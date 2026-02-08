"""
Model Builders for Buddy Trading System.

This module provides factory functions and classes for building various model architectures
used in the Buddy FX trading system. Extracted from main.py for better modularity.

Model Types:
- TCN (Temporal Convolutional Network): Fastest on Apple Silicon M1/M2/M3
- LSTM (Shared Encoder): Good baseline, compatible everywhere
- Multi-Head LSTM: 5 parallel heads (legacy, requires custom engine layers)
- XGBoost: Gradient boosting for tabular data

M1 Metal Optimizations:
- TCN is 2-3x faster than LSTM (fully parallelizable)
- Use recurrent_dropout=0.0 on Metal (non-zero causes massive slowdown)
- Batch sizes of 64-256 are optimal for unified memory
- Avoid recurrent_dropout > 0 with Apple Silicon
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


# =============================================================================
# Model Configuration
# =============================================================================

class ModelType(str, Enum):
    """Supported model architecture types."""
    TCN = "tcn"
    LSTM = "lstm"
    SHARED_ENCODER = "shared_encoder"
    ATTENTION_LSTM = "attention_lstm"
    MULTI_HEAD = "multi_head"
    XGBOOST = "xgboost"


@dataclass
class ModelConfig:
    """
    Configuration for model building.

    Attributes:
        input_dim: Feature dimension (number of input features)
        seq_len: Sequence length (number of timesteps)
        num_layers: Number of encoder layers (LSTM/TCN)
        d_model: Hidden dimension / model dimension
        num_heads: Number of attention heads (if applicable)
        dff: Feed-forward dimension (if applicable)
        dropout: Dropout rate for regularization
        noise_std: Gaussian noise stddev for input augmentation
        dense_hidden: Hidden units in dense head
        dense_dropout: Dropout rate in dense head
        kernel_regularizer: L2 regularization strength
        kernel_size: Convolution kernel size (TCN only)
    """
    input_dim: int = 64
    seq_len: int = 60
    num_layers: int = 2
    d_model: int = 64
    num_heads: int = 4
    dff: int = 128
    dropout: float = 0.3
    noise_std: float = 0.05
    dense_hidden: int = 128
    dense_dropout: float = 0.3
    kernel_regularizer: float = 0.005
    kernel_size: int = 3

    @classmethod
    def from_yaml_config(cls, config: dict) -> "ModelConfig":
        """Create ModelConfig from YAML configuration dictionary."""
        transformer_cfg = config.get("transformer", {})
        # training_cfg available if needed: config.get("training", {})

        return cls(
            num_layers=transformer_cfg.get("num_layers", 2),
            d_model=transformer_cfg.get("d_model", 64),
            num_heads=transformer_cfg.get("num_heads", 4),
            dff=transformer_cfg.get("dff", 128),
            dropout=transformer_cfg.get("dropout", 0.4),
            noise_std=transformer_cfg.get("noise_std", 0.05),
            dense_hidden=transformer_cfg.get("dense_hidden", 128),
            dense_dropout=transformer_cfg.get("dense_dropout", 0.3),
            kernel_regularizer=transformer_cfg.get("kernel_regularizer", 0.005),
            kernel_size=transformer_cfg.get("kernel_size", 3),
        )


# =============================================================================
# Model Builder Functions
# =============================================================================

def build_tcn_model(
    *,
    feature_dim: int,
    seq_len: int,
    hidden_size: int = 64,
    num_layers: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.5,
    dense_hidden: int = 128,
    dense_dropout: float = 0.4,
    kernel_regularizer: float = 0.005,
    noise_std: float = 0.05,
) -> tf.keras.Model:
    """
    Create a Buddy model with TCN encoder - FASTEST on M1 Metal.

    TCN uses dilated causal convolutions which are fully parallelizable,
    making them 2-3x faster than LSTMs on Apple Silicon.

    Args:
        feature_dim: Number of input features
        seq_len: Sequence length
        hidden_size: Number of filters per TCN layer
        num_layers: Number of TCN layers (dilation doubles each layer)
        kernel_size: Convolution kernel size
        dropout: Dropout rate in TCN layers
        dense_hidden: Hidden units in classification head
        dense_dropout: Dropout in classification head
        kernel_regularizer: L2 regularization strength
        noise_std: Gaussian noise stddev for input augmentation

    Returns:
        Keras Model with 'direction' and 'confidence' outputs
    """
    l2_reg = l2(kernel_regularizer)
    inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")

    # Input regularization (M1-compatible alternative to recurrent_dropout)
    x = layers.GaussianNoise(noise_std)(inp)
    x = layers.SpatialDropout1D(dropout * 0.5)(x)

    # TCN layers with exponentially increasing dilation
    for i in range(num_layers):
        dilation_rate = 2 ** i

        # Causal convolution with dilation
        conv_out = layers.Conv1D(
            filters=hidden_size,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation=None,
            kernel_regularizer=l2_reg,
            name=f'tcn_conv_{i}'
        )(x)
        conv_out = layers.BatchNormalization(name=f'tcn_bn_{i}')(conv_out)
        conv_out = layers.Activation('relu', name=f'tcn_relu_{i}')(conv_out)
        conv_out = layers.Dropout(dropout, name=f'tcn_dropout_{i}')(conv_out)

        # Residual connection
        if x.shape[-1] != hidden_size:
            x = layers.Conv1D(hidden_size, 1, name=f'tcn_residual_{i}')(x)
        x = layers.Add(name=f'tcn_add_{i}')([x, conv_out])

    # Global pooling to get fixed-size representation
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)

    # Dense head
    x = layers.Dense(dense_hidden, activation='relu', name='dense_0', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dense_dropout, name='dense_dropout')(x)
    x = layers.Dense(dense_hidden // 2, activation='relu', name='dense_1', kernel_regularizer=l2_reg)(x)

    # Output heads (float32 for numerical stability with mixed precision)
    direction = layers.Dense(1, activation='sigmoid', name='direction', dtype='float32')(x)
    confidence = layers.Dense(1, activation='sigmoid', name='confidence', dtype='float32')(x)

    return tf.keras.Model(
        inputs=inp,
        outputs={'direction': direction, 'confidence': confidence},
        name='buddy_model_tcn',
    )


def build_shared_encoder_model(
    *,
    feature_dim: int,
    seq_len: int,
    encoder_hidden: int = 64,
    encoder_layers: int = 2,
    encoder_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
) -> tf.keras.Model:
    """
    Create a Buddy model with a shared LSTM encoder + dense heads.

    This is a simpler, faster alternative to the multi-head LSTM model.
    Uses a single LSTM stack shared for both direction and confidence prediction.

    Args:
        feature_dim: Number of input features
        seq_len: Sequence length
        encoder_hidden: LSTM hidden units
        encoder_layers: Number of LSTM layers
        encoder_dropout: Dropout in LSTM layers (keep 0.0 on Metal!)
        dense_hidden: Hidden units in classification head
        dense_dropout: Dropout in classification head

    Returns:
        Keras Model with 'direction' and 'confidence' outputs

    Note:
        On Apple Silicon (M1/M2/M3), keep encoder_dropout=0.0 to avoid
        massive slowdowns caused by non-zero recurrent_dropout.
    """
    inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")
    x = inp

    n_layers = max(1, int(encoder_layers))
    for i in range(n_layers):
        return_sequences = i < (n_layers - 1)
        x = layers.LSTM(
            int(encoder_hidden),
            return_sequences=bool(return_sequences),
            dropout=float(encoder_dropout),
            name=f"enc_lstm_{i}",
        )(x)

    x = layers.Dense(int(dense_hidden), activation="relu", name="dense_0")(x)
    x = layers.Dropout(float(dense_dropout), name="dense_dropout")(x)
    x = layers.Dense(int(dense_hidden // 2), activation="relu", name="dense_1")(x)

    # Force float32 outputs for numerically stable losses/metrics under mixed precision.
    direction = layers.Dense(1, activation="sigmoid", name="direction", dtype="float32")(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence", dtype="float32")(x)

    return tf.keras.Model(
        inputs=inp,
        outputs={"direction": direction, "confidence": confidence},
        name="buddy_model_shared_encoder",
    )


def build_lstm_model(
    *,
    feature_dim: int,
    seq_len: int,
    head_hidden: int = 64,
    head_layers: int = 2,
    head_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
) -> tf.keras.Model:
    """
    Create the Buddy model with 5 parallel LSTM heads + shared dense + outputs.

    This is the original multi-head architecture requiring custom engine layers.
    For most use cases, prefer build_shared_encoder_model() or build_tcn_model().

    NOTE: This function requires custom layer imports (MLEngineHead, etc.) which
    may be in legacy/quarantine. Consider using build_shared_encoder_model() instead.

    Args:
        feature_dim: Number of input features
        seq_len: Sequence length
        head_hidden: Hidden units per head
        head_layers: Number of LSTM layers per head
        head_dropout: Dropout in head LSTMs
        dense_hidden: Hidden units in classification head
        dense_dropout: Dropout in classification head

    Returns:
        Keras Model with 'direction' and 'confidence' outputs

    Raises:
        ImportError: If custom engine head layers are not available
    """
    try:
        from ml_head_engine import MLEngineHead
        from mr_engine import MREngineHead
        from mt_engine import MTEngineHead
        from ms_head_engine import MSEngineHead
        from mx_head_engine import MXEngineHead
    except ImportError as e:
        raise ImportError(
            f"Multi-head LSTM requires custom engine layers which are not available. "
            f"Use build_shared_encoder_model() or build_tcn_model() instead. "
            f"Original error: {e}"
        ) from e

    inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")
    h1 = MLEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="ml")(inp)
    h2 = MREngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mr")(inp)
    h3 = MTEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mt")(inp)
    h4 = MSEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="ms")(inp)
    h5 = MXEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mx")(inp)

    merged = layers.Concatenate(name="concat")([h1, h2, h3, h4, h5])
    x = layers.Dense(int(dense_hidden), activation="relu", name="dense_0")(merged)
    x = layers.Dropout(float(dense_dropout), name="dense_dropout")(x)
    x = layers.Dense(int(dense_hidden // 2), activation="relu", name="dense_1")(x)

    # Force float32 outputs for numerically stable losses/metrics under mixed precision.
    direction = layers.Dense(1, activation="sigmoid", name="direction", dtype="float32")(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence", dtype="float32")(x)

    return tf.keras.Model(
        inputs=inp,
        outputs={"direction": direction, "confidence": confidence},
        name="buddy_model",
    )


def build_xgboost_model(
    feature_dim: int,
    seq_len: int,
    config: Optional[dict] = None,
) -> Any:
    """
    Build an XGBoost model for Buddy training.

    Returns a wrapper that's compatible with the Keras training interface.
    Note: XGBoost training is handled separately from Keras models.

    Args:
        feature_dim: Number of input features (for API compatibility)
        seq_len: Sequence length (for API compatibility)
        config: Optional XGBoost configuration dict

    Returns:
        XGBoostTradingModel instance

    Raises:
        ImportError: If XGBoost is not available
    """
    from src.models.xgboost_model import XGBoostTradingModel, XGBoostConfig

    xgb_config = XGBoostConfig()
    if config:
        for key, value in config.items():
            if hasattr(xgb_config, key):
                setattr(xgb_config, key, value)

    return XGBoostTradingModel(xgb_config)


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """
    Factory class for building Buddy trading models.

    Provides a unified interface for building different model architectures
    with proper configuration handling.

    Example:
        >>> config = ModelConfig(input_dim=64, seq_len=60, d_model=64, num_layers=3)
        >>> factory = ModelFactory()
        >>> model = factory.build(ModelType.TCN, config)
        >>> model.summary()

    M1 Metal Recommendations:
        - ModelType.TCN: Fastest (2-3x faster than LSTM)
        - ModelType.LSTM/SHARED_ENCODER: Good baseline
        - ModelType.XGBOOST: CPU-only, good for small datasets
    """

    # Registry of model builders
    _builders: dict[ModelType, callable] = {
        ModelType.TCN: build_tcn_model,
        ModelType.LSTM: build_shared_encoder_model,
        ModelType.SHARED_ENCODER: build_shared_encoder_model,
        ModelType.ATTENTION_LSTM: build_shared_encoder_model,  # Falls back to shared encoder
        ModelType.MULTI_HEAD: build_lstm_model,
        ModelType.XGBOOST: build_xgboost_model,
    }

    @classmethod
    def build(
        cls,
        model_type: Union[str, ModelType],
        config: ModelConfig,
    ) -> tf.keras.Model:
        """
        Build a model based on the specified architecture type.

        Args:
            model_type: One of 'tcn', 'lstm', 'shared_encoder', 'attention_lstm',
                       'multi_head', 'xgboost' or a ModelType enum
            config: ModelConfig with architecture parameters

        Returns:
            Configured Keras model (or XGBoost wrapper for 'xgboost' type)

        Raises:
            ValueError: If model_type is not recognized
        """
        # Normalize model type
        if isinstance(model_type, str):
            model_type = model_type.lower().strip()
            try:
                model_type = ModelType(model_type)
            except ValueError:
                # Handle slight variations
                if model_type in ("encoder", "shared"):
                    model_type = ModelType.SHARED_ENCODER
                elif model_type in ("attention", "attn_lstm"):
                    model_type = ModelType.ATTENTION_LSTM
                elif model_type in ("heads", "parallel", "parallel_heads"):
                    model_type = ModelType.MULTI_HEAD
                else:
                    raise ValueError(
                        f"Unknown model type: '{model_type}'. "
                        f"Supported: {[t.value for t in ModelType]}"
                    )

        builder = cls._builders.get(model_type)
        if builder is None:
            raise ValueError(
                f"No builder registered for model type: {model_type}. "
                f"Supported: {[t.value for t in ModelType]}"
            )

        # Build with appropriate kwargs based on model type
        if model_type == ModelType.TCN:
            return builder(
                feature_dim=config.input_dim,
                seq_len=config.seq_len,
                hidden_size=config.d_model,
                num_layers=config.num_layers,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
                dense_hidden=config.dense_hidden,
                dense_dropout=config.dense_dropout,
                kernel_regularizer=config.kernel_regularizer,
                noise_std=config.noise_std,
            )
        elif model_type in (ModelType.LSTM, ModelType.SHARED_ENCODER, ModelType.ATTENTION_LSTM):
            return builder(
                feature_dim=config.input_dim,
                seq_len=config.seq_len,
                encoder_hidden=config.d_model,
                encoder_layers=config.num_layers,
                encoder_dropout=config.dropout,
                dense_hidden=config.dense_hidden,
                dense_dropout=config.dense_dropout,
            )
        elif model_type == ModelType.MULTI_HEAD:
            return builder(
                feature_dim=config.input_dim,
                seq_len=config.seq_len,
                head_hidden=config.d_model,
                head_layers=config.num_layers,
                head_dropout=config.dropout,
                dense_hidden=config.dense_hidden,
                dense_dropout=config.dense_dropout,
            )
        elif model_type == ModelType.XGBOOST:
            return builder(
                feature_dim=config.input_dim,
                seq_len=config.seq_len,
                config=None,  # Use defaults
            )

        # Should not reach here
        raise ValueError(f"Unhandled model type: {model_type}")

    @classmethod
    def build_from_type_string(
        cls,
        model_type: str,
        *,
        feature_dim: int,
        seq_len: int,
        head_hidden: int = 64,
        head_layers: int = 2,
        head_dropout: float = 0.1,
        dense_hidden: int = 128,
        dense_dropout: float = 0.2,
        kernel_regularizer: float = 0.005,
        noise_std: float = 0.05,
    ) -> tf.keras.Model:
        """
        Build a Buddy model based on the specified architecture type (backward compatible).

        This method provides backward compatibility with the original
        _build_buddy_model_for_type() function from main.py.

        Args:
            model_type: One of 'lstm', 'tcn', 'attention_lstm', 'shared_encoder', 'xgboost'
            Other args: Model configuration parameters

        Returns:
            Configured Keras model

        M1 Metal Recommendations:
            - 'tcn': Fastest on Metal (2-3x faster than LSTM)
            - 'lstm': Good baseline, compatible
            - 'attention_lstm': Better accuracy, slightly slower
        """
        model_type = model_type.lower().strip()

        if model_type == 'tcn':
            return build_tcn_model(
                feature_dim=feature_dim,
                seq_len=seq_len,
                hidden_size=head_hidden,
                num_layers=head_layers,
                dropout=head_dropout,
                dense_hidden=dense_hidden,
                dense_dropout=dense_dropout,
                kernel_regularizer=kernel_regularizer,
                noise_std=noise_std,
            )
        elif model_type == 'xgboost':
            return build_xgboost_model(
                feature_dim=feature_dim,
                seq_len=seq_len,
                config=None,
            )
        else:
            # LSTM variants: 'lstm', 'shared_encoder', 'attention_lstm', or fallback
            # All use shared encoder (attention_lstm can be extended later)
            return build_shared_encoder_model(
                feature_dim=feature_dim,
                seq_len=seq_len,
                encoder_hidden=head_hidden,
                encoder_layers=head_layers,
                encoder_dropout=head_dropout,
                dense_hidden=dense_hidden,
                dense_dropout=dense_dropout,
            )


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# These functions maintain the original API from main.py
def _build_buddy_model(**kwargs) -> tf.keras.Model:
    """Backward compatible alias for build_lstm_model()."""
    return build_lstm_model(**kwargs)


def _build_buddy_model_shared_encoder(**kwargs) -> tf.keras.Model:
    """Backward compatible alias for build_shared_encoder_model()."""
    return build_shared_encoder_model(**kwargs)


def _build_buddy_model_tcn(**kwargs) -> tf.keras.Model:
    """Backward compatible alias for build_tcn_model()."""
    return build_tcn_model(**kwargs)


def _build_xgboost_model(feature_dim: int, seq_len: int, config: Optional[dict] = None):
    """Backward compatible alias for build_xgboost_model()."""
    return build_xgboost_model(feature_dim, seq_len, config)


def _build_buddy_model_for_type(model_type: str, **kwargs) -> tf.keras.Model:
    """Backward compatible alias for ModelFactory.build_from_type_string()."""
    return ModelFactory.build_from_type_string(model_type, **kwargs)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums and config
    "ModelType",
    "ModelConfig",

    # Factory class
    "ModelFactory",

    # Builder functions (new API)
    "build_tcn_model",
    "build_lstm_model",
    "build_shared_encoder_model",
    "build_xgboost_model",

    # Backward compatibility aliases
    "_build_buddy_model",
    "_build_buddy_model_shared_encoder",
    "_build_buddy_model_tcn",
    "_build_xgboost_model",
    "_build_buddy_model_for_type",
]
