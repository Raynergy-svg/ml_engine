#!/usr/bin/env python3
"""Model architecture builders for ML Engine Trading Bot."""
from __future__ import annotations


def _build_buddy_model(
    *,
    feature_dim: int,
    seq_len: int,
    head_hidden: int = 64,
    head_layers: int = 2,
    head_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
):
    """Create the Buddy model: 5 parallel LSTM heads + shared dense + direction+confidence outputs."""
    import tensorflow as tf

    from ml_head_engine import MLEngineHead
    from mr_engine import MREngineHead
    from mt_engine import MTEngineHead
    from ms_head_engine import MSEngineHead
    from mx_head_engine import MXEngineHead

    inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")
    h1 = MLEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="ml")(inp)
    h2 = MREngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mr")(inp)
    h3 = MTEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mt")(inp)
    h4 = MSEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="ms")(inp)
    h5 = MXEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mx")(inp)

    merged = tf.keras.layers.Concatenate(name="concat")([h1, h2, h3, h4, h5])
    x = tf.keras.layers.Dense(int(dense_hidden), activation="relu", name="dense_0")(merged)
    x = tf.keras.layers.Dropout(float(dense_dropout), name="dense_dropout")(x)
    x = tf.keras.layers.Dense(int(dense_hidden // 2), activation="relu", name="dense_1")(x)

    # Force float32 outputs for numerically stable losses/metrics under mixed precision.
    direction = tf.keras.layers.Dense(1, activation="sigmoid", name="direction", dtype="float32")(x)
    # Confidence is trained as a regression-style score (scaled |next return|, nominally in [0,1]).
    # Use a bounded head so training can't diverge to extreme values.
    confidence = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence", dtype="float32")(x)

    return tf.keras.Model(inputs=inp, outputs={"direction": direction, "confidence": confidence}, name="buddy_model")


def _build_buddy_model_shared_encoder(
    *,
    feature_dim: int,
    seq_len: int,
    encoder_hidden: int = 64,
    encoder_layers: int = 2,
    encoder_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
):
    """Create a faster Buddy model with a shared LSTM encoder + dense heads."""
    import tensorflow as tf

    inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")
    x = inp

    n_layers = max(1, int(encoder_layers))
    for i in range(n_layers):
        return_sequences = i < (n_layers - 1)
        x = tf.keras.layers.LSTM(
            int(encoder_hidden),
            return_sequences=bool(return_sequences),
            dropout=float(encoder_dropout),
            name=f"enc_lstm_{i}",
        )(x)

    x = tf.keras.layers.Dense(int(dense_hidden), activation="relu", name="dense_0")(x)
    x = tf.keras.layers.Dropout(float(dense_dropout), name="dense_dropout")(x)
    x = tf.keras.layers.Dense(int(dense_hidden // 2), activation="relu", name="dense_1")(x)

    # Force float32 outputs for numerically stable losses/metrics under mixed precision.
    direction = tf.keras.layers.Dense(1, activation="sigmoid", name="direction", dtype="float32")(x)
    confidence = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence", dtype="float32")(x)

    return tf.keras.Model(
        inputs=inp,
        outputs={"direction": direction, "confidence": confidence},
        name="buddy_model_shared_encoder",
    )


def _build_buddy_model_tcn(
    *,
    feature_dim: int,
    seq_len: int,
    hidden_size: int = 64,
    num_layers: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.5,           # Increased from 0.2 to combat overfitting
    dense_hidden: int = 128,
    dense_dropout: float = 0.4,     # Increased from 0.2
    kernel_regularizer: float = 0.005,  # Increased from 0.002 (stronger L2)
    noise_std: float = 0.05,        # Increased from 0.03 (more input noise)
):
    """Create a Buddy model with TCN encoder - FASTER on M1 Metal than LSTM.

    TCN uses dilated causal convolutions which are fully parallelizable,
    making them 2-3x faster than LSTMs on Apple Silicon.
    """
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.regularizers import l2

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


# Valid model types for type hints and validation
class ModelType:
    """Enumeration of supported model types."""
    TCN = 'tcn'
    LSTM = 'lstm'
    SHARED_ENCODER = 'shared_encoder'
    XGBOOST = 'xgboost'
    ATTENTION_LSTM = 'attention_lstm'
    
    @classmethod
    def all(cls) -> list[str]:
        """Return list of all valid model types."""
        return [cls.TCN, cls.LSTM, cls.SHARED_ENCODER, cls.XGBOOST, cls.ATTENTION_LSTM]
    
    @classmethod
    def is_valid(cls, model_type: str) -> bool:
        """Check if a model type string is valid."""
        return model_type.lower().strip() in cls.all()


def _validate_model_params(
    feature_dim: int,
    seq_len: int,
    head_hidden: int,
    head_layers: int,
    dense_hidden: int,
) -> None:
    """Validate model configuration parameters.
    
    Args:
        feature_dim: Number of input features
        seq_len: Sequence length
        head_hidden: Hidden layer size
        head_layers: Number of hidden layers
        dense_hidden: Dense layer size
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if head_hidden <= 0:
        raise ValueError(f"head_hidden must be positive, got {head_hidden}")
    if head_layers < 1:
        raise ValueError(f"head_layers must be >= 1, got {head_layers}")
    if dense_hidden <= 0:
        raise ValueError(f"dense_hidden must be positive, got {dense_hidden}")


def _build_shared_encoder_model(
    *,
    feature_dim: int,
    seq_len: int,
    encoder_hidden: int,
    encoder_layers: int,
    encoder_dropout: float,
    dense_hidden: int,
    dense_dropout: float,
):
    """Build a shared encoder model with given parameters.
    
    This is a helper to reduce code duplication.
    """
    return _build_buddy_model_shared_encoder(
        feature_dim=feature_dim,
        seq_len=seq_len,
        encoder_hidden=encoder_hidden,
        encoder_layers=encoder_layers,
        encoder_dropout=encoder_dropout,
        dense_hidden=dense_hidden,
        dense_dropout=dense_dropout,
    )


def _build_xgboost_model(
    feature_dim: int,
    seq_len: int,
    config: dict | None = None,
):
    """Build an XGBoost model for Buddy training.

    Returns a wrapper that's compatible with the Keras training interface.
    Note: XGBoost training is handled separately in _train_buddy_xgboost.
    """
    from src.models.xgboost_model import XGBoostTradingModel, XGBoostConfig

    xgb_config = XGBoostConfig()
    if config:
        for key, value in config.items():
            if hasattr(xgb_config, key):
                setattr(xgb_config, key, value)

    return XGBoostTradingModel(xgb_config)


def _build_buddy_model_for_type(
    model_type: str,
    *,
    feature_dim: int,
    seq_len: int,
    head_hidden: int = 64,
    head_layers: int = 2,
    head_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
    kernel_regularizer: float = 0.002,
    noise_std: float = 0.03,
):
    """Build a Buddy model based on the specified architecture type.

    Args:
        model_type: One of 'tcn', 'lstm', 'shared_encoder', 'xgboost', 'attention_lstm'
        feature_dim: Number of input features (must be positive)
        seq_len: Sequence length (must be positive)
        head_hidden: Hidden layer size for encoder heads (default: 64)
        head_layers: Number of encoder layers (default: 2)
        head_dropout: Dropout rate for encoder layers (default: 0.1)
        dense_hidden: Dense layer size (default: 128)
        dense_dropout: Dropout rate for dense layers (default: 0.2)
        kernel_regularizer: L2 regularization strength (default: 0.002)
        noise_std: Input noise standard deviation (default: 0.03)

    Returns:
        Configured Keras model or XGBoost wrapper

    Raises:
        ValueError: If model_type is invalid or parameters are out of range

    M1 Metal Performance Recommendations:
        - 'tcn': Fastest on Metal (2-3x faster than LSTM), recommended for production
        - 'lstm': Good baseline, fully compatible
        - 'attention_lstm': Better accuracy potential, slightly slower
        - 'xgboost': Gradient boosting, requires separate training pipeline
    """
    # Normalize and validate model type
    normalized_type = model_type.lower().strip()
    
    if not ModelType.is_valid(normalized_type):
        valid_types = ", ".join(f"'{t}'" for t in ModelType.all())
        raise ValueError(
            f"Invalid model_type '{model_type}'. "
            f"Valid options: {valid_types}"
        )
    
    # Validate all numeric parameters
    _validate_model_params(
        feature_dim=feature_dim,
        seq_len=seq_len,
        head_hidden=head_hidden,
        head_layers=head_layers,
        dense_hidden=dense_hidden,
    )
    
    # Validate dropout and regularization parameters
    if not 0 <= head_dropout <= 1:
        raise ValueError(f"head_dropout must be in [0, 1], got {head_dropout}")
    if not 0 <= dense_dropout <= 1:
        raise ValueError(f"dense_dropout must be in [0, 1], got {dense_dropout}")
    if kernel_regularizer < 0:
        raise ValueError(f"kernel_regularizer must be >= 0, got {kernel_regularizer}")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}")
    
    # Build model based on type
    if normalized_type == ModelType.TCN:
        return _build_buddy_model_tcn(
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
    elif normalized_type in (ModelType.LSTM, ModelType.SHARED_ENCODER, ModelType.ATTENTION_LSTM):
        # Note: ATTENTION_LSTM currently uses shared encoder as fallback
        # This can be extended to a true attention-based LSTM implementation
        return _build_shared_encoder_model(
            feature_dim=feature_dim,
            seq_len=seq_len,
            encoder_hidden=head_hidden,
            encoder_layers=head_layers,
            encoder_dropout=head_dropout,
            dense_hidden=dense_hidden,
            dense_dropout=dense_dropout,
        )
    elif normalized_type == ModelType.XGBOOST:
        # XGBoost model - returns a wrapper, actual training handled separately
        # Note: XGBoost training is handled in _train_buddy_xgboost
        return _build_xgboost_model(
            feature_dim=feature_dim,
            seq_len=seq_len,
            config=None,
        )
    
    # This should never be reached due to validation above
    raise RuntimeError(f"Unhandled model_type: {normalized_type}")
