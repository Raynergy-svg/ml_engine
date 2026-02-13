"""
Transformer Direction Trainer Module.

Transformer model for direction prediction with advanced continual learning:
- EMA shadow weights for stable inference
- EWC for multi-instrument learning without forgetting
- Replay buffer to retain past market patterns
- Training lineage tracking
- Warm-start with LR reduction
- Prediction collapse detection and recovery
- Tier-2 calibration

This is a large class (~2,400 lines) extracted from modular_trainers.py
for better code organization.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import OverfitPreventionConfig, TrainerConfig
from src.training.trainers.callbacks import (
    EWCTrainingCallback,
    EMACallback,
    EWCPenalty,
    OverfitPreventionCallback,
    QuietProgressCallback,
    RichEpochCallback,
    GradualUnfreezeCallback,
    ReplayBuffer,
    DriftDetector,
    TrainingLineage,
)
from src.training.trainers.utils import (
    ARCH_JSON_SUFFIX,
    EMA_PKL_SUFFIX,
    EWC_PKL_SUFFIX,
    META_PKL_SUFFIX,
    MODEL_NOT_TRAINED_ERROR,
    WEIGHTS_H5_SUFFIX,
    WEIGHTS_LOADED_FULL_MODEL_MSG,
    _get_numpy_dtype,
    _safe_get_learning_rate,
    _safe_load_weights_ignoring_optimizer,
    _safe_reset_optimizer_state,
    _safe_set_learning_rate,
    _validate_weight_shapes,
    compute_auto_variance_weight,
    create_ewc_loss,
    create_sequences_with_weights,
    get_config_seq_len,
)

logger = logging.getLogger(__name__)


class TransformerDirectionTrainer(BaseTrainer):
    """
    Transformer model for direction prediction with advanced continual learning.

    Self-attention captures long-range dependencies in price trends,
    making it better suited for direction prediction than TCN.

    Features (2025):
    - EMA shadow weights for stable inference
    - EWC for multi-instrument learning without forgetting
    - Replay buffer to retain past market patterns
    - Training lineage tracking
    - Warm-start with LR reduction

    Input: Directional features (ADX, MACD, SMA crosses, market structure)
    Output: Binary direction (0=down, 1=up)
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.seq_len = None
        self.selected_indices = None  # Feature selection indices (set during training)

        # Transformer-specific config - defaults match TrainerConfig for proven 60.9% config
        self.transformer_d_model = (
            getattr(config, "transformer_d_model", 16) if config else 16
        )
        self.transformer_num_heads = (
            getattr(config, "transformer_num_heads", 2) if config else 2
        )
        self.transformer_num_layers = (
            getattr(config, "transformer_num_layers", 1) if config else 1
        )
        self.transformer_dff = getattr(config, "transformer_dff", 32) if config else 32
        self.transformer_dropout = (
            getattr(config, "transformer_dropout", 0.2) if config else 0.2
        )  # Reduced from 0.4

        # === CONTINUAL LEARNING COMPONENTS ===

        # EMA for stable inference
        self.ema: Optional[EMACallback] = None
        self._use_ema = getattr(config, "use_ema", True) if config else True

        # EWC for multi-instrument learning
        self.ewc: Optional[EWCPenalty] = None
        self._use_ewc = getattr(config, "use_ewc", True) if config else True

        # Replay buffer
        self.replay_buffer: Optional[ReplayBuffer] = None
        self._use_replay = (
            getattr(config, "use_replay_buffer", True) if config else True
        )

        # Drift detector
        self.drift_detector: Optional[DriftDetector] = None
        self._drift_threshold = (
            getattr(config, "drift_threshold", 0.03) if config else 0.03
        )

        # Training lineage
        self.lineage: Optional[TrainingLineage] = None

        # Track if this is a warm-start session
        self._is_warm_start = False

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        Build Transformer model architecture.

        Key changes for anti-collapse:
        1. Very light L2 regularization (0.001) - too much suppresses outputs
        2. Moderate dropout (0.25)
        3. Small model capacity (d_model=32, 2 layers)
        4. NO regularization on output layer - let it learn freely
        5. Output bias initialized to small positive value (assumes ~52% UP)
        """
        from tensorflow import keras

        # Focal Loss and Class-Balanced Loss are imported later when needed for class imbalance handling

        seq_len, n_features = input_shape

        # L2 regularization — read from config (wired from YAML transformer.l2_reg)
        l2_weight = getattr(self.config, "transformer_l2_reg", 0.003) if self.config else 0.003
        l2_reg = keras.regularizers.l2(l2_weight)

        # Input
        inp = keras.Input(shape=(seq_len, n_features), name="features")

        # Input noise — read from config (wired from YAML transformer.input_noise)
        noise_level = getattr(self.config, "transformer_input_noise", 0.02) if self.config else 0.02
        x = keras.layers.GaussianNoise(noise_level)(inp)

        # Spatial dropout on input sequence — read from config (wired from YAML)
        spatial_dropout = getattr(self.config, "transformer_spatial_dropout", 0.10) if self.config else 0.10
        x = keras.layers.SpatialDropout1D(spatial_dropout)(x)

        # Project features to d_model dimension (with L2)
        x = keras.layers.Dense(
            self.transformer_d_model, kernel_regularizer=l2_reg, name="input_projection"
        )(x)
        proj_dropout = getattr(self.config, "transformer_projection_dropout", 0.10) if self.config else 0.10
        x = keras.layers.Dropout(proj_dropout)(x)

        # Add positional encoding
        x = self._add_positional_encoding(x, seq_len, self.transformer_d_model)

        # Transformer encoder layers
        for i in range(self.transformer_num_layers):
            x = self._transformer_encoder_layer(
                x,
                self.transformer_d_model,
                self.transformer_num_heads,
                self.transformer_dff,
                self.transformer_dropout,  # Uses config dropout (0.4)
                l2_reg,
                name_prefix=f"transformer_{i}",
            )

        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Use tanh instead of ReLU - tanh outputs [-1, 1] which allows both positive
        # and negative contributions to the sigmoid input, making it easier to balance around 0.5
        x = keras.layers.Dense(
            self.config.final_dense_units,
            activation=self.config.final_dense_activation,
            kernel_regularizer=l2_reg,
        )(x)
        x = keras.layers.Dropout(self.config.final_dense_dropout)(x)

        # Binary direction output
        # With tanh inputs ranging [-1, 1], the dot product with weights can be near 0
        # Use small kernel init and zero bias to start at sigmoid(0) = 0.5
        direction = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="direction",
            dtype="float32",
            kernel_initializer=keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.05
            ),
            bias_initializer=keras.initializers.Zeros(),  # Start at sigmoid(0) = 0.5
        )(x)

        model = keras.Model(inputs=inp, outputs=direction, name="transformer_direction")

        # Adam optimizer with standard learning rate
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        logger.info(f"Using Adam optimizer with lr={self.config.learning_rate:.2e}")

        # Standard binary cross entropy - we'll handle calibration post-training
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
            metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)],
        )

        return model

    def _add_positional_encoding(self, x, seq_len: int, d_model: int):
        """Add sinusoidal positional encoding."""
        # Create positional encoding
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]

        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)

        # Apply sin to even indices, cos to odd
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        pos_encoding = pos_encoding[np.newaxis, :, :].astype(np.float32)

        # Add positional encoding to input (cast to match input dtype for mixed precision)
        pos_tensor = tf.constant(pos_encoding)
        pos_tensor = tf.cast(pos_tensor, x.dtype)
        return x + pos_tensor

    def _transformer_encoder_layer(
        self,
        x,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float,
        l2_reg,
        name_prefix: str,
    ):
        """Single transformer encoder layer with multi-head attention and feedforward."""
        from tensorflow import keras

        # Multi-head self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name_prefix}_mha"
        )(x, x)
        attn_output = keras.layers.Dropout(dropout)(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(
            x + attn_output
        )

        # Feedforward network with L2 regularization
        ffn = keras.layers.Dense(
            dff,
            activation="relu",
            kernel_regularizer=l2_reg,
            name=f"{name_prefix}_ffn1",
        )(x)
        ffn = keras.layers.Dense(
            d_model, kernel_regularizer=l2_reg, name=f"{name_prefix}_ffn2"
        )(ffn)
        ffn = keras.layers.Dropout(dropout)(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(
            x + ffn
        )

        return x

    # =========================================================================
    # TRAIN METHOD HELPER FUNCTIONS (Refactored for cognitive complexity < 15)
    # =========================================================================

    def _initialize_training_state(
        self, instrument: str, data_range: str
    ) -> None:
        """Initialize training lineage and drift detector."""
        self.lineage = TrainingLineage()
        self.lineage.generate_checkpoint_id()
        self.lineage.instrument = instrument
        self.lineage.data_range = data_range
        self.lineage.granularity = (
            getattr(self.config, "granularity", "H1") if self.config else "H1"
        )

        self.drift_detector = DriftDetector(
            performance_threshold=self._drift_threshold,
            feature_drift_threshold=0.10,
            window_size=5,
        )

    def _scale_features(
        self,
        X_train: np.ndarray,
        x_val: np.ndarray,
        skip_scaling: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        from sklearn.preprocessing import StandardScaler

        if skip_scaling:
            logger.info("Skipping scaling (data is already pre-scaled)")
            self.scaler = None
            x_train_scaled = X_train.reshape(-1, X_train.shape[-1])
            x_val_scaled = x_val.reshape(-1, x_val.shape[-1])
        else:
            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            )
            x_val_scaled = self.scaler.transform(x_val.reshape(-1, x_val.shape[-1]))

        return x_train_scaled, x_val_scaled

    def _load_warm_start_feature_meta(
        self, warm_start_path: Optional[str]
    ) -> Tuple[Optional[list], Optional[Any]]:
        """Load feature names and scaler from warm-start checkpoint."""
        if not warm_start_path or not Path(warm_start_path).exists():
            return None, None

        meta_path = Path(warm_start_path).with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return None, None

        try:
            with open(meta_path, "rb") as f:
                warm_meta = pickle.load(f)
            feature_names = warm_meta.get("feature_names")
            scaler = warm_meta.get("scaler")
            if feature_names:
                logger.info(
                    f"🔥 WARM-START: Loaded {len(feature_names)} feature names from checkpoint"
                )
                logger.info(f"   Features: {feature_names[:5]}...")
            return feature_names, scaler
        except Exception as e:
            logger.warning(f"Could not load warm-start meta for features: {e}")
            return None, None

    def _apply_warm_start_features(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        warm_start_feature_names: list,
        warm_start_scaler: Any,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply warm-start feature selection if compatible.

        Returns (x_train, x_val, is_compatible).
        """
        saved_features_set = set(warm_start_feature_names)
        current_features_set = set(self.feature_names)
        common_features = saved_features_set.intersection(current_features_set)

        # Check for exact match (all saved features available)
        if len(common_features) == len(warm_start_feature_names):
            return self._apply_exact_feature_match(
                x_train_scaled, x_val_scaled,
                warm_start_feature_names, warm_start_scaler
            )

        # Check for partial match (80%+ overlap)
        if len(common_features) >= len(warm_start_feature_names) * 0.8:
            return self._apply_partial_feature_match(
                x_train_scaled, x_val_scaled,
                warm_start_feature_names, common_features
            )

        # Insufficient overlap - run fresh selection
        logger.warning(
            f"⚠️ Only {len(common_features)}/{len(warm_start_feature_names)} "
            "saved features available. Running fresh selection."
        )
        return x_train_scaled, x_val_scaled, False

    def _apply_exact_feature_match(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        warm_start_feature_names: list,
        warm_start_scaler: Any,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply exact feature match from warm-start."""
        from sklearn.preprocessing import StandardScaler

        selected_indices = [
            self.feature_names.index(f)
            for f in warm_start_feature_names
            if f in self.feature_names
        ]

        x_train_scaled = x_train_scaled[:, selected_indices]
        x_val_scaled = x_val_scaled[:, selected_indices]

        self.selected_indices = selected_indices  # Store for inference/validation
        self.feature_names = [self.feature_names[i] for i in selected_indices]
        self.n_features = len(selected_indices)

        if warm_start_scaler is not None:
            self.scaler = warm_start_scaler
            logger.info("✓ Using saved scaler from checkpoint")
        elif self.scaler is not None:
            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(x_train_scaled)
            x_val_scaled = self.scaler.transform(x_val_scaled)

        logger.info(
            f"🔥 WARM-START: Using ALL {len(selected_indices)} saved features (weights compatible)"
        )
        logger.info(f"   Top features: {self.feature_names[:5]}")
        return x_train_scaled, x_val_scaled, True

    def _apply_partial_feature_match(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        warm_start_feature_names: list,
        common_features: set,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply partial feature match when 80%+ overlap exists."""
        from sklearn.preprocessing import StandardScaler

        saved_features_set = set(warm_start_feature_names)
        missing_features = saved_features_set - common_features
        logger.warning(
            f"⚠️ Feature dimension mismatch: saved model has {len(warm_start_feature_names)} features, "
            f"but only {len(common_features)} available. Missing: {list(missing_features)[:5]}..."
        )
        logger.warning("⚠️ Disabling warm-start weight loading (incompatible architecture)")

        selected_indices = [
            self.feature_names.index(f)
            for f in warm_start_feature_names
            if f in self.feature_names
        ]

        x_train_scaled = x_train_scaled[:, selected_indices]
        x_val_scaled = x_val_scaled[:, selected_indices]

        self.selected_indices = selected_indices  # Store for inference/validation
        self.feature_names = [self.feature_names[i] for i in selected_indices]
        self.n_features = len(selected_indices)

        if self.scaler is not None:
            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(x_train_scaled)
            x_val_scaled = self.scaler.transform(x_val_scaled)

        logger.info(
            f"🔥 WARM-START: Using {len(selected_indices)} available features (fresh model, no weight transfer)"
        )
        return x_train_scaled, x_val_scaled, False

    def _apply_feature_selection(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        y_train: np.ndarray,
        top_k_features: int,
        method: str = "random_forest",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature selection using RF importance or F-test."""
        logger.info(
            f"🔍 Feature Selection: Reducing from {x_train_scaled.shape[-1]} to "
            f"top {top_k_features} features using {method}"
        )

        x_train_flat = x_train_scaled
        x_val_flat = x_val_scaled
        y_train_flat = np.tile(y_train, (x_train_scaled.shape[0], 1)).flatten()[
            : len(x_train_flat)
        ]

        if method == "random_forest":
            selected_indices, x_train_selected, x_val_selected = \
                self._rf_feature_selection(x_train_flat, x_val_flat, y_train_flat, top_k_features)
        else:
            selected_indices, x_train_selected, x_val_selected = \
                self._ftest_feature_selection(x_train_flat, x_val_flat, y_train_flat, top_k_features)

        self.selected_indices = selected_indices
        self._update_features_after_selection(selected_indices, x_train_selected)

        return x_train_selected, x_val_selected

    def _rf_feature_selection(
        self,
        x_train_flat: np.ndarray,
        x_val_flat: np.ndarray,
        y_train_flat: np.ndarray,
        top_k: int,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Apply Random Forest importance-based feature selection."""
        from src.data.feature_engineering import FeatureEngineering
        import pandas as pd

        feature_cols = (
            self.feature_names
            if self.feature_names
            else [f"feat_{i}" for i in range(x_train_flat.shape[-1])]
        )
        df_train = pd.DataFrame(x_train_flat, columns=feature_cols)
        df_train["_target_"] = y_train_flat

        fe = FeatureEngineering()
        _, selected_features = fe.select_features(
            df_train,
            target_col="_target_",
            method="random_forest",
            top_k=top_k,
        )

        selected_indices = [
            feature_cols.index(f) for f in selected_features if f in feature_cols
        ]
        x_train_selected = x_train_flat[:, selected_indices]
        x_val_selected = x_val_flat[:, selected_indices]

        logger.info(f"🌲 RF Feature Selection: {len(selected_indices)} features selected")
        logger.info(f"   Top features: {selected_features[:5]}")

        return selected_indices, x_train_selected, x_val_selected

    def _ftest_feature_selection(
        self,
        x_train_flat: np.ndarray,
        x_val_flat: np.ndarray,
        y_train_flat: np.ndarray,
        top_k: int,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Apply F-test based feature selection."""
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(score_func=f_classif, k=top_k)
        x_train_selected = selector.fit_transform(x_train_flat, y_train_flat)
        x_val_selected = selector.transform(x_val_flat)

        selected_indices = list(selector.get_support(indices=True))
        logger.info(f"✓ F-test Selection: {len(selected_indices)} features selected")

        return selected_indices, x_train_selected, x_val_selected

    def _update_features_after_selection(
        self, selected_indices: list, x_train_selected: np.ndarray
    ) -> None:
        """Update feature names and scaler after selection."""
        from sklearn.preprocessing import StandardScaler

        if self.feature_names is not None:
            self.feature_names = [self.feature_names[i] for i in selected_indices]
            logger.info(
                f"✓ Updated feature names: {len(self.feature_names)} features selected"
            )

        self.n_features = len(selected_indices)

        if self.scaler is not None:
            self.scaler = StandardScaler()
            # Re-fitting will be done by caller
            logger.info(
                f"✓ Scaler will be re-fitted on {x_train_selected.shape[-1]} selected features"
            )

    def _compute_class_statistics(
        self, y_train_filtered: np.ndarray, y_val_filtered: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Compute class distribution and imbalance ratios."""
        train_up_pct = (y_train_filtered == 1).mean() * 100
        val_up_pct = (y_val_filtered == 1).mean() * 100

        train_imbalance = (
            max(train_up_pct, 100 - train_up_pct)
            / min(train_up_pct, 100 - train_up_pct)
            if min(train_up_pct, 100 - train_up_pct) > 0
            else float("inf")
        )
        val_imbalance = (
            max(val_up_pct, 100 - val_up_pct) / min(val_up_pct, 100 - val_up_pct)
            if min(val_up_pct, 100 - val_up_pct) > 0
            else float("inf")
        )

        logger.info(
            "Class distribution: train=%.1f%% up (imbalance=%.2fx), val=%.1f%% up (imbalance=%.2fx)",
            train_up_pct, train_imbalance, val_up_pct, val_imbalance,
        )
        if train_imbalance > 2.0 or val_imbalance > 2.0:
            logger.warning(
                "High class imbalance detected (>2x). Consider adjusting direction_threshold."
            )

        return train_up_pct, val_up_pct, train_imbalance, val_imbalance

    def _compute_sample_weights(
        self, y_train_filtered: np.ndarray, max_weight: float = 3.0
    ) -> np.ndarray:
        """Compute sample weights for class balancing."""
        n_samples = len(y_train_filtered)
        n_up = int((y_train_filtered == 1).sum())
        n_down = n_samples - n_up

        if n_up > 0 and n_down > 0:
            up_weight = min(n_samples / (2 * n_up), max_weight)
            down_weight = min(n_samples / (2 * n_down), max_weight)
            sample_weights = np.where(y_train_filtered == 1, up_weight, down_weight)
            sample_weights = sample_weights / sample_weights.mean()
            logger.info(
                f"📊 Sample weights: UP={up_weight:.2f}x, DOWN={down_weight:.2f}x (cap={max_weight}x)"
            )
            logger.info(f"🎯 Using AntiCollapseFocalLoss (gamma={self.config.focal_gamma}, alpha={self.config.focal_alpha}, entropy_weight=0.1)")
        else:
            sample_weights = np.ones(n_samples)
            logger.warning("⚠️ Single class in training data - using uniform weights")

        return sample_weights

    def _handle_replay_buffer(
        self,
        x_train_filtered: np.ndarray,
        y_train_filtered: np.ndarray,
        sample_weights: np.ndarray,
        instrument: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and mix replay buffer samples with training data."""
        self.replay_buffer = ReplayBuffer(
            capacity_ratio=self.config.replay_buffer_ratio,
            mix_ratio=self.config.replay_mix_ratio,
            buffer_dir=self.config.replay_buffer_dir,
        )

        if self.replay_buffer.load(instrument):
            x_replay, y_replay, w_replay = self.replay_buffer.get_replay_samples(
                len(x_train_filtered)
            )

            if x_replay is not None and len(x_replay) > 0:
                buffer_n_features = x_replay.shape[2]
                if buffer_n_features == x_train_filtered.shape[2]:
                    x_train_filtered = np.vstack([x_train_filtered, x_replay])
                    y_train_filtered = np.concatenate([y_train_filtered, y_replay])
                    if sample_weights is not None and w_replay is not None:
                        sample_weights = np.concatenate([sample_weights, w_replay])
                    logger.info(
                        f"📦 Mixed {len(x_replay)} replay samples, new train size: {len(x_train_filtered)}"
                    )
                else:
                    logger.warning(
                        f"⚠️ Replay buffer feature mismatch: buffer has {buffer_n_features} features, "
                        f"current data has {x_train_filtered.shape[2]}. Clearing buffer."
                    )
                    self.replay_buffer.clear()

        # Always add current training data to buffer for future sessions
        self.replay_buffer.add_samples(
            x_train_filtered,
            y_train_filtered,
            sample_weights,
            data_id=f"{instrument}_{self.lineage.checkpoint_id}",
        )

        return x_train_filtered, y_train_filtered, sample_weights

    def _create_loss_function(
        self, auto_variance_weight: float, label_smoothing: float
    ) -> Any:
        """Create appropriate loss function based on config.

        Returns the base loss function (before EWC wrapping).
        Tries loss types in priority order: MADL > Hybrid CB > Anti-Collapse > CB Focal > Focal > BCE.
        """
        # Priority list of loss function attempts
        loss_attempts = [
            (getattr(self.config, "use_madl_loss", False) if self.config else False,
             lambda: self._try_madl_loss(label_smoothing)),
            (self.config and getattr(self.config, "use_hybrid_cb_anticollapse", True),
             lambda: self._try_hybrid_cb_loss(auto_variance_weight, label_smoothing)),
            (self.config and getattr(self.config, "use_anti_collapse_loss", True),
             lambda: self._try_anti_collapse_loss(auto_variance_weight)),
            (self.config and self.config.use_class_balanced_loss,
             lambda: self._try_cb_focal_loss(label_smoothing)),
        ]

        for should_try, try_func in loss_attempts:
            if should_try:
                loss = try_func()
                if loss is not None:
                    return loss

        # Fallback to standard Focal Loss or BCE
        return self._get_fallback_loss(label_smoothing)

    def _try_madl_loss(self, label_smoothing: float) -> Optional[Any]:
        """Try to create MADL Loss."""
        try:
            from src.models.tensorflow_models import MADLLoss
            madl_direction_weight = (
                getattr(self.config, "madl_direction_weight", 0.7) if self.config else 0.7
            )
            logger.info(
                f"💰 Using MADL Loss for directional profitability "
                f"(direction_weight={madl_direction_weight}, label_smoothing={label_smoothing})"
            )
            return MADLLoss(direction_weight=madl_direction_weight, label_smoothing=label_smoothing)
        except ImportError:
            logger.warning("⚠️ MADLLoss not found, falling back to Class-Balanced Focal Loss")
            return None

    def _try_hybrid_cb_loss(
        self, auto_variance_weight: float, label_smoothing: float
    ) -> Optional[Any]:
        """Try to create Hybrid CB + Anti-Collapse Loss."""
        try:
            from src.models.tensorflow_models import HybridClassBalancedAntiCollapseLoss
            cb_beta = getattr(self.config, "cb_beta", 0.9999)
            cb_gamma = getattr(self.config, "cb_gamma", 2.0)
            variance_weight = auto_variance_weight * 2.5
            variance_target = 0.12
            logger.info(
                f"🔥 Using HybridClassBalancedAntiCollapseLoss "
                f"(beta={cb_beta}, gamma={cb_gamma}, "
                f"variance_weight={variance_weight:.3f}, variance_target={variance_target})"
            )
            return HybridClassBalancedAntiCollapseLoss(
                gamma=cb_gamma, beta=cb_beta,
                variance_weight=variance_weight, variance_target=variance_target,
                label_smoothing=label_smoothing,
            )
        except Exception as e:
            logger.warning(f"⚠️ HybridClassBalancedAntiCollapseLoss failed: {e}")
            return None

    def _try_anti_collapse_loss(
        self, auto_variance_weight: float
    ) -> Optional[Any]:
        """Try to create Anti-Collapse Focal Loss."""
        try:
            from src.models.tensorflow_models import AntiCollapseFocalLoss
            focal_gamma = getattr(self.config, "focal_gamma", 2.0) if self.config else 2.0
            ac_label_smoothing = (
                getattr(self.config, "anti_collapse_label_smoothing", 0.05)
                if self.config else 0.05
            )
            logger.info(
                f"🛡️ Using AntiCollapseFocalLoss (gamma={focal_gamma}, "
                f"variance_weight={auto_variance_weight:.3f}, label_smoothing={ac_label_smoothing})"
            )
            return AntiCollapseFocalLoss(
                gamma=focal_gamma, base_alpha=0.5,
                entropy_weight=auto_variance_weight, label_smoothing=ac_label_smoothing,
            )
        except Exception as e:
            logger.warning(f"⚠️ AntiCollapseFocalLoss failed: {e}")
            return None

    def _try_cb_focal_loss(self, label_smoothing: float) -> Optional[Any]:
        """Try to create Class-Balanced Focal Loss."""
        try:
            from src.models.tensorflow_models import ClassBalancedFocalLoss
            cb_beta = getattr(self.config, "cb_beta", 0.9999) if self.config else 0.9999
            cb_gamma = getattr(self.config, "cb_gamma", 2.0) if self.config else 2.0
            logger.info(
                f"🎯 Using Class-Balanced Focal Loss "
                f"(beta={cb_beta}, gamma={cb_gamma}, label_smoothing={label_smoothing})"
            )
            return ClassBalancedFocalLoss(beta=cb_beta, gamma=cb_gamma, label_smoothing=label_smoothing)
        except ImportError:
            logger.warning("⚠️ ClassBalancedFocalLoss not found, falling back to Focal Loss")
            return None

    def _get_fallback_loss(self, label_smoothing: float) -> Any:
        """Get fallback loss function (Focal or BCE)."""
        from tensorflow import keras
        try:
            from src.models.tensorflow_models import BinaryFocalLoss
            logger.info(
                f"🎯 Using Focal Loss (gamma=2.0, alpha=0.5, label_smoothing={label_smoothing})"
            )
            return BinaryFocalLoss(gamma=2.0, alpha=0.5, label_smoothing=label_smoothing)
        except ImportError:
            logger.warning("⚠️ BinaryFocalLoss not found, falling back to BinaryCrossentropy")
            return keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)

    def _load_warm_start_weights(self, warm_start_path: str) -> bool:
        """Try to load warm-start weights using multiple strategies.

        Returns True if weights were successfully loaded.
        """
        logger.info(f"🔥 WARM-START: Loading weights from {warm_start_path}")

        # Strategy 1: Try .weights.h5 companion file or direct load_weights
        if self._try_load_weights_h5(warm_start_path):
            return True

        # Strategy 2: Try loading full model and extracting weights
        if self._try_load_full_model_weights(warm_start_path):
            return True

        # Strategy 3: Try loading from meta.pkl
        return bool(self._try_load_meta_weights(warm_start_path))

    def _try_load_weights_h5(self, warm_start_path: str) -> bool:
        """Strategy 1: Try loading weights from .h5 file.

        Uses _safe_load_weights_ignoring_optimizer to suppress optimizer-state
        mismatch warnings that are benign during warm-start (the optimizer is
        rebuilt with a fresh Adam before training begins).
        """
        weights_h5_path = Path(warm_start_path).with_suffix(WEIGHTS_H5_SUFFIX)
        if weights_h5_path.exists():
            target = str(weights_h5_path)
        else:
            target = str(warm_start_path)

        if _safe_load_weights_ignoring_optimizer(self.model, target):
            logger.info(f"✓ Loaded model weights from {target}")
            # Reset optimizer slot variables – they belong to the old architecture
            _safe_reset_optimizer_state(self.model)
            return True
        return False

    def _try_load_full_model_weights(self, warm_start_path: str) -> bool:
        """Strategy 2: Try loading full model and extracting weights.

        When architectures differ (e.g. different layer count or frozen layers),
        falls back to *name-based partial weight transfer* so that compatible
        layers still benefit from the checkpoint.
        """
        from tensorflow import keras

        try:
            existing_model = keras.models.load_model(warm_start_path, compile=False)
        except Exception as e:
            logger.debug(f"Full model load failed: {e}")
            return False

        try:
            checkpoint_weights = existing_model.get_weights()
            model_weights = self.model.get_weights()

            is_compatible, shape_error = _validate_weight_shapes(
                model_weights, checkpoint_weights, context="model weights"
            )

            if is_compatible:
                self.model.set_weights(checkpoint_weights)
                logger.info(WEIGHTS_LOADED_FULL_MODEL_MSG)
                return True

            # --- Partial weight transfer via name-based matching ---
            logger.warning(f"Architecture mismatch: {shape_error}")
            logger.info("🔄 Attempting partial weight transfer by layer name...")

            ckpt_weight_map = {
                w.name: w.numpy() for w in existing_model.weights
            }
            loaded, skipped = 0, 0
            for w in self.model.weights:
                if w.name in ckpt_weight_map:
                    ckpt_val = ckpt_weight_map[w.name]
                    if w.shape == ckpt_val.shape:
                        w.assign(ckpt_val.astype(_get_numpy_dtype(w.dtype)))
                        loaded += 1
                    else:
                        skipped += 1
                        logger.debug(
                            f"  Shape mismatch for {w.name}: "
                            f"model={w.shape}, checkpoint={ckpt_val.shape}"
                        )
                else:
                    # Try partial (base) name match
                    base = w.name.split("/")[-1].split(":")[0]
                    matched = False
                    for cn, cv in ckpt_weight_map.items():
                        cb = cn.split("/")[-1].split(":")[0]
                        if base == cb and w.shape == cv.shape:
                            w.assign(cv.astype(_get_numpy_dtype(w.dtype)))
                            loaded += 1
                            matched = True
                            break
                    if not matched:
                        skipped += 1

            if loaded > 0:
                logger.info(
                    f"✓ Partial weight transfer: {loaded} weights loaded, "
                    f"{skipped} re-initialized from scratch"
                )
                _safe_reset_optimizer_state(self.model)
                return True
            else:
                logger.warning("⚠️ No compatible weights found – starting fresh")
                return False
        finally:
            del existing_model

    def _try_load_meta_weights(self, warm_start_path: str) -> bool:
        """Strategy 3: Try loading weights from meta.pkl."""
        meta_path = Path(warm_start_path).with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return False

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if "model_weights" in meta:
                self.model.set_weights(meta["model_weights"])
                logger.info("✓ Loaded weights from meta.pkl")
                return True
            return False
        except Exception as e:
            logger.debug(f"Meta weights load failed: {e}")
            return False

    def _freeze_encoder_layers(self) -> Tuple[int, list]:
        """Freeze encoder layers for warm-start training.

        Returns (frozen_count, trainable_head_layers).
        """
        from tensorflow import keras

        frozen_count = 0
        trainable_head_layers = []

        encoder_patterns = [
            "transformer_", "input_projection", "positional", "multi_head",
            "attention", "ffn", "layer_norm", "spatial_dropout",
            "gaussian_noise", "global_average",
        ]

        for layer in self.model.layers:
            layer_name = layer.name.lower()
            is_encoder_layer = any(pattern in layer_name for pattern in encoder_patterns)

            # Check if this is the classification head
            try:
                output_units = (
                    layer.output.shape[-1] if hasattr(layer, "output")
                    else getattr(layer, "units", 32)
                )
            except (AttributeError, TypeError):
                output_units = getattr(layer, "units", 32)

            is_classification_head = (
                "direction" in layer_name or
                (isinstance(layer, keras.layers.Dense) and output_units <= 16
                 and "projection" not in layer_name)
            )

            if is_encoder_layer and not is_classification_head:
                layer.trainable = False
                frozen_count += 1
            elif layer.trainable:
                trainable_head_layers.append(layer.name)

        return frozen_count, trainable_head_layers

    def _log_frozen_layers(self, frozen_count: int, trainable_head_layers: list) -> None:
        """Log information about frozen layers."""
        import tensorflow as tf

        if frozen_count > 0:
            logger.info(f"🔒 WARM-START: Froze {frozen_count} encoder layers")
            trainable_params = sum(
                [tf.size(w).numpy() for w in self.model.trainable_weights]
            )
            total_params = self.model.count_params()
            pct = 100 * trainable_params / total_params
            logger.info(f"   Trainable: {trainable_params:,}/{total_params:,} params ({pct:.1f}%)")
            if trainable_head_layers:
                suffix = "..." if len(trainable_head_layers) > 5 else ""
                logger.info(f"   Trainable layers: {trainable_head_layers[:5]}{suffix}")
        else:
            logger.warning("⚠️ WARM-START: No layers frozen! This may cause catastrophic forgetting.")

    def _handle_warm_start(self, warm_start_path: str) -> float:
        """Handle warm-start loading: weights, layer freezing, lineage, EWC, EMA.

        Returns the effective learning rate to use.
        """
        try:
            weights_loaded = self._load_warm_start_weights(warm_start_path)

            if weights_loaded:
                self._is_warm_start = True
                self._warm_start_weights = self.model.get_weights()
                logger.info(f"✓ Successfully loaded {self.model.count_params():,} parameters from checkpoint")

                # Freeze encoder layers if configured
                if self.config.warm_start_freeze_encoder:
                    frozen_count, trainable_head_layers = self._freeze_encoder_layers()
                    self._log_frozen_layers(frozen_count, trainable_head_layers)

                # Load metadata (lineage, metrics, training state)
                self._load_warm_start_metadata(warm_start_path)

                # Load EWC state
                self._load_warm_start_ewc(warm_start_path)

                # Load EMA weights
                self._load_warm_start_ema(warm_start_path)

                # Compute effective learning rate
                effective_lr = self.config.learning_rate * self.config.warm_start_lr_factor
                logger.info(
                    f"🔥 Warm-start LR reduction: {self.config.learning_rate} → "
                    f"{effective_lr} (factor={self.config.warm_start_lr_factor})"
                )
                return effective_lr
            else:
                logger.warning("Could not load warm-start weights. Starting fresh.")
                return self.config.learning_rate
        except Exception as e:
            logger.warning(f"Could not load warm-start checkpoint: {e}. Starting fresh.")
            return self.config.learning_rate

    def _load_warm_start_metadata(self, warm_start_path: str) -> None:
        """Load lineage, metrics, and training state from warm-start checkpoint."""
        meta_path = Path(warm_start_path).with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # Load lineage
        if "lineage" in meta:
            parent_lineage = TrainingLineage.from_dict(meta["lineage"])
            self.lineage.parent_checkpoint_id = parent_lineage.checkpoint_id
            self.lineage.cumulative_epochs = parent_lineage.cumulative_epochs
            self.lineage.cumulative_samples = parent_lineage.cumulative_samples
            self.lineage.metric_history = parent_lineage.metric_history.copy()
            self._loaded_model_instrument = parent_lineage.instrument
            logger.info(
                f"📊 Loaded lineage from parent: {parent_lineage.checkpoint_id} "
                f"(cumulative epochs: {self.lineage.cumulative_epochs}, "
                f"instrument: {self._loaded_model_instrument})"
            )

            # Restore dynamic training state
            self._restored_variance_weight = getattr(parent_lineage, "auto_variance_weight", 0.0)
            self._restored_lr_reductions = getattr(parent_lineage, "lr_reductions_count", 0)
            self._restored_lr = getattr(parent_lineage, "final_learning_rate", 0.0)
            if self._restored_variance_weight > 0 or self._restored_lr_reductions > 0:
                logger.info(
                    f"🔄 Restored training state: variance_weight={self._restored_variance_weight:.3f}, "
                    f"lr_reductions={self._restored_lr_reductions}, last_lr={self._restored_lr:.2e}"
                )

        # Load previous best accuracy
        prev_metrics = meta.get("metrics", {})
        self._warm_start_val_acc = prev_metrics.get("val_accuracy", 0.0)
        if self._warm_start_val_acc > 0:
            logger.info(f"🎯 Previous best val_accuracy: {self._warm_start_val_acc:.1%} (will not save worse)")

    def _load_warm_start_ewc(self, warm_start_path: str) -> None:
        """Load EWC Fisher information from previous training."""
        if not self._use_ewc:
            return
        ewc_path = Path(warm_start_path).with_suffix(EWC_PKL_SUFFIX)
        self.ewc = EWCPenalty(
            self.model,
            ewc_lambda=self.config.ewc_lambda,
            gamma=self.config.ewc_gamma,
        )
        if self.ewc.load(str(ewc_path)):
            self.lineage.ewc_n_tasks = self.ewc._n_tasks
            logger.info(f"🧠 EWC loaded: {self.ewc._n_tasks} prior task(s) will be protected")

    def _load_warm_start_ema(self, warm_start_path: str) -> None:
        """Load EMA weights from previous training."""
        if not self._use_ema:
            return
        ema_meta_path = Path(warm_start_path).with_suffix(EMA_PKL_SUFFIX)
        if ema_meta_path.exists():
            with open(ema_meta_path, "rb") as f:
                ema_data = pickle.load(f)

            # Quick check: if tensor count differs drastically, skip loading
            # and remove the stale file so it won't warn on future runs.
            n_model = len(self.model.trainable_weights)
            n_ckpt = len(ema_data.get("ema_weights", []))
            ratio = max(n_model, n_ckpt) / max(min(n_model, n_ckpt), 1)
            if ratio > 3.0:
                logger.info(
                    f"📊 Stale EMA checkpoint ({n_ckpt} tensors) doesn't match "
                    f"current model ({n_model} tensors) — starting fresh EMA."
                )
                try:
                    ema_meta_path.unlink()
                except OSError:
                    pass
                return

            self.ema = EMACallback(
                self.model,
                decay=self.config.ema_decay,
                update_every=self.config.ema_update_every,
            )
            self.ema.set_ema_weights(
                ema_data["ema_weights"],
                weight_names=ema_data.get("ema_weight_names"),
            )
            logger.info("📊 EMA weights loaded from checkpoint")

    def _prepare_sequences_and_filter(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        w_train: Optional[np.ndarray],
        w_val: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences and filter by clear labels."""
        seq_len = get_config_seq_len(self.config)
        self.seq_len = seq_len

        x_train_seq, y_train_seq, w_train_seq = create_sequences_with_weights(
            x_train_scaled, y_train, w_train, seq_len
        )
        x_val_seq, y_val_seq, w_val_seq = create_sequences_with_weights(
            x_val_scaled, y_val, w_val, seq_len
        )

        train_clear_mask = w_train_seq > 0
        val_clear_mask = w_val_seq > 0

        x_train_filtered = x_train_seq[train_clear_mask]
        y_train_filtered = y_train_seq[train_clear_mask]
        x_val_filtered = x_val_seq[val_clear_mask]
        y_val_filtered = y_val_seq[val_clear_mask]

        logger.info(
            f"Filtered training: {train_clear_mask.sum()}/{len(train_clear_mask)} sequences with clear labels"
        )
        logger.info(
            f"Filtered validation: {val_clear_mask.sum()}/{len(val_clear_mask)} sequences with clear labels"
        )

        return x_train_filtered, y_train_filtered, x_val_filtered, y_val_filtered

    def _compute_auto_variance_weight(
        self, instrument: str, train_up_pct: float
    ) -> float:
        """Compute auto-tuned variance weight for loss function."""
        class_ratio = train_up_pct / 100.0
        if hasattr(self, "_restored_variance_weight") and self._restored_variance_weight > 0:
            auto_variance_weight = self._restored_variance_weight
            logger.info(
                f"🔄 Using restored variance_weight={auto_variance_weight:.3f} from warm-start"
            )
        else:
            auto_variance_weight = compute_auto_variance_weight(
                instrument=instrument,
                class_ratio=class_ratio,
                base_weight=self.config.anti_collapse_base_variance_weight if self.config else 0.1,
            )
            logger.info(
                f"🎯 Auto-tuned variance_weight={auto_variance_weight:.3f} "
                f"for {instrument} (class_ratio={class_ratio:.2f})"
            )
        return auto_variance_weight

    def _setup_optimizer_with_warmup(
        self, effective_lr: float, x_train_filtered: np.ndarray
    ) -> Any:
        """Setup optimizer with warmup learning rate schedule."""
        from tensorflow import keras

        warmup_epochs = getattr(self.config, "warmup_epochs", 5) if self.config else 5
        steps_per_epoch = max(1, len(x_train_filtered) // self.config.batch_size)
        total_steps = self.config.epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        try:
            from src.training.m1_metal_optimizer import WarmupCosineDecaySchedule
            lr_schedule = WarmupCosineDecaySchedule(
                initial_learning_rate=effective_lr * 0.1,
                warmup_steps=warmup_steps,
                decay_steps=total_steps - warmup_steps,
                min_learning_rate=1e-6,
                warmup_target=effective_lr,
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            logger.info(
                f"🔥 Warmup LR: {warmup_epochs} epochs ({warmup_steps} steps), target LR={effective_lr:.6f}"
            )
        except ImportError:
            logger.warning("⚠️ WarmupCosineDecaySchedule not available, using constant LR")
            optimizer = keras.optimizers.Adam(learning_rate=effective_lr)

        return optimizer

    def _create_augmentation_fn(self) -> Any:
        """
        Create time-series data augmentation function.
        
        Applies augmentations during training to improve generalization:
        1. Gaussian noise injection
        2. Random scaling
        3. Time masking (like SpecAugment)
        
        Returns:
            TensorFlow augmentation function or None if disabled
        """
        if not getattr(self.config, "use_augmentation", False):
            return None
        
        noise_std = getattr(self.config, "augmentation_noise_std", 0.01)
        scale_range = getattr(self.config, "augmentation_scale_range", (0.98, 1.02))
        time_mask_prob = getattr(self.config, "augmentation_time_mask_prob", 0.1)
        time_mask_max_len = getattr(self.config, "augmentation_time_mask_max_len", 5)
        
        @tf.function
        def augment(x, y):
            """Apply augmentation to a single batch."""
            # Cast to float32 for operations
            x = tf.cast(x, tf.float32)
            
            # 1. Add Gaussian noise
            if noise_std > 0:
                noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std)
                x = x + noise
            
            # 2. Random scaling
            scale = tf.random.uniform([], scale_range[0], scale_range[1])
            x = x * scale
            
            # 3. Time masking (randomly mask some timesteps)
            if time_mask_prob > 0 and time_mask_max_len > 0:
                # Apply time masking with probability
                apply_mask = tf.random.uniform([]) < time_mask_prob
                if apply_mask:
                    batch_size = tf.shape(x)[0]
                    seq_len = tf.shape(x)[1]
                    
                    # Random mask length and start position for each batch item
                    mask_len = tf.random.uniform([batch_size], 1, time_mask_max_len + 1, dtype=tf.int32)
                    mask_start = tf.random.uniform([batch_size], 0, seq_len - time_mask_max_len, dtype=tf.int32)
                    
                    # Create mask for each batch item
                    indices = tf.range(seq_len)  # [seq_len]
                    indices = tf.tile(tf.expand_dims(indices, 0), [batch_size, 1])  # [batch, seq_len]
                    
                    mask_start_expanded = tf.expand_dims(mask_start, 1)  # [batch, 1]
                    mask_len_expanded = tf.expand_dims(mask_len, 1)  # [batch, 1]
                    
                    # Mask: True where NOT masked, False where masked
                    mask = tf.logical_or(
                        indices < mask_start_expanded,
                        indices >= mask_start_expanded + mask_len_expanded
                    )
                    mask = tf.cast(mask, tf.float32)  # [batch, seq_len]
                    mask = tf.expand_dims(mask, -1)  # [batch, seq_len, 1]
                    
                    # Apply mask (zero out masked timesteps)
                    x = x * mask
            
            return x, y
        
        return augment

    def _compile_model_with_loss(
        self,
        optimizer: Any,
        base_loss: Any,
        instrument: str,
    ) -> None:
        """Compile model with appropriate loss function."""
        use_ewc_loss = (
            self._is_warm_start
            and self._use_ewc
            and self.ewc is not None
            and self.ewc.fisher_diagonal is not None
            and getattr(self, "_loaded_model_instrument", None) != instrument
        )

        if use_ewc_loss:
            logger.info(
                f"🧠 EWC loss enabled (cross-pair): λ={self.ewc.ewc_lambda}, "
                f"protecting {self.ewc._n_tasks} prior task(s)"
            )
            ewc_loss = create_ewc_loss(base_loss, self.ewc.penalty, ewc_weight=1.0)
            self.model.compile(optimizer=optimizer, loss=ewc_loss, metrics=["accuracy"])
        else:
            if self._is_warm_start and self._use_ewc:
                logger.info("🧠 EWC disabled for same-pair warm-start (prevents over-regularization)")
            self.model.compile(optimizer=optimizer, loss=base_loss, metrics=["accuracy"])

    def _evaluate_warm_start_baseline(
        self, x_val_filtered: np.ndarray, y_val_filtered: np.ndarray, instrument: str
    ) -> None:
        """Evaluate and set warm-start baseline accuracy."""
        if not self._is_warm_start:
            return

        try:
            loaded_model_instrument = getattr(self, "_loaded_model_instrument", None)
            is_cross_pair = loaded_model_instrument and loaded_model_instrument != instrument

            eval_results = self.model.evaluate(x_val_filtered, y_val_filtered, verbose=0)
            actual_baseline_acc = eval_results[1] if len(eval_results) > 1 else eval_results[0]

            if is_cross_pair:
                logger.info(f"🔄 Cross-pair training ({loaded_model_instrument} → {instrument})")
                logger.info(
                    f"   Stored baseline: {self._warm_start_val_acc:.1%}, "
                    f"Actual on {instrument}: {actual_baseline_acc:.1%}"
                )
                self._warm_start_val_acc = actual_baseline_acc
                logger.info(f"🎯 Using actual baseline on {instrument}: {self._warm_start_val_acc:.1%}")
            else:
                logger.info(f"🔄 Same-pair training ({instrument})")
                logger.info(
                    f"   Stored baseline: {self._warm_start_val_acc:.1%}, Current eval: {actual_baseline_acc:.1%}"
                )
                if self._warm_start_val_acc > 0:
                    logger.info(f"🎯 Using STORED baseline: {self._warm_start_val_acc:.1%} (ignoring data drift)")
                else:
                    self._warm_start_val_acc = actual_baseline_acc
                    logger.info(f"🎯 No stored baseline, using actual: {self._warm_start_val_acc:.1%}")
        except Exception as e:
            logger.warning(f"Could not evaluate baseline: {e}")

    def _initialize_ema(self) -> None:
        """Initialize EMA callback if not already loaded."""
        if self._use_ema and self.ema is None:
            self.ema = EMACallback(
                self.model,
                decay=self.config.ema_decay,
                update_every=self.config.ema_update_every,
            )
            self.lineage.ema_enabled = True

    def _create_training_callbacks(
        self,
        x_val_filtered: np.ndarray,
        y_val_filtered: np.ndarray,
    ) -> list:
        """Create all training callbacks."""
        from tensorflow import keras

        early_stop_patience = (
            self.config.patience // 2 if self._is_warm_start else self.config.patience
        )
        lr_reduce_patience = (
            max(5, self.config.patience // 4) if self._is_warm_start
            else max(4, self.config.patience // 4)
        )

        if self._is_warm_start:
            logger.info("📊 Warm-start callback adjustments:")
            logger.info(f"   Early stopping patience: {early_stop_patience} (reduced from {self.config.patience})")
            logger.info(f"   LR reduction patience: {lr_reduce_patience}")

        callbacks = [
            self._create_epoch_callback(),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=early_stop_patience,
                mode="max",
                restore_best_weights=True,
                verbose=0,
            ),
            OverfitPreventionCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                model_name="transformer_direction",
                config=OverfitPreventionConfig(
                    overfit_threshold=0.08,
                    critical_threshold=0.12,
                    severe_threshold=0.20,
                    max_acceptable_gap=0.10,
                    patience_epochs=2,
                    auto_adjust_dropout=True,
                    auto_reduce_lr=True,
                    enable_swa=True,
                    swa_start_fraction=0.5,
                    enable_cosine_restarts=True,
                    restart_period=15,
                    warm_start_best_acc=self._warm_start_val_acc,
                ),
            ),
        ]

        # Add collapse detection callbacks
        callbacks.extend(self._create_collapse_callbacks(x_val_filtered, y_val_filtered))

        # Add gradual unfreeze callback for warm-start
        self._add_unfreeze_callback(callbacks)

        # Add EMA and EWC callbacks
        self._add_ema_ewc_callbacks(callbacks)

        return callbacks

    def _create_epoch_callback(self) -> Any:
        """Create appropriate epoch display callback."""
        if not self.config.quiet:
            return RichEpochCallback(
                model_name="Transformer Direction",
                total_epochs=self.config.epochs,
                warm_start_best_acc=self._warm_start_val_acc,
                quiet=self.config.quiet,
            )
        return QuietProgressCallback(
            model_name="Transformer Direction",
            total_epochs=self.config.epochs,
        )

    def _create_collapse_callbacks(
        self, x_val_filtered: np.ndarray, y_val_filtered: np.ndarray
    ) -> list:
        """Create collapse detection and prevention callbacks."""

        # Define callbacks inline to avoid class definition complexity
        self._collapse_callback = self._make_prediction_collapse_callback(
            x_val_filtered, y_val_filtered
        )
        self._proactive_collapse_callback = self._make_proactive_collapse_callback()

        return [self._collapse_callback, self._proactive_collapse_callback]

    def _make_prediction_collapse_callback(
        self, x_val: np.ndarray, y_val: np.ndarray
    ) -> Any:
        """Create prediction collapse detection callback."""
        from tensorflow import keras

        trainer_self = self

        class PredictionCollapseCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.x_val = x_val
                self.y_val = y_val
                self.check_every = 5
                self.max_consecutive = 5
                self.warmup_epochs = 15
                self.consecutive_collapses = 0
                self.lr_reductions = getattr(trainer_self, "_restored_lr_reductions", 0)
                self.recovery_count = 0
                self.weight_perturbations = 0
                self.best_weights = None
                self.best_diversity = 0.0

            def on_epoch_end(self, epoch, logs=None):
                if epoch < self.warmup_epochs or (epoch + 1) % self.check_every != 0:
                    return
                self._check_collapse(epoch)

            def _check_collapse(self, epoch):
                try:
                    preds = self.model.predict(self.x_val, verbose=0).flatten()
                    pred_up_pct = (preds > 0.5).mean() * 100
                    diversity = min(pred_up_pct, 100 - pred_up_pct)

                    if diversity > self.best_diversity:
                        self.best_diversity = diversity
                        self.best_weights = self.model.get_weights()

                    if pred_up_pct > 95 or pred_up_pct < 5:
                        self._handle_collapse(epoch, pred_up_pct)
                    elif self.consecutive_collapses > 0:
                        self._handle_recovery(diversity)
                except Exception as e:
                    logger.warning(f"⚠️ Collapse check failed at epoch {epoch + 1}: {e}")

            def _handle_collapse(self, epoch, pred_up_pct):
                self.consecutive_collapses += 1
                collapse_dir = "UP" if pred_up_pct > 95 else "DOWN"
                logger.warning(
                    f"⚠️ PREDICTION COLLAPSE #{self.consecutive_collapses}/{self.max_consecutive} "
                    f"at epoch {epoch + 1}: {pred_up_pct:.1f}% {collapse_dir}"
                )
                self._apply_recovery_strategy()

            def _apply_recovery_strategy(self):
                if self.consecutive_collapses <= 2 and self.lr_reductions < 2:
                    self._reduce_learning_rate()
                elif self.consecutive_collapses <= 4 and self.weight_perturbations < 2:
                    self._perturb_weights()

                if self.consecutive_collapses >= self.max_consecutive:
                    self._stop_training()

            def _reduce_learning_rate(self):
                old_lr = _safe_get_learning_rate(self.model.optimizer)
                new_lr = old_lr * 0.5
                if _safe_set_learning_rate(self.model.optimizer, new_lr):
                    self.lr_reductions += 1
                    logger.warning(f"🔻 LR reduced #{self.lr_reductions}: {old_lr:.2e} → {new_lr:.2e}")

            def _perturb_weights(self):
                try:
                    weights = self.model.get_weights()
                    rng = np.random.default_rng(seed=42 + self.weight_perturbations)
                    noise_scale = 0.01 * (1 + self.weight_perturbations)
                    perturbed = [w + rng.normal(0, noise_scale, w.shape).astype(w.dtype) for w in weights]
                    self.model.set_weights(perturbed)
                    self.weight_perturbations += 1
                    logger.warning(f"🎲 Weight perturbation #{self.weight_perturbations} applied")
                except Exception as e:
                    logger.warning(f"⚠️ Weight perturbation failed: {e}")

            def _stop_training(self):
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    logger.warning(f"🔄 Restored best weights (diversity={self.best_diversity:.1f}%)")
                logger.error(f"🛑 STOPPING TRAINING: {self.consecutive_collapses} consecutive collapses")
                self.model.stop_training = True

            def _handle_recovery(self, diversity):
                self.recovery_count += 1
                logger.info(f"✅ Recovered from collapse (diversity={diversity:.1f}%)")
                self.consecutive_collapses = 0

        return PredictionCollapseCallback()

    def _make_proactive_collapse_callback(self) -> Any:
        """Create proactive collapse prevention callback."""
        from tensorflow import keras

        class ProactiveCollapsePreventionCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.check_every = 50
                self.bias_threshold = 1.5
                self.batch_count = 0
                self.interventions = 0

            def on_train_batch_end(self, batch, logs=None):
                self.batch_count += 1
                if self.batch_count % self.check_every != 0:
                    return
                self._check_bias()

            def _check_bias(self):
                try:
                    output_layer = next(
                        (layer for layer in self.model.layers if layer.name == "direction"), None
                    )
                    if output_layer is None or not hasattr(output_layer, 'bias'):
                        return

                    bias_val = float(output_layer.bias.numpy()[0])
                    if abs(bias_val) > self.bias_threshold:
                        self._intervene(output_layer, bias_val)
                except Exception:
                    pass

            def _intervene(self, output_layer, bias_val):
                self.interventions += 1
                new_bias = bias_val * 0.3
                output_layer.bias.assign([new_bias])

                kernel = output_layer.kernel.numpy()
                rng = np.random.default_rng(seed=42)
                noise = rng.normal(0, 0.01, kernel.shape)
                output_layer.kernel.assign(kernel + noise)

                logger.warning(
                    f"🔧 Proactive collapse intervention #{self.interventions}: "
                    f"bias {bias_val:.3f} → {new_bias:.3f}"
                )

        return ProactiveCollapsePreventionCallback()

    def _add_unfreeze_callback(self, callbacks: list) -> None:
        """Add gradual unfreeze callback for warm-start training."""
        if self._is_warm_start and self.config.warm_start_unfreeze_epochs > 0:
            unfreeze_callback = GradualUnfreezeCallback(
                unfreeze_after_epochs=self.config.warm_start_unfreeze_epochs,
                gradual=self.config.warm_start_gradual_unfreeze,
                learning_rate_boost=1.5,
            )
            callbacks.append(unfreeze_callback)
            logger.info(
                f"🔓 Gradual unfreeze enabled: will unfreeze after epoch "
                f"{self.config.warm_start_unfreeze_epochs}"
            )

    def _add_ema_ewc_callbacks(self, callbacks: list) -> None:
        """Add EMA and EWC monitoring callbacks."""
        from tensorflow import keras

        if self._use_ema and self.ema is not None:
            class EMAUpdateCallback(keras.callbacks.Callback):
                def __init__(self, ema_callback):
                    super().__init__()
                    self.ema_callback = ema_callback

                def on_train_batch_end(self, batch, logs=None):
                    if self.ema_callback:
                        self.ema_callback.update()

            callbacks.append(EMAUpdateCallback(self.ema))

        if (self._is_warm_start and self._use_ewc and self.ewc is not None
                and self.ewc.fisher_diagonal is not None):
            callbacks.append(EWCTrainingCallback(self.ewc, log_every=50))

    def _handle_warm_start_recovery(
        self, x_val_filtered: np.ndarray, y_val_filtered: np.ndarray
    ) -> None:
        """Restore original weights if training degraded."""
        if not (self._is_warm_start and self._warm_start_weights is not None
                and self._warm_start_val_acc > 0):
            return

        current_val_pred = (self.model.predict(x_val_filtered, verbose=0) > 0.5).astype(float)
        current_val_acc = np.mean(current_val_pred.flatten() == y_val_filtered)

        logger.info(
            f"🔍 Post-training check: current_val_acc={current_val_acc:.1%}, "
            f"warm_start_baseline={self._warm_start_val_acc:.1%}"
        )

        if current_val_acc < self._warm_start_val_acc - 0.01:
            from rich.console import Console
            console = Console()
            console.print("  [bold red]⚠️ WARM-START RECOVERY TRIGGERED[/bold red]")
            console.print(f"  [red]   Current: {current_val_acc:.1%} < Baseline: {self._warm_start_val_acc:.1%}[/red]")
            console.print("  [yellow]   Restoring original warm-start weights...[/yellow]")
            self.model.set_weights(self._warm_start_weights)
            console.print(f"  [green]✓ Model preserved at {self._warm_start_val_acc:.1%} accuracy[/green]")

    def _update_ewc_and_ema(
        self, x_train_filtered: np.ndarray, y_train_filtered: np.ndarray
    ) -> None:
        """Update EWC Fisher information and EMA weights."""
        if self._use_ewc:
            if self.ewc is None:
                self.ewc = EWCPenalty(
                    self.model,
                    ewc_lambda=self.config.ewc_lambda,
                    gamma=self.config.ewc_gamma,
                )
            self.ewc.compute_fisher(x_train_filtered, y_train_filtered, n_samples=1000)
            self.lineage.ewc_n_tasks = self.ewc._n_tasks

        if self._use_ema and self.ema is not None:
            self.ema.update(force=True)

    def _update_lineage(
        self, history: Any, x_train_filtered: np.ndarray
    ) -> None:
        """Update training lineage with session info."""
        self.lineage.session_epochs = len(history.history["loss"])
        self.lineage.cumulative_epochs += self.lineage.session_epochs
        self.lineage.cumulative_samples += len(x_train_filtered)
        if self.replay_buffer:
            self.lineage.replay_buffer_size = (
                len(self.replay_buffer.x_buffer) if self.replay_buffer.x_buffer is not None else 0
            )

    def _compute_final_metrics(
        self,
        history: Any,
        x_val_filtered: np.ndarray,
        y_val_filtered: np.ndarray,
        x_train_filtered: np.ndarray,
    ) -> Dict[str, float]:
        """Compute final training metrics."""
        import tensorflow as tf

        val_raw_pred = self.model.predict(x_val_filtered, verbose=0)
        val_pred = (val_raw_pred > 0.5).astype(float)
        val_acc = np.mean(val_pred.flatten() == y_val_filtered)

        y_true = y_val_filtered.flatten()
        y_pred = val_pred.flatten()
        np.mean(y_pred[y_true == 1] == 1) if (y_true == 1).sum() > 0 else 0
        np.mean(y_pred[y_true == 0] == 0) if (y_true == 0).sum() > 0 else 0

        raw_mean = float(np.mean(val_raw_pred))
        raw_std = float(np.std(val_raw_pred))
        raw_median = float(np.median(val_raw_pred))

        self._log_prediction_distribution(val_raw_pred, y_pred, raw_mean, raw_median, raw_std)

        # Calibration
        self.output_calibration = {
            "threshold": raw_median,
            "mean": raw_mean,
            "std": max(raw_std, 0.01),
            "enabled": abs(raw_mean - 0.5) > 0.05,
        }

        # Calibrated metrics
        val_pred_cal = (val_raw_pred.flatten() > raw_median).astype(float)
        up_acc_cal = np.mean(val_pred_cal[y_true == 1] == 1) if (y_true == 1).sum() > 0 else 0
        down_acc_cal = np.mean(val_pred_cal[y_true == 0] == 0) if (y_true == 0).sum() > 0 else 0
        balanced_acc = (up_acc_cal + down_acc_cal) / 2

        self._log_calibrated_distribution(val_pred_cal, raw_median, up_acc_cal, down_acc_cal, balanced_acc)

        metrics = {
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(val_acc),
            "val_balanced_accuracy": float(balanced_acc),
            "val_up_accuracy": float(up_acc_cal),
            "val_down_accuracy": float(down_acc_cal),
            "epochs_trained": len(history.history["loss"]),
            "n_train_samples": len(x_train_filtered),
            "n_val_samples": len(x_val_filtered),
        }

        self._add_weight_norm_metrics(metrics, tf)
        return metrics

    def _log_prediction_distribution(
        self, val_raw_pred: np.ndarray, y_pred: np.ndarray,
        raw_mean: float, raw_median: float, raw_std: float
    ) -> None:
        """Log prediction distribution for debugging."""
        long_preds = (y_pred == 1).sum()
        short_preds = (y_pred == 0).sum()
        logger.info("📊 Final validation prediction distribution:")
        logger.info(
            f"   Raw prob: mean={raw_mean:.4f}, median={raw_median:.4f}, "
            f"std={raw_std:.4f}, min={float(np.min(val_raw_pred)):.4f}, "
            f"max={float(np.max(val_raw_pred)):.4f}"
        )
        long_pct = 100 * long_preds / len(y_pred)
        short_pct = 100 * short_preds / len(y_pred)
        logger.info(f"   Predictions (thresh=0.5): LONG={long_preds} ({long_pct:.1f}%), SHORT={short_preds} ({short_pct:.1f}%)")
        if long_preds == 0 or short_preds == 0:
            logger.warning(f"   ⚠️ MODEL COLLAPSE DETECTED: Always predicting {'LONG' if long_preds > 0 else 'SHORT'}!")

    def _log_calibrated_distribution(
        self, val_pred_cal: np.ndarray, threshold: float,
        up_acc: float, down_acc: float, balanced_acc: float
    ) -> None:
        """Log calibrated prediction distribution."""
        long_preds = (val_pred_cal == 1).sum()
        short_preds = (val_pred_cal == 0).sum()
        long_pct = 100 * long_preds / len(val_pred_cal)
        short_pct = 100 * short_preds / len(val_pred_cal)
        logger.info(f"📐 Calibrated (thresh={threshold:.4f}): LONG={long_preds} ({long_pct:.1f}%), SHORT={short_preds} ({short_pct:.1f}%)")
        logger.info(f"📐 Calibrated balanced accuracy: {balanced_acc:.4f} (up={up_acc:.4f}, down={down_acc:.4f})")

    def _add_weight_norm_metrics(self, metrics: Dict[str, float], tf: Any) -> None:
        """Add weight norm metrics for regularization monitoring."""
        try:
            total_weight_norm = 0.0
            trainable_params = 0
            for layer in self.model.layers:
                for w in layer.trainable_weights:
                    total_weight_norm += float(tf.norm(w).numpy())
                    trainable_params += int(tf.size(w).numpy())
            avg_weight_norm = total_weight_norm / max(
                1, len([w for layer in self.model.layers for w in layer.trainable_weights])
            )
            metrics["total_weight_norm"] = total_weight_norm
            metrics["avg_weight_norm"] = avg_weight_norm
            logger.info(f"Weight norms: total={total_weight_norm:.2f}, avg={avg_weight_norm:.4f}")
        except Exception as e:
            logger.debug(f"Could not compute weight norms: {e}")

    def _check_and_record_drift(
        self, val_acc: float, instrument: str, x_train_filtered: np.ndarray
    ) -> None:
        """Check for drift and record training result."""
        if self.drift_detector is None:
            return

        feature_means = (
            x_train_filtered.mean(axis=(0, 1)) if len(x_train_filtered.shape) == 3
            else x_train_filtered.mean(axis=0)
        )
        self.drift_detector.record_training_result(
            val_accuracy=val_acc,
            instrument=instrument,
            data_hash=self.lineage.data_hash if self.lineage else "",
            feature_means=feature_means,
        )

        drift_detected, drift_reason = self.drift_detector.check_drift()
        if drift_detected:
            logger.warning(f"⚠️ DRIFT DETECTED: {drift_reason}")
            self.metrics["drift_detected"] = True
            self.metrics["drift_reason"] = drift_reason
            if self.lineage:
                self.lineage.drift_detected = True
                self.lineage.drift_reason = drift_reason
                self.lineage.last_drift_check = datetime.now().isoformat()
        else:
            self.metrics["drift_detected"] = False
            if self.lineage:
                self.lineage.drift_detected = False
                self.lineage.last_drift_check = datetime.now().isoformat()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        w_train: Optional[np.ndarray] = None,
        w_val: Optional[np.ndarray] = None,
        warm_start_path: Optional[str] = None,
        instrument: str = "UNKNOWN",
        data_range: str = "",
        skip_scaling: bool = False,
    ) -> Dict[str, float]:
        """Train Transformer for direction prediction with continual learning."""
        logger.info("Training Transformer (Direction)...")

        # Initialize state
        self._initialize_training_state(instrument, data_range)
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        x_train_scaled, x_val_scaled = self._scale_features(X_train, x_val, skip_scaling)

        # Handle warm-start features and feature selection
        x_train_scaled, x_val_scaled, _warm_start_features_compatible = \
            self._handle_feature_preparation(
                x_train_scaled, x_val_scaled, y_train, warm_start_path
            )

        # Create sequences and filter
        x_train_filtered, y_train_filtered, x_val_filtered, y_val_filtered = \
            self._prepare_sequences_and_filter(
                x_train_scaled, x_val_scaled, y_train, y_val, w_train, w_val
            )

        # Compute class stats and variance weight
        train_up_pct, _, _, _ = self._compute_class_statistics(y_train_filtered, y_val_filtered)
        auto_variance_weight = self._compute_auto_variance_weight(instrument, train_up_pct)
        self.lineage.auto_variance_weight = auto_variance_weight

        # Sample weights and replay buffer
        sample_weights = self._compute_sample_weights(y_train_filtered)
        logger.info(f"Sequence shape: train={x_train_filtered.shape}, val={x_val_filtered.shape}")

        if self._use_replay:
            x_train_filtered, y_train_filtered, sample_weights = self._handle_replay_buffer(
                x_train_filtered, y_train_filtered, sample_weights, instrument
            )

        self.lineage.data_hash = TrainingLineage.compute_data_hash(x_train_filtered, y_train_filtered)

        # Build and configure model
        self.model = self._build_model((self.seq_len, self.n_features))
        effective_lr = self._setup_warm_start(warm_start_path, _warm_start_features_compatible)

        # Setup optimizer and compile
        optimizer = self._setup_optimizer_with_warmup(effective_lr, x_train_filtered)
        label_smoothing = getattr(self.config, "label_smoothing", 0.05) if self.config else 0.05
        base_loss = self._create_loss_function(auto_variance_weight, label_smoothing)
        self._compile_model_with_loss(optimizer, base_loss, instrument)

        # Evaluate warm-start baseline
        self._evaluate_warm_start_baseline(x_val_filtered, y_val_filtered, instrument)
        self.model.summary(print_fn=logger.info)

        # Initialize EMA and create callbacks
        self._initialize_ema()
        callbacks = self._create_training_callbacks(x_val_filtered, y_val_filtered)

        # === PHASE 4: DATA AUGMENTATION FOR SMALLER DATASET ===
        augment_fn = self._create_augmentation_fn()
        if augment_fn is not None:
            logger.info("🎨 Time-series augmentation enabled (noise, scaling, time masking)")
            # Create tf.data.Dataset with augmentation
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train_filtered, y_train_filtered))
            train_dataset = train_dataset.shuffle(buffer_size=len(x_train_filtered))
            train_dataset = train_dataset.batch(self.config.batch_size)
            train_dataset = train_dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
            
            # Validation dataset (no augmentation)
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val_filtered, y_val_filtered))
            val_dataset = val_dataset.batch(self.config.batch_size)
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
            
            # Train with augmented dataset
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=0,
            )
        else:
            # Train without augmentation (original method)
            history = self.model.fit(
                x_train_filtered, y_train_filtered,
                validation_data=(x_val_filtered, y_val_filtered),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=0,
                sample_weight=sample_weights,
            )

        self.is_trained = True

        # Post-training: recovery, EWC, EMA, lineage
        self._handle_warm_start_recovery(x_val_filtered, y_val_filtered)
        self._update_ewc_and_ema(x_train_filtered, y_train_filtered)
        self._update_lineage(history, x_train_filtered)

        # Compute metrics
        self.metrics = self._compute_final_metrics(
            history, x_val_filtered, y_val_filtered, x_train_filtered
        )

        # Drift detection
        self._check_and_record_drift(self.metrics["val_accuracy"], instrument, x_train_filtered)

        logger.info(
            f"Transformer trained [canonical]: val_accuracy={self.metrics['val_accuracy']:.4f}, "
            f"balanced_acc={self.metrics['val_balanced_accuracy']:.4f}"
        )
        return self.metrics

    def _handle_feature_preparation(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        y_train: np.ndarray,
        warm_start_path: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Handle feature selection and warm-start feature compatibility."""
        use_feature_selection = getattr(self.config, "use_feature_selection", True) if self.config else True
        feature_selection_method = getattr(self.config, "feature_selection_method", "random_forest") if self.config else "random_forest"
        top_k_features = getattr(self.config, "top_k_features", 50) if self.config else 50

        _warm_start_feature_names, _warm_start_scaler = self._load_warm_start_feature_meta(warm_start_path)
        _warm_start_features_compatible = False

        if _warm_start_feature_names and self.feature_names:
            x_train_scaled, x_val_scaled, _warm_start_features_compatible = \
                self._apply_warm_start_features(
                    x_train_scaled, x_val_scaled, _warm_start_feature_names, _warm_start_scaler
                )
            if _warm_start_features_compatible:
                use_feature_selection = False

        if use_feature_selection and x_train_scaled.shape[-1] > top_k_features:
            x_train_scaled, x_val_scaled = self._apply_feature_selection(
                x_train_scaled, x_val_scaled, y_train, top_k_features, feature_selection_method
            )
            if self.scaler is not None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                x_train_scaled = self.scaler.fit_transform(x_train_scaled)
                x_val_scaled = self.scaler.transform(x_val_scaled)
                logger.info(f"✓ Re-fitted scaler on {x_train_scaled.shape[-1]} selected features")
            logger.info(f"🔍 Feature Selection Complete: train={x_train_scaled.shape}, val={x_val_scaled.shape}")

        return x_train_scaled, x_val_scaled, _warm_start_features_compatible

    def _setup_warm_start(
        self, warm_start_path: Optional[str], features_compatible: bool
    ) -> float:
        """Setup warm-start training if applicable."""
        self._is_warm_start = False
        self._warm_start_val_acc = 0.0
        self._warm_start_weights = None
        self._loaded_model_instrument = None
        effective_lr = self.config.learning_rate

        if warm_start_path and Path(warm_start_path).exists():
            if not features_compatible:
                logger.warning("🔥 WARM-START SKIPPED: Feature dimensions incompatible")
                logger.info("   Training will start fresh with new architecture")
            else:
                effective_lr = self._handle_warm_start(warm_start_path)
        elif warm_start_path:
            logger.info(f"No checkpoint found at {warm_start_path}. Starting fresh training.")

        return effective_lr

    def predict(self, X: np.ndarray, use_ema: bool = True) -> Dict[str, Any]:
        """
        Predict direction (0 or 1) with probability.

        Applies output calibration if enabled to correct for systematic bias.

        Args:
            X: Input features
            use_ema: If True and EMA is available, use EMA weights for stable inference

        Returns:
            Dict with 'direction' (0 or 1) and 'probability' (0.0 to 1.0)
        """
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Apply feature selection if used during training
        x_reshaped = X.reshape(-1, X.shape[-1])

        # Get expected number of features from model input shape
        model_n_features = self.model.input_shape[-1]
        current_n_features = x_reshaped.shape[-1]

        # Apply feature selection if input has more features than model expects
        if current_n_features > model_n_features:
            if (
                self.selected_indices is not None
                and len(self.selected_indices) == model_n_features
            ):
                try:
                    x_reshaped = x_reshaped[:, self.selected_indices]
                except (IndexError, ValueError) as e:
                    logger.warning(f"Feature selection failed: {e}")
            else:
                # No valid selection indices - try to use first N features as fallback
                logger.warning(
                    f"Feature count mismatch: input has {current_n_features}, "
                    f"model expects {model_n_features}. Using first {model_n_features} features."
                )
                x_reshaped = x_reshaped[:, :model_n_features]

        # Scale
        x_scaled = self.scaler.transform(x_reshaped)

        # Check model input shape to determine if sequence or flat input
        model_input_shape = self.model.input_shape
        is_flat_model = len(model_input_shape) == 2  # (None, n_features)

        if is_flat_model:
            # Flat input model - use last row only
            x_input = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled
        else:
            # Sequence model - create sequence from last seq_len rows
            if len(x_scaled) >= self.seq_len:
                x_input = x_scaled[-self.seq_len :].reshape(1, self.seq_len, -1)
            else:
                # Pad with zeros if not enough data
                pad_len = self.seq_len - len(x_scaled)
                x_padded = np.vstack([np.zeros((pad_len, x_scaled.shape[1])), x_scaled])
                x_input = x_padded.reshape(1, self.seq_len, -1)

        # Use EMA weights for stable inference if available
        use_ema_weights = (
            use_ema
            and self._use_ema
            and self.ema is not None
            and self.ema._initialized
            and self.config.use_ema_for_inference
        )

        if use_ema_weights:
            self.ema.apply()  # Apply EMA weights

        try:
            prob_raw = float(self.model.predict(x_input, verbose=0)[0, 0])
        finally:
            if use_ema_weights:
                self.ema.restore()  # Restore training weights

        # === APPLY OUTPUT CALIBRATION ===
        # Use adaptive threshold instead of shifting probabilities
        calibration = getattr(self, "output_calibration", None)
        threshold = 0.5  # Default threshold
        if calibration and calibration.get("enabled", False):
            threshold = calibration.get("threshold", 0.5)

        direction = 1 if prob_raw > threshold else 0

        # For confidence, measure distance from threshold (not from 0.5)
        # Normalize to 0-1 range based on typical distribution
        std = calibration.get("std", 0.15) if calibration else 0.15
        confidence_distance = abs(prob_raw - threshold) / (
            2 * std
        )  # Normalize by 2 std
        confidence = min(1.0, confidence_distance)  # Cap at 1.0

        return {
            "direction": direction,
            "probability": prob_raw,  # Raw probability
            "probability_raw": prob_raw,  # Alias for debugging
            "confidence": confidence,  # Normalized confidence
            "threshold": threshold,  # Calibrated threshold
            "ema_used": use_ema_weights,
            "calibration_applied": calibration.get("enabled", False)
            if calibration
            else False,
        }

    def save(self, path: str, instrument: str = "UNKNOWN") -> None:
        """
        Save Transformer model with all continual learning state.

        Saves:
        - .keras: Main model weights
        - .meta.pkl: Scaler, config, metrics, lineage
        - .ema.pkl: EMA shadow weights
        - .ewc.pkl: EWC Fisher information + reference weights
        - replay buffer to trained_data/replay/<instrument>/
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # === Persist callback state to lineage for warm-start ===
        if hasattr(self, "_collapse_callback") and self._collapse_callback is not None:
            self.lineage.lr_reductions_count = self._collapse_callback.lr_reductions
            self.lineage.collapse_recovery_count = (
                self._collapse_callback.recovery_count
            )
        try:
            self.lineage.final_learning_rate = _safe_get_learning_rate(
                self.model.optimizer, default=0.0
            )
        except Exception:
            self.lineage.final_learning_rate = 0.0

        logger.info(
            f"💾 Persisting training state: "
            f"variance_weight={getattr(self.lineage, 'auto_variance_weight', 0.0):.3f}, "
            f"lr_reductions={getattr(self.lineage, 'lr_reductions_count', 0)}, "
            f"final_lr={getattr(self.lineage, 'final_learning_rate', 0.0):.2e}"
        )

        # Save Keras model in native format
        self.model.save(str(path))

        # === CROSS-VERSION COMPATIBILITY: Save weights and architecture separately ===
        # This allows loading on different Keras versions (2.x vs 3.x)

        # 1. Save weights in H5 format (portable across Keras versions)
        weights_path = path.with_suffix(WEIGHTS_H5_SUFFIX)
        try:
            self.model.save_weights(str(weights_path))
            logger.debug(f"Saved portable weights to {weights_path}")
        except Exception as e:
            logger.warning(f"Could not save portable weights: {e}")

        # 2. Save architecture as JSON (for rebuilding on different Keras version)
        arch_path = path.with_suffix(ARCH_JSON_SUFFIX)
        try:
            arch_json = self.model.to_json()
            with open(arch_path, "w") as f:
                f.write(arch_json)
            logger.debug(f"Saved architecture to {arch_path}")
        except Exception as e:
            logger.warning(f"Could not save architecture JSON: {e}")

        # Update lineage metrics before saving
        if self.lineage:
            self.lineage.add_metrics(self.metrics)

        # Save scaler, config, and lineage (with architecture config for cross-version rebuild)
        meta = {
            "scaler": self.scaler,
            "seq_len": self.seq_len,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "selected_indices": getattr(
                self, "selected_indices", None
            ),  # Feature selection indices
            "model_type": "transformer",
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "is_warm_start": self._is_warm_start,
            "ema_enabled": self._use_ema,
            "ewc_enabled": self._use_ewc,
            "replay_enabled": self._use_replay,
            "output_calibration": getattr(
                self, "output_calibration", None
            ),  # Save calibration params
            # Store transformer architecture config for programmatic rebuild
            "architecture": {
                "input_shape": (self.seq_len, self.n_features),
                "transformer_d_model": self.transformer_d_model,
                "transformer_num_heads": self.transformer_num_heads,
                "transformer_num_layers": self.transformer_num_layers,
                "transformer_dff": self.transformer_dff,
                "transformer_dropout": self.transformer_dropout,
            },
        }
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        # Save EMA weights if available
        if self._use_ema and self.ema is not None and self.ema._initialized:
            ema_data = {
                "ema_weights": self.ema.get_ema_weights(),
                "ema_weight_names": [
                    w.name for w in self.model.trainable_weights
                ],  # For graceful loading
                "decay": self.ema.decay,
                "update_every": self.ema.update_every,
                "step_counter": self.ema.step_counter,
            }
            ema_path = path.with_suffix(EMA_PKL_SUFFIX)
            with open(ema_path, "wb") as f:
                pickle.dump(ema_data, f)
            logger.info(f"📊 EMA weights saved to {ema_path}")

        # Save EWC state if available
        if (
            self._use_ewc
            and self.ewc is not None
            and self.ewc.fisher_diagonal is not None
        ):
            ewc_path = path.with_suffix(EWC_PKL_SUFFIX)
            self.ewc.save(str(ewc_path))

        # Save replay buffer
        if self._use_replay and self.replay_buffer is not None:
            self.replay_buffer.save(instrument)

        logger.info(
            f"✅ Transformer saved to {path} "
            f"(EMA={self._use_ema}, EWC={self._use_ewc}, Replay={self._use_replay}, cross-version=True)"
        )

    def load(self, path: str, instrument: str = "UNKNOWN") -> None:
        """
        Load Transformer model with all continual learning state.

        Loads:
        - .keras: Main model weights
        - .meta.pkl: Scaler, config, metrics, lineage
        - .ema.pkl: EMA shadow weights (for inference)
        - .ewc.pkl: EWC state (for future warm-start)
        - replay buffer from trained_data/replay/<instrument>/

        Handles Keras 2.x/3.x compatibility for models trained on Colab.
        """
        import tensorflow as tf
        from tensorflow import keras

        path = Path(path)

        # Try multiple loading strategies for Keras 2.x/3.x compatibility
        model = None
        load_errors = []

        # Strategy 0: Use cross-version loader (handles Keras 2.15.0 -> 3.x migration)
        try:
            from src.utils.keras_model_loader import load_keras_model

            model, load_metadata = load_keras_model(
                str(path),
                compile=False,
                # Let the loader auto-detect the best approach based on Keras version
                # For Keras 3.x, this will try keras_native first
            )
            if load_metadata.get("success"):
                logger.info(
                    f"✓ Model loaded with cross-version loader ({load_metadata.get('approach_used')})"
                )
            else:
                model = None
        except ImportError:
            load_errors.append("Cross-version loader not available")
        except Exception as e:
            load_errors.append(f"Cross-version: {e}")
            model = None

        # Strategy 1: Standard load (works if same Keras version)
        if model is None:
            try:
                model = keras.models.load_model(str(path), compile=False)
                logger.info("✓ Model loaded with standard loader")
            except Exception as e:
                load_errors.append(f"Standard: {e}")

        # Strategy 2: Use tf.keras.models.load_model (TF-native)
        if model is None:
            try:
                model = tf.keras.models.load_model(str(path), compile=False)
                logger.info("✓ Model loaded with tf.keras loader")
            except Exception as e:
                load_errors.append(f"TF-native: {e}")

        # Strategy 3: Load with safe_mode=False for Keras 3 models
        if model is None:
            try:
                model = keras.models.load_model(
                    str(path), compile=False, safe_mode=False
                )
                logger.info("✓ Model loaded with safe_mode=False")
            except Exception as e:
                load_errors.append(f"Safe-mode: {e}")

        # Strategy 3.5: Load from arch.json + weights.h5 (cross-version portable format)
        arch_path = path.with_suffix(ARCH_JSON_SUFFIX)
        weights_path = path.with_suffix(WEIGHTS_H5_SUFFIX)
        if model is None and arch_path.exists() and weights_path.exists():
            try:
                with open(arch_path) as f:
                    arch_json = f.read()
                model = keras.models.model_from_json(arch_json)
                model.load_weights(str(weights_path))
                logger.info(
                    "✓ Model loaded from arch.json + weights.h5 (cross-version)"
                )
            except Exception as e:
                load_errors.append(f"Arch+Weights: {e}")

        # Strategy 4: Rebuild model from metadata and load weights only
        if model is None:
            meta_path = path.with_suffix(META_PKL_SUFFIX)
            if meta_path.exists():
                try:
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)

                    n_features = meta.get("n_features", 59)
                    seq_len = meta.get("seq_len", 60)
                    config = meta.get("config", {})

                    # Get transformer hyperparams from config
                    d_model = config.get("transformer_d_model", 32)
                    num_heads = config.get("transformer_num_heads", 4)
                    num_layers = config.get("transformer_num_layers", 2)
                    dff = config.get("transformer_dff", 64)
                    dropout = config.get("transformer_dropout", 0.2)

                    # Get output head config (critical for EMA weight compatibility)
                    final_dense_units = config.get("final_dense_units", 16)
                    final_dense_activation = config.get(
                        "final_dense_activation", "tanh"
                    )
                    final_dense_dropout = config.get("final_dense_dropout", 0.15)

                    logger.info(
                        f"Rebuilding Transformer: n_features={n_features}, seq_len={seq_len}, "
                        f"d_model={d_model}, final_dense={final_dense_units}"
                    )

                    # Rebuild the actual Transformer architecture
                    inp = keras.Input(shape=(seq_len, n_features), name="features")
                    x = keras.layers.GaussianNoise(0.15)(inp)
                    x = keras.layers.SpatialDropout1D(0.2)(x)

                    # Input projection
                    x = keras.layers.Dense(d_model, name="input_projection")(x)
                    x = keras.layers.Dropout(0.3)(x)

                    # Add positional encoding (simplified for rebuild)
                    positions = np.arange(seq_len)[:, np.newaxis]
                    dims = np.arange(d_model)[np.newaxis, :]
                    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
                    pos_encoding = np.zeros((seq_len, d_model))
                    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
                    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
                    pos_encoding = pos_encoding[np.newaxis, :, :].astype(np.float32)
                    x = x + tf.constant(pos_encoding)

                    # Transformer encoder layers
                    for i in range(num_layers):
                        # Multi-head attention
                        attn_output = keras.layers.MultiHeadAttention(
                            num_heads=num_heads,
                            key_dim=d_model // num_heads,
                            name=f"transformer_{i}_mha",
                        )(x, x)
                        attn_output = keras.layers.Dropout(dropout)(attn_output)
                        x = keras.layers.LayerNormalization(
                            epsilon=1e-6, name=f"transformer_{i}_ln1"
                        )(x + attn_output)

                        # FFN
                        ffn = keras.layers.Dense(
                            dff, activation="relu", name=f"transformer_{i}_ffn1"
                        )(x)
                        ffn = keras.layers.Dense(d_model, name=f"transformer_{i}_ffn2")(
                            ffn
                        )
                        ffn = keras.layers.Dropout(dropout)(ffn)
                        x = keras.layers.LayerNormalization(
                            epsilon=1e-6, name=f"transformer_{i}_ln2"
                        )(x + ffn)

                    # Global pooling and output
                    x = keras.layers.GlobalAveragePooling1D()(x)
                    x = keras.layers.Dense(
                        final_dense_units, activation=final_dense_activation
                    )(x)
                    x = keras.layers.Dropout(final_dense_dropout)(x)
                    direction = keras.layers.Dense(
                        1, activation="sigmoid", name="direction", dtype="float32"
                    )(x)

                    model = keras.Model(
                        inputs=inp, outputs=direction, name="transformer_direction"
                    )
                    logger.info("✓ Model architecture rebuilt from metadata")

                    # If model was rebuilt, we need to load weights from EMA
                    ema_path = path.with_suffix(EMA_PKL_SUFFIX)
                    if ema_path.exists():
                        with open(ema_path, "rb") as f:
                            ema_data = pickle.load(f)
                        ema_weights = ema_data.get("ema_weights", [])

                        # Try to apply EMA weights directly to rebuilt model
                        # Cast to match dtype for Metal/Keras 3.x compatibility
                        model_weights = model.trainable_weights
                        if len(ema_weights) == len(model_weights):
                            loaded_count = 0
                            skipped_count = 0
                            for w, ema_w in zip(model_weights, ema_weights):
                                # Shape validation safety net
                                if w.shape != tuple(ema_w.shape):
                                    logger.warning(
                                        f"Shape mismatch for {w.name}: model={w.shape}, EMA={ema_w.shape}"
                                    )
                                    skipped_count += 1
                                    continue
                                try:
                                    w.assign(ema_w.astype(_get_numpy_dtype(w.dtype)))
                                    loaded_count += 1
                                except Exception as assign_err:
                                    logger.warning(
                                        f"Could not assign EMA weight to {w.name}: {assign_err}"
                                    )
                                    skipped_count += 1
                            if skipped_count > 0:
                                logger.warning(
                                    f"⚠️ Loaded {loaded_count} EMA weights, "
                                    f"skipped {skipped_count} due to shape mismatch"
                                )
                            else:
                                logger.info(
                                    f"✓ Loaded {loaded_count} EMA weights into rebuilt model"
                                )
                        else:
                            logger.debug(
                                f"EMA weights count ({len(ema_weights)}) != "
                                f"model weights ({len(model_weights)}), will re-init"
                            )

                except Exception as e:
                    load_errors.append(f"Rebuild: {e}")

        if model is None:
            all_errors = "; ".join(load_errors)
            raise RuntimeError(
                f"Failed to load model from {path}. Errors: {all_errors}"
            )

        self.model = model

        # Load metadata
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.scaler = meta["scaler"]
        self.seq_len = meta["seq_len"]
        self.metrics = meta["metrics"]
        self.feature_names = meta.get("feature_names")
        self.n_features = meta.get("n_features")
        self.selected_indices = meta.get(
            "selected_indices"
        )  # Feature selection indices
        self._is_warm_start = meta.get("is_warm_start", False)
        self._use_ema = meta.get("ema_enabled", True)
        self._use_ewc = meta.get("ewc_enabled", True)
        self._use_replay = meta.get("replay_enabled", True)

        # Load output calibration (for bias correction)
        self.output_calibration = meta.get("output_calibration", None)
        if self.output_calibration and self.output_calibration.get("enabled"):
            logger.info(
                f"📐 Output calibration loaded: bias={self.output_calibration['bias']:.4f}"
            )

        # Load lineage
        if meta.get("lineage"):
            self.lineage = TrainingLineage.from_dict(meta["lineage"])
            logger.info(
                f"📊 Lineage loaded: checkpoint={self.lineage.checkpoint_id}, "
                f"cumulative_epochs={self.lineage.cumulative_epochs}"
            )

        # Load EMA weights
        ema_path = path.with_suffix(EMA_PKL_SUFFIX)
        if ema_path.exists() and self._use_ema:
            with open(ema_path, "rb") as f:
                ema_data = pickle.load(f)
            self.ema = EMACallback(
                self.model,
                decay=ema_data.get("decay", self.config.ema_decay),
                update_every=ema_data.get("update_every", self.config.ema_update_every),
            )
            # Pass weight names for graceful mismatch handling
            self.ema.set_ema_weights(
                ema_data["ema_weights"], weight_names=ema_data.get("ema_weight_names")
            )
            self.ema.step_counter = ema_data.get("step_counter", 0)
            logger.info(f"📊 EMA weights loaded (decay={self.ema.decay})")

        # Load EWC state
        ewc_path = path.with_suffix(EWC_PKL_SUFFIX)
        if ewc_path.exists() and self._use_ewc:
            self.ewc = EWCPenalty(
                self.model,
                ewc_lambda=self.config.ewc_lambda,
                gamma=self.config.ewc_gamma,
            )
            self.ewc.load(str(ewc_path))

        # Load replay buffer
        if self._use_replay:
            self.replay_buffer = ReplayBuffer(
                capacity_ratio=self.config.replay_buffer_ratio,
                mix_ratio=self.config.replay_mix_ratio,
                buffer_dir=self.config.replay_buffer_dir,
            )
            self.replay_buffer.load(instrument)

        self.is_trained = True

        logger.info(f"✅ Transformer loaded from {path}")

        # Check for drift if lineage exists
        if self.lineage and self.metrics.get("val_accuracy"):
            if self.lineage.check_drift(
                self.metrics["val_accuracy"], self.config.drift_threshold
            ):
                logger.info(
                    "📊 Model performance below historical best - consider retraining if persistent"
                )

# EOF
