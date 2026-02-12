"""
TCN Trainer for Volatility Regime Classification.

This module implements a Temporal Convolutional Network (TCN) trainer for
predicting current volatility regimes. The model serves as an entry timing
filter, allowing trades only when volatility is HIGH or EXTREME.

Architecture:
    - Dilated causal convolutions for multi-scale pattern recognition
    - 4-class classification: LOW, NORMAL, HIGH, EXTREME
    - Moderate regularization (volatility has cleaner patterns than direction)

Usage:
    trainer = TCNTrainer(config)
    metrics = trainer.train(X_train, y_train, X_val, y_val)
    result = trainer.predict(X_new)

Classes:
    TCNTrainer: Main trainer class for volatility regime classification
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig, OverfitPreventionConfig
from src.training.trainers.callbacks import (
    RichEpochCallback,
    QuietProgressCallback,
    OverfitPreventionCallback,
)
from src.training.trainers.utils import (
    MODEL_NOT_TRAINED_ERROR,
    META_PKL_SUFFIX,
    WEIGHTS_H5_SUFFIX,
    ARCH_JSON_SUFFIX,
    WEIGHTS_LOADED_FULL_MODEL_MSG,
    create_sequences,
    get_config_seq_len,
    _safe_load_weights_ignoring_optimizer,
    _safe_reset_optimizer_state,
    _validate_weight_shapes,
    _get_numpy_dtype,
)

logger = logging.getLogger(__name__)


class TCNTrainer(BaseTrainer):
    """
    TCN model for volatility regime classification.

    Purpose: Entry timing filter - only trade when volatility is HIGH or EXTREME.
    Addresses timing issues where trades are closed early due to unfavorable conditions.

    Input: OHLCV features with ATR-based volatility indicators
    Output: 4-class regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)

    The Transformer handles direction prediction alone - TCN is purely a timing gate.
    """

    # Class constants for regime names
    REGIME_NAMES = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None  # Save feature names for inference
        self.n_classes = 4  # LOW, NORMAL, HIGH, EXTREME

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        Build TCN model for 4-class volatility regime classification.

        Architecture optimized for volatility pattern recognition:
        - Dilated causal convolutions capture multi-scale volatility patterns
        - 4-class softmax output for regime classification
        - Moderate regularization (volatility has cleaner patterns than direction)
        """
        from tensorflow import keras
        regularizers = keras.regularizers

        seq_len, n_features = input_shape

        # Get config values with sensible defaults
        filters = getattr(
            self.config, "tcn_hidden_size", 32
        )  # Increased for regime patterns
        kernel_size = getattr(self.config, "tcn_kernel_size", 3)
        num_layers = getattr(
            self.config, "tcn_num_layers", 2
        )  # More layers for volatility
        dropout = getattr(
            self.config, "tcn_dropout", 0.4
        )  # Reduced - regimes are more learnable
        l2_reg = getattr(self.config, "tcn_l2_reg", 0.005)  # Lighter regularization
        spatial_dropout = getattr(self.config, "tcn_spatial_dropout", 0.2)
        noise_std = getattr(self.config, "tcn_noise_std", 0.03)

        # L2 regularizer for kernels
        kernel_reg = regularizers.l2(l2_reg)

        inp = keras.Input(shape=(seq_len, n_features), name="features")

        # === INPUT REGULARIZATION ===
        x = keras.layers.GaussianNoise(noise_std)(inp)
        x = keras.layers.SpatialDropout1D(spatial_dropout)(x)

        # === TCN LAYERS with dilated causal convolutions ===
        for i in range(num_layers):
            dilation_rate = 2**i
            x = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="causal",
                dilation_rate=dilation_rate,
                activation="relu",
                kernel_regularizer=kernel_reg,
                name=f"tcn_conv_{i}",
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(dropout)(x)

        # === OUTPUT HEAD for 4-class regime classification ===
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Dense layer with moderate capacity
        x = keras.layers.Dense(32, activation="relu", kernel_regularizer=kernel_reg)(x)
        x = keras.layers.Dropout(dropout)(x)

        # 4-class softmax output for volatility regime
        volatility_regime = keras.layers.Dense(
            self.n_classes,  # 4 classes: LOW, NORMAL, HIGH, EXTREME
            activation="softmax",
            name="volatility_regime",
            dtype="float32",
        )(x)

        model = keras.Model(
            inputs=inp, outputs=volatility_regime, name="tcn_volatility_regime"
        )

        # Compile with sparse categorical crossentropy (integer labels)
        lr = self.config.learning_rate

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute inverse-frequency class weights for ALL 4 volatility regimes.

        Always returns weights for classes 0-3 (LOW, NORMAL, HIGH, EXTREME),
        even if some classes are not present in the training data.
        Missing classes get weight 1.0 (neutral).
        """
        unique, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_expected_classes = 4  # Always 4 classes: LOW, NORMAL, HIGH, EXTREME

        # Start with neutral weights for ALL 4 classes
        weights = dict.fromkeys(range(n_expected_classes), 1.0)

        # Update weights for classes that exist in training data
        for cls, count in zip(unique, counts):
            cls_int = int(cls)
            if 0 <= cls_int < n_expected_classes:
                # Inverse frequency with cap
                weight = n_samples / (n_expected_classes * count)
                weights[cls_int] = min(weight, 5.0)  # Cap at 5x

        logger.info(f"Volatility regime class weights: {weights}")
        return weights

    def _validate_labels(self, y_train: np.ndarray) -> None:
        """Validate that training labels are valid regime classes."""
        unique_labels = np.unique(y_train)
        valid_classes = set(range(self.n_classes))
        invalid_labels = [
            label
            for label in unique_labels
            if label not in valid_classes and int(label) != label
        ]
        if invalid_labels:
            label_dist = dict(zip(*np.unique(y_train, return_counts=True)))
            logger.error(f"Invalid labels found: {invalid_labels}. Expected integers 0-{self.n_classes - 1}.")
            logger.error(f"Label distribution: {label_dist}")
            logger.error(f"Label dtype: {y_train.dtype}, min={np.min(y_train):.4f}, max={np.max(y_train):.4f}")
            logger.error("HINT: If you see 0.5 labels, direction data (0.0/0.5/1.0) is being passed to TCN.")
            raise ValueError(
                f"TCN labels must be integers 0-{self.n_classes - 1}, but found: {invalid_labels}.\n"
                f"Label distribution: {label_dist}\n"
                f"Fix: Ensure load_volatility_regime_data() is used when training TCN volatility filter."
            )

    def _log_class_distribution(self, y_train: np.ndarray) -> None:
        """Log the class distribution for training data."""
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            logger.info(f"  Train {self.REGIME_NAMES.get(int(cls), cls)}: {count} ({pct:.1f}%)")

    def _try_load_h5_weights(self, warm_start_path: str) -> bool:
        """Try loading weights from .weights.h5 companion file."""
        weights_h5_path = Path(warm_start_path).with_suffix(WEIGHTS_H5_SUFFIX)
        if not weights_h5_path.exists():
            return False
        if _safe_load_weights_ignoring_optimizer(self.model, str(weights_h5_path)):
            logger.info(f"✓ Loaded weights from {weights_h5_path}")
            _safe_reset_optimizer_state(self.model)
            return True
        return False

    def _try_load_direct_weights(self, warm_start_path: str) -> bool:
        """Try loading weights from .keras file directly."""
        if _safe_load_weights_ignoring_optimizer(self.model, str(warm_start_path)):
            logger.info(f"✓ Loaded weights from {warm_start_path}")
            _safe_reset_optimizer_state(self.model)
            return True
        return False

    def _try_load_full_model_weights(self, warm_start_path: str, keras_module: Any) -> bool:
        """Try full model load and extract weights (bypasses optimizer state)."""
        try:
            existing_model = keras_module.models.load_model(warm_start_path, compile=False)
            checkpoint_weights = existing_model.get_weights()
            model_weights = self.model.get_weights()

            is_compatible, shape_error = _validate_weight_shapes(
                model_weights, checkpoint_weights, context="model weights"
            )

            if is_compatible:
                self.model.set_weights(checkpoint_weights)
                logger.info(WEIGHTS_LOADED_FULL_MODEL_MSG)
                del existing_model
                return True

            # Partial transfer by name matching
            logger.warning(f"Architecture mismatch: {shape_error}")
            logger.info("🔄 Attempting partial weight transfer by layer name...")
            ckpt_weight_map = {w.name: w.numpy() for w in existing_model.weights}
            loaded, skipped = 0, 0
            for w in self.model.weights:
                if w.name in ckpt_weight_map and w.shape == ckpt_weight_map[w.name].shape:
                    w.assign(ckpt_weight_map[w.name].astype(_get_numpy_dtype(w.dtype)))
                    loaded += 1
                else:
                    skipped += 1
            del existing_model

            if loaded > 0:
                logger.info(
                    f"✓ Partial weight transfer: {loaded} loaded, {skipped} re-initialized"
                )
                return True
            logger.warning("⚠️ No compatible weights found in full model")
            return False
        except Exception as e:
            logger.debug(f"Full model load failed: {e}")
            return False

    def _load_warm_start_metadata(self, warm_start_path: str) -> float:
        """Load previous best accuracy from metadata file."""
        meta_path = Path(warm_start_path).with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return 0.0
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            prev_val_acc = meta.get("metrics", {}).get("val_accuracy", 0.0)
            if prev_val_acc > 0:
                logger.info(f"🎯 Warm-start baseline: {prev_val_acc:.1%}")
            return prev_val_acc
        except Exception as e:
            logger.warning(f"Could not load warm-start metadata: {e}")
            return 0.0

    def _load_warm_start_weights(
        self, warm_start_path: str, keras_module: Any
    ) -> Tuple[bool, float]:
        """Load weights from warm-start checkpoint.

        Returns:
            Tuple of (weights_loaded, previous_val_acc)
        """
        # Try loading strategies in order of preference
        weights_loaded = (
            self._try_load_h5_weights(warm_start_path)
            or self._try_load_direct_weights(warm_start_path)
            or self._try_load_full_model_weights(warm_start_path, keras_module)
        )

        # Load previous best accuracy from metadata if weights loaded
        prev_val_acc = self._load_warm_start_metadata(warm_start_path) if weights_loaded else 0.0

        return weights_loaded, prev_val_acc

    def _create_tcn_callbacks(
        self, keras_module: Any
    ) -> List[Any]:
        """Create callbacks for TCN training."""
        return [
            RichEpochCallback(
                model_name="TCN Volatility Regime",
                total_epochs=self.config.epochs,
                warm_start_best_acc=self._warm_start_val_acc,
                quiet=self.config.quiet,
            )
            if not self.config.quiet
            else QuietProgressCallback(
                model_name="TCN Volatility Regime",
                total_epochs=self.config.epochs,
            ),
            keras_module.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                mode="min",
                restore_best_weights=True,
                verbose=0,
            ),
            OverfitPreventionCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                model_name="tcn_volatility_regime",
                config=OverfitPreventionConfig(
                    overfit_threshold=0.10,
                    critical_threshold=0.15,
                    severe_threshold=0.25,
                    max_acceptable_gap=0.12,
                    patience_epochs=2,
                    auto_adjust_dropout=True,
                    auto_reduce_lr=True,
                    enable_swa=True,
                    swa_start_fraction=0.5,
                    enable_cosine_restarts=True,
                    restart_period=15,
                ),
            ),
        ]

    def _load_model_native(self, path: Path, keras_module: Any) -> bool:
        """Try loading model in native .keras format."""
        try:
            self.model = keras_module.models.load_model(str(path), compile=False)
            logger.info(f"TCN Volatility Regime loaded from {path} (native format, compile=False)")
            return True
        except Exception as e:
            logger.warning(f"Could not load native .keras format: {e}")
            logger.info("Attempting cross-version load from weights + architecture...")
            return False

    def _load_model_from_arch_json(
        self, arch_path: Path, weights_path: Path, meta: Dict, keras_module: Any
    ) -> bool:
        """Try loading model from architecture JSON and weights."""
        if not (arch_path.exists() and weights_path.exists()):
            return False
        try:
            with open(arch_path) as f:
                arch_json = f.read()
            self.model = keras_module.models.model_from_json(arch_json)
            self.model.load_weights(str(weights_path))
            lr = meta.get("config", {}).get("learning_rate", 0.0003)
            self.model.compile(
                optimizer=keras_module.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.info(f"TCN Volatility Regime loaded from {weights_path} (cross-version: arch.json + weights)")
            return True
        except Exception as e:
            logger.warning(f"Could not load from arch.json + weights: {e}")
            return False

    def _load_model_rebuild_architecture(
        self, weights_path: Path, meta: Dict
    ) -> bool:
        """Try rebuilding architecture and loading weights."""
        if not weights_path.exists():
            return False
        try:
            arch_cfg = meta.get("architecture", {})
            input_shape = arch_cfg.get("input_shape", (self.seq_len, self.n_features))
            if self.config is None:
                self.config = TrainerConfig()
            for key, value in arch_cfg.items():
                if key != "input_shape":
                    setattr(self.config, key, value)
            self.model = self._build_model(input_shape)
            self.model.load_weights(str(weights_path))
            logger.info(f"TCN Volatility Regime loaded from {weights_path} (cross-version: rebuilt architecture)")
            return True
        except Exception as e:
            logger.warning(f"Could not rebuild and load weights: {e}")
            return False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        warm_start_path: Optional[str] = None,
        instrument: str = "UNKNOWN",
    ) -> Dict[str, float]:
        """Train TCN for volatility regime classification.

        Args:
            warm_start_path: Path to existing model to load weights from (warm-start training)
            instrument: Trading instrument (e.g., "EUR_USD") for logging
        """
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler

        logger.info("Training TCN (Volatility Regime Filter)...")
        logger.info(f"  Classes: {self.REGIME_NAMES}")

        # Initialize warm-start tracking
        self._is_warm_start = False
        self._warm_start_val_acc = 0.0

        # Validate and log labels
        self._validate_labels(y_train)
        self._log_class_distribution(y_train)

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        )
        x_val_scaled = self.scaler.transform(x_val.reshape(-1, x_val.shape[-1]))

        # Get seq_len from config (validated)
        seq_len = get_config_seq_len(self.config)

        # Create sequences using shared helper
        x_train_seq, y_train_seq = create_sequences(x_train_scaled, y_train, seq_len)
        x_val_seq, y_val_seq = create_sequences(x_val_scaled, y_val, seq_len)

        # Ensure labels are integers for sparse categorical crossentropy
        y_train_seq = y_train_seq.astype(np.int32)
        y_val_seq = y_val_seq.astype(np.int32)

        # Build model
        self.model = self._build_model((seq_len, X_train.shape[-1]))

        # === WARM-START: Load existing weights if available ===
        effective_lr = self.config.learning_rate
        if warm_start_path and Path(warm_start_path).exists():
            try:
                logger.info(f"🔥 WARM-START: Loading weights from {warm_start_path}")
                weights_loaded, prev_val_acc = self._load_warm_start_weights(warm_start_path, keras)

                if weights_loaded:
                    self._is_warm_start = True
                    self._warm_start_val_acc = prev_val_acc

                    # Reduce learning rate for warm-start (use 10% of base LR)
                    warm_start_lr_factor = getattr(self.config, 'warm_start_lr_factor', 0.1)
                    effective_lr = self.config.learning_rate * warm_start_lr_factor
                    logger.info(
                        f"🔥 Warm-start LR reduction: {self.config.learning_rate} → "
                        f"{effective_lr} (factor={warm_start_lr_factor})"
                    )
                else:
                    logger.warning(f"⚠️ Could not load warm-start weights from {warm_start_path}")
            except Exception as e:
                logger.warning(f"⚠️ Warm-start failed: {e}. Training from scratch.")

        # === WARMUP LR SCHEDULE ===
        # Calculate steps for warmup schedule
        steps_per_epoch = len(x_train_seq) // self.config.batch_size
        warmup_epochs = getattr(self.config, "warmup_epochs", 5)  # 5 epochs warmup
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = self.config.epochs * steps_per_epoch

        # Use WarmupCosineDecaySchedule for stable training
        try:
            from src.training.m1_metal_optimizer import WarmupCosineDecaySchedule

            lr_schedule = WarmupCosineDecaySchedule(
                initial_learning_rate=effective_lr * 0.1,  # Start at 10% of target
                warmup_steps=warmup_steps,
                decay_steps=total_steps - warmup_steps,
                min_learning_rate=1e-6,
                warmup_target=effective_lr,  # Use effective_lr (reduced for warm-start)
            )
            # Recompile model with warmup schedule
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            lr_display = f"{effective_lr:.2e}" if self._is_warm_start else f"{self.config.learning_rate}"
            logger.info(
                f"✓ TCN using warmup LR schedule: {warmup_epochs} warmup epochs, target LR={lr_display}"
            )
        except ImportError as e:
            logger.warning(
                f"WarmupCosineDecaySchedule not available: {e}. Using constant LR."
            )

        # Compute class weights for imbalanced regimes
        class_weight = self._compute_class_weights(y_train_seq)

        # Callbacks
        callbacks = self._create_tcn_callbacks(keras)

        # Train with class weights
        history = self.model.fit(
            x_train_seq,
            y_train_seq,
            validation_data=(x_val_seq, y_val_seq),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )

        self.is_trained = True
        self.seq_len = seq_len

        # Calculate metrics
        val_pred_probs = self.model.predict(x_val_seq, verbose=0)
        val_pred = np.argmax(val_pred_probs, axis=1)
        val_acc = np.mean(val_pred == y_val_seq)

        # Per-class accuracy
        for cls in range(self.n_classes):
            mask = y_val_seq == cls
            if np.sum(mask) > 0:
                cls_acc = np.mean(val_pred[mask] == cls)
                logger.info(f"  {self.REGIME_NAMES[cls]} accuracy: {cls_acc:.1%}")

        self.metrics = {
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(val_acc),
            "epochs_trained": len(history.history["loss"]),
        }

        logger.info(f"TCN Volatility Regime trained: val_accuracy={val_acc:.1%}")
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict volatility regime (0-3) and probabilities."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        x_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))

        # Create sequence from last seq_len rows
        if len(x_scaled) >= self.seq_len:
            x_seq = x_scaled[-self.seq_len :].reshape(1, self.seq_len, -1)
        else:
            # Pad with zeros if not enough data
            pad_len = self.seq_len - len(x_scaled)
            x_padded = np.vstack([np.zeros((pad_len, x_scaled.shape[1])), x_scaled])
            x_seq = x_padded.reshape(1, self.seq_len, -1)

        probs = self.model.predict(x_seq, verbose=0)[0]  # Shape: (4,)
        regime = int(np.argmax(probs))
        regime_name = self.REGIME_NAMES[regime]

        return {
            "volatility_regime": regime,
            "volatility_regime_name": regime_name,
            "regime_probabilities": probs.tolist(),
            "regime_confidence": float(probs[regime]),
        }

    def save(self, path: str) -> None:
        """Save TCN model with cross-Keras-version compatibility.

        Saves:
        - .keras file (native format for current Keras version)
        - .weights.h5 file (portable weights for cross-version loading)
        - .arch.json file (model architecture JSON for rebuild)
        - .meta.pkl file (scaler, feature names, config)

        This allows loading on both Keras 2.x and 3.x environments.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model in native format
        self.model.save(str(path))

        # === CROSS-VERSION COMPATIBILITY: Save weights and architecture separately ===
        # This allows loading on different Keras versions

        # 1. Save weights in H5 format (portable across Keras 2.x/3.x)
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

        # Save scaler and config
        meta = {
            "scaler": self.scaler,
            "seq_len": self.seq_len,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "model_type": "volatility_regime",  # Identify model type
            # Store architecture config for programmatic rebuild
            "architecture": {
                "input_shape": (self.seq_len, self.n_features),
                "tcn_hidden_size": getattr(self.config, "tcn_hidden_size", 32),
                "tcn_kernel_size": getattr(self.config, "tcn_kernel_size", 3),
                "tcn_num_layers": getattr(self.config, "tcn_num_layers", 2),
                "tcn_dropout": getattr(self.config, "tcn_dropout", 0.4),
                "tcn_l2_reg": getattr(self.config, "tcn_l2_reg", 0.005),
                "tcn_spatial_dropout": getattr(self.config, "tcn_spatial_dropout", 0.2),
                "tcn_noise_std": getattr(self.config, "tcn_noise_std", 0.03),
            },
        }
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        logger.info(
            f"TCN Volatility Regime saved to {path} (with cross-version compatibility)"
        )

    def load(self, path: str) -> None:
        """Load TCN model with cross-Keras-version compatibility.

        Loading strategy (in order):
        1. Try native .keras format (same Keras version)
        2. Try rebuilding from .arch.json + .weights.h5 (cross-version)
        3. Try rebuilding from meta['architecture'] + .weights.h5 (cross-version, no JSON)

        This allows models trained on Keras 3.x to load on Keras 2.x and vice versa.
        """
        from tensorflow import keras

        path = Path(path)
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        weights_path = path.with_suffix(WEIGHTS_H5_SUFFIX)
        arch_path = path.with_suffix(ARCH_JSON_SUFFIX)

        # Load metadata first (always needed)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.scaler = meta["scaler"]
        self.seq_len = meta["seq_len"]
        self.metrics = meta["metrics"]
        self.feature_names = meta.get("feature_names")
        self.n_features = meta.get("n_features")
        self.n_classes = meta.get("n_classes", 4)

        # Try loading strategies in order
        model_loaded = self._load_model_native(path, keras)

        if not model_loaded:
            model_loaded = self._load_model_from_arch_json(arch_path, weights_path, meta, keras)

        if not model_loaded:
            model_loaded = self._load_model_rebuild_architecture(weights_path, meta)

        if not model_loaded:
            raise RuntimeError(
                f"Failed to load TCN model from {path}. "
                f"Tried: native .keras, arch.json + weights.h5, rebuild + weights.h5. "
                f"Ensure model was saved with cross-version compatibility support."
            )

        self.is_trained = True
