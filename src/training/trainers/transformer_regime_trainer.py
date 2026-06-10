"""
Transformer Regime Trainer Module.

Transformer model for market regime classification (3 classes):
- Trend: Strong directional movement
- Chop: Sideways/ranging market
- Mean Revert: Price oscillating around mean

The Transformer acts as a "bouncer" - it determines WHAT KIND of market we're in,
filtering out unfavorable conditions for the directional strategy.

This class (~360 lines) was extracted from modular_trainers.py
for better code organization.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig
from src.training.trainers.callbacks import QuietProgressCallback, RichEpochCallback
from src.training.trainers.utils import (
    META_PKL_SUFFIX,
    MODEL_NOT_TRAINED_ERROR,
    atomic_keras_save,
    atomic_pickle_dump,
    create_sequences,
    get_config_seq_len,
    predict_with_named_input_if_needed,
)

logger = logging.getLogger(__name__)


class TransformerRegimeTrainer(BaseTrainer):
    """
    Transformer model for market regime classification (3 classes).

    The Transformer acts as a "bouncer" - it determines WHAT KIND of market we're in,
    not which direction to trade. This is a much more tractable problem than direction.

    Regimes:
    - 0 = TREND: Strong directional movement, let gates decide direction
    - 1 = CHOP: Sideways noise, skip trading entirely
    - 2 = MEAN_REVERT: Overextended, fade 2-bar momentum

    Input: Regime indicators (ADX, RSI, volatility, z-scores, etc.)
    Output: Softmax over 3 classes
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.seq_len = None
        self.class_names = ["trend", "chop", "mean_revert"]

        # Transformer hyperparameters - match proven direction config as baseline
        self.d_model = getattr(config, "transformer_d_model", 16) if config else 16
        self.num_heads = getattr(config, "transformer_num_heads", 2) if config else 2
        self.ff_dim = getattr(config, "transformer_ff_dim", 32) if config else 32
        self.num_blocks = getattr(config, "transformer_num_blocks", 1) if config else 1
        self.transformer_dropout = (
            getattr(config, "transformer_dropout", 0.2) if config else 0.2
        )  # Reduced from 0.4

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build Transformer model for 3-class regime classification."""
        from tensorflow import keras

        seq_len, n_features = input_shape

        inp = keras.Input(shape=(seq_len, n_features), name="features")

        # Project input to d_model dimensions
        x = keras.layers.Dense(self.d_model, name="input_projection")(inp)

        # Add positional encoding
        x = self._add_positional_encoding(x, seq_len, self.d_model)

        # Transformer encoder blocks
        for i in range(self.num_blocks):
            x = self._transformer_encoder_layer(
                x,
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.ff_dim,
                dropout=self.transformer_dropout,
                name_prefix=f"transformer_{i}",
            )

        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dropout(self.transformer_dropout)(x)

        # 3-class regime output (softmax)
        regime = keras.layers.Dense(
            3, activation="softmax", name="regime", dtype="float32"
        )(x)

        model = keras.Model(inputs=inp, outputs=regime, name="transformer_regime")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def _add_positional_encoding(self, x, seq_len: int, d_model: int):
        """Add sinusoidal positional encoding."""
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]

        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)

        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        pos_encoding = pos_encoding[np.newaxis, :, :].astype(np.float32)

        # Cast to match input dtype for mixed precision
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
        name_prefix: str,
    ):
        """Single transformer encoder layer."""
        from tensorflow import keras

        # Multi-head self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name_prefix}_mha"
        )(x, x)
        attn_output = keras.layers.Dropout(dropout)(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(
            x + attn_output
        )

        # Feedforward network
        ffn = keras.layers.Dense(dff, activation="relu", name=f"{name_prefix}_ffn1")(x)
        ffn = keras.layers.Dense(d_model, name=f"{name_prefix}_ffn2")(ffn)
        ffn = keras.layers.Dropout(dropout)(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(
            x + ffn
        )

        return x

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Train Transformer for 3-class regime classification.
        Reports F1 score (macro) as primary metric.
        """
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import f1_score, classification_report
        from sklearn.utils.class_weight import compute_class_weight

        logger.info("Training Transformer (Regime Classification)...")

        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        if class_names:
            self.class_names = class_names

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        )
        x_val_scaled = self.scaler.transform(x_val.reshape(-1, x_val.shape[-1]))

        # Get seq_len from config (validated)
        seq_len = get_config_seq_len(self.config)
        self.seq_len = seq_len

        # Create sequences using shared helper
        x_train_seq, y_train_seq = create_sequences(x_train_scaled, y_train, seq_len)
        x_val_seq, y_val_seq = create_sequences(x_val_scaled, y_val, seq_len)

        # Log class distribution
        unique, counts = np.unique(y_train_seq, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Training class distribution: {class_dist}")

        unique_val, counts_val = np.unique(y_val_seq, return_counts=True)
        class_dist_val = dict(zip(unique_val, counts_val))
        logger.info(f"Validation class distribution: {class_dist_val}")

        logger.info(f"Sequence shape: train={x_train_seq.shape}, val={x_val_seq.shape}")

        # Compute class weights for imbalanced classes
        classes = np.unique(y_train_seq)
        if len(classes) > 1:
            weights = compute_class_weight("balanced", classes=classes, y=y_train_seq)
            class_weight = {int(c): w for c, w in zip(classes, weights)}
            logger.info(f"Class weights: {class_weight}")
        else:
            class_weight = None

        # Build model
        self.model = self._build_model((seq_len, self.n_features))
        # Compact model info for console
        try:
            from rich.console import Console as _RichConsole
            _rc = _RichConsole()
            _rc.print(
                f"  [dim]Model:[/dim] {self.model.name} | "
                f"[green]{self.model.count_params():,}[/green] params | "
                f"input=({seq_len}, {self.n_features})"
            )
        except Exception:
            pass
        # Full summary to log file (console handler at WARNING skips INFO)
        self.model.summary(print_fn=logger.info)

        # Callbacks - use config patience values
        callbacks = [
            # Rich-formatted epoch display with color coding (or quiet progress bar)
            RichEpochCallback(
                model_name="Transformer Regime",
                total_epochs=self.config.epochs,
                quiet=self.config.quiet,
            )
            if not self.config.quiet
            else QuietProgressCallback(
                model_name="Transformer Regime",
                total_epochs=self.config.epochs,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                mode="min",
                restore_best_weights=True,
                verbose=0,  # Suppress - Rich callback handles display
                start_from_epoch=self.config.min_epochs,  # Enforce minimum epochs before early stopping
            ),
            # NOTE: ReduceLROnPlateau removed - incompatible with LearningRateSchedule
        ]

        # Train
        history = self.model.fit(
            x_train_seq,
            y_train_seq,
            validation_data=(x_val_seq, y_val_seq),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0,  # Suppress Keras output - RichEpochCallback handles display
            class_weight=class_weight,
        )

        self.is_trained = True

        # Calculate metrics
        val_pred_probs = self.model.predict(x_val_seq, verbose=0)
        val_pred = np.argmax(val_pred_probs, axis=1)

        # F1 score (macro - treats all classes equally)
        f1_macro = f1_score(y_val_seq, val_pred, average="macro")
        f1_weighted = f1_score(y_val_seq, val_pred, average="weighted")

        # Per-class F1
        f1_per_class = f1_score(y_val_seq, val_pred, average=None)

        # Accuracy
        val_acc = np.mean(val_pred == y_val_seq)

        # Classification report
        report = classification_report(
            y_val_seq, val_pred, target_names=self.class_names
        )
        logger.info(f"\nClassification Report:\n{report}")

        self.metrics = {
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(val_acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "f1_trend": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            "f1_chop": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            "f1_mean_revert": float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
            "epochs_trained": len(history.history["loss"]),
            "n_train_samples": len(x_train_seq),
            "n_val_samples": len(x_val_seq),
        }

        logger.info(
            f"Regime Transformer trained: val_acc={val_acc:.4f}, F1_macro={f1_macro:.4f}"
        )
        logger.info(
            f"  F1 per class: trend={self.metrics['f1_trend']:.3f}, "
            f"chop={self.metrics['f1_chop']:.3f}, mean_revert={self.metrics['f1_mean_revert']:.3f}"
        )

        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict regime (0=trend, 1=chop, 2=mean_revert) with probabilities.
        """
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
            x_padded = np.vstack([np.zeros((pad_len, x_scaled.shape[-1])), x_scaled])
            x_seq = x_padded.reshape(1, self.seq_len, -1)

        # Predict
        probs = predict_with_named_input_if_needed(self.model, x_seq, verbose=0)[0]
        regime = int(np.argmax(probs))

        return {
            "regime": regime,
            "regime_name": self.class_names[regime],
            "prob_trend": float(probs[0]),
            "prob_chop": float(probs[1]),
            "prob_mean_revert": float(probs[2]),
            "confidence": float(np.max(probs)),
        }

    def save(self, path: str) -> None:
        """Save Transformer regime model and scaler (atomic writes)."""
        from src.core.modular_data_loaders import FEATURE_PIPELINE_VERSION

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        atomic_keras_save(self.model, path)

        meta = {
            "scaler": self.scaler,
            "seq_len": self.seq_len,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "class_names": self.class_names,
            "model_type": "transformer_regime",
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        }
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        atomic_pickle_dump(meta, meta_path)

        logger.info(f"Transformer Regime saved to {path}")

    def load(self, path: str) -> None:
        """Load Transformer regime model and scaler."""
        import tensorflow as tf
        from tensorflow import keras

        path = Path(path)
        model = None
        load_errors = []

        try:
            from src.utils.keras_model_loader import load_keras_model

            model, meta = load_keras_model(str(path), compile=False)
            if not meta.get("success"):
                model = None
        except Exception as e:
            load_errors.append(f"cross_version: {e}")

        if model is None:
            try:
                model = keras.models.load_model(str(path), compile=False)
            except Exception as e:
                load_errors.append(f"keras_compile_false: {e}")

        if model is None:
            try:
                model = tf.keras.models.load_model(str(path), compile=False)
            except Exception as e:
                load_errors.append(f"tf_keras_compile_false: {e}")

        if model is None:
            try:
                model = keras.models.load_model(str(path), compile=False, safe_mode=False)
            except Exception as e:
                load_errors.append(f"safe_mode_false: {e}")

        if model is None:
            raise RuntimeError(
                f"Failed to load Transformer regime model from {path}. Errors: {'; '.join(load_errors)}"
            )
        self.model = model

        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.scaler = meta["scaler"]
        self.seq_len = meta["seq_len"]
        self.metrics = meta["metrics"]
        self.feature_names = meta.get("feature_names")
        self.n_features = meta.get("n_features")
        self.class_names = meta.get("class_names", ["trend", "chop", "mean_revert"])
        self.is_trained = True
        logger.info(f"Transformer Regime loaded from {path}")

# EOF
