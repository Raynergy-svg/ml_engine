"""
TCN Volatility Regime Trainer for Forward-Looking Prediction.

This module implements a dual-head TCN trainer for predicting FUTURE volatility
regimes. Unlike the standard TCN trainer which predicts current volatility, this
trainer looks ahead (default: 48 bars) to predict future market conditions.

Key Features:
    - Forward-looking 4-class prediction (QUIET, STABLE, ACTIVE, EXTREME)
    - Dual-head architecture: classification + regression
    - Research-backed TCN with residual connections and weight normalization
    - Anti-collapse mechanisms (focal loss, class weights, sample weights)
    - Receptive field validation

Architecture:
    Based on Bai et al. / Unit8 research:
    - Dilated causal convolutions with exponential dilation
    - Residual connections for stable deep training
    - Weight normalization to prevent gradient explosion
    - Full receptive field coverage (≥ seq_len)

Usage:
    trainer = TCNVolatilityRegimeTrainer(config)
    metrics = trainer.train(
        X_train, y_train, X_val, y_val,
        y_train_reg=vol_pct_train,
        y_val_reg=vol_pct_val
    )
    result = trainer.predict(X_new)

Classes:
    TCNVolatilityRegimeTrainer: Dual-head forward volatility predictor
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig
from src.training.trainers.callbacks import (
    RichEpochCallback,
    AutoAdjustCallback,
)
from src.training.trainers.display import TrainingDisplay
from src.training.trainers.utils import (
    atomic_keras_save,
    atomic_pickle_dump,
    predict_with_named_input_if_needed,
    train_accuracy_at_best_epoch,
)

# Import custom LR schedules early so they're registered with Keras before
# any pickle deserialization that might contain Keras models with these schedules
try:
    from src.training.m1_metal_optimizer import (
        WarmupCosineDecaySchedule,  # noqa: F401
        CosineDecayRestarts,  # noqa: F401
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TCNVolatilityRegimeTrainer(BaseTrainer):
    """
    TCN model for FORWARD-LOOKING 4-class volatility regime prediction.

    CRITICAL CHANGE (2025): Now predicts FUTURE volatility regime, not current.
    Uses dual-head architecture: classification + regression for robustness.

    Research-backed architecture (Bai et al. / Unit8):
    - Dilated causal convolutions with exponential dilation
    - Residual connections for stable deep training
    - Weight normalization to prevent gradient explosion
    - Full receptive field coverage (≥ seq_len)

    Forward Volatility Regimes (predicted 48 bars ahead):
        - 0 = QUIET_NEXT: Future ATR < 25th percentile (skip trading)
        - 1 = STABLE_NEXT: Future ATR 25th-60th percentile (normal)
        - 2 = ACTIVE_NEXT: Future ATR 60th-85th percentile (opportunity!)
        - 3 = EXTREME_NEXT: Future ATR > 85th percentile (caution)

    Dual-Head Output:
        - Classification: 4-class softmax for regime
        - Regression: % change in volatility (fallback/tiebreaker)

    Anti-Collapse Mechanisms:
        - PredictionCollapseCallback: Detects >80% single-class predictions
        - CategoricalFocalCrossentropy: Boosts minority classes
        - Class weights: Inverse frequency weighting
        - Sample weights: Higher weight for large volatility changes

    Success Criteria:
        - Classification accuracy >60% on validation (harder task than current regime)
        - All 4 classes represented in predictions (no collapse)
        - F1-score >0.50 for ACTIVE_NEXT and EXTREME_NEXT classes
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.seq_len = None
        self.n_classes = 4
        self.class_names = ['QUIET_NEXT', 'STABLE_NEXT', 'ACTIVE_NEXT', 'EXTREME_NEXT']

        # Forward-looking parameters
        self.lookahead = getattr(self.config, 'tcn_lookahead', 48)  # 48 bars = 2 days for H1

        # TCN architecture hyperparameters
        self.kernel_size = getattr(self.config, 'tcn_kernel_size', 5)
        self.dilation_base = getattr(self.config, 'tcn_dilation_base', 2)
        self.num_filters = getattr(self.config, 'tcn_num_filters', 64)
        self.num_residual_blocks = getattr(self.config, 'tcn_num_residual_blocks', 5)
        self.dropout = getattr(self.config, 'tcn_dropout', 0.2)
        self.use_weight_norm = getattr(self.config, 'tcn_weight_norm', True)

        # Loss weights for dual-head
        self.classification_weight = 0.7
        self.regression_weight = 0.3

        # Focal loss parameters - will be overridden by class_weights if provided
        self.focal_gamma = 2.0
        self.focal_alpha = None  # Will use class_weights or config

        # Regression thresholds for fallback mapping
        self.reg_thresholds = {
            'quiet': -0.15,
            'stable_high': 0.15,
            'active_high': 0.40,
        }

        # Dual-head model flag
        self.use_dual_head = True

    def _compute_receptive_field(self) -> int:
        """Compute receptive field size for current architecture."""
        k = self.kernel_size
        b = self.dilation_base
        n = self.num_residual_blocks
        receptive_field = 1 + 2 * (k - 1) * (b ** n - 1) // (b - 1)
        return receptive_field

    def _build_model(self, input_shape: tuple[int, int]) -> Any:
        """
        Build dual-head TCN model for forward volatility prediction.

        Uses TCNVolatilityDualHead from tensorflow_models.py.
        """
        # Import the dual-head model
        try:
            from src.models.tensorflow_models import TCNVolatilityDualHead
        except ImportError:
            from models.tensorflow_models import TCNVolatilityDualHead

        seq_len, n_features = input_shape

        # Verify receptive field coverage
        receptive_field = self._compute_receptive_field()
        if receptive_field < seq_len:
            logger.warning(f"Receptive field ({receptive_field}) < seq_len ({seq_len}). "
                           f"Consider increasing num_residual_blocks.")

        # Build dual-head model
        model = TCNVolatilityDualHead(
            n_features=n_features,
            seq_len=seq_len,
            n_classes=self.n_classes,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            num_residual_blocks=self.num_residual_blocks,
            dilation_base=self.dilation_base,
            dropout=self.dropout,
        )

        # Build model by calling it once
        dummy_input = tf.zeros((1, seq_len, n_features))
        _ = model(dummy_input)

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        class_weights: Optional[Dict[int, float]] = None,
        sample_weights: Optional[np.ndarray] = None,
        seq_len: int = 60,
        y_train_reg: Optional[np.ndarray] = None,
        y_val_reg: Optional[np.ndarray] = None,
        w_train: Optional[np.ndarray] = None,
        w_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train dual-head TCN for forward 4-class volatility regime prediction.

        Args:
            X_train: Training sequences (batch, seq_len, features)
            y_train: Training labels (batch,) with values 0-3
            X_val: Validation sequences
            y_val: Validation labels
            feature_names: Feature names for interpretability
            class_weights: Class weights for imbalanced data
            sample_weights: Per-sample weights (deprecated, use w_train)
            seq_len: Sequence length
            y_train_reg: Regression targets for training (% vol change)
            y_val_reg: Regression targets for validation
            w_train: Sample weights for training
            w_val: Sample weights for validation

        Returns:
            Dict with training metrics
        """
        from tensorflow import keras
        from sklearn.metrics import f1_score

        # Initialize clean display
        display = TrainingDisplay("TCN Forward Volatility")

        # Save metadata
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        self.seq_len = seq_len if X_train.ndim == 3 else 60
        self.scaler = None  # Assume pre-scaled from data loader

        # Ensure correct shape
        if X_train.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, features), got {X_train.shape}")

        # Use provided sample weights or fall back to deprecated parameter
        if w_train is None:
            w_train = sample_weights if sample_weights is not None else np.ones(len(y_train))
        if w_val is None:
            w_val = np.ones(len(y_val))

        # Create regression targets if not provided (default to zeros)
        if y_train_reg is None:
            y_train_reg = np.zeros(len(y_train), dtype=np.float32)
        if y_val_reg is None:
            y_val_reg = np.zeros(len(y_val), dtype=np.float32)

        # Convert to one-hot for classification
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=self.n_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=self.n_classes)
        logger.debug(f"Labels shape: {y_train_onehot.shape}")

        # Build model
        self.model = self._build_model((self.seq_len, self.n_features))

        # === COMPILE WITH CUSTOM TRAINING STEP ===
        # We need custom training because of dual outputs

        # Learning rate
        tcn_lr = min(self.config.learning_rate, 0.001)  # Conservative for forward prediction

        # Determine focal alpha from class_weights (sklearn balanced) or use default
        # Class weights from sklearn are inverse frequency, so higher = rarer class
        if class_weights is not None:
            # Normalize class weights to sum to 1 for focal alpha
            cw_values = [class_weights.get(i, 1.0) for i in range(self.n_classes)]
            cw_sum = sum(cw_values)
            effective_alpha = [v / cw_sum for v in cw_values]
            logger.debug(f"Focal alpha from class weights: {[f'{a:.3f}' for a in effective_alpha]}")
        else:
            # Default focal alpha: boost minority classes
            effective_alpha = [0.30, 0.20, 0.25, 0.25]  # QUIET, STABLE, ACTIVE, EXTREME
            logger.debug(f"Using default focal alpha: {effective_alpha}")

        self.focal_alpha = effective_alpha

        # Create separate losses
        classification_loss_fn = keras.losses.CategoricalFocalCrossentropy(
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            from_logits=False,
        )
        regression_loss_fn = keras.losses.MeanSquaredError()

        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=tcn_lr)

        # Assign optimizer to model so callbacks can access it
        self.model.optimizer = optimizer

        # Show clean configuration summary
        train_valid_mask = w_train > 0
        valid_labels = y_train[train_valid_mask]
        class_counts = np.bincount(valid_labels, minlength=self.n_classes) if len(valid_labels) > 0 else [0]*4

        display.show_config({
            "Lookahead": f"{self.lookahead} bars",
            "Params": f"{self.model.count_params():,}",
            "LR": f"{tcn_lr:.1e}",
            "Loss": f"FocalCE(γ={self.focal_gamma})",
            "Classes": f"QUI:{class_counts[0]/len(valid_labels):.0%} STA:{class_counts[1]/len(valid_labels):.0%} ACT:{class_counts[2]/len(valid_labels):.0%} EXT:{class_counts[3]/len(valid_labels):.0%}",
        })

        # === PREDICTION COLLAPSE CALLBACK (4-class version) ===
        class RegimeCollapseCallback(keras.callbacks.Callback):
            def __init__(self, X_val, y_val, class_names, check_every=5):
                super().__init__()
                self.X_val = X_val
                self.y_val = y_val
                self.class_names = class_names
                self.check_every = check_every
                self.collapse_warned = False
                self.display = display

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.check_every != 0:
                    return

                outputs = self.model.predict(self.X_val, verbose=0)
                if isinstance(outputs, dict):
                    preds = outputs['classification']
                else:
                    preds = outputs

                pred_classes = np.argmax(preds, axis=1)
                pred_dist = np.bincount(pred_classes, minlength=4) / len(pred_classes)

                # Check for collapse (>80% same prediction)
                max_pct = max(pred_dist)
                if max_pct > 0.80:
                    dominant = np.argmax(pred_dist)
                    if not self.collapse_warned:
                        self.display.warn(f"Collapse detected: {max_pct:.0%} -> {self.class_names[dominant]}")
                        self.collapse_warned = True
                else:
                    self.collapse_warned = False

        # Callbacks
        callbacks = [
            RichEpochCallback(
                model_name="TCN Forward Volatility",
                total_epochs=self.config.epochs,
            ),
            AutoAdjustCallback(
                patience=8,           # Stuck for 8 epochs → adjust
                lr_factor=0.5,        # Halve LR when stuck
                min_lr=1e-6,
                max_adjustments=4,    # Up to 4 LR reductions
                min_delta=0.005,      # 0.5% improvement threshold
                verbose=True
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                mode='min',
                restore_best_weights=True,
                verbose=0,
                start_from_epoch=max(10, self.config.min_epochs),
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(5, self.config.patience // 3),
                min_lr=1e-6,
                verbose=0,
            ),
            RegimeCollapseCallback(X_val, y_val, self.class_names, check_every=5),
        ]

        # === CUSTOM TRAINING LOOP for dual-head ===
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            X_train,
            {'classification': y_train_onehot, 'regression': y_train_reg.reshape(-1, 1)},
            w_train
        )).shuffle(1024).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            X_val,
            {'classification': y_val_onehot, 'regression': y_val_reg.reshape(-1, 1)},
            w_val
        )).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        # Training metrics
        train_loss_metric = keras.metrics.Mean(name='train_loss')
        train_acc_metric = keras.metrics.CategoricalAccuracy(name='train_accuracy')
        val_loss_metric = keras.metrics.Mean(name='val_loss')
        val_acc_metric = keras.metrics.CategoricalAccuracy(name='val_accuracy')

        # Class weights are now integrated into focal_alpha above

        @tf.function
        def train_step(x, y, sample_weight):
            with tf.GradientTape() as tape:
                outputs = self.model(x, training=True)

                # Classification loss
                class_loss = classification_loss_fn(
                    y['classification'],
                    outputs['classification'],
                    sample_weight=sample_weight
                )

                # Regression loss
                reg_loss = regression_loss_fn(y['regression'], outputs['regression'])

                # Combined loss
                total_loss = (self.classification_weight * class_loss +
                              self.regression_weight * reg_loss)

                # Add regularization losses
                if self.model.losses:
                    total_loss += tf.add_n(self.model.losses)

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            train_loss_metric.update_state(total_loss)
            train_acc_metric.update_state(y['classification'], outputs['classification'])

            return total_loss

        @tf.function
        def val_step(x, y, sample_weight):
            outputs = self.model(x, training=False)

            class_loss = classification_loss_fn(
                y['classification'],
                outputs['classification'],
                sample_weight=sample_weight
            )
            reg_loss = regression_loss_fn(y['regression'], outputs['regression'])
            total_loss = (self.classification_weight * class_loss +
                          self.regression_weight * reg_loss)

            val_loss_metric.update_state(total_loss)
            val_acc_metric.update_state(y['classification'], outputs['classification'])

            return total_loss

        # Training loop
        best_val_loss = float('inf')
        best_weights = None
        best_epoch_idx = -1  # epoch whose weights get restored below
        patience_counter = 0
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(self.config.epochs):
            # Reset metrics
            train_loss_metric.reset_state()
            train_acc_metric.reset_state()
            val_loss_metric.reset_state()
            val_acc_metric.reset_state()

            # Training
            for x_batch, y_batch, w_batch in train_dataset:
                train_step(x_batch, y_batch, w_batch)

            # Validation
            for x_batch, y_batch, w_batch in val_dataset:
                val_step(x_batch, y_batch, w_batch)

            # Get metrics
            train_loss = train_loss_metric.result().numpy()
            train_acc = train_acc_metric.result().numpy()
            val_loss = val_loss_metric.result().numpy()
            val_acc = val_acc_metric.result().numpy()

            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            # Call callbacks
            logs = {'loss': train_loss, 'accuracy': train_acc,
                    'val_loss': val_loss, 'val_accuracy': val_acc}
            for callback in callbacks:
                # Set model via set_model() for Keras callbacks, or _model for custom
                if hasattr(callback, 'set_model'):
                    callback.set_model(self.model)
                elif hasattr(callback, '_model'):
                    callback._model = self.model
                callback.on_epoch_end(epoch, logs)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                best_epoch_idx = epoch  # 2026-06-10: track the saved epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience and epoch >= self.config.min_epochs:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        if best_weights is not None:
            self.model.set_weights(best_weights)

        self.is_trained = True

        # Evaluate on validation set
        val_outputs = self.model.predict(X_val, verbose=0)
        if isinstance(val_outputs, dict):
            val_pred_probs = val_outputs['classification']
        else:
            val_pred_probs = val_outputs

        val_pred_classes = np.argmax(val_pred_probs, axis=1)
        val_acc = np.mean(val_pred_classes == y_val)

        # Calculate distributions (logged via display)
        pred_dist = np.bincount(val_pred_classes, minlength=self.n_classes) / len(val_pred_classes)

        # Calculate F1 scores per class
        f1_scores = f1_score(y_val, val_pred_classes, average=None, zero_division=0)
        f1_macro = f1_score(y_val, val_pred_classes, average='macro', zero_division=0)

        # ACTIVE/EXTREME detection accuracy (classes 2 and 3 - the actionable ones)
        active_extreme_mask = (y_val >= 2)
        if active_extreme_mask.sum() > 0:
            active_extreme_acc = np.mean(val_pred_classes[active_extreme_mask] >= 2)
        else:
            active_extreme_acc = 0.0

        # Check for collapse
        all_classes_present = all(pred_dist[i] > 0.05 for i in range(4))

        # 2026-06-10 fix (training-infra audit): best weights are restored from
        # the best-val-loss epoch above — report train_accuracy at THAT epoch,
        # not the last (potentially overfit) one. history[-1] produced a
        # phantom train/val gap on the saved checkpoint.
        train_accuracy_at_best = train_accuracy_at_best_epoch(
            history['accuracy'], best_epoch_idx=best_epoch_idx
        )

        self.metrics = {
            'train_accuracy': train_accuracy_at_best,
            'val_accuracy': float(val_acc),
            'val_f1_macro': float(f1_macro),
            'val_f1_quiet': float(f1_scores[0]) if len(f1_scores) > 0 else 0.0,
            'val_f1_stable': float(f1_scores[1]) if len(f1_scores) > 1 else 0.0,
            'val_f1_active': float(f1_scores[2]) if len(f1_scores) > 2 else 0.0,
            'val_f1_extreme': float(f1_scores[3]) if len(f1_scores) > 3 else 0.0,
            'active_extreme_detection': float(active_extreme_acc),
            'all_classes_present': all_classes_present,
            'epochs_trained': len(history['loss']),
            'receptive_field': self._compute_receptive_field(),
            'lookahead': self.lookahead,
        }

        # Show clean results summary
        display.show_summary({
            'Val Accuracy': f"{val_acc:.1%}",
            'F1 Macro': f"{f1_macro:.3f}",
            'F1 (QUI/STA/ACT/EXT)': f"{f1_scores[0]:.2f} / {f1_scores[1]:.2f} / {f1_scores[2]:.2f} / {f1_scores[3]:.2f}",
            'Active/Extreme Det': f"{active_extreme_acc:.1%}",
            'Epochs': len(history['loss']),
            'Status': "✓ Healthy" if all_classes_present else "⚠ Collapse",
        }, title="Training Complete")

        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict forward volatility regime for input sequence.

        Returns:
            Dict with:
                - regime: int (0=QUIET_NEXT, 1=STABLE_NEXT, 2=ACTIVE_NEXT, 3=EXTREME_NEXT)
                - regime_name: str
                - probabilities: np.ndarray of 4 class probabilities
                - confidence: float (max probability)
                - vol_change_pct: float (regression prediction)
                - regression_regime: int (regime from regression fallback)
                - is_opportunity: bool (ACTIVE_NEXT or STABLE_NEXT - allow trading)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        X = np.asarray(X, dtype=np.float32)
        if self.scaler is not None:
            if X.ndim == 2:
                X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            elif X.ndim == 3:
                original_shape = X.shape
                X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(original_shape)

        # Handle input shape
        if X.ndim == 2:
            if len(X) >= self.seq_len:
                X_seq = X[-self.seq_len:].reshape(1, self.seq_len, -1)
            else:
                pad_len = self.seq_len - len(X)
                X_padded = np.vstack([np.zeros((pad_len, X.shape[1])), X])
                X_seq = X_padded.reshape(1, self.seq_len, -1)
        elif X.ndim == 3:
            X_seq = X
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")

        # Get predictions
        outputs = predict_with_named_input_if_needed(self.model, X_seq, verbose=0)

        if isinstance(outputs, dict):
            probs = outputs['classification'][0]
            vol_change = float(outputs['regression'][0, 0])
        elif isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            probs = outputs[0][0]
            vol_change = float(outputs[1][0, 0])
        else:
            probs = outputs[0]
            vol_change = 0.0

        regime = int(np.argmax(probs))
        confidence = float(probs[regime])

        # Regression fallback mapping
        if vol_change < self.reg_thresholds['quiet']:
            reg_regime = 0  # QUIET_NEXT
        elif vol_change < self.reg_thresholds['stable_high']:
            reg_regime = 1  # STABLE_NEXT
        elif vol_change < self.reg_thresholds['active_high']:
            reg_regime = 2  # ACTIVE_NEXT
        else:
            reg_regime = 3  # EXTREME_NEXT

        # Use regression fallback if classification confidence is low
        final_regime = regime
        if confidence < 0.60:
            logger.debug(f"Low classification confidence ({confidence:.1%}), using regression fallback")
            final_regime = reg_regime

        return {
            'regime': final_regime,
            'regime_name': self.class_names[final_regime],
            'probabilities': probs,
            'confidence': confidence,
            'vol_change_pct': vol_change,
            'regression_regime': reg_regime,
            'classification_regime': regime,
            'is_opportunity': final_regime in [1, 2],  # STABLE or ACTIVE - allow trading
            'is_high_volatility': final_regime >= 2,  # ACTIVE or EXTREME
        }

    def save(self, path: str) -> None:
        """Save TCN Forward Volatility model (atomic tmp+rename writes)."""
        from src.core.modular_data_loaders import FEATURE_PIPELINE_VERSION

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        atomic_keras_save(self.model, path)

        meta = {
            'scaler': self.scaler,
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'class_names': self.class_names,
            'metrics': self.metrics,
            'feature_names': self.feature_names,
            'lookahead': self.lookahead,
            'reg_thresholds': self.reg_thresholds,
            'config': {
                'kernel_size': self.kernel_size,
                'dilation_base': self.dilation_base,
                'num_filters': self.num_filters,
                'num_residual_blocks': self.num_residual_blocks,
                'dropout': self.dropout,
                'use_weight_norm': self.use_weight_norm,
                'receptive_field': self._compute_receptive_field(),
                'focal_gamma': self.focal_gamma,
                'focal_alpha': self.focal_alpha,
            },
            # Inference contract: pipeline version stamp so a future pipeline
            # change can't silently feed this model out-of-distribution input.
            'feature_pipeline_version': FEATURE_PIPELINE_VERSION,
        }

        meta_path = path.with_suffix('.meta.pkl')
        atomic_pickle_dump(meta, meta_path)

        logger.info(f"TCN Forward Volatility saved to {path}")

    def load(self, path: str) -> None:
        """Load TCN Forward Volatility model."""
        import tensorflow as tf
        from tensorflow import keras

        path = Path(path)

        # Try to load with custom objects
        try:
            from src.models.tensorflow_models import TCNVolatilityDualHead
        except ImportError:
            try:
                from models.tensorflow_models import TCNVolatilityDualHead
            except ImportError:
                TCNVolatilityDualHead = None

        custom_objects = {}
        if TCNVolatilityDualHead is not None:
            custom_objects['TCNVolatilityDualHead'] = TCNVolatilityDualHead

        # Cross-version loading strategies (Keras 2.x/3.x compatibility)
        model = None
        load_errors = []

        try:
            from src.utils.keras_model_loader import load_keras_model

            model, load_metadata = load_keras_model(
                str(path),
                custom_objects=custom_objects,
                compile=False,
            )
            if not load_metadata.get("success"):
                model = None
        except Exception as e:
            load_errors.append(f"cross_version: {e}")
            model = None

        if model is None:
            try:
                model = keras.models.load_model(
                    str(path),
                    custom_objects=custom_objects,
                    compile=False,
                )
            except Exception as e:
                load_errors.append(f"keras_compile_false: {e}")

        if model is None:
            try:
                model = tf.keras.models.load_model(
                    str(path),
                    custom_objects=custom_objects,
                    compile=False,
                )
            except Exception as e:
                load_errors.append(f"tf_keras_compile_false: {e}")

        if model is None:
            try:
                model = keras.models.load_model(
                    str(path),
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False,
                )
            except Exception as e:
                load_errors.append(f"safe_mode_false: {e}")

        if model is None:
            raise RuntimeError(
                f"Failed to load TCN volatility model from {path}. Errors: {'; '.join(load_errors)}"
            )

        self.model = model

        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.scaler = meta.get('scaler')
        self.seq_len = meta['seq_len']
        self.n_features = meta['n_features']
        self.n_classes = meta.get('n_classes', 4)
        self.class_names = meta.get('class_names', ['QUIET_NEXT', 'STABLE_NEXT', 'ACTIVE_NEXT', 'EXTREME_NEXT'])
        self.metrics = meta.get('metrics', {})
        self.feature_names = meta.get('feature_names')
        self.lookahead = meta.get('lookahead', 48)
        self.reg_thresholds = meta.get('reg_thresholds', {
            'quiet': -0.15, 'stable_high': 0.15, 'active_high': 0.40
        })

        # Restore architecture config
        arch_config = meta.get('config', {})
        self.kernel_size = arch_config.get('kernel_size', 5)
        self.dilation_base = arch_config.get('dilation_base', 2)
        self.num_filters = arch_config.get('num_filters', 64)
        self.num_residual_blocks = arch_config.get('num_residual_blocks', 5)
        self.dropout = arch_config.get('dropout', 0.2)
        self.use_weight_norm = arch_config.get('use_weight_norm', True)
        self.focal_gamma = arch_config.get('focal_gamma', 2.0)
        self.focal_alpha = arch_config.get('focal_alpha', [0.35, 0.25, 0.25, 0.15])

        self.is_trained = True

        logger.info(f"TCN Forward Volatility loaded from {path}")
