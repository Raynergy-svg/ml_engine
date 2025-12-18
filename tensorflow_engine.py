"""
TensorFlow Training Engine with TensorBoard Visualization.
Provides a complete training pipeline with real-time visual monitoring.

Features:
- Native Keras training with model.fit()
- TensorBoard integration for visual training monitoring
- Early stopping, model checkpointing, LR scheduling
- Mixed precision training support
- Custom callbacks for trading-specific metrics
"""

raise SystemExit(
    "Retired: standalone TF training engine disabled. Buddy training runs only via main.py. "
    "Use: python main.py train-buddy"
)

import os
import datetime
import time
import numpy as np
import platform
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks as keras_callbacks
from tensorflow.keras import optimizers, losses, metrics

from tensorflow_models import create_tensorflow_model


from tensorflow.keras.saving import register_keras_serializable

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CHECKPOINT_DIR = 'trained_data/checkpoints'
DEFAULT_TENSORBOARD_DIR = 'trained_data/tensorboard'


# =============================================================================
# Custom Loss Functions
# =============================================================================

@register_keras_serializable()
class BinaryFocalLoss(losses.Loss):
    """
    Binary Focal Loss for addressing class imbalance in binary classification.
    
    Reduces loss contribution from easy examples, focusing training
    on hard, misclassified examples. Perfect for direction prediction
    where up/down may be imbalanced.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha: Class weight for positive class (default 0.25). Set to None to disable.
        label_smoothing: Smoothing factor for labels (default 0.0).
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.0,
        name: str = 'binary_focal_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        # Apply label smoothing
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weight: (1 - pt)^gamma
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_weight
        
        focal_loss = focal_weight * bce
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
        })
        return config


def compute_class_weights(y_direction: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced up/down distribution.
    
    Args:
        y_direction: Binary direction labels (1=up, 0=down)
    
    Returns:
        dict with 'up_weight' and 'down_weight' for loss weighting
    """
    y_flat = np.array(y_direction).flatten()
    n_up = np.sum(y_flat == 1)
    n_down = np.sum(y_flat == 0)
    total = len(y_flat)
    
    if n_up == 0 or n_down == 0:
        return {'up_weight': 1.0, 'down_weight': 1.0, 'alpha': 0.5}
    
    # Inverse frequency weighting
    up_weight = total / (2 * n_up)
    down_weight = total / (2 * n_down)
    
    # Alpha for focal loss (weight for positive/up class)
    alpha = n_down / total  # More weight to minority class
    
    print(f"  Class distribution: {n_up} up ({100*n_up/total:.1f}%), {n_down} down ({100*n_down/total:.1f}%)")
    print(f"  Class weights: up={up_weight:.3f}, down={down_weight:.3f}, alpha={alpha:.3f}")
    
    return {'up_weight': up_weight, 'down_weight': down_weight, 'alpha': alpha}


def _direction_is_balanced(y_direction: np.ndarray, tolerance: float = 0.05) -> bool:
    """True when up-rate is within tolerance of 50%."""
    y_flat = np.array(y_direction).flatten()
    if y_flat.size == 0:
        return True
    up_rate = float(np.mean(y_flat == 1))
    return abs(up_rate - 0.5) <= float(tolerance)


# =============================================================================
# Custom Callbacks
# =============================================================================

class TradingMetricsCallback(keras_callbacks.Callback):
    """
    Custom callback to log trading-specific metrics to TensorBoard.
    Tracks directional accuracy, profit factor, etc.
    Supports both single-task and multi-task models.
    """
    
    def __init__(self, validation_data: Tuple[np.ndarray, np.ndarray], log_dir: str, multi_task: bool = False):
        super().__init__()
        self.validation_data = validation_data
        self.multi_task = multi_task
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'trading_metrics'))
    
    def _extract_price_from_predictions(self, predictions):
        """Extract price predictions from model output."""
        if isinstance(predictions, dict):
            return predictions.get('price', predictions[list(predictions.keys())[0]])
        if isinstance(predictions, (list, tuple)):
            return predictions[0]
        return predictions
    
    def _extract_price_from_targets(self, y_val):
        """Extract price targets from validation data."""
        if isinstance(y_val, dict):
            return y_val.get('price', y_val[list(y_val.keys())[0]])
        return y_val
    
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        predictions = self.model.predict(x_val, verbose=0)

        # Always extract price series for histograms (even if we compute direction-head accuracy).
        pred_price = self._extract_price_from_predictions(predictions)
        true_price = self._extract_price_from_targets(y_val)
        
        # Prefer true direction-head accuracy when available (multi-task).
        directional_accuracy = None
        if self.multi_task and isinstance(predictions, dict) and isinstance(y_val, dict):
            if 'direction' in predictions and 'direction' in y_val:
                pred_dir = (np.array(predictions['direction']).reshape(-1) >= 0.5).astype(np.int32)
                true_dir = (np.array(y_val['direction']).reshape(-1) >= 0.5).astype(np.int32)
                directional_accuracy = float(np.mean(pred_dir == true_dir))

        # Fallback (single-task only): keep legacy sign-based metric.
        if directional_accuracy is None:
            pred_price = np.array(pred_price).flatten()
            true_price = np.array(true_price).flatten()
            pred_direction = np.sign(pred_price)
            true_direction = np.sign(true_price)
            directional_accuracy = float(np.mean(pred_direction == true_direction))
        
        # Log to TensorBoard
        with self.writer.as_default():
            tf.summary.scalar('directional_accuracy', directional_accuracy, step=epoch)
            
            # Distribution of predictions
            tf.summary.histogram('predictions', pred_price, step=epoch)
            tf.summary.histogram('targets', true_price, step=epoch)
        
        logs = logs or {}
        logs['directional_accuracy'] = directional_accuracy


class GradientLoggingCallback(keras_callbacks.Callback):
    """Log gradient statistics to TensorBoard for debugging."""
    
    def __init__(self, log_dir: str, log_frequency: int = 10):
        super().__init__()
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'gradients'))
        self.log_frequency = log_frequency
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_frequency != 0:
            return
        
        with self.writer.as_default():
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    tf.summary.histogram(
                        f'{layer.name}/kernel', 
                        layer.kernel, 
                        step=batch
                    )
                if hasattr(layer, 'bias') and layer.bias is not None:
                    tf.summary.histogram(
                        f'{layer.name}/bias', 
                        layer.bias, 
                        step=batch
                    )


class LearningRateLoggerCallback(keras_callbacks.Callback):
    """Log learning rate to TensorBoard."""
    
    def __init__(self, log_dir: str):
        super().__init__()
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'lr'))
    
    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        with self.writer.as_default():
            tf.summary.scalar('learning_rate', lr, step=epoch)


class BatchProgressCallback(keras_callbacks.Callback):
    """Print periodic batch progress so long CPU epochs don't look hung."""

    def __init__(self, every_n_batches: int = 50):
        super().__init__()
        self.every_n_batches = int(every_n_batches)
        self._t0 = None
        self._epoch_t0 = None

    def on_train_begin(self, logs=None):
        self._t0 = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self.every_n_batches <= 0:
            return
        if batch == 0 or (batch + 1) % self.every_n_batches == 0:
            elapsed = time.time() - (self._epoch_t0 or time.time())
            loss = None if logs is None else logs.get('loss')
            if loss is not None:
                print(f"  [batch {batch+1}] loss={float(loss):.4f} elapsed={elapsed:.1f}s")
            else:
                print(f"  [batch {batch+1}] elapsed={elapsed:.1f}s")


class SavedModelExportCallback(keras_callbacks.Callback):
    """
    Also save model in TensorFlow SavedModel format for portable inference.
    
    The .keras format can have issues with custom layers across sessions.
    SavedModel format is more portable and can be loaded without class definitions.
    This is used by Buddy for live trading predictions.
    """
    
    def __init__(self, saved_model_dir: str, monitor: str = 'val_loss', best_only: bool = True):
        super().__init__()
        self.saved_model_dir = saved_model_dir
        self.monitor = monitor
        self.best_only = best_only
        self.best_val = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val = logs.get(self.monitor)
        
        if current_val is None:
            return
        
        if not self.best_only or current_val < self.best_val:
            self.best_val = current_val
            
            # Save weights in Keras 3 format (.weights.h5 is required)
            # This is the most reliable for custom layers
            weights_path = os.path.join(self.saved_model_dir, 'tf_model_weights.weights.h5')
            try:
                self.model.save_weights(weights_path)
                print(f"\n💾 Saved weights to {weights_path}")
            except Exception as e:
                print(f"\n⚠ Weights save failed: {e}")
            
            # Also save weights as NPZ for fallback loading
            # Use model.weights to get ALL weights including sublayers
            npz_path = os.path.join(self.saved_model_dir, 'tf_model_weights.npz')
            try:
                import numpy as np
                weight_dict = {}
                for var in self.model.weights:
                    # Use full path name for reliable matching
                    name = var.name.replace(':', '_')
                    weight_dict[name] = var.numpy()
                np.savez(npz_path, **weight_dict)
                print(f"💾 Saved NPZ weights to {npz_path} ({len(weight_dict)} arrays)")
            except Exception as e:
                print(f"⚠ NPZ weights save failed: {e}")
            
            # Try SavedModel export (may fail with untracked variables from dropout)
            export_path = os.path.join(self.saved_model_dir, 'tf_model_savedmodel')
            try:
                # Use tf.saved_model.save for maximum compatibility
                tf.saved_model.save(self.model, export_path)
                print(f"📦 Exported SavedModel to {export_path}/")
            except Exception:
                # This is expected to fail with Keras 3 seed generators
                print("⚠ SavedModel export skipped (Keras 3 seed generator issue)")


class AdaptiveLossWeightCallback(keras_callbacks.Callback):
    """
    Adaptive loss weight adjustment for multi-task learning.
    
    Monitors performance of each head and adjusts loss weights to balance
    training. Heads with higher loss get higher weights to improve their training.
    
    This helps achieve balanced training across all 4 heads (price, trend, risk, state).
    """
    
    def __init__(
        self,
        log_dir: str,
        initial_weights: Dict[str, float] = None,
        adjustment_factor: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        adjust_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.initial_weights = initial_weights or {
            'price': 1.0,
            'trend': 0.5,
            'direction': 10.0,
            'risk': 5.0,
            'state_logits': 7.0,
        }
        self.current_weights = dict(self.initial_weights)
        self.adjustment_factor = adjustment_factor
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adjust_every_n_epochs = adjust_every_n_epochs
        self.loss_history = {key: [] for key in self.initial_weights}
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'loss_weights'))
    
    def _record_head_losses(self, logs: Dict):
        """Record individual head losses from training logs."""
        for head in self.initial_weights:
            loss_key = f'{head}_loss'
            if loss_key in logs:
                self.loss_history[head].append(logs[loss_key])
    
    def _log_current_weights(self, epoch: int):
        """Log current weights to TensorBoard."""
        with self.writer.as_default():
            for head, weight in self.current_weights.items():
                tf.summary.scalar(f'loss_weight/{head}', weight, step=epoch)
    
    def _update_model_loss_weights(self):
        """Update model's loss weights (Keras 3.x compatible)."""
        if not hasattr(self.model, '_loss_weights'):
            return
        for head, weight in self.current_weights.items():
            if head in self.model._loss_weights:
                self.model._loss_weights[head] = weight
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        self._record_head_losses(logs)
        self._log_current_weights(epoch)
        
        # Adjust weights periodically
        should_adjust = (epoch + 1) % self.adjust_every_n_epochs == 0 and epoch > 0
        if should_adjust:
            self._adjust_weights()
            self._update_model_loss_weights()
    
    def _adjust_weights(self):
        """Adjust loss weights based on relative loss magnitudes."""
        # Get recent average losses for each head
        recent_losses = {}
        for head, loss_values in self.loss_history.items():
            if len(loss_values) >= self.adjust_every_n_epochs:
                recent_losses[head] = np.mean(loss_values[-self.adjust_every_n_epochs:])
        
        if not recent_losses:
            return
        
        # Normalize losses (higher loss = needs more attention)
        max_loss = max(recent_losses.values()) + 1e-8
        loss_ratios = {h: l / max_loss for h, l in recent_losses.items()}
        
        # Adjust weights: increase weight for heads with higher relative loss
        for head, ratio in loss_ratios.items():
            if head in self.current_weights:
                # Scale adjustment by how far loss is from average
                adjustment = self.adjustment_factor * (ratio - 0.5)
                new_weight = self.current_weights[head] * (1 + adjustment)
                
                # Clamp to valid range
                self.current_weights[head] = np.clip(
                    new_weight, self.min_weight, self.max_weight
                )
        
        print("\n📊 Adaptive loss weights updated:")
        for head, weight in self.current_weights.items():
            print(f"   {head}: {weight:.3f}")


# =============================================================================
# TensorFlow Training Engine
# =============================================================================

class TensorFlowEngine:
    """
    Complete TensorFlow training engine with TensorBoard visualization.
    
    Usage:
        engine = TensorFlowEngine(config)
        history = engine.train(X_train, y_train, X_val, y_val)
        
        # Launch TensorBoard:
        # tensorboard --logdir=trained_data/tensorboard
    """
    
    def __init__(self, config: Dict):
        """
        Initialize training engine.
        
        Args:
            config: Configuration dictionary with:
                - model: Model configuration
                - optimizer: Optimizer settings
                - training: Training parameters
                - paths: Directory paths
                - tensorboard: TensorBoard settings
        """
        self.config = config
        self.model = None
        self.history = None
        
        # Setup paths
        self.paths = config.get('paths', {})
        self.tensorboard_dir = self._setup_tensorboard_dir()
        self.checkpoint_dir = self.paths.get('checkpoint_dir', DEFAULT_CHECKPOINT_DIR)
        
        # Training settings
        self.training_config = config.get('training', {})
        self.epochs = self.training_config.get('epochs', config.get('epochs', 100))
        self.batch_size = config.get('batch_size', 64)
        self.early_stopping_patience = self.training_config.get(
            'early_stopping_patience', 
            config.get('early_stopping_patience', 15)
        )
        
        # Mixed precision
        if config.get('mixed_precision', False):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision training enabled (float16)")
        
        # GPU runtime tuning (Metal included)
        self._configure_gpu_runtime()
        
        # Setup strategy for multi-GPU if available
        self.strategy = self._setup_strategy()

    def _configure_gpu_runtime(self) -> None:
        """Best-effort GPU configuration for TensorFlow (incl. Apple Metal plugin)."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    # Some runtimes/devices may not support memory growth.
                    pass
        except Exception:
            pass
    
    def _setup_tensorboard_dir(self) -> str:
        """Create timestamped TensorBoard log directory."""
        base_dir = self.paths.get('tensorboard_dir', DEFAULT_TENSORBOARD_DIR)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        model_type = self.config.get('model', {}).get('type', 'model')
        log_dir = os.path.join(base_dir, f"{model_type}_{timestamp}")
        
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def _setup_strategy(self) -> tf.distribute.Strategy:
        """Setup distribution strategy for multi-GPU training."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"✓ Multi-GPU training enabled ({len(gpus)} GPUs)")
        elif len(gpus) == 1:
            strategy = tf.distribute.get_strategy()
            print(f"✓ Single GPU training: {gpus[0].name}")
        else:
            strategy = tf.distribute.get_strategy()
            print("⚠ No GPU found, using CPU")
        
        return strategy
    
    def build_model(self) -> keras.Model:
        """Build and compile the model."""
        model_config = self.config.get('model', {})
        multi_task = model_config.get('multi_task', True)

        training_cfg = self.config.get('training', {})
        model_type = (model_config.get('type', '') or '').lower()
        has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        run_eagerly_cfg = training_cfg.get('run_eagerly', self.config.get('run_eagerly', None))
        if run_eagerly_cfg is None:
            # TFT can take a long time to trace/compile the first train step on CPU.
            # Eager mode avoids the "stuck at model.fit" experience.
            run_eagerly = (not has_gpu) and (model_type in ['tft', 'temporal_fusion_transformer'])
        else:
            run_eagerly = bool(run_eagerly_cfg)
        jit_compile = bool(training_cfg.get('jit_compile', self.config.get('jit_compile', False)))
        
        with self.strategy.scope():
            # Create model
            self.model = create_tensorflow_model(model_config)
            
            # Setup optimizer
            optimizer = self._create_optimizer()
            
            if multi_task:
                # Multi-task loss configuration (like UnifiedMarketNet)
                # Enhanced with direction head for improved directional accuracy
                # Using Focal Loss + high weight for hard example mining
                loss_weights = self.config.get('unified_head_loss_weights', {
                    'price': 1.0,
                    'trend': 1.0,
                    'direction': 20.0,  # Very high: direction is critical for trading
                    'risk': 5.0,
                    'state_logits': 7.0,
                })
                
                # Get alpha from config or use default (0.5 = balanced)
                focal_alpha = self.config.get('focal_alpha', 0.5)
                focal_gamma = self.config.get('focal_gamma', 2.0)

                direction_loss_type = (self.config.get('direction_loss', 'focal') or 'focal').lower()
                if direction_loss_type in ('bce', 'binary_crossentropy'):
                    direction_loss = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
                else:
                    direction_loss = BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha, label_smoothing=0.0)
                
                self.model.compile(
                    optimizer=optimizer,
                    loss={
                        'price': losses.Huber(delta=self.config.get('huber_delta', 1.0)),
                        'trend': losses.Huber(delta=0.5),  # Smaller delta for finer predictions
                        'direction': direction_loss,
                        'risk': losses.MeanSquaredError(),  # Risk is continuous [0,1] regression
                        'state_logits': losses.CategoricalCrossentropy(label_smoothing=0.0),
                    },
                    loss_weights=loss_weights,
                    metrics={
                        'price': [metrics.MeanAbsoluteError(name='mae')],
                        'trend': [metrics.MeanAbsoluteError(name='trend_mae')],
                        'direction': [metrics.BinaryAccuracy(name='dir_acc')],  # Directional accuracy!
                        'risk': [metrics.MeanAbsoluteError(name='risk_mae')],  # Risk is continuous [0,1], not binary
                        'state_logits': [metrics.CategoricalAccuracy(name='state_acc')],
                    },
                    run_eagerly=run_eagerly,
                    jit_compile=jit_compile,
                )
                print("✓ Multi-task model compiled with 5 heads: price, trend, direction, risk, state")
                print(f"  Direction head loss: {direction_loss_type}")
                print(f"  Direction weight: {loss_weights.get('direction', 20.0)} (highest priority)")
                if run_eagerly:
                    print("  Note: run_eagerly=True to avoid long CPU tracing")
            else:
                # Single task (price only)
                loss = self._create_loss()
                self.model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=[
                        metrics.MeanAbsoluteError(name='mae'),
                        metrics.RootMeanSquaredError(name='rmse'),
                    ],
                    run_eagerly=run_eagerly,
                    jit_compile=jit_compile,
                )
        
        return self.model
    
    def _create_optimizer(self) -> optimizers.Optimizer:
        """Create optimizer from config with gradient clipping."""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw').lower()
        lr = opt_config.get('learning_rate', self.config.get('learning_rate', 0.001))
        weight_decay = opt_config.get('weight_decay', 0.0)
        
        # Gradient clipping (matches PyTorch's clip_grad_norm_ for stability)
        clipnorm = opt_config.get('clipnorm', self.config.get('gradient_clip', 1.0))

        is_darwin = platform.system() == 'Darwin'
        legacy = getattr(optimizers, 'legacy', None) if is_darwin else None

        def _pick_legacy(name: str, fallback):
            if legacy is not None and hasattr(legacy, name):
                return getattr(legacy, name)
            return fallback

        if opt_type == 'adamw':
            # On Apple Silicon / Metal, the v2.11+ non-legacy AdamW can be very slow.
            # Prefer the legacy implementation when available.
            if legacy is not None and hasattr(legacy, 'AdamW'):
                adamw_cls = legacy.AdamW
                return adamw_cls(
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    beta_1=opt_config.get('betas', [0.9, 0.999])[0],
                    beta_2=opt_config.get('betas', [0.9, 0.999])[1],
                    clipnorm=clipnorm,
                )

            if is_darwin:
                adam_cls = _pick_legacy('Adam', optimizers.Adam)
                print('Note: legacy.AdamW unavailable on this macOS build; using Adam (weight_decay ignored).')
                return adam_cls(learning_rate=lr, clipnorm=clipnorm)

            return optimizers.AdamW(
                learning_rate=lr,
                weight_decay=weight_decay,
                beta_1=opt_config.get('betas', [0.9, 0.999])[0],
                beta_2=opt_config.get('betas', [0.9, 0.999])[1],
                clipnorm=clipnorm,
            )

        if opt_type == 'adam':
            adam_cls = _pick_legacy('Adam', optimizers.Adam)
            return adam_cls(learning_rate=lr, clipnorm=clipnorm)

        if opt_type == 'sgd':
            sgd_cls = _pick_legacy('SGD', optimizers.SGD)
            return sgd_cls(
                learning_rate=lr,
                momentum=opt_config.get('momentum', 0.9),
                clipnorm=clipnorm,
            )

        if opt_type == 'rmsprop':
            rmsprop_cls = _pick_legacy('RMSprop', optimizers.RMSprop)
            return rmsprop_cls(learning_rate=lr, clipnorm=clipnorm)

        adam_cls = _pick_legacy('Adam', optimizers.Adam)
        return adam_cls(learning_rate=lr, clipnorm=clipnorm)
    
    def _create_loss(self) -> losses.Loss:
        """Create loss function from config."""
        loss_type = self.config.get('loss_type', 'mse').lower()
        
        if loss_type == 'huber':
            delta = self.config.get('huber_delta', 1.0)
            return losses.Huber(delta=delta)
        elif loss_type == 'mae':
            return losses.MeanAbsoluteError()
        elif loss_type == 'mse':
            return losses.MeanSquaredError()
        else:
            return losses.MeanSquaredError()
    
    def _create_callbacks(
        self, 
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> List[keras_callbacks.Callback]:
        """Create training callbacks including TensorBoard."""
        callback_list = []
        
        # TensorBoard callback - THE MAIN VISUALIZATION
        tb_config = self.config.get('tensorboard', {})
        tensorboard_callback = keras_callbacks.TensorBoard(
            log_dir=self.tensorboard_dir,
            histogram_freq=tb_config.get('histogram_freq', 1),
            write_graph=tb_config.get('write_graph', True),
            write_images=tb_config.get('write_images', False),
            update_freq='epoch',
            profile_batch=tb_config.get('profile_batch', 0),
        )
        callback_list.append(tensorboard_callback)
        
        # Early stopping
        early_stopping = keras_callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )
        callback_list.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            'tf_model_best.keras'
        )
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Use initial_value_threshold if resuming from a previous checkpoint
        # This ensures we only save if we beat the previous best val_loss
        initial_threshold = self.config.get('resume_best_val_loss', None)
        
        model_checkpoint = keras_callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            initial_value_threshold=initial_threshold,  # Only save if better than previous best
        )
        callback_list.append(model_checkpoint)
        
        # Also save in SavedModel format for Buddy inference
        # SavedModel is more portable than .keras for custom layers
        saved_model_callback = SavedModelExportCallback(
            saved_model_dir=self.checkpoint_dir,
            monitor='val_loss',
            best_only=True,
        )
        callback_list.append(saved_model_callback)
        
        # Learning rate reduction on plateau
        lr_reducer = keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        )
        callback_list.append(lr_reducer)
        
        # Learning rate logger
        callback_list.append(LearningRateLoggerCallback(self.tensorboard_dir))

        # Batch progress heartbeat (helps on CPU / first-step tracing stalls)
        every_n = int(self.training_config.get('batch_progress_every', self.config.get('batch_progress_every', 0)) or 0)
        if every_n > 0:
            callback_list.append(BatchProgressCallback(every_n_batches=every_n))
        
        # Trading metrics (if validation data provided)
        multi_task = self.config.get('model', {}).get('multi_task', False)
        if validation_data is not None:
            callback_list.append(
                TradingMetricsCallback(validation_data, self.tensorboard_dir, multi_task=multi_task)
            )
        
        # Adaptive loss weights for multi-task learning
        if multi_task and self.config.get('adaptive_loss_weights', True):
            initial_weights = self.config.get('unified_head_loss_weights', {
                'price': 1.0,
                'trend': 0.5,
                'risk': 5.0,
                'state_logits': 7.0,
            })
            callback_list.append(
                AdaptiveLossWeightCallback(
                    log_dir=self.tensorboard_dir,
                    initial_weights=initial_weights,
                    adjustment_factor=0.1,
                    adjust_every_n_epochs=5,
                )
            )
        
        # CSV logger for backup
        csv_path = os.path.join(self.tensorboard_dir, 'training_log.csv')
        callback_list.append(keras_callbacks.CSVLogger(csv_path))
        
        return callback_list
    
    def train(  # noqa: C901
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        validation_split: float = 0.1,
    ) -> keras.callbacks.History:
        """
        Train the model with TensorBoard visualization.
        
        Args:
            x_train: Training features [samples, timesteps, features]
            y_train: Training targets [samples, 1]
            x_val: Validation features (optional)
            y_val: Validation targets (optional)
            validation_split: Fraction for validation if X_val not provided
        
        Returns:
            Training history
        
        To visualize training:
            tensorboard --logdir=trained_data/tensorboard
        """
        # If multi-task, auto-tune focal-loss alpha from train direction balance
        # BEFORE compiling the model.
        if self.model is None:
            multi_task = self.config.get('model', {}).get('multi_task', False)
            if multi_task and isinstance(y_train, dict) and 'direction' in y_train:
                if 'direction_loss' not in self.config:
                    is_balanced = _direction_is_balanced(y_train['direction'], tolerance=0.05)
                    self.config['direction_loss'] = 'bce' if is_balanced else 'focal'
                if self.config.get('auto_focal_alpha', True) and 'focal_alpha' not in self.config:
                    cw = compute_class_weights(y_train['direction'])
                    self.config['focal_alpha'] = float(cw.get('alpha', 0.5))
            self.build_model()
        if self.model is None:
            multi_task = self.config.get('model', {}).get('multi_task', False)
            if multi_task and isinstance(y_train, dict) and 'direction' in y_train:
                if 'direction_loss' not in self.config:
                    is_balanced = _direction_is_balanced(y_train['direction'], tolerance=0.05)
                    self.config['direction_loss'] = 'bce' if is_balanced else 'focal'
                if self.config.get('auto_focal_alpha', True) and 'focal_alpha' not in self.config:
                    cw = compute_class_weights(y_train['direction'])
                    self.config['focal_alpha'] = float(cw.get('alpha', 0.5))
            self.build_model()
        
        # Prepare validation data
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        else:
            validation_data = None
        
        # Create callbacks
        callbacks = self._create_callbacks(validation_data)
        
        # Print training info
        print("\n" + "="*60)
        print("TensorFlow Training Engine")
        print("="*60)
        print(f"Model: {type(self.model).__name__}")
        print(f"Training samples: {len(x_train):,}")
        if validation_data:
            print(f"Validation samples: {len(x_val):,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"\n📊 TensorBoard logs: {self.tensorboard_dir}")
        print(f"   Run: tensorboard --logdir={self.tensorboard_dir}")
        print("="*60 + "\n")
        
        print("Starting model.fit()… (first batch can be slow on CPU)")
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else 0,
            callbacks=callbacks,
            verbose=1,
        )
        
        print("\n✓ Training complete!")
        print(f"📊 View results: tensorboard --logdir={self.tensorboard_dir}")
        
        return self.history
    
    def evaluate(self, x_test: np.ndarray, y_test, verbose: int = 1) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            x_test: Test features [samples, timesteps, features]
            y_test: Test targets - can be dict for multi-task or array for single-task
            verbose: Verbosity level
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = self.model.evaluate(x_test, y_test, verbose=verbose, return_dict=True)
        
        # Calculate additional metrics based on task type
        multi_task = self.config.get('model', {}).get('multi_task', False)
        predictions = self.model.predict(x_test, verbose=0)
        
        if multi_task:
            self._add_multi_task_metrics(results, predictions, y_test, x_test)
        else:
            self._add_single_task_metrics(results, predictions, y_test)
        
        return results
    
    def _extract_price_predictions(self, predictions, n_samples: int):
        """Extract price predictions from model output."""
        if isinstance(predictions, dict):
            return predictions.get('price', np.zeros(n_samples))
        if isinstance(predictions, (list, tuple)):
            return predictions[0]
        return predictions
    
    def _extract_price_targets(self, y_test, n_samples: int):
        """Extract price targets from test data."""
        if isinstance(y_test, dict):
            return y_test.get('price', np.zeros(n_samples))
        return y_test
    
    def _compute_directional_accuracy(self, pred_price, true_price) -> float:
        """Compute directional accuracy between predictions and targets."""
        pred_price = np.array(pred_price).flatten()
        true_price = np.array(true_price).flatten()
        pred_direction = np.sign(pred_price)
        true_direction = np.sign(true_price)
        return float(np.mean(pred_direction == true_direction))
    
    def _add_multi_task_metrics(self, results: Dict, predictions, y_test, x_test):
        """Add metrics for multi-task model evaluation."""
        n_samples = len(x_test)
        pred_price = self._extract_price_predictions(predictions, n_samples)
        true_price = self._extract_price_targets(y_test, n_samples)
        
        results['price_directional_accuracy'] = self._compute_directional_accuracy(pred_price, true_price)
        
        # State accuracy (if state predictions available)
        if isinstance(predictions, dict) and 'state_logits' in predictions:
            if isinstance(y_test, dict) and 'state_logits' in y_test:
                state_preds = np.argmax(predictions['state_logits'], axis=-1)
                state_true = np.argmax(y_test['state_logits'], axis=-1)
                results['state_head_accuracy'] = float(np.mean(state_preds == state_true))
    
    def _add_single_task_metrics(self, results: Dict, predictions, y_test):
        """Add metrics for single-task model evaluation."""
        pred_price = np.array(predictions).flatten()
        true_price = np.array(y_test).flatten()
        results['directional_accuracy'] = self._compute_directional_accuracy(pred_price, true_price)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(x, verbose=0)
    
    def save(self, path: Optional[str] = None, scalers: Dict = None):
        """Save the model and optionally scalers.
        
        Args:
            path: Path to save model. If None, uses default checkpoint_dir.
            scalers: Optional dict of scalers to save alongside model.
        """
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'tf_model_final.keras')
        
        self.model.save(path)
        print(f"✓ Model saved to {path}")
        
        # Save scalers if provided
        if scalers:
            scaler_path = os.path.join(os.path.dirname(path), 'scalers')
            Path(scaler_path).mkdir(parents=True, exist_ok=True)
            
            import joblib
            for name, scaler in scalers.items():
                if scaler is not None:
                    scaler_file = os.path.join(scaler_path, f'{name}_scaler.joblib')
                    joblib.dump(scaler, scaler_file)
            print(f"✓ Scalers saved to {scaler_path}")
    
    def load(self, path: str, load_scalers: bool = False) -> Optional[Dict]:
        """Load a saved model and optionally scalers.
        
        Args:
            path: Path to saved model.
            load_scalers: If True, also load scalers from same directory.
            
        Returns:
            Dict of scalers if load_scalers=True, else None.
        """
        self.model = keras.models.load_model(path)
        print(f"✓ Model loaded from {path}")
        
        if load_scalers:
            scaler_path = os.path.join(os.path.dirname(path), 'scalers')
            if os.path.exists(scaler_path):
                import joblib
                scalers = {}
                for scaler_file in Path(scaler_path).glob('*_scaler.joblib'):
                    name = scaler_file.stem.replace('_scaler', '')
                    scalers[name] = joblib.load(scaler_file)
                print(f"✓ Scalers loaded from {scaler_path}")
                return scalers
            else:
                print(f"⚠ No scalers found at {scaler_path}")
        
        return None
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")


# =============================================================================
# Quick Training Function
# =============================================================================

def quick_train_tensorflow(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'attention_lstm',
    epochs: int = 100,
    batch_size: int = 64,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
) -> Tuple[keras.Model, keras.callbacks.History, str]:
    """
    Quick function to train a TensorFlow model with TensorBoard.
    
    Returns:
        (model, history, tensorboard_dir)
    
    Example:
        model, history, tb_dir = quick_train_tensorflow(
            x_train, y_train, x_val, y_val,
            model_type='tft',
            epochs=50
        )
        
        # View training:
        # tensorboard --logdir={tb_dir}
    """
    config = {
        'model': {
            'type': model_type,
            'input_size': x_train.shape[-1],
            'hidden_size': hidden_size,
            'num_layers': 3,
            'dropout': 0.2,
            'num_heads': 4,
            'sequence_length': x_train.shape[1],
        },
        'optimizer': {
            'type': 'adamw',
            'learning_rate': learning_rate,
        },
        'training': {
            'epochs': epochs,
            'early_stopping_patience': 15,
        },
        'batch_size': batch_size,
        'loss_type': 'huber',
        'paths': {
            'tensorboard_dir': DEFAULT_TENSORBOARD_DIR,
            'checkpoint_dir': DEFAULT_CHECKPOINT_DIR,
        },
        'tensorboard': {
            'histogram_freq': 1,
            'write_graph': True,
        },
    }
    
    engine = TensorFlowEngine(config)
    engine.build_model()
    history = engine.train(x_train, y_train, x_val, y_val)
    
    return engine.model, history, engine.tensorboard_dir


# =============================================================================
# Launch TensorBoard Helper
# =============================================================================

def launch_tensorboard(log_dir: str = DEFAULT_TENSORBOARD_DIR, port: int = 6006):
    """
    Launch TensorBoard in a subprocess.
    
    Note: This will block. For non-blocking, run in terminal:
        tensorboard --logdir=trained_data/tensorboard --port=6006
    """
    import subprocess
    import webbrowser
    import time
    
    print("\n🚀 Launching TensorBoard...")
    print(f"   Log directory: {log_dir}")
    print(f"   URL: http://localhost:{port}")
    print("   Press Ctrl+C to stop\n")
    
    # Start TensorBoard
    process = subprocess.Popen(
        ['tensorboard', '--logdir', log_dir, '--port', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait a bit then open browser
    time.sleep(3)
    webbrowser.open(f'http://localhost:{port}')
    
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\n✓ TensorBoard stopped")


if __name__ == "__main__":
    print("TensorFlow Training Engine")
    print("="*60)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Example configuration
    example_config = {
        'model': {
            'type': 'tft',  # temporal_fusion_transformer
            'input_size': 104,
            'hidden_size': 64,
            'dropout': 0.2,
            'num_heads': 4,
            'sequence_length': 60,
        },
        'optimizer': {
            'type': 'adamw',
            'learning_rate': 0.0005,
        },
        'training': {
            'epochs': 100,
            'early_stopping_patience': 15,
        },
        'batch_size': 64,
        'loss_type': 'huber',
        'paths': {
            'tensorboard_dir': DEFAULT_TENSORBOARD_DIR,
            'checkpoint_dir': DEFAULT_CHECKPOINT_DIR,
        },
    }
    
    print("\nExample usage:")
    print("-"*60)
    print("""
from tensorflow_engine import TensorFlowEngine

# Create engine
engine = TensorFlowEngine(config)
engine.build_model()

# Train with TensorBoard visualization
history = engine.train(X_train, y_train, X_val, y_val)

# View training progress:
# tensorboard --logdir=trained_data/tensorboard

# Evaluate
results = engine.evaluate(X_test, y_test)
print(results)
""")
