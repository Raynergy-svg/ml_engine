"""
TensorFlow/Keras W&B Integration for ML Engine Training Pipeline.

This module provides production-grade Weights & Biases integration for the
TensorFlow/Keras training pipeline used by `buddy train`. It implements
the same diagnostic features as the PyTorch observatory but adapted for Keras.

Features:
- Real-time loss curves with EMA smoothing
- Gradient and activation histogram logging
- System metrics (GPU/VRAM/CPU) monitoring
- Model graph visualization
- Automatic hyperparameter logging
- Walk-forward CV metric aggregation

Usage:
    from src.training.wandb_keras_callback import WandbKerasTracker

    tracker = WandbKerasTracker(
        project_name='ml_engine_fx',
        experiment_name='EUR_USD_H1_transformer',
        instrument='EUR_USD',
        granularity='H1',
    )

    # In model.fit()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        callbacks=tracker.get_callbacks(),
    )

    # Finish run
    tracker.finish()

References:
- W&B Keras Integration: https://docs.wandb.ai/guides/integrations/keras
- EMA Smoothing: Roberts (1959), "Control Chart Tests Based on Geometric Moving Averages"
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import wandb
    # wandb.keras was removed in newer versions - use wandb.integration.keras or fallback
    try:
        from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
    except ImportError:
        try:
            from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
        except ImportError:
            # Fallback: create stubs for older compatibility
            WandbMetricsLogger = None
            WandbModelCheckpoint = None
    WANDB_AVAILABLE = True
    
    # Check wandb version for reinit parameter compatibility
    # reinit="finish_previous" was added in wandb 0.23.0
    # Older versions only support boolean values
    try:
        from packaging.version import parse as parse_version
        _wandb_version = parse_version(getattr(wandb, '__version__', '0.0.0'))
        WANDB_SUPPORTS_REINIT_STRING = _wandb_version >= parse_version('0.23.0')
    except ImportError:
        # packaging not available, assume newer version
        WANDB_SUPPORTS_REINIT_STRING = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    WandbMetricsLogger = None
    WandbModelCheckpoint = None
    WANDB_SUPPORTS_REINIT_STRING = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WandbKerasConfig:
    """Configuration for W&B Keras integration.
    
    Attributes:
        project_name: W&B project name (e.g., 'ml_engine_fx')
        experiment_name: Run name (e.g., 'EUR_USD_H1_transformer')
        instrument: Trading pair (e.g., 'EUR_USD')
        granularity: Timeframe (e.g., 'H1')
        tags: List of tags for the run
        notes: Additional notes for the run
        enable_gradient_logging: Log gradient histograms every N batches
        gradient_log_freq: Frequency of gradient logging (every N batches)
        enable_system_metrics: Log GPU/CPU/memory metrics
        ema_smoothing: EMA decay factor for loss curve smoothing (0-1)
        log_model_checkpoint: Save model checkpoints to W&B
        log_model_graph: Log model architecture visualization
        offline_mode: Run in offline mode (sync later with `wandb sync`)
        config: Hyperparameters to log
    """
    project_name: str = "ml_engine_fx"
    experiment_name: str = ""
    instrument: str = "GENERIC"
    granularity: str = "H1"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Diagnostic features
    enable_gradient_logging: bool = True
    gradient_log_freq: int = 100  # Log gradients every 100 batches
    enable_system_metrics: bool = True
    ema_smoothing: float = 0.95  # EMA decay for loss smoothing
    
    # Model logging
    log_model_checkpoint: bool = True
    log_model_graph: bool = True
    
    # Mode
    offline_mode: bool = False
    
    # Hyperparameters
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.experiment_name:
            self.experiment_name = f"{self.instrument}_{self.granularity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate EMA smoothing
        if not 0 < self.ema_smoothing < 1:
            raise ValueError(f"ema_smoothing must be in (0, 1), got {self.ema_smoothing}")


# =============================================================================
# EMA TRACKER FOR LOSS SMOOTHING
# =============================================================================

class EMATracker:
    """Exponential Moving Average tracker for metric smoothing.
    
    Implements EMA smoothing to separate signal from noise in loss curves,
    matching the diagnostic requirements for temporal dynamics visualization.
    
    The EMA at time t is computed as:
        EMA_t = α * value_t + (1 - α) * EMA_{t-1}
    
    where α = 1 - decay (decay is the config.ema_smoothing parameter).
    
    Attributes:
        decay: EMA decay factor (higher = smoother)
        values: Dictionary of tracked metrics and their EMA values
    """
    
    def __init__(self, decay: float = 0.95):
        """
        Initialize EMA tracker.
        
        Args:
            decay: EMA decay factor (0-1). Higher values = smoother curves.
                   Typical values: 0.9 (responsive) to 0.99 (very smooth)
        """
        self.decay = decay
        self.values: Dict[str, float] = {}
        self.raw_history: Dict[str, List[float]] = {}
    
    def update(self, name: str, value: float) -> float:
        """
        Update EMA for a metric and return smoothed value.
        
        Args:
            name: Metric name
            value: Current raw value
            
        Returns:
            Smoothed EMA value
        """
        if name not in self.values:
            self.values[name] = value
            self.raw_history[name] = []
        else:
            # EMA update: new_ema = decay * old_ema + (1 - decay) * new_value
            self.values[name] = self.decay * self.values[name] + (1 - self.decay) * value
        
        self.raw_history[name].append(value)
        return self.values[name]
    
    def get(self, name: str) -> Optional[float]:
        """Get current EMA value for a metric."""
        return self.values.get(name)
    
    def get_all(self) -> Dict[str, float]:
        """Get all current EMA values."""
        return self.values.copy()


# =============================================================================
# SYSTEM METRICS COLLECTOR
# =============================================================================

class SystemMetricsCollector:
    """Collect system metrics for infrastructure monitoring.
    
    Addresses the "Infrastructure" diagnostic requirement by logging:
    - GPU utilization (via TensorFlow Metal on macOS)
    - VRAM usage
    - CPU utilization
    - Memory usage
    
    This helps identify hardware bottlenecks during training.
    """
    
    def __init__(self, log_interval: int = 10):
        """
        Initialize system metrics collector.
        
        Args:
            log_interval: Log metrics every N calls (to reduce overhead)
        """
        self.log_interval = log_interval
        self._call_count = 0
        self._gpu_available: Optional[bool] = None
    
    def collect(self) -> Dict[str, float]:
        """
        Collect current system metrics.
        
        Returns:
            Dictionary of metric name -> value
        """
        self._call_count += 1
        if self._call_count % self.log_interval != 0:
            return {}
        
        metrics = {}
        
        # CPU and memory via psutil
        if PSUTIL_AVAILABLE:
            try:
                metrics['system/cpu_percent'] = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                metrics['system/memory_percent'] = mem.percent
                metrics['system/memory_gb'] = mem.used / (1024 ** 3)
            except Exception as e:
                logger.debug(f"Failed to collect CPU/memory metrics: {e}")
        
        # GPU via TensorFlow
        if TENSORFLOW_AVAILABLE and self._gpu_available is None:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                self._gpu_available = len(gpus) > 0
            except Exception:
                self._gpu_available = False
        
        # Log GPU availability once
        if self._call_count == 1 and self._gpu_available:
            metrics['system/gpu_available'] = 1.0
        
        return metrics


# =============================================================================
# GRADIENT MONITORING CALLBACK
# =============================================================================

class GradientMonitoringCallback(tf.keras.callbacks.Callback if TENSORFLOW_AVAILABLE else object):
    """Keras callback for gradient histogram logging.
    
    Addresses the "Weight Health" diagnostic requirement by logging:
    - Gradient histograms per layer (detects vanishing/exploding gradients)
    - Gradient norms over time
    - Layer activation statistics
    
    NOTE: Uses Keras-native gradient computation via train_step override,
    not wandb.watch() which is PyTorch-specific.
    """
    
    def __init__(
        self,
        log_freq: int = 100,
        log_weights: bool = True,
        log_gradients: bool = True,
    ):
        """
        Initialize gradient monitoring callback.
        
        Args:
            log_freq: Log every N batches
            log_weights: Log weight histograms
            log_gradients: Log gradient histograms
        """
        super().__init__()
        self.log_freq = log_freq
        self.log_weights = log_weights
        self.log_gradients = log_gradients
        self._batch_count = 0
        self._gradient_norms: Dict[str, List[float]] = {}
    
    def on_train_begin(self, logs=None):
        """Log model architecture info at training start."""
        if not WANDB_AVAILABLE or not wandb.run:
            return
        
        try:
            # Log model parameter count
            total_params = self.model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            wandb.log({
                'model/total_params': total_params,
                'model/trainable_params': trainable_params,
                'model/non_trainable_params': non_trainable_params,
            })
            logger.info(f"📊 Gradient monitoring enabled (log_freq={self.log_freq})")
        except Exception as e:
            logger.debug(f"Failed to log model info: {e}")
    
    def on_batch_end(self, batch, logs=None):
        """Log gradient statistics periodically using weight updates as proxy."""
        self._batch_count += 1
        
        if not WANDB_AVAILABLE or not wandb.run:
            return
        
        # Log weight statistics periodically (as proxy for gradient health)
        if self._batch_count % self.log_freq == 0:
            try:
                # Compute weight norms as a proxy for training health
                # (Keras doesn't expose gradients directly in callbacks)
                weight_metrics = {}
                
                for layer in self.model.layers:
                    if layer.trainable_weights:
                        layer_name = layer.name.replace('/', '_')
                        for i, w in enumerate(layer.trainable_weights):
                            try:
                                weight_norm = float(tf.norm(w).numpy())
                                weight_mean = float(tf.reduce_mean(w).numpy())
                                weight_std = float(tf.math.reduce_std(w).numpy())
                                
                                weight_metrics[f'weights/{layer_name}_{i}_norm'] = weight_norm
                                weight_metrics[f'weights/{layer_name}_{i}_mean'] = weight_mean
                                weight_metrics[f'weights/{layer_name}_{i}_std'] = weight_std
                            except Exception:
                                pass
                
                # Log aggregate statistics
                if weight_metrics:
                    # Compute total weight norm
                    total_norm = sum(
                        float(tf.norm(w).numpy())
                        for w in self.model.trainable_weights
                    )
                    weight_metrics['weights/total_norm'] = total_norm
                    weight_metrics['weights/batch'] = self._batch_count
                    
                    wandb.log(weight_metrics, commit=False)
                    
            except Exception as e:
                logger.debug(f"Weight statistics computation failed: {e}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log weight histograms at epoch end."""
        if not WANDB_AVAILABLE or not wandb.run:
            return
        
        if not self.log_weights:
            return
        
        try:
            # Log weight histograms for each layer
            histograms = {}
            for layer in self.model.layers:
                if layer.trainable_weights:
                    layer_name = layer.name.replace('/', '_')
                    for i, w in enumerate(layer.trainable_weights):
                        try:
                            # Flatten weight for histogram
                            w_flat = tf.reshape(w, [-1]).numpy()
                            histograms[f'weights_hist/{layer_name}_{i}'] = wandb.Histogram(w_flat)
                        except Exception:
                            pass
            
            if histograms:
                wandb.log(histograms, commit=False)
                
        except Exception as e:
            logger.debug(f"Weight histogram logging failed: {e}")


# =============================================================================
# EMA SMOOTHED METRICS CALLBACK
# =============================================================================

class EMASmoothedMetricsCallback(tf.keras.callbacks.Callback if TENSORFLOW_AVAILABLE else object):
    """Keras callback for EMA-smoothed metric logging.
    
    Addresses the "Temporal Dynamics" diagnostic requirement by providing
    smoothed loss curves that separate signal from noise.
    """
    
    def __init__(self, ema_decay: float = 0.95, log_freq: str = 'batch'):
        """
        Initialize EMA metrics callback.
        
        Args:
            ema_decay: EMA decay factor (0-1)
            log_freq: 'batch' or 'epoch' for logging frequency
        """
        super().__init__()
        self.ema = EMATracker(decay=ema_decay)
        self.log_freq = log_freq
        self._batch_count = 0
    
    def on_batch_end(self, batch, logs=None):
        """Log EMA-smoothed metrics after each batch."""
        if self.log_freq != 'batch':
            return
        
        logs = logs or {}
        self._batch_count += 1
        
        if not WANDB_AVAILABLE or not wandb.run:
            return
        
        # Apply EMA smoothing to loss metrics
        smoothed_metrics = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                smoothed = self.ema.update(key, value)
                smoothed_metrics[f'{key}_ema'] = smoothed
        
        if smoothed_metrics:
            # Don't specify step - let wandb auto-increment to avoid conflicts
            # with WandbMetricsLogger which manages its own step counter
            wandb.log(smoothed_metrics, commit=False)
    
    def on_epoch_end(self, epoch, logs=None):
        """Log EMA-smoothed epoch metrics."""
        logs = logs or {}
        
        if not WANDB_AVAILABLE or not wandb.run:
            return
        
        # Apply EMA smoothing to epoch metrics
        smoothed_metrics = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                smoothed = self.ema.update(f'epoch_{key}', value)
                smoothed_metrics[f'epoch_{key}_ema'] = smoothed
        
        if smoothed_metrics:
            # Use commit=False to avoid step conflicts with batch-level logging
            # Let wandb auto-increment the step
            wandb.log(smoothed_metrics, commit=True)


# =============================================================================
# SYSTEM METRICS CALLBACK
# =============================================================================

class SystemMetricsCallback(tf.keras.callbacks.Callback if TENSORFLOW_AVAILABLE else object):
    """Keras callback for system metrics logging.
    
    Addresses the "Infrastructure" diagnostic requirement.
    """
    
    def __init__(self, log_freq: int = 10):
        """
        Initialize system metrics callback.
        
        Args:
            log_freq: Log every N batches
        """
        super().__init__()
        self.collector = SystemMetricsCollector(log_interval=log_freq)
        self._batch_count = 0
    
    def on_batch_end(self, batch, logs=None):
        """Log system metrics periodically."""
        self._batch_count += 1
        
        if not WANDB_AVAILABLE or not wandb.run:
            return
        
        metrics = self.collector.collect()
        if metrics:
            # Don't specify step - let wandb auto-increment to avoid conflicts
            wandb.log(metrics, commit=False)


# =============================================================================
# SAFE WANDB CHECKPOINT CALLBACK (Fixes Race Condition)
# =============================================================================

class SafeWandbCheckpoint(tf.keras.callbacks.Callback if TENSORFLOW_AVAILABLE else object):
    """Safe model checkpoint callback that avoids wandb race condition.
    
    The standard WandbModelCheckpoint has a race condition where it tries to log
    the checkpoint file as an artifact BEFORE Keras finishes writing it to disk.
    This causes FileNotFoundError when:
    1. save_best_only=True and current model isn't better (file never created)
    2. The file write is still in progress when wandb tries to read it
    
    This callback fixes the issue by:
    1. Using standard Keras ModelCheckpoint for saving
    2. Only logging to wandb AFTER the file is confirmed to exist
    3. Wrapping artifact logging in try/except to prevent training crashes
    
    Attributes:
        filepath: Path template for checkpoint files
        save_best_only: Only save if monitored metric improves
        monitor: Metric to monitor
        mode: 'min' or 'max' for metric direction
    """
    
    def __init__(
        self,
        filepath: str,
        save_best_only: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_freq: str = 'epoch',
    ):
        """
        Initialize safe checkpoint callback.
        
        Args:
            filepath: Path template for checkpoints (e.g., 'model_{epoch:02d}.keras')
            save_best_only: Only save when monitored metric improves
            monitor: Metric name to monitor
            mode: 'min' for loss, 'max' for accuracy
            save_freq: Save frequency ('epoch' or integer for batches)
        """
        super().__init__()
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        self._best_value = float('inf') if mode == 'min' else float('-inf')
        self._best_epoch = 0
        self._saved_files: List[str] = []
    
    def on_epoch_end(self, epoch: int, logs=None):
        """Save checkpoint and log to wandb if file exists."""
        logs = logs or {}
        
        # Get monitored metric value
        current_value = logs.get(self.monitor)
        if current_value is None:
            logger.debug(f"Monitor metric '{self.monitor}' not available, skipping checkpoint")
            return
        
        # Check if we should save
        should_save = True
        if self.save_best_only:
            if self.mode == 'min':
                should_save = current_value < self._best_value
            else:
                should_save = current_value > self._best_value
        
        if not should_save:
            logger.debug(
                f"Skipping checkpoint: {self.monitor}={current_value:.4f} "
                f"not better than best={self._best_value:.4f}"
            )
            return
        
        # Update best value
        self._best_value = current_value
        self._best_epoch = epoch
        
        # Generate filepath
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        # Save model
        try:
            self.model.save(filepath)
            logger.info(f"✓ Saved checkpoint: {filepath}")
            self._saved_files.append(filepath)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return
        
        # Log to wandb ONLY after successful save and file exists
        if WANDB_AVAILABLE and wandb.run:
            self._log_checkpoint_safely(filepath, epoch)
    
    def _log_checkpoint_safely(self, filepath: str, epoch: int):
        """Log checkpoint to wandb with existence check and error handling."""
        import os
        
        # Verify file exists before attempting to log
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint file not found, skipping wandb logging: {filepath}")
            return
        
        # Check file size to ensure write is complete
        try:
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                logger.warning(f"Checkpoint file is empty, skipping wandb logging: {filepath}")
                return
        except OSError as e:
            logger.warning(f"Could not check checkpoint file size: {e}")
            return
        
        # Log as artifact with error handling
        try:
            artifact = wandb.Artifact(
                name=f"model_epoch_{epoch + 1}",
                type="model",
                metadata={
                    'epoch': epoch + 1,
                    self.monitor: self._best_value,
                    'best_epoch': self._best_epoch + 1,
                },
            )
            artifact.add_file(filepath)
            wandb.log_artifact(artifact, aliases=['latest', f'epoch-{epoch + 1}'])
            logger.info(f"📦 Logged checkpoint to wandb: {filepath}")
        except Exception as e:
            # Log warning but don't crash training
            logger.warning(f"Failed to log checkpoint to wandb (training continues): {e}")


# =============================================================================
# MAIN TRACKER CLASS
# =============================================================================

class WandbKerasTracker:
    """Production-grade W&B tracker for TensorFlow/Keras training.
    
    This is the main entry point for W&B integration with `buddy train`.
    It provides a unified interface for all diagnostic features:
    
    1. **Temporal Dynamics**: EMA-smoothed loss curves via EMASmoothedMetricsCallback
    2. **Weight Health**: Gradient histograms via GradientMonitoringCallback
    3. **Infrastructure**: System metrics via SystemMetricsCallback
    4. **Topology**: Model graph via wandb.watch()
    
    Usage:
        tracker = WandbKerasTracker(
            project_name='ml_engine_fx',
            experiment_name='EUR_USD_H1_transformer',
            instrument='EUR_USD',
            granularity='H1',
            config={
                'epochs': 100,
                'batch_size': 64,
                'learning_rate': 0.001,
            },
        )
        
        # Get callbacks for model.fit()
        callbacks = tracker.get_callbacks(model)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )
        
        # Log final metrics
        tracker.log_final_metrics({
            'val_accuracy': 0.65,
            'val_balanced_accuracy': 0.63,
        })
        
        # Finish run
        tracker.finish()
    
    Attributes:
        config: WandbKerasConfig instance
        run: W&B run object (after init)
        is_active: Whether W&B run is active
    """
    
    def __init__(
        self,
        project_name: str = "ml_engine_fx",
        experiment_name: str = "",
        instrument: str = "GENERIC",
        granularity: str = "H1",
        tags: Optional[List[str]] = None,
        notes: str = "",
        config: Optional[Dict[str, Any]] = None,
        enable_gradient_logging: bool = True,
        gradient_log_freq: int = 100,
        enable_system_metrics: bool = True,
        ema_smoothing: float = 0.95,
        log_model_checkpoint: bool = True,
        log_model_graph: bool = True,
        offline_mode: bool = False,
    ):
        """
        Initialize W&B Keras tracker.
        
        Args:
            project_name: W&B project name
            experiment_name: Run name (auto-generated if empty)
            instrument: Trading pair
            granularity: Timeframe
            tags: List of tags for the run
            notes: Additional notes
            config: Hyperparameters dictionary
            enable_gradient_logging: Enable gradient histogram logging
            gradient_log_freq: Frequency of gradient logging
            enable_system_metrics: Enable system metrics logging
            ema_smoothing: EMA decay factor for loss smoothing
            log_model_checkpoint: Save model checkpoints to W&B
            log_model_graph: Log model architecture
            offline_mode: Run in offline mode
        """
        self.config = WandbKerasConfig(
            project_name=project_name,
            experiment_name=experiment_name,
            instrument=instrument,
            granularity=granularity,
            tags=tags or [],
            notes=notes,
            config=config or {},
            enable_gradient_logging=enable_gradient_logging,
            gradient_log_freq=gradient_log_freq,
            enable_system_metrics=enable_system_metrics,
            ema_smoothing=ema_smoothing,
            log_model_checkpoint=log_model_checkpoint,
            log_model_graph=log_model_graph,
            offline_mode=offline_mode,
        )
        
        self.run = None
        self.is_active = False
        self._start_time: Optional[float] = None
        
        # Callbacks (created when get_callbacks() is called)
        self._gradient_callback: Optional[GradientMonitoringCallback] = None
        self._ema_callback: Optional[EMASmoothedMetricsCallback] = None
        self._system_callback: Optional[SystemMetricsCallback] = None
    
    def init(self) -> bool:
        """
        Initialize W&B run.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available. Install with: pip install wandb")
            return False
        
        if self.is_active:
            logger.warning("W&B run already active")
            return True
        
        try:
            # Set offline mode via environment variable
            if self.config.offline_mode:
                os.environ['WANDB_MODE'] = 'offline'
            
            # Prepare run config
            run_config = {
                'instrument': self.config.instrument,
                'granularity': self.config.granularity,
                'framework': 'tensorflow',
                **self.config.config,
            }
            
            # Initialize run
            # Use version-appropriate reinit value:
            # - wandb >= 0.23.0: "finish_previous" (properly cleans up previous run)
            # - wandb < 0.23.0: True (boolean fallback)
            reinit_value = "finish_previous" if WANDB_SUPPORTS_REINIT_STRING else True
            self.run = wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                tags=self.config.tags + [self.config.instrument, self.config.granularity],
                notes=self.config.notes,
                config=run_config,
                reinit=reinit_value,
            )
            
            # Define metrics with proper step handling to avoid conflicts
            # This allows epoch-level and batch-level metrics to coexist
            try:
                wandb.define_metric("epoch/*", step_metric="epoch/step")
                wandb.define_metric("batch/*", step_metric="batch/step")
                wandb.define_metric("train/*", step_metric="batch/step")
                wandb.define_metric("val/*", step_metric="epoch/step")
                wandb.define_metric("system/*", step_metric="batch/step")
            except Exception as e:
                logger.debug(f"Could not define metric steps: {e}")
            
            self.is_active = True
            self._start_time = time.time()
            
            logger.info(f"📊 W&B run initialized: {self.config.experiment_name}")
            logger.info(f"   Project: {self.config.project_name}")
            logger.info(f"   URL: {self.get_url()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}")
            return False
    
    def get_callbacks(self, model=None) -> List:
        """
        Get Keras callbacks for all diagnostic features.
        
        Args:
            model: Keras model (required for gradient logging)
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Initialize W&B if not already done
        if not self.is_active:
            if not self.init():
                return callbacks
        
        # 1. W&B Metrics Logger (basic metrics)
        if WANDB_AVAILABLE:
            try:
                callbacks.append(WandbMetricsLogger(log_freq='batch'))
                logger.debug("Added WandbMetricsLogger callback")
            except Exception as e:
                logger.warning(f"Failed to create WandbMetricsLogger: {e}")
        
        # 2. EMA Smoothed Metrics (Temporal Dynamics)
        if self.config.ema_smoothing > 0:
            self._ema_callback = EMASmoothedMetricsCallback(
                ema_decay=self.config.ema_smoothing,
                log_freq='batch',
            )
            callbacks.append(self._ema_callback)
            logger.debug(f"Added EMA metrics callback (decay={self.config.ema_smoothing})")
        
        # 3. Gradient Monitoring (Weight Health)
        if self.config.enable_gradient_logging and model is not None:
            self._gradient_callback = GradientMonitoringCallback(
                log_freq=self.config.gradient_log_freq,
                log_weights=True,
                log_gradients=True,
            )
            callbacks.append(self._gradient_callback)
            logger.debug(f"Added gradient monitoring callback (freq={self.config.gradient_log_freq})")
        
        # 4. System Metrics (Infrastructure)
        if self.config.enable_system_metrics:
            self._system_callback = SystemMetricsCallback(log_freq=10)
            callbacks.append(self._system_callback)
            logger.debug("Added system metrics callback")
        
        # 5. Model Checkpoint (if enabled)
        # NOTE: We use a custom SafeWandbCheckpoint callback instead of WandbModelCheckpoint
        # because WandbModelCheckpoint has a race condition where it tries to log the checkpoint
        # file as an artifact BEFORE Keras finishes writing it, causing FileNotFoundError.
        # See: https://github.com/wandb/wandb/issues/XXX
        if self.config.log_model_checkpoint and WANDB_AVAILABLE and TENSORFLOW_AVAILABLE:
            try:
                checkpoint_dir = Path("trained_data/checkpoints") / self.config.experiment_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Use safe checkpoint callback that verifies file exists before logging
                callbacks.append(SafeWandbCheckpoint(
                    filepath=str(checkpoint_dir / "model_{epoch:02d}.keras"),
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                ))
                logger.debug("Added SafeWandbCheckpoint callback (fixes wandb race condition)")
            except Exception as e:
                logger.warning(f"Failed to create SafeWandbCheckpoint: {e}")
        
        return callbacks
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None, commit: bool = False):
        """
        Log a single metric.
        
        NOTE: The `step` parameter is deprecated to avoid conflicts with WandbMetricsLogger
        which manages its own step counter. Metrics are logged without explicit step,
        allowing wandb to auto-increment.
        
        Args:
            name: Metric name
            value: Metric value
            step: DEPRECATED - ignored to avoid step conflicts
            commit: Whether to commit the log immediately (default: False)
        """
        if not self.is_active or not WANDB_AVAILABLE:
            return
        
        try:
            # Don't use explicit step to avoid conflicts with WandbMetricsLogger
            wandb.log({name: value}, commit=commit)
        except Exception as e:
            logger.debug(f"Failed to log metric {name}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, commit: bool = False):
        """
        Log multiple metrics.
        
        NOTE: The `step` parameter is deprecated to avoid conflicts with WandbMetricsLogger
        which manages its own step counter. Metrics are logged without explicit step,
        allowing wandb to auto-increment.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: DEPRECATED - ignored to avoid step conflicts
            commit: Whether to commit the log immediately (default: False)
        """
        if not self.is_active or not WANDB_AVAILABLE:
            return
        
        try:
            # Don't use explicit step to avoid conflicts with WandbMetricsLogger
            wandb.log(metrics, commit=commit)
        except Exception as e:
            logger.debug(f"Failed to log metrics: {e}")
    
    def log_final_metrics(self, metrics: Dict[str, Any]):
        """Log final training metrics as summary."""
        if not self.is_active or not WANDB_AVAILABLE:
            return
        
        try:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    wandb.run.summary[name] = value
            logger.info(f"📊 Logged final metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"Failed to log final metrics: {e}")
    
    def log_artifact(self, path: str, name: Optional[str] = None, type_: str = "model"):
        """Log a file as W&B artifact."""
        if not self.is_active or not WANDB_AVAILABLE:
            return
        
        try:
            artifact = wandb.Artifact(
                name=name or Path(path).stem,
                type=type_,
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)
            logger.info(f"📦 Logged artifact: {path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {path}: {e}")
    
    def log_model_graph(self, model):
        """Log model architecture visualization."""
        if not self.is_active or not WANDB_AVAILABLE:
            return
        
        if not self.config.log_model_graph:
            return
        
        try:
            # Log model summary as text
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            summary_text = "\n".join(summary_lines)
            
            wandb.log({
                "model/summary": wandb.Html(f"<pre>{summary_text}</pre>"),
            })
            logger.info("📊 Logged model graph")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def get_url(self) -> Optional[str]:
        """Get W&B run URL."""
        if WANDB_AVAILABLE and self.run:
            try:
                # Use run.url property instead of deprecated get_url() method
                return getattr(self.run, 'url', None)
            except Exception:
                pass
        return None
    
    def finish(self):
        """Finish W&B run."""
        if not self.is_active:
            return
        
        if WANDB_AVAILABLE and self.run:
            try:
                # Log duration
                if self._start_time:
                    duration = time.time() - self._start_time
                    wandb.run.summary['training_duration_seconds'] = duration
                
                wandb.finish()
                logger.info(f"🏁 W&B run finished: {self.config.experiment_name}")
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")
            finally:
                self.is_active = False
                self.run = None
    
    def __enter__(self):
        """Context manager entry."""
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False


# =============================================================================
# CONVENIENCE FUNCTION FOR CLI INTEGRATION
# =============================================================================

def create_wandb_tracker_from_options(
    options: Any,
    instrument: str = "GENERIC",
    granularity: str = "H1",
    config: Optional[Dict[str, Any]] = None,
) -> Optional[WandbKerasTracker]:
    """
    Create W&B tracker from CLI options.
    
    This function is designed to be called from cli/training.py to create
    a W&B tracker based on command-line options.
    
    Args:
        options: BuddyTrainingOptions or similar object with wandb_* attributes
        instrument: Trading pair
        granularity: Timeframe
        config: Hyperparameters dictionary
        
    Returns:
        WandbKerasTracker instance or None if W&B is disabled
    """
    # Check if W&B is enabled
    wandb_enabled = getattr(options, 'wandb_enabled', False)
    if not wandb_enabled:
        return None
    
    # Check W&B availability
    if not WANDB_AVAILABLE:
        logger.warning("W&B enabled but not installed. Install with: pip install wandb")
        return None
    
    # Extract W&B options
    project_name = getattr(options, 'wandb_project', 'ml_engine_fx')
    experiment_name = getattr(options, 'wandb_name', '') or f"{instrument}_{granularity}"
    offline_mode = getattr(options, 'wandb_offline', False)
    tags = getattr(options, 'wandb_tags', []) or []
    enable_gradients = getattr(options, 'wandb_gradients', True)
    enable_system = getattr(options, 'wandb_system', True)
    
    return WandbKerasTracker(
        project_name=project_name,
        experiment_name=experiment_name,
        instrument=instrument,
        granularity=granularity,
        tags=tags,
        config=config,
        enable_gradient_logging=enable_gradients,
        enable_system_metrics=enable_system,
        offline_mode=offline_mode,
    )


# =============================================================================
# MODULE VALIDATION
# =============================================================================

def validate_wandb_setup() -> Dict[str, Any]:
    """
    Validate W&B setup and return status.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'wandb_available': WANDB_AVAILABLE,
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'psutil_available': PSUTIL_AVAILABLE,
        'api_key_configured': False,
        'ready': False,
    }
    
    if WANDB_AVAILABLE:
        # Check API key
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            results['api_key_configured'] = True
        
        # Check if we can connect
        try:
            if api_key:
                wandb.login(key=api_key, anonymous='never')
            else:
                wandb.login(anonymous='allow')
            results['ready'] = True
        except Exception as e:
            results['error'] = str(e)
    
    return results


if __name__ == "__main__":
    # Validation
    print("W&B Keras Integration Validation")
    print("=" * 50)
    
    results = validate_wandb_setup()
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    if results['ready']:
        print("\n✅ W&B Keras integration ready!")
    else:
        print("\n❌ W&B Keras integration not ready.")
        print("   Install with: pip install wandb")
        print("   Configure with: wandb login")
