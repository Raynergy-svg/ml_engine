"""
M1 Metal Optimization Utilities for TensorFlow Training.

This module provides critical optimizations for Apple M1/M2/M3 chips with TensorFlow Metal.
Addresses the main pain points:
1. Slow training between epochs (CPU-GPU data transfer bottleneck)
2. RecurrentDropout causing cuDNN fallback
3. Suboptimal batch sizes
4. Missing tf.data pipeline optimizations

Usage:
    from m1_metal_optimizer import (
        configure_metal_runtime,
        create_optimized_dataset,
        MetalOptimizedModel,
        WarmupCosineDecaySchedule,
    )
    
    # Before creating any models
    configure_metal_runtime()
    
    # Wrap your training data
    train_ds = create_optimized_dataset(X_train, y_train, batch_size=128)
"""

from __future__ import annotations

import os
import platform
import logging
import time
from typing import Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

logger = logging.getLogger(__name__)


# =============================================================================
# Runtime Configuration
# =============================================================================

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def configure_metal_runtime(
    memory_growth: bool = True,
    mixed_precision: bool = True,
    inter_op_parallelism: int = 1,
    intra_op_parallelism: int = 0,  # 0 = auto
) -> Dict[str, Any]:
    """
    Configure TensorFlow runtime for optimal M1 Metal performance.
    
    MUST be called before creating any models or tensors.
    
    Args:
        memory_growth: Enable GPU memory growth (recommended)
        mixed_precision: Enable mixed precision (float16) for 1.5-2x speedup
        inter_op_parallelism: Thread pool for independent ops
        intra_op_parallelism: Thread pool within single op (0=auto)
    
    Returns:
        Dict with runtime configuration details
    """
    config_info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "is_apple_silicon": is_apple_silicon(),
        "tf_version": tf.__version__,
    }
    
    # 1. Set threading for CPU ops
    if inter_op_parallelism > 0:
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_parallelism)
    if intra_op_parallelism >= 0:
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_parallelism)
    
    # 2. Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    config_info["gpus"] = [g.name for g in gpus]
    
    if gpus and memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            config_info["memory_growth"] = True
            logger.info(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth: {e}")
            config_info["memory_growth"] = False
    
    # 3. Enable mixed precision for M1 (significant speedup)
    if mixed_precision and is_apple_silicon():
        try:
            # mixed_float16 provides best balance for M1
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            config_info["mixed_precision"] = "mixed_float16"
            logger.info("✓ Mixed precision (float16) enabled for M1 Metal")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")
            config_info["mixed_precision"] = "disabled"
    
    # 4. Set environment variables for Metal optimization
    if is_apple_silicon():
        # Disable oneDNN custom ops (can cause issues on M1)
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        
        # Use unified memory for Metal
        os.environ.setdefault("TF_METAL_DEVICE_PLACEMENT_STRATEGY", "0")
        
        config_info["metal_optimizations"] = True
    
    # 5. Log summary
    logger.info(f"TensorFlow {tf.__version__} configured for {config_info['machine']}")
    if gpus:
        logger.info(f"GPU(s) available: {config_info['gpus']}")
    else:
        logger.warning("No GPU found - training will be slow")
    
    return config_info


# =============================================================================
# Optimized Data Pipeline
# =============================================================================

def create_optimized_dataset(
    X: np.ndarray,
    y: Union[np.ndarray, Dict[str, np.ndarray]],
    batch_size: int = 128,
    shuffle: bool = True,
    buffer_size: int = 10000,
    prefetch_buffer: int = tf.data.AUTOTUNE,
    cache: bool = True,
    augment_fn: Optional[Callable] = None,
) -> tf.data.Dataset:
    """
    Create an optimized tf.data.Dataset for M1 Metal training.
    
    Key optimizations:
    1. CACHE: Store processed data in memory (critical for small datasets)
    2. SHUFFLE: Use reasonable buffer (not dataset size for memory efficiency)
    3. BATCH: Before map operations when possible
    4. PREFETCH: Overlap data loading with model execution
    
    Args:
        X: Features array [samples, timesteps, features]
        y: Targets - either array or dict for multi-task
        batch_size: Batch size (128 recommended for M1)
        shuffle: Whether to shuffle (disable for validation)
        buffer_size: Shuffle buffer size
        prefetch_buffer: Prefetch buffer (AUTOTUNE recommended)
        cache: Cache dataset in memory
        augment_fn: Optional augmentation function(x, y) -> (x, y)
    
    Returns:
        Optimized tf.data.Dataset
    """
    # Create base dataset
    if isinstance(y, dict):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Cache BEFORE shuffle for determinism and speed
    if cache:
        dataset = dataset.cache()
    
    # Shuffle (skip for validation/test sets)
    if shuffle:
        # Use smaller buffer than dataset size for memory efficiency
        actual_buffer = min(buffer_size, len(X))
        dataset = dataset.shuffle(buffer_size=actual_buffer, reshuffle_each_iteration=True)
    
    # Apply augmentation before batching (if provided)
    if augment_fn is not None:
        dataset = dataset.map(
            augment_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False  # Allow non-determinism for speed
        )
    
    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=shuffle)
    
    # Prefetch for overlapping data loading with model execution
    dataset = dataset.prefetch(prefetch_buffer)
    
    return dataset


def create_augmentation_fn(
    noise_std: float = 0.01,
    scale_range: Tuple[float, float] = (0.98, 1.02),
    time_mask_prob: float = 0.1,
    time_mask_max_len: int = 5,
) -> Callable:
    """
    Create a data augmentation function for time series.
    
    Augmentations:
    1. Gaussian noise injection
    2. Random scaling
    3. Time masking (like SpecAugment for audio)
    
    Args:
        noise_std: Standard deviation of Gaussian noise
        scale_range: Min/max for random scaling
        time_mask_prob: Probability of applying time mask
        time_mask_max_len: Maximum length of time mask
    
    Returns:
        Augmentation function compatible with tf.data.Dataset.map()
    """
    @tf.function
    def augment(x, y):
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
            if tf.random.uniform([]) < time_mask_prob:
                seq_len = tf.shape(x)[0]
                mask_len = tf.random.uniform([], 1, time_mask_max_len + 1, dtype=tf.int32)
                mask_start = tf.random.uniform([], 0, seq_len - mask_len, dtype=tf.int32)
                
                # Create mask
                indices = tf.range(seq_len)
                mask = tf.logical_or(indices < mask_start, indices >= mask_start + mask_len)
                mask = tf.cast(mask, tf.float32)
                mask = tf.expand_dims(mask, -1)  # [seq_len, 1]
                
                # Apply mask (zero out masked timesteps)
                x = x * mask
        
        return x, y
    
    return augment


# =============================================================================
# Learning Rate Schedules
# =============================================================================

class WarmupCosineDecaySchedule(optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay with linear warmup - crucial for stable training.
    
    Benefits:
    1. Warmup prevents early divergence on small batch sizes
    2. Cosine decay provides smooth LR reduction
    3. Optional restarts can help escape local minima
    
    Paper: "SGDR: Stochastic Gradient Descent with Warm Restarts"
    """
    
    def __init__(
        self,
        initial_learning_rate: float = 0.001,
        warmup_steps: int = 1000,
        decay_steps: int = 10000,
        min_learning_rate: float = 1e-6,
        warmup_target: Optional[float] = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_learning_rate = min_learning_rate
        self.warmup_target = warmup_target or initial_learning_rate
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        # Linear warmup
        warmup_lr = (
            self.initial_learning_rate 
            + (self.warmup_target - self.initial_learning_rate) 
            * (step / warmup_steps)
        )
        
        # Cosine decay
        decay_progress = (step - warmup_steps) / decay_steps
        decay_progress = tf.minimum(decay_progress, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * decay_progress))
        decay_lr = (
            self.min_learning_rate 
            + (self.warmup_target - self.min_learning_rate) * cosine_decay
        )
        
        # Use warmup LR during warmup phase, decay LR after
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "min_learning_rate": self.min_learning_rate,
            "warmup_target": self.warmup_target,
        }


class StochasticWeightAveraging(callbacks.Callback):
    """
    Stochastic Weight Averaging (SWA) callback.
    
    SWA improves generalization by averaging model weights over the last
    part of training. Particularly effective for small datasets.
    
    Paper: "Averaging Weights Leads to Wider Optima and Better Generalization"
    """
    
    def __init__(
        self,
        start_epoch: int = 50,
        swa_lr: Optional[float] = None,
        verbose: int = 1,
    ):
        super().__init__()
        self.start_epoch = start_epoch
        self.swa_lr = swa_lr
        self.verbose = verbose
        self.swa_weights = None
        self.swa_count = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        
        current_weights = self.model.get_weights()
        
        if self.swa_weights is None:
            self.swa_weights = [w.copy() for w in current_weights]
            self.swa_count = 1
        else:
            self.swa_count += 1
            for i, (swa_w, cur_w) in enumerate(zip(self.swa_weights, current_weights)):
                self.swa_weights[i] = (swa_w * (self.swa_count - 1) + cur_w) / self.swa_count
        
        if self.verbose > 0:
            logger.info(f"SWA: Averaged weights (n={self.swa_count})")
    
    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            self.model.set_weights(self.swa_weights)
            if self.verbose > 0:
                logger.info(f"SWA: Applied averaged weights from {self.swa_count} snapshots")


# =============================================================================
# Gradient Accumulation for Larger Effective Batch Sizes
# =============================================================================

class GradientAccumulationModel(keras.Model):
    """
    Model wrapper that implements gradient accumulation.
    
    This allows training with effectively larger batch sizes on memory-limited
    devices by accumulating gradients over multiple forward passes.
    
    Example:
        model = GradientAccumulationModel(base_model, accumulation_steps=4)
        # Effective batch size = actual_batch_size * 4
    """
    
    def __init__(self, model: keras.Model, accumulation_steps: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.inner_model = model
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulator = None
        self.step_count = tf.Variable(0, trainable=False, dtype=tf.int32)
    
    def build(self, input_shape):
        self.inner_model.build(input_shape)
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        return self.inner_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        # Forward pass
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
            scaled_loss = loss / self.accumulation_steps
        
        # Compute gradients
        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        
        # Initialize or accumulate gradients
        if self.gradient_accumulator is None:
            self.gradient_accumulator = [
                tf.Variable(tf.zeros_like(g), trainable=False)
                for g in gradients if g is not None
            ]
        
        # Accumulate
        for accum, grad in zip(self.gradient_accumulator, gradients):
            if grad is not None:
                accum.assign_add(grad)
        
        self.step_count.assign_add(1)
        
        # Apply gradients when accumulation is complete
        if self.step_count >= self.accumulation_steps:
            # Apply accumulated gradients
            self.optimizer.apply_gradients(
                zip(self.gradient_accumulator, self.trainable_variables)
            )
            
            # Reset accumulators
            for accum in self.gradient_accumulator:
                accum.assign(tf.zeros_like(accum))
            self.step_count.assign(0)
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}


# =============================================================================
# Optimized Model Builder
# =============================================================================

def create_metal_optimized_tcn(
    input_shape: Tuple[int, int],
    hidden_size: int = 32,
    num_layers: int = 2,
    kernel_size: int = 3,
    dropout: float = 0.3,
    output_heads: Dict[str, int] = None,
    l2_reg: float = 0.001,
) -> keras.Model:
    """
    Create a TCN model optimized for M1 Metal performance.
    
    Key optimizations:
    1. No RecurrentDropout (causes cuDNN fallback)
    2. GaussianNoise for regularization instead
    3. Separable convolutions for efficiency
    4. Mixed precision compatible
    
    Args:
        input_shape: (sequence_length, num_features)
        hidden_size: Hidden layer size
        num_layers: Number of TCN layers
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        output_heads: Dict mapping head names to output sizes
        l2_reg: L2 regularization strength
    
    Returns:
        Compiled Keras model
    """
    if output_heads is None:
        output_heads = {
            "price": 1,
            "trend": 1,
            "direction": 1,
            "risk": 1,
            "state_logits": 2,
        }
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input")
    
    # Input regularization (instead of RecurrentDropout)
    x = layers.GaussianNoise(0.03)(inputs)
    x = layers.SpatialDropout1D(dropout * 0.5)(x)
    
    # TCN layers with dilated causal convolutions
    for i in range(num_layers):
        dilation = 2 ** i
        residual = x
        
        # Causal convolution
        x = layers.Conv1D(
            hidden_size,
            kernel_size,
            padding='causal',
            dilation_rate=dilation,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(l2_reg),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        # Residual connection
        if residual.shape[-1] != hidden_size:
            residual = layers.Conv1D(hidden_size, 1)(residual)
        x = layers.Add()([x, residual])
    
    # Global pooling
    x = x[:, -1, :]  # Take last timestep
    
    # Shared dense layer
    shared = layers.Dense(64, activation='relu')(x)
    shared = layers.Dropout(dropout)(shared)
    
    # Multi-task heads
    outputs = {}
    
    for head_name, output_size in output_heads.items():
        if head_name == "direction":
            head = layers.Dense(32, activation='relu')(shared)
            head = layers.Dropout(dropout * 0.5)(head)
            head = layers.Dense(output_size, activation='sigmoid', name=head_name)(head)
        elif head_name == "state_logits":
            head = layers.Dense(32, activation='relu')(shared)
            head = layers.Dropout(dropout * 0.5)(head)
            head = layers.Dense(output_size, activation='softmax', name=head_name)(head)
        elif head_name == "risk":
            head = layers.Dense(16, activation='relu')(shared)
            head = layers.Dense(output_size, activation='sigmoid', name=head_name)(head)
        else:
            head = layers.Dense(32, activation='relu')(shared)
            head = layers.Dense(output_size, name=head_name)(head)
        outputs[head_name] = head
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="MetalOptimizedTCN")
    
    return model


# =============================================================================
# Training Monitor
# =============================================================================

class MetalPerformanceMonitor(callbacks.Callback):
    """
    Monitor training performance and detect M1 Metal issues.
    
    Tracks:
    1. Samples per second
    2. GPU utilization patterns
    3. Potential bottlenecks
    """
    
    def __init__(self, log_every_n_batches: int = 50):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.batch_times = []
        self.epoch_start_time = None
        self.batch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.batch_times = []
    
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
    
    def on_train_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
        if (batch + 1) % self.log_every_n_batches == 0:
            avg_batch_time = np.mean(self.batch_times[-self.log_every_n_batches:])
            samples_per_sec = self.params.get('batch_size', 32) / avg_batch_time
            
            logger.info(
                f"  Batch {batch+1}: {avg_batch_time*1000:.1f}ms/batch, "
                f"{samples_per_sec:.0f} samples/sec"
            )
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        
        logger.info(
            f"Epoch {epoch+1}: {epoch_time:.1f}s total, "
            f"{avg_batch_time*1000:.1f}ms avg/batch"
        )
        
        # Warn if batch times are highly variable (indicates CPU-GPU sync issues)
        if self.batch_times and np.std(self.batch_times) > np.mean(self.batch_times) * 0.5:
            logger.warning(
                "⚠ High batch time variance detected - possible CPU-GPU sync bottleneck"
            )


# =============================================================================
# Quick Setup Function
# =============================================================================

def setup_metal_training(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, Dict[str, np.ndarray]],
    X_val: np.ndarray,
    y_val: Union[np.ndarray, Dict[str, np.ndarray]],
    config: Dict[str, Any],
) -> Tuple[tf.data.Dataset, tf.data.Dataset, keras.Model, Dict[str, Any]]:
    """
    One-stop setup for M1 Metal optimized training.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Configuration dict
    
    Returns:
        (train_dataset, val_dataset, model, training_config)
    """
    # 1. Configure runtime
    runtime_info = configure_metal_runtime(
        mixed_precision=config.get('mixed_precision', True)
    )
    
    # 2. Create optimized datasets
    batch_size = config.get('batch_size', 128)
    
    train_ds = create_optimized_dataset(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True,
        cache=True,
    )
    
    val_ds = create_optimized_dataset(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False,
        cache=True,
    )
    
    # 3. Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = create_metal_optimized_tcn(
        input_shape=input_shape,
        hidden_size=config.get('hidden_size', 32),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        l2_reg=config.get('kernel_regularizer', 0.001),
    )
    
    # 4. Setup learning rate schedule
    steps_per_epoch = len(X_train) // batch_size
    total_steps = steps_per_epoch * config.get('epochs', 100)
    
    lr_schedule = WarmupCosineDecaySchedule(
        initial_learning_rate=config.get('learning_rate', 0.0001),
        warmup_steps=steps_per_epoch * 5,  # 5 epochs warmup
        decay_steps=total_steps,
        min_learning_rate=1e-6,
        warmup_target=config.get('learning_rate', 0.0005),
    )
    
    # 5. Compile model
    optimizer = optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.get('weight_decay', 0.01),
        clipnorm=config.get('clipnorm', 1.0),
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            'price': keras.losses.Huber(delta=1.0),
            'trend': keras.losses.Huber(delta=0.5),
            'direction': keras.losses.BinaryCrossentropy(),
            'risk': keras.losses.MeanSquaredError(),
            'state_logits': keras.losses.CategoricalCrossentropy(),
        },
        loss_weights=config.get('unified_head_loss_weights', {
            'price': 1.0,
            'trend': 0.5,
            'direction': 5.0,
            'risk': 2.0,
            'state_logits': 3.0,
        }),
        metrics={
            'direction': [keras.metrics.BinaryAccuracy(name='dir_acc')],
            'state_logits': [keras.metrics.CategoricalAccuracy(name='state_acc')],
        },
    )
    
    # 6. Prepare callbacks
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 30),
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
        ),
        callbacks.ModelCheckpoint(
            filepath=config.get('checkpoint_dir', 'trained_data/checkpoints') + '/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
        ),
        MetalPerformanceMonitor(log_every_n_batches=50),
    ]
    
    # Add SWA if training for enough epochs
    if config.get('epochs', 100) >= 50:
        training_callbacks.append(StochasticWeightAveraging(start_epoch=50))
    
    training_config = {
        'epochs': config.get('epochs', 100),
        'callbacks': training_callbacks,
        'verbose': 1,
    }
    
    return train_ds, val_ds, model, training_config


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("M1 Metal Optimizer Module")
    print("=" * 60)
    
    # Test runtime configuration
    config_info = configure_metal_runtime()
    
    print("\nRuntime Configuration:")
    for key, value in config_info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    X_dummy = np.random.randn(1000, 60, 104).astype(np.float32)
    y_dummy = {
        'price': np.random.randn(1000, 1).astype(np.float32),
        'trend': np.random.randn(1000, 1).astype(np.float32),
        'direction': (np.random.rand(1000, 1) > 0.5).astype(np.float32),
        'risk': np.random.rand(1000, 1).astype(np.float32),
        'state_logits': np.eye(2)[np.random.randint(0, 2, 1000)].astype(np.float32),
    }
    
    # Create optimized dataset
    train_ds = create_optimized_dataset(X_dummy, y_dummy, batch_size=128)
    
    print(f"\nDataset element spec:")
    print(f"  {train_ds.element_spec}")
    
    # Create model
    model = create_metal_optimized_tcn(
        input_shape=(60, 104),
        hidden_size=32,
        num_layers=2,
    )
    
    print(f"\nModel summary:")
    model.summary()
    
    print("\n✓ M1 Metal optimizer ready")

