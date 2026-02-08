#!/usr/bin/env python3
"""
Optimized Training Script for M1 Metal.

This script provides a complete training pipeline optimized for Apple M1/M2/M3 chips
with TensorFlow Metal acceleration.

Usage:
    # Quick training with defaults
    python train_optimized_m1.py

    # With custom data file
    python train_optimized_m1.py --data market_data/EUR_USD_M5_5000.csv

    # Multi-instrument training
    python train_optimized_m1.py --multi-instrument

    # With walk-forward validation
    python train_optimized_m1.py --walkforward

Features:
- M1 Metal optimizations (2-3x faster training)
- Proper data pipeline with caching and prefetching
- Walk-forward time-series validation
- Automatic class balancing for direction prediction
- TensorBoard integration
- Model checkpointing with best weights
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import yaml

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Configure environment before TensorFlow import."""
    # Suppress TensorFlow warnings
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    # Disable oneDNN for M1 (can cause issues)
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')


# Setup environment before importing TensorFlow
setup_environment()

from tensorflow import keras  # noqa: E402
from tensorflow.keras import callbacks  # noqa: E402

# Local imports
try:
    from m1_metal_optimizer import (
        configure_metal_runtime,
        create_optimized_dataset,
        create_metal_optimized_tcn,
        WarmupCosineDecaySchedule,
        StochasticWeightAveraging,
        MetalPerformanceMonitor,
    )
except ImportError:
    logger.error("m1_metal_optimizer.py not found. Run from ml_engine directory.")
    sys.exit(1)

try:
    from tensorflow_data_pipeline import (
        load_tensorflow_multitask_data,
        load_multi_instrument_data,
        find_data_files,
    )
except ImportError:
    logger.error("tensorflow_data_pipeline.py not found.")
    sys.exit(1)

try:
    from walkforward_validation import (
        WalkForwardValidator,
        calculate_trading_metrics,
    )
except ImportError:
    logger.warning("walkforward_validation.py not found. Walk-forward disabled.")
    WalkForwardValidator = None


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    'sequence_length': 60,
    'target_shift': 1,
    'risk_window': 14,
    'state_classes': 2,
    'enable_data_scaling': True,

    # Model config
    'hidden_size': 32,
    'num_layers': 2,
    'dropout': 0.35,
    'kernel_regularizer': 0.002,

    # Training config
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.0005,
    'weight_decay': 0.01,
    'early_stopping_patience': 30,
    'min_epochs': 20,

    # Loss weights (balanced for multi-task)
    'unified_head_loss_weights': {
        'price': 1.0,
        'trend': 0.5,
        'direction': 5.0,
        'risk': 2.0,
        'state_logits': 3.0,
    },

    # M1 optimizations
    'mixed_precision': True,
    'cache_dataset': True,

    # Paths
    'checkpoint_dir': 'trained_data/checkpoints',
    'tensorboard_dir': 'trained_data/tensorboard',
    'model_dir': 'trained_data/models',
}


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Could not load {config_path}: {e}")
        return {}


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge two config dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# Model Building
# =============================================================================

def create_model(
    input_shape: Tuple[int, int],
    config: Dict[str, Any],
) -> keras.Model:
    """Create the optimized TCN model."""
    model = create_metal_optimized_tcn(
        input_shape=input_shape,
        hidden_size=config.get('hidden_size', 32),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.35),
        l2_reg=config.get('kernel_regularizer', 0.002),
        output_heads={
            'price': 1,
            'trend': 1,
            'direction': 1,
            'risk': 1,
            'state_logits': config.get('state_classes', 2),
        },
    )

    return model


def compile_model(
    model: keras.Model,
    config: Dict[str, Any],
    steps_per_epoch: int,
) -> keras.Model:
    """Compile model with optimized settings."""
    total_steps = steps_per_epoch * config.get('epochs', 100)
    warmup_steps = steps_per_epoch * 5  # 5 epochs warmup

    # Learning rate schedule with warmup
    lr_schedule = WarmupCosineDecaySchedule(
        initial_learning_rate=config.get('learning_rate', 0.0005) * 0.1,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        min_learning_rate=1e-7,
        warmup_target=config.get('learning_rate', 0.0005),
    )

    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.get('weight_decay', 0.01),
        clipnorm=1.0,
    )

    # Loss weights
    loss_weights = config.get('unified_head_loss_weights', DEFAULT_CONFIG['unified_head_loss_weights'])

    # Compile - Use Huber for risk to handle small values better than MSE
    model.compile(
        optimizer=optimizer,
        loss={
            'price': keras.losses.Huber(delta=1.0),
            'trend': keras.losses.Huber(delta=0.5),
            'direction': keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            'risk': keras.losses.Huber(delta=0.01),  # Use Huber for stability with small values
            'state_logits': keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        },
        loss_weights=loss_weights,
        metrics={
            'direction': [keras.metrics.BinaryAccuracy(name='dir_acc')],
            'state_logits': [keras.metrics.CategoricalAccuracy(name='state_acc')],
        },
    )

    return model


def create_callbacks(config: Dict[str, Any], run_name: str) -> list:
    """Create training callbacks."""
    callback_list = []

    # Checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', 'trained_data/checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    tb_dir = Path(config.get('tensorboard_dir', 'trained_data/tensorboard'))
    tb_log_dir = tb_dir / run_name
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    callback_list.append(
        callbacks.TensorBoard(
            log_dir=str(tb_log_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0,
        )
    )

    # Early stopping (with min_epochs)
    min_epochs = config.get('min_epochs', 20)

    class MinEpochEarlyStopping(callbacks.EarlyStopping):
        def on_epoch_end(self, epoch, logs=None):
            if epoch < min_epochs:
                return
            super().on_epoch_end(epoch, logs)

    callback_list.append(
        MinEpochEarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 30),
            restore_best_weights=True,
            verbose=1,
        )
    )

    # Model checkpoint - primary (best overall loss)
    callback_list.append(
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        )
    )

    # Multi-metric checkpointing: save best per metric for flexibility
    if config.get('multi_checkpoint', True):
        # Best direction accuracy checkpoint
        callback_list.append(
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / 'best_direction.keras'),
                monitor='val_direction_dir_acc',
                mode='max',
                save_best_only=True,
                verbose=1,
            )
        )

        # Best price prediction loss
        callback_list.append(
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / 'best_price.keras'),
                monitor='val_price_loss',
                mode='min',
                save_best_only=True,
                verbose=1,
            )
        )

        # Best state classification accuracy
        callback_list.append(
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / 'best_state.keras'),
                monitor='val_state_logits_state_acc',
                mode='max',
                save_best_only=True,
                verbose=1,
            )
        )

    # Learning rate reduction
    callback_list.append(
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
        )
    )

    # Performance monitoring
    callback_list.append(MetalPerformanceMonitor(log_every_n_batches=50))

    # Stochastic Weight Averaging (for last 20% of training)
    swa_start = int(config.get('epochs', 100) * 0.8)
    callback_list.append(StochasticWeightAveraging(start_epoch=swa_start))

    # CSV logger
    csv_path = tb_log_dir / 'training_history.csv'
    callback_list.append(callbacks.CSVLogger(str(csv_path)))

    return callback_list


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    data_path: Optional[str],
    config: Dict[str, Any],
    multi_instrument: bool = False,
) -> Tuple[np.ndarray, Dict, np.ndarray, Dict, np.ndarray, Dict, Dict]:
    """Load and prepare training data."""

    if multi_instrument:
        # Find all CSV files in market_data
        csv_files = find_data_files('market_data', '*.csv')
        if not csv_files:
            logger.error("No CSV files found in market_data/")
            sys.exit(1)

        logger.info(f"Loading multi-instrument data from {len(csv_files)} files")

        # Limit to 3 instruments for reasonable training time
        csv_files = csv_files[:3]

        X_train, y_train, X_val, y_val, X_test, y_test, meta = load_multi_instrument_data(
            csv_files,
            config,
            state_classes=config.get('state_classes', 2),
            validation_split=0.2,
            add_all_features=True,
        )
    else:
        # Single instrument
        if data_path is None:
            # Find first available CSV
            csv_files = find_data_files('market_data', '*.csv')
            if not csv_files:
                logger.error("No CSV files found. Specify --data or add files to market_data/")
                sys.exit(1)
            data_path = csv_files[0]

        logger.info(f"Loading data from {data_path}")

        X_train, y_train, X_val, y_val, X_test, y_test, meta = load_tensorflow_multitask_data(
            data_path,
            config,
            state_classes=config.get('state_classes', 2),
            validation_split=0.2,
            add_all_features=True,
        )

    # Log data statistics
    logger.info("Data loaded:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")

    # Check class balance
    train_dir = y_train['direction'].flatten()
    up_pct = 100 * np.mean(train_dir)
    logger.info(f"  Direction balance: {up_pct:.1f}% up / {100-up_pct:.1f}% down")

    return X_train, y_train, X_val, y_val, X_test, y_test, meta


# =============================================================================
# Training
# =============================================================================

def train_model(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train the model with optimized settings."""

    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"m1_optimized_{timestamp}"

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Training Run: {run_name}")
    logger.info(f"{'='*60}")

    # Create optimized datasets
    batch_size = config.get('batch_size', 128)

    train_ds = create_optimized_dataset(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True,
        cache=config.get('cache_dataset', True),
    )

    val_ds = create_optimized_dataset(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False,
        cache=config.get('cache_dataset', True),
    )

    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_model(input_shape, config)

    # Compile
    steps_per_epoch = len(X_train) // batch_size
    model = compile_model(model, config, steps_per_epoch)

    # Print model summary
    logger.info("\nModel Summary:")
    model.summary(print_fn=logger.info)

    # Create callbacks
    training_callbacks = create_callbacks(config, run_name)

    # Train
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.get('epochs', 100),
        callbacks=training_callbacks,
        verbose=1,
    )

    train_time = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info("Training Complete")
    logger.info(f"  Total time: {train_time/60:.1f} minutes")
    logger.info(f"  Best val_loss: {min(history.history['val_loss']):.4f}")
    logger.info(f"{'='*60}")

    return model, history


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate model on test set."""

    logger.info("\n" + "="*60)
    logger.info("Evaluating on Test Set")
    logger.info("="*60)

    # Standard evaluation
    results = model.evaluate(X_test, y_test, verbose=1, return_dict=True)

    # Get predictions
    predictions = model.predict(X_test, verbose=0)

    # Calculate trading metrics
    if isinstance(predictions, dict) and 'direction' in predictions:
        direction_pred = predictions['direction'].flatten()
        direction_true = y_test['direction'].flatten()

        trading_metrics = calculate_trading_metrics(
            direction_true,
            direction_pred,
            prices=None,  # Would need price data for full metrics
        )

        results.update(trading_metrics)

    # Print results
    logger.info("\nTest Results:")
    for key, value in sorted(results.items()):
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")

    return results


def run_walkforward(
    data_path: str,
    config: Dict[str, Any],
    n_splits: int = 5,
) -> None:
    """Run walk-forward validation."""

    if WalkForwardValidator is None:
        logger.error("Walk-forward validation not available. Install dependencies.")
        return

    logger.info("\n" + "="*60)
    logger.info("Walk-Forward Validation")
    logger.info("="*60)

    # Load all data
    X_train, y_train, X_val, y_val, X_test, y_test, meta = load_data(
        data_path, config, multi_instrument=False
    )

    # Combine all data
    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = {
        k: np.concatenate([y_train[k], y_val[k], y_test[k]], axis=0)
        for k in y_train
    }

    # Walk-forward splits
    validator = WalkForwardValidator(
        n_splits=n_splits,
        train_size=0.6,
        val_size=0.1,
        test_size=0.1,
        gap=config.get('sequence_length', 60),
    )

    fold_results = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
        logger.info(f"\n{'='*40}")
        logger.info(f"Fold {fold + 1}/{n_splits}")
        logger.info(f"{'='*40}")

        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = {k: v[train_idx] for k, v in y.items()}
        X_val_fold = X[val_idx]
        y_val_fold = {k: v[val_idx] for k, v in y.items()}
        X_test_fold = X[test_idx]
        y_test_fold = {k: v[test_idx] for k, v in y.items()}

        # Train
        model, history = train_model(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            config,
        )

        # Evaluate
        results = evaluate_model(model, X_test_fold, y_test_fold, config)
        fold_results.append(results)

        # Clean up
        del model
        keras.backend.clear_session()

    # Aggregate results
    logger.info("\n" + "="*60)
    logger.info("Walk-Forward Summary")
    logger.info("="*60)

    # Compute mean and std for each metric
    all_metrics = {}
    for result in fold_results:
        for key, value in result.items():
            if key not in all_metrics:
                all_metrics[key] = []
            if isinstance(value, (int, float)):
                all_metrics[key].append(value)

    for key, values in sorted(all_metrics.items()):
        mean = np.mean(values)
        std = np.std(values)
        logger.info(f"  {key}: {mean:.4f} ± {std:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized M1 Metal Training for FX Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to CSV data file (default: auto-detect from market_data/)',
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config_m1_optimized.yaml',
        help='Path to config file (default: config_m1_optimized.yaml)',
    )

    parser.add_argument(
        '--multi-instrument',
        action='store_true',
        help='Train on multiple instruments for better generalization',
    )

    parser.add_argument(
        '--walkforward',
        action='store_true',
        help='Run walk-forward validation instead of single training',
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size',
    )

    parser.add_argument(
        '--no-mixed-precision',
        action='store_true',
        help='Disable mixed precision (for debugging)',
    )

    args = parser.parse_args()

    # Load configuration
    config = DEFAULT_CONFIG.copy()

    if Path(args.config).exists():
        file_config = load_yaml_config(args.config)
        config = merge_configs(config, file_config)

    # Apply command-line overrides
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.no_mixed_precision:
        config['mixed_precision'] = False

    # Configure M1 Metal runtime
    logger.info("\nConfiguring M1 Metal Runtime...")
    runtime_info = configure_metal_runtime(
        mixed_precision=config.get('mixed_precision', True)
    )

    for key, value in runtime_info.items():
        logger.info(f"  {key}: {value}")

    # Run training
    if args.walkforward:
        run_walkforward(args.data, config)
    else:
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test, meta = load_data(
            args.data, config, args.multi_instrument
        )

        # Train
        model, history = train_model(X_train, y_train, X_val, y_val, config)

        # Evaluate
        evaluate_model(model, X_test, y_test, config)

        # Save model
        model_dir = Path(config.get('model_dir', 'trained_data/models'))
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f'optimized_tcn_{timestamp}.keras'
        model.save(str(model_path))
        logger.info(f"\nModel saved to: {model_path}")

        # Print TensorBoard command
        tb_dir = config.get('tensorboard_dir', 'trained_data/tensorboard')
        logger.info("\nTo view training progress:")
        logger.info(f"  tensorboard --logdir {tb_dir}")


if __name__ == '__main__':
    main()
