#!/usr/bin/env python3
raise SystemExit(
    "Retired: Buddy training/inference/trading is only supported via main.py. "
    "Use: python main.py train-buddy"
)

"""
Visual Training Demo with TensorBoard (Multi-Task Support).
Demonstrates TensorFlow training with real-time visualization.

Supports multi-task training with 5 heads:
- price_head: Next price prediction
- trend_head: Market direction/return
- direction_head: Binary up/down for prediction horizon
- risk_head: Volatility/risk level
- state_head: Market regime classification

Usage:
    python train_visual.py --framework tensorflow --model tft --multi-task
    python train_visual.py --framework tensorflow --model tcn --multi-task
    python train_visual.py --framework pytorch --model attention_lstm
    
    # Then open TensorBoard:
    tensorboard --logdir=trained_data/tensorboard
"""

import os
import argparse
import numpy as np
import pandas as pd
import yaml
import logging
import platform
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from typing import Tuple, List

# Configure logging to show info
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _suppress_irrelevant_tf_warnings_on_macos() -> None:
    """Hide TensorFlow warnings that are expected/irrelevant on Apple Metal."""
    if platform.system() != 'Darwin':
        return

    suppressed_substrings = (
        'will not use cuDNN kernels',
        'optimizer `tf.keras.optimizers.AdamW` runs slowly on M1/M2 Macs',
        'optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs',
        'Mixed precision compatibility check (mixed_float16): WARNING',
        'compute capability of at least 7.0',
        'METAL, no compute capability',
    )

    class _SubstringFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return not any(s in msg for s in suppressed_substrings)

    filt = _SubstringFilter()
    for logger_name in ('', 'tensorflow', 'absl', 'absl.logging'):
        logging.getLogger(logger_name).addFilter(filt)


_suppress_irrelevant_tf_warnings_on_macos()

# Constants
DEFAULT_DATA_DIR = 'trained_data/data'
DEFAULT_CHECKPOINT_PATH = 'trained_data/checkpoints/tensorflow/tf_model_best.keras'


@dataclass
class TrainingConfig:
    """Configuration for training functions to reduce parameter count."""
    model_type: str = 'tft'
    epochs: int = 50
    batch_size: int = 64
    multi_task: bool = False
    state_classes: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.0003
    patience: int = 20
    ensemble_size: int = 5
    base_seed: int = 42
    cyclic_lr: bool = True
    run_eagerly: Optional[bool] = None
    resume: bool = False
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH


def fetch_fresh_oanda_data(
    instruments: List[str] = None,
    granularity: str = 'M5',
    count: int = 5000,
    output_dir: str = DEFAULT_DATA_DIR,
) -> List[str]:
    """Fetch fresh data from OANDA before training.
    
    Args:
        instruments: List of instruments to fetch (default: USD_JPY, EUR_USD, GBP_USD)
        granularity: Candle granularity (default: M5)
        count: Number of candles to fetch (default: 5000)
        output_dir: Directory to save CSV files
        
    Returns:
        List of paths to saved CSV files
    """
    from oanda_practice import OandaPracticeClient
    
    if instruments is None:
        instruments = ['USD_JPY', 'EUR_USD', 'GBP_USD']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    client = OandaPracticeClient.from_env()
    saved_files = []
    
    print("\n🔄 Fetching fresh OANDA data...")
    
    for inst in instruments:
        print(f"   Fetching {inst}...", end=' ')
        try:
            resp = client.get_candles(inst, granularity=granularity, count=count, price='MBA')
            
            candles = resp.get('candles', [])
            rows = []
            for c in candles:
                if not c.get('complete', False):
                    continue
                mid = c.get('mid', {})
                bid = c.get('bid', {})
                ask = c.get('ask', {})
                rows.append({
                    'time': c['time'],
                    'open': float(mid.get('o', 0)),
                    'high': float(mid.get('h', 0)),
                    'low': float(mid.get('l', 0)),
                    'close': float(mid.get('c', 0)),
                    'volume': int(c.get('volume', 0)),
                    'bid_close': float(bid.get('c', 0)),
                    'ask_close': float(ask.get('c', 0)),
                })
            
            df = pd.DataFrame(rows)
            out_file = output_path / f'oanda_{inst}_{granularity}.csv'
            df.to_csv(out_file, index=False)
            saved_files.append(str(out_file))
            
            # Show date range
            start_time = df['time'].iloc[0][:10] if len(df) > 0 else 'N/A'
            end_time = df['time'].iloc[-1][:16] if len(df) > 0 else 'N/A'
            print(f"✅ {len(df)} candles ({start_time} to {end_time})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"   📁 Saved {len(saved_files)} files to {output_dir}/")
    return saved_files


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _dataset_to_numpy(ds, num_samples: int, batch_size: int) -> Tuple[np.ndarray, dict]:
    """Convert tf.data.Dataset to numpy arrays."""
    x_list, y_dict_lists = [], {'price': [], 'trend': [], 'direction': [], 'risk': [], 'state_logits': []}
    
    for batch_x, batch_y in ds.take((num_samples // batch_size) + 1):
        x_list.append(batch_x.numpy())
        for key in y_dict_lists:
            y_dict_lists[key].append(batch_y[key].numpy())
    
    x_arr = np.concatenate(x_list, axis=0)[:num_samples]
    y_dict = {key: np.concatenate(vals, axis=0)[:num_samples] for key, vals in y_dict_lists.items()}
    
    return x_arr, y_dict


def _load_multi_instrument_data(
    csv_files: list,
    sequence_length: int,
    state_classes: int,
    config: dict,
    load_multi_instrument_data_fn,
) -> Tuple[np.ndarray, dict, np.ndarray, dict, np.ndarray, dict, dict]:
    """Load and process multi-instrument data."""
    print(f"   📊 MULTI-INSTRUMENT MODE: {len(csv_files)} files found")
    for f in csv_files:
        print(f"      - {Path(f).name}")
    
    pipeline_config = {
        'sequence_length': sequence_length,
        'target_shift': 1,
        'risk_window': 14,
        'enable_data_scaling': True,
    }
    if config:
        pipeline_config.update(config)
    
    x_train, y_train, x_val, y_val, x_test, y_test, metadata = load_multi_instrument_data_fn(
        csv_paths=csv_files,
        config=pipeline_config,
        state_classes=state_classes,
        validation_split=0.15,
        add_all_features=True,
    )
    
    _print_multi_instrument_stats(x_train, x_val, x_test, y_train, metadata, state_classes)
    return x_train, y_train, x_val, y_val, x_test, y_test, metadata


def _print_multi_instrument_stats(x_train, x_val, x_test, y_train, metadata, state_classes):
    """Print statistics for multi-instrument data."""
    print("\n✓ Multi-instrument data loaded:")
    print(f"   Instruments: {metadata.get('instruments', [])}")
    print(f"   Train: {x_train.shape[0]:,} samples, {x_train.shape[-1]} features")
    print(f"   Val:   {x_val.shape[0]:,} samples")
    print(f"   Test:  {x_test.shape[0]:,} samples")
    print(f"   State classes: {state_classes}")
    
    if 'direction' in y_train:
        up_pct = 100 * np.mean(y_train['direction'])
        print(f"   Direction: UP={up_pct:.1f}% | DOWN={100-up_pct:.1f}%")


def _load_single_file_data(
    data_file: str,
    data_dir: str,
    sequence_length: int,
    state_classes: int,
    config: dict,
    pipeline_class,
) -> Tuple[np.ndarray, dict, np.ndarray, dict, np.ndarray, dict, dict]:
    """Load data from single file or directory using pipeline."""
    if data_file:
        print(f"   File: {data_file}")
    elif data_dir:
        print(f"   Directory: {data_dir}")
    
    batch_size = config.get('batch_size', 64) if config else 64
    pipeline_config = {
        'data_dir': data_dir or '.',
        'sequence_length': sequence_length,
        'price_col': 'close',
        'state_classes': state_classes,
        'volatility_window': 20,
        'train_split': 0.7,
        'val_split': 0.15,
        'batch_size': batch_size,
    }
    
    pipeline = pipeline_class(pipeline_config)
    result = pipeline.load_multitask_data(data_file=data_file, data_dir=data_dir)
    
    if result is None:
        raise ValueError("Failed to load data from pipeline")
    
    train_ds, val_ds, test_ds, metadata = result
    
    n_train = metadata.get('n_train', 0)
    n_val = metadata.get('n_val', 0)
    n_test = metadata.get('n_test', 0)
    
    print("   Converting datasets to numpy...")
    
    x_train, y_train = _dataset_to_numpy(train_ds, n_train, batch_size)
    x_val, y_val = _dataset_to_numpy(val_ds, n_val, batch_size)
    x_test, y_test = _dataset_to_numpy(test_ds, n_test, batch_size)
    
    _print_single_file_stats(x_train, x_val, x_test, y_train, state_classes)
    _save_scalers(pipeline)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, metadata


def _print_single_file_stats(x_train, x_val, x_test, y_train, state_classes):
    """Print statistics for single-file data loading."""
    print("\n✓ Real data loaded:")
    print(f"   Train: {x_train.shape[0]:,} samples, {x_train.shape[-1]} features")
    print(f"   Val:   {x_val.shape[0]:,} samples")
    print(f"   Test:  {x_test.shape[0]:,} samples")
    print(f"   State classes: {state_classes}")
    
    if 'state_logits' in y_train:
        state_train = np.argmax(y_train['state_logits'], axis=1)
        state_counts = np.bincount(state_train, minlength=state_classes)
        print(f"   State distribution (train): {state_counts}")


def _save_scalers(pipeline):
    """Save scalers from pipeline for later use."""
    scaler_path = 'trained_data/scalers'
    os.makedirs(scaler_path, exist_ok=True)
    pipeline.save_scalers(scaler_path)
    print(f"   Scalers saved to: {scaler_path}")


def load_real_data(
    data_file: str = None,
    data_dir: str = None,
    sequence_length: int = 60,
    state_classes: int = 3,
    config: dict = None,
    multi_instrument: bool = True,
) -> Tuple[np.ndarray, dict, np.ndarray, dict, np.ndarray, dict, dict]:
    """Load real market data using the TensorFlow data pipeline.
    
    This bridges the PyTorch data processing utilities (prepare_sequences, 
    build_multitask_targets, FeatureEngineering) to TensorFlow format.
    
    Supports multi-instrument training (Step 6) - combining ~15,000 samples from
    multiple FX pairs to improve generalization and reach 93%+ state accuracy.
    
    Args:
        data_file: Path to single CSV file (e.g., oanda_USD_JPY_M5.csv)
        data_dir: Directory containing multiple CSV files for multi-instrument training
        sequence_length: Sequence length for time series
        state_classes: Number of market state classes (3 for 93%+ accuracy like PyTorch)
        config: Configuration dict
        multi_instrument: If True and data_dir provided, load all CSVs and combine
    
    Returns:
        X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict, metadata
    """
    try:
        from tensorflow_data_pipeline import (
            TensorFlowDataPipeline, 
            load_multi_instrument_data,
            find_data_files,
        )
    except ImportError as e:
        print(f"❌ Error: Could not import TensorFlowDataPipeline: {e}")
        print("   Make sure tensorflow_data_pipeline.py is in the project root.")
        raise
    
    print("\n📁 Loading real data...")
    
    # Check if multi-instrument loading is requested
    if data_dir and multi_instrument:
        csv_files = find_data_files(data_dir)
        if len(csv_files) > 1:
            return _load_multi_instrument_data(
                csv_files, sequence_length, state_classes, config, load_multi_instrument_data
            )
    
    # Single file or fallback to original pipeline
    return _load_single_file_data(
        data_file, data_dir, sequence_length, state_classes, config, TensorFlowDataPipeline
    )


def generate_synthetic_data(
    n_samples: int = 5000,
    n_features: int = 104,
    sequence_length: int = 60,
    multi_task: bool = False,
    state_classes: int = 3,
) -> tuple:
    """Generate synthetic time series data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        sequence_length: Sequence length for LSTM/Transformer
        multi_task: If True, generate targets for all 4 heads
        state_classes: Number of market state classes (default 3)
    
    Returns:
        If multi_task=False: X_train, y_train, X_val, y_val, X_test, y_test
        If multi_task=True: X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict
    """
    print(f"Generating synthetic data: {n_samples} samples, {n_features} features...")
    if multi_task:
        print("  Multi-task mode: 4 heads (price, trend, risk, state)")
    
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    # Generate realistic-looking price data with trends and noise
    time = np.linspace(0, 100, n_samples)
    
    # Base price with multiple components
    trend = 0.01 * time
    seasonality = 5 * np.sin(2 * np.pi * time / 20)
    noise = np.cumsum(rng.standard_normal(n_samples) * 0.1)
    base_price = 100 + trend + seasonality + noise
    
    # Generate features (technical indicators simulation)
    features = []
    for i in range(n_features):
        # Mix of lagged price, moving averages, and oscillators
        if i < 5:  # OHLCV-like
            feat = base_price + rng.standard_normal(n_samples) * 0.5
        elif i < 20:  # Moving average-like
            window = 5 + i
            feat = np.convolve(base_price, np.ones(window)/window, mode='same')
        elif i < 40:  # Oscillators
            feat = np.sin(2 * np.pi * time / (5 + i)) + rng.standard_normal(n_samples) * 0.1
        else:  # Random technical indicators
            feat = rng.standard_normal(n_samples)
        
        features.append(feat)
    
    features = np.array(features).T  # [samples, features]
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Create sequences and targets
    X, y_price, y_trend, y_risk, y_state = [], [], [], [], []
    
    # Pre-compute returns and volatility
    returns = np.zeros_like(base_price)
    returns[1:] = (base_price[1:] / np.clip(base_price[:-1], 1e-12, None)) - 1.0
    
    # Rolling volatility (risk proxy)
    risk_window = 14
    volatility = np.zeros_like(returns)
    for i in range(len(returns)):
        start = max(0, i - risk_window + 1)
        volatility[i] = np.std(returns[start:i+1])
    
    # Normalize volatility to [0, 1] for risk target
    vol_min, vol_max = volatility.min(), volatility.max()
    if vol_max > vol_min:
        volatility_norm = (volatility - vol_min) / (vol_max - vol_min)
    else:
        volatility_norm = np.zeros_like(volatility)
    
    # Create state labels (volatility regime quantiles)
    quantiles = np.linspace(0, 1, state_classes + 1)
    edges = np.quantile(volatility, quantiles)
    state_labels = np.digitize(volatility, edges[1:-1], right=False).astype(np.int64)
    
    for i in range(sequence_length, len(features)):
        X.append(features[i - sequence_length:i])
        
        # Price target: next price change (normalized)
        y_price.append((base_price[i] - base_price[i-1]) / base_price[i-1])
        
        # Trend target: return at target time
        y_trend.append(returns[i])
        
        # Risk target: normalized volatility
        y_risk.append(volatility_norm[i])
        
        # State target: volatility regime class
        y_state.append(state_labels[i])
    
    X = np.array(X, dtype=np.float32)
    y_price = np.array(y_price, dtype=np.float32)
    y_trend = np.array(y_trend, dtype=np.float32)
    y_risk = np.array(y_risk, dtype=np.float32)
    y_state = np.array(y_state, dtype=np.int64)
    
    # Split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    x_train = X[:train_size]
    x_val = X[train_size:train_size+val_size]
    x_test = X[train_size+val_size:]
    
    print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    if multi_task:
        # Compute direction labels (1 = up, 0 = down)
        # Direction is based on whether trend is positive or negative
        y_direction = (y_trend > 0).astype(np.float32)
        
        # Return dict targets for multi-task training
        y_train = {
            'price': y_price[:train_size],
            'trend': y_trend[:train_size],
            'direction': y_direction[:train_size],  # Binary up/down target
            'risk': y_risk[:train_size],
            'state_logits': np.eye(state_classes, dtype=np.float32)[y_state[:train_size]],  # One-hot
        }
        y_val = {
            'price': y_price[train_size:train_size+val_size],
            'trend': y_trend[train_size:train_size+val_size],
            'direction': y_direction[train_size:train_size+val_size],
            'risk': y_risk[train_size:train_size+val_size],
            'state_logits': np.eye(state_classes, dtype=np.float32)[y_state[train_size:train_size+val_size]],
        }
        y_test = {
            'price': y_price[train_size+val_size:],
            'trend': y_trend[train_size+val_size:],
            'direction': y_direction[train_size+val_size:],
            'risk': y_risk[train_size+val_size:],
            'state_logits': np.eye(state_classes, dtype=np.float32)[y_state[train_size+val_size:]],
        }
        
        # Print state and direction distribution
        print(f"  State distribution (train): {np.bincount(y_state[:train_size], minlength=state_classes)}")
        up_count = int(y_direction[:train_size].sum())
        down_count = train_size - up_count
        print(f"  Direction distribution (train): Up={up_count}, Down={down_count} ({100*up_count/train_size:.1f}% up)")
    else:
        # Single-task: just price
        y_train = y_price[:train_size]
        y_val = y_price[train_size:train_size+val_size]
        y_test = y_price[train_size+val_size:]
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def _compute_ensemble_predictions(all_predictions, y_val, multi_task):
    """Compute and evaluate ensemble predictions."""
    print(f"\n{'='*60}")
    print("📊 ENSEMBLE RESULTS")
    print(f"{'='*60}")
    
    if not (multi_task and isinstance(all_predictions[0], dict)):
        return None
    
    # Average each head's predictions
    ensemble_preds = {}
    for key in all_predictions[0].keys():
        preds_stack = np.stack([p[key] for p in all_predictions], axis=0)
        ensemble_preds[key] = np.mean(preds_stack, axis=0)
    
    _evaluate_ensemble_direction(ensemble_preds, all_predictions, y_val)
    return ensemble_preds


def _evaluate_ensemble_direction(ensemble_preds, all_predictions, y_val):
    """Evaluate ensemble directional accuracy."""
    if 'direction' not in ensemble_preds:
        return
    
    dir_preds = (ensemble_preds['direction'] > 0.5).astype(float).flatten()
    if not (isinstance(y_val, dict) and 'direction' in y_val):
        return
    
    dir_true = np.array(y_val['direction']).flatten()
    ensemble_dir_acc = np.mean(dir_preds == dir_true)
    
    # Compare with individual models
    individual_accs = []
    for preds in all_predictions:
        ind_preds = (preds['direction'] > 0.5).astype(float).flatten()
        individual_accs.append(np.mean(ind_preds == dir_true))
    
    print("\n  Direction Accuracy:")
    print(f"    Individual models: {[f'{a*100:.1f}%' for a in individual_accs]}")
    print(f"    Ensemble (avg):    {ensemble_dir_acc*100:.1f}%")
    improvement = (ensemble_dir_acc - np.mean(individual_accs)) * 100
    print(f"    Improvement:       +{improvement:.2f}%")


def _compute_cyclic_lr(training_config: TrainingConfig, model_index: int) -> float:
    """Compute learning rate for cyclic schedule."""
    if not training_config.cyclic_lr:
        return training_config.learning_rate
    
    # Cosine annealing: start high, decay, then repeat
    lr = training_config.learning_rate * (
        1 + np.cos(np.pi * model_index / training_config.ensemble_size)
    ) / 2
    lr = max(lr, training_config.learning_rate * 0.1)  # Minimum 10% of base LR
    print(f"  Cyclic LR: {lr:.6f}")
    return lr


def train_ensemble(
    x_train, y_train, x_val, y_val,
    training_config: TrainingConfig = None,
    config: dict = None,
):
    """
    Train multiple models with different seeds and combine predictions.
    
    Implements snapshot ensemble with optional cyclic learning rate.
    Improves directional accuracy by +0.5-1% through prediction averaging.
    
    Args:
        x_train, y_train, x_val, y_val: Training and validation data
        training_config: TrainingConfig instance with all training parameters
        config: Additional configuration dict
    
    Returns:
        List of trained engines, list of histories, ensemble predictions
    """
    import tensorflow as tf
    
    if training_config is None:
        training_config = TrainingConfig()
    
    print("\n" + "="*60)
    print(f"🎯 ENSEMBLE TRAINING ({training_config.ensemble_size} models)")
    print("="*60)
    print("  Strategy: Multi-seed averaging for +0.5-1% accuracy boost")
    cyclic_status = 'Enabled (cosine annealing)' if training_config.cyclic_lr else 'Disabled'
    print(f"  Cyclic LR: {cyclic_status}")
    
    engines = []
    histories = []
    all_predictions = []
    
    for i in range(training_config.ensemble_size):
        seed = training_config.base_seed + i * 7  # Different seed per model
        print(f"\n{'='*40}")
        print(f"📦 Training Model {i+1}/{training_config.ensemble_size} (seed={seed})")
        print(f"{'='*40}")
        
        # Set seeds for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Adjust learning rate for cyclic schedule
        lr = _compute_cyclic_lr(training_config, i)
        
        # Create a modified config for this model
        model_config = TrainingConfig(
            model_type=training_config.model_type,
            epochs=training_config.epochs,
            batch_size=training_config.batch_size,
            multi_task=training_config.multi_task,
            state_classes=training_config.state_classes,
            dropout=training_config.dropout,
            learning_rate=lr,
            patience=training_config.patience,
        )
        
        # Train single model
        engine, history = train_tensorflow(
            x_train, y_train, x_val, y_val,
            training_config=model_config,
            config=config,
        )
        
        engines.append(engine)
        histories.append(history)
        
        # Get validation predictions
        val_preds = engine.model.predict(x_val, verbose=0)
        all_predictions.append(val_preds)
    
    ensemble_preds = _compute_ensemble_predictions(
        all_predictions, y_val, training_config.multi_task
    )
    
    return engines, histories, ensemble_preds


def train_tensorflow(  # noqa: C901
    x_train, y_train, x_val, y_val,
    training_config: TrainingConfig = None,
    config: dict = None,
):
    """Train with TensorFlow and TensorBoard visualization.
    
    Args:
        x_train: Training features [samples, timesteps, features]
        y_train: Training targets - scalar or dict for multi-task
        x_val: Validation features
        y_val: Validation targets
        training_config: TrainingConfig instance with all training parameters
        config: Configuration dict from config.yaml
    """
    if training_config is None:
        training_config = TrainingConfig()
    
    print("\n" + "="*60)
    print("🧠 TensorFlow Training with TensorBoard")
    if training_config.multi_task:
        print("   Mode: Multi-Task (5 heads: price, trend, direction, risk, state)")
    print("="*60)
    
    import tensorflow as tf
    from tensorflow_engine import TensorFlowEngine

    # Optional tracing (OpenTelemetry). Safe no-op unless ML_ENGINE_TRACING=1.
    try:
        from tracing_setup import setup_tracing, span, gpu_probe_attributes, add_event

        tracing_active = setup_tracing()
    except Exception:
        tracing_active = False
        span = None  # type: ignore
        add_event = None  # type: ignore

        def gpu_probe_attributes():  # type: ignore
            return {}

    root_span_cm = None
    if tracing_active and span is not None:
        root_span_cm = span(
            "train_visual.train_tensorflow",
            attributes={
                "multi_task": bool(getattr(training_config, "multi_task", False)),
                "epochs": int(getattr(training_config, "epochs", 0) or 0),
                "batch_size": int(getattr(training_config, "batch_size", 0) or 0),
            },
        )

    from contextlib import nullcontext

    root_ctx = root_span_cm if root_span_cm is not None else nullcontext()

    with root_ctx:
        # GPU/Metal check (must happen before creating/compiling the model)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU detected: {gpus[0].name}")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        else:
            print("⚠ No GPU detected, using CPU")

        if tracing_active and span is not None:
            with span("tensorflow.gpu_probe", attributes=gpu_probe_attributes()):
                if add_event is not None:
                    add_event("tensorflow.gpu_probe.complete")
        
        # Prepare config
        tf_config = _build_tf_config(x_train, training_config, config)

        # Default to mixed precision when a GPU is available.
        # On macOS/Apple Metal, Keras emits a CUDA-centric compatibility warning for mixed_float16.
        # So we keep mixed precision opt-in on Darwin to avoid confusing users.
        if gpus and not bool(tf_config.get('mixed_precision', False)):
            if platform.system() != 'Darwin':
                tf_config['mixed_precision'] = True

        # IMPORTANT: train_visual builds the model before calling engine.train().
        # So any auto-tuning that depends on y_train must happen here, before build_model().
        if training_config.multi_task and isinstance(y_train, dict) and 'direction' in y_train:
            try:
                y_dir = np.array(y_train['direction']).reshape(-1)
                if y_dir.size > 0:
                    up_rate = float(np.mean(y_dir >= 0.5))
                    is_balanced = abs(up_rate - 0.5) <= 0.05
                    tf_config.setdefault('direction_loss', 'bce' if is_balanced else 'focal')
                    # Match tensorflow_engine.compute_class_weights(): alpha = n_down / total = 1 - up_rate
                    tf_config.setdefault('focal_alpha', float(1.0 - up_rate))
            except Exception:
                pass
        
        # Create engine and train
        if tracing_active and span is not None:
            with span("engine.init"):
                engine = TensorFlowEngine(tf_config)
            with span("engine.build_model"):
                engine.build_model()
        else:
            engine = TensorFlowEngine(tf_config)
            engine.build_model()
        
        # Build the model with a sample input to get parameter count
        sample_input = tf.zeros((1, x_train.shape[1], x_train.shape[2]))
        _ = engine.model(sample_input)  # This builds the model
        
        # Resume from checkpoint if requested
        _handle_checkpoint_resume(engine, x_val, y_val, training_config)

        print(f"\nModel: {type(engine.model).__name__}")
        print(f"Parameters: {engine.model.count_params():,}")
        
        if training_config.multi_task:
            print("\nOutput heads: price, trend, direction, risk, state_logits")
        
        if tracing_active and span is not None:
            with span("engine.train"):
                history = engine.train(x_train, y_train, x_val, y_val)
        else:
            history = engine.train(x_train, y_train, x_val, y_val)
        
        # Evaluate
        if tracing_active and span is not None:
            with span("engine.evaluate"):
                _print_validation_results(engine, x_val, y_val)
        else:
            _print_validation_results(engine, x_val, y_val)
        
        return engine, history


def _build_tf_config(x_train, training_config: TrainingConfig, config: dict) -> dict:
    """Build TensorFlow configuration dictionary."""
    tf_config = {
        'model': {
            'type': training_config.model_type,
            'input_size': x_train.shape[-1],
            'hidden_size': config.get('hidden_size', 64) if config else 64,
            'num_layers': config.get('num_layers', 3) if config else 3,
            'dropout': training_config.dropout,
            'num_heads': 4,
            'sequence_length': x_train.shape[1],
            'multi_task': training_config.multi_task,
            'state_classes': training_config.state_classes,
        },
        'optimizer': {
            'type': 'adamw',
            'learning_rate': training_config.learning_rate,
        },
        'training': {
            'epochs': training_config.epochs,
            'early_stopping_patience': training_config.patience,
            # Print a heartbeat every N batches so CPU runs don't look frozen.
            # (Set to 0 to disable.)
            'batch_progress_every': 50,
            # TFT can take a long time to trace/compile the first train step on CPU.
            # Eager mode avoids looking stuck at "Starting model.fit()".
            'run_eagerly': bool(training_config.run_eagerly) if training_config.run_eagerly is not None else False,
        },
        'batch_size': training_config.batch_size,
        'loss_type': config.get('loss_type', 'huber') if config else 'huber',
        # Default set below based on GPU availability + config.
        'mixed_precision': False,
        'paths': {
            'tensorboard_dir': 'trained_data/tensorboard',
            'checkpoint_dir': 'trained_data/checkpoints/tensorflow',
        },
        'tensorboard': {
            'histogram_freq': 1,
            'write_graph': True,
        },
    }

    # Apply TF performance toggles from config.
    tf_section = (config or {}).get('tensorflow', {}) or {}
    requested_mixed_precision = bool(tf_section.get('mixed_precision', (config or {}).get('mixed_precision', False)))
    requested_xla = bool(tf_section.get('xla_compile', False))
    tf_config['training']['jit_compile'] = requested_xla

    # Default to eager mode for CPU+TFT unless explicitly overridden.
    try:
        import tensorflow as tf
        if training_config.run_eagerly is None:
            has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
            model_type = (training_config.model_type or '').lower()
            if (not has_gpu) and model_type in ['tft', 'temporal_fusion_transformer']:
                tf_config['training']['run_eagerly'] = True
        else:
            has_gpu = len(tf.config.list_physical_devices('GPU')) > 0

        # Only enable mixed precision when a GPU is available.
        tf_config['mixed_precision'] = bool(has_gpu and requested_mixed_precision)
    except Exception:
        pass
    
    if training_config.multi_task:
        _add_multitask_loss_weights(tf_config, config)
    
    return tf_config


def _add_multitask_loss_weights(tf_config: dict, config: dict):
    """Add loss weights for multi-task learning."""
    loss_weights = config.get('unified_head_loss_weights', {}) if config else {}
    tf_config['unified_head_loss_weights'] = {
        'price': loss_weights.get('price', 5.0),
        'trend': loss_weights.get('trend', 1.0),
        'direction': loss_weights.get('direction', 1.0),
        'risk': loss_weights.get('risk', 1.0),
        'state_logits': loss_weights.get('state', 1.0),
    }
    print(f"  Loss weights: {tf_config['unified_head_loss_weights']}")


def _handle_checkpoint_resume(engine, x_val, y_val, training_config: TrainingConfig):
    """Handle checkpoint resume logic."""
    if not training_config.resume:
        return
    
    checkpoint_path = training_config.checkpoint_path
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠ Checkpoint not found: {checkpoint_path}")
        print("  Starting fresh training...")
        return
    
    print(f"\n🔄 RESUMING FROM CHECKPOINT: {checkpoint_path}")
    try:
        from tensorflow import keras
        loaded_model = keras.models.load_model(checkpoint_path)
        engine.model.set_weights(loaded_model.get_weights())
        print("✓ Weights loaded successfully from checkpoint")
        mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        print(f"  Checkpoint modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("  Evaluating checkpoint to get baseline val_loss...")
        eval_results = engine.model.evaluate(x_val, y_val, verbose=0, return_dict=True)
        best_val_loss = eval_results.get('loss', eval_results.get('val_loss'))
        print(f"  ✓ Previous best val_loss: {best_val_loss:.6f}")
        print(f"  → Only saving if new val_loss < {best_val_loss:.6f}")
        
        engine.config['resume_best_val_loss'] = best_val_loss
        
    except Exception as e:
        print(f"⚠ Could not load checkpoint: {e}")
        print("  Starting fresh training instead...")


def _print_validation_results(engine, x_val, y_val):
    """Print validation results."""
    results = engine.evaluate(x_val, y_val)
    print("\nValidation Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def train_pytorch(
    x_train, y_train, x_val, y_val,
    model_type: str = 'attention_lstm',
    epochs: int = 50,
    batch_size: int = 64,
    config: dict = None,
):
    """Train with PyTorch and TensorBoard visualization."""
    print("\n" + "="*60)
    print("🔥 PyTorch Training with TensorBoard")
    print("="*60)
    raise RuntimeError(
        "train_pytorch is retired because PyTorch was removed. Use the TensorFlow-only Buddy pipeline via main.py."
    )



def _create_argument_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description='Visual Training with TensorBoard (Multi-Task Support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-task training with REAL data (recommended for 90%+ accuracy)
  python train_visual.py --framework tensorflow --model tft --multi-task \\
      --data-file market_data/oanda_USD_JPY_M5.csv --state-classes 2
  
  # Multi-task training with synthetic data
  python train_visual.py --framework tensorflow --model tft --multi-task
  python train_visual.py --framework tensorflow --model tcn --multi-task
  
  # Single-task training (price only)
  python train_visual.py --framework tensorflow --model tft
  python train_visual.py --framework pytorch --model attention_lstm
  
After training, view results:
  tensorboard --logdir=trained_data/tensorboard
        """
    )
    
    _add_framework_args(parser)
    _add_training_args(parser)
    _add_data_args(parser)
    _add_ensemble_args(parser)
    _add_oanda_args(parser)
    
    return parser


def _add_framework_args(parser):
    """Add framework-related arguments."""
    parser.add_argument(
        '--framework', '-f',
        choices=['tensorflow', 'pytorch', 'tf', 'pt'],
        default='tensorflow',
        help='ML framework to use (default: tensorflow)'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['lstm', 'attention_lstm', 'transformer', 'tft', 'tcn'],
        default='tft',
        help='Model architecture (default: tft)'
    )


def _add_training_args(parser):
    """Add training-related arguments."""
    parser.add_argument(
        '--multi-task', '--mt',
        action='store_true',
        help='Enable multi-task learning with 5 heads (price, trend, direction, risk, state)'
    )
    parser.add_argument(
        '--state-classes',
        type=int,
        default=3,
        help='Number of market state classes (default: 3 for PyTorch-matching 93%+ accuracy)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate for regularization (default: 0.3)'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=0.0003,
        help='Initial learning rate (default: 0.0003)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    eager_group = parser.add_mutually_exclusive_group()
    eager_group.add_argument(
        '--run-eagerly',
        dest='run_eagerly',
        action='store_true',
        help='Force eager execution (can avoid long first-step tracing/compilation)'
    )
    eager_group.add_argument(
        '--no-run-eagerly',
        dest='run_eagerly',
        action='store_false',
        help='Force graph execution (disable eager execution)'
    )
    parser.set_defaults(run_eagerly=None)
    parser.add_argument(
        '--tensorboard', '-t',
        action='store_true',
        help='Launch TensorBoard after training'
    )


def _add_data_args(parser):
    """Add data-related arguments."""
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=5000,
        help='Number of synthetic samples (default: 5000)'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=104,
        help='Number of features (default: 104)'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to real data CSV file (e.g., market_data/oanda_USD_JPY_M5.csv). '
             'When provided, uses real data pipeline instead of synthetic data.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory containing multiple data CSV files. When provided with multiple files, '
             'enables MULTI-INSTRUMENT training (~15,000 samples) for better generalization.'
    )
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume training from best checkpoint (trained_data/checkpoints/tensorflow/tf_model_best.keras)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='trained_data/checkpoints/tensorflow/tf_model_best.keras',
        help='Path to checkpoint file for resume (default: trained_data/checkpoints/tensorflow/tf_model_best.keras)'
    )
    parser.add_argument(
        '--resume-lr',
        type=float,
        default=None,
        help='Learning rate to use when resuming (default: 1/10th of --lr for fine-tuning)'
    )


def _add_ensemble_args(parser):
    """Add ensemble-related arguments."""
    parser.add_argument(
        '--ensemble-size',
        type=int,
        default=1,
        help='Number of models to train with different seeds (default: 1, set 3-5 for +1%% accuracy)'
    )
    parser.add_argument(
        '--cyclic-lr',
        action='store_true',
        help='Use cyclic learning rate with cosine annealing (snapshot ensemble)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed for reproducibility (default: 42)'
    )


def _add_oanda_args(parser):
    """Add OANDA data fetching arguments."""
    parser.add_argument(
        '--fetch-data', '--fetch',
        action='store_true',
        help='Fetch fresh data from OANDA before training (recommended for up-to-date models)'
    )
    parser.add_argument(
        '--fetch-only',
        action='store_true',
        help='Fetch fresh OANDA data and exit (no training)'
    )
    parser.add_argument(
        '--instruments',
        type=str,
        default='USD_JPY,EUR_USD,GBP_USD',
        help='Comma-separated list of OANDA instruments to fetch (default: USD_JPY,EUR_USD,GBP_USD)'
    )
    parser.add_argument(
        '--candles',
        type=int,
        default=5000,
        help='Number of candles to fetch per instrument (default: 5000)'
    )


def _fetch_oanda_data_if_needed(args):
    """Fetch fresh OANDA data if requested."""
    should_fetch = args.fetch_data or getattr(args, 'fetch_only', False) or (args.data_dir and not args.data_file)
    if not should_fetch:
        return
    
    try:
        instruments = [s.strip() for s in args.instruments.split(',')]
        fetch_fresh_oanda_data(
            instruments=instruments,
            granularity='M5',
            count=args.candles,
            output_dir=args.data_dir or DEFAULT_DATA_DIR,
        )
        if not args.data_dir:
            args.data_dir = DEFAULT_DATA_DIR
    except Exception as e:
        print(f"\n⚠ Warning: Could not fetch fresh OANDA data: {e}")
        print("   Continuing with existing data files...")


def _validate_args(args):
    """Validate and adjust arguments as needed."""
    if args.multi_task and args.model not in ['tft', 'tcn']:
        print("\n⚠ Warning: Multi-task heads are only implemented for 'tft' and 'tcn' models.")
        print(f"  Requested model '{args.model}' will use single-task mode.")
        args.multi_task = False


def _load_training_data(args, config):
    """Load training data (real or synthetic)."""
    if args.data_file or args.data_dir:
        if not args.multi_task:
            print("\n⚠ Note: Real data pipeline requires --multi-task flag. Enabling automatically.")
            args.multi_task = True
        
        x_train, y_train, x_val, y_val, _, _, _ = load_real_data(
            data_file=args.data_file,
            data_dir=args.data_dir,
            sequence_length=60,
            state_classes=args.state_classes,
            config=config,
        )
    else:
        x_train, y_train, x_val, y_val, _, _ = generate_synthetic_data(
            n_samples=args.samples,
            n_features=args.features,
            multi_task=args.multi_task,
            state_classes=args.state_classes,
        )
    
    return x_train, y_train, x_val, y_val


def _train_tensorflow_model(args, x_train, y_train, x_val, y_val, config):
    """Train TensorFlow model (single or ensemble)."""
    if args.ensemble_size > 1:
        training_config = TrainingConfig(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            multi_task=args.multi_task,
            state_classes=args.state_classes,
            dropout=args.dropout,
            learning_rate=args.lr,
            patience=args.patience,
            ensemble_size=args.ensemble_size,
            base_seed=args.seed,
            cyclic_lr=args.cyclic_lr,
            run_eagerly=args.run_eagerly,
        )
        train_ensemble(x_train, y_train, x_val, y_val, training_config, config)
    else:
        effective_lr = _get_effective_learning_rate(args)
        training_config = TrainingConfig(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            multi_task=args.multi_task,
            state_classes=args.state_classes,
            dropout=args.dropout,
            learning_rate=effective_lr,
            patience=args.patience,
            run_eagerly=args.run_eagerly,
            resume=args.resume,
            checkpoint_path=args.checkpoint,
        )
        train_tensorflow(x_train, y_train, x_val, y_val, training_config, config)


def _get_effective_learning_rate(args) -> float:
    """Determine effective learning rate based on resume mode."""
    if not args.resume:
        return args.lr
    
    if args.resume_lr is not None:
        effective_lr = args.resume_lr
    else:
        effective_lr = args.lr / 10.0
    
    print(f"📝 Resume mode: using learning rate {effective_lr:.6f}")
    return effective_lr


def _train_pytorch_model(args, x_train, y_train, x_val, y_val, config):
    """Train PyTorch model."""
    if args.multi_task:
        print("\n⚠ Warning: Multi-task training is only supported with TensorFlow.")
        print("  Running single-task PyTorch training...")
        if isinstance(y_train, dict):
            y_train = y_train['price']
            y_val = y_val['price']
    
    train_pytorch(
        x_train, y_train, x_val, y_val,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        config=config,
    )


def _print_completion_message(args):
    """Print training completion message."""
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    if args.multi_task:
        print("\n📊 Multi-task model trained with 5 heads:")
        print("   - price_head: Price prediction")
        print("   - trend_head: Horizon return")
        print("   - direction_head: Up/Down classification")
        print("   - risk_head: Volatility/risk")
        print(f"   - state_head: Market regime ({args.state_classes} classes)")
    print("\n📊 View training visualization:")
    print("   tensorboard --logdir=trained_data/tensorboard")
    print("\n   Then open: http://localhost:6006")


def _launch_tensorboard_if_requested(args):
    """Launch TensorBoard if requested."""
    if not args.tensorboard:
        return
    
    print("\n🚀 Launching TensorBoard...")
    import subprocess
    import webbrowser
    import time
    
    import sys
    process = subprocess.Popen(
        [sys.executable, '-m', 'tensorboard', '--logdir', 'trained_data/tensorboard', '--port', '6006'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    time.sleep(3)
    webbrowser.open('http://localhost:6006')
    
    print("Press Ctrl+C to stop TensorBoard")
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()


def main():
    parser = _create_argument_parser()
    args = parser.parse_args()
    
    # Load config if exists
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
    
    # Fetch fresh data if needed
    _fetch_oanda_data_if_needed(args)

    if getattr(args, 'fetch_only', False):
        print("\n✓ Fetch complete (--fetch-only). Exiting without training.")
        return
    
    # Validate arguments
    _validate_args(args)
    
    # Load data
    x_train, y_train, x_val, y_val = _load_training_data(args, config)
    
    # Train based on framework
    framework = args.framework.lower()
    if framework in ['tensorflow', 'tf']:
        _train_tensorflow_model(args, x_train, y_train, x_val, y_val, config)
    else:
        _train_pytorch_model(args, x_train, y_train, x_val, y_val, config)
    
    # Print completion and launch TensorBoard
    _print_completion_message(args)
    _launch_tensorboard_if_requested(args)


if __name__ == '__main__':
    main()
