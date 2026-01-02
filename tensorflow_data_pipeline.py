"""
TensorFlow Data Pipeline for Multi-Task Training.
Bridges existing PyTorch data processing with TensorFlow training.

This module provides:
- Integration with existing data_processing.py, feature_engineering.py, multitask_labels.py
- tf.data.Dataset creation from real market data
- Scaler persistence for inference consistency
- Same data pipeline that achieved 93%+ state accuracy in PyTorch
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from data_processing import prepare_sequences
from multitask_labels import build_multitask_targets, split_time_series, MultitaskTargets
from feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)


def load_market_dataframe(
    csv_path: str,
    *,
    add_all_features: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load market data from CSV and optionally add engineered features.
    
    Args:
        csv_path: Path to CSV file with OHLCV data
        add_all_features: If True, add all engineered features (RSI, MACD, etc.)
        config: Optional config dict for feature engineering
    
    Returns:
        DataFrame with normalized column names and optional features
    """
    df = pd.read_csv(csv_path)
    
    # Normalize columns to lowercase
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    
    if add_all_features:
        logger.info("Adding all engineered features...")
        fe = FeatureEngineering(config=config)
        df = fe.create_features(df, include_all=True)
        logger.info(f"Total features after engineering: {len(df.columns)}")
    
    return df


def load_tensorflow_multitask_data(
    csv_path: str,
    config: Dict[str, Any],
    *,
    state_classes: int = 2,
    validation_split: float = 0.2,
    add_all_features: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load real market data using same pipeline as PyTorch unified_multitask_training.
    
    This replicates the data pipeline that achieved 93%+ state classification accuracy.
    
    Args:
        csv_path: Path to CSV file with market data
        config: Configuration dictionary
        state_classes: Number of volatility regime classes (2 for binary like PyTorch)
        validation_split: Fraction for validation set (chronological split)
        add_all_features: Add engineered features (RSI, MACD, etc.)
    
    Returns:
        X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict, meta
        
        Where y_dict contains:
            - 'price': (n,) float32
            - 'trend': (n,) float32
            - 'risk': (n,) float32
            - 'state_logits': (n, state_classes) float32 one-hot
    """
    # Load and preprocess data
    df = load_market_dataframe(csv_path, add_all_features=add_all_features, config=config)
    
    # Get configuration values
    seq_len = int(config.get('sequence_length', config.get('model', {}).get('sequence_length', 60)))
    target_shift = int(config.get('target_shift', 1))
    risk_window = int(config.get('risk_window', 14))
    
    # Get feature columns (exclude non-numeric)
    exclude_cols = {'ticker', 'date', 'timestamp', 'datetime', 'time'}
    feature_cols = [c for c in df.columns if c.lower() not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    logger.info(f"Using {len(feature_cols)} features for training")
    
    # Prepare sequences using existing function
    x, y_price, meta = prepare_sequences(
        df=df,
        sequence_length=seq_len,
        target_column="close",
        feature_columns=feature_cols,
        target_shift=target_shift,
        scale_features=bool(config.get("enable_data_scaling", True)),
        scale_target=bool(config.get("enable_data_scaling", True)),
    )
    
    # Handle NaN/Inf in feature data for numerical stability
    nan_count = np.isnan(x).sum()
    inf_count = np.isinf(x).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"Found {nan_count} NaN and {inf_count} Inf values in features, replacing...")
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Build multi-task targets using existing function
    targets: MultitaskTargets = build_multitask_targets(
        df,
        sequence_length=seq_len,
        target_shift=target_shift,
        close_col="close",
        risk_window=risk_window,
        state_classes=state_classes,
    )
    
    # Validate alignment
    if len(x) != len(y_price) or len(x) != len(targets.trend):
        raise RuntimeError(f"Target alignment mismatch: x={len(x)}, y_price={len(y_price)}, trend={len(targets.trend)}")
    
    # Chronological train/val split (same as PyTorch)
    train_idx, val_idx = split_time_series(len(x), val_fraction=validation_split)
    
    # Further split val into val and test (50/50)
    val_split_point = len(val_idx) // 2
    test_idx = val_idx[val_split_point:]
    val_idx = val_idx[:val_split_point]
    
    # Print state distribution
    train_state = targets.state[train_idx]
    val_state = targets.state[val_idx]
    unique_train, counts_train = np.unique(train_state, return_counts=True)
    unique_val, counts_val = np.unique(val_state, return_counts=True)
    logger.info(f"[State distribution] Train: {dict(zip(unique_train, counts_train))}")
    logger.info(f"[State distribution] Val: {dict(zip(unique_val, counts_val))}")
    
    # Scale risk values to [0, 1] range for numerical stability
    # Risk is rolling std of returns, typically in range [0, 0.01]
    risk_values = targets.risk.astype(np.float32)
    risk_max = np.nanmax(risk_values)
    if risk_max > 0:
        risk_scaled = risk_values / risk_max
    else:
        risk_scaled = np.zeros_like(risk_values)
    # Handle any NaN values
    risk_scaled = np.nan_to_num(risk_scaled, nan=0.5, posinf=1.0, neginf=0.0)
    meta['risk_scale'] = float(risk_max) if risk_max > 0 else 1.0
    logger.info(f"Risk scaling: max={risk_max:.6f}, scaled to [0, 1]")
    
    def create_target_dict(indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Create target dictionary with one-hot encoded state."""
        return {
            'price': y_price[indices].astype(np.float32).reshape(-1, 1),
            'trend': targets.trend[indices].astype(np.float32).reshape(-1, 1),
            'direction': targets.direction[indices].astype(np.float32).reshape(-1, 1),
            'risk': risk_scaled[indices].astype(np.float32).reshape(-1, 1),
            'state_logits': np.eye(state_classes, dtype=np.float32)[targets.state[indices]],  # One-hot
        }
    
    x_train = x[train_idx].astype(np.float32)
    x_val = x[val_idx].astype(np.float32)
    x_test = x[test_idx].astype(np.float32)
    
    y_train = create_target_dict(train_idx)
    y_val = create_target_dict(val_idx)
    y_test = create_target_dict(test_idx)
    
    # Add scaler info to meta for persistence
    meta['state_classes'] = state_classes
    meta['train_size'] = len(train_idx)
    meta['val_size'] = len(val_idx)
    meta['test_size'] = len(test_idx)
    
    logger.info(f"Data loaded: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test, meta


def load_multi_instrument_data(
    csv_paths: List[str],
    config: Dict[str, Any],
    *,
    state_classes: int = 3,
    validation_split: float = 0.2,
    add_all_features: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load and combine data from multiple instruments for improved generalization.
    
    This implements Step 6: Multi-instrument training — combining ~15,000 samples 
    from 3 instruments (5000 candles each) to improve model generalization.
    
    Each instrument is processed independently with its own feature scaling,
    then combined. This helps the model learn patterns that generalize across
    different FX pairs rather than overfitting to a single instrument.
    
    Args:
        csv_paths: List of paths to CSV files (e.g., EUR_USD, GBP_USD, USD_JPY)
        config: Configuration dictionary
        state_classes: Number of volatility regime classes (3 for 93%+ accuracy)
        validation_split: Fraction for validation set (chronological split per instrument)
        add_all_features: Add engineered features (RSI, MACD, etc.)
    
    Returns:
        X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict, meta
        
        Where y_dict contains:
            - 'price': (n,) float32
            - 'trend': (n,) float32
            - 'risk': (n,) float32
            - 'direction': (n,) float32  # Binary up/down
            - 'state_logits': (n, state_classes) float32 one-hot
    """
    all_x_train, all_x_val, all_x_test = [], [], []
    all_y_train = {'price': [], 'trend': [], 'risk': [], 'direction': [], 'state_logits': []}
    all_y_val = {'price': [], 'trend': [], 'risk': [], 'direction': [], 'state_logits': []}
    all_y_test = {'price': [], 'trend': [], 'risk': [], 'direction': [], 'state_logits': []}
    
    combined_meta = {
        'instruments': [],
        'samples_per_instrument': [],
        'total_samples': 0,
        'feature_columns': None,
        'state_classes': state_classes,
    }
    
    logger.info("\n" + "=" * 60)
    logger.info(f"MULTI-INSTRUMENT LOADING: {len(csv_paths)} instruments")
    logger.info("=" * 60)
    
    for i, csv_path in enumerate(csv_paths):
        instrument_name = Path(csv_path).stem
        logger.info(f"\n[{i+1}/{len(csv_paths)}] Loading: {instrument_name}")
        
        try:
            # Load this instrument's data
            x_train, y_train, x_val, y_val, x_test, y_test, meta = load_tensorflow_multitask_data(
                csv_path,
                config,
                state_classes=state_classes,
                validation_split=validation_split,
                add_all_features=add_all_features,
            )
            
            # Direction labels are already provided by build_multitask_targets
            # (aligned to the same prediction horizon as the price target).
            
            # Accumulate data
            all_x_train.append(x_train)
            all_x_val.append(x_val)
            all_x_test.append(x_test)
            
            for key in all_y_train:
                all_y_train[key].append(y_train[key])
                all_y_val[key].append(y_val[key])
                all_y_test[key].append(y_test[key])
            
            # Update metadata
            combined_meta['instruments'].append(instrument_name)
            combined_meta['samples_per_instrument'].append({
                'train': len(x_train),
                'val': len(x_val),
                'test': len(x_test),
            })
            
            if combined_meta['feature_columns'] is None:
                combined_meta['feature_columns'] = meta.get('feature_columns')
                combined_meta['sequence_length'] = meta.get('sequence_length')
            
            logger.info(f"   Loaded: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
            
        except Exception as e:
            logger.error(f"   Failed to load {csv_path}: {e}")
            print(f"❌ Error loading {csv_path}: {e}")  # Ensure user sees this
            import traceback
            traceback.print_exc()
            continue
    
    if not all_x_train:
        raise ValueError("No data was loaded from any instrument!")
    
    # Concatenate all instruments
    x_train_combined = np.concatenate(all_x_train, axis=0)
    x_val_combined = np.concatenate(all_x_val, axis=0)
    x_test_combined = np.concatenate(all_x_test, axis=0)
    
    y_train_combined = {key: np.concatenate(vals, axis=0) for key, vals in all_y_train.items()}
    y_val_combined = {key: np.concatenate(vals, axis=0) for key, vals in all_y_val.items()}
    y_test_combined = {key: np.concatenate(vals, axis=0) for key, vals in all_y_test.items()}
    
    # Shuffle training data (important for multi-instrument to mix samples)
    rng = np.random.default_rng(seed=42)
    train_indices = rng.permutation(len(x_train_combined))
    x_train_combined = x_train_combined[train_indices]
    y_train_combined = {key: val[train_indices] for key, val in y_train_combined.items()}
    
    # Update metadata
    combined_meta['train_size'] = len(x_train_combined)
    combined_meta['val_size'] = len(x_val_combined)
    combined_meta['test_size'] = len(x_test_combined)
    combined_meta['total_samples'] = len(x_train_combined) + len(x_val_combined) + len(x_test_combined)
    combined_meta['n_features'] = x_train_combined.shape[-1]
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-INSTRUMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Instruments: {combined_meta['instruments']}")
    logger.info(f"  Total samples: {combined_meta['total_samples']:,}")
    logger.info(f"  Train: {len(x_train_combined):,} | Val: {len(x_val_combined):,} | Test: {len(x_test_combined):,}")
    logger.info(f"  Features: {combined_meta['n_features']}")
    logger.info(f"  State classes: {state_classes}")
    
    # Direction distribution
    train_dir = y_train_combined['direction']
    up_pct = 100 * np.mean(train_dir)
    logger.info(f"  Direction distribution: UP={up_pct:.1f}% | DOWN={100-up_pct:.1f}%")
    
    return x_train_combined, y_train_combined, x_val_combined, y_val_combined, x_test_combined, y_test_combined, combined_meta


def create_tf_dataset(
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    batch_size: int = 64,
    shuffle: bool = True,
    buffer_size: int = 10000,
) -> tf.data.Dataset:
    """
    Create tf.data.Dataset from numpy arrays with multi-task targets.
    
    Args:
        X: Features array [samples, timesteps, features]
        y: Dict of target arrays {'price': ..., 'trend': ..., 'risk': ..., 'state_logits': ...}
        batch_size: Batch size
        shuffle: Whether to shuffle
        buffer_size: Shuffle buffer size
    
    Returns:
        tf.data.Dataset configured for training
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def save_scalers(
    meta: Dict[str, Any],
    save_path: str,
) -> None:
    """
    Save scalers from metadata for inference consistency.
    
    Args:
        meta: Metadata dict containing feature_scaler and target_scaler
        save_path: Path to save pickle file (if directory, saves as scalers.pkl inside)
    """
    save_data = {
        'feature_scaler': meta.get('feature_scaler'),
        'target_scaler': meta.get('target_scaler'),
        'feature_columns': meta.get('feature_columns'),
        'target_column': meta.get('target_column'),
        'sequence_length': meta.get('sequence_length'),
        'target_shift': meta.get('target_shift'),
        'state_classes': meta.get('state_classes'),
    }
    
    # Handle directory or file path
    save_path = Path(save_path)
    if save_path.is_dir() or not save_path.suffix:
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / 'scalers.pkl'
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"Scalers saved to {save_path}")


def load_scalers(load_path: str) -> Dict[str, Any]:
    """
    Load saved scalers for inference.
    
    Args:
        load_path: Path to pickle file or directory containing scalers.pkl
    
    Returns:
        Dict with scalers and metadata
    """
    load_path = Path(load_path)
    if load_path.is_dir():
        load_path = load_path / 'scalers.pkl'
    
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Scalers loaded from {load_path}")
    return data


def find_data_files(
    data_dir: str = "market_data",
    pattern: str = "*.csv",
) -> List[str]:
    """
    Find all CSV files in data directory.
    
    Args:
        data_dir: Directory to search
        pattern: Glob pattern
    
    Returns:
        List of file paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        # Try trained_data/data
        data_path = Path("trained_data/data")
    
    if not data_path.exists():
        return []
    
    return sorted([str(p) for p in data_path.glob(pattern)])


def get_default_data_file(config: Dict[str, Any]) -> Optional[str]:
    """
    Get default data file from config or find first available.
    
    Args:
        config: Configuration dict
    
    Returns:
        Path to data file or None
    """
    # Check config for explicit path
    data_path = (
        config.get('data', {}).get('csv_path') or
        config.get('data_path') or
        config.get('csv_path')
    )
    
    if data_path and Path(data_path).exists():
        return data_path
    
    # Check standard directories
    data_dirs = [
        config.get('data', {}).get('data_dir', 'market_data'),
        'market_data',
        'trained_data/data',
    ]
    
    for data_dir in data_dirs:
        files = find_data_files(data_dir)
        if files:
            logger.info(f"Using default data file: {files[0]}")
            return files[0]
    
    return None


# =============================================================================
# TensorFlowDataPipeline Class (matches train_visual.py interface)
# =============================================================================

class TensorFlowDataPipeline:
    """
    High-level data pipeline class for TensorFlow multi-task training.
    
    This class provides a simple interface that matches the train_visual.py 
    load_real_data() function expectations.
    
    Usage:
        pipeline = TensorFlowDataPipeline(config)
        result = pipeline.load_multitask_data(data_file='path/to/data.csv')
        train_ds, val_ds, test_ds, metadata = result
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data pipeline.
        
        Args:
            config: Configuration dict with keys:
                - data_dir: Directory containing CSV files
                - sequence_length: Time series sequence length
                - price_col: Price column name (default: 'close')
                - state_classes: Number of state classes
                - train_split: Training data fraction
                - val_split: Validation data fraction  
                - batch_size: Batch size for datasets
        """
        self.config = config
        self.scalers = {}
        self.metadata = {}
    
    def load_multitask_data(
        self,
        data_file: str = None,
        data_dir: str = None,
    ) -> Optional[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]]:
        """
        Load multi-task data from CSV files.
        
        Args:
            data_file: Specific CSV file to load
            data_dir: Directory containing CSV files (uses first file found)
        
        Returns:
            (train_dataset, val_dataset, test_dataset, metadata) or None if failed
        """
        # Determine data source
        if data_file:
            csv_path = data_file
        elif data_dir:
            files = find_data_files(data_dir)
            if not files:
                logger.error(f"No CSV files found in {data_dir}")
                return None
            csv_path = files[0]
        else:
            csv_path = get_default_data_file(self.config)
            
        if not csv_path or not Path(csv_path).exists():
            logger.error(f"Data file not found: {csv_path}")
            return None
        
        logger.info(f"Loading data from: {csv_path}")
        
        # Get configuration
        state_classes = self.config.get('state_classes', 2)
        val_split = self.config.get('val_split', 0.15)
        batch_size = self.config.get('batch_size', 64)
        
        # Load data using main function
        try:
            x_train, y_train, x_val, y_val, x_test, y_test, meta = load_tensorflow_multitask_data(
                csv_path,
                self.config,
                state_classes=state_classes,
                validation_split=val_split + (1 - self.config.get('train_split', 0.7)),  # Include test split
                add_all_features=True,
            )
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Store scalers for later
        self.scalers = {
            'feature_scaler': meta.get('feature_scaler'),
            'target_scaler': meta.get('target_scaler'),
        }
        
        # Update metadata
        self.metadata = {
            'n_train': len(x_train),
            'n_val': len(x_val),
            'n_test': len(x_test),
            'n_features': x_train.shape[-1],
            'sequence_length': x_train.shape[1],
            'state_classes': state_classes,
            'csv_path': csv_path,
            **meta,
        }
        
        # Create tf.data.Dataset objects
        train_ds = create_tf_dataset(x_train, y_train, batch_size=batch_size, shuffle=True)
        val_ds = create_tf_dataset(x_val, y_val, batch_size=batch_size, shuffle=False)
        test_ds = create_tf_dataset(x_test, y_test, batch_size=batch_size, shuffle=False)
        
        return train_ds, val_ds, test_ds, self.metadata
    
    def save_scalers(self, save_path: str) -> None:
        """Save scalers to directory for inference."""
        if not self.scalers:
            logger.warning("No scalers to save")
            return
            
        save_scalers({'scalers': self.scalers, **self.metadata}, save_path)
        logger.info(f"Scalers saved to {save_path}")
    
    def load_scalers(self, load_path: str) -> Dict[str, Any]:
        """Load scalers from directory."""
        data = load_scalers(load_path)
        self.scalers = data.get('scalers', {})
        return data


class AdaptiveLossWeightCallback(tf.keras.callbacks.Callback):
    """
    Callback to adaptively adjust loss weights during training.
    Boosts weights for underperforming heads (like PyTorch _adjust_weights).
    
    This helps balance multi-task learning when some heads train faster than others.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adjustment_factor: float = 1.1,
        patience: int = 3,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ):
        """
        Args:
            initial_weights: Starting loss weights {'price': 1.0, ...}
            adjustment_factor: Factor to boost lagging heads
            patience: Epochs without improvement before boosting
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        super().__init__()
        self.weights = dict(initial_weights)
        self.adjustment_factor = adjustment_factor
        self.patience = patience
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Track best losses per head
        self.best_losses = dict.fromkeys(initial_weights, float('inf'))
        self.no_improvement_count = dict.fromkeys(initial_weights, 0)
    
    def _get_head_losses(self, logs: dict) -> Dict[str, float]:
        """Extract validation losses for each head from logs."""
        return {
            'price': logs.get('val_price_loss', float('inf')),
            'trend': logs.get('val_trend_loss', float('inf')),
            'risk': logs.get('val_risk_loss', float('inf')),
            'state_logits': logs.get('val_state_logits_loss', float('inf')),
        }
    
    def _update_head_tracking(self, head: str, loss: float) -> None:
        """Update tracking for a single head and boost weight if needed."""
        if loss < self.best_losses[head]:
            self.best_losses[head] = loss
            self.no_improvement_count[head] = 0
        else:
            self.no_improvement_count[head] += 1
        
        # Boost weight if no improvement for patience epochs
        if self.no_improvement_count[head] >= self.patience:
            new_weight = min(self.weights[head] * self.adjustment_factor, self.max_weight)
            if new_weight != self.weights[head]:
                logger.info(f"Boosting {head} weight: {self.weights[head]:.2f} -> {new_weight:.2f}")
                self.weights[head] = new_weight
            self.no_improvement_count[head] = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        head_losses = self._get_head_losses(logs)
        
        for head, loss in head_losses.items():
            if head in self.weights:
                self._update_head_tracking(head, loss)
        
        # Update model loss weights if supported
        if hasattr(self.model, 'loss_weights') and self.model.loss_weights:
            for key in self.weights:
                if key in self.model.loss_weights:
                    self.model.loss_weights[key] = self.weights[key]


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test finding data files
    files = find_data_files("market_data")
    print(f"Found {len(files)} data files in market_data/")
    
    files = find_data_files("trained_data/data")
    print(f"Found {len(files)} data files in trained_data/data/")
    
    if files:
        print(f"\nTesting with: {files[0]}")
        
        config = {
            'sequence_length': 60,
            'target_shift': 1,
            'risk_window': 14,
            'enable_data_scaling': True,
        }
        
        try:
            x_train, y_train, x_val, y_val, x_test, y_test, meta = load_tensorflow_multitask_data(
                files[0],
                config,
                state_classes=2,  # Binary like PyTorch 93% result
            )
            
            print("\nData shapes:")
            print(f"  x_train: {x_train.shape}")
            print(f"  y_train['price']: {y_train['price'].shape}")
            print(f"  y_train['state_logits']: {y_train['state_logits'].shape}")
            print(f"\nFeatures: {len(meta['feature_columns'])}")
            
            # Test tf.data.Dataset creation
            dataset = create_tf_dataset(x_train, y_train, batch_size=32)
            for x_batch, y_batch in dataset.take(1):
                print("\nBatch shapes:")
                print(f"  X: {x_batch.shape}")
                print(f"  y['price']: {y_batch['price'].shape}")
                print(f"  y['state_logits']: {y_batch['state_logits'].shape}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
