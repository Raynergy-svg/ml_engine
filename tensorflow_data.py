"""
TensorFlow Data Pipeline using tf.data.Dataset API.
Provides efficient data loading with prefetching and caching for optimal GPU utilization.

Features:
- High-performance tf.data.Dataset pipeline
- Automatic batching and prefetching
- Data augmentation for time series
- Compatible with existing DataLoader preprocessing
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any, Generator

# Import existing data loading utilities
try:
    from data_loader import DataLoader
except ImportError:
    DataLoader = None


# =============================================================================
# TensorFlow Data Pipeline
# =============================================================================

class TensorFlowDataPipeline:
    """
    High-performance data pipeline using tf.data.Dataset.
    
    Advantages over PyTorch DataLoader:
    - Automatic prefetching to GPU
    - Parallel data loading
    - Built-in caching
    - Seamless integration with TensorFlow training
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        batch_size: int = 64,
        shuffle_buffer: int = 10000,
        prefetch_buffer: int = tf.data.AUTOTUNE,
        cache: bool = True,
    ):
        """
        Initialize data pipeline.
        
        Args:
            config: Configuration dictionary
            batch_size: Batch size for training
            shuffle_buffer: Size of shuffle buffer
            prefetch_buffer: Number of batches to prefetch (AUTOTUNE recommended)
            cache: Whether to cache dataset in memory
        """
        self.config = config
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.cache = cache
        
        # Initialize existing DataLoader for preprocessing
        if DataLoader is not None:
            self.data_loader = DataLoader(config)
        else:
            self.data_loader = None
    
    def create_dataset_from_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        training: bool = True,
    ) -> tf.data.Dataset:
        """
        Create tf.data.Dataset from numpy arrays.
        
        Args:
            X: Features array [samples, timesteps, features]
            y: Targets array [samples, 1] or [samples,]
            training: Whether this is for training (enables shuffling)
        
        Returns:
            Optimized tf.data.Dataset
        """
        # Ensure correct dtypes
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Apply optimizations
        if self.cache:
            dataset = dataset.cache()
        
        if training:
            dataset = dataset.shuffle(
                buffer_size=min(self.shuffle_buffer, len(X)),
                reshuffle_each_iteration=True
            )
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer)
        
        return dataset
    
    def create_dataset_from_generator(
        self,
        generator_fn: Generator,
        output_signature: Tuple[tf.TensorSpec, tf.TensorSpec],
        training: bool = True,
    ) -> tf.data.Dataset:
        """
        Create dataset from a generator function.
        Useful for very large datasets that don't fit in memory.
        
        Args:
            generator_fn: Generator function yielding (X, y) tuples
            output_signature: TensorSpec for outputs
            training: Whether this is for training
        
        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_generator(
            generator_fn,
            output_signature=output_signature
        )
        
        if training:
            dataset = dataset.shuffle(self.shuffle_buffer)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer)
        
        return dataset
    
    def load_and_preprocess(
        self,
        file_path: str,
        sequence_length: int = 60,
        validation_split: float = 0.1,
        test_split: float = 0.1,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load data from CSV and create train/val/test datasets.
        
        Args:
            file_path: Path to CSV file
            sequence_length: Length of input sequences
            validation_split: Fraction for validation
            test_split: Fraction for testing
        
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        if self.data_loader is None:
            raise RuntimeError("DataLoader not available. Install data_loader module.")
        
        # Load and preprocess using existing DataLoader
        x_train, x_val, x_test, y_train, y_val, y_test = self.data_loader.preprocess(
            file_path,
            sequence_length=sequence_length,
            validation_split=validation_split,
            test_split=test_split,
        )
        
        # Create datasets
        train_dataset = self.create_dataset_from_arrays(x_train, y_train, training=True)
        val_dataset = self.create_dataset_from_arrays(x_val, y_val, training=False)
        test_dataset = self.create_dataset_from_arrays(x_test, y_test, training=False)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int = 60,
        target_column: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input data array [samples, features]
            sequence_length: Length of input sequences
            target_column: Index of target column (default: last)
        
        Returns:
            (X, y) where X is [samples, sequence_length, features]
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i - sequence_length:i])
            y.append(data[i, target_column])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def apply_augmentation(
        self,
        dataset: tf.data.Dataset,
        noise_std: float = 0.02,
        time_mask_prob: float = 0.1,
        scale_range: Tuple[float, float] = (0.95, 1.05),
    ) -> tf.data.Dataset:
        """
        Apply comprehensive data augmentation for time series.
        
        Enhanced augmentations for small datasets to prevent overfitting:
        - Gaussian noise injection
        - Random time masking
        - Random scaling
        - Random feature dropout
        
        Args:
            dataset: Input dataset
            noise_std: Standard deviation of Gaussian noise
            time_mask_prob: Probability of masking timesteps
            scale_range: (min, max) for random scaling
        
        Returns:
            Augmented dataset
        """
        def augment(x, y):
            # 1. Add Gaussian noise
            noise = tf.random.normal(tf.shape(x), stddev=noise_std)
            x_aug = x + noise
            
            # 2. Random time masking (simulate missing data)
            if time_mask_prob > 0:
                mask = tf.random.uniform(tf.shape(x)[:2]) > time_mask_prob
                mask = tf.cast(mask, tf.float32)
                mask = tf.expand_dims(mask, -1)
                x_aug = x_aug * mask
            
            # 3. Random scaling (magnitude perturbation)
            scale = tf.random.uniform([], scale_range[0], scale_range[1])
            x_aug = x_aug * scale
            
            # 4. Random feature dropout (simulate sensor failures)
            feature_mask = tf.random.uniform([tf.shape(x)[-1]]) > 0.1
            feature_mask = tf.cast(feature_mask, tf.float32)
            x_aug = x_aug * feature_mask
            
            return x_aug, y
        
        return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    def apply_window_slicing(
        self,
        dataset: tf.data.Dataset,
        slice_ratio: float = 0.9,
    ) -> tf.data.Dataset:
        """
        Apply window slicing augmentation - randomly crop time windows.
        
        Args:
            dataset: Input dataset
            slice_ratio: Ratio of window to keep (0.8 = keep 80% of timesteps)
        
        Returns:
            Augmented dataset with sliced windows
        """
        def window_slice(x, y):
            seq_len = tf.shape(x)[0]
            target_len = tf.cast(tf.cast(seq_len, tf.float32) * slice_ratio, tf.int32)
            start = tf.random.uniform([], 0, seq_len - target_len + 1, dtype=tf.int32)
            
            # Slice and pad back to original length
            x_sliced = x[start:start + target_len]
            padding = tf.zeros([seq_len - target_len, tf.shape(x)[-1]], dtype=x.dtype)
            x_aug = tf.concat([x_sliced, padding], axis=0)
            
            return x_aug, y
        
        return dataset.map(window_slice, num_parallel_calls=tf.data.AUTOTUNE)
    
    def apply_mixup(
        self,
        dataset: tf.data.Dataset,
        alpha: float = 0.2,
    ) -> tf.data.Dataset:
        """
        Apply Mixup augmentation for time series.
        Mixes pairs of samples to create interpolated training examples.
        
        Args:
            dataset: Input dataset (must be batched)
            alpha: Mixup interpolation strength (higher = more mixing)
        
        Returns:
            Augmented dataset with mixup applied
        """
        def mixup(x, y):
            batch_size = tf.shape(x)[0]
            
            # Sample lambda from Beta distribution
            lam = tf.random.uniform([], 0, alpha)
            
            # Shuffle indices for mixing
            indices = tf.random.shuffle(tf.range(batch_size))
            
            # Mix samples
            x_mixed = lam * x + (1 - lam) * tf.gather(x, indices)
            y_mixed = lam * y + (1 - lam) * tf.gather(y, indices)
            
            return x_mixed, y_mixed
        
        return dataset.map(mixup, num_parallel_calls=tf.data.AUTOTUNE)
    
    def apply_magnitude_warping(
        self,
        dataset: tf.data.Dataset,
        sigma: float = 0.2,
        knot: int = 4,
    ) -> tf.data.Dataset:
        """
        Apply magnitude warping - smooth random distortion of magnitudes.
        
        Args:
            dataset: Input dataset
            sigma: Standard deviation for warping curve
            knot: Number of knots for cubic spline
        
        Returns:
            Augmented dataset
        """
        def magnitude_warp(x, y):
            seq_len = tf.shape(x)[0]
            
            # Generate random warping curve values
            warp_values = tf.random.normal([knot + 2], 1.0, sigma)
            
            # Interpolate to sequence length
            indices = tf.linspace(0.0, tf.cast(knot + 1, tf.float32), seq_len)
            lower_indices = tf.cast(tf.floor(indices), tf.int32)
            lower_indices = tf.minimum(lower_indices, knot)
            upper_indices = tf.minimum(lower_indices + 1, knot + 1)
            
            t = indices - tf.cast(lower_indices, tf.float32)
            lower_values = tf.gather(warp_values, lower_indices)
            upper_values = tf.gather(warp_values, upper_indices)
            warp_curve = lower_values * (1 - t) + upper_values * t
            
            # Apply warping
            warp_curve = tf.expand_dims(warp_curve, -1)
            x_warped = x * warp_curve
            
            return x_warped, y
        
        return dataset.map(magnitude_warp, num_parallel_calls=tf.data.AUTOTUNE)
    
    def create_augmented_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation_factor: int = 3,
        training: bool = True,
    ) -> tf.data.Dataset:
        """
        Create a heavily augmented dataset for small data scenarios.
        
        Args:
            X: Features array
            y: Targets array
            augmentation_factor: How many augmented versions to create
            training: Whether this is for training
        
        Returns:
            Augmented tf.data.Dataset
        """
        # Create base dataset
        base_dataset = self.create_dataset_from_arrays(X, y, training=training)
        
        if not training:
            return base_dataset
        
        # Apply multiple augmentation passes
        augmented_datasets = [base_dataset]
        
        for i in range(augmentation_factor - 1):
            # Apply different augmentation combinations
            aug_ds = base_dataset
            
            # Gaussian noise (always apply)
            aug_ds = self.apply_augmentation(
                aug_ds, 
                noise_std=0.02 * (i + 1),  # Increasing noise levels
                time_mask_prob=0.05 * (i + 1),
            )
            
            # Apply magnitude warping on some versions
            if i % 2 == 0:
                aug_ds = self.apply_magnitude_warping(aug_ds, sigma=0.15)
            
            # Apply mixup on some versions
            if i % 2 == 1:
                aug_ds = self.apply_mixup(aug_ds, alpha=0.2)
            
            augmented_datasets.append(aug_ds)
        
        # Concatenate all augmented datasets
        combined = augmented_datasets[0]
        for ds in augmented_datasets[1:]:
            combined = combined.concatenate(ds)
        
        # Shuffle combined dataset
        combined = combined.shuffle(buffer_size=len(X) * augmentation_factor)
        
        return combined


# =============================================================================
# Utility Functions
# =============================================================================

def numpy_to_tensorflow_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 64,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
    """
    Convert numpy arrays to optimized tf.data.Datasets.
    
    Example:
        train_ds, val_ds, test_ds = numpy_to_tensorflow_datasets(
            x_train, y_train, x_val, y_val, x_test, y_test
        )
    """
    pipeline = TensorFlowDataPipeline({}, batch_size=batch_size)
    
    train_ds = pipeline.create_dataset_from_arrays(x_train, y_train, training=True)
    val_ds = pipeline.create_dataset_from_arrays(x_val, y_val, training=False)
    
    test_ds = None
    if x_test is not None and y_test is not None:
        test_ds = pipeline.create_dataset_from_arrays(x_test, y_test, training=False)
    
    return train_ds, val_ds, test_ds


def create_time_series_dataset(
    df: pd.DataFrame,
    target_column: str = 'close',
    feature_columns: Optional[list] = None,
    sequence_length: int = 60,
    batch_size: int = 64,
    validation_split: float = 0.1,
    test_split: float = 0.1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Create time series datasets from a pandas DataFrame.
    
    Args:
        df: DataFrame with time series data
        target_column: Name of target column
        feature_columns: List of feature columns (None = all except target)
        sequence_length: Length of input sequences
        batch_size: Batch size
        validation_split: Fraction for validation
        test_split: Fraction for testing
    
    Returns:
        (train_ds, val_ds, test_ds, info_dict)
    """
    # Prepare features
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    features = df[feature_columns].values.astype(np.float32)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i - sequence_length:i])
        # Target is next value of target column
        y.append(df[target_column].iloc[i])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Split data (time series aware - no shuffling before split)
    n_samples = len(X)
    test_size = int(n_samples * test_split)
    val_size = int(n_samples * validation_split)
    train_size = n_samples - test_size - val_size
    
    x_train, y_train = X[:train_size], y[:train_size]
    x_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    x_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    # Create datasets
    train_ds, val_ds, test_ds = numpy_to_tensorflow_datasets(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size
    )
    
    info = {
        'n_samples': n_samples,
        'n_train': train_size,
        'n_val': val_size,
        'n_test': test_size,
        'n_features': len(feature_columns),
        'sequence_length': sequence_length,
        'feature_columns': feature_columns,
        'scaler': scaler,
    }
    
    return train_ds, val_ds, test_ds, info


# =============================================================================
# Dataset Info and Debugging
# =============================================================================

def inspect_dataset(dataset: tf.data.Dataset, name: str = "Dataset"):
    """Print information about a tf.data.Dataset."""
    print(f"\n{name} Info:")
    print("-" * 40)
    
    # Get one batch to inspect shape
    for x, y in dataset.take(1):
        print(f"  Batch X shape: {x.shape}")
        print(f"  Batch y shape: {y.shape}")
        print(f"  X dtype: {x.dtype}")
        print(f"  y dtype: {y.dtype}")
        print(f"  X range: [{tf.reduce_min(x):.4f}, {tf.reduce_max(x):.4f}]")
        print(f"  y range: [{tf.reduce_min(y):.4f}, {tf.reduce_max(y):.4f}]")
    
    # Count total batches
    n_batches = sum(1 for _ in dataset)
    print(f"  Total batches: {n_batches}")


if __name__ == "__main__":
    print("TensorFlow Data Pipeline")
    print("=" * 60)
    
    # Example with synthetic data
    print("\nCreating synthetic time series data...")
    
    # Generate synthetic data
    n_samples = 5000
    n_features = 10
    sequence_length = 60
    
    # Synthetic features
    rng = np.random.default_rng(seed=42)
    raw_data = np.cumsum(rng.standard_normal((n_samples, n_features)), axis=0)
    
    # Create sequences
    features, targets = [], []
    for i in range(sequence_length, n_samples):
        features.append(raw_data[i - sequence_length:i])
        targets.append(raw_data[i, 0])  # Predict first feature
    
    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    # Split
    split_idx = int(len(features) * 0.8)
    x_train, x_val = features[:split_idx], features[split_idx:]
    y_train, y_val = targets[:split_idx], targets[split_idx:]
    
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    # Create datasets
    print("\nCreating tf.data.Dataset pipelines...")
    
    pipeline = TensorFlowDataPipeline({}, batch_size=64)
    
    train_ds = pipeline.create_dataset_from_arrays(x_train, y_train, training=True)
    val_ds = pipeline.create_dataset_from_arrays(x_val, y_val, training=False)
    
    # Inspect
    inspect_dataset(train_ds, "Training Dataset")
    inspect_dataset(val_ds, "Validation Dataset")
    
    print("\n✓ TensorFlow data pipeline ready!")
    print("\nUsage with TensorFlow training:")
    print("-" * 60)
    print("""
# Train with tf.data.Dataset
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[tensorboard_callback]
)
""")
