import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scaler = None
        self.feature_names = None

    def load_from_csv(
        self, filepath: str, target_column: str = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")

            if target_column and target_column in df.columns:
                X = df.drop(columns=[target_column]).values
                y = df[target_column].values
                self.feature_names = df.drop(columns=[target_column]).columns.tolist()
            else:
                X = df.values
                y = None
                self.feature_names = df.columns.tolist()

            return X, y

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def load_from_numpy(
        self, x_path: str, y_path: str = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load data from numpy files"""
        X = np.load(x_path)
        y = np.load(y_path) if y_path else None
        logger.info(f"Loaded numpy data: X shape {X.shape}")
        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Dict[str, np.ndarray]:
        """Split data into train/val/test sets"""
        rng = np.random.default_rng(random_state)
        n_samples = len(X)
        indices = rng.permutation(n_samples)

        # Calculate split points
        test_split = int(n_samples * (1 - test_size))
        val_split = int(test_split * (1 - val_size))

        train_idx = indices[:val_split]
        val_idx = indices[val_split:test_split]
        test_idx = indices[test_split:]

        return {
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
            "X_test": X[test_idx],
            "y_test": y[test_idx],
        }

    def preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess features with normalization"""
        # Handle missing values
        X = self._handle_missing(X)

        # Normalize
        if fit or self.scaler is None:
            self.scaler = self._fit_scaler(X)

        x_scaled = self._apply_scaler(X)

        return x_scaled

    def _handle_missing(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values"""
        if np.isnan(X).any():
            logger.warning("Missing values detected, filling with column means")
            col_means = np.nanmean(X, axis=0)
            nan_idx = np.nonzero(np.isnan(X))
            X[nan_idx] = np.take(col_means, nan_idx[1])
        return X

    def _fit_scaler(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit standard scaler"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        return {"mean": mean, "std": std}

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Apply standard scaling"""
        return (X - self.scaler["mean"]) / self.scaler["std"]

    def create_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        """Create data batches for training"""
        n_samples = len(X)
        for i in range(0, n_samples, batch_size):
            yield X[i : i + batch_size], y[i : i + batch_size]

    def augment_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.01,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment data with noise"""
        rng = np.random.default_rng(random_state)
        x_augmented = X + rng.standard_normal(size=X.shape) * noise_level
        y_augmented = y.copy()

        x_combined = np.vstack([X, x_augmented])
        y_combined = np.hstack([y, y_augmented])

        return x_combined, y_combined
