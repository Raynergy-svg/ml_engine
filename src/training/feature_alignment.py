"""
Feature Alignment Utilities for ML Pipeline.

This module provides utilities to ensure consistent feature dimensions
between training and inference, addressing the "X has 80 features, but 
expecting 45" StandardScaler error.

Key Features:
1. Save feature selection indices during training
2. Load and apply indices during inference
3. Handle scaler state with feature indices

Usage:
    from src.training.feature_alignment import (
        save_feature_indices,
        load_feature_indices,
        align_features_to_model,
        save_scaler_with_indices,
        load_scaler_with_indices,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Default directory for feature indices
DEFAULT_MODEL_DIR = "trained_data/models"


def save_feature_indices(
    selected_indices: np.ndarray,
    model_dir: str,
    filename: str = "feature_indices.pkl",
) -> Path:
    """
    Save feature selection indices to disk.

    Args:
        selected_indices: Array of selected feature indices
        model_dir: Directory to save the indices
        filename: Name of the file (default: feature_indices.pkl)

    Returns:
        Path to the saved file
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    indices_path = model_path / filename
    joblib.dump({
        'selected_indices': np.array(selected_indices),
        'n_features': len(selected_indices),
    }, indices_path)
    
    logger.info(f"Saved {len(selected_indices)} feature indices to {indices_path}")
    return indices_path


def load_feature_indices(
    model_dir: str,
    filename: str = "feature_indices.pkl",
) -> Optional[np.ndarray]:
    """
    Load feature selection indices from disk.

    Args:
        model_dir: Directory containing the indices file
        filename: Name of the file (default: feature_indices.pkl)

    Returns:
        Array of selected feature indices, or None if not found
    """
    indices_path = Path(model_dir) / filename
    
    if not indices_path.exists():
        logger.debug(f"No feature indices found at {indices_path}")
        return None
    
    try:
        data = joblib.load(indices_path)
        indices = data.get('selected_indices', data)
        logger.info(f"Loaded {len(indices)} feature indices from {indices_path}")
        return np.array(indices)
    except Exception as e:
        logger.warning(f"Failed to load feature indices from {indices_path}: {e}")
        return None


def align_features_to_model(
    X: np.ndarray,
    model_dir: str,
    indices_filename: str = "feature_indices.pkl",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Align features to match model expectations.

    This function loads the saved feature indices and applies them
    to the input features to ensure dimension compatibility.

    Args:
        X: Input feature array (n_samples, n_features)
        model_dir: Directory containing the feature indices
        indices_filename: Name of the indices file

    Returns:
        Tuple of (aligned_features, selected_indices)
        - aligned_features: Feature array with correct dimensions
        - selected_indices: The indices used for selection (or None)

    Raises:
        ValueError: If feature dimensions are incompatible
    """
    selected_indices = load_feature_indices(model_dir, indices_filename)
    
    if selected_indices is None:
        # No feature selection was used during training
        return X, None
    
    n_input_features = X.shape[1] if X.ndim == 2 else X.shape[-1]
    n_expected_features = len(selected_indices)
    
    # Check if indices are valid for input
    max_index = int(np.max(selected_indices))
    
    if n_input_features == n_expected_features:
        # Features already aligned (selection was already applied)
        logger.debug(f"Features already aligned: {n_input_features} features")
        return X, selected_indices
    
    if n_input_features > n_expected_features:
        # Need to apply feature selection
        if max_index >= n_input_features:
            raise ValueError(
                f"Feature index mismatch: indices max ({max_index}) >= "
                f"input features ({n_input_features}). "
                f"Model may have been trained on different feature set."
            )
        
        if X.ndim == 2:
            X_aligned = X[:, selected_indices]
        else:
            # Handle 3D input (sequences)
            X_aligned = X[..., selected_indices]
        
        logger.info(
            f"Aligned features: {n_input_features} -> {n_expected_features} "
            f"using saved indices"
        )
        return X_aligned, selected_indices
    
    # n_input_features < n_expected_features
    raise ValueError(
        f"Feature mismatch: input has {n_input_features} features, "
        f"but model expects {n_expected_features}. "
        f"Missing {n_expected_features - n_input_features} features. "
        f"Ensure you're using the same feature engineering pipeline."
    )


def save_scaler_with_indices(
    scaler: Any,
    model_dir: str,
    selected_indices: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Path:
    """
    Save scaler with feature indices for consistent inference.

    Args:
        scaler: Fitted sklearn scaler
        model_dir: Directory to save the scaler
        selected_indices: Feature selection indices (optional)
        prefix: Prefix for the filename (e.g., "direction_")

    Returns:
        Path to the saved scaler file
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    scaler_path = model_path / f"{prefix}scaler.pkl"
    
    save_data = {
        'scaler': scaler,
        'n_features_in_': getattr(scaler, 'n_features_in_', None),
        'feature_indices': selected_indices,
    }
    
    joblib.dump(save_data, scaler_path)
    logger.info(f"Saved scaler with feature info to {scaler_path}")
    
    return scaler_path


def load_scaler_with_indices(
    model_dir: str,
    prefix: str = "",
) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """
    Load scaler with feature indices.

    Args:
        model_dir: Directory containing the scaler
        prefix: Prefix for the filename (e.g., "direction_")

    Returns:
        Tuple of (scaler, feature_indices)
    """
    scaler_path = Path(model_dir) / f"{prefix}scaler.pkl"
    
    if not scaler_path.exists():
        logger.debug(f"No scaler found at {scaler_path}")
        return None, None
    
    try:
        data = joblib.load(scaler_path)
        
        # Handle both old format (direct scaler) and new format (dict)
        if isinstance(data, dict):
            scaler = data.get('scaler')
            feature_indices = data.get('feature_indices')
            logger.info(f"Loaded scaler from {scaler_path} (with feature info)")
        else:
            scaler = data
            feature_indices = None
            logger.info(f"Loaded scaler from {scaler_path} (legacy format)")
        
        return scaler, feature_indices
    
    except Exception as e:
        logger.warning(f"Failed to load scaler from {scaler_path}: {e}")
        return None, None


class FeatureAligner:
    """
    Class to manage feature alignment for a model.

    Usage:
        aligner = FeatureAligner(model_dir="trained_data/models/joint")
        X_aligned = aligner.transform(X)
    """
    
    def __init__(
        self,
        model_dir: str,
        indices_filename: str = "feature_indices.pkl",
    ):
        """
        Initialize feature aligner.

        Args:
            model_dir: Directory containing model artifacts
            indices_filename: Name of the feature indices file
        """
        self.model_dir = model_dir
        self.indices_filename = indices_filename
        self.selected_indices: Optional[np.ndarray] = None
        self._loaded = False
    
    def load(self) -> bool:
        """
        Load feature indices from disk.

        Returns:
            True if indices were loaded, False otherwise
        """
        self.selected_indices = load_feature_indices(
            self.model_dir, self.indices_filename
        )
        self._loaded = True
        return self.selected_indices is not None
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Align features to match model expectations.

        Args:
            X: Input feature array

        Returns:
            Aligned feature array
        """
        if not self._loaded:
            self.load()
        
        if self.selected_indices is None:
            return X
        
        if X.ndim == 2:
            return X[:, self.selected_indices]
        else:
            return X[..., self.selected_indices]
    
    def get_n_features(self) -> Optional[int]:
        """
        Get the number of features expected by the model.

        Returns:
            Number of features, or None if not loaded
        """
        if not self._loaded:
            self.load()
        
        if self.selected_indices is not None:
            return len(self.selected_indices)
        return None


def ensure_feature_compatibility(
    X: np.ndarray,
    expected_n_features: int,
    model_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Ensure feature compatibility between training and inference.

    This is a convenience function that handles common feature alignment
    scenarios with informative error messages.

    Args:
        X: Input feature array
        expected_n_features: Number of features the model expects
        model_dir: Optional model directory to load indices from

    Returns:
        Feature array with correct dimensions

    Raises:
        ValueError: If features cannot be aligned
    """
    actual_n_features = X.shape[1] if X.ndim == 2 else X.shape[-1]
    
    if actual_n_features == expected_n_features:
        return X
    
    # Try to load and apply feature indices
    if model_dir:
        try:
            X_aligned, _ = align_features_to_model(X, model_dir)
            return X_aligned
        except Exception as e:
            logger.warning(f"Could not align features using indices: {e}")
    
    # Provide helpful error message
    if actual_n_features > expected_n_features:
        # Try using first N features as fallback
        logger.warning(
            f"Feature mismatch: {actual_n_features} features provided, "
            f"{expected_n_features} expected. Using first {expected_n_features} features."
        )
        if X.ndim == 2:
            return X[:, :expected_n_features]
        else:
            return X[..., :expected_n_features]
    
    raise ValueError(
        f"Feature mismatch: {actual_n_features} features provided, "
        f"but model expects {expected_n_features}. "
        f"Missing {expected_n_features - actual_n_features} features. "
        f"Ensure you're using the same feature engineering pipeline as training."
    )


# Backward compatibility - expose functions at module level
__all__ = [
    'save_feature_indices',
    'load_feature_indices',
    'align_features_to_model',
    'save_scaler_with_indices',
    'load_scaler_with_indices',
    'FeatureAligner',
    'ensure_feature_compatibility',
]
