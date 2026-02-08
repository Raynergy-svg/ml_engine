"""
XGBoost Trading Model for FX Prediction.

Research shows gradient boosting often outperforms deep learning on tabular financial data.
This module provides an XGBoost-based alternative to the TCN model.

Key advantages:
1. Better handling of tabular features
2. Built-in feature importance
3. Faster training
4. Less prone to overfitting on small datasets
5. No GPU required

Usage:
    from xgboost_model import XGBoostTradingModel

    model = XGBoostTradingModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import xgboost, fall back gracefully
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    xgb = None  # type: ignore
    # Check if it's a libomp issue on macOS
    err_str = str(e).lower()
    if "libomp" in err_str or "openmp" in err_str:
        logger.warning("XGBoost requires OpenMP. Run: brew install libomp")
    else:
        logger.warning(f"XGBoost not available: {e}")


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost trading model."""

    # Core parameters
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 3

    # Regularization
    gamma: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization

    # Training
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"

    # Multi-output
    direction_weight: float = 1.0
    confidence_weight: float = 0.5

    def to_xgb_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameter dict."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "eval_metric": self.eval_metric,
            "use_label_encoder": False,
            "verbosity": 0,
            "n_jobs": -1,
            "random_state": 42,
        }


class XGBoostTradingModel:
    """
    XGBoost-based trading model for direction and confidence prediction.

    This model trains two separate XGBoost classifiers:
    1. Direction model: Predicts up/down
    2. Confidence model: Predicts probability of correct prediction
    """

    def __init__(self, config: Optional[XGBoostConfig] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.config = config or XGBoostConfig()
        self.direction_model: Optional[xgb.XGBClassifier] = None
        self.confidence_model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self._feature_importance: Optional[Dict[str, float]] = None

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten sequence data (batch, seq_len, features) to (batch, seq_len * features).

        For XGBoost, we need 2D input. We flatten sequences and add lag features.
        """
        if X.ndim == 3:
            batch_size, seq_len, n_features = X.shape
            # Take only the last few timesteps to avoid feature explosion
            # Use last 5 timesteps (or all if seq_len < 5)
            use_steps = min(5, seq_len)
            X_recent = X[:, -use_steps:, :]

            # Flatten: (batch, use_steps, features) -> (batch, use_steps * features)
            X_flat = X_recent.reshape(batch_size, -1)

            # Also add aggregated features from full sequence
            X_mean = X.mean(axis=1)  # Mean across time
            X_std = X.std(axis=1)    # Std across time
            X_last = X[:, -1, :]     # Last timestep

            # Combine: recent flattened + aggregates
            X_combined = np.concatenate([X_flat, X_mean, X_std, X_last], axis=1)
            return X_combined

        return X

    def fit(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_confidence: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val_direction: Optional[np.ndarray] = None,
        y_val_confidence: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the XGBoost models.

        Args:
            X: Features (batch, seq_len, features) or (batch, features)
            y_direction: Binary direction labels (0/1)
            y_confidence: Binary confidence labels (optional)
            X_val, y_val_*: Validation data (optional)
            verbose: Print training progress

        Returns:
            Training history dict
        """
        # Flatten sequences
        X_flat = self._flatten_sequences(X)
        X_val_flat = self._flatten_sequences(X_val) if X_val is not None else None

        # Ensure labels are binary
        y_dir = (np.asarray(y_direction).flatten() > 0.5).astype(np.int32)
        y_val_dir = (np.asarray(y_val_direction).flatten() > 0.5).astype(np.int32) if y_val_direction is not None else None

        if verbose:
            logger.info(f"Training XGBoost: X_flat shape = {X_flat.shape}, y_dir shape = {y_dir.shape}")

        # Build direction model
        params = self.config.to_xgb_params()
        params["objective"] = "binary:logistic"

        self.direction_model = xgb.XGBClassifier(**params)

        # Fit with early stopping if validation data provided
        fit_params = {}
        if X_val_flat is not None and y_val_dir is not None:
            fit_params["eval_set"] = [(X_val_flat, y_val_dir)]
            fit_params["verbose"] = verbose

        self.direction_model.fit(X_flat, y_dir, **fit_params)

        # Get direction predictions on training data for confidence model
        dir_preds = self.direction_model.predict_proba(X_flat)[:, 1]
        dir_correct = ((dir_preds > 0.5) == y_dir).astype(np.int32)

        # Build confidence model (predicts if direction prediction is correct)
        if y_confidence is not None:
            # Use provided confidence labels
            y_conf = (np.asarray(y_confidence).flatten() > 0.5).astype(np.int32)
        else:
            # Use correctness as confidence target
            y_conf = dir_correct

        # Add direction prediction as feature for confidence model
        X_conf = np.column_stack([X_flat, dir_preds])

        self.confidence_model = xgb.XGBClassifier(**params)

        if X_val_flat is not None:
            dir_preds_val = self.direction_model.predict_proba(X_val_flat)[:, 1]
            X_val_conf = np.column_stack([X_val_flat, dir_preds_val])

            if y_val_confidence is not None:
                y_val_conf = (np.asarray(y_val_confidence).flatten() > 0.5).astype(np.int32)
            else:
                y_val_conf = ((dir_preds_val > 0.5) == y_val_dir).astype(np.int32)

            fit_params["eval_set"] = [(X_val_conf, y_val_conf)]

        self.confidence_model.fit(X_conf, y_conf, **fit_params)

        self.is_fitted = True

        # Calculate feature importance
        self._calculate_feature_importance()

        # Calculate metrics
        history = self._calculate_metrics(X_flat, y_dir, X_val_flat, y_val_dir)

        return history

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions.

        Args:
            X: Features (batch, seq_len, features) or (batch, features)

        Returns:
            Dict with 'direction' and 'confidence' predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_flat = self._flatten_sequences(X)

        # Direction prediction
        dir_proba = self.direction_model.predict_proba(X_flat)[:, 1]

        # Confidence prediction (using direction prediction as feature)
        X_conf = np.column_stack([X_flat, dir_proba])
        conf_proba = self.confidence_model.predict_proba(X_conf)[:, 1]

        return {
            "direction": dir_proba,
            "confidence": conf_proba,
        }

    def _calculate_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Calculate training and validation metrics."""
        history = {}

        # Training metrics
        train_preds = self.direction_model.predict_proba(X_train)[:, 1]
        train_acc = ((train_preds > 0.5) == y_train).mean()
        history["train_direction_accuracy"] = float(train_acc)

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_preds = self.direction_model.predict_proba(X_val)[:, 1]
            val_acc = ((val_preds > 0.5) == y_val).mean()
            history["val_direction_accuracy"] = float(val_acc)

        return history

    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        if self.direction_model is None:
            return

        importance = self.direction_model.feature_importances_

        # Create feature names if not set
        n_features = len(importance)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        self._feature_importance = dict(zip(feature_names, importance.tolist()))

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if self._feature_importance is None:
            return []

        sorted_features = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "config": self.config.__dict__,
            "direction_model": self.direction_model,
            "confidence_model": self.confidence_model,
            "feature_names": self.feature_names,
            "feature_importance": self._feature_importance,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        logger.info(f"Saved XGBoost model to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "XGBoostTradingModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        config = XGBoostConfig(**save_dict["config"])
        model = cls(config)
        model.direction_model = save_dict["direction_model"]
        model.confidence_model = save_dict["confidence_model"]
        model.feature_names = save_dict["feature_names"]
        model._feature_importance = save_dict["feature_importance"]
        model.is_fitted = save_dict["is_fitted"]

        return model


def train_xgboost_buddy(
    X_train: np.ndarray,
    y_train_direction: np.ndarray,
    y_train_confidence: np.ndarray,
    X_val: np.ndarray,
    y_val_direction: np.ndarray,
    y_val_confidence: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[XGBoostTradingModel, Dict[str, Any]]:
    """
    Train XGBoost model with same interface as TCN training.

    This function provides compatibility with the existing training pipeline.

    Args:
        X_train, y_train_*: Training data
        X_val, y_val_*: Validation data
        config: Optional config dict
        verbose: Print progress

    Returns:
        Tuple of (trained model, training history)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    # Build config from dict
    xgb_config = XGBoostConfig()
    if config:
        for key, value in config.items():
            if hasattr(xgb_config, key):
                setattr(xgb_config, key, value)

    # Create and train model
    model = XGBoostTradingModel(xgb_config)

    history = model.fit(
        X_train,
        y_train_direction,
        y_train_confidence,
        X_val,
        y_val_direction,
        y_val_confidence,
        verbose=verbose,
    )

    return model, history


class XGBoostKerasWrapper:
    """
    Wrapper to make XGBoost model compatible with Keras-style interface.

    This allows using XGBoost in the existing buddy training pipeline
    without major changes.
    """

    def __init__(self, xgb_model: XGBoostTradingModel):
        self.xgb_model = xgb_model
        self._is_built = True

    def __call__(self, inputs: np.ndarray, training: bool = False) -> Dict[str, np.ndarray]:
        """Make predictions (Keras-style call)."""
        return self.xgb_model.predict(inputs)

    def predict(self, X: np.ndarray, verbose: int = 0) -> Dict[str, np.ndarray]:
        """Keras-style predict method."""
        return self.xgb_model.predict(X)

    def save(self, path: str) -> None:
        """Save model."""
        # Convert .keras extension to .pkl for XGBoost
        path = str(path).replace(".keras", ".xgb.pkl")
        self.xgb_model.save(path)

    @classmethod
    def load(cls, path: str) -> "XGBoostKerasWrapper":
        """Load model."""
        path = str(path).replace(".keras", ".xgb.pkl")
        xgb_model = XGBoostTradingModel.load(path)
        return cls(xgb_model)

    @property
    def inputs(self):
        """Dummy inputs property for compatibility."""
        return None

    def get_layer(self, name: str):
        """Dummy get_layer for compatibility."""
        return


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing XGBoost Trading Model...")

    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    seq_len = 60
    n_features = 100

    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y_dir = (np.random.rand(n_samples) > 0.5).astype(np.float32)
    y_conf = (np.random.rand(n_samples) > 0.5).astype(np.float32)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_dir_train, y_dir_val = y_dir[:split], y_dir[split:]
    y_conf_train, y_conf_val = y_conf[:split], y_conf[split:]

    # Train
    model = XGBoostTradingModel()
    history = model.fit(
        X_train, y_dir_train, y_conf_train,
        X_val, y_dir_val, y_conf_val,
        verbose=True
    )

    print(f"\nTraining history: {history}")

    # Predict
    preds = model.predict(X_val[:5])
    print("\nSample predictions:")
    print(f"  Direction: {preds['direction']}")
    print(f"  Confidence: {preds['confidence']}")

    # Feature importance
    print(f"\nTop 10 features: {model.get_feature_importance(10)}")

    print("\nXGBoost model test complete!")
