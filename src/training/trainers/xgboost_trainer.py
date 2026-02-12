"""
XGBoost Trainer for Momentum Analysis.

This trainer uses XGBoost models to predict:
- Momentum score (0-1): How fast price is moving
- Acceleration (bool): Whether momentum is growing

Input features: Lagged returns + spread dynamics
Output: Two predictions for Gate 3 in the ensemble system
"""

from __future__ import annotations

import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig

logger = logging.getLogger(__name__)

# Constants
MODEL_NOT_TRAINED_ERROR = "Model not trained"
SERIALIZED_MODEL_WARNING = ".*serialized model.*"


class XGBoostTrainer(BaseTrainer):
    """
    XGBoost model for momentum analysis.

    Input: Lagged returns + spread dynamics
    Output:
        - momentum_score (0-1): How fast price is moving
        - acceleration (bool): Is momentum growing?
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.momentum_model = None
        self.accel_model = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.momentum_norm_factor = None  # Saved from training for reference

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        momentum_norm_factor: Optional[float] = None,
    ) -> Dict[str, float]:
        """Train XGBoost for momentum analysis (2 models) with GPU support."""
        self.momentum_norm_factor = momentum_norm_factor
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        from sklearn.preprocessing import StandardScaler

        # Detect GPU availability
        use_gpu = self.config.use_gpu if hasattr(self.config, "use_gpu") else False
        tree_method = "gpu_hist" if use_gpu else "auto"

        logger.info(
            f"Training XGBoost (Momentum) - GPU: {use_gpu}, tree_method: {tree_method}"
        )

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # Split targets: y[:, 0] = momentum_score, y[:, 1] = acceleration
        y_train_momentum = y_train[:, 0]
        y_train_accel = y_train[:, 1].astype(int)
        y_val_momentum = y_val[:, 0]
        y_val_accel = y_val[:, 1].astype(int)

        # Train momentum regressor with GPU acceleration
        self.momentum_model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            tree_method=tree_method,  # GPU-accelerated if available
            predictor="gpu_predictor" if use_gpu else "auto",
            verbosity=0,
            n_jobs=-1 if not use_gpu else 1,  # GPU doesn't need multi-threading
            random_state=42,
        )
        self.momentum_model.fit(
            x_train_scaled,
            y_train_momentum,
            eval_set=[(x_val_scaled, y_val_momentum)],
            verbose=False,
        )

        # Train acceleration classifier with GPU acceleration
        self.accel_model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            tree_method=tree_method,  # GPU-accelerated if available
            predictor="gpu_predictor" if use_gpu else "auto",
            verbosity=0,
            n_jobs=-1 if not use_gpu else 1,  # GPU doesn't need multi-threading
            random_state=42,
        )
        self.accel_model.fit(
            x_train_scaled,
            y_train_accel,
            eval_set=[(x_val_scaled, y_val_accel)],
            verbose=False,
        )

        self.is_trained = True

        # Calculate metrics
        momentum_pred = self.momentum_model.predict(x_val_scaled)
        accel_pred = self.accel_model.predict(x_val_scaled)

        momentum_mae = float(np.mean(np.abs(momentum_pred - y_val_momentum)))
        accel_acc = float(np.mean(accel_pred == y_val_accel))

        self.metrics = {
            "momentum_mae": momentum_mae,
            "acceleration_accuracy": accel_acc,
        }

        logger.info(
            f"XGBoost trained: momentum_mae={momentum_mae:.4f}, accel_acc={accel_acc:.4f}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict momentum score and acceleration."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # DEBUG: Log scaled features once
        if not hasattr(self, "_predict_debug_logged"):
            logger.info(
                f"XGB Predict - Input shape: {X.shape}, Scaled shape: {x_scaled.shape}"
            )
            logger.info(f"XGB Predict - x_last scaled: {x_last.flatten()}")
            self._predict_debug_logged = True

        momentum = float(self.momentum_model.predict(x_last)[0])
        acceleration = bool(self.accel_model.predict(x_last)[0])

        # Clamp momentum to 0-1
        momentum = max(0.0, min(1.0, momentum))

        return {
            "momentum": momentum,
            "acceleration": acceleration,
        }

    def save(self, path: str) -> None:
        """Save XGBoost models with version metadata."""
        import sklearn
        import xgboost

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "momentum_model": self.momentum_model,
            "accel_model": self.accel_model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "momentum_norm_factor": self.momentum_norm_factor,
            # Version metadata for compatibility checks
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(
            f"XGBoost saved to {path} (sklearn={sklearn.__version__}, xgboost={xgboost.__version__})"
        )

    def load(self, path: str) -> None:
        """Load XGBoost models."""
        import warnings

        # Suppress XGBoost version warnings (common when loading older serialized models)
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)
        warnings.filterwarnings("ignore", message=".*older version of XGBoost.*")

        with open(path, "rb") as f:
            data = pickle.load(f)

        # DEBUG: Log scaler info
        scaler = data.get("scaler")
        if scaler is not None:
            logger.info(f"XGB Scaler - mean_: {scaler.mean_}")
            logger.info(f"XGB Scaler - scale_: {scaler.scale_}")

        self.momentum_model = data["momentum_model"]
        self.accel_model = data["accel_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get("sklearn_version")
        self._saved_xgboost_version = data.get("xgboost_version")
        self._saved_at = data.get("saved_at")

        logger.info(f"XGBoost loaded from {path}")
