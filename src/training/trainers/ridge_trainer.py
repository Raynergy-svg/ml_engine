"""
Ridge Trainer for Confidence/Stability Scoring.

UPGRADED: Now uses LightGBM instead of ElasticNetCV for better accuracy.
Falls back to ElasticNetCV if LightGBM is not installed.

Features:
- GPU acceleration with CPU fallback
- Better handling of non-linear relationships
- Faster training than ElasticNetCV
- Backward compatible loading (handles both old ElasticNet and new LightGBM)

Input features: Rolling variance, volume changes, technical indicators
Output: Confidence/stability score (0-100) for Gate 2 in the ensemble system
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

# Import custom LR schedules early so they're registered with Keras before
# any pickle deserialization that might contain Keras models with these schedules
try:
    from src.training.m1_metal_optimizer import (
        WarmupCosineDecaySchedule,  # noqa: F401
        CosineDecayRestarts,  # noqa: F401
    )
except ImportError:
    pass  # Not required for ridge training, only for loading models that use them

logger = logging.getLogger(__name__)

# Constants
MODEL_NOT_TRAINED_ERROR = "Model not trained"
UNPICKLE_ESTIMATOR_WARNING = ".*unpickle estimator.*"


def _create_lgbm_regressor(**kwargs) -> Any:
    """
    Create LightGBM regressor with GPU→CPU fallback.

    Tries GPU first (faster), falls back to CPU if unavailable.
    GPU uses smaller max_bin for speedup.

    Args:
        **kwargs: Additional LGBMRegressor parameters

    Returns:
        Configured LGBMRegressor instance
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.warning("LightGBM not installed, falling back to ElasticNetCV")
        return None

    import platform

    # Default hyperparameters optimized for confidence scoring
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "num_leaves": 31,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }
    default_params.update(kwargs)

    # On macOS, LightGBM GPU is not supported (Metal not supported, only CUDA/OpenCL)
    # Skip GPU attempts to avoid [Fatal] error messages
    is_macos = platform.system() == "Darwin"
    devices_to_try = ["cpu"] if is_macos else ["gpu", "cuda", "cpu"]

    for device in devices_to_try:
        try:
            params = default_params.copy()
            params["device"] = device
            # Smaller bins for GPU speedup
            params["max_bin"] = 63 if device != "cpu" else 255

            model = LGBMRegressor(**params)

            # Test fit with tiny data to verify device works
            import numpy as np

            rng = np.random.default_rng(42)
            x_test = rng.random((10, 5)).astype(np.float32)
            y_test = rng.random(10).astype(np.float32)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_test, y_test)

            logger.info(f"✓ LightGBM using device: {device}")

            # Return fresh model (the test model is discarded)
            return LGBMRegressor(**params)

        except Exception as e:
            if device != "cpu":
                logger.debug(f"LightGBM {device} unavailable: {e}")
            continue

    # Final fallback - CPU without test
    logger.info("⚠ LightGBM falling back to CPU (no GPU available)")
    params = default_params.copy()
    params["device"] = "cpu"
    params["max_bin"] = 255
    return LGBMRegressor(**params)


class RidgeTrainer(BaseTrainer):
    """
    LightGBM regression model for confidence/stability scoring.

    UPGRADED: Now uses LightGBM instead of ElasticNetCV for better accuracy.
    Falls back to ElasticNetCV if LightGBM is not installed.

    Features:
    - GPU acceleration with CPU fallback
    - Better handling of non-linear relationships
    - Faster training than ElasticNetCV
    - Backward compatible loading (handles both old ElasticNet and new LightGBM)

    Input: Rolling variance, volume changes, technical indicators
    Output: Confidence/stability score (0-100)
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.best_alpha = None
        self.best_l1_ratio = None
        self.n_nonzero_coefs = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        label_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Train LightGBM for confidence scoring (falls back to ElasticNet if unavailable).

        Args:
            label_metadata: optional dict from `compute_realized_confidence_labels`
                with `n_real`, `n_pseudo`, `class_balance_*` etc. Stored in
                `self.metrics` for downstream visibility and used for leak
                detection (R² > expected band → ERROR log).
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # === FIX: Convert to DataFrame with feature names ===
        # This ensures LightGBM stores feature names consistently to prevent
        # sklearn UserWarning: "X does not have valid feature names"
        if self.feature_names is not None:
            x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
            x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)
        else:
            # Fallback: generate default feature names
            self.feature_names = [
                f"feature_{i}" for i in range(x_train_scaled.shape[1])
            ]
            x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
            x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)

        # Try LightGBM first (GPU-accelerated)
        lgbm_model = _create_lgbm_regressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
        )

        if lgbm_model is not None:
            # Use LightGBM
            logger.info("Training LightGBM (Confidence) - GPU-accelerated...")

            self.model = lgbm_model
            self.model.fit(
                x_train_df,
                y_train,  # Use DataFrame with feature names
                eval_set=[(x_val_df, y_val)],  # Use DataFrame with feature names
            )

            self.is_trained = True
            self._model_type = "lightgbm"

            # Calculate metrics - use DataFrame to avoid feature name warning
            y_pred = self.model.predict(x_val_df)
            mae = float(np.mean(np.abs(y_pred - y_val)))

            # R² score
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Get feature importances
            importances = self.model.feature_importances_
            n_important = int(np.sum(importances > 0.01))

            # Realized-outcome label metrics (leak-fix 2026-04-30).
            # Expected R² band on real outcome labels is [0.05, 0.30] for
            # noisy financial regression — a higher value triggers the
            # leak-detection ERROR below (logged, not raised, so legitimate
            # edge cases can be investigated rather than crashing training).
            expected_r2_band = [0.05, 0.30]
            label_meta = dict(label_metadata or {})

            # Class balance sanity (for binary-rescaled labels)
            unique_y = np.unique(np.round(y_train, 2))
            class_balance = None
            try:
                # If labels are binary {20, 95}, compute fraction at win-side
                if len(unique_y) <= 2:
                    class_balance = float(np.mean(y_train > 50.0))
            except Exception:
                pass

            self.metrics = {
                "confidence_mae": mae,
                "r2_score": float(r2),
                "model_type": "lightgbm",
                "n_estimators": self.model.n_estimators,
                "n_important_features": n_important,
                "n_total_features": self.n_features,
                "label_mode": label_meta.get("label_mode"),
                "n_real_labels": label_meta.get("n_real"),
                "n_pseudo_labels": label_meta.get("n_pseudo"),
                "class_balance_real": label_meta.get("class_balance_real"),
                "class_balance_pseudo": label_meta.get("class_balance_pseudo"),
                "class_balance": class_balance,
                "expected_r2_band": expected_r2_band,
                "score_range": label_meta.get("score_range"),
                "score_formula": label_meta.get("score_formula"),
                "leak_fix_version": label_meta.get("leak_fix_version"),
            }

            # LEAK-DETECTION: log loud ERROR if R² blows past expected band.
            # We log rather than raise so legitimate edge cases (e.g. tiny
            # synthetic test set) don't crash training. The 2026-04-30 leak
            # showed R²=0.997 — anything > 0.30 on this label is suspect.
            if r2 > expected_r2_band[1]:
                logger.error(
                    "SUSPECTED LABEL LEAK: confidence R²=%.4f exceeds expected "
                    "band %s for realized-outcome label. Investigate features "
                    "and label generator for closed-form derivability. See "
                    "docs/confidence_model_leak_investigation_2026-04-30.md.",
                    r2, expected_r2_band,
                )

            logger.info(
                f"LightGBM trained: MAE={mae:.2f}, R²={r2:.4f}, "
                f"important_features={n_important}/{self.n_features}, "
                f"label_mode={label_meta.get('label_mode')}, "
                f"n_real={label_meta.get('n_real')}, n_pseudo={label_meta.get('n_pseudo')}"
            )
            return self.metrics

        # Fallback to ElasticNetCV
        from sklearn.linear_model import ElasticNetCV

        logger.info("Training ElasticNet (Confidence) with TimeSeriesSplit CV...")

        # Configure TimeSeriesSplit for temporal CV (prevents leakage)
        tscv = TimeSeriesSplit(n_splits=self.config.elasticnet_cv_splits)

        # Auto-generate alphas if not specified
        alphas = self.config.elasticnet_alphas
        if alphas is None:
            alphas = np.logspace(-4, 2, 50).tolist()

        # Train ElasticNetCV with automatic hyperparameter tuning
        self.model = ElasticNetCV(
            l1_ratio=self.config.elasticnet_l1_ratios,
            alphas=alphas,
            cv=tscv,
            max_iter=self.config.elasticnet_max_iter,
            n_jobs=-1,  # Parallel CV
            selection="random",  # Faster convergence
        )
        self.model.fit(x_train_scaled, y_train)

        self.is_trained = True
        self._model_type = "elasticnet"

        # Extract best hyperparameters
        self.best_alpha = float(self.model.alpha_)
        self.best_l1_ratio = float(self.model.l1_ratio_)
        self.n_nonzero_coefs = int(np.sum(self.model.coef_ != 0))

        # Calculate metrics
        y_pred = self.model.predict(x_val_scaled)
        mae = float(np.mean(np.abs(y_pred - y_val)))
        r2 = float(self.model.score(x_val_scaled, y_val))

        # Same metadata pass-through as the LightGBM path (leak-fix 2026-04-30).
        expected_r2_band = [0.05, 0.30]
        label_meta = dict(label_metadata or {})
        self.metrics = {
            "confidence_mae": mae,
            "r2_score": r2,
            "model_type": "elasticnet",
            "best_alpha": self.best_alpha,
            "best_l1_ratio": self.best_l1_ratio,
            "n_nonzero_coefs": self.n_nonzero_coefs,
            "n_total_coefs": self.n_features,
            "sparsity_ratio": 1.0 - (self.n_nonzero_coefs / self.n_features)
            if self.n_features > 0
            else 0.0,
            "label_mode": label_meta.get("label_mode"),
            "n_real_labels": label_meta.get("n_real"),
            "n_pseudo_labels": label_meta.get("n_pseudo"),
            "class_balance_real": label_meta.get("class_balance_real"),
            "class_balance_pseudo": label_meta.get("class_balance_pseudo"),
            "expected_r2_band": expected_r2_band,
            "score_range": label_meta.get("score_range"),
            "score_formula": label_meta.get("score_formula"),
            "leak_fix_version": label_meta.get("leak_fix_version"),
        }
        if r2 > expected_r2_band[1]:
            logger.error(
                "SUSPECTED LABEL LEAK (ElasticNet path): confidence R²=%.4f "
                "exceeds expected band %s.", r2, expected_r2_band,
            )

        logger.info(
            f"ElasticNet trained: MAE={mae:.2f}, R²={r2:.4f}, "
            f"alpha={self.best_alpha:.4f}, l1_ratio={self.best_l1_ratio:.2f}, "
            f"sparse={self.n_nonzero_coefs}/{self.n_features} features"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict confidence score (0-100)."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        import pandas as pd

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # === FIX: Always use DataFrame for LightGBM ===
        # This prevents sklearn UserWarning: "X does not have valid feature names"
        # The model was trained with DataFrames that have feature names,
        # so predictions must also use DataFrames with the same feature names.
        model_type = getattr(self, "_model_type", "elasticnet")
        if model_type == "lightgbm":
            # Ensure feature_names are available
            if self.feature_names is None:
                # Generate default feature names if not set
                self.feature_names = [f"feature_{i}" for i in range(x_last.shape[1])]

            # Convert to DataFrame with feature names
            x_last = pd.DataFrame(x_last, columns=self.feature_names)

        confidence = float(self.model.predict(x_last)[0])

        # Clamp to 0-100
        confidence = max(0.0, min(100.0, confidence))

        return {
            "confidence": confidence,
        }

    def save(self, path: str) -> None:
        """Save confidence model with version metadata."""
        import sklearn

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Detect model type
        model_type = getattr(self, "_model_type", "elasticnet")
        if model_type == "lightgbm":
            try:
                import lightgbm

                lgbm_version = lightgbm.__version__
            except ImportError:
                lgbm_version = "unknown"
        else:
            lgbm_version = None

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            # Model type and version metadata
            "model_type": model_type,
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgbm_version,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Confidence model ({model_type}) saved to {path}")

    def load(self, path: str) -> None:
        """Load confidence model (supports both LightGBM and ElasticNet)."""
        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", message=UNPICKLE_ESTIMATOR_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        # Detect model type from saved data or model class
        self._model_type = data.get("model_type")
        if self._model_type is None:
            # Infer from model class name
            model_class = type(self.model).__name__
            if "LGBM" in model_class or "LightGBM" in model_class:
                self._model_type = "lightgbm"
            else:
                self._model_type = "elasticnet"

        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get("sklearn_version")
        self._saved_lightgbm_version = data.get("lightgbm_version")
        self._saved_at = data.get("saved_at")

        logger.info(f"Confidence model ({self._model_type}) loaded from {path}")
