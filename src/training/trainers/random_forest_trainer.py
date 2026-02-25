"""
Random Forest Trainer for Risk Assessment.

This trainer uses Random Forest models to predict:
- Expected drawdown percentage: Maximum adverse excursion in next N bars
- Streak probability: Probability that a losing streak continues

Input features: ATR, historical drawdowns, streak patterns
Output: Two risk metrics for Gate 4 in the ensemble system
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
    pass

logger = logging.getLogger(__name__)

# Constants
MODEL_NOT_TRAINED_ERROR = "Model not trained"
UNPICKLE_ESTIMATOR_WARNING = ".*unpickle estimator.*"


class RandomForestTrainer(BaseTrainer):
    """
    Random Forest model for risk assessment.

    Input: ATR, historical drawdowns, streak patterns
    Output:
        - expected_drawdown_pips: Max adverse excursion in next N bars
        - streak_prob: Probability losing streak continues
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.drawdown_model = None
        self.streak_model = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """Train Random Forest for risk assessment (2 models)."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        logger.info("Training Random Forest (Risk)...")

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # Split targets: y[:, 0] = drawdown_pips, y[:, 1] = streak_prob
        y_train_drawdown = y_train[:, 0]
        y_train_streak = y_train[:, 1]
        y_val_drawdown = y_val[:, 0]
        y_val_streak = y_val[:, 1]

        # Train drawdown regressor
        self.drawdown_model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
        self.drawdown_model.fit(x_train_scaled, y_train_drawdown)

        # Train streak probability regressor (0-1 range, so regression not classification)
        self.streak_model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
        self.streak_model.fit(x_train_scaled, y_train_streak)

        self.is_trained = True

        # Calculate metrics
        drawdown_pred = self.drawdown_model.predict(x_val_scaled)
        streak_pred = self.streak_model.predict(x_val_scaled)

        # Clip predictions to valid range (0-10% for drawdown, 0-1 for streak)
        drawdown_pred = np.clip(drawdown_pred, 0, 0.10)
        streak_pred = np.clip(streak_pred, 0, 1.0)

        drawdown_mae = float(np.mean(np.abs(drawdown_pred - y_val_drawdown)))
        streak_mae = float(np.mean(np.abs(streak_pred - y_val_streak)))

        # Convert to basis points for meaningful display (0.001 = 10 bps)
        drawdown_mae_bps = drawdown_mae * 10000

        # === PHASE 4: MAE TARGET TRACKING ===
        # Target: Drawdown MAE < 10 bps (0.001 in decimal)
        target_bps = 10.0
        target_achieved = drawdown_mae_bps <= target_bps
        target_gap_bps = max(0, drawdown_mae_bps - target_bps)

        self.metrics = {
            "drawdown_mae_pct": drawdown_mae,  # Raw percentage (0-1)
            "drawdown_mae_bps": drawdown_mae_bps,  # Basis points for display
            "streak_prob_mae": streak_mae,
            # Phase 4 target tracking
            "target_achieved": target_achieved,
            "target_gap_bps": target_gap_bps,
        }

        logger.info(
            f"RF trained: drawdown_mae={drawdown_mae_bps:.1f} bps ({drawdown_mae * 100:.3f}%), "
            f"streak_mae={streak_mae:.4f}"
        )
        
        if target_achieved:
            logger.info(f"✅ Phase 4 Target ACHIEVED: Drawdown MAE {drawdown_mae_bps:.1f} bps ≤ {target_bps} bps")
        else:
            logger.warning(
                f"⚠️ Phase 4 Target NOT MET: Drawdown MAE {drawdown_mae_bps:.1f} bps exceeds {target_bps} bps "
                f"by {target_gap_bps:.1f} bps. Consider hyperparameter tuning."
            )
        
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict expected drawdown and streak probability."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # Model outputs drawdown as PERCENTAGE (instrument-agnostic)
        expected_drawdown_pct = float(self.drawdown_model.predict(x_last)[0])
        streak_prob = float(self.streak_model.predict(x_last)[0])

        # Clamp values to realistic ranges
        expected_drawdown_pct = max(
            0.0, min(0.10, expected_drawdown_pct)
        )  # 0-10% max drawdown
        streak_prob = max(0.0, min(1.0, streak_prob))  # 0-100% probability

        return {
            "expected_drawdown_pct": expected_drawdown_pct,
            # Keep legacy key for backward compatibility
            "expected_drawdown_pips": expected_drawdown_pct
            * 10000,  # Rough conversion for display
            "streak_prob": streak_prob,
        }

    def save(self, path: str) -> None:
        """Save Random Forest models with version metadata."""
        import sklearn

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "drawdown_model": self.drawdown_model,
            "streak_model": self.streak_model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            # Version metadata for compatibility checks
            "sklearn_version": sklearn.__version__,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Random Forest saved to {path} (sklearn={sklearn.__version__})")

    def load(self, path: str) -> None:
        """Load Random Forest models."""
        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", message=UNPICKLE_ESTIMATOR_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.drawdown_model = data["drawdown_model"]
        self.streak_model = data["streak_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get("sklearn_version")
        self._saved_at = data.get("saved_at")

        logger.info(f"Random Forest loaded from {path}")
