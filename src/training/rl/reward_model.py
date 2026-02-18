"""Learned reward modeling for RL integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RewardModelConfig:
    """Configuration for reward model training."""

    use_ensemble_predictions: bool = True
    include_risk_metrics: bool = True
    reward_horizon_bars: int = 24
    normalize_rewards: bool = True
    alpha: float = 1.0


class EnsembleRewardModel:
    """
    Small supervised model that predicts risk-adjusted reward targets.

    Uses Ridge regression for stability and interpretability.
    """

    def __init__(self, config: Optional[RewardModelConfig] = None):
        self.config = config or RewardModelConfig()
        self.model = None

    def _construct_features(self, ensemble_predictions: np.ndarray) -> np.ndarray:
        features = np.asarray(ensemble_predictions, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return features

    def train(
        self,
        ensemble_predictions: np.ndarray,
        actual_returns: np.ndarray,
        risk_adjusted_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train reward model on ensemble outputs."""
        from sklearn.linear_model import Ridge

        x = self._construct_features(ensemble_predictions)
        y = (
            np.asarray(risk_adjusted_returns, dtype=np.float32)
            if risk_adjusted_returns is not None
            else np.asarray(actual_returns, dtype=np.float32)
        )
        if self.config.normalize_rewards:
            std = float(np.std(y))
            if std > 1e-12:
                y = (y - float(np.mean(y))) / std

        model = Ridge(alpha=self.config.alpha)
        model.fit(x, y)
        self.model = model
        return {"r2_score": float(model.score(x, y))}

    def predict_reward(self, ensemble_predictions: np.ndarray) -> float:
        """Predict a scalar reward signal for current ensemble output."""
        if self.model is None:
            return 0.0
        x = self._construct_features(ensemble_predictions)
        pred = self.model.predict(x)
        return float(pred[0])

