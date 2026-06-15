"""
Meta-ensemble weight computation.

When two models have decorrelated errors, equal-weight voting is SUBOPTIMAL.
The optimal weight minimizes the combined error variance:

    w* = argmin Var(w * error_leg + (1-w) * error_sota)

Closed-form solution (Markowitz-style):
    w_leg = (var_sota - cov(leg, sota)) / (var_leg + var_sota - 2*cov(leg, sota))
    w_sota = 1 - w_leg

We also support:
  - Regime-specific weights (different blend per volatility regime)
  - Outcome-driven online updates (adjust weights after each trade)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model_comparison import BarPrediction

logger = logging.getLogger(__name__)


@dataclass
class MetaEnsemble:
    """Meta-learned ensemble weights for legacy + SOTA models."""

    pair: str
    # Global weights (fallback when regime-specific not available)
    legacy_weight: float = 0.5
    sota_weight: float = 0.5

    # Regime-specific weights: regime -> (legacy_w, sota_w)
    regime_weights: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Learning rate for online updates
    online_lr: float = 0.05
    min_weight: float = 0.1
    max_weight: float = 0.9

    # History for diagnostics
    weight_history: List[Dict[str, Any]] = field(default_factory=list)

    def predict(
        self,
        legacy_pred: BarPrediction,
        sota_pred: BarPrediction,
        regime: str = "NORMAL",
    ) -> BarPrediction:
        """Blend two predictions into a single meta-prediction.

        Blending logic:
          - If both agree on direction: use the MORE confident one
          - If they disagree: weighted average of raw scores
          - Direction threshold: blended score >= 0.55 → LONG, <= 0.45 → SHORT
        """
        leg_w, sota_w = self.regime_weights.get(regime, (self.legacy_weight, self.sota_weight))

        # Normalize weights to sum to 1
        total = leg_w + sota_w
        if total < 1e-6:
            leg_w = sota_w = 0.5
        else:
            leg_w /= total
            sota_w /= total

        # Convert directions to signed scores [-1, 1]
        leg_score = 1.0 if legacy_pred.direction == "LONG" else (-1.0 if legacy_pred.direction == "SHORT" else 0.0)
        sota_score = 1.0 if sota_pred.direction == "LONG" else (-1.0 if sota_pred.direction == "SHORT" else 0.0)

        # Weighted blend
        blended_score = leg_w * leg_score + sota_w * sota_score

        if blended_score >= 0.55:
            direction = "LONG"
            confidence = min(blended_score, 1.0)
        elif blended_score <= -0.55:
            direction = "SHORT"
            confidence = min(abs(blended_score), 1.0)
        else:
            direction = "HOLD"
            confidence = abs(blended_score)

        return BarPrediction(
            timestamp=legacy_pred.timestamp,
            direction=direction,
            confidence=confidence,
            score=(blended_score + 1.0) / 2.0,  # map back to [0, 1]
        )

    def update_from_outcome(
        self,
        legacy_pred: BarPrediction,
        sota_pred: BarPrediction,
        actual_ret: float,
        regime: str = "NORMAL",
    ) -> None:
        """Online gradient-based weight update after a trade closes.

        Reward the model that was RIGHT, penalize the one that was WRONG.
        This is a multiplicative weights update (Hedge algorithm).
        """
        leg_correct = (legacy_pred.direction == "LONG" and actual_ret > 0) or \
                      (legacy_pred.direction == "SHORT" and actual_ret < 0)
        sota_correct = (sota_pred.direction == "LONG" and actual_ret > 0) or \
                       (sota_pred.direction == "SHORT" and actual_ret < 0)

        # Multiplicative update
        if leg_correct and not sota_correct:
            leg_w, sota_w = self.regime_weights.get(regime, (self.legacy_weight, self.sota_weight))
            new_leg = min(leg_w * (1 + self.online_lr), self.max_weight)
            new_sota = max(1.0 - new_leg, self.min_weight)
            self.regime_weights[regime] = (new_leg, new_sota)
        elif sota_correct and not leg_correct:
            leg_w, sota_w = self.regime_weights.get(regime, (self.legacy_weight, self.sota_weight))
            new_sota = min(sota_w * (1 + self.online_lr), self.max_weight)
            new_leg = max(1.0 - new_sota, self.min_weight)
            self.regime_weights[regime] = (new_leg, new_sota)

        self.weight_history.append({
            "regime": regime,
            "legacy_correct": leg_correct,
            "sota_correct": sota_correct,
            "weights": {"legacy": self.regime_weights.get(regime, (self.legacy_weight, self.sota_weight))[0], "sota": self.regime_weights.get(regime, (self.legacy_weight, self.sota_weight))[1]},
        })

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pair": self.pair,
            "legacy_weight": self.legacy_weight,
            "sota_weight": self.sota_weight,
            "regime_weights": {k: list(v) for k, v in self.regime_weights.items()},
            "weight_history": self.weight_history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetaEnsemble":
        with open(path) as f:
            data = json.load(f)
        me = cls(pair=data["pair"])
        me.legacy_weight = data.get("legacy_weight", 0.5)
        me.sota_weight = data.get("sota_weight", 0.5)
        me.regime_weights = {k: tuple(v) for k, v in data.get("regime_weights", {}).items()}
        me.weight_history = data.get("weight_history", [])
        return me


def compute_meta_weights(
    legacy_errors: np.ndarray,
    sota_errors: np.ndarray,
    min_weight: float = 0.1,
    max_weight: float = 0.9,
) -> Tuple[float, float]:
    """Compute optimal static weights that minimize combined error variance.

    Args:
        legacy_errors: array of {1=correct, 0=wrong} for legacy model
        sota_errors:   array of {1=correct, 0=wrong} for SOTA model

    Returns:
        (legacy_weight, sota_weight) that minimize Var(w*e_leg + (1-w)*e_sota)
    """
    if len(legacy_errors) < 20 or len(sota_errors) < 20:
        return 0.5, 0.5

    # Errors as {-1, +1} for variance computation
    leg_e = 2.0 * legacy_errors - 1.0
    sota_e = 2.0 * sota_errors - 1.0

    var_leg = float(np.var(leg_e))
    var_sota = float(np.var(sota_e))
    cov = float(np.cov(leg_e, sota_e)[0, 1])

    denom = var_leg + var_sota - 2 * cov
    if abs(denom) < 1e-8:
        return 0.5, 0.5

    w_leg = (var_sota - cov) / denom
    w_leg = max(min_weight, min(max_weight, w_leg))
    w_sota = 1.0 - w_leg

    logger.info(f"Meta-weights: legacy={w_leg:.3f}, sota={w_sota:.3f} (cov={cov:.3f})")
    return w_leg, w_sota
