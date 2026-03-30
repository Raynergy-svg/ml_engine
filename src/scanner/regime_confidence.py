"""
Regime Confidence Scorer — Phase 54 (US-338).

Scales position sizing by regime classification certainty.
Uses ADX + ATR percentile + trend strength to estimate regime probability.

Confidence tiers:
  - High (>0.75): 100% position size
  - Medium (0.50-0.75): 80% position size
  - Low (<0.50): 60% position size
  - Transition (max_prob drop >15%): 50% for 2 cycles

Wires into calculate_position_size() AFTER adaptive sizing, BEFORE EWMA.

Usage:
    scorer = RegimeConfidenceScorer()
    result = scorer.score(adx=25.0, atr_percentile=0.6, trend_strength=0.7)
    sizing_factor = result.sizing_factor
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PERSISTENCE_PATH = "trained_data/regime_confidence_history.json"


@dataclass
class RegimeScoreResult:
    """Result of regime confidence scoring."""
    regime_confidence: float = 0.5      # max(P(bull), P(bear), P(sideways))
    sizing_factor: float = 1.0          # Multiplier for position size
    tier: str = "medium"                # "high", "medium", "low", "transition"
    probabilities: Dict[str, float] = field(default_factory=dict)
    is_transition: bool = False
    transition_cycles_remaining: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_confidence": round(self.regime_confidence, 4),
            "sizing_factor": round(self.sizing_factor, 4),
            "tier": self.tier,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "is_transition": self.is_transition,
            "transition_cycles_remaining": self.transition_cycles_remaining,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeScoreResult":
        return cls(
            regime_confidence=float(d.get("regime_confidence", 0.5)),
            sizing_factor=float(d.get("sizing_factor", 1.0)),
            tier=str(d.get("tier", "medium")),
            probabilities=d.get("probabilities", {}),
            is_transition=bool(d.get("is_transition", False)),
            transition_cycles_remaining=int(d.get("transition_cycles_remaining", 0)),
            timestamp=str(d.get("timestamp", "")),
        )


@dataclass
class RegimeConfidenceConfig:
    """Configuration for regime confidence scorer."""
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.50
    high_sizing_factor: float = 1.0
    medium_sizing_factor: float = 0.80
    low_sizing_factor: float = 0.60
    transition_sizing_factor: float = 0.50
    transition_drop_threshold: float = 0.15   # Max prob drop to trigger transition
    transition_cycles: int = 2                 # Cycles to hold transition sizing
    max_history: int = 50
    persistence_path: str = _DEFAULT_PERSISTENCE_PATH


class RegimeConfidenceScorer:
    """Scores regime classification confidence and scales position sizing."""

    def __init__(self, config: Optional[RegimeConfidenceConfig] = None):
        self.config = config or RegimeConfidenceConfig()
        self._last_max_prob: Optional[float] = None
        self._transition_cycles_remaining: int = 0
        self._history: List[RegimeScoreResult] = []
        self._load_state()

    def score(
        self,
        adx: float = 25.0,
        atr_percentile: float = 0.5,
        trend_strength: float = 0.5,
        current_regime: str = "NORMAL",
    ) -> RegimeScoreResult:
        """Score regime confidence and compute sizing factor.

        Args:
            adx: Average Directional Index (0-100)
            atr_percentile: ATR percentile rank (0-1)
            trend_strength: Trend strength indicator (0-1)
            current_regime: Current regime classification

        Returns:
            RegimeScoreResult with confidence and sizing factor
        """
        now = datetime.now(timezone.utc).isoformat()

        # Estimate regime probabilities from indicators
        probs = self._estimate_probabilities(adx, atr_percentile, trend_strength)
        max_prob = max(probs.values())

        # Check for regime transition
        is_transition = False
        if self._last_max_prob is not None:
            drop = self._last_max_prob - max_prob
            if drop > self.config.transition_drop_threshold:
                self._transition_cycles_remaining = self.config.transition_cycles
                is_transition = True
                logger.info(
                    "Phase 54 (US-338): Regime transition detected — "
                    "prob drop %.3f→%.3f (>%.3f threshold), "
                    "reducing sizing for %d cycles",
                    self._last_max_prob, max_prob,
                    self.config.transition_drop_threshold,
                    self.config.transition_cycles,
                )

        self._last_max_prob = max_prob

        # Determine tier and sizing factor
        if self._transition_cycles_remaining > 0:
            tier = "transition"
            sizing_factor = self.config.transition_sizing_factor
            self._transition_cycles_remaining -= 1
        elif max_prob >= self.config.high_confidence_threshold:
            tier = "high"
            sizing_factor = self.config.high_sizing_factor
        elif max_prob >= self.config.medium_confidence_threshold:
            tier = "medium"
            sizing_factor = self.config.medium_sizing_factor
        else:
            tier = "low"
            sizing_factor = self.config.low_sizing_factor

        result = RegimeScoreResult(
            regime_confidence=max_prob,
            sizing_factor=sizing_factor,
            tier=tier,
            probabilities=probs,
            is_transition=is_transition,
            transition_cycles_remaining=self._transition_cycles_remaining,
            timestamp=now,
        )

        self._history.append(result)
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]

        logger.info(
            "Phase 54 (US-338): Regime confidence=%.3f tier=%s sizing=%.2f "
            "probs=%s transition_remaining=%d",
            max_prob, tier, sizing_factor,
            {k: f"{v:.3f}" for k, v in probs.items()},
            self._transition_cycles_remaining,
        )

        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """Get scoring history."""
        return [r.to_dict() for r in self._history]

    def get_status(self) -> Dict[str, Any]:
        """Get current scorer status."""
        return {
            "last_max_prob": round(self._last_max_prob, 4) if self._last_max_prob is not None else None,
            "transition_cycles_remaining": self._transition_cycles_remaining,
            "history_length": len(self._history),
        }

    def _estimate_probabilities(
        self, adx: float, atr_percentile: float, trend_strength: float
    ) -> Dict[str, float]:
        """Estimate regime probability distribution from indicators.

        Uses a simple heuristic combination:
        - High ADX + high trend → bullish/bearish (trending)
        - Low ADX + low trend → sideways
        - High ATR → more volatile regime uncertainty
        """
        # Normalize ADX to 0-1 range (typical range 10-60)
        adx_norm = min(1.0, max(0.0, (adx - 10) / 50))

        # Trending probability = ADX + trend_strength blend
        p_trending = 0.6 * adx_norm + 0.4 * trend_strength
        p_trending = min(1.0, max(0.0, p_trending))

        # Sideways is complement of trending
        p_sideways = 1.0 - p_trending

        # Split trending into bull/bear based on trend_strength sign
        # (trend_strength > 0.5 = bullish, < 0.5 = bearish)
        bull_bias = trend_strength  # 0-1 range
        p_bull = p_trending * bull_bias
        p_bear = p_trending * (1.0 - bull_bias)

        # ATR adds uncertainty (reduces max probability)
        uncertainty = atr_percentile * 0.2  # 0-0.2 range
        total = p_bull + p_bear + p_sideways
        if total > 0:
            p_bull = p_bull / total * (1.0 - uncertainty)
            p_bear = p_bear / total * (1.0 - uncertainty)
            p_sideways = p_sideways / total * (1.0 - uncertainty) + uncertainty

        # Re-normalize
        total = p_bull + p_bear + p_sideways
        if total > 0:
            p_bull /= total
            p_bear /= total
            p_sideways /= total

        return {
            "bull": round(p_bull, 4),
            "bear": round(p_bear, 4),
            "sideways": round(p_sideways, 4),
        }

    def save_state(self) -> None:
        """Persist scorer state."""
        path = self.config.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_max_prob": self._last_max_prob,
                "transition_cycles_remaining": self._transition_cycles_remaining,
                "history": [r.to_dict() for r in self._history[-self.config.max_history:]],
            }
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, path)
        except Exception as e:
            logger.error("Phase 54 (US-338): Failed to save regime confidence state: %s", e)

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self.config.persistence_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                lmp = data.get("last_max_prob")
                if lmp is not None:
                    self._last_max_prob = float(lmp)
                self._transition_cycles_remaining = int(
                    data.get("transition_cycles_remaining", 0)
                )
                history = data.get("history", [])
                if isinstance(history, list):
                    self._history = [
                        RegimeScoreResult.from_dict(r)
                        for r in history if isinstance(r, dict)
                    ]
                logger.info(
                    "Phase 54 (US-338): Loaded regime confidence scorer — "
                    "%d history entries, last_max_prob=%s",
                    len(self._history),
                    f"{self._last_max_prob:.3f}" if self._last_max_prob else "None",
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 54 (US-338): Could not load regime confidence state: %s", e)
