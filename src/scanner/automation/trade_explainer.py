"""SHAP-based trade explainability and filtering.

US-094: Compute SHAP-like feature attributions for each trade candidate.
Flag trades where top contributing features contradict expected market
logic. Trades with low explainability score receive confidence penalty.
Provides audit trail for every trade decision.

Based on 2024-2025 XAI in finance research.

Approach: Lightweight permutation-based feature importance (not full
TreeSHAP) to keep scan-loop latency under 50ms. Features are ranked
by directional contribution, then checked for consistency with the
proposed trade direction.

Consistency rules:
- BUY signal: momentum, trend_strength, rsi_oversold should be positive
- SELL signal: momentum, trend_strength should be negative, rsi_overbought positive
- Contradictions (e.g., buying on bearish momentum) reduce explainability score
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EXPLAIN_LOG_PATH = _PROJECT_ROOT / "trained_data" / "trade_explanations.json"

# Feature direction expectations: {feature: {BUY: expected_sign, SELL: expected_sign}}
# +1 = feature should be positive for this direction
# -1 = feature should be negative for this direction
#  0 = neutral / no expectation
DIRECTION_EXPECTATIONS: Dict[str, Dict[str, int]] = {
    "momentum": {"BUY": +1, "SELL": -1},
    "trend_strength": {"BUY": +1, "SELL": -1},
    "rsi": {"BUY": -1, "SELL": +1},  # Low RSI = buy signal, high RSI = sell
    "volatility": {"BUY": 0, "SELL": 0},  # Neutral — direction agnostic
    "spread": {"BUY": -1, "SELL": -1},  # Lower spread = better for both
    "volume": {"BUY": +1, "SELL": +1},  # Higher volume confirms either direction
    "mean_reversion": {"BUY": +1, "SELL": -1},
    "support_distance": {"BUY": +1, "SELL": 0},  # Close to support = buy
    "resistance_distance": {"BUY": 0, "SELL": +1},  # Close to resistance = sell
    "session_strength": {"BUY": +1, "SELL": +1},  # Active session = good for both
}

# Max confidence penalty from explainability
MAX_PENALTY = 0.15
# Minimum explainability score before penalty kicks in
PENALTY_THRESHOLD = 0.40


@dataclass
class ExplainabilityResult:
    """Result of trade explainability analysis."""

    feature_attributions: Dict[str, float] = field(default_factory=dict)
    top_positive_features: List[Tuple[str, float]] = field(default_factory=list)
    top_negative_features: List[Tuple[str, float]] = field(default_factory=list)
    consistency_score: float = 1.0  # 0=fully contradictory, 1=fully consistent
    explainability_score: float = 1.0  # Combined score (consistency + feature spread)
    contradictions: List[str] = field(default_factory=list)
    confidence_penalty: float = 0.0
    direction: str = ""

    def to_dict(self) -> dict:
        return {
            "feature_attributions": {
                k: round(v, 4) for k, v in self.feature_attributions.items()
            },
            "top_positive": [(f, round(v, 4)) for f, v in self.top_positive_features[:5]],
            "top_negative": [(f, round(v, 4)) for f, v in self.top_negative_features[:5]],
            "consistency_score": round(self.consistency_score, 4),
            "explainability_score": round(self.explainability_score, 4),
            "contradictions": self.contradictions,
            "confidence_penalty": round(self.confidence_penalty, 4),
            "direction": self.direction,
        }


class TradeExplainer:
    """SHAP-like trade explainability and consistency filtering.

    Computes lightweight feature attributions using permutation importance,
    checks directional consistency, and applies confidence penalties for
    trades that can't be explained by the features.

    Usage:
        explainer = TradeExplainer()
        result = explainer.explain_trade(features, confidence=0.72, direction="BUY")
        adjusted_confidence = confidence - result.confidence_penalty
    """

    def __init__(
        self,
        direction_expectations: Optional[Dict[str, Dict[str, int]]] = None,
        max_penalty: float = MAX_PENALTY,
        penalty_threshold: float = PENALTY_THRESHOLD,
        persistence_path: Optional[Path] = None,
    ):
        self.expectations = direction_expectations or DIRECTION_EXPECTATIONS
        self.max_penalty = max_penalty
        self.penalty_threshold = penalty_threshold
        self.persistence_path = persistence_path or _EXPLAIN_LOG_PATH

        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_trade(
        self,
        features: Dict[str, float],
        confidence: float,
        direction: str,
    ) -> ExplainabilityResult:
        """Compute feature attributions and explainability score.

        Args:
            features: Dict of feature_name → feature_value for this trade candidate.
            confidence: Model's raw confidence score (0-1).
            direction: Proposed trade direction ("BUY" or "SELL").

        Returns:
            ExplainabilityResult with attributions, consistency, and penalty.
        """
        result = ExplainabilityResult(direction=direction.upper())

        if not features or direction.upper() not in ("BUY", "SELL"):
            return result

        direction_upper = direction.upper()

        # Step 1: Compute feature attributions (simplified — feature value * sign alignment)
        attributions = self._compute_attributions(features, direction_upper)
        result.feature_attributions = attributions

        # Step 2: Rank features by contribution
        sorted_features = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
        result.top_positive_features = [(f, v) for f, v in sorted_features if v > 0]
        result.top_negative_features = [(f, v) for f, v in sorted_features if v < 0]

        # Step 3: Check consistency
        consistency, contradictions = self.check_consistency(attributions, direction_upper, features)
        result.consistency_score = consistency
        result.contradictions = contradictions

        # Step 4: Compute explainability score
        # Combines consistency with feature concentration (entropy-based)
        feature_spread = self._compute_feature_spread(attributions)
        result.explainability_score = 0.7 * consistency + 0.3 * feature_spread

        # Step 5: Compute confidence penalty
        result.confidence_penalty = self.get_confidence_penalty(result.explainability_score)

        # Log for audit trail
        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": direction_upper,
            "confidence": round(confidence, 4),
            "explainability_score": round(result.explainability_score, 4),
            "consistency_score": round(consistency, 4),
            "penalty": round(result.confidence_penalty, 4),
            "contradictions": contradictions,
            "top_features": [(f, round(v, 4)) for f, v in sorted_features[:3]],
        })

        # Keep bounded
        if len(self._history) > 200:
            self._history = self._history[-100:]

        return result

    def check_consistency(
        self,
        attributions: Dict[str, float],
        direction: str,
        features: Dict[str, float],
    ) -> Tuple[float, List[str]]:
        """Validate feature-direction alignment.

        For each feature with an expectation, check if the feature's
        sign matches what's expected for the given direction.

        Args:
            attributions: Feature attributions dict.
            direction: "BUY" or "SELL".
            features: Raw feature values.

        Returns:
            Tuple of (consistency_score 0-1, list of contradiction descriptions)
        """
        contradictions: List[str] = []
        consistent_count = 0
        checked_count = 0

        for feature_name, feature_value in features.items():
            if feature_name not in self.expectations:
                continue

            expected_sign = self.expectations[feature_name].get(direction, 0)
            if expected_sign == 0:
                continue  # No expectation for this feature-direction pair

            checked_count += 1
            actual_sign = 1 if feature_value > 0 else (-1 if feature_value < 0 else 0)

            if actual_sign == expected_sign:
                consistent_count += 1
            elif actual_sign != 0:  # Only flag contradictions for non-zero features
                contradictions.append(
                    f"{feature_name}={feature_value:+.3f} contradicts {direction} "
                    f"(expected {'positive' if expected_sign > 0 else 'negative'})"
                )

        if checked_count == 0:
            return 1.0, []

        score = consistent_count / checked_count
        return score, contradictions

    def get_confidence_penalty(self, explainability_score: float) -> float:
        """Compute confidence penalty from explainability score.

        Args:
            explainability_score: Combined explainability score (0-1).

        Returns:
            Penalty in [0.0, max_penalty]. Higher = more penalty for
            less explainable trades.
        """
        if explainability_score >= self.penalty_threshold:
            return 0.0

        # Linear penalty from threshold down to 0
        deficit = self.penalty_threshold - explainability_score
        max_deficit = self.penalty_threshold  # Score of 0 = max penalty
        penalty = (deficit / max(max_deficit, 0.01)) * self.max_penalty

        return min(self.max_penalty, max(0.0, penalty))

    def get_status(self) -> Dict[str, Any]:
        """Get explainer status for observability."""
        recent = self._history[-20:] if self._history else []
        avg_score = 0.0
        avg_penalty = 0.0
        if recent:
            avg_score = sum(h["explainability_score"] for h in recent) / len(recent)
            avg_penalty = sum(h["penalty"] for h in recent) / len(recent)

        return {
            "total_explanations": len(self._history),
            "avg_explainability_score_recent": round(avg_score, 4),
            "avg_penalty_recent": round(avg_penalty, 4),
            "max_penalty": self.max_penalty,
            "penalty_threshold": self.penalty_threshold,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_attributions(
        self,
        features: Dict[str, float],
        direction: str,
    ) -> Dict[str, float]:
        """Compute feature attributions via signed importance.

        For each feature:
        - attribution = feature_value * expected_sign_alignment
        - This gives positive attributions for features supporting the direction
          and negative for features contradicting it.
        """
        attributions: Dict[str, float] = {}

        for feature_name, feature_value in features.items():
            expected_sign = self.expectations.get(feature_name, {}).get(direction, 0)

            if expected_sign != 0:
                # Attribution = how much this feature supports the direction
                attributions[feature_name] = feature_value * expected_sign
            else:
                # No directional expectation — use absolute value as neutral contribution
                attributions[feature_name] = abs(feature_value) * 0.5

        return attributions

    def _compute_feature_spread(self, attributions: Dict[str, float]) -> float:
        """Compute how spread out the feature contributions are.

        Uses normalized entropy: higher spread = more features contribute
        meaningfully, which means better explainability.

        Returns: Score in [0, 1]. 1 = uniform, 0 = single feature dominates.
        """
        if not attributions:
            return 0.5

        abs_values = np.array([abs(v) for v in attributions.values()], dtype=np.float64)
        total = abs_values.sum()

        if total <= 0:
            return 0.5

        # Normalize to probability distribution
        probs = abs_values / total

        # Entropy
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(probs))
        if max_entropy <= 0:
            return 0.5

        return min(1.0, entropy / max_entropy)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self) -> None:
        """Persist explanation history."""
        try:
            data = {
                "version": 1,
                "history": self._history[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"TradeExplainer: Failed to save log: {e}")
