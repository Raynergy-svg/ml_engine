"""
Setup Quality Filter — Phase 51 (US-321).

Composite weak-setup rejection filter that scores each trade on 4 dimensions:
1. Agent disagreement (from ensemble_conflict)
2. Uncertainty (from uncertainty agent)
3. Recent pair form (from fitness/journal data)
4. Confidence band (where raw confidence falls)

Rejects setups with quality_score < threshold (default 0.35).
Tracks rejection rate to prevent over-filtering.

Usage:
    sqf = SetupQualityFilter()
    result = sqf.evaluate(
        pair="EUR_USD",
        confidence=0.55,
        agent_disagreement=0.35,
        uncertainty_score=0.40,
        pair_recent_win_rate=0.30,
    )
    if not result.passed:
        logger.info("Setup rejected: %s", result.rejection_reasons)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class SetupQualityConfig:
    """Configuration for setup quality filter."""
    # Dimension weights (must sum to 1.0)
    weight_disagreement: float = 0.30
    weight_uncertainty: float = 0.25
    weight_recent_form: float = 0.25
    weight_confidence_band: float = 0.20

    # Rejection threshold
    quality_threshold: float = 0.35

    # Per-dimension thresholds for converting raw values to 0-1 scores
    # Disagreement: 0.0 (perfect agreement) → 1.0 score; 0.5 (total disagreement) → 0.0 score
    disagreement_max: float = 0.50
    # Uncertainty: 0.0 (certain) → 1.0 score; 0.60 (very uncertain) → 0.0 score
    uncertainty_max: float = 0.60
    # Recent form: 0.0 win rate → 0.0 score; 0.60+ → 1.0 score
    form_excellent: float = 0.60
    # Confidence band: 0.40 → 0.0 score; 0.80+ → 1.0 score
    confidence_low: float = 0.40
    confidence_high: float = 0.80

    # Over-filtering protection
    max_rejection_rate: float = 0.40    # Warn if rejecting > 40% of setups
    tracking_window: int = 50           # Last N evaluations for rejection rate


@dataclass
class SetupQualityResult:
    """Result of setup quality evaluation."""
    quality_score: float = 0.0
    passed: bool = True
    rejection_reasons: List[str] = field(default_factory=list)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class SetupQualityFilter:
    """Composite filter for rejecting low-quality trade setups."""

    def __init__(self, config: Optional[SetupQualityConfig] = None):
        self.config = config or SetupQualityConfig()
        self._evaluation_history: List[bool] = []  # True = passed, False = rejected

    def evaluate(
        self,
        pair: str,
        confidence: float,
        agent_disagreement: float = 0.0,
        uncertainty_score: float = 0.0,
        pair_recent_win_rate: float = 0.5,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> SetupQualityResult:
        """Evaluate a trade setup's quality.

        Args:
            pair: Currency pair
            confidence: Raw confidence 0.0-1.0
            agent_disagreement: Agent disagreement 0.0-1.0 (higher = more disagreement)
            uncertainty_score: Uncertainty 0.0-1.0 (higher = more uncertain)
            pair_recent_win_rate: Recent win rate for this pair 0.0-1.0
            extra_context: Optional additional context for logging

        Returns:
            SetupQualityResult with pass/fail and dimension breakdown.
        """
        c = self.config

        # 1. Agreement score (inverse of disagreement)
        agreement_score = max(0.0, 1.0 - (agent_disagreement / c.disagreement_max))
        agreement_score = min(1.0, agreement_score)

        # 2. Certainty score (inverse of uncertainty)
        certainty_score = max(0.0, 1.0 - (uncertainty_score / c.uncertainty_max))
        certainty_score = min(1.0, certainty_score)

        # 3. Recent form score
        if pair_recent_win_rate >= c.form_excellent:
            form_score = 1.0
        elif pair_recent_win_rate <= 0.0:
            form_score = 0.0
        else:
            form_score = pair_recent_win_rate / c.form_excellent
        form_score = min(1.0, form_score)

        # 4. Confidence band score
        if confidence >= c.confidence_high:
            band_score = 1.0
        elif confidence <= c.confidence_low:
            band_score = 0.0
        else:
            band_score = (confidence - c.confidence_low) / (c.confidence_high - c.confidence_low)

        # Composite quality score
        quality = (
            c.weight_disagreement * agreement_score
            + c.weight_uncertainty * certainty_score
            + c.weight_recent_form * form_score
            + c.weight_confidence_band * band_score
        )
        quality = round(max(0.0, min(1.0, quality)), 4)

        # Determine pass/fail
        passed = quality >= c.quality_threshold
        reasons = []

        if not passed:
            if agreement_score < 0.4:
                reasons.append(f"High disagreement ({agent_disagreement:.2f})")
            if certainty_score < 0.4:
                reasons.append(f"High uncertainty ({uncertainty_score:.2f})")
            if form_score < 0.3:
                reasons.append(f"Poor recent form ({pair_recent_win_rate:.2%})")
            if band_score < 0.3:
                reasons.append(f"Low confidence ({confidence:.2f})")

        result = SetupQualityResult(
            quality_score=quality,
            passed=passed,
            rejection_reasons=reasons,
            dimension_scores={
                "agreement": round(agreement_score, 4),
                "certainty": round(certainty_score, 4),
                "recent_form": round(form_score, 4),
                "confidence_band": round(band_score, 4),
            },
            details={
                "pair": pair,
                "raw_confidence": confidence,
                "agent_disagreement": agent_disagreement,
                "uncertainty_score": uncertainty_score,
                "pair_recent_win_rate": pair_recent_win_rate,
                "threshold": c.quality_threshold,
            },
        )

        # Track for rejection rate monitoring
        self._evaluation_history.append(passed)
        if len(self._evaluation_history) > c.tracking_window:
            self._evaluation_history = self._evaluation_history[-c.tracking_window:]

        # Check rejection rate
        rejection_rate = self.get_rejection_rate()
        if rejection_rate > c.max_rejection_rate:
            logger.warning(
                "Phase 51 US-321: Rejection rate %.1f%% exceeds %.1f%% — consider loosening threshold",
                rejection_rate * 100, c.max_rejection_rate * 100,
            )

        if not passed:
            logger.info(
                "Phase 51 US-321: %s REJECTED (quality=%.3f < %.3f) — %s",
                pair, quality, c.quality_threshold, "; ".join(reasons),
            )

        return result

    def get_rejection_rate(self) -> float:
        """Get current rejection rate from recent evaluations."""
        if not self._evaluation_history:
            return 0.0
        rejected = sum(1 for p in self._evaluation_history if not p)
        return rejected / len(self._evaluation_history)

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        n = len(self._evaluation_history)
        return {
            "total_evaluations": n,
            "passed": sum(1 for p in self._evaluation_history if p),
            "rejected": sum(1 for p in self._evaluation_history if not p),
            "rejection_rate": round(self.get_rejection_rate(), 4),
            "threshold": self.config.quality_threshold,
        }
