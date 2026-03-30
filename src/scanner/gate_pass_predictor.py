"""Phase 58 US-360: GatePassPredictor.

Simulates how many raw signals would pass the min_confidence gate
under penalty-free conditions. Answers the diagnostic question:
  'If we removed all penalties, would ANY signals pass?'

If penalty_free_pass_rate >= 5%, penalties are the root blocker.
If penalty_free_pass_rate < 5%, signals are intrinsically too weak
or the threshold needs recalibration.

Design: pure analysis class — binary-search on the recorded distribution.
Wired into continuous.py stalled block to log actionable intel.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SIGNAL_EXISTS_THRESHOLD = 5.0   # penalty_free_pass_rate_pct >= 5 → signal exists


class GatePassPredictor:
    """Predict gate pass rate under penalty-free conditions.

    Usage:
        predictor = GatePassPredictor()
        report = predictor.generate_report(recorder, min_confidence=58.0)
        if report['signal_exists']:
            logger.warning("Signals exist — penalties are blocking them")
        else:
            logger.warning("Signals too weak — threshold recalibration needed")
    """

    def predict_penalty_free_pass_rate(
        self,
        recorder: object,
        min_confidence: float,
    ) -> float:
        """Return fraction of raw scores that would pass min_confidence gate.

        Args:
            recorder: RawConfidenceRecorder instance.
            min_confidence: Gate threshold (0-100 scale).

        Returns:
            Float in [0.0, 1.0].
        """
        dist = recorder.get_distribution()
        count = dist.get("count", 0)
        if count == 0:
            return 0.0

        recent = recorder.get_recent(n=count)
        above = sum(
            1 for r in recent
            if r.get("raw_confidence", 0.0) >= min_confidence
        )
        return above / count

    def predict_pass_rate_at_threshold(
        self,
        recorder: object,
        threshold: float,
    ) -> float:
        """Return fraction of raw scores that would pass a given threshold."""
        return self.predict_penalty_free_pass_rate(recorder, threshold)

    def find_threshold_for_target_pass_rate(
        self,
        recorder: object,
        target_rate: float = 0.10,
    ) -> float:
        """Binary-search the threshold that yields approximately target_rate pass rate.

        Returns the lowest threshold (to 1 decimal precision) at which
        at least target_rate fraction of recorded signals would pass.
        Returns 0.0 if count is insufficient or all signals fail at 0.0.

        Args:
            recorder: RawConfidenceRecorder instance.
            target_rate: Desired pass rate fraction (e.g. 0.10 = 10%).

        Returns:
            Float threshold value (0-100 scale).
        """
        dist = recorder.get_distribution()
        if dist.get("count", 0) == 0:
            return 0.0

        # Binary search between min and max of recorded values
        lo = dist.get("min", 0.0)
        hi = dist.get("max", 100.0)

        # If even at lo we don't meet target, return lo
        if self.predict_pass_rate_at_threshold(recorder, lo) < target_rate:
            return lo

        # Search for threshold where pass_rate >= target_rate
        for _ in range(30):  # ~30 iterations → 0.1 pt precision
            mid = (lo + hi) / 2.0
            rate = self.predict_pass_rate_at_threshold(recorder, mid)
            if rate >= target_rate:
                lo = mid   # can push threshold higher and still meet target
            else:
                hi = mid   # threshold too high, lower it

        return round(lo, 1)

    def generate_report(
        self,
        recorder: object,
        min_confidence: float,
        stall_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a full diagnostic report for the stalled state.

        Args:
            recorder: RawConfidenceRecorder instance.
            min_confidence: Current engine threshold.
            stall_info: Optional dict with stall context (scans_since_last_trade, etc.).

        Returns:
            Dict with: penalty_free_pass_rate_pct, current_pass_rate_pct,
                       threshold_for_10pct, threshold_for_5pct,
                       signal_exists, recommendation.
        """
        penalty_free_rate = self.predict_penalty_free_pass_rate(recorder, min_confidence)
        penalty_free_pct = round(penalty_free_rate * 100.0, 2)

        threshold_for_10pct = self.find_threshold_for_target_pass_rate(recorder, 0.10)
        threshold_for_5pct = self.find_threshold_for_target_pass_rate(recorder, 0.05)

        signal_exists = penalty_free_pct >= SIGNAL_EXISTS_THRESHOLD

        # Recommendation
        if signal_exists:
            recommendation = "PENALTIES_BLOCKING_SIGNALS"
        elif threshold_for_5pct > 0.0:
            recommendation = "THRESHOLD_RECALIBRATION_NEEDED"
        else:
            recommendation = "NO_SIGNALS_IN_DISTRIBUTION"

        report = {
            "penalty_free_pass_rate_pct": penalty_free_pct,
            "current_pass_rate_pct": penalty_free_pct,  # same for raw; penalized scores not tracked here
            "threshold_for_10pct": threshold_for_10pct,
            "threshold_for_5pct": threshold_for_5pct,
            "signal_exists": signal_exists,
            "recommendation": recommendation,
            "min_confidence_used": min_confidence,
        }

        if stall_info:
            report["stall_info"] = stall_info

        logger.warning(
            "GatePassPredictor: penalty_free_pass_rate=%.1f%% signal_exists=%s "
            "threshold_for_10pct=%.1f rec=%s",
            penalty_free_pct, signal_exists, threshold_for_10pct, recommendation,
        )
        return report
