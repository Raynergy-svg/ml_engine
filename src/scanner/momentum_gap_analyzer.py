"""Phase 61 US-372: MomentumGapAnalyzer.

Analyzes the raw XGBoost momentum score distribution from MomentumScoreRecorder
relative to min_momentum threshold, classifying the gap and producing a
recommendation.

Mirrors ThresholdGapAnalyzer (Phase 58 US-359) but for the momentum domain.
Momentum scores live in 0.0-1.0 range (XGBoost output), so near-miss
thresholds are 0.05 / 0.15 (not 3.0 / 8.0 as in confidence domain).

Classification:
  NEAR_THRESHOLD  — gap_at_p50 <= 0.05  → small threshold change may help
  MODERATE        — 0.05 < gap_at_p50 <= 0.15 → investigate XGBoost quality
  FAR_BELOW       — gap_at_p50 > 0.15   → structural weakness, retrain
  INSUFFICIENT_DATA — fewer than 3 fail records
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─── classification constants ─────────────────────────────────────────────────

CLASS_NEAR_THRESHOLD = "NEAR_THRESHOLD"
CLASS_MODERATE = "MODERATE"
CLASS_FAR_BELOW = "FAR_BELOW"
CLASS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# ─── gap thresholds (momentum is 0-1 scale) ───────────────────────────────────

NEAR_MISS_GAP = 0.05    # ≤ 0.05 gap → near miss
MODERATE_GAP = 0.15     # ≤ 0.15 gap → moderate

# ─── recommendations ─────────────────────────────────────────────────────────

_RECS = {
    CLASS_NEAR_THRESHOLD: (
        "Consider 0.02-0.05 momentum threshold reduction — many pairs are near the gate"
    ),
    CLASS_MODERATE: (
        "Investigate XGBoost momentum signal quality — pairs are moderately below threshold"
    ),
    CLASS_FAR_BELOW: (
        "XGBoost momentum structurally weak — retraining likely required"
    ),
    CLASS_INSUFFICIENT_DATA: (
        "Insufficient momentum fail data — accumulate more scans before diagnosing"
    ),
}

MIN_FAIL_COUNT = 3  # minimum fails before classification is meaningful


def analyze(recorder, min_momentum: float) -> Dict[str, Any]:
    """Analyze momentum score distribution vs threshold.

    Args:
        recorder: MomentumScoreRecorder instance
        min_momentum: the current momentum gate threshold (typically 0.40-0.60)

    Returns:
        dict with: count, pass_rate, gap_at_p50, gap_at_p75, near_miss_count,
                   near_miss_rate_pct, classification, recommendation
    """
    try:
        fail_dist = recorder.get_fail_distribution()
        pass_rate = recorder.get_pass_rate()
        all_dist = recorder.get_distribution()
    except Exception as exc:
        logger.error("MomentumGapAnalyzer.analyze() failed reading recorder: %s", exc)
        return _empty_result(min_momentum)

    total_count = all_dist.get("count", 0)
    fail_count = fail_dist.get("count", 0)

    if fail_count < MIN_FAIL_COUNT:
        return {
            "count": total_count,
            "fail_count": fail_count,
            "pass_rate": round(pass_rate, 4),
            "gap_at_p50": 0.0,
            "gap_at_p75": 0.0,
            "near_miss_count": 0,
            "near_miss_rate_pct": 0.0,
            "classification": CLASS_INSUFFICIENT_DATA,
            "recommendation": _RECS[CLASS_INSUFFICIENT_DATA],
        }

    p50 = fail_dist.get("p50", 0.0)
    p75 = fail_dist.get("p75", 0.0)
    gap_at_p50 = round(float(min_momentum) - p50, 6)
    gap_at_p75 = round(float(min_momentum) - p75, 6)

    # Classify
    if gap_at_p50 <= NEAR_MISS_GAP:
        classification = CLASS_NEAR_THRESHOLD
    elif gap_at_p50 <= MODERATE_GAP:
        classification = CLASS_MODERATE
    else:
        classification = CLASS_FAR_BELOW

    # Near-miss counting: fails where gap (min_momentum - score) <= 0.05
    try:
        # Re-read fail records to count near misses
        with recorder._lock:
            fail_records = [
                r for r in recorder._records
                if not r.get("momentum_passed", True)
            ]
        near_miss_count = sum(
            1 for r in fail_records
            if float(min_momentum) - float(r.get("momentum_score", 0.0)) <= NEAR_MISS_GAP
        )
    except Exception:
        near_miss_count = 0

    near_miss_rate_pct = round(
        near_miss_count / fail_count * 100.0 if fail_count > 0 else 0.0, 2
    )

    return {
        "count": total_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 4),
        "gap_at_p50": gap_at_p50,
        "gap_at_p75": gap_at_p75,
        "near_miss_count": near_miss_count,
        "near_miss_rate_pct": near_miss_rate_pct,
        "classification": classification,
        "recommendation": _RECS[classification],
    }


def _empty_result(min_momentum: float) -> Dict[str, Any]:
    return {
        "count": 0,
        "fail_count": 0,
        "pass_rate": 0.0,
        "gap_at_p50": 0.0,
        "gap_at_p75": 0.0,
        "near_miss_count": 0,
        "near_miss_rate_pct": 0.0,
        "classification": CLASS_INSUFFICIENT_DATA,
        "recommendation": _RECS[CLASS_INSUFFICIENT_DATA],
    }


class MomentumGapAnalyzer:
    """Wrapper class for analyze() — holds reference to recorder for convenience.

    Usage:
        analyzer = MomentumGapAnalyzer(recorder, min_momentum=0.45)
        report = analyzer.analyze_current()
        signals = analyzer.get_momentum_signals()  # for ScanHealthSynthesizer
    """

    def __init__(self, recorder=None, min_momentum: float = 0.45) -> None:
        self._recorder = recorder
        self._min_momentum = min_momentum

    def analyze_current(self) -> Dict[str, Any]:
        """Run analysis with current recorder state."""
        if self._recorder is None:
            return _empty_result(self._min_momentum)
        return analyze(self._recorder, self._min_momentum)

    def get_momentum_signals(self) -> Dict[str, Any]:
        """Return a signals dict consumable by ScanHealthSynthesizer."""
        report = self.analyze_current()
        return {
            "momentum_gap_at_p50": report.get("gap_at_p50", 0.0),
            "momentum_near_miss_rate_pct": report.get("near_miss_rate_pct", 0.0),
            "momentum_classification": report.get("classification", CLASS_INSUFFICIENT_DATA),
            "momentum_pass_rate": report.get("pass_rate", 0.0),
        }
