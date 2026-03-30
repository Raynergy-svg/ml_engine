"""Phase 63 US-379: RiskGateAnalyzer.

Classifies risk gate rejects by proximity to the drawdown threshold.
Mirrors MomentumGapAnalyzer (Phase 61) but for the risk domain.

Gap direction is INVERTED vs confidence/momentum analyzers:
  gap = rf_drawdown_pct - max_drawdown_pct  (positive = over threshold = fail)

Classification thresholds (on 0.025 drawdown scale):
  NEAR_THRESHOLD  : gap_at_p50 <= 0.005   (pairs just over the gate — easy fix)
  MODERATE        : gap_at_p50 <= 0.015   (possible RF calibration issue)
  FAR_ABOVE       : gap_at_p50 > 0.015    (structurally high drawdown prediction)
  INSUFFICIENT_DATA: < 3 fail records

near_miss threshold: gap <= 0.005 per record (20% of the threshold value,
  analogous to Phase 61 momentum 0.05 on 0.45 scale = 11%).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from src.scanner.risk_reject_recorder import RiskRejectRecorder

logger = logging.getLogger(__name__)

# Classification labels
CLASS_NEAR_THRESHOLD = "NEAR_THRESHOLD"
CLASS_MODERATE = "MODERATE"
CLASS_FAR_ABOVE = "FAR_ABOVE"
CLASS_INSUFFICIENT = "INSUFFICIENT_DATA"

# Gap thresholds (rf_drawdown_pct - max_drawdown_pct)
_NEAR_THRESHOLD_LIMIT = 0.005
_MODERATE_LIMIT = 0.015

# Minimum fail records required for classification
_MIN_RECORDS = 3

# Recommendation strings per classification
_RECOMMENDATIONS: Dict[str, str] = {
    CLASS_NEAR_THRESHOLD: (
        "Consider 0.002-0.005 drawdown threshold increase — "
        "predicted drawdowns are near the gate"
    ),
    CLASS_MODERATE: (
        "Investigate RF drawdown prediction quality — "
        "moderate systematic overestimation possible"
    ),
    CLASS_FAR_ABOVE: (
        "RF predicts structurally high drawdown — "
        "verify market conditions or retrain RF model"
    ),
    CLASS_INSUFFICIENT: "Insufficient data for classification — continue collecting",
}


def analyze(
    recorder: "RiskRejectRecorder",
    max_drawdown_pct: float,
) -> Dict[str, Any]:
    """Analyze risk gate rejects and classify by drawdown proximity.

    Args:
        recorder:        RiskRejectRecorder with recorded snapshots.
        max_drawdown_pct: Current drawdown threshold (e.g. 0.025).

    Returns:
        Dict with keys: count, fail_count, pass_rate, gap_at_p50, gap_at_p75,
            near_miss_count, near_miss_rate_pct, drawdown_kill_pct,
            streak_kill_pct, classification, recommendation.
    """
    dist = recorder.get_distribution()
    fail_dist = recorder.get_fail_distribution()
    pass_rate = recorder.get_pass_rate()
    kill_breakdown = recorder.get_kill_breakdown()

    count = dist.get("count", 0)
    fail_count = fail_dist.get("count", 0)

    # Default safe result
    safe_defaults: Dict[str, Any] = {
        "count": count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 4),
        "gap_at_p50": 0.0,
        "gap_at_p75": 0.0,
        "near_miss_count": 0,
        "near_miss_rate_pct": 0.0,
        "drawdown_kill_pct": kill_breakdown.get("drawdown_kill_pct", 0.0),
        "streak_kill_pct": kill_breakdown.get("streak_kill_pct", 0.0),
        "classification": CLASS_INSUFFICIENT,
        "recommendation": _RECOMMENDATIONS[CLASS_INSUFFICIENT],
    }

    if fail_count < _MIN_RECORDS:
        return safe_defaults

    # Compute gap at p50 and p75 of FAIL distribution
    # Inverted: gap = score - threshold (positive = over threshold)
    p50_fail = fail_dist.get("p50", 0.0)
    p75_fail = fail_dist.get("p75", 0.0)
    gap_at_p50 = round(float(p50_fail) - float(max_drawdown_pct), 8)
    gap_at_p75 = round(float(p75_fail) - float(max_drawdown_pct), 8)

    # near_miss: fail records where (rf_drawdown_pct - max_drawdown_pct) <= 0.005
    # We need raw records from the recorder — use fail_distribution is insufficient,
    # we count via a helper approach: compare each fail score to threshold.
    # Since recorder only exposes distribution stats, we use a compatible approach:
    # get_fail_distribution() gives us percentiles. To count near_misses we need
    # the recorder's fail records, but we only have the analyzer interface.
    # PRD says: near_miss_count: fail records where (rf_drawdown_pct - max_drawdown_pct) <= 0.005
    # We compute this from the recorder's internal state via get_fail_distribution
    # metadata — but that only gives us stats, not raw records. So we call the
    # recorder's get_kill_breakdown (which iterates fails) as a reference,
    # and compute near_miss via a dedicated pass over fail records.
    # To keep RiskGateAnalyzer decoupled, we expose this via recorder's fail records.
    near_miss_count = _count_near_misses(recorder, max_drawdown_pct, _NEAR_THRESHOLD_LIMIT)
    near_miss_rate_pct = (
        round(near_miss_count / fail_count * 100.0, 2) if fail_count > 0 else 0.0
    )

    # Classify by gap at p50 of fail scores
    if gap_at_p50 <= _NEAR_THRESHOLD_LIMIT:
        classification = CLASS_NEAR_THRESHOLD
    elif gap_at_p50 <= _MODERATE_LIMIT:
        classification = CLASS_MODERATE
    else:
        classification = CLASS_FAR_ABOVE

    recommendation = _RECOMMENDATIONS.get(classification, "")

    return {
        "count": count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 4),
        "gap_at_p50": gap_at_p50,
        "gap_at_p75": gap_at_p75,
        "near_miss_count": near_miss_count,
        "near_miss_rate_pct": near_miss_rate_pct,
        "drawdown_kill_pct": kill_breakdown.get("drawdown_kill_pct", 0.0),
        "streak_kill_pct": kill_breakdown.get("streak_kill_pct", 0.0),
        "classification": classification,
        "recommendation": recommendation,
    }


def _count_near_misses(
    recorder: "RiskRejectRecorder",
    max_drawdown_pct: float,
    near_miss_limit: float,
) -> int:
    """Count fail records where (rf_drawdown_pct - max_drawdown_pct) <= near_miss_limit."""
    # Access recorder's internal records safely
    with recorder._lock:
        fails = [r for r in recorder._records if not r.get("risk_passed", True)]
    count = 0
    for r in fails:
        gap = float(r.get("rf_drawdown_pct", 0.0)) - float(max_drawdown_pct)
        if gap <= near_miss_limit:
            count += 1
    return count


def get_risk_signals(
    recorder: "RiskRejectRecorder",
    max_drawdown_pct: float,
) -> Dict[str, Any]:
    """Return dict consumable by ScanHealthSynthesizer.

    Keys: risk_gap_at_p50, risk_near_miss_rate_pct, risk_classification,
          risk_pass_rate, risk_drawdown_kill_pct.
    """
    result = analyze(recorder, max_drawdown_pct)
    return {
        "risk_gap_at_p50": result.get("gap_at_p50", 0.0),
        "risk_near_miss_rate_pct": result.get("near_miss_rate_pct", 0.0),
        "risk_classification": result.get("classification", CLASS_INSUFFICIENT),
        "risk_pass_rate": result.get("pass_rate", 0.0),
        "risk_drawdown_kill_pct": result.get("drawdown_kill_pct", 0.0),
    }


class RiskGateAnalyzer:
    """Class wrapper for RiskGateAnalyzer — delegates to module-level analyze().

    Provides analyze_current() and get_risk_signals() as instance methods
    for callers that prefer OOP interface.
    """

    def analyze_current(
        self,
        recorder: "RiskRejectRecorder",
        max_drawdown_pct: float,
    ) -> Dict[str, Any]:
        """Run analysis on the current recorder state."""
        return analyze(recorder, max_drawdown_pct)

    def get_risk_signals(
        self,
        recorder: "RiskRejectRecorder",
        max_drawdown_pct: float,
    ) -> Dict[str, Any]:
        """Return signals dict for ScanHealthSynthesizer."""
        return get_risk_signals(recorder, max_drawdown_pct)
