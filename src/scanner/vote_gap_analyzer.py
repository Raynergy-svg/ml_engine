"""Phase 66 US-389: VoteGapAnalyzer.

Analyzes the weighted_vote_score distribution from VoteScoreRecorder
relative to weighted_vote_threshold, classifying the gap and producing a
recommendation.

Mirrors MomentumGapAnalyzer (Phase 61 US-372) but for the agent consensus domain.
Weighted vote scores live in 0.0-1.0 range (agent team weighted average), so
near-miss thresholds are 0.05 / 0.15 (same scale as momentum domain).

Classification:
  NEAR_THRESHOLD  — gap_at_p50 <= 0.05  → small threshold change may help
  MODERATE        — 0.05 < gap_at_p50 <= 0.15 → investigate agent weight quality
  FAR_BELOW       — gap_at_p50 > 0.15   → structural agent weakness
  INSUFFICIENT_DATA — fewer than 3 fail records
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ─── classification constants ─────────────────────────��───────────────────────

CLASS_NEAR_THRESHOLD = "NEAR_THRESHOLD"
CLASS_MODERATE = "MODERATE"
CLASS_FAR_BELOW = "FAR_BELOW"
CLASS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# ─── gap thresholds (vote score is 0-1 scale) ────────────────────────────────

NEAR_MISS_GAP = 0.05    # ≤ 0.05 gap → near miss
MODERATE_GAP = 0.15     # ≤ 0.15 gap → moderate

# ─── recommendations ─────────────────────────────────────────────────────────

_RECS = {
    CLASS_NEAR_THRESHOLD: (
        "Consider 0.02-0.03 vote threshold reduction — many pairs are near the agent consensus gate"
    ),
    CLASS_MODERATE: (
        "Investigate agent weight quality — pairs are moderately below vote threshold"
    ),
    CLASS_FAR_BELOW: (
        "Agent consensus structurally weak — review agent weights and sub-inference pipeline"
    ),
    CLASS_INSUFFICIENT_DATA: (
        "Insufficient vote fail data — accumulate more scans before diagnosing"
    ),
}

MIN_FAIL_COUNT = 3  # minimum fails before classification is meaningful


def analyze(recorder, weighted_vote_threshold: float) -> Dict[str, Any]:
    """Analyze vote score distribution vs threshold.

    Args:
        recorder: VoteScoreRecorder instance
        weighted_vote_threshold: the current agent vote gate threshold (typically 0.50-0.60)

    Returns:
        dict with: count, pass_rate, gap_at_p50, gap_at_p75, near_miss_count,
                   near_miss_rate_pct, classification, recommendation
    """
    try:
        fail_dist = recorder.get_fail_distribution()
        pass_rate = recorder.get_pass_rate()
        all_dist = recorder.get_distribution()
    except Exception as exc:
        logger.error("VoteGapAnalyzer.analyze() failed reading recorder: %s", exc)
        return _empty_result(weighted_vote_threshold)

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
    gap_at_p50 = round(float(weighted_vote_threshold) - p50, 6)
    gap_at_p75 = round(float(weighted_vote_threshold) - p75, 6)

    # Classify
    if gap_at_p50 <= NEAR_MISS_GAP:
        classification = CLASS_NEAR_THRESHOLD
    elif gap_at_p50 <= MODERATE_GAP:
        classification = CLASS_MODERATE
    else:
        classification = CLASS_FAR_BELOW

    # Near-miss counting: fails where gap (threshold - score) <= 0.05
    try:
        with recorder._lock:
            fail_records = [
                r for r in recorder._records
                if not r.get("agent_passed", True)
            ]
        near_miss_count = sum(
            1 for r in fail_records
            if float(weighted_vote_threshold) - float(r.get("vote_score", 0.0)) <= NEAR_MISS_GAP
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


def _empty_result(weighted_vote_threshold: float) -> Dict[str, Any]:
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


class VoteGapAnalyzer:
    """Wrapper class for analyze() — holds reference to recorder for convenience.

    Usage:
        analyzer = VoteGapAnalyzer(recorder, weighted_vote_threshold=0.55)
        report = analyzer.analyze_current()
        signals = analyzer.get_vote_signals()  # for ScanHealthSynthesizer
    """

    def __init__(self, recorder=None, weighted_vote_threshold: float = 0.55) -> None:
        self._recorder = recorder
        self._weighted_vote_threshold = weighted_vote_threshold

    def analyze_current(self) -> Dict[str, Any]:
        """Run analysis with current recorder state."""
        if self._recorder is None:
            return _empty_result(self._weighted_vote_threshold)
        return analyze(self._recorder, self._weighted_vote_threshold)

    def get_vote_signals(self) -> Dict[str, Any]:
        """Return a signals dict consumable by ScanHealthSynthesizer."""
        report = self.analyze_current()
        return {
            "vote_gap_at_p50": report.get("gap_at_p50", 0.0),
            "vote_near_miss_rate_pct": report.get("near_miss_rate_pct", 0.0),
            "vote_classification": report.get("classification", CLASS_INSUFFICIENT_DATA),
            "vote_pass_rate": report.get("pass_rate", 0.0),
        }
