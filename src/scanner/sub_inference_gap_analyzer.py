"""Phase 67 US-392: SubInferenceGapAnalyzer.

Analyzes the sub-inference consensus ratio distribution from SubInferenceRecorder
relative to sub_inference_vote_threshold, classifying the gap.

The sub-inference gate computes:
  vote_required = ceil(total * sub_inference_vote_threshold)
  agent_passed = votes >= vote_required

consensus_ratio = votes/total is in 0.0-1.0 range.
sub_inference_vote_threshold is typically 0.60-0.72.

Classification (mirrors MomentumGapAnalyzer):
  NEAR_THRESHOLD    — gap_at_p50 <= 0.10  → small threshold change may help
  MODERATE          — 0.10 < gap_at_p50 <= 0.25 → investigate model agreement
  FAR_BELOW         — gap_at_p50 > 0.25   → structural disagreement
  INSUFFICIENT_DATA — fewer than 3 fail records

Note: near-miss thresholds are wider (0.10/0.25) because consensus_ratio
is quantized (0, 0.33, 0.67, 1.0 for 3 window checks) — 0.05 is too tight.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ─── classification constants ─────────────────────────────────────────────────

CLASS_NEAR_THRESHOLD = "NEAR_THRESHOLD"
CLASS_MODERATE = "MODERATE"
CLASS_FAR_BELOW = "FAR_BELOW"
CLASS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# ─── gap thresholds (consensus ratio is 0-1 scale, quantized) ─────────────────

NEAR_MISS_GAP = 0.10    # ≤ 0.10 gap → near miss (wider than momentum due to quantization)
MODERATE_GAP = 0.25     # ≤ 0.25 gap → moderate

# ─── recommendations ─────────────────────────────────────────────────────────

_RECS = {
    CLASS_NEAR_THRESHOLD: (
        "Consider 0.05-0.10 sub-inference vote threshold reduction — "
        "window checks are close to consensus"
    ),
    CLASS_MODERATE: (
        "Investigate sub-inference model agreement — window checks show "
        "moderate directional disagreement"
    ),
    CLASS_FAR_BELOW: (
        "Sub-inference structurally failing — models disagree on direction across "
        "most time windows. Review model quality or reduce window_checks"
    ),
    CLASS_INSUFFICIENT_DATA: (
        "Insufficient sub-inference fail data — accumulate more scans"
    ),
}

MIN_FAIL_COUNT = 3


def analyze(recorder, sub_inference_vote_threshold: float) -> Dict[str, Any]:
    """Analyze sub-inference consensus ratio distribution vs threshold.

    Args:
        recorder: SubInferenceRecorder instance
        sub_inference_vote_threshold: current threshold (typically 0.60-0.72)

    Returns:
        dict with: count, pass_rate, gap_at_p50, gap_at_p75, near_miss_count,
                   near_miss_rate_pct, classification, recommendation
    """
    try:
        fail_dist = recorder.get_fail_distribution()
        pass_rate = recorder.get_pass_rate()
        all_dist = recorder.get_distribution()
    except Exception as exc:
        logger.error("SubInferenceGapAnalyzer.analyze() failed: %s", exc)
        return _empty_result(sub_inference_vote_threshold)

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
    gap_at_p50 = round(float(sub_inference_vote_threshold) - p50, 6)
    gap_at_p75 = round(float(sub_inference_vote_threshold) - p75, 6)

    # Classify
    if gap_at_p50 <= NEAR_MISS_GAP:
        classification = CLASS_NEAR_THRESHOLD
    elif gap_at_p50 <= MODERATE_GAP:
        classification = CLASS_MODERATE
    else:
        classification = CLASS_FAR_BELOW

    # Near-miss counting: fails where gap <= NEAR_MISS_GAP
    try:
        with recorder._lock:
            fail_records = [
                r for r in recorder._records
                if not r.get("agent_passed", True)
            ]
        near_miss_count = sum(
            1 for r in fail_records
            if float(sub_inference_vote_threshold) - float(r.get("consensus_ratio", 0.0)) <= NEAR_MISS_GAP
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


def _empty_result(threshold: float) -> Dict[str, Any]:
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
