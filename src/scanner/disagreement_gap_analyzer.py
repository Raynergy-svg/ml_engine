"""Phase 75/76: DisagreementGapAnalyzer.

Analyzes heuristic disagreement distribution relative to disagreement_hard_floor.

With 5 heuristics (Phase 76), disagreement is quantized: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0.
With hard_floor=0.50 (Phase 81):
  - 1/5=0.20, 2/5=0.40 pass naturally (harmless disagreement)
  - 3/5=0.60 blocks → NEAR_THRESHOLD (gap=0.10, within NEAR_MISS_GAP=0.35)
  - 4/5=0.80 blocks → NEAR_THRESHOLD (gap=0.30, within NEAR_MISS_GAP=0.35)
  - 5/5=1.00 blocks → MODERATE (gap=0.50, within MODERATE_GAP=0.50)

Near-miss gap is 0.35 (covers two full quantization steps above floor=0.50).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

CLASS_NEAR_THRESHOLD = "NEAR_THRESHOLD"
CLASS_MODERATE = "MODERATE"
CLASS_FAR_ABOVE = "FAR_ABOVE"
CLASS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Gap direction is INVERTED: gap = score - threshold (positive = blocked)
NEAR_MISS_GAP = 0.35
MODERATE_GAP = 0.50
MIN_FAIL_COUNT = 3

_RECS = {
    CLASS_NEAR_THRESHOLD: "Raise disagreement_hard_floor to next quantized boundary (e.g. 0.50→0.61 to allow 3/5 disagreement)",
    CLASS_MODERATE: "2/3 heuristics oppose — investigate signal quality or reduce heuristic weight",
    CLASS_FAR_ABOVE: "All heuristics oppose direction — ML models may need retraining",
    CLASS_INSUFFICIENT_DATA: "Insufficient data — accumulate more scans",
}


def analyze(recorder, disagreement_hard_floor: float) -> Dict[str, Any]:
    """Analyze disagreement distribution vs hard floor."""
    try:
        fail_dist = recorder.get_fail_distribution()
        pass_rate = recorder.get_pass_rate()
        all_dist = recorder.get_distribution()
    except Exception as exc:
        logger.error("DisagreementGapAnalyzer.analyze() failed: %s", exc)
        return _empty_result()

    total_count = all_dist.get("count", 0)
    fail_count = fail_dist.get("count", 0)

    if fail_count < MIN_FAIL_COUNT:
        return {
            "count": total_count, "fail_count": fail_count,
            "pass_rate": round(pass_rate, 4),
            "gap_at_p50": 0.0, "gap_at_p75": 0.0,
            "near_miss_count": 0, "near_miss_rate_pct": 0.0,
            "classification": CLASS_INSUFFICIENT_DATA,
            "recommendation": _RECS[CLASS_INSUFFICIENT_DATA],
        }

    p50 = fail_dist.get("p50", 0.0)
    # Gap direction INVERTED: score - threshold (positive = fail)
    gap_at_p50 = round(p50 - float(disagreement_hard_floor), 6)

    if gap_at_p50 <= NEAR_MISS_GAP:
        classification = CLASS_NEAR_THRESHOLD
    elif gap_at_p50 <= MODERATE_GAP:
        classification = CLASS_MODERATE
    else:
        classification = CLASS_FAR_ABOVE

    # Near-miss: blocked entries where gap <= NEAR_MISS_GAP
    try:
        with recorder._lock:
            fail_records = [r for r in recorder._records if r.get("hard_blocked", False)]
        near_miss_count = sum(
            1 for r in fail_records
            if float(r.get("disagreement", 0.0)) - float(disagreement_hard_floor) <= NEAR_MISS_GAP
        )
    except Exception:
        near_miss_count = 0

    near_miss_rate_pct = round(near_miss_count / fail_count * 100.0 if fail_count > 0 else 0.0, 2)

    return {
        "count": total_count, "fail_count": fail_count,
        "pass_rate": round(pass_rate, 4),
        "gap_at_p50": gap_at_p50,
        "gap_at_p75": round(fail_dist.get("p75", 0.0) - float(disagreement_hard_floor), 6),
        "near_miss_count": near_miss_count,
        "near_miss_rate_pct": near_miss_rate_pct,
        "classification": classification,
        "recommendation": _RECS[classification],
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "count": 0, "fail_count": 0, "pass_rate": 0.0,
        "gap_at_p50": 0.0, "gap_at_p75": 0.0,
        "near_miss_count": 0, "near_miss_rate_pct": 0.0,
        "classification": CLASS_INSUFFICIENT_DATA,
        "recommendation": _RECS[CLASS_INSUFFICIENT_DATA],
    }
