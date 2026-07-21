"""Segment expectancy with a built-in out-of-sample gate.

Finds trade populations with positive net expectancy AFTER execution costs --
and refuses to call any of them actionable until the verdict survives a
temporal out-of-sample split.

Why the gate is in the code rather than the process
---------------------------------------------------
Ranking 9 instruments by P&L and suppressing the worst is guaranteed to improve
the in-sample number. On this corpus it looked like a +$1,499 improvement from
dropping one pair; out-of-sample that same rule delivered +$51.47, because the
pair's at-mid edge flipped sign between halves. Two of four pairs with adequate
samples flipped. A finding that cannot survive a split is a description of the
past, not a rule.

So ``segment_report`` will not return ``PROFITABLE_AFTER_COST`` or
``NO_EDGE_AT_MID`` as actionable unless both halves agree on the sign. Callers
that want the raw in-sample numbers can still read them; they just cannot
mistake them for validated.

The cost model (this is the part that is easy to get wrong)
-----------------------------------------------------------
``gross_pl_home`` from the attribution layer is ALREADY net of spread -- you
buy at the ask and sell at the bid, so the spread is embedded in the fill
prices. ``execution_cost_home`` is the measured half-spread, reported for
attribution, NOT a further deduction.

    realized = gross_pl_home                        <- what actually happened
    at_mid   = gross_pl_home + execution_cost_home  <- counterfactual, zero spread

Subtracting execution cost from realized P&L double-counts the spread and makes
every segment look roughly twice as bad as it is. ``at_mid`` is what separates
"this strategy has no edge" from "this strategy has edge that costs consumed".
Those two diagnoses have completely different remedies.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Minimum observations per half before a split verdict means anything.
MIN_N_PER_HALF = 3

# The band of cut points a verdict must survive, as fractions of the ordered
# segment. Testing only the exact midpoint lets an unstable segment pass on
# boundary luck; testing the whole range would test cuts with almost no data on
# one side. The middle 40% is the range where both sides carry real weight.
_CUT_BAND_LO = 0.30
_CUT_BAND_HI = 0.70

# Verdicts. The distinction between the first two is the whole point: one is a
# signal problem, the other is a cost problem, and they need opposite fixes.
NO_EDGE_AT_MID = "NO_EDGE_AT_MID"                      # negative even at zero cost -> signal is bad
EDGE_CONSUMED_BY_COST = "EDGE_CONSUMED_BY_COST"        # positive at mid, negative realized -> cost problem
PROFITABLE_AFTER_COST = "PROFITABLE_AFTER_COST"        # positive realized
INSUFFICIENT = "INSUFFICIENT"                          # not enough data to say anything


def _sum(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(float(r["attribution"][field]) for r in rows)


def segment_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Realized vs at-mid economics for one population of ledger rows."""
    if not rows:
        # Full shape with None, not a 2-key dict: a consumer reading any metric
        # would otherwise KeyError on an empty segment. None means UNDEFINED --
        # 0.0 would read as "measured, and it was zero".
        return {
            "n": 0, "verdict": INSUFFICIENT,
            "realized_home": None, "spread_home": None, "at_mid_home": None,
            "realized_per_trade": None, "spread_per_trade": None,
            "win_rate": None, "spread_share_of_edge": None,
        }

    realized = _sum(rows, "gross_pl_home")
    spread = _sum(rows, "execution_cost_home")
    at_mid = realized + spread
    wins = sum(1 for r in rows if float(r["attribution"]["gross_pl_home"]) > 0)

    if at_mid <= 0:
        verdict = NO_EDGE_AT_MID
    elif realized > 0:
        verdict = PROFITABLE_AFTER_COST
    else:
        verdict = EDGE_CONSUMED_BY_COST

    return {
        "n": len(rows),
        "realized_home": realized,
        "spread_home": spread,
        "at_mid_home": at_mid,
        "realized_per_trade": realized / len(rows),
        "spread_per_trade": spread / len(rows),
        "win_rate": wins / len(rows),
        # Share of the zero-cost edge eaten by spread. None when there is no
        # edge to eat -- a percentage of a negative number is meaningless.
        "spread_share_of_edge": (spread / at_mid) if at_mid > 0 else None,
        "verdict": verdict,
    }


def temporal_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], str] = lambda r: r["outcome"]["entry_time"],
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Split chronologically into two halves. Earlier half first."""
    ordered = sorted(rows, key=key)
    midpoint = len(ordered) // 2
    return ordered[:midpoint], ordered[midpoint:]


def validate_segment(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_n_per_half: int = MIN_N_PER_HALF,
) -> Dict[str, Any]:
    """Does this segment's at-mid edge keep its sign no matter where you cut it?

    A SINGLE midpoint split is not enough. On the real corpus, USD_CHF's
    verdict agreed at cut 12 and 14 but flipped at 13, 15, 16 and 17 -- the
    "validated" answer was decided by where the midpoint happened to land, not
    by anything about the strategy. A gate that can be passed by boundary luck
    is not a gate.

    So the sign is tested at EVERY cut point in the middle band
    (``_CUT_BAND_LO``..``_CUT_BAND_HI`` of the ordered rows) and the segment is
    actionable only if it agrees at ALL of them. ``stability`` reports the
    fraction that agreed, so a near-miss is visible rather than merely refused.

    Fail-closed: too few rows to form the band at all means not actionable.
    """
    ordered = sorted(rows, key=lambda r: r["outcome"]["entry_time"])
    n = len(ordered)

    lo = max(min_n_per_half, int(n * _CUT_BAND_LO))
    hi = min(n - min_n_per_half, int(n * _CUT_BAND_HI))
    cuts = list(range(lo, hi + 1))

    midpoint_first, midpoint_second = temporal_split(ordered)
    a, b = segment_stats(midpoint_first), segment_stats(midpoint_second)
    adequate = bool(cuts) and a["n"] >= min_n_per_half and b["n"] >= min_n_per_half

    agreements: List[bool] = []
    for cut in cuts:
        first_stats = segment_stats(ordered[:cut])
        second_stats = segment_stats(ordered[cut:])
        agreements.append(
            (first_stats["at_mid_home"] > 0) == (second_stats["at_mid_home"] > 0)
        )

    stability = (sum(agreements) / len(agreements)) if agreements else 0.0
    sign_agrees = bool(agreements) and all(agreements)

    if not adequate:
        reason = (
            f"insufficient: need n>={min_n_per_half} per side across the cut band "
            f"(n={n}, cuts tested={len(cuts)})"
        )
    elif sign_agrees:
        reason = (
            f"validated: at-mid edge keeps its sign at all {len(cuts)} tested cut points"
        )
    else:
        reason = (
            f"REJECTED: at-mid edge sign is unstable -- agreed at only "
            f"{sum(agreements)}/{len(agreements)} cut points. In-sample artifact; "
            "do not act on it."
        )

    return {
        "overall": segment_stats(rows),
        "first_half": a,
        "second_half": b,
        "adequate_samples": adequate,
        "sign_agrees": sign_agrees,
        "stability": stability,
        "cuts_tested": len(cuts),
        "actionable": bool(adequate and sign_agrees),
        "reason": reason,
    }


def segment_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    dimensions: Optional[Mapping[str, Callable[[Mapping[str, Any]], str]]] = None,
    min_n_per_half: int = MIN_N_PER_HALF,
) -> Dict[str, Any]:
    """Full segment sweep with the out-of-sample gate applied to every segment.

    Args:
        rows: attributed ledger rows.
        dimensions: name -> function extracting a segment label from a row.
            Defaults to instrument and exit bucket.
        min_n_per_half: sample floor per half.

    Returns:
        ``{"overall": ..., "dimensions": {dim: {label: validation}},
           "actionable_segments": [...]}``. ``actionable_segments`` is the ONLY
        list a caller should consider acting on.
    """
    dims = dimensions or {
        "instrument": lambda r: r["outcome"]["instrument"],
        "exit_bucket": lambda r: r["attribution"]["exit_bucket"],
    }

    valid = [r for r in rows if isinstance(r, Mapping) and "attribution" in r and "outcome" in r]
    if len(valid) != len(list(rows)):
        logger.warning("segment_analysis: skipped %d malformed rows", len(list(rows)) - len(valid))

    out: Dict[str, Any] = {
        "overall": segment_stats(valid),
        "dimensions": {},
        "actionable_segments": [],
        "note": (
            "at_mid = realized + spread. realized is ALREADY net of spread; do not "
            "subtract execution cost from it. Only `actionable_segments` survived "
            "the temporal out-of-sample split."
        ),
    }

    for dim_name, extract in dims.items():
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in valid:
            grouped[extract(row)].append(row)

        dim_out: Dict[str, Any] = {}
        for label, seg_rows in sorted(grouped.items()):
            validation = validate_segment(seg_rows, min_n_per_half=min_n_per_half)
            dim_out[label] = validation
            if validation["actionable"]:
                out["actionable_segments"].append({
                    "dimension": dim_name,
                    "label": label,
                    "verdict": validation["overall"]["verdict"],
                    "n": validation["overall"]["n"],
                    "realized_home": validation["overall"]["realized_home"],
                    "at_mid_home": validation["overall"]["at_mid_home"],
                })
        out["dimensions"][dim_name] = dim_out

    return out


__all__ = [
    "EDGE_CONSUMED_BY_COST",
    "INSUFFICIENT",
    "MIN_N_PER_HALF",
    "NO_EDGE_AT_MID",
    "PROFITABLE_AFTER_COST",
    "segment_report",
    "segment_stats",
    "temporal_split",
    "validate_segment",
]
