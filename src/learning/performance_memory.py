"""Deterministic performance memory — rollups over the attributed outcome ledger.

Everything here is arithmetic over ledger rows. It updates on every close at
zero token cost and is the statistical substrate a reasoning layer would later
interpret. No model, no network, no clock.

Two things make this more than a stats dump:

**Residual health is a first-class alarm.** ``residual_health.status`` is
``ALARM`` if any row failed to reconcile. A residual is the gap between the
broker's own realized P&L and what the attribution model can account for, so a
non-zero one means the MODEL is wrong -- a bug to fix in code, never a question
to hand to an LLM. Surfacing it as a headline field stops a quietly-broken
decomposition from feeding confident-looking statistics downstream.

**Unavailable components stay unavailable.** Rows without a DecisionRecord
cannot attribute sizing or de-gross P&L. Those rows are counted in
``attribution_coverage`` rather than averaged in as zero -- averaging absent
data as zero is how "sizing costs nothing" becomes an accepted false fact.

Sample-size discipline: every bucket carries ``n``. Nothing here decides
anything; per ``.claude/rules/improvement.md`` a pattern needs 3+ observations
before it can even be proposed as a rule.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# A residual should sit at the float-noise floor. Anything above the ledger's
# own reconcile tolerance means the decomposition is wrong.
RESIDUAL_ALARM_TOLERANCE = 0.01

# Buckets below this are reported but explicitly marked not actionable.
MIN_N_FOR_ACTIONABLE = 3


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    """Divide, or return None. Never returns 0.0 for an undefined ratio."""
    return numerator / denominator if denominator else None


def _bucket_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Expectancy-style summary for one group of ledger rows."""
    if not rows:
        return {"n": 0, "actionable": False}

    pls = [float(r["attribution"]["gross_pl_home"]) for r in rows]
    wins = [p for p in pls if p > 0]
    losses = [p for p in pls if p < 0]
    total = sum(pls)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    holding = [
        float(r["attribution"]["holding_period_seconds"])
        for r in rows
        if r["attribution"].get("holding_period_seconds") is not None
    ]

    return {
        "n": len(rows),
        "actionable": len(rows) >= MIN_N_FOR_ACTIONABLE,
        "total_pl_home": total,
        "expectancy_home": total / len(rows),
        "win_rate": len(wins) / len(rows),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "avg_win_home": _safe_div(gross_win, len(wins)),
        "avg_loss_home": _safe_div(-gross_loss, len(losses)),
        "profit_factor": _safe_div(gross_win, gross_loss),
        "execution_cost_home": sum(float(r["attribution"]["execution_cost_home"]) for r in rows),
        "financing_home": sum(float(r["attribution"]["financing_home"]) for r in rows),
        "fees_home": sum(float(r["attribution"]["fees_home"]) for r in rows),
        "median_holding_seconds": (
            sorted(holding)[len(holding) // 2] if holding else None
        ),
    }


def _residual_health(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Is the attribution model itself trustworthy?"""
    if not rows:
        return {"status": "NO_DATA", "n": 0}

    residuals = [abs(float(r["attribution"]["residual_home"])) for r in rows]
    unreconciled = [r for r in rows if not r["attribution"].get("reconciled")]

    if unreconciled:
        status = "ALARM"
    elif max(residuals) > RESIDUAL_ALARM_TOLERANCE / 10:
        status = "DEGRADED"
    else:
        status = "OK"

    return {
        "status": status,
        "n": len(rows),
        "max_abs_residual_home": max(residuals),
        "mean_abs_residual_home": sum(residuals) / len(residuals),
        "unreconciled_count": len(unreconciled),
        "unreconciled_trade_ids": [r["outcome"]["trade_id"] for r in unreconciled][:20],
        "interpretation": (
            "Attribution reconciles against broker-realized P&L."
            if status == "OK"
            else "Residual exceeds the noise floor -- the attribution MODEL is wrong. "
                 "Fix the decomposition in code; do NOT widen the tolerance and do NOT "
                 "route this to a reasoning layer."
        ),
    }


def _attribution_coverage(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """How much of the decomposition is actually available, and why not."""
    if not rows:
        return {"n": 0}
    missing: Dict[str, int] = defaultdict(int)
    for row in rows:
        for field in row["attribution"].get("unavailable", []):
            missing[field] += 1
    with_decision = sum(1 for r in rows if r.get("decision_id"))
    return {
        "n": len(rows),
        "rows_with_decision_record": with_decision,
        "rows_without_decision_record": len(rows) - with_decision,
        "unavailable_component_counts": dict(sorted(missing.items())),
        "note": (
            "Components listed here are UNAVAILABLE, not zero. Sizing and de-gross "
            "attribution require the pre-trade DecisionRecord; rows predating the "
            "decision ledger cannot supply it."
        ),
    }


def build_memory(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute the full deterministic rollup over attributed ledger rows.

    Args:
        rows: ledger rows as produced by ``outcome_ledger.build_row``.

    Returns:
        A JSON-serialisable snapshot. ``residual_health.status`` should be
        checked before trusting any other figure in it.
    """
    valid: List[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or "attribution" not in row or "outcome" not in row:
            logger.warning("performance_memory: skipping malformed ledger row")
            continue
        valid.append(row)

    by_instrument: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_exit: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_lane: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in valid:
        by_instrument[row["outcome"]["instrument"]].append(row)
        by_exit[row["attribution"]["exit_bucket"]].append(row)
        by_lane[row.get("lane") or "unknown"].append(row)

    overall = _bucket_stats(valid)
    execution_drag = None
    if overall.get("n"):
        gross_wins = sum(
            float(r["attribution"]["gross_pl_home"])
            for r in valid
            if float(r["attribution"]["gross_pl_home"]) > 0
        )
        execution_drag = _safe_div(overall["execution_cost_home"], gross_wins)

    return {
        "schema_version": 1,
        "residual_health": _residual_health(valid),
        "attribution_coverage": _attribution_coverage(valid),
        "overall": overall,
        "execution_drag_vs_gross_wins": execution_drag,
        "by_instrument": {k: _bucket_stats(v) for k, v in sorted(by_instrument.items())},
        "by_exit_bucket": {k: _bucket_stats(v) for k, v in sorted(by_exit.items())},
        "by_lane": {k: _bucket_stats(v) for k, v in sorted(by_lane.items())},
        "caveats": [
            f"Buckets with n < {MIN_N_FOR_ACTIONABLE} are marked actionable=false.",
            "A pattern needs 3+ observations before it may be proposed as a rule "
            "(.claude/rules/improvement.md). This snapshot proposes nothing.",
            "Unavailable attribution components are excluded, never averaged as zero.",
        ],
    }


__all__ = [
    "MIN_N_FOR_ACTIONABLE",
    "RESIDUAL_ALARM_TOLERANCE",
    "build_memory",
]
