"""US-007 — Mechanical ship gate.

The deploy decision is made by code against a pre-registered bar, so judgment
(and hope) cannot leak in. Changing any constant below requires an operator-signed
commit message ("gate-change: ...") per the PRD.

A FAIL on any single criterion means the project answer is "no deployable edge at
this bar" — and that is a legitimate, evidence-backed outcome, not a failure of
the work.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

# --- Pre-registered ship criteria (PRD: tasks/prd-fx-factor-portfolio.md) ---
MIN_NET_SHARPE = 0.40
MIN_POSITIVE_YEARS = 6
MIN_TOTAL_YEARS = 10
MAX_DRAWDOWN = 0.25
# ---------------------------------------------------------------------------


@dataclass
class GateVerdict:
    passed: bool
    criteria: Dict[str, dict]
    summary: str


def evaluate_gate(report: dict) -> GateVerdict:
    """Evaluate a backtest report dict (``asdict(BacktestResult)``) against the bar."""
    net_sharpe = float(report.get("net_sharpe", 0.0))
    positive_years = int(report.get("positive_years", 0))
    total_years = int(report.get("total_years", 0))
    max_dd = float(report.get("max_drawdown", 1.0))
    walk_forward = bool(report.get("walk_forward", False))

    criteria = {
        "net_sharpe": {
            "value": net_sharpe,
            "threshold": MIN_NET_SHARPE,
            "passed": net_sharpe >= MIN_NET_SHARPE,
        },
        "positive_years": {
            "value": positive_years,
            "threshold": MIN_POSITIVE_YEARS,
            "passed": positive_years >= MIN_POSITIVE_YEARS,
        },
        "history_length": {
            "value": total_years,
            "threshold": MIN_TOTAL_YEARS,
            "passed": total_years >= MIN_TOTAL_YEARS,
        },
        "max_drawdown": {
            "value": max_dd,
            "threshold": MAX_DRAWDOWN,
            "passed": max_dd <= MAX_DRAWDOWN,
        },
        "walk_forward": {
            "value": walk_forward,
            "threshold": True,
            "passed": walk_forward is True,
        },
    }
    passed = all(c["passed"] for c in criteria.values())
    failed = [k for k, c in criteria.items() if not c["passed"]]
    if passed:
        summary = (
            f"PASS — net Sharpe {net_sharpe:.2f}, {positive_years}/{total_years} "
            f"positive years, maxDD {max_dd:.1%}. Eligible for US-008 practice deploy."
        )
    else:
        summary = (
            f"FAIL — failing criteria: {', '.join(failed)}. "
            "No deployable edge at the pre-registered bar."
        )
    return GateVerdict(passed=passed, criteria=criteria, summary=summary)


def verdict_to_dict(v: GateVerdict) -> dict:
    return asdict(v)
