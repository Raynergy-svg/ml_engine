"""Strategy-agnostic return-series metrics and hard gates.

Drawdown convention: **positive fraction** — see
:mod:`src.research.drawdown_convention` for why, and for the collision this
module used to be one half of. Until 2026-07-30 :func:`max_drawdown` here
returned a NEGATIVE value while every other producer in the repo returned a
positive magnitude under the same ``max_drawdown`` key, and :func:`hard_gate`
compared with ``>= -abs(limit)`` — so a positive-convention 0.863 (86.3%
drawdown) satisfied a 25% gate. Both sides of that comparison now go through
the shared fail-closed validator, which refuses wrong-signed input instead of
comparing it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.research.drawdown_convention import (
    CANONICAL_DRAWDOWN_KEY,
    DRAWDOWN_CONVENTION,
    drawdown_fraction,
    drawdown_limit,
    read_drawdown,
)


def validate_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    clean = returns.dropna().astype(float)
    if clean.empty or not np.isfinite(clean.to_numpy()).all():
        raise ValueError("returns must contain finite observations")
    if isinstance(clean.index, pd.DatetimeIndex) and not clean.index.is_monotonic_increasing:
        raise ValueError("returns must be chronological")
    return clean


def max_drawdown(returns: pd.Series) -> float:
    """Max peak-to-trough decline as a NON-NEGATIVE fraction (0.25 == 25%).

    Canonical convention (:data:`DRAWDOWN_CONVENTION`). Validated on the way
    out so this function can never itself become the wrong-signed producer.
    """
    clean = validate_returns(returns)
    equity = (1.0 + clean).cumprod()
    decline = float((equity / equity.cummax() - 1.0).min())
    return drawdown_fraction(
        max(0.0, -decline),
        key=CANONICAL_DRAWDOWN_KEY,
        source="gated_harness.backtest.max_drawdown",
    )


def summarize_returns(returns: pd.Series, *, periods_per_year: int = 252) -> dict[str, Any]:
    clean = validate_returns(returns)
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    sharpe = float(clean.mean() / std * math.sqrt(periods_per_year)) if std > 1e-12 else 0.0
    return {
        "n_observations": int(len(clean)),
        "annualized_return": float(clean.mean() * periods_per_year),
        "annualized_volatility": float(std * math.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(clean),
        "drawdown_convention": DRAWDOWN_CONVENTION,
        "final_equity": float((1.0 + clean).prod()),
    }


def hard_gate(
    metrics: dict[str, Any],
    *,
    minimum_history: int,
    maximum_drawdown: float,
    minimum_sharpe: float,
) -> dict[str, bool]:
    """Hard admissibility gate over a ``summarize_returns``-shaped metrics dict.

    ``metrics["max_drawdown"]`` and ``maximum_drawdown`` must both be canonical
    positive fractions. A wrong-signed or percent-scaled value raises
    :class:`~src.research.drawdown_convention.DrawdownConventionError` — it is
    never silently compared, because a silently satisfied guard is how an 86.3%
    drawdown used to clear a 25% budget.
    """
    source = "gated_harness.backtest.hard_gate"
    observed_drawdown = read_drawdown(metrics, source=source)
    budget = drawdown_limit(maximum_drawdown, source=source)
    checks = {
        "minimum_history": int(metrics["n_observations"]) >= minimum_history,
        "maximum_drawdown": observed_drawdown <= budget,
        "minimum_sharpe": float(metrics["sharpe"]) >= minimum_sharpe,
    }
    return {**checks, "passed": all(checks.values())}
