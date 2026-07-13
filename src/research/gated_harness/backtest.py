"""Strategy-agnostic return-series metrics and hard gates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


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
    clean = validate_returns(returns)
    equity = (1.0 + clean).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


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
        "final_equity": float((1.0 + clean).prod()),
    }


def hard_gate(
    metrics: dict[str, Any],
    *,
    minimum_history: int,
    maximum_drawdown: float,
    minimum_sharpe: float,
) -> dict[str, bool]:
    checks = {
        "minimum_history": int(metrics["n_observations"]) >= minimum_history,
        "maximum_drawdown": float(metrics["max_drawdown"]) >= -abs(maximum_drawdown),
        "minimum_sharpe": float(metrics["sharpe"]) >= minimum_sharpe,
    }
    return {**checks, "passed": all(checks.values())}

