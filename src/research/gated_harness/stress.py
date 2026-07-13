"""Placebo, causality, sensitivity, and cost stress controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from .backtest import summarize_returns
from .costs import evaluate_cost_scenarios


def placebo_control(
    observed: pd.Series,
    placebo: pd.Series,
    *,
    maximum_absolute_placebo_sharpe: float,
) -> dict[str, Any]:
    observed_metrics = summarize_returns(observed)
    placebo_metrics = summarize_returns(placebo)
    return {
        "observed": observed_metrics,
        "placebo": placebo_metrics,
        "passed": abs(float(placebo_metrics["sharpe"])) <= maximum_absolute_placebo_sharpe,
    }


def assert_prefix_causality(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    original: pd.DataFrame,
    future_mutated: pd.DataFrame,
    *,
    cutoff: Any,
) -> bool:
    """Prove that modifying only future inputs cannot alter past outputs."""
    left = build(original).loc[:cutoff]
    right = build(future_mutated).loc[:cutoff]
    try:
        pd.testing.assert_frame_equal(left, right)
    except AssertionError:
        return False
    return True


def stress_matrix(
    gross_returns: pd.Series,
    turnover: pd.Series,
    *,
    cost_scenarios_bps: tuple[float, ...],
    parameter_returns: Mapping[str, pd.Series],
) -> dict[str, Any]:
    costs = evaluate_cost_scenarios(gross_returns, turnover, cost_scenarios_bps)
    return {
        "cost_scenarios": {
            name: {**summarize_returns(result.net_returns), "total_cost": result.total_cost}
            for name, result in costs.items()
        },
        "parameter_sensitivity": {
            name: summarize_returns(series) for name, series in sorted(parameter_returns.items())
        },
    }
