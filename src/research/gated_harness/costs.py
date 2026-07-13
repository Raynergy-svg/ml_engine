"""Declared, reproducible transaction-cost scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostScenarioResult:
    cost_bps: float
    total_cost: float
    net_returns: pd.Series


def apply_turnover_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    *,
    cost_bps: float,
) -> CostScenarioResult:
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    gross, turns = gross_returns.align(turnover, join="left")
    turns = turns.fillna(0.0)
    if bool((turns < 0).any()) or not np.isfinite(turns.to_numpy()).all():
        raise ValueError("turnover must be finite and non-negative")
    costs = turns * (cost_bps / 10_000.0)
    return CostScenarioResult(
        cost_bps=float(cost_bps),
        total_cost=float(costs.sum()),
        net_returns=gross - costs,
    )


def evaluate_cost_scenarios(
    gross_returns: pd.Series,
    turnover: pd.Series,
    scenarios_bps: tuple[float, ...],
) -> dict[str, CostScenarioResult]:
    if not scenarios_bps:
        raise ValueError("at least one cost scenario is required")
    return {
        f"{float(cost):g}bps": apply_turnover_costs(
            gross_returns, turnover, cost_bps=float(cost)
        )
        for cost in scenarios_bps
    }

