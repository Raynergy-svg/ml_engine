"""Deterministic, authority-free scorecard for crypto cash-and-carry evidence.

The parallel unit is ``venue set × cost model × regime``.  Return metrics and
risk evidence are deliberately independent: no Sharpe ratio can substitute for
complete liquidation-tail and counterparty classifications (roadmap J4).
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from src.research.drawdown_convention import drawdown_fraction, drawdown_limit

ANN = 365.0
MIN_PERIODS = 8
MIN_CELLS = 3
NET_SHARPE_FLOOR = 0.5
MAX_DRAWDOWN_LIMIT = 0.25
MAX_MARGIN_UTILIZATION = 0.70
MAX_TRACKING_ERROR = 0.05
MIN_CAPACITY_USD = 10_000.0

REQUIRED_TAIL_SCENARIOS = (
    "basis_blowout",
    "liquidation",
    "venue_failure",
    "withdrawal_suspension",
    "stablecoin_depeg",
    "cross_venue_settlement",
)

VERDICT_CLEARS = "carry_evidence_clears"
VERDICT_INSUFFICIENT = "insufficient_cells"
VERDICT_FAILS_RETURN = "fails_cost_adjusted_return"
VERDICT_MISSING_TAIL = "missing_tail_evidence"
VERDICT_MISSING_COUNTERPARTY = "missing_counterparty_evidence"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _sharpe(values: Sequence[float]) -> float | None:
    if len(values) < MIN_PERIODS:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance <= 0.0:
        return None
    return mean / math.sqrt(variance) * math.sqrt(ANN)


def _max_drawdown(values: Sequence[float]) -> float:
    """Max peak-to-trough decline as a NON-NEGATIVE fraction (0.25 == 25%).

    Canonical convention -- see :mod:`src.research.drawdown_convention`.
    """
    equity = peak = 1.0
    maximum = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0:
            maximum = max(maximum, (peak - equity) / peak)
    return drawdown_fraction(maximum, source="crypto.carry_scorecard._max_drawdown")


def _evidence_digest(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("evidence_digest"), str)
        and _SHA256.fullmatch(record["evidence_digest"]) is not None
    )


def _counterparty_record_valid(record: Any) -> bool:
    return (
        _evidence_digest(record)
        and isinstance(record.get("classification"), str)
        and bool(record["classification"].strip())
        and isinstance(record.get("as_of"), str)
        and bool(record["as_of"].strip())
    )


def _tail_record_valid(record: Any) -> bool:
    if not _evidence_digest(record):
        return False
    if not isinstance(record.get("source"), str) or not record["source"].strip():
        return False
    loss = record.get("loss_fraction")
    return (
        isinstance(loss, (int, float))
        and not isinstance(loss, bool)
        and math.isfinite(float(loss))
        and float(loss) >= 0.0
    )


def compute_cell_scorecard(cell_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one venue-set/cost-model/regime cell, failing closed on lineage."""
    if not rows:
        raise ValueError(f"cell {cell_id!r} has no rows")
    first = rows[0]
    carry_id = first.get("carry_id")
    venue_set = first.get("venue_set")
    cost_model = first.get("cost_model")
    regime = first.get("regime")
    for name, value in (
        ("carry_id", carry_id),
        ("venue_set", venue_set),
        ("cost_model", cost_model),
        ("regime", regime),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"cell {cell_id!r} missing non-empty {name}")
        if "::" in value:
            raise ValueError(f"cell {cell_id!r} {name} cannot contain '::'")
    expected_cell_id = f"{carry_id}::{venue_set}::{cost_model}::{regime}"
    if cell_id != expected_cell_id:
        raise ValueError(
            f"cell id {cell_id!r} does not match row identity {expected_cell_id!r}"
        )

    classifications = first.get("counterparty_classifications")
    if not isinstance(classifications, dict):
        raise ValueError(f"cell {cell_id!r} counterparty_classifications must be an object")
    raw_venues = str(venue_set).split("+")
    if any(not part.strip() for part in raw_venues):
        raise ValueError(f"cell {cell_id!r} venue_set has an empty venue component")
    venues = tuple(part.strip() for part in raw_venues)
    if len(venues) != len(set(venues)):
        raise ValueError(f"cell {cell_id!r} venue_set contains duplicate venues")
    counterparty_complete = bool(venues) and all(
        _counterparty_record_valid(classifications.get(venue))
        for venue in venues
    )
    tail = first.get("tail_evidence")
    if not isinstance(tail, dict):
        raise ValueError(f"cell {cell_id!r} tail_evidence must be an object")
    tail_complete = all(
        _tail_record_valid(tail.get(scenario)) for scenario in REQUIRED_TAIL_SCENARIOS
    )

    net_returns: list[float] = []
    gross_capture: list[float] = []
    costs: list[float] = []
    mark_pnl: list[float] = []
    tracking_errors: list[float] = []
    margin: list[float] = []
    capacity: list[float] = []
    previous_period: float | None = None
    for line_number, row in enumerate(rows, start=1):
        for key, expected in (
            ("cell_id", cell_id),
            ("carry_id", carry_id),
            ("venue_set", venue_set),
            ("cost_model", cost_model),
            ("regime", regime),
        ):
            if row.get(key) != expected:
                raise ValueError(f"cell {cell_id!r} row {line_number} has inconsistent {key}")
        if row.get("counterparty_classifications") != classifications:
            raise ValueError(f"cell {cell_id!r} row {line_number} changes counterparty evidence")
        if row.get("tail_evidence") != tail:
            raise ValueError(f"cell {cell_id!r} row {line_number} changes tail evidence")
        period = _finite(row.get("period_index"), "period_index")
        if previous_period is not None and period <= previous_period:
            raise ValueError(f"cell {cell_id!r} period_index is not strictly increasing")
        previous_period = period
        net_return = _finite(row.get("net_return"), "net_return")
        if net_return <= -1.0:
            raise ValueError("net_return must be greater than -1")
        net_returns.append(net_return)
        gross = _finite(row.get("gross_funding_capture"), "gross_funding_capture")
        gross_capture.append(gross)
        cost = _finite(row.get("two_leg_cost"), "two_leg_cost")
        if cost < 0:
            raise ValueError("two_leg_cost must be non-negative")
        costs.append(cost)
        mark = _finite(row.get("spot_perp_mark_pnl"), "spot_perp_mark_pnl")
        mark_pnl.append(mark)
        expected_net = gross + mark - cost
        if not math.isclose(net_return, expected_net, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "net_return must equal gross_funding_capture + "
                "spot_perp_mark_pnl - two_leg_cost"
            )
        tracking_errors.append(abs(_finite(row.get("tracking_error"), "tracking_error")))
        margin_value = _finite(row.get("margin_utilization"), "margin_utilization")
        if not 0.0 <= margin_value <= 1.0:
            raise ValueError("margin_utilization must be within [0, 1]")
        margin.append(margin_value)
        capacity_value = _finite(row.get("capacity_usd"), "capacity_usd")
        if capacity_value < 0:
            raise ValueError("capacity_usd must be non-negative")
        capacity.append(capacity_value)

    return {
        "cell_id": cell_id,
        "carry_id": carry_id,
        "venue_set": venue_set,
        "cost_model": cost_model,
        "regime": regime,
        "n_periods": len(rows),
        "net_sharpe": _sharpe(net_returns),
        "net_annual_return": sum(net_returns) / len(net_returns) * ANN,
        "gross_funding_capture": sum(gross_capture),
        "two_leg_cost": sum(costs),
        "max_drawdown": _max_drawdown(net_returns),
        "max_tracking_error": max(tracking_errors),
        "max_margin_utilization": max(margin),
        "min_capacity_usd": min(capacity),
        "tail_evidence_complete": tail_complete,
        "missing_tail_scenarios": [
            scenario
            for scenario in REQUIRED_TAIL_SCENARIOS
            if not _tail_record_valid(tail.get(scenario))
        ],
        "counterparty_evidence_complete": counterparty_complete,
        "unclassified_venues": [
            venue for venue in venues
            if not _counterparty_record_valid(classifications.get(venue))
        ],
    }


def aggregate_carry_scorecard(
    carry_id: str,
    cards: Sequence[Mapping[str, Any]],
    expected_cell_ids: Sequence[str],
    *,
    min_cells: int = MIN_CELLS,
    net_sharpe_floor: float = NET_SHARPE_FLOOR,
    max_drawdown_limit: float = MAX_DRAWDOWN_LIMIT,
    max_margin_utilization: float = MAX_MARGIN_UTILIZATION,
    max_tracking_error: float = MAX_TRACKING_ERROR,
    min_capacity_usd: float = MIN_CAPACITY_USD,
) -> dict[str, Any]:
    """Aggregate all declared cells and refuse missing or duplicate results."""
    expected = list(expected_cell_ids)
    if len(expected) != len(set(expected)):
        raise ValueError(f"carry {carry_id!r} declares duplicate expected cells")
    produced = [str(card.get("cell_id")) for card in cards]
    if len(produced) != len(set(produced)):
        raise ValueError(f"carry {carry_id!r} received duplicate cells")
    if set(produced) != set(expected):
        raise ValueError(
            f"carry {carry_id!r} cell set mismatch: "
            f"missing={sorted(set(expected)-set(produced))}, "
            f"unexpected={sorted(set(produced)-set(expected))}"
        )
    if any(card.get("carry_id") != carry_id for card in cards):
        raise ValueError(f"carry {carry_id!r} received a foreign cell")

    sharpes = [float(card["net_sharpe"]) for card in cards if card.get("net_sharpe") is not None]
    all_return_cells_defined = len(sharpes) == len(cards)
    minimum_sharpe = min(sharpes) if sharpes else None
    maximum_drawdown = max(
        (
            drawdown_fraction(
                card["max_drawdown"], source="crypto.carry_scorecard.aggregate"
            )
            for card in cards
        ),
        default=None,
    )
    maximum_margin = max((float(card["max_margin_utilization"]) for card in cards), default=None)
    maximum_tracking = max((float(card["max_tracking_error"]) for card in cards), default=None)
    minimum_capacity = min((float(card["min_capacity_usd"]) for card in cards), default=None)
    return_gate = (
        len(cards) >= min_cells
        and all_return_cells_defined
        and minimum_sharpe is not None
        and minimum_sharpe >= net_sharpe_floor
        and maximum_drawdown is not None
        and maximum_drawdown
        <= drawdown_limit(
            max_drawdown_limit, source="crypto.carry_scorecard.aggregate"
        )
        and maximum_margin is not None
        and maximum_margin <= max_margin_utilization
        and maximum_tracking is not None
        and maximum_tracking <= max_tracking_error
        and minimum_capacity is not None
        and minimum_capacity >= min_capacity_usd
    )
    tail_gate = bool(cards) and all(bool(card["tail_evidence_complete"]) for card in cards)
    counterparty_gate = bool(cards) and all(
        bool(card["counterparty_evidence_complete"]) for card in cards
    )
    if len(cards) < min_cells:
        verdict = VERDICT_INSUFFICIENT
    elif not tail_gate:
        verdict = VERDICT_MISSING_TAIL
    elif not counterparty_gate:
        verdict = VERDICT_MISSING_COUNTERPARTY
    elif not return_gate:
        verdict = VERDICT_FAILS_RETURN
    else:
        verdict = VERDICT_CLEARS
    return {
        "carry_id": carry_id,
        "n_cells": len(cards),
        "cell_ids": sorted(produced),
        "minimum_cell_net_sharpe": minimum_sharpe,
        "maximum_cell_drawdown": maximum_drawdown,
        "maximum_margin_utilization": maximum_margin,
        "maximum_tracking_error": maximum_tracking,
        "minimum_capacity_usd": minimum_capacity,
        "return_gate_passed": return_gate,
        "tail_evidence_gate_passed": tail_gate,
        "counterparty_evidence_gate_passed": counterparty_gate,
        "missing_tail_by_cell": {
            str(card["cell_id"]): card["missing_tail_scenarios"]
            for card in cards if card["missing_tail_scenarios"]
        },
        "unclassified_venues_by_cell": {
            str(card["cell_id"]): card["unclassified_venues"]
            for card in cards if card["unclassified_venues"]
        },
        "bars": {
            "min_cells": min_cells,
            "net_sharpe_floor": net_sharpe_floor,
            "max_drawdown_limit": max_drawdown_limit,
            "max_margin_utilization": max_margin_utilization,
            "max_tracking_error": max_tracking_error,
            "min_capacity_usd": min_capacity_usd,
        },
        "decision": {"verdict": verdict},
    }


__all__ = [
    "REQUIRED_TAIL_SCENARIOS",
    "VERDICT_CLEARS",
    "VERDICT_INSUFFICIENT",
    "VERDICT_FAILS_RETURN",
    "VERDICT_MISSING_TAIL",
    "VERDICT_MISSING_COUNTERPARTY",
    "compute_cell_scorecard",
    "aggregate_carry_scorecard",
]
