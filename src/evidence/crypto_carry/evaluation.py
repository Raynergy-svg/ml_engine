"""Authority-free evaluation for crypto-carry venue/cost/regime cells."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.crypto import carry_scorecard as scorecard
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import GateResult, GateStatus

from .models import CarryHeadResult, lane_id_for_carry


@dataclass(frozen=True)
class EvaluationParams:
    expected_cells_by_carry: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    min_cells: int = scorecard.MIN_CELLS
    net_sharpe_floor: float = scorecard.NET_SHARPE_FLOOR
    max_drawdown_limit: float = scorecard.MAX_DRAWDOWN_LIMIT
    max_margin_utilization: float = scorecard.MAX_MARGIN_UTILIZATION
    max_tracking_error: float = scorecard.MAX_TRACKING_ERROR
    min_capacity_usd: float = scorecard.MIN_CAPACITY_USD
    replay_tolerance: float = 1e-12


def _parse_cell(cell_id: str, data: bytes) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"cell {cell_id!r} is not valid UTF-8: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cell {cell_id!r} line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"cell {cell_id!r} line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"cell {cell_id!r} contains no rows")
    return rows


def _gate(gate_id: str, ok: bool, observed: Any, threshold: Any, reason: str) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        observed=observed,
        threshold=threshold,
        reason=reason,
    )


def _metrics(aggregate: Mapping[str, Any]) -> dict[str, float]:
    output = {"n_cells": float(aggregate["n_cells"])}
    for key in (
        "minimum_cell_net_sharpe",
        "maximum_cell_drawdown",
        "maximum_margin_utilization",
        "maximum_tracking_error",
        "minimum_capacity_usd",
    ):
        value = aggregate.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number == number and number not in (float("inf"), float("-inf")):
                output[key] = number
    output["tail_evidence_gate_passed"] = float(bool(aggregate["tail_evidence_gate_passed"]))
    output["counterparty_evidence_gate_passed"] = float(
        bool(aggregate["counterparty_evidence_gate_passed"])
    )
    return output


def evaluate_carry(
    campaign_id: str,
    carry_id: str,
    cards: list[dict[str, Any]],
    expected_cells: list[str],
    params: EvaluationParams,
) -> CarryHeadResult:
    aggregate = scorecard.aggregate_carry_scorecard(
        carry_id,
        cards,
        expected_cells,
        min_cells=params.min_cells,
        net_sharpe_floor=params.net_sharpe_floor,
        max_drawdown_limit=params.max_drawdown_limit,
        max_margin_utilization=params.max_margin_utilization,
        max_tracking_error=params.max_tracking_error,
        min_capacity_usd=params.min_capacity_usd,
    )
    verdict = aggregate["decision"]["verdict"]
    return_ok = bool(aggregate["return_gate_passed"])
    tail_ok = bool(aggregate["tail_evidence_gate_passed"])
    counterparty_ok = bool(aggregate["counterparty_evidence_gate_passed"])
    gates = (
        _gate(
            "cost_adjusted_return_and_capacity",
            return_ok,
            {
                "minimum_net_sharpe": aggregate["minimum_cell_net_sharpe"],
                "maximum_drawdown": aggregate["maximum_cell_drawdown"],
                "maximum_margin": aggregate["maximum_margin_utilization"],
                "maximum_tracking_error": aggregate["maximum_tracking_error"],
                "minimum_capacity_usd": aggregate["minimum_capacity_usd"],
            },
            aggregate["bars"],
            "every venue/cost/regime cell must clear after-cost, drawdown, margin, tracking and capacity bars",
        ),
        _gate(
            "mandatory_tail_evidence",
            tail_ok,
            aggregate["missing_tail_by_cell"],
            "all roadmap J4 tail scenarios evidenced in every cell",
            "Sharpe cannot compensate for missing basis, liquidation, venue, withdrawal, depeg or settlement evidence",
        ),
        _gate(
            "mandatory_counterparty_evidence",
            counterparty_ok,
            aggregate["unclassified_venues_by_cell"],
            "every venue has a counterparty-risk classification",
            "carry cannot pass without explicit venue counterparty evidence",
        ),
    )
    passed = all(gate.status == GateStatus.PASS for gate in gates)
    metrics = _metrics(aggregate)
    lane_id = lane_id_for_carry(campaign_id, carry_id)
    return CarryHeadResult(
        head_id=lane_id,
        lane_id=lane_id,
        campaign_id=campaign_id,
        carry_id=carry_id,
        verdict=verdict,
        verdict_basis=(
            "all return, tail and counterparty gates cleared"
            if passed
            else "one or more independent return, tail or counterparty gates failed"
        ),
        metrics=metrics,
        metric_tolerances={name: params.replay_tolerance for name in metrics},
        gates=gates,
        passed=passed,
        artifact_bytes=canonical_bytes(aggregate),
        media_type="application/json",
        n_cells=int(aggregate["n_cells"]),
        cell_ids=tuple(sorted(expected_cells)),
        temporal_holdout=f"{len(expected_cells)} venue-set/cost-model/regime cells",
        incumbent_comparison={"has_incumbent": False, "lane_mode": "shadow"},
    )


def evaluate_partitions(
    partitions: Mapping[str, bytes],
    params: EvaluationParams | None = None,
    *,
    campaign_id: str,
) -> tuple[CarryHeadResult, ...]:
    if not partitions:
        raise ValueError("no crypto-carry cell partitions supplied")
    params = params or EvaluationParams()
    cards_by_carry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell_id in sorted(partitions):
        rows = _parse_cell(cell_id, partitions[cell_id])
        card = scorecard.compute_cell_scorecard(cell_id, rows)
        cards_by_carry[str(card["carry_id"])].append(card)
    declared = params.expected_cells_by_carry
    results: list[CarryHeadResult] = []
    for carry_id in sorted(cards_by_carry):
        if declared:
            if carry_id not in declared:
                raise ValueError(f"undeclared carry {carry_id!r} produced cells")
            expected = list(declared[carry_id])
        else:
            expected = sorted(str(card["cell_id"]) for card in cards_by_carry[carry_id])
        results.append(
            evaluate_carry(campaign_id, carry_id, cards_by_carry[carry_id], expected, params)
        )
    if declared:
        missing = sorted(set(declared) - set(cards_by_carry))
        if missing:
            raise ValueError(f"declared carries with no supplied cells: {missing}")
    return tuple(results)


__all__ = ["EvaluationParams", "evaluate_carry", "evaluate_partitions"]
