"""Authority-free evaluation for crypto-carry venue/cost/regime cells."""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.crypto import carry_scorecard as scorecard
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import GateResult, GateStatus

from .manifests import FORWARD_LEDGER_PREFIX
from .models import CarryHeadResult, lane_id_for_carry

_LEDGER_REQUIRED_FIELDS = ("asof_date", "today_net_return", "gross_leverage", "today_turnover")


@dataclass(frozen=True)
class EvaluationParams:
    expected_cells_by_carry: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    expected_ledger_carries: tuple[str, ...] = ()
    min_cells: int = scorecard.MIN_CELLS
    net_sharpe_floor: float = scorecard.NET_SHARPE_FLOOR
    max_drawdown_limit: float = scorecard.MAX_DRAWDOWN_LIMIT
    max_margin_utilization: float = scorecard.MAX_MARGIN_UTILIZATION
    max_tracking_error: float = scorecard.MAX_TRACKING_ERROR
    min_capacity_usd: float = scorecard.MIN_CAPACITY_USD
    replay_tolerance: float = 1e-12


def _is_forward_ledger_partition(partition_id: str) -> bool:
    return partition_id.startswith(FORWARD_LEDGER_PREFIX)


def _carry_from_ledger_partition_id(partition_id: str) -> str:
    return partition_id[len(FORWARD_LEDGER_PREFIX):]


def _parse_ledger_rows(partition_id: str, data: bytes) -> list[dict[str, Any]]:
    """Parse one carry's REAL forward-shadow ledger partition (JSONL,
    src.crypto.carry_shadow.record_shadow_cycle row shape). Every required
    field is always written by the producer — none are optional here."""
    rows: list[dict[str, Any]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"forward ledger {partition_id!r} is not valid UTF-8: {exc}") from exc
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"forward ledger {partition_id!r} line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"forward ledger {partition_id!r} line {line_number} is not a JSON object")
        missing = sorted(set(_LEDGER_REQUIRED_FIELDS) - set(row))
        if missing:
            raise ValueError(f"forward ledger {partition_id!r} line {line_number} missing fields {missing}")
        if not isinstance(row["asof_date"], str) or not row["asof_date"]:
            raise ValueError(f"forward ledger {partition_id!r} line {line_number} has an invalid asof_date")
        for name in ("today_net_return", "gross_leverage", "today_turnover"):
            value = row[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"forward ledger {partition_id!r} line {line_number} has non-finite {name}")
        if row["gross_leverage"] < 0 or row["today_turnover"] < 0:
            raise ValueError(f"forward ledger {partition_id!r} line {line_number} has a negative leverage or turnover")
        rows.append(row)
    if not rows:
        raise ValueError(f"forward ledger {partition_id!r} contains no rows")
    return rows


def build_forward_ledger_return_contract(rows: list[dict[str, Any]]) -> bytes:
    """Roadmap §14 standardized strategy-return/exposure contract, built from
    a carry's real forward-shadow ledger rows — one JSONL row per trading
    day, sorted ascending, deduplicated by ``asof_date`` (last-occurrence-
    wins). All four contract fields are populated from real, always-present
    ledger data."""
    by_date: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_date[row["asof_date"]] = {
            "date": row["asof_date"],
            "net_return": float(row["today_net_return"]),
            "gross_exposure": float(row["gross_leverage"]),
            "turnover": float(row["today_turnover"]),
        }
    lines = [json.dumps(by_date[date], sort_keys=True) for date in sorted(by_date)]
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


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
    cell_partitions = {k: v for k, v in partitions.items() if not _is_forward_ledger_partition(k)}
    ledger_partitions = {k: v for k, v in partitions.items() if _is_forward_ledger_partition(k)}
    if not cell_partitions:
        raise ValueError("no crypto-carry cell partitions supplied")
    params = params or EvaluationParams()
    cards_by_carry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell_id in sorted(cell_partitions):
        rows = _parse_cell(cell_id, cell_partitions[cell_id])
        card = scorecard.compute_cell_scorecard(cell_id, rows)
        cards_by_carry[str(card["carry_id"])].append(card)

    # The forward-ledger partition set is a signed declaration exactly like
    # the cell set: a carry cannot silently gain or lose its promised return-
    # contract data relative to what was signed.
    if len(params.expected_ledger_carries) != len(set(params.expected_ledger_carries)):
        raise ValueError(f"expected_ledger_carries declares duplicates: {params.expected_ledger_carries}")
    declared_ledger = set(params.expected_ledger_carries)
    supplied_ledger = {_carry_from_ledger_partition_id(pid) for pid in ledger_partitions}
    if declared_ledger != supplied_ledger:
        raise ValueError(
            f"forward ledger carries {sorted(supplied_ledger)} do not match "
            f"the declared set {sorted(declared_ledger)}"
        )
    ledger_contract_by_carry: dict[str, bytes] = {}
    for pid, data in ledger_partitions.items():
        carry_id = _carry_from_ledger_partition_id(pid)
        rows = _parse_ledger_rows(pid, data)
        ledger_contract_by_carry[carry_id] = build_forward_ledger_return_contract(rows)

    declared = params.expected_cells_by_carry
    results: list[CarryHeadResult] = []
    for carry_id in sorted(cards_by_carry):
        if declared:
            if carry_id not in declared:
                raise ValueError(f"undeclared carry {carry_id!r} produced cells")
            expected = list(declared[carry_id])
        else:
            expected = sorted(str(card["cell_id"]) for card in cards_by_carry[carry_id])
        head = evaluate_carry(campaign_id, carry_id, cards_by_carry[carry_id], expected, params)
        contract_bytes = ledger_contract_by_carry.get(carry_id)
        if contract_bytes:
            head = dataclasses.replace(head, strategy_return_bytes=contract_bytes)
        results.append(head)
    if declared:
        missing = sorted(set(declared) - set(cards_by_carry))
        if missing:
            raise ValueError(f"declared carries with no supplied cells: {missing}")
    return tuple(results)


__all__ = [
    "EvaluationParams", "evaluate_carry", "evaluate_partitions",
    "build_forward_ledger_return_contract",
]
