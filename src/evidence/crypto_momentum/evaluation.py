"""Authority-free deterministic evaluation for crypto-momentum cells.

Each partition is one ``construction × fold × stress`` cell.  The scorer works
on already-realized, cost-and-funding-aware returns; the aggregator refuses any
cell set that differs from the signed declaration.  A complete but statistically
insignificant construction is preserved as an honest negative.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.crypto import momentum_scorecard as scorecard
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import GateResult, GateStatus

from .models import ConstructionHeadResult, lane_id_for_construction


@dataclass(frozen=True)
class EvaluationParams:
    """Signed bars and expected cells used by producer and independent replay."""

    expected_cells_by_construction: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    min_oos_folds: int = scorecard.MIN_OOS_FOLDS_FOR_VERDICT
    sharpe_bar: float = scorecard.SHARPE_BAR
    sharpe_tstat_bar: float = scorecard.SHARPE_TSTAT_BAR
    max_drawdown_limit: float = scorecard.MAX_DRAWDOWN_LIMIT
    stress_sharpe_floor: float = scorecard.STRESS_SHARPE_FLOOR
    drop_one_sharpe_floor: float = scorecard.DROP_ONE_SHARPE_FLOOR
    replay_tolerance: float = 1e-12


def _parse_cell(cell_id: str, data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"cell {cell_id!r} is not valid UTF-8: {exc}") from exc
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cell {cell_id!r} line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"cell {cell_id!r} line {line_number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"cell {cell_id!r} contains no rows")
    return rows


def _construction_from_cell_id(cell_id: str) -> str:
    construction = cell_id.split("::", 1)[0]
    if not construction or construction == cell_id:
        raise ValueError(
            f"cell partition id {cell_id!r} is not '<construction>::<stress>::fold<i>'"
        )
    return construction


def _gate(gate_id: str, ok: bool, observed: Any, threshold: Any, reason: str) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        observed=observed,
        threshold=threshold,
        reason=reason,
    )


def _metrics(aggregate: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {
        "n_folds": float(aggregate["n_folds"]),
        "n_baseline_oos_folds": float(aggregate["n_baseline_oos_folds"]),
    }
    for key in (
        "pooled_oos_sharpe",
        "pooled_is_sharpe",
        "oos_sharpe_tstat",
        "pooled_oos_max_drawdown",
        "drop_one_oos_sharpe",
    ):
        value = aggregate.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number == number and number not in (float("inf"), float("-inf")):
                output[key] = number
    for stress, value in sorted((aggregate.get("stress_oos_sharpe") or {}).items()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number == number and number not in (float("inf"), float("-inf")):
                output[f"stress_oos_sharpe.{stress}"] = number
    return output


def evaluate_construction(
    campaign_id: str,
    construction: str,
    cards: list[dict[str, Any]],
    expected_cell_ids: list[str],
    params: EvaluationParams,
) -> ConstructionHeadResult:
    aggregate = scorecard.aggregate_construction_scorecard(
        construction,
        cards,
        expected_cell_ids,
        min_oos_folds=params.min_oos_folds,
        sharpe_bar=params.sharpe_bar,
        sharpe_tstat_bar=params.sharpe_tstat_bar,
        max_drawdown_limit=params.max_drawdown_limit,
        stress_sharpe_floor=params.stress_sharpe_floor,
        drop_one_sharpe_floor=params.drop_one_sharpe_floor,
    )
    verdict = aggregate["decision"]["verdict"]
    admissible = verdict == scorecard.VERDICT_SIGNIFICANCE_CLEARS
    reason = aggregate["decision"].get("reason")
    basis = reason or (
        "all frozen momentum gates cleared"
        if admissible
        else "scorecard supplied no reason for its non-passing verdict"
    )
    n_oos = int(aggregate["n_baseline_oos_folds"])
    stress_values = aggregate.get("stress_oos_sharpe") or {}
    stresses_ok = bool(stress_values) and all(
        value is not None and float(value) >= params.stress_sharpe_floor
        for value in stress_values.values()
    )
    gates = (
        _gate(
            "sufficient_forward_folds",
            n_oos >= params.min_oos_folds,
            n_oos,
            params.min_oos_folds,
            "adequate baseline out-of-sample folds are mandatory",
        ),
        _gate(
            "significance_clears",
            admissible,
            verdict,
            scorecard.VERDICT_SIGNIFICANCE_CLEARS,
            "positive Sharpe alone is insufficient; significance, drawdown and drop-one bars all apply",
        ),
        _gate(
            "all_declared_stresses_survive",
            stresses_ok,
            dict(stress_values),
            params.stress_sharpe_floor,
            "every declared adverse funding/basis/outage stress must remain above the floor",
        ),
    )
    # Quarantine is not promotion. A future complete scorecard may clear these
    # numerical gates and enter quarantine, but this slice has no promotion or
    # champion-pointer capability. The currently frozen construction remains a
    # signed negative because its real significance result does not clear.
    passed = all(g.status == GateStatus.PASS for g in gates)
    metrics = _metrics(aggregate)
    lane_id = lane_id_for_construction(campaign_id, construction)
    return ConstructionHeadResult(
        head_id=lane_id,
        lane_id=lane_id,
        campaign_id=campaign_id,
        construction=construction,
        verdict=verdict,
        verdict_basis=basis,
        metrics=metrics,
        metric_tolerances={name: params.replay_tolerance for name in metrics},
        gates=gates,
        passed=passed,
        artifact_bytes=canonical_bytes(aggregate),
        media_type="application/json",
        n_folds=int(aggregate["n_folds"]),
        n_oos_folds=n_oos,
        fold_ids=tuple(sorted(expected_cell_ids)),
        temporal_holdout=f"{n_oos} baseline forward fold(s); stresses={sorted(stress_values)}",
        incumbent_comparison={"has_incumbent": False, "lane_mode": "shadow"},
    )


def evaluate_partitions(
    partitions: Mapping[str, bytes],
    params: EvaluationParams | None = None,
    *,
    campaign_id: str,
) -> tuple[ConstructionHeadResult, ...]:
    if not partitions:
        raise ValueError("no crypto-momentum cell partitions supplied")
    params = params or EvaluationParams()
    cards_by_construction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell_id in sorted(partitions):
        construction = _construction_from_cell_id(cell_id)
        card = scorecard.compute_fold_scorecard(cell_id, _parse_cell(cell_id, partitions[cell_id]))
        if card["construction"] != construction:
            raise ValueError(
                f"cell {cell_id!r} declares construction {card['construction']!r}, not {construction!r}"
            )
        cards_by_construction[construction].append(card)

    declared = params.expected_cells_by_construction
    results: list[ConstructionHeadResult] = []
    for construction in sorted(cards_by_construction):
        if declared:
            if construction not in declared:
                raise ValueError(f"undeclared construction {construction!r} produced cells")
            expected = list(declared[construction])
        else:
            expected = sorted(card["fold_id"] for card in cards_by_construction[construction])
        results.append(
            evaluate_construction(
                campaign_id, construction, cards_by_construction[construction], expected, params
            )
        )
    if declared:
        missing = sorted(set(declared) - set(cards_by_construction))
        if missing:
            raise ValueError(f"declared constructions with no supplied cells: {missing}")
    return tuple(results)


__all__ = ["EvaluationParams", "evaluate_construction", "evaluate_partitions"]
