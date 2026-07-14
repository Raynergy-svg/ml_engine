"""Authority-free equity-research evaluation.

Given a set of already-verified per-fold partitions (each partition is the JSONL
bytes of one fold's per-name research cross-section), this module:

1. parses each fold partition,
2. scores each fold into a cross-sectional rank-IC scorecard,
3. groups folds by universe tier and runs the DETERMINISTIC AGGREGATOR — which
   refuses an incomplete or duplicated fold set — to produce one gated tier
   result per tier.

It imports only the analysis-only rank-IC scorecard, the evidence canonicalizer,
and the evidence contracts — never the broker path, the trade execution/gate
path, or the state engine. It is handed its data as bytes and has no
repository-root or control-file parameter, so it can neither read
``.claude/state.json`` nor reach a broker credential.

Determinism is structural, not incidental: the scorecard is pure float
arithmetic over the parsed rows, so two runs on identical partition bytes produce
identical metrics and identical verdicts — the property the local metric-replay
verification depends on.

The admissibility gate encodes the honest research question directly: a tier's
research candidate is admissible for quarantine only when its OUT-OF-SAMPLE
pooled rank-IC clears the pre-registered bar with an across-fold t-statistic that
distinguishes it from zero. A tier whose out-of-sample edge fades to
insignificance as the universe widens (``signal_fades_with_breadth``), shows no
signal, or has too few out-of-sample folds all fail the gate and are preserved as
first-class honest negatives — never dropped, never rescued by a strong
in-sample or curated number.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.equity.research import rank_ic_scorecard as ric
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import GateResult, GateStatus

from .models import TierHeadResult, lane_id_for_tier


@dataclass(frozen=True)
class EvaluationParams:
    """Frozen equity-research evaluation configuration.

    ``expected_folds_by_tier`` is the authoritative declared fold set the
    aggregator refuses incomplete/duplicate aggregations against (empty ⇒ derive
    the expectation from the partitions actually supplied, used only in unit
    tests). ``min_oos_folds`` / ``ic_bar`` / ``ic_tstat_bar`` mirror the
    scorecard's own bars. ``replay_tolerance`` is the absolute per-metric bound
    the local importer replays against; the scorecard is exact float arithmetic
    re-parsed from identical bytes, so the delta is ~0.
    """

    expected_folds_by_tier: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    min_oos_folds: int = ric.MIN_OOS_FOLDS_FOR_VERDICT
    ic_bar: float = ric.IC_BAR
    ic_tstat_bar: float = ric.IC_TSTAT_BAR
    replay_tolerance: float = 1e-12


def _parse_fold_partition(fold_id: str, data: bytes) -> list[dict[str, Any]]:
    """Parse one fold's JSONL cross-section into row dicts (fail-closed).

    Every non-blank line must be a JSON object; a malformed line is a hard
    evaluation failure, not a silently-skipped row, because the partition bytes
    were hash-verified against the signed dataset manifest before we got here.
    """
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(data.decode("utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"fold {fold_id!r} line {lineno} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"fold {fold_id!r} line {lineno} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"fold {fold_id!r} contains no cross-section rows")
    return rows


def _tier_of_partition(fold_id: str) -> str:
    """Extract the tier from a ``{tier}::fold{i}`` partition id."""
    tier = fold_id.split("::", 1)[0]
    if not tier or tier == fold_id:
        raise ValueError(f"fold partition id {fold_id!r} is not of the form '<tier>::fold<i>'")
    return tier


def _gate(gate_id: str, ok: bool, observed: Any, threshold: Any, reason: str) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        observed=observed,
        threshold=threshold,
        reason=reason,
    )


def _collect_metrics(agg: Mapping[str, Any]) -> dict[str, float]:
    """Flatten the aggregated tier scorecard into a flat float metric map.

    Only resolved, finite figures are included: an undefined pooled rank-IC or an
    uncomputable t-statistic is simply absent, never coerced to a number. The
    fold-count metrics are always present so the report is never empty.
    """
    metrics: dict[str, float] = {
        "n_folds": float(agg["n_folds"]),
        "n_oos_folds": float(agg["n_oos_folds"]),
        "n_is_folds": float(agg["n_is_folds"]),
    }
    optional_keys = (
        "breadth_mean_names", "pooled_oos_rank_ic", "pooled_is_rank_ic",
        "oos_ic_tstat", "mean_oos_q5_q1_spread",
    )
    for key in optional_keys:
        value = agg.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fv = float(value)
            # Guard against a (theoretical) infinite t-statistic reaching the
            # Canonical-JSON encoder — an infinite reading is not a metric.
            if fv == fv and fv not in (float("inf"), float("-inf")):
                metrics[key] = fv
    return metrics


def _holdout(agg: Mapping[str, Any], tier: str) -> str:
    return f"tier {tier}: {agg['n_oos_folds']} out-of-sample fold(s) of {agg['n_folds']} total"


def evaluate_tier(
    hypothesis_id: str,
    tier: str,
    fold_cards: list[dict[str, Any]],
    expected_fold_ids: list[str],
    params: EvaluationParams,
) -> TierHeadResult:
    """Aggregate one tier's fold scorecards into a gated, deterministic head."""
    agg = ric.aggregate_tier_scorecard(
        tier, fold_cards, expected_fold_ids,
        min_oos_folds=params.min_oos_folds, ic_bar=params.ic_bar, ic_tstat_bar=params.ic_tstat_bar,
    )
    decision = agg["decision"]
    verdict = decision["verdict"]
    basis = decision["basis"]

    n_oos = int(agg["n_oos_folds"])
    persists = verdict == ric.VERDICT_SIGNAL_PERSISTS

    gates = (
        _gate(
            "sufficient_oos_folds", n_oos >= params.min_oos_folds, n_oos, params.min_oos_folds,
            "at least min out-of-sample folds with a defined rank-IC are required to conclude",
        ),
        _gate(
            "signal_persists_oos", persists, verdict, ric.VERDICT_SIGNAL_PERSISTS,
            "pooled OOS rank-IC must clear ic_bar with an across-fold t-stat clearing ic_tstat_bar",
        ),
    )
    passed = all(g.status == GateStatus.PASS for g in gates)

    metrics = _collect_metrics(agg)
    tolerances = {name: params.replay_tolerance for name in metrics}
    artifact_bytes = canonical_bytes(agg)
    sorted_folds = tuple(sorted(expected_fold_ids))

    return TierHeadResult(
        head_id=lane_id_for_tier(hypothesis_id, tier),
        lane_id=lane_id_for_tier(hypothesis_id, tier),
        hypothesis_id=hypothesis_id,
        tier=tier,
        verdict=verdict,
        verdict_basis=basis,
        metrics=metrics,
        metric_tolerances=tolerances,
        gates=gates,
        passed=passed,
        artifact_bytes=artifact_bytes,
        media_type="application/json",
        n_folds=int(agg["n_folds"]),
        n_oos_folds=n_oos,
        fold_ids=sorted_folds,
        temporal_holdout=_holdout(agg, tier),
        incumbent_comparison={"has_incumbent": False},
    )


def evaluate_partitions(
    partitions: Mapping[str, bytes],
    params: EvaluationParams | None = None,
    *,
    hypothesis_id: str,
) -> tuple[TierHeadResult, ...]:
    """Score every fold partition and aggregate one head per universe tier.

    Deterministic: folds are grouped by tier (from their ``{tier}::fold{i}``
    partition id, cross-checked against each row's declared ``tier``), scored
    independently, then aggregated per tier against the declared expected fold
    set — refusing any tier whose produced folds do not match its declaration.
    """
    if not partitions:
        raise ValueError("no fold partitions supplied to equity-research evaluator")
    params = params or EvaluationParams()

    cards_by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold_id in sorted(partitions):
        partition_tier = _tier_of_partition(fold_id)
        rows = _parse_fold_partition(fold_id, partitions[fold_id])
        card = ric.compute_fold_scorecard(fold_id, rows)
        if card["tier"] != partition_tier:
            raise ValueError(
                f"fold {fold_id!r} rows declare tier {card['tier']!r} "
                f"but the partition id names tier {partition_tier!r}"
            )
        cards_by_tier[partition_tier].append(card)

    # Expected fold set per tier: the signed declaration when supplied, otherwise
    # derived from the partitions present (unit-test convenience only).
    declared = params.expected_folds_by_tier
    results: list[TierHeadResult] = []
    for tier in sorted(cards_by_tier):
        if declared:
            if tier not in declared:
                raise ValueError(f"tier {tier!r} produced folds but is not in the declared expectation")
            expected = list(declared[tier])
        else:
            expected = sorted(card["fold_id"] for card in cards_by_tier[tier])
        results.append(evaluate_tier(hypothesis_id, tier, cards_by_tier[tier], expected, params))

    # If the declaration names a tier for which no folds were supplied at all,
    # that is an incomplete aggregation — refuse rather than silently omit a head.
    if declared:
        missing_tiers = sorted(set(declared) - set(cards_by_tier))
        if missing_tiers:
            raise ValueError(f"declared tiers with no supplied folds: {missing_tiers}")

    return tuple(results)


__all__ = [
    "EvaluationParams",
    "evaluate_tier",
    "evaluate_partitions",
]
