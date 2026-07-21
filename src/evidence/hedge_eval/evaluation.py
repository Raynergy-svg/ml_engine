"""Authority-free hedge evaluation.

Given a set of already-verified per-strategy ledger partitions (each partition
is the JSONL bytes of one strategy's raw-vs-hedged shadow ledger), this module
scores each strategy with the existing pure hedge scorecard and returns
deterministic per-strategy results. It imports only the analysis-only hedge
scorecard, the evidence canonicalizer, and the evidence contracts — never the
broker path, the trade execution/gate path, or the state engine. It is handed
its data as bytes and has no repository-root or control-file parameter, so it
can neither read ``.claude/state.json`` nor reach a broker credential.

Determinism is structural, not incidental: ``compute_strategy_scorecard`` is
pure float arithmetic over the logged rows, so two runs on identical partition
bytes produce identical metrics and identical verdicts — the property the
local metric-replay verification depends on. Unlike a trained model there is no
device-dependent nondeterminism to guard against.

The admissibility gate encodes the honest hedge question directly: a strategy's
hedge candidate is admissible for quarantine only when hedging *demonstrably*
improves its after-cost expectancy AND cuts its drawdown on a cost-known basis
(verdict ``signal_real_but_noisy``). A return that dies once its market/sector
beta is hedged out (``return_was_beta_not_alpha``), a hedge that only cuts vol
on a book with no positive expectancy to protect (``weak_or_dead``), a venue
whose hedge cost cannot be modelled (``cost_unmodeled:*``), or a history too
thin to conclude anything all fail the gate and are REJECTED — preserved as
first-class honest negatives, never dropped or rescued.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import GateResult, GateStatus
from src.hedge import hedge_scorecard as hsc

from .models import HedgeHeadResult, lane_id_for_strategy


@dataclass(frozen=True)
class EvaluationParams:
    """Frozen hedge-evaluation configuration.

    ``min_history`` mirrors the scorecard's own ``MIN_HISTORY_FOR_VERDICT`` bar
    (no verdict is forced on a thinner sample). ``replay_tolerance`` is the
    absolute per-metric bound the local importer replays against; the scorecard
    is exact float arithmetic re-parsed from identical bytes, so the delta is
    ~0 and the tolerance is only a guard against float re-parse noise.
    """

    min_history: int = hsc.MIN_HISTORY_FOR_VERDICT
    replay_tolerance: float = 1e-12


# The numeric return leaves the scorecard reads. json.loads accepts NaN/Infinity
# by default, and a non-finite value would later blow up Canonical-JSON encoding
# with an uncaught error on the PRODUCER path; reject it at parse time so both
# the producer and the importer's replay fail closed symmetrically.
_RETURN_LEAVES = (("raw", "net_return"), ("raw", "gross_return"),
                  ("hedged", "net_return"), ("hedged", "gross_return"))


def _reject_non_finite_returns(strategy: str, lineno: int, row: dict[str, Any]) -> None:
    for section, key in _RETURN_LEAVES:
        node = row.get(section)
        value = node.get(key) if isinstance(node, dict) else None
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"partition {strategy!r} line {lineno} has non-finite {section}.{key}"
            )


def _parse_partition(strategy: str, data: bytes) -> list[dict[str, Any]]:
    """Parse one strategy's JSONL ledger partition into row dicts.

    Every non-blank line must be a JSON object carrying this strategy's own
    rows — a foreign or malformed row is a hard evaluation failure (fail-closed),
    not a silently-skipped line, because the partition bytes were already
    hash-verified against the signed dataset manifest before we got here.
    """
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(data.decode("utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"partition {strategy!r} line {lineno} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"partition {strategy!r} line {lineno} is not a JSON object")
        row_strategy = row.get("strategy")
        if row_strategy != strategy:
            raise ValueError(
                f"partition {strategy!r} line {lineno} carries strategy {row_strategy!r}"
            )
        _reject_non_finite_returns(strategy, lineno, row)
        rows.append(row)
    if not rows:
        raise ValueError(f"partition {strategy!r} contains no ledger rows")
    return rows


def _gate(gate_id: str, ok: bool, observed: Any, threshold: Any, reason: str) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        observed=observed,
        threshold=threshold,
        reason=reason,
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _collect_metrics(card: Mapping[str, Any]) -> dict[str, float]:
    """Flatten the scorecard's numeric readings into a flat float metric map.

    Only resolved, finite figures are included: a ``None`` expectancy or an
    unfabricated (below-min-history) Sharpe is simply absent, never coerced to a
    number. n-of-cycles counters are always present so the report is never empty.
    """
    metrics: dict[str, float] = {
        "n_cycles": float(card["n_cycles"]),
        "resolved_raw_cycles": float(card["raw"]["n"]),
    }
    lane_metric_keys = (
        "after_cost_expectancy", "hit_rate", "max_drawdown",
        "sharpe_annualized", "sortino_annualized",
    )
    for lane_name, lane in (("raw", card["raw"]), ("hedged_net", card["hedged_net"])):
        for key in lane_metric_keys:
            value = lane.get(key)
            if _finite(value):
                metrics[f"{lane_name}_{key}"] = float(value)
    drift = card.get("exposure_drift", {})
    for key in ("net_beta_mean", "net_beta_std", "avg_concentration_warnings"):
        value = drift.get(key)
        if _finite(value):
            metrics[f"exposure_{key}"] = float(value)
    return metrics


def _holdout(card: Mapping[str, Any]) -> str:
    asof = card.get("asof_range") or [None, None]
    start, end = (asof + [None, None])[:2]
    if start and end:
        return f"logged raw-vs-hedged ledger cycles {start}..{end}"
    return "full logged raw-vs-hedged ledger history"


def evaluate_strategy(strategy: str, rows: list[dict[str, Any]], params: EvaluationParams) -> HedgeHeadResult:
    """Score one strategy's ledger into a gated, deterministic head result."""
    card = hsc.compute_strategy_scorecard(rows)
    decision = card["decision"]
    verdict = decision["verdict"]
    basis = decision["basis"]

    resolved_n = int(card["raw"]["n"])
    hedge_applied = not card["hedge_unavailable"]
    cost_unmodeled = bool(card["cost_unmodeled_for_venue"])
    # Admissible iff hedging demonstrably helps after-cost on a cost-KNOWN basis.
    # cost_unmodeled venues report a "cost_unmodeled:..." verdict on a "gross"
    # basis and can never clear the after-cost bar (honest by construction).
    hedge_admissible = (
        verdict == hsc.VERDICT_SIGNAL_REAL_BUT_NOISY and basis == "net" and not cost_unmodeled
    )

    gates = (
        _gate(
            "sufficient_history", resolved_n >= params.min_history, resolved_n, params.min_history,
            "at least min-history resolved cycles are required to conclude anything",
        ),
        _gate(
            "hedge_actually_applied", hedge_applied, hedge_applied, True,
            "at least one cycle must have an applied hedge (raw==hedged otherwise)",
        ),
        _gate(
            "hedge_improves_after_cost", hedge_admissible, verdict, hsc.VERDICT_SIGNAL_REAL_BUT_NOISY,
            "hedge must improve after-cost expectancy and cut drawdown on a cost-known basis",
        ),
    )
    passed = all(g.status == GateStatus.PASS for g in gates)

    metrics = _collect_metrics(card)
    tolerances = {name: params.replay_tolerance for name in metrics}
    artifact_bytes = canonical_bytes(card)

    return HedgeHeadResult(
        head_id=strategy,
        lane_id=lane_id_for_strategy(strategy),
        strategy=strategy,
        verdict=verdict,
        verdict_basis=basis,
        metrics=metrics,
        metric_tolerances=tolerances,
        gates=gates,
        passed=passed,
        artifact_bytes=artifact_bytes,
        media_type="application/json",
        n_cycles=int(card["n_cycles"]),
        resolved_raw_cycles=resolved_n,
        temporal_holdout=_holdout(card),
        cost_unmodeled_for_venue=cost_unmodeled,
        incumbent_comparison={"has_incumbent": False},
    )


def evaluate_partitions(
    partitions: Mapping[str, bytes],
    params: EvaluationParams | None = None,
) -> tuple[HedgeHeadResult, ...]:
    """Score every declared strategy partition (deterministic, one head each)."""
    if not partitions:
        raise ValueError("no ledger partitions supplied to hedge evaluator")
    params = params or EvaluationParams()
    return tuple(
        evaluate_strategy(strategy, _parse_partition(strategy, partitions[strategy]), params)
        for strategy in sorted(partitions)
    )


__all__ = [
    "EvaluationParams",
    "evaluate_strategy",
    "evaluate_partitions",
]
