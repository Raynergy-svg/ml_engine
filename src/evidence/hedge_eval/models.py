"""Plain in-process dataclasses that carry per-strategy hedge-evaluation
results through the evidence slice.

These are NOT signed contracts (those live in :mod:`src.evidence.contracts`).
They are the mutable working values the producer computes and the local
importer reproduces, before/after they are frozen into an ``EvaluationReport``
and an ``EvidencePackage``.

The hedge slice's unit of evaluation is one *strategy*: a strategy's
raw-vs-hedged ledger is scored into an alpha-vs-beta verdict, gated on its own
after-cost admissibility, packaged, dispositioned, and quarantined/rejected
fully independently of every other strategy — exactly the per-head isolation
the risk-target slice gives its volatility and drawdown heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.evidence.contracts import (
    DispositionState,
    EvidencePackage,
    GateResult,
    LocalImportVerdict,
    SignedEnvelope,
)


def lane_id_for_strategy(strategy: str) -> str:
    """The evidence lane id under which a strategy's hedge verdict lives."""
    return f"hedge_eval_{strategy}"


STRATEGY_RETURN_MEDIA_TYPE = "application/x-ndjson"


@dataclass(frozen=True)
class HedgeHeadResult:
    """Deterministic hedge-evaluation of one strategy on a declared ledger.

    ``metrics`` values are plain floats (only resolved, finite figures are
    included — a ``None`` Sharpe/expectancy is simply absent, never a fabricated
    number); ``metric_tolerances`` declares the per-metric absolute reproduction
    tolerance the local importer replays against. ``gates`` is the per-strategy
    admissibility verdict; ``passed`` is ``True`` iff no gate failed. A strategy
    whose return dies once its market/sector beta is hedged out
    (``return_was_beta_not_alpha``), or that has too thin a history, carries
    ``passed=False`` and is preserved as a first-class honest negative result,
    never silently dropped.

    ``artifact_bytes`` are the Canonical-JSON scorecard the producer computed —
    plain data (no pickle, no executable payload), content-addressed and signed
    in the package; the slice never re-executes it.

    ``strategy_return_bytes`` is the standardized strategy-return/exposure
    contract (roadmap §14) — one JSONL row per resolved trading day:
    ``{"date", "net_return", "gross_exposure", "turnover"}``. ``net_return`` is
    ``row["raw"]["net_return"]``; ``gross_exposure`` is the ledger's
    ``raw.gross_leverage`` (a gross-leverage proxy, the closest concept this
    ledger already logs); ``turnover`` is always JSON ``null`` — per-position
    weight history isn't logged by the shadow ledger today, so it is honestly
    reported as not-yet-modeled rather than fabricated from a leverage delta
    (a leverage-level change is not the same thing as position churn). Rows
    are deduplicated by ``asof_date`` (last-occurrence-wins, since the shadow
    ledger nominally logs one cycle per day but a backfill/re-run could repeat
    a date) and sorted ascending.
    """

    head_id: str
    lane_id: str
    strategy: str
    verdict: str
    verdict_basis: str
    metrics: Mapping[str, float]
    metric_tolerances: Mapping[str, float]
    gates: tuple[GateResult, ...]
    passed: bool
    artifact_bytes: bytes
    media_type: str
    strategy_return_bytes: bytes
    n_cycles: int
    resolved_raw_cycles: int
    temporal_holdout: str
    cost_unmodeled_for_venue: bool = False
    incumbent_comparison: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PackagedHedgeHead:
    """One strategy head, packaged and signed by the producer, ready to import."""

    head_id: str
    lane_id: str
    package: EvidencePackage
    package_digest: str
    package_envelope: SignedEnvelope
    files: Mapping[str, bytes]
    evaluation_report_envelope: SignedEnvelope
    created_event_envelope: SignedEnvelope
    result_passed: bool


@dataclass(frozen=True)
class HedgeWorkerOutput:
    """Everything one signed hedge-evaluation job produced on the worker side."""

    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    capability_profile_envelope: SignedEnvelope
    heads: tuple[PackagedHedgeHead, ...]


@dataclass(frozen=True)
class HedgeImportOutcome:
    """The local authority's verdict on one imported strategy head package."""

    lane_id: str
    strategy: str
    package_digest: str
    verdict: LocalImportVerdict
    final_state: DispositionState
    reason: str
    disposition_head_digest: str


__all__ = [
    "lane_id_for_strategy",
    "HedgeHeadResult",
    "PackagedHedgeHead",
    "HedgeWorkerOutput",
    "HedgeImportOutcome",
]
