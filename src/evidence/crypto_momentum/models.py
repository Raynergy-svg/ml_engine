"""Plain in-process dataclasses that carry per-construction crypto-momentum
results through the evidence slice.

These are NOT signed contracts (those live in :mod:`src.evidence.contracts`).
They are the mutable working values the producer computes and the local importer
reproduces, before/after they are frozen into an ``EvaluationReport`` and an
``EvidencePackage``.

The crypto-momentum slice's unit of evaluation is one *construction* of one
research campaign — a frozen cross-sectional-momentum configuration identity. A
construction's verdict is produced by a DETERMINISTIC AGGREGATOR that pools many
per-cell forward-return scorecards (the Phase-H parallel unit is
``construction × fold × stress``) and refuses an incomplete or duplicated cell
set. Each construction is scored, gated, packaged, dispositioned and
quarantined/rejected fully independently. The lane stays SHADOW: the frozen
construction's significance never cleared its pre-registration, so an honest
forward scorecard REJECTS it as a first-class null rather than admitting it — the
honest negative is preserved, never rescued (roadmap J3).
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


def lane_id_for_construction(campaign_id: str, construction: str) -> str:
    """The evidence lane id under which one campaign-construction verdict lives."""
    if not campaign_id or not construction or "__" in campaign_id or "__" in construction:
        raise ValueError(
            "campaign_id and construction must be non-empty and cannot contain '__'"
        )
    return f"crypto_momentum_{campaign_id}__{construction}"


@dataclass(frozen=True)
class ConstructionHeadResult:
    """Deterministic aggregated evaluation of one construction over its cells.

    ``metrics`` values are plain floats — only resolved, finite figures are
    included (an undefined pooled Sharpe or an uncomputable t-statistic is simply
    absent, never a fabricated number). ``metric_tolerances`` declares the
    per-metric absolute reproduction tolerance the local importer replays
    against. ``gates`` is the per-construction admissibility verdict; ``passed``
    is ``True`` iff no gate failed. A construction whose out-of-sample Sharpe is
    positive but not significant (``shadow_insufficient_significance``), that
    breaks under stress, that shows no edge, or that has too few out-of-sample
    folds carries ``passed=False`` and is preserved as a first-class honest
    negative — the shadow-only null the roadmap requires — never silently dropped
    or rescued by a strong in-sample number.

    ``artifact_bytes`` are the Canonical-JSON aggregated construction scorecard
    the producer computed — plain data (no pickle, no executable payload),
    content-addressed and signed in the package; the slice never re-executes it.
    """

    head_id: str
    lane_id: str
    campaign_id: str
    construction: str
    verdict: str
    verdict_basis: str
    metrics: Mapping[str, float]
    metric_tolerances: Mapping[str, float]
    gates: tuple[GateResult, ...]
    passed: bool
    artifact_bytes: bytes
    media_type: str
    n_folds: int
    n_oos_folds: int
    fold_ids: tuple[str, ...]
    temporal_holdout: str
    incumbent_comparison: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PackagedConstructionHead:
    """One construction, packaged and signed by the producer, ready to import."""

    head_id: str
    lane_id: str
    campaign_id: str
    construction: str
    package: EvidencePackage
    package_digest: str
    package_envelope: SignedEnvelope
    files: Mapping[str, bytes]
    evaluation_report_envelope: SignedEnvelope
    created_event_envelope: SignedEnvelope
    result_passed: bool


@dataclass(frozen=True)
class CryptoMomentumWorkerOutput:
    """Everything one signed crypto-momentum job produced on the worker side."""

    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    capability_profile_envelope: SignedEnvelope
    heads: tuple[PackagedConstructionHead, ...]


@dataclass(frozen=True)
class ConstructionImportOutcome:
    """The local authority's verdict on one imported construction package."""

    lane_id: str
    campaign_id: str
    construction: str
    package_digest: str
    verdict: LocalImportVerdict
    final_state: DispositionState
    reason: str
    disposition_head_digest: str


__all__ = [
    "lane_id_for_construction",
    "ConstructionHeadResult",
    "PackagedConstructionHead",
    "CryptoMomentumWorkerOutput",
    "ConstructionImportOutcome",
]
