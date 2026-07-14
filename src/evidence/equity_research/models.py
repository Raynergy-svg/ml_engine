"""Plain in-process dataclasses that carry per-universe-tier equity-research
results through the evidence slice.

These are NOT signed contracts (those live in :mod:`src.evidence.contracts`).
They are the mutable working values the producer computes and the local importer
reproduces, before/after they are frozen into an ``EvaluationReport`` and an
``EvidencePackage``.

The equity-research slice's unit of evaluation is one *universe tier* of one
research hypothesis — ``curated``, ``wide`` (survivorship-corrected) or
``full_history``. A tier's verdict is produced by a DETERMINISTIC AGGREGATOR that
pools many per-fold cross-sectional rank-IC scorecards (the Phase-H parallel unit
is ``hypothesis × universe-snapshot × fold``) and refuses an incomplete or
duplicated fold set. Each tier is scored, gated, packaged, dispositioned and
quarantined/rejected fully independently of every other tier — so the attractive
curated-universe result and the defensible wide-universe result are preserved
side by side as first-class, materially-different honest results (roadmap J2).
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


def lane_id_for_tier(hypothesis_id: str, tier: str) -> str:
    """The evidence lane id under which one hypothesis-tier verdict lives."""
    return f"equity_research_{hypothesis_id}_{tier}"


@dataclass(frozen=True)
class TierHeadResult:
    """Deterministic aggregated evaluation of one universe tier over its folds.

    ``metrics`` values are plain floats — only resolved, finite figures are
    included (an undefined pooled rank-IC or an uncomputable t-statistic is
    simply absent, never a fabricated number). ``metric_tolerances`` declares the
    per-metric absolute reproduction tolerance the local importer replays
    against. ``gates`` is the per-tier admissibility verdict; ``passed`` is
    ``True`` iff no gate failed. A tier whose out-of-sample rank-IC fades to
    statistical insignificance as the universe widens
    (``signal_fades_with_breadth``), or that has too few out-of-sample folds,
    carries ``passed=False`` and is preserved as a first-class honest negative,
    never silently dropped or rescued by a strong in-sample/curated number.

    ``artifact_bytes`` are the Canonical-JSON aggregated tier scorecard the
    producer computed — plain data (no pickle, no executable payload),
    content-addressed and signed in the package; the slice never re-executes it.
    """

    head_id: str
    lane_id: str
    hypothesis_id: str
    tier: str
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
class PackagedTierHead:
    """One universe tier, packaged and signed by the producer, ready to import."""

    head_id: str
    lane_id: str
    hypothesis_id: str
    tier: str
    package: EvidencePackage
    package_digest: str
    package_envelope: SignedEnvelope
    files: Mapping[str, bytes]
    evaluation_report_envelope: SignedEnvelope
    created_event_envelope: SignedEnvelope
    result_passed: bool


@dataclass(frozen=True)
class EquityResearchWorkerOutput:
    """Everything one signed equity-research job produced on the worker side."""

    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    capability_profile_envelope: SignedEnvelope
    heads: tuple[PackagedTierHead, ...]


@dataclass(frozen=True)
class TierImportOutcome:
    """The local authority's verdict on one imported universe-tier package."""

    lane_id: str
    hypothesis_id: str
    tier: str
    package_digest: str
    verdict: LocalImportVerdict
    final_state: DispositionState
    reason: str
    disposition_head_digest: str


__all__ = [
    "lane_id_for_tier",
    "TierHeadResult",
    "PackagedTierHead",
    "EquityResearchWorkerOutput",
    "TierImportOutcome",
]
