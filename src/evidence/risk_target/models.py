"""Plain in-process dataclasses that carry per-head risk-target results
through the evidence slice.

These are NOT signed contracts (those live in :mod:`src.evidence.contracts`).
They are the mutable working values the producer computes and the local
importer reproduces, before/after they are frozen into an ``EvaluationReport``
and an ``EvidencePackage``.
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

# One head == one Phase-H parallel unit ("target head x fold"). This slice is
# single-fold, so the heads ARE the folds: the drawdown head and the vol head
# are packaged, dispositioned, and gated fully independently of each other.
VOLATILITY_HEAD_ID = "forward_volatility"
DRAWDOWN_HEAD_ID = "drawdown_state"


@dataclass(frozen=True)
class HeadResult:
    """Deterministic evaluation of one risk-target head on a declared dataset.

    ``metrics`` values are plain floats; ``metric_tolerances`` declares the
    per-metric absolute reproduction tolerance the local importer replays
    against. ``gates`` is the per-head admissibility verdict; ``passed`` is
    ``True`` iff no gate failed (an honestly-failed head — e.g. the
    not-learnable drawdown head — carries ``passed=False`` and is preserved as
    a first-class negative result, never silently dropped).
    """

    head_id: str
    lane_id: str
    target: str
    metrics: Mapping[str, float]
    metric_tolerances: Mapping[str, float]
    gates: tuple[GateResult, ...]
    passed: bool
    model_bytes: bytes
    media_type: str
    trial_count: int
    effective_sample_size: float
    temporal_holdout: str
    purge_observations: int
    embargo_observations: int
    incumbent_comparison: Mapping[str, object] = field(default_factory=dict)
    regime_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PackagedHead:
    """One head, packaged and signed by the producer, ready to be imported."""

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
class WorkerOutput:
    """Everything one signed risk-target job produced on the worker side."""

    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    capability_profile_envelope: SignedEnvelope
    heads: tuple[PackagedHead, ...]


@dataclass(frozen=True)
class HeadImportOutcome:
    """The local authority's verdict on one imported head package."""

    lane_id: str
    package_digest: str
    verdict: LocalImportVerdict
    final_state: DispositionState
    reason: str
    disposition_head_digest: str
