"""In-process values for the portfolio-book (combined-book allocator) evidence slice."""

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

LANE_ID = "portfolio_book"
HEAD_ID = "combined_book_oos"


@dataclass(frozen=True)
class BookResult:
    metrics: Mapping[str, float]
    gates: tuple[GateResult, ...]
    passed: bool
    artifact_bytes: bytes
    effective_sample_size: float
    temporal_holdout: str
    incumbent_comparison: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PackagedBook:
    package: EvidencePackage
    package_digest: str
    package_envelope: SignedEnvelope
    files: Mapping[str, bytes]
    evaluation_report_envelope: SignedEnvelope
    created_event_envelope: SignedEnvelope


@dataclass(frozen=True)
class WorkerOutput:
    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    capability_profile_envelope: SignedEnvelope
    book: PackagedBook


@dataclass(frozen=True)
class ImportOutcome:
    package_digest: str
    verdict: LocalImportVerdict
    final_state: DispositionState
    reason: str
    disposition_head_digest: str


__all__ = ["LANE_ID", "HEAD_ID", "BookResult", "PackagedBook", "WorkerOutput", "ImportOutcome"]
