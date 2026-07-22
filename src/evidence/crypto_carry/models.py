"""Working dataclasses for the crypto-carry evidence slice."""

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


def lane_id_for_carry(campaign_id: str, carry_id: str) -> str:
    if not campaign_id or not carry_id or "__" in campaign_id or "__" in carry_id:
        raise ValueError("campaign_id and carry_id must be non-empty and cannot contain '__'")
    return f"crypto_carry_{campaign_id}__{carry_id}"


@dataclass(frozen=True)
class CarryHeadResult:
    head_id: str
    lane_id: str
    campaign_id: str
    carry_id: str
    verdict: str
    verdict_basis: str
    metrics: Mapping[str, float]
    metric_tolerances: Mapping[str, float]
    gates: tuple[GateResult, ...]
    passed: bool
    artifact_bytes: bytes
    media_type: str
    n_cells: int
    cell_ids: tuple[str, ...]
    temporal_holdout: str
    incumbent_comparison: Mapping[str, object] = field(default_factory=dict)
    # Roadmap §14 standardized strategy-return/exposure contract, sourced from
    # this carry's REAL forward-shadow ledger (src.crypto.carry_shadow
    # record_shadow_cycle) when supplied — empty for carries with no forward
    # ledger partition yet (fully backward compatible). See
    # evaluation.py::build_forward_ledger_return_contract for the exact schema.
    strategy_return_bytes: bytes = b""


@dataclass(frozen=True)
class PackagedCarryHead:
    head_id: str
    lane_id: str
    campaign_id: str
    carry_id: str
    package: EvidencePackage
    package_digest: str
    package_envelope: SignedEnvelope
    files: Mapping[str, bytes]
    evaluation_report_envelope: SignedEnvelope
    created_event_envelope: SignedEnvelope
    result_passed: bool


@dataclass(frozen=True)
class CryptoCarryWorkerOutput:
    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    capability_profile_envelope: SignedEnvelope
    heads: tuple[PackagedCarryHead, ...]


@dataclass(frozen=True)
class CarryImportOutcome:
    lane_id: str
    campaign_id: str
    carry_id: str
    package_digest: str
    verdict: LocalImportVerdict
    final_state: DispositionState
    reason: str
    disposition_head_digest: str


__all__ = [
    "lane_id_for_carry",
    "CarryHeadResult",
    "PackagedCarryHead",
    "CryptoCarryWorkerOutput",
    "CarryImportOutcome",
]
