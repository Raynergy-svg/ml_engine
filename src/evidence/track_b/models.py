"""Working dataclasses for Track B scoring and evaluation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.evidence.contracts import (
    DispositionState, EvidencePackage, GateResult, LocalImportVerdict, SignedEnvelope,
)


def lane_id(stage: str, campaign_id: str, unit_id: str) -> str:
    if stage not in {"scoring", "evaluation"}:
        raise ValueError("stage must be scoring or evaluation")
    if any(not value or "__" in value for value in (campaign_id, unit_id)):
        raise ValueError("campaign_id and unit_id must be non-empty and cannot contain '__'")
    return f"track_b_{stage}_{campaign_id}__{unit_id}"


@dataclass(frozen=True)
class TrackBHeadResult:
    stage: str
    campaign_id: str
    unit_id: str
    lane_id: str
    metrics: Mapping[str, float]
    passed: bool
    gates: tuple[GateResult, ...]
    artifact_bytes: bytes
    extra_artifacts: Mapping[str, bytes]
    temporal_holdout: str
    derived_from_package_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expected = lane_id(self.stage, self.campaign_id, self.unit_id)
        if self.lane_id != expected:
            raise ValueError(f"head lane_id {self.lane_id!r} does not match {expected!r}")


@dataclass(frozen=True)
class PackagedTrackBHead:
    stage: str
    campaign_id: str
    unit_id: str
    lane_id: str
    package: EvidencePackage
    package_digest: str
    package_envelope: SignedEnvelope
    files: Mapping[str, bytes]
    evaluation_report_envelope: SignedEnvelope
    created_event_envelope: SignedEnvelope
    result_passed: bool

    def __post_init__(self) -> None:
        expected = lane_id(self.stage, self.campaign_id, self.unit_id)
        if self.lane_id != expected:
            raise ValueError(f"packaged lane_id {self.lane_id!r} does not match {expected!r}")


@dataclass(frozen=True)
class TrackBWorkerOutput:
    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    capability_profile_envelope: SignedEnvelope
    heads: tuple[PackagedTrackBHead, ...]


@dataclass(frozen=True)
class TrackBImportOutcome:
    lane_id: str
    package_digest: str
    verdict: LocalImportVerdict
    final_state: DispositionState
    reason: str
    disposition_head_digest: str


__all__ = [
    "lane_id", "TrackBHeadResult", "PackagedTrackBHead", "TrackBWorkerOutput",
    "TrackBImportOutcome",
]
