"""Formal AXIOM evidence contracts.

These models are payload contracts. Content digests and producer/local signatures
live in :class:`SignedEnvelope`, avoiding self-referential hashes.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator

from .base import StrictContract

Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40,64}$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=200)]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


class AssetClass(str, Enum):
    FX = "fx"
    EQUITY = "equity"
    CRYPTO = "crypto"
    FILINGS = "filings"
    MULTI_ASSET = "multi_asset"
    PORTFOLIO = "portfolio"
    OUTCOMES = "outcomes"


class ResourceClass(str, Enum):
    CPU_SMALL = "cpu-small"
    CPU_MEMORY = "cpu-memory"
    GPU_SMALL = "gpu-small"
    GPU_LARGE = "gpu-large"
    DOCUMENT_PROCESSING = "document-processing"
    VERIFICATION = "verification"


class DispositionState(str, Enum):
    CREATED = "CREATED"
    RECEIVED = "RECEIVED"
    HASH_VERIFIED = "HASH_VERIFIED"
    POLICY_VERIFIED = "POLICY_VERIFIED"
    METRIC_REPLAYED = "METRIC_REPLAYED"
    QUARANTINED = "QUARANTINED"
    SHADOW = "SHADOW"
    OPERATOR_APPROVED = "OPERATOR_APPROVED"
    CHAMPION = "CHAMPION"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class AuthorityRole(str, Enum):
    PRODUCER = "producer"
    LOCAL_IMPORTER = "local_importer"
    INDEPENDENT_VERIFIER = "independent_verifier"
    OPERATOR = "operator"
    PROMOTION_SERVICE = "promotion_service"
    SAFETY_MONITOR = "safety_monitor"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ImportDecision(str, Enum):
    ACCEPT_FOR_QUARANTINE = "ACCEPT_FOR_QUARANTINE"
    REJECT = "REJECT"


class PartitionRef(StrictContract):
    partition_id: Identifier
    uri: NonEmptyString
    digest: Sha256Digest
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)


class ArtifactRef(StrictContract):
    artifact_id: Identifier
    relative_path: NonEmptyString
    digest: Sha256Digest
    size_bytes: int = Field(ge=0)
    media_type: NonEmptyString

    @field_validator("relative_path")
    @classmethod
    def path_must_be_relative_and_safe(cls, value: str) -> str:
        parts = value.replace("\\", "/").split("/")
        if value.startswith(("/", "\\")) or ".." in parts:
            raise ValueError("artifact path must be relative and must not traverse parents")
        return value


class MetricValue(StrictContract):
    value: float
    unit: str | None = None
    tolerance: float | None = Field(default=None, ge=0.0)


class GateResult(StrictContract):
    gate_id: Identifier
    status: GateStatus
    observed: JsonValue | None = None
    threshold: JsonValue | None = None
    reason: str | None = None


class SafetyAssertion(StrictContract):
    assertion_id: Identifier
    passed: bool
    details: JsonValue | None = None


class DatasetManifest(StrictContract):
    dataset_id: Identifier
    asset_class: AssetClass
    domain: Identifier
    provider: Identifier
    retrieved_at: datetime
    coverage_start: date
    coverage_end: date
    instruments: tuple[Identifier, ...] = Field(min_length=1)
    data_schema_version: NonEmptyString
    point_in_time_rules: tuple[NonEmptyString, ...] = Field(min_length=1)
    exclusions: tuple[str, ...] = ()
    license_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    normalized_manifest_digests: tuple[Sha256Digest, ...] = ()
    partitions: tuple[PartitionRef, ...] = Field(min_length=1)

    _utc_retrieved = field_validator("retrieved_at")(_require_utc)

    @model_validator(mode="after")
    def validate_dataset_bounds(self) -> "DatasetManifest":
        if self.coverage_end < self.coverage_start:
            raise ValueError("coverage_end must not precede coverage_start")
        if len(set(self.instruments)) != len(self.instruments):
            raise ValueError("instruments must be unique")
        ids = [partition.partition_id for partition in self.partitions]
        if len(set(ids)) != len(ids):
            raise ValueError("partition_id values must be unique")
        return self


class StrategyManifest(StrictContract):
    strategy_id: Identifier
    lane_id: Identifier
    experiment_id: Identifier
    hypothesis: NonEmptyString
    exploratory: bool
    frozen_at: datetime
    universe: tuple[Identifier, ...] = Field(min_length=1)
    causal_lag: NonEmptyString
    parameters: dict[str, JsonValue]
    cost_model: dict[str, JsonValue]
    evaluation_windows: tuple[NonEmptyString, ...] = Field(min_length=1)
    search_space: dict[str, JsonValue]
    trial_budget: int = Field(ge=1)
    metrics: tuple[Identifier, ...] = Field(min_length=1)
    promotion_criteria: tuple[NonEmptyString, ...] = Field(min_length=1)

    _utc_frozen = field_validator("frozen_at")(_require_utc)


class CapabilityProfile(StrictContract):
    profile_id: Identifier
    read_scopes: tuple[NonEmptyString, ...] = ()
    write_scopes: tuple[NonEmptyString, ...] = ()
    network_endpoints: tuple[NonEmptyString, ...] = ()
    may_read_broker_credentials: Literal[False] = False
    may_place_or_cancel_orders: Literal[False] = False
    may_read_operator_keys: Literal[False] = False
    may_change_halts: Literal[False] = False
    may_change_live_gate: Literal[False] = False
    may_write_champion_pointer: Literal[False] = False
    may_modify_local_models: Literal[False] = False
    may_approve_evidence: Literal[False] = False


class ModelTrainingSpec(StrictContract):
    target: Identifier
    features: tuple[Identifier, ...] = Field(min_length=1)
    causal_lag: NonEmptyString
    search_space: dict[str, JsonValue]
    metrics: tuple[Identifier, ...] = Field(min_length=1)
    promotion_criteria: tuple[NonEmptyString, ...] = Field(min_length=1)


class JobManifest(StrictContract):
    job_id: Identifier
    git_commit: GitCommit
    container_digest: Sha256Digest
    dataset_manifest_digests: tuple[Sha256Digest, ...] = Field(min_length=1)
    strategy_manifest_digest: Sha256Digest | None = None
    model_training_spec: ModelTrainingSpec | None = None
    feature_pipeline_version: NonEmptyString
    configuration_digest: Sha256Digest
    random_seeds: tuple[int, ...] = Field(min_length=1)
    assigned_unit: NonEmptyString
    resource_class: ResourceClass
    capability_profile_digest: Sha256Digest
    trial_budget: int = Field(ge=1)
    expected_outputs: tuple[NonEmptyString, ...] = Field(min_length=1)
    created_at: datetime

    _utc_created = field_validator("created_at")(_require_utc)

    @model_validator(mode="after")
    def exactly_one_workload_spec(self) -> "JobManifest":
        count = int(self.strategy_manifest_digest is not None) + int(self.model_training_spec is not None)
        if count != 1:
            raise ValueError("exactly one of strategy_manifest_digest or model_training_spec is required")
        return self


class EvaluationReport(StrictContract):
    report_id: Identifier
    job_manifest_digest: Sha256Digest
    evaluator_id: Identifier
    independent: bool
    created_at: datetime
    temporal_holdout: NonEmptyString
    purge_observations: int = Field(ge=0)
    embargo_observations: int = Field(ge=0)
    trial_count: int = Field(ge=1)
    effective_sample_size: float = Field(gt=0.0)
    metrics: dict[Identifier, MetricValue] = Field(min_length=1)
    regime_metrics: dict[Identifier, dict[Identifier, MetricValue]] = Field(default_factory=dict)
    gates: tuple[GateResult, ...] = Field(min_length=1)
    incumbent_comparison: dict[str, JsonValue]
    passed: bool

    _utc_created = field_validator("created_at")(_require_utc)

    @model_validator(mode="after")
    def verdict_matches_gates(self) -> "EvaluationReport":
        gate_ids = [gate.gate_id for gate in self.gates]
        if len(set(gate_ids)) != len(gate_ids):
            raise ValueError("gate_id values must be unique")
        any_failed = any(gate.status == GateStatus.FAIL for gate in self.gates)
        if self.passed == any_failed:
            raise ValueError("passed must be false exactly when one or more gates fail")
        return self


class EvidencePackage(StrictContract):
    package_id: Identifier
    lane_id: Identifier
    producer_id: Identifier
    created_at: datetime
    job_manifest_digest: Sha256Digest
    dataset_manifest_digests: tuple[Sha256Digest, ...] = Field(min_length=1)
    strategy_manifest_digest: Sha256Digest | None = None
    evaluation_report_digests: tuple[Sha256Digest, ...] = Field(min_length=1)
    derived_from_package_digests: tuple[Sha256Digest, ...] = ()
    artifacts: tuple[ArtifactRef, ...]
    logs: tuple[ArtifactRef, ...] = ()
    safety_assertions: tuple[SafetyAssertion, ...] = Field(min_length=1)
    checksums: dict[NonEmptyString, Sha256Digest]

    _utc_created = field_validator("created_at")(_require_utc)

    @model_validator(mode="after")
    def validate_package_contents(self) -> "EvidencePackage":
        refs = self.artifacts + self.logs
        paths = [ref.relative_path for ref in refs]
        if len(set(paths)) != len(paths):
            raise ValueError("artifact and log paths must be unique")
        artifact_ids = [ref.artifact_id for ref in refs]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifact_id values must be unique")
        for ref in refs:
            if self.checksums.get(ref.relative_path) != ref.digest:
                raise ValueError(f"checksums must bind {ref.relative_path!r} to its digest")
        if set(self.checksums) != set(paths):
            raise ValueError("checksums must contain exactly the declared artifact and log paths")
        if not all(assertion.passed for assertion in self.safety_assertions):
            raise ValueError("an EvidencePackage cannot assert failed safety controls")
        return self


class DispositionEvent(StrictContract):
    package_digest: Sha256Digest
    sequence: int = Field(ge=0)
    previous_event_digest: Sha256Digest | None
    from_state: DispositionState | None
    to_state: DispositionState
    authority: AuthorityRole
    actor_id: Identifier
    signer_key_id: Identifier
    occurred_at: datetime
    reason: NonEmptyString
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _utc_occurred = field_validator("occurred_at")(_require_utc)

    @model_validator(mode="after")
    def validate_genesis_shape(self) -> "DispositionEvent":
        if self.sequence == 0:
            if self.previous_event_digest is not None or self.from_state is not None:
                raise ValueError("genesis event must have no predecessor or from_state")
        elif self.previous_event_digest is None or self.from_state is None:
            raise ValueError("non-genesis event requires previous_event_digest and from_state")
        return self


class ImportCheck(StrictContract):
    check_id: Identifier
    passed: bool
    details: str | None = None


class LocalImportVerdict(StrictContract):
    verdict_id: Identifier
    package_digest: Sha256Digest
    importer_id: Identifier
    created_at: datetime
    checks: tuple[ImportCheck, ...] = Field(min_length=1)
    decision: ImportDecision
    rejection_reason: str | None = None

    _utc_created = field_validator("created_at")(_require_utc)

    @model_validator(mode="after")
    def decision_matches_checks(self) -> "LocalImportVerdict":
        failed = [check.check_id for check in self.checks if not check.passed]
        if failed and self.decision != ImportDecision.REJECT:
            raise ValueError("a failed import check requires REJECT")
        if not failed and self.decision != ImportDecision.ACCEPT_FOR_QUARANTINE:
            raise ValueError("all passing checks require ACCEPT_FOR_QUARANTINE")
        if self.decision == ImportDecision.REJECT and not self.rejection_reason:
            raise ValueError("REJECT requires rejection_reason")
        if self.decision != ImportDecision.REJECT and self.rejection_reason is not None:
            raise ValueError("accepted verdict cannot include rejection_reason")
        return self


class ChampionPointer(StrictContract):
    pointer_id: Identifier
    lane_id: Identifier
    package_digest: Sha256Digest
    artifact_id: Identifier
    artifact_digest: Sha256Digest
    disposition_head_digest: Sha256Digest
    actor_id: Identifier
    signer_key_id: Identifier
    created_at: datetime

    _utc_created = field_validator("created_at")(_require_utc)


class SignatureRecord(StrictContract):
    algorithm: Literal["Ed25519"] = "Ed25519"
    key_id: Identifier
    payload_type: Identifier
    signed_digest: Sha256Digest
    signature_b64: NonEmptyString
    created_at: datetime

    _utc_created = field_validator("created_at")(_require_utc)


class SignedEnvelope(StrictContract):
    payload_type: Identifier
    payload: dict[str, Any]
    payload_digest: Sha256Digest
    signature: SignatureRecord


class DispositionReceipt(StrictContract):
    """Local immutable observation time paired with one signed disposition."""

    envelope: SignedEnvelope
    received_at: datetime

    _utc_received = field_validator("received_at")(_require_utc)
