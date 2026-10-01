"""Task 5: signed research history in the existing EvidenceStore.

Only the configured local evidence authority receives a signer. Researchers
supply proposals, not storage/key capabilities. Every read replays authenticated
history; no mutable index supplies attempt counts or frozen identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta
import json
import re
from types import MappingProxyType
from typing import Callable, Literal, Mapping

from pydantic import ConfigDict, Field, JsonValue, ValidationInfo, field_validator

from src.axiom2.contracts.research_proposal import DataDependency, ResearchProposal, validate_research_proposal
from src.axiom2.data.temporal import _as_utc
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole, SignedEnvelope, StrictContract
from src.evidence.contracts.models import Identifier, Sha256Digest
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope
from src.evidence.store import ConcurrentHeadError, EvidenceStore, StoreCorruptionError
from src.evidence.transition_policy import AuthorityRegistry


class ResearchEvent(StrictContract):
    """One versioned append; semantics and privileges are checked during replay."""

    # Proposal text is evidence: do not inherit whitespace normalization.
    model_config = ConfigDict(str_strip_whitespace=False)

    sequence: int = Field(ge=0)
    previous_event_digest: Sha256Digest | None
    kind: Literal["REGISTERED", "FAILED", "FROZEN", "HOLDOUT_DECLARED", "HOLDOUT_OPENED", "HOLDOUT_CONSUMED"]
    actor_id: Identifier
    occurred_at: datetime
    body: dict[str, JsonValue]

    @field_validator("occurred_at", mode="before")
    @classmethod
    def aware_time(cls, value: object, info: ValidationInfo) -> datetime:
        if info.mode == "json" and type(value) is str:
            value = datetime.fromisoformat(value)
        return _as_utc(value, "occurred_at")


class ResearchReceipt(StrictContract):
    """A locally trusted receipt, using the same envelope primitives as dispositions."""

    envelope: SignedEnvelope
    received_at: datetime

    @field_validator("received_at", mode="before")
    @classmethod
    def aware_time(cls, value: object, info: ValidationInfo) -> datetime:
        if info.mode == "json" and type(value) is str:
            value = datetime.fromisoformat(value)
        return _as_utc(value, "received_at")


@dataclass(frozen=True, slots=True)
class RegisteredExperiment:
    proposal: ResearchProposal
    registration_digest: str
    registered_sequence: int
    failure_reason: str | None = None
    candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class FrozenCandidate:
    candidate_id: str
    experiment_id: str
    registration_digest: str
    artifact_hash: str
    code_commit: str
    attempt_count: int
    frozen_at: datetime
    freeze_sequence: int
    actor_id: str
    signer_key_id: str


@dataclass(frozen=True, slots=True)
class HoldoutRecord:
    holdout_id: str
    dataset_digest: str
    coverage_start: datetime
    coverage_end: datetime
    declaration_digest: str
    declared_sequence: int
    candidate_id: str | None = None
    opened_digest: str | None = None
    opened_by: str | None = None
    attempt_count: int | None = None
    result_hash: str | None = None
    consumption_digest: str | None = None

    @property
    def status(self) -> str:
        if self.consumption_digest is not None:
            return "CONSUMED"
        return "OPENED" if self.opened_digest is not None else "UNTOUCHED"


@dataclass(frozen=True, slots=True)
class ResearchState:
    experiments: Mapping[str, RegisteredExperiment]
    candidates: Mapping[str, FrozenCandidate]
    holdouts: Mapping[str, HoldoutRecord]
    head_digest: str | None
    event_count: int

    @property
    def attempt_count(self) -> int:
        return len(self.experiments)


def _text(value: object, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a plain string")
    if not value or value != value.strip():
        raise ValueError(f"{field} must be nonempty without surrounding whitespace")
    return value


def _digest(value: object, field: str) -> str:
    value = _text(value, field)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a full lowercase SHA-256 digest")
    return value


def _commit(value: object) -> str:
    value = _text(value, "code_commit")
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None:
        raise ValueError("code_commit must be a full lowercase Git identity")
    return value


def _body(data: dict, expected: set[str]) -> None:
    if set(data) != expected:
        raise ValueError("research event body has missing or unknown fields")


def _proposal(data: dict) -> ResearchProposal:
    if type(data) is not dict or set(data) != {f.name for f in fields(ResearchProposal)}:
        raise ValueError("registered proposal has missing or unknown fields")
    values = dict(data)
    for name in ("feature_families", "baseline_ids", "primary_metrics", "data_dependencies"):
        if type(values[name]) is not list:
            raise ValueError("serialized proposal collections must be arrays")
        values[name] = tuple(values[name])
    values["data_dependencies"] = tuple(DataDependency(**item) for item in values["data_dependencies"])
    return ResearchProposal(**values)


def required_role(kind: str) -> AuthorityRole:
    if kind in ("REGISTERED", "FAILED", "FROZEN"):
        return AuthorityRole.LOCAL_IMPORTER
    if kind == "HOLDOUT_DECLARED":
        return AuthorityRole.OPERATOR
    if kind in ("HOLDOUT_OPENED", "HOLDOUT_CONSUMED"):
        return AuthorityRole.INDEPENDENT_VERIFIER
    raise ValueError("unsupported research event kind")


def _apply_event(
    event: ResearchEvent, digest: str, signer_key_id: str, experiments: dict, candidates: dict, holdouts: dict
) -> None:
    # Materialize plain JSON, never call provider coercion hooks during replay.
    data = json.loads(canonical_bytes(event.body))
    if event.kind == "REGISTERED":
        _body(data, {"proposal"})
        item = _proposal(data["proposal"])
        if item.experiment_id in experiments:
            raise ValueError("experiment ID is already registered")
        experiments[item.experiment_id] = RegisteredExperiment(item, digest, event.sequence)
        return
    if event.kind in ("FAILED", "FROZEN"):
        expected = (
            {"experiment_id", "reason"}
            if event.kind == "FAILED"
            else {"experiment_id", "registration_digest", "artifact_hash", "code_commit", "attempt_count"}
        )
        _body(data, expected)
        identifier = _text(data["experiment_id"], "experiment_id")
        registered = experiments.get(identifier)
        if registered is None:
            raise ValueError("experiment must be registered first")
        if registered.failure_reason is not None:
            raise ValueError("experiment has already failed")
        if event.kind == "FAILED":
            experiments[identifier] = replace(registered, failure_reason=_text(data["reason"], "reason"))
            return
        if registered.candidate_id is not None:
            raise ValueError("experiment already has a frozen candidate")
        if data["registration_digest"] != registered.registration_digest:
            raise ValueError("frozen candidate does not bind the registered proposal")
        if _commit(data["code_commit"]) != registered.proposal.code_commit:
            raise ValueError("code_commit differs from preregistration; register a new attempt")
        if type(data["attempt_count"]) is not int or data["attempt_count"] != len(experiments):
            raise ValueError("attempt_count does not match complete registry history")
        candidate = FrozenCandidate(
            digest,
            identifier,
            registered.registration_digest,
            _digest(data["artifact_hash"], "artifact_hash"),
            data["code_commit"],
            len(experiments),
            event.occurred_at,
            event.sequence,
            event.actor_id,
            signer_key_id,
        )
        candidates[digest] = candidate
        experiments[identifier] = replace(registered, candidate_id=digest)
        return
    # Holdout lifecycle is defined once, in the dedicated Task 5 authority module.
    from .holdout import apply_holdout_event

    apply_holdout_event(event, digest, signer_key_id, data, experiments, candidates, holdouts)


def reconstruct_research(
    receipts: tuple[ResearchReceipt, ...], *, trust_store: TrustStore, authorities: AuthorityRegistry
) -> ResearchState:
    """Authenticate and replay the complete global research journal; fail closed."""
    experiments, candidates, holdouts = {}, {}, {}
    prior_digest = None
    prior_time = prior_receipt_time = None
    for sequence, receipt in enumerate(receipts):
        try:
            envelope = receipt.envelope
            event = verify_envelope(envelope, ResearchEvent, trust_store)
            if event.sequence != sequence or event.previous_event_digest != prior_digest:
                raise ValueError("research sequence or previous digest mismatch")
            if envelope.signature.created_at != event.occurred_at:
                raise ValueError("research signing and event timestamps disagree")
            if abs(receipt.received_at - event.occurred_at) > timedelta(seconds=30):
                raise ValueError("research receipt exceeds allowed clock skew")
            if prior_time is not None and (event.occurred_at < prior_time or receipt.received_at < prior_receipt_time):
                raise ValueError("research timestamps regressed")
            trust_store.require_trusted_at_receipt(envelope.signature.key_id, receipt.received_at)
            authorities.authorize_identity(
                actor_id=event.actor_id, role=required_role(event.kind), key_id=envelope.signature.key_id
            )
            _apply_event(event, envelope.payload_digest, envelope.signature.key_id, experiments, candidates, holdouts)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise StoreCorruptionError(f"invalid research event at sequence {sequence}: {exc}") from exc
        prior_digest, prior_time, prior_receipt_time = envelope.payload_digest, event.occurred_at, receipt.received_at
    return ResearchState(
        MappingProxyType(experiments),
        MappingProxyType(candidates),
        MappingProxyType(holdouts),
        prior_digest,
        len(receipts),
    )


class ExperimentRegistry:
    """Local evidence service. Inject keys explicitly; never create or discover them."""

    def __init__(self, store: EvidenceStore, *, signer: Ed25519Signer, actor_id: str) -> None:
        self.store = store
        self.signer = signer
        self.actor_id = _text(actor_id, "actor_id")

    def snapshot(self) -> ResearchState:
        return reconstruct_research(
            self.store.load_research_ledger(), trust_store=self.store.trust_store, authorities=self.store.authorities
        )

    def _record(self, kind: str, build: Callable[[ResearchState], dict | None]) -> ResearchState:
        for _ in range(32):
            state = self.snapshot()
            now = _as_utc(self.store._trusted_clock(), "trusted_clock")
            self.store.trust_store.require_trusted_at_receipt(self.signer.key_id, now)
            self.store.authorities.authorize_identity(
                actor_id=self.actor_id, role=required_role(kind), key_id=self.signer.key_id
            )
            body = build(state)
            if body is None:  # Exact idempotent retry; never counts as a new attempt.
                return state
            event = ResearchEvent(
                sequence=state.event_count,
                previous_event_digest=state.head_digest,
                kind=kind,
                actor_id=self.actor_id,
                occurred_at=now,
                body=body,
            )
            envelope = self.signer.sign(event, created_at=now)
            # Run semantic validation before asking the store to commit. Store
            # independently repeats verification under its lock before publication.
            _apply_event(
                event,
                envelope.payload_digest,
                self.signer.key_id,
                dict(state.experiments),
                dict(state.candidates),
                dict(state.holdouts),
            )
            try:
                self.store.append_research_event(envelope, expected_head_digest=state.head_digest)
            except ConcurrentHeadError:
                continue
            return self.snapshot()
        raise ConcurrentHeadError("research journal contention; retry the operation")

    def register_experiment(self, proposal: ResearchProposal) -> RegisteredExperiment:
        validate_research_proposal(proposal)
        data = json.loads(canonical_bytes(asdict(proposal)))
        # Detach caller-owned objects before signing, and revalidate the snapshot.
        item = _proposal(data)

        def build(state: ResearchState) -> dict | None:
            previous = state.experiments.get(item.experiment_id)
            if previous is not None:
                if previous.proposal != item:
                    raise ValueError("experiment ID is already registered with a different proposal")
                return None
            return {"proposal": data}

        return self._record("REGISTERED", build).experiments[item.experiment_id]

    def record_failure(self, experiment_id: str, reason: str) -> RegisteredExperiment:
        identifier, reason = _text(experiment_id, "experiment_id"), _text(reason, "reason")

        def build(state: ResearchState) -> dict | None:
            previous = state.experiments.get(identifier)
            if previous is not None and previous.failure_reason == reason:
                return None
            return {"experiment_id": identifier, "reason": reason}

        return self._record("FAILED", build).experiments[identifier]

    def freeze_candidate(self, experiment_id: str, artifact_hash: str, code_commit: str) -> FrozenCandidate:
        identifier = _text(experiment_id, "experiment_id")
        artifact_hash, code_commit = _digest(artifact_hash, "artifact_hash"), _commit(code_commit)

        def build(state: ResearchState) -> dict | None:
            previous = state.experiments.get(identifier)
            if previous is None:
                raise ValueError("experiment must be registered first")
            if previous.failure_reason is not None:
                raise ValueError("experiment has already failed")
            if previous.candidate_id is not None:
                candidate = state.candidates[previous.candidate_id]
                if (candidate.artifact_hash, candidate.code_commit) != (artifact_hash, code_commit):
                    raise ValueError("frozen candidate identity is immutable")
                return None
            return {
                "experiment_id": identifier,
                "registration_digest": previous.registration_digest,
                "artifact_hash": artifact_hash,
                "code_commit": code_commit,
                "attempt_count": state.attempt_count,
            }

        state = self._record("FROZEN", build)
        return state.candidates[state.experiments[identifier].candidate_id]


def register_experiment(proposal: ResearchProposal, *, registry: ExperimentRegistry) -> RegisteredExperiment:
    """Explicit-context functional entrypoint; no ambient authority or default path."""
    return registry.register_experiment(proposal)


def freeze_candidate(
    experiment_id: str, artifact_hash: str, code_commit: str, *, registry: ExperimentRegistry
) -> FrozenCandidate:
    """Freeze one exact registered candidate; does not promote it."""
    return registry.freeze_candidate(experiment_id, artifact_hash, code_commit)
