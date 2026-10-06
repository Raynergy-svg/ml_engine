"""Immutable, authority-free contracts for candidate monitoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any


class CandidateState(str, Enum):
    WATCHING = "WATCHING"
    WAITING = "WAITING"
    TRIGGERED = "TRIGGERED"
    REVALIDATING = "REVALIDATING"
    READY = "READY"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


TERMINAL_STATES = frozenset(
    {CandidateState.INVALIDATED, CandidateState.EXPIRED}
)
NONTERMINAL_STATES = frozenset(
    {
        CandidateState.WATCHING,
        CandidateState.WAITING,
        CandidateState.TRIGGERED,
        CandidateState.REVALIDATING,
        CandidateState.READY,
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{stable_digest(parts)[:32]}"


def _required_text(name: str, value: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _required_digest(name: str, value: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _immutable_facts(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("facts must be a mapping")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    candidate_id: str
    candidate_version: str
    observation_id: str
    observed_at_ns: int
    freshness_deadline_ns: int
    evidence_digest: str
    raw_observation_digest: str
    confirmation: bool = False
    invalidated: bool = False
    invalidation_reason: str | None = None
    facts: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_id", self.candidate_id),
            ("candidate_version", self.candidate_version),
            ("observation_id", self.observation_id),
        ):
            _required_text(name, value)
        if type(self.observed_at_ns) is not int or self.observed_at_ns < 0:
            raise ValueError("observed_at_ns must be a non-negative integer")
        if (
            type(self.freshness_deadline_ns) is not int
            or self.freshness_deadline_ns < self.observed_at_ns
        ):
            raise ValueError(
                "freshness_deadline_ns must be >= observed_at_ns"
            )
        _required_digest("evidence_digest", self.evidence_digest)
        _required_digest("raw_observation_digest", self.raw_observation_digest)
        if type(self.confirmation) is not bool or type(self.invalidated) is not bool:
            raise ValueError("confirmation and invalidated must be boolean")
        if self.invalidated and not self.invalidation_reason:
            raise ValueError("invalidated observations require a reason")
        if self.invalidation_reason is not None:
            _required_text("invalidation_reason", self.invalidation_reason)
        object.__setattr__(self, "facts", _immutable_facts(self.facts))

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_version": self.candidate_version,
            "observation_id": self.observation_id,
            "observed_at_ns": self.observed_at_ns,
            "freshness_deadline_ns": self.freshness_deadline_ns,
            "evidence_digest": self.evidence_digest,
            "raw_observation_digest": self.raw_observation_digest,
            "confirmation": self.confirmation,
            "invalidated": self.invalidated,
            "invalidation_reason": self.invalidation_reason,
            "facts": dict(self.facts),
        }


@dataclass(frozen=True, slots=True)
class MaterialEvent:
    event_id: str
    candidate_id: str
    candidate_version: str
    source_observation_id: str
    kind: str
    occurred_at_ns: int
    evidence_digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("event_id", self.event_id),
            ("candidate_id", self.candidate_id),
            ("candidate_version", self.candidate_version),
            ("source_observation_id", self.source_observation_id),
            ("kind", self.kind),
        ):
            _required_text(name, value)
        if type(self.occurred_at_ns) is not int or self.occurred_at_ns < 0:
            raise ValueError("occurred_at_ns must be a non-negative integer")
        _required_digest("evidence_digest", self.evidence_digest)

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "candidate_id": self.candidate_id,
            "candidate_version": self.candidate_version,
            "source_observation_id": self.source_observation_id,
            "kind": self.kind,
            "occurred_at_ns": self.occurred_at_ns,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class ResearchWakeup:
    wakeup_id: str
    material_event_id: str
    candidate_id: str
    candidate_version: str
    created_at_ns: int
    bound_evidence_digest: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "wakeup_id": self.wakeup_id,
            "material_event_id": self.material_event_id,
            "candidate_id": self.candidate_id,
            "candidate_version": self.candidate_version,
            "created_at_ns": self.created_at_ns,
            "bound_evidence_digest": self.bound_evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class ResearchResult:
    wakeup_id: str
    candidate_id: str
    candidate_version: str
    completed_at_ns: int
    fresh_until_ns: int
    evidence_digest: str
    qualifies: bool
    invalidated: bool = False
    facts: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        for name, value in (
            ("wakeup_id", self.wakeup_id),
            ("candidate_id", self.candidate_id),
            ("candidate_version", self.candidate_version),
        ):
            _required_text(name, value)
        if (
            type(self.completed_at_ns) is not int
            or type(self.fresh_until_ns) is not int
            or self.completed_at_ns < 0
            or self.fresh_until_ns < self.completed_at_ns
        ):
            raise ValueError("research timestamps are not ordered")
        _required_digest("evidence_digest", self.evidence_digest)
        if type(self.qualifies) is not bool or type(self.invalidated) is not bool:
            raise ValueError("qualifies and invalidated must be boolean")
        object.__setattr__(self, "facts", _immutable_facts(self.facts))

    def to_payload(self) -> dict[str, Any]:
        return {
            "wakeup_id": self.wakeup_id,
            "candidate_id": self.candidate_id,
            "candidate_version": self.candidate_version,
            "completed_at_ns": self.completed_at_ns,
            "fresh_until_ns": self.fresh_until_ns,
            "evidence_digest": self.evidence_digest,
            "qualifies": self.qualifies,
            "invalidated": self.invalidated,
            "facts": dict(self.facts),
        }
