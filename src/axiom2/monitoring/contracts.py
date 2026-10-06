"""Immutable contracts for the deterministic Axiom monitoring boundary."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.evidence.contracts import StrictContract


class CandidateState(str, Enum):
    DISCOVERED = "DISCOVERED"
    WATCHING = "WATCHING"
    WAITING = "WAITING"
    TRIGGERED = "TRIGGERED"
    REVALIDATING = "REVALIDATING"
    READY = "READY"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    EXECUTED = "EXECUTED"
    MONITORING_POSITION = "MONITORING_POSITION"
    EXIT_TRIGGERED = "EXIT_TRIGGERED"
    CLOSED = "CLOSED"


class MaterialEventKind(str, Enum):
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    PRICE_CROSSED = "PRICE_CROSSED"
    VOLUME_CROSSED = "VOLUME_CROSSED"
    FRESHNESS_FAILURE = "FRESHNESS_FAILURE"
    CONNECTION_LOST = "CONNECTION_LOST"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"
    THESIS_INVALIDATED = "THESIS_INVALIDATED"
    CANDIDATE_EXPIRED = "CANDIDATE_EXPIRED"
    EXIT_TRIGGERED = "EXIT_TRIGGERED"


class MonitorEventKind(str, Enum):
    CANDIDATE_REGISTERED = "CANDIDATE_REGISTERED"
    WATCHING_STARTED = "WATCHING_STARTED"
    WAIT_ENTERED = "WAIT_ENTERED"
    OBSERVATION_ACCEPTED = "OBSERVATION_ACCEPTED"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    FRESHNESS_FAILURE = "FRESHNESS_FAILURE"
    CONNECTION_LOST = "CONNECTION_LOST"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"
    THESIS_INVALIDATED = "THESIS_INVALIDATED"
    CANDIDATE_EXPIRED = "CANDIDATE_EXPIRED"
    RESEARCH_RESPONSE = "RESEARCH_RESPONSE"
    REVALIDATION_STARTED = "REVALIDATION_STARTED"
    REVALIDATION_PASSED = "REVALIDATION_PASSED"
    ENTRY_EXECUTED = "ENTRY_EXECUTED"
    POSITION_RECONCILED = "POSITION_RECONCILED"
    EXIT_TRIGGERED = "EXIT_TRIGGERED"
    EXIT_CLOSED = "EXIT_CLOSED"
    RESEARCH_WAKEUP_DELIVERED = "RESEARCH_WAKEUP_DELIVERED"


class ThresholdRule(StrictContract):
    metric: Literal["PRICE", "VOLUME"]
    operator: Literal["AT_OR_ABOVE", "AT_OR_BELOW"]
    threshold: int = Field(gt=0)
    role: Literal["ENTRY", "INVALIDATION", "EXIT"]

    @model_validator(mode="after")
    def validate_metric(self) -> "ThresholdRule":
        if self.metric == "VOLUME" and self.operator == "AT_OR_BELOW":
            raise ValueError("volume conditions may only use AT_OR_ABOVE")
        return self


class CandidateRegistration(StrictContract):
    candidate_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    thesis_version: str = Field(min_length=1)
    thesis_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    registered_at: datetime
    expires_at: datetime
    freshness_seconds: int = Field(gt=0)
    entry_rules: tuple[ThresholdRule, ...] = Field(min_length=1)
    invalidation_rules: tuple[ThresholdRule, ...] = ()
    exit_rules: tuple[ThresholdRule, ...] = ()
    state: CandidateState = CandidateState.DISCOVERED

    @field_validator("registered_at", "expires_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "CandidateRegistration":
        if self.expires_at <= self.registered_at:
            raise ValueError("candidate must expire after registration")
        if any(rule.role != "ENTRY" for rule in self.entry_rules):
            raise ValueError("entry_rules must contain ENTRY rules")
        if any(rule.role != "INVALIDATION" for rule in self.invalidation_rules):
            raise ValueError("invalidation_rules must contain INVALIDATION rules")
        if any(rule.role != "EXIT" for rule in self.exit_rules):
            raise ValueError("exit_rules must contain EXIT rules")
        return self


class MarketObservation(StrictContract):
    candidate_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    source_sequence: int = Field(ge=0)
    observed_at: datetime
    received_at: datetime
    connected: bool
    price_micros: int | None = Field(default=None, gt=0)
    volume: int | None = Field(default=None, ge=0)

    @field_validator("observed_at", "received_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_observation(self) -> "MarketObservation":
        if self.received_at < self.observed_at:
            raise ValueError("received_at cannot precede observed_at")
        if self.connected and self.price_micros is None and self.volume is None:
            raise ValueError("connected observation must contain price or volume")
        return self


class ResearchWakeup(StrictContract):
    wakeup_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    material_event_id: str = Field(min_length=1)
    candidate_version: int = Field(ge=0)
    reason: MaterialEventKind
    observation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchResponse(StrictContract):
    response_id: str = Field(min_length=1)
    wakeup_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    candidate_version: int = Field(ge=0)
    decision: Literal["REVALIDATE", "INVALIDATE", "IGNORE"]


class ReconciliationReceipt(StrictContract):
    receipt_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    kind: Literal["ENTRY_EXECUTED", "POSITION_RECONCILED", "EXIT_CLOSED"]
    external_execution_id: str = Field(min_length=1)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class MonitorEvent(StrictContract):
    sequence: int = Field(default=0, ge=0)
    previous_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    event_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    kind: MonitorEventKind
    source_sequence: int | None = Field(default=None, ge=0)
    occurred_at: datetime
    payload: dict[str, object] = {}

    @field_validator("occurred_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class MonitorReceipt(StrictContract):
    event: MonitorEvent
    received_at: datetime
    event_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("received_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value
