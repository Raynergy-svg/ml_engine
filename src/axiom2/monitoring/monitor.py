"""Deterministic candidate monitoring with no order or broker authority.

The monitor consumes normalized observations, persists every state transition, and
emits at-least-once research wakeups. It does not import or call execution,
broker, portfolio, capital, or approval implementations. A READY candidate is
only eligible for the existing Axiom validation/risk pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from threading import RLock

from src.evidence.hashing import content_digest

from .contracts import (
    CandidateRegistration,
    CandidateState,
    MarketObservation,
    MaterialEventKind,
    MonitorEvent,
    MonitorEventKind,
    ReconciliationReceipt,
    ResearchResponse,
    ResearchWakeup,
)
from .store import MonitorStore, MonitorStoreCorruption, MonitorStoreOrderingError


class MonitorTransitionError(ValueError):
    """Raised when a candidate cannot take the requested monitor transition."""


class ResearchWakeupSink(Protocol):
    """Provider-neutral, bounded advisor wakeup boundary."""

    def wake(self, wakeup: ResearchWakeup) -> None:
        """Accept one wakeup; implementations must be idempotent by wakeup_id."""


@dataclass(frozen=True)
class _Projection:
    candidate: CandidateRegistration
    state: CandidateState
    version: int
    last_observation: MarketObservation | None
    last_observation_digest: str | None


class AxiomMonitor:
    """Replayable deterministic state machine for one monitor journal."""

    _OBSERVATION_STATE = frozenset(
        {
            CandidateState.WAITING,
            CandidateState.TRIGGERED,
            CandidateState.REVALIDATING,
            CandidateState.READY,
            CandidateState.MONITORING_POSITION,
        }
    )
    _PRE_ENTRY_STATE = frozenset(
        {
            CandidateState.WAITING,
            CandidateState.TRIGGERED,
            CandidateState.REVALIDATING,
            CandidateState.READY,
        }
    )
    _EXPIRABLE_STATE = frozenset(
        {
            CandidateState.DISCOVERED,
            CandidateState.WATCHING,
            CandidateState.WAITING,
            CandidateState.TRIGGERED,
            CandidateState.REVALIDATING,
            CandidateState.READY,
        }
    )
    _OBSERVATION_EVENT_KINDS = frozenset(
        {
            MonitorEventKind.OBSERVATION_ACCEPTED,
            MonitorEventKind.ENTRY_TRIGGERED,
            MonitorEventKind.FRESHNESS_FAILURE,
            MonitorEventKind.CONNECTION_LOST,
            MonitorEventKind.CONNECTION_RESTORED,
            MonitorEventKind.THESIS_INVALIDATED,
            MonitorEventKind.EXIT_TRIGGERED,
        }
    )

    def __init__(
        self,
        store: MonitorStore,
        *,
        wakeup_sink: ResearchWakeupSink | None = None,
    ):
        self.store = store
        self.wakeup_sink = wakeup_sink
        self._mutex = RLock()

    def _events(self) -> tuple[MonitorEvent, ...]:
        return self.store.replay()

    def _event(self, event_id: str) -> MonitorEvent | None:
        for event in self._events():
            if event.event_id == event_id:
                return event
        return None

    def _candidate_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for event in self._events():
            if event.kind != MonitorEventKind.CANDIDATE_REGISTERED:
                continue
            if event.candidate_id not in seen:
                seen.append(event.candidate_id)
        return tuple(seen)

    def _projection(self, candidate_id: str) -> _Projection:
        candidate: CandidateRegistration | None = None
        state: CandidateState | None = None
        version = 0
        last_observation: MarketObservation | None = None
        last_observation_digest: str | None = None

        for event in self._events():
            if event.candidate_id != candidate_id:
                continue
            payload = event.payload
            if event.kind == MonitorEventKind.CANDIDATE_REGISTERED:
                raw_candidate = payload.get("candidate")
                if not isinstance(raw_candidate, dict):
                    raise MonitorStoreCorruption("candidate registration payload is malformed")
                candidate = CandidateRegistration.model_validate(raw_candidate)
            raw_state = payload.get("state")
            if raw_state is not None:
                try:
                    state = CandidateState(str(raw_state))
                except ValueError as exc:
                    raise MonitorStoreCorruption("monitor state payload is malformed") from exc
            raw_version = payload.get("candidate_version")
            if raw_version is not None:
                if not isinstance(raw_version, int) or raw_version < 0:
                    raise MonitorStoreCorruption("candidate version payload is malformed")
                version = raw_version
            if event.kind in self._OBSERVATION_EVENT_KINDS:
                raw_observation = payload.get("observation")
                if not isinstance(raw_observation, dict):
                    raise MonitorStoreCorruption("observation payload is malformed")
                last_observation = MarketObservation.model_validate(raw_observation)
                raw_digest = payload.get("observation_digest")
                expected_digest = content_digest(last_observation)
                if raw_digest != expected_digest:
                    raise MonitorStoreCorruption("observation digest mismatch")
                last_observation_digest = expected_digest

        if candidate is None or state is None:
            raise KeyError(f"unknown candidate: {candidate_id}")
        return _Projection(
            candidate=candidate,
            state=state,
            version=version,
            last_observation=last_observation,
            last_observation_digest=last_observation_digest,
        )

    def state(self, candidate_id: str) -> CandidateState:
        """Return the state reconstructed from durable events."""
        return self._projection(candidate_id).state

    @staticmethod
    def _require_aware(value: datetime, name: str) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise MonitorTransitionError(f"{name} must be timezone-aware")

    def _timeline_time(self, candidate_id: str) -> datetime:
        projection = self._projection(candidate_id)
        latest = projection.candidate.registered_at
        for event in self._events():
            if event.candidate_id == candidate_id and event.occurred_at > latest:
                latest = event.occurred_at
        return latest

    def _append(
        self,
        *,
        event_id: str,
        candidate_id: str,
        kind: MonitorEventKind,
        occurred_at: datetime,
        payload: dict[str, object],
        source_sequence: int | None = None,
        received_at: datetime | None = None,
    ) -> str:
        self._require_aware(occurred_at, "occurred_at")
        requested_received_at = received_at or occurred_at
        self._require_aware(requested_received_at, "received_at")
        receipt_watermark = self.store.latest_received_at()
        effective_received_at = max(
            requested_received_at,
            occurred_at,
            receipt_watermark or requested_received_at,
        )
        event = MonitorEvent(
            event_id=event_id,
            candidate_id=candidate_id,
            kind=kind,
            source_sequence=source_sequence,
            occurred_at=occurred_at,
            payload=payload,
        )
        return self.store.append(event, received_at=effective_received_at)

    @staticmethod
    def _state_payload(
        state: CandidateState,
        version: int,
        **extra: object,
    ) -> dict[str, object]:
        return {
            "state": state.value,
            "candidate_version": version,
            **extra,
        }

    @staticmethod
    def _observation_payload(
        observation: MarketObservation,
        state: CandidateState,
        version: int,
        **extra: object,
    ) -> dict[str, object]:
        digest = content_digest(observation)
        return {
            "observation": observation.model_dump(mode="json"),
            "observation_digest": digest,
            **AxiomMonitor._state_payload(state, version, **extra),
        }

    def register(
        self,
        candidate: CandidateRegistration,
        command_id: str,
    ) -> CandidateState:
        """Durably register a candidate; this cannot authorize an order."""
        if candidate.state != CandidateState.DISCOVERED:
            raise MonitorTransitionError("new candidates must start in DISCOVERED")
        registration_event_id = f"register:{command_id}"
        prior = self._event(registration_event_id)
        if prior is not None:
            return self.state(candidate.candidate_id)
        if candidate.candidate_id in self._candidate_ids():
            raise MonitorTransitionError("candidate is already registered")

        self._append(
            event_id=registration_event_id,
            candidate_id=candidate.candidate_id,
            kind=MonitorEventKind.CANDIDATE_REGISTERED,
            occurred_at=candidate.registered_at,
            payload=self._state_payload(
                CandidateState.DISCOVERED,
                0,
                candidate=candidate.model_dump(mode="json"),
            ),
        )
        return CandidateState.DISCOVERED

    def start_watching(self, candidate_id: str, command_id: str) -> CandidateState:
        """Move DISCOVERED to WATCHING."""
        event_id = f"watch:{command_id}"
        prior = self._event(event_id)
        if prior is not None:
            return self.state(candidate_id)
        projection = self._projection(candidate_id)
        if projection.state != CandidateState.DISCOVERED:
            raise MonitorTransitionError("candidate is not DISCOVERED")
        state = CandidateState.WATCHING
        self._append(
            event_id=event_id,
            candidate_id=candidate_id,
            kind=MonitorEventKind.WATCHING_STARTED,
            occurred_at=self._timeline_time(candidate_id),
            payload=self._state_payload(state, projection.version + 1),
        )
        return state

    def wait(self, candidate_id: str, command_id: str) -> CandidateState:
        """Enter the durable WAITING state."""
        event_id = f"wait:{command_id}"
        prior = self._event(event_id)
        if prior is not None:
            return self.state(candidate_id)
        projection = self._projection(candidate_id)
        if projection.state != CandidateState.WATCHING:
            raise MonitorTransitionError("candidate is not WATCHING")
        state = CandidateState.WAITING
        self._append(
            event_id=event_id,
            candidate_id=candidate_id,
            kind=MonitorEventKind.WAIT_ENTERED,
            occurred_at=self._timeline_time(candidate_id),
            payload=self._state_payload(state, projection.version + 1),
        )
        return state

    @staticmethod
    def _value(rule_metric: str, observation: MarketObservation) -> int | None:
        if rule_metric == "PRICE":
            return observation.price_micros
        if rule_metric == "VOLUME":
            return observation.volume
        raise MonitorTransitionError(f"unsupported threshold metric: {rule_metric}")

    @classmethod
    def _satisfies(
        cls,
        rule: object,
        observation: MarketObservation | None,
    ) -> bool:
        if observation is None or not observation.connected:
            return False
        value = cls._value(rule.metric, observation)
        if value is None:
            return False
        if rule.operator == "AT_OR_ABOVE":
            return value >= rule.threshold
        if rule.operator == "AT_OR_BELOW":
            return value <= rule.threshold
        raise MonitorTransitionError(f"unsupported threshold operator: {rule.operator}")

    @classmethod
    def _all_rules(
        cls,
        rules: tuple[object, ...],
        observation: MarketObservation | None,
    ) -> bool:
        return bool(rules) and all(cls._satisfies(rule, observation) for rule in rules)

    @classmethod
    def _any_rule(
        cls,
        rules: tuple[object, ...],
        observation: MarketObservation | None,
    ) -> bool:
        return any(cls._satisfies(rule, observation) for rule in rules)

    @staticmethod
    def _fresh(candidate: CandidateRegistration, observation: MarketObservation) -> bool:
        return observation.received_at - observation.observed_at <= timedelta(
            seconds=candidate.freshness_seconds
        )

    @staticmethod
    def _material_reason(kind: MonitorEventKind) -> MaterialEventKind:
        reasons = {
            MonitorEventKind.ENTRY_TRIGGERED: MaterialEventKind.ENTRY_TRIGGERED,
            MonitorEventKind.FRESHNESS_FAILURE: MaterialEventKind.FRESHNESS_FAILURE,
            MonitorEventKind.CONNECTION_LOST: MaterialEventKind.CONNECTION_LOST,
            MonitorEventKind.CONNECTION_RESTORED: MaterialEventKind.CONNECTION_RESTORED,
            MonitorEventKind.THESIS_INVALIDATED: MaterialEventKind.THESIS_INVALIDATED,
            MonitorEventKind.CANDIDATE_EXPIRED: MaterialEventKind.CANDIDATE_EXPIRED,
            MonitorEventKind.EXIT_TRIGGERED: MaterialEventKind.EXIT_TRIGGERED,
        }
        try:
            return reasons[kind]
        except KeyError as exc:
            raise MonitorTransitionError(f"{kind.value} is not material") from exc

    def _append_material(
        self,
        *,
        event_id: str,
        candidate_id: str,
        kind: MonitorEventKind,
        state: CandidateState,
        version: int,
        occurred_at: datetime,
        observation: MarketObservation | None = None,
        observation_digest: str | None = None,
        received_at: datetime | None = None,
    ) -> CandidateState:
        if observation is not None:
            observation_digest = content_digest(observation)
        if observation_digest is None:
            observation_digest = content_digest(self._projection(candidate_id).candidate)
        wakeup = ResearchWakeup(
            wakeup_id=f"wakeup:{event_id}",
            candidate_id=candidate_id,
            material_event_id=event_id,
            candidate_version=version,
            reason=self._material_reason(kind),
            observation_digest=observation_digest,
        )
        payload: dict[str, object] = {
            "wakeup_id": wakeup.wakeup_id,
            "wakeup": wakeup.model_dump(mode="json"),
            "material_event": wakeup.reason.value,
            **self._state_payload(state, version),
        }
        source_sequence = None
        if observation is not None:
            payload.update(
                self._observation_payload(
                    observation,
                    state,
                    version,
                    wakeup_id=wakeup.wakeup_id,
                    wakeup=wakeup.model_dump(mode="json"),
                    material_event=wakeup.reason.value,
                )
            )
            source_sequence = observation.source_sequence
        self._append(
            event_id=event_id,
            candidate_id=candidate_id,
            kind=kind,
            occurred_at=occurred_at,
            source_sequence=source_sequence,
            received_at=received_at,
            payload=payload,
        )
        self.process_pending_wakeups()
        return state

    def _source_watermark(self, candidate_id: str) -> int | None:
        values = [
            event.source_sequence
            for event in self._events()
            if event.candidate_id == candidate_id and event.source_sequence is not None
        ]
        return max(values) if values else None

    def _validate_observation_identity(
        self,
        projection: _Projection,
        observation: MarketObservation,
    ) -> None:
        if observation.candidate_id != projection.candidate.candidate_id:
            raise MonitorTransitionError("observation candidate mismatch")
        if observation.instrument_id != projection.candidate.instrument_id:
            raise MonitorTransitionError("observation instrument mismatch")
        prior = self._event(observation.event_id)
        if prior is not None:
            raw_observation = prior.payload.get("observation")
            if not isinstance(raw_observation, dict):
                raise MonitorStoreCorruption("duplicate observation has no payload")
            if content_digest(MarketObservation.model_validate(raw_observation)) != content_digest(
                observation
            ):
                raise MonitorStoreCorruption("conflicting duplicate observation")
            return
        watermark = self._source_watermark(projection.candidate.candidate_id)
        if watermark is not None and observation.source_sequence <= watermark:
            raise MonitorStoreOrderingError("source watermark regression")

    def observe(self, observation: MarketObservation) -> CandidateState:
        """Serialize observations so trigger races re-read durable state."""
        with self._mutex:
            return self._observe(observation)

    def _observe(self, observation: MarketObservation) -> CandidateState:
        """Consume one normalized observation and emit at most one material wakeup."""
        projection = self._projection(observation.candidate_id)
        self._validate_observation_identity(projection, observation)
        if self._event(observation.event_id) is not None:
            return projection.state
        if projection.state not in self._OBSERVATION_STATE:
            raise MonitorTransitionError(
                f"observations are not accepted in {projection.state.value}"
            )

        previous = projection.last_observation
        if not observation.connected:
            if previous is None or previous.connected:
                return self._append_material(
                    event_id=observation.event_id,
                    candidate_id=observation.candidate_id,
                    kind=MonitorEventKind.CONNECTION_LOST,
                    state=projection.state,
                    version=projection.version,
                    occurred_at=observation.observed_at,
                    observation=observation,
                    received_at=observation.received_at,
                )
            return self._append_observation(
                observation,
                projection,
                MonitorEventKind.OBSERVATION_ACCEPTED,
            )

        if previous is not None and not previous.connected:
            return self._append_material(
                event_id=observation.event_id,
                candidate_id=observation.candidate_id,
                kind=MonitorEventKind.CONNECTION_RESTORED,
                state=projection.state,
                version=projection.version,
                occurred_at=observation.observed_at,
                observation=observation,
                received_at=observation.received_at,
            )

        if not self._fresh(projection.candidate, observation):
            return self._append_material(
                event_id=observation.event_id,
                candidate_id=observation.candidate_id,
                kind=MonitorEventKind.FRESHNESS_FAILURE,
                state=projection.state,
                version=projection.version,
                occurred_at=observation.observed_at,
                observation=observation,
                received_at=observation.received_at,
            )

        if projection.state in self._PRE_ENTRY_STATE and self._any_rule(
            projection.candidate.invalidation_rules,
            observation,
        ):
            return self._append_material(
                event_id=observation.event_id,
                candidate_id=observation.candidate_id,
                kind=MonitorEventKind.THESIS_INVALIDATED,
                state=CandidateState.INVALIDATED,
                version=projection.version + 1,
                occurred_at=observation.observed_at,
                observation=observation,
                received_at=observation.received_at,
            )

        if projection.state == CandidateState.WAITING:
            entry_now = self._all_rules(projection.candidate.entry_rules, observation)
            entry_before = self._all_rules(projection.candidate.entry_rules, previous)
            if entry_now and not entry_before:
                return self._append_material(
                    event_id=observation.event_id,
                    candidate_id=observation.candidate.candidate_id,
                    kind=MonitorEventKind.ENTRY_TRIGGERED,
                    state=CandidateState.TRIGGERED,
                    version=projection.version + 1,
                    occurred_at=observation.observed_at,
                    observation=observation,
                    received_at=observation.received_at,
                )

        if projection.state == CandidateState.MONITORING_POSITION:
            exit_now = self._any_rule(projection.candidate.exit_rules, observation)
            exit_before = self._any_rule(projection.candidate.exit_rules, previous)
            if exit_now and not exit_before:
                return self._append_material(
                    event_id=observation.event_id,
                    candidate_id=observation.candidate.candidate_id,
                    kind=MonitorEventKind.EXIT_TRIGGERED,
                    state=CandidateState.EXIT_TRIGGERED,
                    version=projection.version + 1,
                    occurred_at=observation.observed_at,
                    observation=observation,
                    received_at=observation.received_at,
                )

        return self._append_observation(
            observation,
            projection,
            MonitorEventKind.OBSERVATION_ACCEPTED,
        )

    def _append_observation(
        self,
        observation: MarketObservation,
        projection: _Projection,
        kind: MonitorEventKind,
    ) -> CandidateState:
        self._append(
            event_id=observation.event_id,
            candidate_id=observation.candidate_id,
            kind=kind,
            source_sequence=observation.source_sequence,
            occurred_at=observation.observed_at,
            received_at=observation.received_at,
            payload=self._observation_payload(
                observation,
                projection.state,
                projection.version,
            ),
        )
        return projection.state

    def expire(self, now: datetime) -> tuple[str, ...]:
        """Expire candidates whose durable deadline has passed."""
        self._require_aware(now, "now")
        expired: list[str] = []
        for candidate_id in self._candidate_ids():
            projection = self._projection(candidate_id)
            if projection.state not in self._EXPIRABLE_STATE:
                continue
            if now <= projection.candidate.expires_at:
                continue
            event_id = f"expire:{candidate_id}"
            if self._event(event_id) is not None:
                continue
            self._append_material(
                event_id=event_id,
                candidate_id=candidate_id,
                kind=MonitorEventKind.CANDIDATE_EXPIRED,
                state=CandidateState.EXPIRED,
                version=projection.version + 1,
                occurred_at=now,
                observation_digest=content_digest(projection.candidate),
                received_at=now,
            )
            expired.append(candidate_id)
        return tuple(expired)

    def _wakeup_exists(self, wakeup_id: str, candidate_id: str) -> bool:
        return any(
            event.candidate_id == candidate_id
            and (
                event.payload.get("wakeup_id") == wakeup_id
                or (
                    isinstance(event.payload.get("wakeup"), dict)
                    and event.payload["wakeup"].get("wakeup_id") == wakeup_id
                )
            )
            for event in self._events()
        )

    def apply_research_response(self, response: ResearchResponse) -> CandidateState:
        """Apply an advisor response without allowing it to authorize execution."""
        event_id = f"research:{response.response_id}"
        prior = self._event(event_id)
        if prior is not None:
            raw_response = prior.payload.get("response")
            if raw_response != response.model_dump(mode="json"):
                raise MonitorStoreCorruption("conflicting duplicate research response")
            return self.state(response.candidate_id)

        projection = self._projection(response.candidate_id)
        if not self._wakeup_exists(response.wakeup_id, response.candidate_id):
            raise MonitorTransitionError("research response references unknown wakeup")
        if projection.state in {
            CandidateState.INVALIDATED,
            CandidateState.EXPIRED,
            CandidateState.EXECUTED,
            CandidateState.MONITORING_POSITION,
            CandidateState.EXIT_TRIGGERED,
            CandidateState.CLOSED,
        }:
            raise MonitorTransitionError("research response cannot revive terminal state")
        if projection.state != CandidateState.TRIGGERED:
            raise MonitorTransitionError("research response requires TRIGGERED state")
        if response.candidate_version != projection.version:
            raise MonitorTransitionError("research response version is stale")

        if response.decision == "REVALIDATE":
            state = CandidateState.REVALIDATING
            version = projection.version + 1
            kind = MonitorEventKind.REVALIDATION_STARTED
        elif response.decision == "INVALIDATE":
            state = CandidateState.INVALIDATED
            version = projection.version + 1
            kind = MonitorEventKind.RESEARCH_RESPONSE
        else:
            state = CandidateState.TRIGGERED
            version = projection.version
            kind = MonitorEventKind.RESEARCH_RESPONSE

        self._append(
            event_id=event_id,
            candidate_id=response.candidate_id,
            kind=kind,
            occurred_at=self._timeline_time(response.candidate_id),
            payload=self._state_payload(
                state,
                version,
                response=response.model_dump(mode="json"),
                wakeup_id=response.wakeup_id,
            ),
        )
        return state

    def revalidate(
        self,
        candidate_id: str,
        observation: MarketObservation,
        command_id: str,
    ) -> CandidateState:
        """Serialize revalidation against concurrent invalidation."""
        with self._mutex:
            return self._revalidate(candidate_id, observation, command_id)

    def _revalidate(
        self,
        candidate_id: str,
        observation: MarketObservation,
        command_id: str,
    ) -> CandidateState:
        """Require fresh current observations before producing READY."""
        event_id = f"revalidate:{command_id}"
        prior = self._event(event_id)
        if prior is not None:
            return self.state(candidate_id)

        projection = self._projection(candidate_id)
        if projection.state != CandidateState.REVALIDATING:
            raise MonitorTransitionError("candidate is not REVALIDATING")
        self._validate_observation_identity(projection, observation)
        if self._event(observation.event_id) is not None:
            raise MonitorTransitionError("revalidation observation was already consumed")
        if not observation.connected:
            self._append_material(
                event_id=observation.event_id,
                candidate_id=candidate_id,
                kind=MonitorEventKind.CONNECTION_LOST,
                state=projection.state,
                version=projection.version,
                occurred_at=observation.observed_at,
                observation=observation,
                received_at=observation.received_at,
            )
            raise MonitorTransitionError("revalidation requires a connected observation")
        if not self._fresh(projection.candidate, observation):
            self._append_material(
                event_id=observation.event_id,
                candidate_id=candidate_id,
                kind=MonitorEventKind.FRESHNESS_FAILURE,
                state=projection.state,
                version=projection.version,
                occurred_at=observation.observed_at,
                observation=observation,
                received_at=observation.received_at,
            )
            raise MonitorTransitionError("revalidation observation is stale")
        if self._any_rule(projection.candidate.invalidation_rules, observation):
            self._append_material(
                event_id=observation.event_id,
                candidate_id=candidate_id,
                kind=MonitorEventKind.THESIS_INVALIDATED,
                state=CandidateState.INVALIDATED,
                version=projection.version + 1,
                occurred_at=observation.observed_at,
                observation=observation,
                received_at=observation.received_at,
            )
            raise MonitorTransitionError("thesis invalidated during revalidation")
        if not self._all_rules(projection.candidate.entry_rules, observation):
            self._append_observation(
                observation,
                projection,
                MonitorEventKind.OBSERVATION_ACCEPTED,
            )
            raise MonitorTransitionError("fresh revalidation does not satisfy entry rules")

        self._append_observation(
            observation,
            projection,
            MonitorEventKind.OBSERVATION_ACCEPTED,
        )
        after_observation = self._projection(candidate_id)
        state = CandidateState.READY
        self._append(
            event_id=event_id,
            candidate_id=candidate_id,
            kind=MonitorEventKind.REVALIDATION_PASSED,
            occurred_at=observation.observed_at,
            received_at=observation.received_at,
            payload=self._state_payload(
                state,
                after_observation.version + 1,
                observation_digest=content_digest(observation),
                command_id=command_id,
            ),
        )
        return state

    def _external_execution_id(self, candidate_id: str) -> str | None:
        for event in reversed(self._events()):
            if event.candidate_id != candidate_id:
                continue
            raw_receipt = event.payload.get("receipt")
            if isinstance(raw_receipt, dict) and raw_receipt.get("kind") == "ENTRY_EXECUTED":
                value = raw_receipt.get("external_execution_id")
                return value if isinstance(value, str) else None
        return None

    def apply_reconciliation(self, receipt: ReconciliationReceipt) -> CandidateState:
        """Consume reconciliation truth; never submit, cancel, or write to a broker."""
        event_id = f"reconcile:{receipt.receipt_id}"
        prior = self._event(event_id)
        raw_receipt = receipt.model_dump(mode="json")
        if prior is not None:
            if prior.payload.get("receipt") != raw_receipt:
                raise MonitorStoreCorruption("conflicting duplicate reconciliation receipt")
            return self.state(receipt.candidate_id)

        projection = self._projection(receipt.candidate_id)
        current_external_id = self._external_execution_id(receipt.candidate_id)
        if receipt.kind == "ENTRY_EXECUTED":
            if projection.state != CandidateState.READY:
                raise MonitorTransitionError("only READY may receive an entry reconciliation")
            state = CandidateState.EXECUTED
            kind = MonitorEventKind.ENTRY_EXECUTED
            version = projection.version + 1
        elif receipt.kind == "POSITION_RECONCILED":
            if projection.state != CandidateState.EXECUTED:
                raise MonitorTransitionError("position reconciliation requires EXECUTED")
            if current_external_id != receipt.external_execution_id:
                raise MonitorTransitionError("position reconciliation execution id mismatch")
            state = CandidateState.MONITORING_POSITION
            kind = MonitorEventKind.POSITION_RECONCILED
            version = projection.version + 1
        else:
            if projection.state != CandidateState.EXIT_TRIGGERED:
                raise MonitorTransitionError("exit close reconciliation requires EXIT_TRIGGERED")
            if current_external_id != receipt.external_execution_id:
                raise MonitorTransitionError("exit reconciliation execution id mismatch")
            state = CandidateState.CLOSED
            kind = MonitorEventKind.EXIT_CLOSED
            version = projection.version + 1

        self._append(
            event_id=event_id,
            candidate_id=receipt.candidate_id,
            kind=kind,
            occurred_at=receipt.occurred_at,
            received_at=receipt.occurred_at,
            payload=self._state_payload(
                state,
                version,
                receipt=raw_receipt,
            ),
        )
        return state

    def process_pending_wakeups(self) -> tuple[str, ...]:
        """Deliver durable wakeups at least once and journal successful delivery."""
        if self.wakeup_sink is None:
            return ()
        delivered: list[str] = []
        for event in self.store.pending_wakeups():
            raw_wakeup = event.payload.get("wakeup")
            if not isinstance(raw_wakeup, dict):
                raise MonitorStoreCorruption("material event has malformed wakeup")
            wakeup = ResearchWakeup.model_validate(raw_wakeup)
            try:
                self.wakeup_sink.wake(wakeup)
            except Exception:
                continue
            delivery_id = f"delivery:{wakeup.wakeup_id}"
            if self._event(delivery_id) is None:
                self._append(
                    event_id=delivery_id,
                    candidate_id=wakeup.candidate_id,
                    kind=MonitorEventKind.RESEARCH_WAKEUP_DELIVERED,
                    occurred_at=event.occurred_at,
                    received_at=self.store.latest_received_at() or event.occurred_at,
                    payload={
                        "wakeup_id": wakeup.wakeup_id,
                        "material_event_id": wakeup.material_event_id,
                    },
                )
            delivered.append(wakeup.wakeup_id)
        return tuple(delivered)


__all__ = [
    "AxiomMonitor",
    "MonitorTransitionError",
    "ResearchWakeupSink",
]
