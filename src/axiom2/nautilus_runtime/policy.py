"""Axiom-owned candidate policy and state transition rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import (
    CandidateObservation,
    CandidateState,
    MaterialEvent,
    ResearchResult,
    ResearchWakeup,
    TERMINAL_STATES,
    stable_id,
)


ALLOWED_TRANSITIONS: dict[CandidateState | None, frozenset[CandidateState]] = {
    None: frozenset({CandidateState.WATCHING}),
    CandidateState.WATCHING: frozenset(
        {
            CandidateState.WAITING,
            CandidateState.TRIGGERED,
            CandidateState.INVALIDATED,
            CandidateState.EXPIRED,
        }
    ),
    CandidateState.WAITING: frozenset(
        {
            CandidateState.TRIGGERED,
            CandidateState.INVALIDATED,
            CandidateState.EXPIRED,
        }
    ),
    CandidateState.TRIGGERED: frozenset(
        {
            CandidateState.REVALIDATING,
            CandidateState.INVALIDATED,
            CandidateState.EXPIRED,
        }
    ),
    CandidateState.REVALIDATING: frozenset(
        {CandidateState.READY, CandidateState.INVALIDATED, CandidateState.EXPIRED}
    ),
    CandidateState.READY: frozenset(
        {CandidateState.INVALIDATED, CandidateState.EXPIRED}
    ),
    CandidateState.INVALIDATED: frozenset(),
    CandidateState.EXPIRED: frozenset(),
}


def transition_allowed(
    previous: CandidateState | None, new: CandidateState
) -> bool:
    return new in ALLOWED_TRANSITIONS[previous]


def observation_state_path(
    previous: CandidateState | None, observation: CandidateObservation
) -> tuple[CandidateState, ...]:
    if previous in TERMINAL_STATES:
        return ()
    if observation.invalidated:
        first = CandidateState.WATCHING if previous is None else previous
        return (
            (first, CandidateState.INVALIDATED)
            if previous is None
            else (CandidateState.INVALIDATED,)
        )
    if observation.observed_at_ns >= observation.freshness_deadline_ns:
        first = CandidateState.WATCHING if previous is None else previous
        return (
            (first, CandidateState.EXPIRED)
            if previous is None
            else (CandidateState.EXPIRED,)
        )
    if previous is None:
        return (
            CandidateState.WATCHING,
            CandidateState.TRIGGERED
            if observation.confirmation
            else CandidateState.WAITING,
        )
    if previous is CandidateState.WATCHING:
        return (
            CandidateState.TRIGGERED
            if observation.confirmation
            else CandidateState.WAITING,
        )
    if previous is CandidateState.WAITING and observation.confirmation:
        return (CandidateState.TRIGGERED,)
    return ()


@dataclass(frozen=True, slots=True)
class ObservationDecision:
    state_path: tuple[CandidateState, ...]
    material_event: MaterialEvent | None
    wakeup: ResearchWakeup | None


class CandidateMonitor:
    """Apply Axiom candidate policy to durable admitted observations.

    This class has no order, account, capital, credential, or submission API.
    """

    def __init__(self, journal: Any) -> None:
        self.journal = journal

    def observe(self, observation: CandidateObservation) -> MaterialEvent | None:
        snapshot = self.journal.snapshot(
            observation.candidate_id, observation.candidate_version
        )
        previous = snapshot.state if snapshot is not None else None
        if snapshot is not None and observation.observed_at_ns < snapshot.updated_at_ns:
            raise ValueError("observation timestamp regressed")
        path = observation_state_path(previous, observation)
        if path and any(
            not transition_allowed(
                previous if index == 0 else path[index - 1], state
            )
            for index, state in enumerate(path)
        ):
            raise ValueError("candidate transition is not allowed")

        material_event = None
        wakeup = None
        if CandidateState.TRIGGERED in path and previous is not CandidateState.TRIGGERED:
            event_id = stable_id(
                "material",
                observation.candidate_id,
                observation.candidate_version,
                observation.observation_id,
            )
            material_event = MaterialEvent(
                event_id=event_id,
                candidate_id=observation.candidate_id,
                candidate_version=observation.candidate_version,
                source_observation_id=observation.observation_id,
                kind="CANDIDATE_TRIGGERED",
                occurred_at_ns=observation.observed_at_ns,
                evidence_digest=observation.evidence_digest,
            )
            wakeup = ResearchWakeup(
                wakeup_id=stable_id("research-wakeup", event_id),
                material_event_id=event_id,
                candidate_id=observation.candidate_id,
                candidate_version=observation.candidate_version,
                created_at_ns=observation.observed_at_ns,
                bound_evidence_digest=observation.evidence_digest,
            )
        result = self.journal.append_observation(
            observation,
            state_path=path,
            material_event=material_event,
            wakeup=wakeup,
        )
        return result.material_event

    def state(
        self, candidate_id: str, candidate_version: str
    ) -> CandidateState | None:
        snapshot = self.journal.snapshot(candidate_id, candidate_version)
        return snapshot.state if snapshot is not None else None

    def revalidate(self, result: ResearchResult) -> Any:
        return self.journal.consume_research_wakeup(result)
