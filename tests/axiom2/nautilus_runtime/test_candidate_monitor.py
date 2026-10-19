from __future__ import annotations

import hashlib

import pytest

from src.axiom2.nautilus_runtime import (
    CandidateJournal,
    CandidateMonitor,
    CandidateObservation,
    CandidateState,
    ResearchResult,
    ResearchWakeupConsumer,
)
from src.axiom2.nautilus_runtime.policy import ALLOWED_TRANSITIONS


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def observation(
    observation_id: str,
    *,
    observed_at_ns: int,
    deadline_ns: int = 100,
    confirmation: bool = False,
    invalidated: bool = False,
    reason: str | None = None,
    version: str = "v1",
) -> CandidateObservation:
    return CandidateObservation(
        candidate_id="candidate-1",
        candidate_version=version,
        observation_id=observation_id,
        observed_at_ns=observed_at_ns,
        freshness_deadline_ns=deadline_ns,
        evidence_digest=digest(f"evidence-{observation_id}"),
        raw_observation_digest=digest(f"raw-{observation_id}"),
        confirmation=confirmation,
        invalidated=invalidated,
        invalidation_reason=reason,
    )


def test_waiting_candidate_invalidates_and_keeps_no_wakeup(tmp_path):
    journal = CandidateJournal(tmp_path / "candidate.sqlite")
    monitor = CandidateMonitor(journal)

    assert monitor.observe(observation("watch", observed_at_ns=1)) is None
    assert monitor.state("candidate-1", "v1") is CandidateState.WAITING

    assert (
        monitor.observe(
            observation(
                "deteriorated",
                observed_at_ns=2,
                invalidated=True,
                reason="confirmation window closed",
            )
        )
        is None
    )
    assert monitor.state("candidate-1", "v1") is CandidateState.INVALIDATED
    assert journal.pending_wakeups() == ()
    assert [row["new_state"] for row in journal.state_history("candidate-1", "v1")] == [
        "WATCHING",
        "WAITING",
        "INVALIDATED",
    ]
    journal.close()


def test_qualified_trigger_revalidates_to_ready_without_order_surface(tmp_path):
    journal = CandidateJournal(tmp_path / "candidate.sqlite")
    monitor = CandidateMonitor(journal)
    consumer = ResearchWakeupConsumer(journal)

    monitor.observe(observation("watch", observed_at_ns=1))
    material = monitor.observe(
        observation("confirmed", observed_at_ns=2, confirmation=True)
    )
    assert material is not None
    assert monitor.state("candidate-1", "v1") is CandidateState.TRIGGERED
    pending = consumer.pending()
    assert len(pending) == 1

    result = ResearchResult(
        wakeup_id=pending[0].wakeup_id,
        candidate_id="candidate-1",
        candidate_version="v1",
        completed_at_ns=3,
        fresh_until_ns=50,
        evidence_digest=digest("fresh-research"),
        qualifies=True,
    )
    outcome = consumer.consume(result)
    assert outcome.outcome == "READY"
    assert monitor.state("candidate-1", "v1") is CandidateState.READY
    assert [row["new_state"] for row in journal.state_history("candidate-1", "v1")] == [
        "WATCHING",
        "WAITING",
        "TRIGGERED",
        "REVALIDATING",
        "READY",
    ]
    assert not hasattr(monitor, "submit_order")
    assert not hasattr(monitor, "authorize_order")
    assert journal.pending_wakeups() == ()
    journal.close()


def test_invalidated_candidate_cannot_be_resurrected_by_late_research(tmp_path):
    journal = CandidateJournal(tmp_path / "candidate.sqlite")
    monitor = CandidateMonitor(journal)
    consumer = ResearchWakeupConsumer(journal)

    monitor.observe(observation("watch", observed_at_ns=1))
    material = monitor.observe(
        observation("confirmed", observed_at_ns=2, confirmation=True)
    )
    assert material is not None
    wakeup = consumer.pending()[0]

    monitor.observe(
        observation(
            "deteriorated",
            observed_at_ns=3,
            invalidated=True,
            reason="conditions deteriorated",
        )
    )
    assert monitor.state("candidate-1", "v1") is CandidateState.INVALIDATED

    outcome = consumer.consume(
        ResearchResult(
            wakeup_id=wakeup.wakeup_id,
            candidate_id="candidate-1",
            candidate_version="v1",
            completed_at_ns=4,
            fresh_until_ns=50,
            evidence_digest=digest("late"),
            qualifies=True,
        )
    )
    assert outcome.outcome == "STALE"
    assert monitor.state("candidate-1", "v1") is CandidateState.INVALIDATED
    assert journal.pending_wakeups() == ()
    journal.close()


def test_recovery_keeps_pending_wakeup_and_idempotent_consumption(tmp_path):
    path = tmp_path / "candidate.sqlite"
    journal = CandidateJournal(path)
    monitor = CandidateMonitor(journal)
    monitor.observe(observation("watch", observed_at_ns=1))
    monitor.observe(observation("confirmed", observed_at_ns=2, confirmation=True))
    wakeup = CandidateMonitor(journal).journal.pending_wakeups()[0]
    journal.close()

    restarted = CandidateJournal(path)
    consumer = ResearchWakeupConsumer(restarted)
    assert [item.wakeup_id for item in consumer.pending()] == [wakeup.wakeup_id]

    claimed = consumer.claim(wakeup.wakeup_id)
    assert claimed.outcome == "CLAIMED"
    restarted.close()

    after_claim_restart = CandidateJournal(path)
    recovered = ResearchWakeupConsumer(after_claim_restart)
    assert [item.wakeup_id for item in recovered.pending()] == [wakeup.wakeup_id]
    result = ResearchResult(
        wakeup_id=wakeup.wakeup_id,
        candidate_id="candidate-1",
        candidate_version="v1",
        completed_at_ns=3,
        fresh_until_ns=50,
        evidence_digest=digest("fresh"),
        qualifies=True,
    )
    assert recovered.complete(result).outcome == "READY"
    assert recovered.complete(result).duplicate is True
    after_claim_restart.close()


def test_duplicate_and_conflicting_observation_ids_are_durable(tmp_path):
    journal = CandidateJournal(tmp_path / "candidate.sqlite")
    monitor = CandidateMonitor(journal)
    first = observation("same", observed_at_ns=1)
    assert monitor.observe(first) is None
    assert monitor.observe(first) is not None or monitor.state("candidate-1", "v1") is CandidateState.WAITING

    conflicting = CandidateObservation(
        candidate_id="candidate-1",
        candidate_version="v1",
        observation_id="same",
        observed_at_ns=1,
        freshness_deadline_ns=100,
        evidence_digest=digest("different"),
        raw_observation_digest=digest("different-raw"),
    )
    with pytest.raises(Exception, match="observation id conflict"):
        monitor.observe(conflicting)
    journal.close()


@pytest.mark.parametrize(
    "state",
    [
        CandidateState.WATCHING,
        CandidateState.WAITING,
        CandidateState.TRIGGERED,
        CandidateState.REVALIDATING,
        CandidateState.READY,
    ],
)
def test_invalidation_is_allowed_from_every_nonterminal_state(state):
    assert CandidateState.INVALIDATED in ALLOWED_TRANSITIONS[state]


def test_expiration_is_terminal_from_waiting(tmp_path):
    journal = CandidateJournal(tmp_path / "candidate.sqlite")
    monitor = CandidateMonitor(journal)
    monitor.observe(observation("watch", observed_at_ns=1, deadline_ns=5))
    monitor.observe(observation("expired", observed_at_ns=5, deadline_ns=5))
    assert monitor.state("candidate-1", "v1") is CandidateState.EXPIRED
    journal.close()
