from __future__ import annotations

import hashlib
from pathlib import Path

from axiom2.nautilus_runtime import (
    CandidateJournal,
    CandidateMonitor,
    CandidateObservation,
    CandidateState,
    NautilusReplayRuntime,
    ResearchResult,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def observation(
    observation_id: str,
    *,
    observed_at_ns: int,
    confirmation: bool = False,
    invalidated: bool = False,
    deadline_ns: int = 100,
) -> CandidateObservation:
    return CandidateObservation(
        candidate_id="runtime-candidate",
        candidate_version="v1",
        observation_id=observation_id,
        observed_at_ns=observed_at_ns,
        freshness_deadline_ns=deadline_ns,
        evidence_digest=digest(f"evidence:{observation_id}"),
        raw_observation_digest=digest(f"raw:{observation_id}"),
        confirmation=confirmation,
        invalidated=invalidated,
        invalidation_reason="conditions deteriorated" if invalidated else None,
        facts={"source": "deterministic-replay"},
    )


def test_replay_runtime_no_entry_deteriorates_without_wakeup(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "no-entry.sqlite")
    monitor = CandidateMonitor(journal)
    runtime = NautilusReplayRuntime(monitor, journal)

    runtime.start()
    assert runtime.is_running
    assert runtime.publish(observation("watch", observed_at_ns=1)) is None
    assert monitor.state("runtime-candidate", "v1") is CandidateState.WAITING

    assert runtime.publish(
        observation("deteriorated", observed_at_ns=2, invalidated=True)
    ) is None
    assert monitor.state("runtime-candidate", "v1") is CandidateState.INVALIDATED
    assert journal.pending_wakeups() == ()

    runtime.stop()
    runtime.dispose()
    journal.close()


def test_replay_runtime_qualified_trigger_restarts_and_reaches_ready(
    tmp_path: Path,
) -> None:
    database = tmp_path / "qualified.sqlite"
    journal = CandidateJournal(database)
    monitor = CandidateMonitor(journal)
    runtime = NautilusReplayRuntime(monitor, journal)

    runtime.start()
    assert runtime.publish(observation("watch", observed_at_ns=1)) is None
    material = runtime.publish(
        observation("confirmed", observed_at_ns=2, confirmation=True)
    )
    assert material is not None
    assert monitor.state("runtime-candidate", "v1") is CandidateState.TRIGGERED
    runtime.set_time(2)
    runtime.stop()
    runtime.dispose()
    journal.close()

    restarted_journal = CandidateJournal(database)
    restarted_monitor = CandidateMonitor(restarted_journal)
    pending = restarted_journal.pending_wakeups()
    assert len(pending) == 1
    result = ResearchResult(
        wakeup_id=pending[0].wakeup_id,
        candidate_id="runtime-candidate",
        candidate_version="v1",
        completed_at_ns=3,
        fresh_until_ns=50,
        evidence_digest=digest("research:confirmed"),
        qualifies=True,
        facts={"source": "bounded-research"},
    )
    outcome = restarted_monitor.revalidate(result)
    assert outcome.outcome == CandidateState.READY.value
    assert restarted_monitor.state("runtime-candidate", "v1") is CandidateState.READY
    assert restarted_monitor.revalidate(result).duplicate
    assert not hasattr(runtime, "submit_order")
    restarted_journal.close()


def test_replay_runtime_stale_result_cannot_resurrect_invalidated_version(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stale.sqlite"
    journal = CandidateJournal(database)
    monitor = CandidateMonitor(journal)
    runtime = NautilusReplayRuntime(monitor, journal)

    runtime.start()
    runtime.publish(observation("watch", observed_at_ns=1))
    material = runtime.publish(
        observation("confirmed", observed_at_ns=2, confirmation=True)
    )
    assert material is not None
    assert runtime.publish(
        observation("invalidated", observed_at_ns=3, invalidated=True)
    ) is None
    assert monitor.state("runtime-candidate", "v1") is CandidateState.INVALIDATED
    wakeup = journal.pending_wakeups()[0]
    stale_result = ResearchResult(
        wakeup_id=wakeup.wakeup_id,
        candidate_id="runtime-candidate",
        candidate_version="v1",
        completed_at_ns=4,
        fresh_until_ns=50,
        evidence_digest=digest("research:late"),
        qualifies=True,
    )
    assert monitor.revalidate(stale_result).outcome == "STALE"
    assert monitor.state("runtime-candidate", "v1") is CandidateState.INVALIDATED

    runtime.dispose()
    journal.close()
