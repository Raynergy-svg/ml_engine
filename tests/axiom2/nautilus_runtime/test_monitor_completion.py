from dataclasses import replace

import pytest

from src.axiom2.nautilus_runtime import CandidateJournal, CandidateMonitor, CandidateObservation, CandidateState, ResearchResult


def observation(identity='trigger', time=2, version='v1', deadline=100):
    return CandidateObservation('c', version, identity, time, deadline, 'a'*64, 'b'*64, confirmation=True)


def result(journal, **changes):
    wakeup = journal.pending_wakeups()[0]
    return ResearchResult(wakeup.wakeup_id, 'c', wakeup.candidate_version, 3, 5, 'c'*64, True, **changes)


def test_ready_expires_at_research_deadline_after_restart(tmp_path):
    path = tmp_path/'monitor.db'
    journal = CandidateJournal(path)
    monitor = CandidateMonitor(journal)
    monitor.observe(observation())
    monitor.revalidate(result(journal))
    journal.close()
    journal = CandidateJournal(path)
    journal.advance_time(5)
    assert CandidateMonitor(journal).state('c', 'v1') is CandidateState.EXPIRED


def test_superseded_version_cannot_resurrect_after_restart(tmp_path):
    path = tmp_path/'monitor.db'
    journal = CandidateJournal(path)
    monitor = CandidateMonitor(journal)
    monitor.observe(observation())
    old = result(journal)
    monitor.observe(observation('v2-trigger', 3, 'v2'))
    journal.close()
    journal = CandidateJournal(path)
    assert CandidateMonitor(journal).revalidate(old).outcome == 'STALE'
    assert CandidateMonitor(journal).state('c', 'v1') is CandidateState.INVALIDATED


def test_old_duplicate_remains_idempotent_after_newer_observation(tmp_path):
    journal = CandidateJournal(tmp_path/'monitor.db')
    monitor = CandidateMonitor(journal)
    first = observation()
    monitor.observe(first)
    monitor.observe(observation('later', 3))
    assert monitor.observe(first) is not None
    assert len(journal.pending_wakeups()) == 1


def test_conflicting_result_is_rejected_durably(tmp_path):
    journal = CandidateJournal(tmp_path/'monitor.db')
    monitor = CandidateMonitor(journal)
    monitor.observe(observation())
    research = result(journal)
    monitor.revalidate(research)
    with pytest.raises(Exception, match='result.*conflict'):
        monitor.revalidate(replace(research, qualifies=False))
    assert monitor.state('c', 'v1') is CandidateState.READY


def test_wrong_version_result_does_not_claim_valid_wakeup(tmp_path):
    journal = CandidateJournal(tmp_path/'monitor.db')
    monitor = CandidateMonitor(journal)
    monitor.observe(observation())
    with pytest.raises(Exception, match='version mismatch'):
        monitor.revalidate(replace(result(journal), candidate_version='wrong'))
    assert monitor.state('c', 'v1') is CandidateState.TRIGGERED


def test_changed_trigger_evidence_cannot_be_revalidated_by_old_wakeup(tmp_path):
    journal = CandidateJournal(tmp_path/'monitor.db')
    monitor = CandidateMonitor(journal)
    monitor.observe(observation())
    research = result(journal)
    monitor.observe(replace(observation('changed', 3), evidence_digest='d'*64))
    assert monitor.revalidate(research).outcome == 'STALE'
    assert monitor.state('c', 'v1') is CandidateState.INVALIDATED


def test_immutable_nested_observation_facts():
    facts = {'nested': {'values': [1]}}
    admitted = replace(observation(), facts=facts)
    facts['nested']['values'].append(2)
    assert admitted.to_payload()['facts']['nested']['values'] == (1,)
