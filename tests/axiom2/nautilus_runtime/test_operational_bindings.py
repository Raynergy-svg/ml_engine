"""Offline composition contracts for existing interfaces; installs no binding."""
from io import StringIO
import hashlib
import json

import pytest

from axiom2.nautilus_runtime import CandidateJournal, CandidateMonitor, CandidateObservation, CandidateState, ResearchResult
from axiom2.nautilus_runtime.wakeup import ResearchWakeupConsumer
from dashboard.server.training_jobs import LocalTrainingJobRunner, TrainingJobJournal
from dashboard.server.training_cockpit import read_jobs
from src.axiom2.execution.authority import ExecutionAuthority
from src.axiom2.research.development import serve_development_stdio


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize('action', ['observe', 'runtime_status', 'restart'])
def test_existing_research_transport_cannot_be_repurposed_as_native_control(action):
    reader = StringIO(json.dumps({'action': action, 'snapshot_id': 'configured'}) + '\n')
    writer = StringIO()
    # No registry/signing context is touched: unsupported actions deny first.
    serve_development_stdio(reader, writer, registry=object(), bundles={'configured': object()}, code_commit='verified')
    result = json.loads(writer.getvalue())
    assert result['status'] == 'REJECTED'
    assert 'unsupported bounded development action' in result['reason']
    assert result['execution_enabled'] is False
    assert result['holdout_accessed'] is False


def test_existing_finite_worker_consumes_outbox_without_gateway_authority(tmp_path):
    journal = CandidateJournal(tmp_path / 'candidate.sqlite')
    jobs = TrainingJobJournal(tmp_path / 'jobs')
    runner = LocalTrainingJobRunner(jobs)
    try:
        observation = CandidateObservation('candidate', 'v1', 'receipt', 1, 100,
            digest('already-admitted-evidence'), digest('retained-source'), confirmation=True)
        # Serialization is transport shape only; this test grants no source admission.
        copied = CandidateObservation(**json.loads(json.dumps(observation.to_payload())))
        CandidateMonitor(journal).observe(copied)
        consumer = ResearchWakeupConsumer(journal)
        wakeup, = consumer.pending()
        assert consumer.claim(wakeup.wakeup_id).duplicate is False
        jobs.transition(wakeup.wakeup_id, 'submitted')
        # Already-computed fixture result; no model/provider/inference invocation.
        result = ResearchResult(wakeup.wakeup_id, 'candidate', 'v1', 2, 90,
            digest('verified-result-fixture'), True)
        future = runner.submit(wakeup.wakeup_id, lambda: {'result': result.to_payload()})
        delivered = future.result(timeout=3)
        assert consumer.complete(ResearchResult(**delivered['result'])).duplicate is False
        assert journal.snapshot('candidate', 'v1').state is CandidateState.READY
        assert consumer.complete(result).duplicate is True
        assert read_jobs({}, jobs.root)['counts']['completed'] == 1
        # Monitoring READY never becomes a review, submit or cancellation grant.
        authority = ExecutionAuthority()
        for decision in (authority.review(delivered, None), authority.submit(delivered, None), authority.cancel(delivered, None)):
            assert decision.status == 'BLOCKED'
            assert decision.execution_enabled is False
            assert decision.capital_authorized is False
    finally:
        runner.shutdown()
        journal.close()


def test_reopened_job_projection_is_not_a_runtime_heartbeat(tmp_path):
    jobs = TrainingJobJournal(tmp_path / 'jobs')
    jobs.transition('interrupted', 'submitted')
    jobs.transition('interrupted', 'running')
    # Durable readback alone cannot tell whether its original process survived.
    reopened = TrainingJobJournal(jobs.root)
    row, = read_jobs({}, reopened.root)['jobs']
    assert row['status'] == 'running'
    assert not {'heartbeat_ns', 'lease_until_ns', 'generation', 'native_handle_running'} & row.keys()
    # An operational adapter must obtain native health separately, never infer it.
    assert read_jobs({}, reopened.root)['counts']['running'] == 1
