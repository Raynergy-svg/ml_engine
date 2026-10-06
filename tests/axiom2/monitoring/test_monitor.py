from datetime import datetime, timedelta, timezone

import pytest

from src.axiom2.monitoring.contracts import (
    CandidateRegistration,
    CandidateState,
    MarketObservation,
    ReconciliationReceipt,
    ResearchResponse,
    ThresholdRule,
)
from src.axiom2.monitoring.monitor import AxiomMonitor, MonitorTransitionError
from src.axiom2.monitoring.store import MonitorStore


NOW = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)


class Sink:
    def __init__(self):
        self.wakeups = []
        self.fail = False

    def wake(self, wakeup):
        if self.fail:
            raise RuntimeError("advisor unavailable")
        self.wakeups.append(wakeup)


def candidate(*, expires_after=timedelta(hours=2)):
    return CandidateRegistration(
        candidate_id="candidate-spy-1",
        instrument_id="SPY",
        thesis_version="thesis-v1",
        thesis_digest="a" * 64,
        registered_at=NOW,
        expires_at=NOW + expires_after,
        freshness_seconds=30,
        entry_rules=(
            ThresholdRule(
                metric="PRICE",
                operator="AT_OR_ABOVE",
                threshold=500_000_000,
                role="ENTRY",
            ),
            ThresholdRule(
                metric="VOLUME",
                operator="AT_OR_ABOVE",
                threshold=10_000,
                role="ENTRY",
            ),
        ),
        invalidation_rules=(
            ThresholdRule(
                metric="PRICE",
                operator="AT_OR_BELOW",
                threshold=495_000_000,
                role="INVALIDATION",
            ),
        ),
        exit_rules=(
            ThresholdRule(
                metric="PRICE",
                operator="AT_OR_BELOW",
                threshold=498_000_000,
                role="EXIT",
            ),
        ),
    )


def observation(
    event_id,
    sequence,
    *,
    price=499_000_000,
    volume=9_000,
    observed_at=NOW,
    connected=True,
):
    return MarketObservation(
        candidate_id="candidate-spy-1",
        instrument_id="SPY",
        event_id=event_id,
        source_sequence=sequence,
        observed_at=observed_at,
        received_at=NOW if observed_at <= NOW else observed_at,
        connected=connected,
        price_micros=price,
        volume=volume,
    )


def waiting_monitor(tmp_path, sink=None, *, expires_after=timedelta(hours=2)):
    monitor = AxiomMonitor(MonitorStore(tmp_path), wakeup_sink=sink)
    monitor.register(candidate(expires_after=expires_after), "register-1")
    monitor.start_watching("candidate-spy-1", "watch-1")
    monitor.wait("candidate-spy-1", "wait-1")
    return monitor


def trigger(monitor, sequence=2):
    return monitor.observe(
        observation(
            f"trigger-{sequence}",
            sequence,
            price=500_000_001,
            volume=10_001,
        )
    )


def test_registration_watching_wait_is_durable_across_restart(tmp_path):
    monitor = waiting_monitor(tmp_path)

    assert monitor.state("candidate-spy-1") == CandidateState.WAITING
    restarted = AxiomMonitor(MonitorStore(tmp_path))

    assert restarted.state("candidate-spy-1") == CandidateState.WAITING


def test_price_and_volume_crossing_triggers_once(tmp_path):
    sink = Sink()
    monitor = waiting_monitor(tmp_path, sink)

    assert monitor.observe(observation("quiet-1", 1)).value == "WAITING"
    assert trigger(monitor).value == "TRIGGERED"
    assert len(sink.wakeups) == 1

    assert monitor.observe(observation("trigger-2", 2, price=500_000_002, volume=10_002)).value == "TRIGGERED"
    assert len(sink.wakeups) == 1


def test_stale_observation_is_rejected_and_fresh_observation_can_trigger(tmp_path):
    sink = Sink()
    monitor = waiting_monitor(tmp_path, sink)

    stale = observation(
        "stale-1",
        1,
        price=500_000_001,
        volume=10_001,
        observed_at=NOW - timedelta(seconds=31),
    )
    assert monitor.observe(stale) == CandidateState.WAITING
    assert len(sink.wakeups) == 1
    assert sink.wakeups[0].reason.value == "FRESHNESS_FAILURE"

    fresh = observation("fresh-2", 2, price=500_000_001, volume=10_001)
    assert monitor.observe(fresh) == CandidateState.TRIGGERED
    assert len(sink.wakeups) == 2


def test_trigger_then_immediate_thesis_invalidation_never_becomes_ready(tmp_path):
    sink = Sink()
    monitor = waiting_monitor(tmp_path, sink)
    trigger(monitor)

    assert monitor.observe(
        observation("invalidate-3", 3, price=494_999_999, volume=11_000)
    ) == CandidateState.INVALIDATED

    response = ResearchResponse(
        response_id="response-1",
        wakeup_id=sink.wakeups[0].wakeup_id,
        candidate_id="candidate-spy-1",
        candidate_version=1,
        decision="REVALIDATE",
    )
    with pytest.raises(MonitorTransitionError):
        monitor.apply_research_response(response)
    assert monitor.state("candidate-spy-1") == CandidateState.INVALIDATED


def test_invalidation_during_revalidation_blocks_ready(tmp_path):
    sink = Sink()
    monitor = waiting_monitor(tmp_path, sink)
    trigger(monitor)
    response = ResearchResponse(
        response_id="response-1",
        wakeup_id=sink.wakeups[0].wakeup_id,
        candidate_id="candidate-spy-1",
        candidate_version=1,
        decision="REVALIDATE",
    )

    assert monitor.apply_research_response(response) == CandidateState.REVALIDATING
    assert monitor.observe(
        observation("invalidate-3", 3, price=494_999_999, volume=11_000)
    ) == CandidateState.INVALIDATED
    with pytest.raises(MonitorTransitionError):
        monitor.revalidate(
            "candidate-spy-1",
            observation("revalidate-4", 4, price=500_000_001, volume=11_000),
            "revalidation-1",
        )
    assert monitor.state("candidate-spy-1") == CandidateState.INVALIDATED


def test_disconnect_waits_and_reconnect_does_not_duplicate_trigger(tmp_path):
    sink = Sink()
    monitor = waiting_monitor(tmp_path, sink)

    assert monitor.observe(
        observation("disconnect-1", 1, connected=False, price=None, volume=None)
    ) == CandidateState.WAITING
    assert sink.wakeups[0].reason.value == "CONNECTION_LOST"

    assert monitor.observe(observation("reconnect-2", 2)) == CandidateState.WAITING
    assert sink.wakeups[1].reason.value == "CONNECTION_RESTORED"
    assert monitor.observe(observation("entry-3", 3, price=500_000_001, volume=10_001)) == CandidateState.TRIGGERED
    assert len([w for w in sink.wakeups if w.reason.value == "ENTRY_TRIGGERED"]) == 1


def test_candidate_expires_without_trigger(tmp_path):
    monitor = waiting_monitor(tmp_path, expires_after=timedelta(seconds=10))

    assert monitor.expire(NOW + timedelta(seconds=10, microseconds=1)) == ("candidate-spy-1",)
    assert monitor.state("candidate-spy-1") == CandidateState.EXPIRED


def test_reconciliation_drives_post_entry_and_exit_lifecycle(tmp_path):
    sink = Sink()
    monitor = waiting_monitor(tmp_path, sink)
    trigger(monitor)
    response = ResearchResponse(
        response_id="response-1",
        wakeup_id=sink.wakeups[0].wakeup_id,
        candidate_id="candidate-spy-1",
        candidate_version=1,
        decision="REVALIDATE",
    )
    monitor.apply_research_response(response)
    assert monitor.revalidate(
        "candidate-spy-1",
        observation("revalidate-3", 3, price=500_000_002, volume=11_000),
        "revalidation-1",
    ) == CandidateState.READY

    assert monitor.apply_reconciliation(
        ReconciliationReceipt(
            receipt_id="entry-receipt",
            candidate_id="candidate-spy-1",
            kind="ENTRY_EXECUTED",
            external_execution_id="external-entry-1",
            occurred_at=NOW,
        )
    ) == CandidateState.EXECUTED
    assert monitor.apply_reconciliation(
        ReconciliationReceipt(
            receipt_id="position-receipt",
            candidate_id="candidate-spy-1",
            kind="POSITION_RECONCILED",
            external_execution_id="external-entry-1",
            occurred_at=NOW + timedelta(seconds=1),
        )
    ) == CandidateState.MONITORING_POSITION
    assert monitor.observe(
        observation("exit-4", 4, price=497_999_999, volume=11_000)
    ) == CandidateState.EXIT_TRIGGERED
    assert monitor.apply_reconciliation(
        ReconciliationReceipt(
            receipt_id="close-receipt",
            candidate_id="candidate-spy-1",
            kind="EXIT_CLOSED",
            external_execution_id="external-entry-1",
            occurred_at=NOW + timedelta(seconds=2),
        )
    ) == CandidateState.CLOSED
