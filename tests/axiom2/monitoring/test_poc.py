from datetime import datetime, timedelta, timezone

from src.axiom2.monitoring.contracts import (
    CandidateRegistration,
    CandidateState,
    MarketObservation,
    ResearchResponse,
    ThresholdRule,
)
from src.axiom2.monitoring.monitor import AxiomMonitor
from src.axiom2.monitoring.store import MonitorStore


NOW = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)


class Sink:
    def __init__(self):
        self.wakeups = []

    def wake(self, wakeup):
        self.wakeups.append(wakeup)


def candidate():
    return CandidateRegistration(
        candidate_id="candidate-poc",
        instrument_id="SPY",
        thesis_version="bullish-v1",
        thesis_digest="d" * 64,
        registered_at=NOW,
        expires_at=NOW + timedelta(hours=1),
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
    )


def observation(event_id, sequence, *, price, volume):
    point = NOW + timedelta(seconds=sequence)
    return MarketObservation(
        candidate_id="candidate-poc",
        instrument_id="SPY",
        event_id=event_id,
        source_sequence=sequence,
        observed_at=point,
        received_at=point,
        connected=True,
        price_micros=price,
        volume=volume,
    )


def waiting(tmp_path, sink):
    monitor = AxiomMonitor(MonitorStore(tmp_path), wakeup_sink=sink)
    monitor.register(candidate(), "register")
    monitor.start_watching("candidate-poc", "watch")
    monitor.wait("candidate-poc", "wait")
    return monitor


def test_poc_waits_then_invalidates_deteriorating_bullish_candidate(tmp_path):
    sink = Sink()
    monitor = waiting(tmp_path, sink)

    transitions = [
        monitor.state("candidate-poc").value,
        monitor.observe(
            observation("quiet", 1, price=499_000_000, volume=9_000)
        ).value,
        monitor.observe(
            observation("deterioration", 2, price=494_999_999, volume=11_000)
        ).value,
    ]

    assert transitions == ["WAITING", "WAITING", "INVALIDATED"]
    assert monitor.state("candidate-poc") != CandidateState.READY
    assert not [
        row
        for row in MonitorStore(tmp_path).replay()
        if row.kind.value == "ENTRY_EXECUTED"
    ]


def test_poc_positive_path_stops_at_ready_without_order(tmp_path):
    sink = Sink()
    monitor = waiting(tmp_path, sink)

    triggered = monitor.observe(
        observation("trigger", 1, price=500_000_001, volume=10_001)
    )
    response = ResearchResponse(
        response_id="research",
        wakeup_id=sink.wakeups[0].wakeup_id,
        candidate_id="candidate-poc",
        candidate_version=1,
        decision="REVALIDATE",
    )
    revalidating = monitor.apply_research_response(response)
    ready = monitor.revalidate(
        "candidate-poc",
        observation("fresh-revalidation", 2, price=500_000_002, volume=11_000),
        "revalidate",
    )

    assert [triggered.value, revalidating.value, ready.value] == [
        "TRIGGERED",
        "REVALIDATING",
        "READY",
    ]
    assert not [
        row
        for row in MonitorStore(tmp_path).replay()
        if row.kind.value == "ENTRY_EXECUTED"
    ]
