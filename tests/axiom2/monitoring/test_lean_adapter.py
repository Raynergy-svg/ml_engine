from datetime import datetime, timedelta, timezone

import pytest

from src.axiom2.monitoring.contracts import (
    CandidateRegistration,
    CandidateState,
    ThresholdRule,
)
from src.axiom2.monitoring.lean_adapter import (
    LeanMarketEvent,
    LeanObservationAdapter,
)
from src.axiom2.monitoring.monitor import AxiomMonitor
from src.axiom2.monitoring.store import MonitorStore


NOW = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)


def candidate():
    return CandidateRegistration(
        candidate_id="candidate-lean-1",
        instrument_id="SPY",
        thesis_version="thesis-v1",
        thesis_digest="b" * 64,
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
    )


def test_lean_event_is_normalized_and_delivered_to_monitor(tmp_path):
    monitor = AxiomMonitor(MonitorStore(tmp_path))
    monitor.register(candidate(), "register")
    monitor.start_watching("candidate-lean-1", "watch")
    monitor.wait("candidate-lean-1", "wait")
    adapter = LeanObservationAdapter(monitor)

    state = adapter.on_market_event(
        LeanMarketEvent(
            source="LEAN",
            event_id="lean-event-1",
            candidate_id="candidate-lean-1",
            instrument_id="SPY",
            source_sequence=1,
            observed_at=NOW,
            received_at=NOW,
            connected=True,
            price_micros=500_000_001,
            volume=10_001,
        )
    )

    assert state == CandidateState.TRIGGERED


def test_lean_disconnect_reconnect_is_event_driven(tmp_path):
    monitor = AxiomMonitor(MonitorStore(tmp_path))
    monitor.register(candidate(), "register")
    monitor.start_watching("candidate-lean-1", "watch")
    monitor.wait("candidate-lean-1", "wait")
    adapter = LeanObservationAdapter(monitor)

    assert adapter.on_market_event(
        LeanMarketEvent(
            source="LEAN",
            event_id="disconnect",
            candidate_id="candidate-lean-1",
            instrument_id="SPY",
            source_sequence=1,
            observed_at=NOW,
            received_at=NOW,
            connected=False,
        )
    ) == CandidateState.WAITING
    assert adapter.on_market_event(
        LeanMarketEvent(
            source="LEAN",
            event_id="reconnect",
            candidate_id="candidate-lean-1",
            instrument_id="SPY",
            source_sequence=2,
            observed_at=NOW + timedelta(seconds=1),
            received_at=NOW + timedelta(seconds=1),
            connected=True,
            price_micros=499_000_000,
            volume=9_000,
        )
    ) == CandidateState.WAITING


def test_lean_event_contract_rejects_non_lean_or_malformed_data():
    with pytest.raises(ValueError):
        LeanMarketEvent(
            source="BROKER",
            event_id="event",
            candidate_id="candidate",
            instrument_id="SPY",
            source_sequence=1,
            observed_at=NOW,
            received_at=NOW,
            connected=True,
            price_micros=500_000_000,
            volume=10_000,
        )
