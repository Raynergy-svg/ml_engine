from datetime import datetime, timedelta, timezone
from pathlib import Path
import ast

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
from src.axiom2.monitoring.store import MonitorStore, MonitorStoreOrderingError


NOW = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)


class Sink:
    def __init__(self):
        self.wakeups = []

    def wake(self, wakeup):
        self.wakeups.append(wakeup)


def candidate():
    return CandidateRegistration(
        candidate_id="candidate-boundary",
        instrument_id="SPY",
        thesis_version="thesis-v1",
        thesis_digest="c" * 64,
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
        exit_rules=(
            ThresholdRule(
                metric="PRICE",
                operator="AT_OR_BELOW",
                threshold=498_000_000,
                role="EXIT",
            ),
        ),
    )


def observation(event_id, sequence, *, price=499_000_000, volume=9_000):
    return MarketObservation(
        candidate_id="candidate-boundary",
        instrument_id="SPY",
        event_id=event_id,
        source_sequence=sequence,
        observed_at=NOW + timedelta(seconds=sequence),
        received_at=NOW + timedelta(seconds=sequence),
        connected=True,
        price_micros=price,
        volume=volume,
    )


def waiting(tmp_path, sink=None):
    monitor = AxiomMonitor(MonitorStore(tmp_path), wakeup_sink=sink)
    monitor.register(candidate(), "register")
    monitor.start_watching("candidate-boundary", "watch")
    monitor.wait("candidate-boundary", "wait")
    return monitor


def trigger_ready(monitor, sink):
    monitor.observe(observation("entry", 1, price=500_000_001, volume=10_001))
    response = ResearchResponse(
        response_id="research",
        wakeup_id=sink.wakeups[0].wakeup_id,
        candidate_id="candidate-boundary",
        candidate_version=1,
        decision="REVALIDATE",
    )
    monitor.apply_research_response(response)
    return monitor.revalidate(
        "candidate-boundary",
        observation("revalidation", 2, price=500_000_001, volume=11_000),
        "revalidate",
    )


def test_out_of_order_observation_is_rejected_without_transition(tmp_path):
    monitor = waiting(tmp_path)

    monitor.observe(observation("newer", 2))
    with pytest.raises(MonitorStoreOrderingError):
        monitor.observe(observation("older", 1))

    assert monitor.state("candidate-boundary") == CandidateState.WAITING


def test_crash_after_durable_material_event_before_wakeup_recovers(tmp_path, monkeypatch):
    sink = Sink()
    monitor = waiting(tmp_path, sink)
    original = monitor.process_pending_wakeups

    def crash():
        raise RuntimeError("crash before wakeup delivery")

    monkeypatch.setattr(monitor, "process_pending_wakeups", crash)
    with pytest.raises(RuntimeError):
        monitor.observe(observation("entry", 1, price=500_000_001, volume=10_001))

    restarted = AxiomMonitor(MonitorStore(tmp_path), wakeup_sink=sink)
    assert restarted.process_pending_wakeups() == ("wakeup:entry",)
    assert restarted.process_pending_wakeups() == ()
    assert len(sink.wakeups) == 1


def test_duplicate_research_response_is_idempotent(tmp_path):
    sink = Sink()
    monitor = waiting(tmp_path, sink)
    monitor.observe(observation("entry", 1, price=500_000_001, volume=10_001))
    response = ResearchResponse(
        response_id="research",
        wakeup_id=sink.wakeups[0].wakeup_id,
        candidate_id="candidate-boundary",
        candidate_version=1,
        decision="REVALIDATE",
    )

    assert monitor.apply_research_response(response) == CandidateState.REVALIDATING
    assert monitor.apply_research_response(response) == CandidateState.REVALIDATING
    assert len(
        [
            row
            for row in MonitorStore(tmp_path).replay()
            if row.kind.value == "REVALIDATION_STARTED"
        ]
    ) == 1


def test_invalidation_immediately_before_execution_blocks_reconciliation(tmp_path):
    sink = Sink()
    monitor = waiting(tmp_path, sink)
    assert trigger_ready(monitor, sink) == CandidateState.READY

    assert monitor.observe(
        observation("invalidation", 3, price=494_999_999, volume=11_000)
    ) == CandidateState.INVALIDATED
    with pytest.raises(MonitorTransitionError):
        monitor.apply_reconciliation(
            ReconciliationReceipt(
                receipt_id="entry",
                candidate_id="candidate-boundary",
                kind="ENTRY_EXECUTED",
                external_execution_id="external-entry",
                occurred_at=NOW + timedelta(seconds=4),
            )
        )


def test_duplicate_reconciliation_and_distinct_execution_are_blocked(tmp_path):
    sink = Sink()
    monitor = waiting(tmp_path, sink)
    assert trigger_ready(monitor, sink) == CandidateState.READY
    receipt = ReconciliationReceipt(
        receipt_id="entry",
        candidate_id="candidate-boundary",
        kind="ENTRY_EXECUTED",
        external_execution_id="external-entry",
        occurred_at=NOW + timedelta(seconds=3),
    )

    assert monitor.apply_reconciliation(receipt) == CandidateState.EXECUTED
    assert monitor.apply_reconciliation(receipt) == CandidateState.EXECUTED
    with pytest.raises(MonitorTransitionError):
        monitor.apply_reconciliation(
            receipt.model_copy(
                update={
                    "receipt_id": "entry-duplicate",
                    "external_execution_id": "external-entry-duplicate",
                }
            )
        )


def test_monitoring_package_has_no_broker_write_or_execution_imports():
    root = Path(__file__).parents[3] / "src" / "axiom2" / "monitoring"
    forbidden = (
        "src.axiom2.execution",
        "src.axiom2.brokers",
        "src.scanner.execution",
        "broker",
        "place_order",
        "submit",
        "cancel_order",
    )
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                assert not any(name.startswith(forbidden[:3]) for name in names)
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                assert not any(name.startswith(prefix) for prefix in forbidden[:3])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in forbidden[3:]
                elif isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden[3:]


def test_ready_is_not_an_execution_surface(tmp_path):
    sink = Sink()
    monitor = waiting(tmp_path, sink)
    assert trigger_ready(monitor, sink) == CandidateState.READY
    assert not hasattr(monitor, "submit")
    assert not hasattr(monitor, "place_order")
    assert not hasattr(monitor, "broker")
    assert not hasattr(monitor, "execution_authority")
