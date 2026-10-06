from datetime import datetime, timedelta, timezone

import pytest

from src.axiom2.monitoring.contracts import MonitorEvent, MonitorEventKind
from src.axiom2.monitoring.store import (
    MonitorStore,
    MonitorStoreCorruption,
    MonitorStoreOrderingError,
)


NOW = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)


def event(
    event_id: str,
    *,
    candidate_id: str = "candidate-spy-1",
    source_sequence: int | None = None,
    kind: MonitorEventKind = MonitorEventKind.OBSERVATION_ACCEPTED,
    payload: dict[str, object] | None = None,
) -> MonitorEvent:
    return MonitorEvent(
        event_id=event_id,
        candidate_id=candidate_id,
        kind=kind,
        source_sequence=source_sequence,
        occurred_at=NOW,
        payload=payload or {"value": event_id},
    )


def test_append_replay_preserves_sequence_chain_and_digest(tmp_path):
    store = MonitorStore(tmp_path)
    first = store.append(event("event-1", source_sequence=1), received_at=NOW)
    second = store.append(
        event("event-2", source_sequence=2),
        received_at=NOW + timedelta(seconds=1),
    )

    rows = store.replay()

    assert first != second
    assert [row.sequence for row in rows] == [0, 1]
    assert rows[1].previous_digest == second if False else rows[1].previous_digest == first
    assert store.head_digest() == second


def test_duplicate_event_is_idempotent_and_conflicting_duplicate_fails(tmp_path):
    store = MonitorStore(tmp_path)
    digest = store.append(event("event-1", source_sequence=1), received_at=NOW)

    assert store.append(event("event-1", source_sequence=1), received_at=NOW) == digest
    with pytest.raises(MonitorStoreCorruption, match="conflicting duplicate"):
        store.append(
            event("event-1", source_sequence=1, payload={"different": True}),
            received_at=NOW,
        )
    assert len(store.replay()) == 1


def test_out_of_order_observation_is_rejected_without_state_change(tmp_path):
    store = MonitorStore(tmp_path)
    store.append(event("event-1", source_sequence=3), received_at=NOW)

    with pytest.raises(MonitorStoreOrderingError, match="source watermark"):
        store.append(event("event-older", source_sequence=2), received_at=NOW)

    assert [row.event_id for row in store.replay()] == ["event-1"]


def test_replay_rejects_tampered_receipt(tmp_path):
    store = MonitorStore(tmp_path)
    store.append(event("event-1", source_sequence=1), received_at=NOW)
    path = next((tmp_path / "events").iterdir())
    path.write_text(path.read_text().replace("event-1", "event-tampered"))

    with pytest.raises(MonitorStoreCorruption):
        MonitorStore(tmp_path).replay()


def test_crash_after_durable_create_recovers_idempotently(tmp_path, monkeypatch):
    store = MonitorStore(tmp_path)
    original = store._atomic_create_bytes

    def write_then_raise(path, data):
        original(path, data)
        raise RuntimeError("simulated crash after commit")

    monkeypatch.setattr(store, "_atomic_create_bytes", write_then_raise)
    with pytest.raises(RuntimeError):
        store.append(event("event-1", source_sequence=1), received_at=NOW)

    restarted = MonitorStore(tmp_path)
    assert restarted.append(event("event-1", source_sequence=1), received_at=NOW) == restarted.head_digest()
    assert len(restarted.replay()) == 1


def test_pending_wakeup_is_recovered_from_event_history(tmp_path):
    store = MonitorStore(tmp_path)
    store.append(
        event(
            "trigger-1",
            source_sequence=1,
            kind=MonitorEventKind.ENTRY_TRIGGERED,
            payload={"wakeup_id": "wakeup-1"},
        ),
        received_at=NOW,
    )

    assert [row.payload["wakeup_id"] for row in store.pending_wakeups()] == ["wakeup-1"]

    store.append(
        event(
            "delivery-1",
            kind=MonitorEventKind.RESEARCH_WAKEUP_DELIVERED,
            payload={"wakeup_id": "wakeup-1"},
        ),
        received_at=NOW + timedelta(seconds=1),
    )
    assert store.pending_wakeups() == ()
