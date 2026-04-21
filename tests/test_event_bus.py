"""Tests for the trading EventBus (US-501 — asyncio.Queue-based pub/sub)."""

from __future__ import annotations

import asyncio
import threading
import pytest

from src.scanner.automation.event_bus import EventBus, get_event_bus, reset_event_bus


@pytest.fixture(autouse=True)
def clean_bus():
    reset_event_bus()
    yield
    reset_event_bus()


# ── helpers ──────────────────────────────────────────────────────────

async def collect_n(bus: EventBus, n: int) -> list[dict]:
    """Collect exactly n events from the bus, then return."""
    events: list[dict] = []
    async for event in bus.subscribe():
        events.append(event)
        if len(events) >= n:
            break
    return events


# ── publish / subscribe ───────────────────────────────────────────────

def test_publish_subscribe_basic():
    """subscribe() yields a published event."""
    bus = EventBus()
    bus.publish("signal.pending", {"pair": "EUR_USD"})

    events = asyncio.run(collect_n(bus, 1))

    assert len(events) == 1
    assert events[0]["event_type"] == "signal.pending"
    assert events[0]["payload"]["pair"] == "EUR_USD"
    assert "ts" in events[0]


def test_multiple_event_types():
    """Different event types are all delivered."""
    bus = EventBus()
    bus.publish("signal.executed", {"pair": "GBP_USD"})
    bus.publish("control.pause", {})
    bus.publish("adjustment.proposed", {"key": "min_confidence", "value": 0.7})

    events = asyncio.run(collect_n(bus, 3))

    types = [e["event_type"] for e in events]
    assert types == ["signal.executed", "control.pause", "adjustment.proposed"]


def test_event_contains_timestamp():
    """Every event has a non-empty ts field."""
    bus = EventBus()
    bus.publish("trade.manual_close", {"trade_id": "42"})

    events = asyncio.run(collect_n(bus, 1))
    assert events[0]["ts"] != ""


# ── replay buffer ─────────────────────────────────────────────────────

def test_replay_buffer_late_subscriber():
    """Late subscriber receives all buffered events."""
    bus = EventBus()
    for i in range(5):
        bus.publish("signal.pending", {"i": i})

    events = asyncio.run(collect_n(bus, 5))

    assert len(events) == 5
    for idx, ev in enumerate(events):
        assert ev["payload"]["i"] == idx


def test_replay_buffer_caps_at_100():
    """Replay buffer holds at most 100 events."""
    bus = EventBus()
    for i in range(150):
        bus.publish("signal.pending", {"i": i})

    assert len(bus.replay_buffer()) == 100
    events = asyncio.run(collect_n(bus, 100))
    assert len(events) == 100
    # The first event in replay should be index 50 (last 100 of 150)
    assert events[0]["payload"]["i"] == 50


def test_replay_buffer_empty_on_fresh_bus():
    """New bus has an empty replay buffer."""
    bus = EventBus()
    assert bus.replay_buffer() == []


# ── thread safety ─────────────────────────────────────────────────────

def test_thread_safety_10x100():
    """10 threads × 100 events each — no corruption, replay intact."""
    bus = EventBus()
    n_threads = 10
    n_per_thread = 100
    errors: list[str] = []

    def publish_batch(tid: int) -> None:
        for i in range(n_per_thread):
            try:
                bus.publish("signal.pending", {"tid": tid, "i": i})
            except Exception as exc:
                errors.append(str(exc))

    threads = [threading.Thread(target=publish_batch, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Publish errors: {errors}"
    # Replay buffer should be exactly 100 (capped)
    assert len(bus.replay_buffer()) == 100


def test_publish_from_background_thread():
    """Events published from a non-main thread are delivered to async subscriber."""
    bus = EventBus()

    async def subscribe_and_verify() -> str:
        # Start collecting before the thread publishes
        collected: list[dict] = []

        async def collect_one() -> None:
            async for ev in bus.subscribe():
                collected.append(ev)
                break

        # Give the subscriber a moment to register, then publish from a thread
        task = asyncio.ensure_future(collect_one())

        def publish_from_thread() -> None:
            bus.publish("signal.executed", {"source": "thread"})

        await asyncio.sleep(0.01)
        t = threading.Thread(target=publish_from_thread)
        t.start()
        t.join()

        await asyncio.wait_for(task, timeout=2.0)
        assert len(collected) == 1
        assert collected[0]["payload"]["source"] == "thread"

    asyncio.run(subscribe_and_verify())


# ── singleton ─────────────────────────────────────────────────────────

def test_get_event_bus_singleton():
    """get_event_bus() returns the same instance on repeated calls."""
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2


def test_reset_event_bus_creates_new_instance():
    """reset_event_bus() causes the next get_event_bus() to return a fresh instance."""
    bus1 = get_event_bus()
    reset_event_bus()
    bus2 = get_event_bus()
    assert bus1 is not bus2


def test_reset_clears_replay_buffer():
    """After reset, the new bus has an empty replay buffer."""
    bus = get_event_bus()
    bus.publish("signal.pending", {})
    reset_event_bus()
    fresh = get_event_bus()
    assert fresh.replay_buffer() == []
