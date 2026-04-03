"""Tests for TradingEventBus — thread safety, priority, idempotency, history."""

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from src.scanner.automation.trading_event_bus import (
    TradingEventBus,
    get_trading_event_bus,
    reset_trading_event_bus,
)


@pytest.fixture(autouse=True)
def _isolate_bus():
    """Reset singleton and patch history path for every test."""
    reset_trading_event_bus()
    yield
    reset_trading_event_bus()


@pytest.fixture
def tmp_history(tmp_path):
    """Patch _HISTORY_PATH to a temp file so tests don't touch real data."""
    p = tmp_path / "events.jsonl"
    with patch("src.scanner.automation.trading_event_bus._HISTORY_PATH", p):
        yield p


@pytest.fixture
def bus(tmp_history):
    return TradingEventBus()


# ── 1. subscribe + emit delivers to handler ─────────────────────────

def test_subscribe_and_emit(bus):
    received = []
    bus.subscribe("trade_closed", lambda e: received.append(e))
    event = {"event_type": "trade_closed", "event_id": "t1", "pair": "EUR_USD"}
    bus.emit(event)
    assert len(received) == 1
    assert received[0]["pair"] == "EUR_USD"


# ── 2. unsubscribe stops delivery ───────────────────────────────────

def test_unsubscribe(bus):
    received = []
    bus.subscribe("trade_closed", lambda e: received.append(e), handler_name="h1")
    bus.unsubscribe("h1")
    bus.emit({"event_type": "trade_closed", "event_id": "u1"})
    assert received == []


# ── 3. priority ordering (lower number fires first) ─────────────────

def test_priority_ordering(bus):
    order = []
    bus.subscribe("sig", lambda e: order.append("C"), priority=300, handler_name="c")
    bus.subscribe("sig", lambda e: order.append("A"), priority=100, handler_name="a")
    bus.subscribe("sig", lambda e: order.append("B"), priority=200, handler_name="b")
    bus.emit({"event_type": "sig", "event_id": "p1"})
    assert order == ["A", "B", "C"]


# ── 4. idempotency — same event_id only fires once ──────────────────

def test_idempotency(bus):
    count = []
    bus.subscribe("tick", lambda e: count.append(1))
    event = {"event_type": "tick", "event_id": "dup1"}
    bus.emit(event)
    bus.emit(event)
    assert len(count) == 1


# ── 5. get_history returns emitted events ────────────────────────────

def test_get_history(bus, tmp_history):
    bus.emit({"event_type": "trade_closed", "event_id": "h1", "pnl": 42.0})
    bus.emit({"event_type": "trade_opened", "event_id": "h2", "pair": "GBP_USD"})

    history = bus.get_history()
    assert len(history) == 2
    assert history[0]["event_id"] == "h1"

    filtered = bus.get_history(event_type="trade_opened")
    assert len(filtered) == 1
    assert filtered[0]["pair"] == "GBP_USD"


# ── 6. singleton returns same instance ──────────────────────────────

def test_singleton_same_instance():
    a = get_trading_event_bus()
    b = get_trading_event_bus()
    assert a is b


# ── 7. reset creates fresh instance ─────────────────────────────────

def test_reset_creates_fresh_instance():
    a = get_trading_event_bus()
    reset_trading_event_bus()
    b = get_trading_event_bus()
    assert a is not b


# ── 8. handler exception does not crash bus ──────────────────────────

def test_handler_exception_does_not_crash(bus):
    results = []

    def bad_handler(e):
        raise RuntimeError("boom")

    def good_handler(e):
        results.append("ok")

    bus.subscribe("x", bad_handler, priority=1, handler_name="bad")
    bus.subscribe("x", good_handler, priority=2, handler_name="good")

    bus.emit({"event_type": "x", "event_id": "err1"})
    assert results == ["ok"]


# ── 9. thread safety — concurrent emits don't lose events ───────────

def test_thread_safety(bus):
    counter = {"n": 0}
    lock = threading.Lock()

    def handler(e):
        with lock:
            counter["n"] += 1

    bus.subscribe("ts", handler)

    threads = []
    for i in range(50):
        t = threading.Thread(
            target=bus.emit,
            args=({"event_type": "ts", "event_id": f"ts-{i}"},),
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    assert counter["n"] == 50


# ── 10. events without event_id always fire (no idempotency) ────────

def test_no_event_id_always_fires(bus):
    count = []
    bus.subscribe("raw", lambda e: count.append(1))
    bus.emit({"event_type": "raw"})
    bus.emit({"event_type": "raw"})
    assert len(count) == 2
