"""Unit tests for FlushGate state machine."""

import pytest

from src.scanner.automation.event_queue import FlushGate


class TestFlushGate:
    """Tests for FlushGate buffer/ordering behaviour."""

    def test_inactive_gate_returns_false_on_enqueue(self):
        """Pass-through: enqueue returns False when gate is not active."""
        gate = FlushGate()
        assert gate.enqueue("item") is False

    def test_active_gate_returns_true_on_enqueue(self):
        """Buffered: enqueue returns True when gate is active."""
        gate = FlushGate()
        gate.start()
        assert gate.enqueue("item") is True

    def test_end_returns_buffered_items_in_fifo_order(self):
        """end() drains items in the order they were enqueued."""
        gate = FlushGate()
        gate.start()
        gate.enqueue("a")
        gate.enqueue("b")
        gate.enqueue("c")
        assert gate.end() == ["a", "b", "c"]

    def test_end_clears_buffer(self):
        """Second end() returns an empty list after first drain."""
        gate = FlushGate()
        gate.start()
        gate.enqueue("x")
        gate.end()
        assert gate.end() == []

    def test_drop_discards_all_items(self):
        """drop() discards buffered items; subsequent end() returns empty."""
        gate = FlushGate()
        gate.start()
        gate.enqueue("a")
        gate.enqueue("b")
        gate.drop()
        assert gate.end() == []

    def test_reactivation_after_end(self):
        """Gate can be restarted after end(), buffering new items."""
        gate = FlushGate()
        gate.start()
        gate.enqueue("old")
        gate.end()

        gate.start()
        gate.enqueue("new1")
        gate.enqueue("new2")
        assert gate.end() == ["new1", "new2"]
