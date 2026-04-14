"""Tests for TriggerRegistry — trigger queueing, draining, and sweep scheduling."""

import time
from unittest.mock import patch

import pytest

from src.scanner.automation.trigger_registry import (
    ScanTrigger,
    TriggerRegistry,
    TriggerType,
)


def _make_trigger(**kwargs) -> ScanTrigger:
    defaults = {"trigger_type": TriggerType.MANUAL}
    defaults.update(kwargs)
    return ScanTrigger(**defaults)


class TestTriggerRegistry:
    """Core TriggerRegistry behaviour."""

    def test_fire_adds_to_queue(self):
        reg = TriggerRegistry()
        assert not reg.has_pending()
        reg.fire(_make_trigger())
        assert reg.has_pending()

    def test_drain_returns_all_and_clears(self):
        reg = TriggerRegistry()
        t1 = _make_trigger(source="a")
        t2 = _make_trigger(source="b")
        reg.fire(t1)
        reg.fire(t2)

        drained = reg.drain()
        assert len(drained) == 2
        assert drained[0] is t1
        assert drained[1] is t2
        # Queue should be empty after drain
        assert not reg.has_pending()
        assert reg.drain() == []

    def test_has_pending_correct(self):
        reg = TriggerRegistry()
        assert reg.has_pending() is False
        reg.fire(_make_trigger())
        assert reg.has_pending() is True
        reg.drain()
        assert reg.has_pending() is False

    def test_schedule_sweep_creates_scheduled_sweep_trigger(self):
        reg = TriggerRegistry()
        pairs = ["EUR_USD", "GBP_USD"]
        reg.schedule_sweep(pairs=pairs)

        triggers = reg.drain()
        assert len(triggers) == 1
        t = triggers[0]
        assert t.trigger_type == TriggerType.SCHEDULED_SWEEP
        assert t.pairs == pairs

    def test_should_sweep_respects_interval(self):
        reg = TriggerRegistry(sweep_interval_minutes=10.0)
        # Just created — last_sweep is now, so should_sweep is False
        assert reg.should_sweep() is False

        # Simulate time passing beyond the interval
        reg._last_sweep = time.time() - 601  # 10min + 1s ago
        assert reg.should_sweep() is True
        # Timer resets after returning True
        assert reg.should_sweep() is False

    def test_queue_bounded_at_maxlen_100_oldest_dropped(self):
        reg = TriggerRegistry()
        triggers = []
        for i in range(110):
            t = _make_trigger(source=f"src_{i}")
            triggers.append(t)
            reg.fire(t)

        drained = reg.drain()
        assert len(drained) == 100
        # Oldest 10 should have been dropped; first in queue is src_10
        assert drained[0].source == "src_10"
        assert drained[-1].source == "src_109"

    def test_fire_multiple_trigger_types(self):
        """Verify different trigger types coexist in the queue."""
        reg = TriggerRegistry()
        reg.fire(_make_trigger(trigger_type=TriggerType.SLOT_OPENED))
        reg.fire(_make_trigger(trigger_type=TriggerType.VOLATILITY_SPIKE))
        reg.schedule_sweep()

        drained = reg.drain()
        types = [t.trigger_type for t in drained]
        assert types == [
            TriggerType.SLOT_OPENED,
            TriggerType.VOLATILITY_SPIKE,
            TriggerType.SCHEDULED_SWEEP,
        ]
