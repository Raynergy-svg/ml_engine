"""Tests for TradeSlot state machine and SlotManager."""

import time
from unittest.mock import patch

import pytest

from src.scanner.automation.trade_slots import (
    LEGAL_TRANSITIONS,
    SlotManager,
    TradeSlot,
    TradeSlotState,
)


class TestTradeSlotLifecycle:
    """Test the full state machine lifecycle."""

    def test_full_lifecycle_idle_to_idle(self):
        """Full lifecycle: IDLE->SCANNING->CANDIDATE->ARMED->OPEN->SUPERVISING->CLOSING->DIAGNOSING->ADAPTING->IDLE."""
        mgr = SlotManager(max_concurrent=3)
        pair = "EUR_USD"

        assert mgr.get_state(pair) == TradeSlotState.IDLE

        transitions = [
            TradeSlotState.SCANNING,
            TradeSlotState.CANDIDATE,
            TradeSlotState.ARMED,
            TradeSlotState.OPEN,
            TradeSlotState.SUPERVISING,
            TradeSlotState.CLOSING,
            TradeSlotState.DIAGNOSING,
            TradeSlotState.ADAPTING,
            TradeSlotState.IDLE,
        ]
        for state in transitions:
            mgr.transition(pair, state)
            assert mgr.get_state(pair) == state

    def test_illegal_transition_raises_valueerror(self):
        """Illegal transitions raise ValueError."""
        mgr = SlotManager()
        pair = "GBP_USD"

        # IDLE -> OPEN is illegal (must go through SCANNING->CANDIDATE->ARMED first)
        with pytest.raises(ValueError, match="Illegal transition"):
            mgr.transition(pair, TradeSlotState.OPEN)

    def test_multiple_illegal_transitions(self):
        """Various illegal transitions all raise ValueError."""
        mgr = SlotManager()
        pair = "USD_JPY"

        mgr.transition(pair, TradeSlotState.SCANNING)

        # SCANNING -> OPEN is illegal
        with pytest.raises(ValueError):
            mgr.transition(pair, TradeSlotState.OPEN)

        # SCANNING -> ARMED is illegal (must go through CANDIDATE)
        with pytest.raises(ValueError):
            mgr.transition(pair, TradeSlotState.ARMED)


class TestSlotCapacity:
    """Test slot capacity and availability."""

    def test_available_slots_respects_max_concurrent(self):
        """available_slots never exceeds max_concurrent."""
        mgr = SlotManager(max_concurrent=2)
        assert mgr.available_slots() == 2

        mgr.transition("EUR_USD", TradeSlotState.SCANNING)
        assert mgr.available_slots() == 1

        mgr.transition("GBP_USD", TradeSlotState.SCANNING)
        assert mgr.available_slots() == 0

    def test_concurrent_slot_limit_enforcement(self):
        """Non-idle slots consume capacity correctly."""
        mgr = SlotManager(max_concurrent=3)

        # Fill all 3 slots
        for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
            mgr.transition(pair, TradeSlotState.SCANNING)
            mgr.transition(pair, TradeSlotState.CANDIDATE)
            mgr.transition(pair, TradeSlotState.ARMED)
            mgr.transition(pair, TradeSlotState.OPEN)

        assert mgr.available_slots() == 0

        # Free one slot
        mgr.transition("EUR_USD", TradeSlotState.IDLE)  # reset
        assert mgr.available_slots() == 1

    def test_open_positions_returns_only_open_supervising(self):
        """open_positions returns only slots in OPEN or SUPERVISING state."""
        mgr = SlotManager(max_concurrent=5)

        # EUR_USD -> OPEN
        for state in [TradeSlotState.SCANNING, TradeSlotState.CANDIDATE,
                      TradeSlotState.ARMED, TradeSlotState.OPEN]:
            mgr.transition("EUR_USD", state)

        # GBP_USD -> SUPERVISING
        for state in [TradeSlotState.SCANNING, TradeSlotState.CANDIDATE,
                      TradeSlotState.ARMED, TradeSlotState.OPEN,
                      TradeSlotState.SUPERVISING]:
            mgr.transition("GBP_USD", state)

        # USD_JPY -> SCANNING (should NOT appear)
        mgr.transition("USD_JPY", TradeSlotState.SCANNING)

        positions = mgr.open_positions()
        pairs = {s.pair for s in positions}
        assert pairs == {"EUR_USD", "GBP_USD"}
        assert len(positions) == 2


class TestCooldown:
    """Test cooldown mechanics."""

    def test_cooldown_blocks_availability(self):
        """A pair in COOLDOWN state consumes a slot."""
        mgr = SlotManager(max_concurrent=2)

        # Walk EUR_USD through to ADAPTING, then set cooldown
        for state in [TradeSlotState.SCANNING, TradeSlotState.CANDIDATE,
                      TradeSlotState.ARMED, TradeSlotState.OPEN,
                      TradeSlotState.SUPERVISING, TradeSlotState.CLOSING,
                      TradeSlotState.DIAGNOSING, TradeSlotState.ADAPTING]:
            mgr.transition("EUR_USD", state)

        mgr.set_cooldown("EUR_USD", duration_seconds=60)
        assert mgr.get_state("EUR_USD") == TradeSlotState.COOLDOWN
        # COOLDOWN is non-idle, so it consumes a slot
        assert mgr.available_slots() == 1

    def test_check_cooldowns_auto_frees_expired(self):
        """check_cooldowns transitions expired COOLDOWN pairs back to IDLE."""
        mgr = SlotManager(max_concurrent=2)

        # Walk through to ADAPTING
        for state in [TradeSlotState.SCANNING, TradeSlotState.CANDIDATE,
                      TradeSlotState.ARMED, TradeSlotState.OPEN,
                      TradeSlotState.SUPERVISING, TradeSlotState.CLOSING,
                      TradeSlotState.DIAGNOSING, TradeSlotState.ADAPTING]:
            mgr.transition("EUR_USD", state)

        # Set cooldown that has already expired
        mgr.set_cooldown("EUR_USD", duration_seconds=0.0)
        # Manually set cooldown_until to the past
        mgr._slots["EUR_USD"].cooldown_until = time.time() - 1

        freed = mgr.check_cooldowns()
        assert "EUR_USD" in freed
        assert mgr.get_state("EUR_USD") == TradeSlotState.IDLE
        assert mgr.available_slots() == 2

    def test_set_cooldown_sets_timer(self):
        """set_cooldown sets the cooldown_until timestamp correctly."""
        mgr = SlotManager()

        for state in [TradeSlotState.SCANNING, TradeSlotState.CANDIDATE,
                      TradeSlotState.ARMED, TradeSlotState.OPEN,
                      TradeSlotState.SUPERVISING, TradeSlotState.CLOSING,
                      TradeSlotState.DIAGNOSING, TradeSlotState.ADAPTING]:
            mgr.transition("EUR_USD", state)

        before = time.time()
        mgr.set_cooldown("EUR_USD", duration_seconds=30)
        after = time.time()

        slot = mgr._slots["EUR_USD"]
        assert slot.cooldown_until is not None
        assert slot.cooldown_until >= before + 30
        assert slot.cooldown_until <= after + 30


class TestResetAndDefaults:
    """Test reset and default state behavior."""

    def test_reset_any_to_idle(self):
        """Transitioning to IDLE works from any state (reset)."""
        mgr = SlotManager()

        # Test reset from each non-IDLE state
        all_states = [
            TradeSlotState.SCANNING,
            TradeSlotState.CANDIDATE,
            TradeSlotState.ARMED,
            TradeSlotState.OPEN,
            TradeSlotState.SUPERVISING,
            TradeSlotState.CLOSING,
            TradeSlotState.DIAGNOSING,
            TradeSlotState.ADAPTING,
            TradeSlotState.COOLDOWN,
        ]

        for i, state in enumerate(all_states):
            pair = f"PAIR_{i}"
            # Force state directly for states we can't easily reach via transitions
            mgr._get_or_create_slot(pair).state = state
            # Reset should always work
            mgr.transition(pair, TradeSlotState.IDLE)
            assert mgr.get_state(pair) == TradeSlotState.IDLE

    def test_get_state_defaults_to_idle_for_unknown_pair(self):
        """get_state returns IDLE for pairs that have never been tracked."""
        mgr = SlotManager()
        assert mgr.get_state("UNKNOWN_PAIR") == TradeSlotState.IDLE
        assert mgr.get_state("NZD_USD") == TradeSlotState.IDLE

    def test_transition_stores_metadata(self):
        """Metadata passed to transition is stored on the slot."""
        mgr = SlotManager()
        mgr.transition("EUR_USD", TradeSlotState.SCANNING, signal_strength=0.85)
        slot = mgr._slots["EUR_USD"]
        assert slot.metadata["signal_strength"] == 0.85
