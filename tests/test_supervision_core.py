"""Tests for SupervisionCore — the supervise-first loop brain.

Covers: process_close_event, slot transitions, adaptive policies,
consecutive loss tracking, tick housekeeping, and status reporting.
"""

import time
from typing import Optional
from unittest.mock import patch

import pytest

from src.scanner.automation.close_diagnostician import (
    CloseDiagnosis,
    CloseDiagnostician,
    CloseReason,
)
from src.scanner.automation.supervision_core import SupervisionCore
from src.scanner.automation.supervision_policy_engine import (
    PolicyType,
    SupervisionPolicyEngine,
)
from src.scanner.automation.trade_slots import SlotManager, TradeSlotState
from src.scanner.automation.trigger_registry import TriggerRegistry, TriggerType


# ── Helpers ────────────────────────────────────────────────────

def _make_outcome(
    pair: str = "EUR_USD",
    realized_pl: float = 50.0,
    exit_reason: str = "tp_hit",
    duration_minutes: float = 15.0,
    regime_at_entry: str = "trending",
    regime_at_exit: str = "trending",
    trade_id: str = "T001",
    **overrides,
) -> dict:
    """Build a minimal trade_outcome dict."""
    base = {
        "trade_id": trade_id,
        "pair": pair,
        "direction": "long",
        "exit_reason": exit_reason,
        "entry_price": 1.1000,
        "exit_price": 1.1050,
        "realized_pl": realized_pl,
        "pnl_pips": realized_pl / 10,
        "duration_minutes": duration_minutes,
        "regime_at_entry": regime_at_entry,
        "regime_at_exit": regime_at_exit,
        "confidence_at_entry": 0.72,
    }
    base.update(overrides)
    return base


def _build_core(config: Optional[dict] = None) -> SupervisionCore:
    """Construct a SupervisionCore with fresh dependencies."""
    sm = SlotManager(max_concurrent=3)
    diag = CloseDiagnostician()
    pe = SupervisionPolicyEngine()
    tr = TriggerRegistry(sweep_interval_minutes=30.0)
    return SupervisionCore(sm, diag, pe, tr, config=config)


def _prepare_slot_for_close(core: SupervisionCore, pair: str) -> None:
    """Walk a slot through IDLE -> ... -> CLOSING so process_close_event can run."""
    sm = core.slot_manager
    sm.transition(pair, TradeSlotState.SCANNING)
    sm.transition(pair, TradeSlotState.CANDIDATE)
    sm.transition(pair, TradeSlotState.ARMED)
    sm.transition(pair, TradeSlotState.OPEN)
    sm.transition(pair, TradeSlotState.SUPERVISING)
    sm.transition(pair, TradeSlotState.CLOSING)


# ── Tests ──────────────────────────────────────────────────────


class TestProcessCloseEvent:
    """process_close_event returns CloseDiagnosis and manages slot lifecycle."""

    def test_returns_close_diagnosis(self):
        core = _build_core()
        _prepare_slot_for_close(core, "EUR_USD")
        outcome = _make_outcome(pair="EUR_USD", realized_pl=30.0)

        result = core.process_close_event(outcome)

        assert isinstance(result, CloseDiagnosis)
        assert result.pair == "EUR_USD"
        assert result.realized_pl == 30.0
        assert result.close_reason == CloseReason.TP_HIT

    def test_slot_transitions_diagnosing_adapting_idle_on_win(self):
        """Win with no consecutive losses -> DIAGNOSING -> ADAPTING -> IDLE."""
        core = _build_core()
        _prepare_slot_for_close(core, "GBP_USD")
        outcome = _make_outcome(pair="GBP_USD", realized_pl=20.0)

        core.process_close_event(outcome)

        assert core.slot_manager.get_state("GBP_USD") == TradeSlotState.IDLE

    def test_win_resets_consecutive_loss_counter(self):
        """A win should reset the pair's consecutive loss count to 0."""
        core = _build_core(config={"consecutive_loss_cooldown": 5})
        # Accumulate 2 losses
        for i in range(2):
            _prepare_slot_for_close(core, "AUD_USD")
            core.process_close_event(
                _make_outcome(pair="AUD_USD", realized_pl=-25.0, exit_reason="sl_hit", trade_id=f"L{i}")
            )

        assert core._consecutive_losses.get("AUD_USD", 0) == 2

        # Now a win
        _prepare_slot_for_close(core, "AUD_USD")
        core.process_close_event(
            _make_outcome(pair="AUD_USD", realized_pl=40.0, trade_id="W1")
        )

        assert core._consecutive_losses["AUD_USD"] == 0


class TestConsecutiveLossCooldown:
    """Consecutive losses >= threshold trigger PAIR_COOLDOWN policy."""

    def test_consecutive_losses_trigger_pair_cooldown(self):
        core = _build_core(config={"consecutive_loss_cooldown": 2})

        for i in range(2):
            _prepare_slot_for_close(core, "USD_JPY")
            core.process_close_event(
                _make_outcome(pair="USD_JPY", realized_pl=-30.0, exit_reason="sl_hit", trade_id=f"L{i}")
            )

        # Pair should be in COOLDOWN state
        assert core.slot_manager.get_state("USD_JPY") == TradeSlotState.COOLDOWN

        # PAIR_COOLDOWN policy should exist
        policies = core.policy_engine.get_active_policies(
            pair="USD_JPY", policy_type=PolicyType.PAIR_COOLDOWN
        )
        assert len(policies) >= 1
        assert policies[0].parameter == "pair_block"


class TestRegimeChangeLoss:
    """Regime change on a loss triggers GATE_TIGHTEN policy."""

    def test_regime_change_loss_triggers_gate_tighten(self):
        core = _build_core()
        _prepare_slot_for_close(core, "EUR_USD")

        outcome = _make_outcome(
            pair="EUR_USD",
            realized_pl=-40.0,
            exit_reason="sl_hit",
            regime_at_entry="trending",
            regime_at_exit="ranging",
        )
        core.process_close_event(outcome)

        policies = core.policy_engine.get_active_policies(
            pair="EUR_USD", policy_type=PolicyType.GATE_TIGHTEN
        )
        assert len(policies) == 1
        assert policies[0].parameter == "min_confidence"
        assert policies[0].adjustment == 0.05


class TestFastTPWin:
    """Quick TP hit win triggers SIZING_ADJUST policy."""

    def test_fast_tp_win_triggers_sizing_adjust(self):
        core = _build_core()
        _prepare_slot_for_close(core, "USD_CHF")

        outcome = _make_outcome(
            pair="USD_CHF",
            realized_pl=25.0,
            exit_reason="tp_hit",
            duration_minutes=10.0,  # < 30 min threshold
        )
        core.process_close_event(outcome)

        policies = core.policy_engine.get_active_policies(
            pair="USD_CHF", policy_type=PolicyType.SIZING_ADJUST
        )
        assert len(policies) == 1
        assert policies[0].parameter == "atr_tp_multiplier"
        assert policies[0].adjustment == 0.1


class TestTick:
    """tick() expires stale policies and fires triggers."""

    def test_tick_expires_stale_policies(self):
        core = _build_core()
        # Manually add an already-expired policy
        from src.scanner.automation.supervision_policy_engine import AdaptivePolicy

        stale = AdaptivePolicy(
            policy_type=PolicyType.GATE_TIGHTEN,
            parameter="min_confidence",
            adjustment=0.05,
            reason="test stale",
            expires_at=time.time() - 10,  # already expired
            pair="EUR_USD",
        )
        core.policy_engine.add_policy(stale)
        assert len(core.policy_engine.get_active_policies()) == 0  # filtered out by time

        # tick should clean internal list
        core.tick()
        assert len(core.policy_engine._policies) == 0

    def test_tick_fires_scheduled_sweep_when_due(self):
        core = _build_core()
        # Force sweep interval to 0 and last_sweep into the past so should_sweep() returns True
        core.trigger_registry._sweep_interval = 0.0
        core.trigger_registry._last_sweep = time.time() - 1

        core.tick()

        triggers = core.trigger_registry.drain()
        sweep_triggers = [t for t in triggers if t.trigger_type == TriggerType.SCHEDULED_SWEEP]
        assert len(sweep_triggers) == 1


class TestGetStatus:
    """get_status returns correct supervision state snapshot."""

    def test_get_status_returns_correct_counts(self):
        core = _build_core()

        # Put one pair in OPEN (counts as open_position)
        sm = core.slot_manager
        sm.transition("EUR_USD", TradeSlotState.SCANNING)
        sm.transition("EUR_USD", TradeSlotState.CANDIDATE)
        sm.transition("EUR_USD", TradeSlotState.ARMED)
        sm.transition("EUR_USD", TradeSlotState.OPEN)

        # Set some consecutive losses
        core._consecutive_losses["GBP_USD"] = 3

        status = core.get_status()

        assert status["open_positions"] == 1
        assert status["available_slots"] == 2  # 3 max - 1 open
        assert status["active_policies"] == 0
        assert status["consecutive_losses"] == {"GBP_USD": 3}


class TestGetSupervisionInterval:
    """get_supervision_interval returns correct tier string."""

    def test_returns_correct_tier(self):
        core = _build_core()

        # Default tiers: {15: "tick", 120: "1m", 480: "5m"}
        assert core.get_supervision_interval(5.0) == "tick"    # below 15 -> default "tick"
        assert core.get_supervision_interval(15.0) == "tick"   # == 15 -> "tick"
        assert core.get_supervision_interval(60.0) == "tick"   # between 15 and 120
        assert core.get_supervision_interval(120.0) == "1m"    # == 120
        assert core.get_supervision_interval(300.0) == "1m"    # between 120 and 480
        assert core.get_supervision_interval(480.0) == "5m"    # == 480
        assert core.get_supervision_interval(999.0) == "5m"    # above 480
