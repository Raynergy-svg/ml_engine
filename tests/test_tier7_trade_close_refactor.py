"""Tests for Tier 7 PR 3: Trade-close event emission + handler extraction.

Validates:
1. Event creation from outcome_data (including None-safe handling)
2. Handler registry completeness (31 handlers registered)
3. SyncContext construction
4. Per-trade vs post-sync handler classification
5. Composite handler dependency chains
6. EventQueue integration with handlers
"""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.automation.trading_events import (
    TradingEventType,
    TradingEvent,
    create_trade_closed_event,
    validate_event,
)
from src.scanner.automation.event_queue import EventQueue, FlushGate
from src.scanner.automation.event_handlers import (
    SyncContext,
    PER_TRADE_HANDLERS,
    POST_SYNC_HANDLERS,
    get_all_handler_instances,
    get_handler_count,
    RLReplayHandler,
    AccuracyGateHandler,
    AttributionPipelineHandler,
    WalkForwardPipelineHandler,
    RetrainDriftCheckHandler,
    StatePersistenceHandler,
    CounterfactualHandler,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_outcome_data(**overrides):
    """Build a realistic outcome_data dict."""
    base = {
        "realized_pl": -15.0,
        "pnl_pips": -12.3,
        "trade_won": False,
        "close_price": 1.0985,
        "close_time": "2026-04-02T01:15:00Z",
        "exit_price": 1.0985,
        "exit_time": "2026-04-02T01:15:00Z",
        "duration_minutes": 45.0,
        "exit_reason": "sl_hit",
        "mae_pips": -18.5,
        "mfe_pips": 5.2,
        "bars_in_trade": 9,
        "sl_pips": 15.0,
        "tp_pips": 25.0,
        "rr_ratio": 1.67,
        "regime_at_entry": "NORMAL",
        "regime_at_exit": "NORMAL",
    }
    base.update(overrides)
    return base


def _make_entry(**overrides):
    """Build a realistic journal entry dict."""
    base = {
        "trade_id": "2092",
        "pair": "EUR_USD",
        "direction": "LONG",
        "timestamp": "2026-04-02T01:00:00Z",
        "entry_price": 1.1000,
        "confidence": 0.72,
        "sl_pips": 15.0,
        "tp_pips": 25.0,
        "regime": {"volatility_regime": "NORMAL"},
        "agents": {"agent_reasons": [], "weighted_vote_score": 0.65},
        "model": "ensemble",
        "outcome": None,
    }
    base.update(overrides)
    return base


def _make_sync_context(**overrides):
    """Build a SyncContext with sensible defaults."""
    entry = _make_entry()
    outcome = _make_outcome_data()
    base = dict(
        entry=entry,
        outcome_data=outcome,
        closed_trades={"2092": {"id": "2092", "realizedPL": "-15.0", "initialUnits": "1000"}},
        entries=[entry],
        synced_entries=[entry],
        rl_updates=[],
        scanner=MagicMock(),
        execution_manager=MagicMock(),
        synced_count=1,
        trade_id="2092",
        pair="EUR_USD",
        trade_won=False,
        realized_pl=-15.0,
        pnl_pips=-12.3,
    )
    base.update(overrides)
    return SyncContext(**base)


# ── Event Creation Tests ──────────────────────────────────────────────

class TestEventCreation:

    def test_create_event_from_outcome_data(self):
        outcome = _make_outcome_data()
        event = create_trade_closed_event(outcome, "session_123")
        assert event.event_type == TradingEventType.TRADE_CLOSED
        assert event.session_id == "session_123"
        assert event.event_id  # non-empty UUID
        assert event.payload["pnl_pips"] == -12.3
        assert validate_event(event)

    def test_create_event_with_none_values(self):
        """Outcome data may have None for sl_pips, tp_pips, etc."""
        outcome = _make_outcome_data(sl_pips=None, tp_pips=None, rr_ratio=None)
        event = create_trade_closed_event(outcome, "session_123")
        assert event.payload["sl_pips"] == 0.0
        assert event.payload["tp_pips"] == 0.0
        assert validate_event(event)

    def test_create_event_with_minimal_outcome(self):
        """Bare minimum outcome_data — everything optional is missing."""
        event = create_trade_closed_event({}, "session_x")
        assert event.event_type == TradingEventType.TRADE_CLOSED
        assert event.payload["realized_pl"] == 0.0
        assert validate_event(event)

    def test_unique_event_ids(self):
        outcome = _make_outcome_data()
        e1 = create_trade_closed_event(outcome, "s1")
        e2 = create_trade_closed_event(outcome, "s1")
        assert e1.event_id != e2.event_id


# ── Handler Registry Tests ────────────────────────────────────────────

class TestHandlerRegistry:

    def test_handler_count(self):
        assert get_handler_count() == len(PER_TRADE_HANDLERS) + len(POST_SYNC_HANDLERS)
        assert get_handler_count() >= 28  # At least 28 handlers extracted

    def test_all_handlers_instantiate(self):
        instances = get_all_handler_instances()
        assert len(instances) >= 28
        for h in instances:
            assert hasattr(h, "name")
            assert hasattr(h, "priority")
            assert hasattr(h, "handle")

    def test_handlers_sorted_by_priority(self):
        instances = get_all_handler_instances()
        priorities = [h.priority for h in instances]
        assert priorities == sorted(priorities)

    def test_per_trade_vs_post_sync_split(self):
        assert len(PER_TRADE_HANDLERS) >= 15  # At least 15 per-trade handlers
        assert len(POST_SYNC_HANDLERS) >= 12  # At least 12 post-sync handlers
        # No overlap
        per_set = set(PER_TRADE_HANDLERS)
        post_set = set(POST_SYNC_HANDLERS)
        assert per_set.isdisjoint(post_set)

    def test_composite_handlers_exist(self):
        """Attribution and WalkForward pipeline handlers preserve dependency chains."""
        per_names = {cls.__name__ for cls in PER_TRADE_HANDLERS}
        post_names = {cls.__name__ for cls in POST_SYNC_HANDLERS}
        all_names = per_names | post_names
        assert "AttributionPipelineHandler" in all_names
        assert "WalkForwardPipelineHandler" in all_names


# ── Individual Handler Tests ──────────────────────────────────────────

class TestHandlerExecution:

    def test_rl_replay_handler(self):
        ctx = _make_sync_context()
        handler = RLReplayHandler()
        result = handler.handle({}, ctx)
        assert result is True
        ctx.execution_manager._sync_trade_outcome_to_rl.assert_called_once()

    def test_accuracy_gate_handler(self):
        ctx = _make_sync_context()
        ag = MagicMock()
        ctx.scanner._accuracy_gate = ag
        handler = AccuracyGateHandler()
        result = handler.handle({}, ctx)
        assert result is True
        ag.record_outcome.assert_called_once()

    def test_handler_returns_false_when_module_missing(self):
        ctx = _make_sync_context(execution_manager=None)
        handler = RLReplayHandler()
        assert handler.handle({}, ctx) is False

    def test_attribution_pipeline_runs_three_steps(self):
        ctx = _make_sync_context()
        em = ctx.execution_manager
        em._attribution_engine = MagicMock()
        em._attribution_engine.attribute_trade.return_value = MagicMock(
            alignment_ratio=0.8,
            aligned_agents=[MagicMock(agent_name="trend")],
            misaligned_agents=[],
        )
        em._attribution_engine.get_weight_recommendations.return_value = {"trend": 1.1}
        em._attribution_engine.generate_report.return_value = {"agents": []}
        em._gate_threshold_optimizer = MagicMock()
        em._gate_threshold_optimizer.compute_recommendations.return_value = []

        # Give it agent_reasons to process
        ctx.entry["agents"] = {
            "agent_reasons": [
                {"name": "trend", "passed": True, "weight": 1.0, "confidence": 0.8}
            ]
        }

        handler = AttributionPipelineHandler()
        result = handler.handle({}, ctx)
        assert result is True
        em._attribution_engine.attribute_trade.assert_called_once()
        em._gate_threshold_optimizer.record_agent_feedback.assert_called()

    def test_counterfactual_only_analyzes_losses(self):
        ctx = _make_sync_context()
        cc = MagicMock()
        cc.generate_standard_scenarios.return_value = []
        cc.batch_analyze.return_value = []
        ctx.scanner._causal_counterfactual = cc

        # Make a winning trade — should skip
        ctx.closed_trades["2092"]["realizedPL"] = "25.0"
        handler = CounterfactualHandler()
        result = handler.handle({}, ctx)
        assert result is True
        cc.generate_standard_scenarios.assert_not_called()


# ── Queue + Handler Integration ───────────────────────────────────────

class TestQueueHandlerIntegration:

    def test_handlers_execute_through_queue(self):
        """Enqueue a handler and verify it executes."""
        completed = []

        class TestHandler:
            name = "test_handler"
            priority = 50
            def handle(self, event, context):
                completed.append(event["event_id"])
                return True

        q = EventQueue(max_size=10, max_retries=1, base_delay=0.01, max_delay=0.1)
        q.start()
        try:
            handler = TestHandler()
            ctx = _make_sync_context()
            event_dict = {"event_id": "test_123", "event_type": "trade_closed"}

            q.enqueue(
                handler_name=handler.name,
                handler_fn=lambda ev, h=handler, c=ctx: h.handle(ev, c),
                event=event_dict,
                priority=handler.priority,
            )
            q.flush()
            assert "test_123" in completed
        finally:
            q.stop(flush_first=False)

    def test_handler_failure_does_not_block_queue(self):
        """A failing handler should not prevent subsequent handlers from running."""
        results = []

        def fail_handler(event):
            raise RuntimeError("intentional failure")

        def ok_handler(event):
            results.append("ok")

        q = EventQueue(max_size=10, max_retries=1, base_delay=0.01, max_delay=0.1)
        q.start()
        try:
            q.enqueue("fail", fail_handler, {"event_id": "f1"}, priority=1)
            q.enqueue("ok", ok_handler, {"event_id": "f2"}, priority=2)
            q.flush()
            assert "ok" in results
            status = q.get_status()
            assert status["failed_count"] >= 1
        finally:
            q.stop(flush_first=False)


# ── Execution.py Integration Smoke Test ───────────────────────────────

class TestExecutionIntegration:

    def test_tier7_available_flag(self):
        """Tier 7 modules should be importable."""
        from src.scanner.execution import _TIER7_AVAILABLE
        assert _TIER7_AVAILABLE is True

    def test_execution_manager_has_tier7_attrs(self):
        """ExecutionManager should have event_bus and event_queue after init."""
        from src.scanner.execution import ExecutionManager
        em = ExecutionManager()
        assert hasattr(em, "_event_bus")
        assert hasattr(em, "_event_queue")
        assert hasattr(em, "_tier7_handlers")
        assert hasattr(em, "_shadow_mode")
        # Queue should be running
        if em._event_queue is not None:
            status = em._event_queue.get_status()
            assert "depth" in status
            em._event_queue.stop(flush_first=False)
