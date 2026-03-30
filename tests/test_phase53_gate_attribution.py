"""
Tests for Phase 53 US-328: Gate Attribution Engine.

Tests cover:
  - GateAttributionEngine initialization and state management
  - Decision recording (pass/reject)
  - Daily summary aggregation
  - Good rejection rate computation
  - Report generation with effectiveness flags
  - Persistence (save/load)
  - Rolling window trimming
  - Lazy imports
  - Wiring verification
"""

import json
import os
import tempfile
import time
from unittest.mock import patch, MagicMock

import pytest

from src.scanner.gate_attribution import (
    GateAttributionEngine,
    GateAttributionConfig,
    GateDecision,
    GateStats,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def engine(tmp_dir):
    """Create a GateAttributionEngine with temp persistence."""
    config = GateAttributionConfig(
        persistence_path=os.path.join(tmp_dir, "gate_attribution.json"),
        max_decisions=100,
    )
    return GateAttributionEngine(config)


@pytest.fixture
def journal_path(tmp_dir):
    """Create a mock trade journal for good_rejection_rate testing."""
    path = os.path.join(tmp_dir, "trade_journal_rl.json")
    # Create journal with known outcomes by confidence level
    # Low confidence (0.45-0.55) trades mostly lose
    # High confidence (0.65-0.75) trades mostly win
    trades = []
    # Decile 4 (0.40-0.49): 80% losses
    for i in range(10):
        trades.append({
            "confidence": 0.45,
            "outcome": "loss" if i < 8 else "win",
        })
    # Decile 5 (0.50-0.59): 60% losses
    for i in range(10):
        trades.append({
            "confidence": 0.55,
            "outcome": "loss" if i < 6 else "win",
        })
    # Decile 6 (0.60-0.69): 60% wins
    for i in range(10):
        trades.append({
            "confidence": 0.65,
            "outcome": "win" if i < 6 else "loss",
        })
    # Decile 7 (0.70-0.79): 80% wins
    for i in range(10):
        trades.append({
            "confidence": 0.72,
            "outcome": "win" if i < 8 else "loss",
        })
    with open(path, "w") as f:
        json.dump(trades, f)
    return path


# ── Unit Tests ────────────────────────────────────────────────────────


class TestGateDecision:
    """Tests for GateDecision dataclass."""

    def test_to_dict_roundtrip(self):
        d = GateDecision(
            gate_name="spread_filter",
            rejected=True,
            reason="too wide",
            confidence=0.58,
            pair="EUR_USD",
            timestamp=1234567890.0,
        )
        as_dict = d.to_dict()
        restored = GateDecision.from_dict(as_dict)
        assert restored.gate_name == "spread_filter"
        assert restored.rejected is True
        assert restored.reason == "too wide"
        assert restored.confidence == 0.58
        assert restored.pair == "EUR_USD"

    def test_from_dict_defaults(self):
        d = GateDecision.from_dict({})
        assert d.gate_name == ""
        assert d.rejected is False
        assert d.confidence == 0.0


class TestGateAttributionEngine:
    """Tests for GateAttributionEngine core logic."""

    def test_init_no_file(self, engine):
        """Engine starts empty when no persistence file exists."""
        assert engine.get_decisions_count() == 0
        assert engine.get_gate_names() == []

    def test_record_rejection(self, engine):
        """Recording a rejection increments counts correctly."""
        engine.record_decision(
            "spread_filter", rejected=True,
            reason="spread 2.1 > 1.5", confidence=0.58, pair="EUR_USD",
        )
        assert engine.get_decisions_count() == 1
        summary = engine.get_daily_summary()
        assert "spread_filter" in summary
        assert summary["spread_filter"]["rejections"] == 1
        assert summary["spread_filter"]["passes"] == 0

    def test_record_pass(self, engine):
        """Recording a pass increments pass counts."""
        engine.record_decision(
            "calendar_filter", rejected=False,
            confidence=0.62, pair="GBP_USD",
        )
        summary = engine.get_daily_summary()
        assert summary["calendar_filter"]["passes"] == 1
        assert summary["calendar_filter"]["rejections"] == 0

    def test_multiple_gates(self, engine):
        """Multiple gates tracked independently."""
        engine.record_decision("spread_filter", rejected=True, confidence=0.5, pair="EUR_USD")
        engine.record_decision("spread_filter", rejected=False, confidence=0.6, pair="EUR_USD")
        engine.record_decision("calendar_filter", rejected=True, confidence=0.55, pair="GBP_USD")
        engine.record_decision("fitness_detector", rejected=False, confidence=0.7, pair="AUD_USD")

        assert sorted(engine.get_gate_names()) == [
            "calendar_filter", "fitness_detector", "spread_filter",
        ]

    def test_rolling_window_trim(self, tmp_dir):
        """Decisions trimmed to max_decisions."""
        config = GateAttributionConfig(
            persistence_path=os.path.join(tmp_dir, "ga.json"),
            max_decisions=10,
        )
        engine = GateAttributionEngine(config)
        for i in range(25):
            engine.record_decision("test_gate", rejected=False, confidence=0.5 + i * 0.01)
        assert engine.get_decisions_count() == 10

    def test_good_rejection_rate_high_loss_confidence(self, engine, journal_path):
        """Rejections at low-confidence (high-loss) levels are 'good'."""
        # Record rejections at confidence 0.45 (decile 4 = 80% loss rate)
        for _ in range(5):
            engine.record_decision(
                "test_gate", rejected=True, confidence=0.45, pair="EUR_USD",
            )
        rate = engine.compute_good_rejection_rate("test_gate", journal_path)
        # 80% loss rate at this decile → win_rate 0.20 < 0.50 → good rejection
        assert rate == 1.0

    def test_good_rejection_rate_high_win_confidence(self, engine, journal_path):
        """Rejections at high-confidence (high-win) levels are 'bad'."""
        # Record rejections at confidence 0.72 (decile 7 = 80% win rate)
        for _ in range(5):
            engine.record_decision(
                "bad_gate", rejected=True, confidence=0.72, pair="EUR_USD",
            )
        rate = engine.compute_good_rejection_rate("bad_gate", journal_path)
        # 80% win rate at this decile → win_rate 0.80 > 0.50 → bad rejection
        assert rate == 0.0

    def test_good_rejection_rate_no_journal(self, engine, tmp_dir):
        """Missing journal returns 0.0."""
        engine.record_decision("x", rejected=True, confidence=0.5)
        rate = engine.compute_good_rejection_rate(
            "x", os.path.join(tmp_dir, "nonexistent.json"),
        )
        assert rate == 0.0

    def test_good_rejection_rate_no_rejections(self, engine, journal_path):
        """No rejections returns 0.0."""
        engine.record_decision("y", rejected=False, confidence=0.5)
        rate = engine.compute_good_rejection_rate("y", journal_path)
        assert rate == 0.0

    def test_generate_report(self, engine, journal_path):
        """Report includes all gate stats with effectiveness flags."""
        # Gate that rejects at low confidence = good
        for _ in range(10):
            engine.record_decision("good_gate", rejected=True, confidence=0.45, pair="EUR_USD")
        for _ in range(5):
            engine.record_decision("good_gate", rejected=False, confidence=0.65, pair="EUR_USD")

        # Gate that rejects at high confidence = bad
        for _ in range(10):
            engine.record_decision("bad_gate", rejected=True, confidence=0.72, pair="EUR_USD")

        report = engine.generate_report(journal_path)
        assert "good_gate" in report
        assert "bad_gate" in report
        assert report["good_gate"]["rejections"] == 10
        assert report["good_gate"]["passes"] == 5
        assert report["good_gate"]["effectiveness_flag"] == "PROTECTING_PROFIT"
        assert report["bad_gate"]["effectiveness_flag"] == "KILLING_GOOD_TRADES"

    def test_generate_report_no_rejections_flag(self, engine, journal_path):
        """Gate with only passes gets NO_REJECTIONS flag."""
        engine.record_decision("passive_gate", rejected=False, confidence=0.6)
        report = engine.generate_report(journal_path)
        assert report["passive_gate"]["effectiveness_flag"] == "NO_REJECTIONS"

    def test_persistence_save_load(self, tmp_dir):
        """State round-trips through save/load."""
        path = os.path.join(tmp_dir, "ga_persist.json")
        config = GateAttributionConfig(persistence_path=path)
        engine1 = GateAttributionEngine(config)
        engine1.record_decision("gate_a", rejected=True, confidence=0.55, pair="EUR_USD")
        engine1.record_decision("gate_b", rejected=False, confidence=0.65, pair="GBP_USD")
        engine1.save_state()

        # Load into a new engine
        engine2 = GateAttributionEngine(config)
        assert engine2.get_decisions_count() == 2
        assert sorted(engine2.get_gate_names()) == ["gate_a", "gate_b"]

    def test_persistence_corrupted_file(self, tmp_dir):
        """Corrupted file doesn't crash — starts fresh."""
        path = os.path.join(tmp_dir, "ga_corrupt.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        config = GateAttributionConfig(persistence_path=path)
        engine = GateAttributionEngine(config)
        assert engine.get_decisions_count() == 0

    def test_daily_summary_specific_date(self, engine):
        """Daily summary returns data for specified date."""
        engine.record_decision("gate_a", rejected=True, confidence=0.5)
        # Get today's summary
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        summary = engine.get_daily_summary(today)
        assert "gate_a" in summary

        # Non-existent date returns empty
        assert engine.get_daily_summary("2020-01-01") == {}

    def test_flagged_gates(self, engine, journal_path):
        """get_flagged_gates identifies problematic gates."""
        for _ in range(10):
            engine.record_decision("killer", rejected=True, confidence=0.72)
        for _ in range(10):
            engine.record_decision("protector", rejected=True, confidence=0.45)

        flags = engine.get_flagged_gates(journal_path)
        assert "killer" in flags["killing_good_trades"]
        assert "protector" in flags["protecting_profit"]

    def test_clear(self, engine):
        """clear() removes all data."""
        engine.record_decision("gate_a", rejected=True, confidence=0.5)
        engine.record_decision("gate_b", rejected=False, confidence=0.6)
        engine.clear()
        assert engine.get_decisions_count() == 0
        assert engine.get_gate_names() == []
        assert engine.get_daily_summary() == {}


class TestGateAttributionLazyImports:
    """Verify lazy imports work in __init__.py."""

    def test_gate_attribution_engine_import(self):
        from src.scanner import GateAttributionEngine
        assert GateAttributionEngine is not None

    def test_gate_attribution_config_import(self):
        from src.scanner import GateAttributionConfig
        assert GateAttributionConfig is not None

    def test_gate_decision_import(self):
        from src.scanner import GateDecision
        assert GateDecision is not None

    def test_gate_stats_import(self):
        from src.scanner import GateStats
        assert GateStats is not None


class TestGateAttributionWiring:
    """Verify GateAttributionEngine is wired into execution.py."""

    def test_execution_manager_has_gate_attribution(self):
        """ExecutionManager initializes _gate_attribution attribute."""
        try:
            from src.scanner.execution import ExecutionManager
            em = ExecutionManager.__new__(ExecutionManager)
            # The attribute should be set during __init__, but we can check
            # the source references it
            import inspect
            source = inspect.getsource(ExecutionManager.__init__)
            assert "_gate_attribution" in source
            assert "GateAttributionEngine" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")

    def test_gate_attribution_record_calls_in_execution(self):
        """Verify record_decision calls exist in execute_trade."""
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager)
            # Should have record_decision calls for each gate
            assert source.count("record_decision") >= 8, (
                "Expected at least 8 record_decision calls (pass+reject for 4+ gates)"
            )
            # Verify gate names are tracked
            for gate in ["drawdown_adapter", "spread_filter", "calendar_filter",
                         "fitness_detector", "setup_quality"]:
                assert f'"{gate}"' in source, f"Missing gate attribution for {gate}"
        except ImportError:
            pytest.skip("ExecutionManager not importable")

    def test_save_state_in_sync(self):
        """Verify save_state is called in sync_closed_trades_rl."""
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.sync_closed_trades_rl)
            assert "_gate_attribution" in source
            assert "save_state" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")
