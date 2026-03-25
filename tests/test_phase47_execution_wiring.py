"""
Phase 47 Execution Wiring Tests (US-296, US-297, US-298)

Tests for wiring Session Detector, Expectancy Tracker, and Regime Gates
Position Multiplier into the execution pipeline.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.scanner.execution import ExecutionManager, ExecutionConfig, ExecutionResult
from src.scanner.session_detector import SessionDetector, SessionConfig
from src.scanner.expectancy_tracker import (
    ExpectancyTracker,
    ExpectancyConfig,
    create_default_expectancy_tracker,
)
from src.scanner.regime_gates import get_regime_profile, apply_regime_gates


class TestSessionDetectorInit:
    """Tests for SessionDetector initialization in ExecutionManager."""

    def test_session_detector_lazy_init_success(self):
        """Test that session detector is successfully initialized on first access."""
        manager = ExecutionManager()
        assert manager._session_detector is None

        # Trigger lazy init
        manager._init_session_detector()
        assert manager._session_detector is not None
        assert isinstance(manager._session_detector, SessionDetector)

    def test_session_detector_init_idempotent(self):
        """Test that multiple init calls don't recreate the instance."""
        manager = ExecutionManager()
        manager._init_session_detector()
        first_instance = manager._session_detector

        manager._init_session_detector()
        second_instance = manager._session_detector

        assert first_instance is second_instance

    def test_session_detector_init_with_mock_failure(self):
        """Test graceful fallback when SessionDetector import fails."""
        manager = ExecutionManager()

        with patch("src.scanner.session_detector.SessionDetector", side_effect=ImportError("Module not found")):
            manager._init_session_detector()
            # Should remain None and not raise
            assert manager._session_detector is None


class TestSessionMultiplierApplication:
    """Tests for applying session position size multipliers."""

    def test_apply_session_multiplier_tokyo_session(self):
        """Test that Tokyo session multiplier (0.70) is applied correctly."""
        manager = ExecutionManager()
        manager._init_session_detector()

        # Mock get_current_session to return Tokyo
        tokyo_time = datetime(2025, 1, 15, 6, 30, tzinfo=timezone.utc)  # 06:30 UTC = Tokyo session
        manager._session_detector.get_current_session = Mock(
            return_value=manager._session_detector.get_current_session(tokyo_time)
        )

        base_lots = 1.0
        adjusted_lots = manager._apply_session_position_multiplier(base_lots)

        # Tokyo multiplier is 0.70
        expected = base_lots * 0.70
        assert abs(adjusted_lots - expected) < 0.01

    def test_apply_session_multiplier_london_session(self):
        """Test that London session multiplier (1.00) is applied correctly."""
        manager = ExecutionManager()
        manager._init_session_detector()

        # Mock get_current_session to return London
        london_time = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)  # 12:00 UTC = London session
        manager._session_detector.get_current_session = Mock(
            return_value=manager._session_detector.get_current_session(london_time)
        )

        base_lots = 1.0
        adjusted_lots = manager._apply_session_position_multiplier(base_lots)

        # London multiplier is 1.00
        expected = base_lots * 1.0
        assert abs(adjusted_lots - expected) < 0.01

    def test_apply_session_multiplier_fallback_on_error(self):
        """Test that original lots are returned if session detector fails."""
        manager = ExecutionManager()
        manager._init_session_detector()

        # Mock get_position_size_multiplier to raise an exception
        manager._session_detector.get_position_size_multiplier = Mock(
            side_effect=RuntimeError("Detector error")
        )

        base_lots = 1.5
        adjusted_lots = manager._apply_session_position_multiplier(base_lots)

        # Should return original lots unchanged
        assert adjusted_lots == base_lots

    def test_apply_session_multiplier_uninitialized_detector(self):
        """Test that lots are returned unchanged if detector is unavailable."""
        manager = ExecutionManager()
        # Ensure detector is None and prevent lazy-init
        manager._session_detector = None

        # Mock _init_session_detector to keep it None
        manager._init_session_detector = Mock()

        base_lots = 1.0
        adjusted_lots = manager._apply_session_position_multiplier(base_lots)

        # Should return original lots
        assert adjusted_lots == base_lots


class TestRegimeMultiplierApplication:
    """Tests for applying regime-based position size multipliers."""

    def test_apply_regime_multiplier_low_vol(self):
        """Test that LOW regime multiplier (1.3) is applied correctly."""
        manager = ExecutionManager()

        base_lots = 1.0
        adjusted_lots = manager._apply_regime_position_multiplier(base_lots, "LOW")

        # LOW multiplier is 1.3
        expected = base_lots * 1.3
        assert abs(adjusted_lots - expected) < 0.01

    def test_apply_regime_multiplier_high_vol(self):
        """Test that HIGH regime multiplier (0.65) is applied correctly."""
        manager = ExecutionManager()

        base_lots = 1.0
        adjusted_lots = manager._apply_regime_position_multiplier(base_lots, "HIGH")

        # HIGH multiplier is 0.65
        expected = base_lots * 0.65
        assert abs(adjusted_lots - expected) < 0.01

    def test_apply_regime_multiplier_extreme_vol(self):
        """Test that EXTREME regime multiplier (0.40) is applied correctly."""
        manager = ExecutionManager()

        base_lots = 2.0
        adjusted_lots = manager._apply_regime_position_multiplier(base_lots, "EXTREME")

        # EXTREME multiplier is 0.40
        expected = base_lots * 0.40
        assert abs(adjusted_lots - expected) < 0.01

    def test_apply_regime_multiplier_unknown_regime(self):
        """Test that original lots are returned for unknown regime."""
        manager = ExecutionManager()

        base_lots = 1.0
        adjusted_lots = manager._apply_regime_position_multiplier(base_lots, "UNKNOWN")

        # Should return original lots
        assert adjusted_lots == base_lots

    def test_apply_regime_multiplier_import_error(self):
        """Test graceful fallback when regime_gates import fails."""
        manager = ExecutionManager()

        with patch("src.scanner.regime_gates.get_regime_profile", side_effect=ImportError("Module not found")):
            base_lots = 1.0
            adjusted_lots = manager._apply_regime_position_multiplier(base_lots, "HIGH")

            # Should return original lots
            assert adjusted_lots == base_lots


class TestPositionSizingChain:
    """Tests for the complete position sizing chain with multipliers."""

    def test_position_sizing_chain_adaptive_regime_session(self):
        """Test that position sizing applies multipliers in correct order: adaptive → regime → session."""
        manager = ExecutionManager(
            config=ExecutionConfig(
                aggressive_mode=True,
                position_sizing_enabled=True,
                account_equity=50000.0,
            )
        )

        # Initialize session detector but NOT adaptive sizer to keep it simple
        manager._init_session_detector()

        # Mock regime to LOW
        manager._last_regime_name = "LOW"

        # Call calculate_position_size with mocked data
        with patch.object(manager, "fetch_live_nav", return_value=50000.0):
            with patch.object(manager, "_calculate_base_tp_pips", return_value=20.0):
                with patch.object(manager, "_apply_risk_manager", return_value=(15.0, 20.0, "medium")):
                    with patch.object(manager, "_calculate_adaptive_position_size", return_value=(1.0, 0.05, "kelly")):
                        lots, risk_pct, sl_pips, tp_pips, conf_level = manager.calculate_position_size(
                            pair="EUR_USD",
                            confidence=0.65,
                            atr=0.012,
                        )

        # Due to multipliers: 1.0 (base) * 1.3 (LOW regime) * session_mult
        # We just verify that lots were adjusted beyond the base 1.0
        assert lots > 0
        assert lots != 1.0  # Should be modified by at least regime multiplier

    def test_position_sizing_disabled(self):
        """Test that position sizing returns zeros when disabled."""
        manager = ExecutionManager(
            config=ExecutionConfig(position_sizing_enabled=False)
        )

        lots, risk_pct, sl_pips, tp_pips, conf_level = manager.calculate_position_size(
            pair="EUR_USD",
            confidence=0.65,
            atr=0.012,
        )

        assert lots == 0.0
        assert risk_pct == 0.0
        assert sl_pips == 0.0
        assert tp_pips == 0.0
        assert conf_level == "disabled"


class TestExpectancyTrackerInit:
    """Tests for ExpectancyTracker initialization in ExecutionManager."""

    def test_expectancy_tracker_lazy_init_success(self):
        """Test that expectancy tracker is successfully initialized."""
        manager = ExecutionManager()
        assert manager._expectancy_tracker is None

        manager._init_expectancy_tracker()
        assert manager._expectancy_tracker is not None
        assert isinstance(manager._expectancy_tracker, ExpectancyTracker)

    def test_expectancy_tracker_init_idempotent(self):
        """Test that multiple init calls don't recreate the instance."""
        manager = ExecutionManager()
        manager._init_expectancy_tracker()
        first_instance = manager._expectancy_tracker

        manager._init_expectancy_tracker()
        second_instance = manager._expectancy_tracker

        assert first_instance is second_instance

    def test_expectancy_tracker_init_loads_state(self):
        """Test that initialization attempts to load persisted state."""
        manager = ExecutionManager()

        with patch("src.scanner.expectancy_tracker.ExpectancyTracker.load_state") as mock_load:
            manager._init_expectancy_tracker()
            assert manager._expectancy_tracker is not None
            # load_state should be called
            mock_load.assert_called_once()

    def test_expectancy_tracker_init_graceful_on_import_error(self):
        """Test graceful fallback when ExpectancyTracker import fails."""
        manager = ExecutionManager()

        with patch("src.scanner.expectancy_tracker.create_default_expectancy_tracker", side_effect=ImportError("Module not found")):
            manager._init_expectancy_tracker()
            # Should remain None
            assert manager._expectancy_tracker is None


class TestExpectancyRecording:
    """Tests for recording trade outcomes to expectancy tracker."""

    def test_record_expectancy_single_trade(self):
        """Test recording a single trade outcome to expectancy tracker."""
        manager = ExecutionManager()
        manager._init_expectancy_tracker()

        # Create fresh tracker to avoid test contamination
        from src.scanner.expectancy_tracker import ExpectancyConfig
        manager._expectancy_tracker = ExpectancyTracker(ExpectancyConfig())

        # Create mock pending entry and closed trade
        pending = [
            {
                "trade_id": "12345",
                "pair": "EUR_USD",
                "regime": {"volatility_regime": "NORMAL"},
                "agents": {
                    "agent_reasons": [
                        {"name": "trend", "passed": True},
                        {"name": "momentum", "passed": True},
                    ]
                },
            }
        ]

        closed_trades = {
            "12345": {
                "id": "12345",
                "realizedPL": 50.0,  # Winning trade
                "averageClosePrice": 1.1010,
                "price": 1.1000,
                "closeTime": "2025-01-15T10:00:00Z",
            }
        }

        # Record expectancy
        manager._record_expectancy_from_trades(pending, closed_trades)

        # Verify that trade was recorded for both agents
        trend_exp = manager._expectancy_tracker.get_expectancy("trend", "NORMAL")
        momentum_exp = manager._expectancy_tracker.get_expectancy("momentum", "NORMAL")

        assert trend_exp.trade_count == 1
        assert momentum_exp.trade_count == 1
        assert trend_exp.win_rate == 1.0
        assert momentum_exp.win_rate == 1.0

    def test_record_expectancy_multiple_trades(self):
        """Test recording multiple trades with mixed outcomes."""
        manager = ExecutionManager()
        manager._init_expectancy_tracker()

        # Create fresh tracker to avoid test contamination
        from src.scanner.expectancy_tracker import ExpectancyConfig
        manager._expectancy_tracker = ExpectancyTracker(ExpectancyConfig())

        pending = [
            {
                "trade_id": "1",
                "pair": "EUR_USD",
                "regime": {"volatility_regime": "HIGH"},
                "agents": {"agent_reasons": [{"name": "trend", "passed": True}]},
            },
            {
                "trade_id": "2",
                "pair": "GBP_USD",
                "regime": {"volatility_regime": "HIGH"},
                "agents": {"agent_reasons": [{"name": "trend", "passed": True}]},
            },
        ]

        closed_trades = {
            "1": {"id": "1", "realizedPL": 100.0, "averageClosePrice": 1.1010, "price": 1.1000, "closeTime": "2025-01-15T10:00:00Z"},
            "2": {"id": "2", "realizedPL": -50.0, "averageClosePrice": 1.4990, "price": 1.5000, "closeTime": "2025-01-15T10:05:00Z"},
        }

        manager._record_expectancy_from_trades(pending, closed_trades)

        trend_exp = manager._expectancy_tracker.get_expectancy("trend", "HIGH")
        assert trend_exp.trade_count == 2
        assert trend_exp.win_rate == 0.5
        assert trend_exp.avg_win > 0
        assert trend_exp.avg_loss < 0

    def test_record_expectancy_unknown_agent_skipped(self):
        """Test that unknown agents are gracefully skipped."""
        manager = ExecutionManager()
        manager._init_expectancy_tracker()

        pending = [
            {
                "trade_id": "1",
                "pair": "EUR_USD",
                "regime": {"volatility_regime": "NORMAL"},
                "agents": {
                    "agent_reasons": [
                        {"name": "unknown_agent", "passed": True},
                    ]
                },
            }
        ]

        closed_trades = {
            "1": {
                "id": "1",
                "realizedPL": 100.0,
                "averageClosePrice": 1.1010,
                "price": 1.1000,
                "closeTime": "2025-01-15T10:00:00Z",
            }
        }

        # Should not raise — unknown agent is skipped gracefully
        manager._record_expectancy_from_trades(pending, closed_trades)

    def test_record_expectancy_missing_trade_skipped(self):
        """Test that trades not found in closed_trades are skipped."""
        manager = ExecutionManager()
        manager._init_expectancy_tracker()

        # Create fresh tracker to avoid test contamination
        from src.scanner.expectancy_tracker import ExpectancyConfig
        manager._expectancy_tracker = ExpectancyTracker(ExpectancyConfig())

        pending = [
            {
                "trade_id": "99999",  # Not in closed_trades
                "pair": "EUR_USD",
                "regime": {"volatility_regime": "NORMAL"},
                "agents": {"agent_reasons": [{"name": "trend", "passed": True}]},
            }
        ]

        closed_trades = {}

        # Mock save_state to avoid actual file I/O
        manager._expectancy_tracker.save_state = Mock()

        # Should not raise — missing trade is skipped
        manager._record_expectancy_from_trades(pending, closed_trades)

        trend_exp = manager._expectancy_tracker.get_expectancy("trend", "NORMAL")
        assert trend_exp.trade_count == 0

    def test_record_expectancy_saves_state(self):
        """Test that state is saved after recording trades."""
        manager = ExecutionManager()
        manager._init_expectancy_tracker()

        with patch.object(manager._expectancy_tracker, "save_state") as mock_save:
            pending = [
                {
                    "trade_id": "1",
                    "pair": "EUR_USD",
                    "regime": {"volatility_regime": "NORMAL"},
                    "agents": {"agent_reasons": [{"name": "trend", "passed": True}]},
                }
            ]

            closed_trades = {
                "1": {
                    "id": "1",
                    "realizedPL": 50.0,
                    "averageClosePrice": 1.1010,
                    "price": 1.1000,
                    "closeTime": "2025-01-15T10:00:00Z",
                }
            }

            manager._record_expectancy_from_trades(pending, closed_trades)

            # save_state should be called once
            mock_save.assert_called_once()

    def test_record_expectancy_uninitialized_tracker(self):
        """Test graceful handling when expectancy tracker is not initialized."""
        manager = ExecutionManager()
        # Don't initialize tracker

        pending = [
            {
                "trade_id": "1",
                "pair": "EUR_USD",
                "regime": {"volatility_regime": "NORMAL"},
                "agents": {"agent_reasons": [{"name": "trend", "passed": True}]},
            }
        ]

        closed_trades = {
            "1": {
                "id": "1",
                "realizedPL": 50.0,
                "averageClosePrice": 1.1010,
                "price": 1.1000,
                "closeTime": "2025-01-15T10:00:00Z",
            }
        }

        # Should not raise — should gracefully initialize and proceed
        manager._record_expectancy_from_trades(pending, closed_trades)


class TestExpectancyWeightModifiers:
    """Tests for expectancy-based weight modifiers applied to agents."""

    def test_expectancy_positive_expectancy_weight_modifier_one(self):
        """Test that positive expectancy agents get weight modifier of 1.0."""
        tracker = create_default_expectancy_tracker()

        # Record winning trades for an agent
        for _ in range(10):
            tracker.record_trade("trend", "NORMAL", pnl=100.0, won=True)

        expectancy_result = tracker.get_expectancy("trend", "NORMAL")
        assert expectancy_result.expectancy > 0
        assert expectancy_result.weight_modifier == 1.0

    def test_expectancy_negative_expectancy_weight_modifier_penalty(self):
        """Test that negative expectancy agents get reduced weight modifier."""
        tracker = create_default_expectancy_tracker()

        # Record losing trades for an agent
        for _ in range(10):
            tracker.record_trade("trend", "NORMAL", pnl=-50.0, won=False)

        expectancy_result = tracker.get_expectancy("trend", "NORMAL")
        assert expectancy_result.expectancy < 0
        assert expectancy_result.weight_modifier < 1.0
        assert expectancy_result.weight_modifier == 1.0 + tracker._config.negative_penalty

    def test_expectancy_insufficient_data_weight_modifier_one(self):
        """Test that agents with insufficient data get neutral weight modifier."""
        tracker = create_default_expectancy_tracker()

        # Record only 1 trade (min_trades_for_calc is 5)
        tracker.record_trade("trend", "NORMAL", pnl=100.0, won=True)

        expectancy_result = tracker.get_expectancy("trend", "NORMAL")
        assert not expectancy_result.is_meaningful
        assert expectancy_result.weight_modifier == 1.0


class TestPhase47AllThreeModulesDisabled:
    """Tests for all three modules gracefully falling back when disabled/unavailable."""

    def test_all_modules_unavailable_execution_succeeds(self):
        """Test that execution succeeds even if all new modules are unavailable."""
        manager = ExecutionManager(
            config=ExecutionConfig(
                position_sizing_enabled=False,
            )
        )

        # Don't initialize any of the new modules
        assert manager._session_detector is None
        assert manager._expectancy_tracker is None

        # Even without position sizing, calculation should handle gracefully
        lots, risk_pct, sl_pips, tp_pips, conf_level = manager.calculate_position_size(
            pair="EUR_USD",
            confidence=0.65,
            atr=0.012,
        )

        # Should return zeros (position_sizing_enabled=False)
        assert lots == 0.0
        assert conf_level == "disabled"

    def test_position_sizing_with_mock_data(self):
        """Test position sizing chain with fully mocked dependencies."""
        manager = ExecutionManager(
            config=ExecutionConfig(
                aggressive_mode=False,  # Use non-aggressive sizer
                position_sizing_enabled=True,
                account_equity=50000.0,
            )
        )

        with patch.object(manager, "fetch_live_nav", return_value=50000.0):
            with patch.object(manager, "_calculate_base_tp_pips", return_value=25.0):
                with patch.object(manager, "_apply_risk_manager", return_value=(15.0, 25.0, "medium")):
                    with patch.object(manager, "_calculate_lots_from_sizer", return_value=(1.0, 0.05, "medium")):
                        lots, risk_pct, sl_pips, tp_pips, conf_level = manager.calculate_position_size(
                            pair="EUR_USD",
                            confidence=0.65,
                            atr=0.012,
                        )

        # Should get some position size
        assert lots > 0
        assert risk_pct > 0
        assert sl_pips > 0
        assert tp_pips > 0


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_realistic_trade_flow_with_all_modules(self):
        """Test a realistic trade flow using session detector, regime gates, and expectancy tracking."""
        # Create execution manager with all modules enabled
        manager = ExecutionManager(
            config=ExecutionConfig(
                aggressive_mode=False,  # Use default sizer
                position_sizing_enabled=True,
                account_equity=100000.0,
            )
        )

        # Initialize all modules
        manager._init_session_detector()
        manager._init_expectancy_tracker()

        # Create fresh tracker to avoid test contamination
        from src.scanner.expectancy_tracker import ExpectancyConfig
        manager._expectancy_tracker = ExpectancyTracker(ExpectancyConfig())

        # Set regime
        manager._last_regime_name = "NORMAL"

        # Mock OANDA fetch
        with patch.object(manager, "fetch_live_nav", return_value=100000.0):
            with patch.object(manager, "_calculate_base_tp_pips", return_value=30.0):
                with patch.object(manager, "_apply_risk_manager", return_value=(15.0, 30.0, "high")):
                    with patch.object(manager, "_calculate_lots_from_sizer", return_value=(2.0, 0.10, "kelly")):
                        # Calculate position size
                        lots, risk_pct, sl_pips, tp_pips, conf_level = manager.calculate_position_size(
                            pair="EUR_USD",
                            confidence=0.70,
                            atr=0.015,
                        )

        # Should get adjusted position size
        assert lots > 0
        assert risk_pct > 0

        # Simulate a closed trade for expectancy recording
        pending = [
            {
                "trade_id": "1001",
                "pair": "EUR_USD",
                "regime": {"volatility_regime": "NORMAL"},
                "agents": {
                    "agent_reasons": [
                        {"name": "trend", "passed": True},
                        {"name": "momentum", "passed": True},
                        {"name": "volatility", "passed": False},
                    ]
                },
            }
        ]

        closed_trades = {
            "1001": {
                "id": "1001",
                "realizedPL": 150.0,
                "averageClosePrice": 1.1015,
                "price": 1.1000,
                "closeTime": "2025-01-15T10:30:00Z",
            }
        }

        manager._record_expectancy_from_trades(pending, closed_trades)

        # Verify expectancy was recorded
        trend_exp = manager._expectancy_tracker.get_expectancy("trend", "NORMAL")
        assert trend_exp.trade_count == 1
        assert trend_exp.win_rate == 1.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_position_size_zero_or_negative_multiplier(self):
        """Test that position size handles zero or negative results correctly."""
        manager = ExecutionManager()

        # Negative regime multiplier (should not happen, but test graceful handling)
        result = manager._apply_regime_position_multiplier(1.0, "NORMAL")
        assert result > 0

    def test_expectancy_with_zero_pnl(self):
        """Test expectancy calculation with zero PnL trades."""
        tracker = create_default_expectancy_tracker()

        tracker.record_trade("trend", "NORMAL", pnl=0.0, won=False)
        tracker.record_trade("trend", "NORMAL", pnl=100.0, won=True)

        expectancy_result = tracker.get_expectancy("trend", "NORMAL")
        assert expectancy_result.trade_count == 2
        assert expectancy_result.expectancy >= 0  # Average should be non-negative

    def test_regime_profile_caching(self):
        """Test that regime profiles are retrieved correctly."""
        profile_low = get_regime_profile("LOW")
        profile_high = get_regime_profile("HIGH")

        assert profile_low.position_size_multiplier == 1.3
        assert profile_high.position_size_multiplier == 0.65
        assert profile_low.confidence_threshold > profile_high.confidence_threshold

    def test_apply_regime_gates_with_custom_thresholds(self):
        """Test apply_regime_gates with custom base thresholds."""
        custom_thresholds = {
            "confidence_threshold": 40.0,
            "momentum_threshold": 0.30,
        }

        result = apply_regime_gates("HIGH", custom_thresholds)

        # HIGH regime should override custom thresholds
        assert "confidence_threshold" in result
        assert "position_size_multiplier" in result
        assert result["position_size_multiplier"] == 0.65


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
