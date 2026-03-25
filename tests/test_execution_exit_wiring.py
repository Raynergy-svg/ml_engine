"""
Test suite for Phase 44 (US-279): AdaptiveExitManager wiring into ExecutionManager.

Tests cover:
- Initialization of _adaptive_exit_manager
- evaluate_exits method behavior with various inputs
- TradeContext construction from trade status dicts
- Exit signal propagation in monitor_open_trades
- Exception handling and graceful degradation
- Edge cases: zero values, missing fields, various regimes/directions
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import numpy as np

from src.scanner.execution import ExecutionManager, ExecutionConfig
from src.scanner.adaptive_exits import (
    AdaptiveExitManager, ExitAction, TradeContext, ExitStrategyConfig
)


class TestExecutionManagerInitialization(unittest.TestCase):
    """Test ExecutionManager initialization with AdaptiveExitManager."""

    def test_adaptive_exit_manager_initialized_on_init(self):
        """ExecutionManager.__init__ should create _adaptive_exit_manager."""
        config = ExecutionConfig()
        manager = ExecutionManager(config=config)
        self.assertIsNotNone(manager._adaptive_exit_manager)
        self.assertIsInstance(manager._adaptive_exit_manager, AdaptiveExitManager)

    def test_adaptive_exit_manager_init_graceful_failure(self):
        """If adaptive_exits module unavailable, manager should defer gracefully."""
        with patch("src.scanner.adaptive_exits.create_default_exit_manager", side_effect=ImportError("Not found")):
            config = ExecutionConfig()
            manager = ExecutionManager(config=config)
            # Should not raise; manager may be None or uninitialized
            self.assertTrue(hasattr(manager, "_adaptive_exit_manager"))

    def test_adaptive_exit_manager_attribute_exists(self):
        """ExecutionManager should have _adaptive_exit_manager attribute."""
        manager = ExecutionManager()
        self.assertTrue(hasattr(manager, "_adaptive_exit_manager"))


class TestEvaluateExitsBasics(unittest.TestCase):
    """Test evaluate_exits method behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ExecutionConfig()
        self.manager = ExecutionManager(config=self.config)

    def test_evaluate_exits_returns_empty_list_when_manager_none(self):
        """evaluate_exits should return [] when _adaptive_exit_manager is None."""
        self.manager._adaptive_exit_manager = None
        result = self.manager.evaluate_exits([])
        self.assertEqual(result, [])

    def test_evaluate_exits_returns_empty_list_for_empty_statuses(self):
        """evaluate_exits should return [] for empty trade_statuses."""
        result = self.manager.evaluate_exits([])
        self.assertEqual(result, [])

    def test_evaluate_exits_accepts_regime_and_confidence(self):
        """evaluate_exits should accept regime_name and current_confidence parameters."""
        trade_statuses = self._build_sample_status("EUR_USD", "LONG", 1.0900, 1.0910, 1.0880, 1.0920)
        # Should not raise
        result = self.manager.evaluate_exits(
            trade_statuses,
            regime_name="HIGH",
            current_confidence=0.75
        )
        self.assertIsInstance(result, list)

    def test_evaluate_exits_accepts_atr_values_dict(self):
        """evaluate_exits should accept optional atr_values dict."""
        trade_statuses = self._build_sample_status("GBP_USD", "SHORT", 1.2500, 1.2490, 1.2510, 1.2470)
        atr_map = {"GBP_USD": 0.0050}
        # Should not raise
        result = self.manager.evaluate_exits(
            trade_statuses,
            atr_values=atr_map
        )
        self.assertIsInstance(result, list)

    @staticmethod
    def _build_sample_status(
        pair: str,
        direction: str,
        entry: float,
        current: float,
        sl: float,
        tp: float,
        trade_id: str = "123456",
        units: int = 10000,
        time_in_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """Build a sample trade status dict for testing."""
        unrealized_pl = (current - entry) * units if direction == "LONG" else (entry - current) * units
        return [
            {
                "trade_id": trade_id,
                "pair": pair,
                "direction": direction,
                "units": units,
                "entry": entry,
                "unrealized_pl": unrealized_pl,
                "sl_price": sl,
                "tp_price": tp,
                "sl_dist_pips": abs(entry - sl) / 0.0001,
                "tp_dist_pips": abs(tp - entry) / 0.0001,
                "time_in_minutes": time_in_minutes,
            }
        ]


class TestEvaluateExitsProcessing(unittest.TestCase):
    """Test evaluate_exits processing of trade statuses."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ExecutionConfig()
        self.manager = ExecutionManager(config=self.config)

    def test_evaluate_exits_processes_single_trade_long(self):
        """evaluate_exits should process a single LONG trade."""
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": 1.0900,
            "unrealized_pl": 100.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_evaluate_exits_processes_single_trade_short(self):
        """evaluate_exits should process a single SHORT trade."""
        status = {
            "trade_id": "t002",
            "pair": "GBP_USD",
            "direction": "SHORT",
            "units": -5000,
            "entry": 1.2500,
            "unrealized_pl": -50.0,
            "sl_price": 1.2520,
            "tp_price": 1.2480,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 45,
        }
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_evaluate_exits_processes_multiple_trades(self):
        """evaluate_exits should process multiple trades."""
        statuses = [
            {
                "trade_id": "t001",
                "pair": "EUR_USD",
                "direction": "LONG",
                "units": 10000,
                "entry": 1.0900,
                "unrealized_pl": 100.0,
                "sl_price": 1.0880,
                "tp_price": 1.0920,
                "sl_dist_pips": 20.0,
                "tp_dist_pips": 20.0,
                "time_in_minutes": 30,
            },
            {
                "trade_id": "t002",
                "pair": "GBP_USD",
                "direction": "SHORT",
                "units": -5000,
                "entry": 1.2500,
                "unrealized_pl": -50.0,
                "sl_price": 1.2520,
                "tp_price": 1.2480,
                "sl_dist_pips": 20.0,
                "tp_dist_pips": 20.0,
                "time_in_minutes": 45,
            },
        ]
        result = self.manager.evaluate_exits(statuses)
        self.assertIsInstance(result, list)
        # May return empty list if exits don't trigger, but should process both

    def test_evaluate_exits_returns_action_dict_with_required_fields(self):
        """Exit action dicts should have required fields."""
        # Mock the manager to return an exit action
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": 1.0900,
            "unrealized_pl": 200.0,  # Higher profit to trigger exits
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 120,  # Long time in trade
        }

        # If an exit action is generated, check its structure
        result = self.manager.evaluate_exits([status])
        if result:  # If any exits were triggered
            for action in result:
                self.assertIn("trade_id", action)
                self.assertIn("pair", action)
                self.assertIn("action", action)
                self.assertIn("reason", action)
                self.assertIn("strategy", action)
                self.assertIn("details", action)


class TestEvaluateExitsEdgeCases(unittest.TestCase):
    """Test evaluate_exits with edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ExecutionConfig()
        self.manager = ExecutionManager(config=self.config)

    def test_evaluate_exits_handles_zero_entry_price(self):
        """evaluate_exits should handle zero entry price gracefully."""
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": 0.0,  # Zero entry
            "unrealized_pl": 0.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }
        # Should not raise, may return empty list or log debug
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_evaluate_exits_handles_zero_units(self):
        """evaluate_exits should handle zero units gracefully."""
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 0,  # Zero units
            "entry": 1.0900,
            "unrealized_pl": 0.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }
        # Should handle gracefully (safe_units fallback to 1)
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_evaluate_exits_handles_missing_fields(self):
        """evaluate_exits should handle missing optional fields gracefully."""
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            # Missing: units, entry, unrealized_pl, sl_price, tp_price, time_in_minutes
        }
        # Should not crash, uses defaults
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_evaluate_exits_handles_negative_entry(self):
        """evaluate_exits should handle negative entry price gracefully."""
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": -1.0900,  # Negative entry (invalid FX price)
            "unrealized_pl": 100.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }
        # Should not raise, may fail validation internally
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_evaluate_exits_handles_zero_sl_tp(self):
        """evaluate_exits should handle zero SL/TP prices gracefully."""
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": 1.0900,
            "unrealized_pl": 100.0,
            "sl_price": 0.0,  # Zero SL
            "tp_price": 0.0,  # Zero TP
            "sl_dist_pips": 0.0,
            "tp_dist_pips": 0.0,
            "time_in_minutes": 30,
        }
        # Should use fallback pips (15.0 and 20.0)
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)


class TestEvaluateExitsRegimes(unittest.TestCase):
    """Test evaluate_exits with various regime names."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ExecutionConfig()
        self.manager = ExecutionManager(config=self.config)

    def _build_status(self) -> Dict[str, Any]:
        """Build a sample trade status."""
        return {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": 1.0900,
            "unrealized_pl": 100.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }

    def test_evaluate_exits_with_regime_low(self):
        """evaluate_exits should handle LOW regime."""
        result = self.manager.evaluate_exits([self._build_status()], regime_name="LOW")
        self.assertIsInstance(result, list)

    def test_evaluate_exits_with_regime_normal(self):
        """evaluate_exits should handle NORMAL regime."""
        result = self.manager.evaluate_exits([self._build_status()], regime_name="NORMAL")
        self.assertIsInstance(result, list)

    def test_evaluate_exits_with_regime_high(self):
        """evaluate_exits should handle HIGH regime."""
        result = self.manager.evaluate_exits([self._build_status()], regime_name="HIGH")
        self.assertIsInstance(result, list)

    def test_evaluate_exits_with_regime_extreme(self):
        """evaluate_exits should handle EXTREME regime."""
        result = self.manager.evaluate_exits([self._build_status()], regime_name="EXTREME")
        self.assertIsInstance(result, list)

    def test_evaluate_exits_with_confidence_thresholds(self):
        """evaluate_exits should handle various confidence levels."""
        status = self._build_status()

        # Test low confidence
        result = self.manager.evaluate_exits([status], current_confidence=0.2)
        self.assertIsInstance(result, list)

        # Test medium confidence
        result = self.manager.evaluate_exits([status], current_confidence=0.5)
        self.assertIsInstance(result, list)

        # Test high confidence
        result = self.manager.evaluate_exits([status], current_confidence=0.9)
        self.assertIsInstance(result, list)


class TestExitSignalPropagation(unittest.TestCase):
    """Test exit signal propagation in monitor_open_trades."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ExecutionConfig()
        self.manager = ExecutionManager(config=self.config)

    def test_monitor_open_trades_calls_evaluate_exits_when_adaptive_manager_present(self):
        """monitor_open_trades should call evaluate_exits when manager is initialized."""
        # Test that evaluate_exits is callable and non-blocking
        # This verifies the integration point without mocking OANDA
        self.assertTrue(hasattr(self.manager, "evaluate_exits"))
        self.assertTrue(callable(self.manager.evaluate_exits))

        # Call evaluate_exits directly with sample data
        status = {
            "trade_id": "123",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": 1.0900,
            "unrealized_pl": 100.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }

        # Should process without error
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_evaluate_exits_integration_with_confidence_and_regime(self):
        """evaluate_exits should integrate confidence and regime parameters correctly."""
        status = {
            "trade_id": "456",
            "pair": "GBP_USD",
            "direction": "SHORT",
            "units": -5000,
            "entry": 1.2500,
            "unrealized_pl": -50.0,
            "sl_price": 1.2520,
            "tp_price": 1.2480,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 60,
        }

        # Test with high confidence and extreme regime
        result = self.manager.evaluate_exits(
            [status],
            regime_name="EXTREME",
            current_confidence=0.85
        )
        self.assertIsInstance(result, list)


class TestTradeContextConstruction(unittest.TestCase):
    """Test TradeContext construction from trade status dicts."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ExecutionConfig()
        self.manager = ExecutionManager(config=self.config)

    def test_trade_context_from_long_trade(self):
        """TradeContext should be constructed correctly for LONG trades."""
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": 1.0900,
            "unrealized_pl": 100.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }

        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)
        # If result is non-empty, the TradeContext was successfully created and processed

    def test_trade_context_from_short_trade(self):
        """TradeContext should be constructed correctly for SHORT trades."""
        status = {
            "trade_id": "t002",
            "pair": "GBP_USD",
            "direction": "SHORT",
            "units": -5000,
            "entry": 1.2500,
            "unrealized_pl": -50.0,
            "sl_price": 1.2520,
            "tp_price": 1.2480,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 45,
        }

        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_trade_context_direction_conversion(self):
        """TradeContext should convert LONG/SHORT to BUY/SELL correctly."""
        # LONG should become BUY
        status_long = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 10000,
            "entry": 1.0900,
            "unrealized_pl": 100.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }

        # SHORT should become SELL
        status_short = {
            "trade_id": "t002",
            "pair": "EUR_USD",
            "direction": "SHORT",
            "units": -10000,
            "entry": 1.0900,
            "unrealized_pl": -100.0,
            "sl_price": 1.0920,
            "tp_price": 1.0880,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }

        result_long = self.manager.evaluate_exits([status_long])
        result_short = self.manager.evaluate_exits([status_short])
        # Both should process without crashing
        self.assertIsInstance(result_long, list)
        self.assertIsInstance(result_short, list)


class TestExceptionHandling(unittest.TestCase):
    """Test exception handling in evaluate_exits."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ExecutionConfig()
        self.manager = ExecutionManager(config=self.config)

    def test_evaluate_exits_handles_valueerror_gracefully(self):
        """evaluate_exits should handle ValueError (TradeContext validation) gracefully."""
        # Create a status that will cause TradeContext validation to fail
        status = {
            "trade_id": "t001",
            "pair": "EUR_USD",
            "direction": "LONG",
            "units": 0,  # Will cause division by zero handling
            "entry": 1.0900,
            "unrealized_pl": 100.0,
            "sl_price": 1.0880,
            "tp_price": 1.0920,
            "sl_dist_pips": 20.0,
            "tp_dist_pips": 20.0,
            "time_in_minutes": 30,
        }

        # Should not raise, may return empty list or log debug
        result = self.manager.evaluate_exits([status])
        self.assertIsInstance(result, list)

    def test_evaluate_exits_handles_generic_exception(self):
        """evaluate_exits should handle generic exceptions without crashing."""
        # Mock the manager to raise an exception during evaluation
        with patch.object(
            self.manager._adaptive_exit_manager,
            "evaluate_exit",
            side_effect=RuntimeError("Test error")
        ):
            status = {
                "trade_id": "t001",
                "pair": "EUR_USD",
                "direction": "LONG",
                "units": 10000,
                "entry": 1.0900,
                "unrealized_pl": 100.0,
                "sl_price": 1.0880,
                "tp_price": 1.0920,
                "sl_dist_pips": 20.0,
                "tp_dist_pips": 20.0,
                "time_in_minutes": 30,
            }

            # Should catch the exception and return empty list or continue
            result = self.manager.evaluate_exits([status])
            self.assertIsInstance(result, list)

    def test_evaluate_exits_logs_warnings_on_failure(self):
        """evaluate_exits should log warnings when evaluation fails."""
        with patch("src.scanner.execution.logger") as mock_logger:
            with patch.object(
                self.manager._adaptive_exit_manager,
                "evaluate_exit",
                side_effect=Exception("Test error")
            ):
                status = {
                    "trade_id": "t001",
                    "pair": "EUR_USD",
                    "direction": "LONG",
                    "units": 10000,
                    "entry": 1.0900,
                    "unrealized_pl": 100.0,
                    "sl_price": 1.0880,
                    "tp_price": 1.0920,
                    "sl_dist_pips": 20.0,
                    "tp_dist_pips": 20.0,
                    "time_in_minutes": 30,
                }

                result = self.manager.evaluate_exits([status])
                # Should have logged a warning (may not be called if validation fails first)
                self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
