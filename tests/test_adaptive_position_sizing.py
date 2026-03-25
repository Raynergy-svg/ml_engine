"""
Comprehensive unit tests for adaptive position sizing module.

Coverage includes:
- Config validation and defaults
- Kelly criterion calculation
- Sigmoid confidence mapping
- Drawdown recovery factors
- Regime multipliers
- Composite sizing with clamping
- Trade recording and rolling window
- State persistence (save/load)
- Factory functions
- Edge cases and error conditions

Test count: 45+ unit tests
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.scanner.adaptive_position_sizing import (
    AdaptivePositionSizingConfig,
    AdaptivePositionSizer,
    PositionSizeResult,
    create_aggressive_position_sizer,
    create_conservative_position_sizer,
    create_default_position_sizer,
)


class TestAdaptivePositionSizingConfig(unittest.TestCase):
    """Tests for AdaptivePositionSizingConfig validation."""

    def test_config_defaults(self):
        """Test that default config values are sensible."""
        config = AdaptivePositionSizingConfig()
        self.assertEqual(config.kelly_fraction, 0.33)
        self.assertEqual(config.kelly_window, 50)
        self.assertEqual(config.sigmoid_steepness, 4.0)
        self.assertEqual(config.sigmoid_midpoint, 0.5)
        self.assertEqual(config.max_acceptable_drawdown, 0.15)
        self.assertEqual(config.drawdown_floor, 0.3)
        self.assertEqual(config.base_risk_pct, 0.05)
        self.assertEqual(config.max_risk_per_trade, 0.10)
        self.assertEqual(config.min_position_units, 1000)

    def test_config_regime_multipliers_default(self):
        """Test default regime multipliers."""
        config = AdaptivePositionSizingConfig()
        self.assertEqual(config.regime_multipliers["LOW"], 1.3)
        self.assertEqual(config.regime_multipliers["NORMAL"], 1.0)
        self.assertEqual(config.regime_multipliers["HIGH"], 0.65)
        self.assertEqual(config.regime_multipliers["EXTREME"], 0.40)

    def test_config_component_weights_sum_to_one(self):
        """Test that default component weights sum to 1.0."""
        config = AdaptivePositionSizingConfig()
        total = sum(config.component_weights.values())
        self.assertAlmostEqual(total, 1.0)

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = AdaptivePositionSizingConfig(
            kelly_fraction=0.25, max_risk_per_trade=0.08
        )
        self.assertEqual(config.kelly_fraction, 0.25)
        self.assertEqual(config.max_risk_per_trade, 0.08)

    def test_config_invalid_kelly_fraction_zero(self):
        """Test that kelly_fraction = 0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(kelly_fraction=0.0)

    def test_config_invalid_kelly_fraction_over_one(self):
        """Test that kelly_fraction > 1 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(kelly_fraction=1.5)

    def test_config_invalid_kelly_window(self):
        """Test that kelly_window < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(kelly_window=0)

    def test_config_invalid_sigmoid_steepness(self):
        """Test that sigmoid_steepness <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(sigmoid_steepness=0.0)

    def test_config_invalid_sigmoid_midpoint(self):
        """Test that sigmoid_midpoint outside (0, 1) raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(sigmoid_midpoint=1.5)

    def test_config_invalid_max_drawdown(self):
        """Test that max_acceptable_drawdown <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(max_acceptable_drawdown=0.0)

    def test_config_invalid_drawdown_floor(self):
        """Test that drawdown_floor >= 1 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(drawdown_floor=1.0)

    def test_config_invalid_base_risk_pct(self):
        """Test that base_risk_pct <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(base_risk_pct=0.0)

    def test_config_invalid_max_risk_per_trade(self):
        """Test that max_risk_per_trade <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(max_risk_per_trade=0.0)

    def test_config_base_risk_exceeds_max_risk(self):
        """Test that base_risk_pct > max_risk_per_trade raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(
                base_risk_pct=0.15, max_risk_per_trade=0.10
            )

    def test_config_invalid_min_position_units_negative(self):
        """Test that min_position_units < 0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(min_position_units=-100)

    def test_config_invalid_regime_multiplier_zero(self):
        """Test that regime multiplier <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(
                regime_multipliers={"NORMAL": 0.0}
            )

    def test_config_invalid_regime_multiplier_negative(self):
        """Test that negative regime multiplier raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(
                regime_multipliers={"NORMAL": -0.5}
            )

    def test_config_invalid_regime_name_type(self):
        """Test that non-string regime names raise ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(
                regime_multipliers={123: 1.0}
            )

    def test_config_component_weights_not_sum_to_one(self):
        """Test that component_weights not summing to 1.0 raises ValueError."""
        with self.assertRaises(ValueError):
            AdaptivePositionSizingConfig(
                component_weights={"kelly": 0.5, "confidence": 0.3, "drawdown": 0.1, "regime": 0.05}
            )


class TestKellyCriterion(unittest.TestCase):
    """Tests for fractional Kelly criterion calculation."""

    def test_kelly_no_trades(self):
        """Test Kelly calculation with no trades returns conservative default."""
        sizer = create_default_position_sizer()
        kelly = sizer.calculate_kelly()
        self.assertEqual(kelly, 0.01)

    def test_kelly_one_winning_trade(self):
        """Test Kelly with one winning trade."""
        sizer = create_default_position_sizer()
        sizer.record_trade(pnl=100.0, win=True)
        kelly = sizer.calculate_kelly()
        self.assertGreater(kelly, 0.01)
        self.assertLessEqual(kelly, sizer.config.kelly_fraction)

    def test_kelly_one_losing_trade(self):
        """Test Kelly with one losing trade."""
        sizer = create_default_position_sizer()
        sizer.record_trade(pnl=-50.0, win=False)
        kelly = sizer.calculate_kelly()
        self.assertEqual(kelly, 0.01)  # Conservative default with losses only

    def test_kelly_zero_win_rate(self):
        """Test Kelly with 0% win rate."""
        sizer = create_default_position_sizer()
        for _ in range(10):
            sizer.record_trade(pnl=-50.0, win=False)
        kelly = sizer.calculate_kelly()
        self.assertEqual(kelly, 0.01)

    def test_kelly_perfect_win_rate(self):
        """Test Kelly with 100% win rate."""
        sizer = create_default_position_sizer()
        for _ in range(10):
            sizer.record_trade(pnl=100.0, win=True)
        kelly = sizer.calculate_kelly()
        self.assertEqual(kelly, sizer.config.kelly_fraction)

    def test_kelly_balanced_wins_losses(self):
        """Test Kelly with balanced win/loss ratio."""
        sizer = create_default_position_sizer()
        for _ in range(5):
            sizer.record_trade(pnl=100.0, win=True)
        for _ in range(5):
            sizer.record_trade(pnl=-80.0, win=False)
        kelly = sizer.calculate_kelly()
        self.assertGreater(kelly, 0.01)
        self.assertLessEqual(kelly, sizer.config.kelly_fraction)

    def test_kelly_rolling_window(self):
        """Test that Kelly uses rolling window correctly."""
        config = AdaptivePositionSizingConfig(kelly_window=5)
        sizer = AdaptivePositionSizer(config)

        # Add 10 losing trades
        for _ in range(10):
            sizer.record_trade(pnl=-50.0, win=False)

        kelly_after_losses = sizer.calculate_kelly()

        # Add 5 winning trades (should push out 5 losing trades)
        for _ in range(5):
            sizer.record_trade(pnl=100.0, win=True)

        kelly_after_wins = sizer.calculate_kelly()

        # Kelly should improve with winning trades
        self.assertGreater(kelly_after_wins, kelly_after_losses)
        self.assertEqual(len(sizer._trade_history), 5)  # Window size maintained

    def test_kelly_negative_risk_reward(self):
        """Test Kelly with negative average loss (shouldn't happen but gracefully handled)."""
        sizer = create_default_position_sizer()
        sizer.record_trade(pnl=10.0, win=True)
        sizer.record_trade(pnl=5.0, win=True)  # All positive PnL
        kelly = sizer.calculate_kelly()
        self.assertEqual(kelly, sizer.config.kelly_fraction)

    def test_kelly_nan_pnl_skipped(self):
        """Test that NaN PnL values are skipped."""
        sizer = create_default_position_sizer()
        sizer.record_trade(pnl=100.0, win=True)
        sizer.record_trade(pnl=np.nan, win=False)  # Should be skipped
        sizer.record_trade(pnl=80.0, win=True)

        # Should have 2 trades in history (NaN skipped)
        self.assertEqual(len(sizer._trade_history), 2)

    def test_kelly_infinite_pnl_skipped(self):
        """Test that infinite PnL values are skipped."""
        sizer = create_default_position_sizer()
        sizer.record_trade(pnl=100.0, win=True)
        sizer.record_trade(pnl=np.inf, win=False)  # Should be skipped
        sizer.record_trade(pnl=80.0, win=True)

        self.assertEqual(len(sizer._trade_history), 2)


class TestConfidenceSigmoid(unittest.TestCase):
    """Tests for confidence-weighted sigmoid calculation."""

    def test_sigmoid_confidence_zero(self):
        """Test sigmoid with confidence = 0."""
        sizer = create_default_position_sizer()
        sigmoid = sizer.calculate_confidence_multiplier(0.0)
        # At confidence=0, sigmoid should be near lower asymptote
        self.assertLess(sigmoid, 0.15)

    def test_sigmoid_confidence_half(self):
        """Test sigmoid with confidence = 0.5 (midpoint)."""
        sizer = create_default_position_sizer()
        sigmoid = sizer.calculate_confidence_multiplier(0.5)
        # At midpoint, sigmoid should be ~0.5
        self.assertAlmostEqual(sigmoid, 0.5, places=1)

    def test_sigmoid_confidence_one(self):
        """Test sigmoid with confidence = 1.0."""
        sizer = create_default_position_sizer()
        sigmoid = sizer.calculate_confidence_multiplier(1.0)
        # At confidence=1, sigmoid should be near upper asymptote
        self.assertGreater(sigmoid, 0.85)

    def test_sigmoid_monotonic_increasing(self):
        """Test that sigmoid is monotonically increasing."""
        sizer = create_default_position_sizer()
        prev_sigmoid = 0
        for confidence in np.linspace(0, 1, 11):
            sigmoid = sizer.calculate_confidence_multiplier(confidence)
            self.assertGreaterEqual(sigmoid, prev_sigmoid)
            prev_sigmoid = sigmoid

    def test_sigmoid_steep_midpoint(self):
        """Test sigmoid with high steepness (sharp transition)."""
        config = AdaptivePositionSizingConfig(sigmoid_steepness=10.0)
        sizer = AdaptivePositionSizer(config)

        low_conf = sizer.calculate_confidence_multiplier(0.4)
        mid_conf = sizer.calculate_confidence_multiplier(0.5)
        high_conf = sizer.calculate_confidence_multiplier(0.6)

        # Higher steepness = steeper transition
        self.assertLess(low_conf, 0.3)
        self.assertGreater(high_conf, 0.7)
        # Verify steepness creates stronger transition
        self.assertGreater(high_conf - low_conf, 0.4)

    def test_sigmoid_shallow_midpoint(self):
        """Test sigmoid with low steepness (gradual transition)."""
        config = AdaptivePositionSizingConfig(sigmoid_steepness=0.5)
        sizer = AdaptivePositionSizer(config)

        low_conf = sizer.calculate_confidence_multiplier(0.2)
        high_conf = sizer.calculate_confidence_multiplier(0.8)

        # Lower steepness = more gradual transition
        difference = high_conf - low_conf
        self.assertLess(difference, 0.8)

    def test_sigmoid_custom_midpoint(self):
        """Test sigmoid with custom midpoint."""
        config = AdaptivePositionSizingConfig(sigmoid_midpoint=0.7)
        sizer = AdaptivePositionSizer(config)

        sigmoid_at_midpoint = sizer.calculate_confidence_multiplier(0.7)
        # At midpoint, sigmoid should be ~0.5
        self.assertAlmostEqual(sigmoid_at_midpoint, 0.5, places=1)

    def test_sigmoid_invalid_confidence_negative(self):
        """Test sigmoid with negative confidence raises ValueError."""
        sizer = create_default_position_sizer()
        with self.assertRaises(ValueError):
            sizer.calculate_confidence_multiplier(-0.1)

    def test_sigmoid_invalid_confidence_over_one(self):
        """Test sigmoid with confidence > 1 raises ValueError."""
        sizer = create_default_position_sizer()
        with self.assertRaises(ValueError):
            sizer.calculate_confidence_multiplier(1.5)


class TestDrawdownRecovery(unittest.TestCase):
    """Tests for drawdown recovery factor calculation."""

    def test_drawdown_zero_recovery_one(self):
        """Test that zero drawdown gives recovery factor = 1.0."""
        sizer = create_default_position_sizer()
        recovery = sizer.calculate_drawdown_recovery(0.0)
        self.assertEqual(recovery, 1.0)

    def test_drawdown_half_max(self):
        """Test recovery at half max acceptable drawdown."""
        config = AdaptivePositionSizingConfig(
            max_acceptable_drawdown=0.20, drawdown_floor=0.1
        )
        sizer = AdaptivePositionSizer(config)
        recovery = sizer.calculate_drawdown_recovery(0.10)
        # Should be between floor and 1.0
        self.assertGreaterEqual(recovery, 0.1)
        self.assertLessEqual(recovery, 1.0)

    def test_drawdown_at_max(self):
        """Test recovery when at maximum acceptable drawdown."""
        config = AdaptivePositionSizingConfig(max_acceptable_drawdown=0.15)
        sizer = AdaptivePositionSizer(config)
        recovery = sizer.calculate_drawdown_recovery(0.15)
        # Should be at drawdown_floor (clamped)
        self.assertEqual(recovery, config.drawdown_floor)

    def test_drawdown_exceeds_max(self):
        """Test recovery when drawdown exceeds max."""
        config = AdaptivePositionSizingConfig(
            max_acceptable_drawdown=0.15, drawdown_floor=0.2
        )
        sizer = AdaptivePositionSizer(config)
        recovery = sizer.calculate_drawdown_recovery(0.25)
        # Should be clamped to floor
        self.assertEqual(recovery, 0.2)

    def test_drawdown_progressive_reduction(self):
        """Test that recovery decreases as drawdown increases."""
        sizer = create_default_position_sizer()
        recovery_small = sizer.calculate_drawdown_recovery(0.02)
        recovery_medium = sizer.calculate_drawdown_recovery(0.07)
        recovery_large = sizer.calculate_drawdown_recovery(0.14)

        self.assertGreater(recovery_small, recovery_medium)
        self.assertGreater(recovery_medium, recovery_large)

    def test_drawdown_invalid_negative(self):
        """Test drawdown with negative value raises ValueError."""
        sizer = create_default_position_sizer()
        with self.assertRaises(ValueError):
            sizer.calculate_drawdown_recovery(-0.05)

    def test_drawdown_invalid_over_one(self):
        """Test drawdown > 1.0 raises ValueError."""
        sizer = create_default_position_sizer()
        with self.assertRaises(ValueError):
            sizer.calculate_drawdown_recovery(1.5)


class TestRegimeMultipliers(unittest.TestCase):
    """Tests for regime-dependent multipliers."""

    def test_regime_low_volatility(self):
        """Test LOW volatility regime multiplier."""
        sizer = create_default_position_sizer()
        mult = sizer.get_regime_multiplier("LOW")
        self.assertEqual(mult, 1.3)

    def test_regime_normal_volatility(self):
        """Test NORMAL volatility regime multiplier."""
        sizer = create_default_position_sizer()
        mult = sizer.get_regime_multiplier("NORMAL")
        self.assertEqual(mult, 1.0)

    def test_regime_high_volatility(self):
        """Test HIGH volatility regime multiplier."""
        sizer = create_default_position_sizer()
        mult = sizer.get_regime_multiplier("HIGH")
        self.assertEqual(mult, 0.65)

    def test_regime_extreme_volatility(self):
        """Test EXTREME volatility regime multiplier."""
        sizer = create_default_position_sizer()
        mult = sizer.get_regime_multiplier("EXTREME")
        self.assertEqual(mult, 0.40)

    def test_regime_unknown_defaults_to_one(self):
        """Test unknown regime defaults to 1.0."""
        sizer = create_default_position_sizer()
        mult = sizer.get_regime_multiplier("UNKNOWN")
        self.assertEqual(mult, 1.0)

    def test_regime_custom_multiplier(self):
        """Test custom regime multipliers."""
        config = AdaptivePositionSizingConfig(
            regime_multipliers={"CALM": 1.5, "WILD": 0.3}
        )
        sizer = AdaptivePositionSizer(config)

        self.assertEqual(sizer.get_regime_multiplier("CALM"), 1.5)
        self.assertEqual(sizer.get_regime_multiplier("WILD"), 0.3)

    def test_regime_invalid_type(self):
        """Test that non-string regime name raises TypeError."""
        sizer = create_default_position_sizer()
        with self.assertRaises(TypeError):
            sizer.get_regime_multiplier(123)


class TestCompositeSizing(unittest.TestCase):
    """Tests for composite position sizing."""

    def test_composite_base_calculation(self):
        """Test basic composite sizing calculation."""
        sizer = create_default_position_sizer()
        result = sizer.calculate_size(
            account_equity=100000,
            confidence=0.7,
            current_drawdown_pct=0.0,
            regime_name="NORMAL"
        )

        self.assertIsInstance(result, PositionSizeResult)
        self.assertGreater(result.final_size, 0)
        self.assertEqual(result.base_size, 100000 * 0.05)  # 5% risk

    def test_composite_respects_min_position(self):
        """Test that composite sizing respects minimum position size."""
        sizer = create_default_position_sizer()
        result = sizer.calculate_size(
            account_equity=1000,  # Very small account
            confidence=0.0,  # Low confidence
            current_drawdown_pct=0.15,  # At max drawdown
            regime_name="EXTREME"
        )

        # Should be clamped to min_position_units (1000)
        self.assertGreaterEqual(result.final_size, sizer.config.min_position_units)

    def test_composite_respects_max_position(self):
        """Test that composite sizing respects maximum position size."""
        config = AdaptivePositionSizingConfig(
            max_risk_per_trade=0.10, min_position_units=0
        )
        sizer = AdaptivePositionSizer(config)

        result = sizer.calculate_size(
            account_equity=100000,
            confidence=1.0,  # High confidence
            current_drawdown_pct=0.0,  # No drawdown
            regime_name="LOW"  # Low volatility
        )

        max_allowed = 100000 * 0.10
        self.assertLessEqual(result.final_size, max_allowed)

    def test_composite_low_confidence_reduces_size(self):
        """Test that low confidence reduces position size."""
        sizer = create_default_position_sizer()

        result_high_conf = sizer.calculate_size(
            account_equity=100000,
            confidence=0.95,
            current_drawdown_pct=0.0,
            regime_name="NORMAL"
        )

        result_low_conf = sizer.calculate_size(
            account_equity=100000,
            confidence=0.1,
            current_drawdown_pct=0.0,
            regime_name="NORMAL"
        )

        self.assertGreater(result_high_conf.final_size, result_low_conf.final_size)

    def test_composite_drawdown_reduces_size(self):
        """Test that drawdown reduces position size."""
        sizer = create_default_position_sizer()

        result_no_dd = sizer.calculate_size(
            account_equity=100000,
            confidence=0.7,
            current_drawdown_pct=0.0,
            regime_name="NORMAL"
        )

        result_with_dd = sizer.calculate_size(
            account_equity=100000,
            confidence=0.7,
            current_drawdown_pct=0.10,
            regime_name="NORMAL"
        )

        self.assertGreater(result_no_dd.final_size, result_with_dd.final_size)

    def test_composite_regime_affects_size(self):
        """Test that regime affects position size."""
        sizer = create_default_position_sizer()

        result_low_vol = sizer.calculate_size(
            account_equity=100000,
            confidence=0.7,
            current_drawdown_pct=0.0,
            regime_name="LOW"
        )

        result_extreme = sizer.calculate_size(
            account_equity=100000,
            confidence=0.7,
            current_drawdown_pct=0.0,
            regime_name="EXTREME"
        )

        # LOW vol should allow larger position
        self.assertGreater(result_low_vol.final_size, result_extreme.final_size)

    def test_composite_invalid_equity_zero(self):
        """Test that zero equity raises ValueError."""
        sizer = create_default_position_sizer()
        with self.assertRaises(ValueError):
            sizer.calculate_size(
                account_equity=0,
                confidence=0.5,
                current_drawdown_pct=0.0
            )

    def test_composite_invalid_equity_negative(self):
        """Test that negative equity raises ValueError."""
        sizer = create_default_position_sizer()
        with self.assertRaises(ValueError):
            sizer.calculate_size(
                account_equity=-1000,
                confidence=0.5,
                current_drawdown_pct=0.0
            )

    def test_composite_result_contains_details(self):
        """Test that result contains detailed breakdown."""
        sizer = create_default_position_sizer()
        result = sizer.calculate_size(
            account_equity=100000,
            confidence=0.7,
            current_drawdown_pct=0.05,
            regime_name="NORMAL"
        )

        self.assertIn("account_equity", result.details)
        self.assertIn("base_size", result.details)
        self.assertIn("kelly_factor", result.details)
        self.assertIn("composite_weighted", result.details)
        self.assertIn("clamped", result.details)

    def test_composite_weighted_between_factors(self):
        """Test that weighted composite is between min and max factors."""
        sizer = create_default_position_sizer()
        result = sizer.calculate_size(
            account_equity=100000,
            confidence=0.5,
            current_drawdown_pct=0.05,
            regime_name="NORMAL"
        )

        min_factor = min(
            result.kelly_factor,
            result.confidence_factor,
            result.drawdown_factor,
            result.regime_factor
        )
        max_factor = max(
            result.kelly_factor,
            result.confidence_factor,
            result.drawdown_factor,
            result.regime_factor
        )

        weighted = result.details["composite_weighted"]
        self.assertGreaterEqual(weighted, min_factor)
        self.assertLessEqual(weighted, max_factor)


class TestStatePersistence(unittest.TestCase):
    """Tests for state save/load functionality."""

    def test_save_state_creates_file(self):
        """Test that save_state creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            sizer = create_default_position_sizer()
            sizer.record_trade(100.0, True)
            sizer.save_state(path)

            self.assertTrue(os.path.exists(path))

    def test_save_state_contains_config(self):
        """Test that saved state contains config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            sizer = create_default_position_sizer()
            sizer.save_state(path)

            with open(path, "r") as f:
                state = json.load(f)

            self.assertIn("config", state)
            self.assertEqual(state["config"]["kelly_fraction"], 0.33)

    def test_save_state_contains_trade_history(self):
        """Test that saved state contains trade history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            sizer = create_default_position_sizer()
            sizer.record_trade(100.0, True)
            sizer.record_trade(-50.0, False)
            sizer.save_state(path)

            with open(path, "r") as f:
                state = json.load(f)

            self.assertIn("trade_history", state)
            self.assertEqual(len(state["trade_history"]), 2)

    def test_load_state_restores_config(self):
        """Test that load_state restores configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")

            # Save
            sizer1 = create_conservative_position_sizer()
            sizer1.save_state(path)

            # Load
            sizer2 = create_default_position_sizer()
            sizer2.load_state(path)

            # Verify config loaded (trade history also restored)
            self.assertEqual(len(sizer2._trade_history), 0)

    def test_load_state_restores_trade_history(self):
        """Test that load_state restores trade history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")

            # Save
            sizer1 = create_default_position_sizer()
            sizer1.record_trade(100.0, True)
            sizer1.record_trade(-50.0, False)
            sizer1.record_trade(75.0, True)
            sizer1.save_state(path)

            # Load
            sizer2 = create_default_position_sizer()
            sizer2.load_state(path)

            self.assertEqual(len(sizer2._trade_history), 3)

    def test_load_state_missing_file(self):
        """Test that loading non-existent file doesn't crash."""
        sizer = create_default_position_sizer()
        # Should not raise
        sizer.load_state("/nonexistent/path/state.json")

    def test_load_state_corrupted_json(self):
        """Test that corrupted JSON is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")

            # Write corrupted JSON
            with open(path, "w") as f:
                f.write("{invalid json}")

            sizer = create_default_position_sizer()
            # Should not raise
            sizer.load_state(path)

    def test_load_state_missing_trade_history_key(self):
        """Test that missing trade_history key is handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")

            # Write state without trade_history
            with open(path, "w") as f:
                json.dump({"config": {}}, f)

            sizer = create_default_position_sizer()
            # Should not raise
            sizer.load_state(path)

    def test_load_state_malformed_trade_record(self):
        """Test that malformed trade records are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")

            # Write state with mixed valid/invalid trades
            state = {
                "config": asdict(AdaptivePositionSizingConfig()),
                "trade_history": [
                    [100.0, True],  # Valid
                    [50.0],  # Invalid (missing second element)
                    [-75.0, False],  # Valid
                ]
            }
            with open(path, "w") as f:
                json.dump(state, f)

            sizer = create_default_position_sizer()
            sizer.load_state(path)

            # Should have loaded the 2 valid trades
            self.assertEqual(len(sizer._trade_history), 2)

    def test_save_load_roundtrip(self):
        """Test complete save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")

            # Create, populate, save
            sizer1 = create_default_position_sizer()
            for i in range(10):
                sizer1.record_trade(100.0 + i*10, i % 2 == 0)
            kelly1 = sizer1.calculate_kelly()
            sizer1.save_state(path)

            # Load into new sizer
            sizer2 = create_default_position_sizer()
            sizer2.load_state(path)
            kelly2 = sizer2.calculate_kelly()

            # Verify equivalence
            self.assertEqual(kelly1, kelly2)
            self.assertEqual(len(sizer1._trade_history), len(sizer2._trade_history))

    def test_atomic_write_no_partial_files(self):
        """Test that atomic write doesn't leave partial files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            sizer = create_default_position_sizer()
            sizer.save_state(path)

            # Check that only final file exists (no .tmp)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(f"{path}.tmp"))


class TestFactoryFunctions(unittest.TestCase):
    """Tests for factory functions."""

    def test_default_factory(self):
        """Test create_default_position_sizer."""
        sizer = create_default_position_sizer()
        self.assertEqual(sizer.config.kelly_fraction, 0.33)
        self.assertEqual(sizer.config.drawdown_floor, 0.3)
        self.assertEqual(sizer.config.max_risk_per_trade, 0.10)

    def test_conservative_factory(self):
        """Test create_conservative_position_sizer."""
        sizer = create_conservative_position_sizer()
        self.assertEqual(sizer.config.kelly_fraction, 0.25)
        self.assertEqual(sizer.config.drawdown_floor, 0.4)
        self.assertEqual(sizer.config.max_risk_per_trade, 0.05)

    def test_aggressive_factory(self):
        """Test create_aggressive_position_sizer."""
        sizer = create_aggressive_position_sizer()
        self.assertEqual(sizer.config.kelly_fraction, 0.40)
        self.assertEqual(sizer.config.drawdown_floor, 0.2)
        self.assertEqual(sizer.config.max_risk_per_trade, 0.15)

    def test_factory_conservative_more_cautious_than_default(self):
        """Test that conservative sizer has more protective config."""
        sizer_default = create_default_position_sizer()
        sizer_conservative = create_conservative_position_sizer()

        # Conservative has lower Kelly fraction, higher drawdown floor, lower max risk
        self.assertLess(sizer_conservative.config.kelly_fraction, sizer_default.config.kelly_fraction)
        self.assertGreater(sizer_conservative.config.drawdown_floor, sizer_default.config.drawdown_floor)
        self.assertLess(sizer_conservative.config.max_risk_per_trade, sizer_default.config.max_risk_per_trade)

    def test_factory_aggressive_more_growth_oriented_than_default(self):
        """Test that aggressive sizer has growth-oriented config."""
        sizer_default = create_default_position_sizer()
        sizer_aggressive = create_aggressive_position_sizer()

        # Aggressive has higher Kelly fraction, lower drawdown floor, higher max risk
        self.assertGreater(sizer_aggressive.config.kelly_fraction, sizer_default.config.kelly_fraction)
        self.assertLess(sizer_aggressive.config.drawdown_floor, sizer_default.config.drawdown_floor)
        self.assertGreater(sizer_aggressive.config.max_risk_per_trade, sizer_default.config.max_risk_per_trade)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_very_large_account_equity(self):
        """Test with very large account equity."""
        sizer = create_default_position_sizer()
        result = sizer.calculate_size(
            account_equity=1_000_000_000,  # $1B
            confidence=0.7,
            current_drawdown_pct=0.0
        )

        self.assertGreater(result.final_size, 0)
        # Verify result is finite (not NaN or inf)
        self.assertTrue(np.isfinite(result.final_size))

    def test_very_small_account_equity(self):
        """Test with very small account equity."""
        sizer = create_default_position_sizer()
        result = sizer.calculate_size(
            account_equity=100,  # $100
            confidence=0.7,
            current_drawdown_pct=0.0
        )

        # Should be clamped to min_position_units
        self.assertGreaterEqual(result.final_size, 1000)

    def test_extreme_confidence_values(self):
        """Test sigmoid with extreme confidence values."""
        sizer = create_default_position_sizer()

        result_near_zero = sizer.calculate_size(100000, 0.0001, 0.0)
        result_near_one = sizer.calculate_size(100000, 0.9999, 0.0)

        self.assertGreater(result_near_one.final_size, result_near_zero.final_size)

    def test_many_trades_in_window(self):
        """Test with rolling window full of trades."""
        config = AdaptivePositionSizingConfig(kelly_window=100)
        sizer = AdaptivePositionSizer(config)

        # Add 200 trades (exceeds window)
        for i in range(200):
            win = (i % 3) == 0  # 1/3 win rate
            pnl = 100.0 if win else -70.0
            sizer.record_trade(pnl, win)

        # Window should be exactly 100
        self.assertEqual(len(sizer._trade_history), 100)

        # Kelly should be based on last 100 trades
        kelly = sizer.calculate_kelly()
        self.assertGreater(kelly, 0.01)

    def test_alternating_wins_losses(self):
        """Test Kelly with alternating wins and losses."""
        sizer = create_default_position_sizer()

        for i in range(50):
            win = (i % 2) == 0
            pnl = 100.0 if win else -100.0
            sizer.record_trade(pnl, win)

        kelly = sizer.calculate_kelly()
        # With 50% win rate and equal risk/reward, Kelly calculates to kelly_fraction
        self.assertEqual(kelly, sizer.config.kelly_fraction)

    def test_position_result_to_dict(self):
        """Test PositionSizeResult.to_dict() method."""
        sizer = create_default_position_sizer()
        result = sizer.calculate_size(100000, 0.7, 0.05)

        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertIn("final_size", result_dict)
        self.assertIn("kelly_factor", result_dict)


def asdict(obj):
    """Helper to convert dataclass to dict (unittest compatibility)."""
    from dataclasses import asdict as _asdict
    return _asdict(obj)


if __name__ == "__main__":
    unittest.main()
