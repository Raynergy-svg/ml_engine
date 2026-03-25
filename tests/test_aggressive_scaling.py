"""
Comprehensive tests for Aggressive Scaling module (US-254, US-255, US-259).

Test coverage:
- US-259: Config Dataclasses (ScalingPhase, TradingStats, KellyConfig, ExecutionConfig, etc.)
- US-254: Kelly Calculator + Position Sizing
- US-255: Risk Management & Execution (SmartOrderExecutor, IntradayCompounder, AggressiveRiskManager)
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.aggressive_scaling import (
    ScalingPhase,
    TradingStats,
    KellyConfig,
    ExecutionConfig,
    CompoundingConfig,
    DrawdownConfig,
    AggressiveScalingConfig,
    KellyCalculator,
    SmartOrderExecutor,
    OrderPlan,
    IntradayCompounder,
    IntradayState,
    AggressiveRiskManager,
    RiskState,
    AggressiveScalingEngine,
    create_validation_phase_engine,
    create_scaling_phase_engine,
    create_full_kelly_engine,
)


# ============================================================================
# US-259: Config Dataclasses Tests (10+ tests)
# ============================================================================

class TestScalingPhaseEnum:
    """Test ScalingPhase enum values."""

    def test_scaling_phase_validation_value(self):
        """Test VALIDATION phase has correct value."""
        assert ScalingPhase.VALIDATION.value == "validation"

    def test_scaling_phase_optimization_value(self):
        """Test OPTIMIZATION phase has correct value."""
        assert ScalingPhase.OPTIMIZATION.value == "optimization"

    def test_scaling_phase_scaling_value(self):
        """Test SCALING phase has correct value."""
        assert ScalingPhase.SCALING.value == "scaling"

    def test_scaling_phase_acceleration_value(self):
        """Test ACCELERATION phase has correct value."""
        assert ScalingPhase.ACCELERATION.value == "acceleration"

    def test_scaling_phase_all_phases_exist(self):
        """Test all four phases exist."""
        phases = list(ScalingPhase)
        assert len(phases) == 4
        assert ScalingPhase.VALIDATION in phases
        assert ScalingPhase.OPTIMIZATION in phases
        assert ScalingPhase.SCALING in phases
        assert ScalingPhase.ACCELERATION in phases


class TestTradingStatsDefaults:
    """Test TradingStats dataclass defaults."""

    def test_trading_stats_default_win_rate(self):
        """Test default win rate is 78%."""
        stats = TradingStats()
        assert stats.win_rate == 0.78

    def test_trading_stats_default_avg_win_pips(self):
        """Test default avg win pips is 41.5."""
        stats = TradingStats()
        assert stats.avg_win_pips == 41.5

    def test_trading_stats_default_avg_loss_pips(self):
        """Test default avg loss pips is 24.9."""
        stats = TradingStats()
        assert stats.avg_loss_pips == 24.9

    def test_trading_stats_default_total_trades(self):
        """Test default total trades is 6."""
        stats = TradingStats()
        assert stats.total_trades == 6

    def test_trading_stats_default_slippage_pips(self):
        """Test default slippage is 6.0 pips."""
        stats = TradingStats()
        assert stats.avg_slippage_pips == 6.0

    def test_trading_stats_default_spread_pips(self):
        """Test default spread is 1.5 pips."""
        stats = TradingStats()
        assert stats.spread_pips == 1.5

    def test_trading_stats_min_trades_for_kelly(self):
        """Test minimum trades for Kelly is 20."""
        stats = TradingStats()
        assert stats.min_trades_for_kelly == 20

    def test_trading_stats_confidence_factor(self):
        """Test confidence factor default is 0.5."""
        stats = TradingStats()
        assert stats.confidence_factor == 0.5


class TestTradingStatsProperties:
    """Test TradingStats calculated properties."""

    def test_risk_reward_ratio_calculation(self):
        """Test risk_reward_ratio property calculates correctly."""
        stats = TradingStats(
            avg_win_pips=41.5,
            avg_loss_pips=24.9
        )
        expected_rr = 41.5 / 24.9
        assert pytest.approx(stats.risk_reward_ratio, rel=1e-3) == expected_rr

    def test_risk_reward_ratio_default_on_zero_loss(self):
        """Test risk_reward_ratio returns 2.0 when avg_loss_pips is 0."""
        stats = TradingStats(avg_loss_pips=0)
        assert stats.risk_reward_ratio == 2.0

    def test_risk_reward_ratio_default_on_negative_loss(self):
        """Test risk_reward_ratio returns 2.0 when avg_loss_pips is negative."""
        stats = TradingStats(avg_loss_pips=-1.0)
        assert stats.risk_reward_ratio == 2.0

    def test_effective_risk_reward_with_slippage(self):
        """Test effective_risk_reward accounts for slippage."""
        stats = TradingStats(
            avg_win_pips=41.5,
            avg_loss_pips=24.9,
            avg_slippage_pips=6.0
        )
        # effective_win = 41.5 - 6.0 = 35.5
        # effective_loss = 24.9 + 6.0 = 30.9
        # effective_rr = 35.5 / 30.9
        expected = 35.5 / 30.9
        assert pytest.approx(stats.effective_risk_reward, rel=1e-3) == expected

    def test_effective_risk_reward_returns_1_on_zero_effective_loss(self):
        """Test effective_risk_reward returns 1.0 on zero or negative effective loss."""
        stats = TradingStats(
            avg_win_pips=10.0,
            avg_loss_pips=5.0,
            avg_slippage_pips=10.0  # Will make effective_loss = 5 + 10 = 15, but win = 10 - 10 = 0
        )
        # effective_win = 0, effective_loss = 15, so returns 0/15 = 0.0
        # But actually we need a case where effective_loss <= 0
        stats2 = TradingStats(
            avg_win_pips=10.0,
            avg_loss_pips=5.0,
            avg_slippage_pips=6.0  # effective_loss = 5 + 6 = 11 (still positive)
        )
        # To actually hit the <= 0 case, slippage must exceed loss
        stats3 = TradingStats(
            avg_win_pips=10.0,
            avg_loss_pips=5.0,
            avg_slippage_pips=5.0  # effective_loss = 5 + 5 = 10 > 0, still positive
        )
        # Let's test the actual implementation: when effective_loss <= 0
        # The implementation checks: if effective_loss <= 0: return 1.0
        stats4 = TradingStats(
            avg_win_pips=10.0,
            avg_loss_pips=0.0,  # This will cause division by zero prevention
            avg_slippage_pips=0.0
        )
        assert stats4.effective_risk_reward == 1.0


class TestTradingStatsUpdateFromTrades:
    """Test TradingStats.update_from_trades method."""

    def test_update_from_trades_empty_list(self):
        """Test update_from_trades with empty list does nothing."""
        stats = TradingStats()
        original_count = stats.total_trades
        stats.update_from_trades([])
        assert stats.total_trades == original_count

    def test_update_from_trades_single_win(self):
        """Test update_from_trades with one winning trade."""
        stats = TradingStats(total_trades=0, winning_trades=0, losing_trades=0)
        trades = [
            {"pnl": 500, "pnl_pips": 50}
        ]
        stats.update_from_trades(trades)
        assert stats.total_trades == 1
        assert stats.winning_trades == 1
        assert stats.losing_trades == 0
        assert stats.win_rate == 1.0

    def test_update_from_trades_single_loss(self):
        """Test update_from_trades with one losing trade."""
        stats = TradingStats(total_trades=0, winning_trades=0, losing_trades=0)
        trades = [
            {"pnl": -300, "pnl_pips": -30}
        ]
        stats.update_from_trades(trades)
        assert stats.total_trades == 1
        assert stats.winning_trades == 0
        assert stats.losing_trades == 1
        assert stats.win_rate == 0.0

    def test_update_from_trades_mixed_wins_losses(self):
        """Test update_from_trades with mixed wins and losses."""
        stats = TradingStats(total_trades=0, winning_trades=0, losing_trades=0)
        trades = [
            {"pnl": 500, "pnl_pips": 50},
            {"pnl": -300, "pnl_pips": -30},
            {"pnl": 600, "pnl_pips": 60},
        ]
        stats.update_from_trades(trades)
        assert stats.total_trades == 3
        assert stats.winning_trades == 2
        assert stats.losing_trades == 1
        assert pytest.approx(stats.win_rate, rel=1e-3) == 2/3

    def test_update_from_trades_calculates_avg_win_pips(self):
        """Test update_from_trades calculates avg_win_pips correctly."""
        stats = TradingStats(total_trades=0)
        trades = [
            {"pnl": 500, "pnl_pips": 50},
            {"pnl": 600, "pnl_pips": 60},
        ]
        stats.update_from_trades(trades)
        assert pytest.approx(stats.avg_win_pips, rel=1e-3) == 55.0

    def test_update_from_trades_calculates_avg_loss_pips(self):
        """Test update_from_trades calculates avg_loss_pips correctly."""
        stats = TradingStats(total_trades=0)
        trades = [
            {"pnl": -300, "pnl_pips": -30},
            {"pnl": -200, "pnl_pips": -20},
        ]
        stats.update_from_trades(trades)
        assert pytest.approx(stats.avg_loss_pips, rel=1e-3) == 25.0

    def test_update_from_trades_updates_slippage(self):
        """Test update_from_trades updates avg_slippage_pips."""
        stats = TradingStats(total_trades=0)
        trades = [
            {"pnl": 500, "slippage_pips": 2.0},
            {"pnl": 600, "slippage_pips": 4.0},
        ]
        stats.update_from_trades(trades)
        assert pytest.approx(stats.avg_slippage_pips, rel=1e-3) == 3.0


class TestKellyConfigDefaults:
    """Test KellyConfig dataclass defaults."""

    def test_kelly_config_full_kelly_fraction(self):
        """Test full_kelly_fraction defaults to 1.0."""
        config = KellyConfig()
        assert config.full_kelly_fraction == 1.0

    def test_kelly_config_max_kelly_fraction(self):
        """Test max_kelly_fraction defaults to 0.65."""
        config = KellyConfig()
        assert config.max_kelly_fraction == 0.65

    def test_kelly_config_min_kelly_fraction(self):
        """Test min_kelly_fraction defaults to 0.05."""
        config = KellyConfig()
        assert config.min_kelly_fraction == 0.05

    def test_kelly_config_phase_multipliers(self):
        """Test phase multipliers are correct."""
        config = KellyConfig()
        assert config.validation_multiplier == 0.25
        assert config.optimization_multiplier == 0.50
        assert config.scaling_multiplier == 0.75
        assert config.acceleration_multiplier == 1.0

    def test_kelly_config_safety_flags(self):
        """Test safety flags default to True."""
        config = KellyConfig()
        assert config.slippage_adjustment is True
        assert config.sample_size_scaling is True


class TestExecutionConfigDefaults:
    """Test ExecutionConfig dataclass defaults."""

    def test_execution_config_use_limit_orders(self):
        """Test use_limit_orders defaults to True."""
        config = ExecutionConfig()
        assert config.use_limit_orders is True

    def test_execution_config_use_split_orders(self):
        """Test use_split_orders defaults to True."""
        config = ExecutionConfig()
        assert config.use_split_orders is True

    def test_execution_config_split_threshold_lots(self):
        """Test split_threshold_lots defaults to 5.0."""
        config = ExecutionConfig()
        assert config.split_threshold_lots == 5.0

    def test_execution_config_split_ratios_sum_to_one(self):
        """Test split_ratios sum to 1.0."""
        config = ExecutionConfig()
        assert pytest.approx(sum(config.split_ratios), rel=1e-10) == 1.0

    def test_execution_config_split_ratios_are_correct(self):
        """Test split_ratios are 40%, 30%, 30%."""
        config = ExecutionConfig()
        assert config.split_ratios == (0.4, 0.3, 0.3)


class TestCompoundingConfigDefaults:
    """Test CompoundingConfig dataclass defaults."""

    def test_compounding_config_enabled(self):
        """Test compounding is enabled by default."""
        config = CompoundingConfig()
        assert config.enabled is True

    def test_compounding_config_profit_reserve_pct(self):
        """Test profit_reserve_pct defaults to 10%."""
        config = CompoundingConfig()
        assert config.profit_reserve_pct == 0.10

    def test_compounding_config_min_compound_amount(self):
        """Test min_compound_amount defaults to $1000."""
        config = CompoundingConfig()
        assert config.min_compound_amount == 1000


class TestDrawdownConfigDefaults:
    """Test DrawdownConfig dataclass defaults."""

    def test_drawdown_config_max_daily_drawdown_pct(self):
        """Test max_daily_drawdown_pct defaults to 10%."""
        config = DrawdownConfig()
        assert config.max_daily_drawdown_pct == 0.10

    def test_drawdown_config_max_trades_per_day(self):
        """Test max_trades_per_day defaults to 6."""
        config = DrawdownConfig()
        assert config.max_trades_per_day == 6

    def test_drawdown_config_losing_streak_reduction_dict(self):
        """Test losing_streak_reduction has correct values."""
        config = DrawdownConfig()
        assert config.losing_streak_reduction[1] == 0.75
        assert config.losing_streak_reduction[2] == 0.50
        assert config.losing_streak_reduction[3] == 0.25

    def test_drawdown_config_recovery_win_streak(self):
        """Test recovery_win_streak defaults to 2."""
        config = DrawdownConfig()
        assert config.recovery_win_streak == 2

    def test_drawdown_config_pause_after_max_streak(self):
        """Test pause_after_max_streak defaults to 3."""
        config = DrawdownConfig()
        assert config.pause_after_max_streak == 3


class TestAggressiveScalingConfigDefaults:
    """Test AggressiveScalingConfig dataclass defaults."""

    def test_aggressive_scaling_config_initial_phase(self):
        """Test current_phase defaults to VALIDATION."""
        config = AggressiveScalingConfig()
        assert config.current_phase == ScalingPhase.VALIDATION

    def test_aggressive_scaling_config_standard_lot_units(self):
        """Test standard_lot_units defaults to 100K."""
        config = AggressiveScalingConfig()
        assert config.standard_lot_units == 100_000

    def test_aggressive_scaling_config_pip_value_per_lot(self):
        """Test pip_value_per_lot defaults to $10."""
        config = AggressiveScalingConfig()
        assert config.pip_value_per_lot == 10.0

    def test_aggressive_scaling_config_jpy_pip_value(self):
        """Test jpy_pip_value_per_lot defaults to ~$9.3."""
        config = AggressiveScalingConfig()
        assert config.jpy_pip_value_per_lot == 9.3

    def test_aggressive_scaling_config_max_leverage(self):
        """Test max_leverage defaults to 50."""
        config = AggressiveScalingConfig()
        assert config.max_leverage == 50.0

    def test_aggressive_scaling_config_min_lot_size(self):
        """Test min_lot_size defaults to 0.01."""
        config = AggressiveScalingConfig()
        assert config.min_lot_size == 0.01

    def test_aggressive_scaling_config_has_sub_configs(self):
        """Test AggressiveScalingConfig has all sub-configurations."""
        config = AggressiveScalingConfig()
        assert isinstance(config.stats, TradingStats)
        assert isinstance(config.kelly, KellyConfig)
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.compounding, CompoundingConfig)
        assert isinstance(config.drawdown, DrawdownConfig)


# ============================================================================
# US-254: Kelly Calculator + Position Sizing Tests (15+ tests)
# ============================================================================

class TestKellyFormula:
    """Test KellyCalculator._kelly_formula."""

    def test_kelly_formula_60_percent_win_rate_2_to_1_rr(self):
        """Test Kelly formula with known values: 60% win rate, 2:1 R:R."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        # f* = (0.6 * 2 - 0.4) / 2 = (1.2 - 0.4) / 2 = 0.8 / 2 = 0.4
        kelly = calculator._kelly_formula(win_rate=0.60, rr_ratio=2.0)
        assert pytest.approx(kelly, rel=1e-3) == 0.4

    def test_kelly_formula_50_percent_win_rate_1_to_1_rr_equals_zero(self):
        """Test Kelly formula: 50% win rate, 1:1 R:R = 0 Kelly."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        # f* = (0.5 * 1 - 0.5) / 1 = 0 / 1 = 0
        kelly = calculator._kelly_formula(win_rate=0.50, rr_ratio=1.0)
        assert pytest.approx(kelly, rel=1e-10) == 0.0

    def test_kelly_formula_100_percent_win_rate_positive(self):
        """Test Kelly formula with 100% win rate."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        # f* = (1.0 * 2 - 0) / 2 = 2 / 2 = 1.0
        kelly = calculator._kelly_formula(win_rate=1.0, rr_ratio=2.0)
        assert pytest.approx(kelly, rel=1e-10) == 1.0

    def test_kelly_formula_zero_rr_ratio_returns_zero(self):
        """Test Kelly formula returns 0 when R:R ratio is 0."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        kelly = calculator._kelly_formula(win_rate=0.60, rr_ratio=0.0)
        assert kelly == 0.0

    def test_kelly_formula_negative_kelly_clamped_to_zero(self):
        """Test Kelly formula clamps negative Kelly to 0."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        # With low win rate and poor R:R, Kelly can be negative
        kelly = calculator._kelly_formula(win_rate=0.30, rr_ratio=1.0)
        assert kelly >= 0.0


class TestCalculateKellyFraction:
    """Test KellyCalculator.calculate_kelly_fraction."""

    def test_calculate_kelly_fraction_returns_dict_with_expected_keys(self):
        """Test calculate_kelly_fraction returns dict with all expected keys."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        result = calculator.calculate_kelly_fraction()

        expected_keys = {
            "raw_kelly", "raw_rr", "slippage_adjusted_kelly", "effective_rr",
            "phase", "phase_multiplier", "final_kelly_fraction", "recommended_risk_pct",
            "stats_used"
        }
        assert expected_keys.issubset(result.keys())

    def test_calculate_kelly_fraction_validation_phase_uses_25_percent_multiplier(self):
        """Test VALIDATION phase uses 0.25 multiplier."""
        config = AggressiveScalingConfig(current_phase=ScalingPhase.VALIDATION)
        config.kelly.full_kelly_fraction = 1.0
        config.kelly.slippage_adjustment = False
        config.kelly.sample_size_scaling = False
        calculator = KellyCalculator(config)

        result = calculator.calculate_kelly_fraction()
        assert result["phase_multiplier"] == 0.25

    def test_calculate_kelly_fraction_optimization_phase_uses_50_percent_multiplier(self):
        """Test OPTIMIZATION phase uses 0.50 multiplier."""
        config = AggressiveScalingConfig(current_phase=ScalingPhase.OPTIMIZATION)
        config.kelly.full_kelly_fraction = 1.0
        config.kelly.slippage_adjustment = False
        config.kelly.sample_size_scaling = False
        calculator = KellyCalculator(config)

        result = calculator.calculate_kelly_fraction()
        assert result["phase_multiplier"] == 0.50

    def test_calculate_kelly_fraction_scaling_phase_uses_75_percent_multiplier(self):
        """Test SCALING phase uses 0.75 multiplier."""
        config = AggressiveScalingConfig(current_phase=ScalingPhase.SCALING)
        config.kelly.full_kelly_fraction = 1.0
        config.kelly.slippage_adjustment = False
        config.kelly.sample_size_scaling = False
        calculator = KellyCalculator(config)

        result = calculator.calculate_kelly_fraction()
        assert result["phase_multiplier"] == 0.75

    def test_calculate_kelly_fraction_acceleration_phase_uses_100_percent_multiplier(self):
        """Test ACCELERATION phase uses 1.0 multiplier."""
        config = AggressiveScalingConfig(current_phase=ScalingPhase.ACCELERATION)
        config.kelly.full_kelly_fraction = 1.0
        config.kelly.slippage_adjustment = False
        config.kelly.sample_size_scaling = False
        calculator = KellyCalculator(config)

        result = calculator.calculate_kelly_fraction()
        assert result["phase_multiplier"] == 1.0

    def test_calculate_kelly_fraction_slippage_adjustment_reduces_kelly(self):
        """Test slippage adjustment reduces final Kelly fraction."""
        config = AggressiveScalingConfig()
        config.kelly.slippage_adjustment = True
        config.kelly.sample_size_scaling = False
        config.kelly.full_kelly_fraction = 1.0
        calculator_with_slippage = KellyCalculator(config)

        config.kelly.slippage_adjustment = False
        calculator_without_slippage = KellyCalculator(config)

        result_with = calculator_with_slippage.calculate_kelly_fraction()
        result_without = calculator_without_slippage.calculate_kelly_fraction()

        # Slippage should reduce Kelly
        assert result_with["final_kelly_fraction"] <= result_without["final_kelly_fraction"]

    def test_calculate_kelly_fraction_sample_size_scaling_with_few_trades(self):
        """Test sample_size_scaling scales down Kelly when < 20 trades."""
        config = AggressiveScalingConfig()
        config.stats = TradingStats(total_trades=10)  # Only 10 trades
        config.kelly.sample_size_scaling = True
        config.kelly.slippage_adjustment = False
        config.kelly.full_kelly_fraction = 1.0
        calculator = KellyCalculator(config)

        result = calculator.calculate_kelly_fraction()

        # Should be scaled by 10/20 = 0.5
        expected_min_scaling = 0.5
        assert result["final_kelly_fraction"] <= (expected_min_scaling * result["raw_kelly"])

    def test_calculate_kelly_fraction_result_clamped_to_min_max(self):
        """Test final Kelly fraction is clamped to [min, max] bounds."""
        config = AggressiveScalingConfig()
        config.stats = TradingStats(win_rate=0.99, avg_win_pips=100, avg_loss_pips=1)
        config.kelly.min_kelly_fraction = 0.05
        config.kelly.max_kelly_fraction = 0.65
        config.kelly.full_kelly_fraction = 1.0
        config.kelly.slippage_adjustment = False
        config.kelly.sample_size_scaling = False
        calculator = KellyCalculator(config)

        result = calculator.calculate_kelly_fraction()

        # Final Kelly should be clamped between min and max
        assert result["final_kelly_fraction"] <= config.kelly.max_kelly_fraction
        assert result["final_kelly_fraction"] >= config.kelly.min_kelly_fraction


class TestCalculatePositionSize:
    """Test KellyCalculator.calculate_position_size."""

    def test_calculate_position_size_returns_dict_with_expected_keys(self):
        """Test calculate_position_size returns dict with expected keys."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        result = calculator.calculate_position_size(
            account_equity=100_000,
            stop_loss_pips=25.0
        )

        expected_keys = {"lots", "units", "risk_amount_usd", "risk_pct", "leverage_used"}
        assert expected_keys.issubset(result.keys())

    def test_calculate_position_size_jpy_pair_uses_different_pip_value(self):
        """Test JPY pair uses jpy_pip_value_per_lot."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        result_jpy = calculator.calculate_position_size(
            account_equity=100_000,
            stop_loss_pips=25.0,
            instrument="USD_JPY"
        )

        result_eur = calculator.calculate_position_size(
            account_equity=100_000,
            stop_loss_pips=25.0,
            instrument="EUR_USD"
        )

        # JPY should have different pip value, so different lots
        assert result_jpy["lots"] != result_eur["lots"]

    def test_calculate_position_size_leverage_tracked(self):
        """Test position size calculates and tracks leverage used."""
        config = AggressiveScalingConfig()
        config.stats = TradingStats(total_trades=30, win_rate=0.78)
        config.max_leverage = 50.0
        config.kelly.full_kelly_fraction = 0.3
        config.kelly.max_kelly_fraction = 0.65
        config.kelly.sample_size_scaling = False
        config.kelly.slippage_adjustment = False
        calculator = KellyCalculator(config)

        result = calculator.calculate_position_size(
            account_equity=100_000,
            stop_loss_pips=25.0
        )

        # Position should be calculated successfully
        assert "lots" in result
        assert "leverage_used" in result
        assert result["lots"] > 0
        assert result["leverage_used"] >= 0

    def test_calculate_position_size_enforces_minimum_lot_size(self):
        """Test position size enforces minimum lot size."""
        config = AggressiveScalingConfig()
        config.min_lot_size = 0.01
        config.kelly.full_kelly_fraction = 0.001  # Very small Kelly
        calculator = KellyCalculator(config)

        result = calculator.calculate_position_size(
            account_equity=100_000,
            stop_loss_pips=25.0
        )

        # Lots should be at least min_lot_size
        assert result["lots"] >= config.min_lot_size

    def test_calculate_position_size_zero_stop_loss_returns_error(self):
        """Test position size with zero stop_loss_pips returns error."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        result = calculator.calculate_position_size(
            account_equity=100_000,
            stop_loss_pips=0.0
        )

        assert "error" in result
        assert result["lots"] == 0
        assert result["units"] == 0

    def test_calculate_position_size_negative_stop_loss_returns_error(self):
        """Test position size with negative stop_loss_pips returns error."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        result = calculator.calculate_position_size(
            account_equity=100_000,
            stop_loss_pips=-10.0
        )

        assert "error" in result


# ============================================================================
# US-255: Risk Management & Execution Tests (15+ tests)
# ============================================================================

class TestSmartOrderExecutor:
    """Test SmartOrderExecutor.plan_execution."""

    def test_plan_execution_single_order_when_lots_below_threshold(self):
        """Test single order when lots < split_threshold_lots."""
        config = AggressiveScalingConfig()
        config.execution.split_threshold_lots = 5.0
        executor = SmartOrderExecutor(config)

        plan = executor.plan_execution(
            direction="long",
            total_lots=3.0,
            current_price=1.0850,
            atr=0.0012
        )

        assert len(plan.split_orders) == 1
        assert plan.split_orders[0]["lots"] == 3.0

    def test_plan_execution_split_orders_when_lots_above_threshold(self):
        """Test split orders when lots > split_threshold_lots."""
        config = AggressiveScalingConfig()
        config.execution.split_threshold_lots = 5.0
        executor = SmartOrderExecutor(config)

        plan = executor.plan_execution(
            direction="long",
            total_lots=10.0,
            current_price=1.0850,
            atr=0.0012
        )

        assert len(plan.split_orders) > 1
        # Check ratios sum to total
        total_lots_planned = sum(o["lots"] for o in plan.split_orders)
        assert pytest.approx(total_lots_planned, rel=1e-2) == 10.0

    def test_plan_execution_estimate_slippage_limit_less_than_market(self):
        """Test slippage estimation: limit < market < passive."""
        config = AggressiveScalingConfig()
        executor = SmartOrderExecutor(config)

        limit_slippage = executor._estimate_slippage(lots=5.0, order_type="limit")
        market_slippage = executor._estimate_slippage(lots=5.0, order_type="market")
        passive_slippage = executor._estimate_slippage(lots=5.0, order_type="passive")

        # Limit should have lowest slippage, passive highest reduction
        assert limit_slippage < market_slippage
        # Passive gets the best fill (maker rebate)
        assert passive_slippage < limit_slippage

    def test_plan_execution_long_limit_price_below_current(self):
        """Test long trade limit price is below current price."""
        config = AggressiveScalingConfig()
        config.execution.use_limit_orders = True
        config.execution.limit_offset_atr_mult = 0.3
        executor = SmartOrderExecutor(config)

        plan = executor.plan_execution(
            direction="long",
            total_lots=1.0,
            current_price=1.0850,
            atr=0.0100  # $0.01 ATR
        )

        # For long, limit should be below current price
        assert plan.limit_price < 1.0850

    def test_plan_execution_short_limit_price_above_current(self):
        """Test short trade limit price is above current price."""
        config = AggressiveScalingConfig()
        config.execution.use_limit_orders = True
        config.execution.limit_offset_atr_mult = 0.3
        executor = SmartOrderExecutor(config)

        plan = executor.plan_execution(
            direction="short",
            total_lots=1.0,
            current_price=1.0850,
            atr=0.0100  # $0.01 ATR
        )

        # For short, limit should be above current price
        assert plan.limit_price > 1.0850

    def test_plan_execution_returns_order_plan_with_savings(self):
        """Test plan_execution returns OrderPlan with estimated savings."""
        config = AggressiveScalingConfig()
        executor = SmartOrderExecutor(config)

        plan = executor.plan_execution(
            direction="long",
            total_lots=5.0,
            current_price=1.0850,
            atr=0.0012
        )

        assert isinstance(plan, OrderPlan)
        assert isinstance(plan.estimated_savings_usd, (int, float))


class TestIntradayCompounder:
    """Test IntradayCompounder."""

    def test_start_session_initializes_state(self):
        """Test start_session initializes IntradayState."""
        config = AggressiveScalingConfig()
        compounder = IntradayCompounder(config)

        state = compounder.start_session(starting_equity=100_000)

        assert isinstance(state, IntradayState)
        assert state.starting_equity == 100_000
        assert state.current_equity == 100_000
        assert state.trades_today == 0

    def test_update_after_trade_updates_equity_on_win(self):
        """Test update_after_trade increases equity on winning trade."""
        config = AggressiveScalingConfig()
        compounder = IntradayCompounder(config)
        compounder.start_session(100_000)

        result = compounder.update_after_trade(pnl=1_000, is_win=True)

        assert result["current_equity"] == 101_000
        assert result["trades_today"] == 1

    def test_update_after_trade_updates_equity_on_loss(self):
        """Test update_after_trade decreases equity on losing trade."""
        config = AggressiveScalingConfig()
        compounder = IntradayCompounder(config)
        compounder.start_session(100_000)

        result = compounder.update_after_trade(pnl=-500, is_win=False)

        assert result["current_equity"] == 99_500
        assert result["trades_today"] == 1

    def test_update_after_trade_deducts_profit_reserve(self):
        """Test update_after_trade deducts profit reserve from usable equity."""
        config = AggressiveScalingConfig()
        config.compounding.profit_reserve_pct = 0.10
        compounder = IntradayCompounder(config)
        compounder.start_session(100_000)

        result = compounder.update_after_trade(pnl=10_000, is_win=True)

        # usable_equity should deduct 10% of profit (10% of 10_000 = 1_000)
        expected_usable = 110_000 - 1_000
        assert result["usable_equity"] == expected_usable

    def test_update_after_trade_raises_if_session_not_started(self):
        """Test update_after_trade raises if session not started."""
        config = AggressiveScalingConfig()
        compounder = IntradayCompounder(config)

        with pytest.raises(ValueError):
            compounder.update_after_trade(pnl=1_000, is_win=True)


class TestAggressiveRiskManager:
    """Test AggressiveRiskManager."""

    def test_check_can_trade_passes_when_fresh(self, tmp_path):
        """Test check_can_trade returns True when fresh."""
        config = AggressiveScalingConfig()
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"

        can_trade, reason = manager.check_can_trade(account_equity=100_000)

        assert can_trade is True
        assert reason == "OK"

    def test_check_can_trade_blocks_after_max_daily_trades(self, tmp_path):
        """Test check_can_trade blocks after max daily trades."""
        config = AggressiveScalingConfig()
        config.drawdown.max_trades_per_day = 5
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"
        manager.state.daily_trades = 5

        can_trade, reason = manager.check_can_trade(account_equity=100_000)

        assert can_trade is False
        assert "Daily trade limit" in reason

    def test_check_can_trade_blocks_on_daily_drawdown_limit(self, tmp_path):
        """Test check_can_trade blocks when daily drawdown exceeded."""
        config = AggressiveScalingConfig()
        config.drawdown.max_daily_drawdown_pct = 0.10  # 10%
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"
        manager.state.daily_pnl = -15_000  # -15% on 100K account

        can_trade, reason = manager.check_can_trade(account_equity=100_000)

        assert can_trade is False
        assert "drawdown" in reason.lower()

    def test_check_can_trade_blocks_when_paused(self, tmp_path):
        """Test check_can_trade blocks when manually paused."""
        config = AggressiveScalingConfig()
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"
        manager.state.is_paused = True
        manager.state.pause_reason = "Manual pause"

        can_trade, reason = manager.check_can_trade(account_equity=100_000)

        assert can_trade is False
        assert "paused" in reason.lower()

    def test_adjust_size_for_streak_no_adjustment_at_zero_losses(self, tmp_path):
        """Test adjust_size_for_streak returns full size with no losses."""
        config = AggressiveScalingConfig()
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"
        manager.state.losing_streak = 0

        adjusted, reason = manager.adjust_size_for_streak(base_lots=10.0)

        assert adjusted == 10.0
        assert "no losing streak" in reason.lower()

    def test_adjust_size_for_streak_reduces_on_losing_streak(self, tmp_path):
        """Test adjust_size_for_streak reduces size after losses."""
        config = AggressiveScalingConfig()
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"
        manager.state.losing_streak = 1

        adjusted, reason = manager.adjust_size_for_streak(base_lots=10.0)

        # After 1 loss, should be reduced to 75%
        assert adjusted == 7.5
        assert "75%" in reason

    def test_update_after_trade_tracks_winning_streak(self, tmp_path):
        """Test update_after_trade increments winning streak on win."""
        config = AggressiveScalingConfig()
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"

        result = manager.update_after_trade(pnl=1_000, is_win=True)

        assert result["winning_streak"] == 1
        assert result["losing_streak"] == 0

    def test_update_after_trade_tracks_losing_streak(self, tmp_path):
        """Test update_after_trade increments losing streak on loss."""
        config = AggressiveScalingConfig()
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"

        result = manager.update_after_trade(pnl=-500, is_win=False)

        assert result["losing_streak"] == 1
        assert result["winning_streak"] == 0

    def test_update_after_trade_resets_losing_streak_on_win(self, tmp_path):
        """Test update_after_trade resets losing streak on win."""
        config = AggressiveScalingConfig()
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"
        manager.state.losing_streak = 2

        result = manager.update_after_trade(pnl=1_000, is_win=True)

        assert result["winning_streak"] == 1
        assert result["losing_streak"] == 0

    def test_update_after_trade_restores_full_size_on_recovery_streak(self, tmp_path):
        """Test update_after_trade restores full size after recovery_win_streak."""
        config = AggressiveScalingConfig()
        config.drawdown.recovery_win_streak = 2
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"
        manager.state.position_multiplier = 0.5
        manager.state.winning_streak = 0

        # First win
        manager.update_after_trade(pnl=1_000, is_win=True)
        assert manager.state.position_multiplier == 0.5

        # Second win - should restore
        manager.update_after_trade(pnl=1_000, is_win=True)
        assert manager.state.position_multiplier == 1.0

    def test_reset_daily_clears_daily_state(self, tmp_path):
        """Test reset_daily clears daily counters."""
        config = AggressiveScalingConfig()
        manager = AggressiveRiskManager(config)
        manager._state_path = tmp_path / "risk_state.json"
        manager.state.daily_pnl = 5_000
        manager.state.daily_trades = 3
        manager.state.is_paused = True

        manager.reset_daily()

        assert manager.state.daily_pnl == 0.0
        assert manager.state.daily_trades == 0
        assert manager.state.is_paused is False


# ============================================================================
# Factory Functions Tests
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions for creating phase-specific engines."""

    def test_create_validation_phase_engine(self):
        """Test create_validation_phase_engine returns configured engine."""
        engine = create_validation_phase_engine()

        assert engine.config.current_phase == ScalingPhase.VALIDATION
        assert engine.config.kelly.full_kelly_fraction == 0.25
        assert engine.config.execution.use_limit_orders is True
        assert engine.config.execution.use_split_orders is False

    def test_create_scaling_phase_engine(self):
        """Test create_scaling_phase_engine returns configured engine."""
        engine = create_scaling_phase_engine()

        assert engine.config.current_phase == ScalingPhase.SCALING
        assert engine.config.kelly.full_kelly_fraction == 0.75
        assert engine.config.execution.use_split_orders is True

    def test_create_full_kelly_engine(self):
        """Test create_full_kelly_engine returns configured engine."""
        engine = create_full_kelly_engine()

        assert engine.config.current_phase == ScalingPhase.ACCELERATION
        assert engine.config.kelly.full_kelly_fraction == 1.0


# ============================================================================
# AggressiveScalingEngine Integration Tests
# ============================================================================

class TestAggressiveScalingEngine:
    """Test AggressiveScalingEngine main orchestration."""

    def test_advance_phase_progresses_through_phases(self, tmp_path):
        """Test advance_phase moves to next scaling phase."""
        config = AggressiveScalingConfig(current_phase=ScalingPhase.VALIDATION)
        engine = AggressiveScalingEngine(config)
        engine.risk_manager._state_path = tmp_path / "risk_state.json"

        phase1 = engine.advance_phase()
        assert phase1 == ScalingPhase.OPTIMIZATION.value

        phase2 = engine.advance_phase()
        assert phase2 == ScalingPhase.SCALING.value

        phase3 = engine.advance_phase()
        assert phase3 == ScalingPhase.ACCELERATION.value

        # Should stay at ACCELERATION
        phase4 = engine.advance_phase()
        assert phase4 == ScalingPhase.ACCELERATION.value

    def test_engine_start_session_initializes_components(self, tmp_path):
        """Test start_session initializes all components."""
        config = AggressiveScalingConfig()
        engine = AggressiveScalingEngine(config)
        engine.risk_manager._state_path = tmp_path / "risk_state.json"

        session = engine.start_session(account_equity=100_000)

        assert session["starting_equity"] == 100_000
        assert session["phase"] == ScalingPhase.VALIDATION.value
        assert engine.compounder.state is not None
        assert engine.compounder.state.starting_equity == 100_000

    def test_engine_calculate_position_returns_complete_plan(self, tmp_path):
        """Test calculate_position returns complete position plan."""
        config = AggressiveScalingConfig()
        engine = AggressiveScalingEngine(config)
        engine.risk_manager._state_path = tmp_path / "risk_state.json"
        engine.start_session(100_000)

        position = engine.calculate_position(
            account_equity=100_000,
            stop_loss_pips=24.9,
            current_price=1.0850,
            atr=0.0012
        )

        assert position["can_trade"] is True
        assert "lots" in position
        assert "units" in position
        assert "execution_plan" in position
        assert "kelly_calculation" in position

    def test_engine_update_after_trade_updates_all_components(self, tmp_path):
        """Test update_after_trade updates risk manager and compounder."""
        config = AggressiveScalingConfig()
        engine = AggressiveScalingEngine(config)
        engine.risk_manager._state_path = tmp_path / "risk_state.json"
        engine.start_session(100_000)

        result = engine.update_after_trade(pnl=1_000, is_win=True)

        assert "risk_state" in result
        assert "compound_state" in result
        assert result["risk_state"]["daily_trades"] == 1


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_kelly_with_all_zero_stats(self):
        """Test Kelly calculation with all-zero stats."""
        config = AggressiveScalingConfig()
        config.stats = TradingStats(
            win_rate=0.0,
            avg_win_pips=0.0,
            avg_loss_pips=0.0
        )
        calculator = KellyCalculator(config)

        result = calculator.calculate_kelly_fraction()

        # Should return valid dict with 0 Kelly
        assert "final_kelly_fraction" in result
        assert result["final_kelly_fraction"] >= 0.0

    def test_position_sizing_with_zero_account_equity(self):
        """Test position sizing with zero account equity."""
        config = AggressiveScalingConfig()
        calculator = KellyCalculator(config)

        result = calculator.calculate_position_size(
            account_equity=0,
            stop_loss_pips=25.0
        )

        # With zero equity, Kelly calculation returns 0 risk amount,
        # but minimum lot size still applies
        assert result["risk_pct"] == 0
        # Units will be min_lot_size * standard_lot_units
        assert result["units"] == int(config.min_lot_size * config.standard_lot_units)

    def test_multiple_consecutive_trades_with_compounding(self, tmp_path):
        """Test multiple consecutive trades with compounding."""
        config = AggressiveScalingConfig()
        config.compounding.enabled = True
        engine = AggressiveScalingEngine(config)
        engine.risk_manager._state_path = tmp_path / "risk_state.json"
        engine.start_session(100_000)

        # Three trades
        engine.update_after_trade(pnl=1_000, is_win=True)
        engine.update_after_trade(pnl=500, is_win=True)
        engine.update_after_trade(pnl=-300, is_win=False)

        # Check final state
        assert engine.compounder.state.trades_today == 3
        assert engine.compounder.state.current_equity == pytest.approx(101_200, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
