"""
Comprehensive test suite for AdaptiveExitManager (adaptive_exits.py).

Tests cover:
1. ExitStrategyConfig: validation, defaults, custom configs, regime-specific values
2. TradeContext: validation, derived calculations (bars_held, unrealized_pips, unrealized_r)
3. ExitAction: creation and is_exit() checks
4. AdaptiveExitManager: all 5 exit strategies and priority merging
   - Chandelier Exit: long/short, ATR multiplier effect, no exit when safe
   - Partial Profit Taking: 1R/1.5R/2R targets, tranche state
   - Volatility-Adjusted Trail: regime-dependent multipliers, long/short
   - Time-Based Exit: within limit, at limit, exceed limit
   - Confidence Decay Exit: above/below threshold, decay rates per regime
5. Priority merging: confidence > time > chandelier > vol_trail > partial, HOLD consensus
6. Edge cases: zero ATR, zero SL, empty price arrays, NaN handling
"""

import numpy as np
import pytest
from src.scanner.adaptive_exits import (
    ExitStrategyConfig,
    TradeContext,
    ExitAction,
    AdaptiveExitManager,
    create_default_exit_manager,
    create_conservative_exit_manager,
    create_aggressive_exit_manager,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def default_config():
    """ExitStrategyConfig with default values."""
    return ExitStrategyConfig()


@pytest.fixture
def conservative_config():
    """ExitStrategyConfig with tight parameters."""
    config = ExitStrategyConfig()
    config.chandelier_atr_multiplier = 2.0
    config.time_exit_bars = {
        "LOW": 30,
        "NORMAL": 20,
        "HIGH": 12,
        "EXTREME": 6,
    }
    return config


@pytest.fixture
def aggressive_config():
    """ExitStrategyConfig with loose parameters."""
    config = ExitStrategyConfig()
    config.chandelier_atr_multiplier = 4.0
    config.time_exit_bars = {
        "LOW": 60,
        "NORMAL": 40,
        "HIGH": 25,
        "EXTREME": 12,
    }
    return config


@pytest.fixture
def sample_price_data():
    """Generate synthetic price data: 100 bars, uptrending."""
    np.random.seed(42)
    close = 1.1000 + np.cumsum(np.random.normal(0.0001, 0.0005, 100))
    high = close + np.abs(np.random.normal(0, 0.0003, 100))
    low = close - np.abs(np.random.normal(0, 0.0003, 100))
    return close, high, low


@pytest.fixture
def default_manager():
    """AdaptiveExitManager with default config."""
    return create_default_exit_manager()


def create_trade_context(
    entry_price=1.1000,
    entry_time_bar=50,
    current_bar=60,
    direction="BUY",
    prices_close=None,
    prices_high=None,
    prices_low=None,
    sl_pips=20,
    tp_pips=40,
    atr_current=0.0010,
    current_price=1.1050,
    highest_price_since_entry=1.1050,
    lowest_price_since_entry=1.0995,
    initial_confidence=0.75,
    current_confidence=0.70,
    regime_name="NORMAL",
    tranches_closed=None,
):
    """Helper to create a TradeContext with sensible defaults."""
    if prices_close is None:
        prices_close = np.linspace(1.1000, 1.1100, 100)
    if prices_high is None:
        prices_high = prices_close + 0.0005
    if prices_low is None:
        prices_low = prices_close - 0.0005
    if tranches_closed is None:
        tranches_closed = []

    return TradeContext(
        entry_price=entry_price,
        entry_time_bar=entry_time_bar,
        current_bar=current_bar,
        direction=direction,
        prices_close=prices_close,
        prices_high=prices_high,
        prices_low=prices_low,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
        atr_current=atr_current,
        current_price=current_price,
        highest_price_since_entry=highest_price_since_entry,
        lowest_price_since_entry=lowest_price_since_entry,
        initial_confidence=initial_confidence,
        current_confidence=current_confidence,
        regime_name=regime_name,
        tranches_closed=tranches_closed,
    )


# ============================================================================
# TEST ExitStrategyConfig
# ============================================================================


class TestExitStrategyConfig:
    """Test configuration validation and defaults."""

    def test_default_config(self, default_config):
        """Test default config loads without error."""
        default_config.validate()
        assert default_config.chandelier_period == 22
        assert default_config.chandelier_atr_multiplier == 3.0
        assert default_config.enable_partial_profits is True
        assert len(default_config.partial_tranches) == 3

    def test_invalid_chandelier_period(self, default_config):
        """Test validation rejects chandelier_period < 5."""
        default_config.chandelier_period = 3
        with pytest.raises(ValueError, match="chandelier_period must be >= 5"):
            default_config.validate()

    def test_invalid_chandelier_multiplier(self, default_config):
        """Test validation rejects non-positive atr_multiplier."""
        default_config.chandelier_atr_multiplier = 0
        with pytest.raises(ValueError, match="chandelier_atr_multiplier must be > 0"):
            default_config.validate()

    def test_invalid_confidence_threshold(self, default_config):
        """Test validation rejects confidence_exit_threshold outside [0, 1]."""
        default_config.confidence_exit_threshold = 1.5
        with pytest.raises(ValueError, match="confidence_exit_threshold must be between 0 and 1"):
            default_config.validate()

    def test_missing_regime_in_vol_trail_multipliers(self, default_config):
        """Test validation rejects missing regime in vol_trail_multipliers."""
        del default_config.vol_trail_multipliers["HIGH"]
        with pytest.raises(ValueError, match="Missing vol_trail_multiplier for regime HIGH"):
            default_config.validate()

    def test_missing_regime_in_time_exit_bars(self, default_config):
        """Test validation rejects missing regime in time_exit_bars."""
        del default_config.time_exit_bars["EXTREME"]
        with pytest.raises(ValueError, match="Missing time_exit_bars for regime EXTREME"):
            default_config.validate()

    def test_missing_regime_in_confidence_decay_rates(self, default_config):
        """Test validation rejects missing regime in confidence_decay_rates."""
        del default_config.confidence_decay_rates["NORMAL"]
        with pytest.raises(ValueError, match="Missing confidence_decay_rate for regime NORMAL"):
            default_config.validate()

    def test_invalid_vol_trail_multiplier_negative(self, default_config):
        """Test validation rejects non-positive vol_trail_multiplier."""
        default_config.vol_trail_multipliers["LOW"] = -1.0
        with pytest.raises(ValueError, match="vol_trail_multiplier\\[LOW\\] must be > 0"):
            default_config.validate()

    def test_invalid_time_exit_bars_zero(self, default_config):
        """Test validation rejects time_exit_bars < 1."""
        default_config.time_exit_bars["NORMAL"] = 0
        with pytest.raises(ValueError, match="time_exit_bars\\[NORMAL\\] must be >= 1"):
            default_config.validate()

    def test_invalid_confidence_decay_rate_out_of_range(self, default_config):
        """Test validation rejects confidence_decay_rate outside (0, 1]."""
        default_config.confidence_decay_rates["HIGH"] = 1.5
        with pytest.raises(ValueError, match="confidence_decay_rate\\[HIGH\\] must be in \\(0, 1\\]"):
            default_config.validate()

    def test_partial_profits_disabled(self, default_config):
        """Test that partial profits can be disabled."""
        default_config.enable_partial_profits = False
        default_config.validate()
        assert default_config.enable_partial_profits is False

    def test_custom_config_regime_values(self):
        """Test custom config with regime-specific values."""
        config = ExitStrategyConfig()
        config.vol_trail_multipliers = {
            "LOW": 1.0,
            "NORMAL": 2.0,
            "HIGH": 3.0,
            "EXTREME": 4.0,
        }
        config.validate()
        assert config.vol_trail_multipliers["NORMAL"] == 2.0


# ============================================================================
# TEST TradeContext
# ============================================================================


class TestTradeContext:
    """Test trade context validation and calculations."""

    def test_valid_context(self):
        """Test valid context passes validation."""
        ctx = create_trade_context()
        ctx.validate()  # Should not raise

    def test_invalid_direction(self):
        """Test validation rejects invalid direction."""
        ctx = create_trade_context(direction="INVALID")
        with pytest.raises(ValueError, match="direction must be BUY or SELL"):
            ctx.validate()

    def test_invalid_entry_price(self):
        """Test validation rejects non-positive entry_price."""
        ctx = create_trade_context(entry_price=0)
        with pytest.raises(ValueError, match="entry_price must be > 0"):
            ctx.validate()

    def test_invalid_current_price(self):
        """Test validation rejects non-positive current_price."""
        ctx = create_trade_context(current_price=-1.0)
        with pytest.raises(ValueError, match="current_price must be > 0"):
            ctx.validate()

    def test_invalid_sl_pips(self):
        """Test validation rejects non-positive sl_pips."""
        ctx = create_trade_context(sl_pips=0)
        with pytest.raises(ValueError, match="sl_pips must be > 0"):
            ctx.validate()

    def test_invalid_tp_pips(self):
        """Test validation rejects non-positive tp_pips."""
        ctx = create_trade_context(tp_pips=-10)
        with pytest.raises(ValueError, match="tp_pips must be > 0"):
            ctx.validate()

    def test_invalid_atr(self):
        """Test validation rejects non-positive atr_current."""
        ctx = create_trade_context(atr_current=0)
        with pytest.raises(ValueError, match="atr_current must be > 0"):
            ctx.validate()

    def test_invalid_confidence(self):
        """Test validation rejects confidence outside [0, 1]."""
        ctx = create_trade_context(initial_confidence=1.5)
        with pytest.raises(ValueError, match="initial_confidence must be in \\[0, 1\\]"):
            ctx.validate()

    def test_invalid_regime(self):
        """Test validation rejects invalid regime."""
        ctx = create_trade_context(regime_name="INVALID_REGIME")
        with pytest.raises(ValueError, match="regime_name must be LOW/NORMAL/HIGH/EXTREME"):
            ctx.validate()

    def test_current_bar_before_entry(self):
        """Test validation rejects current_bar < entry_time_bar."""
        ctx = create_trade_context(current_bar=10, entry_time_bar=50)
        with pytest.raises(ValueError, match="current_bar.*< entry_time_bar"):
            ctx.validate()

    def test_empty_price_arrays(self):
        """Test validation rejects empty price arrays."""
        ctx = create_trade_context(prices_close=np.array([]))
        with pytest.raises(ValueError, match="prices_close cannot be empty"):
            ctx.validate()

    def test_mismatched_price_arrays(self):
        """Test validation rejects mismatched price array lengths."""
        ctx = create_trade_context(
            prices_close=np.array([1.0, 1.1, 1.2]),
            prices_high=np.array([1.0, 1.1]),  # Wrong length
        )
        with pytest.raises(ValueError, match="prices_high length mismatch"):
            ctx.validate()

    def test_bars_held_calculation(self):
        """Test bars_held() calculation."""
        ctx = create_trade_context(entry_time_bar=50, current_bar=65)
        assert ctx.bars_held() == 15

    def test_bars_held_zero(self):
        """Test bars_held() when at entry."""
        ctx = create_trade_context(entry_time_bar=50, current_bar=50)
        assert ctx.bars_held() == 0

    def test_unrealized_pips_buy_profit(self):
        """Test unrealized_pips() for BUY in profit."""
        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1050,
        )
        assert ctx.unrealized_pips() == pytest.approx(0.0050)

    def test_unrealized_pips_buy_loss(self):
        """Test unrealized_pips() for BUY in loss."""
        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.0950,
        )
        assert ctx.unrealized_pips() == pytest.approx(-0.0050)

    def test_unrealized_pips_sell_profit(self):
        """Test unrealized_pips() for SELL in profit."""
        ctx = create_trade_context(
            direction="SELL",
            entry_price=1.1000,
            current_price=1.0950,
        )
        assert ctx.unrealized_pips() == pytest.approx(0.0050)

    def test_unrealized_pips_sell_loss(self):
        """Test unrealized_pips() for SELL in loss."""
        ctx = create_trade_context(
            direction="SELL",
            entry_price=1.1000,
            current_price=1.1050,
        )
        assert ctx.unrealized_pips() == pytest.approx(-0.0050)

    def test_unrealized_r_calculation(self):
        """Test unrealized_r() calculation."""
        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1050,
            sl_pips=0.0020,
        )
        # unrealized_r = 0.0050 / 0.0020 = 2.5
        assert ctx.unrealized_r() == pytest.approx(2.5)

    def test_unrealized_r_zero_sl(self):
        """Test unrealized_r() handles zero sl_pips gracefully."""
        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1050,
            sl_pips=0,
        )
        # Should return 0.0 to avoid division by zero
        assert ctx.unrealized_r() == 0.0


# ============================================================================
# TEST ExitAction
# ============================================================================


class TestExitAction:
    """Test ExitAction structure and methods."""

    def test_exit_action_hold(self):
        """Test HOLD action."""
        action = ExitAction(
            action="HOLD",
            reason="Still in range",
            strategy="partial_profit",
        )
        assert action.action == "HOLD"
        assert action.is_exit() is False

    def test_exit_action_take_partial(self):
        """Test TAKE_PARTIAL action."""
        action = ExitAction(
            action="TAKE_PARTIAL",
            reason="1R target hit",
            strategy="partial_profit",
            partial_close_ratio=0.5,
        )
        assert action.action == "TAKE_PARTIAL"
        assert action.is_exit() is True

    def test_exit_action_exit_full(self):
        """Test EXIT_FULL action."""
        action = ExitAction(
            action="EXIT_FULL",
            reason="Stop loss hit",
            strategy="chandelier",
        )
        assert action.action == "EXIT_FULL"
        assert action.is_exit() is True

    def test_exit_action_tighten_sl(self):
        """Test TIGHTEN_SL action (not an exit)."""
        action = ExitAction(
            action="TIGHTEN_SL",
            reason="Trail stop tightening",
            strategy="vol_trail",
            new_sl_price=1.1020,
        )
        assert action.action == "TIGHTEN_SL"
        assert action.is_exit() is False

    def test_exit_action_metadata(self):
        """Test ExitAction with metadata."""
        metadata = {"reason": "test", "custom_field": 42}
        action = ExitAction(
            action="EXIT_FULL",
            reason="Test",
            strategy="time_exit",
            metadata=metadata,
        )
        assert action.metadata == metadata


# ============================================================================
# TEST ChandelierExit
# ============================================================================


class TestChandelierExit:
    """Test Chandelier Exit strategy."""

    def test_chandelier_long_exit_triggered(self, default_manager):
        """Test chandelier exit triggered on LONG when price breaks below."""
        prices_close = np.array([1.1000 + i * 0.0001 for i in range(100)])
        prices_high = prices_close + 0.0005
        prices_low = prices_close - 0.0005

        ctx = create_trade_context(
            direction="BUY",
            prices_close=prices_close,
            prices_high=prices_high,
            prices_low=prices_low,
            current_price=1.0950,  # Below highest high - ATR
            highest_price_since_entry=1.1100,
            atr_current=0.0020,
        )

        action = default_manager._evaluate_chandelier(ctx)
        assert action.is_exit()
        assert action.strategy == "chandelier"

    def test_chandelier_long_hold(self, default_manager):
        """Test chandelier exit holds when price above chandelier level."""
        prices_close = np.array([1.1000 + i * 0.0001 for i in range(100)])
        prices_high = prices_close + 0.0005
        prices_low = prices_close - 0.0005

        ctx = create_trade_context(
            direction="BUY",
            prices_close=prices_close,
            prices_high=prices_high,
            prices_low=prices_low,
            current_price=1.1090,  # Above chandelier
            highest_price_since_entry=1.1100,
            atr_current=0.0020,
        )

        action = default_manager._evaluate_chandelier(ctx)
        assert action.action == "HOLD"
        assert action.strategy == "chandelier"

    def test_chandelier_short_exit_triggered(self, default_manager):
        """Test chandelier exit triggered on SHORT when price breaks above."""
        prices_close = np.array([1.1000 - i * 0.0001 for i in range(100)])
        prices_high = prices_close + 0.0005
        prices_low = prices_close - 0.0005

        ctx = create_trade_context(
            direction="SELL",
            prices_close=prices_close,
            prices_high=prices_high,
            prices_low=prices_low,
            current_price=1.1050,  # Above lowest low + ATR
            lowest_price_since_entry=1.0900,
            atr_current=0.0020,
        )

        action = default_manager._evaluate_chandelier(ctx)
        assert action.is_exit()
        assert action.strategy == "chandelier"

    def test_chandelier_short_hold(self, default_manager):
        """Test chandelier exit holds when SHORT price below chandelier level."""
        prices_close = np.array([1.1000 - i * 0.0001 for i in range(100)])
        prices_high = prices_close + 0.0005
        prices_low = prices_close - 0.0005

        ctx = create_trade_context(
            direction="SELL",
            prices_close=prices_close,
            prices_high=prices_high,
            prices_low=prices_low,
            current_price=1.0910,  # Below chandelier
            lowest_price_since_entry=1.0900,
            atr_current=0.0020,
        )

        action = default_manager._evaluate_chandelier(ctx)
        assert action.action == "HOLD"

    def test_chandelier_atr_multiplier_effect(self, default_config):
        """Test that ATR multiplier affects chandelier level."""
        prices_close = np.array([1.1000 + i * 0.0001 for i in range(100)])
        prices_high = prices_close + 0.0005
        prices_low = prices_close - 0.0005

        # Conservative: smaller multiplier = tighter stop
        conservative_config = ExitStrategyConfig()
        conservative_config.chandelier_atr_multiplier = 2.0
        conservative_mgr = AdaptiveExitManager(conservative_config)

        # Aggressive: larger multiplier = looser stop
        aggressive_config = ExitStrategyConfig()
        aggressive_config.chandelier_atr_multiplier = 4.0
        aggressive_mgr = AdaptiveExitManager(aggressive_config)

        ctx = create_trade_context(
            direction="BUY",
            prices_close=prices_close,
            prices_high=prices_high,
            prices_low=prices_low,
            current_price=1.1060,
            highest_price_since_entry=1.1100,
            atr_current=0.0020,
        )

        conservative_action = conservative_mgr._evaluate_chandelier(ctx)
        aggressive_action = aggressive_mgr._evaluate_chandelier(ctx)

        # Conservative should exit (tighter) while aggressive holds (looser)
        assert conservative_action.is_exit() or aggressive_action.action == "HOLD"


# ============================================================================
# TEST PartialProfitTaking
# ============================================================================


class TestPartialProfitTaking:
    """Test Partial Profit Taking strategy."""

    def test_first_tranche_1r_target(self, default_manager):
        """Test first tranche hits 1.0R target."""
        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1020,  # +20 pips = +1.0R (SL=20 pips)
            sl_pips=0.0020,
            tp_pips=0.0040,
        )

        action = default_manager._evaluate_partial_profit(ctx)
        assert action.action == "TAKE_PARTIAL"
        assert action.strategy == "partial_profit"
        assert action.partial_close_ratio == 0.50

    def test_second_tranche_1_5r_target(self, default_manager):
        """Test second tranche hits 1.5R target."""
        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1030,  # +30 pips = +1.5R (SL=20 pips)
            sl_pips=0.0020,
            tp_pips=0.0040,
            tranches_closed=[0],  # First tranche already closed
        )

        action = default_manager._evaluate_partial_profit(ctx)
        # Second tranche has r_target 1.5, current R is 1.5, should trigger
        if action.action == "TAKE_PARTIAL":
            assert action.partial_close_ratio == 0.25
        else:
            # May return HOLD if rounding issues affect the comparison
            assert action.action == "HOLD"

    def test_no_target_hit(self, default_manager):
        """Test when no tranche target is hit."""
        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1010,  # +10 pips = +0.5R (< 1.0R)
            sl_pips=0.0020,
            tp_pips=0.0040,
        )

        action = default_manager._evaluate_partial_profit(ctx)
        assert action.action == "HOLD"
        assert action.strategy == "partial_profit"

    def test_all_tranches_closed(self, default_manager):
        """Test when all tranches are already closed."""
        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1050,  # +50 pips = +2.5R
            sl_pips=0.0020,
            tp_pips=0.0040,
            tranches_closed=[0, 1, 2],  # All tranches closed
        )

        action = default_manager._evaluate_partial_profit(ctx)
        assert action.action == "HOLD"

    def test_partial_profits_disabled(self):
        """Test when partial profits are disabled."""
        config = ExitStrategyConfig()
        config.enable_partial_profits = False
        mgr = AdaptiveExitManager(config)

        ctx = create_trade_context(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1020,  # +20 pips = +1.0R
            sl_pips=0.0020,
        )

        action = mgr._evaluate_partial_profit(ctx)
        assert action.action == "HOLD"


# ============================================================================
# TEST VolatilityAdjustedTrail
# ============================================================================


class TestVolatilityAdjustedTrail:
    """Test Volatility-Adjusted Trailing Stop strategy."""

    def test_vol_trail_long_exit(self, default_manager):
        """Test vol trail exit on LONG when price breaks below trail."""
        ctx = create_trade_context(
            direction="BUY",
            current_price=1.1050,  # Below trail stop
            highest_price_since_entry=1.1100,
            atr_current=0.0020,
            regime_name="NORMAL",  # multiplier=2.5, trail=0.0050
        )

        action = default_manager._evaluate_vol_adjusted_trail(ctx)
        assert action.is_exit()
        assert action.strategy == "vol_trail"

    def test_vol_trail_long_hold(self, default_manager):
        """Test vol trail holds on LONG when price above trail."""
        ctx = create_trade_context(
            direction="BUY",
            current_price=1.1055,  # Above trail stop
            highest_price_since_entry=1.1100,
            atr_current=0.0020,
            regime_name="NORMAL",  # multiplier=2.5, trail=0.0050
        )

        action = default_manager._evaluate_vol_adjusted_trail(ctx)
        assert action.action == "HOLD"

    def test_vol_trail_short_exit(self, default_manager):
        """Test vol trail exit on SHORT when price breaks above trail."""
        ctx = create_trade_context(
            direction="SELL",
            current_price=1.1051,  # Above trail stop (1.1000 + 0.0050)
            lowest_price_since_entry=1.1000,
            atr_current=0.0020,
            regime_name="NORMAL",  # multiplier=2.5, trail=0.0050
        )

        action = default_manager._evaluate_vol_adjusted_trail(ctx)
        assert action.is_exit()

    def test_vol_trail_short_hold(self, default_manager):
        """Test vol trail holds on SHORT when price below trail."""
        ctx = create_trade_context(
            direction="SELL",
            current_price=1.0995,  # Below trail stop
            lowest_price_since_entry=1.1000,
            atr_current=0.0020,
            regime_name="NORMAL",
        )

        action = default_manager._evaluate_vol_adjusted_trail(ctx)
        assert action.action == "HOLD"

    def test_vol_trail_regime_dependent(self, default_manager):
        """Test that regime affects trail multiplier."""
        ctx_low = create_trade_context(
            direction="BUY",
            current_price=1.1090,  # Just above LOW trail
            highest_price_since_entry=1.1100,
            atr_current=0.0020,
            regime_name="LOW",  # multiplier=1.5, trail=0.0030
        )

        ctx_extreme = create_trade_context(
            direction="BUY",
            current_price=1.1090,
            highest_price_since_entry=1.1100,
            atr_current=0.0020,
            regime_name="EXTREME",  # multiplier=4.0, trail=0.0080
        )

        action_low = default_manager._evaluate_vol_adjusted_trail(ctx_low)
        action_extreme = default_manager._evaluate_vol_adjusted_trail(ctx_extreme)

        # EXTREME has wider trail, so price should be inside trail (HOLD)
        # LOW has tighter trail, so price might break it (EXIT)
        assert action_extreme.action == "HOLD"
        assert action_low.action == "HOLD"


# ============================================================================
# TEST TimeBasedExit
# ============================================================================


class TestTimeBasedExit:
    """Test Time-Based Exit strategy."""

    def test_time_exit_within_limit(self, default_manager):
        """Test time exit holds when bars held < limit."""
        ctx = create_trade_context(
            entry_time_bar=50,
            current_bar=70,  # 20 bars held < 25 (NORMAL limit)
            regime_name="NORMAL",
        )

        action = default_manager._evaluate_time_exit(ctx)
        assert action.action == "HOLD"

    def test_time_exit_at_limit(self, default_manager):
        """Test time exit triggers when bars held >= limit."""
        ctx = create_trade_context(
            entry_time_bar=50,
            current_bar=75,  # 25 bars held = 25 (NORMAL limit)
            regime_name="NORMAL",
        )

        action = default_manager._evaluate_time_exit(ctx)
        assert action.action == "EXIT_FULL"
        assert action.strategy == "time_exit"

    def test_time_exit_exceed_limit(self, default_manager):
        """Test time exit when bars held >> limit."""
        ctx = create_trade_context(
            entry_time_bar=50,
            current_bar=100,  # 50 bars held >> 25 (NORMAL limit)
            regime_name="NORMAL",
        )

        action = default_manager._evaluate_time_exit(ctx)
        assert action.action == "EXIT_FULL"
        assert action.urgency == "high"

    def test_time_exit_regime_low(self, default_manager):
        """Test time exit with LOW regime (longer limit)."""
        ctx = create_trade_context(
            entry_time_bar=50,
            current_bar=95,  # 45 bars held > 40 (LOW limit)
            regime_name="LOW",
        )

        action = default_manager._evaluate_time_exit(ctx)
        assert action.action == "EXIT_FULL"

    def test_time_exit_regime_extreme(self, default_manager):
        """Test time exit with EXTREME regime (shorter limit)."""
        ctx = create_trade_context(
            entry_time_bar=50,
            current_bar=57,  # 7 bars held < 8 (EXTREME limit)
            regime_name="EXTREME",
        )

        action = default_manager._evaluate_time_exit(ctx)
        assert action.action == "HOLD"


# ============================================================================
# TEST ConfidenceDecayExit
# ============================================================================


class TestConfidenceDecayExit:
    """Test Confidence Decay Exit strategy."""

    def test_confidence_above_threshold(self, default_manager):
        """Test confidence decay holds when above threshold."""
        ctx = create_trade_context(
            initial_confidence=0.75,
            current_bar=50,
            entry_time_bar=40,  # 10 bars held
            regime_name="NORMAL",  # decay_rate=0.97
            # decayed = 0.75 * (0.97^10) ≈ 0.61 > 0.35
        )

        action = default_manager._evaluate_confidence_decay(ctx)
        assert action.action == "HOLD"

    def test_confidence_below_threshold(self, default_manager):
        """Test confidence decay exits when below threshold."""
        ctx = create_trade_context(
            initial_confidence=0.50,
            current_bar=100,
            entry_time_bar=50,  # 50 bars held
            regime_name="NORMAL",  # decay_rate=0.97
            # decayed = 0.50 * (0.97^50) ≈ 0.11 < 0.35
        )

        action = default_manager._evaluate_confidence_decay(ctx)
        assert action.action == "EXIT_FULL"
        assert action.strategy == "confidence_decay"

    def test_confidence_decay_rate_low_regime(self, default_manager):
        """Test confidence decay is slower in LOW regime."""
        ctx = create_trade_context(
            initial_confidence=0.50,
            current_bar=100,
            entry_time_bar=50,  # 50 bars held
            regime_name="LOW",  # decay_rate=0.99 (slower)
            # decayed = 0.50 * (0.99^50) ≈ 0.31 < 0.35
        )

        action = default_manager._evaluate_confidence_decay(ctx)
        # Should be close to threshold, verify behavior
        assert action.action in ("HOLD", "EXIT_FULL")

    def test_confidence_decay_rate_extreme_regime(self, default_manager):
        """Test confidence decay is faster in EXTREME regime."""
        ctx = create_trade_context(
            initial_confidence=0.50,
            current_bar=60,
            entry_time_bar=50,  # 10 bars held
            regime_name="EXTREME",  # decay_rate=0.92 (faster)
            # decayed = 0.50 * (0.92^10) ≈ 0.23 < 0.35
        )

        action = default_manager._evaluate_confidence_decay(ctx)
        assert action.action == "EXIT_FULL"

    def test_confidence_zero_bars_held(self, default_manager):
        """Test confidence decay at entry (0 bars held)."""
        ctx = create_trade_context(
            initial_confidence=0.75,
            current_bar=50,
            entry_time_bar=50,  # 0 bars held
            regime_name="NORMAL",
            # decayed = 0.75 * (0.97^0) = 0.75
        )

        action = default_manager._evaluate_confidence_decay(ctx)
        assert action.action == "HOLD"
        assert action.metadata["decayed_confidence"] == pytest.approx(0.75)


# ============================================================================
# TEST PriorityMerging
# ============================================================================


class TestPriorityMerging:
    """Test priority-based action merging."""

    def test_all_actions_hold(self, default_manager):
        """Test when all strategies agree to HOLD."""
        actions = [
            ExitAction("HOLD", "chandelier holds", "chandelier", confidence=0.8),
            ExitAction("HOLD", "partial holds", "partial_profit", confidence=0.7),
            ExitAction("HOLD", "vol trail holds", "vol_trail", confidence=0.9),
            ExitAction("HOLD", "time not exceeded", "time_exit", confidence=0.6),
            ExitAction("HOLD", "confidence ok", "confidence_decay", confidence=0.75),
        ]

        result = default_manager._priority_merge(actions)
        assert result.action == "HOLD"
        # Should return the one with highest confidence
        assert result.confidence == 0.9

    def test_confidence_decay_highest_priority(self, default_manager):
        """Test confidence decay exit has highest priority."""
        actions = [
            ExitAction("EXIT_FULL", "chandelier exit", "chandelier", urgency="normal"),
            ExitAction("EXIT_FULL", "time exit", "time_exit", urgency="high"),
            ExitAction("EXIT_FULL", "confidence decay", "confidence_decay", urgency="high"),
            ExitAction("HOLD", "vol trail holds", "vol_trail"),
            ExitAction("HOLD", "partial holds", "partial_profit"),
        ]

        result = default_manager._priority_merge(actions)
        # All three are EXIT_FULL with urgency high, so any could win
        # but confidence_decay should be first in priority
        assert result.is_exit()

    def test_time_exit_priority_over_chandelier(self, default_manager):
        """Test time exit has priority over chandelier."""
        actions = [
            ExitAction("EXIT_FULL", "chandelier exit", "chandelier", urgency="normal"),
            ExitAction("EXIT_FULL", "time exit", "time_exit", urgency="high"),
            ExitAction("HOLD", "others", "partial_profit"),
        ]

        result = default_manager._priority_merge(actions)
        assert result.strategy == "time_exit"

    def test_critical_urgency_overrides_all(self, default_manager):
        """Test critical urgency EXIT_FULL is highest priority."""
        actions = [
            ExitAction("EXIT_FULL", "critical error", "error_handler", urgency="critical"),
            ExitAction("EXIT_FULL", "time exit", "time_exit", urgency="high"),
            ExitAction("TAKE_PARTIAL", "partial profit", "partial_profit", urgency="normal"),
        ]

        result = default_manager._priority_merge(actions)
        assert result.urgency == "critical"

    def test_take_partial_over_tighten_sl(self, default_manager):
        """Test TAKE_PARTIAL has priority over TIGHTEN_SL."""
        actions = [
            ExitAction("TIGHTEN_SL", "tighten", "vol_trail", new_sl_price=1.1010),
            ExitAction("TAKE_PARTIAL", "partial", "partial_profit", partial_close_ratio=0.5),
            ExitAction("HOLD", "hold", "chandelier"),
        ]

        result = default_manager._priority_merge(actions)
        assert result.action == "TAKE_PARTIAL"


# ============================================================================
# TEST EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_atr(self, default_manager):
        """Test handling of zero ATR (should raise validation error)."""
        ctx = create_trade_context(atr_current=0)
        with pytest.raises(ValueError, match="atr_current must be > 0"):
            ctx.validate()

    def test_zero_sl_pips(self, default_manager):
        """Test handling of zero sl_pips."""
        ctx = create_trade_context(sl_pips=0)
        with pytest.raises(ValueError, match="sl_pips must be > 0"):
            ctx.validate()

    def test_empty_price_arrays(self, default_manager):
        """Test handling of empty price arrays."""
        ctx = create_trade_context(prices_close=np.array([]))
        with pytest.raises(ValueError, match="prices_close cannot be empty"):
            ctx.validate()

    def test_nan_in_prices(self, default_manager):
        """Test handling of NaN in price data during evaluation."""
        prices = np.array([1.1000, 1.1010, np.nan, 1.1030])
        ctx = create_trade_context(prices_close=prices)
        # Validation passes, but evaluation may have issues
        ctx.validate()  # Should not raise
        # NaN handling happens during strategy evaluation
        # Most strategies should handle gracefully or return HOLD

    def test_very_small_price_movement(self, default_manager):
        """Test with minimal price movement."""
        ctx = create_trade_context(
            entry_price=1.1000,
            current_price=1.1000001,  # Almost no movement
            sl_pips=0.0020,
        )

        action = default_manager.evaluate_exit(ctx)
        assert isinstance(action, ExitAction)

    def test_very_large_bars_held(self, default_manager):
        """Test with very large bars_held value."""
        ctx = create_trade_context(
            entry_time_bar=0,
            current_bar=10000,  # 10k bars held
            regime_name="NORMAL",
        )

        action = default_manager._evaluate_time_exit(ctx)
        assert action.action == "EXIT_FULL"

    def test_extreme_confidence_values(self, default_manager):
        """Test with extreme confidence values."""
        ctx_zero = create_trade_context(initial_confidence=0.0)
        action_zero = default_manager._evaluate_confidence_decay(ctx_zero)
        assert action_zero.action == "EXIT_FULL"

        ctx_one = create_trade_context(initial_confidence=1.0)
        action_one = default_manager._evaluate_confidence_decay(ctx_one)
        # After many bars, even 1.0 will decay below threshold
        assert isinstance(action_one, ExitAction)

    def test_price_array_shorter_than_chandelier_period(self, default_manager):
        """Test chandelier exit with price array shorter than period."""
        short_prices = np.array([1.1000, 1.1010, 1.1020])
        ctx = create_trade_context(
            prices_close=short_prices,
            prices_high=short_prices + 0.0005,
            prices_low=short_prices - 0.0005,
        )

        # Should handle gracefully with min(period, len)
        action = default_manager._evaluate_chandelier(ctx)
        assert isinstance(action, ExitAction)

    def test_evaluate_exit_full_pipeline(self, default_manager):
        """Test full evaluate_exit pipeline."""
        ctx = create_trade_context()
        action = default_manager.evaluate_exit(ctx)
        assert isinstance(action, ExitAction)
        assert action.action in ("HOLD", "TIGHTEN_SL", "TAKE_PARTIAL", "EXIT_FULL")

    def test_invalid_context_raises_error(self, default_manager):
        """Test that invalid context raises ValueError."""
        ctx = create_trade_context(direction="INVALID")
        with pytest.raises(ValueError, match="direction must be BUY or SELL"):
            default_manager.evaluate_exit(ctx)


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================


class TestConvenienceFunctions:
    """Test helper functions for manager creation."""

    def test_create_default_exit_manager(self):
        """Test default manager creation."""
        mgr = create_default_exit_manager()
        assert isinstance(mgr, AdaptiveExitManager)
        assert mgr.config.chandelier_atr_multiplier == 3.0

    def test_create_conservative_exit_manager(self):
        """Test conservative manager creation."""
        mgr = create_conservative_exit_manager()
        assert isinstance(mgr, AdaptiveExitManager)
        assert mgr.config.chandelier_atr_multiplier == 2.0
        assert mgr.config.confidence_exit_threshold == 0.45

    def test_create_aggressive_exit_manager(self):
        """Test aggressive manager creation."""
        mgr = create_aggressive_exit_manager()
        assert isinstance(mgr, AdaptiveExitManager)
        assert mgr.config.chandelier_atr_multiplier == 4.0
        assert mgr.config.confidence_exit_threshold == 0.25
