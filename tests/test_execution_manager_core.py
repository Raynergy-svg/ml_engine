"""Phase 38 (US-242 + US-244): Unit tests for ExecutionManager position sizing,
risk checks, and dataclass contracts.

Tests ExecutionConfig defaults, ExecutionResult construction,
calculate_position_size, can_trade, _check_portfolio_risk_limit,
_check_correlation_exposure, _check_spread_conditions, _calculate_base_tp_pips,
_apply_risk_manager. All tests use mock OANDA client — no real API calls.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.execution import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_manager(**config_overrides):
    """Create ExecutionManager with mock OANDA client."""
    cfg = ExecutionConfig(**config_overrides)
    mock_oanda = MagicMock()
    return ExecutionManager(config=cfg, oanda_client=mock_oanda)


# ---------------------------------------------------------------------------
# Tests: ExecutionConfig defaults
# ---------------------------------------------------------------------------
class TestExecutionConfig:
    def test_default_risk_per_trade(self):
        cfg = ExecutionConfig()
        assert cfg.risk_per_trade_pct == 0.05

    def test_default_leverage(self):
        cfg = ExecutionConfig()
        assert cfg.leverage == 50

    def test_default_max_trades_per_day(self):
        cfg = ExecutionConfig()
        assert cfg.max_trades_per_day == 30

    def test_default_max_open_risk_pct(self):
        cfg = ExecutionConfig()
        assert cfg.max_open_risk_pct == 0.15

    def test_default_atr_multipliers(self):
        cfg = ExecutionConfig()
        assert cfg.atr_sl_multiplier == 1.0
        assert cfg.atr_tp_multiplier == 1.5

    def test_default_sl_tp_bounds(self):
        cfg = ExecutionConfig()
        assert cfg.min_sl_pips == 15.0
        assert cfg.max_sl_pips == 15.0
        assert cfg.min_tp_pips == 20.0
        assert cfg.max_tp_pips == 30.0

    def test_default_position_sizing_enabled(self):
        cfg = ExecutionConfig()
        assert cfg.position_sizing_enabled is True
        assert cfg.aggressive_mode is True

    def test_custom_config(self):
        cfg = ExecutionConfig(
            risk_per_trade_pct=0.02,
            leverage=100,
            max_trades_per_day=10,
        )
        assert cfg.risk_per_trade_pct == 0.02
        assert cfg.leverage == 100
        assert cfg.max_trades_per_day == 10

    def test_execution_recovery_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.max_order_attempts == 2
        assert cfg.rejection_downsize_factor == 0.5
        assert cfg.min_lot_size == 0.01
        assert cfg.max_lot_size == 50.0


# ---------------------------------------------------------------------------
# Tests: ExecutionResult dataclass
# ---------------------------------------------------------------------------
class TestExecutionResult:
    def test_success_result(self):
        result = ExecutionResult(
            success=True,
            trade_id="12345",
            fill_price=1.1050,
            units=100000,
            lots=1.0,
            sl_price=1.1000,
            tp_price=1.1100,
        )
        assert result.success is True
        assert result.trade_id == "12345"
        assert result.lots == 1.0

    def test_failure_result(self):
        result = ExecutionResult(
            success=False,
            error="Insufficient margin",
        )
        assert result.success is False
        assert result.error == "Insufficient margin"
        assert result.trade_id is None

    def test_default_regime_fields(self):
        result = ExecutionResult(success=True)
        assert result.regime_scale == 1.0
        assert result.regime_name == "UNKNOWN"
        assert result.aggressive_scaling_reason == ""

    def test_default_fill_status(self):
        result = ExecutionResult(success=True)
        assert result.fill_status == "UNKNOWN"
        assert result.slippage_pips == 0.0

    def test_confidence_level_default(self):
        result = ExecutionResult(success=True)
        assert result.confidence_level == "medium"


# ---------------------------------------------------------------------------
# Tests: ExecutionManager init
# ---------------------------------------------------------------------------
class TestExecutionManagerInit:
    def test_default_init(self):
        mgr = _make_manager()
        assert mgr.config.risk_per_trade_pct == 0.05
        assert mgr._oanda is not None

    def test_init_with_custom_config(self):
        mgr = _make_manager(risk_per_trade_pct=0.01, leverage=100)
        assert mgr.config.risk_per_trade_pct == 0.01
        assert mgr.config.leverage == 100

    def test_lazy_components_none(self):
        mgr = _make_manager()
        assert mgr._position_sizer is None
        assert mgr._risk_manager is None
        assert mgr._rl_sizer is None


# ---------------------------------------------------------------------------
# Tests: can_trade
# ---------------------------------------------------------------------------
class TestCanTrade:
    def test_can_trade_below_limit(self):
        mgr = _make_manager(max_trades_per_day=30)
        # Mock fetch_trades_today to return 5
        with patch.object(mgr, 'fetch_trades_today', return_value=5):
            allowed, reason = mgr.can_trade()
        assert allowed is True
        assert "5/30" in reason

    def test_cannot_trade_at_limit(self):
        mgr = _make_manager(max_trades_per_day=10)
        with patch.object(mgr, 'fetch_trades_today', return_value=10):
            allowed, reason = mgr.can_trade()
        assert allowed is False
        assert "limit" in reason.lower()

    def test_cannot_trade_above_limit(self):
        mgr = _make_manager(max_trades_per_day=5)
        with patch.object(mgr, 'fetch_trades_today', return_value=8):
            allowed, reason = mgr.can_trade()
        assert allowed is False


# ---------------------------------------------------------------------------
# Tests: _check_portfolio_risk_limit
# ---------------------------------------------------------------------------
class TestPortfolioRiskLimit:
    def test_no_open_trades_ok(self):
        mgr = _make_manager()
        with patch.object(mgr, 'monitor_open_trades', return_value=[]):
            allowed, reason = mgr._check_portfolio_risk_limit()
        assert allowed is True
        assert "no open trades" in reason

    def test_risk_limit_disabled(self):
        mgr = _make_manager(max_open_risk_pct=0)
        allowed, reason = mgr._check_portfolio_risk_limit()
        assert allowed is True
        assert "disabled" in reason

    def test_within_risk_limit(self):
        mgr = _make_manager(max_open_risk_pct=0.15, account_equity=100000)
        statuses = [
            {"pair": "EUR_USD", "sl_dist_pips": 15, "units": 10000},
        ]
        with patch.object(mgr, 'monitor_open_trades', return_value=statuses):
            with patch.object(mgr, 'fetch_live_nav', return_value=100000.0):
                allowed, reason = mgr._check_portfolio_risk_limit()
        assert allowed is True

    def test_exceeds_risk_limit(self):
        mgr = _make_manager(max_open_risk_pct=0.01, account_equity=10000)
        # Large position: 15 pips * 0.0001 * 500000 units = $750
        # On $10000 NAV = 7.5% >> 1% limit
        statuses = [
            {"pair": "EUR_USD", "sl_dist_pips": 15, "units": 500000},
        ]
        with patch.object(mgr, 'monitor_open_trades', return_value=statuses):
            with patch.object(mgr, 'fetch_live_nav', return_value=10000.0):
                allowed, reason = mgr._check_portfolio_risk_limit()
        assert allowed is False
        assert "risk limit" in reason.lower()

    def test_risk_check_exception_returns_ok(self):
        """On error, risk check defaults to allowing trade (fail-open)."""
        mgr = _make_manager()
        with patch.object(mgr, 'monitor_open_trades', side_effect=Exception("API down")):
            allowed, reason = mgr._check_portfolio_risk_limit()
        assert allowed is True
        assert "unavailable" in reason


# ---------------------------------------------------------------------------
# Tests: _check_spread_conditions
# ---------------------------------------------------------------------------
class TestCheckSpreadConditions:
    def test_no_oanda_client_skips(self):
        cfg = ExecutionConfig()
        mgr = ExecutionManager(config=cfg, oanda_client=None)
        ok, spread, reason = mgr._check_spread_conditions("EUR_USD")
        assert ok is True
        assert "skipped" in reason

    def test_acceptable_spread(self):
        mgr = _make_manager()
        mgr._oanda.get_pricing.return_value = [{
            "asks": [{"price": "1.1052"}],
            "bids": [{"price": "1.1050"}],
        }]
        ok, spread, reason = mgr._check_spread_conditions("EUR_USD")
        assert ok is True
        assert spread == 2.0  # (1.1052-1.1050)/0.0001 = 2.0 pips
        assert "OK" in reason

    def test_wide_spread_blocked(self):
        mgr = _make_manager()
        mgr.config.max_spread_pips = 2.0
        mgr._oanda.get_pricing.return_value = [{
            "asks": [{"price": "1.1060"}],
            "bids": [{"price": "1.1050"}],
        }]
        ok, spread, reason = mgr._check_spread_conditions("EUR_USD")
        assert ok is False
        assert spread == 10.0  # (1.1060-1.1050)/0.0001
        assert "too wide" in reason.lower()

    def test_no_pricing_data(self):
        mgr = _make_manager()
        mgr._oanda.get_pricing.return_value = None
        ok, spread, reason = mgr._check_spread_conditions("EUR_USD")
        assert ok is True
        assert "skipped" in reason

    def test_pricing_exception_handled(self):
        mgr = _make_manager()
        mgr._oanda.get_pricing.side_effect = Exception("Connection refused")
        ok, spread, reason = mgr._check_spread_conditions("EUR_USD")
        assert ok is True
        assert "unavailable" in reason


# ---------------------------------------------------------------------------
# Tests: _calculate_base_tp_pips
# ---------------------------------------------------------------------------
class TestCalculateBaseTPPips:
    def test_normal_atr_calculation(self):
        mgr = _make_manager(atr_tp_multiplier=1.5, min_tp_pips=20.0, max_tp_pips=30.0)
        # ATR=0.0025, pip=0.0001 → base_tp = (0.0025*1.5)/0.0001 = 37.5 → clamped to 30
        tp = mgr._calculate_base_tp_pips(atr=0.0025, pip_value=0.0001, confidence=0.5)
        assert tp == 30.0  # Clamped to max

    def test_small_atr_uses_minimum(self):
        mgr = _make_manager(min_tp_pips=20.0)
        # ATR=0.0005, pip=0.0001 → base_tp = (0.0005*1.5)/0.0001 = 7.5 → clamped to 20
        tp = mgr._calculate_base_tp_pips(atr=0.0005, pip_value=0.0001, confidence=0.3)
        assert tp == 20.0  # Clamped to min

    def test_zero_atr_uses_fallback(self):
        mgr = _make_manager(fallback_tp_pips=25.0, min_tp_pips=20.0, max_tp_pips=30.0)
        tp = mgr._calculate_base_tp_pips(atr=0.0, pip_value=0.0001, confidence=0.5)
        assert tp == 25.0  # fallback_tp_pips

    def test_high_prob_bonus_added(self):
        mgr = _make_manager(
            atr_tp_multiplier=1.5, min_tp_pips=20.0, max_tp_pips=30.0,
            high_prob_threshold=0.65, high_prob_tp_bonus=20.0,
        )
        # Base = 30 (clamped), bonus +20 = 50
        tp = mgr._calculate_base_tp_pips(atr=0.0025, pip_value=0.0001, confidence=0.70)
        assert tp == 50.0

    def test_no_bonus_below_threshold(self):
        mgr = _make_manager(
            atr_tp_multiplier=1.5, min_tp_pips=20.0, max_tp_pips=30.0,
            high_prob_threshold=0.65, high_prob_tp_bonus=20.0,
        )
        tp = mgr._calculate_base_tp_pips(atr=0.0025, pip_value=0.0001, confidence=0.60)
        assert tp == 30.0  # No bonus


# ---------------------------------------------------------------------------
# Tests: _apply_risk_manager
# ---------------------------------------------------------------------------
class TestApplyRiskManager:
    def test_no_risk_manager_returns_original(self):
        mgr = _make_manager()
        mgr._risk_manager = None
        sl, tp, level = mgr._apply_risk_manager("EUR_USD", 0.7, 15.0, 30.0)
        assert sl == 15.0
        assert tp == 30.0
        assert level == "medium"

    def test_risk_manager_applied(self):
        mgr = _make_manager()
        mock_rm = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.stop_loss_pips = 12.0
        mock_result.take_profit_pips = 35.0
        mock_result.confidence_level = "high"
        mock_rm.calculate_risk_levels.return_value = mock_result
        mgr._risk_manager = mock_rm

        sl, tp, level = mgr._apply_risk_manager("EUR_USD", 0.8, 15.0, 30.0)
        assert sl == 12.0
        assert tp == 35.0
        assert level == "high"

    def test_risk_manager_invalid_falls_back(self):
        mgr = _make_manager()
        mock_rm = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_rm.calculate_risk_levels.return_value = mock_result
        mgr._risk_manager = mock_rm

        sl, tp, level = mgr._apply_risk_manager("EUR_USD", 0.5, 15.0, 30.0)
        assert sl == 15.0
        assert tp == 30.0
        assert level == "medium"

    def test_risk_manager_exception_falls_back(self):
        mgr = _make_manager()
        mock_rm = MagicMock()
        mock_rm.calculate_risk_levels.side_effect = Exception("Model error")
        mgr._risk_manager = mock_rm

        sl, tp, level = mgr._apply_risk_manager("EUR_USD", 0.5, 15.0, 30.0)
        assert sl == 15.0
        assert tp == 30.0


# ---------------------------------------------------------------------------
# Tests: _check_correlation_exposure
# ---------------------------------------------------------------------------
class TestCheckCorrelationExposure:
    def test_correlation_check_no_open_trades(self):
        mgr = _make_manager()
        with patch.object(mgr, 'monitor_open_trades', return_value=[]):
            with patch('src.scanner.execution.ExecutionManager._check_correlation_exposure') as mock:
                # Call real method to test flow
                pass
        # Use actual method with mock
        mgr2 = _make_manager()
        with patch.object(mgr2, 'monitor_open_trades', return_value=[]):
            allowed, reason = mgr2._check_correlation_exposure("EUR_USD")
        assert allowed is True

    def test_correlation_limit_disabled(self):
        mgr = _make_manager()
        mgr.config.max_correlated_exposure = 0
        allowed, reason = mgr._check_correlation_exposure("EUR_USD")
        assert allowed is True
        assert "disabled" in reason


# ---------------------------------------------------------------------------
# Tests: _init_oanda_client
# ---------------------------------------------------------------------------
class TestInitOandaClient:
    def test_already_initialized(self):
        mgr = _make_manager()
        assert mgr._init_oanda_client() is True

    def test_none_client_tries_import(self):
        cfg = ExecutionConfig()
        mgr = ExecutionManager(config=cfg, oanda_client=None)
        # Will fail because no OANDA env vars, but should not crash
        result = mgr._init_oanda_client()
        # Either True (if env is set) or False (expected in test env)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tests: Slippage history tracking
# ---------------------------------------------------------------------------
class TestSlippageTracking:
    def test_default_slippage_window(self):
        mgr = _make_manager()
        assert mgr._slippage_window == 20
        assert mgr._slippage_threshold_pips == 2.5
        assert mgr._slippage_size_reduction == 0.15

    def test_empty_slippage_history(self):
        mgr = _make_manager()
        assert mgr._slippage_history == {}
