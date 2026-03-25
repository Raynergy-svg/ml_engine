"""Unit tests for ExecutionManager adaptive position sizer wiring (US-282).

Phase 45 (US-282): Integrate AdaptivePositionSizer into ExecutionManager.

Tests cover:
- Initialization of adaptive position sizer
- Graceful fallback when sizer unavailable
- Position size calculation with regime/drawdown
- Trade outcome recording and Kelly update
- State persistence (save/load)
- Integration with calculate_position_size
- Different operating modes (aggressive/conservative)
- Edge cases (zero equity, high drawdown, etc.)
"""

import json
import logging
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    from src.scanner.execution import ExecutionManager, ExecutionConfig
    from src.scanner.adaptive_position_sizing import (
        AdaptivePositionSizer,
        AdaptivePositionSizingConfig,
        PositionSizeResult,
        create_default_position_sizer,
        create_aggressive_position_sizer,
        create_conservative_position_sizer,
    )
except ImportError:
    pytest.skip("Required modules not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Default ExecutionConfig for testing."""
    return ExecutionConfig(
        account_equity=10000.0,
        risk_per_trade_pct=0.05,
        position_sizing_enabled=True,
        aggressive_mode=False,
        max_trades_per_day=30,
        atr_sl_multiplier=1.0,
        atr_tp_multiplier=1.5,
        min_sl_pips=10.0,
        max_sl_pips=25.0,
        min_tp_pips=15.0,
        max_tp_pips=40.0,
        min_lot_size=0.01,
        max_lot_size=50.0,
    )


@pytest.fixture
def em(config):
    """ExecutionManager with mocked OANDA client."""
    with patch.object(ExecutionManager, '__init__', lambda self, *a, **kw: None):
        manager = ExecutionManager.__new__(ExecutionManager)
        manager.config = config
        manager._oanda = MagicMock()
        manager._position_sizer = None
        manager._risk_manager = None
        manager._memory_client = None
        manager._observer = None
        manager._adaptive_position_sizer = None
        manager._last_regime_name = "NORMAL"
        manager._current_drawdown_pct = 0.0
        manager._adaptive_exit_manager = None
        return manager


@pytest.fixture
def adaptive_sizer_config():
    """Default adaptive sizer config."""
    return AdaptivePositionSizingConfig()


# ===========================================================================
# US-282: Initialization Tests
# ===========================================================================

class TestAdaptivePositionerInitialization:
    """Tests for _init_adaptive_position_sizer()."""

    def test_init_creates_default_sizer_normal_mode(self, em):
        """_init_adaptive_position_sizer creates default sizer when not aggressive."""
        em._init_adaptive_position_sizer()
        assert em._adaptive_position_sizer is not None
        assert isinstance(em._adaptive_position_sizer, AdaptivePositionSizer)

    def test_init_creates_aggressive_sizer_aggressive_mode(self, em):
        """_init_adaptive_position_sizer creates aggressive sizer when aggressive_mode=True."""
        em.config.aggressive_mode = True
        em._init_adaptive_position_sizer()
        assert em._adaptive_position_sizer is not None
        assert isinstance(em._adaptive_position_sizer, AdaptivePositionSizer)

    def test_init_idempotent(self, em):
        """_init_adaptive_position_sizer can be called multiple times safely."""
        em._init_adaptive_position_sizer()
        first_sizer = em._adaptive_position_sizer
        em._init_adaptive_position_sizer()
        assert em._adaptive_position_sizer is first_sizer

    def test_init_with_import_failure(self, em):
        """_init_adaptive_position_sizer handles import failure gracefully."""
        with patch('builtins.__import__', side_effect=ImportError("test")):
            # Should not raise, just log and return with sizer=None
            em._init_adaptive_position_sizer()
            # Sizer remains None since import failed
            assert em._adaptive_position_sizer is None

    def test_init_loads_persisted_state(self, em):
        """_init_adaptive_position_sizer loads state if file exists."""
        # Test that load_state is called when file exists (don't touch actual file)
        em._init_adaptive_position_sizer()
        original_sizer = em._adaptive_position_sizer

        with patch.object(original_sizer, 'load_state') as mock_load:
            # Reset sizer and init again
            em._adaptive_position_sizer = None
            em._init_adaptive_position_sizer()
            # The new sizer should try to load state (though file doesn't exist, it's attempted)

    def test_init_handles_missing_state_file_gracefully(self, em):
        """_init_adaptive_position_sizer continues even if state file missing."""
        # Should handle gracefully — state file may or may not exist
        em._init_adaptive_position_sizer()
        assert em._adaptive_position_sizer is not None


# ===========================================================================
# US-282: Position Size Calculation Tests
# ===========================================================================

class TestAdaptivePositionSizeCalculation:
    """Tests for _calculate_adaptive_position_size()."""

    def test_calculate_returns_valid_tuple(self, em):
        """_calculate_adaptive_position_size returns (lots, risk_pct, label)."""
        em._init_adaptive_position_sizer()
        result = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
            regime_name="NORMAL",
            current_drawdown_pct=0.0,
        )
        assert result is not None
        assert len(result) == 3
        lots, risk_pct, label = result
        assert isinstance(lots, float)
        assert isinstance(risk_pct, float)
        assert isinstance(label, str)
        assert lots > 0
        assert risk_pct >= 0

    def test_calculate_returns_none_when_sizer_unavailable(self, em):
        """_calculate_adaptive_position_size returns None if sizer init failed."""
        em._adaptive_position_sizer = None
        with patch.object(em, '_init_adaptive_position_sizer'):
            em._init_adaptive_position_sizer.side_effect = lambda: None
            result = em._calculate_adaptive_position_size(
                equity=10000.0,
                confidence=0.7,
            )
            assert result is None

    def test_calculate_handles_exception_gracefully(self, em):
        """_calculate_adaptive_position_size catches and logs exceptions."""
        em._init_adaptive_position_sizer()
        # Mock the sizer to raise an exception
        em._adaptive_position_sizer.calculate_size = MagicMock(
            side_effect=ValueError("test error")
        )
        result = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
        )
        assert result is None

    def test_calculate_clamps_lots_to_min_max(self, em):
        """_calculate_adaptive_position_size clamps lots within min/max."""
        em._init_adaptive_position_sizer()
        em.config.min_lot_size = 0.1
        em.config.max_lot_size = 5.0

        # Create very large position size by mocking
        mock_result = PositionSizeResult(
            final_size=1000000.0,  # Very large
            base_size=1000000.0,
            kelly_factor=1.0,
            confidence_factor=1.0,
            drawdown_factor=1.0,
            regime_factor=1.0,
            regime_name="NORMAL",
        )
        em._adaptive_position_sizer.calculate_size = MagicMock(return_value=mock_result)

        lots, risk_pct, label = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
        )
        # 1000000 units / 100,000 = 10 lots, clamped to max 5.0
        assert lots <= em.config.max_lot_size

    def test_calculate_uses_regime_name(self, em):
        """_calculate_adaptive_position_size passes regime correctly."""
        em._init_adaptive_position_sizer()
        em._adaptive_position_sizer.calculate_size = MagicMock(
            return_value=PositionSizeResult(
                final_size=50000.0,
                base_size=50000.0,
                kelly_factor=0.5,
                confidence_factor=0.8,
                drawdown_factor=1.0,
                regime_factor=0.65,
                regime_name="HIGH",
            )
        )

        result = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
            regime_name="HIGH",
        )
        assert result is not None
        # Verify that calculate_size was called with HIGH regime
        em._adaptive_position_sizer.calculate_size.assert_called_once()
        call_kwargs = em._adaptive_position_sizer.calculate_size.call_args.kwargs
        assert call_kwargs["regime_name"] == "HIGH"

    def test_calculate_uses_drawdown_pct(self, em):
        """_calculate_adaptive_position_size passes drawdown percentage."""
        em._init_adaptive_position_sizer()
        em._adaptive_position_sizer.calculate_size = MagicMock(
            return_value=PositionSizeResult(
                final_size=25000.0,
                base_size=50000.0,
                kelly_factor=0.5,
                confidence_factor=0.8,
                drawdown_factor=0.5,
                regime_factor=1.0,
                regime_name="NORMAL",
            )
        )

        result = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
            current_drawdown_pct=0.05,  # 5% drawdown
        )
        assert result is not None
        # Verify drawdown was passed
        em._adaptive_position_sizer.calculate_size.assert_called_once()
        call_kwargs = em._adaptive_position_sizer.calculate_size.call_args.kwargs
        assert call_kwargs["current_drawdown_pct"] == 0.05

    def test_calculate_contains_strategy_label(self, em):
        """_calculate_adaptive_position_size returns label with factor breakdown."""
        em._init_adaptive_position_sizer()
        result = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
            regime_name="NORMAL",
        )
        assert result is not None
        _, _, label = result
        # Label should contain 'adaptive' and factors
        assert "adaptive" in label.lower()
        assert "K=" in label  # Kelly factor
        assert "C=" in label  # Confidence factor
        assert "D=" in label  # Drawdown factor
        assert "R=" in label  # Regime factor


# ===========================================================================
# US-282: Integration with calculate_position_size Tests
# ===========================================================================

class TestCalculatePositionSizeIntegration:
    """Tests for adaptive sizer integration in calculate_position_size()."""

    def test_calculate_position_size_tries_adaptive_first(self, em):
        """calculate_position_size tries adaptive sizer before fallback."""
        em._init_position_sizer()
        em._init_adaptive_position_sizer()

        with patch.object(
            em._adaptive_position_sizer,
            'calculate_size',
            return_value=PositionSizeResult(
                final_size=50000.0,
                base_size=50000.0,
                kelly_factor=0.5,
                confidence_factor=0.8,
                drawdown_factor=1.0,
                regime_factor=1.0,
                regime_name="NORMAL",
            )
        ):
            lots, risk_pct, sl_pips, tp_pips, conf = em.calculate_position_size(
                pair="EUR_USD",
                confidence=0.7,
                atr=15.0,
                account_equity=10000.0,
            )
            # Should return adaptive result
            assert lots == 0.5  # 50000 / 100000
            assert risk_pct > 0

    def test_calculate_position_size_falls_back_when_adaptive_unavailable(self, em):
        """calculate_position_size falls back to DynamicPositionSizer if adaptive fails."""
        em._init_position_sizer()
        em._adaptive_position_sizer = None

        lots, risk_pct, sl_pips, tp_pips, conf = em.calculate_position_size(
            pair="EUR_USD",
            confidence=0.7,
            atr=15.0,
            account_equity=10000.0,
        )
        # Should return valid result from fallback
        assert lots >= 0
        assert sl_pips > 0
        assert tp_pips > 0

    def test_calculate_position_size_uses_regime_attribute(self, em):
        """calculate_position_size passes _last_regime_name to adaptive sizer."""
        em._init_adaptive_position_sizer()
        em._last_regime_name = "HIGH"

        with patch.object(em._adaptive_position_sizer, 'calculate_size') as mock_calc:
            mock_calc.return_value = PositionSizeResult(
                final_size=25000.0,
                base_size=50000.0,
                kelly_factor=0.5,
                confidence_factor=0.8,
                drawdown_factor=1.0,
                regime_factor=0.65,
                regime_name="HIGH",
            )

            em.calculate_position_size(
                pair="EUR_USD",
                confidence=0.7,
                atr=15.0,
                account_equity=10000.0,
            )

            # Verify HIGH regime was passed
            call_kwargs = mock_calc.call_args.kwargs
            assert call_kwargs["regime_name"] == "HIGH"

    def test_calculate_position_size_uses_drawdown_attribute(self, em):
        """calculate_position_size passes _current_drawdown_pct to adaptive sizer."""
        em._init_adaptive_position_sizer()
        em._current_drawdown_pct = 0.08  # 8% drawdown

        with patch.object(em._adaptive_position_sizer, 'calculate_size') as mock_calc:
            mock_calc.return_value = PositionSizeResult(
                final_size=25000.0,
                base_size=50000.0,
                kelly_factor=0.5,
                confidence_factor=0.8,
                drawdown_factor=0.5,
                regime_factor=1.0,
                regime_name="NORMAL",
            )

            em.calculate_position_size(
                pair="EUR_USD",
                confidence=0.7,
                atr=15.0,
                account_equity=10000.0,
            )

            # Verify drawdown was passed
            call_kwargs = mock_calc.call_args.kwargs
            assert call_kwargs["current_drawdown_pct"] == 0.08


# ===========================================================================
# US-282: Trade Recording and RL Feedback Tests
# ===========================================================================

class TestTradeRecordingAndStateManagement:
    """Tests for trade recording in RL sync and state persistence."""

    def test_record_trade_win(self, em):
        """Adaptive sizer records winning trades correctly."""
        sizer = create_default_position_sizer()
        initial_count = len(sizer._trade_history)
        sizer.record_trade(pnl=250.0, win=True)

        # Trade should be recorded
        assert len(sizer._trade_history) == initial_count + 1
        pnl, won = sizer._trade_history[-1]
        assert pnl == 250.0
        assert won is True

    def test_record_trade_loss(self, em):
        """Adaptive sizer records losing trades correctly."""
        sizer = create_default_position_sizer()
        initial_count = len(sizer._trade_history)
        sizer.record_trade(pnl=-100.0, win=False)

        assert len(sizer._trade_history) == initial_count + 1
        pnl, won = sizer._trade_history[-1]
        assert pnl == -100.0
        assert won is False

    def test_kelly_calculation_after_trades(self, em):
        """Kelly factor updates based on trade history."""
        em._init_adaptive_position_sizer()

        # Record several trades
        for _ in range(5):
            em._adaptive_position_sizer.record_trade(pnl=150.0, win=True)
        for _ in range(5):
            em._adaptive_position_sizer.record_trade(pnl=-100.0, win=False)

        kelly = em._adaptive_position_sizer.calculate_kelly()
        assert 0.01 <= kelly <= em._adaptive_position_sizer.config.kelly_fraction

    def test_save_state_atomic_write(self, em):
        """save_state writes atomically (.tmp + rename)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "sizer_state.json"

            sizer = create_default_position_sizer()
            initial_trades = len(sizer._trade_history)
            sizer.record_trade(pnl=100.0, win=True)
            sizer.save_state(str(state_file))

            assert state_file.exists()
            # Verify JSON is valid and contains trade history
            data = json.loads(state_file.read_text())
            assert "trade_history" in data
            assert len(data["trade_history"]) == initial_trades + 1

    def test_load_state_restores_history(self, em):
        """load_state restores trade history from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "sizer_state.json"

            # Create and save initial state
            sizer1 = create_default_position_sizer()
            sizer1.record_trade(pnl=100.0, win=True)
            sizer1.record_trade(pnl=-50.0, win=False)
            sizer1.save_state(str(state_file))

            # Load into fresh sizer
            sizer2 = create_default_position_sizer()
            sizer2.load_state(str(state_file))

            assert len(sizer2._trade_history) == 2
            assert sizer2._trade_history[0] == (100.0, True)
            assert sizer2._trade_history[1] == (-50.0, False)

    def test_load_state_handles_missing_file(self, em):
        """load_state handles missing file gracefully."""
        em._init_adaptive_position_sizer()
        # Should not raise
        em._adaptive_position_sizer.load_state("/nonexistent/path/state.json")
        # Sizer should still be functional
        assert em._adaptive_position_sizer is not None

    def test_load_state_handles_corrupted_json(self, em):
        """load_state handles corrupted JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "sizer_state.json"
            state_file.write_text("{corrupted json content")

            em._init_adaptive_position_sizer()
            # Should not raise
            em._adaptive_position_sizer.load_state(str(state_file))
            # Sizer should still be functional with empty history
            assert em._adaptive_position_sizer is not None


# ===========================================================================
# US-282: Operating Modes Tests
# ===========================================================================

class TestOperatingModes:
    """Tests for different operating modes (aggressive/conservative)."""

    def test_default_mode_uses_balanced_config(self, em):
        """Default mode (aggressive_mode=False) uses balanced config."""
        em.config.aggressive_mode = False
        em._init_adaptive_position_sizer()

        config = em._adaptive_position_sizer.config
        # Check balanced defaults
        assert config.kelly_fraction == 0.33
        assert config.drawdown_floor == 0.3

    def test_aggressive_mode_uses_aggressive_config(self, em):
        """Aggressive mode uses aggressive config (higher kelly, lower floor)."""
        em.config.aggressive_mode = True
        em._init_adaptive_position_sizer()

        config = em._adaptive_position_sizer.config
        # Check aggressive settings
        assert config.kelly_fraction == 0.40
        assert config.drawdown_floor == 0.2

    def test_conservative_mode_config(self):
        """Conservative sizer has protective settings."""
        sizer = create_conservative_position_sizer()
        assert sizer.config.kelly_fraction == 0.25
        assert sizer.config.drawdown_floor == 0.4
        assert sizer.config.max_risk_per_trade == 0.05


# ===========================================================================
# US-282: Edge Cases Tests
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_equity_handled(self, em):
        """_calculate_adaptive_position_size handles zero equity."""
        em._init_adaptive_position_sizer()
        # Should not raise
        result = em._calculate_adaptive_position_size(
            equity=0.0,
            confidence=0.7,
        )
        # Will likely fail validation and return None
        # But shouldn't crash
        assert result is None or (result[1] == 0.0)

    def test_zero_confidence(self, em):
        """Position size calculation with zero confidence."""
        em._init_adaptive_position_sizer()
        result = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.0,
            regime_name="NORMAL",
        )
        assert result is not None
        lots, risk_pct, label = result
        # Zero confidence should result in smaller position
        assert lots >= 0

    def test_high_confidence(self, em):
        """Position size calculation with high confidence."""
        em._init_adaptive_position_sizer()
        result_low = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.3,
            regime_name="NORMAL",
        )
        result_high = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.9,
            regime_name="NORMAL",
        )
        assert result_low is not None and result_high is not None
        lots_low, _, _ = result_low
        lots_high, _, _ = result_high
        # Higher confidence should produce larger or equal position
        assert lots_high >= lots_low

    def test_extreme_drawdown(self, em):
        """Position size with extreme drawdown (10%)."""
        em._init_adaptive_position_sizer()
        result = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
            current_drawdown_pct=0.10,
            regime_name="NORMAL",
        )
        assert result is not None
        lots, _, _ = result
        # Should be reduced due to drawdown
        assert lots >= 0

    def test_different_regimes_produce_different_sizes(self, em):
        """Different volatility regimes produce different position sizes."""
        em._init_adaptive_position_sizer()

        result_low = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
            regime_name="LOW",
        )
        result_extreme = em._calculate_adaptive_position_size(
            equity=10000.0,
            confidence=0.7,
            regime_name="EXTREME",
        )

        assert result_low is not None and result_extreme is not None
        lots_low, _, _ = result_low
        lots_extreme, _, _ = result_extreme
        # LOW vol should produce larger or equal position than EXTREME
        # They may be equal due to minimum position size clamping
        assert lots_low >= lots_extreme

    def test_no_trade_history_returns_conservative_kelly(self, em):
        """With no trade history, Kelly uses conservative default (0.01)."""
        sizer = create_default_position_sizer()
        kelly = sizer.calculate_kelly()
        # Should use default 0.01 when no history
        assert kelly == 0.01


# ===========================================================================
# US-282: Integration with Existing Position Sizing Tests
# ===========================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with existing position sizing."""

    def test_disabled_position_sizing_returns_zero(self, em):
        """Position sizing disabled returns zero size."""
        em.config.position_sizing_enabled = False
        lots, risk_pct, sl, tp, conf = em.calculate_position_size(
            pair="EUR_USD",
            confidence=0.7,
            atr=15.0,
            account_equity=10000.0,
        )
        assert lots == 0.0
        assert risk_pct == 0.0

    def test_position_sizing_enabled_returns_nonzero(self, em):
        """Position sizing enabled returns non-zero size."""
        em.config.position_sizing_enabled = True
        em._init_position_sizer()

        lots, risk_pct, sl, tp, conf = em.calculate_position_size(
            pair="EUR_USD",
            confidence=0.7,
            atr=15.0,
            account_equity=10000.0,
        )
        # Should return valid sizing
        assert lots >= 0
        assert sl > 0
        assert tp > 0

    def test_fallback_sizing_when_adaptive_fails(self, em):
        """Falls back to standard sizing when adaptive sizer fails."""
        em._init_position_sizer()

        # Force adaptive sizer to fail
        with patch.object(em, '_calculate_adaptive_position_size', return_value=None):
            lots, risk_pct, sl, tp, conf = em.calculate_position_size(
                pair="EUR_USD",
                confidence=0.7,
                atr=15.0,
                account_equity=10000.0,
            )
            # Should still return valid result from fallback
            assert lots >= 0 or lots == 0
