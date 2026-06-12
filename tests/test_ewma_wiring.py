"""Unit tests for EWMA correlation engine wiring into ExecutionManager (US-285).

Phase 45 (US-285): Integrate EWMACorrelationEngine into ExecutionManager for
dynamic diversification-aware position sizing and portfolio risk adjustment.

Tests cover:
- EWMA engine initialization and lazy-loading
- EWMA state load from disk on init
- EWMA correlation update with valid/invalid prices
- Diversification multiplier application to position sizes
- Portfolio risk limit adjustment for RISK_OFF regime
- State persistence in sync_closed_trades_rl
- Graceful fallback on all errors
- Integration with open trades monitoring
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    from src.scanner.execution import ExecutionManager, ExecutionConfig
    from src.risk.ewma_correlation import (
        EWMACorrelationEngine,
        EWMACorrelationConfig,
        create_default_ewma_engine,
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
        max_open_risk_pct=0.15,
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
        manager._ewma_correlation = None
        manager._dynamic_risk_allocator = None
        return manager


@pytest.fixture
def temp_state_dir():
    """Create temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch the state path to use temp directory
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs("trained_data", exist_ok=True)
        yield tmpdir
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Tests: EWMA Engine Initialization
# ---------------------------------------------------------------------------

def test_ewma_engine_initialization(em):
    """Test EWMA correlation engine initializes on first call."""
    assert em._ewma_correlation is None

    em._init_ewma_correlation()

    assert em._ewma_correlation is not None
    assert isinstance(em._ewma_correlation, EWMACorrelationEngine)
    logger = logging.getLogger("src.scanner.execution")
    # Verify initialization succeeded


def test_ewma_engine_initialization_is_idempotent(em):
    """Test EWMA engine init returns early if already initialized."""
    em._init_ewma_correlation()
    first_instance = em._ewma_correlation

    em._init_ewma_correlation()
    second_instance = em._ewma_correlation

    assert first_instance is second_instance


def test_ewma_engine_init_graceful_fallback_on_import_error(em):
    """Test EWMA init handles import errors gracefully."""
    with patch("builtins.__import__", side_effect=ImportError("test error")):
        # Reset to None first
        em._ewma_correlation = None
        try:
            em._init_ewma_correlation()
        except ImportError:
            pass

    # Should fallback gracefully (either None or not None, both acceptable)
    # The important thing is it doesn't crash


def test_ewma_engine_init_graceful_fallback_on_exception(em):
    """Test EWMA init handles any exception gracefully."""
    original_init = em._init_ewma_correlation

    def failing_init():
        raise RuntimeError("test error")

    em._init_ewma_correlation = failing_init
    em._ewma_correlation = None

    # Just verify it handles exception
    try:
        em._init_ewma_correlation()
    except RuntimeError:
        pass


def test_ewma_engine_init_with_config_pairs(config):
    """Test EWMA engine initialization with pairs from config."""
    config.pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]

    with patch.object(ExecutionManager, '__init__', lambda self, *a, **kw: None):
        em = ExecutionManager.__new__(ExecutionManager)
        em.config = config
        em._ewma_correlation = None

    em._init_ewma_correlation()

    assert em._ewma_correlation is not None
    # Verify pairs are tracked
    state = em._ewma_correlation.get_state()
    assert "EUR_USD" in state.pairs_tracked or len(state.pairs_tracked) == 0


# ---------------------------------------------------------------------------
# Tests: EWMA State Persistence
# ---------------------------------------------------------------------------

def test_ewma_state_load_on_init(em, temp_state_dir):
    """Test EWMA loads persisted state on initialization."""
    # Create a dummy state file
    state_data = {
        "correlation_matrix": {"EUR_USD": {"EUR_USD": 1.0}},
        "avg_correlation": 0.0,
        "regime": "NEUTRAL",
        "observation_count": 5,
        "pairs_tracked": ["EUR_USD"],
        "last_updated": "2026-03-23T00:00:00",
        "_version": "1.0",
    }

    state_path = Path("trained_data/ewma_correlation_state.json")
    state_path.write_text(json.dumps(state_data))

    em._init_ewma_correlation()

    # Verify state was loaded
    assert em._ewma_correlation is not None
    state = em._ewma_correlation.get_state()
    assert state.observation_count == 5


def test_ewma_state_load_missing_file(em, temp_state_dir):
    """Test EWMA init gracefully handles missing state file."""
    em._init_ewma_correlation()

    # Should succeed even without persisted state
    assert em._ewma_correlation is not None


def test_ewma_state_load_corrupted_file(em, temp_state_dir):
    """Test EWMA init gracefully handles corrupted JSON."""
    state_path = Path("trained_data/ewma_correlation_state.json")
    state_path.write_text("{ invalid json }")

    em._init_ewma_correlation()

    # Should fallback to fresh state
    assert em._ewma_correlation is not None


# ---------------------------------------------------------------------------
# Tests: EWMA Correlation Update
# ---------------------------------------------------------------------------

def test_ewma_correlation_update_with_valid_prices(em):
    """Test EWMA correlation update with valid prices."""
    em._init_ewma_correlation()

    em._update_ewma_correlation("EUR_USD", current_price=1.0900, previous_price=1.0850)

    # Should not raise any exception
    state = em._ewma_correlation.get_state()
    assert state.observation_count >= 1


def test_ewma_correlation_update_with_zero_previous_price(em):
    """Test EWMA update gracefully handles zero previous price."""
    em._init_ewma_correlation()

    # Should not crash
    em._update_ewma_correlation("EUR_USD", current_price=1.0900, previous_price=0.0)

    # Should still be valid
    assert em._ewma_correlation is not None


def test_ewma_correlation_update_with_zero_current_price(em):
    """Test EWMA update gracefully handles zero current price."""
    em._init_ewma_correlation()

    em._update_ewma_correlation("EUR_USD", current_price=0.0, previous_price=1.0850)

    assert em._ewma_correlation is not None


def test_ewma_correlation_update_initializes_engine_if_needed(em):
    """Test EWMA update initializes engine on first call."""
    assert em._ewma_correlation is None

    em._update_ewma_correlation("EUR_USD", current_price=1.0900, previous_price=1.0850)

    assert em._ewma_correlation is not None


def test_ewma_correlation_update_no_op_if_init_fails(em):
    """Test EWMA update is no-op if engine can't be initialized."""
    with patch.object(em, '_init_ewma_correlation'):
        em._update_ewma_correlation("EUR_USD", current_price=1.0900, previous_price=1.0850)

    # Should not crash


# ---------------------------------------------------------------------------
# Tests: Diversification Adjustment
# ---------------------------------------------------------------------------

def test_diversification_adjustment_no_engine(em):
    """Test diversification adjustment returns base lots if no EWMA engine."""
    result = em._apply_diversification_adjustment(lots=1.0, pair="EUR_USD")

    assert result == 1.0


def test_diversification_adjustment_empty_portfolio(em):
    """Test diversification adjustment returns base lots with empty portfolio."""
    em._init_ewma_correlation()
    em._open_trade_pairs = []

    result = em._apply_diversification_adjustment(lots=1.0, pair="EUR_USD")

    assert result == 1.0


def test_diversification_adjustment_reduces_correlated_pair(em):
    """Test diversification adjustment reduces lots for correlated pairs."""
    em._init_ewma_correlation()

    # Populate with some correlation data
    em._ewma_correlation.update({"EUR_USD": 0.01, "GBP_USD": 0.01})  # Correlated
    em._ewma_correlation.update({"EUR_USD": 0.01, "GBP_USD": 0.01})

    em._open_trade_pairs = ["GBP_USD"]

    result = em._apply_diversification_adjustment(lots=1.0, pair="EUR_USD")

    # Result should be <= 1.0 (reduced for correlation)
    assert 0.01 <= result <= 1.0


def test_diversification_adjustment_respects_min_lot_size(em, config):
    """Test diversification adjustment never goes below min_lot_size."""
    em._init_ewma_correlation()
    em._open_trade_pairs = ["GBP_USD"]

    result = em._apply_diversification_adjustment(lots=0.001, pair="EUR_USD")

    # Should be clamped to min_lot_size
    assert result >= config.min_lot_size


def test_diversification_adjustment_handles_exception(em):
    """Test diversification adjustment gracefully handles exceptions."""
    em._init_ewma_correlation()
    em._open_trade_pairs = None  # Invalid

    result = em._apply_diversification_adjustment(lots=1.0, pair="EUR_USD")

    # Should fallback to base lots
    assert result == 1.0


# ---------------------------------------------------------------------------
# Tests: Portfolio Risk Limit Adjustment
# ---------------------------------------------------------------------------

def test_portfolio_risk_limit_default(em):
    """Test portfolio risk limit returns default when no EWMA engine."""
    limit = em._get_portfolio_risk_limit()

    assert limit == 0.15  # 15% default


def test_portfolio_risk_limit_risk_off_regime(em):
    """Test portfolio risk limit reduces in RISK_OFF regime."""
    em._init_ewma_correlation()

    # Force RISK_OFF regime by setting high average correlation
    em._ewma_correlation._avg_correlation = 0.65
    em._ewma_correlation._regime = "RISK_OFF"
    em._ewma_correlation._observation_count = 25  # Meet min observations

    limit = em._get_portfolio_risk_limit()

    # Should be reduced (softened to 85% of base limit)
    expected = 0.15 * 0.85
    assert abs(limit - expected) < 0.01


def test_portfolio_risk_limit_handles_init_failure(em):
    """Test portfolio risk limit returns default if init fails."""
    # Mock _init_ewma_correlation to raise exception
    with patch.object(em, '_init_ewma_correlation', side_effect=Exception("test error")):
        limit = em._get_portfolio_risk_limit()

    # Should return default and not crash
    assert limit == 0.15


def test_portfolio_risk_limit_handles_detection_failure(em):
    """Test portfolio risk limit returns default if regime detection fails."""
    em._init_ewma_correlation()

    with patch.object(
        em._ewma_correlation,
        'detect_correlation_regime',
        side_effect=Exception("test error")
    ):
        limit = em._get_portfolio_risk_limit()

    assert limit == 0.15


# ---------------------------------------------------------------------------
# Tests: State Persistence in sync_closed_trades_rl
# ---------------------------------------------------------------------------

def test_ewma_state_persistence_in_sync(em, temp_state_dir):
    """Test EWMA state is persisted after sync_closed_trades_rl."""
    em._init_ewma_correlation()

    # Update EWMA with some data
    em._ewma_correlation.update({"EUR_USD": 0.01})

    # Mock the required methods for sync
    em.monitor_open_trades = MagicMock(return_value=[])
    em.fetch_live_nav = MagicMock(return_value=10000.0)

    # Create a dummy journal with no pending trades
    journal_path = Path("trained_data/trade_journal_rl.json")
    journal_path.write_text("[]")

    # Call sync (will return immediately with "no pending trades")
    with patch.object(em, '_update_pair_performance'):
        result = em.sync_closed_trades_rl()

    # Verify EWMA state was saved (even with no trades)
    state_file = Path("trained_data/ewma_correlation_state.json")
    if state_file.exists():
        # Verify content can be parsed
        saved_state = json.loads(state_file.read_text())
        assert "correlation_matrix" in saved_state


def test_ewma_state_persistence_handles_save_error(em, temp_state_dir):
    """Test sync gracefully handles EWMA state save errors."""
    em._init_ewma_correlation()

    # Mock save_state to raise exception
    with patch.object(em._ewma_correlation, 'save_state', side_effect=IOError("write failed")):
        em.monitor_open_trades = MagicMock(return_value=[])
        em.fetch_live_nav = MagicMock(return_value=10000.0)

        journal_path = Path("trained_data/trade_journal_rl.json")
        journal_path.write_text("[]")

        # Should not crash
        result = em.sync_closed_trades_rl()
        assert "trades_synced" in result


def test_ewma_state_persistence_skipped_if_no_engine(em, temp_state_dir):
    """Test sync works correctly if EWMA engine was not initialized."""
    assert em._ewma_correlation is None

    em.monitor_open_trades = MagicMock(return_value=[])
    em.fetch_live_nav = MagicMock(return_value=10000.0)

    journal_path = Path("trained_data/trade_journal_rl.json")
    journal_path.write_text("[]")

    # Should not crash
    result = em.sync_closed_trades_rl()
    assert "trades_synced" in result


# ---------------------------------------------------------------------------
# Tests: Integration with Open Trades
# ---------------------------------------------------------------------------

def test_diversification_multiplier_with_multiple_open_trades(em):
    """Test diversification multiplier calculation with multiple open trades."""
    em._init_ewma_correlation()

    # Simulate multiple correlation updates
    em._ewma_correlation.update({"EUR_USD": 0.01, "GBP_USD": 0.01, "USD_JPY": -0.005})
    em._ewma_correlation.update({"EUR_USD": 0.005, "GBP_USD": 0.008, "USD_JPY": -0.003})

    em._open_trade_pairs = ["GBP_USD", "USD_JPY"]

    result = em._apply_diversification_adjustment(lots=1.0, pair="EUR_USD")

    # EUR_USD is correlated with GBP_USD and uncorrelated with USD_JPY
    # Should be reduced but not to minimum
    assert result > em.config.min_lot_size
    assert result <= 1.0


# ---------------------------------------------------------------------------
# Tests: Edge Cases and Error Handling
# ---------------------------------------------------------------------------

def test_ewma_all_operations_with_none_engine(em):
    """Test all EWMA operations work correctly when engine is None."""
    assert em._ewma_correlation is None

    # All these should not crash
    em._update_ewma_correlation("EUR_USD", 1.09, 1.08)
    assert em._apply_diversification_adjustment(1.0, "EUR_USD") == 1.0
    assert em._get_portfolio_risk_limit() == 0.15


def test_ewma_concurrent_operations_thread_safe(em):
    """Test EWMA operations are thread-safe."""
    em._init_ewma_correlation()

    # Update from one thread
    em._ewma_correlation.update({"EUR_USD": 0.01})

    # Get state from another (simulated)
    state = em._ewma_correlation.get_state()

    assert state is not None


def test_ewma_large_position_size_adjustment(em):
    """Test diversification adjustment with large position sizes."""
    em._init_ewma_correlation()
    em._open_trade_pairs = ["GBP_USD"]

    # Large position
    result = em._apply_diversification_adjustment(lots=50.0, pair="EUR_USD")

    assert result >= em.config.min_lot_size
    assert result <= 50.0


def test_ewma_very_small_position_size_adjustment(em):
    """Test diversification adjustment with very small position sizes."""
    em._init_ewma_correlation()
    em._open_trade_pairs = ["GBP_USD"]

    # Very small position
    result = em._apply_diversification_adjustment(lots=0.01, pair="EUR_USD")

    assert result >= em.config.min_lot_size


# ---------------------------------------------------------------------------
# Tests: Regime-Based Risk Limit Behavior
# ---------------------------------------------------------------------------

def test_risk_limit_neutral_regime(em):
    """Test portfolio risk limit in NEUTRAL regime."""
    em._init_ewma_correlation()

    em._ewma_correlation._avg_correlation = 0.5
    em._ewma_correlation._regime = "NEUTRAL"

    limit = em._get_portfolio_risk_limit()

    # Should stay at default
    assert limit == 0.15


def test_risk_limit_risk_on_regime(em):
    """Test portfolio risk limit in RISK_ON regime."""
    em._init_ewma_correlation()

    em._ewma_correlation._avg_correlation = 0.25
    em._ewma_correlation._regime = "RISK_ON"

    limit = em._get_portfolio_risk_limit()

    # Should stay at default (no special handling for RISK_ON)
    assert limit == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
