"""
Tests for Phase 53 US-330: Signal Freshness Guard.

Tests cover:
  - RSI computation correctness
  - ADX computation correctness
  - Stale signal detection (delta > 15 points)
  - Confidence penalty on stale
  - Rejection when confidence drops below 0.45
  - Fresh signal passthrough
  - Staleness rate tracking
  - Graceful handling of insufficient data
  - Lazy imports and wiring
"""

import numpy as np
import pytest

from src.scanner.signal_freshness import (
    SignalFreshnessGuard,
    FreshnessConfig,
    FreshnessResult,
)


@pytest.fixture
def guard():
    return SignalFreshnessGuard()


@pytest.fixture
def prices_uptrend():
    """Generate 30 close prices in an uptrend."""
    np.random.seed(42)
    base = 1.1000
    return [base + i * 0.0010 + np.random.normal(0, 0.0002) for i in range(30)]


@pytest.fixture
def prices_downtrend():
    """Generate 30 close prices in a downtrend."""
    np.random.seed(42)
    base = 1.1500
    return [base - i * 0.0010 + np.random.normal(0, 0.0002) for i in range(30)]


class TestRSIComputation:

    def test_rsi_uptrend_high(self, prices_uptrend):
        """Uptrend should produce RSI > 50."""
        rsi = SignalFreshnessGuard.compute_rsi(prices_uptrend)
        assert rsi > 50.0
        assert rsi <= 100.0

    def test_rsi_downtrend_low(self, prices_downtrend):
        """Downtrend should produce RSI < 50."""
        rsi = SignalFreshnessGuard.compute_rsi(prices_downtrend)
        assert rsi < 50.0
        assert rsi >= 0.0

    def test_rsi_insufficient_data(self):
        """Fewer than period+1 prices returns -1."""
        rsi = SignalFreshnessGuard.compute_rsi([1.0, 1.1, 1.2])
        assert rsi == -1.0

    def test_rsi_all_gains(self):
        """All gains should produce RSI=100."""
        prices = [float(i) for i in range(20)]  # 0,1,2,...,19
        rsi = SignalFreshnessGuard.compute_rsi(prices)
        assert rsi == 100.0

    def test_rsi_range(self, prices_uptrend):
        """RSI should always be in [0, 100]."""
        rsi = SignalFreshnessGuard.compute_rsi(prices_uptrend)
        assert 0 <= rsi <= 100


class TestADXComputation:

    def test_adx_trending_market(self):
        """Strong trend should produce high ADX."""
        np.random.seed(42)
        n = 30
        closes = [1.1 + i * 0.005 for i in range(n)]
        highs = [c + 0.002 for c in closes]
        lows = [c - 0.001 for c in closes]
        adx = SignalFreshnessGuard.compute_adx(highs, lows, closes)
        assert adx > 0  # Should detect trend

    def test_adx_insufficient_data(self):
        """Fewer than period+1 prices returns -1."""
        adx = SignalFreshnessGuard.compute_adx([1.0, 1.1], [0.9, 1.0], [1.0, 1.05])
        assert adx == -1.0

    def test_adx_range(self):
        """ADX should be >= 0."""
        np.random.seed(42)
        n = 30
        closes = [1.1 + np.random.normal(0, 0.005) for _ in range(n)]
        highs = [c + abs(np.random.normal(0.002, 0.001)) for c in closes]
        lows = [c - abs(np.random.normal(0.002, 0.001)) for c in closes]
        adx = SignalFreshnessGuard.compute_adx(highs, lows, closes)
        assert adx >= 0


class TestFreshnessCheck:

    def test_fresh_signal_no_penalty(self, guard, prices_uptrend):
        """Non-stale signal passes without penalty."""
        rsi = SignalFreshnessGuard.compute_rsi(prices_uptrend)
        result = guard.check(
            pair="EUR_USD",
            scan_rsi=rsi,  # Same as computed → delta = 0
            scan_adx=0.0,
            current_prices=prices_uptrend,
            confidence=0.60,
        )
        assert result.stale is False
        assert result.rejected is False
        assert result.adjusted_confidence == 0.60
        assert result.confidence_penalty == 0.0

    def test_stale_rsi_penalty(self, guard, prices_uptrend):
        """Stale RSI reduces confidence by 0.10."""
        real_rsi = SignalFreshnessGuard.compute_rsi(prices_uptrend)
        # Set scan RSI 20 points off from actual
        result = guard.check(
            pair="EUR_USD",
            scan_rsi=real_rsi + 20.0,
            scan_adx=0.0,
            current_prices=prices_uptrend,
            confidence=0.60,
        )
        assert result.stale is True
        assert abs(result.confidence_penalty - 0.10) < 0.001
        assert abs(result.adjusted_confidence - 0.50) < 0.001

    def test_stale_rejection(self, guard, prices_uptrend):
        """Stale signal + low confidence → rejection."""
        real_rsi = SignalFreshnessGuard.compute_rsi(prices_uptrend)
        result = guard.check(
            pair="EUR_USD",
            scan_rsi=real_rsi + 20.0,
            scan_adx=0.0,
            current_prices=prices_uptrend,
            confidence=0.50,  # 0.50 - 0.10 = 0.40 < 0.45
        )
        assert result.stale is True
        assert result.rejected is True
        assert result.adjusted_confidence < 0.45

    def test_no_scan_values_passthrough(self, guard):
        """No scan RSI/ADX → passthrough (no check possible)."""
        result = guard.check(
            pair="EUR_USD",
            scan_rsi=0.0,
            scan_adx=0.0,
            confidence=0.60,
        )
        assert result.stale is False
        assert result.adjusted_confidence == 0.60

    def test_no_current_prices_passthrough(self, guard):
        """No current prices → passthrough (can't recompute)."""
        result = guard.check(
            pair="EUR_USD",
            scan_rsi=65.0,
            scan_adx=28.0,
            current_prices=None,
            confidence=0.60,
        )
        assert result.stale is False
        assert result.adjusted_confidence == 0.60

    def test_staleness_rate_tracking(self, guard, prices_uptrend):
        """Staleness rate tracked across checks."""
        real_rsi = SignalFreshnessGuard.compute_rsi(prices_uptrend)

        # 2 fresh checks
        for _ in range(2):
            guard.check(scan_rsi=real_rsi, current_prices=prices_uptrend, confidence=0.6)

        # 1 stale check
        guard.check(scan_rsi=real_rsi + 25, current_prices=prices_uptrend, confidence=0.6)

        rate = guard.get_staleness_rate()
        assert abs(rate - 1.0 / 3.0) < 0.01  # 1 out of 3

    def test_custom_config(self):
        """Custom config thresholds work."""
        config = FreshnessConfig(
            stale_threshold=5.0,  # Very sensitive
            confidence_penalty=0.20,
            rejection_floor=0.50,
        )
        guard = SignalFreshnessGuard(config)
        np.random.seed(42)
        prices = [1.1 + i * 0.001 for i in range(30)]
        real_rsi = SignalFreshnessGuard.compute_rsi(prices)

        result = guard.check(
            scan_rsi=real_rsi + 8.0,  # 8 > 5 threshold
            current_prices=prices,
            confidence=0.60,
        )
        assert result.stale is True
        assert abs(result.confidence_penalty - 0.20) < 0.001


class TestFreshnessLazyImports:

    def test_signal_freshness_guard_import(self):
        from src.scanner import SignalFreshnessGuard
        assert SignalFreshnessGuard is not None

    def test_freshness_config_import(self):
        from src.scanner import FreshnessConfig
        assert FreshnessConfig is not None

    def test_freshness_result_import(self):
        from src.scanner import FreshnessResult
        assert FreshnessResult is not None


class TestFreshnessWiring:

    def test_execution_manager_has_signal_freshness(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.__init__)
            assert "_signal_freshness" in source
            assert "SignalFreshnessGuard" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")

    def test_freshness_check_in_execute_trade(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager)
            assert "_signal_freshness" in source
            assert "staleness_guard" in source  # gate attribution name
        except ImportError:
            pytest.skip("ExecutionManager not importable")
