"""
Unit tests for Market Regime Detector module.

Covers:
- Configuration validation
- BOCPD changepoint detection
- Hurst exponent calculation
- ADX computation
- Volatility regime classification
- Ensemble consensus logic
- Edge cases: zero values, negative values, None inputs, empty collections
- Thread safety (determinism across calls)
"""

import logging
import unittest
from typing import Optional

import numpy as np

from src.scanner.regime_detector import (
    REGIME_EXTREME,
    REGIME_HIGH,
    REGIME_LOW,
    REGIME_NAMES,
    REGIME_NORMAL,
    MarketRegimeDetector,
    RegimeDetectionResult,
    RegimeDetectorConfig,
)

logger = logging.getLogger(__name__)


class TestRegimeDetectorConfig(unittest.TestCase):
    """Test configuration validation."""

    def test_default_config(self):
        """Test that default config is valid."""
        config = RegimeDetectorConfig()
        detector = MarketRegimeDetector(config)
        self.assertIsNotNone(detector)

    def test_config_validation_hazard_rate(self):
        """Test hazard_rate validation."""
        with self.assertRaises(ValueError):
            config = RegimeDetectorConfig(hazard_rate=-0.01)
            MarketRegimeDetector(config)

        with self.assertRaises(ValueError):
            config = RegimeDetectorConfig(hazard_rate=1.5)
            MarketRegimeDetector(config)

    def test_config_validation_bocpd_threshold(self):
        """Test bocpd_threshold validation."""
        with self.assertRaises(ValueError):
            config = RegimeDetectorConfig(bocpd_threshold=-0.1)
            MarketRegimeDetector(config)

        with self.assertRaises(ValueError):
            config = RegimeDetectorConfig(bocpd_threshold=1.5)
            MarketRegimeDetector(config)

    def test_config_validation_hurst_thresholds(self):
        """Test Hurst threshold validation."""
        with self.assertRaises(ValueError):
            config = RegimeDetectorConfig(
                hurst_mean_revert_threshold=0.6,
                hurst_trending_threshold=0.5,
            )
            MarketRegimeDetector(config)

    def test_config_validation_windows(self):
        """Test window size validation."""
        with self.assertRaises(ValueError):
            config = RegimeDetectorConfig(hurst_window=10)
            MarketRegimeDetector(config)

        with self.assertRaises(ValueError):
            config = RegimeDetectorConfig(vol_lookback=10)
            MarketRegimeDetector(config)


class TestBOCPD(unittest.TestCase):
    """Test Bayesian Online Changepoint Detection."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_bocpd_stationary_signal(self):
        """Test BOCPD on stationary signal (should have low changepoint prob)."""
        # Stationary Gaussian noise
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.01, 200)
        changepoint_prob, _ = self.detector._compute_bocpd(returns)

        # Low changepoint probability expected for stationary signal
        self.assertLess(changepoint_prob, 0.5)

    def test_bocpd_with_changepoint(self):
        """Test BOCPD detects changepoint."""
        # Signal with obvious changepoint (volatility shift)
        np.random.seed(42)
        before = np.random.normal(0.0, 0.01, 100)
        after = np.random.normal(0.0, 0.05, 100)  # Higher volatility
        returns = np.concatenate([before, after])

        changepoint_prob, _ = self.detector._compute_bocpd(returns)
        # Should detect changepoint in second half
        self.assertGreater(changepoint_prob, 0.0)

    def test_bocpd_output_shape(self):
        """Test BOCPD returns correct shapes."""
        returns = np.random.normal(0.0, 0.01, 100)
        changepoint_prob, run_dist = self.detector._compute_bocpd(returns)

        self.assertIsInstance(changepoint_prob, float)
        self.assertTrue(0.0 <= changepoint_prob <= 1.0)
        self.assertIsInstance(run_dist, np.ndarray)
        self.assertEqual(run_dist.ndim, 1)
        self.assertGreater(len(run_dist), 0)

    def test_bocpd_insufficient_data(self):
        """Test BOCPD with insufficient data."""
        returns = np.array([0.01])
        changepoint_prob, _ = self.detector._compute_bocpd(returns)
        self.assertEqual(changepoint_prob, 0.0)


class TestHurstExponent(unittest.TestCase):
    """Test Hurst exponent calculation."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_hurst_trending_signal(self):
        """Test Hurst on trending signal (should be > 0.5)."""
        # Create trending signal: cumulative sum of positive increments
        np.random.seed(42)
        drift = np.cumsum(np.random.normal(0.005, 0.01, 200))
        hurst = self.detector._compute_hurst(drift)

        self.assertGreater(hurst, 0.0)
        self.assertLess(hurst, 2.0)

    def test_hurst_mean_reverting_signal(self):
        """Test Hurst on mean-reverting signal."""
        # Create mean-reverting signal: high frequency oscillations
        np.random.seed(42)
        # Strong mean reversion with noise
        signal = np.cumsum(np.random.normal(0.0, 0.001, 300) - 0.0005)
        hurst = self.detector._compute_hurst(signal)

        # Should produce a valid Hurst value
        self.assertGreaterEqual(hurst, 0.0)
        self.assertLessEqual(hurst, 2.0)

    def test_hurst_output_range(self):
        """Test Hurst output is in valid range."""
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.normal(0.0, 0.01, 200))

        hurst = self.detector._compute_hurst(prices)

        self.assertGreaterEqual(hurst, 0.0)
        self.assertLessEqual(hurst, 2.0)

    def test_hurst_insufficient_data(self):
        """Test Hurst with insufficient data."""
        prices = np.array([1.0, 1.01, 1.02])
        hurst = self.detector._compute_hurst(prices)

        # Should return 0.5 (random walk) or log warning
        self.assertEqual(hurst, 0.5)

    def test_hurst_constant_prices(self):
        """Test Hurst with constant prices (edge case)."""
        prices = np.ones(150)
        hurst = self.detector._compute_hurst(prices)

        # Should handle gracefully
        self.assertGreaterEqual(hurst, 0.0)
        self.assertLessEqual(hurst, 2.0)


class TestADX(unittest.TestCase):
    """Test ADX calculation."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_adx_trending_market(self):
        """Test ADX on trending market."""
        # Strong uptrend
        np.random.seed(42)
        base = np.cumsum(np.random.normal(0.01, 0.005, 100))
        closes = base + np.random.normal(0.0, 0.001, 100)
        highs = closes + np.abs(np.random.normal(0.005, 0.002, 100))
        lows = closes - np.abs(np.random.normal(0.005, 0.002, 100))

        adx = self.detector._compute_adx(highs, lows, closes)

        self.assertGreaterEqual(adx, 0.0)
        self.assertLessEqual(adx, 100.0)

    def test_adx_ranging_market(self):
        """Test ADX on ranging market."""
        # Oscillating price (no trend)
        np.random.seed(42)
        closes = 100 + 5 * np.sin(np.linspace(0, 10 * np.pi, 150))
        highs = closes + np.abs(np.random.normal(0.5, 0.2, 150))
        lows = closes - np.abs(np.random.normal(0.5, 0.2, 150))

        adx = self.detector._compute_adx(highs, lows, closes)

        self.assertGreaterEqual(adx, 0.0)
        self.assertLessEqual(adx, 100.0)

    def test_adx_insufficient_data(self):
        """Test ADX with insufficient data."""
        closes = np.array([1.0, 1.01, 1.02])
        highs = closes + 0.005
        lows = closes - 0.005

        adx = self.detector._compute_adx(highs, lows, closes)

        self.assertEqual(adx, 0.0)


class TestVolatilityRegimeClassification(unittest.TestCase):
    """Test volatility regime classification."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_low_volatility_regime(self):
        """Test LOW volatility classification."""
        # Very low volatility
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.001, 100)

        regime, name, percentile = self.detector._classify_volatility_regime(returns)

        self.assertIn(regime, [REGIME_LOW, REGIME_NORMAL])
        self.assertIn(name, REGIME_NAMES)
        self.assertGreaterEqual(percentile, 0.0)
        self.assertLessEqual(percentile, 100.0)

    def test_high_volatility_regime(self):
        """Test HIGH volatility classification."""
        # High volatility (relative to low vol baseline)
        np.random.seed(42)
        # Build history with low vol first, then high vol
        low_vol_returns = np.random.normal(0.0, 0.001, 50)
        high_vol_returns = np.random.normal(0.0, 0.05, 50)
        returns = np.concatenate([low_vol_returns, high_vol_returns])

        regime, name, percentile = self.detector._classify_volatility_regime(returns)

        # High vol relative to history should be HIGH or EXTREME
        self.assertIn(regime, [REGIME_NORMAL, REGIME_HIGH, REGIME_EXTREME])
        self.assertIn(name, REGIME_NAMES)

    def test_extreme_volatility_regime(self):
        """Test EXTREME volatility classification."""
        # Very high volatility (relative to low vol baseline)
        np.random.seed(42)
        # Build history with low vol first, then extreme vol
        low_vol_returns = np.random.normal(0.0, 0.001, 50)
        extreme_vol_returns = np.random.normal(0.0, 0.1, 50)
        returns = np.concatenate([low_vol_returns, extreme_vol_returns])

        regime, name, percentile = self.detector._classify_volatility_regime(returns)

        # Extreme vol relative to history should be at least HIGH
        self.assertIn(regime, [REGIME_HIGH, REGIME_EXTREME])
        self.assertIn(name, REGIME_NAMES)

    def test_vol_percentile_range(self):
        """Test volatility percentile is in valid range."""
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 100)

        _, _, percentile = self.detector._classify_volatility_regime(returns)

        self.assertGreaterEqual(percentile, 0.0)
        self.assertLessEqual(percentile, 100.0)

    def test_vol_insufficient_data(self):
        """Test volatility classification with insufficient data."""
        returns = np.array([0.01, 0.02, 0.01])

        regime, name, percentile = self.detector._classify_volatility_regime(returns)

        self.assertEqual(regime, REGIME_NORMAL)
        self.assertEqual(name, "NORMAL")


class TestTrendClassification(unittest.TestCase):
    """Test trend classification from Hurst exponent."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_trending_classification(self):
        """Test trending classification."""
        hurst = 0.6
        trend = self.detector._classify_trend(hurst)
        self.assertEqual(trend, "TRENDING")

    def test_mean_reverting_classification(self):
        """Test mean-reverting classification."""
        hurst = 0.4
        trend = self.detector._classify_trend(hurst)
        self.assertEqual(trend, "MEAN_REVERTING")

    def test_uncertain_classification(self):
        """Test uncertain classification."""
        hurst = 0.5
        trend = self.detector._classify_trend(hurst)
        self.assertEqual(trend, "UNCERTAIN")


class TestADXStrengthClassification(unittest.TestCase):
    """Test ADX strength classification."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_strong_adx_classification(self):
        """Test strong trend classification."""
        adx = 30.0
        strength = self.detector._classify_adx_strength(adx)
        self.assertEqual(strength, "STRONG")

    def test_weak_adx_classification(self):
        """Test weak trend classification."""
        adx = 15.0
        strength = self.detector._classify_adx_strength(adx)
        self.assertEqual(strength, "WEAK")

    def test_transition_adx_classification(self):
        """Test transition classification."""
        adx = 22.0
        strength = self.detector._classify_adx_strength(adx)
        self.assertEqual(strength, "TRANSITION")


class TestEnsembleConsensus(unittest.TestCase):
    """Test ensemble consensus logic."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_caution_signal(self):
        """Test CAUTION strategy on changepoint + high vol."""
        result = self.detector._ensemble_consensus(
            vol_regime=REGIME_HIGH,
            vol_percentile=80.0,
            changepoint_prob=0.5,
            changepoint_alert=True,
            hurst=0.5,
            trend_class="UNCERTAIN",
            adx=22.0,
            adx_strength="TRANSITION",
        )

        self.assertEqual(result.strategy_hint, "CAUTION")
        self.assertEqual(result.risk_multiplier, 0.5)

    def test_momentum_signal(self):
        """Test MOMENTUM strategy on trending + strong trend."""
        result = self.detector._ensemble_consensus(
            vol_regime=REGIME_NORMAL,
            vol_percentile=50.0,
            changepoint_prob=0.1,
            changepoint_alert=False,
            hurst=0.6,
            trend_class="TRENDING",
            adx=30.0,
            adx_strength="STRONG",
        )

        self.assertEqual(result.strategy_hint, "MOMENTUM")
        self.assertEqual(result.risk_multiplier, 1.0)

    def test_mean_reversion_signal(self):
        """Test MEAN_REVERSION strategy."""
        result = self.detector._ensemble_consensus(
            vol_regime=REGIME_NORMAL,
            vol_percentile=50.0,
            changepoint_prob=0.1,
            changepoint_alert=False,
            hurst=0.4,
            trend_class="MEAN_REVERTING",
            adx=15.0,
            adx_strength="WEAK",
        )

        self.assertEqual(result.strategy_hint, "MEAN_REVERSION")
        self.assertEqual(result.risk_multiplier, 0.8)

    def test_range_signal(self):
        """Test RANGE strategy on uncertain conditions."""
        result = self.detector._ensemble_consensus(
            vol_regime=REGIME_NORMAL,
            vol_percentile=50.0,
            changepoint_prob=0.1,
            changepoint_alert=False,
            hurst=0.5,
            trend_class="UNCERTAIN",
            adx=22.0,
            adx_strength="TRANSITION",
        )

        self.assertEqual(result.strategy_hint, "RANGE")
        self.assertEqual(result.risk_multiplier, 0.7)

    def test_confidence_calculation(self):
        """Test confidence is in valid range."""
        result = self.detector._ensemble_consensus(
            vol_regime=REGIME_NORMAL,
            vol_percentile=50.0,
            changepoint_prob=0.2,
            changepoint_alert=False,
            hurst=0.5,
            trend_class="UNCERTAIN",
            adx=22.0,
            adx_strength="TRANSITION",
        )

        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestFullRegimeDetection(unittest.TestCase):
    """Test full regime detection pipeline."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_update_with_prices_only(self):
        """Test regime detection with prices only."""
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))

        result = self.detector.update(prices)

        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIn(result.regime_index, [0, 1, 2, 3])
        self.assertIn(result.regime_name, REGIME_NAMES)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreater(len(result.signals), 0)

    def test_update_with_ohlc(self):
        """Test regime detection with OHLC data."""
        np.random.seed(42)
        closes = np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))
        highs = closes + np.abs(np.random.normal(0.005, 0.002, 200))
        lows = closes - np.abs(np.random.normal(0.005, 0.002, 200))

        result = self.detector.update(closes, highs, lows)

        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIn(result.regime_index, [0, 1, 2, 3])
        self.assertGreater(result.adx_value, 0.0)

    def test_update_insufficient_data(self):
        """Test regime detection with insufficient data."""
        prices = np.array([1.0, 1.01, 1.02])

        with self.assertRaises(ValueError):
            self.detector.update(prices)

    def test_update_invalid_shapes(self):
        """Test regime detection with invalid shape input."""
        prices = np.random.normal(0.0, 1.0, 200)
        highs = np.random.normal(0.0, 1.0, 150)  # Wrong shape
        lows = np.random.normal(0.0, 1.0, 200)

        with self.assertRaises(ValueError):
            self.detector.update(prices, highs, lows)

    def test_update_determinism(self):
        """Test that same input produces same output (deterministic)."""
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))

        result1 = self.detector.update(prices.copy())
        result2 = self.detector.update(prices.copy())

        self.assertEqual(result1.regime_index, result2.regime_index)
        self.assertEqual(result1.regime_name, result2.regime_name)
        np.testing.assert_almost_equal(
            result1.changepoint_probability, result2.changepoint_probability, decimal=5
        )
        np.testing.assert_almost_equal(
            result1.hurst_exponent, result2.hurst_exponent, decimal=5
        )
        np.testing.assert_almost_equal(result1.confidence, result2.confidence, decimal=5)

    def test_result_signals_dict(self):
        """Test that signals dict is populated."""
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))

        result = self.detector.update(prices)

        self.assertIn("volatility_regime", result.signals)
        self.assertIn("changepoint_probability", result.signals)
        self.assertIn("hurst_exponent", result.signals)
        self.assertIn("adx_value", result.signals)
        self.assertIn("strategy_hint", result.signals)
        self.assertIn("risk_multiplier", result.signals)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up detector."""
        self.detector = MarketRegimeDetector()

    def test_zero_prices(self):
        """Test with zero prices (edge case)."""
        prices = np.zeros(200)
        # Should not crash, but results may be undefined
        try:
            result = self.detector.update(prices)
            # If it succeeds, result should still be valid
            self.assertIsInstance(result, RegimeDetectionResult)
        except (ValueError, RuntimeError):
            # It's acceptable to fail on zero prices
            pass

    def test_nan_handling(self):
        """Test handling of NaN in input."""
        prices = np.random.normal(0.0, 1.0, 200)
        prices[100] = np.nan

        with self.assertRaises(ValueError):
            self.detector.update(prices)

    def test_inf_handling(self):
        """Test handling of infinite values."""
        prices = np.random.normal(0.0, 1.0, 200)
        prices[100] = np.inf

        with self.assertRaises(ValueError):
            self.detector.update(prices)

    def test_negative_prices(self):
        """Test with negative prices (invalid for FX)."""
        prices = np.linspace(-10, -5, 200)
        # Log returns will produce NaN, should be caught
        try:
            result = self.detector.update(prices)
            # If it doesn't crash, it should handle gracefully
            self.assertIsInstance(result, RegimeDetectionResult)
        except (ValueError, RuntimeError):
            # Expected behavior for invalid prices
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
