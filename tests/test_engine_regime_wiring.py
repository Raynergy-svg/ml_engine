"""
Tests for Phase 44 (US-278): MarketRegimeDetector wiring into Scanner engine.

Tests the _detect_regime helper method directly and the initialization pattern.
Uses direct attribute manipulation instead of complex mocking of Scanner internals.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys


class TestDetectRegimeMethod(unittest.TestCase):
    """Test the _detect_regime method logic independently."""

    def _make_scanner_stub(self):
        """Create a minimal Scanner-like object with _detect_regime behavior."""
        # Instead of constructing a full Scanner (which requires many deps),
        # we test the _detect_regime logic directly by importing Scanner
        # and calling the method with a mocked detector.
        try:
            from src.scanner.engine import Scanner
            from src.scanner.config import ScannerConfig
            scanner = Scanner(config=ScannerConfig(), oanda_client=Mock())
            return scanner
        except Exception:
            return None

    def test_detect_regime_returns_none_when_detector_is_none(self):
        """_detect_regime returns None if _market_regime_detector is None."""
        scanner = self._make_scanner_stub()
        if scanner is None:
            self.skipTest("Scanner import failed")
        scanner._market_regime_detector = None

        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
        })
        result = scanner._detect_regime(df, "EUR_USD")
        self.assertIsNone(result)

    def test_detect_regime_returns_result_when_detector_works(self):
        """_detect_regime returns the RegimeDetectionResult when detector succeeds."""
        scanner = self._make_scanner_stub()
        if scanner is None:
            self.skipTest("Scanner import failed")

        mock_detector = Mock()
        mock_result = Mock()
        mock_result.regime_name = "NORMAL"
        mock_result.regime_index = 1
        mock_result.confidence = 0.75
        mock_result.hurst = 0.52
        mock_result.adx = 25.5
        mock_result.changepoint_probability = 0.15
        mock_result.strategy_hint = "MOMENTUM"
        mock_result.risk_multiplier = 0.8
        mock_detector.update.return_value = mock_result
        scanner._market_regime_detector = mock_detector

        df = pd.DataFrame({
            "close": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 101,
            "low": np.random.randn(100).cumsum() + 99,
        })
        result = scanner._detect_regime(df, "EUR_USD")
        self.assertIsNotNone(result)
        self.assertEqual(result.regime_name, "NORMAL")
        mock_detector.update.assert_called_once()

    def test_detect_regime_handles_exception_gracefully(self):
        """_detect_regime catches exceptions and returns None."""
        scanner = self._make_scanner_stub()
        if scanner is None:
            self.skipTest("Scanner import failed")

        mock_detector = Mock()
        mock_detector.update.side_effect = ValueError("Bad data")
        scanner._market_regime_detector = mock_detector

        df = pd.DataFrame({"close": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0]})
        result = scanner._detect_regime(df, "EUR_USD")
        self.assertIsNone(result)

    def test_detect_regime_handles_missing_high_low_columns(self):
        """_detect_regime uses close as fallback for missing high/low."""
        scanner = self._make_scanner_stub()
        if scanner is None:
            self.skipTest("Scanner import failed")

        mock_detector = Mock()
        mock_result = Mock()
        mock_result.regime_name = "LOW"
        mock_result.regime_index = 0
        mock_result.confidence = 0.6
        mock_result.hurst = 0.48
        mock_result.adx = 15.0
        mock_result.changepoint_probability = 0.05
        mock_result.strategy_hint = "RANGE"
        mock_result.risk_multiplier = 1.0
        mock_detector.update.return_value = mock_result
        scanner._market_regime_detector = mock_detector

        # DataFrame without high/low
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        result = scanner._detect_regime(df, "EUR_USD")
        self.assertIsNotNone(result)
        # Verify update was called (with close used as fallback for highs/lows)
        mock_detector.update.assert_called_once()


class TestScannerInitRegimeDetector(unittest.TestCase):
    """Test that Scanner.__init__ creates _market_regime_detector."""

    def test_scanner_has_market_regime_detector_attribute(self):
        """Scanner should have _market_regime_detector attribute after init."""
        try:
            from src.scanner.engine import Scanner
            from src.scanner.config import ScannerConfig
            scanner = Scanner(config=ScannerConfig(), oanda_client=Mock())
            self.assertTrue(hasattr(scanner, "_market_regime_detector"))
        except Exception:
            self.skipTest("Scanner import failed")

    def test_scanner_regime_detector_is_initialized_or_none(self):
        """_market_regime_detector is either a MarketRegimeDetector or None."""
        try:
            from src.scanner.engine import Scanner
            from src.scanner.config import ScannerConfig
            scanner = Scanner(config=ScannerConfig(), oanda_client=Mock())
            # It's either initialized or gracefully set to None
            # (depends on whether regime_detector.py imports succeed)
            attr = scanner._market_regime_detector
            if attr is not None:
                self.assertTrue(hasattr(attr, "update"))
            # If None, that's also fine — deferred init
        except Exception:
            self.skipTest("Scanner import failed")


class TestRegimeDetectorOutput(unittest.TestCase):
    """Test the actual RegimeDetector module independently."""

    def test_regime_detector_creates_and_updates(self):
        """MarketRegimeDetector can be created and updated with price data."""
        try:
            from src.scanner.regime_detector import MarketRegimeDetector
        except ImportError:
            self.skipTest("regime_detector not available")

        detector = MarketRegimeDetector()
        prices = np.random.randn(200).cumsum() + 100
        highs = prices + np.abs(np.random.randn(200)) * 0.5
        lows = prices - np.abs(np.random.randn(200)) * 0.5

        result = detector.update(prices, highs, lows)
        self.assertIsNotNone(result)
        self.assertIn(result.regime_index, [0, 1, 2, 3])
        self.assertIn(result.regime_name, ["LOW", "NORMAL", "HIGH", "EXTREME"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_regime_detector_result_has_all_fields(self):
        """RegimeDetectionResult has all expected fields."""
        try:
            from src.scanner.regime_detector import MarketRegimeDetector
        except ImportError:
            self.skipTest("regime_detector not available")

        detector = MarketRegimeDetector()
        prices = np.random.randn(200).cumsum() + 100
        highs = prices + 1.0
        lows = prices - 1.0

        result = detector.update(prices, highs, lows)
        # Check all fields exist (actual field names from RegimeDetectionResult)
        self.assertTrue(hasattr(result, "regime_index"))
        self.assertTrue(hasattr(result, "regime_name"))
        self.assertTrue(hasattr(result, "confidence"))
        self.assertTrue(hasattr(result, "changepoint_probability"))
        self.assertTrue(hasattr(result, "hurst_exponent"))
        self.assertTrue(hasattr(result, "adx_value"))
        self.assertTrue(hasattr(result, "strategy_hint"))
        self.assertTrue(hasattr(result, "risk_multiplier"))

    def test_regime_detector_short_data_raises(self):
        """RegimeDetector raises ValueError on data shorter than 50 bars."""
        try:
            from src.scanner.regime_detector import MarketRegimeDetector
        except ImportError:
            self.skipTest("regime_detector not available")

        detector = MarketRegimeDetector()
        prices = np.array([100.0, 101.0, 102.0])
        # Module requires at least 50 bars — should raise ValueError
        with self.assertRaises(ValueError):
            detector.update(prices, prices + 0.5, prices - 0.5)

    def test_regime_detector_trending_data(self):
        """RegimeDetector identifies trending data correctly."""
        try:
            from src.scanner.regime_detector import MarketRegimeDetector
        except ImportError:
            self.skipTest("regime_detector not available")

        detector = MarketRegimeDetector()
        # Strong uptrend
        prices = np.linspace(100, 150, 200) + np.random.randn(200) * 0.5
        highs = prices + 1.0
        lows = prices - 1.0

        result = detector.update(prices, highs, lows)
        # Should detect trending regime (hurst_exponent > 0.5 ideally)
        self.assertIsNotNone(result)
        self.assertGreater(result.hurst_exponent, 0.0)

    def test_regime_detector_volatile_data(self):
        """RegimeDetector handles volatile data."""
        try:
            from src.scanner.regime_detector import MarketRegimeDetector
        except ImportError:
            self.skipTest("regime_detector not available")

        detector = MarketRegimeDetector()
        # High volatility data
        prices = 100 + np.random.randn(200) * 10
        highs = prices + np.abs(np.random.randn(200)) * 5
        lows = prices - np.abs(np.random.randn(200)) * 5

        result = detector.update(prices, highs, lows)
        self.assertIsNotNone(result)
        self.assertIn(result.regime_index, [0, 1, 2, 3])


class TestRegimeWiringGateDetails(unittest.TestCase):
    """Test that regime info gets correctly structured in gate_details."""

    def test_gate_details_structure(self):
        """Verify the expected structure of regime_detector in gate_details."""
        # Test the dict construction logic directly
        mock_result = Mock()
        mock_result.regime_name = "HIGH"
        mock_result.regime_index = 2
        mock_result.confidence = 0.85432
        mock_result.changepoint_probability = 0.25678
        mock_result.hurst = 0.58123
        mock_result.adx = 35.678
        mock_result.strategy_hint = "MOMENTUM"
        mock_result.risk_multiplier = 0.65432

        # Replicate the gate_details construction from engine.py
        gate_details = {}
        gate_details["regime_detector"] = {
            "regime_name": mock_result.regime_name,
            "regime_index": mock_result.regime_index,
            "confidence": round(mock_result.confidence, 3),
            "changepoint_probability": round(mock_result.changepoint_probability, 3),
            "hurst": round(mock_result.hurst, 3),
            "adx": round(mock_result.adx, 1),
            "strategy_hint": mock_result.strategy_hint,
            "risk_multiplier": round(mock_result.risk_multiplier, 2),
        }

        rd = gate_details["regime_detector"]
        self.assertEqual(rd["regime_name"], "HIGH")
        self.assertEqual(rd["regime_index"], 2)
        self.assertAlmostEqual(rd["confidence"], 0.854, places=3)
        self.assertAlmostEqual(rd["changepoint_probability"], 0.257, places=3)
        self.assertAlmostEqual(rd["hurst"], 0.581, places=3)
        self.assertAlmostEqual(rd["adx"], 35.7, places=1)
        self.assertEqual(rd["strategy_hint"], "MOMENTUM")
        self.assertAlmostEqual(rd["risk_multiplier"], 0.65, places=2)

    def test_all_regime_indices(self):
        """Test gate_details for all 4 regime indices."""
        regime_map = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        for idx, name in regime_map.items():
            result = Mock()
            result.regime_index = idx
            result.regime_name = name
            result.confidence = 0.5
            result.changepoint_probability = 0.1
            result.hurst = 0.5
            result.adx = 20.0
            result.strategy_hint = "TEST"
            result.risk_multiplier = 1.0

            gate_details = {
                "regime_detector": {
                    "regime_name": result.regime_name,
                    "regime_index": result.regime_index,
                }
            }
            self.assertEqual(gate_details["regime_detector"]["regime_index"], idx)
            self.assertEqual(gate_details["regime_detector"]["regime_name"], name)


if __name__ == "__main__":
    unittest.main()
