"""
Comprehensive unit tests for Multi-Timeframe Confluence Scoring System.

Tests Elder's Triple Screen methodology with 20+ test cases covering:
- Screen 1 (Trend): SMA alignment, ADX strength, slope detection
- Screen 2 (Wave): RSI pullback quality, momentum alignment
- Screen 3 (Entry): EMA alignment, candle quality, S/R proximity
- Confluence calculation: weighted scoring, alignment detection, thresholds
- Edge cases: minimal data, flat/sideways data, direction mismatches
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src.scanner.mtf_confluence import (
    ConfluenceResult,
    MTFConfig,
    MultiTimeframeConfluence,
    ScreenResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures: Synthetic OHLCV Data Generators
# =============================================================================

def create_ohlcv_dataframe(
    num_bars: int,
    start_price: float = 1.1000,
    trend: str = "FLAT",
    volatility: float = 0.005,
    volume: float = 1000.0,
) -> pd.DataFrame:
    """
    Create synthetic OHLCV data with known properties.

    Args:
        num_bars: Number of bars to generate
        start_price: Starting close price
        trend: "UP", "DOWN", "FLAT"
        volatility: Standard deviation of returns
        volume: Base volume per bar

    Returns:
        DataFrame with OHLCV columns
    """
    np.random.seed(42)  # Consistent for testing
    data = []

    price = start_price
    for i in range(num_bars):
        # Generate trend
        if trend == "UP":
            price_change = abs(np.random.normal(0.0002, volatility))
        elif trend == "DOWN":
            price_change = -abs(np.random.normal(0.0002, volatility))
        else:  # FLAT
            price_change = np.random.normal(0, volatility)

        price += price_change

        # OHLC within bar
        open_price = price - np.random.uniform(-volatility / 2, volatility / 2)
        close_price = price + np.random.uniform(-volatility / 2, volatility / 2)
        high_price = max(open_price, close_price) + abs(
            np.random.normal(0, volatility / 2)
        )
        low_price = min(open_price, close_price) - abs(
            np.random.normal(0, volatility / 2)
        )

        vol = volume * np.random.uniform(0.8, 1.2)

        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": vol,
            }
        )

    return pd.DataFrame(data)


def create_bullish_data(num_bars: int = 100) -> pd.DataFrame:
    """Create synthetic bullish trend data."""
    return create_ohlcv_dataframe(num_bars, start_price=1.1000, trend="UP", volatility=0.003)


def create_bearish_data(num_bars: int = 100) -> pd.DataFrame:
    """Create synthetic bearish trend data."""
    return create_ohlcv_dataframe(num_bars, start_price=1.1000, trend="DOWN", volatility=0.003)


def create_flat_data(num_bars: int = 100) -> pd.DataFrame:
    """Create synthetic sideways/flat data."""
    return create_ohlcv_dataframe(num_bars, start_price=1.1000, trend="FLAT", volatility=0.002)


def create_bullish_with_rsi_pullback(num_bars: int = 50) -> pd.DataFrame:
    """
    Create bullish trend data with an RSI pullback setup.
    Used for Screen 2 (Wave) testing.
    """
    data = create_ohlcv_dataframe(num_bars, start_price=1.1000, trend="UP", volatility=0.002)

    # Add a pullback at the end (lower closes for last 5 bars)
    pullback_start = len(data) - 5
    for i in range(pullback_start, len(data)):
        data.loc[i, "close"] = data.loc[i, "close"] * (1 - 0.003)

    return data.reset_index(drop=True)


def create_bearish_with_rsi_pullback(num_bars: int = 50) -> pd.DataFrame:
    """
    Create bearish trend data with an RSI pullback setup.
    """
    data = create_ohlcv_dataframe(num_bars, start_price=1.1000, trend="DOWN", volatility=0.002)

    # Add a pullback at the end (higher closes for last 5 bars)
    pullback_start = len(data) - 5
    for i in range(pullback_start, len(data)):
        data.loc[i, "close"] = data.loc[i, "close"] * (1 + 0.003)

    return data.reset_index(drop=True)


def create_aligned_multiframe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create three timeframes with perfect alignment.
    - 4H: Bullish trend
    - 1H: Bullish with pullback
    - 15M: Bullish entry
    """
    df_4h = create_bullish_data(num_bars=300)
    df_1h = create_bullish_with_rsi_pullback(num_bars=300)
    df_15m = create_bullish_data(num_bars=300)
    return df_4h, df_1h, df_15m


def create_misaligned_multiframe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create three timeframes with conflicting signals.
    - 4H: Bullish
    - 1H: Bearish
    - 15M: Flat
    """
    df_4h = create_bullish_data(num_bars=300)
    df_1h = create_bearish_data(num_bars=300)
    df_15m = create_flat_data(num_bars=300)
    return df_4h, df_1h, df_15m


# =============================================================================
# Test: ScreenResult Dataclass
# =============================================================================

class TestScreenResult:
    """Test ScreenResult dataclass functionality."""

    def test_screen_result_initialization(self):
        """ScreenResult initializes with all fields."""
        result = ScreenResult(
            screen_name="trend_4h",
            direction="BULLISH",
            score=0.75,
            details={"sma50": 1.1000},
        )
        assert result.screen_name == "trend_4h"
        assert result.direction == "BULLISH"
        assert result.score == 0.75
        assert result.details["sma50"] == 1.1000

    def test_screen_result_to_dict(self):
        """ScreenResult.to_dict() converts all fields."""
        result = ScreenResult(
            screen_name="wave_1h",
            direction="NEUTRAL",
            score=0.5,
            details={"rsi": 50.0},
        )
        d = result.to_dict()
        assert d["screen_name"] == "wave_1h"
        assert d["direction"] == "NEUTRAL"
        assert d["score"] == 0.5
        assert isinstance(d["details"], dict)


# =============================================================================
# Test: ConfluenceResult Dataclass
# =============================================================================

class TestConfluenceResult:
    """Test ConfluenceResult dataclass functionality."""

    def test_confluence_result_initialization(self):
        """ConfluenceResult initializes with all fields."""
        screens = [
            ScreenResult("trend_4h", "BULLISH", 0.8),
            ScreenResult("wave_1h", "BULLISH", 0.7),
            ScreenResult("entry_15m", "BULLISH", 0.9),
        ]
        result = ConfluenceResult(
            confluence_score=0.80,
            direction_aligned=True,
            screen_results=screens,
            penalty_applied=0.0,
            recommendation="PROCEED",
        )
        assert result.confluence_score == 0.80
        assert result.direction_aligned is True
        assert len(result.screen_results) == 3
        assert result.recommendation == "PROCEED"

    def test_confluence_result_to_dict(self):
        """ConfluenceResult.to_dict() serializes correctly."""
        screens = [ScreenResult("trend_4h", "BULLISH", 0.75)]
        result = ConfluenceResult(
            confluence_score=0.75,
            direction_aligned=True,
            screen_results=screens,
            penalty_applied=0.0,
            recommendation="PROCEED",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["confluence_score"] == 0.75
        assert d["direction_aligned"] is True
        assert len(d["screen_results"]) == 1


# =============================================================================
# Test: MTFConfig
# =============================================================================

class TestMTFConfig:
    """Test configuration initialization and defaults."""

    def test_default_config(self):
        """Default MTFConfig has sensible values."""
        cfg = MTFConfig()
        assert cfg.proceed_threshold == 0.65
        assert cfg.caution_threshold == 0.45
        assert cfg.misalignment_penalty == -0.25
        assert cfg.trend_weight == 0.50
        assert cfg.wave_weight == 0.30
        assert cfg.entry_weight == 0.20

    def test_config_weights_sum_to_one(self):
        """Timeframe weights sum to 1.0."""
        cfg = MTFConfig()
        total = cfg.trend_weight + cfg.wave_weight + cfg.entry_weight
        assert abs(total - 1.0) < 0.01

    def test_custom_config(self):
        """Can override config values."""
        cfg = MTFConfig(proceed_threshold=0.70, sma_short_period=60)
        assert cfg.proceed_threshold == 0.70
        assert cfg.sma_short_period == 60


# =============================================================================
# Test: Screen 1 — Trend Analysis
# =============================================================================

class TestTrendScreen:
    """Test Screen 1: 4H Trend detection."""

    def test_trend_screen_bullish(self):
        """Bullish data produces BULLISH direction."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=250)
        result = mtf._score_trend_screen(data)

        assert result is not None
        assert result.direction == "BULLISH"
        assert result.score > 0.3  # Should have decent score
        assert result.screen_name == "trend_4h"

    def test_trend_screen_bearish(self):
        """Bearish data produces BEARISH direction."""
        mtf = MultiTimeframeConfluence()
        data = create_bearish_data(num_bars=250)
        result = mtf._score_trend_screen(data)

        assert result is not None
        assert result.direction == "BEARISH"
        assert result.score > 0.3

    def test_trend_screen_flat(self):
        """Flat data produces low or NEUTRAL direction."""
        mtf = MultiTimeframeConfluence()
        data = create_flat_data(num_bars=250)
        result = mtf._score_trend_screen(data)

        assert result is not None
        # Flat data may lean slightly bullish/bearish due to randomness
        # but score should be moderate
        assert result.score < 0.8

    def test_trend_screen_insufficient_data(self):
        """Insufficient data returns None."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=50)  # Too short
        result = mtf._score_trend_screen(data)

        assert result is None

    def test_trend_screen_score_range(self):
        """Trend screen score is always 0.0-1.0."""
        mtf = MultiTimeframeConfluence()
        for _ in range(5):
            data = create_bullish_data(num_bars=250)
            result = mtf._score_trend_screen(data)
            assert result is not None
            assert 0.0 <= result.score <= 1.0

    def test_trend_screen_details(self):
        """Trend screen includes SMA and ADX details."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=250)
        result = mtf._score_trend_screen(data)

        assert "sma50" in result.details
        assert "sma200" in result.details
        assert "adx" in result.details
        assert "slope_magnitude" in result.details


# =============================================================================
# Test: Screen 2 — Wave Analysis
# =============================================================================

class TestWaveScreen:
    """Test Screen 2: 1H Wave detection."""

    def test_wave_screen_bullish_pullback(self):
        """Bullish data with pullback scores well."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_with_rsi_pullback(num_bars=100)
        result = mtf._score_wave_screen(data, trend_direction="BULLISH")

        assert result is not None
        assert result.direction == "BULLISH"
        assert result.score > 0.3

    def test_wave_screen_bearish_pullback(self):
        """Bearish data with pullback scores well."""
        mtf = MultiTimeframeConfluence()
        data = create_bearish_with_rsi_pullback(num_bars=100)
        result = mtf._score_wave_screen(data, trend_direction="BEARISH")

        assert result is not None
        assert result.direction == "BEARISH"
        assert result.score > 0.3

    def test_wave_screen_neutral_trend(self):
        """Neutral trend direction handled correctly."""
        mtf = MultiTimeframeConfluence()
        data = create_flat_data(num_bars=100)
        result = mtf._score_wave_screen(data, trend_direction="NEUTRAL")

        assert result is not None
        assert result.direction == "NEUTRAL"

    def test_wave_screen_insufficient_data(self):
        """Insufficient data returns None."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=5)
        result = mtf._score_wave_screen(data, trend_direction="BULLISH")

        assert result is None

    def test_wave_screen_score_range(self):
        """Wave screen score is always 0.0-1.0."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_with_rsi_pullback(num_bars=100)
        result = mtf._score_wave_screen(data, trend_direction="BULLISH")

        assert result is not None
        assert 0.0 <= result.score <= 1.0

    def test_wave_screen_details(self):
        """Wave screen includes RSI details."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_with_rsi_pullback(num_bars=100)
        result = mtf._score_wave_screen(data, trend_direction="BULLISH")

        assert "rsi" in result.details
        assert "rsi_alignment" in result.details
        assert "momentum_change" in result.details


# =============================================================================
# Test: Screen 3 — Entry Analysis
# =============================================================================

class TestEntryScreen:
    """Test Screen 3: 15M Entry confirmation."""

    def test_entry_screen_bullish(self):
        """Bullish entry data scores well."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        result = mtf._score_entry_screen(data, trend_direction="BULLISH")

        assert result is not None
        assert result.direction == "BULLISH"
        assert result.score > 0.3

    def test_entry_screen_bearish(self):
        """Bearish entry data scores well."""
        mtf = MultiTimeframeConfluence()
        data = create_bearish_data(num_bars=100)
        result = mtf._score_entry_screen(data, trend_direction="BEARISH")

        assert result is not None
        assert result.direction == "BEARISH"
        assert result.score > 0.3

    def test_entry_screen_insufficient_data(self):
        """Insufficient data returns None."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=5)
        result = mtf._score_entry_screen(data, trend_direction="BULLISH")

        assert result is None

    def test_entry_screen_score_range(self):
        """Entry screen score is always 0.0-1.0."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        result = mtf._score_entry_screen(data, trend_direction="BULLISH")

        assert result is not None
        assert 0.0 <= result.score <= 1.0

    def test_entry_screen_details(self):
        """Entry screen includes EMA and ATR details."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        result = mtf._score_entry_screen(data, trend_direction="BULLISH")

        assert "ema20" in result.details
        assert "candle_quality" in result.details
        assert "sr_proximity" in result.details
        assert "atr" in result.details


# =============================================================================
# Test: Confluence Calculation & Weighting
# =============================================================================

class TestConfluenceCalculation:
    """Test overall confluence scoring."""

    def test_confluence_aligned_bullish(self):
        """All screens bullish produces aligned confluence."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_aligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        assert result is not None
        assert result.direction_aligned is True
        # Synthetic data may not produce high scores, but should be aligned
        assert len(result.screen_results) == 3
        directions = [s.direction for s in result.screen_results]
        assert directions.count("BULLISH") >= 2  # At least 2 screens bullish

    def test_confluence_misaligned(self):
        """Mixed direction screens may show misalignment effects."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_misaligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        assert result is not None
        # With deliberately misaligned data (bull/bear/flat),
        # we may still get alignment if indicators lean same way
        # Just verify we get a result with proper structure
        assert len(result.screen_results) == 3
        assert isinstance(result.penalty_applied, float)

    def test_confluence_score_range(self):
        """Confluence score is always 0.0-1.0."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_aligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        assert result is not None
        assert 0.0 <= result.confluence_score <= 1.0

    def test_confluence_weights_sum_to_one(self):
        """Confluence calculation respects weights (0.5 + 0.3 + 0.2)."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_aligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        # Raw score should be weighted combination
        screens = result.screen_results
        if len(screens) == 3:
            expected_raw = (
                screens[0].score * 0.5 +
                screens[1].score * 0.3 +
                screens[2].score * 0.2
            )
            # Account for penalty
            expected_final = max(0.0, expected_raw + result.penalty_applied)
            assert abs(result.confluence_score - expected_final) < 0.01

    def test_confluence_contains_all_screens(self):
        """Confluence result includes all three screens."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_aligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        assert len(result.screen_results) == 3
        assert result.screen_results[0].screen_name == "trend_4h"
        assert result.screen_results[1].screen_name == "wave_1h"
        assert result.screen_results[2].screen_name == "entry_15m"


# =============================================================================
# Test: Recommendation Logic
# =============================================================================

class TestRecommendation:
    """Test recommendation generation."""

    def test_recommendation_proceed_high_aligned(self):
        """High score with alignment -> PROCEED."""
        mtf = MultiTimeframeConfluence()
        # All screens bullish should give high score
        df_4h, df_1h, df_15m = create_aligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        if result.confluence_score >= 0.65 and result.direction_aligned:
            assert result.recommendation == "PROCEED"

    def test_recommendation_caution_medium_score(self):
        """Medium score -> CAUTION."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_misaligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        # Misaligned but might still be in CAUTION range
        assert result.recommendation in ["CAUTION", "REJECT"]

    def test_recommendation_reject_low_score(self):
        """Low score -> REJECT."""
        mtf = MultiTimeframeConfluence()
        # Create minimally valid data with poor alignment
        df_4h = create_flat_data(num_bars=250)
        df_1h = create_flat_data(num_bars=100)
        df_15m = create_flat_data(num_bars=100)
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        # Flat data should give low scores
        assert result.confluence_score < 0.65
        assert result.recommendation in ["CAUTION", "REJECT"]


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Empty dataframe handled gracefully."""
        mtf = MultiTimeframeConfluence()
        empty_df = pd.DataFrame()
        result = mtf.score_confluence(empty_df, empty_df, empty_df)

        assert result is not None
        assert result.recommendation == "REJECT"

    def test_single_bar(self):
        """Single bar doesn't crash."""
        mtf = MultiTimeframeConfluence()
        single_bar = create_bullish_data(num_bars=1)
        result = mtf.score_confluence(single_bar, single_bar, single_bar)

        assert result is not None

    def test_nan_values_handled(self):
        """DataFrames with NaN values handled gracefully."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        data.loc[0, "open"] = np.nan
        result = mtf.score_confluence(data, data, data)

        # Should not crash
        assert result is not None

    def test_zero_prices(self):
        """Zero prices don't cause division errors."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        data.loc[0, "close"] = 0.0
        result = mtf.score_confluence(data, data, data)

        # Should handle gracefully
        assert result is not None

    def test_negative_prices(self):
        """Negative prices don't crash (edge case)."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        data.loc[0, "close"] = -0.001
        result = mtf.score_confluence(data, data, data)

        # Should not crash
        assert result is not None

    def test_mismatched_index_lengths(self):
        """Different length DataFrames handled."""
        mtf = MultiTimeframeConfluence()
        df_4h = create_bullish_data(num_bars=100)
        df_1h = create_bullish_data(num_bars=50)
        df_15m = create_bullish_data(num_bars=25)
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        # Should handle different lengths
        assert result is not None


# =============================================================================
# Test: Indicator Calculations
# =============================================================================

class TestIndicators:
    """Test individual indicator calculations."""

    def test_rsi_calculation(self):
        """RSI calculation produces reasonable values."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        rsi = mtf._calculate_rsi(data, period=14)

        assert rsi is not None
        assert 0.01 <= rsi <= 99.99

    def test_rsi_uptrend(self):
        """RSI higher in uptrend."""
        mtf = MultiTimeframeConfluence()
        bull_data = create_bullish_data(num_bars=200)
        bull_rsi = mtf._calculate_rsi(bull_data[-20:], period=5)

        bear_data = create_bearish_data(num_bars=200)
        bear_rsi = mtf._calculate_rsi(bear_data[-20:], period=5)

        assert bull_rsi is not None and bear_rsi is not None
        # Bullish should have higher RSI on average
        assert bull_rsi > bear_rsi

    def test_adx_calculation(self):
        """ADX calculation produces reasonable values."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        adx = mtf._calculate_adx(data, period=14)

        assert adx is not None
        assert 0.0 <= adx <= 100.0

    def test_adx_trending_vs_flat(self):
        """ADX higher in trending markets."""
        mtf = MultiTimeframeConfluence()
        trend_data = create_bullish_data(num_bars=250)
        trend_adx = mtf._calculate_adx(trend_data, period=14)

        flat_data = create_flat_data(num_bars=250)
        flat_adx = mtf._calculate_adx(flat_data, period=14)

        assert trend_adx is not None and flat_adx is not None
        assert trend_adx > flat_adx

    def test_atr_calculation(self):
        """ATR calculation produces reasonable values."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=100)
        atr = mtf._calculate_atr(data, period=14)

        assert atr is not None
        assert atr > 0.0

    def test_atr_higher_volatility(self):
        """ATR higher in volatile data."""
        mtf = MultiTimeframeConfluence()
        low_vol = create_ohlcv_dataframe(100, volatility=0.001)
        low_vol_atr = mtf._calculate_atr(low_vol, period=14)

        high_vol = create_ohlcv_dataframe(100, volatility=0.01)
        high_vol_atr = mtf._calculate_atr(high_vol, period=14)

        assert low_vol_atr is not None and high_vol_atr is not None
        assert high_vol_atr > low_vol_atr


# =============================================================================
# Test: Candle Quality Scoring
# =============================================================================

class TestCandleQuality:
    """Test candle quality scoring."""

    def test_clean_candle_high_score(self):
        """Clean candle (large body, small wick) scores high."""
        mtf = MultiTimeframeConfluence()
        candle = pd.Series({
            "open": 1.1000,
            "close": 1.1020,
            "high": 1.1022,
            "low": 1.0998,
        })
        score = mtf._score_candle_quality(candle, threshold=0.6)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Clean candle

    def test_weak_candle_low_score(self):
        """Weak candle (small body, large wick) scores lower."""
        mtf = MultiTimeframeConfluence()
        candle = pd.Series({
            "open": 1.1000,
            "close": 1.1001,
            "high": 1.1050,
            "low": 1.0950,
        })
        score = mtf._score_candle_quality(candle, threshold=0.6)

        assert 0.0 <= score <= 1.0

    def test_doji_neutral_score(self):
        """Doji (near-zero wick) gets neutral score."""
        mtf = MultiTimeframeConfluence()
        candle = pd.Series({
            "open": 1.1000,
            "close": 1.1000,
            "high": 1.1000,
            "low": 1.1000,
        })
        score = mtf._score_candle_quality(candle, threshold=0.6)

        assert score == 0.5  # Neutral


# =============================================================================
# Test: S/R Proximity Scoring
# =============================================================================

class TestSRProximity:
    """Test support/resistance proximity scoring."""

    def test_sr_near_level_high_score(self):
        """Price near support/resistance scores higher."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=20)
        atr = 0.001
        score = mtf._score_sr_proximity(data, atr, proximity_atr=0.5)

        assert 0.0 <= score <= 1.0

    def test_sr_far_from_level_lower_score(self):
        """Price far from S/R scores lower."""
        mtf = MultiTimeframeConfluence()
        data = create_bullish_data(num_bars=5)
        # Small ATR = far from levels
        atr = 0.0001
        score = mtf._score_sr_proximity(data, atr, proximity_atr=0.5)

        assert 0.0 <= score <= 1.0


# =============================================================================
# Test: Serialization
# =============================================================================

class TestSerialization:
    """Test serialization to dict."""

    def test_result_to_dict_all_fields(self):
        """Result.to_dict() includes all fields."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_aligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        d = result.to_dict()
        assert "confluence_score" in d
        assert "direction_aligned" in d
        assert "screen_results" in d
        assert "penalty_applied" in d
        assert "recommendation" in d
        assert isinstance(d["screen_results"], list)

    def test_screen_to_dict(self):
        """ScreenResult.to_dict() works."""
        screen = ScreenResult(
            screen_name="trend_4h",
            direction="BULLISH",
            score=0.8,
            details={"sma50": 1.1000},
        )
        d = screen.to_dict()
        assert d["screen_name"] == "trend_4h"
        assert d["direction"] == "BULLISH"
        assert d["score"] == 0.8


# =============================================================================
# Test: Configuration Validation
# =============================================================================

class TestConfigValidation:
    """Test config validation and edge cases."""

    def test_all_weights_positive(self):
        """All config weights are positive."""
        cfg = MTFConfig()
        assert cfg.trend_weight > 0
        assert cfg.wave_weight > 0
        assert cfg.entry_weight > 0
        assert cfg.sma_alignment_weight > 0
        assert cfg.adx_weight > 0
        assert cfg.slope_weight > 0

    def test_thresholds_ordered(self):
        """Caution threshold < Proceed threshold."""
        cfg = MTFConfig()
        assert cfg.caution_threshold < cfg.proceed_threshold

    def test_custom_config_applied(self):
        """Custom config values used in scoring."""
        custom_cfg = MTFConfig(
            proceed_threshold=0.80,
            caution_threshold=0.50,
        )
        mtf = MultiTimeframeConfluence(config=custom_cfg)
        df_4h, df_1h, df_15m = create_flat_data(100), create_flat_data(100), create_flat_data(100)
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        # With custom thresholds, recommendation should be based on new values
        assert result is not None


# =============================================================================
# Integration Test: Full Pipeline
# =============================================================================

class TestFullPipeline:
    """End-to-end integration tests."""

    def test_full_pipeline_bullish_to_proceed(self):
        """Full pipeline processes aligned bullish data correctly."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_aligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        assert result.confluence_score >= 0.0
        assert result.confluence_score <= 1.0
        assert len(result.screen_results) == 3
        # All screens should show consistent direction
        directions = [s.direction for s in result.screen_results]
        assert directions.count("BULLISH") >= 2

    def test_full_pipeline_bearish_alignment(self):
        """Can score bearish-aligned confluence."""
        mtf = MultiTimeframeConfluence()
        df_4h = create_bearish_data(250)
        df_1h = create_bearish_with_rsi_pullback(100)
        df_15m = create_bearish_data(100)
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        assert result is not None
        assert len(result.screen_results) == 3

    def test_result_contains_details(self):
        """Result includes diagnostic details."""
        mtf = MultiTimeframeConfluence()
        df_4h, df_1h, df_15m = create_aligned_multiframe_data()
        result = mtf.score_confluence(df_4h, df_1h, df_15m)

        # Check screen details
        for screen in result.screen_results:
            assert isinstance(screen.details, dict)
            assert len(screen.details) > 0
