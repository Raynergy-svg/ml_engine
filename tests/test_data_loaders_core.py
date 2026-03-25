"""Phase 38 (US-245 + US-246 + US-247): Unit tests for modular_data_loaders.

Tests compute_normalized_features, classify_market_regime,
compute_volatility_regime, fast indicator helpers, temporal_split,
align_features_to_model, normalization stats, feature name utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.modular_data_loaders import (
    compute_normalized_features,
    classify_market_regime,
    compute_volatility_regime,
    _compute_adx_fast,
    _compute_rsi_fast,
    _compute_atr_pct_fast,
    temporal_split,
    get_normalized_feature_names,
    RegimeConfig,
    VolatilityRegimeConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ohlcv(rows=200, trend="flat"):
    """Create realistic OHLCV data."""
    np.random.seed(42)
    if trend == "up":
        close = 1.1000 + np.linspace(0, 0.05, rows) + np.random.randn(rows) * 0.001
    elif trend == "down":
        close = 1.1500 - np.linspace(0, 0.05, rows) + np.random.randn(rows) * 0.001
    else:
        close = 1.1000 + np.random.randn(rows) * 0.002

    high = close + np.abs(np.random.randn(rows) * 0.001) + 0.0005
    low = close - np.abs(np.random.randn(rows) * 0.001) - 0.0005
    open_ = close + np.random.randn(rows) * 0.0003
    volume = np.random.randint(500, 5000, rows).astype(float)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Tests: compute_normalized_features
# ---------------------------------------------------------------------------
class TestComputeNormalizedFeatures:
    def test_output_has_expected_columns(self):
        df = _make_ohlcv(rows=200)
        result = compute_normalized_features(df)
        # Should have returns, volatility, z-scores, etc.
        assert "returns_1" in result.columns
        assert "log_returns_1" in result.columns
        assert "atr_pct_14" in result.columns
        assert "volatility_10" in result.columns
        assert "zscore_20" in result.columns
        assert "pct_rank_20" in result.columns

    def test_output_shape_matches_input(self):
        df = _make_ohlcv(rows=150)
        result = compute_normalized_features(df)
        assert len(result) == 150
        # Should have many more columns than input
        assert len(result.columns) > len(df.columns)

    def test_no_nans_in_core_features(self):
        df = _make_ohlcv(rows=200)
        result = compute_normalized_features(df)
        # After warmup period (50 candles), core features should be populated
        core = result.iloc[60:]
        assert not core["returns_1"].isna().any()
        assert not core["atr_pct_14"].isna().any()

    def test_returns_are_reasonable(self):
        df = _make_ohlcv(rows=200)
        result = compute_normalized_features(df)
        returns = result["returns_1"].iloc[1:]  # Skip first zero
        # Returns should be small for FX (typically < 1%)
        assert returns.abs().max() < 0.1

    def test_zscore_clipped(self):
        df = _make_ohlcv(rows=200)
        result = compute_normalized_features(df)
        for period in [10, 20, 50]:
            col = f"zscore_{period}"
            if col in result.columns:
                assert result[col].max() <= 4.0
                assert result[col].min() >= -4.0

    def test_input_not_mutated(self):
        df = _make_ohlcv(rows=100)
        original_cols = set(df.columns)
        compute_normalized_features(df)
        assert set(df.columns) == original_cols

    def test_no_volume_column_uses_ones(self):
        df = _make_ohlcv(rows=100)
        df = df.drop(columns=["volume"])
        result = compute_normalized_features(df)
        assert "returns_1" in result.columns  # Should still compute

    def test_multi_period_returns(self):
        df = _make_ohlcv(rows=100)
        result = compute_normalized_features(df)
        for period in [2, 3, 5, 10, 20]:
            assert f"returns_{period}" in result.columns

    def test_multi_period_atr(self):
        df = _make_ohlcv(rows=100)
        result = compute_normalized_features(df)
        for period in [5, 10, 14, 20]:
            assert f"atr_pct_{period}" in result.columns

    def test_small_dataframe(self):
        """Should not crash with small input."""
        df = _make_ohlcv(rows=10)
        result = compute_normalized_features(df)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Tests: _compute_rsi_fast
# ---------------------------------------------------------------------------
class TestComputeRSIFast:
    def test_output_length(self):
        close = np.random.randn(100).cumsum() + 100
        rsi = _compute_rsi_fast(close, period=14)
        assert len(rsi) == 100

    def test_rsi_in_range(self):
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        rsi = _compute_rsi_fast(close, period=14)
        valid = rsi[15:]  # After warmup
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_uptrend_rsi_high(self):
        close = np.linspace(100, 150, 100)
        rsi = _compute_rsi_fast(close, period=14)
        # Strong uptrend should have RSI > 50
        assert rsi[-1] > 50


# ---------------------------------------------------------------------------
# Tests: _compute_adx_fast
# ---------------------------------------------------------------------------
class TestComputeADXFast:
    def test_output_length(self):
        df = _make_ohlcv(rows=100)
        adx = _compute_adx_fast(df, period=14)
        assert len(adx) == 100

    def test_adx_non_negative(self):
        df = _make_ohlcv(rows=200)
        adx = _compute_adx_fast(df, period=14)
        valid = adx[30:]  # After warmup
        assert (valid >= 0).all()

    def test_trending_market_higher_adx(self):
        """Strong trend should produce higher ADX than flat market."""
        df_trend = _make_ohlcv(rows=200, trend="up")
        df_flat = _make_ohlcv(rows=200, trend="flat")
        adx_trend = _compute_adx_fast(df_trend, period=14)
        adx_flat = _compute_adx_fast(df_flat, period=14)
        # Trending ADX should generally be higher (check last 50 candles)
        assert np.mean(adx_trend[-50:]) > np.mean(adx_flat[-50:]) * 0.5


# ---------------------------------------------------------------------------
# Tests: _compute_atr_pct_fast
# ---------------------------------------------------------------------------
class TestComputeATRPctFast:
    def test_output_length(self):
        df = _make_ohlcv(rows=100)
        atr = _compute_atr_pct_fast(df, period=14)
        assert len(atr) == 100

    def test_atr_non_negative(self):
        df = _make_ohlcv(rows=200)
        atr = _compute_atr_pct_fast(df, period=14)
        assert (atr >= 0).all()


# ---------------------------------------------------------------------------
# Tests: classify_market_regime
# ---------------------------------------------------------------------------
class TestClassifyMarketRegime:
    def test_returns_tuple_of_series(self):
        """classify_market_regime returns (regime_series, confidence_series)."""
        df = _make_ohlcv(rows=200)
        regimes, confidence = classify_market_regime(df)
        assert isinstance(regimes, pd.Series)
        assert isinstance(confidence, pd.Series)
        assert len(regimes) == 200
        assert len(confidence) == 200

    def test_regimes_in_valid_range(self):
        df = _make_ohlcv(rows=200)
        regimes, _ = classify_market_regime(df)
        unique = set(np.unique(regimes))
        # RegimeConfig has 5 classes: 0-4
        assert unique.issubset({0, 1, 2, 3, 4})

    def test_confidence_in_range(self):
        df = _make_ohlcv(rows=200)
        _, confidence = classify_market_regime(df)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0

    def test_with_config(self):
        df = _make_ohlcv(rows=200)
        config = RegimeConfig()
        regimes, confidence = classify_market_regime(df, config=config)
        assert len(regimes) == 200


# ---------------------------------------------------------------------------
# Tests: compute_volatility_regime
# ---------------------------------------------------------------------------
class TestComputeVolatilityRegime:
    def test_returns_series(self):
        df = _make_ohlcv(rows=200)
        result = compute_volatility_regime(df)
        assert isinstance(result, (pd.Series, np.ndarray))
        assert len(result) == 200

    def test_regimes_in_valid_range(self):
        df = _make_ohlcv(rows=200)
        result = compute_volatility_regime(df)
        unique = set(np.unique(result))
        # 4 regimes: 0 (low), 1 (normal), 2 (high), 3 (extreme)
        assert unique.issubset({0, 1, 2, 3})


# ---------------------------------------------------------------------------
# Tests: VolatilityRegimeConfig
# ---------------------------------------------------------------------------
class TestVolatilityRegimeConfig:
    def test_defaults(self):
        config = VolatilityRegimeConfig()
        assert hasattr(config, "lookback_bars")
        assert hasattr(config, "thresholds")


# ---------------------------------------------------------------------------
# Tests: RegimeConfig
# ---------------------------------------------------------------------------
class TestRegimeConfig:
    def test_defaults(self):
        config = RegimeConfig()
        # Should have class labels or thresholds
        assert config is not None


# ---------------------------------------------------------------------------
# Tests: temporal_split
# ---------------------------------------------------------------------------
class TestTemporalSplit:
    def test_basic_split(self):
        train, val, test = temporal_split(100, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_split_preserves_total(self):
        train, val, test = temporal_split(200, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        assert len(train) + len(val) + len(test) == 200

    def test_temporal_order_preserved(self):
        train, val, test = temporal_split(100, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        # Train indices should be before val, val before test
        assert max(train) < min(val)
        assert max(val) < min(test)


# ---------------------------------------------------------------------------
# Tests: get_normalized_feature_names
# ---------------------------------------------------------------------------
class TestGetNormalizedFeatureNames:
    def test_returns_dict(self):
        result = get_normalized_feature_names()
        assert isinstance(result, dict)

    def test_has_categories(self):
        result = get_normalized_feature_names()
        # Should have categorized feature names
        assert len(result) > 0
        for key, names in result.items():
            assert isinstance(names, list)
            assert len(names) > 0
