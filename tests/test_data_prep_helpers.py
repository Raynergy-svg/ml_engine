"""
Tests for data preparation utilities (US-256, US-257).

Covers:
- US-256: augment_time_series, augment_time_series_batch, PreparedData, ScalerState
- US-257: Training helper utilities (_require_ohlcv_columns, _apply_min_volume_filter, etc.)
"""

import numpy as np
import pandas as pd
import pytest

from src.training.data_preparation import (
    augment_time_series,
    augment_time_series_batch,
    PreparedData,
    ScalerState,
)
from src.training.buddy_training_helpers import (
    _require_ohlcv_columns,
    _apply_min_volume_filter,
    _apply_spread_filter,
    _normalize_pair_dataframe,
    _combine_pair_dataframes,
    _get_available_features,
    _compute_direction_targets,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_mock_console():
    """Simple mock console with print method."""
    class MockConsole:
        def __init__(self):
            self.messages = []

        def print(self, *args, **kwargs):
            self.messages.append(args)

    return MockConsole()


@pytest.fixture
def sample_features():
    """Create sample feature array for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def sample_ohlcv_dataframe():
    """Create a sample OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    df = pd.DataFrame({
        "time": dates,
        "open": np.random.uniform(1.0, 1.1, 100),
        "high": np.random.uniform(1.0, 1.1, 100),
        "low": np.random.uniform(0.9, 1.0, 100),
        "close": np.random.uniform(1.0, 1.1, 100),
        "volume": np.random.randint(1000, 10000, 100),
    })
    return df


# =============================================================================
# US-256: Data Augmentation Tests
# =============================================================================

class TestAugmentTimeSeries:
    """Test augment_time_series function."""

    def test_augment_with_zero_probability_unchanged(self, sample_features):
        """Output shape matches input when augment_prob=0."""
        X, y = sample_features
        X_aug, y_aug = augment_time_series(X, y, augment_prob=0.0)

        assert X_aug.shape == X.shape
        assert y_aug.shape == y.shape

    def test_augment_with_high_probability_differs(self, sample_features):
        """With augment_prob=1.0, output differs from input."""
        np.random.seed(42)
        X, y = sample_features

        # Run augmentation with prob=1.0
        X_aug, y_aug = augment_time_series(
            X, y,
            augment_prob=1.0,
            jittering_config={"enabled": True, "apply_prob": 1.0},
        )

        # Should be different (jitter was applied)
        assert not np.allclose(X_aug, X)
        assert X_aug.shape == X.shape

    def test_augment_labels_unchanged(self, sample_features):
        """Labels (y) should remain unchanged."""
        X, y = sample_features
        y_original = y.copy()

        X_aug, y_aug = augment_time_series(X, y, augment_prob=0.5)

        np.testing.assert_array_equal(y_aug, y_original)

    def test_augment_jittering_enabled(self, sample_features):
        """Jittering should add noise when enabled."""
        np.random.seed(42)
        X, y = sample_features

        X_aug, y_aug = augment_time_series(
            X, y,
            augment_prob=1.0,
            jittering_config={
                "enabled": True,
                "apply_prob": 1.0,
                "noise_std_min": 0.01,
                "noise_std_max": 0.05,
            },
        )

        # Should differ from original
        assert not np.allclose(X_aug, X)
        # But should have similar scale
        assert np.abs(np.mean(X_aug) - np.mean(X)) < 0.5

    def test_augment_scaling_enabled(self, sample_features):
        """Scaling should multiply features when enabled."""
        np.random.seed(42)
        X, y = sample_features

        X_aug, y_aug = augment_time_series(
            X, y,
            augment_prob=1.0,
            jittering_config={"enabled": False},
            scaling_config={
                "enabled": True,
                "apply_prob": 1.0,
                "scale_min": 0.95,
                "scale_max": 1.05,
            },
        )

        # Should be scaled version
        assert X_aug.shape == X.shape
        # Check that scaling was applied (not identical, within 5%)
        assert not np.allclose(X_aug, X)

    def test_augment_disables_both_when_neither(self, sample_features):
        """When both augmentations disabled, should return copy."""
        X, y = sample_features
        X_original = X.copy()

        X_aug, y_aug = augment_time_series(
            X, y,
            augment_prob=1.0,
            jittering_config={"enabled": False},
            scaling_config={"enabled": False},
        )

        # Should be identical (no augmentation applied)
        np.testing.assert_array_almost_equal(X_aug, X_original)


class TestAugmentTimeSeriesBatch:
    """Test augment_time_series_batch function."""

    def test_batch_disabled_returns_copy(self, sample_features):
        """With config disabled, returns unchanged copy."""
        X, y = sample_features

        config = {"enabled": False}
        X_aug, y_aug = augment_time_series_batch(X, y, config=config)

        assert X_aug.shape == X.shape
        np.testing.assert_array_almost_equal(X_aug, X)

    def test_batch_time_series_disabled(self, sample_features):
        """When time_series disabled in config, returns unchanged."""
        X, y = sample_features

        config = {
            "enabled": True,
            "time_series": {"enabled": False},
        }
        X_aug, y_aug = augment_time_series_batch(X, y, config=config)

        np.testing.assert_array_almost_equal(X_aug, X)

    def test_batch_enabled_augments(self, sample_features):
        """With config enabled, augments features."""
        np.random.seed(42)
        X, y = sample_features

        config = {
            "enabled": True,
            "time_series": {
                "enabled": True,
                "augment_prob": 1.0,
                "jittering": {"enabled": True, "apply_prob": 1.0},
            },
        }
        X_aug, y_aug = augment_time_series_batch(X, y, config=config)

        # Should be augmented
        assert not np.allclose(X_aug, X)

    def test_batch_with_none_config(self, sample_features):
        """With None config, returns copy unchanged."""
        X, y = sample_features

        X_aug, y_aug = augment_time_series_batch(X, y, config=None)

        np.testing.assert_array_almost_equal(X_aug, X)


# =============================================================================
# US-256: PreparedData Tests
# =============================================================================

class TestPreparedDataProperties:
    """Test PreparedData container properties."""

    def test_n_train_property(self):
        """n_train property returns correct training sample count."""
        X_train = np.random.randn(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(5),
            scale_=np.ones(5),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            scaler=scaler,
            df=df,
        )

        assert prep.n_train == 80

    def test_n_val_property(self):
        """n_val property returns correct validation sample count."""
        X_train = np.random.randn(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(5),
            scale_=np.ones(5),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            scaler=scaler,
            df=df,
        )

        assert prep.n_val == 20

    def test_n_features_property(self):
        """n_features property returns correct feature count."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(10),
            scale_=np.ones(10),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=[f"f{i}" for i in range(10)],
            scaler=scaler,
            df=df,
        )

        assert prep.n_features == 10

    def test_n_features_empty_training(self):
        """n_features returns 0 for empty training set."""
        X_train = np.array([]).reshape(0, 5)
        y_train = np.array([])
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(5),
            scale_=np.ones(5),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            scaler=scaler,
            df=df,
        )

        assert prep.n_features == 0


class TestPreparedDataValidate:
    """Test PreparedData.validate() method."""

    def test_validate_valid_data(self):
        """validate() returns empty list for valid data."""
        X_train = np.random.randn(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(5),
            scale_=np.ones(5),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            scaler=scaler,
            df=df,
        )

        warnings = prep.validate()
        assert warnings == []

    def test_validate_nan_in_train_features(self):
        """validate() detects NaN in training features."""
        X_train = np.random.randn(80, 5)
        X_train[0, 0] = np.nan
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(5),
            scale_=np.ones(5),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            scaler=scaler,
            df=df,
        )

        warnings = prep.validate()
        assert any("Non-finite" in w for w in warnings)

    def test_validate_inf_in_val_features(self):
        """validate() detects inf in validation features."""
        X_train = np.random.randn(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.randn(20, 5)
        X_val[0, 0] = np.inf
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(5),
            scale_=np.ones(5),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            scaler=scaler,
            df=df,
        )

        warnings = prep.validate()
        assert any("Non-finite" in w for w in warnings)

    def test_validate_empty_training(self):
        """validate() warns about empty training set."""
        X_train = np.array([]).reshape(0, 5)
        y_train = np.array([])
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(5),
            scale_=np.ones(5),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            scaler=scaler,
            df=df,
        )

        warnings = prep.validate()
        assert any("Empty training set" in w for w in warnings)

    def test_validate_extreme_class_imbalance(self):
        """validate() warns about extreme class imbalance."""
        X_train = np.random.randn(100, 5)
        y_train = np.array([0] * 95 + [1] * 5)  # 95% class 0
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)
        scaler = ScalerState(
            mean_=np.zeros(5),
            scale_=np.ones(5),
        )
        df = pd.DataFrame()

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            scaler=scaler,
            df=df,
        )

        warnings = prep.validate()
        assert any("class imbalance" in w for w in warnings)


# =============================================================================
# US-256: ScalerState Tests
# =============================================================================

class TestScalerState:
    """Test ScalerState standardization container."""

    def test_transform_standardizes(self):
        """transform() applies standardization correctly."""
        mean = np.array([1.0, 2.0, 3.0])
        scale = np.array([0.5, 1.0, 2.0])
        scaler = ScalerState(mean_=mean, scale_=scale)

        X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 7.0]])
        X_scaled = scaler.transform(X)

        # First sample should be [0, 0, 0]
        np.testing.assert_array_almost_equal(X_scaled[0], [0.0, 0.0, 0.0])
        # Second sample should be [(2-1)/0.5, (4-2)/1.0, (7-3)/2.0]
        np.testing.assert_array_almost_equal(X_scaled[1], [2.0, 2.0, 2.0])

    def test_inverse_transform_reverses(self):
        """inverse_transform() reverses standardization."""
        mean = np.array([1.0, 2.0, 3.0])
        scale = np.array([0.5, 1.0, 2.0])
        scaler = ScalerState(mean_=mean, scale_=scale)

        X_scaled = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        X_recovered = scaler.inverse_transform(X_scaled)

        # First sample should be [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(X_recovered[0], [1.0, 2.0, 3.0])
        # Second sample should be [2.0, 4.0, 7.0]
        np.testing.assert_array_almost_equal(X_recovered[1], [2.0, 4.0, 7.0])

    def test_round_trip_identity(self):
        """transform then inverse_transform is identity."""
        np.random.seed(42)
        mean = np.array([1.0, 2.0, 3.0])
        scale = np.array([0.5, 1.0, 2.0])
        scaler = ScalerState(mean_=mean, scale_=scale)

        X_original = np.array([[1.5, 3.0, 5.0], [2.0, 4.0, 7.0], [0.5, 1.0, 1.0]])
        X_scaled = scaler.transform(X_original)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X_recovered, X_original)

    def test_zero_scale_handled(self):
        """Handles zero scale gracefully (division behavior)."""
        mean = np.array([1.0, 2.0, 3.0])
        scale = np.array([0.0, 1.0, 2.0])  # Zero scale for first feature
        scaler = ScalerState(mean_=mean, scale_=scale)

        X = np.array([[1.0, 2.0, 3.0]])
        # This will produce NaN or inf for first feature due to division by zero
        X_scaled = scaler.transform(X)

        # First feature should be NaN or inf (0.0 / 0.0 = NaN, or 0.0 / 0.0)
        assert np.isnan(X_scaled[0, 0]) or np.isinf(X_scaled[0, 0])
        # Others should be normal
        np.testing.assert_array_almost_equal(X_scaled[0, 1:], [0.0, 0.0])


# =============================================================================
# US-257: Training Helper Tests
# =============================================================================

class TestRequireOHLCVColumns:
    """Test _require_ohlcv_columns function."""

    def test_valid_ohlcv_passes(self, sample_ohlcv_dataframe):
        """Passes with valid OHLCV DataFrame."""
        # Should not raise
        _require_ohlcv_columns(sample_ohlcv_dataframe)

    def test_missing_open_raises(self):
        """Raises ValueError when open column missing."""
        df = pd.DataFrame({
            "high": [1.0, 2.0],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [100, 200],
        })

        with pytest.raises(ValueError, match="missing required columns"):
            _require_ohlcv_columns(df)

    def test_missing_volume_raises(self):
        """Raises ValueError when volume column missing."""
        df = pd.DataFrame({
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
        })

        with pytest.raises(ValueError, match="missing required columns"):
            _require_ohlcv_columns(df)

    def test_multiple_missing_columns(self):
        """Raises ValueError with all missing columns listed."""
        df = pd.DataFrame({
            "open": [1.0, 2.0],
            "close": [1.0, 2.0],
        })

        with pytest.raises(ValueError, match="missing required columns"):
            _require_ohlcv_columns(df)


class TestApplyMinVolumeFilter:
    """Test _apply_min_volume_filter function."""

    def test_filters_low_volume_rows(self, simple_mock_console):
        """Filters low volume rows."""
        df = pd.DataFrame({
            "volume": [100, 500, 1000, 200, 5000],
        })

        result = _apply_min_volume_filter(
            df=df,
            min_volume=300,
            console=simple_mock_console,
        )

        # Should keep only rows with volume >= 300
        assert len(result) == 3
        assert list(result["volume"]) == [500, 1000, 5000]

    def test_keeps_all_high_volume(self, simple_mock_console):
        """Keeps all rows when volume is high."""
        df = pd.DataFrame({
            "volume": [1000, 5000, 10000],
        })

        result = _apply_min_volume_filter(
            df=df,
            min_volume=100,
            console=simple_mock_console,
        )

        assert len(result) == 3

    def test_none_min_volume_returns_unchanged(self, simple_mock_console):
        """Returns unchanged DataFrame when min_volume is None."""
        df = pd.DataFrame({
            "volume": [100, 200, 300],
        })

        result = _apply_min_volume_filter(
            df=df,
            min_volume=None,
            console=simple_mock_console,
        )

        pd.testing.assert_frame_equal(result, df)

    def test_prints_filtered_count(self, simple_mock_console):
        """Prints message about filtered rows."""
        df = pd.DataFrame({
            "volume": [100, 500, 1000],
        })

        _apply_min_volume_filter(
            df=df,
            min_volume=300,
            console=simple_mock_console,
        )

        # Should print a message
        assert len(simple_mock_console.messages) > 0


class TestApplySpreadFilter:
    """Test _apply_spread_filter function."""

    def test_filters_wide_spread_candles(self, simple_mock_console):
        """Filters wide-spread candles."""
        df = pd.DataFrame({
            "bid_close": [1.0, 1.0, 1.0, 1.0],
            "ask_close": [1.01, 1.005, 1.015, 1.002],
        })

        result = _apply_spread_filter(
            df=df,
            spread_filter=True,
            spread_pctl=0.99,
            spread_mult=1.0,
            console=simple_mock_console,
        )

        # Should filter based on spread threshold
        assert len(result) <= len(df)

    def test_no_filter_when_disabled(self, simple_mock_console):
        """Returns unchanged when spread_filter is False."""
        df = pd.DataFrame({
            "bid_close": [1.0, 1.0],
            "ask_close": [1.1, 1.2],
        })

        result = _apply_spread_filter(
            df=df,
            spread_filter=False,
            spread_pctl=0.99,
            spread_mult=1.0,
            console=simple_mock_console,
        )

        pd.testing.assert_frame_equal(result, df)

    def test_missing_bid_ask_columns(self, simple_mock_console):
        """Skips filter when bid/ask columns missing."""
        df = pd.DataFrame({
            "open": [1.0, 2.0],
            "close": [1.0, 2.0],
        })

        result = _apply_spread_filter(
            df=df,
            spread_filter=True,
            spread_pctl=0.99,
            spread_mult=1.0,
            console=simple_mock_console,
        )

        pd.testing.assert_frame_equal(result, df)


class TestNormalizePairDataFrame:
    """Test _normalize_pair_dataframe function."""

    def test_computes_pct_returns(self):
        """Computes percentage returns correctly."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3),
            "open": [1.0, 1.01, 1.02],
            "high": [1.05, 1.06, 1.07],
            "low": [0.95, 0.96, 0.97],
            "close": [1.01, 1.02, 1.03],
            "volume": [1000, 2000, 3000],
        })

        result = _normalize_pair_dataframe(df, "USD_JPY")

        # Should have pct columns
        assert "open_pct" in result.columns
        assert "close_pct" in result.columns
        assert "volume_zscore" in result.columns

    def test_adds_pair_column(self):
        """Adds pair column to DataFrame."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3),
            "open": [1.0, 1.01, 1.02],
            "high": [1.05, 1.06, 1.07],
            "low": [0.95, 0.96, 0.97],
            "close": [1.01, 1.02, 1.03],
            "volume": [1000, 2000, 3000],
        })

        result = _normalize_pair_dataframe(df, "EUR_USD")

        assert "pair" in result.columns
        assert all(result["pair"] == "EUR_USD")

    def test_drops_first_row(self):
        """Drops first row (pct_change has NaN)."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=5),
            "open": [1.0, 1.01, 1.02, 1.03, 1.04],
            "high": [1.05, 1.06, 1.07, 1.08, 1.09],
            "low": [0.95, 0.96, 0.97, 0.98, 0.99],
            "close": [1.01, 1.02, 1.03, 1.04, 1.05],
            "volume": [1000, 2000, 3000, 4000, 5000],
        })

        result = _normalize_pair_dataframe(df, "GBP_USD")

        assert len(result) == 4  # 5 - 1


class TestCombinePairDataFrames:
    """Test _combine_pair_dataframes function."""

    def test_concatenates_pair_dataframes(self, simple_mock_console):
        """Concatenates multiple pair DataFrames."""
        df1 = pd.DataFrame({
            "open": [1.0, 1.01],
            "close": [1.01, 1.02],
        })
        df2 = pd.DataFrame({
            "open": [2.0, 2.01],
            "close": [2.01, 2.02],
        })

        result = _combine_pair_dataframes([df1, df2], simple_mock_console)

        assert len(result) == 4

    def test_drops_time_and_pair_columns(self, simple_mock_console):
        """Drops time and pair columns."""
        df1 = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=2),
            "pair": ["USD_JPY", "USD_JPY"],
            "open": [1.0, 1.01],
            "close": [1.01, 1.02],
        })
        df2 = pd.DataFrame({
            "time": pd.date_range("2024-01-03", periods=2),
            "pair": ["EUR_USD", "EUR_USD"],
            "open": [2.0, 2.01],
            "close": [2.01, 2.02],
        })

        result = _combine_pair_dataframes([df1, df2], simple_mock_console)

        assert "time" not in result.columns
        assert "pair" not in result.columns
        assert "open" in result.columns
        assert "close" in result.columns

    def test_shuffles_data(self, simple_mock_console):
        """Shuffles data for training."""
        # Create predictable data
        df1 = pd.DataFrame({
            "value": [1.0, 1.0, 1.0, 1.0],
        })
        df2 = pd.DataFrame({
            "value": [2.0, 2.0, 2.0, 2.0],
        })

        np.random.seed(42)
        result = _combine_pair_dataframes([df1, df2], simple_mock_console)

        # Should be mixed (not all 1s then all 2s)
        values = result["value"].tolist()
        # Verify shuffling by checking it's not in original order
        assert values != [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]

    def test_raises_on_empty_list(self, simple_mock_console):
        """Raises ValueError when no data provided."""
        with pytest.raises(ValueError, match="No data loaded"):
            _combine_pair_dataframes([], simple_mock_console)


class TestGetAvailableFeatures:
    """Test _get_available_features function."""

    def test_returns_preferred_columns(self):
        """Returns columns from preferred list if available."""
        df = pd.DataFrame({
            "returns_1": [0.01, 0.02],
            "returns_2": [0.02, 0.03],
            "atr_pct_5": [0.05, 0.06],
            "other": [1.0, 2.0],
        })
        preferred = ["returns_1", "returns_2", "atr_pct_5", "missing"]

        result = _get_available_features(df, preferred)

        # Should return only available preferred columns
        assert "returns_1" in result
        assert "returns_2" in result
        assert "atr_pct_5" in result
        assert "missing" not in result

    def test_fallback_to_numeric_columns(self):
        """Falls back to numeric columns when preferred unavailable."""
        df = pd.DataFrame({
            "x1": [1.0, 2.0],
            "x2": [3.0, 4.0],
            "x3": [5.0, 6.0],
            "x4": [7.0, 8.0],
            "x5": [9.0, 10.0],
            "x6": [11.0, 12.0],
            "x7": [13.0, 14.0],
            "x8": [15.0, 16.0],
            "x9": [17.0, 18.0],
            "x10": [19.0, 20.0],
            "x11": [21.0, 22.0],
            "text": ["a", "b"],
        })
        preferred = ["not_exist_1", "not_exist_2"]

        result = _get_available_features(df, preferred)

        # Should return numeric columns
        assert len(result) > 0
        assert all(c.startswith("x") for c in result)
        assert "text" not in result

    def test_excludes_ohlcv_and_target(self):
        """Fallback excludes OHLCV and target columns."""
        df = pd.DataFrame({
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.1, 2.1],
            "volume": [1000, 2000],
            "direction_target": [0, 1],
            "feature_1": [0.1, 0.2],
            "feature_2": [0.2, 0.3],
            "feature_3": [0.3, 0.4],
        })
        preferred = ["not_exist_1"]

        result = _get_available_features(df, preferred)

        # Should not include OHLCV or direction_target
        assert "open" not in result
        assert "volume" not in result
        assert "direction_target" not in result
        # Should include features
        assert any(c.startswith("feature") for c in result)


class TestComputeDirectionTargets:
    """Test _compute_direction_targets function."""

    def test_returns_correct_shape(self):
        """Returns correct output shape."""
        df = pd.DataFrame({
            "close": np.linspace(1.0, 1.1, 100),
        })

        y, valid_mask = _compute_direction_targets(df, lookahead=24, threshold=0.003)

        assert y.shape == (100,)
        assert valid_mask.shape == (100,)

    def test_up_move_returns_1(self):
        """Up moves return label 1."""
        close_prices = np.array([1.0, 1.01, 1.02, 1.03, 1.05, 1.1])
        df = pd.DataFrame({"close": close_prices})

        y, valid_mask = _compute_direction_targets(df, lookahead=2, threshold=0.003)

        # Fourth value (index 3): (1.05 - 1.03) / 1.03 ≈ 0.0194 > 0.003, so 1
        assert y[3] == 1

    def test_down_move_returns_0(self):
        """Down moves return label 0."""
        close_prices = np.array([1.01, 1.009, 1.008, 1.005, 1.0, 0.995])
        df = pd.DataFrame({"close": close_prices})

        y, valid_mask = _compute_direction_targets(df, lookahead=2, threshold=0.003)

        # Third value: (1.005 - 1.008) / 1.008 ≈ -0.003, so 0
        assert y[2] == 0

    def test_ambiguous_returns_minus_1(self):
        """Ambiguous moves (within threshold) return -1."""
        close_prices = np.array([1.0, 1.0001, 1.0002, 1.0003, 1.0004])
        df = pd.DataFrame({"close": close_prices})

        y, valid_mask = _compute_direction_targets(df, lookahead=1, threshold=0.003)

        # All returns are very small, should be -1
        assert all(y[i] == -1 for i in range(len(y) - 1))

    def test_valid_mask_filters_ambiguous(self):
        """valid_mask filters out ambiguous labels (-1)."""
        close_prices = np.array([1.0, 1.001, 1.01, 1.0, 0.99])
        df = pd.DataFrame({"close": close_prices})

        y, valid_mask = _compute_direction_targets(df, lookahead=1, threshold=0.003)

        # valid_mask should be True where y >= 0 (i.e., not -1)
        assert valid_mask[0] == (y[0] >= 0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestDataPrepIntegration:
    """Integration tests for data preparation workflow."""

    def test_augmentation_preserves_shape(self, sample_features):
        """Augmentation through batch pipeline preserves shape."""
        X, y = sample_features
        config = {
            "enabled": True,
            "time_series": {
                "enabled": True,
                "augment_prob": 0.5,
                "jittering": {"enabled": True},
            },
        }

        X_aug, y_aug = augment_time_series_batch(X, y, config=config)

        assert X_aug.shape == X.shape
        assert y_aug.shape == y.shape

    def test_scaler_with_prepared_data(self):
        """ScalerState works with PreparedData."""
        np.random.seed(42)
        mean = np.mean(np.random.randn(100, 5), axis=0)
        scale = np.std(np.random.randn(100, 5), axis=0)

        X_train = np.random.randn(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)

        scaler = ScalerState(mean_=mean, scale_=scale)

        prep = PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=[f"f{i}" for i in range(5)],
            scaler=scaler,
            df=pd.DataFrame(),
        )

        # Should be valid
        assert prep.n_train == 80
        assert prep.n_val == 20
        assert prep.n_features == 5
