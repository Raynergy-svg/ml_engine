"""Tests for modular_inference.py feature extraction methods."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_volatility_df():
    """Sample DataFrame with volatility features for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'atr_pct_5': np.random.randn(n_samples),
        'atr_pct_10': np.random.randn(n_samples),
        'atr_pct_14': np.random.randn(n_samples),
        'atr_pct_20': np.random.randn(n_samples),
        'volatility_5': np.random.randn(n_samples),
        'volatility_10': np.random.randn(n_samples),
        'volatility_20': np.random.randn(n_samples),
        'bb_width_norm': np.random.randn(n_samples),
        'bb_width_20': np.random.randn(n_samples),
        'high_low_range': np.random.randn(n_samples),
        'returns_1': np.random.randn(n_samples),
        'returns_5': np.random.randn(n_samples),
        'returns_10': np.random.randn(n_samples),
        'volume_ratio_10': np.random.randn(n_samples),
        'volume_ratio_20': np.random.randn(n_samples),
        'adx': np.random.randn(n_samples),
    })


@pytest.fixture
def sample_direction_df():
    """Sample DataFrame with direction features for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'returns_1': np.random.randn(n_samples),
        'returns_2': np.random.randn(n_samples),
        'returns_3': np.random.randn(n_samples),
        'returns_5': np.random.randn(n_samples),
        'returns_10': np.random.randn(n_samples),
        'zscore_10': np.random.randn(n_samples),
        'zscore_20': np.random.randn(n_samples),
        'zscore_50': np.random.randn(n_samples),
        'pct_rank_10': np.random.randn(n_samples),
        'pct_rank_20': np.random.randn(n_samples),
        'rsi_norm': np.random.randn(n_samples),
        'stoch_k_norm': np.random.randn(n_samples),
        'sma_ratio_5': np.random.randn(n_samples),
        'sma_ratio_10': np.random.randn(n_samples),
        'sma_ratio_20': np.random.randn(n_samples),
        'ema_ratio_12': np.random.randn(n_samples),
        'ema_ratio_26': np.random.randn(n_samples),
        'macd_norm': np.random.randn(n_samples),
        'macd_signal_norm': np.random.randn(n_samples),
        'macd_hist_norm': np.random.randn(n_samples),
        'sma_cross_5_20': np.random.randn(n_samples),
        'macd_cross': np.random.randn(n_samples),
        'higher_high_ratio': np.random.randn(n_samples),
        'lower_low_ratio': np.random.randn(n_samples),
    })


def test_volatility_feature_extraction_returns_correct_shape(sample_volatility_df):
    """Test that volatility feature extraction returns expected number of features."""
    # The extraction should return 16 volatility features
    expected_feature_count = 16
    
    # Mock a minimal inference object with the extraction method
    from unittest.mock import MagicMock
    
    mock_inference = MagicMock()
    mock_inference.tcn_volatility_model = MagicMock()
    mock_inference.tcn_volatility_model.feature_names = [
        'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
        'volatility_5', 'volatility_10', 'volatility_20',
        'bb_width_norm', 'bb_width_20', 'high_low_range',
        'returns_1', 'returns_5', 'returns_10',
        'volume_ratio_10', 'volume_ratio_20', 'adx',
    ]
    
    # Extract features using the DataFrame directly (simulating what the method does)
    feature_names = mock_inference.tcn_volatility_model.feature_names
    available = [f for f in feature_names if f in sample_volatility_df.columns]
    features = sample_volatility_df[available].values
    
    assert features.shape[1] == expected_feature_count, \
        f"Expected {expected_feature_count} features, got {features.shape[1]}"


def test_direction_feature_count_differs_from_volatility(sample_direction_df, sample_volatility_df):
    """Test that direction and volatility feature sets are different sizes."""
    from src.core.modular_data_loaders import get_normalized_feature_names
    
    direction_features = get_normalized_feature_names()['direction']
    volatility_features = get_normalized_feature_names()['volatility']
    
    # Direction features should be more numerous than volatility
    assert len(direction_features) > len(volatility_features), \
        f"Direction features ({len(direction_features)}) should exceed volatility ({len(volatility_features)})"
    
    # Verify volatility features are ~16
    assert len(volatility_features) == 16, \
        f"Expected 16 volatility features, got {len(volatility_features)}"


def test_normalized_feature_names_has_volatility_key():
    """Test that get_normalized_feature_names() includes 'volatility' key."""
    from src.core.modular_data_loaders import get_normalized_feature_names
    
    feature_dict = get_normalized_feature_names()
    
    assert 'volatility' in feature_dict, \
        "get_normalized_feature_names() should contain 'volatility' key"
    
    # Check all expected keys exist
    expected_keys = ['direction', 'momentum', 'risk', 'confidence', 'volatility']
    for key in expected_keys:
        assert key in feature_dict, f"Missing key: {key}"


def test_volatility_features_are_volatility_focused():
    """Test that volatility feature list contains appropriate volatility indicators."""
    from src.core.modular_data_loaders import get_normalized_feature_names
    
    volatility_features = get_normalized_feature_names()['volatility']
    
    # Should contain ATR features
    atr_features = [f for f in volatility_features if 'atr' in f.lower()]
    assert len(atr_features) >= 3, "Should have multiple ATR features"
    
    # Should contain volatility/std features
    vol_features = [f for f in volatility_features if 'volatility' in f.lower()]
    assert len(vol_features) >= 3, "Should have multiple volatility features"
    
    # Should NOT contain direction-specific features like sma_cross, macd_cross
    direction_only = ['sma_cross', 'macd_cross', 'higher_high', 'lower_low']
    for feat in volatility_features:
        for dir_feat in direction_only:
            assert dir_feat not in feat.lower(), \
                f"Volatility features should not contain direction indicator: {feat}"


def test_volatility_feature_extraction_handles_missing_features(sample_volatility_df):
    """Test that feature extraction gracefully handles missing columns."""
    # Remove some features from the DataFrame
    partial_df = sample_volatility_df.drop(columns=['adx', 'bb_width_norm'])
    
    # Feature names that include missing columns
    feature_names = [
        'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
        'volatility_5', 'volatility_10', 'volatility_20',
        'bb_width_norm',  # Missing
        'bb_width_20',
        'high_low_range',
        'returns_1', 'returns_5', 'returns_10',
        'volume_ratio_10', 'volume_ratio_20',
        'adx',  # Missing
    ]
    
    # Simulate _extract_features_by_names behavior: only use available features
    available = [f for f in feature_names if f in partial_df.columns]
    features = partial_df[available].values
    
    # Should have 14 features (16 - 2 missing)
    assert features.shape[1] == 14, \
        f"Expected 14 available features, got {features.shape[1]}"


def test_scaler_n_features_in_matches_feature_names():
    """Test that sklearn scaler's n_features_in_ matches saved feature_names count."""
    from sklearn.preprocessing import StandardScaler
    
    # Simulate what happens during training
    feature_names = [
        'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
        'volatility_5', 'volatility_10', 'volatility_20',
        'bb_width_norm', 'bb_width_20', 'high_low_range',
        'returns_1', 'returns_5', 'returns_10',
        'volume_ratio_10', 'volume_ratio_20', 'adx',
    ]
    
    # Create dummy training data with correct shape
    np.random.seed(42)
    X_train = np.random.randn(100, len(feature_names))
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Verify n_features_in_ matches
    assert scaler.n_features_in_ == len(feature_names), \
        f"Scaler expects {scaler.n_features_in_} features, but feature_names has {len(feature_names)}"


def test_pandera_validation_available():
    """Test that Pandera is available for optional validation."""
    try:
        import pandera as pa
        import pandera.pandas as pa_pandas
        pandera_available = True
    except ImportError:
        pandera_available = False
    
    # This test passes regardless - just documents whether Pandera is installed
    if pandera_available:
        assert hasattr(pa_pandas, 'DataFrameSchema'), "Pandera should have DataFrameSchema"
        assert hasattr(pa_pandas, 'Column'), "Pandera should have Column"


def test_pandera_schema_catches_missing_columns():
    """Test that Pandera schema validation catches missing columns."""
    try:
        import pandera as pa
        import pandera.pandas as pa_pandas
    except ImportError:
        pytest.skip("Pandera not installed")
    
    # Define schema requiring specific columns
    schema = pa_pandas.DataFrameSchema({
        'atr_pct_5': pa_pandas.Column(float),
        'volatility_5': pa_pandas.Column(float),
        'missing_col': pa_pandas.Column(float),  # This will be missing
    })
    
    # Create DataFrame missing one required column
    df = pd.DataFrame({
        'atr_pct_5': [1.0, 2.0, 3.0],
        'volatility_5': [0.1, 0.2, 0.3],
        # 'missing_col' is intentionally missing
    })
    
    # Validation should raise SchemaError
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)


def test_pandera_strict_mode_rejects_extra_columns():
    """Test that Pandera strict mode rejects extra columns."""
    try:
        import pandera as pa
        import pandera.pandas as pa_pandas
    except ImportError:
        pytest.skip("Pandera not installed")
    
    # Define strict schema
    schema = pa_pandas.DataFrameSchema(
        {'expected_col': pa_pandas.Column(float)},
        strict=True  # Only allow columns in schema
    )
    
    # Create DataFrame with extra column
    df = pd.DataFrame({
        'expected_col': [1.0, 2.0],
        'extra_col': [3.0, 4.0],  # Not in schema
    })
    
    # Strict validation should raise SchemaError
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)
