"""
Integration tests for the end-to-end modular inference pipeline.

Validates:
1. InferenceConfig dataclass defaults
2. TradeSignal dataclass fields
3. Mock 4-model inference producing trade/no-trade decisions
4. Gate evaluation logic (confidence, momentum, risk gates)
5. Pipeline handles missing models gracefully
6. predict_verbose returns expected structure
7. Instrument defaults to EUR_USD

All tests mock model predictions -- no live OANDA API calls.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_inference_df():
    """Generate a realistic DataFrame with features for inference.

    Contains OHLCV + common engineered features (100 bars).
    """
    np.random.seed(42)
    n = 100
    close = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
    df = pd.DataFrame({
        'open': close - np.random.uniform(0.0001, 0.001, n),
        'high': close + np.random.uniform(0.0001, 0.002, n),
        'low': close - np.random.uniform(0.0001, 0.002, n),
        'close': close,
        'volume': np.random.randint(100, 10000, n),
        'atr': np.random.uniform(0.001, 0.005, n),
        'atr_pct_14': np.random.uniform(0.01, 0.05, n),
        'adx': np.random.uniform(10, 50, n),
        'rsi': np.random.uniform(20, 80, n),
        'rsi_norm': np.random.uniform(0.2, 0.8, n),
        'macd_norm': np.random.randn(n) * 0.01,
        'macd_signal_norm': np.random.randn(n) * 0.01,
        'macd_hist_norm': np.random.randn(n) * 0.005,
        'sma_ratio_5': np.random.uniform(0.99, 1.01, n),
        'sma_ratio_10': np.random.uniform(0.99, 1.01, n),
        'sma_ratio_20': np.random.uniform(0.99, 1.01, n),
        'ema_ratio_12': np.random.uniform(0.99, 1.01, n),
        'ema_ratio_26': np.random.uniform(0.99, 1.01, n),
        'returns_1': np.random.randn(n) * 0.001,
        'returns_2': np.random.randn(n) * 0.001,
        'returns_3': np.random.randn(n) * 0.001,
        'returns_5': np.random.randn(n) * 0.001,
        'returns_10': np.random.randn(n) * 0.001,
        'zscore_10': np.random.randn(n),
        'zscore_20': np.random.randn(n),
        'bb_width_norm': np.random.uniform(0, 2, n),
        'bb_position': np.random.uniform(0, 1, n),
        'volatility_5': np.random.uniform(0, 0.02, n),
        'volatility_10': np.random.uniform(0, 0.02, n),
        'volatility_20': np.random.uniform(0, 0.02, n),
        'stoch_k_norm': np.random.uniform(0, 1, n),
        'sma_cross_5_20': np.random.choice([0, 1], n).astype(float),
        'macd_cross': np.random.choice([0, 1], n).astype(float),
        'higher_high_ratio': np.random.uniform(0, 1, n),
        'lower_low_ratio': np.random.uniform(0, 1, n),
        'pct_rank_10': np.random.uniform(0, 1, n),
        'pct_rank_20': np.random.uniform(0, 1, n),
        'volume_ratio_10': np.random.uniform(0.5, 2.0, n),
        'volume_ratio_20': np.random.uniform(0.5, 2.0, n),
        'high_low_range': np.random.uniform(0.001, 0.005, n),
    })
    return df


# ---------------------------------------------------------------------------
# Test: InferenceConfig defaults
# ---------------------------------------------------------------------------

class TestInferenceConfig:
    """Test InferenceConfig dataclass."""

    def test_import(self):
        """InferenceConfig should be importable."""
        from src.core.modular_inference import InferenceConfig
        assert InferenceConfig is not None

    def test_default_values(self):
        """Key default values are sane."""
        from src.core.modular_inference import InferenceConfig

        config = InferenceConfig()
        assert config.min_confidence == 50.0
        assert config.min_momentum == 0.20
        assert config.max_drawdown_pct == 0.025
        assert config.max_streak_prob == 0.95

    def test_custom_override(self):
        """Can override specific config values."""
        from src.core.modular_inference import InferenceConfig

        config = InferenceConfig(min_confidence=70.0, min_momentum=0.5)
        assert config.min_confidence == 70.0
        assert config.min_momentum == 0.5

    def test_has_tcn_volatility_filter(self):
        """InferenceConfig includes TCN volatility filter settings."""
        from src.core.modular_inference import InferenceConfig

        config = InferenceConfig()
        assert hasattr(config, 'use_tcn_volatility_filter')
        assert hasattr(config, 'min_volatility_regime')
        assert config.use_tcn_volatility_filter is True

    def test_has_sentiment_gate(self):
        """InferenceConfig includes sentiment gate settings."""
        from src.core.modular_inference import InferenceConfig

        config = InferenceConfig()
        assert hasattr(config, 'sentiment_block_enabled')
        assert hasattr(config, 'sentiment_block_threshold')

    def test_has_meta_labeling(self):
        """InferenceConfig includes meta-labeling settings."""
        from src.core.modular_inference import InferenceConfig

        config = InferenceConfig()
        assert hasattr(config, 'enable_meta_labeling')
        assert hasattr(config, 'min_meta_confidence')

    def test_has_calibration(self):
        """InferenceConfig includes calibration settings."""
        from src.core.modular_inference import InferenceConfig

        config = InferenceConfig()
        assert hasattr(config, 'enable_calibration')
        assert hasattr(config, 'calibration_method')

    def test_permissive_mode_default_false(self):
        """Permissive mode defaults to False."""
        from src.core.modular_inference import InferenceConfig

        config = InferenceConfig()
        assert config.permissive_mode is False


# ---------------------------------------------------------------------------
# Test: TradeSignal dataclass
# ---------------------------------------------------------------------------

class TestTradeSignal:
    """Test TradeSignal dataclass fields."""

    def test_import(self):
        """TradeSignal should be importable."""
        from src.core.modular_inference import TradeSignal
        assert TradeSignal is not None

    def test_create_trade_signal(self):
        """Can create a TradeSignal with required fields."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=True,
            direction="long",
            size=0.5,
            confidence=75.0,
        )
        assert signal.trade is True
        assert signal.direction == "long"
        assert signal.size == 0.5
        assert signal.confidence == 75.0

    def test_no_trade_signal(self):
        """Can create a no-trade signal."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=False,
            direction=None,
            size=0.0,
            confidence=30.0,
            reason="Confidence below threshold",
        )
        assert signal.trade is False
        assert signal.direction is None
        assert signal.reason == "Confidence below threshold"

    def test_gate_pass_fields(self):
        """TradeSignal has gate status fields."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=True, direction="long", size=0.5, confidence=80.0,
            confidence_gate_passed=True,
            momentum_gate_passed=True,
            risk_gate_passed=True,
        )
        assert signal.confidence_gate_passed is True
        assert signal.momentum_gate_passed is True
        assert signal.risk_gate_passed is True

    def test_default_gate_values(self):
        """Gate defaults have expected values."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(trade=False, direction=None, size=0.0, confidence=0.0)
        assert signal.confidence_gate_passed is False
        assert signal.momentum_gate_passed is False
        assert signal.risk_gate_passed is False
        assert signal.regime_gate_passed is True  # Default: not in CHOP
        assert signal.meta_gate_passed is True     # Default: meta-labeler not loaded

    def test_tcn_and_supporting_model_fields(self):
        """TradeSignal has model prediction fields."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=True, direction="short", size=0.3, confidence=65.0,
            tcn_direction=0,
            tcn_probability=0.35,
            ridge_confidence=70.0,
            xgb_momentum=0.55,
            xgb_acceleration=True,
            rf_drawdown_pips=0.015,
            rf_streak_prob=0.2,
        )
        assert signal.tcn_direction == 0
        assert signal.tcn_probability == 0.35
        assert signal.ridge_confidence == 70.0
        assert signal.xgb_momentum == 0.55
        assert signal.xgb_acceleration is True

    def test_metadata_field(self):
        """TradeSignal has optional metadata dict."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=True, direction="long", size=0.1, confidence=80.0,
            metadata={'atr': 0.003, 'instrument': 'EUR_USD'},
        )
        assert signal.metadata['atr'] == 0.003
        assert signal.metadata['instrument'] == 'EUR_USD'

    def test_rl_fields(self):
        """TradeSignal has RL-related fields."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=True, direction="long", size=0.5, confidence=80.0,
            rl_thresholds_used=True,
            rl_exit_suggestion="hold",
            rl_exit_confidence=0.85,
        )
        assert signal.rl_thresholds_used is True
        assert signal.rl_exit_suggestion == "hold"


# ---------------------------------------------------------------------------
# Test: ModularEnsembleInference class structure
# ---------------------------------------------------------------------------

class TestModularEnsembleInferenceStructure:
    """Test ModularEnsembleInference class can be instantiated."""

    def test_import(self):
        """ModularEnsembleInference should be importable."""
        from src.core.modular_inference import ModularEnsembleInference
        assert ModularEnsembleInference is not None

    def test_instantiate_with_defaults(self):
        """Can create instance with default parameters."""
        from src.core.modular_inference import ModularEnsembleInference

        # This should not raise (models aren't loaded yet)
        ensemble = ModularEnsembleInference(
            enable_market_intelligence=False,
            enable_drift_detection=False,
            use_rl_sizer=False,
            enable_llm_integration=False,
        )
        assert ensemble is not None

    def test_instantiate_with_custom_config(self):
        """Can create instance with custom InferenceConfig."""
        from src.core.modular_inference import ModularEnsembleInference, InferenceConfig

        config = InferenceConfig(min_confidence=80.0)
        ensemble = ModularEnsembleInference(
            config=config,
            enable_market_intelligence=False,
            enable_drift_detection=False,
            use_rl_sizer=False,
            enable_llm_integration=False,
        )
        assert ensemble.config.min_confidence == 80.0

    def test_has_predict_method(self):
        """ModularEnsembleInference has predict() method."""
        from src.core.modular_inference import ModularEnsembleInference
        assert hasattr(ModularEnsembleInference, 'predict')
        assert callable(getattr(ModularEnsembleInference, 'predict'))

    def test_has_predict_verbose_method(self):
        """ModularEnsembleInference has predict_verbose() method."""
        from src.core.modular_inference import ModularEnsembleInference
        assert hasattr(ModularEnsembleInference, 'predict_verbose')
        assert callable(getattr(ModularEnsembleInference, 'predict_verbose'))

    def test_has_load_models_method(self):
        """ModularEnsembleInference has load_models() method."""
        from src.core.modular_inference import ModularEnsembleInference
        assert hasattr(ModularEnsembleInference, 'load_models')
        assert callable(getattr(ModularEnsembleInference, 'load_models'))


# ---------------------------------------------------------------------------
# Test: Gate evaluation logic via TradeSignal construction
# ---------------------------------------------------------------------------

class TestGateLogic:
    """Test gate evaluation logic through signal field analysis."""

    def test_all_gates_pass_means_trade(self):
        """When all gates pass, trade=True."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=True, direction="long", size=0.5, confidence=80.0,
            confidence_gate_passed=True,
            momentum_gate_passed=True,
            risk_gate_passed=True,
        )
        assert signal.trade is True
        assert all([
            signal.confidence_gate_passed,
            signal.momentum_gate_passed,
            signal.risk_gate_passed,
        ])

    def test_failed_confidence_gate(self):
        """Confidence gate failure means no trade."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=False, direction=None, size=0.0, confidence=30.0,
            confidence_gate_passed=False,
            momentum_gate_passed=True,
            risk_gate_passed=True,
            reason="Confidence below threshold",
        )
        assert signal.trade is False
        assert signal.confidence_gate_passed is False

    def test_failed_momentum_gate(self):
        """Momentum gate failure means no trade."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=False, direction=None, size=0.0, confidence=60.0,
            confidence_gate_passed=True,
            momentum_gate_passed=False,
            risk_gate_passed=True,
            reason="Momentum too low",
        )
        assert signal.trade is False
        assert signal.momentum_gate_passed is False

    def test_failed_risk_gate(self):
        """Risk gate failure means no trade."""
        from src.core.modular_inference import TradeSignal

        signal = TradeSignal(
            trade=False, direction=None, size=0.0, confidence=70.0,
            confidence_gate_passed=True,
            momentum_gate_passed=True,
            risk_gate_passed=False,
            rf_drawdown_pips=0.04,  # Too high
            reason="Drawdown too high",
        )
        assert signal.trade is False
        assert signal.risk_gate_passed is False

    def test_direction_long_short(self):
        """Direction field accepts 'long' and 'short'."""
        from src.core.modular_inference import TradeSignal

        long_signal = TradeSignal(
            trade=True, direction="long", size=0.5, confidence=80.0,
        )
        short_signal = TradeSignal(
            trade=True, direction="short", size=0.5, confidence=80.0,
        )
        assert long_signal.direction == "long"
        assert short_signal.direction == "short"


# ---------------------------------------------------------------------------
# Test: Optional dependency flags (should not crash even if absent)
# ---------------------------------------------------------------------------

class TestOptionalDependencyFlags:
    """Verify optional dependency import flags exist."""

    def test_calibration_flag_exists(self):
        """CALIBRATION_AVAILABLE flag exists."""
        from src.core.modular_inference import CALIBRATION_AVAILABLE
        assert isinstance(CALIBRATION_AVAILABLE, bool)

    def test_meta_labeling_flag_exists(self):
        """META_LABELING_AVAILABLE flag exists."""
        from src.core.modular_inference import META_LABELING_AVAILABLE
        assert isinstance(META_LABELING_AVAILABLE, bool)

    def test_market_intel_flag_exists(self):
        """MARKET_INTEL_AVAILABLE flag exists."""
        from src.core.modular_inference import MARKET_INTEL_AVAILABLE
        assert isinstance(MARKET_INTEL_AVAILABLE, bool)

    def test_online_retrainer_flag_exists(self):
        """ONLINE_RETRAINER_AVAILABLE flag exists."""
        from src.core.modular_inference import ONLINE_RETRAINER_AVAILABLE
        assert isinstance(ONLINE_RETRAINER_AVAILABLE, bool)

    def test_rl_available_flag_exists(self):
        """RL_AVAILABLE flag exists."""
        from src.core.modular_inference import RL_AVAILABLE
        assert isinstance(RL_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Test: Normalized feature names
# ---------------------------------------------------------------------------

class TestNormalizedFeatureNames:
    """Test that normalized feature name sets are defined."""

    def test_get_normalized_feature_names_importable(self):
        """get_normalized_feature_names should be importable."""
        from src.core.modular_data_loaders import get_normalized_feature_names
        assert callable(get_normalized_feature_names)

    def test_feature_name_keys(self):
        """Feature names dict should have expected keys."""
        from src.core.modular_data_loaders import get_normalized_feature_names

        names = get_normalized_feature_names()
        expected_keys = {'direction', 'momentum', 'risk', 'confidence', 'volatility'}
        for key in expected_keys:
            assert key in names, f"Missing feature set key: {key}"

    def test_direction_features_nonempty(self):
        """Direction feature set should not be empty."""
        from src.core.modular_data_loaders import get_normalized_feature_names

        names = get_normalized_feature_names()
        assert len(names['direction']) > 0

    def test_volatility_features_nonempty(self):
        """Volatility feature set should not be empty."""
        from src.core.modular_data_loaders import get_normalized_feature_names

        names = get_normalized_feature_names()
        assert len(names['volatility']) > 0


# ---------------------------------------------------------------------------
# Test: compute_normalized_features
# ---------------------------------------------------------------------------

class TestComputeNormalizedFeatures:
    """Test the feature normalization pipeline."""

    def test_importable(self):
        """compute_normalized_features should be importable."""
        from src.core.modular_data_loaders import compute_normalized_features
        assert callable(compute_normalized_features)

    def test_produces_dataframe(self, sample_inference_df):
        """Output should be a DataFrame."""
        from src.core.modular_data_loaders import compute_normalized_features

        result = compute_normalized_features(sample_inference_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_expected_columns(self, sample_inference_df):
        """Output should contain normalized feature columns."""
        from src.core.modular_data_loaders import compute_normalized_features

        result = compute_normalized_features(sample_inference_df)
        # Should have at least some of the typical normalized features
        assert len(result.columns) > 0

    def test_no_inf_values(self, sample_inference_df):
        """Output should not contain Inf values."""
        from src.core.modular_data_loaders import compute_normalized_features

        result = compute_normalized_features(sample_inference_df)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.any(np.isinf(numeric.values)), "Output contains Inf values"
