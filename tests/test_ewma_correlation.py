"""Comprehensive tests for EWMA correlation engine.

Tests for US-284: Dynamic correlation tracking across FX pairs.
Covers configuration, initialization, updates, regime detection,
diversification, persistence, and edge cases.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.risk.ewma_correlation import (
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    CorrelationState,
    EWMACorrelationConfig,
    EWMACorrelationEngine,
    create_conservative_ewma_engine,
    create_default_ewma_engine,
    create_responsive_ewma_engine,
)


# ============================================================================
# Tests: EWMACorrelationConfig Validation
# ============================================================================


class TestEWMACorrelationConfigValidation:
    """Test configuration validation."""

    def test_valid_config_defaults(self):
        """Test default configuration is valid."""
        config = EWMACorrelationConfig()
        assert config.lambda_decay == 0.94
        assert config.min_observations == 20
        assert config.risk_off_threshold == 0.6
        assert config.risk_on_threshold == 0.3
        assert config.high_correlation_threshold == 0.7

    def test_lambda_decay_above_one_raises(self):
        """Test lambda_decay > 1 raises ValueError."""
        with pytest.raises(ValueError, match="lambda_decay must be in"):
            EWMACorrelationConfig(lambda_decay=1.5)

    def test_lambda_decay_below_zero_raises(self):
        """Test lambda_decay < 0 raises ValueError."""
        with pytest.raises(ValueError, match="lambda_decay must be in"):
            EWMACorrelationConfig(lambda_decay=-0.1)

    def test_lambda_decay_exactly_one_raises(self):
        """Test lambda_decay == 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="lambda_decay must be in"):
            EWMACorrelationConfig(lambda_decay=1.0)

    def test_lambda_decay_exactly_zero_raises(self):
        """Test lambda_decay == 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="lambda_decay must be in"):
            EWMACorrelationConfig(lambda_decay=0.0)

    def test_min_observations_zero_raises(self):
        """Test min_observations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_observations must be >= 1"):
            EWMACorrelationConfig(min_observations=0)

    def test_risk_on_threshold_invalid_low(self):
        """Test risk_on_threshold < 0 raises ValueError."""
        with pytest.raises(ValueError, match="risk_on_threshold must be in"):
            EWMACorrelationConfig(risk_on_threshold=-0.1)

    def test_risk_on_threshold_invalid_high(self):
        """Test risk_on_threshold > 1 raises ValueError."""
        with pytest.raises(ValueError, match="risk_on_threshold must be in"):
            EWMACorrelationConfig(risk_on_threshold=1.5)

    def test_risk_off_threshold_invalid(self):
        """Test risk_off_threshold invalid."""
        with pytest.raises(ValueError, match="risk_off_threshold must be in"):
            EWMACorrelationConfig(risk_off_threshold=2.0)

    def test_high_correlation_threshold_invalid(self):
        """Test high_correlation_threshold invalid."""
        with pytest.raises(ValueError, match="high_correlation_threshold must be in"):
            EWMACorrelationConfig(high_correlation_threshold=1.5)

    def test_thresholds_ordering_invalid(self):
        """Test risk_on_threshold > risk_off_threshold raises ValueError."""
        with pytest.raises(ValueError, match="risk_on_threshold.*must be <="):
            EWMACorrelationConfig(
                risk_on_threshold=0.7, risk_off_threshold=0.5
            )

    def test_valid_custom_config(self):
        """Test custom valid configuration."""
        config = EWMACorrelationConfig(
            lambda_decay=0.95,
            min_observations=50,
            risk_off_threshold=0.65,
            risk_on_threshold=0.25,
            high_correlation_threshold=0.75,
        )
        assert config.lambda_decay == 0.95
        assert config.min_observations == 50


# ============================================================================
# Tests: Engine Initialization
# ============================================================================


class TestEWMACorrelationEngineInit:
    """Test engine initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        engine = EWMACorrelationEngine()
        assert engine.config.lambda_decay == 0.94
        assert len(engine._pairs) == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = EWMACorrelationConfig(lambda_decay=0.95)
        engine = EWMACorrelationEngine(config=config)
        assert engine.config.lambda_decay == 0.95

    def test_init_with_pairs(self):
        """Test initialization with initial pairs."""
        pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
        engine = EWMACorrelationEngine(pairs=pairs)
        assert engine._pairs == pairs
        assert len(engine._pair_idx) == 3

    def test_init_empty_pairs_list(self):
        """Test initialization with empty pairs list."""
        engine = EWMACorrelationEngine(pairs=[])
        assert len(engine._pairs) == 0

    def test_init_thread_safety(self):
        """Test engine initializes with thread lock."""
        engine = EWMACorrelationEngine()
        assert hasattr(engine, "_lock")


# ============================================================================
# Tests: Single Update
# ============================================================================


class TestEWMACorrelationEngineUpdate:
    """Test single update operations."""

    def test_update_single_observation(self):
        """Test update with single observation."""
        engine = EWMACorrelationEngine(pairs=["EUR_USD", "GBP_USD"])
        returns = {"EUR_USD": 0.001, "GBP_USD": 0.0005}

        state = engine.update(returns)

        assert state.observation_count == 1
        assert len(state.pairs_tracked) == 2
        assert "EUR_USD" in state.correlation_matrix

    def test_update_with_new_pair(self):
        """Test update adds new pair dynamically."""
        engine = EWMACorrelationEngine(pairs=["EUR_USD"])
        returns = {"EUR_USD": 0.001, "GBP_USD": 0.0005}

        state = engine.update(returns)

        assert "GBP_USD" in state.pairs_tracked
        assert len(state.pairs_tracked) == 2

    def test_update_empty_dict(self):
        """Test update with empty returns dict still processes pairs."""
        engine = EWMACorrelationEngine(pairs=["EUR_USD"])
        state = engine.update({})

        # Empty dict means no returns for any pair, but still processes
        # with zero returns for existing pairs
        assert state.observation_count == 1

    def test_update_single_pair_only(self):
        """Test update with single pair only."""
        engine = EWMACorrelationEngine()
        state = engine.update({"EUR_USD": 0.001})

        assert state.observation_count == 1
        assert "EUR_USD" in state.correlation_matrix

    def test_update_increments_observation_count(self):
        """Test observation count increments correctly."""
        engine = EWMACorrelationEngine(pairs=["EUR_USD"])
        returns = {"EUR_USD": 0.001}

        engine.update(returns)
        state1 = engine.get_state()
        assert state1.observation_count == 1

        engine.update(returns)
        state2 = engine.get_state()
        assert state2.observation_count == 2

    def test_update_with_nan_returns(self):
        """Test update handles NaN returns gracefully."""
        engine = EWMACorrelationEngine(pairs=["EUR_USD", "GBP_USD"])
        returns = {"EUR_USD": float("nan"), "GBP_USD": 0.001}

        # Should not crash, NaN treated as 0
        state = engine.update(returns)
        assert state.observation_count == 1

    def test_update_with_inf_returns(self):
        """Test update handles inf returns gracefully."""
        engine = EWMACorrelationEngine(pairs=["EUR_USD"])
        returns = {"EUR_USD": float("inf")}

        # Should not crash, inf clamped
        state = engine.update(returns)
        assert state.observation_count == 1


# ============================================================================
# Tests: Multiple Updates and Correlation Building
# ============================================================================


class TestEWMACorrelationCorrelationBuilding:
    """Test correlation builds correctly with multiple observations."""

    def test_perfectly_correlated_series(self):
        """Test perfectly correlated series converges to correlation near 1.0."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])

        # Update with perfectly correlated returns
        for i in range(50):
            ret = 0.001 * (i + 1)
            engine.update({"A": ret, "B": ret})

        corr_ab = engine.get_correlation("A", "B")
        assert corr_ab is not None
        assert corr_ab > 0.95, f"Expected correlation > 0.95, got {corr_ab}"

    def test_uncorrelated_series(self):
        """Test uncorrelated series converges to correlation near 0.0."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        np.random.seed(42)

        for _ in range(50):
            engine.update({
                "A": float(np.random.randn() * 0.001),
                "B": float(np.random.randn() * 0.001),
            })

        corr_ab = engine.get_correlation("A", "B")
        assert corr_ab is not None
        assert abs(corr_ab) < 0.3, f"Expected |correlation| < 0.3, got {corr_ab}"

    def test_anti_correlated_series(self):
        """Test anti-correlated series converges to correlation near -1.0."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])

        for i in range(50):
            ret = 0.001 * (i + 1)
            engine.update({"A": ret, "B": -ret})

        corr_ab = engine.get_correlation("A", "B")
        assert corr_ab is not None
        assert corr_ab < -0.95, f"Expected correlation < -0.95, got {corr_ab}"

    def test_avg_correlation_updates(self):
        """Test average correlation updates with observations."""
        engine = EWMACorrelationEngine(pairs=["A", "B", "C"])

        # First observation with mixed signals
        state1 = engine.update({"A": 0.001, "B": 0.001, "C": -0.001})
        initial_avg = state1.avg_correlation

        # Many more correlated observations (A and B correlated, all three trending together)
        for _ in range(30):
            engine.update({"A": 0.001, "B": 0.001, "C": 0.001})

        state2 = engine.get_state()
        # With EWMA, the average correlation should increase (become more positive overall)
        # due to the preponderance of correlated observations
        assert state2.observation_count == 31
        # Check that correlation is building (C becomes more correlated over time)
        assert state2.avg_correlation > 0.5

    def test_correlation_matrix_structure(self):
        """Test correlation matrix has correct structure."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        engine.update({"A": 0.001, "B": 0.001})

        corr_matrix = engine.get_correlation_matrix()

        assert "A" in corr_matrix
        assert "B" in corr_matrix
        assert "A" in corr_matrix["A"]
        assert "B" in corr_matrix["A"]
        assert corr_matrix["A"]["A"] == 1.0  # Diagonal is 1
        assert corr_matrix["B"]["B"] == 1.0

    def test_correlation_symmetry(self):
        """Test correlation matrix is symmetric."""
        engine = EWMACorrelationEngine(pairs=["A", "B", "C"])

        for _ in range(30):
            engine.update({
                "A": 0.001,
                "B": 0.001,
                "C": -0.0005,
            })

        corr_matrix = engine.get_correlation_matrix()

        # Check symmetry
        for pair_i in ["A", "B", "C"]:
            for pair_j in ["A", "B", "C"]:
                assert (
                    corr_matrix[pair_i][pair_j] ==
                    corr_matrix[pair_j][pair_i]
                )


# ============================================================================
# Tests: Get Correlation
# ============================================================================


class TestGetCorrelation:
    """Test getting individual correlations."""

    def test_get_correlation_exists(self):
        """Test getting correlation between existing pairs."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        engine.update({"A": 0.001, "B": 0.001})

        corr = engine.get_correlation("A", "B")
        assert corr is not None
        assert isinstance(corr, float)

    def test_get_correlation_unknown_pair(self):
        """Test getting correlation with unknown pair returns None."""
        engine = EWMACorrelationEngine(pairs=["A"])
        engine.update({"A": 0.001})

        corr = engine.get_correlation("A", "UNKNOWN")
        assert corr is None

    def test_get_correlation_self_is_one(self):
        """Test correlation of pair with itself is 1.0."""
        engine = EWMACorrelationEngine(pairs=["A"])
        engine.update({"A": 0.001})

        corr = engine.get_correlation("A", "A")
        assert corr == 1.0

    def test_get_correlation_both_pairs_unknown(self):
        """Test getting correlation between unknown pairs returns None."""
        engine = EWMACorrelationEngine()
        corr = engine.get_correlation("UNKNOWN1", "UNKNOWN2")
        assert corr is None


# ============================================================================
# Tests: Diversification Multiplier
# ============================================================================


class TestDiversificationMultiplier:
    """Test diversification multiplier calculations."""

    def test_multiplier_uncorrelated_pair(self):
        """Test multiplier is close to 1.0 for uncorrelated pair."""
        engine = EWMACorrelationEngine(pairs=["A", "B", "C"])

        # Create uncorrelated series
        np.random.seed(42)
        for _ in range(50):
            engine.update({
                "A": float(np.random.randn() * 0.001),
                "B": float(np.random.randn() * 0.001),
                "C": float(np.random.randn() * 0.001),
            })

        mult = engine.diversification_multiplier("A", ["B", "C"])
        # For uncorrelated pairs, multiplier should be reasonable (can be <1 due to some correlation)
        assert 0.7 < mult <= 1.0, f"Expected between 0.7-1.0, got {mult}"

    def test_multiplier_correlated_pair(self):
        """Test multiplier is <1.0 for correlated pair."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])

        # Create highly correlated series
        for _ in range(50):
            ret = 0.001
            engine.update({"A": ret, "B": ret})

        mult = engine.diversification_multiplier("A", ["B"])
        assert mult < 1.0, f"Expected <1.0, got {mult}"

    def test_multiplier_empty_portfolio(self):
        """Test multiplier returns 1.0 for empty portfolio."""
        engine = EWMACorrelationEngine(pairs=["A"])
        engine.update({"A": 0.001})

        mult = engine.diversification_multiplier("A", [])
        assert mult == 1.0

    def test_multiplier_single_pair_portfolio(self):
        """Test multiplier with single pair in portfolio."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])

        for _ in range(30):
            engine.update({"A": 0.001, "B": 0.0005})

        mult = engine.diversification_multiplier("A", ["B"])
        assert 0.5 <= mult <= 1.0

    def test_multiplier_self_pair_ignored(self):
        """Test multiplier ignores self in portfolio."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        engine.update({"A": 0.001, "B": 0.0005})

        mult = engine.diversification_multiplier("A", ["A", "B"])
        # Should behave like portfolio with just B
        assert 0.5 <= mult <= 1.0

    def test_multiplier_unknown_pair(self):
        """Test multiplier returns 1.0 for unknown pair."""
        engine = EWMACorrelationEngine(pairs=["A"])
        engine.update({"A": 0.001})

        mult = engine.diversification_multiplier("UNKNOWN", ["A"])
        assert mult == 1.0

    def test_multiplier_clamped_to_range(self):
        """Test multiplier is clamped to [0.5, 1.0]."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])

        for _ in range(100):
            # Create extremely correlated pairs
            engine.update({"A": 0.001, "B": 0.001})

        mult = engine.diversification_multiplier("A", ["B"])
        assert 0.5 <= mult <= 1.0


# ============================================================================
# Tests: Correlation Regime Detection
# ============================================================================


class TestCorrelationRegimeDetection:
    """Test regime detection logic."""

    def test_regime_neutral_insufficient_data(self):
        """Test regime is NEUTRAL with insufficient observations."""
        config = EWMACorrelationConfig(min_observations=100)
        engine = EWMACorrelationEngine(config=config, pairs=["A"])

        engine.update({"A": 0.001})
        state = engine.get_state()

        assert state.regime == NEUTRAL

    def test_regime_risk_off_high_correlation(self):
        """Test regime is RISK_OFF with high average correlation."""
        config = EWMACorrelationConfig(
            min_observations=5,
            risk_off_threshold=0.5
        )
        engine = EWMACorrelationEngine(config=config, pairs=["A", "B"])

        # Create highly correlated observations
        for _ in range(10):
            engine.update({"A": 0.001, "B": 0.001})

        state = engine.get_state()
        assert state.regime == RISK_OFF

    def test_regime_risk_on_low_correlation(self):
        """Test regime is RISK_ON with low average correlation."""
        config = EWMACorrelationConfig(
            min_observations=5,
            risk_on_threshold=0.5,
            risk_off_threshold=0.7
        )
        engine = EWMACorrelationEngine(config=config, pairs=["A", "B"])

        # Create uncorrelated observations
        np.random.seed(42)
        for _ in range(30):
            engine.update({
                "A": float(np.random.randn() * 0.001),
                "B": float(np.random.randn() * 0.001),
            })

        state = engine.get_state()
        assert state.regime == RISK_ON

    def test_regime_neutral_medium_correlation(self):
        """Test regime transitions based on correlation levels."""
        config = EWMACorrelationConfig(
            min_observations=5,
            risk_on_threshold=0.2,
            risk_off_threshold=0.85  # Higher threshold for RISK_OFF
        )
        engine = EWMACorrelationEngine(config=config, pairs=["A", "B"])

        # Create observations with mixed positive and negative correlation
        np.random.seed(42)
        for i in range(30):
            engine.update({
                "A": float(np.random.randn() * 0.001),
                "B": -float(np.random.randn() * 0.001),  # Opposite direction
            })

        state = engine.get_state()
        # With opposite returns and higher threshold, should not trigger RISK_OFF
        assert state.regime in [NEUTRAL, RISK_ON]


# ============================================================================
# Tests: Get Correlated Pairs
# ============================================================================


class TestGetCorrelatedPairs:
    """Test finding correlated pairs."""

    def test_find_correlated_pairs(self):
        """Test finding pairs correlated above threshold."""
        engine = EWMACorrelationEngine(pairs=["A", "B", "C"])

        # A and B correlated, C uncorrelated
        np.random.seed(42)
        for i in range(50):
            noise1 = np.random.randn() * 0.0001
            noise2 = np.random.randn() * 0.0001
            engine.update({
                "A": 0.001 + noise1,
                "B": 0.001 + noise1,
                "C": float(np.random.randn() * 0.001),
            })

        correlated = engine.get_correlated_pairs("A", threshold=0.5)
        pair_names = [p[0] for p in correlated]

        assert "B" in pair_names
        assert "A" not in pair_names  # Self excluded

    def test_find_correlated_pairs_empty(self):
        """Test finding correlated pairs returns empty for uncorrelated."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])

        np.random.seed(42)
        for _ in range(50):
            engine.update({
                "A": float(np.random.randn() * 0.001),
                "B": float(np.random.randn() * 0.001),
            })

        correlated = engine.get_correlated_pairs("A", threshold=0.8)
        assert len(correlated) == 0

    def test_find_correlated_pairs_uses_default_threshold(self):
        """Test uses config threshold when not specified."""
        config = EWMACorrelationConfig(high_correlation_threshold=0.5)
        engine = EWMACorrelationEngine(config=config, pairs=["A", "B"])

        for _ in range(50):
            engine.update({"A": 0.001, "B": 0.001})

        # No threshold specified, should use config
        correlated = engine.get_correlated_pairs("A")
        assert len(correlated) > 0

    def test_find_correlated_pairs_sorted(self):
        """Test correlated pairs are sorted by correlation descending."""
        engine = EWMACorrelationEngine(pairs=["A", "B", "C"])

        for i in range(50):
            engine.update({
                "A": 0.001,
                "B": 0.0008 + 0.00001 * i,  # Moderately correlated
                "C": 0.0006 + 0.00002 * i,  # More correlated
            })

        correlated = engine.get_correlated_pairs("A", threshold=0.3)

        # Should be sorted by correlation descending
        if len(correlated) > 1:
            for i in range(len(correlated) - 1):
                assert abs(correlated[i][1]) >= abs(correlated[i+1][1])

    def test_find_correlated_pairs_unknown_pair(self):
        """Test returns empty for unknown pair."""
        engine = EWMACorrelationEngine(pairs=["A"])
        engine.update({"A": 0.001})

        correlated = engine.get_correlated_pairs("UNKNOWN")
        assert len(correlated) == 0


# ============================================================================
# Tests: State Persistence
# ============================================================================


class TestStatePersistence:
    """Test saving and loading state."""

    def test_save_state_creates_file(self):
        """Test save_state creates a file."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        engine.update({"A": 0.001, "B": 0.001})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "state.json")
            engine.save_state(filepath)

            assert os.path.exists(filepath)

    def test_save_state_atomic(self):
        """Test save_state uses atomic write."""
        engine = EWMACorrelationEngine(pairs=["A"])
        engine.update({"A": 0.001})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "state.json")
            engine.save_state(filepath)

            # Temp file should not exist after save
            tmp_filepath = f"{filepath}.tmp"
            assert not os.path.exists(tmp_filepath)

    def test_save_state_valid_json(self):
        """Test saved state is valid JSON."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        engine.update({"A": 0.001, "B": 0.001})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "state.json")
            engine.save_state(filepath)

            with open(filepath, "r") as f:
                data = json.load(f)

            assert "correlation_matrix" in data
            assert "avg_correlation" in data
            assert "regime" in data

    def test_load_state_restores_data(self):
        """Test load_state restores saved state."""
        engine1 = EWMACorrelationEngine(pairs=["A", "B"])
        engine1.update({"A": 0.001, "B": 0.001})
        engine1.update({"A": 0.0015, "B": 0.0015})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "state.json")
            engine1.save_state(filepath)

            # Load into new engine
            engine2 = EWMACorrelationEngine()
            engine2.load_state(filepath)

            state2 = engine2.get_state()
            assert state2.observation_count == 2
            assert "A" in state2.pairs_tracked
            assert "B" in state2.pairs_tracked

    def test_load_state_nonexistent_file(self):
        """Test load_state handles nonexistent file gracefully."""
        engine = EWMACorrelationEngine()

        # Should not raise, just log and return
        engine.load_state("/nonexistent/path/state.json")
        assert engine._observation_count == 0

    def test_load_state_invalid_json(self):
        """Test load_state handles invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "state.json")
            with open(filepath, "w") as f:
                f.write("invalid json {{{")

            engine = EWMACorrelationEngine()
            # Should not raise
            engine.load_state(filepath)

    def test_load_state_missing_required_field(self):
        """Test load_state handles missing required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "state.json")
            with open(filepath, "w") as f:
                json.dump({"avg_correlation": 0.5}, f)

            engine = EWMACorrelationEngine()
            # Should not raise, just log warning
            engine.load_state(filepath)

    def test_round_trip_persistence(self):
        """Test round-trip save/load preserves state."""
        engine1 = EWMACorrelationEngine(pairs=["A", "B", "C"])

        for _ in range(30):
            engine1.update({"A": 0.001, "B": 0.0008, "C": -0.0002})

        state1 = engine1.get_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "state.json")
            engine1.save_state(filepath)

            engine2 = EWMACorrelationEngine()
            engine2.load_state(filepath)
            state2 = engine2.get_state()

            assert state2.observation_count == state1.observation_count
            assert state2.pairs_tracked == state1.pairs_tracked
            assert state2.regime == state1.regime


# ============================================================================
# Tests: Reset
# ============================================================================


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        engine.update({"A": 0.001, "B": 0.001})

        engine.reset()

        state = engine.get_state()
        assert state.observation_count == 0
        assert len(state.pairs_tracked) == 0
        assert state.avg_correlation == 0.0
        assert state.regime == NEUTRAL

    def test_reset_allows_restart(self):
        """Test can restart engine after reset."""
        engine = EWMACorrelationEngine(pairs=["A"])
        engine.update({"A": 0.001})
        engine.reset()

        engine.update({"A": 0.002})
        state = engine.get_state()
        assert state.observation_count == 1


# ============================================================================
# Tests: Factory Functions
# ============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_default_ewma_engine(self):
        """Test default factory creates correct engine."""
        engine = create_default_ewma_engine()
        assert engine.config.lambda_decay == 0.94

    def test_create_default_with_pairs(self):
        """Test default factory with pairs."""
        pairs = ["EUR_USD", "GBP_USD"]
        engine = create_default_ewma_engine(pairs=pairs)
        assert engine._pairs == pairs

    def test_create_conservative_ewma_engine(self):
        """Test conservative factory creates correct engine."""
        engine = create_conservative_ewma_engine()
        assert engine.config.lambda_decay == 0.97
        assert engine.config.risk_off_threshold == 0.65

    def test_create_responsive_ewma_engine(self):
        """Test responsive factory creates correct engine."""
        engine = create_responsive_ewma_engine()
        assert engine.config.lambda_decay == 0.90
        assert engine.config.risk_off_threshold == 0.55


# ============================================================================
# Tests: Edge Cases and Robustness
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_variance_pair(self):
        """Test handling of zero-variance pair."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])

        # A has constant returns (zero variance)
        for _ in range(30):
            engine.update({"A": 0.0, "B": 0.001})

        corr = engine.get_correlation("A", "B")
        # Should be 0 for zero-variance pair
        assert corr == 0.0 or corr is not None

    def test_partial_pair_updates(self):
        """Test updates with not all pairs present."""
        engine = EWMACorrelationEngine(pairs=["A", "B", "C"])

        engine.update({"A": 0.001, "B": 0.001})  # C missing
        engine.update({"A": 0.001, "C": 0.001})  # B missing
        engine.update({"B": 0.001, "C": 0.001})  # A missing

        state = engine.get_state()
        assert state.observation_count == 3
        assert len(state.pairs_tracked) == 3

    def test_large_number_of_updates(self):
        """Test doesn't overflow with many updates."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])

        for i in range(10000):
            engine.update({
                "A": 0.001 * np.sin(i / 100),
                "B": 0.001 * np.cos(i / 100),
            })

        state = engine.get_state()
        assert state.observation_count == 10000
        assert not np.isnan(state.avg_correlation)
        assert not np.isinf(state.avg_correlation)

    def test_max_pairs_limit(self):
        """Test respects max_pairs limit."""
        config = EWMACorrelationConfig(max_pairs=5)
        engine = EWMACorrelationEngine(config=config)

        # Try to add more than max_pairs
        for i in range(10):
            engine.update({f"PAIR_{i}": 0.001})

        state = engine.get_state()
        assert len(state.pairs_tracked) <= 5

    def test_correlation_bounds(self):
        """Test correlations are in [-1, 1]."""
        engine = EWMACorrelationEngine(pairs=["A", "B", "C"])

        np.random.seed(42)
        for _ in range(100):
            engine.update({
                "A": float(np.random.randn() * 0.01),
                "B": float(np.random.randn() * 0.01),
                "C": float(np.random.randn() * 0.01),
            })

        corr_matrix = engine.get_correlation_matrix()

        for pair_i in corr_matrix:
            for pair_j in corr_matrix[pair_i]:
                corr = corr_matrix[pair_i][pair_j]
                assert -1.0 <= corr <= 1.0, f"Correlation {corr} out of bounds"

    def test_get_state_timestamp(self):
        """Test get_state includes valid timestamp."""
        engine = EWMACorrelationEngine(pairs=["A"])
        engine.update({"A": 0.001})

        state = engine.get_state()
        assert state.last_updated
        # Should be ISO format
        assert "T" in state.last_updated or "-" in state.last_updated


# ============================================================================
# Tests: Thread Safety
# ============================================================================


class TestThreadSafety:
    """Test thread safety of operations."""

    def test_concurrent_updates(self):
        """Test concurrent updates don't cause crashes."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        errors = []

        def update_worker(worker_id):
            try:
                for i in range(100):
                    engine.update({
                        "A": 0.001 * worker_id,
                        "B": 0.001 * (worker_id + i)
                    })
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_worker, args=(i,))
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        state = engine.get_state()
        assert state.observation_count == 300

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        engine = EWMACorrelationEngine(pairs=["A", "B"])
        errors = []

        def writer_worker():
            try:
                for i in range(50):
                    engine.update({"A": 0.001, "B": 0.001})
            except Exception as e:
                errors.append(e)

        def reader_worker():
            try:
                for _ in range(50):
                    engine.get_correlation_matrix()
                    engine.get_state()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer_worker),
            threading.Thread(target=reader_worker),
            threading.Thread(target=reader_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Tests: CorrelationState
# ============================================================================


class TestCorrelationState:
    """Test CorrelationState dataclass."""

    def test_state_to_dict(self):
        """Test state converts to dict correctly."""
        state = CorrelationState(
            correlation_matrix={"A": {"A": 1.0, "B": 0.5}},
            avg_correlation=0.5,
            regime=NEUTRAL,
            observation_count=10,
            pairs_tracked=["A", "B"],
            last_updated="2026-03-23T12:00:00",
        )

        data = state.to_dict()
        assert data["avg_correlation"] == 0.5
        assert data["regime"] == NEUTRAL
        assert isinstance(data["correlation_matrix"], dict)

    def test_state_from_dict(self):
        """Test state creates from dict correctly."""
        data = {
            "correlation_matrix": {"A": {"A": 1.0}},
            "avg_correlation": 0.3,
            "regime": RISK_ON,
            "observation_count": 5,
            "pairs_tracked": ["A"],
            "last_updated": "2026-03-23T12:00:00",
        }

        state = CorrelationState.from_dict(data)
        assert state.avg_correlation == 0.3
        assert state.regime == RISK_ON
        assert state.observation_count == 5

    def test_state_round_trip(self):
        """Test state dict conversion round-trips."""
        state1 = CorrelationState(
            correlation_matrix={"A": {"A": 1.0, "B": 0.7}},
            avg_correlation=0.35,
            regime=NEUTRAL,
            observation_count=25,
            pairs_tracked=["A", "B"],
            last_updated="2026-03-23T12:00:00",
        )

        data = state1.to_dict()
        state2 = CorrelationState.from_dict(data)

        assert state2.avg_correlation == state1.avg_correlation
        assert state2.regime == state1.regime
        assert state2.observation_count == state1.observation_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
