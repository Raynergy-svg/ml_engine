"""
Integration tests for RL feature computation.

Validates:
1. recent_drawdown computed from close prices (not hardcoded 0.0)
2. BB position in [0, 1] range
3. returns_2 is a rolling sum (not zeros)
4. Curriculum scheduler advancement logic
5. Regime detection from synthetic features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_up_prices():
    """Generate a clear uptrend followed by a pullback for drawdown testing."""
    np.random.seed(42)
    # 100 bars uptrend + 20 bars pullback
    up = np.cumsum(np.abs(np.random.randn(100) * 0.001)) + 1.1
    peak = up[-1]
    pullback = peak - np.cumsum(np.abs(np.random.randn(20) * 0.002))
    prices = np.concatenate([up, pullback])
    return prices


@pytest.fixture
def sample_market_df(trending_up_prices):
    """DataFrame with close prices suitable for extract_gate_features."""
    return pd.DataFrame({
        'close': trending_up_prices,
        'adx': np.linspace(15, 35, len(trending_up_prices)),
        'atr': np.random.uniform(0.001, 0.005, len(trending_up_prices)),
    })


@pytest.fixture
def easy_curriculum_params():
    """Curriculum params for the EASY difficulty level."""
    from src.rl.curriculum import CurriculumScheduler
    scheduler = CurriculumScheduler()
    return scheduler.get_current_params()


# ---------------------------------------------------------------------------
# Test: recent_drawdown from extract_gate_features
# ---------------------------------------------------------------------------

class TestRecentDrawdown:
    """Verify that recent_drawdown is computed from close prices, not hardcoded."""

    def test_drawdown_nonzero_on_pullback(self, sample_market_df):
        """After an uptrend + pullback, recent_drawdown must be > 0."""
        from src.rl.utils import extract_gate_features

        features = extract_gate_features(sample_market_df, lookback=20)
        dd = features['recent_drawdown']

        # The last 20 bars include the pullback, so drawdown must be positive
        assert dd > 0.0, (
            f"recent_drawdown should be > 0 on a pullback, got {dd}"
        )

    def test_drawdown_zero_on_perfect_uptrend(self):
        """A monotonically increasing series has zero drawdown."""
        from src.rl.utils import extract_gate_features

        prices = np.linspace(1.0, 1.5, 50)
        df = pd.DataFrame({'close': prices})
        features = extract_gate_features(df, lookback=20)
        dd = features['recent_drawdown']

        assert dd == pytest.approx(0.0, abs=1e-12), (
            f"Perfect uptrend should have 0 drawdown, got {dd}"
        )

    def test_drawdown_bounded_between_0_and_1(self, sample_market_df):
        """Drawdown fraction must be in [0, 1]."""
        from src.rl.utils import extract_gate_features

        features = extract_gate_features(sample_market_df, lookback=20)
        dd = features['recent_drawdown']

        assert 0.0 <= dd <= 1.0, f"Drawdown should be in [0,1], got {dd}"

    def test_drawdown_default_when_insufficient_data(self):
        """With fewer bars than lookback, drawdown defaults to 0.0."""
        from src.rl.utils import extract_gate_features

        df = pd.DataFrame({'close': [1.0, 1.1, 1.05]})
        features = extract_gate_features(df, lookback=20)
        dd = features['recent_drawdown']

        assert dd == 0.0, f"Expected default 0.0 with insufficient data, got {dd}"

    def test_drawdown_uses_cummax_logic(self):
        """Verify drawdown matches manual cummax calculation."""
        from src.rl.utils import extract_gate_features

        # Deliberate pattern: rise to 100, drop to 90 -> 10% drawdown
        prices = list(range(80, 101)) + [95, 90]
        df = pd.DataFrame({'close': prices})
        features = extract_gate_features(df, lookback=len(prices))
        dd = features['recent_drawdown']

        # Peak=100, trough=90 -> dd = (100-90)/100 = 0.10
        assert dd == pytest.approx(0.10, abs=0.001), (
            f"Expected ~0.10 drawdown, got {dd}"
        )


# ---------------------------------------------------------------------------
# Test: Bollinger Band position from curriculum data
# ---------------------------------------------------------------------------

class TestBBPosition:
    """Verify BB position is properly normalized to [0, 1]."""

    def test_bb_position_in_range(self, easy_curriculum_params):
        """All BB position values should be in [0, 1]."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=500,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        # BB position is column index 6 in the features array
        bb_col = data['features'][:, 6]

        assert np.all(bb_col >= 0.0), "BB position has values < 0"
        assert np.all(bb_col <= 1.0), "BB position has values > 1"

    def test_bb_position_not_all_same(self, easy_curriculum_params):
        """BB position should vary across bars (not stuck at 0.5)."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=500,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        bb_col = data['features'][:, 6]

        # After the first 20 bars (warm-up), values should vary
        bb_active = bb_col[20:]
        assert np.std(bb_active) > 0.01, (
            f"BB position std too low ({np.std(bb_active):.6f}), may be stuck at default"
        )

    def test_bb_position_defaults_are_05_for_warmup(self, easy_curriculum_params):
        """First bb_window (20) bars should default to 0.5."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=500,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        bb_col = data['features'][:, 6]

        # First 20 bars should be the default 0.5
        np.testing.assert_array_almost_equal(
            bb_col[:20], 0.5,
            err_msg="First 20 bars of BB position should be default 0.5"
        )


# ---------------------------------------------------------------------------
# Test: returns_2 rolling sum
# ---------------------------------------------------------------------------

class TestReturns2:
    """Verify returns_2 is a rolling sum over 2 periods, not zeros."""

    def test_returns_2_not_all_zeros(self, easy_curriculum_params):
        """returns_2 column should contain non-zero values."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=500,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        # returns_2 is column index 1 in the features array
        returns_2 = data['features'][:, 1]

        # After 2 warm-up bars, returns_2 should have non-zero values
        active = returns_2[2:]
        non_zero_count = np.count_nonzero(active)
        assert non_zero_count > len(active) * 0.5, (
            f"returns_2 has too many zeros: {non_zero_count}/{len(active)} non-zero"
        )

    def test_returns_2_is_sum_of_consecutive_returns(self, easy_curriculum_params):
        """returns_2[i] should equal returns[i] + returns[i-1]."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=200,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        returns_col = data['features'][:, 0]   # returns
        returns_2_col = data['features'][:, 1]  # returns_2

        # Check from index 2 onward
        for i in range(2, 50):
            expected = returns_col[i] + returns_col[i - 1]
            actual = returns_2_col[i]
            assert actual == pytest.approx(expected, abs=1e-12), (
                f"returns_2[{i}] = {actual}, expected {expected} "
                f"(returns[{i}]={returns_col[i]}, returns[{i-1}]={returns_col[i-1]})"
            )

    def test_returns_2_first_two_are_zero(self, easy_curriculum_params):
        """First 2 bars of returns_2 should be zero (no lookback available)."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=200,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        returns_2_col = data['features'][:, 1]

        assert returns_2_col[0] == 0.0, "returns_2[0] should be 0.0"
        assert returns_2_col[1] == 0.0, "returns_2[1] should be 0.0"


# ---------------------------------------------------------------------------
# Test: generate_curriculum_data output structure
# ---------------------------------------------------------------------------

class TestCurriculumDataStructure:
    """Verify generate_curriculum_data returns correct structure."""

    def test_output_keys(self, easy_curriculum_params):
        """Output dict must have all expected keys."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=100,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )

        expected_keys = {
            'prices', 'features', 'ensemble_preds', 'momentum',
            'atr', 'directions', 'regimes', 'curriculum_level',
        }
        assert set(data.keys()) == expected_keys

    def test_features_shape(self, easy_curriculum_params):
        """Features array should have 7 columns."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=100,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        assert data['features'].shape == (100, 7), (
            f"Expected (100, 7), got {data['features'].shape}"
        )

    def test_ensemble_preds_shape(self, easy_curriculum_params):
        """Ensemble predictions should have 4 columns (tcn, ridge, xgb, rf)."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=100,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        assert data['ensemble_preds'].shape == (100, 4), (
            f"Expected (100, 4), got {data['ensemble_preds'].shape}"
        )

    def test_no_nan_or_inf_in_features(self, easy_curriculum_params):
        """Feature arrays must not contain NaN or Inf."""
        from src.rl.curriculum import generate_curriculum_data

        data = generate_curriculum_data(
            n_samples=500,
            curriculum_params=easy_curriculum_params,
            seed=42,
        )
        assert np.all(np.isfinite(data['features'])), "features contains NaN or Inf"
        assert np.all(np.isfinite(data['ensemble_preds'])), "ensemble_preds contains NaN or Inf"
        assert np.all(np.isfinite(data['prices'])), "prices contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test: CurriculumScheduler advancement
# ---------------------------------------------------------------------------

class TestCurriculumScheduler:
    """Test curriculum difficulty advancement logic."""

    def test_starts_at_easy(self):
        """Scheduler should start at EASY level."""
        from src.rl.curriculum import CurriculumScheduler, DifficultyLevel

        scheduler = CurriculumScheduler()
        assert scheduler.current_level == DifficultyLevel.EASY

    def test_does_not_advance_before_min_episodes(self):
        """Should not advance before min_episodes_per_level (default 50)."""
        from src.rl.curriculum import CurriculumScheduler, DifficultyLevel

        scheduler = CurriculumScheduler()

        # Record 49 high-reward episodes (below min_episodes_per_level=50)
        for _ in range(49):
            scheduler.record_episode(reward=0.9, max_possible_reward=1.0)

        assert scheduler.current_level == DifficultyLevel.EASY

    def test_advances_after_sustained_performance(self):
        """Should advance after min episodes with good performance."""
        from src.rl.curriculum import CurriculumScheduler, DifficultyLevel

        # Use low min_episodes for faster test
        from src.rl.curriculum import CurriculumConfig
        config = CurriculumConfig(min_episodes_per_level=10, advancement_window=5)
        scheduler = CurriculumScheduler(config=config)

        for _ in range(15):
            scheduler.record_episode(reward=0.8, max_possible_reward=1.0)

        assert scheduler.current_level == DifficultyLevel.MEDIUM

    def test_does_not_advance_on_poor_performance(self):
        """Should stay at current level with low rewards."""
        from src.rl.curriculum import CurriculumScheduler, DifficultyLevel, CurriculumConfig

        config = CurriculumConfig(min_episodes_per_level=10, advancement_window=5)
        scheduler = CurriculumScheduler(config=config)

        for _ in range(50):
            scheduler.record_episode(reward=0.2, max_possible_reward=1.0)

        assert scheduler.current_level == DifficultyLevel.EASY

    def test_cannot_exceed_max_level(self):
        """Should cap at EXPERT level."""
        from src.rl.curriculum import CurriculumScheduler, DifficultyLevel, CurriculumConfig

        config = CurriculumConfig(
            min_episodes_per_level=5,
            advancement_window=5,
            min_reward_threshold=0.5,
        )
        scheduler = CurriculumScheduler(config=config)

        # Record many high-reward episodes to advance through all levels
        for _ in range(200):
            scheduler.record_episode(reward=0.9, max_possible_reward=1.0)

        assert scheduler.current_level == DifficultyLevel.EXPERT

    def test_get_stats(self):
        """get_stats returns expected keys."""
        from src.rl.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler()
        scheduler.record_episode(0.5)

        stats = scheduler.get_stats()
        assert 'current_level' in stats
        assert 'total_episodes' in stats
        assert stats['total_episodes'] == 1

    def test_reset(self):
        """reset() restores initial state."""
        from src.rl.curriculum import CurriculumScheduler, DifficultyLevel

        scheduler = CurriculumScheduler()
        for _ in range(10):
            scheduler.record_episode(0.5)

        scheduler.reset()
        assert scheduler.current_level == DifficultyLevel.EASY
        assert scheduler.episode_count == 0
        assert len(scheduler.reward_history) == 0


# ---------------------------------------------------------------------------
# Test: detect_regime
# ---------------------------------------------------------------------------

class TestDetectRegime:
    """Test market regime detection."""

    def test_trend_regime(self):
        """High ADX + momentum -> TREND regime."""
        from src.rl.utils import detect_regime

        features = np.array([30.0, 0.5, 0.5])  # adx=30, vol=0.5, mom=0.5
        one_hot, name = detect_regime(features, adx_idx=0, volatility_idx=1, momentum_idx=2)

        assert name == 'TREND'
        np.testing.assert_array_equal(one_hot, [1.0, 0.0, 0.0])

    def test_chop_regime(self):
        """Low ADX + low vol -> CHOP regime."""
        from src.rl.utils import detect_regime

        features = np.array([15.0, 0.1, 0.05])  # adx=15, vol=0.1, mom=0.05
        one_hot, name = detect_regime(features, adx_idx=0, volatility_idx=1, momentum_idx=2)

        assert name == 'CHOP'
        np.testing.assert_array_equal(one_hot, [0.0, 1.0, 0.0])

    def test_mean_revert_regime(self):
        """Neither TREND nor CHOP -> MEAN_REVERT."""
        from src.rl.utils import detect_regime

        features = np.array([22.0, 0.5, 0.1])  # adx=22, vol=0.5, mom=0.1
        one_hot, name = detect_regime(features, adx_idx=0, volatility_idx=1, momentum_idx=2)

        assert name == 'MEAN_REVERT'
        np.testing.assert_array_equal(one_hot, [0.0, 0.0, 1.0])

    def test_insufficient_features_defaults_trend(self):
        """With too few features, defaults to TREND."""
        from src.rl.utils import detect_regime

        features = np.array([30.0])  # Only 1 feature
        one_hot, name = detect_regime(features, adx_idx=0, volatility_idx=1, momentum_idx=2)

        assert name == 'TREND'


# ---------------------------------------------------------------------------
# Test: performance ratio calculations
# ---------------------------------------------------------------------------

class TestPerformanceMetrics:
    """Test Sharpe, Sortino, Calmar calculations."""

    def test_sharpe_positive_returns(self):
        """Positive returns should produce positive Sharpe."""
        from src.rl.utils import calculate_sharpe

        returns = [0.001, 0.002, 0.0015, 0.001, 0.002, 0.0018]
        sharpe = calculate_sharpe(returns)
        assert sharpe > 0, f"Expected positive Sharpe, got {sharpe}"

    def test_sharpe_insufficient_data(self):
        """With < 2 returns, Sharpe should be 0."""
        from src.rl.utils import calculate_sharpe

        assert calculate_sharpe([0.01]) == 0.0
        assert calculate_sharpe([]) == 0.0

    def test_sortino_positive_returns(self):
        """Positive returns should produce positive Sortino."""
        from src.rl.utils import calculate_sortino

        returns = [0.001, 0.002, -0.0005, 0.001, 0.002, -0.0003]
        sortino = calculate_sortino(returns)
        assert sortino > 0, f"Expected positive Sortino, got {sortino}"

    def test_calmar_zero_drawdown(self):
        """Zero drawdown should return 0 Calmar."""
        from src.rl.utils import calculate_calmar

        assert calculate_calmar([0.01, 0.02], max_drawdown=0.0) == 0.0

    def test_normalize_features_clips(self):
        """normalize_features_for_rl clips extreme values."""
        from src.rl.utils import normalize_features_for_rl

        features = np.array([100.0, -100.0, 0.5, np.nan, np.inf])
        normalized = normalize_features_for_rl(features)

        assert np.all(np.isfinite(normalized)), "Output should have no NaN/Inf"
        assert np.all(normalized >= -3), "Clipped values should be >= -3"
        assert np.all(normalized <= 3), "Clipped values should be <= 3"
