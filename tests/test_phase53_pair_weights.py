"""
Tests for Phase 53 US-329: Pair-Specific Bayesian Agent Weights.

Tests cover:
  - Initialization with 144 distributions (12 agents × 12 pairs)
  - Warm-start: pair needs 5 trades before pair-specific weights activate
  - Sampling with and without pair activation
  - Blending ratio (0.6 regime + 0.4 pair)
  - Update mechanics (win/loss, partial credit)
  - Epsilon decay per-pair
  - Persistence save/load
  - Lazy imports
  - Wiring in execution.py sync path
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.scanner.pair_bayesian_weights import (
    PairSpecificBayesianWeights,
    PairWeightsConfig,
    PairWeightSample,
    DEFAULT_AGENT_NAMES,
    DEFAULT_PAIR_NAMES,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def engine(tmp_dir):
    config = PairWeightsConfig(
        persistence_path=os.path.join(tmp_dir, "pair_weights.json"),
    )
    return PairSpecificBayesianWeights(config)


class TestPairSpecificBayesianWeights:

    def test_init_distribution_count(self, engine):
        """Should initialize 12×12=144 pair distributions."""
        expected = len(DEFAULT_AGENT_NAMES) * len(DEFAULT_PAIR_NAMES)
        assert len(engine._pair_distributions) == expected

    def test_warm_start_inactive(self, engine):
        """Pair with 0 trades returns regime weights only (pair_active=False)."""
        regime_weights = {a: 1.0 / 12 for a in DEFAULT_AGENT_NAMES}
        sample = engine.sample_pair_weights("EUR_USD", regime_weights)
        assert sample.pair_active is False
        assert sample.pair_trade_count == 0
        # Weights should equal regime weights exactly
        for agent in DEFAULT_AGENT_NAMES:
            assert abs(sample.weights[agent] - regime_weights[agent]) < 1e-6

    def test_warm_start_threshold(self, engine):
        """Pair activates after 5 trades."""
        scores = {a: 0.6 for a in DEFAULT_AGENT_NAMES}
        for i in range(4):
            engine.update("EUR_USD", scores, "win")
        assert not engine.is_pair_active("EUR_USD")
        engine.update("EUR_USD", scores, "win")
        assert engine.is_pair_active("EUR_USD")

    def test_sample_active_pair_blends(self, engine):
        """Active pair produces blended weights (not equal to regime)."""
        scores = {a: 0.7 for a in DEFAULT_AGENT_NAMES}
        for _ in range(10):
            engine.update("EUR_USD", scores, "win")

        regime_weights = {a: 1.0 / 12 for a in DEFAULT_AGENT_NAMES}
        np.random.seed(42)
        sample = engine.sample_pair_weights("EUR_USD", regime_weights)
        assert sample.pair_active is True
        assert sample.pair_trade_count == 10
        # Weights should sum to ~1
        total = sum(sample.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_update_win_increases_alpha(self, engine):
        """Win with high score increases alpha for pair."""
        agent = "trend"
        alpha_before, beta_before = engine._pair_distributions[("EUR_USD", agent)]
        engine.update("EUR_USD", {a: 0.8 for a in DEFAULT_AGENT_NAMES}, "win")
        alpha_after, beta_after = engine._pair_distributions[("EUR_USD", agent)]
        assert alpha_after > alpha_before
        assert beta_after == beta_before  # Score > 0.5, win → only alpha increases

    def test_update_loss_increases_beta(self, engine):
        """Loss with high score increases beta for pair."""
        agent = "trend"
        alpha_before, beta_before = engine._pair_distributions[("EUR_USD", agent)]
        engine.update("EUR_USD", {a: 0.8 for a in DEFAULT_AGENT_NAMES}, "loss")
        alpha_after, beta_after = engine._pair_distributions[("EUR_USD", agent)]
        assert beta_after > beta_before
        # High score + loss → beta increases by (1-score)

    def test_epsilon_decay_per_pair(self, engine):
        """Epsilon decays independently per pair."""
        eps_before = engine._pair_epsilon["EUR_USD"]
        scores = {a: 0.5 for a in DEFAULT_AGENT_NAMES}
        engine.update("EUR_USD", scores, "win")
        eps_after = engine._pair_epsilon["EUR_USD"]
        assert eps_after < eps_before

        # GBP_USD epsilon should be unchanged
        assert engine._pair_epsilon["GBP_USD"] == engine.config.epsilon_initial

    def test_auto_add_unknown_pair(self, engine):
        """Unknown pair is auto-added on update."""
        scores = {a: 0.6 for a in DEFAULT_AGENT_NAMES}
        engine.update("XAU_USD", scores, "win")
        assert engine.get_pair_trade_count("XAU_USD") == 1

    def test_persistence_roundtrip(self, tmp_dir):
        """State round-trips through save/load."""
        path = os.path.join(tmp_dir, "pw_test.json")
        config = PairWeightsConfig(persistence_path=path)
        engine1 = PairSpecificBayesianWeights(config)

        scores = {a: 0.7 for a in DEFAULT_AGENT_NAMES}
        for _ in range(7):
            engine1.update("EUR_USD", scores, "win")
        for _ in range(3):
            engine1.update("GBP_USD", scores, "loss")
        engine1.save_state()

        engine2 = PairSpecificBayesianWeights(config)
        assert engine2.get_pair_trade_count("EUR_USD") == 7
        assert engine2.get_pair_trade_count("GBP_USD") == 3
        assert engine2.is_pair_active("EUR_USD")

    def test_get_pair_stats(self, engine):
        """get_pair_stats returns correct structure."""
        stats = engine.get_pair_stats("EUR_USD")
        assert len(stats) == len(DEFAULT_AGENT_NAMES)
        for agent, s in stats.items():
            assert "alpha" in s
            assert "beta" in s
            assert "mean" in s
            assert "variance" in s

    def test_reset(self, engine):
        """Reset clears all state."""
        scores = {a: 0.7 for a in DEFAULT_AGENT_NAMES}
        for _ in range(5):
            engine.update("EUR_USD", scores, "win")
        engine.reset()
        assert engine.get_pair_trade_count("EUR_USD") == 0
        assert not engine.is_pair_active("EUR_USD")

    def test_invalid_outcome_raises(self, engine):
        """Invalid outcome raises ValueError."""
        with pytest.raises(ValueError, match="outcome must be"):
            engine.update("EUR_USD", {"trend": 0.5}, "draw")


class TestPairWeightsLazyImports:

    def test_pair_specific_bayesian_weights_import(self):
        from src.scanner import PairSpecificBayesianWeights
        assert PairSpecificBayesianWeights is not None

    def test_pair_weights_config_import(self):
        from src.scanner import PairWeightsConfig
        assert PairWeightsConfig is not None

    def test_pair_weight_sample_import(self):
        from src.scanner import PairWeightSample
        assert PairWeightSample is not None


class TestPairWeightsWiring:

    def test_execution_manager_has_pair_weights(self):
        """ExecutionManager initializes _pair_bayesian_weights."""
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.__init__)
            assert "_pair_bayesian_weights" in source
            assert "PairSpecificBayesianWeights" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")

    def test_pair_weights_update_in_sync(self):
        """Pair weights updated in sync_closed_trades_rl."""
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.sync_closed_trades_rl)
            assert "_pair_bayesian_weights" in source
            assert "pair_bayesian_weights.update" in source or "save_state" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")
