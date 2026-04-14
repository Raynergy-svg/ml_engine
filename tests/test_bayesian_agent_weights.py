"""
Comprehensive test suite for Bayesian agent weight updater.

Tests cover:
- Configuration validation
- Weight sampling (uniform weights, Thompson sampling, epsilon-greedy)
- Regime-conditional distributions
- Bayesian updates (wins/losses, partial credit)
- Weight decay
- Epsilon decay
- Agent statistics
- Top agents ranking
- State persistence (save/load, corrupted files)
- Factory functions
- Convergence properties
- Edge cases
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.scanner.bayesian_agent_weights import (
    BayesianAgentWeights,
    BayesianWeightsConfig,
    WeightSample,
    create_default_bayesian_weights,
    create_exploration_heavy_weights,
    create_exploitation_heavy_weights,
    DEFAULT_AGENT_NAMES,
    DEFAULT_REGIME_NAMES,
)


class TestBayesianWeightsConfig(unittest.TestCase):
    """Test configuration validation."""

    def test_config_defaults(self):
        """Test that config has sensible defaults."""
        config = BayesianWeightsConfig()
        self.assertEqual(len(config.agent_names), 12)
        self.assertEqual(len(config.regime_names), 4)
        self.assertEqual(config.prior_alpha, 2.0)
        self.assertEqual(config.prior_beta, 2.0)
        self.assertEqual(config.decay_factor, 0.995)
        self.assertEqual(config.epsilon_initial, 0.15)

    def test_config_custom_agents(self):
        """Test config with custom agent names."""
        agents = ["agent_a", "agent_b", "agent_c"]
        config = BayesianWeightsConfig(agent_names=agents)
        self.assertEqual(config.agent_names, agents)

    def test_config_custom_regimes(self):
        """Test config with custom regime names."""
        regimes = ["BULL", "BEAR", "SIDEWAYS"]
        config = BayesianWeightsConfig(regime_names=regimes)
        self.assertEqual(config.regime_names, regimes)

    def test_config_validation_empty_agents(self):
        """Test that empty agent_names raises ValueError."""
        with self.assertRaises(ValueError):
            BayesianWeightsConfig(agent_names=[])

    def test_config_validation_empty_regimes(self):
        """Test that empty regime_names raises ValueError."""
        with self.assertRaises(ValueError):
            BayesianWeightsConfig(regime_names=[])

    def test_config_validation_prior_alpha_bounds(self):
        """Test prior_alpha must be in (0, 100)."""
        with self.assertRaises(ValueError):
            BayesianWeightsConfig(prior_alpha=0)
        with self.assertRaises(ValueError):
            BayesianWeightsConfig(prior_alpha=100)

    def test_config_validation_decay_factor_bounds(self):
        """Test decay_factor must be in (0.9, 1.0)."""
        with self.assertRaises(ValueError):
            BayesianWeightsConfig(decay_factor=0.9)
        with self.assertRaises(ValueError):
            BayesianWeightsConfig(decay_factor=1.0)

    def test_config_validation_epsilon_min_le_initial(self):
        """Test epsilon_min must be <= epsilon_initial."""
        with self.assertRaises(ValueError):
            BayesianWeightsConfig(epsilon_initial=0.1, epsilon_min=0.2)

    def test_config_decay_interval_positive(self):
        """Test decay_interval must be >= 1."""
        with self.assertRaises(ValueError):
            BayesianWeightsConfig(decay_interval=0)


class TestWeightSampleValidation(unittest.TestCase):
    """Test WeightSample validation."""

    def test_weights_must_sum_to_one(self):
        """Test that weights summing to != 1.0 raises ValueError."""
        weights = {"a": 0.5, "b": 0.4}  # sum = 0.9
        with self.assertRaises(ValueError):
            WeightSample(
                weights=weights,
                regime="NORMAL",
                exploration=False,
                raw_samples=weights,
                epsilon=0.1,
            )

    def test_valid_weight_sample(self):
        """Test creating valid WeightSample."""
        weights = {"a": 0.5, "b": 0.5}
        sample = WeightSample(
            weights=weights,
            regime="NORMAL",
            exploration=False,
            raw_samples=weights,
            epsilon=0.1,
        )
        self.assertEqual(sample.weights, weights)
        self.assertEqual(sample.regime, "NORMAL")
        self.assertFalse(sample.exploration)


class TestWeightSampling(unittest.TestCase):
    """Test weight sampling functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = create_default_bayesian_weights()

    def test_sample_weights_returns_normalized(self):
        """Test that sampled weights sum to 1.0."""
        sample = self.weights.sample_weights()
        total = sum(sample.weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_sample_weights_all_agents_present(self):
        """Test that all agents are present in sample."""
        sample = self.weights.sample_weights()
        self.assertEqual(set(sample.weights.keys()), set(self.weights.config.agent_names))

    def test_sample_weights_exploration_uniform(self):
        """Test exploration samples return uniform weights."""
        # Force exploration by patching np.random.rand
        with patch("numpy.random.rand", return_value=0.01):  # < epsilon
            sample = self.weights.sample_weights()
            self.assertTrue(sample.exploration)
            expected_uniform = 1.0 / len(self.weights.config.agent_names)
            for weight in sample.weights.values():
                self.assertAlmostEqual(weight, expected_uniform, places=6)

    def test_sample_weights_exploitation(self):
        """Test exploitation samples use Thompson sampling."""
        with patch("numpy.random.rand", return_value=0.5):  # > epsilon
            sample = self.weights.sample_weights()
            self.assertFalse(sample.exploration)
            # Weights should not all be equal (very unlikely if sampling works)
            total = sum(sample.weights.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_sample_weights_invalid_regime(self):
        """Test sampling with invalid regime raises ValueError."""
        with self.assertRaises(ValueError):
            self.weights.sample_weights(regime="INVALID")

    def test_sample_weights_all_regimes(self):
        """Test sampling works for all defined regimes."""
        for regime in self.weights.config.regime_names:
            sample = self.weights.sample_weights(regime=regime)
            self.assertEqual(sample.regime, regime)
            total = sum(sample.weights.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_sample_weights_epsilon_tracked(self):
        """Test epsilon value is tracked in sample."""
        sample = self.weights.sample_weights()
        self.assertEqual(sample.epsilon, self.weights._epsilon)

    def test_load_state_migrates_nested_legacy_schema(self):
        """Legacy nested regime->agent state should be migrated instead of reset."""
        weights = create_default_bayesian_weights()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bayesian_weights.json"
            path.write_text(json.dumps({
                "version": 0,
                "regimes": {
                    "NORMAL": {
                        "trend": {"alpha": 7.0, "beta": 3.0},
                    }
                },
                "epsilon": 0.07,
            }))

            weights.load_state(str(path))

            self.assertEqual(weights._distributions[("NORMAL", "trend")], (7.0, 3.0))
            # Missing entries should be filled from priors, not wiped/reset.
            self.assertEqual(weights._distributions[("LOW", "trend")], (2.0, 2.0))
            self.assertAlmostEqual(weights._epsilon, 0.07)

    def test_load_state_quarantines_bad_schema(self):
        """Unrecoverable schema should be moved aside and reset cleanly."""
        weights = create_default_bayesian_weights()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bayesian_weights.json"
            path.write_text(json.dumps({"version": 1, "foo": "bar"}))

            weights.load_state(str(path))

            backups = list(Path(td).glob("bayesian_weights.json.bad.schema.*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(weights._distributions[("NORMAL", "trend")], (2.0, 2.0))


class TestBayesianUpdates(unittest.TestCase):
    """Test Bayesian update logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = create_default_bayesian_weights()
        self.agents = self.weights.config.agent_names

    def test_update_win_increases_alpha(self):
        """Test win outcome increases alpha for high-scoring agents."""
        scores = {agent: 0.7 for agent in self.agents}
        regime = "NORMAL"

        alpha_before, beta_before = self.weights._distributions[(regime, self.agents[0])]
        self.weights.update(scores, "win", regime)
        alpha_after, beta_after = self.weights._distributions[(regime, self.agents[0])]

        self.assertGreater(alpha_after, alpha_before)

    def test_update_loss_increases_beta(self):
        """Test loss outcome increases beta for high-scoring agents."""
        scores = {agent: 0.7 for agent in self.agents}
        regime = "NORMAL"

        alpha_before, beta_before = self.weights._distributions[(regime, self.agents[0])]
        self.weights.update(scores, "loss", regime)
        alpha_after, beta_after = self.weights._distributions[(regime, self.agents[0])]

        self.assertGreater(beta_after, beta_before)

    def test_update_partial_credit_win(self):
        """Test partial credit: score-proportional update on win."""
        config = BayesianWeightsConfig(partial_credit=True)
        weights = BayesianAgentWeights(config)

        agent_a = self.agents[0]
        agent_b = self.agents[1]

        scores = {agent_a: 0.9, agent_b: 0.1}
        scores.update({a: 0.5 for a in self.agents[2:]})

        weights.update(scores, "win", "NORMAL")

        alpha_a, beta_a = weights._distributions[("NORMAL", agent_a)]
        alpha_b, beta_b = weights._distributions[("NORMAL", agent_b)]

        # agent_a (score=0.9) should have increased alpha by ~0.9
        # agent_b (score=0.1) should have increased beta by ~0.9
        self.assertGreater(alpha_a, 2.0)
        self.assertGreater(beta_b, 2.0)

    def test_update_binary_credit(self):
        """Test binary credit: threshold-based update."""
        config = BayesianWeightsConfig(partial_credit=False)
        weights = BayesianAgentWeights(config)

        agent_a = self.agents[0]
        agent_b = self.agents[1]

        scores = {agent_a: 0.7, agent_b: 0.3}
        scores.update({a: 0.5 for a in self.agents[2:]})

        weights.update(scores, "win", "NORMAL")

        alpha_a, beta_a = weights._distributions[("NORMAL", agent_a)]
        alpha_b, beta_b = weights._distributions[("NORMAL", agent_b)]

        # agent_a (score > 0.5) should have increased alpha by 1
        # agent_b (score < 0.5) should have increased beta by 1
        self.assertEqual(alpha_a, 3.0)  # prior 2 + 1
        self.assertEqual(beta_b, 3.0)  # prior 2 + 1

    def test_update_invalid_outcome(self):
        """Test invalid outcome raises ValueError."""
        scores = {agent: 0.5 for agent in self.agents}
        with self.assertRaises(ValueError):
            self.weights.update(scores, "draw", "NORMAL")

    def test_update_invalid_agent(self):
        """Test unknown agent in scores raises ValueError."""
        scores = {"unknown_agent": 0.5}
        scores.update({agent: 0.5 for agent in self.agents})
        with self.assertRaises(ValueError):
            self.weights.update(scores, "win", "NORMAL")

    def test_update_invalid_regime(self):
        """Test unknown regime raises ValueError."""
        scores = {agent: 0.5 for agent in self.agents}
        with self.assertRaises(ValueError):
            self.weights.update(scores, "win", "INVALID")

    def test_update_invalid_score_range(self):
        """Test score outside [0, 1] raises ValueError."""
        scores = {self.agents[0]: 1.5}
        scores.update({a: 0.5 for a in self.agents[1:]})
        with self.assertRaises(ValueError):
            self.weights.update(scores, "win", "NORMAL")

    def test_update_empty_scores(self):
        """Test update with missing agents (partial scores) fills defaults."""
        # Only provide scores for some agents; rest should default to 0
        scores = {self.agents[0]: 0.7}
        self.weights.update(scores, "win", "NORMAL")
        # Should not raise an error

    def test_update_epsilon_decays(self):
        """Test epsilon decays after each update."""
        epsilon_before = self.weights._epsilon
        scores = {agent: 0.5 for agent in self.agents}
        self.weights.update(scores, "win", "NORMAL")
        epsilon_after = self.weights._epsilon

        self.assertLess(epsilon_after, epsilon_before)

    def test_update_respects_epsilon_min(self):
        """Test epsilon does not decay below epsilon_min."""
        # Perform many updates to drive epsilon down
        scores = {agent: 0.5 for agent in self.agents}
        for _ in range(1000):
            self.weights.update(scores, "win", "NORMAL")

        self.assertGreaterEqual(self.weights._epsilon, self.weights.config.epsilon_min)


class TestWeightDecay(unittest.TestCase):
    """Test weight decay functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = create_default_bayesian_weights()

    def test_decay_reduces_alpha_beta(self):
        """Test decay multiplies alpha and beta by decay_factor."""
        agent = self.weights.config.agent_names[0]
        regime = "NORMAL"

        # Set high alpha/beta
        self.weights._distributions[(regime, agent)] = (100, 80)

        alpha_before, beta_before = 100, 80
        self.weights.apply_decay()
        alpha_after, beta_after = self.weights._distributions[(regime, agent)]

        expected_alpha = 100 * self.weights.config.decay_factor
        expected_beta = 80 * self.weights.config.decay_factor

        self.assertAlmostEqual(alpha_after, expected_alpha, places=5)
        self.assertAlmostEqual(beta_after, expected_beta, places=5)

    def test_decay_respects_minimum(self):
        """Test decay never takes alpha/beta below 1.0."""
        agent = self.weights.config.agent_names[0]
        regime = "NORMAL"

        # Set very low alpha/beta
        self.weights._distributions[(regime, agent)] = (1.0, 1.0)
        self.weights.apply_decay()
        alpha_after, beta_after = self.weights._distributions[(regime, agent)]

        self.assertGreaterEqual(alpha_after, 1.0)
        self.assertGreaterEqual(beta_after, 1.0)

    def test_decay_applied_at_interval(self):
        """Test decay is applied after decay_interval updates."""
        config = BayesianWeightsConfig(decay_interval=3, partial_credit=True)
        weights = BayesianAgentWeights(config)

        # Set high values
        agent = weights.config.agent_names[0]
        weights._distributions[("NORMAL", agent)] = (100, 80)

        scores = {a: 0.5 for a in weights.config.agent_names}

        # Updates 1-2: no decay
        weights.update(scores, "win", "NORMAL")
        alpha1, _ = weights._distributions[("NORMAL", agent)]

        weights.update(scores, "win", "NORMAL")
        # Still no decay yet (update_count = 2)

        # Update 3: decay is applied
        weights.update(scores, "win", "NORMAL")
        # Now decay should have been applied
        alpha, beta = weights._distributions[("NORMAL", agent)]
        # After update 2: both increased with partial credit (0.5 each), then decayed on update 3
        # Just verify decay was applied by checking values aren't pure sums
        # Without decay: 100 + 0.5 + 0.5 + 0.5 = 101.5
        self.assertLess(alpha, 101.5)  # Would be >= 101.5 without decay


class TestAgentStatistics(unittest.TestCase):
    """Test agent statistics retrieval."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = create_default_bayesian_weights()

    def test_get_agent_stats_all_regimes(self):
        """Test getting stats aggregated across all regimes."""
        stats = self.weights.get_agent_stats()

        # Should have stats for all agents
        self.assertEqual(len(stats), len(self.weights.config.agent_names))

        # Each agent should have required fields
        for agent, agent_stats in stats.items():
            self.assertIn("alpha", agent_stats)
            self.assertIn("beta", agent_stats)
            self.assertIn("mean", agent_stats)
            self.assertIn("variance", agent_stats)

            # Mean should be in [0, 1]
            self.assertGreaterEqual(agent_stats["mean"], 0.0)
            self.assertLessEqual(agent_stats["mean"], 1.0)

            # Variance should be non-negative
            self.assertGreaterEqual(agent_stats["variance"], 0.0)

    def test_get_agent_stats_per_regime(self):
        """Test getting stats for specific regime."""
        for regime in self.weights.config.regime_names:
            stats = self.weights.get_agent_stats(regime=regime)
            self.assertEqual(len(stats), len(self.weights.config.agent_names))

    def test_get_agent_stats_invalid_regime(self):
        """Test invalid regime raises ValueError."""
        with self.assertRaises(ValueError):
            self.weights.get_agent_stats(regime="INVALID")

    def test_mean_calculation_correctness(self):
        """Test mean is correctly calculated as α/(α+β)."""
        agent = self.weights.config.agent_names[0]
        self.weights._distributions[("NORMAL", agent)] = (3.0, 7.0)

        stats = self.weights.get_agent_stats(regime="NORMAL")
        expected_mean = 3.0 / 10.0
        self.assertAlmostEqual(stats[agent]["mean"], expected_mean, places=5)


class TestTopAgents(unittest.TestCase):
    """Test top agents ranking."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = create_default_bayesian_weights()

    def test_get_top_agents_returns_n_agents(self):
        """Test get_top_agents returns correct number of agents."""
        top_5 = self.weights.get_top_agents(n=5)
        self.assertEqual(len(top_5), 5)

    def test_get_top_agents_ranked_by_mean(self):
        """Test agents are ranked by mean performance."""
        # Set up distinct alpha/beta to have clear ordering
        agents = self.weights.config.agent_names
        for i, agent in enumerate(agents):
            alpha = 10 + i  # Increasing alpha
            beta = 5
            self.weights._distributions[("NORMAL", agent)] = (alpha, beta)

        top_agents = self.weights.get_top_agents(regime="NORMAL", n=len(agents))

        # Last agent should have highest alpha/(alpha+beta) ratio
        self.assertEqual(top_agents[0], agents[-1])

    def test_get_top_agents_invalid_regime(self):
        """Test invalid regime raises ValueError."""
        with self.assertRaises(ValueError):
            self.weights.get_top_agents(regime="INVALID")

    def test_get_top_agents_n_exceeds_agent_count(self):
        """Test n > num_agents just returns all agents."""
        top = self.weights.get_top_agents(n=100)
        self.assertEqual(len(top), len(self.weights.config.agent_names))


class TestStatePersistence(unittest.TestCase):
    """Test state save/load functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = create_default_bayesian_weights()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_state_creates_file(self):
        """Test save_state creates a file."""
        path = os.path.join(self.temp_dir, "test_state.json")
        self.weights.save_state(path)
        self.assertTrue(os.path.exists(path))

    def test_save_state_valid_json(self):
        """Test saved state is valid JSON."""
        path = os.path.join(self.temp_dir, "test_state.json")
        self.weights.save_state(path)

        with open(path) as f:
            data = json.load(f)

        self.assertIn("version", data)
        self.assertIn("update_count", data)
        self.assertIn("distributions", data)

    def test_load_state_restores_values(self):
        """Test load_state restores saved state."""
        path = os.path.join(self.temp_dir, "test_state.json")

        # Perform some updates
        scores = {a: 0.5 for a in self.weights.config.agent_names}
        self.weights.update(scores, "win", "NORMAL")
        self.weights.update(scores, "loss", "NORMAL")

        update_count_before = self.weights._update_count
        epsilon_before = self.weights._epsilon
        self.weights.save_state(path)

        # Create new instance and load
        weights2 = create_default_bayesian_weights()
        weights2.load_state(path)

        self.assertEqual(weights2._update_count, update_count_before)
        self.assertAlmostEqual(weights2._epsilon, epsilon_before, places=5)

    def test_load_state_missing_file(self):
        """Test load_state handles missing file gracefully."""
        path = os.path.join(self.temp_dir, "nonexistent.json")
        self.weights.load_state(path)
        # Should not raise, should reinitialize to prior
        self.assertEqual(self.weights._update_count, 0)

    def test_load_state_corrupted_json(self):
        """Test load_state handles corrupted JSON gracefully."""
        path = os.path.join(self.temp_dir, "corrupted.json")
        with open(path, "w") as f:
            f.write("{ invalid json")

        self.weights.load_state(path)
        # Should not raise, should reinitialize to prior
        self.assertEqual(self.weights._update_count, 0)

    def test_load_state_missing_version(self):
        """Test load_state rejects state without version field."""
        path = os.path.join(self.temp_dir, "no_version.json")
        with open(path, "w") as f:
            json.dump({"distributions": {}}, f)

        self.weights.load_state(path)
        # Should reinitialize to prior
        self.assertEqual(self.weights._update_count, 0)

    def test_save_load_roundtrip(self):
        """Test full save/load roundtrip preserves state."""
        path = os.path.join(self.temp_dir, "roundtrip.json")

        # Modify state
        scores = {a: (i + 1) / len(self.weights.config.agent_names)
                  for i, a in enumerate(self.weights.config.agent_names)}
        for _ in range(5):
            self.weights.update(scores, "win", "NORMAL")
        for _ in range(3):
            self.weights.update(scores, "loss", "NORMAL")

        distributions_before = {
            k: v for k, v in self.weights._distributions.items()
        }
        self.weights.save_state(path)

        # Load into new instance
        weights2 = create_default_bayesian_weights()
        weights2.load_state(path)

        # Verify distributions match
        for key in distributions_before:
            alpha_before, beta_before = distributions_before[key]
            alpha_after, beta_after = weights2._distributions[key]
            self.assertAlmostEqual(alpha_before, alpha_after, places=5)
            self.assertAlmostEqual(beta_before, beta_after, places=5)

    def test_save_state_atomic_write(self):
        """Test save_state uses atomic write (no .tmp file left behind)."""
        path = os.path.join(self.temp_dir, "atomic.json")
        self.weights.save_state(path)

        # Check no .tmp files in directory
        files = os.listdir(self.temp_dir)
        tmp_files = [f for f in files if f.startswith(".tmp")]
        self.assertEqual(len(tmp_files), 0)

    def test_save_state_creates_parent_dirs(self):
        """Test save_state creates parent directories if needed."""
        path = os.path.join(self.temp_dir, "a", "b", "c", "state.json")
        self.weights.save_state(path)
        self.assertTrue(os.path.exists(path))


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_create_default_weights(self):
        """Test create_default_bayesian_weights."""
        weights = create_default_bayesian_weights()
        self.assertEqual(weights.config.epsilon_initial, 0.15)
        self.assertEqual(weights.config.epsilon_min, 0.02)

    def test_create_exploration_heavy_weights(self):
        """Test create_exploration_heavy_weights."""
        weights = create_exploration_heavy_weights()
        self.assertEqual(weights.config.epsilon_initial, 0.30)
        self.assertEqual(weights.config.epsilon_min, 0.05)
        self.assertGreater(weights.config.epsilon_initial,
                          create_default_bayesian_weights().config.epsilon_initial)

    def test_create_exploitation_heavy_weights(self):
        """Test create_exploitation_heavy_weights."""
        weights = create_exploitation_heavy_weights()
        self.assertEqual(weights.config.epsilon_initial, 0.05)
        self.assertEqual(weights.config.epsilon_min, 0.01)
        self.assertLess(weights.config.epsilon_initial,
                       create_default_bayesian_weights().config.epsilon_initial)


class TestConvergence(unittest.TestCase):
    """Test convergence properties."""

    def test_convergence_to_best_agent(self):
        """Test that best agent gets highest weight after many wins."""
        weights = create_exploitation_heavy_weights()
        agents = weights.config.agent_names

        # Create scenario: agent[0] always scores high on wins
        for _ in range(50):
            scores = {agents[0]: 0.9}
            scores.update({a: 0.3 for a in agents[1:]})
            weights.update(scores, "win", "NORMAL")

        top_agents = weights.get_top_agents(n=3)
        self.assertEqual(top_agents[0], agents[0])

    def test_convergence_loses_poor_agent(self):
        """Test that agent with consistently wrong predictions loses relative weight."""
        weights = create_exploitation_heavy_weights()
        agents = weights.config.agent_names

        # Scenario: agent[0] is "pessimistic" (low scores), but price goes UP (win)
        # This means agent[0] was wrong. On a win, low-score agents increase β
        for _ in range(50):
            scores = {agents[0]: 0.2}  # agent[0] was pessimistic
            scores.update({a: 0.8 for a in agents[1:]})  # others were optimistic
            weights.update(scores, "win", "NORMAL")  # price went up, so optimistic was right
            # Result: agent[0] gets β += (1-0.2) = 0.8, others get α += 0.8

        # Get statistics: agent[0] should have lower mean than others
        stats = weights.get_agent_stats()
        agent0_mean = stats[agents[0]]["mean"]
        other_means = [stats[a]["mean"] for a in agents[1:]]

        # agent[0] should have significantly lower mean due to high β
        self.assertLess(agent0_mean, sum(other_means) / len(other_means))

    def test_regime_conditional_weights_differ(self):
        """Test that different regimes develop different weights after training."""
        weights = BayesianAgentWeights()

        # Train LOW regime with first agent dominant
        for _ in range(30):
            scores = {weights.config.agent_names[0]: 0.9}
            scores.update({a: 0.2 for a in weights.config.agent_names[1:]})
            weights.update(scores, "win", "LOW")

        # Train HIGH regime with second agent dominant
        for _ in range(30):
            scores = {weights.config.agent_names[1]: 0.9}
            scores.update({a: 0.2 for a in weights.config.agent_names[2:]})
            weights.update(scores, "win", "HIGH")

        # Check top agents differ by regime
        top_low = weights.get_top_agents(regime="LOW", n=1)
        top_high = weights.get_top_agents(regime="HIGH", n=1)

        self.assertEqual(top_low[0], weights.config.agent_names[0])
        self.assertEqual(top_high[0], weights.config.agent_names[1])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = create_default_bayesian_weights()

    def test_update_with_zero_scores(self):
        """Test update with all agents scoring 0.0."""
        scores = {a: 0.0 for a in self.weights.config.agent_names}
        self.weights.update(scores, "win", "NORMAL")
        # Should not raise

    def test_update_with_uniform_scores(self):
        """Test update with all agents scoring 0.5."""
        scores = {a: 0.5 for a in self.weights.config.agent_names}
        self.weights.update(scores, "win", "NORMAL")
        # Should not raise

    def test_sample_weights_consistency_same_regime(self):
        """Test multiple samples from same regime are consistent in distribution."""
        samples = [self.weights.sample_weights(regime="NORMAL") for _ in range(10)]

        # All should be for NORMAL regime
        self.assertTrue(all(s.regime == "NORMAL" for s in samples))

        # All should be normalized
        self.assertTrue(all(abs(sum(s.weights.values()) - 1.0) < 0.01 for s in samples))

    def test_reset_functionality(self):
        """Test reset() returns to initial state."""
        # Perform updates
        scores = {a: 0.5 for a in self.weights.config.agent_names}
        self.weights.update(scores, "win", "NORMAL")
        self.weights.update(scores, "loss", "NORMAL")

        self.assertGreater(self.weights._update_count, 0)

        # Reset
        self.weights.reset()

        self.assertEqual(self.weights._update_count, 0)
        self.assertEqual(self.weights._epsilon, self.weights.config.epsilon_initial)

        # Distributions should be back to prior
        for (regime, agent) in self.weights._distributions:
            alpha, beta = self.weights._distributions[(regime, agent)]
            self.assertEqual(alpha, self.weights.config.prior_alpha)
            self.assertEqual(beta, self.weights.config.prior_beta)

    def test_many_sequential_updates(self):
        """Test system stability under many updates."""
        scores = {a: 0.5 + 0.1 * (i % 10) / 10
                  for i, a in enumerate(self.weights.config.agent_names)}

        for _ in range(100):
            outcome = "win" if np.random.rand() > 0.4 else "loss"
            self.weights.update(scores, outcome, "NORMAL")

        # Should still be able to sample
        sample = self.weights.sample_weights()
        total = sum(sample.weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_config_with_single_agent(self):
        """Test system works with single agent (degenerate case)."""
        config = BayesianWeightsConfig(agent_names=["only_agent"])
        weights = BayesianAgentWeights(config)

        sample = weights.sample_weights()
        self.assertEqual(sample.weights["only_agent"], 1.0)

    def test_config_with_many_agents(self):
        """Test system works with many agents."""
        many_agents = [f"agent_{i}" for i in range(100)]
        config = BayesianWeightsConfig(agent_names=many_agents)
        weights = BayesianAgentWeights(config)

        sample = weights.sample_weights()
        self.assertEqual(len(sample.weights), 100)

    def test_large_update_counts(self):
        """Test epsilon decay stability with very large update counts."""
        weights = create_exploitation_heavy_weights()
        scores = {a: 0.5 for a in weights.config.agent_names}

        for _ in range(10000):
            weights.update(scores, "win", "NORMAL")

        # Epsilon should have decayed but not gone below min
        # Exploitation-heavy has epsilon_min=0.01, so it may converge to exactly 0.01
        self.assertGreaterEqual(weights._epsilon, weights.config.epsilon_min)
        self.assertLessEqual(weights._epsilon, weights.config.epsilon_initial)


class TestRegimeConditionalBehavior(unittest.TestCase):
    """Test regime-specific behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = BayesianAgentWeights()

    def test_all_regimes_sampled_independently(self):
        """Test each regime has independent Beta distributions."""
        agent = self.weights.config.agent_names[0]

        # Set different alpha/beta for each regime
        for regime in self.weights.config.regime_names:
            alpha = self.weights.config.regime_names.index(regime) + 2
            self.weights._distributions[(regime, agent)] = (alpha, 2.0)

        # Sample from each regime
        for regime in self.weights.config.regime_names:
            sample = self.weights.sample_weights(regime=regime)
            self.assertEqual(sample.regime, regime)

    def test_update_affects_only_target_regime(self):
        """Test updating one regime doesn't affect others."""
        # Use explicit non-0.5 scores so there's a clear change
        agent = self.weights.config.agent_names[0]
        scores = {a: 0.8 for a in self.weights.config.agent_names}  # High score

        # Get baseline
        baseline = {}
        for regime in self.weights.config.regime_names:
            baseline[regime] = self.weights._distributions[(regime, agent)]

        # Update only NORMAL with a win (high score agents get α += score)
        self.weights.update(scores, "win", "NORMAL")

        # Check NORMAL changed, others didn't
        for regime in self.weights.config.regime_names:
            alpha, beta = self.weights._distributions[(regime, agent)]
            baseline_alpha, baseline_beta = baseline[regime]

            if regime == "NORMAL":
                # With score=0.8 and outcome=win, alpha should increase
                self.assertGreater(alpha, baseline_alpha)
            else:
                # Other regimes should be unchanged
                self.assertEqual(alpha, baseline_alpha)
                self.assertEqual(beta, baseline_beta)


if __name__ == "__main__":
    unittest.main()
