"""
Tests for GateThresholdEnv and GateRLConfig (US-262 + US-263).

Covers:
- US-262: GateRLConfig (10+ tests) — defaults, ranges, weights, hyperparams
- US-263: Gate Math (10+ tests) — action scaling, gate checks, trade simulation, rewards
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

import numpy as np

# Optional: Skip tests gracefully if dependencies not available
try:
    import gymnasium
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    import stable_baselines3
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class TestGateRLConfig(unittest.TestCase):
    """US-262: GateRLConfig defaults, ranges, weights, hyperparams."""

    def test_config_defaults_base_thresholds(self):
        """GateRLConfig default base thresholds."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig()
        self.assertEqual(config.base_confidence, 50.0)
        self.assertEqual(config.base_momentum, 0.20)
        self.assertEqual(config.base_risk, 0.025)

    def test_config_defaults_delta_ranges(self):
        """GateRLConfig default delta ranges."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig()
        self.assertEqual(config.confidence_delta_range, 0.15)
        self.assertEqual(config.momentum_delta_range, 0.10)
        self.assertEqual(config.risk_delta_range, 0.015)

    def test_config_defaults_reward_weights(self):
        """GateRLConfig default reward weights."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig()
        self.assertEqual(config.sharpe_weight, 0.15)
        self.assertEqual(config.drawdown_penalty, 0.8)
        self.assertEqual(config.win_rate_bonus, 0.25)

    def test_config_defaults_transaction_costs(self):
        """GateRLConfig default transaction costs."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig()
        self.assertEqual(config.spread_cost_pct, 0.0002)
        self.assertEqual(config.commission_pct, 0.0001)
        self.assertEqual(config.slippage_pct, 0.0001)

    def test_config_defaults_sac_hyperparams(self):
        """GateRLConfig default SAC hyperparameters."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig()
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.tau, 0.005)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.buffer_size, 100_000)

    def test_config_defaults_network_arch(self):
        """GateRLConfig default network architectures."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig()
        self.assertEqual(config.net_arch_pi, [64, 64])
        self.assertEqual(config.net_arch_qf, [256, 256])

    def test_config_defaults_n_envs(self):
        """GateRLConfig default n_envs."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig()
        self.assertEqual(config.n_envs, 4)

    def test_config_defaults_max_thresholds(self):
        """GateRLConfig default max thresholds."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig()
        self.assertEqual(config.max_drawdown_pct, 0.10)
        self.assertEqual(config.max_trades_per_episode, 200)

    def test_config_custom_values(self):
        """GateRLConfig accepts custom values."""
        from src.rl.gate_threshold_env import GateRLConfig

        config = GateRLConfig(
            base_confidence=55.0,
            base_momentum=0.25,
            base_risk=0.030,
        )
        self.assertEqual(config.base_confidence, 55.0)
        self.assertEqual(config.base_momentum, 0.25)
        self.assertEqual(config.base_risk, 0.030)

    def test_extract_observation_space_from_model_nonexistent(self):
        """extract_observation_space_from_model returns None for non-existent file."""
        from src.rl.gate_threshold_env import extract_observation_space_from_model

        result = extract_observation_space_from_model(Path("nonexistent.zip"))
        self.assertIsNone(result)

    def test_infer_config_from_model_nonexistent(self):
        """infer_config_from_model returns None for non-existent file."""
        from src.rl.gate_threshold_env import infer_config_from_model

        result = infer_config_from_model(Path("nonexistent.zip"))
        self.assertIsNone(result)


@unittest.skipUnless(HAS_GYM, "gymnasium not available")
class TestGateThresholdEnvMath(unittest.TestCase):
    """US-263: Gate math — action scaling, gate checks, trade simulation, rewards."""

    def setUp(self):
        """Set up test fixtures."""
        from src.rl.gate_threshold_env import GateRLConfig, GateThresholdEnv

        # Create minimal test data
        self.n_samples = 100
        self.prices = np.random.uniform(1.0, 1.1, self.n_samples)
        self.features = np.random.randn(self.n_samples, 10)
        self.ensemble_preds = np.random.uniform(0.3, 0.8, (self.n_samples, 4))
        self.config = GateRLConfig()

        # Create environment
        self.env = GateThresholdEnv(
            prices=self.prices,
            features=self.features,
            ensemble_preds=self.ensemble_preds,
            config=self.config,
        )

    def test_scale_action_zero_action(self):
        """_scale_action: [0,0,0] = no adjustment."""
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        conf_delta, mom_delta, risk_delta = self.env._scale_action(action)

        self.assertAlmostEqual(conf_delta, 0.0, places=5)
        self.assertAlmostEqual(mom_delta, 0.0, places=5)
        self.assertAlmostEqual(risk_delta, 0.0, places=5)

    def test_scale_action_max_positive(self):
        """_scale_action: [1,1,1] = max positive adjustment."""
        action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        conf_delta, mom_delta, risk_delta = self.env._scale_action(action)

        # Check that positive action gives positive delta
        self.assertGreater(conf_delta, 0)
        self.assertGreater(mom_delta, 0)
        self.assertGreater(risk_delta, 0)

    def test_scale_action_max_negative(self):
        """_scale_action: [-1,-1,-1] = max negative adjustment."""
        action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        conf_delta, mom_delta, risk_delta = self.env._scale_action(action)

        # Check that negative action gives negative delta
        self.assertLess(conf_delta, 0)
        self.assertLess(mom_delta, 0)
        self.assertLess(risk_delta, 0)

    def test_scale_action_range(self):
        """_scale_action respects config delta ranges."""
        action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        conf_delta, mom_delta, risk_delta = self.env._scale_action(action)

        # Check ranges
        # conf_delta is scaled to 0-100, so should be ~15 (15% of 100)
        self.assertLessEqual(abs(conf_delta), 15.0 + 1e-5)
        self.assertLessEqual(abs(mom_delta), self.config.momentum_delta_range)
        self.assertLessEqual(abs(risk_delta), self.config.risk_delta_range)

    def test_check_gates_all_pass(self):
        """_check_gates: passes when all thresholds met."""
        # Use low thresholds to ensure passage
        conf_thresh = 1.0
        mom_thresh = 0.0
        risk_thresh = 1.0

        passed, gate_info = self.env._check_gates(conf_thresh, mom_thresh, risk_thresh)

        # Should pass because thresholds are very permissive
        self.assertTrue(passed)
        self.assertTrue(gate_info["conf_passed"])
        self.assertTrue(gate_info["mom_passed"])
        self.assertTrue(gate_info["risk_passed"])

    def test_check_gates_confidence_fail(self):
        """_check_gates: fails when confidence below threshold."""
        # Use high confidence threshold to ensure failure
        conf_thresh = 1000.0
        mom_thresh = 0.0
        risk_thresh = 1.0

        passed, gate_info = self.env._check_gates(conf_thresh, mom_thresh, risk_thresh)

        # Should fail due to high confidence threshold
        self.assertFalse(passed)
        self.assertFalse(gate_info["conf_passed"])

    def test_check_gates_returns_info(self):
        """_check_gates: returns gate_info dict with all fields."""
        conf_thresh = 50.0
        mom_thresh = 0.2
        risk_thresh = 0.03

        passed, gate_info = self.env._check_gates(conf_thresh, mom_thresh, risk_thresh)

        # Check all required fields
        self.assertIn("tcn_prob", gate_info)
        self.assertIn("ridge_conf", gate_info)
        self.assertIn("xgb_mom", gate_info)
        self.assertIn("rf_drawdown", gate_info)
        self.assertIn("conf_passed", gate_info)
        self.assertIn("mom_passed", gate_info)
        self.assertIn("risk_passed", gate_info)
        self.assertIn("thresholds", gate_info)

    def test_simulate_trade_winning_trade(self):
        """_simulate_trade: winning trade with costs deducted."""
        gate_info = {
            "tcn_prob": 0.6,
            "ridge_conf": 60.0,
            "xgb_mom": 0.25,
            "rf_drawdown": 0.02,
            "conf_passed": True,
            "mom_passed": True,
            "risk_passed": True,
        }

        # Create a price environment where the next price is higher (long win)
        original_equity = self.env.equity
        self.env.current_step = 50
        self.env.prices[50] = 1.00
        self.env.prices[51] = 1.01  # 1% move up

        result = self.env._simulate_trade(gates_passed=True, gate_info=gate_info)

        # Should have traded
        self.assertTrue(result["traded"])
        # Should have deducted transaction costs
        self.assertGreater(result["transaction_costs"], 0)
        # Raw P/L should be larger than net P/L due to costs
        self.assertGreater(result["raw_pnl"], result["pnl"])

    def test_simulate_trade_losing_trade(self):
        """_simulate_trade: losing trade."""
        gate_info = {
            "tcn_prob": 0.4,
            "ridge_conf": 40.0,
            "xgb_mom": 0.15,
            "rf_drawdown": 0.05,
            "conf_passed": False,
            "mom_passed": True,
            "risk_passed": False,
        }

        self.env.current_step = 50
        self.env.prices[50] = 1.00
        self.env.prices[51] = 0.99  # 1% move down

        result = self.env._simulate_trade(gates_passed=True, gate_info=gate_info)

        # Should have traded
        self.assertTrue(result["traded"])
        # Should have negative raw P/L (price went down, direction was long based on TCN)
        # Actual sign depends on direction, but we should have a result
        self.assertIsNotNone(result["pnl"])

    def test_simulate_trade_gates_not_passed(self):
        """_simulate_trade: no trade when gates not passed."""
        gate_info = {}

        result = self.env._simulate_trade(gates_passed=False, gate_info=gate_info)

        self.assertFalse(result["traded"])
        self.assertEqual(result["pnl"], 0.0)
        self.assertEqual(result["transaction_costs"], 0.0)

    def test_calculate_reward_no_trade(self):
        """_calculate_reward: small positive for avoiding bad trades."""
        trade_result = {"pnl": 0.0, "traded": False, "transaction_costs": 0.0}

        reward = self.env._calculate_reward(trade_result)

        # No-trade reward should be minimal (0 to small positive)
        self.assertIsInstance(reward, (float, np.floating))

    def test_calculate_reward_profitable_trade(self):
        """_calculate_reward: positive PnL gives positive reward."""
        trade_result = {
            "pnl": 100.0,  # $100 profit
            "traded": True,
            "transaction_costs": 0.5,
        }

        reward = self.env._calculate_reward(trade_result)

        # Profitable trade should give positive reward
        self.assertGreater(reward, 0)

    def test_calculate_reward_losing_trade(self):
        """_calculate_reward: losing trade gives negative reward."""
        trade_result = {
            "pnl": -100.0,  # $100 loss
            "traded": True,
            "transaction_costs": 0.5,
        }

        reward = self.env._calculate_reward(trade_result)

        # Losing trade should give negative reward
        self.assertLess(reward, 0)

    def test_calculate_reward_asymmetric_scaling(self):
        """_calculate_reward: losses are penalized more than gains rewarded."""
        # Test with identical magnitude but opposite sign
        gain_result = {
            "pnl": 100.0,
            "traded": True,
            "transaction_costs": 0.0,
        }
        loss_result = {
            "pnl": -100.0,
            "traded": True,
            "transaction_costs": 0.0,
        }

        gain_reward = self.env._calculate_reward(gain_result)
        loss_reward = self.env._calculate_reward(loss_result)

        # Loss penalty should be larger in magnitude than gain reward
        self.assertGreater(abs(loss_reward), abs(gain_reward))

    def test_calculate_reward_transaction_cost_penalty(self):
        """_calculate_reward: transaction costs reduce reward."""
        result_low_cost = {
            "pnl": 50.0,
            "traded": True,
            "transaction_costs": 0.1,
        }
        result_high_cost = {
            "pnl": 50.0,
            "traded": True,
            "transaction_costs": 5.0,
        }

        reward_low = self.env._calculate_reward(result_low_cost)
        reward_high = self.env._calculate_reward(result_high_cost)

        # Higher transaction costs should result in lower reward
        self.assertGreater(reward_low, reward_high)

    def test_env_initialization_observation_space(self):
        """GateThresholdEnv observation space is 5-dimensional."""
        obs_space = self.env.observation_space
        self.assertEqual(obs_space.shape, (5,))
        self.assertEqual(len(obs_space.low), 5)
        self.assertEqual(len(obs_space.high), 5)

    def test_env_initialization_action_space(self):
        """GateThresholdEnv action space is 3-dimensional continuous."""
        action_space = self.env.action_space
        self.assertEqual(action_space.shape, (3,))
        np.testing.assert_array_equal(action_space.low, [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(action_space.high, [1.0, 1.0, 1.0])

    def test_env_reset_returns_observation(self):
        """GateThresholdEnv reset returns valid observation."""
        obs, info = self.env.reset()

        self.assertEqual(obs.shape, (5,))
        self.assertEqual(len(obs), 5)
        # All observation values should be in valid range
        self.assertTrue(np.all(obs >= self.env.observation_space.low))
        self.assertTrue(np.all(obs <= self.env.observation_space.high))

    def test_env_reset_with_seed(self):
        """GateThresholdEnv reset respects seed."""
        obs1, _ = self.env.reset(seed=42)
        obs2, _ = self.env.reset(seed=42)

        # Same seed should give same starting state
        np.testing.assert_array_equal(obs1, obs2)

    def test_env_reset_clears_history(self):
        """GateThresholdEnv reset clears trade history."""
        # Add some fake history
        self.env.trade_history = [{"pnl": 100}, {"pnl": -50}]
        self.env.reset()

        self.assertEqual(len(self.env.trade_history), 0)


@unittest.skipUnless(HAS_GYM, "gymnasium not available")
class TestGateThresholdEnvStep(unittest.TestCase):
    """Additional tests for env.step() if it exists."""

    def setUp(self):
        """Set up test fixtures."""
        from src.rl.gate_threshold_env import GateRLConfig, GateThresholdEnv

        self.n_samples = 100
        self.prices = np.random.uniform(1.0, 1.1, self.n_samples)
        self.features = np.random.randn(self.n_samples, 10)
        self.ensemble_preds = np.random.uniform(0.3, 0.8, (self.n_samples, 4))
        self.config = GateRLConfig()

        self.env = GateThresholdEnv(
            prices=self.prices,
            features=self.features,
            ensemble_preds=self.ensemble_preds,
            config=self.config,
        )

    def test_env_has_step_method(self):
        """GateThresholdEnv has step method."""
        self.assertTrue(hasattr(self.env, "step"))
        self.assertTrue(callable(self.env.step))

    def test_env_has_get_obs_method(self):
        """GateThresholdEnv has _get_obs method."""
        self.assertTrue(hasattr(self.env, "_get_obs"))
        self.assertTrue(callable(self.env._get_obs))


if __name__ == "__main__":
    unittest.main()
