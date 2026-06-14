"""Tests for SAC execution agent (US-012)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.rl.sac_execution import SACConfig, SACExecutionAgent, PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer:
    def test_add_and_sample(self):
        buf = PrioritizedReplayBuffer(capacity=100, state_dim=6, action_dim=1)
        for i in range(10):
            buf.add(np.ones(6) * i, np.array([0.5]), 1.0, np.ones(6) * (i + 1), False)
        assert buf.size == 10
        s, a, r, s_next, d, idx, weights = buf.sample(5)
        assert s.shape == (5, 6)
        assert a.shape == (5, 1)
        assert weights.shape == (5,)
        assert np.all(weights > 0)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=10, state_dim=3, action_dim=1)
        buf.add(np.ones(3), np.array([0.1]), 0.0, np.ones(3), False)
        old_prio = buf.priorities[0]
        buf.update_priorities(np.array([0]), np.array([5.0]))
        assert buf.priorities[0] > old_prio


class TestSACExecutionAgent:
    def test_init(self):
        agent = SACExecutionAgent()
        assert agent.cfg.state_dim == 6
        assert agent.cfg.action_dim == 1

    def test_select_action_range(self):
        agent = SACExecutionAgent(SACConfig(state_dim=4, action_dim=1))
        state = np.zeros(4)
        for _ in range(20):
            action = agent.select_action(state)
            assert action.shape == (1,)
            assert 0.0 <= action[0] <= 1.0

    def test_train_step_empty_buffer(self):
        agent = SACExecutionAgent()
        metrics = agent.train_step()
        assert metrics["actor_loss"] == 0.0
        assert metrics["critic_loss"] == 0.0

    def test_fit_and_predict(self):
        cfg = SACConfig(state_dim=4, action_dim=1, epochs=20)
        agent = SACExecutionAgent(cfg)
        dataset = []
        for i in range(200):
            s = np.random.randn(4)
            a = np.array([np.random.rand()])
            r = -np.random.rand()  # negative reward (cost)
            s_next = np.random.randn(4)
            done = i > 190
            dataset.append({"state": s, "action": a, "reward": r, "next_state": s_next, "done": done})
        agent.fit(dataset)
        result = agent.predict(np.zeros(4))
        assert "action" in result
        assert "q_value" in result
        assert 0.0 <= result["action"] <= 1.0

    def test_deterministic_action_reproducible(self):
        cfg = SACConfig(state_dim=3, action_dim=1, epochs=10)
        agent = SACExecutionAgent(cfg)
        for i in range(50):
            agent.buffer.add(np.ones(3), np.array([0.5]), -0.1, np.ones(3), False)
        for _ in range(5):
            agent.train_step()
        a1 = agent.select_action(np.ones(3), deterministic=True)
        a2 = agent.select_action(np.ones(3), deterministic=True)
        np.testing.assert_array_equal(a1, a2)
