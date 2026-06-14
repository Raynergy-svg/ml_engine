"""Tests for CQL portfolio rebalancer (US-010)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.rl.cql_rebalancer import CQLConfig, CQLRebalancer, ReplayBuffer


class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(capacity=100, state_dim=5)
        for i in range(10):
            buf.add(np.ones(5) * i, i % 3, float(i), np.ones(5) * (i + 1), i == 9)
        assert buf.size == 10
        s, a, r, s_next, d = buf.sample(5)
        assert s.shape == (5, 5)
        assert a.shape == (5,)
        assert r.shape == (5,)

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=5, state_dim=3)
        for i in range(10):
            buf.add(np.ones(3), 0, 0.0, np.ones(3), False)
        assert buf.size == 5


class TestCQLRebalancer:
    def test_init(self):
        agent = CQLRebalancer()
        assert agent.cfg.n_actions == 5
        assert not agent._trained  # buffer is empty

    def test_select_action_epsilon_greedy(self):
        agent = CQLRebalancer(CQLConfig(state_dim=4, n_actions=3))
        state = np.zeros(4)
        actions = [agent.select_action(state, epsilon=1.0) for _ in range(100)]
        assert all(0 <= a < 3 for a in actions)
        assert len(set(actions)) > 1  # some exploration

    def test_train_step_empty_buffer(self):
        agent = CQLRebalancer()
        metrics = agent.train_step()
        assert metrics["q_loss"] == 0.0
        assert metrics["cql_loss"] == 0.0

    def test_fit_and_predict(self):
        cfg = CQLConfig(state_dim=4, n_actions=3, epochs=20)
        agent = CQLRebalancer(cfg)

        dataset = []
        for i in range(200):
            s = np.random.randn(4)
            a = np.random.randint(0, 3)
            r = np.random.randn()
            s_next = np.random.randn(4)
            done = i > 190
            dataset.append({"state": s, "action": a, "reward": r, "next_state": s_next, "done": done})

        agent.fit(dataset)
        result = agent.predict(np.zeros(4))
        assert "action" in result
        assert "q_values" in result
        assert "confidence" in result
        assert 0 <= result["action"] < cfg.n_actions

    def test_cql_loss_penalty(self):
        cfg = CQLConfig(state_dim=3, n_actions=4, epochs=10, alpha_cql=1.0)
        agent = CQLRebalancer(cfg)
        for i in range(100):
            agent.buffer.add(np.random.randn(3), i % 4, 0.0, np.random.randn(3), False)
        metrics = agent.train_step()
        assert metrics["cql_loss"] >= 0.0
        assert metrics["total_loss"] >= metrics["q_loss"]
