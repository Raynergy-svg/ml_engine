"""Conservative Q-Learning (CQL) portfolio rebalancer.

US-010: Offline RL rebalancer that learns from historical trade outcomes
without requiring online interaction with the market.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CQLConfig:
    """Configuration for CQL rebalancer."""

    state_dim: int = 20
    n_actions: int = 5
    gamma: float = 0.99
    tau: float = 0.005
    alpha_cql: float = 1.0  # CQL conservative penalty weight
    lr: float = 3e-4
    batch_size: int = 64
    hidden_dim: int = 128
    epochs: int = 50
    buffer_capacity: int = 10000


class ReplayBuffer:
    """Simple replay buffer for (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.size = 0
        self.ptr = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class CQLRebalancer:
    """Conservative Q-Learning portfolio rebalancer.

    Uses a conservative Q-loss that penalizes overestimation of Q-values
    for out-of-distribution actions, making it suitable for offline RL
    on historical trading data.
    """

    def __init__(self, config: Optional[CQLConfig] = None):
        self.cfg = config or CQLConfig()
        self.buffer = ReplayBuffer(self.cfg.buffer_capacity, self.cfg.state_dim)

        # Q-network weights (simple 2-layer MLP)
        self._trained = False
        self._init_networks()

        # Target network (soft-updated)
        self._target_Q1 = self._copy_weights(self._W1, self._b1, self._W2, self._b2)

    def _init_networks(self) -> None:
        d = self.cfg.state_dim
        h = self.cfg.hidden_dim
        a = self.cfg.n_actions

        limit1 = np.sqrt(6.0 / (d + h))
        self._W1 = np.random.uniform(-limit1, limit1, (d, h))
        self._b1 = np.zeros(h)

        limit2 = np.sqrt(6.0 / (h + h))
        self._W2 = np.random.uniform(-limit2, limit2, (h, h))
        self._b2 = np.zeros(h)

        limit3 = np.sqrt(6.0 / (h + a))
        self._W3 = np.random.uniform(-limit3, limit3, (h, a))
        self._b3 = np.zeros(a)

    def _copy_weights(self, W1, b1, W2, b2):
        return {
            "W1": W1.copy(),
            "b1": b1.copy(),
            "W2": W2.copy(),
            "b2": b2.copy(),
        }

    def _q_forward(self, state: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """Forward pass through Q-network. Returns Q(s, a) for all actions."""
        h1 = np.maximum(0, state @ W1 + b1)  # ReLU
        h2 = np.maximum(0, h1 @ W2 + b2)       # ReLU
        q = h2 @ self._W3 + self._b3
        return q

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.cfg.n_actions)
        q_values = self._q_forward(state, self._W1, self._b1, self._W2, self._b2)
        return int(np.argmax(q_values))

    def train_step(self) -> Dict[str, float]:
        """One gradient step on CQL loss.

        Returns metrics dict with keys: q_loss, cql_loss, total_loss.
        """
        if self.buffer.size < self.cfg.batch_size:
            return {"q_loss": 0.0, "cql_loss": 0.0, "total_loss": 0.0}

        s, a, r, s_next, d = self.buffer.sample(self.cfg.batch_size)

        # Current Q-values
        q_all = self._q_forward(s, self._W1, self._b1, self._W2, self._b2)
        q_sa = q_all[np.arange(self.cfg.batch_size), a]

        # Target Q-values (Bellman backup)
        q_next_all = self._q_forward(s_next, *self._target_Q1.values())
        q_next_max = np.max(q_next_all, axis=1)
        target_q = r + self.cfg.gamma * (1 - d) * q_next_max

        # Standard Q-loss (MSE)
        q_loss = np.mean((q_sa - target_q) ** 2)

        # CQL conservative penalty: push down Q-values for all actions
        # Penalty = E[log(sum_a exp(Q(s,a)))] - E[Q(s,a)]
        logsumexp = np.log(np.sum(np.exp(q_all), axis=1) + 1e-8)
        cql_penalty = np.mean(logsumexp - q_sa)

        total_loss = q_loss + self.cfg.alpha_cql * cql_penalty

        # Backprop (manual numpy)
        grad_q_sa = 2 * (q_sa - target_q) / self.cfg.batch_size
        grad_cql = self.cfg.alpha_cql * (
            (np.exp(q_all) / (np.sum(np.exp(q_all), axis=1, keepdims=True) + 1e-8))
            - np.eye(self.cfg.n_actions)[a]
        ) / self.cfg.batch_size

        grad_total = grad_q_sa[:, None] * np.eye(self.cfg.n_actions)[a] + grad_cql

        # Gradients w.r.t W3, b3
        h1 = np.maximum(0, s @ self._W1 + self._b1)
        h2 = np.maximum(0, h1 @ self._W2 + self._b2)
        grad_W3 = h2.T @ grad_total
        grad_b3 = np.sum(grad_total, axis=0)

        # Gradients w.r.t h2
        grad_h2 = grad_total @ self._W3.T
        grad_h2[h2 <= 0] = 0  # ReLU backprop

        grad_W2 = h1.T @ grad_h2
        grad_b2 = np.sum(grad_h2, axis=0)

        # Gradients w.r.t h1
        grad_h1 = grad_h2 @ self._W2.T
        grad_h1[h1 <= 0] = 0  # ReLU backprop

        grad_W1 = s.T @ grad_h1
        grad_b1 = np.sum(grad_h1, axis=0)

        # Update weights
        lr = self.cfg.lr
        self._W3 -= lr * grad_W3
        self._b3 -= lr * grad_b3
        self._W2 -= lr * grad_W2
        self._b2 -= lr * grad_b2
        self._W1 -= lr * grad_W1
        self._b1 -= lr * grad_b1

        # Soft update target network
        tau = self.cfg.tau
        self._target_Q1["W1"] = tau * self._W1 + (1 - tau) * self._target_Q1["W1"]
        self._target_Q1["b1"] = tau * self._b1 + (1 - tau) * self._target_Q1["b1"]
        self._target_Q1["W2"] = tau * self._W2 + (1 - tau) * self._target_Q1["W2"]
        self._target_Q1["b2"] = tau * self._b2 + (1 - tau) * self._target_Q1["b2"]

        return {
            "q_loss": float(q_loss),
            "cql_loss": float(cql_penalty),
            "total_loss": float(total_loss),
        }

    def fit(self, dataset: List[Dict[str, Any]]) -> "CQLRebalancer":
        """Train on offline dataset of transitions.

        Args:
            dataset: List of dicts with keys 'state', 'action', 'reward',
                     'next_state', 'done'.
        """
        for transition in dataset:
            self.buffer.add(
                np.array(transition["state"], dtype=np.float32),
                int(transition["action"]),
                float(transition["reward"]),
                np.array(transition["next_state"], dtype=np.float32),
                bool(transition["done"]),
            )

        for epoch in range(self.cfg.epochs):
            metrics = self.train_step()
            if epoch % 10 == 0:
                logger.debug("CQL epoch %d: %s", epoch, metrics)

        logger.info("CQLRebalancer trained on %d transitions", len(dataset))
        return self

    def predict(self, state: np.ndarray) -> Dict[str, Any]:
        """Return action and Q-values for a given state."""
        q_values = self._q_forward(state, self._W1, self._b1, self._W2, self._b2)
        action = int(np.argmax(q_values))
        return {
            "action": action,
            "q_values": q_values.tolist(),
            "confidence": float(q_values[action] - np.mean(q_values)),
        }
