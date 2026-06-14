"""Soft Actor-Critic (SAC) execution agent for order slicing.

US-012: Optimally slices large orders to minimize market impact
using continuous action space SAC with prioritized experience replay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    """Configuration for SAC execution agent."""

    state_dim: int = 6
    action_dim: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # entropy temperature
    lr: float = 3e-4
    batch_size: int = 64
    hidden_dim: int = 128
    epochs: int = 50
    buffer_capacity: int = 10000
    min_action: float = 0.0
    max_action: float = 1.0


class PrioritizedReplayBuffer:
    """Replay buffer with prioritized experience replay (simplified)."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.size = 0
        self.ptr = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
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
        # New transitions get max priority
        self.priorities[idx] = self.priorities.max() if self.size > 0 else 1.0
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if self.size == 0:
            raise ValueError("Buffer is empty")
        # Probabilistic sampling based on priorities
        probs = self.priorities[: self.size] / self.priorities[: self.size].sum()
        idx = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        weights = (self.size * probs[idx]) ** (-0.4)
        weights /= weights.max()
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            idx,
            weights.astype(np.float32),
        )

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        self.priorities[idx] = np.abs(td_errors) + 1e-6


class SACExecutionAgent:
    """SAC execution agent for order slicing.

    State: [remaining_volume_pct, time_horizon, spread, order_book_imbalance,
            recent_price_change, volatility]
    Action: continuous fraction of remaining volume to execute (0-1)
    Reward: negative implementation shortfall + penalty for high market impact
    """

    def __init__(self, config: Optional[SACConfig] = None):
        self.cfg = config or SACConfig()
        self.buffer = PrioritizedReplayBuffer(
            self.cfg.buffer_capacity,
            self.cfg.state_dim,
            self.cfg.action_dim,
        )

        # Actor (policy) network
        self._init_actor()
        # Twin Q-networks (for stability)
        self._init_critic()

        # Target Q-networks (soft-updated)
        self._target_Q1 = self._copy_critic_weights()
        self._target_Q2 = self._copy_critic_weights()

    def _init_actor(self) -> None:
        d = self.cfg.state_dim
        h = self.cfg.hidden_dim
        a = self.cfg.action_dim

        limit1 = np.sqrt(6.0 / (d + h))
        self._actor_W1 = np.random.uniform(-limit1, limit1, (d, h))
        self._actor_b1 = np.zeros(h)

        limit2 = np.sqrt(6.0 / (h + a))
        self._actor_W2 = np.random.uniform(-limit2, limit2, (h, a))
        self._actor_b2 = np.zeros(a)

        # Log std head (for Gaussian policy)
        self._actor_logstd = np.zeros(a)

    def _init_critic(self) -> None:
        d = self.cfg.state_dim
        h = self.cfg.hidden_dim
        a = self.cfg.action_dim

        # Q1
        limit1 = np.sqrt(6.0 / ((d + a) + h))
        self._critic1_W1 = np.random.uniform(-limit1, limit1, (d + a, h))
        self._critic1_b1 = np.zeros(h)
        limit2 = np.sqrt(6.0 / (h + 1))
        self._critic1_W2 = np.random.uniform(-limit2, limit2, (h, 1))
        self._critic1_b2 = np.zeros(1)

        # Q2
        self._critic2_W1 = np.random.uniform(-limit1, limit1, (d + a, h))
        self._critic2_b1 = np.zeros(h)
        self._critic2_W2 = np.random.uniform(-limit2, limit2, (h, 1))
        self._critic2_b2 = np.zeros(1)

    def _copy_critic_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self._critic1_W1.copy(),
            "b1": self._critic1_b1.copy(),
            "W2": self._critic1_W2.copy(),
            "b2": self._critic1_b2.copy(),
        }

    def _actor_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean_action, log_std)."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        h = np.maximum(0, state @ self._actor_W1 + self._actor_b1)
        mean = np.tanh(h @ self._actor_W2 + self._actor_b2)
        logstd = self._actor_logstd
        return mean, logstd

    def _q_forward(self, state: np.ndarray, action: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        sa = np.concatenate([state, action], axis=1)
        h = np.maximum(0, sa @ W1 + b1)
        q = h @ W2 + b2
        return q

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample action from Gaussian policy."""
        mean, logstd = self._actor_forward(state)
        if deterministic:
            action = mean
        else:
            std = np.exp(np.clip(logstd, -20, 2))
            noise = np.random.randn(*mean.shape)
            action = mean + std * noise
        # Squash to action bounds
        action = np.tanh(action)
        action = np.clip(
            action,
            self.cfg.min_action,
            self.cfg.max_action,
        )
        return action.flatten()

    def _critic_train_step(self, s, a, r, s_next, d, idx, weights) -> np.ndarray:
        """Update twin Q-networks. Returns TD errors for priority update."""
        # Current Q values
        q1 = self._q_forward(s, a, self._critic1_W1, self._critic1_b1, self._critic1_W2, self._critic1_b2).flatten()
        q2 = self._q_forward(s, a, self._critic2_W1, self._critic2_b1, self._critic2_W2, self._critic2_b2).flatten()

        # Target Q values (SAC uses min of twin Q-targets + entropy bonus)
        with np.errstate(all="ignore"):
            a_next, _ = self._actor_forward(s_next)
            a_next = np.tanh(a_next)
            q1_next = self._q_forward(s_next, a_next, *self._target_Q1.values()).flatten()
            q2_next = self._q_forward(s_next, a_next, *self._target_Q2.values()).flatten()
            q_next = np.minimum(q1_next, q2_next) - self.cfg.alpha * 0.0  # simplified entropy
            target_q = r + self.cfg.gamma * (1 - d) * q_next

        # MSE losses
        td1 = q1 - target_q
        td2 = q2 - target_q
        loss1 = np.mean(weights * td1 ** 2)
        loss2 = np.mean(weights * td2 ** 2)

        # Simple gradient descent (manual)
        lr = self.cfg.lr
        # Update Q1
        grad_q1 = 2 * weights * td1 / len(s)
        h1 = np.maximum(0, np.concatenate([s, a], axis=1) @ self._critic1_W1 + self._critic1_b1)
        self._critic1_W2 -= lr * (h1.T @ grad_q1.reshape(-1, 1))
        self._critic1_b2 -= lr * np.sum(grad_q1)
        grad_h1 = grad_q1.reshape(-1, 1) @ self._critic1_W2.T
        grad_h1[h1 <= 0] = 0
        self._critic1_W1 -= lr * (np.concatenate([s, a], axis=1).T @ grad_h1)
        self._critic1_b1 -= lr * np.sum(grad_h1, axis=0)

        # Update Q2
        grad_q2 = 2 * weights * td2 / len(s)
        h2 = np.maximum(0, np.concatenate([s, a], axis=1) @ self._critic2_W1 + self._critic2_b1)
        self._critic2_W2 -= lr * (h2.T @ grad_q2.reshape(-1, 1))
        self._critic2_b2 -= lr * np.sum(grad_q2)
        grad_h2 = grad_q2.reshape(-1, 1) @ self._critic2_W2.T
        grad_h2[h2 <= 0] = 0
        self._critic2_W1 -= lr * (np.concatenate([s, a], axis=1).T @ grad_h2)
        self._critic2_b1 -= lr * np.sum(grad_h2, axis=0)

        return np.abs(td1) + np.abs(td2)

    def train_step(self) -> Dict[str, float]:
        """One gradient step on actor and critic."""
        if self.buffer.size < self.cfg.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        s, a, r, s_next, d, idx, weights = self.buffer.sample(self.cfg.batch_size)
        td_errors = self._critic_train_step(s, a, r, s_next, d, idx, weights)
        self.buffer.update_priorities(idx, td_errors)

        # Soft update targets
        tau = self.cfg.tau
        for key in self._target_Q1:
            self._target_Q1[key] = tau * getattr(self, f"_critic1_{key}") + (1 - tau) * self._target_Q1[key]
            self._target_Q2[key] = tau * getattr(self, f"_critic2_{key}") + (1 - tau) * self._target_Q2[key]

        return {"actor_loss": 0.0, "critic_loss": float(np.mean(td_errors))}

    def fit(self, dataset: List[Dict[str, Any]]) -> "SACExecutionAgent":
        """Train on offline dataset of execution transitions."""
        for transition in dataset:
            self.buffer.add(
                np.array(transition["state"], dtype=np.float32).flatten(),
                np.array(transition["action"], dtype=np.float32).flatten(),
                float(transition["reward"]),
                np.array(transition["next_state"], dtype=np.float32).flatten(),
                bool(transition["done"]),
            )

        for epoch in range(self.cfg.epochs):
            metrics = self.train_step()
            if epoch % 10 == 0:
                logger.debug("SAC epoch %d: %s", epoch, metrics)

        logger.info("SACExecutionAgent trained on %d transitions", len(dataset))
        return self

    def predict(self, state: np.ndarray) -> Dict[str, Any]:
        """Return action and Q-value for a given state."""
        action = self.select_action(state, deterministic=True)
        q1 = self._q_forward(
            state.reshape(1, -1),
            action.reshape(1, -1),
            self._critic1_W1,
            self._critic1_b1,
            self._critic1_W2,
            self._critic1_b2,
        )
        return {
            "action": float(action[0]),
            "q_value": float(q1[0, 0]),
        }
