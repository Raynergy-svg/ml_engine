"""RLHF reward shaping for trading policy alignment.

US-011: Expert trader preferences shape the reward function via
a learned reward model trained on pairwise comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """Configuration for RLHF reward shaper."""

    state_dim: int = 20
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 20
    min_comparisons: int = 10
    preference_temperature: float = 1.0


class RLHFRewardShaper:
    """Reward shaping via preference learning (RLHF).

    Learns a reward model from pairwise comparisons of trades.
    When expert feedback is sparse, falls back to the base reward.
    """

    def __init__(self, config: Optional[RLHFConfig] = None):
        self.cfg = config or RLHFConfig()
        self._comparisons: List[Dict[str, Any]] = []
        self._trained = False

        # Reward model weights (2-layer MLP)
        self._init_model()

    def _init_model(self) -> None:
        d = self.cfg.state_dim
        h = self.cfg.hidden_dim
        limit1 = np.sqrt(6.0 / (d + h))
        self._W1 = np.random.uniform(-limit1, limit1, (d, h))
        self._b1 = np.zeros(h)
        limit2 = np.sqrt(6.0 / (h + 1))
        self._W2 = np.random.uniform(-limit2, limit2, (h, 1))
        self._b2 = np.zeros(1)

    def _reward_forward(self, state: np.ndarray) -> float:
        """Predict reward for a single state."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        h = np.maximum(0, state @ self._W1 + self._b1)
        r = h @ self._W2 + self._b2
        return float(r[0, 0])

    def add_comparison(
        self,
        state_a: np.ndarray,
        state_b: np.ndarray,
        preferred: str,  # "A" or "B"
    ) -> None:
        """Add a pairwise comparison from expert feedback.

        Args:
            state_a: Features for trade A.
            state_b: Features for trade B.
            preferred: "A" if A is preferred, "B" if B is preferred.
        """
        self._comparisons.append({
            "state_a": np.array(state_a, dtype=np.float32).flatten(),
            "state_b": np.array(state_b, dtype=np.float32).flatten(),
            "preferred": preferred,
        })

    def fit(self) -> "RLHFRewardShaper":
        """Train reward model on collected comparisons."""
        if len(self._comparisons) < self.cfg.min_comparisons:
            logger.info(
                "RLHF: only %d comparisons (min %d), skipping training",
                len(self._comparisons),
                self.cfg.min_comparisons,
            )
            self._trained = False
            return self

        for epoch in range(self.cfg.epochs):
            total_loss = 0.0
            for comp in self._comparisons:
                sa = comp["state_a"].reshape(1, -1)
                sb = comp["state_b"].reshape(1, -1)
                pref = 1.0 if comp["preferred"] == "A" else 0.0

                ra = self._reward_forward(sa)
                rb = self._reward_forward(sb)

                # Bradley-Terry model: P(pref=A) = sigmoid((rA - rB) / temp)
                diff = (ra - rb) / self.cfg.preference_temperature
                prob_a = 1.0 / (1.0 + np.exp(-diff))

                # Cross-entropy loss
                loss = -(pref * np.log(prob_a + 1e-8) + (1 - pref) * np.log(1 - prob_a + 1e-8))
                total_loss += loss

                # Gradients
                grad_prob = prob_a - pref
                grad_diff = grad_prob / self.cfg.preference_temperature

                # Backprop through diff = ra - rb
                h_a = np.maximum(0, sa @ self._W1 + self._b1)
                h_b = np.maximum(0, sb @ self._W1 + self._b1)

                grad_W2 = grad_diff * (h_a - h_b).T
                grad_b2 = grad_diff * np.ones((1, 1))

                grad_h_a = grad_diff * self._W2.T
                grad_h_b = -grad_diff * self._W2.T
                grad_h_a[h_a <= 0] = 0
                grad_h_b[h_b <= 0] = 0

                grad_W1 = sa.T @ grad_h_a + sb.T @ grad_h_b
                grad_b1 = np.sum(grad_h_a + grad_h_b, axis=0)

                lr = self.cfg.lr
                self._W2 -= lr * grad_W2
                self._b2 -= lr * grad_b2.flatten()
                self._W1 -= lr * grad_W1
                self._b1 -= lr * grad_b1

            if epoch % 5 == 0:
                logger.debug("RLHF epoch %d loss %.4f", epoch, total_loss / len(self._comparisons))

        self._trained = True
        logger.info("RLHF reward model trained on %d comparisons", len(self._comparisons))
        return self

    def shape_reward(self, state: np.ndarray, base_reward: float) -> float:
        """Return shaped reward = base_reward + learned_reward (if trained).

        Falls back to base_reward when RLHF data is sparse.
        """
        if not self._trained:
            return float(base_reward)
        learned = self._reward_forward(state)
        return float(base_reward + learned)

    def get_preference_probability(
        self, state_a: np.ndarray, state_b: np.ndarray
    ) -> float:
        """Return P(A > B) according to learned reward model."""
        if not self._trained:
            return 0.5
        ra = self._reward_forward(state_a)
        rb = self._reward_forward(state_b)
        diff = (ra - rb) / self.cfg.preference_temperature
        return float(1.0 / (1.0 + np.exp(-diff)))
