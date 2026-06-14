"""Continual learning with episodic memory replay.

US-018: Updates models incrementally without catastrophic forgetting
using replay buffers and elastic weight consolidation (EWC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContinualLearnerConfig:
    """Configuration for continual learner."""

    memory_size: int = 500
    replay_ratio: float = 0.3
    ewc_lambda: float = 1.0
    epochs_per_update: int = 5
    forgetting_threshold: float = 0.05


class ContinualLearner:
    """Continual learner with episodic memory and EWC regularization.

    Stores representative past examples in episodic memory and replays
them alongside new data during incremental updates. Uses simplified
    EWC to protect important weights from catastrophic forgetting.
    """

    def __init__(self, config: Optional[ContinualLearnerConfig] = None):
        self.cfg = config or ContinualLearnerConfig()
        self._episodic_memory: List[Dict[str, Any]] = []
        self._task_count = 0
        self._ewc_importance: Optional[Dict[str, np.ndarray]] = None
        self._old_params: Optional[Dict[str, np.ndarray]] = None

    def add_to_memory(self, examples: List[Dict[str, Any]]) -> None:
        """Add examples to episodic memory (reservoir sampling)."""
        for ex in examples:
            if len(self._episodic_memory) < self.cfg.memory_size:
                self._episodic_memory.append(ex)
            else:
                # Reservoir sampling
                idx = np.random.randint(0, len(self._episodic_memory))
                self._episodic_memory[idx] = ex

    def prepare_training_batch(
        self,
        new_examples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Combine new examples with replay memory.

        Returns:
            (combined_batch, replay_batch)
        """
        n_replay = int(len(new_examples) * self.cfg.replay_ratio)
        if len(self._episodic_memory) > 0 and n_replay > 0:
            replay = [
                self._episodic_memory[i]
                for i in np.random.choice(len(self._episodic_memory), size=min(n_replay, len(self._episodic_memory)), replace=False)
            ]
        else:
            replay = []
        return new_examples + replay, replay

    def compute_ewc_penalty(
        self,
        current_params: Dict[str, np.ndarray],
    ) -> float:
        """Compute EWC penalty for parameter deviation from old task optimum.

        Returns:
            Scalar penalty value.
        """
        if self._ewc_importance is None or self._old_params is None:
            return 0.0
        penalty = 0.0
        for key in current_params:
            if key in self._ewc_importance and key in self._old_params:
                diff = current_params[key] - self._old_params[key]
                penalty += np.sum(self._ewc_importance[key] * (diff ** 2))
        return float(penalty) * self.cfg.ewc_lambda

    def update_ewc_importance(
        self,
        params: Dict[str, np.ndarray],
        gradients: Dict[str, np.ndarray],
    ) -> None:
        """Update EWC importance based on Fisher information approximation."""
        if self._ewc_importance is None:
            self._ewc_importance = {}
        for key in params:
            grad_sq = gradients.get(key, np.zeros_like(params[key])) ** 2
            if key in self._ewc_importance:
                self._ewc_importance[key] = 0.9 * self._ewc_importance[key] + 0.1 * grad_sq
            else:
                self._ewc_importance[key] = grad_sq
        self._old_params = {k: v.copy() for k, v in params.items()}
        self._task_count += 1
        logger.info("EWC importance updated for task %d", self._task_count)

    def get_forgetting_metric(self, old_validation_loss: float, current_validation_loss: float) -> float:
        """Return forgetting metric: increase in loss on old validation set."""
        return max(0.0, current_validation_loss - old_validation_loss)

    def get_memory_size(self) -> int:
        return len(self._episodic_memory)
