"""Meta-agent for dynamic strategy selection.

US-017: Selects which sub-strategy to activate based on current regime
and recent performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetaAgentConfig:
    """Configuration for meta-strategy agent."""

    strategies: List[str] = field(default_factory=lambda: ["trend", "mean_reversion", "momentum"])
    top_k: int = 2
    performance_lookback: int = 20
    exploration_rate: float = 0.1


class MetaStrategyAgent:
    """Meta-agent that dynamically selects sub-strategies per regime.

    Observes regime + recent performance of each sub-strategy and
    activates the top-K strategies for the current scan cycle.
    """

    def __init__(self, config: Optional[MetaAgentConfig] = None):
        self.cfg = config or MetaAgentConfig()
        self._weights = {s: 1.0 / len(self.cfg.strategies) for s in self.cfg.strategies}
        self._performance_history: Dict[str, List[float]] = {s: [] for s in self.cfg.strategies}

    def observe(self, strategy: str, pnl: float) -> None:
        """Record performance of a strategy."""
        if strategy not in self._performance_history:
            self._performance_history[strategy] = []
        self._performance_history[strategy].append(pnl)
        # Keep only recent history
        self._performance_history[strategy] = self._performance_history[strategy][-self.cfg.performance_lookback :]

    def select(self, regime: str) -> Dict[str, Any]:
        """Select top-K strategies for current regime.

        Args:
            regime: Current market regime (LOW, NORMAL, HIGH, EXTREME)

        Returns:
            Dict with 'selected_strategies', 'weights', 'regime', 'uncertain'
        """
        # Compute average recent performance per strategy
        scores = {}
        for s in self.cfg.strategies:
            hist = self._performance_history.get(s, [])
            if len(hist) >= 3:
                scores[s] = np.mean(hist[-self.cfg.performance_lookback :])
            else:
                scores[s] = 0.0  # insufficient data

        # Epsilon-greedy exploration
        if np.random.rand() < self.cfg.exploration_rate:
            selected = np.random.choice(self.cfg.strategies, size=self.cfg.top_k, replace=False).tolist()
            weights = {s: 1.0 / self.cfg.top_k for s in selected}
            return {
                "selected_strategies": selected,
                "weights": weights,
                "regime": regime,
                "uncertain": True,
                "reason": "exploration",
            }

        # Select top-K by score
        sorted_strategies = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        top = sorted_strategies[: self.cfg.top_k]

        # If scores are too close, fallback to all strategies
        if len(top) >= 2 and abs(scores[top[0]] - scores[top[1]]) < 1e-6:
            selected = self.cfg.strategies.copy()
            weights = {s: 1.0 / len(selected) for s in selected}
            return {
                "selected_strategies": selected,
                "weights": weights,
                "regime": regime,
                "uncertain": True,
                "reason": "scores_too_close",
            }

        total_score = sum(scores[t] for t in top)
        if total_score <= 0:
            weights = {s: 1.0 / len(top) for s in top}
        else:
            weights = {s: scores[s] / total_score for s in top}
        return {
            "selected_strategies": top,
            "weights": weights,
            "regime": regime,
            "uncertain": False,
            "reason": "performance_based",
        }

    def get_weights(self) -> Dict[str, float]:
        return self._weights.copy()
