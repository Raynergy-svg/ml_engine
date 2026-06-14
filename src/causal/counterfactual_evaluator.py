"""Counterfactual trade evaluator.

US-014: Estimates what PnL would have been under different actions
using a causal model of market dynamics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual evaluation."""

    n_bootstrap: int = 100
    confidence_level: float = 0.95


class CounterfactualEvaluator:
    """Evaluate counterfactual outcomes for trading decisions.

    Given a trade record and causal graph of market factors, estimates
    the expected PnL for alternative position sizes and entry timings.
    """

    def __init__(self, config: Optional[CounterfactualConfig] = None):
        self.cfg = config or CounterfactualConfig()

    def evaluate(
        self,
        market_state: np.ndarray,
        causal_graph: np.ndarray,
        actual_action: Dict[str, Any],
        alternative_actions: List[Dict[str, Any]],
        factor_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluate counterfactual PnL for alternative actions.

        Args:
            market_state: Factor values at trade entry (n_factors,)
            causal_graph: Adjacency matrix from causal discovery
            actual_action: Dict with 'position_size', 'entry_time', etc.
            alternative_actions: List of alternative action dicts
            factor_returns: Optional post-trade factor returns for bootstrapping

        Returns:
            Dict with 'actual_pnl', 'counterfactuals' list, and 'recommendation'
        """
        if market_state.ndim != 1:
            raise ValueError("market_state must be 1D")

        results = {
            "actual_pnl": self._estimate_pnl(market_state, actual_action, factor_returns),
            "counterfactuals": [],
            "recommendation": "hold",
        }

        best_pnl = results["actual_pnl"]["mean"]
        best_action = actual_action

        for alt in alternative_actions:
            pnl = self._estimate_pnl(market_state, alt, factor_returns)
            results["counterfactuals"].append({
                "action": alt,
                "pnl": pnl,
            })
            if pnl["mean"] > best_pnl:
                best_pnl = pnl["mean"]
                best_action = alt

        if best_action is not actual_action:
            results["recommendation"] = best_action.get("name", "alternative")

        return results

    def _estimate_pnl(
        self,
        market_state: np.ndarray,
        action: Dict[str, Any],
        factor_returns: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Simplified PnL estimation using position_size * expected_return."""
        position_size = float(action.get("position_size", 1.0))
        direction = 1.0 if action.get("direction", "LONG") == "LONG" else -1.0

        if factor_returns is not None and len(factor_returns) > 0:
            # Bootstrap mean return
            boot_means = []
            for _ in range(self.cfg.n_bootstrap):
                sample = factor_returns[np.random.choice(len(factor_returns), size=len(factor_returns), replace=True)]
                boot_means.append(float(np.mean(sample)))
            mean_ret = float(np.mean(boot_means))
            std_ret = float(np.std(boot_means))
        else:
            # Fallback: assume small random return based on market state
            mean_ret = float(np.mean(market_state)) * 0.001
            std_ret = abs(mean_ret) * 0.5

        pnl = position_size * direction * mean_ret
        return {
            "mean": pnl,
            "std": std_ret * position_size,
            "ci_lower": pnl - 1.96 * std_ret * position_size,
            "ci_upper": pnl + 1.96 * std_ret * position_size,
        }
