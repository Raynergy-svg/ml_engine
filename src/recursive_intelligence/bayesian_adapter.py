"""Online Bayesian adapter using grid-based Thompson Sampling.

For each (agent, regime) pair, maintains a posterior distribution over
possible HyperparameterSets. After each trade outcome, updates the posterior.
Uses Thompson Sampling to balance exploration vs exploitation.

Algorithm:
1. Grid: 5×5×5×5 = 625 possible hyperparameter combinations per agent/regime
2. Posterior: Dirichlet-like belief over grid points (represented as unnormalized weights)
3. Update: posterior[i] *= likelihood(outcome | hyperparam_set_i)
4. Normalize: posterior becomes probability distribution
5. Sample: draw from posterior (Thompson) or take argmax (greedy)

Time: <100ms per update for the 15-agent team (see ``ScannerAgentTeam._BASE_WEIGHTS``)
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .meta_parameters import HyperparameterSet, MetaParametersRegistry, DEFAULT_GRID, REGIMES

logger = logging.getLogger(__name__)


class OnlineBayesianAdapter:
    """Bayesian posterior over hyperparameter grid per (agent, regime).

    Posterior initialized as uniform (equal belief in all grid points).
    After each trade: posterior updated by likelihood of outcome given hyperparam set.

    Thompson Sampling: draw from posterior → explore more in uncertain regions.
    Greedy: take argmax → pure exploitation after sufficient data.
    """

    def __init__(
        self,
        registry: MetaParametersRegistry,
        grid: Optional[Dict[str, List[float]]] = None,
        exploration_rate: float = 0.3,  # fraction of updates that use Thompson sampling
    ):
        self._registry = registry
        self._grid = grid or DEFAULT_GRID
        self._exploration_rate = exploration_rate

        # Pre-compute all grid combinations: List[Tuple[float, ...]]
        self._param_names = list(self._grid.keys())
        self._grid_points: List[Tuple[float, ...]] = list(
            product(*[self._grid[k] for k in self._param_names])
        )
        n_points = len(self._grid_points)

        # Posterior per (agent, regime): shape [n_grid_points]
        # Initialized as uniform (all equal weight)
        self._posteriors: Dict[str, Dict[str, np.ndarray]] = {}
        self._update_counts: Dict[str, Dict[str, int]] = {}

        logger.info(
            "BayesianAdapter: initialized with %d grid points over %s",
            n_points, self._param_names
        )

    def _get_posterior(self, agent_name: str, regime: str) -> np.ndarray:
        """Get or initialize uniform posterior for (agent, regime)."""
        if agent_name not in self._posteriors:
            self._posteriors[agent_name] = {}
            self._update_counts[agent_name] = {}
        if regime not in self._posteriors[agent_name]:
            n = len(self._grid_points)
            self._posteriors[agent_name][regime] = np.ones(n, dtype=float) / n
            self._update_counts[agent_name][regime] = 0
        return self._posteriors[agent_name][regime]

    def _compute_responsibility(
        self,
        agent_score: float,
        trade_outcome: float,
    ) -> float:
        """How well did this agent's score predict the outcome?
        Returns 0.0 (perfect mismatch) to 1.0 (perfect alignment).
        """
        # Normalize agent_score to [-1, 1] if it's in [0, 1]
        if 0.0 <= agent_score <= 1.0:
            normalized_score = agent_score * 2.0 - 1.0
        else:
            normalized_score = np.clip(agent_score, -1.0, 1.0)

        # Responsibility: how aligned was the sign and magnitude?
        alignment = normalized_score * trade_outcome  # in [-1, 1]
        responsibility = (alignment + 1.0) / 2.0  # in [0, 1]
        return float(responsibility)

    def _likelihood(
        self,
        grid_point: Tuple[float, ...],
        trade_outcome: float,
        agent_score: float,
        responsibility: float,
    ) -> float:
        """P(outcome | this hyperparameter set).

        Key insight: higher confidence_threshold means agent only fires when
        very confident. If outcome was positive (win), we reward sets with
        higher thresholds (they would have filtered out noise).
        If outcome was negative (loss), we reward sets with lower thresholds
        (they would have caught the signal earlier for a safer stop).

        This is a heuristic approximation — true likelihood would require
        simulating the agent with these hyperparams, which is too expensive.
        """
        params = dict(zip(self._param_names, grid_point))

        likelihood = 1.0

        # confidence_threshold: high threshold → good when winning
        ct = params.get("confidence_threshold", 0.6)
        if trade_outcome > 0:
            # Winning trade: reward higher thresholds (quality filter worked)
            likelihood *= (0.5 + 0.5 * (ct - 0.3) / 0.6)
        else:
            # Losing trade: reward lower thresholds (earlier entry, earlier stop)
            likelihood *= (0.5 + 0.5 * (0.9 - ct) / 0.6)

        # voting_veto_power: when responsibility is high, reward higher veto power
        vvp = params.get("voting_veto_power", 1.0)
        if responsibility > 0.6:
            # Agent predicted well → reward giving it more power
            likelihood *= (0.7 + 0.3 * (vvp - 0.5) / 1.0)
        else:
            # Agent predicted poorly → reward reducing its power
            likelihood *= (0.7 + 0.3 * (1.5 - vvp) / 1.0)

        # Clip likelihood to [0.01, 10.0] to prevent numerical collapse
        return float(np.clip(likelihood, 0.01, 10.0))

    def update(
        self,
        agent_name: str,
        regime: str,
        agent_score: float,
        trade_outcome: float,  # -1.0 (loss) to +1.0 (win)
        recent_trades: Optional[List[Dict[str, float]]] = None,
    ) -> HyperparameterSet:
        """Update posterior and return best hyperparameter set for next scan.

        Args:
            agent_name: name of the agent (e.g., "trend", "momentum")
            regime: current market regime (NORMAL, VOLATILE, etc.)
            agent_score: what the agent predicted (0.0-1.0 or -1.0 to 1.0)
            trade_outcome: actual outcome (-1.0 = loss, +1.0 = win)
            recent_trades: last N trade outcomes for context (optional)

        Returns:
            Best HyperparameterSet for the next scan cycle
        """
        regime = regime.upper()
        posterior = self._get_posterior(agent_name, regime)
        responsibility = self._compute_responsibility(agent_score, trade_outcome)

        # Update posterior: multiply by likelihood, normalize
        for i, grid_point in enumerate(self._grid_points):
            likelihood = self._likelihood(grid_point, trade_outcome, agent_score, responsibility)
            posterior[i] *= likelihood

        # Normalize (prevent collapse)
        total = posterior.sum()
        if total < 1e-10:
            # Reset to uniform if collapsed
            posterior[:] = 1.0 / len(posterior)
            logger.warning("BayesianAdapter: posterior collapsed for %s/%s — reset to uniform", agent_name, regime)
        else:
            posterior /= total

        self._update_counts[agent_name][regime] = self._update_counts[agent_name].get(regime, 0) + 1

        # Select: Thompson sampling (explore) or greedy (exploit)
        n_updates = self._update_counts[agent_name][regime]
        use_thompson = (n_updates < 10) or (np.random.random() < self._exploration_rate)

        if use_thompson:
            # Sample from posterior (multinomial draw)
            idx = int(np.random.choice(len(self._grid_points), p=posterior))
        else:
            # Greedy: take argmax
            idx = int(np.argmax(posterior))

        best_point = self._grid_points[idx]
        params_dict = dict(zip(self._param_names, best_point))

        # Build HyperparameterSet and persist in registry
        new_params = HyperparameterSet(
            confidence_threshold=params_dict["confidence_threshold"],
            weight_decay_rate=params_dict["weight_decay_rate"],
            regime_switch_sensitivity=params_dict["regime_switch_sensitivity"],
            voting_veto_power=params_dict["voting_veto_power"],
        )
        self._registry.update_fitness(agent_name, regime, trade_outcome)
        self._registry.set(agent_name, regime, new_params)

        logger.debug(
            "BayesianAdapter: %s/%s updated (outcome=%.2f, resp=%.2f) → ct=%.2f vvp=%.2f",
            agent_name, regime, trade_outcome, responsibility,
            new_params.confidence_threshold, new_params.voting_veto_power
        )

        return new_params

    def get_best(self, agent_name: str, regime: str) -> HyperparameterSet:
        """Return current best hyperparams (greedy, no update)."""
        regime = regime.upper()
        posterior = self._get_posterior(agent_name, regime)
        idx = int(np.argmax(posterior))
        best_point = self._grid_points[idx]
        params_dict = dict(zip(self._param_names, best_point))
        return HyperparameterSet(
            confidence_threshold=params_dict["confidence_threshold"],
            weight_decay_rate=params_dict["weight_decay_rate"],
            regime_switch_sensitivity=params_dict["regime_switch_sensitivity"],
            voting_veto_power=params_dict["voting_veto_power"],
        )

    def reset_posterior(self, agent_name: str, regime: str) -> None:
        """Reset posterior to uniform (rollback)."""
        n = len(self._grid_points)
        if agent_name in self._posteriors and regime in self._posteriors[agent_name]:
            self._posteriors[agent_name][regime] = np.ones(n) / n
            self._update_counts[agent_name][regime] = 0
        logger.info("BayesianAdapter: reset posterior for %s/%s", agent_name, regime)
