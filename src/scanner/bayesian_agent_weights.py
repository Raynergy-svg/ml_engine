"""
Bayesian Agent Weight Updater using Thompson Sampling.

Implements regime-conditional Beta distributions for 12 specialist agents in FX trading.
Uses Thompson Sampling for weight selection and Bayesian updates for learning.

Key features:
- Thompson Sampling: sample θ_i ~ Beta(α_i, β_i), normalize to get weights
- Regime-conditional: separate Beta(α,β) per agent per regime (12 × 4 = 48 distributions)
- Partial credit: score-proportional updates for agents on winning/losing trades
- Weight decay: periodic multiplicative decay to handle non-stationarity
- Epsilon-greedy: exploration probability ε that decays over time
- Atomic persistence: JSON save/load with .tmp + os.rename pattern
- Thread-safe sampling: no shared mutable state during sampling operation

Trade outcomes feed RL updates to agent weights:
- Win: agents with score > 0.5 get α += score
       agents with score < 0.5 get β += (1 - score)
- Loss: agents with score > 0.5 get β += (1 - score)
        agents with score < 0.5 get α += score
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BayesianWeightsConfig",
    "WeightSample",
    "BayesianAgentWeights",
    "create_default_bayesian_weights",
    "create_exploration_heavy_weights",
    "create_exploitation_heavy_weights",
]

# Default core-12 agents (subset of the 15-agent team; extended agents
# order_flow / trader_readiness / devil_advocate are not tracked here).
# Source of truth for the full roster: ``ScannerAgentTeam._BASE_WEIGHTS``.
DEFAULT_AGENT_NAMES = [
    "trend",
    "mean_reversion",
    "volatility",
    "risk_sentinel",
    "uncertainty",
    "execution_quality",
    "momentum",
    "news_risk",
    "multi_timeframe",
    "pair_performance",
    "session_timing",
    "support_resistance",
]

# Market regimes
DEFAULT_REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]


@dataclass
class BayesianWeightsConfig:
    """Configuration for Bayesian agent weights."""

    agent_names: List[str] = field(default_factory=lambda: DEFAULT_AGENT_NAMES.copy())
    regime_names: List[str] = field(default_factory=lambda: DEFAULT_REGIME_NAMES.copy())
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    decay_factor: float = 0.995
    decay_interval: int = 20
    epsilon_initial: float = 0.15
    epsilon_decay: float = 0.9995
    epsilon_min: float = 0.02
    partial_credit: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.agent_names) == 0:
            raise ValueError("agent_names cannot be empty")
        if len(self.regime_names) == 0:
            raise ValueError("regime_names cannot be empty")
        if not (0 < self.prior_alpha and self.prior_alpha < 100):
            raise ValueError("prior_alpha must be in (0, 100)")
        if not (0 < self.prior_beta and self.prior_beta < 100):
            raise ValueError("prior_beta must be in (0, 100)")
        if not (0.9 < self.decay_factor < 1.0):
            raise ValueError("decay_factor must be in (0.9, 1.0)")
        if self.decay_interval < 1:
            raise ValueError("decay_interval must be >= 1")
        if not (0 <= self.epsilon_initial <= 1):
            raise ValueError("epsilon_initial must be in [0, 1]")
        if not (0 <= self.epsilon_decay <= 1):
            raise ValueError("epsilon_decay must be in [0, 1]")
        if not (0 <= self.epsilon_min <= self.epsilon_initial):
            raise ValueError("epsilon_min must be <= epsilon_initial")


@dataclass
class WeightSample:
    """Result of sampling agent weights."""

    weights: Dict[str, float]
    regime: str
    exploration: bool
    raw_samples: Dict[str, float]
    epsilon: float

    def __post_init__(self) -> None:
        """Validate that weights sum to ~1."""
        total = sum(self.weights.values())
        if not (0.99 < total < 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class BayesianAgentWeights:
    """Bayesian agent weight updater using Thompson Sampling with Beta distributions."""

    def __init__(self, config: Optional[BayesianWeightsConfig] = None) -> None:
        """Initialize Bayesian weights.

        Args:
            config: BayesianWeightsConfig instance. Uses defaults if None.
        """
        self.config = config or BayesianWeightsConfig()

        # Validate agents and regimes
        if len(self.config.agent_names) == 0:
            raise ValueError("agent_names cannot be empty")
        if len(self.config.regime_names) == 0:
            raise ValueError("regime_names cannot be empty")

        # Initialize Beta distributions: (regime, agent) -> (alpha, beta)
        self._distributions: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._reset_distributions()

        # Bookkeeping
        self._update_count = 0
        self._epsilon = self.config.epsilon_initial

        logger.info(
            f"Initialized BayesianAgentWeights: {len(self.config.agent_names)} agents, "
            f"{len(self.config.regime_names)} regimes, "
            f"{len(self.config.agent_names) * len(self.config.regime_names)} distributions"
        )

    def _reset_distributions(self) -> None:
        """Initialize all Beta distributions to prior."""
        for regime in self.config.regime_names:
            for agent in self.config.agent_names:
                self._distributions[(regime, agent)] = (
                    self.config.prior_alpha,
                    self.config.prior_beta,
                )

    def sample_weights(self, regime: str = "NORMAL") -> WeightSample:
        """Sample agent weights via Thompson Sampling.

        Algorithm:
        1. Decide exploration vs exploitation via epsilon-greedy
        2. If exploration: return uniform weights
        3. If exploitation: sample θ_i ~ Beta(α_i, β_i) for each agent
        4. Normalize: w_i = θ_i / sum(θ_j)

        Args:
            regime: Market regime name (must be in config.regime_names)

        Returns:
            WeightSample with normalized weights, regime, exploration flag

        Raises:
            ValueError: if regime not in config.regime_names
        """
        if regime not in self.config.regime_names:
            raise ValueError(
                f"Unknown regime '{regime}'. Must be one of {self.config.regime_names}"
            )

        # Epsilon-greedy decision
        is_exploration = np.random.rand() < self._epsilon

        if is_exploration:
            # Uniform weights (exploration)
            uniform_weight = 1.0 / len(self.config.agent_names)
            weights = {agent: uniform_weight for agent in self.config.agent_names}
            raw_samples = {agent: uniform_weight for agent in self.config.agent_names}
            logger.debug(f"Exploration sample (ε={self._epsilon:.4f})")
        else:
            # Thompson sample
            raw_samples = {}
            for agent in self.config.agent_names:
                alpha, beta = self._distributions[(regime, agent)]
                theta = np.random.beta(alpha, beta)
                raw_samples[agent] = theta

            # Normalize
            total = sum(raw_samples.values())
            weights = {agent: theta / total for agent, theta in raw_samples.items()}

        return WeightSample(
            weights=weights,
            regime=regime,
            exploration=is_exploration,
            raw_samples=raw_samples,
            epsilon=self._epsilon,
        )

    def update(
        self, agent_scores: Dict[str, float], outcome: str, regime: str = "NORMAL"
    ) -> None:
        """Update Beta distributions based on trade outcome.

        Bayesian update logic:
        - Win outcome:
          - Agents with score > 0.5: α += score (correct vote)
          - Agents with score < 0.5: β += (1 - score) (incorrect vote)
        - Loss outcome:
          - Agents with score > 0.5: β += (1 - score) (incorrect vote)
          - Agents with score < 0.5: α += score (correct vote on loss)

        Args:
            agent_scores: {agent_name: float score in [0, 1]}
            outcome: "win" or "loss"
            regime: Market regime name

        Raises:
            ValueError: if unknown agent, invalid scores, or invalid outcome
        """
        if outcome not in ("win", "loss"):
            raise ValueError(f"outcome must be 'win' or 'loss', got '{outcome}'")

        if regime not in self.config.regime_names:
            raise ValueError(
                f"Unknown regime '{regime}'. Must be one of {self.config.regime_names}"
            )

        # Validate agent_scores
        unknown_agents = set(agent_scores.keys()) - set(self.config.agent_names)
        if unknown_agents:
            raise ValueError(f"Unknown agents: {unknown_agents}")

        if self.config.partial_credit:
            # Score-proportional partial credit
            for agent in self.config.agent_names:
                score = agent_scores.get(agent, 0.0)

                # Validate score
                if not (0 <= score <= 1):
                    raise ValueError(
                        f"Agent score must be in [0, 1], got {score} for {agent}"
                    )

                alpha, beta = self._distributions[(regime, agent)]

                if outcome == "win":
                    # Higher score → more confidence → increase α
                    if score > 0.5:
                        alpha += score
                    else:
                        beta += 1 - score
                else:  # loss
                    # Higher score → more confident wrong answer → increase β
                    if score > 0.5:
                        beta += 1 - score
                    else:
                        alpha += score

                self._distributions[(regime, agent)] = (alpha, beta)
        else:
            # Binary: score > 0.5 is a vote for the outcome
            for agent in self.config.agent_names:
                score = agent_scores.get(agent, 0.0)

                if not (0 <= score <= 1):
                    raise ValueError(
                        f"Agent score must be in [0, 1], got {score} for {agent}"
                    )

                alpha, beta = self._distributions[(regime, agent)]

                if outcome == "win":
                    if score > 0.5:
                        alpha += 1
                    else:
                        beta += 1
                else:  # loss
                    if score > 0.5:
                        beta += 1
                    else:
                        alpha += 1

                self._distributions[(regime, agent)] = (alpha, beta)

        self._update_count += 1

        # Decay epsilon
        self._epsilon = max(
            self.config.epsilon_min,
            self._epsilon * self.config.epsilon_decay,
        )

        # Apply weight decay if interval reached
        if self._update_count % self.config.decay_interval == 0:
            self.apply_decay()

        logger.debug(
            f"Updated weights: outcome={outcome}, regime={regime}, "
            f"epsilon={self._epsilon:.4f}, update_count={self._update_count}"
        )

    def apply_decay(self) -> None:
        """Apply multiplicative decay to all α, β to handle non-stationarity.

        Decay formula: α_new = max(1.0, α_old * decay_factor)
        This allows the system to forget old observations and adapt to regime changes.
        """
        for key in self._distributions:
            alpha, beta = self._distributions[key]
            alpha = max(1.0, alpha * self.config.decay_factor)
            beta = max(1.0, beta * self.config.decay_factor)
            self._distributions[key] = (alpha, beta)

        logger.info(
            f"Applied weight decay (factor={self.config.decay_factor}) to all "
            f"{len(self._distributions)} distributions"
        )

    def get_agent_stats(self, regime: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get statistics for agents.

        Returns mean (α/(α+β)) and variance for Beta distributions.

        Args:
            regime: If provided, return stats only for this regime.
                   If None, aggregate across all regimes.

        Returns:
            {agent_name: {alpha, beta, mean, variance, regime}}
        """
        stats = {}

        if regime is not None:
            if regime not in self.config.regime_names:
                raise ValueError(
                    f"Unknown regime '{regime}'. Must be one of {self.config.regime_names}"
                )
            regimes = [regime]
        else:
            regimes = self.config.regime_names

        for agent in self.config.agent_names:
            alphas = []
            betas = []

            for reg in regimes:
                alpha, beta = self._distributions[(reg, agent)]
                alphas.append(alpha)
                betas.append(beta)

            # Aggregate: use mean of alpha/beta across regimes
            alpha_agg = np.mean(alphas)
            beta_agg = np.mean(betas)

            mean = alpha_agg / (alpha_agg + beta_agg)
            # Variance of Beta(α, β) = αβ / ((α+β)²(α+β+1))
            variance = (
                (alpha_agg * beta_agg)
                / ((alpha_agg + beta_agg) ** 2 * (alpha_agg + beta_agg + 1))
            )

            stats[agent] = {
                "alpha": alpha_agg,
                "beta": beta_agg,
                "mean": mean,
                "variance": variance,
            }

        return stats

    def get_top_agents(self, regime: str = "NORMAL", n: int = 5) -> List[str]:
        """Get top N agents by mean performance in given regime.

        Args:
            regime: Market regime
            n: Number of top agents to return

        Returns:
            List of agent names, sorted by mean performance (highest first)
        """
        if regime not in self.config.regime_names:
            raise ValueError(
                f"Unknown regime '{regime}'. Must be one of {self.config.regime_names}"
            )

        agent_means = []
        for agent in self.config.agent_names:
            alpha, beta = self._distributions[(regime, agent)]
            mean = alpha / (alpha + beta)
            agent_means.append((agent, mean))

        # Sort by mean (descending)
        agent_means.sort(key=lambda x: x[1], reverse=True)

        return [agent for agent, _ in agent_means[:n]]

    def save_state(self, path: str) -> None:
        """Save state to JSON file atomically.

        Uses .tmp + os.rename pattern to avoid partial writes.
        JSON is formatted with indent=2 and sort_keys=True for readability.

        Args:
            path: File path to save state
        """
        state = {
            "version": 1,
            "update_count": self._update_count,
            "epsilon": self._epsilon,
            "distributions": {
                f"{regime}:{agent}": {"alpha": alpha, "beta": beta}
                for (regime, agent), (alpha, beta) in self._distributions.items()
            },
        }

        # Atomic write: write to .tmp, then rename
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file in same directory to ensure same filesystem
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(path_obj.parent),
                prefix=".tmp_",
                suffix=".json",
            )

            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(state, f, indent=2, sort_keys=True)
                os.rename(tmp_path, path)
                logger.info(f"Saved Bayesian weights state to {path}")
            except Exception as e:
                os.close(tmp_fd)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise e

        except Exception as e:
            logger.error(f"Failed to save Bayesian weights state to {path}: {e}")
            raise

    def load_state(self, path: str) -> None:
        """Load state from JSON file with graceful error handling.

        If file missing or corrupted, logs warning and reinitializes to prior.

        Args:
            path: File path to load state from
        """
        try:
            if not os.path.exists(path):
                logger.warning(
                    f"State file {path} not found. Initializing to prior."
                )
                self._reset_distributions()
                return

            with open(path, "r") as f:
                state = json.load(f)

            state = self._normalize_loaded_state(state)

            # Restore distributions
            self._distributions = {}
            for key_str, dist in state["distributions"].items():
                try:
                    regime, agent = key_str.split(":", 1)
                    alpha = dist.get("alpha")
                    beta = dist.get("beta")

                    if alpha is None or beta is None:
                        raise ValueError(f"Invalid distribution entry: {key_str}")

                    self._distributions[(regime, agent)] = (float(alpha), float(beta))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid distribution entry {key_str}: {e}")

            self._fill_missing_distributions()

            self._update_count = state.get("update_count", 0)
            self._epsilon = state.get("epsilon", self.config.epsilon_initial)

            logger.info(
                f"Loaded Bayesian weights state from {path} "
                f"(update_count={self._update_count}, epsilon={self._epsilon:.4f})"
            )

            # Normalize older schemas in place so future loads are clean.
            if state.get("version", 1) != 1 or len(self._distributions) != len(self.config.agent_names) * len(self.config.regime_names):
                try:
                    self.save_state(path)
                except Exception:
                    logger.debug("Failed to normalize Bayesian state in place: %s", path)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON state file {path}: {e}")
            logger.warning("Reinitializing to prior")
            self._quarantine_bad_state_file(path, reason="json")
            self.reset()
        except Exception as e:
            logger.error(f"Unexpected error loading state from {path}: {e}")
            logger.warning("Reinitializing to prior")
            self._quarantine_bad_state_file(path, reason="schema")
            self.reset()

    def _normalize_loaded_state(self, state: Any) -> Dict[str, Any]:
        """Normalize/migrate persisted state into the current schema."""
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")

        if "distributions" in state:
            if "version" not in state:
                state["version"] = 1
            return state

        # Backward-compatibility: older snapshots may persist flat alpha/beta maps
        # under keys like `weights`, `stats`, or regime->agent nests.
        migrated = self._migrate_legacy_state(state)
        if migrated is not None:
            logger.warning("Migrated legacy Bayesian weights state to current schema")
            return migrated

        raise ValueError("State missing 'distributions' field")

    def _migrate_legacy_state(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to recover older persisted state layouts."""
        candidate_maps = [
            state.get("weights"),
            state.get("stats"),
            state.get("regimes"),
            state,
        ]

        distributions: Dict[str, Dict[str, float]] = {}
        for candidate in candidate_maps:
            if not isinstance(candidate, dict):
                continue

            for key, value in candidate.items():
                # Already normalized format under a nested map.
                if isinstance(value, dict) and "alpha" in value and "beta" in value and ":" in str(key):
                    distributions[str(key)] = {
                        "alpha": float(value["alpha"]),
                        "beta": float(value["beta"]),
                    }
                    continue

                # Nested regime -> agent -> {alpha,beta}
                if str(key) in self.config.regime_names and isinstance(value, dict):
                    for agent, dist in value.items():
                        if (
                            isinstance(dist, dict)
                            and "alpha" in dist
                            and "beta" in dist
                            and str(agent) in self.config.agent_names
                        ):
                            distributions[f"{key}:{agent}"] = {
                                "alpha": float(dist["alpha"]),
                                "beta": float(dist["beta"]),
                            }

        if not distributions:
            return None

        return {
            "version": int(state.get("version", 1) or 1),
            "update_count": int(state.get("update_count", 0) or 0),
            "epsilon": float(state.get("epsilon", self.config.epsilon_initial) or self.config.epsilon_initial),
            "distributions": distributions,
        }

    def _fill_missing_distributions(self) -> None:
        """Ensure every configured regime/agent pair has a distribution."""
        for regime in self.config.regime_names:
            for agent in self.config.agent_names:
                self._distributions.setdefault(
                    (regime, agent),
                    (self.config.prior_alpha, self.config.prior_beta),
                )

    def _quarantine_bad_state_file(self, path: str, *, reason: str) -> None:
        """Best-effort move aside of unreadable legacy/corrupt state."""
        try:
            if not os.path.exists(path):
                return
            backup = f"{path}.bad.{reason}.{int(time.time())}.json"
            os.rename(path, backup)
            logger.warning("Moved unreadable Bayesian state to %s", backup)
        except Exception:
            logger.debug("Failed to quarantine bad Bayesian state file: %s", path)

    def reset(self) -> None:
        """Reset all distributions to prior and bookkeeping to initial state."""
        self._reset_distributions()
        self._update_count = 0
        self._epsilon = self.config.epsilon_initial
        logger.info("Reset Bayesian weights to prior")


def create_default_bayesian_weights() -> BayesianAgentWeights:
    """Factory: Create Bayesian weights with balanced exploration/exploitation.

    Config: ε₀=0.15, min_ε=0.02, decay=0.9995
    """
    return BayesianAgentWeights(BayesianWeightsConfig())


def create_exploration_heavy_weights() -> BayesianAgentWeights:
    """Factory: Create Bayesian weights emphasizing exploration.

    Config: ε₀=0.30, min_ε=0.05, decay=0.99
    Useful for discovering new agent patterns.
    """
    config = BayesianWeightsConfig(
        epsilon_initial=0.30,
        epsilon_min=0.05,
        epsilon_decay=0.99,
    )
    return BayesianAgentWeights(config)


def create_exploitation_heavy_weights() -> BayesianAgentWeights:
    """Factory: Create Bayesian weights emphasizing exploitation.

    Config: ε₀=0.05, min_ε=0.01, decay=0.9998
    Useful for converging to proven agent weights.
    """
    config = BayesianWeightsConfig(
        epsilon_initial=0.05,
        epsilon_min=0.01,
        epsilon_decay=0.9998,
    )
    return BayesianAgentWeights(config)
