"""
Pair-Specific Bayesian Agent Weights — Phase 53 (US-329).

Extends BayesianAgentWeights with pair-level Beta distributions.
12 agents × 12 pairs = 144 additional distributions, blended with
global regime weights:

    final_weight = 0.6 × regime_weight + 0.4 × pair_weight

Warm-start: pair needs 5 trades before activating pair-specific weights;
falls back to global regime weight otherwise.

Epsilon-decay exploration per-pair (ε₀=0.15, decay=0.9995 per pair).

Persistence: trained_data/agent_weights_by_pair.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default 12 agents from the system
DEFAULT_AGENT_NAMES = [
    "trend", "mean_reversion", "volatility", "risk_sentinel",
    "uncertainty", "execution_quality", "momentum", "news_risk",
    "multi_timeframe", "pair_performance", "session_timing", "support_resistance",
]

# Default 12 major/cross pairs
DEFAULT_PAIR_NAMES = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD",
]

_DEFAULT_PERSISTENCE_PATH = "trained_data/agent_weights_by_pair.json"


@dataclass
class PairWeightsConfig:
    """Configuration for pair-specific Bayesian weights."""
    agent_names: List[str] = field(default_factory=lambda: DEFAULT_AGENT_NAMES.copy())
    pair_names: List[str] = field(default_factory=lambda: DEFAULT_PAIR_NAMES.copy())
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    regime_blend: float = 0.6   # Weight on global regime weights
    pair_blend: float = 0.4     # Weight on pair-specific weights
    warm_start_min_trades: int = 5  # Min trades before pair weights activate
    epsilon_initial: float = 0.15
    epsilon_decay: float = 0.9995
    epsilon_min: float = 0.02
    persistence_path: str = _DEFAULT_PERSISTENCE_PATH
    partial_credit: bool = True


@dataclass
class PairWeightSample:
    """Result of sampling blended pair-specific weights."""
    weights: Dict[str, float]           # Final blended weights
    pair: str
    regime: str
    pair_active: bool                    # Whether pair-specific weights were used
    pair_trade_count: int                # Trades seen for this pair
    regime_weights: Dict[str, float]     # Regime-level weights component
    pair_weights: Dict[str, float]       # Pair-level weights component
    epsilon: float


class PairSpecificBayesianWeights:
    """Pair-level Bayesian agent weights blended with global regime weights.

    For each (pair, agent) combination, maintains a Beta(α, β) distribution.
    Blends these with regime-level weights via configurable blend ratio.
    """

    def __init__(self, config: Optional[PairWeightsConfig] = None):
        self.config = config or PairWeightsConfig()

        # Pair-level distributions: (pair, agent) → (alpha, beta)
        self._pair_distributions: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._init_pair_distributions()

        # Per-pair bookkeeping
        self._pair_trade_counts: Dict[str, int] = {p: 0 for p in self.config.pair_names}
        self._pair_epsilon: Dict[str, float] = {
            p: self.config.epsilon_initial for p in self.config.pair_names
        }

        self._loaded = False
        self._load_state()

    def _init_pair_distributions(self) -> None:
        """Initialize all pair-level distributions to prior."""
        for pair in self.config.pair_names:
            for agent in self.config.agent_names:
                self._pair_distributions[(pair, agent)] = (
                    self.config.prior_alpha,
                    self.config.prior_beta,
                )

    def sample_pair_weights(
        self,
        pair: str,
        regime_weights: Optional[Dict[str, float]] = None,
        regime: str = "NORMAL",
    ) -> PairWeightSample:
        """Sample blended weights for a specific pair.

        If pair has < warm_start_min_trades, returns regime_weights only.
        Otherwise blends: 0.6 × regime_weight + 0.4 × pair_weight.

        Args:
            pair: Instrument name
            regime_weights: Pre-sampled regime-level weights dict {agent: weight}
            regime: Regime name (for metadata)

        Returns:
            PairWeightSample with blended weights
        """
        c = self.config
        pair_trade_count = self._pair_trade_counts.get(pair, 0)

        # If no regime weights provided, use uniform
        if regime_weights is None:
            uniform = 1.0 / len(c.agent_names)
            regime_weights = {a: uniform for a in c.agent_names}

        # If pair not warm-started, return regime weights directly
        if pair_trade_count < c.warm_start_min_trades:
            return PairWeightSample(
                weights=dict(regime_weights),
                pair=pair,
                regime=regime,
                pair_active=False,
                pair_trade_count=pair_trade_count,
                regime_weights=dict(regime_weights),
                pair_weights={},
                epsilon=self._pair_epsilon.get(pair, c.epsilon_initial),
            )

        # Epsilon-greedy per-pair
        eps = self._pair_epsilon.get(pair, c.epsilon_initial)
        is_exploration = np.random.rand() < eps

        if is_exploration:
            uniform = 1.0 / len(c.agent_names)
            pair_raw = {a: uniform for a in c.agent_names}
        else:
            # Thompson sample from pair-level distributions
            pair_raw = {}
            for agent in c.agent_names:
                key = (pair, agent)
                if key in self._pair_distributions:
                    alpha, beta = self._pair_distributions[key]
                else:
                    alpha, beta = c.prior_alpha, c.prior_beta
                pair_raw[agent] = float(np.random.beta(alpha, beta))

        # Normalize pair weights
        total = sum(pair_raw.values())
        if total > 0:
            pair_weights = {a: v / total for a, v in pair_raw.items()}
        else:
            uniform = 1.0 / len(c.agent_names)
            pair_weights = {a: uniform for a in c.agent_names}

        # Blend: regime × 0.6 + pair × 0.4
        blended = {}
        for agent in c.agent_names:
            rw = regime_weights.get(agent, 0.0)
            pw = pair_weights.get(agent, 0.0)
            blended[agent] = c.regime_blend * rw + c.pair_blend * pw

        # Re-normalize blended weights
        blend_total = sum(blended.values())
        if blend_total > 0:
            blended = {a: v / blend_total for a, v in blended.items()}

        return PairWeightSample(
            weights=blended,
            pair=pair,
            regime=regime,
            pair_active=True,
            pair_trade_count=pair_trade_count,
            regime_weights=dict(regime_weights),
            pair_weights=pair_weights,
            epsilon=eps,
        )

    def update(
        self,
        pair: str,
        agent_scores: Dict[str, float],
        outcome: str,
    ) -> None:
        """Update pair-specific Beta distributions after a trade closes.

        Also increments pair trade count and decays epsilon.

        Args:
            pair: Instrument name
            agent_scores: {agent_name: score in [0,1]}
            outcome: "win" or "loss"
        """
        if outcome not in ("win", "loss"):
            raise ValueError(f"outcome must be 'win' or 'loss', got '{outcome}'")

        # Auto-add pair if not in config
        if pair not in self._pair_trade_counts:
            self._pair_trade_counts[pair] = 0
            self._pair_epsilon[pair] = self.config.epsilon_initial
            for agent in self.config.agent_names:
                self._pair_distributions[(pair, agent)] = (
                    self.config.prior_alpha, self.config.prior_beta,
                )

        c = self.config
        for agent in c.agent_names:
            score = agent_scores.get(agent, 0.0)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            key = (pair, agent)
            alpha, beta = self._pair_distributions.get(
                key, (c.prior_alpha, c.prior_beta)
            )

            if c.partial_credit:
                if outcome == "win":
                    if score > 0.5:
                        alpha += score
                    else:
                        beta += 1 - score
                else:  # loss
                    if score > 0.5:
                        beta += 1 - score
                    else:
                        alpha += score
            else:
                # Binary credit
                if outcome == "win":
                    if score > 0.5:
                        alpha += 1
                    else:
                        beta += 1
                else:
                    if score > 0.5:
                        beta += 1
                    else:
                        alpha += 1

            self._pair_distributions[key] = (alpha, beta)

        # Increment trade count
        self._pair_trade_counts[pair] = self._pair_trade_counts.get(pair, 0) + 1

        # Decay epsilon for this pair
        old_eps = self._pair_epsilon.get(pair, c.epsilon_initial)
        self._pair_epsilon[pair] = max(c.epsilon_min, old_eps * c.epsilon_decay)

        logger.debug(
            "Phase 53 (US-329): Updated pair weights for %s (%s) — "
            "trade_count=%d, epsilon=%.4f",
            pair, outcome,
            self._pair_trade_counts[pair],
            self._pair_epsilon[pair],
        )

    def get_pair_stats(self, pair: str) -> Dict[str, Dict[str, float]]:
        """Get agent stats for a specific pair.

        Returns:
            {agent: {alpha, beta, mean, variance}}
        """
        stats = {}
        for agent in self.config.agent_names:
            key = (pair, agent)
            alpha, beta = self._pair_distributions.get(
                key, (self.config.prior_alpha, self.config.prior_beta)
            )
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            variance = (
                (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                if (alpha + beta + 1) > 0 else 0.0
            )
            stats[agent] = {
                "alpha": round(alpha, 3),
                "beta": round(beta, 3),
                "mean": round(mean, 4),
                "variance": round(variance, 6),
            }
        return stats

    def get_pair_trade_count(self, pair: str) -> int:
        """Get trade count for a pair."""
        return self._pair_trade_counts.get(pair, 0)

    def is_pair_active(self, pair: str) -> bool:
        """Check if pair has enough trades for pair-specific weights."""
        return self.get_pair_trade_count(pair) >= self.config.warm_start_min_trades

    def save_state(self) -> None:
        """Persist state to disk with retry (US-342)."""
        path = self.config.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            state = {
                "version": 1,
                "pair_distributions": {
                    f"{pair}:{agent}": {"alpha": alpha, "beta": beta}
                    for (pair, agent), (alpha, beta) in self._pair_distributions.items()
                },
                "pair_trade_counts": dict(self._pair_trade_counts),
                "pair_epsilon": {k: round(v, 6) for k, v in self._pair_epsilon.items()},
            }
            try:
                from src.scanner.safe_io import resilient_save
                resilient_save(path, state, context="pair_bayesian_weights")
            except ImportError:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(state, f, indent=2, sort_keys=True)
                os.rename(tmp_path, path)
            logger.info("Phase 53 (US-329): Saved pair-specific weights to %s", path)
        except Exception as e:
            logger.error("Phase 53 (US-329): Failed to save pair weights: %s", e)

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        path = self.config.persistence_path
        if not os.path.exists(path):
            self._loaded = True
            return
        try:
            with open(path, "r") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                self._loaded = True
                return

            # Load distributions
            raw_dists = state.get("pair_distributions", {})
            for key_str, dist in raw_dists.items():
                try:
                    pair, agent = key_str.split(":", 1)
                    alpha = float(dist.get("alpha", self.config.prior_alpha))
                    beta = float(dist.get("beta", self.config.prior_beta))
                    self._pair_distributions[(pair, agent)] = (alpha, beta)
                except (ValueError, KeyError):
                    continue

            # Load trade counts
            raw_counts = state.get("pair_trade_counts", {})
            for pair, count in raw_counts.items():
                self._pair_trade_counts[pair] = int(count)

            # Load epsilon
            raw_eps = state.get("pair_epsilon", {})
            for pair, eps in raw_eps.items():
                self._pair_epsilon[pair] = float(eps)

            self._loaded = True
            logger.info(
                "Phase 53 (US-329): Loaded pair weights — %d distributions, %d pairs with trades",
                len(self._pair_distributions),
                sum(1 for c in self._pair_trade_counts.values() if c > 0),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 53 (US-329): Could not load pair weights: %s", e)
            self._loaded = True

    def reset(self) -> None:
        """Reset all pair distributions and bookkeeping."""
        self._init_pair_distributions()
        self._pair_trade_counts = {p: 0 for p in self.config.pair_names}
        self._pair_epsilon = {
            p: self.config.epsilon_initial for p in self.config.pair_names
        }
