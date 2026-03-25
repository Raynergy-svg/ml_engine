"""Bandit-based online model selection using EXP3 algorithm.

US-093: Dynamically select which ensemble member to trust using EXP3
multi-armed bandit. Each model (TCN/Ridge/RF) is an arm. Reward =
recent trade P/L contribution. Explore-exploit balance adapts to
concept drift. Replaces static ensemble weighting for per-scan model
routing. Based on 2024 OMS-MAB research.

EXP3 Algorithm:
- Maintain weight vector W over K arms (models)
- At each round: sample arm proportional to (1-gamma)*W/sum(W) + gamma/K
- Observe reward r for chosen arm
- Update: W[arm] *= exp(gamma * r_hat / K) where r_hat = r / p(arm)
- gamma controls exploration rate (higher = more uniform sampling)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BANDIT_PATH = _PROJECT_ROOT / "trained_data" / "model_bandit.json"

# Default models (arms)
DEFAULT_MODELS = ["TCN", "Ridge", "RF"]

# EXP3 defaults
DEFAULT_GAMMA = 0.10        # Exploration rate: 10% uniform exploration
DEFAULT_MIN_GAMMA = 0.02    # Minimum exploration (never fully exploit)
DEFAULT_GAMMA_DECAY = 0.999  # Slow decay of exploration over time


@dataclass
class SelectionResult:
    """Result of bandit model selection."""

    selected_model: str = ""
    is_exploration: bool = False
    selection_probability: float = 0.0
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "selected_model": self.selected_model,
            "is_exploration": self.is_exploration,
            "selection_probability": round(self.selection_probability, 6),
            "model_weights": {k: round(v, 6) for k, v in self.model_weights.items()},
            "model_probabilities": {k: round(v, 6) for k, v in self.model_probabilities.items()},
        }


class ModelBanditSelector:
    """EXP3 multi-armed bandit for online model selection.

    Each ensemble model (TCN, Ridge, RF) is an arm. The bandit learns
    which model to trust based on realized trade P/L, adapting to
    concept drift through continuous exploration.

    Usage:
        bandit = ModelBanditSelector()
        result = bandit.select_model()
        # ... use result.selected_model for prediction weighting ...
        bandit.update_reward(result.selected_model, realized_pnl)
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        gamma: float = DEFAULT_GAMMA,
        min_gamma: float = DEFAULT_MIN_GAMMA,
        gamma_decay: float = DEFAULT_GAMMA_DECAY,
        persistence_path: Optional[Path] = None,
    ):
        """Initialize EXP3 bandit selector.

        Args:
            models: List of model names (arms). Defaults to TCN/Ridge/RF.
            gamma: Initial exploration rate (0-1). Higher = more uniform.
            min_gamma: Minimum exploration rate floor.
            gamma_decay: Per-round multiplicative decay of gamma.
            persistence_path: Path for persisting arm statistics.
        """
        self.models = models or DEFAULT_MODELS[:]
        self.K = len(self.models)
        self.gamma = gamma
        self.min_gamma = min_gamma
        self.gamma_decay = gamma_decay
        self.persistence_path = persistence_path or _BANDIT_PATH

        # EXP3 weights — initialized uniformly
        self._weights: Dict[str, float] = {m: 1.0 for m in self.models}

        # Statistics for observability
        self._selection_counts: Dict[str, int] = {m: 0 for m in self.models}
        self._total_reward: Dict[str, float] = {m: 0.0 for m in self.models}
        self._rounds: int = 0

        # Recent reward history for diagnostics
        self._reward_history: List[Dict[str, Any]] = []

        # Try to load persisted state
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_model(
        self,
        context_features: Optional[Dict[str, float]] = None,
    ) -> SelectionResult:
        """Select a model using EXP3 probability distribution.

        Args:
            context_features: Optional context (reserved for contextual
                bandits extension). Currently unused in pure EXP3.

        Returns:
            SelectionResult with selected model and metadata.
        """
        # Compute EXP3 probability distribution
        probabilities = self._compute_probabilities()

        # Sample arm
        model_names = list(probabilities.keys())
        probs = [probabilities[m] for m in model_names]

        # Numpy sampling
        chosen_idx = int(np.random.choice(len(model_names), p=probs))
        chosen_model = model_names[chosen_idx]
        chosen_prob = probs[chosen_idx]

        # Determine if this was exploration (uniform component dominated)
        uniform_prob = 1.0 / self.K
        exploit_prob = (1.0 - self.gamma) * self._weights[chosen_model] / sum(self._weights.values())
        is_exploration = (self.gamma * uniform_prob) > exploit_prob

        self._selection_counts[chosen_model] += 1
        self._rounds += 1

        return SelectionResult(
            selected_model=chosen_model,
            is_exploration=is_exploration,
            selection_probability=chosen_prob,
            model_weights=dict(self._weights),
            model_probabilities=probabilities,
        )

    def update_reward(
        self,
        model_name: str,
        realized_pnl: float,
        selection_probability: Optional[float] = None,
    ) -> None:
        """Update arm weights based on observed reward.

        Args:
            model_name: Which model was selected.
            realized_pnl: The realized P/L from the trade (reward signal).
            selection_probability: Probability the model was selected
                (for importance weighting). If None, recomputes from current probs.
        """
        if model_name not in self._weights:
            logger.warning(f"ModelBandit: Unknown model '{model_name}', skipping update")
            return

        # Normalize reward to [-1, 1] range using tanh squash
        reward = float(np.tanh(realized_pnl / 50.0))  # 50 pips → ~0.76 reward

        # Get selection probability for importance weighting
        if selection_probability is None:
            probs = self._compute_probabilities()
            selection_probability = probs.get(model_name, 1.0 / self.K)

        # EXP3 importance-weighted reward estimate
        # r_hat = r / p(arm)  — unbiased estimator of true reward
        p = max(selection_probability, 0.01)  # Floor to avoid division by ~zero
        r_hat = reward / p

        # Weight update: W[arm] *= exp(gamma * r_hat / K)
        update = self.gamma * r_hat / self.K
        # Clamp update to prevent weight explosion
        update = max(-2.0, min(2.0, update))
        self._weights[model_name] *= math.exp(update)

        # Normalize weights to prevent drift (keep sum ≈ K)
        weight_sum = sum(self._weights.values())
        if weight_sum > 0:
            scale = self.K / weight_sum
            for m in self._weights:
                self._weights[m] *= scale

        # Floor weights to prevent any arm from being starved
        for m in self._weights:
            self._weights[m] = max(0.01, self._weights[m])

        # Decay exploration rate
        self.gamma = max(self.min_gamma, self.gamma * self.gamma_decay)

        # Track statistics
        self._total_reward[model_name] += reward

        # Log for audit
        self._reward_history.append({
            "round": self._rounds,
            "model": model_name,
            "pnl": round(realized_pnl, 2),
            "reward": round(reward, 4),
            "r_hat": round(r_hat, 4),
            "new_weight": round(self._weights[model_name], 6),
            "gamma": round(self.gamma, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Keep history bounded
        if len(self._reward_history) > 200:
            self._reward_history = self._reward_history[-100:]

        logger.debug(
            f"ModelBandit: {model_name} reward={reward:+.3f} "
            f"r_hat={r_hat:+.3f} weight→{self._weights[model_name]:.4f} "
            f"gamma={self.gamma:.4f}"
        )

    def get_model_weights(self) -> Dict[str, float]:
        """Return current arm weights (raw EXP3 weights)."""
        return dict(self._weights)

    def get_model_probabilities(self) -> Dict[str, float]:
        """Return current selection probabilities."""
        return self._compute_probabilities()

    def get_status(self) -> Dict[str, Any]:
        """Get bandit status for observability."""
        probs = self._compute_probabilities()
        avg_reward = {}
        for m in self.models:
            count = self._selection_counts.get(m, 0)
            total = self._total_reward.get(m, 0.0)
            avg_reward[m] = round(total / max(count, 1), 4)

        return {
            "rounds": self._rounds,
            "gamma": round(self.gamma, 4),
            "weights": {m: round(w, 4) for m, w in self._weights.items()},
            "probabilities": {m: round(p, 4) for m, p in probs.items()},
            "selection_counts": dict(self._selection_counts),
            "avg_reward": avg_reward,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_probabilities(self) -> Dict[str, float]:
        """Compute EXP3 mixed strategy.

        p(arm) = (1 - gamma) * W[arm] / sum(W) + gamma / K
        """
        weight_sum = sum(self._weights.values())
        if weight_sum <= 0:
            # Degenerate case: uniform
            return {m: 1.0 / self.K for m in self.models}

        probs = {}
        for m in self.models:
            exploit = (1.0 - self.gamma) * self._weights[m] / weight_sum
            explore = self.gamma / self.K
            probs[m] = exploit + explore

        # Normalize (should already sum to 1, but ensure numerical stability)
        prob_sum = sum(probs.values())
        if prob_sum > 0:
            probs = {m: p / prob_sum for m, p in probs.items()}

        return probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist bandit state to disk."""
        try:
            data = {
                "version": 1,
                "weights": {m: round(w, 6) for m, w in self._weights.items()},
                "selection_counts": dict(self._selection_counts),
                "total_reward": {m: round(r, 4) for m, r in self._total_reward.items()},
                "rounds": self._rounds,
                "gamma": round(self.gamma, 6),
                "reward_history": self._reward_history[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ModelBandit: Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load persisted bandit state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return

            saved_weights = data.get("weights", {})
            for m in self.models:
                if m in saved_weights:
                    self._weights[m] = float(saved_weights[m])

            saved_counts = data.get("selection_counts", {})
            for m in self.models:
                if m in saved_counts:
                    self._selection_counts[m] = int(saved_counts[m])

            saved_reward = data.get("total_reward", {})
            for m in self.models:
                if m in saved_reward:
                    self._total_reward[m] = float(saved_reward[m])

            self._rounds = data.get("rounds", 0)
            self.gamma = data.get("gamma", self.gamma)
            self._reward_history = data.get("reward_history", [])

            logger.debug(
                f"ModelBandit: Loaded state — rounds={self._rounds}, "
                f"gamma={self.gamma:.4f}, weights={self._weights}"
            )
        except Exception as e:
            logger.debug(f"ModelBandit: Failed to load state: {e}")
