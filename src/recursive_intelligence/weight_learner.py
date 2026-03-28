"""Domain-agnostic weight learning engine.

Extracted from Buddy's OnlineWeightUpdater and ScannerAgentTeam RL logic.
Generalizes the outcome → agent weight adjustment feedback loop.

Buddy instantiation:
    agents = ["trend", "mean_reversion", "volatility", ...]
    outcome_signal = trade_won (bool)
    regime_awareness = volatility regime

Aura instantiation:
    agents = ["emotional_state", "cognitive_load", "decision_pattern", ...]
    outcome_signal = user_validated (bool) — did the user confirm the insight was accurate?
    regime_awareness = life_context (e.g., "high_stress", "transition", "stable")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.scanner.automation.safe_json import safe_json_write
    _HAS_SAFE_JSON = True
except ImportError:
    _HAS_SAFE_JSON = False

logger = logging.getLogger(__name__)

# Guard rails
MAX_ADJUSTMENT = 0.10  # Max ±10% per update
MIN_WEIGHT = 0.05      # No agent below 5%
MAX_WEIGHT = 3.0       # No agent above 300%


class WeightLearner:
    """Incremental RL-style weight learning from outcome signals.

    Args:
        domain: Name of the domain ("market", "human", "bridge")
        agents: List of agent names to track weights for
        base_weights: Starting weight for each agent
        learning_rate: Step size for weight adjustments
        weights_path: Where to persist learned weights
    """

    def __init__(
        self,
        domain: str,
        agents: List[str],
        base_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.02,
        weights_path: Optional[Path] = None,
    ):
        self.domain = domain
        self.agents = agents
        self.base_weights = base_weights or {a: 1.0 for a in agents}
        self.learning_rate = learning_rate
        self.weights_path = weights_path or Path(f".aura/weights_{domain}.json")
        self._weights: Dict[str, float] = self._load_weights()
        self._update_count: int = 0

    def get_weight(self, agent: str) -> float:
        """Get current weight for an agent."""
        return self._weights.get(agent, self.base_weights.get(agent, 1.0))

    def get_all_weights(self) -> Dict[str, float]:
        """Get all current weights."""
        return dict(self._weights)

    def update_from_outcome(
        self,
        agent_scores: Dict[str, float],
        outcome_positive: bool,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """Update weights based on an outcome signal.

        Agents that scored high on a positive outcome get weight increases.
        Agents that scored high on a negative outcome get weight decreases.

        Args:
            agent_scores: Dict of agent_name -> score (0-1) from this observation
            outcome_positive: Whether the outcome was positive
            regime: Optional context regime for future regime-aware learning

        Returns:
            Dict of agent -> new weight after update
        """
        direction = 1.0 if outcome_positive else -1.0

        for agent, score in agent_scores.items():
            if agent not in self._weights:
                continue

            current = self._weights[agent]
            # Agents that scored high get rewarded/penalized based on outcome
            # Agents that scored low are unaffected
            delta = direction * self.learning_rate * (score - 0.5)
            delta = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, delta))

            new_weight = current + delta
            new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
            self._weights[agent] = round(new_weight, 4)

        self._update_count += 1
        self._persist_weights()

        return dict(self._weights)

    def apply_decay(self, decay_rate: float = 0.02) -> Dict[str, float]:
        """Decay weights toward baseline to prevent drift."""
        for agent in list(self._weights.keys()):
            base = self.base_weights.get(agent, 1.0)
            current = self._weights[agent]
            decayed = current + decay_rate * (base - current)
            self._weights[agent] = round(decayed, 4)

        self._persist_weights()
        return dict(self._weights)

    def _load_weights(self) -> Dict[str, float]:
        if not self.weights_path.exists():
            return dict(self.base_weights)
        try:
            data = json.loads(self.weights_path.read_text())
            if isinstance(data, dict) and "weights" in data:
                return data["weights"]
            elif isinstance(data, dict):
                return data
            return dict(self.base_weights)
        except Exception as e:
            logger.warning(f"[{self.domain}] Failed to load weights: {e}")
            return dict(self.base_weights)

    def _persist_weights(self) -> None:
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "domain": self.domain,
                "weights": self._weights,
                "update_count": self._update_count,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            if _HAS_SAFE_JSON:
                success = safe_json_write(self.weights_path, data)
                if not success:
                    logger.error(f"[{self.domain}] safe_json_write failed for weights")
            else:
                # Fallback: atomic temp+rename without locking
                import os
                import tempfile
                json_str = json.dumps(data, indent=2, default=str)
                fd, tmp = tempfile.mkstemp(dir=str(self.weights_path.parent))
                try:
                    os.write(fd, json_str.encode("utf-8"))
                    os.fsync(fd)
                    os.close(fd)
                    os.rename(tmp, str(self.weights_path))
                except Exception:
                    os.close(fd)
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
        except Exception as e:
            logger.error(f"[{self.domain}] Failed to persist weights: {e}")
