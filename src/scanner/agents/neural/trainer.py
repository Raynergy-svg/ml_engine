"""
Multi-agent neural policy trainer.

Trains all neural agents simultaneously from trade outcomes.

Workflow:
  1. After each trade, collect all agents' experiences + outcome.
  2. Build per-agent supervised datasets:
     - features -> target_score, where target_score is shaped by reward.
  3. Run lightweight online updates (1 epoch, small batch) per agent.
  4. Persist updated policies to disk.

The trainer hooks into the existing `update_weights_from_outcome()` path
in ScannerAgentTeam so no external orchestration is needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .neural_agent_base import NeuralAgentBase
from .policies import TrendPolicy, MomentumPolicy, MeanReversionPolicy, VolatilityPolicy

logger = logging.getLogger(__name__)


@dataclass
class NeuralTrainerConfig:
    """Trainer hyperparameters."""

    min_samples_per_agent: int = 50
    batch_size: int = 32
    epochs_per_update: int = 3
    learning_rate_decay: float = 0.95
    max_history_per_agent: int = 2000
    save_dir: str = "trained_data/models/neural_agents"


class NeuralAgentTrainer:
    """Orchestrates training for a collection of neural agents."""

    def __init__(self, config: Optional[NeuralTrainerConfig] = None):
        self.cfg = config or NeuralTrainerConfig()
        self.agents: Dict[str, NeuralAgentBase] = {}
        self._init_default_agents()

    def _init_default_agents(self) -> None:
        """Instantiate the default neural agent roster."""
        self.agents = {
            "neural_trend": TrendPolicy(),
            "neural_momentum": MomentumPolicy(),
            "neural_mean_reversion": MeanReversionPolicy(),
            "neural_volatility": VolatilityPolicy(),
        }

    def load_policies(self, save_dir: Optional[str] = None) -> None:
        """Load existing policies from disk if available."""
        sd = Path(save_dir or self.cfg.save_dir)
        for name, agent in self.agents.items():
            path = sd / f"{name}.keras"
            if path.exists():
                agent.load(str(path))
            else:
                logger.debug(f"No saved policy for {name} at {path}")

    def save_policies(self, save_dir: Optional[str] = None) -> None:
        """Persist all agent policies to disk."""
        sd = Path(save_dir or self.cfg.save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        for name, agent in self.agents.items():
            path = sd / f"{name}.keras"
            agent.save(str(path))
        logger.info(f"Saved {len(self.agents)} neural agent policies to {sd}")

    # ------------------------------------------------------------------
    # Outcome-driven training
    # ------------------------------------------------------------------
    def update_from_trade(
        self,
        agent_verdicts: List[Dict[str, Any]],
        trade_won: bool,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Update all neural agents after a trade closes.

        Args:
            agent_verdicts: List of verdict dicts (same format as
                            ScannerAgentTeam.update_weights_from_outcome).
            trade_won: True if TP hit, False if SL hit.

        Returns:
            Dict mapping agent_name -> loss from the online update.
        """
        results: Dict[str, float] = {}
        for vd in agent_verdicts:
            name = str(vd.get("name", ""))
            if name not in self.agents:
                continue
            agent = self.agents[name]
            agent.record_outcome(vd.get("metadata", {}), trade_won)
            # Trigger online update if enough history
            if len(agent._trade_history) >= self.cfg.min_samples_per_agent:
                agent._online_update()
                results[name] = float(agent.policy.history.history["loss"][-1]) if agent.policy and hasattr(agent.policy, "history") else 0.0
        return results

    def batch_train(
        self,
        trade_history: List[Dict[str, Any]],
        epochs: int = 10,
    ) -> Dict[str, float]:
        """Offline batch training on a full trade history dataset.

        Useful for initial warm-start or periodic retraining.
        Each trade record should contain:
          - agent_name: str
          - features: List[float]
          - score: float (agent's score at entry)
          - passed: bool
          - trade_won: bool
          - pnl_pips: float
        """
        # Group by agent
        per_agent: Dict[str, List[Dict[str, Any]]] = {}
        for rec in trade_history:
            name = rec.get("agent_name", "")
            if name not in self.agents:
                continue
            per_agent.setdefault(name, []).append(rec)

        results: Dict[str, float] = {}
        for name, records in per_agent.items():
            if len(records) < self.cfg.min_samples_per_agent:
                continue
            agent = self.agents[name]
            X = np.array([r["features"] for r in records], dtype=np.float32)
            # Target: 1.0 if (passed and won) or (not passed and lost)
            #         0.0 if (passed and lost) or (not passed and won)
            y = np.array([
                1.0 if (r["passed"] == r["trade_won"]) else 0.0
                for r in records
            ], dtype=np.float32)

            if agent.policy is None:
                agent.policy = agent.build_policy(X.shape[1])

            history = agent.policy.fit(
                X, y,
                epochs=epochs,
                batch_size=self.cfg.batch_size,
                validation_split=0.2,
                verbose=0,
            )
            final_loss = float(history.history["loss"][-1])
            results[name] = final_loss
            logger.info(f"{name}: batch-trained on {len(records)} trades, final_loss={final_loss:.4f}")

        return results

    # ------------------------------------------------------------------
    # Evaluation / diagnostics
    # ------------------------------------------------------------------
    def evaluate_policy_calibration(self) -> Dict[str, Dict[str, float]]:
        """Compute per-agent calibration metrics on held-out recent history."""
        metrics: Dict[str, Dict[str, float]] = {}
        for name, agent in self.agents.items():
            hist = agent._trade_history
            if len(hist) < 50:
                continue
            # Simple calibration: bucket scores into 5 bins, compute win rate per bin
            bins: Dict[int, List[bool]] = {}
            for rec in hist[-500:]:
                score = rec.get("score", 0.5)
                bin_idx = min(int(score * 5), 4)
                bins.setdefault(bin_idx, []).append(rec["trade_won"])
            calibration = {}
            for b, outcomes in sorted(bins.items()):
                calibration[f"bin_{b}"] = sum(outcomes) / len(outcomes) if outcomes else 0.0
            metrics[name] = calibration
        return metrics
