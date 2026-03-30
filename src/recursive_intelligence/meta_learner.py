"""Meta-learning orchestrator.

Reads trade outcomes, adapts agent hyperparameters, injects them into next scan.
Provides rollback safety: if 3 consecutive trades with adapted params lose >5%,
reverts to baseline and skips adaptation for 10 cycles.

Integration:
    # After each trade close:
    meta_learner.on_trade_close(pair, regime, trade_result, agent_scores, agent_weights)

    # Before each scan cycle:
    overrides = meta_learner.get_active_overrides(agent_name, regime)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .meta_parameters import MetaParametersRegistry, HyperparameterSet, REGIMES
from .bayesian_adapter import OnlineBayesianAdapter

logger = logging.getLogger(__name__)

# Rollback triggers
ROLLBACK_LOSS_THRESHOLD = -0.05        # -5% return = bad outcome
ROLLBACK_CONSECUTIVE_LOSSES = 3        # 3 bad trades in a row → rollback
COOLDOWN_CYCLES_AFTER_ROLLBACK = 10    # Skip adaptation for this many scan cycles


class MetaLearner:
    """Orchestrates meta-learning: reads outcomes, adapts hyperparams.

    Does NOT replace the weight_learner — it runs alongside it.
    The weight_learner adjusts *which* agents vote.
    The MetaLearner adjusts *how* each agent evaluates.
    """

    def __init__(
        self,
        registry: Optional[MetaParametersRegistry] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._registry = registry or MetaParametersRegistry()
        self._adapter = OnlineBayesianAdapter(self._registry)

        # Rollback tracking: {agent_name → consecutive_loss_count}
        self._loss_streaks: Dict[str, int] = {}
        # Cooldown tracking: {agent_name → cycles_remaining}
        self._cooldowns: Dict[str, int] = {}
        # Total stats
        self._total_updates: int = 0
        self._total_rollbacks: int = 0

    def on_trade_close(
        self,
        pair: str,
        regime: str,
        trade_result: Dict[str, Any],
        agent_scores: Dict[str, float],       # {agent_name → score 0-1}
        agent_weights: Optional[Dict[str, float]] = None,  # {agent_name → weight}
    ) -> Dict[str, HyperparameterSet]:
        """Called after a trade closes. Returns updated hyperparams for next cycle.

        trade_result must contain:
            - 'pnl_pct': float (profit/loss as fraction, e.g. 0.02 = +2%)
            OR
            - 'outcome': 'win' | 'loss'

        Returns:
            {agent_name → active_hyperparams_for_next_scan}
        """
        if not self.enabled:
            return {}

        # Normalize outcome to [-1, +1]
        if "pnl_pct" in trade_result:
            raw = float(trade_result["pnl_pct"])
            trade_outcome = float(max(-1.0, min(1.0, raw * 10.0)))  # scale: 10% = ±1.0
        elif "outcome" in trade_result:
            trade_outcome = 1.0 if trade_result["outcome"] == "win" else -1.0
        else:
            logger.warning("MetaLearner: no outcome in trade_result for %s — skipping", pair)
            return {}

        updated: Dict[str, HyperparameterSet] = {}

        for agent_name, agent_score in agent_scores.items():
            # Check cooldown
            if self._cooldowns.get(agent_name, 0) > 0:
                self._cooldowns[agent_name] -= 1
                updated[agent_name] = self._registry.get(agent_name, regime)
                continue

            # Detect loss streak → rollback
            if trade_outcome < ROLLBACK_LOSS_THRESHOLD:
                self._loss_streaks[agent_name] = self._loss_streaks.get(agent_name, 0) + 1
                if self._loss_streaks[agent_name] >= ROLLBACK_CONSECUTIVE_LOSSES:
                    self._rollback(agent_name, regime)
                    updated[agent_name] = self._registry.get(agent_name, regime)
                    continue
            else:
                self._loss_streaks[agent_name] = 0  # Reset streak on non-loss

            # Update posterior and get new hyperparams
            try:
                new_params = self._adapter.update(
                    agent_name=agent_name,
                    regime=regime,
                    agent_score=float(agent_score),
                    trade_outcome=trade_outcome,
                )
                updated[agent_name] = new_params
                self._total_updates += 1
            except Exception as e:
                logger.error("MetaLearner: update failed for %s/%s: %s", agent_name, regime, e)
                updated[agent_name] = self._registry.get(agent_name, regime)

        # Persist to disk
        try:
            self._registry.save()
        except Exception as e:
            logger.error("MetaLearner: failed to persist registry: %s", e)

        return updated

    def get_active_overrides(self, agent_name: str, regime: str) -> Dict[str, float]:
        """Get current hyperparam overrides for injection into agent logic.

        Returns:
            {
                'confidence_threshold': float,
                'weight_decay_rate': float,
                'regime_switch_sensitivity': float,
                'voting_veto_power': float,
            }
        """
        if not self.enabled:
            return {}
        try:
            params = self._registry.get(agent_name, regime)
            return {
                "confidence_threshold": params.confidence_threshold,
                "weight_decay_rate": params.weight_decay_rate,
                "regime_switch_sensitivity": params.regime_switch_sensitivity,
                "voting_veto_power": params.voting_veto_power,
            }
        except Exception as e:
            logger.warning("MetaLearner: get_active_overrides failed for %s/%s: %s", agent_name, regime, e)
            return {}

    def _rollback(self, agent_name: str, regime: str) -> None:
        """Reset agent's hyperparams to baseline after loss streak."""
        self._registry.reset_to_default(agent_name, regime)
        self._adapter.reset_posterior(agent_name, regime)
        self._loss_streaks[agent_name] = 0
        self._cooldowns[agent_name] = COOLDOWN_CYCLES_AFTER_ROLLBACK
        self._total_rollbacks += 1
        logger.warning(
            "MetaLearner: ROLLBACK for %s/%s after %d consecutive losses — cooldown %d cycles",
            agent_name, regime, ROLLBACK_CONSECUTIVE_LOSSES, COOLDOWN_CYCLES_AFTER_ROLLBACK
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_updates": self._total_updates,
            "total_rollbacks": self._total_rollbacks,
            "active_cooldowns": {k: v for k, v in self._cooldowns.items() if v > 0},
            "loss_streaks": {k: v for k, v in self._loss_streaks.items() if v > 0},
            "enabled": self.enabled,
        }
