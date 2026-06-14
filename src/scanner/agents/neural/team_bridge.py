"""
Bridge: drop-in replacement for ScannerAgentTeam that uses neural policies.

NeuralAgentTeam mirrors the public API of ScannerAgentTeam:
  - __init__(config)
  - evaluate(analysis, df_raw, df_feat, gate_details) -> analysis (with verdicts)
  - update_weights_from_outcome(agent_verdicts, trade_won, regime) -> dict

It can be toggled via config flag `use_neural_agents=True`.  When enabled,
the scanner instantiates NeuralAgentTeam instead of the rule-based
ScannerAgentTeam, but all downstream code (gates, execution, journaling)
continues to work unchanged.

DESIGN NOTE — No Scalar Weight Arithmetic
─────────────────────────────────────────
The legacy ScannerAgentTeam updates per-agent scalar weights via arithmetic
(+0.10 on win, -0.15 on loss, EMA damping, etc.).  NeuralAgentTeam does
NOT do this.  Instead, each agent's internal policy network is updated via
gradient descent on a supervised outcome dataset.  The "weight" field on each
AgentVerdict is a STATIC base multiplier for the voting formula only; the
actual learning happens inside the policy's weights, not in a scalar lookup
table.  This is the key architectural difference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.scanner.agents._team import AgentDecisionContext, AgentVerdict
from src.scanner.results import PairAnalysis
from .neural_agent_base import NeuralAgentBase
from .policies import TrendPolicy, MomentumPolicy, MeanReversionPolicy, VolatilityPolicy
from .trainer import NeuralAgentTrainer

logger = logging.getLogger(__name__)


class NeuralAgentTeam:
    """Learned-agent consensus layer.

    Replaces the rule-based specialist agents with neural policies.
    The weighted voting framework remains, but the scalar weight-learning
    arithmetic (ScannerAgentTeam.update_weights_from_outcome) is replaced
    by gradient-based policy updates via NeuralAgentTrainer.
    """

    # Static base weights for the voting formula.  These are NOT updated
    # by scalar arithmetic; the learning happens inside each agent's
    # policy network (neural_agent_base.py::_online_update).
    _BASE_WEIGHTS: Dict[str, float] = {
        "neural_trend": 1.15,
        "neural_momentum": 1.05,
        "neural_mean_reversion": 0.90,
        "neural_volatility": 1.00,
    }

    # Deliberately OMITTED compared to ScannerAgentTeam:
    #   - _learned_weights  (replaced by policy network parameters)
    #   - _apply_time_decay (irrelevant — network learns temporal dynamics)
    #   - _apply_confidence_scaling (irrelevant — calibration is implicit)
    #   - update_weights_from_outcome scalar arithmetic (replaced by backprop)

    def __init__(self, config: Any):
        self.config = config
        self.agents: Dict[str, NeuralAgentBase] = {
            "neural_trend": TrendPolicy(),
            "neural_momentum": MomentumPolicy(),
            "neural_mean_reversion": MeanReversionPolicy(),
            "neural_volatility": VolatilityPolicy(),
        }
        self.trainer = NeuralAgentTrainer()
        self.trainer.agents = self.agents
        # Attempt to load existing policies
        self.trainer.load_policies()

        # MED-5 FIX: Cache Devil's Advocate agent to avoid re-instantiation
        self._da_agent = None
        try:
            from src.scanner.agents.agent_da import DevilsAdvocateAgent
            self._da_agent = DevilsAdvocateAgent()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API — mirrors ScannerAgentTeam
    # ------------------------------------------------------------------
    def evaluate(
        self,
        analysis: PairAnalysis,
        df_raw: Any,
        df_feat: Any,
        gate_details: Dict[str, Any],
    ) -> PairAnalysis:
        """Run all neural agents and compute weighted consensus.

        The logic here is intentionally simpler than ScannerAgentTeam.evaluate()
        because the neural policies have *already learned* specialization.
        We don't need hard-coded regime multipliers or composite vetos —
        the network discovers those dynamics from data.
        """
        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details=gate_details,
            config=self.config,
        )

        verdicts: List[AgentVerdict] = []
        for name, agent in self.agents.items():
            try:
                vd = agent.evaluate(ctx)
                verdicts.append(vd)
            except Exception as e:
                logger.warning(f"{name} evaluation failed: {e}")

        if not verdicts:
            return analysis

        # Weighted vote score (same formula as rule-based team)
        total_weight = sum(v.weight for v in verdicts)
        weighted_sum = sum(v.score * v.weight for v in verdicts if v.passed)
        vote_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Any block_trade?
        any_block = any(v.block_trade for v in verdicts)
        passed = vote_score >= getattr(self.config, "min_agent_consensus_ratio", 0.50)

        analysis.weighted_vote_score = round(vote_score, 4)
        analysis.weighted_vote_threshold = float(getattr(self.config, "weighted_vote_threshold", 0.55))
        analysis.agent_passed = passed and not any_block
        analysis.agent_score = round(vote_score, 4)
        analysis.agent_votes = sum(1 for v in verdicts if v.passed)
        analysis.agent_total = len(verdicts)

        # Attach verdicts for downstream weight learning
        analysis.agent_verdicts = [v.to_dict() for v in verdicts]

        # Devil's Advocate (keep the adversarial agent even in neural mode)
        if self._da_agent is not None:
            try:
                da_verdict = self._da_agent.evaluate(ctx)
                verdicts.append(da_verdict)
                if da_verdict.block_trade:
                    analysis.agent_passed = False
                    analysis.kill_reason = "devils_advocate_block"
            except Exception as e:
                logger.debug(f"Devil's Advocate skipped in neural mode: {e}")
        else:
            logger.debug("Devil's Advocate not available in neural mode")

        # Cache verdicts on self for outcome updates
        self._last_verdicts = [v.to_dict() for v in verdicts]

        return analysis

    def update_weights_from_outcome(
        self,
        agent_verdicts: List[Dict[str, Any]],
        trade_won: bool,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """Update neural policies from trade outcome via gradient descent.

        REPLACES the legacy arithmetic:
            weight += 0.10 if (voted FOR and won) else -0.15
        with actual supervised learning on the agent's feature vectors.
        """
        return self.trainer.update_from_trade(agent_verdicts, trade_won)

    # ------------------------------------------------------------------
    # Backward-compat stubs
    # ------------------------------------------------------------------
    def reload_learned_weights(self) -> None:
        """Reload agent policies from disk (replaces scalar weight reload)."""
        self.trainer.load_policies()

    def get_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """Return static base weights (neural agents don't need regime-specific scalars)."""
        return dict(self._BASE_WEIGHTS)

    def _weight_for(self, agent_name: str) -> float:
        return self._BASE_WEIGHTS.get(agent_name, 1.0)
