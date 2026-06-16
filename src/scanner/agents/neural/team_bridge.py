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

DESIGN NOTE — Complete Replacement of 15-Agent Layer
────────────────────────────────────────────────────
All 15 specialist agents (core 12 + extended 3) are now neural policies.
The legacy ScannerAgentTeam scalar weight arithmetic is fully replaced by
gradient-based policy updates via NeuralAgentTrainer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.scanner.agents._team import AgentDecisionContext, AgentVerdict
from src.scanner.results import PairAnalysis
from .neural_agent_base import NeuralAgentBase
from .policies import (
    TrendPolicy, MomentumPolicy, MeanReversionPolicy, VolatilityPolicy,
    RiskSentinelPolicy, UncertaintyPolicy, ExecutionQualityPolicy,
    NewsRiskPolicy, MultiTimeframePolicy, PairPerformancePolicy,
    SessionTimingPolicy, SupportResistancePolicy, OrderFlowPolicy,
    TraderReadinessPolicy, DevilsAdvocatePolicy,
)
from .trainer import NeuralAgentTrainer

logger = logging.getLogger(__name__)


class NeuralAgentTeam:
    """Learned-agent consensus layer — full 15-agent neural replacement."""

    # Base weights aligned with legacy _BASE_WEIGHTS in _team.py.
    # These are static multipliers for the voting formula only.
    _BASE_WEIGHTS: Dict[str, float] = {
        "neural_trend": 1.15,
        "neural_momentum": 1.05,
        "neural_mean_reversion": 0.90,
        "neural_volatility": 1.00,
        "neural_risk_sentinel": 1.25,
        "neural_uncertainty": 1.10,
        "neural_execution_quality": 1.05,
        "neural_news_risk": 0.95,
        "neural_multi_timeframe": 1.10,
        "neural_pair_performance": 0.85,
        "neural_session_timing": 0.80,
        "neural_support_resistance": 1.00,
        "neural_order_flow": 0.95,
        "neural_trader_readiness": 0.50,
        "neural_devil_advocate": 1.30,
    }

    def __init__(self, config: Any):
        self.config = config
        self.agents: Dict[str, NeuralAgentBase] = {
            "neural_trend": TrendPolicy(),
            "neural_momentum": MomentumPolicy(),
            "neural_mean_reversion": MeanReversionPolicy(),
            "neural_volatility": VolatilityPolicy(),
            "neural_risk_sentinel": RiskSentinelPolicy(),
            "neural_uncertainty": UncertaintyPolicy(),
            "neural_execution_quality": ExecutionQualityPolicy(),
            "neural_news_risk": NewsRiskPolicy(),
            "neural_multi_timeframe": MultiTimeframePolicy(),
            "neural_pair_performance": PairPerformancePolicy(),
            "neural_session_timing": SessionTimingPolicy(),
            "neural_support_resistance": SupportResistancePolicy(),
            "neural_order_flow": OrderFlowPolicy(),
            "neural_trader_readiness": TraderReadinessPolicy(),
            "neural_devil_advocate": DevilsAdvocatePolicy(),
        }
        self.trainer = NeuralAgentTrainer()
        self.trainer.agents = self.agents
        # Attempt to load existing policies
        self.trainer.load_policies()

    def evaluate(
        self,
        analysis: PairAnalysis,
        df_raw: Any,
        df_feat: Any,
        gate_details: Dict[str, Any],
    ) -> PairAnalysis:
        """Run all 15 neural agents and compute weighted consensus."""
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

        total_weight = sum(v.weight for v in verdicts)
        weighted_sum = sum(v.score * v.weight for v in verdicts if v.passed)
        vote_score = weighted_sum / total_weight if total_weight > 0 else 0.0

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

        # Devil's Advocate runs LAST and can veto (already included in verdicts,
        # but we give it explicit kill-power here for backward compat).
        da_verdicts = [v for v in verdicts if v.name == "neural_devil_advocate"]
        if da_verdicts and da_verdicts[0].block_trade:
            analysis.agent_passed = False
            analysis.kill_reason = "neural_devil_advocate_block"

        self._last_verdicts = [v.to_dict() for v in verdicts]
        return analysis

    def update_weights_from_outcome(
        self,
        agent_verdicts: List[Dict[str, Any]],
        trade_won: bool,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """Update neural policies from trade outcome via gradient descent."""
        return self.trainer.update_from_trade(agent_verdicts, trade_won)

    def reload_learned_weights(self) -> None:
        """Reload agent policies from disk."""
        self.trainer.load_policies()

    def get_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """Return static base weights (neural agents don't need regime-specific scalars)."""
        return dict(self._BASE_WEIGHTS)

    def _weight_for(self, agent_name: str) -> float:
        return self._BASE_WEIGHTS.get(agent_name, 1.0)

    # ------------------------------------------------------------------
    # Compatibility bridge for engine.py record_trade_outcome_phase18
    # (US-004: NeuralAgentTeam learns from live trade outcomes)
    # ------------------------------------------------------------------
    def cache_trade_verdicts(self, episode_id: str, verdicts: List[Dict[str, Any]]) -> None:
        """Cache agent verdicts by episode_id for later outcome binding."""
        if not hasattr(self, "_trade_verdict_cache"):
            self._trade_verdict_cache: Dict[str, Dict[str, Any]] = {}
        if episode_id:
            self._trade_verdict_cache[episode_id] = {"verdicts": list(verdicts)}

    def record_trade_outcome(
        self,
        episode_id: str,
        outcome: str,
        pnl_pips: float,
    ) -> bool:
        """Record trade outcome and trigger policy updates.

        Compatible interface with ScannerAgentTeam.record_trade_outcome
        so engine.py::record_trade_outcome_phase18 works unchanged.
        """
        if not hasattr(self, "_trade_verdict_cache"):
            return False
        cached = self._trade_verdict_cache.get(episode_id)
        if not cached:
            return False
        verdicts = cached.get("verdicts", [])
        trade_won = outcome in ("WIN", "BREAKEVEN")
        try:
            self.update_weights_from_outcome(verdicts, trade_won)
        except Exception as e:
            logger.warning("NeuralAgentTeam outcome update failed: %s", e)
            return False
        # Evict to prevent unbounded growth
        self._trade_verdict_cache.pop(episode_id, None)
        return True
