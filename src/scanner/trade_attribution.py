"""
Trade Outcome Attribution Engine for Buddy FX Trading Bot.

After each closed trade, attributes the outcome to agent contributions
using a lightweight SHAP-inspired approach. Identifies which agents
drove correct/incorrect decisions.

US-304 (Phase 48): Trade Intelligence & Adaptive Filtering.

Research basis:
- SHAP (SHapley Additive exPlanations) for feature attribution
- Simplified to: compare each agent's signal direction with outcome
- Feeds attribution data back into expectancy tracker for weight optimization
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class AgentContribution:
    """A single agent's contribution to a trade decision."""
    agent_name: str
    verdict: str         # "LONG", "SHORT", "NEUTRAL", "SKIP"
    weight: float        # Agent weight at time of decision
    confidence: float    # Agent's confidence score (0-1)
    aligned_with_outcome: bool = False  # Did the agent's signal match the result?
    contribution_score: float = 0.0     # Weighted alignment score


@dataclass
class TradeAttribution:
    """Full attribution report for a single trade."""
    trade_id: str
    pair: str
    direction: str       # "BUY" or "SELL"
    outcome: str         # "WIN" or "LOSS"
    pnl: float
    agents: List[AgentContribution]
    timestamp: float = 0.0

    @property
    def aligned_agents(self) -> List[AgentContribution]:
        return [a for a in self.agents if a.aligned_with_outcome]

    @property
    def misaligned_agents(self) -> List[AgentContribution]:
        return [a for a in self.agents if not a.aligned_with_outcome]

    @property
    def alignment_ratio(self) -> float:
        """Fraction of agents aligned with outcome (weighted)."""
        if not self.agents:
            return 0.0
        total_weight = sum(a.weight for a in self.agents)
        if total_weight <= 0:
            return 0.0
        aligned_weight = sum(a.weight for a in self.agents if a.aligned_with_outcome)
        return aligned_weight / total_weight


@dataclass
class AgentPerformance:
    """Aggregate performance metrics for an agent across trades."""
    agent_name: str
    trades_participated: int = 0
    times_aligned: int = 0
    times_misaligned: int = 0
    total_contribution: float = 0.0

    @property
    def accuracy_rate(self) -> float:
        if self.trades_participated == 0:
            return 0.0
        return self.times_aligned / self.trades_participated

    @property
    def avg_contribution(self) -> float:
        if self.trades_participated == 0:
            return 0.0
        return self.total_contribution / self.trades_participated


@dataclass
class AttributionConfig:
    """Configuration for trade attribution engine."""

    # Maximum attributions to keep in memory
    max_history: int = 100

    # Persistence
    state_file: str = "trained_data/trade_attributions.json"

    # Rolling window for aggregate stats
    aggregate_window: int = 50

    enabled: bool = True
    version: int = 1

    def validate(self) -> None:
        if self.max_history < 10:
            raise ValueError(f"max_history must be >= 10, got {self.max_history}")
        if self.aggregate_window < 5:
            raise ValueError(f"aggregate_window must be >= 5, got {self.aggregate_window}")


class TradeAttributionEngine:
    """Attributes trade outcomes to individual agent contributions.

    For each closed trade, records which agents' signals aligned with
    the actual outcome. Computes per-agent accuracy and contribution
    scores to inform weight optimization.

    Usage:
        engine = TradeAttributionEngine()
        attribution = engine.attribute_trade(
            trade_id="T001", pair="EUR_USD", direction="BUY",
            outcome="WIN", pnl=25.5,
            agent_verdicts={"trend": ("LONG", 0.85, 1.0), ...}
        )
        performance = engine.get_agent_performance()
    """

    def __init__(self, config: Optional[AttributionConfig] = None):
        self._config = config or AttributionConfig()
        try:
            self._config.validate()
        except ValueError as e:
            logger.warning(f"Attribution config invalid: {e}, using defaults")
            self._config = AttributionConfig()

        self._history: List[TradeAttribution] = []
        self._load_state()

    @property
    def config(self) -> AttributionConfig:
        return self._config

    @property
    def history(self) -> List[TradeAttribution]:
        return self._history

    def attribute_trade(
        self,
        trade_id: str,
        pair: str,
        direction: str,
        outcome: str,
        pnl: float,
        agent_verdicts: Dict[str, tuple],
    ) -> TradeAttribution:
        """Attribute a trade outcome to agent contributions.

        Args:
            trade_id: Unique trade identifier
            pair: Currency pair
            direction: "BUY" or "SELL"
            outcome: "WIN" or "LOSS"
            pnl: Trade PnL
            agent_verdicts: Dict of {agent_name: (verdict, confidence, weight)}
                verdict: "LONG", "SHORT", "NEUTRAL", "SKIP"
                confidence: 0.0-1.0
                weight: Agent's weight

        Returns:
            TradeAttribution with full attribution details
        """
        if not self._config.enabled:
            return TradeAttribution(
                trade_id=trade_id, pair=pair, direction=direction,
                outcome=outcome, pnl=pnl, agents=[],
                timestamp=time.time(),
            )

        agents = []
        is_win = outcome.upper() == "WIN"

        for agent_name, (verdict, confidence, weight) in agent_verdicts.items():
            # Determine alignment
            aligned = self._check_alignment(
                verdict, direction, is_win
            )

            # Contribution score: weight * confidence * alignment_sign
            sign = 1.0 if aligned else -1.0
            contribution = weight * confidence * sign

            agents.append(AgentContribution(
                agent_name=agent_name,
                verdict=verdict,
                weight=weight,
                confidence=confidence,
                aligned_with_outcome=aligned,
                contribution_score=contribution,
            ))

        attribution = TradeAttribution(
            trade_id=trade_id,
            pair=pair,
            direction=direction,
            outcome=outcome,
            pnl=pnl,
            agents=agents,
            timestamp=time.time(),
        )

        self._history.append(attribution)

        # Maintain history window
        if len(self._history) > self._config.max_history:
            self._history = self._history[-self._config.max_history:]

        return attribution

    def _check_alignment(
        self,
        verdict: str,
        direction: str,
        is_win: bool,
    ) -> bool:
        """Check if an agent's verdict aligned with the trade outcome.

        An agent is aligned if:
        - It voted in the trade's direction AND the trade won
        - It voted against the trade's direction AND the trade lost
        - NEUTRAL/SKIP agents are considered aligned if trade won (they didn't block)
        """
        verdict_upper = verdict.upper()
        direction_upper = direction.upper()

        if verdict_upper in ("NEUTRAL", "SKIP"):
            return is_win  # Neutral/skip aligned with wins

        # Map direction to verdict
        dir_to_verdict = {"BUY": "LONG", "SELL": "SHORT"}
        expected_verdict = dir_to_verdict.get(direction_upper, "LONG")

        voted_same_direction = verdict_upper == expected_verdict

        if is_win:
            return voted_same_direction  # Voted same direction + won = aligned
        else:
            return not voted_same_direction  # Voted against + lost = aligned (correctly bearish)

    def get_agent_performance(
        self,
        window: Optional[int] = None,
    ) -> Dict[str, AgentPerformance]:
        """Compute aggregate performance metrics per agent.

        Args:
            window: Number of recent trades to consider (None = config default)

        Returns:
            Dict of agent_name → AgentPerformance
        """
        window = window or self._config.aggregate_window
        recent = self._history[-window:] if len(self._history) > window else self._history

        performance: Dict[str, AgentPerformance] = {}

        for attribution in recent:
            for agent in attribution.agents:
                name = agent.agent_name
                if name not in performance:
                    performance[name] = AgentPerformance(agent_name=name)

                perf = performance[name]
                perf.trades_participated += 1
                perf.total_contribution += agent.contribution_score

                if agent.aligned_with_outcome:
                    perf.times_aligned += 1
                else:
                    perf.times_misaligned += 1

        return performance

    def get_weight_recommendations(self) -> Dict[str, float]:
        """Generate weight adjustment recommendations based on attribution.

        Returns dict of agent_name → multiplier (>1.0 = increase, <1.0 = decrease)
        """
        perf = self.get_agent_performance()
        recommendations = {}

        for name, p in perf.items():
            if p.trades_participated < 10:
                recommendations[name] = 1.0  # Insufficient data
                continue

            acc = p.accuracy_rate
            if acc >= 0.65:
                recommendations[name] = 1.10  # Good performer, slight boost
            elif acc >= 0.55:
                recommendations[name] = 1.0   # Average, no change
            elif acc >= 0.45:
                recommendations[name] = 0.90  # Below average, slight reduction
            else:
                recommendations[name] = 0.75  # Poor performer, significant reduction

        return recommendations

    def _load_state(self) -> None:
        """Load persisted attribution history."""
        state_path = Path(self._config.state_file)
        if not state_path.exists():
            return

        try:
            with open(state_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return

            for item in data.get("history", []):
                try:
                    agents = []
                    for ad in item.get("agents", []):
                        agents.append(AgentContribution(
                            agent_name=ad["agent_name"],
                            verdict=ad["verdict"],
                            weight=ad.get("weight", 1.0),
                            confidence=ad.get("confidence", 0.5),
                            aligned_with_outcome=ad.get("aligned_with_outcome", False),
                            contribution_score=ad.get("contribution_score", 0.0),
                        ))

                    self._history.append(TradeAttribution(
                        trade_id=item["trade_id"],
                        pair=item["pair"],
                        direction=item["direction"],
                        outcome=item["outcome"],
                        pnl=item.get("pnl", 0.0),
                        agents=agents,
                        timestamp=item.get("timestamp", 0.0),
                    ))
                except (KeyError, TypeError) as e:
                    logger.debug(f"Skipping malformed attribution: {e}")

            logger.info(f"Loaded {len(self._history)} trade attributions")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load attribution state: {e}")

    def save_state(self) -> None:
        """Persist attribution history (atomic write)."""
        state_path = Path(self._config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        history_data = []
        for attr in self._history:
            history_data.append({
                "trade_id": attr.trade_id,
                "pair": attr.pair,
                "direction": attr.direction,
                "outcome": attr.outcome,
                "pnl": attr.pnl,
                "timestamp": attr.timestamp,
                "agents": [
                    {
                        "agent_name": a.agent_name,
                        "verdict": a.verdict,
                        "weight": a.weight,
                        "confidence": a.confidence,
                        "aligned_with_outcome": a.aligned_with_outcome,
                        "contribution_score": a.contribution_score,
                    }
                    for a in attr.agents
                ],
            })

        data = {
            "version": self._config.version,
            "saved_at": time.time(),
            "history": history_data,
        }

        tmp_path = str(state_path) + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, str(state_path))
        except OSError as e:
            logger.warning(f"Failed to save attribution state: {e}")


def create_default_attribution_engine(
    config: Optional[AttributionConfig] = None,
) -> TradeAttributionEngine:
    """Factory function for TradeAttributionEngine."""
    return TradeAttributionEngine(config=config)
