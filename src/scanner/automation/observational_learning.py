"""Observational diversity learning from synthetic agents.

US-078: Extract patterns from synthetic trade simulations to accelerate RL
convergence. Synthetic agents (5 types) generate trade scenarios on historical
data. ObservationalLearner identifies winning conditions and agent combinations,
then blends these into the live agent weight system.

Blend ratio: 20% synthetic weight + 80% live weight (configurable).

Key patterns extracted:
1. Which synthetic agent types win in which regimes
2. Feature combinations that predict wins across agent types
3. Optimal agent blends per regime
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default blend ratio: 20% synthetic, 80% live
DEFAULT_SYNTHETIC_WEIGHT = 0.20
DEFAULT_LIVE_WEIGHT = 0.80

# Minimum trades from a synthetic agent to consider patterns valid
MIN_SYNTHETIC_TRADES = 10

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PATTERNS_PATH = _PROJECT_ROOT / "trained_data" / "observational_patterns.json"


@dataclass
class AgentPerformance:
    """Performance summary of a synthetic agent type."""

    agent_type: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_pnl_pips: float = 0.0
    best_regime: str = "NORMAL"
    worst_regime: str = "NORMAL"
    regime_win_rates: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_pnl_pips": round(self.avg_pnl_pips, 2),
            "best_regime": self.best_regime,
            "worst_regime": self.worst_regime,
            "regime_win_rates": {k: round(v, 4) for k, v in self.regime_win_rates.items()},
        }


@dataclass
class ObservationalPattern:
    """An extracted pattern from synthetic trade analysis."""

    pattern_type: str = ""  # regime_preference, feature_threshold, agent_blend
    description: str = ""
    agent_types: List[str] = field(default_factory=list)
    regime: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    sample_size: int = 0

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "agent_types": self.agent_types,
            "regime": self.regime,
            "conditions": self.conditions,
            "confidence": round(self.confidence, 4),
            "sample_size": self.sample_size,
        }


@dataclass
class ObservationalResult:
    """Result of observational learning analysis."""

    total_synthetic_trades: int = 0
    agent_performances: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    suggested_weight_adjustments: Dict[str, float] = field(default_factory=dict)
    blend_ratio: float = DEFAULT_SYNTHETIC_WEIGHT

    def to_dict(self) -> dict:
        return {
            "total_synthetic_trades": self.total_synthetic_trades,
            "agent_performances": self.agent_performances,
            "patterns": self.patterns,
            "suggested_weight_adjustments": {
                k: round(v, 4) for k, v in self.suggested_weight_adjustments.items()
            },
            "blend_ratio": self.blend_ratio,
        }


class ObservationalLearner:
    """Extracts patterns from synthetic trade simulations.

    Pipeline:
    1. Receive synthetic trades from all 5 agent types
    2. Compute per-agent performance (overall and per-regime)
    3. Extract patterns (winning conditions, feature thresholds)
    4. Map synthetic agent types to live agent weights
    5. Produce blended weight adjustments (20% synthetic + 80% live)
    """

    # Mapping from synthetic agent types to live agent names
    SYNTHETIC_TO_LIVE_MAP: Dict[str, List[str]] = {
        "momentum_chaser": ["trend", "momentum"],
        "mean_reversion_scalper": ["mean_reversion"],
        "risk_averse": ["risk_sentinel", "uncertainty"],
        "macro_cyclical": ["session_timing", "multi_timeframe"],
        "news_reactive": ["news_risk", "volatility"],
    }

    def __init__(
        self,
        synthetic_weight: float = DEFAULT_SYNTHETIC_WEIGHT,
        persistence_path: Optional[Path] = None,
    ):
        self.synthetic_weight = max(0.0, min(synthetic_weight, 0.50))  # Cap at 50%
        self.live_weight = 1.0 - self.synthetic_weight
        self.persistence_path = persistence_path or _PATTERNS_PATH
        self._cached_patterns: List[ObservationalPattern] = []

    def extract_patterns(
        self,
        synthetic_trades: List[Any],
    ) -> ObservationalResult:
        """Analyze synthetic trades and extract actionable patterns.

        Args:
            synthetic_trades: List of SyntheticTrade objects (or dicts)

        Returns:
            ObservationalResult with patterns and weight suggestions
        """
        result = ObservationalResult(
            total_synthetic_trades=len(synthetic_trades),
            blend_ratio=self.synthetic_weight,
        )

        if not synthetic_trades:
            return result

        # Normalize to dicts
        trades = []
        for t in synthetic_trades:
            if hasattr(t, "to_dict"):
                trades.append(t.to_dict())
            elif isinstance(t, dict):
                trades.append(t)

        # Step 1: Compute per-agent performance
        agent_perfs = self._compute_agent_performances(trades)
        result.agent_performances = [p.to_dict() for p in agent_perfs]

        # Step 2: Extract patterns
        patterns = self._extract_regime_patterns(agent_perfs)
        patterns.extend(self._extract_feature_patterns(trades))
        result.patterns = [p.to_dict() for p in patterns]
        self._cached_patterns = patterns

        # Step 3: Compute weight adjustments
        result.suggested_weight_adjustments = self._compute_weight_suggestions(
            agent_perfs
        )

        # Persist patterns
        self._save_patterns(result)

        logger.info(
            "ObservationalLearner: %d synthetic trades → %d patterns, %d weight adjustments",
            len(trades), len(patterns), len(result.suggested_weight_adjustments),
        )

        return result

    def get_blended_weights(
        self,
        live_weights: Dict[str, float],
        suggested_adjustments: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Blend live agent weights with synthetic-derived adjustments.

        Args:
            live_weights: Current live agent weights
            suggested_adjustments: From extract_patterns() result

        Returns:
            Blended weights dict
        """
        adjustments = suggested_adjustments or {}
        blended = {}

        for agent, live_w in live_weights.items():
            synth_adj = adjustments.get(agent, 0.0)
            # Blend: 80% live + 20% synthetic adjustment
            blended_w = self.live_weight * live_w + self.synthetic_weight * (live_w + synth_adj)
            blended[agent] = max(0.1, min(2.0, blended_w))

        return blended

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def _compute_agent_performances(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[AgentPerformance]:
        """Compute per-agent-type performance stats."""
        # Group by agent type
        by_agent: Dict[str, List[Dict]] = {}
        for t in trades:
            agent = t.get("agent_type", "unknown")
            by_agent.setdefault(agent, []).append(t)

        performances = []
        for agent_type, agent_trades in by_agent.items():
            perf = AgentPerformance(agent_type=agent_type)
            perf.total_trades = len(agent_trades)
            perf.wins = sum(1 for t in agent_trades if t.get("outcome") == "WIN")
            perf.losses = sum(1 for t in agent_trades if t.get("outcome") == "LOSS")
            perf.win_rate = perf.wins / max(perf.total_trades, 1)
            perf.avg_pnl_pips = (
                sum(t.get("pnl_pips", 0) for t in agent_trades) / max(perf.total_trades, 1)
            )

            # Per-regime breakdown
            by_regime: Dict[str, List[Dict]] = {}
            for t in agent_trades:
                regime = t.get("regime", "NORMAL")
                by_regime.setdefault(regime, []).append(t)

            regime_wr: Dict[str, float] = {}
            for regime, r_trades in by_regime.items():
                wins = sum(1 for t in r_trades if t.get("outcome") == "WIN")
                regime_wr[regime] = wins / max(len(r_trades), 1)
            perf.regime_win_rates = regime_wr

            if regime_wr:
                perf.best_regime = max(regime_wr, key=regime_wr.get)
                perf.worst_regime = min(regime_wr, key=regime_wr.get)

            performances.append(perf)

        return performances

    def _extract_regime_patterns(
        self,
        agent_perfs: List[AgentPerformance],
    ) -> List[ObservationalPattern]:
        """Extract which agent types excel in which regimes."""
        patterns = []

        for perf in agent_perfs:
            if perf.total_trades < MIN_SYNTHETIC_TRADES:
                continue

            # Pattern: agent excels in specific regime
            for regime, wr in perf.regime_win_rates.items():
                if wr > 0.60:  # Above 60% win rate in a regime
                    patterns.append(
                        ObservationalPattern(
                            pattern_type="regime_preference",
                            description=(
                                f"{perf.agent_type} has {wr:.0%} win rate in {regime} regime"
                            ),
                            agent_types=[perf.agent_type],
                            regime=regime,
                            conditions={"win_rate": wr},
                            confidence=min(wr, 0.95),
                            sample_size=perf.total_trades,
                        )
                    )

        return patterns

    def _extract_feature_patterns(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[ObservationalPattern]:
        """Extract feature thresholds that predict wins."""
        patterns = []

        # Check if ATR thresholds matter
        winning_atrs = [
            t.get("features", {}).get("atr_pips", 0)
            for t in trades
            if t.get("outcome") == "WIN" and t.get("features", {}).get("atr_pips", 0) > 0
        ]
        losing_atrs = [
            t.get("features", {}).get("atr_pips", 0)
            for t in trades
            if t.get("outcome") == "LOSS" and t.get("features", {}).get("atr_pips", 0) > 0
        ]

        if winning_atrs and losing_atrs:
            avg_win_atr = sum(winning_atrs) / len(winning_atrs)
            avg_loss_atr = sum(losing_atrs) / len(losing_atrs)

            if abs(avg_win_atr - avg_loss_atr) > 1.0:
                patterns.append(
                    ObservationalPattern(
                        pattern_type="feature_threshold",
                        description=(
                            f"Winning trades avg ATR={avg_win_atr:.1f} vs "
                            f"losing ATR={avg_loss_atr:.1f}"
                        ),
                        conditions={
                            "avg_win_atr": avg_win_atr,
                            "avg_loss_atr": avg_loss_atr,
                            "recommended_atr_min": min(avg_win_atr, avg_loss_atr),
                        },
                        confidence=0.60,
                        sample_size=len(winning_atrs) + len(losing_atrs),
                    )
                )

        return patterns

    def _compute_weight_suggestions(
        self,
        agent_perfs: List[AgentPerformance],
    ) -> Dict[str, float]:
        """Map synthetic agent performance to live weight adjustments.

        Returns:
            Dict mapping live agent names to suggested weight deltas.
        """
        suggestions: Dict[str, float] = {}

        for perf in agent_perfs:
            if perf.total_trades < MIN_SYNTHETIC_TRADES:
                continue

            # Map synthetic agent to live agents
            live_agents = self.SYNTHETIC_TO_LIVE_MAP.get(perf.agent_type, [])

            # Scale adjustment by how much the agent beats/misses 50% baseline
            edge = perf.win_rate - 0.50  # positive = above breakeven
            adjustment = edge * 0.2  # Scale down: 10% edge → 0.02 weight boost

            for live_agent in live_agents:
                current = suggestions.get(live_agent, 0.0)
                suggestions[live_agent] = current + adjustment

        return suggestions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_patterns(self, result: ObservationalResult) -> None:
        """Persist patterns to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_trades_analyzed": result.total_synthetic_trades,
                "patterns": result.patterns,
                "weight_adjustments": result.suggested_weight_adjustments,
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save observational patterns: {e}")

    def load_patterns(self) -> List[ObservationalPattern]:
        """Load previously extracted patterns from disk."""
        if not self.persistence_path.exists():
            return []
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text())

            patterns = []
            for p in data.get("patterns", []):
                patterns.append(
                    ObservationalPattern(
                        pattern_type=p.get("pattern_type", ""),
                        description=p.get("description", ""),
                        agent_types=p.get("agent_types", []),
                        regime=p.get("regime", ""),
                        conditions=p.get("conditions", {}),
                        confidence=p.get("confidence", 0.0),
                        sample_size=p.get("sample_size", 0),
                    )
                )
            return patterns
        except Exception as e:
            logger.debug(f"Failed to load observational patterns: {e}")
            return []
