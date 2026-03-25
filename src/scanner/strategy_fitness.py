"""
Strategy Fitness Detector for Buddy FX Trading Bot.

Monitors rolling strategy fitness per pair per regime. Detects when
the system's edge has disappeared and reduces exposure accordingly.

US-302 (Phase 48): Trade Intelligence & Adaptive Filtering.

Research basis:
- Strategy alpha decay: swing systems lose edge in 6-18 months
- Rolling profit factor < 1.0 for 20+ trades = dead edge
- Agent disagreement trending up = model confusion
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FitnessMetrics:
    """Rolling fitness metrics for a pair/regime combo."""
    rolling_profit_factor: float = 1.0  # sum(wins) / sum(losses)
    prediction_correlation: float = 0.5  # how well predictions match outcomes
    avg_agent_disagreement: float = 0.2  # average ensemble disagreement
    trade_count: int = 0
    win_count: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    @property
    def fitness_score(self) -> float:
        """Weighted composite fitness score (0.0 - 1.0).

        Weights: profit_factor 0.4, correlation 0.3, disagreement 0.3
        """
        # Normalize profit factor to 0-1 range (PF 0→0, PF 2→1, PF 3+→1)
        pf_norm = min(self.rolling_profit_factor / 2.0, 1.0)

        # Correlation already 0-1
        corr_norm = max(0.0, min(self.prediction_correlation, 1.0))

        # Disagreement inverted (low disagreement = high fitness)
        disagree_norm = max(0.0, 1.0 - self.avg_agent_disagreement * 2.0)

        return 0.4 * pf_norm + 0.3 * corr_norm + 0.3 * disagree_norm


@dataclass
class FitnessCheckResult:
    """Result of a fitness evaluation."""
    action: str  # "NORMAL", "REDUCE_SIZE", "SKIP"
    fitness_score: float
    metrics: FitnessMetrics
    pair: str
    regime: str
    reason: str
    size_multiplier: float = 1.0

    @property
    def should_trade(self) -> bool:
        return self.action != "SKIP"


@dataclass
class StrategyFitnessConfig:
    """Configuration for strategy fitness detection."""

    # Rolling window size (trades)
    window_size: int = 20

    # Minimum trades before evaluation
    min_trades: int = 10

    # Fitness thresholds
    skip_threshold: float = 0.40     # Below this = SKIP pair
    reduce_threshold: float = 0.60   # Below this = REDUCE_SIZE
    # Above reduce_threshold = NORMAL

    # Size multiplier for REDUCE_SIZE
    reduce_size_multiplier: float = 0.50

    # Persistence
    state_file: str = "trained_data/strategy_fitness.json"

    # Enable/disable
    enabled: bool = True

    version: int = 1

    def validate(self) -> None:
        if self.window_size < 5:
            raise ValueError(f"window_size must be >= 5, got {self.window_size}")
        if self.min_trades < 3:
            raise ValueError(f"min_trades must be >= 3, got {self.min_trades}")
        if not 0.0 <= self.skip_threshold <= 1.0:
            raise ValueError(f"skip_threshold must be 0-1, got {self.skip_threshold}")
        if not 0.0 <= self.reduce_threshold <= 1.0:
            raise ValueError(f"reduce_threshold must be 0-1, got {self.reduce_threshold}")
        if self.skip_threshold >= self.reduce_threshold:
            raise ValueError("skip_threshold must be < reduce_threshold")


@dataclass
class TradeRecord:
    """Lightweight trade record for fitness tracking."""
    pair: str
    regime: str
    pnl: float
    won: bool
    predicted_direction: str  # "BUY" or "SELL"
    actual_direction: str     # "BUY" or "SELL" (based on outcome)
    agent_disagreement: float  # ensemble disagreement at entry
    timestamp: float = 0.0


class StrategyFitnessDetector:
    """Monitors rolling strategy fitness per pair per regime.

    Tracks profit factor, prediction-outcome correlation, and agent
    disagreement to detect when edge has disappeared.

    Usage:
        detector = StrategyFitnessDetector()
        detector.record_trade(trade_record)
        result = detector.check_fitness("EUR_USD", "NORMAL")
        if result.action == "SKIP":
            logger.info(f"Skipping {pair}: fitness too low")
    """

    def __init__(self, config: Optional[StrategyFitnessConfig] = None):
        self._config = config or StrategyFitnessConfig()
        try:
            self._config.validate()
        except ValueError as e:
            logger.warning(f"StrategyFitness config invalid: {e}, using defaults")
            self._config = StrategyFitnessConfig()

        # trades_by_key[pair:regime] = list of TradeRecord (bounded by window_size)
        self._trades: Dict[str, List[TradeRecord]] = {}
        self._load_state()

    @property
    def config(self) -> StrategyFitnessConfig:
        return self._config

    def _key(self, pair: str, regime: str) -> str:
        return f"{pair}:{regime}"

    def record_trade(self, record: TradeRecord) -> None:
        """Record a completed trade for fitness tracking."""
        key = self._key(record.pair, record.regime)
        if key not in self._trades:
            self._trades[key] = []

        self._trades[key].append(record)

        # Maintain rolling window
        if len(self._trades[key]) > self._config.window_size:
            self._trades[key] = self._trades[key][-self._config.window_size:]

    def _compute_metrics(self, trades: List[TradeRecord]) -> FitnessMetrics:
        """Compute fitness metrics from a list of trades."""
        if not trades:
            return FitnessMetrics()

        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]

        total_profit = sum(t.pnl for t in wins) if wins else 0.0
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0

        # Profit factor (avoid division by zero)
        if total_loss > 0:
            pf = total_profit / total_loss
        elif total_profit > 0:
            pf = 3.0  # All wins, cap at 3.0
        else:
            pf = 1.0  # No trades or all zero

        # Prediction correlation (direction match rate)
        direction_matches = sum(
            1 for t in trades
            if t.predicted_direction == t.actual_direction
        )
        pred_corr = direction_matches / len(trades) if trades else 0.5

        # Average agent disagreement
        avg_disagree = (
            sum(t.agent_disagreement for t in trades) / len(trades)
            if trades else 0.2
        )

        return FitnessMetrics(
            rolling_profit_factor=pf,
            prediction_correlation=pred_corr,
            avg_agent_disagreement=avg_disagree,
            trade_count=len(trades),
            win_count=len(wins),
            total_profit=total_profit,
            total_loss=total_loss,
        )

    def check_fitness(self, pair: str, regime: str) -> FitnessCheckResult:
        """Check strategy fitness for a pair/regime combination.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            regime: Market regime (e.g., "NORMAL", "HIGH")

        Returns:
            FitnessCheckResult with action and details
        """
        if not self._config.enabled:
            return FitnessCheckResult(
                action="NORMAL",
                fitness_score=1.0,
                metrics=FitnessMetrics(),
                pair=pair,
                regime=regime,
                reason="fitness_detector_disabled",
                size_multiplier=1.0,
            )

        key = self._key(pair, regime)
        trades = self._trades.get(key, [])

        if len(trades) < self._config.min_trades:
            return FitnessCheckResult(
                action="NORMAL",
                fitness_score=1.0,
                metrics=FitnessMetrics(trade_count=len(trades)),
                pair=pair,
                regime=regime,
                reason=f"insufficient_data ({len(trades)}/{self._config.min_trades} trades)",
                size_multiplier=1.0,
            )

        metrics = self._compute_metrics(trades)
        score = metrics.fitness_score

        if score < self._config.skip_threshold:
            action = "SKIP"
            multiplier = 0.0
            reason = (
                f"fitness_too_low: {score:.3f} < {self._config.skip_threshold} "
                f"(PF={metrics.rolling_profit_factor:.2f}, "
                f"corr={metrics.prediction_correlation:.2f}, "
                f"disagree={metrics.avg_agent_disagreement:.2f})"
            )
            logger.info(
                f"StrategyFitness SKIP {pair}/{regime}: score={score:.3f} "
                f"PF={metrics.rolling_profit_factor:.2f}"
            )
        elif score < self._config.reduce_threshold:
            action = "REDUCE_SIZE"
            multiplier = self._config.reduce_size_multiplier
            reason = (
                f"fitness_reduced: {score:.3f} < {self._config.reduce_threshold} "
                f"(PF={metrics.rolling_profit_factor:.2f})"
            )
        else:
            action = "NORMAL"
            multiplier = 1.0
            reason = f"fitness_ok: {score:.3f}"

        return FitnessCheckResult(
            action=action,
            fitness_score=score,
            metrics=metrics,
            pair=pair,
            regime=regime,
            reason=reason,
            size_multiplier=multiplier,
        )

    def get_all_fitness(self) -> Dict[str, FitnessCheckResult]:
        """Get fitness results for all tracked pair/regime combos."""
        results = {}
        for key in self._trades:
            pair, regime = key.split(":", 1)
            results[key] = self.check_fitness(pair, regime)
        return results

    def _load_state(self) -> None:
        """Load persisted fitness state."""
        state_path = Path(self._config.state_file)
        if not state_path.exists():
            return

        try:
            with open(state_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning("Fitness state is not a dict, ignoring")
                return

            # Check freshness
            saved_at = data.get("saved_at", 0)
            age_hours = (time.time() - saved_at) / 3600.0
            if age_hours > 1.0:
                logger.warning(f"Fitness state is {age_hours:.1f}h old (stale)")

            trades_data = data.get("trades", {})
            for key, trade_list in trades_data.items():
                self._trades[key] = []
                for td in trade_list:
                    try:
                        self._trades[key].append(TradeRecord(
                            pair=td["pair"],
                            regime=td["regime"],
                            pnl=td["pnl"],
                            won=td["won"],
                            predicted_direction=td.get("predicted_direction", "BUY"),
                            actual_direction=td.get("actual_direction", "BUY"),
                            agent_disagreement=td.get("agent_disagreement", 0.2),
                            timestamp=td.get("timestamp", 0.0),
                        ))
                    except (KeyError, TypeError) as e:
                        logger.debug(f"Skipping malformed trade record: {e}")

            logger.info(f"Loaded fitness state: {len(self._trades)} pair/regime combos")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load fitness state: {e}")

    def save_state(self) -> None:
        """Persist fitness state (atomic write)."""
        state_path = Path(self._config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        trades_data = {}
        for key, trade_list in self._trades.items():
            trades_data[key] = [
                {
                    "pair": t.pair,
                    "regime": t.regime,
                    "pnl": t.pnl,
                    "won": t.won,
                    "predicted_direction": t.predicted_direction,
                    "actual_direction": t.actual_direction,
                    "agent_disagreement": t.agent_disagreement,
                    "timestamp": t.timestamp,
                }
                for t in trade_list
            ]

        data = {
            "version": self._config.version,
            "saved_at": time.time(),
            "trades": trades_data,
        }

        tmp_path = str(state_path) + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, str(state_path))
        except OSError as e:
            logger.warning(f"Failed to save fitness state: {e}")


def create_default_fitness_detector(
    config: Optional[StrategyFitnessConfig] = None,
) -> StrategyFitnessDetector:
    """Factory function for StrategyFitnessDetector."""
    return StrategyFitnessDetector(config=config)
