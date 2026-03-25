"""
Walk-Forward Gate Parameter Optimizer for Buddy FX Trading Bot.

Implements anchored walk-forward optimization for gate thresholds and
ATR multipliers. Uses rolling in-sample/out-of-sample windows to find
optimal parameters while detecting overfitting via Walk-Forward Efficiency.

US-305 (Phase 48): Trade Intelligence & Adaptive Filtering.

Research basis:
- Walk-forward analysis prevents curve-fitting to historical data
- WFE (Walk-Forward Efficiency) = OOS_metric / IS_metric
- WFE < 0.30 indicates overfitting
- Grid search on small parameter space (brute force, not Optuna)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Simplified trade outcome for walk-forward optimization."""
    pair: str
    direction: str       # "BUY" or "SELL"
    pnl: float
    confidence: float    # Gate confidence at entry
    momentum_score: float  # Gate momentum at entry
    atr_sl_pips: float   # SL in pips
    atr_tp_pips: float   # TP in pips
    regime: str
    won: bool
    timestamp: float = 0.0


@dataclass
class ParameterSet:
    """A candidate set of gate parameters."""
    confidence_threshold: float = 0.50
    momentum_threshold: float = 0.50
    atr_sl_multiplier: float = 1.0
    atr_tp_multiplier: float = 1.5

    def as_dict(self) -> Dict:
        return {
            "confidence_threshold": self.confidence_threshold,
            "momentum_threshold": self.momentum_threshold,
            "atr_sl_multiplier": self.atr_sl_multiplier,
            "atr_tp_multiplier": self.atr_tp_multiplier,
        }


@dataclass
class OptimizationResult:
    """Result of a walk-forward optimization run."""
    best_params: ParameterSet
    is_profit_factor: float   # In-sample profit factor
    oos_profit_factor: float  # Out-of-sample profit factor
    wfe: float                # Walk-Forward Efficiency
    is_overfitting: bool      # WFE < 0.30
    is_trade_count: int
    oos_trade_count: int
    recommendation: str       # "APPLY", "REVIEW", "REJECT"
    timestamp: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "best_params": self.best_params.as_dict(),
            "is_profit_factor": round(self.is_profit_factor, 3),
            "oos_profit_factor": round(self.oos_profit_factor, 3),
            "wfe": round(self.wfe, 3),
            "is_overfitting": self.is_overfitting,
            "is_trade_count": self.is_trade_count,
            "oos_trade_count": self.oos_trade_count,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
        }


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimizer."""

    # Window sizes
    in_sample_size: int = 60     # IS window (trades)
    out_of_sample_size: int = 15  # OOS window (trades)
    trigger_interval: int = 75    # Run every N closed trades

    # Parameter grid ranges
    confidence_range: Tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60)
    momentum_range: Tuple[float, ...] = (0.40, 0.45, 0.50, 0.55)
    atr_sl_range: Tuple[float, ...] = (0.8, 1.0, 1.2, 1.5)
    atr_tp_range: Tuple[float, ...] = (1.2, 1.5, 1.8, 2.0)

    # WFE thresholds
    wfe_overfit_threshold: float = 0.30  # Below this = overfitting
    wfe_good_threshold: float = 0.50     # Above this = good generalization

    # Minimum profit factor to consider
    min_profit_factor: float = 1.0

    # Persistence
    state_file: str = "trained_data/walkforward_state.json"
    recommendations_file: str = ".claude/config_adjustments.json"

    enabled: bool = True
    version: int = 1

    def validate(self) -> None:
        if self.in_sample_size < 20:
            raise ValueError(f"in_sample_size must be >= 20, got {self.in_sample_size}")
        if self.out_of_sample_size < 5:
            raise ValueError(f"out_of_sample_size must be >= 5, got {self.out_of_sample_size}")
        if self.wfe_overfit_threshold < 0 or self.wfe_overfit_threshold > 1:
            raise ValueError(f"wfe_overfit_threshold must be 0-1, got {self.wfe_overfit_threshold}")


class WalkForwardOptimizer:
    """Anchored walk-forward optimization for gate parameters.

    Uses rolling windows of historical trade outcomes to find
    optimal gate thresholds via grid search, then validates on
    held-out data. Reports Walk-Forward Efficiency to detect overfitting.

    IMPORTANT: Does NOT auto-apply parameters. Writes recommendations
    to config_adjustments.json for manual review.

    Usage:
        optimizer = WalkForwardOptimizer()
        optimizer.record_outcome(trade_outcome)
        if optimizer.should_optimize():
            result = optimizer.run_optimization()
            if result.recommendation == "APPLY":
                logger.info(f"Recommended params: {result.best_params}")
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self._config = config or WalkForwardConfig()
        try:
            self._config.validate()
        except ValueError as e:
            logger.warning(f"WalkForward config invalid: {e}, using defaults")
            self._config = WalkForwardConfig()

        self._outcomes: List[TradeOutcome] = []
        self._trades_since_last_opt: int = 0
        self._last_result: Optional[OptimizationResult] = None
        self._load_state()

    @property
    def config(self) -> WalkForwardConfig:
        return self._config

    @property
    def last_result(self) -> Optional[OptimizationResult]:
        return self._last_result

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """Record a trade outcome for optimization."""
        self._outcomes.append(outcome)
        self._trades_since_last_opt += 1

        # Keep bounded history (2x the required window)
        max_size = (self._config.in_sample_size + self._config.out_of_sample_size) * 2
        if len(self._outcomes) > max_size:
            self._outcomes = self._outcomes[-max_size:]

    def should_optimize(self) -> bool:
        """Check if we have enough data and it's time to optimize."""
        if not self._config.enabled:
            return False
        min_needed = self._config.in_sample_size + self._config.out_of_sample_size
        if len(self._outcomes) < min_needed:
            return False
        return self._trades_since_last_opt >= self._config.trigger_interval

    def run_optimization(self) -> OptimizationResult:
        """Run walk-forward optimization.

        Splits data into IS/OOS windows, grid-searches on IS,
        validates on OOS, computes WFE.
        """
        min_needed = self._config.in_sample_size + self._config.out_of_sample_size
        if len(self._outcomes) < min_needed:
            return OptimizationResult(
                best_params=ParameterSet(),
                is_profit_factor=1.0,
                oos_profit_factor=1.0,
                wfe=1.0,
                is_overfitting=False,
                is_trade_count=0,
                oos_trade_count=0,
                recommendation="REJECT",
                timestamp=time.time(),
            )

        # Split into IS and OOS
        oos_trades = self._outcomes[-self._config.out_of_sample_size:]
        is_trades = self._outcomes[
            -(self._config.in_sample_size + self._config.out_of_sample_size):
            -self._config.out_of_sample_size
        ]

        # Grid search on IS
        best_params, best_is_pf = self._grid_search(is_trades)

        # Evaluate best params on OOS
        oos_pf = self._evaluate_params(best_params, oos_trades)

        # Calculate WFE
        if best_is_pf > 0:
            wfe = oos_pf / best_is_pf
        else:
            wfe = 0.0

        is_overfitting = wfe < self._config.wfe_overfit_threshold

        # Determine recommendation
        if is_overfitting:
            recommendation = "REJECT"
        elif wfe >= self._config.wfe_good_threshold and oos_pf >= self._config.min_profit_factor:
            recommendation = "APPLY"
        elif oos_pf >= self._config.min_profit_factor:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"

        result = OptimizationResult(
            best_params=best_params,
            is_profit_factor=best_is_pf,
            oos_profit_factor=oos_pf,
            wfe=wfe,
            is_overfitting=is_overfitting,
            is_trade_count=len(is_trades),
            oos_trade_count=len(oos_trades),
            recommendation=recommendation,
            timestamp=time.time(),
        )

        self._last_result = result
        self._trades_since_last_opt = 0

        logger.info(
            f"WalkForward optimization complete: "
            f"IS_PF={best_is_pf:.2f}, OOS_PF={oos_pf:.2f}, "
            f"WFE={wfe:.2f}, rec={recommendation}"
        )

        # Write recommendation (does NOT auto-apply)
        self._write_recommendation(result)

        return result

    def _grid_search(
        self,
        trades: List[TradeOutcome],
    ) -> Tuple[ParameterSet, float]:
        """Grid search over parameter space on IS trades."""
        best_pf = 0.0
        best_params = ParameterSet()

        grid = product(
            self._config.confidence_range,
            self._config.momentum_range,
            self._config.atr_sl_range,
            self._config.atr_tp_range,
        )

        for conf, mom, sl, tp in grid:
            params = ParameterSet(
                confidence_threshold=conf,
                momentum_threshold=mom,
                atr_sl_multiplier=sl,
                atr_tp_multiplier=tp,
            )
            pf = self._evaluate_params(params, trades)
            if pf > best_pf:
                best_pf = pf
                best_params = params

        return best_params, best_pf

    def _evaluate_params(
        self,
        params: ParameterSet,
        trades: List[TradeOutcome],
    ) -> float:
        """Evaluate a parameter set against trade outcomes.

        Simulates which trades would have been taken/filtered
        with the given parameters and computes profit factor.
        """
        total_profit = 0.0
        total_loss = 0.0

        for trade in trades:
            # Would this trade pass the gates with these params?
            if trade.confidence < params.confidence_threshold:
                continue  # Would have been filtered
            if trade.momentum_score < params.momentum_threshold:
                continue  # Would have been filtered

            # R:R check with these ATR multipliers
            rr = params.atr_tp_multiplier / params.atr_sl_multiplier
            if rr < 1.2:
                continue  # Below minimum R:R

            if trade.won:
                total_profit += abs(trade.pnl)
            else:
                total_loss += abs(trade.pnl)

        if total_loss <= 0:
            return 3.0 if total_profit > 0 else 1.0

        return total_profit / total_loss

    def _write_recommendation(self, result: OptimizationResult) -> None:
        """Write recommendation to config_adjustments.json (does NOT auto-apply)."""
        rec_path = Path(self._config.recommendations_file)
        rec_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing adjustments
        existing = {}
        if rec_path.exists():
            try:
                with open(rec_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        # Add walk-forward recommendation
        if "walkforward_recommendations" not in existing:
            existing["walkforward_recommendations"] = []

        existing["walkforward_recommendations"].append(result.as_dict())

        # Keep last 10 recommendations
        existing["walkforward_recommendations"] = existing["walkforward_recommendations"][-10:]

        tmp_path = str(rec_path) + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(existing, f, indent=2, sort_keys=True)
            os.rename(tmp_path, str(rec_path))
        except OSError as e:
            logger.warning(f"Failed to write walkforward recommendation: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        state_path = Path(self._config.state_file)
        if not state_path.exists():
            return

        try:
            with open(state_path, "r") as f:
                data = json.load(f)

            for item in data.get("outcomes", []):
                try:
                    self._outcomes.append(TradeOutcome(
                        pair=item["pair"],
                        direction=item["direction"],
                        pnl=item["pnl"],
                        confidence=item.get("confidence", 0.5),
                        momentum_score=item.get("momentum_score", 0.5),
                        atr_sl_pips=item.get("atr_sl_pips", 15.0),
                        atr_tp_pips=item.get("atr_tp_pips", 25.0),
                        regime=item.get("regime", "NORMAL"),
                        won=item.get("won", False),
                        timestamp=item.get("timestamp", 0.0),
                    ))
                except (KeyError, TypeError) as e:
                    logger.debug(f"Skipping malformed outcome: {e}")

            self._trades_since_last_opt = data.get("trades_since_last_opt", 0)
            logger.info(f"Loaded walkforward state: {len(self._outcomes)} outcomes")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load walkforward state: {e}")

    def save_state(self) -> None:
        """Persist state (atomic write)."""
        state_path = Path(self._config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        outcomes_data = [
            {
                "pair": o.pair,
                "direction": o.direction,
                "pnl": o.pnl,
                "confidence": o.confidence,
                "momentum_score": o.momentum_score,
                "atr_sl_pips": o.atr_sl_pips,
                "atr_tp_pips": o.atr_tp_pips,
                "regime": o.regime,
                "won": o.won,
                "timestamp": o.timestamp,
            }
            for o in self._outcomes
        ]

        data = {
            "version": self._config.version,
            "saved_at": time.time(),
            "outcomes": outcomes_data,
            "trades_since_last_opt": self._trades_since_last_opt,
        }

        tmp_path = str(state_path) + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, str(state_path))
        except OSError as e:
            logger.warning(f"Failed to save walkforward state: {e}")


def create_default_walkforward_optimizer(
    config: Optional[WalkForwardConfig] = None,
) -> WalkForwardOptimizer:
    """Factory function for WalkForwardOptimizer."""
    return WalkForwardOptimizer(config=config)
