"""Adaptive position timeout with time-decay exits.

US-086: Regime-dependent time-decay for open trades. Trades that age
without reaching TP lose confidence. Regime-specific schedules:
- trending: 100 bars max, slow decay (0.98/bar)
- mean_revert: 30 bars max, fast decay (0.92/bar)
- crisis: 10 bars max, very fast decay (0.85/bar)
- choppy: 40 bars max, moderate decay (0.95/bar)

Force-exit when confidence decays below 0.20 or exceeds regime max_bars.
Based on triple-barrier method research.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Regime-specific decay schedules
DECAY_SCHEDULES: Dict[str, Dict[str, float]] = {
    "trending": {
        "max_bars": 100,
        "decay_rate": 0.98,      # Slow: 2% per bar
        "force_exit_threshold": 0.20,
        "tp_decay_start_bars": 50,  # Start adjusting TP after 50 bars
    },
    "mean_revert": {
        "max_bars": 30,
        "decay_rate": 0.92,      # Fast: 8% per bar
        "force_exit_threshold": 0.20,
        "tp_decay_start_bars": 15,
    },
    "crisis": {
        "max_bars": 10,
        "decay_rate": 0.85,      # Very fast: 15% per bar
        "force_exit_threshold": 0.25,
        "tp_decay_start_bars": 3,
    },
    "choppy": {
        "max_bars": 40,
        "decay_rate": 0.95,      # Moderate: 5% per bar
        "force_exit_threshold": 0.20,
        "tp_decay_start_bars": 20,
    },
    "normal": {
        "max_bars": 50,
        "decay_rate": 0.96,
        "force_exit_threshold": 0.20,
        "tp_decay_start_bars": 25,
    },
}

# Minimum confidence decay before any penalty applies
GRACE_PERIOD_BARS = 5


@dataclass
class TimeoutResult:
    """Result of position timeout evaluation."""

    trade_id: str = ""
    bars_open: int = 0
    confidence_decay: float = 1.0  # Multiplier 0.0-1.0
    should_exit: bool = False
    exit_reason: str = ""
    adjusted_tp_multiplier: float = 1.0  # TP gets tighter as trade ages
    regime: str = "normal"

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "bars_open": self.bars_open,
            "confidence_decay": round(self.confidence_decay, 4),
            "should_exit": self.should_exit,
            "exit_reason": self.exit_reason,
            "adjusted_tp_multiplier": round(self.adjusted_tp_multiplier, 4),
            "regime": self.regime,
        }


@dataclass
class TrackedPosition:
    """Internal tracking for an open position."""

    trade_id: str = ""
    entry_bar: int = 0
    regime: str = "normal"
    initial_confidence: float = 0.5
    pair: str = ""
    direction: str = ""


class PositionTimeoutManager:
    """Manages time-decay for open trading positions.

    Tracks all open positions and evaluates decay on each scan cycle.
    Regime-specific schedules control how quickly confidence erodes
    and when force-exits are triggered.
    """

    def __init__(self):
        self._positions: Dict[str, TrackedPosition] = {}
        self._bar_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_trade(
        self,
        trade_id: str,
        entry_bar: Optional[int] = None,
        regime: str = "normal",
        confidence: float = 0.5,
        pair: str = "",
        direction: str = "",
    ) -> None:
        """Start tracking a new open position.

        Args:
            trade_id: Unique trade identifier
            entry_bar: Bar index at entry (uses internal counter if None)
            regime: Current volatility regime
            confidence: Entry confidence
            pair: FX pair
            direction: LONG or SHORT
        """
        self._positions[trade_id] = TrackedPosition(
            trade_id=trade_id,
            entry_bar=entry_bar if entry_bar is not None else self._bar_counter,
            regime=regime.lower() if regime else "normal",
            initial_confidence=confidence,
            pair=pair,
            direction=direction,
        )
        logger.debug(
            f"PositionTimeout: Registered {trade_id} ({pair} {direction}) "
            f"at bar {self._bar_counter}, regime={regime}"
        )

    def evaluate(
        self,
        trade_id: str,
        current_bar: Optional[int] = None,
        pnl_pips: float = 0.0,
    ) -> TimeoutResult:
        """Evaluate timeout for an open position.

        Args:
            trade_id: Trade to evaluate
            current_bar: Current bar index (uses internal counter if None)
            pnl_pips: Current PnL in pips

        Returns:
            TimeoutResult with decay, exit signal, and adjusted TP
        """
        result = TimeoutResult(trade_id=trade_id)

        pos = self._positions.get(trade_id)
        if not pos:
            result.exit_reason = "Unknown trade ID"
            return result

        bar = current_bar if current_bar is not None else self._bar_counter
        bars_open = max(0, bar - pos.entry_bar)
        result.bars_open = bars_open
        result.regime = pos.regime

        # Get regime schedule
        schedule = DECAY_SCHEDULES.get(pos.regime, DECAY_SCHEDULES["normal"])

        # Grace period: no decay in first N bars
        if bars_open <= GRACE_PERIOD_BARS:
            result.confidence_decay = 1.0
            return result

        # Compute exponential decay
        effective_bars = bars_open - GRACE_PERIOD_BARS
        decay = schedule["decay_rate"] ** effective_bars
        result.confidence_decay = max(0.0, min(1.0, decay))

        # TP adjustment: move TP closer as trade ages
        if bars_open > schedule["tp_decay_start_bars"]:
            tp_bars = bars_open - schedule["tp_decay_start_bars"]
            tp_decay = 0.99 ** tp_bars  # Gentle TP tightening
            result.adjusted_tp_multiplier = max(0.5, tp_decay)

        # Force exit conditions
        if bars_open > schedule["max_bars"]:
            result.should_exit = True
            result.exit_reason = (
                f"Exceeded max bars for {pos.regime} regime "
                f"({bars_open}/{schedule['max_bars']})"
            )
        elif result.confidence_decay < schedule["force_exit_threshold"]:
            result.should_exit = True
            result.exit_reason = (
                f"Confidence decayed to {result.confidence_decay:.2f} "
                f"(threshold={schedule['force_exit_threshold']})"
            )
        elif pnl_pips < 0 and result.confidence_decay < 0.30:
            result.should_exit = True
            result.exit_reason = (
                f"Negative PnL ({pnl_pips:.1f}) + low decay ({result.confidence_decay:.2f})"
            )

        if result.should_exit:
            logger.info(
                f"PositionTimeout: EXIT signal for {trade_id} — "
                f"{result.exit_reason} (bars={bars_open})"
            )

        return result

    def evaluate_all(
        self,
        current_bar: Optional[int] = None,
        pnl_by_trade: Optional[Dict[str, float]] = None,
    ) -> List[TimeoutResult]:
        """Evaluate all tracked positions.

        Returns:
            List of TimeoutResults, one per tracked position
        """
        results = []
        pnls = pnl_by_trade or {}

        for trade_id in list(self._positions.keys()):
            pnl = pnls.get(trade_id, 0.0)
            result = self.evaluate(trade_id, current_bar, pnl)
            results.append(result)

        return results

    def close_trade(self, trade_id: str) -> None:
        """Remove a closed trade from tracking."""
        if trade_id in self._positions:
            del self._positions[trade_id]

    def tick(self) -> None:
        """Advance the internal bar counter by 1."""
        self._bar_counter += 1

    def get_open_count(self) -> int:
        """Number of currently tracked positions."""
        return len(self._positions)

    def get_status(self) -> Dict[str, Any]:
        """Summary of all tracked positions."""
        return {
            "open_positions": self.get_open_count(),
            "bar_counter": self._bar_counter,
            "positions": {
                tid: {
                    "pair": p.pair,
                    "direction": p.direction,
                    "regime": p.regime,
                    "bars_open": self._bar_counter - p.entry_bar,
                }
                for tid, p in self._positions.items()
            },
        }
