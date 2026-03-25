"""Dynamic risk allocation from P/L distribution.

US-111: Adapt position sizing based on recent P/L distribution and
trade duration patterns. Analyzes rolling P/L to detect hot/cold
streaks and adjusts risk allocation with Kelly-inspired but capped sizing.

Approach:
1. Record trade outcomes (pnl, duration, regime)
2. Compute rolling P/L statistics per regime
3. Detect streaks: 3+ consecutive wins or losses
4. Compute risk multiplier [0.50, 1.30]:
   - Hot streak + good Sharpe → increase (up to 1.30)
   - Cold streak + negative momentum → decrease (down to 0.50)
   - Normal → 1.0
5. Kelly fraction as upper bound (half-Kelly for safety)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "dynamic_risk_allocator_log.json"

# Rolling window parameters
ROLLING_WINDOW = 30
MIN_TRADES = 10  # Minimum trades before allocation activates

# Risk multiplier bounds
MIN_MULTIPLIER = 0.50
MAX_MULTIPLIER = 1.30

# Streak detection
STREAK_THRESHOLD = 3  # Consecutive wins/losses to trigger adjustment

# Kelly parameters
KELLY_FRACTION = 0.5  # Half-Kelly for safety
MAX_KELLY_MULTIPLIER = 1.50  # Cap Kelly-derived multiplier

# Momentum detection
MOMENTUM_WINDOW = 10  # Recent trades for momentum calculation


class DynamicRiskAllocator:
    """Adapts position sizing based on P/L distribution and streaks.

    Bridges trade outcome analysis with position sizing. Uses streak
    detection and rolling P/L statistics to dynamically adjust risk
    exposure per regime.

    Usage:
        allocator = DynamicRiskAllocator()
        allocator.record_trade(15.3, 12, "NORMAL")  # +15.3 pips, 12 bars
        multiplier = allocator.get_risk_multiplier("NORMAL")
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        # Trade history: {regime: [records]}
        self._trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Allocation log
        self._allocation_log: List[Dict[str, Any]] = []
        self._total_trades: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(
        self,
        pnl_pips: float,
        duration_bars: int,
        regime: str,
    ) -> None:
        """Record a completed trade for P/L distribution tracking.

        Args:
            pnl_pips: Profit/loss in pips (positive=win, negative=loss).
            duration_bars: Number of bars the trade was open.
            regime: Market regime during the trade.
        """
        record = {
            "pnl": pnl_pips,
            "duration": max(1, duration_bars),
            "won": pnl_pips > 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._trades[regime].append(record)

        # Trim to rolling window
        if len(self._trades[regime]) > ROLLING_WINDOW * 2:
            self._trades[regime] = self._trades[regime][-ROLLING_WINDOW:]

        self._total_trades += 1

    def get_risk_multiplier(self, regime: str) -> float:
        """Compute position size multiplier based on P/L analysis.

        Args:
            regime: Current market regime.

        Returns:
            Multiplier [0.50, 1.30] for position sizing.
        """
        trades = self._trades.get(regime, [])

        if len(trades) < MIN_TRADES:
            return 1.0

        recent = trades[-ROLLING_WINDOW:]

        # 1. Streak analysis
        streak_mult = self._compute_streak_multiplier(recent)

        # 2. P/L momentum
        momentum_mult = self._compute_momentum_multiplier(recent)

        # 3. Kelly fraction (capped)
        kelly_mult = self._compute_kelly_multiplier(recent)

        # Combine: average of streak and momentum, bounded by Kelly
        raw_mult = 0.5 * streak_mult + 0.5 * momentum_mult
        bounded_mult = min(raw_mult, kelly_mult)

        final = float(np.clip(bounded_mult, MIN_MULTIPLIER, MAX_MULTIPLIER))
        final = round(final, 4)

        # Log allocation decision
        self._allocation_log.append({
            "regime": regime,
            "multiplier": final,
            "streak_mult": round(streak_mult, 4),
            "momentum_mult": round(momentum_mult, 4),
            "kelly_mult": round(kelly_mult, 4),
            "n_trades": len(recent),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._allocation_log) > 300:
            self._allocation_log = self._allocation_log[-150:]

        return final

    def get_pnl_statistics(self, regime: str) -> Dict[str, Any]:
        """Get P/L distribution statistics for a regime."""
        trades = self._trades.get(regime, [])
        if not trades:
            return {"n_trades": 0}

        recent = trades[-ROLLING_WINDOW:]
        pnls = [r["pnl"] for r in recent]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "n_trades": len(recent),
            "win_rate": round(len(wins) / len(recent), 4) if recent else 0.0,
            "avg_pnl": round(float(np.mean(pnls)), 2),
            "std_pnl": round(float(np.std(pnls)), 2) if len(pnls) > 1 else 0.0,
            "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
            "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
            "current_streak": self._get_current_streak(recent),
            "sharpe": self._compute_sharpe(pnls),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get allocator status for observability."""
        return {
            "total_trades": self._total_trades,
            "regimes": {
                regime: len(trades)
                for regime, trades in self._trades.items()
            },
            "recent_allocations": self._allocation_log[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_streak_multiplier(self, recent: List[Dict[str, Any]]) -> float:
        """Compute multiplier based on current win/loss streak."""
        streak = self._get_current_streak(recent)

        if streak >= STREAK_THRESHOLD:
            # Winning streak → cautious increase
            boost = 0.05 * min(streak - STREAK_THRESHOLD + 1, 4)
            return 1.0 + boost
        elif streak <= -STREAK_THRESHOLD:
            # Losing streak → reduce
            reduction = 0.08 * min(abs(streak) - STREAK_THRESHOLD + 1, 4)
            return 1.0 - reduction
        else:
            return 1.0

    def _compute_momentum_multiplier(self, recent: List[Dict[str, Any]]) -> float:
        """Compute multiplier based on recent P/L momentum."""
        if len(recent) < MOMENTUM_WINDOW:
            return 1.0

        momentum_trades = recent[-MOMENTUM_WINDOW:]
        pnls = [r["pnl"] for r in momentum_trades]
        avg_pnl = float(np.mean(pnls))

        # Normalize: positive avg → boost, negative → reduce
        if avg_pnl > 5.0:
            return min(1.0 + avg_pnl / 100.0, MAX_MULTIPLIER)
        elif avg_pnl < -5.0:
            return max(1.0 + avg_pnl / 80.0, MIN_MULTIPLIER)
        else:
            return 1.0

    def _compute_kelly_multiplier(self, recent: List[Dict[str, Any]]) -> float:
        """Compute Kelly-derived upper bound on risk multiplier."""
        wins = [r for r in recent if r["won"]]
        losses = [r for r in recent if not r["won"]]

        if not wins or not losses:
            return 1.0

        win_rate = len(wins) / len(recent)
        avg_win = float(np.mean([r["pnl"] for r in wins]))
        avg_loss = abs(float(np.mean([r["pnl"] for r in losses])))

        if avg_loss == 0:
            return MAX_KELLY_MULTIPLIER

        # Kelly fraction: f* = (bp - q) / b where b = avg_win/avg_loss
        b = avg_win / avg_loss
        kelly = (b * win_rate - (1 - win_rate)) / b

        # Negative Kelly = strategy has negative expected value in this regime
        if kelly < 0:
            logger.warning(
                f"DynamicRiskAllocator: Negative Kelly ({kelly:.3f}) — "
                f"strategy has negative EV (win_rate={win_rate:.2f}, b={b:.2f}). "
                f"Returning MIN_MULTIPLIER."
            )
            return MIN_MULTIPLIER

        # Half-Kelly for safety, capped
        half_kelly = kelly * KELLY_FRACTION
        multiplier = 1.0 + half_kelly

        return float(np.clip(multiplier, MIN_MULTIPLIER, MAX_KELLY_MULTIPLIER))

    def _get_current_streak(self, recent: List[Dict[str, Any]]) -> int:
        """Get current streak. Positive = winning, negative = losing."""
        if not recent:
            return 0

        streak = 0
        last_won = recent[-1]["won"]

        for trade in reversed(recent):
            if trade["won"] == last_won:
                streak += 1
            else:
                break

        return streak if last_won else -streak

    def _compute_sharpe(self, pnls: List[float]) -> float:
        """Compute simple Sharpe ratio of P/L series."""
        if len(pnls) < 2:
            return 0.0
        mean = float(np.mean(pnls))
        std = float(np.std(pnls))
        if std == 0:
            return 0.0
        return round(mean / std, 4)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist allocation state."""
        try:
            trades_plain = {
                regime: records[-ROLLING_WINDOW:]
                for regime, records in self._trades.items()
            }

            data = {
                "version": 1,
                "total_trades": self._total_trades,
                "trades": trades_plain,
                "allocation_log": self._allocation_log[-150:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"DynamicRiskAllocator: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                self._total_trades = data.get("total_trades", 0)
                self._allocation_log = data.get("allocation_log", [])
                raw_trades = data.get("trades", {})
                for regime, records in raw_trades.items():
                    if not isinstance(records, list):
                        continue
                    valid = [
                        r for r in records
                        if isinstance(r, dict)
                        and all(k in r for k in ("pnl", "won", "duration"))
                    ]
                    if len(valid) < len(records):
                        logger.warning(
                            f"DynamicRiskAllocator: Skipped {len(records) - len(valid)} "
                            f"malformed records for regime {regime}"
                        )
                    self._trades[regime] = valid
        except Exception as e:
            logger.warning(f"DynamicRiskAllocator: Failed to load: {e}")
