"""
Dynamic Pair Performance Tracker & Blacklisting for Buddy FX Trading Bot.

Tracks per-pair performance metrics and auto-blacklists underperformers.
Prevents continued trading of pairs where the system consistently loses.

US-303 (Phase 48): Trade Intelligence & Adaptive Filtering.
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
class PairMetrics:
    """Performance metrics for a single pair."""
    pair: str
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    peak_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_count: int = 0
    last_updated: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.wins / self.trade_count

    @property
    def profit_factor(self) -> float:
        if self.total_loss <= 0:
            return 3.0 if self.total_profit > 0 else 1.0
        return self.total_profit / self.total_loss

    @property
    def sharpe_estimate(self) -> float:
        """Simplified Sharpe estimate: avg_pnl / rough_stddev.

        Uses profit factor as a proxy since we don't store per-trade PnL.
        """
        if self.trade_count < 5:
            return 0.0
        avg_pnl = self.total_pnl / self.trade_count
        # Rough stddev proxy from PF and trade count
        if self.total_loss <= 0:
            return 2.0 if avg_pnl > 0 else 0.0
        avg_win = self.total_profit / max(self.wins, 1)
        avg_loss = self.total_loss / max(self.losses, 1)
        rough_std = (avg_win + avg_loss) / 2.0
        if rough_std <= 0:
            return 0.0
        return avg_pnl / rough_std


@dataclass
class BlacklistCheckResult:
    """Result of a blacklist check."""
    is_blacklisted: bool
    pair: str
    reason: str
    metrics: Optional[PairMetrics] = None
    cooldown_remaining: int = 0  # scan cycles remaining


@dataclass
class PairBlacklistConfig:
    """Configuration for pair blacklisting."""

    # Blacklist triggers
    min_sharpe: float = 0.5          # Below this = blacklist candidate
    max_consecutive_losses: int = 4   # 4+ consecutive losses = blacklist
    max_drawdown_pct: float = 10.0    # 10%+ drawdown = blacklist

    # Minimum trades before blacklisting
    min_trades: int = 10

    # Cooldown (scan cycles before retrying a blacklisted pair)
    cooldown_cycles: int = 2

    # Persistence
    state_file: str = "trained_data/models/pair_tracker.json"

    enabled: bool = True
    version: int = 1

    def validate(self) -> None:
        if self.min_trades < 3:
            raise ValueError(f"min_trades must be >= 3, got {self.min_trades}")
        if self.cooldown_cycles < 1:
            raise ValueError(f"cooldown_cycles must be >= 1, got {self.cooldown_cycles}")
        if not 0.0 <= self.max_drawdown_pct <= 100.0:
            raise ValueError(f"max_drawdown_pct must be 0-100, got {self.max_drawdown_pct}")


class PairPerformanceTracker:
    """Tracks per-pair performance and manages automatic blacklisting.

    Usage:
        tracker = PairPerformanceTracker()
        tracker.update_from_trade("EUR_USD", pnl=15.5, won=True)
        result = tracker.check_pair("EUR_USD")
        if result.is_blacklisted:
            logger.info(f"Skipping {pair}: {result.reason}")
    """

    def __init__(self, config: Optional[PairBlacklistConfig] = None):
        self._config = config or PairBlacklistConfig()
        try:
            self._config.validate()
        except ValueError as e:
            logger.warning(f"PairBlacklist config invalid: {e}, using defaults")
            self._config = PairBlacklistConfig()

        self._metrics: Dict[str, PairMetrics] = {}
        # blacklist[pair] = remaining_cooldown_cycles
        self._blacklist: Dict[str, int] = {}
        self._load_state()

    @property
    def config(self) -> PairBlacklistConfig:
        return self._config

    @property
    def metrics(self) -> Dict[str, PairMetrics]:
        return self._metrics

    def update_from_trade(self, pair: str, pnl: float, won: bool) -> None:
        """Record a completed trade outcome for a pair."""
        if pair not in self._metrics:
            self._metrics[pair] = PairMetrics(pair=pair)

        m = self._metrics[pair]
        m.trade_count += 1
        m.total_pnl += pnl
        m.last_updated = time.time()

        if won:
            m.wins += 1
            m.total_profit += abs(pnl)
            m.consecutive_losses = 0
        else:
            m.losses += 1
            m.total_loss += abs(pnl)
            m.consecutive_losses += 1
            m.max_consecutive_losses = max(
                m.max_consecutive_losses, m.consecutive_losses
            )

        # Update peak and drawdown
        if m.total_pnl > m.peak_pnl:
            m.peak_pnl = m.total_pnl
        if m.peak_pnl > 0:
            dd = (m.peak_pnl - m.total_pnl) / m.peak_pnl * 100.0
            m.max_drawdown_pct = max(m.max_drawdown_pct, dd)

        # Check if should be blacklisted
        self._evaluate_blacklist(pair)

    def _evaluate_blacklist(self, pair: str) -> None:
        """Evaluate if a pair should be blacklisted."""
        m = self._metrics.get(pair)
        if not m or m.trade_count < self._config.min_trades:
            return

        reasons = []

        if m.sharpe_estimate < self._config.min_sharpe:
            reasons.append(f"sharpe={m.sharpe_estimate:.2f}<{self._config.min_sharpe}")

        if m.consecutive_losses >= self._config.max_consecutive_losses:
            reasons.append(
                f"consec_losses={m.consecutive_losses}>="
                f"{self._config.max_consecutive_losses}"
            )

        if m.max_drawdown_pct > self._config.max_drawdown_pct:
            reasons.append(
                f"drawdown={m.max_drawdown_pct:.1f}%>"
                f"{self._config.max_drawdown_pct}%"
            )

        if reasons:
            self._blacklist[pair] = self._config.cooldown_cycles
            logger.info(
                f"PairBlacklist: {pair} BLACKLISTED for "
                f"{self._config.cooldown_cycles} cycles. "
                f"Reasons: {', '.join(reasons)}"
            )

    def check_pair(self, pair: str) -> BlacklistCheckResult:
        """Check if a pair is currently blacklisted."""
        if not self._config.enabled:
            return BlacklistCheckResult(
                is_blacklisted=False,
                pair=pair,
                reason="pair_blacklist_disabled",
            )

        if pair in self._blacklist and self._blacklist[pair] > 0:
            return BlacklistCheckResult(
                is_blacklisted=True,
                pair=pair,
                reason=f"blacklisted ({self._blacklist[pair]} cycles remaining)",
                metrics=self._metrics.get(pair),
                cooldown_remaining=self._blacklist[pair],
            )

        return BlacklistCheckResult(
            is_blacklisted=False,
            pair=pair,
            reason="pair_clear",
            metrics=self._metrics.get(pair),
        )

    def tick_cooldown(self) -> List[str]:
        """Decrement cooldown for all blacklisted pairs. Call once per scan cycle.

        Returns list of pairs removed from blacklist.
        """
        removed = []
        for pair in list(self._blacklist.keys()):
            self._blacklist[pair] -= 1
            if self._blacklist[pair] <= 0:
                del self._blacklist[pair]
                removed.append(pair)
                logger.info(f"PairBlacklist: {pair} cooldown expired, re-enabled")
        return removed

    def get_blacklisted_pairs(self) -> List[str]:
        """Get list of currently blacklisted pairs."""
        return [p for p, c in self._blacklist.items() if c > 0]

    def get_pair_summary(self) -> Dict[str, Dict]:
        """Get summary of all tracked pairs."""
        summary = {}
        for pair, m in self._metrics.items():
            summary[pair] = {
                "trades": m.trade_count,
                "win_rate": round(m.win_rate, 3),
                "profit_factor": round(m.profit_factor, 2),
                "sharpe": round(m.sharpe_estimate, 2),
                "consec_losses": m.consecutive_losses,
                "max_dd_pct": round(m.max_drawdown_pct, 1),
                "blacklisted": pair in self._blacklist and self._blacklist[pair] > 0,
            }
        return summary

    def _load_state(self) -> None:
        """Load persisted state."""
        state_path = Path(self._config.state_file)
        if not state_path.exists():
            return

        try:
            with open(state_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return

            # Load metrics
            for pair, md in data.get("metrics", {}).items():
                try:
                    self._metrics[pair] = PairMetrics(
                        pair=pair,
                        wins=md.get("wins", 0),
                        losses=md.get("losses", 0),
                        total_pnl=md.get("total_pnl", 0.0),
                        total_profit=md.get("total_profit", 0.0),
                        total_loss=md.get("total_loss", 0.0),
                        consecutive_losses=md.get("consecutive_losses", 0),
                        max_consecutive_losses=md.get("max_consecutive_losses", 0),
                        peak_pnl=md.get("peak_pnl", 0.0),
                        max_drawdown_pct=md.get("max_drawdown_pct", 0.0),
                        trade_count=md.get("trade_count", 0),
                        last_updated=md.get("last_updated", 0.0),
                    )
                except (TypeError, KeyError) as e:
                    logger.debug(f"Skipping malformed pair metrics: {e}")

            # Load blacklist
            self._blacklist = data.get("blacklist", {})

            logger.info(
                f"Loaded pair tracker: {len(self._metrics)} pairs, "
                f"{len(self._blacklist)} blacklisted"
            )

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load pair tracker state: {e}")

    def save_state(self) -> None:
        """Persist state (atomic write)."""
        state_path = Path(self._config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = {}
        for pair, m in self._metrics.items():
            metrics_data[pair] = {
                "wins": m.wins,
                "losses": m.losses,
                "total_pnl": m.total_pnl,
                "total_profit": m.total_profit,
                "total_loss": m.total_loss,
                "consecutive_losses": m.consecutive_losses,
                "max_consecutive_losses": m.max_consecutive_losses,
                "peak_pnl": m.peak_pnl,
                "max_drawdown_pct": m.max_drawdown_pct,
                "trade_count": m.trade_count,
                "last_updated": m.last_updated,
            }

        data = {
            "version": self._config.version,
            "saved_at": time.time(),
            "metrics": metrics_data,
            "blacklist": self._blacklist,
        }

        tmp_path = str(state_path) + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, str(state_path))
        except OSError as e:
            logger.warning(f"Failed to save pair tracker state: {e}")


def create_default_pair_tracker(
    config: Optional[PairBlacklistConfig] = None,
) -> PairPerformanceTracker:
    """Factory function for PairPerformanceTracker."""
    return PairPerformanceTracker(config=config)
