"""
Dynamic Pair Rotation — Rolling Sharpe Ratio based pair ranking and rotation.

Prioritizes pairs with best recent risk-adjusted performance. Uses rolling Sharpe
ratio to rank pairs and rotates underperformers to observe-only mode (scan but don't trade).

Features:
- Rolling Sharpe ratio calculation (last N trades per pair)
- Dynamic ranking with edge cases (new pairs, insufficient data)
- USD exposure clustering detection
- ObservationLog integration for rotation decisions
- File locking (fcntl) for safe concurrent access
- Graceful degradation on corrupted/missing trade journal
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PairRanking:
    """Ranking result for a single pair."""

    pair: str
    sharpe_ratio: float  # Rolling Sharpe ratio (NaN if insufficient data)
    rolling_pnl: float  # Sum of last N trades P/L
    trade_count: int  # Total trades for this pair
    win_rate: float  # Wins / total trades
    status: str  # "active" or "observe"
    reason: str  # Why this status

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "pair": self.pair,
            "sharpe_ratio": self.sharpe_ratio if not math.isnan(self.sharpe_ratio) else None,
            "rolling_pnl": round(self.rolling_pnl, 2),
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "status": self.status,
            "reason": self.reason,
        }


class PortfolioOptimizer:
    """Ranks pairs by rolling Sharpe and rotates underperformers."""

    # USD-based pairs (all major pairs involve USD)
    USD_PAIRS = {"EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD", "USD_CAD", "USD_CHF"}

    def __init__(
        self,
        trade_journal_path: str = "trained_data/trade_journal_rl.json",
        pair_rankings_path: str = "trained_data/models/pair_rankings.json",
        rolling_window: int = 50,  # Last N trades per pair
        max_active_pairs: int = 7,  # Top N by Sharpe
        max_usd_exposure: float = 0.40,  # Max share of active pairs that are USD
        rotation_interval: int = 10,  # Check every N scan cycles
    ):
        """Initialize portfolio optimizer.

        Args:
            trade_journal_path: Path to trade journal JSON file
            pair_rankings_path: Path to save rankings
            rolling_window: Number of recent trades to analyze per pair
            max_active_pairs: Maximum number of pairs in active trading
            max_usd_exposure: Maximum fraction of active pairs that can be USD-based
            rotation_interval: Number of scan cycles between rotation checks
        """
        self.trade_journal_path = Path(trade_journal_path)
        self.pair_rankings_path = Path(pair_rankings_path)
        self.rolling_window = rolling_window
        self.max_active_pairs = max_active_pairs
        self.max_usd_exposure = max_usd_exposure
        self.rotation_interval = rotation_interval

    def calculate_sharpe(
        self, returns: list[float], risk_free_rate: float = 0.0, annualization_factor: float = 252.0
    ) -> float:
        """Calculate Sharpe ratio from list of returns (P/L values).

        Args:
            returns: List of trade P/L values (in pips or dollars)
            risk_free_rate: Risk-free rate (default 0%)
            annualization_factor: For daily returns, 252; hourly=252*24; etc.

        Returns:
            Sharpe ratio (0.0 if insufficient data or zero std dev)
        """
        if not returns or len(returns) < 2:
            return 0.0

        try:
            # Calculate mean return
            mean_return = sum(returns) / len(returns)

            # Calculate standard deviation
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = math.sqrt(variance)

            # Handle zero volatility
            if std_dev == 0.0:
                return 0.0

            # Sharpe ratio = (mean - rf) / std * sqrt(periods_per_year)
            sharpe = (mean_return - risk_free_rate) / std_dev * math.sqrt(annualization_factor)
            return sharpe if not math.isnan(sharpe) else 0.0

        except Exception as e:
            logger.debug(f"Sharpe calculation error: {e}")
            return 0.0

    def rank_pairs(self, trade_journal: Optional[list[dict]] = None) -> list[PairRanking]:
        """Rank all pairs by rolling Sharpe ratio.

        Args:
            trade_journal: List of trade records (loads from file if None)

        Returns:
            List of PairRanking objects, sorted by Sharpe descending
        """
        if trade_journal is None:
            trade_journal = self._load_trade_journal()

        if not trade_journal:
            logger.debug("No trade journal data; returning empty rankings")
            return []

        # Group trades by pair
        pair_trades: Dict[str, list[dict]] = {}
        for trade in trade_journal:
            pair = trade.get("pair")
            if not pair:
                continue
            if pair not in pair_trades:
                pair_trades[pair] = []
            pair_trades[pair].append(trade)

        rankings: list[PairRanking] = []

        for pair, trades in sorted(pair_trades.items()):
            # Filter to closed trades only
            closed_trades = [t for t in trades if t.get("outcome") is not None]

            if not closed_trades:
                # No closed trades yet for this pair
                rankings.append(
                    PairRanking(
                        pair=pair,
                        sharpe_ratio=0.0,
                        rolling_pnl=0.0,
                        trade_count=0,
                        win_rate=0.0,
                        status="observe",
                        reason="No closed trades yet",
                    )
                )
                continue

            # Calculate metrics on rolling window (last N trades)
            rolling = closed_trades[-self.rolling_window :]
            returns = []
            wins = 0

            for trade in rolling:
                outcome = trade.get("outcome", {})
                pnl = outcome.get("realized_pl", 0.0)
                returns.append(pnl)
                if outcome.get("trade_won"):
                    wins += 1

            rolling_pnl = sum(returns)
            win_rate = wins / len(rolling) if rolling else 0.0
            sharpe = self.calculate_sharpe(returns)

            # Status logic: need at least 5 closed trades to be considered "active"
            if len(closed_trades) < 5:
                status = "observe"
                reason = f"Insufficient history ({len(closed_trades)}/5 min)"
            else:
                status = "active"  # Tentative; will filter by top N later
                reason = f"Sharpe {sharpe:.2f}"

            rankings.append(
                PairRanking(
                    pair=pair,
                    sharpe_ratio=sharpe,
                    rolling_pnl=rolling_pnl,
                    trade_count=len(closed_trades),
                    win_rate=win_rate,
                    status=status,
                    reason=reason,
                )
            )

        # Sort by Sharpe ratio descending
        rankings.sort(key=lambda r: r.sharpe_ratio, reverse=True)

        return rankings

    def get_active_pairs(self, trade_journal: Optional[list[dict]] = None) -> list[str]:
        """Return list of pairs that should be actively traded.

        Algorithm:
        1. Rank all pairs by Sharpe ratio
        2. Select top max_active_pairs with sufficient trade history
        3. Check USD exposure: if > max_usd_exposure, demote lowest-Sharpe USD pair to observe
        4. Return list of active pair names

        Args:
            trade_journal: List of trade records (loads from file if None)

        Returns:
            List of pair names to actively trade
        """
        rankings = self.rank_pairs(trade_journal)

        if not rankings:
            logger.debug("No pair rankings available; returning empty active list")
            return []

        # Filter to pairs with sufficient trade history (status="active")
        active_candidates = [r for r in rankings if r.status == "active"]

        # If not enough active pairs, pad with observe-only pairs (up to top N)
        if len(active_candidates) < self.max_active_pairs:
            observe_candidates = [r for r in rankings if r.status != "active"]
            needed = self.max_active_pairs - len(active_candidates)
            active_candidates.extend(observe_candidates[:needed])

        # Limit to max_active_pairs
        active_candidates = active_candidates[: self.max_active_pairs]

        # Check USD exposure
        usd_active = [r for r in active_candidates if r.pair in self.USD_PAIRS]
        usd_fraction = len(usd_active) / len(active_candidates) if active_candidates else 0.0

        if usd_fraction > self.max_usd_exposure:
            # Find lowest-Sharpe USD pair and demote it
            usd_active.sort(key=lambda r: r.sharpe_ratio)
            if usd_active:
                demoted = usd_active[0]
                active_candidates = [r for r in active_candidates if r.pair != demoted.pair]
                logger.info(
                    f"USD exposure check: demoting {demoted.pair} "
                    f"(was {usd_fraction:.0%} of active pairs, limit {self.max_usd_exposure:.0%})"
                )

        return [r.pair for r in active_candidates]

    def get_observe_only_pairs(
        self, trade_journal: Optional[list[dict]] = None, all_pairs: Optional[list[str]] = None
    ) -> list[str]:
        """Return pairs that should be in observe-only mode (scan but don't trade).

        Args:
            trade_journal: List of trade records (loads from file if None)
            all_pairs: Full list of pairs to consider (uses DEFAULT_PAIRS if None)

        Returns:
            List of pair names in observe-only mode
        """
        active = set(self.get_active_pairs(trade_journal))

        if all_pairs is None:
            # Default to all pairs defined in config
            try:
                from src.scanner.config import DEFAULT_PAIRS

                all_pairs = DEFAULT_PAIRS
            except ImportError:
                logger.debug("Could not import DEFAULT_PAIRS")
                all_pairs = []

        observe = [p for p in all_pairs if p not in active]
        return observe

    def should_rotate(self, scan_cycle: int) -> bool:
        """Check if it's time to re-evaluate pair rankings.

        Args:
            scan_cycle: Current scan cycle number

        Returns:
            True if rotation check is due
        """
        return scan_cycle > 0 and scan_cycle % self.rotation_interval == 0

    def save_rankings(self, rankings: list[PairRanking]) -> None:
        """Persist current rankings to JSON file with file locking.

        Args:
            rankings: List of PairRanking objects
        """
        try:
            self.pair_rankings_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to JSON
            data = {
                "timestamp": datetime.now().isoformat(),
                "rankings": [r.to_dict() for r in rankings],
            }

            # Write with file locking (fcntl)
            with open(self.pair_rankings_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(data, indent=2))
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.debug(f"Saved {len(rankings)} pair rankings to {self.pair_rankings_path}")

        except Exception as e:
            logger.error(f"Failed to save pair rankings: {e}")

    def load_rankings(self) -> list[PairRanking]:
        """Load saved rankings from JSON file.

        Returns:
            List of PairRanking objects (empty if file doesn't exist)
        """
        if not self.pair_rankings_path.exists():
            return []

        try:
            with open(self.pair_rankings_path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            rankings_data = data.get("rankings", [])
            rankings = []

            for r in rankings_data:
                rankings.append(
                    PairRanking(
                        pair=r.get("pair", ""),
                        sharpe_ratio=r.get("sharpe_ratio", 0.0) or 0.0,
                        rolling_pnl=r.get("rolling_pnl", 0.0),
                        trade_count=r.get("trade_count", 0),
                        win_rate=r.get("win_rate", 0.0),
                        status=r.get("status", "observe"),
                        reason=r.get("reason", ""),
                    )
                )

            logger.debug(f"Loaded {len(rankings)} pair rankings from {self.pair_rankings_path}")
            return rankings

        except Exception as e:
            logger.error(f"Failed to load pair rankings: {e}")
            return []

    def _load_trade_journal(self) -> list[dict]:
        """Load trade journal from JSON file with graceful error handling.

        Returns:
            List of trade records (empty list if file missing or corrupted)
        """
        if not self.trade_journal_path.exists():
            logger.debug(f"Trade journal not found: {self.trade_journal_path}")
            return []

        try:
            with open(self.trade_journal_path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            if not isinstance(data, list):
                logger.warning(f"Trade journal is not a list: {type(data)}")
                return []

            logger.debug(f"Loaded {len(data)} trades from {self.trade_journal_path}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Trade journal JSON parsing error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load trade journal: {e}")
            return []

    def log_rotation_decision(
        self,
        scan_cycle: int,
        active_pairs: list[str],
        observe_pairs: list[str],
    ) -> None:
        """Log pair rotation decision to ObservationLog.

        Args:
            scan_cycle: Current scan cycle number
            active_pairs: Pairs selected for active trading
            observe_pairs: Pairs in observe-only mode
        """
        try:
            from src.scanner.automation.observation_log import ObservationLog

            obs_log = ObservationLog()

            obs_log.log_observation(
                pair="PORTFOLIO",
                category="pair_rotation",
                description=f"Pair rotation at cycle {scan_cycle}: {len(active_pairs)} active, "
                f"{len(observe_pairs)} observe",
                metadata={
                    "scan_cycle": scan_cycle,
                    "active_pairs": active_pairs,
                    "observe_pairs": observe_pairs,
                    "active_count": len(active_pairs),
                    "observe_count": len(observe_pairs),
                },
            )

        except Exception as e:
            logger.debug(f"Observation logging error: {e}")
