"""Trade Cluster Analyzer — Detects when trades cluster in similar conditions.

Prevents repeated exposure to the same market setup by detecting patterns:
- If 3+ recent trades share the same (regime, session, direction, confidence_band)
  and they all lost → apply confidence penalty to avoid the 4th
- If 3+ recent trades share the same cluster profile and won → small confidence boost

Lightweight: runs in memory during scan, not persisted.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Session windows (UTC hours)
SESSION_WINDOWS = {
    "ASIA": (0, 8),
    "LONDON": (8, 16),
    "NY": (13, 21),
    "LATE_NY": (21, 24),
}

# Confidence bands
CONFIDENCE_BANDS = [
    ("LOW", 0.0, 0.55),
    ("MED", 0.55, 0.65),
    ("HIGH", 0.65, 0.80),
    ("VERY_HIGH", 0.80, 1.01),
]


def _get_session(hour_utc: int) -> str:
    """Map UTC hour to trading session."""
    for name, (start, end) in SESSION_WINDOWS.items():
        if start <= hour_utc < end:
            return name
    return "LATE_NY"


def _get_confidence_band(confidence: float) -> str:
    """Map confidence to a band."""
    for name, low, high in CONFIDENCE_BANDS:
        if low <= confidence < high:
            return name
    return "LOW"


class TradeClusterAnalyzer:
    """Detects trade clustering and computes confidence adjustments.

    Usage:
        analyzer = TradeClusterAnalyzer()
        analyzer.add_trade(trade_entry)  # from trade journal
        penalty = analyzer.get_cluster_penalty(pair, direction, regime, confidence)
    """

    def __init__(
        self,
        window_hours: int = 24,
        min_cluster_size: int = 3,
        loss_penalty: float = -0.10,
        win_boost: float = 0.05,
    ):
        self.window_hours = window_hours
        self.min_cluster_size = min_cluster_size
        self.loss_penalty = loss_penalty
        self.win_boost = win_boost
        # cluster_key -> list of (timestamp, won)
        self._clusters: Dict[str, List[Tuple[datetime, bool]]] = defaultdict(list)

    def add_trade(self, trade_entry: dict) -> None:
        """Record a trade for clustering analysis.

        Args:
            trade_entry: Trade journal entry with pair, direction, regime, confidence, outcome
        """
        try:
            pair = trade_entry.get("pair", "UNKNOWN")
            direction = str(trade_entry.get("direction", "HOLD")).upper()
            regime = str(trade_entry.get("volatility_regime", "NORMAL")).upper()
            confidence = float(trade_entry.get("confidence", 0.5))

            outcome = trade_entry.get("outcome", {})
            won = float(outcome.get("realized_pl", 0)) > 0

            # Determine session from timestamp
            ts_str = trade_entry.get("timestamp") or trade_entry.get("entry_time", "")
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)

            session = _get_session(ts.hour)
            conf_band = _get_confidence_band(confidence)

            # Create cluster key: (pair, direction, regime, session, conf_band)
            cluster_key = f"{pair}|{direction}|{regime}|{session}|{conf_band}"

            self._clusters[cluster_key].append((ts, won))

            # Prune old entries
            self._prune()

        except Exception as e:
            logger.debug(f"Trade cluster add failed: {e}")

    def get_cluster_penalty(
        self,
        pair: str,
        direction: str,
        regime: str,
        confidence: float,
    ) -> float:
        """Compute confidence penalty/boost based on cluster analysis.

        Args:
            pair: Currency pair
            direction: "LONG" or "SHORT"
            regime: Volatility regime name
            confidence: Current confidence score

        Returns:
            Float adjustment: negative = penalty, positive = boost, 0 = neutral
        """
        session = _get_session(datetime.now(timezone.utc).hour)
        conf_band = _get_confidence_band(confidence)
        cluster_key = f"{pair}|{direction}|{regime}|{session}|{conf_band}"

        entries = self._clusters.get(cluster_key, [])
        if len(entries) < self.min_cluster_size:
            return 0.0

        # Count wins and losses in cluster
        wins = sum(1 for _, won in entries if won)
        losses = sum(1 for _, won in entries if not won)

        if losses >= self.min_cluster_size:
            logger.info(
                f"Losing cluster detected: {cluster_key} "
                f"({losses} losses in {self.window_hours}h)"
            )
            return self.loss_penalty

        if wins >= self.min_cluster_size:
            return self.win_boost

        return 0.0

    def get_active_clusters(self) -> List[Dict]:
        """Get all active clusters with 2+ trades for reporting."""
        self._prune()
        clusters = []
        for key, entries in self._clusters.items():
            if len(entries) >= 2:
                parts = key.split("|")
                wins = sum(1 for _, won in entries if won)
                losses = len(entries) - wins
                clusters.append({
                    "key": key,
                    "pair": parts[0] if len(parts) > 0 else "",
                    "direction": parts[1] if len(parts) > 1 else "",
                    "regime": parts[2] if len(parts) > 2 else "",
                    "session": parts[3] if len(parts) > 3 else "",
                    "conf_band": parts[4] if len(parts) > 4 else "",
                    "count": len(entries),
                    "wins": wins,
                    "losses": losses,
                })
        return sorted(clusters, key=lambda c: c["count"], reverse=True)

    def _prune(self) -> None:
        """Remove entries older than window_hours."""
        now = datetime.now(timezone.utc)
        for key in list(self._clusters.keys()):
            self._clusters[key] = [
                (ts, won)
                for ts, won in self._clusters[key]
                if (now - ts.replace(tzinfo=timezone.utc if ts.tzinfo is None else ts.tzinfo)).total_seconds()
                < self.window_hours * 3600
            ]
            if not self._clusters[key]:
                del self._clusters[key]

    def load_from_journal(self, journal_path: str = "trained_data/trade_journal_rl.json") -> int:
        """Load recent trades from journal for cluster analysis.

        Returns:
            Number of trades loaded
        """
        try:
            import json
            from pathlib import Path

            path = Path(journal_path)
            if not path.exists():
                return 0

            entries = json.loads(path.read_text())
            count = 0
            for entry in entries:
                if entry.get("outcome") is not None:
                    self.add_trade(entry)
                    count += 1
            return count
        except Exception as e:
            logger.debug(f"Journal load for clustering failed: {e}")
            return 0
