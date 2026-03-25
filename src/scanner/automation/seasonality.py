"""Hour-of-day seasonality tracker for per-pair confidence adjustment.

US-064: Extract per-pair hour-of-day win rates from journal. Apply
confidence multiplier to boost/penalize based on historically profitable
or unprofitable trading hours.

Usage:
    tracker = SeasonalityTracker()
    tracker.update_from_journal()
    modifier = tracker.get_hour_modifier("EUR_USD", hour=10)  # 1.15
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
JOURNAL_PATH = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
SEASONALITY_PATH = _PROJECT_ROOT / "trained_data" / "models" / "hour_seasonality.json"

# Modifier bounds
MIN_MODIFIER = 0.70   # Maximum penalty: 30% confidence reduction
MAX_MODIFIER = 1.30   # Maximum boost: 30% confidence increase
MIN_TRADES_PER_HOUR = 5  # Need at least 5 trades in a given hour to compute modifier


class SeasonalityTracker:
    """Tracks per-pair hour-of-day trading performance.

    Builds a matrix of {pair: {hour: {wins, losses, total, win_rate}}}
    from trade journal entries. Provides confidence modifiers.
    """

    def __init__(
        self,
        journal_path: Optional[Path] = None,
        seasonality_path: Optional[Path] = None,
    ):
        self.journal_path = journal_path or JOURNAL_PATH
        self.seasonality_path = seasonality_path or SEASONALITY_PATH
        # {pair: {hour: {wins: int, losses: int}}}
        self._data: Dict[str, Dict[int, Dict[str, int]]] = {}
        self._load()

    def _load(self) -> None:
        """Load persisted seasonality data."""
        if not self.seasonality_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                raw = safe_json_read(self.seasonality_path, default={})
            except ImportError:
                raw = json.loads(self.seasonality_path.read_text(encoding="utf-8"))

            # Convert string hour keys back to int
            for pair, hours in raw.get("data", {}).items():
                self._data[pair] = {
                    int(h): v for h, v in hours.items()
                }
        except Exception as e:
            logger.debug(f"Failed to load seasonality data: {e}")

    def _save(self) -> None:
        """Persist seasonality data."""
        data = {
            "data": self._data,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.seasonality_path, data)
            except ImportError:
                self.seasonality_path.parent.mkdir(parents=True, exist_ok=True)
                self.seasonality_path.write_text(
                    json.dumps(data, indent=2, default=str), encoding="utf-8"
                )
        except Exception as e:
            logger.debug(f"Failed to save seasonality data: {e}")

    def update_from_journal(self) -> int:
        """Rebuild seasonality data from trade journal.

        Returns:
            Number of trades processed.
        """
        trades = self._load_journal()
        if not trades:
            return 0

        self._data = {}
        processed = 0

        for trade in trades:
            outcome = trade.get("outcome")
            if not isinstance(outcome, dict):
                continue
            if outcome.get("trade_won") is None:
                continue

            pair = trade.get("pair", "")
            ts = trade.get("timestamp", "")
            if not pair or not ts:
                continue

            # Extract hour from entry timestamp
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                hour = dt.hour
            except Exception:
                continue

            if pair not in self._data:
                self._data[pair] = {}
            if hour not in self._data[pair]:
                self._data[pair][hour] = {"wins": 0, "losses": 0}

            if outcome["trade_won"]:
                self._data[pair][hour]["wins"] += 1
            else:
                self._data[pair][hour]["losses"] += 1

            processed += 1

        if processed > 0:
            self._save()
            logger.info(f"Seasonality: processed {processed} trades across {len(self._data)} pairs")

        return processed

    def get_hour_modifier(self, pair: str, hour: int) -> float:
        """Get confidence modifier for a pair at a specific hour.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            hour: Hour of day in UTC (0-23)

        Returns:
            Float 0.70-1.30. >1.0 = historically profitable hour,
            <1.0 = historically unprofitable hour. 1.0 = neutral/no data.
        """
        pair_data = self._data.get(pair, {})
        hour_data = pair_data.get(hour)

        if not hour_data:
            return 1.0

        wins = hour_data.get("wins", 0)
        losses = hour_data.get("losses", 0)
        total = wins + losses

        if total < MIN_TRADES_PER_HOUR:
            return 1.0  # Not enough data

        win_rate = wins / total

        # Compute modifier: center around 0.50 win rate
        # win_rate 0.70 → modifier 1.20
        # win_rate 0.30 → modifier 0.80
        modifier = 1.0 + (win_rate - 0.50) * 1.0

        return round(max(MIN_MODIFIER, min(MAX_MODIFIER, modifier)), 4)

    def get_pair_seasonality(self, pair: str) -> Dict[int, Dict[str, Any]]:
        """Get full seasonality profile for a pair.

        Returns:
            Dict mapping hour → {wins, losses, total, win_rate, modifier}
        """
        pair_data = self._data.get(pair, {})
        result: Dict[int, Dict[str, Any]] = {}

        for hour in range(24):
            hour_data = pair_data.get(hour, {"wins": 0, "losses": 0})
            wins = hour_data.get("wins", 0)
            losses = hour_data.get("losses", 0)
            total = wins + losses
            win_rate = wins / total if total > 0 else 0.0

            result[hour] = {
                "wins": wins,
                "losses": losses,
                "total": total,
                "win_rate": round(win_rate, 4),
                "modifier": self.get_hour_modifier(pair, hour),
            }

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        total_trades = sum(
            h.get("wins", 0) + h.get("losses", 0)
            for pair_data in self._data.values()
            for h in pair_data.values()
        )
        return {
            "pairs_tracked": len(self._data),
            "total_trade_entries": total_trades,
            "has_data": bool(self._data),
        }

    def _load_journal(self):
        """Load trade journal."""
        if not self.journal_path.exists():
            return []
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                return safe_json_read(self.journal_path, default=[])
            except ImportError:
                return json.loads(self.journal_path.read_text(encoding="utf-8"))
        except Exception:
            return []
