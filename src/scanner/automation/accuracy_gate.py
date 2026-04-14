"""Per-pair accuracy gating system for live trading.

Automatically tracks directional prediction accuracy per pair and blocks
pairs whose accuracy falls below a configurable threshold. This prevents
overfitting and stops trading pairs that have deteriorated in live performance.

Features:
- Tracks win/loss outcomes per pair with confidence scores
- Calculates rolling directional accuracy per pair
- Auto-blocks pairs below accuracy threshold
- Maintains minimum trade count before evaluation (avoid blocking after 1 loss)
- Human-readable JSON persistence and reporting
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AccuracyGate:
    """
    Per-pair accuracy tracking and gating system.

    Monitors directional prediction accuracy for each pair and auto-blocks
    pairs that fall below minimum accuracy threshold. Prevents trading pairs
    that have deteriorated during live execution.
    """

    def __init__(
        self,
        min_accuracy: float = 0.55,
        min_trades: int = 5,
        rolling_window: int = 20,
        data_path: str = "trained_data/pair_accuracy.json",
    ):
        """Initialize AccuracyGate.

        Args:
            min_accuracy: Minimum directional accuracy to remain unblocked.
                         0.55 = better than 50% random baseline.
            min_trades: Minimum trades required before blocking a pair.
                       Prevents blocking after single bad trade.
            rolling_window: Number of recent trades to use for accuracy calculation.
                           Uses all-time if total trades < rolling_window.
                           This ensures model retraining can unlock a blocked pair
                           within rolling_window trades rather than being anchored to
                           historical performance forever.
            data_path: Path to JSON file for persistence.
        """
        self.min_accuracy = min_accuracy
        self.min_trades = min_trades
        self.rolling_window = rolling_window
        self.data_path = Path(data_path)
        self._data: Dict[str, Dict] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load pair accuracy data from JSON file."""
        try:
            if self.data_path.exists():
                self._data = json.loads(self.data_path.read_text())
                logger.debug(f"Loaded accuracy data for {len(self._data)} pairs")
            else:
                self._data = {}
        except Exception as e:
            logger.warning(f"Failed to load accuracy data: {e}. Starting fresh.")
            self._data = {}

    def _save_data(self) -> None:
        """Persist pair accuracy data to JSON file."""
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.data_path.write_text(json.dumps(self._data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save accuracy data: {e}")

    def record_outcome(
        self,
        pair: str,
        predicted_direction: str,
        actual_outcome: bool,
        confidence: float = 1.0,
        trade_id: Optional[str] = None,
    ) -> None:
        """Record a trade outcome for accuracy calculation.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            predicted_direction: Direction predicted ("LONG" or "SHORT")
            actual_outcome: True if prediction was correct, False otherwise
            confidence: Confidence score (0.0 to 1.0) of the prediction

        Returns:
            None. Updates internal state and persists to JSON.
        """
        if not pair or not isinstance(pair, str):
            logger.warning(f"Invalid pair: {pair}")
            return

        try:
            # Initialize pair if not present
            if pair not in self._data:
                self._data[pair] = {
                    "trades": [],
                    "accuracy": None,
                    "total": 0,
                    "wins": 0,
                }

            if trade_id is not None:
                trade_id = str(trade_id)
                for existing in self._data[pair]["trades"]:
                    if str(existing.get("trade_id", "")) == trade_id:
                        logger.debug("%s: skipping duplicate accuracy outcome for trade_id=%s", pair, trade_id)
                        return

            # Append trade record
            trade_record = {
                "direction": str(predicted_direction),
                "outcome": bool(actual_outcome),
                "confidence": float(min(1.0, max(0.0, confidence))),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            if trade_id is not None:
                trade_record["trade_id"] = trade_id
            self._data[pair]["trades"].append(trade_record)

            # Prune in-memory list to rolling window (prevent unbounded growth)
            max_keep = self.rolling_window * 2  # keep 2x window for safety
            if len(self._data[pair]["trades"]) > max_keep:
                self._data[pair]["trades"] = self._data[pair]["trades"][-max_keep:]

            # Recalculate accuracy
            self._recalculate_accuracy(pair)

            # Persist
            self._save_data()

            acc = self._data[pair]["accuracy"]
            acc_str = f"{acc:.1%}" if acc is not None else "N/A"
            logger.debug(
                f"{pair}: recorded outcome={actual_outcome}, "
                f"total={self._data[pair]['total']}, "
                f"rolling_accuracy={acc_str}"
            )

        except Exception as e:
            logger.error(f"Error recording outcome for {pair}: {e}")

    def rebuild_from_journal(self, journal_path: str = "trained_data/trade_journal_rl.json") -> int:
        """Rebuild canonical pair accuracy state from closed journal entries."""
        jp = Path(journal_path)
        if not jp.exists():
            return 0
        try:
            raw = json.loads(jp.read_text())
        except Exception as e:
            logger.warning("Failed to rebuild accuracy from journal: %s", e)
            return 0
        if not isinstance(raw, list):
            return 0

        rebuilt: Dict[str, Dict] = {}
        count = 0
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            outcome = entry.get("outcome")
            if not isinstance(outcome, dict):
                continue
            pair = str(entry.get("pair", "") or "")
            if not pair:
                continue
            rebuilt.setdefault(pair, {
                "trades": [],
                "accuracy": None,
                "total": 0,
                "wins": 0,
            })
            rebuilt[pair]["trades"].append({
                "trade_id": str(entry.get("trade_id", "") or ""),
                "direction": str(entry.get("direction", "")),
                "outcome": bool(outcome.get("trade_won", False)),
                "confidence": float(min(1.0, max(0.0, float(entry.get("confidence", 0.0) or 0.0)))),
                "timestamp": str(
                    outcome.get("close_time")
                    or outcome.get("exit_time")
                    or entry.get("timestamp")
                    or ""
                ),
            })
            count += 1

        self._data = rebuilt
        for pair in list(self._data.keys()):
            self._recalculate_accuracy(pair)
        self._save_data()
        logger.info("Rebuilt pair accuracy from %d closed journal trade(s) across %d pair(s)", count, len(self._data))
        return count

    def _recalculate_accuracy(self, pair: str) -> None:
        """Recalculate directional accuracy for a pair using rolling window.

        Uses the most recent rolling_window trades for accuracy calculation,
        so model retraining can unlock blocked pairs within rolling_window trades
        rather than being anchored to historical performance indefinitely.

        Args:
            pair: Currency pair to update
        """
        if pair not in self._data:
            return

        trades = self._data[pair].get("trades", [])
        if not trades:
            self._data[pair]["accuracy"] = None
            self._data[pair]["total"] = 0
            self._data[pair]["wins"] = 0
            self._data[pair]["rolling_total"] = 0
            self._data[pair]["rolling_wins"] = 0
            return

        total = len(trades)
        wins = sum(1 for t in trades if t.get("outcome", False))

        # Rolling window: use last N trades for the accuracy gate decision
        recent_trades = trades[-self.rolling_window :]
        rolling_total = len(recent_trades)
        rolling_wins = sum(1 for t in recent_trades if t.get("outcome", False))

        self._data[pair]["total"] = total
        self._data[pair]["wins"] = wins
        self._data[pair]["rolling_total"] = rolling_total
        self._data[pair]["rolling_wins"] = rolling_wins
        # accuracy reflects rolling window (what the gate actually evaluates)
        self._data[pair]["accuracy"] = (
            rolling_wins / rolling_total if rolling_total > 0 else None
        )

    def check_pair(self, pair: str) -> Tuple[bool, Optional[float], str]:
        """Check if a pair is allowed to trade.

        Args:
            pair: Currency pair to check

        Returns:
            Tuple of (is_allowed, accuracy, reason)
            - is_allowed: True if pair meets criteria to trade
            - accuracy: Calculated accuracy (None if insufficient data)
            - reason: Human-readable explanation
        """
        if pair not in self._data:
            return (True, None, "new pair, no history")

        pair_data = self._data[pair]
        total = pair_data.get("total", 0)
        accuracy = pair_data.get("accuracy")

        rolling_total = pair_data.get("rolling_total", total)

        # Insufficient data — allow (use rolling_total for gate decisions)
        if rolling_total < self.min_trades:
            return (
                True,
                accuracy,
                f"insufficient data ({rolling_total}/{self.min_trades} recent trades)",
            )

        # Above threshold — allow
        if accuracy is not None and accuracy >= self.min_accuracy:
            return (
                True,
                accuracy,
                f"rolling accuracy {accuracy:.1%} >= {self.min_accuracy:.1%} "
                f"({rolling_total} recent trades)",
            )

        # Below threshold — block
        if accuracy is not None:
            return (
                False,
                accuracy,
                f"rolling accuracy {accuracy:.1%} < {self.min_accuracy:.1%} "
                f"after {rolling_total} recent trades (of {total} total)",
            )

        # Should not reach here
        return (True, None, "accuracy calculation failed")

    def get_blocked_pairs(self) -> List[str]:
        """Get list of all pairs currently blocked by accuracy gate.

        Returns:
            List of pair codes that should be blocked
        """
        blocked: List[str] = []
        for pair in self._data.keys():
            is_allowed, _, _ = self.check_pair(pair)
            if not is_allowed:
                blocked.append(pair)

        return blocked

    def get_report(self) -> str:
        """Generate formatted accuracy report for all pairs.

        Shows rolling window accuracy (used for gating) and all-time accuracy
        for reference. Rolling window = last {rolling_window} trades.

        Returns:
            Multi-line string with pair accuracy summary
        """
        if not self._data:
            return "No accuracy data recorded yet."

        lines: List[str] = [
            f"AccuracyGate Report (min_accuracy={self.min_accuracy:.1%}, "
            f"min_trades={self.min_trades}, rolling_window={self.rolling_window})",
            "=" * 90,
            f"{'Pair':12} {'Status':12} {'Rolling Acc':>12} {'All-time':>10} {'Reason'}",
            "-" * 90,
        ]

        # Sort by total trades (descending) for relevance
        sorted_pairs = sorted(
            self._data.items(),
            key=lambda x: x[1].get("total", 0),
            reverse=True,
        )

        for pair, data in sorted_pairs:
            total = data.get("total", 0)
            wins = data.get("wins", 0)
            rolling_total = data.get("rolling_total", total)
            rolling_wins = data.get("rolling_wins", wins)
            accuracy = data.get("accuracy")

            is_allowed, _, reason = self.check_pair(pair)
            status = "✓ ALLOWED" if is_allowed else "✗ BLOCKED"

            if accuracy is None:
                rolling_str = "N/A"
            else:
                rolling_str = f"{accuracy:.1%} ({rolling_wins}/{rolling_total})"

            alltime_str = (
                f"{wins/total:.1%} ({wins}/{total})" if total > 0 else "N/A"
            )

            lines.append(
                f"{pair:12} {status:12} {rolling_str:>12} {alltime_str:>10} — {reason}"
            )

        # Add summary
        blocked = self.get_blocked_pairs()
        lines.extend([
            "=" * 80,
            f"Blocked pairs: {len(blocked)} of {len(self._data)} "
            f"({len(blocked)/len(self._data):.0%})" if self._data else "0 pairs",
        ])

        if blocked:
            lines.append(f"  {', '.join(blocked)}")

        return "\n".join(lines)

    def reset_pair(self, pair: str) -> None:
        """Reset accuracy history for a specific pair.

        Use after config changes or pair retraining to start fresh evaluation.

        Args:
            pair: Currency pair to reset
        """
        if pair in self._data:
            del self._data[pair]
            self._save_data()
            logger.info(f"Reset accuracy history for {pair}")

    def get_pair_stats(self, pair: str) -> Optional[Dict]:
        """Get detailed statistics for a specific pair.

        Args:
            pair: Currency pair to retrieve stats for

        Returns:
            Dict with total, wins, accuracy, and trade history, or None if pair not found
        """
        if pair not in self._data:
            return None

        data = self._data[pair].copy()
        total = data.get("total", 0)
        wins = data.get("wins", 0)
        return {
            "pair": pair,
            "total_trades": total,
            "wins": wins,
            "alltime_accuracy": wins / total if total > 0 else None,
            "rolling_total": data.get("rolling_total", total),
            "rolling_wins": data.get("rolling_wins", wins),
            "rolling_accuracy": data.get("accuracy"),  # gate uses rolling
            "recent_trades": data.get("trades", [])[-10:],  # Last 10 trades
        }
