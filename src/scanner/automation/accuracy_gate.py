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
        data_path: str = "trained_data/pair_accuracy.json",
    ):
        """Initialize AccuracyGate.

        Args:
            min_accuracy: Minimum directional accuracy to remain unblocked.
                         0.55 = better than 50% random baseline.
            min_trades: Minimum trades required before blocking a pair.
                       Prevents blocking after single bad trade.
            data_path: Path to JSON file for persistence.
        """
        self.min_accuracy = min_accuracy
        self.min_trades = min_trades
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

            # Append trade record
            trade_record = {
                "direction": str(predicted_direction),
                "outcome": bool(actual_outcome),
                "confidence": float(min(1.0, max(0.0, confidence))),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self._data[pair]["trades"].append(trade_record)

            # Recalculate accuracy
            self._recalculate_accuracy(pair)

            # Persist
            self._save_data()

            logger.debug(
                f"{pair}: recorded outcome={actual_outcome}, "
                f"total={self._data[pair]['total']}, "
                f"accuracy={self._data[pair]['accuracy']:.1%}"
            )

        except Exception as e:
            logger.error(f"Error recording outcome for {pair}: {e}")

    def _recalculate_accuracy(self, pair: str) -> None:
        """Recalculate directional accuracy for a pair.

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
            return

        total = len(trades)
        wins = sum(1 for t in trades if t.get("outcome", False))

        self._data[pair]["total"] = total
        self._data[pair]["wins"] = wins
        self._data[pair]["accuracy"] = wins / total if total > 0 else None

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

        # Insufficient data — allow
        if total < self.min_trades:
            return (
                True,
                accuracy,
                f"insufficient data ({total}/{self.min_trades} trades)",
            )

        # Above threshold — allow
        if accuracy is not None and accuracy >= self.min_accuracy:
            return (True, accuracy, f"accuracy {accuracy:.1%} >= {self.min_accuracy:.1%}")

        # Below threshold — block
        if accuracy is not None:
            return (
                False,
                accuracy,
                f"accuracy {accuracy:.1%} < {self.min_accuracy:.1%} after {total} trades",
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

        Returns:
            Multi-line string with pair accuracy summary
        """
        if not self._data:
            return "No accuracy data recorded yet."

        lines: List[str] = [
            f"AccuracyGate Report (min_accuracy={self.min_accuracy:.1%}, min_trades={self.min_trades})",
            "=" * 80,
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
            accuracy = data.get("accuracy")

            is_allowed, _, reason = self.check_pair(pair)
            status = "✓ ALLOWED" if is_allowed else "✗ BLOCKED"

            if accuracy is None:
                acc_str = "N/A"
            else:
                acc_str = f"{accuracy:.1%}"

            lines.append(
                f"{pair:12} {status:12} {acc_str:>8} ({wins:2}/{total:2}) — {reason}"
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
        return {
            "pair": pair,
            "total_trades": data.get("total", 0),
            "wins": data.get("wins", 0),
            "accuracy": data.get("accuracy"),
            "recent_trades": data.get("trades", [])[-10:],  # Last 10 trades
        }
