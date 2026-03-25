"""Pair affinity matrix — tracks which pairs win/lose together.

US-062: Build historical co-occurrence matrix for portfolio optimization.
Pairs that consistently co-win have positive affinity; those that co-lose
have negative affinity. Used to recommend optimal pair combinations.

Usage:
    tracker = PairAffinityTracker()
    tracker.update_from_journal()
    score = tracker.get_affinity_score("EUR_USD", "GBP_USD")  # 0.73
    recommendations = tracker.get_best_companions("EUR_USD", top_n=3)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
JOURNAL_PATH = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
AFFINITY_PATH = _PROJECT_ROOT / "trained_data" / "models" / "pair_affinity_matrix.json"


class PairAffinityTracker:
    """Tracks pair co-occurrence outcomes for portfolio optimization.

    Builds a symmetric affinity matrix from overlapping trade outcomes:
    - Two trades are "co-occurring" if their open periods overlap
    - Affinity = (co_wins - co_losses) / total_co_occurrences
    - Range: -1.0 (always opposite outcomes) to +1.0 (always same outcome)
    """

    def __init__(
        self,
        journal_path: Optional[Path] = None,
        affinity_path: Optional[Path] = None,
    ):
        self.journal_path = journal_path or JOURNAL_PATH
        self.affinity_path = affinity_path or AFFINITY_PATH
        self._matrix: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._load()

    def _load(self) -> None:
        """Load persisted affinity data."""
        if not self.affinity_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.affinity_path, default={})
            except ImportError:
                data = json.loads(self.affinity_path.read_text(encoding="utf-8"))

            self._matrix = data.get("matrix", {})
        except Exception as e:
            logger.debug(f"Failed to load affinity matrix: {e}")

    def _save(self) -> None:
        """Persist affinity matrix."""
        data = {
            "matrix": self._matrix,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.affinity_path, data)
            except ImportError:
                self.affinity_path.parent.mkdir(parents=True, exist_ok=True)
                self.affinity_path.write_text(
                    json.dumps(data, indent=2), encoding="utf-8"
                )
        except Exception as e:
            logger.debug(f"Failed to save affinity matrix: {e}")

    def update_from_journal(self) -> int:
        """Rebuild affinity matrix from trade journal.

        Finds overlapping trades and records co-occurrence outcomes.

        Returns:
            Number of co-occurrence pairs processed.
        """
        trades = self._load_journal()
        if len(trades) < 2:
            return 0

        # Filter to trades with outcomes and timestamps
        completed = []
        for t in trades:
            outcome = t.get("outcome")
            if not isinstance(outcome, dict):
                continue
            if outcome.get("trade_won") is None:
                continue
            ts = t.get("timestamp", "")
            close_time = outcome.get("close_time", "")
            if not ts or not close_time:
                continue
            completed.append({
                "pair": t.get("pair", ""),
                "won": outcome["trade_won"],
                "open": ts,
                "close": close_time,
            })

        # Find overlapping trades and record co-occurrences
        co_count = 0
        self._matrix = {}

        for i in range(len(completed)):
            for j in range(i + 1, len(completed)):
                a, b = completed[i], completed[j]
                if a["pair"] == b["pair"]:
                    continue  # Skip same-pair comparisons

                # Check time overlap
                if a["open"] <= b["close"] and b["open"] <= a["close"]:
                    pair_a, pair_b = sorted([a["pair"], b["pair"]])
                    key = f"{pair_a}|{pair_b}"

                    if key not in self._matrix:
                        self._matrix[key] = {"co_win": 0, "co_loss": 0, "mixed": 0}

                    if a["won"] and b["won"]:
                        self._matrix[key]["co_win"] += 1
                    elif not a["won"] and not b["won"]:
                        self._matrix[key]["co_loss"] += 1
                    else:
                        self._matrix[key]["mixed"] += 1

                    co_count += 1

        if co_count > 0:
            self._save()
            logger.info(f"Pair affinity: processed {co_count} co-occurrence pairs")

        return co_count

    def get_affinity_score(self, pair_a: str, pair_b: str) -> float:
        """Get affinity score between two pairs.

        Returns:
            Float -1.0 to 1.0. Positive = pairs tend to co-win.
            Returns 0.0 if no data.
        """
        key = "|".join(sorted([pair_a, pair_b]))
        data = self._matrix.get(key, {})
        co_win = data.get("co_win", 0)
        co_loss = data.get("co_loss", 0)
        mixed = data.get("mixed", 0)
        total = co_win + co_loss + mixed

        if total < 3:
            return 0.0  # Not enough data

        # Affinity: co-wins are positive, co-losses are negative, mixed is neutral
        return round((co_win - co_loss) / total, 4)

    def get_best_companions(
        self, pair: str, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get the best companion pairs for portfolio diversification.

        Returns pairs with highest POSITIVE affinity (co-win tendency).
        """
        scores: List[Tuple[str, float]] = []
        for key, data in self._matrix.items():
            parts = key.split("|")
            if len(parts) != 2:
                continue
            if pair not in parts:
                continue

            other = parts[0] if parts[1] == pair else parts[1]
            score = self.get_affinity_score(pair, other)
            if score != 0.0:
                scores.append((other, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        return {
            "pair_combinations": len(self._matrix),
            "has_data": bool(self._matrix),
        }

    def _load_journal(self) -> List[Dict[str, Any]]:
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
