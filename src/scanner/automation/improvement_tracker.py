"""Improvement tracker — measures whether the learning system itself is improving.

US-007: Add session improvement summary and metrics tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

IMPROVEMENT_LOG_PATH = Path("trained_data/improvement_log.jsonl")


class ImprovementTracker:
    """Records and analyzes session-level improvement metrics."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or IMPROVEMENT_LOG_PATH

    def record_session(
        self,
        trades: List[Dict[str, Any]],
        learnings_added: int = 0,
        rules_promoted: int = 0,
        config_adjustments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Append a session record to improvement_log.jsonl."""
        closed = [t for t in trades if t.get("outcome")]
        wins = sum(1 for t in closed if t["outcome"].get("trade_won", False))
        losses = len(closed) - wins
        net_pnl = sum(t["outcome"].get("realized_pl", 0) for t in closed)
        avg_pips = 0.0
        if closed:
            avg_pips = sum(abs(t["outcome"].get("pnl_pips", 0)) for t in closed) / len(closed)

        # Snapshot agent weights
        weights_snapshot: Dict[str, float] = {}
        try:
            wp = Path("trained_data/models/agent_weights.json")
            if wp.exists():
                weights_snapshot = json.loads(wp.read_text())
        except Exception:
            pass

        record = {
            "date": datetime.now(timezone.utc).isoformat(),
            "session_trades": len(closed),
            "wins": wins,
            "losses": losses,
            "net_pnl": round(net_pnl, 2),
            "win_rate": round(wins / len(closed), 3) if closed else 0.0,
            "avg_pips": round(avg_pips, 1),
            "learnings_added": learnings_added,
            "rules_promoted": rules_promoted,
            "config_adjustments": config_adjustments or [],
            "agent_weights_snapshot": weights_snapshot,
        }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        logger.info("Session recorded: %d trades, %.0f%% win, $%.2f P/L", len(closed), record["win_rate"] * 100, net_pnl)
        return record

    def get_trend(self, window: int = 10) -> Dict[str, str]:
        """Return rolling metrics over last N sessions."""
        records = self._load_records()
        if len(records) < 2:
            return {"win_rate_trend": "insufficient_data", "avg_pnl_trend": "insufficient_data", "learning_velocity": "0"}

        recent = records[-window:]
        older = records[:-window] if len(records) > window else records[:1]

        recent_wr = _avg([r.get("win_rate", 0) for r in recent])
        older_wr = _avg([r.get("win_rate", 0) for r in older])
        recent_pnl = _avg([r.get("net_pnl", 0) for r in recent])
        older_pnl = _avg([r.get("net_pnl", 0) for r in older])
        learning_vel = _avg([r.get("learnings_added", 0) for r in recent])

        def _trend(recent_val: float, older_val: float) -> str:
            diff = recent_val - older_val
            if abs(diff) < 0.02:
                return "stable"
            return "improving" if diff > 0 else "declining"

        return {
            "win_rate_trend": _trend(recent_wr, older_wr),
            "avg_pnl_trend": _trend(recent_pnl, older_pnl),
            "learning_velocity": f"{learning_vel:.1f}",
        }

    def generate_report(self) -> str:
        """Return formatted improvement report string."""
        records = self._load_records()
        if not records:
            return "No improvement data recorded yet."

        total_trades = sum(r.get("session_trades", 0) for r in records)
        total_wins = sum(r.get("wins", 0) for r in records)
        total_pnl = sum(r.get("net_pnl", 0) for r in records)
        overall_wr = total_wins / total_trades if total_trades else 0

        # Best/worst pair (from recent records)
        pair_pnl: Dict[str, float] = {}
        for r in records:
            for adj in r.get("config_adjustments", []):
                pass  # adjustments don't have pair info directly

        trend = self.get_trend()
        total_learnings = sum(r.get("learnings_added", 0) for r in records)
        total_promotions = sum(r.get("rules_promoted", 0) for r in records)

        lines = [
            "=== Improvement Report ===",
            f"Sessions: {len(records)}",
            f"Total trades: {total_trades} ({total_wins}W / {total_trades - total_wins}L)",
            f"Overall win rate: {overall_wr:.0%}",
            f"Total P/L: ${total_pnl:.2f}",
            f"Total learnings extracted: {total_learnings}",
            f"Total rules promoted: {total_promotions}",
            f"Win rate trend: {trend['win_rate_trend']}",
            f"P/L trend: {trend['avg_pnl_trend']}",
            f"Learning velocity: {trend['learning_velocity']}/session",
        ]
        return "\n".join(lines)

    def _load_records(self) -> List[Dict[str, Any]]:
        """Load all records from JSONL file."""
        if not self.log_path.exists():
            return []
        records = []
        for line in self.log_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
