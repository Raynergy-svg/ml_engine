"""Session improvement metrics tracking.

Tracks win rates, learning velocity, and improvement trends over time.
Implementation target: PRD US-007.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ImprovementTracker:
    """Records and tracks improvement metrics across sessions."""

    LOG_FILE = "trained_data/improvement_log.jsonl"

    def __init__(self, project_root: Optional[str] = None):
        self._root = Path(project_root) if project_root else Path.cwd()
        self._log_path = self._root / self.LOG_FILE
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_session(
        self,
        session_id: str = None,
        scan_count: int = 0,
        trade_count: int = 0,
        wins: int = 0,
        losses: int = 0,
        total_pnl: float = 0.0,
        learnings_extracted: int = 0,
        rules_promoted: int = 0,
        config_adjustments: int = 0,
        duration_minutes: float = 0.0,
        **extra,
    ) -> None:
        """Record a session's improvement metrics to the JSONL log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "scan_count": scan_count,
            "trade_count": trade_count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(trade_count, 1), 4),
            "total_pnl": round(total_pnl, 2),
            "learnings_extracted": learnings_extracted,
            "rules_promoted": rules_promoted,
            "config_adjustments": config_adjustments,
            "duration_minutes": round(duration_minutes, 1),
        }
        entry.update(extra)

        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load all JSONL entries."""
        if not self._log_path.exists():
            return []
        entries = []
        for line in self._log_path.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def get_trend(self, window: int = 10) -> dict:
        """Compute improvement trend over the last N sessions.

        Returns:
            Dict with trend metrics: win_rate_trend, pnl_trend,
            learning_velocity, avg_trades_per_session.
        """
        entries = self._load_entries()
        if not entries:
            return {
                "sessions_analyzed": 0,
                "win_rate_trend": "insufficient_data",
                "pnl_trend": "insufficient_data",
                "learning_velocity": 0,
            }

        recent = entries[-window:]
        older = entries[:-window] if len(entries) > window else []

        recent_wr = (
            sum(e.get("win_rate", 0) for e in recent) / len(recent)
            if recent
            else 0
        )
        older_wr = (
            sum(e.get("win_rate", 0) for e in older) / len(older)
            if older
            else recent_wr
        )

        recent_pnl = sum(e.get("total_pnl", 0) for e in recent)
        older_pnl = sum(e.get("total_pnl", 0) for e in older) if older else 0

        total_learnings = sum(e.get("learnings_extracted", 0) for e in recent)
        total_days = max(len(recent), 1)

        wr_delta = recent_wr - older_wr
        if wr_delta > 0.02:
            wr_trend = "improving"
        elif wr_delta < -0.02:
            wr_trend = "declining"
        else:
            wr_trend = "stable"

        pnl_delta = recent_pnl - older_pnl
        if pnl_delta > 50:
            pnl_trend = "improving"
        elif pnl_delta < -50:
            pnl_trend = "declining"
        else:
            pnl_trend = "stable"

        return {
            "sessions_analyzed": len(entries),
            "window": min(window, len(entries)),
            "recent_win_rate": round(recent_wr, 4),
            "win_rate_trend": wr_trend,
            "win_rate_delta": round(wr_delta, 4),
            "recent_pnl": round(recent_pnl, 2),
            "pnl_trend": pnl_trend,
            "learning_velocity": round(total_learnings / total_days, 1),
            "avg_trades_per_session": round(
                sum(e.get("trade_count", 0) for e in recent) / len(recent), 1
            ),
            "total_rules_promoted": sum(
                e.get("rules_promoted", 0) for e in entries
            ),
            "total_config_adjustments": sum(
                e.get("config_adjustments", 0) for e in entries
            ),
        }

    def generate_report(self) -> str:
        """Generate a human-readable improvement report."""
        trend = self.get_trend()
        entries = self._load_entries()

        if not entries:
            return "No session data recorded yet."

        lines = [
            "# Improvement Report",
            f"",
            f"**Sessions tracked:** {trend['sessions_analyzed']}",
            f"**Recent win rate:** {trend['recent_win_rate']:.1%} ({trend['win_rate_trend']})",
            f"**Recent P/L:** ${trend['recent_pnl']:.2f} ({trend['pnl_trend']})",
            f"**Learning velocity:** {trend['learning_velocity']:.1f} learnings/session",
            f"**Avg trades/session:** {trend['avg_trades_per_session']:.1f}",
            f"**Total rules promoted:** {trend['total_rules_promoted']}",
            f"**Total config adjustments:** {trend['total_config_adjustments']}",
        ]

        # Last 5 sessions summary
        recent = entries[-5:]
        if recent:
            lines.append("")
            lines.append("## Recent Sessions")
            for e in reversed(recent):
                ts = e.get("timestamp", "?")[:10]
                wr = e.get("win_rate", 0)
                pnl = e.get("total_pnl", 0)
                tc = e.get("trade_count", 0)
                lines.append(f"- {ts}: {tc} trades, {wr:.0%} WR, ${pnl:.2f} P/L")

        return "\n".join(lines)
