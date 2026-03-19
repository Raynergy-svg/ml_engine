"""Session state engine for cross-session continuity.

Persists portfolio state, goals, and progress to .claude/state.json.
Implementation target: PRD US-002.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StateEngine:
    """Manages cross-session state persistence.

    Reads and writes .claude/state.json to maintain continuity across
    scanner sessions. Tracks portfolio snapshots, scan history, and
    learnings with bounded list sizes to prevent unbounded growth.
    """

    STATE_FILE = ".claude/state.json"

    _MAX_SCAN_HISTORY = 100
    _MAX_RECENT_LEARNINGS = 20

    _DEFAULT_STATE: dict[str, Any] = {
        "goal": "Autonomous scan-trade-learn loop",
        "status": "in-progress",
        "done": [],
        "next": "",
        "open_questions": [],
        "last_updated": "",
        "portfolio_snapshot": {
            "nav": 0.0,
            "open_trades": 0,
            "total_realized_pnl": 0.0,
            "session_trades": 0,
            "session_wins": 0,
            "session_losses": 0,
            "win_rate": 0.0,
        },
        "improvement_focus": "",
        "scan_history": [],
        "recent_learnings": [],
    }

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initialize the state engine.

        Args:
            base_dir: Project root directory. Defaults to the current
                working directory. The state file path is resolved
                relative to this directory.
        """
        self._base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._state_path = self._base_dir / self.STATE_FILE

    @property
    def state_path(self) -> Path:
        """Absolute path to the state file."""
        return self._state_path

    def load_state(self) -> dict[str, Any]:
        """Read and return the persisted state.

        Returns the full state dictionary from disk. If the file does
        not exist or contains invalid JSON, returns a default skeleton
        that preserves the expected schema.
        """
        if not self._state_path.exists():
            return self._build_default_state()

        try:
            raw = self._state_path.read_text(encoding="utf-8")
            state = json.loads(raw)
            if not isinstance(state, dict):
                return self._build_default_state()
            return state
        except (json.JSONDecodeError, OSError):
            return self._build_default_state()

    def save_state(self, **kwargs: Any) -> None:
        """Merge keyword arguments into existing state and persist.

        Performs a shallow merge of kwargs into the current state,
        stamps ``last_updated`` with the current UTC time, and writes
        the result back to disk atomically.

        Args:
            **kwargs: Key-value pairs to merge into state. Values
                overwrite existing keys at the top level.
        """
        state = self.load_state()
        state.update(kwargs)
        state["last_updated"] = self._utc_now_iso()
        self._write_state(state)

    def update_portfolio_snapshot(
        self,
        nav: float,
        open_trades: int,
        total_realized_pnl: float,
        session_trades: int,
        session_wins: int,
        session_losses: int,
    ) -> None:
        """Update the portfolio_snapshot sub-object.

        Computes win_rate from session_wins and session_trades, then
        persists the updated snapshot.

        Args:
            nav: Current net asset value.
            open_trades: Number of currently open positions.
            total_realized_pnl: Cumulative realized profit/loss.
            session_trades: Total trades in the current session.
            session_wins: Winning trades in the current session.
            session_losses: Losing trades in the current session.
        """
        win_rate = round(session_wins / session_trades, 2) if session_trades > 0 else 0.0

        snapshot = {
            "nav": nav,
            "open_trades": open_trades,
            "total_realized_pnl": total_realized_pnl,
            "session_trades": session_trades,
            "session_wins": session_wins,
            "session_losses": session_losses,
            "win_rate": win_rate,
        }

        self.save_state(portfolio_snapshot=snapshot)

    def record_scan_cycle(
        self,
        scan_result_count: int,
        tradeable_count: int,
        duration_secs: float,
    ) -> None:
        """Record a completed scan cycle in the scan history.

        Appends a timestamped entry and trims the list to the most
        recent ``_MAX_SCAN_HISTORY`` entries.

        Args:
            scan_result_count: Number of pairs scanned.
            tradeable_count: Number of tradeable setups found.
            duration_secs: Wall-clock duration of the scan in seconds.
        """
        state = self.load_state()
        history = state.get("scan_history", [])

        entry = {
            "timestamp": self._utc_now_iso(),
            "scan_result_count": scan_result_count,
            "tradeable_count": tradeable_count,
            "duration_secs": round(duration_secs, 2),
        }
        history.append(entry)
        history = history[-self._MAX_SCAN_HISTORY :]

        state["scan_history"] = history
        state["last_updated"] = self._utc_now_iso()
        self._write_state(state)

    def record_learning(self, learning_text: str) -> None:
        """Record a learning insight.

        Appends a timestamped learning entry and trims the list to the
        most recent ``_MAX_RECENT_LEARNINGS`` entries.

        Args:
            learning_text: Free-text description of the learning.
        """
        state = self.load_state()
        learnings = state.get("recent_learnings", [])

        entry = {
            "timestamp": self._utc_now_iso(),
            "text": learning_text,
        }
        learnings.append(entry)
        learnings = learnings[-self._MAX_RECENT_LEARNINGS :]

        state["recent_learnings"] = learnings
        state["last_updated"] = self._utc_now_iso()
        self._write_state(state)

    def get_session_summary(self) -> dict[str, Any]:
        """Return a summary dictionary of the current session state.

        Provides a snapshot suitable for logging or display, including
        portfolio metrics, recent scan activity, and learning count.
        """
        state = self.load_state()
        snapshot = state.get("portfolio_snapshot", {})
        scan_history = state.get("scan_history", [])
        learnings = state.get("recent_learnings", [])

        summary: dict[str, Any] = {
            "status": state.get("status", "unknown"),
            "goal": state.get("goal", ""),
            "next": state.get("next", ""),
            "last_updated": state.get("last_updated", ""),
            "portfolio": {
                "nav": snapshot.get("nav", 0.0),
                "open_trades": snapshot.get("open_trades", 0),
                "total_realized_pnl": snapshot.get("total_realized_pnl", 0.0),
                "win_rate": snapshot.get("win_rate", 0.0),
                "session_trades": snapshot.get("session_trades", 0),
            },
            "scan_cycles_recorded": len(scan_history),
            "recent_learnings_count": len(learnings),
            "improvement_focus": state.get("improvement_focus", ""),
        }

        if scan_history:
            last_scan = scan_history[-1]
            summary["last_scan"] = {
                "timestamp": last_scan.get("timestamp", ""),
                "pairs_scanned": last_scan.get("scan_result_count", 0),
                "tradeable": last_scan.get("tradeable_count", 0),
                "duration_secs": last_scan.get("duration_secs", 0.0),
            }

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_default_state(self) -> dict[str, Any]:
        """Return a fresh copy of the default state skeleton."""
        state = json.loads(json.dumps(self._DEFAULT_STATE))
        state["last_updated"] = self._utc_now_iso()
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        """Write state to disk, creating parent directories if needed."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(state, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(self._state_path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @staticmethod
    def _utc_now_iso() -> str:
        """Return the current UTC time as an ISO 8601 string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
