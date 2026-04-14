"""Session state engine for cross-session continuity.

Persists trading state between sessions so the next session can resume
intelligently without re-discovering context.

US-002: Create session state engine for cross-session continuity.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_PATH = Path(".claude/state.json")

_DEFAULT_STATE: Dict[str, Any] = {
    "goal": "",
    "status": "ready",
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
}


class StateEngine:
    """Manages .claude/state.json for cross-session continuity."""

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or STATE_PATH

    def load_state(self) -> Dict[str, Any]:
        """Read .claude/state.json and return dict (or empty default if missing)."""
        if not self.state_path.exists():
            return dict(_DEFAULT_STATE)
        try:
            data = json.loads(self.state_path.read_text())
            return data
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return dict(_DEFAULT_STATE)

    def save_state(
        self,
        goal: str,
        status: str,
        done: List[str],
        next_action: str,
        open_questions: Optional[List[str]] = None,
        portfolio: Optional[Dict[str, Any]] = None,
        improvement_focus: str = "",
    ) -> None:
        """Write state to .claude/state.json."""
        state = {
            "goal": goal,
            "status": status,
            "done": done,
            "next": next_action,
            "open_questions": open_questions or [],
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "portfolio_snapshot": portfolio or _DEFAULT_STATE["portfolio_snapshot"],
            "improvement_focus": improvement_focus,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str))
        logger.info("State saved to %s", self.state_path)

    def update_portfolio_snapshot(self) -> Dict[str, Any]:
        """Fetch NAV from OANDA and open trade count, update state."""
        import requests

        state = self.load_state()
        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = "https://api-fxpractice.oanda.com"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        snapshot = state.get("portfolio_snapshot", dict(_DEFAULT_STATE["portfolio_snapshot"]))

        try:
            resp = requests.get(
                f"{base}/v3/accounts/{acct}/summary",
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                acct_data = resp.json().get("account", {})
                snapshot["nav"] = float(acct_data.get("NAV", 0))
                snapshot["open_trades"] = int(acct_data.get("openTradeCount", 0))
                snapshot["total_realized_pnl"] = float(acct_data.get("pl", 0))
        except Exception as e:
            logger.warning(f"Failed to fetch OANDA account summary: {e}")

        # Update trade stats from journal
        try:
            journal_path = Path("trained_data/trade_journal_rl.json")
            if journal_path.exists():
                entries = json.loads(journal_path.read_text())
                closed = [e for e in entries if e.get("outcome") is not None]
                wins = sum(1 for e in closed if e["outcome"].get("trade_won", False))
                losses = len(closed) - wins
                snapshot["session_trades"] = len(closed)
                snapshot["session_wins"] = wins
                snapshot["session_losses"] = losses
                snapshot["win_rate"] = round(wins / len(closed), 2) if closed else 0.0
        except Exception as e:
            logger.debug(f"Journal stats error: {e}")

        state["portfolio_snapshot"] = snapshot
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str))
        logger.info("Portfolio snapshot updated: NAV=%.2f, open=%d", snapshot["nav"], snapshot["open_trades"])
        return snapshot

    def increment_scan_cycle(self) -> int:
        """Increment and return the scan cycle count (stored in state)."""
        state = self.load_state()
        count = state.get("scan_cycle_count", 0) + 1
        state["scan_cycle_count"] = count
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_path.write_text(json.dumps(state, indent=2, default=str))
        return count
