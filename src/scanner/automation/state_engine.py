"""Session state engine for cross-session continuity.

Persists trading state between sessions so the next session can resume
intelligently without re-discovering context.

US-002: Create session state engine for cross-session continuity.
US-501: schema_version=2 adds supervisor flags (halted, scanner_paused, mode, last_actor).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_PATH = Path(".claude/state.json")

SCHEMA_VERSION = "2"

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
    # US-501 supervisor flags
    "halted": False,
    "scanner_paused": False,
    "mode": "dry_run",
    "last_actor": "",
    "schema_version": SCHEMA_VERSION,
}

# Schema v2 fields added during migration
_V2_FIELDS: Dict[str, Any] = {
    "halted": False,
    "scanner_paused": False,
    "mode": "dry_run",
    "last_actor": "",
    "schema_version": SCHEMA_VERSION,
}


class StateEngine:
    """Manages .claude/state.json for cross-session continuity."""

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or STATE_PATH
        self._lock = threading.Lock()

    # ── read / write ────────────────────────────────────────────────

    def load_state(self) -> Dict[str, Any]:
        """Read .claude/state.json, migrating to schema v2 if necessary."""
        if not self.state_path.exists():
            return dict(_DEFAULT_STATE)
        try:
            data = json.loads(self.state_path.read_text())
            migrated, changed = self._migrate(data)
            if changed:
                # Persist the migrated state so the file reflects v2 schema
                try:
                    self._atomic_write(migrated)
                    logger.info("StateEngine: migrated state.json to schema v%s", SCHEMA_VERSION)
                except Exception as e:
                    logger.warning("StateEngine: could not persist migration: %s", e)
            return migrated
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return dict(_DEFAULT_STATE)

    def _migrate(self, data: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """Add v2 fields if missing. Returns (data, changed)."""
        version = data.get("schema_version", "1")
        if version < SCHEMA_VERSION:
            for key, default in _V2_FIELDS.items():
                if key not in data:
                    data[key] = default
            data["schema_version"] = SCHEMA_VERSION
            return data, True
        return data, False

    def _atomic_write(self, state: Dict[str, Any]) -> None:
        """Write state atomically via a .tmp file to prevent corruption."""
        tmp = self.state_path.with_suffix(".json.tmp")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(state, indent=2, default=str))
            os.replace(str(tmp), str(self.state_path))
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

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
        import requests  # type: ignore[import-untyped]

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

    # ── US-501 supervisor flags ─────────────────────────────────────

    def _update_flag(self, key: str, value: Any) -> None:
        """Read-modify-write a single flag atomically under instance lock."""
        with self._lock:
            state = self.load_state()
            state[key] = value
            state["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._atomic_write(state)

    def get_halted(self) -> bool:
        """Return whether the scanner is halted."""
        return bool(self.load_state().get("halted", False))

    def set_halted(self, value: bool) -> None:
        """Set the halted flag and persist atomically."""
        self._update_flag("halted", value)
        logger.info("StateEngine: halted=%s", value)

    def set_last_actor(self, actor: str) -> None:
        """Record the last supervisor/control actor in state."""
        self._update_flag("last_actor", str(actor or ""))
        logger.info("StateEngine: last_actor=%s", actor)

    def get_paused(self) -> bool:
        """Return whether the scanner is paused."""
        return bool(self.load_state().get("scanner_paused", False))

    def set_paused(self, value: bool) -> None:
        """Set the scanner_paused flag and persist atomically."""
        self._update_flag("scanner_paused", value)
        logger.info("StateEngine: scanner_paused=%s", value)

    def get_mode(self) -> str:
        """Return the current trading mode ('dry_run' or 'live')."""
        return str(self.load_state().get("mode", "dry_run"))

    def set_mode(self, mode: str) -> None:
        """Set the trading mode and persist atomically."""
        if mode not in ("dry_run", "live"):
            raise ValueError(f"Invalid mode '{mode}': must be 'dry_run' or 'live'")
        self._update_flag("mode", mode)
        logger.info("StateEngine: mode=%s", mode)
