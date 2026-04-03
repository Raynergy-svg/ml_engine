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

try:
    from src.scanner.automation.safe_json import safe_json_write as _safe_json_write
except ImportError:
    _safe_json_write = None  # fallback to atomic inline write if not available


def _atomic_write(path: Path, data: Any) -> None:
    """Atomic JSON write: write to .tmp then os.rename. H-1 fix."""
    if _safe_json_write is not None:
        _safe_json_write(path, data)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception as e:
        logger.error(f"_atomic_write failed for {path}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise

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
    # Tier 7: Control plane state
    "control_plane": {
        "session_id": "",
        "transport_state": "disconnected",
        "degraded_mode": False,
        "degraded_reason": "",
        "last_heartbeat": "",
        "reconnect_attempts": 0,
    },
    "queue_summary": {
        "depth": 0,
        "in_flight": 0,
        "failed_count": 0,
        "last_failure": "",
    },
    "last_policy_block": {
        "action_type": "",
        "reason": "",
        "timestamp": "",
    },
}


class StateEngine:
    """Manages .claude/state.json for cross-session continuity."""

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or STATE_PATH

    def load_state(self) -> Dict[str, Any]:
        """Read .claude/state.json with shared file lock (or empty default if missing)."""
        if not self.state_path.exists():
            return dict(_DEFAULT_STATE)
        try:
            import fcntl
            with open(self.state_path, "r", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.loads(f.read())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except ImportError:
            # fcntl not available — fall back to unlocked read
            data = json.loads(self.state_path.read_text())
        except json.JSONDecodeError as e:
            logger.warning(f"State file corrupted: {e}")
            return dict(_DEFAULT_STATE)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return dict(_DEFAULT_STATE)

        # Validate required keys exist
        required_keys = {"goal", "status", "done", "next", "last_updated"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            logger.warning(f"State missing keys {missing_keys}, merging with defaults")
            defaults = dict(_DEFAULT_STATE)
            defaults.update(data)
            data = defaults
        return data

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
        _atomic_write(self.state_path, state)  # H-1: atomic write prevents corruption
        logger.info("State saved to %s", self.state_path)

    def update_portfolio_snapshot(self) -> Dict[str, Any]:
        """Fetch NAV from OANDA and open trade count, update state."""
        import requests

        state = self.load_state()
        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")

        # Validate env vars before API call
        if not token or not isinstance(token, str):
            logger.warning("OANDA_API_TOKEN not set or invalid, skipping portfolio update")
            return state.get("portfolio_snapshot", dict(_DEFAULT_STATE["portfolio_snapshot"]))
        if not acct or not isinstance(acct, str):
            logger.warning("OANDA_ACCOUNT_ID not set or invalid, skipping portfolio update")
            return state.get("portfolio_snapshot", dict(_DEFAULT_STATE["portfolio_snapshot"]))

        base = "https://api-fxpractice.oanda.com"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        snapshot = state.get("portfolio_snapshot", dict(_DEFAULT_STATE["portfolio_snapshot"]))

        try:
            resp = requests.get(
                f"{base}/v3/accounts/{acct}/summary",
                headers=headers,
                timeout=(5, 30),
            )
            if resp.status_code == 200:
                resp_json = resp.json()
                # Validate response structure
                if not isinstance(resp_json, dict) or "account" not in resp_json:
                    logger.warning("OANDA response missing 'account' key, skipping update")
                else:
                    acct_data = resp_json.get("account", {})
                    if isinstance(acct_data, dict):
                        snapshot["nav"] = float(acct_data.get("NAV", 0))
                        snapshot["open_trades"] = int(acct_data.get("openTradeCount", 0))
                        snapshot["total_realized_pnl"] = float(acct_data.get("pl", 0))
            elif resp.status_code == 401:
                logger.warning("OANDA API: Unauthorized (401) — check credentials")
            elif resp.status_code == 429:
                logger.warning("OANDA API: Rate limit (429) — backing off")
            else:
                logger.warning(f"OANDA API returned status {resp.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"OANDA request failed: {e}")
        except Exception as e:
            logger.warning(f"Failed to fetch OANDA account summary: {e}")

        # Update trade stats from journal
        try:
            journal_path = Path("trained_data/trade_journal_rl.json")
            if journal_path.exists():
                journal_text = journal_path.read_text()
                if not journal_text.strip():
                    logger.debug("Journal file is empty, using default stats")
                else:
                    entries = json.loads(journal_text)
                    if not isinstance(entries, list):
                        logger.warning("Journal file is not a JSON list, skipping stats")
                    else:
                        closed = [e for e in entries if e.get("outcome") is not None]
                        # C-6: outcome can be a legacy string ("win") or a dict — guard both
                        wins = sum(
                            1 for e in closed
                            if isinstance(e.get("outcome"), dict) and e["outcome"].get("trade_won", False)
                        )
                        losses = len(closed) - wins
                        snapshot["session_trades"] = len(closed)
                        snapshot["session_wins"] = wins
                        snapshot["session_losses"] = losses
                        snapshot["win_rate"] = round(wins / len(closed), 2) if closed else 0.0
            else:
                logger.debug("Journal file does not exist, using default stats")
        except json.JSONDecodeError as e:
            logger.warning(f"Journal file contains invalid JSON: {e}, using default stats")
        except Exception as e:
            logger.debug(f"Journal stats error: {e}")

        state["portfolio_snapshot"] = snapshot
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.state_path, state)  # H-1: atomic write prevents corruption
        logger.info("Portfolio snapshot updated: NAV=%.2f, open=%d", snapshot["nav"], snapshot["open_trades"])
        return snapshot

    def increment_scan_cycle(self) -> int:
        """Increment and return the scan cycle count (stored in state)."""
        state = self.load_state()
        count = state.get("scan_cycle_count", 0) + 1
        state["scan_cycle_count"] = count
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(self.state_path, state)  # H-1: atomic write prevents corruption
        return count
