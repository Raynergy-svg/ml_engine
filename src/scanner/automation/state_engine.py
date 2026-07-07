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

# US-per-lane-halt: the known trading/research lanes that can be halted
# independently of one another (once the legacy global ``halted`` flag is
# False). Any lane name outside this set is rejected — typo'd/orphan lane
# keys must fail loud, not silently create a dead ``halted_lanes`` entry
# nothing reads (mirrors the ScannerConfig key-validation convention).
KNOWN_LANES = ("oanda_fx", "equity", "brain", "crypto_momentum", "track_b", "crypto_carry")

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

    def __init__(self, state_path: Optional[Path] = None, lane: Optional[str] = None):
        self.state_path = state_path or STATE_PATH
        self._lock = threading.Lock()
        if lane is not None and lane not in KNOWN_LANES:
            raise ValueError(f"Unknown lane {lane!r}: must be one of {KNOWN_LANES}")
        self._lane = lane  # optional default lane for get_halted()/set_halted() no-arg calls

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

    def get_halted(self, lane: Optional[str] = None) -> bool:
        """Return whether the scanner (or a specific lane) is halted.

        ``lane`` defaults to the lane this instance was constructed with
        (``StateEngine(lane="oanda_fx")``), if any — so a lane-scoped
        instance can be read with the exact same ``get_halted()`` call every
        pre-existing caller already uses. An explicit ``lane=`` argument here
        always overrides the instance's bound lane.

        ``lane=None`` (default, unchanged from before per-lane support):
        legacy global check — fail-OPEN on a missing/corrupt state file
        (matches long-standing behavior; ``src.equity.decision_gate``
        deliberately reads the file directly instead of via StateEngine to
        be fail-CLOSED, and keeps doing so).

        ``lane=<name>``: per-lane check, layered on top of the same
        ``load_state()`` read. The legacy global ``halted=True`` flag still
        halts every lane (fail-safe OR — a lane can never be "more unhalted"
        than the global switch). Two distinct "unknown" cases are handled
        differently, deliberately:

          * ``halted_lanes`` is entirely ABSENT (a legacy state.json that
            predates per-lane support, or a brand-new/corrupt file that
            ``load_state()`` silently falls back to defaults for) -> defers
            to the (already-checked, already-False-here) global flag. This
            preserves every pre-existing StateEngine consumer's fail-OPEN
            contract on a missing/corrupt file — the equity harvester's
            ``src.equity.decision_gate`` is the module that reads the file
            directly for a genuinely fail-CLOSED guarantee; StateEngine
            itself has never made that promise (see its class docstring
            precedent) and this method does not newly invent one for it.
          * ``halted_lanes`` IS present but has no valid boolean entry for
            THIS lane -> fail-closed (halted). Once the per-lane concept is
            in use at all, an unlisted/malformed lane is never assumed safe.
        """
        lane = lane if lane is not None else self._lane
        state = self.load_state()
        if bool(state.get("halted", False)):
            return True
        if lane is None:
            return False
        if lane not in KNOWN_LANES:
            raise ValueError(f"Unknown lane {lane!r}: must be one of {KNOWN_LANES}")
        lanes = state.get("halted_lanes")
        if lanes is None:
            return False  # per-lane concept not configured at all -> defer to global (False here)
        if not isinstance(lanes, dict):
            return True  # corrupt type where a dict was expected -> fail-closed
        val = lanes.get(lane)
        if not isinstance(val, bool):
            return True  # fail-closed: halted_lanes exists but this lane's entry is missing/corrupt
        return val

    def get_halted_strict(self, lane: Optional[str] = None) -> bool:
        """Fail-CLOSED halt check for last-line-of-defense guards immediately
        before an order reaches the broker (execution.py execute_trade /
        close_trade mid-cycle re-checks, embedded_scanner's pre-cycle check).

        Unlike ``get_halted()`` — which treats a missing/corrupt state.json as
        "not halted" so early orchestration checks don't false-positive-block
        on a fresh install — this method treats ANY unreadable state as
        halted: missing file, corrupt JSON, a non-dict payload, a missing
        ``halted_lanes`` dict, or a missing/non-bool entry for this lane. A
        lane must be explicitly told "not halted" to run; it is never assumed
        safe by omission or by read failure.

        Reads the file directly (not via ``load_state()``, which swallows
        JSON errors and returns defaults) so a corrupt file cannot silently
        resolve to "not halted". Mirrors ``src.equity.decision_gate.
        _lane_halted``'s fail-closed contract (kept deliberately in sync —
        see that function's docstring for the original design rationale).

        Promoted to StateEngine 2026-07-02 (operator decision, not a silent
        widening) so the FX mid-cycle guards get the same fail-closed
        guarantee the equity harvester's decision gate has always had.

        Raises ``ValueError`` if no lane is bound (constructor or arg) or the
        lane is unknown — there is no "global-only" strict variant, since a
        real global ``halted=True`` already fails closed via ``get_halted()``.
        """
        lane = lane if lane is not None else self._lane
        if lane is None:
            raise ValueError(
                "get_halted_strict requires a lane (bind via StateEngine(lane=...) or pass lane=)"
            )
        if lane not in KNOWN_LANES:
            raise ValueError(f"Unknown lane {lane!r}: must be one of {KNOWN_LANES}")
        if not self.state_path.exists():
            logger.warning(
                "StateEngine.get_halted_strict: %s missing — fail-closed (halted), lane=%s",
                self.state_path, lane,
            )
            return True
        try:
            payload = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "StateEngine.get_halted_strict: %s unreadable (%s) — fail-closed (halted), lane=%s",
                self.state_path, e, lane,
            )
            return True
        if not isinstance(payload, dict):
            logger.warning(
                "StateEngine.get_halted_strict: %s malformed (not a dict) — fail-closed (halted), lane=%s",
                self.state_path, lane,
            )
            return True
        if bool(payload.get("halted", False)):
            return True
        lanes = payload.get("halted_lanes")
        if not isinstance(lanes, dict) or not isinstance(lanes.get(lane), bool):
            return True
        return bool(lanes[lane])

    def get_lane_status(self) -> Dict[str, bool]:
        """Return the effective halted state of every known lane (for
        dashboard/TUI readback) — each computed via the same fail-closed
        ``get_halted(lane=...)`` path used to gate execution."""
        return {lane: self.get_halted(lane) for lane in KNOWN_LANES}

    def set_halted(self, value: bool, lane: Optional[str] = None) -> None:
        """Set the halted flag and persist atomically.

        ``lane=None`` (default): sets the legacy global ``halted`` flag AND
        cascades the same value to every known lane's ``halted_lanes``
        entry. This is the "master switch" a plain ``set_halted(value)``
        call has always been — every existing autonomous halt path
        (execution.py flatten_all, engine.py auto-halt, brain_loop derisk)
        and every existing operator unhalt path keeps working unmodified,
        and now also keeps the per-lane view consistent for callers that
        have not opted into per-lane control.

        ``lane=<name>``: overrides ONLY that lane's entry in
        ``halted_lanes``, without touching the global flag or any other
        lane. This is how a lane is halted/unhalted independently — note
        the global flag must ALSO be False for a lane-level unhalt to have
        any effect (``halted=True`` always wins; see ``get_halted``).

        ``lane`` defaults to the lane this instance was constructed with
        (see ``get_halted`` for the same convention), so a lane-scoped
        instance can be written with the exact same ``set_halted(value)``
        call every pre-existing caller already uses.
        """
        lane = lane if lane is not None else self._lane
        with self._lock:
            state = self.load_state()
            if lane is None:
                state["halted"] = bool(value)
                state["halted_lanes"] = {ln: bool(value) for ln in KNOWN_LANES}
            else:
                if lane not in KNOWN_LANES:
                    raise ValueError(f"Unknown lane {lane!r}: must be one of {KNOWN_LANES}")
                lanes = state.get("halted_lanes")
                lanes = dict(lanes) if isinstance(lanes, dict) else {}
                lanes[lane] = bool(value)
                state["halted_lanes"] = lanes
            state["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._atomic_write(state)
        logger.info("StateEngine: halted=%s lane=%s", value, lane or "ALL (global)")

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
