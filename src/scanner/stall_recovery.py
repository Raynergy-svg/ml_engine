"""Stall Recovery Manager — Phase 57 US-354.

When ScanDiagnosticsReporter reports that the engine has been stalled for
>= STALLED_THRESHOLD scans (0 tradeable pairs), this module enters RECOVERY
state for RECOVERY_WINDOW scans and suspends non-essential penalties
(overconfidence, concept drift) to test whether those penalties are the
cause of the suppression.

The recovery cycle logic:
    NORMAL  → (scans_since_last_trade >= threshold) → RECOVERY
    RECOVERY → (recovery_scans_remaining > 0)         → RECOVERY
    RECOVERY → (recovery_scans_remaining == 0)         → NORMAL
    NORMAL  → (still stalled after recovery)           → RECOVERY (again)

When in RECOVERY, penalties_suspended() returns True. engine.py checks
this before applying overconfidence (-3pt) and concept_drift multiplier.

Usage:
    recovery = StallRecoveryManager()
    recovery.tick(scans_since_last_trade=55)
    if recovery.penalties_suspended():
        # skip overconfidence and drift penalties this scan
        pass
    state = recovery.get_state()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_STATE_PATH = _PROJECT_ROOT / "trained_data" / "stall_recovery_state.json"

STALL_THRESHOLD = 50          # Scans without trade before entering recovery
RECOVERY_WINDOW = 5           # Number of scans penalties are suspended

_STATE_VERSION = 1


class StallRecoveryState:
    NORMAL = "NORMAL"
    RECOVERY = "RECOVERY"


class StallRecoveryManager:
    """Manages stall detection and penalty suspension for stall recovery cycles.

    Usage:
        manager = StallRecoveryManager()
        # Called once per scan cycle:
        manager.tick(scans_since_last_trade)
        if manager.penalties_suspended():
            # Skip overconfidence and drift penalties
    """

    def __init__(
        self,
        stall_threshold: int = STALL_THRESHOLD,
        recovery_window: int = RECOVERY_WINDOW,
        state_path: Optional[Path] = None,
    ):
        self.stall_threshold = int(stall_threshold)
        self.recovery_window = int(recovery_window)
        self.state_path = Path(state_path or _DEFAULT_STATE_PATH)

        self._state: str = StallRecoveryState.NORMAL
        self._recovery_scans_remaining: int = 0
        self._total_recovery_cycles: int = 0
        self._total_ticks: int = 0

        self._load_state()

    # ── Public API ─────────────────────────────────────────────────────────

    def tick(self, scans_since_last_trade: int) -> None:
        """Advance the state machine by one scan cycle.

        Args:
            scans_since_last_trade: Number of consecutive scans with no tradeable pair.
        """
        self._total_ticks += 1

        if self._state == StallRecoveryState.NORMAL:
            if scans_since_last_trade >= self.stall_threshold:
                self._enter_recovery()

        elif self._state == StallRecoveryState.RECOVERY:
            self._recovery_scans_remaining -= 1
            if self._recovery_scans_remaining <= 0:
                self._exit_recovery()
                # Check immediately if still stalled → re-enter
                if scans_since_last_trade >= self.stall_threshold:
                    self._enter_recovery()

    def penalties_suspended(self) -> bool:
        """Return True when in RECOVERY state — non-essential penalties should be skipped."""
        return self._state == StallRecoveryState.RECOVERY

    def is_in_recovery(self) -> bool:
        """Alias for penalties_suspended() — more readable in caller context."""
        return self._state == StallRecoveryState.RECOVERY

    def get_state(self) -> Dict[str, Any]:
        """Return current state as a diagnostic dict."""
        return {
            "state": self._state,
            "recovery_scans_remaining": self._recovery_scans_remaining,
            "total_recovery_cycles": self._total_recovery_cycles,
            "total_ticks": self._total_ticks,
            "stall_threshold": self.stall_threshold,
            "recovery_window": self.recovery_window,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def save_state(self) -> None:
        """Atomically persist state to disk."""
        data = {
            "version": _STATE_VERSION,
            "state": self._state,
            "recovery_scans_remaining": self._recovery_scans_remaining,
            "total_recovery_cycles": self._total_recovery_cycles,
            "total_ticks": self._total_ticks,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write(self.state_path, data)

    # ── Internal ────────────────────────────────────────────────────────────

    def _enter_recovery(self) -> None:
        self._state = StallRecoveryState.RECOVERY
        self._recovery_scans_remaining = self.recovery_window
        self._total_recovery_cycles += 1
        logger.warning(
            "Stall recovery mode activated — penalties suspended for %d scans "
            "(recovery cycle #%d)",
            self.recovery_window, self._total_recovery_cycles,
        )

    def _exit_recovery(self) -> None:
        self._state = StallRecoveryState.NORMAL
        self._recovery_scans_remaining = 0
        logger.info("Stall recovery mode ended — returning to NORMAL penalty mode")

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text())
            if not isinstance(raw, dict):
                return
            self._state = str(raw.get("state", StallRecoveryState.NORMAL))
            self._recovery_scans_remaining = int(raw.get("recovery_scans_remaining", 0))
            self._total_recovery_cycles = int(raw.get("total_recovery_cycles", 0))
            self._total_ticks = int(raw.get("total_ticks", 0))
            logger.debug(
                "StallRecoveryManager: Loaded state=%s remaining=%d cycles=%d",
                self._state, self._recovery_scans_remaining, self._total_recovery_cycles,
            )
        except Exception as e:
            logger.warning("StallRecoveryManager: Load state failed: %s", e)

    def _atomic_write(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, path)
        except Exception as e:
            logger.warning("StallRecoveryManager: Write to %s failed: %s", path, e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
