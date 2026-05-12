"""Hermes watchdog + daily brief state persistence.

One on-disk JSON at .claude/hermes_state.json. Atomic writes via the
cloud-branch safe_json utilities. Forward-compatible: missing fields
load with their dataclass defaults so adding fields in future patches
never breaks an older state file.

Buddy policies honored:
- safe_json_read with try/except + graceful fallback to defaults
- safe_json_write (atomic temp+rename + fcntl lock under the hood)
- No bare except: clauses
- Default values are sensible (None / empty dict) so first-run is safe
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from src.scanner.automation.safe_json import safe_json_read, safe_json_write


_DEFAULT_STATE_PATH = (
    Path(__file__).resolve().parents[3] / ".claude" / "hermes_state.json"
)


@dataclass
class HermesState:
    """Cross-tick state for the watchdog + daily brief."""

    # Watchdog dedup — rate-limit silent entries to once per 4h
    last_silent_at_iso: Optional[str] = None

    # Watchdog dedup — alert_type → ISO of last watch entry written for it
    # Prevents the same active alert from re-firing every 30 min
    last_alert_keys: Dict[str, str] = field(default_factory=dict)

    # Watchdog dedup — job_id → ISO of last watch entry written for it
    # Prevents repeated job failures from spamming the digest
    last_job_failure_keys: Dict[str, str] = field(default_factory=dict)

    # Brief — for cycle_count delta computation
    last_brief_at_iso: Optional[str] = None
    cycle_count_at_last_brief: Optional[int] = None

    # ── persistence ────────────────────────────────────────────────────

    @classmethod
    def from_path(cls, path: Path = _DEFAULT_STATE_PATH) -> "HermesState":
        """Load from disk; return defaults on missing/corrupt/parse-error.

        Forward-compatible — unknown keys in the file are ignored, missing
        keys load as their dataclass defaults.
        """
        raw = safe_json_read(path, default=None)
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known}
        try:
            return cls(**filtered)
        except (TypeError, ValueError):
            # Schema drift in the file — start fresh rather than crash the
            # watchdog. Old state is overwritten on next save.
            return cls()

    def save_to(self, path: Path = _DEFAULT_STATE_PATH) -> bool:
        """Atomic write. Returns True on success, False on I/O failure."""
        return bool(safe_json_write(path, asdict(self), sort_keys=True))
