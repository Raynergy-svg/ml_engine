"""Heartbeat writer (US-603).

Writes ``.claude/heartbeat.json`` atomically so an external watchdog daemon
can detect a hung TUI and trigger a restart via launchctl.

Schema:
    {
        "ts_iso": "2026-04-24T22:38:00+00:00",  # UTC ISO8601
        "pid": 12345,
        "cycle_count": 42,
        "mode": "live" | "demo",
        "scanner_alive": true,
        "last_error_ts": null | "2026-04-24T22:30:00+00:00"
    }
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.scanner.automation.safe_json import safe_json_write


HEARTBEAT_RELPATH = ".claude/heartbeat.json"


@dataclass
class HeartbeatPayload:
    ts_iso: str
    pid: int
    cycle_count: int
    mode: str
    scanner_alive: bool
    last_error_ts: Optional[str]

    def to_dict(self) -> dict:
        return {
            "ts_iso": self.ts_iso,
            "pid": self.pid,
            "cycle_count": self.cycle_count,
            "mode": self.mode,
            "scanner_alive": self.scanner_alive,
            "last_error_ts": self.last_error_ts,
        }


def heartbeat_path(repo_root: Path) -> Path:
    return repo_root / HEARTBEAT_RELPATH


def write_heartbeat(
    repo_root: Path,
    *,
    cycle_count: int,
    mode: str,
    scanner_alive: bool,
    last_error_ts: Optional[str] = None,
    pid: Optional[int] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Atomically write heartbeat.json via safe_json_write to avoid torn reads.

    ``extra`` merges ADDITIVE keys into the payload (e.g. ``writer``,
    ``supervisor_alive`` from the tier7 supervisor). It can never override the
    core schema keys — a heartbeat-schema mismatch once silently blocked every
    unhalt attempt (2026-05-12 incident), so the core contract is protected.
    """
    target = heartbeat_path(repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = HeartbeatPayload(
        ts_iso=datetime.now(timezone.utc).isoformat(),
        pid=pid if pid is not None else os.getpid(),
        cycle_count=int(cycle_count),
        mode=str(mode),
        scanner_alive=bool(scanner_alive),
        last_error_ts=last_error_ts,
    )
    body = payload.to_dict()
    if extra:
        for key, value in extra.items():
            if key not in body:  # additive only — never clobber the core schema
                body[key] = value

    # High-frequency write (every 10s) — disable .bak to avoid extra fsync per beat.
    if not safe_json_write(target, body, sort_keys=True, create_backup=False):
        raise IOError(f"heartbeat: safe_json_write failed for {target}")
    return target


def read_heartbeat(repo_root: Path) -> Optional[dict]:
    """Read heartbeat.json. Returns None if missing or corrupt."""
    target = heartbeat_path(repo_root)
    try:
        with target.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data
