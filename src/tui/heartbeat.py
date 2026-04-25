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
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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
) -> Path:
    """Atomically write heartbeat.json — tmp file + os.replace to avoid torn reads."""
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

    fd, tmp_path = tempfile.mkstemp(
        prefix=".heartbeat.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
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
