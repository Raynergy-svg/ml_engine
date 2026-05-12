"""T2: LivenessBadge — pure-function + real-disk tests.

NO MOCKS — these tests follow ``.claude/rules/improvement.md`` "No-Mock Rule"
(promoted 2026-05-01). Every JSON read is against a real ``tmp_path`` disk
state written by the test itself; the time argument is injected via the
``now=`` parameter on ``compute_liveness`` for determinism.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.tui.widgets.liveness_badge import (
    STALE_THRESHOLD_SECONDS,
    LivenessState,
    compute_liveness,
)


def _write_heartbeat(claude_dir: Path, *, age_seconds: float, scanner_alive: bool,
                     cycle_count: int = 7, now: datetime) -> datetime:
    """Write a real heartbeat.json with ts_iso = now - age_seconds."""
    claude_dir.mkdir(parents=True, exist_ok=True)
    ts = now - timedelta(seconds=age_seconds)
    payload = {
        "ts_iso": ts.isoformat(),
        "pid": 12345,
        "cycle_count": cycle_count,
        "mode": "live",
        "scanner_alive": scanner_alive,
        "last_error_ts": None,
    }
    (claude_dir / "heartbeat.json").write_text(json.dumps(payload, indent=2))
    return ts


def _write_state(claude_dir: Path, *, halted: bool) -> None:
    """Write a real state.json with the given halted flag."""
    claude_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "halted": halted,
        "mode": "live",
        "scan_cycle_count": 100,
    }
    (claude_dir / "state.json").write_text(json.dumps(payload, indent=2))


def test_live_when_heartbeat_fresh_and_not_halted(tmp_path: Path) -> None:
    """Fresh heartbeat (<15s) + scanner_alive + halted=false → LIVE."""
    claude_dir = tmp_path / ".claude"
    now = datetime(2026, 5, 12, 0, 0, 0, tzinfo=timezone.utc)
    _write_heartbeat(claude_dir, age_seconds=3.0, scanner_alive=True,
                     cycle_count=42, now=now)
    _write_state(claude_dir, halted=False)

    state = compute_liveness(claude_dir, now=now)

    assert state.label == "LIVE"
    assert "42" in state.detail  # cycle count shown
    assert state.color  # non-empty


def test_halted_overrides_fresh_heartbeat(tmp_path: Path) -> None:
    """halted=true wins even when the heartbeat is fresh + scanner_alive."""
    claude_dir = tmp_path / ".claude"
    now = datetime(2026, 5, 12, 0, 0, 0, tzinfo=timezone.utc)
    _write_heartbeat(claude_dir, age_seconds=1.0, scanner_alive=True, now=now)
    _write_state(claude_dir, halted=True)

    state = compute_liveness(claude_dir, now=now)

    assert state.label == "HALTED"


def test_stale_when_heartbeat_too_old(tmp_path: Path) -> None:
    """Heartbeat older than 15s → STALE even if scanner_alive=true."""
    claude_dir = tmp_path / ".claude"
    now = datetime(2026, 5, 12, 0, 0, 0, tzinfo=timezone.utc)
    _write_heartbeat(
        claude_dir,
        age_seconds=STALE_THRESHOLD_SECONDS + 5.0,  # ~20s old
        scanner_alive=True,
        now=now,
    )
    _write_state(claude_dir, halted=False)

    state = compute_liveness(claude_dir, now=now)

    assert state.label == "STALE"
    assert "20s" in state.detail or "since heartbeat" in state.detail


def test_init_when_heartbeat_missing(tmp_path: Path) -> None:
    """No heartbeat.json file at all → INIT (state.json may or may not exist)."""
    claude_dir = tmp_path / ".claude"
    # Deliberately do NOT write heartbeat.json. state.json present but not halted.
    _write_state(claude_dir, halted=False)

    state = compute_liveness(claude_dir, now=datetime.now(timezone.utc))

    assert state.label == "INIT"


def test_error_when_heartbeat_corrupt(tmp_path: Path) -> None:
    """Corrupt heartbeat.json (not valid JSON) → ERROR, no crash."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    # Write invalid JSON — must NOT raise from compute_liveness.
    (claude_dir / "heartbeat.json").write_text("{not valid json")
    _write_state(claude_dir, halted=False)

    state = compute_liveness(claude_dir, now=datetime.now(timezone.utc))

    assert state.label == "ERROR"
    # Sanity: the function returned a real LivenessState, didn't raise.
    assert isinstance(state, LivenessState)
