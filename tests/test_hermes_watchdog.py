"""Hermes watchdog — decision tree + dedup."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.scanner.automation.hermes_watchdog import WatchdogContext, run_once


UTC = timezone.utc
FIXED_NOW = datetime(2026, 5, 12, 14, 30, 0, tzinfo=UTC)


def _seed_clean_state(root: Path) -> None:
    """Plant a fully-healthy set of input files under root."""
    claude = root / ".claude"
    claude.mkdir(parents=True, exist_ok=True)
    (claude / "heartbeat.json").write_text(json.dumps({
        "scanner_alive": True,
        "ts_iso": (FIXED_NOW - timedelta(seconds=2)).isoformat(),
        "cycle_count": 100,
        "pid": 1234,
        "mode": "live",
        "last_error_ts": None,
    }))
    (claude / "state.json").write_text(json.dumps({"halted": False, "mode": "live"}))
    (claude / "alert_state.json").write_text(json.dumps({
        "active_alerts": [],
        "last_fired": {},
        "last_updated": FIXED_NOW.isoformat(),
    }))
    (root / "trained_data").mkdir(parents=True, exist_ok=True)
    (root / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({}))


def _ctx(root: Path) -> WatchdogContext:
    return WatchdogContext(
        repo_root=root,
        now=FIXED_NOW,
    )


def test_clean_state_with_no_recent_silent_writes_silent(tmp_path: Path):
    _seed_clean_state(tmp_path)
    result = run_once(_ctx(tmp_path))
    assert result.entries_written == ["silent"]
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "### 14:30Z — silent" in digest


def test_clean_state_with_recent_silent_writes_nothing(tmp_path: Path):
    _seed_clean_state(tmp_path)
    # Mark a recent silent (2h ago) in the state file.
    state_path = tmp_path / ".claude" / "hermes_state.json"
    state_path.write_text(json.dumps({
        "last_silent_at_iso": (FIXED_NOW - timedelta(hours=2)).isoformat(),
    }))
    result = run_once(_ctx(tmp_path))
    assert result.entries_written == []
    # No digest file created at all.
    digest_path = tmp_path / ".claude" / "brain" / "hermes_watchdog.md"
    assert not digest_path.exists()


def test_active_unacknowledged_alert_writes_watch(tmp_path: Path):
    _seed_clean_state(tmp_path)
    alert_state = json.loads((tmp_path / ".claude" / "alert_state.json").read_text())
    alert_state["active_alerts"] = [{
        "alert_type": "consecutive_losses",
        "severity": "WARNING",
        "message": "3 consecutive losses (threshold: 3)",
        "timestamp": FIXED_NOW.isoformat(),
        "value": 3.0,
        "threshold": 3.0,
        "pair": "",
        "acknowledged": False,
    }]
    (tmp_path / ".claude" / "alert_state.json").write_text(json.dumps(alert_state))
    result = run_once(_ctx(tmp_path))
    assert "watch" in result.entries_written
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "consecutive_losses" in digest
    assert "3 consecutive losses" in digest


def test_acknowledged_alert_does_not_write_watch(tmp_path: Path):
    _seed_clean_state(tmp_path)
    alert_state = json.loads((tmp_path / ".claude" / "alert_state.json").read_text())
    alert_state["active_alerts"] = [{
        "alert_type": "consecutive_losses",
        "severity": "WARNING",
        "message": "3 consecutive losses",
        "timestamp": FIXED_NOW.isoformat(),
        "value": 3.0,
        "threshold": 3.0,
        "pair": "",
        "acknowledged": True,
    }]
    (tmp_path / ".claude" / "alert_state.json").write_text(json.dumps(alert_state))
    result = run_once(_ctx(tmp_path))
    assert "watch" not in result.entries_written


def test_same_alert_within_dedup_window_does_not_re_fire(tmp_path: Path):
    _seed_clean_state(tmp_path)
    alert_state = json.loads((tmp_path / ".claude" / "alert_state.json").read_text())
    alert_state["active_alerts"] = [{
        "alert_type": "consecutive_losses",
        "severity": "WARNING",
        "message": "3 consecutive losses",
        "timestamp": FIXED_NOW.isoformat(),
        "value": 3.0, "threshold": 3.0, "pair": "", "acknowledged": False,
    }]
    (tmp_path / ".claude" / "alert_state.json").write_text(json.dumps(alert_state))
    # Mark a previous watch for this alert key 10 min ago.
    state_path = tmp_path / ".claude" / "hermes_state.json"
    state_path.write_text(json.dumps({
        "last_alert_keys": {
            "consecutive_losses": (FIXED_NOW - timedelta(minutes=10)).isoformat(),
        },
    }))
    result = run_once(_ctx(tmp_path))
    assert "watch" not in result.entries_written


def test_heartbeat_stale_writes_watch(tmp_path: Path):
    _seed_clean_state(tmp_path)
    hb = json.loads((tmp_path / ".claude" / "heartbeat.json").read_text())
    hb["ts_iso"] = (FIXED_NOW - timedelta(seconds=90)).isoformat()  # > 60s old
    (tmp_path / ".claude" / "heartbeat.json").write_text(json.dumps(hb))
    result = run_once(_ctx(tmp_path))
    assert "watch" in result.entries_written
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "heartbeat_stale" in digest


def test_job_failure_writes_watch_excluding_self(tmp_path: Path):
    """A failing job triggers an alert — but the watchdog must skip itself."""
    _seed_clean_state(tmp_path)
    (tmp_path / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({
        "nightly_audit": {
            "state": "active",
            "last_status": "failure",
            "last_error": "exit 1: disk full at /tmp",
            "last_status_at": FIXED_NOW.isoformat(),
            "run_count": 5,
        },
        "hermes_watchdog": {  # OUR job; must be skipped
            "state": "active",
            "last_status": "failure",
            "last_error": "recursive failure",
            "last_status_at": FIXED_NOW.isoformat(),
            "run_count": 1,
        },
    }))
    result = run_once(_ctx(tmp_path))
    assert "watch" in result.entries_written
    digest = (tmp_path / ".claude" / "brain" / "hermes_watchdog.md").read_text()
    assert "nightly_audit" in digest
    # The watchdog must NEVER alert on its own failure (would be recursive).
    last_entry_block = digest.split("### ")[-1] if "### " in digest else ""
    assert "hermes_watchdog" not in last_entry_block


def test_paused_failing_job_does_not_alert(tmp_path: Path):
    _seed_clean_state(tmp_path)
    (tmp_path / "trained_data" / "jobs_runtime_state.json").write_text(json.dumps({
        "nightly_audit": {
            "state": "paused",  # paused jobs are not actively failing
            "last_status": "failure",
            "last_error": "old error",
            "last_status_at": (FIXED_NOW - timedelta(hours=24)).isoformat(),
            "run_count": 5,
        },
    }))
    result = run_once(_ctx(tmp_path))
    assert "watch" not in result.entries_written


def test_corrupt_alert_state_does_not_crash(tmp_path: Path):
    _seed_clean_state(tmp_path)
    (tmp_path / ".claude" / "alert_state.json").write_text("not valid json {{{")
    # Should not raise. Should still complete (silent or no-op).
    result = run_once(_ctx(tmp_path))
    assert result is not None
