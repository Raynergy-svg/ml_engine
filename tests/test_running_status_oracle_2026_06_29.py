"""running_status.py oracle — live-lane detection logic (no mocks, real files/pids).

Locks the fix: the oracle must report the LIVE lane (OANDA trend + Tier 7), not only the
dormant harvester. Tests the pure freshness/pid helpers + that live_lane_running returns
the labelled shape. Imports the oracle by path (.claude/loop is not on sys.path).
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "running_status", _REPO / ".claude" / "loop" / "running_status.py")
rs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rs)


def test_file_fresh(tmp_path, monkeypatch):
    f = tmp_path / "a.json"
    f.write_text("{}")
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    fresh, age = rs._file_fresh("a.json", 7200)
    assert fresh is True and age is not None and age < 60
    # age it past the window
    old = time.time() - 10000
    os.utime(f, (old, old))
    fresh2, _ = rs._file_fresh("a.json", 7200)
    assert fresh2 is False
    # missing file -> not fresh, no crash
    assert rs._file_fresh("missing.json", 7200) == (False, None)


def test_pid_alive():
    assert rs._pid_alive(os.getpid()) is True       # our own pid is alive
    assert rs._pid_alive(2_000_000_000) is False     # implausible pid -> dead
    assert rs._pid_alive(0) is False
    assert rs._pid_alive("bad") is False


def test_tier7_heartbeat_fresh_vs_stale(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timedelta, timezone
    cl = tmp_path / ".claude"
    cl.mkdir()
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    now = datetime.now(timezone.utc)
    # fresh beacon with our live pid
    (cl / "heartbeat.json").write_text(json.dumps(
        {"ts_iso": now.isoformat(), "pid": os.getpid()}))
    fresh, age, pid_alive = rs._tier7_heartbeat()
    assert fresh is True and pid_alive is True and age < 90
    # stale beacon -> not fresh
    (cl / "heartbeat.json").write_text(json.dumps(
        {"ts_iso": (now - timedelta(hours=4)).isoformat(), "pid": os.getpid()}))
    fresh2, _, _ = rs._tier7_heartbeat()
    assert fresh2 is False


def test_live_lane_running_shape():
    d = rs.live_lane_running()
    assert "running" in d and isinstance(d["running"], bool)
    for k in ("oanda_trend_proc", "account_state_fresh", "tier7_heartbeat_fresh", "tier7_pid_alive"):
        assert k in d
