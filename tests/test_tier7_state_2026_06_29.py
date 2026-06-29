"""Tier 7 read-only state exporter — honest running detection (no mocks, real disk).

running:true ONLY if heartbeat is fresh AND pid alive; a stale beacon or dead pid
=> running:false with the reason. Uses real .claude/* files on tmp_path.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from src.scanner.automation.tier7_state import build_tier7_state, write_tier7_state


def _seed(root, *, hb_age_s, pid):
    cl = root / ".claude"
    (cl / "meta").mkdir(parents=True)
    now = datetime.now(timezone.utc)
    ts = (now - timedelta(seconds=hb_age_s)).isoformat()
    (cl / "heartbeat.json").write_text(json.dumps(
        {"pid": pid, "cycle_count": 7, "scanner_alive": True, "ts_iso": ts, "mode": "live"}))
    (cl / "state.json").write_text(json.dumps(
        {"halted": False, "mode": "live", "status": "running", "goal": "harvest",
         "next": "scan", "scan_cycle_count": 42}))
    (cl / "meta" / "changes.jsonl").write_text(
        json.dumps({"change_id": "abc", "stage": "aborted", "event": "duplicate_dropped",
                    "kind": "config", "deploy_target": "shadow", "updated_at": "t"}) + "\n")
    return now


def test_fresh_heartbeat_live_pid_is_running(tmp_path):
    now = _seed(tmp_path, hb_age_s=5, pid=os.getpid())  # our own pid is alive
    s = build_tier7_state(tmp_path, now=now)
    assert s["running"] is True
    assert "fresh" in s["running_reason"]
    assert s["scan_cycle_count"] == 42
    assert s["current_action"] == "scan"
    assert s["meta_last_event"]["event"] == "duplicate_dropped"


def test_stale_heartbeat_is_not_running(tmp_path):
    now = _seed(tmp_path, hb_age_s=4 * 24 * 3600, pid=os.getpid())  # 4-day-old beacon
    s = build_tier7_state(tmp_path, now=now)
    assert s["running"] is False
    assert "stale" in s["running_reason"]


def test_dead_pid_is_not_running(tmp_path):
    now = _seed(tmp_path, hb_age_s=5, pid=2_000_000_000)  # implausible -> dead
    s = build_tier7_state(tmp_path, now=now)
    assert s["running"] is False
    assert "not alive" in s["running_reason"]


def test_write_tier7_state_atomic_and_readonly_note(tmp_path):
    _seed(tmp_path, hb_age_s=5, pid=os.getpid())
    snap = write_tier7_state(tmp_path)
    out = tmp_path / ".claude" / "tier7_state.json"
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == snap
    assert "READ-ONLY" in on_disk["note"]
