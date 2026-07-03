"""Autonomy-layer fix (2026-07-03) — no-mock tests on real disk.

Covers the three operator-reported autonomy failures:
1. code pickup    — CodeFreshness detects git-HEAD / watched-file changes (debounced)
2. honest beacon  — supervisor heartbeat says scanner_alive=False + writer keys;
                    a fresh foreign (TUI) heartbeat is never clobbered
3. diagnostics    — maintenance report counts adjustment backlog + unacked alerts
                    and lands in tier7_state for AXIOM

No unittest.mock / no patching (repo No-Mock rule): real tmp git repos, real JSON
files under tmp_path, real function calls.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scanner.automation.code_freshness import CodeFreshness  # noqa: E402
from src.scanner.automation.maintenance_report import (  # noqa: E402
    build_maintenance_report,
    write_maintenance_report,
)
from src.scanner.automation.tier7_state import build_tier7_state  # noqa: E402
from src.tui.heartbeat import read_heartbeat, write_heartbeat  # noqa: E402

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=_GIT_ENV,
                   capture_output=True)


def _make_git_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "f.txt").write_text("v1\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


# ---------------------------------------------------------------------------
# 1. CodeFreshness
# ---------------------------------------------------------------------------

def test_freshness_no_change_is_none(tmp_path: Path) -> None:
    repo = _make_git_repo(tmp_path / "r")
    fresh = CodeFreshness(repo)
    assert fresh.changed() is None
    assert fresh.changed() is None


def test_freshness_detects_new_commit_after_debounce(tmp_path: Path) -> None:
    repo = _make_git_repo(tmp_path / "r")
    fresh = CodeFreshness(repo)
    (repo / "f.txt").write_text("v2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change")
    # Debounce: first poll arms, second poll fires with the reason.
    assert fresh.changed() is None
    reason = fresh.changed()
    assert reason is not None and "git HEAD moved" in reason


def test_freshness_detects_watched_file_mtime(tmp_path: Path) -> None:
    repo = _make_git_repo(tmp_path / "r")
    watched = repo / "daemon.py"
    watched.write_text("print('v1')\n")
    fresh = CodeFreshness(repo, watch_files=[watched])
    time.sleep(0.02)
    watched.write_text("print('v2')\n")
    os.utime(watched, (time.time(), time.time() + 1))  # force a distinct mtime
    assert fresh.changed() is None   # first sighting arms
    reason = fresh.changed()
    assert reason is not None and "watched file changed" in reason


def test_freshness_no_git_no_crash_and_flapping_change_never_fires(tmp_path: Path) -> None:
    plain = tmp_path / "nogit"
    plain.mkdir()
    watched = plain / "w.py"
    watched.write_text("a\n")
    fresh = CodeFreshness(plain, watch_files=[watched])
    assert fresh.baseline_head is None          # no git => signal skipped, no crash
    assert fresh.changed() is None
    # A change that REVERTS before the confirming poll must not fire (debounce).
    os.utime(watched, (time.time(), time.time() + 5))
    assert fresh.changed() is None              # armed
    os.utime(watched, fresh.baseline_mtimes[0][1:] * 2)  # restore baseline mtime
    assert fresh.changed() is None              # reverted -> disarmed


def test_freshness_missing_watch_file_is_skipped(tmp_path: Path) -> None:
    repo = _make_git_repo(tmp_path / "r")
    fresh = CodeFreshness(repo, watch_files=[repo / "does_not_exist.py"])
    assert fresh.baseline_mtimes == ()
    assert fresh.changed() is None


# ---------------------------------------------------------------------------
# 2. Honest heartbeat
# ---------------------------------------------------------------------------

def test_heartbeat_extra_keys_are_additive_and_cannot_override_core(tmp_path: Path) -> None:
    write_heartbeat(
        tmp_path, cycle_count=7, mode="live", scanner_alive=False, pid=os.getpid(),
        extra={"writer": "tier7_supervisor", "supervisor_alive": True,
               "scanner_alive": True},  # attempted core override must be ignored
    )
    hb = read_heartbeat(tmp_path)
    assert hb is not None
    assert hb["scanner_alive"] is False          # core schema protected
    assert hb["writer"] == "tier7_supervisor"    # additive key landed
    assert hb["supervisor_alive"] is True
    assert hb["cycle_count"] == 7 and hb["pid"] == os.getpid()


def _load_supervisor_module():
    spec = importlib.util.spec_from_file_location(
        "run_tier7_loop_under_test", REPO_ROOT / "scripts" / "run_tier7_loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_foreign_fresh_heartbeat_is_not_clobbered(tmp_path: Path) -> None:
    sup = _load_supervisor_module()
    foreign_pid = os.getppid()  # alive and != this test process
    assert foreign_pid != os.getpid()
    write_heartbeat(tmp_path, cycle_count=1, mode="live", scanner_alive=True,
                    pid=foreign_pid)
    assert sup._foreign_writer_owns_heartbeat(tmp_path) is True

    # Own pid => not foreign; supervisor may write.
    write_heartbeat(tmp_path, cycle_count=2, mode="live", scanner_alive=True,
                    pid=os.getpid())
    assert sup._foreign_writer_owns_heartbeat(tmp_path) is False


def test_stale_foreign_heartbeat_releases_ownership(tmp_path: Path) -> None:
    sup = _load_supervisor_module()
    hb_dir = tmp_path / ".claude"
    hb_dir.mkdir(parents=True, exist_ok=True)
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    (hb_dir / "heartbeat.json").write_text(json.dumps({
        "ts_iso": stale_ts, "pid": os.getppid(), "cycle_count": 1,
        "mode": "live", "scanner_alive": True, "last_error_ts": None,
    }))
    assert sup._foreign_writer_owns_heartbeat(tmp_path) is False

    (hb_dir / "heartbeat.json").write_text("{not json")
    assert sup._foreign_writer_owns_heartbeat(tmp_path) is False  # corrupt => write honestly


# ---------------------------------------------------------------------------
# 3. Maintenance report
# ---------------------------------------------------------------------------

def _seed_maintenance_fixtures(root: Path) -> None:
    claude = root / ".claude"
    claude.mkdir(parents=True, exist_ok=True)
    (claude / "config_adjustments.json").write_text(json.dumps({
        "version": 2,
        "history": [
            {"proposal_id": "aaa", "key": "min_confidence", "new_value": 0.5,
             "timestamp": "2026-07-01T00:00:00+00:00"},
            {"proposal_id": "bbb", "key": "atr_sl_multiplier", "new_value": 1.4,
             "timestamp": "2026-07-02T00:00:00+00:00"},
        ],
        "applied_ids": ["aaa"],
    }))
    (claude / "pending_adjustments.json").write_text(json.dumps({
        "proposals": [
            {"id": "p1", "status": "pending", "key": "k1",
             "timestamp": "2026-07-02T01:00:00+00:00"},
            {"id": "p2", "status": "invalid", "key": "nope",
             "timestamp": "2026-07-02T02:00:00+00:00"},
        ],
    }))
    (claude / "alert_state.json").write_text(json.dumps({
        "active_alerts": [
            {"alert_type": "consecutive_losses", "severity": "WARNING",
             "message": "7 consecutive losses (threshold: 3)",
             "timestamp": "2026-07-02T12:58:07", "acknowledged": False},
            {"alert_type": "drawdown", "severity": "WARNING",
             "message": "acked", "timestamp": "2026-07-01T00:00:00",
             "acknowledged": True},
        ],
    }))
    (claude / "state.json").write_text(json.dumps({
        "halted": True,
        "halted_lanes": {"oanda_fx": True, "equity": True, "brain": True,
                         "crypto_momentum": True, "track_b": True},
    }))
    (claude / "self_heal_action_budget.json").write_text(json.dumps({
        "actions_today": ["2026-07-02T09:40:19+00:00", "2026-07-02T12:59:33+00:00"],
        "last_action_at": "2026-07-02T12:59:33+00:00",
        "last_action_label": "reduce_risk_per_trade_pct",
    }))


def test_maintenance_report_counts_backlogs_and_alerts(tmp_path: Path) -> None:
    _seed_maintenance_fixtures(tmp_path)
    report = build_maintenance_report(tmp_path, supervisor_pid=os.getpid())

    adj = report["adjustments"]
    assert adj["approved_unconsumed"] == 1               # bbb not in applied_ids
    assert adj["approved_unconsumed_keys"] == ["atr_sl_multiplier"]
    assert adj["pending_proposals"] == 1 and adj["invalid_proposals"] == 1

    alerts = report["alerts"]
    assert alerts["unacknowledged"] == 1                 # acked one excluded
    assert alerts["active"][0]["type"] == "consecutive_losses"

    lanes = report["lanes"]
    assert lanes["global_halted"] is True
    assert lanes["halted_lanes"]["oanda_fx"] is True
    assert lanes["trend_artifact_stale"] is True         # artifact absent => stale

    sh = report["self_heal"]
    assert sh["actions_today"] == 2
    assert sh["last_action_label"] == "reduce_risk_per_trade_pct"

    assert len(report["needs_attention"]) == 3           # unconsumed + pending + alert


def test_maintenance_report_degrades_gracefully_on_empty_root(tmp_path: Path) -> None:
    report = build_maintenance_report(tmp_path)          # nothing on disk at all
    assert report["adjustments"]["approved_unconsumed"] == 0
    assert report["alerts"]["unacknowledged"] == 0
    assert report["lanes"]["global_halted"] is True      # missing state => halted (fail-closed)
    assert report["self_heal"]["available"] is False
    assert report["needs_attention"] == []


def test_maintenance_report_corrupt_sources_degrade_per_section(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    claude.mkdir(parents=True)
    (claude / "alert_state.json").write_text("{corrupt")
    (claude / "config_adjustments.json").write_text("[]")
    report = build_maintenance_report(tmp_path)
    assert report["alerts"]["unacknowledged"] == 0       # corrupt json -> empty, not raise
    assert report["adjustments"]["approved_unconsumed"] == 0


def test_maintenance_report_lands_in_tier7_state(tmp_path: Path) -> None:
    _seed_maintenance_fixtures(tmp_path)
    report = build_maintenance_report(tmp_path, supervisor_pid=os.getpid())
    path = write_maintenance_report(tmp_path, report)
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["writer"] == "tier7_supervisor"

    snap = build_tier7_state(tmp_path)
    assert snap["maintenance"]["adjustments"]["approved_unconsumed"] == 1
    assert snap["maintenance"]["alerts"]["unacknowledged"] == 1


def test_tier7_state_maintenance_none_when_absent(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)
    snap = build_tier7_state(tmp_path)
    assert snap["maintenance"] is None


# ---------------------------------------------------------------------------
# 4. Trend-runner fast code pickup (verifier finding: poll inside the sleep)
# ---------------------------------------------------------------------------

def test_trend_sleep_wakes_early_on_stop_check() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_oanda_trend_under_test", REPO_ROOT / "scripts" / "run_oanda_trend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    start = time.monotonic()
    mod._sleep_until_next_cycle(30.0, poll_s=0.05, stop_check=lambda: True)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0        # woke on the first poll, not the 30s cadence

    # Without a stop_check the short sleep still completes normally.
    start = time.monotonic()
    mod._sleep_until_next_cycle(0.1, poll_s=0.05)
    assert time.monotonic() - start < 2.0
