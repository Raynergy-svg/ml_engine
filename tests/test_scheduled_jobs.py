"""Tests for src/scanner/automation/scheduled_jobs.py (pick #5)."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scanner.automation.scheduled_jobs import (
    JobConfig,
    JobRuntimeState,
    ScheduledJobsRegistry,
    _parse_iso,
    compute_next_run,
    default_jobs,
)


UTC = timezone.utc


# ────────────────────────── Schedule parsing ──────────────────────────


class TestComputeNextRun:

    def test_daily_advances_to_next_day_when_slot_already_past(self):
        now = datetime(2026, 5, 12, 14, 30, tzinfo=UTC)
        nxt = compute_next_run("daily_10:00", now=now)
        assert nxt == datetime(2026, 5, 13, 10, 0, tzinfo=UTC)

    def test_daily_fires_today_if_slot_in_future(self):
        now = datetime(2026, 5, 12, 9, 0, tzinfo=UTC)
        nxt = compute_next_run("daily_10:00", now=now)
        assert nxt == datetime(2026, 5, 12, 10, 0, tzinfo=UTC)

    def test_daily_anchors_off_last_run_when_provided(self):
        last = datetime(2026, 5, 12, 10, 0, tzinfo=UTC)
        now = datetime(2026, 5, 12, 11, 0, tzinfo=UTC)
        nxt = compute_next_run("daily_10:00", last_run=last, now=now)
        # Next slot is tomorrow at 10:00, not today (we just ran).
        assert nxt == datetime(2026, 5, 13, 10, 0, tzinfo=UTC)

    def test_weekly_next_dow(self):
        # 2026-05-12 is a Tuesday (weekday=1). Next SUN (weekday=6) is 2026-05-17.
        now = datetime(2026, 5, 12, 12, 0, tzinfo=UTC)
        nxt = compute_next_run("weekly_SUN_02:00", now=now)
        assert nxt == datetime(2026, 5, 17, 2, 0, tzinfo=UTC)

    def test_weekly_advances_seven_days_when_same_dow_slot_past(self):
        # Sunday 03:00 — past Sunday 02:00 slot → next Sunday.
        now = datetime(2026, 5, 17, 3, 0, tzinfo=UTC)
        nxt = compute_next_run("weekly_SUN_02:00", now=now)
        assert nxt == datetime(2026, 5, 24, 2, 0, tzinfo=UTC)

    def test_every_n_minutes(self):
        now = datetime(2026, 5, 12, 10, 0, tzinfo=UTC)
        assert compute_next_run("every_30_minutes", now=now) == now + timedelta(minutes=30)

    def test_every_n_hours_anchored_to_last_run(self):
        last = datetime(2026, 5, 12, 10, 0, tzinfo=UTC)
        now = datetime(2026, 5, 12, 14, 30, tzinfo=UTC)
        assert compute_next_run("every_6_hours", last_run=last, now=now) == datetime(
            2026, 5, 12, 16, 0, tzinfo=UTC
        )

    def test_naive_datetime_is_assumed_utc(self):
        # Naive `now` shouldn't blow up; comparisons happen in UTC.
        now = datetime(2026, 5, 12, 14, 30)  # naive
        nxt = compute_next_run("daily_10:00", now=now)
        assert nxt.tzinfo == UTC

    @pytest.mark.parametrize("bad", [
        "daily_25:00",            # bad hour
        "daily_10:99",            # bad minute
        "weekly_XYZ_02:00",       # bad DOW
        "every_0_minutes",        # N < 1
        "every_0_hours",
        "every_minute",           # missing N
        "hourly_30",              # unsupported keyword
        "",                       # empty
    ])
    def test_invalid_schedules_raise(self, bad):
        with pytest.raises(ValueError):
            compute_next_run(bad)


# ────────────────────────── Helpers ──────────────────────────


class TestParseIso:

    def test_none_returns_none(self):
        assert _parse_iso(None) is None
        assert _parse_iso("") is None

    def test_garbage_returns_none(self):
        assert _parse_iso("not a date") is None

    def test_naive_iso_assumed_utc(self):
        dt = _parse_iso("2026-05-12T10:00:00")
        assert dt is not None and dt.tzinfo == UTC

    def test_aware_iso_preserved(self):
        dt = _parse_iso("2026-05-12T10:00:00+00:00")
        assert dt == datetime(2026, 5, 12, 10, 0, tzinfo=UTC)


# ────────────────────────── Registry ──────────────────────────


def _mk_job(**over) -> JobConfig:
    return JobConfig(
        job_id=over.get("job_id", "j1"),
        name=over.get("name", "test"),
        schedule=over.get("schedule", "daily_10:00"),
        command=over.get("command", "echo ok"),
        enabled=over.get("enabled", True),
        description=over.get("description", ""),
    )


@pytest.fixture
def tmp_paths(tmp_path):
    cfg = tmp_path / "claude" / "jobs.json"
    state = tmp_path / "trained" / "jobs_state.json"
    cfg.parent.mkdir(parents=True)
    state.parent.mkdir(parents=True)
    return cfg, state


@pytest.fixture
def fake_executor():
    """Records calls but never spawns a real subprocess."""
    calls: list[str] = []

    def _executor(cmd, *, cwd):
        calls.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = MagicMock(return_value=(b"", b""))
        return proc

    _executor.calls = calls
    return _executor


class TestRegistryLoad:

    def test_missing_config_falls_back_to_defaults(self, tmp_paths, fake_executor):
        cfg, state = tmp_paths
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        assert {j.job_id for j in r.jobs()} == {j.job_id for j in default_jobs()}
        # Defaults are disabled — operator must opt in.
        assert all(j.enabled is False for j in r.jobs())

    def test_loads_from_config_file(self, tmp_paths, fake_executor):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({
            "jobs": [
                {"job_id": "x", "name": "X", "schedule": "daily_10:00", "command": "echo x", "enabled": True},
                {"job_id": "y", "name": "Y", "schedule": "every_30_minutes", "command": "echo y"},
            ]
        }), encoding="utf-8")
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        ids = {j.job_id for j in r.jobs()}
        assert ids == {"x", "y"}

    def test_malformed_config_falls_back_to_defaults(self, tmp_paths, fake_executor):
        cfg, state = tmp_paths
        cfg.write_text('{"not_jobs": []}', encoding="utf-8")
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        # Defaults kicked in:
        assert {j.job_id for j in r.jobs()} == {j.job_id for j in default_jobs()}

    def test_individual_malformed_entries_skipped(self, tmp_paths, fake_executor):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({
            "jobs": [
                {"job_id": "ok", "schedule": "daily_10:00", "command": "echo ok"},
                {"name": "no_id"},   # missing job_id
                "not a dict",
            ]
        }), encoding="utf-8")
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        assert [j.job_id for j in r.jobs()] == ["ok"]


class TestDueJobs:

    def test_disabled_jobs_never_due(self, tmp_paths, fake_executor):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({"jobs": [{
            "job_id": "x", "schedule": "every_1_minutes", "command": "echo x", "enabled": False,
        }]}), encoding="utf-8")
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        # Never-run, every_1_minutes, way in the future "now" — still disabled.
        future = datetime(2030, 1, 1, tzinfo=UTC)
        assert r.due_jobs(now=future) == []

    def test_unrun_enabled_job_with_past_slot_is_due(self, tmp_paths, fake_executor):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({"jobs": [{
            "job_id": "x", "schedule": "daily_10:00", "command": "echo x", "enabled": True,
        }]}), encoding="utf-8")
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        # 14:00 same day, never run — last_run anchors to now, schedule advances → tomorrow 10:00.
        # Not yet due, since compute_next_run anchors to `now` when last_run is missing.
        now = datetime(2026, 5, 12, 14, 0, tzinfo=UTC)
        assert r.due_jobs(now=now) == []

        # But run it once at 09:00, then ask at 11:00 → due (today 10:00 already passed).
        r._state["x"] = JobRuntimeState(last_run_at="2026-05-11T10:00:00+00:00")
        now = datetime(2026, 5, 12, 11, 0, tzinfo=UTC)
        due = r.due_jobs(now=now)
        assert [j.job_id for j in due] == ["x"]

    def test_unparseable_schedule_skipped(self, tmp_paths, fake_executor, caplog):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({"jobs": [{
            "job_id": "bad", "schedule": "nonsense", "command": "echo bad", "enabled": True,
        }]}), encoding="utf-8")
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        assert r.due_jobs(now=datetime(2030, 1, 1, tzinfo=UTC)) == []


class TestTick:

    def test_tick_fires_due_job_and_persists_state(self, tmp_paths, fake_executor):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({"jobs": [{
            "job_id": "x", "schedule": "every_1_minutes", "command": "echo hi", "enabled": True,
        }]}), encoding="utf-8")
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        # Prime last_run so the job is overdue.
        r._state["x"] = JobRuntimeState(last_run_at="2020-01-01T00:00:00+00:00")

        fired = r.tick(now=datetime(2030, 1, 1, tzinfo=UTC))
        assert fired == ["x"]

        # Wait briefly for the daemon thread to finish.
        _wait_for(lambda: r.state("x").run_count == 1, timeout=2.0)

        assert fake_executor.calls == ["echo hi"]
        s = r.state("x")
        assert s.last_status == "success"
        assert s.run_count == 1
        # State file was written:
        on_disk = json.loads(state.read_text(encoding="utf-8"))
        assert on_disk["x"]["last_status"] == "success"

    def test_tick_records_failure_when_command_exits_nonzero(self, tmp_paths):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({"jobs": [{
            "job_id": "x", "schedule": "every_1_minutes", "command": "echo no", "enabled": True,
        }]}), encoding="utf-8")

        def fail_exec(cmd, *, cwd):
            proc = MagicMock()
            proc.returncode = 2
            proc.communicate = MagicMock(return_value=(b"", b"boom\n"))
            return proc

        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fail_exec)
        r.load()
        r._state["x"] = JobRuntimeState(last_run_at="2020-01-01T00:00:00+00:00")

        r.tick(now=datetime(2030, 1, 1, tzinfo=UTC))
        _wait_for(lambda: r.state("x").run_count == 1, timeout=2.0)

        s = r.state("x")
        assert s.last_status == "failure"
        assert s.last_error is not None and "boom" in s.last_error

    def test_tick_catches_executor_exceptions(self, tmp_paths):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({"jobs": [{
            "job_id": "x", "schedule": "every_1_minutes", "command": "echo no", "enabled": True,
        }]}), encoding="utf-8")

        def broken(cmd, *, cwd):
            raise OSError("permission denied")

        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=broken)
        r.load()
        r._state["x"] = JobRuntimeState(last_run_at="2020-01-01T00:00:00+00:00")

        r.tick(now=datetime(2030, 1, 1, tzinfo=UTC))
        _wait_for(lambda: r.state("x").run_count == 1, timeout=2.0)

        s = r.state("x")
        assert s.last_status == "failure"
        assert "permission denied" in (s.last_error or "")

    def test_tick_with_no_due_jobs_is_noop(self, tmp_paths, fake_executor):
        cfg, state = tmp_paths
        cfg.write_text(json.dumps({"jobs": []}), encoding="utf-8")
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_executor)
        r.load()
        assert r.tick(now=datetime(2030, 1, 1, tzinfo=UTC)) == []
        assert fake_executor.calls == []


def _wait_for(predicate, *, timeout: float) -> None:
    """Spin up to `timeout` seconds waiting for `predicate()` → truthy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for predicate")
