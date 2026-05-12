"""Tests for scheduled_jobs observability additions (Tier 1 T1)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.scanner.automation.scheduled_jobs import (
    JobConfig,
    JobRuntimeState,
    ScheduledJobsRegistry,
    compute_next_run,
)

UTC = timezone.utc


class TestJobRuntimeStateExtensions:
    def test_state_field_defaults_to_active(self):
        s = JobRuntimeState()
        assert s.state == "active"

    def test_state_field_round_trips_paused(self):
        s = JobRuntimeState(state="paused")
        d = s.to_dict()
        s2 = JobRuntimeState.from_dict(d)
        assert s2.state == "paused"

    def test_state_field_round_trips_active(self):
        s = JobRuntimeState(state="active")
        s2 = JobRuntimeState.from_dict(s.to_dict())
        assert s2.state == "active"

    def test_next_run_at_iso_field_is_optional(self):
        s = JobRuntimeState()
        assert s.next_run_at_iso is None

    def test_last_status_at_field_is_optional(self):
        s = JobRuntimeState()
        assert s.last_status_at is None

    def test_legacy_state_dict_loads_with_defaults(self):
        legacy = {"last_run_at": "2026-05-01T00:00:00+00:00", "last_status": "success", "run_count": 5}
        s = JobRuntimeState.from_dict(legacy)
        assert s.state == "active"
        assert s.next_run_at_iso is None
        assert s.last_status_at is None
        assert s.run_count == 5


class TestPauseResume:
    def test_pause_marks_state_paused(self, tmp_path: Path):
        cfg = tmp_path / "jobs.json"
        state = tmp_path / "state.json"
        cfg.write_text(json.dumps({"jobs": [
            {"job_id": "j1", "name": "j1", "schedule": "every_5_minutes",
             "command": "echo hi", "enabled": True}
        ]}))
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state)
        r.load()
        r.pause_job("j1")
        assert r.state("j1").state == "paused"

    def test_paused_job_not_in_due_jobs(self, tmp_path: Path):
        cfg = tmp_path / "jobs.json"
        state = tmp_path / "state.json"
        cfg.write_text(json.dumps({"jobs": [
            {"job_id": "j1", "name": "j1", "schedule": "every_1_minutes",
             "command": "echo hi", "enabled": True}
        ]}))
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state)
        r.load()
        r.pause_job("j1")
        future = datetime.now(UTC) + timedelta(hours=1)
        assert r.due_jobs(now=future) == []

    def test_resume_makes_job_due_again(self, tmp_path: Path):
        cfg = tmp_path / "jobs.json"
        state = tmp_path / "state.json"
        cfg.write_text(json.dumps({"jobs": [
            {"job_id": "j1", "name": "j1", "schedule": "every_1_minutes",
             "command": "echo hi", "enabled": True}
        ]}))
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state)
        r.load()
        # Seed last_run_at well in the past so the every_1_minutes schedule is due.
        r._state["j1"] = JobRuntimeState(last_run_at="2020-01-01T00:00:00+00:00")
        r.pause_job("j1")
        # Even with old last_run, paused job is not due.
        assert r.due_jobs() == []
        r.resume_job("j1")
        future = datetime.now(UTC) + timedelta(hours=1)
        due = r.due_jobs(now=future)
        assert any(j.job_id == "j1" for j in due)


class TestTriggerNow:
    def test_trigger_now_runs_job_synchronously(self, tmp_path: Path):
        cfg = tmp_path / "jobs.json"
        state = tmp_path / "state.json"
        ran = []

        def fake_runner(cmd, *, cwd):
            class P:
                returncode = 0
                def communicate(self): return (b"", b"")
            ran.append(cmd)
            return P()

        cfg.write_text(json.dumps({"jobs": [
            {"job_id": "j1", "name": "j1", "schedule": "daily_03:00",
             "command": "echo hi", "enabled": False}
        ]}))
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_runner)
        r.load()
        r.trigger_now("j1")
        time.sleep(0.5)
        assert ran == ["echo hi"]
        assert r.state("j1").last_status == "success"

    def test_trigger_now_records_last_status_at(self, tmp_path: Path):
        cfg = tmp_path / "jobs.json"
        state = tmp_path / "state.json"

        def fake_runner(cmd, *, cwd):
            class P:
                returncode = 0
                def communicate(self): return (b"", b"")
            return P()

        cfg.write_text(json.dumps({"jobs": [
            {"job_id": "j1", "name": "j1", "schedule": "daily_03:00",
             "command": "echo hi", "enabled": False}
        ]}))
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_runner)
        r.load()
        # last_status_at is rounded to whole seconds via .replace(microsecond=0),
        # so subtract one second when comparing.
        before = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=1)
        r.trigger_now("j1")
        time.sleep(0.5)
        s = r.state("j1")
        assert s.last_status_at is not None
        ts = datetime.fromisoformat(s.last_status_at)
        assert ts >= before


class TestNextRunAtIso:
    def test_tick_persists_next_run_at_iso(self, tmp_path: Path):
        cfg = tmp_path / "jobs.json"
        state = tmp_path / "state.json"

        def fake_runner(cmd, *, cwd):
            class P:
                returncode = 0
                def communicate(self): return (b"", b"")
            return P()

        cfg.write_text(json.dumps({"jobs": [
            {"job_id": "j1", "name": "j1", "schedule": "every_30_minutes",
             "command": "echo hi", "enabled": True}
        ]}))
        r = ScheduledJobsRegistry(config_path=cfg, state_path=state, executor=fake_runner)
        r.load()
        # Seed last_run_at well in the past so the every_30_minutes schedule is due.
        r._state["j1"] = JobRuntimeState(last_run_at="2020-01-01T00:00:00+00:00")
        future = datetime.now(UTC) + timedelta(hours=1)
        r.tick(now=future)
        time.sleep(0.5)
        s = r.state("j1")
        assert s.next_run_at_iso is not None
        nxt = datetime.fromisoformat(s.next_run_at_iso)
        assert nxt > future
