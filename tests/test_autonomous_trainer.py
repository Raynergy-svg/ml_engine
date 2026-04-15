"""Tests for autonomous_trainer — proves the retrain spawn fires correctly,
honors cooldown, and handles single-flight + state persistence."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trained_data").mkdir()
    (tmp_path / "scripts").mkdir()
    # Create a stub scheduled_retrain.py so the existence check passes
    (tmp_path / "scripts" / "scheduled_retrain.py").write_text("# stub\n")
    return tmp_path


def _patch_today(monkeypatch, weekday: int, day_iso: str = "2026-04-15"):
    """Force the autonomous_trainer to think today is `weekday` (0=Mon).

    Patches datetime.now() inside the autonomous_trainer module so the
    schedule + 'already fired today' checks are deterministic.
    """
    from datetime import datetime as real_dt, timezone as real_tz
    from src.scanner.automation import autonomous_trainer as at

    class _FixedDateTime(real_dt):
        @classmethod
        def now(cls, tz=None):
            # Build a real datetime with the wanted weekday at noon
            # (start from a known Monday and add weekday days)
            base_mon = real_dt(2026, 4, 13, 12, 0, tzinfo=real_tz.utc)  # Apr 13 2026 = Mon
            from datetime import timedelta as _td
            return base_mon + _td(days=weekday)

    monkeypatch.setattr(at, "datetime", _FixedDateTime)


# ── Decision logic ────────────────────────────────────────────────────


def test_skips_when_status_fresh(iso):
    from src.scanner.automation import autonomous_trainer as at
    fired = []
    result = at.maybe_spawn_autonomous_retrain(
        freshness={"status": "FRESH", "oldest_age_days": 2.0},
        brain_callback=fired.append,
    )
    assert result["spawned"] is False
    assert "FRESH" in (result["skipped_reason"] or "")


def test_skips_when_status_aging(iso):
    from src.scanner.automation import autonomous_trainer as at
    fired = []
    result = at.maybe_spawn_autonomous_retrain(
        freshness={"status": "AGING", "oldest_age_days": 10.0},
        brain_callback=fired.append,
    )
    assert result["spawned"] is False


def test_stale_now_fires_by_default(iso, monkeypatch):
    """STALE auto-fires now (forex weekly cadence). Opt out via env."""
    monkeypatch.delenv("BUDDY_DISABLE_AUTOTRAIN_ON_STALE", raising=False)
    # Pin to a non-train day so schedule trigger doesn't interfere
    _patch_today(monkeypatch, weekday=2)  # Wednesday
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    fake_proc = MagicMock(pid=42)
    fake_proc.poll.return_value = None
    with patch.object(at.subprocess, "Popen", return_value=fake_proc):
        result = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "STALE", "oldest_age_days": 6.0},
            brain_callback=lambda m: None,
        )
    assert result["spawned"] is True


def test_stale_skipped_when_env_disabled(iso, monkeypatch):
    """Opt-out env disables STALE auto-fire (still fires on CRITICAL)."""
    monkeypatch.setenv("BUDDY_DISABLE_AUTOTRAIN_ON_STALE", "1")
    _patch_today(monkeypatch, weekday=2)  # Wednesday
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    fired = []
    result = at.maybe_spawn_autonomous_retrain(
        freshness={"status": "STALE", "oldest_age_days": 6.0},
        brain_callback=fired.append,
    )
    assert result["spawned"] is False
    assert "STALE" in (result["skipped_reason"] or "")


def test_fires_when_status_critical(iso, monkeypatch):
    """CRITICAL >30d ALWAYS triggers retrain."""
    from src.scanner.automation import autonomous_trainer as at
    # Reset any module-level state from prior tests
    at._active_retrain_process = None
    fired = []
    fake_proc = MagicMock(pid=99999)
    fake_proc.poll.return_value = None  # still running
    with patch.object(at.subprocess, "Popen", return_value=fake_proc) as mock_popen:
        result = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "CRITICAL", "oldest_age_days": 45.0},
            brain_callback=fired.append,
        )
    assert result["spawned"] is True, f"expected spawn, got {result}"
    assert result["pid"] == 99999
    mock_popen.assert_called_once()
    # Verify the command is correct
    cmd = mock_popen.call_args[0][0]
    assert "scheduled_retrain.py" in str(cmd[1])
    assert "--pairs" in cmd
    assert any("AUTONOMOUS RETRAIN STARTED" in m for m in fired)


def test_fires_when_status_stale_with_env_optin(iso, monkeypatch):
    monkeypatch.setenv("BUDDY_AUTOTRAIN_ON_STALE", "1")
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    fake_proc = MagicMock(pid=12345)
    fake_proc.poll.return_value = None
    with patch.object(at.subprocess, "Popen", return_value=fake_proc):
        result = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "STALE", "oldest_age_days": 20.0},
            brain_callback=lambda m: None,
        )
    assert result["spawned"] is True


# ── Cooldown ───────────────────────────────────────────────────────────


def test_cooldown_skips_recent_retrain(iso):
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None

    # Pre-populate state to look like a retrain finished 1h ago
    state = at.RetrainState(
        last_started_at=time.time() - 3600,  # 1h ago
        last_completed_at=time.time() - 3000,
        last_pid=12345,
        last_pairs="EUR_USD",
        last_status="success",
    )
    at._save_state(state)

    fired = []
    result = at.maybe_spawn_autonomous_retrain(
        freshness={"status": "CRITICAL", "oldest_age_days": 45.0},
        brain_callback=fired.append,
        cooldown_seconds=24 * 3600,  # 24h cooldown
    )
    assert result["spawned"] is False
    assert "cooldown" in (result["skipped_reason"] or "").lower()


def test_force_bypasses_cooldown(iso):
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    state = at.RetrainState(last_started_at=time.time() - 60)
    at._save_state(state)

    fake_proc = MagicMock(pid=88888)
    fake_proc.poll.return_value = None
    with patch.object(at.subprocess, "Popen", return_value=fake_proc):
        result = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "FRESH", "oldest_age_days": 1},
            brain_callback=lambda m: None,
            force=True,
        )
    assert result["spawned"] is True


# ── Single-flight ──────────────────────────────────────────────────────


def test_skips_when_already_running(iso):
    from src.scanner.automation import autonomous_trainer as at
    # Simulate an in-process retrain
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # Still alive
    at._active_retrain_process = fake_proc

    fired = []
    result = at.maybe_spawn_autonomous_retrain(
        freshness={"status": "CRITICAL", "oldest_age_days": 60},
        brain_callback=fired.append,
    )
    assert result["spawned"] is False
    assert "already running" in (result["skipped_reason"] or "").lower()
    at._active_retrain_process = None  # cleanup


# ── Completion polling ────────────────────────────────────────────────


def test_poll_completion_detects_success(iso):
    from src.scanner.automation import autonomous_trainer as at
    fake_proc = MagicMock(pid=1234)
    fake_proc.poll.return_value = 0  # exit 0 = success
    at._active_retrain_process = fake_proc

    state = at.RetrainState(last_started_at=time.time() - 600, last_status="running")
    at._save_state(state)

    fired = []
    result = at.poll_completion(fired.append)
    assert result == "success"
    assert any("SUCCESS" in m.upper() for m in fired)
    new_state = at._load_state()
    assert new_state.last_status == "success"


def test_poll_completion_detects_failure(iso):
    from src.scanner.automation import autonomous_trainer as at
    fake_proc = MagicMock(pid=2345)
    fake_proc.poll.return_value = 2  # non-zero = failure
    at._active_retrain_process = fake_proc

    state = at.RetrainState(last_started_at=time.time() - 600, last_status="running")
    at._save_state(state)

    fired = []
    result = at.poll_completion(fired.append)
    assert result == "failed"
    assert any("FAILED" in m.upper() for m in fired)


def test_poll_completion_running_returns_running(iso):
    from src.scanner.automation import autonomous_trainer as at
    fake_proc = MagicMock(pid=3456)
    fake_proc.poll.return_value = None  # still running
    at._active_retrain_process = fake_proc

    fired = []
    result = at.poll_completion(fired.append)
    assert result == "running"
    # No completion message yet
    assert not any("RETRAIN" in m and "✓" in m for m in fired)


# ── Dry run ────────────────────────────────────────────────────────────


def test_dry_run_does_not_spawn(iso):
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    fired = []
    with patch.object(at.subprocess, "Popen") as mock_popen:
        result = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "CRITICAL", "oldest_age_days": 50},
            brain_callback=fired.append,
            dry_run=True,
        )
    assert result["spawned"] is False
    assert result["skipped_reason"] == "dry_run"
    mock_popen.assert_not_called()


# ── Missing script ────────────────────────────────────────────────────


def test_missing_script_fails_gracefully(iso):
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    # Remove the stub script
    (iso / "scripts" / "scheduled_retrain.py").unlink()

    fired = []
    result = at.maybe_spawn_autonomous_retrain(
        freshness={"status": "CRITICAL", "oldest_age_days": 50},
        brain_callback=fired.append,
    )
    assert result["spawned"] is False
    assert "missing script" in (result["skipped_reason"] or "")
    assert any("missing" in m for m in fired)


# ── Mon/Fri schedule ──────────────────────────────────────────────────


def test_schedule_fires_on_monday_with_fresh_models(iso, monkeypatch):
    """Mon trigger fires regardless of model freshness — weekly retrain."""
    _patch_today(monkeypatch, weekday=0)  # Monday
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    # Clear state so no cooldown
    at._save_state(at.RetrainState())

    fake_proc = MagicMock(pid=111)
    fake_proc.poll.return_value = None
    fired = []
    with patch.object(at.subprocess, "Popen", return_value=fake_proc):
        result = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "FRESH", "oldest_age_days": 1.0},
            brain_callback=fired.append,
        )
    assert result["spawned"] is True
    # State should record the schedule day
    state = at._load_state()
    assert state.last_schedule_day_iso is not None
    # Reason mentions schedule
    assert any("trigger=schedule" in m or "Mon" in m for m in fired)


def test_schedule_fires_on_friday(iso, monkeypatch):
    _patch_today(monkeypatch, weekday=4)  # Friday
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    at._save_state(at.RetrainState())

    fake_proc = MagicMock(pid=222)
    fake_proc.poll.return_value = None
    with patch.object(at.subprocess, "Popen", return_value=fake_proc):
        result = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "FRESH", "oldest_age_days": 1.0},
            brain_callback=lambda m: None,
        )
    assert result["spawned"] is True


def test_schedule_does_not_fire_on_wednesday(iso, monkeypatch):
    """Mid-week with FRESH models = nothing to do."""
    _patch_today(monkeypatch, weekday=2)  # Wednesday
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    at._save_state(at.RetrainState())

    fired = []
    result = at.maybe_spawn_autonomous_retrain(
        freshness={"status": "FRESH", "oldest_age_days": 1.0},
        brain_callback=fired.append,
    )
    assert result["spawned"] is False
    assert "no trigger" in (result["skipped_reason"] or "").lower()


def test_schedule_only_fires_once_per_day(iso, monkeypatch):
    """Two cycles on the same Mon = one spawn, not two."""
    _patch_today(monkeypatch, weekday=0)
    from src.scanner.automation import autonomous_trainer as at
    at._active_retrain_process = None
    at._save_state(at.RetrainState())

    fake_proc = MagicMock(pid=333)
    fake_proc.poll.return_value = None
    with patch.object(at.subprocess, "Popen", return_value=fake_proc):
        r1 = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "FRESH", "oldest_age_days": 0.5},
            brain_callback=lambda m: None,
        )
        # Process is "still running" → second call sees in-flight
        r2 = at.maybe_spawn_autonomous_retrain(
            freshness={"status": "FRESH", "oldest_age_days": 0.5},
            brain_callback=lambda m: None,
        )
    assert r1["spawned"] is True
    assert r2["spawned"] is False  # blocked by single-flight + same-day


def test_schedule_env_override_picks_different_days(iso, monkeypatch):
    """BUDDY_TRAIN_DAYS=tue,thu changes the schedule."""
    monkeypatch.setenv("BUDDY_TRAIN_DAYS", "tue,thu")
    # Re-import so the module-level SCHEDULED_TRAIN_DAYS recomputes
    import importlib
    from src.scanner.automation import autonomous_trainer as at
    importlib.reload(at)
    assert at.SCHEDULED_TRAIN_DAYS == {1, 3}  # Tue, Thu
    # Reload defaults back so other tests aren't affected
    monkeypatch.delenv("BUDDY_TRAIN_DAYS", raising=False)
    importlib.reload(at)
