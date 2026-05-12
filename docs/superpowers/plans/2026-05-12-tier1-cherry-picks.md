# Tier 1 Cherry-Picks Implementation Plan (T1–T6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship six standalone production-quality lifts to Buddy: scheduler observability, liveness badge, work-unit counter, phase indicator, ConfigAdjuster TTL cache, and inline error banner. Each task is an independent PR.

**Architecture:** Each pattern adds a new Reactive widget or extends an existing scanner contract. No new processes, no HTTP boundary, no LLM in any hot path. All new files follow existing `src/tui/widgets/` / `src/tui/screens/` / `src/scanner/automation/` conventions. Tests use real disk via `tmp_path` per CLAUDE.md "no-mock" rule.

**Tech Stack:** Python 3.11, Textual (reactive TUI), pytest, `dataclasses`, `safe_json_read` / `safe_json_write` from the cloud branch.

**Prerequisite:** The cloud branch `origin/claude/cherry-pick-ml-engine-upgrade-hKlIu` must be merged to main first. It provides `scheduled_jobs.py`, `brain_caps.py`, `safe_json.py`, `log_tailer.py`, the live log viewer modal, the command palette, and the FTS5 trade search. T1 extends `scheduled_jobs.py`; T6 deep-links to the log viewer. If you're executing this plan and those files don't exist, stop and ask for the merge first.

---

## Task 1: Scheduler observability — JobRuntimeState extension + pause/trigger + Jobs screen

**Files:**
- Modify: `src/scanner/automation/scheduled_jobs.py` (extend `JobRuntimeState`, add `ScheduledJobsRegistry.pause_job`, `resume_job`, `trigger_now`)
- Create: `src/tui/screens/jobs_screen.py`
- Modify: `src/tui/app.py` (mount JobsScreen behind F9 binding)
- Create: `tests/test_scheduled_jobs_observability.py`

- [ ] **Step 1: Read the existing `JobRuntimeState` dataclass for context**

Run: `grep -n "class JobRuntimeState\|class JobConfig\|class ScheduledJobsRegistry\|def tick\|def due_jobs" src/scanner/automation/scheduled_jobs.py`
Expected: confirms `JobRuntimeState` has `last_run_at`, `last_status`, `last_error`, `run_count`; registry class at line 210 has `tick`, `due_jobs`, `_run_job`, `_persist_state`.

- [ ] **Step 2: Write the failing tests for `JobRuntimeState` schema extension**

Create `tests/test_scheduled_jobs_observability.py`:

```python
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
        r.pause_job("j1")
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
        before = datetime.now(UTC)
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
        future = datetime.now(UTC) + timedelta(hours=1)
        r.tick(now=future)
        time.sleep(0.5)
        s = r.state("j1")
        assert s.next_run_at_iso is not None
        nxt = datetime.fromisoformat(s.next_run_at_iso)
        assert nxt > future
```

- [ ] **Step 3: Run tests — confirm they fail**

Run: `pytest tests/test_scheduled_jobs_observability.py -v`
Expected: ALL tests fail with `AttributeError: 'JobRuntimeState' object has no attribute 'state'` (or similar).

- [ ] **Step 4: Extend `JobRuntimeState` and `ScheduledJobsRegistry`**

In `src/scanner/automation/scheduled_jobs.py`, replace the existing `JobRuntimeState` dataclass with the extended version:

```python
@dataclass
class JobRuntimeState:
    """Per-job runtime state. Persisted to trained_data/jobs_runtime_state.json."""

    last_run_at: Optional[str] = None
    last_status: str = "pending"
    last_error: Optional[str] = None
    run_count: int = 0
    # T1 additions:
    state: str = "active"
    next_run_at_iso: Optional[str] = None
    last_status_at: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: dict) -> "JobRuntimeState":
        return cls(
            last_run_at=raw.get("last_run_at"),
            last_status=str(raw.get("last_status", "pending")),
            last_error=raw.get("last_error"),
            run_count=int(raw.get("run_count", 0)),
            state=str(raw.get("state", "active")),
            next_run_at_iso=raw.get("next_run_at_iso"),
            last_status_at=raw.get("last_status_at"),
        )

    def to_dict(self) -> dict:
        return {
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "run_count": self.run_count,
            "state": self.state,
            "next_run_at_iso": self.next_run_at_iso,
            "last_status_at": self.last_status_at,
        }
```

Then add three methods to `ScheduledJobsRegistry` (after the existing `_run_job` method):

```python
    # ── T1: pause / resume / trigger / next_run_at persistence ─────────

    def pause_job(self, job_id: str) -> bool:
        """Mark a job paused. Returns True if changed; False if unknown id."""
        with self._lock:
            if job_id not in self._jobs:
                return False
            s = self._state.setdefault(job_id, JobRuntimeState())
            s.state = "paused"
            self._persist_state()
            return True

    def resume_job(self, job_id: str) -> bool:
        """Mark a job active. Returns True if changed; False if unknown id."""
        with self._lock:
            if job_id not in self._jobs:
                return False
            s = self._state.setdefault(job_id, JobRuntimeState())
            s.state = "active"
            self._persist_state()
            return True

    def trigger_now(self, job_id: str) -> bool:
        """Run a job immediately in a daemon thread. Returns True if started."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        now = datetime.now(timezone.utc)
        t = threading.Thread(
            target=self._run_job, args=(job, now),
            daemon=True, name=f"job_{job.job_id}_manual",
        )
        t.start()
        return True
```

Modify `due_jobs` to skip paused jobs — find the line `if not cfg.enabled:` and add a second guard:

```python
                if not cfg.enabled:
                    continue
                state = self._state.get(jid, JobRuntimeState())
                if state.state == "paused":
                    continue
```

Modify the `_run_job` finalization block to compute and persist `next_run_at_iso` + `last_status_at`:

```python
        with self._lock:
            state = self._state.setdefault(job.job_id, JobRuntimeState())
            state.last_run_at = started_at.replace(microsecond=0).isoformat()
            state.last_status = status
            state.last_error = error
            state.last_status_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            state.run_count += 1
            try:
                nxt = compute_next_run(job.schedule, last_run=started_at)
                state.next_run_at_iso = nxt.replace(microsecond=0).isoformat()
            except ValueError:
                state.next_run_at_iso = None
            self._persist_state()
```

- [ ] **Step 5: Run tests — confirm they pass**

Run: `pytest tests/test_scheduled_jobs_observability.py -v`
Expected: ALL tests pass.

- [ ] **Step 6: Commit the scheduler-side changes**

```bash
git add src/scanner/automation/scheduled_jobs.py tests/test_scheduled_jobs_observability.py
git commit -m "feat(scheduled_jobs): add state/next_run_at_iso/last_status_at + pause/resume/trigger_now (T1 backend)"
```

- [ ] **Step 7: Write the failing test for the Jobs screen**

Create `tests/test_jobs_screen.py`:

```python
"""Smoke test for the Jobs TUI screen (Tier 1 T1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tui.screens.jobs_screen import JobsScreen
from src.scanner.automation.scheduled_jobs import ScheduledJobsRegistry


@pytest.mark.asyncio
async def test_jobs_screen_renders_active_and_paused(tmp_path: Path):
    cfg = tmp_path / "jobs.json"
    state = tmp_path / "state.json"
    cfg.write_text(json.dumps({"jobs": [
        {"job_id": "j1", "name": "Active job", "schedule": "daily_03:00",
         "command": "echo a", "enabled": True},
        {"job_id": "j2", "name": "Paused job", "schedule": "every_30_minutes",
         "command": "echo b", "enabled": True},
    ]}))
    state.write_text(json.dumps({
        "j1": {"state": "active",  "last_status": "success", "run_count": 3},
        "j2": {"state": "paused",  "last_status": "failure", "run_count": 1,
               "last_error": "exit 1: disk full"},
    }))
    registry = ScheduledJobsRegistry(config_path=cfg, state_path=state)
    registry.load()

    from textual.app import App

    class _Harness(App):
        def on_mount(self):
            self.push_screen(JobsScreen(registry=registry))

    async with _Harness().run_test() as pilot:
        await pilot.pause()
        rendered = str(pilot.app.screen.tree)
        assert "Active job" in rendered
        assert "Paused job" in rendered
        assert "disk full" in rendered
```

- [ ] **Step 8: Run the screen test — confirm it fails**

Run: `pytest tests/test_jobs_screen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tui.screens.jobs_screen'`.

- [ ] **Step 9: Implement `JobsScreen`**

Create `src/tui/screens/jobs_screen.py`:

```python
"""Tier 1 T1: Jobs screen — list/pause/resume/trigger scheduled jobs."""
from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from src.scanner.automation.scheduled_jobs import ScheduledJobsRegistry


class JobsScreen(Screen):
    """List scheduled jobs with state badge + pause/resume/trigger actions."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("p", "pause_selected", "Pause"),
        ("r", "resume_selected", "Resume"),
        ("t", "trigger_selected", "Trigger Now"),
        ("f5", "refresh", "Refresh"),
    ]

    def __init__(self, *, registry: ScheduledJobsRegistry, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._registry = registry
        self._table: Optional[DataTable] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("Scheduled Jobs — [p]ause / [r]esume / [t]rigger / [F5] refresh / [Esc] back", classes="hint")
            yield DataTable(id="jobs_table", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self._table = self.query_one("#jobs_table", DataTable)
        self._table.add_columns("Job", "State", "Schedule", "Last Status", "Last Run", "Next Run", "Last Error")
        self._refresh()
        self.set_interval(5.0, self._refresh)

    def _refresh(self) -> None:
        if self._table is None:
            return
        self._table.clear()
        for job in self._registry.jobs():
            s = self._registry.state(job.job_id)
            badge = "▶ active" if s.state == "active" else "⏸ paused"
            self._table.add_row(
                job.name,
                badge,
                job.schedule,
                s.last_status,
                s.last_run_at or "—",
                s.next_run_at_iso or "—",
                (s.last_error or "")[:80],
                key=job.job_id,
            )

    def _selected_job_id(self) -> Optional[str]:
        if self._table is None or self._table.row_count == 0:
            return None
        try:
            return self._table.coordinate_to_cell_key(self._table.cursor_coordinate).row_key.value
        except Exception:
            return None

    def action_pause_selected(self) -> None:
        jid = self._selected_job_id()
        if jid:
            self._registry.pause_job(jid)
            self._refresh()

    def action_resume_selected(self) -> None:
        jid = self._selected_job_id()
        if jid:
            self._registry.resume_job(jid)
            self._refresh()

    def action_trigger_selected(self) -> None:
        jid = self._selected_job_id()
        if jid:
            self._registry.trigger_now(jid)
            self._refresh()

    def action_refresh(self) -> None:
        self._refresh()
```

- [ ] **Step 10: Run screen test — confirm it passes**

Run: `pytest tests/test_jobs_screen.py -v`
Expected: PASS.

- [ ] **Step 11: Wire JobsScreen behind an F9 binding in `tui/app.py`**

In `src/tui/app.py`, find the `BINDINGS` block (around line 814) and add after the F8 line:

```python
        Binding("f9", "switch_tab('jobs')", "Jobs", show=True),
```

In `action_switch_tab` (around line 1475), wire `'jobs'` to push the `JobsScreen` with the live `ScheduledJobsRegistry` from `orchestrator._scheduled_jobs`. If the orchestrator handle isn't easily accessible from app.py, resolve lazily inside the action handler.

- [ ] **Step 12: Manual smoke test in TUI**

```bash
./buddy --demo
# Press F9 — Jobs screen renders.
# Press T — manual trigger fires.
# Press P — badge flips to "⏸ paused".
# Press R — badge flips back to "▶ active".
# Press Esc — return.
```

- [ ] **Step 13: Commit the screen**

```bash
git add src/tui/screens/jobs_screen.py src/tui/app.py tests/test_jobs_screen.py
git commit -m "feat(tui): Jobs screen with pause/resume/trigger (T1 frontend)"
```

---

## Task 2: Liveness badge — Reactive widget over heartbeat.json

**Files:**
- Create: `src/tui/widgets/liveness_badge.py`
- Modify: `src/tui/app.py` (mount widget in footer/header area)
- Create: `tests/test_liveness_badge.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_liveness_badge.py`:

```python
"""Tier 1 T2: Tests for the liveness badge state computation."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tui.widgets.liveness_badge import compute_liveness


UTC = timezone.utc


def _heartbeat(path: Path, *, alive: bool, age_sec: float, halted: bool, cycle: int) -> None:
    ts = datetime.now(UTC) - timedelta(seconds=age_sec)
    payload = {
        "scanner_alive": alive,
        "ts_iso": ts.isoformat(),
        "cycle_count": cycle,
        "pid": 1234,
        "mode": "live",
        "last_error_ts": None,
    }
    path.write_text(json.dumps(payload))
    (path.parent / "state.json").write_text(json.dumps({"halted": halted, "mode": "live"}))


def test_live_when_fresh_and_alive_and_not_halted(tmp_path: Path):
    _heartbeat(tmp_path / "heartbeat.json", alive=True, age_sec=2, halted=False, cycle=10)
    res = compute_liveness(tmp_path)
    assert res.label == "LIVE"
    assert res.color == "green"
    assert res.cycles == 10


def test_halted_when_state_halted_true(tmp_path: Path):
    _heartbeat(tmp_path / "heartbeat.json", alive=True, age_sec=2, halted=True, cycle=42)
    res = compute_liveness(tmp_path)
    assert res.label == "HALTED"
    assert res.color == "yellow"


def test_stale_when_heartbeat_older_than_30s(tmp_path: Path):
    _heartbeat(tmp_path / "heartbeat.json", alive=True, age_sec=45, halted=False, cycle=10)
    res = compute_liveness(tmp_path)
    assert res.label == "STALE"
    assert res.color == "red"


def test_missing_heartbeat_returns_init(tmp_path: Path):
    (tmp_path / "state.json").write_text(json.dumps({"halted": False}))
    res = compute_liveness(tmp_path)
    assert res.label == "INIT"
    assert res.color == "cyan"


def test_dead_when_scanner_alive_false(tmp_path: Path):
    _heartbeat(tmp_path / "heartbeat.json", alive=False, age_sec=2, halted=False, cycle=10)
    res = compute_liveness(tmp_path)
    assert res.label == "ERROR"
    assert res.color == "red"
```

- [ ] **Step 2: Run test — confirm failure**

Run: `pytest tests/test_liveness_badge.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the badge**

Create `src/tui/widgets/liveness_badge.py`:

```python
"""Tier 1 T2: Liveness badge — reactive heartbeat.json reader."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from textual.reactive import reactive
from textual.widgets import Static

logger = logging.getLogger(__name__)

_STALE_THRESHOLD_SEC = 30.0
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLAUDE_DIR = _REPO_ROOT / ".claude"


@dataclass(frozen=True)
class LivenessResult:
    label: str
    color: str
    cycles: int
    age_sec: float


def compute_liveness(claude_dir: Path = _CLAUDE_DIR) -> LivenessResult:
    """Pure function — read heartbeat + state, decide label/color."""
    hb_path = claude_dir / "heartbeat.json"
    state_path = claude_dir / "state.json"

    halted = False
    try:
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as f:
                halted = bool(json.load(f).get("halted", False))
    except (OSError, json.JSONDecodeError):
        pass

    if not hb_path.exists():
        return LivenessResult(label="INIT", color="cyan", cycles=0, age_sec=float("inf"))

    try:
        with hb_path.open("r", encoding="utf-8") as f:
            hb = json.load(f)
    except (OSError, json.JSONDecodeError):
        return LivenessResult(label="ERROR", color="red", cycles=0, age_sec=float("inf"))

    cycles = int(hb.get("cycle_count", 0))
    alive = bool(hb.get("scanner_alive", False))
    ts_iso = hb.get("ts_iso")
    age = float("inf")
    if ts_iso:
        try:
            ts = datetime.fromisoformat(ts_iso)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
        except ValueError:
            pass

    if not alive:
        return LivenessResult(label="ERROR", color="red", cycles=cycles, age_sec=age)
    if age > _STALE_THRESHOLD_SEC:
        return LivenessResult(label="STALE", color="red", cycles=cycles, age_sec=age)
    if halted:
        return LivenessResult(label="HALTED", color="yellow", cycles=cycles, age_sec=age)
    return LivenessResult(label="LIVE", color="green", cycles=cycles, age_sec=age)


class LivenessBadge(Static):
    """Footer/sidebar widget — auto-refreshes every 5s."""

    label: reactive[str] = reactive("INIT")
    color: reactive[str] = reactive("cyan")
    cycles: reactive[int] = reactive(0)

    def __init__(self, *, claude_dir: Path = _CLAUDE_DIR, refresh_sec: float = 5.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._claude_dir = claude_dir
        self._refresh_sec = refresh_sec

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(self._refresh_sec, self._tick)

    def _tick(self) -> None:
        try:
            res = compute_liveness(self._claude_dir)
        except Exception as e:
            logger.debug("LivenessBadge tick failed: %s", e)
            return
        self.label = res.label
        self.color = res.color
        self.cycles = res.cycles
        self.update(self.render_label())

    def render_label(self) -> str:
        return f"[{self.color}]● {self.label}[/] · cycles={self.cycles}"
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_liveness_badge.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Mount in `tui/app.py` footer area**

Find where `yield Footer()` is rendered in the main compose tree. Above it:

```python
from src.tui.widgets.liveness_badge import LivenessBadge
# ...
        yield LivenessBadge(id="liveness_badge")
        yield Footer()
```

- [ ] **Step 6: Manual smoke**

```bash
./buddy --demo
# Confirm badge appears, says LIVE (or HALTED if state.json:halted=true).
# Touch .claude/heartbeat.json with an old ts_iso (60s back); wait 5s; badge flips to STALE.
```

- [ ] **Step 7: Commit**

```bash
git add src/tui/widgets/liveness_badge.py src/tui/app.py tests/test_liveness_badge.py
git commit -m "feat(tui): liveness badge reads heartbeat.json + state.json (T2)"
```

---

## Task 3: Cumulative work-unit counter

**Files:**
- Modify: `src/tui/embedded_scanner.py` (add `ScanCounters` instance + increment logic)
- Create: `src/tui/widgets/stats_bar.py`
- Modify: `src/tui/app.py` (mount stats bar in F1 overview area)
- Create: `tests/test_stats_bar.py`

- [ ] **Step 1: Write failing test for ScanCounters increment logic**

Create `tests/test_stats_bar.py`:

```python
"""Tier 1 T3: Tests for cumulative work-unit counter."""
from __future__ import annotations

from src.tui.widgets.stats_bar import ScanCounters


def test_counters_default_zero():
    c = ScanCounters()
    assert c.cycles == 0
    assert c.pairs_scanned == 0
    assert c.gates_checked == 0
    assert c.trades_executed == 0


def test_increment_cycle_only():
    c = ScanCounters()
    c.bump_cycle()
    assert c.cycles == 1
    assert c.pairs_scanned == 0


def test_increment_pair_and_gate_and_trade():
    c = ScanCounters()
    c.bump_pair(3)
    c.bump_gates_checked(8)
    c.bump_trade(2)
    assert c.pairs_scanned == 3
    assert c.gates_checked == 8
    assert c.trades_executed == 2


def test_format_compact():
    c = ScanCounters(cycles=42, pairs_scanned=100, gates_checked=560, trades_executed=7)
    s = c.format_compact()
    assert "42" in s
    assert "100" in s
    assert "7" in s


def test_format_detailed_has_all_fields():
    c = ScanCounters(cycles=42, pairs_scanned=100, gates_checked=560, trades_executed=7)
    s = c.format_detailed()
    assert "cycles" in s.lower()
    assert "pairs" in s.lower()
    assert "gates" in s.lower()
    assert "trades" in s.lower()
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_stats_bar.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ScanCounters` and `StatsBar`**

Create `src/tui/widgets/stats_bar.py`:

```python
"""Tier 1 T3: Cumulative work-unit counter widget."""
from __future__ import annotations

from dataclasses import dataclass
from textual.reactive import reactive
from textual.widgets import Static


@dataclass
class ScanCounters:
    """Cumulative work-unit counters, lifetime of TUI session."""

    cycles: int = 0
    pairs_scanned: int = 0
    gates_checked: int = 0
    trades_executed: int = 0

    def bump_cycle(self, n: int = 1) -> None:
        self.cycles += n

    def bump_pair(self, n: int = 1) -> None:
        self.pairs_scanned += n

    def bump_gates_checked(self, n: int = 1) -> None:
        self.gates_checked += n

    def bump_trade(self, n: int = 1) -> None:
        self.trades_executed += n

    def format_compact(self) -> str:
        return f"cycles {self.cycles} · pairs {self.pairs_scanned} · gates {self.gates_checked} · trades {self.trades_executed}"

    def format_detailed(self) -> str:
        return (
            f"cycles: {self.cycles}\n"
            f"pairs scanned: {self.pairs_scanned}\n"
            f"gates checked: {self.gates_checked}\n"
            f"trades executed: {self.trades_executed}"
        )


class StatsBar(Static):
    """Compact stats line. Tooltip shows detailed breakdown."""

    text: reactive[str] = reactive("")

    def __init__(self, *, counters: ScanCounters, refresh_sec: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._counters = counters
        self._refresh_sec = refresh_sec
        self.tooltip = ""

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(self._refresh_sec, self._tick)

    def _tick(self) -> None:
        self.text = self._counters.format_compact()
        self.tooltip = self._counters.format_detailed()
        self.update(self.text)
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_stats_bar.py -v`
Expected: PASS.

- [ ] **Step 5: Wire counters in EmbeddedScanner**

In `src/tui/embedded_scanner.py`, in `__init__` (near `self._brain = brain_callback` line 122):

```python
        from src.tui.widgets.stats_bar import ScanCounters
        self.counters = ScanCounters()
```

In `run_one_cycle()` (line 465), at the top of the method, add `self.counters.bump_cycle()`. After the per-pair loop, call `self.counters.bump_pair(scanned_count)` using `ScanEnrichment.scanned_count`. At the gate-check phase increment `bump_gates_checked`. At successful execution increment `bump_trade(ok)` using the existing `ok` variable near line 817.

Locate each insertion point by grepping `self._brain` calls inside `run_one_cycle` — counter bumps go inline beside them.

- [ ] **Step 6: Mount StatsBar in `tui/app.py` overview area**

```python
        from src.tui.widgets.stats_bar import StatsBar
        # in compose() where F1 overview is built:
        yield StatsBar(counters=self._scanner.counters, id="stats_bar")
```

(Adapt `self._scanner` to whatever attribute holds the EmbeddedScanner instance.)

- [ ] **Step 7: Manual smoke**

```bash
./buddy --demo
# F1 overview — stats bar shows "cycles N · pairs M · gates K · trades T".
# After 2 cycles run, numbers visibly increment.
```

- [ ] **Step 8: Commit**

```bash
git add src/tui/widgets/stats_bar.py src/tui/embedded_scanner.py src/tui/app.py tests/test_stats_bar.py
git commit -m "feat(tui): cumulative work-unit counter (T3)"
```

---

## Task 4: Transient phase indicator

**Files:**
- Modify: `src/tui/embedded_scanner.py` (add `phase_state` field + fire `.set(...)` at phase boundaries)
- Create: `src/tui/widgets/phase_indicator.py`
- Modify: `src/tui/app.py` (mount indicator near brain feed)
- Create: `tests/test_phase_indicator.py`

- [ ] **Step 1: Write failing test for PhaseIndicator state transitions**

Create `tests/test_phase_indicator.py`:

```python
"""Tier 1 T4: Tests for the transient phase indicator widget."""
from __future__ import annotations

from src.tui.widgets.phase_indicator import PhaseState


def test_default_is_idle():
    p = PhaseState()
    assert p.phase == "idle"
    assert p.detail == ""


def test_set_phase_updates_both_fields():
    p = PhaseState()
    p.set("scanning", "EUR_USD")
    assert p.phase == "scanning"
    assert p.detail == "EUR_USD"


def test_clear_resets_to_idle():
    p = PhaseState()
    p.set("scanning", "EUR_USD")
    p.clear()
    assert p.phase == "idle"
    assert p.detail == ""


def test_format_idle_returns_dim_placeholder():
    p = PhaseState()
    s = p.format()
    assert "idle" in s.lower() or s.strip() == "" or "—" in s


def test_format_active_includes_phase_and_detail():
    p = PhaseState(phase="gate-check", detail="agent 7/15 devil_advocate")
    s = p.format()
    assert "gate-check" in s.lower()
    assert "devil_advocate" in s
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_phase_indicator.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement PhaseIndicator + PhaseState**

Create `src/tui/widgets/phase_indicator.py`:

```python
"""Tier 1 T4: Transient phase indicator — "currently doing X" widget."""
from __future__ import annotations

from dataclasses import dataclass
from textual.reactive import reactive
from textual.widgets import Static


@dataclass
class PhaseState:
    phase: str = "idle"
    detail: str = ""

    def set(self, phase: str, detail: str = "") -> None:
        self.phase = phase
        self.detail = detail

    def clear(self) -> None:
        self.phase = "idle"
        self.detail = ""

    def format(self) -> str:
        if self.phase == "idle":
            return "[dim]— idle —[/]"
        body = f"[cyan]▸ {self.phase}[/]"
        if self.detail:
            body += f" · {self.detail}"
        return body


class PhaseIndicator(Static):
    """Inline transient phase tag; reads PhaseState every 0.5s."""

    text: reactive[str] = reactive("[dim]— idle —[/]")

    def __init__(self, *, state: PhaseState, refresh_sec: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._refresh_sec = refresh_sec

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(self._refresh_sec, self._tick)

    def _tick(self) -> None:
        new_text = self._state.format()
        if new_text != self.text:
            self.text = new_text
            self.update(self.text)
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_phase_indicator.py -v`
Expected: PASS.

- [ ] **Step 5: Wire `phase_state` into EmbeddedScanner**

In `src/tui/embedded_scanner.py`, in `__init__` (near the counters from T3):

```python
        from src.tui.widgets.phase_indicator import PhaseState
        self.phase_state = PhaseState()
```

In `run_one_cycle()`, at natural phase boundaries (adjacent to existing `self._brain` calls):

```python
        self.phase_state.set("scanning", f"cycle #{self._scan_count}")
        # ... after the per-pair scan completes:
        self.phase_state.set("gate-check", f"{scanned_count} pairs")
        # ... after execution attempts:
        self.phase_state.set("executing", f"{ok} trades")
        # ... at the very end of the cycle:
        self.phase_state.clear()
```

- [ ] **Step 6: Mount PhaseIndicator in `tui/app.py` overview area**

Beside or above the F1 brain feed:

```python
        from src.tui.widgets.phase_indicator import PhaseIndicator
        yield PhaseIndicator(state=self._scanner.phase_state, id="phase_indicator")
```

- [ ] **Step 7: Manual smoke**

```bash
./buddy --demo
# F1 — observe the indicator transitioning: "scanning · cycle #N" → "gate-check · N pairs" → "idle".
```

- [ ] **Step 8: Commit**

```bash
git add src/tui/widgets/phase_indicator.py src/tui/embedded_scanner.py src/tui/app.py tests/test_phase_indicator.py
git commit -m "feat(tui): transient phase indicator (T4)"
```

---

## Task 5: TTL cache for ConfigAdjuster._load_state

**Files:**
- Modify: `src/scanner/automation/config_adjuster.py`
- Create: `tests/test_config_adjuster_ttl_cache.py`

- [ ] **Step 1: Write failing test for TTL cache behavior**

Create `tests/test_config_adjuster_ttl_cache.py`:

```python
"""Tier 1 T5: TTL cache around ConfigAdjuster._load_state."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.scanner.automation.config_adjuster import ConfigAdjuster


def _seed(path: Path, history: list, pending: list) -> None:
    path.write_text(json.dumps({
        "history": history,
        "pending": pending,
        "last_applied": None,
    }))


def test_cache_hit_within_ttl_does_not_reread(tmp_path: Path):
    p = tmp_path / "config_adjustments.json"
    _seed(p, history=[{"key": "x", "value": 1}], pending=[])
    a = ConfigAdjuster(persistence_path=p, ttl_seconds=5.0)
    a._load_state()
    _seed(p, history=[{"key": "x", "value": 2}], pending=[])
    a._load_state()
    assert a._load_count == 1


def test_cache_expires_after_ttl(tmp_path: Path):
    p = tmp_path / "config_adjustments.json"
    _seed(p, history=[{"key": "x", "value": 1}], pending=[])
    a = ConfigAdjuster(persistence_path=p, ttl_seconds=0.1)
    a._load_state()
    time.sleep(0.15)
    a._load_state()
    assert a._load_count == 2


def test_invalidate_forces_reload(tmp_path: Path):
    p = tmp_path / "config_adjustments.json"
    _seed(p, history=[{"key": "x", "value": 1}], pending=[])
    a = ConfigAdjuster(persistence_path=p, ttl_seconds=60.0)
    a._load_state()
    a._invalidate_cache()
    a._load_state()
    assert a._load_count == 2
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_config_adjuster_ttl_cache.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'ttl_seconds'`.

- [ ] **Step 3: Read existing ConfigAdjuster shape to plan the patch**

Run: `grep -n "def __init__\|def _load_state\|def _save\|class ConfigAdjuster" src/scanner/automation/config_adjuster.py`
Expected: `class ConfigAdjuster` at line 44, `__init__` at 53, `_load_state` at 310.

- [ ] **Step 4: Add TTL cache to ConfigAdjuster**

In `__init__` add the new kwarg + counters:

```python
    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        # ... existing kwargs ...
        ttl_seconds: float = 5.0,
    ) -> None:
        # ... existing init ...
        self._ttl_seconds = float(ttl_seconds)
        self._last_load_ts: float = 0.0
        self._load_count: int = 0
```

Modify `_load_state` to check cache freshness at entry:

```python
    def _load_state(self) -> None:
        import time as _time
        now = _time.monotonic()
        if (now - self._last_load_ts) < self._ttl_seconds and self._last_load_ts > 0:
            return
        # --- existing _load_state body unchanged below ---
        self._last_load_ts = now
        self._load_count += 1
```

Add the invalidation helper:

```python
    def _invalidate_cache(self) -> None:
        self._last_load_ts = 0.0
```

Locate every method that writes to `self.persistence_path` (e.g. `_save`, `_save_approved`, `add_pending`, any write path inside `apply_adjustments`). After each write, call `self._invalidate_cache()`.

- [ ] **Step 5: Run tests — confirm pass**

Run: `pytest tests/test_config_adjuster_ttl_cache.py -v`
Expected: PASS.

- [ ] **Step 6: Run the existing ConfigAdjuster test suite to make sure nothing regressed**

Run: `pytest tests/ -k config_adjuster -v`
Expected: ALL prior ConfigAdjuster tests still pass. If any depend on each `_load_state()` re-reading disk, either set `ttl_seconds=0` in those tests or call `_invalidate_cache()` between reads.

- [ ] **Step 7: Commit**

```bash
git add src/scanner/automation/config_adjuster.py tests/test_config_adjuster_ttl_cache.py
git commit -m "perf(config_adjuster): 5s TTL cache on _load_state with invalidate-on-write (T5)"
```

---

## Task 6: Inline error banner with View-Log deeplink

**Files:**
- Modify: `src/tui/embedded_scanner.py` (set `error_banner` on caught non-gate exceptions)
- Modify: `src/tui/app.py` (Reactive watcher → Textual `notify` with action)
- Create: `tests/test_error_banner.py`

- [ ] **Step 1: Write failing test for `error_banner` field surfacing**

Create `tests/test_error_banner.py`:

```python
"""Tier 1 T6: Tests for inline error banner surface."""
from __future__ import annotations

import pytest

from src.tui.embedded_scanner import EmbeddedScanner


def _capture_brain():
    msgs: list[str] = []
    def cb(line: str) -> None:
        msgs.append(str(line))
    return cb, msgs


def test_error_banner_field_exists_and_starts_none():
    cb, _ = _capture_brain()
    es = EmbeddedScanner(brain_callback=cb)
    assert hasattr(es, "error_banner")
    assert es.error_banner is None


def test_error_banner_set_on_non_gate_exception(monkeypatch):
    cb, _ = _capture_brain()
    es = EmbeddedScanner(brain_callback=cb)

    def boom(*a, **kw):
        raise FileNotFoundError("model file missing: foo.pkl")
    monkeypatch.setattr(es, "_init_scanner", boom, raising=False)
    try:
        es.run_one_cycle()
    except Exception:
        pass
    assert es.error_banner is not None
    assert "model file missing" in es.error_banner or "foo.pkl" in es.error_banner


def test_error_banner_clearable():
    cb, _ = _capture_brain()
    es = EmbeddedScanner(brain_callback=cb)
    es.error_banner = "something broke"
    es.dismiss_error_banner()
    assert es.error_banner is None
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_error_banner.py -v`
Expected: FAIL — `AttributeError: 'EmbeddedScanner' object has no attribute 'error_banner'`.

- [ ] **Step 3: Add `error_banner` to EmbeddedScanner**

In `src/tui/embedded_scanner.py`, in `__init__`:

```python
        self.error_banner: Optional[str] = None
```

Add the dismiss helper near the bottom of the class:

```python
    def dismiss_error_banner(self) -> None:
        self.error_banner = None
```

Locate every `except` block in `run_one_cycle` and `_init_scanner` that calls `self._brain(...)` with a red error message. Add a sibling `self.error_banner = ...` assignment:

```python
        except FileNotFoundError as e:
            msg = f"Model file missing: {e}"
            self._brain(f"[red]✗ {msg}[/]")
            self.error_banner = msg
        except Exception as e:
            msg = f"Scan #{self._scan_count} failed: {e}"
            self._brain(f"[red]✗ {msg}[/]")
            self.error_banner = msg
```

Gate-rejection paths stay untouched — gate failures are expected outcomes, not errors.

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_error_banner.py -v`
Expected: PASS.

- [ ] **Step 5: Wire the watcher in `tui/app.py`**

In `src/tui/app.py`, find the periodic refresh hook (search for `tradeable_count` or `Update Trades screen`). Append:

```python
        banner = getattr(self._scanner, "error_banner", None)
        if banner and banner != getattr(self, "_last_banner", None):
            self._last_banner = banner
            self.notify(
                f"{banner}\n\n[dim]Ctrl+L to view full log[/]",
                title="Scanner error",
                severity="error",
                timeout=15,
            )
```

Add `self._last_banner = None` to the app `__init__` so deduplication works.

- [ ] **Step 6: Manual smoke**

```bash
./buddy --demo
# Force a failure: temporarily rename trained_data/models/EUR_USD/transformer_direction.pkl.
# Boot. F1. A notification toast appears: "Model file missing... Ctrl+L to view full log".
# Press Ctrl+L — log viewer opens (pick #7 dependency).
# Restore the file.
```

- [ ] **Step 7: Commit**

```bash
git add src/tui/embedded_scanner.py src/tui/app.py tests/test_error_banner.py
git commit -m "feat(tui): inline error banner with View-Log deeplink (T6)"
```

---

## Self-review checklist

1. **Spec coverage:** every T1–T6 row in `docs/superpowers/specs/2026-05-12-hermes-pattern-absorption-design.md` has a task above. ✓
2. **Placeholder scan:** no "TBD", "implement later", or "add error handling" without showing the code. ✓
3. **Type consistency:** `ScanCounters` named consistently across T3 widget and EmbeddedScanner attribute. `PhaseState` / `PhaseIndicator` consistent across T4 widget and EmbeddedScanner attribute. `error_banner` field name consistent across T6 EmbeddedScanner field and app.py reader. ✓
4. **Test patterns:** all new tests use real disk via `tmp_path`, real classes — no `MagicMock`. (The grandfathered `test_scheduled_jobs.py` from cloud branch keeps its existing mocks.) ✓
5. **CLAUDE.md alignment:** no Claude in hot path (Tier 1 is all in-process scanner/TUI work), atomic writes via existing `safe_json_write`, no bare excepts (each new `except` is typed). ✓

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task (T1–T6), review between tasks. Six PRs.
2. **Inline Execution** — execute T1–T6 in one session using `superpowers:executing-plans`, batch with checkpoints.

Recommend (1) for Tier 1 since each task is independently PR-able.
