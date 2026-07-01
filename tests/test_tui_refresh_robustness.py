"""Cluster-2: screen auto-refresh robustness (B4, B5, B7).

B5  diagnostics_screen._load_live_log_entries: a corrupt non-numeric
    cycles_rejected must not crash the 5s refresh tick (int(...) guarded).
B4  same method: health_registry_log.json removed between read and stat()
    must degrade to now(), not raise (covered by inspection + the B5 test
    which exercises the same health block with a real file present).
B7  jobs_screen._refresh: the 5s DataTable.clear()+rebuild must preserve the
    operator's row selection so a pending pause/resume/trigger keypress acts
    on the hovered job, not whatever lands at row 0.

No mocks: real DiagnosticsScreen/JobsScreen, real ScheduledJobsRegistry, real
disk via tmp_path, real Textual Pilot mount.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.app import App

from src.tui.screens.diagnostics_screen import DiagnosticsScreen
from src.tui.screens.jobs_screen import JobsScreen
from src.scanner.automation.scheduled_jobs import ScheduledJobsRegistry


def test_diagnostics_survives_corrupt_cycles_rejected(tmp_path: Path):
    """B5: a non-numeric cycles_rejected must not raise inside the refresh."""
    health = tmp_path / "trained_data" / "health_registry_log.json"
    health.parent.mkdir(parents=True)
    health.write_text(json.dumps({
        "cycles_rejected": "abc",     # corrupt — pre-fix int() raised ValueError
        "total_preflights": 5,
        "module_states": {},
    }))
    screen = DiagnosticsScreen(project_root=str(tmp_path))
    entries = screen._load_live_log_entries()  # must not raise
    msgs = " | ".join(str(e.get("msg", "")) for e in entries)
    assert "Health registry" in msgs
    assert "5 preflights" in msgs        # rendered with rejected coerced to 0


def test_jobs_refresh_preserves_cursor_selection(tmp_path: Path):
    """B7: cursor stays on the selected job across an auto-refresh rebuild."""
    cfg = tmp_path / "jobs.json"
    state = tmp_path / "state.json"
    cfg.write_text(json.dumps({"jobs": [
        {"job_id": "j1", "name": "Job One", "schedule": "daily_03:00",
         "command": "echo a", "enabled": True},
        {"job_id": "j2", "name": "Job Two", "schedule": "every_30_minutes",
         "command": "echo b", "enabled": True},
        {"job_id": "j3", "name": "Job Three", "schedule": "weekly_mon_09:00",
         "command": "echo c", "enabled": True},
    ]}))
    state.write_text(json.dumps({
        "j1": {"state": "active", "last_status": "success"},
        "j2": {"state": "active", "last_status": "success"},
        "j3": {"state": "paused", "last_status": "failure"},
    }))
    registry = ScheduledJobsRegistry(config_path=cfg, state_path=state)
    registry.load()

    class _Harness(App):
        def on_mount(self) -> None:
            self.push_screen(JobsScreen(registry=registry))

    async def _run() -> None:
        async with _Harness().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, JobsScreen)
            screen._table.move_cursor(row=1)         # operator hovers job j2
            assert screen._selected_job_id() == "j2"
            screen._refresh()                        # simulate the 5s tick
            await pilot.pause()
            # Pre-fix: cursor reset to row 0 -> "j1". Post-fix: still "j2".
            assert screen._selected_job_id() == "j2"

    asyncio.run(_run())
