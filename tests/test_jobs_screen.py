"""Smoke test for the Jobs TUI screen (Tier 1 T1)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.tui.screens.jobs_screen import JobsScreen
from src.scanner.automation.scheduled_jobs import ScheduledJobsRegistry


def test_jobs_screen_renders_active_and_paused(tmp_path: Path):
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

    async def _run():
        from textual.widgets import DataTable
        async with _Harness().run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, JobsScreen)
            table = screen.query_one("#jobs_table", DataTable)
            # Collect all cell values across all rows for substring assertions.
            cells: list[str] = []
            for row_key in table.rows:
                for col_key in table.columns:
                    try:
                        cells.append(str(table.get_cell(row_key, col_key)))
                    except Exception:
                        pass
            joined = " | ".join(cells)
            assert "Active job" in joined
            assert "Paused job" in joined
            assert "disk full" in joined
            # Verify state badges rendered
            assert "▶ active" in joined
            assert "⏸ paused" in joined

    asyncio.run(_run())


def test_jobs_screen_pause_resume_actions(tmp_path: Path):
    """Verify pause/resume actions mutate registry state via screen action handlers."""
    cfg = tmp_path / "jobs.json"
    state = tmp_path / "state.json"
    cfg.write_text(json.dumps({"jobs": [
        {"job_id": "j1", "name": "J1", "schedule": "daily_03:00",
         "command": "echo a", "enabled": True},
    ]}))
    state.write_text(json.dumps({"j1": {"state": "active", "last_status": "success", "run_count": 0}}))
    registry = ScheduledJobsRegistry(config_path=cfg, state_path=state)
    registry.load()

    from textual.app import App

    class _Harness(App):
        def on_mount(self):
            self.push_screen(JobsScreen(registry=registry))

    async def _run():
        async with _Harness().run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, JobsScreen)
            # Pause via action handler
            screen.action_pause_selected()
            await pilot.pause()
            assert registry.state("j1").state == "paused"
            # Resume via action handler
            screen.action_resume_selected()
            await pilot.pause()
            assert registry.state("j1").state == "active"

    asyncio.run(_run())
