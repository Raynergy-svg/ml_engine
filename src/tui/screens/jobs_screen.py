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
        # Preserve the operator's row selection across the 5s auto-refresh:
        # DataTable.clear() resets the cursor to row 0, which would yank a
        # hovered job out from under a pending pause/resume/trigger keypress.
        prev_key = self._selected_job_id()
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
        # Restore the cursor to the previously-selected job if it still exists.
        if prev_key is not None:
            for row_index, row_key in enumerate(self._table.rows):
                if row_key.value == prev_key:
                    self._table.move_cursor(row=row_index)
                    break

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
