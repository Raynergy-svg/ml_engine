"""US-007: JournalScreen gate-trace row-selection tests.

NO MOCKS per CLAUDE.md. Uses real JournalScreen with real tmp_path journal
files. Async tests use asyncio.run() (no pytest-asyncio dependency needed).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from src.tui.screens.gate_trace_modal import GateTraceModal
from src.tui.screens.journal_screen import JournalScreen


# ---------------------------------------------------------------------------
# Sample trade fixtures
# ---------------------------------------------------------------------------

_TRADE_WITH_ID: dict = {
    "trade_id": "T-001",
    "pair": "EUR_USD",
    "direction": "BUY",
    "entry_price": 1.08230,
    "exit_price": 1.08450,
    "entry_time": "2026-04-01T10:00:00",
    "agents": {"weighted_vote_score": 0.75},
    "outcome": {"realized_pl": 150.0, "trade_won": True, "exit_reason": "tp_hit"},
}

_TRADE_NO_ID: dict = {
    "pair": "GBP_USD",
    "direction": "SELL",
    "entry_price": 1.25000,
    "exit_price": 1.24800,
    "entry_time": "2026-04-02T10:00:00",
    "agents": {"weighted_vote_score": 0.60},
    "outcome": {"realized_pl": -50.0, "trade_won": False, "exit_reason": "sl_hit"},
}


def _setup_project(tmp_path: Path, trades: list[dict]) -> None:
    journal = tmp_path / "trained_data" / "trade_journal_rl.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps(trades))
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "learnings.md").write_text("- test learning\n")


# ---------------------------------------------------------------------------
# Test harness App
# ---------------------------------------------------------------------------


class _JournalTestApp(App):
    """Minimal app that mounts JournalScreen for testing."""

    def __init__(self, project_root: Path, **kw: Any) -> None:
        super().__init__(**kw)
        self._project_root = project_root
        self.captured_toasts: list[str] = []

    def compose(self) -> ComposeResult:
        yield JournalScreen(project_root=str(self._project_root), id="journal")

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: Literal["information", "warning", "error"] = "information",
        timeout: float | None = None,
        markup: bool = True,
    ) -> None:
        self.captured_toasts.append(message)
        super().notify(message, title=title, severity=severity, timeout=timeout, markup=markup)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJournalGateTrace:
    """AC coverage for US-007."""

    def test_row_trade_ids_populated_from_journal(self, tmp_path: Path) -> None:
        """Baseline: _row_trade_ids is filled correctly from journal file."""
        _setup_project(tmp_path, [_TRADE_WITH_ID])

        async def _run() -> None:
            app = _JournalTestApp(project_root=tmp_path)
            async with app.run_test(headless=True, size=(120, 40)):
                screen = app.query_one("#journal", JournalScreen)
                assert screen._row_trade_ids == ["T-001"]

        asyncio.run(_run())

    def test_row_selected_pushes_gate_trace_modal(self, tmp_path: Path) -> None:
        """AC-1 + AC-2: selecting a journal row pushes GateTraceModal with correct trade_id."""
        _setup_project(tmp_path, [_TRADE_WITH_ID])

        async def _run() -> None:
            app = _JournalTestApp(project_root=tmp_path)
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                table = app.query_one("#journal-table", DataTable)
                table.focus()
                await pilot.press("enter")
                await pilot.pause(delay=0.1)

                # push_screen adds GateTraceModal to screen_stack, not the widget tree
                modal_screens = [s for s in app.screen_stack if isinstance(s, GateTraceModal)]
                assert modal_screens, "GateTraceModal must be on screen stack after Enter on journal row"
                assert modal_screens[0]._trade_id == "T-001"

        asyncio.run(_run())

    def test_stale_row_shows_toast_not_modal(self, tmp_path: Path) -> None:
        """AC-3: row with no trade_id shows toast; no modal is pushed."""
        _setup_project(tmp_path, [_TRADE_NO_ID])

        async def _run() -> None:
            app = _JournalTestApp(project_root=tmp_path)
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                screen = app.query_one("#journal", JournalScreen)
                assert screen._row_trade_ids == [""], "Row with no id should be empty string"

                table = app.query_one("#journal-table", DataTable)
                table.focus()
                await pilot.press("enter")
                await pilot.pause(delay=0.1)

                modal_screens = [s for s in app.screen_stack if isinstance(s, GateTraceModal)]
                assert not modal_screens, "No modal should be pushed for stale/empty trade_id"
                assert "gate trace unavailable for this trade" in app.captured_toasts

        asyncio.run(_run())

    def test_multi_trade_correct_id_from_row_index(self, tmp_path: Path) -> None:
        """AC-2: second row selects the second trade's trade_id, not the first."""
        trade_a = dict(_TRADE_WITH_ID, trade_id="T-AAA")
        trade_b = dict(_TRADE_WITH_ID, trade_id="T-BBB", entry_time="2026-04-02T10:00:00")
        # reversed order: T-BBB first, T-AAA second
        _setup_project(tmp_path, [trade_a, trade_b])

        async def _run() -> None:
            app = _JournalTestApp(project_root=tmp_path)
            async with app.run_test(headless=True, size=(120, 40)) as pilot:
                screen = app.query_one("#journal", JournalScreen)
                assert len(screen._row_trade_ids) == 2
                # _load_journal uses reversed(entries[-50:]) → T-BBB row 0, T-AAA row 1
                assert screen._row_trade_ids[0] == "T-BBB"
                assert screen._row_trade_ids[1] == "T-AAA"

                table = app.query_one("#journal-table", DataTable)
                table.focus()
                # Move cursor down to row 1
                await pilot.press("down")
                await pilot.press("enter")
                await pilot.pause(delay=0.1)

                modal_screens = [s for s in app.screen_stack if isinstance(s, GateTraceModal)]
                assert modal_screens, "GateTraceModal must be on screen stack"
                assert modal_screens[0]._trade_id == "T-AAA"

        asyncio.run(_run())
