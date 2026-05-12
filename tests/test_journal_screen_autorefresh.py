"""JournalScreen auto-refresh on file mtime change.

Prior behavior: JournalScreen read `trade_journal_rl.json` once at mount and
`update_from_snapshot` was a no-op (commented "journal is mostly static").
Result: closed trades never showed up until TUI restart.

New behavior: each snapshot tick stats the journal + learnings files;
reload happens only when mtime advances. This pins:
  1. Initial mount populates rows + captures mtime.
  2. update_from_snapshot with unchanged mtime is a no-op.
  3. update_from_snapshot after a journal write reloads + does NOT duplicate rows.
  4. Missing journal file does not crash the refresh path.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import DataTable

from src.tui.screens.journal_screen import JournalScreen


def _make_trade(trade_id: str, pair: str = "EUR_USD", pl: float = 50.0, won: bool = True) -> dict:
    return {
        "trade_id": trade_id,
        "pair": pair,
        "direction": "long",
        "entry_price": 1.08000,
        "exit_price": 1.08050,
        "confidence": 0.7,
        "entry_time": "2026-05-12T18:00:00",
        "outcome": {
            "realized_pl": pl,
            "trade_won": won,
            "exit_reason": "tp_hit",
        },
    }


def _write_journal(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


class _Harness(App):
    """Bare app that just mounts JournalScreen — no other screens or tabs."""

    def __init__(self, project_root: Path, **kwargs):
        super().__init__(**kwargs)
        self._project_root = project_root

    def compose(self) -> ComposeResult:
        yield JournalScreen(project_root=str(self._project_root), id="journal-screen")


def test_initial_mount_populates_and_captures_mtime(tmp_path: Path):
    journal = tmp_path / "trained_data" / "trade_journal_rl.json"
    _write_journal(journal, [_make_trade("t1"), _make_trade("t2", won=False, pl=-30.0)])

    async def _run():
        async with _Harness(project_root=tmp_path).run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.query_one("#journal-screen", JournalScreen)
            table = pilot.app.query_one("#journal-table", DataTable)
            assert table.row_count == 2
            assert screen._journal_mtime > 0.0

    asyncio.run(_run())


def test_update_from_snapshot_noop_when_unchanged(tmp_path: Path):
    journal = tmp_path / "trained_data" / "trade_journal_rl.json"
    _write_journal(journal, [_make_trade("t1")])

    async def _run():
        async with _Harness(project_root=tmp_path).run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.query_one("#journal-screen", JournalScreen)
            table = pilot.app.query_one("#journal-table", DataTable)
            mtime_before = screen._journal_mtime
            rows_before = table.row_count

            # No write happened — mtime unchanged — refresh should be a no-op.
            screen.update_from_snapshot(snap=None)  # snap is unused
            await pilot.pause()

            assert table.row_count == rows_before
            assert screen._journal_mtime == mtime_before

    asyncio.run(_run())


def test_update_from_snapshot_reloads_after_write(tmp_path: Path):
    journal = tmp_path / "trained_data" / "trade_journal_rl.json"
    _write_journal(journal, [_make_trade("t1")])

    async def _run():
        async with _Harness(project_root=tmp_path).run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.query_one("#journal-screen", JournalScreen)
            table = pilot.app.query_one("#journal-table", DataTable)
            assert table.row_count == 1

            # Simulate a trade close: rewrite with two entries, advance mtime.
            time.sleep(0.05)  # ensure mtime strictly advances on coarse fs clocks
            _write_journal(journal, [_make_trade("t1"), _make_trade("t2")])

            screen.update_from_snapshot(snap=None)
            await pilot.pause()

            # Auto-refresh picked up the new entry AND did not duplicate the old one.
            assert table.row_count == 2

    asyncio.run(_run())


def test_update_from_snapshot_no_crash_when_files_missing(tmp_path: Path):
    # No journal file, no learnings file — refresh should swallow OSError cleanly.
    async def _run():
        async with _Harness(project_root=tmp_path).run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.query_one("#journal-screen", JournalScreen)
            # Should not raise. Mtimes stay at 0; no reload occurs.
            screen.update_from_snapshot(snap=None)
            await pilot.pause()
            assert screen._journal_mtime == 0.0
            assert screen._learnings_mtime == 0.0

    asyncio.run(_run())
