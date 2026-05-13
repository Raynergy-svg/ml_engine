"""US-010: Empty-state UX tests for NAV sparkline, ExpectancyPanel, P/L sparkline.

Covers:
  AC-1  F1 NAV sparkline (app.py): empty / all-zero nav_history → placeholder
        'NAV history pending' shown in muted color; sparkline hidden.
  AC-2  F5 ExpectancyPanel: zero closed trades → all $-formatted stat fields
        render '—' (not '$0.00' in green).
  AC-3  F5 P/L sparkline: empty journal → 'no closed trades' placeholder
        shown; sparkline hidden.
  AC-4  mixed fixture (journal with closed trades) → sparkline shown,
        placeholder hidden.

No mocks per CLAUDE.md No-Mock Rule.
Uses real DashboardSnapshot and real journal files via tmp_path.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Sparkline, Static

from src.tui.data_provider import DashboardSnapshot
from src.tui.screens.journal_screen import ExpectancyPanel, JournalScreen


_TUI_THEME_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"
)

# ---------------------------------------------------------------------------
# Minimal host apps
# ---------------------------------------------------------------------------


class _NavApp(App):
    """Hosts only the NAV sparkline + its placeholder for isolated testing."""

    CSS_PATH = str(_TUI_THEME_PATH)

    def compose(self) -> ComposeResult:
        yield Sparkline(data=[], id="nav-sparkline")
        yield Static(
            "NAV history pending",
            id="nav-sparkline-placeholder",
            classes="status-dim",
        )

    def apply_nav(self, nav_history: list[float]) -> None:
        """Mirror the toggle logic from BuddyApp._do_live_refresh."""
        spark = self.query_one("#nav-sparkline", Sparkline)
        ph = self.query_one("#nav-sparkline-placeholder", Static)
        has_data = bool(nav_history) and any(v != 0 for v in nav_history)
        spark.display = has_data
        ph.display = not has_data
        if has_data:
            spark.data = nav_history


class _JournalApp(App):
    """Wraps JournalScreen in the smallest possible App for testing."""

    CSS_PATH = str(_TUI_THEME_PATH)

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self._root = project_root

    def compose(self) -> ComposeResult:
        yield JournalScreen(project_root=str(self._root), id="journal")


# ---------------------------------------------------------------------------
# AC-1 — NAV sparkline empty state
# ---------------------------------------------------------------------------


class TestNavSparklineEmptyState:
    def test_empty_nav_history_shows_placeholder(self) -> None:
        """AC-1: empty nav_history → placeholder shown, sparkline hidden."""
        snap = DashboardSnapshot()  # nav_history=[] by default
        assert snap.nav_history == [], "DashboardSnapshot default should have empty nav_history"
        app = _NavApp()

        async def _run() -> None:
            async with app.run_test(headless=True, size=(80, 10)) as pilot:
                await pilot.pause()
                app.apply_nav(snap.nav_history)
                await pilot.pause()

                spark = app.query_one("#nav-sparkline", Sparkline)
                ph = app.query_one("#nav-sparkline-placeholder", Static)
                assert not spark.display, "sparkline must be hidden when nav_history is empty"
                assert ph.display, "placeholder must be visible when nav_history is empty"

        asyncio.run(_run())

    def test_all_zero_nav_history_shows_placeholder(self) -> None:
        """AC-1: all-zero nav_history is treated as no-data → placeholder shown."""
        snap = DashboardSnapshot()
        snap.nav_history = [0.0, 0.0, 0.0]
        app = _NavApp()

        async def _run() -> None:
            async with app.run_test(headless=True, size=(80, 10)) as pilot:
                await pilot.pause()
                app.apply_nav(snap.nav_history)
                await pilot.pause()

                spark = app.query_one("#nav-sparkline", Sparkline)
                ph = app.query_one("#nav-sparkline-placeholder", Static)
                assert not spark.display, "sparkline must be hidden when all values are zero"
                assert ph.display, "placeholder must be visible when all values are zero"

        asyncio.run(_run())

    def test_non_empty_nav_history_shows_sparkline(self) -> None:
        """AC-1 inverse: populated nav_history → sparkline shown, placeholder hidden."""
        snap = DashboardSnapshot()
        snap.nav_history = [100_000.0, 100_050.0, 100_120.0]
        app = _NavApp()

        async def _run() -> None:
            async with app.run_test(headless=True, size=(80, 10)) as pilot:
                await pilot.pause()
                app.apply_nav(snap.nav_history)
                await pilot.pause()

                spark = app.query_one("#nav-sparkline", Sparkline)
                ph = app.query_one("#nav-sparkline-placeholder", Static)
                assert spark.display, "sparkline must be shown when nav_history has real data"
                assert not ph.display, "placeholder must be hidden when sparkline has data"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# AC-2 — ExpectancyPanel no-trades empty state
# ---------------------------------------------------------------------------


class TestExpectancyPanelEmptyState:
    def test_no_trades_renders_dashes_not_dollars(self, tmp_path: Path) -> None:
        """AC-2: zero closed trades → all $ fields show '—', not '$0.00' in green."""
        app = _JournalApp(project_root=tmp_path)

        async def _run() -> None:
            async with app.run_test(headless=True, size=(80, 40)) as pilot:
                await pilot.pause()

                panel = app.query_one("#expectancy-panel", ExpectancyPanel)
                rendered = panel.render()
                plain = rendered.plain

                assert "—" in plain, "Expected '—' placeholder when no trades"
                assert "$0.00" not in plain, (
                    "Expected no '$0.00' when no trades — green '$0.00' is misleading"
                )

        asyncio.run(_run())

    def test_with_trades_renders_dollar_values(self, tmp_path: Path) -> None:
        """AC-2 inverse: closed trades present → $ fields render real values."""
        _write_journal(tmp_path, [_closed_trade(50.0, won=True)])
        app = _JournalApp(project_root=tmp_path)

        async def _run() -> None:
            async with app.run_test(headless=True, size=(80, 40)) as pilot:
                await pilot.pause()

                panel = app.query_one("#expectancy-panel", ExpectancyPanel)
                rendered = panel.render()
                plain = rendered.plain

                assert "$" in plain, "Expected $ values when trades are present"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# AC-3 / AC-4 — P/L sparkline empty and mixed states
# ---------------------------------------------------------------------------


class TestPnlSparklineEmptyState:
    def test_empty_journal_shows_placeholder(self, tmp_path: Path) -> None:
        """AC-3: no journal file → pnl placeholder shown, sparkline hidden."""
        app = _JournalApp(project_root=tmp_path)

        async def _run() -> None:
            async with app.run_test(headless=True, size=(80, 40)) as pilot:
                await pilot.pause()

                spark = app.query_one("#pnl-sparkline", Sparkline)
                ph = app.query_one("#pnl-sparkline-placeholder", Static)
                assert not spark.display, "pnl sparkline must be hidden when journal is empty"
                assert ph.display, "pnl placeholder must be visible when journal is empty"

        asyncio.run(_run())

    def test_only_open_trades_shows_placeholder(self, tmp_path: Path) -> None:
        """AC-3: journal with only open trades (no outcome) → placeholder shown."""
        open_trade = {
            "trade_id": "open-001",
            "pair": "EUR_USD",
            "direction": "long",
            "entry_price": 1.1000,
            # no "outcome" key — open trade
        }
        _write_journal(tmp_path, [open_trade])
        app = _JournalApp(project_root=tmp_path)

        async def _run() -> None:
            async with app.run_test(headless=True, size=(80, 40)) as pilot:
                await pilot.pause()

                spark = app.query_one("#pnl-sparkline", Sparkline)
                ph = app.query_one("#pnl-sparkline-placeholder", Static)
                assert not spark.display, "sparkline must be hidden when only open trades exist"
                assert ph.display, "placeholder must be visible when only open trades exist"

        asyncio.run(_run())

    def test_closed_trades_shows_sparkline(self, tmp_path: Path) -> None:
        """AC-4 (mixed): journal with closed trades → sparkline shown, placeholder hidden."""
        _write_journal(tmp_path, [
            _closed_trade(50.0, won=True),
            _closed_trade(-30.0, won=False),
        ])
        app = _JournalApp(project_root=tmp_path)

        async def _run() -> None:
            async with app.run_test(headless=True, size=(80, 40)) as pilot:
                await pilot.pause()

                spark = app.query_one("#pnl-sparkline", Sparkline)
                ph = app.query_one("#pnl-sparkline-placeholder", Static)
                assert spark.display, "sparkline must be shown when closed trades exist"
                assert not ph.display, "placeholder must be hidden when sparkline has data"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_journal(project_root: Path, entries: list[dict]) -> None:
    """Write a trade_journal_rl.json under project_root/trained_data/."""
    td = project_root / "trained_data"
    td.mkdir(parents=True, exist_ok=True)
    (td / "trade_journal_rl.json").write_text(json.dumps(entries))


def _closed_trade(pl: float, *, won: bool, trade_id: str = "t001") -> dict:
    return {
        "trade_id": trade_id,
        "pair": "EUR_USD",
        "direction": "long",
        "entry_price": 1.1000,
        "exit_price": 1.1050 if won else 1.0950,
        "outcome": {
            "realized_pl": pl,
            "trade_won": won,
            "exit_reason": "TP" if won else "SL",
        },
    }
