"""Light Pilot tests for TradeSearchModal — modal mounts, filters, dispatches."""
from __future__ import annotations

import pytest

from src.tui.screens.trade_search_modal import TradeSearchModal, _row_cells
from src.tui.trade_search_index import TradeSearchIndex


def _mk(**over):
    base = {
        "trade_id": "100",
        "pair": "EUR_USD",
        "direction": "LONG",
        "timestamp": "2026-04-15T10:00:00+00:00",
        "regime": {"volatility_regime": "NORMAL"},
        "outcome": {"trade_won": True, "realized_pl": 150.0},
    }
    base.update(over)
    return base


class TestRowCells:

    def test_win_row(self):
        cells = _row_cells(_mk())
        assert cells[0] == "100"
        assert "EUR/USD" in cells[2]
        assert cells[3] == "LONG"
        assert "win" in cells[4]
        assert "+$150" in cells[5]
        assert cells[6] == "NORMAL"

    def test_loss_row(self):
        cells = _row_cells(_mk(outcome={"trade_won": False, "realized_pl": -82.0}))
        assert "loss" in cells[4]
        assert "-$82" in cells[5]

    def test_open_row(self):
        cells = _row_cells(_mk(outcome=None))
        assert cells[4] == "open"
        assert cells[5] == "—"

    def test_handles_missing_regime(self):
        cells = _row_cells(_mk(regime=None))
        assert cells[6] == ""


class TestModalLifecycle:

    @pytest.mark.asyncio
    async def test_modal_dismisses_with_trade_id_on_enter(self):
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        captured: list[str | None] = []
        idx = TradeSearchIndex.from_entries([_mk(trade_id="100"), _mk(trade_id="1220", pair="EUR_AUD")])

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield Static("host")

            def on_mount(self) -> None:
                self.push_screen(TradeSearchModal(idx), captured.append)

        async with _Host().run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            await pilot.press(*"1220")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert captured == ["1220"]

    @pytest.mark.asyncio
    async def test_modal_dismisses_with_none_on_escape(self):
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        captured: list[str | None] = []
        idx = TradeSearchIndex.from_entries([_mk(trade_id="100")])

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield Static("host")

            def on_mount(self) -> None:
                self.push_screen(TradeSearchModal(idx), captured.append)

        async with _Host().run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert captured == [None]
