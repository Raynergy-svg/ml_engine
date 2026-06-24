"""Trades `c` halt-aware confirm (front-end defense-in-depth).

When the bot is halted, the close-trade confirm must carry an explicit
halt-aware warning ("System is HALTED — close position anyway?") before the
operator can confirm. This modal does NOT call close_trade — it only gates the
confirm; the deterministic code-layer halt guard lives in the backend.

These tests verify:
  1. halted=True  -> the halt warning + "Close anyway" button are rendered.
  2. halted=False -> no halt warning; normal "Confirm Close" button.
  3. TradesScreen.action_close_selected_trade forwards its _halted state (set
     from DashboardSnapshot.halted) into the modal.

No mocks: real TradeCloseModal/TradesScreen mounted via Textual Pilot; the
push_screen capture is a real subclass override (the project's no-mock pattern).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual.app import App
from textual.widgets import Button, Label

from src.tui.data_provider import DashboardSnapshot
from src.tui.screens.trades_screen import TradeCloseModal, TradesScreen, TradeRow

_THEME = Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"


def _trade() -> TradeRow:
    return TradeRow(
        pair="EUR_USD", direction="BUY", pnl_pips=12.0,
        entry_price=1.10, sl_price=1.09, tp_price=1.12,
        quantity=10000, trade_id="t-1",
    )


class _ModalApp(App):
    CSS_PATH = str(_THEME)

    def __init__(self, halted: bool) -> None:
        super().__init__()
        self._halted = halted

    async def on_mount(self) -> None:
        await self.push_screen(TradeCloseModal(_trade(), halted=self._halted))


def _confirm_label(app) -> str:
    for btn in app.screen.query(Button):
        if btn.id == "close-confirm":
            return str(btn.label)
    return ""


def test_halted_close_modal_shows_halt_warning():
    app = _ModalApp(halted=True)

    async def _run() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            warnings = [
                str(lbl.render()) for lbl in app.screen.query(Label)
                if lbl.id == "close-halt-warning"
            ]
            assert warnings, "halt warning label missing when halted"
            assert "HALTED" in warnings[0]
            assert "anyway" in _confirm_label(app).lower()

    asyncio.run(_run())


def test_unhalted_close_modal_has_no_halt_warning():
    app = _ModalApp(halted=False)

    async def _run() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            halt_labels = [
                lbl for lbl in app.screen.query(Label) if lbl.id == "close-halt-warning"
            ]
            assert not halt_labels, "halt warning shown when not halted"
            assert "Confirm Close" in _confirm_label(app)

    asyncio.run(_run())


class _CaptureApp(App):
    """Hosts a TradesScreen and captures the modal handed to push_screen."""

    CSS_PATH = str(_THEME)

    def __init__(self) -> None:
        super().__init__()
        self.captured_modal: Any = None

    def compose(self):
        yield TradesScreen(id="trades-screen")

    def push_screen(self, screen, *args, **kwargs):  # type: ignore[override]
        if isinstance(screen, TradeCloseModal):
            self.captured_modal = screen
            return None  # don't actually mount the modal in this capture test
        return super().push_screen(screen, *args, **kwargs)


def test_action_close_forwards_halt_state_to_modal():
    app = _CaptureApp()

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.query_one("#trades-screen", TradesScreen)
            # Feed a halted snapshot carrying one open trade.
            snap = DashboardSnapshot()
            snap.halted = True
            snap.trades = [_trade()]
            screen.update_from_snapshot(snap)
            await pilot.pause()
            assert screen._halted is True
            screen._selected_index = 0
            screen.action_close_selected_trade()
            await pilot.pause()
            assert app.captured_modal is not None, "no TradeCloseModal pushed"
            assert app.captured_modal._halted is True, (
                "halt state not forwarded into TradeCloseModal"
            )

    asyncio.run(_run())
