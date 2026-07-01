"""Part B: the operator_override is set ONLY on the explicit confirmed close.

Backend close_trade now blocks autonomous closes while halted unless
operator_override=True. The front end must pass operator_override=True ONLY on
the genuine operator-confirmed close (the one behind the halt-aware
TradeCloseModal) — never on an automatic/default path.

Proves:
  * confirm (modal -> True) -> _do_close_trade(operator_override=True)
  * cancel (modal -> False) -> _do_close_trade NOT called (no override)
  * _do_close_trade default is False, and forwards the flag to close_trade
  * operator_override=True appears exactly ONCE in the module — in the
    confirmed-callback path.

No mocks: real TradesScreen subclass override (records args), real Pilot mount,
real TradeCloseModal; static checks via inspect.
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from textual.app import App

import src.tui.screens.trades_screen as ts_mod
from src.tui.screens.trades_screen import TradeCloseModal, TradesScreen, TradeRow

_THEME = Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"


def _trade() -> TradeRow:
    return TradeRow(
        pair="EUR_USD", direction="BUY", pnl_pips=5.0,
        entry_price=1.10, sl_price=1.09, tp_price=1.12,
        quantity=10000, trade_id="t-1",
    )


class _RecTrades(TradesScreen):
    """Real subclass that records _do_close_trade args instead of running the
    worker / hitting the broker (no mock — a genuine method override)."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.close_calls: list[tuple[str, bool]] = []

    def _do_close_trade(self, trade_id, operator_override=False):  # type: ignore[override]
        self.close_calls.append((trade_id, operator_override))


class _Host(App):
    CSS_PATH = str(_THEME)

    def compose(self):
        yield _RecTrades(id="trades-screen")


def _run(coro_factory) -> None:
    app = _Host()

    async def _main() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.query_one("#trades-screen", _RecTrades)
            screen._trades = [_trade()]
            screen._selected_index = 0
            screen._halted = True  # halted: the override is what lets it through
            await coro_factory(pilot, app, screen)

    asyncio.run(_main())


def test_confirmed_close_passes_operator_override_true():
    def factory(pilot, app, screen):
        async def _inner():
            screen.action_close_selected_trade()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, TradeCloseModal)
            modal.dismiss(True)          # operator clicks "Close anyway"
            await pilot.pause()
            assert screen.close_calls == [("t-1", True)]
        return _inner()
    _run(factory)


def test_cancelled_close_sets_no_override_and_no_close():
    def factory(pilot, app, screen):
        async def _inner():
            screen.action_close_selected_trade()
            await pilot.pause()
            app.screen.dismiss(False)    # operator cancels
            await pilot.pause()
            assert screen.close_calls == []   # nothing closed, no override
        return _inner()
    _run(factory)


def test_do_close_trade_default_is_false():
    sig = inspect.signature(TradesScreen._do_close_trade)
    assert sig.parameters["operator_override"].default is False


def test_do_close_trade_forwards_override_to_close_trade():
    src = inspect.getsource(TradesScreen._do_close_trade)
    assert "operator_override=operator_override" in src


def test_override_true_set_in_exactly_one_place():
    # The ONLY operator_override=True in the whole screen module is the
    # confirmed-callback path — no autonomous path can set it.
    module_src = inspect.getsource(ts_mod)
    assert module_src.count("operator_override=True") == 1
    confirm_src = inspect.getsource(TradesScreen.action_close_selected_trade)
    assert confirm_src.count("operator_override=True") == 1
