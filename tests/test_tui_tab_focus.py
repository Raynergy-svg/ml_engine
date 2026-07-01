"""B1/B2: switching a tab moves focus into that tab's content.

Before the fix, HeaderBar.can_focus=True retained focus after mount and
action_switch_tab never moved focus into the tab, so the tab-container
BINDINGS (Inbox a/r/e/s/v, Trades c, Config s/q/r, Rules r/g) never
participated in key resolution — the keys were dead. on_tabbed_content_tab_
activated now focuses the first focusable descendant of the active pane, so
its ancestor screen's BINDINGS resolve.

No mocks: real BuddyApp(live=False) via Textual Pilot; we assert on the real
focus chain.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from src.tui.app import BuddyApp

_THEME = Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"


class _App(BuddyApp):
    CSS_PATH = str(_THEME)

    def __init__(self) -> None:
        super().__init__(live=False)


async def _switch(pilot, app, fkey: str, tab: str) -> None:
    await pilot.press(fkey)
    for _ in range(40):
        await pilot.pause()
        if app._active_tab_id() == tab:
            return
    raise AssertionError(f"tab did not switch to {tab!r}; got {app._active_tab_id()!r}")


def _focus_within(app, screen_id: str) -> bool:
    """True if the app's focused widget is the screen #screen_id or a descendant
    of it (i.e. that screen's BINDINGS are in the focus chain)."""
    focused = app.focused
    if focused is None:
        return False
    target = app.query_one(f"#{screen_id}")
    return focused is target or target in focused.ancestors


def test_inbox_tab_focus_enables_screen_bindings():
    app = _App()

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _switch(pilot, app, "f2", "inbox")
            assert _focus_within(app, "inbox-screen"), (
                f"focus not inside inbox-screen; focused={app.focused!r}"
            )
            # The focused widget must actually be focusable (sanity).
            assert app.focused is not None and app.focused.focusable

    asyncio.run(_run())


def test_trades_tab_focus_enables_close_binding():
    app = _App()

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _switch(pilot, app, "f3", "trades")
            assert _focus_within(app, "trades-screen"), (
                f"focus not inside trades-screen; focused={app.focused!r}"
            )

    asyncio.run(_run())
