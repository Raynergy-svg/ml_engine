"""US-005 hotkey routing tests — real BuddyApp Pilot, no mocks.

Per CLAUDE.md No-Mock Rule, the test uses Textual's ``run_test`` Pilot
harness against a real ``BuddyApp`` subclass that records the calls we
care about. Specifically:

* ``_emit_hotkey_routed(key, tab)``  → captured so we can assert the
  global handler took the early-return path.
* ``_emit_signal_veto`` (module-level helper) → captured via a
  ``BuddyApp.push_screen`` override that records the modal type. The
  ``action_supervisor_abort`` early-return must NOT push the
  ``AbortConfirmModal`` (the only path that can publish
  ``control.signal_veto``).
* ``src.scanner.automation.event_bus.EventBus.publish`` → asserted
  empty for the global ``signal_veto`` channel. We use a real
  ``EventBus`` singleton and record every ``publish`` call by reading
  its ``replay_buffer()`` afterwards.

Tabs are switched via ``pilot.press("f2"|"f3"|"f7")`` (the live
``switch_tab`` bindings). Keys (a/c/r) are then fired via
``pilot.press("a"|"c"|"r")`` and we assert that:

* The global handler's brain-feed marker fired with the right tab id
  (i.e. early-return was taken) OR did not fire at all (i.e. the
  screen-local binding intercepted first — equally acceptable; either
  way the global handler did NOT publish).
* ``EventBus.replay_buffer()`` contains zero ``control.signal_veto``
  events at the end of the run.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import pytest

from src.scanner.automation.event_bus import (
    EventBus,
    get_event_bus,
    reset_event_bus,
)
from src.tui.app import BuddyApp


_TUI_THEME_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"
)


@pytest.fixture
def fresh_bus() -> Iterator[EventBus]:
    """Reset the EventBus singleton between tests for isolation."""
    reset_event_bus()
    bus = get_event_bus()
    yield bus
    reset_event_bus()


class _RoutingApp(BuddyApp):
    """Real BuddyApp subclass that records hotkey-routing observability calls.

    Captures both:
      * every ``_emit_hotkey_routed(key, tab)`` invocation so the test can
        assert the global early-return path was taken with the correct
        (key, tab) tuple;
      * every ``push_screen`` call (modal type only) so the test can
        verify ``action_supervisor_abort`` did NOT push ``AbortConfirmModal``
        when the inbox tab owned ``a``.

    Both are real method overrides — no ``unittest.mock`` involved.

    CSS_PATH is repointed at the real ``src/tui/theme.tcss`` because
    Textual resolves relative CSS paths against the *defining module's*
    directory; the subclass would otherwise look in ``tests/``.
    """

    CSS_PATH = str(_TUI_THEME_PATH)

    def __init__(self) -> None:
        super().__init__(live=False)
        self.routed: List[Tuple[str, str]] = []
        self.pushed_modals: List[str] = []

    def _emit_hotkey_routed(self, key: str, tab: str) -> None:  # type: ignore[override]
        self.routed.append((key, tab))
        super()._emit_hotkey_routed(key, tab)

    def push_screen(self, screen: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        try:
            self.pushed_modals.append(type(screen).__name__)
        except Exception:
            self.pushed_modals.append(repr(screen))
        return super().push_screen(screen, *args, **kwargs)


async def _switch_to_tab(pilot, fkey: str, expected_tab: str, app: BuddyApp) -> None:
    """Press an F-key and wait until ``_active_tab_id()`` reflects the target."""
    await pilot.press(fkey)
    deadline = asyncio.get_event_loop().time() + 1.0
    while asyncio.get_event_loop().time() < deadline:
        await pilot.pause()
        if app._active_tab_id() == expected_tab:
            return
    raise AssertionError(
        f"tab did not switch to {expected_tab!r} after pressing {fkey!r}; "
        f"active tab is {app._active_tab_id()!r}"
    )


def _bus_has_signal_veto(bus: EventBus) -> bool:
    for evt in bus.replay_buffer():
        if evt.get("event_type") == "control.signal_veto":
            return True
    return False


class TestHotkeyRouting:
    """AC-1, AC-2, AC-3, AC-4, AC-5 — hotkey routing precedence."""

    def test_a_on_inbox_tab_does_not_publish_signal_veto(self, fresh_bus):
        """AC-1: pressing `a` on F2 Inbox tab does NOT publish control.signal_veto.

        Mechanism: ``BuddyApp.action_supervisor_abort`` checks
        ``_active_tab_id() == "inbox"`` and early-returns before ever
        looking up the event_bus — so no signal_veto event can be emitted.
        Either the screen-local Approve handler intercepts first or the
        global handler early-returns. Both outcomes satisfy the AC.
        """
        host = _RoutingApp()

        async def _run() -> None:
            async with host.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                host.routed.clear()
                host.pushed_modals.clear()

                await _switch_to_tab(pilot, "f2", "inbox", host)
                await pilot.press("a")
                await pilot.pause()

                assert not _bus_has_signal_veto(fresh_bus), (
                    "control.signal_veto MUST NOT be published while inbox tab owns `a`"
                )
                assert "AbortConfirmModal" not in host.pushed_modals, (
                    "AbortConfirmModal MUST NOT be pushed while inbox tab is active"
                )

        asyncio.run(_run())

    def test_action_supervisor_abort_early_returns_on_inbox(self, fresh_bus):
        """AC-1+AC-4: direct call to action_supervisor_abort while inbox active.

        Drives the action method directly (bypassing Textual's binding
        precedence) and asserts:
          * no AbortConfirmModal was pushed
          * no signal_veto was published
          * the brain-feed marker recorded (key='a', tab='inbox')
        """
        host = _RoutingApp()

        async def _run() -> None:
            async with host.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _switch_to_tab(pilot, "f2", "inbox", host)
                host.routed.clear()
                host.pushed_modals.clear()

                host.action_supervisor_abort()
                await pilot.pause()

                assert ("a", "inbox") in host.routed, (
                    f"expected ('a','inbox') in routed log; got {host.routed!r}"
                )
                assert "AbortConfirmModal" not in host.pushed_modals
                assert not _bus_has_signal_veto(fresh_bus)

        asyncio.run(_run())

    def test_action_copy_snapshot_early_returns_on_trades(self, fresh_bus, tmp_path):
        """AC-2+AC-4: action_copy_snapshot early-returns when F3 trades tab active.

        The screen-local TradesScreen.action_close_selected_trade owns `c`.
        Driving the action directly while the trades tab is active must
        record the (c, trades) brain marker and write no snapshot file.
        """
        host = _RoutingApp()

        async def _run() -> None:
            async with host.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _switch_to_tab(pilot, "f3", "trades", host)
                host.routed.clear()

                # Snapshot the .claude/snapshots dir state pre-action — the
                # early-return MUST NOT write any file even though
                # _dump_focused_widget / build_snapshot would normally do so.
                snapshots_dir = Path(".claude") / "snapshots"
                pre_files = (
                    {p.name for p in snapshots_dir.iterdir()}
                    if snapshots_dir.exists()
                    else set()
                )

                host.action_copy_snapshot()
                await pilot.pause()

                assert ("c", "trades") in host.routed, (
                    f"expected ('c','trades') in routed log; got {host.routed!r}"
                )
                post_files = (
                    {p.name for p in snapshots_dir.iterdir()}
                    if snapshots_dir.exists()
                    else set()
                )
                # Early-return prevents any new snapshot file.
                new_files = post_files - pre_files
                assert not new_files, (
                    f"action_copy_snapshot wrote files while trades tab active: {new_files}"
                )

        asyncio.run(_run())

    def test_action_refresh_rules_early_returns_on_inbox(self, fresh_bus):
        """AC-3+AC-4: action_refresh_rules defers to inbox's `r` (Reject)."""
        host = _RoutingApp()

        async def _run() -> None:
            async with host.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _switch_to_tab(pilot, "f2", "inbox", host)
                host.routed.clear()

                host.action_refresh_rules()
                await pilot.pause()

                assert ("r", "inbox") in host.routed, (
                    f"expected ('r','inbox') in routed log; got {host.routed!r}"
                )
                assert not _bus_has_signal_veto(fresh_bus)

        asyncio.run(_run())

    def test_action_refresh_rules_early_returns_on_rules(self, fresh_bus):
        """AC-3+AC-4: action_refresh_rules defers to rules's `r` (Refresh)."""
        host = _RoutingApp()

        async def _run() -> None:
            async with host.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _switch_to_tab(pilot, "f7", "rules", host)
                host.routed.clear()

                host.action_refresh_rules()
                await pilot.pause()

                assert ("r", "rules") in host.routed, (
                    f"expected ('r','rules') in routed log; got {host.routed!r}"
                )
                assert not _bus_has_signal_veto(fresh_bus)

        asyncio.run(_run())

    def test_action_refresh_rules_noop_on_other_tabs(self, fresh_bus):
        """AC-3 contract: on non-owning tabs the method is a deliberate no-op.

        Switching to F1 Overview and calling action_refresh_rules must NOT
        emit a brain-feed marker for any tab — the early-return checks
        only inbox/rules, and outside those branches the method falls
        through without observable effect.
        """
        host = _RoutingApp()

        async def _run() -> None:
            async with host.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _switch_to_tab(pilot, "f1", "overview", host)
                host.routed.clear()

                host.action_refresh_rules()
                await pilot.pause()

                # No brain marker for inbox or rules on the overview tab.
                assert ("r", "inbox") not in host.routed
                assert ("r", "rules") not in host.routed
                assert not _bus_has_signal_veto(fresh_bus)

        asyncio.run(_run())

    def test_global_handlers_still_work_outside_owning_tabs(self, fresh_bus):
        """Negative case: action_supervisor_abort does NOT early-return on overview.

        Switching to F1 Overview (which has no screen-local `a`) and
        calling action_supervisor_abort must NOT record an ('a', 'inbox')
        marker — the early-return branch is keyed on tab id and overview
        falls through to the normal abort flow.
        """
        host = _RoutingApp()

        async def _run() -> None:
            async with host.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await _switch_to_tab(pilot, "f1", "overview", host)
                host.routed.clear()

                host.action_supervisor_abort()
                await pilot.pause()

                # The hotkey-routed marker is keyed on the owning tab. On
                # overview, neither the inbox nor trades marker should
                # appear.
                assert ("a", "inbox") not in host.routed, (
                    "action_supervisor_abort must NOT take the inbox early-return path "
                    "while on overview tab"
                )

        asyncio.run(_run())
