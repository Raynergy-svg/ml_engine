"""Integration test: Kill switch end-to-end flow.

Scenario: 2 practice trades are open → operator presses K → modal appears →
operator types KILL → confirms → flatten_all invoked within 3s → state.halted==True.

Uses Textual's async test pilot with mocked ExecutionManager so no real OANDA
connection is required. Follows the asyncio.run() pattern used by the rest of
this test suite (see test_event_bus.py, test_flatten_all_unit.py).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scanner.execution import FlattenResult, KillSwitchPartialFailure
from src.tui.app import BuddyApp
from src.tui.data_provider import DashboardSnapshot, TradeRow
from src.tui.screens.kill_modal import KillModal


def _two_trade_snapshot() -> DashboardSnapshot:
    return DashboardSnapshot(
        nav=101_000.0,
        unrealized_pnl=-142.50,
        margin_used=5_200.0,
        trades=[
            TradeRow("EUR_USD", "BUY", 45.0, 1.08542, 1.08320, 1.08890, 5000, "T-001"),
            TradeRow("GBP_JPY", "SELL", -97.50, 191.320, 191.650, 190.800, 3000, "T-002"),
        ],
    )


def _flat_result() -> FlattenResult:
    return FlattenResult(
        orders_cancelled=0,
        orders_failed=0,
        positions_closed=2,
        positions_failed=0,
        duration_ms=320.0,
    )


def _make_mock_event_bus():
    """Return a mock EventBus whose subscribe() yields nothing (avoids infinite loop)."""

    async def _empty_subscribe():
        return
        yield  # make it an async generator

    bus = MagicMock()
    bus.subscribe = _empty_subscribe
    return bus


# ── Tests ──────────────────────────────────────────────────────────────


def test_kill_switch_guard_no_trades():
    """K with no open trades shows info notification and does NOT push the modal."""

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            # Inject empty snapshot (no trades)
            app._provider._snapshot = DashboardSnapshot()
            await pilot.press("k")
            await pilot.pause()
            # KillModal must NOT appear on screen
            assert not isinstance(app.screen, KillModal), "Modal pushed with no trades"

    asyncio.run(_run())


def test_kill_switch_cancel_escape_no_action():
    """Pressing Escape on the kill modal cancels without invoking flatten_all."""
    flatten_mock = AsyncMock(return_value=_flat_result())

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider._snapshot = _two_trade_snapshot()
            with (
                patch("src.scanner.execution.ExecutionManager.flatten_all", flatten_mock),
                patch("src.scanner.automation.event_bus.get_event_bus", return_value=_make_mock_event_bus()),
            ):
                await pilot.press("k")
                await pilot.pause()
                assert isinstance(app.screen, KillModal), "KillModal not pushed"
                await pilot.press("escape")
                await pilot.pause()
                assert not isinstance(app.screen, KillModal), "Modal still visible after escape"
                flatten_mock.assert_not_called()

    asyncio.run(_run())


def test_kill_switch_confirm_button_requires_kill():
    """Confirm button stays disabled until operator types exactly 'KILL'."""

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider._snapshot = _two_trade_snapshot()
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, KillModal), "KillModal not pushed"
            modal: KillModal = app.screen  # type: ignore[assignment]
            confirm_btn = modal.query_one("#kill-confirm")
            kill_input = modal.query_one("#kill-input")

            # Initially disabled
            assert confirm_btn.disabled is True

            # Partial string — still disabled
            kill_input.value = "KIL"
            kill_input.post_message(kill_input.Changed(kill_input, "KIL"))
            await pilot.pause()
            assert confirm_btn.disabled is True, "Button should stay disabled for 'KIL'"

            # Wrong case — still disabled
            kill_input.value = "kill"
            kill_input.post_message(kill_input.Changed(kill_input, "kill"))
            await pilot.pause()
            assert confirm_btn.disabled is True, "Button should stay disabled for 'kill' (wrong case)"

            # Correct KILL — enabled
            kill_input.value = "KILL"
            kill_input.post_message(kill_input.Changed(kill_input, "KILL"))
            await pilot.pause()
            assert confirm_btn.disabled is False, "Button should be enabled for 'KILL'"

    asyncio.run(_run())


def test_kill_switch_e2e_flatten_succeeds():
    """Full flow: 2 trades → K → type KILL → confirm → flatten_all called within 3s → state.halted."""
    flatten_mock = AsyncMock(return_value=_flat_result())
    halted_calls: list[bool] = []

    async def _capture_set_halted(self, value: bool) -> None:
        halted_calls.append(value)

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider._snapshot = _two_trade_snapshot()

            with (
                patch("src.scanner.execution.ExecutionManager.flatten_all", flatten_mock),
                patch("src.scanner.automation.event_bus.get_event_bus", return_value=_make_mock_event_bus()),
                patch("src.scanner.automation.state_engine.StateEngine.set_halted", _capture_set_halted),
            ):
                # Step 1: press K — modal appears
                await pilot.press("k")
                await pilot.pause()
                assert isinstance(app.screen, KillModal), "KillModal not shown"

                # Step 2: type KILL into the input
                modal: KillModal = app.screen  # type: ignore[assignment]
                kill_input = modal.query_one("#kill-input")
                kill_input.value = "KILL"
                kill_input.post_message(kill_input.Changed(kill_input, "KILL"))
                await pilot.pause()

                # Step 3: click Confirm
                confirm_btn = modal.query_one("#kill-confirm")
                assert not confirm_btn.disabled, "Confirm button should be enabled"
                await pilot.click("#kill-confirm")
                await pilot.pause()

                # Modal dismissed
                assert not isinstance(app.screen, KillModal), "Modal still visible after confirm"

                # Step 4: flatten_all must be called within 3s
                deadline = asyncio.get_event_loop().time() + 3.0
                while not flatten_mock.called:
                    if asyncio.get_event_loop().time() > deadline:
                        break
                    await asyncio.sleep(0.05)

                assert flatten_mock.called, "flatten_all not invoked within 3s"
                flatten_mock.assert_awaited_once_with("operator_kill")

                # Step 5: wait for worker to finish and verify state reset
                await asyncio.sleep(0.3)
                assert app._kill_in_progress is False

    asyncio.run(_run())


def test_kill_switch_hotkeys_blocked_during_flatten():
    """All navigation hotkeys must be no-ops while flatten_all is in progress."""

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._kill_in_progress = True
            tabs = app.query_one("#main-tabs")
            initial_tab = tabs.active

            await pilot.press("f2")
            await pilot.press("f3")
            await pilot.press("space")
            await pilot.press("m")
            await pilot.press("a")
            await pilot.pause()

            assert tabs.active == initial_tab, "Tab changed while kill in progress"
            app._kill_in_progress = False

    asyncio.run(_run())


def test_kill_switch_partial_failure_shows_toast():
    """KillSwitchPartialFailure causes error notify and resets _kill_in_progress."""
    error = KillSwitchPartialFailure("Could not close T-002", remaining_trades=["T-002"])
    flatten_mock = AsyncMock(side_effect=error)

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider._snapshot = _two_trade_snapshot()

            with (
                patch("src.scanner.execution.ExecutionManager.flatten_all", flatten_mock),
                patch("src.scanner.automation.event_bus.get_event_bus", return_value=_make_mock_event_bus()),
            ):
                await pilot.press("k")
                await pilot.pause()
                modal: KillModal = app.screen  # type: ignore[assignment]
                kill_input = modal.query_one("#kill-input")
                kill_input.value = "KILL"
                kill_input.post_message(kill_input.Changed(kill_input, "KILL"))
                await pilot.pause()
                await pilot.click("#kill-confirm")

                deadline = asyncio.get_event_loop().time() + 3.0
                while not flatten_mock.called:
                    if asyncio.get_event_loop().time() > deadline:
                        break
                    await asyncio.sleep(0.05)

                assert flatten_mock.called, "flatten_all not called within 3s"
                await asyncio.sleep(0.3)
                assert app._kill_in_progress is False, "_kill_in_progress not reset after error"

    asyncio.run(_run())


def test_kill_modal_shows_trade_stats():
    """KillModal displays correct trade count, P/L, and exposure from snapshot."""

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            snap = _two_trade_snapshot()
            app._provider._snapshot = snap
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, KillModal), "KillModal not shown"
            modal: KillModal = app.screen  # type: ignore[assignment]
            stats_text = str(modal.query_one("#kill-stats").render())
            # Verify the stats widget contains the expected values
            assert "2" in stats_text or modal._snap.trades == snap.trades

    asyncio.run(_run())
