"""Integration and unit tests for US-507: mode toggle dry_run ↔ live (hotkey M).

Scenarios covered:
  - dry_run → live: credential precheck blocks missing creds (no modal pushed)
  - dry_run → live: wrong confirmation text keeps confirm button disabled
  - dry_run → live: full flow — press M, type LIVE, confirm → mode == 'live', state strip LIVE
  - live → dry_run: immediate (no modal), mode == 'dry_run'
  - StateEngine persists mode correctly across the toggle

Uses asyncio.run() + Textual pilot — same pattern as test_kill_switch_e2e.py.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

from textual.app import App, ComposeResult

from src.tui.app import BuddyApp
from src.tui.data_provider import DashboardSnapshot, TradeRow
from src.tui.screens.mode_modal import ModeConfirmModal, check_oanda_credentials
from src.tui.widgets.state_strip import StateStrip


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _cred_env() -> dict[str, str]:
    return {"OANDA_API_TOKEN": "test-token-abcdef", "OANDA_ACCOUNT_ID": "001-001-999-001"}


def _no_cred_env() -> dict[str, str]:
    """Return a clean env with all OANDA credential keys removed."""
    keys = {"OANDA_API_TOKEN", "OANDA_API_KEY", "OANDA_ACCOUNT_ID"}
    return {k: v for k, v in os.environ.items() if k not in keys}


class _MockStateEngine:
    """In-memory StateEngine — avoids filesystem I/O during tests."""

    def __init__(self, initial_mode: str = "dry_run") -> None:
        self._mode = initial_mode
        self._paused = False
        self._halted = False

    def get_mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("dry_run", "live"):
            raise ValueError(f"Invalid mode '{mode}'")
        self._mode = mode

    def get_paused(self) -> bool:
        return self._paused

    def set_paused(self, v: bool) -> None:
        self._paused = v

    def get_halted(self) -> bool:
        return self._halted

    def load_state(self) -> dict:
        return {
            "mode": self._mode,
            "scanner_paused": self._paused,
            "halted": self._halted,
            "schema_version": 2,
        }


class _ModalTestApp(App):
    """Minimal test app that pushes ModeConfirmModal on mount for isolated UI tests."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.modal_result: bool | None = None

    def compose(self) -> ComposeResult:
        return iter([])  # no widgets — modal is pushed in on_mount

    def on_mount(self) -> None:
        def _capture(result: bool) -> None:
            self.modal_result = result

        self.push_screen(ModeConfirmModal(nav=100_000.0, open_trades=0), _capture)


# ─────────────────────────────────────────────────────────────────────
# Unit Tests: check_oanda_credentials
# ─────────────────────────────────────────────────────────────────────

class TestCheckOandaCredentials:
    def test_both_missing(self) -> None:
        with patch.dict(os.environ, _no_cred_env(), clear=True):
            ok, msg = check_oanda_credentials()
        assert not ok
        assert "OANDA_API_TOKEN" in msg
        assert "OANDA_ACCOUNT_ID" in msg

    def test_api_key_missing_account_present(self) -> None:
        env = {**_no_cred_env(), "OANDA_ACCOUNT_ID": "001"}
        with patch.dict(os.environ, env, clear=True):
            ok, msg = check_oanda_credentials()
        assert not ok
        assert "OANDA_API" in msg

    def test_account_id_missing_api_key_present(self) -> None:
        env = {**_no_cred_env(), "OANDA_API_TOKEN": "tok"}
        with patch.dict(os.environ, env, clear=True):
            ok, msg = check_oanda_credentials()
        assert not ok
        assert "OANDA_ACCOUNT_ID" in msg

    def test_both_present_via_api_token(self) -> None:
        env = {**_no_cred_env(), "OANDA_API_TOKEN": "mytoken123", "OANDA_ACCOUNT_ID": "001-001-123456-001"}
        with patch.dict(os.environ, env, clear=True):
            ok, msg = check_oanda_credentials()
        assert ok
        assert "001-001-123456-001" in msg

    def test_fallback_to_oanda_api_key(self) -> None:
        """OANDA_API_KEY accepted as fallback when OANDA_API_TOKEN absent."""
        env = {**_no_cred_env(), "OANDA_API_KEY": "key456", "OANDA_ACCOUNT_ID": "acct"}
        with patch.dict(os.environ, env, clear=True):
            ok, msg = check_oanda_credentials()
        assert ok


# ─────────────────────────────────────────────────────────────────────
# Unit Tests: ModeConfirmModal — input validation (via _ModalTestApp)
# ─────────────────────────────────────────────────────────────────────

def test_modal_confirm_disabled_by_default() -> None:
    """Confirm button starts disabled."""

    async def _run() -> None:
        app = _ModalTestApp()
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            from textual.widgets import Button
            modal: ModeConfirmModal = app.screen  # type: ignore[assignment]
            btn = modal.query_one("#mode-confirm", Button)
            assert btn.disabled, "Confirm button should be disabled by default"

    asyncio.run(_run())


def test_modal_confirm_enabled_on_exact_live() -> None:
    """Confirm button enables exactly when input value is 'LIVE'."""

    async def _run() -> None:
        app = _ModalTestApp()
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            from textual.widgets import Button, Input
            modal: ModeConfirmModal = app.screen  # type: ignore[assignment]
            inp = modal.query_one("#mode-input", Input)
            inp.value = "LIVE"
            inp.post_message(inp.Changed(inp, "LIVE"))
            await pilot.pause()
            btn = modal.query_one("#mode-confirm", Button)
            assert not btn.disabled, "Confirm button should be enabled for 'LIVE'"

    asyncio.run(_run())


def test_modal_wrong_confirmation_text_no_change() -> None:
    """Wrong / partial text keeps confirm button disabled (no change)."""

    async def _run() -> None:
        app = _ModalTestApp()
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            from textual.widgets import Button, Input
            modal: ModeConfirmModal = app.screen  # type: ignore[assignment]
            inp = modal.query_one("#mode-input", Input)

            for bad_value in ("live", "LIV", "LIVE ", " LIVE", "Live"):
                inp.value = bad_value
                inp.post_message(inp.Changed(inp, bad_value))
                await pilot.pause()
                btn = modal.query_one("#mode-confirm", Button)
                assert btn.disabled, f"Button should stay disabled for '{bad_value}'"

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────
# Unit Test: missing credentials → rejection toast, no modal
# ─────────────────────────────────────────────────────────────────────

def test_missing_credentials_rejection_no_modal() -> None:
    """dry_run → live blocked when credentials missing; ModeConfirmModal not pushed."""
    mock_se = _MockStateEngine(initial_mode="dry_run")

    async def _run() -> None:
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider._snapshot = DashboardSnapshot()

            with (
                patch("src.scanner.automation.state_engine.StateEngine", return_value=mock_se),
                patch.dict(os.environ, _no_cred_env(), clear=True),
            ):
                await pilot.press("m")
                await pilot.pause()

            # Modal must NOT appear
            assert not isinstance(app.screen, ModeConfirmModal), (
                "ModeConfirmModal pushed despite missing credentials"
            )
            # Mode must remain dry_run
            assert mock_se.get_mode() == "dry_run"

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────
# Integration Test: dry_run → LIVE (type LIVE) → dry_run (immediate M)
# ─────────────────────────────────────────────────────────────────────

def test_mode_toggle_dry_to_live_and_back() -> None:
    """
    Integration:
      1. Start in dry_run → press M → ModeConfirmModal appears.
      2. Type 'LIVE' → confirm → mode == 'live', state strip shows 'LIVE'.
      3. Press M again → mode == 'dry_run' immediately (no modal).
    """
    mock_se = _MockStateEngine(initial_mode="dry_run")
    mock_bus = MagicMock()

    async def _run() -> None:
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider._snapshot = DashboardSnapshot(nav=101_000.0, trades=[])

            with (
                patch("src.scanner.automation.state_engine.StateEngine", return_value=mock_se),
                patch("src.scanner.automation.event_bus.get_event_bus", return_value=mock_bus),
                patch.dict(os.environ, {**_no_cred_env(), **_cred_env()}),
            ):
                # ── Step 1: press M in dry_run → modal appears ──
                await pilot.press("m")
                await pilot.pause()
                assert isinstance(app.screen, ModeConfirmModal), (
                    "ModeConfirmModal not pushed from dry_run"
                )

                # ── Step 2: type LIVE into input, click confirm ──
                modal: ModeConfirmModal = app.screen  # type: ignore[assignment]
                from textual.widgets import Button, Input
                inp = modal.query_one("#mode-input", Input)
                inp.value = "LIVE"
                inp.post_message(inp.Changed(inp, "LIVE"))
                await pilot.pause()

                confirm_btn = modal.query_one("#mode-confirm", Button)
                assert not confirm_btn.disabled, "Confirm button should be enabled"
                await pilot.click("#mode-confirm")
                await pilot.pause()

                # Modal dismissed
                assert not isinstance(app.screen, ModeConfirmModal), (
                    "ModeConfirmModal still visible after confirm"
                )

                # mode == 'live'
                assert mock_se.get_mode() == "live", (
                    f"Expected mode='live', got '{mock_se.get_mode()}'"
                )

                # State strip updated to LIVE
                state_strip = app.query_one("#state-strip", StateStrip)
                assert state_strip.mode == "LIVE", (
                    f"StateStrip.mode expected 'LIVE', got '{state_strip.mode}'"
                )

                # Event bus received control.mode_change
                mock_bus.publish.assert_called()
                last_call = mock_bus.publish.call_args
                assert last_call[0][0] == "control.mode_change"
                payload = last_call[0][1]
                assert payload["from"] == "dry_run"
                assert payload["to"] == "live"
                assert payload["actor"] == "operator"
                assert "nav" in payload
                assert "open_trade_count" in payload

                # ── Step 3: press M again → immediate dry_run (no modal) ──
                mock_bus.reset_mock()
                await pilot.press("m")
                await pilot.pause()

                assert not isinstance(app.screen, ModeConfirmModal), (
                    "ModeConfirmModal pushed for live → dry_run (should be immediate)"
                )
                assert mock_se.get_mode() == "dry_run", (
                    f"Expected mode='dry_run', got '{mock_se.get_mode()}'"
                )
                assert state_strip.mode == "DRY-RUN", (
                    f"StateStrip.mode expected 'DRY-RUN', got '{state_strip.mode}'"
                )

                # Event bus notified of the reverse toggle
                mock_bus.publish.assert_called()
                rev = mock_bus.publish.call_args
                assert rev[0][0] == "control.mode_change"
                assert rev[0][1]["from"] == "live"
                assert rev[0][1]["to"] == "dry_run"

    asyncio.run(_run())


def test_mode_toggle_immediate_live_to_dry_no_modal() -> None:
    """live → dry_run via M key is immediate — no ModeConfirmModal."""
    mock_se = _MockStateEngine(initial_mode="live")
    mock_bus = MagicMock()

    async def _run() -> None:
        app = BuddyApp(live=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider._snapshot = DashboardSnapshot(nav=101_000.0, trades=[])

            with (
                patch("src.scanner.automation.state_engine.StateEngine", return_value=mock_se),
                patch("src.scanner.automation.event_bus.get_event_bus", return_value=mock_bus),
            ):
                await pilot.press("m")
                await pilot.pause()

            assert not isinstance(app.screen, ModeConfirmModal), "Modal pushed for live→dry_run"
            assert mock_se.get_mode() == "dry_run"
            state_strip = app.query_one("#state-strip", StateStrip)
            assert state_strip.mode == "DRY-RUN"

    asyncio.run(_run())
