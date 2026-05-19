"""US-012: ConfigScreen keyboard BINDINGS tests.

Acceptance criteria:
  AC-1  BINDINGS list declares s/q/r with show=True.
  AC-2  Pressing s calls action_stage_profile.
  AC-3  Pressing q calls action_queue_for_approval.
  AC-4  Pressing r (non-Input focus) calls action_reset_form.
  AC-5  Pressing r when an Input inside ConfigScreen has focus does NOT call
        action_reset_form (Input consumes the printable character before the
        BINDINGS chain is reached — Textual's default non-priority dispatch).
  AC-6  on_button_pressed delegates to the same action methods as the bindings
        (shared dispatch contract verified at the source level).

No mocks per CLAUDE.md No-Mock Rule.  _TrackingConfigScreen is a real
ConfigScreen subclass that records action invocations via list append.
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import List

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input

from src.tui.screens.config_screen import ConfigScreen

_TUI_THEME_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"
)


# ---------------------------------------------------------------------------
# Instrumented subclass — records which action methods fired
# ---------------------------------------------------------------------------


class _TrackingConfigScreen(ConfigScreen):
    """Real ConfigScreen subclass that appends to called_actions on each action."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.called_actions: List[str] = []

    def action_stage_profile(self) -> None:
        self.called_actions.append("action_stage_profile")
        super().action_stage_profile()

    def action_queue_for_approval(self) -> None:
        self.called_actions.append("action_queue_for_approval")
        super().action_queue_for_approval()

    def action_reset_form(self) -> None:
        self.called_actions.append("action_reset_form")
        super().action_reset_form()


# ---------------------------------------------------------------------------
# Minimal host App
# ---------------------------------------------------------------------------


class _ConfigApp(App):
    """Smallest App that wraps _TrackingConfigScreen for headless testing."""

    CSS_PATH = str(_TUI_THEME_PATH)

    def __init__(self, project_root: Path | None = None) -> None:
        super().__init__()
        self._root = project_root
        self.config_screen: _TrackingConfigScreen | None = None

    def compose(self) -> ComposeResult:
        self.config_screen = _TrackingConfigScreen(
            project_root=str(self._root) if self._root else ""
        )
        yield self.config_screen


# ---------------------------------------------------------------------------
# AC-1: static BINDINGS check (no pilot needed)
# ---------------------------------------------------------------------------


def test_bindings_declared_with_show_true() -> None:
    """AC-1: s/q/r are in BINDINGS with show=True."""
    binding_map: dict[str, Binding] = {b.key: b for b in ConfigScreen.BINDINGS}
    assert "s" in binding_map, "BINDINGS must declare key 's'"
    assert "q" in binding_map, "BINDINGS must declare key 'q'"
    assert "r" in binding_map, "BINDINGS must declare key 'r'"
    assert binding_map["s"].show is True, "s binding must have show=True"
    assert binding_map["q"].show is True, "q binding must have show=True"
    assert binding_map["r"].show is True, "r binding must have show=True"


def test_binding_actions_map_to_correct_methods() -> None:
    """AC-1: BINDINGS action names resolve to the expected action methods."""
    binding_map = {b.key: b for b in ConfigScreen.BINDINGS}
    assert binding_map["s"].action == "stage_profile"
    assert binding_map["q"].action == "queue_for_approval"
    assert binding_map["r"].action == "reset_form"


# ---------------------------------------------------------------------------
# AC-6: on_button_pressed source delegates to action methods (no pilot needed)
# ---------------------------------------------------------------------------


def test_on_button_pressed_source_delegates_to_action_methods() -> None:
    """AC-6: on_button_pressed calls action_* methods, not private helpers directly.

    Inspects the source of on_button_pressed to confirm the shared-dispatch
    contract: button clicks and keyboard bindings go through the same action
    method entry points.
    """
    src = inspect.getsource(ConfigScreen.on_button_pressed)
    assert "action_queue_for_approval" in src, (
        "on_button_pressed must call action_queue_for_approval() for shared dispatch"
    )
    assert "action_reset_form" in src, (
        "on_button_pressed must call action_reset_form() for shared dispatch"
    )
    assert "action_stage_profile" in src, (
        "on_button_pressed must call action_stage_profile() for shared dispatch"
    )


# ---------------------------------------------------------------------------
# AC-2: s → action_stage_profile
# ---------------------------------------------------------------------------


def test_s_fires_action_stage_profile(tmp_path: Path) -> None:
    """AC-2: pressing s calls action_stage_profile."""
    app = _ConfigApp(project_root=tmp_path)

    async def _run() -> None:
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            assert app.config_screen is not None
            # Ensure a non-Input widget has focus (RadioSet doesn't consume s/q/r)
            app.config_screen.query_one("#config-profile-radios").focus()
            await pilot.pause(0.1)
            app.config_screen.called_actions.clear()
            await pilot.press("s")
            await pilot.pause(0.1)

    asyncio.run(_run())
    assert app.config_screen is not None
    assert "action_stage_profile" in app.config_screen.called_actions, (
        f"Expected action_stage_profile to be called after pressing s; "
        f"got: {app.config_screen.called_actions}"
    )


# ---------------------------------------------------------------------------
# AC-3: q → action_queue_for_approval
# ---------------------------------------------------------------------------


def test_q_fires_action_queue_for_approval(tmp_path: Path) -> None:
    """AC-3: pressing q calls action_queue_for_approval."""
    app = _ConfigApp(project_root=tmp_path)

    async def _run() -> None:
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            assert app.config_screen is not None
            app.config_screen.query_one("#config-profile-radios").focus()
            await pilot.pause(0.1)
            app.config_screen.called_actions.clear()
            await pilot.press("q")
            await pilot.pause(0.1)

    asyncio.run(_run())
    assert app.config_screen is not None
    assert "action_queue_for_approval" in app.config_screen.called_actions, (
        f"Expected action_queue_for_approval to be called after pressing q; "
        f"got: {app.config_screen.called_actions}"
    )


# ---------------------------------------------------------------------------
# AC-4: r → action_reset_form (non-Input focus)
# ---------------------------------------------------------------------------


def test_r_fires_action_reset_form_when_radioset_focused(tmp_path: Path) -> None:
    """AC-4: pressing r with RadioSet focused calls action_reset_form."""
    app = _ConfigApp(project_root=tmp_path)

    async def _run() -> None:
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            assert app.config_screen is not None
            app.config_screen.query_one("#config-profile-radios").focus()
            await pilot.pause(0.1)
            app.config_screen.called_actions.clear()
            await pilot.press("r")
            await pilot.pause(0.1)

    asyncio.run(_run())
    assert app.config_screen is not None
    assert "action_reset_form" in app.config_screen.called_actions, (
        f"Expected action_reset_form to be called after pressing r (RadioSet focused); "
        f"got: {app.config_screen.called_actions}"
    )


# ---------------------------------------------------------------------------
# AC-5: r does NOT fire when an Input has focus
# ---------------------------------------------------------------------------


def test_r_does_not_fire_action_reset_form_when_input_focused(tmp_path: Path) -> None:
    """AC-5: pressing r with a gate Input focused does NOT call action_reset_form.

    Input widgets consume all printable characters for text editing before the
    BINDINGS chain is reached (Textual non-priority dispatch).  The Input will
    receive 'r' as a text character; ConfigScreen's reset_form binding will not
    fire.
    """
    app = _ConfigApp(project_root=tmp_path)
    input_value_after: list[str] = []

    async def _run() -> None:
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            assert app.config_screen is not None

            # Explicitly focus a gate Input widget
            inp = app.config_screen.query_one("#gate-min_confidence", Input)
            inp.focus()
            await pilot.pause(0.1)

            app.config_screen.called_actions.clear()
            await pilot.press("r")
            await pilot.pause(0.1)

            # Capture the input value so we can confirm the key was consumed as text
            input_value_after.append(inp.value)

    asyncio.run(_run())
    assert app.config_screen is not None
    assert "action_reset_form" not in app.config_screen.called_actions, (
        "action_reset_form MUST NOT fire when an Input widget inside ConfigScreen "
        f"has focus; called_actions={app.config_screen.called_actions}"
    )
    # Confirm 'r' was consumed by the Input (appended to its text value)
    if input_value_after:
        assert input_value_after[0].endswith("r"), (
            f"Input should have received 'r' as text; value={input_value_after[0]!r}"
        )
