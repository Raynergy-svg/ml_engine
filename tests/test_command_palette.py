"""Tests for the TUI command palette (pick #4).

Two layers:
  1. Pure filter logic (no Textual runtime) — fast, covers most logic.
  2. Pilot-based smoke test that mounts the modal — verifies wiring.
"""
from __future__ import annotations

import pytest

from src.tui.screens.command_palette_modal import (
    Command,
    CommandPaletteModal,
    default_commands,
    filter_commands,
)


class TestFilter:

    def test_empty_query_returns_all(self):
        cmds = default_commands()
        assert filter_commands("", cmds) == cmds
        assert filter_commands("   ", cmds) == cmds

    def test_token_in_label_matches(self):
        cmds = default_commands()
        out = filter_commands("trades", cmds)
        assert any(c.id == "nav.trades" for c in out)

    def test_token_in_keyword_matches(self):
        cmds = default_commands()
        # "panic" is a keyword for supervisor.kill — not in label.
        out = filter_commands("panic", cmds)
        assert [c.id for c in out] == ["supervisor.kill"]

    def test_token_in_id_matches(self):
        cmds = default_commands()
        out = filter_commands("broker", cmds)
        assert any(c.id == "broker.cycle" for c in out)

    def test_case_insensitive(self):
        cmds = default_commands()
        assert filter_commands("KILL", cmds) == filter_commands("kill", cmds)

    def test_all_tokens_must_match(self):
        cmds = default_commands()
        # "live" appears in supervisor.mode (label) AND supervisor.kill (label is
        # 'Kill switch — flatten all', no 'live'). Token AND-ing should narrow.
        narrow = filter_commands("live mode", cmds)
        assert [c.id for c in narrow] == ["supervisor.mode"]

    def test_no_match_returns_empty(self):
        cmds = default_commands()
        assert filter_commands("zzzzzz-no-such-cmd", cmds) == []

    def test_filter_is_stable_ordering(self):
        # Filtered output preserves the registry's declared order.
        cmds = default_commands()
        out = filter_commands("go to", cmds)
        ids = [c.id for c in out]
        assert ids == [c.id for c in cmds if c.id in ids]


class TestRegistry:

    def test_command_ids_are_unique(self):
        cmds = default_commands()
        ids = [c.id for c in cmds]
        assert len(ids) == len(set(ids))

    def test_every_nav_id_maps_to_a_known_tab(self):
        """nav.<x> must match the BuddyApp tab ids: overview/inbox/trades/...

        Catches the case where someone renames a tab but forgets the palette.
        """
        known_tabs = {"overview", "inbox", "trades", "agents", "journal", "config", "rules", "diag"}
        for cmd in default_commands():
            if cmd.id.startswith("nav."):
                tab = cmd.id.split(".", 1)[1]
                assert tab in known_tabs, f"palette {cmd.id} → unknown tab '{tab}'"

    def test_every_non_nav_id_has_a_router_entry(self):
        """app.action_open_command_palette routes non-nav ids via a dict.

        Mirror that dict here so adding a Command without wiring it surfaces
        in CI rather than silently no-op'ing at runtime.
        """
        routable = {
            "supervisor.pause",
            "supervisor.kill",
            "supervisor.mode",
            "supervisor.abort",
            "broker.cycle",
            "app.quit",
        }
        for cmd in default_commands():
            if cmd.id.startswith("nav."):
                continue
            assert cmd.id in routable, f"palette {cmd.id} has no router entry"


class TestModalLifecycle:
    """Light Pilot test — modal mounts, filters, dispatches the chosen id."""

    @pytest.mark.asyncio
    async def test_modal_dismisses_with_selected_id(self):
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        captured: list[str | None] = []

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield Static("host")

            def on_mount(self) -> None:
                self.push_screen(CommandPaletteModal(), captured.append)

        async with _Host().run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            # Type to filter down to a single match, then Enter.
            await pilot.press(*"panic")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert captured == ["supervisor.kill"]

    @pytest.mark.asyncio
    async def test_modal_dismisses_with_none_on_escape(self):
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        captured: list[str | None] = []

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield Static("host")

            def on_mount(self) -> None:
                self.push_screen(CommandPaletteModal(), captured.append)

        async with _Host().run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert captured == [None]
