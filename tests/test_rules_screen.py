"""Tests for RulesScreen (US-514) — read-only Rules / Learnings / Brain viewer.

Asserts:
- Tree mounts with rules/, learnings/, brain/ sections + expected files
- Markdown widget renders selected file content
- In-view search finds known terms
- No write operations exist anywhere in rules_screen.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.tui.app import BuddyApp
from src.tui.screens.rules_screen import (
    GrepModal,
    RulesScreen,
    SearchModal,
    _collect_files,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_collect_files_returns_expected_sections():
    files = _collect_files(PROJECT_ROOT)
    assert set(files.keys()) == {"rules", "learnings", "brain"}
    # rules dir holds at least improvement.md + trading.md
    rule_names = {fp.name for fp in files["rules"]}
    assert "trading.md" in rule_names
    assert "improvement.md" in rule_names
    # learnings.md present
    learn_names = {fp.name for fp in files["learnings"]}
    assert "learnings.md" in learn_names
    # brain dir has at least briefing.md + open_questions.md
    brain_names = {fp.name for fp in files["brain"]}
    assert "briefing.md" in brain_names
    assert "open_questions.md" in brain_names


def test_no_write_operations_in_rules_screen():
    """Acceptance criterion: NO write operations — grep open('w') == 0."""
    src = (PROJECT_ROOT / "src" / "tui" / "screens" / "rules_screen.py").read_text()
    assert "open('w')" not in src
    assert 'open("w")' not in src
    assert ", 'w')" not in src
    assert ', "w")' not in src
    assert ".write_text(" not in src
    # no os.write, file.write — except inside docstring/comment is fine but rules_screen.py has none
    for forbidden in ("write_text(", "write_bytes(", "writelines("):
        assert forbidden not in src, f"forbidden write op present: {forbidden}"


def test_rules_screen_mounts_with_tree_and_markdown():
    """Mount the screen via BuddyApp and verify Tree + Markdown render."""
    from textual.widgets import Markdown, Tree

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            # Switch to the Rules tab via F7
            await pilot.press("f7")
            await pilot.pause()

            screen = app.query_one("#rules-screen", RulesScreen)
            tree = screen.query_one("#rules-tree", Tree)
            md = screen.query_one("#rules-md", Markdown)
            assert tree is not None
            assert md is not None

            # Tree root should have 3 child sections (rules/, learnings/, brain/)
            child_labels = [str(c.label) for c in tree.root.children]
            assert "rules/" in child_labels
            assert "learnings/" in child_labels
            assert "brain/" in child_labels

            # Each section should have at least one leaf file
            for child in tree.root.children:
                assert len(child.children) >= 1, f"section {child.label} empty"

    asyncio.run(_run())


def test_rules_screen_renders_selected_file():
    """Selecting a node in the tree should populate the Markdown widget."""
    from textual.widgets import Markdown

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.press("f7")
            await pilot.pause()

            screen = app.query_one("#rules-screen", RulesScreen)

            # Find trading.md path in the screen's path map and render directly
            target = None
            for fp in screen._path_by_node.values():
                if fp.name == "trading.md":
                    target = fp
                    break
            assert target is not None, "trading.md not in tree"

            screen._render_file(target)
            await pilot.pause()

            md = screen.query_one("#rules-md", Markdown)
            # Markdown widget exposes its source via the `_markdown` attribute
            # (Textual ≥0.50). Fall back to checking the screen's own cache.
            assert "Trading Rules" in screen._current_text or \
                   "trading" in screen._current_text.lower()

    asyncio.run(_run())


def test_search_finds_known_term():
    """In-view search highlight bar should report match counts for a known term."""

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.press("f7")
            await pilot.pause()

            screen = app.query_one("#rules-screen", RulesScreen)
            # Render trading.md so the search bar has content to search
            target = next(
                fp for fp in screen._path_by_node.values() if fp.name == "trading.md"
            )
            screen._render_file(target)
            await pilot.pause()

            # Apply search highlight directly (skips the modal flow)
            screen._apply_search_highlight("ATR")
            await pilot.pause()

            from textual.widgets import Static
            bar = screen.query_one("#rules-search-bar", Static)
            # Static.update stores the renderable; render() returns it
            rendered = str(bar.render())
            assert "ATR" in rendered
            assert "match" in rendered.lower()

    asyncio.run(_run())


def test_grep_modal_finds_cross_file_matches():
    """GrepModal._grep should locate known content across rule + brain files."""
    modal = GrepModal(PROJECT_ROOT)
    results = modal._grep("ATR")
    assert len(results) >= 1
    # Every result is (Path, int line, str text)
    for fp, ln, text in results:
        assert fp.exists()
        assert isinstance(ln, int) and ln >= 1
        assert "atr" in text.lower()


def test_f7_binding_routes_to_rules_tab():
    """F7 must activate the rules tab (Diagnostics moves to F8)."""
    from textual.widgets import TabbedContent

    async def _run():
        app = BuddyApp(live=False)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.press("f7")
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            assert tabs.active == "rules"

            await pilot.press("f8")
            await pilot.pause()
            assert tabs.active == "diag"

    asyncio.run(_run())
