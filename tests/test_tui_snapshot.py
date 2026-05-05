"""Real-disk test: TUI snapshot captures every canonical surface.

Mythos audit 2026-05-04 — operator request: 'c' hotkey copies all
visible state to a markdown file + clipboard for paste-back debugging.
The snapshot must include each canonical verification surface from
CLAUDE.md "Honesty & verification" section so Claude has full context
in one paste.

NO MOCKS. The snapshot module reads from PROJECT_ROOT — the real disk.
This test runs build_snapshot() against the real working tree and
asserts the output structure contains the expected sections. Surface
content is allowed to be empty/missing (test environment may not
have a live TUI process), but the SECTION HEADERS must be present so
the operator/Claude knows what to look at.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tui.snapshot import (
    build_snapshot,
    copy_to_clipboard,
    write_and_copy,
    write_snapshot,
)


REQUIRED_SECTIONS = [
    "# Buddy TUI Snapshot",
    "## Runtime state",
    "### Heartbeat",
    "### State (.claude/state.json)",
    "### Policy state",
    "### AlertManager",
    "## Autonomous loop",
    "### Meta-pipeline",
    "### Config adjustments",
    "## Models & agents",
    "### Model health",
    "### Agents",
    "## Trading evidence",
    "### Trade journal",
    "### Virtual trades",
    "## Logs",
    "### Brain feed",
    "### System activity",
    "### Autonomous trainer log tail",
    "### Debug log",
]


def test_snapshot_contains_every_required_section() -> None:
    """The snapshot must surface every canonical verification surface
    so the operator's 'c'-paste gives Claude the full picture.
    """
    snap = build_snapshot()
    missing = [s for s in REQUIRED_SECTIONS if s not in snap]
    assert not missing, (
        f"Snapshot missing {len(missing)} required section(s): {missing}"
    )


def test_snapshot_is_reasonable_size() -> None:
    """A real-disk snapshot of a developed project should be 5–200KB.
    Smaller = sections silently failing. Larger = unbounded log dump
    that won't paste cleanly.
    """
    snap = build_snapshot()
    size_kb = len(snap.encode("utf-8")) / 1024
    assert 5 <= size_kb <= 200, (
        f"Snapshot size {size_kb:.1f}KB outside 5–200KB envelope. "
        f"Either sections are silently failing (too small) or a log "
        f"section is unbounded (too large)."
    )


def test_write_snapshot_lands_on_disk(tmp_path: Path) -> None:
    """write_snapshot creates a file under the requested directory."""
    snapshots_dir = tmp_path / "snapshots"
    path = write_snapshot(snapshots_dir=snapshots_dir)
    assert path.exists()
    assert path.parent == snapshots_dir
    assert path.suffix == ".md"
    assert path.stat().st_size > 0
    content = path.read_text(encoding="utf-8")
    assert "Buddy TUI Snapshot" in content


def test_write_and_copy_returns_clipboard_tool_when_available(
    tmp_path: Path,
) -> None:
    """write_and_copy reports which clipboard tool succeeded.

    On a macOS dev box pbcopy is always present; on a Linux CI runner
    it depends on the env. Either result is valid — what we're testing
    is the contract that the function returns the tool name (string)
    or None, not that any specific tool exists.
    """
    snapshots_dir = tmp_path / "snapshots"
    path, tool = write_and_copy(snapshots_dir=snapshots_dir)
    assert path.exists()
    assert tool is None or isinstance(tool, str)


def test_copy_to_clipboard_handles_no_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no clipboard tool is on PATH, copy_to_clipboard returns None
    without raising. Snapshot file write must still succeed independently.
    """
    monkeypatch.setattr("shutil.which", lambda _name: None)
    result = copy_to_clipboard("hello world")
    assert result is None


def test_focusable_static_panels_have_can_focus() -> None:
    """Click-to-focus contract: every Static panel that operators
    interact with must declare can_focus=True so the magenta focus
    border renders on click. Without can_focus, mouse clicks go
    nowhere and the operator can't target a box for the 'c' copy.
    """
    from src.tui.app import (
        AgentPanel,
        HeaderBar,
        MTFConfluencePanel,
        RiskPanel,
        SystemHealthBar,
    )

    for cls in (HeaderBar, AgentPanel, RiskPanel, MTFConfluencePanel, SystemHealthBar):
        assert getattr(cls, "can_focus", False) is True, (
            f"{cls.__name__} is not focusable — operator can't click "
            f"into it to highlight + copy. Add `can_focus = True`."
        )


def test_per_box_section_helpers_are_callable() -> None:
    """The widget-id-to-section map in app.py:_dump_focused_widget
    expects these snapshot module functions to exist. If any are
    renamed or removed without updating the map, the per-box copy
    silently falls through to the generic widget.render() path.
    """
    from src.tui import snapshot as _snap

    required = (
        "_heartbeat_section",
        "_agents_section",
        "_brain_feed_section",
        "_reflection_log_section",
        "_trades_section",
        "_debug_log_section",
    )
    for fname in required:
        fn = getattr(_snap, fname, None)
        assert callable(fn), (
            f"snapshot.{fname} is not callable — the per-box copy "
            f"map in app.py:_dump_focused_widget will fail for the "
            f"widget mapped to it."
        )
        # Each section should produce some output.
        out = fn()
        assert isinstance(out, str) and len(out) > 0, (
            f"snapshot.{fname}() returned empty/non-string output"
        )
