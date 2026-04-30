"""Mythos audit 2026-04-30 — brain-feed disk tee.

Operator's verification challenge surfaced the gap: the F1 brain feed
in the running TUI was in-memory only. Confirming whether a specific
commit's code path fires required reading unrelated disk artifacts
(state.json, .claude/meta/changes/*.json, reflection_log.jsonl)
instead of the actual feed the operator watches scroll.

This commit tees every _write_brain call to .claude/brain/feed.jsonl
as plain-text JSONL — making the operator's verification surface and
the audit verification surface identical.

Tests pin:
  * Plain-text stripping of Rich markup (``[cyan]hi[/]`` → ``hi``)
  * JSONL schema (ts, msg, raw all present)
  * Rotation on size threshold (no unbounded growth)
  * Failure isolation (degraded fs MUST NOT crash the TUI render)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tui.app import (
    _BRAIN_FEED_MAX_BYTES,
    _strip_rich_markup,
    _tee_brain_to_disk,
)


@pytest.fixture
def disk_path(tmp_path, monkeypatch):
    feed = tmp_path / ".claude/brain/feed.jsonl"
    monkeypatch.setattr("src.tui.app._BRAIN_FEED_PATH", feed)
    return feed


def test_markup_stripper_removes_simple_tags():
    assert _strip_rich_markup("[cyan]hi[/]") == "hi"
    assert _strip_rich_markup("[bold red]ALERT[/]") == "ALERT"
    assert _strip_rich_markup("plain text") == "plain text"


def test_markup_stripper_preserves_content():
    """Double-check: only brackets are dropped, content intact."""
    assert _strip_rich_markup(
        "[cyan]▸ meta ⟪89dabc3f⟫ intake event=duplicate_dropped[/]"
    ) == "▸ meta ⟪89dabc3f⟫ intake event=duplicate_dropped"


def test_markup_stripper_handles_nested_close():
    assert _strip_rich_markup("[bold]A[/bold]B[/]") == "AB"


def test_tee_writes_jsonl_with_required_schema(disk_path):
    _tee_brain_to_disk("[cyan]▸ meta ⟪abc12345⟫ intake[/]")
    assert disk_path.exists()
    line = disk_path.read_text(encoding="utf-8").strip()
    blob = json.loads(line)
    assert "ts" in blob
    assert "msg" in blob
    assert "raw" in blob
    assert blob["msg"] == "▸ meta ⟪abc12345⟫ intake"
    assert blob["raw"] == "[cyan]▸ meta ⟪abc12345⟫ intake[/]"


def test_tee_appends_multiple_lines(disk_path):
    _tee_brain_to_disk("[cyan]first[/]")
    _tee_brain_to_disk("[red]second[/]")
    _tee_brain_to_disk("[green]third[/]")
    lines = disk_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    msgs = [json.loads(l)["msg"] for l in lines]
    assert msgs == ["first", "second", "third"]


def test_tee_creates_parent_directory(tmp_path, monkeypatch):
    """Should mkdir(parents=True, exist_ok=True) — first-ever call must
    not fail because .claude/brain/ doesn't exist yet."""
    feed = tmp_path / "deeply/nested/path/feed.jsonl"
    monkeypatch.setattr("src.tui.app._BRAIN_FEED_PATH", feed)
    _tee_brain_to_disk("[cyan]bootstrap[/]")
    assert feed.exists()


def test_tee_rotates_when_oversize(disk_path, monkeypatch):
    """When the feed exceeds the rotation ceiling, rotate to .1
    and start fresh. Operator can cat both for full history."""
    # Drop the threshold so we don't have to write 5 MB in a test
    monkeypatch.setattr("src.tui.app._BRAIN_FEED_MAX_BYTES", 200)

    # Fill above threshold
    for i in range(20):
        _tee_brain_to_disk(f"[cyan]message {i:02d} padding padding[/]")

    # First call after threshold should rotate
    _tee_brain_to_disk("[red]post-rotation[/]")

    rotated = disk_path.with_suffix(".jsonl.1")
    assert rotated.exists(), "rotation file must be written"
    assert disk_path.exists(), "fresh feed file must be created"
    fresh = disk_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("post-rotation" in l for l in fresh)


def test_tee_swallows_io_errors(monkeypatch):
    """Disk failures during the tee MUST NOT raise — observability is
    never load-bearing on TUI rendering."""
    def boom_open(*args, **kwargs):
        raise OSError("disk full")

    fake_path = Path("/tmp/this/path/should/not/be/written/feed.jsonl")
    monkeypatch.setattr("src.tui.app._BRAIN_FEED_PATH", fake_path)
    monkeypatch.setattr(Path, "open", boom_open, raising=False)

    # Must not raise.
    _tee_brain_to_disk("[cyan]should-not-crash[/]")


def test_strip_handles_empty_string():
    assert _strip_rich_markup("") == ""


def test_strip_handles_no_brackets():
    assert _strip_rich_markup("plain") == "plain"
