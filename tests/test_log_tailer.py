"""Tests for src/tui/log_tailer.py (pick #7)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.tui.log_tailer import (
    DEFAULT_HISTORY_LINES,
    LogTailer,
    filter_lines,
    known_log_files,
    tail_last_n_lines,
)


# ────────────────────────── tail_last_n_lines ──────────────────────────


class TestTailLastNLines:

    def test_missing_file_returns_empty(self, tmp_path):
        assert tail_last_n_lines(tmp_path / "no.log") == []

    def test_returns_last_n(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
        out = tail_last_n_lines(path, n=10)
        assert out == [str(i) for i in range(90, 100)]

    def test_n_larger_than_file_returns_all(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("a\nb\nc", encoding="utf-8")
        assert tail_last_n_lines(path, n=10) == ["a", "b", "c"]

    def test_long_lines_are_truncated(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("y" * 200, encoding="utf-8")
        out = tail_last_n_lines(path, per_line_cap=50)
        assert out[0].endswith("…[truncated]")
        assert len(out[0]) <= 50 + len("…[truncated]")


# ────────────────────────── filter_lines ──────────────────────────


class TestFilterLines:

    def test_empty_query_is_identity(self):
        assert filter_lines(["a", "b"], "") == ["a", "b"]
        assert filter_lines(["a", "b"], "   ") == ["a", "b"]

    def test_substring_match(self):
        out = filter_lines(["ERROR: bad", "INFO: ok", "ERROR: worse"], "error")
        assert out == ["ERROR: bad", "ERROR: worse"]

    def test_case_insensitive(self):
        assert filter_lines(["FoO"], "foo") == ["FoO"]

    def test_no_match_returns_empty(self):
        assert filter_lines(["a", "b"], "zzz") == []


# ────────────────────────── known_log_files ──────────────────────────


class TestKnownLogFiles:

    def test_returns_only_existing(self, tmp_path):
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "buddy_tui.stderr.log").write_text("", encoding="utf-8")
        # boot_trace.log + reflection_log.jsonl deliberately missing.
        files = known_log_files(tmp_path)
        assert [p.name for p in files] == ["buddy_tui.stderr.log"]

    def test_empty_when_logs_dir_missing(self, tmp_path):
        assert known_log_files(tmp_path) == []


# ────────────────────────── LogTailer ──────────────────────────


class TestLogTailer:

    def test_first_read_seeks_to_tail(self, tmp_path):
        path = tmp_path / "x.log"
        # 100 KB of content; with default 64KB initial-tail, we only see the
        # last ~64KB (a few thousand short lines).
        path.write_text("\n".join(f"line-{i:05d}" for i in range(20_000)), encoding="utf-8")
        tailer = LogTailer(path, initial_tail_bytes=4096)
        chunk = tailer.read_new_lines()
        assert chunk.rotated is False
        # The very first line of the file is NOT in the chunk:
        assert "line-00000" not in chunk.lines
        # But the very last line IS:
        assert "line-19999" in chunk.lines

    def test_incremental_reads_pick_up_appends(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("seed\n", encoding="utf-8")
        tailer = LogTailer(path, initial_tail_bytes=0)
        tailer.read_new_lines()  # consume initial state

        with path.open("a", encoding="utf-8") as f:
            f.write("one\ntwo\n")
        chunk = tailer.read_new_lines()
        assert chunk.lines == ["one", "two"]
        assert chunk.rotated is False

    def test_no_new_data_returns_empty(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("a\nb\n", encoding="utf-8")
        tailer = LogTailer(path, initial_tail_bytes=0)
        tailer.read_new_lines()
        assert tailer.read_new_lines().lines == []

    def test_truncation_resets_offset_and_signals_rotation(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("first\nsecond\nthird\n", encoding="utf-8")
        tailer = LogTailer(path, initial_tail_bytes=0)
        tailer.read_new_lines()  # offset now at end

        # Simulate rotation: file shrinks to a fresh single line.
        path.write_text("rotated\n", encoding="utf-8")
        chunk = tailer.read_new_lines()
        assert chunk.rotated is True
        assert chunk.lines == ["rotated"]

    def test_missing_file_returns_empty_without_crash(self, tmp_path):
        tailer = LogTailer(tmp_path / "never.log")
        assert tailer.read_new_lines().lines == []

    def test_file_reappears_after_disappearing(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("first\n", encoding="utf-8")
        # Non-zero initial_tail_bytes so that the post-recovery "first read"
        # actually surfaces the contents — initial_tail_bytes=0 would seek
        # straight to EOF on every first-read and skip everything that was
        # there at the moment of reopen.
        tailer = LogTailer(path, initial_tail_bytes=1024)
        tailer.read_new_lines()

        os.remove(path)
        assert tailer.read_new_lines().lines == []

        path.write_text("reborn\n", encoding="utf-8")
        chunk = tailer.read_new_lines()
        assert "reborn" in chunk.lines

    def test_long_line_truncation(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("a" * 500 + "\n", encoding="utf-8")
        tailer = LogTailer(path, initial_tail_bytes=0, per_line_cap=100)
        # offset starts at end on first read → no lines.
        tailer.read_new_lines()
        with path.open("a", encoding="utf-8") as f:
            f.write("b" * 500 + "\n")
        chunk = tailer.read_new_lines()
        assert chunk.lines and chunk.lines[0].endswith("…[truncated]")
        assert len(chunk.lines[0]) <= 100 + len("…[truncated]")

    def test_reset_re_seeds_tail(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_text("only line\n", encoding="utf-8")
        tailer = LogTailer(path, initial_tail_bytes=0)
        tailer.read_new_lines()
        assert tailer.read_new_lines().lines == []

        tailer.reset()
        # After reset, first read should re-emit (initial_tail_bytes=0 means
        # 'start from end' would be empty, so this is more nuanced).
        # Use bigger initial_tail_bytes to verify reset behavior:
        tailer = LogTailer(path, initial_tail_bytes=1024)
        first = tailer.read_new_lines()
        assert "only line" in first.lines
        # Second read returns nothing new:
        assert tailer.read_new_lines().lines == []
        # Reset, then read again — should emit the tail again.
        tailer.reset()
        again = tailer.read_new_lines()
        assert "only line" in again.lines


# ────────────────────────── LogViewerModal lifecycle ──────────────────────────


class TestLogViewerModalLifecycle:

    @pytest.mark.asyncio
    async def test_modal_mounts_and_dismisses_on_escape(self, tmp_path):
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        from src.tui.screens.log_viewer_modal import LogViewerModal

        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "buddy_tui.stderr.log").write_text("hello\nworld\n", encoding="utf-8")

        captured: list = []

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield Static("host")

            def on_mount(self) -> None:
                self.push_screen(LogViewerModal(tmp_path), captured.append)

        async with _Host().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert captured == [None]
