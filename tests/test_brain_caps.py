"""Tests for src/scanner/automation/brain_caps.py (pick #6)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.scanner.automation.brain_caps import (
    BrainCapWarning,
    caps,
    check_brain_caps,
    format_warnings,
)


@pytest.fixture
def brain_dir(tmp_path) -> Path:
    d = tmp_path / "brain"
    d.mkdir()
    return d


class TestCheckBrainCaps:

    def test_missing_brain_dir_is_clean(self, tmp_path):
        # Pre-bootstrap state: directory doesn't exist yet.
        assert check_brain_caps(brain_dir=tmp_path / "no-such-dir") == []

    def test_missing_files_are_skipped(self, brain_dir):
        # Empty dir — known files just aren't present yet.
        assert check_brain_caps(brain_dir=brain_dir) == []

    def test_under_warn_threshold_is_quiet(self, brain_dir):
        # briefing.md hard cap 3000, warn @ 1.2x = 3600. 2000 chars is fine.
        (brain_dir / "briefing.md").write_text("x" * 2000, encoding="utf-8")
        assert check_brain_caps(brain_dir=brain_dir) == []

    def test_exactly_at_warn_threshold_is_quiet(self, brain_dir):
        # Equal-to threshold is still OK (warning fires on strict >).
        (brain_dir / "briefing.md").write_text("x" * 3600, encoding="utf-8")
        assert check_brain_caps(brain_dir=brain_dir) == []

    def test_above_warn_threshold_warns(self, brain_dir):
        (brain_dir / "briefing.md").write_text("x" * 3601, encoding="utf-8")
        warnings = check_brain_caps(brain_dir=brain_dir)
        assert len(warnings) == 1
        w = warnings[0]
        assert w.filename == "briefing.md"
        assert w.char_count == 3601
        assert w.hard_cap == 3000
        assert w.severity == "warn"

    def test_severity_high_at_1_5x_hard_cap(self, brain_dir):
        # briefing.md hard cap 3000; 1.5x = 4500.
        (brain_dir / "briefing.md").write_text("x" * 4500, encoding="utf-8")
        warnings = check_brain_caps(brain_dir=brain_dir)
        assert warnings[0].severity == "high"

    def test_multiple_files_each_reported(self, brain_dir):
        (brain_dir / "briefing.md").write_text("x" * 5000, encoding="utf-8")
        (brain_dir / "open_questions.md").write_text("x" * 3000, encoding="utf-8")
        warnings = check_brain_caps(brain_dir=brain_dir)
        names = {w.filename for w in warnings}
        assert names == {"briefing.md", "open_questions.md"}

    def test_unknown_files_are_ignored(self, brain_dir):
        # Only files in the caps table are checked; arbitrary .md files don't fire.
        (brain_dir / "scratch.md").write_text("x" * 999_999, encoding="utf-8")
        assert check_brain_caps(brain_dir=brain_dir) == []

    def test_unreadable_file_does_not_crash(self, brain_dir, monkeypatch):
        path = brain_dir / "briefing.md"
        path.write_text("x" * 4000, encoding="utf-8")

        # Simulate an OS read failure.
        original_read = Path.read_text

        def boom(self, *a, **kw):
            if self == path:
                raise OSError("simulated read failure")
            return original_read(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", boom)
        # Function returns [] instead of raising.
        assert check_brain_caps(brain_dir=brain_dir) == []

    def test_custom_caps_override_defaults(self, brain_dir):
        (brain_dir / "briefing.md").write_text("x" * 200, encoding="utf-8")
        custom = {"briefing.md": (100, 1.20)}  # warn @ 120 chars
        warnings = check_brain_caps(brain_dir=brain_dir, caps=custom)
        assert len(warnings) == 1 and warnings[0].char_count == 200


class TestBrainCapWarning:

    def test_pct_rounds_correctly(self):
        w = BrainCapWarning("x.md", char_count=3300, hard_cap=3000, warn_threshold=3600, severity="warn")
        assert w.pct == 110

    def test_message_contains_path_and_counts(self):
        w = BrainCapWarning("briefing.md", char_count=3300, hard_cap=3000, warn_threshold=3600, severity="warn")
        msg = w.message()
        assert "briefing.md" in msg and "3,300" in msg and "3,000" in msg


class TestFormatWarnings:

    def test_empty_returns_empty_string(self):
        assert format_warnings([]) == ""

    def test_includes_severity_per_line(self):
        ws = [
            BrainCapWarning("a.md", 5000, 3000, 3600, "high"),
            BrainCapWarning("b.md", 2000, 1500, 1800, "warn"),
        ]
        out = format_warnings(ws)
        assert "2 file(s) over cap" in out
        assert "[HIGH]" in out and "[WARN]" in out


class TestCapsTable:

    def test_caps_returns_a_copy(self):
        a = caps()
        a.clear()
        assert caps()  # Original untouched.

    def test_all_expected_files_present(self):
        expected = {"briefing.md", "session_handoff.md", "open_questions.md",
                    "strategic_log.md", "trade_narrative.md"}
        assert set(caps().keys()) == expected
