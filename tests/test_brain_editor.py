"""Tier 2 T7: Tests for the brain section editor."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tui.widgets.brain_editor import BriefingDocument, BriefingSection


SAMPLE = """# Briefing

## Current Situation

Halted on Trade 1261 loss streak.

## Hypotheses

1. RL position sizer stale
2. Trend agent ADX threshold wrong

## Next Actions

- [ ] Retrain RL sizer
- [ ] Audit ADX
"""


def test_parse_three_sections(tmp_path: Path):
    p = tmp_path / "briefing.md"
    p.write_text(SAMPLE)
    doc = BriefingDocument.from_path(p)
    assert len(doc.sections) == 3
    titles = [s.title for s in doc.sections]
    assert titles == ["Current Situation", "Hypotheses", "Next Actions"]


def test_section_body_preserves_content(tmp_path: Path):
    p = tmp_path / "briefing.md"
    p.write_text(SAMPLE)
    doc = BriefingDocument.from_path(p)
    s = next(s for s in doc.sections if s.title == "Hypotheses")
    assert "RL position sizer stale" in s.body
    assert "Trend agent ADX threshold wrong" in s.body


def test_serialize_round_trip(tmp_path: Path):
    p = tmp_path / "briefing.md"
    p.write_text(SAMPLE)
    doc = BriefingDocument.from_path(p)
    out = doc.to_text()
    assert "## Current Situation" in out
    assert "## Hypotheses" in out
    assert "## Next Actions" in out


def test_update_section_changes_body(tmp_path: Path):
    p = tmp_path / "briefing.md"
    p.write_text(SAMPLE)
    doc = BriefingDocument.from_path(p)
    doc.update_section("Hypotheses", "1. only one hypothesis now")
    out = doc.to_text()
    assert "only one hypothesis now" in out
    assert "RL position sizer stale" not in out


def test_save_pre_write_validation_refuses_over_cap(tmp_path: Path):
    # briefing.md hard_cap is 3000 chars per brain_caps.py.
    p = tmp_path / "briefing.md"
    big_body = "x" * 5000
    p.write_text(f"# Briefing\n\n## Big\n\n{big_body}\n")
    doc = BriefingDocument.from_path(p)
    doc.update_section("Big", big_body + "Y" * 2000)
    with pytest.raises(ValueError) as ei:
        doc.save_to_path(p)
    assert "cap" in str(ei.value).lower() or "limit" in str(ei.value).lower()


def test_reset_to_template_overwrites_file(tmp_path: Path):
    p = tmp_path / "briefing.md"
    tmpl = tmp_path / "briefing.md.default"
    tmpl.write_text("# Briefing\n\n## Current Situation\n\n(empty)\n")
    p.write_text("# Briefing\n\n## Stale\n\nold content\n")
    doc = BriefingDocument.from_path(p)
    doc.reset_to_template(tmpl, target=p)
    fresh = BriefingDocument.from_path(p)
    assert any(s.title == "Current Situation" for s in fresh.sections)
    assert not any(s.title == "Stale" for s in fresh.sections)
