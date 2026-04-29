"""Mythos audit 2026-04-29 Tier-7 step C — meta-pipeline events surface
in the TUI brain log via a tail-watcher on .claude/meta/changes.jsonl.

Pre-fix: brain feed was in-memory only and disconnected from the
meta-pipeline. The cybernetic loop ran invisibly even after step A
(orchestrator routing) and step B (Inbox [Meta] filter) made packages
exist and be visible in the queue. The brain feed is the F1 live
running narrative — until meta events flowed there, the operator
couldn't *watch* the loop fire in real time.

Tests cover the format function directly (extracted as a free
function for unit testing — Textual App lifecycle is heavyweight).
The tail-loop integration is covered by black-box manual smoke;
extracted helpers ensure the formatting logic is provably correct.
"""
from __future__ import annotations

import pytest

from src.tui.app import format_meta_brain_line


def test_intake_stage_uses_cyan_routine_color():
    """Default stages (intake, proposing, awaiting_approval) → cyan ▸."""
    line = format_meta_brain_line({
        "change_id": "abcdef0123456789",
        "stage": "intake",
        "kind": "config",
    })
    assert "[cyan]" in line
    assert "▸" in line
    assert "⟪abcdef01⟫" in line  # truncated to 8 chars
    assert "intake" in line
    assert "kind=config" in line


def test_rejected_stage_uses_red_failure_color():
    line = format_meta_brain_line({
        "change_id": "deadbeef",
        "stage": "rejected",
        "event": "constitution_blocked",
    })
    assert "[red]" in line
    assert "✗" in line
    assert "event=constitution_blocked" in line


def test_aborted_stage_uses_red():
    line = format_meta_brain_line({
        "change_id": "ed9d3336cd2e",
        "stage": "aborted",
        "event": "throttled",
        "kind": "config",
    })
    assert "[red]" in line
    assert "✗" in line


def test_deployed_live_stage_uses_green_success():
    line = format_meta_brain_line({
        "change_id": "feedface",
        "stage": "deployed_live",
        "kind": "config",
    })
    assert "[green]" in line
    assert "✓" in line


def test_approved_stage_uses_green():
    line = format_meta_brain_line({
        "change_id": "cafebabe",
        "stage": "approved",
    })
    assert "[green]" in line
    assert "✓" in line


def test_shadow_canary_stages_use_magenta():
    """Shadow / canary deploys are visually distinct from final live —
    magenta ◆ marks "in-flight rollout"."""
    for stage in ("deployed_shadow", "deployed_canary"):
        line = format_meta_brain_line({
            "change_id": "0badc0de",
            "stage": stage,
        })
        assert "[magenta]" in line, f"stage={stage} did not get magenta"
        assert "◆" in line


def test_missing_change_id_renders_question_mark():
    """Defensive — corrupt or partial ledger entry must not crash the
    formatter."""
    line = format_meta_brain_line({"stage": "intake"})
    assert "⟪?⟫" in line


def test_missing_stage_renders_question_mark():
    line = format_meta_brain_line({"change_id": "12345678abcd"})
    assert "?" in line


def test_unknown_stage_falls_to_cyan_routine():
    """A stage value the formatter doesn't know about (forward-compat
    when meta_types adds new stages) defaults to cyan ▸ — operator
    still sees the event."""
    line = format_meta_brain_line({
        "change_id": "11223344",
        "stage": "some_future_stage",
    })
    assert "[cyan]" in line
    assert "▸" in line


def test_event_field_appears_when_present():
    line = format_meta_brain_line({
        "change_id": "aabbccdd",
        "stage": "intake",
        "event": "duplicate_dropped",
    })
    assert "event=duplicate_dropped" in line


def test_kind_field_appears_when_present():
    line = format_meta_brain_line({
        "change_id": "11112222",
        "stage": "proposing",
        "kind": "model",
    })
    assert "kind=model" in line


def test_change_id_truncated_to_eight_chars():
    """Long change_ids would blow up the brain log line — truncate."""
    line = format_meta_brain_line({
        "change_id": "abcdef0123456789ffffffff",
        "stage": "intake",
    })
    assert "⟪abcdef01⟫" in line
    assert "23456789" not in line


def test_short_change_id_renders_in_full():
    """change_id under 8 chars must still render correctly (e.g. test
    fixtures, hand-crafted incident)."""
    line = format_meta_brain_line({
        "change_id": "ab",
        "stage": "intake",
    })
    assert "⟪ab⟫" in line


def test_format_includes_meta_keyword_for_filterability():
    """Brain log already shows many event types; tagging meta events
    with the literal 'meta' lets operators grep."""
    line = format_meta_brain_line({
        "change_id": "abcdef01",
        "stage": "intake",
    })
    assert "meta" in line


def test_no_unclosed_markup_tags():
    """Rich markup must be balanced — unclosed [color] tags break the
    log rendering for subsequent lines."""
    for stage in ("intake", "rejected", "deployed_live", "deployed_shadow", "weird"):
        line = format_meta_brain_line({
            "change_id": "12345678",
            "stage": stage,
        })
        # Count opening vs closing tags at our brackets
        opens = line.count("[")
        closes = line.count("]")
        assert opens == closes, f"stage={stage} unbalanced markup: {line!r}"
        # Markup must end with [/] closing tag
        assert line.endswith("[/]")
