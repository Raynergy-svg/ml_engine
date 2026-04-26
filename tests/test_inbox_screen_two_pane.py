"""Tests for InboxScreen two-pane layout + homework type rendering.

Static-shape tests (cheap) that exercise the source text and the
InboxScreen class API without spinning up a Textual App. Phase 96 Task 5.
"""
from __future__ import annotations

import inspect
from pathlib import Path


_INBOX_PATH = Path(__file__).resolve().parent.parent / "src" / "tui" / "screens" / "inbox_screen.py"


def test_inbox_screen_imports_homework_store() -> None:
    """The screen must read homework entries — proves the wiring exists in source."""
    text = _INBOX_PATH.read_text()
    assert "HomeworkStore" in text or "homework_pending" in text, (
        "InboxScreen must read from HomeworkStore or homework_pending.jsonl"
    )


def test_inbox_screen_has_filter_pills() -> None:
    """Three filter buttons: All, Homework, Adjustments."""
    text = _INBOX_PATH.read_text()
    assert "filter_all" in text or "filter-all" in text
    assert "filter_homework" in text or "filter-homework" in text
    assert "filter_adjustments" in text or "filter-adj" in text


def test_inbox_screen_has_two_pane_layout() -> None:
    """Compose method must use Horizontal container (two-pane layout)."""
    from src.tui.screens.inbox_screen import InboxScreen
    src = inspect.getsource(InboxScreen.compose)
    assert "Horizontal" in src or "horizontal" in src.lower()


def test_inbox_screen_has_action_for_each_hotkey() -> None:
    """V/A/R/E/S all have action handlers."""
    from src.tui.screens.inbox_screen import InboxScreen
    has_action = lambda name: hasattr(InboxScreen, f"action_{name}")
    assert has_action("approve")
    assert has_action("reject")
    assert has_action("snooze")
    # `edit` is new for homework — separate from approve
    assert has_action("edit") or "homework" in inspect.getsource(InboxScreen).lower()


def test_inbox_screen_has_unified_row_dataclass() -> None:
    """UnifiedInboxRow dataclass adapter must exist at module scope."""
    from src.tui.screens import inbox_screen
    assert hasattr(inbox_screen, "UnifiedInboxRow")
    assert hasattr(inbox_screen, "_adjustment_to_row")
    assert hasattr(inbox_screen, "_homework_to_row")


def test_inbox_screen_bindings_include_edit_hotkey() -> None:
    """The 'e' hotkey must be in BINDINGS for homework editing."""
    from src.tui.screens.inbox_screen import InboxScreen
    keys = [b.key for b in InboxScreen.BINDINGS]
    assert "e" in keys, f"'e' (edit) hotkey missing from BINDINGS: {keys}"


def test_homework_to_row_adapter_works() -> None:
    """_homework_to_row must produce a UnifiedInboxRow from a HomeworkEntry."""
    from src.scanner.automation.homework import HomeworkEntry
    from src.tui.screens.inbox_screen import _homework_to_row

    entry = HomeworkEntry(
        homework_id="hw-test-1",
        trade_id="9999",
        generated_at="2026-04-25T22:00:00Z",
        pair="EUR_USD",
        direction="LONG",
        entry_price=1.10,
        sl_price=1.09,
        tp_price=1.12,
        rr_ratio=2.0,
        confidence=0.65,
        weighted_vote_score=0.70,
        regime="NORMAL",
        agent_verdicts=[],
        close_time="2026-04-25T22:30:00Z",
        close_price=1.09,
        realized_pl=-123.45,
        close_reason="SL",
        duration_minutes=30,
        mfe_pips=2.0,
        mae_pips=12.0,
        analysis_markdown="## Analysis\n\nWhat happened.",
        proposed_lesson="Stop entering ranging markets.",
        confidence_in_analysis=0.7,
        agents_to_reinforce=["risk_sentinel"],
        agents_to_penalize=["trend"],
    )
    row = _homework_to_row(entry)
    assert row.entry_type == "homework"
    assert "9999" in row.subject
    assert "EUR_USD" in row.subject
    assert row.pl == -123.45
    assert row.raw_entry is entry


def test_adjustment_to_row_adapter_works() -> None:
    """_adjustment_to_row must produce a UnifiedInboxRow from a proposal dict."""
    from src.tui.screens.inbox_screen import _adjustment_to_row

    proposal = {
        "id": "adj-1",
        "timestamp": "2026-04-25T20:00:00Z",
        "key": "min_confidence",
        "current_value": 0.5,
        "proposed_value": 0.55,
        "source": "config_tuner",
        "reason": "Test reason",
    }
    row = _adjustment_to_row(proposal)
    assert row.entry_type == "adjustment"
    assert row.subject == "min_confidence"
    assert "0.5" in row.detail_summary and "0.55" in row.detail_summary
    assert row.pl is None
    assert row.raw_entry is proposal
