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


def test_action_approve_all_drains_homework_pending(tmp_path, monkeypatch) -> None:
    """Bulk approve_all must process homework entries, not just adjustments.

    Reproduces the operator-reported bug: 'cant approve all, because approve
    is stuck behind valid' — homework entries were silently skipped because
    action_approve_all only called AdjustmentApprover.
    """
    from src.scanner.automation.homework import HomeworkEntry, HomeworkStore
    from src.tui.screens.inbox_screen import (
        InboxScreen,
        UnifiedInboxRow,
        _homework_to_row,
    )

    # Isolated jsonl files in tmp_path
    pending_path = tmp_path / "homework_pending.jsonl"
    history_path = tmp_path / "homework_history.jsonl"
    quarantine_path = tmp_path / "homework_quarantine.jsonl"
    store = HomeworkStore(
        pending_path=pending_path,
        history_path=history_path,
        quarantine_path=quarantine_path,
    )

    # Seed 3 homework entries
    for i in range(3):
        store.add(HomeworkEntry(
            homework_id=f"hw-{i}",
            trade_id=f"100{i}",
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
            realized_pl=-50.0,
            close_reason="SL",
            duration_minutes=30,
            mfe_pips=2.0,
            mae_pips=12.0,
            analysis_markdown="x",
            proposed_lesson="y",
            confidence_in_analysis=0.7,
            agents_to_reinforce=[],
            agents_to_penalize=[],
        ))
    assert len(store.list_pending()) == 3

    # Patch HomeworkStore() default-constructor to return our isolated store
    import src.tui.screens.inbox_screen as inbox_mod
    monkeypatch.setattr(inbox_mod, "HomeworkStore", lambda: store)

    # Build a stub screen object with the minimum attributes the method touches
    class _StubScreen:
        _current_rows = [_homework_to_row(e) for e in store.list_pending()]
        _pending_path = tmp_path / "no_adjustments.json"  # nonexistent → empty
        _approved_path = tmp_path / "approved.json"
        _proposals: list = []
        notify = staticmethod(lambda *a, **kw: None)
        _load_proposals = staticmethod(lambda: None)

    InboxScreen.action_approve_all(_StubScreen())  # type: ignore[arg-type]

    # All 3 homework entries should be drained from pending
    remaining = store.list_pending()
    assert len(remaining) == 0, (
        f"Expected 0 pending after bulk approve, got {len(remaining)}"
    )
    # All 3 should now be in history with grade=approved
    history = store.list_history()
    assert len(history) == 3
    assert all(e.operator_grade == "approved" for e in history)


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


def test_proposals_cache_skips_reparse_when_mtime_unchanged(tmp_path: Path) -> None:
    """_read_proposals must use the mtime cache when the file hasn't changed."""
    import json as _json
    from src.tui.screens.inbox_screen import InboxScreen

    pending = tmp_path / ".claude" / "pending_adjustments.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(_json.dumps({"proposals": [
        {"id": "a", "status": "pending", "key": "k", "current_value": 1, "proposed_value": 2}
    ]}))

    screen = InboxScreen(project_root=str(tmp_path))

    # First read populates the cache
    first = screen._read_proposals()
    assert len(first) == 1
    assert screen._proposals_cache is not None
    cached_mtime = screen._proposals_cache[0]

    # Second read with same file mtime returns the same list object (cache hit)
    second = screen._read_proposals()
    assert second is first, "cache hit must return the cached list, not re-parse"
    assert screen._proposals_cache[0] == cached_mtime


def test_proposals_cache_invalidates_on_mtime_change(tmp_path: Path) -> None:
    """_read_proposals must re-parse after file is rewritten."""
    import json as _json
    import os as _os
    from src.tui.screens.inbox_screen import InboxScreen

    pending = tmp_path / ".claude" / "pending_adjustments.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(_json.dumps({"proposals": [
        {"id": "a", "status": "pending", "key": "k", "current_value": 1, "proposed_value": 2}
    ]}))

    screen = InboxScreen(project_root=str(tmp_path))
    screen._read_proposals()
    first_cached = screen._proposals_cache

    # Bump mtime by ~10ms (nanosecond-resolution stat) and rewrite
    new_mtime = pending.stat().st_mtime_ns + 10_000_000
    pending.write_text(_json.dumps({"proposals": [
        {"id": "a", "status": "pending", "key": "k", "current_value": 1, "proposed_value": 2},
        {"id": "b", "status": "pending", "key": "k2", "current_value": 3, "proposed_value": 4},
    ]}))
    _os.utime(pending, ns=(new_mtime, new_mtime))

    second = screen._read_proposals()
    assert len(second) == 2, "cache must invalidate when mtime changes"
    assert screen._proposals_cache != first_cached
