"""Tests for HomeworkStore — atomic .jsonl I/O for pending and history files."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.homework.store import HomeworkStore
from src.scanner.automation.homework.types import HomeworkEntry


def _make_entry(trade_id: str = "1220", **overrides) -> HomeworkEntry:
    """Minimal HomeworkEntry factory for tests."""
    base = dict(
        homework_id=f"hw-{trade_id}",
        trade_id=trade_id,
        generated_at="2026-04-25T22:47:35Z",
        pair="EUR_AUD", direction="SHORT",
        entry_price=1.6543, sl_price=1.6580, tp_price=1.6470, rr_ratio=1.97,
        confidence=0.68, weighted_vote_score=0.76, regime="NORMAL",
        agent_verdicts=[],
        close_time="2026-04-15T02:46:03Z",
        close_price=1.6580, realized_pl=-354.56, close_reason="SL",
        duration_minutes=32, mfe_pips=4.0, mae_pips=39.0,
        analysis_markdown="# Analysis...", proposed_lesson="hard-veto trend",
        confidence_in_analysis=0.70,
        agents_to_reinforce=["trend"], agents_to_penalize=["weighted_vote_score"],
    )
    base.update(overrides)
    return HomeworkEntry(**base)


class TestHomeworkStoreAddAndList:
    def test_add_entry_appears_in_pending(self, tmp_path: Path) -> None:
        store = HomeworkStore(
            pending_path=tmp_path / "homework_pending.jsonl",
            history_path=tmp_path / "homework_history.jsonl",
        )
        entry = _make_entry(trade_id="1220")
        store.add(entry)
        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0].trade_id == "1220"

    def test_atomic_write_uses_tmp_rename(self, tmp_path: Path) -> None:
        """Verify add() does not leave .tmp files behind on success."""
        store = HomeworkStore(
            pending_path=tmp_path / "homework_pending.jsonl",
            history_path=tmp_path / "homework_history.jsonl",
        )
        store.add(_make_entry(trade_id="1"))
        store.add(_make_entry(trade_id="2"))
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []

    def test_list_pending_skips_corrupt_lines(self, tmp_path: Path) -> None:
        """Corrupt JSONL lines must be quarantined, not crash the read."""
        pending = tmp_path / "homework_pending.jsonl"
        pending.write_text(
            json.dumps({"homework_id": "ok", "trade_id": "1", "generated_at": "z",
                        "pair": "EUR_USD", "direction": "LONG", "entry_price": 1.0,
                        "sl_price": 1.0, "tp_price": 1.0, "rr_ratio": 1.0,
                        "confidence": 0.5, "weighted_vote_score": 0.5,
                        "regime": "NORMAL", "agent_verdicts": [],
                        "close_time": "", "close_price": 1.0, "realized_pl": 0.0,
                        "close_reason": "TP", "duration_minutes": 0,
                        "mfe_pips": 0.0, "mae_pips": 0.0,
                        "analysis_markdown": "", "proposed_lesson": "",
                        "confidence_in_analysis": 0.0,
                        "agents_to_reinforce": [], "agents_to_penalize": []}) + "\n"
            "{not valid json garbage\n"
        )
        store = HomeworkStore(
            pending_path=pending,
            history_path=tmp_path / "homework_history.jsonl",
            quarantine_path=tmp_path / "quarantine.jsonl",
        )
        entries = store.list_pending()
        assert len(entries) == 1
        assert entries[0].trade_id == "1"
        # Corrupt line went to quarantine
        quarantine_text = (tmp_path / "quarantine.jsonl").read_text()
        assert "not valid json garbage" in quarantine_text


class TestHomeworkStoreMoveToHistory:
    def test_move_to_history_removes_from_pending(self, tmp_path: Path) -> None:
        store = HomeworkStore(
            pending_path=tmp_path / "homework_pending.jsonl",
            history_path=tmp_path / "homework_history.jsonl",
        )
        store.add(_make_entry(trade_id="1"))
        store.add(_make_entry(trade_id="2"))
        moved = store.move_to_history(
            "hw-1",
            grade="approved",
            note=None,
            edits=None,
        )
        assert moved is True
        remaining = store.list_pending()
        assert len(remaining) == 1
        assert remaining[0].trade_id == "2"
        history = store.list_history()
        assert len(history) == 1
        assert history[0].trade_id == "1"
        assert history[0].operator_grade == "approved"

    def test_move_to_history_unknown_id_returns_false(self, tmp_path: Path) -> None:
        store = HomeworkStore(
            pending_path=tmp_path / "homework_pending.jsonl",
            history_path=tmp_path / "homework_history.jsonl",
        )
        moved = store.move_to_history("nonexistent", grade="approved", note=None, edits=None)
        assert moved is False


class TestHomeworkStoreCrashRecovery:
    """Simulate the crash window between history-append and pending-rewrite."""

    def test_list_pending_dedupes_against_history(self, tmp_path: Path) -> None:
        """If an id is in BOTH pending and history (crash mid-move), list_pending hides it."""
        store = HomeworkStore(
            pending_path=tmp_path / "homework_pending.jsonl",
            history_path=tmp_path / "homework_history.jsonl",
        )
        # Simulate the crash state: same entry in both files
        store.add(_make_entry(trade_id="1220"))
        # Manually append a history copy to simulate post-history-pre-rewrite crash
        import dataclasses
        entry = _make_entry(trade_id="1220")
        store._append_locked(store.history_path, dataclasses.asdict(entry))

        # Crash-recovery contract: list_pending no longer returns the duplicate
        pending = store.list_pending()
        assert pending == [], "list_pending must filter out ids already in history"

    def test_move_to_history_retry_after_crash_is_idempotent(self, tmp_path: Path) -> None:
        """Calling move_to_history a second time after a partial crash converges."""
        store = HomeworkStore(
            pending_path=tmp_path / "homework_pending.jsonl",
            history_path=tmp_path / "homework_history.jsonl",
        )
        store.add(_make_entry(trade_id="1220"))

        # Simulate crash state: history has the entry but pending was not rewritten
        import dataclasses
        graded = dataclasses.replace(
            _make_entry(trade_id="1220"),
            status="approved",
            operator_grade="approved",
            reviewed_at="2026-04-26T01:00:00Z",
        )
        store._append_locked(store.history_path, dataclasses.asdict(graded))

        # Retry: the second move should NOT double-append to history,
        # but SHOULD finish the rewrite-pending step.
        moved = store.move_to_history("hw-1220", grade="approved", note=None, edits=None)
        assert moved is True

        history = store.list_history()
        assert len(history) == 1, "history must not double-count after recovery retry"
        # Pending file is now clean
        # (use raw read to bypass dedupe filter — file should be empty)
        assert store._read_jsonl(store.pending_path) == []
