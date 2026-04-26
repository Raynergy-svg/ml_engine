"""Tests for HomeworkReviewer — A/R/E/S transitions + training signal emit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from src.scanner.automation.homework.applicator import TrainingSignalApplicator
from src.scanner.automation.homework.generator import HomeworkGenerator
from src.scanner.automation.homework.reviewer import HomeworkReviewer
from src.scanner.automation.homework.store import HomeworkStore
from src.scanner.automation.homework.types import HomeworkEntry, TrainingSignal


@pytest.fixture
def store(tmp_path: Path) -> HomeworkStore:
    return HomeworkStore(
        pending_path=tmp_path / "homework_pending.jsonl",
        history_path=tmp_path / "homework_history.jsonl",
        quarantine_path=tmp_path / "quarantine.jsonl",
    )


@pytest.fixture(autouse=True)
def isolate_weights(tmp_path: Path, monkeypatch) -> Path:
    """Redirect TrainingSignalApplicator's default weights path into tmp_path so
    HomeworkReviewer(store=store) (no explicit applicator) cannot mutate production
    agent_weights.json during reviewer unit tests."""
    weights_file = tmp_path / "agent_weights.json"
    weights_file.write_text(json.dumps({
        "NORMAL": {"trend": 1.0, "mean_reversion": 0.9},
    }))
    monkeypatch.setattr(
        "src.scanner.automation.homework.applicator.DEFAULT_WEIGHTS_PATH",
        weights_file,
    )
    return weights_file


@pytest.fixture
def sample_entry(store: HomeworkStore) -> HomeworkEntry:
    gen = HomeworkGenerator()
    trade = {
        "trade_id": "1220", "pair": "EUR_AUD", "direction": "SHORT",
        "entry_price": 1.6543, "sl_price": 1.6580, "tp_price": 1.6470,
        "sl_pips": 37.0, "tp_pips": 73.0, "rr_ratio": 1.97,
        "confidence": 0.68, "weighted_vote_score": 0.76, "regime": "NORMAL",
        "agents": [{"name": "trend", "passed": False, "score": 0.45, "weight": 1.15}],
        "gate_details": {"adx": 1.0, "rsi": 48.0, "atr_pips": 12.3,
                         "model_disagreement": 0.20, "disagreement_hard_floor": 0.50},
    }
    outcome = {
        "close_time": "2026-04-15T02:46:03Z",
        "close_price": 1.6580, "realized_pl": -354.56,
        "close_reason": "SL", "duration_minutes": 32,
        "mfe_pips": 4.0, "mae_pips": 39.0,
    }
    entry = gen.generate(trade, outcome)
    store.add(entry)
    return entry


class TestReviewerApprove:
    def test_approve_moves_entry_to_history(
        self, store: HomeworkStore, sample_entry: HomeworkEntry
    ) -> None:
        reviewer = HomeworkReviewer(store=store)
        signal = reviewer.approve(sample_entry.homework_id)
        assert signal is not None
        assert signal.operator_action == "approved"
        # Entry no longer in pending
        assert all(e.homework_id != sample_entry.homework_id for e in store.list_pending())
        # Entry IS in history
        history = store.list_history()
        assert any(e.homework_id == sample_entry.homework_id for e in history)
        graded = next(e for e in history if e.homework_id == sample_entry.homework_id)
        assert graded.operator_grade == "approved"

    def test_approve_emits_signal_with_proposed_deltas(
        self, store: HomeworkStore, sample_entry: HomeworkEntry
    ) -> None:
        reviewer = HomeworkReviewer(store=store)
        signal = reviewer.approve(sample_entry.homework_id)
        # trend voted NO on a SL outcome → should be reinforced
        assert "trend" in signal.agent_weight_deltas
        assert signal.agent_weight_deltas["trend"] > 0


class TestReviewerReject:
    def test_reject_requires_note(
        self, store: HomeworkStore, sample_entry: HomeworkEntry
    ) -> None:
        reviewer = HomeworkReviewer(store=store)
        with pytest.raises(ValueError, match="reject.*note"):
            reviewer.reject(sample_entry.homework_id, note="")

    def test_reject_records_note_in_history(
        self, store: HomeworkStore, sample_entry: HomeworkEntry, tmp_path: Path
    ) -> None:
        reviewer = HomeworkReviewer(
            store=store,
            rejected_log_path=tmp_path / "rejected_heuristics.jsonl",
        )
        signal = reviewer.reject(sample_entry.homework_id, note="trend wasn't the issue, ADX was 1 by luck")
        history = store.list_history()
        graded = next(e for e in history if e.homework_id == sample_entry.homework_id)
        assert graded.operator_grade == "rejected"
        assert "ADX was 1 by luck" in (graded.operator_note or "")
        # Rejected log captured the heuristic that fired
        if (tmp_path / "rejected_heuristics.jsonl").exists():
            log_text = (tmp_path / "rejected_heuristics.jsonl").read_text()
            assert sample_entry.homework_id in log_text


class TestReviewerEdit:
    def test_edit_replaces_proposed_deltas(
        self, store: HomeworkStore, sample_entry: HomeworkEntry
    ) -> None:
        reviewer = HomeworkReviewer(store=store)
        custom_edits = {
            "agent_weight_deltas": {"trend": 0.05, "weighted_vote_score": -0.03},
            "note": "trend deserves bigger reinforce here",
        }
        signal = reviewer.edit(sample_entry.homework_id, edits=custom_edits)
        assert signal.operator_action == "edited"
        assert signal.agent_weight_deltas["trend"] == 0.05
        assert signal.agent_weight_deltas["weighted_vote_score"] == -0.03


class TestReviewerSnooze:
    def test_snooze_keeps_entry_in_pending(
        self, store: HomeworkStore, sample_entry: HomeworkEntry
    ) -> None:
        reviewer = HomeworkReviewer(store=store)
        ok = reviewer.snooze(sample_entry.homework_id, hours=24)
        assert ok is True
        # Still in pending
        pending = store.list_pending()
        snoozed = next((e for e in pending if e.homework_id == sample_entry.homework_id), None)
        assert snoozed is not None
        assert snoozed.status.startswith("snoozed_until_")


class TestReviewerErrors:
    def test_unknown_id_returns_none(self, store: HomeworkStore) -> None:
        reviewer = HomeworkReviewer(store=store)
        signal = reviewer.approve("does-not-exist")
        assert signal is None
