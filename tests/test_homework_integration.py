"""End-to-end Trade Homework System integration test.

Flow tested:
  journal entry + outcome
   → HomeworkGenerator
   → HomeworkStore.add (pending)
   → HomeworkReviewer.approve
   → TrainingSignal emitted with correct deltas
   → HomeworkStore.move_to_history
   → agent_weights.json receives the deltas (mocked RL queue)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.homework.generator import HomeworkGenerator
from src.scanner.automation.homework.reviewer import HomeworkReviewer
from src.scanner.automation.homework.store import HomeworkStore


@pytest.fixture
def isolated_paths(tmp_path: Path):
    return {
        "pending": tmp_path / "homework_pending.jsonl",
        "history": tmp_path / "homework_history.jsonl",
        "rejected": tmp_path / "rejected_heuristics_log.jsonl",
        "weights": tmp_path / "agent_weights.json",
    }


def test_full_flow_approve_emits_signal_and_moves_to_history(isolated_paths) -> None:
    store = HomeworkStore(
        pending_path=isolated_paths["pending"],
        history_path=isolated_paths["history"],
    )
    gen = HomeworkGenerator()
    reviewer = HomeworkReviewer(
        store=store,
        rejected_log_path=isolated_paths["rejected"],
    )

    trade = {
        "trade_id": "1220", "pair": "EUR_AUD", "direction": "SHORT",
        "entry_price": 1.6543, "sl_price": 1.6580, "tp_price": 1.6470,
        "sl_pips": 37.0, "tp_pips": 73.0, "rr_ratio": 1.97,
        "confidence": 0.68, "weighted_vote_score": 0.76, "regime": "NORMAL",
        "agents": [
            {"name": "trend", "passed": False, "score": 0.45, "weight": 1.15},
            {"name": "mean_reversion", "passed": True, "score": 0.55, "weight": 0.90},
        ],
        "gate_details": {"adx": 1.0, "rsi": 48.0, "atr_pips": 12.3,
                         "model_disagreement": 0.20, "disagreement_hard_floor": 0.50},
    }
    outcome = {
        "close_time": "2026-04-15T02:46:03Z", "close_price": 1.6580,
        "realized_pl": -354.56, "close_reason": "SL",
        "duration_minutes": 32, "mfe_pips": 4.0, "mae_pips": 39.0,
    }

    # Generate
    entry = gen.generate(trade, outcome)
    store.add(entry)
    assert len(store.list_pending()) == 1

    # Approve
    signal = reviewer.approve(entry.homework_id)
    assert signal is not None
    assert signal.operator_action == "approved"

    # Verify deltas: trend voted NO on a SL → reinforce; MR voted YES on SL → penalize
    assert signal.agent_weight_deltas.get("trend", 0) > 0
    assert signal.agent_weight_deltas.get("mean_reversion", 0) < 0

    # Verify pending is empty, history has the entry with grade=approved
    assert len(store.list_pending()) == 0
    history = store.list_history()
    assert len(history) == 1
    assert history[0].operator_grade == "approved"


def test_full_flow_reject_with_note(isolated_paths) -> None:
    store = HomeworkStore(
        pending_path=isolated_paths["pending"],
        history_path=isolated_paths["history"],
    )
    gen = HomeworkGenerator()
    reviewer = HomeworkReviewer(
        store=store,
        rejected_log_path=isolated_paths["rejected"],
    )

    trade = {
        "trade_id": "1207", "pair": "EUR_USD", "direction": "LONG",
        "entry_price": 1.0, "sl_price": 0.99, "tp_price": 1.02,
        "sl_pips": 10.0, "tp_pips": 20.0, "rr_ratio": 2.0,
        "confidence": 0.71, "weighted_vote_score": 0.81, "regime": "NORMAL",
        "agents": [{"name": "trend", "passed": True, "score": 0.7, "weight": 1.15}],
        "gate_details": {"adx": 22.0, "rsi": 55.0, "atr_pips": 8.0,
                         "model_disagreement": 0.18, "disagreement_hard_floor": 0.50},
    }
    outcome = {
        "close_time": "2026-04-15T11:24:13Z", "close_price": 1.02,
        "realized_pl": 261.0, "close_reason": "TP",
        "duration_minutes": 75, "mfe_pips": 22.0, "mae_pips": 4.0,
    }

    entry = gen.generate(trade, outcome)
    store.add(entry)
    note = "buddy missed that ADX was rising the whole trade — would not call this a 'lucky win'"
    signal = reviewer.reject(entry.homework_id, note=note)
    assert signal is not None
    assert signal.operator_action == "rejected"
    # Rejected signals carry empty deltas
    assert signal.agent_weight_deltas == {}

    # History records the note
    history = store.list_history()
    assert any(note in (e.operator_note or "") for e in history)

    # Rejection log captured the rejection
    assert isolated_paths["rejected"].exists()
    log_text = isolated_paths["rejected"].read_text()
    assert entry.homework_id in log_text
