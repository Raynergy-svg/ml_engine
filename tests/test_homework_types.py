"""Tests for HomeworkEntry + Heuristic + TrainingSignal dataclasses."""
from __future__ import annotations

import pytest

from src.scanner.automation.homework.types import (
    HomeworkEntry,
    Heuristic,
    TrainingSignal,
)


class TestHomeworkEntry:
    def test_required_fields_present(self) -> None:
        entry = HomeworkEntry(
            homework_id="abc-123",
            trade_id="1220",
            generated_at="2026-04-25T22:47:35Z",
            pair="EUR_AUD",
            direction="SHORT",
            entry_price=1.6543,
            sl_price=1.6580,
            tp_price=1.6470,
            rr_ratio=1.97,
            confidence=0.68,
            weighted_vote_score=0.76,
            regime="NORMAL",
            agent_verdicts=[],
            close_time="2026-04-15T02:46:03Z",
            close_price=1.6580,
            realized_pl=-354.56,
            close_reason="SL",
            duration_minutes=32,
            mfe_pips=4.0,
            mae_pips=39.0,
            analysis_markdown="...",
            proposed_lesson="hard-veto trend when ADX < 5",
            confidence_in_analysis=0.70,
            agents_to_reinforce=["trend"],
            agents_to_penalize=["weighted_vote_score"],
        )
        assert entry.status == "pending"
        assert entry.schema_version == 1
        assert entry.operator_grade is None

    def test_frozen_dataclass(self) -> None:
        """Core entry is immutable — review state transitions create new entries."""
        entry = HomeworkEntry(
            homework_id="x", trade_id="y", generated_at="z",
            pair="EUR_USD", direction="LONG", entry_price=1.0, sl_price=1.0,
            tp_price=1.0, rr_ratio=1.0, confidence=0.5, weighted_vote_score=0.5,
            regime="NORMAL", agent_verdicts=[], close_time="",
            close_price=1.0, realized_pl=0.0, close_reason="TP",
            duration_minutes=0, mfe_pips=0.0, mae_pips=0.0,
            analysis_markdown="", proposed_lesson="", confidence_in_analysis=0.0,
            agents_to_reinforce=[], agents_to_penalize=[],
        )
        with pytest.raises((AttributeError, Exception)):
            entry.confidence = 0.9  # type: ignore[misc]


class TestHeuristic:
    def test_heuristic_dataclass(self) -> None:
        h = Heuristic(
            id="A1",
            name="setup_adx_trend_mismatch",
            category="A",
            predicate=lambda t, o: True,
            lesson_template="ADX={adx} too low for directional trade",
            confidence=0.85,
            source="Bellafiore One Good Trade Ch.4",
        )
        assert h.category == "A"
        assert h.confidence == 0.85
        assert h.source.startswith("Bellafiore")


class TestTrainingSignal:
    def test_training_signal_payload(self) -> None:
        sig = TrainingSignal(
            homework_id="abc",
            trade_id="1220",
            outcome="SL",
            agent_weight_deltas={"trend": 0.02, "weighted_vote_score": -0.01},
            regime_prior_deltas={"NORMAL": {"min_confidence": 1.0}},
            heuristic_fired="C1",
            operator_action="approved",
            operator_note=None,
        )
        assert sig.outcome == "SL"
        assert sig.agent_weight_deltas["trend"] == 0.02
