"""Tests for HomeworkGenerator — closed trade + outcome → HomeworkEntry."""
from __future__ import annotations

import pytest

from src.scanner.automation.homework.generator import HomeworkGenerator
from src.scanner.automation.homework.types import HomeworkEntry


# Synthetic trade journal entry mirroring trained_data/trade_journal_rl.json shape
def _trade_dict_04_15_streak_loss() -> dict:
    return {
        "trade_id": "1220",
        "pair": "EUR_AUD",
        "direction": "SHORT",
        "entry_price": 1.6543,
        "sl_price": 1.6580,
        "tp_price": 1.6470,
        "sl_pips": 37.0,
        "tp_pips": 73.0,
        "rr_ratio": 1.97,
        "confidence": 0.68,
        "weighted_vote_score": 0.76,
        "regime": "NORMAL",
        "agents": [
            {"name": "trend", "passed": False, "score": 0.45, "weight": 1.15, "reason": "ADX 1"},
            {"name": "mean_reversion", "passed": True, "score": 0.55, "weight": 0.90, "reason": "RSI 48"},
            {"name": "volatility", "passed": True, "score": 0.71, "weight": 1.0, "reason": "atr ok"},
            {"name": "risk_sentinel", "passed": True, "score": 0.83, "weight": 1.05, "reason": "drawdown ok"},
        ],
        "gate_details": {
            "model_disagreement": 0.20,
            "disagreement_hard_floor": 0.50,
            "adx": 1.0,
            "rsi": 48.0,
            "atr_pips": 12.3,
        },
        "spread_pips": 1.4,
    }


def _outcome_dict_stop_loss() -> dict:
    return {
        "close_time": "2026-04-15T02:46:03Z",
        "close_price": 1.6580,
        "realized_pl": -354.56,
        "close_reason": "SL",
        "duration_minutes": 32,
        "mfe_pips": 4.0,
        "mae_pips": 39.0,
    }


class TestHomeworkGeneratorBasics:
    def test_generate_returns_homework_entry(self) -> None:
        gen = HomeworkGenerator()
        entry = gen.generate(_trade_dict_04_15_streak_loss(), _outcome_dict_stop_loss())
        assert isinstance(entry, HomeworkEntry)
        assert entry.trade_id == "1220"
        assert entry.realized_pl == -354.56

    def test_generated_id_is_unique(self) -> None:
        gen = HomeworkGenerator()
        e1 = gen.generate(_trade_dict_04_15_streak_loss(), _outcome_dict_stop_loss())
        e2 = gen.generate(_trade_dict_04_15_streak_loss(), _outcome_dict_stop_loss())
        assert e1.homework_id != e2.homework_id

    def test_status_is_pending(self) -> None:
        gen = HomeworkGenerator()
        entry = gen.generate(_trade_dict_04_15_streak_loss(), _outcome_dict_stop_loss())
        assert entry.status == "pending"


class TestHomeworkGeneratorPicksPrimaryLesson:
    def test_04_15_loss_fingerprint_picks_C1_or_A1(self) -> None:
        """The trade #1220 loss should match A1 (ADX=1) and C1 (trend veto unhonored)."""
        gen = HomeworkGenerator()
        entry = gen.generate(_trade_dict_04_15_streak_loss(), _outcome_dict_stop_loss())
        assert entry.proposed_lesson  # not empty
        # Both A1 and C1 should fire; C1 has higher confidence (0.90 vs 0.85)
        assert "trend" in entry.proposed_lesson.lower() or "adx" in entry.proposed_lesson.lower()


class TestHomeworkGeneratorAgentScoring:
    def test_winner_reinforces_passing_agents(self) -> None:
        gen = HomeworkGenerator()
        trade = _trade_dict_04_15_streak_loss()
        outcome = _outcome_dict_stop_loss()
        outcome["close_reason"] = "TP"
        outcome["realized_pl"] = 200.0
        entry = gen.generate(trade, outcome)
        # Agents that passed=True on a TP should be reinforced
        assert "mean_reversion" in entry.agents_to_reinforce
        assert "volatility" in entry.agents_to_reinforce

    def test_loss_penalizes_passing_agents(self) -> None:
        gen = HomeworkGenerator()
        entry = gen.generate(_trade_dict_04_15_streak_loss(), _outcome_dict_stop_loss())
        # On SL, agents that voted YES were wrong — penalized
        assert "mean_reversion" in entry.agents_to_penalize
        # trend voted NO (which was correct) — should be reinforced
        assert "trend" in entry.agents_to_reinforce


class TestHomeworkGeneratorMarkdown:
    def test_markdown_contains_outcome_block(self) -> None:
        gen = HomeworkGenerator()
        entry = gen.generate(_trade_dict_04_15_streak_loss(), _outcome_dict_stop_loss())
        md = entry.analysis_markdown
        assert "EUR_AUD" in md
        assert "STOPPED OUT" in md or "SL" in md
        assert "-354" in md or "−354" in md or "354.56" in md

    def test_markdown_contains_setup_section(self) -> None:
        gen = HomeworkGenerator()
        entry = gen.generate(_trade_dict_04_15_streak_loss(), _outcome_dict_stop_loss())
        md = entry.analysis_markdown
        assert "Setup" in md or "setup" in md
        assert "1.6543" in md  # entry price
