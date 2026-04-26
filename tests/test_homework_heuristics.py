"""Tests for HEURISTIC_CATALOG. Validates per-category coverage + per-heuristic firing."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.scanner.automation.homework.heuristics import HEURISTIC_CATALOG
from src.scanner.automation.homework.types import Heuristic


def _trade(**kw):
    """Lightweight TradeView for predicate testing."""
    base = dict(
        trade_id="t1", pair="EUR_USD", direction="LONG",
        entry_price=1.0, sl_price=0.99, tp_price=1.02,
        rr_ratio=2.0, confidence=0.65, weighted_vote_score=0.70,
        regime="NORMAL", adx=20.0, rsi=50.0, atr_pips=10.0,
        sl_pips=10.0, tp_pips=20.0, spread_pips=1.0,
        agent_verdicts=[
            {"name": "trend", "passed": True, "score": 0.6, "weight": 1.15},
            {"name": "mean_reversion", "passed": True, "score": 0.55, "weight": 0.90},
        ],
        gate_details={"model_disagreement": 0.20, "disagreement_hard_floor": 0.50},
        oldest_age_days=3.0, news_window=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _outcome(**kw):
    """Lightweight OutcomeView for predicate testing."""
    base = dict(
        close_reason="SL", realized_pl=-100.0, duration_minutes=30,
        mfe_pips=2.0, mae_pips=12.0, close_price=0.99,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestCatalogStructure:
    def test_all_six_categories_present(self) -> None:
        cats = {h.category for h in HEURISTIC_CATALOG}
        assert cats == {"A", "B", "C", "D", "E", "F"}

    def test_total_heuristic_count(self) -> None:
        # Spec §4.3 declares ~25 heuristics. Lock at >= 22 to allow minor pruning.
        assert len(HEURISTIC_CATALOG) >= 22

    def test_unique_ids(self) -> None:
        ids = [h.id for h in HEURISTIC_CATALOG]
        assert len(ids) == len(set(ids)), "Heuristic IDs must be unique"

    def test_every_heuristic_has_source(self) -> None:
        """Per spec §4.6 — every heuristic carries a source citation."""
        for h in HEURISTIC_CATALOG:
            assert h.source.strip(), f"{h.id} {h.name} missing source citation"

    def test_confidence_in_range(self) -> None:
        for h in HEURISTIC_CATALOG:
            assert 0.0 <= h.confidence <= 1.0, f"{h.id} confidence out of range"


class TestSetupValidityHeuristics:
    """Category A — Setup Validity."""

    def test_A1_adx_trend_mismatch_fires_on_low_adx_directional_loss(self) -> None:
        h = _by_id("A1")
        trade = _trade(direction="LONG", adx=4.0)
        outcome = _outcome(close_reason="SL")
        assert h.predicate(trade, outcome) is True

    def test_A1_does_not_fire_when_adx_high(self) -> None:
        h = _by_id("A1")
        trade = _trade(direction="LONG", adx=25.0)
        outcome = _outcome(close_reason="SL")
        assert h.predicate(trade, outcome) is False

    def test_A1_does_not_fire_on_winner(self) -> None:
        h = _by_id("A1")
        trade = _trade(direction="LONG", adx=4.0)
        outcome = _outcome(close_reason="TP")
        assert h.predicate(trade, outcome) is False


class TestRiskCalibrationHeuristics:
    """Category B — Risk Calibration."""

    def test_B3_low_regime_sl_violation_fires(self) -> None:
        h = _by_id("B3")
        trade = _trade(regime="LOW", sl_pips=8.0, atr_pips=10.0)  # sl_mult = 0.8 < 1.2
        outcome = _outcome(close_reason="SL")
        assert h.predicate(trade, outcome) is True

    def test_B3_does_not_fire_when_sl_mult_compliant(self) -> None:
        h = _by_id("B3")
        trade = _trade(regime="LOW", sl_pips=15.0, atr_pips=10.0)  # sl_mult = 1.5 >= 1.2
        outcome = _outcome(close_reason="SL")
        assert h.predicate(trade, outcome) is False


class TestConsensusHeuristics:
    """Category C — Agent Consensus Quality."""

    def test_C1_trend_veto_unhonored_fires(self) -> None:
        h = _by_id("C1")
        trade = _trade(
            direction="LONG",
            agent_verdicts=[
                {"name": "trend", "passed": False, "score": 0.45, "weight": 1.15},
            ],
        )
        outcome = _outcome(close_reason="SL")
        assert h.predicate(trade, outcome) is True

    def test_C2_mr_composite_match_fires(self) -> None:
        h = _by_id("C2")
        trade = _trade(
            direction="LONG",
            agent_verdicts=[
                {"name": "mean_reversion", "passed": False, "score": 0.45, "weight": 0.90},
            ],
            gate_details={"model_disagreement": 0.30, "disagreement_hard_floor": 0.50},
        )
        outcome = _outcome(close_reason="SL")
        assert h.predicate(trade, outcome) is True


class TestExecutionHeuristics:
    """Category D — Execution Quality."""

    def test_D1_mfe_zero_fires_on_directional_loss(self) -> None:
        h = _by_id("D1")
        trade = _trade(direction="LONG", atr_pips=10.0)
        outcome = _outcome(close_reason="SL", mfe_pips=1.0)  # 1/10 = 0.1 < 0.2
        assert h.predicate(trade, outcome) is True

    def test_D4_fast_sl_bad_timing_fires(self) -> None:
        h = _by_id("D4")
        outcome = _outcome(close_reason="SL", duration_minutes=3)
        assert h.predicate(_trade(), outcome) is True


class TestContextHeuristics:
    """Category E — Regime / Context Drift."""

    def test_E1_stale_models_fires_on_loss_with_stale_models(self) -> None:
        h = _by_id("E1")
        trade = _trade(oldest_age_days=8.0)
        outcome = _outcome(close_reason="SL")
        assert h.predicate(trade, outcome) is True


class TestMetaPatterns:
    """Category F — Meta-Patterns."""

    def test_F2_lucky_winner_fires(self) -> None:
        h = _by_id("F2")
        trade = _trade(sl_pips=10.0)
        outcome = _outcome(close_reason="TP", mae_pips=8.0)  # 8/10 = 0.8 > 0.7
        assert h.predicate(trade, outcome) is True


def _by_id(hid: str) -> Heuristic:
    matches = [h for h in HEURISTIC_CATALOG if h.id == hid]
    assert len(matches) == 1, f"Heuristic {hid} not found"
    return matches[0]
