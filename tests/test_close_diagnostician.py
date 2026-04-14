"""Tests for CloseDiagnostician: raw trade journal → CloseDiagnosis mapping."""

import pytest

from src.scanner.automation.close_diagnostician import (
    CloseDiagnostician,
    CloseDiagnosis,
    CloseReason,
)


@pytest.fixture
def diag():
    return CloseDiagnostician()


def _base_outcome(**overrides) -> dict:
    """Minimal valid trade outcome with sensible defaults."""
    base = {
        "trade_id": "T-001",
        "pair": "EUR_USD",
        "direction": "long",
        "exit_reason": "sl_hit",
        "entry_price": 1.0800,
        "exit_price": 1.0780,
        "realized_pl": -20.0,
        "pnl_pips": -20.0,
        "duration_minutes": 45.0,
        "regime_at_entry": "trending",
        "regime_at_exit": "trending",
        "confidence_at_entry": 0.72,
        "timestamp": "2026-04-01T12:00:00Z",
    }
    base.update(overrides)
    return base


# --- AC: SL hit maps to CloseReason.SL_HIT ---
class TestCloseReasonMapping:
    def test_sl_hit_maps_correctly(self, diag):
        result = diag.diagnose(_base_outcome(exit_reason="sl_hit"))
        assert result.close_reason == CloseReason.SL_HIT

    def test_sl_shorthand_maps_correctly(self, diag):
        result = diag.diagnose(_base_outcome(exit_reason="sl"))
        assert result.close_reason == CloseReason.SL_HIT

    # --- AC: TP hit maps to TP_HIT ---
    def test_tp_hit_maps_correctly(self, diag):
        result = diag.diagnose(_base_outcome(exit_reason="tp_hit"))
        assert result.close_reason == CloseReason.TP_HIT

    def test_tp_shorthand_maps_correctly(self, diag):
        result = diag.diagnose(_base_outcome(exit_reason="tp"))
        assert result.close_reason == CloseReason.TP_HIT

    # --- AC: unknown exit_reason defaults to MANUAL ---
    def test_unknown_exit_reason_defaults_to_manual(self, diag):
        result = diag.diagnose(_base_outcome(exit_reason="something_random"))
        assert result.close_reason == CloseReason.MANUAL

    def test_empty_exit_reason_defaults_to_manual(self, diag):
        result = diag.diagnose(_base_outcome(exit_reason=""))
        assert result.close_reason == CloseReason.MANUAL


# --- AC: regime_changed detected correctly ---
class TestRegimeChange:
    def test_regime_changed_true_when_different(self, diag):
        result = diag.diagnose(
            _base_outcome(regime_at_entry="trending", regime_at_exit="ranging")
        )
        assert result.regime_changed is True

    def test_regime_changed_false_when_same(self, diag):
        result = diag.diagnose(
            _base_outcome(regime_at_entry="trending", regime_at_exit="trending")
        )
        assert result.regime_changed is False

    def test_regime_changed_false_when_entry_unknown(self, diag):
        result = diag.diagnose(
            _base_outcome(regime_at_entry="unknown", regime_at_exit="trending")
        )
        assert result.regime_changed is False

    def test_regime_changed_false_when_exit_unknown(self, diag):
        result = diag.diagnose(
            _base_outcome(regime_at_entry="trending", regime_at_exit="unknown")
        )
        assert result.regime_changed is False


# --- AC: agents split into correct/wrong based on direction and trade_won ---
class TestAgentSplit:
    def test_winning_long_agents_agree_are_correct(self, diag):
        outcome = _base_outcome(
            direction="long",
            trade_won=True,
            agent_reasons="trend:long, momentum:long",
        )
        result = diag.diagnose(outcome)
        assert "trend" in result.agents_correct
        assert "momentum" in result.agents_correct
        assert result.agents_wrong == []

    def test_winning_long_agents_disagree_are_wrong(self, diag):
        outcome = _base_outcome(
            direction="long",
            trade_won=True,
            agent_reasons="trend:short, risk:short",
        )
        result = diag.diagnose(outcome)
        assert "trend" in result.agents_wrong
        assert "risk" in result.agents_wrong
        assert result.agents_correct == []

    def test_losing_trade_flips_classification(self, diag):
        outcome = _base_outcome(
            direction="long",
            trade_won=False,
            agent_reasons="trend:long, risk:short",
        )
        result = diag.diagnose(outcome)
        # trend agreed with direction but trade lost → wrong
        assert "trend" in result.agents_wrong
        # risk disagreed with direction and trade lost → correct
        assert "risk" in result.agents_correct

    def test_no_agent_reasons_returns_empty(self, diag):
        outcome = _base_outcome(agent_reasons="")
        result = diag.diagnose(outcome)
        assert result.agents_correct == []
        assert result.agents_wrong == []


# --- AC: missing fields get sensible defaults ---
class TestMissingFieldDefaults:
    def test_minimal_outcome_gets_defaults(self, diag):
        result = diag.diagnose({})
        assert result.trade_id == ""
        assert result.pair == ""
        assert result.direction == ""
        assert result.close_reason == CloseReason.MANUAL
        assert result.entry_price == 0.0
        assert result.exit_price == 0.0
        assert result.realized_pl == 0.0
        assert result.pnl_pips == 0.0
        assert result.duration_minutes == 0.0
        assert result.regime_at_entry == "unknown"
        assert result.regime_at_exit == "unknown"
        assert result.regime_changed is False
        assert result.confidence_at_entry == 0.0
        assert result.agents_correct == []
        assert result.agents_wrong == []
        assert result.entry_still_valid_1h is None
        assert result.market_moved_after_close_pips is None
        assert result.timestamp == ""


# --- AC: batch_diagnose processes multiple outcomes ---
class TestBatchDiagnose:
    def test_batch_diagnose_returns_list_of_diagnoses(self, diag):
        outcomes = [
            _base_outcome(trade_id="T-001", exit_reason="sl_hit"),
            _base_outcome(trade_id="T-002", exit_reason="tp_hit"),
            _base_outcome(trade_id="T-003", exit_reason="trailing_sl"),
        ]
        results = diag.batch_diagnose(outcomes)
        assert len(results) == 3
        assert all(isinstance(r, CloseDiagnosis) for r in results)
        assert results[0].close_reason == CloseReason.SL_HIT
        assert results[1].close_reason == CloseReason.TP_HIT
        assert results[2].close_reason == CloseReason.TRAILING_SL

    def test_batch_diagnose_empty_list(self, diag):
        assert diag.batch_diagnose([]) == []


# --- AC: pnl_pips and realized_pl pass through correctly ---
class TestPnlPassthrough:
    def test_pnl_pips_passthrough(self, diag):
        result = diag.diagnose(_base_outcome(pnl_pips=42.5))
        assert result.pnl_pips == 42.5

    def test_realized_pl_passthrough(self, diag):
        result = diag.diagnose(_base_outcome(realized_pl=-150.75))
        assert result.realized_pl == -150.75

    def test_negative_pnl_pips_passthrough(self, diag):
        result = diag.diagnose(_base_outcome(pnl_pips=-33.0))
        assert result.pnl_pips == -33.0
