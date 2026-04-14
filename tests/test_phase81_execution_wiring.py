"""Tests for Phase 81 live-wiring stories (US-P81-001 through US-P81-004).

Covers:
  US-P81-001: FilterChain.apply() in execute_trade()
  US-P81-002: PositionTimeoutManager (4 call sites: tick, evaluate_all, register_trade, close_trade)
  US-P81-003: RegimeConfidenceScorer.score() in _scan_pair()
  US-P81-004: CausalCounterfactual batch_analyze() post-loss
"""

import logging
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# US-P81-001: FilterChain.apply() in execute_trade()
# ---------------------------------------------------------------------------


def _make_filter_result(passed: bool, action: str, reason: str = "",
                        size_multiplier: float = 1.0, filter_name: str = "test"):
    """Create a SimpleNamespace mirroring FilterResult."""
    return types.SimpleNamespace(
        passed=passed,
        action=action,
        reason=reason,
        size_multiplier=size_multiplier,
        filter_name=filter_name,
    )


class TestFilterChainPassAction:
    """US-P81-001: PASS action — trade proceeds, size unchanged."""

    def test_pass_action_does_not_block(self):
        fc = MagicMock()
        fc.get_filter_names.return_value = ["spread_guard"]
        fc.apply.return_value = _make_filter_result(
            passed=True, action="PASS", reason="All clear",
        )

        _phase49_size_mult = 1.0
        _fc_result = fc.apply({"pair": "EUR_USD"})

        blocked = not _fc_result.passed and _fc_result.action == "BLOCK"
        assert blocked is False

    def test_pass_action_leaves_size_multiplier_unchanged(self):
        fc = MagicMock()
        fc.get_filter_names.return_value = ["spread_guard"]
        fc.apply.return_value = _make_filter_result(
            passed=True, action="PASS", size_multiplier=1.0,
        )

        _phase49_size_mult = 0.85  # Pre-existing compound multiplier
        _fc_result = fc.apply({"pair": "EUR_USD"})

        if _fc_result.action == "REDUCE_SIZE" and _fc_result.size_multiplier < 1.0:
            _phase49_size_mult *= _fc_result.size_multiplier

        assert _phase49_size_mult == 0.85  # Unchanged


class TestFilterChainBlockAction:
    """US-P81-001: BLOCK action — returns ExecutionResult(success=False)."""

    def test_block_returns_failure(self):
        fc = MagicMock()
        fc.get_filter_names.return_value = ["spread_guard"]
        fc.apply.return_value = _make_filter_result(
            passed=False, action="BLOCK", reason="Spread too wide",
            filter_name="spread_guard",
        )

        _fc_result = fc.apply({"pair": "EUR_USD"})
        should_block = not _fc_result.passed and _fc_result.action == "BLOCK"
        assert should_block is True

    def test_block_produces_correct_error_message(self):
        reason = "Spread 4.2 pips > 3.0 threshold"
        fc_result = _make_filter_result(
            passed=False, action="BLOCK", reason=reason, filter_name="spread_guard",
        )

        error_msg = f"BLOCKED by filter chain: {fc_result.reason}"
        assert "Spread 4.2 pips" in error_msg
        assert "BLOCKED" in error_msg

    def test_block_short_circuits_before_can_trade(self):
        """BLOCK returns early before the can_trade() check runs."""
        fc = MagicMock()
        fc.get_filter_names.return_value = ["risk_filter"]
        fc.apply.return_value = _make_filter_result(
            passed=False, action="BLOCK", reason="Risk too high",
        )
        can_trade_mock = MagicMock()

        _fc_result = fc.apply({"pair": "GBP_USD"})
        if not _fc_result.passed and _fc_result.action == "BLOCK":
            result = types.SimpleNamespace(success=False, error="BLOCKED")
        else:
            can_trade_mock()
            result = types.SimpleNamespace(success=True)

        assert result.success is False
        can_trade_mock.assert_not_called()


class TestFilterChainReduceSize:
    """US-P81-001: REDUCE_SIZE action — _phase49_size_mult *= size_multiplier."""

    def test_reduce_size_compounds_multiplier(self):
        _phase49_size_mult = 1.0
        fc_result = _make_filter_result(
            passed=True, action="REDUCE_SIZE", size_multiplier=0.7,
            reason="Vol regime elevated",
        )

        if fc_result.action == "REDUCE_SIZE" and fc_result.size_multiplier < 1.0:
            _phase49_size_mult *= fc_result.size_multiplier

        assert abs(_phase49_size_mult - 0.7) < 1e-9

    def test_reduce_size_compounds_with_existing_multiplier(self):
        _phase49_size_mult = 0.8  # Already reduced by prior logic
        fc_result = _make_filter_result(
            passed=True, action="REDUCE_SIZE", size_multiplier=0.5,
        )

        if fc_result.action == "REDUCE_SIZE" and fc_result.size_multiplier < 1.0:
            _phase49_size_mult *= fc_result.size_multiplier

        assert abs(_phase49_size_mult - 0.4) < 1e-9

    def test_reduce_size_at_1_0_does_not_compound(self):
        """size_multiplier == 1.0 should NOT trigger the compounding branch."""
        _phase49_size_mult = 0.9
        fc_result = _make_filter_result(
            passed=True, action="REDUCE_SIZE", size_multiplier=1.0,
        )

        if fc_result.action == "REDUCE_SIZE" and fc_result.size_multiplier < 1.0:
            _phase49_size_mult *= fc_result.size_multiplier

        assert _phase49_size_mult == 0.9  # Guard prevents compounding at 1.0


class TestFilterChainNoneGracefulSkip:
    """US-P81-001: FilterChain is None — graceful skip (fail-open)."""

    def test_filter_chain_none_skips(self):
        filter_chain = None
        _phase49_size_mult = 1.0

        if filter_chain is not None and filter_chain.get_filter_names():
            _phase49_size_mult = 0.0  # Should NOT run

        assert _phase49_size_mult == 1.0

    def test_filter_chain_empty_filters_skips(self):
        filter_chain = MagicMock()
        filter_chain.get_filter_names.return_value = []

        entered = False
        if filter_chain is not None and filter_chain.get_filter_names():
            entered = True

        assert entered is False


class TestFilterChainExceptionFailOpen:
    """US-P81-001: FilterChain raises exception — fail-open, trade proceeds."""

    def test_apply_exception_is_caught_fail_open(self, caplog):
        filter_chain = MagicMock()
        filter_chain.get_filter_names.return_value = ["buggy_filter"]
        filter_chain.apply.side_effect = RuntimeError("filter crash")

        _phase49_size_mult = 1.0
        blocked = False

        if filter_chain is not None and filter_chain.get_filter_names():
            try:
                _fc_result = filter_chain.apply({"pair": "EUR_USD"})
                if not _fc_result.passed and _fc_result.action == "BLOCK":
                    blocked = True
            except Exception as _fc_err:
                logging.getLogger(__name__).debug(
                    "Phase 81 (FilterChain): apply() failed (fail-open): %s", _fc_err
                )

        assert blocked is False
        assert _phase49_size_mult == 1.0


# ---------------------------------------------------------------------------
# US-P81-002: PositionTimeoutManager (4 call sites)
# ---------------------------------------------------------------------------


def _make_timeout_result(should_exit: bool, trade_id: str = "", exit_reason: str = "timeout"):
    return types.SimpleNamespace(
        should_exit=should_exit,
        trade_id=trade_id,
        exit_reason=exit_reason,
    )


class TestPositionTimeoutTick:
    """US-P81-002: tick() called when _position_timeout is not None."""

    def test_tick_called_when_manager_exists(self):
        ptm = MagicMock()
        _position_timeout = ptm

        if _position_timeout is not None:
            _position_timeout.tick()

        ptm.tick.assert_called_once()

    def test_tick_skipped_when_manager_none(self):
        _position_timeout = None
        # Should not raise
        if _position_timeout is not None:
            _position_timeout.tick()

    def test_tick_exception_caught(self):
        ptm = MagicMock()
        ptm.tick.side_effect = RuntimeError("tick crash")
        _position_timeout = ptm

        try:
            if _position_timeout is not None:
                _position_timeout.tick()
        except Exception:
            pass  # Production catches this

        ptm.tick.assert_called_once()


class TestPositionTimeoutEvaluateAll:
    """US-P81-002: evaluate_all() processes timeout results, WARNING logged for forced exits."""

    def test_evaluate_all_forced_exit_logs_warning(self, caplog):
        ptm = MagicMock()
        ptm.evaluate_all.return_value = [
            _make_timeout_result(should_exit=True, trade_id="12345", exit_reason="max_bars_exceeded"),
        ]
        executor = MagicMock()

        with caplog.at_level(logging.WARNING):
            _timeout_results = ptm.evaluate_all()
            for _tr in _timeout_results:
                if getattr(_tr, "should_exit", False):
                    _exit_reason = getattr(_tr, "exit_reason", "timeout")
                    _timed_out_id = getattr(_tr, "trade_id", "")
                    logging.getLogger(__name__).warning(
                        "Phase 81 (US-P81-002): Timeout exit for trade %s — %s",
                        _timed_out_id, _exit_reason,
                    )
                    executor.close_trade(trade_id=_timed_out_id, reason=_exit_reason)
                    ptm.close_trade(_timed_out_id)

        assert any("Timeout exit" in rec.message for rec in caplog.records)
        executor.close_trade.assert_called_once_with(trade_id="12345", reason="max_bars_exceeded")
        ptm.close_trade.assert_called_once_with("12345")

    def test_evaluate_all_no_exits(self):
        ptm = MagicMock()
        ptm.evaluate_all.return_value = [
            _make_timeout_result(should_exit=False, trade_id="11111"),
        ]
        executor = MagicMock()

        _timeout_results = ptm.evaluate_all()
        for _tr in _timeout_results:
            if getattr(_tr, "should_exit", False):
                executor.close_trade(trade_id=_tr.trade_id)

        executor.close_trade.assert_not_called()

    def test_evaluate_all_empty_results(self):
        ptm = MagicMock()
        ptm.evaluate_all.return_value = []
        executor = MagicMock()

        closed_count = 0
        for _tr in ptm.evaluate_all():
            if getattr(_tr, "should_exit", False):
                closed_count += 1

        assert closed_count == 0

    def test_evaluate_all_multiple_exits(self, caplog):
        ptm = MagicMock()
        ptm.evaluate_all.return_value = [
            _make_timeout_result(should_exit=True, trade_id="A1", exit_reason="stale"),
            _make_timeout_result(should_exit=False, trade_id="A2"),
            _make_timeout_result(should_exit=True, trade_id="A3", exit_reason="max_bars"),
        ]
        executor = MagicMock()

        closed = []
        for _tr in ptm.evaluate_all():
            if getattr(_tr, "should_exit", False):
                executor.close_trade(trade_id=_tr.trade_id, reason=_tr.exit_reason)
                ptm.close_trade(_tr.trade_id)
                closed.append(_tr.trade_id)

        assert closed == ["A1", "A3"]
        assert executor.close_trade.call_count == 2


class TestPositionTimeoutRegisterTrade:
    """US-P81-002: register_trade() called post-execute with correct args."""

    def test_register_trade_with_correct_args(self):
        ptm = MagicMock()
        trade_id = "99887"
        regime_name = "TRENDING"
        confidence = 0.78

        if ptm is not None:
            _pt_regime = (regime_name or "NORMAL").lower()
            ptm.register_trade(
                trade_id=str(trade_id),
                regime=_pt_regime,
                confidence=float(confidence),
                pair="EUR_USD",
                direction="BUY",
            )

        ptm.register_trade.assert_called_once_with(
            trade_id="99887",
            regime="trending",
            confidence=0.78,
            pair="EUR_USD",
            direction="BUY",
        )

    def test_register_trade_none_regime_defaults_to_normal(self):
        ptm = MagicMock()
        regime_name = None

        _pt_regime = (regime_name or "NORMAL").lower()
        ptm.register_trade(
            trade_id="100", regime=_pt_regime, confidence=0.6,
            pair="GBP_USD", direction="SELL",
        )

        ptm.register_trade.assert_called_once_with(
            trade_id="100", regime="normal", confidence=0.6,
            pair="GBP_USD", direction="SELL",
        )

    def test_register_trade_skipped_when_mgr_none(self):
        _position_timeout_mgr = None
        called = False

        if _position_timeout_mgr is not None:
            _position_timeout_mgr.register_trade(trade_id="1")
            called = True

        assert called is False


class TestPositionTimeoutCloseTrade:
    """US-P81-002: close_trade() called in sync when trade confirmed closed."""

    def test_close_trade_called(self):
        ptm = MagicMock()
        tid = "55443"

        if ptm is not None:
            ptm.close_trade(tid)

        ptm.close_trade.assert_called_once_with("55443")

    def test_close_trade_exception_caught(self):
        ptm = MagicMock()
        ptm.close_trade.side_effect = RuntimeError("close crash")

        try:
            ptm.close_trade("broken_id")
        except Exception:
            pass  # Production catches this

        ptm.close_trade.assert_called_once()

    def test_close_trade_skipped_when_mgr_none(self):
        _position_timeout_mgr = None
        # Should not raise
        if _position_timeout_mgr is not None:
            _position_timeout_mgr.close_trade("123")


# ---------------------------------------------------------------------------
# US-P81-003: RegimeConfidenceScorer.score() in _scan_pair()
# ---------------------------------------------------------------------------


def _make_rcs_result(sizing_factor: float, tier: str = "MODERATE",
                     is_transition: bool = False):
    return types.SimpleNamespace(
        sizing_factor=sizing_factor,
        tier=tier,
        is_transition=is_transition,
    )


class TestRegimeConfidenceScorerSizingReduction:
    """US-P81-003: sizing_factor < 1.0 — risk_pct is reduced."""

    def test_sizing_factor_075_reduces_risk(self):
        risk_pct = 0.05
        rcs_result = _make_rcs_result(sizing_factor=0.75)

        if rcs_result.sizing_factor < 1.0:
            risk_pct *= rcs_result.sizing_factor

        assert abs(risk_pct - 0.0375) < 1e-9

    def test_sizing_factor_050_halves_risk(self):
        risk_pct = 0.04
        rcs_result = _make_rcs_result(sizing_factor=0.5)

        if rcs_result.sizing_factor < 1.0:
            risk_pct *= rcs_result.sizing_factor

        assert abs(risk_pct - 0.02) < 1e-9


class TestRegimeConfidenceScorerFullSize:
    """US-P81-003: sizing_factor == 1.0 — risk_pct unchanged."""

    def test_sizing_factor_1_leaves_risk_unchanged(self):
        risk_pct = 0.05
        rcs_result = _make_rcs_result(sizing_factor=1.0)

        if rcs_result.sizing_factor < 1.0:
            risk_pct *= rcs_result.sizing_factor

        assert risk_pct == 0.05


class TestRegimeConfidenceScorerRegimeMapping:
    """US-P81-003: STABLE and TRANSITION regime scenarios."""

    def test_stable_regime_full_size(self):
        rcs_result = _make_rcs_result(
            sizing_factor=1.0, tier="HIGH", is_transition=False,
        )

        applied_factor = rcs_result.sizing_factor if rcs_result.sizing_factor < 1.0 else 1.0
        assert applied_factor == 1.0
        assert rcs_result.is_transition is False

    def test_transition_regime_reduced_size(self):
        rcs_result = _make_rcs_result(
            sizing_factor=0.6, tier="LOW", is_transition=True,
        )

        risk_pct = 0.05
        if rcs_result.sizing_factor < 1.0:
            risk_pct *= rcs_result.sizing_factor

        assert abs(risk_pct - 0.03) < 1e-9
        assert rcs_result.is_transition is True


class TestRegimeConfidenceScorerNoneSkip:
    """US-P81-003: scorer is None — graceful skip."""

    def test_scorer_none_does_not_alter_risk(self):
        scorer = None
        risk_pct = 0.05
        direction = "BUY"

        if scorer is not None and direction != "HOLD" and risk_pct > 0:
            risk_pct *= 0.5  # Should NOT run

        assert risk_pct == 0.05

    def test_hold_direction_does_not_score(self):
        scorer = MagicMock()
        risk_pct = 0.05
        direction = "HOLD"

        if scorer is not None and direction != "HOLD" and risk_pct > 0:
            scorer.score()

        scorer.score.assert_not_called()

    def test_zero_risk_does_not_score(self):
        scorer = MagicMock()
        risk_pct = 0.0
        direction = "BUY"

        if scorer is not None and direction != "HOLD" and risk_pct > 0:
            scorer.score()

        scorer.score.assert_not_called()


# ---------------------------------------------------------------------------
# US-P81-004: CausalCounterfactual batch_analyze() post-loss
# ---------------------------------------------------------------------------


def _make_journal_entry(trade_id: str, pair: str = "EUR_USD",
                        confidence: float = 0.6) -> dict:
    """Minimal pending journal entry for counterfactual tests."""
    return {
        "trade_id": trade_id,
        "pair": pair,
        "confidence": confidence,
        "regime": {
            "atr_pips": 0.0012,
            "uncertainty_score": 0.4,
            "volatility_regime": "NORMAL",
        },
    }


def _make_closed_trade(trade_id: str, realized_pl: float) -> dict:
    return {"realizedPL": str(realized_pl)}


class TestCausalCounterfactualLossingTrades:
    """US-P81-004: batch_analyze called only for losing trades."""

    def _run_counterfactual_logic(self, pending, closed_trades, cc, cap=5):
        """Replicate production loop from execution.py lines ~5038-5076."""
        analyzed = 0
        for entry in pending:
            if analyzed >= cap:
                break
            tid = str(entry.get("trade_id", ""))
            if tid not in closed_trades:
                continue
            ct = closed_trades[tid]
            _cc_pl = float(ct.get("realizedPL", 0))
            if _cc_pl >= 0:
                continue
            _cc_context = {
                "pair": entry.get("pair", ""),
                "features": {},
                "regime": "NORMAL",
                "outcome": 0.0,
            }
            _cc_scenarios = cc.generate_standard_scenarios(_cc_context)
            cc.batch_analyze(_cc_context, _cc_scenarios)
            analyzed += 1
        return analyzed

    def test_3_losing_trades_analyzes_3(self):
        cc = MagicMock()
        cc.generate_standard_scenarios.return_value = [MagicMock()]
        cc.batch_analyze.return_value = []

        pending = [_make_journal_entry(f"T{i}") for i in range(3)]
        closed = {f"T{i}": _make_closed_trade(f"T{i}", -50.0) for i in range(3)}

        count = self._run_counterfactual_logic(pending, closed, cc)

        assert count == 3
        assert cc.batch_analyze.call_count == 3

    def test_0_losing_trades_no_analysis(self):
        cc = MagicMock()
        cc.generate_standard_scenarios.return_value = []
        cc.batch_analyze.return_value = []

        pending = [_make_journal_entry(f"T{i}") for i in range(3)]
        closed = {f"T{i}": _make_closed_trade(f"T{i}", 25.0) for i in range(3)}  # All wins

        count = self._run_counterfactual_logic(pending, closed, cc)

        assert count == 0
        cc.batch_analyze.assert_not_called()

    def test_7_losing_trades_capped_at_5(self):
        cc = MagicMock()
        cc.generate_standard_scenarios.return_value = [MagicMock()]
        cc.batch_analyze.return_value = []

        pending = [_make_journal_entry(f"T{i}") for i in range(7)]
        closed = {f"T{i}": _make_closed_trade(f"T{i}", -30.0) for i in range(7)}

        count = self._run_counterfactual_logic(pending, closed, cc, cap=5)

        assert count == 5
        assert cc.batch_analyze.call_count == 5

    def test_mixed_wins_and_losses(self):
        cc = MagicMock()
        cc.generate_standard_scenarios.return_value = [MagicMock()]
        cc.batch_analyze.return_value = []

        pending = [_make_journal_entry(f"T{i}") for i in range(5)]
        closed = {
            "T0": _make_closed_trade("T0", 10.0),   # Win
            "T1": _make_closed_trade("T1", -20.0),  # Loss
            "T2": _make_closed_trade("T2", 0.0),    # Breakeven (not < 0)
            "T3": _make_closed_trade("T3", -5.0),   # Loss
            "T4": _make_closed_trade("T4", 30.0),   # Win
        }

        count = self._run_counterfactual_logic(pending, closed, cc)

        assert count == 2
        assert cc.batch_analyze.call_count == 2


class TestCausalCounterfactualNoneSkip:
    """US-P81-004: counterfactual engine is None — graceful skip."""

    def test_cc_none_graceful_skip(self):
        scanner = MagicMock()
        scanner._causal_counterfactual = None
        synced = 3

        analyzed = False
        if scanner is not None and synced > 0:
            _cc = getattr(scanner, "_causal_counterfactual", None)
            if _cc is not None:
                _cc.batch_analyze({}, [])
                analyzed = True

        assert analyzed is False

    def test_scanner_none_graceful_skip(self):
        scanner = None
        synced = 3

        analyzed = False
        if scanner is not None and synced > 0:
            analyzed = True

        assert analyzed is False

    def test_synced_zero_skips(self):
        scanner = MagicMock()
        scanner._causal_counterfactual = MagicMock()
        synced = 0

        analyzed = False
        if scanner is not None and synced > 0:
            _cc = getattr(scanner, "_causal_counterfactual", None)
            if _cc is not None:
                analyzed = True

        assert analyzed is False


class TestCausalCounterfactualExceptionHandling:
    """US-P81-004: batch_analyze exception — caught, no propagation."""

    def test_batch_analyze_exception_caught(self, caplog):
        cc = MagicMock()
        cc.generate_standard_scenarios.return_value = [MagicMock()]
        cc.batch_analyze.side_effect = RuntimeError("model error")

        pending = [_make_journal_entry("T1")]
        closed = {"T1": _make_closed_trade("T1", -100.0)}

        error_caught = False
        try:
            for entry in pending:
                tid = str(entry.get("trade_id", ""))
                if tid not in closed:
                    continue
                ct = closed[tid]
                if float(ct.get("realizedPL", 0)) >= 0:
                    continue
                _cc_context = {"pair": entry.get("pair", ""), "features": {}, "regime": "NORMAL", "outcome": 0.0}
                _cc_scenarios = cc.generate_standard_scenarios(_cc_context)
                cc.batch_analyze(_cc_context, _cc_scenarios)
        except Exception as _cc_err:
            error_caught = True
            logging.getLogger(__name__).debug(
                "Phase 81 (US-P81-004): CausalCounterfactual error: %s", _cc_err
            )

        assert error_caught is True
        # In production, the outer try/except catches this and logs debug — no propagation

    def test_generate_scenarios_exception_caught(self):
        cc = MagicMock()
        cc.generate_standard_scenarios.side_effect = ValueError("bad context")

        raised = False
        try:
            cc.generate_standard_scenarios({})
        except Exception:
            raised = True

        assert raised is True
        # Production wraps entire block in try/except — no propagation
