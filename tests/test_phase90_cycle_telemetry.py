"""Phase 90 US-404: Tests for extended _log_scan_cycle gate-kill telemetry."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.scanner.results import PairAnalysis, ScanResult


def _make_analysis(pair: str, tradeable: bool = False, direction: str = "HOLD",
                   kill_reason: str | None = None, error: str | None = None) -> PairAnalysis:
    """Build a minimal PairAnalysis for testing."""
    a = PairAnalysis(
        pair=pair,
        direction=direction,
        kill_reason=kill_reason,
        error=error,
        gates_passed=tradeable,
        agent_passed=tradeable,
        volatility_regime="NORMAL" if tradeable else "UNKNOWN",
        model_disagreement=0.1,
    )
    return a


def _make_continuous_scanner(analyses: list[PairAnalysis], auto_execute: bool = True):
    """Create a minimal ContinuousScanner mock with _log_scan_cycle accessible."""
    from src.scanner.automation.continuous import ContinuousScanner

    cs = ContinuousScanner.__new__(ContinuousScanner)
    cs._scan_count = 1
    # All optional sub-systems that _log_scan_cycle checks must exist (as None to skip their blocks)
    for attr in (
        "_scan_diagnostics", "_adaptive_floor", "_penalty_auditor", "_gate_pass_predictor",
        "_signal_funnel", "_scan_health", "_pair_rotation_enabled",
    ):
        setattr(cs, attr, None)

    result = ScanResult(analyses=analyses)
    return cs, result


def test_log_scan_cycle_record_has_gate_kill_breakdown(tmp_path):
    """_log_scan_cycle() record must contain 'gate_kill_breakdown' key."""
    analyses = [
        _make_analysis("EUR_USD", tradeable=False, direction="LONG", kill_reason="confidence_gate"),
        _make_analysis("GBP_USD", tradeable=False, direction="SHORT", kill_reason="momentum_gate"),
        _make_analysis("USD_JPY", tradeable=True, direction="LONG"),
    ]
    cs, result = _make_continuous_scanner(analyses, auto_execute=True)

    with patch("src.scanner.automation.safe_json.safe_jsonl_append") as mock_append:
        cs._log_scan_cycle(result, auto_execute=True)

    call_args = mock_append.call_args
    assert call_args is not None, "_log_scan_cycle should have called safe_jsonl_append"
    record = call_args[0][1]  # positional arg[1]
    assert "gate_kill_breakdown" in record
    assert "trades_attempted" in record
    assert "trades_executed" in record


def test_gate_kill_breakdown_counts_correctly(tmp_path):
    """gate_kill_breakdown correctly tallies kill reasons."""
    analyses = [
        _make_analysis("EUR_USD", tradeable=False, direction="LONG", kill_reason="confidence_gate"),
        _make_analysis("GBP_USD", tradeable=False, direction="SHORT", kill_reason="confidence_gate"),
        _make_analysis("USD_JPY", tradeable=False, direction="LONG", kill_reason="momentum_gate"),
        _make_analysis("AUD_USD", tradeable=True, direction="LONG"),
    ]
    cs, result = _make_continuous_scanner(analyses, auto_execute=True)

    with patch("src.scanner.automation.safe_json.safe_jsonl_append") as mock_append:
        cs._log_scan_cycle(result, auto_execute=True)

    record = mock_append.call_args[0][1]
    breakdown = record["gate_kill_breakdown"]
    assert breakdown["confidence_gate"] == 2
    assert breakdown["momentum_gate"] == 1
    assert breakdown["disagreement_gate"] == 0


def test_trades_attempted_and_executed_counts(tmp_path):
    """trades_attempted and trades_executed are correctly computed."""
    analyses = [
        _make_analysis("EUR_USD", tradeable=False, direction="LONG", kill_reason="confidence_gate"),
        _make_analysis("GBP_USD", tradeable=False, direction="SHORT", kill_reason="momentum_gate"),
        _make_analysis("USD_JPY", tradeable=True, direction="LONG"),
    ]
    cs, result = _make_continuous_scanner(analyses, auto_execute=True)

    with patch("src.scanner.automation.safe_json.safe_jsonl_append") as mock_append:
        cs._log_scan_cycle(result, auto_execute=True)

    record = mock_append.call_args[0][1]
    # trades_attempted: any non-HOLD direction
    assert record["trades_attempted"] >= 2
    # trades_executed: tradeable count when auto_execute=True
    assert record["trades_executed"] == 1


def test_warning_emitted_when_attempted_but_zero_executed(caplog):
    """WARNING logged when trades_attempted > 0 and trades_executed == 0."""
    analyses = [
        _make_analysis("EUR_USD", tradeable=False, direction="LONG", kill_reason="confidence_gate"),
        _make_analysis("GBP_USD", tradeable=False, direction="SHORT", kill_reason="momentum_gate"),
    ]
    cs, result = _make_continuous_scanner(analyses, auto_execute=False)

    with patch("src.scanner.automation.safe_json.safe_jsonl_append"):
        with caplog.at_level(logging.WARNING, logger="src.scanner.automation.continuous"):
            cs._log_scan_cycle(result, auto_execute=False)

    assert any("0 executed" in r.message for r in caplog.records), (
        "Expected WARNING about 0 executed trades"
    )


def test_record_is_valid_json_compatible():
    """Record dict produced by _log_scan_cycle is JSON-serializable."""
    analyses = [
        _make_analysis("EUR_USD", tradeable=False, direction="LONG", kill_reason="drawdown_gate"),
        _make_analysis("GBP_USD", tradeable=True, direction="SHORT"),
    ]
    cs, result = _make_continuous_scanner(analyses, auto_execute=True)

    with patch("src.scanner.automation.safe_json.safe_jsonl_append") as mock_append:
        cs._log_scan_cycle(result, auto_execute=True)

    record = mock_append.call_args[0][1]
    serialized = json.dumps(record)
    parsed = json.loads(serialized)
    assert "gate_kill_breakdown" in parsed
    assert "trades_attempted" in parsed
    assert "trades_executed" in parsed


def test_missing_kill_reason_debug_warning(caplog):
    """Debug log emitted when kill_reason attribute is missing on rejected pair."""
    a = _make_analysis("EUR_USD", tradeable=False, direction="LONG")
    # Simulate a PairAnalysis that lacks kill_reason entirely
    del a.kill_reason
    cs, result = _make_continuous_scanner([a], auto_execute=False)

    with patch("src.scanner.automation.safe_json.safe_jsonl_append"):
        with caplog.at_level(logging.DEBUG, logger="src.scanner.automation.continuous"):
            cs._log_scan_cycle(result, auto_execute=False)

    assert any("kill_reason" in r.message and "defaulting" in r.message for r in caplog.records), (
        "Expected DEBUG warning about missing kill_reason"
    )


def test_existing_fields_not_removed():
    """Existing fields (tradeable_count, model_type, etc.) still present."""
    analyses = [
        _make_analysis("EUR_USD", tradeable=True, direction="LONG"),
    ]
    cs, result = _make_continuous_scanner(analyses, auto_execute=True)
    result.model_type = "xgboost"

    with patch("src.scanner.automation.safe_json.safe_jsonl_append") as mock_append:
        cs._log_scan_cycle(result, auto_execute=True)

    record = mock_append.call_args[0][1]
    assert "tradeable_count" in record
    assert "tradeable_pairs" in record
    assert "model_type" in record
    assert "scan_number" in record
    assert "top_score" in record
