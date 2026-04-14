"""Phase 90 US-406: Integration and regression test suite for all Phase 90 changes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import threading

import pytest


# ── US-401: disagreement_gap_analyzer constants regression ────────────────


def test_disagreement_gap_analyzer_near_miss_gap_regression():
    """Regression: NEAR_MISS_GAP must be 0.35 after Phase 90 recalibration."""
    from src.scanner.disagreement_gap_analyzer import NEAR_MISS_GAP
    assert NEAR_MISS_GAP == pytest.approx(0.35)


def test_disagreement_gap_analyzer_moderate_gap_regression():
    """Regression: MODERATE_GAP must be 0.50 after Phase 90 recalibration."""
    from src.scanner.disagreement_gap_analyzer import MODERATE_GAP
    assert MODERATE_GAP == pytest.approx(0.50)


def test_disagreement_analyze_floor050_p50_060_near_threshold():
    """Regression: analyze() with floor=0.50, fail p50=0.60 → NEAR_THRESHOLD."""
    from src.scanner.disagreement_gap_analyzer import analyze, CLASS_NEAR_THRESHOLD, MIN_FAIL_COUNT

    recorder = MagicMock()
    recorder.get_fail_distribution.return_value = {"count": MIN_FAIL_COUNT, "p50": 0.60, "p75": 0.70}
    recorder.get_pass_rate.return_value = 0.5
    recorder.get_distribution.return_value = {"count": 10}
    recorder._lock = threading.Lock()
    recorder._records = []

    result = analyze(recorder, 0.50)
    assert result["classification"] == CLASS_NEAR_THRESHOLD


def test_disagreement_analyze_floor050_p50_080_near_threshold():
    """Regression: analyze() with floor=0.50, fail p50=0.80 → NEAR_THRESHOLD (was MODERATE)."""
    from src.scanner.disagreement_gap_analyzer import analyze, CLASS_NEAR_THRESHOLD, MIN_FAIL_COUNT

    recorder = MagicMock()
    recorder.get_fail_distribution.return_value = {"count": MIN_FAIL_COUNT, "p50": 0.80, "p75": 0.90}
    recorder.get_pass_rate.return_value = 0.5
    recorder.get_distribution.return_value = {"count": 10}
    recorder._lock = threading.Lock()
    recorder._records = []

    result = analyze(recorder, 0.50)
    assert result["classification"] == CLASS_NEAR_THRESHOLD, (
        f"4/5 disagreement (gap=0.30) should be NEAR_THRESHOLD with NEAR_MISS_GAP=0.35, got {result['classification']}"
    )


# ── US-402: ConfidencePenaltyCeiling DEFAULT_CEILING_FLOOR regression ─────


def test_ceiling_default_floor_is_040():
    """Regression: ConfidencePenaltyCeiling DEFAULT_CEILING_FLOOR == 0.40."""
    from src.scanner.confidence_penalty_ceiling import DEFAULT_CEILING_FLOOR
    assert DEFAULT_CEILING_FLOOR == pytest.approx(0.40)


def test_ceiling_check_subtraction_active_at_042():
    """Regression: ConfidencePenaltyCeiling(0.40).check_subtraction(0.42, 0.05) → ceiling_active=True."""
    from src.scanner.confidence_penalty_ceiling import ConfidencePenaltyCeiling
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40)
    result = ceiling.check_subtraction(current_confidence=0.42, proposed_sub=0.05)
    assert result.ceiling_active is True


# ── US-403: PairAnalysis kill_reason regression ────────────────────────────


def test_pair_analysis_accepts_kill_reason_none():
    """Regression: PairAnalysis accepts kill_reason=None without error."""
    from src.scanner.results import PairAnalysis
    a = PairAnalysis(pair="EUR_USD", kill_reason=None)
    assert a.kill_reason is None


def test_pair_analysis_accepts_kill_reason_confidence_gate():
    """Regression: PairAnalysis accepts kill_reason='confidence_gate' without error."""
    from src.scanner.results import PairAnalysis
    a = PairAnalysis(pair="EUR_USD", kill_reason="confidence_gate")
    assert a.kill_reason == "confidence_gate"


# ── US-404: _log_scan_cycle gate_kill_breakdown regression ────────────────


def test_log_scan_cycle_produces_gate_kill_breakdown():
    """Regression: _log_scan_cycle() produces record with 'gate_kill_breakdown' key."""
    from src.scanner.results import PairAnalysis, ScanResult
    from src.scanner.automation.continuous import ContinuousScanner

    cs = ContinuousScanner.__new__(ContinuousScanner)
    cs._scan_count = 1
    for attr in (
        "_scan_diagnostics", "_adaptive_floor", "_penalty_auditor", "_gate_pass_predictor",
        "_signal_funnel", "_scan_health", "_pair_rotation_enabled",
    ):
        setattr(cs, attr, None)

    analyses = [
        PairAnalysis(pair="EUR_USD", direction="LONG", kill_reason="confidence_gate"),
        PairAnalysis(pair="GBP_USD", direction="HOLD", kill_reason=None),
    ]
    result = ScanResult(analyses=analyses)

    with patch("src.scanner.automation.safe_json.safe_jsonl_append") as mock_append:
        cs._log_scan_cycle(result, auto_execute=False)

    record = mock_append.call_args[0][1]
    assert "gate_kill_breakdown" in record
    assert isinstance(record["gate_kill_breakdown"], dict)


# ── US-405: adaptive_floor_audit.run() regression ─────────────────────────


def test_adaptive_floor_audit_run_returns_all_keys(tmp_path):
    """Regression: adaptive_floor_audit.run() returns dict with all three component keys."""
    from src.scanner.tools.adaptive_floor_audit import run

    result = run(
        confidence_floor_path=tmp_path / "cf.json",
        momentum_floor_path=tmp_path / "mf.json",
        drawdown_ceiling_path=tmp_path / "dc.json",
    )
    assert "confidence_floor" in result
    assert "momentum_floor" in result
    assert "drawdown_ceiling" in result


# ── Module importability ───────────────────────────────────────────────────


def test_all_phase90_modules_importable():
    """All Phase 90 modules importable without error."""
    import src.scanner.disagreement_gap_analyzer
    import src.scanner.confidence_penalty_ceiling
    import src.scanner.results
    import src.scanner.automation.continuous
    import src.scanner.tools.adaptive_floor_audit


# ── Total assertion count guard ────────────────────────────────────────────


def test_suite_has_minimum_assertions():
    """Meta-test: this file contains >= 8 test functions (acceptance criterion)."""
    import inspect
    import sys
    module = sys.modules[__name__]
    test_fns = [name for name in dir(module) if name.startswith("test_")]
    assert len(test_fns) >= 8, f"Expected >= 8 tests, found {len(test_fns)}"
