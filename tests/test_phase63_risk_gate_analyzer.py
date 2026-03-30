"""Phase 63 US-379: RiskGateAnalyzer unit tests.

Minimum 8 tests required per PRD acceptance criteria.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scanner.risk_reject_recorder import RiskRejectRecorder
from src.scanner.risk_gate_analyzer import (
    analyze,
    get_risk_signals,
    RiskGateAnalyzer,
    CLASS_NEAR_THRESHOLD,
    CLASS_MODERATE,
    CLASS_FAR_ABOVE,
    CLASS_INSUFFICIENT,
)

MAX_DD = 0.025  # default max_drawdown_pct


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_recorder(tmp_path: Path) -> RiskRejectRecorder:
    return RiskRejectRecorder(state_path=tmp_path / "state.jsonl")


def add_fails(rec, drawdown_pct: float = 0.030, count: int = 10, max_dd=MAX_DD):
    for _ in range(count):
        rec.record("EUR_USD", rf_drawdown_pct=drawdown_pct, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=max_dd)


def add_passes(rec, count: int = 5, max_dd=MAX_DD):
    for _ in range(count):
        rec.record("EUR_USD", rf_drawdown_pct=0.010, rf_streak_prob=0.40,
                   risk_passed=True, max_drawdown_pct=max_dd)


# ------------------------------------------------------------------ #
# Test 1: NEAR_THRESHOLD classification                                #
# ------------------------------------------------------------------ #

def test_near_threshold_classification(tmp_path):
    """10 fail records at rf_drawdown=0.028 vs max=0.025 → gap=0.003 → NEAR_THRESHOLD."""
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.028)  # gap = 0.028 - 0.025 = 0.003 <= 0.005
    result = analyze(rec, MAX_DD)
    assert result["classification"] == CLASS_NEAR_THRESHOLD
    assert abs(result["gap_at_p50"] - 0.003) < 1e-7


# ------------------------------------------------------------------ #
# Test 2: MODERATE classification                                      #
# ------------------------------------------------------------------ #

def test_moderate_classification(tmp_path):
    """10 fail records at rf_drawdown=0.040 vs max=0.025 → gap=0.015 → MODERATE boundary."""
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.036)  # gap = 0.011 → MODERATE (0.005 < 0.011 <= 0.015)
    result = analyze(rec, MAX_DD)
    assert result["classification"] == CLASS_MODERATE
    assert 0.005 < result["gap_at_p50"] <= 0.015


# ------------------------------------------------------------------ #
# Test 3: FAR_ABOVE classification                                     #
# ------------------------------------------------------------------ #

def test_far_above_classification(tmp_path):
    """10 fail records at rf_drawdown=0.060 vs max=0.025 → gap=0.035 → FAR_ABOVE."""
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.060)  # gap = 0.035 > 0.015
    result = analyze(rec, MAX_DD)
    assert result["classification"] == CLASS_FAR_ABOVE
    assert abs(result["gap_at_p50"] - 0.035) < 1e-7


# ------------------------------------------------------------------ #
# Test 4: INSUFFICIENT_DATA when < 3 fail records                      #
# ------------------------------------------------------------------ #

def test_insufficient_data_empty(tmp_path):
    rec = make_recorder(tmp_path)
    result = analyze(rec, MAX_DD)
    assert result["classification"] == CLASS_INSUFFICIENT


def test_insufficient_data_two_fails(tmp_path):
    rec = make_recorder(tmp_path)
    for _ in range(2):
        rec.record("EUR_USD", rf_drawdown_pct=0.030, rf_streak_prob=0.5,
                   risk_passed=False, max_drawdown_pct=MAX_DD)
    result = analyze(rec, MAX_DD)
    assert result["classification"] == CLASS_INSUFFICIENT


# ------------------------------------------------------------------ #
# Test 5: near_miss_count and near_miss_rate_pct                       #
# ------------------------------------------------------------------ #

def test_near_miss_rate_50_pct(tmp_path):
    """5 near-miss fails (gap<=0.005) + 5 far fails → near_miss_rate_pct=50.0."""
    rec = make_recorder(tmp_path)
    # Near misses: drawdown = 0.028, gap = 0.003 <= 0.005
    for _ in range(5):
        rec.record("EUR_USD", rf_drawdown_pct=0.028, rf_streak_prob=0.5,
                   risk_passed=False, max_drawdown_pct=MAX_DD)
    # Far fails: drawdown = 0.060, gap = 0.035 > 0.005
    for _ in range(5):
        rec.record("EUR_USD", rf_drawdown_pct=0.060, rf_streak_prob=0.5,
                   risk_passed=False, max_drawdown_pct=MAX_DD)
    result = analyze(rec, MAX_DD)
    assert result["near_miss_count"] == 5
    assert abs(result["near_miss_rate_pct"] - 50.0) < 0.01


def test_near_miss_rate_all_near(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.028)  # all near misses
    result = analyze(rec, MAX_DD)
    assert result["near_miss_count"] == 10
    assert abs(result["near_miss_rate_pct"] - 100.0) < 0.01


def test_near_miss_rate_none_near(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.060)  # far above — no near misses
    result = analyze(rec, MAX_DD)
    assert result["near_miss_count"] == 0
    assert result["near_miss_rate_pct"] == 0.0


# ------------------------------------------------------------------ #
# Test 6: Recommendation strings correct                               #
# ------------------------------------------------------------------ #

def test_recommendation_near_threshold(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.028)
    result = analyze(rec, MAX_DD)
    assert "drawdown threshold increase" in result["recommendation"]
    assert "near the gate" in result["recommendation"]


def test_recommendation_moderate(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.036)
    result = analyze(rec, MAX_DD)
    assert "systematic overestimation" in result["recommendation"]


def test_recommendation_far_above(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.060)
    result = analyze(rec, MAX_DD)
    assert "retrain RF model" in result["recommendation"]


def test_recommendation_insufficient(tmp_path):
    rec = make_recorder(tmp_path)
    result = analyze(rec, MAX_DD)
    assert result["recommendation"] != ""  # has a safe default recommendation


# ------------------------------------------------------------------ #
# Test 7: All required output keys present                             #
# ------------------------------------------------------------------ #

def test_all_required_keys_present(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.030)
    result = analyze(rec, MAX_DD)
    required_keys = [
        "count", "fail_count", "pass_rate", "gap_at_p50", "gap_at_p75",
        "near_miss_count", "near_miss_rate_pct", "drawdown_kill_pct",
        "streak_kill_pct", "classification", "recommendation",
    ]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


# ------------------------------------------------------------------ #
# Test 8: gap_at_p50 calculation correct                               #
# ------------------------------------------------------------------ #

def test_gap_at_p50_calculation(tmp_path):
    """gap_at_p50 = p50_of_fail_drawdown - max_drawdown_pct."""
    rec = make_recorder(tmp_path)
    # All fails at exactly 0.035
    add_fails(rec, drawdown_pct=0.035, count=10)
    result = analyze(rec, MAX_DD)
    # gap = 0.035 - 0.025 = 0.010
    assert abs(result["gap_at_p50"] - 0.010) < 1e-7


def test_gap_at_p75_calculation(tmp_path):
    rec = make_recorder(tmp_path)
    # Uniform fails — p75 ≈ 0.035 → gap ≈ 0.010
    add_fails(rec, drawdown_pct=0.035, count=10)
    result = analyze(rec, MAX_DD)
    assert abs(result["gap_at_p75"] - 0.010) < 1e-6


# ------------------------------------------------------------------ #
# Test 9: drawdown_kill_pct and streak_kill_pct                        #
# ------------------------------------------------------------------ #

def test_kill_pcts_in_result(tmp_path):
    rec = make_recorder(tmp_path)
    # 8 drawdown kills
    for _ in range(8):
        rec.record("EUR_USD", rf_drawdown_pct=0.030, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=MAX_DD)
    # 2 streak kills (drawdown under threshold)
    for _ in range(2):
        rec.record("EUR_USD", rf_drawdown_pct=0.010, rf_streak_prob=0.97,
                   risk_passed=False, max_drawdown_pct=MAX_DD, max_streak_prob=0.95)
    result = analyze(rec, MAX_DD)
    assert abs(result["drawdown_kill_pct"] - 80.0) < 0.01
    assert abs(result["streak_kill_pct"] - 20.0) < 0.01


# ------------------------------------------------------------------ #
# Test 10: pass_rate in analyze() result                               #
# ------------------------------------------------------------------ #

def test_pass_rate_in_result(tmp_path):
    rec = make_recorder(tmp_path)
    add_passes(rec, count=7)
    add_fails(rec, count=3)
    result = analyze(rec, MAX_DD)
    assert abs(result["pass_rate"] - 0.70) < 1e-4


# ------------------------------------------------------------------ #
# Test 11: get_risk_signals() returns required keys                    #
# ------------------------------------------------------------------ #

def test_get_risk_signals_keys(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.028)
    signals = get_risk_signals(rec, MAX_DD)
    required = [
        "risk_gap_at_p50",
        "risk_near_miss_rate_pct",
        "risk_classification",
        "risk_pass_rate",
        "risk_drawdown_kill_pct",
    ]
    for k in required:
        assert k in signals, f"Missing key: {k}"


def test_get_risk_signals_near_threshold_value(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.028)
    signals = get_risk_signals(rec, MAX_DD)
    assert signals["risk_classification"] == CLASS_NEAR_THRESHOLD
    assert abs(signals["risk_gap_at_p50"] - 0.003) < 1e-7


# ------------------------------------------------------------------ #
# Test 12: RiskGateAnalyzer class interface                            #
# ------------------------------------------------------------------ #

def test_risk_gate_analyzer_class_interface(tmp_path):
    rec = make_recorder(tmp_path)
    add_fails(rec, drawdown_pct=0.060)
    analyzer = RiskGateAnalyzer()
    result = analyzer.analyze_current(rec, MAX_DD)
    assert result["classification"] == CLASS_FAR_ABOVE
    signals = analyzer.get_risk_signals(rec, MAX_DD)
    assert "risk_classification" in signals
    assert signals["risk_classification"] == CLASS_FAR_ABOVE
