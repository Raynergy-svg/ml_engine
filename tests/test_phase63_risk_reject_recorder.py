"""Phase 63 US-378: RiskRejectRecorder unit tests.

Minimum 8 tests required per PRD acceptance criteria.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scanner.risk_reject_recorder import RiskRejectRecorder


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_recorder(tmp_path: Path, max_size: int = 500) -> RiskRejectRecorder:
    return RiskRejectRecorder(
        max_size=max_size,
        state_path=tmp_path / "risk_reject_state.jsonl",
    )


def record_pass(rec, pair="EUR_USD", drawdown=0.010, streak=0.50, max_dd=0.025):
    rec.record(pair=pair, rf_drawdown_pct=drawdown, rf_streak_prob=streak,
               risk_passed=True, max_drawdown_pct=max_dd)


def record_fail(rec, pair="EUR_USD", drawdown=0.030, streak=0.50, max_dd=0.025):
    rec.record(pair=pair, rf_drawdown_pct=drawdown, rf_streak_prob=streak,
               risk_passed=False, max_drawdown_pct=max_dd)


# ------------------------------------------------------------------ #
# Test 1: record stores snapshot correctly                             #
# ------------------------------------------------------------------ #

def test_record_stores_snapshot(tmp_path):
    rec = make_recorder(tmp_path)
    rec.record(
        pair="EUR_USD",
        rf_drawdown_pct=0.030,
        rf_streak_prob=0.40,
        risk_passed=False,
        max_drawdown_pct=0.025,
        scan_id="scan_001",
    )
    dist = rec.get_distribution()
    assert dist["count"] == 1
    assert abs(dist["p50"] - 0.030) < 1e-7


# ------------------------------------------------------------------ #
# Test 2: drawdown_kill and streak_kill flags correct                  #
# ------------------------------------------------------------------ #

def test_kill_flags_drawdown_kill(tmp_path):
    rec = make_recorder(tmp_path)
    # drawdown over threshold, streak under → drawdown_kill=True, streak_kill=False
    rec.record(
        pair="GBP_USD",
        rf_drawdown_pct=0.030,  # > 0.025 threshold
        rf_streak_prob=0.50,    # < 0.95 threshold
        risk_passed=False,
        max_drawdown_pct=0.025,
    )
    with rec._lock:
        snap = list(rec._records)[0]
    assert snap["drawdown_kill"] is True
    assert snap["streak_kill"] is False
    assert snap["risk_passed"] is False


def test_kill_flags_streak_kill(tmp_path):
    rec = make_recorder(tmp_path)
    # drawdown under threshold, streak over → drawdown_kill=False, streak_kill=True
    rec.record(
        pair="USD_JPY",
        rf_drawdown_pct=0.010,  # < 0.025
        rf_streak_prob=0.96,    # > 0.95
        risk_passed=False,
        max_drawdown_pct=0.025,
        max_streak_prob=0.95,
    )
    with rec._lock:
        snap = list(rec._records)[0]
    assert snap["drawdown_kill"] is False
    assert snap["streak_kill"] is True


def test_kill_flags_both_kills(tmp_path):
    rec = make_recorder(tmp_path)
    rec.record(
        pair="EUR_JPY",
        rf_drawdown_pct=0.030,  # > 0.025
        rf_streak_prob=0.98,    # > 0.95
        risk_passed=False,
        max_drawdown_pct=0.025,
        max_streak_prob=0.95,
    )
    with rec._lock:
        snap = list(rec._records)[0]
    assert snap["drawdown_kill"] is True
    assert snap["streak_kill"] is True


# ------------------------------------------------------------------ #
# Test 3: get_distribution() correct                                   #
# ------------------------------------------------------------------ #

def test_distribution_multiple_records(tmp_path):
    rec = make_recorder(tmp_path)
    values = [0.010, 0.020, 0.030, 0.040]
    for v in values:
        record_pass(rec, drawdown=v)
    dist = rec.get_distribution()
    assert dist["count"] == 4
    assert abs(dist["min"] - 0.010) < 1e-7
    assert abs(dist["max"] - 0.040) < 1e-7
    # mean = (0.010+0.020+0.030+0.040)/4 = 0.025
    assert abs(dist["mean"] - 0.025) < 1e-7
    # p50 on sorted [0.010, 0.020, 0.030, 0.040] ≈ 0.025
    assert 0.020 < dist["p50"] < 0.035


# ------------------------------------------------------------------ #
# Test 4: get_fail_distribution() filters correctly                    #
# ------------------------------------------------------------------ #

def test_fail_distribution_filters(tmp_path):
    rec = make_recorder(tmp_path)
    record_pass(rec, drawdown=0.010)   # should be excluded
    record_fail(rec, drawdown=0.030)   # included
    record_fail(rec, drawdown=0.040)   # included
    fail_dist = rec.get_fail_distribution()
    assert fail_dist["count"] == 2
    assert abs(fail_dist["min"] - 0.030) < 1e-7
    assert abs(fail_dist["max"] - 0.040) < 1e-7


def test_fail_distribution_all_pass_returns_empty(tmp_path):
    rec = make_recorder(tmp_path)
    record_pass(rec)
    record_pass(rec)
    fail_dist = rec.get_fail_distribution()
    assert fail_dist["count"] == 0
    assert fail_dist["p50"] == 0.0


# ------------------------------------------------------------------ #
# Test 5: get_kill_breakdown() counts correctly                        #
# ------------------------------------------------------------------ #

def test_kill_breakdown_drawdown_vs_streak(tmp_path):
    rec = make_recorder(tmp_path)
    # 8 drawdown kills (drawdown > max, streak < max)
    for _ in range(8):
        rec.record("EUR_USD", rf_drawdown_pct=0.030, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=0.025)
    # 2 streak kills (drawdown < max, streak > max)
    for _ in range(2):
        rec.record("EUR_USD", rf_drawdown_pct=0.010, rf_streak_prob=0.97,
                   risk_passed=False, max_drawdown_pct=0.025, max_streak_prob=0.95)
    breakdown = rec.get_kill_breakdown()
    assert breakdown["total_fails"] == 10
    assert breakdown["drawdown_kills"] == 8
    assert breakdown["streak_kills"] == 2
    assert breakdown["both_kills"] == 0
    assert abs(breakdown["drawdown_kill_pct"] - 80.0) < 0.01
    assert abs(breakdown["streak_kill_pct"] - 20.0) < 0.01


def test_kill_breakdown_both_kills(tmp_path):
    rec = make_recorder(tmp_path)
    # 3 both-kills
    for _ in range(3):
        rec.record("EUR_USD", rf_drawdown_pct=0.030, rf_streak_prob=0.97,
                   risk_passed=False, max_drawdown_pct=0.025, max_streak_prob=0.95)
    breakdown = rec.get_kill_breakdown()
    assert breakdown["total_fails"] == 3
    assert breakdown["drawdown_kills"] == 3
    assert breakdown["streak_kills"] == 3
    assert breakdown["both_kills"] == 3


# ------------------------------------------------------------------ #
# Test 6: empty safe defaults                                          #
# ------------------------------------------------------------------ #

def test_empty_recorder_safe_defaults(tmp_path):
    rec = make_recorder(tmp_path)
    dist = rec.get_distribution()
    assert dist["count"] == 0
    assert dist["mean"] == 0.0
    assert dist["p50"] == 0.0
    fail_dist = rec.get_fail_distribution()
    assert fail_dist["count"] == 0
    breakdown = rec.get_kill_breakdown()
    assert breakdown["total_fails"] == 0
    assert breakdown["drawdown_kill_pct"] == 0.0
    assert rec.get_pass_rate() == 0.0


# ------------------------------------------------------------------ #
# Test 7: get_pass_rate() correct                                      #
# ------------------------------------------------------------------ #

def test_pass_rate_correct(tmp_path):
    rec = make_recorder(tmp_path)
    for _ in range(7):
        record_pass(rec)
    for _ in range(3):
        record_fail(rec)
    assert abs(rec.get_pass_rate() - 0.70) < 1e-9


def test_pass_rate_all_fail(tmp_path):
    rec = make_recorder(tmp_path)
    for _ in range(5):
        record_fail(rec)
    assert rec.get_pass_rate() == 0.0


# ------------------------------------------------------------------ #
# Test 8: save / load round-trip                                       #
# ------------------------------------------------------------------ #

def test_save_load_roundtrip(tmp_path):
    state_path = tmp_path / "state.jsonl"
    rec1 = RiskRejectRecorder(state_path=state_path)
    rec1.record("EUR_USD", rf_drawdown_pct=0.028, rf_streak_prob=0.40,
                risk_passed=False, max_drawdown_pct=0.025, scan_id="s1")
    rec1.record("GBP_USD", rf_drawdown_pct=0.012, rf_streak_prob=0.55,
                risk_passed=True, max_drawdown_pct=0.025, scan_id="s2")
    rec1.save_state()

    rec2 = RiskRejectRecorder(state_path=state_path)
    dist = rec2.get_distribution()
    assert dist["count"] == 2
    assert abs(rec2.get_pass_rate() - 0.5) < 1e-9


# ------------------------------------------------------------------ #
# Test 9: atomic JSONL write (no partial state)                        #
# ------------------------------------------------------------------ #

def test_atomic_write_no_tmp_left(tmp_path):
    state_path = tmp_path / "state.jsonl"
    rec = RiskRejectRecorder(state_path=state_path)
    record_fail(rec, drawdown=0.030)
    rec.save_state()
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0
    assert state_path.exists()


# ------------------------------------------------------------------ #
# Test 10: max_size enforced                                           #
# ------------------------------------------------------------------ #

def test_max_size_enforced(tmp_path):
    rec = RiskRejectRecorder(max_size=5, state_path=tmp_path / "s.jsonl")
    for i in range(10):
        rec.record("EUR_USD", rf_drawdown_pct=0.01 * i, rf_streak_prob=0.5,
                   risk_passed=True, max_drawdown_pct=0.025)
    dist = rec.get_distribution()
    assert dist["count"] == 5
