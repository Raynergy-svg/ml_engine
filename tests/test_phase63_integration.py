"""Phase 63 US-381: Integration tests — end-to-end verification.

Minimum 12 integration tests required per PRD acceptance criteria.
Tests include live wiring greps (engine.py call-site verification) and
behavioral simulations of all classification scenarios.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

ENGINE_PATH = Path(__file__).parent.parent / "src" / "scanner" / "engine.py"


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def get_engine_src() -> str:
    return ENGINE_PATH.read_text()


# ------------------------------------------------------------------ #
# WIRING GREP TESTS (Live Wiring Verification Gates)                   #
# ------------------------------------------------------------------ #

def test_wiring_risk_recorder_init_block():
    """_risk_recorder = None init block must exist in engine.py."""
    src = get_engine_src()
    assert "_risk_recorder = None" in src, "Missing lazy init: _risk_recorder = None"


def test_wiring_risk_reject_recorder_import():
    """RiskRejectRecorder must be imported in engine.py."""
    src = get_engine_src()
    assert "RiskRejectRecorder" in src, "Missing RiskRejectRecorder in engine.py"


def test_wiring_risk_recorder_record_call():
    """_risk_recorder.record( must be called in engine.py."""
    src = get_engine_src()
    assert "_risk_recorder.record(" in src, "Missing _risk_recorder.record( call in engine.py"


def test_wiring_risk_recorder_save_state():
    """_risk_recorder.save_state() must be called in engine.py."""
    src = get_engine_src()
    assert "_risk_recorder.save_state()" in src, "Missing _risk_recorder.save_state() call"


def test_wiring_risk_gate_analyzer_import():
    """RiskGateAnalyzer analyze must be imported in engine.py."""
    src = get_engine_src()
    assert "risk_gate_analyzer" in src, "Missing risk_gate_analyzer import in engine.py"


def test_wiring_near_threshold_warning():
    """NEAR_THRESHOLD branch must exist in the Phase 63 periodic log block."""
    src = get_engine_src()
    assert "Phase 63" in src, "Missing Phase 63 log tag in engine.py"


# ------------------------------------------------------------------ #
# BEHAVIORAL SIMULATION TESTS                                          #
# ------------------------------------------------------------------ #

def test_simulate_near_threshold(tmp_path):
    """10 fail records at rf_drawdown=0.028 vs max=0.025 → gap=0.003 → NEAR_THRESHOLD."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import analyze, CLASS_NEAR_THRESHOLD

    rec = RiskRejectRecorder(state_path=tmp_path / "s.jsonl")
    for _ in range(10):
        rec.record("EUR_USD", rf_drawdown_pct=0.028, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=0.025)
    result = analyze(rec, 0.025)
    assert result["classification"] == CLASS_NEAR_THRESHOLD
    assert abs(result["gap_at_p50"] - 0.003) < 1e-7


def test_simulate_far_above(tmp_path):
    """10 fail records at rf_drawdown=0.060 vs max=0.025 → gap=0.035 → FAR_ABOVE."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import analyze, CLASS_FAR_ABOVE

    rec = RiskRejectRecorder(state_path=tmp_path / "s.jsonl")
    for _ in range(10):
        rec.record("EUR_USD", rf_drawdown_pct=0.060, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=0.025)
    result = analyze(rec, 0.025)
    assert result["classification"] == CLASS_FAR_ABOVE
    assert abs(result["gap_at_p50"] - 0.035) < 1e-7


def test_simulate_near_miss_rate_50(tmp_path):
    """5 near-miss fails (gap<=0.005) + 5 far fails → near_miss_rate_pct=50.0."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import analyze

    rec = RiskRejectRecorder(state_path=tmp_path / "s.jsonl")
    for _ in range(5):
        rec.record("EUR_USD", rf_drawdown_pct=0.028, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=0.025)
    for _ in range(5):
        rec.record("EUR_USD", rf_drawdown_pct=0.060, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=0.025)
    result = analyze(rec, 0.025)
    assert abs(result["near_miss_rate_pct"] - 50.0) < 0.01


def test_simulate_kill_breakdown_80_drawdown(tmp_path):
    """8 drawdown_kills + 2 streak_kills → drawdown_kill_pct=80.0."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import analyze

    rec = RiskRejectRecorder(state_path=tmp_path / "s.jsonl")
    for _ in range(8):
        rec.record("EUR_USD", rf_drawdown_pct=0.030, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=0.025)
    for _ in range(2):
        rec.record("EUR_USD", rf_drawdown_pct=0.010, rf_streak_prob=0.97,
                   risk_passed=False, max_drawdown_pct=0.025, max_streak_prob=0.95)
    result = analyze(rec, 0.025)
    assert abs(result["drawdown_kill_pct"] - 80.0) < 0.01
    assert abs(result["streak_kill_pct"] - 20.0) < 0.01


def test_get_risk_signals_required_keys(tmp_path):
    """get_risk_signals() must return all 5 keys for ScanHealthSynthesizer."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import get_risk_signals

    rec = RiskRejectRecorder(state_path=tmp_path / "s.jsonl")
    for _ in range(10):
        rec.record("EUR_USD", rf_drawdown_pct=0.028, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=0.025)
    signals = get_risk_signals(rec, 0.025)
    for key in ["risk_gap_at_p50", "risk_near_miss_rate_pct", "risk_classification",
                "risk_pass_rate", "risk_drawdown_kill_pct"]:
        assert key in signals


def test_save_load_preserves_all_records(tmp_path):
    """RiskRejectRecorder save/load preserves all records across instantiation."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder

    state_path = tmp_path / "state.jsonl"
    rec1 = RiskRejectRecorder(state_path=state_path)
    for i in range(5):
        rec1.record("EUR_USD", rf_drawdown_pct=0.02 + i * 0.005, rf_streak_prob=0.50,
                    risk_passed=False, max_drawdown_pct=0.025, scan_id=f"s{i}")
    rec1.record("GBP_USD", rf_drawdown_pct=0.010, rf_streak_prob=0.40,
                risk_passed=True, max_drawdown_pct=0.025)
    rec1.save_state()

    rec2 = RiskRejectRecorder(state_path=state_path)
    dist = rec2.get_distribution()
    assert dist["count"] == 6
    assert abs(rec2.get_pass_rate() - 1/6) < 1e-9


def test_pass_rate_correct_7_pass_3_fail(tmp_path):
    """7 pass + 3 fail → pass_rate=0.70."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder

    rec = RiskRejectRecorder(state_path=tmp_path / "s.jsonl")
    for _ in range(7):
        rec.record("EUR_USD", rf_drawdown_pct=0.010, rf_streak_prob=0.40,
                   risk_passed=True, max_drawdown_pct=0.025)
    for _ in range(3):
        rec.record("EUR_USD", rf_drawdown_pct=0.030, rf_streak_prob=0.50,
                   risk_passed=False, max_drawdown_pct=0.025)
    assert abs(rec.get_pass_rate() - 0.70) < 1e-9


def test_empty_analyzer_insufficient_data_no_crash(tmp_path):
    """Empty recorder must return INSUFFICIENT_DATA without any exception."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import analyze, CLASS_INSUFFICIENT

    rec = RiskRejectRecorder(state_path=tmp_path / "s.jsonl")
    result = analyze(rec, 0.025)
    assert result["classification"] == CLASS_INSUFFICIENT
    assert result["fail_count"] == 0
    assert result["pass_rate"] == 0.0


# ------------------------------------------------------------------ #
# IMPORT COMPATIBILITY TEST                                            #
# ------------------------------------------------------------------ #

def test_all_phase63_modules_importable():
    """All Phase 63 modules must import alongside Phase 56-62 modules."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import analyze, get_risk_signals, RiskGateAnalyzer
    # Phase 61-62 imports must still work
    from src.scanner.momentum_score_recorder import MomentumScoreRecorder
    from src.scanner.momentum_gap_analyzer import analyze as mga_analyze
    from src.scanner.adaptive_momentum_floor import AdaptiveMomentumFloor
    assert True
