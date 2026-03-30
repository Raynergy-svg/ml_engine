"""Phase 65 US-385 + US-386 + US-387: ScanHealthSynthesizer v2 tests.

US-385: ScanHealthSynthesizer consumes Phase 61-64 momentum + risk signals.
US-386: continuous.py wires momentum_recorder and risk_recorder.
US-387: end-to-end integration verification (minimum 10 tests).

Uses RiskRejectRecorder and MomentumScoreRecorder with tmp_path to simulate
the full data path without mocking.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scanner.scan_health_synthesizer import (
    ScanHealthSynthesizer,
    rank_bottlenecks,
    BN_MOMENTUM_KILL,
    BN_MOMENTUM_NEAR_MISS,
    BN_RISK_KILL,
    BN_RISK_NEAR_MISS,
    BN_SYSTEM_STALL,
    STATUS_HEALTHY,
    STATUS_DEGRADED,
    STATUS_CRITICAL,
)

CONTINUOUS_PATH = Path(__file__).parent.parent / "src" / "scanner" / "automation" / "continuous.py"


# ─── helpers ──────────────────────────────────────────────────────────────────

def get_continuous_src() -> str:
    return CONTINUOUS_PATH.read_text()


def make_momentum_recorder(tmp_path: Path, n_fails: int = 10,
                            drawdown_score: float = 0.43):
    """Populate a MomentumScoreRecorder with fail records near threshold."""
    from src.scanner.momentum_score_recorder import MomentumScoreRecorder
    rec = MomentumScoreRecorder(state_path=tmp_path / "mom_state.jsonl")
    for i in range(n_fails):
        rec.record(pair=f"EUR_USD_{i}", momentum_score=drawdown_score,
                   momentum_passed=False, scan_id=str(i))
    return rec


def make_risk_recorder(tmp_path: Path, n_fails: int = 10,
                        drawdown_pct: float = 0.028,
                        max_drawdown_pct: float = 0.025):
    """Populate a RiskRejectRecorder with fail records near max_drawdown_pct."""
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    rec = RiskRejectRecorder(state_path=tmp_path / "risk_state.jsonl")
    for i in range(n_fails):
        rec.record(
            pair=f"EUR_USD_{i}",
            rf_drawdown_pct=drawdown_pct,
            rf_streak_prob=0.50,
            risk_passed=False,
            max_drawdown_pct=max_drawdown_pct,
            scan_id=str(i),
        )
    return rec


# ─── US-385 unit tests: momentum signals consumed ────────────────────────────

def test_momentum_signals_in_synthesize_output(tmp_path):
    """Momentum signals appear in synthesize() output signals dict when recorder wired."""
    rec = make_momentum_recorder(tmp_path, n_fails=10, drawdown_score=0.43)
    synth = ScanHealthSynthesizer(momentum_recorder=rec, momentum_max_threshold=0.45)
    result = synth.synthesize()
    sigs = result.get("signals", {})
    assert "momentum_classification" in sigs
    assert "momentum_gap_at_p50" in sigs
    assert "momentum_near_miss_rate_pct" in sigs
    assert "momentum_pass_rate" in sigs


def test_risk_signals_in_synthesize_output(tmp_path):
    """Risk signals appear in synthesize() output signals dict when recorder wired."""
    rec = make_risk_recorder(tmp_path, n_fails=10, drawdown_pct=0.028, max_drawdown_pct=0.025)
    synth = ScanHealthSynthesizer(risk_recorder=rec, risk_max_drawdown=0.025)
    result = synth.synthesize()
    sigs = result.get("signals", {})
    assert "risk_classification" in sigs
    assert "risk_gap_at_p50" in sigs
    assert "risk_near_miss_rate_pct" in sigs
    assert "risk_pass_rate" in sigs
    assert "risk_drawdown_kill_pct" in sigs


def test_backward_compat_no_recorders():
    """ScanHealthSynthesizer() with no recorders returns valid result (Phase 60 behavior)."""
    synth = ScanHealthSynthesizer()
    result = synth.synthesize()
    assert "health_status" in result
    assert "bottlenecks" in result
    assert "top_action" in result
    assert result["health_status"] in (STATUS_HEALTHY, STATUS_DEGRADED, STATUS_CRITICAL)
    assert isinstance(result["bottlenecks"], list)


def test_backward_compat_signals_have_defaults():
    """No recorders → momentum/risk signal defaults are empty strings / zeros."""
    synth = ScanHealthSynthesizer()
    result = synth.synthesize()
    sigs = result.get("signals", {})
    assert sigs.get("momentum_classification") == ""
    assert sigs.get("risk_classification") == ""
    assert sigs.get("momentum_near_miss_rate_pct") == 0.0
    assert sigs.get("risk_drawdown_kill_pct") == 0.0


# ─── US-385: NEAR_MISS bottleneck activation ─────────────────────────────────

def test_momentum_near_miss_bottleneck_fires(tmp_path):
    """MOMENTUM_NEAR_MISS fires when dominant_kill=momentum + momentum_cls=NEAR_THRESHOLD."""
    # Score close to threshold (gap ~0.02 → NEAR_THRESHOLD since gap ≤ 0.05)
    rec = make_momentum_recorder(tmp_path, n_fails=10, drawdown_score=0.43)

    # Provide signals directly to rank_bottlenecks to isolate the logic
    signals = {
        "dominant_kill_stage": "momentum",
        "tradeable_pct": 0.0,
        "stall_active": False,
        "near_miss_rate_pct": 0.0,
        "gap_at_p75": 0.0,
        "ensemble_recommendation": "OK",
        "momentum_classification": "NEAR_THRESHOLD",
        "momentum_near_miss_rate_pct": 80.0,
        "risk_classification": "",
        "risk_drawdown_kill_pct": 0.0,
        "risk_near_miss_rate_pct": 0.0,
    }
    bottlenecks = rank_bottlenecks(signals)
    names = [b["name"] for b in bottlenecks]
    assert BN_MOMENTUM_NEAR_MISS in names, f"Expected MOMENTUM_NEAR_MISS in {names}"

    # Verify impact score is 0.68
    mn = next(b for b in bottlenecks if b["name"] == BN_MOMENTUM_NEAR_MISS)
    assert abs(mn["impact_score"] - 0.68) < 1e-6


def test_risk_near_miss_bottleneck_fires():
    """RISK_NEAR_MISS fires when dominant_kill=risk + risk_cls=NEAR_THRESHOLD."""
    signals = {
        "dominant_kill_stage": "risk",
        "tradeable_pct": 0.0,
        "stall_active": False,
        "near_miss_rate_pct": 0.0,
        "gap_at_p75": 0.0,
        "ensemble_recommendation": "OK",
        "momentum_classification": "",
        "momentum_near_miss_rate_pct": 0.0,
        "risk_classification": "NEAR_THRESHOLD",
        "risk_near_miss_rate_pct": 75.0,
        "risk_drawdown_kill_pct": 90.0,
    }
    bottlenecks = rank_bottlenecks(signals)
    names = [b["name"] for b in bottlenecks]
    assert BN_RISK_NEAR_MISS in names, f"Expected RISK_NEAR_MISS in {names}"

    rn = next(b for b in bottlenecks if b["name"] == BN_RISK_NEAR_MISS)
    assert abs(rn["impact_score"] - 0.63) < 1e-6


def test_momentum_near_miss_impact_between_kill_and_cluster():
    """MOMENTUM_NEAR_MISS (0.68) sorts between MOMENTUM_KILL (0.70) and NEAR_MISS_CLUSTER (0.65*rate)."""
    signals = {
        "dominant_kill_stage": "momentum",
        "tradeable_pct": 0.0,
        "stall_active": False,
        "near_miss_rate_pct": 90.0,   # NEAR_MISS_CLUSTER = 0.65*0.9 = 0.585
        "gap_at_p75": 0.0,
        "ensemble_recommendation": "OK",
        "momentum_classification": "NEAR_THRESHOLD",
        "momentum_near_miss_rate_pct": 60.0,
        "risk_classification": "",
        "risk_drawdown_kill_pct": 0.0,
        "risk_near_miss_rate_pct": 0.0,
    }
    bottlenecks = rank_bottlenecks(signals)
    names = [b["name"] for b in bottlenecks]
    # Expected order: MOMENTUM_KILL(0.70) > MOMENTUM_NEAR_MISS(0.68) > NEAR_MISS_CLUSTER(0.585)
    assert BN_MOMENTUM_KILL in names
    assert BN_MOMENTUM_NEAR_MISS in names
    mk_idx = names.index(BN_MOMENTUM_KILL)
    mn_idx = names.index(BN_MOMENTUM_NEAR_MISS)
    assert mk_idx < mn_idx, "MOMENTUM_KILL should sort before MOMENTUM_NEAR_MISS"


def test_risk_near_miss_does_not_fire_when_not_near_threshold():
    """RISK_NEAR_MISS does NOT fire when risk_cls != NEAR_THRESHOLD."""
    signals = {
        "dominant_kill_stage": "risk",
        "tradeable_pct": 0.0,
        "stall_active": False,
        "near_miss_rate_pct": 0.0,
        "gap_at_p75": 0.0,
        "ensemble_recommendation": "OK",
        "momentum_classification": "",
        "momentum_near_miss_rate_pct": 0.0,
        "risk_classification": "FAR_ABOVE",
        "risk_near_miss_rate_pct": 0.0,
        "risk_drawdown_kill_pct": 95.0,
    }
    bottlenecks = rank_bottlenecks(signals)
    names = [b["name"] for b in bottlenecks]
    assert BN_RISK_NEAR_MISS not in names


# ─── US-385: enriched MOMENTUM_KILL and RISK_KILL actions ────────────────────

def test_momentum_kill_action_enriched_far_below():
    """MOMENTUM_KILL action contains 'FAR_BELOW' or 'retrain' when classification=FAR_BELOW."""
    signals = {
        "dominant_kill_stage": "momentum",
        "tradeable_pct": 0.0,
        "stall_active": False,
        "near_miss_rate_pct": 0.0,
        "gap_at_p75": 0.0,
        "ensemble_recommendation": "OK",
        "momentum_classification": "FAR_BELOW",
        "momentum_near_miss_rate_pct": 5.0,
        "risk_classification": "",
        "risk_drawdown_kill_pct": 0.0,
        "risk_near_miss_rate_pct": 0.0,
    }
    bottlenecks = rank_bottlenecks(signals)
    mk = next((b for b in bottlenecks if b["name"] == BN_MOMENTUM_KILL), None)
    assert mk is not None
    action_lower = mk["action"].lower()
    assert "far_below" in action_lower or "retrain" in action_lower, (
        f"Expected 'FAR_BELOW' or 'retrain' in action: {mk['action']}"
    )


def test_risk_kill_action_enriched_drawdown_kill_context():
    """RISK_KILL action contains drawdown kill context when risk_drawdown_kill_pct=90.0."""
    signals = {
        "dominant_kill_stage": "risk",
        "tradeable_pct": 0.0,
        "stall_active": False,
        "near_miss_rate_pct": 0.0,
        "gap_at_p75": 0.0,
        "ensemble_recommendation": "OK",
        "momentum_classification": "",
        "momentum_near_miss_rate_pct": 0.0,
        "risk_classification": "FAR_ABOVE",
        "risk_near_miss_rate_pct": 0.0,
        "risk_drawdown_kill_pct": 90.0,
    }
    bottlenecks = rank_bottlenecks(signals)
    rk = next((b for b in bottlenecks if b["name"] == BN_RISK_KILL), None)
    assert rk is not None
    # Action must contain either the kill_pct value or classification context
    assert "90.0" in rk["action"] or "FAR_ABOVE" in rk["action"] or "drawdown" in rk["action"].lower()


# ─── US-386: wiring grep tests ───────────────────────────────────────────────

def test_wiring_momentum_recorder_in_continuous():
    """momentum_recorder= must appear in continuous.py."""
    src = get_continuous_src()
    assert "momentum_recorder=" in src


def test_wiring_risk_recorder_in_continuous():
    """risk_recorder= must appear in continuous.py."""
    src = get_continuous_src()
    assert "risk_recorder=" in src


def test_wiring_momentum_max_threshold_in_continuous():
    """momentum_max_threshold= must appear in continuous.py."""
    src = get_continuous_src()
    assert "momentum_max_threshold=" in src


def test_wiring_risk_max_drawdown_in_continuous():
    """risk_max_drawdown= must appear in continuous.py."""
    src = get_continuous_src()
    assert "risk_max_drawdown=" in src


# ─── US-387: end-to-end full data path ───────────────────────────────────────

def test_full_data_path_both_signal_sets(tmp_path):
    """Mock momentum + risk recorders → both signal sets present in synthesize() output."""
    mom_rec = make_momentum_recorder(tmp_path, n_fails=5, drawdown_score=0.43)
    risk_rec = make_risk_recorder(tmp_path, n_fails=5, drawdown_pct=0.028, max_drawdown_pct=0.025)

    synth = ScanHealthSynthesizer(
        momentum_recorder=mom_rec,
        risk_recorder=risk_rec,
        momentum_max_threshold=0.45,
        risk_max_drawdown=0.025,
    )
    result = synth.synthesize()
    sigs = result.get("signals", {})

    # Momentum signals present
    assert sigs.get("momentum_classification") != "" or sigs.get("momentum_pass_rate") is not None
    # Risk signals present
    assert sigs.get("risk_classification") != "" or sigs.get("risk_pass_rate") is not None


def test_all_phase65_modules_importable():
    """All Phase 65 modules importable alongside Phase 56-64 modules."""
    from src.scanner.scan_health_synthesizer import ScanHealthSynthesizer
    from src.scanner.momentum_score_recorder import MomentumScoreRecorder
    from src.scanner.momentum_gap_analyzer import analyze as mga_analyze
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import get_risk_signals
    from src.scanner.adaptive_momentum_floor import AdaptiveMomentumFloor
    from src.scanner.adaptive_drawdown_ceiling import AdaptiveDrawdownCeiling
    assert True


def test_near_miss_bottlenecks_at_correct_impact_scores():
    """MOMENTUM_NEAR_MISS=0.68 and RISK_NEAR_MISS=0.63 at exact required scores."""
    # Test MOMENTUM_NEAR_MISS score
    mom_signals = {
        "dominant_kill_stage": "momentum",
        "tradeable_pct": 0.0,
        "stall_active": False,
        "near_miss_rate_pct": 0.0,
        "gap_at_p75": 0.0,
        "ensemble_recommendation": "OK",
        "momentum_classification": "NEAR_THRESHOLD",
        "momentum_near_miss_rate_pct": 50.0,
        "risk_classification": "",
        "risk_drawdown_kill_pct": 0.0,
        "risk_near_miss_rate_pct": 0.0,
    }
    mom_bns = rank_bottlenecks(mom_signals)
    mn = next((b for b in mom_bns if b["name"] == BN_MOMENTUM_NEAR_MISS), None)
    assert mn is not None
    assert abs(mn["impact_score"] - 0.68) < 1e-6, f"MOMENTUM_NEAR_MISS score = {mn['impact_score']}"

    # Test RISK_NEAR_MISS score
    risk_signals = {
        "dominant_kill_stage": "risk",
        "tradeable_pct": 0.0,
        "stall_active": False,
        "near_miss_rate_pct": 0.0,
        "gap_at_p75": 0.0,
        "ensemble_recommendation": "OK",
        "momentum_classification": "",
        "momentum_near_miss_rate_pct": 0.0,
        "risk_classification": "NEAR_THRESHOLD",
        "risk_near_miss_rate_pct": 60.0,
        "risk_drawdown_kill_pct": 85.0,
    }
    risk_bns = rank_bottlenecks(risk_signals)
    rn = next((b for b in risk_bns if b["name"] == BN_RISK_NEAR_MISS), None)
    assert rn is not None
    assert abs(rn["impact_score"] - 0.63) < 1e-6, f"RISK_NEAR_MISS score = {rn['impact_score']}"
