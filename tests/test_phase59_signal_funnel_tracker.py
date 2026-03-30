"""Tests for Phase 59 US-363: SignalFunnelTracker."""

import pytest
from types import SimpleNamespace
from src.scanner.signal_funnel_tracker import (
    SignalFunnelTracker,
    KILL_STAGE_CONFIDENCE, KILL_STAGE_MOMENTUM, KILL_STAGE_RISK, KILL_STAGE_NONE,
)


def _make_analysis(conf_passed, mom_passed, risk_passed, tradeable, confidence=55.0):
    return SimpleNamespace(
        confidence_gate_passed=conf_passed,
        momentum_gate_passed=mom_passed,
        risk_gate_passed=risk_passed,
        is_tradeable=tradeable,
        confidence=confidence,
        pair="EUR_USD",
    )


class TestRecordScan:

    def test_all_pass_funnel(self, tmp_path):
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        analyses = [_make_analysis(True, True, True, True) for _ in range(5)]
        funnel = tracker.record_scan(analyses)
        assert funnel["total_evaluated"] == 5
        assert funnel["is_tradeable"] == 5
        assert funnel["tradeable_pct"] == pytest.approx(100.0, abs=0.1)
        assert funnel["dominant_kill_stage"] == KILL_STAGE_NONE

    def test_all_fail_at_confidence(self, tmp_path):
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        analyses = [_make_analysis(False, False, False, False) for _ in range(10)]
        funnel = tracker.record_scan(analyses)
        assert funnel["passed_confidence"] == 0
        assert funnel["dominant_kill_stage"] == KILL_STAGE_CONFIDENCE

    def test_pass_conf_fail_momentum(self, tmp_path):
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        # 5 clear confidence, 0 clear momentum
        analyses = [_make_analysis(True, False, False, False) for _ in range(5)]
        funnel = tracker.record_scan(analyses)
        assert funnel["passed_confidence"] == 5
        assert funnel["passed_momentum"] == 0
        assert funnel["dominant_kill_stage"] == KILL_STAGE_MOMENTUM

    def test_dominant_stage_risk(self, tmp_path):
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        # 10 evaluated: 8 clear conf, 7 clear mom, 0 clear risk
        analyses = (
            [_make_analysis(True, True, False, False)] * 7
            + [_make_analysis(True, False, False, False)] * 1
            + [_make_analysis(False, False, False, False)] * 2
        )
        funnel = tracker.record_scan(analyses)
        # drops: conf=2, mom=1, risk=7 → dominant=risk
        assert funnel["dominant_kill_stage"] == KILL_STAGE_RISK

    def test_empty_analyses_safe(self, tmp_path):
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        funnel = tracker.record_scan([])
        assert funnel["total_evaluated"] == 0
        assert funnel["dominant_kill_stage"] == KILL_STAGE_NONE

    def test_get_funnel_summary_aggregates(self, tmp_path):
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        # Scan 1: 10 total, 5 pass conf, 3 pass mom, 2 pass risk, 1 tradeable
        s1 = (
            [_make_analysis(True, True, True, True)] * 1
            + [_make_analysis(True, True, True, False)] * 1
            + [_make_analysis(True, True, False, False)] * 1
            + [_make_analysis(True, False, False, False)] * 2
            + [_make_analysis(False, False, False, False)] * 5
        )
        tracker.record_scan(s1)
        summary = tracker.get_funnel_summary()
        assert summary["scan_count"] == 1
        assert summary["total_evaluated"] == 10
        assert summary["tradeable_pct"] == pytest.approx(10.0, abs=0.1)

    def test_get_recent_funnels(self, tmp_path):
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        for _ in range(5):
            tracker.record_scan([_make_analysis(False, False, False, False)])
        recent = tracker.get_recent_funnels(3)
        assert len(recent) == 3


class TestPersistence:

    def test_save_load_round_trip(self, tmp_path):
        p = tmp_path / "sf.jsonl"
        t1 = SignalFunnelTracker(state_path=p)
        t1.record_scan([_make_analysis(True, True, True, True)])
        t1.save_state()
        t2 = SignalFunnelTracker(state_path=p)
        assert len(t2.get_recent_funnels(10)) == 1

    def test_load_missing_file_no_error(self, tmp_path):
        t = SignalFunnelTracker(state_path=tmp_path / "nonexistent.jsonl")
        assert t.get_funnel_summary()["scan_count"] == 0
