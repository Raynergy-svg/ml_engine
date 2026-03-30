"""Tests for Phase 58 US-360: GatePassPredictor.

Covers:
  - all-above-threshold returns 1.0
  - all-below-threshold returns 0.0
  - mixed returns correct fraction
  - find_threshold_for_target binary search
  - empty recorder returns 0.0 safely
  - generate_report includes all required keys
  - signal_exists True when penalty_free >= 5%
  - signal_exists False when penalty_free < 5%
"""

import pytest

from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
from src.scanner.gate_pass_predictor import GatePassPredictor


def _make_recorder(values, tmp_path):
    r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
    for v in values:
        r.record("EUR_USD", v, scan_id="s1")
    return r


class TestPredictPenaltyFreePassRate:

    def test_all_above_returns_one(self, tmp_path):
        r = _make_recorder([65.0] * 20, tmp_path)
        p = GatePassPredictor()
        assert p.predict_penalty_free_pass_rate(r, 58.0) == pytest.approx(1.0)

    def test_all_below_returns_zero(self, tmp_path):
        r = _make_recorder([50.0] * 20, tmp_path)
        p = GatePassPredictor()
        assert p.predict_penalty_free_pass_rate(r, 58.0) == pytest.approx(0.0)

    def test_mixed_fraction(self, tmp_path):
        # 10 above, 10 below → 0.5
        r = _make_recorder([65.0] * 10 + [50.0] * 10, tmp_path)
        p = GatePassPredictor()
        rate = p.predict_penalty_free_pass_rate(r, 58.0)
        assert rate == pytest.approx(0.5, abs=0.01)

    def test_empty_recorder_returns_zero(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        p = GatePassPredictor()
        assert p.predict_penalty_free_pass_rate(r, 58.0) == pytest.approx(0.0)

    def test_predict_at_custom_threshold(self, tmp_path):
        r = _make_recorder([55.0] * 10 + [45.0] * 10, tmp_path)
        p = GatePassPredictor()
        # At threshold=50.0: 10/20 = 0.5
        assert p.predict_pass_rate_at_threshold(r, 50.0) == pytest.approx(0.5, abs=0.01)


class TestFindThreshold:

    def test_find_threshold_for_10pct(self, tmp_path):
        """50 signals uniform 50..99 → threshold for 10% should be around 95."""
        r = _make_recorder(list(range(50, 100)), tmp_path)  # 50 values
        p = GatePassPredictor()
        t = p.find_threshold_for_target_pass_rate(r, target_rate=0.10)
        # At threshold t, ~10% (5 of 50) should pass: values >= 95 → t~95
        pass_rate = p.predict_pass_rate_at_threshold(r, t)
        assert pass_rate >= 0.10 - 0.02  # slight tolerance for binary search

    def test_find_threshold_for_5pct(self, tmp_path):
        """50 signals at 60.0 → 5% target → threshold <= 60.0."""
        r = _make_recorder([60.0] * 50, tmp_path)
        p = GatePassPredictor()
        t = p.find_threshold_for_target_pass_rate(r, target_rate=0.05)
        assert t <= 60.0 + 0.1

    def test_find_threshold_empty_recorder(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        p = GatePassPredictor()
        t = p.find_threshold_for_target_pass_rate(r)
        assert t == 0.0


class TestGenerateReport:

    def test_report_has_all_required_keys(self, tmp_path):
        r = _make_recorder([60.0] * 30, tmp_path)
        p = GatePassPredictor()
        report = p.generate_report(r, min_confidence=58.0)
        for key in ["penalty_free_pass_rate_pct", "current_pass_rate_pct",
                    "threshold_for_10pct", "threshold_for_5pct",
                    "signal_exists", "recommendation", "min_confidence_used"]:
            assert key in report, f"Missing key: {key}"

    def test_signal_exists_true_when_above_5pct(self, tmp_path):
        r = _make_recorder([65.0] * 30, tmp_path)
        p = GatePassPredictor()
        report = p.generate_report(r, min_confidence=58.0)
        assert report["signal_exists"] is True
        assert report["penalty_free_pass_rate_pct"] == pytest.approx(100.0, abs=1.0)

    def test_signal_exists_false_when_below_5pct(self, tmp_path):
        r = _make_recorder([50.0] * 30, tmp_path)
        p = GatePassPredictor()
        report = p.generate_report(r, min_confidence=58.0)
        assert report["signal_exists"] is False
        assert report["penalty_free_pass_rate_pct"] == pytest.approx(0.0, abs=0.1)

    def test_stall_info_included_when_provided(self, tmp_path):
        r = _make_recorder([60.0] * 20, tmp_path)
        p = GatePassPredictor()
        stall_info = {"scans_since_last_trade": 55}
        report = p.generate_report(r, min_confidence=58.0, stall_info=stall_info)
        assert "stall_info" in report
        assert report["stall_info"]["scans_since_last_trade"] == 55

    def test_recommendation_penalties_blocking_when_signal_exists(self, tmp_path):
        r = _make_recorder([65.0] * 30, tmp_path)
        p = GatePassPredictor()
        report = p.generate_report(r, min_confidence=58.0)
        assert report["recommendation"] == "PENALTIES_BLOCKING_SIGNALS"

    def test_recommendation_recalibration_when_no_signal(self, tmp_path):
        r = _make_recorder([50.0] * 30, tmp_path)
        p = GatePassPredictor()
        report = p.generate_report(r, min_confidence=58.0)
        assert report["recommendation"] in (
            "THRESHOLD_RECALIBRATION_NEEDED", "NO_SIGNALS_IN_DISTRIBUTION"
        )
