"""Tests for Phase 58 US-359: ThresholdGapAnalyzer.

Covers:
  - OK when >= 10% signals above threshold
  - THRESHOLD_TOO_HIGH when gap_at_p75 > 5.0 and < 5% above
  - NEAR_THRESHOLD when 0 < gap_at_p75 <= 5.0 and < 10% above
  - INSUFFICIENT_DATA when count < 20
  - should_lower_threshold False when not stalled
  - should_lower_threshold True when stalled + recommendation warrants it
  - compute_suggested_threshold respects safety_floor
  - compute_suggested_threshold uses step correctly
"""

import pytest

from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer, REC_OK, \
    REC_THRESHOLD_TOO_HIGH, REC_NEAR_THRESHOLD, REC_INSUFFICIENT_DATA


def _make_recorder(values, tmp_path):
    r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
    for v in values:
        r.record("EUR_USD", v, scan_id="s1")
    return r


class TestAnalyzeRecommendations:

    def test_ok_when_most_signals_above_threshold(self, tmp_path):
        """50 signals at 65.0, threshold=58.0 → >10% above → OK."""
        r = _make_recorder([65.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert report.recommendation == REC_OK
        assert report.signals_above_threshold_pct == pytest.approx(100.0, abs=1.0)

    def test_threshold_too_high_when_large_gap(self, tmp_path):
        """50 signals at 50.0, threshold=58.0 → gap_at_p75=8.0 > 5.0, 0% above → THRESHOLD_TOO_HIGH."""
        r = _make_recorder([50.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert report.recommendation == REC_THRESHOLD_TOO_HIGH
        assert report.gap_at_p75 == pytest.approx(8.0, abs=0.1)
        assert report.signals_above_threshold_pct == pytest.approx(0.0, abs=0.1)

    def test_near_threshold_when_small_gap(self, tmp_path):
        """50 signals at 55.0, threshold=58.0 → gap_at_p75=3.0 (<=5), 0% above → NEAR_THRESHOLD."""
        r = _make_recorder([55.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert report.recommendation == REC_NEAR_THRESHOLD
        assert report.gap_at_p75 == pytest.approx(3.0, abs=0.1)

    def test_insufficient_data_when_count_below_20(self, tmp_path):
        r = _make_recorder([60.0] * 10, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert report.recommendation == REC_INSUFFICIENT_DATA
        assert report.count == 10

    def test_ok_with_mixed_signals_above_10pct(self, tmp_path):
        """10 out of 25 signals above threshold = 40% → OK."""
        values = [60.0] * 10 + [50.0] * 15
        r = _make_recorder(values, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert report.recommendation == REC_OK

    def test_report_fields_all_present(self, tmp_path):
        r = _make_recorder([60.0] * 30, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        for field in ["count", "mean_raw", "median_raw", "p75_raw",
                      "current_threshold", "gap_at_median", "gap_at_p75",
                      "signals_above_threshold_pct", "recommendation"]:
            assert hasattr(report, field), f"Missing field: {field}"


class TestShouldLowerThreshold:

    def test_returns_false_when_not_stalled(self, tmp_path):
        r = _make_recorder([50.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert report.recommendation == REC_THRESHOLD_TOO_HIGH
        # Not stalled yet
        assert analyzer.should_lower_threshold(report, scans_since_last_trade=30) is False

    def test_returns_true_when_stalled_and_too_high(self, tmp_path):
        r = _make_recorder([50.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert analyzer.should_lower_threshold(report, scans_since_last_trade=55) is True

    def test_returns_false_when_ok_even_if_stalled(self, tmp_path):
        r = _make_recorder([65.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert report.recommendation == REC_OK
        assert analyzer.should_lower_threshold(report, scans_since_last_trade=60) is False

    def test_returns_true_for_near_threshold_when_stalled(self, tmp_path):
        r = _make_recorder([55.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(r, min_confidence=58.0)
        assert report.recommendation == REC_NEAR_THRESHOLD
        assert analyzer.should_lower_threshold(report, scans_since_last_trade=50) is True


class TestComputeSuggestedThreshold:

    def test_reduces_by_step(self, tmp_path):
        r = _make_recorder([50.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer(safety_floor=45.0)
        report = analyzer.analyze(r, min_confidence=58.0)
        suggested = analyzer.compute_suggested_threshold(report, step=1.0)
        assert suggested == pytest.approx(57.0, abs=0.01)

    def test_respects_safety_floor(self, tmp_path):
        r = _make_recorder([40.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer(safety_floor=45.0)
        report = analyzer.analyze(r, min_confidence=45.5)
        # 45.5 - 1.0 = 44.5 < floor=45.0 → floor wins
        suggested = analyzer.compute_suggested_threshold(report, step=1.0)
        assert suggested == pytest.approx(45.0, abs=0.01)

    def test_custom_step(self, tmp_path):
        r = _make_recorder([50.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer(safety_floor=40.0)
        report = analyzer.analyze(r, min_confidence=60.0)
        suggested = analyzer.compute_suggested_threshold(report, step=2.0)
        assert suggested == pytest.approx(58.0, abs=0.01)

    def test_custom_safety_floor_override(self, tmp_path):
        r = _make_recorder([40.0] * 50, tmp_path)
        analyzer = ThresholdGapAnalyzer(safety_floor=45.0)
        report = analyzer.analyze(r, min_confidence=47.0)
        # Using custom floor override of 46.5
        suggested = analyzer.compute_suggested_threshold(report, step=1.0, safety_floor=46.5)
        assert suggested == pytest.approx(46.5, abs=0.01)
