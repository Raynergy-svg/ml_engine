"""Tests for Phase 57 US-353: CalibrationGuard.

Covers:
  - Valid report with sufficient predictions and bins passes
  - Insufficient n_predictions blocks penalty
  - Zero valid_bins blocks penalty
  - None overconfidence_ratio blocks penalty
  - Exactly at threshold passes
  - should_apply_penalty returns False when guard blocks
  - should_apply_penalty returns False when ratio <= 0.15 (not overconfident)
  - get_status returns correct suppression_reason
  - get_stats tracks suppression count
  - Non-dict report returns False gracefully
"""

import pytest

from src.scanner.calibration_guard import CalibrationGuard, MIN_VALID_PREDICTIONS, MIN_VALID_BINS


def make_valid_report(
    n_predictions: int = 25,
    valid_bins: int = 3,
    overconfidence_ratio: float = 0.40,
) -> dict:
    return {
        "model": "TCN",
        "n_predictions": n_predictions,
        "valid_bins": valid_bins,
        "overconfidence_ratio": overconfidence_ratio,
        "calibration_error": 0.05,
    }


class TestIsValid:

    def test_valid_report_passes(self):
        guard = CalibrationGuard()
        assert guard.is_valid(make_valid_report()) is True

    def test_insufficient_predictions_blocked(self):
        guard = CalibrationGuard(min_valid_predictions=20)
        report = make_valid_report(n_predictions=10)
        assert guard.is_valid(report) is False

    def test_exactly_at_threshold_passes(self):
        guard = CalibrationGuard(min_valid_predictions=20, min_valid_bins=2)
        report = make_valid_report(n_predictions=20, valid_bins=2)
        assert guard.is_valid(report) is True

    def test_zero_valid_bins_blocked(self):
        guard = CalibrationGuard()
        report = make_valid_report(valid_bins=0)
        assert guard.is_valid(report) is False

    def test_one_valid_bin_blocked(self):
        guard = CalibrationGuard(min_valid_bins=2)
        report = make_valid_report(valid_bins=1)
        assert guard.is_valid(report) is False

    def test_none_overconfidence_ratio_blocked(self):
        guard = CalibrationGuard()
        report = make_valid_report(overconfidence_ratio=None)
        assert guard.is_valid(report) is False

    def test_non_dict_report_blocked(self):
        guard = CalibrationGuard()
        assert guard.is_valid(None) is False
        assert guard.is_valid("not_a_dict") is False


class TestShouldApplyPenalty:

    def test_valid_report_high_ratio_applies(self):
        guard = CalibrationGuard()
        report = make_valid_report(overconfidence_ratio=0.80)
        assert guard.should_apply_penalty(report) is True

    def test_valid_report_low_ratio_no_penalty(self):
        """Even with valid data, ratio <= 0.15 means no overconfidence."""
        guard = CalibrationGuard()
        report = make_valid_report(overconfidence_ratio=0.10)
        assert guard.should_apply_penalty(report) is False

    def test_insufficient_predictions_suppresses(self):
        guard = CalibrationGuard(min_valid_predictions=20)
        report = make_valid_report(n_predictions=6, overconfidence_ratio=0.80)
        assert guard.should_apply_penalty(report) is False

    def test_zero_bins_suppresses(self):
        guard = CalibrationGuard()
        report = make_valid_report(valid_bins=0, overconfidence_ratio=0.80)
        assert guard.should_apply_penalty(report) is False


class TestGetStatus:

    def test_status_valid_report(self):
        guard = CalibrationGuard()
        status = guard.get_status(make_valid_report(overconfidence_ratio=0.50))
        assert status["is_valid"] is True
        assert status["suppression_reason"] is None
        assert status["will_apply_penalty"] is True

    def test_status_insufficient_predictions(self):
        guard = CalibrationGuard(min_valid_predictions=20)
        status = guard.get_status(make_valid_report(n_predictions=6))
        assert status["is_valid"] is False
        assert "insufficient_predictions" in status["suppression_reason"]

    def test_status_invalid_report_format(self):
        guard = CalibrationGuard()
        status = guard.get_status(None)
        assert status["is_valid"] is False
        assert status["suppression_reason"] == "invalid_report_format"


class TestGetStats:

    def test_suppression_count_increments(self):
        guard = CalibrationGuard(min_valid_predictions=20)
        guard.should_apply_penalty(make_valid_report(n_predictions=6))  # suppressed
        guard.should_apply_penalty(make_valid_report(n_predictions=25))  # not suppressed
        stats = guard.get_stats()
        assert stats["total_checks"] == 2
        assert stats["suppression_count"] == 1
        assert stats["suppression_rate"] == pytest.approx(0.5, abs=0.01)
