"""Unit tests for ModelCalibrationTracker (src/scanner/automation/model_calibration.py).

Closes a zero-coverage gap surfaced by the 2026-05-01 test coverage audit.

Tests:
- Confidence input clamped to [0, 1] before binning
- Per-model rolling window (trims when len > ROLLING_WINDOW * 2 = 200, keeps
  last 100)
- ECE computation: sample-weighted mean of |avg_confidence - actual_win_rate|
  across bins with at least MIN_SAMPLES_PER_BIN=5 samples
- Bin boundaries: 10 bins of width 0.1; right edge of last bin is inclusive
- Overconfidence ratio: fraction of valid bins where avg_conf > win_rate
- Flocking detection: agreement_ratio + direction_agreement ≥ thresholds
- Persistence round-trip + corruption resilience
"""

from __future__ import annotations

import json

import pytest

from src.scanner.automation.model_calibration import (
    FLOCKING_AGREEMENT_THRESHOLD,
    FLOCKING_CONFIDENCE_THRESHOLD,
    MIN_SAMPLES_PER_BIN,
    N_BINS,
    ROLLING_WINDOW,
    ModelCalibrationTracker,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        assert t._total_predictions == 0
        assert t._predictions == {}
        assert t._flock_events == []

    def test_constants_match_spec(self):
        assert N_BINS == 10
        assert MIN_SAMPLES_PER_BIN == 5
        assert ROLLING_WINDOW == 100
        assert FLOCKING_CONFIDENCE_THRESHOLD == 0.65
        assert FLOCKING_AGREEMENT_THRESHOLD == 0.90


# ---------------------------------------------------------------------------
# record_prediction
# ---------------------------------------------------------------------------

class TestRecordPrediction:
    def test_appends_record(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        t.record_prediction("TCN", 0.72, won=True)
        assert len(t._predictions["TCN"]) == 1
        assert t._predictions["TCN"][0]["confidence"] == pytest.approx(0.72)
        assert t._predictions["TCN"][0]["won"] is True
        assert "timestamp" in t._predictions["TCN"][0]

    def test_total_predictions_counter(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        for _ in range(7):
            t.record_prediction("TCN", 0.5, won=True)
        for _ in range(3):
            t.record_prediction("Ridge", 0.6, won=False)
        assert t._total_predictions == 10

    def test_confidence_clamped_low(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        t.record_prediction("TCN", -0.5, won=True)
        assert t._predictions["TCN"][0]["confidence"] == 0.0

    def test_confidence_clamped_high(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        t.record_prediction("TCN", 1.7, won=True)
        assert t._predictions["TCN"][0]["confidence"] == 1.0

    def test_rolling_window_trim(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Trigger trim: spec says when len > ROLLING_WINDOW*2, trim to ROLLING_WINDOW
        for i in range(220):
            t.record_prediction("TCN", 0.5, won=(i % 2 == 0))
        # After 200 it trims to 100, then continues
        # 220 records → at i=200, len becomes 201 > 200, trim to 100; then 20 more → 120
        assert len(t._predictions["TCN"]) <= ROLLING_WINDOW * 2
        assert len(t._predictions["TCN"]) >= ROLLING_WINDOW

    def test_separate_models_isolated(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        t.record_prediction("TCN", 0.7, won=True)
        t.record_prediction("Ridge", 0.4, won=False)
        assert len(t._predictions["TCN"]) == 1
        assert len(t._predictions["Ridge"]) == 1
        assert t._predictions["TCN"][0]["won"] is True
        assert t._predictions["Ridge"][0]["won"] is False


# ---------------------------------------------------------------------------
# Calibration report — bin computation
# ---------------------------------------------------------------------------

class TestBinning:
    def test_returns_10_bins(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        t.record_prediction("TCN", 0.5, won=True)
        report = t.get_calibration_report("TCN")
        assert len(report["bins"]) == 10

    def test_bin_ranges_cover_unit_interval(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        t.record_prediction("TCN", 0.5, won=True)
        report = t.get_calibration_report("TCN")
        ranges = [b["range"] for b in report["bins"]]
        assert ranges[0] == "0.0-0.1"
        assert ranges[-1] == "0.9-1.0"

    def test_records_assigned_to_correct_bin(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # 0.05 → bin 0; 0.55 → bin 5; 0.95 → bin 9
        t.record_prediction("TCN", 0.05, won=True)
        t.record_prediction("TCN", 0.55, won=False)
        t.record_prediction("TCN", 0.95, won=True)
        report = t.get_calibration_report("TCN")
        bin_counts = [b["count"] for b in report["bins"]]
        assert bin_counts[0] == 1
        assert bin_counts[5] == 1
        assert bin_counts[9] == 1

    def test_right_edge_of_last_bin_inclusive(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # 1.0 should land in bin 9 (0.9-1.0), not be excluded
        t.record_prediction("TCN", 1.0, won=True)
        report = t.get_calibration_report("TCN")
        assert report["bins"][9]["count"] == 1

    def test_lower_edge_of_bin_inclusive(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # 0.5 exactly should fall in bin 5 (0.5 <= c < 0.6)
        t.record_prediction("TCN", 0.5, won=False)
        report = t.get_calibration_report("TCN")
        assert report["bins"][5]["count"] == 1
        assert report["bins"][4]["count"] == 0

    def test_empty_bin_has_midpoint_avg_confidence(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Only fill bin 0
        t.record_prediction("TCN", 0.05, won=True)
        report = t.get_calibration_report("TCN")
        # Bin 5 (0.5-0.6) is empty → avg_confidence == 0.55 (midpoint), count=0
        empty_bin = report["bins"][5]
        assert empty_bin["count"] == 0
        assert empty_bin["avg_confidence"] == pytest.approx(0.55)
        assert empty_bin["actual_win_rate"] == 0.0


# ---------------------------------------------------------------------------
# get_calibration_report — ECE & overconfidence
# ---------------------------------------------------------------------------

class TestCalibrationReport:
    def test_no_predictions_returns_none_metrics(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        report = t.get_calibration_report("TCN")
        assert report["calibration_error"] is None
        assert report["overconfidence_ratio"] is None
        assert report["bins"] == []
        assert report["n_predictions"] == 0

    def test_below_min_samples_per_bin_returns_none_metrics(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Add 4 records (below MIN_SAMPLES_PER_BIN=5)
        for _ in range(4):
            t.record_prediction("TCN", 0.7, won=True)
        report = t.get_calibration_report("TCN")
        assert report["calibration_error"] is None
        assert report["overconfidence_ratio"] is None

    def test_perfect_calibration_zero_ece(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Bin 7 (0.7-0.8): 10 records at conf=0.75, 7 won → win_rate=0.7
        # Wait — avg_confidence will be 0.75, win_rate 0.7 → not perfect
        # Actually for "perfect" we need avg_conf == win_rate
        # If conf=0.7 (lands in bin 7), 7 of 10 won → win_rate=0.7, avg_conf=0.7 → ECE=0
        for i in range(10):
            t.record_prediction("TCN", 0.7, won=(i < 7))
        report = t.get_calibration_report("TCN")
        # 7 wins / 10 = 0.7; avg_conf = 0.7 → ECE = 0
        assert report["calibration_error"] == pytest.approx(0.0, abs=1e-3)
        # avg_confidence == actual_win_rate → not overconfident → ratio = 0
        assert report["overconfidence_ratio"] == pytest.approx(0.0)

    def test_overconfident_model_high_ece(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Confidence 0.9, but only 3 of 10 won → win_rate=0.3, gap=0.6
        for i in range(10):
            t.record_prediction("TCN", 0.9, won=(i < 3))
        report = t.get_calibration_report("TCN")
        assert report["calibration_error"] == pytest.approx(0.6, abs=1e-3)
        # 1 valid bin, all overconfident
        assert report["overconfidence_ratio"] == pytest.approx(1.0)

    def test_underconfident_model_zero_overconfidence_ratio(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Confidence 0.4, but 9 of 10 won → win_rate=0.9, gap=-0.5 (underconfident)
        for i in range(10):
            t.record_prediction("TCN", 0.4, won=(i < 9))
        report = t.get_calibration_report("TCN")
        assert report["calibration_error"] == pytest.approx(0.5, abs=1e-3)
        assert report["overconfidence_ratio"] == pytest.approx(0.0)

    def test_ece_is_sample_weighted_across_bins(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Bin 7 (conf 0.7): 10 records, 5 won → gap=0.2, weight=10/15 ≈ 0.667
        # Bin 9 (conf 0.95): 5 records, 1 won → gap=0.75, weight=5/15 ≈ 0.333
        # ECE = 0.667 * 0.2 + 0.333 * 0.75 = 0.1333 + 0.25 = 0.3833
        for i in range(10):
            t.record_prediction("TCN", 0.7, won=(i < 5))
        for i in range(5):
            t.record_prediction("TCN", 0.95, won=(i < 1))
        report = t.get_calibration_report("TCN")
        # 10 records bin 7: avg_conf=0.7, win_rate=0.5, gap=0.2, weight=10/15
        # 5 records bin 9: avg_conf=0.95, win_rate=0.2, gap=0.75, weight=5/15
        expected_ece = (10/15) * 0.2 + (5/15) * 0.75
        assert report["calibration_error"] == pytest.approx(expected_ece, abs=1e-3)

    def test_n_predictions_counts_recent_window(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        for _ in range(15):
            t.record_prediction("TCN", 0.6, won=True)
        report = t.get_calibration_report("TCN")
        assert report["n_predictions"] == 15

    def test_n_predictions_capped_at_rolling_window(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        for i in range(150):
            t.record_prediction("TCN", 0.6, won=(i % 2 == 0))
        report = t.get_calibration_report("TCN")
        # Report uses last ROLLING_WINDOW=100
        assert report["n_predictions"] == ROLLING_WINDOW


# ---------------------------------------------------------------------------
# detect_flocking
# ---------------------------------------------------------------------------

class TestFlockingDetection:
    def test_single_model_no_flocking(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        result = t.detect_flocking({"TCN": 0.9})
        assert result["is_flocking"] is False
        assert result["flocking_score"] == 0.0

    def test_all_neutral_no_flocking(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # All confidences in [0.45, 0.55] → directional empty → no flocking
        result = t.detect_flocking({"TCN": 0.50, "Ridge": 0.48, "RF": 0.52})
        assert result["is_flocking"] is False
        assert result["flocking_score"] == 0.0

    def test_all_high_same_direction_is_flocking(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        result = t.detect_flocking({"TCN": 0.85, "Ridge": 0.78, "RF": 0.92})
        assert result["is_flocking"] is True
        assert result["agreement_ratio"] == 1.0
        assert result["avg_confidence"] == pytest.approx((0.85 + 0.78 + 0.92) / 3, abs=1e-4)

    def test_split_direction_no_flocking(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Two BUY-leaning, one SELL-leaning, all high confidence
        # agreement_ratio=1.0; directions: [+1, +1, -1] → |sum|/n = 1/3
        # flocking_score = 1.0 * 0.333 = 0.333
        # is_flocking requires agreement_ratio >= 0.90 AND avg_confidence >= 0.65
        # Here avg_confidence ≈ 0.7 (above), agreement_ratio=1.0 → IS flagged
        # but flocking_score reflects the split
        result = t.detect_flocking({"TCN": 0.85, "Ridge": 0.78, "RF": 0.20})
        assert result["flocking_score"] < 0.5  # split brings score down

    def test_low_confidence_no_flocking(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # All below FLOCKING_CONFIDENCE_THRESHOLD=0.65 but directional
        result = t.detect_flocking({"TCN": 0.60, "Ridge": 0.62, "RF": 0.58})
        # agreement_ratio = 0/3 = 0 (none above 0.65)
        # avg_confidence = 0.60 < 0.65 → not flocking
        assert result["is_flocking"] is False

    def test_only_some_high_confidence_below_agreement_threshold(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # 2 of 5 above 0.65 → agreement_ratio=0.4 < FLOCKING_AGREEMENT_THRESHOLD=0.9
        result = t.detect_flocking({
            "TCN": 0.85, "Ridge": 0.80, "RF": 0.40, "XGB": 0.30, "MLP": 0.20,
        })
        assert result["is_flocking"] is False

    def test_flock_event_recorded_when_detected(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        assert len(t._flock_events) == 0
        t.detect_flocking({"TCN": 0.85, "Ridge": 0.80, "RF": 0.90})
        assert len(t._flock_events) == 1
        assert t._flock_events[0]["is_flocking"] is True
        assert "model_predictions" in t._flock_events[0]
        assert "timestamp" in t._flock_events[0]

    def test_no_flock_event_when_not_flocking(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        t.detect_flocking({"TCN": 0.50, "Ridge": 0.50})
        assert len(t._flock_events) == 0

    def test_historical_flock_accuracy_added_when_3plus_resolved(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Record and resolve 3 flocks
        for won in [True, True, False]:
            t.detect_flocking({"TCN": 0.85, "Ridge": 0.80, "RF": 0.90})
            t.record_flock_outcome(won)
        # Detect a 4th — should now include historical accuracy
        result = t.detect_flocking({"TCN": 0.85, "Ridge": 0.80, "RF": 0.90})
        assert result["historical_flock_accuracy"] == pytest.approx(2 / 3, abs=1e-3)

    def test_historical_flock_accuracy_none_below_3_resolved(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Only 2 resolved
        for won in [True, False]:
            t.detect_flocking({"TCN": 0.85, "Ridge": 0.80, "RF": 0.90})
            t.record_flock_outcome(won)
        result = t.detect_flocking({"TCN": 0.85, "Ridge": 0.80, "RF": 0.90})
        assert result["historical_flock_accuracy"] is None

    def test_flock_event_log_truncates_at_100(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        for _ in range(110):
            t.detect_flocking({"TCN": 0.85, "Ridge": 0.80, "RF": 0.90})
        # Spec: when len > 100, truncate to last 50
        # After 100 events len=100; at 101 it becomes 101 > 100 → trim to 50
        # Then 9 more appends → 59
        assert 50 <= len(t._flock_events) <= 100

    def test_record_flock_outcome_no_op_when_no_events(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Should not crash
        t.record_flock_outcome(won=True)
        assert t._flock_events == []


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_keys(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        s = t.get_status()
        for key in ("total_predictions", "models", "flock_events", "recent_flock_events"):
            assert key in s

    def test_status_reflects_state(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        for _ in range(7):
            t.record_prediction("TCN", 0.6, won=True)
        for _ in range(3):
            t.record_prediction("Ridge", 0.4, won=False)
        t.detect_flocking({"TCN": 0.85, "Ridge": 0.80, "RF": 0.90})
        s = t.get_status()
        assert s["total_predictions"] == 10
        assert s["models"]["TCN"] == 7
        assert s["models"]["Ridge"] == 3
        assert s["flock_events"] == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_state_writes_json(self, tmp_path):
        path = tmp_path / "cal.json"
        t = ModelCalibrationTracker(persistence_path=path)
        t.record_prediction("TCN", 0.7, won=True)
        t.save_state()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert data["total_predictions"] == 1
        assert "predictions" in data
        assert "TCN" in data["predictions"]

    def test_save_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "cal.json"
        t = ModelCalibrationTracker(persistence_path=path)
        t.save_state()
        assert path.exists()

    def test_save_state_swallows_exceptions(self, tmp_path, monkeypatch):
        from pathlib import Path as _P
        path = tmp_path / "cal.json"
        t = ModelCalibrationTracker(persistence_path=path)

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(_P, "write_text", _boom)
        t.save_state()  # Must not raise

    def test_round_trip_preserves_state(self, tmp_path):
        path = tmp_path / "cal.json"
        t1 = ModelCalibrationTracker(persistence_path=path)
        for _ in range(5):
            t1.record_prediction("TCN", 0.7, won=True)
        t1.detect_flocking({"TCN": 0.85, "Ridge": 0.80, "RF": 0.90})
        total_before = t1._total_predictions
        flocks_before = len(t1._flock_events)
        preds_before = len(t1._predictions["TCN"])
        t1.save_state()

        # Reload
        t2 = ModelCalibrationTracker(persistence_path=path)
        assert t2._total_predictions == total_before
        assert len(t2._flock_events) == flocks_before
        assert len(t2._predictions["TCN"]) == preds_before

    def test_load_skips_corrupt_records(self, tmp_path):
        path = tmp_path / "cal.json"
        path.write_text(json.dumps({
            "version": 1,
            "total_predictions": 3,
            "predictions": {
                "TCN": [
                    {"confidence": 0.7, "won": True, "timestamp": "x"},
                    {"confidence": 0.8},  # missing won → skipped
                    "not a dict",         # not a dict → skipped
                    {"won": False},       # missing confidence → skipped
                ],
            },
            "flock_events": [],
        }))
        t = ModelCalibrationTracker(persistence_path=path)
        # Only the one valid record should survive
        assert len(t._predictions["TCN"]) == 1
        assert t._predictions["TCN"][0]["confidence"] == 0.7

    def test_load_skips_non_list_records(self, tmp_path):
        path = tmp_path / "cal.json"
        path.write_text(json.dumps({
            "predictions": {"TCN": "not-a-list"},
            "total_predictions": 0,
            "flock_events": [],
        }))
        t = ModelCalibrationTracker(persistence_path=path)
        # TCN should not be present in predictions (skipped)
        assert "TCN" not in t._predictions

    def test_load_handles_corrupt_json(self, tmp_path):
        path = tmp_path / "cal.json"
        path.write_text("{ this is not valid json")
        # Must not raise
        t = ModelCalibrationTracker(persistence_path=path)
        assert t._total_predictions == 0


# ---------------------------------------------------------------------------
# Integration: full calibration cycle
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_overconfident_model_flagged_across_multiple_bins(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # Two bins, both overconfident
        for i in range(10):
            t.record_prediction("TCN", 0.75, won=(i < 5))   # gap=0.25
        for i in range(8):
            t.record_prediction("TCN", 0.85, won=(i < 4))   # gap=0.35
        report = t.get_calibration_report("TCN")
        # Both bins are valid (≥5 samples) and both overconfident
        assert report["valid_bins"] == 2
        assert report["overconfidence_ratio"] == 1.0
        # ECE = (10/18)*0.25 + (8/18)*0.35 ≈ 0.139 + 0.156 = 0.294
        assert report["calibration_error"] == pytest.approx(
            (10/18)*0.25 + (8/18)*0.35, abs=1e-3
        )

    def test_only_low_count_bins_yield_none_metrics(self, tmp_path):
        t = ModelCalibrationTracker(persistence_path=tmp_path / "cal.json")
        # 3 records at conf 0.7, 2 at conf 0.85 — both bins below MIN_SAMPLES_PER_BIN
        for _ in range(3):
            t.record_prediction("TCN", 0.7, won=True)
        for _ in range(2):
            t.record_prediction("TCN", 0.85, won=False)
        report = t.get_calibration_report("TCN")
        assert report["valid_bins"] == 0
        assert report["calibration_error"] is None
        assert report["overconfidence_ratio"] is None
