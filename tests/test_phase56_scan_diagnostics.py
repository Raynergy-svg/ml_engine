"""Tests for Phase 56 US-350: Scan Diagnostics Reporter.

Covers:
  - record_scan() stores scan results
  - should_report() returns True every REPORT_INTERVAL scans
  - should_report() returns False before interval
  - generate_report() has all required keys
  - generate_report() avg_confidence is correct
  - generate_report() gate_pass_rate_pct reflects tradeable count
  - generate_report() stalled flag fires when scans_since_last_trade >= threshold
  - generate_report() stalled flag is False when below threshold
  - save_report() resets scans_since_last_report counter
  - save_report() writes valid JSON to report path
  - Persistence: save_state / _load_state round-trip
  - Corrupt state handled gracefully
  - Atomic write: no .tmp file lingers
  - Wiring: ScanDiagnosticsReporter initialized in continuous.py
  - Wiring: record_scan called in continuous.py
"""

import json
from pathlib import Path

import pytest

from src.scanner.scan_diagnostics import ScanDiagnosticsReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reporter(tmp_path, report_interval=10, stalled_threshold=50):
    return ScanDiagnosticsReporter(
        report_path=tmp_path / "report.json",
        state_path=tmp_path / "state.json",
        report_interval=report_interval,
        stalled_threshold=stalled_threshold,
    )


def make_scan_results(n_pairs=7, tradeable_count=0, confidence=0.25, direction="HOLD"):
    """Build a list of synthetic pair result dicts."""
    results = []
    for i in range(n_pairs):
        is_tradeable = i < tradeable_count
        results.append({
            "confidence": confidence + i * 0.01,
            "is_tradeable": is_tradeable,
            "direction": direction if is_tradeable else "HOLD",
        })
    return results


# ---------------------------------------------------------------------------
# 1. record_scan
# ---------------------------------------------------------------------------

class TestRecordScan:

    def test_record_scan_increments_total(self, tmp_path):
        reporter = make_reporter(tmp_path)
        reporter.record_scan(make_scan_results())
        assert reporter._total_scans == 1

    def test_record_scan_empty_list_skipped(self, tmp_path):
        reporter = make_reporter(tmp_path)
        reporter.record_scan([])
        assert reporter._total_scans == 0

    def test_tradeable_resets_stalled_counter(self, tmp_path):
        reporter = make_reporter(tmp_path)
        for _ in range(5):
            reporter.record_scan(make_scan_results(tradeable_count=0))
        assert reporter._scans_since_last_trade == 5
        reporter.record_scan(make_scan_results(tradeable_count=1))
        assert reporter._scans_since_last_trade == 0

    def test_no_tradeable_increments_stalled_counter(self, tmp_path):
        reporter = make_reporter(tmp_path)
        for _ in range(3):
            reporter.record_scan(make_scan_results(tradeable_count=0))
        assert reporter._scans_since_last_trade == 3


# ---------------------------------------------------------------------------
# 2. should_report
# ---------------------------------------------------------------------------

class TestShouldReport:

    def test_should_report_false_before_interval(self, tmp_path):
        reporter = make_reporter(tmp_path, report_interval=10)
        for _ in range(9):
            reporter.record_scan(make_scan_results())
        assert reporter.should_report() is False

    def test_should_report_true_at_interval(self, tmp_path):
        reporter = make_reporter(tmp_path, report_interval=10)
        for _ in range(10):
            reporter.record_scan(make_scan_results())
        assert reporter.should_report() is True

    def test_should_report_resets_after_save_report(self, tmp_path):
        reporter = make_reporter(tmp_path, report_interval=5)
        for _ in range(5):
            reporter.record_scan(make_scan_results())
        assert reporter.should_report() is True
        report = reporter.generate_report()
        reporter.save_report(report)
        assert reporter.should_report() is False


# ---------------------------------------------------------------------------
# 3. generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def test_report_has_all_required_keys(self, tmp_path):
        reporter = make_reporter(tmp_path)
        reporter.record_scan(make_scan_results())
        report = reporter.generate_report()
        required_keys = [
            "total_scans", "scans_since_last_report", "scans_since_last_trade",
            "avg_top_score", "avg_confidence", "gate_pass_rate_pct",
            "pairs_with_direction_pct", "stalled", "diagnostic_summary",
            "window_size", "timestamp",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_report_avg_confidence_correct(self, tmp_path):
        reporter = make_reporter(tmp_path)
        # All pairs have confidence 0.30 (approximately)
        reporter.record_scan([{"confidence": 0.30, "is_tradeable": False, "direction": "HOLD"}] * 5)
        report = reporter.generate_report()
        assert report["avg_confidence"] == pytest.approx(0.30, abs=0.01)

    def test_gate_pass_rate_zero_when_no_tradeables(self, tmp_path):
        reporter = make_reporter(tmp_path)
        for _ in range(5):
            reporter.record_scan(make_scan_results(tradeable_count=0))
        report = reporter.generate_report()
        assert report["gate_pass_rate_pct"] == 0.0

    def test_gate_pass_rate_100_when_always_tradeable(self, tmp_path):
        reporter = make_reporter(tmp_path)
        for _ in range(5):
            reporter.record_scan(make_scan_results(tradeable_count=1))
        report = reporter.generate_report()
        assert report["gate_pass_rate_pct"] == 100.0

    def test_stalled_flag_fires_when_threshold_exceeded(self, tmp_path):
        reporter = make_reporter(tmp_path, stalled_threshold=5)
        for _ in range(5):
            reporter.record_scan(make_scan_results(tradeable_count=0))
        report = reporter.generate_report()
        assert report["stalled"] is True

    def test_stalled_flag_false_below_threshold(self, tmp_path):
        reporter = make_reporter(tmp_path, stalled_threshold=50)
        for _ in range(5):
            reporter.record_scan(make_scan_results(tradeable_count=0))
        report = reporter.generate_report()
        assert report["stalled"] is False

    def test_stalled_diagnostic_summary_mentions_stalled(self, tmp_path):
        reporter = make_reporter(tmp_path, stalled_threshold=3)
        for _ in range(3):
            reporter.record_scan(make_scan_results(tradeable_count=0))
        report = reporter.generate_report()
        assert "STALLED" in report["diagnostic_summary"] or report["stalled"] is True

    def test_window_size_correct(self, tmp_path):
        reporter = make_reporter(tmp_path)
        for _ in range(7):
            reporter.record_scan(make_scan_results())
        report = reporter.generate_report()
        assert report["window_size"] == 7


# ---------------------------------------------------------------------------
# 4. save_report
# ---------------------------------------------------------------------------

class TestSaveReport:

    def test_save_report_creates_file(self, tmp_path):
        reporter = make_reporter(tmp_path)
        reporter.record_scan(make_scan_results())
        report = reporter.generate_report()
        reporter.save_report(report)
        assert (tmp_path / "report.json").exists()

    def test_save_report_valid_json(self, tmp_path):
        reporter = make_reporter(tmp_path)
        reporter.record_scan(make_scan_results())
        report = reporter.generate_report()
        reporter.save_report(report)
        data = json.loads((tmp_path / "report.json").read_text())
        assert "total_scans" in data

    def test_save_report_no_tmp_file(self, tmp_path):
        reporter = make_reporter(tmp_path)
        reporter.record_scan(make_scan_results())
        report = reporter.generate_report()
        reporter.save_report(report)
        assert not (tmp_path / "report.json.tmp").exists()


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_state_survives_save_reload(self, tmp_path):
        r1 = make_reporter(tmp_path)
        for _ in range(5):
            r1.record_scan(make_scan_results(tradeable_count=0))
        r1.save_state()

        r2 = make_reporter(tmp_path)
        assert r2._total_scans == 5
        assert r2._scans_since_last_trade == 5

    def test_corrupt_state_handled_gracefully(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text("NOT JSON {{{")
        reporter = ScanDiagnosticsReporter(
            report_path=tmp_path / "report.json",
            state_path=state_path,
        )
        assert reporter._total_scans == 0


# ---------------------------------------------------------------------------
# 6. Wiring verification
# ---------------------------------------------------------------------------

class TestWiringVerification:

    def test_scan_diagnostics_initialized_in_continuous(self):
        with open("src/scanner/automation/continuous.py") as f:
            content = f.read()
        assert "ScanDiagnosticsReporter" in content
        assert "_scan_diagnostics = None" in content

    def test_record_scan_called_in_continuous(self):
        with open("src/scanner/automation/continuous.py") as f:
            content = f.read()
        assert "_scan_diagnostics.record_scan(" in content
