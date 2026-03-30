"""Tests for Phase 56 US-347: Confidence Penalty Monitor.

Covers:
  - record_penalties() stores penalty sources per pair
  - get_total_penalty() returns correct sum
  - is_floor_breach() returns True/False at threshold
  - is_floor_breach() respects custom threshold override
  - generate_report() has required keys for all pairs
  - floor_breach flag in report is correct
  - get_breach_rate() returns correct fraction
  - flush_scan() resets current scan and increments counters
  - Multiple sources accumulate correctly
  - Persistence: save_state / _load_state round-trip
  - Corrupt state handled gracefully
  - Atomic write: no .tmp file lingers
  - Wiring: ConfidencePenaltyMonitor initialized in engine.py
  - Wiring: record_penalties called in engine.py
"""

import json
import os
from pathlib import Path

import pytest

from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monitor(tmp_path):
    return ConfidencePenaltyMonitor(
        persistence_path=tmp_path / "cpm.json",
        breach_threshold=0.20,
    )


# ---------------------------------------------------------------------------
# 1. record_penalties / get_total_penalty
# ---------------------------------------------------------------------------

class TestRecordPenalties:

    def test_single_source_penalty_stored(self, monitor):
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.08})
        assert monitor.get_total_penalty("EUR_USD") == pytest.approx(0.08, abs=0.001)

    def test_multiple_sources_sum_correctly(self, monitor):
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.08, "drift": 0.06, "disagreement": 0.04})
        assert monitor.get_total_penalty("EUR_USD") == pytest.approx(0.18, abs=0.001)

    def test_unknown_pair_returns_zero(self, monitor):
        assert monitor.get_total_penalty("UNKNOWN_PAIR") == 0.0

    def test_second_call_merges_sources(self, monitor):
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.08})
        monitor.record_penalties("EUR_USD", {"drift": 0.05})
        total = monitor.get_total_penalty("EUR_USD")
        assert total == pytest.approx(0.13, abs=0.001)

    def test_negative_penalties_floored_to_zero(self, monitor):
        """Negative values should not reduce the cumulative penalty."""
        monitor.record_penalties("EUR_USD", {"bad_source": -0.10})
        assert monitor.get_total_penalty("EUR_USD") == 0.0


# ---------------------------------------------------------------------------
# 2. is_floor_breach
# ---------------------------------------------------------------------------

class TestFloorBreach:

    def test_no_breach_below_threshold(self, monitor):
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.10})
        assert monitor.is_floor_breach("EUR_USD") is False

    def test_breach_at_threshold(self, monitor):
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.20})
        assert monitor.is_floor_breach("EUR_USD") is True

    def test_breach_above_threshold(self, monitor):
        monitor.record_penalties("EUR_USD", {"a": 0.12, "b": 0.12})
        assert monitor.is_floor_breach("EUR_USD") is True

    def test_custom_threshold_override(self, monitor):
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.15})
        # Default threshold = 0.20, custom = 0.10
        assert monitor.is_floor_breach("EUR_USD", threshold=0.10) is True
        assert monitor.is_floor_breach("EUR_USD", threshold=0.20) is False

    def test_unknown_pair_never_breaches(self, monitor):
        assert monitor.is_floor_breach("NO_PAIR") is False


# ---------------------------------------------------------------------------
# 3. generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def test_report_has_all_pairs(self, monitor):
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.08})
        monitor.record_penalties("GBP_USD", {"drift": 0.12})
        report = monitor.generate_report()
        assert "EUR_USD" in report
        assert "GBP_USD" in report

    def test_report_has_required_keys(self, monitor):
        monitor.record_penalties("EUR_USD", {"overconfidence": 0.08})
        report = monitor.generate_report()
        pair_data = report["EUR_USD"]
        for key in ("total_penalty", "sources", "floor_breach"):
            assert key in pair_data, f"Missing key: {key}"

    def test_report_floor_breach_flag_correct(self, monitor):
        monitor.record_penalties("EUR_USD", {"a": 0.25})  # above 0.20 threshold
        monitor.record_penalties("GBP_USD", {"b": 0.05})  # below threshold
        report = monitor.generate_report()
        assert report["EUR_USD"]["floor_breach"] is True
        assert report["GBP_USD"]["floor_breach"] is False

    def test_empty_report_when_no_penalties(self, monitor):
        assert monitor.generate_report() == {}


# ---------------------------------------------------------------------------
# 4. flush_scan and get_breach_rate
# ---------------------------------------------------------------------------

class TestFlushScanAndBreachRate:

    def test_flush_resets_current_scan(self, monitor):
        monitor.record_penalties("EUR_USD", {"a": 0.25})
        monitor.flush_scan()
        assert monitor.get_total_penalty("EUR_USD") == 0.0

    def test_flush_increments_scan_counter(self, monitor):
        monitor.flush_scan()
        monitor.flush_scan()
        assert monitor._total_scans == 2

    def test_flush_increments_breach_counter_when_breach(self, monitor):
        monitor.record_penalties("EUR_USD", {"a": 0.25})  # breach
        monitor.flush_scan()
        assert monitor._total_breaches == 1

    def test_flush_no_increment_when_no_breach(self, monitor):
        monitor.record_penalties("EUR_USD", {"a": 0.05})  # no breach
        monitor.flush_scan()
        assert monitor._total_breaches == 0

    def test_breach_rate_correct(self, monitor):
        # 2 scans with breach, 1 scan without
        monitor.record_penalties("EUR_USD", {"a": 0.25})
        monitor.flush_scan()
        monitor.record_penalties("EUR_USD", {"a": 0.05})
        monitor.flush_scan()
        monitor.record_penalties("EUR_USD", {"a": 0.25})
        monitor.flush_scan()
        assert monitor.get_breach_rate() == pytest.approx(2/3, abs=0.01)

    def test_breach_rate_zero_when_no_scans(self, monitor):
        assert monitor.get_breach_rate() == 0.0


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_state_survives_save_reload(self, tmp_path):
        path = tmp_path / "state.json"
        m1 = ConfidencePenaltyMonitor(persistence_path=path, breach_threshold=0.20)
        m1.record_penalties("EUR_USD", {"a": 0.25})
        m1.flush_scan()  # 1 breach
        m1.save_state()

        m2 = ConfidencePenaltyMonitor(persistence_path=path)
        assert m2._total_scans == 1
        assert m2._total_breaches == 1

    def test_corrupt_state_handled_gracefully(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("NOT JSON {{{")
        monitor = ConfidencePenaltyMonitor(persistence_path=path)
        assert monitor._total_scans == 0

    def test_atomic_write_no_tmp_lingers(self, tmp_path):
        path = tmp_path / "atomic.json"
        monitor = ConfidencePenaltyMonitor(persistence_path=path)
        monitor._total_scans = 5
        monitor.save_state()
        assert path.exists()
        assert not Path(str(path) + ".tmp").exists()
        data = json.loads(path.read_text())
        assert data["total_scans"] == 5


# ---------------------------------------------------------------------------
# 6. Wiring verification
# ---------------------------------------------------------------------------

class TestWiringVerification:

    def test_confidence_penalty_monitor_initialized_in_engine(self):
        with open("src/scanner/engine.py") as f:
            content = f.read()
        assert "ConfidencePenaltyMonitor" in content
        assert "_confidence_penalty_monitor = None" in content

    def test_record_penalties_called_in_engine(self):
        with open("src/scanner/engine.py") as f:
            content = f.read()
        assert "_confidence_penalty_monitor.record_penalties(" in content
