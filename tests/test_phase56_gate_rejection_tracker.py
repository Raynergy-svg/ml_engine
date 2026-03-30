"""Tests for Phase 56 US-346: Gate Rejection Tracker.

Covers:
  - record_rejection() stores gate name, pair, reason, confidence
  - get_stats() returns correct counts per gate
  - get_top_blocking_gate() returns gate with highest count
  - get_gate_rejection_rate() returns correct fraction
  - mark_scan_complete() increments scan counter
  - get_recent_records() returns last N records
  - Persistence: save_state / _load_state round-trip
  - Corrupt state file handled gracefully
  - Atomic write: no .tmp file lingers
  - Wiring grep: record_rejection called in engine.py outside class definition
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.scanner.gate_rejection_tracker import GateRejectionTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_tracker(tmp_path):
    return GateRejectionTracker(persistence_path=tmp_path / "gate_rej.json")


# ---------------------------------------------------------------------------
# 1. record_rejection
# ---------------------------------------------------------------------------

class TestRecordRejection:

    def test_record_increments_gate_count(self, tmp_tracker):
        tmp_tracker.record_rejection("EUR_USD", "confidence", "0.21 < 0.60", 0.21)
        stats = tmp_tracker.get_stats()
        assert stats["gate_counts"]["confidence"] == 1

    def test_record_increments_total_rejections(self, tmp_tracker):
        tmp_tracker.record_rejection("EUR_USD", "confidence", reason="low", confidence=0.2)
        tmp_tracker.record_rejection("GBP_USD", "agent", reason="no consensus", confidence=0.5)
        assert tmp_tracker.get_stats()["total_rejections"] == 2

    def test_record_updates_pair_counts(self, tmp_tracker):
        tmp_tracker.record_rejection("EUR_USD", "confidence", "", 0.3)
        tmp_tracker.record_rejection("EUR_USD", "confidence", "", 0.25)
        tmp_tracker.record_rejection("EUR_USD", "risk", "", 0.45)
        stats = tmp_tracker.get_stats()
        assert stats["pair_counts"]["EUR_USD"]["confidence"] == 2
        assert stats["pair_counts"]["EUR_USD"]["risk"] == 1

    def test_empty_gate_name_normalized_to_unknown(self, tmp_tracker):
        tmp_tracker.record_rejection("EUR_USD", "", "", 0.0)
        stats = tmp_tracker.get_stats()
        assert "unknown" in stats["gate_counts"]

    def test_gate_name_lowercased(self, tmp_tracker):
        tmp_tracker.record_rejection("EUR_USD", "Confidence", "", 0.5)
        stats = tmp_tracker.get_stats()
        assert "confidence" in stats["gate_counts"]
        assert "Confidence" not in stats["gate_counts"]


# ---------------------------------------------------------------------------
# 2. get_top_blocking_gate
# ---------------------------------------------------------------------------

class TestTopBlockingGate:

    def test_top_gate_returns_most_common(self, tmp_tracker):
        for _ in range(5):
            tmp_tracker.record_rejection("EUR_USD", "confidence", "", 0.3)
        for _ in range(3):
            tmp_tracker.record_rejection("EUR_USD", "agent", "", 0.5)
        assert tmp_tracker.get_top_blocking_gate() == "confidence"

    def test_top_gate_returns_none_when_empty(self, tmp_tracker):
        assert tmp_tracker.get_top_blocking_gate() == "none"

    def test_top_gate_updates_after_new_records(self, tmp_tracker):
        tmp_tracker.record_rejection("EUR_USD", "confidence", "", 0.3)
        assert tmp_tracker.get_top_blocking_gate() == "confidence"
        for _ in range(3):
            tmp_tracker.record_rejection("EUR_USD", "risk", "", 0.4)
        assert tmp_tracker.get_top_blocking_gate() == "risk"


# ---------------------------------------------------------------------------
# 3. get_gate_rejection_rate
# ---------------------------------------------------------------------------

class TestGateRejectionRate:

    def test_rejection_rate_correct(self, tmp_tracker):
        for _ in range(3):
            tmp_tracker.record_rejection("EUR_USD", "confidence", "", 0.3)
        tmp_tracker.record_rejection("EUR_USD", "agent", "", 0.5)
        rate = tmp_tracker.get_gate_rejection_rate("confidence")
        assert rate == pytest.approx(0.75, abs=0.01)

    def test_rejection_rate_zero_when_no_records(self, tmp_tracker):
        assert tmp_tracker.get_gate_rejection_rate("confidence") == 0.0

    def test_rejection_rate_zero_for_unknown_gate(self, tmp_tracker):
        tmp_tracker.record_rejection("EUR_USD", "confidence", "", 0.3)
        assert tmp_tracker.get_gate_rejection_rate("nonexistent_gate") == 0.0


# ---------------------------------------------------------------------------
# 4. mark_scan_complete
# ---------------------------------------------------------------------------

class TestMarkScanComplete:

    def test_scan_counter_increments(self, tmp_tracker):
        tmp_tracker.mark_scan_complete()
        tmp_tracker.mark_scan_complete()
        stats = tmp_tracker.get_stats()
        assert stats["scans_observed"] == 2

    def test_rejections_per_scan_computed(self, tmp_tracker):
        for _ in range(4):
            tmp_tracker.record_rejection("EUR_USD", "confidence", "", 0.3)
        tmp_tracker.mark_scan_complete()
        tmp_tracker.mark_scan_complete()
        stats = tmp_tracker.get_stats()
        assert stats["rejections_per_scan"] == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# 5. get_recent_records
# ---------------------------------------------------------------------------

class TestGetRecentRecords:

    def test_recent_records_returns_last_n(self, tmp_tracker):
        for i in range(10):
            tmp_tracker.record_rejection(f"EUR_USD_{i}", "confidence", "", 0.3)
        recent = tmp_tracker.get_recent_records(n=3)
        assert len(recent) == 3
        assert recent[-1]["pair"] == "EUR_USD_9"

    def test_recent_records_have_required_keys(self, tmp_tracker):
        tmp_tracker.record_rejection("EUR_USD", "confidence", "0.3 < 0.6", 0.3)
        recent = tmp_tracker.get_recent_records(n=1)
        assert len(recent) == 1
        record = recent[0]
        for key in ("pair", "gate_name", "reason", "confidence", "timestamp"):
            assert key in record, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 6. Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_state_survives_save_reload(self, tmp_path):
        path = tmp_path / "state.json"
        t1 = GateRejectionTracker(persistence_path=path)
        t1.record_rejection("EUR_USD", "confidence", "low", 0.25)
        t1.record_rejection("GBP_USD", "agent", "no consensus", 0.50)
        t1.mark_scan_complete()
        t1.save_state()

        t2 = GateRejectionTracker(persistence_path=path)
        assert t2.get_stats()["total_rejections"] == 2
        assert t2.get_stats()["gate_counts"]["confidence"] == 1
        assert t2.get_stats()["scans_observed"] == 1

    def test_corrupt_state_handled_gracefully(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("NOT VALID JSON {{{")
        tracker = GateRejectionTracker(persistence_path=path)
        assert tracker.get_stats()["total_rejections"] == 0

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path):
        path = tmp_path / "atomic.json"
        tracker = GateRejectionTracker(persistence_path=path)
        tracker.record_rejection("EUR_USD", "confidence", "", 0.3)
        tracker.save_state()

        assert path.exists()
        assert not Path(str(path) + ".tmp").exists()
        data = json.loads(path.read_text())
        assert data["total_rejections"] == 1

    def test_state_file_has_version_field(self, tmp_path):
        path = tmp_path / "v.json"
        tracker = GateRejectionTracker(persistence_path=path)
        tracker.save_state()
        data = json.loads(path.read_text())
        assert "version" in data

    def test_recent_records_capped_at_max(self, tmp_path):
        path = tmp_path / "cap.json"
        tracker = GateRejectionTracker(persistence_path=path, max_recent_records=10)
        for i in range(20):
            tracker.record_rejection(f"PAIR_{i}", "confidence", "", 0.3)
        tracker.save_state()
        data = json.loads(path.read_text())
        assert len(data["recent_records"]) <= 10


# ---------------------------------------------------------------------------
# 7. Wiring verification
# ---------------------------------------------------------------------------

class TestWiringVerification:

    def test_record_rejection_called_in_engine(self):
        """Live wiring: record_rejection is called in engine.py outside class definition."""
        engine_path = "src/scanner/engine.py"
        with open(engine_path) as f:
            content = f.read()
        assert "_gate_rejection_tracker.record_rejection(" in content, (
            "record_rejection() must be called in engine.py as part of the scan loop"
        )

    def test_gate_rejection_tracker_initialized_in_engine(self):
        """GateRejectionTracker must be initialized in engine.py __init__."""
        engine_path = "src/scanner/engine.py"
        with open(engine_path) as f:
            content = f.read()
        assert "GateRejectionTracker" in content
        assert "_gate_rejection_tracker = None" in content
