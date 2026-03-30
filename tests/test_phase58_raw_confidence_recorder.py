"""Tests for Phase 58 US-358: RawConfidenceRecorder.

Covers:
  - record + get_distribution returns correct stats
  - deque max_size eviction
  - get_recent returns correct slice
  - save_state / _load_state round-trip
  - empty recorder returns safe defaults
  - get_distribution_by_pair filters correctly
  - single-entry distribution
"""

import json
from pathlib import Path

import pytest

from src.scanner.raw_confidence_recorder import RawConfidenceRecorder


class TestRecord:

    def test_empty_distribution_safe_defaults(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        dist = r.get_distribution()
        assert dist["count"] == 0
        assert dist["mean"] == 0.0
        assert dist["median"] == 0.0

    def test_record_increases_count(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        r.record("EUR_USD", 62.5, scan_id="s1")
        r.record("GBP_USD", 58.0, scan_id="s1")
        assert r.count() == 2

    def test_distribution_mean_median(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        for val in [50.0, 55.0, 60.0, 65.0, 70.0]:
            r.record("EUR_USD", val, scan_id="s1")
        dist = r.get_distribution()
        assert dist["count"] == 5
        assert abs(dist["mean"] - 60.0) < 0.01
        assert abs(dist["median"] - 60.0) < 0.01

    def test_distribution_percentiles(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        for val in range(0, 100):  # 100 values 0..99
            r.record("EUR_USD", float(val), scan_id="s1")
        dist = r.get_distribution()
        assert dist["p25"] == pytest.approx(24.75, abs=0.5)
        assert dist["p75"] == pytest.approx(74.25, abs=0.5)
        assert dist["p90"] == pytest.approx(89.1, abs=0.5)

    def test_single_entry_distribution(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        r.record("EUR_USD", 57.3, scan_id="s1")
        dist = r.get_distribution()
        assert dist["count"] == 1
        assert dist["mean"] == pytest.approx(57.3, abs=0.01)
        assert dist["min"] == pytest.approx(57.3, abs=0.01)
        assert dist["max"] == pytest.approx(57.3, abs=0.01)


class TestMaxSizeEviction:

    def test_max_size_evicts_oldest(self, tmp_path):
        r = RawConfidenceRecorder(max_size=3, state_path=tmp_path / "rc.jsonl")
        r.record("EUR_USD", 10.0, scan_id="s1")
        r.record("EUR_USD", 20.0, scan_id="s2")
        r.record("EUR_USD", 30.0, scan_id="s3")
        r.record("EUR_USD", 40.0, scan_id="s4")  # evicts 10.0
        assert r.count() == 3
        dist = r.get_distribution()
        assert dist["min"] == pytest.approx(20.0, abs=0.01)

    def test_max_size_respected_after_load(self, tmp_path):
        p = tmp_path / "rc.jsonl"
        r1 = RawConfidenceRecorder(max_size=5, state_path=p)
        for i in range(10):
            r1.record("EUR_USD", float(i), scan_id="s1")
        r1.save_state()
        r2 = RawConfidenceRecorder(max_size=5, state_path=p)
        assert r2.count() == 5


class TestGetRecent:

    def test_get_recent_returns_n_latest(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        for i in range(10):
            r.record("EUR_USD", float(i * 5), scan_id=f"s{i}")
        recent = r.get_recent(3)
        assert len(recent) == 3
        # Most recent should be last recorded (45.0)
        assert recent[-1]["raw_confidence"] == pytest.approx(45.0, abs=0.01)

    def test_get_recent_with_fewer_records(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        r.record("EUR_USD", 55.0, scan_id="s1")
        recent = r.get_recent(20)
        assert len(recent) == 1


class TestPersistence:

    def test_save_load_round_trip(self, tmp_path):
        p = tmp_path / "rc.jsonl"
        r1 = RawConfidenceRecorder(state_path=p)
        r1.record("EUR_USD", 62.0, scan_id="s1")
        r1.record("GBP_USD", 55.5, scan_id="s2")
        r1.save_state()

        r2 = RawConfidenceRecorder(state_path=p)
        assert r2.count() == 2
        dist = r2.get_distribution()
        assert dist["count"] == 2
        assert dist["min"] == pytest.approx(55.5, abs=0.01)
        assert dist["max"] == pytest.approx(62.0, abs=0.01)

    def test_save_no_tmp_file_lingers(self, tmp_path):
        p = tmp_path / "rc.jsonl"
        r = RawConfidenceRecorder(state_path=p)
        r.record("EUR_USD", 60.0, scan_id="s1")
        r.save_state()
        assert not (tmp_path / "rc.jsonl.tmp").exists()
        assert p.exists()

    def test_save_creates_valid_jsonl(self, tmp_path):
        p = tmp_path / "rc.jsonl"
        r = RawConfidenceRecorder(state_path=p)
        r.record("EUR_USD", 61.0, scan_id="s1")
        r.save_state()
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["pair"] == "EUR_USD"
        assert d["raw_confidence"] == pytest.approx(61.0, abs=0.01)

    def test_load_missing_file_no_error(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "nonexistent.jsonl")
        assert r.count() == 0

    def test_save_custom_path(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "default.jsonl")
        r.record("EUR_USD", 57.0, scan_id="s1")
        custom = tmp_path / "custom.jsonl"
        r.save_state(path=custom)
        assert custom.exists()


class TestByPair:

    def test_get_distribution_by_pair_filters(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        r.record("EUR_USD", 60.0, scan_id="s1")
        r.record("EUR_USD", 62.0, scan_id="s1")
        r.record("GBP_USD", 50.0, scan_id="s1")
        dist = r.get_distribution_by_pair("EUR_USD")
        assert dist["count"] == 2
        assert dist["mean"] == pytest.approx(61.0, abs=0.01)

    def test_get_distribution_by_pair_empty(self, tmp_path):
        r = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        dist = r.get_distribution_by_pair("EUR_USD")
        assert dist["count"] == 0
