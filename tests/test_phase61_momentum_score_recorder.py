"""Phase 61 US-371 — MomentumScoreRecorder unit tests.

Minimum 8 tests per acceptance criteria.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.scanner.momentum_score_recorder import MomentumScoreRecorder


# ── helpers ─────────────────────────────────────────────────────────────────


def _tmp_recorder(max_size: int = 500) -> tuple[MomentumScoreRecorder, Path]:
    """Return a recorder backed by a temp file that will be cleaned up."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    path.unlink()  # start empty
    return MomentumScoreRecorder(max_size=max_size, state_path=path), path


# ── US-371 unit tests ────────────────────────────────────────────────────────


class TestMomentumScoreRecorder_Record:
    """record() stores snapshots with correct fields."""

    def test_record_stores_snapshot(self):
        rec, path = _tmp_recorder()
        rec.record("EUR/USD", momentum_score=0.42, momentum_passed=False)
        dist = rec.get_distribution()
        assert dist["count"] == 1
        assert dist["min"] == pytest.approx(0.42, abs=1e-5)

    def test_record_momentum_passed_field(self):
        rec, path = _tmp_recorder()
        rec.record("GBP/USD", momentum_score=0.55, momentum_passed=True)
        rec.record("EUR/USD", momentum_score=0.35, momentum_passed=False)
        pass_rate = rec.get_pass_rate()
        assert pass_rate == pytest.approx(0.5, abs=1e-5)

    def test_record_multiple_pairs(self):
        rec, path = _tmp_recorder()
        pairs = ["EUR/USD", "GBP/USD", "AUD/USD"]
        for i, pair in enumerate(pairs):
            rec.record(pair, momentum_score=0.30 + i * 0.10, momentum_passed=True)
        assert rec.get_distribution()["count"] == 3

    def test_record_with_scan_id(self):
        rec, path = _tmp_recorder()
        rec.record("EUR/USD", momentum_score=0.45, momentum_passed=True, scan_id="scan_42")
        # No exception; snapshot stored (verified via distribution count)
        assert rec.get_distribution()["count"] == 1


class TestMomentumScoreRecorder_Distribution:
    """Distribution statistics computed correctly."""

    def test_distribution_p_values(self):
        """p25/p50/p75 percentiles computed from sorted list."""
        rec, path = _tmp_recorder()
        # 10 evenly spaced scores: 0.1, 0.2, ..., 1.0
        for i in range(1, 11):
            rec.record("EUR/USD", momentum_score=round(i * 0.1, 1), momentum_passed=True)
        dist = rec.get_distribution()
        assert dist["count"] == 10
        # p50 of [0.1..1.0] should be between 0.5 and 0.6
        assert 0.50 <= dist["p50"] <= 0.60
        assert dist["min"] == pytest.approx(0.1, abs=1e-5)
        assert dist["max"] == pytest.approx(1.0, abs=1e-5)

    def test_distribution_empty_safe_defaults(self):
        """Empty recorder returns all-zero distribution, no crash."""
        rec, path = _tmp_recorder()
        dist = rec.get_distribution()
        assert dist["count"] == 0
        assert dist["mean"] == 0.0
        assert dist["p50"] == 0.0
        assert dist["min"] == 0.0

    def test_fail_distribution_filters_correctly(self):
        """get_fail_distribution() returns only momentum_passed=False records."""
        rec, path = _tmp_recorder()
        rec.record("EUR/USD", momentum_score=0.60, momentum_passed=True)
        rec.record("EUR/USD", momentum_score=0.38, momentum_passed=False)
        rec.record("GBP/USD", momentum_score=0.40, momentum_passed=False)
        fail_dist = rec.get_fail_distribution()
        assert fail_dist["count"] == 2
        assert fail_dist["max"] == pytest.approx(0.40, abs=1e-5)

    def test_fail_distribution_empty_when_all_pass(self):
        """get_fail_distribution() returns zero dict when no fails."""
        rec, path = _tmp_recorder()
        rec.record("EUR/USD", momentum_score=0.70, momentum_passed=True)
        rec.record("GBP/USD", momentum_score=0.65, momentum_passed=True)
        fail_dist = rec.get_fail_distribution()
        assert fail_dist["count"] == 0
        assert fail_dist["p50"] == 0.0


class TestMomentumScoreRecorder_PassRate:
    """get_pass_rate() correctness."""

    def test_pass_rate_correct(self):
        """7 pass + 3 fail → pass_rate = 0.70."""
        rec, path = _tmp_recorder()
        for _ in range(7):
            rec.record("EUR/USD", 0.55, momentum_passed=True)
        for _ in range(3):
            rec.record("EUR/USD", 0.35, momentum_passed=False)
        assert rec.get_pass_rate() == pytest.approx(0.70, abs=1e-5)

    def test_pass_rate_empty_is_zero(self):
        rec, path = _tmp_recorder()
        assert rec.get_pass_rate() == 0.0


class TestMomentumScoreRecorder_Persistence:
    """save_state() / _load_state() round-trip."""

    def test_save_load_roundtrip(self):
        """All records survive a save → new instance load cycle."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        path.unlink()

        rec1 = MomentumScoreRecorder(state_path=path)
        for i in range(5):
            rec1.record("EUR/USD", momentum_score=round(0.30 + i * 0.05, 2), momentum_passed=(i % 2 == 0))
        rec1.save_state()

        rec2 = MomentumScoreRecorder(state_path=path)
        assert rec2.get_distribution()["count"] == 5
        assert rec2.get_pass_rate() == pytest.approx(3 / 5, abs=1e-5)
        path.unlink(missing_ok=True)

    def test_save_atomic_write(self):
        """save_state() produces a valid JSONL file (no tmp files left)."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        path.unlink()

        rec = MomentumScoreRecorder(state_path=path)
        rec.record("GBP/USD", 0.45, momentum_passed=True)
        rec.save_state()

        assert path.exists()
        with open(path) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        assert len(lines) == 1
        snap = json.loads(lines[0])
        assert snap["momentum_score"] == pytest.approx(0.45, abs=1e-5)
        path.unlink(missing_ok=True)

    def test_max_size_enforced(self):
        """Rolling buffer never exceeds max_size."""
        rec, path = _tmp_recorder(max_size=10)
        for i in range(20):
            rec.record("EUR/USD", momentum_score=round(i * 0.02, 4), momentum_passed=True)
        assert rec.get_distribution()["count"] == 10
