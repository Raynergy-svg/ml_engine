"""Phase 67 US-391 — SubInferenceRecorder unit tests.

Tests cover:
- record() stores snapshots with consensus_ratio
- get_distribution() returns correct stats
- get_fail_distribution() filters to agent_passed=False
- get_pass_rate() computes correctly
- save_state() / _load_state() round-trip (JSONL)
- Edge cases: empty buffer, all pass, all fail, zero total
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.scanner.sub_inference_recorder import SubInferenceRecorder


def _make_recorder(**kwargs) -> tuple[SubInferenceRecorder, Path]:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    path.unlink()
    return SubInferenceRecorder(state_path=path, **kwargs), path


class TestSubInferenceRecorder_Record:
    def test_record_adds_entry(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", votes=1, total=3, agent_passed=False)
        assert rec.get_distribution()["count"] == 1

    def test_consensus_ratio_computed(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", votes=2, total=3, agent_passed=True)
        dist = rec.get_distribution()
        assert dist["mean"] == pytest.approx(0.6667, abs=1e-3)

    def test_record_respects_max_size(self):
        rec, _ = _make_recorder(max_size=5)
        for i in range(10):
            rec.record(pair=f"P{i}", votes=1, total=3, agent_passed=False)
        assert rec.get_distribution()["count"] == 5

    def test_zero_total_safe(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", votes=0, total=0, agent_passed=False)
        dist = rec.get_distribution()
        assert dist["count"] == 1
        assert dist["mean"] == 0.0


class TestSubInferenceRecorder_Distribution:
    def test_empty_distribution(self):
        rec, _ = _make_recorder()
        assert rec.get_distribution()["count"] == 0

    def test_fail_distribution_filters(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", votes=1, total=3, agent_passed=False)
        rec.record(pair="GBP_USD", votes=2, total=3, agent_passed=True)
        fail_dist = rec.get_fail_distribution()
        assert fail_dist["count"] == 1

    def test_pass_rate(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", votes=2, total=3, agent_passed=True)
        rec.record(pair="GBP_USD", votes=0, total=3, agent_passed=False)
        assert rec.get_pass_rate() == pytest.approx(0.5)


class TestSubInferenceRecorder_Persistence:
    def test_save_load_roundtrip(self):
        rec1, path = _make_recorder()
        rec1.record(pair="EUR_USD", votes=1, total=3, agent_passed=False)
        rec1.record(pair="GBP_USD", votes=2, total=3, agent_passed=True)
        rec1.save_state()

        rec2 = SubInferenceRecorder(state_path=path)
        assert rec2.get_distribution()["count"] == 2
        path.unlink(missing_ok=True)

    def test_save_writes_valid_jsonl(self):
        rec, path = _make_recorder()
        rec.record(pair="EUR_USD", votes=1, total=3, agent_passed=False)
        rec.save_state()

        with open(path) as f:
            parsed = json.loads(f.readline())
        assert "consensus_ratio" in parsed
        assert "votes" in parsed
        assert "total" in parsed
        path.unlink(missing_ok=True)

    def test_importable(self):
        from src.scanner.sub_inference_recorder import SubInferenceRecorder
        assert SubInferenceRecorder is not None
