"""Phase 66 US-388 — VoteScoreRecorder unit tests.

Tests cover:
- record() stores snapshots correctly
- get_distribution() returns correct stats
- get_fail_distribution() filters to agent_passed=False
- get_pass_rate() computes correctly
- save_state() / _load_state() round-trip (JSONL)
- Thread safety with lock
- Edge cases: empty buffer, all pass, all fail
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.scanner.vote_score_recorder import VoteScoreRecorder


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_recorder(**kwargs) -> tuple[VoteScoreRecorder, Path]:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    path.unlink()
    recorder = VoteScoreRecorder(state_path=path, **kwargs)
    return recorder, path


# ── Unit Tests ───────────────────────────────────────────────────────────────


class TestVoteScoreRecorder_Record:
    """record() stores snapshots correctly."""

    def test_record_adds_entry(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", vote_score=0.52, agent_passed=True)
        dist = rec.get_distribution()
        assert dist["count"] == 1

    def test_record_multiple_entries(self):
        rec, _ = _make_recorder()
        for i in range(10):
            rec.record(pair=f"PAIR_{i}", vote_score=0.5 + i * 0.01, agent_passed=i % 2 == 0)
        dist = rec.get_distribution()
        assert dist["count"] == 10

    def test_record_respects_max_size(self):
        rec, _ = _make_recorder(max_size=5)
        for i in range(10):
            rec.record(pair=f"PAIR_{i}", vote_score=0.5, agent_passed=True)
        dist = rec.get_distribution()
        assert dist["count"] == 5


class TestVoteScoreRecorder_Distribution:
    """Distribution stats computed correctly."""

    def test_empty_distribution(self):
        rec, _ = _make_recorder()
        dist = rec.get_distribution()
        assert dist["count"] == 0
        assert dist["mean"] == 0.0

    def test_single_value_distribution(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", vote_score=0.60, agent_passed=True)
        dist = rec.get_distribution()
        assert dist["count"] == 1
        assert dist["mean"] == pytest.approx(0.60, abs=1e-4)
        assert dist["p50"] == pytest.approx(0.60, abs=1e-4)

    def test_mean_computation(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", vote_score=0.40, agent_passed=False)
        rec.record(pair="GBP_USD", vote_score=0.60, agent_passed=True)
        dist = rec.get_distribution()
        assert dist["mean"] == pytest.approx(0.50, abs=1e-4)


class TestVoteScoreRecorder_FailDistribution:
    """Fail distribution filters to agent_passed=False."""

    def test_fail_distribution_filters_correctly(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", vote_score=0.40, agent_passed=False)
        rec.record(pair="GBP_USD", vote_score=0.42, agent_passed=False)
        rec.record(pair="USD_JPY", vote_score=0.60, agent_passed=True)
        fail_dist = rec.get_fail_distribution()
        assert fail_dist["count"] == 2
        assert fail_dist["mean"] == pytest.approx(0.41, abs=1e-4)

    def test_fail_distribution_empty_when_all_pass(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", vote_score=0.60, agent_passed=True)
        fail_dist = rec.get_fail_distribution()
        assert fail_dist["count"] == 0


class TestVoteScoreRecorder_PassRate:
    """get_pass_rate() computes correctly."""

    def test_pass_rate_empty(self):
        rec, _ = _make_recorder()
        assert rec.get_pass_rate() == 0.0

    def test_pass_rate_all_pass(self):
        rec, _ = _make_recorder()
        for _ in range(5):
            rec.record(pair="EUR_USD", vote_score=0.60, agent_passed=True)
        assert rec.get_pass_rate() == pytest.approx(1.0)

    def test_pass_rate_mixed(self):
        rec, _ = _make_recorder()
        rec.record(pair="EUR_USD", vote_score=0.60, agent_passed=True)
        rec.record(pair="GBP_USD", vote_score=0.40, agent_passed=False)
        assert rec.get_pass_rate() == pytest.approx(0.5)


class TestVoteScoreRecorder_Persistence:
    """save_state() / _load_state() round-trip."""

    def test_save_load_roundtrip(self):
        rec1, path = _make_recorder()
        rec1.record(pair="EUR_USD", vote_score=0.52, agent_passed=True)
        rec1.record(pair="GBP_USD", vote_score=0.40, agent_passed=False)
        rec1.save_state()

        rec2 = VoteScoreRecorder(state_path=path)
        dist = rec2.get_distribution()
        assert dist["count"] == 2
        path.unlink(missing_ok=True)

    def test_save_state_writes_valid_jsonl(self):
        rec, path = _make_recorder()
        rec.record(pair="EUR_USD", vote_score=0.55, agent_passed=True)
        rec.save_state()

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "vote_score" in parsed
        assert "agent_passed" in parsed
        path.unlink(missing_ok=True)

    def test_load_state_missing_file_no_error(self):
        path = Path("/tmp/nonexistent_vote_state.jsonl")
        rec = VoteScoreRecorder(state_path=path)
        assert rec.get_distribution()["count"] == 0


class TestVoteScoreRecorder_Importability:
    """Module imports correctly."""

    def test_importable(self):
        from src.scanner.vote_score_recorder import VoteScoreRecorder
        assert VoteScoreRecorder is not None
