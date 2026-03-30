"""Phase 66 US-389 — VoteGapAnalyzer unit tests.

Tests cover:
- NEAR_THRESHOLD classification (gap <= 0.05)
- MODERATE classification (0.05 < gap <= 0.15)
- FAR_BELOW classification (gap > 0.15)
- INSUFFICIENT_DATA when < 3 fail records
- near_miss_count and near_miss_rate_pct accuracy
- Empty recorder returns INSUFFICIENT_DATA
- VoteGapAnalyzer wrapper class methods
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.scanner.vote_score_recorder import VoteScoreRecorder
from src.scanner.vote_gap_analyzer import (
    analyze,
    VoteGapAnalyzer,
    CLASS_NEAR_THRESHOLD,
    CLASS_MODERATE,
    CLASS_FAR_BELOW,
    CLASS_INSUFFICIENT_DATA,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_recorder(**kwargs) -> VoteScoreRecorder:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    path.unlink()
    return VoteScoreRecorder(state_path=path, **kwargs)


def _populate_fails(rec: VoteScoreRecorder, scores: list[float], threshold: float = 0.55):
    """Record failed entries with given scores."""
    for i, s in enumerate(scores):
        rec.record(pair=f"PAIR_{i}", vote_score=s, agent_passed=False)


def _populate_mixed(rec: VoteScoreRecorder, pass_scores: list[float], fail_scores: list[float]):
    for s in pass_scores:
        rec.record(pair="PASS", vote_score=s, agent_passed=True)
    for s in fail_scores:
        rec.record(pair="FAIL", vote_score=s, agent_passed=False)


# ── Classification Tests ─────────────────────────────────────────────────────


class TestVoteGapAnalyzer_Classification:
    """Correct classification based on gap_at_p50."""

    def test_near_threshold_classification(self):
        """Fail scores close to threshold → NEAR_THRESHOLD."""
        rec = _make_recorder()
        # threshold=0.55, fails at 0.52, 0.53, 0.51 → p50=0.52, gap=0.03 <= 0.05
        _populate_fails(rec, [0.52, 0.53, 0.51])
        result = analyze(rec, weighted_vote_threshold=0.55)
        assert result["classification"] == CLASS_NEAR_THRESHOLD

    def test_moderate_classification(self):
        """Fail scores moderately below threshold → MODERATE."""
        rec = _make_recorder()
        # threshold=0.55, fails at 0.42, 0.43, 0.44 → p50=0.43, gap=0.12
        _populate_fails(rec, [0.42, 0.43, 0.44])
        result = analyze(rec, weighted_vote_threshold=0.55)
        assert result["classification"] == CLASS_MODERATE

    def test_far_below_classification(self):
        """Fail scores far below threshold → FAR_BELOW."""
        rec = _make_recorder()
        # threshold=0.55, fails at 0.20, 0.25, 0.30 → p50=0.25, gap=0.30
        _populate_fails(rec, [0.20, 0.25, 0.30])
        result = analyze(rec, weighted_vote_threshold=0.55)
        assert result["classification"] == CLASS_FAR_BELOW

    def test_insufficient_data_with_few_fails(self):
        """Fewer than 3 fails → INSUFFICIENT_DATA."""
        rec = _make_recorder()
        _populate_fails(rec, [0.52, 0.53])  # only 2
        result = analyze(rec, weighted_vote_threshold=0.55)
        assert result["classification"] == CLASS_INSUFFICIENT_DATA

    def test_insufficient_data_empty_recorder(self):
        """Empty recorder → INSUFFICIENT_DATA."""
        rec = _make_recorder()
        result = analyze(rec, weighted_vote_threshold=0.55)
        assert result["classification"] == CLASS_INSUFFICIENT_DATA


class TestVoteGapAnalyzer_NearMiss:
    """Near-miss counting accuracy."""

    def test_near_miss_count(self):
        rec = _make_recorder()
        # threshold=0.55: 0.52 gap=0.03 (near), 0.53 gap=0.02 (near), 0.40 gap=0.15 (not near)
        _populate_fails(rec, [0.52, 0.53, 0.40])
        result = analyze(rec, weighted_vote_threshold=0.55)
        assert result["near_miss_count"] == 2
        assert result["near_miss_rate_pct"] == pytest.approx(66.67, abs=0.1)

    def test_no_near_misses(self):
        rec = _make_recorder()
        _populate_fails(rec, [0.20, 0.25, 0.30])
        result = analyze(rec, weighted_vote_threshold=0.55)
        assert result["near_miss_count"] == 0


class TestVoteGapAnalyzer_PassRate:
    """pass_rate reflects all records."""

    def test_pass_rate_mixed(self):
        rec = _make_recorder()
        _populate_mixed(rec, pass_scores=[0.60, 0.65], fail_scores=[0.40, 0.42, 0.44])
        result = analyze(rec, weighted_vote_threshold=0.55)
        assert result["pass_rate"] == pytest.approx(0.4, abs=0.01)


class TestVoteGapAnalyzer_Wrapper:
    """VoteGapAnalyzer wrapper class."""

    def test_analyze_current(self):
        rec = _make_recorder()
        _populate_fails(rec, [0.52, 0.53, 0.51])
        analyzer = VoteGapAnalyzer(recorder=rec, weighted_vote_threshold=0.55)
        result = analyzer.analyze_current()
        assert result["classification"] == CLASS_NEAR_THRESHOLD

    def test_get_vote_signals(self):
        rec = _make_recorder()
        _populate_fails(rec, [0.52, 0.53, 0.51])
        analyzer = VoteGapAnalyzer(recorder=rec, weighted_vote_threshold=0.55)
        signals = analyzer.get_vote_signals()
        assert "vote_classification" in signals
        assert "vote_gap_at_p50" in signals
        assert "vote_near_miss_rate_pct" in signals
        assert "vote_pass_rate" in signals

    def test_none_recorder_returns_insufficient(self):
        analyzer = VoteGapAnalyzer(recorder=None)
        result = analyzer.analyze_current()
        assert result["classification"] == CLASS_INSUFFICIENT_DATA


class TestVoteGapAnalyzer_Importability:
    def test_importable(self):
        from src.scanner.vote_gap_analyzer import VoteGapAnalyzer, analyze
        assert VoteGapAnalyzer is not None
        assert analyze is not None
