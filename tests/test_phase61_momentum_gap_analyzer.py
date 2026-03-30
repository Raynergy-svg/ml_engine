"""Phase 61 US-372 — MomentumGapAnalyzer unit tests.

Minimum 8 tests per acceptance criteria. Tests cover all three
classification branches, near_miss_rate, empty/safe defaults,
recommendation strings, and gap_at_p50 calculation accuracy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.scanner.momentum_score_recorder import MomentumScoreRecorder
from src.scanner.momentum_gap_analyzer import (
    analyze,
    MomentumGapAnalyzer,
    CLASS_NEAR_THRESHOLD,
    CLASS_MODERATE,
    CLASS_FAR_BELOW,
    CLASS_INSUFFICIENT_DATA,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_recorder(
    fail_scores: list[float],
    pass_scores: list[float] | None = None,
) -> MomentumScoreRecorder:
    """Build a MomentumScoreRecorder with specified fail/pass scores."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    path.unlink()
    rec = MomentumScoreRecorder(state_path=path)
    for s in fail_scores:
        rec.record("EUR/USD", momentum_score=s, momentum_passed=False)
    for s in (pass_scores or []):
        rec.record("EUR/USD", momentum_score=s, momentum_passed=True)
    return rec


# ── Classification tests ─────────────────────────────────────────────────────


class TestMomentumGapAnalyzer_Classification:
    """NEAR_THRESHOLD / MODERATE / FAR_BELOW classification branches."""

    def test_near_threshold_classification(self):
        """10 fail records at 0.42 vs min_momentum=0.45 → gap=0.03 → NEAR_THRESHOLD."""
        rec = _make_recorder(fail_scores=[0.42] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert result["classification"] == CLASS_NEAR_THRESHOLD
        assert result["gap_at_p50"] == pytest.approx(0.03, abs=1e-4)

    def test_moderate_classification(self):
        """Fail scores at 0.35 vs min_momentum=0.45 → gap=0.10 → MODERATE."""
        rec = _make_recorder(fail_scores=[0.35] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert result["classification"] == CLASS_MODERATE
        assert result["gap_at_p50"] == pytest.approx(0.10, abs=1e-4)

    def test_far_below_classification(self):
        """Fail scores at 0.20 vs min_momentum=0.45 → gap=0.25 → FAR_BELOW."""
        rec = _make_recorder(fail_scores=[0.20] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert result["classification"] == CLASS_FAR_BELOW
        assert result["gap_at_p50"] == pytest.approx(0.25, abs=1e-4)

    def test_insufficient_data_classification(self):
        """Fewer than 3 fail records → INSUFFICIENT_DATA, no crash."""
        rec = _make_recorder(fail_scores=[0.40, 0.38])  # only 2 fails
        result = analyze(rec, min_momentum=0.45)
        assert result["classification"] == CLASS_INSUFFICIENT_DATA

    def test_boundary_exactly_near_miss(self):
        """gap_at_p50 == 0.05 exactly → NEAR_THRESHOLD (≤ boundary inclusive)."""
        rec = _make_recorder(fail_scores=[0.40] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert result["gap_at_p50"] == pytest.approx(0.05, abs=1e-4)
        assert result["classification"] == CLASS_NEAR_THRESHOLD

    def test_boundary_just_over_near_miss(self):
        """gap_at_p50 = 0.051 → MODERATE."""
        rec = _make_recorder(fail_scores=[0.399] * 10)
        result = analyze(rec, min_momentum=0.45)
        # gap = 0.45 - 0.399 = 0.051 → MODERATE
        assert result["classification"] == CLASS_MODERATE


# ── Near-miss rate tests ──────────────────────────────────────────────────────


class TestMomentumGapAnalyzer_NearMissRate:
    """near_miss_count and near_miss_rate_pct correctness."""

    def test_near_miss_rate_correct(self):
        """5 near-miss fails (gap ≤ 0.05) + 5 far fails → near_miss_rate_pct=50.0."""
        near_miss_scores = [0.42] * 5   # gap = 0.45 - 0.42 = 0.03 ≤ 0.05 → near miss
        far_scores = [0.10] * 5         # gap = 0.45 - 0.10 = 0.35 > 0.05 → not near miss
        rec = _make_recorder(fail_scores=near_miss_scores + far_scores)
        result = analyze(rec, min_momentum=0.45)
        assert result["near_miss_count"] == 5
        assert result["near_miss_rate_pct"] == pytest.approx(50.0, abs=0.1)

    def test_all_near_miss(self):
        """All fails within near-miss threshold → near_miss_rate_pct=100.0."""
        rec = _make_recorder(fail_scores=[0.43] * 8)  # gap=0.02 ≤ 0.05
        result = analyze(rec, min_momentum=0.45)
        assert result["near_miss_rate_pct"] == pytest.approx(100.0, abs=0.1)

    def test_no_near_miss(self):
        """All fails far below threshold → near_miss_rate_pct=0.0."""
        rec = _make_recorder(fail_scores=[0.20] * 8)  # gap=0.25 > 0.05
        result = analyze(rec, min_momentum=0.45)
        assert result["near_miss_rate_pct"] == pytest.approx(0.0, abs=0.1)


# ── Recommendation strings ────────────────────────────────────────────────────


class TestMomentumGapAnalyzer_Recommendations:
    """Recommendation strings match PRD specifications."""

    def test_near_threshold_recommendation(self):
        rec = _make_recorder(fail_scores=[0.42] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert "0.02-0.05 momentum threshold reduction" in result["recommendation"]

    def test_moderate_recommendation(self):
        rec = _make_recorder(fail_scores=[0.35] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert "XGBoost momentum signal quality" in result["recommendation"]

    def test_far_below_recommendation(self):
        rec = _make_recorder(fail_scores=[0.20] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert "retraining" in result["recommendation"].lower()

    def test_insufficient_data_recommendation(self):
        rec = _make_recorder(fail_scores=[0.40])
        result = analyze(rec, min_momentum=0.45)
        assert "Insufficient" in result["recommendation"]


# ── Output keys ──────────────────────────────────────────────────────────────


class TestMomentumGapAnalyzer_OutputKeys:
    """All required output keys present in result dict."""

    _REQUIRED_KEYS = {
        "count", "fail_count", "pass_rate", "gap_at_p50", "gap_at_p75",
        "near_miss_count", "near_miss_rate_pct", "classification", "recommendation",
    }

    def test_all_required_keys_present_normal(self):
        rec = _make_recorder(fail_scores=[0.40] * 5, pass_scores=[0.55] * 3)
        result = analyze(rec, min_momentum=0.45)
        for key in self._REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_all_required_keys_present_insufficient(self):
        rec = _make_recorder(fail_scores=[0.40])
        result = analyze(rec, min_momentum=0.45)
        for key in self._REQUIRED_KEYS:
            assert key in result, f"Missing key on insufficient: {key}"

    def test_pass_rate_included(self):
        """pass_rate reflects passes relative to total records."""
        rec = _make_recorder(fail_scores=[0.35] * 3, pass_scores=[0.55] * 7)
        result = analyze(rec, min_momentum=0.45)
        assert result["pass_rate"] == pytest.approx(0.70, abs=1e-4)

    def test_gap_at_p75_calculation(self):
        """gap_at_p75 = min_momentum - p75 of fail scores."""
        # With uniform fail scores the p75 equals the score itself
        rec = _make_recorder(fail_scores=[0.30] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert result["gap_at_p75"] == pytest.approx(0.15, abs=1e-4)


# ── MomentumGapAnalyzer class ─────────────────────────────────────────────────


class TestMomentumGapAnalyzerClass:
    """MomentumGapAnalyzer wrapper class and get_momentum_signals()."""

    def test_analyze_current_no_recorder(self):
        """analyze_current() with recorder=None returns INSUFFICIENT_DATA safely."""
        analyzer = MomentumGapAnalyzer(recorder=None, min_momentum=0.45)
        result = analyzer.analyze_current()
        assert result["classification"] == CLASS_INSUFFICIENT_DATA

    def test_get_momentum_signals_keys(self):
        """get_momentum_signals() returns dict consumable by ScanHealthSynthesizer."""
        rec = _make_recorder(fail_scores=[0.42] * 10)
        analyzer = MomentumGapAnalyzer(recorder=rec, min_momentum=0.45)
        signals = analyzer.get_momentum_signals()
        assert "momentum_gap_at_p50" in signals
        assert "momentum_near_miss_rate_pct" in signals
        assert "momentum_classification" in signals
        assert "momentum_pass_rate" in signals

    def test_get_momentum_signals_near_threshold(self):
        """get_momentum_signals() reflects NEAR_THRESHOLD correctly."""
        rec = _make_recorder(fail_scores=[0.42] * 10)
        analyzer = MomentumGapAnalyzer(recorder=rec, min_momentum=0.45)
        signals = analyzer.get_momentum_signals()
        assert signals["momentum_classification"] == CLASS_NEAR_THRESHOLD
        assert signals["momentum_gap_at_p50"] == pytest.approx(0.03, abs=1e-4)

    def test_empty_analyzer_signals_safe(self):
        """get_momentum_signals() on empty recorder returns safe defaults."""
        analyzer = MomentumGapAnalyzer(min_momentum=0.45)
        signals = analyzer.get_momentum_signals()
        assert signals["momentum_gap_at_p50"] == 0.0
        assert signals["momentum_classification"] == CLASS_INSUFFICIENT_DATA
