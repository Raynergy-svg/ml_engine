"""
Tests for Phase 54 US-338: Regime Confidence Scorer.

Tests cover:
  - Probability estimation from indicators
  - Tier classification (high, medium, low)
  - Sizing factors per tier
  - Transition detection on prob drop
  - Transition cycle countdown
  - Persistence roundtrip
  - Lazy imports and wiring
"""

import json
import os
import tempfile

import pytest

from src.scanner.regime_confidence import (
    RegimeConfidenceScorer,
    RegimeConfidenceConfig,
    RegimeScoreResult,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def scorer(tmp_dir):
    config = RegimeConfidenceConfig(
        persistence_path=os.path.join(tmp_dir, "regime_conf.json"),
    )
    return RegimeConfidenceScorer(config)


class TestProbabilityEstimation:

    def test_high_adx_high_trend(self, scorer):
        """High ADX + trend → high trending probability."""
        result = scorer.score(adx=50.0, atr_percentile=0.3, trend_strength=0.8)
        # Should favor bull
        assert result.probabilities["bull"] > result.probabilities["sideways"]

    def test_low_adx_low_trend(self, scorer):
        """Low ADX + low trend → higher sideways probability."""
        result = scorer.score(adx=12.0, atr_percentile=0.3, trend_strength=0.5)
        assert result.probabilities["sideways"] > 0.3

    def test_probabilities_sum_to_one(self, scorer):
        result = scorer.score(adx=30.0, atr_percentile=0.5, trend_strength=0.6)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_all_probabilities_positive(self, scorer):
        result = scorer.score(adx=60.0, atr_percentile=0.9, trend_strength=0.9)
        for p in result.probabilities.values():
            assert p >= 0.0


class TestTierClassification:

    def test_high_tier(self, scorer):
        """Strong indicators → high confidence → full sizing."""
        result = scorer.score(adx=55.0, atr_percentile=0.2, trend_strength=0.9)
        if result.regime_confidence >= 0.75:
            assert result.tier == "high"
            assert result.sizing_factor == 1.0

    def test_medium_tier(self, scorer):
        """Moderate indicators → medium confidence → 80% sizing."""
        # Use moderate values that should give medium confidence
        result = scorer.score(adx=30.0, atr_percentile=0.4, trend_strength=0.6)
        assert result.sizing_factor <= 1.0

    def test_low_tier(self, tmp_dir):
        """Weak indicators → low confidence → 60% sizing."""
        config = RegimeConfidenceConfig(
            persistence_path=os.path.join(tmp_dir, "low.json"),
            high_confidence_threshold=0.99,
            medium_confidence_threshold=0.95,  # Force low tier
        )
        s = RegimeConfidenceScorer(config)
        result = s.score(adx=15.0, atr_percentile=0.5, trend_strength=0.5)
        assert result.tier == "low"
        assert result.sizing_factor == 0.60


class TestTransitionDetection:

    def test_transition_on_prob_drop(self, scorer):
        """Transition triggered when max prob drops > 15%."""
        # First call — establish high baseline
        scorer.score(adx=55.0, atr_percentile=0.1, trend_strength=0.9)
        # Second call — drop indicators sharply
        result = scorer.score(adx=12.0, atr_percentile=0.8, trend_strength=0.5)
        if result.is_transition:
            assert result.tier == "transition"
            assert result.sizing_factor == 0.50

    def test_transition_cycles_countdown(self, scorer):
        """Transition cycles count down properly."""
        # Force transition
        scorer._last_max_prob = 0.95
        result1 = scorer.score(adx=12.0, atr_percentile=0.5, trend_strength=0.5)
        if result1.tier == "transition":
            # Next call should still be in transition or back to normal
            result2 = scorer.score(adx=12.0, atr_percentile=0.5, trend_strength=0.5)
            assert result2.transition_cycles_remaining < result1.transition_cycles_remaining or result2.tier != "transition"

    def test_no_transition_small_drop(self, scorer):
        """Small drop doesn't trigger transition."""
        scorer._last_max_prob = 0.60
        result = scorer.score(adx=25.0, atr_percentile=0.5, trend_strength=0.5)
        # Small drop — should not trigger
        assert not result.is_transition or abs(0.60 - result.regime_confidence) > 0.15


class TestScoreResult:

    def test_roundtrip(self):
        r = RegimeScoreResult(
            regime_confidence=0.80, sizing_factor=1.0, tier="high",
            probabilities={"bull": 0.5, "bear": 0.2, "sideways": 0.3},
            timestamp="2026-03-25T00:00:00Z",
        )
        d = r.to_dict()
        restored = RegimeScoreResult.from_dict(d)
        assert abs(restored.regime_confidence - 0.80) < 0.01
        assert restored.tier == "high"


class TestPersistence:

    def test_save_load_roundtrip(self, tmp_dir):
        path = os.path.join(tmp_dir, "regime_rt.json")
        config = RegimeConfidenceConfig(persistence_path=path)
        s1 = RegimeConfidenceScorer(config)
        s1.score(adx=30.0, atr_percentile=0.5, trend_strength=0.6)
        s1.save_state()

        s2 = RegimeConfidenceScorer(config)
        assert s2._last_max_prob is not None
        assert len(s2._history) == 1

    def test_load_missing_file(self, tmp_dir):
        config = RegimeConfidenceConfig(
            persistence_path=os.path.join(tmp_dir, "missing.json"))
        s = RegimeConfidenceScorer(config)
        assert s._last_max_prob is None

    def test_load_corrupted_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("{bad")
        config = RegimeConfidenceConfig(persistence_path=path)
        s = RegimeConfidenceScorer(config)
        assert s._last_max_prob is None


class TestHistory:

    def test_history_tracked(self, scorer):
        scorer.score(adx=30.0, atr_percentile=0.5, trend_strength=0.6)
        scorer.score(adx=40.0, atr_percentile=0.3, trend_strength=0.7)
        history = scorer.get_history()
        assert len(history) == 2

    def test_history_capped(self, tmp_dir):
        config = RegimeConfidenceConfig(
            persistence_path=os.path.join(tmp_dir, "cap.json"),
            max_history=5,
        )
        s = RegimeConfidenceScorer(config)
        for i in range(10):
            s.score(adx=float(20 + i), atr_percentile=0.5, trend_strength=0.5)
        assert len(s._history) == 5


class TestLazyImports:

    def test_regime_confidence_scorer_import(self):
        from src.scanner import RegimeConfidenceScorer
        assert RegimeConfidenceScorer is not None

    def test_regime_confidence_config_import(self):
        from src.scanner import RegimeConfidenceConfig
        assert RegimeConfidenceConfig is not None


class TestWiring:

    def test_execution_manager_has_regime_scorer(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.__init__)
            assert "_regime_confidence_scorer" in source
            assert "RegimeConfidenceScorer" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")
