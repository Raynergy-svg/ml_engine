"""
Tests for Phase 54 US-334: Dynamic Gate Threshold Optimizer.

Tests cover:
  - Recommendation computation (relax, tighten, no-op)
  - Evidence minimum enforcement
  - Adjustment cap (±15%)
  - Application and history tracking
  - Threshold direction awareness (higher_is_stricter vs lower_is_stricter)
  - Persistence roundtrip
  - Report generation
  - Lazy imports and wiring
"""

import json
import os
import tempfile

import pytest

from src.scanner.gate_threshold_optimizer import (
    GateThresholdOptimizer,
    GateThresholdOptimizerConfig,
    ThresholdRecommendation,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def optimizer(tmp_dir):
    config = GateThresholdOptimizerConfig(
        persistence_path=os.path.join(tmp_dir, "threshold_history.json"),
    )
    return GateThresholdOptimizer(config)


@pytest.fixture
def sample_gate_report():
    """Gate report showing one gate killing good trades, one protecting profit."""
    return {
        "setup_quality": {
            "rejections": 80,
            "passes": 200,
            "rejection_pct": 0.286,
            "good_rejection_rate": 0.20,  # < 0.30 → killing good trades
        },
        "spread_filter": {
            "rejections": 60,
            "passes": 220,
            "rejection_pct": 0.214,
            "good_rejection_rate": 0.85,  # > 0.70 → protecting profit
        },
        "fitness_detector": {
            "rejections": 10,  # < 50 minimum
            "passes": 270,
            "rejection_pct": 0.036,
            "good_rejection_rate": 0.15,
        },
        "calendar_filter": {
            "rejections": 55,
            "passes": 225,
            "rejection_pct": 0.196,
            "good_rejection_rate": 0.50,  # Between thresholds → no action
        },
    }


class TestRecommendationComputation:

    def test_relax_gate_killing_good_trades(self, optimizer, sample_gate_report):
        """Gate with low good_rejection_rate gets relaxed."""
        optimizer._current_thresholds["setup_quality"] = 0.35
        recs = optimizer.compute_recommendations(sample_gate_report)
        sq_recs = [r for r in recs if r.gate_name == "setup_quality"]
        assert len(sq_recs) == 1
        rec = sq_recs[0]
        # setup_quality is higher_is_stricter → relax = lower threshold
        assert rec.recommended_threshold < 0.35
        assert "Killing good trades" in rec.reason

    def test_tighten_gate_protecting_profit(self, optimizer, sample_gate_report):
        """Gate with high good_rejection_rate gets tightened."""
        optimizer._current_thresholds["spread_filter"] = 0.20
        recs = optimizer.compute_recommendations(sample_gate_report)
        sf_recs = [r for r in recs if r.gate_name == "spread_filter"]
        assert len(sf_recs) == 1
        rec = sf_recs[0]
        # spread_filter is lower_is_stricter → tighten = lower threshold
        assert rec.recommended_threshold < 0.20
        assert "Protecting profit" in rec.reason

    def test_no_recommendation_insufficient_evidence(self, optimizer, sample_gate_report):
        """Gate with < 50 rejections gets no recommendation."""
        recs = optimizer.compute_recommendations(sample_gate_report)
        fitness_recs = [r for r in recs if r.gate_name == "fitness_detector"]
        assert len(fitness_recs) == 0

    def test_no_recommendation_neutral_gate(self, optimizer, sample_gate_report):
        """Gate between relax/tighten thresholds gets no recommendation."""
        optimizer._current_thresholds["calendar_filter"] = 30
        recs = optimizer.compute_recommendations(sample_gate_report)
        cal_recs = [r for r in recs if r.gate_name == "calendar_filter"]
        assert len(cal_recs) == 0

    def test_adjustment_capped_at_max(self, tmp_dir):
        """Adjustment never exceeds max_adjustment_pct."""
        config = GateThresholdOptimizerConfig(
            relax_pct=0.50,  # Would want 50%, but max is 15%
            max_adjustment_pct=0.15,
            persistence_path=os.path.join(tmp_dir, "cap.json"),
        )
        opt = GateThresholdOptimizer(config)
        opt._current_thresholds["setup_quality"] = 0.35
        report = {"setup_quality": {"rejections": 100, "good_rejection_rate": 0.10}}
        recs = opt.compute_recommendations(report)
        assert len(recs) == 1
        assert recs[0].adjustment_pct <= 0.15


class TestRecommendationApplication:

    def test_apply_updates_threshold(self, optimizer):
        """Applying recommendation updates current threshold."""
        rec = ThresholdRecommendation(
            gate_name="setup_quality",
            current_threshold=0.35,
            recommended_threshold=0.315,
            adjustment_pct=0.10,
            reason="test",
            evidence_count=80,
            timestamp="2026-03-25T00:00:00Z",
        )
        applied = optimizer.apply_recommendations([rec])
        assert len(applied) == 1
        assert applied[0].applied is True
        assert optimizer._current_thresholds["setup_quality"] == 0.315

    def test_apply_records_history(self, optimizer):
        """Applied recommendations are recorded in history."""
        rec = ThresholdRecommendation(
            gate_name="spread_filter",
            current_threshold=0.20,
            recommended_threshold=0.19,
            adjustment_pct=-0.05,
            reason="test",
            evidence_count=60,
        )
        optimizer.apply_recommendations([rec])
        history = optimizer.get_adjustment_history("spread_filter")
        assert len(history["spread_filter"]) == 1

    def test_apply_skips_low_evidence(self, optimizer):
        """Recommendations with insufficient evidence are skipped."""
        rec = ThresholdRecommendation(
            gate_name="test_gate",
            current_threshold=0.50,
            recommended_threshold=0.45,
            adjustment_pct=0.10,
            reason="test",
            evidence_count=10,  # Below 50 minimum
        )
        applied = optimizer.apply_recommendations([rec])
        assert len(applied) == 0

    def test_history_trimmed_to_max(self, optimizer):
        """History per gate is capped at max_history_per_gate."""
        for i in range(15):
            rec = ThresholdRecommendation(
                gate_name="test_gate",
                current_threshold=float(i),
                recommended_threshold=float(i + 1),
                evidence_count=100,
            )
            optimizer.apply_recommendations([rec])
        history = optimizer.get_adjustment_history("test_gate")
        assert len(history["test_gate"]) == 10


class TestThresholdRecommendation:

    def test_to_dict_roundtrip(self):
        """ThresholdRecommendation round-trips through dict."""
        rec = ThresholdRecommendation(
            gate_name="setup_quality",
            current_threshold=0.35,
            recommended_threshold=0.315,
            adjustment_pct=0.10,
            reason="test reason",
            good_rejection_rate=0.20,
            evidence_count=80,
            applied=True,
            timestamp="2026-03-25T00:00:00Z",
        )
        d = rec.to_dict()
        restored = ThresholdRecommendation.from_dict(d)
        assert restored.gate_name == "setup_quality"
        assert abs(restored.current_threshold - 0.35) < 0.001
        assert restored.applied is True


class TestPersistence:

    def test_save_load_roundtrip(self, tmp_dir):
        """Optimizer state round-trips through save/load."""
        path = os.path.join(tmp_dir, "opt_rt.json")
        config = GateThresholdOptimizerConfig(persistence_path=path)
        opt1 = GateThresholdOptimizer(config)
        opt1._current_thresholds["setup_quality"] = 0.315
        rec = ThresholdRecommendation(
            gate_name="setup_quality",
            current_threshold=0.35,
            recommended_threshold=0.315,
            evidence_count=80,
        )
        opt1.apply_recommendations([rec])
        opt1.save_state()

        opt2 = GateThresholdOptimizer(config)
        assert abs(opt2._current_thresholds["setup_quality"] - 0.315) < 0.001
        history = opt2.get_adjustment_history("setup_quality")
        assert len(history["setup_quality"]) == 1

    def test_load_missing_file(self, tmp_dir):
        """Missing file starts fresh."""
        config = GateThresholdOptimizerConfig(
            persistence_path=os.path.join(tmp_dir, "missing.json"),
        )
        opt = GateThresholdOptimizer(config)
        assert len(opt._current_thresholds) == 0

    def test_load_corrupted_file(self, tmp_dir):
        """Corrupted file gracefully falls back."""
        path = os.path.join(tmp_dir, "corrupt.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        config = GateThresholdOptimizerConfig(persistence_path=path)
        opt = GateThresholdOptimizer(config)
        assert len(opt._current_thresholds) == 0


class TestReportsAndQueries:

    def test_get_recommendations_report(self, optimizer, sample_gate_report):
        """Report returns dict format."""
        optimizer._current_thresholds["setup_quality"] = 0.35
        optimizer._current_thresholds["spread_filter"] = 0.20
        report = optimizer.get_recommendations_report(sample_gate_report)
        assert isinstance(report, dict)
        # Should have setup_quality (relax) and spread_filter (tighten)
        assert "setup_quality" in report or "spread_filter" in report

    def test_get_current_thresholds(self, optimizer):
        """Current thresholds returns copy."""
        optimizer._current_thresholds["test"] = 0.50
        thresholds = optimizer.get_current_thresholds()
        thresholds["test"] = 0.99  # Mutate copy
        assert optimizer._current_thresholds["test"] == 0.50


class TestLazyImports:

    def test_gate_threshold_optimizer_import(self):
        from src.scanner import GateThresholdOptimizer
        assert GateThresholdOptimizer is not None

    def test_gate_threshold_optimizer_config_import(self):
        from src.scanner import GateThresholdOptimizerConfig
        assert GateThresholdOptimizerConfig is not None


class TestWiring:

    def test_execution_manager_has_threshold_optimizer(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.__init__)
            assert "_gate_threshold_optimizer" in source
            assert "GateThresholdOptimizer" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")

    def test_optimizer_called_in_sync(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager)
            assert "compute_recommendations" in source or "gate_threshold_optimizer" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")
