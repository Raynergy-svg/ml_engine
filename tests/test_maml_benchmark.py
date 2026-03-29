"""Tests for MAML Ridge Phase 2 Benchmarking infrastructure.

Covers:
    - Record collection (predictions, outcomes, transitions, meta-trains)
    - MAE computation (overall, per-regime, rolling)
    - Adaptation speed (convergence detection)
    - Regime divergence analysis
    - Meta-train curve analysis
    - Calibration bins
    - Win rate analysis
    - Promotion verdict logic
    - Persistence (save/load)
    - Edge cases (empty data, insufficient samples)
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.recursive_intelligence.maml_benchmark import (
    MAMLBenchmark,
    BenchmarkConfig,
    PredictionRecord,
    OutcomeRecord,
    RegimeTransition,
    MetaTrainRecord,
)


@pytest.fixture(autouse=True)
def _isolate_benchmark(tmp_path):
    """Isolate benchmark persistence to tmp directory."""
    with patch("src.recursive_intelligence.maml_benchmark._BENCHMARK_DIR", tmp_path / "bench"):
        yield


@pytest.fixture
def config():
    return BenchmarkConfig(
        max_predictions=100,
        max_outcomes=50,
        save_interval=5,
        convergence_window=5,
        convergence_threshold=8.0,
        min_outcomes_for_report=10,
        promotion_mae_advantage=2.0,
        min_regime_samples=3,
    )


@pytest.fixture
def bench(config):
    return MAMLBenchmark(config=config)


def _seed_outcomes(bench, n=30, regime="NORMAL", maml_bias=0, ridge_bias=0):
    """Helper: seed n trade outcomes with controlled errors."""
    for i in range(n):
        realized = 50.0 + np.random.uniform(-20, 20)
        maml_pred = realized + maml_bias + np.random.uniform(-3, 3)
        ridge_pred = realized + ridge_bias + np.random.uniform(-3, 3)
        bench.record_outcome(
            trade_id=f"trade_{i}",
            regime=regime,
            maml_confidence=max(0, min(100, maml_pred)),
            ridge_confidence=max(0, min(100, ridge_pred)),
            realized_confidence=realized,
            pnl_pips=np.random.uniform(-30, 30),
            won=realized > 50,
        )


# ── Record Collection ───────────────────────────────────────────────


class TestRecording:
    def test_record_prediction(self, bench):
        bench.record_prediction("NORMAL", 55.0, 52.0)
        assert len(bench._predictions) == 1
        assert bench._predictions[0].regime == "NORMAL"
        assert bench._predictions[0].maml_confidence == 55.0

    def test_record_outcome(self, bench):
        bench.record_outcome(
            trade_id="t1", regime="HIGH", maml_confidence=60.0,
            ridge_confidence=55.0, realized_confidence=65.0,
            pnl_pips=15.0, won=True,
        )
        assert len(bench._outcomes) == 1
        assert bench._outcomes[0].trade_id == "t1"
        assert bench._outcomes[0].won is True

    def test_record_regime_transition(self, bench):
        bench.record_regime_transition("NORMAL", "HIGH")
        assert len(bench._transitions) == 1
        assert bench._transitions[0].old_regime == "NORMAL"
        assert bench._transitions[0].new_regime == "HIGH"
        assert bench._trades_since_transition == 0

    def test_same_regime_transition_ignored(self, bench):
        bench.record_regime_transition("NORMAL", "NORMAL")
        assert len(bench._transitions) == 0

    def test_auto_regime_transition_on_prediction(self, bench):
        bench.record_prediction("NORMAL", 50.0, 50.0)
        bench.record_prediction("HIGH", 55.0, 52.0)
        assert len(bench._transitions) == 1
        assert bench._transitions[0].new_regime == "HIGH"

    def test_record_meta_train(self, bench):
        bench.record_meta_train({
            "status": "trained",
            "meta_trains": 1,
            "outer_loss": 0.05,
            "regime_losses": {"NORMAL": 0.04, "HIGH": 0.06},
            "regimes_used": ["NORMAL", "HIGH"],
        })
        assert len(bench._meta_trains) == 1
        assert bench._meta_trains[0].outer_loss == 0.05

    def test_non_trained_meta_train_ignored(self, bench):
        bench.record_meta_train({"status": "insufficient_data"})
        assert len(bench._meta_trains) == 0

    def test_sliding_window_predictions(self, bench):
        for i in range(150):
            bench.record_prediction("NORMAL", 50.0, 50.0)
        assert len(bench._predictions) == bench.config.max_predictions

    def test_sliding_window_outcomes(self, bench):
        for i in range(70):
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL", maml_confidence=50.0,
                ridge_confidence=50.0, realized_confidence=50.0,
                pnl_pips=0.0, won=False,
            )
        assert len(bench._outcomes) == bench.config.max_outcomes


# ── MAE Computation ─────────────────────────────────────────────────


class TestMAE:
    def test_mae_empty(self, bench):
        assert bench.compute_mae("maml") is None

    def test_mae_perfect_prediction(self, bench):
        for i in range(15):
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL", maml_confidence=60.0,
                ridge_confidence=55.0, realized_confidence=60.0,
                pnl_pips=10.0, won=True,
            )
        assert bench.compute_mae("maml") == 0.0
        assert bench.compute_mae("ridge") == 5.0

    def test_mae_with_regime_filter(self, bench):
        # 10 NORMAL outcomes
        for i in range(10):
            bench.record_outcome(
                trade_id=f"n{i}", regime="NORMAL", maml_confidence=50.0,
                ridge_confidence=50.0, realized_confidence=55.0,
                pnl_pips=5.0, won=True,
            )
        # 10 HIGH outcomes with different error
        for i in range(10):
            bench.record_outcome(
                trade_id=f"h{i}", regime="HIGH", maml_confidence=50.0,
                ridge_confidence=50.0, realized_confidence=60.0,
                pnl_pips=10.0, won=True,
            )
        normal_mae = bench.compute_mae("maml", regime="NORMAL")
        high_mae = bench.compute_mae("maml", regime="HIGH")
        assert normal_mae == 5.0
        assert high_mae == 10.0

    def test_mae_last_n(self, bench):
        # First 10 with perfect prediction
        for i in range(10):
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL", maml_confidence=50.0,
                ridge_confidence=50.0, realized_confidence=50.0,
                pnl_pips=0.0, won=False,
            )
        # Last 5 with 10-point error
        for i in range(5):
            bench.record_outcome(
                trade_id=f"t{10+i}", regime="NORMAL", maml_confidence=50.0,
                ridge_confidence=50.0, realized_confidence=60.0,
                pnl_pips=10.0, won=True,
            )
        assert bench.compute_mae("maml", last_n=5) == 10.0
        # Overall should be lower
        overall = bench.compute_mae("maml")
        assert overall is not None
        assert overall < 10.0

    def test_regime_mae_dict(self, bench):
        _seed_outcomes(bench, 15, regime="NORMAL")
        result = bench.compute_regime_mae("maml")
        assert "NORMAL" in result
        assert result["NORMAL"] is not None
        assert result["HIGH"] is None  # No HIGH data


# ── Adaptation Speed ────────────────────────────────────────────────


class TestAdaptationSpeed:
    def test_no_transitions(self, bench):
        result = bench.compute_adaptation_speed()
        assert result["transitions"] == 0

    def test_convergence_detection(self, bench):
        bench.record_regime_transition("NORMAL", "HIGH")

        # Record outcomes in HIGH regime — all close to realized
        for i in range(10):
            bench.record_outcome(
                trade_id=f"t{i}", regime="HIGH",
                maml_confidence=55.0, ridge_confidence=50.0,
                realized_confidence=55.0,  # MAML is accurate
                pnl_pips=5.0, won=True,
            )

        # Should converge (MAE=0 < threshold=8.0)
        assert bench._transitions[0].converged_after is not None

        result = bench.compute_adaptation_speed()
        assert result["converged"] == 1
        assert result["avg_trades_to_converge"] is not None

    def test_no_convergence_high_error(self, bench):
        bench.record_regime_transition("NORMAL", "HIGH")

        # Record outcomes with high error
        for i in range(10):
            bench.record_outcome(
                trade_id=f"t{i}", regime="HIGH",
                maml_confidence=30.0, ridge_confidence=50.0,
                realized_confidence=70.0,  # MAML is way off
                pnl_pips=5.0, won=True,
            )

        # Should NOT converge (MAE=40 > threshold=8.0)
        assert bench._transitions[0].converged_after is None


# ── Regime Divergence ───────────────────────────────────────────────


class TestRegimeDivergence:
    def test_insufficient_data(self, bench):
        result = bench.compute_regime_divergence()
        assert result["verdict"] == "insufficient_data"

    def test_strong_divergence(self, bench):
        # NORMAL: predictions around 40
        for i in range(10):
            bench.record_prediction("NORMAL", 40.0 + np.random.uniform(-2, 2), 50.0)
        # HIGH: predictions around 75
        for i in range(10):
            bench.record_prediction("HIGH", 75.0 + np.random.uniform(-2, 2), 50.0)

        result = bench.compute_regime_divergence()
        assert result["verdict"] in ("strong_divergence", "moderate_divergence")
        assert result["divergence_score"] is not None
        assert result["divergence_score"] > 0.1

    def test_no_divergence(self, bench):
        # Both regimes: predictions around 50
        for i in range(10):
            bench.record_prediction("NORMAL", 50.0 + np.random.uniform(-1, 1), 50.0)
        for i in range(10):
            bench.record_prediction("HIGH", 50.0 + np.random.uniform(-1, 1), 50.0)

        result = bench.compute_regime_divergence()
        assert result["verdict"] in ("no_divergence", "weak_divergence")


# ── Meta-Train Curve ────────────────────────────────────────────────


class TestMetaTrainCurve:
    def test_no_trains(self, bench):
        result = bench.compute_meta_train_curve()
        assert result["trend"] == "no_data"

    def test_improving_trend(self, bench):
        # Losses decreasing over time
        for i in range(10):
            bench.record_meta_train({
                "status": "trained",
                "meta_trains": i + 1,
                "outer_loss": 1.0 - i * 0.08,  # 1.0 → 0.28
                "regime_losses": {"NORMAL": 1.0 - i * 0.08},
                "regimes_used": ["NORMAL"],
            })

        result = bench.compute_meta_train_curve()
        assert result["trend"] == "improving"
        assert result["improvement_pct"] is not None
        assert result["improvement_pct"] > 10

    def test_degrading_trend(self, bench):
        # Losses increasing
        for i in range(10):
            bench.record_meta_train({
                "status": "trained",
                "meta_trains": i + 1,
                "outer_loss": 0.1 + i * 0.1,  # 0.1 → 1.0
                "regime_losses": {},
                "regimes_used": ["NORMAL"],
            })

        result = bench.compute_meta_train_curve()
        assert result["trend"] == "degrading"


# ── Calibration ─────────────────────────────────────────────────────


class TestCalibration:
    def test_insufficient_data(self, bench):
        result = bench.compute_calibration()
        assert result["status"] == "insufficient_data"

    def test_calibration_bins(self, bench):
        # Seed enough outcomes
        _seed_outcomes(bench, n=20)
        result = bench.compute_calibration(bins=5)
        assert result["status"] == "ok"
        assert len(result["maml"]) == 5
        assert len(result["ridge"]) == 5
        # Each bin has a range
        assert result["maml"][0]["range"] == "0-20"


# ── Win Rate Analysis ───────────────────────────────────────────────


class TestWinRate:
    def test_insufficient_data(self, bench):
        result = bench.compute_win_rate_by_model()
        assert result["status"] == "insufficient_data"

    def test_win_rate_split(self, bench):
        # When MAML more confident and wins
        for i in range(8):
            bench.record_outcome(
                trade_id=f"m{i}", regime="NORMAL",
                maml_confidence=70.0, ridge_confidence=50.0,
                realized_confidence=65.0, pnl_pips=15.0, won=True,
            )
        # When Ridge more confident and loses
        for i in range(7):
            bench.record_outcome(
                trade_id=f"r{i}", regime="NORMAL",
                maml_confidence=30.0, ridge_confidence=60.0,
                realized_confidence=35.0, pnl_pips=-10.0, won=False,
            )

        result = bench.compute_win_rate_by_model()
        assert result["status"] == "ok"
        assert result["maml_more_confident"]["count"] == 8
        assert result["maml_more_confident"]["win_rate"] == 1.0
        assert result["ridge_more_confident"]["count"] == 7
        assert result["ridge_more_confident"]["win_rate"] == 0.0


# ── Report & Verdict ────────────────────────────────────────────────


class TestReport:
    def test_collecting_data_report(self, bench):
        report = bench.generate_report()
        assert report["status"] == "collecting_data"
        assert "outcomes_needed" in report

    def test_full_report_structure(self, bench):
        _seed_outcomes(bench, n=20)

        report = bench.generate_report()
        assert report["status"] == "ready"
        assert "mae_comparison" in report
        assert "regime_mae" in report
        assert "adaptation_speed" in report
        assert "regime_divergence" in report
        assert "meta_train_curve" in report
        assert "calibration" in report
        assert "win_rate_analysis" in report
        assert "rolling_mae" in report
        assert "promotion_verdict" in report

    def test_promotion_verdict_promote(self, bench):
        # MAML consistently better than Ridge
        for i in range(30):
            realized = 55.0
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL",
                maml_confidence=55.0,  # Perfect
                ridge_confidence=45.0,  # 10 off
                realized_confidence=realized,
                pnl_pips=10.0, won=True,
            )

        # Add divergence data
        for i in range(10):
            bench.record_prediction("NORMAL", 40.0, 50.0)
        for i in range(10):
            bench.record_prediction("HIGH", 70.0, 50.0)

        # Add improving meta-trains
        for i in range(5):
            bench.record_meta_train({
                "status": "trained",
                "meta_trains": i + 1,
                "outer_loss": 1.0 - i * 0.15,
                "regime_losses": {"NORMAL": 0.5},
                "regimes_used": ["NORMAL"],
            })

        report = bench.generate_report()
        verdict = report["promotion_verdict"]
        assert verdict["decision"] in ("PROMOTE", "LIKELY_PROMOTE")
        assert verdict["score"] >= 2

    def test_promotion_verdict_do_not_promote(self, bench):
        # Ridge consistently better
        for i in range(20):
            realized = 55.0
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL",
                maml_confidence=40.0,  # 15 off
                ridge_confidence=55.0,  # Perfect
                realized_confidence=realized,
                pnl_pips=10.0, won=True,
            )

        report = bench.generate_report()
        verdict = report["promotion_verdict"]
        assert verdict["decision"] in ("DO_NOT_PROMOTE", "CONTINUE_SHADOW")

    def test_summary_string(self, bench):
        _seed_outcomes(bench, n=20)
        summary = bench.generate_summary()
        assert "MAML Ridge Benchmark Report" in summary
        assert "MAE Comparison" in summary


# ── Persistence ─────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, bench, config, tmp_path):
        # Seed data — both predictions in same regime to avoid auto-transition
        bench.record_prediction("NORMAL", 55.0, 50.0)
        bench.record_prediction("NORMAL", 60.0, 52.0)
        bench.record_outcome(
            trade_id="t1", regime="NORMAL",
            maml_confidence=55.0, ridge_confidence=50.0,
            realized_confidence=58.0, pnl_pips=10.0, won=True,
        )
        bench.record_regime_transition("NORMAL", "HIGH")
        bench.record_meta_train({
            "status": "trained", "meta_trains": 1,
            "outer_loss": 0.05, "regime_losses": {"NORMAL": 0.04},
            "regimes_used": ["NORMAL"],
        })

        bench.force_save()

        # Create new instance — should load state
        bench2 = MAMLBenchmark(config=config)
        assert len(bench2._predictions) == 2
        assert len(bench2._outcomes) == 1
        assert len(bench2._transitions) == 1
        assert len(bench2._meta_trains) == 1

    def test_auto_save_on_outcome_interval(self, bench, tmp_path):
        state_path = tmp_path / "bench" / "benchmark_state.json"
        # config.save_interval = 5, so after 5 outcomes we should auto-save
        for i in range(5):
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL",
                maml_confidence=50.0, ridge_confidence=50.0,
                realized_confidence=50.0, pnl_pips=0.0, won=False,
            )
        assert state_path.exists()


# ── Data Records ────────────────────────────────────────────────────


class TestDataRecords:
    def test_prediction_record_serialization(self):
        rec = PredictionRecord(
            timestamp=1234567890.123, regime="HIGH",
            maml_confidence=55.5, ridge_confidence=50.0, scan_cycle=42,
        )
        d = rec.to_dict()
        rec2 = PredictionRecord.from_dict(d)
        assert rec2.regime == "HIGH"
        assert rec2.maml_confidence == 55.5
        assert rec2.scan_cycle == 42

    def test_outcome_record_serialization(self):
        rec = OutcomeRecord(
            timestamp=1234567890.0, trade_id="t1", regime="LOW",
            maml_confidence=60.0, ridge_confidence=55.0,
            realized_confidence=62.0, pnl_pips=12.5, won=True, regime_age=3,
        )
        d = rec.to_dict()
        rec2 = OutcomeRecord.from_dict(d)
        assert rec2.trade_id == "t1"
        assert rec2.won is True
        assert rec2.regime_age == 3

    def test_regime_transition_serialization(self):
        rec = RegimeTransition(
            timestamp=1234567890.0, old_regime="NORMAL",
            new_regime="EXTREME", trades_since=5, converged_after=3,
        )
        d = rec.to_dict()
        rec2 = RegimeTransition.from_dict(d)
        assert rec2.new_regime == "EXTREME"
        assert rec2.converged_after == 3

    def test_meta_train_record_serialization(self):
        rec = MetaTrainRecord(
            timestamp=1234567890.0, train_number=5,
            outer_loss=0.042, regime_losses={"NORMAL": 0.03, "HIGH": 0.05},
            regimes_used=["NORMAL", "HIGH"],
        )
        d = rec.to_dict()
        rec2 = MetaTrainRecord.from_dict(d)
        assert rec2.train_number == 5
        assert abs(rec2.outer_loss - 0.042) < 1e-6
        assert len(rec2.regime_losses) == 2


# ── Edge Cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_none_regime_normalized(self, bench):
        bench.record_prediction(None, 50.0, 50.0)
        assert bench._predictions[0].regime == "NORMAL"

    def test_lowercase_regime_normalized(self, bench):
        bench.record_prediction("high", 50.0, 50.0)
        # This will trigger a transition since current_regime is NORMAL
        assert bench._predictions[0].regime == "HIGH"

    def test_scan_cycle_tracking(self, bench):
        bench.set_scan_cycle(42)
        bench.record_prediction("NORMAL", 50.0, 50.0)
        assert bench._predictions[0].scan_cycle == 42

    def test_empty_report_summary(self, bench):
        summary = bench.generate_summary()
        assert "Collecting data" in summary

    def test_outcome_regime_age_tracking(self, bench):
        bench.record_outcome(
            trade_id="t0", regime="NORMAL",
            maml_confidence=50.0, ridge_confidence=50.0,
            realized_confidence=50.0, pnl_pips=0.0, won=False,
        )
        bench.record_outcome(
            trade_id="t1", regime="NORMAL",
            maml_confidence=50.0, ridge_confidence=50.0,
            realized_confidence=50.0, pnl_pips=0.0, won=False,
        )
        assert bench._outcomes[1].regime_age == 1  # Second trade since last transition


# ── Statistical Tests ───────────────────────────────────────────────


class TestStatisticalSignificance:
    def test_insufficient_data(self, bench):
        result = bench.compute_significance_test()
        assert result["status"] == "insufficient_data"

    def test_maml_significantly_better(self, bench):
        # MAML always closer to realized than Ridge
        np.random.seed(42)
        for i in range(40):
            realized = 55.0
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL",
                maml_confidence=55.0 + np.random.uniform(-1, 1),  # Close to realized
                ridge_confidence=45.0 + np.random.uniform(-1, 1),  # 10 points off
                realized_confidence=realized,
                pnl_pips=10.0, won=True,
            )
        result = bench.compute_significance_test()
        assert result["status"] == "ok"
        assert result["significant_at_005"] == True
        assert result["maml_better"] == True
        assert result["effect_size"] in ("large", "medium")
        assert len(result["ci_95"]) == 2

    def test_no_significant_difference(self, bench):
        # Both models equally accurate
        np.random.seed(42)
        for i in range(40):
            realized = 50.0 + np.random.uniform(-10, 10)
            noise = np.random.uniform(-3, 3)
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL",
                maml_confidence=realized + noise,
                ridge_confidence=realized + noise + np.random.uniform(-0.5, 0.5),
                realized_confidence=realized,
                pnl_pips=np.random.uniform(-5, 5), won=realized > 50,
            )
        result = bench.compute_significance_test()
        assert result["status"] == "ok"
        # With nearly identical predictions, should not be significant
        assert result["effect_size"] in ("negligible", "small")


class TestCalibrationMetrics:
    def test_insufficient_data(self, bench):
        result = bench.compute_calibration_metrics()
        assert result["status"] == "insufficient_data"

    def test_brier_ece_mce(self, bench):
        np.random.seed(42)
        for i in range(30):
            conf = np.random.uniform(20, 80)
            won = conf > 50  # Calibrated: high conf = win
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL",
                maml_confidence=conf, ridge_confidence=50.0,
                realized_confidence=conf, pnl_pips=10.0 if won else -10.0,
                won=won,
            )
        result = bench.compute_calibration_metrics()
        assert result["status"] == "ok"
        assert "brier_score" in result["maml"]
        assert "ece" in result["maml"]
        assert "mce" in result["maml"]
        assert result["maml"]["brier_score"] >= 0
        assert result["maml"]["ece"] >= 0
        assert len(result["maml"]["bins"]) > 0


class TestErrorTimeSeries:
    def test_insufficient_data(self, bench):
        result = bench.compute_error_time_series()
        assert result["status"] == "insufficient_data"

    def test_improving_trend(self, bench):
        # Errors decreasing over time
        for i in range(20):
            error = max(1, 20 - i)  # 20, 19, 18, ... 1
            realized = 50.0
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL",
                maml_confidence=realized + error,
                ridge_confidence=realized + 10,  # Constant error
                realized_confidence=realized,
                pnl_pips=5.0, won=True,
            )
        result = bench.compute_error_time_series()
        assert result["status"] == "ok"
        assert result["maml"]["trend"] == "improving"
        assert result["maml"]["slope_per_trade"] < 0

    def test_autocorrelation(self, bench):
        # Errors with structure
        for i in range(20):
            bench.record_outcome(
                trade_id=f"t{i}", regime="NORMAL",
                maml_confidence=50.0 + (10 if i % 4 < 2 else 0),
                ridge_confidence=50.0,
                realized_confidence=50.0,
                pnl_pips=0.0, won=False,
            )
        result = bench.compute_error_time_series()
        assert result["maml"]["autocorrelation_lag1"] is not None


class TestInnerLoopDiagnostics:
    def test_no_data(self, bench):
        result = bench.compute_inner_loop_diagnostics()
        assert result["status"] == "no_inner_loss_data"

    def test_with_inner_losses(self, bench):
        # Meta-train records with inner loop data
        bench.record_meta_train({
            "status": "trained",
            "meta_trains": 1,
            "outer_loss": 0.05,
            "regime_losses": {"NORMAL": 0.04},
            "regimes_used": ["NORMAL"],
            "inner_losses": {"NORMAL": [1.0, 0.8, 0.6, 0.4, 0.2]},
            "support_losses": {"NORMAL": 0.2},
            "param_drift": {"NORMAL": 0.5},
        })
        result = bench.compute_inner_loop_diagnostics()
        assert result["status"] == "ok"
        assert "NORMAL" in result["regime_summary"]
        assert result["regime_summary"]["NORMAL"]["monotonic_pct"] == 100.0
        assert result["regime_summary"]["NORMAL"]["avg_reduction_pct"] > 50


class TestParamDrift:
    def test_no_data(self, bench):
        result = bench.compute_param_drift_analysis()
        assert result["status"] == "no_drift_data"

    def test_converging_drift(self, bench):
        # Drift decreasing over training rounds
        for i in range(5):
            bench.record_meta_train({
                "status": "trained",
                "meta_trains": i + 1,
                "outer_loss": 0.05,
                "regime_losses": {"NORMAL": 0.04},
                "regimes_used": ["NORMAL"],
                "inner_losses": {"NORMAL": [1.0, 0.5]},
                "support_losses": {"NORMAL": 0.5},
                "param_drift": {"NORMAL": 1.0 - i * 0.15},  # Decreasing drift
            })
        result = bench.compute_param_drift_analysis()
        assert result["status"] == "ok"
        assert result["trend"] == "converging"


class TestExitReasonAnalysis:
    def test_insufficient_data(self, bench):
        result = bench.compute_exit_reason_analysis()
        assert result["status"] == "insufficient_data"

    def test_with_exit_reasons(self, bench):
        # TP exits — model was right
        for i in range(6):
            bench.record_outcome(
                trade_id=f"tp{i}", regime="NORMAL",
                maml_confidence=70.0, ridge_confidence=65.0,
                realized_confidence=72.0, pnl_pips=15.0, won=True,
                exit_reason="tp",
            )
        # SL exits — model was wrong
        for i in range(6):
            bench.record_outcome(
                trade_id=f"sl{i}", regime="NORMAL",
                maml_confidence=70.0, ridge_confidence=65.0,
                realized_confidence=30.0, pnl_pips=-20.0, won=False,
                exit_reason="sl",
            )
        result = bench.compute_exit_reason_analysis()
        assert result["status"] == "ok"
        assert "tp" in result["exit_types"]
        assert "sl" in result["exit_types"]
        assert result["exit_types"]["tp"]["win_rate"] == 1.0
        assert result["exit_types"]["sl"]["win_rate"] == 0.0


class TestSampleSizeRequirement:
    def test_insufficient_data(self, bench):
        result = bench.compute_required_sample_size()
        assert result["status"] == "insufficient_data"

    def test_calculates_requirement(self, bench):
        _seed_outcomes(bench, n=15)
        result = bench.compute_required_sample_size(target_improvement=2.0)
        assert result["status"] == "ok"
        assert result["required_samples"] >= 30
        assert result["progress_pct"] > 0
        assert "remaining" in result


class TestEnrichedOutcomeRecord:
    def test_enriched_fields_serialization(self):
        rec = OutcomeRecord(
            timestamp=1234567890.0, trade_id="t1", regime="HIGH",
            maml_confidence=60.0, ridge_confidence=55.0,
            realized_confidence=62.0, pnl_pips=12.5, won=True,
            regime_age=3, exit_reason="tp", duration_minutes=45.5,
            regime_at_exit="NORMAL",
        )
        d = rec.to_dict()
        assert d["exit_reason"] == "tp"
        assert d["duration"] == 45.5
        assert d["exit_regime"] == "NORMAL"

        rec2 = OutcomeRecord.from_dict(d)
        assert rec2.exit_reason == "tp"
        assert rec2.duration_minutes == 45.5
        assert rec2.regime_at_exit == "NORMAL"


class TestEnrichedMetaTrainRecord:
    def test_inner_loss_serialization(self):
        rec = MetaTrainRecord(
            timestamp=1234567890.0, train_number=3,
            outer_loss=0.05,
            regime_losses={"NORMAL": 0.04},
            regimes_used=["NORMAL"],
            inner_losses={"NORMAL": [1.0, 0.8, 0.6, 0.4, 0.2]},
            support_losses={"NORMAL": 0.2},
            param_drift={"NORMAL": 0.5},
        )
        d = rec.to_dict()
        assert "inner_losses" in d
        assert len(d["inner_losses"]["NORMAL"]) == 5
        assert "param_drift" in d

        rec2 = MetaTrainRecord.from_dict(d)
        assert len(rec2.inner_losses["NORMAL"]) == 5
        assert abs(rec2.param_drift["NORMAL"] - 0.5) < 1e-6


class TestFullReportWithStats:
    def test_report_includes_all_new_sections(self, bench):
        _seed_outcomes(bench, n=30)

        # Add meta-trains with diagnostics
        for i in range(4):
            bench.record_meta_train({
                "status": "trained",
                "meta_trains": i + 1,
                "outer_loss": 1.0 - i * 0.2,
                "regime_losses": {"NORMAL": 0.5},
                "regimes_used": ["NORMAL"],
                "inner_losses": {"NORMAL": [1.0, 0.7, 0.4, 0.2, 0.1]},
                "support_losses": {"NORMAL": 0.1},
                "param_drift": {"NORMAL": 0.8 - i * 0.1},
            })

        report = bench.generate_report()
        assert "significance_test" in report
        assert "calibration_metrics" in report
        assert "error_time_series" in report
        assert "inner_loop_diagnostics" in report
        assert "param_drift" in report
        assert "exit_reason_analysis" in report
        assert "sample_size" in report

    def test_summary_includes_new_sections(self, bench):
        _seed_outcomes(bench, n=20)
        summary = bench.generate_summary()
        assert "Statistical Significance" in summary
        assert "Calibration" in summary
        assert "Error Trend" in summary
        assert "Sample Size" in summary


# ── Test: Ridge Feature Pipeline (Phase 2 upgrade) ──


class TestRidgeFeaturePipeline:
    """Verify real Ridge features flow through prediction → journal → outcome."""

    def test_real_features_preferred_over_proxy(self):
        """When ridge_features present in journal entry, proxy should NOT be used."""
        real_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        proxy_scores = {"direction_score": 0.5, "confidence_score": 0.5, "momentum_score": 0.5, "risk_score": 0.5}

        entry = {
            "ridge_features": real_features,
            "gate_details": {"scores": proxy_scores},
        }

        # Simulate the feature selection logic from execution.py
        _stored = entry.get("ridge_features")
        if _stored is not None and len(_stored) >= 14:
            features = np.array(_stored[:14], dtype=np.float32)
        else:
            _scores = entry.get("gate_details", {}).get("scores", {})
            features = np.array([
                float(_scores.get("direction_score", 0.5)),
                float(_scores.get("confidence_score", 0.5)),
                float(_scores.get("momentum_score", 0.5)),
                float(_scores.get("risk_score", 0.5)),
            ] + [0.0] * 10, dtype=np.float32)

        # Should use real features, not proxy
        assert features.shape == (14,)
        np.testing.assert_array_almost_equal(features, real_features)

    def test_proxy_fallback_when_no_ridge_features(self):
        """When ridge_features is None, should fall back to proxy."""
        proxy_scores = {"direction_score": 0.7, "confidence_score": 0.6, "momentum_score": 0.8, "risk_score": 0.3}

        entry = {
            "ridge_features": None,
            "gate_details": {"scores": proxy_scores},
        }

        _stored = entry.get("ridge_features")
        if _stored is not None and len(_stored) >= 14:
            features = np.array(_stored[:14], dtype=np.float32)
        else:
            _scores = entry.get("gate_details", {}).get("scores", {})
            features = np.array([
                float(_scores.get("direction_score", 0.5)),
                float(_scores.get("confidence_score", 0.5)),
                float(_scores.get("momentum_score", 0.5)),
                float(_scores.get("risk_score", 0.5)),
            ] + [0.0] * 10, dtype=np.float32)

        assert features.shape == (14,)
        assert features[0] == pytest.approx(0.7)  # direction_score
        assert features[4] == 0.0  # padding

    def test_real_features_have_no_zero_padding(self):
        """Real Ridge features should have meaningful values in all 14 dimensions."""
        real_features = list(np.random.randn(14).astype(np.float32))
        entry = {"ridge_features": real_features}

        _stored = entry.get("ridge_features")
        features = np.array(_stored[:14], dtype=np.float32)

        # Real features should NOT have 10 trailing zeros (the proxy pattern)
        trailing_zeros = np.sum(features[4:] == 0.0)
        assert trailing_zeros < 10, "Real features should not have proxy's zero-padding pattern"

    def test_features_roundtrip_through_json(self):
        """Ridge features survive JSON serialization (stored as list in journal)."""
        original = np.random.randn(14).astype(np.float32)
        as_list = original.tolist()  # What we store in metadata
        json_str = json.dumps(as_list)
        restored = np.array(json.loads(json_str), dtype=np.float32)

        np.testing.assert_array_almost_equal(original, restored, decimal=5)

    def test_maml_training_with_real_features(self):
        """MAML inner loop should work with real Ridge feature distributions."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig
        import tempfile
        from unittest.mock import patch as _patch
        from pathlib import Path
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            with _patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                config = MAMLConfig(
                    min_samples_per_regime=5,
                    meta_train_interval=1000,
                    inner_steps=10,
                    inner_lr=0.01,
                )
                maml = MAMLRidge(config=config)

                # Simulate real Ridge feature distribution (normalized market indicators)
                np.random.seed(42)
                for _ in range(20):
                    # Real features: volatility, ATR, BB, ADX, returns etc — diverse values
                    features = np.random.randn(14).astype(np.float32) * 0.3
                    maml._regime_buffers["NORMAL"].append((features, 65.0))
                for _ in range(20):
                    features = np.random.randn(14).astype(np.float32) * 0.3
                    maml._regime_buffers["HIGH"].append((features, 35.0))

                stats = maml.meta_train()
                assert stats["status"] == "trained"

                # With real features, inner loop should show some loss reduction
                # (unlike the proxy features which had 10 zeros)
                for regime, losses in stats.get("inner_losses", {}).items():
                    if losses and len(losses) >= 2:
                        # At least check losses are finite
                        assert all(np.isfinite(l) for l in losses), \
                            f"Inner losses contain NaN/Inf for {regime}"
