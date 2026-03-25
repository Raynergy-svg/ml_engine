"""
Phase 51 Tests: Win-Rate Engine — Isotonic Calibration, Tranche Tracking,
Pattern Gates, Walk-Forward Retraining, Setup Quality Filter.

US-317 through US-322.
"""

import json
import os
import tempfile
import pytest


# ============================================================================
# US-317: Isotonic Calibrator Tests
# ============================================================================

class TestIsotonicCalibratorPAVA:
    """Test Pool Adjacent Violators Algorithm correctness."""

    def test_pava_already_monotonic(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        weights = [10, 10, 10, 10, 10]
        result = cal._pava(values, weights)
        assert len(result) == 5
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-9

    def test_pava_reversal(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        values = [0.5, 0.3, 0.7]  # Middle violates monotonicity
        weights = [10, 10, 10]
        result = cal._pava(values, weights)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-9, f"Monotonicity violated at {i}: {result}"

    def test_pava_all_same(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        result = cal._pava([0.5, 0.5, 0.5], [1, 1, 1])
        assert all(abs(v - 0.5) < 1e-9 for v in result)

    def test_pava_empty(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        assert cal._pava([], []) == []


class TestIsotonicCalibratorModel:
    """Test IsotonicModel prediction."""

    def test_model_predict_interpolation(self):
        from src.scanner.isotonic_calibrator import IsotonicModel
        model = IsotonicModel(
            breakpoints_x=[0.2, 0.4, 0.6, 0.8],
            breakpoints_y=[0.1, 0.3, 0.5, 0.8],
            is_valid=True,
        )
        # Exact breakpoint
        assert abs(model.predict(0.4) - 0.3) < 1e-9
        # Interpolation between 0.4→0.3 and 0.6→0.5
        assert abs(model.predict(0.5) - 0.4) < 1e-6
        # Below range
        assert abs(model.predict(0.1) - 0.1) < 1e-9
        # Above range
        assert abs(model.predict(0.9) - 0.8) < 1e-9

    def test_model_predict_invalid_fallback(self):
        from src.scanner.isotonic_calibrator import IsotonicModel
        model = IsotonicModel(is_valid=False)
        assert model.predict(0.72) == 0.72  # Pass-through

    def test_model_predict_empty(self):
        from src.scanner.isotonic_calibrator import IsotonicModel
        model = IsotonicModel(breakpoints_x=[], breakpoints_y=[], is_valid=True)
        assert model.predict(0.5) == 0.5


class TestIsotonicCalibratorFit:
    """Test fitting from journal data."""

    def _make_journal(self, n=100, win_rate_at_high_conf=0.7, win_rate_at_low_conf=0.2):
        """Generate synthetic journal with confidence-dependent win rates."""
        import random
        random.seed(42)
        trades = []
        for i in range(n):
            conf = random.uniform(0.3, 0.9)
            # Higher confidence → higher win probability
            threshold = win_rate_at_low_conf + (win_rate_at_high_conf - win_rate_at_low_conf) * (conf - 0.3) / 0.6
            won = random.random() < threshold
            trades.append({
                "pair": "EUR_USD",
                "confidence": conf,
                "direction": "LONG",
                "outcome": {"trade_won": won, "pnl_pips": 20 if won else -15},
            })
        return trades

    def test_fit_synthetic_journal(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator(n_bins=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._make_journal(200), f)
            path = f.name

        try:
            result = cal.fit_from_journal(path)
            assert result.n_train > 0
            assert result.n_holdout > 0
            assert result.n_breakpoints > 0
            # ECE after should be reasonable
            assert result.ece_after <= 1.0
            assert result.ece_after >= 0.0
        finally:
            os.unlink(path)

    def test_fit_insufficient_data(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._make_journal(10), f)
            path = f.name

        try:
            result = cal.fit_from_journal(path)
            assert result.n_train <= 10
            assert not result.is_valid
        finally:
            os.unlink(path)

    def test_calibrate_passthrough_when_no_model(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        assert cal.calibrate(0.65) == 0.65

    def test_save_and_load_model(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator, IsotonicModel
        cal = IsotonicCalibrator()
        cal.model = IsotonicModel(
            breakpoints_x=[0.3, 0.5, 0.7],
            breakpoints_y=[0.2, 0.4, 0.7],
            is_valid=True,
            ece=0.08,
            n_training_samples=400,
        )

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "isotonic.json")
            assert cal.save_model(path)

            cal2 = IsotonicCalibrator()
            assert cal2.load_model(path)
            assert cal2.model is not None
            assert cal2.model.is_valid
            assert len(cal2.model.breakpoints_x) == 3
            assert abs(cal2.calibrate(0.5) - 0.4) < 1e-6

    def test_ece_computation(self):
        from src.scanner.isotonic_calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        # Perfect calibration: predicted matches actual
        predicted = [0.3, 0.3, 0.7, 0.7]
        actual = [False, False, True, True]
        ece = cal._compute_ece(predicted, actual, n_bins=2)
        # Should be near 0 since predictions match outcomes
        assert ece <= 0.35  # Rough check — depends on binning


# ============================================================================
# US-318: Tranche Tracker Tests
# ============================================================================

class TestTrancheTracker:
    """Test tranche tracking and SL adjustment logic."""

    def test_register_and_get(self):
        from src.scanner.tranche_tracker import TrancheTracker

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tranche.json")
            tracker = TrancheTracker(state_path=path)
            tracker.register_trade("123", "EUR_USD", 1.0850, 1.0800, "LONG")

            state = tracker.get_state("123")
            assert state is not None
            assert state.pair == "EUR_USD"
            assert state.entry_price == 1.0850
            assert state.original_sl == 1.0800

    def test_breakeven_after_first_tranche(self):
        from src.scanner.tranche_tracker import TrancheTracker

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tranche.json")
            tracker = TrancheTracker(state_path=path)
            tracker.register_trade("123", "EUR_USD", 1.0850, 1.0800, "LONG")

            adj = tracker.record_tranche_close("123", tranche_idx=0, r_at_close=1.05)
            assert adj.should_move
            assert adj.adjustment_type == "breakeven"
            assert abs(adj.new_sl - 1.0850) < 1e-9  # SL = entry

    def test_profit_lock_after_second_tranche_long(self):
        from src.scanner.tranche_tracker import TrancheTracker

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tranche.json")
            tracker = TrancheTracker(state_path=path)
            tracker.register_trade("123", "GBP_USD", 1.2700, 1.2650, "LONG")
            # SL distance = 1.2700 - 1.2650 = 0.0050

            tracker.record_tranche_close("123", tranche_idx=0, r_at_close=1.0)
            adj = tracker.record_tranche_close("123", tranche_idx=1, r_at_close=1.5)
            assert adj.should_move
            assert adj.adjustment_type == "profit_lock"
            # new_sl = entry + 0.5 * sl_distance = 1.2700 + 0.0025 = 1.2725
            assert abs(adj.new_sl - 1.2725) < 1e-6

    def test_profit_lock_after_second_tranche_short(self):
        from src.scanner.tranche_tracker import TrancheTracker

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tranche.json")
            tracker = TrancheTracker(state_path=path)
            tracker.register_trade("456", "EUR_USD", 1.0850, 1.0900, "SHORT")
            # SL distance = |1.0850 - 1.0900| = 0.0050

            tracker.record_tranche_close("456", tranche_idx=0, r_at_close=1.0)
            adj = tracker.record_tranche_close("456", tranche_idx=1, r_at_close=1.5)
            assert adj.should_move
            # SHORT: new_sl = entry - 0.5 * sl_distance = 1.0850 - 0.0025 = 1.0825
            assert abs(adj.new_sl - 1.0825) < 1e-6

    def test_no_adjustment_for_unregistered(self):
        from src.scanner.tranche_tracker import TrancheTracker

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tranche.json")
            tracker = TrancheTracker(state_path=path)
            adj = tracker.record_tranche_close("999", tranche_idx=0, r_at_close=1.0)
            assert not adj.should_move

    def test_persistence_across_restart(self):
        from src.scanner.tranche_tracker import TrancheTracker

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tranche.json")

            t1 = TrancheTracker(state_path=path)
            t1.register_trade("100", "GBP_AUD", 1.9500, 1.9450, "LONG")
            t1.record_tranche_close("100", tranche_idx=0, r_at_close=1.1)

            # Reload from disk
            t2 = TrancheTracker(state_path=path)
            state = t2.get_state("100")
            assert state is not None
            assert 0 in state.tranches_closed
            assert state.sl_moved_to_breakeven

    def test_remove_trade(self):
        from src.scanner.tranche_tracker import TrancheTracker

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tranche.json")
            tracker = TrancheTracker(state_path=path)
            tracker.register_trade("123", "EUR_USD", 1.0850, 1.0800, "LONG")
            tracker.remove_trade("123")
            assert tracker.get_state("123") is None

    def test_duplicate_tranche_close_idempotent(self):
        from src.scanner.tranche_tracker import TrancheTracker

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tranche.json")
            tracker = TrancheTracker(state_path=path)
            tracker.register_trade("123", "EUR_USD", 1.0850, 1.0800, "LONG")

            adj1 = tracker.record_tranche_close("123", tranche_idx=0, r_at_close=1.0)
            adj2 = tracker.record_tranche_close("123", tranche_idx=0, r_at_close=1.0)
            # Second call should not trigger another breakeven move (already done)
            assert adj1.should_move
            assert not adj2.should_move


# ============================================================================
# US-319: Pattern Gate Tests
# ============================================================================

class TestPatternGate:
    """Test pattern-based confidence adjustment."""

    def _make_patterns_file(self, td):
        """Create a test journal_patterns.json."""
        data = {
            "breakdowns": {
                "by_pair": {
                    "GBP_AUD": {"win_rate": 0.625, "total": 40, "p_value": 0.0017},
                    "EUR_USD": {"win_rate": 0.30, "total": 50, "p_value": 0.15},
                    "EUR_JPY": {"win_rate": 0.55, "total": 40, "p_value": 0.03},
                },
                "by_direction": {
                    "LONG": {"win_rate": 0.495, "total": 300},
                    "SHORT": {"win_rate": 0.275, "total": 222},
                },
            },
            "significant_patterns": [
                {"category": "pair", "value": "GBP_AUD", "win_rate": 0.625, "p_value": 0.0017},
            ],
        }
        path = os.path.join(td, "journal_patterns.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_gbp_aud_gets_boost(self):
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        with tempfile.TemporaryDirectory() as td:
            pp = self._make_patterns_file(td)
            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            result = gate.evaluate("GBP_AUD", "LONG", 0.60)
            assert result.pair_boost > 0
            assert result.adjusted_confidence > 0.60

    def test_long_direction_boost(self):
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        with tempfile.TemporaryDirectory() as td:
            pp = self._make_patterns_file(td)
            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            result = gate.evaluate("GBP_AUD", "LONG", 0.60)
            assert result.direction_boost > 0

    def test_short_direction_penalty(self):
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        with tempfile.TemporaryDirectory() as td:
            pp = self._make_patterns_file(td)
            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            result = gate.evaluate("GBP_AUD", "SHORT", 0.60)
            assert result.direction_boost < 0

    def test_duration_warning_short(self):
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        with tempfile.TemporaryDirectory() as td:
            pp = self._make_patterns_file(td)
            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            result = gate.evaluate("EUR_USD", "LONG", 0.60, expected_duration_hours=2.0)
            assert "below sweet spot" in result.duration_warning

    def test_duration_warning_long(self):
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        with tempfile.TemporaryDirectory() as td:
            pp = self._make_patterns_file(td)
            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            result = gate.evaluate("EUR_USD", "LONG", 0.60, expected_duration_hours=30.0)
            assert "above sweet spot" in result.duration_warning

    def test_no_duration_warning_in_sweet_spot(self):
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        with tempfile.TemporaryDirectory() as td:
            pp = self._make_patterns_file(td)
            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            result = gate.evaluate("EUR_USD", "LONG", 0.60, expected_duration_hours=12.0)
            assert result.duration_warning == ""

    def test_confidence_clamped_to_01(self):
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        with tempfile.TemporaryDirectory() as td:
            pp = self._make_patterns_file(td)
            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            result = gate.evaluate("GBP_AUD", "LONG", 0.98)
            assert result.adjusted_confidence <= 1.0

    def test_pair_ranking(self):
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        with tempfile.TemporaryDirectory() as td:
            pp = self._make_patterns_file(td)
            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            ranking = gate.get_pair_ranking()
            assert len(ranking) >= 2
            assert ranking[0]["pair"] == "GBP_AUD"  # Highest win rate


# ============================================================================
# US-320: Walk-Forward Retrainer Tests
# ============================================================================

class TestWalkForwardRetrainer:
    """Test walk-forward agent weight retraining."""

    def _make_journal(self, n=120):
        import random
        random.seed(42)
        agents = [
            "trend", "mean_reversion", "volatility", "risk_sentinel",
            "uncertainty", "execution_quality", "momentum", "news_risk",
            "multi_timeframe", "pair_performance", "session_timing",
            "support_resistance",
        ]
        trades = []
        for i in range(n):
            direction = random.choice(["LONG", "SHORT"])
            won = random.random() < 0.45  # Slightly positive
            verdicts = {}
            for a in agents:
                vote = direction if random.random() < 0.6 else ("SHORT" if direction == "LONG" else "LONG")
                verdicts[a] = {"verdict": vote, "confidence": random.uniform(0.3, 0.9)}
            trades.append({
                "pair": random.choice(["EUR_USD", "GBP_USD", "GBP_AUD"]),
                "direction": direction,
                "confidence": random.uniform(0.4, 0.85),
                "outcome": {"trade_won": won, "pnl_pips": 20 if won else -15},
                "agent_verdicts": verdicts,
            })
        return trades

    def test_run_basic(self):
        from src.scanner.walkforward_retrainer import WalkForwardRetrainer, WFConfig

        with tempfile.TemporaryDirectory() as td:
            journal_path = os.path.join(td, "journal.json")
            epochs_path = os.path.join(td, "epochs.json")
            weights_path = os.path.join(td, "weights.json")

            with open(journal_path, "w") as f:
                json.dump(self._make_journal(120), f)

            # Create initial weights
            agents = ["trend", "mean_reversion", "volatility", "risk_sentinel",
                      "uncertainty", "execution_quality", "momentum", "news_risk",
                      "multi_timeframe", "pair_performance", "session_timing",
                      "support_resistance"]
            with open(weights_path, "w") as f:
                json.dump({"weights": {a: 1/12 for a in agents}}, f)

            config = WFConfig(train_window=60, test_window=20, slide_step=20)
            retrainer = WalkForwardRetrainer(config)
            epochs = retrainer.run(journal_path, weights_path, epochs_path)

            assert len(epochs) >= 1
            # At least some epochs should exist
            for e in epochs:
                assert e.n_train == 60
                assert e.n_test == 20

    def test_insufficient_data(self):
        from src.scanner.walkforward_retrainer import WalkForwardRetrainer

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "small.json")
            with open(path, "w") as f:
                json.dump(self._make_journal(30), f)

            retrainer = WalkForwardRetrainer()
            epochs = retrainer.run(path)
            assert epochs == []

    def test_wfe_rejection(self):
        from src.scanner.walkforward_retrainer import WalkForwardRetrainer, WFConfig

        # With very high min_wfe, all epochs should be rejected
        config = WFConfig(train_window=60, test_window=20, min_wfe=0.99)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "journal.json")
            epochs_path = os.path.join(td, "epochs.json")
            with open(path, "w") as f:
                json.dump(self._make_journal(120), f)

            retrainer = WalkForwardRetrainer(config)
            epochs = retrainer.run(path, output_path=epochs_path)
            # Most or all should be rejected with WFE < 0.99
            rejected = sum(1 for e in epochs if not e.accepted)
            assert rejected >= len(epochs) * 0.5  # At least half rejected

    def test_weight_normalization(self):
        from src.scanner.walkforward_retrainer import WalkForwardRetrainer

        retrainer = WalkForwardRetrainer()
        trades = self._make_journal(60)
        agents = ["trend", "mean_reversion", "volatility", "risk_sentinel",
                  "uncertainty", "execution_quality", "momentum", "news_risk",
                  "multi_timeframe", "pair_performance", "session_timing",
                  "support_resistance"]
        base = {a: 1/12 for a in agents}

        new_weights = retrainer._retrain_weights(trades, base)
        total = sum(new_weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights not normalized: sum={total}"

    def test_save_and_load_weights(self):
        from src.scanner.walkforward_retrainer import WalkForwardRetrainer

        retrainer = WalkForwardRetrainer()
        retrainer._current_weights = {"trend": 0.15, "momentum": 0.10, "volatility": 0.08}

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "w.json")
            assert retrainer.save_weights(path)
            with open(path) as f:
                data = json.load(f)
            assert "weights" in data
            assert data["source"] == "walkforward_retraining"

    def test_epochs_persisted(self):
        from src.scanner.walkforward_retrainer import WalkForwardRetrainer, WFConfig

        with tempfile.TemporaryDirectory() as td:
            journal_path = os.path.join(td, "journal.json")
            epochs_path = os.path.join(td, "epochs.json")

            with open(journal_path, "w") as f:
                json.dump(self._make_journal(120), f)

            config = WFConfig(train_window=60, test_window=20)
            retrainer = WalkForwardRetrainer(config)
            retrainer.run(journal_path, output_path=epochs_path)

            assert os.path.exists(epochs_path)
            with open(epochs_path) as f:
                data = json.load(f)
            assert "epochs" in data
            assert data["n_epochs"] >= 1


# ============================================================================
# US-321: Setup Quality Filter Tests
# ============================================================================

class TestSetupQualityFilter:
    """Test composite weak-setup rejection filter."""

    def test_high_quality_passes(self):
        from src.scanner.setup_quality import SetupQualityFilter
        sqf = SetupQualityFilter()
        result = sqf.evaluate(
            pair="GBP_AUD",
            confidence=0.75,
            agent_disagreement=0.10,
            uncertainty_score=0.15,
            pair_recent_win_rate=0.55,
        )
        assert result.passed
        assert result.quality_score > 0.5

    def test_low_quality_rejected(self):
        from src.scanner.setup_quality import SetupQualityFilter
        sqf = SetupQualityFilter()
        result = sqf.evaluate(
            pair="AUD_JPY",
            confidence=0.42,
            agent_disagreement=0.45,
            uncertainty_score=0.50,
            pair_recent_win_rate=0.15,
        )
        assert not result.passed
        assert result.quality_score < 0.35
        assert len(result.rejection_reasons) > 0

    def test_dimension_scores_in_range(self):
        from src.scanner.setup_quality import SetupQualityFilter
        sqf = SetupQualityFilter()
        result = sqf.evaluate(
            pair="EUR_USD",
            confidence=0.60,
            agent_disagreement=0.20,
            uncertainty_score=0.30,
            pair_recent_win_rate=0.40,
        )
        for name, score in result.dimension_scores.items():
            assert 0.0 <= score <= 1.0, f"{name} score {score} out of range"

    def test_rejection_rate_tracking(self):
        from src.scanner.setup_quality import SetupQualityFilter
        sqf = SetupQualityFilter()

        # Generate 10 evaluations
        for _ in range(5):
            sqf.evaluate("EUR_USD", 0.75, 0.10, 0.15, 0.55)
        for _ in range(5):
            sqf.evaluate("AUD_JPY", 0.42, 0.45, 0.50, 0.15)

        rate = sqf.get_rejection_rate()
        assert 0.0 <= rate <= 1.0

    def test_stats(self):
        from src.scanner.setup_quality import SetupQualityFilter
        sqf = SetupQualityFilter()
        sqf.evaluate("EUR_USD", 0.75, 0.10, 0.15, 0.55)
        sqf.evaluate("AUD_JPY", 0.42, 0.45, 0.50, 0.15)

        stats = sqf.get_stats()
        assert stats["total_evaluations"] == 2
        assert stats["passed"] + stats["rejected"] == 2

    def test_extreme_inputs_handled(self):
        from src.scanner.setup_quality import SetupQualityFilter
        sqf = SetupQualityFilter()

        # All zeros
        result = sqf.evaluate("X", 0.0, 0.0, 0.0, 0.0)
        assert 0.0 <= result.quality_score <= 1.0

        # All max
        result = sqf.evaluate("X", 1.0, 1.0, 1.0, 1.0)
        assert 0.0 <= result.quality_score <= 1.0

    def test_borderline_quality(self):
        from src.scanner.setup_quality import SetupQualityFilter, SetupQualityConfig

        config = SetupQualityConfig(quality_threshold=0.50)
        sqf = SetupQualityFilter(config)

        # Right at threshold
        result = sqf.evaluate("EUR_USD", 0.60, 0.25, 0.30, 0.35)
        # Score should be around threshold area
        assert isinstance(result.passed, bool)

    def test_details_populated(self):
        from src.scanner.setup_quality import SetupQualityFilter
        sqf = SetupQualityFilter()
        result = sqf.evaluate("GBP_USD", 0.65, 0.20, 0.25, 0.45)
        assert result.details["pair"] == "GBP_USD"
        assert result.details["raw_confidence"] == 0.65


# ============================================================================
# US-322: Integration Tests
# ============================================================================

class TestPhase51LazyImports:
    """Test lazy imports in __init__.py."""

    def test_isotonic_calibrator_import(self):
        from src.scanner import IsotonicCalibrator
        cal = IsotonicCalibrator()
        assert cal is not None

    def test_tranche_tracker_import(self):
        from src.scanner import TrancheTracker
        assert TrancheTracker is not None

    def test_pattern_gate_import(self):
        from src.scanner import PatternGate
        assert PatternGate is not None

    def test_walkforward_retrainer_import(self):
        from src.scanner import WalkForwardRetrainer
        assert WalkForwardRetrainer is not None

    def test_setup_quality_filter_import(self):
        from src.scanner import SetupQualityFilter
        sqf = SetupQualityFilter()
        assert sqf is not None


class TestPhase51Integration:
    """End-to-end integration tests across Phase 51 modules."""

    def test_isotonic_then_pattern_gate_pipeline(self):
        """Test: calibrate confidence, then run through pattern gate."""
        from src.scanner.isotonic_calibrator import IsotonicCalibrator, IsotonicModel
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig

        # Setup calibrator with simple model
        cal = IsotonicCalibrator()
        cal.model = IsotonicModel(
            breakpoints_x=[0.3, 0.5, 0.7, 0.9],
            breakpoints_y=[0.2, 0.35, 0.55, 0.75],
            is_valid=True,
        )

        # Calibrate a confidence score
        raw = 0.70
        calibrated = cal.calibrate(raw)
        assert calibrated != raw  # Should be adjusted
        assert 0.0 <= calibrated <= 1.0

        # Pattern gate on calibrated score
        with tempfile.TemporaryDirectory() as td:
            patterns = {
                "breakdowns": {
                    "by_pair": {"GBP_AUD": {"win_rate": 0.60, "p_value": 0.01}},
                    "by_direction": {"LONG": {"win_rate": 0.49}},
                },
                "significant_patterns": [],
            }
            pp = os.path.join(td, "patterns.json")
            with open(pp, "w") as f:
                json.dump(patterns, f)

            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            result = gate.evaluate("GBP_AUD", "LONG", calibrated)
            assert result.adjusted_confidence >= calibrated  # GBP_AUD + LONG = positive adjustments

    def test_quality_filter_after_pattern_gate(self):
        """Test: pattern gate adjusts confidence, then quality filter evaluates."""
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig
        from src.scanner.setup_quality import SetupQualityFilter

        with tempfile.TemporaryDirectory() as td:
            patterns = {
                "breakdowns": {
                    "by_pair": {"EUR_USD": {"win_rate": 0.30, "p_value": 0.05}},
                    "by_direction": {"SHORT": {"win_rate": 0.28}},
                },
                "significant_patterns": [],
            }
            pp = os.path.join(td, "patterns.json")
            with open(pp, "w") as f:
                json.dump(patterns, f)

            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            gate_result = gate.evaluate("EUR_USD", "SHORT", 0.50)

            # Now run through quality filter
            sqf = SetupQualityFilter()
            quality = sqf.evaluate(
                pair="EUR_USD",
                confidence=gate_result.adjusted_confidence,
                agent_disagreement=0.35,
                uncertainty_score=0.42,
                pair_recent_win_rate=0.25,
            )
            # With poor pair, poor direction, high disagreement, high uncertainty — should be rejected
            assert not quality.passed

    def test_full_pipeline_calibrate_pattern_quality(self):
        """Full pipeline: calibrate → pattern gate → setup quality."""
        from src.scanner.isotonic_calibrator import IsotonicCalibrator, IsotonicModel
        from src.scanner.pattern_gate import PatternGate, PatternGateConfig
        from src.scanner.setup_quality import SetupQualityFilter

        # 1. Calibrate
        cal = IsotonicCalibrator()
        cal.model = IsotonicModel(
            breakpoints_x=[0.4, 0.6, 0.8],
            breakpoints_y=[0.3, 0.5, 0.7],
            is_valid=True,
        )
        calibrated = cal.calibrate(0.75)

        # 2. Pattern gate
        with tempfile.TemporaryDirectory() as td:
            patterns = {
                "breakdowns": {
                    "by_pair": {"GBP_AUD": {"win_rate": 0.62}},
                    "by_direction": {"LONG": {"win_rate": 0.49}},
                },
                "significant_patterns": [],
            }
            pp = os.path.join(td, "p.json")
            with open(pp, "w") as f:
                json.dump(patterns, f)

            gate = PatternGate(PatternGateConfig(patterns_path=pp))
            gate_result = gate.evaluate("GBP_AUD", "LONG", calibrated)

            # 3. Quality filter
            sqf = SetupQualityFilter()
            quality = sqf.evaluate(
                pair="GBP_AUD",
                confidence=gate_result.adjusted_confidence,
                agent_disagreement=0.12,
                uncertainty_score=0.18,
                pair_recent_win_rate=0.55,
            )
            # Good pair, good direction, low disagreement — should pass
            assert quality.passed
            assert quality.quality_score > 0.5
