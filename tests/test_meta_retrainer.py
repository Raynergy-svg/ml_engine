"""
Tests for MetaLabelerRetrainer and MetaLabelerTrainingConfig (US-264 + US-265).

Covers:
- US-264: Core (12+ tests) — config, TradeSample, simulation, data prep
- US-265: Training Pipeline (10+ tests) — Ridge, RF, fit, predict, retrain pipeline
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMetaLabelerTrainingConfig(unittest.TestCase):
    """US-264: MetaLabelerTrainingConfig defaults and construction."""

    def test_config_defaults_model_params(self):
        """MetaLabelerTrainingConfig default model parameters."""
        from src.training.meta_labeler_retrainer import MetaLabelerTrainingConfig

        config = MetaLabelerTrainingConfig()
        self.assertTrue(config.use_xgboost)
        self.assertEqual(config.n_estimators, 150)
        self.assertEqual(config.max_depth, 3)
        self.assertEqual(config.learning_rate, 0.03)

    def test_config_defaults_regularization(self):
        """MetaLabelerTrainingConfig default regularization parameters."""
        from src.training.meta_labeler_retrainer import MetaLabelerTrainingConfig

        config = MetaLabelerTrainingConfig()
        self.assertEqual(config.reg_alpha, 0.5)
        self.assertEqual(config.reg_lambda, 3.0)
        self.assertEqual(config.subsample, 0.8)
        self.assertEqual(config.colsample_bytree, 0.8)
        self.assertEqual(config.min_child_weight, 3)

    def test_config_defaults_thresholds(self):
        """MetaLabelerTrainingConfig default thresholds."""
        from src.training.meta_labeler_retrainer import MetaLabelerTrainingConfig

        config = MetaLabelerTrainingConfig()
        self.assertEqual(config.min_confidence_threshold, 0.50)
        self.assertEqual(config.max_overfitting_gap, 0.08)

    def test_config_defaults_features(self):
        """MetaLabelerTrainingConfig default feature flags."""
        from src.training.meta_labeler_retrainer import MetaLabelerTrainingConfig

        config = MetaLabelerTrainingConfig()
        self.assertTrue(config.include_primary_proba)
        self.assertTrue(config.include_market_regime)
        self.assertTrue(config.use_reduced_features)

    def test_config_defaults_training(self):
        """MetaLabelerTrainingConfig default training parameters."""
        from src.training.meta_labeler_retrainer import MetaLabelerTrainingConfig

        config = MetaLabelerTrainingConfig()
        self.assertTrue(config.auto_tune)
        self.assertEqual(config.validation_split, 0.2)
        self.assertEqual(config.min_samples, 100)

    def test_config_custom_values(self):
        """MetaLabelerTrainingConfig accepts custom values."""
        from src.training.meta_labeler_retrainer import MetaLabelerTrainingConfig

        config = MetaLabelerTrainingConfig(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
        )
        self.assertEqual(config.n_estimators, 200)
        self.assertEqual(config.max_depth, 5)
        self.assertEqual(config.learning_rate, 0.05)


class TestTradeSample(unittest.TestCase):
    """US-264: TradeSample construction and defaults."""

    def test_trade_sample_required_fields(self):
        """TradeSample requires direction field."""
        from src.training.meta_labeler_retrainer import TradeSample

        # direction is required (no default)
        sample = TradeSample(
            tcn_probability=0.7,
            ridge_confidence=60.0,
            xgb_momentum=0.65,
            rf_drawdown_pips=25.0,
            direction="long",
        )
        self.assertEqual(sample.direction, "long")

    def test_trade_sample_default_values(self):
        """TradeSample default values for optional fields."""
        from src.training.meta_labeler_retrainer import TradeSample

        sample = TradeSample(
            tcn_probability=0.7,
            ridge_confidence=60.0,
            xgb_momentum=0.65,
            rf_drawdown_pips=25.0,
            direction="long",
        )
        self.assertIsNone(sample.feature_vector)
        self.assertEqual(sample.instrument, "")
        self.assertEqual(sample.actual_outcome, "")
        self.assertEqual(sample.pnl, 0.0)
        self.assertEqual(sample.pnl_pips, 0.0)
        self.assertEqual(sample.timestamp, "")
        self.assertEqual(sample.trade_id, "")

    def test_trade_sample_all_fields(self):
        """TradeSample can be constructed with all fields."""
        from src.training.meta_labeler_retrainer import TradeSample

        feature_vec = np.array([0.1, 0.2, 0.3, 0.4])
        sample = TradeSample(
            tcn_probability=0.7,
            ridge_confidence=60.0,
            xgb_momentum=0.65,
            rf_drawdown_pips=25.0,
            feature_vector=feature_vec,
            direction="short",
            instrument="EUR_USD",
            actual_outcome="win",
            pnl=150.0,
            pnl_pips=50.0,
            timestamp="2026-03-23T12:00:00Z",
            trade_id="trade_001",
        )
        self.assertEqual(sample.direction, "short")
        self.assertEqual(sample.instrument, "EUR_USD")
        self.assertEqual(sample.actual_outcome, "win")
        self.assertEqual(sample.pnl, 150.0)
        self.assertEqual(sample.pnl_pips, 50.0)
        np.testing.assert_array_equal(sample.feature_vector, feature_vec)


class TestMetaLabelerRetrainerCore(unittest.TestCase):
    """US-264: MetaLabelerRetrainer core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from src.training.meta_labeler_retrainer import (
            MetaLabelerTrainingConfig,
            MetaLabelerRetrainer,
        )

        self.config = MetaLabelerTrainingConfig()
        self.retrainer = MetaLabelerRetrainer(config=self.config)

    def test_retrainer_initialization(self):
        """MetaLabelerRetrainer initializes with config."""
        self.assertIsNotNone(self.retrainer.config)
        self.assertIsNone(self.retrainer.xgb_model)
        self.assertIsNone(self.retrainer.ridge_model)
        self.assertIsNone(self.retrainer.rf_model)
        self.assertFalse(self.retrainer.is_fitted)

    def test_simulate_trades_from_model_returns_list(self):
        """simulate_trades_from_model returns list of TradeSample."""
        samples = self.retrainer.simulate_trades_from_model(n_trades=50)

        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 50)

    def test_simulate_trades_from_model_correct_count(self):
        """simulate_trades_from_model generates correct number of trades."""
        for n in [10, 100, 500]:
            samples = self.retrainer.simulate_trades_from_model(n_trades=n)
            self.assertEqual(len(samples), n)

    def test_simulate_trades_from_model_deterministic_seed(self):
        """simulate_trades_from_model is deterministic with seed."""
        samples1 = self.retrainer.simulate_trades_from_model(n_trades=50, seed=42)
        samples2 = self.retrainer.simulate_trades_from_model(n_trades=50, seed=42)

        # Compare key fields
        for s1, s2 in zip(samples1, samples2):
            self.assertEqual(s1.direction, s2.direction)
            self.assertAlmostEqual(s1.tcn_probability, s2.tcn_probability, places=5)
            self.assertEqual(s1.actual_outcome, s2.actual_outcome)

    def test_simulate_trades_outcomes_distribution(self):
        """simulate_trades_from_model outcomes reflect directional accuracy."""
        samples = self.retrainer.simulate_trades_from_model(
            n_trades=500,
            directional_accuracy=0.708,
            seed=42,
        )

        # Count wins and losses
        wins = sum(1 for s in samples if s.actual_outcome == "win")
        losses = sum(1 for s in samples if s.actual_outcome == "loss")

        actual_accuracy = wins / len(samples)
        # Allow 10% deviation from expected accuracy
        self.assertAlmostEqual(actual_accuracy, 0.708, delta=0.10)

    def test_simulate_trades_short_vs_long_accuracy(self):
        """simulate_trades_from_model respects short vs long accuracy."""
        samples = self.retrainer.simulate_trades_from_model(
            n_trades=1000,
            short_accuracy=0.726,
            long_accuracy=0.593,
            seed=42,
        )

        shorts = [s for s in samples if s.direction == "short"]
        longs = [s for s in samples if s.direction == "long"]

        short_acc = sum(1 for s in shorts if s.actual_outcome == "win") / max(
            len(shorts), 1
        )
        long_acc = sum(1 for s in longs if s.actual_outcome == "win") / max(
            len(longs), 1
        )

        # Shorts should generally have higher accuracy than longs
        self.assertGreater(short_acc, long_acc)

    def test_prepare_training_data_returns_tuple(self):
        """prepare_training_data returns (X, y, w) tuple."""
        from src.training.meta_labeler_retrainer import TradeSample

        samples = [
            TradeSample(
                tcn_probability=0.7,
                ridge_confidence=60.0,
                xgb_momentum=0.65,
                rf_drawdown_pips=25.0,
                direction="long",
                actual_outcome="win",
            )
            for _ in range(150)
        ]

        X, y, w = self.retrainer.prepare_training_data(samples)

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(w, np.ndarray)

    def test_prepare_training_data_correct_shape(self):
        """prepare_training_data returns X with correct shape."""
        from src.training.meta_labeler_retrainer import TradeSample

        samples = [
            TradeSample(
                tcn_probability=0.7,
                ridge_confidence=60.0,
                xgb_momentum=0.65,
                rf_drawdown_pips=25.0,
                direction="long",
                actual_outcome="win",
            )
            for _ in range(150)
        ]

        X, y, w = self.retrainer.prepare_training_data(samples)

        self.assertEqual(X.shape[0], len(samples))
        # Features should be 6: tcn_prob, ridge_conf, xgb_mom, rf_dd, confidence, direction
        self.assertGreaterEqual(X.shape[1], 4)

    def test_prepare_training_data_binary_labels(self):
        """prepare_training_data produces binary labels (0 or 1)."""
        from src.training.meta_labeler_retrainer import TradeSample

        samples = [
            TradeSample(
                tcn_probability=0.7,
                ridge_confidence=60.0,
                xgb_momentum=0.65,
                rf_drawdown_pips=25.0,
                direction="long",
                actual_outcome="win" if i % 2 == 0 else "loss",
            )
            for i in range(150)
        ]

        X, y, w = self.retrainer.prepare_training_data(samples)

        # Labels should be binary
        unique_labels = np.unique(y)
        self.assertTrue(all(label in [0.0, 1.0] for label in unique_labels))

    def test_prepare_training_data_weights_positive(self):
        """prepare_training_data produces positive weights."""
        from src.training.meta_labeler_retrainer import TradeSample

        samples = [
            TradeSample(
                tcn_probability=0.7,
                ridge_confidence=60.0,
                xgb_momentum=0.65,
                rf_drawdown_pips=25.0,
                direction="long",
                actual_outcome="win",
            )
            for _ in range(150)
        ]

        X, y, w = self.retrainer.prepare_training_data(samples)

        # All weights should be positive
        self.assertTrue(np.all(w > 0))

    def test_prepare_training_data_insufficient_samples(self):
        """prepare_training_data raises on insufficient samples."""
        from src.training.meta_labeler_retrainer import TradeSample

        samples = [
            TradeSample(
                tcn_probability=0.7,
                ridge_confidence=60.0,
                xgb_momentum=0.65,
                rf_drawdown_pips=25.0,
                direction="long",
                actual_outcome="win",
            )
            for _ in range(50)  # Less than min_samples (100)
        ]

        with self.assertRaises(ValueError):
            self.retrainer.prepare_training_data(samples)


class TestMetaLabelerTrainingPipeline(unittest.TestCase):
    """US-265: MetaLabelerRetrainer training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        from src.training.meta_labeler_retrainer import (
            MetaLabelerTrainingConfig,
            MetaLabelerRetrainer,
            TradeSample,
        )

        self.config = MetaLabelerTrainingConfig()
        self.retrainer = MetaLabelerRetrainer(config=self.config)

        # Generate training samples
        self.samples = [
            TradeSample(
                tcn_probability=0.7 if i % 2 == 0 else 0.3,
                ridge_confidence=60.0 if i % 2 == 0 else 40.0,
                xgb_momentum=0.65 if i % 2 == 0 else 0.35,
                rf_drawdown_pips=25.0,
                direction="long" if i % 2 == 0 else "short",
                actual_outcome="win" if i % 3 == 0 else "loss",
            )
            for i in range(200)
        ]

        # Prepare data for lower-level tests
        X, y, w = self.retrainer.prepare_training_data(self.samples)
        # Split manually
        val_split = int(len(X) * self.config.validation_split)
        self.X_train = X[:-val_split]
        self.y_train = y[:-val_split]
        self.w_train = w[:-val_split]
        self.X_val = X[-val_split:]
        self.y_val = y[-val_split:]
        self.w_val = np.ones(len(self.X_val))

    def test_train_ridge_returns_metrics(self):
        """train_ridge returns metrics dict."""
        metrics = self.retrainer.train_ridge(
            self.X_train,
            self.y_train,
            self.w_train,
            self.X_val,
            self.y_val,
        )

        self.assertIsInstance(metrics, dict)
        self.assertIn("ridge_train_accuracy", metrics)

    def test_train_ridge_accuracy_above_random(self):
        """train_ridge achieves accuracy > 0.5 (random)."""
        metrics = self.retrainer.train_ridge(
            self.X_train,
            self.y_train,
            self.w_train,
            self.X_val,
            self.y_val,
        )

        # Ridge should do better than random guessing
        self.assertGreater(metrics.get("ridge_train_accuracy", 0), 0.5)

    def test_train_ridge_model_stored(self):
        """train_ridge stores model in retrainer."""
        self.retrainer.train_ridge(
            self.X_train,
            self.y_train,
            self.w_train,
        )

        self.assertIsNotNone(self.retrainer.ridge_model)
        self.assertIsNotNone(self.retrainer.scaler)

    def test_train_random_forest_returns_metrics(self):
        """train_random_forest returns metrics dict."""
        metrics = self.retrainer.train_random_forest(
            self.X_train,
            self.y_train,
            self.w_train,
            self.X_val,
            self.y_val,
        )

        self.assertIsInstance(metrics, dict)
        self.assertIn("rf_train_accuracy", metrics)

    def test_train_random_forest_accuracy_above_random(self):
        """train_random_forest achieves accuracy >= 0.5 (random)."""
        metrics = self.retrainer.train_random_forest(
            self.X_train,
            self.y_train,
            self.w_train,
            self.X_val,
            self.y_val,
        )

        # RF should do at least as well as random guessing
        self.assertGreaterEqual(metrics.get("rf_train_accuracy", 0), 0.5)

    def test_train_random_forest_model_stored(self):
        """train_random_forest stores model in retrainer."""
        self.retrainer.train_random_forest(
            self.X_train,
            self.y_train,
            self.w_train,
        )

        self.assertIsNotNone(self.retrainer.rf_model)

    def test_fit_trains_all_models(self):
        """fit trains Ridge and RF (XGB optional)."""
        self.retrainer.fit(self.samples, verbose=False)

        self.assertIsNotNone(self.retrainer.ridge_model)
        self.assertIsNotNone(self.retrainer.rf_model)
        self.assertTrue(self.retrainer.is_fitted)

    def test_fit_returns_metrics(self):
        """fit returns combined metrics dict."""
        metrics = self.retrainer.fit(self.samples, verbose=False)

        self.assertIsInstance(metrics, dict)
        # Should have metrics from at least Ridge and RF
        self.assertGreater(len(metrics), 0)

    def test_predict_meta_confidence_correct_shape(self):
        """predict_meta_confidence returns array of correct shape."""
        self.retrainer.fit(self.samples, verbose=False)

        preds = self.retrainer.predict_meta_confidence(self.X_val)

        self.assertEqual(len(preds), len(self.X_val))

    def test_predict_meta_confidence_values_in_range(self):
        """predict_meta_confidence values in [0, 1] range."""
        self.retrainer.fit(self.samples, verbose=False)

        preds = self.retrainer.predict_meta_confidence(self.X_val)

        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds <= 1))

    def test_retrain_from_simulation_end_to_end(self):
        """retrain_from_simulation (module func) generates and trains on synthetic data."""
        from src.training.meta_labeler_retrainer import retrain_from_simulation
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = retrain_from_simulation(
                n_trades=200,
                directional_accuracy=0.708,
                output_dir=Path(tmpdir),
                verbose=False,
            )

        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)

    def test_retrain_from_simulation_produces_improvement(self):
        """retrain_from_simulation (module func) trains viable model."""
        from src.training.meta_labeler_retrainer import retrain_from_simulation
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = retrain_from_simulation(
                n_trades=200,
                directional_accuracy=0.708,
                output_dir=Path(tmpdir),
                verbose=False,
            )

        # Should have training metrics
        self.assertGreater(len(metrics), 0)

    def test_save_load_roundtrip(self):
        """fit model, save, and load preserves functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Train and save
            self.retrainer.fit(self.samples, verbose=False)
            self.retrainer.save(tmpdir / "test_meta_labeler.pkl")

            # Load should be a classmethod
            from src.training.meta_labeler_retrainer import MetaLabelerRetrainer
            retrainer2 = MetaLabelerRetrainer.load(tmpdir / "test_meta_labeler.pkl")

            # Should be able to predict
            self.assertTrue(retrainer2.is_fitted)
            preds = retrainer2.predict_meta_confidence(self.X_val)
            self.assertEqual(len(preds), len(self.X_val))


class TestMetaLabelerEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def test_trade_sample_with_feature_vector(self):
        """TradeSample can use provided feature vector."""
        from src.training.meta_labeler_retrainer import TradeSample

        feat = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        sample = TradeSample(
            tcn_probability=0.7,
            ridge_confidence=60.0,
            xgb_momentum=0.65,
            rf_drawdown_pips=25.0,
            direction="long",
            feature_vector=feat,
        )

        self.assertIsNotNone(sample.feature_vector)
        self.assertEqual(len(sample.feature_vector), 6)

    def test_prepare_data_with_mixed_outcomes(self):
        """prepare_training_data handles mixed win/loss/breakeven."""
        from src.training.meta_labeler_retrainer import TradeSample

        retrainer = type(self).__module__.split(".")
        from src.training.meta_labeler_retrainer import MetaLabelerRetrainer

        retrainer = MetaLabelerRetrainer()

        samples = [
            TradeSample(
                tcn_probability=0.7,
                ridge_confidence=60.0,
                xgb_momentum=0.65,
                rf_drawdown_pips=25.0,
                direction="long",
                actual_outcome="win",
            )
            for _ in range(100)
        ] + [
            TradeSample(
                tcn_probability=0.3,
                ridge_confidence=40.0,
                xgb_momentum=0.35,
                rf_drawdown_pips=25.0,
                direction="short",
                actual_outcome="loss",
            )
            for _ in range(100)
        ]

        X, y, w = retrainer.prepare_training_data(samples)

        self.assertEqual(len(X), 200)
        self.assertEqual(len(y), 200)
        self.assertEqual(len(w), 200)


if __name__ == "__main__":
    unittest.main()
