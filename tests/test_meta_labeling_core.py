"""
Tests for MetaLabeling core computation (US-258).

Tests cover:
- MetaLabelingConfig defaults
- MetaLabeler initialization and core methods
- Meta-label generation and feature alignment
- Position sizing logic
"""

import logging
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.training.meta_labeling import (
    MetaLabelingConfig,
    MetaLabeler,
    MetaLabelSample,
)


logger = logging.getLogger(__name__)


class TestMetaLabelingConfig:
    """Tests for MetaLabelingConfig dataclass."""

    def test_config_defaults(self):
        """Test MetaLabelingConfig has expected default values."""
        cfg = MetaLabelingConfig()

        # Meta-model parameters
        assert cfg.use_xgboost is True
        assert cfg.n_estimators == 100
        assert cfg.max_depth == 2
        assert cfg.learning_rate == 0.05

        # Regularization parameters
        assert cfg.reg_alpha == 1.0
        assert cfg.reg_lambda == 5.0
        assert cfg.subsample == 0.7
        assert cfg.colsample_bytree == 0.7
        assert cfg.min_child_weight == 5
        assert cfg.early_stopping_rounds == 30
        assert cfg.max_overfitting_gap == 0.10
        assert cfg.auto_tune is True
        assert cfg.dropout == 0.35

    def test_config_min_confidence_threshold_default(self):
        """Test min_confidence_threshold default is 0.52."""
        cfg = MetaLabelingConfig()
        assert cfg.min_confidence_threshold == 0.52

    def test_config_feature_flags_defaults(self):
        """Test feature inclusion flags have expected defaults."""
        cfg = MetaLabelingConfig()
        assert cfg.include_primary_proba is True
        assert cfg.include_market_regime is True
        assert cfg.include_position_features is True
        assert cfg.use_reduced_features is True
        assert cfg.use_triple_barrier is True

    def test_config_custom_values(self):
        """Test MetaLabelingConfig accepts custom values."""
        cfg = MetaLabelingConfig(
            use_xgboost=False,
            n_estimators=50,
            min_confidence_threshold=0.60,
        )
        assert cfg.use_xgboost is False
        assert cfg.n_estimators == 50
        assert cfg.min_confidence_threshold == 0.60


class TestMetaLabelSample:
    """Tests for MetaLabelSample dataclass."""

    def test_metalabel_sample_construction(self):
        """Test MetaLabelSample can be constructed with all fields."""
        features = np.random.randn(10)
        sample = MetaLabelSample(
            index=0,
            features=features,
            primary_prediction=0.65,
            primary_correct=True,
            actual_pnl=120.5,
        )

        assert sample.index == 0
        assert np.array_equal(sample.features, features)
        assert sample.primary_prediction == 0.65
        assert sample.primary_correct is True
        assert sample.actual_pnl == 120.5


class TestMetaLabelerInit:
    """Tests for MetaLabeler initialization."""

    def test_metalabeler_init_defaults(self):
        """Test MetaLabeler __init__ sets correct defaults."""
        labeler = MetaLabeler()

        assert labeler.meta_model is None
        assert labeler.is_fitted is False
        assert isinstance(labeler.config, MetaLabelingConfig)
        assert labeler.feature_names == []
        assert labeler.primary_selected_indices is None
        assert labeler.pair_name is None
        assert labeler.model_dir is None

    def test_metalabeler_init_with_config(self):
        """Test MetaLabeler __init__ accepts custom config."""
        cfg = MetaLabelingConfig(min_confidence_threshold=0.60)
        labeler = MetaLabeler(config=cfg)

        assert labeler.config.min_confidence_threshold == 0.60
        assert labeler.is_fitted is False

    def test_metalabeler_init_with_pair_and_model_dir(self):
        """Test MetaLabeler __init__ stores pair_name and model_dir."""
        labeler = MetaLabeler(pair_name="EURUSD", model_dir="/models")

        assert labeler.pair_name == "EURUSD"
        assert labeler.model_dir == "/models"


class TestMetaLabelerSetPrimaryFeatureIndices:
    """Tests for set_primary_feature_indices method."""

    def test_set_primary_feature_indices_stores_correctly(self):
        """Test set_primary_feature_indices stores indices as numpy array."""
        labeler = MetaLabeler()
        indices = np.array([0, 2, 4, 6, 8])

        labeler.set_primary_feature_indices(indices)

        assert labeler.primary_selected_indices is not None
        assert np.array_equal(labeler.primary_selected_indices, indices)

    def test_set_primary_feature_indices_accepts_list(self):
        """Test set_primary_feature_indices converts list to array."""
        labeler = MetaLabeler()
        indices = [1, 3, 5, 7]

        labeler.set_primary_feature_indices(indices)

        assert isinstance(labeler.primary_selected_indices, np.ndarray)
        assert np.array_equal(labeler.primary_selected_indices, np.array(indices))


class TestGenerateMetaLabels:
    """Tests for generate_meta_labels method."""

    def test_generate_meta_labels_returns_tuple(self):
        """Test generate_meta_labels returns (meta_features, meta_labels, meta_weights) tuple."""
        np.random.seed(42)
        labeler = MetaLabeler()

        features = np.random.randn(50, 10).astype(np.float32)
        primary_predictions = np.random.rand(50).astype(np.float32)
        actual_outcomes = np.random.rand(50).astype(np.float32)

        result = labeler.generate_meta_labels(features, primary_predictions, actual_outcomes)

        assert isinstance(result, tuple)
        assert len(result) == 3
        meta_features, meta_labels, meta_weights = result
        assert isinstance(meta_features, np.ndarray)
        assert isinstance(meta_labels, np.ndarray)
        assert isinstance(meta_weights, np.ndarray)

    def test_generate_meta_labels_binary_output(self):
        """Test meta_labels are binary (0 or 1)."""
        np.random.seed(42)
        labeler = MetaLabeler()

        features = np.random.randn(50, 10).astype(np.float32)
        primary_predictions = np.random.rand(50).astype(np.float32)
        actual_outcomes = np.random.rand(50).astype(np.float32)

        _, meta_labels, _ = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        assert np.all((meta_labels == 0) | (meta_labels == 1))

    def test_generate_meta_labels_correct_predictions(self):
        """Test meta_labels correctly identify when primary was right."""
        labeler = MetaLabeler()

        # Case 1: primary predicts > 0.5, outcome is > 0.5 -> correct (label=1)
        features = np.array([[1.0, 2.0, 3.0]])
        primary_predictions = np.array([0.7])
        actual_outcomes = np.array([0.8])

        _, meta_labels, _ = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        assert meta_labels[0] == 1.0

        # Case 2: primary predicts > 0.5, outcome is < 0.5 -> incorrect (label=0)
        actual_outcomes = np.array([0.3])
        _, meta_labels, _ = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )
        assert meta_labels[0] == 0.0

    def test_generate_meta_labels_features_expanded(self):
        """Test meta_features has more columns than raw features (adds primary_prediction)."""
        np.random.seed(42)
        labeler = MetaLabeler()

        n_raw_features = 10
        features = np.random.randn(50, n_raw_features).astype(np.float32)
        primary_predictions = np.random.rand(50).astype(np.float32)
        actual_outcomes = np.random.rand(50).astype(np.float32)

        meta_features, _, _ = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        # Should have raw features + primary_prediction + confidence_distance
        assert meta_features.shape[1] > n_raw_features

    def test_generate_meta_labels_with_pnl_outcomes_adjusts_weights(self):
        """Test meta_weights are adjusted based on confidence."""
        np.random.seed(42)
        labeler = MetaLabeler()

        features = np.random.randn(50, 10).astype(np.float32)
        # Very confident predictions (close to 0 or 1)
        primary_predictions = np.concatenate([
            np.ones(25) * 0.9,  # Confident long
            np.ones(25) * 0.1,  # Confident short
        ])
        actual_outcomes = np.random.rand(50).astype(np.float32)

        _, _, meta_weights = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        # Weights for confident predictions should be > 1.0
        assert np.all(meta_weights > 1.0)

    def test_generate_meta_labels_low_confidence_weights(self):
        """Test lower confidence predictions get lower weights."""
        np.random.seed(42)
        labeler = MetaLabeler()

        features = np.random.randn(100, 10).astype(np.float32)
        # Mix of confident and uncertain
        primary_predictions = np.concatenate([
            np.ones(50) * 0.55,  # Uncertain
            np.ones(50) * 0.95,  # Confident
        ])
        actual_outcomes = np.random.rand(100).astype(np.float32)

        _, _, meta_weights = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        uncertain_weights = meta_weights[:50]
        confident_weights = meta_weights[50:]

        # Average weight of uncertain should be less than confident
        assert np.mean(uncertain_weights) < np.mean(confident_weights)

    def test_generate_meta_labels_dimension_checks(self):
        """Test generate_meta_labels validates input dimensions."""
        labeler = MetaLabeler()

        features = np.random.randn(50, 10).astype(np.float32)
        primary_predictions = np.random.rand(50).astype(np.float32)
        actual_outcomes = np.random.rand(40).astype(np.float32)  # Mismatch!

        with pytest.raises(AssertionError):
            labeler.generate_meta_labels(features, primary_predictions, actual_outcomes)

    def test_generate_meta_labels_3d_features(self):
        """Test generate_meta_labels works with 3D sequence features."""
        np.random.seed(42)
        labeler = MetaLabeler()

        # 3D: (samples, timesteps, features)
        features = np.random.randn(50, 10, 15).astype(np.float32)
        primary_predictions = np.random.rand(50).astype(np.float32)
        actual_outcomes = np.random.rand(50).astype(np.float32)

        meta_features, meta_labels, meta_weights = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        assert meta_features.shape[0] == 50
        assert meta_labels.shape[0] == 50
        assert meta_weights.shape[0] == 50


class TestGetPositionSize:
    """Tests for get_position_size method."""

    def test_get_position_size_in_range(self):
        """Test position sizes are within [min_size, max_size] range."""
        labeler = MetaLabeler()

        np.random.seed(42)
        meta_confidence = np.random.rand(100).astype(np.float32)
        min_size = 0.25
        max_size = 2.0

        sizes = labeler.get_position_size(
            meta_confidence,
            min_size=min_size,
            max_size=max_size,
        )

        assert np.all(sizes >= min_size)
        assert np.all(sizes <= max_size)

    def test_get_position_size_higher_confidence_larger(self):
        """Test higher confidence produces larger position sizes."""
        labeler = MetaLabeler()

        # Low confidence
        low_conf = np.array([0.51, 0.52, 0.53])
        # High confidence
        high_conf = np.array([0.95, 0.96, 0.97])

        low_sizes = labeler.get_position_size(low_conf, min_size=0.25, max_size=2.0)
        high_sizes = labeler.get_position_size(high_conf, min_size=0.25, max_size=2.0)

        assert np.mean(high_sizes) > np.mean(low_sizes)

    def test_get_position_size_confidence_0_5_returns_min(self):
        """Test confidence at 0.5 returns min_size."""
        labeler = MetaLabeler()

        meta_confidence = np.array([0.5, 0.5, 0.5])
        min_size = 0.25
        max_size = 2.0

        sizes = labeler.get_position_size(
            meta_confidence,
            min_size=min_size,
            max_size=max_size,
        )

        # At 0.5 confidence, should be at minimum
        assert np.allclose(sizes, min_size)

    def test_get_position_size_confidence_1_0_returns_max(self):
        """Test confidence at 1.0 returns max_size."""
        labeler = MetaLabeler()

        meta_confidence = np.array([1.0, 1.0, 1.0])
        min_size = 0.25
        max_size = 2.0

        sizes = labeler.get_position_size(
            meta_confidence,
            min_size=min_size,
            max_size=max_size,
        )

        # At 1.0 confidence, should be at maximum
        assert np.allclose(sizes, max_size)

    def test_get_position_size_with_base_size(self):
        """Test base_size scaling."""
        labeler = MetaLabeler()

        meta_confidence = np.array([0.75])
        sizes_base_1 = labeler.get_position_size(
            meta_confidence,
            base_size=1.0,
            min_size=0.25,
            max_size=2.0,
        )
        sizes_base_2 = labeler.get_position_size(
            meta_confidence,
            base_size=2.0,
            min_size=0.25,
            max_size=2.0,
        )

        # base_size=2.0 should produce 2x size
        assert np.allclose(sizes_base_2, sizes_base_1 * 2.0)


class TestAlignFeatures:
    """Tests for _align_features method."""

    def test_align_features_pads_shorter_features(self):
        """Test _align_features pads shorter features with zeros."""
        labeler = MetaLabeler()

        # Mock a fitted model with expected 50 features
        labeler.meta_model = MagicMock()
        labeler.meta_model.n_features_in_ = 50

        # Provide only 30 features
        meta_X = np.random.randn(100, 30).astype(np.float32)
        aligned = labeler._align_features(meta_X)

        assert aligned.shape == (100, 50)
        # First 30 columns should match original
        assert np.array_equal(aligned[:, :30], meta_X)
        # Last 20 columns should be zeros
        assert np.all(aligned[:, 30:] == 0)

    def test_align_features_truncates_longer_features(self):
        """Test _align_features truncates longer features."""
        labeler = MetaLabeler()

        # Mock a fitted model with expected 30 features
        labeler.meta_model = MagicMock()
        labeler.meta_model.n_features_in_ = 30

        # Provide 50 features
        meta_X = np.random.randn(100, 50).astype(np.float32)
        aligned = labeler._align_features(meta_X)

        assert aligned.shape == (100, 30)
        # Should keep first 30 features
        assert np.array_equal(aligned, meta_X[:, :30])

    def test_align_features_exact_match(self):
        """Test _align_features returns unchanged array when dimensions match."""
        labeler = MetaLabeler()

        # Mock a fitted model with expected 40 features
        labeler.meta_model = MagicMock()
        labeler.meta_model.n_features_in_ = 40

        meta_X = np.random.randn(100, 40).astype(np.float32)
        aligned = labeler._align_features(meta_X)

        assert aligned is meta_X  # Should return same object

    def test_align_features_no_model(self):
        """Test _align_features returns unchanged when no model fitted."""
        labeler = MetaLabeler()

        # No model set yet
        meta_X = np.random.randn(100, 50).astype(np.float32)
        aligned = labeler._align_features(meta_X)

        assert aligned is meta_X  # Should return unchanged


class TestMetaLabelerIntegration:
    """Integration tests for MetaLabeler."""

    def test_full_workflow_generate_and_predict_not_fitted(self):
        """Test that predict_meta_confidence raises error when not fitted."""
        labeler = MetaLabeler()

        features = np.random.randn(50, 10).astype(np.float32)
        primary_predictions = np.random.rand(50).astype(np.float32)

        with pytest.raises(RuntimeError, match="not fitted"):
            labeler.predict_meta_confidence(features, primary_predictions)

    def test_set_then_use_primary_feature_indices(self):
        """Test set_primary_feature_indices stores indices for later use."""
        labeler = MetaLabeler()

        # Store indices
        indices = np.array([0, 2, 4, 6, 8])
        labeler.set_primary_feature_indices(indices)

        # Verify stored
        assert np.array_equal(labeler.primary_selected_indices, indices)

    def test_meta_labels_accuracy_tracking(self):
        """Test _primary_accuracy is tracked after generate_meta_labels."""
        labeler = MetaLabeler()

        # All predictions correct
        features = np.random.randn(100, 10).astype(np.float32)
        primary_predictions = np.array([0.8] * 100)
        actual_outcomes = np.array([0.9] * 100)  # All > 0.5, so all correct

        labeler.generate_meta_labels(features, primary_predictions, actual_outcomes)

        # Should have 100% accuracy
        assert labeler._primary_accuracy == 1.0

    def test_meta_labels_mixed_accuracy(self):
        """Test _primary_accuracy with mixed correct/incorrect predictions."""
        labeler = MetaLabeler()

        features = np.random.randn(100, 10).astype(np.float32)
        # 60 predictions will be > 0.5, 40 < 0.5
        primary_predictions = np.concatenate([np.ones(60) * 0.7, np.ones(40) * 0.3])
        # 50 outcomes > 0.5, 50 < 0.5 (shuffled)
        actual_outcomes = np.concatenate([np.ones(50) * 0.8, np.ones(50) * 0.2])

        labeler.generate_meta_labels(features, primary_predictions, actual_outcomes)

        # Accuracy should be between 0 and 1
        assert 0.0 <= labeler._primary_accuracy <= 1.0

    def test_reduced_features_mode_fewer_meta_features(self):
        """Test use_reduced_features=True produces fewer meta-features."""
        cfg_reduced = MetaLabelingConfig(use_reduced_features=True)
        labeler_reduced = MetaLabeler(config=cfg_reduced)

        cfg_full = MetaLabelingConfig(use_reduced_features=False)
        labeler_full = MetaLabeler(config=cfg_full)

        # 3D features to trigger aggregation path
        features = np.random.randn(50, 10, 15).astype(np.float32)
        primary_predictions = np.random.rand(50).astype(np.float32)
        actual_outcomes = np.random.rand(50).astype(np.float32)

        meta_features_reduced, _, _ = labeler_reduced.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )
        meta_features_full, _, _ = labeler_full.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        # Reduced should have fewer features (no mean/std aggregates)
        assert meta_features_reduced.shape[1] < meta_features_full.shape[1]


class TestEdgeCases:
    """Edge case tests."""

    def test_generate_meta_labels_single_sample(self):
        """Test generate_meta_labels works with single sample."""
        labeler = MetaLabeler()

        features = np.array([[1.0, 2.0, 3.0]])
        primary_predictions = np.array([0.7])
        actual_outcomes = np.array([0.8])

        meta_features, meta_labels, meta_weights = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        assert meta_features.shape[0] == 1
        assert meta_labels.shape[0] == 1
        assert meta_weights.shape[0] == 1

    def test_position_size_zero_confidence(self):
        """Test position size with confidence near 0."""
        labeler = MetaLabeler()

        meta_confidence = np.array([0.0, 0.1, 0.2])
        sizes = labeler.get_position_size(
            meta_confidence,
            min_size=0.25,
            max_size=2.0,
        )

        # Should be clipped at min_size
        assert np.all(sizes >= 0.25)

    def test_get_position_size_extreme_confidence(self):
        """Test position size with confidence > 1.0 is clipped."""
        labeler = MetaLabeler()

        # Confidence slightly over 1.0 (shouldn't happen but test robustness)
        meta_confidence = np.array([1.1, 1.2])
        sizes = labeler.get_position_size(
            meta_confidence,
            min_size=0.25,
            max_size=2.0,
        )

        # Should be clipped at max_size
        assert np.all(sizes <= 2.0)

    def test_empty_primary_feature_indices(self):
        """Test set_primary_feature_indices with empty array."""
        labeler = MetaLabeler()

        indices = np.array([])
        labeler.set_primary_feature_indices(indices)

        assert len(labeler.primary_selected_indices) == 0

    def test_generate_meta_labels_all_same_prediction(self):
        """Test generate_meta_labels when all predictions are same."""
        labeler = MetaLabeler()

        features = np.random.randn(50, 10).astype(np.float32)
        primary_predictions = np.ones(50) * 0.6  # All same
        actual_outcomes = np.ones(50) * 0.7      # All same

        meta_features, meta_labels, meta_weights = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        # All labels should be 1 (all correct)
        assert np.all(meta_labels == 1.0)
        # All weights should be same (same confidence)
        assert np.allclose(meta_weights, meta_weights[0])

    def test_generate_meta_labels_extreme_confidence(self):
        """Test generate_meta_labels with extreme confidence values (0.0, 1.0)."""
        labeler = MetaLabeler()

        features = np.random.randn(50, 10).astype(np.float32)
        primary_predictions = np.concatenate([
            np.zeros(25),   # 0.0
            np.ones(25),    # 1.0
        ])
        actual_outcomes = np.concatenate([
            np.zeros(25),   # All < 0.5
            np.ones(25),    # All >= 0.5
        ])

        meta_features, meta_labels, meta_weights = labeler.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        # Should handle extreme values gracefully
        assert not np.any(np.isnan(meta_features))
        assert not np.any(np.isnan(meta_labels))
        assert not np.any(np.isnan(meta_weights))
        # Extreme confidence should produce max weights
        assert np.all(meta_weights > 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
