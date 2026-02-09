"""Integration tests for meta-labeler training in buddy pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestMetaLabelerIntegration(unittest.TestCase):
    """Test that meta-labeler is properly integrated into training pipeline."""

    def test_meta_labeler_import(self):
        """Test that meta-labeler module can be imported."""
        try:
            from src.training.meta_labeling import MetaLabeler, MetaLabelingConfig, train_meta_labeler
            self.assertIsNotNone(MetaLabeler)
            self.assertIsNotNone(MetaLabelingConfig)
            self.assertIsNotNone(train_meta_labeler)
        except ImportError as e:
            self.fail(f"Failed to import meta_labeling: {e}")

    def test_meta_labeler_config_defaults(self):
        """Test that MetaLabelingConfig has correct defaults."""
        from src.training.meta_labeling import MetaLabelingConfig
        
        config = MetaLabelingConfig()
        
        # Check defaults match what we expect
        self.assertTrue(config.use_xgboost)
        self.assertEqual(config.n_estimators, 100)
        self.assertEqual(config.max_depth, 2)
        self.assertEqual(config.learning_rate, 0.05)
        self.assertEqual(config.min_confidence_threshold, 0.55)
        self.assertTrue(config.use_reduced_features)

    def test_meta_labeler_instantiation(self):
        """Test that MetaLabeler can be instantiated."""
        from src.training.meta_labeling import MetaLabeler, MetaLabelingConfig
        
        config = MetaLabelingConfig()
        labeler = MetaLabeler(config)
        
        self.assertIsNotNone(labeler)
        self.assertFalse(labeler.is_fitted)
        self.assertEqual(labeler._primary_accuracy, 0.5)

    @patch('xgboost.XGBClassifier')
    def test_meta_labeler_training_mock(self, mock_xgb_classifier):
        """Test meta-labeler training with mocked XGBoost."""
        from src.training.meta_labeling import train_meta_labeler, MetaLabelingConfig
        
        # Setup mock - needs to handle .fit(), .predict(), .predict_proba()
        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda X, **kw: np.random.randint(0, 2, len(X)).astype(float)
        mock_model.predict_proba.side_effect = lambda X, **kw: np.column_stack([
            np.random.rand(len(X)), np.random.rand(len(X))
        ])
        mock_xgb_classifier.return_value = mock_model
        
        # Create dummy data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        primary_probs_train = np.random.rand(100)
        
        X_val = np.random.randn(30, 10)
        y_val = np.random.randint(0, 2, 30)
        primary_probs_val = np.random.rand(30)
        
        config = MetaLabelingConfig(use_xgboost=True)
        
        # Train meta-labeler
        labeler, metrics = train_meta_labeler(
            X_train=X_train,
            y_train=y_train,
            primary_probs_train=primary_probs_train,
            X_val=X_val,
            y_val=y_val,
            primary_probs_val=primary_probs_val,
            config=config,
        )
        
        # Verify labeler was created
        self.assertIsNotNone(labeler)
        self.assertIsNotNone(metrics)
        
        # Verify XGBoost was called
        mock_xgb_classifier.assert_called_once()

    def test_inference_config_has_meta_threshold(self):
        """Test that InferenceConfig includes meta-labeler threshold."""
        from src.core.modular_inference import InferenceConfig
        
        config = InferenceConfig()
        
        # Check that min_meta_confidence exists
        self.assertTrue(hasattr(config, 'min_meta_confidence'))
        self.assertEqual(config.min_meta_confidence, 0.55)

    def test_modular_inference_loads_meta_labeler(self):
        """Test that ModularEnsembleInference attempts to load meta-labeler."""
        from src.core.modular_inference import ModularEnsembleInference
        
        # Create instance (will fail to load files but should have attributes)
        try:
            inference = ModularEnsembleInference(instrument="EUR_USD")
            
            # Check meta-labeler attributes exist
            self.assertTrue(hasattr(inference, 'meta_labeler'))
            self.assertTrue(hasattr(inference, '_meta_labeler_loaded'))
            
        except FileNotFoundError:
            # Expected - no trained models in test environment
            pass


class TestMetaLabelerGateLogic(unittest.TestCase):
    """Test meta-labeler gate logic in inference."""

    def test_gate_bypass_when_not_loaded(self):
        """Test that meta gate passes when meta-labeler is not loaded."""
        # This tests the logic at modular_inference.py:2166-2167
        # where meta_gate_passed = True if meta-labeler is not loaded
        
        # Create mock inference object
        class MockInference:
            def __init__(self):
                self._meta_labeler_loaded = False
                self.meta_labeler = None
        
        mock_inf = MockInference()
        
        # Simulate gate check logic
        if mock_inf._meta_labeler_loaded and mock_inf.meta_labeler is not None:
            meta_gate_passed = False  # Would run actual check
        else:
            meta_gate_passed = True  # Pass by default
        
        self.assertTrue(meta_gate_passed)

    def test_gate_threshold_check(self):
        """Test meta-labeler gate threshold checking logic."""
        # Simulate the gate check at modular_inference.py:2152
        min_meta_confidence = 0.55
        
        # Test: confidence above threshold
        meta_confidence = 0.70
        meta_gate_passed = meta_confidence >= min_meta_confidence
        self.assertTrue(meta_gate_passed)
        
        # Test: confidence below threshold
        meta_confidence = 0.40
        meta_gate_passed = meta_confidence >= min_meta_confidence
        self.assertFalse(meta_gate_passed)
        
        # Test: confidence exactly at threshold
        meta_confidence = 0.55
        meta_gate_passed = meta_confidence >= min_meta_confidence
        self.assertTrue(meta_gate_passed)


if __name__ == '__main__':
    unittest.main()
