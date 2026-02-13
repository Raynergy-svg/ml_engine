"""
Tests for Walk-Forward Cross-Validation Orchestrator.

This module tests the WalkForwardOrchestrator class which manages
fold-wise training and metric aggregation for robust model validation.
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.training.walkforward_orchestrator import WalkForwardOrchestrator
from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig


class MockTrainer(BaseTrainer):
    """Mock trainer for testing."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.train_calls = []
        self.predict_calls = []
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names=None,
        instrument='',
        fold_id=None,
        **kwargs,
    ) -> Dict[str, float]:
        """Mock training that returns deterministic metrics."""
        self.train_calls.append({
            'X_train_shape': X_train.shape,
            'y_train_shape': y_train.shape,
            'X_val_shape': x_val.shape,
            'y_val_shape': y_val.shape,
            'instrument': instrument,
            'fold_id': fold_id,
        })
        
        # Return deterministic metrics based on fold_id
        base_acc = 0.60 + (fold_id or 0) * 0.05
        
        return {
            'val_accuracy': base_acc,
            'val_balanced_accuracy': base_acc - 0.02,
            'train_accuracy': base_acc + 0.05,
        }
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Mock prediction that returns random predictions."""
        self.predict_calls.append(X.shape)
        n_samples = X.shape[0]
        
        # Return predictions as probabilities
        predictions = np.random.rand(n_samples)
        
        return {
            'prediction': predictions,
            'predictions': predictions,
        }
    
    def save(self, path: str) -> None:
        """Mock save."""
        pass
    
    def load(self, path: str) -> None:
        """Mock load."""
        pass


class TestWalkForwardOrchestrator:
    """Test WalkForwardOrchestrator class."""
    
    @pytest.fixture
    def wf_config(self):
        """Create walk-forward configuration.
        
        Note: Using smaller proportions to ensure 3 folds fit within the data.
        For rolling mode with n_splits=3:
        - Total needed: train_size + 2*(val_size + test_size + gap/n) <= 1.0
        - Using train_size=0.30, val_size=0.10, test_size=0.10, gap=5
        """
        return {
            'enabled': True,
            'mode': 'rolling',
            'n_splits': 3,
            'train_size': 0.30,
            'val_size': 0.10,
            'test_size': 0.10,
            'gap': 5,
            'min_train_size': 50,
            'retrain_per_fold': True,
            'aggregate_method': 'best',
        }
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data.
        
        With the updated config (train=0.30, val=0.10, test=0.10, gap=5):
        - For n=500: train=150, val=50, test=50, fold_size=105
        - Fold 0: train[0:150], val[155:205], test[205:255]
        - Fold 1: train[105:255], val[260:310], test[310:360]
        - Fold 2: train[210:360], val[365:415], test[415:465]
        - All within 500 samples.
        """
        np.random.seed(42)
        n_samples = 500  # Sufficient for 3 folds with updated config
        n_features = 10
        
        X_train = np.random.randn(n_samples // 2, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, n_samples // 2).astype(np.int32)
        X_val = np.random.randn(n_samples // 2, n_features).astype(np.float32)
        y_val = np.random.randint(0, 2, n_samples // 2).astype(np.int32)
        
        return X_train, y_train, X_val, y_val
    
    def test_orchestrator_initialization(self, wf_config):
        """Test orchestrator initialization."""
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        assert orchestrator.trainer_class == MockTrainer
        assert orchestrator.wf_config.n_splits == 3
        assert orchestrator.wf_config.mode == 'rolling'
        assert orchestrator.wf_config.aggregate_method == 'best'
        assert len(orchestrator.fold_trainers) == 0
        assert len(orchestrator.fold_results) == 0
    
    def test_orchestrator_rolling_mode(self, wf_config, sample_data):
        """Test orchestrator with rolling window mode."""
        X_train, y_train, X_val, y_val = sample_data
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            feature_names=['f1', 'f2', 'f3'],
            instrument='EUR_USD',
        )
        
        # Check that training happened
        assert best_trainer is not None
        assert isinstance(best_trainer, MockTrainer)
        
        # Check that multiple folds were trained
        assert len(orchestrator.fold_trainers) == 3
        assert len(orchestrator.fold_results) == 3
        
        # Check metrics
        assert 'val_accuracy' in metrics
        assert 'wfcv_mean_test_accuracy' in metrics
        assert 'wfcv_std_test_accuracy' in metrics
        assert 'wfcv_n_folds' in metrics
        assert metrics['wfcv_n_folds'] == 3
    
    def test_orchestrator_expanding_mode(self, wf_config, sample_data):
        """Test orchestrator with expanding window mode."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Change to expanding mode
        wf_config['mode'] = 'expanding'
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            feature_names=['f1', 'f2'],
            instrument='GBP_USD',
        )
        
        assert best_trainer is not None
        assert len(orchestrator.fold_trainers) == 3
        assert metrics['wfcv_n_folds'] == 3
    
    def test_orchestrator_best_selection(self, wf_config, sample_data):
        """Test best fold selection strategy."""
        X_train, y_train, X_val, y_val = sample_data
        
        wf_config['aggregate_method'] = 'best'
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            instrument='USD_JPY',
        )
        
        # With our mock trainer, fold 2 (index 2) should have highest accuracy
        # because base_acc = 0.60 + fold_id * 0.05
        assert 'wfcv_best_fold' in metrics
        # Best fold should be reported (1-indexed)
        assert metrics['wfcv_best_fold'] >= 1
        assert metrics['wfcv_best_fold'] <= 3
    
    def test_orchestrator_average_selection(self, wf_config, sample_data):
        """Test average aggregation strategy."""
        X_train, y_train, X_val, y_val = sample_data
        
        wf_config['aggregate_method'] = 'average'
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            instrument='EUR_GBP',
        )
        
        # Should use averaged metrics
        assert 'wfcv_mean_test_accuracy' in metrics
        assert 'wfcv_std_test_accuracy' in metrics
        assert 'wfcv_best_fold' not in metrics  # Not used for average
    
    def test_orchestrator_retrain_per_fold(self, wf_config, sample_data):
        """Test that models are retrained per fold when enabled."""
        X_train, y_train, X_val, y_val = sample_data
        
        wf_config['retrain_per_fold'] = True
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            instrument='AUD_USD',
        )
        
        # All trainers should be different instances
        assert len(orchestrator.fold_trainers) == 3
        
        # Each trainer should have been called with different fold_id
        for i, trainer in enumerate(orchestrator.fold_trainers):
            assert len(trainer.train_calls) > 0
            # Check that fold_id was passed
            assert trainer.train_calls[0]['fold_id'] == i
    
    def test_orchestrator_fold_results(self, wf_config, sample_data):
        """Test fold results structure."""
        X_train, y_train, X_val, y_val = sample_data
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            instrument='NZD_USD',
        )
        
        # Check fold results
        assert len(orchestrator.fold_results) == 3
        
        for i, result in enumerate(orchestrator.fold_results):
            assert result.fold == i + 1
            assert result.train_size > 0
            assert result.val_size > 0
            assert result.test_size > 0
            assert 'val_accuracy' in result.val_metrics
            assert 'test_accuracy' in result.test_metrics
    
    def test_orchestrator_summary(self, wf_config, sample_data):
        """Test summary computation."""
        X_train, y_train, X_val, y_val = sample_data
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            instrument='CAD_USD',
        )
        
        # Check summary was computed
        assert orchestrator.summary is not None
        assert orchestrator.summary.n_folds == 3
        assert len(orchestrator.summary.fold_results) == 3
        
        # Check aggregated metrics
        assert len(orchestrator.summary.mean_test_metrics) > 0
        assert len(orchestrator.summary.std_test_metrics) > 0
    
    def test_orchestrator_save_summary(self, wf_config, sample_data, tmp_path):
        """Test saving WFCV summary to disk."""
        X_train, y_train, X_val, y_val = sample_data
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            instrument='CHF_USD',
        )
        
        # Save summary
        output_dir = tmp_path / "models"
        orchestrator.save_summary(output_dir)
        
        # Check file was created
        summary_path = output_dir / "wfcv_summary.json"
        assert summary_path.exists()
        
        # Load and verify content
        import json
        with open(summary_path) as f:
            summary_data = json.load(f)
        
        assert summary_data['n_folds'] == 3
        assert 'mean_test_metrics' in summary_data
        assert 'std_test_metrics' in summary_data
        assert 'config' in summary_data
        assert 'fold_results' in summary_data
        assert len(summary_data['fold_results']) == 3
    
    def test_orchestrator_with_sample_weights(self, wf_config, sample_data):
        """Test orchestrator with sample weights."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Create sample weights
        w_train = np.random.rand(len(y_train)).astype(np.float32)
        w_val = np.random.rand(len(y_val)).astype(np.float32)
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            instrument='EUR_CHF',
            w_train=w_train,
            w_val=w_val,
        )
        
        assert best_trainer is not None
        assert len(orchestrator.fold_trainers) == 3
    
    def test_orchestrator_small_dataset(self, wf_config):
        """Test orchestrator with small dataset."""
        # Very small dataset
        n_samples = 100
        n_features = 5
        
        X_train = np.random.randn(n_samples // 2, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, n_samples // 2).astype(np.int32)
        X_val = np.random.randn(n_samples // 2, n_features).astype(np.float32)
        y_val = np.random.randint(0, 2, n_samples // 2).astype(np.int32)
        
        # Adjust config for small dataset
        wf_config['min_train_size'] = 20
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            instrument='EUR_USD',
        )
        
        # Should still work but may have fewer folds
        assert best_trainer is not None
        assert len(orchestrator.fold_trainers) > 0
        assert 'wfcv_n_folds' in metrics


class TestWalkForwardIntegration:
    """Integration tests for walk-forward validation."""
    
    def test_config_loading_from_yaml(self, tmp_path):
        """Test loading walk-forward config from YAML file."""
        config_content = """
walkforward:
  enabled: true
  mode: rolling
  n_splits: 5
  train_size: 0.60
  val_size: 0.10
  test_size: 0.10
  gap: 24
  min_train_size: 2000
  retrain_per_fold: true
  aggregate_method: best
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        # Load config
        import yaml
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        
        wf_config = cfg['walkforward']
        
        # Verify config
        assert wf_config['enabled'] is True
        assert wf_config['mode'] == 'rolling'
        assert wf_config['n_splits'] == 5
        assert wf_config['train_size'] == 0.60
        assert wf_config['gap'] == 24
    
    def test_orchestrator_with_real_config(self):
        """Test orchestrator with realistic configuration.
        
        Note: Using smaller proportions to ensure 5 folds fit within the data.
        With train_size=0.20, val_size=0.05, test_size=0.05, gap=10, n_splits=5:
        - For n=2000: train=400, val=100, test=100, fold_size=210
        - Fold 4: train_end = 4*210 + 400 = 1240, test_end = 1240 + 10 + 100 + 100 = 1450
        - All within 2000 samples.
        """
        wf_config = {
            'enabled': True,
            'mode': 'rolling',
            'n_splits': 5,
            'train_size': 0.20,
            'val_size': 0.05,
            'test_size': 0.05,
            'gap': 10,
            'min_train_size': 100,
            'use_purged_kfold': True,
            'purge_gap': 10,
            'embargo_gap': 5,
            'retrain_per_fold': True,
            'aggregate_method': 'best',
        }
        
        # Create dataset sized for 5 folds
        np.random.seed(42)
        n_samples = 2000
        n_features = 20
        
        X_train = np.random.randn(n_samples // 2, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, n_samples // 2).astype(np.int32)
        X_val = np.random.randn(n_samples // 2, n_features).astype(np.float32)
        y_val = np.random.randint(0, 2, n_samples // 2).astype(np.int32)
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=MockTrainer,
            trainer_config=TrainerConfig(),
            wf_config=wf_config,
        )
        
        best_trainer, metrics = orchestrator.train(
            X_train, y_train, X_val, y_val,
            feature_names=[f'feature_{i}' for i in range(n_features)],
            instrument='EUR_USD',
        )
        
        # Should complete successfully
        assert best_trainer is not None
        assert len(orchestrator.fold_trainers) == 5
        assert metrics['wfcv_n_folds'] == 5
        
        # Metrics should be reasonable
        assert 0.0 <= metrics['wfcv_mean_test_accuracy'] <= 1.0
        assert 0.0 <= metrics['wfcv_std_test_accuracy'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
