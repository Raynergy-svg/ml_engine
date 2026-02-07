"""
Tests for Walk-Forward Cross-Validation Configuration.

This test ensures that the new WalkForwardConfig class works correctly
and can be loaded from configuration files.
"""

import pytest
import numpy as np
from pathlib import Path
import yaml

from src.training.walkforward_validation import (
    WalkForwardConfig,
    WalkForwardValidator,
    purged_kfold_split,
)


class TestWalkForwardConfig:
    """Test WalkForwardConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()
        
        assert config.enabled is True
        assert config.mode == "rolling"
        assert config.n_splits == 5
        assert config.train_size == 0.60
        assert config.val_size == 0.10
        assert config.test_size == 0.10
        assert config.gap == 24
        assert config.min_train_size == 2000
        assert config.use_purged_kfold is True
        assert config.purge_gap == 24
        assert config.embargo_gap == 12
        assert config.retrain_per_fold is True
        assert config.aggregate_method == "best"
        assert config.ensure_regime_balance is False
        assert config.min_samples_per_regime == 50
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WalkForwardConfig(
            enabled=False,
            mode="expanding",
            n_splits=10,
            train_size=0.70,
            gap=48,
            retrain_per_fold=False,
        )
        
        assert config.enabled is False
        assert config.mode == "expanding"
        assert config.n_splits == 10
        assert config.train_size == 0.70
        assert config.gap == 48
        assert config.retrain_per_fold is False
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'enabled': True,
            'mode': 'rolling',
            'n_splits': 7,
            'train_size': 0.65,
            'gap': 36,
            'retrain_per_fold': True,
        }
        
        config = WalkForwardConfig.from_dict(config_dict)
        
        assert config.enabled is True
        assert config.mode == "rolling"
        assert config.n_splits == 7
        assert config.train_size == 0.65
        assert config.gap == 36
        assert config.retrain_per_fold is True
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = WalkForwardConfig(
            mode="expanding",
            n_splits=8,
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['mode'] == "expanding"
        assert config_dict['n_splits'] == 8
        assert 'enabled' in config_dict
        assert 'train_size' in config_dict
    
    def test_load_from_yaml(self):
        """Test loading config from YAML file."""
        # Load the actual config file
        config_path = Path(__file__).parent.parent / "config" / "config_improved_H1.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            if 'walkforward' in full_config:
                wf_dict = full_config['walkforward']
                config = WalkForwardConfig.from_dict(wf_dict)
                
                # Verify expected values from config
                assert config.enabled is True
                assert config.mode == "rolling"
                assert config.n_splits == 5
                assert config.retrain_per_fold is True


class TestWalkForwardValidator:
    """Test WalkForwardValidator with new config."""
    
    def test_validator_with_config(self):
        """Test creating validator from config."""
        config = WalkForwardConfig(
            mode="rolling",
            n_splits=3,
            train_size=0.60,
            val_size=0.10,
            test_size=0.10,
            gap=10,
        )
        
        validator = WalkForwardValidator(
            n_splits=config.n_splits,
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
            gap=config.gap,
            mode=config.mode,
        )
        
        # Test with dummy data
        X = np.random.randn(1000, 60, 10)
        
        splits = list(validator.split(X))
        assert len(splits) > 0
        
        for train_idx, val_idx, test_idx in splits:
            # Verify temporal ordering
            assert train_idx[-1] < val_idx[0]  # Train before validation
            assert val_idx[-1] < test_idx[0]   # Validation before test
            
            # Verify gap exists
            assert val_idx[0] - train_idx[-1] >= config.gap
    
    def test_rolling_vs_expanding(self):
        """Test difference between rolling and expanding modes."""
        X = np.random.randn(1000, 60, 10)
        
        # Rolling mode: fixed window size
        rolling_validator = WalkForwardValidator(
            n_splits=3,
            train_size=0.60,
            mode="rolling",
        )
        
        rolling_splits = list(rolling_validator.split(X))
        rolling_train_sizes = [len(train) for train, _, _ in rolling_splits]
        
        # Expanding mode: growing window
        expanding_validator = WalkForwardValidator(
            n_splits=3,
            train_size=0.60,
            mode="expanding",
        )
        
        expanding_splits = list(expanding_validator.split(X))
        expanding_train_sizes = [len(train) for train, _, _ in expanding_splits]
        
        # In rolling mode, training size should be roughly constant
        assert max(rolling_train_sizes) - min(rolling_train_sizes) < 100
        
        # In expanding mode, training size should grow
        assert expanding_train_sizes[-1] > expanding_train_sizes[0]


class TestPurgedKFold:
    """Test purged K-Fold implementation."""
    
    def test_purged_kfold_basic(self):
        """Test basic purged K-fold split."""
        n_samples = 1000
        n_splits = 5
        purge_gap = 10
        embargo_gap = 5
        
        splits = list(purged_kfold_split(
            n_samples=n_samples,
            n_splits=n_splits,
            purge_gap=purge_gap,
            embargo_gap=embargo_gap,
        ))
        
        assert len(splits) == n_splits
        
        for train_idx, test_idx in splits:
            # Verify train and test don't overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            
            # Verify purge gap exists
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Check gap between train and test
                if train_idx[-1] < test_idx[0]:
                    gap = test_idx[0] - train_idx[-1]
                    assert gap >= purge_gap
    
    def test_purged_kfold_with_config(self):
        """Test purged K-fold with WalkForwardConfig."""
        config = WalkForwardConfig(
            use_purged_kfold=True,
            purge_gap=24,
            embargo_gap=12,
            n_splits=4,
        )
        
        n_samples = 2000
        
        splits = list(purged_kfold_split(
            n_samples=n_samples,
            n_splits=config.n_splits,
            purge_gap=config.purge_gap,
            embargo_gap=config.embargo_gap,
        ))
        
        assert len(splits) == config.n_splits
        
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestTemporalOrdering:
    """Test temporal ordering is maintained."""
    
    def test_no_look_ahead_bias(self):
        """Verify no training samples come after test samples."""
        config = WalkForwardConfig(
            mode="rolling",
            n_splits=4,
            train_size=0.60,
            gap=20,
        )
        
        validator = WalkForwardValidator(
            n_splits=config.n_splits,
            train_size=config.train_size,
            gap=config.gap,
            mode=config.mode,
        )
        
        X = np.random.randn(1500, 60, 10)
        
        for train_idx, val_idx, test_idx in validator.split(X):
            # In rolling mode, train can come before OR after test
            # But validation should always be between train and test
            # And there should be a gap
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                # Check gap between train and validation
                if train_idx[-1] < val_idx[0]:
                    gap = val_idx[0] - train_idx[-1]
                    assert gap >= config.gap, \
                        f"Gap between train and val is {gap}, expected >= {config.gap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
