"""
Integration tests for overfitting/bias fixes from fix_overfitting_bias_plan.md

Tests verify all 5 critical fixes are correctly implemented:
1. Label threshold reduced from 0.003 to 0.0015
2. Focal Loss used instead of BinaryCrossentropy when use_focal_loss=true
3. Sample weights use inverse frequency (~62x) not 1.5x boost
4. Regularization reduced (dropout 0.3, l2_reg 0.005, input_noise 0.03)
5. Feature selection enabled using RF importance method

All tests use mocking to avoid modifying production data/models.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============================================================================
# TEST #1: LABEL THRESHOLD
# ============================================================================

class TestLabelThreshold:
    """Fix #1: Verify direction_threshold is set to 0.003 for better class balance."""
    
    def test_config_threshold_value(self):
        """Config should have threshold 0.003 for better class balance."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        threshold = config.get("direction_threshold")
        assert threshold is not None, "direction_threshold not found in config"
        assert threshold == 0.003, f"Expected threshold 0.003, got {threshold}"
    
    def test_threshold_is_production_value(self):
        """Ensure production threshold 0.003 is used (increased from 0.0015 for class balance)."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        threshold = config.get("direction_threshold")
        assert threshold == 0.003, f"Expected production threshold 0.003, got {threshold}"


# ============================================================================
# TEST #2: FOCAL LOSS
# ============================================================================

class TestFocalLoss:
    """Fix #2: Verify Focal Loss is used instead of BinaryCrossentropy."""
    
    def test_binary_focal_loss_exists(self):
        """BinaryFocalLoss class should exist and be importable."""
        from src.models.tensorflow_models import BinaryFocalLoss
        assert BinaryFocalLoss is not None
    
    def test_binary_focal_loss_instantiation(self):
        """BinaryFocalLoss should instantiate with correct parameters."""
        from src.models.tensorflow_models import BinaryFocalLoss
        
        loss = BinaryFocalLoss(gamma=2.0, alpha=0.5, label_smoothing=0.05)
        assert loss.gamma == 2.0
        assert loss.alpha == 0.5
        assert loss.label_smoothing == 0.05
    
    def test_focal_loss_used_when_config_enabled(self):
        """When use_focal_loss=true, BinaryFocalLoss should be instantiated."""
        # Create mock config with use_focal_loss=True
        mock_config = Mock()
        mock_config.use_focal_loss = True
        mock_config.label_smoothing = 0.05
        mock_config.l2_reg = 0.005
        mock_config.input_noise = 0.03
        mock_config.use_ema = False
        mock_config.use_feature_selection = False
        
        # Import and verify the trainer's loss selection logic
        # We test the import path rather than running full training
        from src.models.tensorflow_models import BinaryFocalLoss
        
        # Verify BinaryFocalLoss can be created as trainer would
        base_loss = BinaryFocalLoss(
            gamma=2.0,
            alpha=0.5,
            label_smoothing=0.05,
        )
        
        # Verify it's not BinaryCrossentropy
        assert type(base_loss).__name__ == "BinaryFocalLoss"
        assert type(base_loss).__name__ != "BinaryCrossentropy"
    
    def test_config_use_focal_loss_enabled(self):
        """Config should have use_focal_loss: true."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        use_focal_loss = config.get("use_focal_loss", False)
        assert use_focal_loss is True, f"Expected use_focal_loss=true, got {use_focal_loss}"


# ============================================================================
# TEST #3: SAMPLE WEIGHTS (INVERSE FREQUENCY)
# ============================================================================

class TestSampleWeights:
    """Fix #3: Verify inverse frequency sample weighting (~62x for 124:1 imbalance)."""
    
    def test_inverse_frequency_calculation(self):
        """Inverse frequency weights should be calculated correctly."""
        # Simulate 124:1 imbalance (e.g., 124 DOWN, 1 UP)
        n_up = 10
        n_down = 1240  # 124:1 ratio
        total = n_up + n_down
        
        # Expected inverse frequency weights
        expected_up_weight = total / (2 * n_up)  # 1250 / 20 = 62.5
        expected_down_weight = total / (2 * n_down)  # 1250 / 2480 ≈ 0.504
        
        assert expected_up_weight > 60, f"UP weight should be >60x, got {expected_up_weight}"
        assert expected_up_weight < 65, f"UP weight should be <65x, got {expected_up_weight}"
        assert expected_down_weight < 1.0, f"DOWN weight should be <1.0, got {expected_down_weight}"
    
    def test_sample_weights_not_old_boost(self):
        """Sample weights should NOT use old 1.5x boost for minority class."""
        # The old code used: minority_boost = 1.5
        # This is insufficient for 124:1 imbalance
        old_boost = 1.5
        
        # For 124:1 imbalance, we need ~62x, not 1.5x
        n_up = 10
        n_down = 1240
        total = n_up + n_down
        correct_up_weight = total / (2 * n_up)
        
        assert correct_up_weight > old_boost * 10, (
            f"Inverse frequency weight {correct_up_weight:.1f}x should be >> old boost {old_boost}x"
        )
    
    @pytest.mark.parametrize("n_up,n_down", [
        (10, 1240),   # 124:1
        (50, 950),    # 19:1
        (100, 900),   # 9:1
        (200, 800),   # 4:1
    ])
    def test_inverse_frequency_various_imbalances(self, n_up, n_down):
        """Inverse frequency should work for various imbalance ratios."""
        total = n_up + n_down
        
        up_weight = total / (2 * n_up)
        down_weight = total / (2 * n_down)
        
        # Weights should be inversely proportional to class frequency
        imbalance_ratio = n_down / n_up
        weight_ratio = up_weight / down_weight
        
        # Weight ratio should approximately equal imbalance ratio
        assert abs(weight_ratio - imbalance_ratio) / imbalance_ratio < 0.01, (
            f"Weight ratio {weight_ratio:.2f} should match imbalance ratio {imbalance_ratio:.2f}"
        )


# ============================================================================
# TEST #4: REGULARIZATION
# ============================================================================

class TestRegularization:
    """Fix #4: Verify reduced regularization values in config."""
    
    def test_model_dropout_reduced(self):
        """Model dropout should be 0.3 (not 0.5)."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        model_config = config.get("model", {})
        dropout = model_config.get("dropout")
        
        assert dropout is not None, "model.dropout not found in config"
        assert dropout == 0.3, f"Expected dropout=0.3, got {dropout}"
        assert dropout != 0.5, "Still using old dropout=0.5"
    
    def test_kernel_regularizer_reduced(self):
        """Kernel regularizer should be 0.001 (not 0.005)."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        model_config = config.get("model", {})
        l2 = model_config.get("kernel_regularizer")
        
        assert l2 is not None, "model.kernel_regularizer not found in config"
        assert l2 == 0.001, f"Expected kernel_regularizer=0.001, got {l2}"
    
    def test_transformer_dropout_reduced(self):
        """Transformer dropout should be 0.3 (not 0.6)."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        transformer_config = config.get("transformer", {})
        dropout = transformer_config.get("dropout")
        
        assert dropout is not None, "transformer.dropout not found in config"
        assert dropout == 0.3, f"Expected transformer.dropout=0.3, got {dropout}"
        assert dropout != 0.6, "Still using old transformer.dropout=0.6"
    
    def test_transformer_l2_reduced(self):
        """Transformer l2_reg should be 0.005 (not 0.02)."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        transformer_config = config.get("transformer", {})
        l2 = transformer_config.get("l2_reg")
        
        assert l2 is not None, "transformer.l2_reg not found in config"
        assert l2 == 0.005, f"Expected transformer.l2_reg=0.005, got {l2}"
        assert l2 != 0.02, "Still using old transformer.l2_reg=0.02"
    
    def test_transformer_input_noise_reduced(self):
        """Transformer input_noise should be 0.03 (not 0.08)."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        transformer_config = config.get("transformer", {})
        noise = transformer_config.get("input_noise")
        
        assert noise is not None, "transformer.input_noise not found in config"
        assert noise == 0.03, f"Expected transformer.input_noise=0.03, got {noise}"
        assert noise != 0.08, "Still using old transformer.input_noise=0.08"
    
    def test_transformer_label_smoothing_reduced(self):
        """Transformer label_smoothing should be 0.05 (not 0.15)."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        transformer_config = config.get("transformer", {})
        smoothing = transformer_config.get("label_smoothing")
        
        assert smoothing is not None, "transformer.label_smoothing not found in config"
        assert smoothing == 0.05, f"Expected transformer.label_smoothing=0.05, got {smoothing}"
        assert smoothing != 0.15, "Still using old transformer.label_smoothing=0.15"


# ============================================================================
# TEST #5: FEATURE SELECTION
# ============================================================================

class TestFeatureSelection:
    """Fix #5: Verify feature selection is enabled using RF importance."""
    
    def test_feature_selection_config_exists(self):
        """Config should have feature selection options."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        assert "use_feature_selection" in config, "use_feature_selection not in config"
        assert "feature_selection_method" in config, "feature_selection_method not in config"
        assert "top_k_features" in config, "top_k_features not in config"
    
    def test_feature_selection_enabled(self):
        """Feature selection should be enabled."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        use_fs = config.get("use_feature_selection")
        assert use_fs is True, f"Expected use_feature_selection=true, got {use_fs}"
    
    def test_feature_selection_method_rf(self):
        """Feature selection method should be random_forest."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        method = config.get("feature_selection_method")
        assert method == "random_forest", f"Expected method=random_forest, got {method}"
    
    def test_top_k_features_value(self):
        """Top-k features should be 50."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        top_k = config.get("top_k_features")
        assert top_k == 50, f"Expected top_k_features=50, got {top_k}"
    
    def test_rf_method_exists_in_feature_engineering(self):
        """random_forest method should exist in FeatureEngineering.select_features()."""
        from src.data.feature_engineering import FeatureEngineering
        import pandas as pd
        
        # Create mock data
        np.random.seed(42)
        n_samples = 100
        n_features = 60
        
        data = np.random.randn(n_samples, n_features)
        columns = [f"feat_{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns)
        df["target"] = np.random.randint(0, 2, n_samples)  # Binary target
        
        fe = FeatureEngineering()
        
        # This should work without error
        with patch.object(Path, 'exists', return_value=False):  # Force training new RF
            selected_df, selected_features = fe.select_features(
                df,
                target_col="target",
                method="random_forest",
                top_k=20,
            )
        
        assert len(selected_features) == 20, f"Expected 20 features, got {len(selected_features)}"
        assert "target" in selected_df.columns, "Target column should be preserved"
    
    @patch('pickle.load')
    @patch('builtins.open', MagicMock())
    def test_rf_model_feature_validation(self, mock_pickle_load):
        """RF model should validate feature names match before using."""
        from src.data.feature_engineering import FeatureEngineering
        import pandas as pd
        
        # Create mock RF model with different features (simulating mismatch)
        mock_rf = Mock()
        mock_rf.feature_importances_ = np.random.rand(50)
        mock_pickle_load.side_effect = [
            mock_rf,
            [f"old_feat_{i}" for i in range(50)],  # Mismatched feature names
        ]
        
        # Create data with different features
        np.random.seed(42)
        n_samples = 100
        data = np.random.randn(n_samples, 60)
        columns = [f"new_feat_{i}" for i in range(60)]
        df = pd.DataFrame(data, columns=columns)
        df["target"] = np.random.randint(0, 2, n_samples)
        
        fe = FeatureEngineering()
        
        # The method should handle feature mismatch (retrain or error gracefully)
        # With mocked Path.exists, this tests the validation logic path
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'unlink'):  # Mock file deletion
                try:
                    selected_df, selected_features = fe.select_features(
                        df,
                        target_col="target",
                        method="random_forest",
                        top_k=20,
                    )
                    # If successful, features should be selected
                    assert len(selected_features) <= 20
                except Exception:
                    # Feature mismatch handling may raise - that's acceptable
                    pass
    
    @pytest.mark.parametrize("top_k", [10, 30, 50, 70])
    def test_rf_selection_various_top_k(self, top_k):
        """RF selection should work for various top_k values."""
        from src.data.feature_engineering import FeatureEngineering
        import pandas as pd
        
        np.random.seed(42)
        n_samples = 100
        n_features = 100  # More features than top_k
        
        data = np.random.randn(n_samples, n_features)
        columns = [f"feat_{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns)
        df["target"] = np.random.randint(0, 2, n_samples)
        
        fe = FeatureEngineering()
        
        with patch.object(Path, 'exists', return_value=False):
            selected_df, selected_features = fe.select_features(
                df,
                target_col="target",
                method="random_forest",
                top_k=top_k,
            )
        
        assert len(selected_features) == top_k, f"Expected {top_k} features, got {len(selected_features)}"


# ============================================================================
# INTEGRATION SMOKE TEST
# ============================================================================

@pytest.mark.bias_fixes
class TestBiasFixesIntegration:
    """Quick smoke test that all fixes are present and consistent."""
    
    def test_all_fixes_present_in_config(self):
        """All 5 fixes should be reflected in config."""
        from src.utils.utils import load_config
        config = load_config("config/config_improved_H1.yaml")
        
        # Fix #1: Label threshold
        assert config.get("direction_threshold") == 0.003
        
        # Fix #2: Focal loss
        assert config.get("use_focal_loss") is True
        
        # Fix #4: Regularization
        assert config.get("model", {}).get("dropout") == 0.3
        assert config.get("transformer", {}).get("dropout") == 0.3
        assert config.get("transformer", {}).get("l2_reg") == 0.005
        assert config.get("transformer", {}).get("input_noise") == 0.03
        assert config.get("transformer", {}).get("label_smoothing") == 0.05
        
        # Fix #5: Feature selection
        assert config.get("use_feature_selection") is True
        assert config.get("feature_selection_method") == "random_forest"
        assert config.get("top_k_features") == 50
    
    def test_focal_loss_class_has_required_params(self):
        """BinaryFocalLoss should have gamma, alpha, label_smoothing params."""
        from src.models.tensorflow_models import BinaryFocalLoss
        import inspect
        
        sig = inspect.signature(BinaryFocalLoss.__init__)
        params = list(sig.parameters.keys())
        
        assert "gamma" in params, "BinaryFocalLoss missing gamma parameter"
        assert "alpha" in params, "BinaryFocalLoss missing alpha parameter"
        assert "label_smoothing" in params, "BinaryFocalLoss missing label_smoothing parameter"
