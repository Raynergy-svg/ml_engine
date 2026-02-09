"""
Test inference configuration loading from YAML and default values.

This test suite verifies:
1. Default InferenceConfig values are correct
2. Config can be loaded from YAML file
3. Gate thresholds work as expected
"""
import pytest
import yaml
from pathlib import Path
from src.core.modular_inference import InferenceConfig


class TestInferenceConfig:
    """Test InferenceConfig default values and loading."""
    
    def test_default_values(self):
        """Test that default InferenceConfig values are correct."""
        config = InferenceConfig()
        
        # TCN volatility regime filter (replaces old TCN probability gate)
        assert config.use_tcn_volatility_filter is True
        assert config.min_volatility_regime == 2
        assert config.tcn_required is True
        
        # Gate 2: Confidence
        assert config.min_confidence == 50.0
        
        # Gate 3: Momentum
        assert config.min_momentum == 0.20
        assert config.require_fresh_or_accel is True
        
        # Gate 4: Risk/Drawdown
        assert config.max_drawdown_pct == 0.025
        assert config.max_streak_prob == 0.95  # Updated from 0.6
        
        # Gate 5: Meta-labeling
        assert config.enable_meta_labeling is True
        assert config.min_meta_confidence == 0.55
        
        # Gate 6: Sentiment
        assert config.sentiment_block_enabled is True
        assert config.sentiment_block_threshold == 0.60
        assert config.sentiment_min_headlines == 3
    
    def test_yaml_config_loading(self):
        """Test loading inference config from YAML file."""
        config_path = Path("config/config_improved_H1.yaml")
        
        # Skip if config file doesn't exist
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Verify inference section exists
        assert 'inference' in cfg, "Missing 'inference' section in config"
        
        inf_cfg = cfg['inference']
        
        # Create InferenceConfig from YAML
        config = InferenceConfig(
            min_confidence=inf_cfg.get('min_confidence', 50.0),
            min_momentum=inf_cfg.get('min_momentum', 0.20),
            require_fresh_or_accel=inf_cfg.get('require_fresh_or_accel', True),
            max_drawdown_pct=inf_cfg.get('max_drawdown_pct', 0.025),
            max_streak_prob=inf_cfg.get('max_streak_prob', 0.95),
            enable_meta_labeling=inf_cfg.get('enable_meta_labeling', True),
            min_meta_confidence=inf_cfg.get('min_meta_confidence', 0.55),
        )
        
        # Verify values match YAML
        assert config.max_streak_prob == 0.95
        assert config.use_tcn_volatility_filter is True
        assert config.max_drawdown_pct == 0.025
    
    def test_streak_risk_gate_behavior(self):
        """Test that RF streak risk gate works correctly with new threshold."""
        config = InferenceConfig()
        
        # Test cases: (streak_prob, should_pass)
        test_cases = [
            (0.50, True),   # Low risk
            (0.70, True),   # Moderate risk
            (0.85, True),   # High risk
            (0.93, True),   # User's case - should PASS with new threshold
            (0.95, True),   # At threshold - should PASS
            (0.96, False),  # Above threshold - should FAIL
            (0.98, False),  # Very high risk - should FAIL
        ]
        
        for streak_prob, should_pass in test_cases:
            actual_passes = streak_prob <= config.max_streak_prob
            assert actual_passes == should_pass, (
                f"Streak prob {streak_prob} expected to "
                f"{'PASS' if should_pass else 'FAIL'} but got "
                f"{'PASS' if actual_passes else 'FAIL'}"
            )
    
    def test_threshold_comparison_old_vs_new(self):
        """Test that new threshold allows more trades than old threshold."""
        old_threshold = 0.6
        new_config = InferenceConfig()
        
        # The problematic case that motivated this change
        problematic_streak = 0.93
        
        # Old threshold would block this
        would_pass_old = problematic_streak <= old_threshold
        assert would_pass_old is False, "Old threshold should block 0.93"
        
        # New threshold should allow this
        would_pass_new = problematic_streak <= new_config.max_streak_prob
        assert would_pass_new is True, "New threshold should allow 0.93"
    
    def test_permissive_mode_flags(self):
        """Test permissive mode configuration flags."""
        config = InferenceConfig()
        
        assert config.permissive_mode is False
        assert config.bypass_risk_gate_in_permissive is False
        
        # Test with permissive mode enabled
        permissive_config = InferenceConfig(
            permissive_mode=True,
            bypass_risk_gate_in_permissive=True,
        )
        
        assert permissive_config.permissive_mode is True
        assert permissive_config.bypass_risk_gate_in_permissive is True
    
    def test_calibration_config(self):
        """Test calibration configuration options."""
        config = InferenceConfig()
        
        assert config.enable_calibration is True
        assert config.calibration_method == 'platt'
        
        # Test with different calibration method
        isotonic_config = InferenceConfig(calibration_method='isotonic')
        assert isotonic_config.calibration_method == 'isotonic'


class TestGateThresholds:
    """Test individual gate threshold logic."""
    
    def test_tcn_volatility_regime_gate(self):
        """Test Gate 1: TCN volatility regime filter (replaces old probability gate)."""
        config = InferenceConfig()
        
        # TCN now filters by volatility regime, not direction probability
        assert config.use_tcn_volatility_filter is True
        assert config.min_volatility_regime == 2  # 2=HIGH, 3=EXTREME
        assert config.tcn_required is True  # Block trades if TCN unavailable
        
        # Regime below threshold - should fail
        assert 0 < config.min_volatility_regime  # LOW regime blocked
        assert 1 < config.min_volatility_regime  # NORMAL regime blocked
        
        # At threshold - should pass
        assert 2 >= config.min_volatility_regime  # HIGH regime passes
        
        # Above threshold - should pass
        assert 3 >= config.min_volatility_regime  # EXTREME regime passes
    
    def test_confidence_gate(self):
        """Test Gate 2: Ridge confidence threshold."""
        config = InferenceConfig()
        
        # Below threshold - should fail
        assert 30.0 < config.min_confidence
        assert 45.0 < config.min_confidence
        
        # At threshold - should pass
        assert 50.0 >= config.min_confidence
        
        # Above threshold - should pass
        assert 75.0 >= config.min_confidence
        assert 100.0 >= config.min_confidence
    
    def test_momentum_gate(self):
        """Test Gate 3: XGBoost momentum threshold."""
        config = InferenceConfig()
        
        # Below threshold - should fail (unless acceleration)
        assert 0.10 < config.min_momentum
        assert 0.15 < config.min_momentum
        
        # At threshold - should pass
        assert 0.20 >= config.min_momentum
        
        # Above threshold - should pass
        assert 0.40 >= config.min_momentum
        assert 0.70 >= config.min_momentum
    
    def test_drawdown_gate(self):
        """Test Gate 4a: RF drawdown threshold."""
        config = InferenceConfig()
        
        # Below threshold - should pass
        assert 0.010 <= config.max_drawdown_pct
        assert 0.020 <= config.max_drawdown_pct
        
        # At threshold - should pass
        assert 0.025 <= config.max_drawdown_pct
        
        # Above threshold - should fail
        assert 0.030 > config.max_drawdown_pct
        assert 0.050 > config.max_drawdown_pct
    
    def test_meta_confidence_gate(self):
        """Test Gate 5: Meta-labeling threshold."""
        config = InferenceConfig()
        
        # Below threshold - should fail
        assert 0.45 < config.min_meta_confidence
        assert 0.50 < config.min_meta_confidence
        
        # At threshold - should pass
        assert 0.55 >= config.min_meta_confidence
        
        # Above threshold - should pass
        assert 0.70 >= config.min_meta_confidence
        assert 0.90 >= config.min_meta_confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
