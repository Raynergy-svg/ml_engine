"""
Test suite for unified configuration defaults.

Verifies that:
1. Constants module defines correct values
2. All code locations use the shared constants
3. Values match config/config_improved_H1.yaml
"""

import pytest
import yaml
from pathlib import Path


def test_direction_defaults_exist():
    """Test that DIRECTION_DEFAULTS constant is defined correctly."""
    from src.core.constants import DIRECTION_DEFAULTS
    
    assert 'threshold' in DIRECTION_DEFAULTS
    assert 'lookahead' in DIRECTION_DEFAULTS
    # Phase4: Stricter direction threshold (75 bps)
    assert DIRECTION_DEFAULTS['threshold'] == 0.0075
    assert DIRECTION_DEFAULTS['lookahead'] == 24


def test_inference_defaults_exist():
    """Test that INFERENCE_DEFAULTS constant is defined correctly."""
    from src.core.constants import INFERENCE_DEFAULTS
    
    assert 'min_tcn_probability' in INFERENCE_DEFAULTS
    assert 'min_confidence' in INFERENCE_DEFAULTS
    assert 'min_momentum' in INFERENCE_DEFAULTS
    assert 'max_drawdown_pct' in INFERENCE_DEFAULTS
    
    assert INFERENCE_DEFAULTS['min_tcn_probability'] == 0.60
    assert INFERENCE_DEFAULTS['min_confidence'] == 50.0
    assert INFERENCE_DEFAULTS['min_momentum'] == 0.20
    assert INFERENCE_DEFAULTS['max_drawdown_pct'] == 0.025


def test_constants_match_yaml_config():
    """Test that constants match config/config_improved_H1.yaml."""
    from src.core.constants import DIRECTION_DEFAULTS, INFERENCE_DEFAULTS
    
    config_path = Path(__file__).parent.parent / 'config' / 'config_improved_H1.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check direction defaults
    assert DIRECTION_DEFAULTS['threshold'] == config['direction_threshold']
    assert DIRECTION_DEFAULTS['lookahead'] == config['direction_lookahead']
    
    # Check inference defaults
    inference_config = config.get('inference', {})
    assert INFERENCE_DEFAULTS['min_tcn_probability'] == inference_config.get('min_tcn_probability')
    assert INFERENCE_DEFAULTS['min_confidence'] == inference_config.get('min_confidence')
    assert INFERENCE_DEFAULTS['min_momentum'] == inference_config.get('min_momentum')
    assert INFERENCE_DEFAULTS['max_drawdown_pct'] == inference_config.get('max_drawdown_pct')


def test_inference_config_uses_constant():
    """Test that InferenceConfig uses INFERENCE_DEFAULTS for min_tcn_probability."""
    from src.core.modular_inference import InferenceConfig
    from src.core.constants import INFERENCE_DEFAULTS
    
    config = InferenceConfig()
    assert config.min_tcn_probability == INFERENCE_DEFAULTS['min_tcn_probability']
    assert config.min_tcn_probability == 0.60


def test_load_direction_data_signature():
    """Test that load_direction_data function signature uses constants."""
    from src.core.modular_data_loaders import load_direction_data
    from src.core.constants import DIRECTION_DEFAULTS
    import inspect
    
    sig = inspect.signature(load_direction_data)
    
    # Check default values in signature
    assert sig.parameters['lookahead'].default == DIRECTION_DEFAULTS['lookahead']
    assert sig.parameters['threshold'].default == DIRECTION_DEFAULTS['threshold']


def test_load_all_modular_data_signature():
    """Test that load_all_modular_data function signature uses constants."""
    from src.core.modular_data_loaders import load_all_modular_data
    from src.core.constants import DIRECTION_DEFAULTS
    import inspect
    
    sig = inspect.signature(load_all_modular_data)
    
    # Check default values in signature
    assert sig.parameters['direction_threshold'].default == DIRECTION_DEFAULTS['threshold']
    assert sig.parameters['direction_lookahead'].default == DIRECTION_DEFAULTS['lookahead']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
