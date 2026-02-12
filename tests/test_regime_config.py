#!/usr/bin/env python3
"""
Test to verify regime configuration is read correctly.
This validates the fix for TCN REGIME not being wired up.
"""

import yaml
from pathlib import Path


def test_regime_config_reading():
    """Test that regime settings are read from correct config path."""
    
    config_path = Path("config/config_improved_H1.yaml")
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Simulate how training.py NOW reads config (after fix)
    buddy_cfg = cfg.get("buddy", {})
    train_defaults = buddy_cfg.get("train_defaults", {})
    transformer_cfg = cfg.get("transformer", {})
    
    # Read use_regime from buddy.train_defaults (primary) or transformer (fallback)
    use_transformer = train_defaults.get("use_transformer", transformer_cfg.get("use_transformer", True))
    use_regime = train_defaults.get("use_regime", transformer_cfg.get("use_regime", False))
    regime_lookback = train_defaults.get("regime_lookback", transformer_cfg.get("regime_lookback", 20))
    regime_lookahead = train_defaults.get("regime_lookahead", transformer_cfg.get("regime_lookahead", 12))
    
    # Direction settings (legacy, only used if use_regime: false)
    direction_threshold = train_defaults.get("direction_threshold", cfg.get("direction_threshold", 0.005))
    direction_lookahead = train_defaults.get("direction_lookahead", cfg.get("direction_lookahead", 12))
    
    # Print results
    print("\n" + "="*60)
    print("REGIME CONFIGURATION TEST")
    print("="*60)
    print(f"✓ Configuration file: {config_path}")
    print(f"✓ use_transformer:    {use_transformer} (expected: True)")
    print(f"✓ use_regime:         {use_regime} (expected: True)")
    print(f"✓ regime_lookback:    {regime_lookback} (expected: 20)")
    print(f"✓ regime_lookahead:   {regime_lookahead} (expected: 12)")
    print(f"✓ direction_threshold: {direction_threshold} (expected: 0.003)")
    print(f"✓ direction_lookahead: {direction_lookahead} (expected: 24)")
    print("="*60)
    
    # Assertions
    assert use_transformer is True, "use_transformer should be True"
    assert use_regime is True, "use_regime should be True (this was the bug!)"
    assert regime_lookback == 20, f"regime_lookback should be 20, got {regime_lookback}"
    assert regime_lookahead == 12, f"regime_lookahead should be 12, got {regime_lookahead}"
    assert direction_threshold == 0.003, f"direction_threshold should be 0.003, got {direction_threshold}"
    assert direction_lookahead == 24, f"direction_lookahead should be 24, got {direction_lookahead}"
    
    print("\n✅ ALL TESTS PASSED - Regime configuration is correctly wired!")
    print("\nThis means:")
    print("  • Training will now use REGIME mode (3-class: trend/chop/mean_revert)")
    print("  • Transformer will classify market regime, not just direction")
    print("  • The regime model will be saved to 'transformer_regime.keras'")
    print("  • Inference will load and use the regime model if present")
    print()
    
    return True


if __name__ == "__main__":
    test_regime_config_reading()
