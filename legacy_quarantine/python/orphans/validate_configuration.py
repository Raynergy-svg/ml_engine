#!/usr/bin/env python3
"""
Configuration Validation Script

This script validates that the tuned configuration file (config_tuned.yaml)
properly integrates with main.py and supports all required functionality
including OANDA services API usage.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_configuration_loading():
    """Test that the configuration loads without errors."""
    print("🔍 Testing configuration loading...")
    
    try:
        from utils import load_config
        config = load_config("config_tuned.yaml")
        print("✅ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return None


def test_required_sections(config):
    """Test that all required sections are present."""
    print("\n🔍 Testing required sections...")

    required_sections = [
        "data", "model", "training", "buddy", "fx",
        "optimizer", "paths", "logging"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ Missing sections: {missing_sections}")
        return False
    else:
        print("✅ All required sections present")
        return True


def test_main_py_integration(config):
    """Test that configuration integrates with main.py expectations."""
    print("\n🔍 Testing main.py integration...")
    
    # Test data section
    data_ok = (
        "data_dir" in config.get("data", {}) and
        "sequence_length" in config.get("data", {}) and
        "required_features" in config.get("data", {})
    )
    
    # Test model section
    model_ok = (
        "hidden_size" in config.get("model", {}) and
        "num_layers" in config.get("model", {}) and
        "dropout" in config.get("model", {})
    )
    
    # Test training section
    training_ok = (
        "epochs" in config.get("training", {}) and
        "early_stopping_patience" in config.get("training", {})
    )
    
    # Test buddy section
    buddy_ok = (
        "stop_loss_pips" in config.get("buddy", {}) and
        "risk_per_trade_pct" in config.get("buddy", {})
    )
    
    # Test fx section
    fx_ok = (
        "enabled" in config.get("fx", {}) and
        "instruments" in config.get("fx", {}) and
        "granularity" in config.get("fx", {})
    )
    
    all_ok = data_ok and model_ok and training_ok and buddy_ok and fx_ok
    
    if all_ok:
        print("✅ All main.py integration points validated")
    else:
        print("❌ Some integration points missing:")
        if not data_ok:
            print("  - Data section incomplete")
        if not model_ok:
            print("  - Model section incomplete")
        if not training_ok:
            print("  - Training section incomplete")
        if not buddy_ok:
            print("  - Buddy section incomplete")
        if not fx_ok:
            print("  - FX section incomplete")
    
    return all_ok


def test_oanda_integration(config):
    """Test that OANDA services configuration is properly structured."""
    print("\n🔍 Testing OANDA services integration...")
    
    fx_config = config.get("fx", {})
    
    # Check FX configuration structure
    fx_structure_ok = (
        fx_config.get("enabled", False) is True and
        "instruments" in fx_config and
        "granularity" in fx_config and
        "risk" in fx_config and
        "execution" in fx_config
    )
    
    # Check instrument list
    instruments_ok = (
        isinstance(fx_config.get("instruments", []), list) and
        len(fx_config.get("instruments", [])) > 0
    )
    
    # Check risk management
    risk_ok = (
        "risk_per_trade_pct" in fx_config.get("risk", {}) and
        "atr_stop_mult" in fx_config.get("risk", {})
    )
    
    all_ok = fx_structure_ok and instruments_ok and risk_ok
    
    if all_ok:
        print("✅ OANDA services configuration validated")
        print(f"   - Instruments: {fx_config['instruments']}")
        print(f"   - Granularity: {fx_config['granularity']}")
        print(f"   - Risk per trade: {fx_config['risk']['risk_per_trade_pct']}")
    else:
        print("❌ OANDA services configuration issues found")
    
    return all_ok


def test_parameter_consistency(config):
    """Test that parameters are consistent with main.py expectations."""
    print("\n🔍 Testing parameter consistency...")
    
    issues = []
    
    # Check training parameters
    training = config.get("training", {})
    if training.get("epochs", 0) != 200:
        issues.append(f"Training epochs: expected 200, got {training.get('epochs')}")
    
    if training.get("early_stopping_patience", 0) != 20:
        issues.append(f"Early stopping patience: expected 20, got {training.get('early_stopping_patience')}")
    
    # Check model parameters
    model = config.get("model", {})
    if model.get("hidden_size", 0) != 48:
        issues.append(f"Model hidden_size: expected 48, got {model.get('hidden_size')}")
    
    if model.get("num_layers", 0) != 3:
        issues.append(f"Model num_layers: expected 3, got {model.get('num_layers')}")
    
    # Check data parameters
    data = config.get("data", {})
    if data.get("sequence_length", 0) != 120:
        issues.append(f"Data sequence_length: expected 120, got {data.get('sequence_length')}")
    
    if issues:
        print("❌ Parameter consistency issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All parameters consistent with main.py expectations")
        return True


def test_buddy_fx_integration(config):
    """Test that Buddy and FX configurations work together."""
    print("\n🔍 Testing Buddy-FX integration...")
    
    buddy = config.get("buddy", {})
    fx = config.get("fx", {})
    
    # Check that Buddy has FX-related settings
    buddy_fx_ok = (
        "stop_loss_pips" in buddy and
        "take_profit_pips" in buddy and
        "risk_per_trade_pct" in buddy and
        "tier2" in buddy
    )
    
    # Check that FX has trading settings
    fx_trading_ok = (
        "instruments" in fx and
        "granularity" in fx and
        "execution" in fx
    )
    
    # Check that risk settings are compatible
    buddy_risk = buddy.get("risk_per_trade_pct", 0)
    fx_risk = fx.get("risk", {}).get("risk_per_trade_pct", 0)
    
    risk_compatible = buddy_risk <= fx_risk if fx_risk > 0 else True
    
    all_ok = buddy_fx_ok and fx_trading_ok and risk_compatible
    
    if all_ok:
        print("✅ Buddy-FX integration validated")
        print(f"   - Buddy risk: {buddy_risk}")
        print(f"   - FX risk: {fx_risk}")
        print(f"   - Instruments: {fx.get('instruments', [])}")
    else:
        print("❌ Buddy-FX integration issues found")
    
    return all_ok


def main():
    """Run all validation tests."""
    print("🚀 Starting Configuration Validation")
    print("=" * 50)
    
    # Test 1: Configuration Loading
    config = test_configuration_loading()
    if not config:
        print("\n❌ VALIDATION FAILED: Cannot load configuration")
        return False
    
    # Test 2: Required Sections
    sections_ok = test_required_sections(config)
    
    # Test 3: Main.py Integration
    integration_ok = test_main_py_integration(config)
    
    # Test 4: OANDA Services Integration
    oanda_ok = test_oanda_integration(config)
    
    # Test 5: Parameter Consistency
    params_ok = test_parameter_consistency(config)
    
    # Test 6: Buddy-FX Integration
    buddy_fx_ok = test_buddy_fx_integration(config)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", config is not None),
        ("Required Sections", sections_ok),
        ("Main.py Integration", integration_ok),
        ("OANDA Services", oanda_ok),
        ("Parameter Consistency", params_ok),
        ("Buddy-FX Integration", buddy_fx_ok),
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("The configuration is ready for use with main.py and OANDA services.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Please review the issues above and fix the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
# — Raynergy-svg —
