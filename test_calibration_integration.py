#!/usr/bin/env python3
"""Test script to verify confidence calibration integration."""

import logging
logging.basicConfig(level=logging.INFO)

from src.core.modular_inference import ModularEnsembleInference, InferenceConfig  # noqa: E402


def test_calibration_integration():
    """Test the calibration integration in modular_inference."""
    print("=" * 60)
    print("CONFIDENCE CALIBRATION INTEGRATION TEST")
    print("=" * 60)
    
    # Create inference with calibration enabled
    config = InferenceConfig(enable_calibration=True, calibration_method='platt')
    engine = ModularEnsembleInference(config=config, enable_market_intelligence=False)
    
    # Check calibration status before loading models
    print("\n=== Before load_models ===")
    status = engine.get_calibration_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    # Load models (this triggers _load_calibration)
    print("\n=== Loading models ===")
    try:
        engine.load_models()
    except Exception as e:
        print(f"Model loading issue (expected if no trained models): {e}")
    
    # Check calibration status after loading
    print("\n=== After load_models ===")
    status = engine.get_calibration_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    # Test _apply_calibration directly
    print("\n=== Testing _apply_calibration ===")
    raw_prob = 0.65
    calibrated, applied = engine._apply_calibration(raw_prob)
    print(f"  Raw probability: {raw_prob}")
    print(f"  Calibrated probability: {calibrated}")
    print(f"  Calibration applied: {applied}")
    
    # Test edge cases
    print("\n=== Edge cases ===")
    for p in [0.0, 0.5, 0.55, 0.75, 0.99, 1.0]:
        cal, app = engine._apply_calibration(p)
        print(f"  {p:.2f} -> {cal:.3f} (applied: {app})")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    # Summary
    if status['calibrator_fitted']:
        print("\n✓ Calibration is ACTIVE and fitted")
        print(f"  Method: {status['method']}")
    elif status['enabled'] and status['calibrator_exists']:
        print("\nℹ Calibration is ENABLED but calibrator not fitted")
        print("  Raw probabilities will be used (graceful fallback)")
        print("  To train calibration: python main.py train-buddy --calibrate")
    else:
        print("\n⚠ Calibration is DISABLED")


if __name__ == "__main__":
    test_calibration_integration()
