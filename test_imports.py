#!/usr/bin/env python3
"""Test if all imports work correctly after restructuring."""

import sys
import traceback

def test_import(module_path, items=None):
    """Test importing a module or specific items from it."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        if items:
            for item in items:
                getattr(mod, item)
            print(f"✓ from {module_path} import {', '.join(items)}")
        else:
            print(f"✓ import {module_path}")
        return True
    except Exception as e:
        print(f"✗ Error importing {module_path}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing imports after restructuring...\n")

    tests = [
        # Core utilities
        ("src.utils", ["setup_logging", "load_config"]),

        # Core modules
        ("src.core.modular_data_loaders", None),
        ("src.core.modular_inference", ["ModularEnsembleInference", "InferenceConfig"]),

        # Models
        ("src.models.tensorflow_models", None),
        ("src.models.tensorflow_engine", None),
        ("src.models.ensemble_model", None),

        # Data
        ("src.data.data_processing", None),
        ("src.data.feature_engineering", None),

        # Risk
        ("src.risk.confidence_calibration", None),
        ("src.risk.fx_guardrails", None),

        # Utils
        ("src.utils.oanda_practice", ["OandaPracticeClient"]),
    ]

    passed = 0
    failed = 0

    for module_path, items in tests:
        if test_import(module_path, items):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)
