#!/usr/bin/env python3
"""Validate repository structure after restructuring."""

import sys
from pathlib import Path


def check_file_exists(path, description):
    """Check if a file exists and report."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} MISSING: {path}")
        return False


def check_dir_exists(path, description):
    """Check if a directory exists and report."""
    if Path(path).is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} MISSING: {path}")
        return False


def check_import_in_file(file_path, import_statement, description):
    """Check if a file contains a specific import."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            if import_statement in content:
                print(f"✓ {description}")
                return True
            else:
                print(f"✗ {description} - Import not found: {import_statement}")
                return False
    except Exception as e:
        print(f"✗ {description} - Error reading file: {e}")
        return False


if __name__ == "__main__":
    print("Validating repository structure...\n")
    print("="*70)

    checks = []

    # Check main entry point
    print("\n📁 Main Entry Point:")
    checks.append(check_file_exists("main.py", "Main entry point"))
    checks.append(check_file_exists("bin/Buddy", "Buddy CLI script"))

    # Check src directory structure
    print("\n📁 Source Directory Structure:")
    checks.append(check_dir_exists("src", "src directory"))
    checks.append(check_dir_exists("src/core", "src/core"))
    checks.append(check_dir_exists("src/models", "src/models"))
    checks.append(check_dir_exists("src/data", "src/data"))
    checks.append(check_dir_exists("src/risk", "src/risk"))
    checks.append(check_dir_exists("src/training", "src/training"))
    checks.append(check_dir_exists("src/utils", "src/utils"))
    checks.append(check_dir_exists("src/inference", "src/inference"))

    # Check core files
    print("\n📄 Core Module Files:")
    checks.append(check_file_exists("src/core/__init__.py", "core __init__.py"))
    checks.append(check_file_exists("src/core/modular_data_loaders.py", "modular_data_loaders.py"))
    checks.append(check_file_exists("src/core/modular_inference.py", "modular_inference.py"))

    # Check model files
    print("\n📄 Model Files:")
    checks.append(check_file_exists("src/models/__init__.py", "models __init__.py"))
    checks.append(check_file_exists("src/models/tensorflow_models.py", "tensorflow_models.py"))
    checks.append(check_file_exists("src/models/tensorflow_engine.py", "tensorflow_engine.py"))
    checks.append(check_file_exists("src/models/ensemble_model.py", "ensemble_model.py"))

    # Check data files
    print("\n📄 Data Processing Files:")
    checks.append(check_file_exists("src/data/__init__.py", "data __init__.py"))
    checks.append(check_file_exists("src/data/data_processing.py", "data_processing.py"))
    checks.append(check_file_exists("src/data/feature_engineering.py", "feature_engineering.py"))

    # Check risk files
    print("\n📄 Risk Management Files:")
    checks.append(check_file_exists("src/risk/__init__.py", "risk __init__.py"))
    checks.append(check_file_exists("src/risk/fx_guardrails.py", "fx_guardrails.py"))
    checks.append(check_file_exists("src/risk/confidence_calibration.py", "confidence_calibration.py"))

    # Check training files
    print("\n📄 Training Files:")
    checks.append(check_file_exists("src/training/__init__.py", "training __init__.py"))
    checks.append(check_file_exists("src/training/modular_trainers.py", "modular_trainers.py"))

    # Check utils files
    print("\n📄 Utility Files:")
    checks.append(check_file_exists("src/utils/__init__.py", "utils __init__.py"))
    checks.append(check_file_exists("src/utils/utils.py", "utils.py"))
    checks.append(check_file_exists("src/utils/oanda_practice.py", "oanda_practice.py"))

    # Check config files
    print("\n📄 Configuration Files:")
    checks.append(check_file_exists("config/config_improved_H1.yaml", "config_improved_H1.yaml"))

    # Check documentation
    print("\n📄 Documentation:")
    checks.append(check_file_exists("README.md", "README.md"))
    checks.append(check_dir_exists("docs", "docs directory"))

    # Check imports in main.py
    print("\n🔍 Import Validation:")
    checks.append(check_import_in_file("main.py", "from src.utils import", "main.py uses src.utils"))
    checks.append(check_import_in_file("main.py", "from src.core.modular_data_loaders", "main.py uses src.core"))
    checks.append(check_import_in_file("main.py", "from src.training.modular_trainers", "main.py uses src.training"))

    # Check no legacy files in root
    print("\n🧹 Clean Root Directory:")
    legacy_files = [
        "ml_engine_enhanced.py",
        "neural_network_integrator_enhanced.py",
        "models_enhanced.py"
    ]
    for file in legacy_files:
        if not Path(file).exists():
            print(f"✓ Legacy file removed: {file}")
            checks.append(True)
        else:
            print(f"✗ Legacy file still exists: {file}")
            checks.append(False)

    # Summary
    print("\n" + "="*70)
    passed = sum(checks)
    total = len(checks)
    print(f"Validation Results: {passed}/{total} checks passed")
    print("="*70)

    if passed == total:
        print("\n🎉 SUCCESS: Repository structure is valid!")
        sys.exit(0)
    else:
        print(f"\n⚠️  WARNING: {total - passed} checks failed")
        sys.exit(1)
