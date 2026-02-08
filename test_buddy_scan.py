#!/usr/bin/env python3
"""
Quick test script for buddy scanner fixes.
Tests the core scanning functionality without needing full models.
"""

import sys
import logging
logging.basicConfig(level=logging.INFO)


def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("TEST 1: Import Validation")
    print("=" * 60)
    
    try:
        # Test buddy_scanner imports
        from buddy_scanner import BuddyScanner, EnhancedScanResult, ScanConfig, ALL_PAIRS, MAJOR_PAIRS, CROSS_PAIRS  # noqa: F401
        print("✓ buddy_scanner imports successful")
        
        # Verify pair counts
        assert len(MAJOR_PAIRS) == 7, f"Expected 7 major pairs, got {len(MAJOR_PAIRS)}"
        assert len(CROSS_PAIRS) == 8, f"Expected 8 cross pairs, got {len(CROSS_PAIRS)}"
        assert len(ALL_PAIRS) == 15, f"Expected 15 total pairs, got {len(ALL_PAIRS)}"
        print(f"✓ Pair counts correct: {len(MAJOR_PAIRS)} majors + {len(CROSS_PAIRS)} crosses = {len(ALL_PAIRS)} total")
        
        # Test cli.buddy_scanning imports
        from cli.buddy_scanning import buddy_scan, buddy_predict_78  # noqa: F401
        print("✓ cli.buddy_scanning imports successful")
        
        # Test internal imports used by buddy_scanner
        from src.utils import load_config  # noqa: F401
        print("✓ src.utils.load_config imports successful")
        
        from src.risk.position_sizing import PositionSizingConfig  # noqa: F401
        print("✓ src.risk.position_sizing imports successful")
        
        from src.risk.risk_management import RiskManagementConfig  # noqa: F401
        print("✓ src.risk.risk_management imports successful")
        
        from src.core.modular_inference import ModularEnsembleInference  # noqa: F401
        print("✓ src.core.modular_inference imports successful")
        
        from src.utils.oanda_practice import OandaPracticeClient  # noqa: F401
        print("✓ src.utils.oanda_practice imports successful")
        
        from src.data.feature_engineering import FeatureEngineering  # noqa: F401
        print("✓ src.data.feature_engineering imports successful")
        
        from src.utils.fx_paper import candles_to_ohlcv_df, pip_size  # noqa: F401
        print("✓ src.utils.fx_paper imports successful")
        
        from src.utils.pair_scanner import PairAnalysis  # noqa: F401
        print("✓ src.utils.pair_scanner imports successful")
        
        print("\n✅ ALL IMPORTS PASSED\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ IMPORT FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_scanner_initialization():
    """Test that BuddyScanner can be initialized."""
    print("=" * 60)
    print("TEST 2: Scanner Initialization")
    print("=" * 60)
    
    try:
        from buddy_scanner import BuddyScanner
        
        # Initialize with default config
        scanner = BuddyScanner(config_path="config/config_improved_H1.yaml")
        print("✓ Scanner initialized with default config")
        
        # Check attributes
        assert hasattr(scanner, '_scan_config'), "Missing _scan_config attribute"
        assert hasattr(scanner, '_correlation_details'), "Missing _correlation_details attribute"
        assert hasattr(scanner, '_pair_returns'), "Missing _pair_returns attribute"
        print("✓ Scanner has required attributes")
        
        # Check correlation_details is initialized as list
        assert isinstance(scanner._correlation_details, list), "correlation_details should be a list"
        assert len(scanner._correlation_details) == 0, "correlation_details should start empty"
        print("✓ Correlation details properly initialized")
        
        print("\n✅ SCANNER INITIALIZATION PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ SCANNER INITIALIZATION FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_scan_config():
    """Test ScanConfig dataclass."""
    print("=" * 60)
    print("TEST 3: Scan Configuration")
    print("=" * 60)
    
    try:
        from buddy_scanner import ScanConfig
        
        # Test default initialization
        config = ScanConfig()
        print("✓ ScanConfig created with defaults")
        
        # Verify key settings
        assert config.parallel_workers == 4, f"Expected 4 workers, got {config.parallel_workers}"
        assert config.backtest_window == 50, f"Expected 50 candle backtest, got {config.backtest_window}"
        assert config.min_sl_pips == 15.0, f"Expected 15 pip SL, got {config.min_sl_pips}"
        assert config.max_sl_pips == 15.0, f"Expected 15 pip max SL, got {config.max_sl_pips}"
        print("✓ Config settings correct:")
        print(f"  - Parallel workers: {config.parallel_workers}")
        print(f"  - Backtest window: {config.backtest_window} candles")
        print(f"  - SL pips: {config.min_sl_pips}-{config.max_sl_pips}")
        print(f"  - TP pips: {config.min_tp_pips}-{config.max_tp_pips}")
        print(f"  - Risk per trade: {config.risk_per_trade_pct:.1%}")
        
        # Test from_dict
        test_dict = {
            "parallel_workers": 8,
            "backtest_window": 100,
        }
        config2 = ScanConfig.from_dict(test_dict)
        assert config2.parallel_workers == 8, "from_dict not working"
        assert config2.backtest_window == 100, "from_dict not working"
        print("✓ ScanConfig.from_dict() working")
        
        print("\n✅ SCAN CONFIGURATION PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ SCAN CONFIGURATION FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_drift_detection_signature():
    """Test that drift detection has correct signature."""
    print("=" * 60)
    print("TEST 4: Drift Detection Method")
    print("=" * 60)
    
    try:
        from buddy_scanner import BuddyScanner
        import inspect
        
        scanner = BuddyScanner(config_path="config/config_improved_H1.yaml")
        
        # Check method exists
        assert hasattr(scanner, '_check_model_drift'), "Missing _check_model_drift method"
        
        # Check signature
        sig = inspect.signature(scanner._check_model_drift)
        params = list(sig.parameters.keys())
        
        assert 'pair' in params, "drift detection missing 'pair' parameter"
        assert 'recent_accuracy' in params, "drift detection missing 'recent_accuracy' parameter"
        print(f"✓ Drift detection signature: {params}")
        
        # Check default for recent_accuracy
        assert sig.parameters['recent_accuracy'].default is None, "recent_accuracy should default to None"
        print("✓ recent_accuracy has correct default (None)")
        
        print("\n✅ DRIFT DETECTION METHOD PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ DRIFT DETECTION METHOD FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BUDDY SCANNER VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Import Validation", test_imports()))
    results.append(("Scanner Initialization", test_scanner_initialization()))
    results.append(("Scan Configuration", test_scan_config()))
    results.append(("Drift Detection Method", test_drift_detection_signature()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Scanner is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
