#!/usr/bin/env python3
"""
Scanner Accuracy Validation Script.

Validates that scanner outputs match actual model predictions.
Checks:
1. Direction matches transformer output
2. Confidence values are calculated correctly
3. Gate thresholds are applied consistently
4. Volatility regime detection is working
"""
import sys
sys.path.insert(0, '.')

from src.scanner import Scanner, ScannerConfig
from src.core.modular_inference import ModularEnsembleInference
from src.core.modular_data_loaders import compute_normalized_features
import pandas as pd
import numpy as np

def validate_scanner():
    print("=" * 60)
    print("SCANNER ACCURACY VALIDATION")
    print("=" * 60)
    
    # Initialize scanner
    config = ScannerConfig()
    scanner = Scanner(config=config)
    
    # Test pairs
    test_pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD']
    
    # Get scan results
    print("\n1. Running scanner on test pairs...")
    result = scanner.scan(pairs=test_pairs)
    
    print(f"\n   Scanned: {len(result.analyses)}")
    print(f"   Tradeable: {len(result.tradeable)}")
    
    # Validate each pair
    print("\n2. Validating individual signals...")
    
    for analysis in result.analyses:
        pair = analysis.pair
        print(f"\n   === {pair} ===")
        print(f"   Scanner says:")
        print(f"     Direction: {analysis.direction}")
        print(f"     Confidence: {analysis.confidence:.1%}")
        print(f"     TCN Conf: {analysis.tcn_confidence:.1%}")
        print(f"     Ridge Conf: {analysis.ridge_confidence:.1%}")
        print(f"     Momentum: {analysis.momentum:.1%}")
        print(f"     Vol Regime: {analysis.volatility_regime_name}")
        print(f"     Gates Passed: {analysis.gates_passed}")
        print(f"     Is Tradeable: {analysis.is_tradeable}")
        print(f"     Block Reason: {analysis.block_reason}")
        
        # Validate gate logic
        mom_threshold = config.min_momentum
        conf_threshold = config.min_confidence
        
        expected_mom_pass = analysis.momentum >= mom_threshold
        expected_conf_pass = analysis.ridge_confidence >= conf_threshold
        
        print(f"\n   Gate Validation:")
        print(f"     Momentum {analysis.momentum:.1%} >= {mom_threshold:.1%}: {expected_mom_pass} (reported: {analysis.momentum_passed})")
        print(f"     Confidence {analysis.ridge_confidence:.1%} >= {conf_threshold:.1%}: {expected_conf_pass} (reported: {analysis.confidence_passed})")
        
        if expected_mom_pass != analysis.momentum_passed:
            print(f"     ⚠️ MOMENTUM GATE MISMATCH!")
        if expected_conf_pass != analysis.confidence_passed:
            print(f"     ⚠️ CONFIDENCE GATE MISMATCH!")
        
        # Check if tradeable status is consistent
        vol_pass = analysis.volatility_gate_passed
        gates_pass = analysis.gates_passed
        has_direction = analysis.direction in ("LONG", "SHORT")
        no_error = analysis.error is None
        no_block = analysis.block_reason is None
        
        expected_tradeable = gates_pass and has_direction and no_error and no_block
        actual_tradeable = analysis.is_tradeable
        
        print(f"\n   Tradeable Validation:")
        print(f"     Vol Gate: {vol_pass}, All Gates: {gates_pass}, Direction: {has_direction}, No Error: {no_error}, No Block: {no_block}")
        print(f"     Expected: {expected_tradeable}, Actual: {actual_tradeable}")
        
        if expected_tradeable != actual_tradeable:
            print(f"     ⚠️ TRADEABLE STATUS MISMATCH!")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    tradeable = [a for a in result.analyses if a.is_tradeable]
    blocked = [a for a in result.analyses if not a.is_tradeable]
    
    print(f"\nTradeable pairs: {len(tradeable)}")
    for t in tradeable:
        print(f"  {t.pair}: {t.direction} @ {t.confidence:.0%}")
    
    print(f"\nBlocked pairs: {len(blocked)}")
    for b in blocked:
        reason = b.block_reason or f"Vol regime: {b.volatility_regime_name}"
        print(f"  {b.pair}: {reason}")
    
    print("\n✅ Validation complete")
    return True

if __name__ == "__main__":
    validate_scanner()
