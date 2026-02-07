#!/usr/bin/env python3
"""Test Exit RL integration with inference."""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.core.modular_inference import ModularEnsembleInference

def main():
    print("=" * 60)
    print("TESTING EXIT RL INTEGRATION WITH INFERENCE")
    print("=" * 60)
    
    # Create test data
    n = 100
    df = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n) * 0.01) + 1.1000,
        'open': np.cumsum(np.random.randn(n) * 0.01) + 1.0998,
        'high': np.cumsum(np.random.randn(n) * 0.01) + 1.1002,
        'low': np.cumsum(np.random.randn(n) * 0.01) + 1.0996,
        'atr': np.abs(np.random.randn(n)) * 0.001,
        'momentum_5': np.random.randn(n) * 0.1,
    })
    
    # Initialize with RL exits enabled
    print("\n1. Loading inference with RL exits enabled...")
    ensemble = ModularEnsembleInference(use_rl_exits=True)
    print(f"   RL Exits Loaded: {ensemble._rl_exits_loaded}")
    
    # Test various scenarios
    test_cases = [
        {"pnl": 12.5, "bars": 8, "direction": "long", "desc": "Small profit, short hold"},
        {"pnl": -8.0, "bars": 3, "direction": "short", "desc": "Small loss, very short hold"},
        {"pnl": 25.0, "bars": 20, "direction": "long", "desc": "Good profit, medium hold"},
        {"pnl": -12.0, "bars": 15, "direction": "long", "desc": "Moderate loss, medium hold"},
        {"pnl": 5.0, "bars": 50, "direction": "short", "desc": "Small profit, long hold"},
    ]
    
    print("\n2. Testing exit check scenarios:")
    print("-" * 60)
    
    for i, tc in enumerate(test_cases):
        result = ensemble.check_open_position_exit(
            df=df,
            unrealized_pnl_pips=tc["pnl"],
            bars_in_trade=tc["bars"],
            direction=tc["direction"],
            entry_price=1.0950,
        )
        
        exit_icon = "🚪" if result["should_exit"] else "🔒"
        rl_icon = "🤖" if result["rl_available"] else "📊"
        
        print(f"\n   Case {i+1}: {tc['desc']}")
        print(f"   Input: PnL={tc['pnl']:.1f} pips, Bars={tc['bars']}, Dir={tc['direction']}")
        print(f"   {exit_icon} Action: {result['action']} (conf={result['confidence']:.2f})")
        print(f"   {rl_icon} Reason: {result['reason']}")
    
    # Test stop loss and take profit triggers
    print("\n3. Testing hard stop/target triggers:")
    print("-" * 60)
    
    # Stop loss hit
    result = ensemble.check_open_position_exit(
        df=df,
        unrealized_pnl_pips=-16.0,  # Past -15 pip stop
        bars_in_trade=5,
        direction="long",
        entry_price=1.0950,
    )
    print(f"\n   Stop Loss Test: PnL=-16 pips")
    print(f"   Should Exit: {result['should_exit']} (expected: True)")
    print(f"   Reason: {result['reason']}")
    
    # Take profit hit
    result = ensemble.check_open_position_exit(
        df=df,
        unrealized_pnl_pips=32.0,  # Past 30 pip target
        bars_in_trade=12,
        direction="short",
        entry_price=1.0950,
    )
    print(f"\n   Take Profit Test: PnL=+32 pips")
    print(f"   Should Exit: {result['should_exit']} (expected: True)")
    print(f"   Reason: {result['reason']}")
    
    print("\n" + "=" * 60)
    print("✅ Exit RL Integration Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
