#!/usr/bin/env python3
"""Test RL training integration with real online learning buffer."""

from rich.console import Console
from pathlib import Path
import json

console = Console()

print("=" * 60)
print("Testing RL Training Integration")
print("=" * 60)

# Check online learning buffer
ol_buffer = Path('trained_data/online_learning/trade_buffer.json')
if ol_buffer.exists():
    with open(ol_buffer) as f:
        data = json.load(f)
    trades_count = len(data.get('trades', []))
    print(f"Online Learning Buffer: {trades_count} trades")
else:
    print("Online Learning Buffer: NOT FOUND")
    exit(1)

# Import and test
print("\nTesting _train_rl_position_sizer_if_ready...")
try:
    from main import _train_rl_position_sizer_if_ready
    
    # Test with lower threshold (we have 25 trades)
    result = _train_rl_position_sizer_if_ready(
        console=console,
        rl_timesteps=5000,  # Quick test
        min_trades=20,  # Lower threshold for testing
    )
    print(f"\nRL Training result: {result}")
    
    if result:
        print("✅ RL Training Integration WORKING!")
    else:
        print("⚠️  RL training did not run (check output above)")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
