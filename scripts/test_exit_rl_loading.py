#!/usr/bin/env python3
"""Test Exit RL loading in inference pipeline."""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("Testing Exit RL Integration")
    print("=" * 60)
    
    # Step 1: Check model file exists
    model_path = Path("trained_data/models/ppo_optimal_exit.zip")
    print(f"\n1. Model file exists: {model_path.exists()}")
    if model_path.exists():
        print(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")
    
    # Step 2: Try direct PPO load
    print("\n2. Testing direct PPO load...")
    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path))
        print(f"   ✅ Direct load successful: {type(model)}")
    except Exception as e:
        print(f"   ❌ Direct load failed: {e}")
        return 1
    
    # Step 3: Import inference config
    print("\n3. Importing InferenceConfig...")
    try:
        from src.core.modular_inference import InferenceConfig, ModularEnsembleInference
        print("   ✅ Import successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return 1
    
    # Step 4: Create ModularEnsembleInference with RL exits (pass to constructor, not config)
    print("\n4. Creating ModularEnsembleInference with RL exits enabled...")
    try:
        mi = ModularEnsembleInference(
            use_rl_exits=True,
        )
        print("   ✅ Instance created")
        print(f"   use_rl_exits = {mi.use_rl_exits}")
        print(f"   rl_exit_optimizer (before load_models): {mi.rl_exit_optimizer}")
    except Exception as e:
        print(f"   ❌ Instance creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Call load_models (may fail due to TF, but test RL loading directly)
    print("\n5. Testing RL Exit loading directly (skipping TF)...")
    try:
        from src.rl.optimal_exit_env import OptimalExitRL
        exit_rl = OptimalExitRL()
        loaded = exit_rl.load()
        print(f"   ✅ OptimalExitRL loaded: {loaded}")
        print(f"   model = {exit_rl.model}")
        mi.rl_exit_optimizer = exit_rl
        mi._rl_exits_loaded = loaded
    except Exception as e:
        print(f"   ❌ RL Exit loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 6: Test check_open_position_exit if model loaded
    if mi.rl_exit_optimizer is not None:
        print("\n6. Testing check_open_position_exit()...")
        try:
            import pandas as pd
            import numpy as np
            # Create minimal test DataFrame
            test_df = pd.DataFrame({
                'close': [1.0855],
                'atr': [0.0015],
                'momentum_5': [0.3],
            })
            result = mi.check_open_position_exit(
                df=test_df,
                unrealized_pnl_pips=5.0,
                bars_in_trade=3,
                direction='long',
                entry_price=1.0850,
                stop_loss_pips=15.0,
                take_profit_pips=30.0
            )
            print(f"   ✅ Result: {result}")
        except Exception as e:
            print(f"   ❌ check_open_position_exit failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n6. Skipping check_open_position_exit - model not loaded")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    # Step 7: Test Position Sizer RL
    print("\n7. Testing Position Sizer RL...")
    try:
        from rl_position_sizing import RLPositionSizer
        sizer = RLPositionSizer()
        loaded = sizer.load()
        print(f"   ✅ Position Sizer loaded: {loaded}")
        if loaded:
            features = np.random.randn(20)  # 20 features expected
            ensemble_pred = np.array([0.65, 0.55])
            size = sizer.get_position_size(features, ensemble_pred, account_equity=100000)
            print(f"   ✅ Position size for $100k account: ${size:.2f}")
    except Exception as e:
        print(f"   ❌ Position Sizer test failed: {e}")
    
    print("\n" + "=" * 60)
    print("ALL 3 RL COMPONENTS TESTED!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
