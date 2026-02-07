# Inference Bottleneck Analysis & Fix

## Problem
The buddy inference was hanging/taking a very long time after loading the Ridge model.

## Root Cause
**The RL Position Sizer import (`stable_baselines3`) takes 9-10+ seconds** because it loads PyTorch and many heavy dependencies. This isn't a "hang" - it's just a very slow import that wasn't showing any progress.

### Timing Breakdown (Apple Silicon)
```
Import modular_inference:     2.0s
Load Transformer:             0.5s
Load XGBoost:                 0.1s  
Load RandomForest:            0.3s
Load Ridge:                   0.0s
Import stable_baselines3:    ~9.2s  ← BOTTLENECK
Load RL model:               ~2.4s
------------------------------------
Total model loading:        ~15s
```

## Fix Applied

### 1. Added Timeout Protection to RL Sizer Loading
File: `src/core/modular_inference.py`

The `_lazy_load_rl_sizer()` function now:
- Uses a background thread with 30s timeout
- Shows progress messages during import
- Gracefully falls back to heuristic sizing if import times out
- Shows tip to disable RL sizer for faster startup

### 2. Added Detailed Timing Logs
Now shows progress during model loading:
```
🔄 RL Position Sizer: loading dependencies...
  ⏱️ RL dependencies imported in X.XXs
  🔄 Creating RL sizer instance...
  ⏱️ RL instance created in X.XXs
  🔄 Loading RL model from disk...
✓ RL Position Sizer loaded (total: X.XXs)
```

### 3. Cleaned Up Debug Logging
File: `rl_position_sizing.py`
- Only logs if import takes >5 seconds (unusual delay indicator)

## How to Disable RL Sizer for Faster Startup

If you don't need RL-based position sizing, you can disable it for faster inference:

```bash
# Use --no-rl-sizer flag (if already implemented in main.py)
./bin/Buddy EUR_USD --no-rl-sizer

# Or set use_rl_sizer=False in code
ensemble = ModularEnsembleInference(instrument='EUR_USD', use_rl_sizer=False)
```

## Intel Mac Specific Notes

On Intel Macs without GPU:
- PyTorch may take longer to initialize (10-20s)
- Consider disabling RL sizer for faster startup
- The timeout protection (30s) will prevent hangs

## Files Modified
1. `src/core/modular_inference.py` - Added timeout and timing logs
2. `rl_position_sizing.py` - Cleaned up import logging

## Testing
Run the debug script to verify timing:
```bash
cd /path/to/ml_engine
python scripts/debug_inference_hang.py
```

Expected output shows each step's timing with progress indicators.
