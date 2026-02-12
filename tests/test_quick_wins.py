#!/usr/bin/env python3
"""Test script for 4 Quick Wins implementation."""

print('Testing Quick Wins Implementation...')
print('=' * 50)

# 1. LightGBM
print('\n1. LightGBM Wrapper:')
try:
    import lightgbm as lgb
    print(f'   ✓ LightGBM {lgb.__version__} available')
    from ensemble_model import LIGHTGBM_AVAILABLE
    print(f'   ✓ LightGBMWrapper imported, LIGHTGBM_AVAILABLE={LIGHTGBM_AVAILABLE}')
except ImportError as e:
    print(f'   ✗ LightGBM not installed: {e}')
    print('   Install with: conda install -n ml_engine_py312 lightgbm')

# 2. Label Smoothing in config
print('\n2. Label Smoothing:')
try:
    import yaml
    with open('config_m1_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f'   ✓ label_smoothing: {config.get("label_smoothing", "NOT SET")}')
    print(f'   ✓ state_label_smoothing: {config.get("state_label_smoothing", "NOT SET")}')
except Exception as e:
    print(f'   ✗ Error: {e}')

# 3. Focal Loss (already implemented)
print('\n3. Focal Loss with Label Smoothing:')
try:
    from tensorflow_engine import BinaryFocalLoss
    fl = BinaryFocalLoss(gamma=2.0, alpha=0.5, label_smoothing=0.1)
    print(f'   ✓ BinaryFocalLoss supports label_smoothing={fl.label_smoothing}')
except Exception as e:
    print(f'   ✗ Error: {e}')

# 4. Model EMA
print('\n4. Model EMA Callback:')
try:
    from tensorflow_engine import ModelEMACallback
    ema = ModelEMACallback(decay=0.999)
    print(f'   ✓ ModelEMACallback created with decay={ema.decay}')
    print(f'   ✓ use_model_ema: {config.get("use_model_ema", "NOT SET")}')
    print(f'   ✓ ema_decay: {config.get("ema_decay", "NOT SET")}')
except Exception as e:
    print(f'   ✗ Error: {e}')

print('\n' + '=' * 50)
print('Quick Wins test complete!')
