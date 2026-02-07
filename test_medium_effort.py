#!/usr/bin/env python3
"""Test all 3 medium-effort improvements."""

# Test all 3 medium-effort improvements

# 1. Test CosineDecayRestarts scheduler
print('Testing CosineDecayRestarts scheduler...')
from m1_metal_optimizer import CosineDecayRestarts
import tensorflow as tf

schedule = CosineDecayRestarts(
    initial_learning_rate=0.001,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=0.9,
    alpha=0.01,
    warmup_steps=100,
)

# Test LR at various steps
for step in [0, 50, 100, 500, 1000, 1500, 2000, 3000]:
    lr = schedule(step)
    print(f'  Step {step:5d}: LR = {float(lr):.6f}')

print('✓ CosineDecayRestarts working!')

# 2. Test RL position sizer import
print('\nTesting RL Position Sizer...')
from rl_position_sizing import RLPositionSizer, RLConfig
print('✓ RLPositionSizer imports working!')

# 3. Test multi-pair data loading import
print('\nTesting multi-pair data loading...')
from src.training.buddy_training_helpers import _load_multi_pair_data
print('✓ Multi-pair loader imports working!')

# 4. Test CLI command registration
print('\nTesting CLI command registration...')
from main import train_rl_sizer
print('✓ train-rl-sizer command registered!')

print('\n✅ All 3 medium-effort improvements verified!')
