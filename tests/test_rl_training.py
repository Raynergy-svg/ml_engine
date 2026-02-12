#!/usr/bin/env python3
"""Test RL training with synthetic data."""

import numpy as np
from rl_position_sizing import RLPositionSizer, RLConfig, SB3_AVAILABLE, GYM_AVAILABLE

print('Testing RL Training with synthetic data...')

if not SB3_AVAILABLE or not GYM_AVAILABLE:
    print('Missing dependencies')
    exit(1)

# Create synthetic training data
n_samples = 150  # More than min_trades=100

features = np.random.randn(n_samples, 20)  # 20 features
predictions = np.column_stack([
    np.random.uniform(0.45, 0.65, n_samples),  # direction_prob
    np.random.uniform(0.4, 0.7, n_samples),   # confidence
])
prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.3)

print(f'Features shape: {features.shape}')
print(f'Predictions shape: {predictions.shape}')
print(f'Prices shape: {prices.shape}')

# Quick training test
config = RLConfig(total_timesteps=2000)  # Very quick for testing
sizer = RLPositionSizer(config)

print('\nStarting training (2000 timesteps)...')
sizer.train(features, predictions, prices)
print('Training complete!')

# Test inference
test_features = np.random.randn(1, 20)
test_pred = np.array([[0.6, 0.7]])
position_size = sizer.get_position_size(test_features[0], test_pred[0])
print(f'\nTest inference - Position size: {position_size:.4f}')

print('\n✅ RL Position Sizer is WORKING!')
