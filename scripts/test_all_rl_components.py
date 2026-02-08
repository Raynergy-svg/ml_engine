#!/usr/bin/env python3
"""Test all 3 RL components work together."""

import numpy as np
import sys
sys.path.insert(0, '.')


def main():
    # Test 1: RL Position Sizer loads and works
    print('=' * 60)
    print('TEST 1: RL Position Sizer')
    print('=' * 60)
    
    from rl_position_sizing import RLPositionSizer
    sizer = RLPositionSizer()
    sizer.load()
    print(f'✓ Loaded: {sizer._is_trained}')
    print(f'✓ Scaler expects: {sizer.scaler.n_features_in_} features')
    
    # Test with correct feature count
    features = np.random.randn(20).astype(np.float32)
    ensemble_pred = np.array([0.65, 0.7])  # [direction_prob, confidence]
    size = sizer.get_position_size(features, ensemble_pred, account_equity=100000)
    print(f'✓ Position size: ${size:.2f} for $100k account')
    
    # Test 2: RL Exit loads and works
    print()
    print('=' * 60)
    print('TEST 2: RL Exit (Optimal Exit)')
    print('=' * 60)
    
    from stable_baselines3 import PPO
    exit_model = PPO.load('trained_data/models/ppo_optimal_exit.zip')
    print('✓ Exit model loaded')
    
    # Simulate exit observation
    obs = np.random.randn(exit_model.observation_space.shape[0]).astype(np.float32)
    action, _ = exit_model.predict(obs, deterministic=True)
    actions = ['HOLD', 'EXIT_PROFIT', 'EXIT_LOSS']
    print(f'✓ Exit decision: {actions[action]}')
    
    # Test 3: RL Gate loads and works
    print()
    print('=' * 60)
    print('TEST 3: RL Gate (Dynamic Thresholds)')
    print('=' * 60)
    
    from stable_baselines3 import SAC
    gate_model = SAC.load('trained_data/models/sac_gate_thresholds.zip')
    print('✓ Gate model loaded')
    
    obs = np.random.randn(gate_model.observation_space.shape[0]).astype(np.float32)
    action, _ = gate_model.predict(obs, deterministic=True)
    print(f'✓ Gate thresholds adjustment: {action}')
    
    print()
    print('=' * 60)
    print('ALL 3 RL COMPONENTS WORKING ✅')
    print('=' * 60)


if __name__ == '__main__':
    main()
