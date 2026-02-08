#!/usr/bin/env python3
"""Test Exit RL reward function after fixes."""

import sys
sys.path.insert(0, '.')

from src.rl.optimal_exit_env import OptimalExitRL, OptimalExitEnv  # noqa: E402

print("=" * 60)
print("TESTING FIXED EXIT RL ENVIRONMENT")
print("=" * 60)

rl = OptimalExitRL()
scenarios = rl._generate_synthetic_scenarios(10)

print('\nSample scenario stats (should show ~1-5% price ranges):')
for i, s in enumerate(scenarios[:5]):
    price_change = (s.prices[-1] - s.entry_price) / s.entry_price * 100
    max_move = (s.prices.max() - s.entry_price) / s.entry_price * 100
    min_move = (s.prices.min() - s.entry_price) / s.entry_price * 100
    print(f'  Scenario {i}: dir={s.direction:+d}, final={price_change:+.2f}%, range=[{min_move:+.2f}%, {max_move:+.2f}%]')

# Test the environment
env = OptimalExitEnv(scenarios=scenarios)
obs, _ = env.reset()

print('\n' + '=' * 60)
print('TESTING REWARD SIGNALS')
print('=' * 60)

# Test 1: Hold then exit with profit
print('\nTest 1: Hold 10 steps then exit')
obs, _ = env.reset()
total_reward = 0
for step in range(15):
    action = 0 if step < 10 else 1  # Hold then exit
    obs, reward, done, trunc, info = env.step(action)
    total_reward += reward
    if step in [0, 5, 9, 10] or 'exit_reason' in info:
        pnl = info.get('pnl', 'N/A')
        print(f'  Step {step}: action={action}, reward={reward:+.4f}, total={total_reward:+.4f}, pnl={pnl}')
    if done:
        break

# Test 2: Exit immediately
print('\nTest 2: Exit immediately (step 0)')
obs, _ = env.reset()
obs, reward, done, trunc, info = env.step(1)  # Exit immediately
print(f'  Immediate exit: reward={reward:+.4f}, pnl={info.get("pnl", "N/A")}')

# Test 3: Multiple trades in episode
print('\nTest 3: Episode with multiple trades')
obs, _ = env.reset()
total_ep_reward = 0
trade_count = 0
for step in range(500):
    # Random strategy
    if step % 10 == 9:  # Exit every 10 steps
        action = 1
    else:
        action = 0
    obs, reward, done, trunc, info = env.step(action)
    total_ep_reward += reward
    if 'exit_reason' in info:
        trade_count += 1
    if done:
        break

print(f'  Completed {trade_count} trades, total reward: {total_ep_reward:+.2f}')

print('\n' + '=' * 60)
print('VERDICT')
print('=' * 60)
if abs(total_ep_reward) > 1.0:
    print('✅ Rewards are now meaningful (non-zero)!')
    print('   Ready for training.')
else:
    print('⚠️  Rewards still too small, may need more tuning.')
