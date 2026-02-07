#!/usr/bin/env python3
"""Analyze RL training evaluation results from NPZ files."""

import numpy as np
from pathlib import Path


def analyze_evaluations(npz_path: Path, name: str):
    """Analyze evaluation results from NPZ file."""
    print("=" * 60)
    print(f"{name} EVALUATION RESULTS")
    print("=" * 60)
    
    if not npz_path.exists():
        print(f"  File not found: {npz_path}")
        return
    
    data = np.load(npz_path)
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']
    
    print(f"Evaluations: {len(timesteps)} checkpoints over {int(timesteps[-1]):,} timesteps")
    print(f"Results shape: {results.shape}")
    print()
    
    # Results is 2D: [eval_num, episode_rewards]
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    
    print("Reward Progression:")
    print(f"  Start ({int(timesteps[0]):,} steps):  {mean_rewards[0]:.2f} +/- {std_rewards[0]:.2f}")
    mid_idx = len(mean_rewards) // 2
    print(f"  Mid ({int(timesteps[mid_idx]):,} steps):    {mean_rewards[mid_idx]:.2f} +/- {std_rewards[mid_idx]:.2f}")
    print(f"  End ({int(timesteps[-1]):,} steps):  {mean_rewards[-1]:.2f} +/- {std_rewards[-1]:.2f}")
    print()
    
    max_idx = mean_rewards.argmax()
    min_idx = mean_rewards.argmin()
    print(f"  Best reward:  {mean_rewards.max():.2f} at step {int(timesteps[max_idx]):,}")
    print(f"  Worst reward: {mean_rewards.min():.2f} at step {int(timesteps[min_idx]):,}")
    print()
    
    mean_lengths = ep_lengths.mean(axis=1)
    print(f"Episode Length: {mean_lengths[0]:.0f} -> {mean_lengths[-1]:.0f} (start -> end)")
    
    # Check for learning signal
    improvement = mean_rewards[-1] - mean_rewards[0]
    print()
    if improvement > 0:
        print(f"✅ LEARNING DETECTED: +{improvement:.2f} reward improvement")
    elif improvement < -1:
        print(f"⚠️  REGRESSION: {improvement:.2f} reward decrease")
    else:
        print(f"⚠️  FLAT: {improvement:.2f} minimal change (may need more training or reward shaping)")
    
    # Check variance
    if std_rewards[-1] < std_rewards[0]:
        print(f"✅ Variance reduced: {std_rewards[0]:.2f} -> {std_rewards[-1]:.2f} (more consistent)")
    else:
        print(f"📊 Variance: {std_rewards[0]:.2f} -> {std_rewards[-1]:.2f}")
    
    return {
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'timesteps': timesteps,
        'improvement': improvement
    }


if __name__ == "__main__":
    base = Path("trained_data/logs")
    
    # Analyze Gate RL (SAC)
    gate_results = analyze_evaluations(
        base / "rl_gates" / "evaluations.npz",
        "GATE RL (SAC)"
    )
    
    print("\n")
    
    # Analyze Exit RL (PPO)
    exit_results = analyze_evaluations(
        base / "rl_exits" / "evaluations.npz", 
        "EXIT RL (PPO)"
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if gate_results:
        print(f"Gate RL:  Final reward = {gate_results['mean_rewards'][-1]:.2f}")
    if exit_results:
        print(f"Exit RL:  Final reward = {exit_results['mean_rewards'][-1]:.2f}")
