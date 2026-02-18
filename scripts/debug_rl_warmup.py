#!/usr/bin/env python3
"""
RL Warm-Up Validation Script.

This script runs a minimal RL training to diagnose where hangs may occur
during PPO training. It tests:
1. Environment initialization
2. PPO model creation
3. First rollout collection
4. Short training run

Usage:
    python scripts/debug_rl_warmup.py

Expected output: Timing for each phase to identify bottlenecks.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def _print_phase(phase: str, elapsed: float | None = None) -> None:
    """Print a phase marker with optional elapsed time."""
    if elapsed is not None:
        print(f"[{time.strftime('%H:%M:%S')}] ✓ {phase} ({elapsed:.2f}s)")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ▶ {phase}...")
    sys.stdout.flush()


def main() -> int:
    """Run minimal RL warm-up test."""
    print("=" * 60)
    print("RL Warm-Up Validation Script")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print()

    total_start = time.perf_counter()

    # Phase 1: Import dependencies
    phase_start = time.perf_counter()
    _print_phase("Importing dependencies")
    
    try:
        from sklearn.preprocessing import StandardScaler
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        _print_phase("Dependencies imported", time.perf_counter() - phase_start)
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("   Install with: pip install stable-baselines3 scikit-learn")
        return 1

    # Phase 2: Create synthetic data
    phase_start = time.perf_counter()
    _print_phase("Creating synthetic data (1000 samples, 20 features)")
    
    n_samples = 1000
    n_features = 20
    np.random.seed(42)
    
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    ensemble_predictions = np.random.randn(n_samples, 2).astype(np.float32)
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    
    _print_phase("Synthetic data created", time.perf_counter() - phase_start)

    # Phase 3: Fit StandardScaler
    phase_start = time.perf_counter()
    _print_phase("Fitting StandardScaler")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    _print_phase("StandardScaler fitted", time.perf_counter() - phase_start)

    # Phase 4: Import and create TradingEnv
    phase_start = time.perf_counter()
    _print_phase("Creating TradingEnv")
    
    try:
        from src.training.rl.position_sizer import RLConfig
        from src.training.rl.environments.trading_env import TradingEnv
        
        config = RLConfig(
            total_timesteps=1000,
            n_steps=64,  # Very small for quick testing
            batch_size=16,
            n_epochs=2,
        )
        
        env = TradingEnv(
            features=features_scaled,
            ensemble_predictions=ensemble_predictions,
            prices=prices,
            config=config,
        )
        _print_phase("TradingEnv created", time.perf_counter() - phase_start)
    except ImportError as e:
        print(f"\n❌ TradingEnv import failed: {e}")
        return 1

    # Phase 5: Wrap in DummyVecEnv
    phase_start = time.perf_counter()
    _print_phase("Wrapping in DummyVecEnv")
    
    vec_env = DummyVecEnv([lambda: env])
    
    _print_phase("DummyVecEnv created", time.perf_counter() - phase_start)

    # Phase 6: Create PPO model
    phase_start = time.perf_counter()
    _print_phase("Creating PPO model (MlpPolicy, device=cpu)")
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=16,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,  # Enable internal logging
        device="cpu",
    )
    
    _print_phase("PPO model created", time.perf_counter() - phase_start)

    # Phase 7: First rollout (this is where hangs typically occur)
    phase_start = time.perf_counter()
    _print_phase("🚨 FIRST ROLLOUT - This may take 5-30s on CPU")
    print("   (Collecting 64 steps for first batch)")
    sys.stdout.flush()
    
    # The first learn() call triggers rollout collection
    # We use a very small timestep count to test quickly
    try:
        model.learn(total_timesteps=64, progress_bar=False)
        _print_phase("First rollout completed", time.perf_counter() - phase_start)
    except Exception as e:
        print(f"\n❌ First rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Phase 8: Short training run
    phase_start = time.perf_counter()
    _print_phase("Running short training (1000 timesteps)")
    
    try:
        model.learn(total_timesteps=1000, progress_bar=False)
        _print_phase("Short training completed", time.perf_counter() - phase_start)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Phase 9: Test inference
    phase_start = time.perf_counter()
    _print_phase("Testing inference")
    
    try:
        # Create a sample observation
        obs = np.concatenate([
            features_scaled[0],
            ensemble_predictions[0],
            [0.0, 0.0, 0.0, 0.5],  # Account placeholder
        ]).astype(np.float32)
        
        action, _ = model.predict(obs, deterministic=True)
        _print_phase(f"Inference successful (action={action})", time.perf_counter() - phase_start)
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        return 1

    # Summary
    total_elapsed = time.perf_counter() - total_start
    print()
    print("=" * 60)
    print(f"✅ ALL PHASES COMPLETED in {total_elapsed:.2f}s")
    print("=" * 60)
    print()
    print("If all phases completed successfully, RL training should work.")
    print("If any phase took >60s, that's where optimization is needed.")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
