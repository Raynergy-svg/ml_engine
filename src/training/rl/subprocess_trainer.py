#!/usr/bin/env python3
"""Subprocess-based RL training to isolate PyTorch from TensorFlow.

This script runs in a separate process to avoid TensorFlow/PyTorch conflicts.
"""

from __future__ import annotations

import os
import sys
import pickle
import tempfile
from pathlib import Path

# Force CPU-only BEFORE any imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_DISABLE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Prevent TensorFlow from being imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train_in_subprocess(data_path: str, output_path: str, config_dict: dict) -> dict:
    """Train PPO in isolated subprocess."""
    import numpy as np
    import torch

    # Force CPU
    torch.set_default_device("cpu")

    print(f"  [subprocess] PyTorch {torch.__version__} on CPU", flush=True)
    print(f"  [subprocess] Loading training data...", flush=True)

    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    ensemble_predictions = data["ensemble_predictions"]
    prices = data["prices"]

    print(f"  [subprocess] Data: {len(features)} samples, {features.shape[1]} features", flush=True)

    # Import SB3 (only in this subprocess)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from sklearn.preprocessing import StandardScaler
    import gymnasium as gym
    from gymnasium import spaces

    # Scale features
    print(f"  [subprocess] Scaling features...", flush=True)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Define environment inline (avoid import issues)
    POSITION_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10]

    class TradingEnv(gym.Env):
        metadata = {"render_modes": ["human"]}

        def __init__(self, features, ensemble_predictions, prices, config):
            super().__init__()
            self.config = config
            self.features = features
            self.ensemble_predictions = ensemble_predictions
            self.prices = prices
            self.n_samples = len(prices)
            self.current_step = config.get("sequence_length", 60)
            self.initial_balance = 10000.0
            self.balance = self.initial_balance
            self.equity = self.initial_balance
            self.position = 0.0
            self.position_price = 0.0
            self.trade_history = []
            self.daily_pnl = 0.0
            self.max_equity = self.initial_balance

            n_features = features.shape[1] if len(features.shape) > 1 else 1
            obs_dim = n_features + 2 + 4
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(len(POSITION_LEVELS))
            self.position_levels = POSITION_LEVELS

        def _get_observation(self):
            market_features = self.features[self.current_step] if len(self.features.shape) > 1 else np.array([self.features[self.current_step]])
            pred = self.ensemble_predictions[self.current_step]
            direction_prob = pred[0] if len(pred) > 0 else 0.5
            confidence = pred[1] if len(pred) > 1 else 0.5
            equity_pct = self.equity / self.initial_balance - 1.0
            drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
            daily_pnl_pct = self.daily_pnl / self.initial_balance
            position_pct = abs(self.position) / self.equity if self.equity > 0 else 0.0

            obs = np.concatenate([
                market_features.astype(np.float32),
                np.array([direction_prob, confidence], dtype=np.float32),
                np.array([equity_pct, drawdown, daily_pnl_pct, position_pct], dtype=np.float32),
            ])
            return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = self.config.get("sequence_length", 60)
            self.balance = self.initial_balance
            self.equity = self.initial_balance
            self.position = 0.0
            self.position_price = 0.0
            self.daily_pnl = 0.0
            self.max_equity = self.initial_balance
            return self._get_observation(), {}

        def step(self, action):
            position_pct = POSITION_LEVELS[action]
            current_price = self.prices[self.current_step]
            pred = self.ensemble_predictions[self.current_step]
            direction = 1 if pred[0] > 0.5 else -1
            new_position = position_pct * self.equity * direction

            if self.position != 0:
                price_change = (current_price - self.position_price) / self.position_price
                pnl = self.position * price_change
                self.balance += pnl
                self.daily_pnl += pnl
                if pnl != 0:
                    self.trade_history.append({"pnl": pnl, "position": self.position})

            self.position = new_position
            self.position_price = current_price if new_position != 0 else 0.0

            unrealized = 0.0
            if self.position != 0 and self.position_price > 0:
                price_change = (current_price - self.position_price) / self.position_price
                unrealized = self.position * price_change
            self.equity = self.balance + unrealized
            self.max_equity = max(self.max_equity, self.equity)

            # Reward
            returns = (self.equity - self.initial_balance) / self.initial_balance
            drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
            reward = returns * 10.0 - drawdown * self.config.get("drawdown_penalty", 5.0)

            self.current_step += 1
            done = self.current_step >= self.n_samples - 1
            truncated = False

            max_dd = self.config.get("max_drawdown_pct", 0.20)
            if drawdown > max_dd:
                done = True
                reward -= 10.0

            return self._get_observation(), reward, done, truncated, {}

    # Create environment
    print(f"  [subprocess] Creating environment...", flush=True)
    env = TradingEnv(features_scaled, ensemble_predictions, prices, config_dict)
    vec_env = DummyVecEnv([lambda: env])

    # Create PPO
    print(f"  [subprocess] Creating PPO model...", flush=True)
    sys.stdout.flush()

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config_dict.get("learning_rate", 3e-4),
        n_steps=config_dict.get("n_steps", 256),
        batch_size=config_dict.get("batch_size", 32),
        n_epochs=config_dict.get("n_epochs", 5),
        gamma=config_dict.get("gamma", 0.99),
        gae_lambda=config_dict.get("gae_lambda", 0.95),
        verbose=1,
        device="cpu",
    )

    print(f"  [subprocess] ✓ PPO model created", flush=True)

    # Training
    total_timesteps = config_dict.get("total_timesteps", 50000)
    print(f"  [subprocess] Training for {total_timesteps:,} timesteps...", flush=True)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    print(f"  [subprocess] ✓ Training complete", flush=True)

    # Save model and scaler
    model_path = Path(output_path) / "rl_position_sizer.zip"
    scaler_path = Path(output_path) / "rl_scaler.pkl"

    model.save(str(model_path))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  [subprocess] ✓ Model saved to {model_path}", flush=True)

    # Calculate win_rate from trade_history
    winning_trades = sum(1 for t in env.trade_history if t.get("pnl", 0) > 0)
    total_trades = len(env.trade_history)
    win_rate = winning_trades / max(total_trades, 1)

    # Return stats
    stats = {
        "total_timesteps": total_timesteps,
        "final_equity": env.equity,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }

    return stats


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pickled training data")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--config", required=True, help="JSON config string")
    args = parser.parse_args()

    config_dict = json.loads(args.config)

    try:
        stats = train_in_subprocess(args.data, args.output, config_dict)
        # Write stats to file
        stats_path = Path(args.output) / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
        print(f"  [subprocess] Stats written to {stats_path}")
        sys.exit(0)
    except Exception as e:
        print(f"  [subprocess] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
