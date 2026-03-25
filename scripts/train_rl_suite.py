#!/usr/bin/env python3
"""
RL Suite Training — M1 Mac optimized.

Trains the three RL components:
  1. Position Sizer (PPO) — optimal position sizing per trade
  2. Gate Thresholds (SAC) — adaptive confidence/momentum/risk gates
  3. Optimal Exits (PPO) — when to take profit or stop loss
  4. Reward Model (Ridge) — learned reward shaping for RL envs

Each component runs in its own subprocess to isolate PyTorch (stable-baselines3)
from TensorFlow (Keras models) — critical on macOS Metal.

Usage:
    python scripts/train_rl_suite.py
    python scripts/train_rl_suite.py --component position_sizer
    python scripts/train_rl_suite.py --timesteps 100000
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

env_path = Path(__file__).resolve().parent.parent / ".env.local"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rl_suite")

CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")
MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"

# Master pairs for building ensemble predictions
MASTER_PAIRS = ["GBP_JPY", "EUR_GBP", "AUD_USD", "USD_CAD", "AUD_NZD"]


def load_ensemble_data_for_rl(pairs: list, candles: int = 25000) -> dict:
    """
    Load cached data and run ensemble predictions to create RL training data.

    Returns dict with:
      - features: np.ndarray of shape (N, feature_dim)
      - ensemble_predictions: np.ndarray of shape (N, 2) — [direction_prob, confidence]
      - prices: np.ndarray of shape (N,) — close prices
    """
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering

    cfg = load_config(CONFIG_PATH)
    fe = FeatureEngineering(cfg)

    all_features = []
    all_prices = []
    all_predictions = []

    for pair in pairs:
        cache_file = CACHE_DIR / f"{pair}_H1_{candles}.csv"
        if not cache_file.exists():
            logger.warning(f"No cached data for {pair}, skipping")
            continue

        df = pd.read_csv(cache_file, parse_dates=["time"])
        df_feat = fe.create_features(df.copy(), include_all=True)

        # Extract numeric features
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        feat_array = df_feat[numeric_cols].fillna(0).values.astype(np.float32)

        # Use close prices for reward calculation
        prices = df_feat["close"].values.astype(np.float32)

        # Generate pseudo-ensemble predictions from trained models
        # (direction probability, confidence score)
        model_dir = MODELS_DIR / pair
        preds = generate_ensemble_predictions(model_dir, df_feat, feat_array)

        all_features.append(feat_array)
        all_prices.append(prices)
        all_predictions.append(preds)

        del df, df_feat
        gc.collect()

    if not all_features:
        return {}

    return {
        "features": np.concatenate(all_features, axis=0),
        "prices": np.concatenate(all_prices, axis=0),
        "ensemble_predictions": np.concatenate(all_predictions, axis=0),
    }


def generate_ensemble_predictions(model_dir: Path, df_feat: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    """Generate ensemble predictions from trained models for RL training data."""
    n = len(features)
    predictions = np.full((n, 2), 0.5, dtype=np.float32)  # [direction_prob, confidence]

    # Try loading transformer direction model
    transformer_path = model_dir / "transformer_direction.keras"
    if transformer_path.exists():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(transformer_path), compile=False)
            # Get input shape
            expected_shape = model.input_shape
            if len(expected_shape) == 3:
                # Needs 3D input — create sequences
                seq_len = expected_shape[1]
                feat_dim = expected_shape[2]
                if features.shape[1] >= feat_dim:
                    X = features[:, :feat_dim]
                else:
                    X = np.pad(features, ((0, 0), (0, feat_dim - features.shape[1])))
                # Create sequences
                sequences = []
                for i in range(seq_len, len(X)):
                    sequences.append(X[i-seq_len:i])
                if sequences:
                    X_seq = np.array(sequences)
                    preds = model.predict(X_seq, verbose=0, batch_size=256).flatten()
                    predictions[seq_len:seq_len+len(preds), 0] = preds
            tf.keras.backend.clear_session()
        except Exception as e:
            logger.warning(f"Could not load transformer for {model_dir.name}: {e}")

    # Try loading ridge confidence
    ridge_path = model_dir / "ridge_confidence.pkl"
    if ridge_path.exists():
        try:
            import pickle
            with open(ridge_path, "rb") as f:
                ridge_model = pickle.load(f)
            if hasattr(ridge_model, "predict_proba"):
                conf = ridge_model.predict_proba(features)
                if conf.ndim == 2:
                    predictions[:, 1] = conf[:, 1]
                else:
                    predictions[:, 1] = conf
            elif hasattr(ridge_model, "predict"):
                predictions[:, 1] = ridge_model.predict(features)
        except Exception as e:
            logger.warning(f"Could not load ridge for {model_dir.name}: {e}")

    return predictions


def train_position_sizer(data: dict, timesteps: int = 50000) -> dict:
    """Train PPO position sizer."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING: RL Position Sizer (PPO)")
    logger.info("="*60)

    from src.training.rl.position_sizer import RLPositionSizer, RLConfig

    config = RLConfig(
        total_timesteps=timesteps,
        learning_rate=3e-4,
        n_steps=512,    # Larger on M1
        batch_size=64,   # Larger on M1
        n_epochs=10,     # More passes on M1
        gamma=0.99,
        gae_lambda=0.95,
        sequence_length=60,
        max_position_pct=0.10,
        min_position_pct=0.01,
        sharpe_weight=0.1,
        drawdown_penalty=0.5,
        win_rate_bonus=0.2,
    )

    sizer = RLPositionSizer(config)

    try:
        result = sizer.train(
            features=data["features"],
            ensemble_predictions=data["ensemble_predictions"],
            prices=data["prices"],
            use_subprocess=True,  # Critical on macOS to isolate PyTorch
        )

        # Save
        sizer.save()
        logger.info(f"✓ Position Sizer trained: {result}")
        return {"status": "success", "metrics": result}
    except Exception as e:
        logger.error(f"✗ Position Sizer failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}


def train_gate_thresholds(data: dict, timesteps: int = 100000) -> dict:
    """Train SAC gate threshold optimizer."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING: Gate Thresholds (SAC)")
    logger.info("="*60)

    try:
        from src.rl.gate_threshold_env import GateThresholdRL

        optimizer = GateThresholdRL()

        # The gate RL needs trade history — use ensemble predictions as proxy
        result = optimizer.train(
            features=data["features"],
            predictions=data["ensemble_predictions"],
            prices=data["prices"],
            total_timesteps=timesteps,
        )

        logger.info(f"✓ Gate Thresholds trained: {result}")
        return {"status": "success", "metrics": result}
    except ImportError as e:
        logger.warning(f"Gate threshold RL not available: {e}")
        return {"status": "skipped", "reason": str(e)}
    except Exception as e:
        logger.error(f"✗ Gate Thresholds failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}


def train_optimal_exits(data: dict, timesteps: int = 100000) -> dict:
    """Train PPO optimal exit timing."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING: Optimal Exit Timing (PPO)")
    logger.info("="*60)

    try:
        from src.rl.optimal_exit_env import OptimalExitRL

        exit_rl = OptimalExitRL()

        result = exit_rl.train(
            features=data["features"],
            predictions=data["ensemble_predictions"],
            prices=data["prices"],
            total_timesteps=timesteps,
        )

        logger.info(f"✓ Optimal Exits trained: {result}")
        return {"status": "success", "metrics": result}
    except ImportError as e:
        logger.warning(f"Optimal exit RL not available: {e}")
        return {"status": "skipped", "reason": str(e)}
    except Exception as e:
        logger.error(f"✗ Optimal Exits failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}


def train_reward_model(data: dict) -> dict:
    """Train learned reward model (Ridge ensemble)."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING: Reward Model (Ridge Ensemble)")
    logger.info("="*60)

    try:
        from src.training.rl.reward_model import EnsembleRewardModel

        reward_model = EnsembleRewardModel()

        # Compute actual returns from prices
        prices = data["prices"]
        returns = np.diff(prices) / prices[:-1]
        returns = np.append(returns, 0)  # pad to match length

        result = reward_model.train(
            ensemble_predictions=data["ensemble_predictions"],
            actual_returns=returns,
        )

        # Save
        import pickle
        save_path = MODELS_DIR / "reward_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(reward_model, f)

        logger.info(f"✓ Reward Model trained: {result}")
        return {"status": "success", "metrics": result}
    except ImportError as e:
        logger.warning(f"Reward model not available: {e}")
        return {"status": "skipped", "reason": str(e)}
    except Exception as e:
        logger.error(f"✗ Reward Model failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}


COMPONENT_TRAINERS = {
    "position_sizer": train_position_sizer,
    "gate_thresholds": train_gate_thresholds,
    "optimal_exits": train_optimal_exits,
    "reward_model": train_reward_model,
}


def main():
    parser = argparse.ArgumentParser(description="Train RL suite (M1 Mac optimized)")
    parser.add_argument("--component", choices=list(COMPONENT_TRAINERS.keys()),
                        help="Train specific component (default: all)")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="PPO/SAC timesteps (default: 50000)")
    parser.add_argument("--candles", type=int, default=25000)
    args = parser.parse_args()

    t0 = time.time()

    # Load ensemble data from cached master pair training data
    logger.info("Loading ensemble data for RL training...")
    data = load_ensemble_data_for_rl(MASTER_PAIRS, args.candles)
    if not data:
        logger.error("No training data available. Run core model training first.")
        sys.exit(1)

    logger.info(f"Loaded {len(data['features']):,} samples across {len(MASTER_PAIRS)} pairs")

    # Train components
    results = {}
    components = [args.component] if args.component else list(COMPONENT_TRAINERS.keys())

    for component in components:
        trainer_fn = COMPONENT_TRAINERS[component]
        if component in ("position_sizer", "gate_thresholds", "optimal_exits"):
            results[component] = trainer_fn(data, timesteps=args.timesteps)
        else:
            results[component] = trainer_fn(data)
        gc.collect()

    # Save report
    total_time = round(time.time() - t0, 1)
    report_path = MODELS_DIR / "rl_suite_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "results": results,
            "total_duration_s": total_time,
            "timesteps": args.timesteps,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, sort_keys=True, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"RL SUITE COMPLETE — {total_time}s")
    for comp, r in results.items():
        logger.info(f"  {comp}: {r.get('status', 'unknown')}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
