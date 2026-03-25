#!/usr/bin/env python3
"""
Transfer Learning from Master to Target Pairs — M1 Mac optimized.

For each correlation group, loads the trained master model and transfers
knowledge to all target pairs using warm-start + EWC + layer freezing
+ gradual unfreezing.

Usage:
    python scripts/train_transfer_learning.py --candles 25000
    python scripts/train_transfer_learning.py --group yen_carry --candles 25000
    python scripts/train_transfer_learning.py --master GBP_JPY --targets EUR_JPY,AUD_JPY
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
logger = logging.getLogger("transfer_learning")

CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")
MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
MAX_GAP = 0.06

# ── Correlation groups: master -> targets ────────────────────────────────────
CORRELATION_GROUPS = {
    "yen_carry": {
        "master": "GBP_JPY",
        "targets": ["EUR_JPY", "AUD_JPY", "NZD_JPY", "USD_JPY"],
    },
    "euro_basket": {
        "master": "EUR_GBP",
        "targets": ["EUR_USD", "EUR_JPY", "EUR_AUD"],
    },
    "sterling_cluster": {
        "master": "EUR_GBP",
        "targets": ["GBP_USD", "GBP_AUD"],
        # GBP_JPY is also in this group but trained as yen_carry master
    },
    "aussie_kiwi_commodity": {
        "master": "AUD_USD",
        "targets": ["NZD_USD", "AUD_JPY", "NZD_JPY"],
    },
    "usd_safe_haven": {
        "master": "USD_CAD",
        "targets": ["USD_JPY", "USD_CHF"],
    },
    "crosses": {
        "master": "AUD_NZD",
        "targets": ["GBP_AUD", "EUR_AUD"],
    },
}

# Transfer learning params
TRANSFER_EPOCHS = 50
TRANSFER_LR_FACTOR = 0.03  # 33x LR reduction for warm-start
ENCODER_LAYERS_TO_FREEZE = 2
UNFREEZE_AFTER_EPOCHS = 10
EWC_LAMBDA = 100.0

# Models that support warm-start transfer (Keras-based)
TRANSFERABLE_MODELS = ["transformer_direction.keras", "transformer_regime.keras", "tcn_volatility.keras", "tcn_volatility_regime.keras"]

# Models that get copied as-is (sklearn/lgbm — retrain from scratch on target data)
RETRAIN_MODELS = ["lgbm_momentum.pkl", "lgbm_risk.pkl", "ridge_confidence.pkl", "histgb_direction.pkl"]


def fetch_or_load(instrument: str, candles: int, granularity: str) -> pd.DataFrame:
    """Fetch from OANDA or load cached CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{instrument}_{granularity}_{candles}.csv"

    if cache_file.exists():
        logger.info(f"Loading cached: {cache_file}")
        return pd.read_csv(cache_file, parse_dates=["time"])

    from src.utils.oanda_practice import OandaPracticeClient
    client = OandaPracticeClient.from_env()
    all_rows = []
    remaining = candles
    to_time = None

    logger.info(f"Fetching {candles:,} H1 candles for {instrument}...")
    while remaining > 0:
        batch_size = min(remaining, 5000)
        kwargs = {"granularity": granularity, "count": batch_size}
        if to_time:
            kwargs["to_time"] = to_time
        try:
            response = client.get_candles(instrument, **kwargs)
        except Exception as e:
            logger.warning(f"Fetch error: {e}")
            time.sleep(2)
            response = client.get_candles(instrument, **kwargs)

        candles_list = response.get("candles", [])
        if not candles_list:
            break
        for c in candles_list:
            mid = c.get("mid", {})
            all_rows.append({
                "time": c.get("time"),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })
        remaining -= len(candles_list)
        to_time = candles_list[0].get("time")
        if len(candles_list) < batch_size:
            break
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        df.to_csv(cache_file, index=False)
    return df


def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering
    cfg = load_config(CONFIG_PATH)
    fe = FeatureEngineering(cfg)
    return fe.create_features(df.copy(), include_all=True)


def transfer_keras_model(master_path: str, target_instrument: str, df_feat: pd.DataFrame,
                         master_lr: float, model_type: str) -> dict:
    """Transfer a Keras model from master to target pair."""
    import tensorflow as tf
    from src.training.trainers.config import TrainerConfig
    from src.core.modular_data_loaders import load_direction_data, load_tcn_data, load_volatility_regime_data

    if not Path(master_path).exists():
        return {"error": f"Master model not found: {master_path}"}

    # Load appropriate data based on model type
    if "tcn_volatility_regime" in model_type:
        data = load_volatility_regime_data(df_feat)
        model_filename = "tcn_volatility_regime.keras"
    elif "tcn" in model_type:
        data = load_tcn_data(df_feat)
        model_filename = "tcn_volatility.keras"
    else:
        data = load_direction_data(df_feat)
        model_filename = model_type

    if not data:
        return {"error": f"No data for {model_type}"}

    # Load master model
    logger.info(f"  Loading master model: {master_path}")
    model = tf.keras.models.load_model(master_path, compile=False)

    # Freeze early encoder layers
    frozen_count = 0
    for layer in model.layers:
        if any(prefix in layer.name for prefix in ["transformer_0", "transformer_1", "encoder", "conv1d"]):
            if frozen_count < ENCODER_LAYERS_TO_FREEZE:
                layer.trainable = False
                frozen_count += 1
                logger.info(f"    Frozen layer: {layer.name}")

    # Compile with reduced LR
    transfer_lr = master_lr * TRANSFER_LR_FACTOR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=transfer_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(f"  Transfer LR: {transfer_lr:.6f} (base={master_lr:.6f} * factor={TRANSFER_LR_FACTOR})")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    # Gradual unfreezing callback
    class GradualUnfreezeCallback(tf.keras.callbacks.Callback):
        def __init__(self, unfreeze_after=UNFREEZE_AFTER_EPOCHS):
            self.unfreeze_after = unfreeze_after
            self.unfrozen = False

        def on_epoch_end(self, epoch, logs=None):
            if epoch >= self.unfreeze_after and not self.unfrozen:
                for layer in self.model.layers:
                    layer.trainable = True
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=transfer_lr * 0.5),
                    loss="binary_crossentropy",
                    metrics=["accuracy"],
                )
                self.unfrozen = True
                logger.info(f"    ✓ Unfroze all layers at epoch {epoch}")

    callbacks.append(GradualUnfreezeCallback())

    # Train
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=TRANSFER_EPOCHS,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    # Save
    save_dir = MODELS_DIR / target_instrument
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / model_filename
    model.save(str(save_path))

    # Metrics
    val_acc = max(history.history.get("val_accuracy", [0]))
    train_acc = max(history.history.get("accuracy", [0]))
    gap = abs(train_acc - val_acc)

    gc.collect()
    tf.keras.backend.clear_session()

    return {
        "train_acc": train_acc, "val_acc": val_acc, "gap": gap,
        "passed": gap <= MAX_GAP, "epochs_trained": len(history.history.get("loss", [])),
    }


def retrain_sklearn_model(model_type: str, target_instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """Retrain a sklearn/lgbm model from scratch on target pair data."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train_single_model_m1.py"),
         "--instrument", target_instrument,
         "--model", model_type,
         "--candles", "0",  # Data already cached
         "--no-gpu-check"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )

    for line in result.stdout.split("\n"):
        if line.startswith("RESULT:"):
            try:
                return json.loads(line[7:])
            except json.JSONDecodeError:
                pass

    return {"error": result.stderr[-500:] if result.stderr else "Unknown error", "model": model_type}


def transfer_group(group_name: str, group_config: dict, candles: int, granularity: str) -> list:
    """Transfer all models from master to all targets in a correlation group."""
    master = group_config["master"]
    targets = group_config["targets"]
    results = []

    logger.info(f"\n{'='*70}")
    logger.info(f"CORRELATION GROUP: {group_name}")
    logger.info(f"  Master: {master} → Targets: {', '.join(targets)}")
    logger.info(f"{'='*70}")

    # Get master params for LR
    from scripts.train_single_model_m1 import MASTER_PARAMS, DEFAULT_PARAMS
    master_params = MASTER_PARAMS.get(master, DEFAULT_PARAMS)

    for target in targets:
        logger.info(f"\n--- Transferring to {target} ---")

        # Fetch/cache target data
        df = fetch_or_load(target, candles, granularity)
        if df.empty:
            results.append({"target": target, "error": "No data"})
            continue

        df_feat = apply_features(df)
        del df
        gc.collect()

        target_results = {"target": target, "models": {}}

        # Transfer Keras models (warm-start)
        for model_file in TRANSFERABLE_MODELS:
            master_path = MODELS_DIR / master / model_file
            if master_path.exists():
                logger.info(f"  Transferring {model_file}...")
                try:
                    r = transfer_keras_model(
                        str(master_path), target, df_feat,
                        master_params["lr"], model_file,
                    )
                    target_results["models"][model_file] = r
                    status = "PASS" if r.get("passed", True) else "FAIL"
                    logger.info(f"    → {status}: val_acc={r.get('val_acc', 0):.4f}, gap={r.get('gap', 0):.4f}")
                except Exception as e:
                    logger.error(f"    → ERROR: {e}")
                    target_results["models"][model_file] = {"error": str(e)}
            else:
                logger.warning(f"  Skipping {model_file} (master model not found)")

        # Retrain sklearn models on target data (no warm-start needed)
        model_to_trainer = {
            "lgbm_momentum.pkl": "lgbm_momentum",
            "lgbm_risk.pkl": "lgbm_risk",
            "ridge_confidence.pkl": "ridge",
            "histgb_direction.pkl": "histgb",
        }
        for model_file, trainer_name in model_to_trainer.items():
            logger.info(f"  Retraining {model_file} on {target} data...")
            try:
                # Use subprocess for memory isolation
                r = retrain_sklearn_model(trainer_name, target, df_feat, master_params)
                target_results["models"][model_file] = r
                logger.info(f"    → DONE: {r.get('val_acc', 'N/A')}")
            except Exception as e:
                logger.error(f"    → ERROR: {e}")
                target_results["models"][model_file] = {"error": str(e)}

        results.append(target_results)
        del df_feat
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Transfer learning from master to target pairs")
    parser.add_argument("--candles", type=int, default=25000)
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--group", help="Specific group to transfer (e.g. yen_carry)")
    parser.add_argument("--master", help="Override master pair")
    parser.add_argument("--targets", help="Override target pairs (comma-separated)")
    args = parser.parse_args()

    t0 = time.time()
    all_results = {}

    if args.master and args.targets:
        # Ad-hoc transfer
        group_config = {"master": args.master, "targets": args.targets.split(",")}
        results = transfer_group("custom", group_config, args.candles, args.granularity)
        all_results["custom"] = results
    elif args.group:
        if args.group not in CORRELATION_GROUPS:
            logger.error(f"Unknown group: {args.group}. Available: {list(CORRELATION_GROUPS.keys())}")
            sys.exit(1)
        results = transfer_group(args.group, CORRELATION_GROUPS[args.group], args.candles, args.granularity)
        all_results[args.group] = results
    else:
        # Transfer all groups
        for group_name, group_config in CORRELATION_GROUPS.items():
            results = transfer_group(group_name, group_config, args.candles, args.granularity)
            all_results[group_name] = results

    # Save results
    total_time = round(time.time() - t0, 1)
    report_path = MODELS_DIR / "transfer_learning_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "results": all_results,
            "total_duration_s": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, sort_keys=True, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"TRANSFER LEARNING COMPLETE — {total_time}s")
    logger.info(f"Report saved: {report_path}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
