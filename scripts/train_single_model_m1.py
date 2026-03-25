#!/usr/bin/env python3
"""
Train a single model for a single instrument — M1 Mac tf-metal optimized.

Restores full wandb-optimized hyperparameters (seq_len 120/90, batch_size 128)
since M1 has ample unified memory vs the constrained VM.

Usage:
    python scripts/train_single_model_m1.py --instrument GBP_JPY --model transformer --candles 25000
    python scripts/train_single_model_m1.py --instrument GBP_JPY --model all --candles 25000
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── tf-metal configuration ───────────────────────────────────────────────────
# Must be set BEFORE importing TensorFlow
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# M1 Metal GPU acceleration — let TF find the Metal plugin automatically
# If you hit Metal issues, uncomment: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load .env.local
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_m1")

CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")
MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
MAX_GAP = 0.06  # 6% max acceptable direction accuracy gap

# ── Full wandb-optimized params per master pair ──────────────────────────────
# Restored from correlation_group_config.py — NO capping for M1 memory
MASTER_PARAMS = {
    "GBP_JPY": {
        "seq_len": 120, "batch_size": 128, "lr": 0.000508, "patience": 10,
        "dropout_rate": 0.236, "l2_reg": 0.000149,
        "tcn_nb_filters": 256, "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4, 8, 16, 32],
        "top_features": 150, "mixed_precision": True,
    },
    "EUR_GBP": {
        "seq_len": 90, "batch_size": 128, "lr": 0.000415, "patience": 25,
        "dropout_rate": 0.363, "l2_reg": 0.000098,
        "tcn_nb_filters": 256, "tcn_kernel_size": 5,
        "tcn_dilations": [1, 2, 4, 8, 16, 32],
        "top_features": 150, "mixed_precision": False,
    },
    "AUD_USD": {
        "seq_len": 90, "batch_size": 128, "lr": 0.000630, "patience": 20,
        "dropout_rate": 0.183, "l2_reg": 0.000405,
        "tcn_nb_filters": 64, "tcn_kernel_size": 5,
        "tcn_dilations": [1, 2, 4, 8, 16, 32],
        "top_features": None, "mixed_precision": True,
    },
    "USD_CAD": {
        "seq_len": 120, "batch_size": 128, "lr": 0.000552, "patience": 25,
        "dropout_rate": 0.468, "l2_reg": 0.000013,
        "tcn_nb_filters": 128, "tcn_kernel_size": 5,
        "tcn_dilations": [1, 2, 4, 8, 16],
        "top_features": 150, "mixed_precision": False,
    },
    "AUD_NZD": {
        "seq_len": 90, "batch_size": 128, "lr": 0.000376, "patience": 20,
        "dropout_rate": 0.420, "l2_reg": 0.000208,
        "tcn_nb_filters": 256, "tcn_kernel_size": 7,
        "tcn_dilations": [1, 2, 4, 8, 16],
        "top_features": 100, "mixed_precision": True,
    },
}

# Default for non-master / target pairs (uses EUR_GBP base)
DEFAULT_PARAMS = {
    "seq_len": 90, "batch_size": 128, "lr": 0.000415, "patience": 25,
    "dropout_rate": 0.363, "l2_reg": 0.000098,
    "tcn_nb_filters": 256, "tcn_kernel_size": 5,
    "tcn_dilations": [1, 2, 4, 8, 16, 32],
    "top_features": 150, "mixed_precision": False,
}


def check_metal_gpu():
    """Check and report tf-metal GPU availability."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"✓ Metal GPU detected: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        else:
            logger.warning("⚠ No Metal GPU found — training will use CPU")
            return False
    except Exception as e:
        logger.warning(f"⚠ GPU check failed: {e}")
        return False


def fetch_or_load(instrument: str, candles: int, granularity: str) -> pd.DataFrame:
    """Fetch from OANDA or load cached CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{instrument}_{granularity}_{candles}.csv"

    if cache_file.exists():
        logger.info(f"Loading cached data: {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["time"])
        return df

    from src.utils.oanda_practice import OandaPracticeClient
    client = OandaPracticeClient.from_env()
    all_rows = []
    remaining = candles
    to_time = None

    logger.info(f"Fetching {candles:,} {granularity} candles for {instrument}...")
    while remaining > 0:
        batch_size = min(remaining, 5000)
        kwargs = {"granularity": granularity, "count": batch_size}
        if to_time:
            kwargs["to_time"] = to_time
        try:
            response = client.get_candles(instrument, **kwargs)
        except Exception as e:
            logger.warning(f"Fetch error (retrying): {e}")
            time.sleep(2)
            try:
                response = client.get_candles(instrument, **kwargs)
            except Exception as e2:
                logger.error(f"Fetch failed after retry: {e2}")
                break

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

    logger.info(f"✓ {instrument}: {len(df):,} candles")
    return df


def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering."""
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering
    cfg = load_config(CONFIG_PATH)
    fe = FeatureEngineering(cfg)
    return fe.create_features(df.copy(), include_all=True)


def gc_cleanup():
    """Force garbage collection + clear TF session."""
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINERS — one function per model type
# ═══════════════════════════════════════════════════════════════════════════════

def train_transformer(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """Transformer Direction Trainer — primary direction prediction model."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
    from src.core.modular_data_loaders import load_direction_data

    cfg = TrainerConfig(
        epochs=200, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=30,
        label_smoothing=0.1, use_augmentation=True, use_class_weights=True,
        use_ema=True, use_ewc=False, use_feature_selection=True,
        use_replay_buffer=False, overfit_threshold=0.04, critical_threshold=0.08,
        max_acceptable_gap=MAX_GAP,
    )

    dir_data = load_direction_data(df_feat)
    if not dir_data:
        return {"error": "No direction data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TransformerDirectionTrainer(cfg)
    trainer.train(dir_data["X_train"], dir_data["y_train"],
                  dir_data["X_val"], dir_data["y_val"],
                  feature_names=dir_data.get("feature_names"))
    trainer.save(str(save_dir / "transformer_direction.keras"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    train_acc = metrics.get("train_direction_accuracy", metrics.get("train_accuracy", 0))
    val_acc = metrics.get("val_direction_accuracy", metrics.get("val_accuracy", 0))
    gc_cleanup()
    return {"train_acc": train_acc, "val_acc": val_acc, "gap": abs(train_acc - val_acc)}


def train_tcn(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """TCN Volatility Regime Trainer — 4-class volatility classification."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.tcn_trainer import TCNTrainer
    from src.core.modular_data_loaders import load_volatility_regime_data

    cfg = TrainerConfig(
        epochs=200, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=30,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )

    tcn_data = load_volatility_regime_data(df_feat, instrument=instrument)
    if not tcn_data:
        return {"error": "No volatility regime data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TCNTrainer(cfg)
    trainer.train(tcn_data["X_train"], tcn_data["y_train"],
                  tcn_data["X_val"], tcn_data["y_val"])
    trainer.save(str(save_dir / "tcn_volatility_regime.keras"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    train_acc = metrics.get("train_accuracy", 0)
    val_acc = metrics.get("val_accuracy", 0)
    gc_cleanup()
    return {"train_acc": train_acc, "val_acc": val_acc, "gap": abs(train_acc - val_acc)}


def train_lgbm_momentum(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """LightGBM Momentum Trainer — momentum signal scoring."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.lightgbm_trainers import LightGBMMomentumTrainer
    from src.core.modular_data_loaders import load_xgboost_data

    cfg = TrainerConfig(batch_size=params["batch_size"], learning_rate=params["lr"])
    data = load_xgboost_data(df_feat)
    if not data:
        return {"error": "No XGBoost data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = LightGBMMomentumTrainer(cfg)
    trainer.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    trainer.save(str(save_dir / "lgbm_momentum.pkl"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    gc_cleanup()
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_lgbm_risk(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """LightGBM Risk Trainer — drawdown risk assessment."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.lightgbm_trainers import LightGBMRiskTrainer
    from src.core.modular_data_loaders import load_rf_data

    cfg = TrainerConfig(batch_size=params["batch_size"], learning_rate=params["lr"])
    data = load_rf_data(df_feat)
    if not data:
        return {"error": "No RF data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = LightGBMRiskTrainer(cfg)
    trainer.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    trainer.save(str(save_dir / "lgbm_risk.pkl"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    gc_cleanup()
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_ridge(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """Ridge/LightGBM Confidence Trainer — confidence/stability scoring."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.ridge_trainer import RidgeTrainer
    from src.core.modular_data_loaders import load_ridge_data

    cfg = TrainerConfig(batch_size=params["batch_size"], learning_rate=params["lr"])
    data = load_ridge_data(df_feat)
    if not data:
        return {"error": "No Ridge data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = RidgeTrainer(cfg)
    trainer.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    trainer.save(str(save_dir / "ridge_confidence.pkl"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    gc_cleanup()
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_histgb(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """HistGradientBoosting Direction Trainer — fast gradient boosting for direction."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.histgb_trainer import HistGradientBoostingDirectionTrainer
    from src.core.modular_data_loaders import load_direction_data

    cfg = TrainerConfig(batch_size=params["batch_size"], learning_rate=params["lr"])
    dir_data = load_direction_data(df_feat)
    if not dir_data:
        return {"error": "No direction data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = dir_data["X_train"]
    X_val = dir_data["X_val"]
    if len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)

    trainer = HistGradientBoostingDirectionTrainer(cfg)
    trainer.train(X_train, dir_data["y_train"], X_val, dir_data["y_val"])
    trainer.save(str(save_dir / "histgb_direction.pkl"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    gc_cleanup()
    return {
        "train_acc": metrics.get("train_accuracy", 0),
        "val_acc": metrics.get("val_accuracy", 0),
        "gap": abs(metrics.get("train_accuracy", 0) - metrics.get("val_accuracy", 0)),
    }


def train_transformer_regime(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """Transformer Regime Trainer — market regime classification via transformer."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_regime_trainer import TransformerRegimeTrainer
    from src.core.modular_data_loaders import load_direction_data

    cfg = TrainerConfig(
        epochs=200, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=30,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )

    dir_data = load_direction_data(df_feat)
    if not dir_data:
        return {"error": "No direction data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TransformerRegimeTrainer(cfg)
    trainer.train(dir_data["X_train"], dir_data["y_train"],
                  dir_data["X_val"], dir_data["y_val"])
    trainer.save(str(save_dir / "transformer_regime.keras"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    gc_cleanup()
    return {"val_acc": metrics.get("val_accuracy", 0)}


def train_tcn_vol_regime(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """TCN Volatility Regime Trainer — volatility regime with 3D sequence input."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.tcn_volatility_trainer import TCNVolatilityRegimeTrainer
    from src.core.modular_data_loaders import load_volatility_regime_data

    cfg = TrainerConfig(
        epochs=200, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=30,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )

    vol_data = load_volatility_regime_data(df_feat)
    if not vol_data:
        return {"error": "No volatility regime data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TCNVolatilityRegimeTrainer(cfg)
    trainer.train(vol_data["X_train"], vol_data["y_train"],
                  vol_data["X_val"], vol_data["y_val"],
                  seq_len=params["seq_len"])
    trainer.save(str(save_dir / "tcn_volatility_regime.keras"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    gc_cleanup()
    return {"val_acc": metrics.get("val_accuracy", 0)}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_TRAINERS = {
    "transformer": train_transformer,
    "tcn": train_tcn,
    "lgbm_momentum": train_lgbm_momentum,
    "lgbm_risk": train_lgbm_risk,
    "ridge": train_ridge,
    "histgb": train_histgb,
    # transformer_regime removed — TCN is the sole regime model (65.8% acc vs broken transformer_regime)
    "tcn_vol_regime": train_tcn_vol_regime,
}

# Direction-critical models that get gap-checked
GAP_CHECKED_MODELS = {"transformer", "tcn", "histgb"}


def main():
    parser = argparse.ArgumentParser(description="Train single model (M1 Mac optimized)")
    parser.add_argument("--instrument", required=True, help="e.g. GBP_JPY")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_TRAINERS.keys()) + ["all"],
                        help="Model type or 'all' for all 8 models")
    parser.add_argument("--candles", type=int, default=25000, help="Number of H1 candles")
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--no-gpu-check", action="store_true", help="Skip Metal GPU check")
    args = parser.parse_args()

    # GPU check
    if not args.no_gpu_check:
        check_metal_gpu()

    t0 = time.time()
    instrument = args.instrument
    params = MASTER_PARAMS.get(instrument, DEFAULT_PARAMS.copy())

    # Fetch/load data
    df = fetch_or_load(instrument, args.candles, args.granularity)
    if df.empty:
        print(json.dumps({"error": "No data", "instrument": instrument, "model": args.model}))
        sys.exit(1)

    # Feature engineering
    df_feat = apply_features(df)
    del df
    gc.collect()

    models_to_train = list(MODEL_TRAINERS.keys()) if args.model == "all" else [args.model]
    results = []

    for model_name in models_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} for {instrument} ({len(df_feat)} rows, {len(df_feat.columns)} features)")
        logger.info(f"{'='*60}")

        trainer_fn = MODEL_TRAINERS[model_name]
        try:
            result = trainer_fn(instrument, df_feat, params)
            result["instrument"] = instrument
            result["model"] = model_name
            result["duration_s"] = round(time.time() - t0, 1)

            gap = result.get("gap", 0)
            if model_name in GAP_CHECKED_MODELS:
                result["passed"] = gap <= MAX_GAP
                status = "✓ PASS" if result["passed"] else "✗ FAIL"
                logger.info(f"{status}: {model_name} gap={gap:.4f} (max={MAX_GAP})")
            else:
                result["passed"] = True
                logger.info(f"✓ DONE: {model_name} (no gap check)")

            results.append(result)
            print(f"RESULT:{json.dumps(result)}")
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            logger.error(traceback.format_exc())
            error_result = {
                "error": str(e), "instrument": instrument, "model": model_name,
                "duration_s": round(time.time() - t0, 1), "passed": False,
            }
            results.append(error_result)
            print(f"RESULT:{json.dumps(error_result)}")

        gc_cleanup()

    # Summary
    total_time = round(time.time() - t0, 1)
    passed = sum(1 for r in results if r.get("passed"))
    failed = sum(1 for r in results if not r.get("passed"))
    errors = sum(1 for r in results if "error" in r)

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE: {instrument}")
    logger.info(f"  Models: {len(results)} total, {passed} passed, {failed} failed, {errors} errors")
    logger.info(f"  Duration: {total_time}s")
    logger.info(f"{'='*60}")

    # Write summary to results file
    summary_path = MODELS_DIR / instrument / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "instrument": instrument,
            "candles": args.candles,
            "params": params,
            "results": results,
            "total_duration_s": total_time,
            "passed": passed,
            "failed": failed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, sort_keys=True, default=str)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
