#!/usr/bin/env python3
"""
Train a single model for a single instrument.
Designed to be called as a subprocess for memory isolation.

Usage:
    python scripts/train_single_model.py --instrument GBP_JPY --model transformer --candles 25000
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
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_single")

CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")
MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
MAX_GAP = 0.06

MASTER_PARAMS = {
    "GBP_JPY": {"seq_len": 60, "batch_size": 64, "lr": 0.000508, "patience": 10},
    "EUR_GBP": {"seq_len": 60, "batch_size": 64, "lr": 0.000415, "patience": 25},
    "AUD_USD": {"seq_len": 60, "batch_size": 64, "lr": 0.000630, "patience": 20},
    "USD_CAD": {"seq_len": 60, "batch_size": 64, "lr": 0.000552, "patience": 25},
    "AUD_NZD": {"seq_len": 60, "batch_size": 64, "lr": 0.000376, "patience": 20},
}


def fetch_or_load(instrument: str, candles: int, granularity: str) -> pd.DataFrame:
    """Fetch from OANDA or load cached."""
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

    logger.info(f"✓ {instrument}: {len(df):,} candles")
    return df


def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering
    cfg = load_config(CONFIG_PATH)
    fe = FeatureEngineering(cfg)
    return fe.create_features(df.copy(), include_all=True)


def train_transformer(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
    from src.core.modular_data_loaders import load_direction_data

    cfg = TrainerConfig(
        epochs=100, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=20,
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
    trainer.train(dir_data["X_train"], dir_data["y_train"], dir_data["X_val"], dir_data["y_val"],
                  feature_names=dir_data.get("feature_names"))
    trainer.save(str(save_dir / "transformer_direction.keras"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    train_acc = metrics.get("train_direction_accuracy", metrics.get("train_accuracy", 0))
    val_acc = metrics.get("val_direction_accuracy", metrics.get("val_accuracy", 0))
    return {"train_acc": train_acc, "val_acc": val_acc, "gap": abs(train_acc - val_acc)}


def train_tcn(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.tcn_trainer import TCNTrainer
    from src.core.modular_data_loaders import load_tcn_data

    cfg = TrainerConfig(
        epochs=100, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=20,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )

    tcn_data = load_tcn_data(df_feat)
    if not tcn_data:
        return {"error": "No TCN data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TCNTrainer(cfg)
    trainer.train(tcn_data["X_train"], tcn_data["y_train"], tcn_data["X_val"], tcn_data["y_val"])
    trainer.save(str(save_dir / "tcn_volatility.keras"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    train_acc = metrics.get("train_accuracy", 0)
    val_acc = metrics.get("val_accuracy", 0)
    return {"train_acc": train_acc, "val_acc": val_acc, "gap": abs(train_acc - val_acc)}


def train_lgbm_momentum(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
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
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_lgbm_risk(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
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
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_ridge(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
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
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_histgb(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
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
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0),
            "gap": abs(metrics.get("train_accuracy", 0) - metrics.get("val_accuracy", 0))}


def train_transformer_regime(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_regime_trainer import TransformerRegimeTrainer
    from src.core.modular_data_loaders import load_direction_data

    cfg = TrainerConfig(
        epochs=100, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=20,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )

    dir_data = load_direction_data(df_feat)
    if not dir_data:
        return {"error": "No direction data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TransformerRegimeTrainer(cfg)
    trainer.train(dir_data["X_train"], dir_data["y_train"], dir_data["X_val"], dir_data["y_val"])
    trainer.save(str(save_dir / "transformer_regime.keras"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    return {"val_acc": metrics.get("val_accuracy", 0)}


def train_tcn_vol_regime(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.tcn_volatility_trainer import TCNVolatilityRegimeTrainer
    from src.core.modular_data_loaders import load_volatility_regime_data

    cfg = TrainerConfig(
        epochs=100, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=20,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )

    vol_data = load_volatility_regime_data(df_feat)
    if not vol_data:
        return {"error": "No volatility regime data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TCNVolatilityRegimeTrainer(cfg)
    trainer.train(vol_data["X_train"], vol_data["y_train"], vol_data["X_val"], vol_data["y_val"],
                  seq_len=params["seq_len"])
    trainer.save(str(save_dir / "tcn_volatility_regime.keras"))

    metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
    return {"val_acc": metrics.get("val_accuracy", 0)}


MODEL_TRAINERS = {
    "transformer": train_transformer,
    "tcn": train_tcn,
    "lgbm_momentum": train_lgbm_momentum,
    "lgbm_risk": train_lgbm_risk,
    "ridge": train_ridge,
    "histgb": train_histgb,
    "transformer_regime": train_transformer_regime,
    "tcn_vol_regime": train_tcn_vol_regime,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--model", required=True, choices=list(MODEL_TRAINERS.keys()))
    parser.add_argument("--candles", type=int, default=25000)
    parser.add_argument("--granularity", default="H1")
    args = parser.parse_args()

    t0 = time.time()
    instrument = args.instrument
    params = MASTER_PARAMS.get(instrument, {"seq_len": 60, "batch_size": 64, "lr": 0.0005, "patience": 20})

    # Fetch/load data
    df = fetch_or_load(instrument, args.candles, args.granularity)
    if df.empty:
        print(json.dumps({"error": "No data", "instrument": instrument, "model": args.model}))
        sys.exit(1)

    # Feature engineering
    df_feat = apply_features(df)
    del df; gc.collect()

    logger.info(f"Training {args.model} for {instrument} ({len(df_feat)} rows, {len(df_feat.columns)} features)")

    # Train
    trainer_fn = MODEL_TRAINERS[args.model]
    try:
        result = trainer_fn(instrument, df_feat, params)
        result["instrument"] = instrument
        result["model"] = args.model
        result["duration_s"] = round(time.time() - t0, 1)
        result["passed"] = result.get("gap", 0) <= MAX_GAP
        print(f"RESULT:{json.dumps(result)}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        print(f"RESULT:{json.dumps({'error': str(e), 'instrument': instrument, 'model': args.model, 'duration_s': round(time.time() - t0, 1)})}")
        sys.exit(1)


if __name__ == "__main__":
    main()
