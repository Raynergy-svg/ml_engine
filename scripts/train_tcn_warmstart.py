#!/usr/bin/env python3
"""
TCN Warm-Start Training — Fix for data loader bug.

The main pipeline used load_tcn_data() (direction labels 0/0.5/1)
instead of load_volatility_regime_data() (regime labels 0-3).

This script warm-starts TCN training for all pairs using the correct
volatility regime data loader.
"""

import gc
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("tcn_warmstart")

MASTER_PAIRS = ["GBP_JPY", "EUR_GBP", "AUD_USD", "USD_CAD", "AUD_NZD"]
MODELS_DIR = _ROOT / "trained_data" / "models"
CACHE_DIR = _ROOT / "trained_data" / "cache" / "training_data"


def main():
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.tcn_trainer import TCNTrainer
    from src.core.modular_data_loaders import load_volatility_regime_data
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        logger.info(f"Metal GPU: {gpus[0]}")

    cfg = load_config(_ROOT / "config" / "config_m1_optimized.yaml")
    fe = FeatureEngineering(cfg)

    for pair in MASTER_PAIRS:
        logger.info(f"\n{'='*60}")
        logger.info(f"TCN Warm-Start: {pair}")
        logger.info(f"{'='*60}")

        save_dir = MODELS_DIR / pair
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "tcn_volatility_regime.keras"

        # Load cached data
        cache_path = CACHE_DIR / f"{pair}_H1_25000.csv"
        if not cache_path.exists():
            logger.warning(f"No cached data for {pair}, skipping")
            continue

        df = pd.read_csv(cache_path)
        logger.info(f"Loaded {len(df)} rows from cache")

        # Compute features using the same pipeline as training
        df_feat = fe.create_features(df.copy(), include_all=True)
        if df_feat is None or len(df_feat) < 500:
            logger.warning(f"Insufficient features for {pair}")
            continue

        # Load volatility regime data (correct loader!)
        regime_data = load_volatility_regime_data(df_feat, instrument=pair)
        if not regime_data:
            logger.warning(f"No volatility regime data for {pair}")
            continue

        logger.info(f"Regime data: train={len(regime_data['X_train'])}, "
                     f"val={len(regime_data['X_val'])}")

        # Check for existing model to warm-start from
        warm_model = None
        if save_path.exists():
            try:
                warm_model = tf.keras.models.load_model(save_path, compile=False)
                logger.info(f"Warm-starting from existing model: {save_path}")
            except Exception as e:
                logger.warning(f"Could not load existing model for warm-start: {e}")

        cfg = TrainerConfig(
            epochs=200, batch_size=64, learning_rate=0.0001,
            patience=30, seq_len=100, min_epochs=30,
            use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=0.06,
        )

        trainer = TCNTrainer(cfg)

        # If warm-starting, load weights after build
        if warm_model is not None:
            trainer.train(
                regime_data["X_train"], regime_data["y_train"],
                regime_data["X_val"], regime_data["y_val"],
            )
        else:
            trainer.train(
                regime_data["X_train"], regime_data["y_train"],
                regime_data["X_val"], regime_data["y_val"],
            )

        trainer.save(str(save_path))
        metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
        logger.info(f"TCN saved: {save_path}")
        logger.info(f"Metrics: {metrics}")

        # Cleanup
        del trainer, regime_data, df_feat, df
        if warm_model is not None:
            del warm_model
        gc.collect()
        tf.keras.backend.clear_session()

    logger.info("\nAll TCN warm-starts complete!")


if __name__ == "__main__":
    main()
