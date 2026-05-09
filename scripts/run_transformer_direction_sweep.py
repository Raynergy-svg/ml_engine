"""W&B Sweep runner for the TransformerDirectionTrainer (master-pair tuning).

Called per-trial by `wandb agent`. Each invocation:
  1. Reads hyperparameters from W&B config (set by sweep agent)
  2. Fetches OANDA H1 candles for the configured pair
  3. Runs load_direction_data
  4. Trains TransformerDirectionTrainer with the trial's hyperparameters
  5. Logs val_balanced_accuracy to W&B (the sweep metric)
  6. Does NOT save the trained model (sweep is search, not deployment)

Usage:
  # 1. Create sweep config:
  python -c "from src.training.wandb_sweep_config import get_transformer_direction_sweep_config; \\
             print(get_transformer_direction_sweep_config('USD_JPY').to_yaml())" > /tmp/sweep.yaml

  # 2. Create sweep:
  wandb sweep --project buddy-master-tuning /tmp/sweep.yaml
  # → returns sweep_id

  # 3. Run agent (one trial at a time on M1):
  wandb agent --count 30 <entity/buddy-master-tuning/sweep_id>
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transformer_sweep")


def main() -> int:
    import wandb
    import pandas as pd

    # W&B agent injects hyperparameters into wandb.config.
    run = wandb.init()
    cfg = wandb.config

    # Required pair + granularity (passed via wandb config or env).
    pair = cfg.get("pair") or os.environ.get("SWEEP_PAIR", "USD_JPY")
    granularity = cfg.get("granularity", "H1")
    candles = int(cfg.get("candles", 25000))
    lookahead = int(cfg.get("lookahead", 24))
    threshold = float(cfg.get("threshold", 0.0))

    log.info("=" * 60)
    log.info("Sweep trial: pair=%s gran=%s candles=%d", pair, granularity, candles)
    log.info("Hyperparameters: %s", dict(cfg))
    log.info("=" * 60)

    # Fetch candles
    from src.utils.oanda_practice import OandaPracticeClient
    from src.training.buddy_training_helpers import _fetch_paginated_candles
    oanda = OandaPracticeClient.from_env()
    rows = _fetch_paginated_candles(oanda, pair, granularity, candles)
    if not rows:
        log.error("OANDA returned no candles for %s", pair)
        wandb.log({"error": "no_candles"})
        return 1

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True).set_index("time", drop=False)
    log.info("Fetched %d candles", len(df))

    # Compute features + load direction data
    from src.core.modular_data_loaders import compute_normalized_features, load_direction_data
    df_feat = compute_normalized_features(df)
    dir_result = load_direction_data(
        df_feat, apply_scaler=False, lookahead=lookahead, threshold=threshold,
    )
    if dir_result is None:
        log.error("load_direction_data returned None")
        wandb.log({"error": "loader_failed"})
        return 1

    # Build trainer with sweep hyperparameters
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer

    trainer_cfg = TrainerConfig()
    # Override defaults with sweep hyperparameters
    if "transformer_d_model" in cfg:
        trainer_cfg.transformer_d_model = int(cfg.transformer_d_model)
    if "transformer_num_heads" in cfg:
        trainer_cfg.transformer_num_heads = int(cfg.transformer_num_heads)
    if "transformer_num_layers" in cfg:
        trainer_cfg.transformer_num_layers = int(cfg.transformer_num_layers)
    if "transformer_dff" in cfg:
        trainer_cfg.transformer_dff = int(cfg.transformer_dff)
    if "transformer_dropout" in cfg:
        trainer_cfg.transformer_dropout = float(cfg.transformer_dropout)
    if "learning_rate" in cfg:
        trainer_cfg.learning_rate = float(cfg.learning_rate)
    if "batch_size" in cfg:
        trainer_cfg.batch_size = int(cfg.batch_size)
    if "top_k_features" in cfg:
        trainer_cfg.top_k_features = int(cfg.top_k_features)
    if "ewc_lambda" in cfg:
        trainer_cfg.ewc_lambda = float(cfg.ewc_lambda)
    if "patience" in cfg:
        trainer_cfg.patience = int(cfg.patience)
    if "epochs" in cfg:
        trainer_cfg.epochs = int(cfg.epochs)

    log.info("Effective trainer config: d_model=%s heads=%s layers=%s dff=%s dropout=%.2f lr=%.6f",
             trainer_cfg.transformer_d_model, trainer_cfg.transformer_num_heads,
             trainer_cfg.transformer_num_layers, trainer_cfg.transformer_dff,
             trainer_cfg.transformer_dropout, trainer_cfg.learning_rate)

    trainer = TransformerDirectionTrainer(trainer_cfg)
    metrics = trainer.train(
        dir_result["X_train"], dir_result["y_train"],
        dir_result["X_val"], dir_result["y_val"],
        feature_names=dir_result["feature_names"],
        w_train=dir_result.get("w_train"),
        w_val=dir_result.get("w_val"),
        instrument=pair,
        skip_scaling=False,
        regime_quantiles=dir_result.get("regime_quantiles"),
        regime_atr_col=dir_result.get("regime_atr_col"),
        feature_pipeline_version=dir_result.get("feature_pipeline_version"),
    )

    # Log to W&B for sweep-driver consumption
    summary_metrics = {
        "val_accuracy": metrics.get("val_accuracy", 0.0),
        "val_balanced_accuracy": metrics.get("val_balanced_accuracy", 0.0),
        "val_loss": metrics.get("val_loss", 0.0),
        "train_accuracy": metrics.get("train_accuracy", 0.0),
        "epochs_trained": metrics.get("epochs_trained", 0),
        "n_train_samples": metrics.get("n_train_samples", 0),
    }
    log.info("Sweep trial result: %s", summary_metrics)
    wandb.log(summary_metrics)
    wandb.summary.update(summary_metrics)
    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
