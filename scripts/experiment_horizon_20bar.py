#!/usr/bin/env python3
"""Test: does 20-bar return direction beat 51%?"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import logging
import numpy as np
from pathlib import Path
from tensorflow import keras

from sota_core.raw_sequence_model import RawSequenceModel, ModelConfig
from sota_core.trainer import SOTATrainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

seq_len = 128
df_path = "trained_data/harvest/EUR_USD_M15.parquet"

# Load data manually
trainer = SOTATrainer()
wins = list(trainer._make_windows(trainer._load_candles_csv(df_path), seq_len, stride=1))

# 20-bar labels
y_dir, y_reg = [], []
import pandas as pd
df = pd.read_parquet(df_path)
for i, w in enumerate(wins):
    end_idx = i + seq_len
    if end_idx + 20 >= len(df):
        continue
    future_ret = (df["close"].iloc[end_idx + 19] / df["close"].iloc[end_idx - 1]) - 1
    y_dir.append(1 if future_ret > 0 else 0)
    future_range = df["close"].iloc[end_idx : end_idx + 20]
    vol = future_range.pct_change().std() * 100
    if vol < 0.03:
        y_reg.append(0)
    elif vol < 0.07:
        y_reg.append(1)
    elif vol < 0.15:
        y_reg.append(2)
    else:
        y_reg.append(3)

X = np.stack(wins[:len(y_dir)])
y_dir = np.array(y_dir, dtype=np.int32)
y_reg = np.array(y_reg, dtype=np.int32)
logger.info(f"Samples: {len(X)}, LONG={(y_dir==1).mean():.2%}")

# Temporal split
n_train = int(len(X) * 0.80)
n_val = int(len(X) * 0.10)
train_X = X[:n_train]
train_y = {"direction": y_dir[:n_train], "regime": y_reg[:n_train]}
val_X = X[n_train : n_train + n_val]
val_y = {"direction": y_dir[n_train : n_train + n_val], "regime": y_reg[n_train : n_train + n_val]}

# Build model
model = RawSequenceModel(ModelConfig())
model.build()

# Load pretrain weights
pt_weights = Path("trained_data/models/sota_pretrain/pretrain_weights.weights.h5")
if pt_weights.exists():
    try:
        model._encoder.load_weights(str(pt_weights))
        logger.info("Loaded pretrain weights")
    except Exception as e:
        logger.warning(f"Pretrain load failed: {e}")

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_direction_loss", mode="min", factor=0.5, patience=2, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_direction_loss", mode="min", patience=3, restore_best_weights=True, verbose=1),
]

history = model.model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=64,
    epochs=10,
    callbacks=callbacks,
    verbose=2,
)

metrics = {k: float(v[-1]) for k, v in history.history.items()}
logger.info(f"RESULTS: {metrics}")

save_dir = Path("trained_data/models/experiments/horizon_20bar_temporal")
save_dir.mkdir(parents=True, exist_ok=True)
model.model.save(f"{save_dir}/sota_model.keras")
logger.info(f"Saved to {save_dir}")
