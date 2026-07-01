#!/usr/bin/env python3
"""Test: does multi-horizon multi-task beat 51%?"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras

from sota_core.raw_sequence_model import RawSequenceModel, ModelConfig
from sota_core.trainer import SOTATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [5, 10, 20]
SEQ_LEN = 128

df_path = "trained_data/harvest/EUR_USD_M15.parquet"

trainer = SOTATrainer()
wins = list(trainer._make_windows(trainer._load_candles_csv(df_path), SEQ_LEN, stride=1))
df = pd.read_parquet(df_path)

labels = {h: [] for h in HORIZONS}
labels_reg = []
valid_wins = []

for i, w in enumerate(wins):
    end_idx = i + SEQ_LEN
    if end_idx + 20 >= len(df):
        continue
    future_range = df["close"].iloc[end_idx : end_idx + 20]
    vol = future_range.pct_change().std() * 100
    if vol < 0.03:
        labels_reg.append(0)
    elif vol < 0.07:
        labels_reg.append(1)
    elif vol < 0.15:
        labels_reg.append(2)
    else:
        labels_reg.append(3)

    for h in HORIZONS:
        future_ret = (df["close"].iloc[end_idx + h - 1] / df["close"].iloc[end_idx - 1]) - 1
        labels[h].append(1 if future_ret > 0 else 0)
    valid_wins.append(w)

X = np.stack(valid_wins)
for h in HORIZONS:
    labels[h] = np.array(labels[h], dtype=np.int32)
labels_reg = np.array(labels_reg, dtype=np.int32)

logger.info(f"Samples: {len(X)}, LONG_5={(labels[5]==1).mean():.2%}, LONG_10={(labels[10]==1).mean():.2%}, LONG_20={(labels[20]==1).mean():.2%}")

n_train = int(len(X) * 0.80)
n_val = int(len(X) * 0.10)
train_X = X[:n_train]
val_X = X[n_train : n_train + n_val]

train_y = {f"direction_{h}": labels[h][:n_train] for h in HORIZONS}
train_y["regime"] = labels_reg[:n_train]
val_y = {f"direction_{h}": labels[h][n_train : n_train + n_val] for h in HORIZONS}
val_y["regime"] = labels_reg[n_train : n_train + n_val]

model = RawSequenceModel(ModelConfig(direction_horizons=tuple(HORIZONS)))
model.build()

pt_weights = Path("trained_data/models/sota_pretrain/pretrain_weights.weights.h5")
if pt_weights.exists():
    try:
        model._encoder.load_weights(str(pt_weights))
        logger.info("Loaded pretrain weights")
    except Exception as e:
        logger.warning(f"Pretrain load failed: {e}")

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_direction_5_loss", mode="min", factor=0.5, patience=2, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_direction_5_loss", mode="min", patience=3, restore_best_weights=True, verbose=1),
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

save_dir = Path("trained_data/models/experiments/multi_horizon_5_10_20")
save_dir.mkdir(parents=True, exist_ok=True)
model.model.save(f"{save_dir}/sota_model.keras")
logger.info(f"Saved to {save_dir}")
