#!/usr/bin/env python3
"""Test: does 3-pair concatenation beat 51%?"""

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

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
SEQ_LEN = 128

# Load and align all pairs
dfs = {}
for pair in PAIRS:
    path = f"trained_data/harvest/{pair}_M15.parquet"
    df = pd.read_parquet(path)
    df = df.sort_index()
    dfs[pair] = df

# Align to common index
common_idx = dfs[PAIRS[0]].index
for pair in PAIRS[1:]:
    common_idx = common_idx.intersection(dfs[pair].index)

logger.info(f"Common bars: {len(common_idx)}")
for pair in PAIRS:
    dfs[pair] = dfs[pair].loc[common_idx]

# Normalize each pair independently (same as _make_windows)
def normalize_df(df):
    df = df.copy()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    vol = df["volume"].replace(0, np.nan)
    vm = vol.rolling(100, min_periods=1).mean()
    vs = vol.rolling(100, min_periods=1).std().replace(0, 1)
    df["volume"] = ((vol - vm) / vs).fillna(0)
    return df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)

arrs = {p: normalize_df(dfs[p]) for p in PAIRS}
arr = np.concatenate([arrs[p] for p in PAIRS], axis=1)  # (N, 15)
arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# Build windows: shape (samples, SEQ_LEN, 15)
windows = []
labels_dir = []
labels_reg = []

for i in range(len(arr) - SEQ_LEN - 5):
    w = arr[i : i + SEQ_LEN]
    windows.append(w)
    # Label from EUR/USD only (channels 0-3 are EUR/USD close)
    future_ret = (dfs["EUR_USD"]["close"].iloc[i + SEQ_LEN + 4] / dfs["EUR_USD"]["close"].iloc[i + SEQ_LEN - 1]) - 1
    labels_dir.append(1 if future_ret > 0 else 0)
    future_range = dfs["EUR_USD"]["close"].iloc[i + SEQ_LEN : i + SEQ_LEN + 5]
    vol = future_range.pct_change().std() * 100
    if vol < 0.03:
        labels_reg.append(0)
    elif vol < 0.07:
        labels_reg.append(1)
    elif vol < 0.15:
        labels_reg.append(2)
    else:
        labels_reg.append(3)

X = np.stack(windows)
y_dir = np.array(labels_dir, dtype=np.int32)
y_reg = np.array(labels_reg, dtype=np.int32)

logger.info(f"Samples: {len(X)}, LONG={(y_dir==1).mean():.2%}")

# Temporal split
n_train = int(len(X) * 0.80)
n_val = int(len(X) * 0.10)
train_X = X[:n_train]
train_y = {"direction": y_dir[:n_train], "regime": y_reg[:n_train]}
val_X = X[n_train : n_train + n_val]
val_y = {"direction": y_dir[n_train : n_train + n_val], "regime": y_reg[n_train : n_train + n_val]}

# Build model with 15 channels
model = RawSequenceModel(ModelConfig(num_inputs=15))
model.build()

# Load pretrain weights (they won't fit because encoder expects different input shape)
# Skip pretrain weights for this experiment — train from scratch
logger.info("Skipping pretrain weights (shape mismatch with 15 channels)")

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

save_dir = Path("trained_data/models/experiments/multi_pair_3concat_temporal")
save_dir.mkdir(parents=True, exist_ok=True)
model.model.save(f"{save_dir}/sota_model.keras")
logger.info(f"Saved to {save_dir}")
