"""
Training pipeline for the SOTA raw-sequence model.

Two-phase training:
  Phase 1 — Self-supervised pre-training on years of unlabeled OHLCV data.
            Tasks: masked candle reconstruction + next-return regression.
  Phase 2 — Supervised fine-tuning on labeled directional outcomes.

The pre-training phase lets the model discover trend, volatility, momentum,
and mean-reversion concepts in latent space without any hand-coded features.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .raw_sequence_model import RawSequenceModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    pretrain_epochs: int = 50
    finetune_epochs: int = 30
    learning_rate: float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience: int = 5
    early_stop_patience: int = 10
    mask_rate: float = 0.15          # fraction of timesteps to mask in pretraining
    min_seq_len: int = 128
    train_split: float = 0.80
    val_split: float = 0.10
    # Data augmentation
    jitter_scale: float = 0.01       # Gaussian noise on returns
    time_warp_sigma: float = 0.1     # small temporal distortions


class SOTATrainer:
    """Orchestrates two-phase training."""

    def __init__(
        self,
        model: Optional[RawSequenceModel] = None,
        config: Optional[TrainerConfig] = None,
    ):
        self.model = model or RawSequenceModel()
        self.cfg = config or TrainerConfig()
        self._pretrain_model: Optional[keras.Model] = None

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    @staticmethod
    def _load_candles_csv(csv_path: str) -> Optional[pd.DataFrame]:
        """Load canonical OHLCV from CSV or Parquet."""
        try:
            path = Path(csv_path)
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(csv_path)
                if "time" in df.columns:
                    df = df.set_index("time")
                df = df.sort_index()
            else:
                df = pd.read_csv(csv_path, parse_dates=["time"])
                df = df.set_index("time").sort_index()
            for col in ("open", "high", "low", "close"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
            return df.dropna(subset=["open", "high", "low", "close"])
        except Exception as e:
            logger.warning(f"Could not load {csv_path}: {e}")
            return None

    def _make_windows(
        self,
        df: pd.DataFrame,
        seq_len: int,
        stride: int = 1,
    ) -> Generator[np.ndarray, None, None]:
        """Yield normalized (seq_len, 5) windows from a DataFrame."""
        if len(df) < seq_len:
            return
        # Normalize exactly as in df_to_tensor
        df = df.copy()
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
        vol = df["volume"].replace(0, np.nan)
        vm = vol.rolling(100, min_periods=1).mean()
        vs = vol.rolling(100, min_periods=1).std().replace(0, 1)
        df["volume"] = ((vol - vm) / vs).fillna(0)

        arr = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(0, len(arr) - seq_len, stride):
            yield arr[i : i + seq_len]

    # ------------------------------------------------------------------
    # Phase 1 — Self-supervised pre-training
    # ------------------------------------------------------------------
    def pretrain(
        self,
        data_paths: List[str],
        save_dir: str = "trained_data/models/sota_pretrain",
    ) -> Dict[str, float]:
        """Pre-train the encoder on unlabeled OHLCV data.

        Args:
            data_paths: List of CSV file paths with OHLCV data.
            save_dir: Where to save the pre-trained weights.

        Returns:
            Training history metrics.
        """
        cfg = self.cfg
        seq_len = self.model.cfg.seq_len

        # Build windows
        windows: List[np.ndarray] = []
        for p in data_paths:
            df = self._load_candles_csv(p)
            if df is not None:
                windows.extend(list(self._make_windows(df, seq_len, stride=max(seq_len // 4, 1))))

        if len(windows) < cfg.batch_size * 2:
            raise RuntimeError(
                f"Insufficient data for pre-training: {len(windows)} windows "
                f"(need at least {cfg.batch_size * 2})"
            )

        # OBS-3 FIX: Temporal split instead of random shuffle.
        # Adjacent windows overlap heavily (seq_len-1 bars shared).
        # Random shuffle leaks validation data into training.
        n_total = len(windows)
        n_train = int(n_total * cfg.train_split)
        n_val = int(n_total * cfg.val_split)
        train_X = np.stack(windows[:n_train])
        val_X = np.stack(windows[n_train : n_train + n_val])

        # Build pre-training model (reconstruction + next-return)
        pt_model = self.model.build_pretraining_model()
        self._pretrain_model = pt_model

        # Create masked targets
        train_ds = self._make_pretrain_dataset(train_X, cfg.batch_size, shuffle=True)
        val_ds = self._make_pretrain_dataset(val_X, cfg.batch_size, shuffle=False)

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=cfg.lr_decay_factor, patience=cfg.lr_patience, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.early_stop_patience, restore_best_weights=True, verbose=1
            ),
        ]

        logger.info(f"Pre-training on {len(train_X)} windows, validating on {len(val_X)}")
        history = pt_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.pretrain_epochs,
            callbacks=callbacks,
            verbose=2,
        )

        # Save encoder weights for fine-tuning
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model._encoder.save_weights(f"{save_dir}/pretrain_weights.weights.h5")
        logger.info(f"Pre-training complete. Encoder weights saved to {save_dir}")

        return {k: float(v[-1]) for k, v in history.history.items()}

    def _make_pretrain_dataset(
        self,
        X: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """Dataset that masks random timesteps and returns reconstruction targets."""
        cfg = self.cfg

        def _generator():
            indices = np.arange(len(X))
            if shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                x = X[idx].copy()  # (seq_len, 5)
                # Random timestep masking
                mask_positions = np.random.rand(self.model.cfg.seq_len) < cfg.mask_rate
                recon_target = x.copy()
                x[mask_positions] = 0.0  # zero-mask (model learns to fill in)

                # Next-return target: log-return of the bar AFTER the window
                # We don't have it in the window, so we approximate from the last close
                next_ret = np.array([x[-1, 3]], dtype=np.float32)  # close return as proxy

                # Data augmentation: jitter
                if cfg.jitter_scale > 0:
                    noise = np.random.normal(0, cfg.jitter_scale, x.shape).astype(np.float32)
                    x = x + noise

                yield x, {"recon_head": recon_target, "next_return": next_ret}

        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(self.model.cfg.seq_len, 5), dtype=tf.float32),
                {
                    "recon_head": tf.TensorSpec(shape=(self.model.cfg.seq_len, 5), dtype=tf.float32),
                    "next_return": tf.TensorSpec(shape=(1,), dtype=tf.float32),
                },
            ),
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------------
    # Phase 2 — Supervised fine-tuning
    # ------------------------------------------------------------------
    def finetune(
        self,
        data_paths: List[str],
        labels: Optional[Dict[str, np.ndarray]] = None,
        pretrain_weights_dir: str = "trained_data/models/sota_pretrain",
        save_dir: str = "trained_data/models/sota_finetuned",
    ) -> Dict[str, float]:
        """Fine-tune on labeled directional outcomes.

        Args:
            data_paths: CSV paths with OHLCV data.
            labels: Optional dict mapping pair_name -> (n_windows,) int array {0=short,1=long}
                    If None, labels are generated from future returns (forward-looking for demo).
            pretrain_weights_dir: Where pre-trained encoder weights live.
            save_dir: Where to save the fine-tuned model.

        Returns:
            Training history metrics.
        """
        cfg = self.cfg
        seq_len = self.model.cfg.seq_len

        # Build windows + labels
        all_X: List[np.ndarray] = []
        all_y_dir: List[int] = []
        all_y_reg: List[int] = []

        for p in data_paths:
            df = self._load_candles_csv(p)
            if df is None or len(df) < seq_len + 5:
                continue
            wins = list(self._make_windows(df, seq_len, stride=1))
            # Direction label: next 5-bar return
            for i, w in enumerate(wins):
                start_idx = i * 1  # stride=1
                end_idx = start_idx + seq_len
                if end_idx + 5 >= len(df):
                    continue
                future_ret = (df["close"].iloc[end_idx + 4] / df["close"].iloc[end_idx - 1]) - 1
                direction = 1 if future_ret > 0 else 0
                # Regime label: realized volatility percentile of next 5 bars
                future_range = df["close"].iloc[end_idx : end_idx + 5]
                vol = future_range.pct_change().std() * 100
                # Simple quartile buckets
                if vol < 0.03:
                    regime = 0  # LOW
                elif vol < 0.07:
                    regime = 1  # NORMAL
                elif vol < 0.15:
                    regime = 2  # HIGH
                else:
                    regime = 3  # EXTREME
                all_X.append(w)
                all_y_dir.append(direction)
                all_y_reg.append(regime)

        if len(all_X) < cfg.batch_size * 4:
            raise RuntimeError(f"Insufficient labeled data: {len(all_X)} samples")

        X = np.stack(all_X)
        y_dir = np.array(all_y_dir, dtype=np.int32)
        y_reg = np.array(all_y_reg, dtype=np.int32)

        # Shuffle + split
        perm = np.random.permutation(len(X))
        X, y_dir, y_reg = X[perm], y_dir[perm], y_reg[perm]
        n_total = len(X)
        n_train = int(n_total * cfg.train_split)
        n_val = int(n_total * cfg.val_split)

        train_X = X[:n_train]
        train_y = {"direction": y_dir[:n_train], "regime": y_reg[:n_train]}
        val_X = X[n_train : n_train + n_val]
        val_y = {"direction": y_dir[n_train : n_train + n_val], "regime": y_reg[n_train : n_train + n_val]}

        # Build supervised model
        if not self.model._built:
            self.model.build()
        # Load encoder weights from pre-training
        pt_weights = Path(pretrain_weights_dir) / "pretrain_weights.weights.h5"
        if pt_weights.exists():
            try:
                self.model._encoder.load_weights(str(pt_weights))
                logger.info("Loaded pre-trained encoder weights for fine-tuning")
            except Exception as e:
                logger.warning(f"Could not load pretrain weights: {e}")
        else:
            logger.info("No pretrain weights found — training from scratch")

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_direction_loss", mode="min", factor=cfg.lr_decay_factor, patience=cfg.lr_patience, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_direction_loss", mode="min", patience=cfg.early_stop_patience, restore_best_weights=True, verbose=1
            ),
        ]

        logger.info(
            f"Fine-tuning: train={len(train_X)}, val={len(val_X)}, "
            f"long_pct={y_dir.mean():.2%}"
        )
        # OBS-1 FIX: Dataset is naturally balanced (~50%); class_weight disabled
        # because Keras 3 rejects class_weight on multi-output models.
        n_long = int((y_dir[:n_train] == 1).sum())
        n_short = int((y_dir[:n_train] == 0).sum())
        logger.info(f"Label balance: LONG={n_long}, SHORT={n_short}")

        history = self.model.model.fit(
            train_X,
            train_y,
            validation_data=(val_X, val_y),
            batch_size=cfg.batch_size,
            epochs=cfg.finetune_epochs,
            callbacks=callbacks,
            verbose=2,
        )

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(f"{save_dir}/sota_model.keras")
        return {k: float(v[-1]) for k, v in history.history.items()}
