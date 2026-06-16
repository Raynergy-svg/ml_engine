"""
Raw-sequence deep learning model for end-to-end market signal generation.

Architecture: Hybrid CNN-Transformer with variable selection attention.
- Raw OHLCV (5 channels) fed directly as a time-series
- 1D convolutions for local pattern extraction (learned instead of SMA/RSI)
- Multi-head self-attention for long-range temporal dependencies
- Variable selection network learns which channels matter per timestep
- Dual heads: direction (sigmoid) + regime (softmax over LOW/NORMAL/HIGH/EXTREME)

No hand-engineered features.  The model discovers trend, mean-reversion,
momentum, volatility, and support/resistance concepts in latent space.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
try:
    from src.sota_core.encoders.itransformer import iTransformerEncoder
except Exception:
    iTransformerEncoder = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Hyperparameters for the raw-sequence model."""

    seq_len: int = 128          # bars of history
    d_model: int = 128          # embedding dimension
    num_heads: int = 4          # transformer heads
    num_layers: int = 3         # transformer encoder layers
    dropout: float = 0.25       # regularization
    cnn_filters: Optional[List[int]] = None  # per-layer conv filters
    cnn_kernel: int = 5         # kernel size for local convolutions
    dense_units: int = 64       # final MLP before heads
    learning_rate: float = 1e-3
    direction_weight: float = 1.0
    regime_weight: float = 0.3  # auxiliary loss
    encoder_type: str = "transformer"  # "transformer" | "itransformer"

    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, self.d_model]


class VariableSelectionNetwork(keras.layers.Layer):
    """Learns per-timestep importance weights for each input channel."""

    def __init__(self, num_inputs: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.d_model = d_model
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.context_gru = keras.layers.GRU(self.d_model, return_sequences=True)
        self.selection_dense = keras.layers.Dense(self.num_inputs, activation="softmax")
        self.input_projections = [
            keras.layers.Dense(self.d_model, activation="relu")
            for _ in range(self.num_inputs)
        ]
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        context = self.context_gru(inputs)
        selection = self.selection_dense(context)
        selection = self.dropout(selection, training=training)
        projected = tf.stack(
            [proj(inputs[..., i:i + 1]) for i, proj in enumerate(self.input_projections)],
            axis=2,
        )
        selection = tf.expand_dims(selection, -1)
        weighted = projected * selection
        output = tf.reduce_sum(weighted, axis=2)
        return output


class GatedResidualNetwork(keras.layers.Layer):
    """GRN from TFT — allows the model to skip layers when they add noise."""

    def __init__(self, units: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = keras.layers.Dense(self.units, activation="elu")
        self.dense2 = keras.layers.Dense(self.units)
        self.gate_dense = keras.layers.Dense(self.units, activation="sigmoid")
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        if input_shape[-1] != self.units:
            self.skip_proj = keras.layers.Dense(self.units)
        else:
            self.skip_proj = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        gate = self.gate_dense(inputs)
        x = x * gate
        skip = self.skip_proj(inputs) if self.skip_proj is not None else inputs
        return self.layer_norm(skip + x)


class RawSequenceModel:
    """End-to-end model: raw OHLCV -> learned representations -> direction + regime."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self.model: Optional[keras.Model] = None
        self._built = False

    def build(self) -> keras.Model:
        cfg = self.cfg
        inputs = keras.Input(shape=(cfg.seq_len, 5), name="ohlcv")

        x = VariableSelectionNetwork(
            num_inputs=5, d_model=cfg.d_model, dropout=cfg.dropout, name="varsel"
        )(inputs)

        for i, filt in enumerate(cfg.cnn_filters):
            x = keras.layers.Conv1D(
                filters=filt,
                kernel_size=cfg.cnn_kernel,
                padding="causal",
                activation="relu",
                name=f"conv_{i}",
            )(x)
            x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = keras.layers.Dropout(cfg.dropout, name=f"conv_drop_{i}")(x)

        pos_emb = keras.layers.Embedding(
            input_dim=cfg.seq_len,
            output_dim=cfg.d_model,
            name="pos_embedding",
        )(tf.range(cfg.seq_len))
        x = x + pos_emb

        if cfg.encoder_type == "itransformer" and iTransformerEncoder is not None:
            x = iTransformerEncoder(
                seq_len=cfg.seq_len,
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                name="itransformer",
            )(x)
        else:
            for i in range(cfg.num_layers):
                attn = keras.layers.MultiHeadAttention(
                    num_heads=cfg.num_heads,
                    key_dim=cfg.d_model // cfg.num_heads,
                    name=f"mha_{i}",
                )(x, x)
                attn = keras.layers.Dropout(cfg.dropout, name=f"attn_drop_{i}")(attn)
                x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x + attn)

                ffn = keras.layers.Dense(cfg.d_model * 4, activation="relu", name=f"ffn1_{i}")(x)
                ffn = keras.layers.Dense(cfg.d_model, name=f"ffn2_{i}")(ffn)
                ffn = keras.layers.Dropout(cfg.dropout, name=f"ffn_drop_{i}")(ffn)
                x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x + ffn)

        seq_repr = x  # (batch, seq_len, d_model)
        pooled = keras.layers.GlobalAveragePooling1D(name="gap")(seq_repr)
        grn = GatedResidualNetwork(cfg.dense_units, dropout=cfg.dropout, name="grn")(pooled)
        grn = keras.layers.Dropout(cfg.dropout, name="grn_drop")(grn)

        direction = keras.layers.Dense(
            1, activation="sigmoid", name="direction", dtype="float32"
        )(grn)
        regime = keras.layers.Dense(
            4, activation="softmax", name="regime", dtype="float32"
        )(grn)

        model = keras.Model(inputs=inputs, outputs=[direction, regime], name="sota_raw_sequence")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss={
                "direction": "binary_crossentropy",
                "regime": "sparse_categorical_crossentropy",
            },
            loss_weights={
                "direction": cfg.direction_weight,
                "regime": cfg.regime_weight,
            },
            metrics={"direction": "accuracy"},
        )
        # CRIT-3 FIX: encoder is an explicit sub-model, no layer traversal.
        encoder = keras.Model(inputs=inputs, outputs=seq_repr, name="sota_encoder")
        self.model = model
        self._encoder = encoder
        self._built = True
        return model

    def build_pretraining_model(self) -> keras.Model:
        """CRIT-3 FIX: encoder is an explicit sub-model, no layer traversal."""
        if not self._built:
            self.build()

        encoder_input = self._encoder.input
        seq_repr = self._encoder(encoder_input)

        recon = keras.layers.TimeDistributed(
            keras.layers.Dense(5, activation="linear"),
            name="recon_head",
        )(seq_repr)

        last_step = seq_repr[:, -1, :]
        next_ret = keras.layers.Dense(1, activation="linear", name="next_return")(last_step)

        pt_model = keras.Model(
            inputs=encoder_input,
            outputs=[recon, next_ret],
            name="sota_pretrain",
        )
        pt_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss={"recon_head": "mse", "next_return": "mse"},
            loss_weights={"recon_head": 1.0, "next_return": 0.5},
        )
        return pt_model

    @staticmethod
    def _load_normalization_stats(path: Optional[str]) -> Optional[Dict[str, Dict[str, float]]]:
        """Load per-pair normalization stats from JSON."""
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("stats")
        except Exception as e:
            logger.debug(f"Failed to load normalization stats from {path}: {e}")
            return None

    @staticmethod
    def df_to_tensor(
        df: pd.DataFrame,
        seq_len: int,
        normalization_stats_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Convert OHLCV DataFrame to model input tensor.

        CRIT-2 FIX: Volume normalization uses fixed per-pair statistics from
        `normalization_stats_path` instead of a rolling window anchored to
        the start of the DataFrame.  If no stats file is provided, falls back
        to computing z-scores from the ENTIRE passed window (still window-
        dependent but not anchored to history).

        Returns (batch=1, seq_len, 5) or None if insufficient data.
        """
        if df is None or len(df) < seq_len:
            return None
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset({c.lower() for c in df.columns}):
            cols_lower = {c.lower(): c for c in df.columns}
            missing = required - set(cols_lower.keys())
            if missing:
                logger.debug(f"Missing columns for raw-sequence model: {missing}")
                return None
            df = df.rename(columns={v: k for k, v in cols_lower.items() if k in required})

        df = df.copy()
        # Price channels: use returns (percentage changes)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Volume: z-score with fixed stats if available, else tail-window stats
        if "volume" in df.columns:
            vol_all = df["volume"].replace(0, np.nan).values.astype(np.float32)
            stats = RawSequenceModel._load_normalization_stats(normalization_stats_path)
            if stats and "volume" in stats:
                # Fixed per-pair stats from training data — fully deterministic
                vm = stats["volume"].get("mean", 0.0)
                vs = stats["volume"].get("std", 1.0)
                if vs <= 0:
                    vs = 1.0
                df["volume"] = pd.Series((vol_all - vm) / vs).fillna(0).values
            else:
                # CRIT-2 FIX: compute stats from the TAIL WINDOW only, not the
                # entire DataFrame.  This ensures two DataFrames with the same
                # last seq_len bars produce identical volume normalizations.
                vol_tail = vol_all[-seq_len:]
                vm = float(np.nanmean(vol_tail))
                vs = float(np.nanstd(vol_tail))
                if vs < 1e-8:
                    vs = 1.0
                df["volume"] = pd.Series((vol_all - vm) / vs).fillna(0).values
        else:
            df["volume"] = 0.0

        arr = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        window = arr[-seq_len:]
        if len(window) < seq_len:
            pad = seq_len - len(window)
            window = np.concatenate([np.zeros((pad, 5), dtype=np.float32), window], axis=0)
        return window[np.newaxis, ...]

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"SOTA model saved to {path}")

    def load(self, path: str) -> bool:
        try:
            self.model = keras.models.load_model(
                path,
                custom_objects={
                    "VariableSelectionNetwork": VariableSelectionNetwork,
                    "GatedResidualNetwork": GatedResidualNetwork,
                },
            )
            self._built = True
            logger.info(f"SOTA model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load SOTA model from {path}: {e}")
            return False
