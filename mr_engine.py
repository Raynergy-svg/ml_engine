"""MR head (pure TensorFlow).

This project is now TensorFlow-only. `MREngineHead` is a reusable LSTM head
that maps a shared sequence input (OHLCV + engineered features) to a fixed-size
embedding vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import tensorflow as tf


@dataclass(frozen=True)
class MREngineHeadConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    projection_dim: Optional[int] = None


@tf.keras.utils.register_keras_serializable(package="buddy")
class MREngineHead(tf.keras.Model):
    """LSTM head that produces a hidden embedding."""

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        projection_dim: Optional[int] = None,
        name: str = "mr_head",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.projection_dim = int(projection_dim) if projection_dim is not None else None

        self._lstms: list[tf.keras.layers.Layer] = []
        for i in range(self.num_layers):
            self._lstms.append(
                tf.keras.layers.LSTM(
                    self.hidden_size,
                    return_sequences=(i < self.num_layers - 1),
                    dropout=self.dropout,
                    recurrent_dropout=0.0,
                    name=f"lstm_{i}",
                )
            )

        self._norm = tf.keras.layers.LayerNormalization(name="norm")
        self._proj = (
            tf.keras.layers.Dense(self.projection_dim, activation="tanh", name="proj")
            if self.projection_dim is not None
            else None
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = x
        for layer in self._lstms:
            h = layer(h, training=training)
        h = self._norm(h, training=training)
        if self._proj is not None:
            h = self._proj(h, training=training)
        return h

    def get_config(self) -> Dict[str, Any]:
        base = super().get_config()
        return {
            **base,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "projection_dim": self.projection_dim,
            "name": self.name,
        }

# — Raynergy-svg —
