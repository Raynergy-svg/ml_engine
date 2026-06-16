"""Inverted Transformer (iTransformer) encoder block.

Reference: iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting (Liu et al., NeurIPS 2023).

Attends over the feature (d_model) dimension; feed-forward operates over
 the temporal (seq_len) dimension per feature independently.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class iTransformerBlock(keras.layers.Layer):
    """Single iTransformer layer: MHA over features, FF over time."""

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        # input_shape: (batch, seq_len, d_model)
        seq_len = int(input_shape[1])
        self._mha = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name=f"{self.name}_mha",
        )
        self._attn_drop = keras.layers.Dropout(self.dropout_rate)
        self._ln1 = keras.layers.LayerNormalization(epsilon=1e-6)

        self._ffn1 = keras.layers.Dense(seq_len * 4, activation="relu")
        self._ffn2 = keras.layers.Dense(seq_len)
        self._ffn_drop = keras.layers.Dropout(self.dropout_rate)
        self._ln2 = keras.layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, seq_len, d_model)
        x = tf.transpose(inputs, perm=[0, 2, 1])  # (batch, d_model, seq_len)

        attn = self._mha(x, x, training=training)
        attn = self._attn_drop(attn, training=training)
        x = self._ln1(x + attn)

        ffn = self._ffn1(x)
        ffn = self._ffn2(ffn)
        ffn = self._ffn_drop(ffn, training=training)
        x = self._ln2(x + ffn)

        return tf.transpose(x, perm=[0, 2, 1])  # (batch, seq_len, d_model)


class iTransformerEncoder(keras.layers.Layer):
    """Stack of iTransformerBlocks.

    Input/Output shape: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.blocks = [
            iTransformerBlock(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate,
                name=f"{self.name}_block_{i}",
            )
            for i in range(self.num_layers)
        ]
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x
