"""Verify iTransformer encoder produces identical output shapes to the default transformer."""

from __future__ import annotations

import numpy as np
import pytest

from src.sota_core.raw_sequence_model import ModelConfig, RawSequenceModel


def _build_and_infer(encoder_type: str):
    cfg = ModelConfig(
        seq_len=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        encoder_type=encoder_type,
    )
    m = RawSequenceModel(cfg)
    model = m.build()
    dummy = np.ones((2, 32, 5), dtype=np.float32)
    out = model(dummy)
    return out, model


class TestEncoderVariants:
    def test_transformer_compiles_and_runs(self):
        out, model = _build_and_infer("transformer")
        assert len(out) == 2  # direction, regime
        assert out[0].shape == (2, 1)
        assert out[1].shape == (2, 4)
        assert model is not None

    def test_itransformer_compiles_and_runs(self):
        out, model = _build_and_infer("itransformer")
        assert len(out) == 2
        assert out[0].shape == (2, 1)
        assert out[1].shape == (2, 4)
        assert model is not None

