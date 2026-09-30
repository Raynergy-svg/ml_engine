"""Verify iTransformer encoder produces identical output shapes to the default transformer."""

from __future__ import annotations

import numpy as np
import pytest

from src.sota_core.raw_sequence_model import ModelConfig, RawSequenceModel


def _build_and_infer(encoder_type: str, *, direction_classes: int = 2, direction_horizons: tuple[int, ...] = (5,)):
    cfg = ModelConfig(
        seq_len=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        encoder_type=encoder_type,
        direction_classes=direction_classes,
        direction_horizons=direction_horizons,
    )
    m = RawSequenceModel(cfg)
    model = m.build()
    dummy = np.ones((2, 32, 5), dtype=np.float32)
    out = model(dummy, training=False)
    if encoder_type == "itransformer":
        assert model.get_layer("itransformer") is not None
    return out, model


class TestEncoderVariants:
    def test_transformer_compiles_and_runs(self):
        out, model = _build_and_infer("transformer")
        assert isinstance(out, dict)
        assert set(out) == {"direction_5", "regime"}
        assert out["direction_5"].shape == (2, 1)
        assert out["regime"].shape == (2, 4)
        assert model is not None

    def test_itransformer_compiles_and_runs(self):
        out, model = _build_and_infer("itransformer")
        assert isinstance(out, dict)
        assert set(out) == {"direction_5", "regime"}
        assert out["direction_5"].shape == (2, 1)
        assert out["regime"].shape == (2, 4)
        assert model is not None


@pytest.mark.parametrize("encoder_type", ["transformer", "itransformer"])
@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("horizons", [(3,), (5, 10)])
def test_named_output_shapes_and_probabilities_match_configured_heads(encoder_type, classes, horizons):
    out, _ = _build_and_infer(encoder_type, direction_classes=classes, direction_horizons=horizons)
    expected_shapes = {f"direction_{h}": (2, 3 if classes == 3 else 1) for h in horizons}
    expected_shapes["regime"] = (2, 4)
    assert isinstance(out, dict)
    assert set(out) == set(expected_shapes)
    for name, shape in expected_shapes.items():
        values = np.asarray(out[name])
        assert values.shape == shape
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
        assert np.all(values >= 0) and np.all(values <= 1)
        if shape[-1] > 1:
            np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-4, rtol=0)
