import numpy as np
import pandas as pd
import pytest


main = pytest.importorskip("main", reason="main module cannot be imported in current environment")


def test_build_integrated_feature_tensors_shapes():
    if not hasattr(main, "build_integrated_feature_tensors"):
        pytest.skip("main.build_integrated_feature_tensors not available")

    rows = 80
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 120, rows),
            "high": np.linspace(101, 121, rows),
            "low": np.linspace(99, 119, rows),
            "close": np.linspace(100, 120, rows) + np.sin(np.linspace(0, 4, rows)),
            "volume": np.linspace(1_000_000, 1_100_000, rows),
        }
    )

    config = {"data": {"sequence_length": 60}, "model": {"input_size": 7}, "device": "cpu"}
    tensors = main.build_integrated_feature_tensors(df, config)

    assert set(tensors.keys()) == {"ml_features", "mt_features", "mr_features"}
    assert tuple(tensors["ml_features"].shape) == (1, 60, 7)
    assert tuple(tensors["mt_features"].shape) == (1, 60, 11)
    assert tuple(tensors["mr_features"].shape) == (1, 60, 11)


def test_build_integrated_feature_tensors_padding_when_allowed():
    if not hasattr(main, "build_integrated_feature_tensors"):
        pytest.skip("main.build_integrated_feature_tensors not available")

    rows = 5
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 101, rows),
            "high": np.linspace(101, 102, rows),
            "low": np.linspace(99, 100, rows),
            "close": np.linspace(100, 101, rows),
            "volume": np.linspace(1_000_000, 1_000_100, rows),
        }
    )

    config = {"data": {"sequence_length": 10}, "model": {"input_size": 7}, "device": "cpu"}
    tensors = main.build_integrated_feature_tensors(df, config, allow_pad=True)

    assert tuple(tensors["ml_features"].shape) == (1, 10, 7)
    assert tuple(tensors["mt_features"].shape) == (1, 10, 11)
    assert tuple(tensors["mr_features"].shape) == (1, 10, 11)
# — Raynergy-svg —
