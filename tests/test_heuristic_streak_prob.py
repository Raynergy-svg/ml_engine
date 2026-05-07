"""Real-disk tests for the streak_prob heuristic bridge.

NO MOCKS — real numpy/pandas; deterministic synthetic vol series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scanner.heuristics.streak_prob import compute_streak_prob


def test_shape_and_range_on_synthetic_vol() -> None:
    rng = np.random.default_rng(7)
    n = 500
    v20 = pd.Series(np.abs(rng.normal(0.001, 0.0003, n)) + 1e-5, name="volatility_20")
    v10 = pd.Series(v20 * (0.6 + rng.normal(0, 0.3, n)), name="volatility_10").clip(lower=1e-5)

    out = compute_streak_prob(v10, v20)
    assert isinstance(out, pd.Series)
    assert len(out) == n
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_formula_exact_clip_bounds() -> None:
    # Hand-computed cases against modular_data_loaders.py:3239-3245 formula:
    #   prob = clip((vol_10/vol_20 - 0.8) / 0.4, 0, 1)
    # ratio=0.5 → (0.5-0.8)/0.4 = -0.75 → clip to 0
    # ratio=0.8 → 0
    # ratio=1.0 → (1.0-0.8)/0.4 = 0.5
    # ratio=1.2 → (1.2-0.8)/0.4 = 1.0
    # ratio=1.5 → 1.75 → clip to 1
    v20 = pd.Series([0.001, 0.001, 0.001, 0.001, 0.001])
    v10 = pd.Series([0.0005, 0.0008, 0.001, 0.0012, 0.0015])
    out = compute_streak_prob(v10, v20)
    expected = [0.0, 0.0, 0.5, 1.0, 1.0]
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-9)


def test_div_by_zero_falls_back_to_half() -> None:
    # When vol_20 ≈ 0, fallback to 0.5 per modular_data_loaders.py:3243.
    v20 = pd.Series([0.0, 1e-12, 0.001, 0.0])
    v10 = pd.Series([0.001, 0.001, 0.001, 1e-12])
    out = compute_streak_prob(v10, v20)
    # First two: vol_20 below tol → 0.5
    assert out.iloc[0] == 0.5
    assert out.iloc[1] == 0.5
    # Third: ratio = 1.0 → 0.5
    assert out.iloc[2] == pytest.approx(0.5)
    # Fourth: vol_20=0 → 0.5
    assert out.iloc[3] == 0.5


def test_index_preserved() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    v10 = pd.Series([0.001] * 10, index=idx, name="volatility_10")
    v20 = pd.Series([0.0012] * 10, index=idx, name="volatility_20")
    out = compute_streak_prob(v10, v20)
    assert out.index.equals(idx)
    assert out.name == "streak_prob"


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must have equal length"):
        compute_streak_prob(pd.Series([0.001, 0.002]), pd.Series([0.001, 0.002, 0.003]))


def test_input_must_be_series() -> None:
    with pytest.raises(TypeError, match="must be pd.Series"):
        compute_streak_prob(np.array([0.001]), pd.Series([0.001]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be pd.Series"):
        compute_streak_prob(pd.Series([0.001]), np.array([0.001]))  # type: ignore[arg-type]


def test_nan_input_uses_fallback() -> None:
    v10 = pd.Series([0.001, np.nan, 0.001])
    v20 = pd.Series([0.001, 0.001, np.nan])
    out = compute_streak_prob(v10, v20)
    # NaN inputs should produce NaN ratio → fallback to 0.5
    assert out.iloc[1] == 0.5
    assert out.iloc[2] == 0.5
