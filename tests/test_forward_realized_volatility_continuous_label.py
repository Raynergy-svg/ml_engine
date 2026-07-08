"""Unit tests for `compute_forward_realized_volatility` (continuous regression
counterpart to the binned `compute_realized_volatility_regime_labels`).

NO MOCKS. Real numpy/pandas only.

Coverage:
    1. Forward-looking property — value[i] depends only on bars > i.
    2. Numerical consistency with the binned label's underlying formula.
    3. Annualization factor applied correctly.
    4. NaN tail — last vol_horizon_bars rows are NaN.
    5. Edge cases — empty df, len(df) <= horizon, missing 'close'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.labels.realized_volatility_regime_label import (
    compute_forward_realized_volatility,
    compute_realized_volatility_regime_labels,
)


def _random_walk_close(n: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.01, size=n)
    return np.exp(np.cumsum(log_rets) + np.log(1.10))


def test_value_depends_only_on_future_bars():
    n = 300
    horizon = 10
    close = _random_walk_close(n)
    df = pd.DataFrame({"close": close})

    values, _ = compute_forward_realized_volatility(df, horizon)

    # Mutate a bar STRICTLY BEFORE the forward window of row i=50 (i.e. a
    # bar at or before i itself) — value[i] must be unchanged.
    df2 = df.copy()
    df2.loc[10, "close"] = df2.loc[10, "close"] * 5.0  # far before row 50's forward window
    values2, _ = compute_forward_realized_volatility(df2, horizon)
    assert np.isclose(values[50], values2[50])

    # Mutate a bar STRICTLY INSIDE row 50's forward window [51, 61) — value[50] must change.
    df3 = df.copy()
    df3.loc[55, "close"] = df3.loc[55, "close"] * 5.0
    values3, _ = compute_forward_realized_volatility(df3, horizon)
    assert not np.isclose(values[50], values3[50])


def test_consistent_with_binned_label_cut_boundaries():
    """The continuous value, when binned with the SAME percentile cuts the
    classification label fits internally, must reproduce the same class
    assignment — proves both share the same underlying formula."""
    n = 2000
    horizon = 24
    close = _random_walk_close(n, seed=3)
    df = pd.DataFrame({"close": close})

    cont_values, _ = compute_forward_realized_volatility(df, horizon)
    binned_labels, meta = compute_realized_volatility_regime_labels(
        df, vol_horizon_bars=horizon, n_classes=4,
    )

    finite = np.isfinite(cont_values)
    cuts = np.asarray(meta["cut_values"])
    rebinned = np.clip(np.searchsorted(cuts, cont_values[finite], side="right"), 0, 3)
    assert np.array_equal(rebinned, binned_labels[finite])


def test_annualization_factor_scales_linearly():
    n = 300
    horizon = 10
    close = _random_walk_close(n, seed=5)
    df = pd.DataFrame({"close": close})

    raw, _ = compute_forward_realized_volatility(df, horizon)
    annualized, meta = compute_forward_realized_volatility(df, horizon, annualization_factor=16.0)

    finite = np.isfinite(raw)
    assert np.allclose(annualized[finite], raw[finite] * 16.0)
    assert meta["annualization_factor"] == 16.0


def test_nan_tail():
    n = 100
    horizon = 20
    close = _random_walk_close(n)
    df = pd.DataFrame({"close": close})
    values, meta = compute_forward_realized_volatility(df, horizon)
    assert np.all(np.isnan(values[-horizon:]))
    assert np.all(np.isfinite(values[:-horizon]))
    assert meta["n_nan_dropped"] == horizon


def test_empty_df():
    df = pd.DataFrame({"close": []})
    values, meta = compute_forward_realized_volatility(df, 10)
    assert len(values) == 0
    assert meta["n_total"] == 0


def test_short_df_all_nan():
    df = pd.DataFrame({"close": _random_walk_close(5)})
    values, meta = compute_forward_realized_volatility(df, 10)
    assert np.all(np.isnan(values))


def test_missing_close_column():
    df = pd.DataFrame({"open": _random_walk_close(50)})
    values, meta = compute_forward_realized_volatility(df, 10)
    assert np.all(np.isnan(values))
    assert meta["n_nan_dropped"] == 50


def test_horizon_too_small_raises():
    df = pd.DataFrame({"close": _random_walk_close(50)})
    with pytest.raises(ValueError):
        compute_forward_realized_volatility(df, 1)
