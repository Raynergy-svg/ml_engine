"""No-mock tests for factor signals (US-002 carry, US-003 trend).

Causality + window-invariance in the style of test_feature_window_invariance.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor.signals import (
    tsmom_signal,
    carry_signal,
    combine_signals,
    TSMOM_LOOKBACKS,
)


def _panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n, tz="UTC")
    pairs = ["EUR_USD", "USD_JPY", "GBP_USD"]
    data = {}
    for i, p in enumerate(pairs):
        steps = rng.normal(0.0002 * (i - 1), 0.005, n)
        data[p] = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
    return pd.DataFrame(data)


def test_tsmom_sign_on_monotone_series():
    idx = pd.bdate_range("2014-01-01", periods=400, tz="UTC")
    up = pd.DataFrame({"A": np.linspace(1, 2, 400)}, index=idx)
    down = pd.DataFrame({"A": np.linspace(2, 1, 400)}, index=idx)
    assert tsmom_signal(up).iloc[-1, 0] == 1.0
    assert tsmom_signal(down).iloc[-1, 0] == -1.0


def test_tsmom_window_invariance():
    # The signal on a given date must not depend on how much history precedes it.
    panel = _panel()
    full = tsmom_signal(panel)
    truncated = tsmom_signal(panel.iloc[50:])  # drop early rows
    common = full.index.intersection(truncated.index)
    common = common[common > panel.index[max(TSMOM_LOOKBACKS) + 50]]
    a = full.loc[common]
    b = truncated.loc[common]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_tsmom_is_nan_before_warmup():
    panel = _panel()
    sig = tsmom_signal(panel)
    assert sig.iloc[: max(TSMOM_LOOKBACKS)].isna().all().all()


def test_carry_rank_sign():
    idx = pd.bdate_range("2014-01-01", periods=5, tz="UTC")
    rd = pd.DataFrame(
        {"HI": [3.0] * 5, "MID": [0.0] * 5, "LO": [-3.0] * 5}, index=idx
    )
    sig = carry_signal(rd)
    # Highest differential -> most positive (long); lowest -> most negative.
    assert sig["HI"].iloc[-1] == 1.0
    assert sig["LO"].iloc[-1] == -1.0
    assert abs(sig["MID"].iloc[-1]) < 1e-9


def test_combine_skips_nan_factor():
    idx = pd.bdate_range("2014-01-01", periods=3, tz="UTC")
    a = pd.DataFrame({"X": [1.0, 1.0, 1.0]}, index=idx)
    b = pd.DataFrame({"X": [np.nan, -1.0, np.nan]}, index=idx)
    out = combine_signals([a, b])
    assert out["X"].iloc[0] == 1.0          # b is NaN -> uses a only
    assert abs(out["X"].iloc[1]) < 1e-9     # mean(1, -1) == 0
