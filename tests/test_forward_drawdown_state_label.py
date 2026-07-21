"""Unit tests for `compute_forward_drawdown_state_labels` (new greenfield
drawdown-state target — 2026-07-08 risk-target ML redirect).

NO MOCKS. Real numpy/pandas + a real LGBMClassifier for the leak-detection
probe, mirroring `test_realized_volatility_regime_label.py`'s convention.

Coverage:
    1. Forward-looking property — label[i] depends only on bars > i.
    2. Leak-detection probe — a tiny LGBM on (X=trailing features,
       y=labels) must score macro-F1 < 0.6 on synthetic random-walk data.
    3. Train-slice cut-fitting — cut matches
       np.quantile(max_dd[train_idx & finite], stressed_quantile).
    4. A KNOWN synthetic crash produces a "stressed" label at the correct
       row and "calm" elsewhere.
    5. NaN tail — last horizon rows get DRAWDOWN_NAN_SENTINEL.
    6. Edge cases — empty df, len(df) <= horizon, missing 'close'.
    7. Bad input — stressed_quantile out of (0,1), horizon < 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.labels.forward_drawdown_state_label import (
    DRAWDOWN_NAN_SENTINEL,
    compute_forward_drawdown_state_labels,
)


def _random_walk_close(n: int, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.008, size=n)
    return np.exp(np.cumsum(log_rets) + np.log(1.10))


def test_label_depends_only_on_future_bars():
    n = 300
    horizon = 15
    close = _random_walk_close(n)
    df = pd.DataFrame({"close": close})
    labels, _ = compute_forward_drawdown_state_labels(df, horizon)

    # Mutate a bar before row 60's forward window — label[60] unchanged.
    df2 = df.copy()
    df2.loc[10, "close"] *= 0.5
    labels2, _ = compute_forward_drawdown_state_labels(df2, horizon)
    assert labels[60] == labels2[60]

    # Crash INSIDE row 60's forward window [61, 76) — max_dd[60] must rise
    # enough to flip calm->stressed for at least a plausible crash size.
    df3 = df.copy()
    df3.loc[65, "close"] *= 0.5  # 50% single-bar crash inside the window
    labels3, meta3 = compute_forward_drawdown_state_labels(df3, horizon)
    assert labels3[60] == 1  # stressed


def test_leak_detection_probe_macro_f1_below_threshold():
    pytest.importorskip("lightgbm")
    from sklearn.metrics import f1_score
    from lightgbm import LGBMClassifier

    n = 4000
    horizon = 20
    close = _random_walk_close(n, seed=21)
    df = pd.DataFrame({"close": close})

    # Build a feature set from data available AT row i only (trailing,
    # never forward) — a leak-free label must not be reconstructable from
    # these even though they're volatility-flavored features.
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    feat = pd.DataFrame({
        "trailing_vol_20": pd.Series(log_ret).rolling(20, min_periods=20).std(),
        "trailing_ret_5": pd.Series(log_ret).rolling(5, min_periods=5).mean(),
        "trailing_range_14": pd.Series(close).rolling(14, min_periods=14).apply(
            lambda w: (w.max() - w.min()) / w.mean()
        ),
    })

    labels, _ = compute_forward_drawdown_state_labels(df, horizon)
    valid = (labels != DRAWDOWN_NAN_SENTINEL) & feat.notna().all(axis=1).values
    X = feat.loc[valid].values
    y = labels[valid]

    split = int(len(X) * 0.7)
    clf = LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1, random_state=0)
    clf.fit(X[:split], y[:split])
    pred = clf.predict(X[split:])
    macro_f1 = f1_score(y[split:], pred, average="macro")
    assert macro_f1 < 0.6, f"leak suspected: macro-F1={macro_f1:.3f} on synthetic random walk"


def test_cut_fit_on_train_slice_only():
    n = 1000
    horizon = 20
    close = _random_walk_close(n, seed=42)
    df = pd.DataFrame({"close": close})

    train_idx = np.arange(0, 700)
    labels, meta = compute_forward_drawdown_state_labels(
        df, horizon, train_idx=train_idx, stressed_quantile=0.75,
    )

    # Recompute max_dd independently and confirm the cut matches
    # np.quantile of the train-slice's finite max_dd values.
    from src.training.labels.forward_drawdown_state_label import _compute_forward_max_drawdown
    max_dd = _compute_forward_max_drawdown(close, horizon)
    train_finite = max_dd[train_idx][np.isfinite(max_dd[train_idx])]
    expected_cut = float(np.quantile(train_finite, 0.75))
    assert np.isclose(meta["cut_value"], expected_cut)


def test_known_crash_produces_stressed_label():
    n = 200
    horizon = 10
    close = np.full(n, 1.0)
    # A clean 30% drawdown starting right after row 50.
    close[51:61] = np.linspace(1.0, 0.70, 10)
    close[61:] = 0.70
    df = pd.DataFrame({"close": close})

    labels, meta = compute_forward_drawdown_state_labels(df, horizon, stressed_quantile=0.5)
    assert labels[50] == 1
    # A row far from any drawdown (deep in the flat tail before the crash,
    # with a flat forward window) should be calm.
    assert labels[20] == 0


def test_nan_tail_uses_sentinel():
    n = 100
    horizon = 15
    close = _random_walk_close(n)
    df = pd.DataFrame({"close": close})
    labels, meta = compute_forward_drawdown_state_labels(df, horizon)
    assert np.all(labels[-horizon:] == DRAWDOWN_NAN_SENTINEL)
    assert np.all(labels[:-horizon] != DRAWDOWN_NAN_SENTINEL)
    assert meta["n_nan_dropped"] == horizon


def test_empty_df():
    df = pd.DataFrame({"close": []})
    labels, meta = compute_forward_drawdown_state_labels(df, 10)
    assert len(labels) == 0


def test_short_df_all_sentinel():
    df = pd.DataFrame({"close": _random_walk_close(5)})
    labels, meta = compute_forward_drawdown_state_labels(df, 10)
    assert np.all(labels == DRAWDOWN_NAN_SENTINEL)


def test_missing_close_column():
    df = pd.DataFrame({"open": _random_walk_close(50)})
    labels, meta = compute_forward_drawdown_state_labels(df, 10)
    assert np.all(labels == DRAWDOWN_NAN_SENTINEL)


def test_stressed_quantile_out_of_range_raises():
    df = pd.DataFrame({"close": _random_walk_close(50)})
    with pytest.raises(ValueError):
        compute_forward_drawdown_state_labels(df, 10, stressed_quantile=1.5)
    with pytest.raises(ValueError):
        compute_forward_drawdown_state_labels(df, 10, stressed_quantile=0.0)


def test_horizon_too_small_raises():
    df = pd.DataFrame({"close": _random_walk_close(50)})
    with pytest.raises(ValueError):
        compute_forward_drawdown_state_labels(df, 0)
