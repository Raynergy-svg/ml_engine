"""Unit tests for `compute_realized_volatility_regime_labels` (B1 leak fix).

NO MOCKS. All tests use real numpy/pandas DataFrames and a real
LGBMClassifier for the leak-detection probe.

Coverage matrix:
    1. Forward-looking property — label[i] depends only on bars > i.
    2. Leak-detection probe — tiny LGBM on (X = TCN features, y = labels)
       must score macro-F1 < 0.6 on synthetic data.
    3. Train-slice cut-fitting — cuts match np.quantile(realized_vol[
       train_idx & finite], [0.25, 0.60, 0.85]).
    4. NaN tail — last vol_horizon_bars rows have label = -1 sentinel.
    5. Class balance — across 5000-row random-walk H1 DataFrame, all 4
       classes have >= 10% representation under default cuts.
    6. Edge cases — empty df, len(df) <= horizon, missing 'close'.
    7. Bad input — n_classes != 4, mismatched cut_quantiles, horizon < 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.labels.realized_volatility_regime_label import (
    NAN_SENTINEL,
    compute_realized_volatility_regime_labels,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data builders. NO MOCKS.
# ---------------------------------------------------------------------------

def _build_random_walk_df(
    n: int,
    seed: int = 42,
    base_price: float = 1.10,
    daily_vol: float = 0.0015,
) -> pd.DataFrame:
    """Build a deterministic H1-bar DataFrame with full OHLCV + features.

    The features mirror the TCN's X-feature list at
    `modular_data_loaders.py:3580-3595` so the leak-detection probe
    operates on the actual production feature set.
    """
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, daily_vol, size=n)
    log_close = np.cumsum(log_rets) + np.log(base_price)
    close = np.exp(log_close)
    # Build OHL around close with small intra-bar noise.
    intrabar = rng.normal(0.0, daily_vol * 0.3, size=n)
    high = close * np.exp(np.abs(intrabar))
    low = close * np.exp(-np.abs(intrabar))
    open_ = np.concatenate([[base_price], close[:-1]])
    volume = rng.lognormal(mean=10.0, sigma=0.4, size=n).astype(np.float64)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    # Indicators that match TCN's volatility_features list.
    rets = pd.Series(close).pct_change().fillna(0.0)
    df["returns_1"] = rets.values
    df["returns_5"] = rets.rolling(5).mean().fillna(0.0).values
    df["returns_10"] = rets.rolling(10).mean().fillna(0.0).values

    for w in (5, 10, 20):
        df[f"volatility_{w}"] = rets.rolling(w).std().fillna(0.0).values

    # ATR family — true range / close.
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.concatenate([[close[0]], close[:-1]])),
        np.abs(low - np.concatenate([[close[0]], close[:-1]])),
    ])
    for w in (5, 10, 14, 20):
        df[f"atr_pct_{w}"] = (
            pd.Series(tr).rolling(w).mean() / pd.Series(close)
        ).fillna(0.0).values

    # Bollinger width.
    sma20 = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    df["bb_width_20"] = ((4 * std20) / sma20).fillna(0.0).values
    df["bb_width_norm"] = df["bb_width_20"]
    df["high_low_range"] = ((high - low) / close)

    df["volume_ratio_10"] = (
        pd.Series(volume) / pd.Series(volume).rolling(10).mean()
    ).fillna(1.0).values
    df["volume_ratio_20"] = (
        pd.Series(volume) / pd.Series(volume).rolling(20).mean()
    ).fillna(1.0).values

    # ADX placeholder (synthetic; real value not critical for leak probe).
    df["adx"] = np.abs(rets.rolling(14).mean()).fillna(0.0).values * 100.0

    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# 1. Forward-looking property
# ---------------------------------------------------------------------------

def test_label_depends_only_on_future_bars():
    """Modifying close[i] must NOT change label[i]; modifying close[i+5]
    (within the forward window) MUST change label[i] for at least one i.
    """
    n = 500
    horizon = 24
    df = _build_random_walk_df(n, seed=7)

    labels_a, _ = compute_realized_volatility_regime_labels(
        df, vol_horizon_bars=horizon
    )

    # Modify a row INSIDE the forward window of row i=100.
    target_i = 100
    df_mod = df.copy()
    # Bump close[i+5] by 5% — should perturb forward-realized vol for row i.
    df_mod.loc[target_i + 5, "close"] *= 1.05

    labels_b, _ = compute_realized_volatility_regime_labels(
        df_mod, vol_horizon_bars=horizon
    )

    # Labels at row i (and nearby rows whose forward window includes i+5)
    # may change. Labels far in the past (e.g., i=10) should not change.
    assert labels_a[target_i] != labels_b[target_i] or any(
        labels_a[max(0, target_i - 5):target_i + 1]
        != labels_b[max(0, target_i - 5):target_i + 1]
    ), (
        "Modifying a forward-window bar must perturb at least one nearby "
        "label — otherwise the label is not forward-looking."
    )

    # Now verify the contrapositive: modifying close[i] WITHOUT touching
    # bars after i must not change label[i] (the label is computed from
    # close[i+1 .. i+1+horizon] only).
    df_mod2 = df.copy()
    # Shift close[i] only — but this also shifts bars before/at i,
    # which the label does not depend on.
    df_mod2.loc[target_i, "close"] *= 1.10
    labels_c, _ = compute_realized_volatility_regime_labels(
        df_mod2, vol_horizon_bars=horizon
    )
    assert labels_a[target_i] == labels_c[target_i], (
        "Modifying close[i] alone changed label[i] — leak!"
    )


# ---------------------------------------------------------------------------
# 2. Leak-detection probe — tiny LGBM macro-F1 < 0.6
# ---------------------------------------------------------------------------

def test_leak_detection_probe_macro_f1_below_threshold():
    """Train a tiny LGBMClassifier on (X = TCN feature columns,
    y = realized labels) on synthetic data. If macro-F1 >= 0.6, there's
    a residual leak — the test fails.
    """
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score

    n = 5000
    horizon = 24
    df = _build_random_walk_df(n, seed=11)

    feature_cols = [
        "atr_pct_5", "atr_pct_10", "atr_pct_14", "atr_pct_20",
        "volatility_5", "volatility_10", "volatility_20",
        "bb_width_norm", "bb_width_20",
        "high_low_range",
        "returns_1", "returns_5", "returns_10",
        "volume_ratio_10", "volume_ratio_20",
        "adx",
    ]
    X = df[feature_cols].values.astype(np.float32)

    # Use a temporal split (70/30) and fit cuts on train.
    n_train = int(n * 0.7)
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n)
    labels, meta = compute_realized_volatility_regime_labels(
        df, vol_horizon_bars=horizon, train_idx=train_idx, val_idx=val_idx,
    )

    # Drop sentinel rows.
    keep = labels != NAN_SENTINEL
    X = X[keep]
    y = labels[keep].astype(int)
    # Re-derive train/val masks against the kept rows.
    kept_train_mask = keep[:n_train]
    kept_val_mask = keep[n_train:]
    n_train_kept = int(kept_train_mask.sum())
    X_train, X_val = X[:n_train_kept], X[n_train_kept:]
    y_train, y_val = y[:n_train_kept], y[n_train_kept:]

    # Tiny LGBM — exactly the leak-detection probe spec.
    clf = LGBMClassifier(
        n_estimators=20,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.1,
        random_state=0,
        verbose=-1,
        objective="multiclass",
        num_class=4,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

    # Stash the macro-F1 for the test report — pytest captures it via -v.
    print(f"\nLeak-detection probe macro-F1 = {macro_f1:.4f}")

    assert macro_f1 < 0.6, (
        f"Leak-detection probe macro-F1 = {macro_f1:.4f} >= 0.6 — "
        f"the realized-vol label is too predictable from the row-t feature "
        f"vector. There may be a residual leak in the loader's X feature "
        f"set or in the label generator. cut_values={meta['cut_values']}, "
        f"class_balance={meta['class_balance']['all']}"
    )


# ---------------------------------------------------------------------------
# 3. Train-slice cut-fitting matches np.quantile on train-only finite vol
# ---------------------------------------------------------------------------

def test_cuts_fit_on_train_slice_only():
    n = 1000
    horizon = 24
    df = _build_random_walk_df(n, seed=3)

    train_idx = np.arange(0, 700)
    labels, meta = compute_realized_volatility_regime_labels(
        df, vol_horizon_bars=horizon, train_idx=train_idx,
    )

    # Re-derive expected cuts independently from the same forward formula.
    close = df["close"].values
    rv = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon - 0):
        end = i + 1 + horizon
        if end > n:
            break
        win = close[i + 1: end]
        log_rets = np.diff(np.log(win))
        if log_rets.size < 1:
            continue
        sd = float(np.std(log_rets, ddof=0))
        rv[i] = sd / max(abs(np.mean(win)), 1e-12)

    expected_cuts = np.quantile(
        rv[train_idx][np.isfinite(rv[train_idx])], [0.25, 0.60, 0.85]
    )
    got = np.array(meta["cut_values"])
    np.testing.assert_allclose(got, expected_cuts, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. NaN tail — last horizon rows = sentinel
# ---------------------------------------------------------------------------

def test_nan_tail_uses_sentinel():
    n = 200
    horizon = 24
    df = _build_random_walk_df(n, seed=5)
    labels, meta = compute_realized_volatility_regime_labels(
        df, vol_horizon_bars=horizon
    )
    # The last `horizon` rows have insufficient forward window.
    assert (labels[-horizon:] == NAN_SENTINEL).all()
    # Earlier rows should mostly be valid (some may be NaN if log returns
    # were degenerate but on synthetic data this is rare).
    assert (labels[:n - horizon] != NAN_SENTINEL).sum() >= int(
        0.95 * (n - horizon)
    )
    assert meta["n_nan_dropped"] >= horizon


# ---------------------------------------------------------------------------
# 5. Class balance — all 4 classes >= 10% on a 5000-row random walk
# ---------------------------------------------------------------------------

def test_class_balance_all_four_present():
    df = _build_random_walk_df(5000, seed=17)
    labels, meta = compute_realized_volatility_regime_labels(
        df, vol_horizon_bars=24,
    )
    cb = meta["class_balance"]["all"]
    for k in ("0", "1", "2", "3"):
        assert cb[k] >= 0.10, (
            f"Class {k} under-represented at {cb[k]:.3f} (<10%); "
            f"cuts={meta['cut_values']}, full balance={cb}"
        )


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

def test_empty_df_returns_empty_labels():
    df = pd.DataFrame({"close": []})
    labels, meta = compute_realized_volatility_regime_labels(df)
    assert labels.size == 0
    assert meta["n_total"] == 0


def test_short_df_all_sentinel():
    df = pd.DataFrame({"close": np.linspace(1.10, 1.11, 10)})
    labels, meta = compute_realized_volatility_regime_labels(
        df, vol_horizon_bars=24,
    )
    assert (labels == NAN_SENTINEL).all()
    assert meta["n_nan_dropped"] == 10


def test_missing_close_column_returns_all_sentinel():
    df = pd.DataFrame({"open": np.linspace(1.10, 1.20, 100)})
    labels, meta = compute_realized_volatility_regime_labels(df)
    assert (labels == NAN_SENTINEL).all()
    assert meta["n_nan_dropped"] == 100


# ---------------------------------------------------------------------------
# 7. Bad input — strict signature
# ---------------------------------------------------------------------------

def test_n_classes_must_be_four():
    df = _build_random_walk_df(200, seed=2)
    with pytest.raises(ValueError, match="n_classes must be 4"):
        compute_realized_volatility_regime_labels(df, n_classes=3)


def test_cut_quantiles_length_mismatch():
    df = _build_random_walk_df(200, seed=2)
    with pytest.raises(ValueError, match="cut_quantiles must have length"):
        compute_realized_volatility_regime_labels(
            df, cut_quantiles=(0.25, 0.50)
        )


def test_horizon_too_small():
    df = _build_random_walk_df(200, seed=2)
    with pytest.raises(ValueError, match="vol_horizon_bars must be >= 2"):
        compute_realized_volatility_regime_labels(df, vol_horizon_bars=1)


# ---------------------------------------------------------------------------
# 8. Per-split class balance metadata populated when indices supplied
# ---------------------------------------------------------------------------

def test_per_split_class_balance_populated():
    n = 1000
    horizon = 24
    df = _build_random_walk_df(n, seed=8)
    train_idx = np.arange(0, 700)
    val_idx = np.arange(700, 850)
    test_idx = np.arange(850, n)
    _, meta = compute_realized_volatility_regime_labels(
        df,
        vol_horizon_bars=horizon,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    for split in ("train", "val", "test"):
        assert set(meta["class_balance"][split].keys()) == {"0", "1", "2", "3"}
