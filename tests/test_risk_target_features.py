"""Tests for `src/training/risk_target_features.py`.

NO MOCKS. Real pandas/numpy only.

Coverage:
    1. Window-invariance canary (L-001) — mirrors
       `tests/test_feature_window_invariance.py`: computing features from
       two different history anchors must produce IDENTICAL values on the
       overlapping tail rows. A feature that depends on where the analysis
       window starts (cumulative-from-start) fails here.
    2. Causality — a feature at row i is unchanged by mutating any row > i
       (backward-looking only, never reads the future).
    3. Warm-up NaN pattern — first 59 rows (60-bar longest window) have at
       least one NaN feature; row 60+ are fully populated on clean data.
    4. `build_pooled_feature_frame` — per-pair rolling windows never blend
       across a pair boundary; pooled frame sorted by (date, pair).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.risk_target_features import (
    FEATURE_COLUMNS,
    build_pooled_feature_frame,
    compute_risk_target_features,
)


def _synthetic_ohlcv(n: int, seed: int = 7, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = 0.004 + 0.01 * (np.sin(np.linspace(0, 8 * np.pi, n)) ** 2)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol))
    high = close * (1 + np.abs(rng.normal(0, 1, n)) * vol)
    low = close * (1 - np.abs(rng.normal(0, 1, n)) * vol)
    openp = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D", tz="UTC"),
        "open": openp, "high": high, "low": low, "close": close, "volume": volume,
    })


def test_window_invariance_across_history_anchors():
    base = _synthetic_ohlcv(2000)
    long_anchor = compute_risk_target_features(base.copy())
    short_anchor = compute_risk_target_features(base.tail(500).reset_index(drop=True).copy())

    k = 200  # compare the final k overlapping rows
    a = long_anchor.tail(k).reset_index(drop=True)
    b = short_anchor.tail(k).reset_index(drop=True)

    offenders = []
    for col in FEATURE_COLUMNS:
        av = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        both_finite = np.isfinite(av) & np.isfinite(bv)
        if both_finite.sum() == 0:
            continue
        scale = max(float(np.nanstd(av[both_finite])), 1e-12)
        max_rel = float(np.max(np.abs(av[both_finite] - bv[both_finite]))) / scale
        if max_rel > 1e-6:
            offenders.append((col, max_rel))

    assert not offenders, f"window-anchored (leaky) features detected: {offenders}"


def test_causal_only_no_future_leak():
    base = _synthetic_ohlcv(300)
    feat_a = compute_risk_target_features(base.copy())

    mutated = base.copy()
    # Mutate every row AFTER row 100 — row 100's own features must be unchanged.
    mutated.loc[101:, ["open", "high", "low", "close", "volume"]] *= 3.0
    feat_b = compute_risk_target_features(mutated)

    for col in FEATURE_COLUMNS:
        a_val = feat_a.loc[100, col]
        b_val = feat_b.loc[100, col]
        if pd.isna(a_val) and pd.isna(b_val):
            continue
        assert np.isclose(float(a_val), float(b_val), rtol=1e-9), (
            f"{col} at row 100 changed when future rows were mutated (leak)"
        )


def test_warmup_nan_pattern():
    base = _synthetic_ohlcv(200)
    feat = compute_risk_target_features(base)
    # Longest warm-up: realized_vol_60 needs 60 log-returns, and the first
    # log-return (row 0) is NaN — so row 60 (0-indexed) is the first
    # fully-populated row.
    assert feat.loc[:59, FEATURE_COLUMNS].isna().any(axis=1).all()
    assert not feat.loc[60:, FEATURE_COLUMNS].isna().any(axis=1).any()


def test_missing_column_raises():
    import pytest
    bad = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "close": np.ones(10)})
    with pytest.raises(ValueError):
        compute_risk_target_features(bad)


def test_pooled_frame_no_cross_pair_contamination():
    pair_a = _synthetic_ohlcv(150, seed=1)
    pair_b = _synthetic_ohlcv(150, seed=2)

    pooled, cols = build_pooled_feature_frame({"PAIR_A": pair_a, "PAIR_B": pair_b})

    # Recompute PAIR_A alone and confirm its rows in the pooled frame match
    # exactly — pooling must not let PAIR_B's history leak into PAIR_A's
    # rolling windows.
    solo_a = compute_risk_target_features(pair_a)
    pooled_a = pooled[pooled["pair"] == "PAIR_A"].sort_values("date").reset_index(drop=True)

    for col in cols:
        av = pd.to_numeric(solo_a[col], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(pooled_a[col], errors="coerce").to_numpy(dtype=float)
        both_finite = np.isfinite(av) & np.isfinite(bv)
        assert np.allclose(av[both_finite], bv[both_finite], rtol=1e-9)

    assert set(pooled["pair"].unique()) == {"PAIR_A", "PAIR_B"}
    assert pooled["date"].is_monotonic_increasing or pooled.sort_values(["date", "pair"]).equals(pooled)
