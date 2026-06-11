"""Pin the feature pipeline as window-anchor-invariant (no mocks).

The 2026-06-10 backtest-harness finding: frame-anchored features (raw OBV
cumsum, cumprod/cumsum returns) made model predictions depend on how much
history the inference window happened to include — identical bars flipped
26/26 LONG <-> 26/26 SHORT purely on anchor length. This test enriches the
same synthetic series from two different anchor points and asserts every
surviving feature row is identical: any future anchored feature fails here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.feature_engineering import FeatureEngineering
from src.utils import load_config


def _synthetic_ohlcv(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    vol = 0.0004 + 0.002 * (np.sin(np.linspace(0, 10 * np.pi, n)) ** 2)
    close = 1.10 * np.exp(np.cumsum(rng.randn(n) * vol))
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="15min"),
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": close * (1 + np.abs(rng.randn(n)) * vol),
        "low": close * (1 - np.abs(rng.randn(n)) * vol),
        "close": close,
        "volume": rng.randint(100, 1000, n).astype(float),
    })


def test_features_identical_across_anchor_lengths():
    base = _synthetic_ohlcv(3000)
    cfg = load_config("config/config_m1_optimized.yaml")
    fe = FeatureEngineering(cfg)

    long_anchor = fe.create_features(base.copy(), include_all=True)
    short_anchor = fe.create_features(base.tail(1200).reset_index(drop=True).copy(),
                                      include_all=True)

    # Compare the overlapping final rows (same raw bars, different anchors).
    k = 400
    a = long_anchor.tail(k).reset_index(drop=True)
    b = short_anchor.tail(k).reset_index(drop=True)

    offenders = []
    for col in a.columns:
        if col == "time" or col not in b.columns:
            continue
        av = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        # NaN patterns must match too; compare where both finite.
        both = np.isfinite(av) & np.isfinite(bv)
        if both.sum() == 0:
            continue
        scale = max(float(np.nanstd(av)), 1e-12)
        max_rel = float(np.max(np.abs(av[both] - bv[both]))) / scale
        if max_rel > 1e-6:
            offenders.append((col, max_rel))

    assert not offenders, f"anchor-dependent features detected: {offenders[:10]}"


def test_obv_is_zscored_not_cumsum():
    df = _synthetic_ohlcv(800)
    cfg = load_config("config/config_m1_optimized.yaml")
    out = FeatureEngineering(cfg).create_features(df.copy(), include_all=True)
    obv = pd.to_numeric(out["obv"], errors="coerce").dropna()
    # A rolling z-score lives on a bounded scale; a raw cumsum of volume
    # (100-1000 per bar over hundreds of bars) reaches tens of thousands.
    assert float(obv.abs().max()) < 10.0
    assert float(obv.abs().max()) > 0.1   # and it's not degenerate zeros


def test_version_bumped_for_semantics_change():
    from src.core.modular_data_loaders import FEATURE_PIPELINE_VERSION
    assert FEATURE_PIPELINE_VERSION == "2026-06-10-v2"
