"""Phase 4-A integration tests — compute_normalized_features news_features_df join.

Verifies the wire from align_news_to_bars output → compute_normalized_features:
- Optional kwarg with default None preserves backward compat
- Left-outer join on bar_timestamp (DatetimeIndex)
- Bars without aligned news rows zero-filled (matches no-event-bar convention)
- Empty news df treated as no-op
- Non-DatetimeIndex on news df logs error and skips (no crash)

No mocks — real DataFrames, real compute_normalized_features, per
.claude/rules/improvement.md No-Mock Rule.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.core.modular_data_loaders import compute_normalized_features


def _make_price_df(n: int = 200, freq: str = "h") -> pd.DataFrame:
    """Synthetic price df with DatetimeIndex (UTC tz-aware)."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = 1.10 + np.cumsum(rng.normal(0.0, 0.0001, n))
    return pd.DataFrame(
        {
            "time": times,
            "open": close + rng.normal(0.0, 0.0001, n),
            "high": close + np.abs(rng.normal(0.0, 0.0002, n)),
            "low": close - np.abs(rng.normal(0.0, 0.0002, n)),
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=times,
    )


def _make_news_features_df(times: pd.DatetimeIndex, sparsity_step: int = 7) -> pd.DataFrame:
    """Synthetic news_features_df: pca_0..pca_63 + ec_0..ec_7 at every Nth bar."""
    rng = np.random.default_rng(7)
    news_idx = times[::sparsity_step]
    cols = [f"pca_{i}" for i in range(64)] + [f"ec_{i}" for i in range(8)]
    return pd.DataFrame(
        rng.normal(0.0, 1.0, (len(news_idx), 72)),
        index=news_idx,
        columns=cols,
    )


def test_compute_normalized_features_news_df_none_is_no_op():
    """Default news_features_df=None must preserve backward compat — no news cols added."""
    df = _make_price_df()
    out = compute_normalized_features(df.copy(), news_features_df=None)
    for prefix in ("pca_", "ec_"):
        assert not any(c.startswith(prefix) for c in out.columns), (
            f"No-op default leaked {prefix} cols"
        )


def test_compute_normalized_features_news_df_left_outer_join_shape():
    """News df present → 72 new cols added; row count preserved."""
    df = _make_price_df(n=200)
    news_df = _make_news_features_df(df.index)
    out = compute_normalized_features(df.copy(), news_features_df=news_df)
    assert "pca_0" in out.columns
    assert "pca_63" in out.columns
    assert "ec_0" in out.columns
    assert "ec_7" in out.columns
    assert len(out) == 200, "Row count must match price df"


def test_compute_normalized_features_news_df_zero_fills_unmatched_bars():
    """Bars without aligned news must be zero-filled — matches no-event-bar convention."""
    df = _make_price_df(n=200)
    news_df = _make_news_features_df(df.index, sparsity_step=10)
    out = compute_normalized_features(df.copy(), news_features_df=news_df)

    # First bar IS in news (sparsity_step=10 → indices 0, 10, 20, ...)
    news_bar = df.index[0]
    expected = news_df.loc[news_bar, "pca_0"]
    assert abs(out.loc[news_bar, "pca_0"] - expected) < 1e-9

    # Second bar is NOT in news → zero
    non_news_bar = df.index[1]
    assert out.loc[non_news_bar, "pca_0"] == 0.0
    assert out.loc[non_news_bar, "ec_3"] == 0.0


def test_compute_normalized_features_empty_news_df_is_no_op():
    """Empty news df must be treated as no-op (defensive)."""
    df = _make_price_df()
    out = compute_normalized_features(df.copy(), news_features_df=pd.DataFrame())
    assert not any(c.startswith("pca_") for c in out.columns)


def test_compute_normalized_features_non_datetime_news_index_skips_with_log(caplog):
    """News df with non-DatetimeIndex must log error and skip the join (not crash)."""
    df = _make_price_df()
    bad_news_df = pd.DataFrame(
        np.zeros((10, 72)),
        index=range(10),  # RangeIndex, not DatetimeIndex
        columns=[f"pca_{i}" for i in range(64)] + [f"ec_{i}" for i in range(8)],
    )
    with caplog.at_level(logging.ERROR, logger="src.core.modular_data_loaders"):
        out = compute_normalized_features(df.copy(), news_features_df=bad_news_df)
    # Should not crash; should not have added news cols
    assert "pca_0" not in out.columns
    # Should have logged the contract violation
    assert any(
        "DatetimeIndex" in record.message
        for record in caplog.records
    )


def test_compute_normalized_features_news_join_does_not_break_existing_features():
    """Joining news must NOT remove any pre-existing feature columns."""
    df = _make_price_df()
    out_no_news = compute_normalized_features(df.copy(), news_features_df=None)
    pre_existing_cols = set(out_no_news.columns)

    news_df = _make_news_features_df(df.index)
    out_with_news = compute_normalized_features(df.copy(), news_features_df=news_df)
    new_cols = set(out_with_news.columns)

    # Every pre-existing column must still be present
    missing = pre_existing_cols - new_cols
    assert not missing, f"News join removed pre-existing columns: {missing}"
    # And exactly 72 new ones
    added = new_cols - pre_existing_cols
    assert len(added) == 72, f"Expected 72 new news cols, got {len(added)}"
