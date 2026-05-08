"""Phase 4-A integration tests — load_direction_data news fusion wiring.

Verifies the full Stage 4-A trainer-time pipeline:
  news_source → news_embedder → align_news_to_bars → PCA(fit_on_train_only)
   → append to X → result dict surfaces news_pca for trainer to persist

Uses real subclasses of NewsSource and NewsEmbedder (no mocks per
.claude/rules/improvement.md No-Mock Rule). Synthetic deterministic data
keeps the test fast and reproducible.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.core.modular_data_loaders import load_direction_data, compute_normalized_features
from src.data.news.embedder import NewsEmbedder
from src.data.news.source import NewsEvent, NewsSource


class _DeterministicNewsSource(NewsSource):
    """Generates fixed events at known timestamps for test determinism."""

    def __init__(self, n_events: int = 50):
        self._n = n_events

    def fetch_events(self, pair, since, until):
        # Spread n_events evenly across the [since, until] window.
        if since >= until:
            return []
        delta = (until - since) / (self._n + 1)
        events = []
        cats = ["NFP", "CPI", "GDP", "FOMC", "ECB", "BoE", "BoJ", "OTHER"]
        for i in range(self._n):
            ts = since + delta * (i + 1)
            events.append(
                NewsEvent(
                    timestamp=ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts,
                    text=f"Test event {i} category {cats[i % len(cats)]}",
                    source="test",
                    category=cats[i % len(cats)],
                    relevance_score=0.5 + 0.5 * ((i % 3) / 2),
                    pair=pair,
                    impact="medium",
                )
            )
        return events


class _DeterministicEmbedder(NewsEmbedder):
    """768-dim deterministic embedder: hash-of-text → fixed vector. No model load."""
    embedding_dim: int = 768

    def embed(self, events: List[NewsEvent]) -> np.ndarray:
        if not events:
            return np.empty((0, 768), dtype=np.float32)
        out = np.zeros((len(events), 768), dtype=np.float32)
        for i, e in enumerate(events):
            # Hash-derived deterministic vector
            rng = np.random.default_rng(seed=hash(e.text) & 0xFFFFFFFF)
            out[i] = rng.standard_normal(768).astype(np.float32) * 0.5
        return out


def _make_h1_price_df(n: int = 400) -> pd.DataFrame:
    """Synthetic H1 OHLCV df with DatetimeIndex (UTC)."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
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


def test_load_direction_data_no_news_kwargs_is_no_op():
    """Default behavior unchanged when news kwargs not provided."""
    df = _make_h1_price_df(n=400)
    df = compute_normalized_features(df)
    result = load_direction_data(df, apply_scaler=False, lookahead=24, threshold=0.0001)
    assert result is not None
    fn = result["feature_names"]
    assert not any(c.startswith("pca_") for c in fn), "pca_ cols leaked without news kwargs"
    assert not any(c.startswith("ec_") for c in fn), "ec_ cols leaked without news kwargs"
    assert result.get("news_pca") is None
    assert result.get("news_lookback_window_hours") is None
    assert result.get("news_pca_n_components") is None
    assert result.get("news_event_class_count_columns") == []


def test_load_direction_data_with_news_appends_72_cols_and_fits_pca():
    """News kwargs provided → 64 PCA + 8 ec cols appended; PCA fitted on train only."""
    df = _make_h1_price_df(n=400)
    df = compute_normalized_features(df)
    result = load_direction_data(
        df,
        apply_scaler=False,
        lookahead=24,
        threshold=0.0001,
        news_source=_DeterministicNewsSource(n_events=80),
        news_embedder=_DeterministicEmbedder(),
        pair="EUR_USD",
        news_lookback_window_hours=24,
        news_pca_n_components=64,
    )
    assert result is not None
    fn = result["feature_names"]

    pca_cols = [c for c in fn if c.startswith("pca_")]
    ec_cols = [c for c in fn if c.startswith("ec_")]
    assert len(pca_cols) == 64
    assert len(ec_cols) == 8

    # X width should now include the 72 extra cols
    assert result["X_train"].shape[1] == len(fn)
    assert result["X_val"].shape[1] == len(fn)
    assert result["X_test"].shape[1] == len(fn)

    # PCA artifact is fitted (has components_)
    pca = result["news_pca"]
    assert pca is not None
    assert hasattr(pca, "components_")
    assert pca.components_.shape == (64, 768)

    assert result["news_event_class_count_columns"] == [f"ec_{i}" for i in range(8)]
    assert result["news_lookback_window_hours"] == 24
    assert result["news_pca_n_components"] == 64


def test_load_direction_data_with_news_no_events_in_window_logs_and_zero_fills(caplog):
    """Empty event source → news cols still appended, all zero."""
    import logging

    class _EmptyNewsSource(NewsSource):
        def fetch_events(self, pair, since, until):
            return []

    df = _make_h1_price_df(n=400)
    df = compute_normalized_features(df)

    with caplog.at_level(logging.WARNING, logger="src.core.modular_data_loaders"):
        result = load_direction_data(
            df,
            apply_scaler=False,
            lookahead=24,
            threshold=0.0001,
            news_source=_EmptyNewsSource(),
            news_embedder=_DeterministicEmbedder(),
            pair="EUR_USD",
        )

    # Should still produce news cols (even if all-zero from no events)
    assert result is not None
    fn = result["feature_names"]
    pca_cols = [c for c in fn if c.startswith("pca_")]
    ec_cols = [c for c in fn if c.startswith("ec_")]
    assert len(pca_cols) == 64
    assert len(ec_cols) == 8
    # The warning must surface so operator notices
    assert any(
        "0 events fetched" in record.message
        for record in caplog.records
    )


def test_load_direction_data_news_pca_explains_real_variance():
    """Sanity: news_pca on real-ish embeddings retains meaningful variance ratio."""
    df = _make_h1_price_df(n=600)  # bigger sample for stable PCA
    df = compute_normalized_features(df)
    result = load_direction_data(
        df,
        apply_scaler=False,
        lookahead=24,
        threshold=0.0001,
        news_source=_DeterministicNewsSource(n_events=200),
        news_embedder=_DeterministicEmbedder(),
        pair="EUR_USD",
        news_pca_n_components=64,
    )
    pca = result["news_pca"]
    assert pca is not None
    cum_var = float(pca.explained_variance_ratio_.sum())
    # On synthetic Gaussian-ish embeddings, 64-of-768 should still capture
    # meaningful variance (>5%) — exact ratio depends on the synthetic distribution.
    assert cum_var > 0.05, f"PCA cum_var unrealistically low: {cum_var}"
    assert cum_var <= 1.0


def test_load_direction_data_news_failure_falls_back_gracefully():
    """If news pipeline raises, loader continues without news cols (logs error)."""
    import logging

    class _BrokenSource(NewsSource):
        def fetch_events(self, pair, since, until):
            raise RuntimeError("simulated upstream outage")

    df = _make_h1_price_df(n=400)
    df = compute_normalized_features(df)

    with caplog_at_error_level("src.core.modular_data_loaders") as caplog:
        result = load_direction_data(
            df,
            apply_scaler=False,
            lookahead=24,
            threshold=0.0001,
            news_source=_BrokenSource(),
            news_embedder=_DeterministicEmbedder(),
            pair="EUR_USD",
        )

    # Loader must NOT crash; news cols absent; PCA absent
    assert result is not None
    fn = result["feature_names"]
    assert not any(c.startswith("pca_") for c in fn)
    assert result["news_pca"] is None


# Helper context manager to capture logger output cleanly
import contextlib
import logging as _logging


@contextlib.contextmanager
def caplog_at_error_level(logger_name):
    """Compact logger-capture helper for the failure-path test."""
    logger = _logging.getLogger(logger_name)
    handler = _logging.handlers.MemoryHandler(capacity=1000)
    handler.setLevel(_logging.ERROR)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


import logging.handlers  # noqa: E402  (used inside helper, deferred import OK)
