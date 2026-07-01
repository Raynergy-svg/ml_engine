"""No-mock tests for the equity daily data loader (US-001).

Project no-mock rule: real classes, real disk via ``tmp_path``. The IBKR and
yfinance fetch paths are exercised against the live broker in higher tiers; here
we test the cache round-trip, source tagging, and fail-loud guards.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.equity import EQUITY_PIPELINE_VERSION
from src.equity.data_loader import (
    ALLOWED_SOURCES,
    EquityCacheMeta,
    EquityDataError,
    _normalize_frame,
    _write_cache,
    load_or_fetch,
    read_cache_meta,
    validate_freshness,
    validate_history,
)


def _synth_equity(n: int, end: str = "2026-06-17") -> pd.DataFrame:
    end_ts = pd.Timestamp(end, tz="UTC")
    idx = pd.bdate_range(end=end_ts, periods=n, tz="UTC")
    base = pd.Series(np.linspace(100.0, 150.0, n), index=idx)
    return pd.DataFrame(
        {
            "open": base.values,
            "high": base.values * 1.005,
            "low": base.values * 0.995,
            "close": base.values,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_normalize_frame_lowercases_and_utc_indexes():
    raw = _synth_equity(50)
    raw.columns = ["Open", "High", "Low", "Close", "Volume"]
    raw.index = raw.index.tz_localize(None)
    out = _normalize_frame(raw, "AAPL")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is not None
    assert out.index.is_monotonic_increasing


def test_normalize_frame_rejects_missing_ohlc():
    df = _synth_equity(10).drop(columns=["high"])
    with pytest.raises(EquityDataError):
        _normalize_frame(df, "AAPL")


def test_validate_history_rejects_short():
    df = _synth_equity(100)
    with pytest.raises(EquityDataError):
        validate_history(df, "AAPL")


def test_validate_history_accepts_long_clean():
    df = _synth_equity(2_600)
    validate_history(df, "AAPL")  # must not raise


def test_validate_history_flags_nan_close():
    df = _synth_equity(2_600)
    df.iloc[10, df.columns.get_loc("close")] = np.nan
    with pytest.raises(EquityDataError, match="NaNs in OHLC"):
        validate_history(df, "AAPL")


def test_validate_freshness_passes_when_recent():
    df = _synth_equity(50)
    now = df.index.max().to_pydatetime() + timedelta(days=2)
    validate_freshness(df, "AAPL", freshness_days=7, now=now)


def test_validate_freshness_rejects_stale():
    df = _synth_equity(50)
    now = df.index.max().to_pydatetime() + timedelta(days=30)
    with pytest.raises(EquityDataError, match="older|old"):
        validate_freshness(df, "AAPL", freshness_days=7, now=now)


def test_cache_round_trip_records_source(tmp_path):
    df = _synth_equity(2_600)
    meta = _write_cache(df, "AAPL", source="ibkr", cache_dir=tmp_path)
    assert (tmp_path / "AAPL_D.csv").exists()
    assert (tmp_path / "AAPL_D.meta.json").exists()
    assert meta.source == "ibkr"
    assert meta.rows == 2_600
    assert meta.pipeline_version == EQUITY_PIPELINE_VERSION

    out = load_or_fetch(
        "AAPL",
        cache_dir=tmp_path,
        validate=True,
        freshness_days=365 * 100,  # disable freshness for the synth fixture
    )
    assert len(out) == 2_600
    assert list(out.columns)[:4] == ["open", "high", "low", "close"]
    assert out.index.tz is not None

    sidecar = read_cache_meta("AAPL", cache_dir=tmp_path)
    assert sidecar.source in ALLOWED_SOURCES
    assert sidecar.source == "ibkr"


def test_load_or_fetch_raises_on_stale_cache(tmp_path):
    # Build a cache whose last row is 30 days old vs. a "now" the test pins.
    end = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end_ts = pd.Timestamp(end)
    idx = pd.bdate_range(end=end_ts, periods=2_600, tz="UTC")
    base = pd.Series(np.linspace(100.0, 150.0, 2_600), index=idx)
    df = pd.DataFrame(
        {
            "open": base.values,
            "high": base.values * 1.005,
            "low": base.values * 0.995,
            "close": base.values,
            "volume": np.full(2_600, 1_000_000.0),
        },
        index=idx,
    )
    _write_cache(df, "MSFT", source="ibkr", cache_dir=tmp_path)

    pinned_now = datetime(2026, 6, 18, tzinfo=timezone.utc)
    with pytest.raises(EquityDataError, match="old|stale|older"):
        load_or_fetch(
            "MSFT",
            cache_dir=tmp_path,
            validate=True,
            freshness_days=7,
            now=pinned_now,
        )


def test_load_or_fetch_raises_on_nan_cache(tmp_path):
    df = _synth_equity(2_600)
    df.iloc[42, df.columns.get_loc("close")] = np.nan
    _write_cache(df, "QQQ", source="yfinance", cache_dir=tmp_path)
    with pytest.raises(EquityDataError, match="NaN"):
        load_or_fetch(
            "QQQ",
            cache_dir=tmp_path,
            validate=True,
            freshness_days=365 * 100,
        )


def test_load_or_fetch_rejects_unknown_source(tmp_path):
    with pytest.raises(EquityDataError, match="unknown source"):
        load_or_fetch("AAPL", source="alpaca", cache_dir=tmp_path)


def test_cache_meta_schema_round_trip(tmp_path):
    df = _synth_equity(2_600)
    written = _write_cache(df, "SPY", source="yfinance", cache_dir=tmp_path)
    loaded = read_cache_meta("SPY", cache_dir=tmp_path)
    assert isinstance(loaded, EquityCacheMeta)
    assert loaded.source == written.source == "yfinance"
    assert loaded.ticker == "SPY"
    assert loaded.rows == 2_600


def test_atomic_write_leaves_no_tmp(tmp_path):
    df = _synth_equity(50)
    _write_cache(df, "IWM", source="ibkr", cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.tmp"))
