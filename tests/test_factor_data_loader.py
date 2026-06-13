"""No-mock tests for the factor daily data loader (US-001).

Network paths are exercised by the real backtest run; here we test the cache
round-trip and validation logic against real disk and real pandas frames — no
test doubles, no mocked OANDA client (project no-mock rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factor.data_loader import (
    FactorDataError,
    _candles_to_frame,
    load_or_fetch,
    validate_history,
)


def _synth_ohlc(n: int, start: str = "2014-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    base = pd.Series(np.linspace(1.0, 1.2, n), index=idx)
    return pd.DataFrame(
        {
            "open": base.values,
            "high": base.values * 1.001,
            "low": base.values * 0.999,
            "close": base.values,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_candles_to_frame_skips_incomplete():
    candles = [
        {"time": "2014-01-01T00:00:00Z", "complete": True,
         "mid": {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.05"}, "volume": 10},
        {"time": "2014-01-02T00:00:00Z", "complete": False,
         "mid": {"o": "1.05", "h": "1.2", "l": "1.0", "c": "1.1"}, "volume": 5},
    ]
    df = _candles_to_frame(candles)
    assert len(df) == 1  # the forming (incomplete) bar must be dropped
    assert df.index.tz is not None


def test_validate_history_rejects_short():
    df = _synth_ohlc(100)
    with pytest.raises(FactorDataError):
        validate_history(df, "EUR_USD")


def test_validate_history_accepts_long_clean():
    df = _synth_ohlc(2700)
    validate_history(df, "EUR_USD")  # should not raise


def test_validate_history_flags_nans():
    df = _synth_ohlc(2700)
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    with pytest.raises(FactorDataError):
        validate_history(df, "EUR_USD")


def test_load_or_fetch_reads_existing_cache(tmp_path):
    # Write a real CSV cache; load_or_fetch must read it without any network.
    df = _synth_ohlc(2700)
    (tmp_path).mkdir(exist_ok=True)
    df.to_csv(tmp_path / "EUR_USD_D.csv", index=True)
    out = load_or_fetch("EUR_USD", cache_dir=tmp_path, validate=True)
    assert len(out) == 2700
    assert list(out.columns)[:4] == ["open", "high", "low", "close"]
    assert out.index.tz is not None
