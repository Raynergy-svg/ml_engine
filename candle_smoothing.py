"""OHLCV candle smoothing utilities.

Implements a simple, deterministic preprocessing step used across the project:
- resample OHLCV to 5-minute bars
- keep raw `close` and add an EMA(14) feature column

This is intentionally lightweight and dependency-free (pandas only).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


_CORE_OHLCV = ("open", "high", "low", "close")


def resample_5min_ohlcv_and_ema_close(
    df: pd.DataFrame,
    *,
    ema_span: int = 14,
    median_window: int | None = None,
    time_col: str = "time",
) -> pd.DataFrame:
    """Return a copy resampled to 5-minute OHLCV and add an EMA-smoothed close feature.

    - If `time_col` exists, it will be parsed as UTC timestamps.
    - Resampling aggregates:
      open=first, high=max, low=min, close=last, volume=sum (if present).
      Optional bid/ask closes are aggregated as last.
    - Output keeps a string `time` column (RFC3339 Z).

    The returned frame keeps the raw resampled `close` and adds:
    - `close_ema_{ema_span}`: EMA(ema_span) computed causally from past-to-present.

    If `median_window` is provided, a causal rolling median (no centering) is applied
    before the EMA calculation (this avoids look-ahead from centered windows).

    If required columns/time parsing are not available, returns the original df copy.
    """

    if df is None or len(df) == 0:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()

    # Locate time column case-insensitively.
    time_col_actual: Optional[str] = None
    for c in out.columns:
        if str(c).strip().lower() == str(time_col).strip().lower():
            time_col_actual = c
            break

    if time_col_actual is None:
        return out

    missing = [c for c in _CORE_OHLCV if c not in [str(x).lower() for x in out.columns]]
    # If columns are not normalized, try a lowercase rename first.
    if missing:
        lower_map = {c: str(c).strip().lower() for c in out.columns}
        out = out.rename(columns=lower_map)
        # update time col after rename
        time_col_actual = str(time_col).strip().lower()

    for c in _CORE_OHLCV:
        if c not in out.columns:
            return df.copy()

    dt = pd.to_datetime(out[time_col_actual], utc=True, errors="coerce")
    if dt.isna().all():
        return df.copy()

    out = out.copy()
    out[time_col_actual] = dt
    out = out.sort_values(time_col_actual)
    out = out.dropna(subset=[time_col_actual])
    out = out.set_index(time_col_actual)

    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in out.columns:
        agg["volume"] = "sum"
    if "bid_close" in out.columns:
        agg["bid_close"] = "last"
    if "ask_close" in out.columns:
        agg["ask_close"] = "last"

    resampled = out.resample("5min").agg(agg)

    # Drop incomplete bars.
    resampled = resampled.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])

    if len(resampled) == 0:
        return df.copy()

    ema_span = int(ema_span)
    if ema_span <= 0:
        ema_span = 14

    close_raw = resampled["close"].astype(float)
    close_for_ema = close_raw

    if median_window is not None:
        try:
            mw = int(median_window)
            if mw > 1:
                # Causal median filter (no centering -> no future leakage)
                close_for_ema = close_for_ema.rolling(window=mw, min_periods=1, center=False).median()
        except Exception:
            pass

    resampled[f"close_ema_{ema_span}"] = close_for_ema.ewm(span=ema_span, adjust=False).mean()

    resampled = resampled.reset_index().rename(columns={time_col_actual: "time"})
    # Stable RFC3339 Z strings.
    resampled["time"] = pd.to_datetime(resampled["time"], utc=True).dt.tz_convert("UTC").dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    # Trim trailing .000000Z when not needed (keeps tests nice).
    resampled["time"] = resampled["time"].str.replace(r"\.000000Z$", "Z", regex=True)

    return resampled
