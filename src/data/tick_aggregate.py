"""Aggregate captured tick parquet into OHLCV candles.

Reads partitioned tick data under trained_data/ticks/{pair}/YYYY/MM/DD.parquet
and resamples to arbitrary granularity (S5, M1, M15, H1, etc.).

Design:
- Parquet-native: reads only the partitions overlapping the query range.
- Vectorised: groupby + agg, no Python loops over bars.
- Volume model: OANDA quote ticks have no trade volume.  We proxy volume
  by counting ticks per bar and scaling by average liquidity.

Usage:
    df_m15 = aggregate_ticks_to_candles("EUR_USD", "M15", start="2026-06-01")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TICK_ROOT = Path("trained_data/ticks")

_GRAN_TO_FREQ = {
    "S5": "5s",
    "S10": "10s",
    "S15": "15s",
    "S30": "30s",
    "M1": "1min",
    "M2": "2min",
    "M5": "5min",
    "M10": "10min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H2": "2h",
    "H4": "4h",
    "D": "1d",
}


def aggregate_ticks_to_candles(
    instrument: str,
    granularity: str = "M15",
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    root: Path = TICK_ROOT,
) -> pd.DataFrame:
    """Load ticks for a pair and resample to OHLCV candles.

    Args:
        instrument: e.g. "EUR_USD".
        granularity: One of S5, S10, S15, S30, M1, M2, M5, M10, M15,
                     M30, H1, H2, H4, D.
        start: ISO date lower bound (inclusive).  e.g. "2026-06-01".
        end:   ISO date upper bound (exclusive).  e.g. "2026-06-16".
        root:  Root of tick parquet partitions.

    Returns:
        DataFrame indexed by UTC timestamp with columns
        [open, high, low, close, volume, tick_count, avg_spread].
    """
    freq = _GRAN_TO_FREQ.get(granularity)
    if freq is None:
        raise ValueError(f"Unsupported granularity: {granularity}")

    df = _load_ticks(instrument, start, end, root)
    if df is None or df.empty:
        logger.warning(f"No tick data found for {instrument}")
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "tick_count", "avg_spread"]
        )

    # Ensure sorted
    df = df.sort_index()

    # Use mid price for OHLC; bid/ask available for spread calculation
    ohlc = df["mid"].resample(freq).agg({"open": "first", "high": "max", "low": "min", "close": "last"})

    # Volume proxy: count of ticks per bar weighted by average liquidity
    # If liq data missing, fall back to raw tick count
    if "bid_liq" in df.columns and "ask_liq" in df.columns:
        liq = (df["bid_liq"] + df["ask_liq"]).clip(lower=1.0)
        volume = liq.resample(freq).sum()
    else:
        volume = df["mid"].resample(freq).count()

    tick_count = df["mid"].resample(freq).count()

    # Average spread per bar (in price units, not pips)
    spread = (df["ask"] - df["bid"]).abs()
    avg_spread = spread.resample(freq).mean()

    merged = pd.concat([ohlc, volume.rename("volume"), tick_count.rename("tick_count"), avg_spread.rename("avg_spread")], axis=1)
    merged = merged.dropna(subset=["open"])
    return merged


def _load_ticks(
    instrument: str,
    start: Optional[str],
    end: Optional[str],
    root: Path,
) -> Optional[pd.DataFrame]:
    """Load tick parquet files for the given date range."""
    inst_dir = root / instrument
    if not inst_dir.exists():
        return None

    # Collect candidate partition files
    files = sorted(inst_dir.rglob("*.parquet"))
    if not files:
        return None

    # Date-range filter based on filepath (YYYY/MM/DD.parquet)
    dfs = []
    for f in files:
        try:
            parts = f.relative_to(inst_dir).parts
            if len(parts) != 3:
                continue
            yr, mo_file, day_file = parts
            mo = day_file.split(".")[0]  # "01.parquet" → "01"
            # Actually path is inst_dir/YYYY/MM/DD.parquet
            # So parts = ("YYYY", "MM", "DD.parquet")
            yr_str = str(yr)
            mo_str = str(mo_file)
            day_str = day_file.replace(".parquet", "")
            file_date = f"{yr_str}-{mo_str}-{day_str}"

            if start and file_date < start[:10]:
                continue
            if end and file_date >= end[:10]:
                continue
        except Exception:
            continue

        try:
            df = pd.read_parquet(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.debug(f"Skipping unreadable parquet {f}: {e}")
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs)
    # Ensure datetime index
    if "time" in combined.columns:
        combined = combined.set_index("time")
    combined.index = pd.to_datetime(combined.index, utc=True)
    combined = combined.sort_index()

    # Optional in-memory time filtering (more precise than file-based)
    if start:
        combined = combined[combined.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        combined = combined[combined.index < pd.Timestamp(end, tz="UTC")]

    return combined
