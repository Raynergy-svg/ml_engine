"""Dukascopy tick data harvester.

Downloads free historical tick data from Dukascopy Bank:
  https://www.dukascopy.com/swiss/english/marketwatch/historical/

Data format:
  URL: https://www.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
  Compression: LZ4 (bi5 = binary + lz4)
  Tick record: 20 bytes
    - time_delta : uint32 BE (ms offset from hour start)
    - ask        : uint32 BE (price * 100_000)
    - bid        : uint32 BE (price * 100_000)
    - ask_volume : uint32 BE (units, often 1_000_000)
    - bid_volume : uint32 BE

Usage:
    from src.data.sources.dukascopy_harvester import DukascopyHarvester
    h = DukascopyHarvester("EUR_USD")
    df = h.download_hour(2023, 1, 1, 0)  # Jan 1, 00:00 UTC
    h.to_parquet(df, "trained_data/dukascopy/EUR_USD_2023_01_01.parquet")
"""

from __future__ import annotations

import gzip
import io
import logging
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Dukascopy base URL template
_DUKASCOPY_URL = "https://www.dukascopy.com/datafeed/{pair}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# Retry config
_MAX_RETRIES = 3
_TIMEOUT = 30.0
_REQUEST_DELAY = 0.5  # polite delay between requests

# LZ4 availability (optional graceful fallback)
try:
    import lz4.frame
    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False
    logger.warning("lz4 not installed; Dukascopy bi5 decompression unavailable")


def _parse_bi5(data: bytes) -> np.ndarray:
    """Decompress LZ4 and parse 20-byte tick records into structured array.

    Returns ndarray with fields:
        time_delta, ask_raw, bid_raw, ask_volume, bid_volume
    """
    if not _HAS_LZ4:
        raise RuntimeError("lz4 is required for bi5 decompression. Install: pip install lz4")

    decompressed = lz4.frame.decompress(data)
    n_ticks = len(decompressed) // 20
    if n_ticks == 0:
        return np.array([], dtype=np.uint32)

    # Unpack as uint32 big-endian, reshape to (n_ticks, 5)
    fmt = ">" + "I" * 5 * n_ticks
    unpacked = struct.unpack(fmt, decompressed[: n_ticks * 20])
    arr = np.array(unpacked, dtype=np.uint32).reshape(n_ticks, 5)
    return arr


class DukascopyHarvester:
    """Harvest tick data from Dukascopy free historical feed."""

    def __init__(
        self,
        pair: str,
        *,
        price_scale: float = 100_000.0,
        volume_scale: float = 1_000_000.0,
        save_dir: Path = Path("trained_data/dukascopy"),
    ) -> None:
        self.pair = str(pair).upper().replace("/", "")
        self.price_scale = price_scale
        self.volume_scale = volume_scale
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "ml_engine-dukascopy-harvester/1.0",
            "Accept-Encoding": "gzip, deflate",
        })

    def _build_url(self, year: int, month: int, day: int, hour: int) -> str:
        return _DUKASCOPY_URL.format(
            pair=self.pair, year=year, month=month, day=day, hour=hour,
        )

    def download_hour(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        *,
        retries: int = _MAX_RETRIES,
    ) -> Optional[pd.DataFrame]:
        """Download and parse one hour of tick data.

        Returns DataFrame with columns:
            timestamp (UTC), ask, bid, mid, ask_volume, bid_volume
        Returns None if file not found or empty.
        """
        if not _HAS_LZ4:
            logger.error("Cannot download Dukascopy data: lz4 not installed")
            return None

        url = self._build_url(year, month, day, hour)
        for attempt in range(1, retries + 1):
            try:
                resp = self._session.get(url, timeout=_TIMEOUT)
                if resp.status_code == 404:
                    logger.debug("Dukascopy 404: %s", url)
                    return None
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logger.warning("Dukascopy fetch attempt %d/%d failed: %s", attempt, retries, e)
                import time
                time.sleep(1.0 * attempt)
        else:
            logger.error("Dukascopy fetch failed after %d retries: %s", retries, url)
            return None

        try:
            raw = resp.content
            if not raw:
                return None

            arr = _parse_bi5(raw)
            if len(arr) == 0:
                return None

            # Build timestamp from year/month/day/hour + time_delta
            hour_start = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
            deltas_ms = arr[:, 0].astype(np.int64)
            timestamps = pd.to_datetime(
                [hour_start.timestamp() + d / 1000.0 for d in deltas_ms],
                unit="s",
                utc=True,
            )

            df = pd.DataFrame({
                "timestamp": timestamps,
                "ask": arr[:, 1].astype(np.float64) / self.price_scale,
                "bid": arr[:, 2].astype(np.float64) / self.price_scale,
                "ask_volume": arr[:, 3].astype(np.float64) / self.volume_scale,
                "bid_volume": arr[:, 4].astype(np.float64) / self.volume_scale,
            })
            df["mid"] = (df["ask"] + df["bid"]) / 2.0
            df["spread"] = df["ask"] - df["bid"]
            return df.set_index("timestamp").sort_index()
        except Exception as e:
            logger.warning("Dukascopy parse failed for %s: %s", url, e)
            return None

    def download_day(
        self,
        year: int,
        month: int,
        day: int,
    ) -> Optional[pd.DataFrame]:
        """Download all 24 hours for a single day and concatenate."""
        dfs = []
        for hour in range(24):
            df = self.download_hour(year, month, day, hour)
            if df is not None and not df.empty:
                dfs.append(df)
            import time
            time.sleep(_REQUEST_DELAY)
        if not dfs:
            return None
        return pd.concat(dfs).sort_index()

    def download_range(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Download tick data for a date range.

        Downloads day-by-day. Forex is closed Sat 22:00 UTC to Sun 22:00 UTC,
        so weekend days will return empty (expected).
        """
        dfs = []
        current = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        end_utc = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end

        while current <= end_utc:
            df = self.download_day(current.year, current.month, current.day)
            if df is not None and not df.empty:
                dfs.append(df)
                logger.info(
                    "Dukascopy %s %s: %d ticks",
                    self.pair, current.strftime("%Y-%m-%d"), len(df),
                )
            else:
                logger.debug("Dukascopy %s %s: no data", self.pair, current.strftime("%Y-%m-%d"))
            current += pd.Timedelta(days=1)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs).sort_index()

    def to_parquet(self, df: pd.DataFrame, path: Path | str) -> None:
        """Save tick DataFrame to parquet with zstd compression."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="zstd")
        logger.info("Saved %d ticks to %s", len(df), path)


def convert_dukascopy_to_harvest_format(
    pair: str,
    dukascopy_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Convert raw Dukascopy ticks to canonical OHLCV parquet (M15 granularity).

    This bridges Dukascopy data into the existing HarvestScheduler pipeline.
    """
    if dukascopy_df.empty:
        return
    ohlcv = dukascopy_df["mid"].resample("15min").agg(
        open="first", high="max", low="min", close="last"
    )
    ohlcv["volume"] = dukascopy_df["ask_volume"].resample("15min").sum().fillna(0)
    ohlcv = ohlcv.dropna()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(output_path, compression="zstd")
    logger.info("Converted %d M15 bars for %s → %s", len(ohlcv), pair, output_path)
