"""Local OANDA candle caching utilities.

Goal: keep a growing, deduplicated OHLCV CSV per instrument/granularity so
training sessions can fetch only new candles and then resume training.

This module is intentionally practice-only via `OandaPracticeClient`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from utils import load_config


@dataclass(frozen=True)
class OandaCacheSpec:
    instrument: str
    granularity: str
    price: str = "MBA"
    initial_count: int = 5000
    update_count: int = 1500


def _data_dir_from_cfg(cfg: Dict[str, Any]) -> Path:
    # Prefer top-level path aliases; fall back to paths.*
    return Path(cfg.get("DATA_DIR") or cfg.get("paths", {}).get("data_dir") or "trained_data/data")


def cache_csv_path(cfg: Dict[str, Any], *, instrument: str, granularity: str) -> Path:
    data_dir = _data_dir_from_cfg(cfg)
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_instrument = str(instrument).replace("/", "_").replace(" ", "").strip()
    safe_gran = str(granularity).replace("/", "_").replace(" ", "").strip()
    return data_dir / f"oanda_{safe_instrument}_{safe_gran}.csv"


def _to_utc_iso_z(ts: pd.Timestamp) -> str:
    # OANDA expects RFC3339; we emit a Z-suffixed UTC timestamp.
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time" not in out.columns:
        raise ValueError("Expected a 'time' column")
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).reset_index(drop=True)
    return out


def merge_dedupe_by_time(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge two OHLCV frames and dedupe by `time` (keep last)."""
    if existing is None or len(existing) == 0:
        out = _normalize_time(incoming)
    elif incoming is None or len(incoming) == 0:
        out = _normalize_time(existing)
    else:
        a = _normalize_time(existing)
        b = _normalize_time(incoming)
        out = pd.concat([a, b], ignore_index=True)

    out = out.sort_values("time", kind="mergesort")
    out = out.drop_duplicates(subset=["time"], keep="last")
    out = out.reset_index(drop=True)
    return out


# OANDA API maximum candles per request
OANDA_MAX_CANDLES_PER_REQUEST = 5000


def _load_existing_cache(out_path: Path) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load existing cache and extract the latest timestamp for incremental fetch."""
    existing: Optional[pd.DataFrame] = None
    from_time: Optional[str] = None

    if not out_path.exists():
        return existing, from_time

    try:
        existing = pd.read_csv(out_path)
    except Exception:
        return None, None

    if existing is None or len(existing) == 0 or "time" not in existing.columns:
        return existing, None

    try:
        existing_norm = _normalize_time(existing)
        if len(existing_norm) > 0:
            from_time = _to_utc_iso_z(pd.to_datetime(existing_norm["time"].max(), utc=True))
    except Exception:
        from_time = None

    return existing, from_time


def _fetch_candle_batch(
    client: Any,
    instrument: str,
    granularity: str,
    batch_size: int,
    price: str,
    from_time: Optional[str],
    to_time: Optional[str],
    is_first_batch: bool,
) -> Optional[pd.DataFrame]:
    """Fetch a single batch of candles from OANDA."""
    from fx_paper import candles_to_ohlcv_df

    payload = client.get_candles(
        instrument,
        granularity=granularity,
        count=batch_size,
        price=price,
        from_time=from_time if is_first_batch else None,
        to_time=to_time,
    )

    batch_df = candles_to_ohlcv_df(payload)
    if batch_df is None or len(batch_df) == 0:
        return None

    return _normalize_time(batch_df)


def _fetch_paginated_candles(
    client: Any,
    instrument: str,
    granularity: str,
    candles: int,
    price: str,
    from_time: Optional[str],
) -> pd.DataFrame:
    """Fetch candles with pagination support for large requests."""
    all_incoming_frames: list[pd.DataFrame] = []
    remaining = int(candles)
    current_to_time: Optional[str] = None

    while remaining > 0:
        batch_size = min(remaining, OANDA_MAX_CANDLES_PER_REQUEST)
        is_first_batch = len(all_incoming_frames) == 0

        batch_df = _fetch_candle_batch(
            client, instrument, granularity, batch_size, price,
            from_time, current_to_time, is_first_batch
        )

        if batch_df is None or len(batch_df) == 0:
            break

        all_incoming_frames.append(batch_df)
        remaining -= len(batch_df)

        # Set to_time to earliest timestamp for backward pagination
        if remaining > 0 and len(batch_df) > 0:
            current_to_time = _to_utc_iso_z(batch_df["time"].min())
        else:
            break

        # If we got fewer than requested, we've hit the end
        if len(batch_df) < batch_size:
            break

    return _combine_candle_frames(all_incoming_frames)


def _combine_candle_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple candle DataFrames into one deduplicated frame."""
    if not frames:
        return pd.DataFrame()

    incoming = pd.concat(frames, ignore_index=True)
    incoming = _normalize_time(incoming)
    incoming = incoming.drop_duplicates(subset=["time"], keep="last")
    return incoming.sort_values("time").reset_index(drop=True)


def _save_merged_cache(
    existing: Optional[pd.DataFrame],
    incoming: pd.DataFrame,
    out_path: Path,
) -> None:
    """Merge incoming candles with existing cache and save to CSV."""
    # Format time as RFC3339 strings for stable CSV
    if len(incoming) > 0:
        incoming["time"] = incoming["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    existing_df = existing if existing is not None else pd.DataFrame()
    merged = merge_dedupe_by_time(existing_df, incoming)
    merged["time"] = pd.to_datetime(merged["time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    merged.to_csv(out_path, index=False)


def update_oanda_cache(
    config_path: str,
    *,
    instrument: str,
    granularity: str,
    candles: int = 5000,
    price: str = "MBA",
) -> Path:
    """Fetch new candles from OANDA and update the local cache CSV.

    - If cache exists: fetch from the latest cached timestamp (inclusive) and merge/dedupe.
    - If not: fetch the most recent `candles` and write.
    - Supports pagination for requests exceeding OANDA's 5000 candle limit.

    Returns the cache CSV path.
    """
    from oanda_practice import OandaPracticeClient

    cfg = load_config(config_path)
    out_path = cache_csv_path(cfg, instrument=instrument, granularity=granularity)

    existing, from_time = _load_existing_cache(out_path)

    client = OandaPracticeClient.from_env()
    incoming = _fetch_paginated_candles(client, instrument, granularity, candles, price, from_time)

    _save_merged_cache(existing, incoming, out_path)
    return out_path
