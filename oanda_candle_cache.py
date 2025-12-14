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


def update_oanda_cache(
    config_path: str,
    *,
    instrument: str,
    granularity: str,
    candles: int = 1500,
    price: str = "MBA",
) -> Path:
    """Fetch new candles from OANDA and update the local cache CSV.

    - If cache exists: fetch from the latest cached timestamp (inclusive) and merge/dedupe.
    - If not: fetch the most recent `candles` and write.

    Returns the cache CSV path.
    """
    cfg = load_config(config_path)
    out_path = cache_csv_path(cfg, instrument=instrument, granularity=granularity)

    existing: Optional[pd.DataFrame] = None
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
        except Exception:
            existing = None

    from_time: Optional[str] = None
    if existing is not None and len(existing) > 0 and "time" in existing.columns:
        try:
            existing_norm = _normalize_time(existing)
            if len(existing_norm) > 0:
                from_time = _to_utc_iso_z(pd.to_datetime(existing_norm["time"].max(), utc=True))
        except Exception:
            from_time = None

    from oanda_practice import OandaPracticeClient
    from fx_paper import candles_to_ohlcv_df

    client = OandaPracticeClient.from_env()
    payload = client.get_candles(
        instrument,
        granularity=granularity,
        count=int(candles),
        price=price,
        from_time=from_time,
    )

    incoming = candles_to_ohlcv_df(payload)
    # Ensure the cache includes time as RFC3339 strings (stable CSV)
    incoming = _normalize_time(incoming)
    incoming["time"] = incoming["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # Avoid ambiguous DataFrame truthiness: use explicit None check
    merged = merge_dedupe_by_time(existing if existing is not None else pd.DataFrame(), incoming)
    merged["time"] = pd.to_datetime(merged["time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    merged.to_csv(out_path, index=False)
    return out_path
