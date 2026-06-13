"""US-001 — Daily OHLC data layer for the FX factor portfolio.

Fetches OANDA daily ('D') mid candles per pair, caches them to
``market_data/factor/{PAIR}_D.csv``, and validates history depth + gaps.

Design rules (project contracts):
- Causal/honest: only COMPLETE candles are kept (OANDA marks the forming candle
  ``complete: false``); a half-formed last bar would leak the future.
- Atomic writes: cache is written to ``.tmp`` then ``os.replace``d.
- Fail-loud: short history or unreadable cache raises a *named* error, never a
  silent empty frame.
- No mocks in tests: the loader round-trips a real tmp_path cache.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import PAIRS

logger = logging.getLogger(__name__)

# ~10 years of business days. OANDA returns ~3,230 daily rows from 2014 today.
MIN_ROWS = 2_600
MAX_GAP_BUSINESS_DAYS = 5
DEFAULT_START = "2014-01-01T00:00:00Z"
DEFAULT_CACHE_DIR = Path("market_data/factor")


class FactorDataError(RuntimeError):
    """Raised when daily data is missing, too short, or structurally invalid."""


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=True)
    os.replace(tmp, path)


def _candles_to_frame(candles: List[dict]) -> pd.DataFrame:
    """Convert OANDA candle dicts to a UTC-indexed OHLCV frame (complete only)."""
    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue  # never include the still-forming bar — that is lookahead
        mid = c.get("mid") or {}
        try:
            rows.append(
                {
                    "date": pd.Timestamp(c["time"]).tz_convert("UTC").normalize(),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": float(c.get("volume", 0) or 0),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FactorDataError(f"Malformed OANDA candle {c!r}: {exc}") from exc
    if not rows:
        raise FactorDataError("No complete candles in OANDA response")
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_daily_candles(
    instrument: str,
    *,
    start: str = DEFAULT_START,
    client: Optional[object] = None,
) -> pd.DataFrame:
    """Fetch complete daily mid candles for one instrument from OANDA practice.

    A single ``count=5000`` request covers >12 years of dailies, so no
    pagination loop is needed today; we still pass ``from`` for reproducibility.
    """
    if client is None:
        from src.utils.oanda_practice import OandaPracticeClient

        client = OandaPracticeClient.from_env()

    payload = client.get_candles(
        instrument,
        granularity="D",
        count=5000,
        price="M",
        from_time=start,
    )
    candles = (payload or {}).get("candles") or []
    if not candles:
        raise FactorDataError(f"{instrument}: OANDA returned zero candles")
    return _candles_to_frame(candles)


def validate_history(
    df: pd.DataFrame,
    instrument: str,
    *,
    min_rows: int = MIN_ROWS,
    max_gap_business_days: int = MAX_GAP_BUSINESS_DAYS,
) -> None:
    """Hard-fail on short history; loudly warn on suspicious calendar gaps."""
    if len(df) < min_rows:
        raise FactorDataError(
            f"{instrument}: only {len(df)} daily rows (< {min_rows} ≈ 10y). "
            "Refusing to build factors on insufficient history."
        )
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise FactorDataError(f"{instrument}: missing column '{col}'")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise FactorDataError(f"{instrument}: NaNs in OHLC after load")

    # Gap detection in business days (weekends are expected and excluded).
    idx = df.index
    bdays = pd.Series(idx).diff().dt.days.fillna(0)
    # A normal Fri->Mon gap is 3 calendar days; flag stretches that imply
    # > max_gap_business_days missing sessions.
    suspicious = bdays[bdays > (max_gap_business_days + 2)]
    if len(suspicious) > 0:
        first = idx[suspicious.index[0]]
        logger.warning(
            "%s: %d calendar gap(s) > %d business days; first near %s",
            instrument,
            len(suspicious),
            max_gap_business_days,
            first.date(),
        )


def load_or_fetch(
    instrument: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    client: Optional[object] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Return daily OHLCV for ``instrument``, using the CSV cache when present."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{instrument}_D.csv"

    if path.exists() and not refresh:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.DatetimeIndex(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        except Exception as exc:  # corrupt cache -> refetch, never crash
            logger.warning("%s: cache unreadable (%s); refetching", instrument, exc)
            df = fetch_daily_candles(instrument, client=client)
            _atomic_write_csv(df, path)
    else:
        df = fetch_daily_candles(instrument, client=client)
        _atomic_write_csv(df, path)

    if validate:
        validate_history(df, instrument)
    return df


def load_panel(
    pairs: Optional[List[str]] = None,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    client: Optional[object] = None,
    field: str = "close",
) -> pd.DataFrame:
    """Load one field (default close) for all pairs into a date×pair frame.

    The returned frame is inner-joined on the common trading calendar so factor
    math sees aligned dates. Per-pair OHLCV frames remain available via
    ``load_or_fetch`` for the backtester (which needs open prices for fills).
    """
    pairs = pairs or PAIRS
    series: Dict[str, pd.Series] = {}
    for p in pairs:
        df = load_or_fetch(p, cache_dir=cache_dir, refresh=refresh, client=client)
        series[p] = df[field].rename(p)
    panel = pd.concat(series.values(), axis=1, join="inner").sort_index()
    if panel.empty:
        raise FactorDataError("Empty panel after aligning pairs on common dates")
    return panel


def load_universe(
    pairs: List[str],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    client: Optional[object] = None,
    min_rows: int = MIN_ROWS,
):
    """Load close+open for a universe, DROPPING instruments without enough history.

    Returns (close_panel, open_panel, dropped) where ``dropped`` is a list of
    (instrument, reason). Validation failures don't crash the run — they exclude
    the instrument with a named warning (FP-1 done-criterion). Aligns survivors
    on their common trading calendar.
    """
    closes: Dict[str, pd.Series] = {}
    opens: Dict[str, pd.Series] = {}
    dropped = []
    for p in pairs:
        try:
            df = load_or_fetch(
                p, cache_dir=cache_dir, refresh=refresh, client=client,
                validate=False,
            )
            validate_history(df, p, min_rows=min_rows)
        except FactorDataError as exc:
            logger.warning("FP-1: dropping %s from universe — %s", p, exc)
            dropped.append((p, str(exc)))
            continue
        closes[p] = df["close"].rename(p)
        opens[p] = df["open"].rename(p)
    if not closes:
        raise FactorDataError("No instruments survived validation for the universe")
    close = pd.concat(closes.values(), axis=1, join="inner").sort_index()
    open_ = pd.concat(opens.values(), axis=1, join="inner").sort_index()
    return close, open_, dropped


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
