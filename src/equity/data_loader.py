"""US-001 — Daily adjusted OHLCV loader for the equity-beta harvester.

Mirrors the contract of ``src/factor/data_loader.py``:

- Atomic cache writes (``.tmp`` + ``os.replace``)
- Fail-loud (``EquityDataError``) on short/stale/NaN data — no silent fixup
- Real classes, real disk in tests (no mocks)
- Source-explicit: a JSON sidecar records which live source produced each CSV
  (``ibkr`` preferred for live trading via ``fetch_candles('D')``; ``yfinance``
  is backtest-only).

The cache lives under ``market_data/equity/{TICKER}_D.csv`` plus a sidecar
``{TICKER}_D.meta.json``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ~10 years of US sessions (~252/yr) for any history-hungry downstream.
MIN_ROWS = 2_500
DEFAULT_FRESHNESS_DAYS = 7
DEFAULT_LOOKBACK_DAYS = 252 * 12
DEFAULT_CACHE_DIR = Path("market_data/equity")

OHLC_COLUMNS = ("open", "high", "low", "close")
ALLOWED_SOURCES = ("ibkr", "yfinance")


class EquityDataError(RuntimeError):
    """Raised when equity data is missing, stale, NaN-bearing, or unloadable."""


@dataclass(frozen=True)
class EquityCacheMeta:
    """JSON sidecar describing the provenance of a cached CSV."""

    ticker: str
    source: str
    fetched_at: str   # UTC ISO-8601
    rows: int
    last_date: str    # UTC ISO-8601 date of the latest row
    pipeline_version: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    @classmethod
    def from_path(cls, path: Path) -> "EquityCacheMeta":
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise EquityDataError(
                f"Cache meta unreadable at {path}: {exc}"
            ) from exc
        try:
            return cls(**payload)
        except TypeError as exc:
            raise EquityDataError(
                f"Cache meta schema mismatch at {path}: {exc}"
            ) from exc


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=True)
    os.replace(tmp, path)


def _normalize_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame with lowercase columns, sorted, deduped."""
    if df.empty:
        raise EquityDataError(f"{ticker}: empty OHLCV frame from source")

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    missing = [c for c in OHLC_COLUMNS if c not in df.columns]
    if missing:
        raise EquityDataError(
            f"{ticker}: source frame missing OHLC columns {missing}"
        )

    if "volume" not in df.columns:
        df["volume"] = 0.0

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.DatetimeIndex(df.index)
        except (TypeError, ValueError) as exc:
            raise EquityDataError(
                f"{ticker}: index is not date-like ({exc})"
            ) from exc

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index = df.index.normalize()
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    keep = list(OHLC_COLUMNS) + ["volume"]
    return df[keep]


def _ibkr_candles_to_frame(candles: List[object], ticker: str) -> pd.DataFrame:
    """Convert IBKR ``CandleData`` objects to a normalized OHLCV frame."""
    if not candles:
        raise EquityDataError(f"{ticker}: IBKR returned zero candles")
    rows = []
    for c in candles:
        try:
            rows.append(
                {
                    "date": pd.Timestamp(c.time),
                    "open": float(c.open),
                    "high": float(c.high),
                    "low": float(c.low),
                    "close": float(c.close),
                    "volume": float(getattr(c, "volume", 0) or 0),
                }
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise EquityDataError(
                f"{ticker}: malformed IBKR candle {c!r}: {exc}"
            ) from exc
    df = pd.DataFrame(rows).set_index("date")
    return _normalize_frame(df, ticker)


def fetch_daily_ibkr(
    ticker: str,
    *,
    ibkr_client: object,
    count: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Fetch adjusted daily candles from IBKR (live-preferred path).

    The caller passes a connected IBKR client (``src/brokers/ibkr.py``). We
    build an ``Instrument`` with ``asset_class='STK'`` semantics on the broker
    side; today the broker abstraction only models FX/FUTURES, so callers can
    also pass a duck-typed client whose ``fetch_candles(symbol, 'D', count)``
    returns ``CandleData``-shaped objects.
    """
    if ibkr_client is None:
        raise EquityDataError(
            f"{ticker}: IBKR source requested but no client provided"
        )
    try:
        candles = ibkr_client.fetch_candles(ticker, "D", count)
    except (ConnectionError, TimeoutError, ValueError) as exc:
        raise EquityDataError(
            f"{ticker}: IBKR fetch_candles failed: {exc}"
        ) from exc
    return _ibkr_candles_to_frame(candles, ticker)


def fetch_daily_yfinance(
    ticker: str,
    *,
    period: str = "max",
) -> pd.DataFrame:
    """Backtest-only path: pull adjusted daily bars via yfinance.

    yfinance is NOT to be used on the live trading path — feed quality and
    survivorship guarantees are insufficient. This helper exists so the
    backtest harness has a self-serve EOD source.
    """
    try:
        import yfinance as yf  # noqa: WPS433 — lazy import; backtest-only path
    except ImportError as exc:
        raise EquityDataError(
            f"{ticker}: yfinance not installed ({exc}); "
            "install it for the backtest path or use the IBKR source"
        ) from exc

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if df is None or df.empty:
        raise EquityDataError(f"{ticker}: yfinance returned no rows")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _normalize_frame(df, ticker)


def validate_history(
    df: pd.DataFrame,
    ticker: str,
    *,
    min_rows: int = MIN_ROWS,
) -> None:
    """Hard-fail on short history or NaN OHLC — no silent forward-fill."""
    if len(df) < min_rows:
        raise EquityDataError(
            f"{ticker}: only {len(df)} daily rows (< {min_rows}). "
            "Refusing to build equity factors on insufficient history."
        )
    for col in OHLC_COLUMNS:
        if col not in df.columns:
            raise EquityDataError(f"{ticker}: missing column '{col}'")
    if df[list(OHLC_COLUMNS)].isna().any().any():
        raise EquityDataError(
            f"{ticker}: NaNs in OHLC — refusing to forward-fill silently"
        )


def validate_freshness(
    df: pd.DataFrame,
    ticker: str,
    *,
    freshness_days: int = DEFAULT_FRESHNESS_DAYS,
    now: Optional[datetime] = None,
) -> None:
    """Raise if the latest row is older than ``freshness_days``."""
    if df.empty:
        raise EquityDataError(f"{ticker}: empty frame at freshness check")
    last = df.index.max()
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_days = (now_utc - last.to_pydatetime()).days
    if age_days > freshness_days:
        raise EquityDataError(
            f"{ticker}: latest row {last.date()} is {age_days}d old "
            f"(> {freshness_days}d freshness window). Refusing to trade stale data."
        )


def _write_cache(
    df: pd.DataFrame,
    ticker: str,
    source: str,
    cache_dir: Path,
) -> EquityCacheMeta:
    if source not in ALLOWED_SOURCES:
        raise EquityDataError(
            f"{ticker}: unknown source '{source}' "
            f"(allowed: {ALLOWED_SOURCES})"
        )
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / f"{ticker}_D.csv"
    meta_path = cache_dir / f"{ticker}_D.meta.json"

    last = df.index.max()
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    from . import EQUITY_PIPELINE_VERSION
    meta = EquityCacheMeta(
        ticker=ticker,
        source=source,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        rows=int(len(df)),
        last_date=last.date().isoformat(),
        pipeline_version=EQUITY_PIPELINE_VERSION,
    )
    _atomic_write_csv(df, csv_path)
    _atomic_write_text(meta.to_json(), meta_path)
    return meta


def _read_cache(ticker: str, cache_dir: Path) -> pd.DataFrame:
    csv_path = Path(cache_dir) / f"{ticker}_D.csv"
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise EquityDataError(
            f"{ticker}: cache CSV unreadable at {csv_path}: {exc}"
        ) from exc
    return _normalize_frame(df, ticker)


def load_or_fetch(
    ticker: str,
    *,
    source: str = "ibkr",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    ibkr_client: Optional[object] = None,
    validate: bool = True,
    min_rows: int = MIN_ROWS,
    freshness_days: int = DEFAULT_FRESHNESS_DAYS,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    """Return adjusted daily OHLCV for ``ticker``.

    Uses the cache when present; otherwise fetches from the requested source
    and writes the cache atomically (with a JSON sidecar tagging the source).

    Fails loud on:
      * cache CSV unreadable (no silent refetch — corruption is signal)
      * short history (< ``min_rows``)
      * NaN OHLC (no silent forward-fill)
      * stale last row (> ``freshness_days`` old) when ``validate=True``
    """
    if source not in ALLOWED_SOURCES:
        raise EquityDataError(
            f"{ticker}: unknown source '{source}' "
            f"(allowed: {ALLOWED_SOURCES})"
        )

    cache_dir = Path(cache_dir)
    csv_path = cache_dir / f"{ticker}_D.csv"

    if csv_path.exists() and not refresh:
        df = _read_cache(ticker, cache_dir)
    else:
        if source == "ibkr":
            df = fetch_daily_ibkr(ticker, ibkr_client=ibkr_client)
        else:
            df = fetch_daily_yfinance(ticker)
        _write_cache(df, ticker, source, cache_dir)

    if validate:
        validate_history(df, ticker, min_rows=min_rows)
        validate_freshness(df, ticker, freshness_days=freshness_days, now=now)
    return df


def read_cache_meta(ticker: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> EquityCacheMeta:
    """Read the JSON sidecar so callers can audit which source produced rows."""
    meta_path = Path(cache_dir) / f"{ticker}_D.meta.json"
    return EquityCacheMeta.from_path(meta_path)


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
