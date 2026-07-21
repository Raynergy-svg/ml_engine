"""Daily adjusted-close price panel loader for the Track B research harness.

Produces the ``date x ticker`` adjusted-close DataFrame that
:mod:`src.equity.research.harness` consumes (``_normalize_prices`` expects a
``pd.DataFrame`` indexed by a UTC ``DatetimeIndex``, columns = tickers, values =
adjusted close). The harness ffill/overlay layer tolerates NaNs, so this loader
returns the RAW aligned adjusted close with ``NaN`` where a ticker had no print
on a given trading date — it does not impute.

Why the Yahoo CHART API and not yfinance
----------------------------------------

``yfinance`` is unusable in this environment: it now defaults to a ``curl_cffi``
transport whose TLS impersonation fails through the agent proxy. The plain
``requests`` path against the public Yahoo *chart* endpoint
(``query1.finance.yahoo.com/v8/finance/chart/{ticker}``) goes through the proxy
CA bundle exactly like the EDGAR path does, and returns a JSON document carrying
both the per-bar epoch timestamps and the aligned adjusted-close series.

Design guarantees
-----------------

* **Fail-soft per ticker.** A ticker that 404s, returns an empty body, or has an
  all-null adjusted-close series is logged and DROPPED (its column is omitted);
  it never crashes the panel. An empty panel (every ticker failed) is returned
  as an empty DataFrame rather than raising.
* **PIT-safe.** Every bar's epoch timestamp is floored to its UTC trading date
  (midnight UTC). Rows strictly after ``end`` are discarded, so the panel never
  carries information dated after the requested window.
* **Polite + bounded.** Explicit ``(connect=5, read=30)`` timeouts, a small
  inter-request sleep (well under 10 req/s), and retries ONLY on transient
  failures (5xx, 429, ``Timeout``, ``ConnectionError``) with exponential backoff
  and jitter. Other 4xx (e.g. 404 unknown ticker) are NOT retried — they mean
  the ticker is bad, so we drop it immediately.
* **No bare excepts.** Every caught exception is a specific type and is logged.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# --- Tunables (module constants, not magic numbers in the body) --------------
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_DEFAULT_CA = "/root/.ccr/ca-bundle.crt"
_USER_AGENT = "Mozilla/5.0"
_CONNECT_TIMEOUT = 5.0
_READ_TIMEOUT = 30.0
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.0
_BACKOFF_CAP = 30.0
_REQUEST_SPACING = 0.15  # seconds between requests (~6.7 req/s, < 10/s ceiling)
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# Errors that justify a retry (transient transport failures only).
_RETRYABLE_EXC = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)


class PriceLoaderError(Exception):
    """Raised only for caller-fault input errors (bad dates / empty tickers)."""


def _resolve_ca_bundle(ca_bundle: Optional[str]) -> Optional[str]:
    """Pick the CA bundle: explicit arg > env > known proxy default."""
    if ca_bundle:
        return ca_bundle
    env_ca = os.environ.get("REQUESTS_CA_BUNDLE")
    if env_ca:
        return env_ca
    if os.path.exists(_DEFAULT_CA):
        return _DEFAULT_CA
    return None  # fall back to certifi default if nothing else is present


def _to_epoch(iso_date: str, *, end_of_day: bool) -> int:
    """Parse an ISO ``YYYY-MM-DD`` date to a UTC epoch second.

    ``end_of_day`` extends an inclusive ``end`` to cover the whole trading day so
    the final bar is fetched; the period2 query param is exclusive-ish at Yahoo.
    """
    try:
        dt = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as exc:
        raise PriceLoaderError(f"invalid ISO date {iso_date!r}: {exc}") from exc
    if end_of_day:
        # +1 day so an inclusive `end` captures that day's bar.
        return int(dt.timestamp()) + 86400
    return int(dt.timestamp())


def parse_chart_payload(
    payload: dict,
) -> Tuple[List[pd.Timestamp], List[float]]:
    """Pure parse: Yahoo chart JSON dict -> (UTC-date timestamps, adjclose).

    Extracts ``result[0].timestamp`` (epoch seconds) aligned with
    ``result[0].indicators.adjclose[0].adjclose``. Each kept timestamp is floored
    to its UTC trading date (midnight UTC). Bars whose adjusted close is ``None``
    (Yahoo gap / halt) are DROPPED in lockstep with their timestamp.

    Returns two parallel, equal-length lists. Returns ``([], [])`` when the
    payload carries no usable data (missing keys, empty series, or all-null
    adjusted closes). This function performs NO network I/O and is the unit
    under the offline parse test.
    """
    if not isinstance(payload, dict):
        return [], []
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        return [], []
    results = chart.get("result")
    if not results or not isinstance(results, list):
        return [], []
    result = results[0]
    if not isinstance(result, dict):
        return [], []

    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not timestamps or not isinstance(indicators, dict):
        return [], []
    adj_block = indicators.get("adjclose")
    if not adj_block or not isinstance(adj_block, list):
        return [], []
    adj_values = adj_block[0].get("adjclose") if isinstance(adj_block[0], dict) else None
    if not adj_values:
        return [], []
    if len(timestamps) != len(adj_values):
        logger.warning(
            "parse_chart_payload: timestamp/adjclose length mismatch (%d vs %d) — "
            "dropping malformed payload",
            len(timestamps), len(adj_values),
        )
        return [], []

    dates: List[pd.Timestamp] = []
    closes: List[float] = []
    for epoch, value in zip(timestamps, adj_values):
        if value is None or epoch is None:
            continue
        # Floor the intraday market-open epoch to its UTC trading date.
        ts = pd.Timestamp(int(epoch), unit="s", tz="UTC").normalize()
        dates.append(ts)
        closes.append(float(value))
    return dates, closes


def _read_cache(cache_path: Path) -> Optional[dict]:
    """Best-effort read of a cached chart payload; never raises on bad cache."""
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("price cache unreadable at %s: %s", cache_path, exc)
        return None


def _write_cache(cache_path: Path, payload: dict) -> None:
    """Atomic cache write (tmp + os.replace); cache failure is non-fatal."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, cache_path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("price cache write failed at %s: %s", cache_path, exc)


def _cache_key(ticker: str, start: str, end: str) -> str:
    return f"{ticker.upper()}__{start}__{end}.json"


def _fetch_chart(
    ticker: str,
    epoch_start: int,
    epoch_end: int,
    *,
    session: requests.Session,
    ca_bundle: Optional[str],
) -> Optional[dict]:
    """Fetch one ticker's chart JSON. Returns the parsed dict, or None on a
    permanent failure (drop the ticker). Retries only transient failures.
    """
    url = _CHART_URL.format(ticker=ticker)
    params = {"period1": epoch_start, "period2": epoch_end, "interval": "1d"}
    headers = {"User-Agent": _USER_AGENT}

    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.get(
                url,
                params=params,
                headers=headers,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                verify=ca_bundle if ca_bundle else True,
            )
        except _RETRYABLE_EXC as exc:
            delay = _backoff(attempt)
            logger.warning(
                "transient fetch error for %s (attempt %d/%d): %s — retrying in %.1fs",
                ticker, attempt + 1, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
            continue
        except requests.exceptions.RequestException as exc:
            # Non-retryable request-layer error (e.g. malformed URL): drop.
            logger.warning("non-retryable fetch error for %s: %s — dropping", ticker, exc)
            return None

        status = resp.status_code
        if status == 200:
            try:
                return resp.json()
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("bad JSON for %s: %s — dropping", ticker, exc)
                return None
        if status in _RETRY_STATUS:
            delay = _backoff(attempt)
            logger.warning(
                "HTTP %d for %s (attempt %d/%d) — retrying in %.1fs",
                status, ticker, attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue
        # Any other 4xx (404 unknown ticker, 422 bad range): permanent, drop.
        logger.warning("HTTP %d for %s — dropping (not retryable)", status, ticker)
        return None

    logger.warning("exhausted retries for %s — dropping", ticker)
    return None


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter, capped."""
    base = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt))
    return random.uniform(0.0, base)


def load_price_panel(
    tickers: Sequence[str],
    start: str,
    end: str,
    *,
    session: Optional[requests.Session] = None,
    ca_bundle: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Build a daily adjusted-close panel for ``tickers`` over ``[start, end]``.

    Parameters
    ----------
    tickers:
        Iterable of Yahoo ticker symbols (e.g. ``["AAPL", "MSFT"]``).
    start, end:
        Inclusive ISO ``YYYY-MM-DD`` window bounds.
    session:
        Optional pre-built :class:`requests.Session` (connection reuse / test
        injection). One is created and closed internally if omitted.
    ca_bundle:
        Optional CA bundle path. Defaults to ``$REQUESTS_CA_BUNDLE`` then the
        known proxy bundle ``/root/.ccr/ca-bundle.crt``.
    cache_dir:
        Optional directory for an atomic on-disk JSON cache keyed by
        ``(ticker, start, end)``. Omit to disable caching.

    Returns
    -------
    pandas.DataFrame
        UTC ``DatetimeIndex`` (one row per trading date), one column per ticker
        that returned usable data, values = adjusted close (NaN where that
        ticker had no print on a date). A ticker that fails entirely is dropped
        (column omitted). Returns an EMPTY DataFrame if every ticker failed.

    Raises
    ------
    PriceLoaderError
        Only on caller-fault input: empty ticker list, ``start > end``, or a
        non-ISO date. Never raised for a per-ticker network failure.
    """
    symbols = [str(t).strip() for t in tickers if str(t).strip()]
    if not symbols:
        raise PriceLoaderError("tickers must be a non-empty sequence")

    epoch_start = _to_epoch(start, end_of_day=False)
    epoch_end = _to_epoch(end, end_of_day=True)
    if epoch_start >= epoch_end:
        raise PriceLoaderError(f"start {start!r} must be before end {end!r}")

    end_ts = pd.Timestamp(end, tz="UTC").normalize()
    resolved_ca = _resolve_ca_bundle(ca_bundle)
    cache_root = Path(cache_dir) if cache_dir else None

    owns_session = session is None
    sess = session if session is not None else requests.Session()
    series_by_ticker: Dict[str, pd.Series] = {}

    try:
        for i, ticker in enumerate(symbols):
            payload: Optional[dict] = None
            cache_path = (
                cache_root / _cache_key(ticker, start, end) if cache_root else None
            )
            if cache_path is not None:
                payload = _read_cache(cache_path)

            if payload is None:
                if i > 0:
                    time.sleep(_REQUEST_SPACING)  # politeness spacing
                payload = _fetch_chart(
                    ticker, epoch_start, epoch_end,
                    session=sess, ca_bundle=resolved_ca,
                )
                if payload is not None and cache_path is not None:
                    _write_cache(cache_path, payload)

            if payload is None:
                continue  # already logged; column dropped

            dates, closes = parse_chart_payload(payload)
            if not dates:
                logger.warning("no usable adjclose for %s — dropping column", ticker)
                continue

            ser = pd.Series(closes, index=pd.DatetimeIndex(dates), name=ticker)
            # De-dup (keep last) + sort + PIT clamp (never a row after `end`).
            ser = ser[~ser.index.duplicated(keep="last")].sort_index()
            ser = ser[ser.index <= end_ts]
            if ser.empty:
                logger.warning("all bars for %s fell outside window — dropping", ticker)
                continue
            series_by_ticker[ticker] = ser
    finally:
        if owns_session:
            sess.close()

    if not series_by_ticker:
        logger.warning(
            "price panel empty: all %d tickers failed/dropped", len(symbols)
        )
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    # Align on the union of dates; preserve requested ticker order for kept cols.
    panel = pd.DataFrame(series_by_ticker)
    panel = panel[[t for t in symbols if t in series_by_ticker]]
    panel = panel.sort_index()
    if panel.index.tz is None:  # defensive; series carry tz already
        panel.index = panel.index.tz_localize("UTC")
    return panel
