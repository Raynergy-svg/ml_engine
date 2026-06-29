"""Free, survivorship-aware crypto data adapter (research/backtest only).

Pulls perp funding rates + OHLCV from three free sources and caches to disk
(parquet) so backtests run offline and deterministically. No API keys, no
secrets, no order path.

Primary depth source  : Binance static dumps (data.binance.vision)
Cross-check (CEX)     : OKX public REST
Cross-check (DEX)     : Hyperliquid public REST

Survivorship: ``list_binance_perp_symbols()`` enumerates ALL symbol directories
in the static archive, which retains delisted coins (LUNAUSDT, FTTUSDT, ...).
A point-in-time universe (see ``universe.py``) therefore includes coins that
later died -- the carry-crash left tail is visible rather than silently deleted.
LIMITATION (flagged honestly): the dumps stop emitting funding/klines when a
symbol delists; we close at last observed price, which UNDERSTATES true
delisting losses (no -100% liquidation gap is modeled). Same class of caveat as
the equity harvester's survivor-only 2008 understatement.
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
# defusedxml: stdlib ElementTree is vulnerable to XXE / billion-laughs. The S3
# listing is from AWS (trusted) but we harden anyway (repo security discipline).
from defusedxml import ElementTree as ET

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
# crypto_cache/ matches the *_cache/ rule in .gitignore -> never committed.
CACHE_DIR = REPO_ROOT / "crypto_cache"

BINANCE_DUMP = "https://data.binance.vision/data/futures/um/monthly"
BINANCE_S3_XML = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
OKX_BASE = "https://www.okx.com/api/v5"
HL_BASE = "https://api.hyperliquid.xyz/info"

# Binance USDⓈ-M funding history begins 2020-01 (verified: 2019-12 -> 404).
BINANCE_FUNDING_START = "2020-01"

_CONNECT_TIMEOUT = 5
_READ_TIMEOUT = 30
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.0
_POLITE_SLEEP = 0.03  # between sequential requests (static CDN bucket tolerates it)


# --------------------------------------------------------------------------- #
# HTTP helpers (exponential backoff, no retry on 4xx per repo robustness gate)
# --------------------------------------------------------------------------- #
def _get(url: str, *, params: dict | None = None, method: str = "GET",
         json_body: dict | None = None) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method, url, params=params, json=json_body,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            if resp.status_code == 404:
                return resp  # caller handles "archive not present yet"
            if 400 <= resp.status_code < 500:
                # client error other than 404: do not retry, surface it.
                resp.raise_for_status()
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} server error")
            return resp
        except (requests.Timeout, requests.ConnectionError,
                requests.HTTPError) as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            delay = min(_BACKOFF_BASE * (2 ** (attempt - 1)), 30.0)
            logger.warning("GET %s attempt %d/%d failed (%s); retry in %.1fs",
                           url, attempt, _MAX_RETRIES, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"GET {url} failed after {_MAX_RETRIES} attempts: {last_exc}")


def _month_range(start: str, end: str | None = None) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end (default: now)."""
    s = datetime.strptime(start, "%Y-%m").replace(tzinfo=timezone.utc)
    e = (datetime.now(timezone.utc) if end is None
         else datetime.strptime(end, "%Y-%m").replace(tzinfo=timezone.utc))
    out, y, m = [], s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)  # atomic on same filesystem


# --------------------------------------------------------------------------- #
# Universe enumeration (survivorship-aware: includes delisted symbols)
# --------------------------------------------------------------------------- #
def list_binance_perp_symbols(quote: str = "USDT") -> list[str]:
    """All perp symbols with a funding archive, INCLUDING delisted ones.

    Reads the S3 XML listing of the static dump. Filters to ``quote``-margined
    (default USDT). This is the survivorship-complete symbol set.
    """
    syms: list[str] = []
    token = ""
    prefix = "data/futures/um/monthly/fundingRate/"
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if token:
            params["marker"] = token
        resp = _get(BINANCE_S3_XML, params=params)
        root = ET.fromstring(resp.content)
        ns = {"s3": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        cps = root.findall(".//s3:CommonPrefixes/s3:Prefix", ns) if ns \
            else root.findall(".//CommonPrefixes/Prefix")
        page = [el.text.split("/")[-2] for el in cps if el.text]
        syms.extend(page)
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=ns)
                     if ns else root.findtext("IsTruncated", default="false"))
        if truncated.lower() != "true" or not page:
            break
        token = page[-1]  # marker pagination by last key prefix
    out = sorted({s for s in syms if s.endswith(quote)})
    logger.info("binance perp symbols (%s, incl delisted): %d", quote, len(out))
    return out


# --------------------------------------------------------------------------- #
# Binance static dumps: funding + klines
# --------------------------------------------------------------------------- #
def _read_zip_csv(content: bytes, header: list[str] | None = None) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(content))
    name = zf.namelist()[0]
    raw = zf.open(name)
    # Some monthly CSVs ship with a header row, some don't; sniff the first byte.
    first = zf.open(name).read(64).decode("utf-8", "replace")
    has_header = any(c.isalpha() for c in first.split(",")[0])
    # Binance added header rows to newer dumps (e.g. klines 2025+) using ITS OWN
    # column names (taker_buy_volume, not our positional taker_base). Always read
    # POSITIONALLY and skip a header row if present, then assign our canonical
    # names -> robust to both header and headerless monthly files.
    return pd.read_csv(raw, header=None, names=header,
                       skiprows=1 if has_header else 0)


def fetch_binance_funding(symbol: str, *, refresh: bool = False) -> pd.DataFrame:
    """Funding-rate series for one symbol from the static dump.

    Returns DataFrame indexed by UTC ``funding_time`` with columns
    ``funding_rate`` (per-interval, decimal) and ``interval_hours``.
    Empty DataFrame if the symbol has no archive.
    """
    cache = CACHE_DIR / "binance" / "funding" / f"{symbol}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for ym in _month_range(BINANCE_FUNDING_START):
        url = f"{BINANCE_DUMP}/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        resp = _get(url)
        if resp.status_code == 404:
            continue
        df = _read_zip_csv(
            resp.content,
            header=["calc_time", "funding_interval_hours", "last_funding_rate"],
        )
        frames.append(df)
        time.sleep(_POLITE_SLEEP)
    if not frames:
        return pd.DataFrame(columns=["funding_rate", "interval_hours"])

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"calc_time": "ts",
                            "funding_interval_hours": "interval_hours",
                            "last_funding_rate": "funding_rate"})
    df["funding_time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = (df.dropna(subset=["funding_rate"])
            .drop_duplicates(subset=["funding_time"])
            .set_index("funding_time")
            .sort_index()[["funding_rate", "interval_hours"]])
    df["funding_rate"] = df["funding_rate"].astype(float)
    _atomic_parquet(df, cache)
    return df


def fetch_binance_klines(symbol: str, interval: str = "1d", *,
                         refresh: bool = False) -> pd.DataFrame:
    """OHLCV for one symbol from the static dump (futures UM perp).

    Returns DataFrame indexed by UTC ``open_time`` with float OHLCV columns.
    """
    cache = CACHE_DIR / "binance" / "klines" / interval / f"{symbol}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    frames = []
    for ym in _month_range(BINANCE_FUNDING_START):
        url = f"{BINANCE_DUMP}/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        resp = _get(url)
        if resp.status_code == 404:
            continue
        df = _read_zip_csv(resp.content, header=cols)
        frames.append(df)
        time.sleep(_POLITE_SLEEP)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume",
                                     "quote_volume", "taker_base"])
    df = pd.concat(frames, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"),
                                     unit="ms", utc=True)
    # taker_base = taker BUY base volume (col 9) -> order-flow imbalance signal (H3)
    keep = ["open", "high", "low", "close", "volume", "quote_volume", "taker_base"]
    df = (df.drop_duplicates(subset=["open_time"]).set_index("open_time")
            .sort_index()[keep].astype(float))
    _atomic_parquet(df, cache)
    return df


# --------------------------------------------------------------------------- #
# OKX cross-check (CEX): OHLCV history + recent funding
# --------------------------------------------------------------------------- #
def fetch_okx_ohlcv(inst_id: str, bar: str = "1D", *,
                    refresh: bool = False) -> pd.DataFrame:
    """Paginated OKX history candles (back to ~2020 for BTC). For cross-check."""
    cache = CACHE_DIR / "okx" / "ohlcv" / bar / f"{inst_id}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows: list[list] = []
    after = ""  # pagination: records older than this ts
    while True:
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after:
            params["after"] = after
        data = _get(f"{OKX_BASE}/market/history-candles", params=params).json().get("data", [])
        if not data:
            break
        rows.extend(data)
        after = data[-1][0]
        time.sleep(_POLITE_SLEEP)
        if len(rows) > 5000:  # ~13y of daily; safety bound
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "volume", "volCcy", "volCcyQuote", "confirm"])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df = (df.drop_duplicates(subset=["ts"]).set_index("ts").sort_index())
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    _atomic_parquet(df, cache)
    return df


def fetch_okx_funding_recent(inst_id: str) -> pd.DataFrame:
    """Most-recent ~3 months of OKX funding (API retention limit). Live-ish check."""
    data = _get(f"{OKX_BASE}/public/funding-rate-history",
                params={"instId": inst_id, "limit": "100"}).json().get("data", [])
    if not data:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(data)
    df["funding_time"] = pd.to_datetime(df["fundingTime"].astype("int64"),
                                        unit="ms", utc=True)
    df["funding_rate"] = df["realizedRate"].astype(float)
    return df.set_index("funding_time").sort_index()[["funding_rate"]]


# --------------------------------------------------------------------------- #
# Hyperliquid cross-check (DEX): hourly funding from 2023-07
# --------------------------------------------------------------------------- #
def fetch_hl_universe() -> list[str]:
    meta = _get(HL_BASE, method="POST", json_body={"type": "meta"}).json()
    return [u["name"] for u in meta["universe"]]


def fetch_hl_funding(coin: str, start_ms: int = 1688169600000, *,
                     refresh: bool = False) -> pd.DataFrame:
    """Hourly funding for one HL coin from 2023-07 (paginated, full retention)."""
    cache = CACHE_DIR / "hyperliquid" / "funding" / f"{coin}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows: list[dict] = []
    cur = start_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    while cur < now_ms:
        batch = _get(HL_BASE, method="POST",
                     json_body={"type": "fundingHistory", "coin": coin,
                                "startTime": cur}).json()
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1]["time"])
        if last <= cur:
            break
        cur = last + 1
        time.sleep(_POLITE_SLEEP)
    if not rows:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = (df.drop_duplicates(subset=["funding_time"]).set_index("funding_time")
            .sort_index()[["funding_rate"]])
    _atomic_parquet(df, cache)
    return df


# --------------------------------------------------------------------------- #
# Coverage report (honest survivorship accounting)
# --------------------------------------------------------------------------- #
@dataclass
class SymbolCoverage:
    symbol: str
    funding_start: str | None
    funding_end: str | None
    funding_rows: int
    delisted: bool  # last funding > 45d before global max -> treated as dead


def coverage_report(symbols: Iterable[str], *, refresh: bool = False,
                    dead_gap_days: int = 45) -> pd.DataFrame:
    """Per-symbol funding coverage + delisting flag. Survivorship transparency."""
    recs: list[SymbolCoverage] = []
    global_end: pd.Timestamp | None = None
    cache_frames: dict[str, pd.DataFrame] = {}
    for s in symbols:
        f = fetch_binance_funding(s, refresh=refresh)
        cache_frames[s] = f
        if not f.empty:
            end = f.index.max()
            global_end = end if global_end is None else max(global_end, end)
    cutoff = (global_end - pd.Timedelta(days=dead_gap_days)) if global_end is not None else None
    for s, f in cache_frames.items():
        if f.empty:
            recs.append(SymbolCoverage(s, None, None, 0, True))
            continue
        end = f.index.max()
        recs.append(SymbolCoverage(
            symbol=s,
            funding_start=str(f.index.min().date()),
            funding_end=str(end.date()),
            funding_rows=len(f),
            delisted=bool(cutoff is not None and end < cutoff),
        ))
    return pd.DataFrame([r.__dict__ for r in recs])
