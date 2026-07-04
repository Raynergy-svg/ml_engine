#!/usr/bin/env python3
"""FINRA daily Reg SHO short-volume downloader — bronze-layer, immutable cache.

Pre-registered data source for docs/prereg-finra-short-volume-2026-07-04.md
(FROZEN before any signal/backtest code runs). This script ONLY downloads and
caches raw files; it does no parsing, no signal construction, no backtesting.

Endpoint (verified live 2026-07-04): the CNMS consolidated-tape daily
short-volume file, requires a browser-like User-Agent or the CDN's bot filter
403s (this is NOT an access restriction — a bare/no-UA request 403s, a
Mozilla-style UA succeeds):

    https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

Format (pipe-delimited, one row per (date, symbol) — the CNMS file is already
consolidated across markets, with ``Market`` as a comma-joined list like
``B,Q,N`` rather than one row per market):

    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

History depth is a ROLLING CDN retention window (~7.5 years as of 2026-07-04),
not a fixed-start archive — dates before ~2019-01 return 403. This script does
not hardcode that assumption beyond the pre-reg's frozen sample window
(2019-01-02 through 2026-06-24); it fetches whatever the trading calendar asks
for and logs/skips 403s with the reason, so a future re-run naturally sees
whatever window the CDN still serves.

Bronze-layer discipline: each day's raw file is cached IMMUTABLY at
``market_data/equity_alt/finra_short_volume/raw/CNMSshvol{YYYYMMDD}.txt``.
Existing cache files are never overwritten or re-fetched — append-only.

Usage:
    python scripts/fetch_finra_short_volume.py
    python scripts/fetch_finra_short_volume.py --start 2019-01-02 --end 2026-06-24
    python scripts/fetch_finra_short_volume.py --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fetch_finra_short_volume")

# Pre-reg frozen sample window (docs/prereg-finra-short-volume-2026-07-04.md).
DEFAULT_START = "2019-01-02"
DEFAULT_END = "2026-06-24"

PRICE_PANEL_PATH = REPO_ROOT / "market_data" / "equity" / "sp500_prices.parquet"
CACHE_DIR = REPO_ROOT / "market_data" / "equity_alt" / "finra_short_volume" / "raw"

FINRA_URL_TMPL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
# CDN bot-filters bare requests; a browser-like UA is required (verified live).
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_TIMEOUT = (5, 30)  # (connect, read) seconds
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 1.5


def trading_calendar(start: str, end: str) -> List[pd.Timestamp]:
    """Trading days from the price panel's own index — never a guessed calendar."""
    if not PRICE_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"price panel not found at {PRICE_PANEL_PATH} — cannot derive a "
            "trading calendar without guessing one"
        )
    prices = pd.read_parquet(PRICE_PANEL_PATH)
    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC")
    days = sorted(d for d in idx.unique() if lo <= d <= hi)
    return days


def _fetch_one(session: requests.Session, day: pd.Timestamp) -> Tuple[str, str]:
    """Fetch and cache one day's file. Returns (status, detail).

    status in {"cached_already", "downloaded", "skipped_403", "skipped_404",
    "skipped_error"}.
    """
    ymd = day.strftime("%Y%m%d")
    dest = CACHE_DIR / f"CNMSshvol{ymd}.txt"
    if dest.exists():
        return "cached_already", str(dest)

    url = FINRA_URL_TMPL.format(ymd=ymd)
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = session.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=_TIMEOUT
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS:
                sleep_s = _BACKOFF_BASE_S * attempt
                logger.warning(
                    "%s: attempt %d/%d network error (%s), retrying in %.1fs",
                    ymd, attempt, _MAX_ATTEMPTS, exc, sleep_s,
                )
                time.sleep(sleep_s)
                continue
            return "skipped_error", f"network error after {attempt} attempts: {exc}"

        if resp.status_code == 200:
            # Bronze layer: write raw bytes atomically (tmp + rename), never
            # partial/overwritten.
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(resp.content)
            tmp.rename(dest)
            return "downloaded", str(dest)
        if resp.status_code == 403:
            # Confirmed (pre-reg, verified live): CDN retention window or
            # bot-filter — not worth retrying, this is a per-day terminal state.
            return "skipped_403", f"HTTP 403 for {ymd} (outside CDN retention window?)"
        if resp.status_code == 404:
            return "skipped_404", f"HTTP 404 for {ymd} (no file published)"
        if resp.status_code >= 500 and attempt < _MAX_ATTEMPTS:
            sleep_s = _BACKOFF_BASE_S * attempt
            logger.warning(
                "%s: attempt %d/%d got HTTP %d, retrying in %.1fs",
                ymd, attempt, _MAX_ATTEMPTS, resp.status_code, sleep_s,
            )
            time.sleep(sleep_s)
            continue
        return "skipped_error", f"HTTP {resp.status_code} for {ymd}"

    return "skipped_error", f"exhausted retries: {last_exc}"


def run(start: str, end: str, limit: int | None) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    days = trading_calendar(start, end)
    if limit is not None:
        days = days[:limit]
    logger.info(
        "trading calendar: %d days in [%s, %s] from %s",
        len(days), start, end, PRICE_PANEL_PATH,
    )

    manifest = {
        "requested": len(days),
        "cached_already": 0,
        "downloaded": 0,
        "skipped_403": 0,
        "skipped_404": 0,
        "skipped_error": 0,
        "skip_reasons": [],
    }
    session = requests.Session()
    for i, day in enumerate(days):
        status, detail = _fetch_one(session, day)
        manifest[status] = manifest.get(status, 0) + 1
        if status.startswith("skipped"):
            manifest["skip_reasons"].append(
                {"date": day.strftime("%Y-%m-%d"), "status": status, "detail": detail}
            )
        if status == "downloaded":
            logger.info("[%d/%d] %s -> downloaded", i + 1, len(days), day.date())
        elif i % 200 == 0:
            logger.info(
                "[%d/%d] %s -> %s", i + 1, len(days), day.date(), status
            )

    logger.info("=" * 70)
    logger.info("MANIFEST: requested=%d cached_already=%d downloaded=%d "
                "skipped_403=%d skipped_404=%d skipped_error=%d",
                manifest["requested"], manifest["cached_already"],
                manifest["downloaded"], manifest["skipped_403"],
                manifest["skipped_404"], manifest["skipped_error"])
    n_ok = manifest["cached_already"] + manifest["downloaded"]
    logger.info("successfully cached (this run + prior): %d / %d days",
                n_ok, manifest["requested"])
    if manifest["skip_reasons"]:
        logger.info("first 10 skip reasons: %s", manifest["skip_reasons"][:10])
    logger.info("=" * 70)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START, help="ISO start date (inclusive)")
    ap.add_argument("--end", default=DEFAULT_END, help="ISO end date (inclusive)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of days (smoke test)")
    args = ap.parse_args()
    run(args.start, args.end, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
