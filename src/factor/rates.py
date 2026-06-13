"""US-002 data layer — short-rate panel for the carry factor.

Source: OECD 3-month interbank rates via FRED (free, no key), one series per
currency. These are directly comparable across countries, which is what a
cross-sectional carry rank needs. Monthly observations are forward-filled to the
daily price calendar and lagged one month so a signal at date T only uses a rate
print that was already published by T (no lookahead).

Cached to ``market_data/factor/rates_{CCY}.csv``; fetch is fail-loud.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import PAIRS
from .data_loader import DEFAULT_CACHE_DIR, FactorDataError

logger = logging.getLogger(__name__)

# 3-month interbank rate, OECD via FRED (monthly, % annualized).
FRED_RATE_IDS: Dict[str, str] = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N",
}
PUBLICATION_LAG_MONTHS = 1


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=True)
    os.replace(tmp, path)


def fetch_rate_series(ccy: str, *, session: Optional[object] = None) -> pd.Series:
    """Fetch one currency's monthly 3m interbank rate from FRED."""
    import requests

    fid = FRED_RATE_IDS[ccy]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}"
    get = (session or requests).get
    last_exc = None
    for _ in range(3):
        try:
            r = get(url, timeout=15)
            if r.status_code == 200 and len(r.text) > 100:
                from io import StringIO

                df = pd.read_csv(StringIO(r.text))
                df.columns = ["date", "rate"]
                df["date"] = pd.to_datetime(df["date"], utc=True)
                df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
                s = df.dropna().set_index("date")["rate"].sort_index()
                if s.empty:
                    raise FactorDataError(f"{ccy}: FRED series {fid} parsed empty")
                return s.rename(ccy)
        except Exception as exc:  # network flake -> retry
            last_exc = exc
    raise FactorDataError(f"{ccy}: FRED fetch failed for {fid}: {last_exc}")


def load_or_fetch_rate(
    ccy: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    session: Optional[object] = None,
) -> pd.Series:
    cache_dir = Path(cache_dir)
    path = cache_dir / f"rates_{ccy}.csv"
    if path.exists() and not refresh:
        try:
            s = pd.read_csv(path, index_col=0, parse_dates=True)["rate"]
            s.index = pd.DatetimeIndex(s.index)
            if s.index.tz is None:
                s.index = s.index.tz_localize("UTC")
            return s.rename(ccy)
        except Exception as exc:
            logger.warning("rates %s cache unreadable (%s); refetching", ccy, exc)
    s = fetch_rate_series(ccy, session=session)
    _atomic_write_csv(s.to_frame("rate"), path)
    return s


def _pair_ccys(pair: str) -> tuple:
    base, quote = pair.split("_")
    return base, quote


def build_rate_diff_panel(
    daily_index: pd.DatetimeIndex,
    pairs: Optional[List[str]] = None,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    session: Optional[object] = None,
) -> pd.DataFrame:
    """Return a date×pair frame of base-minus-quote short-rate differentials (%).

    Positive value => being long the pair earns positive carry. Monthly rates are
    lagged ``PUBLICATION_LAG_MONTHS`` then forward-filled onto ``daily_index``.
    """
    pairs = pairs or PAIRS
    ccys = sorted({c for p in pairs for c in _pair_ccys(p)})
    rates: Dict[str, pd.Series] = {}
    for c in ccys:
        s = load_or_fetch_rate(c, cache_dir=cache_dir, refresh=refresh, session=session)
        s = s.shift(PUBLICATION_LAG_MONTHS)  # only use already-published prints
        rates[c] = s

    out = {}
    for p in pairs:
        base, quote = _pair_ccys(p)
        diff_m = (rates[base] - rates[quote]).dropna()
        daily = diff_m.reindex(diff_m.index.union(daily_index)).sort_index()
        daily = daily.ffill().reindex(daily_index)
        out[p] = daily.rename(p)
    panel = pd.concat(out.values(), axis=1)
    if panel.dropna(how="all").empty:
        raise FactorDataError("Empty rate-diff panel after alignment")
    return panel
