"""Survivorship-aware S&P 500 PIT membership from Wikipedia (free; offline; running:NO).

A cross-sectional quality factor needs the universe AS-OF each date — including
names that were members then and have since been removed/delisted — otherwise the
backtest only ever sees today's survivors (survivorship bias = fake alpha, L-018).

This reconstructs point-in-time membership for free: take Wikipedia's CURRENT S&P
500 constituents + its documented add/remove change log, and roll the set BACKWARD
through the changes to recover who was a member on any past date. The result is a
date×ticker boolean membership matrix wrapped as a ``UniverseSnapshot`` so it plugs
straight into the existing equal-weight / variant machinery.

HONESTY LABELLING (mandatory):
  - Membership source = Wikipedia change log; this is a reconstruction, not a
    vendor index-membership feed. Completeness DEGRADES going back in time (the
    change table thins pre-~2010) and ticker-symbol renames are not perfectly
    tracked. Combined with the EDGAR XBRL ~2010 fundamentals floor, the honest
    clean window is ~2012-2026. These limits are reported, not hidden.
  - It DOES include since-removed names (real survivorship mitigation). Residual
    survivorship can still enter via price-data availability (yfinance often lacks
    delisted tickers) — that is labelled where it bites, not papered over.

OFFLINE only — one Wikipedia fetch + pure reconstruction; nothing trades/unhalts.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.equity import EQUITY_PIPELINE_VERSION
from src.equity.universe import UniverseRules, UniverseSnapshot, compute_universe_hash

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_UA = "ml_engine-research dcertan84@gmail.com"


def canonical_ticker(sym: str) -> str:
    """Normalize to the yfinance/EDGAR convention: class shares use '-' (BRK-B)."""
    return str(sym).strip().upper().replace(".", "-")


def _fetch_wiki_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current_constituents, changes) tables from Wikipedia."""
    resp = requests.get(WIKI_URL, headers={"User-Agent": _UA}, timeout=(5, 30))
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    if len(tables) < 2:
        raise RuntimeError("Wikipedia S&P 500 page layout changed (expected >=2 tables)")
    return tables[0], tables[1]


def _parse_current(constituents: pd.DataFrame) -> List[str]:
    col = "Symbol" if "Symbol" in constituents.columns else constituents.columns[0]
    return sorted({canonical_ticker(s) for s in constituents[col].dropna()})


def _parse_changes(changes: pd.DataFrame) -> List[Dict[str, object]]:
    """Flatten the changes table into sorted [{date, added, removed}] events."""
    df = changes.copy()
    _tick = ("Ticker", "Symbol")
    if isinstance(df.columns, pd.MultiIndex):
        date_col = [c for c in df.columns if "Date" in str(c[0])][0]
        add_col = [c for c in df.columns if c[0] == "Added" and any(t in str(c[1]) for t in _tick)]
        rem_col = [c for c in df.columns if c[0] == "Removed" and any(t in str(c[1]) for t in _tick)]
    else:
        date_col = [c for c in df.columns if "Date" in str(c)][0]
        add_col = [c for c in df.columns if "Added" in str(c)]
        rem_col = [c for c in df.columns if "Removed" in str(c)]
    events: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        try:
            d = pd.Timestamp(row[date_col], tz="UTC").normalize()
        except (ValueError, TypeError):
            continue
        if pd.isna(d):
            continue
        added = canonical_ticker(row[add_col[0]]) if add_col and pd.notna(row[add_col[0]]) else None
        removed = canonical_ticker(row[rem_col[0]]) if rem_col and pd.notna(row[rem_col[0]]) else None
        if added or removed:
            events.append({"date": d, "added": added or None, "removed": removed or None})
    events.sort(key=lambda e: e["date"])
    return events


def reconstruct_membership(
    start: str,
    end: str,
    *,
    freq: str = "MS",
    current: Optional[List[str]] = None,
    changes: Optional[List[Dict[str, object]]] = None,
) -> pd.DataFrame:
    """Date×ticker bool membership over [start, end], rolling the current set back
    through the change log. ``current``/``changes`` injectable for offline testing."""
    if current is None or changes is None:
        cons_tbl, chg_tbl = _fetch_wiki_tables()
        current = current if current is not None else _parse_current(cons_tbl)
        changes = changes if changes is not None else _parse_changes(chg_tbl)
    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize()
    grid = pd.DatetimeIndex(pd.date_range(start_ts, end_ts, freq=freq, tz="UTC")).normalize()

    # 1) Undo every change after `start` to recover the set AS OF start.
    set0 = set(current)
    post = [c for c in changes if c["date"] > start_ts]
    for ch in reversed(post):                       # newest -> oldest, undo each
        if ch["added"]:
            set0.discard(ch["added"])               # it wasn't a member before it was added
        if ch["removed"]:
            set0.add(ch["removed"])                 # it WAS a member before it was removed
    # 2) Walk forward, applying changes at/after their date.
    all_tickers = sorted(set(current) | {c["added"] for c in changes if c["added"]}
                         | {c["removed"] for c in changes if c["removed"]})
    membership = pd.DataFrame(False, index=grid, columns=all_tickers)
    cur = set(set0)
    ci = 0
    fwd = [c for c in changes if c["date"] > start_ts]
    for d in grid:
        while ci < len(fwd) and fwd[ci]["date"] <= d:
            ch = fwd[ci]
            if ch["added"]:
                cur.add(ch["added"])
            if ch["removed"]:
                cur.discard(ch["removed"])
            ci += 1
        present = [t for t in cur if t in membership.columns]
        membership.loc[d, present] = True
    return membership


def build_sp500_snapshot(
    start: str = "2012-01-01",
    end: Optional[str] = None,
    *,
    freq: str = "MS",
    current: Optional[List[str]] = None,
    changes: Optional[List[Dict[str, object]]] = None,
) -> UniverseSnapshot:
    """Build a survivorship-aware PIT ``UniverseSnapshot`` for the S&P 500."""
    end = end or datetime.now(timezone.utc).date().isoformat()
    membership = reconstruct_membership(start, end, freq=freq, current=current, changes=changes)
    uhash = compute_universe_hash(membership)
    return UniverseSnapshot(
        membership=membership,
        rules=UniverseRules(top_n=len(membership.columns), ranking_metric="sp500_wikipedia_pit"),
        built_at=datetime.now(timezone.utc).isoformat(),
        pipeline_version=EQUITY_PIPELINE_VERSION,
        universe_hash=uhash,
        reconstitution_dates=tuple(d.date().isoformat() for d in membership.index),
    )
