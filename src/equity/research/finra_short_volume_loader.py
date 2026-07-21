"""Point-in-time (PIT) loader for cached FINRA daily Reg SHO short-volume files.

Feeds the pre-registered short-volume signal experiment
(``docs/prereg-finra-short-volume-2026-07-04.md``, ``scripts/experiment_finra_short_volume.py``).
Parses the raw pipe-delimited CNMS files cached by
``scripts/fetch_finra_short_volume.py`` into a clean ``(date, ticker) ->
short_volume_ratio`` panel, restricted to tickers present in
``market_data/equity/sp500_prices.parquet``.

Raw file format (verified live 2026-07-04, one row per (date, symbol) — the
CNMS "consolidated" file already folds all Reg NMS markets into a single row
per symbol with ``Market`` as a comma-joined list, e.g. ``"B,Q,N"``, NOT one
row per market):

    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

PIT discipline (load-bearing, mirrors the inference-contract refusal rule
elsewhere in this repo — see ``.claude/rules/improvement.md`` "Train<->
Inference Contract Gates"):

* A given date's row may ONLY come from that date's own file. The date used
  for every row is re-derived from the FILENAME (``CNMSshvol{YYYYMMDD}.txt``)
  and cross-checked against the file's own embedded ``Date`` column — a
  mismatch is a contract violation and raises, it is never silently corrected
  or dropped.
* No forward-fill, no back-fill across dates. A ticker missing from a given
  day's file is simply absent from that day's panel row (NaN) — never
  imputed from a neighbouring day.
* The one-row-per-(date,symbol) assumption is VERIFIED against every parsed
  file, not assumed. If a file ever contains duplicate (date, symbol) rows,
  this loader raises ``FinraShortVolumeContractError`` rather than silently
  summing or dropping — that would be silently deciding an aggregation rule
  the pre-reg never specified.

OFFLINE only: reads already-cached files from disk. No network calls here —
network fetching is entirely ``scripts/fetch_finra_short_volume.py``'s job.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

RAW_FILENAME_RE = re.compile(r"^CNMSshvol(\d{8})\.txt$")

REQUIRED_COLUMNS = (
    "Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market",
)


class FinraShortVolumeContractError(RuntimeError):
    """Raised when a cached raw file violates the loader's PIT/shape contract."""


def _default_cache_dir(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "market_data" / "equity_alt" / "finra_short_volume" / "raw"


def _parse_one_file(path: Path) -> pd.DataFrame:
    """Parse one raw CNMS file, enforcing the filename<->embedded-date PIT check.

    Returns a DataFrame indexed by Symbol with columns short_volume,
    short_exempt_volume, total_volume, market — all rows guaranteed to belong
    to exactly the date encoded in the filename.
    """
    m = RAW_FILENAME_RE.match(path.name)
    if not m:
        raise FinraShortVolumeContractError(
            f"unrecognised raw filename {path.name!r} — expected "
            "CNMSshvol{YYYYMMDD}.txt"
        )
    filename_ymd = m.group(1)

    df = pd.read_csv(path, sep="|", dtype=str)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise FinraShortVolumeContractError(
            f"{path.name}: missing expected columns {missing_cols} "
            f"(got {list(df.columns)})"
        )

    # FINRA appends a trailing row-count line to each file (no delimiters —
    # just a bare integer). Verified on real 2019-01-02 data: last line is a
    # lone number, which pandas parses into a row with the Date column
    # populated (the count) and every other column NaN. Drop rows where every
    # column except Date is NaN BEFORE the PIT date check below, so this
    # known trailer format quirk is never mistaken for a PIT violation.
    non_date_cols = [c for c in df.columns if c != "Date"]
    trailer_mask = df[non_date_cols].isna().all(axis=1)
    if trailer_mask.any():
        df = df.loc[~trailer_mask].copy()

    # PIT check: every row's embedded Date must equal the filename's date.
    # A file whose content date doesn't match its own filename must never be
    # used for that day's cross-section — that would be silent cross-date
    # contamination (the exact failure mode this loader exists to prevent).
    bad_dates = df.loc[df["Date"] != filename_ymd, "Date"].unique()
    if len(bad_dates) > 0:
        raise FinraShortVolumeContractError(
            f"{path.name}: embedded Date column contains {list(bad_dates)[:5]} "
            f"which does not match the filename's date {filename_ymd} — "
            "refusing to use this file (PIT contract violation)"
        )

    # Verify the one-row-per-(date,symbol) assumption on REAL data rather than
    # assuming it. The CNMS ("consolidated") file is documented by FINRA as
    # already folding all Reg NMS markets into one row per symbol (Market is a
    # comma-joined list like "B,Q,N"). If duplicates ever appear, that
    # assumption is broken and we must not silently decide how to aggregate.
    dupe_mask = df.duplicated(subset=["Symbol"], keep=False)
    if dupe_mask.any():
        dupe_symbols = sorted(df.loc[dupe_mask, "Symbol"].unique())
        raise FinraShortVolumeContractError(
            f"{path.name}: {len(dupe_symbols)} symbols have duplicate rows "
            f"(e.g. {dupe_symbols[:5]}) — the one-row-per-(date,symbol) "
            "assumption is violated; refusing to silently sum or drop. "
            "This loader needs an explicit aggregation rule added before "
            "it can handle this file."
        )

    for col in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("Symbol")
    df = df.rename(columns={
        "ShortVolume": "short_volume",
        "ShortExemptVolume": "short_exempt_volume",
        "TotalVolume": "total_volume",
        "Market": "market",
    })
    return df[["short_volume", "short_exempt_volume", "total_volume", "market"]]


def _iter_cached_files(
    cache_dir: Path, dates: Optional[Iterable[pd.Timestamp]] = None
) -> List[Tuple[pd.Timestamp, Path]]:
    """List (date, path) pairs for cached raw files, optionally restricted to ``dates``."""
    wanted_ymd = None
    if dates is not None:
        wanted_ymd = {pd.Timestamp(d).strftime("%Y%m%d") for d in dates}

    out: List[Tuple[pd.Timestamp, Path]] = []
    for path in sorted(cache_dir.glob("CNMSshvol*.txt")):
        m = RAW_FILENAME_RE.match(path.name)
        if not m:
            continue
        ymd = m.group(1)
        if wanted_ymd is not None and ymd not in wanted_ymd:
            continue
        ts = pd.Timestamp(ymd, tz="UTC")
        out.append((ts, path))
    return out


def load_short_volume_ratio_panel(
    tickers: Iterable[str],
    *,
    cache_dir: Optional[Path] = None,
    dates: Optional[Iterable[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Build a date x ticker ``short_volume_ratio`` panel from cached raw files.

    ``short_volume_ratio = ShortVolume / TotalVolume`` per (date, ticker),
    restricted to ``tickers``. PIT-safe by construction: each date's row comes
    ONLY from that date's own cached file (see ``_parse_one_file``); there is
    no forward/back-fill across dates — a ticker/day with no row in the raw
    file is NaN, not imputed.

    Parameters
    ----------
    tickers:
        Universe to restrict columns to (e.g. ``sp500_prices.parquet``'s
        columns). Symbols outside this set are dropped per file before
        concatenation (keeps the panel small).
    cache_dir:
        Override for the raw cache directory (tests pass a ``tmp_path``).
    dates:
        Optional restriction to a specific set of dates; default is every
        cached file found in ``cache_dir``.

    Returns
    -------
    pd.DataFrame
        UTC ``DatetimeIndex`` (one row per date with a cached file), columns
        = ``tickers`` (only those actually present in at least one file),
        values = short_volume_ratio (NaN where absent that day).
    """
    cache_dir = cache_dir or _default_cache_dir()
    wanted = set(str(t) for t in tickers)
    files = _iter_cached_files(cache_dir, dates=dates)
    if not files:
        raise FinraShortVolumeContractError(
            f"no cached FINRA files found in {cache_dir} "
            "(run scripts/fetch_finra_short_volume.py first)"
        )

    rows: Dict[pd.Timestamp, pd.Series] = {}
    for ts, path in files:
        day_df = _parse_one_file(path)
        day_df = day_df[day_df.index.isin(wanted)]
        if day_df.empty:
            continue
        ratio = day_df["short_volume"] / day_df["total_volume"]
        ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).astype(float)
        rows[ts] = ratio

    if not rows:
        raise FinraShortVolumeContractError(
            f"parsed {len(files)} cached files but none contained any of the "
            f"{len(wanted)} requested tickers"
        )

    panel = pd.DataFrame(rows).T.sort_index()
    panel.index.name = "Date"
    panel = panel.reindex(columns=sorted(wanted))
    return panel


__all__ = [
    "FinraShortVolumeContractError",
    "load_short_volume_ratio_panel",
]
