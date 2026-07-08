"""Calendar-date walk-forward split for the pooled multi-pair risk-target
universe.

A per-pair ROW-INDEX split (like `modular_data_loaders.temporal_split`) is
wrong for a pooled cross-pair panel: FX majors share correlated global vol
regimes (e.g. the 2020 COVID shock hit every pair the same week), so
splitting by row position per pair could let one pair's "train" fold
overlap in CALENDAR TIME with another pair's "test" fold. This module
splits on the shared calendar-date axis instead, with an embargo gap (in
trading days) reserved — and DROPPED, not reassigned — on both sides of the
train/val boundary and the val/OOS boundary, so no row's forward-label
window (which reads `H` bars ahead) can read across a split boundary.
"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np
import pandas as pd


def compute_global_date_buckets(
    dates: pd.Series,
    oos_start: str,
    val_frac: float = 0.2,
    embargo_bars: int = 20,
) -> Tuple[Set[pd.Timestamp], Set[pd.Timestamp], Set[pd.Timestamp]]:
    """Partition the UNIQUE calendar dates in `dates` into train/val/test.

    Args:
        dates: a Series of (possibly duplicated, one per pair) datetime64
            values across the pooled universe.
        oos_start: ISO date string; all dates >= this are OOS/test, read
            once and never touched by the train/val split.
        val_frac: fraction of the pre-OOS ("IS") date range reserved for
            validation (early-stopping), applied to USABLE IS dates (after
            the embargo trim below).
        embargo_bars: number of trading dates dropped (not reassigned) on
            each side of the train/val and val/OOS boundaries.

    Returns:
        (train_dates, val_dates, test_dates) — each a set of
        `pd.Timestamp`. Dates falling inside an embargo gap belong to NONE
        of the three sets (by design — they are excluded, not train).
    """
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates.unique(), utc=True)))
    oos_start_ts = pd.Timestamp(oos_start, tz="UTC")

    is_dates = unique_dates[unique_dates < oos_start_ts]
    test_dates = unique_dates[unique_dates >= oos_start_ts]

    m = len(is_dates)
    gap = int(embargo_bars)
    train_n = int(m * (1.0 - val_frac))

    train_end = max(train_n - gap, 0)
    val_start = train_n
    val_end = max(m - gap, val_start)

    train_dates = is_dates[:train_end]
    val_dates = is_dates[val_start:val_end]

    return set(train_dates), set(val_dates), set(test_dates)


def rows_in_bucket(row_dates: pd.Series, bucket: Set[pd.Timestamp]) -> np.ndarray:
    """Row-position indices (0-based, into `row_dates`'s own frame) whose
    date falls in `bucket`."""
    normalized = pd.to_datetime(row_dates, utc=True)
    mask = normalized.isin(bucket)
    return np.where(mask.values)[0]


__all__ = ["compute_global_date_buckets", "rows_in_bucket"]
