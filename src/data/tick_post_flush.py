"""Post-flush tick aggregation hook.

US-002: After raw ticks are persisted to parquet, immediately aggregate
them to S5 and M15 candles and append to the canonical harvest files.
This ensures the training pipeline always sees fresh data without waiting
for a manual harvest run.
"""

from __future__ import annotations

import fcntl
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, List

import pandas as pd

from src.data.tick_aggregate import TICK_ROOT, aggregate_ticks_to_candles
from src.utils.oanda_streaming import TickQuote

logger = logging.getLogger(__name__)

HARVEST_ROOT = Path("trained_data/harvest")

# Pad the incremental aggregation window backwards by one bar-width so the
# bar straddling the batch start boundary is fully recomputed (and overwrites
# its partial prior value on merge via keep="last").
_GRAN_PAD = {
    "S5": timedelta(seconds=5),
    "S10": timedelta(seconds=10),
    "S15": timedelta(seconds=15),
    "S30": timedelta(seconds=30),
    "M1": timedelta(minutes=1),
    "M2": timedelta(minutes=2),
    "M5": timedelta(minutes=5),
    "M10": timedelta(minutes=10),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "H2": timedelta(hours=2),
    "H4": timedelta(hours=4),
    "D": timedelta(days=1),
}


def _harvest_lock_path(pq_path: Path) -> Path:
    """Canonical lock-file path for a harvest parquet.

    Shared contract: any writer of ``harvest/{pair}_{gran}.parquet`` (this
    hook AND src/data/harvest.py) must hold an fcntl.flock on this sibling
    ``.lock`` file for the read-modify-write to be safe. The lock file is a
    zero-byte sentinel; it is never read for content.
    """
    return pq_path.with_suffix(pq_path.suffix + ".lock")


def _atomic_merge_write(pq_path: Path, candles: pd.DataFrame) -> int:
    """Merge ``candles`` into ``pq_path`` atomically under a file lock.

    Reads the existing parquet (if any), concats, de-duplicates on the index
    keeping the latest value, sorts, then writes to a ``.tmp`` sibling and
    os.replace()s it into place (atomic on POSIX). The whole read-modify-write
    is serialized via fcntl.flock on the canonical lock file so a concurrent
    harvest/training writer cannot observe a torn or half-written parquet.

    Returns the number of rows in the final merged frame.
    """
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _harvest_lock_path(pq_path)
    # tmp file in the same directory guarantees os.replace is atomic (same fs).
    tmp_path = pq_path.with_suffix(pq_path.suffix + f".tmp.{os.getpid()}")

    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if pq_path.exists():
            existing = pd.read_parquet(pq_path)
            merged = pd.concat([existing, candles])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()
        else:
            merged = candles
        merged.to_parquet(tmp_path, compression="zstd")
        os.replace(tmp_path, pq_path)
        return len(merged)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def create_post_flush_aggregator(
    pair: str,
    granularities: List[str] = ("S5", "M15"),
) -> Any:
    """Factory for a post-flush callback that aggregates ticks to candles.

    Args:
        pair: Instrument code (e.g. "EUR_USD"). Used only as a default label;
            the actual instrument(s) aggregated are derived from the flushed
            batch, which may contain ticks for several instruments.
        granularities: List of granularities to aggregate and append.

    Returns:
        Callable[[List[TickQuote]], None] suitable for TickPersister.on_flush.
        The returned callable carries a mutable ``error_count`` attribute and
        a ``last_error`` attribute so a persistent aggregation failure is
        observable to callers/monitoring rather than silently swallowed.
    """
    def _aggregate(ticks: List[TickQuote]) -> None:
        if not ticks:
            return

        # Build a temporary DataFrame from ticks
        df = pd.DataFrame(
            [
                {
                    "instrument": t.instrument,
                    "time": t.time,
                    "bid": t.bid,
                    "ask": t.ask,
                    "mid": t.mid,
                    "bid_liq": t.bid_liq,
                    "ask_liq": t.ask_liq,
                    "status": t.status,
                    "source": t.source,
                }
                for t in ticks
            ]
        )
        if df.empty:
            return

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()

        # Aggregate per instrument actually present in this batch — the batch
        # may carry ticks for instruments other than `pair`. Using the batch
        # time span as [start, end] bounds keeps each aggregation incremental
        # instead of rescanning the entire tick history every flush.
        for instrument in df["instrument"].dropna().unique():
            inst_df = df[df["instrument"] == instrument]
            if inst_df.empty:
                continue
            batch_min = inst_df.index.min()
            batch_max = inst_df.index.max()

            for granularity in granularities:
                pad = _GRAN_PAD.get(granularity, timedelta(0))
                # Inclusive end bound: aggregate_ticks_to_candles treats end as
                # exclusive, so push it just past the last tick.
                start_iso = (batch_min - pad).isoformat()
                end_iso = (batch_max + timedelta(seconds=1)).isoformat()
                try:
                    candles = aggregate_ticks_to_candles(
                        instrument=instrument,
                        granularity=granularity,
                        start=start_iso,
                        end=end_iso,
                        root=TICK_ROOT,
                    )
                    if candles is None or candles.empty:
                        continue

                    pq_path = HARVEST_ROOT / f"{instrument}_{granularity}.parquet"
                    total = _atomic_merge_write(pq_path, candles)
                    logger.debug(
                        "Post-flush: appended %d %s candles for %s "
                        "(harvest total=%d)",
                        len(candles), granularity, instrument, total,
                    )
                except Exception as e:
                    # A swallowed-and-forgotten WARNING here is how US-002
                    # silently died before. Surface a counter + last_error so a
                    # persistent failure is visible to monitoring, and log at
                    # ERROR (not WARNING) since this breaks the always-fresh
                    # data guarantee the SOTA training trigger depends on.
                    _aggregate.error_count += 1
                    _aggregate.last_error = repr(e)
                    logger.error(
                        "Post-flush aggregation FAILED for %s %s "
                        "(error_count=%d): %s",
                        instrument, granularity, _aggregate.error_count, e,
                    )

    _aggregate.error_count = 0
    _aggregate.last_error = None
    return _aggregate
