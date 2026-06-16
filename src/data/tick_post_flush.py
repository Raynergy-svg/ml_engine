"""Post-flush tick aggregation hook.

US-002: After raw ticks are persisted to parquet, immediately aggregate
them to S5 and M15 candles and append to the canonical harvest files.
This ensures the training pipeline always sees fresh data without waiting
for a manual harvest run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.data.tick_aggregate import aggregate_ticks_to_candles
from src.utils.oanda_streaming import TickQuote

logger = logging.getLogger(__name__)

HARVEST_ROOT = Path("trained_data/harvest")


def create_post_flush_aggregator(
    pair: str,
    granularities: List[str] = ("S5", "M15"),
) -> Any:
    """Factory for a post-flush callback that aggregates ticks to candles.

    Args:
        pair: Instrument code (e.g. "EUR_USD").
        granularities: List of granularities to aggregate and append.

    Returns:
        Callable[[List[TickQuote]], None] suitable for TickPersister.on_flush.
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

        for granularity in granularities:
            try:
                candles = aggregate_ticks_to_candles(
                    instrument=pair,
                    granularity=granularity,
                    start=None,
                    end=None,
                    root=None,  # uses default tick root
                )
                if candles is None or candles.empty:
                    continue

                pq_path = HARVEST_ROOT / f"{pair}_{granularity}.parquet"
                pq_path.parent.mkdir(parents=True, exist_ok=True)

                if pq_path.exists():
                    existing = pd.read_parquet(pq_path)
                    merged = pd.concat([existing, candles])
                    merged = merged[~merged.index.duplicated(keep="last")]
                    merged = merged.sort_index()
                else:
                    merged = candles

                merged.to_parquet(pq_path, compression="zstd")
                logger.debug(
                    "Post-flush: appended %d %s candles for %s",
                    len(candles), granularity, pair,
                )
            except Exception as e:
                logger.warning("Post-flush aggregation failed for %s %s: %s", pair, granularity, e)

    return _aggregate
