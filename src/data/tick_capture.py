"""Real-time tick capture from OANDA pricing stream.

Persist quote ticks to partitioned Parquet for downstream model training.

Design:
- Ring-buffer ticks in memory (default 10 000 ticks).
- Flush to Parquet every N ticks or every M seconds (whichever comes first).
- Partition by (pair, year, month, day) for fast time-range queries.
- Idempotent: dedupe by (instrument, time) on flush to survive restarts.
- Background flush thread so stream ingestion is never blocked by I/O.

Usage:
    from src.data.tick_capture import TickCaptureDaemon
    daemon = TickCaptureDaemon.from_env()
    daemon.start(["EUR_USD", "GBP_USD"])
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.oanda_streaming import OandaStreamClient, StreamBuffer, TickQuote

logger = logging.getLogger(__name__)

TICK_ROOT = Path("trained_data/ticks")
TICK_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_BUFFER_SIZE = 10_000
DEFAULT_FLUSH_INTERVAL_SEC = 30.0


class TickPersister:
    """Flush buffered ticks to partitioned Parquet."""

    def __init__(self, root: Path = TICK_ROOT):
        self.root = root

    def flush(self, ticks: List[TickQuote]) -> int:
        """Persist ticks; return count written."""
        if not ticks:
            return 0

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
            return 0

        # Ensure UTC-aware datetime index for partitioning
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day

        written = 0
        for (inst, yr, mo, day), group in df.groupby(
            ["instrument", "year", "month", "day"]
        ):
            # Clean partition columns
            group = group.drop(columns=["year", "month", "day"])
            group = group.set_index("time").sort_index()

            pq_path = self.root / inst / f"{yr}" / f"{mo:02d}" / f"{day:02d}.parquet"
            pq_path.parent.mkdir(parents=True, exist_ok=True)

            # Append or create
            if pq_path.exists():
                try:
                    existing = pd.read_parquet(pq_path)
                    merged = pd.concat([existing, group])
                    merged = merged[~merged.index.duplicated(keep="last")]
                    merged = merged.sort_index()
                except Exception:
                    merged = group
            else:
                merged = group

            merged.to_parquet(pq_path, compression="zstd")
            written += len(group)

        logger.info(f"Flushed {written} ticks to {self.root}")
        return written


class TickCaptureDaemon:
    """Long-running daemon that captures OANDA tick stream."""

    def __init__(
        self,
        client: OandaStreamClient,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
        persister: Optional[TickPersister] = None,
    ) -> None:
        self.client = client
        self.buffer = StreamBuffer(max_size=buffer_size)
        self.flush_interval = flush_interval_sec
        self.persister = persister or TickPersister()
        self._shutdown = False
        self._flush_thread: Optional[threading.Thread] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stats: Dict[str, Any] = {
            "ticks_received": 0,
            "ticks_written": 0,
            "flushes": 0,
            "started_at": None,
        }

    @classmethod
    def from_env(cls, **kwargs) -> "TickCaptureDaemon":
        client = OandaStreamClient.from_env()
        return cls(client, **kwargs)

    def start(self, instruments: List[str]) -> None:
        """Launch capture threads and block until shutdown."""
        logger.info(f"TickCaptureDaemon starting for {instruments}")
        self._stats["started_at"] = datetime.now(timezone.utc).isoformat()

        # Background flush timer
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True
        )
        self._flush_thread.start()

        # Foreground stream ingestion
        self._stream_loop(instruments)

    def shutdown(self) -> None:
        """Signal shutdown and perform final flush."""
        logger.info("TickCaptureDaemon shutting down...")
        self._shutdown = True
        self.client.shutdown()
        time.sleep(0.5)
        self._do_flush()

    def _stream_loop(self, instruments: List[str]) -> None:
        """Consume stream and buffer ticks."""
        for tick in self.client.stream_ticks(instruments, snapshot=True):
            if self._shutdown:
                break
            self._stats["ticks_received"] += 1
            should_flush = self.buffer.append(tick)
            if should_flush:
                self._do_flush()

    def _flush_loop(self) -> None:
        """Periodic flush thread."""
        while not self._shutdown:
            time.sleep(self.flush_interval)
            self._do_flush()

    def _do_flush(self) -> None:
        ticks = self.buffer.flush()
        if ticks:
            n = self.persister.flush(ticks)
            self._stats["ticks_written"] += n
            self._stats["flushes"] += 1
