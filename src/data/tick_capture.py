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

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.oanda_streaming import (
    PRACTICE_STREAM_URL,
    OandaStreamClient,
    StreamBuffer,
    TickQuote,
)

logger = logging.getLogger(__name__)

TICK_ROOT = Path("trained_data/ticks")
TICK_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_BUFFER_SIZE = 10_000
DEFAULT_FLUSH_INTERVAL_SEC = 30.0
DEFAULT_RETENTION_DAYS = 180  # bounded storage; None disables pruning
DEFAULT_GAP_THRESHOLD_SEC = 20.0  # idle longer than this between ticks = a logged gap
_PRUNE_MIN_INTERVAL_SEC = 3600.0  # don't re-walk the partition tree more than hourly


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically (tmp + rename) per repo JSON-safety convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def build_practice_stream_client() -> OandaStreamClient:
    """Build an OANDA stream client hard-pinned to practice.

    Tick capture is a read-only research feed (``/pricing/stream`` only — no
    order/trade endpoint exists on this client). Hard NO #1 (practice-only)
    is enforced unconditionally here regardless of ``OANDA_ENVIRONMENT``,
    because this factory must never become a path that could stream (or
    later be repurposed to trade) against the live account.
    """
    api_token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    if not api_token or not account_id:
        raise RuntimeError(
            "Missing OANDA env vars: set OANDA_API_TOKEN (or OANDA_API_KEY) "
            "and OANDA_ACCOUNT_ID."
        )
    client = OandaStreamClient(
        account_id=account_id, api_token=api_token, environment="practice"
    )
    # Explicit check, not `assert` -- assert statements are stripped entirely
    # under `python -O`/`PYTHONOPTIMIZE`, which would silently remove this
    # Hard-NO practice-only guard rather than skip a debug convenience.
    if client.base_url != PRACTICE_STREAM_URL:
        raise RuntimeError("tick capture must never target the live stream URL")
    return client


class TickPersister:
    """Flush buffered ticks to partitioned Parquet."""

    def __init__(self, root: Path = TICK_ROOT, on_flush: Any = None):
        self.root = root
        self.on_flush = on_flush  # US-002: optional post-flush callback

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
        # US-002: Post-flush aggregation hook
        if self.on_flush is not None and written > 0:
            try:
                self.on_flush(ticks)
            except Exception as _hook_err:
                logger.warning("Tick aggregation hook failed: %s", _hook_err)
        return written

    def prune_older_than(self, retention_days: int) -> int:
        """Delete day-partition files older than ``retention_days``. Bounded storage.

        Partition age is read from the ``<instrument>/<year>/<month>/<day>.parquet``
        path itself (not mtime), so re-flushing an old partition never resurrects it
        past its true calendar age. Returns count of files removed.
        """
        if retention_days is None or retention_days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc).date().toordinal() - retention_days
        removed = 0
        if not self.root.exists():
            return 0
        for pq_path in self.root.glob("*/*/*/*.parquet"):
            try:
                day = int(pq_path.stem)
                month = int(pq_path.parent.name)
                year = int(pq_path.parent.parent.name)
                part_ordinal = datetime(year, month, day, tzinfo=timezone.utc).date().toordinal()
            except (ValueError, OSError):
                continue  # not a recognized partition file — leave it alone
            if part_ordinal < cutoff:
                try:
                    pq_path.unlink()
                    removed += 1
                except OSError as unlink_err:
                    logger.warning("Failed to prune %s: %s", pq_path, unlink_err)
        if removed:
            logger.info(f"Pruned {removed} tick partition(s) older than {retention_days}d")
        return removed


class TickCaptureDaemon:
    """Long-running daemon that captures OANDA tick stream."""

    def __init__(
        self,
        client: OandaStreamClient,
        *,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
        persister: Optional[TickPersister] = None,
        retention_days: Optional[int] = DEFAULT_RETENTION_DAYS,
        gap_threshold_sec: float = DEFAULT_GAP_THRESHOLD_SEC,
    ) -> None:
        self.client = client
        self.buffer = StreamBuffer(max_size=buffer_size)
        self.flush_interval = flush_interval_sec
        self.persister = persister or TickPersister()
        self.retention_days = retention_days
        self.gap_threshold_sec = gap_threshold_sec
        self._shutdown = False
        self._flush_thread: Optional[threading.Thread] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stats: Dict[str, Any] = {
            "ticks_received": 0,
            "ticks_written": 0,
            "flushes": 0,
            "started_at": None,
        }
        # US-011: Health tracking
        self._last_tick_time: Optional[float] = None
        self._ticks_this_minute: int = 0
        self._last_minute_ts: float = 0.0
        self._health_alerts: int = 0
        self._stream_status: str = "starting"  # starting | running | stalled | dead
        # Gap honesty: every idle period longer than gap_threshold_sec is logged,
        # not silently absorbed — downstream consumers of the tick archive need to
        # know which windows have no coverage before trusting spread/slippage
        # stats computed across them.
        self._gaps: List[Dict[str, Any]] = []
        self._gaps_path = self.persister.root / "_gaps.jsonl"
        self._health_path = self.persister.root / "_health.json"
        self._last_prune_ts: float = 0.0

    @classmethod
    def from_env(cls, **kwargs) -> "TickCaptureDaemon":
        client = OandaStreamClient.from_env()
        return cls(client, **kwargs)

    @classmethod
    def from_env_practice(cls, **kwargs) -> "TickCaptureDaemon":
        """Build a daemon whose client is hard-pinned to practice (Hard NO #1).

        Prefer this over :meth:`from_env` for any standalone/headless capture
        entrypoint — it ignores ``OANDA_ENVIRONMENT`` entirely rather than
        trusting a process-wide env var that other tooling may set.
        """
        return cls(build_practice_stream_client(), **kwargs)

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
        self._stream_status = "dead"
        self.client.shutdown()
        time.sleep(0.5)
        self._do_flush()

    def get_health(self) -> Dict[str, Any]:
        """US-011: Return health metrics for monitoring."""
        now = time.time()
        idle_seconds = (now - self._last_tick_time) if self._last_tick_time else 999.0
        # Determine stream status heuristically
        if self._stream_status == "starting":
            status = "starting"
        elif idle_seconds > 15.0:
            status = "stalled"
        elif idle_seconds > 5.0:
            status = "slow"
        else:
            status = "running"
        # US-011 / neural-14: the per-minute counter only resets when a new
        # tick arrives, so during a stall it retains its last value and lies
        # about throughput. Report 0 once the stream has been idle past the
        # 60s window — no tick has arrived to fill the current minute.
        ticks_per_minute = 0 if idle_seconds > 60.0 else self._ticks_this_minute
        return {
            "stream_status": status,
            "ticks_received_total": self._stats["ticks_received"],
            "ticks_per_minute": ticks_per_minute,
            "idle_seconds": round(idle_seconds, 1),
            "last_tick_time": self._last_tick_time,
            "flushes": self._stats["flushes"],
            "ticks_written": self._stats["ticks_written"],
            "started_at": self._stats["started_at"],
            "gaps_detected": len(self._gaps),
            "last_gap": self._gaps[-1] if self._gaps else None,
            "retention_days": self.retention_days,
        }

    def _record_gap(self, gap_start: float, gap_end: float) -> None:
        """Append an honest gap record (in-memory + append-only sidecar)."""
        entry = {
            "gap_start": datetime.fromtimestamp(gap_start, tz=timezone.utc).isoformat(),
            "gap_end": datetime.fromtimestamp(gap_end, tz=timezone.utc).isoformat(),
            "duration_sec": round(gap_end - gap_start, 1),
        }
        self._gaps.append(entry)
        if len(self._gaps) > 500:
            self._gaps = self._gaps[-500:]
        try:
            self._gaps_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._gaps_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as gap_write_err:
            logger.warning("Failed to persist tick-capture gap record: %s", gap_write_err)
        logger.warning(
            "Tick stream gap detected: %.1fs idle (%s -> %s)",
            entry["duration_sec"], entry["gap_start"], entry["gap_end"],
        )

    def _stream_loop(self, instruments: List[str]) -> None:
        """Consume stream and buffer ticks."""
        self._stream_status = "running"
        for tick in self.client.stream_ticks(instruments, snapshot=True):
            if self._shutdown:
                break
            now = time.time()
            if self._last_tick_time is not None and (now - self._last_tick_time) > self.gap_threshold_sec:
                self._record_gap(self._last_tick_time, now)
            self._stats["ticks_received"] += 1
            self._last_tick_time = now
            # US-011: Per-minute tick counter
            if now - self._last_minute_ts >= 60.0:
                self._last_minute_ts = now
                self._ticks_this_minute = 1
            else:
                self._ticks_this_minute += 1
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
        # Bounded storage: prune at most once per hour, decoupled from the
        # (possibly frequent) tick-flush cadence.
        now = time.time()
        if self.retention_days and (now - self._last_prune_ts) >= _PRUNE_MIN_INTERVAL_SEC:
            self._last_prune_ts = now
            try:
                self.persister.prune_older_than(self.retention_days)
            except Exception as prune_err:
                logger.warning("Tick partition pruning failed: %s", prune_err)
        try:
            _atomic_write_json(self._health_path, self.get_health())
        except OSError as health_write_err:
            logger.warning("Failed to persist tick-capture health sidecar: %s", health_write_err)
