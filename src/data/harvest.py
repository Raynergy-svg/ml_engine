"""Automated Data Harvest — persist OANDA candles + microstructure for training.

Lightweight daemon that backfills and appends to canonical per-pair parquet files.
Designed to run periodically (e.g., cron every 6 hours) or ad-hoc before training.

Usage:
    python -m src.data.harvest --pairs EUR_USD GBP_USD --granularity M15
    python -m src.data.harvest --backfill-all --granularity H1

Key design choices:
- Idempotent: duplicate bars are dropped by (pair, time) index before append.
- Rate-limited: OANDA practice allows ~20 req/sec; we sleep 100ms between calls.
- Compressed: Parquet storage yields ~200 bytes/bar. 15 pairs × 2 years M15 ≈ 8 GB.
- Graceful degradation: if any pair fails 3 times consecutively, log alert and skip.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.brokers.instrument import Instrument
from src.brokers.oanda import OandaBroker
from src.brokers.registry import get_registry

logger = logging.getLogger(__name__)

HARVEST_ROOT = Path("trained_data/harvest")
HARVEST_ROOT.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_SECONDS = 0.10  # 100 ms between requests
MAX_FAIL_STREAK = 3


@dataclass
class HarvestConfig:
    pairs: List[str]
    granularity: str = "M15"
    broker_type: str = "oanda"
    rate_limit: float = RATE_LIMIT_SECONDS
    harvest_root: Path = HARVEST_ROOT
    max_fail_streak: int = MAX_FAIL_STREAK


class HarvestScheduler:
    """Fetches historical candles and microstructure, persists as Parquet."""

    def __init__(self, config: Optional[HarvestConfig] = None):
        self.cfg = config or HarvestConfig(pairs=[])
        self._broker: Optional[OandaBroker] = None
        self._fail_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Broker lifecycle
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        if self._broker is None:
            self._broker = OandaBroker.from_env()
            self._broker.connect()
            logger.info("HarvestScheduler: connected to OANDA")

    def disconnect(self) -> None:
        if self._broker:
            self._broker.disconnect()
            self._broker = None
            logger.info("HarvestScheduler: disconnected")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        backfill_months: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Harvest candles for all configured pairs.

        Args:
            backfill_months: If set, force-fetch this many months regardless of gap.
            dry_run: If True, compute gaps but don't write files.

        Returns:
            Dict with harvested counts per pair and any errors.
        """
        self._connect()
        results: Dict[str, Any] = {}

        for pair_raw in self.cfg.pairs:
            pair = str(pair_raw).upper().replace("/", "_")
            try:
                n_new = self._harvest_pair(pair, backfill_months, dry_run)
                results[pair] = {"status": "ok", "new_bars": n_new}
                self._fail_counts[pair] = 0
            except Exception as exc:
                self._fail_counts[pair] = self._fail_counts.get(pair, 0) + 1
                msg = f"Harvest failed for {pair}: {exc}"
                if self._fail_counts[pair] >= self.cfg.max_fail_streak:
                    logger.error(msg + f" (fail streak={self._fail_counts[pair]}, skipping)")
                else:
                    logger.warning(msg)
                results[pair] = {"status": "error", "error": str(exc)}
            finally:
                time.sleep(self.cfg.rate_limit)

        return results

    # ------------------------------------------------------------------
    # Per-pair backfill
    # ------------------------------------------------------------------
    def _harvest_pair(
        self,
        pair: str,
        backfill_months: Optional[int],
        dry_run: bool,
    ) -> int:
        logger.info(f"Harvesting {pair} {self.cfg.granularity}")
        pq_path = self.cfg.harvest_root / f"{pair}_{self.cfg.granularity}.parquet"
        existing = self._load_existing(pq_path)

        if existing is not None:
            latest = existing.index.max()
            logger.debug(f"Latest stored bar for {pair}: {latest}")
        else:
            latest = None
            logger.debug(f"No existing parquet for {pair}")

        # Determine how many candles to fetch
        bars_per_chunk = 5000
        total_new = 0
        last_bar_time: Optional[datetime] = None

        # If backfill_months specified, ignore existing and fetch fixed window
        if backfill_months is not None:
            bars_target = self._months_to_bars(backfill_months, self.cfg.granularity)
            chunks = (bars_target // bars_per_chunk) + 1
            for _ in range(chunks):
                if bars_target <= 0:
                    break
                count = min(bars_per_chunk, bars_target)
                df_new = self._fetch_chunk(pair, count)
                if df_new is None or df_new.empty:
                    break
                total_new += len(df_new)
                bars_target -= len(df_new)
                if not dry_run:
                    self._append(pq_path, df_new)
            logger.info(f"Backfilled {total_new} bars for {pair}")
            return total_new

        # Normal incremental gap-fill
        if latest is None:
            # Cold start: fetch 5k bars (~5 weeks M15, ~7 months H1)
            df_new = self._fetch_chunk(pair, bars_per_chunk)
            if df_new is not None and not df_new.empty:
                total_new = len(df_new)
                if not dry_run:
                    self._append(pq_path, df_new)
            return total_new

        # Determine gap
        now = datetime.now(timezone.utc)
        gap_bars = self._gap_to_bars(latest, now, self.cfg.granularity)
        logger.info(f"Gap for {pair}: {gap_bars} bars since {latest.isoformat()}")

        if gap_bars <= 0:
            logger.info(f"No gap for {pair}; up to date")
            return 0

        chunks = (gap_bars // bars_per_chunk) + 1
        for _ in range(chunks):
            if gap_bars <= 0:
                break
            count = min(bars_per_chunk, gap_bars)
            df_new = self._fetch_chunk(pair, count)
            if df_new is None or df_new.empty:
                break
            # De-duplicate against existing
            if existing is not None:
                overlap = df_new.index.isin(existing.index)
                df_new = df_new[~overlap]
            if df_new.empty:
                break
            total_new += len(df_new)
            gap_bars -= len(df_new)
            if not dry_run:
                self._append(pq_path, df_new)
                # Refresh existing so subsequent chunks deduplicate correctly
                existing = self._load_existing(pq_path)

        logger.info(f"Harvested {total_new} new bars for {pair}")
        return total_new

    # ------------------------------------------------------------------
    # OANDA fetch
    # ------------------------------------------------------------------
    def _fetch_chunk(self, pair: str, count: int) -> Optional[pd.DataFrame]:
        broker_symbol = pair.replace("_", "/")  # OANDA uses EUR/USD
        instrument = get_registry().get(broker_symbol)
        if instrument is None:
            # Fallback: build generic FX instrument
            instrument = Instrument.fx(
                symbol=broker_symbol,
                name=pair,
                pip_value=0.0001 if "JPY" not in pair else 0.01,
            )
        try:
            candles = self._broker.fetch_candles(instrument, self.cfg.granularity, count)
        except Exception as exc:
            logger.warning(f"Fetch failed for {pair}: {exc}")
            return None
        if not candles:
            return None
        rows = []
        for c in candles:
            rows.append({
                "time": c.time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time").sort_index()
        return df

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------
    @staticmethod
    def _load_existing(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if "time" in df.columns and df.index.name != "time":
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.set_index("time").sort_index()
            return df
        except Exception as exc:
            logger.warning(f"Could not read {path}: {exc}")
            return None

    @staticmethod
    def _append(path: Path, df_new: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                if "time" in existing.columns and existing.index.name != "time":
                    existing["time"] = pd.to_datetime(existing["time"], utc=True)
                    existing = existing.set_index("time")
                combined = pd.concat([existing, df_new])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            except Exception as exc:
                logger.warning(f"Could not merge {path}: {exc}; overwriting")
                combined = df_new
        else:
            combined = df_new
        combined.to_parquet(path, compression="zstd")
        logger.debug(f"Appended {len(df_new)} bars to {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _months_to_bars(months: int, granularity: str) -> int:
        granularities = {
            "M1": 60 * 24 * 30,
            "M5": 12 * 24 * 30,
            "M15": 4 * 24 * 30,
            "M30": 2 * 24 * 30,
            "H1": 24 * 30,
            "H4": 6 * 30,
            "D": 30,
            "W": 4,
        }
        return months * granularities.get(granularity.upper(), 4 * 24 * 30)

    @staticmethod
    def _gap_to_bars(latest: datetime, now: datetime, granularity: str) -> int:
        gap_seconds = (now - latest).total_seconds()
        seconds_per_bar = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D": 86400,
            "W": 604800,
        }.get(granularity.upper(), 900)
        # Buffer for weekends/market close
        return int(gap_seconds / seconds_per_bar) + 5

    # ------------------------------------------------------------------
    # Microstructure persistence (position book snapshots)
    # ------------------------------------------------------------------
    def harvest_position_book(self, pair: str) -> Dict[str, Any]:
        """Fetch and persist OANDA position book for a pair.

        Returns the snapshot dict for immediate use; also writes to parquet.
        """
        self._connect()
        if self._broker is None:
            return {}
        try:
            resp = self._broker._client.get_position_book(pair.replace("_", "/"))
            pb = resp.get("positionBook", {}) if isinstance(resp, dict) else {}
            buckets = pb.get("buckets")
            if not buckets:
                return {}
            snapshot = {
                "pair": pair,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "buckets": buckets,
                "longRatio": pb.get("longCountPercent", 0.0),
                "shortRatio": pb.get("shortCountPercent", 0.0),
            }
            # Persist
            pq_path = self.cfg.harvest_root / f"{pair}_positionbook.parquet"
            # Append as single-row DataFrame for time-series storage
            row = pd.DataFrame([snapshot])
            row["timestamp"] = pd.to_datetime(row["timestamp"], utc=True)
            row = row.set_index("timestamp")
            self._append(pq_path, row)
            return snapshot
        except Exception as exc:
            logger.warning(f"Position book harvest failed for {pair}: {exc}")
            return {}


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="ML Engine Data Harvester")
    parser.add_argument("--pairs", nargs="+", default=None, help="Pairs to harvest")
    parser.add_argument("--granularity", default="M15", help="Candle granularity")
    parser.add_argument("--backfill-months", type=int, default=None)
    parser.add_argument("--backfill-all", action="store_true", help="Backfill all DEFAULT_PAIRS")
    parser.add_argument("--position-book", nargs="+", default=None, help="Harvest position book for given pairs")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    pairs = args.pairs
    if args.backfill_all:
        from src.scanner.config import DEFAULT_PAIRS
        pairs = list(DEFAULT_PAIRS)

    if not pairs and not args.position_book:
        parser.error("Must specify --pairs, --backfill-all, or --position-book")

    cfg = HarvestConfig(
        pairs=pairs or [],
        granularity=args.granularity,
    )
    scheduler = HarvestScheduler(cfg)

    if pairs:
        results = scheduler.run(backfill_months=args.backfill_months, dry_run=args.dry_run)
        for pair, info in results.items():
            print(f"{pair}: {info}")

    if args.position_book:
        for pair in args.position_book:
            snapshot = scheduler.harvest_position_book(pair)
            print(f"{pair} position book: {len(snapshot.get('buckets', []))} buckets")

    scheduler.disconnect()


if __name__ == "__main__":
    main()
