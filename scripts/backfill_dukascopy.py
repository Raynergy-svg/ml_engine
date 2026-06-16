#!/usr/bin/env python3
"""Bulk backfill Dukascopy tick data for training.

Examples:
    python scripts/backfill_dukascopy.py --pair EUR_USD --start 2023-01-01 --end 2023-01-31
    python scripts/backfill_dukascopy.py --pair EUR_USD --start 2020-01-01 --end 2023-12-31 --output trained_data/dukascopy
    python scripts/backfill_dukascopy.py --pair GBP_USD --start 2022-06-01 --end 2022-06-30 --convert-to-harvest
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.sources.dukascopy_harvester import (
    DukascopyHarvester,
    convert_dukascopy_to_harvest_format,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Dukascopy tick data")
    parser.add_argument("--pair", type=str, required=True, help="FX pair, e.g. EUR_USD")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="trained_data/dukascopy", help="Output directory")
    parser.add_argument("--convert-to-harvest", action="store_true", help="Also convert to M15 OHLCV harvest format")
    parser.add_argument("--chunks", type=int, default=1, help="Split into N monthly chunk files")
    args = parser.parse_args()

    pair_clean = args.pair.upper().replace("/", "_")
    pair_duka = pair_clean.replace("_", "")
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    logger.info("Backfilling %s from %s to %s", pair_clean, args.start, args.end)

    harvester = DukascopyHarvester(pair_duka, save_dir=Path(args.output))
    df = harvester.download_range(start_dt, end_dt)

    if df.empty:
        logger.error("No data downloaded. Check pair name and date range.")
        sys.exit(0)

    # Save raw tick parquet
    out_file = Path(args.output) / f"{pair_clean}_{args.start}_{args.end}_ticks.parquet"
    harvester.to_parquet(df, out_file)
    logger.info("Raw ticks saved: %s (%d rows)", out_file, len(df))

    # Convert to harvest format
    if args.convert_to_harvest:
        harvest_path = Path(args.output) / f"{pair_clean}_M15.parquet"
        convert_dukascopy_to_harvest_format(pair_clean, df, harvest_path)
        logger.info("Harvest M15 saved: %s", harvest_path)

    logger.info("Backfill complete: %d ticks, %d M15 bars", len(df), len(df) // 96)


if __name__ == "__main__":
    main()
