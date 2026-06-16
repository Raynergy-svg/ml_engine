#!/usr/bin/env python3
"""Auto-trigger SOTA pre-training when Dukascopy backfill parquet appears.

Usage:
    python scripts/auto_pretrain_on_backfill.py --poll-interval 300
    nohup python scripts/auto_pretrain_on_backfill.py &
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.harness import train

logger = logging.getLogger(__name__)

TARGET_PARQUET = Path("trained_data/harvest/EUR_USD_M15_dukascopy.parquet")
CHECKPOINT_DIR = Path("trained_data/models/sota_pretrain")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-pretrain on backfill")
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds between checks")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logger.info("Waiting for %s ...", TARGET_PARQUET)
    while not TARGET_PARQUET.exists():
        logger.info("Parquet not found yet; sleeping %ds", args.poll_interval)
        time.sleep(args.poll_interval)

    logger.info("Parquet detected: %s", TARGET_PARQUET)
    result = train(
        model_type="sota",
        data_paths=[str(TARGET_PARQUET)],
        save_dir=str(CHECKPOINT_DIR),
        dry_run=args.dry_run,
        pretrain_only=True,
    )

    if result.get("status") == "ok":
        logger.info("Auto-pretrain succeeded: %s", result)
    else:
        logger.error("Auto-pretrain failed: %s", result.get("error"))
        sys.exit(1)


if __name__ == "__main__":
    main()
