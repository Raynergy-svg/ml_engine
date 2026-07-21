#!/usr/bin/env python3
"""CLI: Capture OANDA real-time tick stream.

PRACTICE-ONLY, read-only: this script never constructs an order/trade path —
it only pulls quotes off ``/pricing/stream`` via
:func:`src.data.tick_capture.build_practice_stream_client`, which hard-pins
the OANDA environment to practice regardless of ``OANDA_ENVIRONMENT``
(Hard NO #1). There is no ``--env`` flag: a flag that could point this
capture process at the live stream is exactly the kind of scaffolding
Hard NO #1 forbids, so the choice isn't exposed at all.

Examples:
    python scripts/run_tick_capture.py --pairs EUR_USD GBP_USD
    python scripts/run_tick_capture.py --pairs ALL_FX --buffer 5000
    python scripts/run_tick_capture.py --pairs EUR_USD --max-seconds 60   # bounded smoke test
    nohup python scripts/run_tick_capture.py --pairs ALL_FX >> logs/tick_capture.out 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import List

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tick_capture import TickCaptureDaemon, TickPersister, build_practice_stream_client
from src.data_platform.forward_capture import get_default_forward_capture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

ALL_MAJOR_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "EUR_CHF", "GBP_CHF", "AUD_JPY", "EUR_AUD", "GBP_AUD",
    "AUD_NZD", "CAD_JPY", "CHF_JPY", "EUR_CAD", "NZD_JPY",
]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture OANDA tick stream")
    parser.add_argument(
        "--pairs",
        type=str,
        default="EUR_USD",
        help="Comma-separated pairs or 'ALL_FX' (default: EUR_USD)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=10_000,
        help="In-memory tick buffer size before flush (default: 10_000)",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=30.0,
        help="Periodic flush interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trained_data/ticks"),
        help="Root directory for tick parquet partitions",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        default=True,
        help="Request initial snapshot before stream (default: True)",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_false",
        dest="snapshot",
        help="Skip initial snapshot",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=180,
        help="Prune tick partitions older than this many days (default: 180; 0 disables pruning)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Auto-shutdown after this many seconds (bounded smoke-test runs). "
             "Omit for an unbounded headless daemon.",
    )
    args = parser.parse_args(argv)

    # Resolve pairs
    if args.pairs.strip().upper() == "ALL_FX":
        pairs = ALL_MAJOR_PAIRS
    else:
        pairs = [p.strip().upper().replace("/", "_") for p in args.pairs.split(",")]

    logger.info(f"Tick capture starting for {len(pairs)} pair(s): {pairs}")
    logger.info(f"Output root: {args.output}")

    client = build_practice_stream_client()
    logger.info(
        "PRACTICE-ONLY, read-only pricing stream confirmed (base_url=%s)", client.base_url
    )
    daemon = TickCaptureDaemon(
        client=client,
        buffer_size=args.buffer,
        flush_interval_sec=args.flush_interval,
        retention_days=args.retention_days or None,
        persister=TickPersister(
            root=args.output,
            forward_capture=get_default_forward_capture(),
        ),
    )

    # Graceful shutdown on SIGINT/SIGTERM
    def _handler(signum, frame):
        logger.info(f"Received signal {signum}; shutting down gracefully...")
        daemon.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    if args.max_seconds:
        timer = threading.Timer(args.max_seconds, daemon.shutdown)
        timer.daemon = True
        timer.start()
        logger.info(f"Bounded run: will auto-shutdown after {args.max_seconds}s")

    try:
        daemon.start(pairs)
    except KeyboardInterrupt:
        daemon.shutdown()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
