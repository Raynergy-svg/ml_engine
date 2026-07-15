#!/usr/bin/env python3
"""Refresh tracked Binance kline/funding caches into immutable capture versions.

The default universe is exactly the symbols already present in the
survivorship-aware local cache. This is data ingestion only: no strategy,
portfolio, broker, exchange-account, or order code is imported.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crypto import data_layer as dl  # noqa: E402

logger = logging.getLogger("refresh_crypto_forward_data")


def discover_cached_symbols(cache_root: Path = dl.CACHE_DIR) -> tuple[str, ...]:
    paths = [
        cache_root / "binance" / "funding",
        cache_root / "binance" / "klines" / "1d",
    ]
    return tuple(sorted({path.stem for root in paths for path in root.glob("*.parquet")}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", help="comma-separated override; default=tracked cache universe")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--limit", type=int, default=0, help="bounded operational smoke-test limit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.symbols:
        symbols = tuple(sorted({item.strip().upper() for item in args.symbols.split(",") if item.strip()}))
    else:
        symbols = discover_cached_symbols()
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols:
        logger.error("no cached crypto symbols discovered; refusing an empty success")
        return 2

    completed = failed = 0
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            dl.fetch_binance_klines(symbol, args.interval, refresh=True)
            dl.fetch_binance_funding(symbol, refresh=True)
            completed += 1
        except Exception as exc:
            failed += 1
            errors[symbol] = f"{type(exc).__name__}: {exc}"
            logger.exception("crypto refresh failed for %s", symbol)
    summary = {
        "completed": completed,
        "failed": failed,
        "interval": args.interval,
        "requested": len(symbols),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if completed > 0 and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
