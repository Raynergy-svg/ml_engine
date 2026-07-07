#!/usr/bin/env python3
"""CLI: build the offline, advisory execution-cost model (P2).

Reads ORDER_FILL history (trained_data/oanda/transactions.jsonl) and captured
ticks (trained_data/ticks/, once populated), writes per-instrument
spread/slippage estimates to trained_data/cost_model/execution_cost_estimates.json,
and prints a holdout validation summary. Read-only against its inputs;
places no orders; never imports a broker client, execution.py, or engine.py
(it does import src.brokers.registry for static pip-value metadata only).

Usage:
    python scripts/build_execution_cost_model.py
    python scripts/build_execution_cost_model.py --instrument EUR_USD
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.execution_cost_model import (  # noqa: E402
    build_cost_model,
    load_fill_qualities,
    save_cost_model,
    validate_against_holdout,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the offline execution-cost model")
    parser.add_argument("--instrument", type=str, default=None, help="Limit to one instrument")
    parser.add_argument("--lookback-days", type=int, default=30, help="Tick lookback window (default: 30)")
    args = parser.parse_args(argv)

    instruments = [args.instrument] if args.instrument else None

    qualities = load_fill_qualities()
    logger.info(f"Loaded {len(qualities)} ORDER_FILL record(s)")

    model = build_cost_model(instruments, lookback_days=args.lookback_days)
    path = save_cost_model(model)
    logger.info(f"Wrote {len(model)} instrument estimate(s) to {path}")

    for inst, est in sorted(model.items()):
        print(
            f"{inst:8s} source={est.source:16s} n_fills={est.sample_count_fills:4d} "
            f"n_ticks={est.sample_count_ticks:6d} confidence={est.confidence:.2f} "
            f"spread_pips={est.spread_pips} slippage_pips={est.slippage_pips}"
        )
        for note in est.notes:
            print(f"           note: {note}")

    validation = validate_against_holdout(qualities)
    print("\n--- Holdout validation (train-mean slippage vs held-out realized) ---")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
