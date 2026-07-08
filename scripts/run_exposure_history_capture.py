#!/usr/bin/env python3
"""Standalone exposure-history capture loop — the exposure-plane counterpart
of tick capture. SHADOW / ANALYSIS-ONLY: reads ``account_state.json`` + tick
parquets, appends ``trained_data/hedge/exposure_history.jsonl``. Imports
nothing from the trade/order plane — enforced structurally by
tests/test_hedge_exposure_history_2026_07_08.py.

Usage:
    python3 scripts/run_exposure_history_capture.py            # one capture
    python3 scripts/run_exposure_history_capture.py --loop 900 # every 15 min
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.hedge.exposure_history import (  # noqa: E402
    ACCOUNT_STATE_PATH,
    EXPOSURE_HISTORY_PATH,
    TICKS_ROOT,
    capture_exposure_snapshot,
)

logger = logging.getLogger("exposure_history_capture")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                        help="repeat every N seconds (0 = capture once and exit)")
    parser.add_argument("--account-state", type=Path, default=ACCOUNT_STATE_PATH)
    parser.add_argument("--ticks-root", type=Path, default=TICKS_ROOT)
    parser.add_argument("--out", type=Path, default=EXPOSURE_HISTORY_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    while True:
        row = capture_exposure_snapshot(account_state_path=args.account_state,
                                        ticks_root=args.ticks_root,
                                        out_path=args.out)
        if row is None:
            logger.info("no row appended (stale/duplicate/unresolvable — see warnings)")
        if args.loop <= 0:
            # one-shot: exit 0 either way — a refused capture is a valid
            # outcome (dedupe), not a failure; warnings carry the reason.
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
