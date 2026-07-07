#!/usr/bin/env python3
"""track_b — SHADOW-ONLY live driver for the SEC filing-text research-alpha
lane (accumulates a genuine forward out-of-sample track record).

HARD LINE: this script places ZERO orders and constructs NO broker/execution
client — grep confirms no `src.brokers` / `execution.py` / `place_*_order`
reference anywhere in this file or in `src.equity.track_b_shadow`. It
respects the `track_b` per-lane halt (fail-closed, same contract as the
`oanda_fx` / `equity` / `brain` / `crypto_momentum` lanes) and re-checks halt
every cycle in `--loop` mode, so a re-halt stops the lane before its next
tick — same pattern as `scripts/run_crypto_momentum_shadow.py`.

The signal itself (LLM filing-text composite -> Q5 long-only book, vol-
targeted, ~monthly rebalance) is reused VERBATIM from the pre-registered,
verifier-confirmed harness (see `src/equity/track_b_shadow.py` docstring) —
this script never tunes or re-derives it. Its own binding verdict is
`overall_verdict=INSUFFICIENT` (underpowered, not a measured NO EDGE — see
`docs/adversarial-review-no-edge-verdicts-2026-07-02.md`) — this lane exists
to let the live forward record answer the open question, not because the
signal is verified.

Usage::

    python scripts/run_track_b_shadow.py                  # one shadow cycle (cached prices)
    python scripts/run_track_b_shadow.py --refresh-prices  # pull a fresh price panel via Yahoo
    python scripts/run_track_b_shadow.py --loop --interval 86400
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.equity import track_b_shadow as tbs  # noqa: E402
# Reused as-is: the equity harvester's fail-closed per-lane halt reader is
# generic over (root, lane) — no equity-specific logic — so this lane reuses
# it directly, same as the crypto momentum lane does.
from src.equity.decision_gate import _lane_halted  # noqa: E402

logger = logging.getLogger("run_track_b_shadow")

LANE = "track_b"


def _tick(project_root: Path, refresh_prices: bool) -> dict:
    halted, readable, reason = _lane_halted(project_root, LANE)
    if halted:
        result = {"ran": False, "reason": f"halted:{reason}", "readable": readable, "orders": 0}
        print(f"CYCLE_RESULT: ran=False reason={result['reason']} lane={LANE} orders=0", flush=True)
        return result

    try:
        cycle_result = tbs.compute_shadow_cycle(refresh_prices=refresh_prices)
        cycle_ts_iso = datetime.now(timezone.utc).isoformat()
        row = tbs.record_shadow_cycle(cycle_result, cycle_ts_iso=cycle_ts_iso)
    except (tbs.TrackBShadowError, OSError) as exc:
        logger.error("track_b shadow cycle failed: %s", exc, exc_info=True)
        result = {"ran": False, "reason": f"error:{exc}", "orders": 0}
        print(f"CYCLE_RESULT: ran=False reason=error lane={LANE} orders=0", flush=True)
        return result
    if row is None:
        result = {
            "ran": True, "reason": "no_new_trading_day", "orders": 0,
            "asof_date": cycle_result.asof_date,
        }
        print(f"CYCLE_RESULT: ran=True reason=no_new_trading_day lane={LANE} orders=0 "
              f"asof={cycle_result.asof_date}", flush=True)
        return result

    result = {
        "ran": True, "reason": "recorded", "orders": 0,
        "asof_date": row["asof_date"], "n_longs": row["n_longs"],
        "n_scored_filings": row["n_scored_filings"],
        "today_net_return": row["today_net_return"],
        "cumulative_shadow_return": row["cumulative_shadow_return"],
        "forward_cycle_seq": row["forward_cycle_seq"],
    }
    print(f"CYCLE_RESULT: ran=True reason=recorded lane={LANE} orders=0 "
          f"asof={row['asof_date']} longs={row['n_longs']} "
          f"n_scored_filings={row['n_scored_filings']} "
          f"today_net_return={row['today_net_return']:+.5f} "
          f"cumulative_shadow_return={row['cumulative_shadow_return']:+.5f} "
          f"forward_cycle_seq={row['forward_cycle_seq']}", flush=True)
    return result


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-prices", action="store_true",
                    help="pull a fresh price panel via src.equity.research.price_loader "
                         "(network-dependent) instead of the cached "
                         "market_data/equity/sp500_prices.parquet snapshot")
    ap.add_argument("--loop", action="store_true", help="run cycles repeatedly (operator decision to invoke)")
    ap.add_argument("--interval", type=float, default=86400.0,
                    help="seconds between cycles when --loop (default 1 day — matches the "
                         "underlying construction's ~monthly rebalance cadence being far "
                         "coarser than any useful polling interval)")
    ap.add_argument("--max-cycles", type=int, default=0, help="stop after N cycles in --loop mode (0 = run until killed)")
    ap.add_argument("--project-root", type=str, default=str(REPO_ROOT))
    args = ap.parse_args(argv)

    project_root = Path(args.project_root)
    assert project_root.exists(), f"project root not found: {project_root}"

    n = 0
    while True:
        result = _tick(project_root, args.refresh_prices)
        n += 1
        if not args.loop:
            return 0 if result["ran"] else 1
        if args.max_cycles and n >= args.max_cycles:
            logger.info("reached --max-cycles=%d, exiting", args.max_cycles)
            return 0
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
