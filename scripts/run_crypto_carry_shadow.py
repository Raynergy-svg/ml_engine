#!/usr/bin/env python3
"""crypto_carry -- SHADOW-ONLY live driver for the cash-and-carry lane
(accumulates a genuine forward out-of-sample track record).

HARD LINE: this script places ZERO orders and constructs NO exchange/broker
client -- grep confirms no `src.brokers` / `execution.py` / `place_*_order`
reference anywhere in this file or in `src.crypto.carry_shadow`. It respects
the `crypto_carry` per-lane halt (fail-closed, same contract as every other
lane) and re-checks halt every cycle in `--loop` mode, so a re-halt stops the
lane before its next tick -- same pattern as
`scripts/run_crypto_momentum_shadow.py`.

The construction itself (positive-funding-only cash-and-carry, delta-neutral
by construction, vol-targeted) is reused VERBATIM from the pre-registered
harness (see `src/crypto/carry_shadow.py` docstring) -- this script never
tunes or re-derives it. It FAILED the ship gate on turnover cost
(docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md section 5) -- this
lane exists to let the live forward record answer whether a real deployment
fares differently, not because the signal is verified. This is a risk
premium with a real exchange-solvency/liquidation tail, not free money.

Usage::

    python scripts/run_crypto_carry_shadow.py                 # one shadow cycle
    python scripts/run_crypto_carry_shadow.py --refresh        # force a fresh data pull
    python scripts/run_crypto_carry_shadow.py --loop --interval 86400
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

from src.crypto import carry_shadow as cs  # noqa: E402
# Reused as-is: the equity harvester's fail-closed per-lane halt reader is
# generic over (root, lane) -- no equity-specific logic -- so this lane
# reuses it directly rather than duplicating the fail-closed contract.
from src.equity.decision_gate import _lane_halted  # noqa: E402

logger = logging.getLogger("run_crypto_carry_shadow")

LANE = "crypto_carry"


def _tick(project_root: Path, refresh: bool) -> dict:
    halted, readable, reason = _lane_halted(project_root, LANE)
    if halted:
        result = {"ran": False, "reason": f"halted:{reason}", "readable": readable, "orders": 0}
        print(f"CYCLE_RESULT: ran=False reason={result['reason']} lane={LANE} orders=0", flush=True)
        return result

    cycle_result = cs.compute_shadow_cycle(refresh_klines=refresh)
    cycle_ts_iso = datetime.now(timezone.utc).isoformat()
    row = cs.record_shadow_cycle(cycle_result, cycle_ts_iso=cycle_ts_iso)
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
        "asof_date": row["asof_date"], "n_longs": row["n_longs"], "n_shorts": row["n_shorts"],
        "today_net_return": row["today_net_return"],
        "cumulative_shadow_return": row["cumulative_shadow_return"],
        "forward_cycle_seq": row["forward_cycle_seq"],
    }
    print(f"CYCLE_RESULT: ran=True reason=recorded lane={LANE} orders=0 "
          f"asof={row['asof_date']} longs={row['n_longs']} shorts={row['n_shorts']} "
          f"today_net_return={row['today_net_return']:+.5f} "
          f"cumulative_shadow_return={row['cumulative_shadow_return']:+.5f} "
          f"forward_cycle_seq={row['forward_cycle_seq']}", flush=True)
    return result


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true",
                    help="force a fresh Binance data pull (default: use cache; the "
                         "monthly-dump source means repeated cycles are often no-ops "
                         "without this flag -- see module docstring)")
    ap.add_argument("--loop", action="store_true", help="run cycles repeatedly (operator decision to invoke)")
    ap.add_argument("--interval", type=float, default=86400.0,
                    help="seconds between cycles when --loop (default 1 day -- matches "
                         "the underlying data source's real cadence)")
    ap.add_argument("--max-cycles", type=int, default=0, help="stop after N cycles in --loop mode (0 = run until killed)")
    ap.add_argument("--project-root", type=str, default=str(REPO_ROOT))
    args = ap.parse_args(argv)

    project_root = Path(args.project_root)
    assert project_root.exists(), f"project root not found: {project_root}"

    n = 0
    while True:
        result = _tick(project_root, args.refresh)
        n += 1
        if not args.loop:
            return 0 if result["ran"] else 1
        if args.max_cycles and n >= args.max_cycles:
            logger.info("reached --max-cycles=%d, exiting", args.max_cycles)
            return 0
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
