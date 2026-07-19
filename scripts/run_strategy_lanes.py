#!/usr/bin/env python
"""Recurring forward-capture runner (2026-07-18 readiness report, step 4).

One command per capture cycle: computes + records each strategy-owned shadow
lane's forward row (duplicate-asof rows are refused by each lane — running
this daily is safe on weekly/monthly strategies), then runs the universal
hedge lane cycle + scorecard/residual/promotion reports so every registered
strategy gets a current committed verdict.

SHADOW ONLY: zero orders, zero broker mutation. Everything this script
touches is a paper ledger or a report artifact. Residual rewards stay
governed by ``enable_residual_alpha_rewards`` (default OFF) — this script
never flips config.

Usage:
    python scripts/run_strategy_lanes.py            # cached data
    python scripts/run_strategy_lanes.py --refresh  # re-pull data sources first

Wire it to a daily scheduler (launchd/cron) on the operator machine; each
lane advances only when its data source actually has a new bar.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("run_strategy_lanes")


def _capture_crypto_ts_trend(refresh: bool) -> str:
    from src.crypto import ts_trend_shadow as ts
    result = ts.compute_shadow_cycle(refresh_klines=refresh)
    row = ts.record_shadow_cycle(
        result, cycle_ts_iso=datetime.now(timezone.utc).isoformat())
    if row is None:
        return f"crypto_ts_trend: no new bar (asof={result.asof_date}) — no-op"
    return (f"crypto_ts_trend: recorded cycle {row['forward_cycle_seq']} "
            f"asof={row['asof_date']} net={row['today_net_return']:.6f}")


def _capture_multi_asset_trend(refresh: bool) -> str:
    from src.equity import multi_asset_trend_lane as lane
    result = lane.compute_lane_cycle(refresh=refresh)
    row = lane.record_lane_cycle(
        result, cycle_ts_iso=datetime.now(timezone.utc).isoformat())
    if row is None:
        return f"multi_asset_trend: no new bar (asof={result.asof_date}) — no-op"
    return (f"multi_asset_trend: recorded cycle {row['forward_cycle_seq']} "
            f"asof={row['asof_date']} net={row['today_net_return']:.6f} "
            f"n_longs={row['n_longs']} lev={row['overlay_leverage']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull data sources before computing (yfinance / Binance dumps)")
    parser.add_argument("--skip-reports", action="store_true",
                        help="record lane cycles only; skip the hedge lane + report rebuild")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    failures = 0
    for name, fn in (("crypto_ts_trend", _capture_crypto_ts_trend),
                     ("multi_asset_trend", _capture_multi_asset_trend)):
        try:
            print(fn(args.refresh))
        except Exception as exc:  # noqa: BLE001 — one lane's data gap must not kill the others
            failures += 1
            logger.error("%s capture failed: %s", name, exc)

    if not args.skip_reports:
        try:
            from src.hedge.hedged_shadow_lane import run_all
            for strategy, row in run_all().items():
                status = "no data — skipped" if row is None else (
                    f"asof={row['asof_date']} hedge={row['hedge']['status']}")
                print(f"hedge lane {strategy}: {status}")
            from src.hedge.hedge_scorecard import main as rebuild_reports
            rebuild_reports()
            print("reports: scorecard + residual attribution + portfolio promotion rebuilt")
        except Exception as exc:  # noqa: BLE001 — report rebuild is fail-soft, lanes already recorded
            failures += 1
            logger.error("hedge lane / report rebuild failed: %s", exc)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
