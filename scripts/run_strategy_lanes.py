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


def _describe(name: str, appended: list) -> str:
    if not appended:
        return f"{name}: no unseen bar — honest no-op"
    if appended[0].get("kind") == "activation":
        return (f"{name}: ACTIVATED at asof={appended[0]['asof_date']} "
                "(baseline snapshot, no realized return — forward bars start next)")
    return (f"{name}: appended {len(appended)} forward bar(s) "
            f"{appended[0]['asof_date']} -> {appended[-1]['asof_date']} "
            f"(cum={appended[-1]['cumulative_shadow_return']:.6f})")


def _capture_crypto_momentum(refresh: bool) -> str:
    from src.crypto import momentum_shadow as h4
    appended = h4.capture_forward(
        refresh_klines=refresh, cycle_ts_iso=datetime.now(timezone.utc).isoformat())
    return _describe("crypto_momentum", appended)


def _capture_crypto_ts_trend(refresh: bool) -> str:
    from src.crypto import ts_trend_shadow as ts
    appended = ts.capture_forward(
        refresh_klines=refresh, cycle_ts_iso=datetime.now(timezone.utc).isoformat())
    return _describe("crypto_ts_trend", appended)


def _capture_multi_asset_trend(refresh: bool) -> str:
    from src.equity import multi_asset_trend_lane as lane
    appended = lane.capture_forward(
        refresh=refresh, cycle_ts_iso=datetime.now(timezone.utc).isoformat())
    return _describe("multi_asset_trend", appended)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull data sources before computing (yfinance / Binance dumps)")
    parser.add_argument("--skip-reports", action="store_true",
                        help="record lane cycles only; skip the hedge lane + report rebuild")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    failures = 0
    for name, fn in (("crypto_momentum", _capture_crypto_momentum),
                     ("crypto_ts_trend", _capture_crypto_ts_trend),
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
