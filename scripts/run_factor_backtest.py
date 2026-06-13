#!/usr/bin/env python
"""Run the daily FX factor-portfolio backtest and the mechanical ship gate.

Usage:
    python scripts/run_factor_backtest.py [--refresh] [--trend-only]

Produces the one number that matters (US-006) and the PASS/FAIL verdict (US-007).
Writes the backtest artifact + gate verdict to trained_data/backtests/ atomically.
No trading is performed; this is research/evaluation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.factor import PAIRS, ALL_PAIRS, FACTOR_PIPELINE_VERSION  # noqa: E402
from src.factor.data_loader import load_universe  # noqa: E402
from src.factor.rates import build_rate_diff_panel  # noqa: E402
from src.factor.signals import (  # noqa: E402
    tsmom_signal,
    carry_signal,
    combine_signals,
)
from src.factor.portfolio import (  # noqa: E402
    target_weights,
    apply_no_trade_band,
    enforce_guards,
    weekly_rebalance_mask,
)
from src.factor.backtest import run_backtest, write_report  # noqa: E402
from src.factor.ship_gate import evaluate_gate, verdict_to_dict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="force re-fetch from OANDA")
    ap.add_argument("--trend-only", action="store_true",
                    help="run trend (TSMOM) only")
    ap.add_argument("--no-carry", action="store_true",
                    help="skip the carry factor (FRED rate fetch)")
    ap.add_argument("--majors-only", action="store_true",
                    help="trade the 7 USD majors only (no G10 crosses)")
    args = ap.parse_args()

    universe = PAIRS if args.majors_only else ALL_PAIRS
    print(f"[factor] pipeline={FACTOR_PIPELINE_VERSION} "
          f"universe={'majors' if args.majors_only else 'majors+crosses'} "
          f"({len(universe)} instruments)")
    close, _open, dropped = load_universe(universe, refresh=args.refresh)
    if dropped:
        print(f"[factor] dropped {len(dropped)} (insufficient history): "
              f"{[d[0] for d in dropped]}")
    print(f"[factor] panel: {close.shape[1]} instruments × {close.shape[0]} "
          f"aligned days {close.index[0].date()} -> {close.index[-1].date()}")

    returns = close.pct_change()
    signals = []
    accrual = None
    if not args.no_carry:
        rd = build_rate_diff_panel(close.index, refresh=args.refresh)
        accrual = rd / 100.0 / 252.0  # daily fractional carry earned per pair
        if not args.trend_only:
            signals.append(carry_signal(rd))
    signals.append(tsmom_signal(close))
    combined = combine_signals(signals)

    raw_w = target_weights(combined, returns)
    rb = weekly_rebalance_mask(close.index)
    held_w = apply_no_trade_band(raw_w, band=0.05, rebalance_mask=rb)
    held_w = enforce_guards(held_w)  # hard gross/per-pair caps, never breached

    result = run_backtest(held_w, close, carry_accrual=accrual)
    report_path = write_report(result)

    report = json.loads(json.dumps(result.__dict__, default=lambda o: o))
    verdict = evaluate_gate(report)
    verdict_path = report_path.with_name(report_path.stem + "_gate.json")
    tmp = verdict_path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(verdict_to_dict(verdict), fh, indent=2, sort_keys=True)
    import os
    os.replace(tmp, verdict_path)

    book = "trend-only" if args.trend_only else (
        "trend (no carry)" if args.no_carry else "carry + trend")
    print(f"\n=== FACTOR PORTFOLIO BACKTEST ({book}) ===")
    print(f"  window         : {result.start} -> {result.end} "
          f"({result.total_years} years, {result.n_days} days)")
    print(f"  net Sharpe     : {result.net_sharpe:.3f}   "
          f"(gross {result.gross_sharpe:.3f})")
    print(f"  net CAGR       : {result.net_cagr:.2%}")
    print(f"  max drawdown   : {result.max_drawdown:.1%}")
    print(f"  positive years : {result.positive_years}/{result.total_years}")
    print(f"  annual turnover: {result.annual_turnover:.1f}x   "
          f"cost drag {result.annual_cost_drag:.2%}/yr")
    print("  per-year       :")
    for y, v in sorted(result.per_year_returns.items()):
        print(f"      {y}: {v:+.2%}")
    print("\n=== SHIP GATE (US-007) ===")
    for k, c in verdict.criteria.items():
        flag = "PASS" if c["passed"] else "FAIL"
        print(f"  [{flag}] {k}: {c['value']} (need {c['threshold']})")
    print(f"\n  VERDICT: {verdict.summary}")
    print(f"\n[factor] report  -> {report_path}")
    print(f"[factor] verdict -> {verdict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
