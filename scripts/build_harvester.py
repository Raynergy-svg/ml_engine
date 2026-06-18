#!/usr/bin/env python3
"""The Harvester — a risk-premium engine (carry + trend) with a causal risk overlay.

This is the SECOND engine: it does not predict direction. It collects the carry
premium (long high-rate / short low-rate) and rides trends (TSMOM), then manages the
fat left tail that sank every prior premium book — via a portfolio-level vol target +
a drawdown / vol-spike de-grossing circuit-breaker.

Every premium book tested this session failed the ship gate on DRAWDOWN, not Sharpe.
So the overlay is the strategy. Rather than claim one setting "passes," this prints the
return-vs-drawdown FRONTIER across risk settings so the operator chooses the mandate.

Research/shadow only — NOT wired to live execution. Reuses src.factor (signals, vol-
targeted weights, faithful cost model) on FRED daily G10 majors (1999-2026).

Usage: python scripts/build_harvester.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.experiment_pre2014_factor import load_fred_close  # noqa: E402
from src.factor.rates import build_rate_diff_panel  # noqa: E402
from src.factor.signals import tsmom_signal, carry_signal, combine_signals  # noqa: E402
from src.factor.portfolio import (  # noqa: E402
    target_weights, apply_no_trade_band, enforce_guards, weekly_rebalance_mask,
)
from src.factor.backtest import run_backtest  # noqa: E402
from src.factor.ship_gate import MIN_NET_SHARPE, MAX_DRAWDOWN  # noqa: E402

ANN = 252


def base_book(close: pd.DataFrame):
    """Vol-targeted carry+trend weights + the daily carry accrual (pre-overlay)."""
    returns = close.pct_change()
    rd = build_rate_diff_panel(close.index, pairs=list(close.columns))
    accrual = rd / 100.0 / 252.0
    combined = combine_signals([carry_signal(rd), tsmom_signal(close)])
    raw = target_weights(combined, returns)
    rb = weekly_rebalance_mask(close.index)
    held = enforce_guards(apply_no_trade_band(raw, band=0.05, rebalance_mask=rb))
    return held, returns, accrual


def exposure_scalar(held, returns, accrual, *, target_vol, dd_soft, dd_hard,
                    vol_lookback=21, vol_spike_mult=2.0):
    """CAUSAL daily exposure multiplier in [0,1]: vol-target * drawdown-breaker.

    Derived from the BASE book's own gross pnl/equity using only info <= t-1 (shift(1)),
    so it cannot peek. Scales ALL positions down in high-vol or drawdown states — the
    carry-crash defense. Never levers up (cap 1.0)."""
    w_eff = held.shift(1).fillna(0.0)
    gross = (w_eff * returns).sum(axis=1) + (w_eff * accrual).sum(axis=1)
    # 1) portfolio vol target (de-risk only)
    rvol = gross.rolling(vol_lookback).std().mul(np.sqrt(ANN)).shift(1)
    s_vol = (target_vol / rvol).clip(upper=1.0).fillna(1.0)
    # 2) vol-spike kill: if realized vol > mult*target, cut hard
    s_spike = np.where(rvol > vol_spike_mult * target_vol, 0.3, 1.0)
    s_spike = pd.Series(s_spike, index=gross.index)
    # 3) drawdown breaker off base equity (causal, shift(1))
    eq = (1.0 + gross).cumprod()
    dd = (1.0 - eq / eq.cummax()).shift(1).fillna(0.0)
    s_dd = np.where(dd <= dd_soft, 1.0,
                    np.where(dd >= dd_hard, 0.0, (dd_hard - dd) / (dd_hard - dd_soft)))
    s_dd = pd.Series(s_dd, index=gross.index)
    return (s_vol * s_spike * s_dd).clip(0.0, 1.0)


def stats(close, held, accrual):
    r = run_backtest(held, close, carry_accrual=accrual)
    return {"net_sharpe": round(r.net_sharpe, 3), "max_dd": round(r.max_drawdown, 3),
            "net_cagr": round(r.net_cagr, 4), "turnover": round(r.annual_turnover, 1),
            "pos_years": f"{r.positive_years}/{r.total_years}"}


def main() -> int:
    t0 = time.time()
    close = load_fred_close(refresh=False)
    held, returns, accrual = base_book(close)
    base = stats(close, held, accrual)

    grid = [(tv, ds, dh) for tv in (0.06, 0.08, 0.10)
            for (ds, dh) in ((0.10, 0.20), (0.15, 0.25), (0.20, 0.35))]
    rows = [{"setting": "BASE (no overlay)", "target_vol": None, **base, "gate_pass":
             bool(base["net_sharpe"] >= MIN_NET_SHARPE and base["max_dd"] <= MAX_DRAWDOWN)}]
    for tv, ds, dh in grid:
        s = exposure_scalar(held, returns, accrual, target_vol=tv, dd_soft=ds, dd_hard=dh)
        managed = held.mul(s, axis=0)
        st = stats(close, managed, accrual)
        rows.append({"setting": f"vol{int(tv*100)}% dd{int(ds*100)}/{int(dh*100)}",
                     "target_vol": tv, **st,
                     "gate_pass": bool(st["net_sharpe"] >= MIN_NET_SHARPE and st["max_dd"] <= MAX_DRAWDOWN)})

    print("=== THE HARVESTER — carry+trend + risk overlay, FRED daily G10 majors ===")
    print(f"span {close.index[0].date()}->{close.index[-1].date()}  "
          f"gate: net Sharpe>={MIN_NET_SHARPE} AND maxDD<={MAX_DRAWDOWN:.0%}")
    print(f"\n{'setting':22s} {'net_Sh':>7s} {'maxDD':>7s} {'CAGR':>7s} {'turn':>6s} {'+yrs':>6s}  gate")
    for r in rows:
        print(f"{r['setting']:22s} {r['net_sharpe']:>7.3f} {r['max_dd']:>7.1%} "
              f"{r['net_cagr']:>+7.1%} {r['turnover']:>6.1f} {r['pos_years']:>6s}  "
              f"{'PASS' if r['gate_pass'] else 'fail'}")
    passers = [r for r in rows if r["gate_pass"]]
    print(f"\nsettings clearing the 25%-maxDD gate: {len(passers)}/{len(rows)}")
    if passers:
        best = max(passers, key=lambda r: r["net_sharpe"])
        print(f"best gate-passing: {best['setting']}  net Sharpe {best['net_sharpe']}  "
              f"maxDD {best['max_dd']:.1%}  CAGR {best['net_cagr']:+.1%}")
    print("\nREAD: this is the return/drawdown frontier. Pick the (vol target, dd-stop) that "
          "matches the drawdown you can stomach. The overlay trades return for tail safety; "
          "the operator owns the mandate. NOT wired to live — shadow/research only.")
    out = {"base": base, "frontier": rows, "gate": {"min_net_sharpe": MIN_NET_SHARPE, "max_dd": MAX_DRAWDOWN},
           "span": f"{close.index[0].date()}->{close.index[-1].date()}", "elapsed_s": round(time.time()-t0, 1)}
    p = REPO_ROOT / "trained_data" / "backtests" / (
        f"harvester_v1_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.rename(p)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
