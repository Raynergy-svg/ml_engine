#!/usr/bin/env python3
"""Equity cross-sectional factor test — is equities a different game than FX?

FX majors are a 52% efficient wall; the carry harvester only reaches Sharpe 0.63 with
36% DD. Equities have documented, persistent cross-sectional premia (momentum, low-vol,
value, quality) that FX lacks. This tests the PRICE-ONLY equity factors (momentum + low-
vol — no fundamentals needed) on a clean ETF universe (no single-name survivorship bias,
near-zero cost), through the SAME mechanical ship gate the harvester was held to.

Long-short (dollar-neutral) AND long-only top-tercile, vol-targeted, cost-aware, causal.
Benchmark = SPY buy-and-hold. Free data via yfinance.

This is the cheap first read. The richer version (fundamental factors: value/quality/PEAD
on single stocks) needs the financialdatasets.ai API key. Usage: python scripts/experiment_equity_factor.py
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

# Equity-only universe: 9 SPDR sectors + small-cap + 4 factor ETFs + intl/EM equity.
UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
            "IWM", "MTUM", "QUAL", "USMV", "VLUE", "EFA", "EEM"]
START, END = "2014-01-01", "2026-06-01"
ANN = 252
COST_BPS = 0.0002          # ~2bps ETF round-trip (vs 15-30bps EM FX)
TARGET_VOL = 0.10
GATE_SHARPE, GATE_MAXDD = 0.40, 0.25


def load_prices() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(UNIVERSE, start=START, end=END, interval="1d",
                     progress=False, auto_adjust=True)
    close = df["Close"].dropna(how="all")
    return close.dropna(axis=1, thresh=int(0.8 * len(close)))  # keep cols w/ >=80% history


def xs_rank(sig: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank -> [-1, 1], centered (NaN-safe)."""
    r = sig.rank(axis=1)
    n = r.count(axis=1)
    return r.sub((n + 1) / 2.0, axis=0).div((n - 1) / 2.0, axis=0).where(sig.notna())


def signals(close: pd.DataFrame) -> pd.DataFrame:
    mom = close.shift(21) / close.shift(252) - 1.0          # 12-1 month momentum
    ret = close.pct_change()
    lowvol = -ret.rolling(63).std()                          # low-vol anomaly (negate)
    combined = (xs_rank(mom) + xs_rank(lowvol)) / 2.0
    return combined.where(close.shift(252).notna())


def backtest(close, weights, *, cost=COST_BPS):
    ret = close.pct_change().reindex(columns=weights.columns)
    w_eff = weights.shift(1).fillna(0.0)                     # 1-day execution lag (causal)
    turn = w_eff.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (w_eff * ret).sum(axis=1) - turn * cost
    pnl = pnl.dropna()
    eq = (1 + pnl).cumprod()
    peak = eq.cummax(); dd = float(((peak - eq) / peak).max())
    sharpe = float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(ANN))
    cagr = float(eq.iloc[-1] ** (ANN / len(pnl)) - 1)
    per_year = {str(y): round(float((1 + g).prod() - 1), 4)
                for y, g in pnl.groupby(pnl.index.year)}
    return {"net_sharpe": round(sharpe, 3), "max_dd": round(dd, 3),
            "cagr": round(cagr, 4), "n_days": len(pnl), "per_year": per_year,
            "pos_years": f"{sum(v>0 for v in per_year.values())}/{len(per_year)}"}


def vol_target(raw_w, close):
    ret = close.pct_change()
    port = (raw_w.shift(1) * ret).sum(axis=1)
    rv = port.rolling(21).std().mul(np.sqrt(ANN)).shift(1).replace(0, np.nan)
    scale = (TARGET_VOL / rv).clip(upper=3.0).fillna(1.0)
    return raw_w.mul(scale, axis=0)


def main() -> int:
    t0 = time.time()
    close = load_prices()
    print(f"[equity] universe: {close.shape[1]} ETFs × {close.shape[0]} days "
          f"{close.index.min().date()}->{close.index.max().date()}")
    sig = signals(close)

    # Long-short dollar-neutral: demean cross-sectionally, gross-normalize, vol-target.
    ls = sig.sub(sig.mean(axis=1), axis=0)
    ls = ls.div(ls.abs().sum(axis=1), axis=0)
    ls_w = vol_target(ls, close)
    ls_stats = backtest(close, ls_w)

    # Long-only top tercile, equal-weight, vol-target.
    thr = sig.quantile(2 / 3.0, axis=1)
    lo = sig.ge(thr, axis=0).astype(float)
    lo = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    lo_w = vol_target(lo, close)
    lo_stats = backtest(close, lo_w)

    # SPY benchmark
    import yfinance as yf
    spy = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)["Close"]
    spy = spy.reindex(close.index).dropna() if isinstance(spy, pd.Series) else spy.iloc[:, 0].reindex(close.index).dropna()
    spyret = spy.pct_change().dropna()
    speq = (1 + spyret).cumprod()
    spy_stats = {"net_sharpe": round(float(spyret.mean()/(spyret.std()+1e-12)*np.sqrt(ANN)), 3),
                 "max_dd": round(float(((speq.cummax()-speq)/speq.cummax()).max()), 3),
                 "cagr": round(float(speq.iloc[-1]**(ANN/len(spyret))-1), 4)}

    print("\n=== EQUITY CROSS-SECTIONAL FACTOR (momentum + low-vol), price-only ===")
    print(f"gate: net Sharpe>={GATE_SHARPE} AND maxDD<={GATE_MAXDD:.0%}  (cost {COST_BPS*1e4:.0f}bps)")
    print(f"\n{'book':22s} {'net_Sh':>7s} {'maxDD':>7s} {'CAGR':>7s} {'+yrs':>7s}  gate")
    for name, s in (("long-short (neutral)", ls_stats), ("long-only top-tercile", lo_stats),
                    ("SPY buy & hold", spy_stats)):
        g = (s["net_sharpe"] >= GATE_SHARPE and s["max_dd"] <= GATE_MAXDD)
        py = s.get("pos_years", "-")
        print(f"{name:22s} {s['net_sharpe']:>7.3f} {s['max_dd']:>7.1%} {s['cagr']:>+7.1%} "
              f"{py:>7s}  {'PASS' if g else 'fail' if name!='SPY buy & hold' else '(bmk)'}")
    print("\nREAD: the question is whether a price-only EQUITY cross-section clears the SAME gate "
          "FX never could. If yes, equities is a categorically different market — and this is the "
          "WEAKEST equity test (no fundamentals; value/quality/PEAD on single names need the API key).")
    out = {"universe": list(close.columns), "span": f"{close.index.min().date()}->{close.index.max().date()}",
           "long_short": ls_stats, "long_only": lo_stats, "spy": spy_stats,
           "gate": {"min_net_sharpe": GATE_SHARPE, "max_dd": GATE_MAXDD},
           "ls_gate_pass": bool(ls_stats["net_sharpe"] >= GATE_SHARPE and ls_stats["max_dd"] <= GATE_MAXDD),
           "lo_gate_pass": bool(lo_stats["net_sharpe"] >= GATE_SHARPE and lo_stats["max_dd"] <= GATE_MAXDD),
           "elapsed_s": round(time.time()-t0, 1)}
    p = REPO_ROOT / "trained_data" / "backtests" / (
        f"equity_factor_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.rename(p)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
