#!/usr/bin/env python3
"""Round-3 LEAD A — higher-frequency (1h) intraday crypto edge test.

PRE-REGISTERED: docs/experiment-edge-hunt-round3-2026-06-29.md (frozen; committed BEFORE
results). Research/backtest only — no live execution, no spend (free Binance 1h dumps).

Two frozen intraday-shaped signals on the top-30 USDT perps by ADV:
  A1 = hourly time-series momentum: sign(trailing-24h return), inverse-vol sized, 10%
       vol-target, hourly hold. Directional TS-trend at 1h.
  A2 = intraday "first-6h predicts last-6h" (Gao et al. retail analog): cross-sectional
       dollar-neutral, long top-quintile / short bottom-quintile by the first-6h-of-UTC-day
       return, held the last 6h of the same day. Causal (signal at h06, position h18-23).

Costs 5 bps/side realistic + 9 bps stress. OOS 2024+. Multiple-testing N=20 (Bonferroni
0.0025). Reuses the verified data layer + crypto significance functions.

Usage: python scripts/experiment_edge_round3_leadA.py [--top 30]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiment_crypto_xs_signals as h  # noqa: E402  data + significance fns
from src.crypto import data_layer as dl   # noqa: E402

N_TRIALS = 20
BONFERRONI_P = 0.05 / N_TRIALS
ANN_H = 365.0 * 24.0      # hourly annualization
ANN_D = 365.0
OOS_START = "2024-01-01"
COST_BPS = 5.0
STRESS_BPS = 9.0
QUINTILE = 0.20
TOP_N = 30


def sig_block(net, ann, label, btc=None):
    net = net.dropna()
    oos = net.loc[net.index >= OOS_START]
    is_ = net.loc[net.index < OOS_START]
    def sh(x, a=ann):
        x = x.dropna()
        return float(np.sqrt(a) * x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0
    dsr, _ = h.deflated_sr(oos, n_trials=N_TRIALS)
    p = h.block_bootstrap_p(oos)
    beta = h.btc_beta(oos, btc) if btc is not None else float("nan")
    oos_sh = sh(oos)
    full_dd = h.max_drawdown(net)
    return {"label": label, "oos_sharpe": round(oos_sh, 3), "is_sharpe": round(sh(is_), 3),
            "full_maxDD": round(full_dd, 3), "oos_maxDD": round(h.max_drawdown(oos), 3),
            "oos_ann_ret": round(float(oos.mean() * ann), 4),
            "oos_dsr_N20": round(dsr, 3) if not np.isnan(dsr) else None,
            "oos_bootstrap_p": round(p, 4) if not np.isnan(p) else None,
            "oos_btc_beta": round(beta, 3) if not np.isnan(beta) else None,
            "gate": {"oos_sharpe>=0.40": bool(oos_sh >= 0.40),
                     "maxDD<=0.25": bool(abs(full_dd) <= 0.25),
                     "oos_confirmed": bool(oos_sh >= 0.40 and np.sign(oos_sh) == np.sign(sh(is_))),
                     "significance(DSR>=.95&p<.0025)": bool((not np.isnan(dsr)) and dsr >= 0.95
                                                            and (not np.isnan(p)) and p < BONFERRONI_P),
                     "btc_neutral(|b|<=0.15)": bool(abs(beta) <= 0.15) if not np.isnan(beta) else None}}


def top_symbols(top_n):
    syms = dl.list_binance_perp_symbols("USDT")
    daily = h._concurrent(lambda s: dl.fetch_binance_klines(s, "1d"), syms, "daily-rank")
    adv = {s: float(k["quote_volume"].median()) for s, k in daily.items()
           if "quote_volume" in k and len(k) > 200}
    ranked = sorted(adv, key=adv.get, reverse=True)[:top_n]
    return ranked


def backtest_A1(close, cost_bps):
    """Hourly TS-momentum: sign(24h ret), inv-vol, gross=1, 10% vol-target, hourly hold."""
    ret = close.pct_change()
    sign = np.sign(close / close.shift(24) - 1.0)
    inv_vol = 1.0 / ret.rolling(24 * 7, min_periods=48).std()
    raw = (sign * inv_vol).shift(1)                      # known at t-1h (causal)
    gross_n = raw.abs().sum(axis=1).replace(0, np.nan)
    W = raw.div(gross_n, axis=0).fillna(0.0)             # gross exposure 1.0
    Wprev = W.shift(1).fillna(0.0)
    gross = (Wprev * ret.fillna(0.0)).sum(axis=1)
    target_h = 0.10 / np.sqrt(ANN_H)
    lev = (target_h / gross.rolling(24 * 7, min_periods=48).std().shift(1)).clip(upper=3.0).fillna(0.0)
    Wl = W.mul(lev, axis=0)
    turn = (Wl - Wl.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = lev * gross - turn * (cost_bps / 1e4)
    return net


def backtest_A2(close, cost_bps):
    """first-6h-of-UTC-day -> last-6h, cross-sectional dollar-neutral quintiles (daily series)."""
    hod = close.index.hour
    def at(h_):
        sub = close[hod == h_].copy()
        sub.index = sub.index.normalize()
        return sub[~sub.index.duplicated(keep="last")]
    c0, c6, c18, c23 = at(0), at(6), at(18), at(23)
    idx = c0.index.intersection(c6.index).intersection(c18.index).intersection(c23.index)
    first6 = (c6.loc[idx] / c0.loc[idx] - 1.0)           # signal, known at h06
    last6 = (c23.loc[idx] / c18.loc[idx] - 1.0)          # return captured h18->h23 (causal)
    rows = []
    for d in idx:
        s = first6.loc[d].dropna()
        r = last6.loc[d]
        w = pd.Series(0.0, index=first6.columns)
        if len(s) >= 10:
            k = max(1, int(len(s) * QUINTILE))
            ranked = s.sort_values()
            w[ranked.index[-k:]] = 0.5 / k               # long top first-6h (momentum)
            w[ranked.index[:k]] = -0.5 / k               # short bottom
        pnl = float((w * r.reindex(w.index).fillna(0.0)).sum())
        cost = float(w.abs().sum() * 2.0 * (cost_bps / 1e4))   # enter+exit
        rows.append({"date": d, "net": pnl - cost})
    return pd.DataFrame(rows).set_index("date")["net"]


def main():
    t0 = time.time()
    top_n = TOP_N
    if "--top" in sys.argv:
        top_n = int(sys.argv[sys.argv.index("--top") + 1])
    print(f"=== ROUND-3 LEAD A: intraday 1h crypto (top-{top_n} by ADV, N={N_TRIALS}) ===")
    syms = top_symbols(top_n)
    print(f"top-{top_n} by ADV: {syms}")
    klines = h._concurrent(lambda s: dl.fetch_binance_klines(s, "1h"), syms, "klines-1h")
    close = pd.DataFrame({s: k["close"] for s, k in klines.items()})
    close.index = pd.to_datetime(close.index, utc=True)
    close = close.sort_index()
    print(f"1h panel: {close.shape[0]} hours x {close.shape[1]} coins, "
          f"{close.index.min()}->{close.index.max()}")
    btc_h = close["BTCUSDT"].pct_change() if "BTCUSDT" in close else None

    results = {"experiment": "round3_leadA_intraday_1h",
               "pre_registered": "docs/experiment-edge-hunt-round3-2026-06-29.md",
               "n_trials": N_TRIALS, "bonferroni_alpha": BONFERRONI_P,
               "universe_top_n": top_n, "universe": syms,
               "panel": {"hours": int(close.shape[0]), "coins": int(close.shape[1]),
                         "span": [str(close.index.min()), str(close.index.max())]},
               "signals": {}}

    for bps, tag in ((COST_BPS, "realistic"), (STRESS_BPS, "stress")):
        a1 = backtest_A1(close, bps)
        btc_d = None
        a2 = backtest_A2(close, bps)
        # A2 btc-beta vs daily BTC return
        btc_daily = close["BTCUSDT"].resample("1D").last().pct_change() if "BTCUSDT" in close else None
        results["signals"][f"A1_hourly_tsmom_{tag}_{int(bps)}bps"] = sig_block(a1, ANN_H, "A1", btc_h)
        results["signals"][f"A2_intraday_first6_last6_{tag}_{int(bps)}bps"] = sig_block(a2, ANN_D, "A2", btc_daily)

    any_clear = any(all(v for k, v in s["gate"].items() if v is not None)
                    for s in results["signals"].values())
    results["any_signal_clears"] = bool(any_clear)
    results["elapsed_s"] = round(time.time() - t0, 1)

    outdir = REPO_ROOT / "trained_data" / "backtests"
    out = outdir / (f"edge_round3_leadA_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True))
    tmp.replace(out)
    for name, s in results["signals"].items():
        print(f"\n{name}:")
        print(f"  OOS Sharpe {s['oos_sharpe']:+.3f} | IS {s['is_sharpe']:+.3f} | "
              f"fullDD {s['full_maxDD']:+.3f} | DSR {s['oos_dsr_N20']} | p {s['oos_bootstrap_p']} | "
              f"beta {s['oos_btc_beta']} | clears={all(v for k,v in s['gate'].items() if v is not None)}")
    print(f"\nany_signal_clears={any_clear}\nwrote {out}\nelapsed {results['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
