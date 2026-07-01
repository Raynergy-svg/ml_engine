#!/usr/bin/env python3
"""H2 INFRA-STRESS — is our infrastructure suppressing the crypto-momentum signal?

NOT a new hypothesis. This is a ROBUSTNESS DECOMPOSITION of the already-pre-registered
H2 result (docs/experiment-crypto-edge-hunt-2026-06-29.md), stress-testing the SHARED
modelling assumptions the from-scratch verifier cannot catch (because both
implementations make them): the cost level, the rebalance frequency / turnover, and the
absence of vol-targeting / drawdown control. It does NOT change the signal or sweep
params to manufacture a positive (anti-p-hacking, L-018) — it asks, for the EXISTING
signal, at what cost / turnover / risk-overlay it crosses the gate, so we can say which
infra assumption (if any) is suppressing real signal vs correctly rejecting a non-edge.

Reuses the verifier-confirmed harness VERBATIM (build_panels / make_signal / metrics from
scripts/experiment_crypto_xs_signals.py); the ONLY new code is a backtest wrapper that
(a) rebalances every N days instead of daily and (b) optionally vol-targets the book using
LAGGED trailing realized vol. Same eligibility, same 1-day signal lag, same costs, same
funding model, same OOS split, same significance math.

Research/backtest only — no live execution, no real money, no order path.

Usage: python scripts/experiment_crypto_h2_infra_stress.py
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

import experiment_crypto_xs_signals as h  # noqa: E402  the verified H2 harness

ANN = h.ANN
QUINTILE = h.QUINTILE
OOS_START = h.OOS_START
TARGET_ANN_VOL = 0.10          # vol-target overlay: 10% annualised
VOL_WINDOW = 30               # trailing window for realized-vol estimate (days)
MAX_LEV = 3.0                 # leverage cap for the vol-target overlay


# --------------------------------------------------------------------------- #
def custom_momentum_signal(close, cols, lookback):
    """Mirror h.make_signal('momentum', ...) but with an arbitrary lookback."""
    raw = close[cols] / close[cols].shift(lookback) - 1.0
    return raw.shift(1)  # known at t-1 (causal) — identical convention to the harness


def backtest_flex(close, funding_daily, eligible, cols, signal_df, direction, *,
                  cost_bps, rebalance_days=1, vol_target=False):
    """Harness backtest() generalised with rebalance frequency + optional vol-target.

    rebalance_days=1 + vol_target=False reproduces h.backtest() exactly (regression-
    checked in main()). On non-rebalance days the book is HELD (zero turnover), which is
    the standard, faithful way to lower turnover. Vol-target scales the held book by
    target_vol / lagged_realized_vol (no look-ahead), recomputing turnover on the levered
    weights so cost scales with leverage too.
    """
    px = close[cols]
    fund = funding_daily[cols]
    elig = eligible[cols]
    price_ret = px.pct_change()
    sig = signal_df.where(elig.shift(1).fillna(False))

    # --- pass 1: unlevered book, rebalancing every `rebalance_days` days ---------- #
    dates = list(px.index)
    target_w = pd.Series(0.0, index=cols)   # current target quintile weights (held)
    weights = {}                            # date -> held weight Series
    for i, d in enumerate(dates):
        if i % rebalance_days == 0:
            s = sig.loc[d].dropna()
            w = pd.Series(0.0, index=cols)
            if len(s) >= 10:
                k = max(1, int(len(s) * QUINTILE))
                ranked = s.sort_values()
                low, high = ranked.index[:k], ranked.index[-k:]
                longs, shorts = (high, low) if direction > 0 else (low, high)
                w[longs] = 0.5 / k
                w[shorts] = -0.5 / k
            target_w = w
        weights[d] = target_w.copy()
    W = pd.DataFrame(weights).T.reindex(columns=cols).fillna(0.0)  # date x cols
    Wprev = W.shift(1).fillna(0.0)

    pr = price_ret.reindex(columns=cols).fillna(0.0)
    fr = fund.reindex(columns=cols).fillna(0.0)
    price_pnl = (Wprev * pr).sum(axis=1)
    carry_pnl = (-Wprev * fr).sum(axis=1)
    gross = price_pnl + carry_pnl

    # --- pass 2: optional vol-target overlay (LAGGED realized vol, no look-ahead) -- #
    if vol_target:
        target_daily = TARGET_ANN_VOL / np.sqrt(ANN)
        realized = gross.rolling(VOL_WINDOW, min_periods=10).std().shift(1)
        lev = (target_daily / realized).clip(upper=MAX_LEV).fillna(0.0)
        Wlev = W.mul(lev, axis=0)
        Wlevprev = Wlev.shift(1).fillna(0.0)
        turnover = (Wlev - Wlevprev).abs().sum(axis=1)
        price_pnl = (Wlevprev * pr).sum(axis=1)
        carry_pnl = (-Wlevprev * fr).sum(axis=1)
        gross = price_pnl + carry_pnl
    else:
        turnover = (W - Wprev).abs().sum(axis=1)

    cost = turnover * (cost_bps / 1e4)
    out = pd.DataFrame({"price": price_pnl, "carry": carry_pnl,
                        "turnover": turnover, "cost": cost,
                        "gross": gross, "net": gross - cost})
    out.index = px.index
    return out


# --------------------------------------------------------------------------- #
def metrics(r, btc_ret, label):
    oos = r.loc[r.index >= OOS_START]
    is_ = r.loc[r.index < OOS_START]
    net_oos = oos["net"]
    dsr, _ = h.deflated_sr(net_oos)
    boot_p = h.block_bootstrap_p(net_oos)
    return {
        "label": label,
        "oos_net_sharpe": round(h.sharpe(net_oos), 3),
        "oos_gross_sharpe": round(h.sharpe(oos["gross"]), 3),
        "is_net_sharpe": round(h.sharpe(is_["net"]), 3),
        "full_max_dd": round(h.max_drawdown(r["net"]), 3),
        "oos_max_dd": round(h.max_drawdown(net_oos), 3),
        "oos_net_ann": round(float(net_oos.mean() * ANN), 4),
        "oos_cost_ann": round(float(oos["cost"].mean() * ANN), 4),
        "avg_turnover": round(float(r["turnover"].mean()), 3),
        "oos_btc_beta": round(h.btc_beta(net_oos, btc_ret), 3),
        "oos_dsr": round(dsr, 3) if not np.isnan(dsr) else None,
        "oos_bootstrap_p": round(boot_p, 4) if not np.isnan(boot_p) else None,
        # gate booleans (history>=10y is FALSE by construction for all crypto)
        "g_sharpe>=0.40": bool(h.sharpe(net_oos) >= 0.40),
        "g_dd<=0.25": bool(abs(h.max_drawdown(r["net"])) <= 0.25),
        "g_mt(DSR>=.95&p<.0167)": bool((not np.isnan(dsr)) and dsr >= 0.95
                                       and (not np.isnan(boot_p)) and boot_p < 0.0167),
        "g_beta(|b|<=0.15)": bool(abs(h.btc_beta(net_oos, btc_ret)) <= 0.15),
    }


def main():
    t0 = time.time()
    print("=== H2 INFRA-STRESS (robustness decomposition, NOT a new hypothesis) ===")
    close, funding_daily, eligible, btc_ret, cols, _ = h.build_panels(False, False)
    base_sig = h.make_signal("momentum", close, None, cols)  # 14d, the frozen H2 signal
    print(f"universe: {len(cols)} | span {close.index.min().date()}->"
          f"{close.index.max().date()}\n")

    results = {"experiment": "H2_infra_stress",
               "note": ("robustness decomposition of the pre-registered H2 "
                        "(docs/experiment-crypto-edge-hunt-2026-06-29.md); reuses the "
                        "verifier-confirmed harness; NOT a new hypothesis, does NOT "
                        "increment the multiple-testing count"),
               "sweeps": {}}

    # --- regression check: flex(daily, no vol-target) == harness backtest() ------- #
    r_ref = h.backtest(close, funding_daily, eligible, cols, base_sig, +1, cost_bps=10.0)
    r_flex = backtest_flex(close, funding_daily, eligible, cols, base_sig, +1,
                           cost_bps=10.0, rebalance_days=1, vol_target=False)
    ref_s = h.sharpe(r_ref.loc[r_ref.index >= OOS_START, "net"])
    flex_s = h.sharpe(r_flex.loc[r_flex.index >= OOS_START, "net"])
    reg_ok = abs(ref_s - flex_s) < 1e-6
    print(f"[regression] harness OOS Sharpe {ref_s:.4f} vs flex {flex_s:.4f} "
          f"-> {'MATCH' if reg_ok else 'MISMATCH!!'}")
    results["regression_check"] = {"harness_oos_sharpe": round(ref_s, 4),
                                   "flex_oos_sharpe": round(flex_s, 4),
                                   "match": bool(reg_ok)}

    # === SWEEP 1: cost level (daily rebalance, 14d momentum, no overlay) ========== #
    print("\n--- SWEEP 1: cost level (per side bps), daily rebalance ---")
    cost_rows = []
    for bps in (2, 4, 6, 10, 20):
        r = backtest_flex(close, funding_daily, eligible, cols, base_sig, +1,
                          cost_bps=float(bps), rebalance_days=1)
        m = metrics(r, btc_ret, f"cost_{bps}bps")
        cost_rows.append(m)
        print(f"  {bps:>2}bps: OOS net Sharpe {m['oos_net_sharpe']:+.3f}  "
              f"net_ann {m['oos_net_ann']:+.3f}  fullDD {m['full_max_dd']:+.3f}")
    results["sweeps"]["cost_level"] = cost_rows

    # === SWEEP 2: rebalance frequency (10bps, 14d momentum, no overlay) =========== #
    print("\n--- SWEEP 2: rebalance frequency (days), 10bps ---")
    reb_rows = []
    for rd in (1, 2, 3, 5, 10):
        r = backtest_flex(close, funding_daily, eligible, cols, base_sig, +1,
                          cost_bps=10.0, rebalance_days=rd)
        m = metrics(r, btc_ret, f"rebalance_{rd}d")
        reb_rows.append(m)
        print(f"  every {rd:>2}d: OOS net Sharpe {m['oos_net_sharpe']:+.3f}  "
              f"turnover {m['avg_turnover']:.3f}  cost_ann {m['oos_cost_ann']:.3f}  "
              f"fullDD {m['full_max_dd']:+.3f}")
    results["sweeps"]["rebalance_frequency"] = reb_rows

    # === SWEEP 3: vol-target overlay (10% ann), daily + weekly rebalance ========== #
    print("\n--- SWEEP 3: vol-target 10% overlay (DD control), 10bps ---")
    vt_rows = []
    for rd in (1, 5):
        r = backtest_flex(close, funding_daily, eligible, cols, base_sig, +1,
                          cost_bps=10.0, rebalance_days=rd, vol_target=True)
        m = metrics(r, btc_ret, f"voltarget_rebalance_{rd}d")
        vt_rows.append(m)
        print(f"  voltgt every {rd:>2}d: OOS net Sharpe {m['oos_net_sharpe']:+.3f}  "
              f"fullDD {m['full_max_dd']:+.3f}  oosDD {m['oos_max_dd']:+.3f}  "
              f"turnover {m['avg_turnover']:.3f}")
    results["sweeps"]["vol_target_overlay"] = vt_rows

    results["elapsed_s"] = round(time.time() - t0, 1)
    outdir = REPO_ROOT / "trained_data" / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / (f"crypto_h2_infra_stress_"
                    f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True))
    tmp.replace(out)
    print(f"\nwrote {out}\nelapsed {results['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
