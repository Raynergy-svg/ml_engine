#!/usr/bin/env python3
"""Crypto Edge Hunt ROUND 2 — H4 (infra-corrected XS momentum) + H5 (TS-trend).

PRE-REGISTERED: docs/experiment-crypto-edge-hunt-round2-2026-06-29.md (frozen; committed
BEFORE this script produced any result). Research/backtest only — no live execution.

Reuses the verifier-confirmed harness (build_panels / make_signal / metrics from
experiment_crypto_xs_signals.py) and the regression-checked backtest_flex
(experiment_crypto_h2_infra_stress.py). The ONLY new code is the H5 time-series-trend
position builder. Multiple-testing correction uses N_TRIALS=15 (the cumulative crypto
search budget — harsher than Round-1's N=3), per the pre-reg §1.

Usage: python scripts/experiment_crypto_round2.py
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

import experiment_crypto_xs_signals as h            # noqa: E402  verified harness
import experiment_crypto_h2_infra_stress as s       # noqa: E402  regression-checked flex

ANN = h.ANN
OOS_START = h.OOS_START
N_TRIALS = 15                 # frozen cumulative search budget (pre-reg §1)
BONFERRONI_P = 0.05 / N_TRIALS   # 0.00333
TARGET_ANN_VOL = 0.10
VOL_WINDOW = 30
MAX_LEV = 3.0
TS_LOOKBACK = 30              # H5 trend lookback (frozen, canonical monthly TS-momentum)
REBALANCE_D = 7              # weekly (7-day) for both H4 and H5 (frozen)


# --------------------------------------------------------------------------- #
def ts_trend_backtest(close, funding_daily, eligible, cols, *, cost_bps,
                      lookback=TS_LOOKBACK, rebalance_days=REBALANCE_D,
                      vol_target=True):
    """H5 time-series trend: per-coin sign(trailing `lookback`-d return), inverse-vol
    sized, portfolio vol-targeted, weekly rebalance. Directional (NOT dollar-neutral) —
    carries market beta by construction. Funding accrues on perp positions (modeled).
    Causal: signal & sizing use data known at t-1 (shift(1))."""
    px = close[cols]
    fund = funding_daily[cols]
    elig = eligible[cols]
    price_ret = px.pct_change()

    sign = np.sign(px / px.shift(lookback) - 1.0)               # +1/-1 per coin
    inv_vol = 1.0 / price_ret.rolling(VOL_WINDOW, min_periods=10).std()
    raw = (sign * inv_vol).shift(1)                             # known at t-1 (causal)
    raw = raw.where(elig.shift(1).fillna(False))

    dates = list(px.index)
    target_w = pd.Series(0.0, index=cols)
    weights = {}
    for i, d in enumerate(dates):
        if i % rebalance_days == 0:
            r = raw.loc[d].dropna()
            w = pd.Series(0.0, index=cols)
            if len(r) >= 10:
                gross = r.abs().sum()
                if gross > 0:
                    w = (r / gross).reindex(cols).fillna(0.0)   # gross-exposure = 1.0
            target_w = w
        weights[d] = target_w.copy()
    W = pd.DataFrame(weights).T.reindex(columns=cols).fillna(0.0)

    pr = price_ret.reindex(columns=cols).fillna(0.0)
    fr = fund.reindex(columns=cols).fillna(0.0)
    Wprev = W.shift(1).fillna(0.0)
    gross_ret = (Wprev * pr).sum(axis=1) + (-Wprev * fr).sum(axis=1)

    if vol_target:
        target_daily = TARGET_ANN_VOL / np.sqrt(ANN)
        realized = gross_ret.rolling(VOL_WINDOW, min_periods=10).std().shift(1)
        lev = (target_daily / realized).clip(upper=MAX_LEV).fillna(0.0)
        Wlev = W.mul(lev, axis=0)
        Wlevprev = Wlev.shift(1).fillna(0.0)
        turnover = (Wlev - Wlevprev).abs().sum(axis=1)
        price_pnl = (Wlevprev * pr).sum(axis=1)
        carry_pnl = (-Wlevprev * fr).sum(axis=1)
    else:
        turnover = (W - Wprev).abs().sum(axis=1)
        price_pnl = (Wprev * pr).sum(axis=1)
        carry_pnl = (-Wprev * fr).sum(axis=1)

    gross = price_pnl + carry_pnl
    cost = turnover * (cost_bps / 1e4)
    out = pd.DataFrame({"price": price_pnl, "carry": carry_pnl, "turnover": turnover,
                        "cost": cost, "gross": gross, "net": gross - cost})
    out.index = px.index
    return out


def full_metrics(r, btc_ret, label):
    oos = r.loc[r.index >= OOS_START]
    is_ = r.loc[r.index < OOS_START]
    net_oos, net_is = oos["net"], is_["net"]
    dsr, dsr_bench = h.deflated_sr(net_oos, n_trials=N_TRIALS)
    boot_p = h.block_bootstrap_p(net_oos)
    oos_sh = h.sharpe(net_oos)
    is_sh = h.sharpe(net_is)
    full_dd = h.max_drawdown(r["net"])
    beta = h.btc_beta(net_oos, btc_ret)
    g_sharpe = oos_sh >= 0.40
    g_dd = abs(full_dd) <= 0.25
    g_oos = (oos_sh >= 0.40) and (np.sign(oos_sh) == np.sign(is_sh))
    g_mt = (not np.isnan(dsr)) and dsr >= 0.95 and (not np.isnan(boot_p)) \
        and boot_p < BONFERRONI_P
    g_beta = abs(beta) <= 0.15
    return {
        "label": label,
        "oos_net_sharpe": round(oos_sh, 3), "is_net_sharpe": round(is_sh, 3),
        "oos_gross_sharpe": round(h.sharpe(oos["gross"]), 3),
        "full_max_dd": round(full_dd, 3), "oos_max_dd": round(h.max_drawdown(net_oos), 3),
        "oos_net_ann": round(float(net_oos.mean() * ANN), 4),
        "oos_cost_ann": round(float(oos["cost"].mean() * ANN), 4),
        "avg_turnover": round(float(r["turnover"].mean()), 3),
        "oos_btc_beta": round(beta, 3),
        "oos_dsr_N15": round(dsr, 3) if not np.isnan(dsr) else None,
        "oos_bootstrap_p": round(boot_p, 4) if not np.isnan(boot_p) else None,
        "bonferroni_alpha_N15": round(BONFERRONI_P, 4),
        "gate": {"oos_sharpe>=0.40": bool(g_sharpe), "max_dd<=0.25": bool(g_dd),
                 "oos_confirmed": bool(g_oos),
                 "mt(DSR_N15>=.95 & p<.0033)": bool(g_mt),
                 "btc_neutral(|b|<=0.15)": bool(g_beta), "history>=10y": False},
        "clears_ex_history": bool(g_sharpe and g_dd and g_oos and g_mt and g_beta),
    }


def main():
    t0 = time.time()
    print("=== CRYPTO ROUND 2: H4 infra-corrected XS + H5 TS-trend (N_TRIALS=15) ===")
    close, funding_daily, eligible, btc_ret, cols, _ = h.build_panels(False, False)
    print(f"universe {len(cols)} | span {close.index.min().date()}->"
          f"{close.index.max().date()}\n")

    # --- H4: infra-corrected XS momentum (14d, vol-target 10%, weekly, 10bps) ----- #
    h2sig = h.make_signal("momentum", close, None, cols)
    r_h4 = s.backtest_flex(close, funding_daily, eligible, cols, h2sig, +1,
                           cost_bps=10.0, rebalance_days=REBALANCE_D, vol_target=True)
    m_h4 = full_metrics(r_h4, btc_ret, "H4_infra_corrected_xs_momentum")

    # --- H5: time-series trend (30d sign, inv-vol, vol-target 10%, weekly, 10bps) -- #
    r_h5 = ts_trend_backtest(close, funding_daily, eligible, cols, cost_bps=10.0)
    m_h5 = full_metrics(r_h5, btc_ret, "H5_ts_trend_momentum")

    # --- H5 risk-control axis: vs buy-hold BTC and equal-weight long-only --------- #
    bh = btc_ret.loc[btc_ret.index >= OOS_START]
    elig_cols = eligible[cols]
    ew = (close[cols].pct_change().where(elig_cols.shift(1).fillna(False))
          .mean(axis=1))
    ew_oos = ew.loc[ew.index >= OOS_START]
    risk_ctrl = {
        "h5_oos_sharpe": m_h5["oos_net_sharpe"], "h5_full_maxDD": m_h5["full_max_dd"],
        "h5_oos_btc_beta": m_h5["oos_btc_beta"],
        "buyhold_btc_oos_sharpe": round(h.sharpe(bh), 3),
        "buyhold_btc_oos_maxDD": round(h.max_drawdown(bh), 3),
        "ew_longonly_oos_sharpe": round(h.sharpe(ew_oos), 3),
        "ew_longonly_oos_maxDD": round(h.max_drawdown(ew_oos), 3),
    }

    results = {"experiment": "crypto_round2_H4_H5",
               "pre_registered": "docs/experiment-crypto-edge-hunt-round2-2026-06-29.md",
               "n_trials": N_TRIALS, "bonferroni_alpha": BONFERRONI_P,
               "frozen_params": {"H4": {"signal": "xs_momentum_14d", "vol_target": 0.10,
                                        "rebalance_days": REBALANCE_D, "cost_bps": 10},
                                 "H5": {"signal": f"ts_trend_sign_{TS_LOOKBACK}d",
                                        "sizing": "inverse_vol", "vol_target": 0.10,
                                        "rebalance_days": REBALANCE_D, "cost_bps": 10}},
               "universe": {"ever_liquid": len(cols),
                            "span": [str(close.index.min().date()),
                                     str(close.index.max().date())]},
               "H4": m_h4, "H5": m_h5, "H5_risk_control_axis": risk_ctrl,
               "elapsed_s": round(time.time() - t0, 1)}

    outdir = REPO_ROOT / "trained_data" / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / (f"crypto_round2_"
                    f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True))
    tmp.replace(out)

    print("--- H4 infra-corrected XS momentum ---")
    print(json.dumps(m_h4, indent=2))
    print("\n--- H5 TS-trend ---")
    print(json.dumps(m_h5, indent=2))
    print("\n--- H5 risk-control axis ---")
    print(json.dumps(risk_ctrl, indent=2))
    print(f"\nwrote {out}\nelapsed {results['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
