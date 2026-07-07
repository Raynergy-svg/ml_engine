#!/usr/bin/env python3
"""Cash-and-carry (long spot proxy / short perp) — pre-registered, frozen.

Pre-registered: docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md (frozen; NO sweep).
Research/backtest only — no live execution, no real money, no order path.

This is NOT a re-run of H1 (scripts/experiment_crypto_funding_carry.py). H1 is a
cross-sectional relative-value book (short highest-funding quintile, long lowest,
dollar-neutral) that failed the ship gate on price adverse-selection (shorting
euphoric/momentum names). This script tests the textbook cash-and-carry
construction instead: per-asset long-spot/short-perp, delta-neutral BY
CONSTRUCTION, harvesting only POSITIVE funding. Universe, signal, and cost/
vol-target constants are IMPORTED VERBATIM from the H1 and H2/H4 frozen
harnesses -- never redefined or tuned here. See the pre-registration doc's
section 0-2 for the full rationale and honest scope limitations (basis risk
assumed zero, positive-funding-only, exchange-solvency/liquidation tail not
modeled).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_crypto_funding_carry as _h1        # noqa: E402  frozen universe+signal (verifier-confirmed)
import experiment_crypto_h2_infra_stress as _h2i      # noqa: E402  frozen vol-target overlay (verifier-confirmed)

# --------------------------------------------------------------------------- #
# Frozen construction -- imported constants ONLY. Never redefine/tune here.
# --------------------------------------------------------------------------- #
ANN: float = _h1.ANN                                   # 365.0
LOOKBACK_D: int = _h1.LOOKBACK_D                        # 3
MIN_HISTORY_D: int = _h1.MIN_HISTORY_D                  # 90
MIN_ADV_USD: float = _h1.MIN_ADV_USD                    # 10_000_000
ADV_WINDOW_D: int = _h1.ADV_WINDOW_D                    # 30
COST_BPS_PER_LEG: float = _h1.COST_BPS                  # 10.0 (imported; doubled below for 2 legs)
COST_BPS_ROUNDTRIP: float = COST_BPS_PER_LEG * 2.0      # 20.0 -- two legs (spot + perp), conservative
OOS_START: str = _h1.OOS_START                          # "2024-01-01"
N_TRIALS: int = _h1.N_TRIALS                            # 3 -- shared multiple-testing budget with H1-H3
TARGET_ANN_VOL: float = _h2i.TARGET_ANN_VOL              # 0.10
VOL_WINDOW: int = _h2i.VOL_WINDOW                        # 30
MAX_LEV: float = _h2i.MAX_LEV                            # 3.0

PRE_REGISTERED = "docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md"


# --------------------------------------------------------------------------- #
# Backtest -- per-asset positive-funding-only cash-and-carry, delta-neutral
# by construction (no price P&L term -- see pre-registration doc section 1-2).
# --------------------------------------------------------------------------- #
def backtest(funding_daily: pd.DataFrame, eligible: pd.DataFrame, cols: list,
             *, cost_bps_roundtrip: float, vol_target: bool = True) -> pd.DataFrame:
    """Returns daily series: carry (=net pre-overlay), cost, gross, net, turnover."""
    fund = funding_daily[cols]
    elig = eligible[cols]

    # signal: trailing LOOKBACK_D mean funding, known at t-1 (causal), same as H1.
    sig = fund.rolling(LOOKBACK_D, min_periods=LOOKBACK_D).mean().shift(1)
    sig = sig.where(elig.shift(1).fillna(False))

    dates = fund.index
    prev_w = pd.Series(0.0, index=cols)
    unlevered_rows = []
    weights: dict = {}
    for d in dates:
        s = sig.loc[d].dropna()
        positive = s[s > 0.0]
        w = pd.Series(0.0, index=cols)
        if len(positive) > 0:
            w[positive.index] = 1.0 / len(positive)  # equal-weight, gross=1.0 pre-overlay
        weights[d] = w
        fr = fund.loc[d].reindex(cols).fillna(0.0)
        # long spot / short perp: collects +funding when perp funding is positive.
        # No price P&L term (delta-neutral by construction -- see pre-reg doc).
        carry_pnl = float((prev_w * fr).sum())
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * (cost_bps_roundtrip / 1e4)
        unlevered_rows.append({"date": d, "carry": carry_pnl, "turnover": turnover,
                               "cost": cost, "net": carry_pnl - cost})
        prev_w = w
    r = pd.DataFrame(unlevered_rows).set_index("date")

    if not vol_target:
        r["gross"] = r["carry"]
        return r

    target_daily = TARGET_ANN_VOL / (ANN ** 0.5)
    realized = r["net"].rolling(VOL_WINDOW, min_periods=10).std().shift(1)
    lev = (target_daily / realized).clip(upper=MAX_LEV).fillna(0.0)
    r["carry"] = r["carry"] * lev
    r["cost"] = r["cost"] * lev
    r["gross"] = r["carry"]
    r["net"] = r["carry"] - r["cost"]
    r["turnover"] = r["turnover"] * lev
    return r


# --------------------------------------------------------------------------- #
# Metrics (reused verbatim -- same formulas as H1)
# --------------------------------------------------------------------------- #
def sharpe(x):
    x = x.dropna()
    return float(np.sqrt(ANN) * x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0


def max_drawdown(x):
    eq = (1 + x.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def report_block(r, label):
    net = r["net"]
    return {
        "label": label, "n_days": int(len(net)),
        "net_sharpe": round(sharpe(net), 3),
        "net_ann_ret": round(float(net.mean() * ANN), 4),
        "carry_ann_ret": round(float(r["carry"].mean() * ANN), 4),
        "cost_ann": round(float(r["cost"].mean() * ANN), 4),
        "avg_turnover": round(float(r["turnover"].mean()), 3),
        "max_drawdown": round(max_drawdown(net), 3),
    }


def main() -> int:
    t0 = time.time()
    print("=== CASH-AND-CARRY (long spot proxy / short perp, positive-funding-only) ===")
    close, adv, funding_daily, eligible, btc_ret, cols = _h1.build_panels()
    print(f"tradeable universe size (ever-liquid): {len(cols)}")
    print(f"backtest span: {close.index.min().date()} -> {close.index.max().date()}")

    r_full = backtest(funding_daily, eligible, cols, cost_bps_roundtrip=COST_BPS_ROUNDTRIP)

    oos = lambda r: r.loc[r.index >= OOS_START]  # noqa: E731
    is_ = lambda r: r.loc[r.index < OOS_START]   # noqa: E731

    full_all = report_block(r_full, "full_all")
    full_is = report_block(is_(r_full), "full_IS_pre2024")
    full_oos = report_block(oos(r_full), "full_OOS_2024+")

    oos_net = oos(r_full)["net"]
    dsr, dsr_bench = _h1.deflated_sr(oos_net, n_trials=N_TRIALS)
    boot_p = _h1.block_bootstrap_p(oos_net)
    psr0 = _h1.psr(oos_net, 0.0)
    beta = _h1.btc_beta(oos_net, btc_ret)

    g_sharpe = full_oos["net_sharpe"] >= 0.40
    g_dd = abs(full_all["max_drawdown"]) <= 0.25
    g_oos_conf = (full_oos["net_sharpe"] >= 0.40
                  and np.sign(full_oos["net_sharpe"] or 1) == np.sign(full_is["net_sharpe"] or 1))
    g_mt = (not np.isnan(dsr)) and dsr >= 0.95 and (not np.isnan(boot_p)) and boot_p < 0.05 / N_TRIALS
    g_beta = (not np.isnan(beta)) and abs(beta) <= 0.15
    g_history = False  # MIN_TOTAL_YEARS=10 cannot be met (pre-registered caveat, shared w/ H1)

    verdict = {
        "experiment": "crypto_cash_and_carry_positive_funding_only",
        "pre_registered": PRE_REGISTERED,
        "frozen_params": {
            "lookback_d": LOOKBACK_D, "min_adv_usd": MIN_ADV_USD, "min_history_d": MIN_HISTORY_D,
            "cost_bps_roundtrip": COST_BPS_ROUNDTRIP, "oos_start": OOS_START, "n_trials": N_TRIALS,
            "vol_target_ann": TARGET_ANN_VOL, "vol_window_d": VOL_WINDOW, "max_leverage": MAX_LEV,
        },
        "universe": {"ever_liquid": len(cols),
                     "span": [str(close.index.min().date()), str(close.index.max().date())]},
        "blocks": {b["label"]: b for b in [full_all, full_is, full_oos]},
        "significance_oos": {
            "psr_gt0": round(psr0, 4), "deflated_sr": round(dsr, 4),
            "dsr_bench_ann_sharpe": round(dsr_bench, 3),
            "bootstrap_p_mean_le0": round(boot_p, 4),
            "bonferroni_alpha": round(0.05 / N_TRIALS, 4),
        },
        "btc_beta_oos": round(beta, 3),
        "gate": {
            "oos_net_sharpe>=0.40": bool(g_sharpe),
            "max_drawdown<=0.25": bool(g_dd),
            "oos_confirmed": bool(g_oos_conf),
            "multiple_testing(DSR>=.95 & p<.0167)": bool(g_mt),
            "btc_beta_neutral(|b|<=0.15)": bool(g_beta),
            "history_depth>=10y": bool(g_history),
        },
        "clears_ex_history": bool(g_sharpe and g_dd and g_oos_conf and g_mt and g_beta),
        "elapsed_s": round(time.time() - t0, 1),
    }

    outdir = REPO_ROOT / "trained_data" / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"crypto_cash_and_carry_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    tmp.replace(p)

    print("\n--- OOS (2024+) headline ---")
    print(json.dumps(full_oos, indent=2))
    print("\n--- significance + BTC-beta (OOS) ---")
    print(json.dumps(verdict["significance_oos"], indent=2))
    print(f"btc_beta_oos={verdict['btc_beta_oos']}")
    print("\n--- gate ---")
    print(json.dumps(verdict["gate"], indent=2))
    print(f"\nclears_ex_history={verdict['clears_ex_history']}  "
          f"(history_depth>=10y is FALSE by construction -> binding caveat)")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
