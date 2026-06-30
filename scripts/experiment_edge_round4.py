#!/usr/bin/env python3
"""Round-4 — ONE pre-registered breadth+history expansion of the cross-asset trend book.

PRE-REGISTERED: docs/experiment-edge-hunt-round4-2026-06-30.md (frozen; committed BEFORE
results). Research/backtest only — no live execution, no spend (free yfinance).

IDENTICAL to Round-3 Lead B in every way EXCEPT the universe (37 -> 59: + long-history index/
futures proxies and new independent metals/FX/EM sleeves) and N_TRIALS (20 -> 21). Reuses the
verifier-checked construction (src/equity/multi_asset_trend) + significance fns + the Round-3
helpers (effective_n, beta_to). ONE run (+5bps stress robustness). Not iterated to pass.

Run with the interpreter that has yfinance:  python scripts/experiment_edge_round4.py [--cost-bps 2]
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

import experiment_crypto_xs_signals as h  # noqa: E402  significance fns
import experiment_edge_round3_leadB as lb  # noqa: E402  effective_n, beta_to (reuse)
from src.equity.multi_asset_trend import (  # noqa: E402
    combined_portfolio, ew_buy_hold, sixty_forty, gate_eval, trend_streams,
)

N_TRIALS = 21
BONFERRONI_P = 0.05 / N_TRIALS  # 0.00238
ANN = 252.0
OOS_FRACTION = 0.35
TARGET_VOL = 0.10  # FROZEN (explicit, not the 0.12 code default)

BASE_37 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "TLT", "IEF", "SHY",
           "LQD", "HYG", "EMB", "TIP", "USO", "UNG", "DBC", "DBA", "CORN", "WEAT",
           "GLD", "SLV", "CPER", "VNQ", "UUP", "FXE", "FXY", "FXB", "FXA",
           "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD", "BCH-USD",
           "DOGE-USD", "ADA-USD"]
ADD_22 = ["^GSPC", "^IXIC", "^RUT", "^N225", "^GDAXI", "^FTSE", "^HSI", "DX-Y.NYB",
          "GC=F", "SI=F", "CL=F", "HG=F", "NG=F", "ZC=F", "ZW=F", "ZS=F",
          "PPLT", "PALL", "FXF", "FXC", "EWZ", "INDA"]
UNIVERSE = BASE_37 + ADD_22
CACHE = REPO_ROOT / "market_data" / "multi_asset" / "panel_round4.parquet"


def fetch_panel():
    if CACHE.exists():
        print(f"loading cached panel {CACHE}")
        return pd.read_parquet(CACHE)
    import yfinance as yf
    print(f"downloading {len(UNIVERSE)}-ticker panel (yfinance, free)...")
    raw = yf.download(UNIVERSE, start="1920-01-01", progress=False, auto_adjust=True,
                      threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index, utc=True).normalize()
    close = close.reindex(columns=UNIVERSE).dropna(how="all")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        close.to_parquet(CACHE)
    except Exception:  # noqa: BLE001
        pass
    return close


def main():
    t0 = time.time()
    cost_bps = 2.0
    if "--cost-bps" in sys.argv:
        cost_bps = float(sys.argv[sys.argv.index("--cost-bps") + 1])
    print(f"=== ROUND-4: 59-asset breadth+history trend (cost {cost_bps}bps, N={N_TRIALS}) ===")
    px = fetch_panel()
    have = [c for c in UNIVERSE if c in px.columns and px[c].notna().sum() > 250]
    px = px[have]
    print(f"panel: {len(px)} dates, {len(have)}/{len(UNIVERSE)} assets")

    streams = trend_streams(px, cost_bps=cost_bps)
    net = combined_portfolio(px, cost_bps=cost_bps, target_vol=TARGET_VOL)
    if net.empty:
        print("empty portfolio")
        return 1
    idx = net.index
    split = int(len(idx) * (1.0 - OOS_FRACTION))
    oos_idx, is_idx = idx[split:], idx[:split]
    net_oos, net_is = net.loc[oos_idx], net.loc[is_idx]

    eff_n = lb.effective_n(streams)
    dsr_full, _ = h.deflated_sr(net, n_trials=N_TRIALS)
    p_full = h.block_bootstrap_p(net)
    dsr_oos, _ = h.deflated_sr(net_oos, n_trials=N_TRIALS)
    p_oos = h.block_bootstrap_p(net_oos)

    g_full, g_oos, g_is = gate_eval(net), gate_eval(net_oos), gate_eval(net_is)
    ew, sf = ew_buy_hold(px), sixty_forty(px)
    spy = px["SPY"].pct_change() if "SPY" in px else pd.Series(dtype=float)
    beta_spy = lb.beta_to(net_oos, spy.loc[spy.index.intersection(oos_idx)])
    ew_oos = ew.loc[ew.index.intersection(oos_idx)]
    sf_oos = sf.loc[sf.index.intersection(oos_idx)] if not sf.empty else pd.Series(dtype=float)

    gate = {
        "oos_net_sharpe>=0.40": bool(g_oos["net_sharpe"] >= 0.40),
        "max_dd<=0.25(full)": bool(abs(g_full["max_dd"]) <= 0.25),
        "majority_years_positive": bool(g_full["positive_years"] > g_full["total_years"] / 2),
        "oos_confirmed": bool(g_oos["net_sharpe"] >= 0.40
                              and np.sign(g_oos["net_sharpe"]) == np.sign(g_is["net_sharpe"])),
        "DSR_OOS(N21)>=.95 & p<.00238": bool((not np.isnan(dsr_oos)) and dsr_oos >= 0.95
                                             and (not np.isnan(p_oos)) and p_oos < BONFERRONI_P),
    }
    clears = all(gate.values())

    result = {
        "experiment": "round4_breadth_history_expansion",
        "pre_registered": "docs/experiment-edge-hunt-round4-2026-06-30.md",
        "cost_bps_per_side": cost_bps, "n_trials": N_TRIALS, "bonferroni_alpha": BONFERRONI_P,
        "universe": have, "n_assets": len(have),
        "effective_n_trend_streams": round(eff_n, 2),
        "effective_n_round3_for_reference": 13.07,
        "window": {"start": str(idx.min().date()), "end": str(idx.max().date()),
                   "n_days": len(idx), "oos_start": str(oos_idx.min().date()),
                   "oos_n_days": len(oos_idx), "oos_fraction": OOS_FRACTION},
        "full_sample": g_full, "in_sample": g_is, "out_of_sample": g_oos,
        "significance": {"dsr_full_N21": round(dsr_full, 3) if not np.isnan(dsr_full) else None,
                         "bootstrap_p_full": round(p_full, 4) if not np.isnan(p_full) else None,
                         "dsr_oos_N21": round(dsr_oos, 3) if not np.isnan(dsr_oos) else None,
                         "bootstrap_p_oos": round(p_oos, 4) if not np.isnan(p_oos) else None,
                         "round3_dsr_oos_for_reference": 0.843},
        "return_vs_risk": {
            "trend_oos_sharpe": g_oos["net_sharpe"], "trend_oos_maxDD": g_oos["max_dd"],
            "trend_oos_beta_to_SPY": round(beta_spy, 3) if not np.isnan(beta_spy) else None,
            "ew_buyhold_oos_sharpe": round(h.sharpe(ew_oos), 3),
            "ew_buyhold_oos_maxDD": round(h.max_drawdown(ew_oos), 3),
            "sixty_forty_oos_sharpe": round(h.sharpe(sf_oos), 3) if not sf_oos.empty else None,
        },
        "gate": gate, "clears_gate": bool(clears),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = (REPO_ROOT / "trained_data" / "backtests" /
           f"edge_round4_{int(cost_bps)}bps_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(out)
    print(json.dumps({k: result[k] for k in
                      ["effective_n_trend_streams", "window", "full_sample", "out_of_sample",
                       "significance", "return_vs_risk", "gate", "clears_gate"]}, indent=2))
    print(f"\nwrote {out}\nelapsed {result['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
