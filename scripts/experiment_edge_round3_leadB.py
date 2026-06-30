#!/usr/bin/env python3
"""Round-3 LEAD B — broad cross-asset TS-trend book + the SIGNIFICANCE test it never faced.

PRE-REGISTERED: docs/experiment-edge-hunt-round3-2026-06-29.md (frozen; committed BEFORE
results). Research/backtest only — no live execution, no spend (yfinance free daily).

Reuses the verifier-checked canonical trend construction VERBATIM
(src/equity/multi_asset_trend.py: single_asset_trend_returns / trend_streams /
combined_portfolio / ew_buy_hold / sixty_forty / gate_eval / decompose) and the crypto
harness significance functions (deflated_sr / block_bootstrap_p, multiple-testing N=20).
The ONLY new analysis is: DSR/Bonferroni significance on the portfolio returns, the
effective-N of the trend-stream matrix, and an equity-beta decomposition — the bars the
prior multi-asset test (factor ship_gate) never applied.

Usage: python scripts/experiment_edge_round3_leadB.py [--cost-bps 2]
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

import experiment_crypto_xs_signals as h  # noqa: E402  significance fns (deflated_sr, bootstrap)
from src.equity.multi_asset_trend import (  # noqa: E402
    combined_portfolio, ew_buy_hold, sixty_forty, gate_eval, trend_streams,
)

N_TRIALS = 20
BONFERRONI_P = 0.05 / N_TRIALS  # 0.0025
ANN = 252.0
OOS_FRACTION = 0.35

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ",
            "TLT", "IEF", "SHY", "LQD", "HYG", "EMB", "TIP",
            "USO", "UNG", "DBC", "DBA", "CORN", "WEAT", "GLD", "SLV", "CPER",
            "VNQ", "UUP", "FXE", "FXY", "FXB", "FXA",
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD", "BCH-USD",
            "DOGE-USD", "ADA-USD"]
CACHE = REPO_ROOT / "market_data" / "multi_asset" / "panel_round3.parquet"


def fetch_panel():
    if CACHE.exists():
        print(f"loading cached panel {CACHE}")
        return pd.read_parquet(CACHE)
    import yfinance as yf
    print("downloading panel (yfinance, free, daily auto-adjusted)...")
    raw = yf.download(UNIVERSE, start="1990-01-01", progress=False, auto_adjust=True,
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


def effective_n(streams: pd.DataFrame) -> float:
    """Participation ratio of the trend-stream correlation eigenvalues: (Σλ)²/Σλ².
    = number of effectively-independent return streams (the verifier's effective-N)."""
    s = streams.dropna(how="all")
    corr = s.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    ev = np.linalg.eigvalsh(corr)
    ev = ev[ev > 0]
    if ev.size == 0:
        return float("nan")
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def beta_to(net: pd.Series, mkt: pd.Series) -> float:
    df = pd.concat([net, mkt], axis=1).dropna()
    df.columns = ["s", "m"]
    if len(df) < 30 or df["m"].var() == 0:
        return float("nan")
    return float(np.cov(df["s"], df["m"])[0, 1] / df["m"].var())


def main():
    t0 = time.time()
    cost_bps = 2.0
    if "--cost-bps" in sys.argv:
        cost_bps = float(sys.argv[sys.argv.index("--cost-bps") + 1])
    print(f"=== ROUND-3 LEAD B: broad cross-asset TS-trend (cost {cost_bps}bps, N={N_TRIALS}) ===")
    px = fetch_panel()
    have = [c for c in UNIVERSE if c in px.columns and px[c].notna().sum() > 250]
    px = px[have]
    print(f"panel: {len(px)} dates, {len(have)}/{len(UNIVERSE)} assets: {have}")

    streams = trend_streams(px, cost_bps=cost_bps)
    net = combined_portfolio(px, cost_bps=cost_bps)
    if net.empty:
        print("empty portfolio")
        return 1
    idx = net.index
    split = int(len(idx) * (1.0 - OOS_FRACTION))
    oos_idx = idx[split:]
    is_idx = idx[:split]
    net_oos, net_is = net.loc[oos_idx], net.loc[is_idx]

    eff_n = effective_n(streams)
    # significance on FULL and OOS portfolio daily returns
    dsr_full, _ = h.deflated_sr(net, n_trials=N_TRIALS)
    p_full = h.block_bootstrap_p(net)
    dsr_oos, _ = h.deflated_sr(net_oos, n_trials=N_TRIALS)
    p_oos = h.block_bootstrap_p(net_oos)

    g_full = gate_eval(net)
    g_oos = gate_eval(net_oos)
    g_is = gate_eval(net_is)

    # baselines + beta (return-vs-risk decomposition)
    ew = ew_buy_hold(px)
    sf = sixty_forty(px)
    spy_ret = px["SPY"].pct_change() if "SPY" in px else pd.Series(dtype=float)
    beta_spy = beta_to(net_oos, spy_ret.loc[spy_ret.index.intersection(oos_idx)])
    ew_oos = ew.loc[ew.index.intersection(oos_idx)]
    sf_oos = sf.loc[sf.index.intersection(oos_idx)] if not sf.empty else pd.Series(dtype=float)

    gate = {
        "oos_net_sharpe>=0.40": bool(g_oos["net_sharpe"] >= 0.40),
        "max_dd<=0.25(full)": bool(abs(g_full["max_dd"]) <= 0.25),
        "majority_years_positive": bool(g_full["positive_years"] > g_full["total_years"] / 2),
        "oos_confirmed": bool(g_oos["net_sharpe"] >= 0.40
                              and np.sign(g_oos["net_sharpe"]) == np.sign(g_is["net_sharpe"])),
        "significance(DSR_N20>=.95 & p<.0025)_OOS": bool(
            (not np.isnan(dsr_oos)) and dsr_oos >= 0.95
            and (not np.isnan(p_oos)) and p_oos < BONFERRONI_P),
        "significance_FULL": bool((not np.isnan(dsr_full)) and dsr_full >= 0.95
                                  and (not np.isnan(p_full)) and p_full < BONFERRONI_P),
    }
    clears = (gate["oos_net_sharpe>=0.40"] and gate["max_dd<=0.25(full)"]
              and gate["majority_years_positive"] and gate["oos_confirmed"]
              and gate["significance(DSR_N20>=.95 & p<.0025)_OOS"])

    result = {
        "experiment": "round3_leadB_broad_cross_asset_trend",
        "pre_registered": "docs/experiment-edge-hunt-round3-2026-06-29.md",
        "cost_bps_per_side": cost_bps, "n_trials": N_TRIALS, "bonferroni_alpha": BONFERRONI_P,
        "universe": have, "n_assets": len(have),
        "effective_n_trend_streams": round(eff_n, 2),
        "window": {"start": str(idx.min().date()), "end": str(idx.max().date()),
                   "n_days": len(idx), "oos_start": str(oos_idx.min().date()),
                   "oos_fraction": OOS_FRACTION},
        "full_sample": g_full, "in_sample": g_is, "out_of_sample": g_oos,
        "significance": {"dsr_full_N20": round(dsr_full, 3) if not np.isnan(dsr_full) else None,
                         "bootstrap_p_full": round(p_full, 4) if not np.isnan(p_full) else None,
                         "dsr_oos_N20": round(dsr_oos, 3) if not np.isnan(dsr_oos) else None,
                         "bootstrap_p_oos": round(p_oos, 4) if not np.isnan(p_oos) else None,
                         "bonferroni_alpha_N20": round(BONFERRONI_P, 4)},
        "return_vs_risk": {
            "trend_oos_sharpe": g_oos["net_sharpe"], "trend_oos_maxDD": g_oos["max_dd"],
            "trend_oos_beta_to_SPY": round(beta_spy, 3) if not np.isnan(beta_spy) else None,
            "ew_buyhold_oos_sharpe": round(h.sharpe(ew_oos), 3),
            "ew_buyhold_oos_maxDD": round(h.max_drawdown(ew_oos), 3),
            "sixty_forty_oos_sharpe": round(h.sharpe(sf_oos), 3) if not sf_oos.empty else None,
            "sixty_forty_oos_maxDD": round(h.max_drawdown(sf_oos), 3) if not sf_oos.empty else None,
        },
        "gate": gate, "clears_gate": bool(clears),
        "elapsed_s": round(time.time() - t0, 1),
    }
    outdir = REPO_ROOT / "trained_data" / "backtests"
    out = outdir / (f"edge_round3_leadB_{int(cost_bps)}bps_"
                    f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(out)
    print(json.dumps({k: result[k] for k in
                      ["effective_n_trend_streams", "full_sample", "out_of_sample",
                       "significance", "return_vs_risk", "gate", "clears_gate"]}, indent=2))
    print(f"\nwrote {out}\nelapsed {result['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
