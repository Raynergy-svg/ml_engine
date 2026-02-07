"""Calibrate `fx_confidence_v1` and confidence trade gate via walk-forward evaluation.

This script performs a time-ordered walk-forward evaluation across historical
FX candles and sweeps confidence thresholds and simple `base_confidence`
parameter from the `confidence_model` section. It reports both:
 - Tier-1 realistic: applies `fx_guardrails` costs (spread + slippage)
 - Model-only debug: ignores costs (diagnostic)

Usage: run from repo root:
    python scripts/calibrate_fx_confidence.py --instrument EUR_USD

The script prints a CSV-like summary and a recommended threshold.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.fx_paper import atr as fx_atr, setup_signal, spread_pips_from_df, pip_size, conservative_slippage_pips
from src.utils.reasoning_enhanced import fx_confidence_v1
import src.risk.fx_guardrails as fxg


def load_market_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect typical OANDA export columns: time, open, high, low, close, volume, bid_close, ask_close
    return df.reset_index(drop=True)


def simulate_trade_row(
    df: pd.DataFrame,
    idx: int,
    signal: str,
    policy: fxg.FxPolicy,
    cfg: Dict,
    apply_costs: bool = True,
) -> Optional[Dict]:
    """Simulate a single rule-based trade starting at index `idx`.

    Returns dict with realized return in quote units and metadata, or None if no trade.
    """
    if signal not in {"buy", "sell"}:
        return None

    last_close = float(df.loc[idx, "close"])
    atr_value = float(fx_atr(df.iloc[: idx + 1], period=14))

    # Spread from last candle
    spread_pips = spread_pips_from_df(df.iloc[: idx + 1], policy.instruments[0])
    if spread_pips is None:
        spread_pips = float(policy.costs.spread_fallback_pips.get(policy.instruments[0], 0.0) or 0.0)

    # Compute confidence using cfg tuning params
    params = dict((cfg.get("fx", {}) or {}).get("confidence_model") or {})
    payload = fx_confidence_v1(
        signal=signal,
        price=last_close,
        atr=atr_value,
        spread_pips=spread_pips,
        max_spread_pips=float(policy.costs.max_spread_pips.get(policy.instruments[0], spread_pips) or spread_pips),
        params=params,
    )
    confidence = float(payload.get("confidence") or 0.0)

    # Entry price approximated as next bar open if available else last_close
    if idx + 1 < len(df):
        entry_price = float(df.loc[idx + 1, "open"])
    else:
        entry_price = last_close

    # Risk/stop/tp
    stop_distance = float(atr_value) * float(getattr(policy.risk, "atr_stop_mult", 1.5))
    tp_mult = float(getattr(policy.risk, "rr_take_profit", 1.5))

    if signal == "buy":
        stop_price = entry_price - stop_distance
        tp_price = entry_price + stop_distance * tp_mult
    else:
        stop_price = entry_price + stop_distance
        tp_price = entry_price - stop_distance * tp_mult

    # Slippage and cost
    slippage_pips = conservative_slippage_pips(spread_pips=spread_pips, min_pips=policy.costs.slippage_pips_min, spread_mult=policy.costs.slippage_pips_spread_mult)
    slippage_price = slippage_pips * pip_size(policy.instruments[0])
    spread_cost = spread_pips * pip_size(policy.instruments[0])

    # Walk-forward until TP or SL or end
    exit_price = None
    exit_idx = None
    exit_reason = None
    # start scanning from the bar after entry
    for j in range(idx + 2, len(df)):
        high = float(df.loc[j, "high"])
        low = float(df.loc[j, "low"])
        if signal == "buy":
            if low <= stop_price <= high:
                exit_price = stop_price
                exit_idx = j
                exit_reason = "stop"
                break
            if low <= tp_price <= high:
                exit_price = tp_price
                exit_idx = j
                exit_reason = "tp"
                break
        else:
            if low <= tp_price <= high:
                exit_price = tp_price
                exit_idx = j
                exit_reason = "tp"
                break
            if low <= stop_price <= high:
                exit_price = stop_price
                exit_idx = j
                exit_reason = "stop"
                break

    if exit_price is None:
        # No TP/SL hit, close at last available close
        exit_price = float(df.loc[len(df) - 1, "close"])
        exit_idx = len(df) - 1
        exit_reason = "timeout"

    # Compute raw return in price units (quote currency for FX)
    ret = (exit_price - entry_price) if signal == "buy" else (entry_price - exit_price)

    cost = 0.0
    if apply_costs:
        cost = spread_cost + slippage_price

    net_ret = ret - cost

    return {
        "confidence": confidence,
        "entry_idx": idx + 1,
        "exit_idx": exit_idx,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "net_ret": net_ret,
        "exit_reason": exit_reason,
        "spread_pips": spread_pips,
        "slippage_pips": slippage_pips,
    }


def walk_forward_eval(df: pd.DataFrame, policy: fxg.FxPolicy, cfg: Dict, instrument: str, n_splits: int = 5):
    N = len(df)
    if N < 100:
        raise ValueError("Not enough data for walk-forward evaluation")

    fold_size = max(1, N // (n_splits + 1))
    results = []

    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        test_start = train_end
        test_end = min(N, test_start + fold_size)

        df_train = df.iloc[:train_end]
        df_test = df.iloc[test_start:test_end].reset_index(drop=True)

        # For each candlestick in test, generate signal and simulate
        fold_rows = []
        for i in range(len(df_test) - 2):
            global_idx = test_start + i
            sig = setup_signal(df.iloc[: global_idx + 1])
            sim_real = simulate_trade_row(df, global_idx, sig, policy, cfg, apply_costs=True)
            sim_model = simulate_trade_row(df, global_idx, sig, policy, cfg, apply_costs=False)
            if sim_real is not None:
                fold_rows.append((sim_real, sim_model))

        results.append(fold_rows)

    return results


def summarize_sweep(results: List[List], thresholds: List[float]):
    # results: list of folds, each is list of tuples (real, model)
    out = []
    for thr in thresholds:
        stats = {"thr": thr, "trades": 0, "trades_model": 0, "wins": 0, "wins_model": 0, "net_ret": 0.0, "net_ret_model": 0.0}
        for fold in results:
            for real, model in fold:
                if real["confidence"] >= thr:
                    stats["trades"] += 1
                    stats["net_ret"] += real["net_ret"]
                    if real["net_ret"] > 0:
                        stats["wins"] += 1
                if model["confidence"] >= thr:
                    stats["trades_model"] += 1
                    stats["net_ret_model"] += model["net_ret"]
                    if model["net_ret"] > 0:
                        stats["wins_model"] += 1

        stats["win_rate"] = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0.0
        stats["win_rate_model"] = (stats["wins_model"] / stats["trades_model"] * 100) if stats["trades_model"] else 0.0
        stats["avg_ret"] = (stats["net_ret"] / stats["trades"]) if stats["trades"] else 0.0
        stats["avg_ret_model"] = (stats["net_ret_model"] / stats["trades_model"]) if stats["trades_model"] else 0.0
        out.append(stats)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--data", default="market_data/oanda_EUR_USD_M5.csv")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--grid", action="store_true", help="Run grid-search over confidence_model params")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades required to consider a threshold/config")
    parser.add_argument("--top-n", type=int, default=5, help="Print top-N configurations from grid-search")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open("config_tuned.yaml", "r"))
    policy = fxg.load_fx_policy(cfg)

    df = load_market_csv(args.data)
    results = walk_forward_eval(df, policy, cfg, args.instrument, n_splits=args.splits)

    thresholds = [round(x, 2) for x in list(np.arange(0.5, 0.96, 0.05))]
    summary = summarize_sweep(results, thresholds)

    # Print CSV-like summary for baseline
    print("Baseline sweep (current confidence_model params):")
    print("thr,trades,win_rate,avg_ret,trades_model,win_rate_model,avg_ret_model")
    for r in summary:
        print(
            f"{r['thr']},{r['trades']},{r['win_rate']:.2f},{r['avg_ret']:.6f},{r['trades_model']},{r['win_rate_model']:.2f},{r['avg_ret_model']:.6f}"
        )

    cand = [r for r in summary if r['trades'] >= args.min_trades]
    if cand:
        best = max(cand, key=lambda x: x['avg_ret'])
        print(f"\nBaseline recommended threshold (Tier-1 realistic): {best['thr']}  avg_ret={best['avg_ret']:.6f} trades={best['trades']}")
    else:
        print("\nBaseline: no threshold had sufficient realistic trades; consider expanding data or lowering threshold range.")

    # Optional grid-search to try to shift the confidence distribution
    if args.grid:
        print("\nRunning grid-search over confidence_model parameters (this may take a while)...")

        # Define small-ish default grids to avoid explosion
        base_conf_grid = [0.75, 0.80, 0.85, 0.90]
        spread_penalty_grid = [0.0, 0.1, 0.2]
        atr_penalty_grid = [0.0, 0.15, 0.30]
        spread_soft_start_grid = [0.5, 0.6, 0.7]
        atr_soft_start_grid = [0.0005, 0.001, 0.002]

        combos = []
        for bc in base_conf_grid:
            for sp_p in spread_penalty_grid:
                for atr_p in atr_penalty_grid:
                    for sp_ss in spread_soft_start_grid:
                        for atr_ss in atr_soft_start_grid:
                            combos.append(
                                {
                                    "base_confidence": float(bc),
                                    "spread_penalty_max": float(sp_p),
                                    "atr_penalty_max": float(atr_p),
                                    "spread_soft_start_ratio": float(sp_ss),
                                    "atr_soft_start_pct": float(atr_ss),
                                }
                            )

        grid_results = []
        # Evaluate each combo
        for idx, combo in enumerate(combos):
            # inject combo into cfg for fx.confidence_model
            cfg_local = dict(cfg)
            if "fx" not in cfg_local:
                cfg_local["fx"] = {}
            if "confidence_model" not in cfg_local["fx"]:
                cfg_local["fx"]["confidence_model"] = {}
            cfg_local["fx"]["confidence_model"].update(combo)

            res = walk_forward_eval(df, policy, cfg_local, args.instrument, n_splits=args.splits)
            summ = summarize_sweep(res, [0.7])
            stat = summ[0]
            stat.update({"combo": combo})
            grid_results.append(stat)

        # Filter by min trades at threshold 0.7
        viable = [g for g in grid_results if g["trades"] >= args.min_trades]
        if not viable:
            print("Grid-search: no configuration produced >= min-trades at threshold 0.7")
        else:
            # Rank by avg_ret then win_rate
            viable_sorted = sorted(viable, key=lambda x: (x["avg_ret"], x["win_rate"]), reverse=True)
            print(f"\nTop {args.top_n} grid-search results (threshold=0.70):")
            print("avg_ret,win_rate,trades,combo")
            for top in viable_sorted[: args.top_n]:
                print(f"{top['avg_ret']:.6f},{top['win_rate']:.2f},{top['trades']},{top['combo']}")

            best_cfg = viable_sorted[0]
            print(f"\nRecommended config to try (best avg_ret at thr=0.70): {best_cfg['combo']}")


if __name__ == "__main__":
    main()
# — Raynergy-svg —
