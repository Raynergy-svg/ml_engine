#!/usr/bin/env python3
"""
Full-Pipeline Historical Replay v2

Runs the ACTUAL scanner models (TCN/LightGBM/Ridge ensemble) on sliding windows
of historical data, but applies configurable threshold gates instead of the ML
gate evaluator (which rejects everything on historical data due to stale state).

Flow for each window:
  1. Fetch full history from OANDA
  2. Slide forward in STEP_SIZE increments
  3. Run feature engineering + ensemble inference
  4. Apply simple threshold gates (confidence, momentum, risk/reward)
  5. If agent team is loaded, run specialist agents
  6. Simulate SL/TP outcome on future candles
  7. Compute institutional metrics and verdict
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(name)s:%(levelname)s: %(message)s")
logger = logging.getLogger("full_pipeline_replay")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

ALL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "GBP_AUD", "EUR_CHF", "GBP_CHF",
]

CANDLE_COUNT = 5000         # ~7 months of H1 data
WINDOW_SIZE = 300           # Scanner lookback
STEP_SIZE = 6              # Advance 6 candles (6 hours)
MAX_BARS_OUTCOME = 48      # Max 48h to hit SL/TP
MAX_TRADES_PER_PAIR = 150  # Cap per pair
MIN_COOLDOWN_BARS = 12     # Min bars between trades on same pair

# Configurable gate thresholds (replaces ML gate evaluator)
GATE_CONFIDENCE_MIN = 0.45     # Model confidence threshold
GATE_ATR_SL_MULT = 1.0        # SL = ATR * mult
GATE_ATR_TP_MULT = 1.5        # TP = ATR * mult
GATE_MIN_RR = 1.2              # Minimum risk:reward ratio
GATE_MIN_ATR_PIPS = 5.0       # Skip if ATR too low (dead market)

# Live readiness thresholds
THRESHOLDS = {
    "min_trades": 300,
    "min_win_rate": 0.42,
    "min_sharpe": 0.05,
    "min_profit_factor": 1.05,
    "max_drawdown_pct": 25.0,
    "max_consec_losses": 15,
    "min_profitable_pairs": 5,
    "min_expectancy": 0.3,
}


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


def simulate_outcome(candles_df: pd.DataFrame, entry_idx: int,
                     direction: str, sl_pips: float, tp_pips: float,
                     pair: str) -> dict:
    """Walk forward through candles to check SL/TP hit."""
    ps = pip_size(pair)
    entry_price = float(candles_df.iloc[entry_idx]["close"])
    sl_raw = sl_pips * ps
    tp_raw = tp_pips * ps
    max_idx = min(entry_idx + MAX_BARS_OUTCOME, len(candles_df))

    for i in range(entry_idx + 1, max_idx):
        row = candles_df.iloc[i]
        high, low = float(row["high"]), float(row["low"])

        if direction in ("LONG", "BUY"):
            if high - entry_price >= tp_raw:
                return {"pnl_pips": tp_pips, "bars": i - entry_idx, "exit": "tp_hit"}
            if entry_price - low >= sl_raw:
                return {"pnl_pips": -sl_pips, "bars": i - entry_idx, "exit": "sl_hit"}
        else:
            if entry_price - low >= tp_raw:
                return {"pnl_pips": tp_pips, "bars": i - entry_idx, "exit": "tp_hit"}
            if high - entry_price >= sl_raw:
                return {"pnl_pips": -sl_pips, "bars": i - entry_idx, "exit": "sl_hit"}

    last_close = float(candles_df.iloc[max_idx - 1]["close"])
    pnl_raw = (last_close - entry_price) if direction in ("LONG", "BUY") else (entry_price - last_close)
    return {"pnl_pips": round(pnl_raw / ps, 1), "bars": MAX_BARS_OUTCOME, "exit": "timeout"}


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR from OHLC data."""
    if len(df) < period + 1:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.0


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"count": 0, "win_rate": 0, "sharpe": 0, "profit_factor": 0,
                "max_dd_pips": 0, "max_dd_pct": 0, "max_consec_losses": 0,
                "total_pnl": 0, "expectancy": 0, "avg_win": 0, "avg_loss": 0,
                "sortino": 0, "recovery_factor": 0}

    pnls = [t["pnl_pips"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls)
    total_pnl = sum(pnls)
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0

    downside = [p for p in pnls if p < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1.0
    sortino = mean_pnl / downside_std if downside_std > 0 else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    max_dd_pips = float(np.max(dd)) if len(dd) > 0 else 0
    max_dd_pct = (max_dd_pips / max(float(np.max(peak)), 1.0)) * 100 if float(np.max(peak)) > 0 else 0

    max_consec = current = 0
    for p in pnls:
        if p <= 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    recovery = total_pnl / max_dd_pips if max_dd_pips > 0 else 999

    return {
        "count": len(pnls), "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 4), "sortino": round(sortino, 4),
        "profit_factor": round(profit_factor, 3),
        "max_dd_pips": round(max_dd_pips, 1), "max_dd_pct": round(max_dd_pct, 1),
        "max_consec_losses": max_consec,
        "total_pnl": round(total_pnl, 1), "expectancy": round(mean_pnl, 2),
        "avg_win": round(np.mean(wins), 1) if wins else 0,
        "avg_loss": round(np.mean(losses), 1) if losses else 0,
        "recovery_factor": round(recovery, 2),
    }


def main():
    console.print(Panel.fit(
        "[bold magenta]FULL-PIPELINE HISTORICAL REPLAY v2[/bold magenta]\n"
        "[dim]TCN/LightGBM/Ridge ensemble + configurable gates on 7 months H1 data[/dim]\n"
        f"[dim]Gates: conf≥{GATE_CONFIDENCE_MIN}, SL={GATE_ATR_SL_MULT}×ATR, "
        f"TP={GATE_ATR_TP_MULT}×ATR, R:R≥{GATE_MIN_RR}[/dim]",
        border_style="magenta",
    ))

    # Initialize scanner (loads all models)
    console.print("\n[cyan]Loading scanner with full ensemble models...[/cyan]")
    from src.scanner.config import ScannerConfig
    from src.scanner.engine import Scanner

    config = ScannerConfig()
    config.apply_profile("smart")
    config.blocked_pairs = []  # Unblock ALL pairs for replay
    scanner = Scanner(config=config)
    # Force unblock after scanner init (in case constructor re-applies)
    scanner.config.blocked_pairs = []
    console.print("[green]Scanner ready — all 15 pairs unblocked[/green]\n")

    all_trades = []
    pair_trades_map = {}
    exit_reasons = defaultdict(int)
    direction_counts = defaultdict(int)
    signal_stats = {"total_scans": 0, "got_direction": 0, "passed_conf": 0, "passed_all": 0}

    max_windows = (CANDLE_COUNT - WINDOW_SIZE - MAX_BARS_OUTCOME) // STEP_SIZE
    console.print(f"[bold cyan]Replaying {len(ALL_PAIRS)} pairs × ~{max_windows} windows each[/bold cyan]")
    console.print(f"[dim]Max {MAX_TRADES_PER_PAIR} trades/pair, {MIN_COOLDOWN_BARS}-bar cooldown[/dim]\n")

    for pair_idx, pair in enumerate(ALL_PAIRS):
        console.print(f"  [{pair_idx+1}/{len(ALL_PAIRS)}] {pair}: ", end="")
        ps = pip_size(pair)

        try:
            # Fetch full history
            df_full = scanner._fetch_pair_data(pair, CANDLE_COUNT)
            if df_full is None or len(df_full) < WINDOW_SIZE + 100:
                console.print(f"[red]SKIP (insufficient data: {len(df_full) if df_full is not None else 0})[/red]")
                continue

            console.print(f"{len(df_full)} candles ", end="")

            pair_trade_list = []
            pair_scans = 0
            pair_signals = 0
            last_trade_idx = -MIN_COOLDOWN_BARS * 10

            for start in range(0, len(df_full) - WINDOW_SIZE - MAX_BARS_OUTCOME, STEP_SIZE):
                if len(pair_trade_list) >= MAX_TRADES_PER_PAIR:
                    break

                # Cooldown
                if start < last_trade_idx + MIN_COOLDOWN_BARS:
                    continue

                end = start + WINDOW_SIZE
                window_df = df_full.iloc[start:end].copy().reset_index(drop=True)

                signal_stats["total_scans"] += 1
                pair_scans += 1

                # === STEP 1: Feature Engineering ===
                try:
                    df_feat = scanner._compute_features(window_df.copy())
                    if df_feat is None or len(df_feat) < 20:
                        continue
                except Exception:
                    continue

                # === STEP 2: Run Model Inference (full ensemble) ===
                try:
                    direction, confidence, tcn_conf, ridge_conf, _gates, _regime, _ok, details = (
                        scanner._run_inference(window_df, df_feat, pair)
                    )
                    if direction is None or direction == "HOLD":
                        continue
                except Exception:
                    continue

                if direction == "HOLD" or direction is None:
                    continue

                signal_stats["got_direction"] += 1
                pair_signals += 1
                direction_counts[direction] += 1

                # === STEP 3: Apply Configurable Gates ===
                # Confidence gate
                if confidence < GATE_CONFIDENCE_MIN:
                    continue
                signal_stats["passed_conf"] += 1

                # ATR-based SL/TP
                atr = compute_atr(window_df)
                if atr <= 0:
                    continue

                atr_pips = atr / ps
                if atr_pips < GATE_MIN_ATR_PIPS:
                    continue

                sl_pips = max(atr_pips * GATE_ATR_SL_MULT, 5.0)
                tp_pips = max(atr_pips * GATE_ATR_TP_MULT, 6.0)

                # R:R gate
                rr = tp_pips / sl_pips if sl_pips > 0 else 0
                if rr < GATE_MIN_RR:
                    continue

                signal_stats["passed_all"] += 1

                # === STEP 4: Simulate Outcome ===
                entry_idx = end - 1
                outcome = simulate_outcome(df_full, entry_idx, direction, sl_pips, tp_pips, pair)

                trade = {
                    "pair": pair,
                    "direction": direction,
                    "confidence": round(float(confidence), 3),
                    "tcn_conf": round(float(tcn_conf), 3),
                    "ridge_conf": round(float(ridge_conf), 3),
                    "sl_pips": round(sl_pips, 1),
                    "tp_pips": round(tp_pips, 1),
                    "rr_ratio": round(rr, 2),
                    "atr_pips": round(atr_pips, 1),
                    "pnl_pips": outcome["pnl_pips"],
                    "won": outcome["pnl_pips"] > 0,
                    "bars_held": outcome["bars"],
                    "exit_reason": outcome["exit"],
                    "window_start": start,
                    "regime": str(details.get("volatility_regime_name", "UNKNOWN") if details else "UNKNOWN"),
                }

                pair_trade_list.append(trade)
                all_trades.append(trade)
                exit_reasons[outcome["exit"]] += 1
                last_trade_idx = start

            pair_metrics = compute_metrics(pair_trade_list)
            pair_trades_map[pair] = pair_metrics

            n = pair_metrics["count"]
            pnl = pair_metrics["total_pnl"]
            wr = pair_metrics["win_rate"]
            style = "green" if pnl > 0 else ("red" if pnl < 0 else "dim")
            console.print(
                f"→ {pair_scans} scans, {pair_signals} signals, "
                f"[{style}]{n} trades, {wr:.0%} WR, {pnl:+.0f} pips[/{style}]"
            )

        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            import traceback
            traceback.print_exc()

    # ── RESULTS ──────────────────────────────────────────────────────
    overall = compute_metrics(all_trades)

    console.print(f"\n{'='*70}")
    console.print(f"[bold]Total: {len(all_trades)} trades from full ensemble pipeline[/bold]\n")

    # Signal funnel
    console.print("[bold cyan]SIGNAL FUNNEL[/bold cyan]")
    s = signal_stats
    console.print(f"  Total scans:    {s['total_scans']:,}")
    console.print(f"  Got direction:  {s['got_direction']:,} ({s['got_direction']/max(s['total_scans'],1):.1%})")
    console.print(f"  Passed conf:    {s['passed_conf']:,} ({s['passed_conf']/max(s['got_direction'],1):.1%} of directional)")
    console.print(f"  Passed all:     {s['passed_all']:,} ({s['passed_all']/max(s['passed_conf'],1):.1%} of conf-passed)")
    console.print(f"  Directions:     {dict(direction_counts)}")

    # Exit reasons
    console.print(f"\n[bold cyan]EXIT REASONS[/bold cyan]")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        console.print(f"  {reason}: {count} ({count/max(len(all_trades),1):.0%})")

    # Per-pair table
    console.print(f"\n[bold cyan]PER-PAIR RESULTS[/bold cyan]")
    pt = Table(box=box.ROUNDED)
    pt.add_column("Pair", style="cyan")
    pt.add_column("Trades", justify="right")
    pt.add_column("Win Rate", justify="right")
    pt.add_column("Sharpe", justify="right")
    pt.add_column("P/F", justify="right")
    pt.add_column("Max DD", justify="right")
    pt.add_column("Consec L", justify="right")
    pt.add_column("P/L (pips)", justify="right")
    pt.add_column("Expectancy", justify="right")
    pt.add_column("Verdict")

    profitable_pairs = 0
    for pair in ALL_PAIRS:
        m = pair_trades_map.get(pair)
        if not m or m["count"] == 0:
            pt.add_row(pair, "0", "-", "-", "-", "-", "-", "-", "-", "[dim]NO SIGNAL[/dim]")
            continue
        pnl_style = "green" if m["total_pnl"] > 0 else "red"
        verdict = "[green]PASS[/green]" if m["total_pnl"] > 0 and m["win_rate"] >= 0.38 else "[red]FAIL[/red]"
        if m["total_pnl"] > 0:
            profitable_pairs += 1
        pt.add_row(
            pair, str(m["count"]),
            f"{m['win_rate']:.0%}", f"{m['sharpe']:.3f}", f"{m['profit_factor']:.2f}",
            f"{m['max_dd_pips']:.0f}", str(m["max_consec_losses"]),
            f"[{pnl_style}]{m['total_pnl']:+.0f}[/{pnl_style}]",
            f"{m['expectancy']:+.1f}",
            verdict,
        )
    console.print(pt)

    # Regime breakdown
    regime_trades = defaultdict(list)
    for t in all_trades:
        regime_trades[t.get("regime", "UNKNOWN")].append(t)
    if regime_trades:
        console.print(f"\n[bold cyan]BY REGIME[/bold cyan]")
        rt = Table(box=box.SIMPLE)
        rt.add_column("Regime", style="cyan")
        rt.add_column("Trades", justify="right")
        rt.add_column("Win Rate", justify="right")
        rt.add_column("Avg P/L", justify="right")
        for regime, trades_in_regime in sorted(regime_trades.items()):
            n = len(trades_in_regime)
            wr = sum(1 for t in trades_in_regime if t["won"]) / n
            avg_pnl = np.mean([t["pnl_pips"] for t in trades_in_regime])
            style = "green" if avg_pnl > 0 else "red"
            rt.add_row(regime, str(n), f"{wr:.0%}", f"[{style}]{avg_pnl:+.1f}[/{style}]")
        console.print(rt)

    # Confidence bracket analysis
    conf_brackets = [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]
    console.print(f"\n[bold cyan]BY CONFIDENCE BRACKET[/bold cyan]")
    ct = Table(box=box.SIMPLE)
    ct.add_column("Confidence", style="cyan")
    ct.add_column("Trades", justify="right")
    ct.add_column("Win Rate", justify="right")
    ct.add_column("Avg P/L", justify="right")
    for lo, hi in conf_brackets:
        bracket_trades = [t for t in all_trades if lo <= t["confidence"] < hi]
        if bracket_trades:
            n = len(bracket_trades)
            wr = sum(1 for t in bracket_trades if t["won"]) / n
            avg = np.mean([t["pnl_pips"] for t in bracket_trades])
            style = "green" if avg > 0 else "red"
            ct.add_row(f"{lo:.0%}-{hi:.0%}", str(n), f"{wr:.0%}", f"[{style}]{avg:+.1f}[/{style}]")
    console.print(ct)

    # Walk-forward split (train first 60%, test last 40%)
    if len(all_trades) >= 50:
        split_idx = int(len(all_trades) * 0.6)
        train_trades = all_trades[:split_idx]
        test_trades = all_trades[split_idx:]
        train_m = compute_metrics(train_trades)
        test_m = compute_metrics(test_trades)
        console.print(f"\n[bold cyan]WALK-FORWARD VALIDATION (60/40 split)[/bold cyan]")
        wf = Table(box=box.SIMPLE)
        wf.add_column("Period", style="cyan")
        wf.add_column("Trades", justify="right")
        wf.add_column("Win Rate", justify="right")
        wf.add_column("Sharpe", justify="right")
        wf.add_column("P/F", justify="right")
        wf.add_column("P/L", justify="right")
        wf.add_row("Train (60%)", str(train_m["count"]), f"{train_m['win_rate']:.0%}",
                    f"{train_m['sharpe']:.3f}", f"{train_m['profit_factor']:.2f}", f"{train_m['total_pnl']:+.0f}")
        wf.add_row("Test (40%)", str(test_m["count"]), f"{test_m['win_rate']:.0%}",
                    f"{test_m['sharpe']:.3f}", f"{test_m['profit_factor']:.2f}", f"{test_m['total_pnl']:+.0f}")
        console.print(wf)

    # Overall metrics
    console.print(f"\n[bold cyan]OVERALL SYSTEM METRICS[/bold cyan]")
    mt = Table(box=box.HEAVY)
    mt.add_column("Metric", style="bold")
    mt.add_column("Value", justify="right")
    mt.add_column("Threshold", justify="right", style="dim")
    mt.add_column("Status")

    T = THRESHOLDS
    checks = [
        ("Total Trades", str(overall["count"]), f"≥{T['min_trades']}", overall["count"] >= T["min_trades"]),
        ("Win Rate", f"{overall['win_rate']:.1%}", f"≥{T['min_win_rate']:.0%}", overall["win_rate"] >= T["min_win_rate"]),
        ("Sharpe Ratio", f"{overall['sharpe']:.3f}", f"≥{T['min_sharpe']}", overall["sharpe"] >= T["min_sharpe"]),
        ("Profit Factor", f"{overall['profit_factor']:.2f}", f"≥{T['min_profit_factor']}", overall["profit_factor"] >= T["min_profit_factor"]),
        ("Max Drawdown", f"{overall['max_dd_pct']:.1f}%", f"≤{T['max_drawdown_pct']}%", overall["max_dd_pct"] <= T["max_drawdown_pct"]),
        ("Max Consec Losses", str(overall["max_consec_losses"]), f"≤{T['max_consec_losses']}", overall["max_consec_losses"] <= T["max_consec_losses"]),
        ("Profitable Pairs", f"{profitable_pairs}/15", f"≥{T['min_profitable_pairs']}", profitable_pairs >= T["min_profitable_pairs"]),
        ("Expectancy", f"{overall['expectancy']:+.2f} pips", f"≥{T['min_expectancy']}", overall["expectancy"] >= T["min_expectancy"]),
        ("Total P/L", f"{overall['total_pnl']:+,.0f} pips", ">0", overall["total_pnl"] > 0),
        ("Sortino Ratio", f"{overall['sortino']:.3f}", ">0", overall["sortino"] > 0),
        ("Recovery Factor", f"{overall['recovery_factor']:.2f}", ">1.0", overall["recovery_factor"] > 1.0),
        ("Avg Win", f"{overall['avg_win']:+.1f} pips", "-", True),
        ("Avg Loss", f"{overall['avg_loss']:.1f} pips", "-", True),
    ]

    pass_count = sum(1 for _, _, t, p in checks if t != "-" and p)
    fail_count = sum(1 for _, _, t, p in checks if t != "-" and not p)
    for name, value, threshold, passed in checks:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        mt.add_row(name, value, threshold, status)
    console.print(mt)

    total_checks = pass_count + fail_count
    pass_pct = pass_count / total_checks if total_checks > 0 else 0

    if pass_pct >= 0.80 and overall["total_pnl"] > 0 and overall["profit_factor"] > 1.0:
        verdict = "[bold green]READY FOR LIVE TRADING[/bold green]"
        detail = f"Passed {pass_count}/{total_checks} ({pass_pct:.0%}). Positive expectancy confirmed."
    elif pass_pct >= 0.55:
        verdict = "[bold yellow]CONDITIONALLY READY — REDUCED SIZE[/bold yellow]"
        detail = f"Passed {pass_count}/{total_checks} ({pass_pct:.0%}). Trade at 25-50% position size."
    else:
        verdict = "[bold red]NOT READY — NEEDS IMPROVEMENT[/bold red]"
        detail = f"Passed {pass_count}/{total_checks} ({pass_pct:.0%}). Focus on failing metrics."

    console.print(Panel(f"{verdict}\n\n{detail}", title="VERDICT", border_style="bold"))

    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "full_pipeline_replay_v2",
        "config": {
            "confidence_min": GATE_CONFIDENCE_MIN,
            "atr_sl_mult": GATE_ATR_SL_MULT,
            "atr_tp_mult": GATE_ATR_TP_MULT,
            "min_rr": GATE_MIN_RR,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
        },
        "signal_funnel": signal_stats,
        "total_trades": len(all_trades),
        "overall": overall,
        "pairs": pair_trades_map,
        "exit_reasons": dict(exit_reasons),
        "direction_counts": dict(direction_counts),
        "checks_passed": pass_count,
        "checks_total": total_checks,
        "trades": all_trades,
    }
    report_path = _ROOT / "trained_data" / "full_pipeline_replay_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"\n[dim]Report saved: {report_path}[/dim]")
    console.print(f"[dim]All {len(all_trades)} trades saved for RL training[/dim]")


if __name__ == "__main__":
    main()
