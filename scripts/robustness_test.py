#!/usr/bin/env python3
"""
Robustness Test Suite for Buddy Trading System

Generates thousands of trades across all pairs via historical replay,
then evaluates with institutional-grade metrics to determine if the
system is ready for live trading.

Tests:
  1. Per-pair profitability (Sharpe, win rate, profit factor)
  2. Walk-forward validation (train/test split, no look-ahead)
  3. Drawdown analysis (max DD, recovery time, consecutive losses)
  4. Regime robustness (performance in different volatility regimes)
  5. Overall system verdict: READY / NOT READY
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

ALL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "GBP_AUD", "EUR_CHF", "GBP_CHF",
]

CANDLES = 5000  # ~7 months H1

# Thresholds for LIVE READINESS
MIN_TOTAL_TRADES = 2000
MIN_OVERALL_WIN_RATE = 0.38
MIN_OVERALL_SHARPE = 0.05
MAX_ALLOWED_DRAWDOWN_PCT = 25.0  # % of peak equity
MIN_PROFIT_FACTOR = 1.05  # gross profit / gross loss
MAX_CONSECUTIVE_LOSSES = 15
MIN_PROFITABLE_PAIRS = 5  # out of 15
MIN_WALK_FORWARD_WIN_RATE = 0.35  # out-of-sample

_scanner = None


def get_scanner():
    global _scanner
    if _scanner is None:
        from src.scanner.config import ScannerConfig
        from src.scanner.engine import Scanner
        config = ScannerConfig()
        _scanner = Scanner(config=config)
    return _scanner


def fetch_candles(pair: str) -> list:
    scanner = get_scanner()
    df = scanner._fetch_pair_data(pair, CANDLES)
    if df is None or df.empty:
        return []
    candles = []
    for idx, row in df.iterrows():
        candles.append({
            "time": str(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row.get("volume", 0)),
        })
    return candles


def run_replay(pair: str, candles: list, config: dict) -> object:
    from src.scanner.automation.replay_validator import ReplayValidator
    validator = ReplayValidator()
    return validator.replay_scan(pair, candles, config, "robustness")


def compute_metrics(trades: list) -> dict:
    """Compute institutional-grade metrics from a trade list."""
    if not trades:
        return {
            "count": 0, "win_rate": 0, "sharpe": 0, "sortino": 0,
            "profit_factor": 0, "max_dd_pips": 0, "max_dd_pct": 0,
            "max_consec_losses": 0, "avg_win": 0, "avg_loss": 0,
            "total_pnl": 0, "expectancy": 0, "recovery_factor": 0,
        }

    pnls = [t.get("pnl_pips", 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Basic stats
    win_rate = len(wins) / len(pnls) if pnls else 0
    total_pnl = sum(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Sharpe ratio (annualized assuming H1 candles)
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0

    # Sortino (downside deviation only)
    downside = [p for p in pnls if p < 0]
    downside_std = np.std(downside) if downside else 1.0
    sortino = mean_pnl / downside_std if downside_std > 0 else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    max_dd_pips = float(np.max(dd)) if len(dd) > 0 else 0
    max_dd_pct = (max_dd_pips / max(float(np.max(peak)), 1.0)) * 100

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for p in pnls:
        if p <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Expectancy (avg pip gain per trade)
    expectancy = mean_pnl

    # Recovery factor = total P/L / max drawdown
    recovery_factor = total_pnl / max_dd_pips if max_dd_pips > 0 else float("inf")

    return {
        "count": len(pnls),
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "profit_factor": round(profit_factor, 4),
        "max_dd_pips": round(max_dd_pips, 1),
        "max_dd_pct": round(max_dd_pct, 1),
        "max_consec_losses": max_consec,
        "avg_win": round(avg_win, 1),
        "avg_loss": round(avg_loss, 1),
        "total_pnl": round(total_pnl, 1),
        "expectancy": round(expectancy, 2),
        "recovery_factor": round(recovery_factor, 2),
    }


def classify_regime(candles: list, idx: int, lookback: int = 20) -> str:
    """Classify volatility regime from recent candles."""
    if idx < lookback:
        return "UNKNOWN"
    recent = candles[idx - lookback:idx]
    ranges = []
    for c in recent:
        h = float(c.get("high", 0))
        l = float(c.get("low", 0))
        if h > 0 and l > 0:
            ranges.append((h - l) / l)
    if not ranges:
        return "UNKNOWN"
    avg_range = np.mean(ranges)
    if avg_range < 0.003:
        return "LOW"
    elif avg_range < 0.008:
        return "NORMAL"
    elif avg_range < 0.015:
        return "HIGH"
    else:
        return "EXTREME"


def main():
    console.print(Panel.fit(
        "[bold magenta]BUDDY ROBUSTNESS TEST SUITE[/bold magenta]\n"
        "[dim]Institutional-grade validation for live trading readiness[/dim]",
        border_style="magenta",
    ))

    # ── CONFIG VARIANTS ──────────────────────────────────────────────
    configs = {
        "tight": {
            "confidence_threshold": 0.50,
            "rr_threshold": 1.5,
            "atr_sl_multiplier": 1.0,
            "atr_tp_multiplier": 2.0,
        },
        "balanced": {
            "confidence_threshold": 0.40,
            "rr_threshold": 1.2,
            "atr_sl_multiplier": 1.0,
            "atr_tp_multiplier": 1.8,
        },
        "loose": {
            "confidence_threshold": 0.35,
            "rr_threshold": 1.2,
            "atr_sl_multiplier": 0.8,
            "atr_tp_multiplier": 1.5,
        },
    }

    all_trades = []
    pair_results = {}
    regime_trades = defaultdict(list)
    walk_forward_results = {}

    console.print(f"\n[bold cyan]Phase 1: Generating trades across {len(ALL_PAIRS)} pairs × 3 configs[/bold cyan]")
    console.print(f"  Target: {MIN_TOTAL_TRADES}+ trades for statistical significance\n")

    for pair in ALL_PAIRS:
        try:
            console.print(f"  {pair}: fetching...", end="")
            candles = fetch_candles(pair)
            if len(candles) < 200:
                console.print(f" [red]SKIP ({len(candles)} candles)[/red]")
                continue
            console.print(f" {len(candles)} candles", end="")

            pair_trades = []
            for config_name, config in configs.items():
                result = run_replay(pair, candles, config)
                for t in result.trades:
                    t["pair"] = pair
                    t["config"] = config_name
                    # Classify regime at entry
                    t["regime"] = classify_regime(candles, t.get("candle_idx", 20))
                pair_trades.extend(result.trades)

            all_trades.extend(pair_trades)

            # Per-pair metrics (combined across configs)
            pair_metrics = compute_metrics(pair_trades)
            pair_results[pair] = pair_metrics

            # Regime classification
            for t in pair_trades:
                regime_trades[t.get("regime", "UNKNOWN")].append(t)

            # Walk-forward: first 60% train, last 40% test
            split_idx = int(len(candles) * 0.6)
            train_candles = candles[:split_idx + 50]  # overlap for ATR lookback
            test_candles = candles[split_idx:]

            wf_train = run_replay(pair, train_candles, configs["balanced"])
            wf_test = run_replay(pair, test_candles, configs["balanced"])
            walk_forward_results[pair] = {
                "train": compute_metrics(wf_train.trades),
                "test": compute_metrics(wf_test.trades),
            }

            wr = pair_metrics["win_rate"]
            pnl = pair_metrics["total_pnl"]
            style = "green" if pnl > 0 else "red"
            console.print(f" → [{style}]{len(pair_trades)} trades, {wr:.0%} WR, {pnl:+.0f} pips[/{style}]")

        except Exception as e:
            console.print(f" [red]ERROR: {e}[/red]")

    # ── RESULTS ──────────────────────────────────────────────────────
    console.print(f"\n[bold]Total trades generated: {len(all_trades)}[/bold]\n")

    # ── 1. Per-Pair Table ────────────────────────────────────────────
    console.print("[bold cyan]1. PER-PAIR ANALYSIS[/bold cyan]")
    pair_table = Table(box=box.ROUNDED, title="Performance by Pair")
    pair_table.add_column("Pair", style="cyan")
    pair_table.add_column("Trades", justify="right")
    pair_table.add_column("Win Rate", justify="right")
    pair_table.add_column("Sharpe", justify="right")
    pair_table.add_column("P/F", justify="right")
    pair_table.add_column("Max DD", justify="right")
    pair_table.add_column("Consec L", justify="right")
    pair_table.add_column("P/L (pips)", justify="right")
    pair_table.add_column("Verdict")

    profitable_pairs = 0
    for pair in ALL_PAIRS:
        m = pair_results.get(pair)
        if not m or m["count"] == 0:
            pair_table.add_row(pair, "0", "-", "-", "-", "-", "-", "-", "[dim]NO DATA[/dim]")
            continue
        pnl_style = "green" if m["total_pnl"] > 0 else "red"
        verdict = "[green]PASS[/green]" if m["total_pnl"] > 0 and m["win_rate"] > 0.30 else "[red]FAIL[/red]"
        if m["total_pnl"] > 0:
            profitable_pairs += 1
        pair_table.add_row(
            pair,
            str(m["count"]),
            f"{m['win_rate']:.0%}",
            f"{m['sharpe']:.2f}",
            f"{m['profit_factor']:.2f}",
            f"{m['max_dd_pips']:.0f}",
            str(m["max_consec_losses"]),
            f"[{pnl_style}]{m['total_pnl']:+.0f}[/{pnl_style}]",
            verdict,
        )
    console.print(pair_table)

    # ── 2. Walk-Forward Validation ───────────────────────────────────
    console.print("\n[bold cyan]2. WALK-FORWARD VALIDATION (60% train / 40% test)[/bold cyan]")
    wf_table = Table(box=box.ROUNDED, title="Out-of-Sample Performance")
    wf_table.add_column("Pair", style="cyan")
    wf_table.add_column("Train WR", justify="right")
    wf_table.add_column("Test WR", justify="right")
    wf_table.add_column("Degradation", justify="right")
    wf_table.add_column("Test P/L", justify="right")
    wf_table.add_column("OOS Verdict")

    wf_pass_count = 0
    for pair in ALL_PAIRS:
        wf = walk_forward_results.get(pair)
        if not wf or wf["train"]["count"] == 0:
            continue
        train_wr = wf["train"]["win_rate"]
        test_wr = wf["test"]["win_rate"]
        degradation = train_wr - test_wr if train_wr > 0 else 0
        test_pnl = wf["test"]["total_pnl"]
        pnl_style = "green" if test_pnl > 0 else "red"
        oos_pass = test_wr >= MIN_WALK_FORWARD_WIN_RATE and test_pnl > -100
        if oos_pass:
            wf_pass_count += 1
        wf_table.add_row(
            pair,
            f"{train_wr:.0%}",
            f"{test_wr:.0%}",
            f"[{'red' if degradation > 0.10 else 'green'}]{degradation:+.0%}[/{'red' if degradation > 0.10 else 'green'}]",
            f"[{pnl_style}]{test_pnl:+.0f}[/{pnl_style}]",
            "[green]PASS[/green]" if oos_pass else "[red]FAIL[/red]",
        )
    console.print(wf_table)

    # ── 3. Regime Analysis ───────────────────────────────────────────
    console.print("\n[bold cyan]3. REGIME ROBUSTNESS[/bold cyan]")
    regime_table = Table(box=box.ROUNDED, title="Performance by Volatility Regime")
    regime_table.add_column("Regime", style="cyan")
    regime_table.add_column("Trades", justify="right")
    regime_table.add_column("Win Rate", justify="right")
    regime_table.add_column("Sharpe", justify="right")
    regime_table.add_column("P/L (pips)", justify="right")

    for regime in ["LOW", "NORMAL", "HIGH", "EXTREME", "UNKNOWN"]:
        trades = regime_trades.get(regime, [])
        if not trades:
            continue
        m = compute_metrics(trades)
        pnl_style = "green" if m["total_pnl"] > 0 else "red"
        regime_table.add_row(
            regime,
            str(m["count"]),
            f"{m['win_rate']:.0%}",
            f"{m['sharpe']:.2f}",
            f"[{pnl_style}]{m['total_pnl']:+.0f}[/{pnl_style}]",
        )
    console.print(regime_table)

    # ── 4. Overall System Metrics ────────────────────────────────────
    console.print("\n[bold cyan]4. OVERALL SYSTEM METRICS[/bold cyan]")
    overall = compute_metrics(all_trades)

    metrics_table = Table(box=box.HEAVY, title="Aggregate Performance")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_column("Threshold", justify="right", style="dim")
    metrics_table.add_column("Status")

    checks = [
        ("Total Trades", str(overall["count"]), f"≥{MIN_TOTAL_TRADES}", overall["count"] >= MIN_TOTAL_TRADES),
        ("Win Rate", f"{overall['win_rate']:.1%}", f"≥{MIN_OVERALL_WIN_RATE:.0%}", overall["win_rate"] >= MIN_OVERALL_WIN_RATE),
        ("Sharpe Ratio", f"{overall['sharpe']:.3f}", f"≥{MIN_OVERALL_SHARPE}", overall["sharpe"] >= MIN_OVERALL_SHARPE),
        ("Profit Factor", f"{overall['profit_factor']:.2f}", f"≥{MIN_PROFIT_FACTOR}", overall["profit_factor"] >= MIN_PROFIT_FACTOR),
        ("Max Drawdown", f"{overall['max_dd_pct']:.1f}%", f"≤{MAX_ALLOWED_DRAWDOWN_PCT}%", overall["max_dd_pct"] <= MAX_ALLOWED_DRAWDOWN_PCT),
        ("Max Consec Losses", str(overall["max_consec_losses"]), f"≤{MAX_CONSECUTIVE_LOSSES}", overall["max_consec_losses"] <= MAX_CONSECUTIVE_LOSSES),
        ("Profitable Pairs", f"{profitable_pairs}/15", f"≥{MIN_PROFITABLE_PAIRS}", profitable_pairs >= MIN_PROFITABLE_PAIRS),
        ("WF OOS Pass Rate", f"{wf_pass_count}/{len(walk_forward_results)}", f"≥50%", wf_pass_count >= len(walk_forward_results) * 0.5),
        ("Total P/L (pips)", f"{overall['total_pnl']:+,.0f}", ">0", overall["total_pnl"] > 0),
        ("Expectancy/Trade", f"{overall['expectancy']:+.2f} pips", ">0", overall["expectancy"] > 0),
        ("Sortino Ratio", f"{overall['sortino']:.3f}", ">0", overall["sortino"] > 0),
        ("Recovery Factor", f"{overall['recovery_factor']:.2f}", ">1.0", overall["recovery_factor"] > 1.0),
        ("Avg Win", f"{overall['avg_win']:+.1f} pips", "-", True),
        ("Avg Loss", f"{overall['avg_loss']:.1f} pips", "-", True),
    ]

    pass_count = 0
    fail_count = 0
    for name, value, threshold, passed in checks:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        if threshold != "-":
            if passed:
                pass_count += 1
            else:
                fail_count += 1
        metrics_table.add_row(name, value, threshold, status)
    console.print(metrics_table)

    # ── 5. FINAL VERDICT ─────────────────────────────────────────────
    total_checks = pass_count + fail_count
    pass_pct = pass_count / total_checks if total_checks > 0 else 0

    if pass_pct >= 0.80 and overall["total_pnl"] > 0 and overall["profit_factor"] > 1.0:
        verdict = "[bold green]READY FOR LIVE TRADING[/bold green]"
        verdict_detail = (
            f"Passed {pass_count}/{total_checks} checks ({pass_pct:.0%}). "
            f"System shows positive expectancy across {profitable_pairs} pairs."
        )
    elif pass_pct >= 0.60:
        verdict = "[bold yellow]CONDITIONALLY READY — REDUCE SIZE[/bold yellow]"
        verdict_detail = (
            f"Passed {pass_count}/{total_checks} checks ({pass_pct:.0%}). "
            f"Consider trading at 50% position size until more live data confirms edge."
        )
    else:
        verdict = "[bold red]NOT READY — NEEDS MORE TRAINING[/bold red]"
        verdict_detail = (
            f"Passed {pass_count}/{total_checks} checks ({pass_pct:.0%}). "
            f"Key failures need resolution before live trading."
        )

    console.print(Panel(
        f"{verdict}\n\n{verdict_detail}",
        title="VERDICT",
        border_style="bold",
    ))

    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trades": len(all_trades),
        "overall_metrics": overall,
        "pair_results": pair_results,
        "regime_results": {r: compute_metrics(t) for r, t in regime_trades.items()},
        "walk_forward": {p: v for p, v in walk_forward_results.items()},
        "checks_passed": pass_count,
        "checks_total": total_checks,
        "verdict": "READY" if pass_pct >= 0.80 else ("CONDITIONAL" if pass_pct >= 0.60 else "NOT_READY"),
    }
    report_path = _ROOT / "trained_data" / "robustness_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"\n[dim]Report saved to {report_path}[/dim]")


if __name__ == "__main__":
    main()
