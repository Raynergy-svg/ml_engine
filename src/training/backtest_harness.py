"""Cost-aware, event-driven backtest harness.

The honest-number machine: walks bars sequentially, asks a signal callable for
a direction on each close while flat, fills at the NEXT bar's open (no
lookahead), manages one ATR-bracketed position at a time (mirroring the live
one-position-per-pair model), charges spread+slippage per trade, and produces
an equity curve + the same fail-closed gate verdict the promotion path uses.

Relationship to expectancy_gate.py: that module scores *prediction sets*
(every prediction an independent bracket — right for holdout gating); this
module simulates *sequential trading* (right for equity curves, drawdown, and
exposure). Both share the pessimistic SL-wins-ties intra-bar rule.

The core is signal-agnostic (`signal_fn(window_df) -> "LONG"|"SHORT"|None`)
so it is fully testable with deterministic callables; scripts/backtest_harness.py
wires the real GateEvaluator transformer path as the signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.expectancy_gate import _atr, gate_decision, ExpectancyReport

logger = logging.getLogger(__name__)

SignalFn = Callable[[pd.DataFrame], Optional[str]]


@dataclass
class BacktestResult:
    """Sequential-simulation outcome + the promotion-style gate verdict."""
    n_trades: int = 0
    win_rate: float = 0.0
    expectancy_pips: float = 0.0
    profit_factor: float = 0.0
    total_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    t_stat: float = 0.0
    balanced_accuracy: float = 0.0
    long_share: float = 0.0
    short_share: float = 0.0
    class_collapse: bool = False
    avg_hold_bars: float = 0.0
    exposure_pct: float = 0.0
    spread_pips: float = 0.0
    slippage_pips: float = 0.0
    gate_passed: bool = False
    gate_reasons: List[str] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def as_report(self) -> Dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "expectancy_pips": round(self.expectancy_pips, 3),
            "profit_factor": round(self.profit_factor, 3),
            "total_pips": round(self.total_pips, 1),
            "max_drawdown_pips": round(self.max_drawdown_pips, 1),
            "t_stat": round(self.t_stat, 2),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "long_share": round(self.long_share, 4),
            "short_share": round(self.short_share, 4),
            "class_collapse": self.class_collapse,
            "avg_hold_bars": round(self.avg_hold_bars, 1),
            "exposure_pct": round(self.exposure_pct, 1),
            "spread_pips": self.spread_pips,
            "slippage_pips": self.slippage_pips,
            "gate_passed": self.gate_passed,
            "gate_reasons": list(self.gate_reasons),
            "equity_curve": [round(e, 2) for e in self.equity_curve],
        }


def run_backtest(
    df: pd.DataFrame,
    signal_fn: SignalFn,
    *,
    warmup_bars: int = 150,
    spread_pips: float,
    pip_size: float,
    slippage_pips: float = 0.2,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
    atr_period: int = 14,
    max_hold_bars: int = 24,
    min_trades: int = 20,
    min_balanced_accuracy: float = 0.52,
    min_expectancy_pips: float = 0.0,
    max_class_share: float = 0.85,
) -> BacktestResult:
    """Event-driven sequential simulation of `signal_fn` over `df`.

    Per bar while FLAT: call signal_fn on the window up to and including the
    current close; on LONG/SHORT, enter at the NEXT bar's open with
    SL/TP = ATR(entry-signal bar) * mults. While IN a position: walk bars,
    pessimistic SL-first on intra-bar double-touch, time-stop exit at the
    window close after max_hold_bars. spread+slippage deducted once per trade.

    Returns a BacktestResult whose gate verdict reuses the promotion gate's
    fail-closed rules (insufficient trades = FAIL).
    """
    closes = df["close"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    atr = _atr(df, atr_period)
    n = len(df)
    cost_pips = float(spread_pips) + float(slippage_pips)

    trades: List[Dict[str, Any]] = []
    equity: List[float] = []
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    bars_in_market = 0
    n_long = 0
    n_short = 0
    al_t = al_c = as_t = as_c = 0  # balanced-accuracy ledgers

    i = max(int(warmup_bars), atr_period)
    while i < n - 2:
        direction = signal_fn(df.iloc[: i + 1])
        if direction not in ("LONG", "SHORT"):
            i += 1
            continue

        bracket = atr[i]
        if not np.isfinite(bracket) or bracket <= 0:
            i += 1
            continue

        entry_bar = i + 1
        entry = opens[entry_bar]
        sl_dist = bracket * sl_atr_mult
        tp_dist = bracket * tp_atr_mult
        if direction == "LONG":
            n_long += 1
            sl_price, tp_price = entry - sl_dist, entry + tp_dist
        else:
            n_short += 1
            sl_price, tp_price = entry + sl_dist, entry - tp_dist

        end = min(entry_bar + max_hold_bars, n - 1)
        exit_bar = end
        gross: Optional[float] = None
        for j in range(entry_bar, end + 1):
            if direction == "LONG":
                hit_sl = lows[j] <= sl_price
                hit_tp = highs[j] >= tp_price
            else:
                hit_sl = highs[j] >= sl_price
                hit_tp = lows[j] <= tp_price
            if hit_sl:  # pessimistic: SL wins intra-bar ties
                gross, exit_bar = -sl_dist / pip_size, j
                break
            if hit_tp:
                gross, exit_bar = tp_dist / pip_size, j
                break
        if gross is None:  # time-stop at window close
            move = closes[end] - entry
            gross = (move if direction == "LONG" else -move) / pip_size
            exit_bar = end

        # Balanced-accuracy ledger vs realized direction over the hold.
        realized_up = closes[exit_bar] > entry
        if realized_up:
            al_t += 1
            al_c += int(direction == "LONG")
        else:
            as_t += 1
            as_c += int(direction == "SHORT")

        net = gross - cost_pips
        cum += net
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        bars_in_market += exit_bar - entry_bar + 1
        equity.append(cum)
        trades.append({
            "entry_bar": int(entry_bar),
            "exit_bar": int(exit_bar),
            "direction": direction,
            "entry_price": float(entry),
            "net_pips": float(net),
            "hold_bars": int(exit_bar - entry_bar + 1),
        })
        i = exit_bar + 1  # sequential: stay flat until past the exit

    res = BacktestResult(spread_pips=float(spread_pips), slippage_pips=float(slippage_pips))
    res.trades = trades
    res.equity_curve = equity
    res.n_trades = len(trades)
    if trades:
        pips = np.asarray([t["net_pips"] for t in trades])
        wins = pips[pips > 0]
        losses = pips[pips <= 0]
        res.win_rate = len(wins) / len(pips)
        res.expectancy_pips = float(pips.mean())
        res.total_pips = float(pips.sum())
        gross_loss = float(-losses.sum()) if len(losses) else 0.0
        res.profit_factor = (float(wins.sum()) / gross_loss) if gross_loss > 0 else float("inf")
        res.max_drawdown_pips = float(max_dd)
        sd = float(pips.std(ddof=1)) if len(pips) > 1 else 0.0
        res.t_stat = float(pips.mean() / (sd / np.sqrt(len(pips)))) if sd > 0 else 0.0
        res.avg_hold_bars = float(np.mean([t["hold_bars"] for t in trades]))
        traded_span = n - max(int(warmup_bars), atr_period)
        res.exposure_pct = 100.0 * bars_in_market / traded_span if traded_span > 0 else 0.0
        total_dir = n_long + n_short
        res.long_share = n_long / total_dir if total_dir else 0.0
        res.short_share = n_short / total_dir if total_dir else 0.0
        res.class_collapse = max(res.long_share, res.short_share) > max_class_share
        recalls = []
        if al_t:
            recalls.append(al_c / al_t)
        if as_t:
            recalls.append(as_c / as_t)
        res.balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0

    # Promotion-style fail-closed verdict (shared thresholds with the
    # expectancy gate so backtest and holdout agree on what "good" means).
    rep = ExpectancyReport(
        n_trades=res.n_trades,
        balanced_accuracy=res.balanced_accuracy,
        long_share=res.long_share,
        short_share=res.short_share,
        class_collapse=res.class_collapse,
        expectancy_pips=res.expectancy_pips,
        profit_factor=res.profit_factor,
        spread_pips=res.spread_pips,
    )
    res.gate_passed, res.gate_reasons = gate_decision(
        rep,
        min_trades=min_trades,
        min_balanced_accuracy=min_balanced_accuracy,
        min_expectancy_pips=min_expectancy_pips,
        max_class_share=max_class_share,
    )
    return res
