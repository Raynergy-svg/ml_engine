"""Cost-aware backtest harness — core engine tests (no mocks).

The harness core is signal-agnostic: it takes a real callable
`signal_fn(window_df) -> "LONG"|"SHORT"|None`, so these tests drive it with
deterministic functions and synthetic OHLC — no keras, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.backtest_harness import BacktestResult, run_backtest


def _ohlc(closes: np.ndarray, wick: float = 0.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=len(closes), freq="15min"),
        "open": opens,
        "high": np.maximum(opens, closes) + wick,
        "low": np.minimum(opens, closes) - wick,
        "close": closes,
        "volume": np.full(len(closes), 100),
    })


def test_long_signal_in_uptrend_profits_after_costs():
    df = _ohlc(1.1000 + 0.0010 * np.arange(300))
    res = run_backtest(
        df, lambda w: "LONG", warmup_bars=50,
        spread_pips=1.0, slippage_pips=0.2, pip_size=0.0001,
        sl_atr_mult=1.5, tp_atr_mult=2.0, max_hold_bars=24,
    )
    assert isinstance(res, BacktestResult)
    assert res.n_trades >= 5                      # sequential: one at a time
    assert res.expectancy_pips > 0
    assert res.total_pips > 0
    assert res.win_rate > 0.8
    assert len(res.equity_curve) == res.n_trades


def test_positions_are_sequential_not_overlapping():
    df = _ohlc(1.1000 + 0.0010 * np.arange(300))
    res = run_backtest(df, lambda w: "LONG", warmup_bars=50,
                       spread_pips=1.0, pip_size=0.0001, max_hold_bars=24)
    # Each trade's entry bar must be strictly after the previous trade's exit.
    for prev, nxt in zip(res.trades, res.trades[1:]):
        assert nxt["entry_bar"] > prev["exit_bar"]


def test_fill_at_next_open_no_lookahead():
    df = _ohlc(1.1000 + 0.0010 * np.arange(300))
    res = run_backtest(df, lambda w: "LONG", warmup_bars=50,
                       spread_pips=0.0, slippage_pips=0.0, pip_size=0.0001)
    t = res.trades[0]
    sig_bar = t["entry_bar"] - 1                  # signal fired on prior close
    assert t["entry_price"] == df["open"].iloc[t["entry_bar"]]
    assert t["entry_bar"] == sig_bar + 1


def test_costs_reduce_expectancy_linearly():
    df = _ohlc(1.1000 + 0.0010 * np.arange(300))
    free = run_backtest(df, lambda w: "LONG", warmup_bars=50,
                        spread_pips=0.0, slippage_pips=0.0, pip_size=0.0001)
    costly = run_backtest(df, lambda w: "LONG", warmup_bars=50,
                          spread_pips=3.0, slippage_pips=1.0, pip_size=0.0001)
    assert abs((free.expectancy_pips - costly.expectancy_pips) - 4.0) < 1e-6


def test_abstaining_signal_produces_no_trades_and_fails_gate():
    df = _ohlc(1.1000 + 0.0010 * np.arange(300))
    res = run_backtest(df, lambda w: None, warmup_bars=50,
                       spread_pips=1.0, pip_size=0.0001)
    assert res.n_trades == 0
    assert res.gate_passed is False               # fail closed on no evidence


def test_short_in_downtrend_profits():
    df = _ohlc(1.2000 - 0.0010 * np.arange(300))
    res = run_backtest(df, lambda w: "SHORT", warmup_bars=50,
                       spread_pips=1.0, pip_size=0.0001)
    assert res.expectancy_pips > 0
    assert res.short_share == 1.0


def test_max_drawdown_tracked_on_losing_streak():
    # Whipsaw: signal LONG into a steady downtrend -> consecutive SL hits.
    df = _ohlc(1.2000 - 0.0010 * np.arange(300), wick=0.0005)
    res = run_backtest(df, lambda w: "LONG", warmup_bars=50,
                       spread_pips=1.0, pip_size=0.0001)
    assert res.expectancy_pips < 0
    assert res.max_drawdown_pips > 0
    assert res.gate_passed is False


def test_report_dict_complete():
    df = _ohlc(1.1000 + 0.0010 * np.arange(300))
    res = run_backtest(df, lambda w: "LONG", warmup_bars=50,
                       spread_pips=1.0, pip_size=0.0001)
    rep = res.as_report()
    for key in ("n_trades", "win_rate", "expectancy_pips", "profit_factor",
                "total_pips", "max_drawdown_pips", "t_stat", "balanced_accuracy",
                "long_share", "short_share", "avg_hold_bars", "exposure_pct",
                "gate_passed", "gate_reasons"):
        assert key in rep, key
