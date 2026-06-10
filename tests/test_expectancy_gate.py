"""Tests for the cost-aware expectancy gate (no mocks — real numpy/pandas).

The expectancy gate is the promotion-time answer to "is this model profitable
after costs?" — the question raw directional accuracy cannot answer (an
all-SHORT predictor scores ~70% on a SHORT-heavy slice while being untradeable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.expectancy_gate import (
    ExpectancyReport,
    gate_decision,
    simulate_trades,
)


def _ohlc(closes: np.ndarray, wick: float = 0.0) -> pd.DataFrame:
    """Build a minimal OHLC frame from a close path (highs/lows pad by wick)."""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) + wick,
        "low": np.minimum(opens, closes) - wick,
        "close": closes,
        "volume": np.full(len(closes), 100),
    })


def _trending_up(n: int = 120, step: float = 0.0010) -> pd.DataFrame:
    """Steady uptrend: +10 pips per bar on a EUR_USD-like price."""
    return _ohlc(1.1000 + step * np.arange(n))


def test_long_in_uptrend_is_profitable_after_small_spread():
    df = _trending_up()
    preds = [(i, "LONG") for i in range(20, 60)]
    report = simulate_trades(
        df, preds, spread_pips=1.0, pip_size=0.0001,
        sl_atr_mult=1.5, tp_atr_mult=2.0, max_hold_bars=24,
    )
    assert isinstance(report, ExpectancyReport)
    assert report.n_trades == 40
    assert report.win_rate > 0.9          # uptrend: longs hit TP
    assert report.expectancy_pips > 0     # profitable after 1-pip spread
    assert report.profit_factor > 1.0


def test_huge_spread_turns_same_signal_unprofitable():
    df = _trending_up()
    preds = [(i, "LONG") for i in range(20, 60)]
    cheap = simulate_trades(df, preds, spread_pips=1.0, pip_size=0.0001)
    pricey = simulate_trades(df, preds, spread_pips=50.0, pip_size=0.0001)
    assert pricey.expectancy_pips < cheap.expectancy_pips
    assert pricey.expectancy_pips < 0     # 50-pip cost swamps the edge


def test_balanced_accuracy_punishes_one_class_collapse():
    # Two-sided market: uptrend then downtrend, so actual directions over the
    # hold window are genuinely mixed (LONG half, SHORT half).
    up = 1.1000 + 0.0010 * np.arange(60)
    down = up[-1] - 0.0010 * np.arange(1, 61)
    df = _ohlc(np.concatenate([up, down]))
    # All-SHORT predictor across both halves: recall_short=1, recall_long=0
    # -> balanced ~0.5, and class_collapse must trip (100% SHORT predictions).
    preds = [(i, "SHORT") for i in range(15, 100)]
    report = simulate_trades(df, preds, spread_pips=1.0, pip_size=0.0001)
    assert report.short_share == 1.0
    assert report.class_collapse is True
    assert report.balanced_accuracy < 0.62  # no real two-sided skill


def test_gate_fails_closed_on_too_few_trades():
    df = _trending_up()
    preds = [(25, "LONG"), (30, "LONG")]   # only 2 trades
    report = simulate_trades(df, preds, spread_pips=1.0, pip_size=0.0001)
    passed, reasons = gate_decision(report, min_trades=20)
    assert passed is False
    assert any("trades" in r for r in reasons)


def test_gate_passes_profitable_balanced_model():
    # Uptrend then downtrend; predict correctly in both halves -> two-sided skill.
    up = 1.1000 + 0.0010 * np.arange(60)
    down = up[-1] - 0.0010 * np.arange(1, 61)
    df = _ohlc(np.concatenate([up, down]))
    preds = [(i, "LONG") for i in range(15, 40)] + [(i, "SHORT") for i in range(75, 100)]
    report = simulate_trades(df, preds, spread_pips=1.0, pip_size=0.0001)
    passed, reasons = gate_decision(report, min_trades=20)
    assert report.balanced_accuracy > 0.9
    assert report.class_collapse is False
    assert report.expectancy_pips > 0
    assert passed is True, reasons


def test_gate_rejects_class_collapse_even_when_expectancy_positive():
    df = _trending_up()
    preds = [(i, "LONG") for i in range(20, 60)]   # 100% LONG, profitable
    report = simulate_trades(df, preds, spread_pips=1.0, pip_size=0.0001)
    assert report.expectancy_pips > 0
    passed, reasons = gate_decision(report, min_trades=20)
    assert passed is False
    assert any("collapse" in r or "share" in r for r in reasons)


def test_jpy_pip_size_respected():
    closes = 150.00 + 0.10 * np.arange(120)   # USD_JPY-like, +10 pips/bar
    df = _ohlc(closes)
    preds = [(i, "LONG") for i in range(20, 50)]
    report = simulate_trades(df, preds, spread_pips=1.5, pip_size=0.01)
    assert report.n_trades == 30
    assert report.expectancy_pips > 0
