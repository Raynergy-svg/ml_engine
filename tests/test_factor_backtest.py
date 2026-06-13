"""No-mock tests for the cost-aware backtest (US-006).

Includes the lookahead canary and the zero-cost vs with-cost sanity spread.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.factor.backtest import run_backtest, write_report, BacktestResult


def _prices(n=600, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n, tz="UTC")
    pairs = ["EUR_USD", "USD_JPY"]
    out = {}
    for p in pairs:
        out[p] = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.006, n))), index=idx)
    return pd.DataFrame(out)


def test_lookahead_canary_cheating_beats_honest():
    """A position aligned to the SAME-bar return (lag 0/neg) must look better than
    the honestly-lagged one. If the honest pipeline ever matched the cheat, a
    future leak would be hiding."""
    close = _prices()
    ret = close.pct_change()
    # "Oracle" weights = sign of the contemporaneous return (uses the future).
    oracle = np.sign(ret).fillna(0.0)

    honest = run_backtest(oracle, close, execution_lag=1,
                          spreads_pips={"EUR_USD": 0, "USD_JPY": 0},
                          financing_markup_annual=0.0)
    cheating = run_backtest(oracle, close, execution_lag=0,
                            spreads_pips={"EUR_USD": 0, "USD_JPY": 0},
                            financing_markup_annual=0.0)
    assert cheating.net_sharpe > honest.net_sharpe + 1.0


def test_zero_cost_dominates_with_cost():
    close = _prices(seed=5)
    rng = np.random.default_rng(9)
    w = pd.DataFrame(rng.uniform(-0.3, 0.3, close.shape),
                     index=close.index, columns=close.columns)
    free = run_backtest(w, close, spreads_pips={"EUR_USD": 0, "USD_JPY": 0},
                        financing_markup_annual=0.0)
    costed = run_backtest(w, close)
    assert free.gross_sharpe == free.net_sharpe        # no costs -> identical
    assert costed.net_sharpe <= free.net_sharpe + 1e-9  # costs only hurt


def test_carry_accrual_credits_pnl():
    close = _prices(seed=11)
    w = pd.DataFrame(0.5, index=close.index, columns=close.columns)  # long both
    accr = pd.DataFrame(0.0001, index=close.index, columns=close.columns)  # +carry
    without = run_backtest(w, close, financing_markup_annual=0.0,
                           spreads_pips={"EUR_USD": 0, "USD_JPY": 0})
    withc = run_backtest(w, close, carry_accrual=accr, financing_markup_annual=0.0,
                         spreads_pips={"EUR_USD": 0, "USD_JPY": 0})
    assert withc.net_cagr > without.net_cagr


def test_write_report_atomic_and_readable(tmp_path):
    close = _prices()
    w = pd.DataFrame(0.1, index=close.index, columns=close.columns)
    res = run_backtest(w, close)
    assert isinstance(res, BacktestResult)
    path = write_report(res, out_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    for key in ("net_sharpe", "max_drawdown", "per_year_returns", "walk_forward"):
        assert key in data
