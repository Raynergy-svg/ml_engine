"""Multi-asset trend — pure-logic tests (no network, no mocks).

Verifies the single-asset trend rule is long-or-flat + strictly causal (a future price
spike cannot change today's weight) and that an uptrend is held while a downtrend is cash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equity.multi_asset_trend import gate_eval, single_asset_trend_returns


def _series(n, drift, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(drift, 0.01, n)), index=idx)


def test_long_or_flat_and_causal():
    px = _series(600, 0.0005, 1)
    # long-or-flat: gross (cost-free) trend return is (lagged 0/1 weight) * asset return,
    # so |gross| <= |asset return| on every bar (never short, never levered)
    base = single_asset_trend_returns(px, sma_window=100, step=21, cost_bps=0.0)
    aligned = px.pct_change().reindex(base.index)
    assert (base.abs() <= aligned.abs() + 1e-9).all()
    # causality: perturb the LAST price only -> ONLY the last return changes (no lookahead)
    p2 = px.copy()
    p2.iloc[-1] *= 1.5
    b2 = single_asset_trend_returns(p2, sma_window=100, step=21, cost_bps=0.0)
    diff = (base - b2).abs()
    assert (diff.iloc[:-1] < 1e-12).all()   # everything before the last bar is identical
    assert diff.iloc[-1] > 1e-9             # only the last return reflects the perturbed price


def test_uptrend_held_downtrend_flat():
    up = _series(500, 0.003, 2)      # strong uptrend -> mostly held -> positive sum
    down = _series(500, -0.003, 3)   # strong downtrend -> mostly flat -> ~0, not deeply negative
    r_up = single_asset_trend_returns(up, sma_window=100, step=21)
    r_down = single_asset_trend_returns(down, sma_window=100, step=21)
    assert r_up.sum() > 0
    # long-or-flat means the downtrend is sat out in cash, not shorted -> small magnitude
    assert r_down.sum() > down.pct_change().sum()   # better than buy-and-hold the downtrend


def test_gate_eval_shape():
    g = gate_eval(_series(800, 0.0004, 5).pct_change())
    assert set(["net_sharpe", "max_dd", "positive_years", "total_years", "gate_pass"]) <= set(g)
