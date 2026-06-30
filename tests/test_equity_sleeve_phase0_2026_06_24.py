"""Phase 0 — multi-asset trend sleeve + HRP-across-sleeves combiner (offline; running:NO).

Verifies the trend sleeve weights and the sleeve combiner are STRICTLY CAUSAL
(perturbing an input at t cannot change weights at any index <= t — the #1 lie
vector), long-only / long-or-flat (cash allowed), and that the combined-book gate
faithfully scores a return series via the real `stats` + `evaluate_gate`.

No mocks: real synthetic price/return panels + real `backtest`/`ship_gate`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equity.backtest import stats
from src.equity.sleeve_combiner import combine_sleeves, gate_combined
from src.equity.trend_sleeve import trend_sleeve_weights


def _prices(seed=0, n=600, cols=("SPY", "TLT", "GLD")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n, tz="UTC").normalize()
    out = {}
    drifts = {0: 0.0007, 1: -0.0005, 2: 0.0002}
    for k, c in enumerate(cols):
        rets = rng.normal(drifts.get(k, 0.0), 0.010, n)
        out[c] = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(out, index=idx)


def _ret(seed=0, n=600):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n, tz="UTC").normalize()
    return pd.Series(rng.normal(0.0004, 0.011, n), index=idx)


def test_trend_weights_causal():
    px = _prices(1)
    base = trend_sleeve_weights(px, sma_window=50, step=21)
    t = 300
    p2 = px.copy()
    # Sustained halving of SPY from t onward: leaves prices <= t-1 untouched (so
    # causality must hold for weights <= t), but flips SPY below its SMA at later
    # rebalances (so the binary signal demonstrably depends on the input forward).
    p2.iloc[t:, 0] = p2.iloc[t:, 0] * 0.5
    pert = trend_sleeve_weights(p2, sma_window=50, step=21)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], pert.iloc[: t + 1])  # no lookahead
    assert not base.iloc[t + 1:].equals(pert.iloc[t + 1:])  # forward dependence


def test_trend_weights_long_or_flat_and_cash():
    px = _prices(2)
    w = trend_sleeve_weights(px, sma_window=50, step=21)
    assert (w.values >= -1e-12).all()                 # long-only
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()         # cash allowed (rows sum <= 1)
    # SPY trends up (held more), TLT trends down (flat/cash more) -> SPY heavier on avg
    tail = w.iloc[200:]
    assert tail["SPY"].mean() > tail["TLT"].mean()


def test_combiner_weights_sum_to_one_and_combined_is_weighted_sum():
    r = pd.DataFrame({"harv": _ret(0), "trend": _ret(1)})
    combined, wp = combine_sleeves(r, lookback=100, step=21)
    assert np.allclose(wp.sum(axis=1).values, 1.0, atol=1e-9)   # fully allocated across sleeves
    assert (wp.values >= -1e-12).all()
    assert np.allclose(combined.values, (wp * r).sum(axis=1).values, atol=1e-12)


def test_combiner_is_causal():
    r = pd.DataFrame({"harv": _ret(0), "trend": _ret(1)})
    _, wp = combine_sleeves(r, lookback=100, step=21)
    t = 200
    r2 = r.copy()
    r2.iloc[t, 0] = r2.iloc[t, 0] + 0.05  # perturb a sleeve return at t
    _, wp2 = combine_sleeves(r2, lookback=100, step=21)
    pd.testing.assert_frame_equal(wp.iloc[: t + 1], wp2.iloc[: t + 1])


def test_gate_combined_matches_stats():
    r = _ret(5, 800)
    g = gate_combined(r)
    s = stats(r.dropna())
    assert abs(g["net_sharpe"] - float(s["net_sharpe"])) < 1e-9
    assert abs(g["max_dd"] - float(s["max_dd"])) < 1e-9
    assert "gate_pass" in g
