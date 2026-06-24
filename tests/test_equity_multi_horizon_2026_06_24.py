"""Multi-horizon harvester variant — pure-function tests (offline; running:NO).

Proves the variant is (1) a strict generalization of the baseline overlay
(horizons=(21,) reproduces `backtest.overlay` exactly), (2) STRICTLY CAUSAL —
perturbing returns at t cannot change the exposure scalar at any index <= t (the
#1 way these backtests lie), (3) a genuinely different signal at multiple
horizons, and (4) that the eval harness reproduces `ship_gate.evaluate_harvester`
on its baseline arm — so the side-by-side gate comparison is apples-to-apples.

No mocks: real synthetic panels + real `ship_gate`/`backtest` functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equity.backtest import overlay as baseline_overlay
from src.equity.multi_horizon import evaluate_multi_horizon, overlay_multi_horizon
from src.equity.ship_gate import evaluate_harvester
from src.equity.universe import UniverseRules, UniverseSnapshot

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _ret(seed=0, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n, tz="UTC")
    return pd.Series(rng.normal(0.0004, 0.011, n), index=idx)


def test_single_horizon_equals_baseline_overlay():
    r = _ret()
    a = overlay_multi_horizon(
        r, target_vol=0.12, dd_soft=0.10, dd_hard=0.20, max_lev=1.0, horizons=(21,)
    )
    b = baseline_overlay(r, target_vol=0.12, dd_soft=0.10, dd_hard=0.20, max_lev=1.0)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_overlay_strictly_causal_no_lookahead():
    r = _ret(1, 300)
    base = overlay_multi_horizon(
        r, target_vol=0.12, dd_soft=0.10, dd_hard=0.20, horizons=(21, 63)
    )
    t = 200
    r2 = r.copy()
    r2.iloc[t] = r2.iloc[t] + 0.5  # huge spike at t
    perturbed = overlay_multi_horizon(
        r2, target_vol=0.12, dd_soft=0.10, dd_hard=0.20, horizons=(21, 63)
    )
    # scalar at every index <= t must be UNCHANGED (depends only on returns <= t-1)
    pd.testing.assert_series_equal(
        base.iloc[: t + 1], perturbed.iloc[: t + 1], check_names=False
    )
    # and the perturbation DOES propagate forward (sanity: not a constant series)
    assert not base.iloc[t + 1:].equals(perturbed.iloc[t + 1:])


def test_multi_horizon_is_genuinely_different_signal():
    r = _ret(2, 500)
    single = overlay_multi_horizon(
        r, target_vol=0.12, dd_soft=0.10, dd_hard=0.20, horizons=(21,)
    )
    multi = overlay_multi_horizon(
        r, target_vol=0.12, dd_soft=0.10, dd_hard=0.20, horizons=(21, 63, 126, 252)
    )
    assert not np.allclose(single.values, multi.values)


def _panels(seed=7, n=600):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n, tz="UTC").normalize()
    membership = pd.DataFrame(
        np.ones((n, len(TICKERS)), dtype=bool), index=dates, columns=TICKERS
    )
    rets = rng.normal(0.0004, 0.011, size=(n, len(TICKERS)))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=TICKERS
    )
    return membership, prices


def _snap(membership):
    return UniverseSnapshot(
        membership=membership,
        rules=UniverseRules(),
        built_at="2026-06-19T00:00:00Z",
        pipeline_version="2026-06-18-eq1",
        universe_hash="abcd" * 16,
    )


def test_eval_baseline_arm_reproduces_ship_gate():
    membership, prices = _panels()
    snap = _snap(membership)
    expected = evaluate_harvester(snap, prices, adv=None)
    got = evaluate_multi_horizon(snap, prices, horizons=(21,), adv=None)
    assert abs(got["net_sharpe"] - expected.net_sharpe) < 1e-9
    assert abs(got["max_dd"] - expected.max_dd) < 1e-9
    assert got["positive_years"] == expected.positive_years
    assert got["total_years"] == expected.total_years
    assert got["gate_pass"] == expected.gate_pass
