"""Widened-universe PIT quality evaluator — staggered-universe NaN-safety (offline).

The engine fail-loud rejects NaN returns (correct). A wide survivorship-aware
universe has late-IPO (leading NaN) and delisted (trailing NaN) names. This proves
evaluate_book_wide tolerates that WITHOUT fabricating exposure: weight is forced to
0 wherever the original price was NaN, so filled cells are inert. No mocks — real
backtest/overlay/gate primitives + a real synthetic price panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equity.pit_quality_eval import _daily_membership_snapshot, evaluate_book_wide
from src.equity.sp500_membership import build_sp500_snapshot
from src.equity.universe import UniverseRules, UniverseSnapshot, compute_universe_hash


def _staggered_prices(n=520):
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=n, tz="UTC")).normalize()
    rng = np.random.default_rng(7)
    df = pd.DataFrame(index=idx)
    df["FULL"] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    late = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    late[: n // 3] = np.nan                      # late IPO: leading NaN
    df["LATEIPO"] = late
    deli = 100 * np.cumprod(1 + rng.normal(0.0003, 0.011, n))
    deli[2 * n // 3:] = np.nan                    # delisted: trailing NaN
    df["DELISTED"] = deli
    return df


def _snap(prices):
    members = list(prices.columns)
    membership = pd.DataFrame(prices.notna().values, index=prices.index, columns=members)
    return UniverseSnapshot(
        membership=membership, rules=UniverseRules(top_n=len(members)),
        built_at="2026-06-25T00:00:00+00:00", pipeline_version="test",
        universe_hash=compute_universe_hash(membership),
    )


def _ew_weights(snapshot, prices):
    active = prices.notna()
    n = active.sum(axis=1).replace(0, np.nan)
    return active.astype(float).div(n, axis=0).fillna(0.0)


def test_evaluate_book_wide_survives_staggered_universe():
    px = _staggered_prices()
    snap = _snap(px)
    res = evaluate_book_wide(snap, px, _ew_weights(snap, px))
    # Runs to a real gate verdict without raising on NaN; sane stats.
    assert "net_sharpe" in res and "gate_pass" in res
    assert -5.0 < res["net_sharpe"] < 5.0
    assert 0.0 <= res["max_dd"] <= 1.0


def test_unpriced_name_is_inert_and_no_crash():
    # An all-NaN (never-priced-in-window) name must change nothing and must NOT crash
    # the engine — it is dropped as 0-weight, never fabricated into exposure.
    px = _staggered_prices()
    snap = _snap(px)
    base = evaluate_book_wide(snap, px, _ew_weights(snap, px))
    px2 = px.copy()
    px2["GHOST"] = np.nan  # never priced in the window
    snap2 = _snap(px2)
    after = evaluate_book_wide(snap2, px2, _ew_weights(snap2, px2))
    assert abs(after["net_sharpe"] - base["net_sharpe"]) < 1e-9  # inert: no leak, no crash


def test_daily_membership_snapshot_ffills_monthly():
    snap_m = build_sp500_snapshot(start="2015-01-01", end="2015-04-01",
                                  current=["A", "B"], changes=[])
    # monthly membership -> reindex onto a daily grid via ffill
    daily_idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", "2015-03-25", tz="UTC")).normalize()
    snap_d = _daily_membership_snapshot(snap_m, daily_idx)
    assert snap_d.membership.index.equals(daily_idx)
    assert snap_d.membership.values.all()  # A,B members throughout (no changes)
