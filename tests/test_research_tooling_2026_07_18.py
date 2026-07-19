"""Research tooling from readiness steps 3 + 5 (runtime-exact OANDA backtest
arm + consolidated strategy metrics). No mocks: synthetic pandas panels, real
tmp_path ledgers, real repo artifacts read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_oanda_trend_runtime import COST_BPS_GRID, backtest_arm  # noqa: E402
from research_metrics import _metrics, _own_ledger_series, build_report  # noqa: E402
from src.hedge.hedged_shadow_lane import STRATEGIES  # noqa: E402


def _panel(n=320, seed=11):
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    up = 1.0 * np.exp(np.cumsum(rng.normal(0.0015, 0.004, n)))
    down = 2.0 * np.exp(np.cumsum(rng.normal(-0.0015, 0.004, n)))
    return pd.DataFrame({"UPP_USD": up, "DWN_USD": down}, index=idx)


# ── step 3: runtime-exact backtest arm ──────────────────────────────────────

def test_backtest_arm_schema_costs_and_direction():
    arm = backtest_arm(_panel(), sma_window=100, gross_leverage=3.0)
    assert arm["sma_window"] == 100 and arm["n_days"] > 0
    assert set(arm["cost_arms"]) == {f"{c:g}bps" for c in COST_BPS_GRID}
    sharpes = [arm["cost_arms"][f"{c:g}bps"]["net_sharpe"] for c in COST_BPS_GRID]
    # cost monotonicity: more cost never improves the result
    assert sharpes[0] >= sharpes[1] >= sharpes[2]
    # long-or-flat on an uptrending name in a 2-name world -> positive gross return
    assert arm["cost_arms"]["0bps"]["ann_return_unlevered"] > 0
    # leverage scales return linearly; Sharpe is reported leverage-invariant
    a0 = arm["cost_arms"]["0bps"]
    assert a0["ann_return_levered"] == round(a0["ann_return_unlevered"] * 3.0, 5) or \
        abs(a0["ann_return_levered"] - a0["ann_return_unlevered"] * 3.0) < 1e-4


def test_backtest_arm_is_causal_next_bar_application():
    # 2026-07-19 audit: assert causality on the RETURN SERIES itself, not a
    # weak aggregate. Perturbing ONLY the final bar's prices must leave every
    # PRIOR bar's daily return bit-for-bit identical (targets apply next bar,
    # no same-bar execution, no forward leakage into history).
    p1 = _panel()
    p2 = p1.copy()
    p2.iloc[-1] = p2.iloc[-1] * 1.10
    _a1, s1 = backtest_arm(p1, sma_window=100, gross_leverage=1.0, with_series=True)
    _a2, s2 = backtest_arm(p2, sma_window=100, gross_leverage=1.0, with_series=True)
    pd.testing.assert_series_equal(s1.iloc[:-1], s2.iloc[:-1], check_exact=True)
    # ...and the perturbed final bar's own return DOES differ (the position
    # entering that bar was held; a 10% price shock must show up there)
    assert s1.iloc[-1] != s2.iloc[-1]


# ── step 5: consolidated metrics ────────────────────────────────────────────

def _dates(n, start="2026-01-05", freq="7D"):
    return [str(d.date()) for d in pd.date_range(start, periods=n, freq=freq)]


def test_metrics_honest_nulls_and_significance_gate():
    empty = _metrics([], [], n_trials=15)
    assert empty["n_observations"] == 0 and empty["sharpe"] is None
    assert "no forward observations" in empty["unavailable_reason"]

    # ONE observation must never be annualized (audit: one +1.024% bar was
    # being reported as a 373% annual return)
    one = _metrics([0.01024], _dates(1), n_trials=15)
    assert one["cumulative_return"] == pytest.approx(0.01024)
    assert one["ann_return"] is None and one["ann_vol"] is None and one["sharpe"] is None
    assert "mechanical extrapolation" in one["unavailable_reason"]

    rng = np.random.default_rng(5)
    good = _metrics(list(rng.normal(0.004, 0.002, 60)), _dates(60), n_trials=15)
    assert good["sharpe"] is not None and good["sharpe"] > 0
    # weekly dates -> derived factor ~52, never a hardcoded per-strategy map
    assert good["ann_factor_derived"] == pytest.approx(365.25 / 7, rel=1e-3)
    assert good["deflated_sr_prob"] is not None and good["dsr_n_trials"] == 15
    assert 0.0 <= good["block_bootstrap_p"] <= 1.0


def test_metrics_dsr_requires_a_registered_trial_count():
    # audit: deflated_sr(n_trials=1) degenerates its benchmark to
    # norm.ppf(0) = -inf, making anything look maximally significant.
    rng = np.random.default_rng(7)
    rets = list(rng.normal(0.004, 0.002, 60))
    no_trials = _metrics(rets, _dates(60), n_trials=None)
    assert no_trials["deflated_sr_prob"] is None
    assert "no registered multiple-testing trial count" in no_trials["significance_reason"]
    assert no_trials["block_bootstrap_p"] is not None  # bootstrap needs no trial count
    one_trial = _metrics(rets, _dates(60), n_trials=1)
    assert one_trial["deflated_sr_prob"] is None       # n_trials=1 is never computed


def test_metrics_drawdown_measures_from_initial_capital():
    # audit: a FIRST observation of -10% is a 10% drawdown — the peak starts
    # at 1.0, never at the first post-return equity value.
    m = _metrics([-0.10], _dates(1), n_trials=None)
    assert m["max_drawdown"] == pytest.approx(0.10)
    m2 = _metrics([-0.10, 0.05], _dates(2), n_trials=None)
    assert m2["max_drawdown"] == pytest.approx(0.10)


def test_metrics_no_annualization_below_coverage_floor():
    from research_metrics import MIN_OBS_FOR_ANNUALIZATION
    rets = [0.004] * (MIN_OBS_FOR_ANNUALIZATION - 1)
    m = _metrics(rets, _dates(len(rets)), n_trials=None)
    assert m["ann_return"] is None and m["sharpe"] is None
    rets2 = list(np.random.default_rng(3).normal(0.004, 0.002, MIN_OBS_FOR_ANNUALIZATION))
    m2 = _metrics(rets2, _dates(len(rets2)), n_trials=None)
    assert m2["ann_return"] is not None


def test_own_ledger_series_reads_real_rows(tmp_path):
    ledger = tmp_path / "lane.jsonl"
    rows = [
        {"kind": "activation", "asof_date": "2026-06-24",
         "today_net_return": None},                       # baseline: never counted
        {"asof_date": "2026-07-01", "today_net_return": 0.002,
         "today_turnover": 0.5, "today_cost": 0.0005},
        {"asof_date": "2026-07-01", "today_net_return": 0.002},  # dup asof: excluded
        {"asof_date": "2026-07-08", "today_net_return": -0.001},
        {"asof_date": "2026-07-15"},   # unresolved cycle: skipped, never zero-filled
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    series = _own_ledger_series(ledger)
    assert series["returns"] == [0.002, -0.001]
    assert series["dates"] == ["2026-07-01", "2026-07-08"]
    assert series["turnover"][0] == 0.5 and series["turnover"][1] is None
    assert series["cadence"]["n_return_bars"] == 3  # cadence counts raw ledger rows pre-dedupe


def test_build_report_covers_all_registered_strategies_uniform_schema():
    report = build_report()
    assert set(report["strategies"]) == set(STRATEGIES)
    for s, m in report["strategies"].items():
        for key in report["schema"]:
            assert key in m or key in ("residual_fraction",) and "residual_fraction" in m, (s, key)
        assert "residual_fraction" in m and "phi" in m["residual_fraction"]
        assert m["beta_reason"] and m["benchmark_reason"]
    assert report["paper_only"] is True and report["runtime_allowed"] is False
