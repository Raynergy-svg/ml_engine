"""Research tooling from readiness steps 3 + 5 (SIGNAL-ONLY OANDA target-rule
backtest + consolidated strategy metrics). No mocks: synthetic pandas panels,
real tmp_path ledgers, real repo artifacts read-only.

2026-07-19 re-review regression coverage lives here for items 3 (evidence-
gated metrics), 4 (annualization correctness), 5 (cost/turnover alignment),
6 (initial-capital drawdown), 7 (signal-only framing) and 8 (causality seam).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_oanda_trend_runtime import (  # noqa: E402
    ANN, COST_BPS_GRID, _max_drawdown, backtest_arm)
from research_metrics import (  # noqa: E402
    ACCEPTANCE_REQUIREMENTS, _derive_ann_factor, _metrics, _own_ledger_series,
    _validated_span, build_report, evidence_eligibility)
from src.equity.oanda_trend import trend_targets  # noqa: E402
from src.equity.trend_sleeve import trend_sleeve_weights  # noqa: E402
from src.hedge.hedged_shadow_lane import STRATEGIES  # noqa: E402

ELIGIBLE = {"eligible": True, "reason_code": None, "detail": "test gate"}
INELIGIBLE = {"eligible": False, "reason_code": "insufficient_independent_periods",
              "detail": "test gate"}


def _panel(n=320, seed=11):
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    up = 1.0 * np.exp(np.cumsum(rng.normal(0.0015, 0.004, n)))
    down = 2.0 * np.exp(np.cumsum(rng.normal(-0.0015, 0.004, n)))
    return pd.DataFrame({"UPP_USD": up, "DWN_USD": down}, index=idx)


# ── step 3: signal-only backtest arm ────────────────────────────────────────

def test_backtest_arm_schema_costs_and_direction():
    arm = backtest_arm(_panel(), sma_window=100, gross_leverage=3.0)
    assert arm["sma_window"] == 100 and arm["n_days"] > 0
    assert set(arm["cost_arms"]) == {f"{c:g}bps" for c in COST_BPS_GRID}
    sharpes = [arm["cost_arms"][f"{c:g}bps"]["net_sharpe"] for c in COST_BPS_GRID]
    # cost monotonicity: more cost never improves the result
    assert sharpes[0] >= sharpes[1] >= sharpes[2]
    # long-or-flat on an uptrending name in a 2-name world -> positive gross return
    assert arm["cost_arms"]["0bps"]["ann_return_unlevered"] > 0
    a0 = arm["cost_arms"]["0bps"]
    assert a0["ann_return_levered"] == round(a0["ann_return_unlevered"] * 3.0, 5) or \
        abs(a0["ann_return_levered"] - a0["ann_return_unlevered"] * 3.0) < 1e-4


def test_backtest_arm_is_causal_next_bar_application():
    # Perturbing ONLY the final bar's prices must leave every PRIOR bar's
    # daily return bit-for-bit identical (targets apply next bar, no
    # same-bar execution, no forward leakage into history).
    p1 = _panel()
    p2 = p1.copy()
    p2.iloc[-1] = p2.iloc[-1] * 1.10
    _a1, s1 = backtest_arm(p1, sma_window=100, gross_leverage=1.0, with_series=True)
    _a2, s2 = backtest_arm(p2, sma_window=100, gross_leverage=1.0, with_series=True)
    pd.testing.assert_series_equal(s1.iloc[:-1], s2.iloc[:-1], check_exact=True)
    assert s1.iloc[-1] != s2.iloc[-1]


def test_causality_holds_at_multiple_perturbation_points():
    # 2026-07-19 re-review item 8: a single perturbed FINAL bar is a weak
    # seam. Perturb bar t at several interior points: every return strictly
    # BEFORE bar t must stay bit-identical (changing bar t cannot alter any
    # target or return before its earliest allowed execution period).
    p1 = _panel()
    _a1, s1 = backtest_arm(p1, sma_window=100, gross_leverage=1.0,
                           with_series=True)
    for k in (len(p1) - 1, len(p1) - 5, len(p1) - 20, len(p1) - 60):
        p2 = p1.copy()
        p2.iloc[k] = p2.iloc[k] * 1.10
        _a2, s2 = backtest_arm(p2, sma_window=100, gross_leverage=1.0,
                               with_series=True)
        bar_t = p1.index[k]
        # everything strictly BEFORE the perturbed bar: bit-identical
        pd.testing.assert_series_equal(
            s1.loc[:bar_t].iloc[:-1], s2.loc[:bar_t].iloc[:-1],
            check_exact=True)


def test_backtest_targets_match_runtime_trend_targets_on_rolling_prefixes():
    # item 8: on EVERY rolling prefix, the backtest's target matrix row must
    # equal what the runtime's own ``oanda_trend.trend_targets`` would emit
    # given exactly that history — the backtest tests the runtime's rule,
    # not a lookalike.
    p = _panel()
    full = trend_sleeve_weights(p.ffill(), sma_window=100, step=1)
    for L in (150, 200, 260, 320):
        runtime_now = trend_targets(p.iloc[:L], sma_window=100)
        backtest_row = {c: float(v) for c, v in full.iloc[L - 1].items()}
        assert runtime_now == backtest_row, f"prefix {L} diverges"


def test_turnover_and_costs_align_to_applied_weights():
    # item 5: costs are charged on the turnover of the APPLIED weights (the
    # series that earns returns), never on the unshifted decision targets.
    p = _panel()
    arm = backtest_arm(p, sma_window=100, gross_leverage=1.0)
    targets = trend_sleeve_weights(p.ffill(), sma_window=100, step=1)
    applied = targets.shift(1).fillna(0.0)
    turn = applied.diff().abs().sum(axis=1).fillna(0.0)
    start = p.index[min(len(p) - 1, 101)]
    assert arm["avg_annual_turnover"] == pytest.approx(
        float(turn.loc[start:].mean() * ANN))


def test_backtest_drawdown_measures_from_initial_capital():
    # item 6: peak starts at 1.0 — a first-bar loss IS a drawdown.
    assert _max_drawdown([-0.10]) == pytest.approx(0.10)
    assert _max_drawdown([-0.10, 0.05]) == pytest.approx(0.10)
    # end-to-end: the reported 0-cost unlevered maxDD equals the compounded
    # drawdown of the returned series itself.
    arm, base = backtest_arm(_panel(), sma_window=100, gross_leverage=1.0,
                             with_series=True)
    assert arm["cost_arms"]["0bps"]["max_dd_unlevered"] == pytest.approx(
        round(_max_drawdown(base.tolist()), 5))


def test_signal_only_framing_in_script_and_artifact_fields():
    # item 7: the script must not present itself as an exact live-rule /
    # signal+sizing runtime test.
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "backtest_oanda_trend_runtime.py").read_text()
    for banned in ("SIGNAL+SIZING", "Signal + sizing", "runtime-exact",
                   "exact live rule"):
        assert banned not in src, f"stale framing {banned!r} back in script"
    assert "SIGNAL-ONLY" in src


# ── step 5: consolidated metrics ────────────────────────────────────────────

def _dates(n, start="2026-01-05", freq="7D"):
    return [str(d.date()) for d in pd.date_range(start, periods=n, freq=freq)]


def test_metrics_honest_nulls_and_gating():
    empty = _metrics([], [], n_trials=15, gate=dict(INELIGIBLE))
    assert empty["n_observations"] == 0 and empty["sharpe"] is None

    # ONE observation must never be annualized
    one = _metrics([0.01024], _dates(1), n_trials=15, gate=dict(INELIGIBLE))
    assert one["cumulative_return"] == pytest.approx(0.01024)
    assert one["ann_return_geometric"] is None and one["sharpe"] is None

    # 2026-07-19 re-review item 3: even 60 observations do NOT unlock
    # annualization/DSR/bootstrap while the acceptance gate is ineligible.
    rng = np.random.default_rng(5)
    rets = list(rng.normal(0.004, 0.002, 60))
    gated = _metrics(rets, _dates(60), n_trials=15, gate=dict(INELIGIBLE))
    assert gated["sharpe"] is None and gated["ann_return_geometric"] is None
    assert gated["deflated_sr_prob"] is None and gated["block_bootstrap_p"] is None
    assert "insufficient_independent_periods" in gated["significance_reason"]

    good = _metrics(rets, _dates(60), n_trials=15, gate=dict(ELIGIBLE))
    assert good["sharpe"] is not None and good["sharpe"] > 0
    # weekly dates -> observations-per-elapsed-year ~52
    assert good["ann_factor_derived"] == pytest.approx(365.25 / 7, rel=1e-3)
    assert good["deflated_sr_prob"] is not None and good["dsr_n_trials"] == 15
    assert 0.0 <= good["block_bootstrap_p"] <= 1.0
    # item 4: BOTH annualized returns present under explicit names, and the
    # geometric one compounds (differs from the arithmetic one in general)
    assert good["ann_return_geometric"] is not None
    assert good["ann_return_arithmetic"] is not None
    assert good["ann_return_geometric"] != good["ann_return_arithmetic"]


def test_ann_factor_maps_business_days_to_business_year():
    # item 4: the median-gap version mapped business-day data to ~365/yr.
    # Observations-per-elapsed-year maps weekday-sampled data to the weekday
    # calendar (~261/yr — pandas "B" has no holiday calendar), NEVER ~365.
    bdays = [str(d.date()) for d in pd.date_range("2024-01-02", periods=253,
                                                  freq="B")]
    f = _derive_ann_factor(bdays)
    assert f is not None and 255 <= f <= 265, f


def test_ann_factor_refuses_invalid_dates_with_reason():
    assert _validated_span(["2026-01-05", "2026-01-04"]) == (None, "invalid_dates")
    assert _validated_span(["2026-01-05", "2026-01-05"]) == (None, "invalid_dates")
    assert _validated_span(["not-a-date", "2026-01-05"]) == (None, "invalid_dates")
    assert _validated_span(["2026-01-05"])[1] == "insufficient_observations"
    assert _derive_ann_factor(["2026-01-05", "2026-01-04"]) is None


def test_metrics_zero_variance_reason_code():
    m = _metrics([0.004] * 60, _dates(60), n_trials=15, gate=dict(ELIGIBLE))
    assert m["sharpe"] is None
    assert m["metrics_gate"]["reason_code"] == "zero_variance"


def test_eligibility_raw_daily_bars_never_annualize():
    # item 3: 8 raw daily bars (no explicit cadence fields) must NOT unlock
    # annualization for a weekly/monthly strategy.
    dates = [str(d.date()) for d in pd.date_range("2026-07-01", periods=8,
                                                  freq="D")]
    cadence = {"n_return_bars": 8, "n_completed_rebalances": None,
               "n_independent_holding_periods": None}
    g = evidence_eligibility("crypto_ts_trend", cadence, dates)
    assert g["eligible"] is False and g["reason_code"] == "cadence_unavailable"


def test_eligibility_meets_frozen_acceptance_targets():
    req = ACCEPTANCE_REQUIREMENTS["crypto_ts_trend"]
    assert req["min_independent_periods"] == 52 and req["min_elapsed_days"] == 365
    dates_53w = _dates(53)                       # 52 weekly gaps = 364d elapsed
    cad = {"n_completed_rebalances": 52, "n_independent_holding_periods": 52}
    g = evidence_eligibility("crypto_ts_trend", cad, dates_53w)
    assert g["eligible"] is False and g["reason_code"] == "insufficient_elapsed_span"
    dates_54w = _dates(54)                       # 371d elapsed, 52+ periods
    g2 = evidence_eligibility(
        "crypto_ts_trend",
        {"n_completed_rebalances": 53, "n_independent_holding_periods": 53},
        dates_54w)
    assert g2["eligible"] is True
    g3 = evidence_eligibility(
        "crypto_ts_trend",
        {"n_completed_rebalances": 51, "n_independent_holding_periods": 51},
        dates_54w)
    assert g3["reason_code"] == "insufficient_independent_periods"
    # multi-asset gates on completed MONTHLY rebalances
    monthly = _dates(30, freq="30D")
    g4 = evidence_eligibility(
        "multi_asset_trend",
        {"n_completed_rebalances": 23, "n_independent_holding_periods": 23},
        monthly)
    assert g4["reason_code"] == "insufficient_independent_periods"
    g5 = evidence_eligibility("no_such_strategy", None, _dates(5))
    assert g5["reason_code"] in ("no_registered_acceptance_target",
                                 "cadence_unavailable")


def test_metrics_dsr_requires_a_registered_trial_count():
    rng = np.random.default_rng(7)
    rets = list(rng.normal(0.004, 0.002, 60))
    no_trials = _metrics(rets, _dates(60), n_trials=None, gate=dict(ELIGIBLE))
    assert no_trials["deflated_sr_prob"] is None
    assert "no registered multiple-testing trial count" in no_trials["significance_reason"]
    one_trial = _metrics(rets, _dates(60), n_trials=1, gate=dict(ELIGIBLE))
    assert one_trial["deflated_sr_prob"] is None   # n_trials=1 never computed


def test_metrics_drawdown_measures_from_initial_capital():
    m = _metrics([-0.10], _dates(1), n_trials=None, gate=dict(INELIGIBLE))
    assert m["max_drawdown"] == pytest.approx(0.10)
    m2 = _metrics([-0.10, 0.05], _dates(2), n_trials=None, gate=dict(INELIGIBLE))
    assert m2["max_drawdown"] == pytest.approx(0.10)


def test_own_ledger_series_reads_engine_rows_only(tmp_path):
    # item 1+2: legacy pre-engine rows (no ``kind``) are EXCLUDED and
    # reported; engine forward rows form the one canonical view.
    ledger = tmp_path / "lane.jsonl"
    rows = [
        {"asof_date": "2026-06-17", "today_net_return": 0.0102},  # legacy
        {"kind": "activation", "asof_date": "2026-06-24",
         "today_net_return": None},
        {"kind": "forward", "asof_date": "2026-07-01",
         "evidence_period_id": "s:2026-06-24->2026-07-01",
         "snapshot_version_id": "v1", "today_net_return": 0.005},
        # explicit supersession: replaces, never counts beside
        {"kind": "forward", "asof_date": "2026-07-01",
         "evidence_period_id": "s:2026-06-24->2026-07-01",
         "snapshot_version_id": "v2", "supersedes": "v1",
         "today_net_return": 0.002, "today_turnover": 0.5,
         "today_cost": 0.0005},
        {"kind": "forward", "asof_date": "2026-07-08",
         "evidence_period_id": "s:2026-07-01->2026-07-08",
         "snapshot_version_id": "v3", "today_net_return": -0.001},
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    series = _own_ledger_series(ledger)
    assert series["returns"] == [0.002, -0.001]
    assert series["dates"] == ["2026-07-01", "2026-07-08"]
    assert series["turnover"][0] == 0.5 and series["turnover"][1] is None
    assert series["legacy_rows_excluded"] == 1
    assert series["cadence"]["n_return_bars"] == 2
    assert series["cadence"]["n_legacy_rows_excluded"] == 1
    # rows lack explicit rebalance fields -> None, never inferred
    assert series["cadence"]["n_completed_rebalances"] is None


def test_build_report_covers_all_registered_strategies_uniform_schema():
    report = build_report()
    assert set(report["strategies"]) == set(STRATEGIES)
    for s, m in report["strategies"].items():
        for key in report["schema"]:
            assert key in m, (s, key)
        assert "residual_fraction" in m and "phi" in m["residual_fraction"]
        assert m["beta_reason"] and m["benchmark_reason"]
        assert "metrics_gate" in m and "reason_code" in m["metrics_gate"]
    assert report["paper_only"] is True and report["runtime_allowed"] is False
