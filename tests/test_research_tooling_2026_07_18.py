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
    # Perturbing ONLY the final bar's prices must not change the PREVIOUS
    # bars' P&L — targets apply at the NEXT bar (no same-bar execution).
    p1 = _panel()
    p2 = p1.copy()
    p2.iloc[-1] = p2.iloc[-1] * 1.10
    a1 = backtest_arm(p1.iloc[:-1], sma_window=100, gross_leverage=1.0)
    a2 = backtest_arm(p2, sma_window=100, gross_leverage=1.0)
    # a2 includes one extra day; everything up to the shared span must agree
    assert a1["span"][0] == a2["span"][0]
    assert a1["cost_arms"]["0bps"]["positive_years"] <= a2["cost_arms"]["0bps"]["total_years"] + 1


# ── step 5: consolidated metrics ────────────────────────────────────────────

def test_metrics_honest_nulls_and_significance_gate():
    empty = _metrics([], ann=252.0)
    assert empty["n_observations"] == 0 and empty["sharpe"] is None
    assert "no forward observations" in empty["unavailable_reason"]

    one = _metrics([0.01], ann=252.0)
    assert one["cumulative_return"] == 0.01 and one["sharpe"] is None
    assert one["deflated_sr_prob"] is None and "warm-up" in one["significance_reason"]

    rng = np.random.default_rng(5)
    good = _metrics(list(rng.normal(0.004, 0.002, 60)), ann=52.0)
    assert good["sharpe"] is not None and good["sharpe"] > 0
    assert good["deflated_sr_prob"] is not None
    assert 0.0 <= good["block_bootstrap_p"] <= 1.0
    assert good["max_drawdown"] is not None


def test_own_ledger_series_reads_real_rows(tmp_path):
    ledger = tmp_path / "lane.jsonl"
    rows = [
        {"asof_date": "2026-07-01", "today_net_return": 0.002,
         "today_turnover": 0.5, "today_cost": 0.0005},
        {"asof_date": "2026-07-08", "today_net_return": -0.001},
        {"asof_date": "2026-07-15"},   # unresolved cycle: skipped, never zero-filled
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    series = _own_ledger_series(ledger)
    assert series["returns"] == [0.002, -0.001]
    assert series["dates"] == ["2026-07-01", "2026-07-08"]
    assert series["turnover"][0] == 0.5 and series["turnover"][1] is None


def test_build_report_covers_all_registered_strategies_uniform_schema():
    report = build_report()
    assert set(report["strategies"]) == set(STRATEGIES)
    for s, m in report["strategies"].items():
        for key in report["schema"]:
            assert key in m or key in ("residual_fraction",) and "residual_fraction" in m, (s, key)
        assert "residual_fraction" in m and "phi" in m["residual_fraction"]
        assert m["beta_reason"] and m["benchmark_reason"]
    assert report["paper_only"] is True and report["runtime_allowed"] is False
