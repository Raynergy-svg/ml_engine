"""No-mock tests for the Kronos-vs-Axiom shootout ledger math.

Covers the pure-logic core of scripts/experiment_kronos_vs_axiom_shootout.py:
calendar selection, outcome indexing, trade segmentation, cost application,
deadband, matched-week comparison, and the Axiom arm's causality. Kronos
inference itself needs huggingface.co weights — exercised only via the script's
full mode, never faked here (no-mock policy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import experiment_kronos_vs_axiom_shootout as ex  # noqa: E402


def _panel(n: int = 700, start: str = "2016-01-01", trend: float = 0.0005) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    close = 1.0 * np.cumprod(1.0 + np.full(n, trend))
    df = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.001, "low": close * 0.998,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)
    df.index.name = "date"
    return df


def test_decision_dates_picks_first_trading_day_per_iso_week():
    df = _panel(60)
    picks = ex.decision_dates(df.index, "2016-01-01", "2016-02-28")
    assert len(picks) >= 7
    seen = {p.isocalendar()[:2] for p in picks}
    assert len(seen) == len(picks)  # one pick per ISO week
    # every pick is the earliest bar of its week present in the index
    for p in picks:
        wk = [d for d in df.index if d.isocalendar()[:2] == p.isocalendar()[:2]]
        assert p == min(wk)


def test_week_outcome_uses_next_open_to_open_and_refuses_missing_forward():
    df = _panel(20)
    t = df.index[5]
    r = ex.week_outcome(df, t)
    expected = float(df["open"].iloc[11] / df["open"].iloc[6] - 1.0)
    assert abs(r - expected) < 1e-12
    assert ex.week_outcome(df, df.index[-3]) is None  # forward bars don't exist


def test_segment_trades_merges_same_direction_and_splits_on_flip():
    dates = pd.bdate_range("2020-01-06", periods=6, freq="W-MON", tz="UTC")
    weekly = [
        ex.WeekDecision("X", dates[0], 1, 0.01),
        ex.WeekDecision("X", dates[1], 1, 0.02),   # merges with week 0
        ex.WeekDecision("X", dates[2], -1, 0.01),  # flip -> new trade
        ex.WeekDecision("X", dates[3], 0, None),   # abstain closes the short
        ex.WeekDecision("X", dates[4], 1, -0.01),
    ]
    trades = ex.segment_trades(weekly)
    assert [t.direction for t in trades] == [1, -1, 1]
    assert trades[0].weeks == 2
    assert abs(trades[0].gross - (1.01 * 1.02 - 1.0)) < 1e-12
    # one round-trip cost per merged trade, not per week
    assert abs(trades[0].net(2.5) - (trades[0].gross - 2.5e-4)) < 1e-15


def test_arm_metrics_counts_wins_net_of_cost():
    dates = pd.bdate_range("2020-01-06", periods=3, freq="W-MON", tz="UTC")
    weekly = [
        ex.WeekDecision("X", dates[0], 1, 0.0002),   # gross win, net loss at 2.5bps
        ex.WeekDecision("X", dates[1], 0, None),
        ex.WeekDecision("X", dates[2], 1, 0.01),     # net win
    ]
    res = ex.ArmResult(weekly=weekly, trades=ex.segment_trades(weekly))
    m = ex.arm_metrics(res, 2.5)
    assert m["trades"] == 2
    assert m["winning_trades"] == 1          # 2bps gross does not survive 2.5bps cost
    assert m["weeks_taken"] == 2 and m["weeks_available"] == 3
    assert abs(m["weekly_hit_rate"] - 1.0) < 1e-12  # both taken weeks were gross-positive


def test_weekly_net_series_charges_cost_only_on_opening_week():
    dates = pd.bdate_range("2020-01-06", periods=3, freq="W-MON", tz="UTC")
    weekly = [
        ex.WeekDecision("X", dates[0], 1, 0.01),
        ex.WeekDecision("X", dates[1], 1, 0.01),   # held: no new cost
        ex.WeekDecision("X", dates[2], -1, 0.01),  # flip: cost again
    ]
    s = ex.weekly_net_series(weekly, 2.5)
    c = 2.5e-4
    assert abs(s.iloc[0] - (0.01 - c)) < 1e-12
    assert abs(s.iloc[1] - 0.01) < 1e-12
    assert abs(s.iloc[2] - (0.01 - c)) < 1e-12


def test_kronos_deadband_long_short_abstain():
    assert ex.kronos_direction_from_pred(1.0010, 1.0) == 1     # +10bps > 2.5bps
    assert ex.kronos_direction_from_pred(0.9990, 1.0) == -1
    assert ex.kronos_direction_from_pred(1.0001, 1.0) == 0     # inside deadband
    assert ex.kronos_direction_from_pred(float("nan"), 1.0) == 0
    assert ex.kronos_direction_from_pred(1.01, 0.0) == 0


def test_matched_week_test_only_uses_weeks_both_arms_took():
    dates = pd.bdate_range("2020-01-06", periods=3, freq="W-MON", tz="UTC")
    ax = [ex.WeekDecision("X", dates[0], 1, 0.01),
          ex.WeekDecision("X", dates[1], 0, None),
          ex.WeekDecision("X", dates[2], 1, -0.01)]
    kr = [ex.WeekDecision("X", dates[0], -1, -0.01),
          ex.WeekDecision("X", dates[1], 1, 0.02),
          ex.WeekDecision("X", dates[2], 1, -0.01)]
    m = ex.matched_week_test(ax, kr)
    assert m["n_matched_weeks"] == 2  # dates[1] excluded: axiom abstained
    assert abs(m["axiom_hit"] - 0.5) < 1e-12
    assert abs(m["kronos_hit"] - 0.0) < 1e-12


def test_axiom_directions_are_causal_and_long_or_flat():
    # steady uptrend: long once SMA warmup passes; a huge SAME-DAY spike at the
    # decision bar must not affect that day's decision (uses <= t-1 only).
    df = _panel(400, trend=0.001)
    dates = ex.decision_dates(df.index, "2016-09-01", "2017-06-30")
    dirs = ex.axiom_directions(df, dates)
    assert dirs and all(d in (0, 1) for d in dirs.values())
    assert any(d == 1 for d in dirs.values())
    t = dates[len(dates) // 2]
    spiked = df.copy()
    spiked.loc[t, "close"] = float(spiked.loc[t, "close"]) * 0.5  # crash AT t
    assert ex.axiom_directions(spiked, [t])[t] == ex.axiom_directions(df, [t])[t]


def test_evaluate_window_axiom_only_end_to_end(tmp_path):
    df = _panel(700, trend=0.0008)
    out = ex.evaluate_window({"SYN_PAIR": df}, ("2017-06-01", "2018-06-01"), None)
    prim = out["cost_scenarios"][str(ex.COST_PRIMARY_BPS)]
    assert prim["kronos"] is None
    a = prim["axiom"]
    assert a["trades"] >= 1
    assert a["weeks_available"] > 40
    # uptrending synthetic panel: the trend arm should be mostly long and winning
    assert a["participation"] > 0.5
    assert a["winning_trades"] >= 1
