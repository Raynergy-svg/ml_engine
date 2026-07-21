"""AXIOM Hedge Layer Phase 4 — raw-vs-hedged scorecard (SHADOW/ANALYSIS-ONLY).

Reproduce-then-resolve tests for the decision readout itself, built on
synthetic (but shape-correct) ledger rows -- exercises the pure-math scoring
layer independent of ``hedged_shadow_lane``'s price/exposure machinery
(that side is covered in ``test_hedged_shadow_lane_2026_07_08.py``).

NO MOCKS: every row is a real dict matching the real ledger schema; every
function under test operates on real Python data structures, no test
doubles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hedge import hedge_scorecard as hsc  # noqa: E402


def _row(asof: str, raw_net: float, hedged_net, hedged_gross, exposure_fail_closed: bool = False,
         net_beta: float = 0.0, status: str = "applied", concentration_warnings=None) -> dict:
    return {
        "strategy": "test_strategy",
        "asof_date": asof,
        "raw": {"net_return": raw_net},
        "hedged": {"net_return": hedged_net, "gross_return": hedged_gross},
        "hedge": {"status": status, "overlay_cost_known": hedged_net is not None},
        "exposure": {"net_beta_exposure": net_beta, "fail_closed": exposure_fail_closed,
                     "concentration_warnings": concentration_warnings or []},
    }


# --------------------------------------------------------------------- #
# Insufficient history — never force a verdict on a thin sample          #
# --------------------------------------------------------------------- #
def test_fewer_than_min_history_never_forces_a_verdict():
    rows = [_row("2026-01-01", 0.01, 0.005, 0.005), _row("2026-01-02", 0.01, 0.005, 0.005)]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["n_cycles"] == 2
    assert card["decision"]["verdict"] == hsc.VERDICT_INSUFFICIENT_HISTORY


def test_empty_history_does_not_crash():
    card = hsc.compute_strategy_scorecard([])
    assert card["n_cycles"] == 0
    assert card["decision"]["verdict"] == hsc.VERDICT_INSUFFICIENT_HISTORY
    assert card["raw"]["after_cost_expectancy"] is None
    assert card["raw"]["sharpe_annualized"] is None


# --------------------------------------------------------------------- #
# The three-way decision readout, net-cost-known case                   #
# --------------------------------------------------------------------- #
def test_hedge_improves_expectancy_and_cuts_drawdown_is_signal_real_but_noisy():
    # raw: choppy, occasionally very negative. hedged: same sign pattern but
    # smaller magnitude losses AND higher expectancy.
    raw = [0.01, -0.05, 0.02, -0.04, 0.01, -0.03]
    hedged = [0.015, -0.01, 0.02, -0.008, 0.012, -0.005]
    rows = [_row(f"2026-01-{i+1:02d}", r, h, h) for i, (r, h) in enumerate(zip(raw, hedged))]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["raw"]["after_cost_expectancy"] < card["hedged_net"]["after_cost_expectancy"]
    assert card["hedged_net"]["max_drawdown"] < card["raw"]["max_drawdown"]
    assert card["decision"]["verdict"] == hsc.VERDICT_SIGNAL_REAL_BUT_NOISY


def test_raw_positive_hedged_negative_is_beta_not_alpha():
    raw = [0.02, 0.015, 0.018, 0.022, 0.019, 0.017]  # steadily positive
    hedged = [-0.001, -0.002, 0.0005, -0.0015, -0.0008, -0.001]  # ~flat/negative
    rows = [_row(f"2026-01-{i+1:02d}", r, h, h) for i, (r, h) in enumerate(zip(raw, hedged))]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["raw"]["after_cost_expectancy"] > 0
    assert card["hedged_net"]["after_cost_expectancy"] <= 0
    assert card["decision"]["verdict"] == hsc.VERDICT_BETA_NOT_ALPHA


def test_hedge_cuts_vol_but_still_negative_is_weak_or_dead():
    # raw: large volatile swings, mean -0.015. hedged: smaller, steadier
    # decline, mean -0.0208 -- hedged does NOT beat raw's expectancy (rules
    # out signal_real_but_noisy), but its drawdown IS smaller (vol cut),
    # and it's still net negative -- the "there was nothing to protect"
    # pattern.
    raw = [0.05, -0.08, 0.06, -0.09, 0.04, -0.07]
    hedged = [-0.02, -0.025, -0.018, -0.022, -0.019, -0.021]
    rows = [_row(f"2026-01-{i+1:02d}", r, h, h) for i, (r, h) in enumerate(zip(raw, hedged))]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["hedged_net"]["after_cost_expectancy"] <= card["raw"]["after_cost_expectancy"]
    assert card["hedged_net"]["max_drawdown"] < card["raw"]["max_drawdown"]
    assert card["hedged_net"]["after_cost_expectancy"] <= 0
    assert card["decision"]["verdict"] == hsc.VERDICT_WEAK_OR_DEAD


# --------------------------------------------------------------------- #
# Cost-unmodeled venue: gross shown, net honestly withheld               #
# --------------------------------------------------------------------- #
def test_cost_unmodeled_venue_reports_gross_verdict_prefixed_never_net():
    raw = [0.02, 0.018, 0.021, 0.019, 0.022, 0.017]
    gross = [-0.001, 0.0005, -0.002, 0.0002, -0.001, -0.0015]
    # hedged.net_return is None on every row -> cost genuinely unknown
    rows = [_row(f"2026-01-{i+1:02d}", r, None, g, status="applied") for i, (r, g) in enumerate(zip(raw, gross))]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["cost_unmodeled_for_venue"] is True
    assert card["hedged_net"]["n"] == 0
    assert card["hedged_net"]["after_cost_expectancy"] is None
    assert card["hedged_gross"]["after_cost_expectancy"] is not None
    assert card["decision"]["verdict"].startswith("cost_unmodeled:")
    assert card["decision"]["basis"] == "gross"
    assert "not after-cost" in card["decision"]["reason"] or "NOT after-cost" in card["decision"]["reason"]


def test_partial_cost_known_across_history_is_not_treated_as_fully_unmodeled():
    """cost_unmodeled_for_venue is only true when EVERY applied cycle came
    back cost_known=False -- a mixed history (some cycles had real fill
    data) must not be mislabeled as fully unknown."""
    rows = [
        _row("2026-01-01", 0.01, 0.005, 0.005),  # cost known (net present)
        _row("2026-01-02", 0.01, None, 0.005),   # cost unknown this cycle
        _row("2026-01-03", 0.01, 0.004, 0.004),
    ]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["cost_unmodeled_for_venue"] is False


def test_raw_sharpe_annualizes_using_only_resolved_raw_dates():
    """Regression for a real bug found in review: pairing the FULL
    (unfiltered) asof_date list with the FILTERED (resolved-only) raw
    return series silently miscomputes the annualization cadence whenever a
    cycle has no resolved raw price (interleaved daily rows here). The
    correct median gap across the 3 RESOLVED dates (Jan 1, 5, 9) is 4 days;
    using all 6 logged dates (daily) would wrongly compute a 1-day cadence
    and inflate/distort the Sharpe by a large, spurious factor."""
    rows = [
        _row("2026-01-01", 0.01, 0.005, 0.005),
        _row("2026-01-02", None, 0.005, 0.005),   # raw unresolved this cycle
        _row("2026-01-03", None, 0.005, 0.005),
        _row("2026-01-05", -0.02, 0.005, 0.005),
        _row("2026-01-07", None, 0.005, 0.005),
        _row("2026-01-09", 0.015, 0.005, 0.005),
    ]
    resolved_dates = ["2026-01-01", "2026-01-05", "2026-01-09"]
    expected_gap = hsc._median_gap_days(resolved_dates)
    assert expected_gap == pytest.approx(4.0)

    card = hsc.compute_strategy_scorecard(rows)
    assert card["raw"]["n"] == 3
    raw_returns = [0.01, -0.02, 0.015]
    expected_std = hsc._sample_std(raw_returns)
    expected_expectancy = hsc._mean(raw_returns)
    expected_sharpe = (expected_expectancy / expected_std) * ((hsc.CALENDAR_DAYS_PER_YEAR / expected_gap) ** 0.5)
    assert card["raw"]["sharpe_annualized"] == pytest.approx(expected_sharpe)


def test_hedged_series_excludes_non_applied_cycles_never_diluted():
    """Regression for a real bug found in review: cycles where no hedge was
    applied set hedged == raw by construction (nothing to hedge with) -- if
    those trivial rows leak into the "hedged" statistics series, a strategy
    with a MOSTLY-unavailable hedge would get its alpha-vs-beta comparison
    silently diluted toward "hedged tracks raw" for reasons that have
    nothing to do with hedge economics. Only status=="applied" cycles may
    contribute to hedged_net/hedged_gross."""
    rows = [
        _row("2026-01-01", 0.05, 0.05, 0.05, status="fail_closed"),   # hedged==raw, NOT a real hedge result
        _row("2026-01-02", 0.04, 0.04, 0.04, status="no_valid_hedge"),  # same
        _row("2026-01-03", 0.02, -0.01, -0.01, status="applied"),     # the only REAL hedge result
    ]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["hedged_net"]["n"] == 1, "only the applied cycle may enter the hedged series"
    assert card["hedged_net"]["after_cost_expectancy"] == pytest.approx(-0.01)


def test_no_hedge_available_verdict_uses_same_n_as_insufficient_history_gate():
    """Regression for a real bug found in review: hedge_unavailable's
    insufficient-history check used len(rows) (total logged cycles) while
    decision_readout used raw_stats['n'] (resolved-raw-return count) -- a
    history with plenty of logged rows but few resolved raw returns and zero
    applied hedges must report insufficient_history, not no_hedge_available."""
    rows = [
        _row("2026-01-01", 0.01, None, None, status="fail_closed"),
        _row("2026-01-02", None, None, None, status="fail_closed"),
        _row("2026-01-03", None, None, None, status="fail_closed"),
        _row("2026-01-04", None, None, None, status="fail_closed"),
        _row("2026-01-05", 0.02, None, None, status="fail_closed"),
    ]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["raw"]["n"] == 2, "only 2 of 5 logged rows have a resolved raw return"
    assert card["decision"]["verdict"] == hsc.VERDICT_INSUFFICIENT_HISTORY


# --------------------------------------------------------------------- #
# No hedge available (crypto / fail-closed) — honest, distinct verdict   #
# --------------------------------------------------------------------- #
def test_no_hedge_ever_applied_reports_no_hedge_available_not_a_fabricated_comparison():
    rows = [_row(f"2026-01-{i+1:02d}", 0.01, 0.01, 0.01, status="unsupported_asset_class")
            for i in range(5)]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["hedge_unavailable"] is True
    assert card["decision"]["verdict"] == "no_hedge_available"


# --------------------------------------------------------------------- #
# Sharpe/Sortino never fabricated on degenerate input                    #
# --------------------------------------------------------------------- #
def test_sharpe_none_on_zero_variance_series_no_div_by_zero():
    rows = [_row(f"2026-01-{i+1:02d}", 0.01, 0.01, 0.01) for i in range(5)]
    card = hsc.compute_strategy_scorecard(rows)
    assert card["raw"]["sharpe_annualized"] is None
    assert card["raw"]["after_cost_expectancy"] == pytest.approx(0.01)


def test_max_drawdown_matches_hand_computed_equity_curve():
    returns = [0.10, -0.20, 0.05]
    # equity: 1.10 -> 0.88 -> 0.924 ; peak=1.10 ; trough=0.88 ; dd=(1.10-0.88)/1.10
    expected_dd = (1.10 - 0.88) / 1.10
    assert hsc._max_drawdown(returns) == pytest.approx(expected_dd)


# --------------------------------------------------------------------- #
# Safety triplet + read-only contract                                    #
# --------------------------------------------------------------------- #
def test_safety_triplet_present_on_every_report():
    report = hsc.build_full_scorecard(ledger_path=Path("/nonexistent/ledger.jsonl"))
    assert report["runtime_allowed"] is False
    assert report["paper_only"] is True
    assert report["human_review_required"] is True
    assert report["strategies_with_history"] == []


def test_missing_ledger_file_returns_empty_not_crash():
    rows = hsc.read_ledger_rows(Path("/nonexistent/ledger_xyz.jsonl"))
    assert rows == []


def test_no_execution_or_broker_import_anywhere_in_module():
    source = Path(hsc.__file__).read_text(encoding="utf-8")
    forbidden = ("import src.brokers", "from src.brokers", "place_market_order",
                 "place_limit_order", "StateEngine", "set_halted", "LiveGate.arm")
    for token in forbidden:
        assert token not in source, f"forbidden execution/control token found: {token}"
