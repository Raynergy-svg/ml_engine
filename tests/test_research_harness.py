"""No-mock tests for the Track B research-alpha backtest harness.

Everything here uses REAL synthetic data: a deterministically-constructed price
panel + real :class:`ResearchScore` objects. No network, no LLM, no mock.

The central fixture builds a world in which the real composite GENUINELY
predicts returns (a name's per-bar drift is a monotone function of its composite
score), so:

* the PRIMARY (full / pre / post) arms show positive Sharpe, and
* the PLACEBO arm — which permutes the score<->ticker link — collapses to ~0.

This is the load-bearing falsification test for the placebo control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity.research.contracts import (
    ARM_FULL,
    ARM_PLACEBO,
    ARM_POST_CUTOFF,
    ARM_PRE_CUTOFF,
    N_TRIALS,
    PRIMARY_WEIGHTS,
    ResearchScore,
)
from src.equity.research import harness as H
from src.equity.research.harness import (
    ResearchHarnessError,
    run_research_backtest,
)
from src.factor.ship_gate import evaluate_gate


# ---------------------------------------------------------------------------
# Synthetic-world builders (deterministic, no RNG-driven labels)
# ---------------------------------------------------------------------------

N_TICKERS = 50
START = "2014-01-01"


def _tickers(n: int = N_TICKERS) -> list:
    return [f"T{i:02d}" for i in range(n)]


def _composite_for(i: int, n: int) -> float:
    """A spread of composites in [-0.5, +0.5] across the n names (i-indexed)."""
    # Linear ramp: name 0 worst, name n-1 best.
    return -0.5 + (i / (n - 1))


def _score_for(ticker: str, i: int, n: int, as_of: str) -> ResearchScore:
    """Build a ResearchScore whose PRIMARY composite equals _composite_for(i,n).

    PRIMARY_WEIGHTS = 0.5*quality - 0.5*red_flags + 0.0*outlook. We set
    red_flags=0 and quality so 0.5*quality == target composite.
    """
    target = _composite_for(i, n)
    quality = 2.0 * target  # 0.5 * quality == target
    quality = float(np.clip(quality, -1.0, 1.0))
    return ResearchScore(
        ticker=ticker,
        as_of=as_of,
        fundamental_quality=quality,
        accounting_red_flags=0.0,
        forward_outlook=0.0,
        conviction=0.6,
        rationale="synthetic",
        spans_used=["span-1"],
    )


def _build_world(
    *,
    years: int = 12,
    signal_strength: float = 0.0008,
    seed: int = 7,
):
    """Build (scores, prices) where composite predicts forward drift.

    Returns (scores, prices, cutoff_iso). Each name's daily return is a small
    deterministic drift proportional to its composite, plus a fixed-seed noise
    term shared by NO label (the noise does not encode the composite). Because
    drift is monotone in composite, a top-quintile book earns the highest names'
    drift -> positive Sharpe. The placebo breaks the score<->ticker map.
    """
    tickers = _tickers()
    n = len(tickers)
    dates = pd.bdate_range(START, periods=years * 252, tz="UTC")
    rng = np.random.default_rng(seed)

    # Per-name constant drift from composite; small common noise.
    drifts = np.array(
        [signal_strength * (_composite_for(i, n) * 2.0) for i in range(n)]
    )
    noise = rng.normal(0.0, 0.006, size=(len(dates), n))
    rets = noise + drifts[None, :]
    log_px = np.cumsum(rets, axis=0)
    px = 100.0 * np.exp(log_px)
    prices = pd.DataFrame(px, index=dates, columns=tickers)

    # One score per ticker, as_of well before the first rebalance, refreshed
    # again mid-sample so both pre- and post-cutoff arms have a current score.
    early = dates[5].date().isoformat()
    mid_idx = len(dates) // 2
    mid = dates[mid_idx].date().isoformat()
    scores = []
    for i, t in enumerate(tickers):
        scores.append(_score_for(t, i, n, early))
        scores.append(_score_for(t, i, n, mid))

    # Cutoff sits ~60% through the sample, so both pre and post arms are non-empty.
    cutoff = dates[int(len(dates) * 0.6)].date().isoformat()
    return scores, prices, cutoff


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_arm_splitting_pre_post_cutoff():
    """Rebalances split correctly around the model cutoff."""
    scores, prices, cutoff = _build_world()
    prices = H._normalize_prices(prices)
    all_rb = H._rebalance_dates(prices.index, 21)
    cutoff_ts = H._to_utc_ts(cutoff)
    pre = [d for d in all_rb if d < cutoff_ts]
    post = [d for d in all_rb if d >= cutoff_ts]

    assert len(pre) > 0 and len(post) > 0
    assert len(pre) + len(post) == len(all_rb)
    assert all(d < cutoff_ts for d in pre)
    assert all(d >= cutoff_ts for d in post)


def test_arm_splitting_through_public_api():
    """The public API routes the right rebalances to pre/post arms."""
    scores, prices, cutoff = _build_world()
    out = run_research_backtest(scores, prices, cutoff)
    # Pre and post arms exist and are not skipped (both windows non-empty).
    assert out[ARM_PRE_CUTOFF].get("skipped") is not True
    assert out[ARM_POST_CUTOFF].get("skipped") is not True
    assert out[ARM_POST_CUTOFF]["is_clean_arm"] is True
    # Full arm spans the whole sample -> more bars than either sub-arm.
    full_bars = out[ARM_FULL]["equity_curve_summary"]["n_bars"]
    pre_bars = out[ARM_PRE_CUTOFF]["equity_curve_summary"]["n_bars"]
    post_bars = out[ARM_POST_CUTOFF]["equity_curve_summary"]["n_bars"]
    assert full_bars >= pre_bars
    assert full_bars >= post_bars


def test_placebo_destroys_signal():
    """PRIMARY arm shows positive Sharpe; PLACEBO collapses to ~0. Deterministic."""
    scores, prices, cutoff = _build_world()
    out = run_research_backtest(scores, prices, cutoff)

    full_sharpe = out[ARM_FULL]["report"]["net_sharpe"]
    placebo_sharpe = out[ARM_PLACEBO]["report"]["net_sharpe"]

    # The real composite predicts returns -> primary book has a real edge.
    assert full_sharpe > 0.3, f"expected positive primary Sharpe, got {full_sharpe}"
    # The placebo breaks score<->ticker -> edge collapses to ~0. Use the
    # prereg's frozen control threshold (|net Sharpe| < 0.15, §1.4).
    assert abs(placebo_sharpe) < 0.15, (
        f"placebo Sharpe {placebo_sharpe} should be ~0 (|.|<0.15) — a non-zero "
        "placebo means the pipeline is leaking signal"
    )
    # And the summary surfaces the frozen 'placebo is clean' control as True.
    assert out["summary"]["placebo_is_clean"] is True
    assert out["summary"]["arm_net_sharpe"][ARM_FULL] == full_sharpe


def test_placebo_is_deterministic():
    """Two runs on identical inputs produce identical placebo Sharpe (no RNG)."""
    scores, prices, cutoff = _build_world()
    out1 = run_research_backtest(scores, prices, cutoff)
    out2 = run_research_backtest(scores, prices, cutoff)
    assert (
        out1[ARM_PLACEBO]["report"]["net_sharpe"]
        == out2[ARM_PLACEBO]["report"]["net_sharpe"]
    )


def test_placebo_permute_preserves_distribution_and_breaks_link():
    """The permutation keeps the multiset of values but moves them off-ticker."""
    composites = {"AAA": -0.4, "BBB": -0.1, "CCC": 0.2, "DDD": 0.5}
    permuted = H._placebo_permute(composites)
    assert sorted(permuted.values()) == sorted(composites.values())
    # Roll-by-one: at least one ticker now carries a different name's value.
    moved = sum(1 for k in composites if composites[k] != permuted[k])
    assert moved >= 2
    # Deterministic.
    assert H._placebo_permute(composites) == permuted


def test_q5_selection_picks_top_composites_and_sums_to_one():
    """Q5 long-only selects the highest composites; weights sum to ~1."""
    n = 20
    composites = {f"T{i:02d}": _composite_for(i, n) for i in range(n)}
    book = H._quintile_long_only_weights(composites)
    # 20 names / 5 => 4 names in the top quintile.
    assert len(book) == 4
    # The selected names are the four highest composites.
    expected_top = {f"T{i:02d}" for i in range(n - 4, n)}
    assert set(book) == expected_top
    assert abs(sum(book.values()) - 1.0) < 1e-9
    # Equal weight.
    assert all(abs(w - 0.25) < 1e-9 for w in book.values())


def test_long_short_book_is_dollar_neutral():
    """Secondary Q5-Q1 book nets to ~0 (long leg +0.5, short leg -0.5)."""
    n = 20
    composites = {f"T{i:02d}": _composite_for(i, n) for i in range(n)}
    book = H._quintile_long_short_weights(composites)
    assert abs(sum(book.values())) < 1e-9
    assert abs(sum(w for w in book.values() if w > 0) - 0.5) < 1e-9
    assert abs(sum(w for w in book.values() if w < 0) + 0.5) < 1e-9


def test_gate_wiring_strong_passes_weak_fails():
    """A strong synthetic return series PASSES evaluate_gate; a weak one FAILS."""
    idx = pd.bdate_range(START, periods=12 * 252, tz="UTC")
    # Strong: steady positive drift, tiny vol -> high Sharpe, shallow DD, 12y.
    strong = pd.Series(0.0006 + 0.0 * np.arange(len(idx)), index=idx)
    strong = strong + np.random.default_rng(1).normal(0, 0.002, len(idx))
    strong_report = H._build_report(strong)
    assert evaluate_gate(strong_report).passed is True

    # Weak: zero-mean noise -> ~0 Sharpe -> fails net_sharpe criterion.
    weak = pd.Series(
        np.random.default_rng(2).normal(0, 0.01, len(idx)), index=idx
    )
    weak_report = H._build_report(weak)
    assert evaluate_gate(weak_report).passed is False


def test_causality_positions_depend_only_on_past():
    """Shuffling FUTURE prices does not change positions stamped from past info.

    Build the weight panel, pick a rebalance date in the first third, scramble
    every price row strictly AFTER that date, rebuild, and assert the stamped
    book on/before the date is byte-identical.
    """
    scores, prices, cutoff = _build_world()
    prices = H._normalize_prices(prices)
    rb_dates = H._rebalance_dates(prices.index, 21)

    scores_by_ticker = {}
    for s in scores:
        scores_by_ticker.setdefault(str(s.ticker), []).append(s)

    panel_a = H._build_weight_panel(
        scores_by_ticker, prices, rb_dates, PRIMARY_WEIGHTS,
        long_only=True, placebo=False,
    )

    # Choose a cut date in the first third of the index.
    cut = prices.index[len(prices) // 3]

    # Scramble FUTURE prices only (rows strictly after cut), deterministically.
    future_mask = prices.index > cut
    scrambled = prices.copy()
    fut = scrambled.loc[future_mask]
    # Reverse the future block row-wise + multiply by a constant -> very different.
    scrambled.loc[future_mask] = (fut.iloc[::-1].to_numpy() * 1.37)

    panel_b = H._build_weight_panel(
        scores_by_ticker, scrambled, rb_dates, PRIMARY_WEIGHTS,
        long_only=True, placebo=False,
    )

    # Positions on/before the cut must be identical (no lookahead).
    past_a = panel_a.loc[:cut]
    past_b = panel_b.loc[:cut].reindex(columns=past_a.columns)
    pd.testing.assert_frame_equal(past_a, past_b)


def test_full_result_shape_and_dsr_computed():
    """The returned dict has all four arms, a summary, and a real §4.5 DSR/Bonferroni block."""
    scores, prices, cutoff = _build_world()
    out = run_research_backtest(scores, prices, cutoff)
    for arm in (ARM_FULL, ARM_PRE_CUTOFF, ARM_POST_CUTOFF, ARM_PLACEBO):
        assert arm in out
    assert "summary" in out and "params" in out
    # DSR-OOS(N=22) is now computed for real (prereg §4.5) — never faked, but no
    # longer a hardcoded TODO None either. It abstains (None) only when there
    # are too few bars to estimate skew/kurtosis/variance meaningfully.
    dsr = out[ARM_FULL]["dsr_oos_n22"]
    assert dsr is not None
    assert "dsr" in dsr and "p_oos_bootstrap" in dsr and "passes_significance" in dsr
    assert out[ARM_FULL]["n_trials"] == N_TRIALS
    # effective-N is reported for a holding book.
    assert out[ARM_FULL]["effective_n"] is not None
    assert out[ARM_FULL]["effective_n"] > 0
    # Summary exposes the frozen control booleans.
    assert "placebo_is_clean" in out["summary"]
    assert "post_cutoff_consistent_with_full" in out["summary"]


def test_out_of_range_score_rejected():
    """A score outside its pre-registered range fails loud (no silent clip)."""
    scores, prices, cutoff = _build_world()
    bad = ResearchScore(
        ticker="T00",
        as_of="2014-01-10",
        fundamental_quality=2.5,  # out of [-1, 1]
        accounting_red_flags=0.0,
        forward_outlook=0.0,
        conviction=0.5,
        rationale="bad",
    )
    with pytest.raises(ValueError):
        run_research_backtest([bad] + scores, prices, cutoff)


def test_empty_scores_rejected():
    """Empty score list is rejected, not silently treated as a flat book."""
    _, prices, cutoff = _build_world()
    with pytest.raises(ResearchHarnessError):
        run_research_backtest([], prices, cutoff)


# ---------------------------------------------------------------------------
# Review fixes (2026-06-30): derangement placebo + binding §4.7 verdict +
# real sign-flip consistency. No mocks — pure logic on real summary dicts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 7, 16, 32, 50, 64, 127, 128, 256])
def test_placebo_derangement_has_no_fixed_points(n):
    """The placebo permutation must be a DERANGEMENT (no ticker keeps its own
    composite) — a fixed point would leak real signal into the falsification
    control. bit-reversal alone fixed up to 16 indices at n=128."""
    order = H._derangement_order(n)
    assert sorted(order) == list(range(n))  # valid permutation
    assert all(order[i] != i for i in range(n))  # zero fixed points


def _arm(sharpe, gate):
    return {"report": {"net_sharpe": float(sharpe)}, "gate_passed": bool(gate)}


def test_binding_verdict_contaminated_when_full_passes_but_placebo_dirty():
    """§4.7 override: a full-arm gate pass with a non-clean placebo MUST read
    LOOKAHEAD_CONTAMINATED, never REAL — the exact failure mode §1 exists for."""
    results = {
        ARM_FULL: _arm(2.0, True),
        ARM_POST_CUTOFF: _arm(1.5, False),
        ARM_PLACEBO: _arm(0.9, False),   # dirty (|Sharpe| >= 0.15)
        ARM_PRE_CUTOFF: _arm(2.0, False),
    }
    summary = H._build_summary(results)
    assert summary["overall_verdict"] == "LOOKAHEAD_CONTAMINATED"


def test_binding_verdict_never_real_while_blinding_audit_uncomputed():
    """Fail-closed: even a clean placebo + confirming post-cutoff cannot yield
    REAL while the human blinding audit (§1.1) is uncomputed."""
    results = {
        ARM_FULL: _arm(2.0, True),
        ARM_POST_CUTOFF: _arm(1.5, False),
        ARM_PLACEBO: _arm(0.05, False),  # clean
        ARM_PRE_CUTOFF: _arm(2.0, False),
    }
    summary = H._build_summary(results)
    assert summary["blinding_audit_clean"] is None
    assert summary["overall_verdict"] != "REAL"
    assert summary["post_cutoff_consistent_with_full"] is True


def test_consistency_false_when_both_arms_negative():
    """A both-negative case is NOT 'consistent' — there is no edge to confirm.
    The old `full<=0 or post>0` form wrongly returned True here."""
    results = {
        ARM_FULL: _arm(-0.5, False),
        ARM_POST_CUTOFF: _arm(-0.3, False),
        ARM_PLACEBO: _arm(0.05, False),
        ARM_PRE_CUTOFF: _arm(-0.5, False),
    }
    summary = H._build_summary(results)
    assert summary["post_cutoff_consistent_with_full"] is False
    assert summary["overall_verdict"] == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# §4.5 — DSR-OOS(N=22) + Bonferroni block-bootstrap (real computation, not TODO)
# ---------------------------------------------------------------------------


def _daily_returns(mean: float, std: float, n: int, *, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(rng.normal(mean, std, size=n), index=idx)


def test_dsr_abstains_below_minimum_bars():
    """Too few bars -> None, never a noise-driven number."""
    ret = _daily_returns(0.01, 0.01, 5, seed=1)
    assert H._deflated_sharpe_ratio(ret, 22) is None
    assert H._circular_block_bootstrap_sharpe_pvalue(ret) is None
    result = H._dsr_oos_n22(ret)
    assert result["dsr"] is None
    assert result["p_oos_bootstrap"] is None
    assert result["passes_significance"] is None
    assert result["insufficient_data"] is True


def test_dsr_high_for_strong_persistent_edge():
    """A strongly positive, low-noise return stream clears both DSR>=0.95 and
    the Bonferroni bootstrap p-value comfortably."""
    ret = _daily_returns(0.004, 0.003, 1000, seed=2)  # SR ~ 0.004/0.003*sqrt(252) ~ huge
    result = H._dsr_oos_n22(ret, n_trials=N_TRIALS)
    assert result["dsr"] is not None and result["dsr"] > 0.95
    assert result["p_oos_bootstrap"] is not None and result["p_oos_bootstrap"] < H.BONFERRONI_ALPHA
    assert result["passes_significance"] is True


def test_dsr_low_for_zero_mean_noise():
    """Pure zero-mean noise must NOT clear the multiple-testing bar."""
    ret = _daily_returns(0.0, 0.01, 1000, seed=3)
    result = H._dsr_oos_n22(ret, n_trials=N_TRIALS)
    assert result["dsr"] is not None and result["dsr"] < 0.95
    assert result["passes_significance"] is False


def test_dsr_bootstrap_deterministic_same_seed():
    """The block bootstrap is reproducible: identical input -> identical p-value."""
    ret = _daily_returns(0.002, 0.01, 500, seed=4)
    a = H._circular_block_bootstrap_sharpe_pvalue(ret)
    b = H._circular_block_bootstrap_sharpe_pvalue(ret)
    assert a == b


def test_build_summary_real_requires_significance_pass():
    """REAL requires the §4.5 significance extension on top of the gate + §1
    controls — a full-arm gate pass with dsr_oos_n22 UNSET (no significance
    result at all) must NOT read REAL even with every other control clean."""
    results = {
        ARM_FULL: _arm(2.0, True),
        ARM_POST_CUTOFF: _arm(1.5, False),
        ARM_PLACEBO: _arm(0.05, False),
        ARM_PRE_CUTOFF: _arm(2.0, False),
    }
    summary = H._build_summary(results, blinding_audit_clean=True)
    assert summary["full_significance_passes"] is False
    assert summary["overall_verdict"] != "REAL"


def test_build_summary_real_when_all_controls_and_significance_clear():
    """The one case that SHOULD read REAL: gate pass + clean placebo + confirming
    post-cutoff + clean blinding audit + a clearing §4.5 significance result."""
    full_arm = _arm(2.0, True)
    full_arm["dsr_oos_n22"] = {"dsr": 0.99, "passes_significance": True}
    results = {
        ARM_FULL: full_arm,
        ARM_POST_CUTOFF: _arm(1.5, False),
        ARM_PLACEBO: _arm(0.05, False),
        ARM_PRE_CUTOFF: _arm(2.0, False),
    }
    summary = H._build_summary(results, blinding_audit_clean=True)
    assert summary["full_significance_passes"] is True
    assert summary["overall_verdict"] == "REAL"
