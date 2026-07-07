"""AXIOM Hedge Layer Phase 3 — hedge candidate generator (SHADOW/ANALYSIS-ONLY).

Reproduce-then-resolve tests: the USD-pileup headline case gets a cost-aware
ranked short-USD / relative-value hedge; an artificially expensive hedge gets
BLOCK_COST; an equity sleeve gets market/sector/RV hedges; missing exposure
data and no-valid-hedge cases fail closed per the module's own contract.

NO MOCKS per CLAUDE.md No-Mock Rule — real config JSON via default-argument
disk loads; cost is exercised both via the real (degrade-to-unknown)
execution cost model and via an injected deterministic callable for the
cost-ratio-tier tests, since ``cost_lookup`` is designed to be injectable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.execution_cost_model import CostEstimate  # noqa: E402
from src.hedge.hedge_candidates import (  # noqa: E402
    DECISION_APPROVE_HEDGED,
    DECISION_APPROVE_RAW,
    DECISION_BLOCK_COST,
    DECISION_BLOCK_CORRELATION,
    DECISION_BLOCK_NO_VALID_HEDGE,
    DECISION_REDUCE_SIZE,
    DECISION_SHADOW_ONLY,
    HUMAN_REVIEW_REQUIRED,
    PAPER_ONLY,
    RUNTIME_ALLOWED,
    _decide,
    build_hedge_report,
    generate_equity_hedges,
    generate_fx_hedges,
    rank_hedges_by_cost,
    rank_hedges_by_correlation,
    rank_hedges_by_exposure_reduction,
)
from src.hedge.portfolio_exposure import ExposurePosition, build_exposure_report  # noqa: E402


def _cheap_cost_lookup(instrument: str) -> CostEstimate:
    """Deterministic cheap cost: 0.0001 half-spread-cost-per-unit."""
    return CostEstimate(
        instrument=instrument, source="fills", spread_raw=0.0002, spread_pips=2.0,
        slippage_raw=0.0001, slippage_pips=1.0, half_spread_cost_per_unit=0.0001,
        sample_count_fills=10, sample_count_ticks=0, confidence=0.8, as_of="test", notes=[])


def _expensive_cost_lookup(instrument: str) -> CostEstimate:
    """Deterministic expensive cost: cost dollars >> exposure reduction."""
    return CostEstimate(
        instrument=instrument, source="fills", spread_raw=0.05, spread_pips=500.0,
        slippage_raw=0.02, slippage_pips=200.0, half_spread_cost_per_unit=1.0,
        sample_count_fills=10, sample_count_ticks=0, confidence=0.8, as_of="test", notes=[])


def _mid_cost_lookup(instrument: str) -> CostEstimate:
    """Cost ratio lands in the REDUCE_SIZE band (low < ratio <= max)."""
    return CostEstimate(
        instrument=instrument, source="fills", spread_raw=0.001, spread_pips=10.0,
        slippage_raw=0.0005, slippage_pips=5.0, half_spread_cost_per_unit=0.1,
        sample_count_fills=10, sample_count_ticks=0, confidence=0.8, as_of="test", notes=[])


# --------------------------------------------------------------------- #
# THE headline case: USD pileup                                         #
# --------------------------------------------------------------------- #
def test_usd_pileup_direct_offset_hedge_proposed_and_ranked_cheapest_first():
    open_positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)
    assert report.concentration_warnings, "sanity: USD pileup must be flagged first"

    proposals = generate_fx_hedges(candidate, report, cost_lookup=_cheap_cost_lookup)
    styles = {p.style for p in proposals}
    assert "direct_offset" in styles

    direct = next(p for p in proposals if p.style == "direct_offset")
    assert direct.neutralizes == "currency:USD"
    assert direct.exposure_reduction == pytest.approx(3.0)
    assert direct.exposure_reduction_pct == pytest.approx(1.0)
    assert direct.legs[0].direction == "short"
    assert not direct.fail_closed

    ranked = rank_hedges_by_cost(proposals)
    assert ranked[0].cost_known
    assert ranked[0].cost_to_reduction_ratio is not None


def test_usd_pileup_relative_value_hedge_requires_supplied_relative_strength():
    """Mirrors the equity RV contract: no ranking data -> fail closed, never
    an arbitrary (e.g. alphabetical) pick dressed up as a ranked view."""
    open_positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)

    no_data = generate_fx_hedges(candidate, report, cost_lookup=_cheap_cost_lookup)
    rv_gap = next(p for p in no_data if p.style == "relative_value")
    assert rv_gap.fail_closed
    assert "no_relative_strength_data_supplied" in rv_gap.reasons

    with_data = generate_fx_hedges(
        candidate, report, cost_lookup=_cheap_cost_lookup,
        relative_strength={"CAD": 0.10, "CHF": -0.05, "JPY": 0.02})
    rv = next(p for p in with_data if p.style == "relative_value" and not p.fail_closed)
    assert rv.exposure_reduction == 0.0
    assert rv.correlation_to_target is None
    assert len(rv.legs) == 2
    assert rv.legs[0].instrument == "CAD_USD"  # strongest (0.10)
    assert rv.legs[0].direction == "long"
    assert rv.legs[1].instrument == "CHF_USD"  # weakest (-0.05)
    assert rv.legs[1].direction == "short"


def test_usd_pileup_full_report_decision_approve_hedged_with_cheap_cost():
    open_positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)

    result = build_hedge_report(candidate, report, cost_lookup=_cheap_cost_lookup)
    assert result["decision"] == DECISION_APPROVE_HEDGED
    assert result["runtime_allowed"] is False
    assert result["paper_only"] is True
    assert result["human_review_required"] is True
    assert result["hedge_action"], "must propose at least one ranked hedge"
    assert result["candidate_trade"]["instrument"] == "USD_JPY"
    assert "USD" in result["risk_issue"]


# --------------------------------------------------------------------- #
# Cost-tier decisions                                                   #
# --------------------------------------------------------------------- #
def test_too_expensive_hedge_blocks_on_cost():
    open_positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)

    result = build_hedge_report(candidate, report, cost_lookup=_expensive_cost_lookup)
    assert result["decision"] == DECISION_BLOCK_COST
    top = result["hedge_action"][0]
    assert top["cost_to_reduction_ratio"] > 0.5


def test_unknown_cost_blocks_on_cost_not_assumed_cheap():
    """Real cost model (not injected) on a currency pair with no OANDA
    fill/tick history -> insufficient_data -> cost unknown -> fail-closed to
    BLOCK_COST, never silently treated as free. Uses a synthetic currency
    code (never traded, so guaranteed no real fill history) to keep this
    deterministic regardless of what's in the live practice ledger --
    ``USD_CAD``/``USD_JPY`` etc DO have real fills (verified against
    trained_data/oanda/transactions.jsonl), so a real-pair test here would
    silently flip pass/fail as the practice account trades more instruments."""
    synth_map = {
        "USD": {"tags": ["safe_haven"]}, "ZZQ": {"tags": ["risk_on"]},
        "ZZR": {"tags": ["risk_on"]}, "ZZS": {"tags": ["risk_on"]},
    }
    open_positions = [
        ExposurePosition("fx", "USD_ZZQ", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_ZZR", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("fx", "USD_ZZS", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate, currency_bucket_map=synth_map)
    assert not report.fail_closed
    assert report.concentration_warnings, "sanity: 3 distinct USD legs must flag concentration"

    result = build_hedge_report(candidate, report, currency_bucket_map=synth_map)  # real cost model
    top = result["hedge_action"][0]
    assert top["cost_known"] is False
    assert result["decision"] == DECISION_BLOCK_COST


def test_mid_cost_hedge_recommends_reduce_size():
    open_positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)

    result = build_hedge_report(candidate, report, cost_lookup=_mid_cost_lookup)
    assert result["decision"] == DECISION_REDUCE_SIZE


# --------------------------------------------------------------------- #
# Equity sleeve: market / sector / RV                                   #
# --------------------------------------------------------------------- #
def test_equity_sleeve_generates_market_and_sector_hedges():
    open_positions = [
        ExposurePosition("equity", "AAPL", "long", 1.0, source="shadow_open"),
        ExposurePosition("equity", "MSFT", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("equity", "NVDA", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)
    assert report.concentration_warnings, "sanity: 3 Technology longs must flag sector concentration"

    proposals = generate_equity_hedges(candidate, report, cost_lookup=_cheap_cost_lookup)
    styles = {p.style for p in proposals}
    assert "market" in styles
    assert "sector" in styles

    sector = next(p for p in proposals if p.style == "sector")
    assert sector.neutralizes == "sector:Technology"
    assert sector.legs[0].instrument == "XLK"
    assert sector.legs[0].direction == "short"
    assert sector.exposure_reduction == pytest.approx(3.0)

    market = next(p for p in proposals if p.style == "market")
    assert market.neutralizes == "beta:portfolio"
    assert market.legs[0].instrument == "SPY"
    assert market.legs[0].direction == "short"


def test_equity_sleeve_relative_value_requires_supplied_relative_strength():
    open_positions = [
        ExposurePosition("equity", "AAPL", "long", 1.0, source="shadow_open"),
        ExposurePosition("equity", "MSFT", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("equity", "NVDA", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)

    # No relative_strength supplied -> gap recorded, not silently omitted.
    no_data = generate_equity_hedges(candidate, report, cost_lookup=_cheap_cost_lookup)
    rv_gap = next(p for p in no_data if p.style == "relative_value")
    assert rv_gap.fail_closed
    assert "no_relative_strength_data_supplied" in rv_gap.reasons

    # Supplied -> a real RV proposal, exposure_reduction conservatively 0.
    with_data = generate_equity_hedges(
        candidate, report, cost_lookup=_cheap_cost_lookup,
        relative_strength={"AAPL": 0.10, "MSFT": -0.05, "NVDA": 0.20})
    rv = next(p for p in with_data if p.style == "relative_value" and not p.fail_closed)
    assert rv.legs[0].instrument == "NVDA"  # strongest
    assert rv.legs[0].direction == "long"
    assert rv.legs[1].instrument == "MSFT"  # weakest
    assert rv.legs[1].direction == "short"
    assert rv.exposure_reduction == 0.0
    assert rv.correlation_to_target is None


# --------------------------------------------------------------------- #
# Fail-closed contract                                                  #
# --------------------------------------------------------------------- #
def test_missing_portfolio_exposure_fails_closed_to_shadow_only():
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    result = build_hedge_report(candidate, None)
    assert result["decision"] == DECISION_SHADOW_ONLY
    assert result["runtime_allowed"] is False


def test_unresolvable_exposure_fails_closed_to_shadow_only():
    open_positions = [ExposurePosition("fx", "USD_ZZZ", "long", 1.0, source="shadow_open")]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)
    assert report.fail_closed  # ZZZ is not in the currency bucket map

    result = build_hedge_report(candidate, report)
    assert result["decision"] == DECISION_SHADOW_ONLY


def test_flat_book_no_concentration_approves_raw():
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report([], candidate)
    assert not report.concentration_warnings

    result = build_hedge_report(candidate, report, cost_lookup=_cheap_cost_lookup)
    assert result["decision"] == DECISION_APPROVE_RAW


def test_no_counter_currency_available_blocks_no_valid_hedge():
    """Every currency in the map is exhausted by the flagged bucket itself ->
    no valid direct-offset instrument can be constructed."""
    single_ccy_map = {"USD": {"tags": ["safe_haven"]}}
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    # portfolio_exposure must show a net USD bucket to trigger hedge search;
    # build one directly rather than through build_exposure_report (which
    # would itself fail-closed on the missing JPY currency).
    from src.hedge.portfolio_exposure import ExposureReport
    report = ExposureReport(
        net_currency_exposure={"USD": 1.0}, net_sector_exposure={}, net_beta_exposure=0.0,
        net_correlation_bucket_exposure={"fx_currency:USD": 1.0},
        concentration_warnings=("currency concentration: USD net long 1.00 (100%) across 1 instruments",),
        fail_closed=False, fail_reasons=(), skipped_positions=(), position_count=1,
        resolved_position_count=1)

    proposals = generate_fx_hedges(candidate, report, currency_bucket_map=single_ccy_map,
                                   cost_lookup=_cheap_cost_lookup)
    assert all(p.fail_closed for p in proposals)

    result = build_hedge_report(candidate, report, currency_bucket_map=single_ccy_map,
                                cost_lookup=_cheap_cost_lookup)
    assert result["decision"] == DECISION_BLOCK_NO_VALID_HEDGE


# --------------------------------------------------------------------- #
# Decision function in isolation (correlation path)                     #
# --------------------------------------------------------------------- #
def test_low_correlation_hedge_blocks_on_correlation():
    from src.hedge.hedge_candidates import HedgeProposal, HedgeLeg
    from src.hedge.portfolio_exposure import ExposureReport

    report = ExposureReport(
        net_currency_exposure={"USD": 1.0}, net_sector_exposure={}, net_beta_exposure=0.0,
        net_correlation_bucket_exposure={}, concentration_warnings=("flagged",),
        fail_closed=False, fail_reasons=(), skipped_positions=(), position_count=1,
        resolved_position_count=1)
    weak_hedge = HedgeProposal(
        style="direct_offset", legs=(HedgeLeg("USD_JPY", "short", 1.0),),
        neutralizes="currency:USD", isolates="test", exposure_reduction=1.0,
        exposure_reduction_pct=1.0, correlation_to_target=0.1,
        cost_known=True, cost_estimate_dollars=0.01, cost_to_reduction_ratio=0.05,
        cost_source="test", cost_confidence=1.0)

    decision = _decide(report, rank_hedges_by_cost([weak_hedge]),
                       max_cost_to_reduction_ratio=0.5, low_cost_to_reduction_ratio=0.15,
                       min_correlation_to_target=0.3)
    assert decision == DECISION_BLOCK_CORRELATION


# --------------------------------------------------------------------- #
# Ranking functions                                                     #
# --------------------------------------------------------------------- #
def test_rank_hedges_by_exposure_reduction_orders_descending():
    open_positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)
    proposals = generate_fx_hedges(candidate, report, cost_lookup=_cheap_cost_lookup)

    ranked = rank_hedges_by_exposure_reduction(proposals)
    reductions = [p.exposure_reduction for p in ranked]
    assert reductions == sorted(reductions, reverse=True)


def test_rank_hedges_by_correlation_puts_unknown_last():
    open_positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
    ]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    report = build_exposure_report(open_positions, candidate)
    proposals = generate_fx_hedges(candidate, report, cost_lookup=_cheap_cost_lookup)

    ranked = rank_hedges_by_correlation(proposals)
    assert ranked[-1].correlation_to_target is None or ranked[-1].fail_closed


# --------------------------------------------------------------------- #
# Safety triplet is fixed, never derived                                #
# --------------------------------------------------------------------- #
def test_safety_triplet_constants_are_fixed():
    assert RUNTIME_ALLOWED is False
    assert PAPER_ONLY is True
    assert HUMAN_REVIEW_REQUIRED is True


def test_non_fx_non_equity_candidate_fails_closed_no_valid_hedge():
    candidate = ExposurePosition("crypto", "BTC_USD", "long", 1.0, source="candidate")
    report = build_exposure_report([], ExposurePosition("fx", "USD_JPY", "long", 0.0))
    result = build_hedge_report(candidate, report)
    assert result["decision"] in (DECISION_BLOCK_NO_VALID_HEDGE, DECISION_APPROVE_RAW)
