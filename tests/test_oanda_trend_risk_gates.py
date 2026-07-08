"""Reproduce-then-resolve: RISK GATE for the OANDA trend lane (2026-07-01).

ROOT CAUSE reproduced from the LIVE practice book (queried 2026-07-01 06:4x):
every one of EUR_JPY/GBP_JPY/USD_CAD/USD_CHF/USD_JPY carried EXACTLY 2 open
trades, opened ~4.5h apart, in an EXACT ~3.7x size ratio across all five
instruments (e.g. EUR_JPY 26,764 -> 99,222). This is not random sizing — it is
the old leverage-based ``target_units`` re-sizing the WHOLE target every
cycle off gross_leverage*NAV; OANDA opens a NEW trade ID per market order
instead of merging into the existing one, so every re-size became a second
correlated ticket. The 6 gate rules below (src/equity/trend_risk_gates.py)
close this: one-position cap, correlation-bucket caps, risk-normalized
sizing, a bias detector, winner management, and pyramid gating.

NO MOCKS per CLAUDE.md No-Mock Rule. Every test drives the real pure
functions with real values (some lifted directly from the live book above);
persisted-state tests use tmp_path for real disk I/O.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.execution_cost_model import CostEstimate  # noqa: E402
from src.equity.trend_risk_gates import (  # noqa: E402
    DEFAULT_MAX_COST_R_FRACTION,
    RiskGateConfig,
    accumulate_bucket_risk,
    bias_detector,
    bucket_cap_gate,
    buckets_over_cap,
    clamp_risk_pct,
    compute_bucket_risk,
    cost_aware_gate,
    currency_legs,
    evaluate_winner_stop,
    leverage_cap_scale,
    load_risk_state,
    one_position_gate,
    pyramid_gate,
    quote_to_home_rate,
    risk_normalized_units,
    save_risk_state,
)


def _cost_estimate(*, source="fills", spread_raw=0.0002, slippage_raw=0.0,
                   instrument="EUR_USD") -> CostEstimate:
    return CostEstimate(
        instrument=instrument, source=source, spread_raw=spread_raw,
        spread_pips=None, slippage_raw=slippage_raw, slippage_pips=None,
        half_spread_cost_per_unit=None, sample_count_fills=5, sample_count_ticks=0,
        confidence=0.5, as_of="2026-07-08T00:00:00+00:00", notes=[])


NAV = 102818.5912
LAST_PX = {
    "USD_JPY": 162.737, "EUR_JPY": 185.675, "GBP_JPY": 215.504,
    "USD_CAD": 1.42157, "USD_CHF": 0.80912, "EUR_USD": 1.0850, "GBP_USD": 1.2650,
}
# Real ATR-derived R distances (sl_mult=2.0 * daily ATR), reverse-engineered
# from the live book (same instruments as the TP/SL fix's test fixtures).
R_DIST = {
    "USD_JPY": 1.15266, "EUR_JPY": 1.15266 * 1.5, "GBP_JPY": 2.35734,
    "USD_CAD": 0.0117, "USD_CHF": 0.00992,
}


# --------------------------------------------------------------------- #
# Rule 3 — risk-normalized sizing: JPY and non-JPY give the SAME dollar #
# risk despite very different price scales (kills the 26k-vs-99k bug)  #
# --------------------------------------------------------------------- #
def test_risk_normalized_units_same_dollar_risk_jpy_and_nonjpy():
    risk_pct = 0.01
    u_jpy = risk_normalized_units(instrument="USD_JPY", r_distance_quote=R_DIST["USD_JPY"],
                                  nav=NAV, risk_pct=risk_pct, last_prices=LAST_PX)
    u_cad = risk_normalized_units(instrument="USD_CAD", r_distance_quote=R_DIST["USD_CAD"],
                                  nav=NAV, risk_pct=risk_pct, last_prices=LAST_PX)
    # Dollar risk realized if stopped out = units * r_distance * quote_to_home.
    risk_home_jpy = u_jpy * R_DIST["USD_JPY"] * quote_to_home_rate("USD_JPY", LAST_PX)
    risk_home_cad = u_cad * R_DIST["USD_CAD"] * quote_to_home_rate("USD_CAD", LAST_PX)
    target = NAV * risk_pct
    assert abs(risk_home_jpy - target) / target < 0.01
    assert abs(risk_home_cad - target) / target < 0.01
    # Same target risk regardless of instrument's absolute price scale.
    assert abs(risk_home_jpy - risk_home_cad) / target < 0.02


def test_risk_normalized_units_refuses_on_missing_atr():
    assert risk_normalized_units(instrument="USD_JPY", r_distance_quote=None,
                                 nav=NAV, risk_pct=0.01, last_prices=LAST_PX) == 0
    assert risk_normalized_units(instrument="USD_JPY", r_distance_quote=0.0,
                                 nav=NAV, risk_pct=0.01, last_prices=LAST_PX) == 0


def test_clamp_risk_pct_bounds_and_nan_safety():
    assert clamp_risk_pct(0.20) == 0.03      # ceiling
    assert clamp_risk_pct(0.0001) == 0.005   # floor
    assert clamp_risk_pct(float("nan")) == 0.01
    assert clamp_risk_pct("garbage") == 0.01


def test_leverage_cap_scale_caps_total_notional_not_per_trade():
    want = {"USD_JPY": 200_000, "USD_CAD": 200_000}  # deliberately oversized
    scaled, factor = leverage_cap_scale(want, LAST_PX, NAV, gross_leverage=3.0)
    assert factor < 1.0
    # Both instruments' BASE is USD (base->home rate = 1.0) -> gross = sum(units).
    gross_notional = sum(abs(u) for u in scaled.values())
    assert gross_notional <= 3.0 * NAV + 1.0


def test_leverage_cap_scale_noop_when_under_cap():
    want = {"USD_JPY": 1000}
    scaled, factor = leverage_cap_scale(want, LAST_PX, NAV, gross_leverage=3.0)
    assert factor == 1.0
    assert scaled == want


# --------------------------------------------------------------------- #
# Rule 1 + 6 — one position per instrument / pyramid gating             #
# --------------------------------------------------------------------- #
def _trade(units, upl, open_time="2026-07-01T00:00:00Z"):
    return {"currentUnits": units, "unrealizedPL": upl, "openTime": open_time,
            "instrument": "EUR_JPY"}


def test_one_position_gate_reproduces_the_live_bug_then_blocks():
    """Reproduce: an instrument with an existing GREEN trade + a naive re-size
    candidate the SAME size as before (not smaller) -> the OLD code allowed
    this (it just called create_market_order unconditionally); the gate must
    now block it."""
    existing = [_trade(26764, 50.0)]
    allow, reason = one_position_gate(
        existing_trades_for_instrument=existing, candidate_units=99222,
        r_distance_quote=1.7, risk_unit_home=1028.19, quote_to_home=0.0054)
    assert allow is False
    assert reason == "pyramid_add_not_smaller_than_last_layer"


def test_one_position_gate_blocks_average_down_on_red_trade():
    existing = [_trade(26764, -41.37)]  # RED, matches the live EUR_JPY trade 1647
    allow, reason = one_position_gate(
        existing_trades_for_instrument=existing, candidate_units=10000,
        r_distance_quote=1.7, risk_unit_home=1028.19, quote_to_home=0.0054)
    assert allow is False
    assert reason == "pyramid_blocked_red_trade_no_averaging_down"


def test_one_position_gate_allows_new_entry_with_no_existing_trade():
    allow, reason = one_position_gate(
        existing_trades_for_instrument=[], candidate_units=50000,
        r_distance_quote=1.7, risk_unit_home=1028.19, quote_to_home=0.0054)
    assert allow is True
    assert reason == "new_entry_no_existing_trade"


def test_pyramid_gate_authorizes_smaller_add_on_green_within_cum_risk_cap():
    existing = [_trade(50000, 25.0)]  # green, risk_home = 50000*1.0*0.0054 = 270
    allow, reason = pyramid_gate(
        existing_trades_for_instrument=existing, candidate_units=10000,
        r_distance_quote=1.0, risk_unit_home=1028.19, quote_to_home=0.0054,
        cum_risk_r_cap=1.0)
    # cum = (50000+10000)*1.0*0.0054 = 324 <= 1028.19 -> OK, smaller than 50000 -> OK
    assert allow is True
    assert reason == "pyramid_add_ok"


def test_pyramid_gate_blocks_when_cum_risk_exceeds_cap():
    existing = [_trade(180000, 25.0)]  # risk_home = 180000*1.0*0.0054=972
    allow, reason = pyramid_gate(
        existing_trades_for_instrument=existing, candidate_units=20000,
        r_distance_quote=1.0, risk_unit_home=1028.19, quote_to_home=0.0054,
        cum_risk_r_cap=1.0)
    # cum = 200000*1.0*0.0054=1080 > 1028.19 -> blocked
    assert allow is False
    assert reason == "pyramid_cum_risk_exceeds_cap"


def test_pyramid_gate_blocks_at_max_layers():
    existing = [_trade(50000, 10.0), _trade(20000, 5.0), _trade(10000, 1.0)]
    allow, reason = pyramid_gate(
        existing_trades_for_instrument=existing, candidate_units=1000,
        r_distance_quote=1.0, risk_unit_home=1028.19, quote_to_home=0.0054,
        max_layers=3)
    assert allow is False
    assert reason == "pyramid_max_layers_reached"


# --------------------------------------------------------------------- #
# Rule 2 — correlation-bucket caps (shared-currency exposure)           #
# --------------------------------------------------------------------- #
def test_currency_legs_decomposes_base_and_quote():
    assert currency_legs("EUR_JPY") == ("EUR", "JPY")
    assert currency_legs("USD_CAD") == ("USD", "CAD")
    assert currency_legs("garbage") is None


def test_compute_bucket_risk_aggregates_yen_cross_longs_into_one_bucket():
    """3 correlated yen-cross longs (USD_JPY, EUR_JPY, GBP_JPY) all add to the
    SAME (JPY, 'short') bucket — this is the exact live-book pattern."""
    trades = [
        {"id": "1", "instrument": "USD_JPY", "currentUnits": 30569},
        {"id": "2", "instrument": "EUR_JPY", "currentUnits": 26764},
        {"id": "3", "instrument": "GBP_JPY", "currentUnits": 23052},
    ]
    fallback = {"USD_JPY": 1.15266, "EUR_JPY": 1.7, "GBP_JPY": 2.35}
    bucket_risk, bucket_insts = compute_bucket_risk(
        trades, {}, LAST_PX, r_distance_fallback_by_instrument=fallback)
    jpy_short = ("JPY", "short")
    assert jpy_short in bucket_risk
    assert len(bucket_insts[jpy_short]) == 3
    # 3 separate base-currency "long" buckets, each with 1 instrument.
    assert bucket_insts[("USD", "long")] == {"USD_JPY"}
    assert bucket_insts[("EUR", "long")] == {"EUR_JPY"}
    assert bucket_insts[("GBP", "long")] == {"GBP_JPY"}


def test_compute_bucket_risk_prefers_entry_anchored_r_over_current_atr_fallback():
    """FIX (verifier, 2026-07-01): an open trade's risk must be measured off its
    ENTRY-anchored R (persisted risk_state, keyed by trade_id) — NOT current ATR.
    Reproduces the ATR-shrink under-count: if volatility contracted after entry,
    using current ATR would understate this trade's real dollar-risk relative to
    what was actually risked at entry."""
    trade_id = "1665"
    trades = [{"id": trade_id, "instrument": "USD_JPY", "currentUnits": 113329}]
    entry_r = 1.15266          # the R actually risked at entry (higher-vol regime)
    shrunk_current_atr_r = 0.40  # ATR contracted hard since entry
    # OLD (buggy) behavior: fallback-only (current ATR) understates the risk.
    buggy_risk, _ = compute_bucket_risk(
        trades, {}, LAST_PX, r_distance_fallback_by_instrument={"USD_JPY": shrunk_current_atr_r})
    # NEW (fixed) behavior: entry-anchored R from risk_state takes precedence.
    fixed_risk, _ = compute_bucket_risk(
        trades, {trade_id: entry_r}, LAST_PX,
        r_distance_fallback_by_instrument={"USD_JPY": shrunk_current_atr_r})
    buggy_total = sum(buggy_risk.values()) / 2.0   # each trade double-counted (2 buckets)
    fixed_total = sum(fixed_risk.values()) / 2.0
    assert fixed_total > buggy_total * 2.5   # entry R is ~2.9x the shrunk current ATR
    assert abs(fixed_total - 113329 * entry_r * quote_to_home_rate("USD_JPY", LAST_PX)) < 0.01


def test_accumulate_bucket_risk_closes_the_same_cycle_bypass():
    """FIX (verifier, 2026-07-01) [HIGH]: 3 yen-cross longs newly flipping on in
    the SAME cycle must be capped at 2R in TOTAL, not each individually checked
    against an empty pre-cycle snapshot (which would let ~3R through — the
    realistic daily-trend case the operator named). Simulates the in-loop
    accumulation run_oanda_trend_cycle now performs between approvals."""
    risk_unit_home = NAV * 0.01
    max_bucket_risk_r = 2.0
    bucket_risk_home = {}   # empty pre-cycle snapshot — nothing open yet this cycle
    candidates = [
        ("USD_JPY", 100000, 1.15266),
        ("EUR_JPY", 100000, 1.7),
        ("GBP_JPY", 100000, 2.35),
    ]
    approved = []
    for inst, units, r_dist in candidates:
        q2h = quote_to_home_rate(inst, LAST_PX)
        allow, reason = bucket_cap_gate(
            instrument=inst, candidate_units=units, r_distance_quote=r_dist,
            last_prices=LAST_PX, bucket_risk_home=bucket_risk_home,
            risk_unit_home=risk_unit_home, max_bucket_risk_r=max_bucket_risk_r)
        if allow:
            approved.append(inst)
            # THE FIX: update the running total immediately, same as the
            # oanda_trend.py call site does after each approval this cycle.
            accumulate_bucket_risk(bucket_risk_home, inst, units * r_dist * q2h)
    # With in-loop accumulation, NOT all 3 can be approved — the cap is real.
    assert len(approved) < 3, "same-cycle bucket bypass reproduced: all 3 slipped through uncapped"
    final_jpy_short = bucket_risk_home.get(("JPY", "short"), 0.0)
    assert final_jpy_short <= max_bucket_risk_r * risk_unit_home + 1e-6


def test_accumulate_bucket_risk_without_the_fix_would_let_all_3_through():
    """Documents the bug this fix closes: WITHOUT in-loop accumulation (a static
    snapshot reused for every candidate this cycle), all 3 correlated adds pass
    individually even though their combined risk breaches the 2R cap."""
    risk_unit_home = NAV * 0.01
    max_bucket_risk_r = 2.0
    static_snapshot = {}  # never updated between candidates -> the bug
    candidates = [
        ("USD_JPY", 100000, 1.15266),
        ("EUR_JPY", 100000, 1.7),
        ("GBP_JPY", 100000, 2.35),
    ]
    approved = []
    for inst, units, r_dist in candidates:
        allow, _ = bucket_cap_gate(
            instrument=inst, candidate_units=units, r_distance_quote=r_dist,
            last_prices=LAST_PX, bucket_risk_home=static_snapshot,
            risk_unit_home=risk_unit_home, max_bucket_risk_r=max_bucket_risk_r)
        if allow:
            approved.append(inst)
    assert len(approved) == 3   # the bug: every one clears the empty snapshot
    total_risk = sum(units * r_dist * quote_to_home_rate(inst, LAST_PX)
                     for inst, units, r_dist in candidates)
    assert total_risk > max_bucket_risk_r * risk_unit_home  # jointly over cap


def test_buckets_over_cap_reports_honestly_when_no_trim_exists():
    risk_unit_home = NAV * 0.01
    over = {("JPY", "short"): 2.5 * risk_unit_home}
    warnings = buckets_over_cap(over, risk_unit_home=risk_unit_home, max_bucket_risk_r=2.0)
    assert len(warnings) == 1
    assert "JPY_short" in warnings[0]
    assert "NO automatic trim" in warnings[0]


def test_buckets_over_cap_silent_when_within_cap():
    risk_unit_home = NAV * 0.01
    within = {("JPY", "short"): 1.5 * risk_unit_home}
    assert buckets_over_cap(within, risk_unit_home=risk_unit_home, max_bucket_risk_r=2.0) == []


def test_bucket_cap_gate_blocks_the_third_correlated_yen_cross():
    """Reproduce the operator's exact scenario: 1%-per-trade risk unit, 2R
    bucket cap. Two yen-cross trades already use ~2R combined in the JPY-short
    bucket; a third correlated add must be BLOCKED even though it would pass
    fine as a standalone ticket."""
    risk_unit_home = NAV * 0.01
    existing_bucket_risk = {("JPY", "short"): 1.9 * risk_unit_home}
    allow, reason = bucket_cap_gate(
        instrument="GBP_JPY", candidate_units=30000, r_distance_quote=2.35,
        last_prices=LAST_PX, bucket_risk_home=existing_bucket_risk,
        risk_unit_home=risk_unit_home, max_bucket_risk_r=2.0)
    assert allow is False
    assert reason.startswith("bucket_cap_exceeded:JPY_short")


def test_bucket_cap_gate_allows_first_two_correlated_trades_under_cap():
    risk_unit_home = NAV * 0.01
    allow, reason = bucket_cap_gate(
        instrument="USD_JPY", candidate_units=8000, r_distance_quote=1.15266,
        last_prices=LAST_PX, bucket_risk_home={}, risk_unit_home=risk_unit_home,
        max_bucket_risk_r=2.0)
    assert allow is True
    assert reason == "bucket_cap_ok"


def test_bucket_cap_gate_catches_usd_long_bucket_from_2_different_pairs():
    """The operator's other real example: USD_CAD + USD_CHF = one long-USD bet."""
    risk_unit_home = NAV * 0.01
    existing_bucket_risk = {("USD", "long"): 2.05 * risk_unit_home}
    allow, reason = bucket_cap_gate(
        instrument="USD_CHF", candidate_units=10000, r_distance_quote=0.00992,
        last_prices=LAST_PX, bucket_risk_home=existing_bucket_risk,
        risk_unit_home=risk_unit_home, max_bucket_risk_r=2.0)
    assert allow is False
    assert reason.startswith("bucket_cap_exceeded:USD_long")


def test_bucket_cap_gate_refuses_when_quote_rate_unresolvable():
    """No fabrication: an instrument whose quote->home rate can't be derived from
    the traded panel must REFUSE, not silently pass with a made-up rate."""
    allow, reason = bucket_cap_gate(
        instrument="ZZZ_QQQ", candidate_units=10000, r_distance_quote=1.0,
        last_prices=LAST_PX, bucket_risk_home={}, risk_unit_home=NAV * 0.01,
        max_bucket_risk_r=2.0)
    assert allow is False
    assert reason == "bucket_cap_no_quote_rate"


def test_pyramid_gate_zero_quote_to_home_would_silently_disable_the_cap():
    """VERIFIER FINDING (2026-07-01): pyramid_gate's cumulative-risk check trusts
    its caller's ``quote_to_home`` input completely — passing 0.0 (e.g. as a naive
    fallback for an unresolvable rate) makes every trade_risk_home() compute as 0
    regardless of position size, so the cum_risk_r_cap check can never trip. This
    is exactly why the call site in run_oanda_trend_cycle (oanda_trend.py) now
    REFUSES the whole candidate (continue, before ever calling this gate) when
    quote_to_home_rate() returns None, instead of substituting 0.0. This test
    locks in the underlying mechanism so the danger can never be "fixed away"
    silently by relaxing pyramid_gate's cap arithmetic instead — the contract is:
    callers MUST supply a real quote_to_home rate, or not call this gate at all."""
    existing = [_trade(180000, 25.0)]  # would normally already be AT the cap
    allow, reason = pyramid_gate(
        existing_trades_for_instrument=existing, candidate_units=170000,  # still smaller
        r_distance_quote=1.0, risk_unit_home=1028.19, quote_to_home=0.0,  # than last layer
        cum_risk_r_cap=1.0)
    assert allow is True  # documents the danger: 0.0 defeats the cap entirely
    assert reason == "pyramid_add_ok"


# --------------------------------------------------------------------- #
# Rule 4 — bias detector (log-only diagnostic)                          #
# --------------------------------------------------------------------- #
def test_bias_detector_flags_concentrated_book_across_multiple_instruments():
    bucket_risk = {("JPY", "short"): 950.0, ("USD", "long"): 300.0,
                   ("EUR", "long"): 25.0, ("GBP", "long"): 25.0}
    bucket_insts = {("JPY", "short"): {"USD_JPY", "EUR_JPY", "GBP_JPY"},
                    ("USD", "long"): {"USD_JPY"},
                    ("EUR", "long"): {"EUR_JPY"}, ("GBP", "long"): {"GBP_JPY"}}
    warnings = bias_detector(bucket_risk, bucket_insts, share_threshold=0.90,
                             min_instruments=2)
    assert any("bias_one_sided_book" in w and "JPY_short" in w for w in warnings)


def test_bias_detector_silent_on_diversified_book():
    bucket_risk = {("JPY", "short"): 250.0, ("USD", "long"): 250.0,
                   ("EUR", "long"): 250.0, ("CHF", "short"): 250.0}
    bucket_insts = {("JPY", "short"): {"USD_JPY"}, ("USD", "long"): {"USD_JPY", "USD_CHF"},
                    ("EUR", "long"): {"EUR_USD"}, ("CHF", "short"): {"USD_CHF"}}
    warnings = bias_detector(bucket_risk, bucket_insts, share_threshold=0.90,
                             min_instruments=2)
    assert warnings == []


def test_bias_detector_ignores_single_instrument_bucket():
    """A lone instrument dominating its own bucket isn't 'disguised' concentration."""
    bucket_risk = {("JPY", "short"): 1000.0, ("USD", "long"): 1000.0}
    bucket_insts = {("JPY", "short"): {"USD_JPY"}, ("USD", "long"): {"USD_JPY"}}
    warnings = bias_detector(bucket_risk, bucket_insts, min_instruments=2)
    assert warnings == []


# --------------------------------------------------------------------- #
# Rule 5 — winner management: breakeven then trail, never loosen        #
# --------------------------------------------------------------------- #
def test_winner_stop_moves_to_breakeven_after_1r():
    entry, r = 162.656, 1.15266
    sl = evaluate_winner_stop(entry_price=entry, r_distance=r,
                              current_price=entry + 1.05 * r, current_sl=161.0)
    assert sl is not None
    assert abs(sl - entry) < 1e-9


def test_winner_stop_trails_after_1_5r_locking_1r_of_profit():
    entry, r = 162.656, 1.15266
    current = entry + 2.0 * r
    sl = evaluate_winner_stop(entry_price=entry, r_distance=r, current_price=current,
                              current_sl=entry)
    assert sl is not None
    assert abs(sl - (current - r)) < 1e-9
    assert sl > entry  # locks MORE than breakeven


def test_winner_stop_never_moves_backward():
    entry, r = 162.656, 1.15266
    # Already trailing at a high stop; a pullback in current price must NOT
    # loosen the stop even though r_gain still nominally qualifies.
    high_water_sl = entry + 3 * r
    sl = evaluate_winner_stop(entry_price=entry, r_distance=r,
                              current_price=entry + 1.6 * r, current_sl=high_water_sl)
    assert sl is None


def test_winner_stop_noop_before_1r():
    sl = evaluate_winner_stop(entry_price=162.656, r_distance=1.15266,
                              current_price=162.656 + 0.5 * 1.15266, current_sl=161.0)
    assert sl is None


def test_winner_stop_refuses_on_bad_r_distance():
    assert evaluate_winner_stop(entry_price=162.656, r_distance=0.0,
                                current_price=170.0, current_sl=None) is None
    assert evaluate_winner_stop(entry_price=0.0, r_distance=1.0,
                                current_price=170.0, current_sl=None) is None


# --------------------------------------------------------------------- #
# Persisted risk state (rule 5 bootstrap) — real disk via tmp_path      #
# --------------------------------------------------------------------- #
def test_risk_state_round_trips_atomically(tmp_path):
    path = tmp_path / "risk_state.json"
    assert load_risk_state(path) == {}
    state = {"1665": {"instrument": "USD_JPY", "entry_price": 162.737,
                      "r_distance": 1.15266, "stage": "initial"}}
    save_risk_state(path, state)
    assert load_risk_state(path) == state
    # tmp file cleaned up, no leftover .tmp siblings
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_risk_state_load_survives_corrupt_json(tmp_path):
    path = tmp_path / "risk_state.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_risk_state(path) == {}


# --------------------------------------------------------------------- #
# Rule 7 (2026-07-08) — cost-awareness: ADDITIVE, fail-closed on missing #
# execution-cost data (src.data.execution_cost_model wiring)            #
# --------------------------------------------------------------------- #
def test_cost_aware_gate_refuses_when_estimate_missing():
    """No fabrication: an instrument with no CostEstimate at all (never
    looked up / never in the traded universe) must REFUSE, exactly like
    risk_normalized_units refuses on a missing ATR."""
    allow, reason = cost_aware_gate(
        instrument="EUR_USD", r_distance_quote=0.004, cost_estimate=None)
    assert allow is False
    assert reason == "cost_unknown_fail_closed"


def test_cost_aware_gate_refuses_on_insufficient_data_source():
    est = _cost_estimate(source="insufficient_data", spread_raw=None, slippage_raw=None)
    allow, reason = cost_aware_gate(
        instrument="EUR_USD", r_distance_quote=0.004, cost_estimate=est)
    assert allow is False
    assert reason == "cost_unknown_fail_closed"


def test_cost_aware_gate_refuses_on_partial_data_spread_known_slippage_unknown():
    """Ticks cleared the sample bar (spread known) but fills didn't (no
    slippage signal yet) — never assume the missing half is free."""
    est = _cost_estimate(source="ticks", spread_raw=0.0002, slippage_raw=None)
    allow, reason = cost_aware_gate(
        instrument="EUR_USD", r_distance_quote=0.004, cost_estimate=est)
    assert allow is False
    assert reason == "cost_partial_data_fail_closed"


def test_cost_aware_gate_refuses_on_non_finite_cost_fields():
    est = _cost_estimate(spread_raw=float("nan"), slippage_raw=0.0)
    allow, reason = cost_aware_gate(
        instrument="EUR_USD", r_distance_quote=0.004, cost_estimate=est)
    assert allow is False
    assert reason == "cost_non_finite_fail_closed"


def test_cost_aware_gate_allows_cheap_cost_relative_to_r():
    """2-pip spread, zero slippage, against a realistic 40-pip (0.004) stop:
    cost_r_fraction = 0.0002/0.004 = 0.05, well under the 0.30 default cap."""
    est = _cost_estimate(spread_raw=0.0002, slippage_raw=0.0)
    allow, reason = cost_aware_gate(
        instrument="EUR_USD", r_distance_quote=0.004, cost_estimate=est)
    assert allow is True
    assert reason == "cost_aware_ok"


def test_cost_aware_gate_blocks_when_cost_eats_the_edge():
    """A wide/illiquid spread against a tight stop eats too much of the R
    budget — must block even though the estimate itself is real, not missing."""
    est = _cost_estimate(spread_raw=0.0015, slippage_raw=0.0005)  # round-trip 0.0025
    allow, reason = cost_aware_gate(
        instrument="EUR_USD", r_distance_quote=0.004,  # 0.0025/0.004 = 0.625 > 0.30
        cost_estimate=est)
    assert allow is False
    assert reason.startswith("cost_exceeds_edge:")


def test_cost_aware_gate_refuses_on_missing_r_distance():
    est = _cost_estimate()
    assert cost_aware_gate(instrument="EUR_USD", r_distance_quote=0.0,
                           cost_estimate=est)[0] is False
    assert cost_aware_gate(instrument="EUR_USD", r_distance_quote=None,
                           cost_estimate=est)[0] is False


def test_cost_aware_gate_respects_custom_max_cost_r_fraction():
    """Same numbers as the 'blocks' case above, but a looser caller-supplied
    cap (documents the knob is real, not hardcoded) — never changes the
    DEFAULT, which stays conservative."""
    est = _cost_estimate(spread_raw=0.0015, slippage_raw=0.0005)  # cost_r = 0.625
    allow, _ = cost_aware_gate(instrument="EUR_USD", r_distance_quote=0.004,
                               cost_estimate=est, max_cost_r_fraction=0.70)
    assert allow is True
    assert DEFAULT_MAX_COST_R_FRACTION == 0.30  # documents the conservative default is untouched


# --------------------------------------------------------------------- #
# Config defaults / clamping sanity                                     #
# --------------------------------------------------------------------- #
def test_risk_gate_config_defaults_are_within_operator_specified_ranges():
    cfg = RiskGateConfig()
    assert 0.005 <= cfg.risk_pct <= 0.03
    assert cfg.max_bucket_risk_r > 0
    assert cfg.pyramid_cum_risk_r_cap <= 1.0
    assert cfg.breakeven_trigger_r == 1.0
    assert 0.0 < cfg.max_cost_r_fraction <= 1.0
