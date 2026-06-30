"""Reproduce-then-resolve: ATR bracket sizing must be anchored to the FILL,
not the last complete candle close, so R:R is a constant 2.0 across JPY and
non-JPY instruments and never breaches the 1.2 invariant.

ROOT CAUSE (2026-06-30): ``run_oanda_trend_cycle`` placed brackets as ABSOLUTE
prices computed from ``last_px`` (the last *complete daily candle close*), while
the market order fills at the live price. The drift between the daily close and
the fill (≈ +68 pips on USD_JPY, +81 on GBP_JPY in the live book) skewed the
realized SL/TP distances per-instrument and dropped USD_JPY to R:R 0.89 < 1.2.

FIX: size brackets as DISTANCES (sl_mult*ATR, tp_mult*ATR) attached via OANDA's
``*OnFill.distance`` so they anchor to the actual fill — exact 2.0 R:R for any
instrument, immune to anchor drift.

NO MOCKS per CLAUDE.md No-Mock Rule. These exercise pure sizing/body-building
functions with real values; the live-order proof is a separate manual step.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.equity.oanda_trend import (  # noqa: E402
    DEFAULT_ATR_SL_MULT,
    DEFAULT_ATR_TP_MULT,
    bracket_distances,
    repair_bracket_prices,
)

# Real ATRs (price units) reverse-engineered from the live book 2026-06-30.
ATR_JPY = 0.57633      # USD_JPY daily ATR
ATR_GBPJPY = 1.17867   # GBP_JPY daily ATR
ATR_NONJPY = 0.00585   # USD_CAD daily ATR
ATR_CHF = 0.00496      # USD_CHF daily ATR


def _rr(sl_dist: float, tp_dist: float) -> float:
    return tp_dist / sl_dist


# --------------------------------------------------------------------- #
# bracket_distances — constant R:R across instrument classes            #
# --------------------------------------------------------------------- #
def test_distances_are_exactly_mult_times_atr():
    for atr in (ATR_JPY, ATR_GBPJPY, ATR_NONJPY, ATR_CHF):
        sl, tp = bracket_distances(atr)
        assert abs(sl - DEFAULT_ATR_SL_MULT * atr) < 1e-12
        assert abs(tp - DEFAULT_ATR_TP_MULT * atr) < 1e-12


def test_rr_is_constant_2_across_jpy_and_nonjpy():
    rrs = []
    for atr in (ATR_JPY, ATR_GBPJPY, ATR_NONJPY, ATR_CHF):
        sl, tp = bracket_distances(atr)
        rr = _rr(sl, tp)
        rrs.append(rr)
        assert rr >= 1.2, f"R:R {rr} breaches the 1.2 invariant for atr={atr}"
    # Every instrument class gets the SAME ratio (4:2 = 2.0) — no per-instrument skew.
    assert max(rrs) - min(rrs) < 1e-9
    assert abs(rrs[0] - 2.0) < 1e-9


def test_distances_immune_to_anchor_drift():
    """The live bug: absolute brackets anchored to a stale close != fill skew R:R.

    Distance-based brackets give R:R 2.0 from ANY fill price, so the +68 pip
    USD_JPY drift that produced 0.89 can no longer occur.
    """
    atr = ATR_JPY
    entry = 162.625          # actual fill
    stale_anchor = 161.944   # last complete daily close (what the bug used)
    # OLD bug: absolute SL/TP from the stale anchor, measured from the real entry.
    old_sl_px = stale_anchor - DEFAULT_ATR_SL_MULT * atr
    old_tp_px = stale_anchor + DEFAULT_ATR_TP_MULT * atr
    old_rr_from_entry = (old_tp_px - entry) / (entry - old_sl_px)
    assert old_rr_from_entry < 1.2  # reproduces the breach

    # NEW: distance-based — R:R is 2.0 regardless of the drift.
    sl_dist, tp_dist = bracket_distances(atr)
    assert abs(_rr(sl_dist, tp_dist) - 2.0) < 1e-9


def test_disable_legs_returns_none():
    assert bracket_distances(ATR_JPY, enable_sl=False)[0] is None
    assert bracket_distances(ATR_JPY, enable_tp=False)[1] is None


def test_no_atr_returns_none():
    assert bracket_distances(0.0) == (None, None)
    assert bracket_distances(None) == (None, None)


# --------------------------------------------------------------------- #
# repair path — anchor to the trade's ENTRY, not the moving current     #
# --------------------------------------------------------------------- #
def test_repair_anchors_to_entry_giving_2_to_1():
    """An open trade missing a bracket should be repaired relative to its ENTRY."""
    entry = 162.625
    current = 162.700      # price has drifted up since entry
    atr = ATR_JPY
    sl, tp = repair_bracket_prices(entry=entry, current=current, atr=atr)
    # Anchored to entry -> exact 2:1 from entry, both on the correct side of market.
    assert abs(sl - (entry - DEFAULT_ATR_SL_MULT * atr)) < 1e-9
    assert abs(tp - (entry + DEFAULT_ATR_TP_MULT * atr)) < 1e-9
    assert sl < current < tp
    rr = (tp - entry) / (entry - sl)
    assert abs(rr - 2.0) < 1e-9


def test_repair_clamps_to_keep_orders_on_correct_side():
    """If price ran past the entry-anchored level, clamp so SL<current<TP."""
    entry = 162.625
    atr = ATR_JPY
    # Price collapsed below the entry-anchored stop -> SL must sit below current.
    current = entry - DEFAULT_ATR_SL_MULT * atr - 0.10
    sl, tp = repair_bracket_prices(entry=entry, current=current, atr=atr)
    assert sl < current < tp


# --------------------------------------------------------------------- #
# order body — *OnFill.distance emitted with correct per-instrument scale #
# --------------------------------------------------------------------- #
from src.utils.oanda_practice import build_market_order_body  # noqa: E402


def test_order_body_emits_distance_brackets_jpy():
    """JPY pair: distance formatted to 3 decimals under *OnFill.distance."""
    sl_dist, tp_dist = bracket_distances(ATR_JPY)  # 1.15266, 2.30532
    body = build_market_order_body(
        instrument="USD_JPY", units=1000, client_order_id="t", client_tag="t",
        stop_loss_distance=sl_dist, take_profit_distance=tp_dist)
    o = body["order"]
    assert o["stopLossOnFill"] == {"distance": "1.153"}
    assert o["takeProfitOnFill"] == {"distance": "2.305"}
    # R:R preserved in the emitted strings.
    assert float(o["takeProfitOnFill"]["distance"]) / float(o["stopLossOnFill"]["distance"]) >= 1.2


def test_order_body_emits_distance_brackets_nonjpy():
    """Non-JPY pair: distance formatted to 5 decimals."""
    sl_dist, tp_dist = bracket_distances(ATR_NONJPY)  # 0.0117, 0.0234
    body = build_market_order_body(
        instrument="USD_CAD", units=1000, client_order_id="t", client_tag="t",
        stop_loss_distance=sl_dist, take_profit_distance=tp_dist)
    o = body["order"]
    assert o["stopLossOnFill"] == {"distance": "0.01170"}
    assert o["takeProfitOnFill"] == {"distance": "0.02340"}


def test_order_body_distance_takes_precedence_over_price():
    """When both supplied for a leg, distance (fill-anchored) wins over absolute price."""
    body = build_market_order_body(
        instrument="USD_JPY", units=1000, client_order_id="t", client_tag="t",
        stop_loss_price=160.0, stop_loss_distance=1.153)
    assert body["order"]["stopLossOnFill"] == {"distance": "1.153"}


def test_order_body_absolute_price_still_supported():
    """Back-compat: absolute price path unchanged when no distance given."""
    body = build_market_order_body(
        instrument="USD_JPY", units=1000, client_order_id="t", client_tag="t",
        stop_loss_price=160.791, take_profit_price=164.249)
    assert body["order"]["stopLossOnFill"] == {"price": "160.791"}
    assert body["order"]["takeProfitOnFill"] == {"price": "164.249"}
