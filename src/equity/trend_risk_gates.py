"""Trade-path RISK GATE for the OANDA trend lane (PRACTICE only).

Operator-directed rebuild (2026-07-01) after a blotter review found the live
book was 2 correlated macro bets (yen-cross longs = short-JPY; USD_CAD +
USD_CHF = long-USD) disguised as 7 trades, with randomly-varying sizing (the
old leverage-based ``target_units`` re-sized the WHOLE target every cycle off
``gross_leverage x NAV``; since OANDA v20 opens a NEW trade ID for every
market order on an instrument instead of merging into the existing one, every
re-size created a second ticket instead of adjusting the first), a losing
average-down add, and no winner protection.

All functions here are PURE (no network I/O) so they are no-mock testable.
The only I/O is the small persisted per-trade state file for winner
management (atomic tmp+rename, matches project JSON-safety convention).

Composition order in the live cycle (tightest constraint wins, each one only
ever REDUCES risk relative to the previous stage — never loosens):
    risk_normalized_units()  (bottom-up, per-instrument, ATR-aware)
      -> leverage_cap_scale()  (top-down ceiling on total exposure)
      -> margin_scale()  (existing margin/liquidation rail, unchanged)
      -> one_position_gate() / pyramid_gate()  (per-order block, rules 1+6)
      -> bucket_cap_gate()  (per-order block, rule 2)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.equity.fx_rates import base_to_home_rate

logger = logging.getLogger(__name__)

# --- Rule 3: risk-normalized sizing -----------------------------------------
DEFAULT_RISK_PCT = 0.01     # 1% of NAV risked per new position
MIN_RISK_PCT = 0.005
MAX_RISK_PCT = 0.03

# --- Rule 2: correlation-bucket caps -----------------------------------------
DEFAULT_MAX_BUCKET_RISK_R = 2.0   # a currency-direction bucket may hold at most 2R

# --- Rule 4: bias detector ----------------------------------------------------
DEFAULT_BIAS_SHARE_THRESHOLD = 0.90
DEFAULT_BIAS_MIN_INSTRUMENTS = 2

# --- Rule 5: winner management -----------------------------------------------
DEFAULT_BREAKEVEN_TRIGGER_R = 1.0
DEFAULT_TRAIL_START_R = 1.5

# --- Rule 6: pyramid gating ---------------------------------------------------
DEFAULT_PYRAMID_MAX_LAYERS = 3
DEFAULT_PYRAMID_CUM_RISK_R_CAP = 1.0


@dataclass(frozen=True)
class RiskGateConfig:
    """All six rules' knobs in one place. Every field independently overridable
    via CLI/env in scripts/run_oanda_trend.py (mirrors the existing gross_leverage
    dial pattern)."""
    risk_pct: float = DEFAULT_RISK_PCT
    max_bucket_risk_r: float = DEFAULT_MAX_BUCKET_RISK_R
    bias_share_threshold: float = DEFAULT_BIAS_SHARE_THRESHOLD
    bias_min_instruments: int = DEFAULT_BIAS_MIN_INSTRUMENTS
    breakeven_trigger_r: float = DEFAULT_BREAKEVEN_TRIGGER_R
    trail_start_r: float = DEFAULT_TRAIL_START_R
    pyramid_max_layers: int = DEFAULT_PYRAMID_MAX_LAYERS
    pyramid_cum_risk_r_cap: float = DEFAULT_PYRAMID_CUM_RISK_R_CAP
    home_ccy: str = "USD"


def clamp_risk_pct(value: float) -> float:
    """Clamp risk_pct to [MIN_RISK_PCT, MAX_RISK_PCT]; non-finite -> default."""
    import math
    try:
        v = float(value)
    except (ValueError, TypeError):
        return DEFAULT_RISK_PCT
    if not math.isfinite(v) or v <= 0:
        return DEFAULT_RISK_PCT
    return max(MIN_RISK_PCT, min(v, MAX_RISK_PCT))


# --------------------------------------------------------------------- #
# Currency-conversion helper (rule 3): risk lives in the QUOTE currency #
# --------------------------------------------------------------------- #
def quote_to_home_rate(instrument: str, last_prices: Dict[str, float], *,
                       home_ccy: str = "USD") -> Optional[float]:
    """Home-ccy value of 1 unit of the instrument's QUOTE currency.

    ATR / stop distances are priced in the QUOTE currency (mid.c/h/l are quote-
    per-base). Dollar risk = units * stop_distance_quote * quote_to_home_rate.
    This is DIFFERENT from ``base_to_home_rate`` (which prices NOTIONAL, i.e.
    the base-currency leg) — reuses it against a synthetic ``QUOTE_HOME``
    instrument string since that helper only reads the piece before the first
    underscore.
    """
    parts = str(instrument).split("_")
    if len(parts) != 2:
        return None
    quote = parts[1]
    if quote == home_ccy:
        return 1.0
    return base_to_home_rate(f"{quote}_{home_ccy}", last_prices, home_ccy=home_ccy)


def risk_normalized_units(*, instrument: str, r_distance_quote: Optional[float],
                          nav: float, risk_pct: float, last_prices: Dict[str, float],
                          home_ccy: str = "USD") -> int:
    """Rule 3 — units sized so that a full stop-out loses exactly ``nav*risk_pct``.

    units = (nav * risk_pct) / (r_distance_quote * quote_to_home_rate).
    Refuses (returns 0) rather than fabricating a size when the stop distance
    or the quote->home rate is unavailable — same no-fabrication discipline as
    the rest of this module.
    """
    if not (r_distance_quote and r_distance_quote > 0):
        return 0
    if risk_pct <= 0 or nav <= 0:
        return 0
    q2h = quote_to_home_rate(instrument, last_prices, home_ccy=home_ccy)
    if q2h is None or q2h <= 0:
        logger.warning("no quote->%s rate for %s — sizing 0 (refuse, no fabrication)",
                       home_ccy, instrument)
        return 0
    risk_home = float(nav) * float(risk_pct)
    risk_per_unit_home = float(r_distance_quote) * float(q2h)
    if risk_per_unit_home <= 0:
        return 0
    return max(int(risk_home / risk_per_unit_home), 1)


def leverage_cap_scale(want_units: Dict[str, int], last_px: Dict[str, float], nav: float, *,
                       gross_leverage: float, home_ccy: str = "USD"):
    """Leverage dial demoted to a CEILING (rule 3 reconciliation): scale the whole
    risk-sized book down (never up) so total gross notional <= gross_leverage*NAV.
    Mirrors ``margin_scale``'s shape exactly. Returns ``(scaled_units, factor)``."""
    gross_notional = 0.0
    for inst, u in want_units.items():
        rate = base_to_home_rate(inst, last_px, home_ccy=home_ccy)
        if rate and rate > 0:
            gross_notional += abs(int(u)) * float(rate)
    cap = float(gross_leverage) * float(nav)
    if gross_notional <= cap or gross_notional <= 0:
        return dict(want_units), 1.0
    factor = cap / gross_notional
    return {i: int(int(u) * factor) for i, u in want_units.items()}, factor


# --------------------------------------------------------------------- #
# Rule 2: correlation buckets — shared-currency exposure, not a fixed   #
# instrument list, so it generalizes to any pair sharing a currency leg #
# --------------------------------------------------------------------- #
def currency_legs(instrument: str) -> Optional[Tuple[str, str]]:
    parts = str(instrument).split("_")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def trade_risk_home(trade: dict, r_distance_quote: float, quote_to_home: float) -> float:
    try:
        units = abs(float(trade.get("currentUnits") or trade.get("initialUnits") or 0))
    except (ValueError, TypeError):
        units = 0.0
    return units * float(r_distance_quote) * float(quote_to_home)


def compute_bucket_risk(open_trades: List[dict], r_distance_by_trade_id: Dict[str, float],
                        last_prices: Dict[str, float], *, home_ccy: str = "USD",
                        r_distance_fallback_by_instrument: Optional[Dict[str, float]] = None
                        ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], set]]:
    """Sum each open (long) trade's dollar-risk into its (base,'long') and
    (quote,'short') buckets. Returns (bucket_risk_home, bucket_instruments) —
    the latter used by the bias detector to require >=2 distinct instruments
    before flagging (a single-instrument book isn't a "disguised" concentration).

    ``r_distance_by_trade_id`` is keyed by OANDA trade ID and should come from
    the SAME entry-anchored ``risk_state`` winner-management uses (rule 5) —
    NOT the current ATR. Using current ATR here would under/over-state a
    trade's real dollar-risk whenever volatility has moved since entry (e.g.
    an ATR contraction makes the CURRENT stop distance look smaller than the
    risk actually taken, silently letting more correlated risk through the
    bucket cap than intended). ``r_distance_fallback_by_instrument`` (current
    ATR) is used ONLY for a trade not yet present in persisted state (the
    first cycle it's ever seen, before winner-management has bootstrapped it)."""
    bucket_risk: Dict[Tuple[str, str], float] = {}
    bucket_insts: Dict[Tuple[str, str], set] = {}
    fallback = r_distance_fallback_by_instrument or {}
    for trade in open_trades or []:
        inst = trade.get("instrument")
        trade_id = str(trade.get("id") or "")
        legs = currency_legs(inst) if inst else None
        if not legs:
            continue
        r_dist = r_distance_by_trade_id.get(trade_id) or fallback.get(inst)
        if not r_dist or r_dist <= 0:
            continue
        q2h = quote_to_home_rate(inst, last_prices, home_ccy=home_ccy)
        if q2h is None or q2h <= 0:
            continue
        risk_home = trade_risk_home(trade, r_dist, q2h)
        base, quote = legs
        for key in ((base, "long"), (quote, "short")):
            bucket_risk[key] = bucket_risk.get(key, 0.0) + risk_home
            bucket_insts.setdefault(key, set()).add(inst)
    return bucket_risk, bucket_insts


def accumulate_bucket_risk(bucket_risk_home: Dict[Tuple[str, str], float], instrument: str,
                           risk_home: float) -> None:
    """Rule 2 fix (verifier, 2026-07-01) — SAME-CYCLE bucket bypass close: mutates
    ``bucket_risk_home`` IN PLACE immediately after an order is approved, so the
    Nth correlated instrument flipping on in the SAME cycle is checked against a
    running total that already includes the (N-1) earlier approvals THIS cycle —
    not just the pre-cycle snapshot. Without this, 3 yen-cross longs newly
    flipping on together could each individually clear a 2R cap (each sees 0
    existing risk) while jointly landing at ~3R."""
    legs = currency_legs(instrument)
    if not legs:
        return
    base, quote = legs
    for key in ((base, "long"), (quote, "short")):
        bucket_risk_home[key] = bucket_risk_home.get(key, 0.0) + float(risk_home)


def bucket_cap_gate(*, instrument: str, candidate_units: int, r_distance_quote: float,
                    last_prices: Dict[str, float], bucket_risk_home: Dict[Tuple[str, str], float],
                    risk_unit_home: float, max_bucket_risk_r: float, home_ccy: str = "USD"
                    ) -> Tuple[bool, str]:
    """Rule 2 — block a candidate order if it would push EITHER of its two
    currency buckets over ``max_bucket_risk_r`` R-units of risk."""
    legs = currency_legs(instrument)
    if not legs:
        return False, "bucket_cap_unresolvable_instrument"
    q2h = quote_to_home_rate(instrument, last_prices, home_ccy=home_ccy)
    if q2h is None or q2h <= 0:
        return False, "bucket_cap_no_quote_rate"
    candidate_risk = abs(candidate_units) * float(r_distance_quote) * float(q2h)
    cap = float(max_bucket_risk_r) * float(risk_unit_home)
    base, quote = legs
    for key in ((base, "long"), (quote, "short")):
        used = bucket_risk_home.get(key, 0.0)
        if used + candidate_risk > cap + 1e-9:
            return False, f"bucket_cap_exceeded:{key[0]}_{key[1]}"
    return True, "bucket_cap_ok"


def buckets_over_cap(bucket_risk_home: Dict[Tuple[str, str], float], *,
                     risk_unit_home: float, max_bucket_risk_r: float) -> List[str]:
    """Honest end-of-cycle check (replaces a prior comment that falsely claimed
    an over-cap bucket "self-corrects next cycle" — there is no trim/reduce path,
    so an over-cap book PERSISTS until enough trades close naturally). Returns
    one warning string per bucket still over ``max_bucket_risk_r`` after this
    cycle's approvals, for the caller to log (LOG-ONLY — no automatic trim is
    implemented; adding one would itself be a new order-placing code path that
    needs its own dedicated review, out of scope for this fix)."""
    cap = float(max_bucket_risk_r) * float(risk_unit_home)
    if cap <= 0:
        return []
    return [f"bucket_over_cap: {key[0]}_{key[1]} risk={risk:.2f} cap={cap:.2f} "
            f"({risk / cap:.0%} of cap) — NO automatic trim exists; will persist "
            f"until a trade closes naturally"
            for key, risk in bucket_risk_home.items() if risk > cap + 1e-9]


def bias_detector(bucket_risk_home: Dict[Tuple[str, str], float],
                  bucket_insts: Dict[Tuple[str, str], set], *,
                  share_threshold: float = DEFAULT_BIAS_SHARE_THRESHOLD,
                  min_instruments: int = DEFAULT_BIAS_MIN_INSTRUMENTS) -> List[str]:
    """Rule 4 — LOG-ONLY diagnostic (rule 2 already enforces the hard cap).

    Returns human-readable warning strings for any bucket whose share of total
    per-trade risk (each trade counted ONCE, buckets double-count by design)
    exceeds ``share_threshold`` across >= ``min_instruments`` distinct tickets —
    i.e. the book is effectively one macro bet wearing multiple instrument
    labels. Caller logs each returned message; never blocks."""
    if not bucket_risk_home:
        return []
    # Total distinct-trade risk = each trade's risk counted once; approximate
    # via the max single-currency bucket's own "long" + "short" split is wrong
    # (double counts), so recompute total from the union of contributing
    # instruments' bucket_risk contributions divided by 2 (every trade adds to
    # exactly 2 buckets with the SAME risk_home amount).
    total_risk_double_counted = sum(bucket_risk_home.values())
    total_risk = total_risk_double_counted / 2.0
    if total_risk <= 0:
        return []
    warnings: List[str] = []
    for key, risk in bucket_risk_home.items():
        insts = bucket_insts.get(key, set())
        if len(insts) < min_instruments:
            continue
        share = risk / total_risk
        if share >= share_threshold:
            warnings.append(
                f"bias_one_sided_book: bucket={key[0]}_{key[1]} share={share:.0%} "
                f"of total open risk across {len(insts)} instruments {sorted(insts)}")
    return warnings


# --------------------------------------------------------------------- #
# Rules 1 + 6: one-position-per-instrument, with pyramid as the only    #
# authorized exception (never average down, never grow past 1R total)  #
# --------------------------------------------------------------------- #
def pyramid_gate(*, existing_trades_for_instrument: List[dict], candidate_units: int,
                 r_distance_quote: float, risk_unit_home: float, quote_to_home: float,
                 max_layers: int = DEFAULT_PYRAMID_MAX_LAYERS,
                 cum_risk_r_cap: float = DEFAULT_PYRAMID_CUM_RISK_R_CAP
                 ) -> Tuple[bool, str]:
    """Rule 6 — an add is authorized ONLY if every existing layer is green,
    the new layer is smaller than the most-recently-opened layer, and the
    combined risk (all layers + candidate) stays within ``cum_risk_r_cap`` R."""
    if not existing_trades_for_instrument:
        return True, "pyramid_not_applicable_no_existing_trade"
    if len(existing_trades_for_instrument) >= max_layers:
        return False, "pyramid_max_layers_reached"
    for t in existing_trades_for_instrument:
        try:
            upl = float(t.get("unrealizedPL", 0) or 0)
        except (ValueError, TypeError):
            upl = 0.0
        if upl <= 0:
            return False, "pyramid_blocked_red_trade_no_averaging_down"
    most_recent = max(existing_trades_for_instrument,
                      key=lambda t: str(t.get("openTime", "")))
    try:
        most_recent_units = abs(float(most_recent.get("currentUnits") or 0))
    except (ValueError, TypeError):
        most_recent_units = 0.0
    if abs(candidate_units) >= most_recent_units:
        return False, "pyramid_add_not_smaller_than_last_layer"
    cum_existing = sum(trade_risk_home(t, r_distance_quote, quote_to_home)
                       for t in existing_trades_for_instrument)
    candidate_risk = abs(candidate_units) * float(r_distance_quote) * float(quote_to_home)
    if cum_existing + candidate_risk > float(cum_risk_r_cap) * float(risk_unit_home) + 1e-9:
        return False, "pyramid_cum_risk_exceeds_cap"
    return True, "pyramid_add_ok"


def one_position_gate(*, existing_trades_for_instrument: List[dict], candidate_units: int,
                      r_distance_quote: float, risk_unit_home: float, quote_to_home: float,
                      pyramid_max_layers: int = DEFAULT_PYRAMID_MAX_LAYERS,
                      pyramid_cum_risk_r_cap: float = DEFAULT_PYRAMID_CUM_RISK_R_CAP
                      ) -> Tuple[bool, str]:
    """Rule 1 — hard cap: no second open ticket on the same instrument, unless
    ``pyramid_gate`` (rule 6) explicitly authorizes it. This is only evaluated
    for size-INCREASE candidates; the live cycle must always allow decreases/
    closes through without calling this gate (risk-reducing orders never
    blocked)."""
    if not existing_trades_for_instrument:
        return True, "new_entry_no_existing_trade"
    return pyramid_gate(
        existing_trades_for_instrument=existing_trades_for_instrument,
        candidate_units=candidate_units, r_distance_quote=r_distance_quote,
        risk_unit_home=risk_unit_home, quote_to_home=quote_to_home,
        max_layers=pyramid_max_layers, cum_risk_r_cap=pyramid_cum_risk_r_cap)


# --------------------------------------------------------------------- #
# Rule 5: winner management — breakeven then trail, never loosen        #
# --------------------------------------------------------------------- #
def evaluate_winner_stop(*, entry_price: float, r_distance: float, current_price: float,
                         current_sl: Optional[float],
                         breakeven_trigger_r: float = DEFAULT_BREAKEVEN_TRIGGER_R,
                         trail_start_r: float = DEFAULT_TRAIL_START_R) -> Optional[float]:
    """Long-only. Returns a NEW stop price if the position has earned protection
    it doesn't already have, else None (no-op). Never returns a level worse
    than the current stop (monotonic non-decreasing) and never worse than
    breakeven once breakeven has triggered — total risk from here on is capped
    at ~0, well inside the original 1R budget."""
    if not (r_distance and r_distance > 0 and entry_price and entry_price > 0):
        return None
    r_gain = (float(current_price) - float(entry_price)) / float(r_distance)
    candidate: Optional[float] = None
    if r_gain >= trail_start_r:
        candidate = float(current_price) - float(r_distance)
    elif r_gain >= breakeven_trigger_r:
        candidate = float(entry_price)
    if candidate is None:
        return None
    candidate = max(candidate, float(entry_price))  # never worse than breakeven post-trigger
    if current_sl is not None and candidate <= float(current_sl) + 1e-9:
        return None  # not an improvement
    return candidate


# --------------------------------------------------------------------- #
# Persisted per-trade state (rule 5 bootstrap: r_distance fixed at      #
# entry so a later ATR change can't retroactively alter a trade's R)    #
# --------------------------------------------------------------------- #
def load_risk_state(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def save_risk_state(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, str(path))
