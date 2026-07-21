"""OANDA practice TREND / managed-futures runner (NON-directional; PRACTICE only).

The strategy here is NOT the retired directional FX transformer (L-016: directional
prediction hit the ~52% ceiling). It is the validated **trend / managed-futures**
logic — price vs a long moving average, long-or-flat, strictly ``shift(1)`` causal —
i.e. the same rule as ``trend_sleeve.trend_sleeve_weights`` that was our one
validated lever (a robust drawdown-reducer). Applied here to OANDA's tradable
instruments (FX majors + whatever CFDs the practice account enables). It predicts
no direction; it follows trend and goes to cash in downtrends.

Data + execution lanes (all OANDA v20 **practice**, ``api-fxpractice``):
  candles  -> trend signal (this module)
  account summary + open positions + transactions -> NAV / P&L / audit
  market-order endpoint -> demo execution (``create_market_order``)

HARD LINE: PRACTICE ONLY. The only client used is ``OandaPracticeClient``, whose
base URL is hard-pinned to ``api-fxpractice.oanda.com/v3`` (no live URL constant
exists in it). This module asserts ``config.oanda_environment == "practice"`` and
re-checks the global halt every cycle (a halted ``.claude/state.json`` REFUSES).

FUTURE (flagged, NOT built): the v20 order-book / position-book SENTIMENT
(``get_order_book`` / ``get_position_book``) is a promising NON-directional
contrarian signal (retail crowding -> fade). Reserve for a pre-registered test.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from src.equity import fx_rates
from src.equity.decision_gate import _lane_halted
from src.learning.decision_record import Disposition
from src.equity.trend_risk_gates import (
    DEFAULT_BIAS_MIN_INSTRUMENTS,
    DEFAULT_BIAS_SHARE_THRESHOLD,
    DEFAULT_BREAKEVEN_TRIGGER_R,
    DEFAULT_MAX_BUCKET_RISK_R,
    DEFAULT_MAX_COST_R_FRACTION,
    DEFAULT_PYRAMID_CUM_RISK_R_CAP,
    DEFAULT_PYRAMID_MAX_LAYERS,
    DEFAULT_RISK_PCT,
    DEFAULT_TRAIL_START_R,
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
    quote_to_home_rate,
    risk_normalized_units,
    save_risk_state,
    trade_risk_home,
)
from src.data.execution_cost_model import build_cost_model
from src.hedge.exposure_tags import CURRENCY_BUCKET_MAP_PATH, load_bucket_map
from src.hedge.portfolio_exposure import ExposurePosition, build_exposure_report

if TYPE_CHECKING:  # pragma: no cover
    from src.data.execution_cost_model import CostEstimate
    from src.scanner.config import ScannerConfig
    from src.utils.oanda_practice import OandaPracticeClient

logger = logging.getLogger(__name__)

DEFAULT_SMA = 100        # ~5-month trend on daily candles (managed-futures canonical)
DEFAULT_GRANULARITY = "D"
DEFAULT_CANDLE_COUNT = 300   # <= OANDA 5000 max; enough history for the SMA + warmup
# Total demo exposure = gross_leverage x NAV, spread across the "on" instruments.
# Operator dial via OANDA_GROSS_LEVERAGE / --gross-leverage (2026-06-25: 0.5 was
# too timid — ~1.9% margin used). Default 3x; capped to keep margin < ~80% of NAV
# (FX majors ~3-5% margin => ~20x is the margin-call wall, so 15x hard cap).
DEFAULT_GROSS_LEVERAGE = 3.0
MAX_GROSS_LEVERAGE = 15.0


def clamp_leverage(value: float) -> float:
    """Clamp gross leverage to [0, MAX_GROSS_LEVERAGE]; warn if it was capped.

    Non-finite (nan/inf) inputs fall back to the default (verifier hardening: a
    nan would otherwise slip both comparisons and reach the sizer)."""
    import math
    try:
        v = float(value)
    except (ValueError, TypeError):
        return DEFAULT_GROSS_LEVERAGE
    if not math.isfinite(v) or v < 0:
        return DEFAULT_GROSS_LEVERAGE
    if v > MAX_GROSS_LEVERAGE:
        logger.warning("gross_leverage %.1f exceeds cap %.1f — clamping (margin-call guard)",
                       v, MAX_GROSS_LEVERAGE)
        return MAX_GROSS_LEVERAGE
    return v


@dataclass(frozen=True)
class OandaTrendResult:
    ran: bool
    reason: str               # "halted" / "no_data" / "no_token" / "executed" / "dry_run"
    targets: Dict[str, float]  # instrument -> target weight (0 = flat/cash)
    orders_placed: int


def candles_to_close_panel(candles_by_instrument: Dict[str, dict]) -> pd.DataFrame:
    """OANDA candle responses -> date x instrument close panel (COMPLETE candles only).

    Each value is a ``get_candles`` response: ``{"candles": [{"time", "complete",
    "mid": {"c": ...}}, ...]}``. Incomplete (forming) candles are dropped so the
    signal only ever sees closed bars — part of the no-lookahead discipline.
    """
    cols: Dict[str, pd.Series] = {}
    for inst, resp in candles_by_instrument.items():
        rows = (resp or {}).get("candles", []) or []
        times, closes = [], []
        for c in rows:
            if not c.get("complete", False):
                continue
            mid = c.get("mid") or c.get("bid") or c.get("ask") or {}
            close = mid.get("c")
            t = c.get("time")
            if close is None or t is None:
                continue
            try:
                closes.append(float(close))
                times.append(pd.Timestamp(t))
            except (ValueError, TypeError):
                continue
        if times:
            s = pd.Series(closes, index=pd.DatetimeIndex(times))
            cols[inst] = s
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols).sort_index()
    panel.index = pd.to_datetime(panel.index, utc=True).normalize()
    return panel


def trend_targets(close_panel: pd.DataFrame, *, sma_window: int = DEFAULT_SMA) -> Dict[str, float]:
    """Latest long-or-flat target weights via the validated trend rule.

    Reuses ``trend_sleeve.trend_sleeve_weights`` (price[t-1] > SMA(<=t-1) -> on,
    equal-weight the on-set else cash; double ``shift(1)`` => strictly causal).
    Returns the most-recent row as {instrument: weight} (weights sum to <= 1).
    """
    from src.equity.trend_sleeve import trend_sleeve_weights

    if close_panel.empty:
        return {}
    w = trend_sleeve_weights(close_panel.ffill(), sma_window=sma_window, step=1)
    if w.empty:
        return {}
    last = w.iloc[-1]
    return {inst: float(v) for inst, v in last.items()}


# Re-exported for backward compat (existing tests/callers import this from
# oanda_trend) — actual implementation lives in fx_rates to avoid a circular
# import with trend_risk_gates (which also needs it).
base_to_home_rate = fx_rates.base_to_home_rate


def target_units(
    targets: Dict[str, float],
    nav: float,
    last_prices: Dict[str, float],
    *,
    gross_leverage: float = DEFAULT_GROSS_LEVERAGE,
    home_ccy: str = "USD",
) -> Dict[str, int]:
    """Convert target weights -> integer OANDA units with PER-BASE-CURRENCY sizing.

    Each held instrument gets EQUAL home-currency notional
    (``weight * NAV * gross_leverage``); units = that notional / the base->home
    rate, so JPY-quote and USD-base pairs are now consistently risk-scaled (no
    more ``/price`` skew). An instrument whose base->home rate cannot be derived is
    sized 0 (refuse, don't fabricate). Long-or-flat => units >= 0.
    """
    out: Dict[str, int] = {}
    leverage = float(gross_leverage)
    for inst, w in targets.items():
        if w <= 0 or leverage <= 0:
            out[inst] = 0
            continue
        rate = base_to_home_rate(inst, last_prices, home_ccy=home_ccy)
        if rate is None or rate <= 0:
            logger.warning("no base->%s rate for %s — sizing 0 (refuse, no fabrication)",
                           home_ccy, inst)
            out[inst] = 0
            continue
        notional_home = float(w) * float(nav) * leverage
        if notional_home <= 0:
            out[inst] = 0
            continue
        out[inst] = max(int(notional_home / float(rate)), 1)
    return out


DEFAULT_DD_HARD = 0.20   # auto-halt the lane if NAV draws down >= 20% from peak


def nav_drawdown_breached(nav: float, peak_path: Path, *, dd_hard: float = DEFAULT_DD_HARD) -> Optional[float]:
    """Track peak NAV on disk; return the drawdown if it breaches ``dd_hard`` (else None).

    Autonomous-safety rail (verifier rec): an unsupervised loop must stop adding risk
    if the demo account bleeds. Peak NAV is persisted atomically and ratchets up only.
    """
    import json
    peak = float(nav)
    try:
        prev = float(json.loads(peak_path.read_text()).get("peak_nav", nav))
        peak = max(peak, prev)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(peak_path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump({"peak_nav": peak}, fh)
    os.replace(tmp, peak_path)
    dd = (peak - float(nav)) / peak if peak > 0 else 0.0
    return dd if dd >= float(dd_hard) else None


def rebalance_delta(target: int, current: int, *, band_pct: float = 0.02, min_units: int = 1) -> int:
    """Order delta with a no-trade band — 0 unless |target-current| exceeds the band.

    Avoids churning tiny ±1-unit orders every cycle as NAV drifts (good-citizen):
    only rebalances when the gap exceeds ``band_pct`` of the position (a flat<->on
    flip is a full-position delta, always far above the band; sub-percent NAV drift
    is ignored). ``min_units`` floors the band so it is never below 1 unit.
    """
    delta = int(target) - int(current)
    band = max(int(band_pct * max(abs(int(target)), abs(int(current)))), int(min_units))
    return delta if abs(delta) >= band else 0


# --- TP/SL brackets + margin/liquidation guard (TASK A) ---
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_SL_MULT = 2.0       # protective stop = entry - sl_mult*ATR (long-or-flat)
DEFAULT_ATR_TP_MULT = 4.0       # take-profit = entry + tp_mult*ATR (2:1 R:R by default)
DEFAULT_ENABLE_SL = True
DEFAULT_ENABLE_TP = True
DEFAULT_MAX_MARGIN_UTIL = 0.50  # margin rail: cap projected margin at 50% of NAV
DEFAULT_MARGIN_RATE = 0.04      # conservative FX-major margin (~25:1) for the guard estimate
MIN_REPAIR_RR = 1.2


def compute_atr(candles_by_instrument: Dict[str, dict], *, period: int = DEFAULT_ATR_PERIOD) -> Dict[str, float]:
    """Average True Range (in price units) per instrument from COMPLETE candle OHLC."""
    out: Dict[str, float] = {}
    for inst, resp in (candles_by_instrument or {}).items():
        trs: List[float] = []
        prev_close: Optional[float] = None
        for c in (resp or {}).get("candles", []) or []:
            if not c.get("complete", False):
                continue
            m = c.get("mid") or {}
            try:
                hi, lo, cl = float(m["h"]), float(m["l"]), float(m["c"])
            except (KeyError, TypeError, ValueError):
                continue
            tr = (hi - lo) if prev_close is None else max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
            trs.append(tr)
            prev_close = cl
        if len(trs) >= period:
            out[inst] = sum(trs[-period:]) / period
    return out


def bracket_prices(entry: float, atr: Optional[float], *, sl_mult: float = DEFAULT_ATR_SL_MULT,
                   tp_mult: float = DEFAULT_ATR_TP_MULT, enable_sl: bool = DEFAULT_ENABLE_SL,
                   enable_tp: bool = DEFAULT_ENABLE_TP):
    """Long-side ATR brackets -> (stop_loss_price, take_profit_price). None when unavailable.

    SL = entry - sl_mult*ATR (protective; default ON). TP = entry + tp_mult*ATR
    (default ON for Tier 7 bounded exits). No ATR -> no bracket (refuse to
    fabricate a stop/target at a made-up distance)."""
    if not (atr and atr > 0 and entry and entry > 0):
        return None, None
    sl = entry - sl_mult * atr if enable_sl else None
    if sl is not None and sl <= 0:
        sl = None
    tp = entry + tp_mult * atr if enable_tp else None
    return sl, tp


def bracket_distances(atr: Optional[float], *, sl_mult: float = DEFAULT_ATR_SL_MULT,
                      tp_mult: float = DEFAULT_ATR_TP_MULT, enable_sl: bool = DEFAULT_ENABLE_SL,
                      enable_tp: bool = DEFAULT_ENABLE_TP):
    """ATR bracket DISTANCES (price units) -> (sl_distance, tp_distance).

    For attaching via OANDA ``stopLossOnFill.distance`` / ``takeProfitOnFill.distance``,
    which anchor to the ACTUAL FILL price. SL distance = sl_mult*ATR, TP distance =
    tp_mult*ATR, so the realized R:R is a constant tp_mult/sl_mult for EVERY
    instrument (JPY or not) and never depends on the gap between the last candle
    close and the live fill. This is the fix for the 2026-06-30 stale-anchor bug
    where absolute brackets computed from ``last_px`` (last complete daily close)
    skewed R:R per-instrument and dropped USD_JPY to 0.89 < 1.2. No ATR -> no
    distances (refuse to fabricate a stop at a made-up distance)."""
    if not (atr and atr > 0):
        return None, None
    sl = sl_mult * atr if enable_sl else None
    tp = tp_mult * atr if enable_tp else None
    return sl, tp


def repair_bracket_prices(*, entry: float, current: float, atr: float,
                          sl_mult: float = DEFAULT_ATR_SL_MULT,
                          tp_mult: float = DEFAULT_ATR_TP_MULT,
                          min_rr: float = MIN_REPAIR_RR):
    """Absolute SL/TP to REPAIR an already-open long, anchored to the trade's ENTRY.

    A trade has already filled, so there is no ``*OnFill`` anchor — the repair must
    use absolute prices. The correct reference is the trade's ENTRY (``trade['price']``),
    NOT the moving ``current`` price (the old bug). Computed levels are clamped so
    ``sl < current < tp`` — a late repair can never post a leg on the wrong side of
    the market and fire immediately — and the TP keeps at least ``min_rr`` R:R against
    the (possibly clamped) stop. Returns ``(sl, tp)``."""
    buffer = 0.10 * sl_mult * atr  # keep legs strictly off the current market
    sl = min(entry - sl_mult * atr, current - buffer)
    tp = entry + tp_mult * atr
    tp = max(tp, current + buffer, current + min_rr * (current - sl))
    return sl, tp


def margin_projection(want_units: Dict[str, int], last_px: Dict[str, float], *,
                      margin_rate: float = DEFAULT_MARGIN_RATE, home_ccy: str = "USD") -> float:
    """Projected margin (home ccy) for a target book: sum(|units| * base->home * margin_rate)."""
    total = 0.0
    for inst, u in want_units.items():
        rate = base_to_home_rate(inst, last_px, home_ccy=home_ccy)
        if rate and rate > 0:
            total += abs(int(u)) * rate * float(margin_rate)
    return total


def margin_scale(want_units: Dict[str, int], last_px: Dict[str, float], nav: float, *,
                 max_margin_util: float = DEFAULT_MAX_MARGIN_UTIL,
                 margin_rate: float = DEFAULT_MARGIN_RATE):
    """Margin/liquidation RAIL: scale the book so projected margin <= max_margin_util*NAV.

    Returns ``(scaled_units, scale_factor)``; ``factor < 1`` means the guard clamped the
    book (refused the excess exposure). This caps margin usage REGARDLESS of the leverage
    dial — the binding safety rail under aggressive sizing."""
    projected = margin_projection(want_units, last_px, margin_rate=margin_rate)
    cap = float(max_margin_util) * float(nav)
    if projected <= cap or projected <= 0:
        return dict(want_units), 1.0
    factor = cap / projected
    return {i: int(int(u) * factor) for i, u in want_units.items()}, factor


def _iso_now() -> str:
    """UTC ISO-8601 stamp for decision identity. Local time would make the
    cycle_id non-comparable across DST and machines."""
    from datetime import timezone

    return datetime.now(timezone.utc).isoformat()


def _order_price(order: Optional[dict]) -> Optional[float]:
    try:
        return float((order or {}).get("price"))
    except (ValueError, TypeError):
        return None


def repair_missing_trade_brackets(
    client: "OandaPracticeClient",
    *,
    last_px: Dict[str, float],
    atr: Dict[str, float],
    enable_sl: bool = DEFAULT_ENABLE_SL,
    enable_tp: bool = DEFAULT_ENABLE_TP,
    atr_sl_mult: float = DEFAULT_ATR_SL_MULT,
    atr_tp_mult: float = DEFAULT_ATR_TP_MULT,
    trades: Optional[List[dict]] = None,
) -> int:
    """Attach missing required brackets to existing open long trades.

    Existing bracket orders are left untouched. Repairs are anchored to the trade's
    ENTRY price (``repair_bracket_prices``), preserving the intended 2:1 R:R, and
    clamped so ``SL < current < TP`` — a late repair can never post a leg on the
    wrong side of the market and fire immediately. If a required bracket is missing
    but cannot be computed safely, the caller should fail the cycle closed.

    ``trades`` lets the caller pass an already-fetched OPEN-trades list (the risk
    gate needs the same snapshot) to avoid a duplicate API round-trip; when
    omitted this fetches it itself (unchanged behavior for existing callers).
    """
    if trades is None:
        trades = (client.get_trades(state="OPEN") or {}).get("trades", []) or []
    repaired = 0
    for trade in trades:
        inst = trade.get("instrument")
        trade_id = str(trade.get("id") or "")
        try:
            units = float(trade.get("currentUnits") or trade.get("initialUnits") or 0)
        except (ValueError, TypeError):
            units = 0.0
        current = float(last_px.get(inst) or 0.0)
        inst_atr = float(atr.get(inst) or 0.0)
        if not inst or not trade_id or units <= 0:
            continue
        has_sl = bool(trade.get("stopLossOrder"))
        has_tp = bool(trade.get("takeProfitOrder"))
        if (not enable_sl or has_sl) and (not enable_tp or has_tp):
            continue
        if current <= 0 or inst_atr <= 0:
            raise ValueError(f"cannot repair missing brackets for {inst} trade {trade_id}: no price/ATR")

        # Anchor the repair to the trade's ENTRY price (trade['price']), not the
        # moving current price — same root-cause fix as the open path. Levels are
        # clamped inside repair_bracket_prices so sl < current < tp (a late repair
        # can't fire immediately) while preserving the entry-anchored 2:1 R:R.
        entry_px = _order_price(trade) or current
        cand_sl, cand_tp = repair_bracket_prices(
            entry=entry_px, current=current, atr=inst_atr,
            sl_mult=atr_sl_mult, tp_mult=atr_tp_mult, min_rr=MIN_REPAIR_RR)
        existing_sl = _order_price(trade.get("stopLossOrder"))
        sl = tp = None
        if enable_sl and not has_sl:
            if cand_sl <= 0 or cand_sl >= current:
                raise ValueError(f"invalid repair SL for {inst} trade {trade_id}: {cand_sl}")
            sl = cand_sl
            existing_sl = sl
        if enable_tp and not has_tp:
            if existing_sl is not None and existing_sl < current:
                cand_tp = max(cand_tp, current + MIN_REPAIR_RR * (current - existing_sl))
            if cand_tp <= current:
                raise ValueError(f"invalid repair TP for {inst} trade {trade_id}: {cand_tp}")
            tp = cand_tp

        client.set_trade_dependent_orders(
            trade_id=trade_id,
            instrument=inst,
            stop_loss_price=sl,
            take_profit_price=tp,
        )
        repaired += 1
        logger.warning("OANDA bracket repair: %s trade=%s SL=%s TP=%s",
                       inst, trade_id, f"{sl:.5f}" if sl else "-", f"{tp:.5f}" if tp else "-")
    return repaired


def _open_trades_to_exposure_positions(
    open_trades: List[dict],
    r_distance_by_trade_id: Dict[str, float],
    r_distance_fallback_by_instrument: Dict[str, float],
    last_prices: Dict[str, float],
) -> List[ExposurePosition]:
    """Translate this cycle's OPEN OANDA trades into ``ExposurePosition``
    records for the portfolio exposure engine (``src.hedge.portfolio_exposure``)
    — an ADDITIONAL, independently-computed correlation check that runs
    ALONGSIDE (never instead of) ``bucket_cap_gate``. Direction is read from
    each trade's actual signed units (not assumed long-only) so a book that
    somehow carries a short is still represented correctly. Mirrors
    ``compute_bucket_risk``'s entry-anchored-R-over-current-ATR-fallback
    discipline exactly, so both engines see the SAME risk_home per trade.
    A trade whose risk can't be resolved (missing R, unresolvable quote rate)
    is SKIPPED here (not fabricated as zero) — the same trade is equally
    skipped by ``compute_bucket_risk``, so neither engine silently under-
    counts relative to the other."""
    positions: List[ExposurePosition] = []
    for trade in open_trades or []:
        inst = trade.get("instrument")
        trade_id = str(trade.get("id") or "")
        if not inst:
            continue
        try:
            units = float(trade.get("currentUnits") or trade.get("initialUnits") or 0)
        except (ValueError, TypeError):
            units = 0.0
        if units == 0:
            continue
        direction = "long" if units > 0 else "short"
        r_dist = r_distance_by_trade_id.get(trade_id) or r_distance_fallback_by_instrument.get(inst)
        if not r_dist or r_dist <= 0:
            continue
        q2h = quote_to_home_rate(inst, last_prices)
        if q2h is None or q2h <= 0:
            continue
        risk_home = trade_risk_home(trade, r_dist, q2h)
        positions.append(ExposurePosition(asset_class="fx", instrument=inst,
                                          direction=direction, risk_home=risk_home,
                                          source="oanda_open_trade"))
    return positions


def exposure_engine_gate(
    *, open_positions: List[ExposurePosition], candidate: ExposurePosition,
    risk_unit_home: float, max_bucket_risk_r: float,
    currency_bucket_map: Optional[Dict[str, dict]] = None,
) -> Tuple[bool, str]:
    """ADDITIONAL portfolio-exposure check (``src.hedge.portfolio_exposure``),
    run ALONGSIDE ``trend_risk_gates.bucket_cap_gate`` — never instead of it.
    ``bucket_cap_gate`` is called unchanged at its existing call site in
    ``run_oanda_trend_cycle``; this is a second, independently-implemented AND
    condition, so the combined result can only be MORE restrictive than
    ``bucket_cap_gate`` alone, never less (an approval requires BOTH to pass).

    Fail-closed: any exposure-engine failure (unresolvable instrument/currency,
    non-finite risk) refuses the candidate rather than silently treating the
    gap as zero exposure — matches ``ExposureReport.fail_closed``'s own
    contract in ``portfolio_exposure.py``.
    """
    # Code reviewer finding (2026-07-08): build_exposure_report's
    # net_currency_exposure NETS opposing directions (a long + a short of the
    # same currency partially cancel), whereas bucket_cap_gate's
    # bucket_risk_home accumulates UNSIGNED per-direction totals that never
    # net against each other. The two are equivalent only because this lane
    # is strictly long-or-flat (never shorts) — verified by construction
    # (see module docstring; trend_targets only ever emits weight >= 0). If a
    # short/hedge leg were ever introduced, signed netting could UNDERSTATE
    # risk relative to bucket_cap_gate, silently weakening this "independent,
    # at-least-as-strict" check. Fail closed instead of trusting a signed net
    # outside the regime this equivalence was proven for.
    all_positions = list(open_positions) + [candidate]
    if any(p.direction == "short" for p in all_positions):
        return False, "exposure_engine_short_leg_unsupported"
    report = build_exposure_report(open_positions, candidate, currency_bucket_map=currency_bucket_map)
    if report.fail_closed:
        return False, f"exposure_engine_fail_closed:{list(report.fail_reasons)}"
    legs = currency_legs(candidate.instrument)
    if legs is None:
        return False, "exposure_engine_unresolvable_instrument"
    cap = float(max_bucket_risk_r) * float(risk_unit_home)
    for ccy in legs:
        net = report.net_currency_exposure.get(ccy, 0.0)
        if abs(net) > cap + 1e-9:
            return False, f"exposure_engine_bucket_exceeded:{ccy}"
    return True, "exposure_engine_ok"


def run_oanda_trend_cycle(
    *,
    client: "OandaPracticeClient",
    config: "ScannerConfig",
    instruments: List[str],
    project_root: Path = Path("."),
    granularity: str = DEFAULT_GRANULARITY,
    sma_window: int = DEFAULT_SMA,
    candle_count: int = DEFAULT_CANDLE_COUNT,
    gross_leverage: float = DEFAULT_GROSS_LEVERAGE,
    enable_sl: bool = DEFAULT_ENABLE_SL,
    enable_tp: bool = DEFAULT_ENABLE_TP,
    atr_sl_mult: float = DEFAULT_ATR_SL_MULT,
    atr_tp_mult: float = DEFAULT_ATR_TP_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
    max_margin_util: float = DEFAULT_MAX_MARGIN_UTIL,
    margin_rate: float = DEFAULT_MARGIN_RATE,
    risk_pct: float = DEFAULT_RISK_PCT,
    max_bucket_risk_r: float = DEFAULT_MAX_BUCKET_RISK_R,
    bias_share_threshold: float = DEFAULT_BIAS_SHARE_THRESHOLD,
    bias_min_instruments: int = DEFAULT_BIAS_MIN_INSTRUMENTS,
    breakeven_trigger_r: float = DEFAULT_BREAKEVEN_TRIGGER_R,
    trail_start_r: float = DEFAULT_TRAIL_START_R,
    pyramid_max_layers: int = DEFAULT_PYRAMID_MAX_LAYERS,
    pyramid_cum_risk_r_cap: float = DEFAULT_PYRAMID_CUM_RISK_R_CAP,
    max_cost_r_fraction: float = DEFAULT_MAX_COST_R_FRACTION,
    cost_model_transactions_path: Optional[Path] = None,
    cost_model_tick_root: Optional[Path] = None,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> OandaTrendResult:
    """One PRACTICE trend cycle: candles -> signal -> (demo) orders. Fail-closed.

    Respects the oanda_fx lane halt (REFUSE; the legacy global halt still wins
    over any lane) and asserts practice-only. With ``dry_run``
    it computes + logs targets without placing orders. Otherwise it reads NAV +
    open positions and sizes each "on" instrument via RISK-NORMALIZED sizing
    (``risk_pct`` of NAV / ATR-based stop distance — operator-directed rebuild,
    2026-07-01, replacing the old leverage-based ``target_units``). ``gross_leverage``
    is now a CEILING on total exposure only (``leverage_cap_scale``), not the sizing
    basis. A RISK GATE (one-position-per-instrument, correlation-bucket caps,
    pyramid rules, cost-awareness, an independent portfolio-exposure-engine
    correlation check) runs before every size-increase order; decreases/closes
    are never blocked (risk-reducing orders always pass). The cost-awareness
    and exposure-engine checks (2026-07-08) are purely ADDITIVE — they can
    only refuse a candidate the prior rules already approved, never approve
    one they blocked.
    """
    gross_leverage = clamp_leverage(gross_leverage)
    assert getattr(config, "oanda_environment", "practice") == "practice", \
        "HARD LINE: oanda_environment must be 'practice'"
    root = Path(project_root)

    halted, readable, reason = _lane_halted(root, "oanda_fx")
    if not readable or halted:
        logger.warning("OANDA trend cycle REFUSED — halt=%s readable=%s reason=%s", halted, readable, reason)
        return OandaTrendResult(False, "halted", {}, 0)

    # 1. candles -> close panel -> trend targets
    candles = {}
    for inst in instruments:
        try:
            candles[inst] = client.get_candles(
                inst, granularity=granularity, count=candle_count, price="M")
        except Exception as exc:  # network/auth surfaced to caller as no_data
            logger.error("OANDA candles failed for %s: %s", inst, exc)
            raise
    panel = candles_to_close_panel(candles)
    if panel.empty:
        return OandaTrendResult(False, "no_data", {}, 0)
    atr = compute_atr(candles, period=atr_period)   # for protective SL/TP brackets
    targets = trend_targets(panel, sma_window=sma_window)
    on = {k: v for k, v in targets.items() if v > 0}
    logger.info("OANDA trend targets: %d on / %d flat — on=%s",
                len(on), len(targets) - len(on), sorted(on))

    if dry_run:
        return OandaTrendResult(True, "dry_run", targets, 0)

    # 2. NAV + current positions -> delta orders (long-or-flat)
    summary = client.get_account_summary() or {}
    account_summary = summary.get("account") or {}
    nav = float(account_summary.get("NAV", 0.0) or 0.0)

    # NAV glitch guard (verifier rec): a transient summary error -> nav<=0 would size
    # every name to the 1-unit floor (noise orders). Refuse the cycle instead.
    if not math.isfinite(nav) or nav <= 0:
        logger.warning("OANDA trend cycle skipped — NAV unavailable/nan/<=0 (transient?)")
        return OandaTrendResult(False, "no_nav", targets, 0)

    # Autonomous-safety drawdown rail: stop adding risk if the demo NAV bleeds.
    dd = nav_drawdown_breached(nav, root / "trained_data" / "oanda" / "peak_nav.json")
    if dd is not None:
        logger.error("OANDA trend cycle HALTED — NAV drawdown %.1f%% >= %.0f%% from peak",
                     dd * 100, DEFAULT_DD_HARD * 100)
        return OandaTrendResult(False, "drawdown_halt", targets, 0)
    last_px = {inst: panel[inst].dropna().iloc[-1] for inst in panel.columns
               if not panel[inst].dropna().empty}
    risk_pct = clamp_risk_pct(risk_pct)
    risk_unit_home = nav * risk_pct

    # Promoted AXIOM risk controller (PRACTICE only). It cannot choose direction
    # and cannot raise the configured risk budget: its sole output is a [0, 1]
    # multiplier applied to the already ATR-normalized target below. A missing
    # promotion preserves legacy sizing; an enabled-but-invalid promotion fails
    # to zero so artifact tampering can never silently restore full risk.
    from src.axiom_training.risk_runtime import (
        evaluate_runtime_risk,
        scale_target_units,
    )
    risk_decision = evaluate_runtime_risk(
        root=root,
        environment=getattr(config, "oanda_environment", "practice"),
        nav=nav,
        open_trade_count=int(account_summary.get("openTradeCount", 0) or 0),
        now=now,
    )
    if risk_decision.promotion_enabled:
        log = logger.info if risk_decision.valid else logger.error
        log(
            "AXIOM RISK MODEL: valid=%s scale=%.4f reason=%s vol=%s target_vol=%s "
            "drawdown=%s loss_streak=%s concentration=%s resolved=%s cluster=%s "
            "strategy_prior=%s transition=%s liquidity=%s confidence_gap=%s "
            "disagreement=%s hash=%s",
            risk_decision.valid,
            risk_decision.scale,
            risk_decision.reason,
            risk_decision.trailing_volatility,
            risk_decision.target_volatility,
            risk_decision.drawdown,
            risk_decision.loss_streak,
            risk_decision.portfolio_concentration,
            risk_decision.exposure_resolved_fraction,
            risk_decision.correlation_cluster_stress,
            risk_decision.strategy_risk_multiplier,
            risk_decision.regime_transition_risk,
            risk_decision.liquidity_stress,
            risk_decision.confidence_shortfall,
            risk_decision.model_disagreement,
            risk_decision.model_sha256,
        )

    # RULE 3 — risk-normalized sizing: units = (nav*risk_pct) / (sl_mult*ATR *
    # quote->home rate). Requires a real ATR-based stop distance to be
    # meaningful; an instrument with SL disabled or no ATR is refused (0), not
    # fabricated, since sizing "risk" with no attached stop is undefined.
    want: Dict[str, int] = {}
    for inst, w in targets.items():
        if w <= 0 or not enable_sl:
            want[inst] = 0
            continue
        sl_dist, _tp_dist_unused = bracket_distances(
            atr.get(inst), sl_mult=atr_sl_mult, tp_mult=atr_tp_mult,
            enable_sl=True, enable_tp=False)
        want[inst] = risk_normalized_units(
            instrument=inst, r_distance_quote=sl_dist, nav=nav, risk_pct=risk_pct,
            last_prices=last_px)
    want = scale_target_units(want, risk_decision.scale)
    # Visibility (verifier rec): an on-signal instrument sized 0 means its ATR or
    # quote->home rate was underivable — surface it so a silently-untraded
    # instrument can't hide as a no-op.
    silently_flat = [i for i, w in targets.items() if w > 0 and want.get(i, 0) == 0]
    if silently_flat:
        logger.warning("on-signal but sized 0 (no ATR or quote->home rate, refuse "
                       "rather than fabricate): %s", silently_flat)

    # RULE 3 reconciliation — gross_leverage is now a CEILING on total exposure
    # only (never the sizing basis): scale the whole risk-sized book down (never
    # up) if it would exceed gross_leverage*NAV.
    want, lev_factor = leverage_cap_scale(want, last_px, nav, gross_leverage=gross_leverage)
    if lev_factor < 1.0:
        logger.warning("LEVERAGE CAP fired — risk-sized book exceeded %.1fx NAV, "
                       "scaled to fit (scale=%.3f)", gross_leverage, lev_factor)

    # MARGIN/LIQUIDATION RAIL: innermost hard rail, clamp so projected margin <=
    # max_margin_util*NAV regardless of the leverage dial or risk sizing.
    want, margin_factor = margin_scale(want, last_px, nav,
                                       max_margin_util=max_margin_util, margin_rate=margin_rate)
    if margin_factor < 1.0:
        logger.warning("MARGIN GUARD fired — clamped book to %.0f%% NAV margin cap "
                       "(scale=%.3f)", max_margin_util * 100, margin_factor)

    pos_resp = client.get_open_positions() or {}
    current: Dict[str, int] = {}
    for p in pos_resp.get("positions", []) or []:
        inst = p.get("instrument")
        net = int(float((p.get("long") or {}).get("units", 0) or 0)) + \
            int(float((p.get("short") or {}).get("units", 0) or 0))
        if inst:
            current[inst] = net

    # Single OPEN-trades snapshot shared by the bracket repair, the RISK GATE,
    # and winner management — one API round-trip, one consistent view per cycle.
    open_trades = (client.get_trades(state="OPEN") or {}).get("trades", []) or []
    trades_by_instrument: Dict[str, List[dict]] = {}
    for t in open_trades:
        ti = t.get("instrument")
        if ti:
            trades_by_instrument.setdefault(ti, []).append(t)

    try:
        repair_missing_trade_brackets(
            client,
            last_px=last_px,
            atr=atr,
            enable_sl=enable_sl,
            enable_tp=enable_tp,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult,
            trades=open_trades,
        )
    except Exception as exc:
        logger.error("OANDA trend cycle REFUSED — missing-bracket repair failed: %s", exc)
        return OandaTrendResult(False, "bracket_repair_failed", targets, 0)

    # r_distance PER INSTRUMENT off current ATR — used (a) as the fallback for a
    # trade not yet in persisted risk_state, and (b) for sizing a brand-new
    # candidate order this cycle (which has no trade_id / entry ATR yet).
    r_dist_by_instrument = {i: (bracket_distances(atr.get(i), sl_mult=atr_sl_mult,
                                                  enable_sl=True, enable_tp=False)[0] or 0.0)
                            for i in set(list(targets.keys()) + list(trades_by_instrument.keys()))}

    # Load persisted per-trade risk state BEFORE computing bucket risk (rule 2
    # fix, verifier 2026-07-01): bucket risk must be measured off each OPEN
    # trade's ENTRY-anchored R (rule 5's risk_state), not current ATR — an ATR
    # contraction since entry would otherwise make the gate think less risk is
    # outstanding than was actually taken, silently under-counting bucket usage.
    risk_state_path = root / "trained_data" / "oanda" / "risk_state.json"
    risk_state = load_risk_state(risk_state_path)
    risk_state_dirty = False
    for trade in open_trades:
        trade_id = str(trade.get("id") or "")
        inst = trade.get("instrument")
        if not trade_id or not inst or trade_id in risk_state:
            continue
        entry_px = _order_price(trade)
        r_dist = r_dist_by_instrument.get(inst) or 0.0
        if entry_px is None or r_dist <= 0:
            continue  # can't establish R yet (first sight, no ATR) — try again next cycle
        risk_state[trade_id] = {"instrument": inst, "entry_price": entry_px, "r_distance": r_dist}
        risk_state_dirty = True
    r_distance_by_trade_id = {tid: st["r_distance"] for tid, st in risk_state.items()
                              if st.get("r_distance")}

    # RULE 2 + RULE 4 — bucket risk snapshot (BEFORE this cycle's new orders,
    # entry-anchored) and the (log-only) bias detector.
    bucket_risk_home, bucket_insts = compute_bucket_risk(
        open_trades, r_distance_by_trade_id, last_px,
        r_distance_fallback_by_instrument=r_dist_by_instrument)
    for warning in bias_detector(bucket_risk_home, bucket_insts,
                                 share_threshold=bias_share_threshold,
                                 min_instruments=bias_min_instruments):
        logger.warning(warning)

    # RULE 7 (2026-07-08, ADDITIVE) — realistic execution-cost estimates for
    # every instrument this cycle might trade, built ONCE (real OANDA fills +
    # captured ticks; never fabricated — an instrument with no history reports
    # source="insufficient_data" and the gate below refuses it, never assumes
    # free). Paths default to this project_root's real artifacts (matches the
    # risk_state_path pattern above); overridable for tests via tmp_path.
    cost_estimates: Dict[str, "CostEstimate"] = build_cost_model(
        instruments,
        transactions_path=cost_model_transactions_path
        or (root / "trained_data" / "oanda" / "transactions.jsonl"),
        tick_root=cost_model_tick_root or (root / "trained_data" / "ticks"),
    )

    # ADDITIONAL portfolio-exposure-engine correlation check (2026-07-08),
    # ALONGSIDE (never instead of) the bucket_cap_gate rule 2 cap below. Seeded
    # from the SAME pre-cycle open book as compute_bucket_risk (same r_distance/
    # quote-rate discipline), then accumulated in-loop on each approval exactly
    # like bucket_risk_home — closes the same-cycle-bypass class for this
    # second, independently-implemented check too.
    currency_bucket_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    exposure_open_positions = _open_trades_to_exposure_positions(
        open_trades, r_distance_by_trade_id, r_dist_by_instrument, last_px)

    # RULE 5 — winner management: move stop to breakeven after +1R, then trail,
    # for every open long trade (uses the SAME risk_state loaded above).
    for trade in open_trades:
        trade_id = str(trade.get("id") or "")
        inst = trade.get("instrument")
        if not trade_id or not inst:
            continue
        st = risk_state.get(trade_id)
        current_px = last_px.get(inst)
        if st is None or current_px is None:
            continue
        current_sl = _order_price(trade.get("stopLossOrder"))
        new_sl = evaluate_winner_stop(
            entry_price=st["entry_price"], r_distance=st["r_distance"],
            current_price=float(current_px), current_sl=current_sl,
            breakeven_trigger_r=breakeven_trigger_r, trail_start_r=trail_start_r)
        if new_sl is not None:
            client.set_trade_dependent_orders(trade_id=trade_id, instrument=inst,
                                              stop_loss_price=new_sl)
            logger.info("WINNER MANAGEMENT: %s trade=%s stop -> %.5f (entry=%.5f r=%.5f)",
                        inst, trade_id, new_sl, st["entry_price"], st["r_distance"])
    if risk_state_dirty:
        save_risk_state(risk_state_path, risk_state)

    # ── Decision capture (2026-07-21) ────────────────────────────────
    # Record the intention BEFORE any order leaves. This lane is the only one
    # with a working broker, so it is the only source of broker-realized
    # closes -- and until now every one of its fills was unattributable (all
    # 124 realized closes in the journal carry no decision_id).
    #
    # Non-blocking by design: orders are placed one-by-one inside the loop
    # below with risk gates already applied, so aborting mid-loop could leave
    # the book half-rebalanced -- worse than an unattributed fill. The equity
    # seam aborts instead because nothing has been sent at that point.
    fx_decisions = []
    try:
        from src.equity.fx_decision_capture import (
            build_fx_cycle_decisions,
            persist_fx_decisions,
        )

        fx_cycle_id = f"oanda_fx:{_iso_now()}"
        fx_decisions = build_fx_cycle_decisions(
            cycle_id=fx_cycle_id,
            decision_timestamp=_iso_now(),
            targets=targets,
            want_units=want,
            current_units=current,
            nav=nav,
            gross_leverage=gross_leverage,
            last_px=last_px,
            atr=atr,
            disposition=Disposition.EXECUTE,
            reason_codes=("trend_sma", f"risk_scale={risk_decision.scale:.4f}"),
            strategy_version=f"trend_sma{sma_window}",
            market_data_asof=_iso_now(),
        )
        persist_fx_decisions(fx_decisions)
        logger.info("FX decision capture: %d decision(s) recorded for cycle %s",
                    len(fx_decisions), fx_cycle_id)
    except Exception as _cap_exc:  # noqa: BLE001 - capture must never break the cycle
        logger.error("FX_DECISION_CAPTURE_FAILED: %r -- this cycle's fills will be "
                     "UNATTRIBUTABLE", _cap_exc)

    _fx_digest_by_instrument = {}
    for _d in fx_decisions:
        try:
            _fx_digest_by_instrument[_d.instrument] = _d.digest()
        except Exception:  # noqa: BLE001
            pass

    placed = 0
    for inst in instruments:
        delta = rebalance_delta(want.get(inst, 0), current.get(inst, 0))
        if delta == 0:
            continue
        existing_for_inst = trades_by_instrument.get(inst, [])
        if delta > 0:
            # RULES 1 + 6 — one-position-per-instrument, pyramid as the only
            # authorized exception (green-only, smaller-than-last, <=1R cumulative).
            sl_dist_gate = r_dist_by_instrument.get(inst) or 0.0
            q2h = quote_to_home_rate(inst, last_px)
            if q2h is None or q2h <= 0:
                # Verifier finding (2026-07-01): an unresolvable rate must REFUSE the
                # add, not fall back to 0.0 — passing quote_to_home=0.0 zeroes every
                # risk_home computation inside pyramid_gate, silently disabling the
                # cumulative-risk cap instead of blocking (same no-fabrication
                # discipline as risk_normalized_units / bucket_cap_gate).
                logger.warning("RISK GATE BLOCKED %s units=%+d — no quote->home rate "
                               "(refuse, no fabrication)", inst, delta)
                continue
            allow, reason = one_position_gate(
                existing_trades_for_instrument=existing_for_inst, candidate_units=delta,
                r_distance_quote=sl_dist_gate, risk_unit_home=risk_unit_home,
                quote_to_home=q2h, pyramid_max_layers=pyramid_max_layers,
                pyramid_cum_risk_r_cap=pyramid_cum_risk_r_cap)
            if not allow:
                logger.warning("RISK GATE BLOCKED %s units=%+d — %s", inst, delta, reason)
                continue

            # RULE 7 (ADDITIVE, 2026-07-08) — cost-awareness: refuse a candidate
            # whose modeled round-trip execution cost eats too much of its own
            # 1R stop budget. Can only refuse, never approve something the
            # prior rules already blocked.
            allow, reason = cost_aware_gate(
                instrument=inst, r_distance_quote=sl_dist_gate,
                cost_estimate=cost_estimates.get(inst),
                max_cost_r_fraction=max_cost_r_fraction)
            if not allow:
                logger.warning("RISK GATE BLOCKED %s units=%+d — %s", inst, delta, reason)
                continue

            # ADDITIONAL portfolio-exposure-engine check (2026-07-08) — runs
            # ALONGSIDE bucket_cap_gate below (an independent AND condition,
            # never a replacement); see exposure_engine_gate's docstring.
            candidate_risk_home = abs(delta) * sl_dist_gate * q2h
            candidate_position = ExposurePosition(
                asset_class="fx", instrument=inst, direction="long",
                risk_home=candidate_risk_home, source="candidate")
            allow, reason = exposure_engine_gate(
                open_positions=exposure_open_positions, candidate=candidate_position,
                risk_unit_home=risk_unit_home, max_bucket_risk_r=max_bucket_risk_r,
                currency_bucket_map=currency_bucket_map)
            if not allow:
                logger.warning("RISK GATE BLOCKED %s units=%+d — %s", inst, delta, reason)
                continue

            # RULE 2 — correlation-bucket cap, checked against a RUNNING total
            # (fix, verifier 2026-07-01): bucket_risk_home is mutated in-loop via
            # accumulate_bucket_risk() immediately below on approval, so the Nth
            # correlated instrument approved THIS cycle is measured against the
            # (N-1) prior approvals from this same cycle, not a stale pre-cycle
            # snapshot — closes the same-cycle bypass where e.g. 3 yen-cross
            # longs newly flipping on together could each individually clear a
            # 2R cap while jointly landing at ~3R.
            allow, reason = bucket_cap_gate(
                instrument=inst, candidate_units=delta, r_distance_quote=sl_dist_gate,
                last_prices=last_px, bucket_risk_home=bucket_risk_home,
                risk_unit_home=risk_unit_home, max_bucket_risk_r=max_bucket_risk_r)
            if not allow:
                logger.warning("RISK GATE BLOCKED %s units=%+d — %s", inst, delta, reason)
                continue
            accumulate_bucket_risk(bucket_risk_home, inst, abs(delta) * sl_dist_gate * q2h)
            exposure_open_positions.append(candidate_position)
        sl_dist = tp_dist = None
        if delta > 0:   # opening/increasing a long -> attach protective brackets
            # Size brackets as DISTANCES (sl_mult*ATR / tp_mult*ATR) and attach via
            # *OnFill.distance so OANDA anchors them to the ACTUAL FILL — not the
            # stale last complete candle close. Guarantees a constant tp:sl R:R for
            # every instrument (JPY and non-JPY) regardless of fill-vs-close drift.
            sl_dist, tp_dist = bracket_distances(atr.get(inst),
                                                 sl_mult=atr_sl_mult, tp_mult=atr_tp_mult,
                                                 enable_sl=enable_sl, enable_tp=enable_tp)
            if (enable_sl and sl_dist is None) or (enable_tp and tp_dist is None):
                logger.error("OANDA trend order REFUSED for %s — missing required bracket(s) "
                             "SLd=%s TPd=%s atr=%s",
                             inst, sl_dist, tp_dist, atr.get(inst))
                continue
        client.create_market_order(instrument=inst, units=delta,
                                   stop_loss_distance=sl_dist, take_profit_distance=tp_dist,
                                   client_tag="ml_engine_trend_demo")
        placed += 1
        # Append-only linkage: the decision row is never rewritten, and every
        # later lifecycle event resolves to this same decision digest.
        _fx_dig = _fx_digest_by_instrument.get(inst)
        if _fx_dig:
            try:
                from src.learning.decision_ledger import link_broker_order

                link_broker_order(_fx_dig,
                                  broker_order_id=f"{inst}:{_iso_now()}",
                                  client_order_id="ml_engine_trend_demo")
            except Exception as _lnk:  # noqa: BLE001 - order is away; do not roll back
                logger.error("FX_DECISION_CAPTURE_FAILED: could not link order for %s: %r",
                             inst, _lnk)
        _dec = 3 if str(inst).endswith("_JPY") else 5
        logger.info("OANDA PAPER order: %s units=%+d (target=%d current=%d) SLdist=%s TPdist=%s",
                    inst, delta, want.get(inst, 0), current.get(inst, 0),
                    f"{sl_dist:.{_dec}f}" if sl_dist else "-", f"{tp_dist:.{_dec}f}" if tp_dist else "-")

    # Honest end-of-cycle check (fix, verifier 2026-07-01): there is NO automatic
    # trim/reduce path for an over-cap bucket — a prior comment falsely claimed
    # this "self-corrects next cycle". Log it loudly instead so it's visible and
    # actionable, rather than silently persisting until trades close naturally.
    for warning in buckets_over_cap(bucket_risk_home, risk_unit_home=risk_unit_home,
                                    max_bucket_risk_r=max_bucket_risk_r):
        logger.error(warning)
    return OandaTrendResult(True, "executed", targets, placed)
