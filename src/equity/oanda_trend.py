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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from src.equity.decision_gate import _global_halt

if TYPE_CHECKING:  # pragma: no cover
    from src.scanner.config import ScannerConfig
    from src.utils.oanda_practice import OandaPracticeClient

logger = logging.getLogger(__name__)

DEFAULT_SMA = 100        # ~5-month trend on daily candles (managed-futures canonical)
DEFAULT_GRANULARITY = "D"
DEFAULT_CANDLE_COUNT = 300   # <= OANDA 5000 max; enough history for the SMA + warmup
DEFAULT_GROSS_LEVERAGE = 0.5  # conservative: total demo exposure <= 0.5x NAV


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


def base_to_home_rate(
    instrument: str,
    last_prices: Dict[str, float],
    *,
    home_ccy: str = "USD",
) -> Optional[float]:
    """USD (home-ccy) value of 1 unit of the instrument's BASE currency.

    OANDA units are denominated in the BASE currency (the left side of XXX_YYY),
    so the home-currency exposure of ``units`` is ``units * base_to_home_rate`` —
    NOT ``units * price`` (price is in the QUOTE currency). Examples (home=USD):
      USD_JPY -> base USD            -> 1.0
      EUR_USD -> base EUR            -> EUR_USD price (~1.10)
      GBP_JPY -> base GBP            -> GBP_USD price (~1.27, via the cross leg)
      USD_CAD -> base USD            -> 1.0
    Resolves the base->home rate from a direct ``BASE_HOME`` price, else an inverse
    ``HOME_BASE`` price. Returns ``None`` if it cannot be derived (caller refuses to
    size rather than fabricate units).
    """
    base = str(instrument).split("_")[0]
    if base == home_ccy:
        return 1.0
    direct = last_prices.get(f"{base}_{home_ccy}")        # e.g. EUR_USD
    if direct and direct > 0:
        return float(direct)
    inverse = last_prices.get(f"{home_ccy}_{base}")       # e.g. USD_CAD for base CAD
    if inverse and inverse > 0:
        return 1.0 / float(inverse)
    return None


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
    for inst, w in targets.items():
        if w <= 0:
            out[inst] = 0
            continue
        rate = base_to_home_rate(inst, last_prices, home_ccy=home_ccy)
        if rate is None or rate <= 0:
            logger.warning("no base->%s rate for %s — sizing 0 (refuse, no fabrication)",
                           home_ccy, inst)
            out[inst] = 0
            continue
        notional_home = float(w) * float(nav) * float(gross_leverage)
        out[inst] = max(int(notional_home / float(rate)), 1)
    return out


def run_oanda_trend_cycle(
    *,
    client: "OandaPracticeClient",
    config: "ScannerConfig",
    instruments: List[str],
    project_root: Path = Path("."),
    granularity: str = DEFAULT_GRANULARITY,
    sma_window: int = DEFAULT_SMA,
    candle_count: int = DEFAULT_CANDLE_COUNT,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> OandaTrendResult:
    """One PRACTICE trend cycle: candles -> signal -> (demo) orders. Fail-closed.

    Respects the global halt (REFUSE) and asserts practice-only. With ``dry_run``
    it computes + logs targets without placing orders. Otherwise it reads NAV +
    open positions and places market orders for the delta to each target.
    """
    assert getattr(config, "oanda_environment", "practice") == "practice", \
        "HARD LINE: oanda_environment must be 'practice'"
    root = Path(project_root)

    halted, readable = _global_halt(root)
    if not readable or halted:
        logger.warning("OANDA trend cycle REFUSED — halt=%s readable=%s", halted, readable)
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
    targets = trend_targets(panel, sma_window=sma_window)
    on = {k: v for k, v in targets.items() if v > 0}
    logger.info("OANDA trend targets: %d on / %d flat — on=%s",
                len(on), len(targets) - len(on), sorted(on))

    if dry_run:
        return OandaTrendResult(True, "dry_run", targets, 0)

    # 2. NAV + current positions -> delta orders (long-or-flat)
    summary = client.get_account_summary() or {}
    nav = float((summary.get("account") or {}).get("NAV", 0.0) or 0.0)
    last_px = {inst: panel[inst].dropna().iloc[-1] for inst in panel.columns
               if not panel[inst].dropna().empty}
    want = target_units(targets, nav, last_px)
    # Visibility (verifier rec): an on-signal instrument sized 0 means its base->home
    # rate was underivable (USD leg absent from the traded panel) — surface it so a
    # silently-untraded cross can't hide as a no-op.
    silently_flat = [i for i, w in targets.items() if w > 0 and want.get(i, 0) == 0]
    if silently_flat:
        logger.warning("on-signal but sized 0 (no base->home rate, add its USD leg): %s",
                       silently_flat)

    pos_resp = client.get_open_positions() or {}
    current: Dict[str, int] = {}
    for p in pos_resp.get("positions", []) or []:
        inst = p.get("instrument")
        net = int(float((p.get("long") or {}).get("units", 0) or 0)) + \
            int(float((p.get("short") or {}).get("units", 0) or 0))
        if inst:
            current[inst] = net

    placed = 0
    for inst in instruments:
        delta = int(want.get(inst, 0)) - int(current.get(inst, 0))
        if abs(delta) < 1:
            continue
        client.create_market_order(instrument=inst, units=delta, client_tag="ml_engine_trend_demo")
        placed += 1
        logger.info("OANDA PAPER order: %s units=%+d (target=%d current=%d)",
                    inst, delta, want.get(inst, 0), current.get(inst, 0))
    return OandaTrendResult(True, "executed", targets, placed)
