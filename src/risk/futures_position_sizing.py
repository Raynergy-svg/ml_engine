"""Futures Contract Position Sizing.

Futures sizing is fundamentally different from FX pip math.

FX sizing:
    units = (NAV * risk_pct) / (sl_pips * pip_value_per_unit)

Futures sizing:
    contracts = floor((NAV * risk_pct) / (|entry - stop| / tick_size * tick_value))

Where:
  - tick_size  = minimum price increment (e.g. 0.25 for ES)
  - tick_value = dollar value of 1 tick move (e.g. $12.50 for ES)
  - So 1 point of ES = (1 / 0.25) * $12.50 = $50/point = multiplier

Safety limits (paper mode defaults):
  - MAX_CONTRACTS_PER_TRADE = 5  (hard cap per position)
  - MAX_MARGIN_UTILIZATION  = 0.50 (50 % of NAV reserved for margin)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from src.brokers.instrument import Instrument
from src.brokers.registry import get_registry

logger = logging.getLogger(__name__)

# ─── Safety constants ──────────────────────────────────────────────────────────
MAX_CONTRACTS_PER_TRADE: int = 5        # Hard ceiling for paper/demo mode
MAX_MARGIN_UTILIZATION: float = 0.50    # Use at most 50 % of NAV as margin
MIN_CONTRACTS: int = 1                  # Never size below 1 contract


# ─── Public API ───────────────────────────────────────────────────────────────

def calculate_futures_contracts(
    nav: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    instrument: Instrument,
    *,
    current_margin_used: float = 0.0,
    max_contracts: int = MAX_CONTRACTS_PER_TRADE,
    max_margin_utilization: float = MAX_MARGIN_UTILIZATION,
) -> int:
    """Calculate the number of futures contracts to trade.

    Uses tick-value math to express dollar risk in contract terms:

        dollar_risk   = NAV * risk_pct
        ticks_at_risk = abs(entry_price - stop_price) / tick_size
        risk_per_contract = ticks_at_risk * tick_value
        raw_contracts = floor(dollar_risk / risk_per_contract)

    Then applies:
      - Hard ceiling: ``max_contracts``
      - Margin ceiling: 50 % of NAV across all open positions
      - Floor: 0 (never trade if margin is exhausted)

    Args:
        nav:                  Net asset value of the account ($).
        risk_pct:             Fraction of NAV to risk per trade (e.g. 0.02 = 2 %).
        entry_price:          Expected entry price.
        stop_price:           Stop-loss price.
        instrument:           Instrument dataclass (FUTURES asset class).
        current_margin_used:  Total margin already committed by open positions ($).
                              Used to enforce the MAX_MARGIN_UTILIZATION ceiling.
        max_contracts:        Per-trade contract ceiling (default: 5 for paper mode).
        max_margin_utilization: Fraction of NAV usable as total margin (default 0.50).

    Returns:
        Integer number of contracts (0 if trade should be skipped).

    Raises:
        ValueError: If instrument is not a FUTURES instrument, or if tick_size /
                    tick_value are None (misconfigured instrument).
    """
    # ── Validate instrument ─────────────────────────────────────────────────
    if instrument.asset_class != "FUTURES":
        raise ValueError(
            f"calculate_futures_contracts requires a FUTURES instrument, "
            f"got asset_class={instrument.asset_class!r} for {instrument.symbol}"
        )
    if instrument.tick_size is None or instrument.tick_value is None:
        raise ValueError(
            f"FUTURES instrument {instrument.symbol} has no tick_size/tick_value — "
            "check InstrumentRegistry definition."
        )
    if instrument.margin_requirement is None or instrument.margin_requirement <= 0:
        raise ValueError(
            f"FUTURES instrument {instrument.symbol} has invalid margin_requirement "
            f"({instrument.margin_requirement})."
        )

    # ── Validate inputs ─────────────────────────────────────────────────────
    if nav <= 0:
        logger.warning("calculate_futures_contracts: nav=%.2f <= 0 — returning 0 contracts", nav)
        return 0
    if risk_pct <= 0 or risk_pct > 1.0:
        logger.warning(
            "calculate_futures_contracts: invalid risk_pct=%.4f — clamping to [0.001, 0.20]",
            risk_pct,
        )
        risk_pct = max(0.001, min(0.20, risk_pct))

    sl_distance = abs(entry_price - stop_price)
    if sl_distance == 0:
        logger.warning(
            "calculate_futures_contracts: entry_price == stop_price for %s — no risk, returning 0",
            instrument.symbol,
        )
        return 0

    # ── Core sizing math ────────────────────────────────────────────────────
    dollar_risk: float = nav * risk_pct

    # Number of ticks between entry and stop
    ticks_at_risk: float = sl_distance / instrument.tick_size

    # Dollar risk per contract (how much 1 contract loses if stop is hit)
    risk_per_contract: float = ticks_at_risk * instrument.tick_value

    if risk_per_contract <= 0:
        logger.error(
            "calculate_futures_contracts: risk_per_contract=%.4f <= 0 for %s — returning 0",
            risk_per_contract, instrument.symbol,
        )
        return 0

    raw_contracts: float = dollar_risk / risk_per_contract
    contracts: int = math.floor(raw_contracts)

    logger.debug(
        "Futures sizing [%s]: NAV=%.0f, risk_pct=%.2f%%, sl_distance=%.4f, "
        "ticks_at_risk=%.1f, risk_per_contract=$%.2f, raw=%.3f → %d contracts",
        instrument.symbol, nav, risk_pct * 100, sl_distance,
        ticks_at_risk, risk_per_contract, raw_contracts, contracts,
    )

    # ── Apply safety caps ───────────────────────────────────────────────────

    # 1. Per-trade contract ceiling
    if contracts > max_contracts:
        logger.info(
            "Futures sizing [%s]: capping %d → %d contracts (MAX_CONTRACTS_PER_TRADE)",
            instrument.symbol, contracts, max_contracts,
        )
        contracts = max_contracts

    # 2. Margin ceiling — total margin used must not exceed max_margin_utilization * NAV
    margin_budget: float = (nav * max_margin_utilization) - current_margin_used
    if margin_budget <= 0:
        logger.warning(
            "Futures sizing [%s]: margin budget exhausted "
            "(current_margin_used=%.0f, budget=%.0f) — returning 0 contracts",
            instrument.symbol, current_margin_used, margin_budget,
        )
        return 0

    max_contracts_by_margin: int = math.floor(
        margin_budget / instrument.margin_requirement
    )
    if contracts > max_contracts_by_margin:
        logger.info(
            "Futures sizing [%s]: margin cap %d → %d contracts "
            "(budget=%.0f, margin_req=%.0f/contract)",
            instrument.symbol, contracts, max_contracts_by_margin,
            margin_budget, instrument.margin_requirement,
        )
        contracts = max_contracts_by_margin

    # 3. Final floor — 0 is valid (skip trade), 1 is minimum meaningful size
    contracts = max(0, contracts)

    logger.info(
        "Futures sizing [%s]: FINAL %d contract(s) "
        "(dollar_risk=$%.0f, risk_per_contract=$%.2f)",
        instrument.symbol, contracts, dollar_risk, risk_per_contract,
    )
    return contracts


def calculate_futures_contracts_from_symbol(
    symbol: str,
    nav: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    *,
    current_margin_used: float = 0.0,
    max_contracts: int = MAX_CONTRACTS_PER_TRADE,
) -> int:
    """Convenience wrapper — looks up the Instrument by symbol from the registry.

    Args:
        symbol:               Futures symbol (e.g. "ES", "NQ", "CL").
        nav:                  Account NAV ($).
        risk_pct:             Fraction of NAV to risk (e.g. 0.02).
        entry_price:          Expected entry price.
        stop_price:           Stop-loss price.
        current_margin_used:  Margin already in use across open positions ($).
        max_contracts:        Per-trade contract ceiling.

    Returns:
        Integer contract count (0 if trade should be skipped).

    Raises:
        KeyError: If symbol not found in InstrumentRegistry.
        ValueError: If instrument is not a FUTURES instrument.
    """
    instrument = get_registry().get(symbol)
    return calculate_futures_contracts(
        nav=nav,
        risk_pct=risk_pct,
        entry_price=entry_price,
        stop_price=stop_price,
        instrument=instrument,
        current_margin_used=current_margin_used,
        max_contracts=max_contracts,
    )


def dollar_risk_per_contract(instrument: Instrument, sl_distance: float) -> float:
    """Return the dollar P&L per contract if price moves ``sl_distance`` against us.

    Useful for display / logging without sizing a full trade.

    Args:
        instrument:   FUTURES instrument.
        sl_distance:  Absolute price distance (entry - stop) in native units.

    Returns:
        Dollar risk per contract.
    """
    if instrument.asset_class != "FUTURES":
        raise ValueError(f"{instrument.symbol} is not a FUTURES instrument")
    if instrument.tick_size is None or instrument.tick_value is None:
        raise ValueError(f"{instrument.symbol} has no tick_size/tick_value")

    ticks = sl_distance / instrument.tick_size
    return ticks * instrument.tick_value


def pnl_per_point(instrument: Instrument) -> float:
    """Return the dollar P&L per 1-point move for a single contract.

    For ES (tick_size=0.25, tick_value=$12.50): 1 point = 4 ticks = $50.
    For NQ (tick_size=0.25, tick_value=$5.00):  1 point = 4 ticks = $20.

    Args:
        instrument: FUTURES instrument.

    Returns:
        Dollar P&L per point per contract.
    """
    if instrument.asset_class != "FUTURES":
        raise ValueError(f"{instrument.symbol} is not a FUTURES instrument")
    if instrument.tick_size is None or instrument.tick_value is None:
        raise ValueError(f"{instrument.symbol} has no tick_size/tick_value")

    ticks_per_point = 1.0 / instrument.tick_size
    return ticks_per_point * instrument.tick_value
