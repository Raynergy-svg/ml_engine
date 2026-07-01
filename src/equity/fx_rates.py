"""Shared FX-rate conversion helpers for the OANDA trend lane.

Split out of ``oanda_trend.py`` so ``trend_risk_gates.py`` can import
``base_to_home_rate`` without a circular import (``oanda_trend`` imports the
risk gate module; the risk gate needs this rate helper).
"""
from __future__ import annotations

from typing import Dict, Optional


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
