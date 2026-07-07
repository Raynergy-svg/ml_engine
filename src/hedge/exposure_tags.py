"""AXIOM Hedge Layer — Phase 1: exposure tagging (SHADOW / ANALYSIS-ONLY).

Decomposes a single trade candidate (or an already-open shadow position) into
its underlying exposure legs:

* FX — split a pair into its two currency legs (Long EUR/USD = +EUR/-USD),
  each tagged risk-on / risk-off (safe-haven) / commodity-linked. Currency-leg
  parsing REUSES ``src.equity.trend_risk_gates.currency_legs`` (the existing
  risk gate's single source of truth for how a pair maps to its two
  currencies) rather than forking that definition.
* Equity — tag a ticker's sector, market beta, factor exposure, and benchmark
  hedge instrument from a config map.

Every tagging function is a PURE function (no network I/O, no broker calls,
no state.json / halt / gate touch) — mirrors the no-mock, real-disk-only
testing discipline used by ``trend_risk_gates.py``.

Fail-closed contract: a currency or ticker absent from its config map is
NEVER silently treated as zero exposure. The tag comes back with
``fail_closed=True`` and a ``reasons`` tuple naming exactly what could not be
resolved; callers (``portfolio_exposure.py``) propagate that into the
portfolo-level report rather than hiding the gap.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.equity.trend_risk_gates import currency_legs  # reuse, do not fork

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent / "config"
CURRENCY_BUCKET_MAP_PATH = _CONFIG_DIR / "currency_bucket_map.json"
SECTOR_BUCKET_MAP_PATH = _CONFIG_DIR / "sector_bucket_map.json"
CORRELATION_BUCKET_MAP_PATH = _CONFIG_DIR / "correlation_bucket_map.json"

VALID_DIRECTIONS = ("long", "short")


def load_bucket_map(path: Path) -> Dict[str, dict]:
    """Load a bucket-map JSON config. Missing/corrupt file -> {} (JSON Safety
    Gates), which — by construction — makes every subsequent lookup miss and
    fail closed rather than silently assuming zero exposure. The ``_meta`` key
    (documentation only) is stripped so callers never mistake it for a real
    currency/ticker entry."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning(
            "hedge: failed to load bucket map %s (%s) — treating as empty "
            "(every lookup will fail closed)", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("hedge: bucket map %s is not a JSON object — treating as empty", path)
        return {}
    return {k: v for k, v in raw.items() if k != "_meta"}


# --------------------------------------------------------------------- #
# FX                                                                     #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class CurrencyLegTag:
    """One currency leg of an FX exposure, with its signed home-currency risk
    (positive = net long that currency, negative = net short)."""
    currency: str
    direction: str          # "long" | "short" (this leg's own direction)
    tags: Tuple[str, ...]   # e.g. ("safe_haven",) or ("risk_on", "commodity_linked")
    signed_home: float


@dataclass(frozen=True)
class FxExposureTag:
    instrument: str
    direction: str
    risk_home: float
    legs: Tuple[CurrencyLegTag, ...] = ()
    fail_closed: bool = False
    reasons: Tuple[str, ...] = ()


def tag_fx_exposure(instrument: str, direction: str, risk_home: float,
                    currency_bucket_map: Optional[Dict[str, dict]] = None) -> FxExposureTag:
    """Decompose an FX instrument + direction + dollar risk into its two
    signed currency legs. Long <BASE>_<QUOTE> = long BASE / short QUOTE;
    short flips both. Fails closed (empty legs, ``fail_closed=True``) on an
    unresolvable instrument, an invalid direction, a non-finite/negative
    risk_home, or a currency absent from ``currency_bucket_map``."""
    if currency_bucket_map is None:
        currency_bucket_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    reasons = []
    if direction not in VALID_DIRECTIONS:
        reasons.append(f"invalid_direction:{direction}")
    try:
        risk_home = float(risk_home)
    except (TypeError, ValueError):
        reasons.append(f"invalid_risk_home:{risk_home}")
        risk_home = 0.0
    if risk_home < 0:
        reasons.append(f"negative_risk_home:{risk_home}")
    legs_parts = currency_legs(instrument)
    if legs_parts is None:
        reasons.append(f"unresolvable_instrument:{instrument}")
        return FxExposureTag(instrument=instrument, direction=direction, risk_home=risk_home,
                             legs=(), fail_closed=True, reasons=tuple(reasons))
    base, quote = legs_parts
    if reasons:
        # Direction/risk already broken — no point resolving currency tags too.
        return FxExposureTag(instrument=instrument, direction=direction, risk_home=risk_home,
                             legs=(), fail_closed=True, reasons=tuple(reasons))

    base_dir = "long" if direction == "long" else "short"
    quote_dir = "short" if direction == "long" else "long"
    legs = []
    for ccy, ccy_dir in ((base, base_dir), (quote, quote_dir)):
        entry = currency_bucket_map.get(ccy)
        if entry is None:
            reasons.append(f"unknown_currency:{ccy}")
            continue
        signed = risk_home if ccy_dir == "long" else -risk_home
        legs.append(CurrencyLegTag(currency=ccy, direction=ccy_dir,
                                   tags=tuple(entry.get("tags", ())), signed_home=signed))
    if reasons:
        return FxExposureTag(instrument=instrument, direction=direction, risk_home=risk_home,
                             legs=(), fail_closed=True, reasons=tuple(reasons))
    return FxExposureTag(instrument=instrument, direction=direction, risk_home=risk_home,
                         legs=tuple(legs), fail_closed=False, reasons=())


# --------------------------------------------------------------------- #
# Equity                                                                 #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class EquityExposureTag:
    ticker: str
    direction: str
    risk_home: float
    sector: Optional[str] = None
    market_beta: Optional[float] = None
    factor_tags: Tuple[str, ...] = field(default_factory=tuple)
    benchmark_hedge: Optional[str] = None
    signed_home: float = 0.0          # +risk_home if long, -risk_home if short
    signed_beta_home: Optional[float] = None  # signed_home * market_beta
    fail_closed: bool = False
    reasons: Tuple[str, ...] = ()


def tag_equity_exposure(ticker: str, direction: str, risk_home: float,
                        sector_bucket_map: Optional[Dict[str, dict]] = None) -> EquityExposureTag:
    """Decompose an equity ticker + direction + dollar risk into sector,
    market-beta, and factor exposure. Fails closed on an unknown ticker,
    invalid direction, or non-finite/negative risk_home."""
    if sector_bucket_map is None:
        sector_bucket_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    reasons = []
    if direction not in VALID_DIRECTIONS:
        reasons.append(f"invalid_direction:{direction}")
    try:
        risk_home = float(risk_home)
    except (TypeError, ValueError):
        reasons.append(f"invalid_risk_home:{risk_home}")
        risk_home = 0.0
    if risk_home < 0:
        reasons.append(f"negative_risk_home:{risk_home}")
    if reasons:
        return EquityExposureTag(ticker=ticker, direction=direction, risk_home=risk_home,
                                 fail_closed=True, reasons=tuple(reasons))

    entry = sector_bucket_map.get(ticker)
    if entry is None:
        return EquityExposureTag(ticker=ticker, direction=direction, risk_home=risk_home,
                                 fail_closed=True, reasons=(f"unknown_ticker:{ticker}",))

    signed_home = risk_home if direction == "long" else -risk_home
    beta = entry.get("market_beta")
    signed_beta_home = signed_home * float(beta) if beta is not None else None
    return EquityExposureTag(
        ticker=ticker, direction=direction, risk_home=risk_home,
        sector=entry.get("sector"), market_beta=beta,
        factor_tags=tuple(entry.get("factor_tags", ())),
        benchmark_hedge=entry.get("benchmark_hedge"),
        signed_home=signed_home, signed_beta_home=signed_beta_home,
        fail_closed=False, reasons=())
