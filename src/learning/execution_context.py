"""ExecutionContext — where a weight intention becomes executable units.

Sits between a persisted, authorised DecisionRecord and the broker call:

    persisted DecisionRecord
    -> validate executable decision
    -> obtain NAV, holdings and quote
    -> convert weights to units
    -> append execution-context row      <- THIS MODULE
    -> submit order
    -> append broker-link row

The decision seam is weight-native on purpose: NAV, price, holdings and lot
rules do not exist there. They exist here. Converting at this layer keeps
execution concerns out of strategy evaluation and keeps DecisionRecord
immutable -- the context is a NEW append-only row, never a mutation.

Raw vs broker-valid quantities
------------------------------

Both are preserved. ``raw_target_units`` is what the capital base and price
imply; ``intended_units`` is what the broker will actually accept after
quantization. The gap between them is sizing that changed for LOT reasons
rather than strategy or risk reasons, and it is invisible unless both are
recorded.

A non-zero weight delta that quantizes to zero units is an explicit
``NO_ORDER_AFTER_QUANTIZATION`` outcome, never an anonymous successful no-op.

Cost estimate scope
-------------------

Only what the quote supports: entry half-spread and commission. Exit and
round-trip estimates stay ``None`` and are named in ``unavailable_fields``
until an explicit exit-cost model exists. Manufacturing a round-trip number
from an entry quote would be a fabricated input to future calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from src.evidence.hashing import content_digest
from src.learning.decision_record import RECORD_KIND_DECISION

EXECUTION_CONTEXT_ROW_TYPE = "execution_context"
EXECUTION_CONTEXT_EVENT = "EXECUTION_CONTEXT_CREATED"
CONTEXT_SCHEMA_VERSION = 1

# Reference-price semantics. A last trade is NOT a mid and must never be
# labelled one -- attribution against a mislabelled reference is silently wrong.
REF_MID = "MID"
REF_ASK = "ASK"
REF_BID = "BID"
REF_LAST = "LAST"

# Constraint reason codes.
FRACTIONAL_NOT_ALLOWED = "FRACTIONAL_NOT_ALLOWED"
ROUNDED_TO_WHOLE_SHARE = "ROUNDED_TO_WHOLE_SHARE"
BELOW_MINIMUM_ORDER = "BELOW_MINIMUM_ORDER"
INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"
ZERO_AFTER_QUANTIZATION = "ZERO_AFTER_QUANTIZATION"
NO_ORDER_AFTER_QUANTIZATION = "NO_ORDER_AFTER_QUANTIZATION"

#: Fields that identify one context. A replay of the same decision, plan and
#: material market snapshot must reproduce this digest exactly.
#:
#: ``bid``/``ask`` are material alongside ``reference_price`` because two quotes
#: can share a mid while differing in spread -- 99/101 and 98/102 both mid at
#: 100 but imply different entry costs. Keying on the mid alone would collapse
#: them into one context and lose the cost difference.
CONTEXT_MATERIAL_FIELDS: Tuple[str, ...] = (
    "decision_digest", "plan_id", "instrument",
    "allocatable_nav", "reference_price", "bid", "ask", "current_units",
    "intended_units", "order_units",
)


class ExecutionContextError(ValueError):
    """The context could not be constructed. No broker call may follow."""


@dataclass(frozen=True)
class QuoteSnapshot:
    """The quote the conversion actually used."""

    quote_source: str
    quote_timestamp: str
    quote_received_at: str
    bid: float
    ask: float
    reference_price: float
    reference_price_type: str
    quote_age_ms: Optional[float] = None
    #: Populated only when the sizing reference differs from mid (e.g. a buy
    #: sized at the ask for affordability while attributed against mid).
    execution_reference_price: Optional[float] = None
    execution_reference_price_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ExecutionContextError(f"non-positive quote bid={self.bid} ask={self.ask}")
        if self.ask < self.bid:
            raise ExecutionContextError(f"crossed quote bid={self.bid} > ask={self.ask}")
        if self.reference_price <= 0:
            raise ExecutionContextError(f"non-positive reference_price {self.reference_price}")
        if self.reference_price_type not in (REF_MID, REF_ASK, REF_BID, REF_LAST):
            raise ExecutionContextError(
                f"unknown reference_price_type {self.reference_price_type!r}"
            )

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_absolute(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return (self.spread_absolute / self.mid) * 10_000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quote_source": self.quote_source,
            "quote_timestamp": self.quote_timestamp,
            "quote_received_at": self.quote_received_at,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "spread_absolute": self.spread_absolute,
            "spread_bps": self.spread_bps,
            "reference_price": self.reference_price,
            "reference_price_type": self.reference_price_type,
            "quote_age_ms": self.quote_age_ms,
            "execution_reference_price": self.execution_reference_price,
            "execution_reference_price_type": self.execution_reference_price_type,
        }


@dataclass(frozen=True)
class LotConstraints:
    """Broker quantity rules applied to the raw sizing."""

    lot_size: float = 1.0
    minimum_order_units: float = 0.0
    quantity_increment: float = 1.0
    fractional_allowed: bool = False
    rounding_mode: str = "FLOOR_ABS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lot_size": self.lot_size,
            "minimum_order_units": self.minimum_order_units,
            "quantity_increment": self.quantity_increment,
            "fractional_allowed": self.fractional_allowed,
            "rounding_mode": self.rounding_mode,
        }


def quantize_units(raw: float, lot: LotConstraints) -> float:
    """Deterministically snap a raw quantity to a broker-valid one.

    ``FLOOR_ABS`` floors the MAGNITUDE and keeps the sign, so quantization can
    only ever reduce exposure. Rounding a short's magnitude up would silently
    increase risk to satisfy a lot rule.
    """
    if raw == 0:
        return 0.0
    increment = lot.quantity_increment if lot.fractional_allowed else max(lot.lot_size, 1.0)
    if increment <= 0:
        raise ExecutionContextError(f"non-positive quantity increment {increment}")
    magnitude = abs(raw)
    if lot.rounding_mode == "FLOOR_ABS":
        snapped = math.floor(magnitude / increment) * increment
    elif lot.rounding_mode == "NEAREST":
        snapped = round(magnitude / increment) * increment
    else:
        raise ExecutionContextError(f"unknown rounding_mode {lot.rounding_mode!r}")
    # Kill floating-point dust from the divide/multiply round trip.
    snapped = round(snapped, 10)
    return math.copysign(snapped, raw)


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable record of the weight -> units conversion for one instrument."""

    # identity
    event_id: str
    decision_id: str
    decision_digest: str
    cycle_id: str
    plan_id: str
    instrument: str
    created_at: str

    # capital base -- portfolio_nav is NOT assumed equal to allocatable_nav
    portfolio_nav: float
    allocatable_nav: float
    cash_available: Optional[float]

    # sizing inputs carried forward from the decision
    pre_degross_weight: float
    target_weight: float
    weight_delta: float
    degross_factor: float
    current_units: float

    # notional + raw (pre-quantization) quantities
    pre_degross_notional: float
    target_notional: float
    current_notional: float
    raw_pre_degross_units: float
    raw_target_units: float

    # broker-valid quantities
    pre_degross_units: float
    intended_units: float
    order_units: float
    rounded_units_delta: float

    quote: QuoteSnapshot
    lot: LotConstraints
    constraint_reason_codes: Tuple[str, ...] = ()

    # entry-only cost estimate
    estimated_entry_half_spread_bps: Optional[float] = None
    estimated_entry_spread_cost: Optional[float] = None
    estimated_commission: Optional[float] = None
    estimated_entry_cost_bps: Optional[float] = None
    cost_estimate_availability: str = "entry_only"

    # deliberately unavailable until an exit-cost model exists
    estimated_exit_cost_bps: Optional[float] = None
    estimated_total_cost_bps: Optional[float] = None
    expected_net_edge_bps: Optional[float] = None
    unavailable_fields: Tuple[str, ...] = ()

    context_schema_version: int = CONTEXT_SCHEMA_VERSION
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_digest:
            raise ExecutionContextError("decision_digest is required")
        if not self.plan_id:
            raise ExecutionContextError("plan_id is required")
        for name in ("estimated_exit_cost_bps", "estimated_total_cost_bps", "expected_net_edge_bps"):
            if getattr(self, name) is not None and name in self.unavailable_fields:
                raise ExecutionContextError(f"{name} declared unavailable but has a value")

    def material(self) -> Dict[str, Any]:
        """Identity fields. ``reference_price`` is sourced from the quote --
        it is the market snapshot the conversion actually used, so a changed
        quote must produce a distinguishable context."""
        out: Dict[str, Any] = {}
        for name in CONTEXT_MATERIAL_FIELDS:
            if name in ("reference_price", "bid", "ask"):
                out[name] = getattr(self.quote, name)
            else:
                out[name] = getattr(self, name)
        return out

    def digest(self) -> str:
        return content_digest(self.material())

    def to_row(self) -> Dict[str, Any]:
        return {
            "record_kind": RECORD_KIND_DECISION,
            "row_type": EXECUTION_CONTEXT_ROW_TYPE,
            "event_type": EXECUTION_CONTEXT_EVENT,
            "context_schema_version": self.context_schema_version,
            # The ledger dedupes on decision_digest; namespace the context so a
            # context row can never collide with its own decision row.
            "decision_digest": f"ctx:{self.digest()}",
            "context_digest": self.digest(),
            "links_decision_digest": self.decision_digest,
            "event_id": self.event_id,
            "decision_id": self.decision_id,
            "cycle_id": self.cycle_id,
            "plan_id": self.plan_id,
            "instrument": self.instrument,
            "created_at": self.created_at,
            "portfolio_nav": self.portfolio_nav,
            "allocatable_nav": self.allocatable_nav,
            "cash_available": self.cash_available,
            "pre_degross_weight": self.pre_degross_weight,
            "target_weight": self.target_weight,
            "weight_delta": self.weight_delta,
            "degross_factor": self.degross_factor,
            "current_units": self.current_units,
            "pre_degross_notional": self.pre_degross_notional,
            "target_notional": self.target_notional,
            "current_notional": self.current_notional,
            "raw_pre_degross_units": self.raw_pre_degross_units,
            "raw_target_units": self.raw_target_units,
            "pre_degross_units": self.pre_degross_units,
            "intended_units": self.intended_units,
            "order_units": self.order_units,
            "rounded_units_delta": self.rounded_units_delta,
            "quote": self.quote.to_dict(),
            "lot": self.lot.to_dict(),
            "constraint_reason_codes": list(self.constraint_reason_codes),
            "estimated_entry_half_spread_bps": self.estimated_entry_half_spread_bps,
            "estimated_entry_spread_cost": self.estimated_entry_spread_cost,
            "estimated_commission": self.estimated_commission,
            "estimated_entry_cost_bps": self.estimated_entry_cost_bps,
            "cost_estimate_availability": self.cost_estimate_availability,
            "estimated_exit_cost_bps": self.estimated_exit_cost_bps,
            "estimated_total_cost_bps": self.estimated_total_cost_bps,
            "expected_net_edge_bps": self.expected_net_edge_bps,
            "unavailable_fields": list(self.unavailable_fields),
            "notes": dict(self.notes),
        }


__all__ = [
    "BELOW_MINIMUM_ORDER",
    "CONTEXT_MATERIAL_FIELDS",
    "CONTEXT_SCHEMA_VERSION",
    "EXECUTION_CONTEXT_EVENT",
    "EXECUTION_CONTEXT_ROW_TYPE",
    "FRACTIONAL_NOT_ALLOWED",
    "INSUFFICIENT_BUYING_POWER",
    "NO_ORDER_AFTER_QUANTIZATION",
    "REF_ASK", "REF_BID", "REF_LAST", "REF_MID",
    "ROUNDED_TO_WHOLE_SHARE",
    "ZERO_AFTER_QUANTIZATION",
    "ExecutionContext",
    "ExecutionContextError",
    "LotConstraints",
    "QuoteSnapshot",
    "quantize_units",
]
