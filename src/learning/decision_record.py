"""DecisionRecord — why a trade was (or was not) attempted, captured pre-execution.

This is the instrument that makes loss attributable. Without it a closed
position can only be scored on direction; sizing, de-grossing, and gate quality
are unrecoverable. On the current realized corpus, 0 of 124 rows carry one, so
``direction_pl_home`` / ``sizing_pl_home`` / ``degross_pl_home`` are
``unavailable`` on every single row.

Two structurally distinct classes
---------------------------------

``EXECUTED_DECISION``       — disposition EXECUTE, an order will be submitted
``CANDIDATE_NOT_EXECUTED``  — disposition NO_ACT / ABSTAIN / HALT

The pairing is enforced in ``__post_init__``: EXECUTE cannot carry the
candidate class and no other disposition can carry the executed class. A
candidate record can therefore never be mistaken for evidence that capital
moved, which is the entire point of separating them.

Candidates are the reason instrumentation pays off immediately. A halted lane
produces no fills, but it still produces signal, cost estimate, sizing
intention and a refusal reason — and later prices can turn those into labelled
counterfactual observations. Those observations are NEVER realized P&L and must
never enter the realized ledger; ``record_kind`` exists so that boundary is
mechanical rather than a matter of care.

Unavailable is not zero
-----------------------

Forecast and quote fields are ``None`` when the loop cannot supply them, and
every such field is named in ``unavailable_fields``. Zero would assert "the
expected move was nil" or "the spread was free", which are claims, not absences.

Weights vs units
----------------

The equity control loop is weight-based end to end (``Order`` carries
``weight_delta`` / ``target_weight``; there are no units and no prices at the
decision seam). Weights are therefore the native, always-present intention.
Unit fields are populated only when a converter supplies a price, and are
reported unavailable otherwise rather than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from src.evidence.hashing import content_digest

# Marks which ledger a row belongs to. The realized ledger refuses rows carrying
# the decision kind and vice versa, so the two corpora cannot be blended by a
# careless caller.
RECORD_KIND_DECISION = "decision"
RECORD_KIND_REALIZED = "realized_outcome"

DECISION_SCHEMA_VERSION = 1


class DecisionClass(str, Enum):
    EXECUTED_DECISION = "EXECUTED_DECISION"
    CANDIDATE_NOT_EXECUTED = "CANDIDATE_NOT_EXECUTED"


class Disposition(str, Enum):
    EXECUTE = "EXECUTE"
    NO_ACT = "NO_ACT"
    ABSTAIN = "ABSTAIN"
    HALT = "HALT"


#: Dispositions that mean no order will be submitted this cycle.
CANDIDATE_DISPOSITIONS = frozenset({Disposition.NO_ACT, Disposition.ABSTAIN, Disposition.HALT})

#: Fields covered by the identity digest. A retry of the same cycle must
#: reproduce this exactly; a materially different decision must not.
MATERIAL_FIELDS: Tuple[str, ...] = (
    "cycle_id",
    "instrument",
    "direction",
    "market_data_asof",
    "strategy_version",
    "pre_degross_units",
    "pre_degross_weight",
    "intended_units",
    "intended_weight",
    "disposition",
)


class DecisionRecordError(ValueError):
    """Structurally invalid decision record."""


@dataclass(frozen=True)
class DecisionRecord:
    """One instrument's decision within one cycle."""

    # identity / classification
    decision_id: str
    cycle_id: str
    decision_class: DecisionClass
    disposition: Disposition
    reason_codes: Tuple[str, ...]
    instrument: str
    direction: str                      # "LONG" | "SHORT" | "FLAT"
    decision_timestamp: str

    # provenance
    market_data_asof: Optional[str] = None
    strategy_version: Optional[str] = None
    risk_policy_version: Optional[str] = None
    input_digest: Optional[str] = None

    # quotes / cost estimate -- None when the loop has no quote source
    reference_mid: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_bps: Optional[float] = None
    estimated_entry_cost_bps: Optional[float] = None
    estimated_exit_cost_bps: Optional[float] = None
    estimated_total_cost_bps: Optional[float] = None

    # forecast -- None when the strategy emits no explicit expectation
    expected_move_bps: Optional[float] = None
    expected_move_horizon: Optional[str] = None
    expected_move_confidence: Optional[float] = None
    expected_net_edge_bps: Optional[float] = None

    # sizing: weights are native to the equity loop, units need a price
    current_weight: Optional[float] = None
    pre_degross_weight: Optional[float] = None
    intended_weight: Optional[float] = None
    weight_delta: Optional[float] = None
    current_units: Optional[float] = None
    pre_degross_units: Optional[float] = None
    intended_units: Optional[float] = None
    order_delta: Optional[float] = None
    degross_factor: Optional[float] = None

    # portfolio context
    portfolio_nav: Optional[float] = None
    gross_exposure_before: Optional[float] = None
    gross_exposure_after: Optional[float] = None
    drawdown_state: Optional[float] = None

    # linkage -- filled in after submission, never at construction
    broker_order_id: Optional[str] = None
    actual_filled_units: Optional[float] = None
    client_order_id: Optional[str] = None

    # explicit availability metadata
    unavailable_fields: Tuple[str, ...] = ()
    lane: Optional[str] = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise DecisionRecordError("decision_id is required")
        if not self.cycle_id:
            raise DecisionRecordError("cycle_id is required")
        if not self.instrument:
            raise DecisionRecordError("instrument is required")

        # The class/disposition pairing is the structural guarantee that a
        # candidate can never masquerade as an execution.
        if self.disposition is Disposition.EXECUTE:
            if self.decision_class is not DecisionClass.EXECUTED_DECISION:
                raise DecisionRecordError(
                    "disposition EXECUTE requires decision_class EXECUTED_DECISION"
                )
        elif self.decision_class is not DecisionClass.CANDIDATE_NOT_EXECUTED:
            raise DecisionRecordError(
                f"disposition {self.disposition.value} requires decision_class "
                "CANDIDATE_NOT_EXECUTED"
            )

        # A candidate has not been submitted, so it cannot carry broker linkage.
        if self.decision_class is DecisionClass.CANDIDATE_NOT_EXECUTED:
            if self.broker_order_id is not None or self.actual_filled_units is not None:
                raise DecisionRecordError(
                    "CANDIDATE_NOT_EXECUTED cannot carry broker_order_id or "
                    "actual_filled_units -- nothing was submitted"
                )

        # A field cannot be both populated and declared unavailable.
        for name in self.unavailable_fields:
            if getattr(self, name, None) is not None:
                raise DecisionRecordError(
                    f"field {name!r} is declared unavailable but has a value"
                )

    # -- serialisation ---------------------------------------------------

    def material(self) -> Dict[str, Any]:
        """The subset that identifies this decision."""
        out: Dict[str, Any] = {}
        for name in MATERIAL_FIELDS:
            value = getattr(self, name)
            out[name] = value.value if isinstance(value, Enum) else value
        return out

    def digest(self) -> str:
        """Deterministic identity. Same cycle + same decision -> same digest."""
        return content_digest(self.material())

    def to_row(self) -> Dict[str, Any]:
        """JSON-serialisable ledger row."""
        payload: Dict[str, Any] = {
            "record_kind": RECORD_KIND_DECISION,
            "schema_version": DECISION_SCHEMA_VERSION,
            "decision_digest": self.digest(),
        }
        for key, value in self.__dict__.items():
            payload[key] = value.value if isinstance(value, Enum) else (
                list(value) if isinstance(value, tuple) else
                dict(value) if isinstance(value, Mapping) else value
            )
        return payload


def unavailable_from(**candidates: Any) -> Tuple[str, ...]:
    """Names of the supplied fields whose value is None.

    Helper so callers declare availability from the same values they pass,
    rather than maintaining a parallel list that can silently drift.
    """
    return tuple(sorted(name for name, value in candidates.items() if value is None))


__all__ = [
    "CANDIDATE_DISPOSITIONS",
    "DECISION_SCHEMA_VERSION",
    "MATERIAL_FIELDS",
    "RECORD_KIND_DECISION",
    "RECORD_KIND_REALIZED",
    "DecisionClass",
    "DecisionRecord",
    "DecisionRecordError",
    "Disposition",
    "unavailable_from",
]
