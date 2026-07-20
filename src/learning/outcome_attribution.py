"""Deterministic outcome attribution — the arithmetic half of the learning loop.

This module answers "why did this position win or lose?" with arithmetic only.
No model, no LLM, no heuristics. It is the step that MUST run before any
reasoning layer sees a closed position, so that a reviewer interprets a
measured residual instead of inventing numbers.

Design rules (all load-bearing):

* **Pure.** No I/O, no broker calls, no clock. Inputs are dataclasses; output
  is a dataclass. Broker-specific parsing lives in an adapter (see
  ``src.learning.oanda_outcome_adapter``), never here.
* **Fail-closed on unreconciled arithmetic.** If the decomposition does not
  reproduce the broker's own realized P&L within tolerance, the report is
  marked ``reconciled=False`` and carries the gap. It is never silently
  rounded away, and a caller must not treat an unreconciled report as fact.
* **Never fabricate an unavailable component.** Sizing attribution requires
  the *intended* size from a ``DecisionRecord``. Where no decision was
  recorded (e.g. every historical trade predating the decision ledger),
  ``sizing_pl_home`` is ``None`` -- not zero. Zero would assert "sizing cost
  nothing", which is a different and unsupported claim.

The P&L identity implemented here was derived empirically, not assumed. It
reconciles across all 124 linkable closed slices in
``trained_data/oanda/transactions.jsonl`` with a worst-case absolute error of
5e-05 home currency (see ``tests/test_outcome_attribution.py``).

Sign conventions
----------------
``units`` is signed: positive is long, negative is short. ``gross_pl_home`` is
therefore ``(exit - entry) * units * conversion``, which is correctly negative
for a long that fell and positive for a short that fell.

Costs (``execution_cost_home``, ``fees_home``) are reported as POSITIVE
magnitudes representing money lost to friction. ``financing_home`` is signed:
positive is financing received, negative is financing paid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

# Absolute tolerance, in home currency, for calling the decomposition
# reconciled. Set from the observed float-precision noise floor (5e-05 across
# the 124-slice corpus) with headroom -- tight enough that a real modelling
# error cannot hide under it.
RECONCILE_TOLERANCE_HOME = 0.01

# Exit reasons that indicate a risk rail fired rather than a discretionary or
# signal-driven close. Used for honest bucketing, not for scoring.
RISK_EXIT_REASONS = frozenset({"STOP_LOSS_ORDER", "TRAILING_STOP_LOSS_ORDER", "MARGIN_CLOSEOUT"})
TARGET_EXIT_REASONS = frozenset({"TAKE_PROFIT_ORDER"})


class AttributionError(ValueError):
    """Raised when an outcome cannot be attributed at all.

    Distinct from an unreconciled report: an unreconciled report is a
    successfully-computed decomposition whose residual exceeds tolerance (a
    finding worth recording). This exception means the input was structurally
    unusable and no decomposition could be produced.
    """


@dataclass(frozen=True)
class DecisionRecord:
    """Why we traded — captured BEFORE execution, immutable afterwards.

    This is the half of the loop that today does not exist: without the
    intended size and the reason it was chosen, a closed position can only be
    scored on direction, never on sizing or gate quality.

    ``intended_units`` is what the strategy asked for *after* deterministic
    risk scaling; ``pre_degross_units`` is what it asked for before. The pair
    is what makes de-grossing measurable rather than invisible: the difference
    between them is the risk layer's realized contribution.
    """

    decision_id: str
    asof: str
    lane: str
    instrument: str
    intended_units: float
    signal_source: str
    pre_degross_units: Optional[float] = None
    degross_factor: Optional[float] = None
    risk_reasons: Tuple[str, ...] = ()
    gates_passed: Tuple[str, ...] = ()
    hedge_expected_units: Optional[float] = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise AttributionError("DecisionRecord requires a non-empty decision_id")
        if not self.instrument:
            raise AttributionError("DecisionRecord requires a non-empty instrument")


@dataclass(frozen=True)
class OutcomeRecord:
    """What actually happened — one closed (or partially closed) position slice.

    Field names are broker-neutral. Every monetary field is in HOME currency
    so the decomposition is additive without further conversion.

    ``realized_pl_home`` is the broker's own figure and is treated as ground
    truth; the decomposition is checked against it rather than replacing it.
    """

    trade_id: str
    instrument: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    units: float
    realized_pl_home: float
    conversion_factor: float = 1.0
    execution_cost_home: float = 0.0
    financing_home: float = 0.0
    commission_home: float = 0.0
    other_fees_home: float = 0.0
    conversion_cost_home: float = 0.0
    exit_reason: str = "UNKNOWN"
    decision_id: Optional[str] = None
    lane: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.trade_id:
            raise AttributionError("OutcomeRecord requires a non-empty trade_id")
        if self.units == 0:
            raise AttributionError(f"OutcomeRecord {self.trade_id}: units must be non-zero")


@dataclass(frozen=True)
class AttributionReport:
    """The deterministic answer, plus an explicit measure of what it can't explain.

    ``residual_home`` is the load-bearing field. It is the gap between the
    broker's realized P&L and the sum of the components this module can
    account for. A persistently large residual means the attribution MODEL is
    wrong -- that is a deterministic alarm to fix in code, never a question to
    hand to a reasoning layer.
    """

    trade_id: str
    instrument: str
    decision_id: Optional[str]
    gross_pl_home: float
    net_pl_home: float
    residual_home: float
    reconciled: bool
    price_move: float
    holding_period_seconds: Optional[float]
    exit_bucket: str
    direction_pl_home: Optional[float]
    sizing_pl_home: Optional[float]
    degross_pl_home: Optional[float]
    execution_cost_home: float
    financing_home: float
    fees_home: float
    conversion_cost_home: float
    unavailable: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable projection for the outcome ledger."""
        return {
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "decision_id": self.decision_id,
            "gross_pl_home": self.gross_pl_home,
            "net_pl_home": self.net_pl_home,
            "residual_home": self.residual_home,
            "reconciled": self.reconciled,
            "price_move": self.price_move,
            "holding_period_seconds": self.holding_period_seconds,
            "exit_bucket": self.exit_bucket,
            "direction_pl_home": self.direction_pl_home,
            "sizing_pl_home": self.sizing_pl_home,
            "degross_pl_home": self.degross_pl_home,
            "execution_cost_home": self.execution_cost_home,
            "financing_home": self.financing_home,
            "fees_home": self.fees_home,
            "conversion_cost_home": self.conversion_cost_home,
            "unavailable": list(self.unavailable),
        }


def _classify_exit(exit_reason: str) -> str:
    """Bucket an exit reason. Unknown reasons stay ``other`` rather than guessing."""
    reason = (exit_reason or "").upper()
    if reason in RISK_EXIT_REASONS:
        return "risk_rail"
    if reason in TARGET_EXIT_REASONS:
        return "target"
    if reason.startswith("MARKET_ORDER"):
        return "discretionary_or_signal"
    return "other"


def _holding_period_seconds(entry_time: str, exit_time: str) -> Optional[float]:
    """Seconds held, or None if either timestamp is unparseable.

    Returns None rather than 0.0 on failure -- a zero holding period is a
    meaningful (scalp) value and must not be confused with "unknown".
    """
    import re
    from datetime import datetime

    # OANDA emits NANOSECOND precision ("...T02:12:50.419308529Z") but
    # datetime.fromisoformat accepts at most 6 fractional digits. Truncate the
    # fraction only, leaving any timezone suffix untouched -- naively
    # collecting "all digits" would swallow the offset's digits too.
    _FRACTION = re.compile(r"\.(\d+)")

    def _parse(raw: str) -> Optional[datetime]:
        if not raw:
            return None
        text = raw.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        text = _FRACTION.sub(lambda m: "." + m.group(1)[:6], text, count=1)
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    start, end = _parse(entry_time), _parse(exit_time)
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def attribute(
    outcome: OutcomeRecord,
    decision: Optional[DecisionRecord] = None,
) -> AttributionReport:
    """Decompose one closed position into its deterministic drivers.

    Args:
        outcome: the realized result, in home currency.
        decision: the pre-trade intent, if it was recorded. When absent,
            sizing and de-gross attribution are reported as unavailable
            rather than assumed to be zero.

    Returns:
        An :class:`AttributionReport`. Check ``reconciled`` before using the
        components: an unreconciled report means the arithmetic did not close
        and the decomposition should not be trusted.

    Raises:
        AttributionError: if the outcome is structurally unusable.
    """
    if decision is not None and decision.instrument != outcome.instrument:
        raise AttributionError(
            f"decision/outcome instrument mismatch for trade {outcome.trade_id}: "
            f"{decision.instrument!r} vs {outcome.instrument!r}"
        )

    price_move = outcome.exit_price - outcome.entry_price
    gross_pl_home = price_move * outcome.units * outcome.conversion_factor

    fees_home = outcome.commission_home + outcome.other_fees_home
    net_pl_home = gross_pl_home + outcome.financing_home - fees_home

    # Reconcile the derived gross against the broker's own realized figure.
    # This is the check that makes every other number in the report credible.
    residual_home = outcome.realized_pl_home - gross_pl_home
    reconciled = abs(residual_home) <= RECONCILE_TOLERANCE_HOME

    unavailable: list[str] = []
    direction_pl_home: Optional[float] = None
    sizing_pl_home: Optional[float] = None
    degross_pl_home: Optional[float] = None

    if decision is None:
        # No recorded intent -> direction cannot be separated from sizing.
        # Reporting zero here would assert "sizing was free", which is a claim
        # the data does not support.
        unavailable.extend(["direction_pl_home", "sizing_pl_home", "degross_pl_home"])
    else:
        unit_pl = price_move * outcome.conversion_factor
        # Direction: what the intended size would have earned.
        direction_pl_home = unit_pl * decision.intended_units
        # Sizing: what the difference between filled and intended cost/earned.
        sizing_pl_home = unit_pl * (outcome.units - decision.intended_units)
        if decision.pre_degross_units is None:
            unavailable.append("degross_pl_home")
        else:
            # De-gross contribution: P&L the risk layer removed from the book.
            # Positive means de-grossing SAVED money (it cut a losing position).
            degross_pl_home = -unit_pl * (decision.pre_degross_units - decision.intended_units)

    return AttributionReport(
        trade_id=outcome.trade_id,
        instrument=outcome.instrument,
        decision_id=outcome.decision_id or (decision.decision_id if decision else None),
        gross_pl_home=gross_pl_home,
        net_pl_home=net_pl_home,
        residual_home=residual_home,
        reconciled=reconciled,
        price_move=price_move,
        holding_period_seconds=_holding_period_seconds(outcome.entry_time, outcome.exit_time),
        exit_bucket=_classify_exit(outcome.exit_reason),
        direction_pl_home=direction_pl_home,
        sizing_pl_home=sizing_pl_home,
        degross_pl_home=degross_pl_home,
        execution_cost_home=outcome.execution_cost_home,
        financing_home=outcome.financing_home,
        fees_home=fees_home,
        conversion_cost_home=outcome.conversion_cost_home,
        unavailable=tuple(unavailable),
    )


__all__ = [
    "AttributionError",
    "AttributionReport",
    "DecisionRecord",
    "OutcomeRecord",
    "RECONCILE_TOLERANCE_HOME",
    "RISK_EXIT_REASONS",
    "TARGET_EXIT_REASONS",
    "attribute",
]
