"""Build an ExecutionContext, and the submission gate that guards the fleet.

Two responsibilities that belong together because both sit on the submission
boundary:

1. ``build_execution_context`` — the weight -> units conversion, with every
   input it used recorded alongside the output.
2. ``SubmissionGate`` — durable state that closes further submissions after a
   broker order is left without durable local linkage.

The gate's rationale
--------------------

If a broker accepts an order and the linkage write then fails, cancelling
blindly is worse than leaving it: the position is real and a blind cancel can
race a fill. But continuing to submit is also wrong -- each further order can
create another unlinked broker position, compounding an already-unreconciled
book. So: the existing order stands, an incident is raised, the gate closes,
and submissions resume only after linkage is recovered from broker state.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.learning.decision_record import DecisionClass, DecisionRecord
from src.learning.execution_context import (
    BELOW_MINIMUM_ORDER,
    FRACTIONAL_NOT_ALLOWED,
    INSUFFICIENT_BUYING_POWER,
    NO_ORDER_AFTER_QUANTIZATION,
    ROUNDED_TO_WHOLE_SHARE,
    ZERO_AFTER_QUANTIZATION,
    ExecutionContext,
    ExecutionContextError,
    LotConstraints,
    QuoteSnapshot,
    quantize_units,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_GATE_PATH = REPO_ROOT / "trained_data" / "learning" / "submission_gate.json"

#: A quote older than this cannot size an order. Deterministic and explicit --
#: "recent enough" must not be a judgement call at the submission boundary.
DEFAULT_MAX_QUOTE_AGE_MS = 5_000.0

LOG_TAG_UNLINKED_INCIDENT = "UNLINKED_BROKER_ORDER_INCIDENT"


def build_execution_context(
    *,
    decision: DecisionRecord,
    decision_digest: str,
    plan_id: str,
    created_at: str,
    quote: QuoteSnapshot,
    portfolio_nav: float,
    allocatable_nav: Optional[float] = None,
    cash_available: Optional[float] = None,
    current_units: float = 0.0,
    lot: Optional[LotConstraints] = None,
    commission_estimate: Optional[float] = None,
    max_quote_age_ms: float = DEFAULT_MAX_QUOTE_AGE_MS,
) -> ExecutionContext:
    """Convert a persisted weight decision into broker-valid units.

    Raises:
        ExecutionContextError: on any condition that must stop the broker call
            -- non-executable decision, stale quote, invalid NAV, bad quote.
    """
    if decision.decision_class is not DecisionClass.EXECUTED_DECISION:
        raise ExecutionContextError(
            f"decision {decision.decision_id} is {decision.decision_class.value}; "
            "only an EXECUTED_DECISION may receive executable context"
        )
    if portfolio_nav is None or not (portfolio_nav > 0) or portfolio_nav != portfolio_nav:
        raise ExecutionContextError(f"invalid portfolio_nav {portfolio_nav!r}")

    # Never assume the scheduler deploys the whole NAV.
    capital_base = float(allocatable_nav if allocatable_nav is not None else portfolio_nav)
    if not (capital_base > 0) or capital_base != capital_base:
        raise ExecutionContextError(f"invalid allocatable_nav {allocatable_nav!r}")

    if quote.quote_age_ms is not None and quote.quote_age_ms > max_quote_age_ms:
        raise ExecutionContextError(
            f"stale quote: age {quote.quote_age_ms}ms exceeds limit {max_quote_age_ms}ms"
        )

    lot = lot or LotConstraints()
    reasons: List[str] = []

    pre_degross_weight = float(decision.pre_degross_weight or 0.0)
    target_weight = float(decision.intended_weight or 0.0)
    reference_price = float(quote.reference_price)

    pre_degross_notional = capital_base * pre_degross_weight
    target_notional = capital_base * target_weight
    current_notional = float(current_units) * reference_price

    raw_pre_degross_units = pre_degross_notional / reference_price
    raw_target_units = target_notional / reference_price

    pre_degross_units = quantize_units(raw_pre_degross_units, lot)
    intended_units = quantize_units(raw_target_units, lot)
    order_units = intended_units - float(current_units)

    if not lot.fractional_allowed:
        reasons.append(FRACTIONAL_NOT_ALLOWED)
        if raw_target_units != intended_units:
            reasons.append(ROUNDED_TO_WHOLE_SHARE)
    rounded_units_delta = intended_units - raw_target_units

    if raw_target_units != 0 and intended_units == 0:
        reasons.append(ZERO_AFTER_QUANTIZATION)
    # A real weight change that produces no order must say so explicitly rather
    # than look like a successful no-op.
    if float(decision.weight_delta or 0.0) != 0.0 and order_units == 0:
        reasons.append(NO_ORDER_AFTER_QUANTIZATION)
    if lot.minimum_order_units and 0 < abs(order_units) < lot.minimum_order_units:
        reasons.append(BELOW_MINIMUM_ORDER)
    if cash_available is not None and order_units > 0:
        exec_price = quote.execution_reference_price or quote.ask
        if abs(order_units) * exec_price > float(cash_available):
            reasons.append(INSUFFICIENT_BUYING_POWER)

    half_spread_bps = quote.spread_bps / 2.0
    notional_traded = abs(order_units) * reference_price
    spread_cost = notional_traded * (half_spread_bps / 10_000.0)
    commission = float(commission_estimate) if commission_estimate is not None else None
    entry_cost_bps = half_spread_bps
    if commission is not None and notional_traded > 0:
        entry_cost_bps = half_spread_bps + (commission / notional_traded) * 10_000.0

    unavailable = ["estimated_exit_cost_bps", "estimated_total_cost_bps", "expected_net_edge_bps"]
    if commission is None:
        unavailable.append("estimated_commission")

    return ExecutionContext(
        event_id=f"ctx:{decision.decision_id}:{plan_id}",
        decision_id=decision.decision_id,
        decision_digest=decision_digest,
        cycle_id=decision.cycle_id,
        plan_id=plan_id,
        instrument=decision.instrument,
        created_at=created_at,
        portfolio_nav=float(portfolio_nav),
        allocatable_nav=capital_base,
        cash_available=None if cash_available is None else float(cash_available),
        pre_degross_weight=pre_degross_weight,
        target_weight=target_weight,
        weight_delta=float(decision.weight_delta or 0.0),
        degross_factor=float(decision.degross_factor or 0.0),
        current_units=float(current_units),
        pre_degross_notional=pre_degross_notional,
        target_notional=target_notional,
        current_notional=current_notional,
        raw_pre_degross_units=raw_pre_degross_units,
        raw_target_units=raw_target_units,
        pre_degross_units=pre_degross_units,
        intended_units=intended_units,
        order_units=order_units,
        rounded_units_delta=rounded_units_delta,
        quote=quote,
        lot=lot,
        constraint_reason_codes=tuple(dict.fromkeys(reasons)),
        estimated_entry_half_spread_bps=half_spread_bps,
        estimated_entry_spread_cost=spread_cost,
        estimated_commission=commission,
        estimated_entry_cost_bps=entry_cost_bps,
        unavailable_fields=tuple(sorted(unavailable)),
    )


class SubmissionGate:
    """Durable open/closed state for order submission.

    Closed by an unlinked-broker-order incident. Deliberately NOT
    auto-reopening: reopening requires linkage to be recovered from broker
    state, which is a reconciliation step, not a timeout.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else SUBMISSION_GATE_PATH

    def is_open(self) -> bool:
        """True when submissions are permitted. Unreadable state fails CLOSED."""
        if not self.path.exists():
            return True
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "submission gate state unreadable (%s) -- failing CLOSED", exc
            )
            return False
        if not isinstance(payload, dict):
            logger.error("submission gate state malformed -- failing CLOSED")
            return False
        return bool(payload.get("open", True))

    def state(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"open": True}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"open": False}
        except (json.JSONDecodeError, OSError):
            return {"open": False}

    def _write(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def raise_unlinked_incident(
        self,
        *,
        broker_order_id: str,
        decision_id: str,
        detail: str,
        at: str,
    ) -> Dict[str, Any]:
        """Close the gate after a broker order was left without linkage.

        The accepted order is NOT cancelled -- the position is real and a blind
        cancel can race a fill. But no further order may be created until the
        book is reconciled, or one unlinked position becomes many.
        """
        payload = {
            "open": False,
            "closed_at": at,
            "incident": "UNLINKED_BROKER_ORDER",
            "severity": "CRITICAL",
            "broker_order_id": broker_order_id,
            "decision_id": decision_id,
            "detail": detail,
            "recovery": (
                "Recover linkage from broker account/order state, repair it "
                "append-only, then reopen. Do NOT auto-resubmit; the existing "
                "order stands and may already be filled."
            ),
        }
        self._write(payload)
        logger.critical(
            "%s broker_order=%s decision=%s -- SUBMISSION GATE CLOSED. %s",
            LOG_TAG_UNLINKED_INCIDENT, broker_order_id, decision_id, detail,
        )
        return payload

    def reopen(self, *, reason: str, at: str) -> None:
        """Reopen after reconciliation. Explicit and recorded."""
        self._write({"open": True, "reopened_at": at, "reason": reason})
        logger.warning("submission gate REOPENED: %s", reason)


__all__ = [
    "DEFAULT_MAX_QUOTE_AGE_MS",
    "LOG_TAG_UNLINKED_INCIDENT",
    "SUBMISSION_GATE_PATH",
    "SubmissionGate",
    "build_execution_context",
]
