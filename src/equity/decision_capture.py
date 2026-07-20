"""Translate one equity control-loop cycle into DecisionRecords.

Sits at the decision seam: after the strategy has produced target weights and
the risk layer has produced its de-gross factor and verdict, but BEFORE any
order is submitted. It is deterministic, has no model dependency, and does not
influence what the loop decides -- it only records what the loop decided.

Persist-or-abstain
------------------

``persist_cycle_decisions`` returns whether the records reached disk. The
caller's obligation differs by class:

* **Executable decision** -- a failed write MUST convert the cycle to ABSTAIN
  and prevent order submission. An execution nobody recorded is an execution
  nobody can attribute, and an unattributable fill is exactly the hole this
  whole subsystem exists to close.
* **Candidate** -- a failed write must NOT change the disposition (the trade was
  already refused; losing the record does not un-refuse it) but must surface a
  visible error.

Weights, not units
------------------

The equity loop is weight-based end to end: ``Order`` carries ``weight_delta``
and ``target_weight``, and no price is available at this seam. Weights are
therefore always populated and units are populated only when a ``unit_converter``
is supplied. Absent one, the unit fields stay ``None`` and are named in
``unavailable_fields`` -- never zero-filled, which would assert a size that was
never intended.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.equity.rebalance import ExecResult, Order
from src.evidence.hashing import content_digest
from src.learning.decision_ledger import (
    DecisionLedgerError,
    append_decisions,
    append_execution_context,
    existing_digests,
    link_broker_order,
)
from src.learning.execution_context import ExecutionContextError, LotConstraints, QuoteSnapshot
from src.learning.execution_context_builder import (
    DEFAULT_MAX_QUOTE_AGE_MS,
    SubmissionGate,
    build_execution_context,
)
from src.learning.decision_record import (
    DecisionClass,
    DecisionRecord,
    Disposition,
    unavailable_from,
)

logger = logging.getLogger(__name__)

LOG_TAG_DECISION_CAPTURE_FAILED = "DECISION_CAPTURE_FAILED"
LOG_TAG_ORDER_UNLINKED = "ORDER_REFUSED_NO_PERSISTED_DECISION"
LOG_TAG_CONTEXT_REFUSED = "ORDER_REFUSED_NO_EXECUTION_CONTEXT"
LOG_TAG_GATE_CLOSED = "ORDER_REFUSED_SUBMISSION_GATE_CLOSED"

#: Fields that simply do not exist at the equity decision seam today. Named
#: once here so every record reports the same honest gap rather than each
#: caller inventing its own.
_STRUCTURALLY_ABSENT = (
    "reference_mid", "bid", "ask", "spread_bps",
    "estimated_entry_cost_bps", "estimated_exit_cost_bps", "estimated_total_cost_bps",
    "expected_move_bps", "expected_move_horizon", "expected_move_confidence",
    "expected_net_edge_bps",
)

UnitConverter = Callable[[str, float], Optional[float]]


def _direction(weight: float) -> str:
    if weight > 0:
        return "LONG"
    if weight < 0:
        return "SHORT"
    return "FLAT"


def _decision_class(disposition: Disposition) -> DecisionClass:
    return (
        DecisionClass.EXECUTED_DECISION
        if disposition is Disposition.EXECUTE
        else DecisionClass.CANDIDATE_NOT_EXECUTED
    )


def build_cycle_decisions(
    *,
    cycle_id: str,
    decision_timestamp: str,
    disposition: Disposition,
    reason_codes: Sequence[str],
    target_weights: Mapping[str, float],
    degross_factor: float,
    current_weights: Optional[Mapping[str, float]] = None,
    portfolio_nav: Optional[float] = None,
    peak_nav: Optional[float] = None,
    market_data_asof: Optional[str] = None,
    strategy_version: Optional[str] = None,
    risk_policy_version: Optional[str] = None,
    unit_converter: Optional[UnitConverter] = None,
    lane: str = "equity",
) -> List[DecisionRecord]:
    """One DecisionRecord per instrument touched by this cycle.

    The instrument set is the UNION of target and current weights, so an exit
    (target 0, currently held) is recorded as deliberately as an entry.
    """
    current = dict(current_weights or {})
    instruments = sorted(set(target_weights) | set(current))

    gross_before = sum(abs(float(w)) for w in current.values())
    gross_after = sum(abs(float(w) * float(degross_factor)) for w in target_weights.values())

    drawdown_state: Optional[float] = None
    if portfolio_nav is not None and peak_nav:
        drawdown_state = max(0.0, (float(peak_nav) - float(portfolio_nav)) / float(peak_nav))

    input_digest = content_digest({
        "target_weights": {k: float(v) for k, v in sorted(target_weights.items())},
        "current_weights": {k: float(v) for k, v in sorted(current.items())},
        "degross_factor": float(degross_factor),
        "portfolio_nav": portfolio_nav,
    })

    klass = _decision_class(disposition)
    records: List[DecisionRecord] = []

    for instrument in instruments:
        pre_degross_weight = float(target_weights.get(instrument, 0.0))
        intended_weight = pre_degross_weight * float(degross_factor)
        current_weight = float(current.get(instrument, 0.0))

        pre_degross_units = intended_units = current_units = order_delta = None
        if unit_converter is not None:
            pre_degross_units = unit_converter(instrument, pre_degross_weight)
            intended_units = unit_converter(instrument, intended_weight)
            current_units = unit_converter(instrument, current_weight)
            if intended_units is not None and current_units is not None:
                order_delta = intended_units - current_units

        unavailable = tuple(sorted(set(_STRUCTURALLY_ABSENT) | set(unavailable_from(
            pre_degross_units=pre_degross_units,
            intended_units=intended_units,
            current_units=current_units,
            order_delta=order_delta,
            drawdown_state=drawdown_state,
            market_data_asof=market_data_asof,
            strategy_version=strategy_version,
            risk_policy_version=risk_policy_version,
            portfolio_nav=portfolio_nav,
        ))))

        records.append(DecisionRecord(
            decision_id=f"{cycle_id}:{instrument}",
            cycle_id=cycle_id,
            decision_class=klass,
            disposition=disposition,
            reason_codes=tuple(reason_codes),
            instrument=instrument,
            direction=_direction(intended_weight),
            decision_timestamp=decision_timestamp,
            market_data_asof=market_data_asof,
            strategy_version=strategy_version,
            risk_policy_version=risk_policy_version,
            input_digest=input_digest,
            current_weight=current_weight,
            pre_degross_weight=pre_degross_weight,
            intended_weight=intended_weight,
            weight_delta=intended_weight - current_weight,
            current_units=current_units,
            pre_degross_units=pre_degross_units,
            intended_units=intended_units,
            order_delta=order_delta,
            degross_factor=float(degross_factor),
            portfolio_nav=portfolio_nav,
            gross_exposure_before=gross_before,
            gross_exposure_after=gross_after,
            drawdown_state=drawdown_state,
            unavailable_fields=unavailable,
            lane=lane,
        ))

    return records


def persist_cycle_decisions(
    records: Iterable[DecisionRecord],
    *,
    executable: bool,
    path: Optional[Any] = None,
) -> Tuple[bool, Dict[str, int]]:
    """Write decisions; report whether the caller may proceed to execute.

    Args:
        records: the cycle's decisions.
        executable: True when an order would follow. Governs the failure
            contract, not the write itself.

    Returns:
        ``(persisted, stats)``. When ``executable`` and ``persisted`` is False,
        the caller MUST abstain and submit nothing.
    """
    try:
        stats = append_decisions(records, path)
        return True, stats
    except (DecisionLedgerError, OSError) as exc:
        if executable:
            logger.error(
                "%s executable decision could not be persisted (%s) -- converting cycle "
                "to ABSTAIN; no order will be submitted", LOG_TAG_DECISION_CAPTURE_FAILED, exc,
            )
        else:
            logger.error(
                "%s candidate decision could not be persisted (%s) -- disposition "
                "preserved, evidence LOST for this cycle", LOG_TAG_DECISION_CAPTURE_FAILED, exc,
            )
        return False, {"appended": 0, "skipped_duplicate": 0}


__all__ = [
    "LOG_TAG_DECISION_CAPTURE_FAILED",
    "LOG_TAG_ORDER_UNLINKED",
    "decision_linked_executor",
    "build_cycle_decisions",
    "persist_cycle_decisions",
]


QuoteProvider = Callable[[str], Optional[QuoteSnapshot]]
NavProvider = Callable[[], Optional[Dict[str, float]]]
HoldingsProvider = Callable[[str], Optional[float]]


def decision_linked_executor(
    inner: Callable[[Order], Any],
    *,
    decisions: Sequence[DecisionRecord],
    ledger_path: Optional[Any] = None,
    plan_id: str = "unknown-plan",
    created_at: str = "",
    quote_provider: Optional[QuoteProvider] = None,
    nav_provider: Optional[NavProvider] = None,
    holdings_provider: Optional[HoldingsProvider] = None,
    lot: Optional[LotConstraints] = None,
    max_quote_age_ms: float = DEFAULT_MAX_QUOTE_AGE_MS,
    submission_gate: Optional[SubmissionGate] = None,
) -> Callable[[Order], Any]:
    """Wrap an executor so nothing reaches the broker unattributably.

    The full boundary, in order:

        submission gate open?
        -> decision exists, is EXECUTED_DECISION, and is ON DISK
        -> quote + NAV + holdings obtained
        -> weights converted to broker-valid units
        -> execution-context row PERSISTED
        -> broker called
        -> broker-link row appended

    Every pre-submission failure returns ``ExecResult.REJECTED`` with no broker
    call. A post-acceptance linkage failure is different in kind: the order is
    real and must not be blindly cancelled, so the existing order stands, a
    CRITICAL incident is raised, and the submission gate CLOSES so no further
    order can compound an unreconciled book.

    When no quote/NAV provider is supplied the context step is skipped, which
    preserves the pre-enrichment behaviour for callers that have not yet been
    given a market-data source.
    """
    by_instrument: Dict[str, DecisionRecord] = {d.instrument: d for d in decisions}
    on_disk = existing_digests(ledger_path)
    gate = submission_gate or SubmissionGate()

    def _guarded(order: Order) -> Any:
        if not gate.is_open():
            state = gate.state()
            logger.error(
                "%s order=%s -- gate closed by %s (%s); NOT submitting",
                LOG_TAG_GATE_CLOSED, order.client_order_id,
                state.get("incident"), state.get("detail"),
            )
            return ExecResult.REJECTED

        record = by_instrument.get(order.ticker)
        if record is None:
            logger.error(
                "%s order=%s ticker=%s -- no decision record for this instrument; "
                "NOT submitting", LOG_TAG_ORDER_UNLINKED, order.client_order_id, order.ticker,
            )
            return ExecResult.REJECTED
        if record.decision_class is not DecisionClass.EXECUTED_DECISION:
            logger.error(
                "%s order=%s ticker=%s -- decision is %s, which documents a refusal "
                "and cannot authorise an order; NOT submitting",
                LOG_TAG_ORDER_UNLINKED, order.client_order_id, order.ticker,
                record.decision_class.value,
            )
            return ExecResult.REJECTED
        digest = record.digest()
        if digest not in on_disk:
            logger.error(
                "%s order=%s ticker=%s -- decision %s is not persisted; NOT submitting",
                LOG_TAG_ORDER_UNLINKED, order.client_order_id, order.ticker, record.decision_id,
            )
            return ExecResult.REJECTED

        # -- execution context: weights become units, or nothing is sent -----
        if quote_provider is not None and nav_provider is not None:
            try:
                quote = quote_provider(order.ticker)
                if quote is None:
                    raise ExecutionContextError(f"no quote available for {order.ticker}")
                nav_info = nav_provider()
                if not nav_info:
                    raise ExecutionContextError("no NAV available")
                held = holdings_provider(order.ticker) if holdings_provider else 0.0
                context = build_execution_context(
                    decision=record,
                    decision_digest=digest,
                    plan_id=plan_id,
                    created_at=created_at,
                    quote=quote,
                    portfolio_nav=float(nav_info.get("portfolio_nav")),
                    allocatable_nav=nav_info.get("allocatable_nav"),
                    cash_available=nav_info.get("cash_available"),
                    current_units=float(held or 0.0),
                    lot=lot,
                    max_quote_age_ms=max_quote_age_ms,
                )
                append_execution_context(context, ledger_path)
            except (ExecutionContextError, DecisionLedgerError, OSError, TypeError, ValueError) as exc:
                logger.error(
                    "%s order=%s ticker=%s -- %s; NOT submitting",
                    LOG_TAG_CONTEXT_REFUSED, order.client_order_id, order.ticker, exc,
                )
                return ExecResult.REJECTED

        outcome = inner(order)

        try:
            link_broker_order(
                digest,
                broker_order_id=order.client_order_id,
                client_order_id=order.client_order_id,
                path=ledger_path,
            )
        except (DecisionLedgerError, OSError) as exc:
            # The broker may now hold an order with no durable local linkage.
            # Do NOT cancel -- the position is real and a blind cancel can race
            # a fill. Do NOT keep submitting either: each further order could
            # add another unlinked position to an already-unreconciled book.
            gate.raise_unlinked_incident(
                broker_order_id=order.client_order_id,
                decision_id=record.decision_id,
                detail=f"linkage write failed after broker acceptance: {exc}",
                at=created_at,
            )
        return outcome

    return _guarded
