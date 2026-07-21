"""DecisionRecord capture for the OANDA FX trend lane.

Why this exists separately from ``src/equity/decision_capture.py``
-----------------------------------------------------------------

The equity harvester and the FX trend lane are different code paths with
different natures, and only one of them can currently produce realized P&L:

* The harvester runs through ``AutonomousLoop`` / ``decide_cycle`` and is
  weight-native. It needs IBKR to place a real order -- ``ib_async`` is not
  installed and nothing listens on :7497 -- so its fills are simulated.
* The FX trend lane (``run_oanda_trend_cycle``) never touches ``decide_cycle``;
  it imports only ``_lane_halted`` and calls ``client.create_market_order``
  directly. It is the ONLY lane with a working broker, so it is the only source
  of broker-realized closes.

Instrumenting only the harvester would therefore record intentions that can
never be scored against real money, while every real fill stayed
unattributable -- which is the gap this whole subsystem exists to close. All
124 realized closes in the journal carry no decision.

Unit-native, so no execution-context event is needed
----------------------------------------------------

The equity seam had to defer units to a later event because NAV, price and lot
rules do not exist there. Here they all do: ``want`` (target units),
``current``, ``delta``, ``last_px`` and NAV are in hand at the decision point.
So a single DecisionRecord carries both the intention and its executable size,
and ``unavailable_fields`` stays small and honest.

What is still unavailable: the strategy emits no explicit expected move or
horizon, so ``expected_move_bps`` and everything derived from it remain None.
Recording that gap is the point -- it is the measured distance to a calibratable
selectivity score, not a defect to paper over.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.evidence.hashing import content_digest
from src.learning.decision_ledger import DecisionLedgerError, append_decisions
from src.learning.decision_record import (
    DecisionClass,
    DecisionRecord,
    Disposition,
    unavailable_from,
)

logger = logging.getLogger(__name__)

LOG_TAG_FX_CAPTURE_FAILED = "FX_DECISION_CAPTURE_FAILED"

#: The FX strategy emits a direction, not a magnitude forecast. Named once so
#: every record reports the same honest gap.
_NO_FORECAST = (
    "expected_move_bps", "expected_move_horizon", "expected_move_confidence",
    "expected_net_edge_bps", "estimated_exit_cost_bps", "estimated_total_cost_bps",
)


def _direction(units: float) -> str:
    if units > 0:
        return "LONG"
    if units < 0:
        return "SHORT"
    return "FLAT"


def build_fx_cycle_decisions(
    *,
    cycle_id: str,
    decision_timestamp: str,
    targets: Mapping[str, float],
    want_units: Mapping[str, float],
    current_units: Mapping[str, float],
    nav: float,
    gross_leverage: float,
    last_px: Optional[Mapping[str, float]] = None,
    atr: Optional[Mapping[str, float]] = None,
    disposition: Disposition = Disposition.EXECUTE,
    reason_codes: Sequence[str] = (),
    blocked: Optional[Mapping[str, str]] = None,
    strategy_version: Optional[str] = None,
    market_data_asof: Optional[str] = None,
) -> List[DecisionRecord]:
    """One DecisionRecord per instrument this cycle considered.

    The instrument set is the union of targets and current holdings, so an exit
    (target 0 while holding) is recorded as deliberately as an entry.

    ``blocked`` maps instrument -> reason for instruments the risk gates refused.
    Those become CANDIDATE_NOT_EXECUTED with the gate's own reason code, so a
    refusal is evidence rather than a silent absence.
    """
    prices = dict(last_px or {})
    atrs = dict(atr or {})
    blocked = dict(blocked or {})
    instruments = sorted(set(targets) | set(want_units) | set(current_units))

    input_digest = content_digest({
        "targets": {k: float(v) for k, v in sorted(targets.items())},
        "current_units": {k: float(v) for k, v in sorted(current_units.items())},
        "nav": float(nav),
        "gross_leverage": float(gross_leverage),
    })

    gross_before = sum(abs(float(u)) * float(prices.get(i, 0.0) or 0.0)
                       for i, u in current_units.items())
    gross_after = sum(abs(float(u)) * float(prices.get(i, 0.0) or 0.0)
                      for i, u in want_units.items())

    records: List[DecisionRecord] = []
    for inst in instruments:
        want = float(want_units.get(inst, 0.0))
        held = float(current_units.get(inst, 0.0))
        delta = want - held
        price = prices.get(inst)

        # A gate refusal makes this a candidate, not an execution.
        gate_reason = blocked.get(inst)
        if gate_reason:
            disp = Disposition.NO_ACT
            reasons: Tuple[str, ...] = (gate_reason,)
        elif delta == 0:
            disp = Disposition.NO_ACT
            reasons = ("no_change",)
        else:
            disp = disposition
            reasons = tuple(reason_codes)

        klass = (DecisionClass.EXECUTED_DECISION
                 if disp is Disposition.EXECUTE
                 else DecisionClass.CANDIDATE_NOT_EXECUTED)

        reference_mid = float(price) if price else None
        unavailable = tuple(sorted(set(_NO_FORECAST) | set(unavailable_from(
            reference_mid=reference_mid,
            bid=None, ask=None, spread_bps=None,
            estimated_entry_cost_bps=None,
            market_data_asof=market_data_asof,
            strategy_version=strategy_version,
        ))))

        records.append(DecisionRecord(
            decision_id=f"{cycle_id}:{inst}",
            cycle_id=cycle_id,
            decision_class=klass,
            disposition=disp,
            reason_codes=reasons,
            instrument=inst,
            direction=_direction(want),
            decision_timestamp=decision_timestamp,
            market_data_asof=market_data_asof,
            strategy_version=strategy_version,
            input_digest=input_digest,
            reference_mid=reference_mid,
            # FX is unit-native: no de-gross step, so pre_degross == intended.
            current_units=held,
            pre_degross_units=want,
            intended_units=want,
            order_delta=delta,
            degross_factor=1.0,
            pre_degross_weight=float(targets.get(inst, 0.0)),
            intended_weight=float(targets.get(inst, 0.0)),
            current_weight=None,
            weight_delta=None,
            portfolio_nav=float(nav),
            gross_exposure_before=gross_before,
            gross_exposure_after=gross_after,
            unavailable_fields=unavailable,
            lane="oanda_fx",
            notes={"gross_leverage": float(gross_leverage),
                   "atr": float(atrs[inst]) if inst in atrs and atrs[inst] else None},
        ))

    return records


def persist_fx_decisions(
    records: Iterable[DecisionRecord],
    *,
    path: Optional[Any] = None,
) -> Tuple[bool, Dict[str, int]]:
    """Write the cycle's decisions.

    Returns ``(persisted, stats)``. The FX cycle treats a failure as
    NON-BLOCKING for the trade: unlike the equity seam, this lane's orders are
    placed one-by-one inside a loop with risk gates already applied, and
    aborting mid-loop could leave the book half-rebalanced -- a worse outcome
    than an unattributed fill. The failure is logged loudly instead.

    That asymmetry is deliberate and worth stating: on the equity path a failed
    write converts the cycle to ABSTAIN because nothing has been sent yet.
    """
    try:
        stats = append_decisions(records, path)
        return True, stats
    except (DecisionLedgerError, OSError) as exc:
        logger.error(
            "%s could not persist %d FX decision(s): %s -- orders will proceed but "
            "this cycle's fills will be UNATTRIBUTABLE",
            LOG_TAG_FX_CAPTURE_FAILED, len(list(records)), exc,
        )
        return False, {"appended": 0, "skipped_duplicate": 0}


__all__ = [
    "LOG_TAG_FX_CAPTURE_FAILED",
    "build_fx_cycle_decisions",
    "persist_fx_decisions",
]
