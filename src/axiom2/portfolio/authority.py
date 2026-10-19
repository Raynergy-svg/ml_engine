"""Pure deterministic portfolio/risk evaluation with no execution capability.

This evaluates normalized observations, not their external authenticity. The
privileged execution admission boundary must later verify the referenced
artifacts, signatures, roles, account state and policy. No result here permits
capital, and no empty-target result is a liquidation instruction.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib

from src.evidence.canonical import canonical_bytes
from .contracts import (
    PHASE1_PORTFOLIO_DIGEST, PortfolioRequest, RiskDecision, RiskPolicy,
    TargetPosition, validate_policy, validate_request,
)


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def risk_policy_digest(policy: RiskPolicy) -> str:
    validate_policy(policy)
    return _hash(asdict(policy))


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _age_reason(observed: datetime, now: datetime, ttl: int, prefix: str) -> str | None:
    # UTC and integer microseconds avoid DST wall-clock ambiguity and float ages.
    delta = _utc(now) - _utc(observed)
    age = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    if age < 0:
        return prefix + "_FUTURE"
    if age > ttl * 1_000_000:
        return prefix + "_STALE"
    return None


def _exposure_reasons(rows: tuple[tuple[str, int, str, str], ...],
                      nav: int, policy: RiskPolicy) -> list[str]:
    reasons = []
    sectors: dict[str, int] = {}
    groups: dict[str, int] = {}
    gross = 0
    for _, value, sector, group in rows:
        gross += value
        sectors[sector] = sectors.get(sector, 0) + value
        groups[group] = groups.get(group, 0) + value
        if value * 10000 > nav * policy.max_name_bps:
            reasons.append("NAME_LIMIT")
    if gross * 10000 > nav * policy.max_gross_bps:
        reasons.append("GROSS_LIMIT")
    if any(value * 10000 > nav * policy.max_sector_bps for value in sectors.values()):
        reasons.append("SECTOR_LIMIT")
    if any(value * 10000 > nav * policy.max_correlation_group_bps for value in groups.values()):
        reasons.append("CORRELATION_LIMIT")
    return reasons


def evaluate_portfolio(request: PortfolioRequest, policy: RiskPolicy) -> RiskDecision:
    """Return a repeatable non-capital decision or reject malformed input.

    Rank/weight/horizon selection is fixed. A violated constraint rejects the
    complete entry cohort instead of quietly constructing a different strategy.
    Existing positions are explicitly held, reconciled or referred for exits;
    the evaluator itself does not calculate or submit broker orders.
    """
    validate_request(request)
    validate_policy(policy)
    snapshot = request.snapshot
    normalized = asdict(request)
    normalized["candidates"] = sorted(normalized["candidates"], key=lambda o: (o["rank"], o["instrument_id"]))
    normalized["snapshot"]["positions"] = sorted(
        normalized["snapshot"]["positions"], key=lambda p: p["instrument_id"],
    )
    input_digest = _hash(normalized)
    policy_digest = risk_policy_digest(policy)

    def decision(status: str, reasons: list[str] | tuple[str, ...] = (), *,
                 targets: tuple[TargetPosition, ...] = (), gross: int | None = None,
                 turnover: int | None = None, costs: int | None = None,
                 cash: int | None = None) -> RiskDecision:
        body = dict(
            status=status, reasons=tuple(sorted(set(reasons))), targets=targets,
            decision_id=request.decision_id, account_alias=request.account_alias,
            as_of=request.as_of, source_kind=request.source_kind,
            policy_digest=policy_digest, input_digest=input_digest,
            gross_trade_cents=gross, turnover_bps_ceiling=turnover,
            estimated_cost_cents=costs, residual_cash_cents=cash,
        )
        digest_body = dict(body, targets=tuple(asdict(target) for target in targets),
                           execution_enabled=False, capital_authorized=False, schema_version=1)
        return RiskDecision(**body, decision_digest=_hash(digest_body))

    if request.risk_state == "HALTED":
        return decision("HALTED", ["KILL_STATE"])

    reasons = []
    if not snapshot.reconciled:
        reasons.append("ACCOUNT_UNRECONCILED")
    if snapshot.unresolved_orders:
        reasons.append("ORDER_STATE_UNKNOWN")
    if snapshot.open_order_count:
        reasons.append("OPEN_ORDERS")
    if not snapshot.corporate_actions_clear:
        reasons.append("ACCOUNT_CORPORATE_ACTION_UNRESOLVED")
    age = _age_reason(snapshot.observed_at, request.as_of, policy.snapshot_ttl_seconds, "SNAPSHOT")
    if age:
        reasons.append(age)
    if reasons:
        return decision("RECONCILE", reasons)

    entry = snapshot.cohort_entry_session
    if (bool(snapshot.positions) != (entry is not None)
            or (entry is not None and (entry > request.session_index
                or entry < policy.rebalance_anchor_session
                or (entry - policy.rebalance_anchor_session) % 5))):
        return decision("RECONCILE", ["COHORT_STATE_UNKNOWN"])

    routing_open = _utc(request.session_open) <= _utc(request.as_of) < _utc(request.session_close)

    risk_reasons = []
    if request.risk_state == "RISK_OFF":
        risk_reasons.append("RISK_OFF")
    flow_adjusted_nav = snapshot.nav_cents - snapshot.net_external_cash_flow_since_peak_cents
    if flow_adjusted_nav < 0:
        return decision("RECONCILE", ["EXTERNAL_CASH_FLOW_STATE_INVALID"])
    if (snapshot.peak_nav_cents - flow_adjusted_nav) * 10000 >= snapshot.peak_nav_cents * policy.max_drawdown_bps:
        risk_reasons.append("DRAWDOWN_LIMIT")
    if snapshot.positions:
        rows = tuple((p.instrument_id, p.market_value_cents, p.sector_id, p.correlation_group_id)
                     for p in snapshot.positions)
        risk_reasons.extend(_exposure_reasons(rows, snapshot.nav_cents, policy))
        if (snapshot.cash_cents - snapshot.reserved_cash_cents) * 10000 < snapshot.nav_cents * policy.cash_reserve_bps:
            risk_reasons.append("CASH_RESERVE")
    if risk_reasons:
        if not routing_open:
            risk_reasons.append("ROUTING_CLOSED")
        return decision("EXIT_REQUIRED" if snapshot.positions else "HALTED", risk_reasons)

    if snapshot.positions:
        if request.session_index - entry >= 5:
            maturity_reasons = ["COHORT_MATURED"]
            if not routing_open:
                maturity_reasons.append("ROUTING_CLOSED")
            return decision("EXIT_REQUIRED", maturity_reasons)
        active_reasons = ["COHORT_ACTIVE"]
        if not routing_open:
            active_reasons.append("OUTSIDE_REGULAR_SESSION")
        return decision("HOLD", active_reasons)

    if not routing_open:
        return decision("HOLD", ["OUTSIDE_REGULAR_SESSION"])

    if request.session_index < policy.rebalance_anchor_session or (request.session_index - policy.rebalance_anchor_session) % 5:
        return decision("HOLD", ["NOT_REBALANCE_SESSION"])

    opportunities = tuple(sorted(request.candidates, key=lambda o: (o.rank, o.instrument_id)))
    if request.campaign_portfolio_digest != PHASE1_PORTFOLIO_DIGEST:
        reasons.append("PORTFOLIO_POLICY_MISMATCH")
    if tuple(o.rank for o in opportunities) != (1, 2, 3, 4, 5):
        reasons.append("COHORT_INCOMPLETE")
    age = _age_reason(request.signal_at, request.as_of, policy.signal_ttl_seconds, "SIGNAL")
    if age:
        reasons.append(age)
    for opportunity in opportunities:
        for valid, reason in (
            (opportunity.eligible, "INELIGIBLE"),
            (opportunity.membership_known, "UNKNOWN_MEMBERSHIP"),
            (opportunity.corporate_actions_clear, "CORPORATE_ACTION_UNRESOLVED"),
        ):
            if not valid:
                reasons.append(reason)
        for at, ttl, prefix in (
            (opportunity.quoted_at, policy.quote_ttl_seconds, "QUOTE"),
            (opportunity.metadata_at, policy.metadata_ttl_seconds, "METADATA"),
        ):
            age = _age_reason(at, request.as_of, ttl, prefix)
            if age:
                reasons.append(age)
        if opportunity.bid_micros > opportunity.ask_micros:
            reasons.append("CROSSED_QUOTE")
        elif (opportunity.ask_micros - opportunity.bid_micros) * 20000 > policy.max_spread_bps * (opportunity.ask_micros + opportunity.bid_micros):
            reasons.append("SPREAD_LIMIT")
        # For a mid-marked target, crossing to the ask incurs at least half
        # the quoted spread. Reject an incoherent fixed buffer; never silently
        # revise the preregistered cost assumption to make entry pass.
        if (opportunity.ask_micros - opportunity.bid_micros) * 10000 > policy.cost_buffer_bps * (opportunity.ask_micros + opportunity.bid_micros):
            reasons.append("COST_BUFFER_BELOW_SPREAD")
        if opportunity.expected_net_edge_bps < policy.min_net_edge_bps:
            reasons.append("EDGE_FLOOR")
    if reasons:
        return decision("REJECTED", reasons)

    # Floor each equally weighted sleeve allocation; never redistribute cents
    # or substitute another name to make a risk constraint pass.
    target_value = snapshot.nav_cents * policy.allocation_bps // 50000
    if target_value <= 0:
        return decision("REJECTED", ["INSUFFICIENT_ALLOCATION"])
    targets = tuple(TargetPosition(instrument_id=o.instrument_id, target_value_cents=target_value)
                    for o in opportunities)
    rows = tuple((o.instrument_id, target_value, o.sector_id, o.correlation_group_id)
                 for o in opportunities)
    reasons.extend(_exposure_reasons(rows, snapshot.nav_cents, policy))
    gross = target_value * 5  # Entry only follows a reconciled flat snapshot.
    costs = (gross * policy.cost_buffer_bps + 9999) // 10000
    turnover = (gross * 10000 + 2 * snapshot.nav_cents - 1) // (2 * snapshot.nav_cents)
    if gross * 10000 > 2 * snapshot.nav_cents * policy.max_turnover_bps:
        reasons.append("TURNOVER_LIMIT")
    for opportunity in opportunities:
        if target_value * 10000 > opportunity.adv_cents * policy.max_participation_bps:
            reasons.append("PARTICIPATION_LIMIT")
    available = snapshot.settled_cash_cents - snapshot.reserved_cash_cents
    if gross + costs > available:
        reasons.append("INSUFFICIENT_SETTLED_CASH")
    if (available - gross - costs) * 10000 < snapshot.nav_cents * policy.cash_reserve_bps:
        reasons.append("CASH_RESERVE")
    if reasons:
        return decision("REJECTED", reasons)
    return decision("READY_SHADOW", targets=targets, gross=gross, turnover=turnover,
                    costs=costs, cash=snapshot.cash_cents - gross - costs)


def verify_risk_decision_integrity(decision: RiskDecision) -> None:
    """Recompute the canonical decision digest at downstream admission."""
    if type(decision) is not RiskDecision:
        raise TypeError("decision must be an exact RiskDecision")
    body=asdict(decision)
    supplied=body.pop("decision_digest")
    if _hash(body)!=supplied:
        raise ValueError("risk decision digest mismatch")
