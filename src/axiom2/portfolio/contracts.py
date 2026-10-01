"""Strict, normalized inputs and non-capital portfolio results.

References bind input identity, not source truth or signature authorization.
This module owns no credentials, broker client, training hook or live defaults.
Its records are not a hostile-code sandbox; privileged consumers must verify
source evidence and process authority independently before creating orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
import hashlib
import re
from zoneinfo import ZoneInfo

from src.evidence.canonical import canonical_bytes

PHASE1_PORTFOLIO_DIGEST = hashlib.sha256(canonical_bytes({
    "top_k": 5, "holding_horizon": 5, "rebalance_every": 5,
    "weighting": "equal", "overlap": False, "schema_version": 1,
})).hexdigest()
_SOURCE_KINDS = frozenset({"SYNTHETIC", "DEVELOPMENT", "LIVE_SHADOW"})
_RISK_STATES = frozenset({"RUN", "RISK_OFF", "HALTED"})
_DECISION_STATES = frozenset({
    "READY_SHADOW", "REJECTED", "HOLD", "HALTED", "RECONCILE", "EXIT_REQUIRED",
})


def _integer(value: object, name: str, minimum: int | None = 0,
             maximum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a plain integer")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")


def _boolean(value: object, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _reference(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a plain string")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError(f"{name} must be a compact reference")


def _digest(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a plain SHA-256 string")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


def _timestamp(value: object, name: str) -> None:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be a datetime")
    if type(value.tzinfo) not in (timezone, ZoneInfo) or value.utcoffset() is None:
        raise ValueError(f"{name} must have an explicit standard timezone")


def _choice(value: object, name: str, allowed: frozenset[str]) -> None:
    _reference(value, name)
    if value not in allowed:
        raise ValueError(f"{name} is unsupported")


def _schema(value: object) -> None:
    _integer(value, "schema_version", 1, 1)


def _exact(value: object, expected: type, name: str) -> None:
    if type(value) is not expected:
        raise TypeError(f"{name} must be an exact {expected.__name__} record")


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskPolicy:
    """Versioned risk overlay; all numerical limits must be supplied explicitly."""
    allocation_bps: int
    max_name_bps: int
    max_sector_bps: int
    max_correlation_group_bps: int
    max_gross_bps: int
    cash_reserve_bps: int
    max_turnover_bps: int
    max_participation_bps: int
    min_net_edge_bps: int
    max_spread_bps: int
    cost_buffer_bps: int
    max_drawdown_bps: int
    quote_ttl_seconds: int
    snapshot_ttl_seconds: int
    signal_ttl_seconds: int
    metadata_ttl_seconds: int
    rebalance_anchor_session: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        validate_policy(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class Position:
    instrument_id: str
    market_value_cents: int
    sector_id: str
    correlation_group_id: str

    def __post_init__(self) -> None:
        validate_position(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PortfolioSnapshot:
    observed_at: datetime
    nav_cents: int
    cash_cents: int
    settled_cash_cents: int
    reserved_cash_cents: int
    positions: tuple[Position, ...]
    open_order_count: int
    unresolved_orders: bool
    reconciled: bool
    corporate_actions_clear: bool
    peak_nav_cents: int
    net_external_cash_flow_since_peak_cents: int = 0
    cohort_entry_session: int | None = None
    source_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        validate_snapshot(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RankedOpportunity:
    """Rank is not economic edge; net edge needs separate evidence in the request."""
    instrument_id: str
    asset_class: str
    rank: int
    sector_id: str
    correlation_group_id: str
    expected_net_edge_bps: int
    adv_cents: int
    bid_micros: int
    ask_micros: int
    quoted_at: datetime
    metadata_at: datetime
    eligible: bool
    membership_known: bool
    corporate_actions_clear: bool

    def __post_init__(self) -> None:
        validate_opportunity(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PortfolioRequest:
    decision_id: str
    account_alias: str
    as_of: datetime
    session_index: int
    session_open: datetime
    session_close: datetime
    risk_state: str
    source_kind: str
    signal_at: datetime
    model_artifact_hash: str
    promotion_decision_hash: str
    signal_digest: str
    economic_evidence_digest: str
    universe_digest: str
    calendar_digest: str
    campaign_portfolio_digest: str
    snapshot: PortfolioSnapshot
    candidates: tuple[RankedOpportunity, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        validate_request(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetPosition:
    instrument_id: str
    target_value_cents: int

    def __post_init__(self) -> None:
        _reference(self.instrument_id, "instrument_id")
        _integer(self.target_value_cents, "target_value_cents", 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskDecision:
    """A desired shadow portfolio, never an order or an authorization token.

    No targets on a non-ready state means no executable instruction. In
    particular it must never be interpreted as a request to liquidate.
    """
    status: str
    reasons: tuple[str, ...]
    targets: tuple[TargetPosition, ...]
    decision_id: str
    account_alias: str
    as_of: datetime
    source_kind: str
    policy_digest: str
    input_digest: str
    decision_digest: str
    gross_trade_cents: int | None
    turnover_bps_ceiling: int | None
    estimated_cost_cents: int | None
    residual_cash_cents: int | None
    execution_enabled: bool = field(default=False, init=False)
    capital_authorized: bool = field(default=False, init=False)
    schema_version: int = field(default=1, init=False)


def validate_policy(policy: RiskPolicy) -> None:
    _exact(policy, RiskPolicy, "policy")
    _schema(policy.schema_version)
    for item in fields(policy):
        if item.name.endswith("_bps"):
            _integer(getattr(policy, item.name), item.name, 0, 10000)
        elif item.name.endswith("_seconds"):
            _integer(getattr(policy, item.name), item.name, 1)
    _integer(policy.allocation_bps, "allocation_bps", 1, 10000)
    _integer(policy.max_drawdown_bps, "max_drawdown_bps", 1, 10000)
    _integer(policy.rebalance_anchor_session, "rebalance_anchor_session")
    if policy.allocation_bps + policy.cash_reserve_bps > 10000:
        raise ValueError("allocation_bps and cash_reserve_bps exceed full capital")


def validate_position(position: Position) -> None:
    _exact(position, Position, "position")
    for name in ("instrument_id", "sector_id", "correlation_group_id"):
        _reference(getattr(position, name), name)
    _integer(position.market_value_cents, "market_value_cents", 0)


def validate_snapshot(snapshot: PortfolioSnapshot) -> None:
    _exact(snapshot, PortfolioSnapshot, "snapshot")
    _schema(snapshot.schema_version)
    _timestamp(snapshot.observed_at, "observed_at")
    _digest(snapshot.source_digest, "source_digest")
    for name in ("nav_cents", "cash_cents", "settled_cash_cents", "reserved_cash_cents",
                 "open_order_count", "peak_nav_cents"):
        _integer(getattr(snapshot, name), name)
    _integer(snapshot.net_external_cash_flow_since_peak_cents,
             "net_external_cash_flow_since_peak_cents", None)
    _integer(snapshot.nav_cents, "nav_cents", 1)
    for name in ("unresolved_orders", "reconciled", "corporate_actions_clear"):
        _boolean(getattr(snapshot, name), name)
    if snapshot.cohort_entry_session is not None:
        _integer(snapshot.cohort_entry_session, "cohort_entry_session")
    if type(snapshot.positions) is not tuple:
        raise TypeError("positions must be an immutable tuple")
    for position in snapshot.positions:
        validate_position(position)
    if len({p.instrument_id for p in snapshot.positions}) != len(snapshot.positions):
        raise ValueError("positions contains duplicate instrument identities")
    if snapshot.nav_cents != snapshot.cash_cents + sum(p.market_value_cents for p in snapshot.positions):
        raise ValueError("nav_cents must equal valued positions plus cash_cents")
    if snapshot.settled_cash_cents > snapshot.cash_cents:
        raise ValueError("settled_cash_cents exceeds cash_cents")
    if snapshot.reserved_cash_cents > snapshot.settled_cash_cents:
        raise ValueError("reserved_cash_cents exceeds settled_cash_cents")
    flow_adjusted_nav = snapshot.nav_cents - snapshot.net_external_cash_flow_since_peak_cents
    if flow_adjusted_nav < 0:
        raise ValueError("external cash flow implies negative flow-adjusted equity")
    if snapshot.peak_nav_cents < flow_adjusted_nav:
        raise ValueError("peak_nav_cents must include current flow-adjusted nav_cents")


def validate_opportunity(opportunity: RankedOpportunity) -> None:
    _exact(opportunity, RankedOpportunity, "opportunity")
    for name in ("instrument_id", "sector_id", "correlation_group_id"):
        _reference(getattr(opportunity, name), name)
    _choice(opportunity.asset_class, "asset_class", frozenset({"EQUITY", "ETF"}))
    _integer(opportunity.rank, "rank", 1, 5)
    _integer(opportunity.expected_net_edge_bps, "expected_net_edge_bps", None)
    for name in ("adv_cents", "bid_micros", "ask_micros"):
        _integer(getattr(opportunity, name), name, 1)
    for name in ("quoted_at", "metadata_at"):
        _timestamp(getattr(opportunity, name), name)
    for name in ("eligible", "membership_known", "corporate_actions_clear"):
        _boolean(getattr(opportunity, name), name)


def validate_request(request: PortfolioRequest) -> None:
    """Revalidate the full nested object at admission, not only on construction."""
    _exact(request, PortfolioRequest, "request")
    _schema(request.schema_version)
    for name in ("decision_id", "account_alias"):
        _reference(getattr(request, name), name)
    for name in ("as_of", "session_open", "session_close", "signal_at"):
        _timestamp(getattr(request, name), name)
    _integer(request.session_index, "session_index")
    if request.session_close <= request.session_open:
        raise ValueError("session_close must be after session_open")
    _choice(request.risk_state, "risk_state", _RISK_STATES)
    _choice(request.source_kind, "source_kind", _SOURCE_KINDS)
    for name in ("model_artifact_hash", "promotion_decision_hash", "signal_digest",
                 "economic_evidence_digest", "universe_digest", "calendar_digest",
                 "campaign_portfolio_digest"):
        _digest(getattr(request, name), name)
    validate_snapshot(request.snapshot)
    if type(request.candidates) is not tuple:
        raise TypeError("candidates must be an immutable tuple")
    for opportunity in request.candidates:
        validate_opportunity(opportunity)
    for name in ("instrument_id", "rank"):
        if len({getattr(o, name) for o in request.candidates}) != len(request.candidates):
            raise ValueError(f"candidates contains duplicate {name}")
