"""Immutable incomplete observations; never live execution readiness."""
from datetime import datetime
from typing import Literal,Protocol
from src.evidence.contracts import StrictContract

UNAVAILABLE=('PROVIDER_SNAPSHOT_TIME_UNAVAILABLE','SETTLED_CASH_UNAVAILABLE',
    'RESERVED_CASH_UNAVAILABLE','EXTERNAL_CASHFLOW_HISTORY_UNAVAILABLE',
    'CORPORATE_ACTION_HISTORY_UNAVAILABLE','ADVANCED_ORDER_COVERAGE_UNAVAILABLE',
    'APPROVAL_SETTING_UNAVAILABLE','REGULAR_SESSION_CALENDAR_UNAVAILABLE')

class CapabilitySnapshot(StrictContract):
    provider:Literal['official-robinhood-trading-mcp']='official-robinhood-trading-mcp'
    declared_contract_sha256:str
    equity_read_bindings_configured:Literal[True]=True
    live_connection_verified:Literal[False]=False
    auth_revocation_verified:Literal[False]=False
    approval_setting:Literal['UNKNOWN']='UNKNOWN'
    provider_snapshot_time_available:Literal[False]=False
    advanced_order_coverage:Literal['UNAVAILABLE']='UNAVAILABLE'
    regular_session_calendar:Literal['UNAVAILABLE']='UNAVAILABLE'
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False
    blocked_prerequisites:tuple[str,...]=UNAVAILABLE

class PositionObservation(StrictContract):
    symbol:str
    quantity:str
    available_for_sells:str
    held_for_sells:str
    held_for_transfer:str
    held_for_options:str
    held_for_stock_grants:str
    pending_from_options:str
    intraday_quantity:str
    average_cost_cents:int|None
    average_cost_field_present:bool
    position_type:str

class AccountSnapshot(StrictContract):
    account_alias:str
    local_observation_id:str
    request_started_at:datetime
    received_at:datetime
    provider_observed_at:None=None
    reported_cash_cents:int
    pending_deposits_cents:int
    portfolio_value_cents:int
    asset_values_cents:dict[str,int]
    unsupported_asset_holdings_present:bool
    reported_unsettled_funds_cents:int|None
    unsettled_funds_field_present:bool
    buying_power_cents:int|None
    unleveraged_buying_power_cents:int|None
    settled_cash_cents:None=None
    reserved_cash_cents:None=None
    external_cashflows:None=None
    positions:tuple[PositionObservation,...]
    positions_complete:bool
    common_snapshot_verified:Literal[False]=False
    advanced_order_coverage:Literal['UNAVAILABLE']='UNAVAILABLE'
    corporate_actions_clear:Literal[False]=False
    execution_ready:Literal[False]=False
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False
    blocked_prerequisites:tuple[str,...]=UNAVAILABLE

    def as_risk_account(self):
        raise ValueError('AUTHORITATIVE_ACCOUNT_FACTS_UNAVAILABLE')

class QuoteObservation(StrictContract):
    symbol:str
    bid_micros:int
    ask_micros:int
    bid_at:datetime
    ask_at:datetime
    bid_source_time:str
    ask_source_time:str
    selected_trade_price_micros:int
    selected_trade_at:datetime
    selected_trade_source_time:str
    selected_trade_session:Literal['REGULAR','NONREGULAR']
    selected_trade_stale:bool
    official_close_price_micros:int|None
    official_close_date:str|None
    official_close_available:bool
    halt_information_complete:Literal[False]=False
    instrument_state:str
    account_type_tradability:str

class MarketSnapshot(StrictContract):
    local_observation_id:str
    request_started_at:datetime
    received_at:datetime
    provider_observed_at:None=None
    quotes:tuple[QuoteObservation,...]
    regular_session_verified:Literal[False]=False
    corporate_actions_clear:Literal[False]=False
    execution_ready:Literal[False]=False
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False
    blocked_prerequisites:tuple[str,...]=UNAVAILABLE

class FillObservation(StrictContract):
    execution_id:str
    price_micros:int
    quantity:str
    fees_cents:int
    occurred_at:datetime
    source_time:str

class OrderObservation(StrictContract):
    order_id:str
    instrument_id:str
    symbol:str
    side:str
    state:str
    state_supported:bool
    quantity:str|None
    cumulative_quantity:str
    fees_cents:int
    created_at:datetime
    last_transaction_at:datetime|None
    broker_ref_id:str|None
    request_identity_kind:Literal['BROKER_REF_UNVERIFIED']='BROKER_REF_UNVERIFIED'
    placed_agent:str
    order_type:str
    trigger:str
    time_in_force:str
    market_hours:str
    limit_price_micros:int|None
    stop_price_micros:int|None
    average_price_micros:int|None
    dollar_based_amount_cents:int|None
    created_source_time:str
    last_transaction_source_time:str|None
    executions:tuple[FillObservation,...]
    execution_details_complete:bool

class OrderObservationPage(StrictContract):
    account_alias:str
    local_observation_id:str
    request_started_at:datetime
    received_at:datetime
    provider_observed_at:None=None
    orders:tuple[OrderObservation,...]
    complete:bool
    coverage:Literal['ordinary-equity-orders-only']='ordinary-equity-orders-only'
    advanced_order_coverage:Literal['UNAVAILABLE']='UNAVAILABLE'
    stable_snapshot_verified:Literal[False]=False
    source_event_watermark:None=None
    account_binding:Literal['trusted-request-scope-only']='trusted-request-scope-only'
    execution_ready:Literal[False]=False
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False
    blocked_prerequisites:tuple[str,...]=UNAVAILABLE


class ReadOnlyBrokerPort(Protocol):
    def capabilities(self)->CapabilitySnapshot: ...
    async def market_snapshot(self,instruments:tuple[str,...])->MarketSnapshot: ...
    async def account_snapshot(self,account_alias:str)->AccountSnapshot: ...
    async def order_observations(self,account_alias:str)->OrderObservationPage: ...
