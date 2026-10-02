"""Broker-neutral equity intent records. No broker transport or capital authority."""
from dataclasses import dataclass, field
from datetime import datetime
import re
from src.axiom2.portfolio.contracts import _boolean,_digest,_integer,_reference,_timestamp
from src.evidence.contracts import StrictContract

@dataclass(frozen=True,slots=True,kw_only=True)
class BrokerCapabilities:
    capability_digest:str
    whole_shares:bool
    regular_session_day_limit:bool
    fractional:bool
    notional:bool
    market_orders:bool
    extended_hours:bool
    schema_version:int=1
    def __post_init__(self):
        _digest(self.capability_digest,"capability_digest")
        for name in ("whole_shares","regular_session_day_limit","fractional","notional","market_orders","extended_hours"):
            _boolean(getattr(self,name),name)
        _integer(self.schema_version,"schema_version",1,1)

@dataclass(frozen=True,slots=True,kw_only=True)
class AccountObservationRef:
    account_alias:str
    revision:str
    observed_at:datetime
    expires_at:datetime
    source_digest:str
    settled_cash_cents:int
    reserved_cash_cents:int
    positions:tuple[tuple[str,int],...]
    schema_version:int=1
    def __post_init__(self):
        _reference(self.account_alias,"account_alias")
        _reference(self.revision,"revision")
        _timestamp(self.observed_at,"observed_at"); _timestamp(self.expires_at,"expires_at")
        if self.expires_at<=self.observed_at: raise ValueError("expires_at must follow observed_at")
        _digest(self.source_digest,"source_digest")
        _integer(self.settled_cash_cents,"settled_cash_cents"); _integer(self.reserved_cash_cents,"reserved_cash_cents")
        if self.reserved_cash_cents>self.settled_cash_cents: raise ValueError("reserved cash exceeds settled cash")
        if type(self.positions) is not tuple: raise TypeError("positions must be immutable")
        seen=set()
        for row in self.positions:
            if type(row) is not tuple or len(row)!=2: raise TypeError("positions entries must be (instrument_id, quantity)")
            _reference(row[0],"instrument_id"); _integer(row[1],"quantity")
            if row[0] in seen:
                raise ValueError("duplicate position")
            seen.add(row[0])
        _integer(self.schema_version,"schema_version",1,1)
class RiskIntentReceipt(StrictContract):
    decision_digest:str
    intents_digest:str
    account_alias:str
    account_revision:str
    capability_digest:str
    mode:str
    intent_count:int

@dataclass(frozen=True,slots=True,kw_only=True)
class EquityOrderIntent:
    intent_id:str
    decision_digest:str
    account_alias:str
    account_revision:str
    capability_digest:str
    mode:str
    instrument_id:str
    side:str
    order_type:str
    time_in_force:str
    quantity:int|None
    notional_cents:int|None
    limit_price_micros:int
    expires_at:datetime
    execution_enabled:bool=field(default=False,init=False)
    capital_authorized:bool=field(default=False,init=False)
    schema_version:int=field(default=1,init=False)
    def __post_init__(self):
        for name in ("intent_id","account_alias","account_revision","instrument_id"):
            _reference(getattr(self,name),name)
        for name in ("decision_digest","capability_digest"): _digest(getattr(self,name),name)
        if self.mode!="SHADOW": raise ValueError("mode must be SHADOW")
        if self.side not in {"BUY","SELL"}: raise ValueError("unsupported side")
        if self.order_type!="LIMIT" or self.time_in_force!="DAY": raise ValueError("unsupported order profile")
        if (self.quantity is None)==(self.notional_cents is None): raise ValueError("exactly one quantity form is required")
        if self.quantity is not None: _integer(self.quantity,"quantity",1)
        if self.notional_cents is not None: _integer(self.notional_cents,"notional_cents",1)
        _integer(self.limit_price_micros,"limit_price_micros",1); _timestamp(self.expires_at,"expires_at")
