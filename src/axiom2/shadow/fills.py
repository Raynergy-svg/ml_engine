"""Integer-only terminal snapshot assumptions, never broker fill evidence."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from src.axiom2.portfolio.contracts import _integer, _reference, _digest, _timestamp, _boolean
from src.axiom2.contracts.equity_orders import EquityOrderIntent

@dataclass(frozen=True, slots=True, kw_only=True)
class FillPolicy:
    slippage_bps: int
    fee_bps: int
    max_participation_bps: int
    quote_ttl_seconds: int
    max_spread_bps: int
    def __post_init__(self):
        for name in ('slippage_bps','fee_bps','max_participation_bps','max_spread_bps'):
            _integer(getattr(self,name),name,0,10000)
        _integer(self.quote_ttl_seconds,'quote_ttl_seconds',1)

@dataclass(frozen=True, slots=True, kw_only=True)
class Quote:
    instrument_id: str
    bid_micros: int
    ask_micros: int
    quoted_at: datetime
    available_at: datetime
    volume_shares: int
    halted: bool
    def __post_init__(self):
        _reference(self.instrument_id,'instrument_id')
        _integer(self.bid_micros,'bid_micros',1); _integer(self.ask_micros,'ask_micros',1)
        _integer(self.volume_shares,'volume_shares'); _boolean(self.halted,'halted')
        _timestamp(self.quoted_at,'quoted_at'); _timestamp(self.available_at,'available_at')
        if self.bid_micros>self.ask_micros: raise ValueError('crossed quote')
        if self.available_at.astimezone(timezone.utc)<self.quoted_at.astimezone(timezone.utc): raise ValueError('availability precedes quote')

@dataclass(frozen=True, slots=True, kw_only=True)
class MarketSnapshot:
    as_of: datetime
    session_open: datetime
    session_close: datetime
    calendar_digest: str
    source_digest: str
    quotes: tuple[Quote,...]
    def __post_init__(self):
        for name in ('as_of','session_open','session_close'): _timestamp(getattr(self,name),name)
        if self.session_close.astimezone(timezone.utc)<=self.session_open.astimezone(timezone.utc): raise ValueError('invalid session')
        _digest(self.calendar_digest,'calendar_digest'); _digest(self.source_digest,'source_digest')
        if type(self.quotes) is not tuple: raise TypeError('quotes must be immutable tuple')
        for quote in self.quotes:
            if type(quote) is not Quote: raise TypeError('exact Quote required')
            quote.__post_init__()
        if len({q.instrument_id for q in self.quotes})!=len(self.quotes): raise ValueError('duplicate quote')

@dataclass(frozen=True, slots=True, kw_only=True)
class HypotheticalFill:
    intent_id: str
    status: str
    quantity: int
    price_micros: int | None
    fee_cents: int
    uncertainty: tuple[str,...]
    hypothetical: bool = field(default=True,init=False)
    execution_enabled: bool = field(default=False,init=False)
    capital_authorized: bool = field(default=False,init=False)
    def __post_init__(self):
        if self.hypothetical is not True or self.execution_enabled is not False or self.capital_authorized is not False:
            raise ValueError('shadow fill authority flags invalid')
        _reference(self.intent_id,'intent_id')
        if self.status not in {'FILLED','PARTIAL','UNFILLED'}: raise ValueError('invalid fill status')
        _integer(self.quantity,'quantity'); _integer(self.fee_cents,'fee_cents')
        if self.quantity:
            _integer(self.price_micros,'price_micros',1)
            if self.status=='UNFILLED': raise ValueError('unfilled quantity')
        elif self.status!='UNFILLED' or self.price_micros is not None or self.fee_cents:
            raise ValueError('empty fill must be unfilled')
        if type(self.uncertainty) is not tuple or not self.uncertainty: raise ValueError('explicit uncertainty required')
        for reason in self.uncertainty: _reference(reason,'uncertainty')

def hypothetical_fill(intent,quote,market,policy):
    """One terminal snapshot. No queue-position or future-volume claims."""
    if type(intent) is not EquityOrderIntent: raise TypeError('exact EquityOrderIntent required')
    if type(market) is not MarketSnapshot or type(policy) is not FillPolicy: raise TypeError('exact fill inputs required')
    intent.__post_init__(); market.__post_init__(); policy.__post_init__()
    if intent.execution_enabled is not False or intent.capital_authorized is not False:
        raise ValueError('intent authority flags invalid')
    if intent.quantity is None or intent.notional_cents is not None: raise ValueError('whole shares required')
    if intent.mode!='SHADOW': raise ValueError('shadow mode required')
    def empty(reason):
        return HypotheticalFill(intent_id=intent.intent_id,status='UNFILLED',quantity=0,
            price_micros=None,fee_cents=0,uncertainty=('HYPOTHETICAL_NOT_BROKER_EVIDENCE',reason))
    if market.as_of.astimezone(timezone.utc)>=intent.expires_at.astimezone(timezone.utc): return empty('INTENT_EXPIRED')
    if not market.session_open.astimezone(timezone.utc)<=market.as_of.astimezone(timezone.utc)<market.session_close.astimezone(timezone.utc): return empty('SESSION_CLOSED')
    if quote is None: return empty('QUOTE_MISSING')
    if type(quote) is not Quote: raise TypeError('exact Quote required')
    quote.__post_init__()
    if quote.instrument_id!=intent.instrument_id: raise ValueError('quote instrument mismatch')
    if quote.available_at.astimezone(timezone.utc)>market.as_of.astimezone(timezone.utc) or quote.quoted_at.astimezone(timezone.utc)>market.as_of.astimezone(timezone.utc): return empty('QUOTE_UNAVAILABLE')
    if (market.as_of.astimezone(timezone.utc)-quote.quoted_at.astimezone(timezone.utc)).total_seconds()>policy.quote_ttl_seconds: return empty('QUOTE_STALE')
    if quote.halted: return empty('HALTED')
    if (quote.ask_micros-quote.bid_micros)*10000>quote.bid_micros*policy.max_spread_bps:
        return empty('SPREAD_LIMIT')
    if intent.side=='BUY':
        price=(quote.ask_micros*(10000+policy.slippage_bps)+9999)//10000
        crossed=price<=intent.limit_price_micros
    else:
        price=quote.bid_micros*(10000-policy.slippage_bps)//10000
        crossed=price>=intent.limit_price_micros and price>0
    if not crossed: return empty('LIMIT_NOT_CROSSED')
    capacity=quote.volume_shares*policy.max_participation_bps//10000
    quantity=min(intent.quantity,capacity)
    if not quantity: return empty('VOLUME_LIMIT')
    fee=(quantity*price*policy.fee_bps+99_999_999)//100_000_000
    uncertainty=('HYPOTHETICAL_NOT_BROKER_EVIDENCE','QUEUE_POSITION_UNKNOWN')
    if quantity<intent.quantity: uncertainty+=('VOLUME_LIMIT',)
    return HypotheticalFill(intent_id=intent.intent_id,status='FILLED' if quantity==intent.quantity else 'PARTIAL',
        quantity=quantity,price_micros=price,fee_cents=fee,uncertainty=uncertainty)
