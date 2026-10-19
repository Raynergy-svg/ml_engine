"""Fixed official MCP reads. Runtime service bindings are trusted, not a sandbox.

No credential file, provider URL, arbitrary forwarding, broker preview, write or
security-setting operation exists. Local receive times never become provider
observation timestamps. Declared contracts are pinned; connection/empty reads
are not live completeness, settled cash, approval or capital qualification.
"""
from dataclasses import dataclass,field
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
from typing import Callable
from fractions import Fraction
from uuid import uuid4
import re
import asyncio
from pydantic import BaseModel,ConfigDict
from src.evidence.canonical import canonical_bytes
from src.axiom2.brokers.contracts import (CapabilitySnapshot,AccountSnapshot,PositionObservation,
    MarketSnapshot,QuoteObservation,OrderObservationPage,OrderObservation,FillObservation)

# SHA256 of the exact six official tool declarations captured at discovery.
DECLARED_CONTRACT_SHA256='a8acfcc10e137fad609ef1467dd329b59e28d4122bd0173f52426c70b8e93be6'

@dataclass(frozen=True,slots=True,kw_only=True)
class OfficialReadTools:
    get_accounts:Callable=field(repr=False)
    get_portfolio:Callable=field(repr=False)
    get_equity_positions:Callable=field(repr=False)
    get_equity_orders:Callable=field(repr=False)
    get_equity_quotes:Callable=field(repr=False)
    get_equity_tradability:Callable=field(repr=False)
    def __post_init__(self):
        if not all(callable(binding) for binding in (self.get_accounts,self.get_portfolio,self.get_equity_positions,self.get_equity_orders,self.get_equity_quotes,self.get_equity_tradability)):
            raise TypeError('EXACT_CALLABLE_READ_BINDINGS_REQUIRED')

class Wire(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True,allow_inf_nan=False)

class Account(Wire):
    account_number:str;agentic_allowed:bool;brokerage_account_type:str
    deactivated:bool;is_default:bool;option_level:str;permanently_deactivated:bool
    rhs_account_number:str;state:str;type:str
    affiliate:str|None=None;management_type:str|None=None;nickname:str|None=None
    rhc_account_number:str|None=None;unsettled_funds:str|None=None;user_option_level:str|None=None
class Accounts(Wire):
    accounts:tuple[Account,...]|None
class BuyingPower(Wire):
    buying_power:str;display_currency:str;unleveraged_buying_power:str
    intraday_buying_power:str|None=None;off_intraday_buying_power:str|None=None
class CryptoBuyingPower(Wire):
    buying_power:str
class Portfolio(Wire):
    buying_power:BuyingPower|None;cash:str;crypto_value:str;currency:str;equity_value:str
    event_contracts_value:str;fixed_income_value:str;futures_value:str;mutual_funds_value:str
    options_value:str;pending_deposits:str;total_value:str
    crypto_buying_power:CryptoBuyingPower|None=None
class Position(Wire):
    intraday_quantity:str;quantity:str;shares_available_for_sells:str
    shares_held_for_asset_transfer:str;shares_held_for_options_events:str
    shares_held_for_sells:str;shares_held_for_stock_grants:str;shares_pending_from_options_events:str
    symbol:str;type:str;average_buy_price:str|None=None
class Positions(Wire):
    positions:tuple[Position,...]|None;next:str=''
class Quote(Wire):
    adjusted_previous_close:str;ask_price:str;bid_price:str;has_traded:bool
    last_non_reg_trade_price:str|None;last_trade_price:str;previous_close:str
    previous_close_date:str|None;state:str;symbol:str;venue_ask_time:str
    venue_bid_time:str;venue_last_non_reg_trade_time:str|None;venue_last_trade_time:str
class Close(Wire):
    date:str|None;interpolated:bool|None;price:str|None;source:str|None;symbol:str
class QuoteResult(Wire):
    quote:Quote|None;close:Close|None=None
class Quotes(Wire):
    results:tuple[QuoteResult,...]|None;closes_error:str|None=None
class AccountTradability(Wire):
    account_type:str;account_type_tradability:str
class Tradability(Wire):
    symbol:str;tradeable:bool;extended_hours_fractional_tradability:bool
    account_type_tradabilities:tuple[AccountTradability,...]|None=None
    all_day_tradability:str|None=None;country:str|None=None;fractional_tradability:str|None=None
    internal_halt_details:str|None=None;internal_halt_end_time:str|None=None
    internal_halt_reason:str|None=None;internal_halt_sessions:tuple[str,...]|None=None
    internal_halt_start_time:str|None=None;name:str|None=None;short_selling_tradability:str|None=None
    simple_name:str|None=None;state:str|None=None;twenty_four_seven_tradability:str|None=None
class Tradabilities(Wire):
    results:tuple[Tradability,...]|None;not_found:tuple[str,...]|None=None
class Execution(Wire):
    fees:str;id:str;price:str;quantity:str;timestamp:str
class DollarAmount(Wire):
    amount:str;currency_code:str
class Order(Wire):
    average_price:str|None;created_at:str;cumulative_quantity:str
    dollar_based_amount:DollarAmount|None;executions:tuple[Execution,...]|None
    fees:str;id:str;instrument_id:str;last_transaction_at:str|None;market_hours:str
    placed_agent:str;price:str|None;quantity:str|None;side:str;state:str
    stop_price:str|None;symbol:str;time_in_force:str;trigger:str;type:str
    ref_id:str|None=None;reject_reason:str|None=None
class Orders(Wire):
    orders:tuple[Order,...]|None;next:str=''


def _number(value):
    if type(value) is not str or not re.fullmatch(r'-?\d+(?:\.\d+)?',value) or len(value)>80:raise ValueError('INVALID_PROVIDER_DECIMAL')
    try:number=Decimal(value)
    except InvalidOperation:raise ValueError('INVALID_PROVIDER_DECIMAL') from None
    if not number.is_finite():raise ValueError('INVALID_PROVIDER_DECIMAL')
    return number

def _units(value,scale):
    scaled=Fraction(_number(value))*scale
    if scaled.denominator!=1:raise ValueError('UNSUPPORTED_PROVIDER_PRECISION')
    return int(scaled)

def _time(value):
    if type(value) is not str or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})',value):raise ValueError('PROVIDER_TIME_UNAVAILABLE')
    try:parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:raise ValueError('INVALID_PROVIDER_TIME') from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:raise ValueError('PROVIDER_TIME_UNAVAILABLE')
    return parsed.astimezone(timezone.utc)

def _age(now,source):
    parsed=_time(source);delta=now-parsed
    age=Fraction(delta.days*86400+delta.seconds)+Fraction(delta.microseconds,10**6)
    fraction=re.search(r'\.(\d+)',source)
    if fraction and len(fraction.group(1))>6:
        digits=fraction.group(1);age-=Fraction(int(digits[6:]),10**len(digits))
    return age

class RobinhoodReadOnly:
    __slots__=('_tools','_alias','_account','_clock','_read_timeout')
    def __init__(self,tools,*,account_alias,account_number,contract_sha256,clock,read_timeout_seconds=10):
        if type(tools) is not OfficialReadTools:raise TypeError('EXACT_OFFICIAL_READ_BINDINGS_REQUIRED')
        if contract_sha256!=DECLARED_CONTRACT_SHA256:raise ValueError('PROVIDER_SCHEMA_DRIFT')
        if type(account_alias) is not str or not re.fullmatch(r'[a-z][a-z0-9-]{0,63}',account_alias):raise ValueError('INVALID_ACCOUNT_ALIAS')
        if type(account_number) is not str or not account_number or not callable(clock):raise ValueError('TRUSTED_ACCOUNT_BINDING_REQUIRED')
        if account_alias==account_number:raise ValueError('PRIVATE_ACCOUNT_CANNOT_BE_ALIAS')
        if type(read_timeout_seconds) not in (int,float) or not 0<read_timeout_seconds<=30:raise ValueError('BOUNDED_READ_TIMEOUT_REQUIRED')
        self._tools=tools;self._alias=account_alias;self._account=account_number;self._clock=clock;self._read_timeout=read_timeout_seconds

    def capabilities(self):return CapabilitySnapshot(declared_contract_sha256=DECLARED_CONTRACT_SHA256)

    def _now(self):
        now=self._clock()
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset()!=timezone.utc.utcoffset(now):raise ValueError('TRUSTED_UTC_CLOCK_REQUIRED')
        return now

    def _window(self,start,identity):
        end=self._now()
        if end<start:raise ValueError('LOCAL_READ_CLOCK_REGRESSED')
        return dict(local_observation_id=identity,request_started_at=start,received_at=end)

    def _alias_check(self,alias):
        if type(alias) is not str or alias!=self._alias:raise ValueError('ACCOUNT_ALIAS_NOT_ALLOWLISTED')

    async def _read(self,binding,arguments,contract):
        # Exception strings/response text/guide are never propagated or logged.
        try:result=await asyncio.wait_for(binding(dict(arguments)),timeout=self._read_timeout)
        except Exception:raise ValueError('PROVIDER_READ_FAILED') from None
        if type(result) is not dict or result.get('isError') is not False:raise ValueError('PROVIDER_READ_FAILED')
        structured=result.get('structuredContent')
        if type(structured) is not dict or set(structured)!={'data','guide'} or type(structured['guide']) is not str:raise ValueError('PROVIDER_STRUCTURED_RESPONSE_UNAVAILABLE')
        try:return contract.model_validate_json(canonical_bytes(structured['data']),strict=True)
        except Exception:raise ValueError('PROVIDER_SCHEMA_OR_VALUE_INVALID') from None

    async def _account_record(self):
        result=await self._read(self._tools.get_accounts,{},Accounts)
        if result.accounts is None:raise ValueError('ACCOUNT_COVERAGE_UNAVAILABLE')
        matches=[account for account in result.accounts if account.account_number==self._account]
        if len(matches)!=1:raise ValueError('ACCOUNT_BINDING_MISMATCH')
        account=matches[0]
        if not account.agentic_allowed or account.state!='active' or account.deactivated or account.permanently_deactivated:raise ValueError('ACCOUNT_NOT_ELIGIBLE')
        return account

    async def _pages(self,binding,contract,field_name):
        cursor=None;seen=set();rows=[]
        for _ in range(100):
            arguments={'account_number':self._account}
            if cursor is not None:arguments['cursor']=cursor
            page=await self._read(binding,arguments,contract)
            values=page.positions if field_name=='positions' else page.orders
            if values is None:raise ValueError('PAGE_COVERAGE_UNAVAILABLE')
            rows.extend(values)
            if len(rows)>10000:raise ValueError('PAGINATION_LIMIT')
            if not page.next:return tuple(rows)
            if page.next in seen:raise ValueError('PAGINATION_CYCLE')
            seen.add(page.next);cursor=page.next
        raise ValueError('PAGINATION_LIMIT')

    async def account_snapshot(self,account_alias):
        self._alias_check(account_alias);start=self._now();identity=str(uuid4());account=await self._account_record()
        portfolio=await self._read(self._tools.get_portfolio,{'account_number':self._account},Portfolio)
        positions=await self._pages(self._tools.get_equity_positions,Positions,'positions')
        if portfolio.currency!='USD' or (portfolio.buying_power and portfolio.buying_power.display_currency!='USD'):raise ValueError('UNSUPPORTED_ACCOUNT_CURRENCY')
        converted=[];seen=set()
        for position in positions:
            if position.symbol in seen:raise ValueError('DUPLICATE_POSITION')
            seen.add(position.symbol)
            for value in (position.quantity,position.intraday_quantity,position.shares_available_for_sells,position.shares_held_for_sells,position.shares_held_for_asset_transfer,position.shares_held_for_options_events,position.shares_held_for_stock_grants,position.shares_pending_from_options_events):_number(value)
            converted.append(PositionObservation(symbol=position.symbol,quantity=position.quantity,available_for_sells=position.shares_available_for_sells,held_for_sells=position.shares_held_for_sells,held_for_transfer=position.shares_held_for_asset_transfer,held_for_options=position.shares_held_for_options_events,held_for_stock_grants=position.shares_held_for_stock_grants,pending_from_options=position.shares_pending_from_options_events,intraday_quantity=position.intraday_quantity,average_cost_cents=_units(position.average_buy_price,100) if position.average_buy_price is not None else None,average_cost_field_present='average_buy_price' in position.model_fields_set,position_type=position.type))
        assets=dict(equity=_units(portfolio.equity_value,100),options=_units(portfolio.options_value,100),crypto=_units(portfolio.crypto_value,100),events=_units(portfolio.event_contracts_value,100),fixed_income=_units(portfolio.fixed_income_value,100),futures=_units(portfolio.futures_value,100),mutual_funds=_units(portfolio.mutual_funds_value,100))
        return AccountSnapshot(account_alias=self._alias,**self._window(start,identity),reported_cash_cents=_units(portfolio.cash,100),pending_deposits_cents=_units(portfolio.pending_deposits,100),portfolio_value_cents=_units(portfolio.total_value,100),asset_values_cents=assets,unsupported_asset_holdings_present=any(value!=0 for name,value in assets.items() if name!='equity'),reported_unsettled_funds_cents=_units(account.unsettled_funds,100) if account.unsettled_funds is not None else None,unsettled_funds_field_present='unsettled_funds' in account.model_fields_set,buying_power_cents=_units(portfolio.buying_power.buying_power,100) if portfolio.buying_power else None,unleveraged_buying_power_cents=_units(portfolio.buying_power.unleveraged_buying_power,100) if portfolio.buying_power else None,positions=tuple(converted),positions_complete=True)

    async def market_snapshot(self,instruments):
        if type(instruments) is not tuple or not 1<=len(instruments)<=10 or len(set(instruments))!=len(instruments) or any(type(symbol) is not str or not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,19}',symbol) for symbol in instruments):raise ValueError('EXACT_EQUITY_SYMBOLS_REQUIRED')
        start=self._now();identity=str(uuid4());account=await self._account_record()
        quotes=await self._read(self._tools.get_equity_quotes,{'symbols':list(instruments)},Quotes)
        tradabilities=await self._read(self._tools.get_equity_tradability,{'account_number':self._account,'symbols':list(instruments)},Tradabilities)
        if quotes.results is None or tradabilities.results is None or tradabilities.not_found:raise ValueError('MARKET_COVERAGE_UNAVAILABLE')
        trads={value.symbol:value for value in tradabilities.results}
        if len(trads)!=len(tradabilities.results) or set(trads)!=set(instruments):raise ValueError('TRADABILITY_COVERAGE_MISMATCH')
        now=self._now();converted=[];seen=set()
        for value in quotes.results:
            quote=value.quote
            if quote is None or quote.symbol in seen or quote.symbol not in instruments:raise ValueError('QUOTE_COVERAGE_MISMATCH')
            seen.add(quote.symbol);trad=trads[quote.symbol]
            matched=[entry.account_type_tradability for entry in trad.account_type_tradabilities or () if entry.account_type==account.brokerage_account_type]
            if not quote.has_traded or quote.state!='active' or trad.state!='active' or not trad.tradeable or matched!=['tradable'] or 'regular_hours' in (trad.internal_halt_sessions or ()):raise ValueError('REGULAR_EQUITY_TRADABILITY_UNAVAILABLE')
            bid=_units(quote.bid_price,10**6);ask=_units(quote.ask_price,10**6);bid_at=_time(quote.venue_bid_time);ask_at=_time(quote.venue_ask_time)
            if bid<=0 or ask<=0 or bid>ask:raise ValueError('INVALID_BID_ASK')
            if any(not 0<=_age(now,source)<=30 for source in (quote.venue_bid_time,quote.venue_ask_time)):raise ValueError('STALE_OR_FUTURE_QUOTE')
            trade_sources=[(quote.venue_last_trade_time,quote.last_trade_price,'REGULAR')]
            if (quote.last_non_reg_trade_price is None)!=(quote.venue_last_non_reg_trade_time is None):raise ValueError('NONREGULAR_TRADE_PAIR_INCOMPLETE')
            if quote.last_non_reg_trade_price is not None:trade_sources.append((quote.venue_last_non_reg_trade_time,quote.last_non_reg_trade_price,'NONREGULAR'))
            if any(_age(now,time)<0 or _units(price,10**6)<=0 for time,price,_ in trade_sources):raise ValueError('INVALID_TRADE_SOURCE')
            trade_time,trade_price,trade_session=min(trade_sources,key=lambda item:_age(now,item[0]))
            close=value.close
            if close is not None and close.symbol!=quote.symbol:raise ValueError('OFFICIAL_CLOSE_SYMBOL_MISMATCH')
            close_available=close is not None and close.price is not None and close.date is not None and not quotes.closes_error
            converted.append(QuoteObservation(symbol=quote.symbol,bid_micros=bid,ask_micros=ask,bid_at=bid_at,ask_at=ask_at,
                bid_source_time=quote.venue_bid_time,ask_source_time=quote.venue_ask_time,selected_trade_price_micros=_units(trade_price,10**6),
                selected_trade_at=_time(trade_time),selected_trade_source_time=trade_time,selected_trade_session=trade_session,
                selected_trade_stale=_age(now,trade_time)>30,official_close_price_micros=_units(close.price,10**6) if close_available else None,
                official_close_date=close.date if close_available else None,official_close_available=close_available,
                instrument_state=quote.state,account_type_tradability=matched[0]))
        if seen!=set(instruments):raise ValueError('QUOTE_COVERAGE_MISMATCH')
        window=self._window(start,identity)
        if any(not 0<=_age(window['received_at'],source)<=30 for quote in converted for source in (quote.bid_source_time,quote.ask_source_time)):raise ValueError('QUOTE_EXPIRED_DURING_NORMALIZATION')
        return MarketSnapshot(**window,quotes=tuple(converted))

    async def order_observations(self,account_alias):
        self._alias_check(account_alias);start=self._now();identity=str(uuid4());await self._account_record()
        orders=await self._pages(self._tools.get_equity_orders,Orders,'orders')
        seen=set();executions=set();converted=[]
        now=self._now()
        for order in orders:
            if order.id in seen or not order.id or not order.instrument_id or not order.symbol:raise ValueError('DUPLICATE_OR_MISSING_ORDER_IDENTITY')
            seen.add(order.id);fills=[]
            for fill in order.executions or ():
                if not fill.id or fill.id in executions:raise ValueError('DUPLICATE_EXECUTION_IDENTITY')
                executions.add(fill.id);occurred=_time(fill.timestamp)
                if _age(now,fill.timestamp)<0:raise ValueError('FUTURE_EXECUTION_TIME')
                if _number(fill.quantity)<=0 or _units(fill.price,10**6)<=0 or _units(fill.fees,100)<0:raise ValueError('INVALID_FILL')
                fills.append(FillObservation(execution_id=fill.id,price_micros=_units(fill.price,10**6),quantity=fill.quantity,fees_cents=_units(fill.fees,100),occurred_at=occurred,source_time=fill.timestamp))
            created=_time(order.created_at);updated=_time(order.last_transaction_at) if order.last_transaction_at else None
            if _age(now,order.created_at)<0 or updated is not None and (_age(now,order.last_transaction_at)<0 or _age(now,order.last_transaction_at)>_age(now,order.created_at)):raise ValueError('INVALID_ORDER_TIME')
            if _number(order.cumulative_quantity)<0 or order.quantity is not None and _number(order.quantity)<0 or _units(order.fees,100)<0:raise ValueError('INVALID_ORDER_VALUES')
            if order.quantity is not None and _number(order.cumulative_quantity)>_number(order.quantity):raise ValueError('ORDER_FILL_QUANTITY_EXCEEDS_REQUEST')
            if any(_age(now,fill.source_time)>_age(now,order.created_at) for fill in fills):raise ValueError('FILL_PRECEDES_ORDER')
            fill_quantity=sum((Fraction(_number(fill.quantity)) for fill in fills),Fraction())
            detail_complete=order.executions is not None and fill_quantity==Fraction(_number(order.cumulative_quantity)) and sum(fill.fees_cents for fill in fills)==_units(order.fees,100) and (not fills or updated is not None and all(_age(now,fill.source_time)>=_age(now,order.last_transaction_at) for fill in fills))
            if order.dollar_based_amount is not None and order.dollar_based_amount.currency_code!='USD':raise ValueError('UNSUPPORTED_ORDER_CURRENCY')
            converted.append(OrderObservation(order_id=order.id,instrument_id=order.instrument_id,symbol=order.symbol,side=order.side,state=order.state,state_supported=order.state in ('new','queued','confirmed','unconfirmed','partially_filled','filled','cancelled','rejected','failed','voided','pending_cancelled','partially_filled_rest_cancelled','locating','locate_failed'),quantity=order.quantity,cumulative_quantity=order.cumulative_quantity,fees_cents=_units(order.fees,100),created_at=created,last_transaction_at=updated,broker_ref_id=order.ref_id,placed_agent=order.placed_agent,order_type=order.type,trigger=order.trigger,time_in_force=order.time_in_force,market_hours=order.market_hours,limit_price_micros=_units(order.price,10**6) if order.price is not None else None,stop_price_micros=_units(order.stop_price,10**6) if order.stop_price is not None else None,average_price_micros=_units(order.average_price,10**6) if order.average_price is not None else None,dollar_based_amount_cents=_units(order.dollar_based_amount.amount,100) if order.dollar_based_amount is not None else None,created_source_time=order.created_at,last_transaction_source_time=order.last_transaction_at,executions=tuple(fills),execution_details_complete=detail_complete))
        return OrderObservationPage(account_alias=self._alias,**self._window(start,identity),orders=tuple(converted),complete=True)
