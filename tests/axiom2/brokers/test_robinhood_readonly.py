"""Synthetic official-shape fixtures only; no private live response data."""
import asyncio
from datetime import datetime,timezone,timedelta
import pytest

NOW=datetime(2026,10,1,15,tzinfo=timezone.utc)
RAW_ACCOUNT='SYNTHETIC_ACCOUNT_NOT_REAL'

def run(awaitable):return asyncio.run(awaitable)

def response(data):return dict(content=[{'type':'text','text':'ignored private-looking text'}],structuredContent={'data':data,'guide':'ignored'},isError=False)

def fixture():
    from src.axiom2.brokers.robinhood_readonly import RobinhoodReadOnly,OfficialReadTools,DECLARED_CONTRACT_SHA256
    calls=[]
    account=dict(account_number=RAW_ACCOUNT,agentic_allowed=True,brokerage_account_type='individual',deactivated=False,is_default=False,option_level='',permanently_deactivated=False,rhs_account_number='SYNTHETIC_NUMERIC',state='active',type='cash')
    portfolio=dict(cash='25.00',pending_deposits='5.00',total_value='25.00',currency='USD',equity_value='0.00',options_value='0.00',crypto_value='0.00',event_contracts_value='0.00',fixed_income_value='0.00',futures_value='0.00',mutual_funds_value='0.00',buying_power={'buying_power':'20.00','unleveraged_buying_power':'20.00','display_currency':'USD'})
    quote=dict(adjusted_previous_close='10.00',ask_price='10.01',bid_price='10.00',has_traded=True,last_non_reg_trade_price=None,last_trade_price='10.00',previous_close='10.00',previous_close_date='2026-09-30',state='active',symbol='FIXTURE',venue_ask_time=NOW.isoformat(),venue_bid_time=NOW.isoformat(),venue_last_non_reg_trade_time=None,venue_last_trade_time=NOW.isoformat())
    payloads=dict(accounts={'accounts':[account]},portfolio=portfolio,positions={'positions':[]},orders={'orders':[]},quotes={'results':[{'quote':quote}]},tradability={'results':[{'symbol':'FIXTURE','tradeable':True,'state':'active','extended_hours_fractional_tradability':False,'account_type_tradabilities':[{'account_type':'individual','account_type_tradability':'tradable'}]}]})
    async def invoke(name,arguments):calls.append((name,dict(arguments)));return response(payloads[name])
    async def accounts(args):return await invoke('accounts',args)
    async def funds(args):return await invoke('portfolio',args)
    async def positions(args):return await invoke('positions',args)
    async def orders(args):return await invoke('orders',args)
    async def quotes(args):return await invoke('quotes',args)
    async def tradability(args):return await invoke('tradability',args)
    tools=OfficialReadTools(get_accounts=accounts,get_portfolio=funds,get_equity_positions=positions,get_equity_orders=orders,get_equity_quotes=quotes,get_equity_tradability=tradability)
    adapter=RobinhoodReadOnly(tools,account_alias='dedicated-fixture',account_number=RAW_ACCOUNT,contract_sha256=DECLARED_CONTRACT_SHA256,clock=lambda:NOW)
    return adapter,payloads,calls,tools


def test_capabilities_keep_unknown_approval_and_source_time():
    adapter,payloads,calls,tools=fixture();cap=adapter.capabilities()
    assert cap.approval_setting=='UNKNOWN' and cap.execution_enabled is False
    assert cap.provider_snapshot_time_available is False and calls==[]


def test_account_read_preserves_unknown_settlement_and_empty_positions():
    adapter,payloads,calls,tools=fixture();snapshot=run(adapter.account_snapshot('dedicated-fixture'))
    assert snapshot.reported_cash_cents==2500 and snapshot.pending_deposits_cents==500
    assert snapshot.settled_cash_cents is None and snapshot.reserved_cash_cents is None
    assert snapshot.provider_observed_at is None and snapshot.received_at==NOW
    assert snapshot.positions==() and snapshot.positions_complete is True
    assert snapshot.execution_ready is False and snapshot.external_cashflows is None
    assert RAW_ACCOUNT not in repr(snapshot) and 'SYNTHETIC_NUMERIC' not in repr(snapshot)
    with pytest.raises(ValueError,match='UNAVAILABLE'):snapshot.as_risk_account()


@pytest.mark.parametrize('change',['wrong-account','null-accounts','inactive','not-eligible','missing','extra','wrong-type','wrapper-error','text-only','missing-iserror','null-positions','extra-position','unknown-currency'])
def test_account_contract_failures_are_sanitized(change):
    adapter,payloads,calls,tools=fixture()
    if change=='wrong-account':payloads['accounts']['accounts'][0]['account_number']='OTHER_FIXTURE'
    if change=='null-accounts':payloads['accounts']['accounts']=None
    if change=='inactive':payloads['accounts']['accounts'][0]['state']='restricted'
    if change=='not-eligible':payloads['accounts']['accounts'][0]['agentic_allowed']=False
    if change=='missing':del payloads['portfolio']['cash']
    if change=='extra':payloads['portfolio']['secret_unknown']='private'
    if change=='wrong-type':payloads['portfolio']['cash']=25
    if change=='null-positions':payloads['positions']['positions']=None
    if change=='extra-position':payloads['positions']['positions']=[{'symbol':'FIXTURE','unknown':True}]
    if change=='unknown-currency':payloads['portfolio']['currency']='EUR'
    if change in ('wrapper-error','text-only','missing-iserror'):
        async def bad(args):
            if change=='wrapper-error':return {'isError':True,'content':[{'type':'text','text':RAW_ACCOUNT}]}
            if change=='text-only':return {'isError':False,'content':[{'type':'text','text':RAW_ACCOUNT}]}
            return {'structuredContent':{'data':payloads['portfolio']}}
        object.__setattr__(tools,'get_portfolio',bad)
    with pytest.raises(ValueError) as error:run(adapter.account_snapshot('dedicated-fixture'))
    assert RAW_ACCOUNT not in str(error.value) and 'private' not in str(error.value)


def test_wrong_alias_is_rejected_before_any_read():
    adapter,payloads,calls,tools=fixture()
    with pytest.raises(ValueError):run(adapter.account_snapshot('other'))
    assert calls==[]


def test_quote_translation_preserves_two_source_timestamps_and_unknown_session():
    adapter,payloads,calls,tools=fixture();snapshot=run(adapter.market_snapshot(('FIXTURE',)))
    assert snapshot.quotes[0].bid_micros==10_000_000 and snapshot.quotes[0].ask_micros==10_010_000
    assert snapshot.quotes[0].bid_at==NOW and snapshot.quotes[0].ask_at==NOW
    assert snapshot.regular_session_verified is False and snapshot.corporate_actions_clear is False


@pytest.mark.parametrize('change',['stale','future','crossed','zero','missing','duplicate','unrequested','inactive','untradable','null','timestamp','halt'])
def test_quote_and_tradability_negative_facts(change):
    adapter,payloads,calls,tools=fixture();quote=payloads['quotes']['results'][0]['quote'];trad=payloads['tradability']['results'][0]
    if change=='stale':quote['venue_bid_time']=(NOW-timedelta(seconds=31)).isoformat()
    if change=='future':quote['venue_ask_time']=(NOW+timedelta(seconds=1)).isoformat()
    if change=='crossed':quote['bid_price']='11'
    if change=='zero':quote['ask_price']='0'
    if change=='missing':payloads['quotes']['results']=[]
    if change=='duplicate':payloads['quotes']['results']*=2
    if change=='unrequested':quote['symbol']='OTHER'
    if change=='inactive':quote['state']='delisted'
    if change=='untradable':trad['tradeable']=False
    if change=='null':payloads['quotes']['results'][0]['quote']=None
    if change=='timestamp':quote['venue_bid_time']='2026-10-01T15:00:00'
    if change=='halt':trad['internal_halt_sessions']=['regular_hours']
    with pytest.raises(ValueError):run(adapter.market_snapshot(('FIXTURE',)))


def test_opaque_order_pagination_requires_complete_chain_and_keeps_identity():
    adapter,payloads,calls,tools=fixture();pages=iter([{'orders':[],'next':'opaque cursor ? secret-like'}, {'orders':[]}])
    async def orders(args):calls.append(('orders',dict(args)));return response(next(pages))
    object.__setattr__(tools,'get_equity_orders',orders)
    page=run(adapter.order_observations('dedicated-fixture'))
    assert page.complete is True and page.orders==() and page.provider_observed_at is None
    assert calls[-1][1]['cursor']=='opaque cursor ? secret-like'
    assert all(set(arguments)<= {'account_number','cursor'} for name,arguments in calls if name=='orders')


def test_pagination_cycle_fails_closed():
    adapter,payloads,calls,tools=fixture();payloads['orders']['next']='repeat'
    with pytest.raises(ValueError,match='PAGINATION'):run(adapter.order_observations('dedicated-fixture'))


def test_transport_exception_is_redacted():
    adapter,payloads,calls,tools=fixture()
    async def fail(args):raise RuntimeError(RAW_ACCOUNT+' secret')
    object.__setattr__(tools,'get_equity_orders',fail)
    with pytest.raises(ValueError,match='PROVIDER_READ_FAILED') as error:run(adapter.order_observations('dedicated-fixture'))
    assert RAW_ACCOUNT not in str(error.value) and error.value.__cause__ is None


def test_schema_pin_and_side_effect_surface_fail_closed():
    from src.axiom2.brokers.robinhood_readonly import RobinhoodReadOnly,OfficialReadTools
    adapter,payloads,calls,tools=fixture()
    with pytest.raises(ValueError,match='SCHEMA'):RobinhoodReadOnly(tools,account_alias='fixture',account_number=RAW_ACCOUNT,contract_sha256='0'*64,clock=lambda:NOW)
    assert not any(hasattr(adapter,name) for name in ('submit','review','cancel','call_tool','place_order'))
    assert not any(hasattr(tools,name) for name in ('review_equity_order','place_equity_order','cancel_equity_order'))
    assert calls==[]


def test_decimal_normalization_never_uses_ambient_rounding():
    from decimal import localcontext
    from src.axiom2.brokers.robinhood_readonly import _units
    with pytest.raises(ValueError):_units('1.00000000000000000000000000001',100)
    with localcontext() as context:
        context.prec=2
        assert _units('123.45',100)==12345


def order_fixture():
    return dict(average_price='10.00',created_at=(NOW-timedelta(seconds=10)).isoformat(),cumulative_quantity='1.00',dollar_based_amount=None,
        executions=[dict(fees='0.01',id='SYNTHETIC_FILL_1',price='10.00',quantity='1.00',timestamp=(NOW-timedelta(seconds=5)).isoformat())],fees='0.01',
        id='SYNTHETIC_ORDER_1',instrument_id='SYNTHETIC_INSTRUMENT_1',last_transaction_at=(NOW-timedelta(seconds=5)).isoformat(),market_hours='regular_hours',
        placed_agent='agentic',price='10.00',quantity='2.00',ref_id='SYNTHETIC_BROKER_APPROVAL_REF',side='buy',state='partially_filled',stop_price=None,
        symbol='FIXTURE',time_in_force='gfd',trigger='immediate',type='limit')


def test_incomplete_execution_list_is_not_attested_complete():
    adapter,payloads,calls,tools=fixture();value=order_fixture();value['executions']=[];payloads['orders']['orders']=[value]
    orders=run(adapter.order_observations('dedicated-fixture'))
    assert orders.orders[0].execution_details_complete is False


def test_nonempty_position_retains_fractional_and_zero_holdings():
    adapter,payloads,calls,tools=fixture()
    def position(symbol,quantity,kind):return dict(symbol=symbol,type=kind,quantity=quantity,intraday_quantity='0.00',shares_available_for_sells=quantity,shares_held_for_asset_transfer='0.00',shares_held_for_options_events='0.00',shares_held_for_sells='0.00',shares_held_for_stock_grants='0.00',shares_pending_from_options_events='0.00',average_buy_price=None)
    payloads['positions']['positions']=[position('FIXTURE','2.50','long'),position('DELISTED_FIXTURE','0.00','empty')]
    result=run(adapter.account_snapshot('dedicated-fixture'))
    assert tuple(item.quantity for item in result.positions)==('2.50','0.00')
    assert result.positions[1].symbol=='DELISTED_FIXTURE' and not result.corporate_actions_clear
    assert result.request_started_at<=result.received_at and result.local_observation_id


def test_null_buying_power_is_distinct_from_available_zero():
    adapter,payloads,calls,tools=fixture();payloads['portfolio']['buying_power']=None
    missing=run(adapter.account_snapshot('dedicated-fixture'));assert missing.buying_power_cents is None
    payloads['portfolio']['buying_power']=dict(buying_power='0.00',unleveraged_buying_power='0.00',display_currency='USD')
    zero=run(adapter.account_snapshot('dedicated-fixture'));assert zero.buying_power_cents==0 and zero.settled_cash_cents is None


@pytest.mark.parametrize('field',['accounts','positions','orders'])
def test_nullable_rows_are_explicit_schema_gaps(field):
    adapter,payloads,calls,tools=fixture();payloads[field][field]=[None]
    with pytest.raises(ValueError,match='SCHEMA'):run(adapter.order_observations('dedicated-fixture') if field=='orders' else adapter.account_snapshot('dedicated-fixture'))


def test_failed_middle_page_never_returns_complete_history():
    adapter,payloads,calls,tools=fixture();count=0
    async def orders(args):
        nonlocal count
        count+=1
        if count==1:return response({'orders':[order_fixture()],'next':'SYNTHETIC_NEXT'})
        raise RuntimeError('private provider response '+RAW_ACCOUNT)
    object.__setattr__(tools,'get_equity_orders',orders)
    with pytest.raises(ValueError,match='PROVIDER_READ_FAILED'):run(adapter.order_observations('dedicated-fixture'))
    assert count==2


@pytest.mark.parametrize('change',['unknown-state','missing-ref','dollar-order','partial-cancel','fill-gap','quantity-overrun','early-fill','fee-gap'])
def test_nonempty_orders_retain_facts_without_claiming_request_or_settlement_truth(change):
    adapter,payloads,calls,tools=fixture();order=order_fixture()
    if change=='unknown-state':order['state']='future_provider_state'
    if change=='missing-ref':del order['ref_id']
    if change=='dollar-order':order.update(quantity=None,cumulative_quantity='0',average_price=None,executions=None,dollar_based_amount={'amount':'20.00','currency_code':'USD'},fees='0.00')
    if change=='partial-cancel':order['state']='partially_filled_rest_cancelled'
    if change=='fill-gap':order['executions'][0]['quantity']='0.50'
    if change=='quantity-overrun':order['cumulative_quantity']='3'
    if change=='early-fill':order['executions'][0]['timestamp']=(NOW-timedelta(minutes=1)).isoformat()
    if change=='fee-gap':order['executions'][0]['fees']='0.00'
    payloads['orders']['orders']=[order]
    if change in ('quantity-overrun','early-fill'):
        with pytest.raises(ValueError):run(adapter.order_observations('dedicated-fixture'))
        return
    result=run(adapter.order_observations('dedicated-fixture'));value=result.orders[0]
    assert result.complete and result.coverage=='ordinary-equity-orders-only' and not result.stable_snapshot_verified
    assert result.advanced_order_coverage=='UNAVAILABLE' and result.provider_observed_at is None
    assert value.request_identity_kind=='BROKER_REF_UNVERIFIED'
    if change=='unknown-state':assert value.state=='future_provider_state'
    if change=='missing-ref':assert value.broker_ref_id is None
    if change=='dollar-order':assert value.quantity is None and value.dollar_based_amount_cents==2000 and not value.execution_details_complete
    if change=='partial-cancel':assert value.state=='partially_filled_rest_cancelled' and value.executions
    if change in ('fill-gap','fee-gap'):assert value.execution_details_complete is False
    assert all('created_at_gte' not in arguments for name,arguments in calls)


def test_preserves_nanoseconds_and_selects_newer_nonregular_trade():
    adapter,payloads,calls,tools=fixture();quote=payloads['quotes']['results'][0]['quote']
    quote.update(venue_bid_time='2026-10-01T14:59:59.987654321Z',venue_ask_time='2026-10-01T14:59:59.123456789Z',venue_last_trade_time='2026-10-01T14:00:00Z',last_non_reg_trade_price='10.02',venue_last_non_reg_trade_time='2026-10-01T14:59:59.111111111Z')
    result=run(adapter.market_snapshot(('FIXTURE',))).quotes[0]
    assert result.bid_source_time.endswith('987654321Z') and result.ask_source_time.endswith('123456789Z')
    assert result.selected_trade_session=='NONREGULAR' and result.selected_trade_price_micros==10_020_000
    assert not result.selected_trade_stale and not result.official_close_available


def test_future_submicrosecond_timestamp_is_not_truncated_into_present():
    adapter,payloads,calls,tools=fixture();payloads['quotes']['results'][0]['quote']['venue_bid_time']='2026-10-01T15:00:00.000000001Z'
    with pytest.raises(ValueError,match='FUTURE'):run(adapter.market_snapshot(('FIXTURE',)))


def test_stale_trade_is_retained_as_stale_even_with_fresh_book():
    adapter,payloads,calls,tools=fixture();payloads['quotes']['results'][0]['quote']['venue_last_trade_time']=(NOW-timedelta(hours=2)).isoformat()
    result=run(adapter.market_snapshot(('FIXTURE',))).quotes[0]
    assert result.selected_trade_stale is True


def test_absent_and_null_average_cost_remain_distinct_and_unsettled_is_informational():
    adapter,payloads,calls,tools=fixture()
    position=dict(symbol='FIXTURE',type='long',quantity='1',intraday_quantity='0',shares_available_for_sells='1',shares_held_for_asset_transfer='0',shares_held_for_options_events='0',shares_held_for_sells='0',shares_held_for_stock_grants='0',shares_pending_from_options_events='0')
    payloads['positions']['positions']=[position]
    absent=run(adapter.account_snapshot('dedicated-fixture'));assert absent.positions[0].average_cost_field_present is False
    position['average_buy_price']=None;payloads['accounts']['accounts'][0]['unsettled_funds']='0.00'
    explicit=run(adapter.account_snapshot('dedicated-fixture'))
    assert explicit.positions[0].average_cost_field_present is True and explicit.positions[0].average_cost_cents is None
    assert explicit.reported_unsettled_funds_cents==0 and explicit.unsettled_funds_field_present is True and explicit.settled_cash_cents is None


def test_null_next_is_a_contract_gap_not_terminal_history():
    adapter,payloads,calls,tools=fixture();payloads['orders']['next']=None
    with pytest.raises(ValueError,match='SCHEMA'):run(adapter.order_observations('dedicated-fixture'))


@pytest.mark.parametrize('change',['fill-future-ns','creation-future-ns','update-future-ns','fill-before-creation-ns'])
def test_order_source_time_comparisons_preserve_nanosecond_precision(change):
    adapter,payloads,calls,tools=fixture();order=order_fixture()
    if change=='fill-future-ns':order['executions'][0]['timestamp']='2026-10-01T15:00:00.000000001Z'
    if change=='creation-future-ns':order.update(created_at='2026-10-01T15:00:00.000000001Z',last_transaction_at=None,executions=[],cumulative_quantity='0',fees='0.00')
    if change=='update-future-ns':order['last_transaction_at']='2026-10-01T15:00:00.000000001Z'
    if change=='fill-before-creation-ns':order.update(created_at='2026-10-01T14:59:59.000000900Z',last_transaction_at='2026-10-01T14:59:59.999999999Z');order['executions'][0]['timestamp']='2026-10-01T14:59:59.000000100Z'
    payloads['orders']['orders']=[order]
    with pytest.raises(ValueError):run(adapter.order_observations('dedicated-fixture'))


def test_read_timeout_is_bounded_and_never_returns_cached_readiness():
    from src.axiom2.brokers.robinhood_readonly import RobinhoodReadOnly,DECLARED_CONTRACT_SHA256
    adapter,payloads,calls,tools=fixture()
    async def blocked(args):await asyncio.Future()
    object.__setattr__(tools,'get_accounts',blocked)
    limited=RobinhoodReadOnly(tools,account_alias='dedicated-fixture',account_number=RAW_ACCOUNT,contract_sha256=DECLARED_CONTRACT_SHA256,clock=lambda:NOW,read_timeout_seconds=.01)
    with pytest.raises(ValueError,match='READ_FAILED'):run(limited.account_snapshot('dedicated-fixture'))
    assert limited.capabilities().live_connection_verified is False


def test_declared_contract_probe_never_establishes_connection(tmp_path):
    import runpy
    from pathlib import Path
    module=runpy.run_path(str(Path(__file__).parents[3]/'scripts'/'axiom2_verify_readonly.py'))
    result=module['verify_contracts'](module['ROOT']/'config/axiom2/robinhood_readonly_contracts.json')
    assert result['status']=='PASS' and not result['live_connection_verified'] and result['broker_actions']==0
    changed=tmp_path/'changed.json';changed.write_text('{}')
    with pytest.raises(ValueError,match='SCHEMA'):module['verify_contracts'](changed)
