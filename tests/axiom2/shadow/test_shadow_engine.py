"""Offline shadow behavior, with explicit absence of production authority."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import runpy
import pytest

def api():
    from src.axiom2.shadow import fills, engine
    return fills, engine

def fixture():
    helper=runpy.run_path(str(Path(__file__).parents[1]/'portfolio'/'test_order_intents.py'))
    _,request,_,decision=helper['base']()
    account,caps=helper['refs'](request)
    return request,decision,account,caps

def test_snapshot_fill_behavior_is_available():
    import src.axiom2.shadow as shadow
    assert hasattr(shadow,'ShadowEngine'), 'deterministic shadow step is missing'

class MemoryJournal:
    def __init__(self): self.rows={}; self.head=None; self.completions=0
    def reserve(self,intent,*,context_digest):
        from types import SimpleNamespace
        prior=self.rows.get(intent.intent_id)
        if prior:
            if prior.context_digest!=context_digest: raise ValueError('changed replay')
            return prior
        row=SimpleNamespace(context_digest=context_digest,state='RESERVED',result=None)
        self.rows[intent.intent_id]=row
        return row
    def complete(self,intent_id,*,result,expected_head):
        row=self.rows[intent_id]; row.result=result; row.state='COMPLETED'
        self.completions+=1; self.head=str(self.completions)
        return row

def inputs():
    f,e=api(); request,decision,account,caps=fixture()
    policy=f.FillPolicy(slippage_bps=0,fee_bps=10,max_participation_bps=1000,quote_ttl_seconds=30,max_spread_bps=50)
    quotes=tuple(f.Quote(instrument_id=o.instrument_id,bid_micros=o.bid_micros,ask_micros=o.ask_micros,
        quoted_at=o.quoted_at,available_at=request.as_of,volume_shares=1000,halted=False) for o in request.candidates)
    market=f.MarketSnapshot(as_of=request.as_of,session_open=request.session_open,session_close=request.session_close,
        calendar_digest=request.calendar_digest,source_digest='a'*64,quotes=quotes)
    return f,e,request,decision,account,caps,policy,market

def test_engine_replay_and_restart_are_identical_and_disabled():
    f,e,request,decision,account,caps,policy,market=inputs(); journal=MemoryJournal()
    first=e.ShadowEngine(request=request,capabilities=caps,policy=policy,journal=journal).step(decision,market,account)
    second=e.ShadowEngine(request=request,capabilities=caps,policy=policy,journal=journal).step(decision,market,account)
    assert first==second and len(first.fills)==5 and journal.completions==5
    assert first.hypothetical and not first.execution_enabled and not first.capital_authorized
    assert all(x.quantity==15 and x.status=='FILLED' and x.fee_cents==151 for x in first.fills)
    with pytest.raises(ValueError,match='replay'):
        e.ShadowEngine(request=request,capabilities=caps,policy=replace(policy,fee_bps=11),journal=journal).step(decision,market,account)

@pytest.mark.parametrize('change,reason',[
    ({'volume_shares':10},'VOLUME_LIMIT'), ({'halted':True},'HALTED'),
    ({'ask_micros':100_200_000},'LIMIT_NOT_CROSSED'),
    ({'available_at':'future'},'QUOTE_UNAVAILABLE'),
    ({'quoted_at':'stale'},'QUOTE_STALE')])
def test_uncertain_fill_conditions(change,reason):
    f,e,request,decision,account,caps,policy,market=inputs()
    change={k:(request.as_of+timedelta(seconds=1) if v=='future' else request.as_of-timedelta(seconds=31) if v=='stale' else v) for k,v in change.items()}
    market=replace(market,quotes=(replace(market.quotes[0],**change),)+market.quotes[1:])
    receipt=e.ShadowEngine(request=request,capabilities=caps,policy=policy,journal=MemoryJournal()).step(decision,market,account)
    first=receipt.fills[0]
    assert reason in first.uncertainty and first.quantity<15
    assert first.status in {'PARTIAL','UNFILLED'}

def test_slippage_respects_limit_instead_of_inventing_fill():
    f,e,r,d,a,c,p,m=inputs()
    result=e.ShadowEngine(request=r,capabilities=c,policy=replace(p,slippage_bps=1),journal=MemoryJournal()).step(d,m,a)
    assert all(x.status=='UNFILLED' and 'LIMIT_NOT_CROSSED' in x.uncertainty for x in result.fills)

def test_missing_quote_is_explicit_unfilled():
    f,e,r,d,a,c,p,m=inputs()
    result=e.ShadowEngine(request=r,capabilities=c,policy=p,journal=MemoryJournal()).step(d,replace(m,quotes=m.quotes[1:]),a)
    assert result.fills[0].status=='UNFILLED' and 'QUOTE_MISSING' in result.fills[0].uncertainty

def test_closed_session_rejected_before_reservation():
    f,e,r,d,a,c,p,m=inputs(); journal=MemoryJournal()
    with pytest.raises(ValueError,match='session'):
        e.ShadowEngine(request=r,capabilities=c,policy=p,journal=journal).step(d,replace(m,as_of=r.session_close),a)
    assert not journal.rows

def test_invalid_quotes_and_boolean_policy_rejected():
    f,e,r,d,a,c,p,m=inputs()
    with pytest.raises(ValueError,match='crossed'):
        replace(m.quotes[0],bid_micros=101_000_000)
    with pytest.raises(TypeError): replace(p,fee_bps=True)
    with pytest.raises(ValueError,match='duplicate'): replace(m,quotes=(m.quotes[0],m.quotes[0]))

def test_crash_after_reservation_resumes_same_fill():
    f,e,r,d,a,c,p,m=inputs()
    class CrashJournal(MemoryJournal):
        fail=True
        def complete(self,*args,**kwargs):
            if self.fail: self.fail=False; raise RuntimeError('crash')
            return super().complete(*args,**kwargs)
    j=CrashJournal(); runner=e.ShadowEngine(request=r,capabilities=c,policy=p,journal=j)
    with pytest.raises(RuntimeError,match='crash'): runner.step(d,m,a)
    assert len(j.rows)==1
    receipt=runner.step(d,m,a)
    assert receipt==runner.step(d,m,a) and j.completions==5

def test_sell_fill_uses_bid_with_adverse_slippage_and_fee_rounding():
    from src.axiom2.portfolio.intents import build_order_intents
    f,e,r,d,a,c,p,m=inputs()
    intent=build_order_intents(d,r,a,c,mode='SHADOW')[0]
    intent=replace(intent,side='SELL',limit_price_micros=99_000_000)
    fill=f.hypothetical_fill(intent,m.quotes[0],m,replace(p,slippage_bps=10))
    assert fill.price_micros==99_900_000 and fill.quantity==15 and fill.fee_cents==150
    assert not fill.execution_enabled and not fill.capital_authorized

def test_direct_fill_validates_mutated_quote_and_expired_intent():
    from src.axiom2.portfolio.intents import build_order_intents
    f,e,r,d,a,c,p,m=inputs()
    intent=build_order_intents(d,r,a,c,mode='SHADOW')[0]
    fill=f.hypothetical_fill(replace(intent,expires_at=r.as_of),m.quotes[0],m,p)
    assert fill.status=='UNFILLED' and 'INTENT_EXPIRED' in fill.uncertainty
    object.__setattr__(m.quotes[0],'bid_micros',101_000_000)
    with pytest.raises(ValueError,match='crossed'): f.hypothetical_fill(intent,m.quotes[0],m,p)

def test_changed_market_replay_does_not_append_terminal_fills():
    f,e,r,d,a,c,p,m=inputs(); j=MemoryJournal()
    runner=e.ShadowEngine(request=r,capabilities=c,policy=p,journal=j)
    runner.step(d,m,a)
    with pytest.raises(ValueError,match='replay'):
        runner.step(d,replace(m,source_digest='b'*64),a)
    assert j.completions==5

def test_journal_terminal_result_tampering_is_denied():
    f,e,r,d,a,c,p,m=inputs(); j=MemoryJournal()
    runner=e.ShadowEngine(request=r,capabilities=c,policy=p,journal=j)
    runner.step(d,m,a)
    next(iter(j.rows.values())).result['capital_authorized']=True
    with pytest.raises(ValueError,match='replay'): runner.step(d,m,a)

def test_process_shadow_step_has_no_network_process_or_broker_submit_side_effect():
    import os, subprocess, sys
    script='''
import sys, runpy
from pathlib import Path
scope=runpy.run_path('tests/axiom2/shadow/test_shadow_engine.py')
f,e,r,d,a,c,p,m=scope['inputs']()
j=scope['MemoryJournal']()
for name in ('src.brokers','src.broker','src.scanner'):
    assert not any(module==name or module.startswith(name+'.') for module in sys.modules)
calls=[]
def audit(event,args):
    if event.startswith('socket.') or event.startswith('subprocess.') or event in {'os.system','os.posix_spawn'}:
        calls.append(event)
        raise RuntimeError('forbidden side effect '+event)
sys.addaudithook(audit)
receipt=e.ShadowEngine(request=r,capabilities=c,policy=p,journal=j).step(d,m,a)
assert len(receipt.fills)==5 and not receipt.execution_enabled and not receipt.capital_authorized
assert not calls
print('ZERO_SIDE_EFFECT_SHADOW_STEPS=1 HYPOTHETICAL_FILLS=5')
'''
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    result=subprocess.run([sys.executable,'-W','error','-c',script],cwd=Path(__file__).resolve().parents[3],env=env,
        capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert 'ZERO_SIDE_EFFECT_SHADOW_STEPS=1 HYPOTHETICAL_FILLS=5' in result.stdout

def test_nonzero_fees_never_exceed_aggregate_settled_cash_budget():
    from src.axiom2.portfolio.intents import build_order_intents
    f,e,r,d,a,c,p,m=inputs()
    intents=build_order_intents(d,r,a,c,mode='SHADOW')
    principal=sum((i.quantity*i.limit_price_micros+9999)//10000 for i in intents)
    a=replace(a,settled_cash_cents=principal)
    j=MemoryJournal(); runner=e.ShadowEngine(request=r,capabilities=c,policy=p,journal=j)
    with pytest.raises(ValueError,match='cash'):
        runner.step(d,m,a)
    assert not j.rows and j.completions==0

def test_direct_fill_refuses_reflected_intent_authority_flags():
    from src.axiom2.portfolio.intents import build_order_intents
    f,e,r,d,a,c,p,m=inputs()
    intent=build_order_intents(d,r,a,c,mode='SHADOW')[0]
    object.__setattr__(intent,'execution_enabled',True)
    with pytest.raises(ValueError,match='authority'): f.hypothetical_fill(intent,m.quotes[0],m,p)

def test_revalidated_fill_and_receipt_refuse_reflected_authority_flags():
    f,e,r,d,a,c,p,m=inputs()
    receipt=e.ShadowEngine(request=r,capabilities=c,policy=p,journal=MemoryJournal()).step(d,m,a)
    object.__setattr__(receipt.fills[0],'capital_authorized',True)
    with pytest.raises(ValueError,match='authority'): receipt.fills[0].__post_init__()
    with pytest.raises(ValueError,match='authority'): receipt.__post_init__()

def test_quote_freshness_uses_utc_elapsed_time_across_dst_fold():
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from src.axiom2.portfolio.intents import build_order_intents
    f,e,r,d,a,c,p,m=inputs()
    zone=ZoneInfo('America/New_York')
    quoted=datetime(2026,11,1,1,10,tzinfo=zone,fold=0)
    now=datetime(2026,11,1,1,10,tzinfo=zone,fold=1)
    quote=replace(m.quotes[0],quoted_at=quoted,available_at=now)
    market=replace(m,as_of=now,session_open=datetime(2026,11,1,6,tzinfo=timezone.utc),
        session_close=datetime(2026,11,1,7,tzinfo=timezone.utc),quotes=(quote,))
    intent=replace(build_order_intents(d,r,a,c,mode='SHADOW')[0],expires_at=market.session_close)
    fill=f.hypothetical_fill(intent,quote,market,p)
    assert fill.status=='UNFILLED' and 'QUOTE_STALE' in fill.uncertainty

def test_excessive_fee_policy_fails_cost_bound_before_any_reservation():
    f,e,r,d,a,c,p,m=inputs(); j=MemoryJournal()
    with pytest.raises(ValueError,match='cost|cash'):
        e.ShadowEngine(request=r,capabilities=c,policy=replace(p,fee_bps=1000),journal=j).step(d,m,a)
    assert not j.rows

def test_midpoint_spread_cost_and_fees_share_frozen_cost_bound():
    f,e,r,d,a,c,p,m=inputs(); j=MemoryJournal()
    # Fees alone fit the 1600-cent risk cost bound; spread plus fees do not.
    with pytest.raises(ValueError,match='cost'):
        e.ShadowEngine(request=r,capabilities=c,policy=replace(p,fee_bps=20),journal=j).step(d,m,a)
    assert not j.rows
