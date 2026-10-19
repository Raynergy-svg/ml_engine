from dataclasses import replace
import pytest
from src.axiom2.execution.reconciliation import (Fill, Ledger, Observations, Account, Policy, Acceptance, reconcile)

def data():
    f = Fill('e1', 'o1', 'AAA', 2, 100, 3)
    return Ledger('a', 'observed', 1000, (), ('o1',), (f,), 1000, 0), Observations('a', 'observed', True, (f,)), Account('a', 797, 797, (('AAA', 2),), True, 1000, 0, valuation_marks_micros=(('AAA',1015000),)), Policy('a'*64, 0, 0)

def test_match_and_native_dedup():
    l,o,a,p=data()
    assert reconcile(l,replace(o,fills=o.fills*2),a,p).state == 'MATCHED'

def test_incomplete_not_zero():
    l,o,a,p=data()
    assert reconcile(l,replace(o,complete=False),a,p).state == 'PENDING'

@pytest.mark.parametrize('change', ['account','mode','unknown','fee','cash','zero','mutation','correction','cashflow'])
def test_fail_closed(change):
    l,o,a,p=data()
    if change=='account': a=replace(a,account='b')
    if change=='mode': o=replace(o,mode='hypothetical')
    if change=='unknown': o=replace(o,fills=(replace(o.fills[0],order_id='manual'),))
    if change=='fee': o=replace(o,fills=(replace(o.fills[0],fee_cents=4),))
    if change=='cash': a=replace(a,total_cash_cents=798)
    if change=='zero': a=replace(a,positions=(('AAA',2),('DELISTED',0)))
    if change=='mutation': object.__setattr__(o.fills[0], 'quantity', True)
    if change=='correction': o=replace(o,fills=(replace(o.fills[0],correction_of='e0'),))
    if change=='cashflow': a=replace(a,external_cashflow_cents=50)
    assert reconcile(l,o,a,p).state=='HALTED'

def test_missing_fill_and_unsettled_pending():
    l,o,a,p=data()
    assert reconcile(l,replace(o,fills=()),a,p).state=='PENDING'
    assert reconcile(l,o,replace(a,settled_cash_cents=796),p).state=='PENDING'

def test_acceptance_calendar_and_cycles():
    r=Acceptance('a'*64,0,0)
    assert not r.qualified(tuple(range(20)), ((0,1,2,3,4),(5,6,7,8,9),(10,11,12,13,14),(15,16,17,18,19)), synthetic=True)
    assert not r.qualified(tuple(range(19)), (), synthetic=False)

@pytest.mark.parametrize('mutation',['timeout','action','unresolved','conflict','mutable','boolcash','lostid','negative','closedholding'])
def test_adversarial_review_inputs(mutation):
    l,o,a,p=data()
    expected='HALTED'
    if mutation=='timeout': o=replace(o,timed_out=True); expected='PENDING'
    if mutation=='action': o=replace(o,corporate_actions=('split',))
    if mutation=='unresolved': o=replace(o,unresolved_orders=('o1',)); expected='PENDING'
    if mutation=='conflict': o=replace(o,fills=o.fills+(replace(o.fills[0],fee_cents=9),))
    if mutation=='mutable': o=replace(o,fills=list(o.fills))
    if mutation=='boolcash': a=replace(a,total_cash_cents=True)
    if mutation=='lostid': o=replace(o,fills=(replace(o.fills[0],event_id=''),))
    if mutation=='negative': l=replace(l,initial_cash_cents=100)
    if mutation=='closedholding': l=replace(l,initial_positions=(('DELISTED',0),))
    assert reconcile(l,o,a,p).state==expected

def test_out_of_order_partial_and_late_fill_after_cancel():
    l,o,a,p=data()
    fills=(Fill('e2','o1','AAA',1,100,0),Fill('e1','o1','AAA',2,100,3))
    l=replace(l,expected_fills=fills)
    o=replace(o,fills=tuple(reversed(fills)))
    a=replace(a,total_cash_cents=697,settled_cash_cents=697,positions=(('AAA',3),),equity_cents=1001)
    d=reconcile(l,o,a,p)
    assert d.state=='MATCHED' and d.costs.turnover_cents==300 and d.costs.fees_cents==3
    # Cancellation is no fill suppression: a late unjournaled event halts.
    assert reconcile(replace(l,expected_fills=fills[:1]),o,a,p).state=='HALTED'

@pytest.mark.parametrize('flow',[50,-50])
def test_signed_cashflows_do_not_raise_performance_peak(flow):
    l,o,a,p=data()
    l=replace(l,external_cashflow_cents=flow)
    a=replace(a,total_cash_cents=797+flow,settled_cash_cents=797+flow,external_cashflow_cents=flow,equity_cents=1000+flow)
    assert reconcile(l,o,a,p).flow_adjusted_peak_cents==1000

def test_preserves_zero_holdings_and_rechecks_policy():
    l,o,a,p=data()
    l=replace(l,initial_positions=(('DELISTED',0),))
    a=replace(a,positions=(('DELISTED',0),('AAA',2)),valuation_marks_micros=(('DELISTED',0),('AAA',1015000)))
    assert reconcile(l,o,a,p).state=='MATCHED'
    object.__setattr__(p,'cash_tolerance_cents',True)
    assert reconcile(l,o,a,p).state=='HALTED'

def test_shadow_acceptance_never_qualifies_unattested_calendar():
    r=Acceptance('a'*64,0,0)
    assert not r.qualified(tuple(range(20)),tuple(tuple(range(i,i+5)) for i in (0,5,10,15)),synthetic=False,audited=True)
    with pytest.raises(ValueError): Acceptance('bad',0,0)


def test_observed_account_cannot_match_hypothetical_ledger():
    l,o,a,p=data()
    assert reconcile(replace(l,mode='hypothetical'),replace(o,mode='hypothetical'),a,p).state=='HALTED'


@pytest.mark.parametrize('equity',[0,1001,10**12])
def test_equity_must_reconcile_to_cash_and_exact_position_marks(equity):
    l,o,a,p=data()
    assert reconcile(l,o,replace(a,equity_cents=equity),p).state=='HALTED'


def test_authoritative_correction_chain_replaces_original_once():
    l,o,a,p=data()
    correction=Fill('e2','o1','AAA',1,100,2,correction_of='e1')
    fills=(o.fills[0],correction)
    l=replace(l,expected_fills=fills)
    o=replace(o,fills=(correction,fills[0],correction))
    a=replace(a,total_cash_cents=898,settled_cash_cents=898,positions=(('AAA',1),),equity_cents=999)
    d=reconcile(l,o,a,p)
    assert d.state=='MATCHED' and d.costs.turnover_cents==100 and d.costs.fees_cents==2


def test_explicit_split_and_dividend_reconcile_without_creating_buying_power():
    from src.axiom2.execution.reconciliation import CorporateAction
    l,o,a,p=data()
    l=replace(l,initial_positions=(('OLD',2),))
    split=CorporateAction('split-one','OLD','SPLIT',2,1,0)
    dividend=CorporateAction('div-one','OLD','DIVIDEND',1,1,50)
    o=replace(o,corporate_actions=(split,dividend))
    l=replace(l,expected_corporate_actions=(split,dividend))
    a=replace(a,total_cash_cents=847,settled_cash_cents=847,positions=(('OLD',4),('AAA',2)),
              valuation_marks_micros=(('OLD',500000),('AAA',1015000)),equity_cents=1250)
    d=reconcile(l,o,a,p)
    assert d.state=='MATCHED'
    assert reconcile(l,o,replace(a,settled_cash_cents=797),p).state=='PENDING'


def test_dangling_correction_ancestor_halts_instead_of_crashing():
    l,o,a,p=data()
    chain=(Fill('child','o1','AAA',1,100,0,correction_of='parent'),
           Fill('parent','o1','AAA',1,100,0,correction_of='missing'))
    assert reconcile(l,replace(o,fills=chain),a,p).state=='HALTED'


def test_superseded_history_conflict_cannot_disappear_in_normalization():
    l,o,a,p=data()
    original=o.fills[0];correction=Fill('e2','o1','AAA',1,100,2,correction_of='e1')
    l=replace(l,expected_fills=(original,correction))
    o=replace(o,fills=(replace(original,quantity=200),correction))
    a=replace(a,total_cash_cents=898,settled_cash_cents=898,positions=(('AAA',1),),equity_cents=999)
    assert reconcile(l,o,a,p).state=='HALTED'
