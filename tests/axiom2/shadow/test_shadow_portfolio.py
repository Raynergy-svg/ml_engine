"""Durable synthetic portfolio evolution; no calendar/live qualification."""
from dataclasses import replace
from datetime import timedelta
import pytest
from tests.axiom2.shadow.test_shadow_engine import inputs
from tests.axiom2.shadow.test_shadow_journal import make_journal
from src.evidence.execution_shadow import AuthorityBindingEvent
from src.axiom2.portfolio.authority import evaluate_portfolio
from src.axiom2.execution.resolution import reconstruct_risk_packet
from tests.axiom2.shadow.test_execution_resolution import packet


def setup(tmp_path):
    from src.axiom2.shadow.portfolio import ShadowPortfolioLedger
    f,e,request,_,account,caps,fill_policy,market=inputs()
    policy,_,_=reconstruct_risk_packet(packet())
    journal,signer,operator=make_journal(tmp_path);journal.store._trusted_clock=lambda:request.as_of
    event=AuthorityBindingEvent(sequence=0,previous_digest=None,action='REGISTER',actor_id='shadow',key_id=signer.key_id,occurred_at=request.as_of)
    journal.configure(operator.sign(event,created_at=request.as_of))
    ledger=ShadowPortfolioLedger(journal);ledger.initialize(request,account,risk_policy=policy,fill_policy=fill_policy)
    snapshot,account=ledger.observations();request=replace(request,snapshot=snapshot)
    return f,e,journal,ledger,request,account,caps,policy,fill_policy,market


def test_buy_restart_hold_maturity_sell_settlement_and_next_rebalance(tmp_path):
    from src.axiom2.shadow.portfolio import ShadowPortfolioLedger
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    decision=evaluate_portfolio(request,policy)
    engine=e.ShadowEngine(request=request,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy)
    receipt=engine.step(decision,market,account)
    assert engine.step(decision,market,account)==receipt
    ledger=ShadowPortfolioLedger(journal)
    snapshot,observed=ledger.observations();assert len(observed.positions)==5
    assert snapshot.cash_cents==request.snapshot.cash_cents-sum((x.quantity*x.price_micros+9999)//10000+x.fee_cents for x in receipt.fills)
    assert snapshot.cohort_entry_session==request.session_index
    for offset in range(1,11):
        now=request.as_of+timedelta(days=offset);journal.store._trusted_clock=lambda:now
        updated_market=replace(market,as_of=now,session_open=market.session_open+timedelta(days=offset),session_close=market.session_close+timedelta(days=offset),quotes=tuple(replace(q,quoted_at=now,available_at=now) for q in market.quotes))
        ledger.observe(updated_market,session_index=request.session_index+offset)
        snapshot,observed=ledger.observations()
        updated=replace(request,as_of=now,signal_at=now,session_open=updated_market.session_open,session_close=updated_market.session_close,session_index=request.session_index+offset,snapshot=snapshot,candidates=tuple(replace(o,quoted_at=now,metadata_at=now) for o in request.candidates))
        decision=evaluate_portfolio(updated,policy)
        if offset==1:assert decision.status=='HOLD';continue
        if offset==5:
            assert decision.status=='EXIT_REQUIRED'
            e.ShadowEngine(request=updated,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy).step(decision,updated_market,observed)
            sold,_=ledger.observations();assert sold.cash_cents>sold.settled_cash_cents and not sold.positions
        if offset==6:assert snapshot.cash_cents==snapshot.settled_cash_cents
        if offset==10:
            assert decision.status=='READY_SHADOW'
            e.ShadowEngine(request=updated,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy).step(decision,updated_market,observed)
    assert ledger.qualified is False


def test_stale_account_and_changed_replay_rejected(tmp_path):
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    decision=evaluate_portfolio(request,policy)
    engine=e.ShadowEngine(request=request,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy)
    engine.step(decision,market,account)
    changed=replace(request,decision_id='another-decision')
    with pytest.raises(ValueError):e.ShadowEngine(request=changed,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy).step(evaluate_portfolio(changed,policy),market,account)
    with pytest.raises(ValueError):ledger.observe(market,session_index=request.session_index-1)


def test_crash_mid_batch_requires_recovery_and_does_not_double_debit(tmp_path,monkeypatch):
    from src.axiom2.shadow.portfolio import ShadowPortfolioLedger
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    engine=e.ShadowEngine(request=request,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy)
    decision=evaluate_portfolio(request,policy);original=journal.complete;calls=[]
    def crash(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('simulated crash')
        return original(*args,**kwargs)
    monkeypatch.setattr(journal,'complete',crash)
    with pytest.raises(RuntimeError):engine.step(decision,market,account)
    with pytest.raises(ValueError):ledger.observations()
    with pytest.raises(ValueError):ledger.observe(market,session_index=request.session_index)
    monkeypatch.setattr(journal,'complete',original)
    receipt=engine.step(decision,market,account);snapshot,_=ShadowPortfolioLedger(journal).observations()
    assert snapshot.cash_cents==request.snapshot.cash_cents-sum((fill.quantity*fill.price_micros+9999)//10000+fill.fee_cents for fill in receipt.fills)
    assert engine.step(decision,market,account)==receipt


def test_wrong_risk_policy_and_future_observation_rejected_before_reservations(tmp_path):
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    decision=evaluate_portfolio(request,policy)
    with pytest.raises(ValueError):e.ShadowEngine(request=request,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=replace(policy,cash_reserve_bps=policy.cash_reserve_bps+1)).step(decision,market,account)
    assert not journal.replay()
    future=replace(market,as_of=market.as_of+timedelta(seconds=1))
    with pytest.raises(ValueError):ledger.observe(future,session_index=request.session_index)


@pytest.mark.parametrize('change',['same-time','jump','calendar','late-clock','regressed-clock'])
def test_observation_cannot_fabricate_settlement_sessions_or_poison_clock(tmp_path,change):
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    old_files=list((tmp_path/'shadow-portfolio').glob('*.json'))
    changed=market;session=request.session_index
    if change=='same-time':session+=1
    if change=='jump':
        session+=2;changed=replace(market,as_of=market.as_of+timedelta(days=2),session_open=market.session_open+timedelta(days=2),session_close=market.session_close+timedelta(days=2));journal.store._trusted_clock=lambda:changed.as_of
    if change=='calendar':changed=replace(market,calendar_digest='f'*64)
    if change=='late-clock':
        clocks=iter((request.as_of,request.as_of+timedelta(seconds=31)));journal.store._trusted_clock=lambda:next(clocks)
    if change=='regressed-clock':journal.store._trusted_clock=lambda:request.as_of-timedelta(seconds=1)
    with pytest.raises(ValueError):ledger.observe(changed,session_index=session)
    assert list((tmp_path/'shadow-portfolio').glob('*.json'))==old_files


def test_revocation_cannot_precede_committed_portfolio_receipt(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    from src.evidence.signing import Ed25519Signer
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    operator=Ed25519Signer.generate();journal.operator_trust_store.add(operator.trusted_key(valid_from=request.as_of-timedelta(days=1)))
    later=request.as_of+timedelta(seconds=2);journal.store._trusted_clock=lambda:later
    ledger.observe(replace(market,as_of=later),session_index=request.session_index)
    revoke_time=request.as_of+timedelta(seconds=1);journal.store._trusted_clock=lambda:revoke_time
    previous=journal._roles()[0][-1][0].payload_digest
    event=AuthorityBindingEvent(sequence=1,previous_digest=previous,action='REVOKE',actor_id=journal.actor_id,key_id=journal.signer.key_id,occurred_at=revoke_time)
    with pytest.raises(ValueError):journal.configure(operator.sign(event,created_at=revoke_time),expected_head=previous)
    assert ledger.observations()[0].observed_at==later


def test_continuous_engine_rejects_hidden_portfolio_callback(tmp_path):
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    class Hidden:
        def validate_step(self,*args,**kwargs):raise AssertionError('hidden callback invoked')
    with pytest.raises(TypeError):e.ShadowEngine(request=request,capabilities=caps,policy=fill_policy,journal=journal,portfolio=Hidden(),risk_policy=policy)


def test_portfolio_crash_after_commit_and_tamper(tmp_path,monkeypatch):
    from src.axiom2.shadow.portfolio import ShadowPortfolioLedger
    from src.evidence.store import StoreCorruptionError
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    engine=e.ShadowEngine(request=request,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy)
    original=journal._write
    def crash(name,*args):
        result=original(name,*args)
        if name=='shadow-portfolio':raise RuntimeError('after commit')
        return result
    monkeypatch.setattr(journal,'_write',crash)
    decision=evaluate_portfolio(request,policy)
    with pytest.raises(RuntimeError):engine.step(decision,market,account)
    monkeypatch.setattr(journal,'_write',original)
    before=ShadowPortfolioLedger(journal).observations();engine.step(decision,market,account)
    assert ledger.observations()==before
    target=sorted((tmp_path/'shadow-portfolio').glob('*.json'))[-1];target.write_bytes(target.read_bytes().replace(b'STEP',b'STEQ'))
    with pytest.raises(StoreCorruptionError):ledger.observations()


def test_step_receipt_cannot_precede_completed_execution(tmp_path,monkeypatch):
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    later=request.as_of+timedelta(seconds=10);journal.store._trusted_clock=lambda:later
    engine=e.ShadowEngine(request=request,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy)
    decision=evaluate_portfolio(request,policy);original=ledger.apply_step
    def regress(*args,**kwargs):
        journal.store._trusted_clock=lambda:request.as_of+timedelta(seconds=5)
        return original(*args,**kwargs)
    monkeypatch.setattr(ledger,'apply_step',regress)
    with pytest.raises(ValueError):engine.step(decision,market,account)
    assert len(list((tmp_path/'shadow-portfolio').glob('*.json')))==1


def test_step_actual_market_quote_must_be_fresh_at_final_receipt(tmp_path,monkeypatch):
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path)
    changed=replace(market,quotes=tuple(replace(q,quoted_at=request.as_of-timedelta(seconds=29),available_at=request.as_of-timedelta(seconds=29)) for q in market.quotes))
    engine=e.ShadowEngine(request=request,capabilities=caps,policy=fill_policy,journal=journal,portfolio=ledger,risk_policy=policy)
    original=ledger.apply_step
    def delay(*args,**kwargs):
        journal.store._trusted_clock=lambda:request.as_of+timedelta(seconds=2)
        return original(*args,**kwargs)
    monkeypatch.setattr(ledger,'apply_step',delay)
    with pytest.raises(ValueError):engine.step(evaluate_portfolio(request,policy),changed,account)
    assert len(list((tmp_path/'shadow-portfolio').glob('*.json')))==1


def test_frozen_long_quote_ttl_is_used_consistently_in_replay(tmp_path):
    f,e,journal,ledger,request,account,caps,policy,fill_policy,market=setup(tmp_path/'unused')
    from src.axiom2.shadow.portfolio import ShadowPortfolioLedger
    from src.evidence.execution_shadow import AuthorityBindingEvent
    fresh_journal,signer,operator=make_journal(tmp_path/'long');fresh_journal.store._trusted_clock=lambda:request.as_of
    event=AuthorityBindingEvent(sequence=0,previous_digest=None,action='REGISTER',actor_id='shadow',key_id=signer.key_id,occurred_at=request.as_of)
    fresh_journal.configure(operator.sign(event,created_at=request.as_of))
    policy=replace(policy,quote_ttl_seconds=60)
    ledger=ShadowPortfolioLedger(fresh_journal)
    ledger.initialize(request,account,risk_policy=policy,fill_policy=fill_policy)
    old=replace(market,quotes=tuple(replace(q,quoted_at=request.as_of-timedelta(seconds=45),available_at=request.as_of-timedelta(seconds=45)) for q in market.quotes))
    ledger.observe(old,session_index=request.session_index)
    assert ShadowPortfolioLedger(fresh_journal).observations()[0].observed_at==request.as_of


def portfolio_initializer(root,private,operator_private,request,account,policy,fill_policy,queue):
    from src.axiom2.shadow.portfolio import ShadowPortfolioLedger
    journal,_,_=make_journal(root,private,operator_private);journal.store._trusted_clock=lambda:request.as_of
    try:
        ShadowPortfolioLedger(journal).initialize(request,account,risk_policy=policy,fill_policy=fill_policy)
        queue.put('committed')
    except ValueError:queue.put('denied')


def test_real_multiprocess_single_baseline_writer(tmp_path):
    import multiprocessing
    f,e,request,_,account,caps,fill_policy,market=inputs();policy,_,_=reconstruct_risk_packet(packet())
    journal,signer,operator=make_journal(tmp_path);journal.store._trusted_clock=lambda:request.as_of
    event=AuthorityBindingEvent(sequence=0,previous_digest=None,action='REGISTER',actor_id='shadow',key_id=signer.key_id,occurred_at=request.as_of)
    journal.configure(operator.sign(event,created_at=request.as_of))
    context=multiprocessing.get_context('spawn');queue=context.Queue()
    processes=[context.Process(target=portfolio_initializer,args=(tmp_path,signer.private_bytes(),operator.private_bytes(),request,account,policy,fill_policy,queue)) for _ in range(4)]
    for process in processes:process.start()
    outcomes=[queue.get(timeout=15) for _ in processes]
    for process in processes:process.join(15);assert process.exitcode==0
    assert outcomes.count('committed')==1 and outcomes.count('denied')==3
    from src.axiom2.shadow.portfolio import ShadowPortfolioLedger
    assert ShadowPortfolioLedger(journal).observations()[0].cash_cents==request.snapshot.cash_cents
