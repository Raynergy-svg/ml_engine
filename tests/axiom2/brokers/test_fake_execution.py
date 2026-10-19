"""Persistent fake contract history; no provider bindings."""
from datetime import timedelta
import multiprocessing
import pytest
from tests.axiom2.shadow.test_shadow_journal import make_journal,register,NOW
from src.axiom2.brokers.fake_execution import FakeRobinhoodTransport,FakeProviderEvent,STREAM
from src.evidence.store import StoreCorruptionError

REQUEST=dict(account_number='FAKE:dedicated',symbol='FIXTURE',side='buy',type='limit',quantity='2',limit_price='1.00',time_in_force='gfd',market_hours='regular_hours')
PLACE={**REQUEST,'ref_id':'ba235fdd-1c08-4d54-b018-faf1cf035a01'}


def test_event_signed_before_registration_is_rejected(tmp_path):
    journal,signer,operator=make_journal(tmp_path);register(journal,signer,operator)
    response=FakeRobinhoodTransport._apply('REVIEW',REQUEST,{},set(),NOW)
    earlier=NOW-timedelta(seconds=1)
    event=FakeProviderEvent(sequence=0,previous_digest=None,actor_id='shadow',occurred_at=earlier,action='REVIEW',request=REQUEST,response=response)
    with journal.store._locked():journal._write(STREAM,signer.sign(event,created_at=earlier),NOW,0)
    with pytest.raises(StoreCorruptionError):FakeRobinhoodTransport(journal).orders()


def test_changed_ref_request_and_unsupported_profiles_are_rejected(tmp_path):
    journal,signer,operator=make_journal(tmp_path);register(journal,signer,operator);fake=FakeRobinhoodTransport(journal)
    fake._commit('REVIEW',REQUEST);first=fake._commit('PLACE',PLACE)
    assert fake._commit('PLACE',PLACE)==first
    with pytest.raises(ValueError,match='CHANGED_BYTES'):fake._commit('PLACE',{**PLACE,'quantity':'3'})
    for change in ({'type':'market'},{'quantity':'0.5'},{'account_number':'real'},{'stop_price':'1'},{'market_hours':'all_day'}):
        with pytest.raises(ValueError):fake._commit('REVIEW',{**REQUEST,**change})


def _place_worker(root,private,operator_private,queue):
    journal,_,_=make_journal(root,private,operator_private)
    try:queue.put(('ok',FakeRobinhoodTransport(journal)._commit('PLACE',PLACE).payload_digest))
    except Exception as exc:queue.put(('error',type(exc).__name__))


def test_four_process_same_ref_has_one_durable_fake_placement(tmp_path):
    journal,signer,operator=make_journal(tmp_path);register(journal,signer,operator);fake=FakeRobinhoodTransport(journal)
    fake._commit('REVIEW',REQUEST)
    context=multiprocessing.get_context('spawn');queue=context.Queue()
    workers=[context.Process(target=_place_worker,args=(tmp_path,signer.private_bytes(),operator.private_bytes(),queue)) for _ in range(4)]
    for worker in workers:worker.start()
    results=[queue.get(timeout=30) for _ in workers]
    for worker in workers:worker.join(timeout=30);assert worker.exitcode==0
    assert all(result[0]=='ok' for result in results) and len({result[1] for result in results})==1
    with journal.store._locked():rows,*_=fake._read()
    assert sum(envelope.payload['action']=='PLACE' for envelope,_ in rows)==1


def test_retirement_keeps_history_but_denies_new_fake_effect(tmp_path):
    from src.evidence.execution_shadow import AuthorityBindingEvent
    journal,signer,operator=make_journal(tmp_path);head=register(journal,signer,operator);fake=FakeRobinhoodTransport(journal)
    fake._commit('REVIEW',REQUEST);fake._commit('PLACE',PLACE)
    later=NOW+timedelta(seconds=1);journal.store._trusted_clock=lambda:later
    event=AuthorityBindingEvent(sequence=1,previous_digest=head,action='REVOKE',actor_id='shadow',key_id=signer.key_id,occurred_at=later)
    journal.configure(operator.sign(event,created_at=later),expected_head=head)
    order=fake.orders()[0]
    with pytest.raises(ValueError,match='RETIRED'):fake.fill(order['id'],quantity=1,price_cents=100,fee_cents=0)
