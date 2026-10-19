"""No runtime auth or broker calls; synthetic services only."""
import pytest
from tests.axiom2.brokers.test_robinhood_readonly import fixture,run
from tests.axiom2.shadow.test_execution_artifacts import resolved_ingress


def test_read_host_composes_only_fixed_reads_and_preserves_unknowns():
    from src.axiom2.brokers.service import TrustedReadHost
    adapter,payloads,calls,tools=fixture();host=TrustedReadHost(adapter)
    snapshot=run(host.account_snapshot('dedicated-fixture'))
    assert snapshot.settled_cash_cents is None and snapshot.provider_observed_at is None
    assert not host.capabilities().execution_enabled
    assert not any(hasattr(host,name) for name in ('submit','review','cancel','call_tool','connect','register_oauth'))


def test_fake_service_full_path_is_hypothetical_and_reconciles_from_signed_journals(tmp_path):
    from src.axiom2.execution.service import OfflineExecutionService
    from src.axiom2.brokers.fake_execution import FakeRobinhoodTransport
    kernel,gateway,observer,envelope,args,request=resolved_ingress(tmp_path)
    fake=FakeRobinhoodTransport(args['fence'].journal)
    service=OfflineExecutionService(kernel,fake,gateway_signer=gateway,observer_signer=observer,
        admission=args,instrument_symbols={args['intent'].instrument_id:'FIXTURE'})
    service.admit(envelope);review=service.review(args['intent'].intent_id)
    assert review.event.state=='REVIEWED' and not review.capital_authorized
    acknowledged=service.submit_fake(args['intent'].intent_id)
    assert acknowledged.event.state=='ACKNOWLEDGED' and acknowledged.event.filled_quantity==0
    order=fake.orders()[0];fake.fill(order['id'],quantity=args['intent'].quantity,price_cents=args['intent'].limit_price_micros//10000,fee_cents=0)
    service.observe(args['intent'].intent_id)
    decision=service.reconcile_hypothetical()
    assert decision.state=='MATCHED' and decision.costs.mode=='hypothetical'
    assert not decision.capital_authorized and not decision.execution_enabled
    assert [receipt.event.state for receipt in kernel.replay()]==['PROPOSED','REVIEWED','SUBMISSION_RESERVED','ACKNOWLEDGED','FILLED']
    from src.axiom2.execution.authority import ExecutionAuthority
    assert ExecutionAuthority().submit(None,None).status=='BLOCKED'


def service_fixture(tmp_path):
    from src.axiom2.execution.service import OfflineExecutionService
    from src.axiom2.brokers.fake_execution import FakeRobinhoodTransport
    kernel,gateway,observer,envelope,args,request=resolved_ingress(tmp_path)
    fake=FakeRobinhoodTransport(args['fence'].journal)
    def make(transport):return OfflineExecutionService(kernel,transport,gateway_signer=gateway,observer_signer=observer,admission=args,instrument_symbols={args['intent'].instrument_id:'FIXTURE'})
    service=make(fake);service.admit(envelope)
    return service,fake,kernel,args,make


def test_ambiguous_response_restart_reconciles_without_second_placement(tmp_path,monkeypatch):
    from src.axiom2.brokers.fake_execution import FakeRobinhoodTransport
    service,fake,kernel,args,make=service_fixture(tmp_path);service.review(args['intent'].intent_id)
    original=FakeRobinhoodTransport._commit
    def lost_ack(self,action,request,**kwargs):
        result=original(self,action,request,**kwargs)
        if action=='PLACE':raise TimeoutError('simulated lost acknowledgement')
        return result
    monkeypatch.setattr(FakeRobinhoodTransport,'_commit',lost_ack)
    with pytest.raises(TimeoutError):service.submit_fake(args['intent'].intent_id)
    assert kernel.replay()[-1].event.state=='SUBMISSION_UNKNOWN'
    monkeypatch.setattr(FakeRobinhoodTransport,'_commit',original)
    restarted=make(FakeRobinhoodTransport(fake.journal));assert restarted.submit_fake(args['intent'].intent_id).event.state=='ACKNOWLEDGED'
    with kernel.store._locked():rows,*_=fake._read()
    assert sum(envelope.payload['action']=='PLACE' for envelope,_ in rows)==1


def test_crash_after_reservation_does_not_blindly_place_on_restart(tmp_path):
    from src.axiom2.brokers.fake_execution import FakeRobinhoodTransport
    service,fake,kernel,args,make=service_fixture(tmp_path);service.review(args['intent'].intent_id)
    service._advance('SUBMISSION_RESERVED',guard=True)
    restarted=make(FakeRobinhoodTransport(fake.journal));result=restarted.submit_fake(args['intent'].intent_id)
    assert result.event.state=='SUBMISSION_RESERVED' and fake.orders()==()


def test_partial_cancel_then_late_fill_remains_hypothetical_and_auditable(tmp_path):
    service,fake,kernel,args,make=service_fixture(tmp_path);identity=args['intent'].intent_id
    service.review(identity);service.submit_fake(identity);order=fake.orders()[0];quantity=args['intent'].quantity
    assert quantity>1
    fake.fill(order['id'],quantity=1,price_cents=args['intent'].limit_price_micros//10000,fee_cents=0);service.observe(identity)
    assert service.cancel_fake(identity).event.state=='CANCEL_REQUESTED'
    assert fake.orders()[0]['state']=='pending_cancelled'
    fake.confirm_cancel(order['id']);assert service.observe(identity).event.state=='CANCELED'
    assert service.reconcile_hypothetical().state=='MATCHED'
    fake.fill(order['id'],quantity=quantity-1,price_cents=args['intent'].limit_price_micros//10000,fee_cents=0)
    assert service.observe(identity).event.state=='FILLED'
    assert service.reconcile_hypothetical().state=='MATCHED'


def test_final_fake_effect_denies_changed_request_and_expired_lease(tmp_path):
    from datetime import timedelta
    service,fake,kernel,args,make=service_fixture(tmp_path);identity=args['intent'].intent_id;service.review(identity)
    request=service._request(True);request['quantity']=str(args['intent'].quantity+1)
    with pytest.raises(ValueError,match='DIFFERS'):fake._commit('PLACE',request,service=service)
    kernel.store._trusted_clock=lambda:args['fence_token'].expires_at+timedelta(seconds=1)
    with pytest.raises(ValueError):service.submit_fake(identity)
    assert fake.orders()==()


def test_real_transport_or_live_classification_cannot_be_injected(tmp_path):
    from src.axiom2.execution.service import OfflineExecutionService
    service,fake,kernel,args,make=service_fixture(tmp_path)
    class RealTransport:
        def submit(self,*args):raise AssertionError('must never run')
    with pytest.raises(TypeError):make(RealTransport())
    changed=dict(args);from src.axiom2.execution.artifacts import SessionArtifactPointer
    pointer=SessionArtifactPointer.from_versioned_payload(changed['artifact_pointer_envelope'].payload).model_copy(update={'source_kind':'LIVE_SHADOW'})
    signer=fake.journal.signer;changed['artifact_pointer_envelope']=signer.sign(pointer,created_at=kernel.store._trusted_clock())
    with pytest.raises(ValueError,match='LIVE'):OfflineExecutionService(kernel,fake,gateway_signer=service.gateway,observer_signer=service.observer,admission=changed,instrument_symbols=dict(service.symbols))


def test_cancel_crash_before_effect_recovers_without_duplicate_effect(tmp_path):
    from src.axiom2.brokers.fake_execution import FakeRobinhoodTransport
    service,fake,kernel,args,make=service_fixture(tmp_path);identity=args['intent'].intent_id
    service.review(identity);service.submit_fake(identity)
    service._advance('CANCEL_REQUESTED',guard=True)
    restarted=make(FakeRobinhoodTransport(fake.journal))
    assert restarted.observe(identity).event.state=='CANCEL_REQUESTED'
    assert restarted.cancel_fake(identity).event.state=='CANCEL_REQUESTED'
    assert restarted.cancel_fake(identity).event.state=='CANCEL_REQUESTED'
    with kernel.store._locked():rows,*_=fake._read()
    assert sum(envelope.payload['action']=='CANCEL' for envelope,_ in rows)==1
    fake.confirm_cancel(fake.orders()[0]['id'])
    assert restarted.observe(identity).event.state=='CANCELED'


def test_fill_while_cancel_pending_preserves_pending_and_updated_quantity(tmp_path):
    service,fake,kernel,args,make=service_fixture(tmp_path);identity=args['intent'].intent_id
    service.review(identity);service.submit_fake(identity);service.cancel_fake(identity)
    order=fake.orders()[0]
    fake.fill(order['id'],quantity=1,price_cents=args['intent'].limit_price_micros//10000,fee_cents=0)
    result=service.observe(identity)
    assert result.event.state=='CANCEL_REQUESTED' and result.event.filled_quantity==1
    fake.confirm_cancel(order['id']);assert service.observe(identity).event.state=='CANCELED'
    assert service.reconcile_hypothetical().state=='MATCHED'


def test_late_partial_fill_keeps_canceled_remainder_and_reconciles(tmp_path):
    service,fake,kernel,args,make=service_fixture(tmp_path);identity=args['intent'].intent_id
    service.review(identity);service.submit_fake(identity);service.cancel_fake(identity)
    order=fake.orders()[0];fake.confirm_cancel(order['id']);service.observe(identity)
    fake.fill(order['id'],quantity=1,price_cents=args['intent'].limit_price_micros//10000,fee_cents=0)
    assert fake.orders()[0]['state']=='partially_filled_rest_cancelled'
    assert service.observe(identity).event.state=='CANCELED'
    assert service.reconcile_hypothetical().state=='MATCHED'


def test_future_provider_checkpoint_is_denied_on_clock_regression(tmp_path):
    from datetime import timedelta
    service,fake,kernel,args,make=service_fixture(tmp_path);identity=args['intent'].intent_id
    service.review(identity);service.submit_fake(identity);start=kernel.store._trusted_clock()
    kernel.store._trusted_clock=lambda:start+timedelta(seconds=2)
    fake.fill(fake.orders()[0]['id'],quantity=args['intent'].quantity,price_cents=args['intent'].limit_price_micros//10000,fee_cents=0)
    kernel.store._trusted_clock=lambda:start+timedelta(seconds=1)
    with pytest.raises(ValueError,match='PROVIDER'):service.observe(identity)
    assert kernel.replay()[-1].event.state=='ACKNOWLEDGED'


@pytest.mark.parametrize('change',['missing','altered'])
def test_provider_checkpoint_digest_and_bytes_verified_at_append_and_replay(tmp_path,change):
    from src.evidence.canonical import canonical_bytes
    from src.evidence.store import StoreCorruptionError
    service,fake,kernel,args,make=service_fixture(tmp_path);identity=args['intent'].intent_id
    service.review(identity);service.submit_fake(identity)
    before=len(kernel.replay())
    if change=='missing':proof=({'invalid':True},{},'0'*64)
    else:
        current=kernel.replay()[-1];proof=({'invalid':True},current.provider_response,current.provider_receipt_digest)
    current=kernel.replay()[-1];now=kernel.store._trusted_clock()
    event=current.event.model_copy(update=dict(sequence=before,previous_digest=kernel.head,state='CANCEL_REQUESTED',role='GATEWAY',actor_id=kernel.actors[('GATEWAY',service.gateway.key_id)]))
    with pytest.raises(ValueError,match='PROVIDER'):kernel.append(service.gateway.sign(event,created_at=now),expected_head=kernel.head,_provider_evidence=proof)
    assert len(kernel.replay())==before
    # Even re-signed outer bytes cannot smuggle an absent/altered checkpoint.
    forged=current.model_copy(update=dict(provider_request=proof[0],provider_response=proof[1],provider_receipt_digest=proof[2]))
    path=next(kernel._directory().glob(f'*{current.event_envelope.payload_digest}.json'))
    path.write_bytes(canonical_bytes(kernel.signer.sign(forged,created_at=forged.received_at)))
    with pytest.raises(StoreCorruptionError):kernel.replay()


def test_valid_other_order_checkpoint_cannot_be_attached_to_lifecycle(tmp_path):
    from uuid import uuid4
    service,fake,kernel,args,make=service_fixture(tmp_path);identity=args['intent'].intent_id
    service.review(identity);service.submit_fake(identity)
    unrelated=fake._commit('PLACE',{**service._request(True),'ref_id':str(uuid4())})
    proof=(unrelated.payload['request'],unrelated.payload['response'],unrelated.payload_digest)
    current=kernel.replay()[-1];now=kernel.store._trusted_clock()
    event=current.event.model_copy(update=dict(sequence=len(kernel.replay()),previous_digest=kernel.head,state='CANCEL_REQUESTED',role='GATEWAY',actor_id=kernel.actors[('GATEWAY',service.gateway.key_id)]))
    with pytest.raises(ValueError,match='PROVIDER_ORDER_SCOPE'):kernel.append(service.gateway.sign(event,created_at=now),expected_head=kernel.head,_provider_evidence=proof)
