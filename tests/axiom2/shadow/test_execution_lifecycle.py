"""Offline signed execution lifecycle, never broker/provider capability."""
from datetime import timedelta
import pytest
from src.evidence.signing import Ed25519Signer,TrustStore
from src.evidence.store import EvidenceStore,ConcurrentHeadError
from src.evidence.transition_policy import AuthorityRegistry
from tests.axiom2.shadow.test_shadow_journal import NOW


def setup(tmp_path):
    from src.axiom2.execution.lifecycle import ExecutionLifecycle,LifecycleEvent
    producer=Ed25519Signer.generate();observer=Ed25519Signer.generate();receipt=Ed25519Signer.generate()
    roles={};actors={}
    for role,signer in [('GATEWAY',producer),('OBSERVER',observer)]:
        trust=TrustStore();trust.add(signer.trusted_key(valid_from=NOW-timedelta(days=1)))
        roles[role]=trust;actors[(role,signer.key_id)]=role.lower()
    trust=TrustStore();trust.add(receipt.trusted_key(valid_from=NOW-timedelta(days=1)))
    store=EvidenceStore(tmp_path,trust_store=trust,authorities=AuthorityRegistry(),trusted_clock=lambda:NOW)
    kernel=ExecutionLifecycle(store,signer=receipt,role_trust_stores=roles,actor_bindings=actors)
    def event(state,quantity=0,*,signer=None,expected_head=None,identity='offline-order-one',intent_digest='a'*64,authorization_digest='b'*64):
        current=kernel.replay();seq=len(current);head=kernel.head
        role='OBSERVER' if state in ('ACKNOWLEDGED','PARTIALLY_FILLED','FILLED','CANCELED','REJECTED') else 'GATEWAY'
        actor=observer if role=='OBSERVER' else producer
        record=LifecycleEvent(sequence=seq,previous_digest=head,order_identity=identity,
            intent_digest=intent_digest,authorization_digest=authorization_digest,account_alias='dedicated',
            maximum_quantity=5,filled_quantity=quantity,state=state,actor_id=role.lower(),role=role,occurred_at=NOW)
        return kernel.append((signer or actor).sign(record,created_at=NOW),expected_head=head if expected_head is None else expected_head)
    return kernel,event,producer,observer,roles


def test_proposal_approval_and_cancel_request_are_not_fills(tmp_path):
    kernel,event,*_=setup(tmp_path)
    for state in ('PROPOSED','APPROVAL_PENDING','REVIEWED','SUBMISSION_RESERVED','ACKNOWLEDGED','CANCEL_REQUESTED'):
        receipt=event(state)
        assert receipt.event.filled_quantity==0
        assert receipt.execution_enabled is False and receipt.capital_authorized is False
    assert event('PARTIALLY_FILLED',2).event.filled_quantity==2
    assert event('CANCELED',2).event.filled_quantity==2
    # A cancel acknowledgement never suppresses a late observed fill.
    assert event('FILLED',5).event.filled_quantity==5
    assert len(kernel.replay())==9


def test_ambiguous_submission_cannot_be_blindly_reserved_again(tmp_path):
    kernel,event,*_=setup(tmp_path)
    for state in ('PROPOSED','REVIEWED','SUBMISSION_RESERVED','SUBMISSION_UNKNOWN'):event(state)
    with pytest.raises(ValueError,match='transition'):event('SUBMISSION_RESERVED')
    with pytest.raises(ValueError,match='transition'):event('REVIEWED')
    assert event('ACKNOWLEDGED').event.state=='ACKNOWLEDGED'


def test_wrong_role_stale_cas_and_fill_regression_are_denied(tmp_path):
    kernel,event,producer,observer,roles=setup(tmp_path)
    event('PROPOSED');event('REVIEWED');event('SUBMISSION_RESERVED')
    with pytest.raises(ValueError):event('ACKNOWLEDGED',signer=producer)
    with pytest.raises(ConcurrentHeadError):event('ACKNOWLEDGED',expected_head='0'*64)
    event('ACKNOWLEDGED');event('PARTIALLY_FILLED',2)
    with pytest.raises(ValueError):event('PARTIALLY_FILLED',1)
    with pytest.raises(ValueError):event('FILLED',6)
    with pytest.raises(ValueError):event('FILLED',4)


def test_renamed_identity_cannot_retry_ambiguous_submission(tmp_path):
    kernel,event,*_=setup(tmp_path)
    for state in ('PROPOSED','REVIEWED','SUBMISSION_RESERVED','SUBMISSION_UNKNOWN'):event(state)
    with pytest.raises(ValueError,match='scope|ambiguous'):
        event('PROPOSED',identity='renamed-order')
    with pytest.raises(ValueError,match='ambiguous'):
        event('PROPOSED',identity='new-exposure',intent_digest='c'*64,authorization_digest='d'*64)


@pytest.mark.parametrize('delayed',[False,True])
def test_proposal_ingress_binds_signed_risk_and_operator_nonce(tmp_path,delayed):
    from dataclasses import asdict
    from src.evidence.hashing import content_digest
    from src.axiom2.contracts.equity_orders import RiskIntentReceipt
    from src.axiom2.execution.lifecycle import LifecycleEvent
    from src.axiom2.execution.authorization import AuthorizationJournal
    from tests.axiom2.shadow.test_operator_authorization import fixture
    from tests.axiom2.shadow.test_shadow_journal import intent
    kernel,event,producer,observer,roles=setup(tmp_path/'lifecycle')
    operator,gateway,trust,receipts,expected,declaration,store=fixture(tmp_path/'authorization')
    order=intent();expected={**expected,'intent_digest':content_digest(asdict(order)),'capability_digest':order.capability_digest}
    declaration=declaration.model_copy(update={'intent_digest':expected['intent_digest'],'capability_digest':order.capability_digest})
    signed_auth=operator.sign(declaration,created_at=NOW)
    risk=Ed25519Signer.generate();risk_trust=TrustStore();risk_trust.add(risk.trusted_key(valid_from=NOW-timedelta(days=1)))
    risk_payload=RiskIntentReceipt(decision_digest=order.decision_digest,intents_digest=content_digest((asdict(order),)),
        account_alias=order.account_alias,account_revision=order.account_revision,capability_digest=order.capability_digest,
        mode='SHADOW',intent_count=1)
    proposal=LifecycleEvent(sequence=0,previous_digest=None,order_identity=order.intent_id,intent_digest=expected['intent_digest'],
        authorization_digest=signed_auth.payload_digest,account_alias=order.account_alias,maximum_quantity=order.quantity,
        filled_quantity=0,state='PROPOSED',actor_id='gateway',role='GATEWAY',occurred_at=NOW)
    journal=AuthorizationJournal(store,signer=gateway,operator_trust_store=trust)
    arguments=dict(intent=order,intents=(order,),risk_trust_store=risk_trust,
        authorization_envelope=signed_auth,authorization_journal=journal,expected_scope=expected,capital_cents=100,expected_head=None)
    with pytest.raises(ValueError):kernel.reserve_proposal(producer.sign(proposal,created_at=NOW),risk_envelope=producer.sign(risk_payload,created_at=NOW),**arguments)
    assert kernel.replay()==()
    if delayed:
        clocks=iter((NOW,NOW+timedelta(seconds=2)))
        kernel.store._trusted_clock=lambda:next(clocks)
        with pytest.raises(ValueError,match='durable admission'):
            kernel.reserve_proposal(producer.sign(proposal,created_at=NOW),risk_envelope=risk.sign(risk_payload,created_at=NOW-timedelta(seconds=29)),**arguments)
        assert kernel.replay()==()
        return
    receipt=kernel.reserve_proposal(producer.sign(proposal,created_at=NOW),risk_envelope=risk.sign(risk_payload,created_at=NOW),**arguments)
    assert receipt.risk_envelope is not None and receipt.authorization_envelope==signed_auth
    assert receipt.nonce_request_digest is not None
    assert receipt.event.order_identity==order.intent_id and receipt.execution_enabled is False
    assert len(list((tmp_path/'authorization'/'offline-authorizations').glob('*.json')))==1


def test_crash_before_and_after_commit_restart(tmp_path,monkeypatch):
    from src.axiom2.execution.lifecycle import ExecutionLifecycle
    kernel,event,*_=setup(tmp_path)
    atomic=kernel.store._atomic_create_bytes
    def before(*args,**kwargs):raise RuntimeError('before commit')
    monkeypatch.setattr(kernel.store,'_atomic_create_bytes',before)
    with pytest.raises(RuntimeError):event('PROPOSED')
    assert kernel.replay()==()
    def after(*args,**kwargs):atomic(*args,**kwargs);raise RuntimeError('after commit')
    monkeypatch.setattr(kernel.store,'_atomic_create_bytes',after)
    with pytest.raises(RuntimeError):event('PROPOSED')
    restarted=ExecutionLifecycle(kernel.store,signer=kernel.signer,role_trust_stores=kernel.roles,actor_bindings=kernel.actors)
    assert len(restarted.replay())==1
    assert restarted.replay()[0].event.state=='PROPOSED'


def test_signed_receipt_tamper_and_key_retirement(tmp_path):
    from src.evidence.store import StoreCorruptionError
    kernel,event,producer,observer,roles=setup(tmp_path)
    event('PROPOSED')
    roles['GATEWAY'].revoke(producer.key_id,NOW+timedelta(seconds=1))
    kernel.store._trusted_clock=lambda:NOW+timedelta(seconds=2)
    assert len(kernel.replay())==1
    with pytest.raises(ValueError):event('REVIEWED')
    path=next((tmp_path/'offline-execution-lifecycle').glob('*.json'))
    path.write_bytes(path.read_bytes().replace(b'PROPOSED',b'REVIEWED'))
    with pytest.raises(StoreCorruptionError):kernel.replay()


def lifecycle_worker(root,keys,envelope,queue):
    from pathlib import Path
    from src.axiom2.execution.lifecycle import ExecutionLifecycle
    from src.evidence.contracts import SignedEnvelope
    producer,observer,receipt=[Ed25519Signer.from_private_bytes(value) for value in keys]
    roles={};actors={}
    for role,signer in [('GATEWAY',producer),('OBSERVER',observer)]:
        trust=TrustStore();trust.add(signer.trusted_key(valid_from=NOW-timedelta(days=1)))
        roles[role]=trust;actors[(role,signer.key_id)]=role.lower()
    receipts=TrustStore();receipts.add(receipt.trusted_key(valid_from=NOW-timedelta(days=1)))
    store=EvidenceStore(Path(root),trust_store=receipts,authorities=AuthorityRegistry(),trusted_clock=lambda:NOW)
    kernel=ExecutionLifecycle(store,signer=receipt,role_trust_stores=roles,actor_bindings=actors)
    try:
        result=kernel.append(SignedEnvelope.model_validate_json(envelope,strict=True),expected_head=None)
        queue.put(('OK',result.event_envelope.payload_digest))
    except Exception as exc:queue.put(('ERROR',repr(exc)))


def test_real_multiprocess_lifecycle_append_race(tmp_path):
    import multiprocessing
    from src.evidence.canonical import canonical_bytes
    from src.axiom2.execution.lifecycle import LifecycleEvent
    kernel,event,producer,observer,roles=setup(tmp_path)
    record=LifecycleEvent(sequence=0,previous_digest=None,order_identity='offline-order-one',intent_digest='a'*64,
        authorization_digest='b'*64,account_alias='dedicated',maximum_quantity=5,filled_quantity=0,
        state='PROPOSED',actor_id='gateway',role='GATEWAY',occurred_at=NOW)
    data=canonical_bytes(producer.sign(record,created_at=NOW));keys=[producer.private_bytes(),observer.private_bytes(),kernel.signer.private_bytes()]
    context=multiprocessing.get_context('spawn');queue=context.Queue()
    workers=[context.Process(target=lifecycle_worker,args=(str(tmp_path),keys,data,queue)) for _ in range(4)]
    for worker in workers:worker.start()
    for worker in workers:worker.join(20);assert worker.exitcode==0
    outcomes=[queue.get(timeout=2) for _ in workers]
    assert all(row[0]=='OK' for row in outcomes),outcomes
    assert len({row[1] for row in outcomes})==1
    assert len(kernel.replay())==1


def test_prequeued_order_cannot_submit_after_account_becomes_ambiguous(tmp_path):
    kernel,event,*_=setup(tmp_path)
    event('PROPOSED');event('PROPOSED',identity='queued',intent_digest='c'*64,authorization_digest='d'*64)
    for state in ('REVIEWED','SUBMISSION_RESERVED','SUBMISSION_UNKNOWN'):event(state)
    with pytest.raises(ValueError,match='ambiguous'):
        event('REVIEWED',identity='queued',intent_digest='c'*64,authorization_digest='d'*64)
