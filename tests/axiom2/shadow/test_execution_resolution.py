"""Recompute complete deterministic risk inputs at offline gateway ingress."""
from dataclasses import asdict,replace
import json,runpy
from pathlib import Path
import pytest
from src.evidence.canonical import canonical_bytes


def packet():
    helper=runpy.run_path(str(Path(__file__).parents[1]/'portfolio'/'test_order_intents.py'))
    _,request,policy,decision=helper['base']()
    return json.loads(canonical_bytes({'policy':asdict(policy),'request':asdict(request),'decision':asdict(decision)}))


def test_complete_risk_packet_recomputed_without_model_import():
    from src.axiom2.execution.resolution import reconstruct_risk_packet
    value=packet();policy,request,decision=reconstruct_risk_packet(value)
    assert decision.status=='READY_SHADOW' and decision.policy_digest
    assert json.loads(canonical_bytes(asdict(decision)))==value['decision']


@pytest.mark.parametrize('change',['target','reason','policy','request','authority','extra','snapshot'])
def test_rehashed_forged_decision_or_changed_input_cannot_pass(change):
    from src.axiom2.execution.resolution import reconstruct_risk_packet
    from src.evidence.hashing import content_digest
    value=packet()
    if change=='target':value['decision']['targets'][0]['target_value_cents']+=1
    if change=='reason':value['decision']['reasons']=['APPROVED_BY_AGENT']
    if change=='policy':value['policy']['cash_reserve_bps']+=1
    if change=='request':value['request']['risk_state']='HALTED'
    if change=='authority':value['decision']['capital_authorized']=True
    if change=='extra':value['request']['hidden_submit_client']='broker'
    if change=='snapshot':value['request']['snapshot']['settled_cash_cents']-=1
    if change in ('target','reason'):
        body={key:val for key,val in value['decision'].items() if key!='decision_digest'}
        value['decision']['decision_digest']=content_digest(body)
    with pytest.raises((ValueError,TypeError)):reconstruct_risk_packet(value)


def test_signed_risk_packet_requires_actor_role_and_current_fresh_inputs():
    from datetime import datetime,timedelta
    from src.evidence.signing import Ed25519Signer,TrustStore
    from src.axiom2.execution.resolution import SignedRiskPacket,verify_risk_packet
    value=packet();now=datetime.fromisoformat(value['request']['as_of'].replace('Z','+00:00'))
    signer=Ed25519Signer.generate();trust=TrustStore();trust.add(signer.trusted_key(valid_from=now-timedelta(days=1)))
    record=SignedRiskPacket(actor_id='risk',occurred_at=now,**value);signed=signer.sign(record,created_at=now)
    assert verify_risk_packet(signed,risk_trust_store=trust,actor_bindings={signer.key_id:'risk'},now=now)[2].status=='READY_SHADOW'
    with pytest.raises(ValueError):verify_risk_packet(signed,risk_trust_store=trust,actor_bindings={signer.key_id:'research'},now=now)
    # A new signature cannot refresh the old source request/quote/account facts.
    later=now+timedelta(seconds=31)
    resigned=signer.sign(record.model_copy(update={'occurred_at':later}),created_at=later)
    with pytest.raises(ValueError,match='source'):
        verify_risk_packet(resigned,risk_trust_store=trust,actor_bindings={signer.key_id:'risk'},now=later)


def test_reconstructed_batch_rejects_partial_batch_and_conflicting_account():
    from src.axiom2.execution.resolution import verify_reconstructed_intents,SignedRiskPacket
    from src.evidence.signing import Ed25519Signer,TrustStore
    from src.axiom2.portfolio.intents import build_order_intents
    from datetime import datetime,timedelta
    from src.axiom2.execution.resolution import reconstruct_risk_packet
    value=packet();policy,request,decision=reconstruct_risk_packet(value)
    helper=runpy.run_path(str(Path(__file__).parents[1]/'portfolio'/'test_order_intents.py'))
    account,caps=helper['refs'](request);intents=build_order_intents(decision,request,account,caps,mode='SHADOW')
    signer=Ed25519Signer.generate();trust=TrustStore();trust.add(signer.trusted_key(valid_from=request.as_of-timedelta(days=1)))
    signed=signer.sign(SignedRiskPacket(actor_id='risk',occurred_at=request.as_of,account=json.loads(canonical_bytes(asdict(account))),capabilities=json.loads(canonical_bytes(asdict(caps))),**value),created_at=request.as_of)
    kwargs=dict(risk_trust_store=trust,actor_bindings={signer.key_id:'risk'},now=request.as_of,account=account,capabilities=caps)
    assert verify_reconstructed_intents(signed,intents=intents,**kwargs)[2]==decision
    with pytest.raises(ValueError):verify_reconstructed_intents(signed,intents=intents[:-1],**kwargs)
    with pytest.raises(ValueError):verify_reconstructed_intents(signed,intents=intents,**{**kwargs,'account':replace(account,settled_cash_cents=account.settled_cash_cents-1)})
    with pytest.raises(ValueError):verify_reconstructed_intents(signed,intents=intents,**{**kwargs,'now':account.expires_at+timedelta(microseconds=1)})


@pytest.mark.parametrize('delay',[False,True])
def test_lifecycle_reconstructed_ingress_retains_packet_and_rechecks_under_lock(tmp_path,delay):
    from tests.axiom2.shadow.test_execution_lifecycle import setup
    from tests.axiom2.shadow.test_operator_authorization import fixture
    from src.axiom2.execution.authorization import AuthorizationJournal
    from src.axiom2.execution.lifecycle import LifecycleEvent
    from src.axiom2.execution.resolution import reconstruct_risk_packet,SignedRiskPacket
    from src.axiom2.portfolio.intents import build_order_intents,sign_risk_intent_receipt
    from src.evidence.signing import Ed25519Signer,TrustStore
    from src.evidence.hashing import content_digest
    from datetime import timedelta
    value=packet();policy,request,decision=reconstruct_risk_packet(value);now=request.as_of
    helper=runpy.run_path(str(Path(__file__).parents[1]/'portfolio'/'test_order_intents.py'))
    account,caps=helper['refs'](request);intents=build_order_intents(decision,request,account,caps,mode='SHADOW');order=intents[0]
    kernel,_,producer,_,_=setup(tmp_path/'life');kernel.store._trusted_clock=lambda:now
    operator,gateway,trust,_,scope,declaration,store=fixture(tmp_path/'auth');store._trusted_clock=lambda:now
    scope={**scope,'account_alias':account.account_alias,'intent_digest':content_digest(asdict(order)),
        'capability_digest':caps.capability_digest,'risk_policy_digest':decision.policy_digest,'model_digest':request.model_artifact_hash}
    declaration=declaration.model_copy(update={**scope,'not_before':now-timedelta(seconds=1),'expires_at':now+timedelta(minutes=1),'maximum_order_quantity':order.quantity,'maximum_capital_cents':10**8})
    signed_auth=operator.sign(declaration,created_at=now)
    risk=Ed25519Signer.generate();risk_trust=TrustStore();risk_trust.add(risk.trusted_key(valid_from=now-timedelta(days=1)))
    signed_packet=risk.sign(SignedRiskPacket(actor_id='risk',occurred_at=now,account=json.loads(canonical_bytes(asdict(account))),capabilities=json.loads(canonical_bytes(asdict(caps))),**value),created_at=now)
    risk_receipt=sign_risk_intent_receipt(decision,intents,account,caps,signer=risk,created_at=now)
    proposal=LifecycleEvent(sequence=0,previous_digest=None,order_identity=order.intent_id,intent_digest=scope['intent_digest'],authorization_digest=signed_auth.payload_digest,account_alias=account.account_alias,maximum_quantity=order.quantity,filled_quantity=0,state='PROPOSED',actor_id='gateway',role='GATEWAY',occurred_at=now)
    kwargs=dict(intent=order,intents=intents,risk_envelope=risk_receipt,risk_trust_store=risk_trust,
        risk_packet_envelope=signed_packet,risk_actor_bindings={risk.key_id:'risk'},account=account,capabilities=caps,
        authorization_envelope=signed_auth,authorization_journal=AuthorizationJournal(store,signer=gateway,operator_trust_store=trust),
        expected_scope=scope,capital_cents=10**8,expected_head=None)
    if delay:
        clocks=iter((now,now,now+timedelta(seconds=31)));kernel.store._trusted_clock=lambda:next(clocks)
        with pytest.raises(ValueError):kernel.reserve_reconstructed_proposal(producer.sign(proposal,created_at=now),**kwargs)
        assert kernel.replay()==()
    else:
        receipt=kernel.reserve_reconstructed_proposal(producer.sign(proposal,created_at=now),**kwargs)
        assert receipt.risk_packet_envelope==signed_packet
        assert kernel.replay()[0].risk_packet_envelope==signed_packet
        assert receipt.execution_enabled is False
        from src.axiom2.execution.resolution import audit_reconstructed_receipt
        assert audit_reconstructed_receipt(kernel.replay()[0],risk_trust_store=risk_trust,actor_bindings={risk.key_id:'risk'})[2]==decision
        changed=receipt.model_copy(update={'reconstructed_inputs':{**dict(receipt.reconstructed_inputs),'intents':[]}})
        with pytest.raises(ValueError):audit_reconstructed_receipt(changed,risk_trust_store=risk_trust,actor_bindings={risk.key_id:'risk'})


def test_historical_risk_audit_rejects_changed_proposal(tmp_path):
    # Reuse the real integrated fixture and capture its durable receipt.
    from src.axiom2.execution.lifecycle import ExecutionLifecycle
    from src.axiom2.execution.resolution import audit_reconstructed_receipt
    from src.evidence.contracts import SignedEnvelope
    from src.axiom2.contracts.equity_orders import RiskIntentReceipt
    from src.evidence.signing import Ed25519Signer,TrustStore
    from datetime import timedelta
    captured=[];original=ExecutionLifecycle.reserve_reconstructed_proposal
    def capture(self,*args,**kwargs):
        result=original(self,*args,**kwargs);captured.append((result,kwargs));return result
    from unittest.mock import patch
    with patch.object(ExecutionLifecycle,'reserve_reconstructed_proposal',capture):
        test_lifecycle_reconstructed_ingress_retains_packet_and_rechecks_under_lock(tmp_path,False)
    receipt,kwargs=captured[0];trust=kwargs['risk_trust_store'];actors=kwargs['risk_actor_bindings']
    # Signed recorder attestation is insufficient for independent audit when
    # proposal metadata diverges from the signed selected intent.
    changed_event=receipt.event.model_copy(update={'maximum_quantity':receipt.event.maximum_quantity+1})
    producer=Ed25519Signer.generate()
    changed=receipt.model_copy(update={'event_envelope':producer.sign(changed_event,created_at=receipt.received_at)})
    with pytest.raises(ValueError):audit_reconstructed_receipt(changed,risk_trust_store=trust,actor_bindings=actors)


@pytest.mark.parametrize('change',['future','stale','account','decision','capability','revision'])
def test_historical_risk_audit_repeats_receipt_time_and_scope_checks(tmp_path,change):
    from src.axiom2.execution.lifecycle import ExecutionLifecycle
    from src.axiom2.execution.resolution import audit_reconstructed_receipt
    from src.axiom2.contracts.equity_orders import RiskIntentReceipt
    from src.evidence.signing import Ed25519Signer,TrustStore
    from datetime import timedelta
    captured=[];original=ExecutionLifecycle.reserve_reconstructed_proposal
    def capture(self,*args,**kwargs):
        result=original(self,*args,**kwargs);captured.append((result,kwargs));return result
    from unittest.mock import patch
    with patch.object(ExecutionLifecycle,'reserve_reconstructed_proposal',capture):
        test_lifecycle_reconstructed_ingress_retains_packet_and_rechecks_under_lock(tmp_path,False)
    receipt,kwargs=captured[0];signer=Ed25519Signer.generate();trust=kwargs['risk_trust_store']
    trust.add(signer.trusted_key(valid_from=receipt.received_at-timedelta(days=1)))
    payload=RiskIntentReceipt.from_versioned_payload(receipt.risk_envelope.payload)
    updates={'account':{'account_alias':'other'},'decision':{'decision_digest':'f'*64},
        'capability':{'capability_digest':'f'*64},'revision':{'account_revision':'other'}}.get(change,{})
    time=receipt.received_at+timedelta(seconds=1) if change=='future' else receipt.received_at-timedelta(seconds=31) if change=='stale' else receipt.received_at
    replacement=signer.sign(payload.model_copy(update=updates),created_at=time)
    from src.axiom2.execution.resolution import SignedRiskPacket
    packet_record=SignedRiskPacket.from_versioned_payload(receipt.risk_packet_envelope.payload)
    replacement_packet=signer.sign(packet_record,created_at=receipt.received_at)
    changed=receipt.model_copy(update={'risk_envelope':replacement,'risk_packet_envelope':replacement_packet})
    actors={**kwargs['risk_actor_bindings'],signer.key_id:'risk'}
    with pytest.raises(ValueError,match='stale or future|scope mismatch'):
        audit_reconstructed_receipt(changed,risk_trust_store=trust,actor_bindings=actors)
