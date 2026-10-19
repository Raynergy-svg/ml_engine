"""Strict full-input risk reconstruction; no training/provider/holdout access."""
from dataclasses import asdict,fields
from datetime import datetime,timezone
import json
from pydantic import Field
from src.evidence.contracts import StrictContract,SignedEnvelope
from src.evidence.canonical import canonical_bytes
from src.evidence.signing import verify_envelope
from src.axiom2.portfolio.contracts import (RiskPolicy,PortfolioRequest,PortfolioSnapshot,
    Position,RankedOpportunity,RiskDecision,TargetPosition)
from src.axiom2.portfolio.authority import evaluate_portfolio,verify_risk_decision_integrity


def _record(cls,value,conversions=None):
    if type(value) is not dict or set(value)!={field.name for field in fields(cls)}:
        raise ValueError('exact '+cls.__name__+' contents required')
    body=dict(value)
    for field in fields(cls):
        if not field.init:
            expected=field.default
            if type(body[field.name]) is not type(expected) or body.pop(field.name)!=expected:
                raise ValueError('risk packet authority/schema mismatch')
    for name,convert in (conversions or {}).items():body[name]=convert(body[name])
    result=cls(**body)
    if canonical_bytes(asdict(result))!=canonical_bytes(value):
        raise ValueError('noncanonical '+cls.__name__+' contents')
    return result


def _time(value):
    if type(value) is not str:raise ValueError('canonical timestamp string required')
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def _tuple(value,convert=lambda v:v):
    if type(value) is not list:raise ValueError('canonical JSON array required')
    return tuple(convert(item) for item in value)


def reconstruct_risk_packet(packet):
    """Re-evaluate the complete request/policy, not merely a decision hash.

    Input is canonical JSON-shaped data. Source truth remains the responsibility
    of authenticated artifact/account services at production admission.
    """
    if type(packet) is not dict or set(packet)!={'policy','request','decision'}:
        raise ValueError('complete exact risk packet required')
    policy=_record(RiskPolicy,packet['policy'])
    snapshot=lambda value:_record(PortfolioSnapshot,value,{
        'observed_at':_time,'positions':lambda rows:_tuple(rows,lambda row:_record(Position,row))})
    opportunity=lambda value:_record(RankedOpportunity,value,{'quoted_at':_time,'metadata_at':_time})
    request=_record(PortfolioRequest,packet['request'],{
        'as_of':_time,'session_open':_time,'session_close':_time,'signal_at':_time,
        'snapshot':snapshot,'candidates':lambda rows:_tuple(rows,opportunity)})
    decision=_record(RiskDecision,packet['decision'],{
        'as_of':_time,'reasons':_tuple,'targets':lambda rows:_tuple(rows,lambda row:_record(TargetPosition,row))})
    verify_risk_decision_integrity(decision)
    computed=evaluate_portfolio(request,policy)
    if canonical_bytes(asdict(computed))!=canonical_bytes(asdict(decision)):
        raise ValueError('risk decision differs from complete deterministic reconstruction')
    return policy,request,decision


class SignedRiskPacket(StrictContract):
    actor_id:str=Field(min_length=1)
    occurred_at:datetime
    policy:dict
    request:dict
    decision:dict
    account:dict|None=None
    capabilities:dict|None=None


def verify_risk_packet(envelope,*,risk_trust_store,actor_bindings,now):
    envelope=SignedEnvelope.model_validate_json(canonical_bytes(envelope),strict=True)
    record=verify_envelope(envelope,SignedRiskPacket,risk_trust_store)
    risk_trust_store.require_trusted_at_receipt(envelope.signature.key_id,now)
    if actor_bindings.get(envelope.signature.key_id)!=record.actor_id:
        raise ValueError('risk actor/key role mismatch')
    if record.occurred_at.tzinfo is None or abs((record.occurred_at-envelope.signature.created_at).total_seconds())>5:
        raise ValueError('risk packet signing time mismatch')
    if record.occurred_at>now or not 0<=(now-envelope.signature.created_at).total_seconds()<=30:
        raise ValueError('risk packet stale or future at receipt')
    policy,request,decision=reconstruct_risk_packet(json.loads(canonical_bytes({
        'policy':record.policy,'request':record.request,'decision':record.decision})))
    validate_reconstructed_sources(policy,request,now)
    return policy,request,decision


def validate_reconstructed_sources(policy,request,now):
    """Check source ages against the trusted current clock after reconstruction."""
    if now.tzinfo is None:
        raise ValueError('trusted receipt clock must be timezone aware')
    current=now.astimezone(timezone.utc)
    sources=[(request.as_of,30),(request.snapshot.observed_at,policy.snapshot_ttl_seconds),
        (request.signal_at,policy.signal_ttl_seconds)]
    sources.extend((item.quoted_at,policy.quote_ttl_seconds) for item in request.candidates)
    sources.extend((item.metadata_at,policy.metadata_ttl_seconds) for item in request.candidates)
    if any(not 0<=(current-stamp.astimezone(timezone.utc)).total_seconds()<=ttl
           for stamp,ttl in sources):
        raise ValueError('risk source inputs stale or future at receipt')
    if not request.session_open.astimezone(timezone.utc)<=current<request.session_close.astimezone(timezone.utc):
        raise ValueError('risk source session closed at receipt')


def verify_reconstructed_intents(envelope,*,intents,account,capabilities,
        risk_trust_store,actor_bindings,now):
    """Bind a complete reconstructed batch to normalized offline observations.

    Signature authenticates the configured risk service, not genuine capital
    artifacts or live broker facts. The public execution gateway stays disabled.
    """
    from src.axiom2.portfolio.intents import build_order_intents
    from src.axiom2.contracts.equity_orders import AccountObservationRef
    policy,request,decision=verify_risk_packet(envelope,risk_trust_store=risk_trust_store,
        actor_bindings=actor_bindings,now=now)
    if type(account) is not AccountObservationRef:
        raise ValueError('exact normalized account observation required')
    if not account.observed_at<=now<account.expires_at:
        raise ValueError('account source expired or future at receipt')
    snapshot=request.snapshot
    record=verify_envelope(envelope,SignedRiskPacket,risk_trust_store)
    if record.account is None or record.capabilities is None or canonical_bytes(record.account)!=canonical_bytes(asdict(account)) or canonical_bytes(record.capabilities)!=canonical_bytes(asdict(capabilities)):
        raise ValueError('risk packet must bind complete account and capability observations')
    expected_positions=tuple(sorted(position.instrument_id for position in snapshot.positions))
    if (account.settled_cash_cents,account.reserved_cash_cents,tuple(sorted(instrument for instrument,quantity in account.positions if quantity>0)))!=(
            snapshot.settled_cash_cents,snapshot.reserved_cash_cents,expected_positions):
        raise ValueError('account source differs from reconstructed risk snapshot')
    computed=build_order_intents(decision,request,account,capabilities,mode='SHADOW')
    if type(intents) is not tuple or canonical_bytes(tuple(asdict(item) for item in intents))!=canonical_bytes(tuple(asdict(item) for item in computed)):
        raise ValueError('complete reconstructed intent batch mismatch')
    return policy,request,decision


def audit_reconstructed_receipt(receipt,*,risk_trust_store,actor_bindings):
    """Independently reconstruct retained offline admission at its receipt time.

    Caller must first authenticate the outer lifecycle receipt. Historical audit
    is separate from current admission; it never grants submission permission.
    """
    from src.axiom2.contracts.equity_orders import AccountObservationRef,BrokerCapabilities,EquityOrderIntent
    from src.axiom2.portfolio.intents import verify_risk_intent_receipt
    if receipt.reconstructed_inputs is None or receipt.risk_packet_envelope is None or receipt.risk_envelope is None:
        raise ValueError('complete retained reconstructed admission required')
    inputs=json.loads(canonical_bytes(receipt.reconstructed_inputs))
    if set(inputs)!={'account','capabilities','intents'}:
        raise ValueError('exact retained reconstruction inputs required')
    account=_record(AccountObservationRef,inputs['account'],{'observed_at':_time,'expires_at':_time,
        'positions':lambda rows:_tuple(rows,lambda row:_tuple(row))})
    capabilities=_record(BrokerCapabilities,inputs['capabilities'])
    intents=_tuple(inputs['intents'],lambda row:_record(EquityOrderIntent,row,{'expires_at':_time}))
    result=verify_reconstructed_intents(receipt.risk_packet_envelope,intents=intents,account=account,
        capabilities=capabilities,risk_trust_store=risk_trust_store,actor_bindings=actor_bindings,now=receipt.received_at)
    risk=verify_risk_intent_receipt(receipt.risk_envelope,intents,trust_store=risk_trust_store)
    risk_trust_store.require_trusted_at_receipt(receipt.risk_envelope.signature.key_id,receipt.received_at)
    if not 0<=(receipt.received_at-receipt.risk_envelope.signature.created_at).total_seconds()<=30:
        raise ValueError('retained risk receipt stale or future at admission')
    for intent in intents:
        if (risk.decision_digest,risk.account_alias,risk.account_revision,risk.capability_digest)!=(
                intent.decision_digest,intent.account_alias,intent.account_revision,intent.capability_digest):
            raise ValueError('retained risk receipt scope mismatch')
        if receipt.received_at>=intent.expires_at:
            raise ValueError('retained risk intent expired at admission')
    if receipt.risk_envelope.signature.key_id!=receipt.risk_packet_envelope.signature.key_id:
        raise ValueError('retained risk actor mismatch')
    selected=[intent for intent in intents if intent.intent_id==receipt.event.order_identity]
    if len(selected)!=1:
        raise ValueError('retained selected intent missing')
    from src.evidence.hashing import content_digest
    event=receipt.event;intent=selected[0]
    if event.state!='PROPOSED' or event.filled_quantity or event.maximum_quantity!=intent.quantity or event.account_alias!=intent.account_alias or receipt.authorization_envelope is None or event.authorization_digest!=receipt.authorization_envelope.payload_digest or receipt.nonce_request_digest is None:
        raise ValueError('retained proposal scope mismatch')
    if event.intent_digest!=content_digest(asdict(intent)):
        raise ValueError('retained selected intent mismatch')
    return result
