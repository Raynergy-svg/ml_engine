"""Durable offline submission-state rehearsal. No broker or capital authority.

Digest references here bind signed fixture bytes, not genuine holdout or provider
truth. The public ExecutionAuthority remains disabled; this journal cannot submit
orders or establish readiness. Actor/key role configuration is a trusted input.
"""
from dataclasses import asdict
from datetime import datetime
import json
from typing import Literal
from pydantic import Field
from src.evidence.contracts import StrictContract,SignedEnvelope
from src.evidence.canonical import canonical_bytes
from src.evidence.signing import verify_envelope
from src.evidence.hashing import content_digest
from src.evidence.store import ConcurrentHeadError,StoreCorruptionError

STATES = Literal['PROPOSED','APPROVAL_PENDING','REVIEWED','SUBMISSION_RESERVED',
    'SUBMISSION_UNKNOWN','ACKNOWLEDGED','PARTIALLY_FILLED','FILLED',
    'CANCEL_REQUESTED','CANCELED','EXPIRED','REJECTED']
OBSERVED_STATES=frozenset(('ACKNOWLEDGED','PARTIALLY_FILLED','FILLED','CANCELED','REJECTED'))
TRANSITIONS={
    None:{'PROPOSED'},
    'PROPOSED':{'APPROVAL_PENDING','REVIEWED','EXPIRED'},
    'APPROVAL_PENDING':{'REVIEWED','EXPIRED'},
    'REVIEWED':{'SUBMISSION_RESERVED','EXPIRED'},
    'SUBMISSION_RESERVED':{'ACKNOWLEDGED','SUBMISSION_UNKNOWN','REJECTED'},
    'SUBMISSION_UNKNOWN':{'ACKNOWLEDGED','REJECTED'},
    'ACKNOWLEDGED':{'PARTIALLY_FILLED','FILLED','CANCEL_REQUESTED','CANCELED'},
    'PARTIALLY_FILLED':{'PARTIALLY_FILLED','FILLED','CANCEL_REQUESTED','CANCELED'},
    'CANCEL_REQUESTED':{'PARTIALLY_FILLED','FILLED','CANCELED'},
    # Authoritatively observed late fills remain visible after cancel acknowledgement.
    'CANCELED':{'PARTIALLY_FILLED','FILLED','CANCELED'},
    'FILLED':set(),'EXPIRED':set(),'REJECTED':set(),
}

class LifecycleEvent(StrictContract):
    sequence:int=Field(ge=0)
    previous_digest:str|None
    order_identity:str=Field(min_length=1)
    intent_digest:str=Field(pattern=r'^[0-9a-f]{64}$')
    authorization_digest:str=Field(pattern=r'^[0-9a-f]{64}$')
    account_alias:str=Field(min_length=1)
    maximum_quantity:int=Field(gt=0)
    filled_quantity:int=Field(ge=0)
    state:STATES
    actor_id:str=Field(min_length=1)
    role:Literal['GATEWAY','OBSERVER']
    occurred_at:datetime
    mode:Literal['OFFLINE_PREPARATION']='OFFLINE_PREPARATION'
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False

class LifecycleReceipt(StrictContract):
    event_envelope:SignedEnvelope
    received_at:datetime
    mode:Literal['OFFLINE_PREPARATION']='OFFLINE_PREPARATION'
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False
    provider_request:dict|None=None
    provider_response:dict|None=None
    provider_receipt_digest:str|None=None
    artifact_pointer_envelope:SignedEnvelope|None=None
    artifact_resolution:dict|None=None
    fence_reservation:dict|None=None
    reconstructed_inputs:dict|None=None
    risk_packet_envelope:SignedEnvelope|None=None
    risk_envelope:SignedEnvelope|None=None
    authorization_envelope:SignedEnvelope|None=None
    nonce_request_digest:str|None=Field(default=None,pattern=r'^[0-9a-f]{64}$')

    @property
    def event(self):
        return LifecycleEvent.from_versioned_payload(self.event_envelope.payload)

class ExecutionLifecycle:
    """Lock/fsync/CAS journal with signed actor trust at each receipt.

    Whole-store rollback needs external checkpointing. No production secret,
    broker client, callback, enable flag or alternative execution route exists.
    """
    def __init__(self,store,*,signer,role_trust_stores,actor_bindings,provider_journal=None):
        if set(role_trust_stores)!={'GATEWAY','OBSERVER'}:
            raise ValueError('exact offline lifecycle role stores required')
        self.store=store;self.signer=signer
        self.roles=role_trust_stores;self.actors=actor_bindings
        self.provider_journal=provider_journal

    def _verify_provider_checkpoint(self,request,response,digest,received_at,event):
        if request is None and response is None and digest is None:return
        from src.axiom2.execution.journal import ExecutionJournal
        from src.axiom2.brokers.fake_execution import FakeRobinhoodTransport
        if type(self.provider_journal) is not ExecutionJournal or self.provider_journal.store is not self.store:
            raise ValueError('EXACT_PROVIDER_JOURNAL_REQUIRED')
        rows,*_=FakeRobinhoodTransport(self.provider_journal)._read()
        matches=[(envelope,time) for envelope,time in rows if envelope.payload_digest==digest]
        if len(matches)!=1:raise ValueError('SIGNED_PROVIDER_CHECKPOINT_MISSING')
        envelope,time=matches[0]
        if time>received_at:raise ValueError('PROVIDER_CHECKPOINT_FROM_FUTURE')
        if canonical_bytes(envelope.payload['request'])!=canonical_bytes(request) or canonical_bytes(envelope.payload['response'])!=canonical_bytes(response):
            raise ValueError('PROVIDER_CHECKPOINT_BYTES_CHANGED')
        from uuid import uuid5
        from src.axiom2.brokers.fake_execution import NAMESPACE
        from src.axiom2.brokers.robinhood_readonly import _units
        ref=str(uuid5(NAMESPACE,event.account_alias+':'+event.intent_digest))
        order_id=str(uuid5(NAMESPACE,ref));action=envelope.payload['action']
        if action=='REVIEW':
            if event.state!='REVIEWED' or request['account_number']!='FAKE:'+event.account_alias or _units(request['quantity'],1)!=event.maximum_quantity:
                raise ValueError('PROVIDER_REVIEW_SCOPE_CHANGED')
        elif action=='CANCEL':
            if event.state!='CANCEL_REQUESTED' or request!={'account_number':'FAKE:'+event.account_alias,'order_id':order_id}:
                raise ValueError('PROVIDER_CANCEL_SCOPE_CHANGED')
        else:
            order=response['data']['order']
            if order['id']!=order_id or order.get('ref_id')!=ref or _units(order['quantity'],1)!=event.maximum_quantity:
                raise ValueError('PROVIDER_ORDER_SCOPE_CHANGED')
            if action=='PLACE' and request['account_number']!='FAKE:'+event.account_alias:
                raise ValueError('PROVIDER_ACCOUNT_SCOPE_CHANGED')
            quantity=_units(order['cumulative_quantity'],1)
            if event.state in ('PARTIALLY_FILLED','FILLED','CANCELED','CANCEL_REQUESTED') and quantity!=event.filled_quantity:
                raise ValueError('PROVIDER_FILL_QUANTITY_CHANGED')
            if event.state=='FILLED' and order['state']!='filled' or event.state=='CANCELED' and order['state'] not in ('cancelled','partially_filled_rest_cancelled'):
                raise ValueError('PROVIDER_TERMINAL_STATE_CHANGED')


    def _directory(self):
        directory=self.store.root/'offline-execution-lifecycle'
        if directory.is_symlink() or directory.exists() and not directory.is_dir():
            raise StoreCorruptionError('invalid lifecycle directory')
        return directory

    def _verify_event(self,envelope,received_at):
        raw=LifecycleEvent.from_versioned_payload(envelope.payload)
        expected_role='OBSERVER' if raw.state in OBSERVED_STATES else 'GATEWAY'
        if raw.role!=expected_role:raise ValueError('wrong lifecycle role')
        trust=self.roles[expected_role]
        event=verify_envelope(envelope,LifecycleEvent,trust)
        trust.require_trusted_at_receipt(envelope.signature.key_id,received_at)
        if self.actors.get((event.role,envelope.signature.key_id))!=event.actor_id:
            raise ValueError('actor/key binding mismatch')
        if event.occurred_at.tzinfo is None or received_at.tzinfo is None:
            raise ValueError('aware lifecycle time required')
        if abs((event.occurred_at-envelope.signature.created_at).total_seconds())>5:
            raise ValueError('signing time mismatch')
        if event.occurred_at>received_at or not 0<=(received_at-envelope.signature.created_at).total_seconds()<=30:
            raise ValueError('receipt time mismatch')
        return event

    @staticmethod
    def _scope(event,states):
        advancing=event.state in ('PROPOSED','APPROVAL_PENDING','REVIEWED','SUBMISSION_RESERVED')
        for old in states.values():
            if old.order_identity!=event.order_identity and old.account_alias==event.account_alias and old.state=='SUBMISSION_UNKNOWN' and advancing:
                raise ValueError('ambiguous account blocks new exposure')
            if event.order_identity not in states and (old.authorization_digest==event.authorization_digest or (old.account_alias,old.intent_digest)==(event.account_alias,event.intent_digest)):
                raise ValueError('execution scope already bound to another order identity')

    @staticmethod
    def _transition(event,previous):
        old_state=previous.state if previous else None
        if event.state not in TRANSITIONS[old_state]:raise ValueError('invalid lifecycle transition')
        if previous:
            fields=('intent_digest','authorization_digest','account_alias','maximum_quantity')
            if any(getattr(event,key)!=getattr(previous,key) for key in fields):
                raise ValueError('changed lifecycle scope')
            if event.filled_quantity<previous.filled_quantity:raise ValueError('fill quantity regressed')
        if event.filled_quantity>event.maximum_quantity:raise ValueError('fill exceeds reserved quantity')
        if event.state=='FILLED' and event.filled_quantity!=event.maximum_quantity:
            raise ValueError('FILLED quantity must match reservation')
        if event.state=='PARTIALLY_FILLED' and not 0<event.filled_quantity<event.maximum_quantity:
            raise ValueError('PARTIAL quantity invalid')
        if event.state not in ('PARTIALLY_FILLED','FILLED','CANCELED','CANCEL_REQUESTED') and event.filled_quantity:
            raise ValueError('proposal/approval/acknowledgement is not a fill')
        if event.state=='CANCEL_REQUESTED' and previous and event.filled_quantity!=previous.filled_quantity:
            raise ValueError('cancel request cannot report fills')

    def _read(self):
        directory=self._directory();rows=[];states={};head=None
        if not directory.exists():return rows,states
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file():raise StoreCorruptionError('invalid lifecycle entry')
            if path.name.startswith('.') and '.tmp-' in path.name:continue
            try:
                data=path.read_bytes();outer=SignedEnvelope.model_validate_json(data,strict=True)
                if canonical_bytes(outer)!=data:raise ValueError('noncanonical lifecycle receipt')
                receipt=verify_envelope(outer,LifecycleReceipt,self.store.trust_store)
                if outer.signature.created_at!=receipt.received_at:raise ValueError('receipt timestamp mismatch')
                self.store.trust_store.require_trusted_at_receipt(outer.signature.key_id,receipt.received_at)
                event=self._verify_event(receipt.event_envelope,receipt.received_at)
                if event.sequence!=len(rows) or event.previous_digest!=head:raise ValueError('lifecycle chain')
                if path.name!=f'{len(rows):020d}-{receipt.event_envelope.payload_digest}.json':raise ValueError('lifecycle address')
                if rows and receipt.received_at<rows[-1].received_at:raise ValueError('lifecycle clock regression')
                self._verify_provider_checkpoint(receipt.provider_request,receipt.provider_response,receipt.provider_receipt_digest,receipt.received_at,event)
                self._scope(event,states)
                self._transition(event,states.get(event.order_identity))
                rows.append(receipt);states[event.order_identity]=event;head=receipt.event_envelope.payload_digest
            except (ValueError,TypeError,OSError) as exc:
                raise StoreCorruptionError('invalid signed lifecycle receipt') from exc
        return rows,states

    def replay(self):
        with self.store._locked():return tuple(self._read()[0])

    @property
    def head(self):
        with self.store._locked():
            rows,_=self._read();return rows[-1].event_envelope.payload_digest if rows else None

    def append(self,envelope,*,expected_head,_risk_validation=None,_authorization_validation=None,_packet_validation=None,_fence_validation=None,_artifact_validation=None,_provider_evidence=None):
        envelope=SignedEnvelope.model_validate_json(canonical_bytes(envelope),strict=True)
        with self.store._locked():
            rows,states=self._read();head=rows[-1].event_envelope.payload_digest if rows else None
            if rows and rows[-1].event_envelope==envelope:return rows[-1]
            if head!=expected_head:raise ConcurrentHeadError('lifecycle head changed')
            now=self.store._trusted_clock();event=self._verify_event(envelope,now)
            if _artifact_validation is not None:
                resolver,pointer,resolution=_artifact_validation
                _,now=resolver._verify_unlocked(pointer,resolution,now,return_time=True)
                self._verify_event(envelope,now)
            if _fence_validation is not None:
                fence,token,reservation=_fence_validation
                fence._verify_unlocked(token,reservation,now)
            if _packet_validation is not None:
                from src.axiom2.execution.resolution import verify_reconstructed_intents
                packet_envelope,packet_arguments=_packet_validation
                verify_reconstructed_intents(packet_envelope,now=now,**packet_arguments)
            if _authorization_validation is not None:
                from src.axiom2.execution.authorization import verify_authorization
                auth_envelope,auth_trust,scope,quantity,capital,nonce_digest=_authorization_validation
                verify_authorization(auth_envelope,auth_trust,expected=scope,now=now,quantity=quantity,capital_cents=capital)
            if _risk_validation is not None:
                from src.axiom2.portfolio.intents import verify_risk_intent_receipt
                risk_envelope,intents,risk_trust_store=_risk_validation
                verify_risk_intent_receipt(risk_envelope,intents,trust_store=risk_trust_store)
                risk_trust_store.require_trusted_at_receipt(risk_envelope.signature.key_id,now)
                if not 0<=(now-risk_envelope.signature.created_at).total_seconds()<=30:
                    raise ValueError('risk receipt stale at durable admission')
                if any(now>=intent.expires_at for intent in intents):
                    raise ValueError('risk intent expired at durable admission')
            if event.sequence!=len(rows) or event.previous_digest!=head:raise ValueError('lifecycle chain')
            if rows and now<rows[-1].received_at:raise ValueError('lifecycle clock regressed')
            self._scope(event,states)
            self._transition(event,states.get(event.order_identity))
            receipt=LifecycleReceipt(provider_request=_provider_evidence[0] if _provider_evidence else None,
                provider_response=_provider_evidence[1] if _provider_evidence else None,
                provider_receipt_digest=_provider_evidence[2] if _provider_evidence else None,event_envelope=envelope,received_at=now,
                artifact_pointer_envelope=_artifact_validation[1] if _artifact_validation else None,
                artifact_resolution=json.loads(canonical_bytes(_artifact_validation[2])) if _artifact_validation else None,
                fence_reservation=json.loads(canonical_bytes(_fence_validation[2])) if _fence_validation else None,
                reconstructed_inputs=json.loads(canonical_bytes({
                    'account':asdict(_packet_validation[1]['account']),
                    'capabilities':asdict(_packet_validation[1]['capabilities']),
                    'intents':tuple(asdict(order) for order in _packet_validation[1]['intents'])})) if _packet_validation else None,
                risk_packet_envelope=_packet_validation[0] if _packet_validation else None,
                risk_envelope=_risk_validation[0] if _risk_validation else None,
                authorization_envelope=_authorization_validation[0] if _authorization_validation else None,
                nonce_request_digest=_authorization_validation[-1] if _authorization_validation else None)
            self._verify_provider_checkpoint(receipt.provider_request,receipt.provider_response,receipt.provider_receipt_digest,now,event)
            self.store.trust_store.require_trusted_at_receipt(self.signer.key_id,now)
            signed=self.signer.sign(receipt,created_at=now)
            directory=self._directory()
            if not directory.exists():
                directory.mkdir(mode=0o700);self.store._fsync_directory(directory);self.store._fsync_directory(self.store.root)
            self.store._atomic_create_bytes(directory/f'{len(rows):020d}-{envelope.payload_digest}.json',canonical_bytes(signed))
            return receipt


    def reserve_proposal(self,envelope,*,intent,intents,risk_envelope,risk_trust_store,
            authorization_envelope,authorization_journal,expected_scope,capital_cents,expected_head,_packet_validation=None,_fence_validation=None,_artifact_validation=None):
        """Offline integrated ingress; verify signed risk bytes and reserve nonce.

        Expected model/candidate/build/policy digests remain externally resolved
        inputs. This does not authenticate capital artifacts, genuine holdouts,
        live account/quotes or provider semantics. No readiness is issued.
        Direct append is the signed fixture transition primitive, not production
        admission. The disabled gateway exposes neither primitive to callers.
        """
        from src.axiom2.contracts.equity_orders import EquityOrderIntent
        from src.axiom2.portfolio.intents import verify_risk_intent_receipt
        from src.axiom2.execution.journal import ExecutionJournal
        from src.axiom2.execution.authorization import AuthorizationJournal
        if type(authorization_journal) is not AuthorizationJournal:
            raise TypeError('exact offline authorization journal required')
        if type(intents) is not tuple or not intents or type(intent) is not EquityOrderIntent:
            raise ValueError('exact immutable risk intents required')
        payloads=[]
        for order in intents:
            if type(order) is not EquityOrderIntent:raise ValueError('exact risk intent required')
            payload=json.loads(canonical_bytes(asdict(order)))
            ExecutionJournal._validate_intent(payload);payloads.append(payload)
        chosen=json.loads(canonical_bytes(asdict(intent)));ExecutionJournal._validate_intent(chosen)
        if len({order.intent_id for order in intents})!=len(intents) or payloads.count(chosen)!=1:
            raise ValueError('selected intent must occur once in signed batch')
        now=self.store._trusted_clock()
        if now>=intent.expires_at:raise ValueError('risk intent expired')
        risk_envelope=SignedEnvelope.model_validate_json(canonical_bytes(risk_envelope),strict=True)
        risk=verify_risk_intent_receipt(risk_envelope,intents,trust_store=risk_trust_store)
        risk_trust_store.require_trusted_at_receipt(risk_envelope.signature.key_id,now)
        if not 0<=(now-risk_envelope.signature.created_at).total_seconds()<=30:
            raise ValueError('risk receipt stale or future')
        for order in intents:
            if (risk.decision_digest,risk.account_alias,risk.account_revision,risk.capability_digest)!=(
                    order.decision_digest,order.account_alias,order.account_revision,order.capability_digest):
                raise ValueError('risk receipt scope mismatch')
        digest=content_digest(chosen)
        if expected_scope.get('intent_digest')!=digest or expected_scope.get('account_alias')!=intent.account_alias or expected_scope.get('capability_digest')!=intent.capability_digest:
            raise ValueError('operator scope does not bind resolved intent')
        if type(capital_cents) is not int or capital_cents<(intent.quantity*intent.limit_price_micros+9999)//10000:
            raise ValueError('offline capital bound below principal')
        envelope=SignedEnvelope.model_validate_json(canonical_bytes(envelope),strict=True)
        event=self._verify_event(envelope,now)
        authorization_envelope=SignedEnvelope.model_validate_json(canonical_bytes(authorization_envelope),strict=True)
        if event.state!='PROPOSED' or event.filled_quantity or event.order_identity!=intent.intent_id or event.intent_digest!=digest or event.authorization_digest!=authorization_envelope.payload_digest or event.maximum_quantity!=intent.quantity or event.account_alias!=intent.account_alias:
            raise ValueError('proposal does not bind exact signed inputs')
        nonce=authorization_journal.reserve(authorization_envelope,expected=expected_scope,
            quantity=intent.quantity,capital_cents=capital_cents)
        # Persisted nonce precedes lifecycle progression. Crash recovery replays
        # the same reservation; CAS/unique scope forbid a second order identity.
        return self.append(envelope,expected_head=expected_head,_risk_validation=(risk_envelope,intents,risk_trust_store),
            _authorization_validation=(authorization_envelope,authorization_journal.operator_trust_store,
                dict(expected_scope),intent.quantity,capital_cents,nonce.request_digest),_packet_validation=_packet_validation,_fence_validation=_fence_validation,_artifact_validation=_artifact_validation)


    def reserve_reconstructed_proposal(self,envelope,*,risk_packet_envelope,risk_actor_bindings,
            account,capabilities,**arguments):
        """Complete signed risk reconstruction for offline proposal admission.

        Account/capability observations and artifact identities still require
        production service resolution. This grants no capital or broker access.
        """
        from src.axiom2.execution.resolution import verify_reconstructed_intents
        packet_arguments=dict(intents=arguments['intents'],account=account,capabilities=capabilities,
            risk_trust_store=arguments['risk_trust_store'],actor_bindings=dict(risk_actor_bindings))
        packet=SignedEnvelope.model_validate_json(canonical_bytes(risk_packet_envelope),strict=True)
        _,request,decision=verify_reconstructed_intents(packet,now=self.store._trusted_clock(),**packet_arguments)
        scope=arguments['expected_scope']
        if scope.get('risk_policy_digest')!=decision.policy_digest or scope.get('model_digest')!=request.model_artifact_hash:
            raise ValueError('operator scope differs from reconstructed policy/model')
        if packet.signature.key_id!=arguments['risk_envelope'].signature.key_id:
            raise ValueError('risk packet and intent receipt actor mismatch')
        return self.reserve_proposal(envelope,**arguments,_packet_validation=(packet,packet_arguments))


    def reserve_fenced_proposal(self,envelope,*,fence,fence_token,**arguments):
        """Persist an offline epoch reservation and recheck before proposal commit.

        This is single shared-filesystem rehearsal, not provider fencing proof.
        No transport or live action exists in this lifecycle journal.
        """
        from src.axiom2.execution.fencing import OfflineExecutionFence
        if type(fence) is not OfflineExecutionFence or fence.journal.store is not self.store:
            raise TypeError('fence must share exact lifecycle EvidenceStore')
        reservation=fence.reserve_effect(fence_token,intent_digest=content_digest(asdict(arguments['intent'])))
        return self.reserve_reconstructed_proposal(envelope,**arguments,_fence_validation=(fence,fence_token,reservation))


    def reserve_resolved_proposal(self,envelope,*,artifact_resolver,artifact_pointer_envelope,**arguments):
        """Offline admission with actual artifact bytes, frozen session and fencing.

        Resolution authenticates metadata and bytes; external source truth,
        service isolation, genuine holdout and broker qualification remain closed.
        """
        from src.axiom2.execution.artifacts import ArtifactResolver,SessionArtifactPointer
        from src.axiom2.execution.resolution import verify_reconstructed_intents
        if type(artifact_resolver) is not ArtifactResolver or artifact_resolver.store is not self.store:
            raise TypeError('artifact resolver must share exact lifecycle EvidenceStore')
        pointer=SignedEnvelope.model_validate_json(canonical_bytes(artifact_pointer_envelope),strict=True)
        resolution=artifact_resolver.resolve(pointer)
        _,request,_=verify_reconstructed_intents(arguments['risk_packet_envelope'],
            intents=arguments['intents'],account=arguments['account'],capabilities=arguments['capabilities'],
            risk_trust_store=arguments['risk_trust_store'],actor_bindings=arguments['risk_actor_bindings'],now=self.store._trusted_clock())
        expected=dict(model_artifact_hash=resolution.model_digest,promotion_decision_hash=resolution.promotion_decision_hash,
            economic_evidence_digest=resolution.calibration_payload_digest,campaign_portfolio_digest=resolution.campaign_portfolio_digest)
        if any(getattr(request,name)!=value for name,value in expected.items()):raise ValueError('risk request differs from resolved artifact bytes')
        if any(candidate.expected_net_edge_bps!=resolution.expected_net_edge_bps for candidate in request.candidates):raise ValueError('risk edge differs from resolved calibration')
        session=SessionArtifactPointer.from_versioned_payload(pointer.payload)
        if session.account_alias!=arguments['intent'].account_alias:raise ValueError('artifact session account mismatch')
        if request.source_kind!=resolution.source_kind or (request.session_open,request.session_close)!=(session.session_open,session.session_close):raise ValueError('artifact classification/session differs from risk request')
        scope=arguments['expected_scope']
        for name,value in dict(candidate_digest=resolution.candidate_digest,model_digest=resolution.model_digest,
                risk_policy_digest=resolution.risk_policy_digest,build_digest=resolution.build_digest).items():
            if scope.get(name)!=value:raise ValueError('operator scope differs from resolved artifacts')
        return self.reserve_fenced_proposal(envelope,**arguments,_artifact_validation=(artifact_resolver,pointer,resolution))
