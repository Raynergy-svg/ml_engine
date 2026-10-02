"""Fake-only offline service composition; public ExecutionAuthority unchanged.

No real provider transport can be injected. This service owns signed rehearsal
roles, not production authority. Independent source truth, deployment isolation,
calendar qualification, OAuth and genuine holdout remain outside this scope.
"""
from dataclasses import asdict
from datetime import datetime
import json
from types import MappingProxyType
from uuid import uuid5
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import SignedEnvelope
from src.evidence.hashing import content_digest
from src.evidence.signing import verify_envelope
from src.axiom2.execution.lifecycle import ExecutionLifecycle,LifecycleEvent
from src.axiom2.execution.artifacts import ArtifactResolver,ArtifactResolution,SessionArtifactPointer
from src.axiom2.execution.authorization import AuthorizationJournal,AuthorizationReservation,SCOPE_FIELDS,verify_authorization
from src.axiom2.execution.fencing import OfflineExecutionFence,FenceReservation
from src.axiom2.execution.resolution import verify_reconstructed_intents
from src.axiom2.execution.reconciliation import Ledger,Observations,Account,Policy,Fill,reconcile
from src.axiom2.brokers.fake_execution import FakeRobinhoodTransport,FakeProviderEvent,NAMESPACE
from src.axiom2.brokers.robinhood_readonly import Order,_units

class OfflineExecutionService:
    __slots__=('kernel','transport','gateway','observer','admission','symbols')
    def __init__(self,kernel,transport,*,gateway_signer,observer_signer,admission,instrument_symbols):
        if type(kernel) is not ExecutionLifecycle or type(transport) is not FakeRobinhoodTransport or transport.journal.store is not kernel.store:raise TypeError('EXACT_SHARED_STORE_FAKE_COMPOSITION_REQUIRED')
        if type(admission) is not dict or type(admission.get('artifact_resolver')) is not ArtifactResolver or type(admission.get('authorization_journal')) is not AuthorizationJournal or type(admission.get('fence')) is not OfflineExecutionFence:raise TypeError('EXACT_OFFLINE_ADMISSION_SERVICES_REQUIRED')
        if any(service.store is not kernel.store for service in (admission['artifact_resolver'],admission['authorization_journal'])) or admission['fence'].journal.store is not kernel.store:raise ValueError('ADMISSION_STORES_MUST_MATCH')
        pointer=SessionArtifactPointer.from_versioned_payload(admission['artifact_pointer_envelope'].payload)
        if pointer.source_kind not in ('SYNTHETIC','DEVELOPMENT'):raise ValueError('FAKE_SERVICE_REJECTS_LIVE_CLASSIFICATION')
        intent=admission['intent']
        if type(instrument_symbols) is not dict or set(instrument_symbols)!={intent.instrument_id} or any(type(value) is not str or not value for value in instrument_symbols.values()):raise ValueError('EXACT_OFFLINE_INSTRUMENT_MAPPING_REQUIRED')
        if intent.quantity is None or intent.limit_price_micros%10000:raise ValueError('FAKE_PROFILE_REQUIRES_WHOLE_SHARES_CENT_LIMIT')
        for role,signer in (('GATEWAY',gateway_signer),('OBSERVER',observer_signer)):
            if (role,signer.key_id) not in kernel.actors:raise ValueError('SERVICE_SIGNER_ROLE_NOT_CONFIGURED')
        if kernel.provider_journal is not None and kernel.provider_journal is not transport.journal:raise ValueError('PROVIDER_JOURNAL_BINDING_CHANGED')
        kernel.provider_journal=transport.journal
        self.kernel=kernel;self.transport=transport;self.gateway=gateway_signer;self.observer=observer_signer
        copied=dict(admission);copied['expected_scope']=dict(admission['expected_scope']);copied['risk_actor_bindings']=dict(admission['risk_actor_bindings'])
        self.admission=MappingProxyType(copied);self.symbols=MappingProxyType(dict(instrument_symbols))

    def _current(self):
        rows=self.kernel.replay();matching=[row for row in rows if row.event.order_identity==self.admission['intent'].intent_id]
        if not matching:raise ValueError('DURABLE_ADMISSION_REQUIRED')
        return matching[0],matching[-1]

    def _validate_unlocked(self,now,action,request):
        args=self.admission
        if action in ('REVIEW','PLACE') and canonical_bytes(request)!=canonical_bytes(self._request(action=='PLACE')):raise ValueError('FAKE_EFFECT_DIFFERS_FROM_ADMITTED_INTENT')
        if action=='CANCEL':
            _,orders,_,_,_,_=self.transport._read();entry=orders.get(self._request(True)['ref_id'])
            if entry is None or request!={'account_number':'FAKE:'+args['intent'].account_alias,'order_id':entry['order']['id']}:raise ValueError('FAKE_CANCEL_SCOPE_MISMATCH')
        rows,_=self.kernel._read()
        proposed=next((row for row in rows if row.event.order_identity==args['intent'].intent_id),None)
        if proposed is None or proposed.event.state!='PROPOSED' or proposed.artifact_pointer_envelope is None:raise ValueError('RESOLVED_DURABLE_ADMISSION_REQUIRED')
        resolution=ArtifactResolution.model_validate_json(canonical_bytes(proposed.artifact_resolution),strict=True)
        reservation=FenceReservation.model_validate_json(canonical_bytes(proposed.fence_reservation),strict=True)
        declaration=verify_authorization(proposed.authorization_envelope,args['authorization_journal'].operator_trust_store,
            expected=args['expected_scope'],now=now,quantity=args['intent'].quantity,capital_cents=args['capital_cents'])
        identity=content_digest(dict(operator_key=proposed.authorization_envelope.signature.key_id,nonce=declaration.nonce))
        path=self.kernel.store.root/'offline-authorizations'/(identity+'.json')
        if path.is_symlink() or not path.is_file():raise ValueError('DURABLE_NONCE_RESERVATION_MISSING')
        raw=path.read_bytes();signed=SignedEnvelope.model_validate_json(raw,strict=True)
        if canonical_bytes(signed)!=raw:raise ValueError('NONCANONICAL_NONCE_RESERVATION')
        nonce=verify_envelope(signed,AuthorizationReservation,self.kernel.store.trust_store)
        self.kernel.store.trust_store.require_trusted_at_receipt(signed.signature.key_id,nonce.received_at)
        request_digest=content_digest(dict(scope=args['expected_scope'],quantity=args['intent'].quantity,capital_cents=args['capital_cents'],authorization_digest=proposed.authorization_envelope.payload_digest))
        if nonce.authorization!=proposed.authorization_envelope or nonce.received_at!=signed.signature.created_at or nonce.request_digest!=request_digest or proposed.nonce_request_digest!=request_digest:raise ValueError('DURABLE_NONCE_SCOPE_CHANGED')
        _,now=args['artifact_resolver']._verify_unlocked(proposed.artifact_pointer_envelope,resolution,now,return_time=True)
        args['fence']._verify_unlocked(args['fence_token'],reservation,now)
        verify_reconstructed_intents(proposed.risk_packet_envelope,intents=args['intents'],account=args['account'],capabilities=args['capabilities'],risk_trust_store=args['risk_trust_store'],actor_bindings=args['risk_actor_bindings'],now=now)
        verify_authorization(proposed.authorization_envelope,args['authorization_journal'].operator_trust_store,expected=args['expected_scope'],now=now,quantity=args['intent'].quantity,capital_cents=args['capital_cents'])
        return now

    def admit(self,envelope):return self.kernel.reserve_resolved_proposal(envelope,**dict(self.admission))

    def _request(self,with_ref):
        intent=self.admission['intent'];price=intent.limit_price_micros
        request=dict(account_number='FAKE:'+intent.account_alias,symbol=self.symbols[intent.instrument_id],side=intent.side.lower(),type='limit',quantity=str(intent.quantity),limit_price=f'{price//10**6}.{price%10**6:06d}',market_hours='regular_hours',time_in_force='gfd')
        if with_ref:request['ref_id']=str(uuid5(NAMESPACE,intent.account_alias+':'+content_digest(asdict(intent))))
        return request

    def _advance(self,state,*,filled=0,evidence=None,guard=False):
        proposed,current=self._current();now=self.kernel.store._trusted_clock()
        signer=self.observer if state in ('ACKNOWLEDGED','PARTIALLY_FILLED','FILLED','CANCELED','REJECTED') else self.gateway
        role='OBSERVER' if signer is self.observer else 'GATEWAY'
        event=current.event.model_copy(update=dict(sequence=len(self.kernel.replay()),previous_digest=self.kernel.head,state=state,filled_quantity=filled,
            occurred_at=now,role=role,actor_id=self.kernel.actors[(role,signer.key_id)]))
        validations={}
        if guard:
            args=self.admission;validations=dict(_artifact_validation=(args['artifact_resolver'],proposed.artifact_pointer_envelope,ArtifactResolution.model_validate_json(canonical_bytes(proposed.artifact_resolution),strict=True)),
                _fence_validation=(args['fence'],args['fence_token'],FenceReservation.model_validate_json(canonical_bytes(proposed.fence_reservation),strict=True)),
                _packet_validation=(proposed.risk_packet_envelope,dict(intents=args['intents'],account=args['account'],capabilities=args['capabilities'],risk_trust_store=args['risk_trust_store'],actor_bindings=args['risk_actor_bindings'])),
                _authorization_validation=(proposed.authorization_envelope,args['authorization_journal'].operator_trust_store,args['expected_scope'],args['intent'].quantity,args['capital_cents'],proposed.nonce_request_digest))
        proof=None
        if evidence is not None:
            value=FakeProviderEvent.from_versioned_payload(evidence.payload);proof=(dict(value.request),dict(value.response),evidence.payload_digest)
        return self.kernel.append(signer.sign(event,created_at=now),expected_head=current.event_envelope.payload_digest,_provider_evidence=proof,**validations)

    def review(self,intent_id):
        if intent_id!=self.admission['intent'].intent_id:raise ValueError('UNKNOWN_SERVICE_INTENT')
        _,current=self._current()
        if current.event.state=='REVIEWED':return current
        if current.event.state!='PROPOSED':raise ValueError('REVIEW_REQUIRES_PROPOSED')
        result=self.transport._commit('REVIEW',self._request(False),service=self)
        return self._advance('REVIEWED',evidence=result,guard=True)

    def submit_fake(self,intent_id):
        if intent_id!=self.admission['intent'].intent_id:raise ValueError('UNKNOWN_SERVICE_INTENT')
        _,current=self._current()
        if current.event.state=='REVIEWED':self._advance('SUBMISSION_RESERVED',guard=True)
        elif current.event.state in ('SUBMISSION_RESERVED','SUBMISSION_UNKNOWN','ACKNOWLEDGED','PARTIALLY_FILLED','FILLED','CANCELED'):
            # Restart reads provider rehearsal truth; it never blindly places.
            return self.observe(intent_id)
        else:raise ValueError('FAKE_SUBMISSION_REQUIRES_REVIEW')
        try:result=self.transport._commit('PLACE',self._request(True),service=self)
        except Exception:
            self._advance('SUBMISSION_UNKNOWN');raise
        return self._advance('ACKNOWLEDGED',evidence=result)

    def observe(self,intent_id):
        if intent_id!=self.admission['intent'].intent_id:raise ValueError('UNKNOWN_SERVICE_INTENT')
        _,current=self._current();ref=self._request(True)['ref_id']
        with self.kernel.store._locked():
            rows,orders,_,_,_,_=self.transport._read()
            entry=orders.get(ref)
            if entry is None:return current
            if canonical_bytes(entry['request'])!=canonical_bytes(self._request(True)):raise ValueError('FAKE_REQUEST_BINDING_CHANGED')
            evidence=next(envelope for envelope,_ in reversed(rows) if envelope.payload['action'] in ('PLACE','FILL','CANCEL','CANCEL_CONFIRMED') and (envelope.payload['request'].get('ref_id')==ref or envelope.payload['request'].get('order_id')==entry['order']['id']))
            order=json.loads(canonical_bytes(entry['order']))
        if current.event.state in ('SUBMISSION_RESERVED','SUBMISSION_UNKNOWN'):
            current=self._advance('ACKNOWLEDGED',evidence=evidence)
        filled=_units(order['cumulative_quantity'],1)
        state={'queued':'ACKNOWLEDGED','partially_filled':'PARTIALLY_FILLED','filled':'FILLED','cancelled':'CANCELED','partially_filled_rest_cancelled':'CANCELED','pending_cancelled':'CANCEL_REQUESTED'}.get(order['state'])
        if state is None:raise ValueError('UNSUPPORTED_FAKE_ORDER_STATE')
        if current.event.state=='CANCEL_REQUESTED' and state in ('ACKNOWLEDGED','PARTIALLY_FILLED','CANCEL_REQUESTED'):
            if filled==current.event.filled_quantity:return current
            if filled>current.event.filled_quantity:
                self._advance('PARTIALLY_FILLED',filled=filled,evidence=evidence)
                return self._advance('CANCEL_REQUESTED',filled=filled,evidence=evidence)
        if current.event.state==state and current.event.filled_quantity==filled:return current
        return self._advance(state,filled=filled,evidence=evidence)

    def cancel_fake(self,intent_id):
        if intent_id!=self.admission['intent'].intent_id:raise ValueError('UNKNOWN_SERVICE_INTENT')
        _,current=self._current()
        if current.event.state=='CANCEL_REQUESTED':
            current=self.observe(intent_id)
            if current.event.state in ('FILLED','CANCELED'):return current
        elif current.event.state in ('ACKNOWLEDGED','PARTIALLY_FILLED'):
            self._advance('CANCEL_REQUESTED',filled=current.event.filled_quantity,guard=True)
        else:raise ValueError('FAKE_CANCEL_REQUIRES_ACKNOWLEDGEMENT')
        order=next(order for order in self.transport.orders() if order['ref_id']==self._request(True)['ref_id'])
        self.transport._commit('CANCEL',dict(account_number='FAKE:'+self.admission['intent'].account_alias,order_id=order['id']),service=self)
        return self._current()[1]

    def reconcile_hypothetical(self):
        args=self.admission;intent=args['intent'];proposed,current=self._current()
        if current.event.state not in ('FILLED','CANCELED'):raise ValueError('UNRESOLVED_FAKE_ORDER')
        with self.kernel.store._locked():
            rows,orders,_,_,_,_=self.transport._read()
            evidence={envelope.payload_digest:FakeProviderEvent.from_versioned_payload(envelope.payload) for envelope,_ in rows}
            if current.provider_receipt_digest not in evidence:raise ValueError('SIGNED_FAKE_PROVIDER_CHECKPOINT_MISSING')
            expected_event=evidence[current.provider_receipt_digest]
            if canonical_bytes(current.provider_request)!=canonical_bytes(expected_event.request) or canonical_bytes(current.provider_response)!=canonical_bytes(expected_event.response):raise ValueError('LIFECYCLE_PROVIDER_CHECKPOINT_CHANGED')
            entry=orders.get(self._request(True)['ref_id'])
            if entry is None or len(orders)!=1:raise ValueError('UNRESOLVED_OR_UNJOURNALED_FAKE_ORDER')
            order=Order.model_validate_json(canonical_bytes(entry['order']),strict=True)
            expected_order=Order.model_validate_json(canonical_bytes(expected_event.response['data']['order']),strict=True) if 'order' in expected_event.response['data'] else None
            if expected_order is None:raise ValueError('FAKE_CANCEL_REQUIRES_FILL_CHECKPOINT')
            if current.event.filled_quantity!=_units(expected_order.cumulative_quantity,1):raise ValueError('LIFECYCLE_FILL_CHECKPOINT_MISMATCH')
        def fills(record):
            return tuple(Fill(fill.id,record.id,intent.instrument_id,_units(fill.quantity,1)*(1 if intent.side=='BUY' else -1),_units(fill.price,100),_units(fill.fees,100)) for fill in record.executions or ())
        expected=fills(expected_order);observed=fills(order);positions=dict(args['account'].positions);cash=args['account'].settled_cash_cents
        for fill in observed:positions[fill.instrument]=positions.get(fill.instrument,0)+fill.quantity;cash-=fill.quantity*fill.price_cents+fill.fee_cents
        # Explicit synthetic marks only. Unsupported pre-existing instruments
        # remain blocked instead of being silently omitted or valued at zero.
        if set(positions)-{intent.instrument_id}:raise ValueError('FAKE_VALUATION_COVERAGE_UNAVAILABLE')
        marks=tuple((name,intent.limit_price_micros) for name in positions)
        account=Account(intent.account_alias,cash,cash,tuple(positions.items()),True,cash+sum(quantity*intent.limit_price_micros for quantity in positions.values())//10000,0,'hypothetical',marks)
        ledger=Ledger(intent.account_alias,'hypothetical',args['account'].settled_cash_cents,args['account'].positions,(order.id,),expected,args['account'].settled_cash_cents,0)
        observation=Observations(intent.account_alias,'hypothetical',True,observed)
        return reconcile(ledger,observation,account,Policy(args['expected_scope']['risk_policy_digest'],0,0))
