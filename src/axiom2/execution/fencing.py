"""Durable offline lease epochs on a single shared POSIX EvidenceStore.

This records hypothetical effect reservations only. It does not fence a broker,
prove cross-host filesystem locks, authenticate a deployment node, or prevent
whole-store rollback; those require operational/provider evidence. Owner labels
are signed shadow-service metadata. The public execution gateway stays disabled.
"""
from datetime import datetime,timedelta
from typing import Literal
from pydantic import Field
from src.evidence.contracts import StrictContract
from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import content_digest
from src.evidence.signing import verify_envelope
from src.evidence.store import StoreCorruptionError
from src.axiom2.execution.journal import ExecutionJournal

class FenceToken(StrictContract):
    realm:str=Field(min_length=1,max_length=128)
    owner:str=Field(min_length=1,max_length=128)
    epoch:int=Field(gt=0)
    token_digest:str=Field(pattern=r'^[0-9a-f]{64}$')
    expires_at:datetime
    mode:Literal['OFFLINE_PREPARATION']='OFFLINE_PREPARATION'
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False

class FenceEvent(StrictContract):
    sequence:int=Field(ge=0)
    previous_digest:str|None
    actor_id:str
    occurred_at:datetime
    action:Literal['ACQUIRE','RENEW','RELEASE','RESERVE']
    realm:str=Field(min_length=1,max_length=128)
    owner:str=Field(min_length=1,max_length=128)
    epoch:int=Field(gt=0)
    token:FenceToken|None=None
    expires_at:datetime|None=None
    intent_digest:str|None=Field(default=None,pattern=r'^[0-9a-f]{64}$')
    mode:Literal['OFFLINE_PREPARATION']='OFFLINE_PREPARATION'
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False

class FenceReservation(StrictContract):
    realm:str
    owner:str
    epoch:int
    intent_digest:str
    record_digest:str
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False

class OfflineExecutionFence:
    def __init__(self,journal,*,realm):
        if type(journal) is not ExecutionJournal:raise TypeError('exact durable shadow journal required')
        if type(realm) is not str or not 1<=len(realm)<=128:raise ValueError('explicit bounded fence realm required')
        self.journal=journal;self.realm=realm

    @staticmethod
    def _require(token,state,now):
        if type(token) is not FenceToken or state is None or state['lease'] is None or canonical_bytes(token)!=canonical_bytes(state['lease']) or now>=token.expires_at:
            raise ValueError('expired, released or superseded fence token')

    @staticmethod
    def _reduce(event,digest,states,receipt):
        state=states.setdefault(event.realm,dict(epoch=0,lease=None,effects={}))
        if event.action=='ACQUIRE':
            if event.token is not None or event.intent_digest is not None or event.expires_at is None or not 0<(event.expires_at-receipt).total_seconds()<=30 or event.epoch!=state['epoch']+1:
                raise ValueError('invalid lease acquisition')
            if state['lease'] is not None and receipt<state['lease'].expires_at:raise ValueError('another active lease owns realm')
            state['epoch']=event.epoch;state['lease']=FenceToken(realm=event.realm,owner=event.owner,epoch=event.epoch,expires_at=event.expires_at,token_digest=digest)
            return state['lease']
        OfflineExecutionFence._require(event.token,state,receipt)
        if (event.realm,event.owner,event.epoch)!=(event.token.realm,event.token.owner,event.token.epoch):raise ValueError('fence event scope mismatch')
        if event.action=='RENEW':
            if event.intent_digest is not None or event.expires_at is None or not 0<(event.expires_at-receipt).total_seconds()<=30 or event.expires_at<state['lease'].expires_at:raise ValueError('invalid lease renewal')
            state['lease']=state['lease'].model_copy(update={'expires_at':event.expires_at});return state['lease']
        if event.expires_at is not None:raise ValueError('unexpected fence expiry field')
        if event.action=='RELEASE':
            if event.intent_digest is not None:raise ValueError('release cannot reserve effect')
            state['lease']=None;return None
        if event.action!='RESERVE' or event.intent_digest is None or event.intent_digest in state['effects']:raise ValueError('duplicate or missing fenced effect')
        reservation=FenceReservation(realm=event.realm,owner=event.owner,epoch=event.epoch,intent_digest=event.intent_digest,record_digest=digest)
        state['effects'][event.intent_digest]=reservation;return reservation

    def _read(self):
        journal=self.journal;authority_rows,bindings=journal._roles();rows=journal._read('shadow-fences');states={};head=None
        for index,(envelope,receipt) in enumerate(rows):
            try:
                event=verify_envelope(envelope,FenceEvent,journal.store.trust_store);journal._timing(envelope,event,receipt,journal.store.trust_store)
                interval=bindings.get((event.actor_id,envelope.signature.key_id))
                if interval is None or min(receipt,event.occurred_at,envelope.signature.created_at)<interval[0] or (interval[1] is not None and max(receipt,event.occurred_at,envelope.signature.created_at)>=interval[1]):raise ValueError('fence actor not authorized')
                if event.sequence!=index or event.previous_digest!=head or (index and receipt<rows[index-1][1]):raise ValueError('fence chain/time')
                self._reduce(event,envelope.payload_digest,states,receipt);head=envelope.payload_digest
            except Exception as exc:raise StoreCorruptionError('invalid signed offline fence history') from exc
        return rows,states,head,authority_rows,bindings

    def _append(self,action,*,owner=None,token=None,ttl_seconds=None,intent_digest=None):
        journal=self.journal
        with journal.store._locked():
            rows,states,head,authority_rows,bindings=self._read();now=journal.store._trusted_clock()
            if (rows and now<rows[-1][1]) or (authority_rows and now<authority_rows[-1][1]):raise ValueError('fence clock regressed')
            state=states.get(self.realm)
            if action!='ACQUIRE':
                self._require(token,state,now)
                if token.realm!=self.realm:raise ValueError('fence realm mismatch')
                owner=token.owner
                if action=='RESERVE' and intent_digest in state['effects']:
                    prior=state['effects'][intent_digest]
                    if (prior.owner,prior.epoch)!=(token.owner,token.epoch):raise ValueError('effect already reserved by another epoch')
                    return prior
            interval=bindings.get((journal.actor_id,journal.signer.key_id))
            if interval is None or now<interval[0] or interval[1] is not None:raise ValueError('fence actor retired/unregistered')
            journal.store.trust_store.require_trusted_at_receipt(journal.signer.key_id,now)
            expiry=None
            if action in ('ACQUIRE','RENEW'):
                if type(ttl_seconds) is not int or not 1<=ttl_seconds<=30:raise ValueError('bounded integer lease TTL required')
                expiry=now+timedelta(seconds=ttl_seconds)
            event=FenceEvent(sequence=len(rows),previous_digest=head,actor_id=journal.actor_id,occurred_at=now,action=action,realm=self.realm,owner=owner,epoch=(state['epoch'] if state else 0)+1 if action=='ACQUIRE' else token.epoch,token=token,expires_at=expiry,intent_digest=intent_digest)
            signed=journal.signer.sign(event,created_at=now);result=self._reduce(event,signed.payload_digest,states,now)
            journal._write('shadow-fences',signed,now,len(rows));return result

    def acquire(self,owner,*,ttl_seconds):return self._append('ACQUIRE',owner=owner,ttl_seconds=ttl_seconds)
    def renew(self,token,*,ttl_seconds):return self._append('RENEW',token=token,ttl_seconds=ttl_seconds)
    def release(self,token):return self._append('RELEASE',token=token)
    def reserve_effect(self,token,*,intent_digest):return self._append('RESERVE',token=token,intent_digest=intent_digest)

    def _verify_unlocked(self,token,reservation,now):
        """Final lease check; caller holds the same EvidenceStore lock."""
        rows,states,_,authority_rows,bindings=self._read();state=states.get(self.realm)
        if (rows and now<rows[-1][1]) or (authority_rows and now<authority_rows[-1][1]):raise ValueError('final admission precedes fence/authority receipt')
        self._require(token,state,now)
        if token.realm!=self.realm or type(reservation) is not FenceReservation or (reservation.realm,reservation.owner,reservation.epoch)!=(token.realm,token.owner,token.epoch) or state['effects'].get(reservation.intent_digest)!=reservation:
            raise ValueError('durable fenced reservation mismatch')
        interval=bindings.get((self.journal.actor_id,self.journal.signer.key_id))
        if interval is None or now<interval[0] or interval[1] is not None:raise ValueError('fence actor retired/unregistered')
        self.journal.store.trust_store.require_trusted_at_receipt(self.journal.signer.key_id,now)
