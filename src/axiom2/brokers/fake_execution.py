"""Private persistent synthetic provider contract; no broker/network bindings.

All effects are fake and mode OFFLINE_PREPARATION. Native identities and declared
wire shapes are exercised, never promoted to provider truth or live readiness.
One EvidenceStore lock/fsync authenticates this stream and its authority history.
"""
from datetime import datetime
import json
from typing import Literal
from uuid import UUID,uuid5
from pydantic import Field
from src.evidence.contracts import StrictContract
from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import content_digest
from src.evidence.signing import verify_envelope
from src.evidence.store import StoreCorruptionError
from src.axiom2.execution.journal import ExecutionJournal
from src.axiom2.brokers.robinhood_readonly import Wire,Order,_units

STREAM='shadow-provider-rehearsal'
NAMESPACE=UUID('d1cf6a75-3b46-4b41-8ecf-e3d3c2456a5b')

class ReviewRequest(Wire):
    account_number:str;symbol:str;side:Literal['buy','sell'];type:Literal['limit']
    quantity:str;limit_price:str;time_in_force:Literal['gfd'];market_hours:Literal['regular_hours']
class PlaceRequest(ReviewRequest):
    ref_id:str
class FillRequest(Wire):
    order_id:str;quantity:int;price_cents:int;fee_cents:int
class OrderIDRequest(Wire):
    order_id:str
class CancelRequest(Wire):
    account_number:str;order_id:str
class FakeProviderEvent(StrictContract):
    sequence:int=Field(ge=0)
    previous_digest:str|None
    actor_id:str
    occurred_at:datetime
    action:Literal['REVIEW','PLACE','FILL','CANCEL','CANCEL_CONFIRMED']
    request:dict
    response:dict
    mode:Literal['OFFLINE_PREPARATION']='OFFLINE_PREPARATION'
    source_kind:Literal['SYNTHETIC']='SYNTHETIC'
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False


def _request(action,value):
    contract={'REVIEW':ReviewRequest,'PLACE':PlaceRequest,'FILL':FillRequest,'CANCEL':CancelRequest,'CANCEL_CONFIRMED':OrderIDRequest}[action]
    request=contract.model_validate_json(canonical_bytes(value),strict=True)
    if action in ('REVIEW','PLACE','CANCEL') and not request.account_number.startswith('FAKE:'):raise ValueError('FAKE_ACCOUNT_NAMESPACE_REQUIRED')
    if action in ('REVIEW','PLACE'):
        quantity=_units(request.quantity,1)
        if quantity<=0 or _units(request.limit_price,100)<=0 or not request.symbol:raise ValueError('UNSUPPORTED_FAKE_ORDER')
    if action=='PLACE' and str(UUID(request.ref_id))!=request.ref_id:raise ValueError('EXACT_FAKE_REQUEST_UUID_REQUIRED')
    if action=='FILL' and (request.quantity<=0 or request.price_cents<=0 or request.fee_cents<0):raise ValueError('INVALID_FAKE_FILL')
    return json.loads(canonical_bytes(request.model_dump()))

class FakeRobinhoodTransport:
    __slots__=('journal',)
    def __init__(self,journal):
        if type(journal) is not ExecutionJournal:raise TypeError('EXACT_SHARED_JOURNAL_REQUIRED')
        self.journal=journal

    @staticmethod
    def _apply(action,request,orders,reviews,now):
        # Return a fresh declared response; no external callbacks or tools exist.
        if action=='REVIEW':
            reviews.add(content_digest(request))
            price=request['limit_price'];quote=dict(adjusted_previous_close=price,ask_price=price,bid_price=price,has_traded=True,last_non_reg_trade_price=None,last_trade_price=price,previous_close=price,previous_close_date=now.date().isoformat(),state='active',symbol=request['symbol'],venue_ask_time=now.isoformat(),venue_bid_time=now.isoformat(),venue_last_non_reg_trade_time=None,venue_last_trade_time=now.isoformat())
            return dict(data=dict(limit_price=price,quantity=request['quantity'],order_checks={},quote_data=quote,side=request['side'],symbol=request['symbol'],type='limit'),guide='OFFLINE SYNTHETIC CONTRACT ONLY')
        if action=='PLACE':
            review={key:value for key,value in request.items() if key!='ref_id'}
            if content_digest(review) not in reviews:raise ValueError('FAKE_REVIEW_REQUIRED')
            if request['ref_id'] in orders:raise ValueError('FAKE_DUPLICATE_PLACEMENT')
            order=dict(average_price=None,created_at=now.isoformat(),cumulative_quantity='0',dollar_based_amount=None,executions=[],fees='0.00',id=str(uuid5(NAMESPACE,request['ref_id'])),instrument_id='FAKE:'+request['symbol'],last_transaction_at=None,market_hours='regular_hours',placed_agent='agentic',price=request['limit_price'],quantity=request['quantity'],ref_id=request['ref_id'],side=request['side'],state='queued',stop_price=None,symbol=request['symbol'],time_in_force='gfd',trigger='immediate',type='limit')
            Order.model_validate_json(canonical_bytes(order),strict=True)
            orders[request['ref_id']]=dict(request=request,order=order)
            return dict(data=dict(order=order),guide='OFFLINE SYNTHETIC CONTRACT ONLY')
        matches=[entry for entry in orders.values() if entry['order']['id']==request['order_id']]
        if len(matches)!=1:raise ValueError('UNKNOWN_FAKE_ORDER')
        entry=matches[0];order=entry['order']
        if action=='CANCEL':
            if entry['request']['account_number']!=request['account_number'] or order['state']=='filled':raise ValueError('INVALID_FAKE_CANCEL')
            entry['cancel_pending']=True;order['state']='pending_cancelled';order['last_transaction_at']=now.isoformat()
            return dict(data=dict(accepted=True),guide='OFFLINE SYNTHETIC CONTRACT ONLY')
        if action=='CANCEL_CONFIRMED':
            if not entry.get('cancel_pending') or order['state']=='filled':raise ValueError('NO_PENDING_FAKE_CANCEL')
            entry['cancel_pending']=False;entry['cancel_confirmed']=True;order['state']='partially_filled_rest_cancelled' if _units(order['cumulative_quantity'],1) else 'cancelled';order['last_transaction_at']=now.isoformat()
            return dict(data=dict(order=order),guide='OFFLINE SYNTHETIC CONTRACT ONLY')
        if order['state']=='filled':raise ValueError('FAKE_ORDER_TERMINAL')
        cumulative=_units(order['cumulative_quantity'],1)+request['quantity']
        if cumulative>_units(order['quantity'],1):raise ValueError('FAKE_FILL_EXCEEDS_ORDER')
        limit=_units(order['price'],100)
        if order['side']=='buy' and request['price_cents']>limit or order['side']=='sell' and request['price_cents']<limit:raise ValueError('FAKE_FILL_OUTSIDE_LIMIT')
        price=f"{request['price_cents']//100}.{request['price_cents']%100:02d}";fees=f"{request['fee_cents']//100}.{request['fee_cents']%100:02d}"
        fill=dict(id=str(uuid5(NAMESPACE,order['id']+':'+str(len(order['executions'])))),fees=fees,price=price,quantity=str(request['quantity']),timestamp=now.isoformat())
        order['executions'].append(fill);order['cumulative_quantity']=str(cumulative)
        total=sum(_units(item['fees'],100) for item in order['executions']);order['fees']=f'{total//100}.{total%100:02d}'
        # The fake profile requires a single fill price; unsupported averaging
        # semantics fail rather than making up provider-rounded precision.
        prices={item['price'] for item in order['executions']}
        if len(prices)!=1:raise ValueError('FAKE_MIXED_FILL_PRICE_UNSUPPORTED')
        order['average_price']=price;order['state']='filled' if cumulative==_units(order['quantity'],1) else ('pending_cancelled' if entry.get('cancel_pending') else ('partially_filled_rest_cancelled' if entry.get('cancel_confirmed') else 'partially_filled'));order['last_transaction_at']=now.isoformat()
        return dict(data=dict(order=order),guide='OFFLINE SYNTHETIC CONTRACT ONLY')

    def _read(self):
        journal=self.journal;authority,bindings=journal._roles();rows=journal._read(STREAM);orders={};reviews=set();head=None
        for index,(envelope,receipt) in enumerate(rows):
            try:
                event=verify_envelope(envelope,FakeProviderEvent,journal.store.trust_store);journal._timing(envelope,event,receipt,journal.store.trust_store)
                interval=bindings.get((event.actor_id,envelope.signature.key_id))
                if interval is None or min(receipt,event.occurred_at,envelope.signature.created_at)<interval[0] or interval[1] is not None and max(receipt,event.occurred_at,envelope.signature.created_at)>=interval[1]:raise ValueError('fake actor not authorized')
                if event.sequence!=index or event.previous_digest!=head or index and receipt<rows[index-1][1]:raise ValueError('fake chain/time')
                request=_request(event.action,event.request)
                response=self._apply(event.action,request,orders,reviews,receipt)
                if canonical_bytes(response)!=canonical_bytes(event.response):raise ValueError('fake response does not reconstruct')
                head=envelope.payload_digest
            except (ValueError,TypeError,KeyError) as exc:raise StoreCorruptionError('invalid signed fake provider history') from exc
        return rows,orders,reviews,head,authority,bindings

    def _commit(self,action,request,*,service=None):
        request=_request(action,request);journal=self.journal
        with journal.store._locked():
            rows,orders,reviews,head,authority,bindings=self._read();now=journal.store._trusted_clock()
            if action=='PLACE' and request['ref_id'] in orders:
                entry=orders[request['ref_id']]
                if canonical_bytes(entry['request'])!=canonical_bytes(request):raise ValueError('FAKE_REQUEST_ID_REUSED_WITH_CHANGED_BYTES')
                # Read-only replay of the original placement, never a new effect.
                return next(envelope for envelope,_ in rows if envelope.payload['action']=='PLACE' and envelope.payload['request']['ref_id']==request['ref_id'])
            if action=='CANCEL':
                old=next((envelope for envelope,_ in rows if envelope.payload['action']=='CANCEL' and canonical_bytes(envelope.payload['request'])==canonical_bytes(request)),None)
                if old is not None:return old
            if action=='REVIEW':
                old=next((envelope for envelope,_ in rows if envelope.payload['action']=='REVIEW' and canonical_bytes(envelope.payload['request'])==canonical_bytes(request)),None)
                if old is not None:return old
            if service is not None:
                from src.axiom2.execution.service import OfflineExecutionService
                if type(service) is not OfflineExecutionService or service.transport is not self:raise TypeError('EXACT_FAKE_SERVICE_REQUIRED')
                now=service._validate_unlocked(now,action,request)
            if rows and now<rows[-1][1] or authority and now<authority[-1][1]:raise ValueError('FAKE_CLOCK_REGRESSED')
            interval=bindings.get((journal.actor_id,journal.signer.key_id))
            if interval is None or now<interval[0] or interval[1] is not None:raise ValueError('FAKE_ACTOR_RETIRED_OR_UNREGISTERED')
            journal.store.trust_store.require_trusted_at_receipt(journal.signer.key_id,now)
            response=self._apply(action,request,orders,reviews,now)
            event=FakeProviderEvent(sequence=len(rows),previous_digest=head,actor_id=journal.actor_id,occurred_at=now,action=action,request=request,response=response)
            envelope=journal.signer.sign(event,created_at=now);journal._write(STREAM,envelope,now,len(rows));return envelope

    def fill(self,order_id,*,quantity,price_cents,fee_cents):
        return self._commit('FILL',dict(order_id=order_id,quantity=quantity,price_cents=price_cents,fee_cents=fee_cents))
    def cancel(self,account_alias,order_id):
        return self._commit('CANCEL',dict(account_number='FAKE:'+account_alias,order_id=order_id))
    def confirm_cancel(self,order_id):
        return self._commit('CANCEL_CONFIRMED',dict(order_id=order_id))
    def orders(self):
        with self.journal.store._locked():
            _,orders,_,_,_,_=self._read();return tuple(json.loads(canonical_bytes(value['order'])) for value in orders.values())
