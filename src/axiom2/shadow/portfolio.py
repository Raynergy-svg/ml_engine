"""Signed continuous hypothetical portfolio; no live or capital qualification.

Shares and T+1 credits reconstruct from immutable shadow streams. Unsupported
corporate actions/external cash flows require explicit future normalized events;
they cannot be inferred from a price change. A shared filesystem's locking is
not proof of distributed deployment fencing.
"""
from dataclasses import asdict
from datetime import datetime,timedelta
import json
from typing import Literal
from pydantic import Field
from src.evidence.contracts import StrictContract
from src.evidence.signing import verify_envelope
from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import content_digest
from src.evidence.store import StoreCorruptionError
from src.evidence.execution_shadow import ExecutionEvent
from src.axiom2.execution.resolution import _record,_time,_tuple,reconstruct_risk_packet,validate_reconstructed_sources
from src.axiom2.contracts.equity_orders import AccountObservationRef,BrokerCapabilities
from src.axiom2.portfolio.contracts import PortfolioSnapshot,Position,RiskPolicy
from src.axiom2.portfolio.intents import build_order_intents
from src.axiom2.shadow.fills import FillPolicy,MarketSnapshot,Quote,hypothetical_fill

class ShadowPortfolioEvent(StrictContract):
    sequence:int=Field(ge=0)
    previous_digest:str|None
    actor_id:str
    occurred_at:datetime
    kind:Literal['INITIALIZE','OBSERVE','STEP']
    body:dict
    mode:Literal['SHADOW']='SHADOW'
    capital_authorized:Literal[False]=False
    execution_enabled:Literal[False]=False


def _market(body):
    return _record(MarketSnapshot,body,{'as_of':_time,'session_open':_time,'session_close':_time,
        'quotes':lambda rows:_tuple(rows,lambda row:_record(Quote,row,{'quoted_at':_time,'available_at':_time}))})


def _account(body):
    return _record(AccountObservationRef,body,{'observed_at':_time,'expires_at':_time,
        'positions':lambda rows:_tuple(rows,lambda row:_tuple(row))})

class ShadowPortfolioLedger:
    """Same EvidenceStore lock/fsync and authenticated shadow actor history."""
    qualified=False
    def __init__(self,journal):
        from src.axiom2.execution.journal import ExecutionJournal
        if type(journal) is not ExecutionJournal:raise TypeError('exact durable execution journal required')
        self.journal=journal

    @staticmethod
    def _source_time(kind,body,receipt,state=None):
        value=body['market']['as_of'] if kind=='OBSERVE' else body['packet']['request']['as_of']
        stamp=_time(value)
        if not 0<=(receipt-stamp).total_seconds()<=30:
            raise ValueError('portfolio source stale or future at receipt')
        if kind!='OBSERVE':
            policy,request,_=reconstruct_risk_packet(body['packet'])
            validate_reconstructed_sources(policy,request,receipt)
        if kind in ('STEP','OBSERVE'):
            market=_market(body['market'])
            ttl=body['fill_policy']['quote_ttl_seconds'] if kind=='STEP' else state['policy']['quote_ttl_seconds'] if state else 30
            if not market.session_open<=receipt<market.session_close or any(quote.available_at>receipt or not 0<=(receipt-quote.quoted_at).total_seconds()<=ttl for quote in market.quotes):
                raise ValueError('portfolio observation source stale/future at receipt')

    @staticmethod
    def _checkpoint_time(body,execution_rows,receipt):
        checkpoint=body.get('execution_head')
        if checkpoint is not None:
            for envelope,received in execution_rows:
                if envelope.payload_digest==checkpoint:
                    if received>receipt:raise ValueError('execution checkpoint follows portfolio receipt')
                    return
            raise ValueError('execution checkpoint unavailable')

    def _read(self):
        journal=self.journal;_,bindings=journal._roles();rows=journal._read('shadow-portfolio');state=None;head=None
        execution_rows,_=journal._replay()
        for index,(envelope,receipt) in enumerate(rows):
            try:
                event=verify_envelope(envelope,ShadowPortfolioEvent,journal.store.trust_store)
                journal._timing(envelope,event,receipt,journal.store.trust_store)
                interval=bindings.get((event.actor_id,envelope.signature.key_id))
                if interval is None or min(receipt,event.occurred_at,envelope.signature.created_at)<interval[0] or (interval[1] is not None and max(receipt,event.occurred_at,envelope.signature.created_at)>=interval[1]):
                    raise ValueError('portfolio actor not authorized')
                if event.sequence!=index or event.previous_digest!=head or (index and receipt<rows[index-1][1]):raise ValueError('portfolio chain/time')
                body=json.loads(canonical_bytes(event.body));self._source_time(event.kind,body,receipt,state)
                self._checkpoint_time(body,execution_rows,receipt)
                state=self._reduce(state,event.kind,body,head,execution_rows)
                head=envelope.payload_digest
            except Exception as exc:raise StoreCorruptionError('invalid signed shadow portfolio transition') from exc
        return rows,state,head

    def _views(self,state,head):
        if state is None:raise ValueError('shadow portfolio not initialized')
        positions=tuple(Position(instrument_id=name,market_value_cents=row['value'],sector_id=row['sector'],correlation_group_id=row['group']) for name,row in sorted(state['positions'].items()))
        nav=state['cash']+sum(row.market_value_cents for row in positions)
        snapshot=PortfolioSnapshot(observed_at=state['time'],nav_cents=nav,cash_cents=state['cash'],settled_cash_cents=state['settled'],reserved_cash_cents=0,positions=positions,open_order_count=0,unresolved_orders=False,reconciled=True,corporate_actions_clear=True,peak_nav_cents=max(state['peak'],nav-state['external']),net_external_cash_flow_since_peak_cents=state['external'],cohort_entry_session=state['entry'],source_digest=head)
        account=AccountObservationRef(account_alias=state['alias'],revision='shadow:'+head,observed_at=state['time'],expires_at=state['time']+timedelta(seconds=state['policy']['snapshot_ttl_seconds']),source_digest=head,settled_cash_cents=state['settled'],reserved_cash_cents=0,positions=tuple((name,row['quantity']) for name,row in sorted(state['positions'].items())))
        return snapshot,account

    @staticmethod
    def _pending(state,execution_rows):
        consumed=set(state['contexts']) if state else set()
        return any(ExecutionEvent.from_versioned_payload(envelope.payload).context_digest not in consumed for envelope,_ in execution_rows)

    def _reduce(self,state,kind,body,head,execution_rows):
        if kind=='INITIALIZE':
            if state is not None or set(body)!={'packet','account','fill_policy'}:raise ValueError('exact single initialization required')
            policy,request,_=reconstruct_risk_packet(body['packet']);account=_account(body['account']);fill_policy=_record(FillPolicy,body['fill_policy']);snapshot=request.snapshot
            if account.account_alias!=request.account_alias or account.source_digest!=snapshot.source_digest or snapshot.cash_cents!=snapshot.settled_cash_cents or snapshot.reserved_cash_cents or account.reserved_cash_cents or account.settled_cash_cents!=snapshot.settled_cash_cents or snapshot.open_order_count or snapshot.unresolved_orders or not snapshot.reconciled or not snapshot.corporate_actions_clear:
                raise ValueError('clear fully settled baseline required')
            quantities=dict(account.positions)
            if set(quantities)!={position.instrument_id for position in snapshot.positions}:raise ValueError('baseline inventory mismatch')
            state=dict(alias=account.account_alias,time=request.as_of,session=request.session_index,session_open=request.session_open,session_close=request.session_close,calendar=request.calendar_digest,cash=snapshot.cash_cents,settled=snapshot.settled_cash_cents,peak=snapshot.peak_nav_cents,external=snapshot.net_external_cash_flow_since_peak_cents,entry=snapshot.cohort_entry_session,positions={p.instrument_id:dict(quantity=quantities[p.instrument_id],value=p.market_value_cents,sector=p.sector_id,group=p.correlation_group_id) for p in snapshot.positions},pending=[],contexts=[],policy=asdict(policy),fill_policy=asdict(fill_policy))
            self._views(state,head or snapshot.source_digest)
            return state
        if state is None:raise ValueError('baseline required')
        state=json.loads(canonical_bytes(state));state['time']=_time(state['time']);state['session_open']=_time(state['session_open']);state['session_close']=_time(state['session_close'])
        if kind=='OBSERVE':
            if set(body)!={'market','session_index','execution_head'} or type(body['session_index']) is not int:raise ValueError('exact observation required')
            market=_market(body['market']);session=body['session_index']
            if market.calendar_digest!=state['calendar']:raise ValueError('frozen calendar drift')
            if session<state['session'] or market.as_of<state['time']:raise ValueError('session/time regression')
            if session==state['session']:
                if (market.session_open,market.session_close)!=(state['session_open'],state['session_close']):raise ValueError('same modeled session identity changed')
            elif session!=state['session']+1 or market.session_open<state['session_close'] or market.as_of<=state['time']:
                raise ValueError('next nonoverlapping modeled session required')
            prefix=[]
            if body['execution_head'] is not None:
                for envelope,receipt in execution_rows:
                    prefix.append((envelope,receipt))
                    if envelope.payload_digest==body['execution_head']:break
                if not prefix or prefix[-1][0].payload_digest!=body['execution_head']:raise ValueError('observation execution checkpoint missing')
            if self._pending(state,prefix):raise ValueError('unapplied execution requires recovery before observation')
            if not market.session_open<=market.as_of<market.session_close:raise ValueError('observation session closed')
            quotes={quote.instrument_id:quote for quote in market.quotes}
            for name,row in state['positions'].items():
                quote=quotes.get(name)
                if quote is None or quote.halted or quote.available_at>market.as_of or not 0<=(market.as_of-quote.quoted_at).total_seconds()<=state['policy']['quote_ttl_seconds']:raise ValueError('held instrument valuation unavailable')
                row['value']=row['quantity']*quote.bid_micros//10000
            state['settled']+=sum(item['credit'] for item in state['pending'] if item['session']<=session)
            state['pending']=[item for item in state['pending'] if item['session']>session]
            state['time']=market.as_of;state['session']=session;state['session_open']=market.session_open;state['session_close']=market.session_close
        elif kind=='STEP':
            if set(body)!={'packet','account','capabilities','fill_policy','market','context','execution_head'}:raise ValueError('exact step required')
            policy,request,decision=reconstruct_risk_packet(body['packet']);account=_account(body['account']);caps=_record(BrokerCapabilities,body['capabilities']);fill_policy=_record(FillPolicy,body['fill_policy']);market=_market(body['market'])
            if market.as_of!=request.as_of or (market.session_open,market.session_close,market.calendar_digest)!=(request.session_open,request.session_close,request.calendar_digest) or market.calendar_digest!=state['calendar'] or not market.session_open<=market.as_of<market.session_close:
                raise ValueError('market decision/session/calendar mismatch')
            if body['context'] in state['contexts']:raise ValueError('duplicate consumed step')
            snapshot,current_account=self._views(state,head)
            if canonical_bytes(asdict(request.snapshot))!=canonical_bytes(asdict(snapshot)) or canonical_bytes(asdict(account))!=canonical_bytes(asdict(current_account)) or request.session_index!=state['session'] or request.as_of!=state['time'] or asdict(policy)!=state['policy'] or asdict(fill_policy)!=state['fill_policy']:raise ValueError('stale portfolio/policy step')
            intents=build_order_intents(decision,request,account,caps,mode='SHADOW')
            context=content_digest({'request':asdict(request),'decision':asdict(decision),'account':asdict(account),'capabilities':asdict(caps),'fill_policy':asdict(fill_policy),'market':asdict(market),'mode':'SHADOW'})
            if context!=body['context']:raise ValueError('step context mismatch')
            prefix=[]
            for envelope,_ in execution_rows:
                prefix.append(envelope)
                if envelope.payload_digest==body['execution_head']:break
            if body['execution_head'] is not None and (not prefix or prefix[-1].payload_digest!=body['execution_head']):raise ValueError('execution checkpoint missing')
            events=[ExecutionEvent.from_versioned_payload(envelope.payload) for envelope in prefix]
            matching={event.intent_id:event for event in events if event.context_digest==context}
            if set(matching)!={intent.intent_id for intent in intents}:raise ValueError('complete execution batch missing')
            quotes={quote.instrument_id:quote for quote in market.quotes};metadata={item.instrument_id:item for item in request.candidates}
            for intent in intents:
                event=matching[intent.intent_id];fill=hypothetical_fill(intent,quotes.get(intent.instrument_id),market,fill_policy)
                if event.state!='COMPLETED' or canonical_bytes(event.intent)!=canonical_bytes(asdict(intent)) or canonical_bytes(event.result)!=canonical_bytes(asdict(fill)):raise ValueError('durable hypothetical completion mismatch')
                if not fill.quantity:continue
                if intent.side=='BUY':
                    debit=(fill.quantity*fill.price_micros+9999)//10000+fill.fee_cents
                    if debit>state['settled']:raise ValueError('unsettled funds cannot buy')
                    state['cash']-=debit;state['settled']-=debit
                    if intent.instrument_id in state['positions']:raise ValueError('overlapping cohorts denied')
                    info=metadata[intent.instrument_id];state['positions'][intent.instrument_id]=dict(quantity=fill.quantity,value=fill.quantity*quotes[intent.instrument_id].bid_micros//10000,sector=info.sector_id,group=info.correlation_group_id)
                    state['entry']=state['session']
                else:
                    row=state['positions'].get(intent.instrument_id)
                    if row is None or fill.quantity>row['quantity']:raise ValueError('oversell denied')
                    credit=fill.quantity*fill.price_micros//10000-fill.fee_cents
                    if credit<0:raise ValueError('negative sale proceeds')
                    state['cash']+=credit;state['pending'].append(dict(session=state['session']+1,credit=credit))
                    row['quantity']-=fill.quantity;row['value']=row['quantity']*quotes[intent.instrument_id].bid_micros//10000
                    if not row['quantity']:del state['positions'][intent.instrument_id]
            if not state['positions']:state['entry']=None
            state['contexts'].append(context)
        else:raise ValueError('unknown portfolio transition')
        nav=state['cash']+sum(row['value'] for row in state['positions'].values());state['peak']=max(state['peak'],nav-state['external'])
        self._views(state,head)
        return state

    def _append(self,kind,body):
        journal=self.journal
        with journal.store._locked():
            rows,state,head=self._read();execution_rows,_=journal._replay()
            if kind=='STEP' and state and body['context'] in state['contexts']:
                for envelope,_ in rows:
                    event=ShadowPortfolioEvent.from_versioned_payload(envelope.payload)
                    if event.kind=='STEP' and canonical_bytes({key:value for key,value in event.body.items() if key!='execution_head'})==canonical_bytes({key:value for key,value in body.items() if key!='execution_head'}):return envelope.payload_digest
                raise ValueError('changed consumed portfolio replay')
            if kind in ('STEP','OBSERVE') and body['execution_head']!=(execution_rows[-1][0].payload_digest if execution_rows else None):
                raise ValueError('execution checkpoint changed before portfolio admission')
            self._source_time(kind,json.loads(canonical_bytes(body)),journal.store._trusted_clock(),state)
            if kind=='INITIALIZE' and execution_rows:raise ValueError('baseline cannot erase prior execution')
            self._reduce(state,kind,json.loads(canonical_bytes(body)),head,execution_rows)
            now=journal.store._trusted_clock();authority_rows,bindings=journal._roles()
            self._source_time(kind,json.loads(canonical_bytes(body)),now,state)
            self._checkpoint_time(body,execution_rows,now)
            if (rows and now<rows[-1][1]) or (authority_rows and now<authority_rows[-1][1]):raise ValueError('portfolio receipt clock regressed')
            interval=bindings.get((journal.actor_id,journal.signer.key_id))
            if interval is None or now<interval[0] or interval[1] is not None:raise ValueError('portfolio actor retired/unregistered')
            journal.store.trust_store.require_trusted_at_receipt(journal.signer.key_id,now)
            event=ShadowPortfolioEvent(sequence=len(rows),previous_digest=head,actor_id=journal.actor_id,occurred_at=now,kind=kind,body=json.loads(canonical_bytes(body)))
            signed=journal.signer.sign(event,created_at=now);journal._write('shadow-portfolio',signed,now,len(rows));return signed.payload_digest

    def initialize(self,request,account,*,risk_policy,fill_policy):
        from src.axiom2.portfolio.authority import evaluate_portfolio
        decision=evaluate_portfolio(request,risk_policy)
        return self._append('INITIALIZE',dict(packet=dict(request=asdict(request),policy=asdict(risk_policy),decision=asdict(decision)),account=asdict(account),fill_policy=asdict(fill_policy)))

    def observe(self,market,*,session_index):return self._append('OBSERVE',dict(market=asdict(market),session_index=session_index,execution_head=self.journal.head))

    def observations(self):
        with self.journal.store._locked():
            _,state,head=self._read();execution_rows,_=self.journal._replay()
            if self._pending(state,execution_rows):raise ValueError('unapplied execution requires recovery')
            return self._views(state,head)

    def validate_step(self,request,account,*,context,decision,risk_policy,fill_policy,market):
        with self.journal.store._locked():
            _,state,head=self._read()
            reconstruct_risk_packet(json.loads(canonical_bytes(dict(request=asdict(request),policy=asdict(risk_policy),decision=asdict(decision)))))
            if state and (asdict(risk_policy)!=state['policy'] or asdict(fill_policy)!=state['fill_policy']):raise ValueError('frozen shadow policy drift')
            if state and context in state['contexts']:return
            self._source_time('STEP',json.loads(canonical_bytes(dict(packet=dict(request=asdict(request),policy=asdict(risk_policy),decision=asdict(decision)),market=asdict(market),fill_policy=asdict(fill_policy)))),self.journal.store._trusted_clock(),state)
            snapshot,current=self._views(state,head)
            if canonical_bytes(asdict(snapshot))!=canonical_bytes(asdict(request.snapshot)) or canonical_bytes(asdict(current))!=canonical_bytes(asdict(account)):raise ValueError('stale continuous shadow account')

    def apply_step(self,request,decision,account,caps,fill_policy,market,*,risk_policy,context):
        return self._append('STEP',dict(packet=dict(request=asdict(request),policy=asdict(risk_policy),decision=asdict(decision)),account=asdict(account),capabilities=asdict(caps),fill_policy=asdict(fill_policy),market=asdict(market),context=context,execution_head=self.journal.head))
