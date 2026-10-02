"""Durable signed shadow stream using the EvidenceStore's lock and fsync primitives."""
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from types import MappingProxyType
from collections.abc import Mapping
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import SignedEnvelope
from src.evidence.hashing import content_digest
from src.evidence.signing import verify_envelope
from src.evidence.store import ConcurrentHeadError, ImmutableConflictError, StoreCorruptionError
from src.evidence.execution_shadow import AuthorityBindingEvent, ExecutionEvent, ShadowJournalReceipt

@dataclass(frozen=True)
class Reservation:
    intent_id: str
    intent_digest: str
    context_digest: str
    state: str
    result: Mapping | None

class ExecutionJournal:
    def __init__(self, store, *, signer, actor_id, operator_trust_store, receipt_trust_store):
        self.store, self.signer, self.actor_id = store, signer, actor_id
        self.operator_trust_store = operator_trust_store
        self.receipt_trust_store = receipt_trust_store

    def _directory(self, name):
        path = self.store.root / name
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise StoreCorruptionError('invalid shadow stream directory')
        return path

    def _read(self, name):
        directory = self._directory(name)
        if not directory.exists(): return []
        rows = []
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file(): raise StoreCorruptionError('invalid stream entry')
            if path.name.startswith('.') and '.tmp-' in path.name: continue
            try:
                raw = path.read_bytes()
                signed_receipt = SignedEnvelope.model_validate_json(raw, strict=True)
                if canonical_bytes(signed_receipt) != raw:
                    raise ValueError('noncanonical receipt')
                record = verify_envelope(signed_receipt, ShadowJournalReceipt, self.receipt_trust_store)
                receipt = record.received_at
                if signed_receipt.signature.created_at != receipt or receipt.tzinfo is None:
                    raise ValueError('receipt timestamp does not match its signature')
                self.receipt_trust_store.require_trusted_at_receipt(signed_receipt.signature.key_id, receipt)
                if record.stream != name:
                    raise ValueError('cross-stream receipt')
                envelope = record.envelope
                if path.name != f'{len(rows):020d}-{envelope.payload_digest}.json':
                    raise ValueError('receipt address')
                rows.append((envelope, receipt))
            except Exception as exc: raise StoreCorruptionError('invalid signed shadow receipt') from exc
        return rows

    def _timing(self, envelope, event, receipt, trust):
        signed = envelope.signature.created_at
        if event.occurred_at.tzinfo is None or abs((signed-event.occurred_at).total_seconds()) > 5:
            raise StoreCorruptionError('event signing time mismatch')
        if event.occurred_at > receipt or signed > receipt or (receipt-signed).total_seconds() > 30:
            raise StoreCorruptionError('event receipt time mismatch')
        trust.require_trusted_at_receipt(envelope.signature.key_id, receipt)

    def _roles(self):
        rows = self._read('shadow-authority'); bindings = {}; previous = None
        for index,(envelope,receipt) in enumerate(rows):
            event = verify_envelope(envelope, AuthorityBindingEvent, self.operator_trust_store)
            self._timing(envelope,event,receipt,self.operator_trust_store)
            if event.sequence != index or event.previous_digest != previous: raise StoreCorruptionError('authority chain')
            if index and receipt < rows[index-1][1]: raise StoreCorruptionError('authority receipt time regression')
            identity = (event.actor_id,event.key_id)
            if event.action == 'REGISTER':
                if identity in bindings: raise StoreCorruptionError('duplicate authority registration')
                bindings[identity] = (receipt,None)
            else:
                if identity not in bindings or bindings[identity][1] is not None: raise StoreCorruptionError('invalid revocation')
                bindings[identity] = (bindings[identity][0],receipt)
            previous = envelope.payload_digest
        return rows,bindings

    def _replay(self):
        _,bindings = self._roles(); rows=self._read('shadow-execution'); states={}; previous=None; decisions={}
        for index,(envelope,receipt) in enumerate(rows):
            event=verify_envelope(envelope,ExecutionEvent,self.store.trust_store)
            self._timing(envelope,event,receipt,self.store.trust_store)
            interval=bindings.get((event.actor_id,envelope.signature.key_id))
            if interval is None or min(receipt, event.occurred_at, envelope.signature.created_at) < interval[0] or (interval[1] is not None and max(receipt, event.occurred_at, envelope.signature.created_at) >= interval[1]):
                raise StoreCorruptionError('shadow role not authorized at receipt')
            if event.sequence != index or event.previous_digest != previous: raise StoreCorruptionError('execution chain')
            if index and receipt < rows[index-1][1]: raise StoreCorruptionError('execution receipt time regression')
            if content_digest(event.intent) != event.intent_digest or event.intent.get('intent_id') != event.intent_id or event.intent.get('mode') != 'SHADOW':
                raise StoreCorruptionError('intent integrity')
            self._validate_intent(event.intent)
            self._bind_account_revision(decisions, event.intent)
            old=states.get(event.intent_id)
            if event.state=='RESERVED':
                if old or event.result is not None: raise StoreCorruptionError('invalid reservation transition')
            else:
                if not old or old.state!='RESERVED' or old.intent_digest!=event.intent_digest or old.context_digest!=event.context_digest:
                    raise StoreCorruptionError('invalid completion transition')
                self._validate_result(event.result,event.intent_id,event.intent)
            states[event.intent_id]=Reservation(event.intent_id,event.intent_digest,event.context_digest,event.state,event.result)
            previous=envelope.payload_digest
        return rows,states

    @staticmethod
    def _bind_account_revision(decisions, intent):
        identity = (intent['account_alias'], intent['account_revision'])
        decision = intent['decision_digest']
        if identity in decisions and decisions[identity] != decision:
            raise ValueError('account revision already consumed by another shadow decision')
        decisions[identity] = decision

    @staticmethod
    def _validate_intent(payload):
        from src.axiom2.contracts.equity_orders import EquityOrderIntent
        values = dict(payload)
        for name, expected in [('execution_enabled', False), ('capital_authorized', False), ('schema_version', 1)]:
            if name not in values or type(values[name]) is not type(expected) or values.pop(name) != expected:
                raise ValueError('intent authority/schema mismatch')
        values['expires_at'] = datetime.fromisoformat(values['expires_at'].replace('Z', '+00:00'))
        intent = EquityOrderIntent(**values)
        if intent.quantity is None:
            raise ValueError('initial shadow profile requires whole-share quantities')
        if canonical_bytes(asdict(intent)) != canonical_bytes(payload):
            raise ValueError('intent canonical contents mismatch')
        return intent

    @staticmethod
    def _validate_result(result, intent_id, intent=None):
        from src.axiom2.portfolio.contracts import _integer, _reference
        if not isinstance(result, Mapping) or result.get('intent_id') != intent_id or result.get('hypothetical') is not True or result.get('execution_enabled') is not False or result.get('capital_authorized') is not False:
            raise ValueError('completion must be explicitly hypothetical with no authority')
        expected = {'intent_id', 'status', 'quantity', 'price_micros', 'fee_cents', 'uncertainty', 'hypothetical', 'execution_enabled', 'capital_authorized'}
        if set(result) != expected:
            raise ValueError('unknown or missing completion fields')
        quantity = result['quantity']
        _integer(quantity, 'quantity')
        _integer(result['fee_cents'], 'fee_cents')
        if result['status'] not in {'FILLED', 'PARTIAL', 'UNFILLED'}:
            raise ValueError('invalid hypothetical fill status')
        reasons = result['uncertainty']
        if not isinstance(reasons, (list, tuple)) or 'HYPOTHETICAL_NOT_BROKER_EVIDENCE' not in reasons:
            raise ValueError('explicit hypothetical uncertainty required')
        for reason in reasons:
            _reference(reason, 'uncertainty')
        if quantity:
            _integer(result['price_micros'], 'price_micros', 1)
            if result['status'] == 'UNFILLED':
                raise ValueError('unfilled result has quantity')
        elif result['status'] != 'UNFILLED' or result['price_micros'] is not None or result['fee_cents'] != 0:
            raise ValueError('empty result must be unfilled without costs')
        if intent is not None:
            if quantity > intent['quantity']:
                raise ValueError('hypothetical fill exceeds reservation')
            if quantity and ((intent['side'] == 'BUY' and result['price_micros'] > intent['limit_price_micros']) or (intent['side'] == 'SELL' and result['price_micros'] < intent['limit_price_micros'])):
                raise ValueError('hypothetical fill violates limit price')
            if result['status'] == 'FILLED' and quantity != intent['quantity']:
                raise ValueError('FILLED must consume full reservation')
            if result['status'] == 'PARTIAL' and not 0 < quantity < intent['quantity']:
                raise ValueError('PARTIAL must consume a strict portion')

    def _write(self,name,envelope,receipt,sequence):
        directory=self._directory(name)
        if not directory.exists():
            directory.mkdir(mode=0o700); self.store._fsync_directory(directory); self.store._fsync_directory(self.store.root)
        self.receipt_trust_store.resolve(self.signer.key_id, receipt)
        self.receipt_trust_store.require_trusted_at_receipt(self.signer.key_id, receipt)
        record = ShadowJournalReceipt(envelope=envelope, received_at=receipt, stream=name)
        signed_receipt = self.signer.sign(record, created_at=receipt)
        self.store._atomic_create_bytes(directory/f'{sequence:020d}-{envelope.payload_digest}.json',canonical_bytes(signed_receipt))

    def configure(self,envelope,*,expected_head=None):
        envelope=SignedEnvelope.model_validate_json(canonical_bytes(envelope),strict=True)
        with self.store._locked():
            rows,bindings=self._roles(); head=rows[-1][0].payload_digest if rows else None
            if rows and rows[-1][0]==envelope: return head
            if head!=expected_head: raise ConcurrentHeadError('authority head changed')
            event=verify_envelope(envelope,AuthorityBindingEvent,self.operator_trust_store); now=self.store._trusted_clock()
            self._timing(envelope,event,now,self.operator_trust_store)
            if event.sequence!=len(rows) or event.previous_digest!=head: raise ValueError('authority chain')
            if rows and now < rows[-1][1]: raise ValueError('authority clock regressed')
            identity=(event.actor_id,event.key_id)
            if event.action=='REGISTER' and identity in bindings: raise ValueError('duplicate registration')
            if event.action=='REVOKE' and (identity not in bindings or bindings[identity][1] is not None): raise ValueError('invalid revocation')
            provider_rows=self._read('shadow-provider-rehearsal')
            if provider_rows and (now<provider_rows[-1][1] or (event.action=='REVOKE' and now<=provider_rows[-1][1])):
                raise ValueError('fake provider clock must precede authority retirement')
            fence_rows=self._read('shadow-fences')
            if fence_rows and (now<fence_rows[-1][1] or (event.action=='REVOKE' and now<=fence_rows[-1][1])):
                raise ValueError('fence clock must precede authority retirement')
            portfolio_rows=self._read('shadow-portfolio')
            if portfolio_rows and (now<portfolio_rows[-1][1] or (event.action=='REVOKE' and now<=portfolio_rows[-1][1])):
                raise ValueError('portfolio clock must precede authority retirement')
            execution_rows, _ = self._replay()
            if execution_rows and now < execution_rows[-1][1]:
                raise ValueError('execution clock regressed before authority configuration')
            if event.action == 'REVOKE' and execution_rows and now <= execution_rows[-1][1]:
                raise ValueError('retirement clock must follow committed execution history')
            self._write('shadow-authority',envelope,now,len(rows)); return envelope.payload_digest

    @property
    def head(self):
        with self.store._locked():
            rows,_=self._replay(); return rows[-1][0].payload_digest if rows else None

    def replay(self):
        with self.store._locked(): return MappingProxyType(self._replay()[1])

    def _append_unlocked(self,envelope,expected_head):
        rows,states=self._replay(); head=rows[-1][0].payload_digest if rows else None
        if rows and rows[-1][0]==envelope: return states[envelope.payload['intent_id']]
        if head!=expected_head: raise ConcurrentHeadError('shadow head changed')
        event=verify_envelope(envelope,ExecutionEvent,self.store.trust_store); now=self.store._trusted_clock()
        self._timing(envelope,event,now,self.store.trust_store)
        role_rows,bindings=self._roles(); interval=bindings.get((event.actor_id,envelope.signature.key_id))
        if role_rows and now < role_rows[-1][1]: raise ValueError('authority clock regressed before execution')
        if interval is None or min(now, event.occurred_at, envelope.signature.created_at) < interval[0] or interval[1] is not None: raise ValueError('shadow signer role retired or unregistered')
        if event.sequence!=len(rows) or event.previous_digest!=head: raise ValueError('shadow chain')
        if rows and now < rows[-1][1]: raise ValueError('execution clock regressed')
        if content_digest(event.intent)!=event.intent_digest or event.intent.get('intent_id')!=event.intent_id or event.intent.get('mode')!='SHADOW': raise ValueError('intent integrity')
        self._validate_intent(event.intent)
        decisions = {}
        for prior, _ in rows:
            self._bind_account_revision(decisions, prior.payload['intent'])
        self._bind_account_revision(decisions, event.intent)
        old=states.get(event.intent_id)
        if event.state=='RESERVED':
            if old or event.result is not None: raise ValueError('invalid reservation')
        else:
            if not old or old.state!='RESERVED' or old.intent_digest!=event.intent_digest or old.context_digest!=event.context_digest: raise ValueError('invalid completion')
            self._validate_result(event.result,event.intent_id,event.intent)
        self._write('shadow-execution',envelope,now,len(rows))
        return Reservation(event.intent_id,event.intent_digest,event.context_digest,event.state,event.result)

    def append(self,event,expected_head=None):
        envelope=SignedEnvelope.model_validate_json(canonical_bytes(event),strict=True)
        with self.store._locked(): return self._append_unlocked(envelope,expected_head)

    def reserve(self,intent,*,context_digest):
        from src.axiom2.contracts.equity_orders import EquityOrderIntent
        if type(intent) is not EquityOrderIntent:
            raise TypeError('exact EquityOrderIntent required')
        payload=json.loads(canonical_bytes(asdict(intent)))
        intent=self._validate_intent(payload)
        digest=content_digest(payload)
        with self.store._locked():
            rows,states=self._replay(); old=states.get(intent.intent_id)
            if old:
                if old.intent_digest!=digest or old.context_digest!=context_digest: raise ImmutableConflictError('changed reservation replay')
                return old
            head=rows[-1][0].payload_digest if rows else None; now=self.store._trusted_clock()
            event=ExecutionEvent(sequence=len(rows),previous_digest=head,actor_id=self.actor_id,occurred_at=now,intent_id=intent.intent_id,intent_digest=digest,context_digest=context_digest,state='RESERVED',intent=payload)
            return self._append_unlocked(self.signer.sign(event,created_at=now),head)

    def complete(self,intent_id,*,result,expected_head):
        with self.store._locked():
            rows,states=self._replay(); old=states.get(intent_id)
            if old is None: raise ValueError('unknown reservation')
            result=json.loads(canonical_bytes(result)); self._validate_result(result,intent_id)
            if old.state=='COMPLETED':
                if canonical_bytes(old.result)!=canonical_bytes(result): raise ImmutableConflictError('changed completion replay')
                return old
            payload=next(e.payload['intent'] for e,_ in rows if e.payload['intent_id']==intent_id)
            now=self.store._trusted_clock()
            event=ExecutionEvent(sequence=len(rows),previous_digest=expected_head,actor_id=self.actor_id,occurred_at=now,intent_id=intent_id,intent_digest=old.intent_digest,context_digest=old.context_digest,state='COMPLETED',intent=payload,result=result)
            return self._append_unlocked(self.signer.sign(event,created_at=now),expected_head)
