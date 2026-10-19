"""Opaque authorization preparation; signed bytes do not grant authority.

A future privileged verifier must bind account, candidate, policy, limits,
window, build, capability and nonce with time-bounded actor/key trust. An offline exact-scope verifier and signed nonce reservation journal are
implemented below; privileged artifact/readiness resolution and production
gateway integration remain absent. No genuine holdout is opened.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class CapitalAuthorization:
    """Unverified immutable envelope, never a runtime activation flag."""

    envelope_bytes: bytes

    def __post_init__(self):
        if type(self.envelope_bytes) is not bytes:
            raise TypeError('authorization envelope must be immutable bytes')
        if not self.envelope_bytes:
            raise ValueError('authorization envelope must not be empty')

# Offline verification primitives. The disabled public gateway never calls these.
# Verification here authenticates the bounded declaration, not readiness or orders.
from datetime import datetime, timezone
from typing import Literal
from pydantic import Field
from src.evidence.contracts import StrictContract, SignedEnvelope
from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import content_digest
from src.evidence.signing import verify_envelope
from src.evidence.store import ImmutableConflictError, StoreCorruptionError


class OperatorAuthorization(StrictContract):
    account_alias: str = Field(min_length=1)
    candidate_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    risk_policy_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    intent_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    build_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    capability_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    nonce: str = Field(min_length=1)
    not_before: datetime
    expires_at: datetime
    maximum_order_quantity: int = Field(gt=0)
    maximum_capital_cents: int = Field(gt=0)
    purpose: Literal['OFFLINE_PREPARATION'] = 'OFFLINE_PREPARATION'
    execution_enabled: Literal[False] = False
    capital_authorized: Literal[False] = False


SCOPE_FIELDS = frozenset(('account_alias', 'candidate_digest', 'model_digest',
    'risk_policy_digest', 'intent_digest', 'build_digest', 'capability_digest'))


def verify_authorization(envelope, operator_trust_store, *, expected, now, quantity, capital_cents):
    """Authenticate exact offline scope using a dedicated operator trust store.

    expected must come from a future privileged resolver; caller-supplied expected
    digests do not authenticate artifact truth. No readiness receipt is issued.
    """
    envelope=SignedEnvelope.model_validate_json(canonical_bytes(envelope),strict=True)
    declaration=verify_envelope(envelope,OperatorAuthorization,operator_trust_store)
    if type(expected) is not dict or set(expected)!=SCOPE_FIELDS:
        raise ValueError('exact authorization scope required')
    if any(type(v) is not str for v in expected.values()):
        raise ValueError('invalid authorization scope')
    if any(getattr(declaration,key)!=value for key,value in expected.items()):
        raise ValueError('authorization scope mismatch')
    for value in (now,declaration.not_before,declaration.expires_at):
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('aware authorization time required')
    instant=now.astimezone(timezone.utc)
    if not declaration.not_before<=envelope.signature.created_at<=instant<declaration.expires_at:
        raise ValueError('authorization expired or not yet valid')
    operator_trust_store.require_trusted_at_receipt(envelope.signature.key_id,instant)
    for value,limit in ((quantity,declaration.maximum_order_quantity),(capital_cents,declaration.maximum_capital_cents)):
        if type(value) is not int or not 0<value<=limit:
            raise ValueError('authorization limit exceeded')
    return declaration


class AuthorizationReservation(StrictContract):
    authorization: SignedEnvelope
    request_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    quantity: int = Field(gt=0)
    capital_cents: int = Field(gt=0)
    received_at: datetime
    state: Literal['RESERVED_OFFLINE'] = 'RESERVED_OFFLINE'
    execution_enabled: Literal[False] = False
    capital_authorized: Literal[False] = False


class AuthorizationJournal:
    """Atomic single-use operator-key/nonce reservation in a signed offline stream.

    No broker side effect or submission state exists. Truncation/whole-store
    rollback requires an external checkpoint as for EvidenceStore. Production
    gateway integration, artifact resolution and distributed leases are absent.
    """
    def __init__(self,store,*,signer,operator_trust_store):
        self.store=store;self.signer=signer;self.operator_trust_store=operator_trust_store

    def reserve(self,envelope,*,expected,quantity,capital_cents):
        envelope=SignedEnvelope.model_validate_json(canonical_bytes(envelope),strict=True)
        with self.store._locked():
            now=self.store._trusted_clock()
            declaration=verify_authorization(envelope,self.operator_trust_store,
                expected=expected,now=now,quantity=quantity,capital_cents=capital_cents)
            request_digest=content_digest({'scope':expected,'quantity':quantity,'capital_cents':capital_cents,
                'authorization_digest':envelope.payload_digest})
            identity=content_digest({'operator_key':envelope.signature.key_id,'nonce':declaration.nonce})
            directory=self.store.root/'offline-authorizations'
            if directory.is_symlink() or directory.exists() and not directory.is_dir():
                raise StoreCorruptionError('invalid authorization stream')
            path=directory/(identity+'.json')
            if path.is_symlink(): raise StoreCorruptionError('invalid authorization reservation')
            if path.exists():
                try:
                    raw=path.read_bytes(); signed=SignedEnvelope.model_validate_json(raw,strict=True)
                    if canonical_bytes(signed)!=raw: raise ValueError('noncanonical reservation')
                    old=verify_envelope(signed,AuthorizationReservation,self.store.trust_store)
                    if old.received_at!=signed.signature.created_at: raise ValueError('receipt-time mismatch')
                    self.store.trust_store.require_trusted_at_receipt(signed.signature.key_id,old.received_at)
                    old_declaration=verify_authorization(old.authorization,self.operator_trust_store,
                        expected=expected,now=old.received_at,quantity=old.quantity,capital_cents=old.capital_cents)
                    old_identity=content_digest({'operator_key':old.authorization.signature.key_id,'nonce':old_declaration.nonce})
                    old_request=content_digest({'scope':expected,'quantity':old.quantity,'capital_cents':old.capital_cents,
                        'authorization_digest':old.authorization.payload_digest})
                    if old_identity!=identity or old_request!=old.request_digest:
                        raise ValueError('reservation integrity')
                except (ValueError,TypeError,OSError) as exc:
                    raise StoreCorruptionError('invalid signed authorization reservation') from exc
                if old.request_digest!=request_digest or old.authorization.payload_digest!=envelope.payload_digest:
                    raise ImmutableConflictError('operator nonce already reserved for a different request')
                return old
            if not directory.exists():
                directory.mkdir(mode=0o700);self.store._fsync_directory(directory);self.store._fsync_directory(self.store.root)
            reservation=AuthorizationReservation(authorization=envelope,request_digest=request_digest,
                quantity=quantity,capital_cents=capital_cents,received_at=now)
            self.store.trust_store.require_trusted_at_receipt(self.signer.key_id,now)
            signed=self.signer.sign(reservation,created_at=now)
            self.store._atomic_create_bytes(path,canonical_bytes(signed))
            return reservation
