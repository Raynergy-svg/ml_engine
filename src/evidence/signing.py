"""Ed25519 detached signatures, signer identity, rotation, and revocation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_bytes
from .contracts import DispositionEvent, SignatureRecord, SignedEnvelope, StrictContract
from .hashing import content_digest, sha256_bytes


class SignatureVerificationError(ValueError):
    """Raised when a signature, digest, signer, or trust interval is invalid."""


MAX_SIGNATURE_EVENT_SKEW = timedelta(seconds=5)
MAX_EVENT_FUTURE_SKEW = timedelta(seconds=30)


@dataclass(frozen=True)
class TrustedKey:
    key_id: str
    public_key: Ed25519PublicKey
    valid_from: datetime
    valid_until: datetime | None = None
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("valid_from", "valid_until", "revoked_at"):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware")
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("valid_until must follow valid_from")
        if self.revoked_at is not None and self.revoked_at < self.valid_from:
            raise ValueError("revoked_at must not precede valid_from")

    def permits(self, signed_at: datetime) -> bool:
        instant = signed_at.astimezone(timezone.utc)
        if instant < self.valid_from.astimezone(timezone.utc):
            return False
        if self.valid_until is not None and instant >= self.valid_until.astimezone(timezone.utc):
            return False
        return self.revoked_at is None or instant < self.revoked_at.astimezone(timezone.utc)


class TrustStore:
    """In-memory public-key registry supporting overlap during key rotation."""

    def __init__(self) -> None:
        self._keys: dict[str, TrustedKey] = {}

    def add(self, key: TrustedKey) -> None:
        if key.key_id in self._keys:
            raise ValueError(f"trusted key already exists: {key.key_id}")
        self._keys[key.key_id] = key

    def resolve(self, key_id: str, signed_at: datetime) -> Ed25519PublicKey:
        key = self._keys.get(key_id)
        if key is None:
            raise SignatureVerificationError(f"untrusted signer key: {key_id}")
        if not key.permits(signed_at):
            raise SignatureVerificationError(f"signer key is outside its trusted interval: {key_id}")
        return key.public_key

    def require_trusted_at_receipt(self, key_id: str, received_at: datetime) -> None:
        """Reject evidence first observed after a key ceased to be trusted."""
        key = self._keys.get(key_id)
        if key is None:
            raise SignatureVerificationError(f"untrusted signer key: {key_id}")
        if not key.permits(received_at):
            raise SignatureVerificationError(
                f"signer key was not trusted at ledger receipt time: {key_id}"
            )

    def revoke(self, key_id: str, revoked_at: datetime) -> None:
        key = self._keys.get(key_id)
        if key is None:
            raise KeyError(key_id)
        replacement = TrustedKey(
            key_id=key.key_id,
            public_key=key.public_key,
            valid_from=key.valid_from,
            valid_until=key.valid_until,
            revoked_at=revoked_at,
        )
        self._keys[key_id] = replacement


class Ed25519Signer:
    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private_key = private_key

    @classmethod
    def generate(cls) -> "Ed25519Signer":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_private_bytes(cls, value: bytes) -> "Ed25519Signer":
        return cls(Ed25519PrivateKey.from_private_bytes(value))

    @classmethod
    def from_private_pem(cls, value: bytes, password: bytes | None = None) -> "Ed25519Signer":
        key = serialization.load_pem_private_key(value, password=password)
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError("private key must be Ed25519")
        return cls(key)

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._private_key.public_key()

    @property
    def key_id(self) -> str:
        raw = self.public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return f"ed25519:{sha256_bytes(raw)[:32]}"

    def private_bytes(self) -> bytes:
        return self._private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )

    def private_pem(self, password: bytes | None = None) -> bytes:
        encryption = serialization.NoEncryption() if password is None else serialization.BestAvailableEncryption(password)
        return self._private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            encryption,
        )

    def trusted_key(
        self,
        *,
        valid_from: datetime,
        valid_until: datetime | None = None,
        revoked_at: datetime | None = None,
    ) -> TrustedKey:
        return TrustedKey(self.key_id, self.public_key, valid_from, valid_until, revoked_at)

    def sign_digest(self, digest: str, *, payload_type: str, created_at: datetime) -> SignatureRecord:
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        try:
            digest_bytes = bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError("digest must be hexadecimal") from exc
        if len(digest_bytes) != 32:
            raise ValueError("digest must be a SHA-256 digest")
        unsigned = SignatureRecord(
            key_id=self.key_id,
            payload_type=payload_type,
            signed_digest=digest,
            signature_b64="pending",
            created_at=created_at,
        )
        signature = self._private_key.sign(_signature_message(unsigned))
        return unsigned.model_copy(update={"signature_b64": base64.b64encode(signature).decode("ascii")})

    def sign(self, payload: StrictContract, *, created_at: datetime) -> SignedEnvelope:
        digest = content_digest(payload)
        return SignedEnvelope(
            payload_type=type(payload).__name__,
            payload=payload.model_dump(mode="json", round_trip=True, warnings=False),
            payload_digest=digest,
            signature=self.sign_digest(digest, payload_type=type(payload).__name__, created_at=created_at),
        )

    def sign_versioned_payload(
        self,
        payload: Mapping[str, Any],
        *,
        payload_type: str,
        created_at: datetime,
    ) -> SignedEnvelope:
        """Sign an original versioned representation before any local migration."""
        digest = content_digest(payload)
        return SignedEnvelope(
            payload_type=payload_type,
            payload=dict(payload),
            payload_digest=digest,
            signature=self.sign_digest(digest, payload_type=payload_type, created_at=created_at),
        )


def _signature_message(record: SignatureRecord) -> bytes:
    statement = {
        "algorithm": record.algorithm,
        "created_at": record.created_at,
        "key_id": record.key_id,
        "payload_type": record.payload_type,
        "signed_digest": record.signed_digest,
    }
    return b"AXIOM-EVIDENCE-SIGNATURE-V1\0" + canonical_bytes(statement)


def verify_signature_record(record: SignatureRecord, trust_store: TrustStore) -> None:
    public_key = trust_store.resolve(record.key_id, record.created_at)
    try:
        signature = base64.b64decode(record.signature_b64, validate=True)
    except ValueError as exc:
        raise SignatureVerificationError("signature is not valid base64") from exc
    try:
        public_key.verify(signature, _signature_message(record))
    except (InvalidSignature, ValueError) as exc:
        raise SignatureVerificationError("Ed25519 signature verification failed") from exc


def verify_envelope(
    envelope: SignedEnvelope,
    expected_type: type[StrictContract],
    trust_store: TrustStore,
) -> StrictContract:
    if envelope.payload_type != expected_type.__name__:
        raise SignatureVerificationError(
            f"payload type mismatch: expected {expected_type.__name__}, got {envelope.payload_type}"
        )
    if envelope.signature.payload_type != envelope.payload_type:
        raise SignatureVerificationError("signature payload_type does not match envelope payload_type")
    # Authenticate the producer's exact historical representation. Migration
    # is a local interpretation step and must never alter the signed bytes.
    actual_digest = content_digest(envelope.payload)
    if actual_digest != envelope.payload_digest:
        raise SignatureVerificationError("payload digest mismatch")
    if envelope.signature.signed_digest != envelope.payload_digest:
        raise SignatureVerificationError("signature is bound to a different digest")
    verify_signature_record(envelope.signature, trust_store)
    return expected_type.from_versioned_payload(envelope.payload)


def verify_disposition_timing(
    envelope: SignedEnvelope,
    event: DispositionEvent,
    *,
    received_at: datetime,
    trust_store: TrustStore,
) -> None:
    """Bind signer time, event time, and the local ledger's trusted receipt time."""
    if received_at.tzinfo is None or received_at.utcoffset() is None:
        raise SignatureVerificationError("ledger receipt time must be timezone-aware")
    signed_at = envelope.signature.created_at.astimezone(timezone.utc)
    occurred_at = event.occurred_at.astimezone(timezone.utc)
    receipt = received_at.astimezone(timezone.utc)
    if abs(signed_at - occurred_at) > MAX_SIGNATURE_EVENT_SKEW:
        raise SignatureVerificationError(
            "signature created_at does not match disposition occurred_at within clock-skew policy"
        )
    if occurred_at > receipt + MAX_EVENT_FUTURE_SKEW:
        raise SignatureVerificationError("disposition occurred_at is too far ahead of ledger receipt time")
    trust_store.require_trusted_at_receipt(envelope.signature.key_id, receipt)
