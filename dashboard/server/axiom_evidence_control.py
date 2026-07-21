"""Evidence-backed dashboard control plane for AXIOM risk-policy governance.

This replaces the legacy constant-header ``axiom_training_control`` write path.
The dashboard may *request* two local actions, but every governance decision is
made by the immutable evidence layer (``src.evidence``), never by this web tier:

* ``run``     — execute the risk-target evidence slice. The producer / local
  importer / independent verifier are distinct server-held signing identities;
  the disposition chain ends at QUARANTINED or REJECTED. This path has **no
  operator or promotion authority** and can never reach OPERATOR_APPROVED /
  CHAMPION.
* ``promote`` — advance a quarantined package one governed step. Promotion is an
  operator-authority action. The web tier holds **no operator private key**: the
  operator signs the exact ``DispositionEvent`` off-server, and this module only
  relays that signed envelope to ``EvidenceStore.append_disposition``, which
  re-derives the signature, the transition policy, and the authority registry.
  A web endpoint therefore cannot mint an approval.

Both actions are gated by a real authorization primitive — an Ed25519 *operator
request credential* bound to the exact action + subject, with issued/expiry
window and a persistent single-use nonce ledger against replay. Any missing or
invalid credential fails closed (roadmap §16).
"""
from __future__ import annotations

import base64
import copy
import fcntl
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole, SignedEnvelope
from src.evidence.hashing import sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustedKey, TrustStore
from src.evidence.store import EvidenceStore
from src.evidence.transition_policy import AuthorityRegistry
from src.evidence.risk_target.evaluation import EvaluationParams, evaluate_partitions
from src.evidence.risk_target.slice import (
    SliceIdentities,
    run_risk_target_evidence_slice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Domain separation: an operator request credential is NOT interchangeable with
# an evidence-layer disposition signature (which uses a different prefix).
OPERATOR_REQUEST_DOMAIN = b"AXIOM-DASHBOARD-OPERATOR-REQUEST-V1\0"

# A credential authorizes a single action within a bounded, near-term window.
MAX_CREDENTIAL_WINDOW = timedelta(seconds=300)
CLOCK_SKEW = timedelta(seconds=30)

DEFAULT_EVIDENCE_ROOT = REPO_ROOT / "trained_data" / "evidence"
DEFAULT_OPERATOR_TRUST = REPO_ROOT / "trained_data" / "axiom" / "operator_trust.json"
DEFAULT_NONCE_LEDGER = REPO_ROOT / "trained_data" / "axiom" / "operator_nonces.jsonl"
DEFAULT_KEYSTORE = REPO_ROOT / "trained_data" / "axiom" / "server_keystore"

_SERVER_ROLES = (
    AuthorityRole.PRODUCER,
    AuthorityRole.LOCAL_IMPORTER,
    AuthorityRole.INDEPENDENT_VERIFIER,
)
_SERVER_ACTORS = {
    AuthorityRole.PRODUCER: "axiom-dashboard-producer",
    AuthorityRole.LOCAL_IMPORTER: "axiom-dashboard-importer",
    AuthorityRole.INDEPENDENT_VERIFIER: "axiom-dashboard-verifier",
}


class AuthorizationError(Exception):
    """A credential is missing, malformed, unauthorized, expired, or replayed."""


class OperatorTrustUnavailable(Exception):
    """No operator trust anchor is configured — the control plane fails closed."""


def _require_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AuthorizationError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_ts(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise AuthorizationError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuthorizationError(f"{field} is not a valid timestamp") from exc
    return _require_utc(parsed, field)


@dataclass(frozen=True)
class OperatorCredential:
    """An operator-signed authorization for exactly one action + subject."""

    action: str
    subject: str
    nonce: str
    issued_at: datetime
    expires_at: datetime
    key_id: str
    signature_b64: str

    def statement_bytes(self) -> bytes:
        """The exact bytes the operator signs — order-independent, canonical."""
        statement = {
            "action": self.action,
            "subject": self.subject,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "key_id": self.key_id,
        }
        return OPERATOR_REQUEST_DOMAIN + canonical_bytes(statement)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OperatorCredential":
        if not isinstance(data, Mapping):
            raise AuthorizationError("credential must be an object")
        required = ("action", "subject", "nonce", "issued_at",
                    "expires_at", "key_id", "signature_b64")
        missing = [key for key in required if key not in data]
        if missing:
            raise AuthorizationError(f"credential missing fields: {', '.join(missing)}")
        for key in ("action", "subject", "nonce", "key_id", "signature_b64"):
            if not isinstance(data[key], str) or not data[key]:
                raise AuthorizationError(f"credential field {key!r} must be a non-empty string")
        return cls(
            action=data["action"],
            subject=data["subject"],
            nonce=data["nonce"],
            issued_at=_parse_ts(data["issued_at"], "issued_at"),
            expires_at=_parse_ts(data["expires_at"], "expires_at"),
            key_id=data["key_id"],
            signature_b64=data["signature_b64"],
        )


@dataclass(frozen=True)
class TrustedOperatorKey:
    """The operator's public trust anchor. No private key ever lives here."""

    actor_id: str
    key_id: str
    public_key: Ed25519PublicKey
    valid_from: datetime
    valid_until: datetime | None = None

    def permits(self, at: datetime) -> bool:
        instant = at.astimezone(timezone.utc)
        if instant < self.valid_from.astimezone(timezone.utc):
            return False
        if self.valid_until is not None and instant >= self.valid_until.astimezone(timezone.utc):
            return False
        return True

    def as_trusted_key(self) -> TrustedKey:
        return TrustedKey(
            key_id=self.key_id,
            public_key=self.public_key,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
        )


def load_operator_trust(path: str | Path = DEFAULT_OPERATOR_TRUST) -> TrustedOperatorKey:
    """Load the operator trust anchor, or fail closed if absent/invalid."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OperatorTrustUnavailable(f"no operator trust anchor at {path}") from exc
    except (OSError, ValueError) as exc:
        raise OperatorTrustUnavailable(f"unreadable operator trust anchor: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise OperatorTrustUnavailable("operator trust anchor must be an object")
    try:
        public_raw = base64.b64decode(str(raw["public_key_b64"]), validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(public_raw)
        actor_id = str(raw["actor_id"])
        key_id = str(raw["key_id"])
    except (KeyError, ValueError, TypeError) as exc:
        raise OperatorTrustUnavailable(f"malformed operator trust anchor: {exc}") from exc
    if not actor_id or not key_id:
        raise OperatorTrustUnavailable("operator trust anchor needs actor_id and key_id")
    # Bind the declared key_id to the actual key material.
    expected = _ed25519_key_id(public_key)
    if key_id != expected:
        raise OperatorTrustUnavailable("operator key_id does not match its public key")
    valid_from = _parse_ts(raw.get("valid_from", "1970-01-01T00:00:00Z"), "valid_from")
    valid_until = None
    if raw.get("valid_until"):
        valid_until = _parse_ts(raw["valid_until"], "valid_until")
    return TrustedOperatorKey(
        actor_id=actor_id, key_id=key_id, public_key=public_key,
        valid_from=valid_from, valid_until=valid_until,
    )


def _ed25519_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return f"ed25519:{sha256_bytes(raw)[:32]}"


class NonceLedger:
    """A durable single-use nonce store — the request-layer replay guard.

    Entries carry their credential's expiry so the ledger self-prunes and stays
    bounded (credentials live at most ``MAX_CREDENTIAL_WINDOW``). All access is
    serialized with an exclusive advisory lock so concurrent workers cannot both
    consume the same nonce.
    """

    def __init__(self, path: str | Path = DEFAULT_NONCE_LEDGER) -> None:
        self._path = Path(path)

    def consume(self, nonce: str, *, expires_at: datetime, now: datetime) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path(), "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                live = self._load_live(now)
                if nonce in live:
                    raise AuthorizationError("credential nonce has already been used")
                live[nonce] = expires_at
                self._rewrite(live)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _lock_path(self) -> Path:
        return self._path.with_suffix(self._path.suffix + ".lock")

    def _load_live(self, now: datetime) -> dict[str, datetime]:
        live: dict[str, datetime] = {}
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return live
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                nonce = str(row["nonce"])
                expires = _parse_ts(row["expires_at"], "expires_at")
            except (ValueError, KeyError, TypeError, AuthorizationError):
                continue
            if expires > now:
                live[nonce] = expires
        return live

    def _rewrite(self, live: Mapping[str, datetime]) -> None:
        lines = [
            json.dumps({"nonce": nonce, "expires_at": expires.astimezone(timezone.utc)
                        .isoformat().replace("+00:00", "Z")}, sort_keys=True)
            for nonce, expires in sorted(live.items())
        ]
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".nonces.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + ("\n" if lines else ""))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


class OperatorAuthorizer:
    """Verify operator request credentials, fail closed on any defect."""

    def __init__(
        self,
        trust: TrustedOperatorKey,
        nonces: NonceLedger,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._trust = trust
        self._nonces = nonces
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def authorize(
        self, credential: OperatorCredential, *, action: str, subject_digest: str
    ) -> None:
        now = self._clock().astimezone(timezone.utc)
        if credential.action != action:
            raise AuthorizationError("credential action does not match the request")
        if credential.subject != subject_digest:
            raise AuthorizationError("credential is not bound to this request payload")
        if credential.key_id != self._trust.key_id:
            raise AuthorizationError("credential signer is not the trusted operator key")
        if not self._trust.permits(credential.issued_at):
            raise AuthorizationError("operator key was not trusted when the credential was issued")
        if credential.expires_at <= credential.issued_at:
            raise AuthorizationError("credential expiry must follow its issue time")
        if credential.expires_at - credential.issued_at > MAX_CREDENTIAL_WINDOW:
            raise AuthorizationError("credential validity window is too long")
        if now < credential.issued_at - CLOCK_SKEW:
            raise AuthorizationError("credential was issued in the future")
        if now >= credential.expires_at:
            raise AuthorizationError("credential has expired")
        try:
            signature = base64.b64decode(credential.signature_b64, validate=True)
        except ValueError as exc:
            raise AuthorizationError("credential signature is not valid base64") from exc
        try:
            self._trust.public_key.verify(signature, credential.statement_bytes())
        except InvalidSignature as exc:
            raise AuthorizationError("operator signature verification failed") from exc
        # Consume the nonce LAST — only a fully valid credential burns a nonce.
        self._nonces.consume(credential.nonce, expires_at=credential.expires_at, now=now)


def build_server_identities(
    now: datetime,
    operator: TrustedOperatorKey,
    *,
    signers: Mapping[AuthorityRole, Ed25519Signer] | None = None,
) -> SliceIdentities:
    """Register the three server signing identities plus the operator anchor.

    The operator public key is added to trust and registered as OPERATOR so the
    store can verify operator-signed disposition events — but the operator's
    private key is never present here.
    """
    signers = dict(signers or {role: Ed25519Signer.generate() for role in _SERVER_ROLES})
    trust = TrustStore()
    authorities = AuthorityRegistry()
    valid_from = now - timedelta(days=1)
    for role in _SERVER_ROLES:
        signer = signers[role]
        trust.add(signer.trusted_key(valid_from=valid_from))
        authorities.register(actor_id=_SERVER_ACTORS[role], role=role, key_ids=(signer.key_id,))
    trust.add(operator.as_trusted_key())
    authorities.register(
        actor_id=operator.actor_id,
        role=AuthorityRole.OPERATOR,
        key_ids=(operator.key_id,),
    )
    return SliceIdentities(
        signers=signers, actors=dict(_SERVER_ACTORS),
        trust_store=trust, authorities=authorities,
    )


def _load_or_create_signers(
    keystore: Path, now: datetime
) -> dict[AuthorityRole, Ed25519Signer]:
    keystore.mkdir(parents=True, exist_ok=True)
    signers: dict[AuthorityRole, Ed25519Signer] = {}
    for role in _SERVER_ROLES:
        pem_path = keystore / f"{role.value}.pem"
        if pem_path.exists():
            signers[role] = Ed25519Signer.from_private_pem(pem_path.read_bytes())
            continue
        signer = Ed25519Signer.generate()
        try:
            fd = os.open(str(pem_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # A concurrent worker created it first — adopt the persisted key.
            signers[role] = Ed25519Signer.from_private_pem(pem_path.read_bytes())
            continue
        with os.fdopen(fd, "wb") as handle:
            handle.write(signer.private_pem())
            handle.flush()
            os.fsync(handle.fileno())
        signers[role] = signer
    return signers


class EvidenceControlPlane:
    """Authorize dashboard requests, then drive the governed evidence layer."""

    def __init__(
        self,
        *,
        store: EvidenceStore,
        identities: SliceIdentities,
        authorizer: OperatorAuthorizer,
    ) -> None:
        self._store = store
        self._identities = identities
        self._authorizer = authorizer

    @classmethod
    def from_config(
        cls,
        *,
        evidence_root: str | Path | None = None,
        operator_trust_path: str | Path | None = None,
        nonce_ledger_path: str | Path | None = None,
        keystore: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> "EvidenceControlPlane":
        """Build a persistent control plane, failing closed without operator trust."""
        clock = clock or (lambda: datetime.now(timezone.utc))
        now = clock().astimezone(timezone.utc)
        operator = load_operator_trust(
            operator_trust_path
            or os.environ.get("AXIOM_OPERATOR_TRUST", DEFAULT_OPERATOR_TRUST)
        )
        keystore = Path(keystore or os.environ.get("AXIOM_KEYSTORE", DEFAULT_KEYSTORE))
        signers = _load_or_create_signers(keystore, now)
        identities = build_server_identities(now, operator, signers=signers)
        root = Path(
            evidence_root or os.environ.get("AXIOM_EVIDENCE_ROOT", DEFAULT_EVIDENCE_ROOT)
        )
        store = EvidenceStore(
            root,
            trust_store=identities.trust_store,
            authorities=identities.authorities,
        )
        nonces = NonceLedger(
            nonce_ledger_path
            or os.environ.get("AXIOM_NONCE_LEDGER", DEFAULT_NONCE_LEDGER)
        )
        authorizer = OperatorAuthorizer(operator, nonces, clock=clock)
        return cls(store=store, identities=identities, authorizer=authorizer)

    @property
    def store(self) -> EvidenceStore:
        return self._store

    def prepare_run(
        self,
        credential: OperatorCredential,
        *,
        request_body: Mapping[str, Any],
        partitions: Mapping[str, bytes],
        slice_kwargs: Mapping[str, Any],
        params: EvaluationParams | None = None,
        evaluator: Callable[..., Any] = evaluate_partitions,
        on_authorized: Callable[[str], None] | None = None,
    ) -> Callable[[], dict[str, Any]]:
        """Authorize now and return a single-use, authority-free work item.

        The operator credential and nonce are consumed synchronously before a
        caller can acknowledge or queue the request.  The returned callable
        holds only the already-authorized evidence inputs; it cannot authorize
        another action, promote a package, or be executed twice.
        """
        authorized_request = copy.deepcopy(dict(request_body))
        authorized_partitions = {str(key): bytes(value) for key, value in partitions.items()}
        authorized_kwargs = copy.deepcopy(dict(slice_kwargs))
        authorized_params = copy.deepcopy(params)
        subject = sha256_bytes(canonical_bytes(authorized_request))
        self._authorizer.authorize(credential, action="run", subject_digest=subject)
        if on_authorized is not None:
            on_authorized(subject)
        execution_lock = threading.Lock()
        executed = False

        def execute() -> dict[str, Any]:
            nonlocal executed
            with execution_lock:
                if executed:
                    raise AuthorizationError("authorized run work item has already been executed")
                executed = True
            result = run_risk_target_evidence_slice(
                self._store,
                self._identities,
                authorized_partitions,
                params=authorized_params,
                evaluator=evaluator,
                **authorized_kwargs,
            )
            return {
                "action": "run",
                "job_manifest_digest": result.job_envelope.payload_digest,
                "job_manifest": {
                    key: result.job_envelope.payload[key]
                    for key in (
                        "job_id",
                        "git_commit",
                        "container_digest",
                        "configuration_digest",
                        "feature_pipeline_version",
                        "resource_class",
                        "assigned_unit",
                        "trial_budget",
                    )
                },
                "outcomes": {
                    lane: {
                        "state": outcome.final_state.value,
                        "package_digest": outcome.package_digest,
                        "decision": outcome.verdict.decision.value,
                    }
                    for lane, outcome in result.outcomes.items()
                },
            }

        return execute

    def run(
        self,
        credential: OperatorCredential,
        *,
        request_body: Mapping[str, Any],
        partitions: Mapping[str, bytes],
        slice_kwargs: Mapping[str, Any],
        params: EvaluationParams | None = None,
        evaluator: Callable[..., Any] = evaluate_partitions,
        on_authorized: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Compatibility wrapper: authorize and execute in the caller thread."""
        work_item = self.prepare_run(
            credential,
            request_body=request_body,
            partitions=partitions,
            slice_kwargs=slice_kwargs,
            params=params,
            evaluator=evaluator,
            on_authorized=on_authorized,
        )
        return work_item()

    def promote(
        self, credential: OperatorCredential, *, disposition: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Authorize, then RELAY one operator-signed disposition transition.

        The web tier never signs the transition: it forwards the operator's own
        signed envelope to the store, which re-derives signature, transition
        policy, and operator authority. Skipping SHADOW (jumping straight to
        OPERATOR_APPROVED) is impossible — that edge is not in the policy.
        """
        # The disposition arrives as JSON-origin data (ISO-string datetimes).
        # Parse it through the JSON validator so the strict contract coerces
        # those strings exactly as the store will when it re-verifies.
        try:
            envelope = SignedEnvelope.model_validate_json(json.dumps(dict(disposition)))
        except Exception as exc:  # noqa: BLE001 - malformed envelope is a rejection
            raise AuthorizationError(f"disposition envelope is malformed: {exc}") from exc
        subject = envelope.payload_digest
        self._authorizer.authorize(credential, action="promote", subject_digest=subject)
        previous = envelope.payload.get("previous_event_digest")
        state = self._store.append_disposition(envelope, expected_head_digest=previous)
        return {
            "action": "promote",
            "package_digest": envelope.payload.get("package_digest"),
            "state": state.state.value,
            "head_event_digest": state.head_event_digest,
        }


__all__ = [
    "AuthorizationError",
    "OperatorTrustUnavailable",
    "OperatorCredential",
    "TrustedOperatorKey",
    "NonceLedger",
    "OperatorAuthorizer",
    "EvidenceControlPlane",
    "build_server_identities",
    "load_operator_trust",
]
