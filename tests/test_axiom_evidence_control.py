"""Governance tests for the evidence-backed dashboard control plane.

These prove the roadmap §16 boundary for ``dashboard.server.axiom_evidence_control``:

* the legacy constant-header authorization is gone (see the companion app test
  ``dashboard/server/test_axiom_evidence_api.py``);
* unauthorized / unsigned / replayed / expired requests are rejected;
* an authorized request produces only legitimate signed disposition
  transitions, and a web caller can NEVER skip the OPERATOR_APPROVED authority
  rules or forge operator authority.

NO MOCKS: real Ed25519 keys, a real ``EvidenceStore`` on ``tmp_path``, the real
disposition chain, and a real (injected) deterministic evaluator — an ordinary
callable returning real contract objects, exactly as the §12 slice tests do.
"""
from __future__ import annotations

import base64
from datetime import date, datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import (
    AuthorityRole,
    DispositionEvent,
    DispositionState,
    GateResult,
    GateStatus,
    SignedEnvelope,
)
from src.evidence.hashing import sha256_bytes
from src.evidence.signing import Ed25519Signer
from src.evidence.store import EvidenceStore, EvidenceStoreError
from src.evidence.risk_target.models import (
    DRAWDOWN_HEAD_ID,
    VOLATILITY_HEAD_ID,
    HeadResult,
)

from dashboard.server.axiom_evidence_control import (
    AuthorizationError,
    EvidenceControlPlane,
    NonceLedger,
    OperatorAuthorizer,
    OperatorCredential,
    OperatorTrustUnavailable,
    TrustedOperatorKey,
    build_server_identities,
    load_operator_trust,
)

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
SLICE_KW = dict(
    dataset_id="fx-daily-test",
    coverage_start=date(2019, 1, 1),
    coverage_end=date(2026, 1, 1),
    retrieved_at=NOW,
    created_at=NOW,
    git_commit="0" * 40,
)


# --------------------------------------------------------------------------- #
# Real, deterministic evaluator (no mocks)                                     #
# --------------------------------------------------------------------------- #

def _head(head_id: str, lane_id: str, passed: bool, score: float = 0.05) -> HeadResult:
    status = GateStatus.PASS if passed else GateStatus.FAIL
    gates = (
        GateResult(gate_id=f"{head_id}_admissibility", status=status, observed=score,
                   threshold=0.1, reason="synthetic admissibility gate"),
    )
    return HeadResult(
        head_id=head_id, lane_id=lane_id, target=head_id,
        metrics={"score": score, "secondary": score * 2.0},
        metric_tolerances={"score": 1e-9, "secondary": 1e-9},
        gates=gates, passed=passed,
        model_bytes=f"deterministic-model::{head_id}".encode(),
        media_type="application/octet-stream",
        trial_count=1, effective_sample_size=120.0, temporal_holdout="OOS bucket",
        purge_observations=20, embargo_observations=20,
    )


def _fixed_evaluator(vol_passed: bool = True, dd_passed: bool = False):
    def evaluator(partitions, params):
        return (
            _head(VOLATILITY_HEAD_ID, "risk_target_vol", vol_passed),
            _head(DRAWDOWN_HEAD_ID, "risk_target_drawdown", dd_passed),
        )
    return evaluator


def _partitions():
    return {
        "EUR_USD": b"EUR_USD synthetic risk-target partition bytes\n",
        "USD_JPY": b"USD_JPY synthetic risk-target partition bytes\n",
    }


# --------------------------------------------------------------------------- #
# Fixtures: a real control plane with a distinct operator key                  #
# --------------------------------------------------------------------------- #

class _Operator:
    """Holds the operator PRIVATE key — simulating the off-server operator tool.

    The server side (control plane / store) only ever sees the public key.
    """

    def __init__(self) -> None:
        self._private = Ed25519PrivateKey.generate()
        self.signer = Ed25519Signer(self._private)
        self.anchor = TrustedOperatorKey(
            actor_id="axiom-operator",
            key_id=self.signer.key_id,
            public_key=self._private.public_key(),
            valid_from=NOW - timedelta(days=1),
        )

    def credential(self, *, action: str, subject: str, nonce: str,
                   issued_at: datetime | None = None,
                   expires_at: datetime | None = None) -> OperatorCredential:
        issued_at = issued_at or NOW
        expires_at = expires_at or (NOW + timedelta(seconds=60))
        unsigned = OperatorCredential(
            action=action, subject=subject, nonce=nonce,
            issued_at=issued_at, expires_at=expires_at,
            key_id=self.anchor.key_id, signature_b64="unsigned",
        )
        signature = self._private.sign(unsigned.statement_bytes())
        return OperatorCredential(
            action=action, subject=subject, nonce=nonce,
            issued_at=issued_at, expires_at=expires_at,
            key_id=self.anchor.key_id,
            signature_b64=base64.b64encode(signature).decode("ascii"),
        )

    def sign_event(self, event: DispositionEvent, *, at: datetime) -> SignedEnvelope:
        return self.signer.sign(event, created_at=at)


def _build_plane(tmp_path, *, operator: _Operator, clock=None):
    clock = clock or (lambda: NOW)
    identities = build_server_identities(NOW, operator.anchor)
    store = EvidenceStore(
        tmp_path / "evidence",
        trust_store=identities.trust_store,
        authorities=identities.authorities,
        trusted_clock=lambda: NOW,
    )
    nonces = NonceLedger(tmp_path / "nonces.jsonl")
    authorizer = OperatorAuthorizer(operator.anchor, nonces, clock=clock)
    plane = EvidenceControlPlane(store=store, identities=identities, authorizer=authorizer)
    return plane, store


def _run_subject(request_body) -> str:
    return sha256_bytes(canonical_bytes(dict(request_body)))


def _quarantine_one(plane, operator, *, request_body=None, nonce="run-1"):
    request_body = request_body if request_body is not None else {"job": "risk-target"}
    cred = operator.credential(
        action="run", subject=_run_subject(request_body), nonce=nonce,
    )
    return plane.run(
        cred,
        request_body=request_body,
        partitions=_partitions(),
        slice_kwargs=SLICE_KW,
        evaluator=_fixed_evaluator(vol_passed=True, dd_passed=False),
    )


def _shadow_envelope(store, operator, package_digest, *, to_state=DispositionState.SHADOW,
                     authority=AuthorityRole.OPERATOR, signer=None):
    ledger = store.load_ledger(package_digest)
    head = ledger[-1].payload_digest
    at = NOW + timedelta(seconds=6)
    event = DispositionEvent(
        package_digest=package_digest,
        sequence=len(ledger),
        previous_event_digest=head,
        from_state=DispositionState.QUARANTINED,
        to_state=to_state,
        authority=authority,
        actor_id=operator.anchor.actor_id,
        signer_key_id=(signer or operator.signer).key_id,
        occurred_at=at,
        reason="operator promotes candidate to shadow",
    )
    return (signer or operator.signer).sign(event, created_at=at)


# --------------------------------------------------------------------------- #
# Authorization: unauthorized / unsigned / replayed / expired                  #
# --------------------------------------------------------------------------- #

def test_run_authorizes_and_reaches_quarantine(tmp_path):
    operator = _Operator()
    plane, _ = _build_plane(tmp_path, operator=operator)
    result = _quarantine_one(plane, operator)
    outcomes = result["outcomes"]
    assert outcomes["risk_target_vol"]["state"] == "QUARANTINED"
    assert outcomes["risk_target_drawdown"]["state"] == "REJECTED"


def test_run_rejects_unsigned_credential(tmp_path):
    operator = _Operator()
    plane, _ = _build_plane(tmp_path, operator=operator)
    body = {"job": "risk-target"}
    forged = OperatorCredential(
        action="run", subject=_run_subject(body), nonce="n",
        issued_at=NOW, expires_at=NOW + timedelta(seconds=60),
        key_id=operator.anchor.key_id,
        signature_b64=base64.b64encode(b"not a real signature" * 4).decode("ascii"),
    )
    with pytest.raises(AuthorizationError, match="signature verification failed"):
        plane.run(forged, request_body=body, partitions=_partitions(),
                  slice_kwargs=SLICE_KW, evaluator=_fixed_evaluator())


def test_run_rejects_credential_signed_by_a_non_operator_key(tmp_path):
    operator = _Operator()
    plane, _ = _build_plane(tmp_path, operator=operator)
    body = {"job": "risk-target"}
    attacker = _Operator()  # a different key entirely
    cred = attacker.credential(action="run", subject=_run_subject(body), nonce="n")
    # key_id belongs to the attacker, not the trusted operator anchor.
    with pytest.raises(AuthorizationError, match="not the trusted operator key"):
        plane.run(cred, request_body=body, partitions=_partitions(),
                  slice_kwargs=SLICE_KW, evaluator=_fixed_evaluator())


def test_credential_bound_to_a_different_action_is_rejected(tmp_path):
    operator = _Operator()
    plane, _ = _build_plane(tmp_path, operator=operator)
    body = {"job": "risk-target"}
    cred = operator.credential(action="promote", subject=_run_subject(body), nonce="n")
    with pytest.raises(AuthorizationError, match="action does not match"):
        plane.run(cred, request_body=body, partitions=_partitions(),
                  slice_kwargs=SLICE_KW, evaluator=_fixed_evaluator())


def test_credential_bound_to_a_different_subject_is_rejected(tmp_path):
    operator = _Operator()
    plane, _ = _build_plane(tmp_path, operator=operator)
    cred = operator.credential(action="run", subject=sha256_bytes(b"other"), nonce="n")
    with pytest.raises(AuthorizationError, match="not bound to this request"):
        plane.run(cred, request_body={"job": "risk-target"}, partitions=_partitions(),
                  slice_kwargs=SLICE_KW, evaluator=_fixed_evaluator())


def test_replayed_credential_is_rejected(tmp_path):
    operator = _Operator()
    plane, _ = _build_plane(tmp_path, operator=operator)
    body = {"job": "risk-target"}
    cred = operator.credential(action="run", subject=_run_subject(body), nonce="reused")
    plane.run(cred, request_body=body, partitions=_partitions(),
              slice_kwargs=SLICE_KW, evaluator=_fixed_evaluator())
    with pytest.raises(AuthorizationError, match="already been used"):
        plane.run(cred, request_body=body, partitions=_partitions(),
                  slice_kwargs=SLICE_KW, evaluator=_fixed_evaluator())


def test_expired_credential_is_rejected(tmp_path):
    operator = _Operator()
    later = {"t": NOW + timedelta(seconds=120)}
    plane, _ = _build_plane(tmp_path, operator=operator, clock=lambda: later["t"])
    body = {"job": "risk-target"}
    cred = operator.credential(
        action="run", subject=_run_subject(body), nonce="n",
        issued_at=NOW, expires_at=NOW + timedelta(seconds=60),
    )
    with pytest.raises(AuthorizationError, match="expired"):
        plane.run(cred, request_body=body, partitions=_partitions(),
                  slice_kwargs=SLICE_KW, evaluator=_fixed_evaluator())


def test_overlong_validity_window_is_rejected(tmp_path):
    operator = _Operator()
    plane, _ = _build_plane(tmp_path, operator=operator)
    body = {"job": "risk-target"}
    cred = operator.credential(
        action="run", subject=_run_subject(body), nonce="n",
        issued_at=NOW, expires_at=NOW + timedelta(hours=1),
    )
    with pytest.raises(AuthorizationError, match="window is too long"):
        plane.run(cred, request_body=body, partitions=_partitions(),
                  slice_kwargs=SLICE_KW, evaluator=_fixed_evaluator())


# --------------------------------------------------------------------------- #
# Promotion: legitimate signed transitions only                               #
# --------------------------------------------------------------------------- #

def test_authorized_promote_advances_quarantine_to_shadow(tmp_path):
    operator = _Operator()
    plane, store = _build_plane(tmp_path, operator=operator)
    result = _quarantine_one(plane, operator)
    package_digest = result["outcomes"]["risk_target_vol"]["package_digest"]

    envelope = _shadow_envelope(store, operator, package_digest)
    cred = operator.credential(
        action="promote", subject=envelope.payload_digest, nonce="promote-1",
    )
    outcome = plane.promote(cred, disposition=envelope.model_dump(mode="json"))
    assert outcome["state"] == "SHADOW"

    ledger = store.load_ledger(package_digest)
    from src.evidence.signing import verify_envelope
    states = [verify_envelope(e, DispositionEvent, store.trust_store).to_state for e in ledger]
    assert states[-1] == DispositionState.SHADOW


def test_promote_cannot_skip_to_operator_approved(tmp_path):
    operator = _Operator()
    plane, store = _build_plane(tmp_path, operator=operator)
    package_digest = _quarantine_one(plane, operator)["outcomes"]["risk_target_vol"][
        "package_digest"
    ]
    # A well-formed, operator-signed event that tries to jump straight past
    # SHADOW. The store's transition policy has no QUARANTINED->OPERATOR_APPROVED
    # edge, so the chain is refused — the web tier cannot skip the ladder.
    envelope = _shadow_envelope(
        store, operator, package_digest, to_state=DispositionState.OPERATOR_APPROVED,
    )
    cred = operator.credential(
        action="promote", subject=envelope.payload_digest, nonce="skip",
    )
    with pytest.raises(EvidenceStoreError):
        plane.promote(cred, disposition=envelope.model_dump(mode="json"))
    # The ledger is unchanged: still QUARANTINED.
    from src.evidence.signing import verify_envelope
    ledger = store.load_ledger(package_digest)
    last = verify_envelope(ledger[-1], DispositionEvent, store.trust_store)
    assert last.to_state == DispositionState.QUARANTINED


def test_promote_rejects_forged_operator_authority(tmp_path):
    operator = _Operator()
    plane, store = _build_plane(tmp_path, operator=operator)
    package_digest = _quarantine_one(plane, operator)["outcomes"]["risk_target_vol"][
        "package_digest"
    ]
    # The event CLAIMS operator authority but is signed by the server importer
    # key — trusted for signatures, yet registered as LOCAL_IMPORTER, never as
    # OPERATOR. The authority registry refuses it: merely writing
    # authority=operator into an event grants nothing.
    importer = plane._identities.signers[AuthorityRole.LOCAL_IMPORTER]
    envelope = _shadow_envelope(store, operator, package_digest, signer=importer)
    cred = operator.credential(
        action="promote", subject=envelope.payload_digest, nonce="forge",
    )
    with pytest.raises(EvidenceStoreError):
        plane.promote(cred, disposition=envelope.model_dump(mode="json"))


def test_promote_rejects_credential_not_bound_to_the_transition(tmp_path):
    operator = _Operator()
    plane, store = _build_plane(tmp_path, operator=operator)
    package_digest = _quarantine_one(plane, operator)["outcomes"]["risk_target_vol"][
        "package_digest"
    ]
    envelope = _shadow_envelope(store, operator, package_digest)
    # Credential subject points at a DIFFERENT digest than the disposition.
    cred = operator.credential(
        action="promote", subject=sha256_bytes(b"unrelated"), nonce="mismatch",
    )
    with pytest.raises(AuthorizationError, match="not bound to this request"):
        plane.promote(cred, disposition=envelope.model_dump(mode="json"))


def test_promote_rejects_malformed_envelope(tmp_path):
    operator = _Operator()
    plane, _ = _build_plane(tmp_path, operator=operator)
    cred = operator.credential(action="promote", subject=sha256_bytes(b"x"), nonce="bad")
    with pytest.raises(AuthorizationError, match="malformed"):
        plane.promote(cred, disposition={"not": "an envelope"})


# --------------------------------------------------------------------------- #
# Fail-closed trust configuration + nonce durability                          #
# --------------------------------------------------------------------------- #

def test_missing_operator_trust_fails_closed(tmp_path):
    with pytest.raises(OperatorTrustUnavailable):
        load_operator_trust(tmp_path / "does_not_exist.json")


def test_from_config_fails_closed_without_operator_trust(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_OPERATOR_TRUST", str(tmp_path / "absent.json"))
    monkeypatch.setenv("AXIOM_EVIDENCE_ROOT", str(tmp_path / "evidence"))
    monkeypatch.setenv("AXIOM_KEYSTORE", str(tmp_path / "keystore"))
    monkeypatch.setenv("AXIOM_NONCE_LEDGER", str(tmp_path / "nonces.jsonl"))
    with pytest.raises(OperatorTrustUnavailable):
        EvidenceControlPlane.from_config()


def test_operator_trust_roundtrip_and_key_binding(tmp_path):
    import json

    from cryptography.hazmat.primitives import serialization

    operator = _Operator()
    raw = operator.anchor.public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    good = tmp_path / "trust.json"
    good.write_text(json.dumps({
        "actor_id": operator.anchor.actor_id,
        "key_id": operator.anchor.key_id,
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
    }))
    loaded = load_operator_trust(good)
    assert loaded.key_id == operator.anchor.key_id
    assert loaded.actor_id == operator.anchor.actor_id

    # A key_id that does not match the key material fails closed.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "actor_id": "axiom-operator",
        "key_id": "ed25519:deadbeef",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
    }))
    with pytest.raises(OperatorTrustUnavailable, match="does not match"):
        load_operator_trust(bad)


def test_nonce_ledger_is_durable_across_instances(tmp_path):
    path = tmp_path / "nonces.jsonl"
    NonceLedger(path).consume("abc", expires_at=NOW + timedelta(seconds=60), now=NOW)
    with pytest.raises(AuthorizationError, match="already been used"):
        NonceLedger(path).consume("abc", expires_at=NOW + timedelta(seconds=60), now=NOW)


def test_nonce_ledger_tolerates_malformed_lines(tmp_path):
    path = tmp_path / "nonces.jsonl"
    # A line with a valid expiry but no 'nonce' key must not crash consume().
    path.write_text('{"expires_at": "2099-01-01T00:00:00Z"}\nnot json\n')
    NonceLedger(path).consume("fresh", expires_at=NOW + timedelta(seconds=60), now=NOW)
    with pytest.raises(AuthorizationError, match="already been used"):
        NonceLedger(path).consume("fresh", expires_at=NOW + timedelta(seconds=60), now=NOW)


def test_nonce_ledger_self_prunes_expired_entries(tmp_path):
    path = tmp_path / "nonces.jsonl"
    NonceLedger(path).consume("old", expires_at=NOW + timedelta(seconds=10), now=NOW)
    # Well after expiry the stale nonce has been pruned, so it can appear again
    # (it would already have been rejected at the expiry check upstream).
    future = NOW + timedelta(seconds=120)
    NonceLedger(path).consume("old", expires_at=future + timedelta(seconds=60), now=future)
