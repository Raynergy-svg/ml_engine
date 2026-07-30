"""Persistent slice-identity tests — real keys, real disk, no mocks.

The property under test: identities created by one process must let a later
process verify every envelope the first one signed (the durable-store replay
contract). All tests use real ``Ed25519Signer`` material against ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from src.evidence.contracts import AuthorityRole, CapabilityProfile
from src.evidence.signing import Ed25519Signer, verify_envelope
from src.evidence.risk_target.manifests import build_capability_profile
from src.evidence.risk_target.persistent_identity import (
    IDENTITY_METADATA_FILENAME,
    IdentityPersistenceError,
    load_or_create_slice_identities,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def test_create_writes_private_keys_0600_and_metadata(tmp_path):
    key_dir = tmp_path / "signing"
    identities = load_or_create_slice_identities(key_dir, now=NOW)

    for role in (
        AuthorityRole.PRODUCER,
        AuthorityRole.LOCAL_IMPORTER,
        AuthorityRole.INDEPENDENT_VERIFIER,
    ):
        key_path = key_dir / f"{role.value}.key"
        assert key_path.exists()
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
        assert len(key_path.read_bytes()) == 32
        assert role in identities.signers

    metadata = json.loads((key_dir / IDENTITY_METADATA_FILENAME).read_text())
    assert set(metadata["roles"]) == {
        "producer", "local_importer", "independent_verifier",
    }
    for role, signer in identities.signers.items():
        assert metadata["roles"][role.value]["key_id"] == signer.key_id


def test_reload_returns_identical_identities(tmp_path):
    key_dir = tmp_path / "signing"
    first = load_or_create_slice_identities(key_dir, now=NOW)
    second = load_or_create_slice_identities(key_dir, now=NOW + timedelta(days=30))

    assert {r: s.key_id for r, s in first.signers.items()} == {
        r: s.key_id for r, s in second.signers.items()
    }
    assert dict(first.actors) == dict(second.actors)


def test_envelope_signed_in_first_process_verifies_in_second(tmp_path):
    """The load-bearing replay property: run-2 trust verifies run-1 signatures."""
    key_dir = tmp_path / "signing"
    run1 = load_or_create_slice_identities(key_dir, now=NOW)
    envelope = run1.producer.sign(build_capability_profile(), created_at=NOW)

    run2 = load_or_create_slice_identities(key_dir, now=NOW + timedelta(days=7))
    payload = verify_envelope(envelope, CapabilityProfile, run2.trust_store)
    assert isinstance(payload, CapabilityProfile)

    # And the authority binding survives too: the producer key is registered
    # to the same actor under the producer role.
    run2.authorities.authorize_identity(
        actor_id=run2.actors[AuthorityRole.PRODUCER],
        role=AuthorityRole.PRODUCER,
        key_id=envelope.signature.key_id,
    )


def test_swapped_key_file_is_refused(tmp_path):
    key_dir = tmp_path / "signing"
    load_or_create_slice_identities(key_dir, now=NOW)

    imposter = Ed25519Signer.generate()
    key_path = key_dir / f"{AuthorityRole.PRODUCER.value}.key"
    key_path.write_bytes(imposter.private_bytes())
    os.chmod(key_path, 0o600)

    with pytest.raises(IdentityPersistenceError, match="swapped key"):
        load_or_create_slice_identities(key_dir, now=NOW)


def test_world_readable_key_is_refused(tmp_path):
    key_dir = tmp_path / "signing"
    load_or_create_slice_identities(key_dir, now=NOW)
    key_path = key_dir / f"{AuthorityRole.PRODUCER.value}.key"
    os.chmod(key_path, 0o644)

    with pytest.raises(IdentityPersistenceError, match="accessible"):
        load_or_create_slice_identities(key_dir, now=NOW)


def test_missing_key_file_is_refused(tmp_path):
    key_dir = tmp_path / "signing"
    load_or_create_slice_identities(key_dir, now=NOW)
    (key_dir / f"{AuthorityRole.INDEPENDENT_VERIFIER.value}.key").unlink()

    with pytest.raises(IdentityPersistenceError, match="missing private key"):
        load_or_create_slice_identities(key_dir, now=NOW)


def test_naive_now_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        load_or_create_slice_identities(tmp_path / "signing", now=datetime(2026, 7, 30))
