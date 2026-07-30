"""Persistent Ed25519 slice identities for cross-process store replay.

``build_slice_identities`` (slice.py) mints fresh in-process keys, which is
correct for tests but makes a durable ``EvidenceStore`` unverifiable across
processes: the store re-verifies every stored envelope against the supplied
``TrustStore`` on each mutation (``EvidenceStore._index_payload_unlocked``),
so a second run with new ephemeral keys would hit ``StoreCorruptionError``
("untrusted signer key") on every package the first run wrote.

This module persists the three slice identities (producer / local importer /
independent verifier) under a private key directory:

- ``{role}.key`` — raw 32-byte Ed25519 private key, mode 0600, created with
  ``O_CREAT|O_EXCL`` (never overwritten);
- ``identities.json`` — public metadata binding role -> actor_id, key_id and
  the trust interval's ``valid_from`` (persisted so envelopes signed in an
  earlier run remain inside the trusted interval on reload).

The default location, ``trained_data/axiom/signing/``, is covered by the
repo's ``trained_data/axiom/`` .gitignore entry — private keys must never be
committed. On load, each key file's derived key_id is checked against the
metadata and a mismatch fails loud (a swapped key must never silently
re-sign under an old identity).
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from src.evidence.contracts import AuthorityRole
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.transition_policy import AuthorityRegistry

from .slice import _DEFAULT_ACTORS, _SLICE_ROLES, SliceIdentities

IDENTITY_METADATA_FILENAME = "identities.json"
IDENTITY_SCHEMA_VERSION = "1.0.0"


class IdentityPersistenceError(RuntimeError):
    """Persisted identity material is missing, malformed, or inconsistent."""


def _key_path(key_dir: Path, role: AuthorityRole) -> Path:
    return key_dir / f"{role.value}.key"


def _write_private_key(path: Path, signer: Ed25519Signer) -> None:
    """Create the raw private-key file 0600, refusing to overwrite."""
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(signer.private_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_private_key(path: Path) -> Ed25519Signer:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise IdentityPersistenceError(
            f"private key {path} is group/world accessible (mode {oct(mode)}); refusing to use it"
        )
    raw = path.read_bytes()
    if len(raw) != 32:
        raise IdentityPersistenceError(f"private key {path} is not a raw 32-byte Ed25519 key")
    return Ed25519Signer.from_private_bytes(raw)


def load_or_create_slice_identities(
    key_dir: str | Path,
    *,
    now: datetime,
    actors: Mapping[AuthorityRole, str] | None = None,
) -> SliceIdentities:
    """Load the persisted slice identities, creating them on first use.

    First call generates three distinct keys (0600, ``O_EXCL``) plus an
    ``identities.json`` metadata file recording actor_id, key_id and the
    trust ``valid_from`` (``now - 1 day``, persisted so later processes keep
    the original interval). Subsequent calls reload the same identities and
    rebuild an equivalent ``TrustStore``/``AuthorityRegistry``, so envelopes
    signed by any earlier run keep verifying.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    key_dir = Path(key_dir)
    key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata_path = key_dir / IDENTITY_METADATA_FILENAME
    actors = dict(actors or _DEFAULT_ACTORS)

    if metadata_path.exists():
        return _load_existing(key_dir, metadata_path)

    signers = {role: Ed25519Signer.generate() for role in _SLICE_ROLES}
    valid_from = (now - timedelta(days=1)).astimezone(timezone.utc)
    for role, signer in signers.items():
        _write_private_key(_key_path(key_dir, role), signer)
    metadata = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "created_at": now.astimezone(timezone.utc).isoformat(),
        "roles": {
            role.value: {
                "actor_id": actors[role],
                "key_id": signers[role].key_id,
                "valid_from": valid_from.isoformat(),
            }
            for role in _SLICE_ROLES
        },
    }
    temporary = metadata_path.parent / f".{metadata_path.name}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(temporary, metadata_path)
    return _assemble(signers, actors, {role: valid_from for role in _SLICE_ROLES})


def _load_existing(key_dir: Path, metadata_path: Path) -> SliceIdentities:
    try:
        metadata = json.loads(metadata_path.read_text())
        role_entries = metadata["roles"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise IdentityPersistenceError(f"unreadable identity metadata {metadata_path}: {exc}") from exc

    signers: dict[AuthorityRole, Ed25519Signer] = {}
    actors: dict[AuthorityRole, str] = {}
    valid_from_by_role: dict[AuthorityRole, datetime] = {}
    for role in _SLICE_ROLES:
        entry = role_entries.get(role.value)
        if not isinstance(entry, dict):
            raise IdentityPersistenceError(
                f"identity metadata {metadata_path} has no entry for role {role.value!r}"
            )
        key_path = _key_path(key_dir, role)
        if not key_path.exists():
            raise IdentityPersistenceError(f"missing private key file: {key_path}")
        signer = _read_private_key(key_path)
        expected_key_id = entry.get("key_id")
        if signer.key_id != expected_key_id:
            raise IdentityPersistenceError(
                f"key file {key_path} derives key_id {signer.key_id}, but metadata "
                f"records {expected_key_id!r} — refusing to sign under a swapped key"
            )
        try:
            valid_from = datetime.fromisoformat(entry["valid_from"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IdentityPersistenceError(
                f"invalid valid_from for role {role.value!r} in {metadata_path}"
            ) from exc
        if valid_from.tzinfo is None or valid_from.utcoffset() is None:
            raise IdentityPersistenceError(
                f"valid_from for role {role.value!r} must be timezone-aware"
            )
        actor_id = entry.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id:
            raise IdentityPersistenceError(
                f"invalid actor_id for role {role.value!r} in {metadata_path}"
            )
        signers[role] = signer
        actors[role] = actor_id
        valid_from_by_role[role] = valid_from
    return _assemble(signers, actors, valid_from_by_role)


def _assemble(
    signers: Mapping[AuthorityRole, Ed25519Signer],
    actors: Mapping[AuthorityRole, str],
    valid_from_by_role: Mapping[AuthorityRole, datetime],
) -> SliceIdentities:
    trust = TrustStore()
    authorities = AuthorityRegistry()
    for role in _SLICE_ROLES:
        signer = signers[role]
        trust.add(signer.trusted_key(valid_from=valid_from_by_role[role]))
        authorities.register(actor_id=actors[role], role=role, key_ids=(signer.key_id,))
    return SliceIdentities(
        signers=dict(signers),
        actors=dict(actors),
        trust_store=trust,
        authorities=authorities,
    )


__all__ = [
    "IdentityPersistenceError",
    "IDENTITY_METADATA_FILENAME",
    "load_or_create_slice_identities",
]
