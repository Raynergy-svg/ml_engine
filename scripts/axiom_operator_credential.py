#!/usr/bin/env python3
"""Provision AXIOM operator trust and issue short-lived dashboard credentials.

The private key lives in the invoking operator's home directory, never in the
repository or dashboard server.  The dashboard receives only a single-use,
action-and-subject-bound Ed25519 credential.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.server.axiom_evidence_control import (
    MAX_CREDENTIAL_WINDOW,
    OperatorCredential,
    OperatorTrustUnavailable,
    load_operator_trust,
)
from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import sha256_bytes
from src.evidence.signing import Ed25519Signer

DEFAULT_PRIVATE_KEY = Path.home() / ".config" / "axiom" / "operator_ed25519.pem"
DEFAULT_TRUST = REPO_ROOT / "trained_data" / "axiom" / "operator_trust.json"
DEFAULT_DATASET = REPO_ROOT / "market_data" / "factor" / "axiom_risk_target_dataset.json"


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_private(path: Path) -> Ed25519PrivateKey:
    try:
        if path.stat().st_mode & 0o077:
            raise ValueError("operator key must not be accessible by group or other users")
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except FileNotFoundError as exc:
        raise ValueError(f"operator key is absent; run init first: {path}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("operator key is not Ed25519")
    return key


def _validate_issuing_trust(private: Ed25519PrivateKey, trust_path: Path) -> None:
    try:
        anchor = load_operator_trust(trust_path)
    except OperatorTrustUnavailable as exc:
        raise ValueError(f"operator trust anchor is unavailable: {exc}") from exc
    signer = Ed25519Signer(private)
    private_public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    trusted_public = anchor.public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    if signer.key_id != anchor.key_id or private_public != trusted_public:
        raise ValueError("operator key does not match the configured trust anchor")


def _init(private_path: Path, trust_path: Path, actor_id: str, force: bool) -> dict[str, str]:
    if private_path.exists() and not force:
        private = _load_private(private_path)
    else:
        private = Ed25519PrivateKey.generate()
        private_bytes = private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        _atomic_write(private_path, private_bytes, 0o600)
    signer = Ed25519Signer(private)
    public_bytes = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    trust = {
        "actor_id": actor_id,
        "key_id": signer.key_id,
        "public_key_b64": base64.b64encode(public_bytes).decode("ascii"),
        "valid_from": _utc_text(datetime.now(timezone.utc) - timedelta(seconds=30)),
    }
    _atomic_write(
        trust_path,
        (json.dumps(trust, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        0o644,
    )
    return {"key_id": signer.key_id, "private_key": str(private_path), "trust_anchor": str(trust_path)}


def _credential(private: Ed25519PrivateKey, *, action: str, subject: str, ttl: int) -> dict[str, str]:
    if ttl <= 0 or timedelta(seconds=ttl) > MAX_CREDENTIAL_WINDOW:
        raise ValueError(f"ttl must be between 1 and {int(MAX_CREDENTIAL_WINDOW.total_seconds())} seconds")
    now = datetime.now(timezone.utc)
    signer = Ed25519Signer(private)
    unsigned = OperatorCredential(
        action=action,
        subject=subject,
        nonce=secrets.token_urlsafe(24),
        issued_at=now,
        expires_at=now + timedelta(seconds=ttl),
        key_id=signer.key_id,
        signature_b64="unsigned",
    )
    signature = private.sign(unsigned.statement_bytes())
    return {
        "action": unsigned.action,
        "subject": unsigned.subject,
        "nonce": unsigned.nonce,
        "issued_at": _utc_text(unsigned.issued_at),
        "expires_at": _utc_text(unsigned.expires_at),
        "key_id": unsigned.key_id,
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)
    parser.add_argument("--trust-anchor", type=Path, default=DEFAULT_TRUST)
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser("init", help="create/reuse the operator key and publish its public trust anchor")
    initialize.add_argument("--actor-id", default="axiom-local-operator")
    initialize.add_argument("--force", action="store_true")
    run = sub.add_parser("issue-run", help="issue a credential for the exact configured risk-target dataset")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--ttl", type=int, default=120)
    promote = sub.add_parser("issue-promote", help="issue a credential bound to one signed disposition payload digest")
    promote.add_argument("--subject", required=True)
    promote.add_argument("--ttl", type=int, default=120)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            print(json.dumps(_init(args.private_key, args.trust_anchor, args.actor_id, args.force), sort_keys=True))
            return 0
        private = _load_private(args.private_key)
        _validate_issuing_trust(private, args.trust_anchor)
        if args.command == "issue-run":
            descriptor = args.dataset.read_bytes()
            request = {"dataset_sha256": sha256_bytes(descriptor), "lane": "risk_target"}
            subject = sha256_bytes(canonical_bytes(request))
            credential = _credential(private, action="run", subject=subject, ttl=args.ttl)
        else:
            credential = _credential(private, action="promote", subject=args.subject, ttl=args.ttl)
        print(json.dumps(credential, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
