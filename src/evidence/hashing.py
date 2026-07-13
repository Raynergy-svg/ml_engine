"""SHA-256 content addressing for canonical evidence payloads and raw artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, BinaryIO

from .canonical import canonical_bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_digest(value: Any) -> str:
    """Digest a structured value using AXIOM Canonical JSON v1."""
    return sha256_bytes(canonical_bytes(value))


def sha256_stream(stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return sha256_stream(handle)
