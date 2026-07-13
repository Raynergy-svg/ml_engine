"""AXIOM Canonical JSON v1.

The format is UTF-8 JSON with sorted keys, no insignificant whitespace, no
non-finite numbers, and Pydantic JSON-mode normalization for contract values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CanonicalizationError(ValueError):
    """Raised when a value has no stable representation in Canonical JSON v1."""


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="json", round_trip=True, warnings=False))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalizationError("naive datetimes are not canonical")
        utc = value.astimezone(timezone.utc)
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        raise CanonicalizationError("Path objects are forbidden; serialize an explicit URI or relative path")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are forbidden")
        if value == 0.0:
            return 0.0
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("canonical object keys must be strings")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    raise CanonicalizationError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the deterministic Canonical JSON v1 representation of *value*."""
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 bytes suitable for hashing and signing."""
    return canonical_json(value).encode("utf-8")
