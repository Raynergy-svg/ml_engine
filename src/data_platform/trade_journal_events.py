"""Append-only authority behind the legacy trade-journal JSON projection."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import content_digest, sha256_bytes

from .platform import DataIntegrityError


def event_path_for(projection_path: str | Path) -> Path:
    path = Path(projection_path)
    return path.with_name(f"{path.stem}.events.jsonl")


def _entry_key(entry: Mapping[str, Any], position: int) -> str:
    for field in ("trade_id", "transaction_id", "id"):
        value = str(entry.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    identity = {
        "direction": entry.get("direction"),
        "pair": entry.get("pair") or entry.get("instrument"),
        "timestamp": entry.get("timestamp") or entry.get("time"),
    }
    return f"legacy:{content_digest(identity)}:{position}"


def _snapshot_map(entries: Iterable[Mapping[str, Any]]) -> dict[str, tuple[int, dict[str, Any]]]:
    result: dict[str, tuple[int, dict[str, Any]]] = {}
    for position, entry in enumerate(entries):
        value = dict(entry)
        key = _entry_key(value, position)
        if key in result:
            raise DataIntegrityError(f"duplicate trade-journal identity: {key}")
        result[key] = (position, value)
    return result


def replay_events(path: str | Path) -> list[dict[str, Any]]:
    event_path = Path(path)
    if not event_path.exists():
        return []
    state: dict[str, tuple[int, dict[str, Any]]] = {}
    previous_digest: str | None = None
    for expected_sequence, raw in enumerate(event_path.read_bytes().splitlines()):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError("invalid trade-journal event JSON") from exc
        if not isinstance(event, dict) or canonical_bytes(event) != raw:
            raise DataIntegrityError("trade-journal event is not canonical")
        digest = event.get("event_digest")
        body = {key: value for key, value in event.items() if key != "event_digest"}
        if not isinstance(digest, str) or content_digest(body) != digest:
            raise DataIntegrityError("trade-journal event digest mismatch")
        if event.get("sequence") != expected_sequence:
            raise DataIntegrityError("trade-journal event sequence gap")
        if event.get("previous_event_digest") != previous_digest:
            raise DataIntegrityError("trade-journal event chain is broken")
        key = event.get("entry_key")
        operation = event.get("operation")
        if not isinstance(key, str):
            raise DataIntegrityError("trade-journal event lacks entry_key")
        if operation == "UPSERT":
            entry = event.get("entry")
            position = event.get("position")
            if not isinstance(entry, dict) or not isinstance(position, int) or position < 0:
                raise DataIntegrityError("invalid trade-journal UPSERT event")
            if content_digest(entry) != event.get("entry_digest"):
                raise DataIntegrityError("trade-journal entry digest mismatch")
            state[key] = (position, entry)
        elif operation == "DELETE":
            state.pop(key, None)
        else:
            raise DataIntegrityError("unknown trade-journal event operation")
        previous_digest = digest
    ordered = sorted(state.values(), key=lambda item: (item[0], _entry_key(item[1], item[0])))
    return [entry for _, entry in ordered]


def append_snapshot_events(path: str | Path, entries: Iterable[Mapping[str, Any]]) -> int:
    """Append the minimal UPSERT/DELETE delta needed to represent *entries*."""
    event_path = Path(path)
    desired = _snapshot_map(entries)
    current_entries = replay_events(event_path)
    current = _snapshot_map(current_entries)
    raw_lines = event_path.read_bytes().splitlines() if event_path.exists() else []
    previous_digest = None
    if raw_lines:
        previous_digest = json.loads(raw_lines[-1])["event_digest"]
    sequence = len(raw_lines)
    events: list[dict[str, Any]] = []
    occurred_at = datetime.now(timezone.utc)

    for key in sorted(set(current) - set(desired)):
        body = {
            "entry_key": key,
            "occurred_at": occurred_at,
            "operation": "DELETE",
            "previous_event_digest": previous_digest,
            "schema_version": "1.0.0",
            "sequence": sequence,
        }
        digest = content_digest(body)
        events.append({**body, "event_digest": digest})
        previous_digest, sequence = digest, sequence + 1

    for key, (position, entry) in sorted(desired.items(), key=lambda item: item[1][0]):
        if current.get(key) == (position, entry):
            continue
        body = {
            "entry": entry,
            "entry_digest": content_digest(entry),
            "entry_key": key,
            "occurred_at": occurred_at,
            "operation": "UPSERT",
            "position": position,
            "previous_event_digest": previous_digest,
            "schema_version": "1.0.0",
            "sequence": sequence,
        }
        digest = content_digest(body)
        events.append({**body, "event_digest": digest})
        previous_digest, sequence = digest, sequence + 1

    if not events:
        return 0
    event_path.parent.mkdir(parents=True, exist_ok=True)
    existed = event_path.exists()
    with event_path.open("ab") as handle:
        for event in events:
            handle.write(canonical_bytes(event) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if not existed:
        descriptor = os.open(event_path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if replay_events(event_path) != [dict(entry) for entry in entries]:
        raise DataIntegrityError("trade-journal event replay differs from requested snapshot")
    return len(events)


def rebuild_projection(event_path: str | Path, projection_path: str | Path) -> list[dict[str, Any]]:
    """Return authoritative state; caller owns its atomic projection write."""
    del projection_path
    return replay_events(event_path)


def event_file_digest(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())
