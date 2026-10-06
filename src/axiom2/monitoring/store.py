"""Crash-safe operational journal for deterministic candidate monitoring.

This journal intentionally remains separate from the evidence, shadow-execution,
and reconciliation authorities. It reuses canonical hashing conventions while
recording only monitor state and delivery facts.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import os
from pathlib import Path
from typing import Iterator

from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import content_digest

from .contracts import MonitorEvent, MonitorEventKind, MonitorReceipt


class MonitorStoreError(ValueError):
    """Base error for invalid monitor-store input or state."""


class MonitorStoreCorruption(MonitorStoreError):
    """Raised when durable monitor history cannot be trusted."""


class MonitorStoreOrderingError(MonitorStoreError):
    """Raised when event or source ordering would regress."""


class MonitorStore:
    """One immutable, process-locked monitor event stream."""

    _MATERIAL_WAKEUP_KINDS = frozenset(
        {
            MonitorEventKind.ENTRY_TRIGGERED,
            MonitorEventKind.FRESHNESS_FAILURE,
            MonitorEventKind.CONNECTION_LOST,
            MonitorEventKind.CONNECTION_RESTORED,
            MonitorEventKind.THESIS_INVALIDATED,
            MonitorEventKind.CANDIDATE_EXPIRED,
            MonitorEventKind.EXIT_TRIGGERED,
        }
    )

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.events_root = self.root / "events"
        self.lock_path = self.root / ".monitor.lock"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._ensure_directory(self.root)
        self.events_root.mkdir(mode=0o700, exist_ok=True)
        self._ensure_directory(self.events_root)

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise MonitorStoreCorruption(f"invalid monitor directory: {path}")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _identity_payload(event: MonitorEvent) -> dict[str, object]:
        payload = event.model_dump(mode="json")
        payload.pop("sequence", None)
        payload.pop("previous_digest", None)
        return payload

    @classmethod
    def _identity_digest(cls, event: MonitorEvent) -> str:
        return content_digest(cls._identity_payload(event))

    @staticmethod
    def _event_digest(event: MonitorEvent) -> str:
        return content_digest(event)

    def _atomic_create_bytes(self, destination: Path, data: bytes) -> None:
        temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor != -1:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _replay_unlocked(self) -> tuple[MonitorEvent, ...]:
        self._ensure_directory(self.events_root)
        paths = []
        for path in self.events_root.iterdir():
            if path.is_symlink():
                raise MonitorStoreCorruption("symlink in monitor event stream")
            if path.name.startswith(".") and ".tmp-" in path.name:
                continue
            if not path.is_file():
                raise MonitorStoreCorruption("non-file in monitor event stream")
            paths.append(path)
        paths.sort(key=lambda path: path.name)

        rows: list[MonitorEvent] = []
        event_ids: dict[str, str] = {}
        source_watermarks: dict[str, int] = {}
        previous_digest: str | None = None
        previous_received_at: datetime | None = None

        for offset, path in enumerate(paths):
            try:
                receipt = MonitorReceipt.model_validate_json(path.read_bytes(), strict=True)
                raw = canonical_bytes(receipt)
                if raw != path.read_bytes():
                    raise ValueError("noncanonical receipt")
                event = receipt.event
                digest = self._event_digest(event)
                if receipt.event_digest != digest:
                    raise ValueError("event digest mismatch")
                if path.name != f"{event.sequence:020d}-{digest}.json":
                    raise ValueError("event address mismatch")
                if event.sequence != offset:
                    raise ValueError("event sequence mismatch")
                if event.previous_digest != previous_digest:
                    raise ValueError("event previous digest mismatch")
                if event.occurred_at > receipt.received_at:
                    raise ValueError("event occurs after receipt")
                if previous_received_at is not None and receipt.received_at < previous_received_at:
                    raise ValueError("receipt time regression")
                identity = self._identity_digest(event)
                prior_identity = event_ids.get(event.event_id)
                if prior_identity is not None:
                    if prior_identity != identity:
                        raise ValueError("conflicting duplicate event id")
                    raise ValueError("duplicate event id")
                event_ids[event.event_id] = identity
                if event.source_sequence is not None:
                    prior_source = source_watermarks.get(event.candidate_id)
                    if prior_source is not None and event.source_sequence <= prior_source:
                        raise ValueError("source watermark regression")
                    source_watermarks[event.candidate_id] = event.source_sequence
            except MonitorStoreError:
                raise
            except Exception as exc:
                raise MonitorStoreCorruption(f"invalid monitor receipt: {path.name}") from exc
            rows.append(event)
            previous_digest = digest
            previous_received_at = receipt.received_at
        return tuple(rows)

    def replay(self) -> tuple[MonitorEvent, ...]:
        with self._locked():
            return self._replay_unlocked()

    def append(self, event: MonitorEvent, *, received_at: datetime) -> str:
        if received_at.tzinfo is None or received_at.utcoffset() is None:
            raise MonitorStoreError("received_at must be timezone-aware")
        if event.occurred_at > received_at:
            raise MonitorStoreError("event occurs after receipt")

        with self._locked():
            rows = self._replay_unlocked()
            identity = self._identity_digest(event)
            for prior in rows:
                if prior.event_id == event.event_id:
                    if self._identity_digest(prior) != identity:
                        raise MonitorStoreCorruption("conflicting duplicate event id")
                    return self._event_digest(prior)

            if event.source_sequence is not None:
                prior_sources = [
                    row.source_sequence
                    for row in rows
                    if row.candidate_id == event.candidate_id
                    and row.source_sequence is not None
                ]
                if prior_sources and event.source_sequence <= max(prior_sources):
                    raise MonitorStoreOrderingError("source watermark regression")

            stored = event.model_copy(
                update={
                    "sequence": len(rows),
                    "previous_digest": self._event_digest(rows[-1]) if rows else None,
                }
            )
            digest = self._event_digest(stored)
            receipt = MonitorReceipt(
                event=stored,
                received_at=received_at,
                event_digest=digest,
            )
            destination = self.events_root / f"{stored.sequence:020d}-{digest}.json"
            if destination.exists():
                raise MonitorStoreCorruption("immutable monitor address already exists")
            self._atomic_create_bytes(destination, canonical_bytes(receipt))
            return digest

    def head_digest(self) -> str | None:
        rows = self.replay()
        return self._event_digest(rows[-1]) if rows else None

    def latest_received_at(self) -> datetime | None:
        """Return the verified receipt watermark for monotonic deliveries."""
        with self._locked():
            self._replay_unlocked()
            paths = sorted(self.events_root.iterdir(), key=lambda path: path.name)
            if not paths:
                return None
            receipt = MonitorReceipt.model_validate_json(paths[-1].read_bytes(), strict=True)
            return receipt.received_at

    def pending_wakeups(self) -> tuple[MonitorEvent, ...]:
        rows = self.replay()
        delivered = {
            str(row.payload.get("wakeup_id"))
            for row in rows
            if row.kind == MonitorEventKind.RESEARCH_WAKEUP_DELIVERED
        }
        return tuple(
            row
            for row in rows
            if row.kind in self._MATERIAL_WAKEUP_KINDS
            and isinstance(row.payload.get("wakeup_id"), str)
            and str(row.payload["wakeup_id"]) not in delivered
        )
