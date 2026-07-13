"""Local-compatible partitioned object store with explicit write capabilities."""

from __future__ import annotations

import json
import re
import shutil
import stat
from io import BytesIO
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import JsonValue, ValidationError
import pyarrow as pa
import pyarrow.ipc as arrow_ipc
import pyarrow.parquet as pq

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import DatasetManifest, SignedEnvelope
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.signing import SignatureVerificationError, TrustStore, verify_envelope

from .contracts import (
    FeatureBundle,
    FeatureOutputRef,
    ImmutableObjectRef,
    NormalizedPartitionRef,
    NormalizedVersion,
    RawObjectReceipt,
    StorageDomain,
    StorageFormat,
)
from .io import (
    atomic_create,
    atomic_replace,
    exclusive_lock,
    publish_directory,
    write_new_file,
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,200}$")


class DataPlatformError(RuntimeError):
    """Base error for canonical data-platform operations."""


class DataIntegrityError(DataPlatformError):
    """Stored or submitted bytes do not match their declared identity."""


class SourceWriteConflict(DataPlatformError):
    """An immutable source address already contains different content."""


class ControlStateConflict(DataPlatformError):
    """Small-control-state compare-and-swap failed."""


@dataclass(frozen=True)
class PartitionPayload:
    data: bytes
    row_count: int | None = None

    def __post_init__(self) -> None:
        if self.row_count is not None and self.row_count < 0:
            raise ValueError("row_count must be non-negative")


@dataclass(frozen=True)
class HistorySegment:
    sequence: int
    digest: str
    path: Path


def _require_digest(value: str, field: str) -> None:
    if not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _component(value: str) -> str:
    if not value:
        raise ValueError("storage identity component must not be empty")
    return sha256_bytes(value.encode("utf-8"))[:32]


def _validate_container(
    data: bytes,
    storage_format: StorageFormat,
    *,
    expected_row_count: int | None = None,
    expected_schema_digest: str | None = None,
) -> tuple[int, str]:
    """Parse a complete columnar file and return authoritative metadata."""
    try:
        if storage_format is StorageFormat.PARQUET:
            container = pq.ParquetFile(BytesIO(data))
            row_count = container.metadata.num_rows
            schema = container.schema_arrow
        else:
            container = arrow_ipc.open_file(pa.BufferReader(data))
            row_count = sum(container.get_batch(index).num_rows for index in range(container.num_record_batches))
            schema = container.schema
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise DataIntegrityError(f"invalid {storage_format.value} columnar container") from exc
    schema_digest = sha256_bytes(schema.serialize().to_pybytes())
    if expected_row_count is not None and row_count != expected_row_count:
        raise DataIntegrityError(
            f"declared row_count {expected_row_count} does not match parsed row_count {row_count}"
        )
    if expected_schema_digest is not None and schema_digest != expected_schema_digest:
        raise DataIntegrityError("declared schema_digest does not match parsed columnar schema")
    return row_count, schema_digest


def _iter_history_root(stream_root: Path) -> Iterator[dict[str, JsonValue]]:
    for expected_sequence, path in enumerate(sorted(stream_root.glob("*.jsonl"))):
        prefix = f"{expected_sequence:020d}-"
        if not path.name.startswith(prefix):
            raise DataIntegrityError(f"history sequence gap or fork at {path}")
        raw = path.read_bytes()
        expected_name = f"{expected_sequence:020d}-{sha256_bytes(raw)}.jsonl"
        if not raw.endswith(b"\n") or path.name != expected_name:
            raise DataIntegrityError(f"history segment integrity failure: {path}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"invalid JSONL history segment: {path}") from exc
        if not isinstance(value, dict) or canonical_bytes(value) + b"\n" != raw:
            raise DataIntegrityError(f"non-canonical JSONL history segment: {path}")
        yield value


class DataPlatform:
    """Own the canonical ``axiom-data`` layout and issue scoped capabilities."""

    def __init__(
        self,
        root: str | Path,
        *,
        trust_store: TrustStore,
        snapshot_signer_key_ids: Iterable[str] = (),
    ) -> None:
        self.root = Path(root)
        self.trust_store = trust_store
        self.snapshot_signer_key_ids = frozenset(snapshot_signer_key_ids)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / ".locks" / "data-platform.lock"
        for domain in StorageDomain:
            domain_root = self.root / domain.value
            for tier in ("raw", "normalized", "snapshots", "features", "history"):
                (domain_root / tier).mkdir(parents=True, mode=0o700, exist_ok=True)

    def ingestion_writer(self) -> "IngestionWriter":
        return IngestionWriter(self)

    def training_view(
        self, domain: StorageDomain, *, read_only_root: str | Path
    ) -> "TrainingDataView":
        """Open an exported read-only training root without writer authority."""
        return TrainingDataView(
            read_only_root,
            domain,
            trust_store=self.trust_store,
            snapshot_signer_key_ids=self.snapshot_signer_key_ids,
        )

    def export_training_root(self, destination: str | Path) -> Path:
        """Create a local read-only replica containing no raw source tier."""
        target = Path(destination)
        if target.exists():
            raise FileExistsError(target)
        target.mkdir(parents=True, mode=0o755)
        for domain in StorageDomain:
            domain_target = target / domain.value
            domain_target.mkdir(mode=0o755)
            for tier in ("normalized", "snapshots", "features", "history"):
                source = self._domain_root(domain) / tier
                shutil.copytree(source, domain_target / tier)
        for path in sorted(target.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        target.chmod(0o555)
        return target

    def feature_writer(self, domain: StorageDomain) -> "FeatureWriter":
        return FeatureWriter(self, domain)

    def iter_history(self, domain: StorageDomain, stream_id: str) -> Iterator[dict[str, JsonValue]]:
        """Trusted local-control read path; training workers use exported roots."""
        stream_root = self._domain_root(domain) / "history" / _component(stream_id)
        yield from _iter_history_root(stream_root)

    def control_state_store(
        self,
        root: str | Path,
        *,
        max_bytes: int = 1024 * 1024,
    ) -> "ControlStateStore":
        return ControlStateStore(root, immutable_data_root=self.root, max_bytes=max_bytes)

    def _domain_root(self, domain: StorageDomain) -> Path:
        return self.root / domain.value

    @staticmethod
    def _uri(domain: StorageDomain, *parts: str) -> str:
        return f"axiom-data://{domain.value}/{'/'.join(parts)}"

    def _resolve_uri(self, uri: str, *, expected_domain: StorageDomain | None = None) -> Path:
        parsed = urlsplit(uri)
        if parsed.scheme != "axiom-data" or parsed.query or parsed.fragment:
            raise DataIntegrityError(f"unsupported immutable object URI: {uri!r}")
        try:
            domain = StorageDomain(parsed.netloc)
        except ValueError as exc:
            raise DataIntegrityError(f"unknown storage domain in URI: {uri!r}") from exc
        if expected_domain is not None and domain is not expected_domain:
            raise DataIntegrityError(
                f"URI domain {domain.value!r} does not match expected {expected_domain.value!r}"
            )
        parts = tuple(part for part in parsed.path.split("/") if part)
        if not parts or any(part in {".", ".."} for part in parts):
            raise DataIntegrityError(f"unsafe immutable object URI: {uri!r}")
        path = self._domain_root(domain).joinpath(*parts)
        if not path.resolve().is_relative_to(self._domain_root(domain).resolve()):
            raise DataIntegrityError(f"object URI escapes domain root: {uri!r}")
        return path

    def _read_verified(
        self,
        uri: str,
        *,
        digest: str,
        size_bytes: int,
        expected_domain: StorageDomain,
    ) -> bytes:
        path = self._resolve_uri(uri, expected_domain=expected_domain)
        if path.is_symlink():
            raise DataIntegrityError(f"immutable object must not be a symlink: {uri}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DataIntegrityError(f"immutable object is unavailable: {uri}") from exc
        if len(data) != size_bytes or sha256_bytes(data) != digest:
            raise DataIntegrityError(f"immutable object integrity failure: {uri}")
        return data

    def _load_normalized_manifest(self, path: Path) -> NormalizedVersion:
        try:
            raw = path.read_bytes()
            model = NormalizedVersion.model_validate_json(raw, strict=True)
        except (OSError, ValidationError, ValueError) as exc:
            raise DataIntegrityError(f"invalid normalized manifest at {path}") from exc
        if canonical_bytes(model) != raw:
            raise DataIntegrityError(f"normalized manifest is not canonical: {path}")
        if content_digest(model) != sha256_bytes(raw):
            raise DataIntegrityError("normalized manifest canonical digest is inconsistent")
        self._validate_raw_lineage(model)
        for partition in model.partitions:
            resolved = self._resolve_uri(partition.uri, expected_domain=model.domain)
            if resolved.parent != path.parent / "partitions":
                raise DataIntegrityError("normalized partition URI escapes its immutable version")
            data = self._read_verified(
                partition.uri,
                digest=partition.digest,
                size_bytes=partition.size_bytes,
                expected_domain=model.domain,
            )
            _validate_container(
                data,
                model.storage_format,
                expected_row_count=partition.row_count,
                expected_schema_digest=model.schema_digest,
            )
        return model

    def _validate_raw_lineage(self, model: NormalizedVersion) -> None:
        domain_root = self._domain_root(model.domain)
        for digest in model.source_raw_digests:
            object_path = domain_root / "raw" / "objects" / digest[:2] / digest
            if not object_path.is_file() or object_path.is_symlink():
                raise DataIntegrityError(f"raw lineage object is unavailable: {digest}")
            data = object_path.read_bytes()
            if sha256_bytes(data) != digest:
                raise DataIntegrityError(f"raw lineage object is corrupt: {digest}")
            receipt_root = domain_root / "raw" / "receipts" / digest
            valid_receipt = False
            for receipt_path in sorted(receipt_root.glob("*.json")):
                try:
                    receipt_raw = receipt_path.read_bytes()
                    receipt = RawObjectReceipt.model_validate_json(receipt_raw, strict=True)
                except (OSError, ValidationError, ValueError):
                    continue
                if canonical_bytes(receipt) != receipt_raw:
                    continue
                if (
                    receipt.domain is model.domain
                    and receipt.object.digest == digest
                    and receipt.object.size_bytes == len(data)
                    and receipt.object.uri
                    == self._uri(model.domain, "raw", "objects", digest[:2], digest)
                    and receipt_path.name == f"{content_digest(receipt)}.json"
                ):
                    valid_receipt = True
                    break
            if not valid_receipt:
                raise DataIntegrityError(f"raw lineage receipt is missing or invalid: {digest}")

    def _snapshot_path(self, domain: StorageDomain, snapshot_digest: str) -> Path:
        _require_digest(snapshot_digest, "snapshot_digest")
        return self._domain_root(domain) / "snapshots" / snapshot_digest / "manifest.json"

    def _load_snapshot(self, domain: StorageDomain, snapshot_digest: str) -> DatasetManifest:
        path = self._snapshot_path(domain, snapshot_digest)
        try:
            raw = path.read_bytes()
            envelope = SignedEnvelope.model_validate_json(raw, strict=True)
            if canonical_bytes(envelope) != raw:
                raise DataIntegrityError("snapshot envelope is not canonical")
            if envelope.payload_digest != snapshot_digest:
                raise DataIntegrityError("snapshot address does not match manifest digest")
            if envelope.signature.key_id not in self.snapshot_signer_key_ids:
                raise DataIntegrityError("snapshot signer is not authorized for snapshot publication")
            manifest = verify_envelope(envelope, DatasetManifest, self.trust_store)
        except (OSError, ValidationError, SignatureVerificationError, ValueError) as exc:
            if isinstance(exc, DataIntegrityError):
                raise
            raise DataIntegrityError(f"invalid snapshot {snapshot_digest}") from exc
        if not isinstance(manifest, DatasetManifest):
            raise DataIntegrityError("snapshot payload is not a DatasetManifest")
        if manifest.asset_class.value != domain.value:
            raise DataIntegrityError("snapshot asset_class does not match its storage domain")
        self._validate_snapshot_membership(domain, manifest)
        return manifest

    def _validate_snapshot_membership(
        self, domain: StorageDomain, manifest: DatasetManifest
    ) -> None:
        if not manifest.normalized_manifest_digests:
            raise DataIntegrityError("snapshot must bind at least one normalized manifest digest")
        declared = set(manifest.normalized_manifest_digests)
        discovered: set[str] = set()
        for partition in manifest.partitions:
            path = self._resolve_uri(partition.uri, expected_domain=domain)
            relative = path.relative_to(self._domain_root(domain)).parts
            if not relative or relative[0] != "normalized" or path.parent.name != "partitions":
                raise DataIntegrityError("training snapshots may reference only normalized source data")
            normalized = self._load_normalized_manifest(path.parent.parent / "manifest.json")
            normalized_digest = content_digest(normalized)
            if normalized_digest not in declared:
                raise DataIntegrityError("snapshot partition is not bound to its normalized manifest")
            matches = [item for item in normalized.partitions if item.partition_id == partition.partition_id]
            if len(matches) != 1:
                raise DataIntegrityError("snapshot partition is absent from normalized manifest")
            source = matches[0]
            if (
                source.uri != partition.uri
                or source.digest != partition.digest
                or source.size_bytes != partition.size_bytes
                or source.row_count != partition.row_count
            ):
                raise DataIntegrityError("snapshot partition does not match normalized manifest entry")
            discovered.add(normalized_digest)
        if discovered != declared:
            raise DataIntegrityError("snapshot normalized manifest digest set is incomplete or unused")


class IngestionWriter:
    """Capability permitted to publish immutable source tiers and history."""

    def __init__(self, platform: DataPlatform) -> None:
        self._platform = platform

    def put_raw(
        self,
        domain: StorageDomain,
        data: bytes,
        *,
        source_id: str,
        retrieved_at: datetime,
        media_type: str,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> RawObjectReceipt:
        digest = sha256_bytes(data)
        object_path = self._platform._domain_root(domain) / "raw" / "objects" / digest[:2] / digest
        uri = self._platform._uri(domain, "raw", "objects", digest[:2], digest)
        reference = ImmutableObjectRef(
            uri=uri,
            digest=digest,
            size_bytes=len(data),
            media_type=media_type,
        )
        receipt = RawObjectReceipt(
            domain=domain,
            source_id=source_id,
            retrieved_at=retrieved_at,
            object=reference,
            metadata=dict(metadata or {}),
        )
        receipt_bytes = canonical_bytes(receipt)
        receipt_digest = content_digest(receipt)
        receipt_path = (
            self._platform._domain_root(domain)
            / "raw"
            / "receipts"
            / digest
            / f"{receipt_digest}.json"
        )
        with exclusive_lock(self._platform._lock_path):
            if object_path.is_symlink():
                raise SourceWriteConflict(f"raw content address must not be a symlink: {digest}")
            if object_path.exists():
                existing = object_path.read_bytes()
                if sha256_bytes(existing) != digest or existing != data:
                    raise SourceWriteConflict(f"raw content address conflict: {digest}")
            else:
                atomic_create(object_path, data)
            if receipt_path.exists():
                if receipt_path.read_bytes() != receipt_bytes:
                    raise SourceWriteConflict(f"raw receipt address conflict: {receipt_digest}")
            else:
                atomic_create(receipt_path, receipt_bytes)
        return receipt

    def publish_normalized(
        self,
        domain: StorageDomain,
        *,
        dataset_id: str,
        version_id: str,
        storage_format: StorageFormat,
        schema_digest: str,
        source_raw_digests: tuple[str, ...],
        created_at: datetime,
        partitions: Mapping[str, PartitionPayload],
    ) -> NormalizedVersion:
        _require_digest(schema_digest, "schema_digest")
        if not partitions:
            raise ValueError("at least one normalized partition is required")
        for raw_digest in source_raw_digests:
            _require_digest(raw_digest, "source_raw_digest")
            raw_path = (
                self._platform._domain_root(domain)
                / "raw"
                / "objects"
                / raw_digest[:2]
                / raw_digest
            )
            if not raw_path.is_file() or sha256_bytes(raw_path.read_bytes()) != raw_digest:
                raise DataIntegrityError(f"normalized source raw object is missing or corrupt: {raw_digest}")

        dataset_component = _component(dataset_id)
        version_component = _component(version_id)
        version_root = (
            self._platform._domain_root(domain)
            / "normalized"
            / dataset_component
            / version_component
        )
        refs: list[NormalizedPartitionRef] = []
        partition_paths: dict[str, str] = {}
        for partition_id, payload in sorted(partitions.items()):
            if payload.row_count is None:
                raise DataIntegrityError("normalized partitions require a declared row_count")
            _validate_container(
                payload.data,
                storage_format,
                expected_row_count=payload.row_count,
                expected_schema_digest=schema_digest,
            )
            part_name = f"{_component(partition_id)}.{storage_format.extension}"
            relative = (
                "normalized",
                dataset_component,
                version_component,
                "partitions",
                part_name,
            )
            digest = sha256_bytes(payload.data)
            refs.append(
                NormalizedPartitionRef(
                    partition_id=partition_id,
                    uri=self._platform._uri(domain, *relative),
                    digest=digest,
                    size_bytes=len(payload.data),
                    row_count=payload.row_count,
                )
            )
            if part_name in partition_paths:
                raise DataIntegrityError("normalized partition storage-key collision")
            partition_paths[part_name] = partition_id

        version = NormalizedVersion(
            domain=domain,
            dataset_id=dataset_id,
            version_id=version_id,
            storage_format=storage_format,
            schema_digest=schema_digest,
            source_raw_digests=source_raw_digests,
            created_at=created_at,
            partitions=tuple(refs),
        )
        manifest_bytes = canonical_bytes(version)
        with exclusive_lock(self._platform._lock_path):
            if version_root.exists():
                existing = self._platform._load_normalized_manifest(version_root / "manifest.json")
                if existing != version:
                    raise SourceWriteConflict(
                        f"normalized version identity already published: {dataset_id}/{version_id}"
                    )
                return existing

            temporary = version_root.parent / f".{version_root.name}.tmp-{uuid4().hex}"
            temporary.mkdir(parents=True, mode=0o700)
            try:
                write_new_file(temporary / "manifest.json", manifest_bytes)
                for ref in refs:
                    part_name = Path(urlsplit(ref.uri).path).name
                    payload = partitions[partition_paths[part_name]]
                    write_new_file(temporary / "partitions" / part_name, payload.data)
                publish_directory(temporary, version_root)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return version

    def publish_snapshot(
        self,
        domain: StorageDomain,
        envelope: SignedEnvelope,
    ) -> str:
        if envelope.signature.key_id not in self._platform.snapshot_signer_key_ids:
            raise DataIntegrityError("snapshot signer is not authorized for snapshot publication")
        try:
            manifest = verify_envelope(envelope, DatasetManifest, self._platform.trust_store)
        except (SignatureVerificationError, ValueError) as exc:
            raise DataIntegrityError(f"invalid signed DatasetManifest: {exc}") from exc
        if not isinstance(manifest, DatasetManifest):
            raise DataIntegrityError("snapshot payload is not a DatasetManifest")
        if manifest.asset_class.value != domain.value:
            raise DataIntegrityError("snapshot asset_class does not match its storage domain")
        self._platform._validate_snapshot_membership(domain, manifest)

        snapshot_digest = envelope.payload_digest
        destination = self._platform._snapshot_path(domain, snapshot_digest).parent
        envelope_bytes = canonical_bytes(envelope)
        with exclusive_lock(self._platform._lock_path):
            if destination.exists():
                existing = destination / "manifest.json"
                if existing.read_bytes() != envelope_bytes:
                    raise SourceWriteConflict(f"snapshot address conflict: {snapshot_digest}")
                return snapshot_digest
            temporary = destination.parent / f".{snapshot_digest}.tmp-{uuid4().hex}"
            temporary.mkdir(mode=0o700)
            try:
                write_new_file(temporary / "manifest.json", envelope_bytes)
                publish_directory(temporary, destination)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return snapshot_digest

    def append_history(
        self,
        domain: StorageDomain,
        stream_id: str,
        record: Mapping[str, JsonValue],
        *,
        expected_sequence: int | None = None,
    ) -> HistorySegment:
        if not record:
            raise ValueError("history record must not be empty")
        line = canonical_bytes(dict(record)) + b"\n"
        digest = sha256_bytes(line)
        stream_root = self._platform._domain_root(domain) / "history" / _component(stream_id)
        with exclusive_lock(self._platform._lock_path):
            existing = sorted(stream_root.glob("*.jsonl")) if stream_root.exists() else []
            list(_iter_history_root(stream_root))
            sequence = len(existing)
            if expected_sequence is not None and sequence != expected_sequence:
                raise ControlStateConflict(
                    f"history sequence changed: expected={expected_sequence}, actual={sequence}"
                )
            destination = stream_root / f"{sequence:020d}-{digest}.jsonl"
            atomic_create(destination, line)
        return HistorySegment(sequence=sequence, digest=digest, path=destination)


class TrainingDataView:
    """Standalone reader over an exported filesystem tree mounted read-only."""

    def __init__(
        self,
        read_only_root: str | Path,
        domain: StorageDomain,
        *,
        trust_store: TrustStore,
        snapshot_signer_key_ids: Iterable[str],
    ) -> None:
        root = Path(read_only_root)
        if not root.is_dir():
            raise DataIntegrityError("training data root is unavailable")
        writable = [
            path
            for path in (root, *root.rglob("*"))
            if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ]
        if writable:
            raise DataIntegrityError("training data root must be filesystem read-only")
        if (root / domain.value / "raw").exists():
            raise DataIntegrityError("training data root must not expose the raw source tier")
        self.__root = root
        self.__trust_store = trust_store
        self.__snapshot_signer_key_ids = frozenset(snapshot_signer_key_ids)
        self.domain = domain

    def _domain_root(self) -> Path:
        return self.__root / self.domain.value

    def _resolve_uri(self, uri: str) -> Path:
        parsed = urlsplit(uri)
        if parsed.scheme != "axiom-data" or parsed.netloc != self.domain.value:
            raise DataIntegrityError("training object URI has the wrong domain")
        parts = tuple(part for part in parsed.path.split("/") if part)
        if not parts or any(part in {".", ".."} for part in parts):
            raise DataIntegrityError("unsafe training object URI")
        path = self._domain_root().joinpath(*parts)
        if not path.resolve().is_relative_to(self._domain_root().resolve()):
            raise DataIntegrityError("training object URI escapes read-only domain root")
        return path

    def _read_verified(self, uri: str, *, digest: str, size_bytes: int) -> bytes:
        path = self._resolve_uri(uri)
        if path.is_symlink():
            raise DataIntegrityError("training object must not be a symlink")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DataIntegrityError("training object is unavailable") from exc
        if len(data) != size_bytes or sha256_bytes(data) != digest:
            raise DataIntegrityError("immutable object integrity failure")
        return data

    def _load_normalized(self, path: Path) -> NormalizedVersion:
        try:
            raw = path.read_bytes()
            version = NormalizedVersion.model_validate_json(raw, strict=True)
        except (OSError, ValidationError, ValueError) as exc:
            raise DataIntegrityError("invalid normalized manifest in training root") from exc
        if canonical_bytes(version) != raw or version.domain is not self.domain:
            raise DataIntegrityError("normalized manifest identity is invalid")
        for partition in version.partitions:
            resolved = self._resolve_uri(partition.uri)
            if resolved.parent != path.parent / "partitions":
                raise DataIntegrityError("normalized partition escapes its version")
            data = self._read_verified(
                partition.uri,
                digest=partition.digest,
                size_bytes=partition.size_bytes,
            )
            _validate_container(
                data,
                version.storage_format,
                expected_row_count=partition.row_count,
                expected_schema_digest=version.schema_digest,
            )
        return version

    def load_snapshot(self, snapshot_digest: str) -> DatasetManifest:
        _require_digest(snapshot_digest, "snapshot_digest")
        path = self._domain_root() / "snapshots" / snapshot_digest / "manifest.json"
        try:
            raw = path.read_bytes()
            envelope = SignedEnvelope.model_validate_json(raw, strict=True)
        except (OSError, ValidationError, ValueError) as exc:
            raise DataIntegrityError("invalid training snapshot") from exc
        if canonical_bytes(envelope) != raw or envelope.payload_digest != snapshot_digest:
            raise DataIntegrityError("training snapshot address is invalid")
        if envelope.signature.key_id not in self.__snapshot_signer_key_ids:
            raise DataIntegrityError("snapshot signer is not authorized for snapshot publication")
        try:
            manifest = verify_envelope(envelope, DatasetManifest, self.__trust_store)
        except (SignatureVerificationError, ValueError) as exc:
            raise DataIntegrityError("training snapshot signature is invalid") from exc
        assert isinstance(manifest, DatasetManifest)
        if manifest.asset_class.value != self.domain.value:
            raise DataIntegrityError("snapshot asset class does not match training domain")
        declared = set(manifest.normalized_manifest_digests)
        discovered: set[str] = set()
        if not declared:
            raise DataIntegrityError("snapshot has no normalized manifest binding")
        for partition in manifest.partitions:
            resolved = self._resolve_uri(partition.uri)
            if resolved.parent.name != "partitions":
                raise DataIntegrityError("snapshot partition is outside normalized storage")
            version = self._load_normalized(resolved.parent.parent / "manifest.json")
            version_digest = content_digest(version)
            matches = [item for item in version.partitions if item.partition_id == partition.partition_id]
            if version_digest not in declared or len(matches) != 1:
                raise DataIntegrityError("snapshot partition lacks normalized membership")
            source = matches[0]
            if (
                source.uri,
                source.digest,
                source.size_bytes,
                source.row_count,
            ) != (
                partition.uri,
                partition.digest,
                partition.size_bytes,
                partition.row_count,
            ):
                raise DataIntegrityError("snapshot partition differs from normalized manifest")
            discovered.add(version_digest)
        if discovered != declared:
            raise DataIntegrityError("snapshot normalized manifest set is incomplete")
        return manifest

    def read_snapshot_partition(self, snapshot_digest: str, partition_id: str) -> bytes:
        manifest = self.load_snapshot(snapshot_digest)
        matches = [partition for partition in manifest.partitions if partition.partition_id == partition_id]
        if len(matches) != 1:
            raise KeyError(f"snapshot partition not found: {partition_id}")
        partition = matches[0]
        return self._read_verified(
            partition.uri,
            digest=partition.digest,
            size_bytes=partition.size_bytes,
        )

    def iter_history(self, stream_id: str) -> Iterator[dict[str, JsonValue]]:
        stream_root = self._domain_root() / "history" / _component(stream_id)
        yield from _iter_history_root(stream_root)

    def load_feature_bundle(self, snapshot_digest: str, feature_pipeline_digest: str) -> FeatureBundle:
        _require_digest(snapshot_digest, "snapshot_digest")
        _require_digest(feature_pipeline_digest, "feature_pipeline_digest")
        path = (
            self._domain_root()
            / "features"
            / snapshot_digest
            / feature_pipeline_digest
            / "manifest.json"
        )
        try:
            raw = path.read_bytes()
            bundle = FeatureBundle.model_validate_json(raw, strict=True)
        except (OSError, ValidationError, ValueError) as exc:
            raise DataIntegrityError("invalid feature bundle manifest") from exc
        if canonical_bytes(bundle) != raw:
            raise DataIntegrityError("feature bundle manifest is not canonical")
        if (
            bundle.domain is not self.domain
            or bundle.snapshot_digest != snapshot_digest
            or bundle.feature_pipeline_digest != feature_pipeline_digest
        ):
            raise DataIntegrityError("feature bundle identity does not match its storage key")
        for output in bundle.outputs:
            resolved = self._resolve_uri(output.uri)
            expected_parent = path.parent / "outputs"
            if resolved.parent != expected_parent:
                raise DataIntegrityError("feature output URI escapes its snapshot/pipeline key")
            data = self._read_verified(
                output.uri,
                digest=output.digest,
                size_bytes=output.size_bytes,
            )
            _validate_container(
                data,
                bundle.storage_format,
                expected_row_count=output.row_count,
            )
        return bundle

    def read_feature_output(
        self,
        snapshot_digest: str,
        feature_pipeline_digest: str,
        output_id: str,
    ) -> bytes:
        bundle = self.load_feature_bundle(snapshot_digest, feature_pipeline_digest)
        matches = [output for output in bundle.outputs if output.output_id == output_id]
        if len(matches) != 1:
            raise KeyError(f"feature output not found: {output_id}")
        output = matches[0]
        return self._read_verified(
            output.uri,
            digest=output.digest,
            size_bytes=output.size_bytes,
        )


class FeatureWriter:
    """Write-only capability restricted to snapshot/pipeline-keyed feature outputs."""

    def __init__(self, platform: DataPlatform, domain: StorageDomain) -> None:
        self._platform = platform
        self.domain = domain

    def publish(
        self,
        *,
        snapshot_digest: str,
        feature_pipeline_digest: str,
        storage_format: StorageFormat,
        created_at: datetime,
        outputs: Mapping[str, PartitionPayload],
    ) -> FeatureBundle:
        _require_digest(feature_pipeline_digest, "feature_pipeline_digest")
        self._platform._load_snapshot(self.domain, snapshot_digest)
        if not outputs:
            raise ValueError("at least one feature output is required")

        key_root = (
            self._platform._domain_root(self.domain)
            / "features"
            / snapshot_digest
            / feature_pipeline_digest
        )
        refs: list[FeatureOutputRef] = []
        output_paths: dict[str, str] = {}
        for output_id, payload in sorted(outputs.items()):
            if payload.row_count is None:
                raise DataIntegrityError("feature outputs require a declared row_count")
            _validate_container(
                payload.data,
                storage_format,
                expected_row_count=payload.row_count,
            )
            filename = f"{_component(output_id)}.{storage_format.extension}"
            uri = self._platform._uri(
                self.domain,
                "features",
                snapshot_digest,
                feature_pipeline_digest,
                "outputs",
                filename,
            )
            refs.append(
                FeatureOutputRef(
                    output_id=output_id,
                    uri=uri,
                    digest=sha256_bytes(payload.data),
                    size_bytes=len(payload.data),
                    row_count=payload.row_count,
                )
            )
            if filename in output_paths:
                raise DataIntegrityError("feature output storage-key collision")
            output_paths[filename] = output_id
        bundle = FeatureBundle(
            domain=self.domain,
            snapshot_digest=snapshot_digest,
            feature_pipeline_digest=feature_pipeline_digest,
            storage_format=storage_format,
            created_at=created_at,
            outputs=tuple(refs),
        )
        manifest_bytes = canonical_bytes(bundle)
        with exclusive_lock(self._platform._lock_path):
            if key_root.exists():
                manifest_path = key_root / "manifest.json"
                try:
                    raw = manifest_path.read_bytes()
                    existing = FeatureBundle.model_validate_json(raw, strict=True)
                except (OSError, ValidationError, ValueError) as exc:
                    raise DataIntegrityError("invalid existing feature bundle") from exc
                if canonical_bytes(existing) != raw:
                    raise DataIntegrityError("existing feature bundle is non-canonical")
                for output in existing.outputs:
                    data = self._platform._read_verified(
                        output.uri,
                        digest=output.digest,
                        size_bytes=output.size_bytes,
                        expected_domain=self.domain,
                    )
                    _validate_container(
                        data,
                        existing.storage_format,
                        expected_row_count=output.row_count,
                    )
                if existing != bundle:
                    raise SourceWriteConflict(
                        "feature key already contains different outputs; change the pipeline digest"
                    )
                return existing
            temporary = key_root.parent / f".{key_root.name}.tmp-{uuid4().hex}"
            temporary.mkdir(parents=True, mode=0o700)
            try:
                write_new_file(temporary / "manifest.json", manifest_bytes)
                for ref in refs:
                    filename = Path(urlsplit(ref.uri).path).name
                    write_new_file(
                        temporary / "outputs" / filename,
                        outputs[output_paths[filename]].data,
                    )
                publish_directory(temporary, key_root)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return bundle


class ControlStateStore:
    """Atomic JSON for small mutable local control state, outside ``axiom-data``."""

    def __init__(
        self,
        root: str | Path,
        *,
        immutable_data_root: str | Path,
        max_bytes: int = 1024 * 1024,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root)
        self.immutable_data_root = Path(immutable_data_root)
        control = self.root.resolve()
        immutable = self.immutable_data_root.resolve()
        if control.is_relative_to(immutable) or immutable.is_relative_to(control):
            raise ValueError("control state and immutable data roots must be disjoint")
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._lock_path = self.root / ".control-state.lock"

    def _path(self, name: str) -> Path:
        if not _CONTROL_NAME_RE.fullmatch(name):
            raise ValueError("control-state name must be a safe local identifier")
        filename = name if name.endswith(".json") else f"{name}.json"
        return self.root / filename

    def read(self, name: str) -> dict[str, JsonValue]:
        path = self._path(name)
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise DataIntegrityError(f"invalid control state: {name}") from exc
        if not isinstance(value, dict) or canonical_bytes(value) != raw:
            raise DataIntegrityError(f"control state is not a canonical JSON object: {name}")
        return value

    def write(
        self,
        name: str,
        payload: Mapping[str, JsonValue],
        *,
        expected_digest: str | None,
    ) -> str:
        path = self._path(name)
        data = canonical_bytes(dict(payload))
        if len(data) > self.max_bytes:
            raise ValueError(f"control state exceeds {self.max_bytes} bytes")
        new_digest = sha256_bytes(data)
        with exclusive_lock(self._lock_path):
            actual_digest = sha256_bytes(path.read_bytes()) if path.exists() else None
            if actual_digest != expected_digest:
                raise ControlStateConflict(
                    f"control state changed: expected={expected_digest!r}, actual={actual_digest!r}"
                )
            atomic_replace(path, data)
        return new_digest
