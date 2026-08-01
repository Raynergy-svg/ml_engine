"""Durable local evidence store with immutable packages and append-only events."""

from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, TypeVar
from uuid import uuid4

from pydantic import ValidationError

from .canonical import canonical_bytes
from .contracts import (
    ArtifactRef,
    AuthorityRole,
    ChampionPointer,
    DispositionEvent,
    DispositionReceipt,
    DispositionState,
    EvidencePackage,
    ImportDecision,
    LocalImportVerdict,
    SignedEnvelope,
    StrictContract,
)
from .event_store import EventChainError, ReconstructedDisposition, reconstruct_disposition
from .hashing import sha256_bytes
from .signing import SignatureVerificationError, TrustStore, verify_envelope
from .transition_policy import AuthorityRegistry, TransitionPolicyError

TContract = TypeVar("TContract", bound=StrictContract)


class EvidenceStoreError(RuntimeError):
    """Base error for durable evidence-store operations."""


class StoreCorruptionError(EvidenceStoreError):
    """Stored bytes contradict their digest, signature, schema, or history."""


class ImmutableConflictError(EvidenceStoreError):
    """A caller attempted to replace content at an immutable address."""


class ConcurrentHeadError(EvidenceStoreError):
    """A compare-and-swap expected head did not match durable history."""


class ChampionPointerError(EvidenceStoreError):
    """A champion pointer is unauthorized or does not resolve exactly."""


class ProjectionStaleError(EvidenceStoreError):
    """Authority committed, but its rebuildable index projection is stale."""


class EvidenceStore:
    """Filesystem-backed AXIOM evidence authority.

    The store lock serializes local mutations. Signed events and explicit
    ``expected_head_digest`` values provide the durable compare-and-swap
    contract; indexes remain rebuildable caches rather than authority.
    """

    DIRECTORY_NAMES = (
        "packages",
        "dispositions",
        "verdicts",
        "indexes",
        "quarantine",
        "champions",
    )

    def __init__(
        self,
        root: str | Path,
        *,
        trust_store: TrustStore,
        authorities: AuthorityRegistry,
        trusted_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = Path(root)
        self.trust_store = trust_store
        self.authorities = authorities
        self._trusted_clock = trusted_clock or (lambda: datetime.now(timezone.utc))
        self.root.mkdir(parents=True, exist_ok=True)
        for name in self.DIRECTORY_NAMES:
            (self.root / name).mkdir(mode=0o700, exist_ok=True)
        self._locks_dir = self.root / ".locks"
        self._locks_dir.mkdir(mode=0o700, exist_ok=True)
        self._lock_path = self._locks_dir / "store.lock"

    @property
    def index_path(self) -> Path:
        return self.root / "indexes" / "current.json"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _atomic_write_bytes(cls, destination: Path, data: bytes, *, mode: int = 0o600) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            cls._fsync_directory(destination.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def _atomic_create_bytes(cls, destination: Path, data: bytes, *, mode: int = 0o600) -> None:
        """Atomically create immutable bytes without an overwrite window."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            temporary.unlink()
            cls._fsync_directory(destination.parent)
        except FileExistsError as exc:
            temporary.unlink(missing_ok=True)
            raise ImmutableConflictError(f"immutable path already exists: {destination}") from exc
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _safe_package_path(base: Path, relative_path: str) -> Path:
        normalized = PurePosixPath(relative_path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
            raise ValueError(f"unsafe package-relative path: {relative_path!r}")
        destination = base.joinpath(*normalized.parts)
        if not destination.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"package path escapes destination: {relative_path!r}")
        return destination

    @classmethod
    def _write_new_file(cls, destination: Path, data: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _read_envelope(path: Path) -> tuple[SignedEnvelope, bytes]:
        try:
            raw = path.read_bytes()
            envelope = SignedEnvelope.model_validate_json(raw, strict=True)
        except (OSError, ValidationError, ValueError) as exc:
            raise StoreCorruptionError(f"invalid signed envelope at {path}: {exc}") from exc
        if canonical_bytes(envelope) != raw:
            raise StoreCorruptionError(f"non-canonical signed envelope at {path}")
        return envelope, raw

    @staticmethod
    def _read_receipt(path: Path) -> tuple[DispositionReceipt, bytes]:
        try:
            raw = path.read_bytes()
            receipt = DispositionReceipt.model_validate_json(raw, strict=True)
        except (OSError, ValidationError, ValueError) as exc:
            raise StoreCorruptionError(f"invalid disposition receipt at {path}: {exc}") from exc
        if canonical_bytes(receipt) != raw:
            raise StoreCorruptionError(f"non-canonical disposition receipt at {path}")
        return receipt, raw

    def _verify_typed_envelope(
        self,
        envelope: SignedEnvelope,
        expected_type: type[TContract],
    ) -> TContract:
        try:
            payload = verify_envelope(envelope, expected_type, self.trust_store)
        except (SignatureVerificationError, ValueError) as exc:
            raise StoreCorruptionError(str(exc)) from exc
        if not isinstance(payload, expected_type):
            raise StoreCorruptionError(f"expected {expected_type.__name__} payload")
        return payload

    def write_package(self, envelope: SignedEnvelope, files: Mapping[str, bytes]) -> str:
        """Verify and atomically publish one complete immutable package directory."""
        package = self._verify_typed_envelope(envelope, EvidencePackage)
        self._authorize_package_producer(package, envelope.signature.key_id)
        expected_paths = set(package.checksums)
        if set(files) != expected_paths:
            missing = sorted(expected_paths - set(files))
            extra = sorted(set(files) - expected_paths)
            raise ValueError(f"package file set mismatch; missing={missing}, extra={extra}")

        refs = {ref.relative_path: ref for ref in package.artifacts + package.logs}
        for relative_path, data in files.items():
            ref = refs[relative_path]
            if len(data) != ref.size_bytes:
                raise ValueError(f"size mismatch for {relative_path!r}")
            if sha256_bytes(data) != ref.digest:
                raise ValueError(f"digest mismatch for {relative_path!r}")

        package_digest = envelope.payload_digest
        final_directory = self.root / "packages" / package_digest
        envelope_bytes = canonical_bytes(envelope)

        with self._locked():
            if final_directory.exists():
                existing, existing_bytes = self._load_package_unlocked(package_digest)
                if existing_bytes != envelope_bytes or existing != package:
                    raise ImmutableConflictError(f"package address already contains different bytes: {package_digest}")
                self._repair_projection_after_commit("package", package_digest)
                return package_digest

            # Refuse to add new authority while existing authority cannot be
            # reconstructed. The new package itself was fully verified above.
            self._index_payload_unlocked()

            temporary = self.root / "packages" / f".{package_digest}.tmp-{uuid4().hex}"
            temporary.mkdir(mode=0o700)
            try:
                self._write_new_file(temporary / "envelope.json", envelope_bytes)
                for relative_path, data in files.items():
                    destination = self._safe_package_path(temporary, relative_path)
                    self._write_new_file(destination, data)
                for directory, _, _ in os.walk(temporary, topdown=False):
                    self._fsync_directory(Path(directory))
                os.rename(temporary, final_directory)
                self._fsync_directory(final_directory.parent)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            self._repair_projection_after_commit("package", package_digest)
        return package_digest

    def _load_package_unlocked(self, package_digest: str) -> tuple[EvidencePackage, bytes]:
        directory = self.root / "packages" / package_digest
        if not directory.is_dir():
            raise FileNotFoundError(f"unknown package: {package_digest}")
        envelope, envelope_bytes = self._read_envelope(directory / "envelope.json")
        if envelope.payload_digest != package_digest:
            raise StoreCorruptionError("package directory name does not match payload digest")
        package = self._verify_typed_envelope(envelope, EvidencePackage)
        self._authorize_package_producer(package, envelope.signature.key_id)

        expected_files = {"envelope.json"}
        for ref in package.artifacts + package.logs:
            try:
                path = self._safe_package_path(directory, ref.relative_path)
            except ValueError as exc:
                raise StoreCorruptionError(str(exc)) from exc
            if path.is_symlink():
                raise StoreCorruptionError(f"stored artifact must not be a symlink: {ref.relative_path!r}")
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise StoreCorruptionError(f"missing package file {ref.relative_path!r}") from exc
            if len(data) != ref.size_bytes or sha256_bytes(data) != ref.digest:
                raise StoreCorruptionError(f"stored artifact mismatch: {ref.relative_path!r}")
            expected_files.add(ref.relative_path.replace("\\", "/"))

        actual_files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise StoreCorruptionError(
                f"stored package file set mismatch; expected={sorted(expected_files)}, actual={sorted(actual_files)}"
            )
        return package, envelope_bytes

    def _authorize_package_producer(self, package: EvidencePackage, signer_key_id: str) -> None:
        try:
            self.authorities.authorize_identity(
                actor_id=package.producer_id,
                role=AuthorityRole.PRODUCER,
                key_id=signer_key_id,
            )
        except TransitionPolicyError as exc:
            raise StoreCorruptionError(str(exc)) from exc

    def load_package(self, package_digest: str) -> EvidencePackage:
        with self._locked():
            package, _ = self._load_package_unlocked(package_digest)
            return package

    @staticmethod
    def _read_verified_member(path: Path, ref: ArtifactRef) -> bytes:
        """Read one declared member and bind the returned bytes to its ref."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise StoreCorruptionError(
                f"missing package file {ref.relative_path!r}"
            ) from exc
        if len(data) != ref.size_bytes or sha256_bytes(data) != ref.digest:
            raise StoreCorruptionError(
                f"stored artifact mismatch: {ref.relative_path!r}"
            )
        return data

    def load_package_file(self, package_digest: str, relative_path: str) -> bytes:
        """Return one verified immutable package member.

        The full package is revalidated first, so callers never receive bytes
        from an incomplete or checksum-inconsistent directory.  This is the
        durable read path used for signed manifests and evaluation reports.
        """
        with self._locked():
            package, _ = self._load_package_unlocked(package_digest)
            declared = {
                ref.relative_path: ref for ref in package.artifacts + package.logs
            }
            ref = declared.get(relative_path)
            if ref is None:
                raise FileNotFoundError(
                    f"package {package_digest} has no declared member {relative_path!r}"
                )
            path = self._safe_package_path(
                self.root / "packages" / package_digest, relative_path
            )
            return self._read_verified_member(path, ref)

    def write_verdict(self, envelope: SignedEnvelope) -> str:
        verdict = self._verify_typed_envelope(envelope, LocalImportVerdict)
        self._authorize_verdict(verdict, envelope.signature.key_id)
        destination = self.root / "verdicts" / f"{envelope.payload_digest}.json"
        data = canonical_bytes(envelope)
        with self._locked():
            self._load_package_unlocked(verdict.package_digest)
            if destination.exists():
                if destination.read_bytes() != data:
                    raise ImmutableConflictError(f"verdict address conflict: {envelope.payload_digest}")
                return envelope.payload_digest
            self._atomic_create_bytes(destination, data)
        return envelope.payload_digest

    def _authorize_verdict(self, verdict: LocalImportVerdict, signer_key_id: str) -> None:
        try:
            self.authorities.authorize_identity(
                actor_id=verdict.importer_id,
                role=AuthorityRole.LOCAL_IMPORTER,
                key_id=signer_key_id,
            )
        except TransitionPolicyError as exc:
            raise EvidenceStoreError(str(exc)) from exc

    def _require_accepted_verdict_unlocked(
        self, package_digest: str, verdict_digest: str
    ) -> LocalImportVerdict:
        path = self.root / "verdicts" / f"{verdict_digest}.json"
        if not path.exists():
            raise EvidenceStoreError(
                f"package {package_digest} has no durable accepted local import verdict "
                f"at {verdict_digest}"
            )
        envelope, _ = self._read_envelope(path)
        if envelope.payload_digest != verdict_digest:
            raise StoreCorruptionError("verdict address does not match its payload digest")
        verdict = self._verify_typed_envelope(envelope, LocalImportVerdict)
        self._authorize_verdict(verdict, envelope.signature.key_id)
        if verdict.package_digest != package_digest:
            raise EvidenceStoreError("referenced local import verdict belongs to another package")
        if verdict.decision != ImportDecision.ACCEPT_FOR_QUARANTINE:
            raise EvidenceStoreError("referenced local import verdict is not accepted")
        return verdict

    def _require_promotion_evidence_unlocked(
        self, package_digest: str, receipts: list[DispositionReceipt]
    ) -> None:
        for receipt in receipts:
            event = self._verify_typed_envelope(receipt.envelope, DispositionEvent)
            if event.to_state != DispositionState.QUARANTINED:
                continue
            verdict_digest = event.metadata.get("verdict_digest")
            if not isinstance(verdict_digest, str):
                raise EvidenceStoreError(
                    "QUARANTINED event must reference an exact verdict_digest"
                )
            self._require_accepted_verdict_unlocked(package_digest, verdict_digest)
            return
        raise EvidenceStoreError("promoted ledger has no QUARANTINED evidence event")

    def _ledger_directory(self, package_digest: str) -> Path:
        return self.root / "dispositions" / package_digest

    def _load_ledger_unlocked(self, package_digest: str) -> list[DispositionReceipt]:
        directory = self._ledger_directory(package_digest)
        if not directory.exists():
            return []
        unexpected = [
            path.name
            for path in directory.iterdir()
            if path.is_file() and not path.name.startswith(".") and path.suffix != ".json"
        ]
        if unexpected:
            raise StoreCorruptionError(f"unexpected disposition files: {sorted(unexpected)}")
        receipts: list[DispositionReceipt] = []
        for path in sorted(directory.glob("*.json")):
            receipt, _ = self._read_receipt(path)
            envelope = receipt.envelope
            event = self._verify_typed_envelope(envelope, DispositionEvent)
            expected_name = f"{event.sequence:020d}-{envelope.payload_digest}.json"
            if path.name != expected_name:
                raise StoreCorruptionError(
                    f"event filename does not match sequence and digest: {path.name!r}"
                )
            receipts.append(receipt)
        if receipts:
            try:
                state = reconstruct_disposition(
                    receipts,
                    trust_store=self.trust_store,
                    authorities=self.authorities,
                )
            except EventChainError as exc:
                raise StoreCorruptionError(f"invalid disposition ledger for {package_digest}: {exc}") from exc
            if state.state in {
                DispositionState.QUARANTINED,
                DispositionState.SHADOW,
                DispositionState.OPERATOR_APPROVED,
                DispositionState.CHAMPION,
                DispositionState.RETIRED,
            }:
                self._require_promotion_evidence_unlocked(package_digest, receipts)
        return receipts

    def load_ledger(self, package_digest: str) -> tuple[SignedEnvelope, ...]:
        with self._locked():
            return tuple(receipt.envelope for receipt in self._load_ledger_unlocked(package_digest))

    def append_disposition(
        self,
        envelope: SignedEnvelope,
        *,
        expected_head_digest: str | None,
    ) -> ReconstructedDisposition:
        event = self._verify_typed_envelope(envelope, DispositionEvent)
        package_digest = event.package_digest
        with self._locked():
            package, _ = self._load_package_unlocked(package_digest)
            if event.to_state == DispositionState.CHAMPION:
                self._validate_champion_event(event, package)
            ledger = self._load_ledger_unlocked(package_digest)
            if ledger and ledger[-1].envelope == envelope:
                state = reconstruct_disposition(
                    ledger,
                    trust_store=self.trust_store,
                    authorities=self.authorities,
                )
                self._repair_projection_after_commit("disposition", envelope.payload_digest)
                return state
            actual_head = ledger[-1].envelope.payload_digest if ledger else None
            if actual_head != expected_head_digest:
                raise ConcurrentHeadError(
                    f"disposition head changed for {package_digest}: "
                    f"expected={expected_head_digest!r}, actual={actual_head!r}"
                )
            received_at = self._trusted_clock()
            receipt = DispositionReceipt(envelope=envelope, received_at=received_at)
            proposed = [*ledger, receipt]
            try:
                state = reconstruct_disposition(
                    proposed,
                    trust_store=self.trust_store,
                    authorities=self.authorities,
                )
            except EventChainError as exc:
                raise EvidenceStoreError(f"refused disposition append: {exc}") from exc

            if state.state in {
                DispositionState.QUARANTINED,
                DispositionState.SHADOW,
                DispositionState.OPERATOR_APPROVED,
                DispositionState.CHAMPION,
                DispositionState.RETIRED,
            }:
                self._require_promotion_evidence_unlocked(package_digest, proposed)

            projection = canonical_bytes(
                self._index_payload_unlocked(ledger_overrides={package_digest: proposed})
            )

            directory = self._ledger_directory(package_digest)
            if not directory.exists():
                directory.mkdir(mode=0o700, parents=True)
                self._fsync_directory(directory)
                self._fsync_directory(directory.parent)
            destination = directory / f"{event.sequence:020d}-{envelope.payload_digest}.json"
            if destination.exists():
                raise ImmutableConflictError(f"event already exists: {envelope.payload_digest}")
            self._atomic_create_bytes(destination, canonical_bytes(receipt))
            self._write_precomputed_projection_after_commit(
                projection, "disposition", envelope.payload_digest
            )
            return state

    @staticmethod
    def _validate_champion_event(event: DispositionEvent, package: EvidencePackage) -> None:
        artifact_id = event.metadata.get("artifact_id")
        artifact_digest = event.metadata.get("artifact_digest")
        lane_id = event.metadata.get("lane_id")
        if not all(isinstance(value, str) for value in (artifact_id, artifact_digest, lane_id)):
            raise EvidenceStoreError(
                "CHAMPION event metadata requires string lane_id, artifact_id, and artifact_digest"
            )
        if lane_id != package.lane_id:
            raise EvidenceStoreError("CHAMPION event lane_id does not match package")
        artifacts = {artifact.artifact_id: artifact for artifact in package.artifacts}
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact.digest != artifact_digest:
            raise EvidenceStoreError("CHAMPION event does not resolve to an exact package artifact")

    def _champion_path(self, lane_id: str) -> Path:
        lane_digest = sha256_bytes(lane_id.encode("utf-8"))
        return self.root / "champions" / f"{lane_digest}.json"

    def _validate_champion_unlocked(
        self,
        envelope: SignedEnvelope,
    ) -> ChampionPointer:
        pointer = self._verify_typed_envelope(envelope, ChampionPointer)
        if pointer.signer_key_id != envelope.signature.key_id:
            raise ChampionPointerError("champion pointer signer does not match envelope signer")
        try:
            self.authorities.authorize_identity(
                actor_id=pointer.actor_id,
                role=AuthorityRole.PROMOTION_SERVICE,
                key_id=pointer.signer_key_id,
            )
        except TransitionPolicyError as exc:
            raise ChampionPointerError(str(exc)) from exc

        package, _ = self._load_package_unlocked(pointer.package_digest)
        if package.lane_id != pointer.lane_id:
            raise ChampionPointerError("champion lane does not match package lane")
        artifacts = {artifact.artifact_id: artifact for artifact in package.artifacts}
        artifact = artifacts.get(pointer.artifact_id)
        if artifact is None or artifact.digest != pointer.artifact_digest:
            raise ChampionPointerError("champion artifact does not resolve to exact package bytes")

        ledger = self._load_ledger_unlocked(pointer.package_digest)
        if not ledger:
            raise ChampionPointerError("champion package has no disposition ledger")
        state = reconstruct_disposition(
            ledger,
            trust_store=self.trust_store,
            authorities=self.authorities,
        )
        if state.state != DispositionState.CHAMPION:
            raise ChampionPointerError(f"package disposition is {state.state.value}, not CHAMPION")
        if pointer.disposition_head_digest != state.head_event_digest:
            raise ChampionPointerError("champion pointer references a stale disposition head")
        return pointer

    def write_champion_pointer(
        self,
        envelope: SignedEnvelope,
        *,
        expected_pointer_digest: str | None,
    ) -> str:
        with self._locked():
            pointer = self._validate_champion_unlocked(envelope)
            destination = self._champion_path(pointer.lane_id)
            actual_digest: str | None = None
            if destination.exists():
                existing, existing_bytes = self._read_envelope(destination)
                actual_digest = existing.payload_digest
                if existing_bytes == canonical_bytes(envelope):
                    self._repair_projection_after_commit(
                        "champion pointer", envelope.payload_digest
                    )
                    return envelope.payload_digest
            if actual_digest != expected_pointer_digest:
                raise ConcurrentHeadError(
                    f"champion pointer changed for {pointer.lane_id}: "
                    f"expected={expected_pointer_digest!r}, actual={actual_digest!r}"
                )
            projection = canonical_bytes(
                self._index_payload_unlocked(pointer_overrides={pointer.lane_id: envelope})
            )
            self._atomic_write_bytes(destination, canonical_bytes(envelope))
            self._write_precomputed_projection_after_commit(
                projection, "champion pointer", envelope.payload_digest
            )
            return envelope.payload_digest

    def load_champion_pointer(self, lane_id: str) -> ChampionPointer:
        with self._locked():
            path = self._champion_path(lane_id)
            if not path.exists():
                raise FileNotFoundError(f"no champion pointer for lane: {lane_id}")
            envelope, _ = self._read_envelope(path)
            pointer = self._validate_champion_unlocked(envelope)
            if pointer.lane_id != lane_id:
                raise StoreCorruptionError("champion pointer lane does not match its address")
            return pointer

    def load_champion_pointer_envelope(self, lane_id: str) -> SignedEnvelope:
        """Return the lane's fully validated champion-pointer envelope.

        Additive read helper for the promotion service (Phase L): the caller
        needs the exact signed envelope (for compare-and-swap digests and for
        rollback preservation), not just the decoded pointer. Validation is
        identical to :meth:`load_champion_pointer`.
        """
        with self._locked():
            path = self._champion_path(lane_id)
            if not path.exists():
                raise FileNotFoundError(f"no champion pointer for lane: {lane_id}")
            envelope, _ = self._read_envelope(path)
            pointer = self._validate_champion_unlocked(envelope)
            if pointer.lane_id != lane_id:
                raise StoreCorruptionError("champion pointer lane does not match its address")
            return envelope

    def clear_champion_pointer(self, lane_id: str, *, expected_pointer_digest: str) -> None:
        """Atomically remove a lane's champion pointer (CAS-guarded).

        Additive Phase-L helper for supervised champion swaps: a new CHAMPION
        disposition cannot commit while the lane's pointer file still selects
        a different (retired) package, because the index projection validates
        the pointer against the champion-state package. Removing the pointer
        is fail-closed — until the replacement pointer is written the lane
        advertises no champion and consumers must abstain. The compare-and-
        swap digest prevents clearing a pointer some other writer replaced.
        """
        with self._locked():
            path = self._champion_path(lane_id)
            if not path.exists():
                raise FileNotFoundError(f"no champion pointer for lane: {lane_id}")
            envelope, _ = self._read_envelope(path)
            if envelope.payload_digest != expected_pointer_digest:
                raise ConcurrentHeadError(
                    f"champion pointer changed for {lane_id}: "
                    f"expected={expected_pointer_digest!r}, actual={envelope.payload_digest!r}"
                )
            path.unlink()
            self._fsync_directory(path.parent)
            self._repair_projection_after_commit(
                "champion pointer removal", expected_pointer_digest
            )

    def peek_champion_pointer_digest(self, lane_id: str) -> str | None:
        """Return the raw payload digest of the lane's pointer file, or None.

        CAS-only helper: it verifies canonical-envelope integrity but does NOT
        re-validate champion resolution, so it also works when the pointer has
        gone stale (e.g. its package was retired during a rollback). The digest
        is only useful as ``expected_pointer_digest`` for
        :meth:`write_champion_pointer`, which re-validates everything itself —
        this helper grants no authority and returns no artifact identity.
        """
        with self._locked():
            path = self._champion_path(lane_id)
            if not path.exists():
                return None
            envelope, _ = self._read_envelope(path)
            return envelope.payload_digest

    def _index_payload_unlocked(
        self,
        *,
        ledger_overrides: Mapping[str, list[DispositionReceipt]] | None = None,
        pointer_overrides: Mapping[str, SignedEnvelope] | None = None,
    ) -> dict[str, object]:
        packages: dict[str, object] = {}
        champions: dict[str, object] = {}
        champion_lanes: set[str] = set()
        ledger_overrides = ledger_overrides or {}
        pointer_overrides = pointer_overrides or {}
        packages_root = self.root / "packages"
        for directory in sorted(packages_root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            package_digest = directory.name
            package, _ = self._load_package_unlocked(package_digest)
            ledger = ledger_overrides.get(package_digest)
            if ledger is None:
                ledger = self._load_ledger_unlocked(package_digest)
            state = None
            if ledger:
                state = reconstruct_disposition(
                    ledger,
                    trust_store=self.trust_store,
                    authorities=self.authorities,
                )
                # The package directory is content-addressed, so its name must equal the
                # digest the ledger reconstructs to. Divergence means the ledger under
                # this directory describes a different package, and indexing it would
                # silently attribute one package's state to another's digest.
                if state.package_digest != package_digest:
                    raise StoreCorruptionError(
                        f"ledger under package directory {package_digest!r} reconstructs to "
                        f"package {state.package_digest!r}"
                    )
            packages[package_digest] = {
                "artifact_digests": {
                    artifact.artifact_id: artifact.digest for artifact in package.artifacts
                },
                "head_event_digest": state.head_event_digest if state else None,
                "lane_id": package.lane_id,
                "package_id": package.package_id,
                "sequence": state.sequence if state else None,
                "state": state.state.value if state else None,
            }
            if state is not None and state.state == DispositionState.CHAMPION:
                event = self._verify_typed_envelope(ledger[-1].envelope, DispositionEvent)
                self._validate_champion_event(event, package)
                lane_id = str(event.metadata["lane_id"])
                if lane_id in champion_lanes:
                    raise StoreCorruptionError(f"multiple champion packages for lane {lane_id!r}")
                champion_lanes.add(lane_id)
                pointer_path = self._champion_path(lane_id)
                pointer_envelope = pointer_overrides.get(lane_id)
                if pointer_envelope is None and pointer_path.exists():
                    pointer_envelope, _ = self._read_envelope(pointer_path)
                if pointer_envelope is None:
                    continue
                pointer = self._validate_champion_unlocked(pointer_envelope)
                if pointer.package_digest != package_digest:
                    raise StoreCorruptionError(
                        f"champion pointer for lane {lane_id!r} selects a different package"
                    )
                champions[lane_id] = {
                    "artifact_digest": pointer.artifact_digest,
                    "artifact_id": pointer.artifact_id,
                    "disposition_head_digest": pointer.disposition_head_digest,
                    "package_digest": package_digest,
                }
        return {
            "champions": champions,
            "packages": packages,
            "schema_version": "1.0.0",
        }

    def _rebuild_indexes_unlocked(self) -> bytes:
        data = canonical_bytes(self._index_payload_unlocked())
        self._atomic_write_bytes(self.index_path, data)
        return data

    def _repair_projection_after_commit(self, kind: str, digest: str) -> bytes:
        try:
            return self._rebuild_indexes_unlocked()
        except Exception as exc:
            raise ProjectionStaleError(
                f"{kind} committed at {digest}, but evidence projection is stale"
            ) from exc

    def _write_precomputed_projection_after_commit(
        self, data: bytes, kind: str, digest: str
    ) -> None:
        try:
            self._atomic_write_bytes(self.index_path, data)
        except Exception as exc:
            raise ProjectionStaleError(
                f"{kind} committed at {digest}, but evidence projection is stale"
            ) from exc

    def rebuild_indexes(self) -> bytes:
        """Recreate byte-deterministic indexes from immutable packages and events."""
        with self._locked():
            return self._rebuild_indexes_unlocked()

    def current_index_bytes(self) -> bytes:
        with self._locked():
            try:
                return self.index_path.read_bytes()
            except OSError as exc:
                raise EvidenceStoreError("current evidence index is unavailable") from exc
