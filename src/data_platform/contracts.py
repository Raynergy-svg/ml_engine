"""Strict contracts for the canonical AXIOM data-platform layout."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator

from src.evidence.contracts import StrictContract

Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=240)]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


class StorageDomain(str, Enum):
    FX = "fx"
    EQUITY = "equity"
    CRYPTO = "crypto"
    FILINGS = "filings"
    MULTI_ASSET = "multi_asset"
    PORTFOLIO = "portfolio"
    OUTCOMES = "outcomes"
    EVIDENCE = "evidence"


class StorageFormat(str, Enum):
    PARQUET = "parquet"
    ARROW_IPC = "arrow"

    @property
    def extension(self) -> str:
        return "parquet" if self is StorageFormat.PARQUET else "arrow"

    @property
    def media_type(self) -> str:
        if self is StorageFormat.PARQUET:
            return "application/vnd.apache.parquet"
        return "application/vnd.apache.arrow.file"


class ImmutableObjectRef(StrictContract):
    uri: NonEmptyString
    digest: Sha256Digest
    size_bytes: int = Field(ge=0)
    media_type: NonEmptyString


class RawObjectReceipt(StrictContract):
    domain: StorageDomain
    source_id: Identifier
    retrieved_at: datetime
    object: ImmutableObjectRef
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _utc_retrieved = field_validator("retrieved_at")(_require_utc)


class NormalizedPartitionRef(StrictContract):
    partition_id: Identifier
    uri: NonEmptyString
    digest: Sha256Digest
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)


class NormalizedVersion(StrictContract):
    domain: StorageDomain
    dataset_id: Identifier
    version_id: Identifier
    storage_format: StorageFormat
    schema_digest: Sha256Digest
    source_raw_digests: tuple[Sha256Digest, ...] = Field(min_length=1)
    created_at: datetime
    partitions: tuple[NormalizedPartitionRef, ...] = Field(min_length=1)

    _utc_created = field_validator("created_at")(_require_utc)

    @model_validator(mode="after")
    def unique_sources_and_partitions(self) -> "NormalizedVersion":
        if len(set(self.source_raw_digests)) != len(self.source_raw_digests):
            raise ValueError("source_raw_digests must be unique")
        ids = [partition.partition_id for partition in self.partitions]
        if len(set(ids)) != len(ids):
            raise ValueError("partition_id values must be unique")
        return self


class FeatureOutputRef(StrictContract):
    output_id: Identifier
    uri: NonEmptyString
    digest: Sha256Digest
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)


class FeatureBundle(StrictContract):
    domain: StorageDomain
    snapshot_digest: Sha256Digest
    feature_pipeline_digest: Sha256Digest
    storage_format: StorageFormat
    created_at: datetime
    outputs: tuple[FeatureOutputRef, ...] = Field(min_length=1)

    _utc_created = field_validator("created_at")(_require_utc)

    @model_validator(mode="after")
    def unique_outputs(self) -> "FeatureBundle":
        ids = [output.output_id for output in self.outputs]
        if len(set(ids)) != len(ids):
            raise ValueError("output_id values must be unique")
        return self
