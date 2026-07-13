from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.ipc as arrow_ipc
import pyarrow.parquet as pq

from src.data_platform import (
    DataPlatform,
    PartitionPayload,
    StorageDomain,
    StorageFormat,
)
from src.data_platform.platform import (
    ControlStateConflict,
    DataIntegrityError,
    SourceWriteConflict,
)
from src.evidence.contracts import AssetClass, DatasetManifest, PartitionRef
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustStore

NOW = datetime(2026, 7, 11, 18, 0, tzinfo=timezone.utc)


def parquet_bytes(label: str, row_count: int) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"label": [label] * row_count}), sink)
    return sink.getvalue().to_pybytes()


def parquet_schema_digest(data: bytes) -> str:
    schema = pq.ParquetFile(pa.BufferReader(data)).schema_arrow
    return sha256_bytes(schema.serialize().to_pybytes())


def arrow_bytes(label: str, row_count: int) -> bytes:
    table = pa.table({"label": [label] * row_count})
    sink = pa.BufferOutputStream()
    with arrow_ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def platform_with_signer(tmp_path: Path) -> tuple[DataPlatform, Ed25519Signer]:
    signer = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(days=1)))
    return DataPlatform(
        tmp_path / "axiom-data",
        trust_store=trust,
        snapshot_signer_key_ids=(signer.key_id,),
    ), signer


def training_view(platform: DataPlatform, domain: StorageDomain, destination: Path):
    root = platform.export_training_root(destination)
    return platform.training_view(domain, read_only_root=root), root


def publish_fx_snapshot(
    platform: DataPlatform,
    signer: Ed25519Signer,
) -> tuple[str, bytes]:
    writer = platform.ingestion_writer()
    raw = writer.put_raw(
        StorageDomain.FX,
        b"provider-response-v1",
        source_id="oanda-daily",
        retrieved_at=NOW,
        media_type="application/json",
        metadata={"license": "internal-use"},
    )
    partition_data = parquet_bytes("EUR_USD-2026-07", 7)
    version = writer.publish_normalized(
        StorageDomain.FX,
        dataset_id="fx-daily",
        version_id="2026-07-11-v1",
        storage_format=StorageFormat.PARQUET,
        schema_digest=parquet_schema_digest(partition_data),
        source_raw_digests=(raw.object.digest,),
        created_at=NOW,
        partitions={"instrument=EUR_USD/year=2026": PartitionPayload(partition_data, row_count=7)},
    )
    normalized = version.partitions[0]
    manifest = DatasetManifest(
        dataset_id="fx-daily-snapshot-2026-07-11",
        asset_class=AssetClass.FX,
        domain="daily-bars",
        provider="oanda",
        retrieved_at=NOW,
        coverage_start=date(2026, 7, 1),
        coverage_end=date(2026, 7, 10),
        instruments=("EUR_USD",),
        data_schema_version="fx-daily-v1",
        point_in_time_rules=("bar close precedes feature as-of",),
        normalized_manifest_digests=(content_digest(version),),
        partitions=(
            PartitionRef(
                partition_id=normalized.partition_id,
                uri=normalized.uri,
                digest=normalized.digest,
                size_bytes=normalized.size_bytes,
                row_count=normalized.row_count,
            ),
        ),
    )
    envelope = signer.sign(manifest, created_at=NOW)
    return writer.publish_snapshot(StorageDomain.FX, envelope), partition_data


def test_platform_creates_all_canonical_domain_roots(tmp_path: Path) -> None:
    platform, _ = platform_with_signer(tmp_path)
    assert {path.name for path in platform.root.iterdir() if not path.name.startswith(".")} == {
        domain.value for domain in StorageDomain
    }
    for domain in StorageDomain:
        assert {
            path.name for path in (platform.root / domain.value).iterdir()
        } == {"raw", "normalized", "snapshots", "features", "history"}


def test_raw_objects_are_content_addressed_immutable_and_traceable(tmp_path: Path) -> None:
    platform, _ = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    data = b"original-provider-payload"
    receipt = writer.put_raw(
        StorageDomain.EQUITY,
        data,
        source_id="sec-edgar",
        retrieved_at=NOW,
        media_type="application/json",
        metadata={"accession": "0001"},
    )
    assert receipt.object.digest == sha256_bytes(data)
    object_path = platform._resolve_uri(receipt.object.uri, expected_domain=StorageDomain.EQUITY)
    assert object_path.read_bytes() == data
    assert writer.put_raw(
        StorageDomain.EQUITY,
        data,
        source_id="sec-edgar",
        retrieved_at=NOW,
        media_type="application/json",
        metadata={"accession": "0001"},
    ) == receipt

    object_path.write_bytes(b"tampered-provider-payload")
    with pytest.raises(SourceWriteConflict, match="raw content address conflict"):
        writer.put_raw(
            StorageDomain.EQUITY,
            data,
            source_id="sec-edgar",
            retrieved_at=NOW,
            media_type="application/json",
        )


def test_normalized_versions_are_immutable_and_require_new_identity_for_changes(tmp_path: Path) -> None:
    platform, _ = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    raw = writer.put_raw(
        StorageDomain.CRYPTO,
        b"funding-source",
        source_id="venue-a",
        retrieved_at=NOW,
        media_type="application/json",
    )
    arguments = {
        "domain": StorageDomain.CRYPTO,
        "dataset_id": "funding-history",
        "version_id": "2026-07-11-v1",
        "storage_format": StorageFormat.PARQUET,
        "schema_digest": parquet_schema_digest(parquet_bytes("funding-v1", 8)),
        "source_raw_digests": (raw.object.digest,),
        "created_at": NOW,
    }
    first = writer.publish_normalized(
        **arguments,
        partitions={"symbol=BTC_USDT": PartitionPayload(parquet_bytes("funding-v1", 8), row_count=8)},
    )
    assert writer.publish_normalized(
        **arguments,
        partitions={"symbol=BTC_USDT": PartitionPayload(parquet_bytes("funding-v1", 8), row_count=8)},
    ) == first
    with pytest.raises(SourceWriteConflict, match="already published"):
        writer.publish_normalized(
            **arguments,
            partitions={"symbol=BTC_USDT": PartitionPayload(parquet_bytes("changed", 8), row_count=8)},
        )


def test_partial_normalized_publish_never_exposes_a_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform, _ = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    raw = writer.put_raw(
        StorageDomain.FILINGS,
        b"filing-source",
        source_id="sec-edgar",
        retrieved_at=NOW,
        media_type="text/html",
    )
    import src.data_platform.platform as platform_module

    original = platform_module.write_new_file

    def fail_partition(path: Path, data: bytes) -> None:
        if path.parent.name == "partitions":
            raise OSError("injected normalized partition failure")
        original(path, data)

    monkeypatch.setattr(platform_module, "write_new_file", fail_partition)
    with pytest.raises(OSError, match="injected normalized"):
        writer.publish_normalized(
            StorageDomain.FILINGS,
            dataset_id="sec-filings",
            version_id="2026-07-11-v1",
            storage_format=StorageFormat.PARQUET,
            schema_digest=parquet_schema_digest(parquet_bytes("filing", 1)),
            source_raw_digests=(raw.object.digest,),
            created_at=NOW,
            partitions={"accession=0001": PartitionPayload(parquet_bytes("filing", 1), row_count=1)},
        )
    normalized_root = platform.root / "filings" / "normalized"
    assert not list(normalized_root.rglob("manifest.json"))
    assert not list(normalized_root.rglob("*.tmp-*"))


def test_signed_snapshot_is_immutable_and_training_view_is_source_read_only(tmp_path: Path) -> None:
    platform, signer = platform_with_signer(tmp_path)
    snapshot_digest, partition_data = publish_fx_snapshot(platform, signer)
    with pytest.raises(DataIntegrityError, match="read-only"):
        platform.training_view(StorageDomain.FX, read_only_root=platform.root)
    view, read_root = training_view(platform, StorageDomain.FX, tmp_path / "training-root")
    manifest = view.load_snapshot(snapshot_digest)
    assert content_digest(manifest) == snapshot_digest
    assert view.read_snapshot_partition(snapshot_digest, manifest.partitions[0].partition_id) == partition_data
    assert not hasattr(view, "put_raw")
    assert not hasattr(view, "publish_normalized")
    assert not hasattr(view, "publish_snapshot")
    assert not hasattr(view, "_platform")
    assert not (read_root / StorageDomain.FX.value / "raw").exists()
    assert not (read_root.stat().st_mode & 0o222)

    partition = manifest.partitions[0]
    from urllib.parse import urlsplit

    normalized_path = read_root / StorageDomain.FX.value / urlsplit(partition.uri).path.lstrip("/")
    normalized_path.chmod(0o644)
    normalized_path.write_bytes(parquet_bytes("tampered-same-tier", 7))
    with pytest.raises(DataIntegrityError, match="integrity failure"):
        view.read_snapshot_partition(snapshot_digest, partition.partition_id)


def test_snapshot_refuses_raw_or_cross_domain_source_paths(tmp_path: Path) -> None:
    platform, signer = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    raw = writer.put_raw(
        StorageDomain.FX,
        b"raw-bars",
        source_id="oanda",
        retrieved_at=NOW,
        media_type="application/json",
    )
    manifest = DatasetManifest(
        dataset_id="invalid-raw-snapshot",
        asset_class=AssetClass.FX,
        domain="daily-bars",
        provider="oanda",
        retrieved_at=NOW,
        coverage_start=date(2026, 7, 1),
        coverage_end=date(2026, 7, 10),
        instruments=("EUR_USD",),
        data_schema_version="v1",
        point_in_time_rules=("causal",),
        normalized_manifest_digests=("f" * 64,),
        partitions=(
            PartitionRef(
                partition_id="raw",
                uri=raw.object.uri,
                digest=raw.object.digest,
                size_bytes=raw.object.size_bytes,
            ),
        ),
    )
    with pytest.raises(DataIntegrityError, match="only normalized"):
        writer.publish_snapshot(StorageDomain.FX, signer.sign(manifest, created_at=NOW))


def test_feature_outputs_are_keyed_by_snapshot_and_pipeline_digest(tmp_path: Path) -> None:
    platform, signer = platform_with_signer(tmp_path)
    snapshot_digest, _ = publish_fx_snapshot(platform, signer)
    pipeline_digest = "d" * 64
    features = parquet_bytes("risk-features-v1", 7)
    writer = platform.feature_writer(StorageDomain.FX)
    bundle = writer.publish(
        snapshot_digest=snapshot_digest,
        feature_pipeline_digest=pipeline_digest,
        storage_format=StorageFormat.PARQUET,
        created_at=NOW,
        outputs={"risk-target-features": PartitionPayload(features, row_count=7)},
    )
    assert bundle.snapshot_digest == snapshot_digest
    assert bundle.feature_pipeline_digest == pipeline_digest
    view, _ = training_view(platform, StorageDomain.FX, tmp_path / "feature-training-root")
    assert view.read_feature_output(snapshot_digest, pipeline_digest, "risk-target-features") == features
    assert f"/{snapshot_digest}/{pipeline_digest}/" in bundle.outputs[0].uri
    assert not hasattr(writer, "put_raw")
    assert not hasattr(writer, "publish_normalized")

    with pytest.raises(SourceWriteConflict, match="change the pipeline digest"):
        writer.publish(
            snapshot_digest=snapshot_digest,
            feature_pipeline_digest=pipeline_digest,
            storage_format=StorageFormat.PARQUET,
            created_at=NOW,
            outputs={"risk-target-features": PartitionPayload(parquet_bytes("changed", 7), row_count=7)},
        )


def test_history_is_append_only_segmented_jsonl_not_rewritten_arrays(tmp_path: Path) -> None:
    platform, _ = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    first = writer.append_history(
        StorageDomain.OUTCOMES,
        "closed-trades",
        {"trade_id": "T1", "pnl": 1.5},
        expected_sequence=0,
    )
    first_bytes = first.path.read_bytes()
    second = writer.append_history(
        StorageDomain.OUTCOMES,
        "closed-trades",
        {"trade_id": "T2", "pnl": -0.5},
        expected_sequence=1,
    )
    assert first.path.read_bytes() == first_bytes
    assert second.sequence == 1
    assert first.path.suffix == second.path.suffix == ".jsonl"
    view, _ = training_view(platform, StorageDomain.OUTCOMES, tmp_path / "history-training-root")
    assert list(view.iter_history("closed-trades")) == [
        {"pnl": 1.5, "trade_id": "T1"},
        {"pnl": -0.5, "trade_id": "T2"},
    ]
    with pytest.raises(ControlStateConflict, match="history sequence changed"):
        writer.append_history(
            StorageDomain.OUTCOMES,
            "closed-trades",
            {"trade_id": "T3", "pnl": 0.1},
            expected_sequence=1,
        )


def test_small_control_state_is_disjoint_atomic_json_with_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform, _ = platform_with_signer(tmp_path)
    with pytest.raises(ValueError, match="must be disjoint"):
        platform.control_state_store(platform.root / "control")
    control = platform.control_state_store(tmp_path / "local-control", max_bytes=1024)
    first_digest = control.write("lane_halts", {"fx": True}, expected_digest=None)
    assert control.read("lane_halts") == {"fx": True}
    second_digest = control.write("lane_halts", {"fx": False}, expected_digest=first_digest)
    assert second_digest != first_digest
    with pytest.raises(ControlStateConflict, match="control state changed"):
        control.write("lane_halts", {"fx": True}, expected_digest=first_digest)
    assert control.read("lane_halts") == {"fx": False}

    import src.data_platform.io as io_module

    original_replace = os.replace

    def fail_replace(source, destination) -> None:
        if Path(destination).name == "lane_halts.json":
            raise OSError("injected control replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected control"):
        control.write("lane_halts", {"fx": True}, expected_digest=second_digest)
    assert control.read("lane_halts") == {"fx": False}
    assert not list(control.root.glob(".*.tmp-*"))


def test_invalid_columnar_container_is_rejected_before_publication(tmp_path: Path) -> None:
    platform, _ = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    raw = writer.put_raw(
        StorageDomain.PORTFOLIO,
        b"portfolio-source",
        source_id="local-book",
        retrieved_at=NOW,
        media_type="application/json",
    )
    with pytest.raises(DataIntegrityError, match="invalid parquet"):
        writer.publish_normalized(
            StorageDomain.PORTFOLIO,
            dataset_id="portfolio-returns",
            version_id="v1",
            storage_format=StorageFormat.PARQUET,
            schema_digest="e" * 64,
            source_raw_digests=(raw.object.digest,),
            created_at=NOW,
            partitions={"lane=fx": PartitionPayload(b"not-parquet", row_count=1)},
        )

    with pytest.raises(DataIntegrityError, match="invalid parquet"):
        writer.publish_normalized(
            StorageDomain.PORTFOLIO,
            dataset_id="portfolio-returns",
            version_id="v2",
            storage_format=StorageFormat.PARQUET,
            schema_digest="e" * 64,
            source_raw_digests=(raw.object.digest,),
            created_at=NOW,
            partitions={
                "lane=fx": PartitionPayload(
                    b"PAR1this-is-not-a-parquet-filePAR1", row_count=1
                )
            },
        )

    with pytest.raises(DataIntegrityError, match="invalid arrow"):
        writer.publish_normalized(
            StorageDomain.PORTFOLIO,
            dataset_id="portfolio-returns",
            version_id="v3",
            storage_format=StorageFormat.ARROW_IPC,
            schema_digest="e" * 64,
            source_raw_digests=(raw.object.digest,),
            created_at=NOW,
            partitions={
                "lane=fx": PartitionPayload(
                    b"ARROW1this-is-not-an-arrow-fileARROW1", row_count=1
                )
            },
        )

    valid_arrow = arrow_bytes("portfolio", 2)
    arrow_schema = arrow_ipc.open_file(pa.BufferReader(valid_arrow)).schema
    published = writer.publish_normalized(
        StorageDomain.PORTFOLIO,
        dataset_id="portfolio-returns",
        version_id="valid-arrow",
        storage_format=StorageFormat.ARROW_IPC,
        schema_digest=sha256_bytes(arrow_schema.serialize().to_pybytes()),
        source_raw_digests=(raw.object.digest,),
        created_at=NOW,
        partitions={"lane=fx": PartitionPayload(valid_arrow, row_count=2)},
    )
    assert published.partitions[0].row_count == 2


def test_snapshot_rejects_orphan_normalized_file(tmp_path: Path) -> None:
    platform, signer = platform_with_signer(tmp_path)
    data = parquet_bytes("orphan", 1)
    orphan = platform.root / "fx" / "normalized" / "orphan" / "partitions" / "orphan.parquet"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(data)
    manifest = DatasetManifest(
        dataset_id="orphan-snapshot",
        asset_class=AssetClass.FX,
        domain="daily-bars",
        provider="manual",
        retrieved_at=NOW,
        coverage_start=date(2026, 7, 1),
        coverage_end=date(2026, 7, 10),
        instruments=("EUR_USD",),
        data_schema_version="v1",
        point_in_time_rules=("causal",),
        normalized_manifest_digests=("f" * 64,),
        partitions=(
            PartitionRef(
                partition_id="orphan",
                uri="axiom-data://fx/normalized/orphan/partitions/orphan.parquet",
                digest=sha256_bytes(data),
                size_bytes=len(data),
                row_count=1,
            ),
        ),
    )
    with pytest.raises(DataIntegrityError, match="normalized manifest"):
        platform.ingestion_writer().publish_snapshot(
            StorageDomain.FX, signer.sign(manifest, created_at=NOW)
        )


def test_corrupt_raw_lineage_invalidates_normalized_and_snapshot_publication(tmp_path: Path) -> None:
    platform, signer = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    raw = writer.put_raw(
        StorageDomain.FX,
        b"provider-source",
        source_id="oanda",
        retrieved_at=NOW,
        media_type="application/json",
    )
    data = parquet_bytes("EUR_USD", 2)
    version = writer.publish_normalized(
        StorageDomain.FX,
        dataset_id="fx-daily",
        version_id="raw-corruption-test",
        storage_format=StorageFormat.PARQUET,
        schema_digest=parquet_schema_digest(data),
        source_raw_digests=(raw.object.digest,),
        created_at=NOW,
        partitions={"instrument=EUR_USD": PartitionPayload(data, row_count=2)},
    )
    raw_path = platform._resolve_uri(raw.object.uri, expected_domain=StorageDomain.FX)
    raw_path.write_bytes(b"corrupt-provider-source")
    normalized = version.partitions[0]
    manifest = DatasetManifest(
        dataset_id="corrupt-lineage-snapshot",
        asset_class=AssetClass.FX,
        domain="daily-bars",
        provider="oanda",
        retrieved_at=NOW,
        coverage_start=date(2026, 7, 1),
        coverage_end=date(2026, 7, 10),
        instruments=("EUR_USD",),
        data_schema_version="v1",
        point_in_time_rules=("causal",),
        normalized_manifest_digests=(content_digest(version),),
        partitions=(
            PartitionRef(
                partition_id=normalized.partition_id,
                uri=normalized.uri,
                digest=normalized.digest,
                size_bytes=normalized.size_bytes,
                row_count=normalized.row_count,
            ),
        ),
    )
    with pytest.raises(DataIntegrityError, match="raw lineage object is corrupt"):
        writer.publish_snapshot(StorageDomain.FX, signer.sign(manifest, created_at=NOW))


def test_missing_raw_receipt_invalidates_published_snapshot_replay(tmp_path: Path) -> None:
    platform, signer = platform_with_signer(tmp_path)
    snapshot_digest, _ = publish_fx_snapshot(platform, signer)
    receipts = list((platform.root / "fx" / "raw" / "receipts").rglob("*.json"))
    assert len(receipts) == 1
    receipts[0].unlink()
    with pytest.raises(DataIntegrityError, match="raw lineage receipt"):
        platform._load_snapshot(StorageDomain.FX, snapshot_digest)


def test_snapshot_requires_dedicated_authorized_signer(tmp_path: Path) -> None:
    platform, authorized = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    snapshot_digest, _ = publish_fx_snapshot(platform, authorized)
    original = platform._load_snapshot(StorageDomain.FX, snapshot_digest)
    unauthorized = Ed25519Signer.generate()
    platform.trust_store.add(unauthorized.trusted_key(valid_from=NOW - timedelta(days=1)))
    with pytest.raises(DataIntegrityError, match="not authorized"):
        writer.publish_snapshot(
            StorageDomain.FX, unauthorized.sign(original, created_at=NOW)
        )


def test_declared_row_count_and_schema_must_match_parsed_metadata(tmp_path: Path) -> None:
    platform, _ = platform_with_signer(tmp_path)
    writer = platform.ingestion_writer()
    raw = writer.put_raw(
        StorageDomain.CRYPTO,
        b"funding-source",
        source_id="venue",
        retrieved_at=NOW,
        media_type="application/json",
    )
    data = parquet_bytes("funding", 3)
    common = dict(
        domain=StorageDomain.CRYPTO,
        dataset_id="funding",
        storage_format=StorageFormat.PARQUET,
        source_raw_digests=(raw.object.digest,),
        created_at=NOW,
    )
    with pytest.raises(DataIntegrityError, match="row_count"):
        writer.publish_normalized(
            **common,
            version_id="wrong-rows",
            schema_digest=parquet_schema_digest(data),
            partitions={"BTC": PartitionPayload(data, row_count=2)},
        )
    with pytest.raises(DataIntegrityError, match="schema_digest"):
        writer.publish_normalized(
            **common,
            version_id="wrong-schema",
            schema_digest="f" * 64,
            partitions={"BTC": PartitionPayload(data, row_count=3)},
        )
