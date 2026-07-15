"""Immutable Parquet partitions with entity/date predicate pushdown."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.evidence.canonical import canonical_bytes

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str, label: str) -> str:
    text = str(value)
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(f"unsafe {label}: {value!r}")
    return text


class ImmutableMonthlyParquetStore:
    """One entity/month per immutable file; readers filter before table materialization."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write(
        self,
        namespace: str,
        entity: str,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "event_time",
    ) -> tuple[Path, ...]:
        namespace = _safe(namespace, "namespace")
        entity = _safe(entity, "entity")
        if timestamp_column not in frame.columns:
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError(
                    f"frame requires {timestamp_column!r} or a DatetimeIndex"
                )
            materialized = frame.reset_index().rename(
                columns={frame.index.name or "index": timestamp_column}
            )
        else:
            materialized = frame.copy()
        materialized[timestamp_column] = pd.to_datetime(
            materialized[timestamp_column], utc=True, errors="raise"
        )
        materialized["entity"] = entity
        materialized["year"] = materialized[timestamp_column].dt.year.astype("int16")
        materialized["month"] = materialized[timestamp_column].dt.month.astype("int8")
        outputs: list[Path] = []
        for (year, month), part in materialized.groupby(["year", "month"], sort=True):
            part = part.sort_values(timestamp_column).reset_index(drop=True)
            stored = part.drop(columns=["entity", "year", "month"])
            stored.columns.name = None
            directory = (
                self.root
                / namespace
                / f"entity={entity}"
                / f"year={int(year):04d}"
                / f"month={int(month):02d}"
            )
            existing_paths = sorted(directory.glob("*.parquet")) if directory.exists() else []
            if existing_paths:
                existing = pd.concat(
                    [pd.read_parquet(path) for path in existing_paths], ignore_index=True
                ).sort_values(timestamp_column).reset_index(drop=True)
                overlap = stored[timestamp_column].isin(existing[timestamp_column])
                if bool(overlap.any()):
                    old_overlap = existing[
                        existing[timestamp_column].isin(stored.loc[overlap, timestamp_column])
                    ].sort_values(timestamp_column).reset_index(drop=True)
                    new_overlap = stored.loc[overlap].sort_values(timestamp_column).reset_index(drop=True)
                    try:
                        pd.testing.assert_frame_equal(
                            old_overlap[new_overlap.columns],
                            new_overlap,
                            check_dtype=True,
                            check_exact=True,
                        )
                    except (AssertionError, KeyError) as exc:
                        raise ValueError(
                            f"immutable partition revision refused for {entity} "
                            f"{int(year):04d}-{int(month):02d}"
                        ) from exc
                stored = stored[
                    ~stored[timestamp_column].isin(existing[timestamp_column])
                ].reset_index(drop=True)
                if stored.empty:
                    outputs.extend(existing_paths)
                    continue
            table = pa.Table.from_pandas(stored, preserve_index=False)
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink, compression="zstd")
            payload = sink.getvalue().to_pybytes()
            digest = hashlib.sha256(payload).hexdigest()
            path = directory / f"{digest}.parquet"
            if path.exists():
                if file_sha256(path) != digest:
                    raise ValueError(f"existing immutable partition is corrupt: {path}")
                outputs.append(path)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            manifest = {
                "schema": "axiom_monthly_partition_v1",
                "namespace": namespace,
                "entity": entity,
                "year": int(year),
                "month": int(month),
                "rows": len(stored),
                "sha256": digest,
                "columns": list(stored.columns),
                "first_event_time": stored[timestamp_column].iloc[0].isoformat(),
                "last_event_time": stored[timestamp_column].iloc[-1].isoformat(),
            }
            path.with_suffix(".manifest.json").write_bytes(canonical_bytes(manifest))
            outputs.append(path)
        return tuple(outputs)

    def query(
        self,
        namespace: str,
        *,
        entities: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        columns: Sequence[str] | None = None,
        timestamp_column: str = "event_time",
    ) -> pd.DataFrame:
        namespace = _safe(namespace, "namespace")
        if not entities:
            raise ValueError("query requires at least one entity predicate")
        safe_entities = tuple(_safe(entity, "entity") for entity in entities)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tz is None else start_ts.tz_convert("UTC")
        end_ts = end_ts.tz_localize("UTC") if end_ts.tz is None else end_ts.tz_convert("UTC")
        if end_ts < start_ts:
            raise ValueError("end must be on or after start")
        base = self.root / namespace
        if not base.exists():
            return pd.DataFrame(columns=list(columns or ()))
        dataset = ds.dataset(base, format="parquet", partitioning="hive", exclude_invalid_files=True)
        predicate = (
            ds.field("entity").isin(safe_entities)
            & (ds.field(timestamp_column) >= pa.scalar(start_ts.to_pydatetime()))
            & (ds.field(timestamp_column) <= pa.scalar(end_ts.to_pydatetime()))
        )
        selected = list(columns) if columns is not None else None
        required = [timestamp_column, "entity"]
        if selected is not None:
            selected = list(dict.fromkeys([*required, *selected]))
        table = dataset.to_table(filter=predicate, columns=selected)
        frame = table.to_pandas()
        if frame.empty:
            return frame
        return frame.sort_values([timestamp_column, "entity"]).reset_index(drop=True)


def bounded_batches(values: Iterable[object], *, batch_size: int) -> Iterable[tuple[object, ...]]:
    """Yield bounded tuples so no worker receives an unbounded corpus."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch: list[object] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)
