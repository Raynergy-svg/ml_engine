"""Phase F partitioned crypto research data and eligibility artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.evidence.canonical import canonical_bytes
from src.training.performance.partition_store import (
    ImmutableMonthlyParquetStore,
    file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "crypto_cache" / "partitioned"


def partition_history(
    symbol: str,
    kind: str,
    frame: pd.DataFrame,
    *,
    root: Path = DEFAULT_ROOT,
) -> tuple[Path, ...]:
    return ImmutableMonthlyParquetStore(root).write(kind, symbol, frame)


def load_history(
    symbols: Sequence[str],
    kind: str,
    *,
    start: str,
    end: str,
    columns: Sequence[str] | None = None,
    root: Path = DEFAULT_ROOT,
) -> pd.DataFrame:
    return ImmutableMonthlyParquetStore(root).query(
        kind,
        entities=symbols,
        start=start,
        end=end,
        columns=columns,
    )


def materialize_eligibility(
    eligible: pd.DataFrame,
    *,
    strategy_id: str,
    parameters: dict[str, object],
    root: Path = DEFAULT_ROOT,
) -> Path:
    """Store strategy-specific eligibility separately from momentum/carry inputs."""
    long = (
        eligible.astype(bool)
        .rename_axis(index="event_time", columns="symbol")
        .stack(future_stack=True)
        .rename("eligible")
        .reset_index()
    )
    payload_identity = {
        "strategy_id": strategy_id,
        "parameters": parameters,
        "rows": len(long),
        "first": pd.Timestamp(long["event_time"].min()).isoformat() if len(long) else None,
        "last": pd.Timestamp(long["event_time"].max()).isoformat() if len(long) else None,
        "eligibility_digest": hashlib.sha256(
            pd.util.hash_pandas_object(long, index=True).values.tobytes()
        ).hexdigest(),
    }
    digest = hashlib.sha256(canonical_bytes(payload_identity)).hexdigest()
    destination = root / "eligibility" / strategy_id / f"{digest}.parquet"
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".parquet.tmp")
    long.to_parquet(tmp, index=False)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    destination.with_suffix(".manifest.json").write_bytes(
        canonical_bytes({**payload_identity, "artifact_sha256": file_sha256(destination)})
    )
    return destination


def materialize_strategy_dataset(
    frame: pd.DataFrame,
    *,
    strategy_id: str,
    dataset_role: str,
    parameters: dict[str, object],
    root: Path = DEFAULT_ROOT,
) -> Path:
    """Freeze momentum/carry inputs in disjoint strategy namespaces."""
    ordered = frame.sort_index().sort_index(axis=1)
    content_digest = hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
    ).hexdigest()
    identity = {
        "strategy_id": strategy_id,
        "dataset_role": dataset_role,
        "parameters": parameters,
        "rows": len(ordered),
        "columns": [str(column) for column in ordered.columns],
        "content_digest": content_digest,
    }
    digest = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    destination = root / "strategies" / strategy_id / dataset_role / f"{digest}.parquet"
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".parquet.tmp")
    ordered.to_parquet(tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    destination.with_suffix(".manifest.json").write_bytes(
        canonical_bytes({**identity, "artifact_sha256": file_sha256(destination)})
    )
    return destination
