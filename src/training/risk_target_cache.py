"""Immutable per-pair risk-target feature cache (Phase F)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.evidence.canonical import canonical_bytes
from src.training.risk_target_features import (
    FEATURE_COLUMNS,
    RISK_TARGET_FEATURE_PIPELINE_VERSION,
    compute_risk_target_features,
)
from src.training.performance.partition_store import file_sha256


def raw_frame_digest(frame: pd.DataFrame) -> str:
    """Stable content identity including column order, dtypes, index, and values."""
    digest = hashlib.sha256()
    digest.update(canonical_bytes(list(frame.columns.astype(str))))
    digest.update(canonical_bytes([str(dtype) for dtype in frame.dtypes]))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


class RiskTargetFeatureCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def get_or_compute(self, pair: str, raw: pd.DataFrame) -> pd.DataFrame:
        raw_digest = raw_frame_digest(raw)
        directory = self.root / pair / RISK_TARGET_FEATURE_PIPELINE_VERSION
        data_path = directory / f"{raw_digest}.parquet"
        manifest_path = directory / f"{raw_digest}.manifest.json"
        if data_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("raw_digest") != raw_digest:
                raise ValueError("risk-target cache raw digest mismatch")
            payload_digest = file_sha256(data_path)
            if payload_digest != manifest.get("artifact_sha256"):
                raise ValueError("risk-target cached feature partition is corrupt")
            cached = pd.read_parquet(data_path)
            missing = set(FEATURE_COLUMNS) - set(cached.columns)
            if missing:
                raise ValueError(f"risk-target cached features missing {sorted(missing)}")
            return cached

        features = compute_risk_target_features(raw)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = data_path.with_suffix(".parquet.tmp")
        features.to_parquet(tmp, index=False)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, data_path)
        manifest = {
            "schema": "axiom_risk_target_feature_partition_v1",
            "pair": pair,
            "pipeline_version": RISK_TARGET_FEATURE_PIPELINE_VERSION,
            "raw_digest": raw_digest,
            "rows": len(features),
            "columns": list(features.columns),
            "artifact_sha256": file_sha256(data_path),
        }
        manifest_path.write_bytes(canonical_bytes(manifest))
        return features


def pool_final_valid_rows(
    per_pair_frames: Mapping[str, pd.DataFrame],
    *,
    required_columns: Sequence[str] = tuple(FEATURE_COLUMNS),
) -> pd.DataFrame:
    """Drop warm-up/invalid rows per pair before pooling, never after cross-pair concat."""
    valid_frames: list[pd.DataFrame] = []
    for pair, frame in sorted(per_pair_frames.items()):
        missing = set(required_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{pair} missing required features {sorted(missing)}")
        valid = frame.loc[frame[list(required_columns)].notna().all(axis=1)].copy()
        valid["pair"] = pair
        valid_frames.append(valid)
    if not valid_frames:
        return pd.DataFrame(columns=[*required_columns, "pair"])
    pooled = pd.concat(valid_frames, ignore_index=True)
    if "date" in pooled.columns:
        pooled["date"] = pd.to_datetime(pooled["date"], utc=True, errors="raise")
        pooled = pooled.sort_values(["date", "pair"]).reset_index(drop=True)
    return pooled
