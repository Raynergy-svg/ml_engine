"""Compute-once, fixed-window, memory-mapped matrices for the legacy RL suite."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.evidence.canonical import canonical_bytes


class BatchPredictor(Protocol):
    def predict(self, values: np.ndarray, *, verbose: int, batch_size: int) -> Any: ...


@dataclass(frozen=True)
class RLMatrixBundle:
    features: np.ndarray
    ensemble_predictions: np.ndarray
    prices: np.ndarray

    def validate(self) -> None:
        n_rows = len(self.features)
        if self.features.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix")
        if self.ensemble_predictions.shape != (n_rows, 2):
            raise ValueError("ensemble_predictions must have shape (rows, 2)")
        if self.prices.shape != (n_rows,):
            raise ValueError("prices must have shape (rows,)")
        for name, values in (
            ("features", self.features),
            ("ensemble_predictions", self.ensemble_predictions),
            ("prices", self.prices),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")


def fixed_window_view(features: np.ndarray, *, sequence_length: int) -> np.ndarray:
    """Return the legacy [i-seq:i] windows as a non-copying, read-only view."""
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if sequence_length < 1 or len(matrix) <= sequence_length:
        raise ValueError("sequence_length must leave at least one prediction window")
    windows = np.lib.stride_tricks.sliding_window_view(
        matrix, window_shape=sequence_length, axis=0
    )
    # NumPy places the new window axis last: (rows, features, sequence).
    # Move it to the model's expected (rows, sequence, features), and discard
    # the final window to preserve the legacy range(seq_len, len(X)) contract.
    view = np.moveaxis(windows, -1, 1)[:-1]
    view.flags.writeable = False
    return view


def predict_fixed_windows(
    model: BatchPredictor,
    features: np.ndarray,
    *,
    sequence_length: int,
    feature_dimension: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Align once and invoke model inference in bounded batches without a window list."""
    if feature_dimension < 1 or batch_size < 1:
        raise ValueError("feature_dimension and batch_size must be positive")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape[1] >= feature_dimension:
        aligned = matrix[:, :feature_dimension]
    else:
        aligned = np.pad(matrix, ((0, 0), (0, feature_dimension - matrix.shape[1])))
    windows = fixed_window_view(aligned, sequence_length=sequence_length)
    outputs: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = np.ascontiguousarray(windows[start : start + batch_size])
        prediction = np.asarray(
            model.predict(batch, verbose=0, batch_size=batch_size)
        ).reshape(-1)
        outputs.append(prediction.astype(np.float32, copy=False))
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(canonical_bytes(list(contiguous.shape)))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _file_digest(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_bundle_digest(bundle: RLMatrixBundle, *, source_identity: str) -> str:
    bundle.validate()
    payload = {
        "source_identity": source_identity,
        "features": _array_digest(bundle.features),
        "ensemble_predictions": _array_digest(bundle.ensemble_predictions),
        "prices": _array_digest(bundle.prices),
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def materialize_rl_matrices(
    bundle: RLMatrixBundle,
    root: Path,
    *,
    source_identity: str,
) -> Path:
    """Write immutable `.npy` matrices once and return their content-addressed directory."""
    digest = matrix_bundle_digest(bundle, source_identity=source_identity)
    destination = root / digest
    if destination.exists():
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=root))
    try:
        arrays = {
            "features": np.asarray(bundle.features, dtype=np.float32),
            "ensemble_predictions": np.asarray(
                bundle.ensemble_predictions, dtype=np.float32
            ),
            "prices": np.asarray(bundle.prices, dtype=np.float32),
        }
        files: dict[str, dict[str, Any]] = {}
        for name, values in arrays.items():
            path = temp / f"{name}.npy"
            with path.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            files[name] = {
                "file": path.name,
                "sha256": _file_digest(path),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        manifest = {
            "schema": "axiom_rl_matrices_v1",
            "dataset_digest": digest,
            "source_identity": source_identity,
            "rows": len(bundle.features),
            "files": files,
        }
        manifest_path = temp / "manifest.json"
        manifest_path.write_bytes(canonical_bytes(manifest))
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.replace(temp, destination)
        except OSError:
            if not destination.exists():
                raise
        return destination
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_rl_matrices(path: Path, *, mmap_mode: str = "r") -> RLMatrixBundle:
    """Verify the manifest and load arrays without copying them into RAM."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "axiom_rl_matrices_v1":
        raise ValueError("unsupported RL matrix manifest")
    arrays: dict[str, np.ndarray] = {}
    for name in ("features", "ensemble_predictions", "prices"):
        meta = manifest["files"][name]
        array_path = path / str(meta["file"])
        if _file_digest(array_path) != meta["sha256"]:
            raise ValueError(f"RL matrix digest mismatch: {name}")
        arrays[name] = np.load(array_path, mmap_mode=mmap_mode, allow_pickle=False)
    bundle = RLMatrixBundle(
        features=arrays["features"],
        ensemble_predictions=arrays["ensemble_predictions"],
        prices=arrays["prices"],
    )
    bundle.validate()
    return bundle


def find_rl_matrices(root: Path, *, source_identity: str) -> Path | None:
    """Resolve an already-materialized source without recomputing any features."""
    if not root.exists():
        return None
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema") == "axiom_rl_matrices_v1"
            and manifest.get("source_identity") == source_identity
        ):
            return manifest_path.parent
    return None
