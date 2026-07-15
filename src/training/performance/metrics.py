"""Reproducible resource baselines for every AXIOM training family."""

from __future__ import annotations

import hashlib
import os
import platform
import resource
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import Field

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts.base import StrictContract

try:
    import psutil
except ImportError:  # optional: stdlib resource fallback remains fully functional
    psutil = None  # type: ignore[assignment]


def _stream_digest(path: Path, *, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class PerformanceBaseline(StrictContract):
    """The Phase F exit-gate report; units are explicit and machine-readable."""

    family: str = Field(min_length=1)
    workload: str = Field(min_length=1)
    measured_at_utc: datetime
    dataset: dict[str, Any]
    phases_seconds: dict[str, float]
    wall_time_seconds: float = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    cpu_utilization_percent: float = Field(ge=0)
    gpu_utilization: dict[str, Any]
    io_volume: dict[str, Any]
    artifact: dict[str, Any]
    environment: dict[str, Any]

    @property
    def baseline_digest(self) -> str:
        return hashlib.sha256(canonical_bytes(self)).hexdigest()


def _io_snapshot(process: Any | None) -> tuple[int, int, str]:
    if process is not None:
        try:
            counters = process.io_counters()
            return int(counters.read_bytes), int(counters.write_bytes), "process_io_counters"
        except (AttributeError, OSError):
            pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_inblock * 512), int(usage.ru_oublock * 512), "resource_blocks_512b"


def _rss_bytes(process: Any | None) -> int:
    if process is not None:
        try:
            return int(process.memory_info().rss)
        except (AttributeError, OSError):
            pass
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux/BSD commonly report KiB.
    return value if platform.system() == "Darwin" else value * 1024


def _gpu_snapshot() -> dict[str, Any]:
    """Best available utilization sample; absence is explicit, never invented as zero."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if values:
            return {
                "available": True,
                "backend": "nvidia-smi",
                "average_percent": sum(values) / len(values),
            }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import torch

        if bool(torch.backends.mps.is_available()):
            return {
                "available": True,
                "backend": "apple_mps",
                "average_percent": None,
                "measurement": "backend exposes no process utilization counter",
            }
    except (ImportError, AttributeError, RuntimeError):
        pass
    return {
        "available": False,
        "backend": "none",
        "average_percent": None,
        "measurement": "no supported GPU telemetry backend",
    }


class PerformanceProbe:
    """Measure one bounded workload, including peak RSS via a sampler thread."""

    def __init__(self, *, sample_interval_seconds: float = 0.01) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self._sample_interval = sample_interval_seconds
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._phases: dict[str, float] = {}
        self._phase_starts: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._start_io = (0, 0, "unsupported")
        self._gpu = _gpu_snapshot()

    def __enter__(self) -> "PerformanceProbe":
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_io = _io_snapshot(self._process)
        self._peak_rss = _rss_bytes(self._process)

        def sample() -> None:
            while not self._stop.wait(self._sample_interval):
                try:
                    self._peak_rss = max(self._peak_rss, _rss_bytes(self._process))
                except OSError:
                    return

        self._thread = threading.Thread(target=sample, name="axiom-rss-probe", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self._sample_interval * 3))
        self._peak_rss = max(self._peak_rss, _rss_bytes(self._process))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not name or name in self._phase_starts:
            raise ValueError(f"phase name must be unique and non-empty: {name!r}")
        self._phase_starts[name] = time.perf_counter()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + (
                time.perf_counter() - self._phase_starts.pop(name)
            )

    def baseline(
        self,
        *,
        family: str,
        workload: str,
        dataset: dict[str, Any],
        artifact_path: Path | None,
    ) -> PerformanceBaseline:
        if self._start_wall == 0.0:
            raise RuntimeError("PerformanceProbe must be entered before producing a baseline")
        wall = max(0.0, time.perf_counter() - self._start_wall)
        cpu = max(0.0, time.process_time() - self._start_cpu)
        end_read, end_write, end_method = _io_snapshot(self._process)
        start_read, start_write, start_method = self._start_io
        artifact: dict[str, Any] = {
            "path": None,
            "size_bytes": 0,
            "sha256": None,
        }
        if artifact_path is not None and artifact_path.exists():
            try:
                artifact_label = str(
                    artifact_path.resolve().relative_to(Path.cwd().resolve())
                )
            except ValueError:
                artifact_label = str(artifact_path)
            if artifact_path.is_file():
                artifact_digest, artifact_size = _stream_digest(artifact_path)
                artifact = {
                    "path": artifact_label,
                    "size_bytes": artifact_size,
                    "sha256": artifact_digest,
                }
            else:
                files = sorted(path for path in artifact_path.rglob("*") if path.is_file())
                digest = hashlib.sha256()
                size = 0
                for path in files:
                    rel = path.relative_to(artifact_path).as_posix().encode()
                    file_digest, file_size = _stream_digest(path)
                    digest.update(len(rel).to_bytes(8, "big"))
                    digest.update(rel)
                    digest.update(bytes.fromhex(file_digest))
                    size += file_size
                artifact = {
                    "path": artifact_label,
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
        return PerformanceBaseline(
            family=family,
            workload=workload,
            measured_at_utc=datetime.now(timezone.utc),
            dataset=dataset,
            phases_seconds={
                "feature_computation": float(self._phases.get("feature_computation", 0.0)),
                "fit": float(self._phases.get("fit", 0.0)),
                "evaluation": float(self._phases.get("evaluation", 0.0)),
                **{key: float(value) for key, value in sorted(self._phases.items())},
            },
            wall_time_seconds=wall,
            peak_rss_bytes=self._peak_rss,
            cpu_utilization_percent=(cpu / wall * 100.0) if wall > 0 else 0.0,
            gpu_utilization=self._gpu,
            io_volume={
                "read_bytes": max(0, end_read - start_read),
                "write_bytes": max(0, end_write - start_write),
                "measurement": end_method if end_method == start_method else "mixed",
                "logical_input_bytes": int(dataset.get("size_bytes", 0)),
                "logical_artifact_bytes": int(artifact["size_bytes"]),
            },
            artifact=artifact,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "logical_cpu_count": os.cpu_count(),
            },
        )


def write_baseline(report: PerformanceBaseline, path: Path) -> Path:
    """Atomically write canonical report bytes; the report itself carries no mutable alias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = canonical_bytes(report)
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path
