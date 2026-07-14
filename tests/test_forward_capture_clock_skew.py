from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data_platform.forward_capture import ForwardCaptureService
from src.data_platform.platform import DataPlatform
from src.evidence.signing import TrustStore


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def build_service(tmp_path: Path, clock: MutableClock) -> ForwardCaptureService:
    platform = DataPlatform(tmp_path / "axiom-data", trust_store=TrustStore())
    return ForwardCaptureService(platform, clock=clock)


def parquet_bytes(label: str) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"label": [label]}), sink)
    return sink.getvalue().to_pybytes()


def test_tick_capture_accepts_bounded_provider_clock_skew(tmp_path: Path) -> None:
    clock = MutableClock(datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc))
    service = build_service(tmp_path, clock)
    provider_time = clock.value + timedelta(milliseconds=250)

    record = service.capture_tick_partition(
        "EUR_USD",
        [{"time": provider_time.isoformat(), "bid": 1.1, "ask": 1.1002}],
        parquet_bytes("bounded-provider-skew"),
    )

    assert record is not None and record.training_eligible
    assert record.source_event_time_utc == provider_time
    assert record.observed_at_utc == provider_time
    assert record.captured_at_utc == provider_time


def test_tick_capture_rejects_materially_future_provider_time(tmp_path: Path) -> None:
    clock = MutableClock(datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc))
    service = build_service(tmp_path, clock)

    with pytest.raises(ValueError, match="cannot be later"):
        service.capture_tick_partition(
            "EUR_USD",
            [{"time": (clock.value + timedelta(minutes=1)).isoformat(), "bid": 1.1, "ask": 1.1002}],
            parquet_bytes("material-provider-skew"),
        )
    assert not list((tmp_path / "axiom-data").rglob("manifest.json"))
