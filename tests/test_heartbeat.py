"""US-603: Heartbeat writer + external watchdog tests."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.tui.heartbeat import (
    HEARTBEAT_RELPATH,
    heartbeat_path,
    read_heartbeat,
    write_heartbeat,
)


def test_heartbeat_schema(tmp_path: Path) -> None:
    """heartbeat.json must contain the required keys with the correct types."""
    write_heartbeat(
        tmp_path,
        cycle_count=7,
        mode="live",
        scanner_alive=True,
        last_error_ts=None,
    )

    target = tmp_path / HEARTBEAT_RELPATH
    assert target.exists(), f"heartbeat file not written to {target}"

    with target.open("r") as f:
        data = json.load(f)

    required = {"ts_iso", "pid", "cycle_count", "mode", "scanner_alive", "last_error_ts"}
    assert required.issubset(data.keys()), f"missing keys: {required - data.keys()}"

    assert isinstance(data["ts_iso"], str)
    # Timestamp must parse as ISO8601 UTC.
    parsed = datetime.fromisoformat(data["ts_iso"])
    assert parsed.tzinfo is not None

    assert isinstance(data["pid"], int) and data["pid"] > 0
    assert data["cycle_count"] == 7
    assert data["mode"] == "live"
    assert data["scanner_alive"] is True
    assert data["last_error_ts"] is None


def test_atomic_write_no_torn_reads(tmp_path: Path) -> None:
    """Concurrent writers + readers must never observe a partial/empty file."""
    # Seed a valid file first so readers always have something.
    write_heartbeat(tmp_path, cycle_count=0, mode="demo", scanner_alive=False)

    target = heartbeat_path(tmp_path)
    stop = threading.Event()
    torn = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            write_heartbeat(
                tmp_path,
                cycle_count=i,
                mode="live" if i % 2 else "demo",
                scanner_alive=bool(i % 2),
            )
            i += 1

    def reader() -> None:
        while not stop.is_set():
            try:
                with target.open("r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                torn.append(repr(exc))
                continue
            # Schema spot-check.
            if not {"ts_iso", "pid", "cycle_count"}.issubset(data.keys()):
                torn.append(f"missing keys: {data!r}")

    threads = [threading.Thread(target=writer) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()

    # Run the race for ~0.75s — plenty of write/read cycles.
    threading.Event().wait(0.75)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert not torn, f"observed torn reads: {torn[:5]}"

    # No leftover .tmp files in the heartbeat directory.
    leftovers = list(target.parent.glob(".heartbeat.*.tmp"))
    assert not leftovers, f"tmp files leaked: {leftovers}"


def test_read_heartbeat_round_trip(tmp_path: Path) -> None:
    """write_heartbeat + read_heartbeat round-trip — proxy for 'TUI launches write
    heartbeat within 15s' (we cannot launch a Textual app in unit tests, so we
    invoke the same writer the TUI's set_interval(10s) calls)."""
    assert read_heartbeat(tmp_path) is None  # nothing yet

    write_heartbeat(
        tmp_path,
        cycle_count=42,
        mode="demo",
        scanner_alive=False,
        last_error_ts="2026-04-24T00:00:00+00:00",
    )

    data = read_heartbeat(tmp_path)
    assert data is not None
    assert data["cycle_count"] == 42
    assert data["mode"] == "demo"
    assert data["scanner_alive"] is False
    assert data["last_error_ts"] == "2026-04-24T00:00:00+00:00"
    # Confirm freshness — written within the last 15 seconds.
    written = datetime.fromisoformat(data["ts_iso"])
    age = (datetime.now(timezone.utc) - written).total_seconds()
    assert 0 <= age < 15, f"heartbeat age {age}s outside 15s window"
