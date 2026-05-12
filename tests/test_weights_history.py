"""Tests for src/scanner/weights_history.py (Bucket D1)."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import time
from pathlib import Path

import pytest

from src.scanner import weights_history


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect HISTORY_PATH + LOCK_PATH into a tmp dir for the test."""
    hist = tmp_path / "weight_history.jsonl"
    lock = tmp_path / "agent_weights.json.lock"
    monkeypatch.setattr(weights_history, "HISTORY_PATH", hist)
    monkeypatch.setattr(weights_history, "LOCK_PATH", lock)
    return hist, lock


def _sample_weights(scale: float = 1.0) -> dict:
    return {
        "_global": {"trend": 1.0 * scale, "mean_reversion": 0.9 * scale},
        "LOW": {"trend": 0.8 * scale, "mean_reversion": 1.1 * scale},
    }


# ---------------------------------------------------------------------------
# record_weight_version
# ---------------------------------------------------------------------------

def test_record_appends_and_skips_duplicate(isolated_paths):
    hist, _ = isolated_paths
    weights = _sample_weights()

    h1 = weights_history.record_weight_version(weights, source="test:first")
    assert h1 is not None
    assert hist.exists()
    lines = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["source"] == "test:first"
    assert lines[0]["snapshot_hash"] == h1
    assert lines[0]["weights_snapshot"] == weights

    # Second call with identical weights — dedup, no second line.
    h2 = weights_history.record_weight_version(weights, source="test:second")
    assert h2 is None
    assert len(hist.read_text().splitlines()) == 1

    # Different weights → appended.
    h3 = weights_history.record_weight_version(_sample_weights(scale=1.5), source="test:third")
    assert h3 is not None and h3 != h1
    assert len(hist.read_text().splitlines()) == 2


def test_record_refuses_non_dict(isolated_paths):
    hist, _ = isolated_paths
    assert weights_history.record_weight_version([1, 2, 3], source="bad") is None  # type: ignore[arg-type]
    assert not hist.exists()


def test_record_under_lock_skips_acquire(isolated_paths):
    """When _under_lock=True we do NOT try to acquire the flock."""
    hist, lock = isolated_paths
    # Pre-acquire the lock in this process to simulate an outer flock holder.
    lock.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock, "a+", encoding="utf-8")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    try:
        # With _under_lock=True this must NOT deadlock (no re-acquire attempt).
        h = weights_history.record_weight_version(
            _sample_weights(), source="under_lock", _under_lock=True
        )
        assert h is not None
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    assert hist.exists()


def _worker_hold_lock(lock_path_str: str, hold_seconds: float, ready_evt, done_evt) -> None:
    """Subprocess helper: acquire the flock, signal ready, sleep, release."""
    fh = open(lock_path_str, "a+", encoding="utf-8")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    ready_evt.set()
    time.sleep(hold_seconds)
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    fh.close()
    done_evt.set()


def test_flock_contention_blocks_then_succeeds(isolated_paths):
    """When another process holds the flock, record_weight_version blocks then completes."""
    hist, lock = isolated_paths
    lock.parent.mkdir(parents=True, exist_ok=True)

    ready = multiprocessing.Event()
    done = multiprocessing.Event()
    p = multiprocessing.Process(
        target=_worker_hold_lock, args=(str(lock), 0.4, ready, done)
    )
    p.start()
    assert ready.wait(timeout=5), "subprocess never acquired lock"

    start = time.monotonic()
    h = weights_history.record_weight_version(_sample_weights(), source="contention")
    elapsed = time.monotonic() - start

    p.join(timeout=5)
    assert h is not None
    # We were blocked for the bulk of the subprocess's 0.4s hold.
    assert elapsed >= 0.2, f"call returned too fast (elapsed={elapsed:.3f}s) — flock not honored"
    assert hist.exists()


# ---------------------------------------------------------------------------
# list_versions / find_version
# ---------------------------------------------------------------------------

def test_list_versions_returns_newest_first(isolated_paths):
    hist, _ = isolated_paths
    for i in range(5):
        weights_history.record_weight_version(_sample_weights(scale=1.0 + 0.1 * i), source=f"s{i}")

    versions = weights_history.list_versions(limit=3)
    assert len(versions) == 3
    # Newest first → s4, s3, s2.
    assert [v["source"] for v in versions] == ["s4", "s3", "s2"]


def test_list_versions_skips_malformed_lines(isolated_paths):
    hist, _ = isolated_paths
    weights_history.record_weight_version(_sample_weights(), source="good")
    # Inject a bad line + a blank line.
    with open(hist, "a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("\n")
        fh.write(json.dumps({"ts_iso": "z", "source": "ok2", "snapshot_hash": "h" * 64,
                             "weights_snapshot": _sample_weights(scale=2.0), "reason": ""}) + "\n")

    versions = weights_history.list_versions(limit=10)
    sources = [v["source"] for v in versions]
    assert "good" in sources and "ok2" in sources
    assert len(versions) == 2  # malformed line excluded


def test_find_version_rejects_short_prefix(isolated_paths):
    weights_history.record_weight_version(_sample_weights(), source="x")
    assert weights_history.find_version("ab") is None  # too short
    assert weights_history.find_version("") is None


def test_find_version_returns_matching_entry(isolated_paths):
    h = weights_history.record_weight_version(_sample_weights(), source="match-me")
    assert h is not None
    rec = weights_history.find_version(h[:8])
    assert rec is not None
    assert rec["snapshot_hash"] == h


# ---------------------------------------------------------------------------
# restore_version (round-trip)
# ---------------------------------------------------------------------------

def test_restore_round_trip(isolated_paths, tmp_path):
    hist, _ = isolated_paths
    weights_path = tmp_path / "agent_weights.json"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    a = _sample_weights(scale=1.0)
    b = _sample_weights(scale=2.0)
    h_a = weights_history.record_weight_version(a, source="a")
    weights_history.record_weight_version(b, source="b")
    weights_path.write_text(json.dumps(b))

    rec = weights_history.restore_version(h_a[:8], weights_path=weights_path)
    assert rec["snapshot_hash"] == h_a

    restored = json.loads(weights_path.read_text())
    assert restored == a

    # A new history entry tagged rollback:<hash_prefix> should be appended.
    versions = weights_history.list_versions(limit=10)
    assert versions[0]["source"].startswith("rollback:")


def test_restore_raises_on_missing_hash(isolated_paths, tmp_path):
    weights_path = tmp_path / "agent_weights.json"
    with pytest.raises(LookupError):
        weights_history.restore_version("deadbeefdead", weights_path=weights_path)


# ---------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------

def test_rotation_at_max_lines(isolated_paths, monkeypatch):
    hist, _ = isolated_paths
    monkeypatch.setattr(weights_history, "MAX_LINES", 3)
    for i in range(5):
        weights_history.record_weight_version(_sample_weights(scale=1.0 + 0.1 * i), source=f"s{i}")

    archives = list(hist.parent.glob("weight_history.jsonl.*.gz"))
    assert len(archives) >= 1, "expected at least one rotated archive"
    # Live file exists but is shorter than the pre-rotation count.
    live_lines = [l for l in hist.read_text().splitlines() if l.strip()]
    assert len(live_lines) <= 3


# ---------------------------------------------------------------------------
# canonical_hash determinism
# ---------------------------------------------------------------------------

def test_canonical_hash_is_order_invariant():
    a = {"LOW": {"trend": 1.0, "mr": 0.9}, "_global": {"trend": 1.0, "mr": 0.9}}
    b = {"_global": {"mr": 0.9, "trend": 1.0}, "LOW": {"mr": 0.9, "trend": 1.0}}
    assert weights_history.canonical_hash(a) == weights_history.canonical_hash(b)
