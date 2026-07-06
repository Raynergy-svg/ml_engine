"""
Phase 1 producer correctness tests.

Verifies that the per-pair candidate produced by the parallel scan
coordinator is identical (analysis fields + side-effect payload) whether
the scan was driven sequentially or via the coordinator. Uses a fake
Scanner stub so we don't need real OANDA / models.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.scanner.automation.parallel_scan import (
    PairCandidate,
    ParallelScanCoordinator,
    resolve_worker_count,
)


# ── Fake Scanner ────────────────────────────────────────────────────


class _FakeAnalysis:
    """Stand-in for PairAnalysis — we only need attributes the coordinator
    reads (none, in fact) plus an identity for assertions."""

    def __init__(self, pair: str, confidence: float = 0.5):
        self.pair = pair
        self.confidence = confidence
        self.direction = "HOLD"
        self.is_tradeable = False


def _make_fake_scanner(
    pairs: List[str],
    *,
    sleep_secs: float = 0.0,
    seed: int = 0,
) -> MagicMock:
    """Build a Scanner mock with the shared dicts the coordinator reads.

    _scan_pair populates the shared dicts as a side effect (matching real
    Scanner._scan_pair behaviour) so the coordinator can hoist them into
    the PairCandidate.
    """
    scanner = MagicMock()
    scanner._raw_snapshots = {}
    scanner._feature_snapshots = {}
    scanner._pair_returns = {}
    scanner._last_prices = {}
    scanner._cached_results = {}
    scanner._cache_contexts = {}
    scanner._last_scan_times = {}

    # Per-pair compute counter — used to verify we don't double-scan.
    scanner._scan_pair_calls: Dict[str, int] = {}
    scanner._scan_pair_lock = threading.Lock()

    def _scan_pair(pair: str) -> _FakeAnalysis:
        if sleep_secs > 0:
            time.sleep(sleep_secs)

        # Deterministic per-pair side-effect payload.
        scanner._raw_snapshots[pair] = f"raw-{pair}-{seed}"
        scanner._feature_snapshots[pair] = f"feat-{pair}-{seed}"
        scanner._pair_returns[pair] = f"returns-{pair}-{seed}"
        scanner._last_prices[pair] = 1.0 + (hash((pair, seed)) % 1000) / 10_000
        scanner._cache_contexts[pair] = (pair, seed)
        scanner._last_scan_times[pair] = time.time()
        analysis = _FakeAnalysis(pair, confidence=0.5)
        scanner._cached_results[pair] = analysis

        with scanner._scan_pair_lock:
            scanner._scan_pair_calls[pair] = scanner._scan_pair_calls.get(pair, 0) + 1

        return analysis

    scanner._scan_pair.side_effect = _scan_pair
    return scanner


# ── Tests ───────────────────────────────────────────────────────────


PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]


def test_phase1_returns_one_candidate_per_pair():
    scanner = _make_fake_scanner(PAIRS)
    coordinator = ParallelScanCoordinator(scanner, max_workers=3)

    candidates, stats = coordinator.run_phase1(PAIRS)

    assert len(candidates) == len(PAIRS)
    assert {c.pair for c in candidates} == set(PAIRS)
    assert stats.pair_count == len(PAIRS)
    assert stats.worker_count == 3
    assert stats.errors == []


def test_phase1_each_pair_scanned_exactly_once():
    scanner = _make_fake_scanner(PAIRS)
    coordinator = ParallelScanCoordinator(scanner, max_workers=4)

    coordinator.run_phase1(PAIRS)

    assert scanner._scan_pair_calls == {p: 1 for p in PAIRS}


def test_phase1_candidate_payload_matches_scanner_state():
    """The hoisted side-effect data must equal what _scan_pair wrote."""
    scanner = _make_fake_scanner(PAIRS, seed=7)
    coordinator = ParallelScanCoordinator(scanner, max_workers=2)

    candidates, _ = coordinator.run_phase1(PAIRS)

    for c in candidates:
        # raw_snapshot, feature_snapshot, pair_returns, current_price,
        # cached_result, cache_context, last_scan_time should all be
        # populated and match what the fake scanner stored.
        assert c.raw_snapshot == f"raw-{c.pair}-7"
        assert c.feature_snapshot == f"feat-{c.pair}-7"
        assert c.pair_returns == f"returns-{c.pair}-7"
        assert c.current_price is not None
        assert c.cache_context == (c.pair, 7)
        assert c.last_scan_time is not None


def test_phase1_parallel_vs_sequential_produces_equal_candidates():
    """Same _scan_pair logic + same pair list → equal candidates regardless
    of worker count."""
    scanner_seq = _make_fake_scanner(PAIRS, seed=42)
    scanner_par = _make_fake_scanner(PAIRS, seed=42)

    coordinator_seq = ParallelScanCoordinator(scanner_seq, max_workers=1)
    coordinator_par = ParallelScanCoordinator(scanner_par, max_workers=5)

    cands_seq, _ = coordinator_seq.run_phase1(PAIRS)
    cands_par, _ = coordinator_par.run_phase1(PAIRS)

    seq_by_pair = {c.pair: c for c in cands_seq}
    par_by_pair = {c.pair: c for c in cands_par}

    assert set(seq_by_pair) == set(par_by_pair)
    for pair, cs in seq_by_pair.items():
        cp = par_by_pair[pair]
        # Side-effect payload must be identical (deterministic on seed).
        assert cs.raw_snapshot == cp.raw_snapshot
        assert cs.feature_snapshot == cp.feature_snapshot
        assert cs.pair_returns == cp.pair_returns
        assert cs.current_price == cp.current_price
        assert cs.cache_context == cp.cache_context


def test_phase1_handles_per_pair_exception():
    scanner = _make_fake_scanner(PAIRS)

    def _scan_pair_with_failure(pair: str):
        if pair == "GBP_USD":
            raise RuntimeError("synthetic failure")
        # Mirror the fake's normal logic for other pairs.
        scanner._raw_snapshots[pair] = f"raw-{pair}"
        scanner._feature_snapshots[pair] = f"feat-{pair}"
        scanner._pair_returns[pair] = f"ret-{pair}"
        scanner._last_prices[pair] = 1.0
        scanner._cache_contexts[pair] = (pair, 0)
        scanner._last_scan_times[pair] = time.time()
        a = _FakeAnalysis(pair)
        scanner._cached_results[pair] = a
        return a

    scanner._scan_pair.side_effect = _scan_pair_with_failure

    coordinator = ParallelScanCoordinator(scanner, max_workers=3)
    candidates, stats = coordinator.run_phase1(PAIRS)

    assert len(candidates) == len(PAIRS)
    failed = [c for c in candidates if c.error is not None]
    assert len(failed) == 1
    assert failed[0].pair == "GBP_USD"
    assert failed[0].analysis is None


def test_phase1_callback_invoked_only_for_successful_analyses():
    scanner = _make_fake_scanner(PAIRS)
    coordinator = ParallelScanCoordinator(scanner, max_workers=2)

    seen: List[Any] = []
    coordinator.run_phase1(PAIRS, on_pair_complete=lambda a: seen.append(a))

    assert len(seen) == len(PAIRS)
    assert all(isinstance(a, _FakeAnalysis) for a in seen)


def test_phase1_wall_clock_under_sequential_estimate():
    """5 pairs × 0.05s sleep with 5 workers should beat sequential floor."""
    sleep = 0.05
    pairs = list(PAIRS)
    scanner = _make_fake_scanner(pairs, sleep_secs=sleep)
    coordinator = ParallelScanCoordinator(scanner, max_workers=len(pairs))

    candidates, stats = coordinator.run_phase1(pairs)

    assert len(candidates) == len(pairs)
    sequential_floor = sleep * len(pairs)
    # Allow generous slack for CI noise — anything below 70% of the
    # sequential floor proves Phase 1 is genuinely concurrent.
    assert stats.phase1_wall_secs < sequential_floor * 0.7, (
        f"phase1 wall {stats.phase1_wall_secs:.3f}s not < 0.7 * {sequential_floor:.3f}s"
    )


def test_resolve_worker_count_caps_at_default_max(monkeypatch):
    monkeypatch.delenv("BUDDY_PARALLEL_SCAN_WORKERS", raising=False)
    # 100 pairs but we cap at min(15, cpu_count()).
    cpu = __import__("os").cpu_count() or 4
    expected = max(1, min(100, cpu, 15))
    assert resolve_worker_count(100) == expected


def test_resolve_worker_count_env_override(monkeypatch):
    monkeypatch.setenv("BUDDY_PARALLEL_SCAN_WORKERS", "3")
    assert resolve_worker_count(20) == 3


def test_resolve_worker_count_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("BUDDY_PARALLEL_SCAN_WORKERS", "9")
    assert resolve_worker_count(20, override=2) == 2


def test_resolve_worker_count_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("BUDDY_PARALLEL_SCAN_WORKERS", "not-a-number")
    assert resolve_worker_count(5) >= 1
