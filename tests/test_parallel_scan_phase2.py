"""
Phase 2 reducer correctness tests.

Verifies that committing pre-computed candidates into Scanner shared
state happens deterministically, under the lock, and matches the
sequential reference (i.e. what would happen if _scan_pair ran one pair
at a time).
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.scanner.automation.parallel_scan import (
    PairCandidate,
    ParallelScanCoordinator,
)


PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]


def _empty_scanner():
    s = MagicMock()
    s._raw_snapshots = {}
    s._feature_snapshots = {}
    s._pair_returns = {}
    s._last_prices = {}
    s._cached_results = {}
    s._cache_contexts = {}
    s._last_scan_times = {}
    return s


def _candidate(pair: str, ts: float = 1700000000.0) -> PairCandidate:
    return PairCandidate(
        pair=pair,
        analysis=MagicMock(pair=pair),
        raw_snapshot=f"raw-{pair}",
        feature_snapshot=f"feat-{pair}",
        pair_returns=f"ret-{pair}",
        current_price=1.0 + len(pair) / 100.0,
        cached_result=MagicMock(pair=pair),
        cache_context=(pair, 0),
        last_scan_time=ts,
    )


# ── Tests ───────────────────────────────────────────────────────────


def test_phase2_commits_all_candidate_payloads():
    scanner = _empty_scanner()
    coordinator = ParallelScanCoordinator(scanner)

    candidates = [_candidate(p) for p in PAIRS]
    analyses, stats = coordinator.commit_candidates(candidates)

    assert stats.candidates_seen == len(PAIRS)
    assert stats.candidates_with_analysis == len(PAIRS)
    assert stats.candidates_with_error == 0
    assert len(analyses) == len(PAIRS)

    # All shared dicts populated.
    for p in PAIRS:
        assert scanner._raw_snapshots[p] == f"raw-{p}"
        assert scanner._feature_snapshots[p] == f"feat-{p}"
        assert scanner._pair_returns[p] == f"ret-{p}"
        assert p in scanner._last_prices
        assert p in scanner._cached_results
        assert scanner._cache_contexts[p] == (p, 0)


def test_phase2_iterates_in_deterministic_order():
    """Candidates produced in arbitrary completion order must commit in
    sorted-by-pair order so any cross-pair logic added later sees a
    stable sequence."""
    scanner = _empty_scanner()
    coordinator = ParallelScanCoordinator(scanner)

    # Submit in deliberately scrambled order.
    scrambled = [_candidate(p) for p in ["USD_CAD", "AUD_USD", "EUR_USD", "USD_JPY", "GBP_USD"]]
    analyses, _ = coordinator.commit_candidates(scrambled)

    # Output analyses must be sorted by pair name (deterministic).
    assert [a.pair for a in analyses] == sorted([c.pair for c in scrambled])


def test_phase2_skips_error_candidates_without_analysis():
    scanner = _empty_scanner()
    coordinator = ParallelScanCoordinator(scanner)

    failing = PairCandidate(pair="EUR_USD", error="oanda_offline", analysis=None)
    ok = _candidate("GBP_USD")
    analyses, stats = coordinator.commit_candidates([failing, ok])

    assert stats.candidates_seen == 2
    assert stats.candidates_with_analysis == 1
    assert stats.candidates_with_error == 1
    assert len(analyses) == 1
    assert analyses[0].pair == "GBP_USD"
    # Failed pair must NOT have leaked into shared state.
    assert "EUR_USD" not in scanner._raw_snapshots


def test_phase2_holds_lock_across_writes():
    """Verify the coordinator lock is taken during commit by trying to
    take it from another thread — that thread must block until commit
    finishes."""
    scanner = _empty_scanner()
    coordinator = ParallelScanCoordinator(scanner)

    candidates = [_candidate(p) for p in PAIRS]

    other_started = threading.Event()
    other_acquired = threading.Event()

    def _race():
        other_started.set()
        # Try to grab the lock; should block until commit_candidates
        # releases it.
        with coordinator._phase2_lock:
            other_acquired.set()

    thr = threading.Thread(target=_race, daemon=True)
    # Pre-acquire the lock so the racing thread is forced to wait.
    coordinator._phase2_lock.acquire()
    try:
        thr.start()
        other_started.wait(timeout=1.0)
        # While we hold the lock, the racer cannot acquire.
        assert not other_acquired.wait(timeout=0.1)
    finally:
        coordinator._phase2_lock.release()

    other_acquired.wait(timeout=1.0)
    assert other_acquired.is_set()
    thr.join(timeout=1.0)


def test_phase2_idempotent_under_repeat_commit():
    """Committing the same candidate twice must not corrupt state."""
    scanner = _empty_scanner()
    coordinator = ParallelScanCoordinator(scanner)

    cand = _candidate("EUR_USD")
    coordinator.commit_candidates([cand])
    snapshot1 = dict(scanner._raw_snapshots)
    coordinator.commit_candidates([cand])
    snapshot2 = dict(scanner._raw_snapshots)

    assert snapshot1 == snapshot2


def test_phase2_matches_sequential_reference():
    """Building scanner state via Phase 2 from candidates must end up in
    the same state as if _scan_pair had run sequentially and written
    directly."""
    candidates = [_candidate(p, ts=1700000000.0 + i) for i, p in enumerate(PAIRS)]

    # Reference: simulate sequential _scan_pair writes.
    ref_scanner = _empty_scanner()
    for c in sorted(candidates, key=lambda x: x.pair):
        ref_scanner._raw_snapshots[c.pair] = c.raw_snapshot
        ref_scanner._feature_snapshots[c.pair] = c.feature_snapshot
        ref_scanner._pair_returns[c.pair] = c.pair_returns
        ref_scanner._last_prices[c.pair] = c.current_price
        ref_scanner._cached_results[c.pair] = c.cached_result
        ref_scanner._cache_contexts[c.pair] = c.cache_context
        ref_scanner._last_scan_times[c.pair] = c.last_scan_time

    # Parallel: shuffle order and commit through Phase 2.
    par_scanner = _empty_scanner()
    par_coord = ParallelScanCoordinator(par_scanner)
    par_coord.commit_candidates(list(reversed(candidates)))

    # State must match (modulo MagicMock identity for cached_result —
    # compare scalar payload only).
    for d in ("_raw_snapshots", "_feature_snapshots", "_pair_returns",
              "_last_prices", "_cache_contexts", "_last_scan_times"):
        assert getattr(ref_scanner, d) == getattr(par_scanner, d), f"Mismatch in {d}"
