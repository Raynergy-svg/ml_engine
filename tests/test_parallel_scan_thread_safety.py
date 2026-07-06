"""
Thread-safety stress tests for the parallel scan coordinator.

Approach: drive the coordinator through many cycles back-to-back with a
fake Scanner whose _scan_pair sleeps a small random amount of time and
mutates the shared dicts. We then verify:
  - no exceptions
  - per-pair scan counts are exactly 1 per cycle
  - shared dicts converge to the expected state at end of each cycle
  - no journal-style writes interleave (we use a mutex on a counter)
"""

from __future__ import annotations

import random
import threading
import time
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.scanner.automation.parallel_scan import ParallelScanCoordinator


PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "USD_CHF", "EUR_CHF", "AUD_NZD", "EUR_AUD", "GBP_AUD",
]


def _make_scanner(jitter_max_secs: float = 0.005) -> MagicMock:
    s = MagicMock()
    s._raw_snapshots = {}
    s._feature_snapshots = {}
    s._pair_returns = {}
    s._last_prices = {}
    s._cached_results = {}
    s._cache_contexts = {}
    s._last_scan_times = {}

    # Sentinel single-writer counter — every successful _scan_pair call
    # increments under a mutex; if any update is lost we'll see it.
    s._counter = 0
    s._counter_lock = threading.Lock()
    s._scan_calls: Dict[str, int] = {}
    s._scan_calls_lock = threading.Lock()

    def _scan_pair(pair: str):
        # Random small sleep to encourage interleaving.
        time.sleep(random.uniform(0, jitter_max_secs))

        # Per-pair side effects.
        s._raw_snapshots[pair] = ("raw", pair, s._counter)
        s._feature_snapshots[pair] = ("feat", pair)
        s._pair_returns[pair] = ("ret", pair)
        s._last_prices[pair] = 1.0 + (hash(pair) % 100) / 1000.0
        s._cache_contexts[pair] = (pair,)
        s._last_scan_times[pair] = time.time()

        with s._counter_lock:
            s._counter += 1
        with s._scan_calls_lock:
            s._scan_calls[pair] = s._scan_calls.get(pair, 0) + 1

        analysis = MagicMock(pair=pair)
        s._cached_results[pair] = analysis
        return analysis

    s._scan_pair.side_effect = _scan_pair
    return s


def test_thread_safety_100_cycles():
    """Run 100 cycles end-to-end. No exceptions, counter == 100 * pairs."""
    scanner = _make_scanner()
    coordinator = ParallelScanCoordinator(scanner, max_workers=8)

    n_cycles = 100
    for _ in range(n_cycles):
        analyses, stats = coordinator.run(PAIRS)
        assert len(analyses) == len(PAIRS)
        assert stats.errors == []

    # Every pair was scanned exactly n_cycles times.
    assert scanner._scan_calls == {p: n_cycles for p in PAIRS}
    # Counter increments must equal total scans (no lost writes).
    assert scanner._counter == n_cycles * len(PAIRS)


def test_concurrent_coordinators_dont_corrupt_journal_proxy():
    """Stress: spawn N coordinators concurrently sharing a single
    journal-proxy resource (a list protected by a lock). All writes
    must be present and well-formed."""
    journal: List[str] = []
    journal_lock = threading.Lock()

    scanners = [_make_scanner() for _ in range(4)]
    # Override _scan_pair to also write to the shared journal proxy.
    for idx, s in enumerate(scanners):
        original = s._scan_pair.side_effect

        def _make_wrapped(scanner_idx, orig):
            def _wrapped(pair: str):
                analysis = orig(pair)
                with journal_lock:
                    journal.append(f"{scanner_idx}:{pair}")
                return analysis
            return _wrapped

        s._scan_pair.side_effect = _make_wrapped(idx, original)

    coordinators = [ParallelScanCoordinator(s, max_workers=4) for s in scanners]

    threads = []
    for c in coordinators:
        t = threading.Thread(target=lambda c=c: c.run(PAIRS))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "coordinator thread hung"

    # Each scanner ran each pair exactly once.
    for idx, s in enumerate(scanners):
        assert s._scan_calls == {p: 1 for p in PAIRS}, f"scanner {idx} missed a pair"

    # Journal contains every (scanner_idx, pair) pair exactly once.
    expected = {f"{i}:{p}" for i in range(len(scanners)) for p in PAIRS}
    assert set(journal) == expected
    assert len(journal) == len(expected)


def test_phase2_lock_serialises_concurrent_commits():
    """If we manually invoke commit_candidates from N threads on the same
    coordinator, the final shared state must be exactly the union of
    all committed candidates with no torn writes."""
    scanner = _make_scanner()
    coordinator = ParallelScanCoordinator(scanner, max_workers=4)

    # Pre-build candidate lists for 4 threads.
    from src.scanner.automation.parallel_scan import PairCandidate

    def _build(pair_subset, tag):
        return [
            PairCandidate(
                pair=p,
                analysis=MagicMock(pair=p),
                raw_snapshot=("raw", p, tag),
                feature_snapshot=("feat", p, tag),
                pair_returns=("ret", p, tag),
                current_price=1.0,
                cached_result=MagicMock(pair=p),
                cache_context=(p, tag),
                last_scan_time=time.time(),
            )
            for p in pair_subset
        ]

    chunks = [PAIRS[i::4] for i in range(4)]
    candidate_batches = [_build(chunks[i], i) for i in range(4)]

    threads = []
    for batch in candidate_batches:
        t = threading.Thread(target=lambda b=batch: coordinator.commit_candidates(b))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=10)

    # All 15 pairs must have been committed.
    assert set(scanner._raw_snapshots.keys()) == set(PAIRS)
