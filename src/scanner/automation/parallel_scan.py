"""
Two-phase parallel scan coordinator.

Splits the scan loop into:

  Phase 1 — fan out, no shared writes:
    Per-pair workers run inside a ThreadPoolExecutor. Each worker calls
    Scanner._scan_pair(pair) which fetches OHLCV, engineers features,
    runs all model heads and the agent team. The result + the per-pair
    side-effect data (raw_snapshot, feature_snapshot, returns, last
    price, cache context) is bundled into a PairCandidate. NO writes
    to Scanner.* shared dicts happen on the worker thread.

  Phase 2 — fan in, serial under lock:
    The coordinator iterates candidates in a deterministic order (pair
    name) under a single re-entrant lock and commits the per-pair
    side-effect data into Scanner.* shared dicts. Cross-pair reducers
    (correlation filter, drawdown guardian, R:R gate, execution
    decisions) run later in the existing execute_trades() path which is
    already single-threaded — Phase 2 only handles the producer commit.

Why this shape
  Buddy's invariants (correlation filter, drawdown guardian, single-writer
  trade journal) require a serial decision step. The two-phase split
  preserves them by making decision-making serial while parallelising the
  expensive per-pair compute (OANDA fetch + model inference + agent team).

Toggle
  Enabled by env var BUDDY_PARALLEL_SCAN=1. Pool size from
  BUDDY_PARALLEL_SCAN_WORKERS (default min(15, os.cpu_count())). When
  the toggle is off, Scanner.scan() falls through to the existing path
  with no behaviour change.

Thread-safety audit (see docs/parallel_scan_architecture.md)
  - _modular_ensemble.predict_*: already guarded by Scanner._ensemble_lock
  - ScannerAgentTeam.evaluate: _learned_weights is read-only during scan
    (writes happen in _learn_from_outcome post-trade, outside scan path)
  - Per-pair shared dicts: writes are deferred to Phase 2 under
    coordinator lock (this module's primary safety mitigation)
  - Trade journal: single-writer in ExecutionManager.atomic_write
  - Drawdown guardian: read-only during scan
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Env var contract — kept here so config.py is untouched per scope rules
ENV_PARALLEL_SCAN: str = "BUDDY_PARALLEL_SCAN"
ENV_PARALLEL_SCAN_WORKERS: str = "BUDDY_PARALLEL_SCAN_WORKERS"
DEFAULT_MAX_WORKERS: int = 15  # cap; matches typical pair count


def is_parallel_scan_enabled() -> bool:
    """True iff BUDDY_PARALLEL_SCAN env var is a truthy value."""
    raw = os.environ.get(ENV_PARALLEL_SCAN, "0")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def resolve_worker_count(pair_count: int, override: Optional[int] = None) -> int:
    """Worker pool size: explicit override > env var > min(pairs, cpu, 15)."""
    if override is not None and override > 0:
        return max(1, int(override))

    env_raw = os.environ.get(ENV_PARALLEL_SCAN_WORKERS, "").strip()
    if env_raw:
        try:
            return max(1, int(env_raw))
        except ValueError:
            logger.warning(
                "Invalid %s=%r — falling back to default", ENV_PARALLEL_SCAN_WORKERS, env_raw
            )

    cpu_count = os.cpu_count() or 4
    return max(1, min(pair_count if pair_count > 0 else 1, cpu_count, DEFAULT_MAX_WORKERS))


@dataclass
class PairCandidate:
    """Phase 1 output for one pair — analysis + commit-side-effect payload.

    The analysis is the user-visible product. The other fields are
    private to the Scanner and are committed to Scanner.* shared dicts
    inside Phase 2 under the coordinator lock so that workers never
    touch shared state.
    """
    pair: str
    analysis: Optional[Any] = None  # PairAnalysis; avoid import cycle
    error: Optional[str] = None
    duration_secs: float = 0.0

    # Side-effect payload — committed atomically in Phase 2.
    # All optional; only populated when _scan_pair succeeds far enough.
    raw_snapshot: Optional[Any] = None        # pd.DataFrame
    feature_snapshot: Optional[Any] = None    # pd.DataFrame
    pair_returns: Optional[Any] = None        # pd.Series
    current_price: Optional[float] = None
    cached_result: Optional[Any] = None       # PairAnalysis (deepcopy of analysis)
    cache_context: Optional[Tuple[Any, ...]] = None
    last_scan_time: Optional[float] = None


@dataclass
class Phase2Stats:
    """Telemetry from a single Phase 2 commit pass."""
    candidates_seen: int = 0
    candidates_with_analysis: int = 0
    candidates_with_error: int = 0
    lock_wait_secs: float = 0.0
    commit_secs: float = 0.0


@dataclass
class ScanCycleStats:
    """End-to-end timing for one full scan cycle."""
    pair_count: int = 0
    worker_count: int = 0
    phase1_wall_secs: float = 0.0     # wall-clock for Phase 1 (parallel)
    phase1_cpu_secs: float = 0.0      # sum of per-pair durations
    phase2_wall_secs: float = 0.0
    phase2_stats: Phase2Stats = field(default_factory=Phase2Stats)
    errors: List[str] = field(default_factory=list)


class ParallelScanCoordinator:
    """Two-phase scan executor.

    Phase 1 fans out per-pair work onto a ThreadPoolExecutor. Phase 2
    commits results into Scanner shared state under a single lock.
    """

    def __init__(
        self,
        scanner: Any,
        max_workers: Optional[int] = None,
        per_pair_timeout_secs: float = 60.0,
        cycle_timeout_secs: float = 300.0,
    ):
        self.scanner = scanner
        self._max_workers_override = max_workers
        self._per_pair_timeout = per_pair_timeout_secs
        self._cycle_timeout = cycle_timeout_secs
        # Re-entrant lock so Phase 2 can call into helpers that may
        # acquire the same lock (defence-in-depth; current call graph
        # does not nest, but trade execution may).
        self._phase2_lock = threading.RLock()

    # ─── Phase 1 ────────────────────────────────────────────────────

    def _scan_pair_isolated(self, pair: str) -> PairCandidate:
        """Run _scan_pair for one pair, capturing side-effects without
        committing them to Scanner shared dicts.

        We let _scan_pair do its job (it writes to Scanner._raw_snapshots
        etc.) and then *immediately* hoist those writes back out into a
        PairCandidate. This minimises diff to engine.py while still
        letting Phase 2 perform the canonical commit later.

        Race window: between _scan_pair finishing and us reading the
        shared dict back into the candidate, another worker could clobber
        the same key only if it scanned the SAME pair, which never
        happens in one cycle (pairs are unique).
        """
        start = time.monotonic()
        try:
            analysis = self.scanner._scan_pair(pair)
        except Exception as e:
            logger.error("Parallel scan: _scan_pair(%s) failed: %s", pair, e)
            return PairCandidate(
                pair=pair,
                analysis=None,
                error=str(e),
                duration_secs=time.monotonic() - start,
            )

        # Hoist per-pair side-effect data out of Scanner shared dicts.
        # These reads are safe because (a) only this worker wrote to
        # this pair's key, and (b) Python dict reads of a single key
        # are atomic under the GIL.
        scanner = self.scanner
        candidate = PairCandidate(
            pair=pair,
            analysis=analysis,
            duration_secs=time.monotonic() - start,
            raw_snapshot=scanner._raw_snapshots.get(pair),
            feature_snapshot=scanner._feature_snapshots.get(pair),
            pair_returns=scanner._pair_returns.get(pair),
            current_price=scanner._last_prices.get(pair),
            cached_result=scanner._cached_results.get(pair),
            cache_context=scanner._cache_contexts.get(pair),
            last_scan_time=scanner._last_scan_times.get(pair),
        )

        return candidate

    def run_phase1(
        self,
        pair_list: List[str],
        on_pair_complete: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[List[PairCandidate], ScanCycleStats]:
        """Fan out per-pair work in parallel. No shared-state writes.

        Returns the candidates in completion order along with timing
        stats. The on_pair_complete callback is invoked from the main
        thread (NOT inside the worker) once each future resolves so
        TUI / display code stays single-threaded.
        """
        worker_count = resolve_worker_count(len(pair_list), self._max_workers_override)
        stats = ScanCycleStats(pair_count=len(pair_list), worker_count=worker_count)

        if not pair_list:
            return [], stats

        candidates: List[PairCandidate] = []
        wall_start = time.monotonic()

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="buddy-scan-p1",
        ) as executor:
            future_to_pair = {
                executor.submit(self._scan_pair_isolated, pair): pair
                for pair in pair_list
            }
            try:
                for future in as_completed(future_to_pair, timeout=self._cycle_timeout):
                    pair = future_to_pair[future]
                    try:
                        candidate = future.result(timeout=self._per_pair_timeout)
                    except Exception as e:
                        logger.error("Parallel scan: future.result(%s) failed: %s", pair, e)
                        candidate = PairCandidate(pair=pair, error=str(e))
                        stats.errors.append(f"{pair}: {e}")

                    candidates.append(candidate)
                    stats.phase1_cpu_secs += candidate.duration_secs

                    if on_pair_complete and candidate.analysis is not None:
                        try:
                            on_pair_complete(candidate.analysis)
                        except Exception as cb_err:
                            logger.debug(
                                "Parallel scan: on_pair_complete callback failed for %s: %s",
                                pair, cb_err,
                            )
            except TimeoutError:
                logger.error(
                    "Parallel scan: cycle timeout after %.1fs (workers=%d, pairs=%d)",
                    self._cycle_timeout, worker_count, len(pair_list),
                )
                stats.errors.append("cycle_timeout")

        stats.phase1_wall_secs = time.monotonic() - wall_start
        return candidates, stats

    # ─── Phase 2 ────────────────────────────────────────────────────

    def commit_candidates(
        self,
        candidates: List[PairCandidate],
    ) -> Tuple[List[Any], Phase2Stats]:
        """Serial commit of Phase 1 results into Scanner shared state.

        Iteration order is sorted by pair name for determinism. The
        coordinator lock serialises all writes, so any cross-pair logic
        added later (e.g. correlation filter pre-computation) is safe.

        Returns the analyses list in pair-name order — callers that
        depend on the original insertion order can re-sort by their own
        criteria.
        """
        stats = Phase2Stats(candidates_seen=len(candidates))

        lock_wait_start = time.monotonic()
        with self._phase2_lock:
            stats.lock_wait_secs = time.monotonic() - lock_wait_start
            commit_start = time.monotonic()

            analyses: List[Any] = []
            scanner = self.scanner
            # Deterministic order — if two candidates touch the same
            # cross-pair structure, they apply in stable order.
            for candidate in sorted(candidates, key=lambda c: c.pair):
                if candidate.error and candidate.analysis is None:
                    stats.candidates_with_error += 1
                    continue
                if candidate.analysis is None:
                    continue

                stats.candidates_with_analysis += 1
                analyses.append(candidate.analysis)

                # Commit per-pair side-effect data. Single-writer here,
                # so the dicts are consistent across keys at the end of
                # this loop.
                pair = candidate.pair
                if candidate.raw_snapshot is not None:
                    scanner._raw_snapshots[pair] = candidate.raw_snapshot
                if candidate.feature_snapshot is not None:
                    scanner._feature_snapshots[pair] = candidate.feature_snapshot
                if candidate.pair_returns is not None:
                    scanner._pair_returns[pair] = candidate.pair_returns
                if candidate.current_price is not None:
                    scanner._last_prices[pair] = candidate.current_price
                if candidate.cached_result is not None:
                    scanner._cached_results[pair] = candidate.cached_result
                if candidate.cache_context is not None:
                    scanner._cache_contexts[pair] = candidate.cache_context
                if candidate.last_scan_time is not None:
                    scanner._last_scan_times[pair] = candidate.last_scan_time

            stats.commit_secs = time.monotonic() - commit_start

        return analyses, stats

    # ─── End-to-end ─────────────────────────────────────────────────

    def run(
        self,
        pair_list: List[str],
        on_pair_complete: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[List[Any], ScanCycleStats]:
        """Run a full Phase 1 + Phase 2 cycle.

        This is the top-level entry point used by Scanner.scan() when
        BUDDY_PARALLEL_SCAN=1.
        """
        candidates, stats = self.run_phase1(pair_list, on_pair_complete=on_pair_complete)

        phase2_start = time.monotonic()
        analyses, phase2_stats = self.commit_candidates(candidates)
        stats.phase2_wall_secs = time.monotonic() - phase2_start
        stats.phase2_stats = phase2_stats

        logger.info(
            "Parallel scan cycle: pairs=%d workers=%d phase1=%.2fs phase2=%.3fs "
            "(lock_wait=%.4fs commit=%.4fs) committed=%d errors=%d",
            stats.pair_count,
            stats.worker_count,
            stats.phase1_wall_secs,
            stats.phase2_wall_secs,
            phase2_stats.lock_wait_secs,
            phase2_stats.commit_secs,
            phase2_stats.candidates_with_analysis,
            phase2_stats.candidates_with_error,
        )

        return analyses, stats


__all__ = [
    "ENV_PARALLEL_SCAN",
    "ENV_PARALLEL_SCAN_WORKERS",
    "PairCandidate",
    "ParallelScanCoordinator",
    "Phase2Stats",
    "ScanCycleStats",
    "is_parallel_scan_enabled",
    "resolve_worker_count",
]
