# Parallel Scan Architecture

**Status:** Implemented behind feature flag (default OFF). Pending production benchmark + smoke validation.
**Toggle:** `BUDDY_PARALLEL_SCAN=1`
**Scope:** scan-loop orchestration only. No strategy / gate / agent / sizer / execution changes.

---

## What it does

Today's scan loop iterates pairs sequentially (well, with a thread pool but feeding into shared writes). For 15 pairs at ~250 ms/pair this lands in the 3-7 s/cycle range — most of which is OANDA fetch + feature engineering + model inference, all parallelisable.

The parallel scan splits the cycle into two phases:

```
Phase 1 — fan out, no shared writes (parallel across pairs)
  ─ OANDA OHLCV fetch
  ─ Feature engineering
  ─ All model heads (direction, confidence, momentum, risk, regime, meta-labeler)
  ─ 15-agent team for that pair
  → produces a PairCandidate (analysis + per-pair side-effect payload)

Phase 2 — fan in, serial under lock (single thread)
  ─ Iterates candidates in deterministic pair-name order
  ─ Commits per-pair side-effect data into Scanner shared dicts
  ─ Cross-pair reducers (correlation filter, drawdown guardian, R:R gate,
    execution decisions) run in the existing execute_trades() path, which
    is already single-threaded
```

Phase 1 is the bulk of the cycle wall-clock; Phase 2 is fast logic under a single re-entrant lock.

## Invariants preserved

- **Correlation filter** — runs in the existing serial decision path; sees all candidates from Phase 2 commit before approving any entry.
- **Drawdown guardian** — single in-memory source of truth, read in Phase 2 / execute path only. No concurrent writes.
- **Trade journal** — single-writer in `ExecutionManager.atomic_write`. Untouched.
- **Agent weights** — `_team.py` `_learned_weights` writes happen in `_learn_from_outcome` post-trade, NOT during scan. Reads during scan are dict lookups, atomic under the GIL.
- **Model objects** — `_modular_ensemble` already guards predict paths with `_ensemble_lock`; LightGBM / TF inference release the GIL during native code, so threading parallelises them.

If any of those invariants flag concerns, fall back via the toggle.

## Enabling

```sh
export BUDDY_PARALLEL_SCAN=1
./buddy --demo   # or --live
```

Worker pool size:
- Default: `min(pair_count, os.cpu_count(), 15)`
- Override: `export BUDDY_PARALLEL_SCAN_WORKERS=8`
- Don't oversize — too many workers causes lock contention and OANDA-side throttling.

## Rolling back

```sh
unset BUDDY_PARALLEL_SCAN
./buddy --demo   # back to sequential, no behaviour change
```

The `Scanner.scan()` path falls through to the original sequential `ThreadPoolExecutor` loop when the env var is absent OR the parallel coordinator raises. **No code revert needed.** If you see anomalies, just unset and restart.

## Telemetry

Each parallel cycle logs:

```
Parallel scan cycle: pairs=15 workers=8 phase1=0.42s phase2=0.018s
  (lock_wait=0.001s commit=0.014s) committed=15 errors=0
```

The cycle stats object is also stored at `Scanner._last_parallel_scan_stats` for in-process inspection.

## Known limitations

1. **GIL ceiling around ~30 pairs.** Pure-Python work in agents and feature pipelines is GIL-bound. Past ~30 pairs, lock contention + GIL serialisation will diminish returns. At that point, multi-process sharding (`option (b)` from the brainstorm) becomes worth the orchestration cost.
2. **No process isolation.** A bug that crashes one pair's scan crashes all of them. Mitigated in code by per-future try/except returning a `PairCandidate` with `error` set, but a pathological native-code crash takes the process down.
3. **OANDA rate limits.** Fan-out to 15 concurrent fetches respects per-second limits, but if OANDA tightens, reduce `BUDDY_PARALLEL_SCAN_WORKERS`.
4. **Smoke test + benchmark not yet captured in repo.** The implementation passes 20 unit tests (Phase 1 producer correctness, Phase 2 reducer determinism, thread-safety stress). Pending operator validation against a live `--demo` cycle and a wall-clock A/B benchmark.

## Pending validation

Before flipping `BUDDY_PARALLEL_SCAN=1` in a live trading environment:

- [ ] Run `./buddy --demo` for one full session with the flag on. Confirm no race-condition errors in logs, no malformed JSON in `trained_data/trade_journal_rl.json` or state files.
- [ ] Capture wall-clock per cycle for sequential and parallel; expect ≥ 3× speedup on Phase 1.
- [ ] Watch first few live cycles for any agent-state inconsistencies (the audit says reads are safe, but production is the truth check).

## Files

- `src/scanner/automation/parallel_scan.py` — coordinator + dataclasses + env-var helpers
- `src/scanner/engine.py` — `Scanner.scan()` wired to coordinator with fallback on exception
- `tests/test_parallel_scan_phase1.py` — Phase 1 producer correctness (11 tests)
- `tests/test_parallel_scan_phase2.py` — Phase 2 reducer determinism (6 tests)
- `tests/test_parallel_scan_thread_safety.py` — concurrent stress (3 tests)
