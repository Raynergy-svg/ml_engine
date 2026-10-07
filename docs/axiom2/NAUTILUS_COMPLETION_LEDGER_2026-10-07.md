# SDD ledger — plan: docs/superpowers/plans/2026-10-06-axiom2-nautilus-runtime.md

Base: 9e996c8bd3e0a37fafed2793be1d1165d91209b3. Isolated completion branch; remote runtime branch unchanged.

Ruling: Resume inherited Tasks 1–4 implementation instead of rewriting completed mechanics — preserve verified checkpoint and upstream pin — inherited defects require explicit regression tests.
Ruling: Keep stable handoff artifacts in docs/axiom2 instead of deleting the ledger — user explicitly requires stable handoff — additional tracked documentation.
Pre-flight: Task 2 raw observation/state/outbox consumed by Task 3; version supersession and freshness need verification.
Pre-flight: Task 4 raw events consumed by comparison; current comparison lives only in a test and needs an actual projection consumer.
Cheap baseline: smoke contract + candidate monitor, 16 passed in 0.14s.
Dependency root cause: runtime order comparison imports Axiom signed lifecycle -> pydantic and cryptography, absent from upstream make sync environment. Exact pins added only after upstream build/cache verification.
Artifact: 11454954090, run 37553124116, GitHub digest sha256:fad40bc654f5965a8bef2317036f1b9a20504138f30e2a516afcfa90414ec721. Download through storage blocked 403; connector transfer exceeds 32 MiB. Use bounded verified cloud cache; cache miss fails, no rebuild.

Task 2/3 completion: Added persistent replay clock, shorter research READY deadline, supersession invalidation, stale evidence denial, duplicate-result conflict rejection, immutable nested facts and durable recovery snapshot. Explicit time advancement supplies replay freshness; this is not a wall-clock live service.
Task 4 completion: LifecycleProjection consumes retained native reports and derived snapshots, checks the unchanged Axiom transition table, exact USD integer-share/cents semantics and fill quantity bounds, and calls the unchanged pure reconciliation helper. Conflicts/rejections/parity gaps remain HALTED. Durable seed and payload/snapshot integrity checks, locked rebuild, event-ID deduplication and transaction rollback verified with actual upstream extension.
Native RED: test constructors used str instead of TraderId/StrategyId; after correction, four lifecycle tests failed (fixed-point raw quantity expectation, missing seed binding, stale cross-connection state, event identity duplication). Seed/integrity/concurrency/dedup fixes then passed. Later overfill and research-deadline extension regressions failed and passed after targeted fixes.
Native focused gate at 0c0dae72: 56 passed in 0.52s. Actual no-credentials/socket-guard public API smoke passed. Original extension SHA256 09832798c20e663d7917a72d427308960925f34594e0345f69351fc2b08df6fb; upstream git diff empty.
Ruling: Original upload omitted hidden manifest and venv (ZIP inventory has exactly one extension); reconstruct source/toolchain manifest and require exact supplied manifest digest, original archive digest and original run/head metadata before unchanged extension reuse — no compile or altered upstream source — package distribution metadata remains unavailable in source-plus-extension tests, and no Mac equivalence is claimed.
Ruling: Keep source inventory gate unchanged as instructed — all existing research/portfolio/execution hashes still match and native read declaration probe passes — complete checkout isolation stays blocked until an explicitly reviewed Nautilus exclusion inventory is authorized. Do not skip or xfail its failure.
Pass 1 self-review: requirements, public API and provenance complete for the replay-only scope; imported actual pinned extension, smoke verified Clock/MessageBus/BacktestEngine lifecycle, no timer callback dispatch claim, native order constructors corrected.
Pass 2 self-review: durable state/outbox rollback, SIGKILL/restart, version supersession, result identity conflicts, READY deadlines/revocation, raw/derived rollback, seed drift, corrupt raw bytes, separate-connection recovery, unsupported currencies/precision, and rejected/late-fill terminality exercised. No execution method or signed authority write added.
Full local Axiom/evidence gate: executing, record exact outcome when finished.
Fresh final review and final cached cloud verification: pending. Reviewed completion cannot be claimed while these are pending or source-inventory gate is red.
