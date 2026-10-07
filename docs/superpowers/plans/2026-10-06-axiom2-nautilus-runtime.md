# Axiom 2.0 Nautilus Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a pinned, execution-free NautilusTrader runtime bridge for deterministic candidate monitoring and offline order-event lifecycle comparison while preserving Axiom’s existing execution authority.

**Architecture:** NautilusTrader remains an unmodified, pinned upstream dependency and supplies only event/runtime mechanics. Axiom owns observation admission, candidate policy, provenance/freshness, durable material events, bounded research wakeups, revalidation, and all authority checks. The first implementation is deterministic replay-only; no broker adapter, credentials, order submission, or live-feed claim is added.

**Tech Stack:** Python 3.12+, upstream NautilusTrader commit `4f021bafc2e99c5490cee204b0fc2bd2c83baab4`, isolated virtual environment, SQLite-backed durable monitor journal, existing Axiom lifecycle/reconciliation contracts, pytest.

**Spec:** User-approved Nautilus integration request dated 2026-10-06; upstream source/build/license evidence recorded in `docs/axiom2/NAUTILUS_RUNTIME_REVIEW_2026-10-06.md`.

## Global Constraints

- Base exactly `64bf4d3780bc193bf122f830b1df04027ba8fe47`; LEAN branch and all LEAN changes remain untouched.
- Nautilus source revision exactly `4f021bafc2e99c5490cee204b0fc2bd2c83baab4`; do not fork, copy, or rewrite its engine/event bus/clock/order state machine.
- Runtime harness starts with no broker credentials, execution clients, or external network access.
- READY is revocable and never order authorization; existing Axiom capital, approval, risk, execution, and reconciliation gates remain unchanged.
- Replay/synthetic inputs prove runtime behavior only; live market connectivity remains an explicit gap.
- Durable state and wakeup delivery must recover together; stable IDs and persistent idempotency are required. No exactly-once claim.
- Raw upstream order observations remain separate from normalized/inferred Axiom state.

## Review Focus

- A crash after material-event persistence but before wakeup delivery must recover one durable pending wakeup.
- A stale, expired, invalidated, or superseded candidate version must not become READY after restart or late research response.
- Duplicate and conflicting observations must be handled durably, not by process memory.
- Nautilus lifecycle start/stop must not open network clients or allow execution surfaces.
- Order-event terminality, late fills, cancel rejection, and conflicting duplicates must fail closed or remain explicitly unresolved.

---

### Task 1: Pinned upstream provenance and actual public-interface smoke harness

**Files:**
- Create: `docs/axiom2/NAUTILUS_RUNTIME_REVIEW_2026-10-06.md`
- Create: `config/axiom2/nautilus_runtime.json`
- Create: `scripts/axiom2_nautilus_smoke.py`
- Create: `tests/axiom2/test_nautilus_smoke_contract.py`
- Create: `.github/workflows/axiom2-nautilus-runtime.yml`

**Interfaces:**
- Smoke harness imports the installed upstream package, exercises event delivery, deterministic clock/timer, and lifecycle start/stop, and exits nonzero if the supported public API is unavailable.
- The harness records package version, source revision, build/artifact identity, enabled features, license, and network/credential denial assertions.
- Order-event capability probing records which public order-event constructors and callbacks are callable; it must not interpret config-only objects as engine integration.

- [x] Write the failing smoke-contract test and run it in the isolated CI environment.
- [x] Add the exact upstream pin, build requirements, LGPL-3.0-only notice, and explicit known API findings.
- [x] Implement the minimal smoke harness using only public Nautilus interfaces.
- [x] Run the real source build/import/smoke workflow with no broker credentials and a network-denial guard.
- [x] Commit the provenance, harness, and workflow as one small increment.

### Task 2: Durable candidate-monitor journal and deterministic policy

**Files:**
- Create: `src/axiom2/nautilus_runtime/__init__.py`
- Create: `src/axiom2/nautilus_runtime/contracts.py`
- Create: `src/axiom2/nautilus_runtime/journal.py`
- Create: `src/axiom2/nautilus_runtime/policy.py`
- Create: `tests/axiom2/nautilus_runtime/test_candidate_monitor.py`

**Interfaces:**
- `CandidateObservation`: immutable admitted observation with candidate/version, evidence digest, observed-at, freshness deadline, confirmation/invalidation facts, and raw-observation digest.
- `CandidateState`: WATCHING, WAITING, TRIGGERED, REVALIDATING, READY, INVALIDATED, EXPIRED.
- `CandidateMonitor.observe(observation) -> MaterialEvent | None`.
- `CandidateMonitor.revalidate(candidate_version, result) -> CandidateState`.
- `CandidateJournal.replay() -> DurableSnapshot` and atomic append/ack methods.

- [x] Write RED tests for valid transitions, invalidation/expiration from every nonterminal state, stale observations, duplicate/conflicting IDs, and READY revocation.
- [x] Implement the journal with an atomic SQLite transaction containing raw observation, state event, material event, and pending wakeup rows.
- [x] Implement deterministic policy with candidate-version binding, monotonic timestamps, evidence freshness, and no order/capital fields.
- [x] Run focused tests, then the full Axiom-scoped test selection.
- [x] Commit the monitor increment.

### Task 3: Nautilus runtime adapter and bounded ResearchWakeup path

**Files:**
- Modify: `src/axiom2/nautilus_runtime/__init__.py`
- Create: `src/axiom2/nautilus_runtime/runtime.py`
- Create: `src/axiom2/nautilus_runtime/wakeup.py`
- Create: `tests/axiom2/nautilus_runtime/test_runtime_recovery.py`

**Interfaces:**
- `NautilusReplayRuntime.start() / stop()` owns only the upstream runtime lifecycle.
- `NautilusReplayRuntime.publish(observation)` sends admitted observations through the upstream event path.
- `ResearchWakeupConsumer.consume(wakeup_id, result)` is durably idempotent and revalidates the exact candidate version.
- No adapter method can submit, cancel, amend, or authorize an order.

- [x] Write RED tests for no-entry invalidation and qualified trigger-to-READY/no-order scenarios.
- [x] Implement the runtime adapter against the verified public upstream event/clock/timer API; do not duplicate Nautilus internals.
- [x] Implement crash/restart recovery with durable pending wakeups and stale-readiness revocation.
- [x] Run multi-observation deterministic replay, kill/restart proof, duplicate-wakeup proof, and authority-negative tests.
- [x] Commit the runtime increment.

### Task 4: Offline order-lifecycle observation adapter

**Files:**
- Create: `src/axiom2/nautilus_runtime/order_events.py`
- Create: `tests/axiom2/nautilus_runtime/test_order_events.py`
- Modify: `docs/axiom2/NAUTILUS_RUNTIME_REVIEW_2026-10-06.md`

**Interfaces:**
- `RawOrderObservation` retains the exact upstream event type/name, payload digest, source timestamp, and receipt sequence.
- `NautilusOrderLifecycleAdapter.ingest(raw) -> LifecycleProjection` derives only a normalized comparison projection.
- The projection must cover acknowledgement, partial/full fill, cancellation, cancel rejection, late fill, duplicate/conflicting reports, and restart.
- It must compare against Axiom lifecycle/reconciliation invariants without writing to execution authority or broker transports.

- [x] Write RED tests for every required lifecycle scenario and raw-vs-derived separation.
- [x] Implement a fail-closed projection and durable replay keyed by stable order/event IDs.
- [x] Compare equivalent invariant results with existing Axiom lifecycle/reconciliation helpers; record any discrepancy as a test failure or explicit unresolved state.
- [x] Run focused and full scoped tests; keep live-feed status separate.
- [x] Commit the offline lifecycle increment.

### Task 5: Three-pass adversarial review and final verification

**Files:**
- Create: `docs/axiom2/NAUTILUS_RUNTIME_VERIFICATION_2026-10-06.md`
- Modify: `docs/axiom2/NAUTILUS_RUNTIME_REVIEW_2026-10-06.md`

- [x] Pass 1: requirements and API correctness, including actual upstream smoke evidence.
- [x] Pass 2: crash recovery, duplicate/conflict handling, stale readiness, credentials/network, and authority containment.
- [x] Pass 3: integration/regression, raw-event retention, lifecycle/reconciliation parity, and live-feed gap.
- [x] Run the complete fresh verification command and record exact counts, failures, branch/HEAD, and clean/dirty status.
- [x] Commit only verified local changes; do not merge, deploy, enable trading, or create infrastructure.

## Explicit non-goals

This plan does not add Robinhood or any other broker client, credentials, live market connectivity, order placement/cancellation, autonomous strategy orders, capital admission, production deployment, or alternative execution authority.

## Completion evidence — 2026-10-07

Resumed the preserved verified runtime checkpoint `9e996c8` on isolated branch
`codex/axiom2-nautilus-cloud-completion-2026-10-07`, as explicitly authorized.
Final reviewed/tested code `6fa68a03` reuses the original unchanged Linux
extension; no upstream reimplementation or native rebuild. Cloud run
`37563968612` passes smoke, 58 focused cases and the complete scoped gate
(1,453 passed; no failures/skips/collection errors). The inherited numeric
mismatch and exact-source registration were repaired in separate bounded,
independently reviewed commits. See
[verification](../../axiom2/NAUTILUS_RUNTIME_VERIFICATION_2026-10-06.md) for the
implemented/tested/reviewed/Mac-pending table, limitations and receipts.
Mac remains independently owned; production was not activated. Original
runtime/Luna workspace and paused LEAN PR63 remain untouched.
