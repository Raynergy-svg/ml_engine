# Axiom 2.0 LEAN Monitoring Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the smallest durable, deterministic Axiom candidate-monitoring slice that consumes LEAN-shaped observations, wakes research only on material events, and cannot authorize or place orders.

**Architecture:** LEAN remains the live subscription, consolidation, scheduling, and connectivity provider. A narrow adapter converts LEAN event envelopes into immutable Axiom observations; `AxiomMonitor` evaluates registered deterministic conditions and persists its event history in an operational monitor journal. Research wakeups, revalidation facts, and reconciliation receipts are typed boundaries; the monitor has no broker or production execution imports.

**Tech Stack:** Python 3.10+, existing Pydantic `StrictContract`, existing Axiom canonical JSON and SHA-256 helpers, POSIX file locking/fsync conventions, pytest, static import-boundary tests. No new runtime dependency and no vendored LEAN source.

**Spec:** `docs/superpowers/specs/2026-10-06-axiom2-lean-monitoring-design.md`

## Global Constraints

- Start from `64bf4d3780bc193bf122f830b1df04027ba8fe47` on `codex/axiom2-lean-monitoring-2026-10-06`.
- Preserve unrelated work; no reset, amend, force-push, merge, deploy, or execution-kernel redesign.
- LEAN revision is `QuantConnect/Lean@80e7843f645673bcbeaab963049f76f20f6785e1`.
- The monitor package must not import broker-write adapters, `src.axiom2.execution`, legacy execution, or broker SDK modules.
- `READY` means only “send through the existing Axiom validation/risk pipeline”; it never means order authorization.
- Existing `ExecutionAuthority` remains unchanged and deny-only.
- The deterministic monitor is continuous; research/advisor code is invoked only through accepted material-event wakeups.
- Every production function is introduced RED -> GREEN; observe each new test fail for the intended missing behavior before implementation.

## Review Focus

- Replayed or conflicting event IDs must be idempotent or fail closed, never double-trigger.
  Test in Task 2: `test_duplicate_event_is_idempotent_and_conflicting_duplicate_fails`.
- A source watermark regression must not mutate candidate state, even if the regressed event would cross a trigger.
  Test in Task 2: `test_out_of_order_observation_is_rejected_without_state_change`.
- A crash after durable material-event acceptance but before advisor delivery must recover the pending wakeup exactly once.
  Test in Task 4: `test_restart_replays_pending_wakeup_after_crash_before_delivery`.
- Invalidation racing with revalidation must win under the monitor store lock and cannot produce `READY`.
  Test in Task 3: `test_invalidation_during_revalidation_blocks_ready`.
- No caller-controlled adapter, advisor, replay, or receipt can reach a broker write or production execution implementation.
  Tests in Task 5: `test_monitoring_import_boundary` and `test_monitor_submit_attempt_cannot_create_order`.

### Task 1: Define immutable monitoring contracts and boundary manifest

**Files:**
- Create: `src/axiom2/monitoring/__init__.py`
- Create: `src/axiom2/monitoring/contracts.py`
- Create: `config/axiom2/monitoring_boundary.json`
- Test: `tests/axiom2/monitoring/test_contracts.py`

**Interfaces:**
- `CandidateState`: `DISCOVERED`, `WATCHING`, `WAITING`, `TRIGGERED`, `REVALIDATING`, `READY`, `INVALIDATED`, `EXPIRED`, `EXECUTED`, `MONITORING_POSITION`, `EXIT_TRIGGERED`, `CLOSED`.
- `MaterialEventKind`: `ENTRY_TRIGGERED`, `PRICE_CROSSED`, `VOLUME_CROSSED`, `FRESHNESS_FAILURE`, `CONNECTION_LOST`, `CONNECTION_RESTORED`, `THESIS_INVALIDATED`, `CANDIDATE_EXPIRED`, `REVALIDATION_PASSED`, `EXIT_TRIGGERED`, `POSITION_CLOSED`.
- `ThresholdRule(metric: Literal['PRICE','VOLUME'], operator: Literal['AT_OR_ABOVE','AT_OR_BELOW'], threshold: int, role: Literal['ENTRY','INVALIDATION','EXIT'])`.
- `CandidateRegistration(candidate_id, instrument_id, thesis_version, thesis_digest, registered_at, expires_at, freshness_seconds, entry_rules, invalidation_rules, exit_rules)`.
- `MarketObservation(candidate_id, instrument_id, event_id, source_sequence, observed_at, received_at, connected, price_micros, volume)`.
- `ResearchWakeup(wakeup_id, candidate_id, material_event_id, candidate_version, reason, observation_digest)`.
- `ResearchResponse(response_id, wakeup_id, candidate_id, candidate_version, decision: Literal['REVALIDATE','INVALIDATE','IGNORE'])`.
- `ReconciliationReceipt(receipt_id, candidate_id, kind: Literal['ENTRY_EXECUTED','POSITION_RECONCILED','EXIT_CLOSED'], external_execution_id, occurred_at)`.
- All contracts inherit existing `src.evidence.contracts.StrictContract`, reject unknown/mutable/naive/nonfinite values, and use canonical integer microprices/volumes.

- Boundary manifest fields: schema version, profile `monitoring-observation-only`, `execution_enabled=false`, `capital_authorized=false`, `broker_write_enabled=false`, exact monitor source inventory and SHA-256 digests.

- [ ] **Step 1: Write failing contract and boundary tests**
- [ ] **Step 2: Run `python -m pytest tests/axiom2/monitoring/test_contracts.py -q`; expected failure because the package/contracts do not exist**
- [ ] **Step 3: Implement the immutable contracts and manifest**
- [ ] **Step 4: Rerun the focused tests and confirm PASS**
- [ ] **Step 5: Commit `feat: define axiom monitoring contracts`**

### Task 2: Implement the crash-safe operational monitor journal

**Files:**
- Create: `src/axiom2/monitoring/store.py`
- Test: `tests/axiom2/monitoring/test_store.py`

**Interfaces:**
- `MonitorStore(root: Path)`.
- `MonitorStore.append(event: MonitorEvent, received_at: datetime) -> str`.
- `MonitorStore.replay() -> tuple[MonitorEvent, ...]`.
- `MonitorStore.pending_wakeups() -> tuple[MonitorEvent, ...]`.
- `MonitorStore.head_digest() -> str | None`.
- Internal `MonitorEvent` and `MonitorReceipt` contracts may live in `contracts.py`.

Use existing `canonical_bytes` and `content_digest` for payload identity. Mirror the repository’s reviewed file-lock, atomic-create, fsync, immutable-address, sequence, and previous-digest behavior, but do not call private `EvidenceStore` methods. Explain in the implementation docstring that execution/shadow journals are unsuitable because they are authority-bound to order intents and hypothetical portfolio state.

Store one immutable event receipt per sequence under the monitor root. Verify canonical bytes, filename address, sequence, previous digest, event digest, duplicate event IDs, and monotonic receipt timestamps during replay. A repeated identical event returns its existing digest; a conflicting event ID, malformed payload, digest mismatch, sequence gap, or watermark regression raises a monitor-store error and leaves the projection unchanged.

- [ ] **Step 1: Write failing tests for append/replay, digest tamper, duplicate IDs, out-of-order watermarks, atomic crash-after-create, and restart reconstruction**
- [ ] **Step 2: Run `python -m pytest tests/axiom2/monitoring/test_store.py -q`; expected RED failures for missing `MonitorStore`**
- [ ] **Step 3: Implement the minimal immutable journal and replay verifier**
- [ ] **Step 4: Rerun the focused store tests and confirm PASS**
- [ ] **Step 5: Commit `feat: add durable monitor event journal`**

### Task 3: Implement deterministic candidate state transitions

**Files:**
- Create: `src/axiom2/monitoring/monitor.py`
- Modify: `src/axiom2/monitoring/contracts.py` only if a state projection contract is needed
- Test: `tests/axiom2/monitoring/test_monitor.py`

**Interfaces:**
- `AxiomMonitor(store: MonitorStore, wakeup_sink: ResearchWakeupPort | None = None)`.
- `AxiomMonitor.register(candidate: CandidateRegistration, command_id: str) -> CandidateState`.
- `AxiomMonitor.start_watching(candidate_id: str, command_id: str) -> CandidateState`.
- `AxiomMonitor.wait(candidate_id: str, command_id: str) -> CandidateState`.
- `AxiomMonitor.observe(observation: MarketObservation) -> CandidateState`.
- `AxiomMonitor.expire(now: datetime) -> tuple[str, ...]`.
- `AxiomMonitor.apply_research_response(response: ResearchResponse) -> CandidateState`.
- `AxiomMonitor.revalidate(candidate_id: str, observation: MarketObservation, command_id: str) -> CandidateState`.
- `AxiomMonitor.apply_reconciliation(receipt: ReconciliationReceipt) -> CandidateState`.
- `AxiomMonitor.state(candidate_id: str) -> CandidateState`.
- `AxiomMonitor.process_pending_wakeups() -> tuple[str, ...]`.
- `ResearchWakeupPort.wake(wakeup: ResearchWakeup) -> None`.

Replay the journal to reconstruct state before every mutation or from a verified in-memory projection rebuilt at initialization. Serialize each command under the store lock. Entry conditions require fresh connected observations and deterministic crossing of all registered price/volume rules. Stale observations produce a freshness material event but cannot trigger. Invalidations take precedence over trigger and revalidation transitions. Expiry is terminal for candidates that have not reached a reconciled position.

`apply_research_response` may move `TRIGGERED` to `REVALIDATING` or invalidate; it may not move directly to `READY`. `revalidate` requires a newer connected observation, freshness within the registered TTL, unchanged thesis version, unexpired candidate, and all entry conditions still true. It returns `READY` only as a monitor fact.

`apply_reconciliation` is the only transition source for `EXECUTED`, `MONITORING_POSITION`, and `CLOSED`. Exit thresholds are evaluated only while `MONITORING_POSITION`; an exit crossing emits `EXIT_TRIGGERED` and a wakeup but does not submit or create an order.

- [ ] **Step 1: Write failing tests for the full state graph, price/volume crossings, stale rejection, stale-to-fresh recovery, invalidation, expiration, disconnect/reconnect, and post-entry exit**
- [ ] **Step 2: Run `python -m pytest tests/axiom2/monitoring/test_monitor.py -q`; expected RED failures for missing `AxiomMonitor`**
- [ ] **Step 3: Implement the minimal deterministic transition reducer and wakeup outbox calls**
- [ ] **Step 4: Rerun focused monitor tests and confirm PASS**
- [ ] **Step 5: Commit `feat: add deterministic axiom candidate monitor`**

### Task 4: Add the narrow LEAN adapter and recovery-safe wakeup delivery

**Files:**
- Create: `src/axiom2/monitoring/lean_adapter.py`
- Test: `tests/axiom2/monitoring/test_lean_adapter.py`
- Modify: `NOTICE`
- Create: `docs/axiom2/LEAN_MONITORING_INTEGRATION.md`

**Interfaces:**
- `LeanObservationEnvelope(subscription_id, event_id, source_sequence, candidate_id, instrument_id, observed_at, received_at, connected, price_micros, volume)`.
- `LeanConnectionEnvelope(subscription_id, event_id, source_sequence, connected, occurred_at, received_at)`.
- `LeanObservationAdapter(monitor: AxiomMonitor)`.
- `LeanObservationAdapter.on_data(envelope: LeanObservationEnvelope) -> CandidateState`.
- `LeanObservationAdapter.on_connection(envelope: LeanConnectionEnvelope) -> CandidateState`.
- `LeanObservationAdapter.on_scheduled_sweep(now: datetime) -> tuple[str, ...]`.

The adapter validates the LEAN-shaped envelope and delegates only normalized observations/connectivity/sweep events. It does not implement subscriptions, consolidation, scheduling, reconnect logic, or order operations. The documentation pins LEAN commit `80e7843f...`, identifies `SubscriptionManager`, consolidators, `ScheduledEvent`, `ScheduleManager`, `LiveTradingDataFeed`, `LiveSynchronizer`, and `LiveTradingRealTimeHandler`, and states that the test fixtures do not prove a live sidecar.

`process_pending_wakeups` is outbox-like: durable material events are accepted before delivery; a sink failure leaves the wakeup pending; restart delivers it once to an idempotent sink and records delivery. Duplicate research responses are accepted only when identical; duplicate reconciliation receipts are idempotent only when identical.

- [ ] **Step 1: Write failing adapter, crash-before-wakeup, duplicate-response, duplicate-reconciliation, and reconnect tests**
- [ ] **Step 2: Run `python -m pytest tests/axiom2/monitoring/test_lean_adapter.py -q`; expected RED failures for missing adapter/outbox behavior**
- [ ] **Step 3: Implement the narrow adapter and durable wakeup delivery**
- [ ] **Step 4: Rerun adapter tests and confirm PASS**
- [ ] **Step 5: Commit `feat: attach lean observation adapter`**

### Task 5: Prove dependency containment and execution authority preservation

**Files:**
- Create: `tests/axiom2/monitoring/test_monitoring_boundary.py`
- Modify: `config/axiom2/monitoring_boundary.json` with final source hashes
- Modify: `NOTICE` only for the LEAN attribution entry

**Interfaces:**
- Boundary test scans the monitoring package AST/import graph and rejects `src.axiom2.execution`, `src.axiom2.brokers`, legacy execution, order SDKs, dynamic import/eval/exec, and direct filesystem/network process capabilities outside the store’s reviewed persistence functions.
- Authority test instantiates `ExecutionAuthority`, proves the monitor exposes no submit/review/cancel method, and verifies any attempted existing authority submission remains `BLOCKED` with `execution_enabled=false` and `capital_authorized=false`.

- [ ] **Step 1: Write failing boundary/authority tests that assert the absence of forbidden imports and order methods**
- [ ] **Step 2: Run `python -m pytest tests/axiom2/monitoring/test_monitoring_boundary.py -q`; expected RED until the boundary inventory/test harness exists**
- [ ] **Step 3: Implement the static boundary test and final manifest hashes without changing execution code**
- [ ] **Step 4: Run focused boundary tests and confirm PASS**
- [ ] **Step 5: Commit `test: enforce monitoring authority boundary`**

### Task 6: Run the proof-of-concept, full verification, and three adversarial reviews

**Files:**
- Test: `tests/axiom2/monitoring/test_monitoring_poc.py`
- Modify: `docs/axiom2/LEAN_MONITORING_INTEGRATION.md` with exact synthetic transcript and remaining gaps

**Interfaces:**
- Scenario A: register bullish candidate, remain `WAITING`, receive deteriorating/stale observation, emit thesis invalidation, end `INVALIDATED`; assert no wakeup reaches an execution function.
- Scenario B: register candidate, cross price and volume entry thresholds, wake advisor, accept fresh revalidation, end `READY`; assert no order is created and existing `ExecutionAuthority` remains blocked.
- Scenario C: apply reconciliation `ENTRY_EXECUTED`, then `POSITION_RECONCILED`, cross exit threshold, apply `EXIT_CLOSED`; assert `EXECUTED -> MONITORING_POSITION -> EXIT_TRIGGERED -> CLOSED`.

- [ ] **Step 1: Write the PoC tests first and run them RED**
- [ ] **Step 2: Implement only the required fixture transcript/assertions**
- [ ] **Step 3: Run focused suite: `python -m pytest tests/axiom2/monitoring -q`**
- [ ] **Step 4: Run the applicable Axiom/evidence regression gate and boundary probes already defined by the repository**
- [ ] **Step 5: Run three adversarial passes: correctness/recovery, authority/bypass, dependency/failure containment; turn every demonstrated bypass into a regression test and fix it before final verification**
- [ ] **Step 6: Run lint/static checks and inspect branch diff/status; record exact results and remaining live gaps**
- [ ] **Step 7: Commit `test: verify bounded lean monitoring slice`**

## Self-review

Coverage mapping:

- Lifecycle and durable WAIT: Tasks 1–3 and 6.
- LEAN event/connection/scheduler boundary: Task 4.
- Freshness, stale-to-fresh, ordering, duplicate, restart, crash, and races: Tasks 2–4.
- Research wakeup without continuous LLM: Task 4.
- Reconciliation-only entry/post-entry/exit: Tasks 3 and 6.
- No second execution path and no direct authority access: Task 5.
- Apache attribution and pinned upstream revision: Task 4.
- Live limitations explicitly retained: spec and Task 6 documentation.

Nothing in this plan changes `ExecutionAuthority`, portfolio/risk authority, promotion, broker read contracts, or reconciliation truth.