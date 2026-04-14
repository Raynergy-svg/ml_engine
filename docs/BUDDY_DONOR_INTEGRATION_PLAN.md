# Buddy Tier 7: Control Plane Integration Plan

> Strengthened from draft plan after deep investigation of donor repo and Buddy codebase.
> Agent-verified against actual line numbers, method signatures, and module architecture.

Primary donor reference:
- [Raynergy-svg/leaked-claude-code](https://github.com/Raynergy-svg/leaked-claude-code)

---

## Executive Summary

Buddy's `sync_closed_trades_rl()` is 1,260 lines of synchronous post-close fan-out (execution.py:4526-5785). Only ~15% is truly critical (broker reconciliation, journal write, RL weight update). The other ~85% (30+ analytics/learning modules) runs inline, blocking the next scan cycle.

Tier 7 introduces:
1. **Event-driven post-close fan-out** — decouple 30+ analytics modules from the critical path
2. **Policy engine** — explicit allow/deny/confirm gates on autonomous actions
3. **Control plane** — session supervision, broker health, degraded-mode handling
4. **Operator visibility** — paged event history, queue state, policy decision log

The donor repo provides proven patterns for serial event queuing, transport abstraction, session supervision, and action policy classification. All will be reimplemented in Python, leveraging mature libraries where they eliminate subtle bugs or save significant code.

---

## Dependency Strategy

### Adopt Immediately (PR 1-2)

| Library | Version | PR | Replaces | Why |
|---------|---------|-----|----------|-----|
| **`tenacity`** | >=8.0 | PR 2 | Hand-rolled retry/backoff | Battle-tested exponential backoff with jitter, retry-on-exception filters, max attempts, stop conditions. Saves ~50 lines of subtle concurrency code. Zero transitive deps. |
| **`blinker`** | >=1.7 | PR 1 | Custom pub/sub in TradingEventBus | Named signals, priority subscribers, thread-safe, weak references for auto-cleanup. Flask uses it internally. Tiny footprint. |
| **`cachetools`** | >=5.0 | PR 1 | Hand-rolled bounded dedup set | `LRUCache(maxsize=10000)` for event idempotency — bounded, O(1) lookup, automatic eviction. Replaces manual deque bookkeeping. |
| **`structlog`** | >=24.0 | PR 1 | Plain logging with manual context | JSON-structured logs with bound context (session_id, trade_id auto-attached to every log line). Makes event history queries trivial. All Tier 7 modules use it from day 1. |

### Adopt at Phase Boundary (PR 4-5)

| Library | Version | PR | Replaces | Why |
|---------|---------|-----|----------|-----|
| **`transitions`** | >=0.9 | PR 5 | Manual `TransportState` if/elif | FSM with guards, on_enter/on_exit callbacks, invalid transition rejection, optional diagram export. Transport state bugs (connecting→degraded vs reconnecting→degraded) are hard to catch without a proper FSM. |

### Evaluate Later (Phase 6+)

| Library | Trigger | Replaces | Why Wait |
|---------|---------|----------|----------|
| **`sqlite3`** (stdlib) | Event history >5000 entries or complex filter queries become slow | JSONL file scan in `event_history.py` | JSONL is fine for append + tail reads. Switch when filter_by(pair, date_range) hits performance wall. Zero new deps (stdlib). |
| **`casbin`** | Policy rules exceed ~30 or need RBAC/inheritance | Simple evaluator in `policy_engine.py` | 10 rules don't need a framework. Revisit if rules grow, need role-based policies, or need external policy management. |
| **`huey`** or **`dramatiq`** | Queue handlers start doing multi-minute work (model retraining) | Thread-based `EventQueue` | Adds Redis dependency. Overkill until actual handler durations exceed queue capacity. |

### Will NOT Adopt

| Library | Why Not |
|---------|---------|
| Any async framework (aiohttp, trio) | Buddy is synchronous Python. Forcing async would require rewriting execution.py. Thread-based queue is sufficient. |
| Celery | Massive dependency tree, needs message broker. Way overkill for 30 lightweight handlers. |
| Any ORM | safe_json.py + sqlite3 stdlib covers all persistence needs. |

### Installation

```bash
# PR 1-2 dependencies (add to requirements.txt)
pip install tenacity>=8.0 blinker>=1.7 cachetools>=5.0 structlog>=24.0

# PR 5 dependency
pip install transitions>=0.9
```

---

## Architecture Target

```
OANDA Broker
    │
    ▼
ExecutionManager.sync_closed_trades_rl()
    │
    ├── [SYNCHRONOUS] Broker reconciliation + P&L calc + journal write + RL weight update
    │
    └── emit(TRADE_CLOSED) ──► TradingEventBus
                                    │
                                    ▼
                              EventQueue (serial, bounded, backpressure)
                                    │
                                    ├── RL replay buffer handler
                                    ├── Accuracy gate handler
                                    ├── Attribution engine handler
                                    ├── Walk-forward retrainer handler
                                    ├── Drift detector handler
                                    ├── Counterfactual handler
                                    ├── Pair performance handler
                                    ├── State persistence handler
                                    └── ... (30+ handlers)

PolicyEngine ◄── continuous.py, orchestrator.py, execution.py
    │
    ├── allow (auto-proceed)
    ├── needs_confirmation (log + block)
    └── deny (block + alert)

TradingControlPlane ◄── wraps ContinuousScanner
    │
    ├── BrokerTransport state machine (connecting/connected/degraded/closed)
    ├── SessionRegistry (persist across restarts)
    ├── Queue health monitoring
    └── Degraded-mode coordination
```

---

## Donor Pattern Mapping

### Pattern 1: Serial Event Queue
- **Donor**: `SerialBatchEventUploader.ts` — generic `<T>` queue, one POST in-flight, bounded, backpressure, exponential backoff with jitter, `flush()`, `close()`
- **Python target**: `queue.Queue` (stdlib, bounded) + single consumer `threading.Thread` + **`tenacity`** for retry/backoff logic
- **Buddy use**: Non-critical post-close fan-out with retry and ordering guarantees
- **Library rationale**: `tenacity` replaces ~50 lines of hand-rolled backoff. `@retry(wait=wait_exponential(multiplier=1, max=30), stop=stop_after_attempt(3), retry=retry_if_exception_type(HandlerError))` covers the donor's entire retry strategy in one decorator.

### Pattern 2: Flush Gate
- **Donor**: `FlushGate.ts` — queue writes during transport rebuilds, drain in order after reconnect
- **Python target**: Simple dataclass with list + bool flag, or `asyncio.Event`-gated queue
- **Buddy use**: Buffer operations during OANDA reconnection, drain in order after

### Pattern 3: Transport Abstraction
- **Donor**: `ReplBridgeTransport` interface — `connect()`, `close()`, `is_connected()`, `flush()`, `get_state_label()`
- **Python target**: ABC/Protocol class with OANDA concrete implementation
- **Buddy use**: Clean broker interface enabling live/practice/mock swap without touching scanner

### Pattern 4: Session Supervision
- **Donor**: `bridgeMain.ts` + `remoteBridgeCore.ts` — credential refresh, transport rebuild, epoch mismatch recovery, FlushGate integration
- **Python target**: Background coroutine with heartbeat + reconnect backoff
- **Buddy use**: Active broker session monitoring, auto-reconnect, degraded-mode detection

### Pattern 5: Action Policy
- **Donor**: `autoMode.ts` — three-tier rules (allow/soft_deny/environment), config merge, LLM critique
- **Python target**: Dataclass rules + evaluator function + JSON config. Start simple; migrate to **`casbin`** if rules exceed ~30 or need RBAC inheritance.
- **Buddy use**: Gate autonomous actions (trade execution, model switch, retrain, threshold adjust)

### Pattern 6: Paged Event History
- **Donor**: `sessionHistory.ts` — cursor-based pagination, typed HistoryPage
- **Python target**: JSONL append log + cursor-based read API initially. Upgrade to **`sqlite3`** (stdlib) when event volume exceeds ~5000 or filter queries become slow.
- **Buddy use**: "What happened around this losing trade?" operator queries

---

## Canonical Ownership Rules (IMMUTABLE)

1. Only `ExecutionManager.sync_closed_trades_rl()` may perform canonical trade-close reconciliation and journal mutation
2. Only the canonical trade-close path may emit `TRADE_CLOSED`
3. Canonical journal writes remain synchronous and atomic via `safe_json_write()`
4. Event handlers may READ enriched outcomes but may NOT rewrite canonical close entries
5. The queue handles non-critical fan-out ONLY
6. PolicyEngine may allow/block/confirm actions but may NOT mutate trade outcomes
7. TradingControlPlane may supervise but may NOT replace execution authority
8. Exactly one close-event emitter, exactly one canonical journal write path

---

## Critical Path Analysis: sync_closed_trades_rl()

### MUST STAY SYNCHRONOUS (execution.py:4545-5568)

| Lines | Component | Why Synchronous |
|-------|-----------|-----------------|
| 4545-4553 | Journal load & parse | Source of truth |
| 4556-4558 | Pending trade filter | Determines work scope |
| 4568-4584 | OANDA auth & trade fetch | Broker reconciliation |
| 4590-4612 | Trade state resolution | Confirms closure |
| 4607-4629 | P&L calculation | Outcome truth |
| 4630-4650 | Duration & exit reason | Outcome categorization |
| 4677-4700 | Outcome dict construction | Journal data |
| 5316-5321 | Atomic journal write | Durability |
| 4903-4949 | Regime reward shaping | Feeds RL update below |
| 5478-5568 | RL agent weight update | Core feedback loop for next trade |

### SAFE TO QUEUE (30+ handlers, execution.py:4703-5778)

| Lines | Handler | Category |
|-------|---------|----------|
| 4703-4707 | RL replay buffer | Learning |
| 4708-4729 | Episodic memory | Learning |
| 4737-4757 | Accuracy gate | Analytics |
| 4758-4770 | Retrain trigger | Monitoring |
| 4771-4805 | Meta-learner | Learning |
| 4806-4833 | Ensemble weighter | Learning |
| 4834-4902 | MAML Ridge & benchmark | Learning |
| 4950-4951 | Expectancy tracker | Analytics |
| 4955-4974 | Pair performance tracker | Analytics |
| 4975-5053 | Attribution engine | Analytics |
| 5055-5073 | Gate threshold optimizer | Learning |
| 5075-5100 | Feature drift detector | Monitoring |
| 5102-5131 | Feature health monitor | Monitoring |
| 5133-5144 | Trade cluster analyzer | Analytics |
| 5146-5205 | Walk-forward optimizer | Tuning |
| 5207-5270 | Walk-forward retrainer | Training |
| 5271-5283 | Tranche tracker | Housekeeping |
| 5285-5314 | Adaptive R:R | Learning |
| 5323-5331 | AccuracyGate rebuild | Post-sync |
| 5333-5348 | Retrain trigger drift check | Monitoring |
| 5350-5387 | TradeOutcomePredictor | Training |
| 5389-5406 | PairAffinityTracker | Analytics |
| 5408-5459 | CausalCounterfactual | Analytics |
| 5461-5476 | Adaptive position sizer | Learning |
| 5572-5602 | Pair Bayesian weights | Learning |
| 5604-5645 | Agent accuracy matrix | Analytics |
| 5647-5681 | Pair-regime-agent matrix | Analytics |
| 5683-5710 | Fast-track A/B tracking | Analytics |
| 5712-5743 | Phase 18 calibration | Learning |
| 5745-5750 | Pair performance summary | Analytics |
| 5752-5778 | State persistence saves | Housekeeping |

---

## New Files

### Phase 1: Event Foundation

#### `src/scanner/automation/trading_events.py`
- `TradingEventType(Enum)` — TRADE_OPENED, TRADE_CLOSED, TRADE_MODIFIED, POLICY_BLOCKED, RETRAIN_REQUESTED, DEGRADED_MODE_ENTERED, DEGRADED_MODE_EXITED, QUEUE_FAILURE
- `TradingEvent(dataclass)` — event_id, event_type, timestamp, source, payload_version, session_id, correlation_id, payload: dict
- `TradeClosedPayload(dataclass)` — trade_id, pair, direction, entry_price, exit_price, realized_pl, pnl_pips, trade_won, exit_reason, duration_minutes, confidence, regime, agent_reasons, model, analysis_context, close_time, sl_pips, tp_pips, weighted_vote_score
- `TradeOpenedPayload(dataclass)` — trade_id, pair, direction, entry_price, sl_pips, tp_pips, lots, confidence, regime, model
- `PolicyBlockedPayload(dataclass)` — action_type, decision, reasons, matched_rules, environment_snapshot
- Helper: `create_trade_closed_event(outcome_data: dict, session_id: str) -> TradingEvent`
- Helper: `validate_event(event: TradingEvent) -> bool`

#### `src/scanner/automation/trading_event_bus.py`
- Built on **`blinker`** signals — one `Signal` per `TradingEventType`
- `TradingEventBus` class (SEPARATE from PRD EventBus in event_bus.py)
  - `subscribe(event_type, handler_fn, priority, handler_name)` — uses `blinker.Signal.connect()` with named receivers
  - `unsubscribe(handler_name)` — `Signal.disconnect()`
  - `emit(event)` — `Signal.send(event)` for sync handlers + optional queue handoff
  - `emit_to_queue(event)` — queue-only emission (non-critical)
  - Idempotency: **`cachetools.LRUCache(maxsize=10000)`** for processed event IDs — bounded, O(1), auto-evicts oldest
  - Thread-safe (blinker signals are inherently thread-safe)
  - History: append to `trained_data/trading_events.jsonl`
  - Logging: **`structlog`** with bound context (`session_id`, `event_type` auto-attached to every log line)
- Singleton: `get_trading_event_bus()`

### Phase 2: Event Queue

#### `src/scanner/automation/event_queue.py`
- Inspired by donor `SerialBatchEventUploader`
- Uses **`tenacity`** for retry logic — replaces hand-rolled exponential backoff
- `EventQueue` class:
  - `__init__(max_size=1000, max_retries=3, base_delay=1.0, max_delay=30.0)` — bounded `queue.Queue`
  - `enqueue(handler_name, handler_fn, event, priority=0)` — backpressure when full (blocks or drops)
  - `start()` — launches single consumer `threading.Thread` (daemon)
  - `stop(flush_first=True)` — graceful shutdown with optional drain
  - `flush()` — blocks until queue empty
  - Internal: serial drain with **`tenacity`** retry decorator per handler:
    ```python
    @retry(
        wait=wait_exponential(multiplier=base_delay, max=max_delay) + wait_random(0, 1),
        stop=stop_after_attempt(max_retries),
        retry=retry_if_exception_type(HandlerError),
        before_sleep=log_retry_attempt,  # structlog integration
    )
    def _execute_handler(self, handler_fn, event): ...
    ```
  - `get_status() -> dict` — depth, in_flight, failed_count, retry_count, last_failure
  - Integration with `BackgroundActivityTracker` for job visibility
  - Logging: **`structlog`** with bound handler_name, event_id, retry_attempt
- `FlushGate` class (from donor `FlushGate.ts`):
  - `start()` — activate gate, queue incoming
  - `end() -> list` — deactivate, return queued items for drain
  - `enqueue(item) -> bool` — True if gated (queued), False if pass-through
  - `drop()` — discard queued items (transport dead)

### Phase 3: Event Handlers

#### `src/scanner/automation/event_handlers.py`
- `EventHandler(Protocol)` — `name: str`, `handle(event: TradingEvent) -> bool`, `priority: int`
- `HandlerRegistry` class:
  - `register(handler)` — add handler
  - `get_handlers_for(event_type) -> list[EventHandler]` — ordered by priority
  - `bootstrap(execution_manager)` — wire all handlers from ExecutionManager modules
- Concrete handlers (each wraps existing module calls):
  - `RLReplayHandler` — wraps execution.py:4703-4707
  - `EpisodicMemoryHandler` — wraps execution.py:4708-4729
  - `AccuracyGateHandler` — wraps execution.py:4737-4757
  - `RetrainTriggerHandler` — wraps execution.py:4758-4770
  - `MetaLearnerHandler` — wraps execution.py:4771-4805
  - `EnsembleWeighterHandler` — wraps execution.py:4806-4833
  - `MAMLHandler` — wraps execution.py:4834-4902
  - `ExpectancyHandler` — wraps execution.py:4950-4951
  - `PairPerformanceHandler` — wraps execution.py:4955-4974
  - `AttributionHandler` — wraps execution.py:4975-5053
  - `GateThresholdHandler` — wraps execution.py:5055-5073
  - `DriftDetectorHandler` — wraps execution.py:5075-5131
  - `ClusterAnalyzerHandler` — wraps execution.py:5133-5144
  - `WalkForwardHandler` — wraps execution.py:5146-5270
  - `TrancheHandler` — wraps execution.py:5271-5283
  - `AdaptiveRRHandler` — wraps execution.py:5285-5314
  - `AccuracyRebuildHandler` — wraps execution.py:5323-5331
  - `PredictorTrainingHandler` — wraps execution.py:5350-5387
  - `AffinityHandler` — wraps execution.py:5389-5406
  - `CounterfactualHandler` — wraps execution.py:5408-5459
  - `AdaptiveSizerHandler` — wraps execution.py:5461-5476
  - `BayesianWeightsHandler` — wraps execution.py:5572-5602
  - `AgentAccuracyMatrixHandler` — wraps execution.py:5604-5645
  - `PairRegimeMatrixHandler` — wraps execution.py:5647-5681
  - `ABTrackingHandler` — wraps execution.py:5683-5710
  - `CalibrationHandler` — wraps execution.py:5712-5743
  - `StatePersistenceHandler` — wraps execution.py:5752-5778

### Phase 4: Policy Engine

#### `src/scanner/automation/policy_types.py`
- `ActionType(Enum)` — EXECUTE_TRADE, CLOSE_TRADE_EARLY, TIGHTEN_STOP, TAKE_PARTIAL, SWITCH_PAIR_MODEL, APPLY_THRESHOLD_ADJUSTMENT, START_BACKGROUND_RETRAIN, START_BACKGROUND_DEBUG, RESCAN_AFTER_CLOSE, FORCE_TRADE_DEGRADED
- `PolicyDecisionType(Enum)` — AUTO_ALLOW, NEEDS_CONFIRMATION, DENY
- `ActionRequest(dataclass)` — action_type, source, context: dict, timestamp
- `PolicyDecision(dataclass)` — decision_id, action_type, decision, reasons: list[str], matched_rules: list[str], environment_snapshot: dict, timestamp
- `PolicyRule(dataclass)` — name, action_types: list[ActionType], decision: PolicyDecisionType, condition: str (human-readable), priority: int

#### `src/scanner/automation/policy_engine.py`
- **V1 (Simple evaluator)**: Hand-rolled rule matching — sufficient for ~10-30 rules
- **V2 (Casbin)**: Migrate to **`casbin`** if rules exceed ~30 or need RBAC-style role inheritance
- `PolicyEngine` class:
  - `__init__(rules_path=None)` — loads rules from JSON. V1 uses simple list matching; V2 uses `casbin.Enforcer`
  - `evaluate(request: ActionRequest) -> PolicyDecision` — matches rules against request + environment
  - `get_environment() -> dict` — account_mode (paper/live), spread_regime, drawdown_pct, broker_state, open_exposure, market_session
  - `log_decision(decision)` — append to `trained_data/policy_decisions.jsonl` via **`structlog`**
  - `get_recent_decisions(n) -> list[PolicyDecision]`
  - `lint_rules() -> list[str]` — validate rules for conflicts/gaps
- Default rules (logging-only initially):
  - allow: tighten_stop (when trade open + drawdown ok), observation logging, background analytics (queue healthy)
  - soft_deny: lower thresholds in live mode, switch model on <10 trades, retrain during degraded broker
  - environment: account_mode, spread_regime, drawdown_threshold, broker_state, exposure_cap, market_session

### Phase 5: Control Plane

#### `src/scanner/automation/broker_transport.py`
- State machine powered by **`transitions`** library — eliminates manual if/elif state tracking
- `TransportState` — CONNECTING, CONNECTED, DEGRADED, RECONNECTING, CLOSED (as string states for `transitions`)
- `BrokerTransport(ABC)`:
  - `connect() -> bool`
  - `close()`
  - `is_connected() -> bool`
  - `get_state() -> str` (transitions state)
  - `get_last_heartbeat() -> datetime`
  - `flush()`
- `OANDATransport(BrokerTransport)` — concrete implementation wrapping existing OANDA client
  - Uses **`transitions.Machine`** for state management:
    ```python
    states = ['connecting', 'connected', 'degraded', 'reconnecting', 'closed']
    transitions = [
        {'trigger': 'connect_success', 'source': ['connecting', 'reconnecting'], 'dest': 'connected', 'after': '_on_connected'},
        {'trigger': 'heartbeat_timeout', 'source': 'connected', 'dest': 'degraded', 'after': '_on_degraded'},
        {'trigger': 'start_reconnect', 'source': 'degraded', 'dest': 'reconnecting', 'before': '_activate_flush_gate'},
        {'trigger': 'reconnect_failed', 'source': 'reconnecting', 'dest': 'degraded', 'after': '_increment_backoff'},
        {'trigger': 'shutdown', 'source': '*', 'dest': 'closed', 'before': '_drain_flush_gate'},
    ]
    Machine(model=self, states=states, transitions=transitions, initial='connecting')
    ```
  - Heartbeat via lightweight `/v3/accounts/{id}/summary` endpoint
  - Reconnect backoff via **`tenacity`**: `wait_exponential(multiplier=2, max=60) + wait_random(0, 2)`
  - State transitions logged via **`structlog`** with `on_enter_*` callbacks
  - `FlushGate` activated during `reconnecting` state, drained on `connected` entry

#### `src/scanner/automation/trading_control_plane.py`
- `TradingControlPlane` class:
  - `__init__(transport, event_bus, event_queue, policy_engine, session_registry)`
  - `start()` — begin supervision loop
  - `stop()` — graceful shutdown
  - `check_health() -> dict` — transport state, queue depth, last heartbeat, degraded status
  - `enter_degraded_mode(reason)` — block new trades, continue monitoring
  - `exit_degraded_mode()`
  - `handle_transport_failure()` — reconnect sequence with FlushGate
  - `get_status() -> dict` — full control plane summary
- Degraded mode triggers: transport disconnect, repeated queue failure, repeated handler failure
- Degraded mode behavior: block new execution (via policy), continue trade monitoring, log cause

#### `src/scanner/automation/session_registry.py`
- `SessionRegistry` class:
  - Persists to `trained_data/session_registry.json` via safe_json
  - Fields: session_id, started_at, transport_state, last_heartbeat, reconnect_attempts, degraded_mode (bool + reason + since), queue_depth_summary, outstanding_jobs, last_processed_event_ids
  - `create_session() -> str` — returns new session_id
  - `update_transport_state(state)`
  - `record_heartbeat()`
  - `record_reconnect_attempt()`
  - `enter_degraded(reason)` / `exit_degraded()`
  - `save()` / `load()`

### Phase 6: Operator Visibility

#### `src/scanner/automation/event_history.py`
- **V1 (JSONL)**: Start with JSONL append log — simple, fast writes, good enough for <5000 events
- **V2 (SQLite)**: Upgrade to `sqlite3` (stdlib) when event volume exceeds ~5000 or filter queries need indexing
- `EventHistory` class:
  - `__init__(backend='jsonl')` — backend-agnostic API, swap implementation without changing callers
  - Reads from `trained_data/trading_events.jsonl` (V1) or `trained_data/trading_events.db` (V2)
  - `get_latest(n) -> list[TradingEvent]`
  - `get_before(cursor_id, n) -> HistoryPage`
  - `filter_by(event_type, pair, trade_id, session_id) -> list[TradingEvent]`
  - `HistoryPage(dataclass)` — events, first_id, has_more
- V1 uses tail-based JSONL reading for efficiency (no full file load)
- V2 SQLite schema (future):
  ```sql
  CREATE TABLE events (
      event_id TEXT PRIMARY KEY,
      event_type TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      session_id TEXT,
      pair TEXT,
      trade_id TEXT,
      payload TEXT NOT NULL  -- JSON blob
  );
  CREATE INDEX idx_events_type ON events(event_type);
  CREATE INDEX idx_events_pair ON events(pair);
  CREATE INDEX idx_events_trade ON events(trade_id);
  CREATE INDEX idx_events_time ON events(timestamp);
  ```

---

## Existing File Modifications

### execution.py — Trade-Close Refactor (Phase 3)

**Target method**: `sync_closed_trades_rl()` (lines 4526-5785)

**Change**: After journal write (line 5321), emit `TRADE_CLOSED` event and route 30+ handlers through queue instead of inline calls.

**Before** (pseudocode):
```python
# Line 5321: journal write
safe_json_write(journal_path, journal)
# Lines 5323-5778: 30+ inline handler calls
self._accuracy_gate.record(...)
self._retrain_trigger.record(...)
self._meta_learner.on_trade_close(...)
# ... 27 more ...
```

**After** (pseudocode):
```python
# Line 5321: journal write (UNCHANGED)
safe_json_write(journal_path, journal)
# Lines 5478-5568: RL weight update (STAYS INLINE — core feedback)
agent_team.update_weights_from_outcome(...)
# NEW: emit event, queue handles the rest
event = create_trade_closed_event(outcome_data, session_id)
trading_event_bus.emit(event)  # triggers queue-backed handlers
```

**What stays inline** (execution.py):
- Lines 4545-5321: Full broker reconciliation + journal write
- Lines 4903-4949: Regime reward shaping (dependency for RL update)
- Lines 5478-5568: RL agent weight update (core feedback for next trade)

**What moves to handlers** (event_handlers.py):
- All 30+ analytics/learning calls listed in "SAFE TO QUEUE" table above

### continuous.py — Control Plane Integration (Phase 5)

**Target methods**:
- `run()` (line 253): Initialize TradingControlPlane before main loop
- `_run_smart_loop()` (line 1383): Add policy gates before position management
- `_perform_shutdown()` (line 867): Persist control plane + flush queue

**Specific changes**:
1. **Line ~290** (after signal handlers): Initialize TradingControlPlane, PolicyEngine, SessionRegistry
2. **Line ~516** (before auto-execute): `policy_engine.evaluate(ActionRequest(EXECUTE_TRADE, ...))`
3. **Line ~641** (before _run_smart_loop): `control_plane.check_health()`
4. **Line ~867** (shutdown): `event_queue.stop(flush_first=True)`, `session_registry.save()`, `control_plane.stop()`

### orchestrator.py — Policy Gates (Phase 4)

**Target methods**:
- `_build_dispatch_table()` (line 338): Add policy gate dispatch steps
- `run_cycle()` (line 585): Policy check before scan dispatch
- `get_system_status()` (line 1204): Include policy/control-plane state

**Specific changes**:
1. **Line ~554** (end of dispatch table): Register `policy_gate_audit` step (interval=5)
2. **Line ~615** (before scan block): Log policy environment snapshot
3. **Line ~1235** (status dict): Add `policy_engine_status`, `control_plane_status`, `queue_status`

### state_engine.py — Schema Extension (Phase 5)

**Target**: `_DEFAULT_STATE` dict (line 46)

**Add fields**:
```python
"control_plane": {
    "session_id": "",
    "transport_state": "disconnected",
    "degraded_mode": False,
    "degraded_reason": "",
    "last_heartbeat": "",
    "reconnect_attempts": 0,
},
"queue_summary": {
    "depth": 0,
    "in_flight": 0,
    "failed_count": 0,
    "last_failure": "",
},
"last_policy_block": {
    "action_type": "",
    "reason": "",
    "timestamp": "",
},
```

**New methods**:
- `update_control_plane(session_id, transport_state, ...)`
- `update_queue_summary(depth, in_flight, failed_count)`
- `record_policy_block(action_type, reason)`

### module_dispatcher.py — Queue Maintenance (Phase 5)

**Target**: `DEFAULT_FREQUENCIES` dict (line 30)

**Add entries**:
```python
"queue_health_check": 1,        # every cycle
"policy_gate_audit": 5,         # every 5 cycles
"control_plane_heartbeat": 1,   # every cycle
"session_registry_persist": 10, # every 10 cycles
```

### background_activity.py — Queue Job Tracking (Phase 2)

**No structural changes needed.** Use existing `start_activity()` / `complete_activity()` / `fail_activity()` API for:
- Queue worker lifecycle tracking
- Individual handler execution tracking
- Control plane session tracking

---

## Rollout Phases

### Phase 0: Schema Freeze
**Deliverables**: Finalize all dataclass schemas, event types, policy action types
**Exit criteria**: Schemas reviewed, no changes after this point
**Risk**: None (no runtime changes)

### Phase 1: Trading Event Foundation — RALPH
**PR 1 scope**: `trading_events.py` + `trading_event_bus.py` + tests
**No existing files modified** (import-only if needed)
**Tests**: Event creation, validation, idempotency, bus subscribe/emit/history
**Exit criteria**: Events can be emitted and validated in isolation
**Risk**: Low
**Ralph PRD**: `tasks/tier7-pr1-events.prd.json` (5 stories)
**Execution**: `cp tasks/tier7-pr1-events.prd.json .claude/ralph/prd.json && bash scripts/ralph.sh`

### Phase 2: Event Queue Foundation — RALPH
**PR 2 scope**: `event_queue.py` (includes FlushGate) + tests
**Optional**: Wire BackgroundActivityTracker for queue job visibility
**Tests**: Ordering, flush, retry/backoff, bounded capacity, graceful shutdown, FlushGate state machine
**Exit criteria**: Queue is reliable and inspectable standalone
**Risk**: Low
**Ralph PRD**: `tasks/tier7-pr2-queue.prd.json` (4 stories)
**Execution**: Merge PR 1 first, then `cp tasks/tier7-pr2-queue.prd.json .claude/ralph/prd.json && bash scripts/ralph.sh`

### Phase 3: Trade-Close Refactor — INTERACTIVE (CRITICAL PR)
**PR 3 scope**: Modify `execution.py` + add `event_handlers.py` + tests
**This is the highest-risk PR — execution.py is the canonical trade path**
**Changes**:
1. Extract 30+ post-journal handlers into `event_handlers.py` classes
2. After journal write + RL update, emit `TRADE_CLOSED` via bus
3. Bus routes to queue, queue executes handlers serially
4. **Shadow mode first**: Run both old inline path AND new queue path, compare results
**Tests**: Emit-once guarantee, journal sync unchanged, handler execution completeness, shadow-mode parity
**Exit criteria**: Trade closes emit exactly one event, journal writes remain synchronous, no regression
**Risk**: Medium-high

### Phase 4: Policy Engine (Logging Mode) — INTERACTIVE
**PR 4 scope**: `policy_types.py` + `policy_engine.py` + modify `continuous.py` + `orchestrator.py`
**Initially logging-only** — decisions recorded but NOT enforced
**Tests**: Rule loading/validation, decision evaluation, environment snapshot, lint
**Exit criteria**: Policy decisions recorded for all target action classes
**Risk**: Medium (touches continuous.py)
**Why interactive**: Wiring into continuous.py and orchestrator.py dispatch table requires full context of the watch loop lifecycle

### Phase 5: Control Plane + Session Registry — INTERACTIVE
**PR 5 scope**: `broker_transport.py` + `trading_control_plane.py` + `session_registry.py` + modify `continuous.py` + `state_engine.py`
**Library**: **`transitions`** for TransportState FSM
**Tests**: Transport state machine, heartbeat, reconnect backoff, degraded mode transitions, registry persistence/recovery
**Exit criteria**: Watch mode has transport/session visibility and explicit degraded-mode tracking
**Risk**: Medium-high (wraps ContinuousScanner)
**Why interactive**: Most interconnected PR — touches 3 existing files, wraps ContinuousScanner, introduces degraded-mode semantics

### Phase 6: Operator Visibility — RALPH
**PR 6 scope**: `event_history.py` + modify `session_snapshot.py`
**Tests**: Paged history, filters, cursor pagination
**Exit criteria**: Operators can query event history, queue state, policy blocks
**Risk**: Low
**Ralph PRD**: `tasks/tier7-pr6-history.prd.json` (3 stories)
**Execution**: After PRs 1-5 merged, `cp tasks/tier7-pr6-history.prd.json .claude/ralph/prd.json && bash scripts/ralph.sh`

### Phase 7: Controlled Enforcement — INTERACTIVE
**PR 7 scope**: Enable policy enforcement incrementally
1. First: non-destructive actions (observation logging, background analytics)
2. Then: model switches, threshold adjustments
3. Last: trade execution gating (after soak period)
**Exit criteria**: No regression under shadow validation, enforcement is incremental and reversible
**Risk**: High if rushed
**Why interactive**: Runtime behavior change — needs judgment calls on enforcement timing and soak period

---

## Execution Strategy Summary

```
RALPH (autonomous)                 INTERACTIVE (with you)
─────────────────                  ──────────────────────
PR 1: Events + Bus (5 stories)
         │
         ▼ merge
PR 2: Queue + Gate (4 stories)
         │
         ▼ merge                   PR 3: Trade-Close Refactor
                                          │
                                          ▼ merge
                                   PR 4: Policy Engine Wiring
                                          │
                                          ▼ merge
                                   PR 5: Control Plane
                                          │
                                          ▼ merge
PR 6: Event History (3 stories)
         │
         ▼ merge                   PR 7: Enforcement Rollout
```

**Total**: 12 Ralph stories (PRs 1, 2, 6) + 4 interactive sessions (PRs 3, 4, 5, 7)

**Ralph PRD files**:
- `tasks/tier7-pr1-events.prd.json` — run first
- `tasks/tier7-pr2-queue.prd.json` — run after PR 1 merged
- `tasks/tier7-pr6-history.prd.json` — run after PR 5 merged

---

## Per-PR Task Lists

### PR 1: Event Schemas + Trading Bus — RALPH (5 stories)

> Ralph PRD: `tasks/tier7-pr1-events.prd.json`
> Includes dependency install (blinker, cachetools, structlog, tenacity) as US-001

**New files:**
- [ ] Create `src/scanner/automation/trading_events.py`
  - [ ] Define `TradingEventType` enum (8 types)
  - [ ] Define `TradingEvent` dataclass with all required fields
  - [ ] Define `TradeClosedPayload` dataclass (20+ fields mapped from outcome_data)
  - [ ] Define `TradeOpenedPayload` dataclass
  - [ ] Define `PolicyBlockedPayload` dataclass
  - [ ] Implement `create_trade_closed_event()` factory
  - [ ] Implement `validate_event()` with schema checks
  - [ ] Add `payload_version = 1` for forward compatibility

- [ ] Create `src/scanner/automation/trading_event_bus.py`
  - [ ] Implement `TradingEventBus` class using **`blinker`** signals
  - [ ] One `blinker.Signal` per `TradingEventType` (dict mapping)
  - [ ] `subscribe(event_type, handler_fn, priority, handler_name)` — `signal.connect(handler_fn)`
  - [ ] `unsubscribe(handler_name)` — `signal.disconnect(handler_fn)`
  - [ ] `emit(event)` — `signal.send(event)` for sync handlers + optional queue handoff
  - [ ] `emit_to_queue(event)` — queue-only
  - [ ] Idempotency via **`cachetools.LRUCache(maxsize=10000)`** for processed event IDs
  - [ ] History append to `trained_data/trading_events.jsonl` via `safe_jsonl_append`
  - [ ] Initialize **`structlog`** logger with bound `session_id` context
  - [ ] Singleton `get_trading_event_bus()`

**Tests:**
- [ ] Create `tests/test_trading_events.py`
  - [ ] Test event creation with all required fields
  - [ ] Test event validation (missing fields, wrong types)
  - [ ] Test payload version forward compatibility
  - [ ] Test `create_trade_closed_event()` from real outcome_data shape

- [ ] Create `tests/test_trading_event_bus.py`
  - [ ] Test subscribe/emit/unsubscribe
  - [ ] Test priority ordering
  - [ ] Test idempotency (same event_id emitted twice)
  - [ ] Test thread safety (concurrent subscribe + emit)
  - [ ] Test history persistence to JSONL
  - [ ] Test singleton behavior

### PR 2: Event Queue

**Dependencies:**
- [ ] Add `tenacity>=8.0` to `requirements.txt`

**New files:**
- [ ] Create `src/scanner/automation/event_queue.py`
  - [ ] Implement `EventQueue` class
    - [ ] `__init__(max_size=1000, max_retries=3, base_delay=1.0, max_delay=30.0)`
    - [ ] `enqueue(handler_name, handler_fn, event, priority=0)`
    - [ ] `start()` — single consumer `threading.Thread(daemon=True)`
    - [ ] `stop(flush_first=True)` — graceful shutdown
    - [ ] `flush()` — block until empty via `threading.Event`
    - [ ] `get_status() -> dict` (depth, in_flight, failed_count, retry_count, last_failure)
    - [ ] Handler execution wrapped with **`tenacity`** retry:
      ```python
      @retry(
          wait=wait_exponential(multiplier=base_delay, max=max_delay) + wait_random(0, 1),
          stop=stop_after_attempt(max_retries),
          retry=retry_if_exception_type((HandlerError, RuntimeError)),
          before_sleep=lambda rs: structlog.get_logger().warning("handler_retry", attempt=rs.attempt_number),
      )
      ```
    - [ ] BackgroundActivityTracker integration (track each handler invocation)
    - [ ] **`structlog`** logging with bound handler_name, event_id
  - [ ] Implement `FlushGate` class (no library needed — 15 lines)
    - [ ] `start()` — activate gating
    - [ ] `end() -> list` — deactivate, return queued items
    - [ ] `enqueue(item) -> bool` — True if gated
    - [ ] `drop()` — discard all
    - [ ] `is_active -> bool`

**Tests:**
- [ ] Create `tests/test_event_queue.py`
  - [ ] Test FIFO ordering
  - [ ] Test bounded capacity (enqueue when full)
  - [ ] Test single-consumer serial execution
  - [ ] Test tenacity retry fires on HandlerError (mock handler that fails then succeeds)
  - [ ] Test max retries exhaustion (handler dropped after N failures — `tenacity.RetryError`)
  - [ ] Test `flush()` blocks until empty
  - [ ] Test graceful shutdown (flush_first=True drains, False drops)
  - [ ] Test handler exception isolation (one failure doesn't block others)
  - [ ] Test `get_status()` accuracy

- [ ] Create `tests/test_flush_gate.py`
  - [ ] Test gate inactive → pass-through
  - [ ] Test gate active → queues items
  - [ ] Test `end()` returns queued items in order
  - [ ] Test `drop()` discards
  - [ ] Test re-activation after end

### PR 3: Trade-Close Event Emission (CRITICAL)

**Modified files:**
- [ ] Modify `src/scanner/execution.py`
  - [ ] Add imports: `TradingEventBus`, `create_trade_closed_event`, `EventQueue`
  - [ ] Add `_event_bus` and `_event_queue` optional attributes to `__init__` (lazy init)
  - [ ] After journal write (line 5321) + RL update (line 5568):
    - [ ] Build `TradeClosedPayload` from `outcome_data`
    - [ ] Call `create_trade_closed_event(outcome_data, session_id)`
    - [ ] Emit via `_event_bus.emit(event)`
  - [ ] **Shadow mode**: Keep existing inline handlers running AND emit to queue
  - [ ] Add `_shadow_mode: bool = True` flag — when True, both paths run
  - [ ] Add comparison logging: count handlers completed inline vs via queue
  - [ ] Add method `set_event_queue(queue)` for dependency injection
  - [ ] Add method `disable_shadow_mode()` to switch to queue-only

**New files:**
- [ ] Create `src/scanner/automation/event_handlers.py`
  - [ ] Define `EventHandler` Protocol: `name`, `handle(event) -> bool`, `priority`
  - [ ] Implement `HandlerRegistry` class
    - [ ] `register(handler)`
    - [ ] `get_handlers_for(event_type) -> list`
    - [ ] `bootstrap(execution_manager)` — auto-wire all 30+ handlers
  - [ ] Implement 28+ handler classes (each wraps ONE existing module call):
    - [ ] `RLReplayHandler` — priority 10 (high)
    - [ ] `EpisodicMemoryHandler` — priority 20
    - [ ] `AccuracyGateHandler` — priority 20
    - [ ] `RetrainTriggerHandler` — priority 30
    - [ ] `MetaLearnerHandler` — priority 30
    - [ ] `EnsembleWeighterHandler` — priority 30
    - [ ] `MAMLHandler` — priority 30
    - [ ] `ExpectancyHandler` — priority 40
    - [ ] `PairPerformanceHandler` — priority 40
    - [ ] `AttributionHandler` — priority 50
    - [ ] `GateThresholdHandler` — priority 50
    - [ ] `DriftDetectorHandler` — priority 50
    - [ ] `ClusterAnalyzerHandler` — priority 60
    - [ ] `WalkForwardHandler` — priority 60
    - [ ] `TrancheHandler` — priority 60
    - [ ] `AdaptiveRRHandler` — priority 60
    - [ ] `AccuracyRebuildHandler` — priority 70
    - [ ] `PredictorTrainingHandler` — priority 70
    - [ ] `AffinityHandler` — priority 70
    - [ ] `CounterfactualHandler` — priority 80 (heavy)
    - [ ] `AdaptiveSizerHandler` — priority 70
    - [ ] `BayesianWeightsHandler` — priority 70
    - [ ] `AgentAccuracyMatrixHandler` — priority 70
    - [ ] `PairRegimeMatrixHandler` — priority 70
    - [ ] `ABTrackingHandler` — priority 70
    - [ ] `CalibrationHandler` — priority 70
    - [ ] `StatePersistenceHandler` — priority 90 (last)

**Tests:**
- [ ] Create `tests/test_trade_close_event_emission.py`
  - [ ] Test exactly ONE event emitted per trade close
  - [ ] Test journal write still synchronous and atomic
  - [ ] Test RL weight update still runs inline (not queued)
  - [ ] Test shadow mode: both inline AND queue paths complete
  - [ ] Test queue receives all 28+ handlers after emit
  - [ ] Test handler execution order (priority)
  - [ ] Test handler failure isolation

- [ ] Create `tests/test_event_handlers.py`
  - [ ] Test each handler individually with mock ExecutionManager
  - [ ] Test HandlerRegistry.bootstrap() wires all handlers
  - [ ] Test handler idempotency (same event processed twice = no side effects)

### PR 4: Policy Engine (Logging Mode)

**New files:**
- [ ] Create `src/scanner/automation/policy_types.py`
  - [ ] `ActionType` enum (10 types)
  - [ ] `PolicyDecisionType` enum (3 types)
  - [ ] `ActionRequest` dataclass
  - [ ] `PolicyDecision` dataclass (with decision_id UUID)
  - [ ] `PolicyRule` dataclass

- [ ] Create `src/scanner/automation/policy_engine.py`
  - [ ] `PolicyEngine` class
    - [ ] `__init__(rules_path=None)` — loads from JSON, falls back to defaults
    - [ ] `evaluate(request: ActionRequest) -> PolicyDecision`
    - [ ] `get_environment() -> dict` — dynamically reads account_mode, spread, drawdown, etc.
    - [ ] `log_decision(decision)` — append to `trained_data/policy_decisions.jsonl`
    - [ ] `get_recent_decisions(n=10) -> list`
    - [ ] `lint_rules() -> list[str]` — check for conflicts, missing coverage
    - [ ] `load_rules(path)` / `save_rules(path)`
  - [ ] Default rules (10+ rules covering all action types)
  - [ ] **Logging mode**: Always returns AUTO_ALLOW but logs the WOULD-BE decision

**Modified files:**
- [ ] Modify `src/scanner/automation/continuous.py`
  - [ ] Import PolicyEngine, ActionRequest, ActionType
  - [ ] Initialize `_policy_engine` in `__init__` or `run()`
  - [ ] Before auto-execute (line ~516): `policy_engine.evaluate(ActionRequest(EXECUTE_TRADE, ...))`
  - [ ] Before model switch: `policy_engine.evaluate(ActionRequest(SWITCH_PAIR_MODEL, ...))`
  - [ ] Before background retrain: `policy_engine.evaluate(ActionRequest(START_BACKGROUND_RETRAIN, ...))`
  - [ ] Log all decisions (logging mode = no enforcement yet)

- [ ] Modify `src/scanner/automation/orchestrator.py`
  - [ ] Import PolicyEngine
  - [ ] Add `policy_engine_status` to `get_system_status()` (line ~1235)
  - [ ] Add `policy_gate_audit` to dispatch table (line ~554)

**Tests:**
- [ ] Create `tests/test_policy_engine.py`
  - [ ] Test rule loading from JSON
  - [ ] Test default rules
  - [ ] Test evaluate() returns correct decision for each action type
  - [ ] Test environment snapshot capture
  - [ ] Test decision logging to JSONL
  - [ ] Test lint_rules() catches conflicts
  - [ ] Test logging mode (always AUTO_ALLOW, but logs correct would-be decision)

### PR 5: Control Plane + Session Registry

**Dependencies:**
- [ ] Add `transitions>=0.9` to `requirements.txt`

**New files:**
- [ ] Create `src/scanner/automation/broker_transport.py`
  - [ ] Define states list: `['connecting', 'connected', 'degraded', 'reconnecting', 'closed']`
  - [ ] Define transitions list with guards and callbacks (see broker_transport.py spec above)
  - [ ] `BrokerTransport(ABC)` with 6 abstract methods
  - [ ] `OANDATransport(BrokerTransport)` — concrete implementation:
    - [ ] Initialize **`transitions.Machine`** in `__init__`:
      ```python
      Machine(model=self, states=states, transitions=transitions, initial='connecting')
      ```
    - [ ] `on_enter_connected()` callback: log via **`structlog`**, drain FlushGate
    - [ ] `on_enter_degraded()` callback: log warning, activate FlushGate
    - [ ] `on_enter_reconnecting()` callback: activate FlushGate, start reconnect loop
    - [ ] Heartbeat via `/v3/accounts/{id}/summary` (lightweight)
    - [ ] Reconnect with **`tenacity`** backoff: `wait_exponential(multiplier=2, max=60) + wait_random(0, 2)`
    - [ ] FlushGate integration: activated on `reconnecting` entry, drained on `connected` entry
    - [ ] Invalid transitions auto-rejected by `transitions` library (no manual guard code needed)

- [ ] Create `src/scanner/automation/trading_control_plane.py`
  - [ ] `TradingControlPlane` class
    - [ ] `__init__(transport, event_bus, event_queue, policy_engine, session_registry)`
    - [ ] `start()` — begin supervision
    - [ ] `stop()` — graceful shutdown
    - [ ] `check_health() -> dict`
    - [ ] `enter_degraded_mode(reason)` / `exit_degraded_mode()`
    - [ ] `handle_transport_failure()` — reconnect with FlushGate
    - [ ] `get_status() -> dict`

- [ ] Create `src/scanner/automation/session_registry.py`
  - [ ] `SessionRegistry` class
    - [ ] Persists to `trained_data/session_registry.json`
    - [ ] `create_session() -> str`
    - [ ] `update_transport_state(state)`
    - [ ] `record_heartbeat()` / `record_reconnect_attempt()`
    - [ ] `enter_degraded(reason)` / `exit_degraded()`
    - [ ] `save()` / `load()`

**Modified files:**
- [ ] Modify `src/scanner/automation/continuous.py`
  - [ ] Initialize TradingControlPlane before main loop (line ~290)
  - [ ] Call `control_plane.check_health()` each cycle
  - [ ] Call `control_plane.stop()` in shutdown (line ~867)
  - [ ] Surface control plane status in watch output

- [ ] Modify `src/scanner/automation/state_engine.py`
  - [ ] Extend `_DEFAULT_STATE` with control_plane, queue_summary, last_policy_block fields
  - [ ] Add `update_control_plane()`, `update_queue_summary()`, `record_policy_block()` methods

**Tests:**
- [ ] Create `tests/test_broker_transport.py`
  - [ ] Test valid state transitions via `transitions` triggers (connect_success, heartbeat_timeout, etc.)
  - [ ] Test INVALID transitions rejected by `transitions` (e.g., `connected` → `closed` without `shutdown` trigger)
  - [ ] Test `on_enter_connected` callback drains FlushGate
  - [ ] Test `on_enter_degraded` callback activates FlushGate
  - [ ] Test heartbeat timing (mock time, verify timeout triggers `heartbeat_timeout`)
  - [ ] Test reconnect backoff progression via `tenacity` (verify delays increase)
  - [ ] Test FlushGate queues during reconnect, drains on reconnect success

- [ ] Create `tests/test_trading_control_plane.py`
  - [ ] Test degraded mode enter/exit
  - [ ] Test health check aggregation
  - [ ] Test transport failure → reconnect sequence

- [ ] Create `tests/test_session_registry.py`
  - [ ] Test session creation and persistence
  - [ ] Test load after restart
  - [ ] Test field updates

### PR 6: Event History + Visibility

**New files:**
- [ ] Create `src/scanner/automation/event_history.py`
  - [ ] `EventHistory` class with backend-agnostic API
    - [ ] `__init__(backend='jsonl')` — start with JSONL, swap to SQLite later without changing callers
    - [ ] `get_latest(n=20) -> list[TradingEvent]`
    - [ ] `get_before(cursor_id, n=20) -> HistoryPage`
    - [ ] `filter_by(event_type, pair, trade_id, session_id) -> list`
  - [ ] `HistoryPage` dataclass (events, first_id, has_more)
  - [ ] `JSONLBackend` — tail-based JSONL reading (no full file scan)
  - [ ] `SQLiteBackend` (stub for now) — **`sqlite3`** (stdlib) with indexed columns
    - [ ] Prep the schema and CREATE TABLE/INDEX statements
    - [ ] Migration utility: `migrate_jsonl_to_sqlite()` for seamless upgrade
    - [ ] Trigger: auto-migrate when JSONL exceeds 5000 lines (or manual CLI flag)
  - [ ] Logging: **`structlog`** for query timing

**Modified files:**
- [ ] Modify `src/scanner/automation/session_snapshot.py`
  - [ ] Extend snapshot schema with control_plane_summary, queue_depth, reconnect_count, degraded_duration, policy_block_count

**Tests:**
- [ ] Create `tests/test_event_history.py`
  - [ ] Test pagination (latest + before cursor) — JSONL backend
  - [ ] Test filters (by type, pair, trade_id) — JSONL backend
  - [ ] Test empty history
  - [ ] Test large history (1000+ events)
  - [ ] Test SQLiteBackend stub creates correct schema
  - [ ] Test migrate_jsonl_to_sqlite() preserves all events

### PR 7: Enforcement Rollout

- [ ] Enable policy enforcement for non-destructive actions first
- [ ] Monitor for false positives (actions incorrectly blocked)
- [ ] Enable for model switches and threshold adjustments
- [ ] Final: enable for trade execution gating (after 7-day soak)
- [ ] Add `_enforcement_mode: bool` flag to PolicyEngine (default False)
- [ ] CLI flag `--enforce-policy` to enable

---

## Non-Negotiable Guardrails

1. Canonical journal writes remain synchronous and atomic
2. Canonical close event emitted exactly once per trade close
3. Queue only handles non-critical downstream work
4. No donor TypeScript in live trading authority path
5. Policy engine starts in logging mode before enforcement
6. Control plane may supervise but not replace execution
7. All new persisted state uses `safe_json_write()` atomic patterns
8. RL agent weight update (`execution.py:5478-5568`) stays inline — core feedback loop
9. Shadow mode mandatory for PR 3 (run both paths, compare results)
10. Each PR must pass all existing tests before merge

---

## Library Integration Notes

### structlog Configuration (PR 1)

All Tier 7 modules use `structlog` from day 1. Configure once at import time in a shared location:

```python
# src/scanner/automation/logging_config.py (new file, ~20 lines)
import structlog

def configure_tier7_logging():
    """Call once at startup. All Tier 7 modules import structlog.get_logger() directly."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,    # auto-attach session_id, trade_id
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),             # switch to JSONRenderer() for prod
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

Usage in any Tier 7 module:
```python
import structlog
log = structlog.get_logger()

# Auto-attached context from contextvars (set once per cycle in continuous.py):
structlog.contextvars.bind_contextvars(session_id=session_id, scan_cycle=cycle_num)

# Then every log line in every module includes session_id and scan_cycle automatically:
log.info("handler_completed", handler="AccuracyGate", event_id=event.event_id, duration_ms=42)
```

### blinker Signal Pattern (PR 1)

```python
from blinker import Signal

class TradingEventBus:
    def __init__(self):
        self._signals = {etype: Signal(etype.name) for etype in TradingEventType}

    def subscribe(self, event_type, handler_fn, handler_name):
        self._signals[event_type].connect(handler_fn, sender=handler_name)

    def emit(self, event):
        self._signals[event.event_type].send(event)
```

### tenacity Retry Pattern (PR 2)

```python
from tenacity import retry, wait_exponential, wait_random, stop_after_attempt, retry_if_exception_type

@retry(
    wait=wait_exponential(multiplier=1, max=30) + wait_random(0, 1),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((HandlerError, RuntimeError)),
    reraise=True,  # surface final failure to queue for dead-letter handling
)
def _execute_handler(handler_fn, event):
    return handler_fn(event)
```

### transitions State Machine Pattern (PR 5)

```python
from transitions import Machine

class OANDATransport(BrokerTransport):
    states = ['connecting', 'connected', 'degraded', 'reconnecting', 'closed']

    def __init__(self):
        self.machine = Machine(
            model=self, states=self.states, initial='connecting',
            transitions=[
                {'trigger': 'connect_success', 'source': ['connecting', 'reconnecting'], 'dest': 'connected'},
                {'trigger': 'heartbeat_timeout', 'source': 'connected', 'dest': 'degraded'},
                {'trigger': 'start_reconnect', 'source': 'degraded', 'dest': 'reconnecting'},
                {'trigger': 'reconnect_failed', 'source': 'reconnecting', 'dest': 'degraded'},
                {'trigger': 'shutdown', 'source': '*', 'dest': 'closed'},
            ],
            after_state_change='_log_transition',
        )

    # transitions auto-creates self.connect_success(), self.heartbeat_timeout(), etc.
    # Invalid transitions raise MachineError — no manual guard code needed
```

### Upgrade Triggers (When to Swap)

| Current | Upgrade To | Trigger Signal |
|---------|-----------|---------------|
| JSONL event history | SQLite | `wc -l trading_events.jsonl` > 5000 OR filter queries >500ms |
| Simple policy evaluator | casbin | Rules count > 30 OR need role-based inheritance |
| Thread-based EventQueue | huey/dramatiq | Handler avg duration > 30s OR need dead-letter queue |

---

## Top Collision Risks

| Risk | Mitigation |
|------|-----------|
| Duplicating control loop | Control plane wraps ContinuousScanner, does not compete with it |
| Double-processing closes | Only execution.py:5321 emits TRADE_CLOSED, bus has idempotency |
| Journal writes off-thread | NEVER — journal write stays synchronous in execution.py |
| Sidecar order authority | No donor component gains order authority |
| Implicit risk bypass | Policy and control plane sit ABOVE existing risk/drawdown, not bypass |
| Silent queue failure | Queue exposes get_status(), integrates with BackgroundActivityTracker |
| Shadow mode divergence | Compare handler counts inline vs queue, alert on mismatch |

---

## Definition of Done

- [ ] Trade-close path is ~200 lines (down from 1,260) — broker recon + journal + RL only
- [ ] 30+ post-close handlers run through inspectable queue
- [ ] Autonomous actions are policy-gated with recorded decisions
- [ ] Watch mode has explicit broker/session supervision
- [ ] Operators can query event history, queue state, policy blocks
- [ ] Restart continuity works for session metadata and queue state
- [ ] All existing tests pass with zero regression
- [ ] Shadow mode validates queue path matches inline path before switchover
