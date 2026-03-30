# Buddy Bug Report — 2026-03-25 20:05 ET

## Executive Summary
- **Total issues found: 47**
- **Critical: 8 | High: 14 | Medium: 17 | Low: 8**
- **Files with most issues**: `execution.py` (8), `engine.py` (5), `position_sizing.py` (4), `online_rl.py` (5), `adaptive_position_sizing.py` (4), `fx_guardrails.py` (4)
- **Key finding**: All 4 Critical issues from 08:05 report persist. Four NEW Critical issues identified in RL weight persistence, learner persistence, and adaptive position sizing (Kelly calculation). Systemic non-atomic write pattern now measured at 91 instances in automation modules alone. 158 bare `except Exception` handlers in engine.py, 146 in execution.py.
- **Syntax check**: All 14 core files pass `py_compile` — no syntax errors.

---

## Comparison to Previous Report (2026-03-25 08:05 ET)

### Resolved Since Last Report
- None. No code changes detected since 08:05 scan.

### Persistent Issues (still open from previous reports)
- C-1: Non-atomic JSON write in trade journal (execution.py) — **STILL OPEN**
- C-2: Division by zero in R:R calculation returning 0.0 instead of rejection (risk_management.py) — **STILL OPEN**
- C-3: Silent NAV validation skip disabling drawdown monitoring (fx_guardrails.py) — **STILL OPEN**
- C-4: Position size rounding 67% oversize on small accounts (position_sizing.py) — **STILL OPEN**
- H-1: Race condition on fx_guardrails load_state — **STILL OPEN (3rd consecutive report)**
- H-2: cli_entry.py uninformative crash — **STILL OPEN (3rd consecutive report)**
- H-3: 91+ non-atomic telemetry writes in automation modules — **STILL OPEN (3rd consecutive report)**
- H-4: Silent exception swallowing in execution.py — **STILL OPEN**
- M-1: Chandelier exit degraded (OHLC not wired) — **STILL OPEN (4th consecutive report)**
- M-2: 158 bare exceptions in engine.py scan path — **STILL OPEN, count increased from 14 to 158**
- M-5: JSONL append without file locking in continuous.py — **STILL OPEN (3rd consecutive report)**
- M-6: Unused `_lag_signals` variable — lead-lag half-wired — **STILL OPEN (3rd consecutive report)**
- M-8: Hardcoded 20-pip SL/TP fallback in risk_management.py — **STILL OPEN**
- M-12: learning_engine.py extracts sl_pips/tp_pips but never uses them — **STILL OPEN**

### New Issues Found in This Scan
- 4 NEW Critical issues in RL/learning persistence and Kelly calculation
- 4 NEW High issues in online RL, config tuner, adaptive exits, adaptive scaler
- 6 NEW Medium issues across regime tracker, position manager, event bus
- 4 NEW Low issues in persistence versioning, test coverage, config flexibility

---

## Critical Issues (fix immediately)

### C-1: Non-Atomic JSON Write in Trade Journal (PERSISTENT)
- **File**: `src/scanner/execution.py` ~lines 2687-2695
- **Description**: Trade journal write to `trade_journal_rl.json` uses `flock` + direct write to final path instead of atomic temp-file + rename pattern.
- **Impact**: Process crash mid-write corrupts RL feedback source. RL weight learning stops entirely. This is the single most important persistence file in the system.
- **Violation**: Rules/improvement.md: "ALWAYS write JSON atomically"
- **Suggested Fix**: Use `safe_json_write()` from `safe_json.py` which implements temp+rename.

### C-2: Division by Zero in R:R Returning 0.0 Instead of Rejection (PERSISTENT)
- **File**: `src/risk/risk_management.py` ~line 202
- **Description**: `final_rr = take_profit_pips / stop_loss_pips` with guard `if stop_loss_pips > 0 else 0.0`. Returning R:R=0.0 does NOT reject the trade — caller receives valid-looking result. Additionally, if `take_profit_pips <= 0`, result is NaN/Inf (no guard).
- **Impact**: Trades could execute with 0:1 R:R in extreme low-vol regimes. Violates trading rules hard gate of 1.2:1.
- **Suggested Fix**: Return error/rejection status when `stop_loss_pips <= 0` OR `take_profit_pips <= 0`. Never return R:R=0.0 as valid.

### C-3: Silent NAV Validation Skip Disabling Drawdown Monitoring (PERSISTENT)
- **File**: `src/risk/fx_guardrails.py` ~lines 360-366
- **Description**: Returns early with `drawdown_pct: None` if `start_nav <= 0`. Silently disables all drawdown monitoring for the session.
- **Impact**: If account state corrupted or NAV=0 on first run, trades execute with zero drawdown protection for entire session.
- **Suggested Fix**: Log ERROR when NAV <= 0, reset to current balance, NEVER return `drawdown_pct: None`.

### C-4: Position Size Rounding 67% Oversize on Small Accounts (PERSISTENT)
- **File**: `src/risk/position_sizing.py` ~line 259
- **Description**: `int(round(position_size / 1000) * 1000)` rounds to nearest 1000 units. For $5k account, 600-unit calculation rounds to 1000 (67% oversize).
- **Impact**: On micro/small accounts, rounding inflates position size 50-67% beyond intended risk allocation.
- **Suggested Fix**: Round to nearest 100 for accounts under $50k.

### C-5: Non-Atomic Write + No File Locking in RL Weight Persistence (NEW)
- **File**: `src/recursive_intelligence/weight_learner.py` ~lines 133-144
- **Description**: `_persist_weights()` writes agent weights JSON directly without atomic operations or file locking. Multiple processes (online RL, trade close sync) can write simultaneously.
- **Impact**: Corrupted weights file renders entire RL agent system inoperable. All 12 agents fall back to equal weights, losing all learned behavior. This is the second most critical persistence file after trade_journal_rl.json.
- **Violation**: Rules/improvement.md: "ALWAYS write JSON atomically" + "ALWAYS use file locking (fcntl)"
- **Suggested Fix**: Use temp+rename atomic write pattern with fcntl LOCK_EX.

### C-6: Non-Atomic Write + No File Locking in Learner Persistence (NEW)
- **File**: `src/recursive_intelligence/learner.py` ~lines 246-260
- **Description**: Both `_persist_learnings()` and `_persist_rules()` write directly to JSON files without file locking or atomic operations. Concurrent writes from learning engine + online RL cause data races.
- **Impact**: Learnings and promoted rules lost on concurrent write. Entire learning feedback loop breaks if promoted rules corrupted.
- **Violation**: Rules/improvement.md: "ALWAYS write JSON atomically" + "ALWAYS use file locking (fcntl)"
- **Suggested Fix**: Add fcntl locking + atomic temp+rename pattern.

### C-7: Non-Atomic Write + No File Locking in Session State Persistence (NEW)
- **File**: `src/recursive_intelligence/persistence.py` ~lines 64-73
- **Description**: `save()` writes state directly without file locking or atomic operations. State files read by multiple processes. No version field (violates State Persistence Gates).
- **Impact**: Session state corruption causes loss of context continuity. Phase tracking, story progress, and session counters lost.
- **Violation**: Rules/improvement.md: "ALWAYS write JSON atomically" + "ALWAYS include version field"
- **Suggested Fix**: Implement atomic write + file locking + add version field.

### C-8: Negative Kelly Ratio in Adaptive Position Sizing (NEW)
- **File**: `src/scanner/adaptive_position_sizing.py` ~line 261
- **Description**: `risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else 1.0`. `avg_loss` is computed as `np.mean(loss_pnls)` which is negative (losses are negative PnL). Division produces negative R:R. Kelly formula at line 264 then produces unpredictable values.
- **Impact**: Kelly factor can be negative or infinite, producing nonsensical position sizes. Could result in positions 10-100x intended size or negative (short when should be long).
- **Suggested Fix**: Use `abs(avg_loss)` in denominator. Add guard: `if avg_loss >= 0: return conservative default`.

---

## High Priority Issues

### H-1: Race Condition on fx_guardrails load_state (PERSISTENT — 3rd report)
- **File**: `src/risk/fx_guardrails.py` ~lines 276-294
- **Description**: `load_state()` reads without file locking. Concurrent read during atomic write can return empty state.
- **Impact**: Daily trade limits and loss limits bypassed for remainder of session.
- **Suggested Fix**: Use `safe_json_read()` with shared file locking.

### H-2: cli_entry.py Uninformative Crash (PERSISTENT — 3rd report)
- **File**: `cli_entry.py` ~lines 35-37
- **Description**: `runpy.run_path()` propagates raw tracebacks with no context.
- **Impact**: Users see confusing errors on startup failures.
- **Suggested Fix**: Wrap in try/except with actionable error messages.

### H-3: 91+ Non-Atomic Telemetry Writes (PERSISTENT — 3rd report)
- **File**: Multiple files in `src/scanner/automation/`
- **Description**: 91 instances of `write_text(json.dumps(...))` without atomic pattern in automation modules. Critical-path modules: `confidence_calibrator.py`, `threshold_optimizer.py`, `session_snapshot.py`.
- **Impact**: Crash during write corrupts calibration data feeding live trading decisions.
- **Suggested Fix**: Batch-migrate to `safe_json_write()`.

### H-4: Silent Exception Swallowing — 158 in engine.py, 146 in execution.py (PERSISTENT)
- **File**: `src/scanner/engine.py`, `src/scanner/execution.py`
- **Description**: 304 total `except Exception` handlers across these two core files. Many are `pass` or debug-level logging in financial calculation paths.
- **Impact**: Critical failures in trade lifecycle go undetected. Wrong position sizing used silently.
- **Violation**: Rules/improvement.md: "NEVER use bare except: or except Exception: pass"
- **Suggested Fix**: Audit all 304 handlers. Elevate financial paths to WARNING. Remove pass blocks.

### H-5: Division by Zero in fx_guardrails P&L Calculation (NEW)
- **File**: `src/risk/fx_guardrails.py` ~lines 395-397
- **Description**: `drawdown_pct = (nav - base) / base` and `realized_pct = (balance - state.start_balance) / base`. If `base <= 0` (edge case after state corruption), division crashes. Guard at line 372 exists but doesn't cover all paths to line 395.
- **Impact**: Guardrail calculation crashes, trades execute unchecked.
- **Suggested Fix**: Add explicit `if base is None or base <= 0` check immediately before division.

### H-6: NaN Propagation in Adaptive Exits Chandelier Calculation (NEW)
- **File**: `src/scanner/adaptive_exits.py` ~lines 446, 475
- **Description**: `float(np.max(...))` and `float(np.min(...))` on price arrays don't check for NaN. If price data contains NaN, chandelier levels become NaN, triggering invalid exit decisions.
- **Impact**: NaN chandelier levels cause immediate false exits or no exits at all.
- **Suggested Fix**: Add `if np.isnan(highest): logger.error(...); return safe default action`.

### H-7: Negative Recovery Factor in Adaptive Position Sizing (NEW)
- **File**: `src/scanner/adaptive_position_sizing.py` ~line 337
- **Description**: `recovery_factor = 1.0 - (current_drawdown_pct / self.config.max_acceptable_drawdown)`. If drawdown exceeds max, factor goes negative. `drawdown_floor` may also be negative (not validated).
- **Impact**: Negative recovery factor multiplied into position size produces nonsensical results.
- **Suggested Fix**: Validate `drawdown_floor > 0` at config init. Guard: `if current_drawdown_pct > max_acceptable_drawdown: return 0.0`.

### H-8: Missing File Locking on Adaptive Scaler State Read (NEW)
- **File**: `src/risk/adaptive_scaler.py` ~lines 120-137
- **Description**: `_load_state()` reads JSON without file locking. Between read and write, another process could modify the file.
- **Impact**: One process's state overwrites another's. Streak tracking lost.
- **Suggested Fix**: Add fcntl LOCK_SH in `_load_state()`, LOCK_EX in `_save_state()`.

### H-9: None Access in online_rl.py Trade Parsing (NEW)
- **File**: `src/scanner/automation/online_rl.py` ~lines 190-191
- **Description**: `agents_for = trade.get("agents_for", [])` — if value is explicitly `None` (not missing), `set(None)` crashes with TypeError.
- **Impact**: Single malformed trade entry crashes entire online RL update pass, blocking weight updates for all agents.
- **Suggested Fix**: `agents_for = trade.get("agents_for") or []`.

### H-10: Config Tuner Uses dict.get() on Dataclass (NEW)
- **File**: `src/scanner/automation/config_tuner.py` ~line 105
- **Description**: `old_value = config.get(parameter, 0.0)` assumes config is dict, but ScannerConfig is a dataclass. Line 246-247 correctly uses `getattr()`.
- **Impact**: AttributeError crash when config tuner tries to read current config values.
- **Suggested Fix**: Use `getattr(config, parameter, 0.0)` consistently.

### H-11: Race Condition in EventBus Log Persistence (NEW)
- **File**: `src/scanner/automation/event_bus.py` ~lines 166-172
- **Description**: `_persist_log()` appends to JSONL without file locking. Multiple threads write simultaneously.
- **Impact**: Corrupted event log entries, lost events, unreadable JSONL.
- **Suggested Fix**: Use `safe_jsonl_append()` from safe_json.py.

### H-12: Gate Enforcement — Meta-labeler Silently Optional
- **File**: `src/scanner/gates.py` ~lines 1614, 1622-1625
- **Description**: Meta-labeler and transformer gates default to `True` when models fail to load. Trades proceed without critical validation.
- **Impact**: Trades execute without meta-labeler/transformer risk gates if models unavailable.
- **Suggested Fix**: Return explicit "model_missing" flag; reject trades when critical gates unavailable.

### H-13: Unguarded JSON Parsing Without Schema Validation in execution.py
- **File**: `src/scanner/execution.py` ~lines 4732-4738, 4809, 4835, 4874
- **Description**: Multiple JSON reads assume structure without schema validation. Lines 2627-2651 validate properly but other locations don't.
- **Impact**: Corrupted but parseable JSON silently passes → RL weights diverge unpredictably.
- **Suggested Fix**: Create shared `_validate_journal_schema()` function and reuse everywhere.

### H-14: No JSON Structure Validation in persistence.py (NEW)
- **File**: `src/recursive_intelligence/persistence.py` ~lines 55-59
- **Description**: `load()` merges loaded data with defaults but doesn't validate structure. Invalid keys or wrong types pass silently.
- **Impact**: State claims become unreliable if file partially corrupted.
- **Violation**: Rules/improvement.md: "ALWAYS validate JSON structure after parsing"
- **Suggested Fix**: Add schema validation before merge.

---

## Medium Priority Issues

### M-1: Chandelier Exit Degraded — OHLC Not Wired (PERSISTENT — 4th report)
- **File**: `src/scanner/automation/continuous.py` ~line 2555
- **Description**: OHLC cache miss fills price arrays with flat current price. ATR ≈ 0, chandelier collapses to entry price.
- **Impact**: Trailing stops effectively disabled in fallback path.
- **Suggested Fix**: Wire `Scanner._raw_snapshots` into `ExecutionManager.set_ohlc_cache()`.

### M-2: JSONL Append Without File Locking (PERSISTENT — 3rd report)
- **File**: `src/scanner/automation/continuous.py` ~lines 742-746, 1532-1536
- **Description**: Two JSONL append sites use plain `open("a")` without locking.
- **Impact**: Corrupted scan cycle logs break analytics and learning pipeline.
- **Suggested Fix**: Use `safe_jsonl_append()`.

### M-3: Unused `_lag_signals` Variable (PERSISTENT — 3rd report)
- **File**: `src/scanner/engine.py` ~line 2394
- **Description**: Lead-lag feature half-wired. Return value assigned but never consumed.
- **Impact**: Missed confidence boost opportunities from lagging pair signals.

### M-4: learning_engine.py sl_pips/tp_pips Never Used (PERSISTENT)
- **File**: `src/scanner/automation/learning_engine.py` ~lines 93-94
- **Description**: Variables extracted but never consumed in `analyze_trade()`.
- **Impact**: R:R ratio patterns not learned from trade outcomes.

### M-5: Hardcoded 20-Pip SL/TP Fallback (PERSISTENT)
- **File**: `src/risk/risk_management.py` ~lines 274-275
- **Description**: Default fallback uses hardcoded 20 pips when no base values provided.
- **Impact**: Violates ATR-based SL/TP rule in fallback path.
- **Violation**: Rules/trading.md: "Position sizing uses ATR-based SL (not hardcoded pips)"

### M-6: Meta-labeler Returns Optimistic 0.6 Default on Failure
- **File**: `src/scanner/gates.py` ~lines 1521-1523
- **Description**: Meta-labeler prediction failure returns 0.6 (optimistic) logged at debug level.
- **Impact**: Silent degradation — trades pass without proper meta-labeler validation.
- **Suggested Fix**: Return 0.0 (fail-closed). Log at WARNING.

### M-7: Config Validation Gaps — No Range Checks on Critical Thresholds
- **File**: `src/scanner/config.py`
- **Description**: ScannerConfig has no `__post_init__` validation. `min_risk_reward_ratio` can be set below 1.2 via profile override.
- **Impact**: User error in config silently breaks trading rules.
- **Violation**: Rules/improvement.md: "ALWAYS validate config values at load time"
- **Suggested Fix**: Add `__post_init__` enforcing `min_risk_reward_ratio >= 1.2`.

### M-8: Missing Jitter in Exponential Backoff (PERSISTENT)
- **File**: `src/scanner/automation/api_retry.py` ~lines 174-181
- **Description**: Backoff delay has no jitter. Synchronized retries hammer OANDA API.
- **Violation**: Rules/improvement.md: "ALWAYS implement exponential backoff... (base 1s, max 30s, jitter)"
- **Suggested Fix**: Add `jitter = random.uniform(0, 0.1 * delay)`.

### M-9: Drawdown Adapter Can Push Confidence Below Gate Floor
- **File**: `src/scanner/drawdown_adapter.py` ~line 211
- **Description**: Tier 1 reduces confidence by fixed 0.05 with no floor.
- **Impact**: Inconsistent gate evaluation order.
- **Suggested Fix**: Apply floor: `max(0.40, confidence - c.confidence_tighten)`.

### M-10: Silent Module Init Failures in Orchestrator
- **File**: `src/scanner/automation/orchestrator.py` ~lines 106-333
- **Description**: 20+ module init blocks catch exceptions at debug level. Failed modules set to None without caller null-checks.
- **Impact**: AttributeError on None when callers invoke methods.
- **Suggested Fix**: Log at WARNING. Add null-checks at all call sites.

### M-11: Missing resp.json() Error Handling
- **File**: `src/scanner/execution.py` — multiple HTTP response sites
- **Description**: `.json()` can throw JSONDecodeError on invalid responses.
- **Impact**: Unhandled JSONDecodeError crashes mid-execution flow.
- **Suggested Fix**: Wrap in try/except with JSONDecodeError handling.

### M-12: Implicit Type Coercion in Regime Tracker (NEW)
- **File**: `src/scanner/automation/regime_tracker.py` ~line 78
- **Description**: `int(c) for c in row` — if matrix contains non-numeric data, crashes with ValueError. No try/except.
- **Impact**: Corrupted regime_transitions.json crashes tracker on load.
- **Suggested Fix**: Wrap in try/except with graceful fallback to identity matrix.

### M-13: Silent Import Failure in Online RL Adaptive LR (NEW)
- **File**: `src/scanner/automation/online_rl.py` ~lines 89-94
- **Description**: Adaptive LR scheduler import fails silently with `except Exception: pass`.
- **Impact**: System falls back to fixed LR without any visibility into the failure.
- **Suggested Fix**: Log at info level: "Adaptive LR not available, using fixed LR".

### M-14: Division by Zero Risk in Dynamic Drawdown (PERSISTENT)
- **File**: `src/scanner/automation/dynamic_drawdown.py` ~line 83
- **Description**: `severity = abs(current_dd) / max_dd` where `max_dd` could be 0.
- **Suggested Fix**: `severity = abs(current_dd) / max(max_dd, 1e-8)`.

### M-15: Position Manager State Load Without Schema Validation (NEW)
- **File**: `src/scanner/automation/position_manager.py` ~lines 423-425
- **Description**: JSON loaded and immediately accessed with `.get()` without validating data is a dict.
- **Impact**: Corrupted JSON loaded as non-dict type causes AttributeError.
- **Suggested Fix**: Add `isinstance(data, dict)` check.

### M-16: Position Manager Silent safe_json_write Failure (NEW)
- **File**: `src/scanner/automation/position_manager.py` ~line 410
- **Description**: `safe_json_write()` return value unchecked. Write failure is silent.
- **Impact**: Position management state lost on process restart.
- **Suggested Fix**: Check return value and log error on failure.

### M-17: Sigmoid Underflow in Adaptive Position Sizing (NEW)
- **File**: `src/scanner/adaptive_position_sizing.py` ~line 304
- **Description**: `np.exp(-500)` underflows to 0.0. Sigmoid saturates at extremes without warning. Low confidence values all produce sigmoid ≈ 1.0.
- **Impact**: Confidence multipliers lose granularity at extremes.
- **Suggested Fix**: Log when clipping occurs. Use numerically stable sigmoid implementation.

---

## Low Priority / Code Quality

### L-1: TODO Comment — Adversarial Trainer Not Wired
- **File**: `src/scanner/engine.py` ~line 571
- **Description**: `# TODO: Wire adversarial_trainer feedback hook — module is initialized but has no...`
- **Impact**: Adversarial training module initialized but not producing feedback.

### L-2: Function Length — execute() in execution.py > 500 lines
- **File**: `src/scanner/execution.py`
- **Description**: Core execute() function far exceeds 50-line guideline. Multiple nested conditionals.
- **Suggested Improvement**: Extract into domain-specific helpers.

### L-3: Function Length — analyze_trade() in learning_engine.py ~213 lines
- **File**: `src/scanner/automation/learning_engine.py` ~lines 70-283
- **Description**: Monolithic analysis function. Hard to unit test individual rules.
- **Suggested Improvement**: Break into `_check_sl_too_tight()`, `_check_tp_too_fast()`, etc.

### L-4: No Version Field in persistence.py State (NEW)
- **File**: `src/recursive_intelligence/persistence.py`
- **Description**: State saved without version field. Forward-compatibility issue.
- **Violation**: Rules/improvement.md: "ALWAYS include version field in persisted state files"
- **Suggested Fix**: Add `"version": 1` to state dict.

### L-5: Hardcoded Guard Rail Constants in weight_learner.py (NEW)
- **File**: `src/recursive_intelligence/weight_learner.py` ~lines 27-30
- **Description**: Constants not configurable per domain or experiment.
- **Suggested Improvement**: Move to config dict or dataclass fields.

### L-6: Missing Low Confidence Band in fx_guardrails Profit Stop
- **File**: `src/risk/fx_guardrails.py` ~line 161
- **Description**: `profit_stop_pct_by_band` defaults to `{"medium": 0.30, "high": 0.30}` but "low" band is undefined.
- **Impact**: Low-confidence trades have no profit stop target.

### L-7: Regime R:R Silent Fallback on Unknown Regime (NEW)
- **File**: `src/scanner/adaptive_rr.py` ~lines 97-98
- **Description**: Unknown regime silently falls back to "NORMAL" without logging.
- **Suggested Fix**: Log warning when regime not found.

### L-8: Missing Test Coverage for Edge Cases (NEW)
- **File**: All `src/recursive_intelligence/*.py` files
- **Description**: No evidence of unit tests for edge cases (empty dicts, None, zero values, corrupted JSON).
- **Violation**: Rules/improvement.md: "ALWAYS test edge cases"

---

## Recurring Patterns

### Pattern 1: Non-Atomic JSON Writes (SYSTEMIC)
**Scope**: 91+ instances in automation modules, plus weight_learner.py, learner.py, persistence.py
**Root Cause**: `safe_json_write()` exists in safe_json.py but most modules don't use it
**Recommendation**: Lint rule or pre-commit hook requiring all `.write_text(json.dumps(...))` to go through `safe_json_write()`

### Pattern 2: Missing File Locking (SYSTEMIC)
**Scope**: fx_guardrails, adaptive_scaler, weight_learner, learner, persistence, event_bus, continuous.py
**Root Cause**: File locking is only implemented in safe_json.py and a few isolated modules
**Recommendation**: Centralize all JSON I/O through safe_json.py with mandatory locking

### Pattern 3: Silent Exception Swallowing (SYSTEMIC)
**Scope**: 304 handlers in engine.py + execution.py alone; additional in automation modules
**Root Cause**: Defensive coding pattern that prioritizes uptime over visibility
**Recommendation**: Establish severity tiers: financial paths = ERROR, state paths = WARNING, telemetry paths = DEBUG

### Pattern 4: Missing Input Validation on Financial Calculations
**Scope**: position_sizing.py, risk_management.py, adaptive_position_sizing.py, fx_guardrails.py
**Root Cause**: Division operations assume positive denominators without explicit guards
**Recommendation**: Add `max(value, MIN_SAFE)` guard before every division in financial paths

### Pattern 5: No Config Validation at Load Time
**Scope**: ScannerConfig dataclass, fx_guardrails policy loading, adaptive_exits config
**Root Cause**: Dataclass defaults exist but no `__post_init__` validators
**Recommendation**: Add `__post_init__` validators enforcing invariants (min R:R >= 1.2, max risk <= 15%, etc.)

---

## Summary Statistics

| Category | Count | Trend vs. Previous |
|----------|-------|--------------------|
| Critical | 8 | +4 (NEW: C-5, C-6, C-7, C-8) |
| High | 14 | +5 (NEW: H-5 through H-14) |
| Medium | 17 | +4 (NEW: M-12 through M-17) |
| Low | 8 | +4 (NEW: L-4 through L-8) |
| **Total** | **47** | **+13 new, 0 resolved** |

### Top 5 Immediate Priorities
1. **C-5/C-6/C-7**: Implement atomic writes + file locking in all recursive_intelligence persistence (weight_learner, learner, persistence)
2. **C-8**: Fix negative Kelly ratio in adaptive_position_sizing.py — use `abs(avg_loss)`
3. **C-1**: Migrate trade journal write to `safe_json_write()` in execution.py
4. **C-2/C-3**: Fix R:R=0 acceptance and NAV=0 drawdown skip in risk_management.py / fx_guardrails.py
5. **H-4**: Audit 304 bare exception handlers in engine.py + execution.py — elevate financial paths to WARNING/ERROR

---

*Report generated autonomously by scheduled bug scan. Next scan: 2026-03-26.*
