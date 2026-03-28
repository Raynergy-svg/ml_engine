# Buddy Bug Report — 2026-03-28 01:46 ET

## Executive Summary
- **Total issues found: 21**
- **Critical: 3 | High: 3 | Medium: 6 | Low: 9**
- **Files with most issues**: `src/scanner/adaptive_position_sizing.py` (2), `src/recursive_intelligence/weight_learner.py` (1), `src/recursive_intelligence/learner.py` (1), `src/recursive_intelligence/persistence.py` (1), `src/scanner/engine.py` (2)
- **Key finding**: 7 issues from the 2026-03-25 report are confirmed FIXED (C-1, C-2, C-3, C-4, H-1, H-3, M-5). Three previous Critical issues remain open (C-5, C-6, C-7: non-atomic writes in recursive_intelligence). One new Critical was added: the C-4 position-size rounding fix is logically broken by a variable shadowing bug. Two new Medium issues: `EnsembleDivergenceMonitor` and `SignalFunnelTracker` are Phase 59 additions that are not wired into the production scan loop. Syntax: all Python files pass `py_compile` — no syntax errors.

---

## Comparison to Previous Report (2026-03-25 20:05 ET)

### Resolved Since Last Report ✅
- **C-1**: Non-atomic JSON write in trade journal (`execution.py`) — **FIXED**: now uses `safe_json_write()` with atomic temp+rename fallback.
- **C-2**: Division by zero in R:R returning 0.0 instead of rejection (`risk_management.py`) — **FIXED**: now returns `is_valid=False` with explicit reason when `stop_loss_pips <= 0`.
- **C-3**: Silent NAV validation skip disabling drawdown monitoring (`fx_guardrails.py`) — **FIXED**: now logs ERROR and resets to current NAV; returns `drawdown_pct=0.0` (not None) when both invalid.
- **C-4**: Position size rounding 67% oversize on small accounts (`position_sizing.py`) — **PARTIALLY FIXED** (see C-4-NEW below — the fix is logically broken by variable shadowing).
- **H-1**: Race condition on `fx_guardrails` load_state — **FIXED**: now uses `fcntl.flock(LOCK_SH)` with ImportError fallback.
- **H-3**: Non-atomic `save_state` in `fx_guardrails.py` — **FIXED**: now uses `_tmp.write_text()` + `os.replace()`.
- **M-5**: JSONL append without file locking in `continuous.py` — **FIXED**: now uses `safe_jsonl_append()` from `safe_json.py`.

### Persistent Issues (still open from 2026-03-25 report)
- **C-5**: Non-atomic write + no file locking in RL weight persistence (`weight_learner.py`) — **STILL OPEN**
- **C-6**: Non-atomic write + no file locking in learner persistence (`learner.py`) — **STILL OPEN**
- **C-7**: Non-atomic write + no file locking in session state persistence (`persistence.py`) — **STILL OPEN**
- **M-2**: 176 bare `except Exception` handlers in `engine.py` scan path — **COUNT INCREASED (was 173)**
- **M-8**: Hardcoded minimum SL (`10.0 pip`) in `risk_management.py` fallback at `_calculate_sl_tp()` — **STILL OPEN**
- **L-TODO**: `AdversarialTrainer` initialized in engine.py with a TODO comment, no call site in scan loop — **STILL OPEN**

### New Issues Found in This Scan
- 1 NEW Critical: C-4-NEW position-size rounding fix broken by variable shadowing
- 1 NEW Critical: Kelly calculation is dead code (unreachable path)
- 1 NEW High: `EnsembleDivergenceMonitor` initialized but never called (Phase 59 wiring gap)
- 1 NEW High: `SignalFunnelTracker` has zero production call sites (Phase 59 wiring gap)
- 1 NEW Medium: Undefined name `Any` in `continuous.py` line 1618 (runtime NameError risk)
- 1 NEW Low: Online RL JSONL fallback path lacks file locking (`online_rl.py` line 346)
- 4 NEW Low: Unused imports in automation modules (causal_filter, drift_remediator, dynamic_hedging, episodic_memory)

---

## Critical Issues (fix immediately)

### C-1 ✅ RESOLVED
Trade journal now uses `safe_json_write()`. Confirmed fixed.

### C-2 ✅ RESOLVED
R:R rejection now returns `is_valid=False`. Confirmed fixed.

### C-3 ✅ RESOLVED
NAV validation gap closed. Confirmed fixed.

### C-4-NEW: Position-Size Rounding Fix Broken by Variable Shadowing (NEW)
- **File**: `src/risk/position_sizing.py` lines 238–263
- **Description**: The fix for C-4 (rounding to 100 units for small accounts) uses the wrong `account_equity`. The function receives `account_equity: float` as a parameter and uses it correctly through line 256 for the risk calculation. Then at line 261:
  ```python
  account_equity = getattr(self.config, 'account_equity', 100000)
  ```
  This **shadows the parameter** with a config attribute that defaults to `100000` if not set. Since `ScannerConfig` and `PositionSizerConfig` do not define `account_equity` as a field, `getattr()` always returns `100000`, so `rounding_unit` is always `1000` (not `100`), and the small-account rounding fix is never applied.
- **Impact**: The C-4 rounding fix is completely inert on small/practice accounts. 600-unit positions still round to 1000 (67% oversize).
- **Suggested Fix**: Replace line 261 with the parameter reference already in scope: use the `account_equity` parameter value, not `getattr(self.config, ...)`. The parameter is passed from the caller with the actual account equity.

### C-5: Non-Atomic Write + No File Locking in RL Weight Persistence (PERSISTENT — 2nd report)
- **File**: `src/recursive_intelligence/weight_learner.py` lines 133–144
- **Description**: `_persist_weights()` calls `self.weights_path.write_text(json.dumps(...))` directly — no atomic temp+rename, no fcntl locking. Two processes (online RL weight update, post-trade sync) can write simultaneously.
- **Impact**: Corrupted `agent_weights.json` resets all 12 agents to equal weights, destroying all learned behavior. This is the second most critical persistence path in the system.
- **Violation**: `improvement.md` "ALWAYS write JSON atomically" + "ALWAYS use file locking (fcntl)"
- **Suggested Fix**: Import and use `safe_json_write()` from `src.scanner.automation.safe_json`.

### C-6: Non-Atomic Write + No File Locking in Learner Persistence (PERSISTENT — 2nd report)
- **File**: `src/recursive_intelligence/learner.py` lines 246–260
- **Description**: Both `_persist_learnings()` and `_persist_rules()` use `write_text(json.dumps(...))` directly — no locking, no atomicity.
- **Impact**: Concurrent writes from learning engine + RL can silently corrupt learnings and promoted rules.
- **Violation**: `improvement.md` "ALWAYS write JSON atomically"
- **Suggested Fix**: Use `safe_json_write()` from safe_json.

### C-7: Non-Atomic Write in Session State Persistence (PERSISTENT — 2nd report)
- **File**: `src/recursive_intelligence/persistence.py` line 71
- **Description**: `save()` uses `self.state_path.write_text(json.dumps(...))` directly. No atomic pattern. State file also lacks a `"version"` field.
- **Impact**: State file corruption causes loss of phase/story tracking continuity.
- **Violation**: `improvement.md` "ALWAYS write JSON atomically" + "ALWAYS include version field"
- **Suggested Fix**: Use atomic temp+rename pattern; add `state["version"] = 1` before persisting.

### C-8-NEW: Kelly Calculation is Dead Code — Adaptive Position Sizing Always Returns Fixed Fraction
- **File**: `src/scanner/adaptive_position_sizing.py` lines 250–275
- **Description**: The previous C-8 bug (negative Kelly from raw negative PnL values) was addressed with a guard at line 257:
  ```python
  if avg_loss <= 0:
      return self.config.kelly_fraction
  ```
  However, `avg_loss` is computed as `np.mean(loss_pnls)` where `loss_pnls` contains the PnL values of losing trades — which are **negative by definition** (e.g., -50, -100). Therefore `avg_loss < 0` always, and the guard fires on every call. The Kelly calculation at lines 261–274 is **permanently unreachable dead code**.
- **Impact**: Adaptive position sizing never adapts — it always returns `self.config.kelly_fraction` (the conservative default). The entire Kelly-based adaptation is silently disabled.
- **Suggested Fix**: Use `abs(avg_loss)` in `avg_loss` computation, then guard on `abs_avg_loss == 0` to prevent division by zero.

---

## High Priority Issues

### H-1 ✅ RESOLVED
`fx_guardrails` load_state race condition fixed with fcntl locking.

### H-2: Phase 59 Wiring Gap — `EnsembleDivergenceMonitor` Initialized but Never Called (NEW)
- **File**: `src/scanner/engine.py` lines 543–546 (init), no call sites
- **Description**: `self._ensemble_divergence = EnsembleDivergenceMonitor()` is initialized in `__init__`, but there are zero calls to any method on `self._ensemble_divergence` anywhere in the scan loop or post-scan hooks. The module defines `record_snapshot()` and `get_summary()` but neither is invoked.
- **Impact**: Ensemble divergence is never tracked. The Phase 59 intent (detecting structural model disagreement as a root cause of confidence suppression) produces zero data. The module is write-only dead code.
- **Violation**: `improvement.md` "ALWAYS verify both the write side AND read side of any feedback/telemetry system"
- **Suggested Fix**: Add a call to `self._ensemble_divergence.record_snapshot(tcn_confidence, ridge_confidence)` in the per-pair inference loop where both confidences are available.

### H-3 ✅ RESOLVED
`fx_guardrails` save_state is now atomic.

### H-4: Phase 59 Wiring Gap — `SignalFunnelTracker` Has No Production Call Sites (NEW)
- **File**: `src/scanner/signal_funnel_tracker.py` (exists), production files (absent)
- **Description**: `SignalFunnelTracker` was added in Phase 59 (US-363) to track gate attrition from evaluated pairs to tradeable pairs. The module has tests (`test_phase59_signal_funnel_tracker.py`) but is not imported or called anywhere in `continuous.py`, `engine.py`, `orchestrator.py`, or any other production file. No `_signal_funnel` attribute exists in any production class.
- **Impact**: Gate attrition (which stage kills signals most often) is never tracked. The diagnostic value of Phase 59's US-363 is zero in production.
- **Violation**: `improvement.md` "ALWAYS check that methods defined for integration are actually CALLED"
- **Suggested Fix**: Wire `SignalFunnelTracker.record_scan(analyses)` into `continuous.py`'s `_log_scan_cycle()` method, alongside the existing `safe_jsonl_append` call.

---

## Medium Priority Issues

### M-1 (PREVIOUSLY): Chandelier exit / OHLC wiring
- **Status**: Not re-verified in this scan — not explicitly changed since last report.

### M-2: 176 Bare `except Exception` Handlers in `engine.py` (PERSISTENT — count increasing)
- **File**: `src/scanner/engine.py`
- **Count**: 176 (was 173 in last report — still growing with each Phase)
- **Description**: Bare `except Exception` with only `logger.debug(...)` or `logger.warning(...)` swallows errors silently. Financial calculation paths should surface errors as trade rejections, not hide them.
- **Violation**: `improvement.md` "NEVER use bare except: or except Exception: pass — always log the error" + "ALWAYS re-raise or return error status after logging"
- **Suggested Fix**: Systematic audit pass; financial/gate paths should re-raise or return rejection status.

### M-3: Undefined Name `Any` in `continuous.py` (NEW)
- **File**: `src/scanner/automation/continuous.py` line 1618
- **Description**: flake8 reports `F821 undefined name 'Any'` at line 1618 in type hint `Optional[Any]` inside `_spawn_background_retrain()`. `Any` is only imported at module top level but this block has a late/conditional `import` scope. At runtime in Python, type hints in function signatures are not evaluated by default (PEP 563 forward refs are strings), but this could break with `from __future__ import annotations` absent.
- **Impact**: If `annotations` evaluation is triggered (e.g., via `get_type_hints()`), raises `NameError`. Low probability in current setup, but fragile.
- **Suggested Fix**: Add `from typing import Any` at the top of the file (it's already imported at top level in most files — verify `continuous.py` has it globally).

### M-4: `_check_interval` Computed but Never Used in `orchestrator.py` (LOW-MEDIUM)
- **File**: `src/scanner/automation/orchestrator.py` line 1084
- **Description**: `_check_interval = getattr(_cfg, "drift_remediation_check_interval", 20)` is computed every dispatch cycle but never used — the remediation is triggered on every call regardless of interval.
- **Impact**: Drift remediation runs on every orchestrator cycle, not on the configured interval. May run more frequently than intended, adding latency.
- **Suggested Fix**: Use `_check_interval` to gate remediation: only call `check_and_remediate()` when `self._scan_count % _check_interval == 0`.

### M-5 ✅ RESOLVED
JSONL append in continuous.py now uses `safe_jsonl_append`.

### M-6: Hardcoded 10-pip SL Fallback in `risk_management.py` (PERSISTENT)
- **File**: `src/risk/risk_management.py` line 294
- **Description**: When no base SL/TP values are provided, fallback uses `max(self.config.min_stop_loss_pips, 10.0)`. The `10.0` pip literal is hardcoded, violating ATR-based SL requirement.
- **Impact**: In edge cases where caller omits ATR-based values, fallback produces hardcoded SL that doesn't reflect market volatility.
- **Suggested Fix**: Log a more prominent WARNING and reject the trade rather than using a hardcoded fallback.

### M-7: `_result` Unused in Orchestrator `_sync_closed_trades_rl_dispatch` (LOW)
- **File**: `src/scanner/automation/orchestrator.py` line 776
- **Description**: `result = self._current_result` is assigned but `result` is never read in the function body.
- **Impact**: Cosmetic dead code; no runtime risk.

### M-8 (Previously M-12): `learning_engine.py` Extracts `sl_pips`/`tp_pips` but Uses Only for R:R Logging
- **Status**: Confirmed still present. `sl_pips` and `tp_pips` are extracted at lines 93–95 and used for `rr_ratio` computation/logging at lines 95 and 134, but never fed back into weight updates or pattern promotion. The learning signal from SL/TP structure is lost.

---

## Low Priority / Code Quality

### L-1: Online RL JSONL Fallback Lacks File Locking
- **File**: `src/scanner/automation/online_rl.py` lines 346–356
- **Description**: `_log_weight_updates()` has a primary path using `safe_jsonl_append` (properly locked), but the `except ImportError` fallback at line 346 uses bare `open(..., "a")` without file locking. Concurrent writes possible if ImportError fires.
- **Suggested Fix**: Fallback path should also acquire a file lock or simply raise after logging rather than falling back to unlocked write.

### L-2: Unused `closed_trades` Assignment in `orchestrator.py` Line 666
- **File**: `src/scanner/automation/orchestrator.py` line 666
- **Description**: `closed_trades = self._dispatch_results.get("closed_trades", [])` is assigned but never read in `_run_dispatch()`. Downstream steps re-fetch from `self._dispatch_results` directly.
- **Suggested Fix**: Remove the dead assignment; downstream methods already read from `_dispatch_results`.

### L-3: Unused Import `CounterfactualScenario` in `causal_filter.py` Line 315
- **File**: `src/scanner/automation/causal_filter.py` line 315
- **Description**: `CounterfactualScenario` is imported inside a function alongside `CounterfactualEngine`, but only `CounterfactualEngine` is used.
- **Suggested Fix**: Remove `CounterfactualScenario` from the import.

### L-4: Unused Import `safe_json_write` in `drift_remediator.py` Line 33
- **File**: `src/scanner/automation/drift_remediator.py` line 33
- **Description**: `safe_json_write` is imported at module level but never called in the file.
- **Suggested Fix**: Remove unused import.

### L-5: Multiple Automation Modules With Unused Imports (Cosmetic but Accumulating)
- **Files**: `dynamic_hedging.py` (json, math, datetime, pathlib.Path, Tuple), `episodic_memory.py` (json, os, dataclasses.field), `memory_manager.py` (os, shutil, OrderedDict, Callable), `agent_accuracy_matrix.py` (numpy), and ~30 others
- **Impact**: Import overhead and confusion about what each module depends on.
- **Suggested Fix**: Periodic `flake8 --select=F401` sweep to remove dead imports.

### L-6: `AdversarialTrainer` Initialized but Not Called (Has Explicit TODO)
- **File**: `src/scanner/engine.py` lines 664–673
- **Description**: The code comment explicitly notes this is unwired:
  ```python
  # TODO: Wire adversarial_trainer feedback hook — module is initialized but has no
  # call site in the scan loop.
  ```
  This has been open since at least the 2026-03-25 report.
- **Suggested Fix**: Wire a periodic `adversarial_trainer.run_test(ensemble_predictions)` call after each scan cycle, or remove the module if not planned.

### L-7: `_check_interval` Variable Computed but Never Used — Remediation Always Runs
- *(Same as M-4 above, escalated if remediation proves expensive in profiling)*

### L-8: `persistence.py` State Missing `version` Field
- **File**: `src/recursive_intelligence/persistence.py` line 64–73
- **Description**: No `"version"` field is written to the state file. Forward-compatibility requirement from `improvement.md` ("ALWAYS include version field in persisted state files") is not met.
- **Suggested Fix**: Add `state["version"] = 1` before writing.

### L-9: `min_take_profit_pips = 20.0` Default Forces Minimum 20-pip TP
- **File**: `src/risk/risk_management.py` line 74
- **Description**: `RiskManagementConfig.min_take_profit_pips = 20.0` is enforced unconditionally in `_apply_tp_constraints()`. For low-volatility H4/D1 setups, ATR-based TP may naturally be 10–15 pips; this floor overrides ATR and distorts R:R.
- **Impact**: Trades with tight setups get TP forced to 20 pips, potentially pushing R:R above what the setup supports.
- **Suggested Fix**: Make `min_take_profit_pips` dynamically derivable from ATR, or expose it as a per-instrument config.

---

## Recurring Patterns

1. **Recursive Intelligence persistence modules lag behind scanner**: `safe_json.py` in `scanner/automation/` provides atomic I/O, but `src/recursive_intelligence/` (weight_learner, learner, persistence) still use raw `write_text()`. These three files should be systematically updated to use the same safe_json infrastructure.

2. **Phase N wiring gap on the read side**: Every Phase introduces new modules, but initialization ≠ wiring. Phase 59 has two new monitoring modules (`EnsembleDivergenceMonitor`, `SignalFunnelTracker`) where the module was created and tested but never called in the production loop. This is the 4th consecutive report with this pattern (adversarial_trainer, chandelier exit, signal_funnel_tracker, ensemble_divergence_monitor).

3. **Variable shadowing masking intent**: `position_sizing.py` C-4-NEW shows that a later local assignment silently shadows a function parameter. This class of bug does not appear in any test because tests typically mock the caller, not the internal scope. These should be caught with `flake8 --select=F841` at minimum.

4. **Dead-code guard conditions**: The Kelly calculation guard `if avg_loss <= 0` is logically always true when losses are negative PnL. This pattern (correct intent, wrong sign assumption) appears again after C-8 from the previous report was "fixed" — the fix introduced a new form of the same root problem.

---

## Files Changed Since Last Report (2026-03-25 20:05 ET)

Key modified files detected (via mtime):
- `src/scanner/engine.py` — Phase 59 module initialization added
- `src/scanner/execution.py` — C-1/C-2 fixes applied
- `src/risk/fx_guardrails.py` — C-3 fix applied
- `src/risk/position_sizing.py` — C-4 fix applied (with shadowing bug introduced)
- `src/scanner/automation/continuous.py` — M-5 fix applied, undefined `Any` introduced
- `src/scanner/automation/orchestrator.py` — drift remediation dispatch added
- `src/scanner/automation/safe_json.py` — schema validators added
- `src/scanner/signal_funnel_tracker.py` — NEW Phase 59 (US-363), not wired
- `src/scanner/gate_proximity_reporter.py` — NEW Phase 59 (US-364), wired in engine.py ✅
- `src/scanner/ensemble_divergence_monitor.py` — NEW Phase 59 (US-365), initialized but not called ⚠️
- `src/scanner/automation/causal_filter.py`, `causal_counterfactual.py`, `counterfactual_learner.py` — new causal analysis modules
- `src/aura/bridge/` — new query_gate.py added
- `src/strategy_invention/` — new genetic algorithm / evolution controller added

---

## Prioritized Fix Order

| Priority | Issue | File | Est. Effort |
|----------|-------|------|-------------|
| 1 | C-4-NEW: Variable shadowing breaks rounding fix | `position_sizing.py:261` | 1 line |
| 2 | C-8-NEW: Kelly always returns fixed fraction (dead code) | `adaptive_position_sizing.py:254` | 2 lines |
| 3 | C-5: Non-atomic weight_learner write | `weight_learner.py:142` | 5 lines |
| 4 | C-6: Non-atomic learner write | `learner.py:250,258` | 10 lines |
| 5 | C-7: Non-atomic persistence write + missing version | `persistence.py:71` | 5 lines |
| 6 | H-2: Wire EnsembleDivergenceMonitor | `engine.py` | 5 lines |
| 7 | H-4: Wire SignalFunnelTracker | `continuous.py` | 10 lines |
| 8 | M-3: Undefined `Any` in continuous.py | `continuous.py:1618` | 1 import |
| 9 | M-4: _check_interval computed but unused | `orchestrator.py:1084` | 3 lines |
| 10 | M-6: Hardcoded 10-pip SL fallback | `risk_management.py:294` | 5 lines |

---

## Exterminator Results — 2026-03-28 ~03:00 ET

**Exterminator run by**: buddy-bug-squash scheduled task (automated)
**Commits**: 5 bug-fix commits on branch `tier6/maml-ridge-prototype`

### Fixed ✅

| Issue | Commit | Notes |
|-------|--------|-------|
| C-4-NEW: Variable shadowing in position_sizing.py | `63c3d29` | Removed `account_equity = getattr(...)` line that shadowed parameter |
| C-5: Non-atomic write in weight_learner.py | `9a1e915` | Now uses `safe_json_write()` with atomic fallback |
| C-6: Non-atomic writes in learner.py | `7662442` | Both `_persist_learnings()` and `_persist_rules()` now use `safe_json_write()` |
| C-7: Non-atomic write + missing version in persistence.py | `6013b9f` | Atomic write via `safe_json_write()`; `state.setdefault('version', 1)` added |
| H-2: EnsembleDivergenceMonitor not wired | N/A — pre-fixed | Already wired in engine.py (record() call at line 1742) — report was stale |
| H-4: SignalFunnelTracker not wired | N/A — pre-fixed | Already wired in continuous.py (record_scan() at line 975) — report was stale |
| M-3: Missing `Any` import in continuous.py | `873acc6` | Added `Any` to typing import line |
| M-4: `_check_interval` unused in orchestrator.py | `873acc6` | Now gates remediation via `_remediation_cycle_counter % _check_interval == 0` |
| M-6: Hardcoded 10.0 pip fallback in risk_management.py | `873acc6` | Removed literal; uses `config.min_stop_loss_pips` only |
| L-1: Unlocked JSONL fallback in online_rl.py | `ca8aa23` | ImportError path now acquires fcntl.LOCK_EX before appending |
| L-2: Dead `closed_trades` assignment in orchestrator.py | `ca8aa23` | Removed unused assignment |
| L-3: Unused `CounterfactualScenario` import in causal_filter.py | `ca8aa23` | Removed from local import |
| L-4: Unused `safe_json_write` import in drift_remediator.py | `ca8aa23` | Removed from module import |

### Skipped ⚠️

| Issue | Reason |
|-------|--------|
| C-8-NEW: Kelly dead code in adaptive_position_sizing.py | Fix (`abs(avg_loss)`) breaks `test_alternating_wins_losses` which expects `kelly_fraction` for 50/50 R:R (test was written to match broken behavior). Cannot modify test files per task rules. Fix requires updating `TestEdgeCases.test_alternating_wins_losses` to assert ≤ 0.01 instead of 0.33. |
| M-2: 176 bare except in engine.py | Systemic/ongoing — too broad for automated fix. Requires manual triage of individual handlers. |
| M-7: Unused `_result` in orchestrator | Cosmetic dead code — no runtime risk. Deferred. |
| M-8: sl_pips/tp_pips not feeding weight updates | Architectural gap — would require learning_engine.py changes and new feedback pathway. Out of scope for bug-squash. |
| L-5: 30+ files with unused imports | Too broad for automated sweep — requires flake8 audit and human review per file. |
| L-6: AdversarialTrainer TODO | Has explicit TODO comment; architectural decision needed. Skip. |
| L-7: Same as M-4 — resolved above | — |
| L-8: persistence.py missing version field | ALREADY FIXED as part of C-7. |
| L-9: min_take_profit_pips = 20.0 default | Config default decision — not a bug. Deferred. |

### New Issues Discovered During Fixing

- None. All fixes were straightforward and localized.

### Systemic Observations

1. The stale git lock files (`index.lock`, `HEAD.lock`) from prior sessions (`-rwx------` mode) required manual workarounds (`mv` to `.stale_*` files) before each commit. Git operations otherwise work normally.
2. H-2 and H-4 were already resolved since the scan report was generated (Phase 59 wiring completed in a session between scan and squash).
3. C-8-NEW (Kelly dead code) has a test that verified the broken behavior. The test needs updating before the real fix can land.
