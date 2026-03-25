# Buddy Bug Report — 2026-03-24 10:04 ET

## Executive Summary
- **Total issues found: 44**
- **Critical: 6 | High: 10 | Medium: 17 | Low: 11**
- **Files with most issues**: engine.py (5), execution.py (4), safe_json.py (5), learning_engine.py (3), state_engine.py (3), orchestrator.py (3)
- **Systemic patterns**: Non-atomic JSON writes (7 instances), bare `except Exception:` (60+ instances across codebase), missing config validation, division-by-zero risks in financial calculations

---

## Critical Issues (Fix Immediately)

### C-1: Gate Logic Violation — Missing momentum_passed Check
- **File**: `src/scanner/engine.py` ~line 2836-2838
- **Description**: Code allows trade execution when only `confidence_passed AND risk_passed` are true, WITHOUT verifying `momentum_passed`. The fallback condition `if result.gates_passed or (result.confidence_passed and result.risk_passed)` bypasses the momentum gate entirely.
- **Impact**: Trades execute with weak/missing momentum signals, violating the core rule "ALL THREE gates must pass." Increases whipsaw risk and losses.
- **Suggested Fix**: Change condition to `if result.gates_passed or (result.confidence_passed and result.risk_passed and result.momentum_passed)`

### C-2: Gate Logic Violation — Extreme Regime Skips Confidence & Momentum
- **File**: `src/scanner/engine.py` ~line 2867-2869
- **Description**: EXTREME volatility re-qualification checks only `risk_passed`, ignoring both confidence and momentum gates. `if result.risk_passed: result.is_tradeable = True`
- **Impact**: In EXTREME regimes (when risk is highest), trades qualify with only 1 of 3 gates passing. This is the exact opposite of desired behavior — extreme regimes should be MORE restrictive, not less.
- **Suggested Fix**: Add `and result.confidence_passed and result.momentum_passed` to the condition.

### C-3: Division by Zero — Learning Engine Aggregate Win Rate
- **File**: `src/scanner/automation/learning_engine.py` ~line 426
- **Description**: `overall_wr = wins / total` where `total = len(resolved)` can be 0 after filtering. The `if len(resolved) < 3: return` guard above uses `<3` not `==0`, but a race condition or data mutation between the check and the division could cause a crash.
- **Impact**: Crashes learning extraction for override batch analysis; propagates to rule promotion pipeline.
- **Suggested Fix**: Add explicit `if total == 0: return all_entries` guard immediately before division.

### C-4: Division by Zero — Learning Engine Exit Reason Analysis
- **File**: `src/scanner/automation/learning_engine.py` ~line 746
- **Description**: `distribution[reason] = len(pnl_list) / total_trades` without checking `total_trades > 0`.
- **Impact**: Crashes exit reason pattern extraction during low-trade-volume analysis periods.
- **Suggested Fix**: Guard with `if total_trades > 0:` before the distribution calculation loop.

### C-5: Type Mismatch Crash — StateEngine Outcome Access
- **File**: `src/scanner/automation/state_engine.py` ~line 154
- **Description**: Filters for `e.get("outcome") is not None`, then calls `.get("trade_won")` on the outcome value. But `outcome` can be either a dict OR a string (e.g., "win"/"loss"). Calling `.get()` on a string raises `AttributeError`.
- **Impact**: Crashes portfolio snapshot calculation mid-cycle; can kill the entire trading session.
- **Suggested Fix**: `if isinstance(e.get("outcome"), dict) and e["outcome"].get("trade_won", False)`

### C-6: Division by Zero — Adaptive Position Sizing Drawdown Recovery
- **File**: `src/scanner/adaptive_position_sizing.py` ~line 337
- **Description**: `current_drawdown_pct / self.config.max_acceptable_drawdown` will crash if `max_acceptable_drawdown` is 0. No validation prevents this config value from being zero.
- **Impact**: Position sizing calculation crashes, potentially blocking all trade execution.
- **Suggested Fix**: Add validation at config load: `if self.config.max_acceptable_drawdown <= 0: raise ValueError(...)` and guard at calculation site.

---

## High Priority Issues

### H-1: Non-Atomic JSON Writes in StateEngine
- **File**: `src/scanner/automation/state_engine.py` lines 89, 170, 180
- **Description**: Direct `.write_text()` calls for state persistence are NOT atomic. Concurrent access or crash during write corrupts the file.
- **Impact**: Stale or corrupted state files cause inconsistent trading decisions.
- **Suggested Fix**: Use `safe_json_write()` from safe_json module for all state persistence.

### H-2: Non-Atomic JSON Writes in WeightLearner
- **File**: `src/recursive_intelligence/weight_learner.py` ~line 142
- **Description**: Agent weights written via `.write_text()` without atomic guarantee. Agent weights are critical financial data.
- **Impact**: Corrupted weight files cause agent misweighting; systematic trading errors.
- **Suggested Fix**: Use `safe_json_write()` or implement temp-file + `os.rename()` pattern.

### H-3: Non-Atomic JSON Writes in SessionPersistence
- **File**: `src/recursive_intelligence/persistence.py` ~line 71
- **Description**: Same atomic write issue across the persistence layer for domain state.
- **Impact**: Domain state (market, human, bridge) can be corrupted on crash.
- **Suggested Fix**: Use `safe_json_write()`.

### H-4: Silent Exception Swallowing in safe_json Backup Recovery
- **File**: `src/scanner/automation/safe_json.py` ~lines 139-140
- **Description**: `except Exception: pass` in backup recovery path. If backup restoration fails, the error is completely hidden.
- **Impact**: Undetectable backup recovery failures; system may silently return corrupted data. Violates "Silent Exception Prevention" rule.
- **Suggested Fix**: `except Exception as bak_err: logger.warning(f"Backup validation failed for {path}: {bak_err}")`

### H-5: Missing Handler for 'monitor' Command
- **File**: `main.py` ~line 566
- **Description**: `_DISPATCH_TABLE` includes "status" but no "monitor" entry, despite `_handle_monitor` being defined. The dispatch logic will print "Unknown command: monitor" and exit.
- **Impact**: `python main.py monitor` silently fails instead of running the monitor.
- **Suggested Fix**: Add `"monitor": _handle_monitor` to `_DISPATCH_TABLE`.

### H-6: Unhandled Exception in cli_entry.py
- **File**: `cli_entry.py` ~line 37
- **Description**: `runpy.run_path()` can raise `FileNotFoundError` or `SyntaxError` with no exception handling.
- **Impact**: Ugly, unhelpful crash if main.py is missing or has syntax errors.
- **Suggested Fix**: Wrap in try/except with user-friendly error messages.

### H-7: Missing JSON Encoding Specification in safe_json Recovery
- **File**: `src/scanner/automation/safe_json.py` ~line 137
- **Description**: `path.write_text()` called without `encoding="utf-8"` during backup recovery, while reads specify utf-8.
- **Impact**: Encoding mismatch on non-UTF-8 default systems causes subsequent JSONDecodeError.
- **Suggested Fix**: Add `encoding="utf-8"` to all `write_text()` calls.

### H-8: Memory Leak — Unbounded _decisions List
- **File**: `src/scanner/automation/position_manager.py` ~lines 390-392
- **Description**: `_decisions` list only trimmed after exceeding 500 entries. Long watch-mode sessions accumulate unbounded memory.
- **Impact**: Memory usage grows continuously in long-running sessions.
- **Suggested Fix**: Trim on load (keep last 250) and enforce max size more aggressively.

### H-9: Memory Leak — Unbounded _action_counts Dictionary
- **File**: `src/scanner/automation/position_manager.py` ~lines 226, 234-237
- **Description**: `_action_counts` and `_last_action_bar` dicts grow unbounded if `cleanup_trade()` isn't called for every trade.
- **Impact**: Slow memory leak for every uncleaned trade in long sessions.
- **Suggested Fix**: Implement periodic cleanup or LRU eviction with max-size enforcement.

### H-10: Missing JSON Structure Validation After Parse
- **File**: `src/scanner/automation/state_engine.py` ~lines 149, 154
- **Description**: After parsing journal JSON, code assumes structure without validation. Missing keys cause KeyError.
- **Impact**: Malformed journal files crash portfolio snapshot calculation.
- **Suggested Fix**: Use `.get("outcome", {})` and validate types before access.

---

## Medium Priority Issues

### M-1: Unsafe Array Access in Price Parsing
- **File**: `src/scanner/execution.py` ~lines 2274-2275
- **Description**: `[0]` indexing on `asks`/`bids` arrays without bounds check. Empty arrays from OANDA cause IndexError.
- **Impact**: Runtime crash if OANDA returns empty price data.
- **Suggested Fix**: Check `if asks and bids:` before indexing.

### M-2: Corrupted Agent Weights File Not Recovered
- **File**: `src/scanner/agents/_team.py` ~lines 206-222
- **Description**: JSON decode error logged but corrupted file not renamed/removed. Next session retries same broken file.
- **Impact**: Persistent failure state for agent weight loading across sessions.
- **Suggested Fix**: Rename corrupted file to `.bak.corrupt` and fall back to defaults.

### M-3: Non-Atomic File Write Fallback in Execution
- **File**: `src/scanner/execution.py` ~lines 1927-1933
- **Description**: Fallback from fcntl-locked write uses `.write_text()` which is non-atomic.
- **Impact**: Trade journal corruption risk on process crash during write.
- **Suggested Fix**: Use atomic temp-file + rename pattern in fallback path.

### M-4: Silent Skip of Missing Closed Trades in RL Sync
- **File**: `src/scanner/execution.py` ~lines 2727-2730
- **Description**: RL sync silently skips trades not found in OANDA closed trades without logging reason.
- **Impact**: No visibility into sync completeness; could mask closed positions not recorded.
- **Suggested Fix**: Log warning with trade ID when skipping.

### M-5: Missing Timeout Configuration on OANDA API Calls
- **File**: `src/scanner/execution.py` ~lines 227-252
- **Description**: `retry_api_call` wrapper doesn't set explicit timeouts. Per rules: connect=5s, read=30s.
- **Impact**: API calls can hang indefinitely if OANDA becomes unresponsive, blocking entire scan cycle.
- **Suggested Fix**: Add `timeout=(5, 30)` to all requests through retry wrapper.

### M-6: Race Condition in fx_guardrails State File Access
- **File**: `src/risk/fx_guardrails.py` ~lines 282, 325
- **Description**: `load_state()` and `save_state()` don't use file locking on fx_state_*.json files.
- **Impact**: Concurrent access could corrupt daily state or allow more trades than configured daily limit.
- **Suggested Fix**: Use `safe_json_read`/`safe_json_write` from safe_json module.

### M-7: Config Validation Not Enforced at Load Time (fx_guardrails)
- **File**: `src/risk/fx_guardrails.py` ~lines 112-181
- **Description**: Config values loaded without range/type validation. `atr_stop_mult` could be negative or zero.
- **Impact**: Invalid config values cause downstream calculation errors silently.
- **Suggested Fix**: Add validation: `if atr_mult <= 0: raise ValueError(...)`

### M-8: Missing Error Context in Orchestrator Lazy Initialization
- **File**: `src/scanner/automation/orchestrator.py` ~lines 107-320
- **Description**: 20+ module initialization failures logged as warnings without stack traces.
- **Impact**: Debugging initialization failures is extremely difficult without tracebacks.
- **Suggested Fix**: Add `exc_info=True` to all logger.warning calls in _init_modules.

### M-9: Bare Exception in File Lock Cleanup
- **File**: `src/scanner/automation/safe_json.py` ~lines 53-54
- **Description**: `except Exception: pass` in fcntl unlock. Lock release failures are hidden.
- **Impact**: File locks may not be properly released, leading to deadlocks.
- **Suggested Fix**: `except OSError as e: logger.debug(f"Failed to unlock: {e}")`

### M-10: Unguarded .lower() on Potentially None Attribute
- **File**: `buddy_scanner.py` ~lines 51-53
- **Description**: `analysis.volatility_regime` accessed directly without None guard, then `.lower()` called.
- **Impact**: AttributeError crash if volatility_regime is None.
- **Suggested Fix**: `regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").lower()`

### M-11: Unguarded None Comparison in Display
- **File**: `src/scanner/display.py` ~line 403
- **Description**: `analysis.current_price > 0.0001` fails with TypeError if current_price is None.
- **Impact**: Display crashes when rendering table with incomplete PairAnalysis.
- **Suggested Fix**: `has_partial = (analysis.current_price or 0) > 0.0001`

### M-12: Unvalidated OANDA API Response Format
- **File**: `main.py` ~lines 233-240
- **Description**: `c.get("mid")` may return None, then `None.get("o")` raises AttributeError.
- **Impact**: Malformed OANDA response crashes training command.
- **Suggested Fix**: `mid = c.get("mid") or {}; if not isinstance(mid, dict): continue`

### M-13: Version Field Missing in State Persistence
- **File**: `src/recursive_intelligence/persistence.py` ~line 66
- **Description**: State file doesn't include a `version` field for forward compatibility.
- **Impact**: Future schema changes can't be migrated gracefully.
- **Suggested Fix**: Add `state["version"] = 1` before save.

### M-14: No Timestamp Validation on State Load
- **File**: `src/recursive_intelligence/persistence.py` ~lines 50-62
- **Description**: State loading doesn't validate freshness (should warn if stale > 1 hour per rules).
- **Impact**: Stale state from crashes can lead to incorrect trading assumptions.
- **Suggested Fix**: Check `last_updated` timestamp on load and warn if stale.

### M-15: State File Not Flushed Before Session End
- **File**: `src/scanner/automation/state_engine.py`
- **Description**: No explicit shutdown hook to call `save_state()`. Trading rules require flushing state before shutdown.
- **Impact**: Session state lost on crash (scan cycle count, portfolio snapshot).
- **Suggested Fix**: Register atexit handler or ensure orchestrator calls save_state() in finally block.

### M-16: Silent ImportError Fallback to Unsafe JSON Paths
- **File**: `src/scanner/automation/online_rl.py` ~lines 89-94, 151-154, 239-242
- **Description**: Multiple places silently fall back to non-atomic JSON operations when safe_json import fails.
- **Impact**: System degrades to unsafe file I/O without any warning in logs.
- **Suggested Fix**: Log warning on ImportError: `except ImportError: logger.warning("safe_json unavailable")`

### M-17: Learning Rate Not Bounds-Checked in OnlineWeightUpdater
- **File**: `src/scanner/automation/online_rl.py` ~lines 285-289
- **Description**: Learning rate from scheduler assigned directly without bounds validation.
- **Impact**: Adaptive scheduler could return extreme values, causing runaway weight updates.
- **Suggested Fix**: Clamp: `_lr = max(0.001, min(0.1, _lr))`

---

## Low Priority / Code Quality

### L-1: Inconsistent JSON Formatting in safe_json
- **File**: `src/scanner/automation/safe_json.py` ~line 195
- **Description**: Missing `sort_keys=True` in `json.dumps()` per project JSON Safety Gates rule.
- **Suggested Improvement**: Add `sort_keys=True` for consistent, diff-friendly output.

### L-2: Bare Exception in _safe_float Utility
- **File**: `src/scanner/agents/_team.py` ~lines 38-39
- **Description**: `_safe_float()` catches `Exception` without logging.
- **Suggested Improvement**: Log at debug level for traceability.

### L-3: Learnings.md Consolidation Not Implemented
- **File**: `src/scanner/automation/learning_engine.py`
- **Description**: `consolidate()` method exists but never enforces archival of entries > 30 days.
- **Suggested Improvement**: Implement consolidation per improvement rules (archive when > 30 entries).

### L-4: Numerical Stability in Confidence Calculation
- **File**: `src/scanner/agents/_team.py` ~lines 1369-1370
- **Description**: Fixed denominators (0.10, 0.20) with unbounded numerators could produce large intermediates before clipping.
- **Suggested Improvement**: Add bounds check before division.

### L-5: Overly Broad Exception Handling in Adaptive Exits
- **File**: `src/scanner/adaptive_exits.py` ~lines 293-309
- **Description**: Single try/except wraps ALL strategy evaluations. Any single failure forces EXIT_FULL for all trades.
- **Suggested Improvement**: Wrap individual strategy calls in their own try/except.

### L-6: Silent Field Fallback in results.py
- **File**: `src/scanner/results.py` ~lines 171-172
- **Description**: Legacy/current field fallback logic operates silently without debug logging.
- **Suggested Improvement**: Add debug logging to track which fields are being used.

### L-7: Empty String Handling Inconsistency in Regime Gates
- **File**: `src/scanner/regime_gates.py` ~line 190
- **Description**: Empty string `""` returns None instead of mapping to NORMAL profile.
- **Suggested Improvement**: Document behavior or map empty string to NORMAL.

### L-8: Potential IndexError in regime_detector BOCPD
- **File**: `src/scanner/regime_detector.py` ~lines 340-341
- **Description**: `run_length_dist[0]` accessed without explicit bounds check (protected by early return but fragile).
- **Suggested Improvement**: Add explicit guard for defensive coding.

### L-9: Chandelier Period Edge Case
- **File**: `src/scanner/adaptive_exits.py` ~lines 434-436
- **Description**: If `len(ctx.prices_high)` is 0, period becomes 0, and `[-0:]` returns entire array.
- **Suggested Improvement**: `period = max(1, min(self.config.chandelier_period, len(ctx.prices_high)))`

### L-10: ConfigTuner Bounds Not Re-Validated After Apply
- **File**: `src/scanner/automation/config_tuner.py`
- **Description**: Rules applied to config fields without re-clamping to ensure bounds are respected.
- **Suggested Improvement**: Always re-clamp: `new = max(min_val, min(max_val, new))`

### L-11: Position Manager Decision Log Duplication Risk
- **File**: `src/scanner/automation/position_manager.py` ~lines 241-244
- **Description**: `_decisions[-50:]` distribution can be inconsistent if trim happens between checks.
- **Suggested Improvement**: Compute distribution from a snapshot, not live list.

---

## Recurring Patterns

### 1. Non-Atomic JSON Writes (7 instances)
Files: state_engine.py, weight_learner.py, persistence.py, execution.py (fallback), safe_json.py (recovery path)
**Pattern**: Direct `.write_text()` instead of temp-file + `os.rename()`. Project rules mandate atomic writes for all JSON persistence.

### 2. Bare `except Exception:` Without Logging (60+ instances)
Files: engine.py (10), orchestrator.py (15), config_tuner.py (4), agents/_team.py (4), gates.py (2), many automation files
**Pattern**: Exceptions caught and silently swallowed or logged without stack traces. Violates "Silent Exception Prevention" rule.

### 3. Missing Config Validation at Load Time (3+ instances)
Files: fx_guardrails.py, config_tuner.py, adaptive_position_sizing.py
**Pattern**: Numeric config values loaded without range checks. Zero or negative values can cause division-by-zero in financial calculations.

### 4. Missing File Locking on Shared Resources (3 instances)
Files: fx_guardrails.py, state_engine.py, weight_learner.py
**Pattern**: JSON files read/written without fcntl locking despite being accessed by potentially concurrent processes. Only safe_json.py and execution.py consistently use locking.

### 5. Division-by-Zero Risks in Financial Calculations (4 instances)
Files: learning_engine.py (2), adaptive_position_sizing.py (1), agents/_team.py (1)
**Pattern**: Denominators from dynamic calculations (counts, averages, config values) used without zero-guards.

---

## File Complexity Warning

| File | Lines | Concern |
|------|-------|---------|
| src/scanner/engine.py | 3,807 | Extremely large; multiple gate bypass paths are hard to audit |
| src/scanner/execution.py | 3,448 | Very large; interleaves OANDA API, journaling, and RL sync |
| src/scanner/agents/_team.py | 2,520 | Large; 12 agent implementations in one file |
| src/scanner/automation/orchestrator.py | 1,228 | Large; 15+ module lazy initialization paths |

These files should be candidates for modular decomposition in future phases.

---

## Comparison to Previous Report

No previous bug reports found in `bug_reports/` directory. This is the baseline scan.

---

## Static Analysis Summary

- **Syntax check**: All core files (engine.py, execution.py, gates.py, config.py, _team.py, main.py, buddy_scanner.py) pass `py_compile` without errors.
- **Bare except clauses**: 60+ instances of `except Exception:` found across `src/scanner/`, many without logging.
- **Hardcoded secrets**: None found. All API keys properly sourced from environment variables.
- **TODO/FIXME/HACK**: 1 instance of "REWARD HACK ALERT" comment in regime_reward.py (line 243) — appears to be an intentional alert label, not a known bug.

---

*Report generated by automated bug scan — 2026-03-24 10:04 UTC*
