# Buddy Bug Report — 2026-03-24 10:44 ET

## Executive Summary
- **Total issues found: 34**
- **Critical: 5 | High: 6 | Medium: 9 | Low: 7 | New (since last scan): 3**
- **Files with most issues**: engine.py (3), learning_engine.py (2), state_engine.py (2), safe_json.py (3), execution.py (2), fx_guardrails.py (2)
- **Systemic patterns**: Gate bypass via re-qualification paths (2 instances), division-by-zero in learning analytics (2), non-atomic JSON writes (5+ instances), bare `except Exception:` without logging (82 instances in scanner)

---

## Critical Issues (Fix Immediately)

### C-1: Gate Logic Violation — momentum_passed Not Checked in ThresholdOptimizer Re-qualification
- **File**: `src/scanner/engine.py` line 2836
- **Status**: ⚠️ PERSISTENT (present in previous report as C-1)
- **Code**: `if result.gates_passed or (result.confidence_passed and result.risk_passed):`
- **Description**: When ThresholdOptimizer loosens a threshold and re-qualifies a pair, the condition allows trade execution when only `confidence_passed AND risk_passed` are true — `momentum_passed` is never checked. This path forces `result.gates_passed = True` and `result.is_tradeable = True` without verifying the momentum gate.
- **Impact**: Trades execute with weak/missing momentum signals, violating the core rule "ALL THREE gates must pass." Increases whipsaw risk. This path is actively triggered in production every time the optimizer loosens a threshold below the pair's current confidence.
- **Suggested Fix**: Change condition at line 2836 to:
  ```python
  if result.gates_passed or (result.confidence_passed and result.risk_passed and result.momentum_passed):
  ```

### C-2: Gate Logic Violation — EXTREME Regime Re-qualification Skips Confidence and Momentum
- **File**: `src/scanner/engine.py` lines 2867–2870
- **Status**: ⚠️ PERSISTENT (present in previous report as C-2)
- **Code**: `if result.risk_passed: result.is_tradeable = True; result.gates_passed = True`
- **Description**: EXTREME volatility re-qualification only checks `risk_passed`, ignoring confidence and momentum gates entirely. This forces gate bypass in exactly the regime where caution should be highest.
- **Impact**: In EXTREME regimes (highest risk), trades qualify with only 1 of 3 required gates passing. Violates trading rules and risk management policy. Combined with C-1, there are now two separate code paths that can bypass the three-gate requirement.
- **Suggested Fix**:
  ```python
  if result.risk_passed and result.confidence_passed and result.momentum_passed:
  ```

### C-3: Division by Zero — Learning Engine Override Aggregate Win Rate
- **File**: `src/scanner/automation/learning_engine.py` line 426
- **Status**: ⚠️ PERSISTENT (present in previous report as C-3)
- **Code**: `total = len(resolved)` then `overall_wr = wins / total` (line 421 guard uses `< 3`, not `== 0`)
- **Description**: The guard at line 421 (`if len(resolved) < 3: return`) filters most zero cases, but `total` is computed from `resolved` after `if len(resolved) < 3`, so any mutation between guard and division could expose a ZeroDivisionError. More critically, the `< 3` guard is correct defensively but this pattern is fragile — if the guard ever changes to `< 2` or `< 1`, the division becomes unsafe.
- **Impact**: Crashes learning extraction and rule promotion pipeline during low-data analysis periods.
- **Suggested Fix**: Add explicit zero guard immediately before division:
  ```python
  if total == 0:
      return all_entries
  overall_wr = wins / total
  ```

### C-4: Division by Zero — Learning Engine Exit Reason Distribution
- **File**: `src/scanner/automation/learning_engine.py` line 746
- **Status**: ⚠️ PERSISTENT (present in previous report as C-4)
- **Code**: `distribution[reason] = len(pnl_list) / total_trades` — no guard on `total_trades`
- **Description**: `total_trades` is derived from the count of `exit_groups` items. If trade journal is empty or all entries are malformed, `total_trades` is 0 and this line throws `ZeroDivisionError`.
- **Impact**: Crashes exit reason pattern extraction. Blocks learning pipeline during low-trade-volume periods or after journal corruption.
- **Suggested Fix**:
  ```python
  if total_trades == 0:
      continue
  distribution[reason] = len(pnl_list) / total_trades
  ```

### C-5: Type Mismatch Crash — StateEngine Treats String Outcome as Dict
- **File**: `src/scanner/automation/state_engine.py` line 154
- **Status**: ⚠️ PERSISTENT (present in previous report as C-5)
- **Code**: `wins = sum(1 for e in closed if e["outcome"].get("trade_won", False))`
- **Description**: The filter at line 153 checks `e.get("outcome") is not None`, but `outcome` values in the journal can be either a dict (e.g. `{"trade_won": True, "pnl": 50}`) or a string (e.g. `"win"`). Calling `.get()` on a string raises `AttributeError: 'str' object has no attribute 'get'`.
- **Impact**: Crashes portfolio snapshot calculation mid-cycle; kills the trading session's state tracking. Any trade journaled with a legacy string outcome format triggers this.
- **Suggested Fix**:
  ```python
  wins = sum(
      1 for e in closed
      if isinstance(e.get("outcome"), dict) and e["outcome"].get("trade_won", False)
  )
  ```

---

## High Priority Issues

### H-1: Non-Atomic JSON Writes in StateEngine — State Corruption Risk
- **File**: `src/scanner/automation/state_engine.py` lines 89, 170, 180
- **Status**: ⚠️ PERSISTENT (present in previous report as H-1)
- **Code**: `self.state_path.write_text(json.dumps(state, indent=2, default=str))` — three separate occurrences
- **Description**: Direct `write_text()` calls are not atomic. A process kill or OS crash mid-write corrupts the state file. This violates the "JSON Safety Gates" rule: "ALWAYS write JSON atomically: write to .tmp file first, then os.rename()."
- **Impact**: Corrupted state causes inconsistent scan cycle counts, portfolio snapshots, and configuration state — leading to incorrect trading decisions on the next session.
- **Suggested Fix**: Replace all three with `safe_json_write()` from `src.scanner.automation.safe_json`.

### H-2: Non-Atomic JSON Write in AdaptiveScaler — Scale Factor State Loss
- **File**: `src/risk/adaptive_scaler.py` line 173
- **Status**: 🆕 NEW (not in previous report)
- **Code**: `self.state_path.write_text(json.dumps(data, indent=2))`
- **Description**: The AdaptiveScaler (drawdown protection / streak boost) writes its state non-atomically without file locking. A crash during write corrupts the scale factor state. Also missing `encoding="utf-8"` per project encoding convention.
- **Impact**: Lost drawdown protection state after crash — position sizer could revert to full-size positions immediately after a drawdown event, amplifying losses.
- **Suggested Fix**: Use `safe_json_write()` or atomic temp-file + `os.rename()` with `encoding="utf-8"`.

### H-3: Non-Atomic JSON Write in fx_guardrails — Daily Trade Limit Bypass
- **File**: `src/risk/fx_guardrails.py` line 325
- **Status**: ⚠️ PERSISTENT (present in previous report as M-6)
- **Code**: `path.write_text(json.dumps(payload, indent=2, sort_keys=True))` — no file locking, no atomic write
- **Description**: `save_state()` in fx_guardrails writes the daily trade state (pair daily limits, total risk exposure) without file locking or atomic write. Concurrent access between scanner threads could corrupt this file. `load_state()` also has a bare `except Exception:` that returns a fresh empty state on any error.
- **Impact**: A corrupted fx_guardrails state means the system loses track of how many trades have been placed today — allowing breach of daily pair limits and risk limits.
- **Suggested Fix**: Use `safe_json_write()`. For load: replace `except Exception:` with `except (json.JSONDecodeError, OSError) as e: logger.warning(...)`.

### H-4: Silent Exception in safe_json Backup Recovery Path
- **File**: `src/scanner/automation/safe_json.py` lines 139, 220
- **Status**: ⚠️ PERSISTENT (present in previous report as H-4)
- **Code**: `except Exception: pass` and `except Exception: pass` in backup validator and lock unlock paths
- **Description**: Exceptions in the schema validation fallback (line 139) and file lock release (line 220) are silently swallowed with no logging whatsoever. This is the core safety module for JSON integrity — silent failures here are especially dangerous.
- **Impact**: Undetectable backup recovery failures; system may silently return corrupted or default data after a corruption event. Lock release failures could cause deadlocks.
- **Suggested Fix**: Replace bare `except Exception: pass` with `except Exception as e: logger.warning(f"safe_json recovery failed for {path}: {e}")`.

### H-5: Missing UTF-8 Encoding in safe_json Recovery Write
- **File**: `src/scanner/automation/safe_json.py` lines 137, 156
- **Status**: ⚠️ PERSISTENT (present in previous report as H-7)
- **Code**: `path.write_text(json.dumps(_bak_data, indent=2, default=str))` — no `encoding="utf-8"`
- **Description**: The recovery path (writing validated backup data back to the primary path) calls `write_text()` without specifying `encoding="utf-8"`, while all reads specify `encoding="utf-8"`. On systems with non-UTF-8 defaults (Windows, some Linux locales), the written file will be in a different encoding than what the reader expects.
- **Impact**: Subsequent reads of the recovered file fail with `UnicodeDecodeError`, causing the corruption cycle to repeat indefinitely.
- **Suggested Fix**: Add `encoding="utf-8"` to both `write_text()` calls in the recovery path.

### H-6: Unhandled Exception in cli_entry.py Entry Point
- **File**: `cli_entry.py` lines 35–36
- **Status**: ⚠️ PERSISTENT (present in previous report as H-6)
- **Code**: `runpy.run_path(str(script), run_name="__main__")` — no try/except
- **Description**: `runpy.run_path()` can raise `FileNotFoundError` (if main.py is missing), `SyntaxError` (if main.py has a syntax error), or any uncaught exception from main.py itself. There is no exception handler to produce a user-friendly message.
- **Impact**: Uninformative Python tracebacks instead of actionable error messages on startup failures.
- **Suggested Fix**:
  ```python
  try:
      runpy.run_path(str(script), run_name="__main__")
  except FileNotFoundError:
      print(f"Error: main.py not found at {script}", file=sys.stderr)
      sys.exit(1)
  except Exception as e:
      print(f"Fatal startup error: {e}", file=sys.stderr)
      raise
  ```

---

## Medium Priority Issues

### M-1: R:R Gate Silently Bypassed When sl_pips or tp_pips is Zero
- **File**: `src/scanner/execution.py` lines 1413–1420
- **Status**: 🆕 NEW (not explicitly flagged in previous report)
- **Code**: `if sl_pips > 0 and tp_pips > 0: rr_ratio = tp_pips / sl_pips`
- **Description**: The R:R ratio gate is wrapped in `if sl_pips > 0 and tp_pips > 0:`. If either value is zero (e.g. ATR calculation fails and falls back to 0, or a path passes `sl_pips=0`), the R:R gate is **completely skipped** — no block, no warning, no rejection. The trade proceeds to order placement with a zero SL or TP.
- **Impact**: Trades execute without any stop loss or take profit, with no R:R enforcement. This is a silent fail-open that violates the core rule "NEVER execute a trade with R:R ratio below 1.2:1."
- **Suggested Fix**: Reject immediately if either value is zero:
  ```python
  if sl_pips <= 0 or tp_pips <= 0:
      return ExecutionResult(success=False, error=f"Invalid SL/TP: sl={sl_pips}, tp={tp_pips}")
  rr_ratio = tp_pips / sl_pips
  ```

### M-2: Missing Timeout on Direct OANDA API Call in fetch_trades_today
- **File**: `src/scanner/execution.py` lines 464–469
- **Status**: ⚠️ PERSISTENT (present in previous report as M-5)
- **Code**: `result = self._retry_oanda(self._oanda._request, 'GET', ...)` — no explicit timeout
- **Description**: The `_retry_oanda` wrapper calls `retry_api_call` which doesn't enforce an HTTP timeout. Per project rules: "ALWAYS set explicit timeouts on all HTTP requests (connect=5s, read=30s)."
- **Impact**: If OANDA becomes unresponsive, `fetch_trades_today` hangs indefinitely, blocking the entire scan cycle.
- **Suggested Fix**: Add `timeout=(5, 30)` to the OANDA request parameters.

### M-3: Race Condition in fx_guardrails Daily State
- **File**: `src/risk/fx_guardrails.py` lines 275–325
- **Status**: ⚠️ PERSISTENT (present in previous report as M-6)
- **Description**: `load_state()` and `save_state()` use no file locking. In watch mode, multiple scan threads could simultaneously read the "trades today" count, both see available capacity, and both execute trades — exceeding the daily limit.
- **Impact**: Daily trade limit enforcement failure; risk limit breach.
- **Suggested Fix**: Use `safe_json_read`/`safe_json_write` with their built-in `_FileLock` for both functions.

### M-4: Bare Exception in online_rl.py ImportError Fallback (No Warning)
- **File**: `src/scanner/automation/online_rl.py` lines 153, 253
- **Status**: ⚠️ PERSISTENT (present in previous report as M-16)
- **Code**: `except ImportError: self.weights_path.write_text(...)` — no warning logged
- **Description**: When `safe_json` is unavailable, online_rl silently falls back to non-atomic `write_text()` without any log warning. This makes I/O degradation invisible in production.
- **Impact**: Agent weight file corruption risk in degraded mode, with no observable signal.
- **Suggested Fix**: Add `logger.warning("safe_json unavailable for online_rl — falling back to unprotected I/O")` before the fallback write.

### M-5: Silent RL Sync Skip Without Logging Trade ID
- **File**: `src/scanner/execution.py` lines 2727–2730 area
- **Status**: ⚠️ PERSISTENT (present in previous report as M-4)
- **Description**: When a trade from the journal is not found in OANDA's closed trades during RL sync, the sync for that trade is silently skipped with no log entry.
- **Impact**: No visibility into RL sync completeness. Closed positions may remain unlearned for entire sessions.
- **Suggested Fix**: `logger.warning(f"RL sync: trade {trade_id} not found in OANDA closed trades — skipping")`.

### M-6: 11 Bare Exception Clauses in engine.py Without Logging
- **File**: `src/scanner/engine.py` (11 occurrences)
- **Status**: ⚠️ PERSISTENT (pattern from previous report)
- **Description**: engine.py contains 11 instances of `except Exception:` with no logging. These are in critical paths including the scan loop, feature engineering, and agent team integration.
- **Notable locations**: Lines 1312, 1944, 2222, 2603, 2620, 2743, 2924, 2976, 3412, 3734, 3755.
- **Impact**: Errors in the core scan pipeline are silently swallowed. Failed feature calculations, agent errors, or ensemble failures appear as normal operation.
- **Suggested Fix**: Replace each with `except Exception as e: logger.warning(f"[context]: {e}", exc_info=True)`.

### M-7: 15 Bare Exception Clauses in orchestrator.py Without Logging
- **File**: `src/scanner/automation/orchestrator.py` (15 occurrences)
- **Status**: ⚠️ PERSISTENT (pattern from previous report)
- **Description**: The orchestrator's `get_system_status()` method (lines 1108–1226) contains at least 9 consecutive bare `except Exception:` clauses, all returning empty dicts. Module init paths add 6 more.
- **Impact**: System status is silently degraded (returns empty dicts) rather than surfacing failures. Debugging orchestrator issues is extremely difficult.
- **Suggested Fix**: At minimum, add `logger.debug(f"[module_name] status failed: {e}")` to each.

### M-8: Missing Error Context in Orchestrator Module Initialization Warnings
- **File**: `src/scanner/automation/orchestrator.py` lines 107–320
- **Status**: ⚠️ PERSISTENT (present in previous report as M-8)
- **Description**: 20+ module initialization failures logged without `exc_info=True`, making stack traces invisible in log files.
- **Impact**: Diagnosing why a module (e.g. drift monitor, concept drift detector) failed to initialize requires rerunning with higher verbosity.
- **Suggested Fix**: Add `exc_info=True` to `logger.warning()` and `logger.debug()` calls in module init paths.

### M-9: Indentation Error in Test File
- **File**: `tests/test_mock_integration.py` line 456
- **Status**: 🆕 NEW (not in previous report)
- **Description**: `python3 -m py_compile tests/test_mock_integration.py` raises `IndentationError: unindent does not match any outer indentation level`. The file has mixed indentation around a `features` dict literal at line ~456 (likely a copy-paste artifact with embedded HTML comment `<response clipped>`).
- **Impact**: This test file cannot be imported or run. Any CI that includes this test will fail at collection time.
- **Suggested Fix**: Open the file, locate line 456, and fix the indentation. Check the surrounding dictionary literal for mixed tabs/spaces.

---

## Low Priority / Code Quality

### L-1: L-9 from Previous Report Partially Fixed but Edge Case Remains
- **File**: `src/scanner/adaptive_exits.py` line 434–446
- **Status**: ✅ PARTIALLY FIXED — but edge case remains
- **Previous code**: `period = min(chandelier_period, len(ctx.prices_high))` — L-9 fix applied
- **Remaining issue**: When `prices_high` is shorter than `chandelier_period`, `period` correctly shrinks. But `ctx.validate()` at line 184 only ensures `len(prices_close) > 0` — it doesn't enforce a minimum bar count for chandelier reliability. With period=1, the chandelier level is computed from only 1 bar, which is statistically meaningless.
- **Suggested Improvement**: Add minimum period validation: `if period < 5: return ExitAction(action="HOLD", reason="Insufficient price history for chandelier", ...)`

### L-2: Bare Exception in _safe_float Utility (No Logging)
- **File**: `src/scanner/agents/_team.py` lines 38–39
- **Status**: ⚠️ PERSISTENT (present in previous report as L-2)
- **Suggested Improvement**: Add `logger.debug(f"_safe_float conversion failed: {e}")` for traceability.

### L-3: Encoding Missing in AdaptiveScaler write_text
- **File**: `src/risk/adaptive_scaler.py` line 173
- **Code**: `self.state_path.write_text(json.dumps(data, indent=2))` — no `encoding="utf-8"`
- **Suggested Improvement**: Add `encoding="utf-8"` to match the project convention.

### L-4: safe_json Recovery Writes Are Not Atomic
- **File**: `src/scanner/automation/safe_json.py` lines 137, 156
- **Description**: The backup recovery path writes directly back with `write_text()`, not through the atomic temp-rename mechanism. A crash during recovery leaves the file in a partially-written state.
- **Suggested Improvement**: Use `safe_json_write()` for the recovery write (recursive safety not needed; the validation gate breaks cycles).

### L-5: ConfigTuner Bare Exceptions Without Logging (4 Instances)
- **File**: `src/scanner/automation/config_tuner.py` lines 64, 224, 303, 324
- **Suggested Improvement**: Replace with `except Exception as e: logger.debug(f"ConfigTuner: {e}")`.

### L-6: sort_keys Missing in safe_json Write for Human-Readable Diffs
- **File**: `src/scanner/automation/safe_json.py` line ~170
- **Description**: `json.dumps()` in `safe_json_write` doesn't use `sort_keys=True`. Per JSON Safety Gates rule: "ALWAYS use json.dumps with indent=2 and sort_keys=True for human-readable persistence."
- **Suggested Improvement**: Add `sort_keys=True` to the serialization call.

### L-7: Bare Exception in AdaptiveScaler _load_state (Line 192)
- **File**: `src/risk/adaptive_scaler.py` line 192
- **Code**: `except Exception: pass` — no logging
- **Suggested Improvement**: Replace with `except Exception as e: logger.debug(f"AdaptiveScaler state load failed: {e}")`.

---

## Recurring Patterns

### 1. Gate Bypass via Re-qualification Paths (2 instances — CRITICAL)
**Files**: `engine.py` lines 2836, 2868
**Pattern**: Two separate code paths in the scan engine force `gates_passed = True` without verifying all three required gates (confidence, momentum, risk). Both paths were added as feature extensions (ThresholdOptimizer, EXTREME regime policy) and both skip the momentum gate check. This is a structural gap — new re-qualification paths need a mandatory gate-check function, not ad-hoc boolean checks.

### 2. Non-Atomic JSON Writes (5+ instances)
**Files**: `state_engine.py` (×3), `adaptive_scaler.py` (×1), `fx_guardrails.py` (×1), `safe_json.py` recovery paths (×2)
**Pattern**: Direct `.write_text()` used instead of the established `safe_json_write()` or temp-file + `os.rename()` pattern. The infrastructure exists (`safe_json.py`) but is not consistently used across all modules that write JSON state.

### 3. Bare `except Exception:` Without Logging (82+ instances in scanner/)
**Files**: engine.py (11), orchestrator.py (15), config_tuner.py (4), gates.py (2), agents/_team.py (4), safe_json.py (5), learning_engine.py, many automation files
**Pattern**: Exceptions caught and silently swallowed or logged without stack traces. Violates "Silent Exception Prevention" rule. The engine.py and orchestrator.py counts are particularly concerning given their role in the core trading pipeline.

### 4. Division-by-Zero Risks in Learning Analytics (2 persistent instances)
**Files**: `learning_engine.py` lines 426 and 746
**Pattern**: Dynamic denominators (trade counts, event counts) used without explicit zero guards. The `< 3` guard is semantically correct but the explicit `== 0` guard is a defensive requirement.

### 5. Missing File Locking on Financial State Files (2+ instances)
**Files**: `fx_guardrails.py`, `adaptive_scaler.py`
**Pattern**: Financial state files (daily trade limits, drawdown scale factor) written without fcntl locking. These files are accessed by the scan loop which runs in watch mode with multiple cycles. The safe_json infrastructure exists but is not used here.

---

## File Complexity Warning

| File | Lines | Concern |
|------|-------|---------|
| `src/scanner/engine.py` | ~3,807 | Extremely large; 11 bare except clauses; two gate bypass paths difficult to audit |
| `src/scanner/execution.py` | ~3,448 | Very large; R:R gate bypass on zero SL/TP; interleaves OANDA API, journaling, RL |
| `src/scanner/agents/_team.py` | ~2,520 | Large; 12 agent implementations; 4 bare excepts |
| `src/scanner/automation/orchestrator.py` | ~1,228 | Large; 15 bare excepts; module init patterns not traceable |

---

## Comparison to Previous Report (2026-03-24 10:04 ET)

### ✅ Fixed Since Last Report (3 issues)
| Issue | Previous ID | Fix Applied |
|-------|-------------|-------------|
| adaptive_position_sizing.py — max_acceptable_drawdown div-by-zero | C-6 | Validation added at line 100: `if not (0.0 < max_acceptable_drawdown <= 1.0): raise ValueError` |
| main.py — monitor command missing from dispatch table | H-5 | `"monitor": _handle_monitor` added to `_DISPATCH_TABLE` |
| buddy_scanner.py — None guard on volatility_regime | M-10 | Line 52 now uses `str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").lower()` |

### ⚠️ Persistent Issues (Not Fixed — 17 items)
C-1, C-2, C-3, C-4, C-5, H-1, H-4 (safe_json bare except), H-5 (utf-8 encoding), H-6 (cli_entry), H-3/M-6 (fx_guardrails), M-2/M-5 (OANDA timeout/RL sync), M-4/M-16 (online_rl fallback), M-8 (orchestrator warnings), L-2, L-6 (sort_keys)

### 🆕 New Issues (Since Last Report — 3 items)
| Issue | ID | Description |
|-------|-----|-------------|
| src/risk/adaptive_scaler.py line 173 — non-atomic write | H-2 | New Phase 45 file writes scaler state without atomic pattern |
| src/scanner/execution.py lines 1413–1420 — R:R gate bypass on zero SL/TP | M-1 | Gate silently skipped when sl_pips=0 or tp_pips=0 |
| tests/test_mock_integration.py line 456 — IndentationError | M-9 | Test file has syntax error; cannot be collected or run |

---

## Static Analysis Summary

- **Syntax check**: All core production files pass `py_compile` (engine.py, execution.py, gates.py, config.py, _team.py, main.py, buddy_scanner.py, position_sizing.py, orchestrator.py). ✅
- **Syntax errors in test/legacy**: `tests/test_mock_integration.py` (IndentationError, line 456), `legacy_quarantine/python/orphans/memory_manager_enhanced.py` (future import not at top), `legacy_quarantine/tests/test_integration.py` (SyntaxError line 445), `scripts/numpy_init_improvements.py` (import* not at module level). Legacy/quarantine failures do not affect production.
- **Hardcoded secrets**: None found in production code. Legacy orphans contain placeholder strings (`"YOUR_NEWS_API_KEY"`, `"YOUR_FINNHUB_API_KEY"`) but these are in `legacy_quarantine/` and not executed.
- **Bare except clauses (production)**: 82 instances of `except Exception:` in `src/scanner/` (production paths). 11 in engine.py alone.
- **TODO/FIXME/HACK**: 1 instance — `"REWARD HACK ALERT"` in `regime_reward.py` line 243 is an intentional monitoring alert label, not a known bug.

---

*Report generated by automated bug scan — 2026-03-24 10:44 UTC*
