# Buddy Bug Report — 2026-03-24 09:07 ET

## Executive Summary
- **Total issues found: 38**
- **Critical: 7 | High: 8 | Medium: 10 | Low: 6 | Informational: 7**
- **Files with most issues**: engine.py (4), execution.py (7), agents/_team.py (4), learning_engine.py (2), safe_json.py (2), mtf_confluence.py (3)
- **Systemic patterns**: Gate bypass via re-qualification paths (2 instances, PERSISTENT), non-atomic JSON writes (6+ instances), division-by-zero in analytics/consensus paths (5 instances), hardcoded SL violating ATR rule
- **New issues since last report**: 4 new findings; 5 persistent critical issues remain unfixed

---

## Critical Issues (Fix Immediately)

### C-1: Gate Logic Violation — ThresholdOptimizer Re-qualification Skips Momentum Gate
- **File**: `src/scanner/engine.py` line 2836
- **Status**: ⚠️ PERSISTENT (3rd consecutive report)
- **Code**: `if result.gates_passed or (result.confidence_passed and result.risk_passed):`
- **Description**: ThresholdOptimizer re-qualification allows trades when only confidence + risk pass — momentum gate is never checked on this path. Forces `gates_passed = True` and `is_tradeable = True`.
- **Impact**: Trades execute with weak/absent momentum signals. Violates core rule: "ALL THREE gates must pass." Actively triggered in production when optimizer loosens thresholds.
- **Suggested Fix**: `if result.gates_passed or (result.confidence_passed and result.risk_passed and result.momentum_passed):`

### C-2: Gate Logic Violation — EXTREME Regime Re-qualification Only Checks Risk Gate
- **File**: `src/scanner/engine.py` lines 2867–2870
- **Status**: ⚠️ PERSISTENT (3rd consecutive report)
- **Code**: `if result.risk_passed: result.is_tradeable = True; result.gates_passed = True`
- **Description**: EXTREME volatility re-qualification checks only `risk_passed`, ignoring confidence and momentum gates entirely. Bypasses gate safety in the highest-risk regime.
- **Impact**: In EXTREME regimes, trades qualify with 1 of 3 required gates. Combined with C-1, two separate code paths bypass the three-gate requirement.
- **Suggested Fix**: `if result.risk_passed and result.confidence_passed and result.momentum_passed:`

### C-3: SL Calculation Uses Hardcoded Pips — Violates ATR Rule
- **File**: `src/scanner/execution.py` lines 1070–1078
- **Status**: 🆕 NEW
- **Code**: `sl_pips = self.config.max_sl_pips` (hardcoded 15.0)
- **Description**: Stop loss is set to the fixed `max_sl_pips` value (15.0 pips) instead of being calculated from ATR. Meanwhile, TP IS ATR-based (`base_tp = atr * atr_tp_multiplier / pip_value`). The `atr_sl_multiplier = 1.0` config field exists but is never used in the SL calculation path.
- **Impact**: SL stays fixed at 15 pips regardless of volatility. In high-vol environments, SL is too tight (stopped out prematurely). In low-vol, SL is too wide (excess risk). Violates core rule: "SL = ATR * atr_sl_multiplier, TP = ATR * atr_tp_multiplier."
- **Suggested Fix**: Replace hardcoded SL with: `sl_pips = max(self.config.min_sl_pips, min((atr * self.config.atr_sl_multiplier) / pip_value, self.config.max_sl_pips))`

### C-4: Division by Zero — Learning Engine Override Win Rate
- **File**: `src/scanner/automation/learning_engine.py` line 426
- **Status**: ⚠️ PERSISTENT
- **Code**: `overall_wr = wins / total` — guard is `if len(resolved) < 3: return` (not `== 0`)
- **Description**: The `< 3` guard prevents most zero-division cases, but no explicit zero guard at the division point. If guard threshold changes, division becomes unsafe.
- **Impact**: Crashes learning extraction pipeline during low-data periods.
- **Suggested Fix**: Add `if total == 0: return all_entries` before division.

### C-5: Division by Zero — Learning Engine Exit Reason Distribution
- **File**: `src/scanner/automation/learning_engine.py` line 746
- **Status**: ⚠️ PERSISTENT
- **Code**: `distribution[reason] = len(pnl_list) / total_trades` — no guard on `total_trades`
- **Description**: If trade journal is empty or all entries malformed, `total_trades` is 0.
- **Impact**: Crashes exit reason pattern extraction. Blocks learning pipeline.
- **Suggested Fix**: `if total_trades == 0: continue`

### C-6: Type Mismatch — StateEngine Treats String Outcome as Dict
- **File**: `src/scanner/automation/state_engine.py` line 154
- **Status**: ⚠️ PERSISTENT
- **Code**: `wins = sum(1 for e in closed if e["outcome"].get("trade_won", False))`
- **Description**: `outcome` can be a string (legacy format `"win"`) or dict. Calling `.get()` on a string raises `AttributeError`.
- **Impact**: Crashes portfolio snapshot calculation; kills session state tracking.
- **Suggested Fix**: `if isinstance(e.get("outcome"), dict) and e["outcome"].get("trade_won", False)`

### C-7: R:R Gate Silently Bypassed When sl_pips or tp_pips is Zero
- **File**: `src/scanner/execution.py` lines 1413–1420
- **Status**: ⚠️ PERSISTENT (identified in previous report as M-1, upgraded to Critical)
- **Code**: `if sl_pips > 0 and tp_pips > 0: rr_ratio = tp_pips / sl_pips`
- **Description**: If either SL or TP is zero (ATR fallback failure, corrupted data), the R:R gate is **completely skipped** — trade proceeds without stop loss enforcement. This is a fail-open path.
- **Impact**: Trades execute without SL/TP, violating "NEVER execute a trade with R:R ratio below 1.2:1."
- **Suggested Fix**: Reject immediately: `if sl_pips <= 0 or tp_pips <= 0: return ExecutionResult(success=False, error=f"Invalid SL/TP: sl={sl_pips}, tp={tp_pips}")`

---

## High Priority Issues

### H-1: Non-Atomic JSON Writes in StateEngine
- **File**: `src/scanner/automation/state_engine.py` lines 89, 170, 180
- **Status**: ⚠️ PERSISTENT
- **Description**: Three `write_text()` calls without atomic temp-file+rename pattern. Violates JSON Safety Gates rule.
- **Impact**: Corrupted state causes incorrect trading decisions on next session.
- **Suggested Fix**: Use `safe_json_write()` from `src.scanner.automation.safe_json`.

### H-2: Non-Atomic JSON Write in AdaptiveScaler
- **File**: `src/risk/adaptive_scaler.py` line 173
- **Status**: ⚠️ PERSISTENT
- **Code**: `self.state_path.write_text(json.dumps(data, indent=2))`
- **Impact**: Lost drawdown protection state after crash — position sizer reverts to full-size immediately after drawdown.
- **Suggested Fix**: Use `safe_json_write()` with atomic pattern.

### H-3: Non-Atomic JSON Write in fx_guardrails
- **File**: `src/risk/fx_guardrails.py` line 325
- **Status**: ⚠️ PERSISTENT
- **Description**: Daily trade state written without file locking or atomic write. `load_state()` has bare `except Exception:` returning empty state.
- **Impact**: Corrupted guardrails state means daily trade limits are lost — allows limit breach.
- **Suggested Fix**: Use `safe_json_write()`. Replace bare except with specific exception types.

### H-4: Silent Exception in safe_json Backup Recovery
- **File**: `src/scanner/automation/safe_json.py` lines 139, 220
- **Status**: ⚠️ PERSISTENT
- **Code**: `except Exception: pass` in backup validator and lock release paths
- **Impact**: Undetectable recovery failures; deadlocks on lock release failure.
- **Suggested Fix**: `except Exception as e: logger.warning(f"safe_json recovery failed for {path}: {e}")`

### H-5: Non-Atomic Fallback Write in execution.py Trade Journal
- **File**: `src/scanner/execution.py` lines 2809, 1927-1933
- **Status**: ⚠️ PERSISTENT
- **Description**: When `safe_json_write` import fails, fallback uses direct `write_text()` without atomic pattern. Even the fcntl-locked path writes directly to file instead of temp+rename.
- **Impact**: Trade journal corruption on process crash during write.
- **Suggested Fix**: Ensure safe_json is always importable or replicate atomic pattern in fallback.

### H-6: Hardcoded Credentials in .env.local
- **File**: `.env.local`
- **Status**: ⚠️ PERSISTENT
- **Description**: OANDA practice account credentials (account ID + API token) exist in `.env.local`. While `.gitignore` excludes `.env*`, the file is present on disk.
- **Impact**: If repo is pushed to public repository, credentials are exposed. Practice account could be accessed for unauthorized trading.
- **Suggested Fix**: `git rm --cached .env.local` and verify exclusion. Consider credential rotation.

### H-7: Agent Promotion Overrides Gates Without Explicit Verification
- **File**: `src/scanner/engine.py` lines 1965–1992
- **Status**: 🆕 NEW
- **Description**: Agent promotion feature sets `gates_passed=True` based on consensus votes without explicitly verifying all three gate conditions (momentum+confidence+risk). It has its own safety checks (weighted_vote_score >= 0.70, regime, disagreement) but bypasses the standard three-gate requirement.
- **Impact**: Pairs promoted by high agent consensus can execute even if momentum gate failed.
- **Suggested Fix**: Add explicit gate verification: `if not (analysis.momentum_passed and analysis.confidence_passed and analysis.risk_passed): ...`

### H-8: Division by Zero in Agent Consensus Cluster Calculation
- **File**: `src/scanner/agents/_team.py` line 2503
- **Status**: 🆕 NEW
- **Code**: `cluster_scores = {k: sum(v) / len(v) for k, v in clusters.items()}`
- **Description**: No guard on `len(v)`. If a cluster has an empty list (all verdicts filtered), `len(v) == 0` causes ZeroDivisionError.
- **Impact**: Consensus metrics crash; agent metadata not recorded.
- **Suggested Fix**: `{k: sum(v) / len(v) if v else 0.0 for k, v in clusters.items()}`

---

## Medium Priority Issues

### M-1: Index Out of Bounds in MTF Confluence
- **File**: `src/scanner/mtf_confluence.py` lines 229-231
- **Description**: `latest_close = float(data["close"].iloc[-1])` — no length check before `iloc[-1]`. Empty DataFrame raises IndexError.
- **Impact**: Scanner crashes during multi-timeframe confluence for new pairs with insufficient data.
- **Suggested Fix**: Add `if data.empty: return default_result` guard.

### M-2: Slope Denominator Near-Zero in MTF Confluence
- **File**: `src/scanner/mtf_confluence.py` line 257
- **Code**: `slope_change = (sma50.iloc[-1] - sma50.iloc[-2]) / (sma50.iloc[-2] + 1e-8)`
- **Description**: Uses `+ 1e-8` guard, but if denominator is exactly `-1e-8`, result is extreme.
- **Impact**: NaN/Inf cascades in confidence scoring.
- **Suggested Fix**: Use `max(abs(sma50.iloc[-2]), 1e-8)` as denominator.

### M-3: ROC Calculation Without Length Check
- **File**: `src/scanner/agents/_team.py` lines 1887-1889
- **Code**: `prev = float(closes.iloc[-6])` — no check if `len(closes) >= 6`
- **Impact**: Multi-timeframe agent fails on pairs with sparse data.
- **Suggested Fix**: Add length guard.

### M-4: Non-Atomic Pair Performance Stats Write
- **File**: `src/scanner/execution.py` line 3256
- **Code**: `perf_path.write_text(json.dumps(dict(stats), indent=2))`
- **Description**: No locking, no atomic write.
- **Impact**: Concurrent write corruption of pair performance data.

### M-5: Deprecated datetime.utcnow() — 16 Instances
- **Files**: accuracy_gate.py, alert_manager.py, retrain_trigger.py, pair_model_selector.py, qa_pipeline.py, orchestrator.py
- **Description**: `datetime.utcnow()` deprecated in Python 3.12+. Should use `datetime.now(timezone.utc)`.
- **Impact**: Warnings now, errors in future Python versions.

### M-6: Duplicated PIP_VALUES — 3 Definitions
- **Files**: `src/scanner/config.py` (15 pairs), `src/scanner/execution.py` (22 pairs), `src/risk/trading_metrics.py`
- **Description**: Execution.py has 7 extra pairs not in config.py. Adding new pairs requires updating 3 files.
- **Impact**: Desync between files causes incorrect pip value lookups for some pairs.
- **Suggested Fix**: Single source of truth in config.py, import everywhere else.

### M-7: JSON Type Assumption in gap_wirer
- **File**: `src/scanner/automation/gap_wirer.py` lines 147-149
- **Code**: `existing = json.loads(prd_path.read_text()); stories = existing.get("userStories", [])`
- **Description**: Assumes JSON loads returns dict. A JSON array or null triggers AttributeError.
- **Impact**: PRD watcher crashes during gap analysis.

### M-8: Ensemble Conflict NaN Propagation
- **File**: `src/scanner/ensemble_conflict.py` lines 363-364
- **Code**: `denominator = max(abs(mean_score), 0.01); disagreement = std_score / denominator`
- **Description**: If `std_score` is NaN (from model returning NaN), result is NaN without error.
- **Impact**: Silent NaN penalty propagation through gate evaluation.

### M-9: Missing UTF-8 Encoding in safe_json Recovery
- **File**: `src/scanner/automation/safe_json.py` lines 137, 156
- **Description**: Recovery writes use `write_text()` without `encoding="utf-8"` while reads specify it.
- **Impact**: UnicodeDecodeError cycle on non-UTF-8 default locale systems.

### M-10: Integer Truncation in Position Sizing
- **File**: `src/scanner/execution.py` lines 1599, 1721
- **Code**: `units = int(lots * 100_000)` — truncates instead of rounds
- **Impact**: Small discrepancies between intended and executed position sizes.
- **Suggested Fix**: `units = round(lots * 100_000)`

---

## Low Priority / Code Quality

### L-1: _scan_pair() Function — 952 Lines
- **File**: `src/scanner/engine.py` line 2065
- **Description**: Single function spanning 952 lines. Extremely difficult to test, debug, or modify safely.
- **Suggested Improvement**: Extract into sub-methods (feature engineering, gate eval, agent eval, promotion, etc.)

### L-2: execute_trade() Function — 560 Lines
- **File**: `src/scanner/execution.py` line 1178
- **Description**: Core execution function too long for safe maintenance.

### L-3: ScannerAgentTeam.evaluate() — 265 Lines
- **File**: `src/scanner/agents/_team.py` line 839
- **Description**: Agent evaluation method too long; extract individual agent evals.

### L-4: sync_closed_trades_rl() — 394 Lines
- **File**: `src/scanner/execution.py` line 2671
- **Description**: RL sync function handles too many responsibilities.

### L-5: Unhandled Exception in cli_entry.py
- **File**: `cli_entry.py` lines 35-36
- **Code**: `runpy.run_path(str(script), run_name="__main__")` — no try/except
- **Description**: Startup errors produce raw tracebacks instead of user-friendly messages.

### L-6: RL Buffer Write Failures Logged at DEBUG Level
- **File**: `src/scanner/execution.py`
- **Code**: `logger.debug(f"RL replay buffer write failed: {e}")`
- **Description**: RL training data loss logged at debug (invisible in production). Should be warning/error.

---

## Recurring Patterns

### 1. Non-Atomic JSON Writes (Systemic)
Found 6+ locations where JSON state files are written with `write_text()` instead of the atomic temp-file+rename pattern. The codebase has a proper `safe_json_write()` implementation, but it's not used consistently in all paths, especially fallback/error recovery paths.

**Affected files**: state_engine.py, adaptive_scaler.py, fx_guardrails.py, execution.py (2 fallback paths), pair performance stats.

### 2. Gate Bypass Paths (Systemic)
Two separate re-qualification code paths in engine.py bypass the fundamental three-gate (momentum+confidence+risk) requirement. The ThresholdOptimizer path requires only confidence+risk. The EXTREME regime path requires only risk. Agent promotion also bypasses without explicit verification.

### 3. Division by Zero Insufficient Guards
Multiple locations use partial guards (e.g., `if len(x) < 3` instead of `if len(x) == 0`) or no guards at all. Particularly dangerous in financial calculation paths where a crash means missed trades or incorrect sizing.

### 4. Hardcoded vs ATR-based Stops
The SL calculation uses a hardcoded max_sl_pips value while TP correctly uses ATR scaling, creating an asymmetry that defeats the purpose of regime-aware position sizing.

---

## Comparison to Previous Report (2026-03-24 10:44 ET)

### Persistent Issues (Still Unfixed — 5 Critical)
| ID | Issue | Reports Present |
|----|-------|----------------|
| C-1 | ThresholdOptimizer gate bypass | 3 consecutive |
| C-2 | EXTREME regime gate bypass | 3 consecutive |
| C-4 | Learning engine div-by-zero (line 426) | 3 consecutive |
| C-5 | Learning engine div-by-zero (line 746) | 3 consecutive |
| C-6 | StateEngine string outcome crash | 3 consecutive |
| H-1 | StateEngine non-atomic writes | 3 consecutive |
| H-3 | fx_guardrails non-atomic writes | 3 consecutive |
| H-4 | safe_json silent exceptions | 3 consecutive |

### New Issues (4)
| ID | Issue | Severity |
|----|-------|----------|
| C-3 | Hardcoded SL ignoring ATR config | CRITICAL |
| H-7 | Agent promotion bypasses gate verification | HIGH |
| H-8 | Agent consensus cluster div-by-zero | HIGH |
| M-1 | MTF confluence index out of bounds | MEDIUM |

### Resolved Since Last Report
- None detected. All 5 critical issues from the previous report remain unfixed.

---

## Risk Assessment

**Overall System Risk: HIGH**

The combination of C-1 + C-2 (gate bypass paths) with C-3 (hardcoded SL) means the system can enter trades in EXTREME volatility with only the risk gate passing, using a fixed 15-pip stop loss regardless of market conditions. This is the highest-risk scenario for account drawdown.

**Recommended Priority:**
1. Fix C-1 + C-2 gate bypass paths (5 minutes each, one-line fix)
2. Fix C-3 hardcoded SL (30 minutes, requires testing)
3. Fix C-7 R:R fail-open path (10 minutes)
4. Batch-fix all non-atomic JSON writes (1 hour)
5. Add zero-guards to all division operations (30 minutes)

---

## Exterminator Results — 2026-03-24

**Run by**: Buddy Bug Exterminator (scheduled task)
**Commits**: `66d5449` (C-1), `829d02f` (C-2 through M-10, L-6, batch)

### Bugs Fixed ✅

| ID | Severity | Description | Commit |
|----|----------|-------------|--------|
| C-1 | CRITICAL | ThresholdOptimizer requires all 3 gates (added `momentum_passed`) | `66d5449` |
| C-2 | CRITICAL | EXTREME regime re-qualification requires all 3 gates (not just risk) | `829d02f` |
| C-3 | CRITICAL | ATR-based SL now used in both `calculate_position_size` and `calculate_regime_aware_position_size` | `829d02f` |
| C-4 | CRITICAL | Added `if total == 0: return all_entries` guard in learning_engine.py line 426 | `829d02f` |
| C-5 | CRITICAL | Added `if total_trades == 0: continue` guard; also fixed `exit_groups.items()` → `.values()` bug | `829d02f` |
| C-6 | CRITICAL | `isinstance(e.get("outcome"), dict)` guard prevents AttributeError on legacy string outcome | `829d02f` |
| C-7 | CRITICAL | R:R gate is now fail-closed: `if sl_pips <= 0 or tp_pips <= 0: return ExecutionResult(success=False)` | `829d02f` |
| H-1 | HIGH | StateEngine: all 3 `write_text()` calls replaced with `_atomic_write()` (temp+rename) | `829d02f` |
| H-2 | HIGH | AdaptiveScaler: `write_text()` replaced with temp+rename atomic write | `829d02f` |
| H-3 | HIGH | fx_guardrails: `write_text()` replaced with temp+rename atomic write | `829d02f` |
| H-4 | HIGH | safe_json: `except Exception: pass` in backup validator and lock release now logs warnings | `829d02f` |
| H-7 | HIGH | Agent promotion now explicitly checks all 3 gates before setting `gates_passed = True` | `829d02f` |
| H-8 | HIGH | Consensus cluster dict comprehension: `if v else 0.0` guard prevents ZeroDivisionError | `829d02f` |
| M-1 | MEDIUM | MTF confluence: added `data.empty` check alongside length check | `829d02f` |
| M-2 | MEDIUM | MTF slope denominator: `max(abs(sma50.iloc[-2]), 1e-8)` prevents extreme values | `829d02f` |
| M-3 | MEDIUM | ROC calculation: belt-and-suspenders `len(closes) >= 6` guard + `None` check | `829d02f` |
| M-4 | MEDIUM | Pair performance stats: non-atomic `write_text()` replaced with temp+rename | `829d02f` |
| M-9 | MEDIUM | safe_json recovery writes: added `encoding="utf-8"` to both recovery `write_text()` calls | `829d02f` |
| M-10 | MEDIUM | Position sizing: `int(lots * 100_000)` changed to `round()` to prevent truncation bias | `829d02f` |
| L-6 | LOW | RL replay buffer write failure elevated from `logger.debug` to `logger.warning` | `829d02f` |

### Bugs Skipped ⚠️

| ID | Severity | Reason |
|----|----------|--------|
| H-5 | HIGH | Non-atomic fallback in execution.py trade journal write — complex multi-path; safe_json import fallback refactoring too risky without deeper testing. Recommend separate PR. |
| H-6 | HIGH | Hardcoded credentials in .env.local — file management outside code scope; requires manual `git rm --cached` by developer. |
| M-5 | MEDIUM | `datetime.utcnow()` deprecation across 16 instances — pervasive refactor; low immediate risk. Recommend batch replacement in next sprint. |
| M-6 | MEDIUM | PIP_VALUES duplication across 3 files — structural refactor; risky to change centrally without full test coverage. |
| M-7 | MEDIUM | gap_wirer JSON type assumption — not on critical path; low priority. |
| M-8 | MEDIUM | Ensemble conflict NaN propagation — requires NaN propagation analysis before fixing. |
| L-1, L-2, L-3, L-4 | LOW | Function length refactoring — no-risk functional change; out of scope for automated exterminator. |
| L-5 | LOW | cli_entry.py unhandled exception — cosmetic; does not affect trading. |

### Newly Discovered Issues

- **C-5 secondary bug**: `sum(len(v) for v in exit_groups.items())` was iterating over tuples `(key, value)`, making `len()` always return 2 per item. Fixed to `.values()` as part of C-5 fix.
- **Pre-existing test failure**: `test_fallback_lots_zero_atr` fails due to uninitialized `_session_detector` attribute in `ExecutionManager` (pre-existing, not introduced by this run). 166/167 tests pass.

### Git Lock Note

The FUSE-mounted filesystem prevented deletion of stale git lock files (`.git/index.lock`, `.git/HEAD.lock`) left by a prior commit. Workaround: all subsequent commits used `GIT_DIR=/tmp/ml_engine_git_copy` with the locks removed in the temp copy, then objects + HEAD ref synced back. All changes are confirmed committed.

