# Codex Daemon Review — SOTA Activation / Execution Path

**Date:** 2026-06-16
**Branch:** `codex/sota-activation-execution` (uncommitted working tree)
**Runtime context:** `state.json` → `halted=false`, `mode=dry_run`
**Scope reviewed:** US-002/006/009/010/012/013 daemon wiring — tick capture → post-flush
aggregation → autonomous retrain → soak → SOTA promotion → execution path, plus neural-agent
online learning.

> Verification convention (per `.claude/rules/honesty.md`): every claim below names a
> `file:line` source and carries a calibrated confidence tag. Findings were read from disk,
> verdicted against disk, and (where safe) auto-applied with `py_compile` + `flake8` gates.

---

## 1. Summary

24 findings were triaged: **23 confirmed real, 1 refuted** (`safety-0`). Of the 23 real, **13
were auto-applied** (correctness / honesty / robustness fixes that cannot place an order, change
halt, or flip `dry_run`) and **11 are flagged for operator decision** because they touch the live
execution or model-promotion path (one finding, `safety-3`, was partially applied — its defensive
half landed, its incumbent-swap half was deferred).

The headline result: the **SOTA auto-promotion loop is decorative end-to-end** — nothing reads the
soak verdict, nothing flips `use_sota_inference`, and the soak it would read is scoring a momentum
toy against a constant-HOLD stub. The **US-013 dry-run safety feature is non-functional** on both
production drivers (and crashed on construction until `bugs-0` was fixed). The documented **SOTA
rollback is a no-op** against live inference. None of this currently loses money — the system is
fail-closed (halted gate at the order boundary holds; SOTA artifact is absent so inference returns
HOLD) — but the safety controls the daemon advertises do not exist as wired.

---

## 2. APPLIED Fixes (13)

These landed on the working tree. Each passed `py_compile` and introduced zero new `flake8`
errors. None touches order placement, halt, or `dry_run` defaults. Confidence **HIGH** unless noted.

| ID | file:line | What was wrong → fix | Diff note |
|----|-----------|----------------------|-----------|
| `bugs-0` | `src/scanner/engine.py:7509` | Dry-run branch built `ExecutionResult(pair=, direction=, entry_price=, mode=)` — none are dataclass fields (`execution.py:301-336`); `TypeError` on the **first** tradeable pair, crashing the scan cycle. Fix: valid fields only — `entry_price→fill_price`, drop `pair/direction/mode`, add `fill_status="DRY_RUN"`. pair/direction/mode already in the journal entry. | `+2/−4`; new regression test `tests/test_dry_run_execution_result_fields_2026_06_16.py` (no-mock: real `ScannerConfig`, real `ExecutionManager` subclass stubbing only `fetch_live_nav`). Test fails on reverted code. |
| `bugs-5` / `neural-2` | `src/data/tick_post_flush.py:67-73` | Post-flush aggregator passed `root=None` → `_load_ticks` does `None / instrument` → `TypeError` **every flush**, swallowed to WARNING; also full-history rescan on each 30s flush. US-002 "always-fresh data" silently never ran. Fix: `root=TICK_ROOT`; incremental `start/end` from batch min/max tick time; iterate instruments actually present in batch. | Error path demoted from silent WARNING to surfaced `logger.error` + `error_count`/`last_error` counters. |
| `bugs-6` | `src/data/tick_post_flush.py:80-88` | Non-atomic read-modify-write on shared `harvest/*.parquet` (also written non-atomically by `harvest.py:270`) from the tick-capture background thread → torn reads / truncated parquet. Fix: `_atomic_merge_write()` — `fcntl.flock` on `{path}.lock` sentinel + tmp write + `os.replace()`. | **Cross-writer guarantee is one-sided** until `harvest.py:_append` adopts the same lock path — flagged below. |
| `bugs-7` / `safety-5` | `src/scanner/automation/cycle_autonomy.py:140-149` | US-006 soak poll built a fresh `SoakOrchestrator()` each cycle; a throwaway instance has `_process=None`, so `poll()` is permanently `idle` — `running`/`complete` branches dead. Fix: poll the module singleton `autonomous_trainer._get_soak_orchestrator()`; removed the bare `except: pass`. | Observability-only today (sole consumers were two brain-log lines). |
| `bugs-8` / `neural-15` / `neural-7` | `src/scanner/automation/continuous.py:379-401` | Throwaway `TickCaptureDaemon` built only to grab `.persister` (`# will be replaced below`), wrapped in `except: pass`; aggregator hard-coded to `pairs[0]`. Fix: deleted the scaffolding; single daemon; new `_create_multi_pair_post_flush_aggregator()` groups flushed ticks by instrument so every streamed pair aggregates. | Removed 2 pre-existing flake8 errors as a side effect. |
| `bugs-9` / `neural-13` | `src/scanner/agents/neural/neural_agent_base.py:301-318` | EWC Fisher computed against `tf.zeros_like(preds)` — biases the diagonal toward the y=0 class, not a valid Fisher estimate. Fix: target from model's own predicted class `tf.stop_gradient(tf.round(preds))`. | Dormant (`use_ewc=False`), latent if ever enabled. |
| `bugs-10` | `src/scanner/agents/neural/neural_agent_base.py:320-345` | `_apply_ewc_penalty` recompiled the policy every `_online_update`, churning optimizer slot state. Fix: compile EWC loss once (`_ewc_compiled` guard); single optimizer; Fisher/anchor updated in-place via `_ewc_var_holders`. Confidence **MEDIUM** (path reachable via `smart` profile but gated dormant by `use_ewc=False`). | Removed dead `original_loss`. |
| `bugs-12` | `src/sota_core/model_versioning.py:94-109` | `_update_manifest` non-atomic write + bare `except: pass` on read → corrupt manifest silently resets to `{"versions": []}`, dropping version history `rollback_sota.py --list` depends on. Fix: typed `except (JSONDecodeError, OSError)` + WARNING; tmp + `os.replace()`. | `rollback_to_version(<ts>)` still works (reads dir directly); only `--list` discovery was at risk. |
| `bugs-13` | `src/scanner/automation/autonomous_trainer.py:363-372` | Retrain `log_file` handle never closed in parent → fd leak across Mon/Fri+staleness spawns. Fix: `log_file=None` guard + `finally:` close (child already dup'd the fd). | |
| `neural-0` | `src/evaluation/soak_orchestrator.py:54-57` | `_sota_signal_fn` called `pred.get(...)` on a `TradeSignal` **dataclass** → `AttributeError` every bar → SOTA arm permanently HOLD. Fix: `getattr(pred,"direction","HOLD")` / `getattr(pred,"confidence",0.5)`, normalize+validate, widen `except` to WARNING + failure counter. | See §5 — this is the load-bearing soak defect. |
| `neural-1` | `src/scanner/agents/neural/neural_agent_base.py:206-219 + :187` | **Inverted training labels** (highest-severity correctness bug): `evaluate()` never wrote `passed` to metadata → `record_outcome` read `passed=False` always → `target=1.0 if (False==trade_won)` → agent labeled "correct" exactly when the trade **lost**. Fix: write `passed`+`score` at `:187`; if `passed` absent, `logger.error` + skip rather than default-False. | Corrupted the policy whenever the trainer ran, independent of shadow mode. |
| `neural-6` | `src/evaluation/soak_orchestrator.py:48-52` | `SOTAInference()` constructed **per bar** inside the comparison loop → 500-2000 full `load_model()` calls per soak → daemon stuck "running". Fix: build once in `_run_soak_worker`, bind via `functools.partial`. | Cross-file stats-cache half (`raw_sequence_model.py`) deferred — out of single-file scope. |
| `neural-10` | `src/scanner/automation/continuous.py:2505-2508` | Same `TradeSignal`-vs-`dict` `.get()` bug in the SOTA shadow A/B log → `sota_direction` always `None` → shadow comparison data permanently empty. Fix: `getattr`-based read + `trade is False → HOLD` guard + case-normalize. | `_check_model_promotion` (`:2558`) now receives non-empty SOTA records — operator should review that consumer. |
| `neural-11` | `src/scanner/agents/neural/neural_agent_base.py:143-150` | `evaluate()` lazily builds a **random** policy on load failure and predicts from it → can emit `block_trade=True` veto on noise. Fix: `_trained` flag (set only on successful `load()`/`_online_update()`); untrained → `_fallback_verdict(score=0.5, block_trade=False)`. | Live-reachable via `NeuralAgentTeam` even under `shadow_neural_agents=True`. |
| `neural-14` | `src/data/tick_capture.py:203-216` | `ticks_per_minute` only resets on tick arrival → during a stream stall it reports a healthy rate while the stream is dead. Fix: `get_health()` returns `0` when `idle_seconds > 60`. | Monitoring honesty only. |

---

## 3. FLAGGED FOR OPERATOR (11) — touches live / promotion, NOT auto-applied

Per `CLAUDE.md` ("halt > break"; never flip execution behavior without operator consent), these
were **not** changed. Each is a decision with its cost-of-being-wrong stated. All confidence **HIGH**
unless noted.

| ID | sev | file:line | Decision & risk |
|----|-----|-----------|-----------------|
| `bugs-1` / `safety-1` / `neural-3` | **critical** | `embedded_scanner.py:1055`; `continuous.py:85,801-804`; `engine.py:7446` | **Dry-run is dead config + unreachable code on every live driver.** Neither `EmbeddedScanner._execute_trades` nor `continuous.py:802` passes `mode=`; `ContinuousConfig.dry_run` is never read; `continuous.py:801` force-sets `enable_execution=True`. **Risk of NOT fixing:** an operator who believes `--watch` / a `dry_run=True` daemon is safe gets **real OANDA orders**. **Risk of fixing wrong:** a thread-through bug here could itself place live orders — why it's operator-gated. Fix-spec: thread `mode='dry_run' if dry_run else 'live'` end-to-end (`main.py` → `ContinuousConfig` → `run()` → `execute_trades`); stop forcing `enable_execution`; add no-mock integration test asserting zero broker calls. |
| `bugs-3` | **critical** | `soak_orchestrator.py:36-61` | **Soak compares a momentum toy vs a HOLD-only stub** (see §5). Baseline `_legacy_signal_fn` is a 2-bar sign-of-return dummy, not the real `ModularEnsembleInference`; SOTA path loads a flat `sota_model.keras` the versioning system never writes. Fix-spec: wire the real incumbent; assert `sota._loaded` before running, else emit `UNDECIDED`; gate promotion on `REPLACE AND n_bars>=MIN_BARS AND real expectancy delta`. **Risk:** a wired consumer could promote on noise. |
| `bugs-2` / `neural-9` / `safety-4` | **critical/med** | `continuous.py:48-70` | **`_persist_sota_config` has zero callers; `get_latest_result()` has zero callers.** The soak→promote→persist gate is unwired — a soak result never flips `use_sota_inference`. If wired, the regex YAML rewrite is unsafe (matches comments/nested keys, non-atomic, silent no-op if key absent — the `$3,527 dead-write` class). Decision: **delete the dead function + drop the auto-promote claim**, OR wire behind an explicit operator-gated, atomic, key-validated path. Do not advertise auto-promote as functional until then. |
| `bugs-4` / `safety-2` | **high** | `model_versioning.py:25,73-85,104-160`; `inference.py:36` | **Rollback is a no-op against live inference.** Versioning writes `sota_versions/<ts>/model.keras` + symlinks `sota_finetuned/latest` (a dir); live inference loads the flat `sota_finetuned/sota_model.keras` and never reads `latest`. `rollback_sota.py` reports success while the running scanner keeps the old weights. **Risk:** a false emergency-rollback control on an autonomous live daemon. Fix-spec: unify the path (repoint inference at `latest/model.keras`, or have rollback atomically rewrite `sota_model.keras`); add a test asserting rollback changes the bytes `SOTAInference()` next loads. |
| `neural-5` | **high** | `model_versioning.py:46-91`; `trainer.py:351` | **Versioning system has zero callers.** `save_versioned_artifact` is never invoked; trainer writes `sota_model.keras` directly. Manifest always empty → `--list` shows nothing, `--to <ts>` always fails. The rollback safety mechanism does not exist in practice. Cross-file fix (trainer + inference config + test) — out of single-file scope; needs operator-sanctioned multi-file change. |
| `neural-4` / `bugs-11` | **high/med** | `autonomous_trainer.py:490-503`; `scripts/scheduled_retrain.py` | **Retrain→soak chain wired to the wrong trainer.** `scheduled_retrain.py` trains per-pair/joint transformer models (0 SOTA refs) but the soak triggers against `sota_model.keras`, which it never produces → permanent HOLD. `status=='success'` only means `rc==0`, not that the artifact exists; only `pairs[0]` is soaked; failure swallowed to `debug`. Fix-spec: resolve+verify the actual produced artifact before triggering; soak per trained pair; raise the swallowed log. (Joint training is also deprecated per `improvement.md`.) |
| `neural-8` | **high** | `cycle_autonomy.py:140-149` | The *observability* half of this was auto-fixed (`bugs-7`). This entry remains flagged because the fix-spec **couples** the poll to routing the recommendation into the promotion decision — a live-promotion change requiring the fail-closed gate. Confidence **HIGH** on the defect; promotion-routing is the operator's call. |

> **`safety-3` (partial):** the defensive half landed (worker forces `recommendation=INSUFFICIENT`
> when SOTA never loaded or produced zero directional predictions, so no STACK/PROMOTE can derive
> from constant-HOLD). The incumbent-swap half (replace `_legacy_signal_fn` with real
> `ModularEnsembleInference`) was **deferred** — it's a substantive behavioral change with an
> unverified constructor/predict contract and its own per-bar perf concern.

---

## 4. Refuted Findings (1)

| ID | claim | verdict |
|----|-------|---------|
| `safety-0` | "Auto-execute path sends live orders while `state.json halted=true`; `execute_trades` has no halt check." | **REFUTED — confidence HIGH.** The observation about `continuous.py:769` being gated only on `auto_execute and not _scanner_paused` (pause ≠ halt) is correct, BUT the conclusion is false: every order funnels through `ExecutionManager.execute_trade` (`execution.py:2090-2100`), whose **first executable statement** is `if StateEngine().get_halted(): return ExecutionResult(success=False, error="BLOCKED: state.halted=True")` — added specifically to close the trade-1306 mid-cycle bypass (2026-05-12). No broker call precedes the guard. Residual lower-severity nit (NOT the reported critical): that guard catches its own exception and proceeds, so a `StateEngine()` failure could leak an order — a robustness gap worth a follow-up, but the path is **not** unguarded as claimed. |

---

## 5. Soak-Gate Verdict — Is auto-promote currently gating on noise?

**No — because it is gating on nothing at all. The auto-promotion loop is unwired end-to-end, and
the soak it would read is structurally invalid.** Confidence **HIGH** (every link verified from disk).

**The promotion loop does not exist as wired:**
- `_persist_sota_config` (`continuous.py:48`) — **zero callers**.
- `SoakOrchestrator.get_latest_result()` (`soak_orchestrator.py:166`) — **zero callers**.
- `save_versioned_artifact` (`model_versioning.py:46`) — **zero callers**.
- The only live poll (`cycle_autonomy.py`, pre-fix) read a throwaway instance that was permanently
  `idle` and only emitted a brain-log line — it never read a recommendation, never promoted.

So **no soak result has ever flipped `use_sota_inference`.** The "auto-promote SOTA" capability the
daemon advertises is decorative.

**Even if it were wired, the soak comparison is invalid in both directions:**
- **Baseline** = `_legacy_signal_fn` (`soak_orchestrator.py:36-45`), a self-documented 2-bar
  sign-of-return momentum toy — not the real `ModularEnsembleInference` the bot trades.
- **SOTA arm** = constant HOLD. Two compounding causes: (1) `pred.get(...)` on a `TradeSignal`
  dataclass raised `AttributeError` every bar (now fixed, `neural-0`); and (2) the model path
  `sota_finetuned/sota_model.keras` is **absent on disk** (versioning never writes it), so
  `SOTAInference` fails closed to HOLD-only.
- HOLD is scored never-correct (`model_comparison.py:100`) and zero-PnL, so `_make_recommendation`
  (`error_correlation.py:198`) hits `sota_acc<0.51 OR sota_pnl<=0 → DISCARD` deterministically.

**What changed in this review:** the soak now (a) reads the SOTA `TradeSignal` correctly
(`neural-0`), (b) constructs the model once instead of per-bar (`neural-6`), and (c) **fails closed
to `INSUFFICIENT`** when SOTA never loads or emits zero directional predictions (`safety-3`
defensive half) — so a future wired consumer **cannot** derive a STACK/PROMOTE from a constant-HOLD
run. What did **not** change (operator-gated): the dummy baseline is still in place, the model-path
contract is still mismatched, and **nothing consumes the verdict**. Net: the gate fails closed today
(good), but it is hollow — it can neither promote a good model nor honestly evaluate one until the
incumbent baseline, the model-path contract (`bugs-4`/`neural-5`), and the promotion wiring
(`bugs-2`/`neural-9`) are addressed under operator sign-off.

---

## 6. Notes & residual gaps (not changed, surfaced for follow-up)

- **`harvest.py:_append` (`:270`)** still does a non-atomic write and does not yet take the
  `_harvest_lock_path` lock — the `bugs-6` cross-process guarantee is one-sided until it adopts the
  shared lock + `os.replace` pattern.
- **`tick_post_flush._aggregate`** (per the `continuous.py` multi-pair dispatcher) still ignores the
  `ticks` arg downstream in some paths — the `bugs-5` fix addresses the file-local aggregator; verify
  the dispatcher and aggregator agree on the batch they pass.
- **`StateEngine()` exception-swallow** at the halt guard (`execution.py:2101-2105`) — a robustness
  follow-up surfaced while refuting `safety-0`; harden so a `StateEngine` failure fails closed.
- **Working tree is uncommitted** on `codex/sota-activation-execution`; runtime `mode=dry_run`,
  `halted=false`. The applied fixes are on disk but not in the running process until restart.
