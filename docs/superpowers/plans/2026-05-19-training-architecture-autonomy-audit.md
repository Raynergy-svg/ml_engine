# Training Architecture — Autonomy Fitness Audit

**Status**: Read-only audit (no code changes)
**Author**: Software Architect agent, 2026-05-19
**Trigger**: Operator directive — *"we now are autonomous quant loop, with agents"* (2026-05-18 23:18 EDT). Training architecture must serve agent invocations, not human-driven script edits.
**Builds on**: `docs/superpowers/plans/2026-05-19-training-architecture-control-plane-wiring.md` (hyperparam contract gap) and `docs/superpowers/plans/2026-05-19-training-run-timeline.md` (empirical val_accuracy timeline).

---

## Executive Summary

**Is the architecture fit-for-autonomous-operation? NO — three load-bearing gaps.** (HIGH confidence.)

1. **No single agent-invocation surface.** There are 16 training entry points across `scripts/` + `src/training/` + `online_retrainer.py`; agents (autonomous loop, ralph, scheduled_retrain, retrain_worker) reach training through 4 of them, and the 4 paths have **different hyperparameters and different lookahead defaults** (see prior wiring audit). An agent that invokes `train_single_model_m1.py` trains a different model class than one that invokes `scheduled_retrain.py`'s correlation-transfer path. There is no canonical `buddy retrain` CLI.

2. **Machine-readable status is incomplete and inconsistent.** `train_single_model_m1.py` emits `RESULT:{json}` lines on stdout AND writes `training_summary.json` per-instrument (`scripts/train_single_model_m1.py:653,679-691`), but `scheduled_retrain.py` emits no JSON status at all — only log files + W&B events. An agent that wants to verify a `scheduled_retrain` outcome must parse `logs/retrain_*.log` or subscribe to `TRAINING_COMPLETED` via the event bus. There is no single per-run status JSON the agent can read.

3. **Fail-silently / fail-confusing modes are unaddressed in the autonomous path.** No pre-flight disk-space check (despite the 2026-05-13 disk-full Bug B); OANDA fetch has a single retry-with-2s-sleep then `break` (`scripts/train_single_model_m1.py:227-235`) with no timeout — observation 1473 confirms it can hang indefinitely; per-pair correlation-transfer hard-stops at `<2 pairs` (`buddy_training_helpers.py:1990-2002`) with no auto-fallback, so a single-pair drift-triggered retrain dies with `insufficient_pairs_for_correlation` (observation 1391); no concurrent-training file lock — `retrain_worker._active_lock` is thread-local (`retrain_worker.py:37`), so two agents invoking `train_single_model_m1.py` directly will trample each other's `MODELS_DIR/{pair}/` writes.

**Top 3 recommendations (cheapest → highest leverage):** (1) **add a per-pair `status.json` write at the end of every training run** (~30 lines, single file, 0 blast radius); (2) **add disk-free + OANDA-timeout pre-flight gate** (~50 lines, gates both fetch hang and disk-full Bug B); (3) **ship `buddy retrain` CLI as Tier 2.B from the prior audit** (the architectural fix; ~250 lines, demotes the bypass paths). Detail in §8.

---

## A) Data Flow Integrity

The full chain: **OANDA fetch → cache CSV → `apply_features` → `load_direction_data` (or volatility/risk/etc.) → split → fit scaler → window/sequence → train → save with meta sidecar**.

### A.1 — Single points of failure (HIGH)

| Stage | Where | Failure mode | Today's coverage |
|---|---|---|---|
| OANDA fetch | `scripts/train_single_model_m1.py:204-263` | Hang on large `--count` calls, no read timeout (observation 1473) | One retry with 2s sleep, then `break` + empty df; no `requests.get(timeout=…)` is plumbed through |
| Cache write | `:260` | Partial write on `OSError` (disk full) leaves a half-written CSV that next run will pd.read_csv-fail | Not handled — `df.to_csv(cache_file)` is not atomic |
| Feature compute | `src/core/modular_data_loaders.py:724 compute_normalized_features` | Silent column-missing → contract-version-bumped artifact written but inference can't find required columns later | `feature_pipeline_version=2026-05-08-v1` saved (`modular_data_loaders.py:43,2349`) AND enforced at gate load (`gates.py:1257-1270` refuses mismatch) — HIGH confidence this layer is enforced |
| Scaler fit | `transformer_trainer.py` | Double-fit identity-scaler bug (`var_=1.0±1e-9`) — Phase 2.A tripwire `_assert_scaler_not_identity` catches this only at trainer ERROR-log level | Tripwire logs ERROR but does NOT fail the build — quarantine-on-identity-scaler is NOT in code today |
| Save meta sidecar | `transformer_trainer.py:3198-3242` | Sidecar write is not atomic (writes directly to `.meta.pkl`, no `tmp + rename`) | Not addressed — direct write |
| Save EMA / EWC / weights H5 / arch JSON | `:3246-3268` | If disk fills between `.keras` save and `.meta.pkl` save, model loads but contract sidecar is missing → gate logs WARNING but inference falls back to legacy untyped path (broken per audit doc) | Best-effort catch on individual files; no transactional save-group |

**Verdict**: contract enforcement on **inference load** (HIGH) is real and works (`gates.py:1257-1294`). Contract enforcement on **training save** is best-effort + non-transactional, so a partial-write under disk-full produces a model that *loads* (because `.keras` was written first) but inference *refuses* (because the sidecar is missing or stale). That's actually the right failure mode — but it's accidental, not designed.

### A.2 — `feature_pipeline_version` enforcement (HIGH)

Verified by grep:
- **Written**: `modular_data_loaders.py:2349` (in `load_direction_data`'s return), `transformer_trainer.py:3230` (in `meta` dict at save).
- **Read**: `gates.py:1244, 1257-1270` (refuses mismatch).
- **Constant source**: `modular_data_loaders.py:43 FEATURE_PIPELINE_VERSION = "2026-05-08-v1"`.

If a future trainer skips writing the version, `gates._load_transformer_contract` logs WARNING at `:1273-1278` ("inference will use legacy untyped path; expected to produce broken predictions until retrain") and continues with `self._transformer_pipeline_version=None`. The artifact is **not refused**, it's flagged. A net-new trainer (e.g. a new head added next quarter) that forgets the version key will silently ship.

**Gap**: no compile-time guarantee that every trainer save() writes the version. The check exists at load time, not at save time.

### A.3 — Per-pair routing fallback (MEDIUM)

`GateEvaluator(use_per_pair_routing=True)` lazily builds a per-pair sub-evaluator from `trained_data/models/{PAIR}/`. When the dir is missing or has no contract-complete artifact, the sub-evaluator falls back to the joint dir (`trained_data/models/joint/`). The joint dir holds a pre-Phase-2.A artifact (no scaler, no regime_quantiles, no version) per `scripts/scheduled_retrain.py:396-405`. That artifact will be refused by the contract gate → transformer returns `(None, 0.5)` → gate falls back to momentum/default SHORT → holdout reports the fallback predictor's accuracy on a SHORT-heavy slice. This was the 2026-05-11 Bug C; the fix shipped (`scheduled_retrain.py:417-421` now constructs the evaluator with per-pair routing on), but the underlying brittleness (joint dir is broken, never refreshed) is still there.

---

## B) Agent-Friendliness

### B.1 — Single CLI surface (FAILED, HIGH)

There is no single canonical CLI an agent calls. Today's surface fans out:

| Agent / trigger | Invokes | Path |
|---|---|---|
| `scheduled_retrain.py` launchd at Mon/Wed/Fri 10:00 UTC | `run_train_joint` → `train_per_pair_correlation_ensemble` (`buddy_training_helpers.py:1901`) | Per-pair correlation transfer |
| `retrain_worker.on_training_proposed` | `subprocess.Popen([scheduled_retrain.py, "--pairs", ...])` (`retrain_worker.py:90-107`) | Same as above |
| Operator ad-hoc / autonomous-loop child | `python scripts/train_single_model_m1.py --instrument X --model Y --candles N` | **Different** trainer config (epochs=200 vs ~50; MASTER_PARAMS HP set; lookahead=24 default) |
| `online_retrainer.trigger_retrain` | In-process sklearn retrain of momentum/risk/confidence heads only — NEVER retrains direction or volatility regime | Lightweight, replay-buffer-driven |
| Stage-4A news fusion | `python scripts/stage_4a_news_retrain.py --pair X` | Yet a third config + bypasses control plane |

Per the prior wiring audit, **3 of 16 entry points pull the W&B control plane correctly**; the 2 paths agents most often touch (`train_single_model_m1.py`, `stage_4a_news_retrain.py`) bypass it.

For an agent: there is no "canonical" — each agent has to know which sub-path it's wired to and its quirks.

### B.2 — Machine-readable output (PARTIAL, HIGH)

`train_single_model_m1.py`:
- Per-model `RESULT:{json}` line on stdout (`:653,662`)
- Per-instrument `training_summary.json` (`:679-691`) — keys: `instrument, candles, params, results[], total_duration_s, passed, failed, timestamp`

`scheduled_retrain.py`:
- No `RESULT:` line
- Writes `trained_data/retrain_all_summary.json` ONLY on success (`:1079-1089`) — keys: `timestamp, pairs, duration_seconds, status: "success"`
- Updates `trained_data/models/modular_ensemble.meta.json` with `candidate_trained_at` / `trained_at` / `holdout_pending` / `last_holdout_failure_at` (`:992-1026`, also `buddy_training_helpers.py:2058-2086`)
- Emits `TRAINING_COMPLETED` event to `trading_events.jsonl` ONLY when `BUDDY_WANDB_REGISTRY_ENABLED=1` and only on success (`:1052-1062`)

**Gap**: there is no single `status.json` per training-run-id that an agent can read to determine `{success, val_acc, balanced_acc, quarantined, contract_version, model_dir, timestamp}` without parsing logs or subscribing to the event bus.

### B.3 — Verifying success without log-grepping (FAILED, HIGH)

The agent has 3 surfaces to check today, each partial:

1. **Per-model `meta.pkl`** — has metrics + lineage + contract version. Loadable but requires Keras / sklearn lib availability + knowing the file path. Heavy.
2. **`training_summary.json`** — only `train_single_model_m1.py` writes it. Light, but path-by-pair, no unified index.
3. **`trading_events.jsonl`** — append-only event log. Has TRAINING_COMPLETED + MODEL_PROMOTED / MODEL_HELD. Requires parsing JSONL + filtering by `event_type` + `correlation_id`. Only fires when `BUDDY_WANDB_REGISTRY_ENABLED=1`.

There is no `trained_data/last_run_status.json` or `<pair>/<head>/last_run.json` the agent can stat-and-read in O(1).

### B.4 — Concurrency (FAILED, HIGH)

- `retrain_worker._active_lock` is a `threading.Lock()` in module scope (`retrain_worker.py:37`) — **only protects against multiple invocations within the same Python process**. Two `python scripts/train_single_model_m1.py --instrument EUR_USD` calls (e.g. from two different agent harnesses) will both run, both call `gc_cleanup()` + clear the TF Keras session, both write to `trained_data/models/EUR_USD/`. The last write wins; the first's outputs are lost.
- No file lock on `trained_data/models/{pair}/`.
- The Metal GPU is shared — TF's `set_memory_growth=True` (`train_single_model_m1.py:194`) doesn't prevent simultaneous device use, just doesn't pre-grab all memory. M1's 8-16 GB unified memory will OOM if 3 transformer trainings collide.

**Verdict**: agents MUST serialize externally today (cron schedule + explicit at-most-one launchd). The system does not enforce serialization.

---

## C) Reproducibility

### C.1 — Bit-exact reproducibility (FAILED, HIGH)

- No global seed contract. Seeds appear ad-hoc inside the transformer trainer (`np.random.default_rng(seed=42 + self.weight_perturbations)` at `transformer_trainer.py:2313`, `seed=42` at `:2374`), inside walkforward (`np.random.seed(42)` at `walkforward_validation.py:655,1417`). No seed is plumbed from CLI → `TrainerConfig` → trainer init.
- TF op-level determinism is not configured (no `tf.config.experimental.enable_op_determinism()`).
- M1 Metal GPU produces non-deterministic outputs for some Keras ops (matmul reductions); even with seeds, two runs of the same config differ by ~0.5% val_accuracy.

**Implication**: an agent cannot reproduce a prior best run to A/B-test a code change. The agent can re-train and *measure*, but cannot reproduce.

### C.2 — Run manifest (PARTIAL, HIGH)

The `meta.pkl` sidecar contains: scaler, config, lineage, architecture, feature_pipeline_version, news contract, regime quantiles. It is the closest thing to a manifest. But it lacks:
- git SHA of the code
- W&B run URL
- exact CLI args used
- environment (`tf-metal` version, numpy version, pandas version)
- random seed used

`training_summary.json` has `params` (the per-pair HP dict that was used) but not the code SHA. `trading_events.jsonl` has event_id, source, session_id but not the SHA either.

**Implication**: if an agent says "retrain EUR_USD M15 with config X", it has no record of which *version* of the code processed config X. A trainer regression between two retrains is invisible.

### C.3 — Walk-forward integration (FAILED, HIGH)

- `src/training/walkforward_validation.py` exists (1417 lines).
- **No grep hit** for `walkforward` or `WalkForward` in `scripts/scheduled_retrain.py`, `scripts/train_single_model_m1.py`, or `online_retrainer.py`.

The walkforward validation module is **not integrated into the autonomous path**. The autonomous retrain uses a single train/val split (`load_direction_data` returns `X_train, X_val, y_train, y_val` — `modular_data_loaders.py:1827`). Walkforward is library-only code today; agents don't trigger it.

---

## D) Failure Modes

| Failure | Today's behavior | Notes |
|---|---|---|
| Disk fills mid-training | Trainer's `model.save()` raises `OSError: No space left on device`. The training subprocess returns non-zero; `retrain_worker._watch_subprocess` publishes `queue_failure` event. **But**: `train_single_model_m1.py` writes saves sequentially (`.keras` → `.weights.h5` → `.arch.json` → `.meta.pkl`). If disk fills between #1 and #4, the model file exists, the contract sidecar doesn't. Inference will see the model and fail at contract load with a WARNING log. **No pre-flight free-disk check exists.** Observation: Bug B (2026-05-11) shipped val_acc=0.4839 to disk under disk-full conditions; the autonomous-retrain pipeline did not refuse. |
| OANDA fetch fails | `train_single_model_m1.py:227-235`: try → `except Exception` → 2s sleep → 1 retry → `break` + empty df. No timeout on `client.get_candles`. Observation 1473: large `--candles` calls hang indefinitely. The retrain worker watcher has no kill-timeout. |
| FinBERT / HuggingFace fetch fails | News fusion is P1 scaffolding only (`docs/superpowers/plans/2026-05-08-news-macro-signal-design.md`). The current `_evaluate_news_risk` agent in `src/scanner/agents/_team.py:2344` uses VADER + a separate EconomicCalendar. So today: no FinBERT in training. When P3 lands and FinBERT becomes a training-time dep, this failure mode reactivates. |
| Saved model is corrupt | `gates._load_transformer` `except Exception` → returns False → `transformer=None` → evaluate_transformer returns `(None, 0.5)` → falls back to momentum/default SHORT. **Silent**. No alert emitted. Operator sees nothing in `.claude/alert_state.json`. |
| Two agents retrain same pair concurrently | Process-level race. Both write to the same dir. Last-write-wins. No detection. No alert. |
| Contract version mismatch on load | `gates.py:1257-1270` logs ERROR and sets `self._transformer = None`. Inference correctly refuses. (Working — HIGH.) |
| Identity-scaler regression (var_=1.0±1e-9) | `_assert_scaler_not_identity` (per CLAUDE.md "Tripwires") logs ERROR but does NOT quarantine. The model ships. |
| `train_per_pair_correlation_ensemble` called with <2 pairs | Hard error `insufficient_pairs_for_correlation`. No auto-fallback to single-pair path. Observation 1391 confirms this catches single-pair drift-triggered retrains. |

**Pattern**: the system has **strong INFERENCE-side refusals** (contract version, missing scaler, missing model file) but **weak TRAINING-side guardrails** (no pre-flight checks, no quarantine on identity-scaler, no auto-fallback on insufficient-pairs).

---

## E) Observability

### E.1 — Metrics landing zones (MEDIUM)

For inspection after a training run:
1. **`meta.pkl`** per artifact — primary truth, but binary.
2. **W&B web** — when `WANDB_DISABLED=0` and `pull_config`/`log_run` succeed. Has run config + metrics. **Not used by `train_single_model_m1.py`** today (no `log_run` call).
3. **`logs/retrain_{ts}.log`** — verbose human log, grep-friendly.
4. **`trained_data/logs/m15_*.log`** — per-experiment ad-hoc logs.
5. **`trading_events.jsonl`** — only when `BUDDY_WANDB_REGISTRY_ENABLED=1`. Has TRAINING_COMPLETED + MODEL_PROMOTED / MODEL_HELD.

**Gap**: there is no unified per-run summary (e.g., `trained_data/runs/{ts}_{pair}_{head}.json`) that all paths converge on. Each path writes to a different surface.

### E.2 — Operator TUI / dashboard surface (PARTIAL, MEDIUM)

- `src/tui/widgets/model_inventory.py:144` reads `feature_pipeline_version` from per-pair meta files and surfaces them — so the TUI knows contract status. Confirmed via grep.
- F1 brain feed (`.claude/brain/feed.jsonl`) does NOT include training events by default. The TRAINING_COMPLETED handler logs to logger.info; whether that lands in the brain feed depends on the operator-side filter.
- The `.claude/alert_state.json` AlertManager surfaces consecutive_losses, drawdown, win_rate_drop, weight_instability — **not training failures**. A quarantined model is not an AlertManager event.

### E.3 — Quarantine notification (FAILED, MEDIUM)

`_quarantine_if_overshipped` (`train_single_model_m1.py:86-123`) moves files to `_quarantine/` and logs ERROR. **No event emitted to trading_event_bus.** Operator finds out by:
- Reading `logs/buddy_debug.log` for the ERROR line, OR
- Listing `trained_data/models/{pair}/_quarantine/`, OR
- Noticing degraded trading performance, then investigating.

There is no proactive "ship-blocked: review needed" notification.

### E.4 — Ship-gate visibility (PARTIAL, MEDIUM)

`scripts/train_single_model_m1.py:118-122` logs ERROR with the 🚫 emoji. That lands in `logs/buddy_debug.log` (if logger is hooked) AND in stdout (`logging.basicConfig(...StreamHandler())`). It does NOT explicitly write to `.claude/brain/feed.jsonl`. The autonomous-loop and ralph agents see it via their stdout-streaming, not via a structured event.

---

## F) Scaling

### F.1 — Pair count (MEDIUM)

Today: 15 active pairs (`scheduled_retrain.py:57-62 DEFAULT_SCANNER_PAIRS`). Each pair trains ~7 heads. The correlation-transfer orchestrator trains ~5-6 master pairs end-to-end + transfer-learns the rest from EWC.

- At **30 pairs**: master count stays ~6 (correlation groups grow ~linearly with N pairs but new pairs mostly fit existing groups); transfer count grows to ~24. Training time scales with transfers, not masters. Each transfer ~1 min on M1 Metal → ~30 min total. Acceptable.
- At **100 pairs**: assuming ~10 master pairs (correlation-group breadth at FX scale), 90 transfers. ~90 min. Begins to bump up against the 6-hour W&B sweep window and the OANDA-fetch rate limit (`time.sleep(0.5)` between 5000-candle batches, `train_single_model_m1.py:254`). The single-process fetch-then-feature-then-train sequencing becomes the bottleneck — no parallelism.
- **Breaks at**: disk space (`df -h` already at 92%, observation in timeline doc), W&B artifact upload bandwidth, and OANDA rate limit (untimed).

### F.2 — Head count per pair (MEDIUM)

Today: 7 heads per pair (direction, confidence, momentum, risk, volatility_regime, trend_regime, meta_labeler). At **10 heads**: control-plane JSON expansion is trivial (one new JSON per head); trainer addition is moderate (one new function in `train_single_model_m1.py` + one new entry in `MODEL_TRAINERS`); the bottleneck is the **manual nature of adding a head** — every new head requires touching ≥3 files (`train_single_model_m1.py`, the trainer module, the gate consumer). No registry-driven head onboarding.

### F.3 — Metal GPU serialization (HIGH)

- M1 unified memory + tf-metal: TF claims the entire Metal device per process. Two concurrent `train_single_model_m1.py` invocations contend for the same Metal queue.
- `tf.config.experimental.set_memory_growth(gpu, True)` (`train_single_model_m1.py:194`) prevents pre-grabbing all memory but does NOT prevent simultaneous use. Two transformer trainings = ~12-14 GB combined active memory + thrashing.
- **No parallel-pair training within one process** — `train_single_model_m1.py` serializes by design (one `--instrument` per invocation, one model loop per instrument).

**Implication**: scaling beyond 15 pairs requires either (a) external orchestration (sequential queue), (b) sharding by CPU (no Metal), or (c) moving heavy heads to non-Metal infra (cloud GPU).

---

## Fitness Matrix

Rows are the 6 audit dimensions. Columns: **today** (what's actually shipped), **gaps** (what's missing for autonomy), **target** (the agent-friendly version).

| Dimension | Today | Gaps | Target |
|---|---|---|---|
| **A. Data flow integrity** | Inference-side contract enforcement works (HIGH); training-side save is non-transactional, no identity-scaler quarantine, no per-trainer save-version check | Atomic save group (`tmp + rename + fsync`); ship-block on identity scaler; compile-time guarantee every trainer writes `feature_pipeline_version` | Transactional artifact group; tripwires fail-hard (quarantine, not just log) |
| **B. Agent-friendliness** | 4 different agent paths into 16 entry points; partial machine-readable status (RESULT line + per-pair JSON); no concurrency control | Single canonical CLI (`buddy retrain`); unified per-run status JSON; cross-process file lock | One CLI, one status JSON path, one lock |
| **C. Reproducibility** | No global seed; no run manifest with git SHA; walkforward exists but is library-only | Plumb seed via `TrainerConfig`; embed git SHA + CLI args + env versions in meta; wire walkforward as opt-in for promotion gate | Bit-reproducible runs (modulo Metal nondeterminism); walkforward as opt-in deploy gate |
| **D. Failure modes** | Strong inference-side refusals; weak training-side guardrails (no disk check, no fetch timeout, no concurrent-write protection, no auto-fallback on <2 pairs) | Pre-flight disk + memory + OANDA-reachability check; `requests.get(timeout=)` plumbing; single-pair fallback for correlation transfer; cross-process lock | Pre-flight guard gate; explicit fallback policy; ship-block on every safety violation |
| **E. Observability** | meta.pkl + W&B (when wired) + `retrain_*.log` + `trading_events.jsonl` (when enabled); quarantine logs ERROR but emits no event | Unified `last_run.json` per pair-head; AlertManager event on quarantine; brain-feed line for every promote/hold/quarantine | One status surface per pair-head; quarantine = AlertManager alert |
| **F. Scaling** | Serialized on Metal GPU; OANDA fetch single-threaded; disk at 92%; 15 pairs ~30 min retrain | Disk-space management (LRU cache eviction); parallel-fetch (rate-limited); job queue for >15 pairs | Job-queue executor with disk + Metal admission control |

---

## Recommendations — ranked by leverage (cheapest information / smallest fix first)

### 1. Per-run unified `status.json` writer (cost: ~30 lines; leverage: HIGH)

**HIGH confidence** this is the smallest cost-high-leverage fix. Add a single helper called by every training entry point at the END of its run:

```
# pseudo-API
write_run_status(
    pair, head, status, val_acc, balanced_acc, train_acc, gap,
    quarantined, contract_version, model_dir, git_sha, cli_args,
    duration_s, timestamp,
) -> Path  # writes trained_data/runs/{ts}_{pair}_{head}.json + updates trained_data/runs/index.jsonl
```

Wire into `train_single_model_m1.py:653` (where `RESULT:{json}` is already emitted), `scheduled_retrain.py:1081` (where `retrain_all_summary.json` is written today), `online_retrainer.py` post-retrain. Agent can now stat-and-read a single index.

**Risk if wrong**: zero — additive. Existing surfaces unchanged. Revert by removing the helper.

### 2. Pre-flight gate (disk + OANDA timeout) (cost: ~50 lines; leverage: HIGH)

**HIGH confidence** this gates the two confirmed Bug B and observation-1473 incidents. Add a `preflight_check()` helper at the start of every training entry point:

```
# pseudo-API
preflight_check(min_free_gb=5, oanda_probe_timeout_s=10) -> (ok, reasons[])
  - shutil.disk_usage("/Users/buddy/Documents/ml_engine/trained_data") -> free GB
  - OandaPracticeClient.get_candles(probe_pair, count=10, timeout=oanda_probe_timeout_s)
  - return (False, ["disk_low:4.2GB", "oanda_unreachable"]) on failure
```

Also: thread `timeout=30` through `OandaPracticeClient.get_candles` (currently no timeout, observation 1473).

**Risk if wrong**: LOW — fails closed on disk-low, which is the conservative behavior. If false-positive (e.g., transient OANDA blip), retrain is skipped + retried next cycle.

### 3. `buddy retrain` CLI as Tier 2.B (cost: ~250 lines; leverage: HIGH)

This is the architectural fix from the prior wiring audit, promoted to a top-3 recommendation here because it's load-bearing for autonomy. The CLI:

1. Single agent-invocation surface (`python -m buddy retrain --pair X --granularity M15`).
2. Internally pulls W&B control plane (closes the 13 of 16 bypass paths).
3. Enforces operator safety floors (`candles >= 25000`, `lookahead` from JSON, hard quarantine on `gap > 10%`).
4. Writes unified `status.json` (recommendation 1).
5. Runs pre-flight check (recommendation 2).
6. Cross-process file lock on `trained_data/models/{pair}/` via `fcntl.flock` on a `.lock` sentinel.
7. Logs every run via `wandb_control_plane.log_run` so agent retrains are A/B-comparable against operator sweeps.

The old scripts stay as compatibility shims with `DeprecationWarning`. Existing `com.mlengine.retrain.plist` keeps working until migrated.

**Risk if wrong**: MEDIUM — biggest blast radius of the three. Mitigation: ship as new file (`src/cli/retrain.py`); keep `scripts/train_single_model_m1.py` working; flip the launchd manifest only after operator validates a smoke run end-to-end.

### Confidence calibration on each

| Recommendation | Confidence on "this fixes the gap" | Confidence on cost estimate | Failure mode if recommendation is wrong |
|---|---|---|---|
| 1. Per-run status JSON | **HIGH** — agents stop log-grepping immediately on landing | **HIGH** — ~30 lines is a generous estimate | Agent has extra noise to ignore (vestigial JSON files) — no real downside |
| 2. Pre-flight gate | **HIGH** — both Bug B (disk-full) and observation 1473 (OANDA hang) directly addressed | **MEDIUM** — 50 lines if OANDA-probe is a one-call get-candles; could be 80 if we add a proper "ping" endpoint | False-positive blocks a legit retrain → next cron tick retries, MEDIUM cost |
| 3. `buddy retrain` CLI | **MEDIUM** — fixes the wiring + concurrency + status, but the val_accuracy gap (51 vs 60) is hypothesized-not-confirmed to be the control-plane lookahead divergence; if hypothesis is wrong, the CLI ships safety + ergonomics but doesn't close the perf gap | **MEDIUM** — 250 lines if it wraps existing trainers; 400+ if it has to refactor `train_single_model_m1.py`'s MASTER_PARAMS path | Operator's existing flow (manual `python scripts/...`) keeps working; deprecation is opt-in. Worst case: 2 weeks of dual maintenance. |

---

## Cross-references

| Doc | Relationship |
|---|---|
| `docs/superpowers/plans/2026-05-19-training-architecture-control-plane-wiring.md` | The HP / control-plane wiring side. This audit builds on it: that doc says "wire `pull_config` into 4 bypassing scripts" (Tier 1 PRs 1-7); this doc says "and then put a single CLI in front of all of them" (Tier 2.B). Neither doc re-litigates the other. |
| `docs/superpowers/plans/2026-05-19-training-run-timeline.md` | Empirical val_accuracy timeline. The "no transformer run >0.60" finding informs this audit's claim that Buddy training is on the back foot — autonomy gaps are not just hypothetical. |
| `docs/superpowers/plans/2026-05-08-pipeline-reconciliation-phase1-audit.md` | The Phase 2.A inference-contract audit. This doc cites the contract enforcement as **the one thing that works**; its enforcement is the rest of the system's safety net when training fails partially. |
| `docs/superpowers/plans/2026-05-05-self-heal-self-train-roadmap.md` | The self-heal / self-train master plan. The audit findings here feed back into that roadmap's "what must work before unhalting" gate. |
| `docs/superpowers/plans/2026-05-08-news-macro-signal-design.md` | P1 news fusion design. Once P3 lands, FinBERT becomes a training-time dep — Failure-Mode D needs revisiting then. |

---

## 10% gap rule — untouched

Per operator directive: HARD_MAX_GAP=0.10 is non-negotiable. Every recommendation here preserves or strengthens it. Recommendation 3's `buddy retrain` CLI moves the quarantine check INTO the CLI (rather than living in `train_single_model_m1.py` only), so a future bypassing path cannot ship a >10% gap model. No recommendation lowers or hides the rule.
