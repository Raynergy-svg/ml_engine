# Improvement Rules

Meta-rules governing how Buddy learns and evolves.

## Learning Triggers
- Every closed trade triggers learning extraction (analyze outcome vs prediction)
- Every losing trade > $100 triggers deep analysis (LLM-assisted if enabled)
- Every 10 scan cycles triggers learnings audit (consolidation check)

## Promotion Criteria
- A pattern observed 3+ times in learnings.md gets promoted to rules/trading.md
- Promoted rules include the date, source count, and specific actionable directive
- Source learnings are marked [PROMOTED] after extraction

## Consolidation
- When learnings.md exceeds 30 entries: group by category, archive old entries
- When rules/trading.md exceeds 50 lines: split by domain (entry rules vs risk rules)
- When config_adjustments.json exceeds 100 entries: archive entries older than 30 days

## Code Quality Gates (promoted 2026-03-18, from 4 robustness observations)
- ALWAYS run code review specialist on new subsystems BEFORE first production use
- ALWAYS validate JSON parsing with try/except and graceful defaults (never crash on corrupted files)
- ALWAYS use file locking (fcntl) when multiple processes may write to shared JSON files
- ALWAYS verify state.json claims against source-of-truth files before acting on them
- ALWAYS use lazy imports in package __init__.py when submodules have heavy dependencies

## JSON Safety Gates (promoted 2026-03-23, from 31 observations)
- ALWAYS wrap JSON file reads in try/except with graceful fallback to empty dict/list
- ALWAYS validate JSON structure after parsing (check expected keys exist before access)
- ALWAYS write JSON atomically: write to .tmp file first, then os.rename() to final path
- NEVER trust JSON file contents without schema validation in production paths
- ALWAYS use json.dumps with indent=2 and sort_keys=True for human-readable persistence

## Retry & Robustness Gates (promoted 2026-03-23, from 27 observations)
- ALWAYS implement exponential backoff for OANDA API calls (base 1s, max 30s, jitter)
- ALWAYS set explicit timeouts on all HTTP requests (connect=5s, read=30s)
- ALWAYS catch specific exceptions (requests.Timeout, ConnectionError) not bare except
- NEVER retry on 4xx client errors — only on 5xx, timeout, and connection failures
- ALWAYS log retry attempts with attempt number, delay, and error context

## State Persistence Gates (promoted 2026-03-23, from 8 observations)
- ALWAYS flush state to disk before shutdown (save_state() in every module with mutable state)
- ALWAYS validate state file freshness on load (check timestamp, warn if stale > 1 hour)
- NEVER assume in-memory state survives process restart — always persist critical state
- ALWAYS include version field in persisted state files for forward compatibility

## Test Coverage Gates (promoted 2026-03-23, from 8 observations)
- ALWAYS write unit tests for new calculation/logic functions before merging
- ALWAYS test edge cases: zero values, negative values, None inputs, empty collections
- ~~ALWAYS use mock-based testing for external API dependencies~~ **SUPERSEDED 2026-05-01: NO MOCKS — see No-Mock Rule below**
- NEVER ship a new subsystem without at least 5 unit tests covering core paths

## No-Mock Rule (promoted 2026-05-01, from 1 catastrophic observation)
- NEVER use `unittest.mock`, `MagicMock`, `patch`, or any test-double class. The 2026-05-01 audit found 38 passing `test_meta_manager.py` tests hid a production wiring gap (`StagedDeployer` constructed without `config_adjuster=ConfigAdjuster()` → 11 ChangePackages walked through shadow→canary→live with zero actual config mutation). Mocks make integration gaps invisible.
- Tests must use real classes against real disk. Pattern: real `ScannerConfig()`, real `ConfigAdjuster(persistence_path=tmp_path / "adj.json")`, real `AdjustmentApprover()`, real `StagedDeployer(...)`, real `MetaManager.intake(...)`. Drive the pipeline; assert on real disk state and real config attribute mutation.
- For external APIs (OANDA, news feeds): skip the test, mark `@pytest.mark.integration`, or use a sandbox. Don't mock.
- For clocks: pass real timestamps or use real `datetime.now()`. Time-based tests use small thresholds, not frozen time.
- If the code-under-test cannot be tested without mocks, the code is too coupled — refactor it. The existing mocked test suite stays as-is (don't rewrite retroactively); never add a new mock; migrate when touching a test file for other reasons.

Source: 1 catastrophic observation — see `docs/incidents.md` "No-Mock catastrophe".

## Config Validation Gates (promoted 2026-03-23, from 6 observations)
- ALWAYS validate config values at load time (range checks, type checks, required fields)
- ALWAYS provide sensible defaults for optional config fields via dataclass defaults
- NEVER silently ignore unknown config keys — log a warning for typo detection
- ALWAYS ensure profile-specific overrides don't violate safety invariants (min SL, max risk)

## Silent Exception Prevention (promoted 2026-03-23, from 4 observations)
- NEVER use bare except: or except Exception: pass — always log the error
- ALWAYS re-raise or return error status after logging — callers must know something failed
- ALWAYS include context in error logs: function name, input parameters, stack trace
- NEVER swallow errors in financial calculation paths — surface them as trade rejections

## Live Wiring Verification Gates (promoted 2026-03-24, from 6 observations in single audit)
- ALWAYS run a live scan smoke test after wiring a new module — verify the module's log output appears in production output, not just in unit tests
- ALWAYS add new config feature flags as dataclass fields FIRST, then profile dict entries, then consumer getattr() calls — missing any one of these three = dead feature silently skipped by apply_profile()
- ALWAYS verify both the write side AND read side of any feedback/telemetry system — if record_X() exists without a corresponding get_X() consumer in the live loop, the module is write-only dead code
- NEVER trust "passes: true" on wiring phases without verifying call sites exist in production files — unit tests mock boundaries and will pass even when the production call is missing
- ALWAYS check that methods defined for integration are actually CALLED, not just defined — grep for the method name outside its own class to confirm at least one live call site exists
- ALWAYS test new feature flag propagation end-to-end: set flag in profile dict → verify it reaches the dataclass field → verify consumer reads True → verify module activates (log line appears)

## Anti-Patterns
- Never create new .claude/ files without justification — edit existing ones
- Never let learnings accumulate without triage (apply / capture / dismiss)
- Never evolve config silently — log every adjustment with reason
- Never guess at stale state — read state.json, ask if unclear

## Config Adjustment Consumer Verification (promoted 2026-04-16, from 4 observations: 3 self-heal dead-letter reports + 1 root cause trace)
- ALWAYS verify that ConfigAdjuster._load_state() loads ALL persisted fields: pending, history, last_applied
- ALWAYS verify that pending config adjustment keys EXACTLY match ScannerConfig dataclass field names — mismatched keys silently create orphan attributes via setattr() that no code reads
- ALWAYS run a round-trip smoke test after adding a new config adjustment source: write pending → restart process → verify apply_adjustments() consumes it → verify config attribute changed
- NEVER assume a feedback loop is closed just because both write and read sides exist — verify the persistence layer connects them (this bug cost $3,527 over 14 trades)
- ALWAYS validate proposed config keys against ScannerConfig dataclass field names BEFORE writing to config_adjustments.json — grep for the exact field name in config.py. Confirmed orphan keys from cycle 3: min_confidence_threshold->min_confidence, atr_sl_multiplier_low_regime->atr_sl_multiplier. 5/9 proposals were dead on arrival.

## Train↔Inference Contract Gates (promoted 2026-05-08, from 1 catastrophic observation: Phase 1 audit)
The inference path must reproduce the training pipeline's feature distribution exactly. A latent skew accumulated 6 simultaneous violations (identity scaler from double-fit, OHLCV passed only at inference, regime one-hots not computed at inference, column ordering different, feature selection skipped at inference, `feature_names` never read by gates) that broke every transformer prediction since C1.A landed. See `docs/superpowers/plans/2026-05-08-pipeline-reconciliation-phase1-audit.md`.

- NEVER fit a scaler twice. Specifically: never call `StandardScaler().fit_transform(x)` on a matrix that has already been through `StandardScaler.fit_transform`. The double-fit produces var_=1.0 exactly across all columns — the saved scaler becomes the identity transform. Tripwire `_assert_scaler_not_identity` in `transformer_trainer.py` detects this; it should never log ERROR in a healthy training run.
- ALWAYS save the inference contract in model meta. Required keys: `feature_names` (ordered), `scaler` (fitted, real per-column stats, not identity), `regime_quantiles` (q25/q50/q75 if regime one-hots in feature_names), `regime_atr_col` (which atr feature drove the quantiles), `feature_pipeline_version` (semver bumped on any change to compute_normalized_features or load_direction_data column set).
- ALWAYS read the inference contract before predict at inference. `gates._load_transformer` must read the meta sidecar and populate `_transformer_feature_names`, `_transformer_scaler`, `_transformer_regime_quantiles`, `_transformer_regime_atr_col`, `_transformer_pipeline_version` on the evaluator instance. Missing meta = inference cannot be trusted.
- NEVER zero-fill missing inference features silently. If a column required by `feature_names` is unavailable, REFUSE (return None, log a contract gap warning) — silent zero-fill produces predictions that look fine but use inputs the model never saw at training, which is the worst possible failure mode.
- ALWAYS assert `feature_pipeline_version` match at load. If the artifact's version differs from the runtime constant, refuse to use the model. The pipeline contract is a hard boundary; cross-version inference produces silent skew that's invisible until it costs money.
- ALWAYS subset (don't refit) a fitted StandardScaler when applying feature selection. `transformer_trainer._subset_scaler(scaler, indices)` returns a fresh StandardScaler with `mean_/scale_/var_` narrowed to the selected columns; applied to RAW selected features at inference, it reproduces the same per-column standardization the model was trained on. This is mathematically only safe because StandardScaler is per-column independent — other transforms (PCA, ICA, anything with cross-column terms) cannot be subset this way and need a different feature-selection-then-fit ordering.
- BEFORE shipping a model, verify the saved scaler stats look real: `var_` should NOT be 1.0 ± 1e-9 across all features; binary one-hot columns should have `mean_ ≈ p` and `scale_ ≈ √(p(1-p))`, not `mean_=0, scale_=1`. The latter pattern means the column was constant-zero at training, which means the feature pipeline silently zeroed it (e.g. broken time-extraction, regime augmentation skipped).

Source: 1 catastrophic observation — see `docs/incidents.md` "C1 train↔inference scaler skew" and `docs/strategy.md` "Inference contract". Re-validate after 30 days of live data with the Phase 2.A+B pipeline.

## Hard Ship Gate — 10% Train/Val Gap (promoted 2026-05-13, operator directive)
"No models is to be shipped higher than 10% gap" — operator rule 2026-05-13. The training stack already has helpers to manage the gap (EarlyStopping, EMA, class-balanced loss, label smoothing, auto-dropout, SWA), but enforcement at the ship boundary was broken on two layers prior to this rule:

1. `train_single_model_m1.py` read metrics via `trainer.get_metrics()`, which **does not exist** on TransformerDirectionTrainer. `hasattr()` returned False → empty dict → train_acc/val_acc/gap all reported as 0 → the cosmetic 6% PASS/FAIL gate (MAX_GAP) was a permanent no-op.
2. The trainer's `_compute_final_metrics` reported `train_accuracy = history.history["accuracy"][-1]` — the LAST epoch's value, not the best-val epoch's. EarlyStoppingCheckpoint restores best-val weights for the saved model, so `train[-1]` (post-overfit) didn't describe the SAVED model state — produced a phantom gap that would have falsely quarantined good models.

- ALWAYS read `trainer.metrics` directly (or via the `_read_trainer_metrics` helper). Never trust `getattr(trainer, "get_metrics", lambda: {})()` to return useful data.
- ALWAYS report `train_accuracy` at the best-val epoch (the epoch whose weights were saved). The transformer trainer now finds `best_epoch_idx = np.argmax(history["val_accuracy"])` and reports `history["accuracy"][best_epoch_idx]`. Any new trainer must follow the same contract.
- ALWAYS compute `gap = abs(train_accuracy - val_accuracy)` AFTER save, then route through `_quarantine_if_overshipped(...)`. If `gap > HARD_MAX_GAP (= 0.10)`, MOVE every artifact sharing the file's stem (e.g. `.keras`, `.meta.pkl`, `.ema.pkl`, `.ewc.pkl`, `.arch.json`, `.weights.h5`) to `trained_data/models/{PAIR}/_quarantine/{model_name}-{utc_ts}/`. Per-pair gate routing CANNOT pick from `_quarantine/`.
- NEVER set `HARD_MAX_GAP` above 0.10 without explicit operator approval. The cosmetic `MAX_GAP=0.06` (PASS/FAIL printed in RESULT lines) can be stricter, but the ship gate is a hard rail.
- ALWAYS emit `RESULT:{...}` lines containing `train_acc`, `val_acc`, `gap`, and `quarantined` (bool + `quarantine_dir`) for downstream automation (W&B, deploy gate, retrain agent). The legacy `gap=0` pattern means the metric-read path is broken — fail loud.

Source: 2026-05-13 operator directive after the 2026-05-12 5-pair M15 retrain produced val_acc=4.7-18.9% with balanced_acc≈50% across all 5 pairs. Root-cause investigation surfaced TWO compounding bugs: the missing-weight forwarding fixed in commit `1a05e75`, AND the broken gap-gate read/write paths fixed here.

## Pending experiments
Dated experiment runbooks (e.g. the 2026-05-18 HistGB capacity-shrink plan) live in
`docs/experiments-log.md` — this rules file holds promoted rules only, not one-off run plans.

## Joint Fallback Deprecation Gates (promoted 2026-05-12, operator directive)
The joint training output (`trained_data/models/joint/`, plus root-level `modular_ensemble.meta.json`) is deprecated as a runtime fallback. Per-pair routing is the only supported gate / inference path going forward.

- NEVER add new code paths that fall back to `models/joint/` for gate evaluation, inference, or runtime model loading. The joint dir is audit-only. Add per-pair coverage for the instrument or drop it from `active_pairs`.
- NEVER include `joint_gates` or `modular_ensemble` in any freshness rollup that gates unhalt / trade execution. They remain visible in `freshness["groups"]` for inspection but are excluded from `oldest_age_days` + `stale_models` by `get_model_freshness_for_pairs` as of 2026-05-12.
- ALWAYS verify per-pair `transformer_direction.keras` exists at startup before treating an instrument as tradeable. Pairs without per-pair coverage hit the DEPRECATED warning log in `gates._get_pair_evaluator` (logged once per pair per process via the `_pair_evaluators` cache) and are heading for explicit refusal once the engine startup filter ships.
- ALWAYS surface the joint fallback firing — `gates._get_pair_evaluator` logs `DEPRECATED joint fallback: instrument=X has no per-pair directory...` at WARNING. If you see this in `logs/buddy_debug.log`, that instrument needs per-pair training (Tier 1 master or Tier 2 transfer) or removal from `active_pairs`.
- Removal sequence (track here as steps land):
  1. ✅ (2026-05-12) Demote `joint_gates` + `modular_ensemble` from `get_model_freshness_for_pairs` rollup — they no longer block unhalt or AGING/STALE/CRITICAL classification.
  2. ✅ (2026-05-12) Add DEPRECATED warning log in `gates._get_pair_evaluator` joint fallback branch.
  3. ⏳ (pending) Engine startup filter: when `use_per_pair_routing=True`, drop pairs without per-pair `transformer_direction.keras` from `self.config.pairs` with one WARNING per dropped pair. Touches engine.py + 4 call sites at 5091/5471/6682/6970 that read `self.config.pairs`.
  4. ⏳ (pending) Flip `_get_pair_evaluator` fallback from `return self` to `return None` (or raise `RuntimeError`). Callers updated to skip the pair cleanly.
  5. ⏳ (pending) Delete joint dir loading code in `GateEvaluator.__init__` and `engine.py:_initialize_models`. Delete `use_joint_models_only` config field. Delete joint training tier from `run_full_training.sh`.

Source: 1 operator directive on 2026-05-12 ("no joint as fall back.. deprecate") plus prior context — joint ensemble's last retrain was holdout-rejected 0/3 on May 4 (modular_ensemble.meta.json `_mythos_heal_note`), so trained_at was rolled back to April 16; joint has been stale-and-uncorrectable for 4+ weeks. The May 11 4-team unblock and per-pair routing fixes prove per-pair is the right path.
