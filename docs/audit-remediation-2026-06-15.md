# Audit Remediation — 2026-06-15

**Summary:** Of 11 audited findings, 10 were confirmed real (all `high` confidence) and 1 (`F3-tcn-regime-fields`) was refuted as by-design; 2 non-trading-semantic fixes in `continuous.py` (B2 logger TypeError + B3 `_error_frequency` relocation) were auto-applied and verified, while the remaining 8 confirmed findings touch trading semantics and are deferred below as ready-to-implement specs pending operator approval.

---

## 1. Confirmed-Real Findings

| ID | Confidence | Evidence (file:line) | Touches trading semantics | Status |
|----|-----------|----------------------|:--:|--------|
| B1-dup-learning-loop | high | `src/scanner/automation/continuous.py:1966` (1st def) + `:2268` (2nd def, identical sig); Python keeps the 2nd, 1st is dead | Yes | Deferred (§5.1) |
| B2-selfheal-logger-typeerror | high | `src/scanner/automation/continuous.py:45` (stdlib `getLogger`) + `:1134-1135,1166-1167,1172,1176-1177,1181` (structlog kwargs) → `TypeError: Logger._log() got an unexpected keyword argument 'error_sig'` | No | **APPLIED (§4)** |
| B3-error-frequency-classlevel | high | `src/scanner/automation/continuous.py:1093` declares `_error_frequency` at class scope; `__init__` (77-298) never inits it; used via `self._error_frequency` (`:1131-1132,1170`) | Yes | **APPLIED (§4)** |
| B4-double-smartloop-obs | high | `src/scanner/automation/continuous.py:517-528` + `:744-754` (identical obs-logging blocks); `:553` + `:854` (two `_run_smart_loop()` calls); `:1957,1960` (dup `_run_learning_loop()` calls) | Yes | Deferred (§5.2) |
| F1-meta-deadwrite | high | `src/scanner/config.py:563` `meta_labeler_threshold`; `src/core/modular_inference.py:514` `min_meta_confidence`, read at `:4812`; `src/scanner/engine.py:1899-1921` builds `InferenceConfig` but never maps it; `src/scanner/sota_integration.py:60-69` legacy path also omits it | Yes | Deferred (§5.3) |
| F2-trainer-feature-names | high | `scripts/train_single_model_m1.py:511-512,547,570,593,626-628` — no `feature_names` passed; loaders return them at `src/core/modular_data_loaders.py:4096,3432,3630,3871` | Yes | Deferred (§5.4) |
| F4-contract-version-asymmetry | high | `src/scanner/gates.py:1290-1303` — ONLY transformer enforces `feature_pipeline_version`; `_load_tcn_volatility:627`, `_load_ridge_confidence:1127`, `_load_rf_risk:1173`, `_load_meta_labeler:1329` load meta.pkl with no version check | Yes | Deferred (§5.5) |
| F5-gapgate-coverage | high | `scripts/train_single_model_m1.py:475-477,520-524,636-638` quarantine (transformer/tcn/histgb); `:548-552,571-575,594-598` lgbm_momentum/lgbm_risk/ridge have NO quarantine; `:779` `GAP_CHECKED_MODELS` excludes them | Yes | Deferred (§5.6) |
| F6-orphans-joint | high | `scripts/train_single_model.py:~151-161` `MODEL_TRAINERS` lacks xgboost/random_forest; `src/scanner/gates.py:469-483` joint fallback reachable when `use_per_pair_routing=True` + no per-pair dir; `:2325-2348` falls back to `self` | Yes | Deferred (§5.7) |

> Note: B1 and B4 overlap on the duplicate `_run_learning_loop` definition (`continuous.py:1966` vs `:2268`); they are tracked as distinct findings because B4 additionally covers the duplicated observation-logging blocks and the doubled `_run_smart_loop()`/`_run_learning_loop()` call sites.

---

## 2. Refuted Findings (`real=false`)

| ID | Confidence | Why refuted (file:line) |
|----|-----------|--------------------------|
| F3-tcn-regime-fields | high | The TCN volatility loader (`src/core/modular_data_loaders.py:3884-4106`, via `train_single_model_m1.py:500-512`) correctly does NOT return `regime_quantiles`/`regime_atr_col`. Those fields belong to the direction pipeline (`load_direction_data`, `:2053-2083`); the regime one-hot + quantile thresholds (`:2065-2083`) are appended ONLY to direction features. TCN is a separate 4-class volatility classifier trained on forward-realized volatility labels (predicting regimes 0-3), not direction-conditioned regimes. Their absence in the TCN loader is by design, not a bug. **No fix required.** |

---

## 3. (intentionally merged into §4)

---

## 4. Applied Fixes

**File:** `src/scanner/automation/continuous.py` · **Diff summary:** 13 insertions, 10 deletions, 1 file. 5 logger calls rewritten + `_error_frequency` moved class-scope → `__init__`. No other code touched.

### 4.1 B2-selfheal-logger-typeerror — APPLIED (all 5 calls)
Converted 5 structlog-style logger calls to stdlib `%`-formatted messages, preserving the same fields:

| Line | Call | Fields preserved |
|------|------|------------------|
| 1134 | `code_repair.deferred` | `error_sig`, `frequency`, `reason` |
| 1166 | `code_repair.success` | `error_sig`, `files`, `fix` |
| 1172 | `code_repair.reverted` | `error_sig`, `reason` |
| 1176 | `code_repair.needs_human` | `error_sig`, `diagnosis` |
| 1181 | `code_repair.failed` | `error_sig`, `error` |

This removes the `TypeError: Logger._log() got an unexpected keyword argument` that would have been raised on the self-heal path. (The deferred-call TypeError fired in the foreground scan thread; the success/reverted/needs_human/failed ones inside `_repair_worker`'s try/except — but all were broken.) These lines emit only status logs around a separately-spawned repair worker and compute/alter no signal, confidence, gate, position size, SL/TP, or risk value — hence `touches_trading_semantics=false`, safe to auto-apply.

### 4.2 B3-error-frequency-classlevel — APPLIED
Moved `_error_frequency: Dict[str, int] = {}` from class scope (was `:1093`) into `__init__` as `self._error_frequency: Dict[str, int] = {}` (after the journal-cache init). Removes the shared-mutable-class-attribute footgun; the dict is now per-instance. All existing `self._error_frequency[...]` access sites were already instance-qualified, so behavior is preserved (and corrected for the multi-instance case).

### 4.3 Verification
- `python -m py_compile src/scanner/automation/continuous.py` → **PASS** (`PY_COMPILE_OK`).
- `python -m flake8 src/scanner/automation/continuous.py --config=.flake8` → exits 1, but **identical to the pre-edit baseline** (8 pre-existing warnings, line numbers shifted +3 by the `__init__` insertion). Number-normalized diff: `FULLY_IDENTICAL_NO_NEW_LINT`. Edits introduce **zero new violations**. The pre-existing `F811 _run_learning_loop` redefinition warning is exactly the duplicate deferred in §5.1 — left in place deliberately.
- Not reverted: the flake8 non-zero exit is entirely pre-existing baseline noise (file was already non-clean on `git stash`), not caused by these edits.

---

## 5. Deferred Fix-Specs (verified real, trading-semantic — NOT auto-applied)

All findings below are `high` confidence and `touches_trading_semantics=true`. They are presented as ready-to-implement specs for operator approval; none has been applied.

### 5.1 B1-dup-learning-loop — duplicate `_run_learning_loop` definition
**File:** `src/scanner/automation/continuous.py`
**What's wrong:** Two identical-signature definitions of `_run_learning_loop` exist at `:1966` (lines 1966-2243) and `:2268` (lines 2268-2366). Python keeps the **second** (`:2268`, the simpler one), making the first dead code.
**Spec (REQUIRES OPERATOR RECONCILIATION DECISION — do not blindly delete):** The two bodies are NOT equivalent. The first def (currently dead) is the richer implementation: ModelManager A/B scoring, AdaptiveRiskScaler, CounterfactualLearner, QA-pipeline audit, full AlertManager + peak-NAV tracking, AccuracyGate merge into `blocked_pairs`, drift detection + `_spawn_background_retrain`, PairModelSelector switching, mtime-cached journal reads. The second def (currently live) is stripped to steps 5a-5f with raw `json.loads(read_text())` and `logger.debug`-swallowed errors. Because the missing components affect signal generation, risk calculation, and position sizing, the operator must decide which feature set should be live BEFORE deletion: deleting the first is runtime-safe but discards the richer behavior; deleting the second changes runtime behavior. This is a reconciliation, not a mechanical dedup.

### 5.2 B4-double-smartloop-obs — observation logging + `_run_smart_loop` run twice per cycle
**File:** `src/scanner/automation/continuous.py`
**What's wrong:** The observation-logging block (`ObservationLog().log_from_analysis()` loop) appears identically at `:517-528` and `:744-754`; `_run_smart_loop()` is called identically at `:553` and `:854`; and `_run_learning_loop()` is called twice at `:1957` and `:1960`. `_run_smart_loop` modifies trades via the OANDA API (drawdown guardian, trailing stops, position management), syncs RL state, and runs the learning loop — so running it twice per cycle corrupts the learning signal and double-applies risk modifications.
**Spec:**
1. DELETE `:517-528` — first observation-logging block (identical to `:744-754`). Retain the `validation_stats` and config-tuning code that follows.
2. DELETE `:553` — first `_run_smart_loop()` call (identical to `:854`).
3. DELETE `:1960` — duplicate second `_run_learning_loop(em, trades_synced)` call (keep `:1957`).
4. (Couples with §5.1) Optionally collapse to one `_run_learning_loop` definition once the §5.1 reconciliation decision is made.
**Verify after:** confirm only one obs-logging block, one `_run_smart_loop()` call, and one `_run_learning_loop()` call execute per scan cycle.

### 5.3 F1-meta-deadwrite — `meta_labeler_threshold` never reaches inference
**File:** `src/scanner/engine.py` (+ `src/scanner/sota_integration.py`)
**What's wrong:** `ScannerConfig.meta_labeler_threshold` (`config.py:563`) is never mapped to `InferenceConfig.min_meta_confidence` (`modular_inference.py:514`), which is the field actually read by the meta gate at `modular_inference.py:4812`. The `InferenceConfig` constructed at `engine.py:1899-1921` (and the legacy path at `sota_integration.py:60-69`) omits it, so the operator-tunable threshold is a dead write — the gate runs on the `InferenceConfig` default.
**Spec:**
- `engine.py:1899-1921`: after line 1920, add `min_meta_confidence=float(self.config.meta_labeler_threshold),` to the `InferenceConfig(...)` construction.
- `sota_integration.py:60-69`: after line 68, add the same `min_meta_confidence=float(...)` mapping.
**Verify after:** round-trip — set `meta_labeler_threshold` in profile/config → confirm `InferenceConfig.min_meta_confidence` carries the value at runtime → confirm the `:4812` gate uses it.

### 5.4 F2-trainer-feature-names — trainers never receive `feature_names`
**File:** `scripts/train_single_model_m1.py`
**What's wrong:** Loaders return `feature_names` in their data dicts (`modular_data_loaders.py:4096` volatility, `:3432` xgboost, `:3630` rf, `:3871` ridge), but the trainer calls at `train_single_model_m1.py:511-512,547,570,593,626-628` never pass them through — breaking the inference-contract requirement that `feature_names` be saved in model meta.
**Spec:**
- `:511-512` (`train_tcn`): pass `feature_names=tcn_data.get("feature_names")`.
- `:547` (`train_lgbm_momentum`): add `feature_names=data.get("feature_names")`.
- `:570` (`train_lgbm_risk`): add `feature_names=data.get("feature_names")`.
- `:593` (`train_ridge`): add `feature_names=data.get("feature_names")`.
- `:626-628` (`train_histgb`): add `feature_names=dir_data.get("feature_names")`.
**Verify after:** confirm each trainer signature accepts `feature_names` and that saved meta sidecars contain an ordered `feature_names` key.

### 5.5 F4-contract-version-asymmetry — only the transformer head enforces `feature_pipeline_version`
**File:** `src/scanner/gates.py`
**What's wrong:** The `feature_pipeline_version` match check exists ONLY in `_load_transformer_contract` (`:1290-1303`). The other four heads load their meta.pkl with no version validation: `_load_tcn_volatility` (`:627`), `_load_ridge_confidence` (`:1127`), `_load_rf_risk` (`:1173`), `_load_meta_labeler` (`:1329`). A train↔runtime pipeline divergence therefore silently feeds OOD features into the momentum/confidence/risk/meta-labeler gates.
**Spec:** In `_load_tcn_volatility` (after `:666`), `_load_ridge_confidence` (after `:1161`), `_load_rf_risk` (after `:1187`), and `_load_meta_labeler` (after `:1349`): (1) read `feature_pipeline_version` from meta.pkl; (2) compare against `_LOADER_PIPELINE_VERSION`; (3) on mismatch, `log.error(... "Refusing to use this model")` and set the model to `None`, mirroring the transformer pattern at `:1295-1302`.
**Verify after:** load a deliberately version-mismatched artifact for each head and confirm refusal + error log.

### 5.6 F5-gapgate-coverage — three trainers skip the 10% gap quarantine gate
**File:** `scripts/train_single_model_m1.py`
**What's wrong:** `_quarantine_if_overshipped()` is called for transformer (`:475-477`), tcn (`:520-524`), and histgb (`:636-638`), and `GAP_CHECKED_MODELS` (`:779`) is `{transformer, tcn, histgb}`. The lgbm_momentum (`:548-552`), lgbm_risk (`:571-575`), and ridge (`:594-598`) functions return `{train_acc, val_acc}` but never call the quarantine gate — so an over-gap momentum/risk/confidence head can ship past the hard 10% ship rail.
**Spec:** After `trainer.save()` in each function, capture metrics and call `_quarantine_if_overshipped([save_path], train_acc, val_acc, instrument, <name>)` before return (mirroring `:550-552`):
- `train_lgbm_momentum` (`:548`): name `"lgbm_momentum"`.
- `train_lgbm_risk` (`:571`): name `"lgbm_risk"`.
- `train_ridge` (`:594`): name `"ridge_confidence"`.
- Add these three model names to `GAP_CHECKED_MODELS` (`:779`).
**Verify after:** force a >10% gap on each and confirm artifacts move to `_quarantine/`.

### 5.7 F6-orphans-joint — joint fallback still reachable; orphan trainers missing
**File:** `src/scanner/gates.py` (+ `scripts/train_single_model.py`)
**What's wrong:** `MODEL_TRAINERS` in `train_single_model.py` (`~:151-161`) lacks `xgboost_trainer` and `random_forest_trainer` entries. The joint fallback branch in `gates.py:469-483` is still reachable when `use_per_pair_routing=True` and no per-pair directory exists, caching `self` and returning it; `evaluate_all_gates` (`:2325-2348`) falls back to `self` when routing finds no per-pair match — contradicting the joint-fallback deprecation sequence in `.claude/rules/improvement.md`.
**Spec (interim fix carries a known breaking risk — operator decision required):** The clean fix completes deprecation step 3 first (an `engine.py` startup filter that drops pairs without per-pair coverage). A minimal interim change at `gates.py:482` — replacing `self._pair_evaluators[instrument] = self; return self` with `raise ValueError(f"Instrument {instrument} has no per-pair models and joint fallback is deprecated...")` — will refuse correctly but **break existing workflows until the startup filter lands**. Lower-risk alternative: add a pre-gate-load validation step that prevents `models/joint/` from being populated with stale xgboost/rf models. Recommend sequencing behind the deprecation steps already tracked in `improvement.md` rather than applying the raise standalone.

---

## 6. Caveats & Confidence Calibration

- **All 11 findings carry `high` verification confidence** in the source inputs — there are no low-confidence verifications in this batch. F3 is `high`-confidence *refuted* (by-design), not unverified.
- **Applied fixes (§4) are verified:** `py_compile` passes and flake8 introduces zero new violations vs. the pre-edit baseline. The fixes are non-trading-semantic and git-reversible.
- **All §5 deferred fixes touch trading semantics** (signal generation, gate thresholds, risk/position sizing, or the ship/inference contract) and are intentionally NOT auto-applied. They require operator review before implementation.
- **§5.1 (B1) and §5.2 (B4) carry reconciliation/behavioral risk:** the two `_run_learning_loop` bodies differ materially, so a naive "delete the dead def" would silently discard the richer learning/risk/sizing behavior. The operator must decide which feature set is canonical before any deletion.
- **§5.7 (F6) carries a known breaking risk:** the interim `raise` will break per-pair-less instruments until the `engine.py` startup filter (deprecation step 3 in `improvement.md`) ships. Do not apply the raise standalone.
- The pre-existing flake8 baseline for `continuous.py` (8 warnings, incl. the `F811` duplicate-definition for the B1/B4 finding) was tolerated as-is and not "fixed while here," per scope discipline.
