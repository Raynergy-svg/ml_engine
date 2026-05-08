# Pipeline Reconciliation — Phase 1 Audit

> **Status.** Audit only. No code changed. Documents every confirmed train↔inference skew point and the inference contract that should exist.
>
> **Prior context.** This audit replaces the partial "Bug 3 = feature schema drift" diagnosis from `docs/superpowers/plans/2026-05-08-bug3-feature-schema-drift.md` and `docs/superpowers/plans/2026-05-08-ml-validation-broken.md`. Those documents identified surface-level gaps; this audit traces the full contract violation.
>
> **Bottom line.** The trained model artifact's saved `scaler` is **mathematically the identity transform** (`var_=1.0`, `mean_≈1e-17` for all 50 features). At inference, the keras model receives raw feature values that are 100×–1000× different from the standardized values it was trained on, producing OOD predictions. This is not a "bug 3" — it is a fundamental train↔inference contract failure.
>
> **Confidence:** 99% on root cause (scaler refit at line 2816). Empirically reproduced.

---

## 1. Root cause: scaler is fit twice, second fit zeros out the useful one

### Code path (current, post-C1.A)

```
joint_trainer.JointTrainer.train(...)
  → load_direction_data(df, apply_scaler=False)            # raw features returned
    → compute_normalized_features(df)                      # 66 cols (60 features + 6 raw)
    → exclude {open,high,low,close,volume,time,...}        # 60 features
    → variance/correlation filter                           # ~60 features (most kept)
    → regime augmentation (X = concat with 4 one-hots)     # 64 features
    → return X (raw values)
  → TransformerDirectionTrainer.train(X_train, y, ...,
                                      skip_scaling=False)
    → _scale_features(X_train, x_val, skip_scaling=False)
        line 480: self.scaler = StandardScaler()
        line 481: x_train_scaled = self.scaler.fit_transform(X_train_raw)
                  # x_train_scaled now has mean=0, std=1; scaler has REAL stats
    → _handle_feature_preparation(x_train_scaled, ...)
        line 2811: _apply_feature_selection(x_train_scaled, ...)
                   # subsets to 50 selected_indices, returns subset of x_train_scaled
        line 2814: if self.scaler is not None:
        line 2816:   self.scaler = StandardScaler()           ← REPLACES fitted scaler
        line 2817:   x_train_scaled = self.scaler.fit_transform(x_train_scaled)
                     # FIT FRESH ON ALREADY-SCALED DATA
                     # result: var_=1.0 exactly, mean_≈1e-17, scale_=1.0
    → save() at end
        # writes the IDENTITY scaler to .meta.pkl
```

### Empirical reproduction (no model run, just math)

```
StandardScaler #1: fit(raw_features)         → mean_≈feature_means, scale_≈feature_stds
                   transform(raw)             → mean=0, std=1 per column
StandardScaler #2: fit(scaler1_output)        → mean_≈0,         scale_=1.0  (var_=1.0)
                   transform(scaler1_output)  → ≈ identity (within floating point)
```

A reproduction script (Python, sklearn) shows the exact `var_=1.0, mean_≈1e-17` fingerprint
matching `trained_data/models/EUR_USD/transformer_direction.meta.pkl`.

### Inference consequence

```
gates.evaluate_transformer:
  features = compute_normalized_features(df)    # raw values
  X = features[numeric_cols].iloc[-60:].values  # raw, ~e-3 to e0 magnitudes
  saved_scaler.transform(X)                      # ≈ identity, X unchanged
  keras.predict(X)                               # model expects N(0,1), sees raw → OOD
```

For EUR_USD M15 returns the magnitude gap is roughly:
- `returns_5` raw std ≈ 1e-3
- training-time scaled value ≈ N(0, 1) (range ~[-3, 3])
- inference-time value ≈ raw (range ~[-3e-3, 3e-3])
- model sees inputs **1000× smaller** than training distribution
- prediction collapses toward whatever the bias term + saturation produce → not random, but uncorrelated with actual signal

This is why every M15 holdout we've run since C1.A landed has reported the **fallback predictor's accuracy** (a constant SHORT or LONG, depending on the test slice's class balance) — when `evaluate_transformer` returns `(None, 0.5)` after the keras layer detects shape/distribution issues, the gate evaluator falls back at `gates.py:1822` to `direction = momentum > 0.5 ? "LONG" : "SHORT"`, which collapses to a constant for the default momentum.

---

## 2. Train↔Inference skew points (complete map)

| # | Skew point | Training does | Inference does | Severity | Status |
|---|---|---|---|---|---|
| 1 | **Scaler is identity** | Two StandardScaler.fit calls; second fits on already-scaled data → identity | Applies identity scaler to raw features | **CRITICAL** | Confirmed |
| 2 | **OHLCV columns** | Excludes `{open,high,low,close,volume}` from feature matrix | `select_dtypes([np.number])` includes them → 5 extra cols | **CRITICAL** | Confirmed |
| 3 | **Regime one-hots** | Appends 4 cols computed from `np.percentile` over full corpus ATR | Doesn't compute or append → 4 missing cols | **CRITICAL** | Confirmed |
| 4 | **Column ordering** | Uses `feature_names` order saved in meta | Uses `compute_normalized_features` insertion order (different) | **CRITICAL** | Confirmed |
| 5 | **Feature selection** | Trainer selects 50 of 64 via RF importance; saves `selected_indices` | Doesn't apply selection; passes all numeric cols | **CRITICAL** | Confirmed |
| 6 | **`feature_names` plumbing** | Trainer saves `feature_names` to `.meta.pkl` | `gates._load_transformer` (line 1016-1053) loads `.keras` only; meta never read | **CRITICAL** | Confirmed |
| 7 | **Regime quantiles** | Computed at training via `np.percentile(_atr_valid, [25,50,75])` over training corpus | Not saved in meta; cannot reproduce at inference | **HIGH** (blocks even a fixed regime path) | Confirmed |
| 8 | **GBP_USD scaler=None** | GBP_USD model trained 2026-05-05 (predates C1.A); `scaler=None` in meta | Inference would crash at `scaler.transform()` | **HIGH** (artifact-level) | Confirmed |
| 9 | **Granularity not propagated** | `lineage.granularity = config.granularity or 'H1'`; config not updated for M15 retrain | Lineage shows 'H1' for an M15-trained model | LOW (metadata only; doesn't break inference) | Confirmed |
| 10 | **Time-feature extraction** (FIXED today, fde458a) | Pre-fix: silent `time_col.hour` AttributeError when df had `time` as column | Post-fix: works for time-as-column | RESOLVED | Confirmed |

**Skew points 1–6 individually break inference.** Skew 7 prevents future correct regime computation. Skew 8 means GBP_USD model needs full retrain regardless. Skew 9 is cosmetic. Skew 10 is fixed.

Even if we fix Skew 1 (the scaler bug), Skews 2–6 still produce a column-mismatch shape error or order-misalignment — the keras model would receive 65 cols of misordered raw values instead of 50 cols of correctly-ordered scaled values.

---

## 3. The inference contract — what `transformer_direction.meta.pkl` SHOULD provide

For inference to faithfully reproduce training-time feature distribution, the meta artifact must be a complete contract. Current meta has 15 keys; below are the keys needed (existing keys marked ✓; new keys marked +).

### Required for correctness

| Key | Type | Purpose |
|---|---|---|
| `feature_names` ✓ | `List[str]` (50) | Column names in EXACT order the model expects. Already saved. |
| `scaler` ✓ | `StandardScaler` | Fitted on the 50 final selected features in `feature_names` order, with **real mean/var stats** (not identity). |
| `regime_quantiles` + | `Dict[str, float]` | `{q25, q50, q75}` from training corpus ATR. Required to compute regime one-hots at inference. |
| `regime_atr_col` + | `str` | Which feature column was used for regime computation (`atr_pct_20` or `atr_pct_14`). |
| `feature_pipeline_version` + | `str` | Semver tag for the feature contract. At load, assert `inference_version == artifact_version` or refuse to predict. |
| `seq_len` ✓ | `int` (60) | Already saved. |
| `architecture` ✓ | `Dict` | Already saved. |
| `n_features` ✓ | `int` (50) | Already saved. |

### Existing but unused / stale

- `selected_indices` — becomes unnecessary if scaler is fit on the final 50 selected features (i.e., feature selection happens BEFORE scaler fit, not after). Keep for forensic/debug only.
- `output_calibration` — not in scope of this audit.
- `lineage` — keep, fix granularity propagation in a follow-up.

---

## 4. Inference path that uses the contract (sketch)

The proposed `gates._load_transformer` path:

1. Locate `transformer_direction.keras` and `transformer_direction.meta.pkl` in `self.model_dir`.
2. Load meta via the existing `_load_pickle_quietly` helper used elsewhere in `gates.py` (e.g. line 1083 for the legacy meta-labeler path).
3. Validate `meta["feature_pipeline_version"]` matches the runtime constant; if not, log an error and return `False` (refuse to predict).
4. Store `meta["feature_names"]`, `meta["scaler"]`, `meta.get("regime_quantiles")`, `meta.get("regime_atr_col", "atr_pct_20")` on `self`.
5. Load the keras model via the existing `load_keras_model` helper.

The proposed `gates.evaluate_transformer` path:

1. If transformer or its `feature_names` are missing, return `(None, 0.5)`.
2. Run `compute_normalized_features(features)` to produce the inference DataFrame.
3. For each name in `feature_names` (in order):
   - If present as a column → use it.
   - If a `regime_*` name and `regime_quantiles` are available → compute one-hot from `regime_atr_col` values via the saved `q25/q50/q75`.
   - Else → log error, return `(None, 0.5)`. **No silent zero-fill.**
4. Stack columns → 2-D array of shape `(n_rows, 50)`.
5. Take last `seq_len` rows; sanitize NaN/Inf.
6. Apply `self._transformer_scaler.transform(X)` (the REAL scaler, post-fix).
7. Reshape to `(1, seq_len, 50)`; call the keras predict helper.
8. Return `(direction, prob_long)`.

(Implementation lands in Phase 2.B; not changed here.)

---

## 5. Training pipeline changes required to honor the contract

### Change A — Fix scaler order (one fit, after selection)

In `transformer_trainer.py` lines 2810–2820:

- **BEFORE (buggy):** `_scale_features` fits scaler #1 on raw → `_apply_feature_selection` subsets the scaled data → line 2816 refits scaler #2 on already-scaled subset (var_=1.0 bug).
- **AFTER (proposed):** Defer scaling to AFTER feature selection. RF importance and SelectKBest(f_classif) are scale-invariant, so this is safe. Then a single `StandardScaler.fit_transform(x_train_selected_raw)` produces a scaler with real stats on the 50 final features.

Implementation outline:
1. Skip the line-480 fit when `use_feature_selection=True` (or pass raw to selection).
2. After selection, fit scaler ONCE on raw selected features.
3. Remove the line-2814 to 2819 refit block.
4. Apply the same fix to `_apply_exact_feature_match` (line 577-579) and `_apply_partial_feature_match` (line 619-621) for the warm-start case.

### Change B — Save regime quantiles in meta

In `modular_data_loaders.load_direction_data` lines 1925–1947, capture `_q25, _q50, _q75` and the `_atr_col` name, return them up to the trainer, which embeds them into meta at save time.

### Change C — Save feature_pipeline_version in meta

Add a constant `FEATURE_PIPELINE_VERSION = "2026-05-08-v1"` in `src/core/modular_data_loaders.py`. Bump on any change to `compute_normalized_features` or `load_direction_data` feature columns. Trainer reads at save; gates reads at load.

### Change D — Tripwire assertions

- **Training**: after `_scale_features`, assert no column has `var_ ∈ (1.0 - 1e-9, 1.0 + 1e-9)` for >50% of features (catches the double-fit pattern).
- **Inference**: at `_load_transformer`, assert pipeline version match. At `evaluate_transformer`, assert `X.shape[-1] == len(feature_names)` before predict.

---

## 6. Open questions / things this audit doesn't yet answer

1. **What's GBP_USD's actual training history?** `scaler=None` predates C1.A. We need to retrain it with the fixed pipeline regardless. No need to investigate further — just retrain.
2. **Is the `JointTrainer` flow (joint multi-pair training) the only entry?** I traced via `joint_trainer` and `scheduled_retrain`. There may be a per-pair trainer path used by `autonomous_trainer`. Phase 2 should grep for all `TransformerDirectionTrainer.train()` call sites and apply the contract uniformly.
3. **Does `is_warm_start=True` path produce a similar identity-scaler?** `_apply_exact_feature_match` (lines 577–579) and `_apply_partial_feature_match` (lines 619–621) have the same refit-on-scaled pattern. If we ever warm-start, we hit the same bug. Phase 2 fix should remove all three refit sites.
4. **What's the price-only ceiling on M15 EUR_USD?** Unknown until we have a fixed pipeline + retrain + honest holdout. CLAUDE.md says ~70%; that's based on previous (broken) training runs. Need to re-establish.
5. **Is the regime augmentation lookahead-prone?** `np.percentile(_atr_valid, [25,50,75])` over the FULL training corpus uses test/val data to compute training-time regime classifications (mild leak). Phase 2 should compute quantiles only on the train fold, save to meta, and apply at inference.

---

## 7. Recommended Phase 2 design

**Goal**: lock down the inference contract, fix the scaler bug, and retrain one pair to verify.

**Phase 2.A — Trainer changes (single PR)**:
1. Reorder `_handle_feature_preparation`: feature select first, then scale (single fit, no refit).
2. Capture and persist `regime_quantiles` + `regime_atr_col` from `load_direction_data`.
3. Add `feature_pipeline_version` constant; embed in meta.
4. Add training-time tripwire: warn if `>50%` of features have `var_=1.0±ε`.
5. Update `_apply_warm_start_features` exact + partial match paths to reuse parent scaler subset, never refit.
6. Tests (no mocks, real disk): synthetic data → train → verify saved scaler has real var stats; verify regime_quantiles present.

**Phase 2.B — Inference changes (single PR)**:
1. `gates._load_transformer` reads meta, stores `feature_names`, `scaler`, `regime_quantiles`, `regime_atr_col`.
2. `gates.evaluate_transformer` builds X via `feature_names` order, computes regime via saved quantiles, applies saved scaler.
3. Pipeline-version check at load; column-count assert at predict.
4. Tests (no mocks, real disk): build a real model artifact, run inference, assert deterministic output.

**Phase 2.C — Per-pair holdout eval changes**:
1. `scripts/per_pair_holdout_eval.py` already passes 60-bar windows post-deedf76; verify it uses GateEvaluator with `use_per_pair_routing=True` and that the new contract path runs.

**Phase 3 — Retrain**:
1. EUR_USD M15 65k candles end-to-end with fixed pipeline.
2. Verify saved scaler has real stats.
3. Verify saved meta has `regime_quantiles`, `feature_pipeline_version`.

**Phase 4 — Honest holdout**:
1. Run `scripts/per_pair_holdout_eval.py --pair EUR_USD --granularity M15`.
2. Result should be within ±2pp of `val_balanced_accuracy`. If not, more bugs.

**Phase 5 — Decision based on the real number**:
- ≥58% holdout: pipeline trustworthy, expand to other pairs.
- 52–57%: pipeline correct, price-only ceiling reached, pivot to news/macro fusion.
- <52%: deeper architecture issue.

**Phase 6 — Lock-down**:
1. Promote contract-validation invariants to `.claude/rules/improvement.md`.
2. Document the inference contract in CLAUDE.md.

---

## 8. Confidence calibration on this audit

| Claim | Confidence | Verification source |
|---|---|---|
| Scaler refit at line 2816 is the cause of `var_=1.0 exactly` | **99%** | Code read + synthetic reproduction exactly matches saved stats |
| Inference path doesn't load `feature_names` from meta | **99%** | grep for `_transformer_feature_names` and `transformer_direction.meta` in `gates.py` returns 0 hits |
| Regime quantiles aren't saved | **99%** | Direct meta key inspection |
| GBP_USD has `scaler=None` | **99%** | Direct meta read |
| Training data flow as described (joint_trainer → loader → trainer) | **95%** | Code traced from `scheduled_retrain.py` to `save()`; one untraced path: `autonomous_trainer` |
| Removing the line-2816 refit + saving regime_quantiles is sufficient to make inference correct | **90%** | Logically follows from 1 and 7; needs Phase 3 retrain to confirm empirically |
| Real EUR_USD holdout post-fix lands in 52–60% | **70%** | `val_balanced_accuracy=0.5265` is the prior; assumes no other latent bug |
| Price-only ceiling for M15 is real (~60%) | **60%** | Inference has been broken since C1.A landed; we've never measured it cleanly |

---

## 9. What this audit does NOT do

- **Does not change code.** All findings are read-only.
- **Does not retrain.** That's Phase 3.
- **Does not unhalt.** Bot stays halted until Phase 4 produces an honest holdout ≥ a threshold the operator approves.
- **Does not pursue news/macro fusion.** That's downstream of Phase 5 decision.

---

## 10. References

- `src/core/modular_data_loaders.py:716-1056` — `compute_normalized_features` + time/session features
- `src/core/modular_data_loaders.py:1772-1820` — `load_direction_data` feature column selection
- `src/core/modular_data_loaders.py:1925-1947` — regime augmentation (the source of the missing quantiles)
- `src/core/modular_data_loaders.py:2024-2065` — RobustScaler + clip path (apply_scaler=True only; not used since C1.A)
- `src/training/trainers/joint_trainer.py:175-180` — `apply_scaler=False` site
- `src/training/trainers/joint_trainer.py:385-397` — `skip_scaling=False` site
- `src/training/trainers/transformer_trainer.py:465-486` — `_scale_features` (1st fit)
- `src/training/trainers/transformer_trainer.py:550-626` — `_apply_exact_feature_match` / `_apply_partial_feature_match` (warm-start refit, lines 577-579, 619-621)
- `src/training/trainers/transformer_trainer.py:728-748` — `_update_features_after_selection` (replaces scaler with empty)
- `src/training/trainers/transformer_trainer.py:2810-2819` — **THE BUG**: post-selection refit on already-scaled data
- `src/scanner/gates.py:1016-1053` — `_load_transformer` (no meta read)
- `src/scanner/gates.py:1608-1665` — `evaluate_transformer` (no `feature_names` usage)
- `trained_data/models/EUR_USD/transformer_direction.meta.pkl` — saved scaler with `var_=1.0 exactly`
- `trained_data/models/GBP_USD/transformer_direction.meta.pkl` — `scaler=None`

---

End of Phase 1 audit. Awaiting operator review before Phase 2.
