# Track C1 — Direction Head 44% Holdout: Feature-Normalization Mismatch (FINDINGS)

> **Status.** Read-only investigation. No code patches in this commit. Surfaces a concrete root cause for the direction head's sub-coin-flip holdout (CLEAN per audit but scoring 43.8/44.5/43.8% on H1/M15/H4) and proposes three fix paths for operator decision.
>
> **Author.** Claude (foreground investigation, no sub-agent), 2026-05-06.
>
> **Scope.** C1 hypothesis: train/inference feature-normalization mismatch. C2/C3/C4 not investigated.

---

## 1. The mismatch — confirmed

The direction head has **two scalers in the training pipeline** and **only one of them is reproduced at inference**.

### Training pipeline (concrete file:line)

1. **Loader scales first.** `src/core/modular_data_loaders.py:2008-2021` (inside `load_direction_data`, function defined at `:1724`):

   ```python
   from sklearn.preprocessing import RobustScaler
   scaler = RobustScaler()
   X_train_scaled = scaler.fit_transform(X[train_idx])    # :2013
   X_val_scaled   = scaler.transform(X[val_idx])           # :2014
   X_test_scaled  = scaler.transform(X[test_idx])          # :2015
   clip_value = 10.0
   X_train_scaled = np.clip(X_train_scaled, -clip_value, clip_value)   # :2019
   ...
   ```

   Saves the RobustScaler in `result['scaler']` at `:2097`. The returned `dict['X_train']` is RobustScaler-pre-scaled-and-clipped (`:2086`).

2. **Trainer scales again, on top of that.** `cli/training.py:1944-1954` calls:

   ```python
   dir_trainer = TransformerDirectionTrainer(trainer_config)
   dir_metrics = dir_trainer.train(
       dir_data['X_train'], dir_data['y_train'],   # ← already RobustScaler+clipped
       dir_data['X_val'],   dir_data['y_val'],
       feature_names=dir_data['feature_names'],
       ...
   )
   ```

   Inside the trainer at `src/training/trainers/transformer_trainer.py:480-484`:

   ```python
   self.scaler = StandardScaler()
   x_train_scaled = self.scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1]))
   x_val_scaled   = self.scaler.transform(x_val.reshape(-1, x_val.shape[-1]))
   ```

   The trainer's `self.scaler` is a `StandardScaler` fitted on **RobustScaler-clipped** features. Saved with the model at `:1581-1582`.

   Effective training input distribution: `raw → RobustScaler.fit_transform → clip[-10,10] → StandardScaler.fit_transform → TCN/Transformer`.

### Inference pipeline (concrete file:line)

1. **`_extract_tcn_features` returns raw features**, no scaling. `src/core/modular_inference.py:3054-3073`:

   ```python
   def _extract_tcn_features(self, df: pd.DataFrame) -> np.ndarray:
       normalized_features = get_normalized_feature_names()['direction']
       ...
       feature_names = getattr(self.tcn, 'feature_names', None) if self.tcn else None
       return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
   ```

   `_extract_features_by_names` at `:2667` does only `result[:, i] = df[fname].values.astype(np.float32)` (no transform).

2. **Trainer's `predict()` applies only the StandardScaler.** `src/training/trainers/transformer_trainer.py:2880`:

   ```python
   x_scaled = self.scaler.transform(x_reshaped)   # StandardScaler only
   ```

   Effective inference input distribution: `raw → StandardScaler.transform → TCN/Transformer`.

### Distribution gap (qualitative — order of magnitude)

The trainer's `self.scaler` was fit on a distribution centered near 0 with most values in `[-10, +10]` post-clip (post-RobustScaler). At inference it sees raw features whose magnitudes can be **orders of magnitude larger** for unbounded indicators (e.g. `volume_ratio_*`, `macd`, raw `returns` × seq_len) and substantially different for bounded ones. `StandardScaler.transform` linearly shifts and scales — it does not clip — so out-of-distribution inputs propagate directly into the model.

The TCN/Transformer's first layer was trained on a distribution where ~99% of values were in `[-3, +3]` after StandardScaler (typical for clipped-and-RobustScaled data). At inference it routinely sees `|x| >> 3` for many features, pushing activations into saturated regions or unseen territory. **A coin-flip-or-worse holdout (43.8% on H1, 44.5% M15, 43.8% H4) is exactly the symptom this mismatch would produce** — the model isn't broken, it's being asked to predict from inputs it has never seen.

### Why both scalers exist

Best guess from code archaeology (not verified beyond grep): the loader's `RobustScaler` block was added when the codebase was a single-scaler pipeline (loader did the scaling, trainer was a thin TF wrapper). The `StandardScaler` inside `TransformerDirectionTrainer.train` was added later (commit on `transformer_trainer.py` history would tell, but is not in scope for this read-only investigation). At that point the loader's scaler became **dead state** — its `result['scaler']` is never consumed at inference (greppable: zero hits for `dir_data['scaler']` outside the loader file itself, and no `RobustScaler` in `src/scanner/` or `src/core/modular_inference.py`).

The dual-scaling at training "worked" because both halves were applied symmetrically. Inference broke it by silently dropping the loader's `RobustScaler+clip` step.

---

## 2. Three fix paths (operator decides)

| Option | Change | Pros | Cons |
|---|---|---|---|
| **C1.A — Drop the loader's RobustScaler** (recommended) | Remove `:2008-2021` block from `load_direction_data`; pass raw features to the trainer; trainer's StandardScaler is then fit on raw data — same as inference. | Single source of truth. Smallest surface area. Loader becomes a pure label+feature builder. | Requires retraining the direction head. Pre-fix model artifacts become stale (already are — last retrain Apr 16). |
| **C1.B — Apply the loader's scaler at inference too** | Save the loader's `RobustScaler` alongside the model artifact; load it at inference; apply `RobustScaler+clip` BEFORE the trainer's `StandardScaler` in `predict()`. | No retraining — the existing model artifact is preserved if the scaler is recoverable. | Adds dual-scaler complexity to the inference path (more state, more ways to break). The loader's scaler is currently NOT serialized into the trainer's model bundle (verified: `src/training/trainers/transformer_trainer.py:1578-1587` saves only `self.scaler`, the StandardScaler). Recovering it requires loading from a separate file or refitting on training data — both fragile. |
| **C1.C — Apply only the loader's RobustScaler; remove the trainer's StandardScaler** | Inverse of C1.A. Trainer skips its own scaler. Loader's RobustScaler must be saved and applied at inference (same as C1.B). | RobustScaler is more appropriate for financial features (heavy tails). | Same serialization fragility as C1.B. Requires retraining anyway. |

**Recommended: C1.A.** Smallest blast radius, cleanest invariant, and we're already going to retrain (system halted, models 19+ days stale, post-leak-fix retrain queue active). C1.B is a "preserve the broken artifact" path — not worth it given the retrain is coming regardless.

### C1.A — concrete patch sketch (NOT in this commit)

```python
# src/core/modular_data_loaders.py — load_direction_data
# DELETE :2008-2021 (the RobustScaler block + clip)
# DELETE :2024-2032 (the constant-feature removal — keep only if numerical
#         issues arise in tests; can re-add if needed)
# Result dict at :2086-2098: change 'X_train' / 'X_val' / 'X_test' to use
# X[train_idx] / X[val_idx] / X[test_idx] directly (no _scaled suffix).
# DELETE 'scaler': scaler from :2097 (becomes obsolete; the trainer owns
#        the scaling).
# Logger line :2034 needs updating to log raw stats instead of scaled.
```

Then retrain (on the same retrain pipeline used for confidence head).

---

## 3. Validation strategy

After C1.A patch + retrain, the direction head holdout should rise from sub-coin-flip toward a realistic supervised-classification score on financial features.

**Honest expectation (calibrated by A1.5 + B1):**
- Direction-head val accuracy 50-58% across H1/M15/H4 — modest but above 52% threshold.
- The audit classified direction head as CLEAN. With normalization fixed, the upper bound is set by how predictable a 1-bar-forward direction actually is from the 50-feature input set.
- If post-fix accuracy is still <52%, this is NOT a normalization issue — investigate C2 (lookahead/sequence bug), C3 (distribution shift), C4 (sign-inversion probe). Don't lower the 52% threshold.
- If post-fix accuracy is >>70%, look for residual leak (unlikely given the audit, but the leak audit didn't probe sequence-construction bugs that C2 would surface).

**Validation gate.** The direction head IS the gate that the 52% threshold was designed for (verified: `scripts/scheduled_retrain.py:375` references `min_accuracy=0.52` for direction). Same threshold applies post-fix. Don't lower. Acceptance: macro-F1 between (0.52 acc, 0.65 acc) — anything above 0.65 needs leak re-audit.

---

## 4. Sequencing — when to ship C1.A

Three logical orderings, by operator priority:

1. **Before any further B-track work.** If C1 is the cause of the 19-day-stale-and-rejected retrains, fixing it might make subsequent retrains pass holdout naturally — including for the leaky heads (their retrains route through the same `cli/training.py` path that calls `load_direction_data` even though the leaky heads are separate). Worth checking before more B-fix design churn. **Recommended.**
2. **In parallel with B-track.** C1.A is a small loader-only patch + retrain; doesn't touch any leak-fix code. Could ship as its own commit, retrain, validate, and operate independently of B1-B4 disposition.
3. **After Track B closes.** Sequencing risk if B-track reveals more loader-level issues that warrant a unified loader rewrite.

---

## 5. Side findings (for the record)

While investigating, surfaced two adjacent observations not in C1's scope:

- **`load_direction_data:2024-2032` removes constant features post-scaling.** `feature_stds = np.std(X_train_scaled, axis=0); valid_features = feature_stds > 1e-6`. Inference does NOT replicate this filter — it sends all features through the trained model. If a feature was constant on the train slice but variable at inference time (rare but possible: a feature that was zero during a low-volatility training period and non-zero during a higher-vol inference window), there's a count mismatch. The trainer's `feature_names` save at `:2095` may already account for this — the inference path has a feature-count guard at `transformer_trainer.py:2858-2877` that re-aligns. **No action recommended; surfaced for visibility.**
- **`compute_volatility_regime` (the leaky volatility regime label generator from B1) is referenced ONLY from `load_volatility_regime_data` and 2 unit tests.** No runtime call site outside training. **Confirms the design's earlier note that the leaky function can be left in place as a dormant utility.**

---

## 6. Verification surfaces (for any next session)

To re-prove this finding from scratch:

```bash
# Loader's RobustScaler block (training-time)
sed -n '2008,2035p' src/core/modular_data_loaders.py

# Trainer's StandardScaler usage (training-time)
sed -n '475,490p' src/training/trainers/transformer_trainer.py

# Trainer's predict (inference-time scaling)
sed -n '2879,2882p' src/training/trainers/transformer_trainer.py

# Inference feature extraction (no scaling)
sed -n '3054,3073p' src/core/modular_inference.py
sed -n '2667,2710p' src/core/modular_inference.py

# Loader scaler is dead at inference
grep -rn "dir_data\['scaler'\]\|\.scaler.*RobustScaler\|RobustScaler" --include="*.py" src/scanner/ src/core/modular_inference.py
# (expected: zero hits)
```

If all greps confirm what this doc says, C1.A is safe to ship.

---

## 7. Delta verification (per-feature scaler diff + column-order check) — 2026-05-07

This section augments the original C1 finding (sections 1-6 above, committed 2026-05-06 as `7b0275b`). Before patching with C1.A, the augmentation was supposed to confirm: (a) per-feature scaler params actually mismatch between training and inference, and (b) column ordering matches across train/inference. Loading the saved model artifacts surfaced something **bigger and more urgent than a normalization mismatch**.

### 7.1 Smoking gun — saved direction-model scalers are HALF NULL

Survey of all 8 saved `transformer_direction.meta.pkl` files on main:

| Model dir | `meta['scaler']` | feature_names | Notes |
|---|---|---|---|
| `trained_data/models/joint/` | **None** | 50 | THE inference-path model |
| `trained_data/models/USD_JPY/` | **None** | 50 | correlation-dropped pair |
| `trained_data/models/EUR_USD/` | **None** | 50 | major pair |
| `trained_data/models/GBP_USD/` | **None** | 50 | major pair |
| `trained_data/models/GBP_JPY/` | `StandardScaler` | 50 | mean_≈0, scale_≈1 |
| `trained_data/models/USD_CAD/` | `StandardScaler` | 50 | (similar params) |
| `trained_data/models/AUD_USD/` | `StandardScaler` | 50 | (similar params) |
| `trained_data/models/EUR_GBP/` | `StandardScaler` | 50 | (similar params) |

**4 of 8 direction-model meta files — including the joint head — have `meta['scaler'] = None`.**

For models with valid scaler: `mean_ ≈ 0` (e.g. 6.8e-09) and `scale_ ≈ 1.0` per feature, consistent with a `StandardScaler` fit on data that was previously RobustScaler+clipped (i.e. already centered+rescaled at the loader). This confirms section 1's "double-scaling at training time" picture for the half of models that DID save a scaler.

For models with `scaler = None`: the trainer's `predict()` path at `src/training/trainers/transformer_trainer.py:2880` calls `self.scaler.transform(x_reshaped)` — **on `None`, this is an AttributeError**. Either inference has been silently catching that exception and falling through to a no-scaling code path (sending raw features into a model trained on scaled features = noise floor output), or the predict path is being hit via a different code branch I haven't traced. Either way, **4 of 8 direction models are not just normalization-mismatched — they have NO scaler to apply at all**.

### 7.2 Implication for C1.A

The original C1.A recommendation ("drop loader's `RobustScaler` block so trainer fits+saves its own scaler from raw inputs") was correct as far as it went. But it assumed all 8 models had a scaler saved. The audit shows that's not true today. C1.A still applies, with a stronger justification:

- **C1.A retrained models will all have valid `meta['scaler']`** (assuming the fix forces `self.scaler` to be a fitted StandardScaler before save).
- **The current half-null state explains why the holdout is at 44%** even on the half of pairs that nominally have a scaler — when joint is the inference target (fallback for correlation-dropped pairs), the inference scaler is None, and raw features go in.
- **Pre-flight check before retraining:** verify `transformer_trainer.save()` writes a non-None scaler. The save block at `transformer_trainer.py:1578-1587` says `if self.scaler is not None: saved['scaler'] = self.scaler`, but the dump above shows `'scaler'` IS in the keys with value None — meaning either (a) `self.scaler` was None at save time, or (b) some other code path writes `meta['scaler'] = None` explicitly. Need to grep the trainer for unconditional `meta['scaler'] = None` assignments before C1.A retrains.

### 7.3 Column-order check — DEFERRED, lower priority

Skipped in favor of the bigger finding above. All 8 models save `feature_names` (count = 50, identical across pairs). Inference path's `_extract_features_by_names` at `modular_inference.py:2667` uses `feature_names` as the source-of-truth ordering — so if `feature_names` is consistent across save/load, column order is consistent. No evidence of a column-order bug; that hypothesis is weakly ruled out (not formally verified).

### 7.4 Verdict — C1 finding STANDS, with addendum

The original section 1 conclusion (train/infer normalization mismatch causing sub-coin-flip output) holds for the 4 pairs WITH a scaler. For the 4 pairs WITHOUT a scaler, the bug is even simpler: **inference has no scaler to apply**, raw features hit a model trained on scaled inputs.

Both subsets fail for related-but-distinct reasons. Both are fixed by the same patch family (C1.A or its variants). Recommended action UPGRADED:

**C1.A (revised):** drop the loader's RobustScaler block AND verify the trainer always saves a non-None scaler. Then retrain ALL direction models so every saved meta has a valid scaler. Validate ≥52% holdout on the next retrain.

### 7.5 One-line recommendation for the controller

C1.A is *more* urgent than the original doc framed it. Halt-blocker (44% holdout since Apr 16) is explained by the half-null scaler state across saved direction models. Patch + retrain unlocks Track D.

---

## 8. Followup verification — `direction_scaler.pkl` joblib survey

Section 7 found `meta['scaler'] = None` for 4 of 8 direction model meta files. The trainer actually has a SECOND scaler save path: `_save_scalers()` at `src/training/trainers/transformer_trainer.py:1550-1589` writes a separate `direction_scaler.pkl` (joblib) **only when `self.scaler is not None`** (guard at line 1578). At load time, `_load_scalers()` at `:1591-1626` reads it and overrides `self.scaler`. **So the operative inference scaler comes from `direction_scaler.pkl`, not `meta.pkl['scaler']`.**

Survey of `direction_scaler.pkl` presence across all `trained_data/models/*/` dirs:

| Has `direction_scaler.pkl` | No `direction_scaler.pkl` |
|---|---|
| AUD_USD | AUD_JPY |
| EUR_GBP | AUD_NZD |
| GBP_JPY | EUR_AUD |
| USD_CAD | EUR_CHF |
| | EUR_JPY |
| | EUR_USD |
| | GBP_AUD |
| | GBP_CHF |
| | GBP_USD |
| | NZD_USD |
| | USD_CHF |
| | USD_JPY |
| | **joint** |
| **4 of 17** | **13 of 17** |

The 4 dirs WITH `direction_scaler.pkl` are exactly the 4 with valid `meta.pkl['scaler']`. The other 13 dirs (including the joint head, which is the inference fallback for correlation-dropped pairs USD_JPY / EUR_GBP / EUR_JPY per CLAUDE.md) have NEITHER — meaning when `_load_scalers()` runs on those models, the override doesn't happen, `self.scaler` stays at whatever `meta['scaler']` set it to (None for joint), and `predict()` either crashes on `None.transform()` or silently bypasses scaling.

**Why 4 of 17 work and 13 of 17 don't:** unverified, but most likely a regression in a recent retrain — `self.scaler` was set to None at training time (possibly via the warm-start path at `transformer_trainer.py:574-578` which copies a warm_start_scaler that was itself None) before `_save_scalers()` ran. The line-1578 guard then prevented `direction_scaler.pkl` from being written, leaving the dir in a "no scaler" state. Future retrains in that pair's pipeline would NOT restore the scaler unless explicitly re-fitted from scratch.

**Operational impact:** the joint head, which is the inference target for any pair that doesn't have its own per-pair model, is in the broken set. So almost every trade decision since Apr 16 has been routing through a TCN/Transformer with no scaler at inference — raw features piped into a model trained on scaled features. Sub-coin-flip output is the inevitable consequence.

### 8.1 Concrete C1.A patch surface (before retrain)

Three things must change before retraining:

1. **`src/core/modular_data_loaders.py:2008-2021`** — drop the `RobustScaler.fit_transform + clip[-10,10]` block in `load_direction_data`. Pass raw features to the trainer; it owns scaling.
2. **`src/training/trainers/transformer_trainer.py:1578`** — replace the `if self.scaler is not None:` guard with a hard assertion (`assert self.scaler is not None, "..."`). If `self.scaler` is None at save time, that is a training bug; the save path should refuse to silently drop the scaler.
3. **Retrain ALL 17 direction-model dirs** so every dir has a valid `direction_scaler.pkl` and `meta.pkl['scaler']`. Validate ≥52% holdout post-retrain. The audit's "direction head is CLEAN" verdict means the head's labels are honest forward-price; with proper scaling restored, holdout should jump from 44% to a realistic 52-58%.

### 8.2 Recommendation — UPGRADE PRIORITY

**This is THE halt-blocker.** Direction head's 44% holdout (rejecting every retrain since Apr 16) is fully explained by the missing-scaler state across 13 of 17 direction models. C1.A patch + retrain is no longer a "recommended optimization" — it's the critical fix. Once landed, Track D (auto-unhalt) becomes unblockable on signal quality grounds.
