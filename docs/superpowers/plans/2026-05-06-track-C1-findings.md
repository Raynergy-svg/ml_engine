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
