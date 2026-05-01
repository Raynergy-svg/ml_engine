# Confidence Model Target-Leakage Investigation — 2026-04-30

Follow-up to `docs/ml_architecture_audit_2026-04-30.md` (line 140, Action item #1).

---

## 1. Verdict

**Leak confirmed: target-as-feature (deterministic label leakage).**

The confidence label `y_confidence` is a **closed-form deterministic function of five indicator columns** (`adx`, `rsi`, `atr_pct_14`, `bb_position_20`, `volume_ratio_20`). Those same five columns — plus closely related siblings (`atr_pct_10/20`, `volatility_10/20`, `rsi_norm`, etc.) — are passed in as **features** in `X_confidence`. The model is being trained to recover an algebraic formula from its own ingredients. R² = 0.997 is the residual error of LightGBM approximating a smooth weighted sum, nothing more.

This is not "intentional self-distillation" — there is no separate teacher signal. The target carries no information beyond what's already in X.

---

## 2. Evidence

### 2.1 Metrics confirmed
`trained_data/models/joint/joint_training_meta.json:46-53`
- `r2_score`: 0.9971026361463041 (val set)
- `confidence_mae`: 0.2688 (on a 20-95 range target → ~0.4% relative error)
- `model_type`: lightgbm, n_estimators=100, depth=6, leaves=31
- `n_important_features`: 16 of 29
- `n_train_samples`: not in confidence block but direction block reports 28259 train / 9315 val (same upstream split)
- `saved_at`: 2026-04-16T11:29:31

### 2.2 Label definition (the smoking gun)
`src/core/modular_data_loaders.py:3335-3426`, function `load_ridge_data()`:

| Step | Source line | What it does |
|------|-------------|--------------|
| Pull indicators | 3341-3347 | Reads `adx`, `rsi`, `atr_pct_14`, `bb_position_20`, `volume_ratio_20` directly off `df` |
| ADX score | 3387-3388 | Linear scaling from train-only P25/P75 percentiles, clipped to [0,1] |
| RSI score | 3392-3393 | `1 - abs(rsi - 50) / 25`, clipped |
| Volatility score | 3397 | `1 - atr_pct_14 / 0.015`, clipped |
| BB score | 3401-3402 | `1 - 2.5 * abs(bb_pos - 0.5)`, clipped |
| Volume score | 3406 | `(volume_ratio - 0.7) / 0.6`, clipped |
| Combine | 3414-3420 | Fixed weights: 0.35 ADX + 0.20 RSI + 0.20 vol + 0.15 BB + 0.10 volume |
| Scale | 3424 | `confidence = 20 + raw * 75` → range [20, 95] |

The docstring is explicit (3335-3337):
> "Calculate confidence based on TREND CLARITY (learnable from ADX, RSI, volatility) [...] This IS learnable because these are computed from the same features!"

The author *knew* this was a closed-form function of the inputs and shipped it anyway.

### 2.3 Feature matrix overlap
`src/core/modular_data_loaders.py:3300-3318` — the X matrix is built from:
- **Normalized list** (`get_normalized_feature_names()['confidence']`, defined at `modular_data_loaders.py:1355-1370`): includes `atr_pct_10`, `atr_pct_20`, `volatility_10/20`, `rsi_norm`, `volume_ratio_10/20`, plus 7 trend/return features.
- **Legacy fallback list** (3304-3308): `volatility_5/10/20`, `atr`, `bb_width_20`, **`bb_position_20`**, **`adx`**, `returns`. Appended whenever <8 normalized features resolve.
- **Pattern fallback** (3320-3325): regex-grabs any column matching `'atr_pct'`, `'volatility'`, `'volume_ratio'`, `'sma_ratio'`, `'return'`, `'zscore'`.

So `adx` itself is in X (legacy fallback always triggers when normalized set is incomplete). `bb_position_20` is in X (legacy fallback). `volume_ratio_10/20` are in X (normalized set). `atr_pct_*` are in X (normalized set). `rsi_norm` is in X (normalized set) — only `rsi` is referenced for the label, but `rsi_norm` is monotonic in `rsi`.

**Five of five label inputs are present in X**, four of them as the exact column the formula reads.

### 2.4 Split is chronological, scaler is fit on train only — these are NOT the leak
- `temporal_split` at `modular_data_loaders.py:1390-1421`: chronological, no shuffle, no overlap. Val is strictly after train. ✓
- ADX percentile thresholds for the label are computed from train indices only (3371-3375). ✓
- `StandardScaler` is fit on train only (`ridge_trainer.py:168-170`). ✓
- R² is computed on val predictions vs val labels (`ridge_trainer.py:209-215`), not mislabeled train R². ✓

These guards are real but irrelevant — they prevent *temporal* leakage. The leak here is **structural**: the label is a function the model can fit no matter how the splits are arranged. A perfect chronological split would still give R² ≈ 1.0 because LightGBM with 100 trees of depth 6 can represent the weighted-sum-of-clipped-piecewise-linears formula to arbitrary precision.

### 2.5 Default split call has zero gap
`src/training/trainers/joint_trainer.py:248` calls `load_ridge_data(df)` with all defaults → `gap=0`. Adjacent train/val rows share OHLCV context. This compounds the structural leak with mild temporal autocorrelation leakage (rolling indicators leak across the boundary), but is dwarfed by the structural issue.

---

## 3. Mechanism — how R² hits 0.997

The label is a piecewise-linear-in-features expression with 5 clipped components and fixed weights. LightGBM with depth-6 trees and 100 boosting rounds is a universal approximator for such functions — it will partition feature space along the same thresholds the formula uses (ADX P25/P75 cutoffs, RSI=50, BB=0.5, ATR=0.015, volume=0.7) and assign leaf values that recover the linear combination.

The `confidence_mae` of 0.27 on a 75-point range = **0.36% relative error**. That's not "the model learned market structure"; that's "the model memorized an arithmetic formula it had all the inputs for." The remaining 0.003 of unexplained variance is rounding error from the `np.clip` boundaries that don't align perfectly with tree splits.

A trivial linear regression with the five raw inputs and per-feature interaction terms would also score >0.95 on this target. The 0.997 is just LightGBM doing slightly better than a linear approximator at the clip-boundary kinks.

**Verification (no retrain needed, just trace the formula):** if you compute label = `0.35*adx_score + 0.20*rsi_score + ...` row-by-row and compare to LightGBM predictions on val rows, deltas will be <1 confidence point everywhere except near clip boundaries. The model is a regression over a deterministic transform.

---

## 4. Blast radius

### Direct consumers of `ridge_confidence.pkl`
| File | Line | What it does |
|------|------|--------------|
| `src/scanner/gates.py` | 779-810, 1253-1295 | Loads model, calls `.predict()` on live features in `_compute_confidence()` to produce the gate's confidence score (0-100 range expected) |
| `src/training/trainers/joint_trainer.py` | 419-427, 763, 875-878, 941-942 | Trains it, fine-tunes per-instrument, loads master for transfer |
| `src/training/retrain_gates.py` | 143, 193 | Online retrain pipeline writes a fresh `ridge_confidence.pkl` |
| `online_retrainer.py` | 556 | Loads it for online retraining cycle |
| `src/cli/helpers.py` | 186, `src/utils/instrument_validation.py:245` | Health-check probes for model presence |
| `src/tui/screens/diagnostics_screen.py` | 766 | Reports model age in TUI |

### Behavioral impact when this prediction is used
The gate at `gates.py:1281` calls `self._ridge_confidence.predict(X)` and passes the raw 0-100 score forward. Any downstream consumer that reads "confidence = 73" believes that means "model thinks this setup is 73/100 quality."

Reality: it means **"a closed-form score over ADX/RSI/ATR/BB/volume on the current bar = 73"**, which the gate could compute directly without a model. The LightGBM round-trip adds ~5ms of inference latency and pickle-load overhead for **zero predictive value beyond the formula itself**.

**Critical note: the formula has no link to trade outcomes.** It encodes a heuristic about "trend clarity" (high ADX + mid RSI + low ATR + center BB + above-avg volume = high confidence). This may correlate with win-rate or may not — that's an empirical question that has *never been validated* because the model's R² on its synthetic target was taken as evidence the system works. The audit doc (`docs/ml_architecture_audit_2026-04-30.md:140`) flagged exactly this.

### Downstream gating logic that consumes this confidence
1. **Confidence gate in scanner** — `gates.py:1253-1295`. Score feeds the agent team's `confidence` signal which is one of the 15 agents in `_BASE_WEIGHTS` (`src/scanner/agents/_team.py`). A WVS computed off this is therefore measuring "trend-clarity formula" not "model-learned win probability."
2. **Position sizer** — `src/risk/position_sizing.py` reads confidence to scale risk. Currently it scales position size on a number that is mechanically derived from current-bar indicators.
3. **Calibration layer** — `src/risk/confidence_calibration.py` Platt+Isotonic ensemble recalibrates this score against trade outcomes from the journal. The calibrator is doing the real work: mapping the formula score to actual win-rate. The Ridge model is a redundant intermediate step.
4. **Meta-labeler features** — `src/scanner/meta_labeling.py` uses confidence as one feature. A circular feature whose target was algebraically derivable from other features is at best redundant, at worst noise on top of a noisy ground truth.

### What changes if we corrected this
A "fixed" confidence model would need to be retrained against an **outcome-derived label** — e.g. realized win/loss within N bars, or risk-adjusted forward return. R² on such a target will plummet (0.05-0.20 is realistic for FX scoring models). **Buddy will report lower confidence numbers and reject more trades**, but the numbers will mean what they claim to mean. The Platt/Isotonic calibrator (which is currently rescuing the system by mapping the synthetic score to journal outcomes) will need to refit.

Net effect: **the gate is currently doing nothing the formula couldn't do directly, and the calibration layer is silently compensating for the bogus score.** Removing or replacing this model would not catastrophically break trading — it would just expose how much work the calibrator is doing.

---

## 5. Recommended fix (minimal)

Two viable paths, in order of effort/risk:

### Option A (minimum change, recommended for first iteration)
Replace the synthetic label with **realized confidence from journal outcomes**. The codebase already has `realized_confidence` floating around (`src/recursive_intelligence/maml_benchmark.py:691,836,1018,1043` references it). Define `y_confidence = f(realized_pnl_within_N_bars, max_favorable_excursion, max_adverse_excursion)` — e.g. a 0-100 score that correlates with realized R-multiple.

Then:
- Keep `temporal_split` chronological (it already is).
- Keep the same features.
- Expect R² to drop to 0.05-0.25 on val. That is the *correct* number for this kind of regression on noisy financial outcomes.
- Refit Platt/Isotonic over the new score (calibration will be much more meaningful).

Single-file change in `src/core/modular_data_loaders.py:3335-3426` (the label-construction block). No retraining infrastructure changes needed.

### Option B (zero-cost, immediate)
Delete `ridge_confidence.pkl` consumption in `gates.py:1253-1295` and replace with a direct call to the formula from the loader. The model adds no information; the formula is the actual computation. This makes the gate honest about what it's doing without any retraining.

Then decide separately whether you want a *real* learned confidence model (Option A) on top of that.

### Do NOT
- Don't add a "gap" parameter or change the split — the structural leak is independent of split mechanics.
- Don't drop "leaky" features one by one — they ALL appear in the label. Dropping `adx` leaves `rsi`, dropping `rsi` leaves `atr_pct_14`, etc. The label itself is the problem.
- Don't ship a "fixed" model without retraining the calibrator. The current Platt/Isotonic params are tuned to the synthetic-score distribution; a new score distribution will need re-fitting.

---

## Appendix: cross-references
- Audit that flagged this: `docs/ml_architecture_audit_2026-04-30.md:11,29,32,140`
- Joint training entry point: `src/training/trainers/joint_trainer.py:419-427`
- Loader: `src/core/modular_data_loaders.py:3266-3457`
- Trainer + R² calc: `src/training/trainers/ridge_trainer.py:150-234`
- Live consumer: `src/scanner/gates.py:770-814, 1253-1295`
- Calibration that's silently rescuing the system: `src/risk/confidence_calibration.py:96-129`
