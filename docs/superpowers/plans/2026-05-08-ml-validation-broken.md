# ML Validation Pipeline — 3 Compounding Bugs (Discovery 2026-05-08)

> **Status.** None of the M15 holdout numbers we've celebrated tonight (70.0%, 70.2%, 72.2%, 70.5%) are valid signal. They were the all-SHORT fallback predictor's accuracy on whatever the test slice's class balance happened to be. The trained model has never actually evaluated in any holdout.
>
> **Significance.** This is a partner-mode honest-accounting moment. The C1.A patch + Option G + primary_tf alignment + 65k-candle wider-window — every "fix" we shipped tonight was real and well-reasoned, but the holdout that "validated" each was bypassing the trained model entirely.

---

## Bug 1: Holdout passed single-row DataFrames to evaluate_transformer (FIXED — commit `deedf76`)

`scripts/scheduled_retrain.py:452` (pre-fix) iterated with `row = features.iloc[i : i + 1]` (1 row).

`src/scanner/gates.py:1629` requires `seq_len=60` rows of context:
```
if len(features) < seq_len:
    return None, 0.5
```

1 row < 60 → transformer returns None → `evaluate_all_gates` falls back at `:1822` to `direction = momentum > 0.5 ? "LONG" : "SHORT"`. With default momentum near 0, fallback was always "SHORT". **The "70% M15 holdout" was the all-SHORT fallback's accuracy on a SHORT-heavy test slice.**

Fix: pass `features.iloc[i - SEQ_LEN + 1 : i + 1]` (60-bar windows).

## Bug 2: GateEvaluator default `use_per_pair_routing=False` (PARTIAL — explicit flag needed)

CLAUDE.md design says auto-enabled when per-pair dirs exist. Verified via probe: `getattr(evaluator, 'use_per_pair_routing', '?')` returns `False` by default.

Even with Bug 1 fixed, eval was using the **scaler-null joint model** instead of the C1.A-fixed per-pair models.

Fix: `GateEvaluator(model_dir=..., use_per_pair_routing=True)` explicitly.

## Bug 3: Feature count mismatch — 50 trained, 59 at inference (OPEN)

After fixing Bugs 1+2, the per-pair model still returns None. Direct trainer.predict() shows:
```
Feature selection failed: index 58 is out of bounds for axis 1 with size 58
ValueError: X has 58 features, but StandardScaler is expecting 50 features as input.
```

The trainer saved `feature_names` with 50 features. `compute_normalized_features` at inference produces 59 columns. The trainer's predict() at `src/training/trainers/transformer_trainer.py:2860-2877` tries feature selection via `self.selected_indices` but the indices are out-of-range.

Likely cause: `compute_normalized_features` was modified between training and inference (extra features added), but the trained model's `selected_indices` references column indices that no longer exist in the feature matrix.

Until fixed: every model.predict() call at runtime AND in eval throws an exception, gets caught by `evaluate_transformer`'s `except Exception`, returns `(None, 0.5)`, falls back to SHORT.

**Implication:** the bot has been unable to actually use any of its trained transformer models since the feature-count drift occurred. Halt=true was the correct posture.

---

## Operator-side accounting

What this means for everything we shipped tonight:

| Commit | Real value | Validation status |
|---|---|---|
| C1.A patch (`e9528fa`) | Fixes scaler-null bug; per-pair models now save proper scalers | Valid — verified directly on disk |
| Option G holdout-lookahead fix (`6477a86`) | Aligns trainer's lookahead=24 with holdout's evaluation horizon | Valid as a CODE fix; the holdout it patches was already broken at a deeper layer (Bug 1) |
| primary_tf alignment (`1a7565c`) | Holdout's primary_tf now tracks `--granularity` | Valid as a CODE fix; same caveat |
| TF switch H1→M15 (`60c8cc7`) | Scanner now configured for M15 | Valid config; runtime untested due to halt=true |
| Pair constraint to EUR_USD,GBP_USD (`f3cf986`) | Bot pair list narrowed | Valid; safety-net config |
| Heuristic bridges (`9024100`) | 4 modules + 25 tests, parity proven | Independently validated |
| News pipeline P1 scaffolding (`f1a45d8`) | Design + stubs | Independently validated |

The infrastructure work all has standalone value. The HOLDOUT NUMBERS were bogus.

---

## What we don't know

After Bug 3 is fixed, what's the actual accuracy of any pair at any timeframe?

The only honest answer: **unknown until the inference pipeline can actually run the model**.

---

## Recommended path forward (operator decides)

1. **Fix Bug 3** — investigate why feature count drifted from 50 (saved) to 59 (current). Either:
   - (a) Force `compute_normalized_features` to return only the 50 features in `feature_names` saved with the model
   - (b) Retrain everything with the current 59-feature output, saving `selected_indices` correctly
   - (c) Bypass feature selection entirely if the model architecture allows it

2. **Re-run holdout per-pair with fixed pipeline** — only THEN do we have honest accuracy numbers

3. **Don't unhalt the bot** until the inference pipeline can actually evaluate the model.

4. **Don't trust the "70% M15" claim** until reproduced through a fixed eval.

---

## Honest accounting for the autonomous run

I committed multiple times tonight claiming "model is validated at 70%". That was wrong. Each retrain DID produce a valid model artifact on disk; the artifact has a valid scaler; the model file has trained weights. What was wrong was the validation that "70% accuracy" claim relied on.

This is the right outcome of the load-bearing-question discipline: keep digging until the test you trust holds up.

The discovery itself was made possible by the per-pair eval script I wrote. Direct probing showed `prob_long = 0.5000` for 100/100 windows, which would have been impossible if the model was actually predicting.

---

## What I'm doing for the rest of the autonomous run

Given the depth of this finding, I'm STOPPING the iteration loop on pair expansion. Continuing to run retrains while the eval is broken just produces more fake numbers.

Instead, until morning I will:

1. Investigate Bug 3 (feature-count drift) — diagnostic only, no code changes without operator approval since this touches inference path
2. Keep halt=true (already correct)
3. Document everything found
4. Surface a clear morning report

The pair-expansion goal ("all 15 must pass") is paused pending Bug 3 resolution. Pretending to make progress while the eval is broken is not partnership.
