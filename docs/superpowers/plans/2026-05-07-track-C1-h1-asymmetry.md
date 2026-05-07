# Track C1 — H1 Asymmetry Investigation

> **Status.** Read-only investigation. Smoking gun found. C1.A patch is correct; H1 holdout's persistent 43% is a SECOND, INDEPENDENT bug — a hardcoded lookahead mismatch between trainer and holdout validator.
>
> **Author.** Claude (foreground, after sub-agent returned weak), 2026-05-07.

---

## 1. The asymmetry

C1.A patch (commit `e9528fa`) shipped the scaler-symmetry fix. 2-pair smoke retrain on EUR_USD,GBP_USD H1 produced:

| TF | pre-fix | post-fix | Δ | gate (52%) |
|---|---|---|---|---|
| H1 | 43.8% | **43.2%** | -0.6 | FAILED |
| M15 | 44.5% | **54.0%** | +9.5 | PASSED |
| H4 | 43.8% | **50.5%** | +6.7 | FAILED |
| aggregate | ~44% | 49.2% | +5pp | informational |

If C1.A only fixed normalization, all 3 TFs should improve symmetrically. They don't.

---

## 2. Hypothesis H6 — residual `skip_scaling=True` for direction training: RULED OUT

`grep -rn "skip_scaling=True" src/` returns 0 matches in active code (only matches are in C1.A patch comments documenting the prior bug). Verified at `src/training/trainers/joint_trainer.py:395` (now `skip_scaling=False`). No other call sites for direction training.

---

## 3. Hypothesis H1 — lookahead config divergence: SMOKING GUN

### 3.1 Trainer's lookahead = 24

`src/training/buddy_training_helpers.py:536`:
```python
lookahead: int = 24,
```

`:544-545`:
```python
for i in range(len(close) - lookahead):
    future_return = (close[i + lookahead] - close[i]) / close[i]
```

The direction trainer (correlation-transfer pipeline used by `scripts/scheduled_retrain.py`) labels each bar `i` with the sign of `close[i+24] / close[i] - 1`. Default lookahead is 24 bars, applied uniformly across all timeframes.

### 3.2 Holdout validator's lookahead = 5 (HARDCODED)

`scripts/scheduled_retrain.py:451-455` — inside `validate_holdout_accuracy`:
```python
for i in range(min(len(features) - 5, 200)):  # cap at 200 for speed
    row = features.iloc[i : i + 1]
    future_close = test_df["close"].iloc[min(i + 5, len(test_df) - 1)]
    current_close = test_df["close"].iloc[i]
    actual_dir = "LONG" if future_close > current_close else "SHORT"
```

The literal `5` is the lookahead — hardcoded, applied uniformly across all timeframes (H1/M15/H4).

### 3.3 The mismatch

The model is trained to predict 24-bar-forward direction, but the holdout scores it against 5-bar-forward actual:

| TF | trainer lookahead | trainer horizon (real time) | holdout lookahead | holdout horizon (real time) | mismatch |
|---|---|---|---|---|---|
| H1 | 24 bars | 24 hours | 5 bars | 5 hours | 4.8× |
| M15 | 24 bars | 6 hours | 5 bars | 75 min | 4.8× |
| H4 | 24 bars | 96 hours (4 days) | 5 bars | 20 hours | 4.8× |

The bar-count ratio is constant (4.8×) but the **statistical properties of "5-bar move" vs "24-bar move" differ per timeframe**. At H1, 5 hours is short enough that mean-reversion dominates the model's 24-hour directional prediction. At M15, 75 minutes is short enough that the model's 6-hour directional bet still has correlated trend. At H4, 20 hours is roughly the model's training horizon (24-bar at H4 = 4 days), so partial agreement.

This produces exactly the asymmetry observed:
- H1 holdout (5h actual vs 24h trained): least correlated → 43% (effectively noise)
- M15 holdout (75min actual vs 6h trained): partial correlation → 54%
- H4 holdout (20h actual vs 96h trained): closer correlation → 50.5%

### 3.4 Why C1.A didn't help H1

C1.A fixed the scaler-null bug, which was producing GARBAGE input features to the model. With the patch, the model now receives properly-scaled features and predicts directionally. **But the holdout is still measuring the model's 24-hour prediction against a 5-hour outcome — the model can be 100% accurate at its own task and still score ~50% on this mismatched evaluation.** The 43.2% post-fix on H1 is the holdout protocol's noise floor; further C1-style fixes can't move it.

The +9.5pp jump on M15 is real: the model's predictions are now informative, and the 75min/6h ratio is favorable enough that 75min direction often agrees with the 6h direction. But it's a coincidence of horizons, not the fix's intended path.

---

## 4. Hypotheses H2-H5: insufficient evidence or contradicted

- **H2 (per-TF threshold/feature variation):** no evidence; the loader is unified.
- **H3 (H1 holdout uses different features):** disproven; `compute_normalized_features` is called with the same args for all TFs (`scheduled_retrain.py:443`).
- **H4 (early-termination/silent fail):** disproven; smoke retrain log shows EUR_USD scaler IS saved (`meta['scaler']=StandardScaler`, `direction_scaler.pkl` present).
- **H5 (sample-size noise):** disproven; all 3 TFs use `holdout_candles=500` and converge on 400 evaluated bars.

---

## 5. Verdict

**H1 (lookahead mismatch) is the smoking gun.** The holdout validator at `scripts/scheduled_retrain.py:451-455` is incorrect: it hardcodes a 5-bar lookahead that doesn't match the trainer's 24-bar default. This has been silently corrupting holdout-accuracy numbers since this script was written — the 44% that's been blocking promotion since Apr 16 was measuring the wrong thing.

C1.A is structurally correct AND has measurably improved the model. The holdout protocol bug is what's blocking promotion, not the model.

---

## 6. Recommended fix path

Two options:

### Option G (minimal symptom fix — recommended)

Replace the hardcoded `5` at `scheduled_retrain.py:451, 453` with a constant that matches the trainer:

```python
HOLDOUT_LOOKAHEAD = 24  # MUST match buddy_training_helpers.py:lookahead default

for i in range(min(len(features) - HOLDOUT_LOOKAHEAD, 200)):
    row = features.iloc[i : i + 1]
    future_close = test_df["close"].iloc[min(i + HOLDOUT_LOOKAHEAD, len(test_df) - 1)]
    ...
```

After the fix, re-run the same 2-pair smoke retrain — H1 holdout should jump significantly (the scaler-fixed model is in place; only the evaluation horizon was wrong). Expected post-fix H1: 50-58% range (the audit's CLEAN-head ceiling at 24-bar horizon).

### Option H (proper architectural fix)

Plumb the trainer's actual lookahead through the artifact metadata. Save `lookahead` in `transformer_direction.meta.pkl`. Holdout validator reads it from the model artifact and uses the same value. Future trainer changes to lookahead automatically propagate to holdout. ~20 LOC change across trainer save + holdout validator load.

**Recommendation: ship Option G first** (single-line trainer-aligned constant), validate the post-fix holdout passes, then file Option H as an architectural follow-up.

---

## 7. Operator action

1. Apply Option G patch to `scripts/scheduled_retrain.py:451, 453` (1 file, ~3 lines + 1 constant).
2. Re-run `python scripts/scheduled_retrain.py --pairs EUR_USD,GBP_USD --granularity H1`.
3. Read the new holdout result. If H1 ≥52%, the joint dir gets PROMOTED automatically and Track D unblocks.
4. If H1 still <52%, the issue is deeper than lookahead — file as a separate investigation.

The 52% threshold itself is **not changing**. The bug is that we've been measuring against the wrong target.

---

## 8. Confidence in finding

**HIGH.** The lookahead mismatch is a code-level fact, citable file:line, no inference required. The mismatch's 4.8× ratio explains the per-TF asymmetry quantitatively (different real-time-horizon ratios produce different cross-correlations).

**Sanity check across the codebase** — `grep -rn "lookahead" --include="*.py" src/training/ src/core/modular_data_loaders.py`:

| Lookahead value | File | Purpose |
|---|---|---|
| **24** | `buddy_training_helpers.py:536` | **direction trainer (the active training path)** |
| 12 | `tensorflow_data_pipeline.py:95` | `target_shift` default (alt path; not the active direction trainer) |
| 12 | `alternative_targets.py:60` | `entry_lookahead` for alternative target heads (not direction) |
| 5 | `feature_analysis.py:73` | analysis tool default — NOT a training pipeline |
| **5** | `scripts/scheduled_retrain.py:451-453` | **holdout validator (HARDCODED, mismatched)** |

The holdout's `5` matches only the analysis tool's default — never the actual trainer. The trainer's `24` is the operative value (direction labels are 24-bar-forward sign). The fix at Option G must use `24` or, better, plumb the actual value via Option H so future trainer changes propagate.
