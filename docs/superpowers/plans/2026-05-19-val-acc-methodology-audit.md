# val_accuracy Methodology Audit — Is Our Ship Metric Honest?

**Date:** 2026-05-19
**Author:** AI Engineer agent (read-only audit)
**Load-bearing question:** is `val_accuracy` (and the `abs(train_acc - val_acc) > 0.10` ship gate derived from it) the correct metric to govern Buddy's autonomous-quant-loop ship decisions?
**TL;DR (HIGH confidence):** **No.** The metric is structurally gameable (filtered clear-label set + raw accuracy on imbalanced classes + train/val computed from different model states). The 10% gap rule is partially-honest but compares apples to oranges. A holdout exists only in `scheduled_retrain.py` and is also accuracy-not-balanced. Walk-forward validation exists in `src/training/walkforward_validation.py` but is **never called** by the ad-hoc agent path.

---

## Executive Summary

The current ship rule — `gap = abs(train_accuracy - val_accuracy) > 0.10 → quarantine` — fires on two metrics that are not directly comparable:

1. **`val_accuracy`** is computed by `self.model.predict(x_val_filtered)` AFTER `restore_best_weights=True` fires — so it reflects the saved best-val checkpoint, evaluated on **clear-label-only** validation sequences with `threshold=0.5` decision rule on a class-imbalanced label set.
2. **`train_accuracy`** is read from `history.history["accuracy"][best_epoch_idx]` — the keras training-time value at the best-val epoch, computed on **augmented** training data with **dropout enabled** and **sample weighting active**.

These two numbers are not measuring the same model on the same distribution. The gap can be misleading in either direction; the 2026-05-13 fix (commit `dbb6131`) only addressed the worst case (history[-1] vs best-val), not the deeper asymmetry.

Worse: `val_accuracy` itself is the wrong headline because the labels are class-imbalanced. The 2026-05-13 HistGB `val_acc=0.95 / balanced=0.50` log line (cited in `2026-05-19-training-run-timeline.md:74`) proves the metric is gamed by majority-class prediction. `val_balanced_accuracy` is already computed (`transformer_trainer.py:2635`, `histgb_trainer.py:202`) but is not the ship gate.

**Bottom line for the autonomous-quant-loop:** the agents currently gate on a metric that says "the prediction matches the label more than 50% of the time", which on imbalanced FX direction labels is satisfied by predicting the majority class. The ship rule fires on a gap between two non-comparable quantities. Both should change.

---

## A) The metric itself

### A1) What does `val_accuracy` measure?

`src/training/trainers/transformer_trainer.py:2608-2610`:
```
val_raw_pred = self.model.predict(x_val_filtered, verbose=0)
val_pred = (val_raw_pred > 0.5).astype(float)
val_acc = np.mean(val_pred.flatten() == y_val_filtered)
```

**HIGH confidence.** It measures fraction-of-correct on the **clear-label-filtered** val set (`transformer_trainer.py:1769-1775`: `val_clear_mask = w_val_seq > 0`, then `x_val_filtered = x_val_seq[val_clear_mask]`). Sequences whose `lookahead`-bar future move was below `threshold` (and thus carry weight=0) are **dropped from val before measurement**. So `val_accuracy` is measured on a subset that excludes ambiguous moves — the easier subset of the val window.

### A2) What does `val_balanced_accuracy` measure?

`transformer_trainer.py:2632-2635`:
```
val_pred_cal = (val_raw_pred.flatten() > raw_median).astype(float)
up_acc_cal = np.mean(val_pred_cal[y_true == 1] == 1) if (y_true == 1).sum() > 0 else 0
down_acc_cal = np.mean(val_pred_cal[y_true == 0] == 0) if (y_true == 0).sum() > 0 else 0
balanced_acc = (up_acc_cal + down_acc_cal) / 2
```

**HIGH confidence.** Mean of per-class accuracies, using the **calibrated threshold** (`raw_median` not 0.5). This is the standard scikit-learn balanced accuracy. Crucially it uses a **different decision threshold** (`raw_median`) than `val_accuracy` (which uses `0.5`). So `val_accuracy` and `val_balanced_accuracy` are not even computed against the same predictions.

### A3) Is the transformer's `val_accuracy` similarly gameable?

**HIGH confidence: yes.** The HistGB case is the extreme (`val=0.9521 / balanced=0.5017` in `m15_pair_expansion.log:23:00:30`, cited in `2026-05-19-training-run-timeline.md:74`). The transformer is less extreme but the same mechanism applies: when filtered val labels are skewed (e.g., 70% LONG in a trending window), a model that always predicts LONG hits 70% val_accuracy with balanced ~50%. The `2022 5:44a` observation already flagged "USD_JPY M15 direction labels balanced in raw split but 20% LONG in training" — clear-label filtering reshapes the class balance asymmetrically across the temporal split. The model collapse detector at `transformer_trainer.py:2690-2691` only catches the 0% / 100% extreme; the silent imbalance gameability persists.

### A4) Should the 10% gap rule use balanced metrics?

**HIGH confidence the current choice is the wrong basis; MEDIUM confidence the operator's reasoning was arbitrary.** Source: `.claude/rules/improvement.md:108-120` documents the rule's history (2026-05-13 operator directive after the 5-pair M15 retrain with `val_acc=4.7-18.9% / balanced≈50%`). The fix focused on **reading** the metric correctly (not zero, not history[-1]), not on **choosing** the right metric. The doc never argues why raw accuracy beats balanced — it inherits the keras default. A balanced-vs-balanced gap would be more honest because train_acc and val_acc would both be class-rebalanced.

**Asymmetry not addressed by the 2026-05-13 fix:** `val_accuracy` is computed by `predict()` on the **restored best-val weights** (`transformer_trainer.py:2150-2155`: `restore_best_weights=True`); `train_accuracy` is read from keras `history.history["accuracy"][best_epoch_idx]`, which was recorded **during training** with **dropout on** and **augmented data** (`transformer_trainer.py:2887-2893`). These are NOT the same model evaluated on the same distribution. The "best-val epoch" only aligns the time axis; it does not align the evaluation conditions. A defensible gap would recompute BOTH at end of training with `self.model.predict(x_train_filtered)` (dropout off, no augmentation) and `self.model.predict(x_val_filtered)`.

---

## B) Train/val/test split methodology

### B1) Split ratio

`src/core/modular_data_loaders.py:1517-1548` (`temporal_split`): default `(0.7, 0.2, 0.1)`. `load_direction_data` at `:1829` uses the same default. **HIGH confidence.**

### B2) Temporal vs random

`modular_data_loaders.py:1540-1547`:
```
train_end = int(n_samples * train_frac)
val_end = int(n_samples * (train_frac + val_frac))
train_idx = np.arange(0, train_end)
val_idx = np.arange(train_end + gap, val_end)
test_idx = np.arange(val_end + gap, n_samples)
```

**HIGH confidence: temporal, sequential, no shuffle.** Correct in principle.

### B3) Embargo / gap between splits

`temporal_split` accepts `gap=0` default (`:1522`); `load_direction_data(df_feat)` at `scripts/train_single_model_m1.py:305,451,504` **does not pass `gap=`**, so it defaults to 0. **HIGH confidence: no embargo in the ad-hoc training path.** With `lookahead=24` (the runtime default per the architecture doc), the last 24 train labels overlap with the first 24 val sequences' future-windows — direct lookahead leakage at the train/val boundary. This is a real, currently-shipping leak.

### B4) Is the test set ever evaluated?

`grep "X_test\|y_test\|test_acc"` in `scripts/train_single_model_m1.py` and `scripts/scheduled_retrain.py`:
- `train_single_model_m1.py`: 0 hits — **the 10% of data designated as `test` is never touched.**
- `scheduled_retrain.py`: hits only its own holdout function (uses last 500 OANDA candles, not the loader's `X_test`).

**HIGH confidence: the in-loader test set is dead weight in the ad-hoc agent path.** Val is both the ship-gate metric AND the early-stopping target — classic test-set leak into model selection.

### B5) Walk-forward validation usage

`grep "walkforward_validation\|WalkForwardValidator"` in `scripts/`:
- `scripts/robustness_test.py`, `scripts/diagnose_training_issues.py`, `scripts/calibrate_fx_confidence.py` — diagnostic-only paths.
- `scripts/train_single_model_m1.py` — **0 hits.**
- `scripts/scheduled_retrain.py` — **0 hits.**
- `scripts/stage_4a_news_retrain.py` — **0 hits.**

**HIGH confidence: walk-forward validation is not on the autonomous-loop training path.** It exists in `src/training/walkforward_validation.py` and is wired into `buddy_training_helpers.py` + `enterprise_training.py` (separate path), but the agent-triggered retrains never use it.

---

## C) Best-checkpoint vs final-epoch (the 2026-05-13 fix audit)

### C1) Was val_accuracy also fixed at best-val epoch?

`transformer_trainer.py:2608` (`val_raw_pred = self.model.predict(x_val_filtered)`) runs after `model.fit()` returns. Because `EarlyStopping(restore_best_weights=True)` (`:2150-2155`) fires inside fit, the weights are restored before predict. **HIGH confidence: yes — `val_accuracy` reflects the saved best-val checkpoint.** This was already correct pre-2026-05-13.

### C2) Saved model = predict's model?

Yes, with one caveat: `_handle_warm_start_recovery` at `:2909` runs **before** `_compute_final_metrics`. If recovery mutates weights (it can: see `:2317-2324`), the metrics-time weights are the recovered weights, not strictly the best-val weights. MEDIUM confidence the saved artifact matches the metrics, because save happens after metrics — but a recovery branch that fires post-restore-best could desynchronize them. Out of scope for this audit; flagged for future investigation.

### C3) train_acc vs val_acc — same model state?

**HIGH confidence: no.** Verified at `transformer_trainer.py:2648-2657`:
```
val_acc_history = history.history.get("val_accuracy", [])
train_acc_history = history.history.get("accuracy", [])
if val_acc_history and train_acc_history:
    best_epoch_idx = int(np.argmax(val_acc_history))
    train_accuracy_at_best = float(train_acc_history[best_epoch_idx])
```

- `train_accuracy_at_best` is the keras `history` recorded value: **dropout ON, augmentation ON, sample_weight ON** (`:2887-2893,:2902-2903`).
- `val_accuracy` is `model.predict()` post-fit: **dropout OFF, no augmentation, no weighting**.

The 2026-05-13 fix aligned the EPOCH but not the EVAL CONDITIONS. The gap measured today is `train_acc_with_regularization_on - val_acc_with_regularization_off`. A well-regularized model will report a smaller train_acc than its true train_acc (dropout hurts train_acc more than val_acc), making the gap **underestimate** overfitting. This means the 10% gate is **less strict than it appears.**

---

## D) Holdout testing

### D1) Is holdout part of every training run?

`grep "holdout\|HOLDOUT\|X_test\|test_acc"` in `scripts/train_single_model_m1.py` → **0 hits.** `grep` in `scripts/stage_4a_news_retrain.py` → only doc strings telling the operator to run `scripts/per_pair_holdout_eval.py` manually after training (`stage_4a_news_retrain.py:155-157`). **HIGH confidence: holdout is NOT part of every training run.** It runs only inside `scheduled_retrain.py` (the deploy-gate harness) and `scripts/per_pair_holdout_eval.py` (separate operator tool).

### D2) Why don't price-only trainings get a holdout?

No documented reason. The asymmetry is operational: news-fusion was the highest-stakes change, so it got a holdout harness; price-only ad-hoc retrains never added one. **MEDIUM confidence: the omission is historical, not principled.** The same reasoning that justifies a holdout for news-fusion (verify out-of-window generalization before promotion) applies equally to price-only retrains. Higher-stakes still, because price-only retrains are what the autonomous loop fires.

### D3) `scheduled_retrain.py` holdout details

`scheduled_retrain.py:373-500`:
- Loads last `holdout_candles=500` candles from OANDA (FRESH PULL, not the loader's `X_test` — so it's truly out-of-window of training data).
- Iterates rolling 60-bar windows, predicts direction, compares vs `t+HOLDOUT_LOOKAHEAD=24` actual move.
- Metric: **raw accuracy** (`correct / total`), `min_accuracy=0.52` deploy gate.

**MEDIUM confidence: this is honest but gameable.** The same class-imbalance critique applies — a 52% gate on raw accuracy is satisfied by a model that predicts the majority class on a 55/45-skewed holdout window. No `balanced_acc` is computed. The fixed `HOLDOUT_LOOKAHEAD=24` is the source of one of the May 2026 audit's flagged issues (it must match the trainer's lookahead — currently desynchronized per `2026-05-19-training-architecture-control-plane-wiring.md:45-48`).

---

## E) What the autonomous-quant-loop actually needs

### E1) Metric most correlated with trade PnL

**MEDIUM confidence: none of the current metrics.** Direction-accuracy on filtered clear labels is a necessary-but-not-sufficient proxy for P&L. P&L depends on: (a) directional correctness on **executable setups** (those that pass gates), (b) average return per correct trade (TP_ATR_MULT efficiency), (c) average loss per wrong trade (SL_ATR_MULT discipline), (d) calibration of `confidence` (the position sizer scales on it), (e) avoidance of trades during low-edge regimes.

`val_accuracy` measures only (a) — and only on the easy clear-label subset of the val window, with no notion of gate-rejection-rate or expected-value per trade.

### E2) Balanced vs Sharpe vs hit-rate-of-high-confidence-trades

The most direct proxy is **balanced accuracy on the HIGH-confidence subset of val predictions**, because gated trades only fire when `gates.evaluate_transformer` passes the calibrated-threshold + uncertainty filter. The current `val_balanced_accuracy` is computed over ALL filtered val sequences, not just the gate-passing ones. A high `val_balanced_accuracy` says "the model is well-calibrated overall" but doesn't say "the model is right on the trades it would actually fire."

The honest metric for the autonomous-loop ship decision is: **balanced accuracy on the subset of val predictions where `confidence > min_confidence_threshold` AND `uncertainty < max_uncertainty_score` AND `model_disagreement < max_model_disagreement`**. This is what gets traded. This is what should be gated on. The infrastructure to compute it exists (gates module, ridge calibration); the trainer just doesn't measure it.

### E3) Minimal change to align ship metric with trade goal

In ascending blast radius:

1. **Promote `val_balanced_accuracy` to the headline log line + ship gate.** One-line change in the trainer's logger.info; downstream `_quarantine_if_overshipped` reads `train_accuracy / val_accuracy` from the metrics dict, switch it to read the balanced variants (or compute `train_balanced_accuracy` for parity — currently NOT computed; `transformer_trainer.py:2660` only emits `train_accuracy`).
2. **Recompute `train_accuracy` post-fit with dropout off**, matching the eval conditions of `val_accuracy`. Single extra `predict(x_train_filtered)` call. Closes the asymmetry from C3.
3. **Add an embargo `gap = lookahead_bars` to every `load_direction_data` call** in the ad-hoc training paths. Closes the boundary leak from B3. Single kwarg change at three call sites in `train_single_model_m1.py`.
4. **Add a `gated_balanced_accuracy` metric** that measures balanced accuracy on the subset of val predictions that would pass the runtime gate filters. New computation, not a config flip. Highest-value change for the autonomous loop.
5. **Make holdout mandatory for every retrain**, not just `scheduled_retrain.py`. Move the holdout harness into `train_single_model_m1.py` as a post-save step. Decouple from OANDA fresh-pull (use loader's `X_test`, which is currently dead).

---

## Metric-mismatch table

| Metric (name) | What it measures | What it's treated as | Mismatch severity |
|---|---|---|---|
| `val_accuracy` | Fraction of clear-label-filtered val sequences where `predict > 0.5` matches y_val. Post-restore-best weights, dropout OFF. | "Out-of-sample performance" / proxy for trade edge | **HIGH** — gameable by majority-class prediction on imbalanced filtered val |
| `train_accuracy` | Keras `history.history["accuracy"][best_epoch_idx]`. Recorded WITH dropout, augmentation, sample weighting active. | "In-sample performance" — paired with val_acc for gap | **HIGH** — different eval conditions than val_acc, so gap measures noise+regularization-effect, not overfitting alone |
| `gap = abs(train_acc - val_acc)` | Difference between two non-comparable quantities (see above) | "Overfitting gauge" — ship rule trip at 0.10 | **HIGH** — underestimates true overfit (regularization hurts train_acc more than val_acc) |
| `val_balanced_accuracy` | Mean of per-class accuracies, using CALIBRATED threshold (raw_median, not 0.5) | Logged but not gated; some operators eyeball it | **MEDIUM** — closer to honest but uses a different threshold than val_acc; not the ship gate |
| `val_direction_gap` | abs(up_acc_cal - down_acc_cal) | Logged at WARNING when > 0.10; cosmetic | **MEDIUM** — useful diagnostic, not part of ship rule |
| `train_balanced_accuracy` | Not computed | n/a | n/a — but ITS ABSENCE is the bug: there's no train-side counterpart to val_balanced_accuracy, so a balanced gap can't be computed |
| `scheduled_retrain` holdout accuracy | Raw accuracy on last 500 OANDA candles, rolling 60-bar windows, lookahead=24 | Deploy gate at 0.52 | **MEDIUM** — class-imbalance gameable like val_acc; lookahead hardcoded; only fires in scheduled path |
| `loader's X_test/y_test` | Last 10% of historical window, temporally separated | Designed to be the true holdout | **CRITICAL** — never evaluated anywhere in the ad-hoc agent path; pure dead weight |

---

## Concrete recommendations

### R1) Should the 10% gap rule switch to balanced metrics?

**YES — HIGH confidence.** Two concrete asks:
- Compute `train_balanced_accuracy` in `_compute_final_metrics` (recompute via `self.model.predict(x_train_filtered)` with the same calibrated threshold used for `val_balanced_accuracy`).
- Switch the ship gate to `gap_balanced = abs(train_balanced - val_balanced)` with the same 10% threshold. Keep raw-accuracy gap as a secondary diagnostic logged in `RESULT:{...}`.
- This is an operator decision per task constraints — do NOT change without operator approval. Audit-level recommendation only.

### R2) Should walk-forward validation be the default for ad-hoc retrains?

**Yes for promotion gating, no for training itself — MEDIUM confidence.** Walk-forward training (rolling-window retrains) is expensive (~5× single-shot training time). But walk-forward EVALUATION (train once, evaluate on multiple chronological windows) is cheap and adds robust generalization signal. `src/training/walkforward_validation.py` already provides the harness; it just needs wiring into `train_single_model_m1.py` as a post-training validation step. Two-three additional held-out windows would give variance bars around `val_accuracy` and surface temporal instability — currently invisible.

### R3) Should holdout be added to every training run?

**YES — HIGH confidence.** The asymmetry (only `scheduled_retrain` runs holdout, ad-hoc retrains don't) is operationally indefensible. The autonomous loop fires `train_single_model_m1.py` subprocess, not `scheduled_retrain.py`. Every retrain must produce a holdout number; the loader's existing `X_test` slot is the cheapest implementation (one extra `predict()` call). Promotion to live should require `(gap < 0.10) AND (holdout_balanced > 0.52) AND (val_balanced > 0.52)` — three independent checks, not one. This closes the dead-weight in B4.

### R4) Minimal change to make the autonomous-loop ship decision honest?

In one PR:
1. Add `train_balanced_accuracy` to the metrics dict.
2. Recompute train_acc post-fit with dropout off (resolves C3 asymmetry).
3. Pass `gap=lookahead_bars` to `load_direction_data` calls in `train_single_model_m1.py` (resolves B3 boundary leak).
4. Add a `holdout_balanced` computation using the loader's `X_test` (resolves D2 + B4 dead-weight).

Blast radius: ~30 lines across 2 files. Reversible. Does NOT touch `HARD_MAX_GAP=0.10` (per operator constraint). Does NOT change the 10% rule's threshold; only swaps the inputs.

### R5) Ship-rule v2 proposal

Current rule (`.claude/rules/improvement.md:108-120`):
> `gap = abs(train_accuracy - val_accuracy) > 0.10 → quarantine`

Proposed v2 (audit-level recommendation, requires operator approval before promotion):
> A model ships only when ALL of:
> 1. `gap_balanced = abs(train_balanced_accuracy - val_balanced_accuracy) < 0.10` (train recomputed with dropout off)
> 2. `val_balanced_accuracy > 0.52` (strict edge over chance on balanced classes)
> 3. `holdout_balanced_accuracy > 0.52` (on the loader's X_test slice, currently dead)
> 4. `val_direction_gap < 0.20` (no extreme class-prediction skew)
>
> Failure of ANY of (1-4) routes to `_quarantine/` with the failure reason in `RESULT:{}`.

Why this is honest: the four conditions are independent — (1) measures overfit, (2) measures absolute edge on balanced classes, (3) measures temporal-generalization to a held-out slice, (4) measures direction-bias. The current rule conflates (1) with (2) via raw accuracy, and skips (3) and (4) entirely.

Confidence per claim:
- "Current rule is gameable via class imbalance" — **HIGH** (HistGB `val=0.95 / bal=0.50` is on disk, cited).
- "train_acc and val_acc are eval-condition-asymmetric" — **HIGH** (`transformer_trainer.py:2648-2660` reads history-recorded train, predict-recorded val).
- "v2 closes the gameability" — **MEDIUM** (balanced is harder to game but not impossible; holdout requires data that genuinely wasn't seen).
- "30-line PR scope" — **LOW** (estimate; the test surface around it could be larger).

---

## Sources cited

- `src/training/trainers/transformer_trainer.py:1769-1775` (val clear-label filter)
- `src/training/trainers/transformer_trainer.py:2150-2155` (EarlyStopping restore_best_weights)
- `src/training/trainers/transformer_trainer.py:2598-2672` (`_compute_final_metrics`)
- `src/training/trainers/transformer_trainer.py:2887-2903` (fit conditions: dropout, augmentation, sample_weight)
- `src/training/trainers/histgb_trainer.py:194-227` (HistGB val_acc + balanced)
- `src/core/modular_data_loaders.py:1517-1548` (temporal_split, gap default 0)
- `src/core/modular_data_loaders.py:1827-1874` (`load_direction_data`, gap default 0)
- `scripts/train_single_model_m1.py:55-123` (MAX_GAP, HARD_MAX_GAP, `_quarantine_if_overshipped`)
- `scripts/train_single_model_m1.py:305,451,504` (load_direction_data calls with no gap=)
- `scripts/scheduled_retrain.py:373-500` (holdout harness, raw accuracy gate)
- `scripts/stage_4a_news_retrain.py:155-157` (manual holdout instructions)
- `src/training/training_defaults/direction_training_config.json:7,20-22` (canonical lookahead=12, lr=3e-4, dropout=0.4)
- `.claude/rules/improvement.md:108-120` (operator 10% gap rule history)
- `docs/superpowers/plans/2026-05-19-training-run-timeline.md:69-76` (HistGB val=0.95/bal=0.50 evidence)
- `docs/superpowers/plans/2026-05-19-training-architecture-control-plane-wiring.md:45-48` (lookahead 12 vs 24 divergence)

---

## Constraints honored

- Read-only audit; no code changes; no config changes.
- No mocks (per `.claude/rules/improvement.md` No-Mock Rule).
- Every causal claim carries an explicit confidence tag + file:line citation.
- `HARD_MAX_GAP = 0.10` rule untouched; ship-rule v2 is a proposal, not an applied change.
- Operator decision required before any of R1-R5 land.
