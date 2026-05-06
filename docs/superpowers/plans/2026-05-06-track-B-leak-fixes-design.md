# Track B — Label-Leak Fixes Design (B1–B4)

> **Status.** Design only. No code in this commit. Track A is on main (HEAD `362f877`); confidence-head fix shipped in A1; per-pair confidence collapsed to noise floor in A1.5; audit on main as `02cb006`. This document specifies how to apply the confidence-head fix template to the four other CONFIRMED LEAK heads.
>
> **Author.** Software Architect sub-agent, 2026-05-06.
>
> **Scope.** B1 volatility regime · B2 momentum_score · B3 acceleration · B4 streak_prob. One commit per head; one head per checkpoint.

---

## 0. Reading map (operator who picks this up)

1. `docs/superpowers/plans/2026-05-05-self-heal-self-train-roadmap.md` — context + Track B box (lines 72–85).
2. `docs/label_leak_audit_2026-05-05.md` — the four leaks, line-cited.
3. `src/training/labels/realized_confidence_label.py` — the *template* every B-head mirrors (function shape, journal join + triple-barrier fallback, blend mode, NaN-row dropping pattern).
4. `src/core/modular_data_loaders.py:3438-3504` — call site of the confidence template.
5. `scripts/retrain_confidence_leak_fix.py` — the joint retrain template (backup → load → train → write meta).
6. This file.

---

## 1. Soberness statement (read before predicting any post-fix metric)

A1.5 retrained the confidence head per-pair on realized-outcome labels and got **R² ∈ [-0.189, -0.046] across 7 pairs, ALL below the joint head's -0.0102 and ALL below the audit's predicted 0.05–0.30 band.** The leak removal collapsed apparent signal to the noise floor.

**Track B should NOT promise that fixing B1–B4 will lift them into a positive band.** It should frame the fixes as: *remove fake signal; expose honest difficulty.* If the four heads end up at noise floor like confidence, that is the truth — not a regression. The system already has a downstream calibration layer (`trained_data/confidence_calibration.json`) that maps low-information scores to outcomes; honest noise can be calibrated even when R² is near zero.

**Key line for the operator:** *"Track B's success criterion is leak-detection passing and labels being non-derivable from features — not R² ≥ X. Post-fix metrics may collapse to noise floor; that is the system being honest about how hard each prediction problem really is."*

---

## 2. Pattern reuse — recommendation: **copy per head, defer extraction.**

Four heads, four shapes:

| Head | Target type | Forward signal source | Journal usable? |
|---|---|---|---|
| B1 volatility_regime | 4-class classification | forward realized vol over W bars | no — journal has trade outcomes, not vol regime |
| B2 momentum_score | regression in [0, 1] | forward |returns| magnitude or risk-adj. return | partial — could weight by trade outcome but signal is mostly market-driven |
| B3 acceleration | binary classification | derived from B2 (forward vs backward windows) | no — derivative of B2 |
| B4 streak_prob | regression in [0, 1] | realized future loss-streak length / probability | **yes — journal is the natural source** |

The confidence template's specific shape — journal join + triple-barrier pseudo-fallback + binary blend — fits **only B4** cleanly (because only B4 is naturally journal-driven). B1/B2/B3 don't have a meaningful journal join (regime/vol labels aren't trade-level outcomes); they're forward-OHLC-derived end to end.

**Recommendation:** copy the file template (`src/training/labels/realized_<head>_label.py`) and the function-signature shape (`compute_realized_<head>_labels(df, instrument, journal=None, lookahead_bars=..., ..., label_mode=...) -> (labels, metadata)`), but do NOT extract a shared base class yet. Reasons:

- Each head's *forward computation* is different (4-class binner vs scalar magnitude vs binary derivative vs streak-counter). A premature `realized_outcome_base.py` would either be an empty interface or would force the four implementations into a procrustean shape.
- After B1–B4 ship, the duplication will be visible. Extract then, when you can see what's actually duplicated (likely: the journal-index helper from `realized_confidence_label.py:54-87`, the df-index normalizer at `:90-106`, and the ATR resolver at `:109-118`). Move those three helpers to `src/training/labels/_common.py` *as a follow-up*, not as a prerequisite.
- The No-Mock Rule means tests for each head will be real-disk integration tests that drive the full loader. Sharing test infrastructure is more valuable than sharing label-generator code.

---

## 3. Sequencing — operator's order is correct.

Operator plan: B1 → (B2+B3 jointly) → B4. Confirmed:

- **B1 first** — TCN trainer is the slowest (Keras + custom train loop at `src/training/trainers/tcn_volatility_trainer.py:394`). Land it first so it's the long pole; no downstream B-head depends on B1.
- **B2 + B3 jointly** — B3's label is `1[mom_score[i] > mom_score[i-W]]`, computed *from the B2 label*. They share a loader (`load_xgboost_data` at `src/core/modular_data_loaders.py:2947`) and a trainer (`LightGBMMomentumTrainer.train` at `src/training/trainers/lightgbm_trainers.py:489`, dual-output). Splitting them into two commits creates a pointless intermediate state where B2 is fixed and B3 still derives from the (now-changed) B2. Ship them as one commit. (This does not violate "one head per commit" — B3 has no independent label source; it is a derivative output of B2 by construction.)
- **B4 last** — `streak_prob` is the lowest-impact head (it's a secondary output of `LightGBMRiskTrainer` at `src/training/trainers/lightgbm_trainers.py:720`; the primary `expected_drawdown_pct` output is CLEAN per audit and stays as-is). The journal-driven label generator can reuse the journal-index helpers from `realized_confidence_label.py` directly without refactor.

**No shared-compute conflicts.** TCN runs on Keras; LightGBM runs on CPU-bound trees. They can be retrained back-to-back on the same machine without contention. There is no inter-head dependency forcing parallelism.

---

## 4. B1 — Volatility regime (TCN, 4-class)

### 4.1 The leak

> Audit `docs/label_leak_audit_2026-05-05.md:38-81`. Quoted formula:
>
> ```
> compute_volatility_regime() at src/core/modular_data_loaders.py:615-649:
>   for each row i:
>     window = atr_pct_14[i-100 : i]
>     pct_rank = sum(window <= atr_pct_14[i]) / len(window)
>     regime[i] = bin(pct_rank, [0.25, 0.60, 0.85])
> ```
>
> X matrix at `src/core/modular_data_loaders.py:3580-3595` includes `atr_pct_14` itself, plus `atr_pct_5/10/20`, `volatility_5/10/20`, `bb_width_norm`, etc. The TCN sees a sliding window of `atr_pct_14[i-seq_len:i]` directly; it can reconstruct the percentile cut almost exactly.

### 4.2 Fix path — option (a) realized-outcome label

**Concrete realized-outcome definition:** label[i] = bin of *forward realized vol* over the next `vol_horizon_bars` bars, computed from forward OHLC only.

Pseudo-spec (NOT real code):

```
def compute_realized_volatility_regime_labels(df, vol_horizon_bars=24, n_classes=4):
    """
    For each row i:
      future_high  = high[i+1 : i+1+vol_horizon_bars]
      future_low   = low[i+1  : i+1+vol_horizon_bars]
      future_close = close[i  : i+vol_horizon_bars]
      realized_vol[i] = std(forward_log_returns) / mean(forward_close)
                        # OR: mean(future_high - future_low) / close[i]
    Bin realized_vol[] into 4 classes using percentile cuts derived
    from TRAINING-SLICE realized_vol values only (NOT a backward-looking
    rolling window per row — a single split-fit then apply).
    Last vol_horizon_bars rows: label = NaN; caller drops them.
    """
    return labels, metadata
```

Two design choices to flag:
- **Cut-fit basis.** Either (i) fit percentile cuts on the train slice's realized_vol distribution and apply to all splits (matches the train-only normalization pattern at `modular_data_loaders.py:3050-3058`), or (ii) keep fixed `[0.25, 0.60, 0.85]` cuts on a global-history percentile but compute on *forward* vol. Recommend (i) — it is the same pattern the momentum loader already uses for its norm_factor and avoids cross-split leakage.
- **Realized-vol formula.** Either standard-deviation of log-returns over the forward window, or mean Parkinson high-low range. Recommend stddev of log-returns (cleaner statistically, matches academic conventions), but Parkinson is acceptable if it shows tighter class-balance histograms.

### 4.3 Files to touch

- **New:** `src/training/labels/realized_volatility_regime_label.py` (~200 LOC, mirror of `realized_confidence_label.py`).
- **Edit:** `src/core/modular_data_loaders.py:3541-3697` (`load_volatility_regime_data`) — replace `compute_volatility_regime(df, config=vol_config)` call at line 3626 with `compute_realized_volatility_regime_labels(df, vol_horizon_bars=...)`. Keep the X-feature list at `:3580-3595` unchanged for now (option (b)/(c) from the audit are NOT being applied; we are doing only option (a)). Drop NaN-label rows at the tail before temporal split, mirroring the confidence loader at `:3461-3469`.
- **Leave untouched:** `compute_volatility_regime()` at `src/core/modular_data_loaders.py:612-685` — keep it as a *runtime utility* for live-inference consumers that compute regime from a current ATR percentile (the function is also used outside training; check call sites before deletion in a later sweep). Adding a docstring deprecation note ("training no longer uses this") is in scope; deletion is not.
- **Edit:** the TCN trainer `src/training/trainers/tcn_volatility_trainer.py:67-186` does NOT need changes — it consumes whatever labels the loader returns; only the labels' source changes.

### 4.4 Backup procedure (mirrors A1's `c790d28`)

Before retraining:

```
mkdir -p trained_data/rollback/volatility_regime_pre_2026-05-06/
cp trained_data/models/joint/tcn_volatility_regime.* \
   trained_data/rollback/volatility_regime_pre_2026-05-06/
# Per-pair if any exist:
for pair_dir in trained_data/models/*/; do
  if [ -f "$pair_dir/tcn_volatility_regime.keras" ]; then
    mkdir -p trained_data/rollback/volatility_regime_pre_2026-05-06/per_pair/$(basename "$pair_dir")
    cp "$pair_dir"tcn_volatility_regime.* \
       trained_data/rollback/volatility_regime_pre_2026-05-06/per_pair/$(basename "$pair_dir")/
  fi
done
```

Commit the backup as `chore(rollback): backup pre-leak-fix volatility regime artifacts` BEFORE the leak-fix code commit.

### 4.5 Retrain command

A custom script is required — `scripts/scheduled_retrain.py` does NOT have a `--head` flag (verified via argparse block at `scripts/scheduled_retrain.py:715-749`; only `--pairs`, `--granularity`, `--candles`, `--dry-run`, `--force-per-pair`). The TCN is wired into the joint trainer (`src/training/trainers/joint_trainer.py:43`) which retrains every head; running the joint trainer just to retrain volatility regime would also stomp on the (good) per-pair confidence pkls from A1.5.

**Proposed (do NOT write yet):** `scripts/retrain_volatility_regime_leak_fix.py`, modelled exactly on `scripts/retrain_confidence_leak_fix.py`:
- Load H1 OHLCV per pair via `_newest_h1_csv_per_pair`.
- Compute normalized features.
- Call `load_volatility_regime_data(df, ...)` (which now returns realized-outcome labels post-fix).
- Train via `TCNVolatilityRegimeTrainer.train` (`src/training/trainers/tcn_volatility_trainer.py:186`).
- Overwrite `trained_data/models/joint/tcn_volatility_regime.keras` (and `.arch.json`) in place.
- Log per-class F1, accuracy, confusion matrix to a side meta JSON.

Run command:

```
python scripts/retrain_volatility_regime_leak_fix.py --pairs all
```

### 4.6 Validation criteria

- **Metric.** Macro-F1 across 4 classes (LOW/NORMAL/HIGH/EXTREME). Accuracy alone is misleading because the audit's old leaky labels had a 25/35/25/15 distribution; the new realized-vol labels will have a different (and split-dependent) distribution.
- **Existing gate?** No fixed accuracy gate for volatility regime in `scripts/scheduled_retrain.py` (the only `min_accuracy=0.52` gate at line 375 applies to direction holdout). The TCN's `min_regime_for_trade=2` runtime gate (at `VolatilityRegimeConfig` in `modular_data_loaders.py:603-609`) is a *consumer* threshold, not a training threshold. Validation here is qualitative.
- **Success.** Fix is SUCCESSFUL if:
  1. Pre-fix accuracy was >>random (audit predicts ~95%+ because the model is approximating the formula); post-fix accuracy drops sharply toward ~25-50% (4-class chance is 25%; some forward-vol predictability is plausible from `volume_ratio_*` and `bb_width_*`).
  2. Class confusion is *broad* (4-class F1 with similar precision/recall per class), not collapsed to one majority class.
  3. No row-`t` feature can predict the realized-vol label deterministically (a sanity-check probe: train a 5-tree tiny LightGBM on the same X and the new label; if it gets >0.6 macro-F1, there is still a leak).

**Soberness.** Realistic post-fix macro-F1: 0.30–0.45. Below 0.25 = TCN is below random and should be replaced with a heuristic. Above 0.55 = double-check for residual leak (likely from `volume_ratio_*` or `volatility_*` which weakly correlate with future vol).

### 4.7 Rollback

```
git revert <fix-commit>
cp -r trained_data/rollback/volatility_regime_pre_2026-05-06/* \
       trained_data/models/joint/
# Per-pair restore as needed.
```

### 4.8 Risks + open questions

- **Q1 (operator):** `vol_horizon_bars` — confidence head used 24 H1 bars. Volatility regime is read by the runtime gate `min_regime_for_trade=2` for *current* trade entry sizing, so the natural horizon is the *expected trade duration*. Trade journal median hold time is the right answer; if unknown, default to 24 (parity with confidence). **Recommend defaulting to 24, but ask the operator before B1 starts.**
- **Risk:** if `vol_horizon_bars` is too short (e.g. 5) the labels will be very noisy and the model will collapse to majority-class. Too long (e.g. 100) and recent rows are NaN-dropped, shrinking the train set.
- **Risk:** the runtime call to `compute_volatility_regime()` in `modular_data_loaders.py:612` may still be used by non-training code (e.g. the scanner's runtime regime-selection in gates). Grep `compute_volatility_regime(` repo-wide before B1 ships; if there's a runtime call site, document that the *runtime path is unchanged* (it remains the rolling-percentile rule) — only the *training labels* moved to forward-realized.

---

## 5. B2 + B3 — Momentum (`momentum_score` + `acceleration`), shipped jointly

### 5.1 The leaks

> Audit `docs/label_leak_audit_2026-05-05.md:124-185`. Quoted formulas:
>
> ```
> # B2 — momentum_score, src/core/modular_data_loaders.py:3026-3029
> raw_momentum_all[i] = mean(|returns[i-momentum_window:i]|)   # default W=10
> momentum_score[i]   = min(raw_momentum_all[i] / norm_factor, 1.0)
>
> # B3 — acceleration, src/core/modular_data_loaders.py:3072-3076
> acceleration[i] = 1 if momentum_score[i] > momentum_score[i-5] else 0
> ```
>
> X matrix at `src/core/modular_data_loaders.py:2982` (via `get_normalized_feature_names()['momentum']` → `:1328-1340`) includes `returns_1, returns_2, returns_3, returns_5, returns_10, returns_20`. The label is a closed-form rolling sum of feature columns; LightGBM 150 trees of depth 6 fit it trivially. B3 is a derivative of B2 — zero forward information.

### 5.2 Fix path — option (a) realized-outcome label, jointly

**Concrete realized-outcome definitions:**

- **B2 momentum_score → forward realized momentum.** label[i] = `mean(|returns[i+1 : i+1+momentum_window]|) / norm_factor_train` where `norm_factor_train` is the P50 of *training-slice forward* momentum (analogous to current `norm_factor` at `:3050-3058` but shifted to forward window).
- **B3 acceleration → forward acceleration.** label[i] = `1[momentum_forward[i] > momentum_backward[i-W:i]]` — i.e. is *future* momentum higher than *past* momentum? This is the audit's explicit recommendation (`label_leak_audit_2026-05-05.md:182-184`).

Pseudo-spec:

```
def compute_realized_momentum_labels(df, momentum_window=10):
    """
    For each row i, where i+momentum_window+1 < n:
      forward_returns = (close[i+1:i+1+W] - close[i:i+W]) / close[i:i+W]
      forward_mom[i]  = mean(|forward_returns|)
      backward_mom[i] = mean(|returns[i-W:i]|)        # for B3 only
      momentum_score_label[i]    = min(forward_mom[i] / norm_train, 1.0)
      acceleration_label[i]      = 1.0 if forward_mom[i] > backward_mom[i] else 0.0
    Last momentum_window rows: label = NaN.
    norm_train computed on train-slice forward_mom only (no leakage).
    """
    return (mom_score_labels, accel_labels), metadata
```

### 5.3 Files to touch

- **New:** `src/training/labels/realized_momentum_label.py` (~150 LOC; computes both outputs in one pass since they share the forward-window scan).
- **Edit:** `src/core/modular_data_loaders.py:2947-3109` (`load_xgboost_data`):
  - Replace label computation at `:3017-3076` with a call to `compute_realized_momentum_labels(df, momentum_window=...)`.
  - Drop the `acceleration[i] = ... momentum_score[i-5]` derivation entirely (B3 is now a forward-window comparison, not a self-derivative).
  - Drop tail rows (`momentum_window` of them) where labels are NaN, mirroring confidence loader pattern.
  - Keep X-feature list as-is (option (b) from the audit — dropping `returns_*` from X — is NOT being applied; rationale: the audit itself notes `returns_*` columns ALSO weakly correlate with forward returns, so dropping them is brittle and may not actually break the leak).
- **Leave untouched:** `LightGBMMomentumTrainer.train` (`src/training/trainers/lightgbm_trainers.py:489-690`). The trainer consumes `y_train[:, 0]` for momentum_score (line 538) and `y_train[:, 1]` for acceleration; the dual-output ordering is preserved.

### 5.4 Backup procedure

```
mkdir -p trained_data/rollback/momentum_pre_2026-05-06/
cp trained_data/models/joint/lgbm_momentum.* \
   trained_data/rollback/momentum_pre_2026-05-06/
cp trained_data/models/joint/xgb_momentum.* \
   trained_data/rollback/momentum_pre_2026-05-06/  2>/dev/null || true
cp trained_data/models/joint/catboost_momentum.* \
   trained_data/rollback/momentum_pre_2026-05-06/  2>/dev/null || true
# Per-pair sweep (15 pairs × 3 algorithms = up to 45 files):
for pair_dir in trained_data/models/*/; do
  pair=$(basename "$pair_dir")
  for f in lgbm_momentum xgb_momentum catboost_momentum; do
    if ls "$pair_dir"$f* >/dev/null 2>&1; then
      mkdir -p trained_data/rollback/momentum_pre_2026-05-06/per_pair/$pair
      cp "$pair_dir"$f.* trained_data/rollback/momentum_pre_2026-05-06/per_pair/$pair/
    fi
  done
done
```

Commit as `chore(rollback): backup pre-leak-fix momentum model artifacts`.

### 5.5 Retrain command

Same constraint as B1 — no `--head` flag in `scheduled_retrain.py`. **Proposed:** `scripts/retrain_momentum_leak_fix.py`, modelled on `scripts/retrain_confidence_leak_fix.py`:
- Load H1 CSV per pair.
- Call `load_xgboost_data(df, momentum_window=10)`.
- Train `LightGBMMomentumTrainer.train(...)`.
- Overwrite `trained_data/models/joint/lgbm_momentum.pkl` (and per-pair if `--per-pair`).
- Write meta JSON with per-output metrics.

Run:

```
python scripts/retrain_momentum_leak_fix.py --pairs all
```

### 5.6 Validation criteria

- **B2 momentum_score (regression).** R² and MAE. No fixed gate exists in `scheduled_retrain.py`. **Sober prediction:** R² collapses from near-1.0 (leaky) to **near-zero or slightly negative**, mirroring confidence A1.5. Forward |returns| magnitude is hard to predict from row-`t` features; this is expected.
- **B3 acceleration (binary).** Accuracy and F1. **Sober prediction:** accuracy collapses from likely 95%+ (because the leaky label was a feature derivative) to **50–55%**, just above coin-flip. The audit itself notes acceleration's leaky version "uses no future data, deterministic in features" — once future data is in the label, predictability evaporates.
- **No threshold gate to lower.** Operator's hard rule applies but is moot — there is no fixed gate for momentum heads to begin with; success is qualitative ("post-fix metric reflects honest predictability of forward momentum from current features").
- **Leak sanity check.** After fix, train a 5-tree LightGBM on X and the new B2/B3 labels. If R² > 0.5 or accuracy > 0.7 respectively, residual leak exists (likely via `atr_pct_*` and `volatility_*` columns; would warrant Track-B-extension to drop those cols from this head's X).

### 5.7 Rollback

```
git revert <fix-commit>
cp -r trained_data/rollback/momentum_pre_2026-05-06/* trained_data/models/joint/
# Per-pair restore as needed.
```

### 5.8 Risks + open questions

- **Q2 (operator):** `momentum_window` for forward window — current loader uses 10 H1 bars *backward*. Should the *forward* window use the same 10, or align with the confidence head's 24? Recommend **10** to keep the head's semantic ("short-horizon momentum") and the existing trainer's `momentum_window=10` default unchanged. Different from confidence's 24 because the use-case is different (momentum agent reads this to gate *current-bar* trade entry, not multi-day position management).
- **Q3 (operator):** B3 acceleration — should the *backward* window in the comparison stay at `i-5` (current loader) or shift to `i-W:i`? Recommend `i-W:i` (W=`momentum_window`=10) for symmetry with the forward window and to match the audit's recommendation.
- **Risk:** the momentum_score label is no longer in [0, 1] guaranteed (forward |returns| can spike during news events). Clipping to [0, 1] post-normalization is fine but check that the trainer at `lightgbm_trainers.py:538-557` doesn't assume in-range targets.
- **Risk:** because B2 and B3 share the forward window, NaN-row drops at the tail apply to BOTH outputs. The trainer expects both outputs to have identical row counts; the loader pattern at `:3079` (`np.column_stack([momentum_score, acceleration])`) preserves that. Verify after fix.

---

## 6. B4 — Risk `streak_prob`

### 6.1 The leak

> Audit `docs/label_leak_audit_2026-05-05.md:188-218`. Quoted formula:
>
> ```
> # streak_prob, src/core/modular_data_loaders.py:3239-3246
> for i in range(20, n):
>   if volatility_20[i] > 1e-10:
>     vol_ratio = volatility_10[i] / volatility_20[i]
>     streak_prob[i] = clip((vol_ratio - 0.8) / 0.4, 0, 1)
> ```
>
> X matrix at `:3151` (via `get_normalized_feature_names()['risk']` → `:1341-1354`) explicitly contains `volatility_5, volatility_10, volatility_20`. Label is a deterministic linear-and-clipped expression of two columns also in X.

### 6.2 Fix path — option (a) realized-outcome label from journal

**Concrete realized-outcome definition:** label[i] = realized loss-streak metric over the window starting at row `i`, as observed in the trade journal.

This is the **only B-head where the journal is the natural label source** — `streak_prob` is conceptually "what's the probability of a losing streak starting from here" and the journal records exactly that.

Pseudo-spec (mirrors `realized_confidence_label.py:199-312`):

```
def compute_realized_streak_prob_labels(
    df, instrument, journal,
    streak_window_bars=72,        # 3 days at H1 — captures multi-day streaks
    streak_threshold=3,           # >=3 consecutive losses = "streak occurred"
    fallback_simulator=True,      # use triple-barrier as pseudo-fallback
    label_mode="blend",
):
    """
    For each row i:
      Real path:
        Walk journal forward from row i's timestamp for streak_window_bars.
        Count consecutive losing trades on `instrument` in that window.
        If consecutive_losses >= streak_threshold:    label = 1.0
        else:                                          label = 0.0
        If no journal trades in the window AT ALL:    label = NaN (delegates to pseudo)

      Pseudo path (when journal has insufficient density):
        Run triple-barrier dual-direction simulation (already implemented at
        realized_confidence_label.py:121-196) over forward window.
        Label = 1.0 if M consecutive simulated SL hits, else 0.0.

      label_mode='blend' (default): real labels override pseudo where present.
    Returns (labels, metadata) — same shape as confidence template.
    """
```

This is a more aggressive reuse of the confidence template than B1/B2/B3 because the journal-join + triple-barrier pattern is exactly applicable.

### 6.3 Files to touch

- **New:** `src/training/labels/realized_streak_prob_label.py` (~250 LOC; can directly import the journal-index helper from `realized_confidence_label.py:_build_journal_index` with a small edit — make that helper public by renaming it `build_journal_index_for_instrument`).
  - **Sub-decision:** whether to make the confidence helpers public *now* (B4) or after the post-B-track common extraction. Recommend now — it's a one-line rename in `realized_confidence_label.py`; saves duplicating ~30 lines.
- **Edit:** `src/core/modular_data_loaders.py:3239-3253` — replace the closed-form formula with a call to `compute_realized_streak_prob_labels(df, instrument, journal=...)`. Pass the journal in via the loader's signature (currently `load_rf_data` does NOT take a journal arg; add `journal: Optional[List[Dict[str, Any]]] = None` param matching `load_ridge_data`'s signature at `modular_data_loaders.py:3438`).
- **Edit:** `load_rf_data` valid-range trimming at `:3261-3266` — keep `valid_start = 20` for the warmup, change `valid_end = n - drawdown_horizon` to `valid_end = n - max(drawdown_horizon, streak_window_bars)` so the label-NaN tail rows are dropped consistently.
- **Leave untouched:** `LightGBMRiskTrainer.train` (`src/training/trainers/lightgbm_trainers.py:720`) — the dual-output `[expected_drawdown_pct, streak_prob]` shape is preserved; only the streak_prob target's *source* changes.
- **CRITICAL — leave `expected_drawdown_pct` untouched.** The audit explicitly says it is CLEAN (`label_leak_audit_2026-05-05.md:212-218`). The fix is to one of two outputs only.

### 6.4 Backup procedure

```
mkdir -p trained_data/rollback/streak_prob_pre_2026-05-06/
cp trained_data/models/joint/rf_risk.* \
   trained_data/rollback/streak_prob_pre_2026-05-06/
cp trained_data/models/joint/lightgbm_risk.* \
   trained_data/rollback/streak_prob_pre_2026-05-06/  2>/dev/null || true
for pair_dir in trained_data/models/*/; do
  pair=$(basename "$pair_dir")
  for f in rf_risk lightgbm_risk; do
    if ls "$pair_dir"$f* >/dev/null 2>&1; then
      mkdir -p trained_data/rollback/streak_prob_pre_2026-05-06/per_pair/$pair
      cp "$pair_dir"$f.* trained_data/rollback/streak_prob_pre_2026-05-06/per_pair/$pair/
    fi
  done
done
```

Commit as `chore(rollback): backup pre-leak-fix risk model artifacts (streak_prob fix)`.

### 6.5 Retrain command

**Proposed:** `scripts/retrain_streak_prob_leak_fix.py`, modelled on `retrain_confidence_leak_fix.py`. Note the risk model is dual-output, but only one output is changing — the script must train and save the full model (you can't retrain just one head of a multi-output LightGBM).

```
python scripts/retrain_streak_prob_leak_fix.py --pairs all
```

### 6.6 Validation criteria

- **streak_prob (regression in [0, 1]).** Brier score and MAE. Brier is the better metric for probability-of-event labels. **Sober prediction:** the realized-outcome label is sparse (most windows do NOT contain a 3-loss streak); class balance will be heavily skewed toward 0.0. Expect Brier ~ 0.05–0.10 (which beats predicting `0.5` for everything but isn't far above predicting `mean(y_train)`). MAE will look low simply because most labels ARE near zero.
- **expected_drawdown_pct (regression).** R² and MAE — should be UNCHANGED from pre-fix because that head's labels and X are unchanged. If it shifts more than ~10%, the loader's row-trimming logic (point 6.3 above) is wrong.
- **No fixed threshold gate.** No `min_accuracy` for risk head in `scheduled_retrain.py`.
- **Leak sanity check.** Train tiny LightGBM on X and new streak_prob label only. If R² > 0.3 or Brier improves >50% over `mean(y_train)` baseline, residual leak (likely from `volatility_*`); investigate before promotion.

### 6.7 Rollback

```
git revert <fix-commit>
cp -r trained_data/rollback/streak_prob_pre_2026-05-06/* trained_data/models/joint/
```

### 6.8 Risks + open questions

- **Q4 (operator):** `streak_window_bars` — confidence used 24 H1 (~1 day). For streak-of-losses, a longer window is justified (3 days = 72 H1 bars catches multi-session losing patterns). Recommend **72**, but ask the operator before B4 starts.
- **Q5 (operator):** `streak_threshold` — what counts as a "streak"? 3 consecutive losses is intuitive but the journal data may have very few 3-streak windows for less-traded pairs (similar to the n_real=40 problem in A1.5). 2 consecutive may give better class balance at the cost of semantic precision. Recommend **3** for parity with the rule promoted at `.claude/rules/trading.md` (`max_consecutive_losses_threshold=3` analogue), but flag for operator decision.
- **Risk:** journal density is low for non-EUR_USD pairs (A1.5: 6 of 7 pairs had only 40 real labels for confidence). Triple-barrier pseudo-fallback will dominate; operator should expect post-fix per-pair streak_prob R² to look like A1.5's confidence (noise-floor across 6 of 7 pairs).
- **Risk:** the runtime gate in `gates.py` consuming `streak_prob` may have been calibrated to the leaky scale (skewed because old labels were `clip((vol_10/vol_20 - 0.8)/0.4, 0, 1)` which pegged at 0 or 1 a lot). Post-fix labels will be much sparser at 1.0 (real streaks are rare). **Verify gate threshold semantics post-fix** before unhalt — if `streak_prob > 0.5` was the runtime block, it may now never fire.

---

## 7. Cross-cutting reminders

### 7.1 Per-commit checklist (one head per commit)

For each B-head:

1. Backup commit FIRST: `chore(rollback): backup pre-leak-fix <head> artifacts`.
2. Code commit: `fix(training): replace <head> leaky label with realized-outcome` — adds new label generator file, edits loader call site, no other diff.
3. Retrain script commit (if new): `feat(scripts): retrain <head> leak fix`.
4. Run retrain. Capture validation metrics in a `docs/plans/retrain_report_<head>_2026-05-06.md` per the A1 precedent.
5. CHECKPOINT LOG entry in `docs/superpowers/plans/2026-05-05-self-heal-self-train-roadmap.md` per the existing format.

### 7.2 What NOT to do

- Do **NOT** lower any holdout threshold. The 52% direction-head gate is unaffected (Track C territory). No B-head has a fixed gate to relax.
- Do **NOT** auto-unhalt after a B-fix lands. Operator's frozen gate (roadmap section "Auto-unhalt strict conjunctive gate") still requires Tracks A + B + C complete.
- Do **NOT** add an LLM call anywhere. All four label generators must be deterministic NumPy/pandas.
- Do **NOT** introduce mocks in tests. Per the No-Mock Rule, tests for each new label file must drive `compute_realized_<head>_labels(df, ...)` against real synthetic OHLCV (constructed in `tmp_path`). Pattern: build a 500-row DataFrame with deterministic high/low/close, call the function, assert on label values directly.
- Do **NOT** squash B1+B2+B3+B4 into one PR/commit. One head per commit per the operator's rule. (B2+B3 ship together because B3 has no independent label source — single commit, two outputs.)
- Do **NOT** modify `expected_drawdown_pct` (it's clean per audit).

### 7.3 Cross-head checks the operator should run after all four ship

- **Smoke retrain.** `python scripts/scheduled_retrain.py --pairs EUR_USD --dry-run` (verify no schema breakage).
- **Holdout regression.** Run direction-head holdout (`validate_holdout_multi_timeframe`); accuracy should not change because none of B1–B4 touch the direction head. If it shifts, a label fix accidentally changed a feature column upstream.
- **Disk audit.** `ls trained_data/rollback/` should contain four `*_pre_2026-05-06/` directories, one per fix.
- **Grep audit.** `grep -n "streak_prob\[i\] = clip" src/core/modular_data_loaders.py` should return nothing post-B4. Same shape grep for the other three formulas.

### 7.4 What this design does NOT cover

- **Track C** (direction-head 44% holdout investigation) — independent track, separate design.
- **Trend/chop/MR regime (SUSPICIOUS, not CONFIRMED)** — audit's option (a)/(b) requires forward-derived label work but the head is partly defensible due to its future-return confirmation. Defer to a Track-B-extension after B1–B4 ship.
- **Common-helper extraction** (`src/training/labels/_common.py`) — recommended as a follow-up commit after B1–B4 land, not as a prerequisite.
- **W&B control plane integration.** Each new label generator should accept its forward-window/threshold params from `pull_config(head)` (A2 surface), but wiring that is a separate post-B refactor — initial fixes hardcode the recommended defaults to keep diffs minimal.

---

## 8. Open questions consolidated (answer before B-track work starts)

| ID | Decision | Default if unanswered |
|---|---|---|
| **B1.Q1** | `vol_horizon_bars` for B1 | 24 (parity with confidence) |
| **B1.Q2** | Realized-vol formula: stddev(log_returns) vs Parkinson HL-range | stddev(log_returns) |
| **B1.Q3** | Cut-fit basis: train-slice percentile vs fixed `[0.25, 0.60, 0.85]` | train-slice percentile (no cross-split leakage) |
| **B2.Q1** | `momentum_window` for forward window | 10 (parity with current backward) |
| **B3.Q1** | Backward-window length in B3 comparison: `i-5` vs `i-W:i` | `i-W:i` (W=10, symmetric with forward) |
| **B4.Q1** | `streak_window_bars` | 72 (3 days at H1) |
| **B4.Q2** | `streak_threshold` (consecutive losses to count as streak) | 3 |
| **B4.Q3** | Make `_build_journal_index` public in `realized_confidence_label.py` for B4 reuse, or duplicate it? | rename to public helper now |
| **Cross** | Use `pull_config(head)` from A2 W&B control plane, or hardcode defaults in label generators? | hardcode defaults; W&B-wire as follow-up |

---

## 9. Verification surfaces (per CLAUDE.md honesty protocol)

For every B-head fix the operator/next session must verify against:

- `git log main..` — confirm exactly one fix commit per head (plus its backup commit).
- `grep -n "<old_formula>" src/core/modular_data_loaders.py` — should return zero hits post-fix.
- `ls trained_data/rollback/<head>_pre_2026-05-06/` — backup artifacts present.
- `trained_data/models/joint/<artifact>` mtime — newer than the fix commit.
- New label generator file exists at `src/training/labels/realized_<head>_label.py`.
- Per-head retrain report at `docs/plans/retrain_report_<head>_2026-05-06.md` with leak-detection sanity check (the "tiny LightGBM probe" from each head's section 4.6/5.6/6.6).
- CHECKPOINT LOG entry appended to `docs/superpowers/plans/2026-05-05-self-heal-self-train-roadmap.md`.

---

*End of design. Begin B1 only after the open-question table above is answered. Remember A1.5: the leak fix is the success; the post-fix metric is the truth, whatever it shows.*
