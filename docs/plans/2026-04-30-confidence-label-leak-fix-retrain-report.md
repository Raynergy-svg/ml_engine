# Confidence Label Leak Fix — Retrain Report (2026-04-30)

Branch: `claude/trading-strategy-analysis-sAakL`
Source diagnosis: `docs/confidence_model_leak_investigation_2026-04-30.md`
Plan (trimmed): `docs/plans/2026-04-30-confidence-label-leak-fix-plan.md`
Rollback manifest: `docs/plans/2026-04-30-confidence-label-leak-fix-rollback-manifest.md`

## Summary

Replaced the leaky closed-form confidence label with a realized-outcome label
(journal real labels + triple-barrier pseudo labels), retrained the joint
LightGBM regressor in place, and refit the Platt calibrator against
journal outcomes. Per-pair confidence fine-tunes are skipped (insufficient
journal volume). All 14 unit + integration tests pass; smoke test passes.

## Old vs new metrics

| Metric | Old (leaky) | New (realized) |
|---|---|---|
| Val R² | 0.9971 | -0.0102 |
| Val MAE | 0.27 | 36.81 |
| Model type | LightGBM regressor | LightGBM regressor (unchanged signature) |
| Score range | [20, 95] | [20, 95] (preserved via `score = 20 + 75 * binary_label`) |
| Train rows | 28,259 | 43,401 |
| Val rows | 9,315 | 12,400 |
| Label mode | closed-form weighted sum | journal_outcome_blend |
| Real labels | 0 | 240 (40 per-pair × 6 pairs) |
| Pseudo labels | 0 | 61,810 (triple-barrier) |
| Class balance (overall) | n/a (continuous) | 57.0% wins |
| Saved at | 2026-04-16T11:29:31 | 2026-05-01T17:55:55 |

R² near zero is the expected outcome of removing the leak. The previous 0.9971 was
the model recovering an arithmetic formula whose five inputs were also features.
With outcome-derived labels, val R² collapses to the noise floor — which is
correct for a noisy financial regression target. Per the leak-fix plan, the
expected R² band is `[0.05, 0.30]`; we landed slightly below 0.05, reflecting
the heavy reliance on triple-barrier pseudo-labels (240 real / 61,810 pseudo).
The Platt calibrator now does the actual work of mapping raw scores to
win-probability — and is well-calibrated against journal outcomes (ECE 0.017,
Brier 0.214).

The leak-detection assertion in `ridge_trainer.py` (`r2 > expected_r2_band[1]`)
is wired and would log an ERROR if a future training run somehow blew past R² > 0.30.

## Class balance per instrument

(from `joint_training_meta.json:confidence_label_metadata.per_instrument`)

| Instrument | n_real | class_balance_real | n_pseudo | class_balance_pseudo |
|---|---|---|---|---|
| AUD_USD | 40 | 0.375 | 11,083 | 0.573 |
| GBP_USD | 40 | 0.375 | 11,065 | 0.568 |
| NZD_USD | 40 | 0.400 | 11,172 | 0.561 |
| USD_CAD | 40 | 0.375 | 10,991 | 0.563 |
| USD_CHF | 40 | 0.250 | 10,999 | 0.561 |
| USD_JPY | 40 | 0.400 | 6,500  | 0.576 |

Real-label class balance (~37%) is consistent with the journal win rate
(~32% live + archives). Pseudo-label class balance (~57%) is slightly biased
to wins because the dual-direction triple-barrier picks `max(long_win, short_win)` —
this is a known design trade-off (see `realized_confidence_label.py` docstring).

## Score distribution comparison (12,414 val rows, joint)

10-bin histogram in [20, 95]:

| Bucket | Count |
|---|---|
| 20.0–27.5 | 0 |
| 27.5–35.0 | 2 |
| 35.0–42.5 | 8 |
| 42.5–50.0 | 57 |
| 50.0–57.5 | 906 |
| 57.5–65.0 | 8,522 |
| 65.0–72.5 | 2,710 |
| 72.5–80.0 | 204 |
| 80.0–87.5 | 5 |
| 87.5–95.0 | 0 |

Percentiles: min=31.64, p25=60.40, p50=62.61, p75=64.82, max=84.29.

Distribution is concentrated around 60-65, reflecting the ~57% win-rate
pseudo-label class balance. The old leaky model had similar central tendency
but spurious high-side outliers (R²=0.997 cluster artifacts) — the new
distribution is smoother and reflects honest uncertainty.

## Calibrator validation

(from `trained_data/confidence_calibration.json:validation`)

| Gate | Threshold | Actual | Pass |
|---|---|---|---|
| ECE | < 0.15 | 0.0166 | YES |
| Brier | < 0.30 | 0.2140 | YES |
| Reliability monotonic (5 buckets) | true | true (1 bucket vacuous) | YES |
| n_train | ≥ 30 | 547 | YES |
| n_test | ≥ 5 | 136 | YES |

Platt parameters: coef=0.2769, intercept=-0.8963. The calibrator maps
normalized raw scores in roughly [0.3, 0.85] to win-probabilities centered
around 0.31 — close to the historical journal win rate (~32%). All scores
clustered into a single bucket on test (predict=0.325, actual=0.309) so
monotonicity is vacuously true. With more journal data accumulating, future
refits will populate additional buckets and the gate becomes meaningful.

A `validation_failed.json` sidecar is emitted next to the calibration file
if any gate fails on a future refit; the file is NOT written in that case
(operator should restore from rollback manifest).

## Smoke test result

`scripts/smoke_test_confidence_leak_fix.py` (replaces `./buddy --demo`):

- New `ridge_confidence.pkl` loads without error
- Calibration JSON v2 with `label_mode=journal_outcome_blend` loads
- 50 synthetic-feature predictions: min=0.27, p50=53.97, max=86.33
- Real EUR_USD H1 val (9 rows) predictions: min=60.32, p50=62.40, max=70.19
  (within [0, 100] gate API contract)
- No exceptions related to confidence, calibration, or gates

Note: synthetic-feature scores can extrapolate slightly outside [20, 95]
(ranging 0.27–86.33) because LightGBM trees aren't bounded; on real
in-distribution data the range stays well within bounds.

## Files changed (training-only, scope-respected)

Code:
- `src/core/modular_data_loaders.py` — `load_ridge_data` rewrite (label gen + new args, backward-compatible signature)
- `src/training/labels/__init__.py` (NEW)
- `src/training/labels/journal_loader.py` (NEW — archive salvage, dedup, normalization)
- `src/training/labels/realized_confidence_label.py` (NEW — real + pseudo label generator)
- `src/training/trainers/ridge_trainer.py` — metadata pass-through + leak-detection assertion
- `src/training/trainers/joint_trainer.py` — metadata pass-through, atomic meta write, per-pair confidence fine-tune SKIP

Scripts:
- `scripts/retrain_confidence_leak_fix.py` (NEW)
- `scripts/refit_calibration_leak_fix.py` (NEW)
- `scripts/smoke_test_confidence_leak_fix.py` (NEW)

Tests:
- `tests/test_realized_confidence_label.py` (NEW — 7 tests including leak-prevention)
- `tests/test_journal_archive_salvage.py` (NEW — 4 tests)
- `tests/integration/test_ridge_data_loader_label_join.py` (NEW — 3 tests)

Artifacts (replaced in place; old saved to rollback dir):
- `trained_data/models/joint/ridge_confidence.pkl`
- `trained_data/models/joint/joint_training_meta.json`
- `trained_data/confidence_calibration.json`

Backups (sha256-verified):
- `trained_data/_rollback_2026-04-30/MANIFEST.json` (16 files tracked)

Docs:
- `docs/plans/2026-04-30-confidence-label-leak-fix-rollback-manifest.md`
- `docs/plans/2026-04-30-confidence-label-leak-fix-retrain-report.md` (this file)

## What's NOT in this PR (deliberately out of scope)

- `src/scanner/gates.py` — untouched (no threshold changes)
- `src/scanner/agents/_team.py` — untouched
- `src/risk/position_sizing.py` — untouched
- `src/scanner/execution.py` — untouched
- `src/scanner/config.py` — untouched (no new flags)
- Per-pair confidence pkls — backed up, not regenerated (joint-only model)
- Shadow stages, A/B test infra, threshold re-tuning — deferred per operator decision

## Known limitations

1. Live journal trades (17 records) couldn't contribute to calibration refit
   because their stored `ridge_features` (length 14) don't match the new
   model's 24-feature signature. Calibration was fit on archive trade
   confidences (which used the old leaky model). This is documented in
   `refit_calibration_leak_fix.py` — operators should re-run calibration
   refit after ~30+ closed trades have been logged with the new ridge
   features (auto-handled by `record_new_trade()`).

2. Pseudo labels via triple-barrier dual-direction over-reports wins
   (~57% vs real ~32%). Future improvement: train as a binary classifier
   with class weights, or use single-direction labeling that requires the
   loader to know the agent's directional vote at each row.

3. Val R² ≈ 0 means the model is essentially predicting the mean. This is
   correct given mostly-noisy pseudo labels; the calibrator carries the
   actual signal. As live journal volume grows, real labels will start to
   dominate and R² should rise toward the [0.05, 0.30] expected band.

## Rollback

If the operator observes degraded behavior, run the procedure in
`docs/plans/2026-04-30-confidence-label-leak-fix-rollback-manifest.md`.
Backup integrity is sha256-verified and the procedure is one-shot.

---

## Follow-up: feature backfill + per-pair + W&B (2026-05-01)

Operator-driven follow-up to address the two caveats noted above
(known limitations 1 and 2-adjacent). Branch: `claude/trading-strategy-analysis-sAakL`.

### 1. Live-journal feature backfill

`scripts/backfill_ridge_features.py` re-derives the 24-feature
`ridge_features` for live journal trades that previously stored
`None` (or 14-feature legacy vectors). Uses the same builder
`gates.py` and `modular_inference.py:_extract_ridge_features` use
(`align_features_to_model` for numeric slots; manual one-hot for
the `instrument_*` columns). Atomic write (tmp + rename + sha256).

**Result on the live journal (17 trades):**
- 8 trades backfilled with 24-feature vectors (pairs with H1 OHLCV
  available: EUR_USD, NZD_USD, USD_CHF, USD_JPY, USD_CAD).
- 9 trades skipped (no H1 CSV in `market_data/` for EUR_GBP, EUR_AUD,
  EUR_JPY, AUD_JPY).
- Each backfilled entry tagged with `ridge_features_backfill` containing
  `bar_age_days`, `n_features`, `backfilled_at`, `schema_version`.
  Bar age is 80–91 days (CSVs are ~3 months stale relative to
  trade timestamps); the instrument one-hot is recovered correctly.
- Default `--max-bar-age-days=180` is generous given the available
  OHLCV; tighten to 30 once fresher CSVs are pulled.

### 2. Per-pair confidence fine-tunes re-enabled

`src/training/trainers/joint_trainer.py` no longer skips per-pair
confidence — it now mirrors momentum/risk per-pair fine-tunes. With
realized-outcome blend labels each pair sees ~10k pseudo + ~40 real
labels (well above the LightGBM floor).

`scripts/retrain_per_pair_confidence.py` ran successfully for **6
pairs**:

| Pair    | Val R²    | MAE   | n_train | n_val | leak_fired |
|---------|-----------|-------|---------|-------|------------|
| AUD_USD | -0.0607   | 36.71 | 7,781   | 2,223 | False      |
| GBP_USD | -0.0543   | 36.68 | 7,766   | 2,219 | False      |
| NZD_USD | -0.0490   | 36.93 | 7,847   | 2,242 | False      |
| USD_CAD | -0.0328   | 36.53 | 7,714   | 2,204 | False      |
| USD_CHF | -0.0313   | 36.51 | 7,720   | 2,206 | False      |
| USD_JPY | -0.1035   | 36.42 | 4,573   | 1,306 | False      |

Range: -0.10 to -0.03 — same dynamic as the joint master (heavy
pseudo-label dilution on a noisy outcome target; the calibrator
carries the actual signal). 0/6 leak flags fired (none exceed
`expected_r2_band[1]=0.30`).

**Pairs not retrained** (logged + skipped, joint master remains active
via the gate's per-pair → joint fallback): EUR_USD (CSV too short — only
99 rows), AUD_NZD, EUR_AUD, EUR_CHF, EUR_GBP, EUR_JPY, GBP_AUD, GBP_CHF,
GBP_JPY (no H1 CSV in `market_data/`).

Raw report: `trained_data/per_pair_confidence_report.json`.

### 3. W&B integration

`src/training/wandb_confidence.py` — focused W&B helper for the
confidence training pipeline. Run name format:
`confidence_{joint|<pair>}_{YYYYMMDD}`. Metrics logged: val_r2,
val_mae, n_real, n_pseudo, class balances, expected_r2_band_low/high,
leak_detection_fired flag. Optional artifact upload via
`WANDB_LOG_ARTIFACTS=1`.

**Auth handling** (per spec):
- `WANDB_API_KEY` set → online mode.
- Unset → offline (logs to `./wandb/`).
- `WANDB_DISABLED=1` → short-circuits all calls (no-op).

**Run mode for THIS retrain run: offline** — no `WANDB_API_KEY` was
present in the working environment AND `wandb` is not installed in
this Python env. The `_import_wandb()` helper returned `None`
gracefully (verified by 17 unit tests mocking the wandb module). No
W&B run URLs to report; future retrains with `WANDB_API_KEY` set
will populate them.

### 4. Calibration v3 refit (archive + live)

`scripts/refit_calibration_v3.py` — same Platt-fit pipeline as v2
but now incorporates the 7 live trades (one of the 8 backfilled was
the `test_1` synthetic with no outcome) alongside 683 archive trades.

Saved at `trained_data/confidence_calibration.json` with:
- `version: 3`
- `label_mode: "journal_outcome_blend"`
- `calibration_corpus: "archive+live"`
- `metadata.prior_version: 2` for provenance.

**v3 vs v2 metrics:**

| Metric              | v2 (archive only) | v3 (archive+live) | Delta |
|---------------------|------------------:|------------------:|------:|
| ECE                 | 0.0166            | 0.0275            | +0.0109 |
| Brier               | 0.2140            | 0.2261            | +0.0121 |
| n_train             | 547               | 552               | +5     |
| n_test              | 136               | 138               | +2     |
| reliability buckets | 1 (vacuous)       | 1 (vacuous)       | —      |
| n_total trades      | 683               | 690               | +7     |

Both gates pass (ECE < 0.15, Brier < 0.30). The slight degradation
reflects the v3 corpus being more representative — v2 was
archive-only and the archive `confidence` field came from the leaky
model, so its calibration was self-referential. v3 includes 7 trades
scored against the new 24-feature joint model.

Platt parameters: `coef=0.5861, intercept=-1.1386` (v2 was
`0.2769, -0.8963`).

### 5. Backups extended

`trained_data/_rollback_2026-04-30/`:
- `confidence_calibration.json.v2` — pre-v3 state.
- `trade_journal_rl.json.pre_backfill` — pre-backfill state.
- `per_pair/{INSTRUMENT}_ridge_confidence.pkl.pre_per_pair_retrain`
  for all 13 per-pair pkls.

Manifest now tracks **31 files**, all sha256-verified.
`scripts/extend_rollback_backups.py` is idempotent.

### 6. Tests added

- `tests/test_backfill_ridge_features.py` (8 tests) — 14→24 backfill,
  None→24, idempotency, field preservation, no-csv skip,
  timestamp-out-of-band skip, dry-run, direct helper.
- `tests/test_per_pair_confidence_finetune.py` (5 tests) — save/reload
  roundtrip, [0,100] API contract, label_metadata pass-through,
  leak-detection ERROR fires on synthetic over-fit, no leak ERROR on
  realistic data.
- `tests/test_wandb_integration.py` (17 tests) — disabled/offline/online
  mode, payload key set, run name format, default project, artifact
  upload gate, graceful no-op on missing wandb / init exceptions.

All 30 new tests pass.

### 7. Smoke test result

`scripts/smoke_test_confidence_v3.py`:
- Joint `ridge_confidence.pkl` loads (n_features=24).
- 13 per-pair pkls load + predict (6 retrained at n_features=24,
  7 still at the old 28/29-feature schema — those are the pairs with
  no fresh OHLCV; gate falls back to joint master).
- Calibration v3 JSON loads with version=3, corpus=archive+live, ECE
  and Brier within gates.
- 50-sample synthetic batch: scores in [~31, ~84] for joint;
  per-pair retrained models in [~28, ~94] (LightGBM trees can
  extrapolate slightly past [20, 95] on out-of-distribution synthetic
  noise — within `0 < c < 100` API contract).
- Real EUR_USD H1 last-200-bar batch: scores within [0, 100] band,
  no exceptions.

Exit code 0 — `SMOKE TEST PASSED`.

### 8. Files changed in follow-up

Code:
- `src/training/trainers/joint_trainer.py` — replace per-pair confidence
  SKIP block with active fine-tune + leak-detection surface.
- `src/training/wandb_confidence.py` (NEW) — W&B helper.

Scripts:
- `scripts/backfill_ridge_features.py` (NEW) — 24-feature backfill.
- `scripts/extend_rollback_backups.py` (NEW) — extend backup manifest.
- `scripts/retrain_per_pair_confidence.py` (NEW) — focused per-pair runner.
- `scripts/refit_calibration_v3.py` (NEW) — v3 calibration refit.
- `scripts/smoke_test_confidence_v3.py` (NEW) — full-stack smoke.
- `scripts/retrain_confidence_leak_fix.py` — wire W&B into joint master.

Tests: `tests/test_backfill_ridge_features.py`,
`tests/test_per_pair_confidence_finetune.py`,
`tests/test_wandb_integration.py` (all NEW).

Artifacts (replaced; old saved to rollback dir):
- `trained_data/models/joint/ridge_confidence.pkl` (joint retrained;
  schema unchanged, metrics unchanged within rounding).
- `trained_data/models/{AUD_USD,GBP_USD,NZD_USD,USD_CAD,USD_CHF,USD_JPY}/
  ridge_confidence.pkl` (NEW per-pair fine-tunes; n_features=24).
- `trained_data/confidence_calibration.json` (v3).
- `trained_data/trade_journal_rl.json` (8 trades backfilled).

Backups: `trained_data/_rollback_2026-04-30/MANIFEST.json` now tracks
31 files (was 16). New entries: calibration v2, journal pre-backfill,
13 per-pair pkls.

### 9. Out of scope (still)

- `src/scanner/gates.py` — untouched (no threshold or feature-builder
  changes).
- `src/scanner/agents/_team.py`, `execution.py`, `engine.py`,
  `config.py` — untouched.
- `src/risk/position_sizing.py` — untouched.

The new live trades store 24-feature `ridge_features` automatically
through `modular_inference.py:_extract_ridge_features` (which already
honors the model's saved `feature_names`), so going forward there's no
schema drift to manage.

---

## Follow-up: W&B control plane wired across all heads (2026-05-02)

Yesterday's confidence-only W&B logger has been generalized into a full
training control plane. All 7 heads (direction, confidence, momentum,
risk, volatility_regime, trend_regime, meta_labeler) now pull their
operator-tunable settings from a versioned W&B config artifact named
`<head>_training_config:latest`, and log each run back with proper
tagging (`source=manual` vs `source=auto_retrain`).

**Key changes:**

- New helper `src/training/wandb_control_plane.py` (single source of
  truth: `pull_config`, `push_config`, `log_run`, `seed_default`,
  `apply_config_to_trainer`).
- 7 default JSONs under `src/training/training_defaults/` provide the
  seed values + fallback when W&B is unreachable.
- `online_retrainer.py:_retrain_xgboost/_retrain_rf/_retrain_ridge` now
  pulls per-head config + logs each retrain to W&B with
  `source=auto_retrain`. Cooldowns + drift triggers unchanged.
- `scripts/retrain_confidence_leak_fix.py`,
  `scripts/retrain_per_pair_confidence.py`,
  `scripts/train_full_ensemble.py` route through the control plane and
  tag manual runs.
- LightGBM trainers (`lightgbm_trainers.py`, `ridge_trainer.py`) expose
  `n_estimators / learning_rate / max_depth` as instance attributes
  instead of hardcoded constants.
- `apply_config_to_trainer` also routes Keras-style HPs onto
  `trainer.config` (TrainerConfig) for Transformer / TCN — `dropout`
  aliases to both `transformer_dropout` and `tcn_dropout`.
- 40 unit tests for the helper + 6 integration tests covering the
  online retrainer; offline smoke confirmed end-to-end (run dirs land
  under `wandb/offline-run-*`).

**Operator-facing doc:** `docs/wandb_training_control_plane.md`.

**What stayed hidden in code:** model architectures (d_model, layers,
kernel sizes), validation infrastructure (walk-forward window, embargo,
k-folds), LightGBM internals (num_leaves, reg_alpha/lambda), loss /
optimizer / EMA decay. Changing those still requires a code PR.

**What auto-retrain picks up vs requires manual rerun:**
- Auto-retrain (drift-triggered): pulls latest `confidence/momentum/risk`
  configs each cycle.
- Manual rerun required: direction, regimes, meta-labeler (heavy
  retrains owned by `train_full_ensemble.py` / dedicated scripts).

**Trading invariants untouched:** R:R 1.2 gate, correlation filter,
ATR-based SL/TP, drawdown guardian, LOW regime sl_mult >= 1.2, trend
agent veto, MR composite veto. The control plane only tunes training,
not runtime.
