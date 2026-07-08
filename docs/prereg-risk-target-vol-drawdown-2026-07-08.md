# Pre-registration — Risk-target ML redirect: forward volatility + drawdown-state (2026-07-08)

Frozen BEFORE running `scripts/experiment_risk_target_vol_drawdown.py` or looking at any OOS
result. Per repo convention (anti-p-hacking pledge — see the crypto/factor pre-reg docs): one
construction, one feature set, one hyperparameter config, no sweep. Results are appended to §6
after the fact, never edited into the frozen sections above.

## 0. Why this task, and why now

`docs/ENGINEERING_BRAIN.md` P3 ("Redirect ML to risk targets, not direction") and operator
directive: the 52% direction wall (L-001/L-016/L-022) is a closed question — no free-data
directional edge exists at this data scale. Volatility forecasting and drawdown-state estimation
are a **different statistical object**: vol clustering (ARCH/GARCH-family autocorrelation in
|returns|) is one of the most robust, replicated findings in empirical finance, and the
efficient-market argument that kills *direction* prediction does not apply to *risk* prediction —
predicting how much a market will move is not the same claim as predicting which way. This task
builds that model family from scratch, pre-registered, walk-forward validated, and reports the
honest OOS number — including an honest FAIL if either target doesn't clear its bar.

This is **not** a retrain of the retired FX direction transformers (L-016 does not apply — this
targets risk, not direction) and **not** a duplicate of the existing live
`TCNVolatilityRegimeTrainer` (which predicts a 4-class *dispersion* regime, forward 48 H1-bars,
per-pair, wired into the FX direction ensemble's gate/filter at
`src/core/modular_inference.py:1684`). This is a new, decoupled model family: (a) a genuinely new
target — forward-realized-volatility as a *continuous regression*, not 4-class dispersion — and
(b) a wholly new target with no prior definition anywhere in the codebase — forward drawdown-state
(calm vs. stressed), which is path-dependent (max peak-to-trough decline) rather than a dispersion
statistic. Universe, horizon, and feature pipeline are also independent of the FX direction/TCN
pipeline (own `RISK_TARGET_FEATURE_PIPELINE_VERSION`, per the infra survey's recommendation) so
this work cannot desync or interact with the existing per-pair gate contract.

## 1. Frozen construction

**Universe**: 19 FX pairs in `market_data/factor/*_D.csv` (daily OHLCV, OANDA-sourced, cached
locally — no live API call needed): AUD_JPY, AUD_NZD, AUD_USD, CAD_JPY, CHF_JPY, EUR_AUD, EUR_CAD,
EUR_CHF, EUR_GBP, EUR_JPY, EUR_USD, GBP_AUD, GBP_JPY, GBP_USD, NZD_JPY, NZD_USD, USD_CAD, USD_CHF,
USD_JPY. ~3,230–3,250 daily rows per pair, 2014-01-01 → 2026-06-11 (~12.5y). FX majors/crosses have
no survivorship bias (no delisting). **Pooled across pairs** (not one model per pair): a
pair-categorical feature carries the identity; pooling gives ~58k rows total vs. ~3.2k per pair,
which is necessary for an honestly-powered walk-forward split on a daily (not intraday) sampling
frequency.

**Targets (2, both pre-registered before any run):**

1. **Forward realized volatility (regression).** For row `i`: `realized_vol[i] =
   stddev(diff(log(close[i+1 .. i+1+H]))) * sqrt(252)` (annualized), `H = 20` trading days. Reuses
   the leak-free forward-window formula already proven in
   `src/training/labels/realized_volatility_regime_label.py` (`_compute_forward_realized_vol`,
   B1 leak-fix 2026-05-06) — exposed as a new public continuous-value function
   `compute_forward_realized_volatility()` added to that same module (extends, does not fork).
   Label depends ONLY on `close[i+1 .. i+1+H]` — never on row `i` or earlier.
2. **Forward drawdown-state (binary classification).** For row `i`: within the forward window
   `close[i+1 .. i+1+H]` (same `H=20`), compute the running-peak-relative max drawdown
   `max_dd[i] = max_t‰[i+1,i+1+H]( (running_peak_t - close_t) / running_peak_t )`. Label = 1
   ("stressed") if `max_dd[i]` exceeds the 75th percentile of `max_dd` **fit on the train slice
   only**; else 0 ("calm"). New label module
   `src/training/labels/forward_drawdown_state_label.py` — greenfield (repo-wide grep confirmed
   no prior "drawdown_state"/"calm"/"stressed" definition exists anywhere), same leak-prevention
   contract and train-only-cut-fitting pattern as the existing vol-regime label, for
   methodological consistency.

**Target explicitly NOT attempted this round**: correlation-regime. The infra survey confirmed
`src/hedge/portfolio_exposure.py` is a point-in-time-current-state function with no historical
position-snapshot store to replay it against — building one is a separate, larger lift (would need
a reconstructed historical multi-asset book, which doesn't exist for the harvester/crypto_carry
shadow lanes at daily granularity going back years). Rather than force a weak proxy, this is
honestly deferred as a future P-item, not silently dropped.

**Features (13, causal/backward-looking only, computed at row `i` from data ≤ `i`; own
`RISK_TARGET_FEATURE_PIPELINE_VERSION = "2026-07-08-v1"`, decoupled from the direction pipeline's
`FEATURE_PIPELINE_VERSION`):** `realized_vol_5/10/20/60` (rolling annualized stdev of daily log
returns, trailing window), `atr_14` (true-range rolling mean / close), `hl_range_pct_14` (rolling
mean of `(high-low)/close`), `return_mean_5/20` (rolling mean daily log return), `vol_of_vol_20`
(rolling stdev of `realized_vol_5`), `volume_zscore_20` (rolling z-score of volume vs. trailing
60-day mean/std), `day_of_week` (0–4, categorical), `pair` (18-level categorical). All windows are
trailing/rolling (never cumulative-from-dataset-start) — window-invariance is a mandatory test
(mirrors `tests/test_feature_window_invariance.py`'s canary, per L-001: a feature whose value
depends on where the analysis window starts is exactly the leak class that produced the false 56–
70% direction numbers). **Tick-derived microstructure features (`src/data/tick_capture.py`) and
execution-cost features (`src/data/execution_cost_model.py`) are explicitly EXCLUDED from this
pre-registered run** — the infra survey confirmed tick capture has only ~1–2 days of history
(started 2026-07-07) and `execution_cost_model`'s persisted artifact is a current-snapshot dict,
not a historical time series joinable on timestamp. Including either would mean training on data
too thin to walk-forward validate honestly, or fabricating a historical join that doesn't exist.
Both are named as P2 follow-ups once tick history accumulates for months, not fabricated now.

**Split**: expanding walk-forward by CALENDAR DATE (not per-pair row index, since all 18 pairs
share correlated global vol regimes — a per-pair row-index split could let one pair's 2020 COVID
train fold overlap in calendar time with another pair's test fold). IS = 2014-01-01 →
2023-12-31, used internally as an 80/20 date-ordered train/validation split for LightGBM early
stopping. **OOS = 2024-01-01 → 2026-06-11**, read once, untouched until §6. This matches the
repo's standing OOS-cutoff convention (same cutoff used in the crypto and pre-2014-factor
pre-registrations). **Embargo gap = H = 20 trading days** between the train/val boundary and
between IS and OOS, so no training row's forward-label window reads into the validation/OOS
period (mirrors `temporal_split(..., gap=vol_horizon_bars)` in
`modular_data_loaders.py:4008-4015`).

**Model**: `RiskTargetTrainer` (new, `src/training/trainers/risk_target_trainer.py`) — two
independent LightGBM models (`_create_lgbm_regressor`/`_create_lgbm_classifier` from
`src/training/trainers/utils.py`, reused not forked), fixed hyperparameters (no tuning after
seeing OOS): `n_estimators=500` with early stopping (`n_iter_no_change=50`) on the IS validation
slice, `max_depth=6`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`. No feature
scaling (tree-based; scale-invariant — noted honestly in meta as `"scaler": "not_applicable"`
rather than fabricating an identity scaler).

**Correction note (disclosed before the reported §6 run, not a post-hoc metric change to win a
bar)**: the first smoke run of this exact frozen construction trained the vol head as an
unconstrained regression on the raw annualized-vol level. That is a modeling-domain bug, not a
metric-tuning choice: an unconstrained regressor has no positivity floor, and it produced
non-positive raw outputs on ~7.6% of OOS rows (verified: `pred.min()==` the clip floor on 890/
11685 OOS rows) — which makes QLIKE (which divides by the prediction) explode to ~3.9×10⁷ even
though R²/pinball/MAE on that same run showed real skill (R² 0.77 OOS, pinball 4× better than the
naive baseline). This is the textbook reason vol forecasting is done in log-space (Andersen &
Bollerslev 1998 log-normal working assumption): fit the regressor on `log(annualized_vol)`,
exponentiate the prediction back. This enforces positivity by construction and is now the frozen
construction reported in §6 — changed BEFORE looking at the log-space run's OOS numbers, and the
same fix would apply regardless of whether it helped or hurt the final verdict (it is a
correctness fix to a broken evaluation metric, not a lever pulled to pass the gate).

## 2. Metrics + pre-registered "learnable" bar (frozen before any run)

**Target 1 (forward realized vol, regression):**
- Primary: **QLIKE** = `mean(actual/pred - log(actual/pred) - 1)` (standard vol-forecast loss,
  lower is better, penalizes under-prediction of variance asymmetrically) and **pinball loss at
  q=0.5** (median forecast calibration). Secondary (interpretability only): R², MAE.
- Baseline: **naive persistence** — predict tomorrow's 20-day-forward realized vol as today's
  trailing 20-day realized vol (`realized_vol_20` feature itself). This is a strong baseline (vol
  clustering means persistence already captures most of the signal) — deliberately hard to beat,
  not a strawman.
- **Learnable bar**: model's OOS QLIKE must beat the naive-persistence baseline's OOS QLIKE, AND
  OOS R² > 0. Both must hold; failing either is reported as "not learnable at this bar", not
  rounded up.

**Target 2 (forward drawdown-state, binary classification):**
- Primary: **AUC-ROC** and **Brier score** (calibration).
- Baseline: predict the train-set base rate (≈0.25, by construction of the 75th-percentile cut)
  for every row. Baseline AUC = 0.5 by definition; baseline Brier = `p(1-p)` at the base rate.
- **Learnable bar**: OOS AUC ≥ 0.55 AND OOS Brier beats the base-rate baseline. 0.55 is a modest,
  explicitly-not-generous bar (vs. the 52% direction wall, which is ~2pp above a 50% coin flip and
  was ruled "no edge" — 0.55 AUC is a comparable-magnitude effect size, so this bar is deliberately
  conservative, not inflated to manufacture a pass).

**No sweep, no multiple-testing inflation**: exactly 2 targets × 1 frozen configuration each = 2
tests total this round. This is a new model family outside the prior return/Sharpe significance
campaign's `N_trials` budget (that budget gates *alpha* claims via DSR/Bonferroni on Sharpe
ratios — QLIKE/AUC skill claims are a different statistical object and are not folded into that
counter), but the "no sweep after seeing results" discipline is identical.

## 3. Hard constraints (repeated here so the gate can be checked against this doc)

- Offline/research only. No execution, unhalt, arm, or broker path touched. No live OANDA API
  call (all data read from the cached `market_data/factor/*_D.csv` panel).
- The trained model outputs a risk ESTIMATE (predicted vol / stressed-probability) for future
  sizing/gating consumption — it must never be reframed as a directional trade signal, must never
  enter the trade hot-path, and must never touch `execution.py`/`gates.py`/`state_engine`/the
  practice pin. Verified structurally (import-boundary test), not just by convention.
- Retrain/write path is gated: a candidate model may only overwrite the incumbent artifact if it
  does not regress on the OOS gate metric, reusing the exact same gate primitives
  (`_cli_gate_verdict`/`_cli_gate_degenerate`/`_log_cli_gate`) already enforced in
  `cli/training_ops.py::retrain_gates()` — imported, not duplicated.
- Nothing armed, nothing live, nothing added to `ScannerConfig`/the hot-path config surface this
  session. The "how it feeds sizing/gating" deliverable is a documented READ-ONLY interface
  (a `predict_risk_state()` function returning a plain dict), analogous to
  `src/hedge/portfolio_exposure.py`'s read-only report pattern — not a live wire-in.

## 4. Independent verification plan

Two separate domain-specialist verifiers, dispatched after implementation is green:
1. **Security Engineer** — structural: zero imports of `execution.py`/`gates.py`/
   `state_engine`/broker clients/OANDA order endpoints from any new module; practice pin and
   `.claude/state.json` untouched by `git diff --stat`; no new hot-path coupling.
2. **Model QA Specialist** — quant/stats: independently re-derive the leak-safety proof for both
   labels (forward-window-only dependency), confirm the walk-forward split has no embargo gap
   violation, confirm QLIKE/pinball/AUC/Brier are computed correctly and the baselines are fair
   (not strawmen), and confirm the reported "learnable"/"not learnable" verdict matches the
   frozen bars in §2 exactly (no post-hoc bar-lowering).

## 5. What ships regardless of the gate result

Whichever target(s) clear their bar, the trained artifact(s) are written to
`trained_data/risk_targets/models/` (a new namespace, decoupled from the FX per-pair
`trained_data/models/{PAIR}/` ensemble dirs `gates.py` scans) with full meta contract
(`feature_names`, `RISK_TARGET_FEATURE_PIPELINE_VERSION`, QLIKE/AUC OOS numbers, "learnable" bool
per target). A target that fails its bar is still reported honestly in §6 — this is a valid,
useful outcome (an honest negative), consistent with the operator's stated preference for a
half-done task reported honestly over a "complete" that papers over a gap.

## 6. Results (appended after the fact — empty at pre-registration time)

Run: `scripts/experiment_risk_target_vol_drawdown.py` →
`trained_data/backtests/risk_target_vol_drawdown_result.json`. Pooled 19-pair universe, n_train=
38,036, n_val=9,526, n_test (OOS, 2024-01-01→2026-06-11)=11,685 rows.

### Target 1 — forward realized volatility (regression): **LEARNABLE (bar cleared)**

| | OOS QLIKE (↓ better) | OOS pinball@median (↓) | OOS R² (↑) | OOS MAE |
|---|---|---|---|---|
| **Model** | **0.0704** | **0.0052** | **0.775** | 0.0104 |
| Naive persistence (trailing realized_vol_20) | 1.537 | 0.0238 | −0.803 | — |

Model beats the naive-persistence baseline by ~22× on QLIKE and clears OOS R² > 0 comfortably
(the naive baseline is actually NEGATIVE R² OOS — trailing-20-day persistence alone does not
track 2024-2026 forward vol well; the model does). **Both frozen bar conditions hold. Verdict:
LEARNABLE.**

### Target 2 — forward drawdown-state (binary classification): **NOT LEARNABLE (bar not cleared)**

| | OOS AUC (↑) | OOS Brier (↓) |
|---|---|---|
| **Model** | **0.625** | **0.232** |
| Baseline (train base-rate ≈0.25 / AUC 0.5) | 0.500 | 0.143 |

AUC clears the 0.55 bar (0.625, a real, if modest, ranking skill above chance). **But** Brier
score is WORSE than the constant base-rate baseline (0.232 vs 0.143) — the model's probability
CALIBRATION is poor even though its ranking has some skill (a common LightGBM-with-imbalanced-
classes-and-no-calibration-step failure mode: `class_weight="balanced"`, inherited from
`_create_lgbm_classifier`'s default, pushes predicted probabilities away from the true base rate).
The pre-registered bar requires BOTH conditions; **since Brier fails, this target is reported as
NOT LEARNABLE at the frozen bar** — an honest negative, not rounded up because AUC alone looked
promising. A follow-up (isotonic/Platt recalibration of the classifier's raw probabilities, or
dropping `class_weight="balanced"`) would very plausibly fix the Brier failure without touching
the ranking skill — but doing that now, having already seen this result, would be exactly the
post-hoc bar-gaming this pre-registration exists to prevent. It is named here as a legitimate
next-round hypothesis, not applied.

### Correction note reference

The reported numbers above are from the log-space vol-regression fix described in §1's
"Correction note" (fit `log(annualized_vol)`, exponentiate predictions) — the first raw-level run
produced a nonsensical QLIKE (~3.9×10⁷) due to unconstrained-regression non-positive predictions,
disclosed and fixed before this reported run, not after seeing these final numbers.

### What shipped

Per §5: since Target 1 cleared its bar, `trained_data/risk_targets/models/risk_target_model.{pkl,
meta.json}` was written (both heads are saved together as one `RiskTargetTrainer` artifact —
Target 2's non-clearance is recorded honestly in the artifact's `metrics` block, not hidden; the
artifact should currently be consumed for its VOLATILITY output only, per the honest verdict
above, until a recalibrated drawdown-state candidate clears the bar).
