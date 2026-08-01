# Pre-registration — Risk-Target Forward-Vol ENSEMBLE Challenger (2026-08-01)

**Status: REGISTERED, not yet run.** Frozen before any challenger model is fitted.
Operator prompt (2026-08-01): "the earlier architecture was way better with an ensemble of
different models" — this experiment is the honest test of that hypothesis on the one target
that has survived its gate. Scope: the forward-volatility head ONLY. The drawdown-state head
failed its pre-registered bar (OOS Brier 0.2363 vs 0.1432 base-rate; evidence package
`d4bed301…df05f8` REJECTED) and is out of scope; an ensemble is not a route around a failed
gate. FX direction remains retired (LESSONS L-016/L-022) and is not touched here.

## 1. Hypothesis (frozen)

H-VE1: An equal-ish ensemble of heterogeneous forward-vol estimators — (a) the incumbent
LightGBM regressor, (b) a HAR-RV linear model (Corsi 2009: RV_d, RV_w, RV_m components),
(c) a scikit-learn GradientBoosting quantile model at q=0.5 — combined by simple average of
log-vol predictions, achieves LOWER pooled OOS QLIKE than the incumbent single LightGBM
champion (evidence package `5d6af6ba…c2b30e`, OOS QLIKE 0.05039).

Promotion bar (frozen): challenger pooled OOS QLIKE < incumbent's by ≥ 2% relative
(i.e. ≤ 0.04938) AND per-pair QLIKE not worse than incumbent on more than 4 of 19 pairs.
A tie or marginal win does not promote — churn without improvement is a cost.

## 2. Frozen design (identical to the 2026-07-08 incumbent prereg except the model family)

- Data: same 19 pairs, same `market_data/factor/*_D.csv` snapshot (dataset_id
  `fx-factor-daily-2014-2026`), same partition hashes as the incumbent's DatasetManifest.
- Features: IDENTICAL feature pipeline and `RISK_TARGET_FEATURE_PIPELINE_VERSION` as the
  incumbent — reused by import from `risk_target_trainer.py`, not forked. No new features
  (a feature change would be a different experiment identity).
- Target: same log forward realized vol, H = 20 trading days.
- Split: same expanding walk-forward by calendar date; IS 2014-01-01→2023-12-31 (80/20
  date-ordered internal split), OOS 2024-01-01→2026-06-11 read once; embargo gap H=20.
- Ensemble members (frozen hyperparameters, no tuning after OOS):
  1. LightGBM: exactly the incumbent's frozen params (n_estimators=500, early stop 50,
     max_depth=6, lr=0.03, subsample=0.8, colsample_bytree=0.8, seed 20260708).
  2. HAR-RV: OLS on [RV_daily, RV_weekly(5d mean), RV_monthly(22d mean)] in log space,
     fit on the same IS rows. No regularization sweep.
  3. GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=300, max_depth=3,
     learning_rate=0.05, subsample=0.8, random_state=20260801).
- Combiner: arithmetic mean of the three log-vol predictions. No learned weights, no
  stacking (a stacked/learned combiner is a separate future experiment identity).
- Seeds: 20260801 for the new member; incumbent member keeps 20260708.

## 3. Trial accounting

This is ONE new confirmatory trial. It must be appended to the authoritative ledger in
`src/research/trial_budget.py` (TRIAL_LEDGER) before the OOS read, raising the family-wise
N accordingly; the DSR/Bonferroni machinery does not apply to QLIKE loss comparison, but the
ledger entry records the attempt so repeated challenger-mining is visible and bounded.
Challenger attempts against the risk-target champion are capped at 2 per quarter (frozen).

## 4. Governance path (same rail as the incumbent — no shortcuts)

Run through `scripts/run_risk_target_evidence_slice.py` extended with a `--challenger vol_ensemble`
mode: DatasetManifest (same hashes) → signed JobManifest → training → EvaluationReport
(challenger AND incumbent metrics side by side) → EvidencePackage → import → QUARANTINED →
promotion ONLY via the Phase L promotion service with explicit operator approval, which
atomically repoints the champion and preserves the incumbent for rollback. If the challenger
loses, its package is REJECTED and preserved as a first-class negative result.

## 5. What would falsify the operator's ensemble hypothesis here

If the ensemble fails the bar, the recorded conclusion is: heterogeneous averaging adds no
material forward-vol accuracy at daily FX scale over a single gradient-boosted model, and
the "ensemble was better" memory belongs to the direction stack whose apparent superiority
was the L-001 artifact. One clean negative closes the question; do not re-run with tweaked
members absent a materially new input (new data family or new frequency).
