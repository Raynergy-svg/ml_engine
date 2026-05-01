# Plan: Confidence Label Target-Leak Fix (Tier-7)

**Date:** 2026-04-30
**Status:** Drafted — pending operator approval at Stage 0 gate
**Source diagnosis:** `docs/confidence_model_leak_investigation_2026-04-30.md`
**Audit anchor:** `docs/ml_architecture_audit_2026-04-30.md` §6 Q1

---

## 0. TL;DR
The confidence label `y_confidence` at `src/core/modular_data_loaders.py:3335-3426` is a closed-form weighted sum of five indicator columns that are also features. We replace it with a journal-outcome-derived label, retrain in shadow, refit the Platt/Isotonic calibrator on the new distribution, re-tune `min_confidence` against measured shadow distributions, and cut over only after gate-rejection-rate stays within ±20% of baseline for one full review window. Old artifacts kept as `*.pre_leak_fix.bak` for rollback.

**Chosen rollout:** Strategy B (Shadow) + Strategy C (threshold re-tune) folded into Stage 4. Hard cut (A) and bandit (D) rejected — see §5.

---

## 1. Pre-flight (Stage 0)

### 1.1 Data inventory established by planner
1. **Live journal** (`trained_data/trade_journal_rl.json`): 17 trades, 16 closed.
2. **Archive journals** (`trained_data/archive/trade_journal_rl_*.json`): six files, each ~365 trades, **JSON-corrupt** (parser breaks at line 11859). Tolerant parse (truncate at last `\n  },\n` and append `]`) yields 364 closed trades from `trade_journal_rl_20260323_203742.json`. Older two archives use a schema with no top-level `sl_pips` — must compute risk distance from `entry_price`/`sl_price`.
3. **Realized-confidence pipeline** (`src/scanner/automation/event_handlers.py:240-269`): writes `target = ridge_conf ± abs(pnl_pips)*0.5` clamped to [0,100] into `maml.record_outcome(... target_confidence=target)`. **This is not a clean realized signal — it's the leaky ridge_conf with a pip-bonus shift.** Cannot be used as the new label without redefinition.
4. **Stored ridge features per closed trade** (`src/scanner/engine.py:7098-7099`, journal field `ridge_features` length≥14): present on the live journal but NOT on archive trades.
5. **Calibrator state** (`trained_data/confidence_calibration.json`): only one Platt model fitted (`_global`, coef≈8.99, intercept≈-5.58). Per-regime models absent. Trade history list = 14 entries.
6. **Joint training meta** (`trained_data/models/joint/joint_training_meta.json`): val R²=0.9971, MAE=0.27, n_train=28259, n_val=9315, saved 2026-04-16.

### 1.2 Inspections to run (read-only) before writing code
- Distribution of realized R-multiples per pair across all archives + live journal.
- Win rate by `pair`, by `regime_at_entry`, by `confidence` bucket → identifies whether the leaky score has ANY signal.
- Count of training rows with `ridge_features` recoverable → bounds the supervised label-set size.
- Distribution of `gate.confidence_passed` by pair over last 30 days → baseline for §6 cut-over check.
- Walk the call sites in §3 below and confirm they're still live (`online_retrainer.py:556`, TUI).

### 1.3 Open questions (require operator answers — see §10)
Block Stage 0 → 1 transition until answered.

### 1.4 Stage 0 deliverable
A short `docs/plans/2026-04-30-confidence-label-leak-fix-stage0-report.md` containing:
- Final closed-trade count (live + salvaged archives).
- R-multiple histogram (10 bins) per pair.
- Recoverable-features count.
- Operator answers to §10 questions.
- Explicit GO/NO-GO recommendation.

**Operator approval gate #1** at end of Stage 0.

---

## 2. Label-replacement design

### 2.1 Label form — recommend binary win/loss + auxiliary continuous, NOT raw R
Rationale: R-multiple distribution is bimodal (-1.0 SL hit / +1.8 TP hit) with almost nothing in between (archive `20260323_203742` shows 231 trades at -1, 132 at +1.8, 1 at -1.4, none in (-1,0) or (0,1)). Regression on this is essentially classification with extra steps and degenerate variance.

**Proposed label primary form:** `y_confidence = win_probability ∈ [0,1]`, where the supervisor target is `1 if outcome.trade_won else 0`. Then the existing 0-100 gate scale is recovered as `100 * y_confidence`.

**Alternative continuous form** (use only if operator answers Q10b in favor): `y_confidence = sigmoid(2 * R_multiple)` clipped to [0.05, 0.95], where `R_multiple = (exit - entry) / |sl - entry|` with sign by direction. Winsorize R at [-1.5, +3] before sigmoid.

**Do NOT** use the existing `realized_confidence` from `event_handlers.py:248` — it's contaminated by `ridge_conf`.

### 2.2 Handling unlabeled rows
Two viable approaches; pick at Stage 0:

**(i) Restrict training to labeled rows only.** Pros: clean supervised signal. Cons: ~364 rows total against current 28k → severely undertrained.

**(ii) Semi-supervised / pseudo-label cascade.** Use a meta-labeler-style approach: for unlabeled rows, derive a "would-this-have-won?" pseudo-label by simulating triple-barrier outcome from forward bars. **Recommend (ii)** because triple-barrier with proper embargo is exactly what the meta-labeler already does (`meta_labeling.py:75-80`).

If (ii) is taken, the new label is essentially "would a hypothetical trade entered at this bar with the configured ATR-based SL/TP have hit TP first within N bars?" — a binary triple-barrier label. This is meaningfully different from the current leaky formula and not a function of the input features.

### 2.3 Join key: training row → trade journal lookup
- Row identity in training: `(instrument, df.index)` where `df.index` is bar timestamp.
- Journal trade identity: `(pair, timestamp)` where `timestamp` is signal-time, plus `entry_price`.
- Resolution: normalize pair names; align timestamps by `pd.Timestamp.floor(granularity)` (granularity = `H1` per `ScannerConfig.granularity` at `src/scanner/config.py:427`).
- Storage: emit a side-car `confidence_label_join.parquet` next to joint training data with columns `(instrument, bar_timestamp, has_real_label, real_label, has_pseudo_label, pseudo_label)`.

### 2.4 Class-balance + sampling
With binary label and ~36% win-rate, class imbalance is mild. No SMOTE. Pass `class_weight='balanced'` to LightGBM.

---

## 3. Training pipeline change — file:line touchpoints

### 3.1 Label generator (new module)
- **NEW**: `src/training/labels/realized_confidence_label.py`
  - `compute_realized_confidence_labels(df, instrument, journal, lookahead_bars, sl_atr_mult, tp_atr_mult) -> np.ndarray` — returns float labels in [0,1] aligned to df rows.
  - Triple-barrier on forward bars for pseudo-labels, journal join for real labels, real overrides pseudo.
- **NEW**: `src/training/labels/__init__.py`

### 3.2 Loader rewrite
- **EDIT**: `src/core/modular_data_loaders.py:3335-3426` — replace closed-form formula with call to `compute_realized_confidence_labels(...)`. Keep signature stable for all 5 callers.
- **EDIT**: `src/core/modular_data_loaders.py:3274-3293` — add `journal_path: Optional[str]` and `label_mode: Literal['real', 'pseudo', 'blend']` params with default `'blend'`.
- **EDIT**: `src/core/modular_data_loaders.py:3450-3454` — extend returned dict with `label_metadata`.

### 3.3 Trainer awareness (metadata only)
- **EDIT**: `src/training/trainers/ridge_trainer.py:221-228` — extend `self.metrics` with `label_mode`, `n_real_labels`, `n_pseudo_labels`, `class_balance`, `expected_r2_band` (e.g. `[0.05, 0.30]`). Auto-detect leak: `if r2 > 0.6: logger.error("Suspected leak: R²=%.3f exceeds expected band", r2)`.
- **EDIT**: `src/training/trainers/joint_trainer.py:419-428` — pass through label-mode metadata.
- **EDIT**: `src/training/trainers/joint_trainer.py:794-808` — write new label fields into `joint_training_meta.json`.

### 3.4 Backups before retrain (Stage 1 task)
Implementer creates `.pre_leak_fix.bak` copies under `trained_data/_rollback_2026-04-30/` of:
- `trained_data/models/joint/ridge_confidence.pkl`
- `trained_data/models/joint/joint_training_meta.json`
- `trained_data/confidence_calibration.json`
- All per-pair `trained_data/models/{INSTRUMENT}/ridge_confidence.pkl`

Manifest: `docs/plans/2026-04-30-confidence-label-leak-fix-rollback-manifest.md`.

---

## 4. Calibration refit

### 4.1 Why refit is mandatory
Current Platt (`coef=8.99, intercept=-5.58`) was fit against the 20-95-range leaky score. New score range will be [0,100] (or [0,1]) and the empirical mapping to win-probability will be entirely different.

### 4.2 Refit procedure (Stage 3)
- Collect `(new_confidence, won)` pairs from shadow run for ≥200 closed trades (target 50+ per regime if regime-conditioned Platt is desired).
- Refit using existing `ConfidenceCalibrator.fit()` at `src/risk/confidence_calibration.py:100-132`.
- Bump version to `2`, add `"label_mode"` field. Move old to `.pre_leak_fix.bak`.

### 4.3 Calibration validation gates
- ECE on held-out 20% slice < 0.10.
- Reliability curve: monotonic non-decreasing across 10 confidence buckets (small inversions <2pp acceptable).
- Brier score < 0.25.
- No bucket has |actual_win_rate − predicted_confidence| > 0.20.

If any gate fails, calibration is rejected and we revert. Block Stage 4.

---

## 5. Shadow vs. cut-over strategy — chosen: **B (Shadow) + C (re-tune) at Stage 4**

### 5.1 Why not (A) hard cut
`min_confidence=42.0` was tuned for the 20-95 leaky range. New range will be [0,100] or [0,1]. Until thresholds are re-fit empirically, the gate would either lock all trades or release everything.

### 5.2 Why not (D) bandit-arbitrate
Bandit weighting (`src/recursive_intelligence/ensemble_weighting.py:152-212`) assumes both arms are at-least-coherent. The leaky model is not coherent — its score has no causal link to outcomes. Defer (D) as Phase 2 only after new model proves out.

### 5.3 Strategy B+C in detail
- Stage 2: train new model in joint pipeline. Save as `ridge_confidence.shadow.pkl` next to live `ridge_confidence.pkl`.
- Stage 3: scanner loads BOTH; live decisions still use leaky model; new model's prediction logged via MAML benchmark infrastructure (`src/recursive_intelligence/maml_benchmark.py`). Persist to `trained_data/leak_fix_shadow_log.jsonl`.
- Stage 3 minimum window: 200 closed trades AND 14 calendar days, whichever later. Live volume ~1 trade/day/pair → ~22 days for 200 closed across 9 active pairs.
- Stage 4: use shadow data to determine new `min_confidence` such that gate-rejection rate falls within ±20% of leaky baseline. Apply new threshold + flip consumer in single deploy.

### 5.4 Trade-offs explicit
| Strategy | Risk | Time | Reversible |
|---|---|---|---|
| A (hard cut) | High — distribution shift, possibly all-or-nothing trades | Days | Yes via .bak |
| **B+C (shadow + re-tune) — chosen** | Low | 3-4 weeks | Yes, trivially |
| D (bandit) | Medium — bandit on non-coherent arm | Weeks | Yes |

---

## 6. Validation gates (must all pass before Stage 4 cut-over)

### 6.1 Model-level
- New val R² in `[0.05, 0.30]`. Higher → suspect leak; lower → suspect undertraining. Both block.
- New val MAE on win-prob target < 0.40 (random baseline ~0.49 at 36% win rate).
- New val Brier score < 0.25.
- LightGBM feature importance: no single feature with `gain >= 60%` (concentration check).

### 6.2 Calibration-level
See §4.3.

### 6.3 System-level (shadow comparison)
- Gate-rejection rate (rolling 100-scan window) of new model within ±20% of leaky baseline at proposed threshold.
- WVS distribution shift: 95th percentile of new WVS within ±0.10 of leaky baseline.
- No regime where gate-rejection rate hits 100% or 0% (class-collapse check).
- Position-sizer multiplier distribution: no >2x shift in mean unit count.
- Meta-labeler precision on shadow trades doesn't drop more than 5pp vs baseline.

### 6.4 Decision rule
ALL of §6.1, §6.2, §6.3 must pass. Any failure → block cut-over, return to Stage 2 or escalate.

---

## 7. Rollback path

### 7.1 Backups
Manifest in `trained_data/_rollback_2026-04-30/MANIFEST.json` listing every backed-up artifact + sha256.

### 7.2 Procedure (any stage)
1. Stop scanner (operator action).
2. Copy rollback artifacts back over live paths.
3. Revert source via git revert.
4. Re-launch scanner; verify `joint_training_meta.json:r2_score` reads 0.997 (leaky model back).
5. Operator-visible alert with rollback rationale + delta metrics.

### 7.3 Partial rollbacks
- New model bad but calibrator fine → revert only `.pkl`; keep new calibration.
- Calibrator bad → fall back to `method='none'`; keep new model. Acceptable for short emergency window.

### 7.4 Time-bound auto-revert (optional safeguard)
Add config flag `confidence_model_auto_revert_after_n_losses: int = 10` to `ScannerConfig`. If 10 consecutive losses occur in first 50 trades after cut-over, auto-rollback. Implement as separate small PR after Stage 4.

---

## 8. Test coverage

### 8.1 Unit tests (Stage 1)
- `tests/test_realized_confidence_label.py`:
  - `test_pure_winner_gets_high_label()`
  - `test_pure_loser_gets_low_label()`
  - `test_unlabeled_row_uses_pseudo()`
  - `test_journal_overrides_pseudo()`
  - `test_winsorization_caps_extremes()`
  - `test_join_key_normalization()`
  - **`test_label_invariance_to_feature_perturbation()`** — perturbing input features by ±10% must NOT change label by more than ε. THE leak-prevention test.

### 8.2 Integration tests
- `tests/integration/test_ridge_data_loader_label_join.py`: end-to-end synthetic df + journal → expected real/pseudo split.

### 8.3 Regression tests
- `tests/test_gate_confidence_distribution_regression.py`: replay 1000 production scans through both models; gate-rejection delta within ±20%; WVS KS-test p > 0.05.

### 8.4 Existing tests to update
- `tests/test_phase53_confidence_pipeline.py`, `test_phase58_raw_confidence_recorder.py`, `test_confidence_calibration.py`, `test_calibration_integration.py` — audit assertions hardcoded to 20-95 range; update to new range.

### 8.5 Smoke test before Stage 4
`./buddy --demo` for one full session (single pair, dry run, 50 scans). Verify new model loaded, predictions in expected range, no exceptions.

---

## 9. CLAUDE.md / rules updates

### 9.1 Promote to `.claude/rules/improvement.md`
**`## Label-from-Features Audit Gates (promoted 2026-04-30, from confidence-leak incident)`**
- ALWAYS audit any new supervised label for closed-form derivability from input features — if the label can be written as `f(x_1, ..., x_k)` where `x_i` are also features, it is a leak even if temporal splits are correct.
- ALWAYS include a `test_label_invariance_to_feature_perturbation` for any new label generator.
- ALWAYS log `r2_score`, `expected_r2_band` in training metadata; emit ERROR if `r2 > expected_r2_band[1]` for noisy financial regression heads.
- NEVER allow training and label-generation to read the same indicator column unless explicit comment justifies why label is causally upstream of feature.

### 9.2 Trading rules ledger (`.claude/rules/trading.md`)
After Stage 4: append "Confidence model rebuilt against journal-outcome labels on 2026-04-XX (replaces closed-form leak). Calibrator refit; baseline `min_confidence=42.0` re-tuned to `<value>`."

### 9.3 CLAUDE.md
Optional: add to "What we never do" — "Ship a learned model whose label is a deterministic function of its inputs."

---

## 10. Open questions blocking execution (must be answered before Stage 0 → 1)

1. **Archive salvage**: Six archive files JSON-corrupt; tolerant-truncate parse recovers ~365 trades each. Salvage them (~2000 closed trades total) or treat as lost (only 364 from most-recent parseable archive)? **Recommendation: salvage** — ~30 LOC of regex-truncate. Affects whether semi-supervised is even necessary.
2. **Label form**: binary win/loss vs. winsorized R-multiple vs. MFE-normalized continuous score? **Recommendation: binary win/loss** given bimodal R distribution.
3. **Acceptable shadow window**: live trade volume ~1/day/pair; minimum-200-closed threshold = ~3 weeks. Acceptable, or compress by replaying salvaged archives offline?
4. **Per-pair fine-tunes**: ~24 closed trades per pair below LightGBM fine-tune floor. (a) skip per-pair fine-tunes for confidence (joint only), (b) defer until volume catches up, or (c) per-pair pseudo-labels?
5. **Auto-revert safeguard**: implement §7.4 N-loss auto-revert in this PR or defer to follow-up?
6. **Triple-barrier params for pseudo-labels**: reuse `meta_labeling.py:51-80` defaults, or align with runtime SL/TP gate logic in `gates.py`? They may differ.

---

## 11. Staged rollout — operator approval gates

| Stage | Goal | Output | Gate |
|---|---|---|---|
| **0. Audit** | Establish data inventory, answer §10 questions | `docs/plans/2026-04-30-confidence-label-leak-fix-stage0-report.md` | **Operator GO/NO-GO #1** |
| **1. Build label generator** | New module + unit tests + backups | `src/training/labels/realized_confidence_label.py`, tests, rollback manifest | All §8.1 unit tests pass; CI green |
| **2. Train shadow** | Joint train with new label, save as `.shadow.pkl`; integration test passes | New `.shadow.pkl`, updated meta | **Operator GO/NO-GO #2**: review §6.1 |
| **3. Measure shadow** | Scanner loads both models; log per-scan deltas; refit calibrator after enough trades | `leak_fix_shadow_log.jsonl`, drafted new calibrator | Min 200 closed + 14 days; **Operator GO/NO-GO #3**: review §6.2 + §6.3 |
| **4. Cut over** | Tune `min_confidence`; flip consumer in `gates.py`; promote new calibrator; smoke test | PR merged, scanner restarted | **Operator GO/NO-GO #4** (final): 24h post-deploy review |
| **5. Cleanup** | Remove old leaky code; promote rules; archive `.shadow.pkl` | Cleanup PR | Stage closed |

NO-GO at any gate triggers §7.2 rollback.

---

## 12. Risk register (top 5)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Insufficient labeled data (~364 closed real trades, ~24/pair) — model undertrains | High | High | Stage 0 archive salvage; semi-supervised pseudo-labels via triple-barrier; gate Stage 4 on N_closed ≥ 1000 |
| 2 | Bimodal R-distribution makes regression ill-posed | High | Medium | Switch to binary win/loss label (§2.1); winsorize if continuous chosen |
| 3 | Calibrator refit produces degenerate curve (only 1 Platt model fitted today) | Medium | High | §4.3 ECE/reliability/Brier gates; keep `.pre_leak_fix.bak`; partial rollback (§7.3) |
| 4 | Threshold re-tune lands on a value that locks all trades or releases everything | Medium | Critical | Strategy B measures rejection rate before flip; §6.3 ±20% gate; §7.4 auto-revert |
| 5 | Downstream consumers (meta-labeler, WVS, position sizer, agents) break in subtle ways | Medium | High | §6.3 system-level gates per consumer; 14d minimum shadow window |

---

## 13. Effort estimate

- Stage 0: 2-3 days (data archeology + operator Q&A roundtrip).
- Stage 1: 3-4 days (label generator + tests + backups).
- Stage 2: 3-5 days (training run + per-instrument fine-tune + meta capture).
- Stage 3: **14-21 calendar days minimum** (live shadow window dominated by trade volume, not engineering hours).
- Stage 4: 2-3 days.
- Stage 5: 1-2 days.
- **Total: 25-35 calendar days, 14-18 engineering days.**

---

## 14. Critical files for implementation

- `src/core/modular_data_loaders.py` (label generation 3335-3426; signature 3266-3293)
- `src/training/trainers/ridge_trainer.py` (metrics 221-228; expected-R²-band check)
- `src/training/trainers/joint_trainer.py` (entrypoint 419-428; metadata save 794-808; per-pair fine-tune 760-784)
- `src/risk/confidence_calibration.py` (refit 100-132; application 213-229)
- `src/scanner/gates.py` (live consumer 1253-1295; cut-over flip lives here)
