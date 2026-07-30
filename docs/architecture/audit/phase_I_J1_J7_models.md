# Phase I / J1 / J7 — evidence-based model audit

**Date:** 2026-07-30 · **Auditor:** AI Engineer sub-agent · **Mode:** read-only (no source changed)
**HEAD:** `074758d` (2026-07-21) · **Working tree:** clean except this new `docs/architecture/audit/` dir.

**Method.** Every claim below was re-derived from files read in this session. `file:line` is cited for
each. Integration-greps were run before any "wired"/"not wired" claim. Confidence is tagged per
load-bearing claim. Where I disagree with the roadmap's own QA note, I say so and give the disk evidence.

**Environment disclosure (affects only the test-run section).** The container's Python had **no**
`pytest`, `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `rich`, `pydantic`, or a working
`cryptography`/`cffi`. I `pip install`ed those to execute the gate tests. That is an environment
mutation, not a repo mutation — no file under version control was modified.

**One incidental honesty flag, outside audit scope.** `CLAUDE.md` states `.claude/state.json` is
`halted: true`. On disk it is `halted: false`, `mode: "live"`, `last_updated: 2026-05-05T01:10:34Z`
(`.claude/state.json`). The file is ~12 weeks stale and contradicts the doctrine file. Not part of this
audit's remit, but it is a Hard-NO-adjacent discrepancy an operator should resolve. Confidence HIGH
(direct read).

---

## 0. Headline

| | Count |
|---|---|
| Phase I vertical-slice stages implemented | **13 / 13** (+ dashboard = 14/14) |
| Trained model artifacts on disk for these phases | 2 (`risk_target_model` vol head + drawdown head, one `.pkl`) |
| Of those, with **zero** production consumers | **2 / 2** |
| Gate tests run, `tests/test_risk_target_training_gate.py` | **4 passed** |
| Supporting evidence tests run | **29 passed** |
| J1 active goals with a trainer | 3 / 8 |
| J1 active goals with a live consumer | 1 / 8 |
| J1 RL prerequisites satisfied | **2 / 9** |
| J7 local-only capabilities LIVE | 4 / 7 |
| J7 promotion requirements enforced in code | **8 / 10** |

---

## 1. Phase I — the 13-stage vertical slice

### 1.1 The roadmap's own QA note is STALE — correct it

`AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md:981-1002` (QA note dated 2026-07-12) asserts:

> "the diagram below is not implemented yet — none of its stages are built (`cli/risk_target_training.py`/`RiskTargetTrainer` never touch `src/evidence/` contracts; there is no remote/local split, no `DatasetManifest`/`JobManifest`/`EvidencePackage`, no dashboard surface)."

**Half of that is now false.** Commit `ae685fa` *"feat(evidence): governed evidence foundation +
Phase I risk-target vertical slice"* landed **2026-07-13**, the day after the note was written, and
built `src/evidence/risk_target/` (8 modules, 1875 LOC). The roadmap file was subsequently touched by
`074758d` on 2026-07-21 **without** updating the note. `git log --format='%h %ad %s' --date=short -- src/evidence/`
and `-- docs/architecture/AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md` are the sources. Confidence **HIGH**.

**The half that is still true, and is the real finding:** the sentence
*"`cli/risk_target_training.py`/`RiskTargetTrainer` never touch `src/evidence/`"* remains **literally
correct**. `grep -n "evidence" src/training/trainers/risk_target_trainer.py cli/risk_target_training.py`
returns exactly one hit — a prose comment at `risk_target_trainer.py:180`. **Zero import edges.** The
dependency runs the other way: `src/evidence/risk_target/evaluation.py:38` imports `RiskTargetTrainer`.

So there are **two disjoint paths**, and only one of them produces the artifact that exists on disk:

| | Path A — artifact producer | Path B — evidence producer |
|---|---|---|
| Entry | `cli/risk_target_training.py:70 train_risk_targets` | `src/evidence/risk_target/slice.py:255 run_risk_target_evidence_slice` |
| Trains | `RiskTargetTrainer` directly | `RiskTargetTrainer` via `evaluation.py:196` |
| Writes | `trained_data/risk_targets/models/risk_target_model.{pkl,meta.json}` **(exists on disk)** | signed EvidencePackage into an `EvidenceStore` **(no store on disk)** |
| Gates | CLI regression gate + absolute drawdown bar | signed per-head gates + hash/policy/replay import checks |
| Signing / lineage / dashboard | **none** | full |
| Ends at | incumbent overwrite | QUARANTINED or REJECTED, never champion |

**The governance chain is real but governs nothing that exists.** The only risk-target artifact on
disk was produced by the ungoverned Path A. `find . -maxdepth 3 -name "*evidence*" -type d` returns
only `./src/evidence` and `./.ci/policy/evidence`; `ls trained_data/evidence` →
*No such file or directory*. Confidence **HIGH**.

### 1.2 Stage-by-stage

| # | Stage | Status | Source |
|---|---|---|---|
| 1 | FX daily snapshot | **PARTIAL** | `manifests.py:63 build_fx_daily_dataset_manifest`; real-partition driver `scripts/run_risk_target_evidence_slice.py:90`. No scheduled invocation — `com.buddy.forward_daily.plist` runs `scripts/run_forward_capture_daily.py`, which calls `scripts/check_risk_target_p2_readiness.py` (readiness only), not the slice. |
| 2 | DatasetManifest | **LIVE** | `manifests.py:63-110`; per-partition sha256 + size + row counts. |
| 3 | Risk-target feature snapshot | **LIVE** | `evaluation.py:107 assemble_frame` → `src/training/risk_target_features` (`RISK_TARGET_FEATURE_PIPELINE_VERSION` = `2026-07-08-v2`, matches the on-disk artifact meta). |
| 4 | Signed JobManifest | **LIVE** | `manifests.py:127 build_risk_target_job_manifest` + `slice.py:222` `producer.sign(job, ...)`; Ed25519 (`src/evidence/signing.py`). |
| 5 | Isolated training | **LIVE** | `worker.py:247 run_worker` verifies job + dataset envelopes, re-hashes partitions (`worker.py:94`), then calls the evaluator. Capability profile forbids broker creds, orders, halts, live gate, champion pointer, model writes, network (`local_import.py:176-190`). Enforced by 4 tests incl. a transitive import-closure scan (`tests/test_risk_target_evidence_worker_no_authority.py:127`). |
| 6 | Per-head EvaluationReport | **LIVE** | `worker.py:109 _build_evaluation_report`; per-head metrics, tolerances, gates, holdout, purge/embargo, trial count, effective N, cost. |
| 7 | EvidencePackage | **LIVE** | `worker.py:142 _package_head`; content-addressed, model bytes + all four signed lineage envelopes persisted as immutable members. |
| 8 | Remote CREATED event | **LIVE** | `local_import.py:333 append_created_from_producer`. |
| 9 | Local RECEIVED event | **LIVE** | `local_import.py:301 _Chain.append`, driven from `import_head`. |
| 10 | Hash verification | **LIVE** | `local_import.py:111 _hash_checks` — 5 checks: artifact bytes, report↔package binding, signed-lineage durability, dataset↔job lineage, importer-side partition re-hash. |
| 11 | Policy verification | **LIVE** | `local_import.py:169 _policy_checks` — 4 checks: no forbidden capabilities, safety assertions, evaluation completeness, cost accounting (recomputes the USD amount from wall-seconds × rate). |
| 12 | Local metric replay | **LIVE** | `local_import.py:251 _replay_checks` — independently re-runs the evaluator and compares every metric within its declared tolerance **and** the gate verdict. Replay failure is a verdict, not a crash (`local_import.py:258`). |
| 13 | LocalImportVerdict → QUARANTINED / REJECTED | **LIVE** | `local_import.py:340 import_head`; `slice.py:12` "the slice never promotes a champion." |
| 14 | Dashboard display | **LIVE (code) / dark (data)** | `dashboard.py:17 risk_target_evidence_view` → `dashboard/server/data_sources.py:1364 read_risk_target_evidence` → `dashboard/server/app.py:258 @app.get("/api/risk_target_evidence")`. Reads `EVIDENCE_DIR = trained_data/evidence` (`data_sources.py:26`), **which does not exist**, so the endpoint returns `available: False` today. |

**Does the trainer touch `src/evidence/`?** **No.** Zero imports (§1.1). Confidence **HIGH**.

**Roadmap §12 acceptance tests — all 10 have a real test.** `tests/test_risk_target_evidence_slice.py`
covers: worse head rejected independently (`:143`), good head doesn't rescue failed head (`:156`),
candidate never overwrites incumbent (`:194`), missing fold fails (`:211`), changed partition → hash
failure (`:224`), changed model bytes rejected (`:248`), local replay reproduces (`:270`, and against
*real* training at `:380`), registry outage doesn't stop verification (`:407`). Worker-authority
acceptance is `tests/test_risk_target_evidence_worker_no_authority.py:103-155`. Confidence **HIGH**
(names read + all 29 executed green, §3).

### 1.3 Defect found: duplicate dashboard reader

`dashboard/server/data_sources.py` defines `read_risk_target_evidence` **twice** — at `:1296` and at
`:1364`, with identical bodies. The second shadows the first; the first is dead code. Harmless today
(same behavior) but it is exactly the kind of silent shadowing that hides a future divergent edit.
Confidence **HIGH** (both definitions read).

---

## 2. THE KEY QUESTION — does anything consume risk-target predictions?

### 2.1 The grep that proves it

```
grep -rn "risk_target\|RiskTarget\|forward_volatility\|predicted_forward_vol\|drawdown_stressed" \
     src/scanner/ src/risk/ src/tui/ src/core/ --include=*.py
→ src/core/modular_data_loaders.py:2860:def load_forward_volatility_data(
```

**One hit, in none of the three requested directories, and it is a data *loader* (training-side
input), not a consumer of predictions.** `src/scanner/`, `src/risk/` and `src/tui/` contain **zero**
references. Confidence **HIGH**.

Corroborating: `grep -rn "risk_target_evidence_view\|EvidenceStore(" src/tui/ src/scanner/ src/risk/ cli/`
returns nothing — the evidence cockpit is read only by the FastAPI dashboard, never by the trading stack.

### 2.2 What is actually sitting on disk, unused

`trained_data/risk_targets/models/risk_target_model.pkl` + `.meta.json` exist. Metrics from the meta:

| metric | value |
|---|---|
| `n_train` / `n_val` | 38036 / 9526 |
| `val_qlike` | 0.0389 |
| `val_r2` | 0.363 |
| `val_mae` | 0.0182 |
| `val_auc_drawdown` | 0.5864 |
| `val_brier_drawdown` | 0.2492 |
| `val_brier_drawdown_baseline` | **absent** |
| `drawdown_learnable` | **absent** |
| `risk_target_feature_pipeline_version` | `2026-07-08-v2` |
| feature count | 12 |

**Classification: ISOLATED (trained, gated, zero consumers).**

**Sub-finding — the incumbent predates its own gate.** The two keys `val_brier_drawdown_baseline` and
`drawdown_learnable` are written by `risk_target_trainer.py:255-256` on every train since the QA fix.
Their absence from this artifact's meta proves it was written by the **pre-fix** trainer — i.e. it *is*
the "not-learnable drawdown candidate [that] could be (and was) written and marked PASSED on first
deploy" the roadmap note describes. Confidence **HIGH** (meta.json read directly).

Operational consequence: the next retrain's *relative* gate reads
`_score_existing_metrics()` → `val_brier_drawdown = 0.2492` (`cli/risk_target_training.py:60-67, 118-122`)
as its incumbent. That number's absolute-bar status is **unknowable from disk** because the baseline
was never recorded. The absolute bar (§4) still fires independently, so this cannot produce a bad
ship — but the incumbent comparison is being made against an unaudited number. Confidence **MEDIUM**
(gate logic read; not executed against this artifact).

### 2.3 Exactly what wiring would make the vol head LIVE

**Only the volatility head is a wiring candidate.** The drawdown head failed its pre-registered bar
(OOS AUC 0.625 clears 0.55 but OOS Brier 0.232 loses to the 0.143 base-rate baseline, per
`docs/prereg-risk-target-vol-drawdown-2026-07-08.md` as quoted at roadmap:986-990; the on-disk val
split agrees directionally at AUC 0.586). Wiring a head that lost to a constant predictor into a
sizing or gating decision would be strictly worse than the status quo. **Do not wire the drawdown
head.** This is the one place where "everything live, shadow is not an answer" must yield to the
evidence — the honest answer for that head is *retire or re-target*, not *ship*.

**Recommended wire for forward-vol — risk-decreasing-only size multiplier.**

- **Consumer:** `src/risk/position_sizing.py` → `DynamicPositionSizer.calculate_regime_scaled_position_size`
  (`position_sizing.py:466`). It already accepts `volatility_regime: int` and already emits
  `regime_scale_applied` / `regime_name` on its result (`position_sizing.py:128-129`), so there is a
  natural, already-plumbed field to attribute the adjustment to.
- **Decision it should change:** the aggressive up-scaling branch at `position_sizing.py:509-510`.
  Today a HIGH/EXTREME *contemporaneous* regime + meta-confidence multiplies size by 1.5-1.75×
  (`RegimeScalingConfig`, `position_sizing.py:59-68`). The forward-vol head predicts the *forward*
  20-bar annualized vol — precisely the quantity that scaling is implicitly betting on. The correct
  first wire is a **veto/damp, not a boost**: when
  `predicted_forward_volatility > k × realized_ATR_implied_vol`, clamp `regime_scale_applied` to
  ≤ 1.0. One new config field (`enable_risk_target_vol_damp`, default False) as a `ScannerConfig`
  dataclass field **first**, then the four profile dicts, then the consumer `getattr` — the exact
  three-step order mandated by `.claude/rules/improvement.md` "Live Wiring Verification Gates".
- **Multiplier discipline:** cap in `[0.5, 1.0]`, mirroring `RiskCalibrationLearner`'s existing
  risk-decreasing-only contract (`risk_calibration_learner.py:21-23`). This makes the wire
  monotonically risk-reducing, which is the only class of change that can be shipped under `halted`
  doctrine without an unhalt decision.
- **Prerequisite that does not exist yet:** an inference adapter. Nothing loads
  `risk_target_model.pkl` at runtime — `RiskTargetTrainer.load()` (`risk_target_trainer.py:299`) has
  no caller outside tests. A `src/risk/risk_target_inference.py` module is needed to load the artifact,
  assert `risk_target_feature_pipeline_version` matches the runtime constant (refuse on mismatch, per
  the Train↔Inference Contract Gates), compute the 12 features at inference, and refuse (return None)
  rather than zero-fill on any missing column.

---

## 3. Real gate-test output

```
$ python -m pytest tests/test_risk_target_training_gate.py -v --tb=short
tests/test_risk_target_training_gate.py::test_first_deploy_always_writes PASSED                                    [ 25%]
tests/test_risk_target_training_gate.py::test_regressing_candidate_is_refused_and_incumbent_untouched PASSED       [ 50%]
tests/test_risk_target_training_gate.py::test_first_deploy_refuses_drawdown_head_that_fails_absolute_prereg_bar PASSED [ 75%]
tests/test_risk_target_training_gate.py::test_improving_candidate_passes_and_overwrites PASSED                     [100%]
======================= 4 passed, 12 warnings in 27.27s ========================
```

The 12 warnings are all one LightGBM deprecation (`eval_set` → `eval_X`/`eval_y`,
`lightgbm/sklearn.py:1106`) raised by `risk_target_trainer.py:213` and `:229`. Cosmetic today; it will
become a breakage on the next LightGBM major. Worth a scheduled fix, not urgent.

Supporting run (evidence chain):

```
$ python -m pytest tests/test_risk_target_evidence_slice.py \
    tests/test_risk_target_evidence_worker_no_authority.py \
    tests/test_risk_target_evidence_dashboard.py \
    tests/test_risk_target_no_hotpath_coupling.py -q
29 passed, 12 warnings in 9.01s
```

Confidence **HIGH** — both runs executed in this session against the working tree.

---

## 4. Is the drawdown learnability gate really enforced?

**Yes — enforced in code, in both paths, with a passing test.** Confidence **HIGH**.

**Definition + computation (trainer):**
- `src/training/trainers/risk_target_trainer.py:100` — `DRAWDOWN_LEARNABLE_MIN_AUC = 0.55`
- `:103 brier_baseline(y_true)` — returns `p(1-p)` at the observed base rate
- `:243` — `val_brier_drawdown_baseline = brier_baseline(y_val_dd)`
- `:244-247` — `drawdown_learnable = bool(val_auc_drawdown >= DRAWDOWN_LEARNABLE_MIN_AUC and val_brier_drawdown < val_brier_drawdown_baseline)` — **both conjuncts, exactly the pre-registered formula**
- `:255-256` — both fields persisted into the artifact meta

**Enforcement at the ship boundary (CLI gate):**
- `cli/risk_target_training.py:134-135` — `if drawdown_gate["verdict"] == CLI_GATE_PASSED and not metrics.get("drawdown_learnable", False): drawdown_gate["verdict"] = CLI_GATE_REFUSED`
- `:136-143` — records a machine-readable reason string with both observed values and both ok-flags
- `:146-149` — `both_pass` requires *both* heads PASSED
- `:158-160` — `trainer.save(...)` is inside `if both_pass:` — **the artifact is not written otherwise**

**Independent enforcement in the evidence lane** (this is the part that makes it a real rail rather
than a single choke-point): `src/evidence/risk_target/evaluation.py:254-257` emits two signed gates,
`oos_auc_ge_bar` (threshold `DRAWDOWN_LEARNABLE_MIN_AUC`, imported at `:39` from the same constant) and
`oos_brier_beats_baseline`. Because the importer's `_replay_checks` re-derives the gate verdict
(`local_import.py:281-284`), a producer cannot narrate around it.

**Degenerate-input handling is correct.** `_safe_auc` (`risk_target_trainer.py:117`) returns 0.5 on a
one-class holdout, which is `< 0.55`, so the AUC conjunct refuses before `brier_baseline`'s
degenerate `0.0` return could be reached — the docstring at `:107-113` states this and the code
matches.

**Test:** `test_first_deploy_refuses_drawdown_head_that_fails_absolute_prereg_bar` — **PASSED** (§3).

**Residual weakness (MEDIUM).** `metrics.get("drawdown_learnable", False)` fails *closed* on a missing
key — correct. But the *incumbent* side (`_score_existing_metrics`, `cli/risk_target_training.py:56`)
never checks whether the incumbent itself ever cleared the bar. The on-disk incumbent has no
`drawdown_learnable` key (§2.2). Since the absolute bar is applied to the candidate unconditionally
this cannot ship a bad head, but "the incumbent is trustworthy" is an unverified assumption in the
regression comparison.

---

## 5. Phase J1 — the 8 active FX goals

Per `CLAUDE.md`, price-only M15 **direction** is a falsified target (~52% val, >10% gap) and is out of
scope here. This section audits only the risk / execution / cost targets.

| # | Goal | Trainer? | Live consumer? | Class |
|---|---|---|---|---|
| 1 | Trend-risk estimation | **No model.** `src/equity/trend_risk_gates.py` is rule-based (`bucket_cap_gate`, `bias_detector`, `currency_legs`) and belongs to the **equity/trend** lane, not FX. | Yes, but for the trend lane (`src/equity/oanda_trend.py:40`), not the FX scanner. | **ABSENT (for FX)** |
| 2 | Volatility forecasting | **Yes ×2.** `risk_target_trainer.py` vol head (LightGBM, log-space, QLIKE 0.0389 / R² 0.363); `tcn_volatility_trainer.py` (TCN 4-class regime). | Risk-target head: **none** (§2.1). TCN regime: **yes** — `src/scanner/gates.py:314, 405, 607, 610` (`require_tcn` hard-fails without it). | Risk-target vol: **ISOLATED**. TCN regime: **LIVE**. |
| 3 | Drawdown-state forecasting | **Yes**, `risk_target_trainer.py` drawdown head. **Fails its own pre-registered bar.** | None. | **ISOLATED — and should stay unwired** |
| 4 | Regime-conditioned sizing | No learned sizing model; `RegimeScalingConfig` is hand-set constants (`position_sizing.py:47-68`). PPO `position_sizer.py` exists but no fresh artifact. | Rule path is **LIVE** (`position_sizing.py:466 calculate_regime_scaled_position_size`), fed by the TCN regime. | **PARTIAL (heuristic live, no learned model)** |
| 5 | Fill-quality modeling | No trainer. Realized `slippage_pips` **is** captured — 18/18 journal records carry it. | Not modeled; only logged. | **ABSENT** |
| 6 | Spread / slippage forecasting | **Empirical estimator, not a trained model:** `scripts/build_execution_cost_model.py` → `src/data/execution_cost_model.py` → `trained_data/cost_model/execution_cost_estimates.json` (`advisory_only: true`, per-pair `confidence` as low as **0.08** on `sample_count_fills: 4`). A governed evidence lane exists (`src/evidence/execution_cost/`, head `realized_execution_cost_bps`). | Consumed by **equity/hedge** only: `src/equity/trend_risk_gates.py:50,74,343`, `src/equity/oanda_trend.py:67`, `src/hedge/hedge_candidates.py:59`, `src/hedge/hedged_shadow_lane.py:47`. **Zero** consumers in `src/scanner/` — the FX execution path still uses live spread only. | **PARTIAL (live in equity/hedge, ISOLATED for FX)** |
| 7 | Currency-exposure analysis | Analytic, not learned: `src/hedge/portfolio_exposure.py` (reuses `trend_risk_gates.currency_legs`), `src/hedge/exposure_history.py`, `src/hedge/exposure_tags.py`. | `grep "portfolio_exposure\|PortfolioExposure" src/scanner/ src/risk/ src/tui/` → **zero hits**. FX-side double-exposure control is the separate pairwise correlation filter (`src/scanner/execution.py:1197, 1235`), not currency-leg netting. | **PARTIAL (hedge lane only; ABSENT for FX)** |
| 8 | Execution / abstention quality | No trainer. Abstention itself is implemented and correct (`direction=None`, `src/scanner/gates.py`, `src/scanner/agents/_team.py`). Rejected setups are logged to `virtual_trades.jsonl`. | Abstention is **LIVE**; abstention *quality* is never scored — no consumer reads `virtual_trades.jsonl` back into any model. | **PARTIAL** |

**J1 summary: 3/8 goals have a trainer; 1/8 (TCN volatility regime) has a live FX consumer.**
Confidence **HIGH** on the greps; **MEDIUM** on goal↔module mapping, since the roadmap names goals in
prose and I matched them to modules by capability.

---

## 6. Phase J1 — the 9 RL prerequisites, against real files

Real counts, measured this session:

- `trained_data/trade_journal_rl.json` — **18 records**, 85 KB, timestamps
  **2026-04-03 → 2026-05-05** (nothing in ~12 weeks)
- `trained_data/virtual_trades.jsonl` — **4427 records**, 954 KB, **2026-03-31 → 2026-05-05**
- `trained_data/rl_replay_buffer.jsonl` 11 KB · `trained_data/online_rl_updates.jsonl` 1.1 KB

| # | Prerequisite | Measured | Class |
|---|---|---|---|
| 1 | Training states reflect the current trend lane | Journal `lane` split exists in the calibration whitelist (`lane_scanner` / `lane_trend`, `risk_calibration_learner.py:51-52`), but `is_calibration_scoreable` **excludes trend-lane entries entirely** (`offline_learning_cycle.py:293` comment; `risk_calibration_learner.py:108-117`). Journal is 100% scanner-lane. | **ABSENT** |
| 2 | Entry context is complete | 18/18 have `agents`, `gates`, `gate_details`, `regime`, `spread_pips`, `slippage_pips`. **But:** `agent_reasons` carries **5 of 15** agents on 17 records and **0** on one — only `trend, mean_reversion, risk_sentinel, uncertainty` (17× each), `execution_quality` (10×), `volatility` (7×). The other 9 agents (momentum, news_risk, multi_timeframe, pair_performance, session_timing, support_resistance, order_flow, trader_readiness, devil_advocate) **never appear**. `gates` has 3 gate booleans + a summary; `gate_details` has only `scores`/`core_score`/`final_score`. `ridge_features` is **empty on 18/18**. | **PARTIAL — ~33% agent coverage, 0% feature coverage** |
| 3 | Outcome labels are reliable | **16/18 (88.9%)** have a non-null `outcome` with `realized_pl`, `trade_won`, `exit_reason`, `mae_pips`, `mfe_pips`. Structurally good. `rl_weights_applied` — the dedicated done-guard mandated by `.claude/rules/improvement.md` — is present on **0/18**, so no trade in this journal has been credited to agent-weight learning. | **PARTIAL** |
| 4 | Rejected setups recorded with full features | **4427/4427 have `gate_failures` and `raw_confidence`. 0/4427 (0.0%) have non-empty `features`. 0/4427 (0.0%) have non-empty `agent_scores`.** Both keys are present on every record and empty on every record. | **ABSENT — the single hardest blocker** |
| 5 | Sample size adequate | 18 closed trades. The learning loop's own floor is `MIN_HOLDOUT + 1 = 16` just to *attempt* a promotion (`offline_learning_cycle.py:83`). For RL this is ~3 orders of magnitude short. | **ABSENT** |
| 6 | Off-policy evaluation available | `grep -rn "off_policy\|off-policy\|doubly.robust\|importance_sampling\|weighted_importance\|snips\|\bOPE\b" src/ --include=*.py` → **zero hits**. | **ABSENT** |
| 7 | Deterministic baseline beats the RL policy | `grep -rn "deterministic_baseline\|baseline_policy" src/` → **zero hits**. No comparison harness. | **ABSENT** |
| 8 | Risk constraints encoded structurally | **Yes.** `src/training/rl/offline_rl_trainer.py:13-17` states the runtime safety filter as non-negotiable: any DT prediction violating `.claude/rules/trading.md` (R:R<1.2, trend-veto, MR-veto, staleness-block) is *discarded*; the DT may refuse but never override. The gate stack's verdict applies. | **LIVE (as a design constraint)** |
| 9 | RL artifacts use the same evidence package + shadow process | `src/evidence/` has lanes for risk_target, crypto_carry, crypto_momentum, equity_research, execution_cost, hedge_eval, track_b — **no RL lane**. `offline_rl_trainer.py:3-5` self-declares "contract stubs only… methods raise `NotImplementedError`". | **ABSENT** |

**Score: 2/9 satisfied (#3 partially, #8 fully). RL correctly remains research-only.** Prereq #4 is
the load-bearing one: 4427 rejected setups were logged with the schema in place and the payload
empty. That is a producer bug — the fields are written as `{}` — and it means 12 weeks of rejection
data is unusable for counterfactual/off-policy work. Confidence **HIGH** (whole-file scan of all 4427
lines, not a sample).

---

## 7. Phase J7 — learning, calibration, agent weights

### 7.1 The seven local-only capabilities

| Capability | Exists | Runs headless | Class | Source |
|---|---|---|---|---|
| Agent-weight synchronization | Yes | **Yes** — `scripts/offline_learning_cycle.py:242 _run_rl_weight_sync` calls `ExecutionManager.apply_pending_rl_weight_updates()`. Before 2026-07-06 its only caller was the TUI (`embedded_scanner.py`), a dormant process — the docstring at `:17-26` documents that dead-write asymmetry and its fix. | **LIVE** | `offline_learning_cycle.py:242-274` |
| Confidence calibration | Yes | Yes — in-process on the scan loop | **LIVE** | Constructed `src/scanner/engine.py:1075-1077`, applied per-signal `engine.py:4239-4251` incl. `should_refit()` → `fit_from_journal()`. State `trained_data/confidence_calibration.json` is real: Platt+isotonic, n=690 (683 archive + 7 live), `prior_brier` 0.2140, `prior_ece` 0.0166, `leak_fix_version` 2026-04-30. |
| Risk calibration | Yes | Yes — Saturday 06/12/18 UTC (`com.buddy.learning_loop.plist:31-33`), `RunAtLoad=false` | **ISOLATED** | `src/training/incremental/risk_calibration_learner.py`. Its own docstring `:13-23`: *"`size_multiplier` … is currently read by NO live position-sizing path — `DynamicPositionSizer` does not consume `risk_calibration_state.json` (grep-confirmed)."* I re-ran that grep: confirmed, zero hits in `src/risk/`. **And `trained_data/models/risk_calibration_state.json` does not exist** — no candidate has ever been promoted. |
| Gate-model retraining | Yes | Partially | **PARTIAL** | Artifacts `trained_data/models/gate_rl_config.pkl` + `gate_rl_scaler.pkl` exist; loaded by `src/core/rl_inference_integration.py:63-68` and applied at `:118, 171, 189 get_adjusted_thresholds`. Env `src/rl/gate_threshold_env.py:91, 665, 881`. No scheduled retrain job. |
| Drift detection | Yes | Yes | **LIVE** | `src/scanner/engine.py:1096-1097` constructs `DriftMonitor`; `engine.py:5888 run_drift_check` on a periodic cadence, CRITICAL/mild logged at `:5895, :5902`, non-blocking catch at `:5908`. Plus `drift_proxy_guard.py`, `feature_health.py`, `drift_projector.py`. |
| Alert routing | Yes | Yes | **LIVE** | `engine.py:1188-1189` constructs `AlertManager`; `engine.py:5522, 5537 check_all(...)`. Second caller `automation/continuous.py:2236-2238`. State `.claude/alert_state.json` present. |
| Retraining requests | Yes | Yes | **LIVE** | Producer: self-heal writes markers into `trained_data/retrain_requests/`. Consumer: `offline_learning_cycle.py:110 drain_retrain_requests` → `_processed/` via `os.replace` at `:124`, audit trail preserved, never deleted. `RetrainTrigger`/`DriftRemediator` pair at `automation/drift_remediator.py:75, 134, 166`. |

**4/7 LIVE, 1 PARTIAL, 1 ISOLATED, plus one caveat:** `trained_data/models/agent_weights.json` — the
file `AGENTS.md` names as the persistence target for learned weights — **does not exist**. Only
`agent_weights.json.lock` and `agent_weights.json.preCI_2026_05_13` are present in
`trained_data/models/`. So agent-weight sync *runs* but has never written a live weights file in this
tree; the team falls back to `_BASE_WEIGHTS`. Confidence **HIGH** (`ls` read directly).

### 7.2 The ten promotion requirements

Assessed against `scripts/offline_learning_cycle.py`, the only code path in the repo that implements a
J7-style local promotion gate.

| # | Requirement | Enforced | Source |
|---|---|---|---|
| 1 | Chronological holdout | **YES** | `:174-189 _new_entries_since_cursor` sorts by the `(timestamp, trade_id)` total order; `:328 holdout = new_entries[-MIN_HOLDOUT:]` takes the chronologically-latest slice; `:329 train_new = new_entries[:-MIN_HOLDOUT]`. No leakage. |
| 2 | Minimum class balance | **YES** | `:94 MIN_MINORITY_HOLDOUT = 2`; `:337-348` refuses with `decision="insufficient_holdout_signal"` if `min(wins, losses) < 2`. The comment at `:85-93` records the adversarial finding that motivated it (1W/17L let an always-lose predictor beat baseline). |
| 3 | Minimum sample size | **YES** | `:83 MIN_HOLDOUT = 15`; needs `MIN_HOLDOUT + 1 = 16` entries to attempt promotion (`:322-326`). |
| 4 | Incumbent comparison | **YES** | `:350-357` loads incumbent state, deep-copies to candidate, fits candidate on `train_new` only, scores **both** on the same holdout. |
| 5 | Calibration metrics | **YES** | Brier score (`risk_calibration_learner.py:207`), applied `:356-357`, margin `:84 PROMOTION_MARGIN = 0.005` enforced `:369`. |
| 6 | Risk-increasing prohibition | **YES, belt-and-braces** | `:362-364` computes `max_multiplier` over the holdout; `:370 and max_multiplier <= 1.0`. Redundant with the learner's own `[0.5, 1.0]` cap (`risk_calibration_learner.py:22`). Deliberately doubled. |
| 7 | Atomic candidate write | **YES** | `:212-216 _write_state` — `tmp.write_text(...)` then `os.replace(tmp, STATE_PATH)`. Same pattern at `:192-200` (cursor) and `:225-229` (brain status). |
| 8 | Full local event trail | **YES** | `:219-223 _append_history` → `trained_data/learning_loop/history.jsonl`; `:225-240 _write_brain_status` → `.claude/brain/learning_loop_status.json` + append to `.claude/brain/feed.jsonl`. Every cycle logs its decision, both Brier scores, `max_size_multiplier`, markers drained. |
| 9 | **Rollback to prior incumbent** | **NO** | `_write_state` (`:212-216`) overwrites `STATE_PATH` with **no** prior copy retained. `grep -n "prev\|backup\|rollback\|\.bak" scripts/offline_learning_cycle.py` → **zero hits**. Once a candidate is promoted, the previous incumbent is unrecoverable from this path. |
| 10 | No in-place silent overwrite | **PARTIAL** | The write is atomic and always logged (#7, #8), so it is not *silent*. But with no prior-version retention (#9) it *is* in-place: there is no versioned candidate store and no champion pointer, unlike the evidence lane's design. |

**Score: 8/10 fully enforced, 1 partial, 1 absent.** Confidence **HIGH** — every line cited was read.

**Live proof the gate actually bites.** `trained_data/learning_loop/history.jsonl` has exactly 2 lines:

- `2026-07-04T15:20:56Z` — `decision: "insufficient_holdout_signal"`, `holdout_wins: 1`,
  `holdout_losses: 14`, `new_outcomes: 18`, `markers_drained: 0`
- `2026-07-07T02:01:35Z` — `decision: "no_new_data"`, `markers_drained: 9`,
  `rl_weight_sync: {applied: 0, weights_updated: false, detail: "no pending backfilled outcomes"}`

The class-balance guard (#2) fired on the first real cycle and refused. That is the gate working as
designed — and it is also why `risk_calibration_state.json` does not exist. The loop is correct and
**data-starved**, which is the same root cause as J1 prereq #5.

---

## 8. Classification roll-up

**ABSENT (7):** RL evidence lane · off-policy evaluation · deterministic RL baseline · fill-quality
model · FX trend-risk model · promotion rollback (J7 #9) · scheduled evidence-slice run.

**ISOLATED — trained/computed, zero consumers (3):**
1. **`trained_data/risk_targets/models/risk_target_model.pkl` — forward-vol head.** Genuinely
   learnable (OOS QLIKE 0.0505 vs 0.0712 naive across 19/19 pairs, per the pre-reg doc; val QLIKE
   0.0389 / R² 0.363 on disk). Nothing reads it. **Highest-value gap in this audit.**
2. **`risk_target_model.pkl` — drawdown head.** Trained, gated, and honestly failed its bar. Zero
   consumers, and that is the *correct* state.
3. **`RiskCalibrationLearner.size_multiplier`.** Gate machinery is 8/10 complete, self-documented as
   unwired, and has never produced a promoted state file.

**PARTIAL (7):** FX daily snapshot stage (built, unscheduled) · regime-conditioned sizing (heuristic
live, no learned model) · spread/slippage cost model (live in equity+hedge, absent from FX scanner) ·
currency-exposure analysis (hedge lane only) · execution/abstention quality (abstains, never scored) ·
gate-model retraining (loaded, no retrain schedule) · J7 requirement #10.

**LIVE (8):** stages 2-13 of the evidence chain as code · TCN volatility regime → gates ·
confidence calibration → engine · drift detection · alert routing · retraining-request drain ·
agent-weight sync (headless) · RL structural risk constraints.

---

## 9. GAP REGISTER

Ordered by dependency. Effort: **S** ≤ 1 day · **M** 2-5 days · **L** > 1 week.
Priority follows the operator's rule — *a real trained model with zero consumers outranks new
research* — with one explicit exception (G3), argued below.

| ID | Gap | Work to make it LIVE | Effort | Depends on |
|---|---|---|---|---|
| **G1** | **Forward-vol head has no inference adapter.** `RiskTargetTrainer.load()` (`risk_target_trainer.py:299`) has zero callers outside tests. | New `src/risk/risk_target_inference.py`: load `trained_data/risk_targets/models/risk_target_model.pkl`; assert `risk_target_feature_pipeline_version == RISK_TARGET_FEATURE_PIPELINE_VERSION` and **refuse** on mismatch; compute the 12 features at inference from the same `src/training/risk_target_features` code path; **refuse (return None) rather than zero-fill** any missing column. Add a real-disk canary test asserting refusal on both a version bump and a missing feature. This is the single unblocking prerequisite for every downstream wire. | **M** | — |
| **G2** | **Forward-vol prediction reaches no sizing decision.** | Wire G1 into `DynamicPositionSizer.calculate_regime_scaled_position_size` (`position_sizing.py:466`) as a **risk-decreasing-only damp on the 1.5-1.75× up-scale branch** (`:509-510`), clamped to `[0.5, 1.0]`, attributed via the existing `regime_scale_applied` / `regime_name` result fields (`:128-129`). Add `enable_risk_target_vol_damp` as a `ScannerConfig` dataclass field **first**, then all four profile dicts, then the consumer `getattr` — the exact three-step order in `.claude/rules/improvement.md`. Verify with a live-scan smoke test that the module's log line appears in `logs/buddy_debug.log`, not just in unit tests. | **M** | G1 |
| **G3** | **`virtual_trades.jsonl`: 0/4427 records carry `features` or `agent_scores`.** Schema present, payload empty on every line. 12 weeks of rejection data is unusable. | Find the writer (grep `virtual_trades` in `src/scanner/`), fix the two empty-dict payloads, add a **write-side assertion** that both dicts are non-empty before append. Backfill is impossible — the features are gone — so **every day this stays broken is a day of permanently lost data.** That is why this outranks the remaining model-wiring gaps despite not itself being an unused model: it is the only item in this register that is actively destroying an asset. | **S** | — |
| **G4** | **`RiskCalibrationLearner` unwired + no rollback (J7 #9).** | (a) Add prior-incumbent retention to `_write_state` (`offline_learning_cycle.py:212-216`): copy the current `risk_calibration_state.json` to `.prev` before `os.replace`, plus a `--rollback` flag. Closes J7 #9 and upgrades #10 to full. (b) Once a state file finally exists, wire `size_multiplier` into the same `position_sizing.py:466` seam as G2 behind its own flag. Note (b) is operator-gated per `CLAUDE.md` (per-trade hot path) and is **data-blocked** regardless — see G6. | **S** (a) / **M** (b) | (b) needs G6 |
| **G5** | **Evidence chain is fully built but has never run against real data.** `trained_data/evidence/` does not exist; `/api/risk_target_evidence` (`app.py:258`) returns `available: False`. | Schedule `scripts/run_risk_target_evidence_slice.py` — add it to `scripts/run_forward_capture_daily.py`'s step list (it already runs `check_risk_target_p2_readiness.py`) or as a new launchd plist beside `com.buddy.forward_daily.plist`. Then re-run the vol-head training **through Path B**, so the artifact that G1/G2 consume carries a signed lineage instead of Path A's ungoverned write. Also: dedupe `read_risk_target_evidence` (`data_sources.py:1296` vs `:1364`). | **M** | G1 (to make Path B's output the consumable artifact) |
| **G6** | **Learning/RL loops are data-starved.** 18 closed trades (16 resolved), none since 2026-05-05; `agent_weights.json` absent; `risk_calibration_state.json` absent; `agent_reasons` covers 5/15 agents; `ridge_features` empty on 18/18; `rl_weights_applied` on 0/18. | Not a code gap — a **volume** gap, and the honest gate for all of it. Two code-side sub-items that *are* actionable now: (a) extend journal `agent_reasons` to persist all 15 agent verdicts, not 4-6, and populate `ridge_features`; (b) add the `rl_weights_applied` dedicated done-guard field to the writer, per `.claude/rules/improvement.md`'s four-times-confirmed dead-write pattern. Both are prerequisites for the journal ever being usable — do them before the data arrives, not after. | **M** | G3 (same writer discipline) |
| **G7** | **FX execution path ignores the cost model.** `trained_data/cost_model/execution_cost_estimates.json` is consumed by equity + hedge (`trend_risk_gates.py:50`, `oanda_trend.py:67`, `hedge_candidates.py:59`) but has **zero** consumers in `src/scanner/`. | Wire `estimate_execution_cost` into the `execution_quality` agent (`src/scanner/agents/_team.py`) as an additional input alongside live spread. **Caveat, and it is real:** the file is `advisory_only: true` with per-pair `confidence` as low as 0.08 on `sample_count_fills: 4`. Ship this only behind a minimum-confidence threshold, or it will inject noise into a hard gate. | **M** | G3 (fill data feeds the estimator) |
| **G8** | **RL prereqs 6, 7, 9 absent** (off-policy evaluation, deterministic baseline, RL evidence lane). | Build only after G3+G6 produce data worth evaluating. Doing OPE against 18 trades would produce a number, and the number would be meaningless. **Correctly deferred.** | **L** | G3, G6 |

**Explicitly NOT recommended.** Wiring the drawdown head into any sizing or gating decision. It loses
to a constant base-rate predictor on Brier in both the frozen OOS split (0.232 vs 0.143, pre-reg doc)
and directionally on the on-disk val split (AUC 0.586). The operator's "everything live, shadow is not
an answer" is the right default, and it is exactly why the *vol* head should ship (G1→G2) — but the
drawdown head has no live state to go to. Its honest dispositions are **retire** or **re-target**
(different horizon, different label definition), each requiring a fresh pre-registration. Shipping it
would be shipping noise into a risk decision. Confidence **HIGH**.

---

## 10. Confidence ledger

| Claim | Level | Basis |
|---|---|---|
| Roadmap QA note at `:981-1002` is stale on the "no stages built" half | **HIGH** | `git log` dates + 8 modules read + 29 tests green |
| Trainer/CLI never import `src/evidence/` — dependency runs the other way | **HIGH** | Direct grep, one prose-comment hit; reverse edge at `evaluation.py:38` |
| Zero production consumers of risk-target predictions in `src/scanner/`, `src/risk/`, `src/tui/` | **HIGH** | Grep of 5 identifier variants across 4 dirs → 1 hit, a loader in `src/core/` |
| Drawdown learnability bar enforced in code, both paths | **HIGH** | 8 cited lines read + `test_..._absolute_prereg_bar` PASSED |
| 4 gate tests + 29 evidence tests pass | **HIGH** | Executed this session; verbatim output in §3 |
| On-disk incumbent predates the QA fix | **HIGH** | `drawdown_learnable` / `val_brier_drawdown_baseline` absent from its meta.json |
| 0/4427 virtual trades carry features or agent_scores | **HIGH** | Whole-file scan, all 4427 lines, not a sample |
| J7 promotion gate: 8/10 enforced, rollback absent | **HIGH** | Every cited line read; `grep prev\|backup\|rollback` → zero |
| J1 goal → module mapping | **MEDIUM** | Roadmap names goals in prose; I matched to modules by capability. A different reading could reassign goals 1 and 7. |
| Incumbent-side trust gap in the regression comparison | **MEDIUM** | Gate logic read, not executed against this artifact |
| "Effort S/M/L" estimates | **LOW** | Judgment from code shape, not from attempting the work |

**Assumption that, if false, invalidates the G1→G2 recommendation:** that the forward-vol head's
pre-registered OOS result (QLIKE 0.0505 vs 0.0712 naive, 19/19 pairs) holds on *current* data. I read
that number from the roadmap's citation of `docs/prereg-risk-target-vol-drawdown-2026-07-08.md`, not
from a re-run. The on-disk artifact's own val metrics (QLIKE 0.0389, R² 0.363) are consistent with a
working head, but they are val-split, not the frozen OOS slice. **Cheapest test that settles it:**
run `scripts/run_risk_target_evidence_slice.py` on current partitions (G5) and read the signed
`oos_qlike_beats_naive` gate verdict — that is one command against machinery already built and
already green in test.
