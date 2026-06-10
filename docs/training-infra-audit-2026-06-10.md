# Training Infrastructure Audit — 2026-06-10

Method: 4 parallel specialist auditors (data/leakage, trainer/metrics, gate/quarantine/contract,
validation/promotion) each verified against code; every CRITICAL/HIGH below independently
re-verified by the lead via grep/read (file:line quoted). Confidence HIGH unless noted.

## Headline

The training infra has three classes of problem, in priority order:
1. **The gates that decide what ships are partly blind or no-op** — bad models can ship green.
2. **Advertised-vs-real gaps** — walk-forward validation is dead code; train↔inference contract
   enforcement covers 1 of 6 heads.
3. **Robustness** — non-atomic saves under a once-only disk check; embargo not wired into 4 loaders.

The single highest-value fix is the balanced-accuracy / class-collapse guard on the promote gate
(C-V2): it is the literal mechanism that let the all-SHORT predictor pass as "70% holdout."

## CRITICAL (verified)

| ID | Finding | Evidence | Fix |
|---|---|---|---|
| **G1** | `train_tcn` ships with a fabricated `gap=0`. No trainer defines `get_metrics`; `train_tcn` (`train_single_model_m1.py:501`) calls `trainer.get_metrics() if hasattr else {}` → `{}` → train_acc=val_acc=gap=0. `"tcn"` IS in `GAP_CHECKED_MODELS:755`, so its 10% quarantine + 6% PASS gate are permanent no-ops. The `_read_trainer_metrics` fix (`:67`) was applied to transformer (`:461`) + histgb (`:609`) only. Live regression of the 2026-05-13 bug. | grep: no `def get_metrics` anywhere; lines 501/526/549/572/649 still use the broken reader | Route lines 501/526/549/572/649 through `_read_trainer_metrics`; assert non-empty metrics for any `GAP_CHECKED` model and fail loud. |
| **V2** | Promote gate is blind to class collapse. `balanced_accuracy` appears **nowhere** in `promotion_policy.py` or `scheduled_retrain.py`; the gate uses raw `accuracy` only (`promotion_policy.py:124`, holdout `scheduled_retrain.py:526 correct/total`). An all-SHORT predictor scores ~0.70 on a SHORT-heavy slice (balanced≈0.50) and clears the 0.52 floor. This is the May-2026 all-SHORT failure mode, unguarded. | grep `balanced_accuracy` in both files = empty | Compute balanced_accuracy (mean of per-class recall) in `validate_holdout_accuracy`; hard-gate `balanced_accuracy >= 0.52` AND class-collapse tripwire (reject if either predicted class <15%). |
| **M1** | Walk-forward validation is dead code. `train_single_model_m1` + `scheduled_retrain` have **0** walkforward refs; `train_with_walkforward_validation` has **0** callers. Every shipped model is gated on a single 70/20/10 temporal split. CLAUDE.md/ML-stack advertises "walk-forward + purged k-fold + embargo" — an f070d39-class advertised-but-dead gap. | grep walkforward both ship scripts = 0 | Operator decision: wire walk-forward into the ship path (report cv_mean/cv_variance, quarantine on high variance) OR delete the modules and correct the claim. If wiring, fix the embargo bug (Vp) first. |

## HIGH (verified)

| ID | Finding | Evidence | Fix |
|---|---|---|---|
| **C1** | Train↔inference contract enforced for transformer only. `feature_pipeline_version` saved by transformer (5 refs) and **0** other heads (histgb/lgbm/ridge/tcn-vol/tcn). gates refuses on version mismatch only for the transformer (`gates.py:1284-1297`); the other 5 heads load + predict with no version guard → silent OOD after any `compute_normalized_features` change. | grep feature_pipeline_version per trainer | Add the version key to every trainer save dict; add load-time mismatch refusal in each `gates._load_*`. |
| **C2** | HistGB inference matrix built from the WRONG head's feature_names. `modular_inference.py:4349 histgb_features = self._extract_tcn_features(df)` — HistGB (a hybrid voter that can override the transformer) is fed a TCN-ordered matrix, not its own saved `feature_names`. Divergence → silent misaligned-column predictions. | line 4349 quote | Build the matrix from `self.histgb.feature_names`; refuse on missing column. |
| **R1** | Non-atomic model saves + once-only preflight = corruption window. No trainer uses `os.replace/os.rename`; all write the final path directly (pickle/keras). `_preflight_check` validates `MIN_FREE_GB` once at start; minutes of training follow. ENOSPC/kill mid-save leaves a truncated `.pkl/.keras` in the LIVE pair dir that routing reads directly — the SIGSEGV-under-load pattern. | grep: 0 os.replace in trainers | Atomic save everywhere: write `path.tmp` → fsync → `os.replace`. Re-check disk immediately before each `.save()` even under `--skip-preflight`. |
| **V-H1** | Holdout validation fails OPEN. `scheduled_retrain.py:521-524 if total < 20: return True  # Don't block if we can't validate` (+ many other `return True` fail-open paths: 100/184/241/541/545/695/706). An un-validatable model promotes. Structural twin of the 2/3-pass gate that shipped val_acc=0.4839. | line 521-524 | Fail closed: `return False` when `total < MIN_HOLDOUT_SAMPLES` and on import/fetch failure. |
| **V-H2** | No reproducibility. `train_single_model_m1` + transformer trainer set no global tf/np seed; `src/utils/seed_manager.py` has 0 consumers. Two retrains of identical data ship different models / different val_acc → the gap-gate at the 10% boundary is a coin-flip. | grep seed = 0 in ship script | Call `seed_manager.set_global_seed(42)` at top of `main()` in both ship scripts; record seed in training_status.json. |
| **Met2/Met3** | Two TCN trainers report last-epoch train_acc against best-epoch val_acc (phantom gap). `tcn_trainer.py:760` + `tcn_volatility_trainer.py:531` use `history['accuracy'][-1]` while EarlyStopping/manual-restore save best-val weights. Can falsely quarantine a good model or mask a real gap once G1 is fixed (feeds the gap gate via `train_tcn_vol_regime`). | lines 760 / 531 | Report `history['accuracy'][best_epoch_idx]` (argmax val_acc / argmin val_loss to match saved weights), mirroring transformer `:2648-2657`. |
| **L-C1** | Embargo not wired into 4 loaders. `load_volatility_regime_data` (no `gap` param at all), `load_tcn_data:2471` (calls load_direction_data w/o gap), `load_forward_volatility_data:3107` (no embargo between seq splits), and the production RF-risk / ridge-confidence call sites (gap=0 while labels are forward). Bounded train/val boundary leak (one lookahead window each) → modestly inflated val. Direction head is correctly embargoed (`train_single_model_m1.py:438 gap=DIRECTION_LOOKAHEAD`). NOTE: momentum target is backward-looking (nowcast) → gap=0 OK there. | signatures / call sites | Add `gap` param to each forward-label loader, default to its lookahead; thread it from ship script. Assert `gap >= lookahead` and fail loud. |

## MEDIUM / LOW (verified by auditors, spot-checked)

- **Vp** — `purged_kfold_split` embargo is mis-implemented (shrinks the test block instead of purging post-test training samples; `walkforward_validation.py:394-404`). Latent (dead path) but must fix before wiring M1.
- **T-M** — no `_assert_scaler_not_identity` tripwire on tree/TCN trainers (each does one clean fit today; no active corruption, coverage gap only).
- **Q-L** — quarantine `restore_dir` rollback wired only for `tcn_vol_regime`; transformer/histgb retrains that over-ship leave the live dir model-less (prior artifact already overwritten before quarantine moves it).
- **G-M** — transformer `.keras` and `.meta.pkl` saved as two separate non-atomic writes; interruption desyncs model↔contract.
- **Th-M** — 3 disagreeing "min accuracy" thresholds (0.52 raw live / 0.53 raw / 0.53 balanced) across promotion_policy, unified_thresholds, deployment_gate; only 0.52-raw gates ships. `unified_thresholds`/`deployment_gate`/`validation_monitor` are dead (imported nowhere live) — a trap (look like the gate, aren't).
- **News-L** — news fusion zero-fills on empty fetch (OOD pattern, violates no-silent-zero-fill); PCA itself correctly fit train-only.

## What's correct (verified, no action)
- The documented historical label leaks (vol-regime percentile, confidence/streak_prob) are genuinely fixed (forward-realized labels, train-only cut fitting, sentinel tail-drop).
- Transformer head fully enforces the C1 contract incl refuse-on-version-mismatch.
- Quarantine is routing-safe (routing never globs `_quarantine/`; moves all stem sidecars).
- RESULT-on-exception is honest (thrown trainer → `passed:false`, not masked).
- Direction-loader scaler + forward-vol scaler fit train-only; sequence label alignment consistent.
- Dual-head TCN wiring (the forward-vol head wired 2026-06-09) is correct.

## Recommended remediation order
1. **V2 + V-H1** (one commit) — balanced-accuracy + class-collapse guard + fail-closed holdout. Closes the all-SHORT-ships hole. *The one that costs money.* (Gate-policy change → operator sign-off.)
2. **G1** — route all gap-checked trainers through `_read_trainer_metrics`, fail loud on empty. (Bug fix, restores intended gate.)
3. **R1** — atomic saves everywhere. (Robustness bug fix.)
4. **V-H2** — global seed. (Reproducibility; prerequisite for trusting any gate.)
5. **C1 + C2** — contract version on all heads + histgb own-feature-names. (Silent-OOD.)
6. **Met2/Met3, L-C1** — best-epoch metrics + embargo wiring.
7. **M1 decision** — wire walk-forward or delete + fix the CLAUDE.md claim. (Operator call.)
