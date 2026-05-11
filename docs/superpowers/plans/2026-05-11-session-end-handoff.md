# 2026-05-11 Session-End Handoff — Trainer Bug Stack + Strategic Pivot Decision

> **Bot stays halted.** No artifact on disk currently produces tradeable
> live inference. The path forward is trainer-bug-fixes, not more retrains
> on the broken pipeline. News fusion confirmed rejected via Phase 5.C
> verdict (3/3 configs fail at H1 lookahead=24).

## 1. What this session shipped (15 commits on local `main`)

| Commit | What |
|---|---|
| `07a1f96` | Smart profile loosening (4 thresholds) |
| `1939e9e` | Momentum cascade health normalization |
| `c38b166` | Correlation: binary block → net-exposure cap |
| `c5cd028` | Disagreement_hard_block diagnostic spec |
| `8f5ce29` | gates.py per-pair calibrated threshold (secondary path) |
| `b8d229b` | CLAUDE.md confidence-calibration rule |
| `4f7001a` | Revert of broken ensemble_conflict three-magnitude semantics |
| `bd908f9` | gitignore hardening + skip-worktree for 71 runtime files |
| `d93e37f` | Deploy gate: M15 mandatory (kill 2/3 majority bypass) |
| `c0530b5` | Bug A fix: `_predict_with_uncertainty` apply scaler + regime |
| `4e88550` | Tab/space fix in `src/tui/app.py` (Buddy could not start) |
| `19676e0` | trainer.load reads `regime_quantiles` / `regime_atr_col` / version |
| `5bb6af3` | Log rotation backupCount 3→1 (disk pressure) |
| `69e1eb2` | scheduled_retrain default `--granularity` H1→M15 |
| `1d2408b` | Holdout harness uses per-pair routing (mirrors Scanner runtime) |

## 2. The bug stack uncovered (NOT fixed this session)

### Bug #1: Trainer ignores `--granularity` flag
- **Evidence**: Team B (5.21pm) — today's `--granularity M15` retrain produced an artifact with `lineage.granularity=H1`, `lineage.instrument=JOINT_EUR_USD`, 11958 train samples.
- **Why**: The joint trainer's internals (somewhere in `src/training/trainers/transformer_trainer.py` / `src/training/buddy_training_helpers.py` / `src/training/joint_trainer.py`) hardcodes H1 regardless of the CLI flag passed to `scripts/scheduled_retrain.py`.
- **Fix-it-next**: Trace `train_per_pair_correlation_ensemble` and joint trainer code paths; find where `granularity` should be threaded but isn't. Likely a default parameter that wasn't surfaced to the CLI.
- **Confidence**: HIGH (lineage data is direct evidence)

### Bug #2: Trainer produces collapsed predictors (90% LONG)
- **Evidence**: Team C (4.10pm) — ablation showed today's EUR_USD model predicts LONG 90% of the time at calibration threshold 0.5; 93.5% at calibrated threshold 0.4077. Slice was 57% LONG-actual. Model scores BELOW the always-LONG baseline.
- **Why**: Class weights not applying OR training collapsed to majority-class predictor. Need to inspect training-time per-class accuracy in the saved `metrics` dict and verify weight application.
- **Fix-it-next**: Inspect `metrics.val_up_accuracy` vs `metrics.val_down_accuracy` for collapsed signal. Trace class weight logic in trainer.
- **Confidence**: HIGH (200-sample run with verified bytes-identical production artifact)

### Bug #3: Meta sidecar doesn't save `scaler` / `regime_quantiles` / `feature_pipeline_version` for new artifacts
- **Evidence**: Production EUR_USD + USD_JPY both have `scaler=None`, `feature_pipeline_version=N/A` in their meta sidecars. Bug A's fix (`c0530b5`) requires these to activate; with them None, the helper falls back to legacy unscaled inference path.
- **Why**: Trainer's `save()` method either doesn't save these OR `trainer.load()` (fixed in `19676e0`) was the wrong layer.
- **Fix-it-next**: Read `transformer_trainer.py:save()` and verify the contract-complete keys (`scaler`, `regime_quantiles`, `regime_atr_col`, `feature_pipeline_version`, `feature_names`) are written to the meta dict before pickling.
- **Confidence**: HIGH (verified via `joblib.load` on both artifacts)

### Bug #4: Meta sidecar doesn't save `lookahead`
- **Evidence**: Team B — no `lookahead`/`horizon`/`forward_bars` key anywhere in the saved meta. Must be inferred from `DIRECTION_DEFAULTS['lookahead']` constant.
- **Why**: Contract gap; trainer doesn't record what it trained for.
- **Fix-it-next**: Add `lookahead` to the meta dict in `transformer_trainer.py:save()`. Pair with #3 (both are contract gaps in same function).
- **Confidence**: HIGH

### Bug #5: `HOLDOUT_LOOKAHEAD=24` is bars-not-hours
- **Evidence**: `scripts/scheduled_retrain.py:489` — `HOLDOUT_LOOKAHEAD=24` used as raw `i+24` indexing on the holdout dataframe. M15 model holdout: 6h forward target. H1 model holdout: 24h forward target. Bug #1's H1-train + M15-holdout creates a 4x horizon mismatch.
- **Fix-it-next**: Parametrize `HOLDOUT_LOOKAHEAD` per granularity (H1→24, M15→96 for "24 hours forward" consistency), OR read it from the loaded model's meta (per Bug #4 fix).
- **Confidence**: HIGH

## 3. What we falsified this session

| Hypothesis I confidently asserted | Verdict |
|---|---|
| "gates.py:1902 hardcoded 0.5 cut is the load-bearing cause of 88% SHORT" | FALSE — `modular_inference.py:1614-1619` was already applying calibrated thresholds. Fix landed in `8f5ce29` was correct but secondary. |
| "Team C's ensemble_conflict resurrection helps" | FALSE — direction-signed magnitudes carry the same sign by construction; resolver fired false-positive blocks. Reverted in `4f7001a`. |
| "Production model is today's failed retrain (val_acc 0.4839)" | FALSE — my own `git stash push -u` swept the failed retrain into the stash and reverted production to the May 4 good artifact (val_acc 0.5116). |
| "M15 ceiling is exhausted at ~50%" | FALSE — today's model wasn't trained at M15 AND it's collapsed (90% LONG). Real M15 ceiling unknown. |
| "Phase 5.D tuned models (58.3% EUR_USD, 63.2% USD_JPY) survive" | FALSE — disk artifacts overwritten / swept; only current production state matters. |
| "News fusion (P1) is the next lever" | FALSE for H1/lookahead=24 — Phase 5.C verdict rejected 3/3 configs. Untested at M15. |

CLAUDE.md confidence-calibration rule promoted this session (`b8d229b`) was operator-driven feedback after my third wrong-causal-claim incident.

## 4. Next-session priority order

1. **Bug #3 (meta contract gap on save)** — quickest single-line audit. Open `transformer_trainer.py:save()`, verify what's written. ~15 min.
2. **Bug #1 (granularity ignored)** — trace joint trainer code paths for where `granularity` should be passed but isn't. ~30-60 min.
3. **Bug #2 (model collapse)** — inspect per-class accuracy + class-weight logic. May be the load-bearing fix if class weights aren't applying. ~30-60 min.
4. **Bug #4 + #5 (lookahead)** — pair these; save `lookahead` to meta in save(), read it in holdout. ~15 min.
5. **Retrain USD_JPY at H1** with fixed trainer (NOT M15 — H1 is where 62% was). ~5 min.
6. **Validate via fixed harness** — should reproduce 60%+ on H1 holdout. Confirms fix worked.
7. **Forward-test USD_JPY on demo** for 1-2 weeks per Phase 5.C Path B recommendation.

## 5. Operator decisions deferred

- **Stash `stash@{0}`** still preserves 404 dirty files including 5 preserved EUR_USD artifacts (`_oos_train`, `_pre_tuned_phase5d`, `_tuned`, `_news_lb4`, `_news_k32`) and the failed retrain. Either pop selectively, drop, or leave.
- **`git push`** never executed this session. Local `main` is ~193 commits ahead of origin/main, 1 commit behind (the test-coverage merge from earlier). Resolve before any push.
- **Worktrees** — 2.4 GB still on disk (3 heal-train + 1 detached-HEAD scratch with operator's scaler-audit scripts). Operator's WIP.

## 6. Halt status

**`state.json:halted=true` correctly preserved.** Current EUR_USD artifact is 49.5% M15 collapsed. Current USD_JPY artifact is 59.6% val_acc but contract-incomplete (scaler=None). Neither is safe to unhalt.

The system is in a better state than session start (deploy gate strict, contract reading wired, granularity default fixed at spawner level, harness now mirrors runtime) but cannot trade until the trainer bug stack is closed.

## 7. Reading order for next session

1. This doc.
2. `docs/superpowers/plans/2026-05-08-phase5c-final-verdict.md` (news rejection)
3. `docs/superpowers/plans/2026-05-09-phase5d-final-verdict.md` (USD_JPY tuned numbers)
4. `docs/superpowers/specs/2026-05-11-disagreement-hard-block-diagnosis-design.md` (the diagnostic chain that started here)
5. `CLAUDE.md` "Honesty & verification protocol" — confidence-calibration rule (promoted this session)
