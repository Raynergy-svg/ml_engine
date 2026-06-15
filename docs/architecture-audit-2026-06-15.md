# Architecture Audit — Loose / Orphaned Pipeline (2026-06-15)

**Scope:** inference · training · meta · autonomous loop. Method: 4 parallel read-only
Explore agents + direct disk verification of the highest-value claims.
**Confidence tags:** `VERIFIED` = grepped from disk this session (HIGH). `REPORTED` =
single subagent verdict, plausible, not independently re-verified (MEDIUM — confirm before acting).

This is an audit, not a fix. Nothing here is changed yet. Recommended fix-now items are in §6.

---

## 1. Live bugs in the autonomous loop (act first — cheap, real, running)

| Finding | Status | Location | Impact |
|---|---|---|---|
| `_run_learning_loop` **defined twice** — 2nd shadows 1st | VERIFIED | [continuous.py:1966](../src/scanner/automation/continuous.py) & [:2268](../src/scanner/automation/continuous.py) | Whichever logic was in the first def is dead; branch behaviour is whatever the 2nd def does. Silent regression. |
| Self-heal `logger.info("code_repair.deferred", error_sig=…, frequency=…)` | VERIFIED | [continuous.py:1134](../src/scanner/automation/continuous.py) (+1162/1166/1172/1176/1181) | stdlib `logging.Logger.info` rejects arbitrary kwargs → **TypeError every time code-repair fires**. The self-heal path is broken; failures are swallowed in a background thread. |
| `_error_frequency: Dict = {}` at **class scope** | VERIFIED | [continuous.py:1093](../src/scanner/automation/continuous.py) | Shared across all `ContinuousScanner` instances; two scanners corrupt each other's error counts and the "fire repair after N occurrences" threshold. |
| `_run_smart_loop` + observation logging run **twice per cycle** | REPORTED | continuous.py ~517–527 / 744–754, calls at 553 & 854 | Wasted CPU every scan cycle; duplicate observation writes. |
| `observation_consumer.save_state()` with **no restore on startup** | REPORTED | continuous.py:797 | Write-only state — persisted but never reloaded; state-persistence rule violated. |

## 2. Meta pipeline — orphan-key dead-write (the $3,527 failure class)

| Finding | Status | Location | Impact |
|---|---|---|---|
| `ScannerConfig.meta_labeler_threshold` (0.52) vs gate read `InferenceConfig.min_meta_confidence` (0.52) — **two different fields**; comment claims one configures the other | VERIFIED (mismatch) / REPORTED (no-bridge) | [config.py:563](../src/scanner/config.py) vs [modular_inference.py:514](../src/core/modular_inference.py) read at [:4812](../src/core/modular_inference.py); only `aggressive_min_meta_confidence` is ever assigned, at [engine.py:1523](../src/scanner/engine.py) | If no code maps `meta_labeler_threshold → min_meta_confidence` at InferenceConfig construction, **tuning that knob does nothing** — exact pattern of the documented dead-write incident. **Confirm InferenceConfig build path before asserting.** |
| `aggressive_min_meta_confidence` field defined, **zero readers** | REPORTED | [config.py:627](../src/scanner/config.py) | Dead config metadata; adjustments to it are inert. |
| `enable_trader_readiness_agent` dormant (Aura writer never shipped) | VERIFIED (known) | config.py (all profiles False) | Already documented; toggle wired, agent unimplemented. Benign. |

## 3. Training pipeline — train↔inference contract gaps

| Finding | Status | Location | Impact |
|---|---|---|---|
| **5 trainers don't receive `feature_names`** in their `train()` calls (TCN, LGBM-momentum, LGBM-risk, Ridge, HistGB) | REPORTED | train_single_model_m1.py (per-trainer call sites) | Heads save `feature_names=None` → gates can't reconstruct feature alignment at inference → silent degradation / abstention. Corroborated by claude-mem obs 7585/7595. |
| `train_tcn` missing `feature_names` **and** `regime_quantiles/atr_col` | REPORTED | train_single_model_m1.py train_tcn | TCN regime one-hot reconstruction can't fire → regime head abstains silently. Worst training orphan per the auditor. |
| Feature-contract enforcement **only on transformer** | REPORTED | gates.py (version check transformer-only) | TCN / momentum / confidence / risk run with no `feature_pipeline_version` check — a pipeline change silently breaks 4 heads while only direction fails closed. |
| Gap-gate **not applied** to LGBM-momentum / LGBM-risk / Ridge | REPORTED | train_single_model_m1.py:779 region | Those heads can ship at any train/val gap (10% rail covers direction/tcn/histgb only). |
| Orphaned trainers: `xgboost_trainer`, `random_forest_trainer`, `transformer_regime_trainer` not in `MODEL_TRAINERS` | REPORTED | src/training/trainers/* | Dead/dormant training code (transformer_regime explicitly shelved as "broken"). |

## 4. Inference pipeline — orphaned modules (mostly fail-safe, but dead)

| Finding | Status | Location | Impact |
|---|---|---|---|
| `CQLRebalancer` + `ContinualLearner` declared `None`, **never initialized** | REPORTED | modular_inference.py:783–784 | Referenced only behind null-checks that always skip. Dead code from commit 286509d. No live effect, but misleading. |
| `enable_llm_integration=False` default; `self.enable_llm` only read in `predict_verbose`, which `predict()` never calls | REPORTED | modular_inference.py:675/695/4668 | Fully dormant feature; also note: LLM-in-decision-path would violate FR-6 if ever enabled. |
| `SOTAInference` never imported by engine; `HybridInference.register_foundation/xgboost` never called | REPORTED | sota_core/inference.py, hybrid_inference.py:95–103 | SOTA inference stack dormant at runtime; registration points dead. Consistent with the freeze-conflict finding. |
| Joint fallback still reachable when `use_per_pair_routing=True` | REPORTED | gates.py:435–483 | Deprecation steps 3–5 (improvement.md) still pending; stale joint models can shadow fresh per-pair under routing. |

## 5. SOTA modules with no off-switch (freeze gap — cross-ref)

`MetaStrategyAgent`, `SACExecutionAgent`, `CausalFeatureSelector`, `CausalDiscovery` initialize
"if available" with **no config flag** (REPORTED). `HybridInference`/`CQLRebalancer` are flag-gated
default-False. None flips a live trade today, but the four flagless ones are incompatible with the
factor-pivot freeze — tracked in `docs/sota-modernization-review-2026-06-15.md` §4.

---

## 6. Recommended disposition

**Fix-now (verified, cheap, scoped — safe in one PR):**
1. Delete the dead first `_run_learning_loop` (continuous.py:1966); keep one definition.
2. Fix the self-heal `logger.info(..., kwargs)` calls → f-strings or a structlog logger (the
   self-heal path is currently broken). 
3. Move `_error_frequency` into `__init__` (instance state, not class state).
4. De-duplicate the double `_run_smart_loop` / observation-logging calls per cycle.

**Verify-then-fix (high value, needs one confirmation each):**
5. Meta dead-write (§2): confirm whether `min_meta_confidence` is populated from
   `meta_labeler_threshold` at InferenceConfig construction. If not, wire it — a live tuning knob
   is currently inert.
6. The 5-trainer `feature_names` gap (§3): confirm the `train()` signatures, then thread
   `feature_names` (and TCN regime fields) through. This is a silent-degradation contract gap.

**Defer / track (orphan cleanup, no live harm):** §3 orphaned trainers, §4 dead SOTA modules,
§5 freeze flags (handled in the SOTA review's US-009), joint-fallback removal (improvement.md ledger).

**Do NOT auto-fix without operator decision:** anything touching gate thresholds, profile values,
or the joint-deprecation sequence — those are operator calls per CLAUDE.md scope guardrails.
