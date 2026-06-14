# Wire SOTA Modules into Scanner Execution Path

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, no subagents available). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect 11 orphaned Tier-7 SOTA modules into the live scanner execution path so they participate in inference, agent voting, execution, feedback, and autonomy loops.

**Architecture:** Lazy-init pattern following existing `_init_*()` conventions in `Scanner`. Each module is stored as `self._*` and invoked only when its config flag is enabled. TDD-first: failing integration test per module, then minimal wiring.

**Tech Stack:** Python 3.13, pytest, existing scanner engine (`src/scanner/engine.py`, `src/core/modular_inference.py`, `src/scanner/execution.py`, `src/scanner/agents/_team.py`, `src/scanner/feedback/post_trade_loop.py`).

---

## Task 2: Signal Layer — HybridInference + TimesFMAdapter

**Files:**
- Modify: `src/scanner/engine.py`
- Create: `tests/test_hybrid_inference_wiring.py`

**Steps:**
- [ ] Write failing test: `test_hybrid_inference_used_when_enabled` asserts that when `use_hybrid_inference=True`, `_run_inference` returns direction from `HybridInference.predict()`
- [ ] Run test → FAIL
- [ ] Add `_init_hybrid_inference()` to `Scanner.__init__`
- [ ] Patch `_run_inference()` to branch on `use_hybrid_inference`
- [ ] Run test → PASS
- [ ] Live-wiring audit: grep for actual `predict()` call in engine.py

## Task 3: Feature Engineering — CausalFeatureSelector

**Files:**
- Modify: `src/scanner/engine.py`
- Create: `tests/test_causal_feature_selector_wiring.py`

**Steps:**
- [ ] Write failing test: `test_causal_features_pruned_when_enabled`
- [ ] Add `_init_causal_feature_selector()` to `Scanner`
- [ ] Patch `_compute_features()` to call `selector.select()`
- [ ] Run test → PASS
- [ ] Live-wiring audit

## Task 4: Agent Layer — MetaStrategyAgent

**Files:**
- Modify: `src/scanner/agents/_team.py`
- Create: `tests/test_meta_strategy_agent_wiring.py`

**Steps:**
- [ ] Write failing test: `test_meta_strategy_overrides_weights`
- [ ] Init `MetaStrategyAgent` in `ScannerAgentTeam.__init__`
- [ ] Patch `vote()` to consult `meta.select(regime)`
- [ ] Run test → PASS
- [ ] Live-wiring audit

## Task 5: Execution Layer — CQLRebalancer

**Files:**
- Modify: `src/core/modular_inference.py`
- Create: `tests/test_cql_rebalancer_wiring.py`

**Steps:**
- [ ] Write failing test: `test_cql_rebalancer_sizing_used_when_enabled`
- [ ] Add `use_cql_rebalancer` flag check in `_calculate_position_size()`
- [ ] Run test → PASS

## Task 6: Execution Layer — SACExecutionAgent

**Files:**
- Modify: `src/scanner/execution.py`
- Create: `tests/test_sac_execution_wiring.py`

**Steps:**
- [ ] Write failing test: `test_sac_slices_large_orders`
- [ ] Patch `execute_trade()` to use SAC for order slicing above threshold
- [ ] Run test → PASS
- [ ] Execution-safety audit

## Task 7: Feedback Loop — RLHFRewardShaper

**Files:**
- Modify: `src/scanner/feedback/post_trade_loop.py`
- Create: `tests/test_rlhf_reward_wiring.py`

**Steps:**
- [ ] Write failing test: `test_rlhf_shapes_reward_when_enabled`
- [ ] Lazy-init `RLHFRewardShaper` in `PostTradeLoop`
- [ ] Patch `run()` to call `shape_reward()`
- [ ] Run test → PASS

## Task 8: Autonomy — SelfHealConfigUpdater

**Files:**
- Modify: `src/scanner/automation/state_engine.py` or equivalent
- Create: `tests/test_self_heal_config_updater_wiring.py`

**Steps:**
- [ ] Write failing test: `test_self_heal_config_updater_proposes_adjustments`
- [ ] Wire `SelfHealConfigUpdater` alongside existing `SelfHeal`
- [ ] Run test → PASS

## Task 9: Autonomy — ContinualLearner

**Files:**
- Modify: training pipeline / `src/training/retrain_worker.py` or similar
- Create: `tests/test_continual_learner_wiring.py`

**Steps:**
- [ ] Write failing test: `test_continual_learner_updates_incrementally`
- [ ] Wire `ContinualLearner.add_to_memory()` after nightly retrain
- [ ] Run test → PASS

## Task 10: Stress Test Harness — DiffusionMarketModel

**Files:**
- Modify: `src/scanner/analysis/backtest.py` or new harness
- Create: `tests/test_diffusion_market_model_wiring.py`

**Steps:**
- [ ] Write failing test: `test_diffusion_generates_synthetic_paths`
- [ ] Wire into backtest/stress test harness
- [ ] Run test → PASS

## Verification

- Full suite: `pytest tests/ -x -q`
- Live-wiring audit each module
- Execution-safety audit for execution.py changes
