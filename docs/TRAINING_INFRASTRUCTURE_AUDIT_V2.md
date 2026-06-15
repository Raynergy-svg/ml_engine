# Training Infrastructure Audit V2 — SOTA Assessment

**Auditor:** Dex (systematic file exploration + handoff reconciliation)
**Scope:** `src/training/`, `src/models/`, `src/rl/`, `src/sota_core/`, `src/scanner/agents/neural/`
**Date:** 2026-06-15

---

## Executive Summary

**Verdict: ALREADY SOTA.** The existing training stack is production-grade and does NOT need rebuilding. The new `src/sota_core/` and `src/scanner/agents/neural/` components are **complementary, not duplicative**.

**Two integration gaps remain** that prevent the new components from participating in the existing training ecosystem.

---

## What Already Exists (Confirmed SOTA)

### Deep Learning Architectures
- **TFT** (`TFTemporalFusionTransformer`) with VSN+GRN — already in `src/models/tensorflow_models.py:1193`
- **TCN** with dilated causal convolutions — `src/models/tensorflow_models.py:1776`
- **Transformer** pure attention encoder — `src/models/tensorflow_models.py:1029`
- **Attentive LSTM** — `src/models/tensorflow_models.py` (residual connections, recurrent dropout)

### Training Framework
- **EWC** (Elastic Weight Consolidation) for continual learning — `TransformerDirectionTrainer`
- **Replay buffer** with training lineage — `TransformerDirectionTrainer`
- **LLRD** (Layer-wise Learning Rate Decay) — `TransformerDirectionTrainer`
- **Prediction-collapse detection** + `AntiCollapseFocalLoss` — custom loss in `src/training/trainers/`
- **Warm-start / drift detector** — `TransformerDirectionTrainer`
- **JointMultiPairTrainer** with contrastive learning across pairs — `src/training/trainers/joint_trainer.py`

### RL Infrastructure
- **SAC** (Soft Actor-Critic) execution — `src/rl/sac_execution.py`
- **PPO Position Sizer** — `src/training/rl/position_sizer.py`
- **Reward modeling** + trajectory loading — `src/training/rl/`

### Observability
- **W&B** sweeps, multi-pair tracking, observatory dashboard
- **MLflow** experiment tracking with Alembic migrations
- **Training stability analyzer** — overfitting/collapse/drift detection

---

## New Components (The Gaps They Fill)

### Gap 1: End-to-End Raw Sequence Learning
**File:** `src/sota_core/raw_sequence_model.py`
- Consumes raw OHLCV (5 channels) instead of engineered features (7+ channels)
- Removes human bias in feature engineering
- **Differentiator:** Existing TFT uses engineered features; new model is raw-to-prediction

### Gap 2: Neural Specialist Agents
**File:** `src/scanner/agents/neural/`
- Replaces 15 rule-based agents with gradient-learned policies
- Each policy is an MLP with binary-crossentropy + PrioritizedReplayBuffer
- **Differentiator:** Existing ensemble learns meta-weights; new agents learn per-specialist policies

---

## Integration Gaps (Needs Work)

### IG-1: `SOTATrainer` does not subclass `BaseTrainer`
**File:** `src/sota_core/trainer.py`

`SOTATrainer` is a standalone class. It does not inherit from `BaseTrainer`, so it cannot:
- Participate in `JointMultiPairTrainer` joint training
- Use EWC for continual learning
- Use the existing W&B callback infrastructure
- Be registered in `MODEL_REGISTRY`

**Recommendation:** Refactor `SOTATrainer` to inherit from `BaseTrainer` and implement `train()`, `predict()`, `save()`, `load()`.

### IG-2: Neural agents have no EWC or joint training
**File:** `src/scanner/agents/neural/trainer.py`

`NeuralAgentTrainer` uses online gradient descent with a local replay buffer. It does not:
- Use EWC to protect previously learned policy weights
- Participate in walkforward validation
- Log to W&B
- Export model artifacts in the canonical format

**Recommendation:** Add optional EWC regularization and W&B logging hooks. Keep the online update path lightweight for live trading.

### IG-3: Evaluation module is not wired into CI
**Files:** `tests/evaluation/*`

Tests exist and pass, but:
- No soak data has been collected yet
- `soak_runner.py` has not been executed end-to-end
- `meta_ensemble.py` has not been trained on real legacy + SOTA errors

**Recommendation:** After model artifacts are trained, run `scripts/train_sota_and_neural.py --phase all` followed by a soak on held-out data.

---

## Duplication Check

| Component | Existing | New | Verdict |
|-----------|----------|-----|---------|
| VSN+GRN | ✅ `tensorflow_models.py` | ✅ `raw_sequence_model.py` | **Different input domains** — keep both |
| Replay buffer | ✅ `TransformerDirectionTrainer` | ✅ `PrioritizedReplayBuffer` | **Different use-cases** — keep both |
| MLP policy | ✅ `TFTCNPredictor` | ✅ `NeuralAgentBase` | **Different scopes** — keep both |
| Loss functions | ✅ `AntiCollapseFocalLoss` | ✅ `binary_crossentropy` | **Different problem formulations** — keep both |

**No code duplication detected.** Every new component serves a distinct purpose.

---

## Recommendations

1. **Do NOT rebuild the training stack.** Integrate into existing `BaseTrainer` framework.
2. **Register `RawSequenceModel`** as a 5th ensemble component (augment mode) for the soak comparison.
3. **Add EWC to `NeuralAgentTrainer`** before deploying to live trading (prevents catastrophic forgetting).
4. **Run the soak** as soon as `trained_data/models/sota_finetuned/sota_model.keras` exists.
