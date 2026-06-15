# Training Infrastructure Audit — SOTA Assessment

**Auditor:** Dex (systematic file exploration)
**Scope:** `src/training/`, `src/models/`, `src/rl/`, `scripts/train_*.py`
**Date:** 2026-06-15

---

## Executive Summary

Your training infrastructure is **already SOTA-grade** in many dimensions. The codebase contains production-quality implementations of virtually every modern technique for time-series trading prediction. What we built in `src/sota_core/` and `src/scanner/agents/neural/` is **complementary, not duplicative** — it fills two specific gaps that the existing infrastructure does not address.

**Verdict:** Do not rebuild the training stack. Integrate the new raw-sequence model and neural agents into the existing `BaseTrainer` / `JointMultiPairTrainer` framework.

---

## What Already Exists (SOTA Capabilities)

### 1. Deep Learning Architectures (`src/models/tensorflow_models.py` — 1,100+ lines)

| Model | Status | Notes |
|-------|--------|-------|
| **TFTemporalFusionTransformer** | ✅ Full implementation | Variable Selection Networks, GRN, static covariates, known future inputs, interpretable attention |
| **TFTemporalFusionTransformerEnhanced** | ✅ Full implementation | Adds instrument embeddings, session embeddings, hour/day embeddings |
| **TFTransformerPredictor** | ✅ Pure Transformer | Multi-head self-attention for time series |
| **TFTCNPredictor** | ✅ TCN with dilations | Dilated causal convolutions, faster than LSTM on Metal |
| **TFAttentiveLSTM** | ✅ LSTM + attention | Residual connections, recurrent dropout |
| **TFEnsemblePredictor** | ✅ Ensemble wrapper | Combines multiple architectures |

**Key insight:** The existing TFT already has `VariableSelectionNetwork` and `GatedResidualNetwork` — the same components we built in `src/sota_core/raw_sequence_model.py`. However, the existing TFT operates on **engineered features** (7+ channels including technical indicators), while our new model operates on **raw OHLCV** (5 channels only).

### 2. Training Framework (`src/training/trainers/`)

| Component | Lines | SOTA Features |
|-----------|-------|---------------|
| `TransformerDirectionTrainer` | 3,635 | EMA shadow weights, EWC continual learning, replay buffer, training lineage, warm-start, prediction collapse detection, LLRD, drift detector |
| `TCNTrainer` | ~600 | Dilated causal convolutions, class-weighted loss, BN + dropout regularization |
| `TransformerRegimeTrainer` | ~400 | Regime classification with Transformer encoder |
| `JointMultiPairTrainer` | 1,119 | **Contrastive learning** across pairs, instrument one-hot encoding, transfer learning, fine-tuning decisions |
| `BaseTrainer` | ~150 | Abstract interface: `train()`, `predict()`, `save()`, `load()` |

**Key insight:** The existing `TransformerDirectionTrainer` already implements many techniques that are considered SOTA for continual learning (EWC, replay buffer, LLRD). Our new trainer in `src/sota_core/trainer.py` does NOT have EWC or replay buffer — it would benefit from inheriting from `BaseTrainer` and reusing these mechanisms.

### 3. Data Pipeline (`src/training/data_preparation.py`)

- Time-series augmentation (jittering, scaling, time masking)
- Feature engineering and normalization
- Train/validation splitting with sequence preparation
- Class-weighted loss computation
- Label smoothing

### 4. RL Infrastructure (`src/training/rl/`, `src/rl/`)

| Component | Status |
|-----------|--------|
| SAC (Soft Actor-Critic) | ✅ `src/rl/sac_execution.py` |
| PPO Position Sizer | ✅ `src/training/rl/position_sizer.py` |
| Decision Transformer | 🟡 Stubs in `src/training/rl/decision_transformer.py` |
| Offline RL Trainer | 🟡 Contract stubs in `src/training/rl/offline_rl_trainer.py` |
| Trajectory Loading | ✅ `src/training/rl/trajectory_loader.py` |
| Reward Modeling | ✅ `src/training/rl/reward_model.py` |

### 5. Observability & Experiment Tracking

- **W&B integration**: Sweeps, multi-pair tracking, observatory dashboard
- **MLflow**: Experiment tracking with Alembic migrations
- **Training stability analyzer**: Detects overfitting, collapse, drift
- **Validation monitor**: Thresholds, criteria, promotion gates

### 6. Custom Loss Functions

- `AntiCollapseFocalLoss`: Focal loss + variance regularization (prevents prediction collapse)
- `HybridSFTLoss`: Supervised fine-tuning + RL hybrid loss
- Class-weighted crossentropy for long/short imbalance

---

## What Our New Code Adds (The Gaps)

### Gap 1: End-to-End Raw Sequence Learning

**Existing:** All DL models (TFT, Transformer, TCN) consume **engineered features** (SMA, RSI, ATR, MACD, etc.) as input channels. The feature engineering pipeline in `modular_data_loaders.py` computes ~40+ normalized features.

**Our addition (`src/sota_core/`):** A model that consumes **raw OHLCV only** (5 channels) and learns its own representations via CNN + Transformer + Variable Selection. No human-designed features.

**Why this matters:** The existing approach is "features first, model second" — which is how most quant systems work. Our approach is "raw data first, model discovers everything" — which is the paradigm shift that made ImageNet and GPT successful. In practice, this means:
- The existing model can only find patterns that exist in the engineered features
- Our model can discover novel patterns (e.g. complex order-flow dynamics) that don't map to any classical indicator

**Integration path:** The raw-sequence model should be registered as a new trainer class (`RawSequenceTrainer`) that inherits from `BaseTrainer`, so it can participate in the joint training framework.

### Gap 2: Self-Supervised Pre-Training

**Existing:** All training is **supervised** — models learn from labeled outcomes (direction, regime). There is no pre-training phase on unlabeled historical data.

**Our addition (`src/sota_core/trainer.py` pretrain phase):** Two-phase training:
1. Self-supervised pre-training: masked candle reconstruction + next-return prediction
2. Supervised fine-tuning: direction + regime labels

**Why this matters:** Self-supervised pre-training lets the model learn general market structure (trend, volatility, momentum) from years of unlabeled data before seeing any trade outcomes. This is the standard approach in NLP (BERT, GPT) and computer vision (MAE, SimCLR) but is rare in finance.

**Integration path:** The pre-training phase should be added as an optional step in the existing `train_all.py` pipeline, running before supervised training on historical OHLCV archives.

### Gap 3: Neural Agent Policies

**Existing:** Scanner agents (`src/scanner/agents/_team.py`) use **hand-coded heuristic formulas** (SMA crossovers, RSI thresholds, ATR comparisons). Agent weights adapt via scalar arithmetic (+0.10/-0.15).

**Our addition (`src/scanner/agents/neural/`):** Small MLP policies that learn from trade outcomes via gradient descent, with a prioritized replay buffer.

**Why this matters:** The existing weight-learning is a crude approximation of what neural policies do. Scalar weights can only say "I'm usually good at trend" — neural policies can learn "I'm good at trend when ADX > 25 and volume is above the 20-bar average."

**Integration path:** The neural agents are already designed as a drop-in replacement (`team_bridge.py`). They should be toggled via the existing `use_neural_agents` config flag.

---

## What Would Be Duplicate (Avoid)

| If We Built... | It Would Duplicate... | Status |
|------------------|------------------------|--------|
| Another Transformer trainer | `TransformerDirectionTrainer` (3,635 lines) | ❌ Don't build |
| Another TCN trainer | `TCNTrainer` | ❌ Don't build |
| Another TFT model | `TFTemporalFusionTransformer` / `Enhanced` | ❌ Don't build |
| Another data augmentation pipeline | `data_preparation.py::augment_time_series` | ❌ Don't build |
| Another W&B sweep runner | `src/training/wandb_sweep_agent.py` | ❌ Don't build |
| Another RL position sizer | `src/training/rl/position_sizer.py` (PPO) | ❌ Don't build |
| Another focal loss | `AntiCollapseFocalLoss` in `tensorflow_models.py` | ❌ Don't build |
| Another class-weighting scheme | `_compute_class_weights` in `tcn_trainer.py` | ❌ Don't build |
| Another EWC continual learning | `EWCTrainingCallback` in `transformer_trainer.py` | ❌ Don't build |
| Another replay buffer | `ReplayBuffer` in `transformer_trainer.py` | ❌ Don't build |

---

## Recommended Integration Architecture

Instead of keeping `src/sota_core/` as a separate module, integrate it into the existing training framework:

```
src/training/trainers/
  raw_sequence_trainer.py     # NEW: inherits BaseTrainer
    - Wraps src/sota_core/raw_sequence_model.py
    - Adds EWC, replay buffer from existing callbacks
    - Participates in JointMultiPairTrainer

src/training/
  pretrain_pipeline.py        # NEW: self-supervised phase
    - Wraps src/sota_core/trainer.py pretrain logic
    - Feeds unlabeled OHLCV archives
    - Outputs encoder weights for fine-tuning

src/scanner/agents/neural/
  (keep as-is — already integrated via team_bridge.py)
```

The key decision: **should the raw-sequence model replace or augment the existing Transformer ensemble?**

- **Replace**: Set `use_sota_inference: true` — the scanner loads only the raw-sequence model
- **Augment**: Add the raw-sequence model as a 5th ensemble component (alongside Transformer, XGBoost, Ridge, RF) — it votes in the existing ensemble

---

## File Map of Existing vs New

| What | Existing File | Our New File | Relationship |
|------|---------------|--------------|--------------|
| TFT model | `src/models/tensorflow_models.py:1193` | `src/sota_core/raw_sequence_model.py:53` | Both have VSN+GRN; existing uses engineered features, ours uses raw OHLCV |
| Transformer trainer | `src/training/trainers/transformer_trainer.py` | `src/sota_core/trainer.py` | Existing has EWC/replay/LLRD; ours has self-supervised pretraining |
| Data augmentation | `src/training/data_preparation.py` | None in our code | Use existing — ours only has jittering |
| Agent policies | `src/scanner/agents/_team.py` (rule-based) | `src/scanner/agents/neural/` (learned) | Drop-in replacement |
| RL position sizing | `src/training/rl/position_sizer.py` (PPO) | None in our code | Use existing |
| Custom losses | `src/models/tensorflow_models.py` (FocalLoss) | None in our code | Use existing — our model uses standard BCE |

---

## Conclusion

Your training infrastructure is **not behind SOTA** — it IS SOTA. The question is not "how do we build a modern training stack?" but rather "how do we integrate end-to-end raw-sequence learning and neural agents into an already-modern stack?"

The recommended next step: register `RawSequenceModel` as a `BaseTrainer` subclass so it can participate in joint training, walkforward validation, and W&B sweeps alongside the existing TFT/Transformer/TCN models.
