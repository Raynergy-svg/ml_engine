# RL Integration Architecture Analysis

**Document Purpose**: Comprehensive analysis of the current training architecture to plan reinforcement learning integration strategy that preserves existing methodologies.

**Date**: February 2026  
**Status**: Architecture Analysis Complete

---

## Table of Contents

1. [Current Architecture Summary](#1-current-architecture-summary)
2. [Optimization Loop Analysis](#2-optimization-loop-analysis)
3. [Integration Points](#3-integration-points)
4. [Protected Components](#4-protected-components)
5. [Recommendations](#5-recommendations)

---

## 1. Current Architecture Summary

### 1.1 High-Level Training Pipeline

The ML trading engine uses a **modular ensemble architecture** with four independent specialist models trained on different feature subsets:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │ Data Loading │───▶│ Feature          │───▶│ Feature Selection       │   │
│  │ - OANDA API  │    │ Engineering      │    │ - RF Importance         │   │
│  │ - CSV files  │    │ - 200+ features  │    │ - F-test scoring        │   │
│  │ - Multi-pair │    │ - Noise reduction│    │ - Top-K selection       │   │
│  └──────────────┘    └──────────────────┘    └─────────────────────────┘   │
│                                                        │                     │
│                                                        ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MODULAR ENSEMBLE TRAINING                          │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │  │
│  │  │ Transformer     │  │ XGBoost         │  │ Random Forest       │  │  │
│  │  │ Direction Model │  │ Momentum Model  │  │ Risk Model          │  │  │
│  │  │                 │  │                 │  │                     │  │  │
│  │  │ - Self-attention│  │ - Gradient boost│  │ - Ensemble trees    │  │  │
│  │  │ - Sequential    │  │ - Tabular optim │  │ - Risk estimation   │  │  │
│  │  │ - EMA/EWC/Replay│  │ - Fast training │  │ - Drawdown/streak   │  │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │  │
│  │           │                    │                     │              │  │
│  │           └────────────────────┼─────────────────────┘              │  │
│  │                                ▼                                    │  │
│  │                    ┌─────────────────────┐                          │  │
│  │                    │ Ridge/ElasticNet    │                          │  │
│  │                    │ Confidence Model    │                          │  │
│  │                    │ - L1/L2 regression  │                          │  │
│  │                    │ - Calibration       │                          │  │
│  │                    └─────────────────────┘                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                │                                            │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    POST-TRAINING                                      │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │  - Meta-labeler training (trade filter)                              │  │
│  │  - Confidence calibration (Platt scaling)                            │  │
│  │  - RL Position Sizer training (optional)                             │  │
│  │  - Deployment validation gates                                       │  │
│  │  - Model persistence (.keras, .pkl, .meta.json)                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Transformer Trainer | [`src/training/trainers/transformer_trainer.py`](src/training/trainers/transformer_trainer.py) | Main neural network trainer with continual learning |
| Training Config | [`src/training/trainers/config.py`](src/training/trainers/config.py) | Configuration dataclasses for all trainers |
| CLI Orchestration | [`cli/training.py`](cli/training.py) | Training pipeline orchestration (3900+ lines) |
| Training Operations | [`cli/training_ops.py`](cli/training_ops.py) | RL training, retraining gates, validation |
| Modular Trainers | [`src/training/modular_trainers.py`](src/training/modular_trainers.py) | XGBoost, RF, Ridge trainer implementations |
| RL Position Sizing | [`rl_position_sizing.py`](rl_position_sizing.py) | Existing PPO-based position sizer |
| Online Retrainer | [`online_retrainer.py`](online_retrainer.py) | Incremental gate model retraining |

### 1.3 Model Types

```python
# From TrainerConfig in src/training/trainers/config.py
@dataclass
class TrainerConfig:
    # Transformer architecture
    transformer_d_model: int = 32        # Model dimension
    transformer_num_heads: int = 4       # Attention heads
    transformer_num_layers: int = 2      # Encoder layers
    transformer_dff: int = 64            # Feedforward dimension
    transformer_dropout: float = 0.2     # Dropout rate
    
    # Continual learning
    use_ema: bool = True                 # EMA shadow weights
    ema_decay: float = 0.999             # EMA decay factor
    use_ewc: bool = True                 # Elastic Weight Consolidation
    ewc_lambda: float = 100.0            # EWC penalty strength
    use_replay_buffer: bool = True       # Experience replay
```

---

## 2. Optimization Loop Analysis

### 2.1 Gradient Flow Architecture

The Transformer trainer uses standard TensorFlow/Keras backpropagation with several enhancements:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZATION LOOP STRUCTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FORWARD PASS                                       │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Input Sequence (batch, seq_len, features)                           │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  Input Projection + Positional Encoding                              │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  Transformer Encoder Layers (x N)                                    │   │
│  │  - Multi-Head Self-Attention                                         │   │
│  │  - Feedforward Network                                               │   │
│  │  - Layer Normalization                                               │   │
│  │  - Dropout                                                           │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  Global Average Pooling (sequence → single vector)                   │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  Output Head (Dense → Sigmoid)                                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  Predictions: direction_prob, confidence_score                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LOSS COMPUTATION                                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  Loss Priority Chain:                                                 │   │
│  │  1. MADL Loss (if enabled) - Directional profitability               │   │
│  │  2. Hybrid CB Anti-Collapse - Class-balanced + variance reg         │   │
│  │  3. Anti-Collapse Focal Loss - Variance regularization              │   │
│  │  4. Class-Balanced Focal Loss - Extreme imbalance handling          │   │
│  │  5. Binary Focal Loss - Hard example focusing                       │   │
│  │  6. Binary Cross-Entropy - Fallback                                 │   │
│  │                                                                       │   │
│  │  Total Loss = Direction_Loss + λ_conf * Confidence_Loss + EWC_Penalty│   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    BACKWARD PASS                                      │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  Optimizer: Adam (legacy on macOS)                                   │   │
│  │  - learning_rate: configurable (default 0.001)                       │   │
│  │  - clipnorm: 1.0 (gradient clipping)                                │   │
│  │                                                                       │   │
│  │  LR Scheduling:                                                       │   │
│  │  - Cosine annealing with warm restarts                              │   │
│  │  - SWA (Stochastic Weight Averaging) in final 25%                   │   │
│  │  - Warm-start LR reduction (33x for continual learning)             │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    WEIGHT UPDATE HOOKS                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  EMA Callback (every N steps):                                       │   │
│  │    θ_ema = α * θ_ema + (1-α) * θ_current                            │   │
│  │                                                                       │   │
│  │  EWC Callback (post-training):                                       │   │
│  │    F = ∂²L/∂θ² (Fisher Information Matrix)                          │   │
│  │    Store F for next task's penalty computation                       │   │
│  │                                                                       │   │
│  │  Replay Buffer (during training):                                    │   │
│  │    Mix 20% replay samples with current batch                         │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Loss Function Architecture

The loss function selection follows a priority chain with sophisticated anti-collapse mechanisms:

```python
# Loss priority from transformer_trainer.py:_create_loss_function()
LOSS_PRIORITY = [
    ("madl_loss", MADLLoss),              # Directional profitability
    ("hybrid_cb_anticollapse", HybridCBAntiCollapseLoss),  # Recommended
    ("anti_collapse_focal", AntiCollapseFocalLoss),        # Variance reg
    ("cb_focal", ClassBalancedFocalLoss),  # Extreme imbalance
    ("focal", BinaryFocalLoss),            # Hard example mining
    ("bce", BinaryCrossentropy),           # Fallback
]

# Anti-collapse variance regularization
# From config:
anti_collapse_base_variance_weight: float = 0.2
sample_weight_max_multiplier: float = 5.0
```

### 2.3 Callback System

The training uses a comprehensive callback system for monitoring and intervention:

| Callback | Purpose | Trigger |
|----------|---------|---------|
| `EarlyStopping` | Stop when validation plateaus | Every epoch |
| `OverfitPreventionCallback` | Dynamic dropout/LR adjustment | Train-val gap > threshold |
| `PredictionCollapseCallback` | Detect prediction variance collapse | Every epoch |
| `ProactiveCollapsePreventionCallback` | Pre-emptive intervention | Variance < threshold |
| `GradualUnfreezeCallback` | Layer-by-layer unfreezing | Warm-start only |
| `EMAUpdateCallback` | Update shadow weights | Every N batches |
| `EWCTrainingCallback` | Monitor EWC penalty | Every epoch |
| `RichEpochCallback` | Console progress display | Every epoch |

---

## 3. Integration Points

### 3.1 Existing RL Integration

The codebase already has RL integration points that serve as reference patterns:

#### 3.1.1 RL Position Sizer Training Hook

```python
# From cli/training.py:_train_rl_position_sizer_if_ready()
# Location: After ensemble training completes

def _train_rl_position_sizer_if_ready(
    console,
    rl_timesteps: int = 500_000,
    min_samples: int = 500,
    *,
    features: np.ndarray | None = None,
    ensemble_predictions: np.ndarray | None = None,
    prices: np.ndarray | None = None,
) -> bool:
    """
    Train RL position sizer using ensemble training data.
    
    Called after ensemble training completes successfully.
    Uses actual features, ensemble predictions, and price data.
    """
```

**Integration Pattern**: Post-training hook with data passthrough

#### 3.1.2 RL Training Operations

```python
# From cli/training_ops.py

def train_rl_sizer(...)     # Full RL position sizer training
def train_rl_gates(...)     # RL gate threshold optimizer (SAC)
def train_rl_exits(...)     # RL exit timing optimizer (PPO)
```

**Integration Pattern**: CLI commands with subprocess isolation for TF/PyTorch compatibility

### 3.2 Identified Integration Points for New RL Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RL INTEGRATION POINTS MAP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT A: Pre-Training Data Preparation                                     │
│  ─────────────────────────────────────                                      │
│  File: cli/training.py:_train_buddy_impl()                                  │
│  Lines: ~2950-3050 (after feature engineering)                              │
│                                                                              │
│  Hook: Feature/label preparation complete, before model training            │
│  Use Case: RL environment setup with prepared data                          │
│  Risk Level: LOW (read-only access to prepared data)                        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT B: Post-Ensemble Training                                            │
│  ─────────────────────────────────                                          │
│  File: cli/training.py:_train_ensemble_models()                             │
│  Lines: ~1040-1075 (after model saving)                                     │
│                                                                              │
│  Hook: All ensemble models trained, before validation                       │
│  Use Case: RL training using ensemble predictions (CURRENT PATTERN)         │
│  Risk Level: LOW (established pattern via _train_rl_after_ensemble)         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT C: Inference Pipeline                                                │
│  ─────────────────────────────                                              │
│  File: src/core/modular_inference.py                                        │
│  Class: ModularEnsembleInference                                            │
│                                                                              │
│  Hook: Signal generation, before position sizing                            │
│  Use Case: RL-based signal filtering, confidence adjustment                 │
│  Risk Level: MEDIUM (affects trading decisions)                             │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT D: Online Retraining Loop                                            │
│  ─────────────────────────────────                                          │
│  File: online_retrainer.py                                                  │
│  Class: OnlineRetrainer                                                     │
│                                                                              │
│  Hook: Drift detection triggers retraining                                  │
│  Use Case: RL-based retraining decisions, adaptive thresholds               │
│  Risk Level: MEDIUM (modifies model state)                                  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT E: Deployment Validation                                             │
│  ─────────────────────────────────                                          │
│  File: src/training/deployment_gate.py                                      │
│  Class: DeploymentValidator                                                 │
│                                                                              │
│  Hook: Pre-deployment checks                                                │
│  Use Case: RL-based validation, stress testing                              │
│  Risk Level: LOW (read-only validation)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Recommended Integration Patterns

#### Pattern 1: Auxiliary RL Trainer (Recommended)

Create a parallel RL training pipeline that consumes ensemble outputs:

```python
# Proposed: src/training/rl_auxiliary_trainer.py

class RLAuxiliaryTrainer:
    """
    Auxiliary RL trainer that runs alongside main training.
    
    Does NOT modify Transformer weights.
    Uses ensemble predictions as observations.
    Outputs position sizing, exit timing, or threshold adjustments.
    """
    
    def train_after_ensemble(
        self,
        features: np.ndarray,
        predictions: Dict[str, np.ndarray],
        prices: np.ndarray,
    ) -> Dict[str, Any]:
        """Train RL components after ensemble training completes."""
        pass
```

#### Pattern 2: RL Callback Integration

Add RL components as Keras callbacks for tight integration:

```python
# Proposed: src/training/callbacks/rl_callback.py

class RLIntegrationCallback(tf.keras.callbacks.Callback):
    """
    Callback that exposes training data to RL components.
    
    - on_epoch_end: Provide predictions to RL buffer
    - on_train_end: Trigger RL training
    """
```

#### Pattern 3: Subprocess Isolation (Current Pattern)

Continue using subprocess isolation for TF/PyTorch compatibility:

```python
# Current pattern from cli/training_ops.py:train_rl_sizer()
# Uses subprocess to avoid TF/PyTorch GPU conflicts
cmd = [python_exe, str(script_path), "--data", str(data_file)]
process = subprocess.Popen(cmd, ...)
```

---

## 4. Protected Components

### 4.1 Critical Training Components (DO NOT MODIFY)

| Component | File | Reason |
|-----------|------|--------|
| Transformer Architecture | [`transformer_trainer.py:_build_model()`](src/training/trainers/transformer_trainer.py) | Core model structure |
| Loss Function Chain | [`transformer_trainer.py:_create_loss_function()`](src/training/trainers/transformer_trainer.py) | Training stability |
| EMA Implementation | [`modular_trainers.py:EMACallback`](src/training/modular_trainers.py) | Continual learning |
| EWC Penalty | [`modular_trainers.py:EWCPenalty`](src/training/modular_trainers.py) | Catastrophic forgetting prevention |
| Replay Buffer | [`modular_trainers.py:ReplayBuffer`](src/training/modular_trainers.py) | Experience replay |
| Training Lineage | [`modular_trainers.py:TrainingLineage`](src/training/modular_trainers.py) | Audit trail |
| Checkpoint Format | [`transformer_trainer.py:save()`](src/training/trainers/transformer_trainer.py) | Model persistence compatibility |

### 4.2 Protected Data Flows

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROTECTED DATA FLOWS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. FEATURE ENGINEERING PIPELINE                                            │
│     src/data/feature_engineering.py → FeatureEngineering.create_features()  │
│     - Must remain deterministic for model compatibility                     │
│     - Feature order and naming must not change                              │
│                                                                              │
│  2. SEQUENCE CREATION                                                        │
│     cli/training.py:make_sequence_dataset()                                 │
│     - Window semantics (label at last timestep)                             │
│     - Train/val split ordering (temporal)                                   │
│                                                                              │
│  3. STANDARDIZATION                                                          │
│     StandardScaler fit on train, applied to val                             │
│     - Scaler must be saved/loaded with model                                │
│     - Feature statistics stored in metadata                                 │
│                                                                              │
│  4. CONTINUAL LEARNING STATE                                                 │
│     EMA weights, EWC Fisher, Replay Buffer                                  │
│     - Must be preserved across training sessions                            │
│     - Stored in .ema.pkl, .ewc.pkl, replay/                                 │
│                                                                              │
│  5. MODEL METADATA                                                           │
│     .meta.json / .meta.pkl files                                            │
│     - Feature columns, scaler params, metrics                               │
│     - Required for inference compatibility                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Configuration Constraints

```python
# From src/training/trainers/config.py
# These parameters affect model architecture and should not be changed
# without retraining from scratch:

ARCHITECTURE_PARAMS = {
    "transformer_d_model": 32,      # Model dimension
    "transformer_num_heads": 4,     # Must divide d_model evenly
    "transformer_num_layers": 2,    # Encoder depth
    "transformer_dff": 64,          # Feedforward dimension
    "seq_len": 60,                  # Input sequence length
}

# These affect continual learning and must be consistent:
CONTINUAL_LEARNING_PARAMS = {
    "use_ema": True,
    "ema_decay": 0.999,
    "use_ewc": True,
    "ewc_lambda": 100.0,
    "ewc_gamma": 0.95,
    "use_replay_buffer": True,
    "replay_buffer_ratio": 0.10,
    "replay_mix_ratio": 0.20,
}
```

---

## 5. Recommendations

### 5.1 RL Framework Placement

Based on the analysis, we recommend the following RL integration strategy:

#### Tier 1: Post-Ensemble RL (Recommended First Step)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED RL ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXISTING PIPELINE (UNCHANGED)                                              │
│  ─────────────────────────────────                                          │
│                                                                              │
│  Data → Features → Transformer → XGBoost → RF → Ridge → Ensemble Output    │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                                                                              │
│  NEW RL LAYER (AUXILIARY)                                                   │
│  ─────────────────────────────                                              │
│                                                                              │
│  Ensemble Output → ┌──────────────────┐ → Final Trading Decision           │
│                    │ RL Position Sizer │                                     │
│                    │ - PPO Agent       │                                     │
│                    │ - Market state    │                                     │
│                    │ - Account state   │                                     │
│                    └──────────────────┘                                     │
│                                                                              │
│                    ┌──────────────────┐                                     │
│                    │ RL Gate Optimizer │                                     │
│                    │ - Threshold adapt │                                     │
│                    │ - Regime detect   │                                     │
│                    └──────────────────┘                                     │
│                                                                              │
│                    ┌──────────────────┐                                     │
│                    │ RL Exit Timer     │                                     │
│                    │ - Hold/TP/SL dec  │                                     │
│                    │ - PnL optimize    │                                     │
│                    └──────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Rationale**:
- Zero modification to existing training pipeline
- Uses ensemble predictions as observations (current pattern)
- Can be enabled/disabled independently
- Subprocess isolation prevents TF/PyTorch conflicts

#### Tier 2: Inference Integration

```python
# Proposed: src/core/rl_inference_integration.py

class RLInferenceIntegration:
    """
    Integrates RL components into the inference pipeline.
    
    Usage:
        ensemble = ModularEnsembleInference()
        rl_integration = RLInferenceIntegration()
        
        signal = ensemble.predict(df)
        enhanced_signal = rl_integration.enhance(signal)
    """
    
    def enhance(self, signal: TradingSignal) -> TradingSignal:
        """Apply RL enhancements to ensemble signal."""
        # 1. Adjust position size using RL sizer
        signal.position_size = self.rl_sizer.get_position_size(...)
        
        # 2. Adjust thresholds using RL gate optimizer
        signal.confidence_threshold = self.rl_gates.get_threshold(...)
        
        # 3. Set exit strategy using RL exit timer
        signal.exit_strategy = self.rl_exits.get_strategy(...)
        
        return signal
```

### 5.2 Implementation Phases

| Phase | Task | Risk | Dependencies |
|-------|------|------|--------------|
| 1 | Document existing RL position sizer usage | Low | None |
| 2 | Add RL gate threshold optimizer | Low | Phase 1 |
| 3 | Add RL exit timing optimizer | Low | Phase 1 |
| 4 | Create unified RL inference integration | Medium | Phases 1-3 |
| 5 | Add RL-based validation checks | Low | Phase 4 |
| 6 | Explore RL for meta-learning | High | Phases 1-5 |

### 5.3 Code Organization Recommendations

```
src/
├── training/
│   ├── trainers/
│   │   ├── transformer_trainer.py  # UNCHANGED
│   │   └── config.py               # UNCHANGED
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── position_sizer.py       # Move from rl_position_sizing.py
│   │   ├── gate_optimizer.py       # NEW: RL gate threshold optimizer
│   │   ├── exit_timer.py           # NEW: RL exit timing optimizer
│   │   └── environments/
│   │       ├── trading_env.py      # Existing TradingEnv
│   │       ├── gate_env.py         # NEW: Gate threshold environment
│   │       └── exit_env.py         # NEW: Exit timing environment
│   └── ...
├── core/
│   ├── modular_inference.py        # UNCHANGED
│   └── rl_inference_integration.py # NEW: RL inference layer
└── ...
```

### 5.4 Key Design Principles

1. **Isolation**: RL components should be isolated from the main training pipeline
2. **Optional**: All RL features should be toggleable via configuration
3. **Compatible**: Use subprocess isolation for TF/PyTorch compatibility
4. **Observable**: Log all RL decisions for analysis
5. **Reversible**: Allow fallback to non-RL behavior on failure

---

## Appendix A: File Reference

### Core Training Files

| File | Lines | Purpose |
|------|-------|---------|
| [`src/training/trainers/transformer_trainer.py`](src/training/trainers/transformer_trainer.py) | 2794 | Main Transformer trainer |
| [`src/training/trainers/config.py`](src/training/trainers/config.py) | 245 | Configuration dataclasses |
| [`cli/training.py`](cli/training.py) | 3939+ | Training CLI orchestration |
| [`cli/training_ops.py`](cli/training_ops.py) | 1097 | Training operations |
| [`src/training/modular_trainers.py`](src/training/modular_trainers.py) | - | Modular trainer implementations |

### Existing RL Files

| File | Lines | Purpose |
|------|-------|---------|
| [`rl_position_sizing.py`](rl_position_sizing.py) | 841 | PPO position sizer |
| [`online_retrainer.py`](online_retrainer.py) | 672 | Incremental retraining |

---

## Appendix B: Configuration Reference

```yaml
# From config/config_improved_H1.yaml

buddy:
  train_defaults:
    warm_start: true
    seq_len: 60
    
training:
  auto_train_rl: true
  
transformer:
  use_transformer: true
  d_model: 32
  num_heads: 4
  num_layers: 2
  dff: 64
  dropout: 0.2
  
continual_learning:
  use_ema: true
  ema_decay: 0.999
  use_ewc: true
  ewc_lambda: 100.0
  use_replay_buffer: true
```

---

**Document Status**: Complete  
**Next Steps**: Review with team, proceed to implementation planning
