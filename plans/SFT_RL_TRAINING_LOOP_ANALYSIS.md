# SFT-RL Training Loop Architecture Analysis

**Document Purpose**: Analyze the existing training loop architecture to identify optimal integration points for Hybrid Supervised-Reinforcement Learning (SFT-RL) paradigm.

**Date**: February 2026  
**Status**: Architecture Analysis Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Training Loop Structure](#2-current-training-loop-structure)
3. [Loss Computation Flow](#3-loss-computation-flow)
4. [Identified Integration Points](#4-identified-integration-points)
5. [Existing Safety Mechanisms](#5-existing-safety-mechanisms)
6. [Existing RL Components](#6-existing-rl-components)
7. [Hybrid Loss Integration Strategy](#7-hybrid-loss-integration-strategy)
8. [Recommendations](#8-recommendations)

---

## 1. Executive Summary

### Key Findings

| Finding | Location | Significance |
|---------|----------|--------------|
| Main training loop | [`transformer_trainer.py:train()`](../src/training/trainers/transformer_trainer.py:2034) | Primary integration point |
| Loss function creation | [`transformer_trainer.py:_create_loss_function()`](../src/training/trainers/transformer_trainer.py:682) | Hybrid loss insertion point |
| Model compilation | [`transformer_trainer.py:_compile_model_with_loss()`](../src/training/trainers/transformer_trainer.py:1422) | Loss wrapping with EWC |
| Existing RL utilities | [`src/rl/`](../src/rl/) | Reusable components available |
| Callback system | [`transformer_trainer.py:_create_training_callbacks()`](../src/training/trainers/transformer_trainer.py:1494) | RL callback integration |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CURRENT TRAINING LOOP FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  train() [Line 2034]                                                        │
│      │                                                                       │
│      ├─► _initialize_training_state() [Line 292]                            │
│      │       └─► TrainingLineage, DriftDetector                             │
│      │                                                                       │
│      ├─► _scale_features() [Line 310]                                       │
│      │       └─► StandardScaler fit/transform                               │
│      │                                                                       │
│      ├─► _handle_feature_preparation() [Line 2164]                          │
│      │       └─► Feature selection, warm-start compatibility                │
│      │                                                                       │
│      ├─► _prepare_sequences_and_filter() [Line 1264]                        │
│      │       └─► Sequence creation, label filtering                         │
│      │                                                                       │
│      ├─► _build_model() [Line 134]                                          │
│      │       └─► Transformer architecture construction                      │
│      │                                                                       │
│      ├─► _setup_warm_start() [Line 2207]                                    │
│      │       └─► Weight loading, layer freezing                             │
│      │                                                                       │
│      ├─► _setup_optimizer_with_warmup() [Line 1323]                         │
│      │       └─► Adam with warmup schedule                                  │
│      │                                                                       │
│      ├─► _create_loss_function() [Line 682] ◄── HYBRID LOSS INSERTION      │
│      │       └─► Loss priority chain selection                              │
│      │                                                                       │
│      ├─► _compile_model_with_loss() [Line 1422]                             │
│      │       └─► EWC wrapping, optimizer binding                            │
│      │                                                                       │
│      ├─► _create_training_callbacks() [Line 1494] ◄── RL CALLBACK HOOK     │
│      │       └─► EarlyStopping, OverfitPrevention, Collapse detection       │
│      │                                                                       │
│      ├─► model.fit() [Line 2124-2141] ◄── MAIN TRAINING LOOP               │
│      │       └─► TensorFlow Keras fit with callbacks                        │
│      │                                                                       │
│      ├─► _handle_warm_start_recovery() [Line 1843]                          │
│      │       └─► Post-training weight restoration                           │
│      │                                                                       │
│      ├─► _update_ewc_and_ema() [Line 1868]                                  │
│      │       └─► Fisher computation, EMA update                             │
│      │                                                                       │
│      └─► _compute_final_metrics() [Line 1897]                               │
│              └─► Validation, calibration, drift detection                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current Training Loop Structure

### 2.1 Main Entry Point

**File**: [`src/training/trainers/transformer_trainer.py`](../src/training/trainers/transformer_trainer.py)  
**Method**: `train()`  
**Lines**: 2034-2162

```python
def train(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[list] = None,
    w_train: Optional[np.ndarray] = None,
    w_val: Optional[np.ndarray] = None,
    warm_start_path: Optional[str] = None,
    instrument: str = "UNKNOWN",
    data_range: str = "",
    skip_scaling: bool = False,
    fold_id: Optional[int] = None,
    **kwargs,
) -> Dict[str, float]:
```

### 2.2 Training Phases

| Phase | Method | Lines | Purpose |
|-------|--------|-------|---------|
| 1. Initialization | `_initialize_training_state()` | 292-308 | Create lineage, drift detector |
| 2. Feature Scaling | `_scale_features()` | 310-331 | StandardScaler fit/transform |
| 3. Feature Selection | `_handle_feature_preparation()` | 2164-2205 | RF/F-test selection |
| 4. Sequence Creation | `_prepare_sequences_and_filter()` | 1264-1299 | Create sequences, filter labels |
| 5. Model Building | `_build_model()` | 134-227 | Construct Transformer |
| 6. Warm-Start Setup | `_setup_warm_start()` | 2207-2226 | Load weights, freeze layers |
| 7. Optimizer Setup | `_setup_optimizer_with_warmup()` | 1323-1351 | Adam with warmup schedule |
| 8. Loss Creation | `_create_loss_function()` | 682-709 | Select loss function |
| 9. Model Compilation | `_compile_model_with_loss()` | 1422-1447 | Bind optimizer, loss, metrics |
| 10. Callback Creation | `_create_training_callbacks()` | 1494-1557 | Create all callbacks |
| 11. Training | `model.fit()` | 2124-2141 | Main training loop |
| 12. Post-Training | Various | 2143-2161 | Recovery, EWC, EMA, metrics |

### 2.3 Base Trainer Interface

**File**: [`src/training/trainers/base.py`](../src/training/trainers/base.py)  
**Lines**: 22-70

```python
class BaseTrainer(ABC):
    """Abstract base class for all modular trainers."""

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.model = None
        self.is_trained = False
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def train(...) -> Dict[str, float]:
        """Train the model and return metrics."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make predictions."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
```

---

## 3. Loss Computation Flow

### 3.1 Loss Function Priority Chain

**File**: [`src/training/trainers/transformer_trainer.py`](../src/training/trainers/transformer_trainer.py)  
**Method**: `_create_loss_function()`  
**Lines**: 682-709

```python
def _create_loss_function(
    self, auto_variance_weight: float, label_smoothing: float
) -> Any:
    """Create appropriate loss function based on config."""
    
    # Priority list of loss function attempts
    loss_attempts = [
        (getattr(self.config, "use_madl_loss", False),
         lambda: self._try_madl_loss(label_smoothing)),
        (self.config and getattr(self.config, "use_hybrid_cb_anticollapse", True),
         lambda: self._try_hybrid_cb_loss(auto_variance_weight, label_smoothing)),
        (self.config and getattr(self.config, "use_anti_collapse_loss", True),
         lambda: self._try_anti_collapse_loss(auto_variance_weight)),
        (self.config and self.config.use_class_balanced_loss,
         lambda: self._try_cb_focal_loss(label_smoothing)),
    ]

    for should_try, try_func in loss_attempts:
        if should_try:
            loss = try_func()
            if loss is not None:
                return loss

    # Fallback to standard Focal Loss or BCE
    return self._get_fallback_loss(label_smoothing)
```

### 3.2 Loss Types Available

| Priority | Loss Type | Method | Lines | Purpose |
|----------|-----------|--------|-------|---------|
| 1 | MADL Loss | `_try_madl_loss()` | 711-725 | Directional profitability |
| 2 | Hybrid CB Anti-Collapse | `_try_hybrid_cb_loss()` | 727-749 | Class-balanced + variance reg |
| 3 | Anti-Collapse Focal | `_try_anti_collapse_loss()` | 751-772 | Variance regularization |
| 4 | Class-Balanced Focal | `_try_cb_focal_loss()` | 774-787 | Extreme imbalance handling |
| 5 | Binary Focal | `_get_fallback_loss()` | 789-800 | Hard example mining |
| 6 | Binary Cross-Entropy | `_get_fallback_loss()` | 789-800 | Final fallback |

### 3.3 Model Compilation with EWC

**File**: [`src/training/trainers/transformer_trainer.py`](../src/training/trainers/transformer_trainer.py)  
**Method**: `_compile_model_with_loss()`  
**Lines**: 1422-1447

```python
def _compile_model_with_loss(
    self,
    optimizer: Any,
    base_loss: Any,
    instrument: str,
) -> None:
    """Compile model with appropriate loss function."""
    use_ewc_loss = (
        self._is_warm_start
        and self._use_ewc
        and self.ewc is not None
        and self.ewc.fisher_diagonal is not None
        and getattr(self, "_loaded_model_instrument", None) != instrument
    )

    if use_ewc_loss:
        logger.info(
            f"🧠 EWC loss enabled (cross-pair): λ={self.ewc.ewc_lambda}, "
            f"protecting {self.ewc._n_tasks} prior task(s)"
        )
        ewc_loss = create_ewc_loss(base_loss, self.ewc.penalty, ewc_weight=1.0)
        self.model.compile(optimizer=optimizer, loss=ewc_loss, metrics=["accuracy"])
    else:
        self.model.compile(optimizer=optimizer, loss=base_loss, metrics=["accuracy"])
```

---

## 4. Identified Integration Points

### 4.1 Primary Integration Points

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SFT-RL INTEGRATION POINTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT 1: Loss Function Creation (RECOMMENDED)                              │
│  ─────────────────────────────────────────────                              │
│  File: src/training/trainers/transformer_trainer.py                         │
│  Method: _create_loss_function()                                            │
│  Lines: 682-709                                                             │
│                                                                              │
│  Integration:                                                                │
│    - Add hybrid loss option to priority chain                               │
│    - Create HybridSFTLoss class combining CE + RL loss                      │
│    - Add epoch-based switching logic                                        │
│                                                                              │
│  Risk Level: MEDIUM (affects loss computation)                              │
│  Dependencies: None (isolated change)                                       │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT 2: Callback System                                                   │
│  ───────────────────────────                                                │
│  File: src/training/trainers/transformer_trainer.py                         │
│  Method: _create_training_callbacks()                                       │
│  Lines: 1494-1557                                                           │
│                                                                              │
│  Integration:                                                                │
│    - Add RLSwitchCallback for epoch-based loss switching                    │
│    - Add RLRewardBaselineCallback for baseline updates                      │
│    - Add RLActionSamplingCallback for action collection                     │
│                                                                              │
│  Risk Level: LOW (callback pattern established)                             │
│  Dependencies: POINT 1                                                      │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT 3: Training Loop (model.fit)                                         │
│  ─────────────────────────────────────                                      │
│  File: src/training/trainers/transformer_trainer.py                         │
│  Lines: 2124-2141                                                           │
│                                                                              │
│  Integration:                                                                │
│    - Use tf.data.Dataset with RL augmentation                               │
│    - Pass additional RL context to fit()                                    │
│                                                                              │
│  Risk Level: MEDIUM (core training loop)                                    │
│  Dependencies: POINT 1, POINT 2                                             │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POINT 4: Configuration                                                     │
│  ────────────────────────                                                   │
│  File: src/training/trainers/config.py                                      │
│  Lines: 1-251                                                               │
│                                                                              │
│  Integration:                                                                │
│    - Add SFT-RL configuration section                                       │
│    - Define hybrid loss parameters                                          │
│    - Define RL training parameters                                          │
│                                                                              │
│  Risk Level: LOW (configuration only)                                       │
│  Dependencies: None                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Detailed Integration Point Analysis

#### POINT 1: Loss Function Creation (Lines 682-709)

**Current Implementation**:
```python
# Loss priority chain
loss_attempts = [
    (use_madl_loss, lambda: self._try_madl_loss(label_smoothing)),
    (use_hybrid_cb_anticollapse, lambda: self._try_hybrid_cb_loss(...)),
    (use_anti_collapse_loss, lambda: self._try_anti_collapse_loss(...)),
    (use_class_balanced_loss, lambda: self._try_cb_focal_loss(...)),
]
```

**Proposed Integration**:
```python
# Add SFT-RL hybrid loss to priority chain
loss_attempts = [
    (use_sft_rl_hybrid, lambda: self._try_sft_rl_loss(
        auto_variance_weight, label_smoothing, rl_config
    )),
    (use_madl_loss, lambda: self._try_madl_loss(label_smoothing)),
    # ... rest of chain
]
```

**Required Changes**:
1. Add `use_sft_rl_hybrid` config option
2. Create `_try_sft_rl_loss()` method
3. Create `HybridSFTRLLoss` class in `src/models/tensorflow_models.py`

#### POINT 2: Callback System (Lines 1494-1557)

**Current Callbacks**:
- `RichEpochCallback` / `QuietProgressCallback`
- `EarlyStopping`
- `OverfitPreventionCallback`
- `PredictionCollapseCallback`
- `ProactiveCollapsePreventionCallback`
- `GradualUnfreezeCallback`
- `EMAUpdateCallback`
- `EWCTrainingCallback`

**Proposed Additions**:
```python
# SFT-RL Callbacks
class RLSwitchCallback(tf.keras.callbacks.Callback):
    """Switch between CE and RL loss at specified epoch."""
    
    def __init__(self, switch_epoch: int, loss_manager: HybridLossManager):
        self.switch_epoch = switch_epoch
        self.loss_manager = loss_manager
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.switch_epoch:
            self.loss_manager.enable_rl_loss()

class RLRewardBaselineCallback(tf.keras.callbacks.Callback):
    """Update reward baseline using exponential moving average."""
    
    def __init__(self, baseline_decay: float = 0.99):
        self.baseline_decay = baseline_decay
        self.reward_baseline = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        # Update baseline with rewards from epoch
        pass

class RLActionSamplingCallback(tf.keras.callbacks.Callback):
    """Collect action samples using torch.multinomial equivalent."""
    
    def on_train_batch_end(self, batch, logs=None):
        # Sample actions from policy distribution
        pass
```

---

## 5. Existing Safety Mechanisms

### 5.1 Warm-Start Recovery

**File**: [`src/training/trainers/transformer_trainer.py`](../src/training/trainers/transformer_trainer.py)  
**Method**: `_handle_warm_start_recovery()`  
**Lines**: 1843-1866

```python
def _handle_warm_start_recovery(
    self, x_val_filtered: np.ndarray, y_val_filtered: np.ndarray
) -> None:
    """Restore original weights if training degraded."""
    if not (self._is_warm_start and self._warm_start_weights is not None
            and self._warm_start_val_acc > 0):
        return

    current_val_pred = (self.model.predict(x_val_filtered, verbose=0) > 0.5).astype(float)
    current_val_acc = np.mean(current_val_pred.flatten() == y_val_filtered)

    if current_val_acc < self._warm_start_val_acc - 0.01:
        # Recovery triggered - restore original weights
        self.model.set_weights(self._warm_start_weights)
```

**Preservation Strategy**: Ensure RL loss does not trigger false recovery by:
1. Using separate validation metrics for RL evaluation
2. Adding RL-specific recovery threshold

### 5.2 SWA Update Logic

**File**: [`src/training/trainers/callbacks.py`](../src/training/trainers/callbacks.py)  
**Class**: `OverfitPreventionCallback`

**Configuration** (from `config.py`):
```python
# SWA settings
enable_swa: bool = True               # Average weights in final 25%
swa_start_fraction: float = 0.75      # Start SWA at 75% of training
swa_lr_factor: float = 0.5            # SWA constant LR = initial_lr * factor
```

**Preservation Strategy**: SWA should only apply to CE phase weights, not RL phase.

### 5.3 Scheduler Step Timing

**File**: [`src/training/trainers/transformer_trainer.py`](../src/training/trainers/transformer_trainer.py)  
**Method**: `_setup_optimizer_with_warmup()`  
**Lines**: 1323-1351

```python
def _setup_optimizer_with_warmup(
    self, effective_lr: float, x_train_filtered: np.ndarray
) -> Any:
    """Setup optimizer with warmup learning rate schedule."""
    warmup_epochs = getattr(self.config, "warmup_epochs", 5)
    steps_per_epoch = max(1, len(x_train_filtered) // self.config.batch_size)
    total_steps = self.config.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    lr_schedule = WarmupCosineDecaySchedule(
        initial_learning_rate=effective_lr * 0.1,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        min_learning_rate=1e-6,
        warmup_target=effective_lr,
    )
```

**Preservation Strategy**: 
1. Continue LR schedule through RL phase
2. Optionally freeze LR during RL fine-tuning

### 5.4 Collapse Detection

**File**: [`src/training/trainers/transformer_trainer.py`](../src/training/trainers/transformer_trainer.py)  
**Method**: `_make_prediction_collapse_callback()`  
**Lines**: 1586-1680

**Mechanism**:
- Monitors prediction diversity (UP/DOWN ratio)
- Triggers recovery if predictions collapse to single class
- Applies LR reduction or weight perturbation

**Preservation Strategy**: RL loss should maintain prediction diversity. Add RL-specific collapse detection.

---

## 6. Existing RL Components

### 6.1 Available RL Utilities

**Directory**: [`src/rl/`](../src/rl/)

| File | Purpose | Reusable for SFT-RL |
|------|---------|---------------------|
| [`__init__.py`](../src/rl/__init__.py) | Lazy imports | ✅ Pattern for SB3 imports |
| [`utils.py`](../src/rl/utils.py) | Utility functions | ✅ `detect_regime()`, `calculate_sharpe()` |
| [`gate_threshold_env.py`](../src/rl/gate_threshold_env.py) | SAC gate optimizer | ✅ Environment pattern |
| [`optimal_exit_env.py`](../src/rl/optimal_exit_env.py) | PPO exit timer | ✅ Environment pattern |
| [`callbacks.py`](../src/rl/callbacks.py) | SB3 callbacks | ⚠️ SB3-specific |
| [`curriculum.py`](../src/rl/curriculum.py) | Curriculum learning | ✅ Progressive difficulty |
| [`rl_worker.py`](../src/rl/rl_worker.py) | Subprocess worker | ✅ TF/PyTorch isolation |

### 6.2 Reusable Components

#### Regime Detection

**File**: [`src/rl/utils.py`](../src/rl/utils.py)  
**Lines**: 19-61

```python
def detect_regime(
    features: np.ndarray,
    adx_idx: int = 0,
    volatility_idx: int = 1,
    momentum_idx: int = 2,
) -> Tuple[np.ndarray, str]:
    """
    Detect market regime from features.

    Regimes:
        - TREND (0): ADX > 25, clear directional momentum
        - CHOP (1): ADX < 20, low volatility, no clear direction
        - MEAN_REVERT (2): High volatility, momentum exhaustion
    """
```

**Reuse Potential**: Use for RL state construction and reward shaping.

#### Sharpe/Sortino Calculations

**File**: [`src/rl/utils.py`](../src/rl/utils.py)  
**Lines**: 64-151

```python
def calculate_sharpe(returns: List[float], ...) -> float:
    """Calculate Sharpe ratio from returns."""

def calculate_sortino(returns: List[float], ...) -> float:
    """Calculate Sortino ratio (downside-risk adjusted)."""

def calculate_calmar(returns: List[float], max_drawdown: float, ...) -> float:
    """Calculate Calmar ratio (return / max drawdown)."""
```

**Reuse Potential**: Use for RL reward signal computation.

#### Feature Normalization

**File**: [`src/rl/utils.py`](../src/rl/utils.py)  
**Lines**: 154-185

```python
def normalize_features_for_rl(
    features: np.ndarray,
    scaler: Optional[object] = None,
) -> np.ndarray:
    """Normalize features for RL environment observation space."""
```

**Reuse Potential**: Use for preparing observations in RL loss.

#### Trade Simulation

**File**: [`src/rl/utils.py`](../src/rl/utils.py)  
**Lines**: 265-331

```python
def simulate_trade(
    entry_price: float,
    direction: int,
    prices: np.ndarray,
    stop_loss_pct: float = 0.01,
    take_profit_pct: float = 0.02,
    max_bars: int = 24,
) -> Dict[str, float]:
    """Simulate a trade outcome."""
```

**Reuse Potential**: Use for computing RL rewards during training.

### 6.3 Environment Patterns

**File**: [`src/rl/gate_threshold_env.py`](../src/rl/gate_threshold_env.py)  
**Class**: `GateThresholdEnv`  
**Lines**: 172-551

**Key Patterns**:
1. Lazy gymnasium/SB3 imports
2. Observation space: regime + win_rate + drawdown
3. Action space: continuous threshold adjustments
4. Reward shaping: P/L + Sharpe + drawdown penalty

**Reuse Potential**: Pattern for creating RL loss environment.

---

## 7. Hybrid Loss Integration Strategy

### 7.1 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID SFT-RL LOSS ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    HybridSFTLoss                                      │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  Phase 1: Supervised Fine-Tuning (SFT)                               │   │
│  │  ─────────────────────────────────────                               │   │
│  │  Loss = CE_Loss + Variance_Regularization                            │   │
│  │                                                                       │   │
│  │  Phase 2: Reinforcement Learning (RL)                                │   │
│  │  ─────────────────────────────────                                   │   │
│  │  Loss = -log_prob(action) * (reward - baseline)                      │   │
│  │                                                                       │   │
│  │  Switching Logic:                                                     │   │
│  │  - switch_epoch: config.sft_rl_switch_epoch (default: 50)           │   │
│  │  - gradual_transition: config.sft_rl_gradual (default: True)        │   │
│  │  - mix_weight: linear interpolation during transition                │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Reward Computation                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  reward = base_pnl + sharpe_component - drawdown_penalty            │   │
│  │                                                                       │   │
│  │  Where:                                                               │   │
│  │  - base_pnl: Simulated trade P/L from direction prediction          │   │
│  │  - sharpe_component: Rolling Sharpe ratio bonus                      │   │
│  │  - drawdown_penalty: Progressive penalty for drawdown               │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Baseline Update                                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  baseline = decay * baseline + (1 - decay) * current_reward         │   │
│  │                                                                       │   │
│  │  Where decay = config.rl_baseline_decay (default: 0.99)             │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Action Sampling                                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  # TensorFlow equivalent of torch.multinomial                        │   │
│  │  logits = model_output[:, action_dim]                                │   │
│  │  probs = tf.nn.softmax(logits)                                       │   │
│  │  action = tf.random.categorical(tf.math.log(probs), 1)              │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Configuration Schema

**File**: [`src/training/trainers/config.py`](../src/training/trainers/config.py)

```python
@dataclass
class TrainerConfig:
    # ... existing fields ...
    
    # === SFT-RL HYBRID LOSS SETTINGS ===
    use_sft_rl_hybrid: bool = False  # Enable SFT-RL hybrid training
    sft_rl_switch_epoch: int = 50    # Epoch to switch from CE to RL loss
    sft_rl_gradual: bool = True      # Gradual transition between losses
    sft_rl_mix_epochs: int = 10      # Epochs for gradual transition
    
    # RL Loss Parameters
    rl_baseline_decay: float = 0.99  # Reward baseline EMA decay
    rl_reward_scale: float = 1.0     # Reward scaling factor
    rl_entropy_weight: float = 0.01  # Entropy bonus for exploration
    
    # Reward Shaping
    rl_sharpe_weight: float = 0.1    # Sharpe component weight
    rl_drawdown_penalty: float = 0.5 # Drawdown penalty weight
    rl_win_rate_bonus: float = 0.2   # Win rate bonus weight
```

### 7.3 Implementation Approach

**Phase 1: Configuration** (LOW RISK)
- Add SFT-RL config fields to `TrainerConfig`
- Add YAML configuration section

**Phase 2: Loss Class** (MEDIUM RISK)
- Create `HybridSFTLoss` class in `src/models/tensorflow_models.py`
- Implement CE phase loss with variance regularization
- Implement RL phase loss with reward baseline
- Implement switching logic

**Phase 3: Callback Integration** (LOW RISK)
- Create `RLSwitchCallback` for epoch-based switching
- Create `RLRewardBaselineCallback` for baseline updates
- Create `RLActionSamplingCallback` for action collection

**Phase 4: Training Loop Integration** (MEDIUM RISK)
- Modify `_create_loss_function()` to include hybrid loss
- Add callbacks to `_create_training_callbacks()`
- Ensure SWA/scheduler compatibility

---

## 8. Recommendations

### 8.1 Implementation Priority

| Priority | Task | Risk | Effort | Dependencies |
|----------|------|------|--------|--------------|
| 1 | Add SFT-RL configuration | LOW | Small | None |
| 2 | Create HybridSFTLoss class | MEDIUM | Medium | 1 |
| 3 | Create RL callbacks | LOW | Small | 1 |
| 4 | Integrate into loss chain | MEDIUM | Small | 2, 3 |
| 5 | Add callbacks to training | LOW | Small | 3 |
| 6 | Testing and validation | MEDIUM | Medium | 1-5 |

### 8.2 Protected Components

**DO NOT MODIFY** during SFT-RL integration:

| Component | File | Reason |
|-----------|------|--------|
| Transformer architecture | `transformer_trainer.py:_build_model()` | Model structure stability |
| EMA implementation | `callbacks.py:EMACallback` | Continual learning |
| EWC penalty | `callbacks.py:EWCPenalty` | Catastrophic forgetting prevention |
| Replay buffer | `callbacks.py:ReplayBuffer` | Experience replay |
| Training lineage | `callbacks.py:TrainingLineage` | Audit trail |
| Checkpoint format | `transformer_trainer.py:save()` | Model persistence |

### 8.3 Testing Strategy

1. **Unit Tests**:
   - Test `HybridSFTLoss` in isolation
   - Test callback switching logic
   - Test reward baseline updates

2. **Integration Tests**:
   - Test with synthetic data
   - Test warm-start compatibility
   - Test SWA/scheduler preservation

3. **Regression Tests**:
   - Verify existing metrics maintained
   - Verify no change to CE-only training
   - Verify backward compatibility

### 8.4 Rollback Plan

1. **Feature Flag**: `use_sft_rl_hybrid: false` by default
2. **Fallback**: If RL loss fails, fall back to CE loss
3. **Recovery**: Warm-start recovery should work with RL-trained models

---

## Appendix: File Reference

### Primary Training Files

| File | Lines | Purpose |
|------|-------|---------|
| [`src/training/trainers/transformer_trainer.py`](../src/training/trainers/transformer_trainer.py) | 2806 | Main Transformer trainer |
| [`src/training/trainers/config.py`](../src/training/trainers/config.py) | 251 | Configuration dataclasses |
| [`src/training/trainers/base.py`](../src/training/trainers/base.py) | 70 | Abstract base trainer |
| [`src/training/trainers/callbacks.py`](../src/training/trainers/callbacks.py) | ~2700 | Training callbacks |

### RL Utility Files

| File | Lines | Purpose |
|------|-------|---------|
| [`src/rl/utils.py`](../src/rl/utils.py) | 331 | RL utility functions |
| [`src/rl/gate_threshold_env.py`](../src/rl/gate_threshold_env.py) | 912 | SAC gate optimizer |
| [`src/rl/optimal_exit_env.py`](../src/rl/optimal_exit_env.py) | ~800 | PPO exit timer |
| [`src/rl/callbacks.py`](../src/rl/callbacks.py) | ~250 | SB3 callbacks |

### Documentation

| File | Purpose |
|------|---------|
| [`docs/RL_INTEGRATION_ARCHITECTURE_ANALYSIS.md`](../docs/RL_INTEGRATION_ARCHITECTURE_ANALYSIS.md) | Existing RL architecture analysis |
| [`docs/RL_INTEGRATION_STRATEGY.md`](../docs/RL_INTEGRATION_STRATEGY.md) | RL integration strategy |

---

**Document Status**: Complete  
**Next Steps**: Review with team, create implementation task breakdown
