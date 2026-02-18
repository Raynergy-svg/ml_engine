# Hybrid SFT-RL Training Module Documentation

**Version**: 1.0.0  
**Date**: February 2026  
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Configuration Guide](#3-configuration-guide)
4. [Algorithm Details](#4-algorithm-details)
5. [Usage Guide](#5-usage-guide)
6. [Callbacks Reference](#6-callbacks-reference)
7. [Monitoring and Debugging](#7-monitoring-and-debugging)
8. [API Reference](#8-api-reference)
9. [Testing Guide](#9-testing-guide)
10. [Implementation Files](#10-implementation-files)

---

## 1. Overview

### 1.1 Purpose and Motivation

The Hybrid Supervised-Reinforcement Learning (SFT-RL) training module addresses a critical challenge in ML training pipelines: **memorization and overfitting**. When models are trained purely with supervised learning, they can memorize training patterns rather than learning generalizable decision boundaries.

The hybrid approach introduces stochastic exploration through reinforcement learning, which:

- **Breaks memorization patterns** by sampling actions from the model's probability distribution
- **Encourages exploration** through entropy bonuses
- **Provides implicit regularization** by introducing controlled randomness
- **Maintains supervised learning benefits** while adding RL-based generalization

### 1.2 High-Level Description

The Hybrid SFT-RL module implements a stochastic switch between two loss functions:

| Loss Type | Purpose | Behavior |
|-----------|---------|----------|
| **Supervised (SFT)** | Learn from ground truth labels | Standard cross-entropy loss |
| **Reinforcement Learning (RL)** | Explore and generalize | REINFORCE policy gradient with entropy bonus |

Each training batch has a configurable probability (`rl_prob`) of using the RL loss instead of the supervised loss. This stochastic switching creates an implicit regularization effect.

### 1.3 Key Benefits

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID SFT-RL BENEFITS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ✅ Implicit Regularization                                                  │
│     └─► Stochastic exploration prevents memorization                        │
│                                                                              │
│  ✅ Maintained Performance                                                   │
│     └─► Supervised learning ensures baseline accuracy                       │
│                                                                              │
│  ✅ Controlled Exploration                                                   │
│     └─► Entropy bonus prevents policy collapse                              │
│                                                                              │
│  ✅ Variance Reduction                                                       │
│     └─► EMA baseline stabilizes gradient estimates                          │
│                                                                              │
│  ✅ Curriculum Learning Support                                              │
│     └─► Schedule rl_prob over training epochs                               │
│                                                                              │
│  ✅ Seamless Integration                                                     │
│     └─► Works with existing callbacks, SWA, schedulers                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID SFT-RL TRAINING ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         TrainerConfig                                 │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │ use_hybrid_sft_rl│  │     rl_prob      │  │   entropy_coef   │   │   │
│  │  │   (bool)         │  │    (0.0-1.0)     │  │    (float)       │   │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    HybridSFTLossWrapper                               │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │                    HybridSFTLoss                                │   │   │
│  │  │                                                                 │   │   │
│  │  │   ┌─────────────────────┐    ┌─────────────────────┐          │   │   │
│  │  │   │   SFT Loss Path     │    │    RL Loss Path     │          │   │   │
│  │  │   │                     │    │                     │          │   │   │
│  │  │   │  Cross-Entropy      │    │  REINFORCE          │          │   │   │
│  │  │   │  + Label Smoothing  │    │  + Entropy Bonus    │          │   │   │
│  │  │   │                     │    │  + Baseline (EMA)   │          │   │   │
│  │  │   └──────────┬──────────┘    └──────────┬──────────┘          │   │   │
│  │  │              │                          │                      │   │   │
│  │  │              └──────────┬───────────────┘                      │   │   │
│  │  │                         │                                      │   │   │
│  │  │                         ▼                                      │   │   │
│  │  │              ┌─────────────────────┐                           │   │   │
│  │  │              │  Stochastic Switch  │                           │   │   │
│  │  │              │   (tf.random < p)   │                           │   │   │
│  │  │              └─────────────────────┘                           │   │   │
│  │  │                                                                 │   │   │
│  │  │   ┌─────────────────────────────────────────────────────────┐ │   │   │
│  │  │   │              RLMetricsTracker                            │ │   │   │
│  │  │   │  • reward_history    • entropy_history                   │ │   │   │
│  │  │   │  • baseline_history  • action_distribution               │ │   │   │
│  │  │   └─────────────────────────────────────────────────────────┘ │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Callbacks                                      │   │
│  │                                                                       │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │ RLMetricsCallback│  │RLBaselineScheduler│ │RLEarlyStopping   │   │   │
│  │  │                  │  │                  │  │Callback          │   │   │
│  │  │ • TensorBoard    │  │ • Curriculum     │  │ • Collapse       │   │   │
│  │  │ • Console logs   │  │ • Scheduling     │  │   detection      │   │   │
│  │  │ • History        │  │ • 3 schedule     │  │ • Weight         │   │   │
│  │  │                  │  │   types          │  │   restoration    │   │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Integration with Training Pipeline

The Hybrid SFT-RL loss integrates at the **loss function creation** point in the training pipeline:

```
Training Pipeline Flow:
─────────────────────

1. Feature Preparation
2. Model Building
3. Warm-Start Setup (if applicable)
4. Optimizer Setup
5. Loss Creation ◄── HYBRID SFT-RL INTEGRATION POINT
6. Model Compilation
7. Callback Creation ◄── RL CALLBACKS INTEGRATION POINT
8. model.fit()
9. Post-Training (SWA, EMA, EWC)
```

### 2.3 Relationship to Existing Components

| Component | Relationship | Notes |
|-----------|--------------|-------|
| **SWA (Stochastic Weight Averaging)** | Compatible | SWA averages weights; hybrid loss provides gradients |
| **Learning Rate Scheduler** | Compatible | Scheduler continues through both SFT and RL phases |
| **Warm-Start Recovery** | Compatible | Recovery monitors validation accuracy |
| **EWC (Elastic Weight Consolidation)** | Compatible | EWC penalty added to hybrid loss |
| **EMA (Exponential Moving Average)** | Compatible | EMA tracks model weights independently |
| **Overfit Prevention Callback** | Compatible | Monitors train-val gap regardless of loss type |

---

## 3. Configuration Guide

### 3.1 Complete Configuration Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `use_hybrid_sft_rl` | bool | `False` | - | Enable Hybrid SFT-RL training |
| `rl_prob` | float | `0.5` | 0.0-1.0 | Probability of using RL loss per batch |
| `entropy_coef` | float | `0.01` | 0.0+ | Entropy bonus coefficient for exploration |
| `initial_baseline` | float | `0.5` | 0.0-1.0 | Initial reward baseline for variance reduction |
| `baseline_momentum` | float | `0.9` | 0.0-1.0 | EMA momentum for baseline updates |
| `rl_curriculum_enabled` | bool | `False` | - | Enable curriculum scheduling of rl_prob |
| `rl_curriculum_type` | str | `"linear"` | linear/cosine/step | Schedule type for curriculum |
| `rl_curriculum_warmup_epochs` | int | `10` | 1+ | Epochs to reach target rl_prob |
| `rl_curriculum_initial_prob` | float | `0.0` | 0.0-1.0 | Starting rl_prob for curriculum |
| `rl_curriculum_final_prob` | float | `0.5` | 0.0-1.0 | Target rl_prob after warmup |
| `rl_early_stopping` | bool | `True` | - | Enable RL-based early stopping |
| `rl_min_entropy` | float | `0.005` | 0.0+ | Minimum entropy threshold |
| `rl_patience` | int | `10` | 1+ | Epochs of low entropy before stopping |

### 3.2 Example YAML Configuration

```yaml
# config/config_hybrid_sft_rl.yaml

training:
  # Enable Hybrid SFT-RL Training
  use_hybrid_sft_rl: true
  
  # Core RL Parameters
  rl_prob: 0.3                    # 30% of batches use RL loss
  entropy_coef: 0.01              # Encourage exploration
  initial_baseline: 0.5           # Start with 50% baseline
  baseline_momentum: 0.9          # Slow baseline updates
  
  # Curriculum Learning (optional)
  rl_curriculum_enabled: true
  rl_curriculum_type: "linear"    # Options: linear, cosine, step
  rl_curriculum_warmup_epochs: 20
  rl_curriculum_initial_prob: 0.0 # Start with pure SFT
  rl_curriculum_final_prob: 0.5   # End with 50% RL
  
  # RL Early Stopping
  rl_early_stopping: true
  rl_min_entropy: 0.005           # Collapse threshold
  rl_patience: 10                 # Wait 10 epochs before stopping
```

### 3.3 Example Python Configuration

```python
from src.training.trainers.config import TrainerConfig

# Basic configuration
config = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.3,
    entropy_coef=0.01,
    initial_baseline=0.5,
    baseline_momentum=0.9,
)

# With curriculum learning
config_with_curriculum = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.5,
    entropy_coef=0.01,
    
    # Curriculum settings
    rl_curriculum_enabled=True,
    rl_curriculum_type="cosine",
    rl_curriculum_warmup_epochs=15,
    rl_curriculum_initial_prob=0.0,
    rl_curriculum_final_prob=0.5,
    
    # Early stopping
    rl_early_stopping=True,
    rl_min_entropy=0.005,
    rl_patience=10,
)
```

### 3.4 Parameter Tuning Recommendations

#### Conservative Setup (Stable Training)

```python
config = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.2,              # Low RL proportion
    entropy_coef=0.005,       # Low exploration
    baseline_momentum=0.95,   # Very stable baseline
)
```

#### Aggressive Setup (Maximum Regularization)

```python
config = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.5,              # 50% RL batches
    entropy_coef=0.02,        # Higher exploration
    baseline_momentum=0.9,    # Faster baseline adaptation
)
```

#### Curriculum Learning Setup (Recommended)

```python
config = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.5,              # Final target
    entropy_coef=0.01,
    
    # Start slow, increase RL
    rl_curriculum_enabled=True,
    rl_curriculum_type="linear",
    rl_curriculum_warmup_epochs=20,
    rl_curriculum_initial_prob=0.0,
    rl_curriculum_final_prob=0.5,
)
```

---

## 4. Algorithm Details

### 4.1 REINFORCE Algorithm Implementation

The RL component uses the REINFORCE policy gradient algorithm:

```
REINFORCE Loss Computation:
───────────────────────────

1. Sample action from policy:
   actions ~ Categorical(softmax(logits))

2. Compute reward:
   reward = 1.0 if action == target_label else 0.0

3. Update baseline (EMA):
   baseline = momentum * baseline + (1 - momentum) * mean_reward

4. Compute advantage:
   advantage = reward - baseline

5. Policy gradient loss:
   pg_loss = -log_prob(action) * advantage

6. Entropy bonus:
   entropy = -Σ p(a) * log(p(a))
   
7. Total RL loss:
   loss = mean(pg_loss) - entropy_coef * mean(entropy)
```

### 4.2 Entropy Bonus for Exploration

Entropy prevents policy collapse by encouraging diverse action selection:

```python
# Entropy calculation
probs = tf.nn.softmax(logits, axis=-1)
probs_safe = tf.clip_by_value(probs, 1e-8, 1.0 - 1e-8)
entropy = -tf.reduce_sum(probs_safe * tf.math.log(probs_safe), axis=-1)

# High entropy = uniform distribution = exploration
# Low entropy = peaked distribution = exploitation (risk of collapse)
```

**Entropy Interpretation:**
- Maximum entropy (binary): `ln(2) ≈ 0.693` (uniform predictions)
- Low entropy: `< 0.1` (confident predictions, potential collapse)
- Zero entropy: `0.0` (deterministic, collapsed policy)

### 4.3 Baseline Variance Reduction (EMA)

The baseline reduces variance in gradient estimates:

```python
# Baseline update (detached from computation graph)
new_baseline = (
    baseline_momentum * current_baseline +
    (1 - baseline_momentum) * tf.stop_gradient(mean_reward)
)

# Advantage computation
advantage = reward - baseline

# Effect:
# - If reward > baseline: positive advantage → reinforce action
# - If reward < baseline: negative advantage → discourage action
```

**Why EMA?**
- Running average of rewards provides a dynamic baseline
- Prevents all rewards from looking "good" or "bad"
- Stabilizes training by reducing gradient variance

### 4.4 Stochastic Switching Mechanism

The switch between SFT and RL loss is stochastic:

```python
def call(self, y_true, y_pred):
    # Stochastic switch
    use_rl = tf.random.uniform([]) < self.rl_prob
    
    # Compute both losses
    rl_loss, reward, entropy, actions = self._compute_rl_loss(y_true, y_pred)
    sft_loss = self._compute_sft_loss(y_true, y_pred)
    
    # Select based on random value
    loss = tf.cond(use_rl, lambda: rl_loss, lambda: sft_loss)
    
    return loss
```

**Probability Interpretation:**
- `rl_prob = 0.0`: Pure supervised learning
- `rl_prob = 0.3`: 30% of batches use RL (recommended starting point)
- `rl_prob = 0.5`: Balanced SFT/RL (moderate regularization)
- `rl_prob = 1.0`: Pure RL learning (not recommended alone)

---

## 5. Usage Guide

### 5.1 Enabling the Feature

**Method 1: Configuration-based (Recommended)**

```python
from src.training.trainers.config import TrainerConfig
from src.training.trainers.hybrid_sft_rl_loss import create_hybrid_sft_rl_loss

# Create configuration
config = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.3,
    entropy_coef=0.01,
)

# Create loss function
loss_fn = create_hybrid_sft_rl_loss(config)

# Compile model
model.compile(optimizer='adam', loss=loss_fn)
```

**Method 2: Direct Instantiation**

```python
from src.training.trainers.hybrid_sft_rl_loss import HybridSFTLoss

# Create loss directly
loss_fn = HybridSFTLoss(
    rl_prob=0.3,
    entropy_coef=0.01,
    initial_baseline=0.5,
    baseline_momentum=0.9,
    num_classes=2,
)

# Compile model
model.compile(optimizer='adam', loss=loss_fn)
```

### 5.2 Basic Usage Example

```python
import tensorflow as tf
import numpy as np
from src.training.trainers.config import TrainerConfig
from src.training.trainers.hybrid_sft_rl_loss import HybridSFTLossWrapper

# 1. Prepare data
X_train = np.random.randn(1000, 20).astype(np.float32)
y_train = np.random.randint(0, 2, 1000).astype(np.int32)

# 2. Create configuration
config = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.3,
    entropy_coef=0.01,
)

# 3. Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)  # Logits (no activation)
])

# 4. Create loss and compile
loss_fn = HybridSFTLossWrapper(config)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn,
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6. Get RL metrics summary
summary = loss_fn.get_metrics_summary()
print(f"Mean reward: {summary['mean_reward']:.3f}")
print(f"Mean entropy: {summary['mean_entropy']:.4f}")
print(f"RL ratio: {summary['rl_ratio']:.1%}")
```

### 5.3 Advanced Usage with Curriculum Learning

```python
import tensorflow as tf
from src.training.trainers.config import TrainerConfig
from src.training.trainers.hybrid_sft_rl_loss import HybridSFTLossWrapper
from src.training.trainers.rl_callbacks import (
    RLMetricsCallback,
    RLBaselineScheduler,
    RLEarlyStoppingCallback
)

# Configuration with curriculum learning
config = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.5,  # Final target
    entropy_coef=0.01,
    
    # Curriculum settings
    rl_curriculum_enabled=True,
    rl_curriculum_type="cosine",
    rl_curriculum_warmup_epochs=20,
    rl_curriculum_initial_prob=0.0,
    rl_curriculum_final_prob=0.5,
    
    # Early stopping
    rl_early_stopping=True,
    rl_min_entropy=0.005,
    rl_patience=10,
)

# Create loss
loss_fn = HybridSFTLossWrapper(config)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn,
    metrics=['accuracy']
)

# Create callbacks
callbacks = [
    # RL metrics logging
    RLMetricsCallback(
        loss_fn=loss_fn,
        log_dir="logs/rl_training",
        log_every_n_epochs=1,
    ),
    
    # Curriculum scheduler
    RLBaselineScheduler(
        loss_fn=loss_fn,
        initial_rl_prob=config.rl_curriculum_initial_prob,
        final_rl_prob=config.rl_curriculum_final_prob,
        warmup_epochs=config.rl_curriculum_warmup_epochs,
        schedule_type=config.rl_curriculum_type,
    ),
    
    # Early stopping
    RLEarlyStoppingCallback(
        loss_fn=loss_fn,
        min_entropy=config.rl_min_entropy,
        patience=config.rl_patience,
        restore_best_weights=True,
    ),
]

# Train with callbacks
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

### 5.4 Integration with Existing Training Workflows

**With Transformer Trainer:**

```python
from src.training.trainers.transformer_trainer import TransformerTrainer
from src.training.trainers.config import TrainerConfig

# Configure with Hybrid SFT-RL
config = TrainerConfig(
    use_hybrid_sft_rl=True,
    rl_prob=0.3,
    entropy_coef=0.01,
    
    # Other trainer settings
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    
    # Compatible with existing features
    enable_swa=True,
    use_ewc=True,
    use_ema=True,
)

# Create trainer
trainer = TransformerTrainer(config=config)

# Train as usual - Hybrid SFT-RL integrates automatically
metrics = trainer.train(
    X_train=X_train,
    y_train=y_train,
    x_val=X_val,
    y_val=y_val,
    instrument="EUR_USD",
)
```

---

## 6. Callbacks Reference

### 6.1 RLMetricsCallback

**Purpose:** Track and log RL-specific metrics during training.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_fn` | `HybridSFTLoss` or `HybridSFTLossWrapper` | `None` | Loss function to monitor |
| `log_dir` | `str` | `None` | Directory for TensorBoard logs |
| `log_every_n_epochs` | `int` | `1` | Logging frequency |
| `collapse_threshold` | `float` | `0.01` | Entropy threshold for collapse detection |
| `tensorboard_update_freq` | `int` | `10` | TensorBoard update frequency |

**Usage:**

```python
from src.training.trainers.rl_callbacks import RLMetricsCallback

callback = RLMetricsCallback(
    loss_fn=hybrid_loss,
    log_dir="logs/rl_metrics",
    log_every_n_epochs=1,
    collapse_threshold=0.01,
)

model.fit(..., callbacks=[callback])

# Get metrics history
history = callback.get_metrics_history()
```

**Logged Metrics:**
- `rl/reward_mean` - Mean reward per epoch
- `rl/entropy` - Policy entropy (exploration measure)
- `rl/baseline` - Current reward baseline
- `rl/rl_ratio` - Ratio of RL to SFT batches
- `rl/actions` - Action distribution histogram

### 6.2 RLBaselineScheduler

**Purpose:** Schedule `rl_prob` over training for curriculum learning.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_fn` | `HybridSFTLoss` or `HybridSFTLossWrapper` | `None` | Loss function to update |
| `initial_rl_prob` | `float` | `0.0` | Starting RL probability |
| `final_rl_prob` | `float` | `0.5` | Target RL probability |
| `warmup_epochs` | `int` | `10` | Epochs to reach target |
| `schedule_type` | `str` | `"linear"` | Schedule type: linear/cosine/step |
| `step_epochs` | `List[int]` | `None` | Epochs for step schedule |
| `step_values` | `List[float]` | `None` | Values for step schedule |
| `min_rl_prob` | `float` | `0.0` | Minimum RL probability |

**Schedule Types:**

```
Linear Schedule:
────────────────
rl_prob
  │
1 │                     ╱─────────────
  │                   ╱
0 │─────────────────╱
  └────────────────────────────────────► epoch
  0              warmup             end

Cosine Schedule:
────────────────
rl_prob
  │
1 │                  ╭──────────────
  │               ╱
0 │─────────────╱
  └────────────────────────────────────► epoch
  0              warmup             end

Step Schedule:
────────────────
rl_prob
  │
1 │                          ┌────────
  │                    ┌─────┘
0 │────────┬───────────┘
  └────────────────────────────────────► epoch
  0    step1    step2    step3
```

**Usage:**

```python
from src.training.trainers.rl_callbacks import RLBaselineScheduler

# Linear schedule
scheduler = RLBaselineScheduler(
    loss_fn=hybrid_loss,
    initial_rl_prob=0.0,
    final_rl_prob=0.5,
    warmup_epochs=20,
    schedule_type="linear",
)

# Cosine schedule (smoother transition)
scheduler = RLBaselineScheduler(
    loss_fn=hybrid_loss,
    initial_rl_prob=0.0,
    final_rl_prob=0.5,
    warmup_epochs=20,
    schedule_type="cosine",
)

# Step schedule
scheduler = RLBaselineScheduler(
    loss_fn=hybrid_loss,
    schedule_type="step",
    step_epochs=[10, 20, 30],
    step_values=[0.1, 0.3, 0.5],
    final_rl_prob=0.5,
)

model.fit(..., callbacks=[scheduler])
```

### 6.3 RLEarlyStoppingCallback

**Purpose:** Stop training when RL metrics indicate policy collapse or poor performance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_fn` | `HybridSFTLoss` or `HybridSFTLossWrapper` | `None` | Loss function to monitor |
| `min_entropy` | `float` | `0.005` | Minimum entropy threshold |
| `min_reward` | `float` | `0.3` | Minimum reward threshold |
| `patience` | `int` | `10` | Epochs to wait before stopping |
| `restore_best_weights` | `bool` | `True` | Restore best weights on stop |

**Usage:**

```python
from src.training.trainers.rl_callbacks import RLEarlyStoppingCallback

callback = RLEarlyStoppingCallback(
    loss_fn=hybrid_loss,
    min_entropy=0.005,
    min_reward=0.3,
    patience=10,
    restore_best_weights=True,
)

model.fit(..., callbacks=[callback])
```

**Stopping Conditions:**
1. **Policy Collapse:** Entropy below `min_entropy` for `patience` epochs
2. **Low Reward:** Reward below `min_reward` for `patience * 2` epochs

---

## 7. Monitoring and Debugging

### 7.1 Metrics to Monitor

| Metric | Healthy Range | Warning | Critical | Action |
|--------|---------------|---------|----------|--------|
| `reward_mean` | 0.5-0.8 | < 0.4 | < 0.3 | Check model capacity, increase `rl_prob` |
| `entropy` | 0.3-0.7 | < 0.1 | < 0.01 | Increase `entropy_coef` |
| `baseline` | 0.4-0.6 | - | - | Should track reward mean |
| `rl_ratio` | ~`rl_prob` | - | - | Verify stochastic switch works |

### 7.2 TensorBoard Integration

**Launch TensorBoard:**

```bash
tensorboard --logdir logs/rl_training
```

**Available Charts:**
- `rl/reward_mean` - Reward trend over epochs
- `rl/entropy` - Entropy trend (exploration)
- `rl/baseline` - Baseline adaptation
- `rl/rl_ratio` - Actual RL/SFT ratio
- `rl/actions` - Action distribution histogram

### 7.3 Common Issues and Troubleshooting

#### Issue: Policy Collapse (Low Entropy)

**Symptoms:**
- Entropy drops below 0.01
- All predictions same class
- Action distribution shows 99%+ one class

**Solutions:**
```python
# Increase entropy coefficient
config = TrainerConfig(
    entropy_coef=0.02,  # Increase from 0.01
)

# Or use curriculum learning to delay RL
config = TrainerConfig(
    rl_curriculum_enabled=True,
    rl_curriculum_initial_prob=0.0,
    rl_curriculum_warmup_epochs=30,  # Longer warmup
)
```

#### Issue: Unstable Training (High Variance)

**Symptoms:**
- Loss oscillates wildly
- Reward fluctuates significantly
- Training doesn't converge

**Solutions:**
```python
# Increase baseline momentum for stability
config = TrainerConfig(
    baseline_momentum=0.95,  # Increase from 0.9
)

# Reduce RL probability
config = TrainerConfig(
    rl_prob=0.2,  # Reduce from 0.5
)
```

#### Issue: No Improvement from RL

**Symptoms:**
- Validation accuracy same or worse
- No generalization improvement

**Solutions:**
```python
# Ensure model has sufficient capacity
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),  # Larger layers
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Use curriculum learning
config = TrainerConfig(
    rl_curriculum_enabled=True,
    rl_curriculum_type="cosine",
)
```

### 7.4 Policy Collapse Detection

The [`RLMetricsTracker`](src/training/trainers/hybrid_sft_rl_loss.py:48) provides automatic collapse detection:

```python
from src.training.trainers.hybrid_sft_rl_loss import HybridSFTLoss

loss = HybridSFTLoss(rl_prob=0.3)

# After training...
summary = loss.get_metrics_summary()

if summary['policy_collapsed']:
    print(f"Collapse detected: {summary['collapse_reason']}")
    # Take corrective action
```

**Detection Criteria:**
1. Mean entropy (last 10 batches) < threshold
2. Single action accounts for > 95% of all actions

---

## 8. API Reference

### 8.1 HybridSFTLoss Class

**Location:** [`src/training/trainers/hybrid_sft_rl_loss.py:145`](src/training/trainers/hybrid_sft_rl_loss.py:145)

```python
class HybridSFTLoss(tf.keras.losses.Loss):
    """
    Hybrid Supervised-Reinforcement Learning Loss.
    
    Stochastically switches between Cross-Entropy (Supervised) and 
    REINFORCE Policy Gradient (RL) loss to mitigate memorization.
    """
```

**Constructor:**

```python
def __init__(
    self,
    rl_prob: float = 0.5,           # Probability of using RL loss per batch
    entropy_coef: float = 0.01,     # Entropy bonus coefficient
    initial_baseline: float = 0.5,  # Initial reward baseline
    baseline_momentum: float = 0.9, # EMA momentum for baseline updates
    num_classes: int = 2,           # Number of output classes
    label_smoothing: float = 0.0,   # Label smoothing for supervised loss
    name: str = "hybrid_sft_rl_loss",
    **kwargs
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `call(y_true, y_pred)` | `tf.Tensor` | Compute hybrid loss |
| `get_config()` | `Dict[str, Any]` | Get configuration for serialization |
| `from_config(config)` | `HybridSFTLoss` | Create from configuration |
| `get_metrics_summary()` | `Dict[str, Any]` | Get RL metrics summary |
| `reset_metrics()` | `None` | Reset metrics tracker |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `reward_baseline` | `float` | Current reward baseline value |
| `metrics_tracker` | `RLMetricsTracker` | Metrics tracking instance |

### 8.2 HybridSFTLossWrapper Class

**Location:** [`src/training/trainers/hybrid_sft_rl_loss.py:431`](src/training/trainers/hybrid_sft_rl_loss.py:431)

```python
class HybridSFTLossWrapper(tf.keras.losses.Loss):
    """
    Wrapper for HybridSFTLoss that provides compatibility with existing training code.
    
    Works with both:
    - Standard Keras training (model.fit)
    - Custom training loops
    """
```

**Constructor:**

```python
def __init__(
    self,
    config: Any,                        # TrainerConfig instance
    name: str = "hybrid_sft_rl_loss_wrapper",
    **kwargs
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `call(y_true, y_pred)` | `tf.Tensor` | Compute loss (delegates to hybrid_loss) |
| `get_config()` | `Dict[str, Any]` | Get configuration |
| `from_config(config)` | `HybridSFTLossWrapper` | Create from configuration |
| `get_metrics_summary()` | `Dict[str, Any]` | Get RL metrics summary |
| `reset_metrics()` | `None` | Reset metrics tracker |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `hybrid_loss` | `HybridSFTLoss` | Underlying loss instance |

### 8.3 RLMetricsTracker Dataclass

**Location:** [`src/training/trainers/hybrid_sft_rl_loss.py:48`](src/training/trainers/hybrid_sft_rl_loss.py:48)

```python
@dataclass
class RLMetricsTracker:
    """
    Tracks RL-specific metrics during training.
    """
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reward_history` | `List[float]` | `[]` | Mean rewards per batch |
| `baseline_history` | `List[float]` | `[]` | Baseline values over time |
| `entropy_history` | `List[float]` | `[]` | Entropy values per batch |
| `action_distribution` | `Dict[int, int]` | `{}` | Counts of actions taken |
| `rl_steps` | `int` | `0` | Batches with RL loss |
| `sft_steps` | `int` | `0` | Batches with SFT loss |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `update(reward_mean, baseline, entropy, actions, is_rl)` | `None` | Update tracked metrics |
| `get_summary()` | `Dict[str, Any]` | Get summary statistics |
| `check_policy_collapse(threshold)` | `Tuple[bool, str]` | Check for policy collapse |

---

## 9. Testing Guide

### 9.1 Running Tests

**Run all Hybrid SFT-RL tests:**

```bash
pytest tests/test_hybrid_sft_rl.py -v
```

**Run with coverage:**

```bash
pytest tests/test_hybrid_sft_rl.py -v --cov=src/training/trainers/hybrid_sft_rl_loss --cov=src/training/trainers/rl_callbacks
```

**Run specific test class:**

```bash
pytest tests/test_hybrid_sft_rl.py::TestHybridSFTLoss -v
pytest tests/test_hybrid_sft_rl.py::TestRLMetricsCallback -v
pytest tests/test_hybrid_sft_rl.py::TestHybridSFTRLIntegration -v
```

### 9.2 Test Coverage Overview

| Module | Coverage | Test Classes |
|--------|----------|--------------|
| `hybrid_sft_rl_loss.py` | 95%+ | `TestRLMetricsTracker`, `TestHybridSFTLoss`, `TestHybridSFTLossWrapper`, `TestCreateHybridSFTRLLoss` |
| `rl_callbacks.py` | 90%+ | `TestRLMetricsCallback`, `TestRLBaselineScheduler`, `TestRLEarlyStoppingCallback` |
| Integration | 85%+ | `TestHybridSFTRLIntegration`, `TestParameterized`, `TestEdgeCases` |

### 9.3 Key Test Scenarios

**Unit Tests:**

| Scenario | Test Method | Purpose |
|----------|-------------|---------|
| Default initialization | `test_initialization_defaults` | Verify default parameters |
| Custom initialization | `test_initialization_custom` | Verify custom parameters |
| Stochastic switching | `test_stochastic_switching` | Verify RL/SFT switching |
| Baseline update | `test_baseline_update` | Verify EMA baseline |
| Entropy calculation | `test_entropy_calculation` | Verify entropy bonus |
| Gradient flow | `test_gradient_flow` | Verify gradients computed |
| NaN handling | `test_nan_handling_probs` | Verify numerical stability |

**Integration Tests:**

| Scenario | Test Method | Purpose |
|----------|-------------|---------|
| Config to loss | `test_config_to_loss_instantiation` | Verify config integration |
| Training with loss | `test_training_with_hybrid_loss` | Verify training completes |
| Training with callbacks | `test_training_with_callbacks` | Verify callback integration |
| End-to-end workflow | `test_end_to_end_workflow` | Verify complete pipeline |

**Parameterized Tests:**

| Parameter | Values | Test Method |
|-----------|--------|-------------|
| `rl_prob` | 0.0, 0.25, 0.5, 0.75, 1.0 | `test_rl_prob_values` |
| `entropy_coef` | 0.0, 0.001, 0.01, 0.1 | `test_entropy_coef_values` |
| `num_classes` | 2, 3, 5, 10 | `test_num_classes` |
| `batch_size` | 1, 2, 8, 32, 64, 128 | `test_batch_sizes` |
| `baseline_momentum` | 0.5, 0.9, 0.95, 0.99 | `test_baseline_momentum_values` |
| `schedule_type` | linear, cosine, step | `test_schedule_types` |

---

## 10. Implementation Files

### 10.1 Files Created/Modified

| File | Lines | Purpose |
|------|-------|---------|
| [`src/training/trainers/hybrid_sft_rl_loss.py`](src/training/trainers/hybrid_sft_rl_loss.py) | 547 | Main Hybrid SFT-RL loss implementation |
| [`src/training/trainers/rl_callbacks.py`](src/training/trainers/rl_callbacks.py) | 653 | RL training callbacks |
| [`src/training/trainers/config.py`](src/training/trainers/config.py) | 271 | Configuration (SFT-RL section added) |
| [`tests/test_hybrid_sft_rl.py`](tests/test_hybrid_sft_rl.py) | 1201 | Comprehensive test suite |

### 10.2 Public API Exports

**`hybrid_sft_rl_loss.py`:**

```python
__all__ = [
    "HybridSFTLoss",
    "HybridSFTLossWrapper",
    "RLMetricsTracker",
    "create_hybrid_sft_rl_loss",
]
```

**`rl_callbacks.py`:**

```python
__all__ = [
    "RLMetricsCallback",
    "RLBaselineScheduler",
    "RLEarlyStoppingCallback",
]
```

### 10.3 Dependencies

| Dependency | Purpose | Required |
|------------|---------|----------|
| `tensorflow >= 2.10` | Deep learning framework | Yes |
| `numpy >= 1.20` | Numerical operations | Yes |
| `rich >= 10.0` | Console formatting (optional) | No |

---

## References

### Academic References

1. Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256.

2. Mnih, V., et al. (2016). "Asynchronous methods for deep reinforcement learning." *International Conference on Machine Learning*.

3. Schulman, J., et al. (2017). "Proximal policy optimization algorithms." *arXiv preprint arXiv:1707.06347*.

### Related Documentation

- [RL Integration Architecture Analysis](RL_INTEGRATION_ARCHITECTURE_ANALYSIS.md)
- [RL Integration Strategy](RL_INTEGRATION_STRATEGY.md)
- [Training Troubleshooting Guide](TRAINING_TROUBLESHOOTING.md)
- [Production ML Pipeline Guide](PRODUCTION_ML_PIPELINE_GUIDE.md)

---

**Document Status**: Complete  
**Last Updated**: February 2026  
**Maintainer**: ML Engineering Team
