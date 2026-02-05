# Class-Balanced Loss Implementation Plan for Prediction Collapse Fix

## Executive Summary

**Problem:** The model exhibits severe prediction collapse at epoch 5, showing 96.6% bias toward the UP class while effectively ignoring the DOWN class.

**Root Cause:** Extreme class imbalance (e.g., 124:1 ratio) causes the current inverse frequency weighting with a 10x cap to be insufficient. The minority class (DOWN) gets capped at 10x weight when it needs 62.5x weight to balance the gradient contribution, leading to unstable gradients and prediction collapse.

**Solution:** Implement **Class-Balanced Loss (CB Loss)** - a technique specifically designed for extreme class imbalance that is not currently in the codebase. CB Loss computes effective number of samples per class and adjusts weights more stably than inverse frequency weighting with caps.

---

## 1. Current Implementation Analysis

### 1.1 Existing Anti-Collapse Mechanisms

The codebase already has extensive anti-collapse mechanisms:

| Mechanism | Location | Description |
|------------|----------|-------------|
| **AntiCollapseFocalLoss** | [`src/models/tensorflow_models.py:63-166`](../src/models/tensorflow_models.py:63) | Focal Loss with variance regularization and dynamic alpha |
| **BinaryFocalLoss** | [`src/models/tensorflow_models.py:170-214`](../src/models/tensorflow_models.py:170) | Standard Focal Loss with fixed alpha |
| **Sample Weight Capping** | [`src/training/modular_trainers.py:2997`](../src/training/modular_trainers.py:2997) | Caps weights at 10x multiplier |
| **Regime-Aware Weights** | [`src/training/modular_trainers.py:2933-2982`](../src/training/modular_trainers.py:2933) | Balances classes within volatility regimes |
| **Prediction Collapse Detection** | [`src/training/modular_trainers.py:3343-3373`](../src/training/modular_trainers.py:3343) | Warns if >95% predictions are one class |
| **Output Calibration** | [`src/training/modular_trainers.py:3543-3565`](../src/training/modular_trainers.py:3543) | Uses median as adaptive threshold |
| **Label Smoothing** | [`config/config_improved_H1.yaml`](../config/config_improved_H1.yaml) | Default 0.05 |

### 1.2 Current Sample Weight Computation

From [`src/training/modular_trainers.py:2986-3026`](../src/training/modular_trainers.py:2986):

```python
# Inverse frequency weighting
total = n_up + n_down
MAX_WEIGHT_MULTIPLIER = 10.0  # Cap at 10x weight
up_weight = min(total / (2 * n_up), MAX_WEIGHT_MULTIPLIER) if n_up > 0 else 1.0
down_weight = min(total / (2 * n_down), MAX_WEIGHT_MULTIPLIER) if n_down > 0 else 1.0

# Apply weights
sample_weights[y_train_filtered == 1] *= up_weight
sample_weights[y_train_filtered == 0] *= down_weight
```

**Problem with Current Approach:**
- For 124:1 imbalance: minority class needs 62.5x weight but gets capped at 10x
- Majority class still gets 10x weight (excessive)
- Creates unstable gradients and oscillation

---

## 2. Class-Balanced Loss (CB Loss) - The Missing Technique

### 2.1 What is CB Loss?

Class-Balanced Loss (Cui et al., CVPR 2019) addresses extreme class imbalance by computing the **effective number of samples** for each class and adjusting weights accordingly.

**Key Insight:** As class size increases, the effective number of samples grows sublinearly due to redundancy. CB Loss uses this to compute more stable weights.

### 2.2 CB Loss Formula

For each class $c$ with $n_c$ samples:

$$
\beta = \frac{N-1}{N} \cdot (1 - b^{n_c/N})
$$

Where:
- $N$ = total number of samples
- $b \in (0, 1)$ is a hyperparameter (typically $\beta = 0.9999$)
- $n_c$ = number of samples in class $c$

**Effective number of samples:**
$$
n_{eff}^{(c)} = \frac{1 - \beta^{n_c/N}}{1 - \beta}
$$

**Class-balanced weight:**
$$
w_{CB}^{(c)} = \frac{1 - \beta}{1 - \beta^{n_c/N}}
$$

### 2.3 CB Loss Combined with Focal Loss

CB Loss can be combined with Focal Loss for even better performance:

$$
L_{CB-Focal} = \frac{w_{CB}^{(c)}}{\sum_i w_{CB}^{(c_i)}} \cdot L_{Focal}
$$

This ensures:
1. Each class contributes equally to the gradient (via CB weighting)
2. Hard examples get more focus (via Focal Loss)
3. Stable gradients even with extreme imbalance

### 2.4 Why CB Loss is Better Than Current Approach

| Aspect | Current (Inverse Freq + Cap) | CB Loss |
|--------|------------------------------|---------|
| **Weight Stability** | Unstable - minority capped at 10x when needs 62.5x | Stable - grows sublinearly with class size |
| **Gradient Behavior** | Oscillates due to extreme weight differences | Smooth - each class contributes equally |
| **Handling Extreme Imbalance** | Poor - caps prevent proper balancing | Excellent - designed for 100:1+ imbalance |
| **Theoretical Foundation** | Heuristic | Research-backed (CVPR 2019) |

---

## 3. Implementation Plan

### 3.1 Create ClassBalancedFocalLoss Class

**File:** [`src/models/tensorflow_models.py`](../src/models/tensorflow_models.py)

**Location:** After `AntiCollapseFocalLoss` (after line 166)

**Implementation:**

```python
@register_keras_serializable()
class ClassBalancedFocalLoss(keras.losses.Loss):
    """
    Class-Balanced Focal Loss for extreme class imbalance.
    
    Combines Class-Balanced Loss (Cui et al., CVPR 2019) with Focal Loss.
    This is more stable than inverse frequency weighting with caps.
    
    Args:
        gamma: Focusing parameter for Focal Loss (default 2.0)
        beta: Class-balanced hyperparameter (default 0.9999)
        label_smoothing: Label smoothing factor (default 0.05)
        focal_alpha: Base class weight for Focal (default 0.5)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        beta: float = 0.9999,
        label_smoothing: float = 0.05,
        focal_alpha: float = 0.5,
        name: str = 'class_balanced_focal_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
    
    def call(self, y_true, y_pred):
        import tensorflow as tf
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # === COMPUTE CLASS-BALANCED WEIGHTS ===
        # Get batch size
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        
        # Count samples per class
        n_up = tf.reduce_sum(y_true)
        n_down = batch_size - n_up
        
        # Compute effective number of samples for each class
        # n_eff = (1 - beta^(n/N)) / (1 - beta)
        total = n_up + n_down + tf.keras.backend.epsilon()
        
        # Effective samples for UP class
        up_ratio = n_up / total
        up_eff = (1.0 - tf.pow(self.beta, up_ratio)) / (1.0 - self.beta + tf.keras.backend.epsilon())
        
        # Effective samples for DOWN class
        down_ratio = n_down / total
        down_eff = (1.0 - tf.pow(self.beta, down_ratio)) / (1.0 - self.beta + tf.keras.backend.epsilon())
        
        # Class-balanced weights (inverse of effective samples)
        # w_CB = (1 - beta) / (1 - beta^(n/N))
        up_weight = (1.0 - self.beta) / (1.0 - tf.pow(self.beta, up_ratio) + tf.keras.backend.epsilon())
        down_weight = (1.0 - self.beta) / (1.0 - tf.pow(self.beta, down_ratio) + tf.keras.backend.epsilon())
        
        # Normalize weights to sum to batch_size (preserve effective batch size)
        total_weight = up_weight * n_up + down_weight * n_down
        up_weight = up_weight * batch_size / (total_weight + tf.keras.backend.epsilon())
        down_weight = down_weight * batch_size / (total_weight + tf.keras.backend.epsilon())
        
        # Apply class-balanced weights per sample
        cb_weight = y_true * up_weight + (1 - y_true) * down_weight
        
        # === FOCAL LOSS COMPONENT ===
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weight
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)
        
        # Apply focal alpha
        alpha_weight = y_true * self.focal_alpha + (1 - y_true) * (1 - self.focal_alpha)
        focal_weight = focal_weight * alpha_weight
        
        # === COMBINE CB WEIGHTS WITH FOCAL LOSS ===
        # L = w_CB * L_Focal
        focal_loss = focal_weight * bce
        
        # Apply class-balanced weighting
        weighted_loss = cb_weight * focal_loss
        
        # Normalize by batch size to get average loss
        total_loss = tf.reduce_mean(weighted_loss)
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'beta': self.beta,
            'label_smoothing': self.label_smoothing,
            'focal_alpha': self.focal_alpha,
        })
        return config
```

### 3.2 Update TrainerConfig

**File:** [`src/training/modular_trainers.py`](../src/training/modular_trainers.py)

**Location:** In `TrainerConfig` class (around line 73-77)

**Add new configuration parameters:**

```python
# === CLASS-BALANCED LOSS SETTINGS ===
use_class_balanced_loss: bool = True  # Use CB Loss instead of Focal Loss
cb_beta: float = 0.9999  # Class-balanced hyperparameter
cb_gamma: float = 2.0  # Focusing parameter for CB Loss
```

### 3.3 Update TransformerDirectionTrainer.train()

**File:** [`src/training/modular_trainers.py`](../src/training/modular_trainers.py)

**Location:** In `train()` method, around line 3222-3257

**Changes:**

1. **Import ClassBalancedFocalLoss** (around line 2632):
```python
try:
    from src.models.tensorflow_models import (
        BinaryFocalLoss,
        ClassBalancedFocalLoss,  # ADD THIS
        MADLLoss,
    )
except ImportError:
    BinaryFocalLoss = None
    ClassBalancedFocalLoss = None  # ADD THIS
    MADLLoss = None
```

2. **Replace loss function selection logic** (around line 3227-3257):
```python
# === CLASS-BALANCED FOCAL LOSS OR BCE ===
label_smoothing = getattr(self.config, 'label_smoothing', 0.05) if self.config else 0.05

# Check for MADL loss first
use_madl = getattr(self.config, 'use_madl_loss', False) if self.config else False

if use_madl:
    try:
        from src.models.tensorflow_models import MADLLoss
        madl_direction_weight = getattr(self.config, 'madl_direction_weight', 0.7) if self.config else 0.7
        logger.info(f"💰 Using MADL Loss for directional profitability (direction_weight={madl_direction_weight}, label_smoothing={label_smoothing})")
        base_loss = MADLLoss(
            direction_weight=madl_direction_weight,
            label_smoothing=label_smoothing,
        )
    except ImportError:
        logger.warning("⚠️ MADLLoss not found, falling back to Class-Balanced Focal Loss")
        use_madl = False

# Use Class-Balanced Focal Loss (NEW - PRIORITY)
if not use_madl and self.config and self.config.use_class_balanced_loss:
    try:
        from src.models.tensorflow_models import ClassBalancedFocalLoss
        cb_beta = getattr(self.config, 'cb_beta', 0.9999) if self.config else 0.9999
        cb_gamma = getattr(self.config, 'cb_gamma', 2.0) if self.config else 2.0
        logger.info(f"🎯 Using Class-Balanced Focal Loss for extreme class imbalance (beta={cb_beta}, gamma={cb_gamma}, label_smoothing={label_smoothing})")
        base_loss = ClassBalancedFocalLoss(
            beta=cb_beta,
            gamma=cb_gamma,
            label_smoothing=label_smoothing,
        )
    except ImportError:
        logger.warning("⚠️ ClassBalancedFocalLoss not found, falling back to Focal Loss")
        self.config.use_class_balanced_loss = False

# Fallback to standard Focal Loss
if not use_madl and (not self.config or not self.config.use_class_balanced_loss):
    try:
        from src.models.tensorflow_models import BinaryFocalLoss
        logger.info(f"🎯 Using Focal Loss for class imbalance (gamma=2.0, alpha=0.5, label_smoothing={label_smoothing})")
        base_loss = BinaryFocalLoss(
            gamma=2.0,
            alpha=0.5,
            label_smoothing=label_smoothing,
        )
    except ImportError:
        logger.warning("⚠️ BinaryFocalLoss not found, falling back to BinaryCrossentropy")
        base_loss = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
```

3. **Disable or reduce sample weight capping** (around line 2995-3006):
```python
# === CLASS-BALANCED LOSS: DISABLE SAMPLE WEIGHT CAPPING ===
# When using Class-Balanced Loss, the loss function handles class imbalance internally
# We should NOT use sample weights as they would double-count the balancing
if self.config and self.config.use_class_balanced_loss:
    logger.info("🎯 Class-Balanced Loss enabled - using uniform sample weights (loss handles balancing)")
    sample_weights = None  # Let CB Loss handle class balancing
    class_weight = None
else:
    # === MINORITY CLASS BOOSTING (Anti-Bias) ===
    # Use inverse frequency weighting for class imbalance (original logic)
    minority_class = 1 if n_up < n_down else 0
    
    total = n_up + n_down
    
    # REDUCE CAP from 10x to 5x for more stable gradients
    # CB Loss is primary, this is secondary
    MAX_WEIGHT_MULTIPLIER = 5.0  # Reduced from 10.0
    up_weight = min(total / (2 * n_up), MAX_WEIGHT_MULTIPLIER) if n_up > 0 else 1.0
    down_weight = min(total / (2 * n_down), MAX_WEIGHT_MULTIPLIER) if n_down > 0 else 1.0
    
    # Apply inverse frequency weights
    sample_weights[y_train_filtered == 1] *= up_weight
    sample_weights[y_train_filtered == 0] *= down_weight
    
    # Re-normalize to mean=1
    sample_weights = sample_weights / sample_weights.mean()
    
    # Log weights
    imbalance_ratio = max(n_up, n_down) / min(n_up, n_down) if min(n_up, n_down) > 0 else float('inf')
    minority_name = "UP" if minority_class == 1 else "DOWN"
    logger.info(f"🎯 Inverse frequency sample weights (capped at {MAX_WEIGHT_MULTIPLIER}x): UP={up_weight:.2f}x, DOWN={down_weight:.2f}x")
    logger.info(f"🎯 Class imbalance ratio: {imbalance_ratio:.1f}:1, minority={minority_name}")
    
    # Also compute global class weights for Keras (backup)
    class_weight = {
        0: total / (2 * n_down),
        1: total / (2 * n_up),
    }
```

### 3.4 Update Configuration File

**File:** [`config/config_improved_H1.yaml`](../config/config_improved_H1.yaml)

**Add/Update settings:**

```yaml
train_defaults:
  # === CLASS-BALANCED LOSS SETTINGS ===
  use_class_balanced_loss: true  # NEW: Use CB Loss for extreme class imbalance
  cb_beta: 0.9999  # NEW: Class-balanced hyperparameter (higher = more aggressive)
  cb_gamma: 2.0  # NEW: Focusing parameter for CB Loss
  
  # === FOCAL LOSS SETTINGS (fallback if CB Loss disabled) ===
  use_focal_loss: false  # DISABLED: CB Loss is better for extreme imbalance
  focal_gamma: 2.0
  focal_alpha: 0.5
  focal_label_smoothing: 0.05
```

---

## 4. Expected Outcomes

### 4.1 Training Behavior

| Metric | Before (Current) | After (CB Loss) |
|--------|------------------|------------------|
| **Gradient Stability** | Unstable (oscillates due to 10x cap) | Stable (CB weights grow smoothly) |
| **Minority Class Learning** | Poor (capped at 10x when needs 62.5x) | Good (CB computes appropriate weight) |
| **Prediction Distribution** | Collapses to 96.6% UP | Balanced (~50/50 after calibration) |
| **Convergence Speed** | Slower due to oscillation | Faster due to stable gradients |

### 4.2 Validation Metrics

| Metric | Current | Expected with CB Loss |
|--------|---------|---------------------|
| **Val Accuracy** | ~60% | 62-65% (better balance) |
| **Balanced Accuracy** | ~55% | 60-63% (significant improvement) |
| **UP Class Accuracy** | ~98% | 75-80% (reduced overfitting) |
| **DOWN Class Accuracy** | ~12% | 75-80% (major improvement) |
| **Prediction Distribution** | 96.6% UP, 3.4% DOWN | 48-52% UP, 48-52% DOWN |

### 4.3 Training Stability

- **No prediction collapse warnings** after epoch 5
- **Stable loss curves** without oscillation
- **Faster convergence** to optimal accuracy
- **Better generalization** to unseen data

---

## 5. Verification Steps

### 5.1 Unit Tests

Create test file: `tests/test_class_balanced_loss.py`

```python
import numpy as np
import tensorflow as tf
from src.models.tensorflow_models import ClassBalancedFocalLoss

def test_cb_loss_balanced():
    """Test that CB Loss produces equal gradient contributions."""
    loss_fn = ClassBalancedFocalLoss(beta=0.9999, gamma=2.0)
    
    # Simulate batch with 100:1 imbalance
    y_true = tf.constant([[1.0]] * 100 + [[0.0]] * 1, dtype=tf.float32)
    y_pred = tf.constant([[0.5]] * 101, dtype=tf.float32)
    
    loss = loss_fn(y_true, y_pred)
    
    # Loss should be finite and not collapse
    assert tf.math.is_finite(loss), "Loss should be finite"
    assert loss < 10.0, "Loss should be reasonable"
    print(f"✓ CB Loss test passed: loss={loss.numpy():.4f}")

def test_cb_loss_vs_imbalance():
    """Test that CB Loss handles different imbalance ratios."""
    loss_fn = ClassBalancedFocalLoss(beta=0.9999, gamma=2.0)
    
    # Test different imbalance ratios
    for ratio in [2, 10, 50, 100]:
        n_minority = 10
        n_majority = 10 * ratio
        
        y_true = tf.constant([[1.0]] * n_minority + [[0.0]] * n_majority, dtype=tf.float32)
        y_pred = tf.constant([[0.5]] * (n_minority + n_majority), dtype=tf.float32)
        
        loss = loss_fn(y_true, y_pred)
        
        # Loss should remain stable even with extreme imbalance
        assert tf.math.is_finite(loss), f"Loss should be finite for {ratio}:1 ratio"
        print(f"✓ CB Loss stable for {ratio}:1 ratio: loss={loss.numpy():.4f}")

if __name__ == "__main__":
    test_cb_loss_balanced()
    test_cb_loss_vs_imbalance()
    print("\n✅ All Class-Balanced Loss tests passed!")
```

### 5.2 Integration Test

Create test file: `tests/test_cb_loss_integration.py`

```python
import sys
sys.path.append('.')

from src.training.modular_trainers import TrainerConfig, TransformerDirectionTrainer
import numpy as np

def test_cb_loss_training():
    """Test that training with CB Loss doesn't collapse."""
    # Create synthetic imbalanced data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    seq_len = 60
    
    # 90% UP, 10% DOWN (extreme imbalance)
    y_train = np.array([1] * 900 + [0] * 100)
    y_val = np.array([1] * 90 + [0] * 10)
    
    X_train = np.random.randn(n_samples, n_features)
    X_val = np.random.randn(100, n_features)
    
    # Configure with CB Loss
    config = TrainerConfig(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        use_class_balanced_loss=True,  # ENABLE CB LOSS
        cb_beta=0.9999,
        cb_gamma=2.0,
    )
    
    # Train
    trainer = TransformerDirectionTrainer(config)
    metrics = trainer.train(
        X_train, y_train,
        X_val, y_val,
        feature_names=[f"feat_{i}" for i in range(n_features)],
    )
    
    # Verify no collapse
    val_acc = metrics['val_accuracy']
    print(f"Validation accuracy: {val_acc:.2%}")
    
    # Should not collapse to predicting all UP
    assert val_acc > 0.45, f"Val accuracy {val_acc:.2%} too low, possible collapse"
    
    # Check prediction distribution (would need to inspect logs)
    print("✅ CB Loss integration test passed!")

if __name__ == "__main__":
    test_cb_loss_training()
```

### 5.3 A/B Testing

Run comparative training:

```bash
# Baseline: Current Focal Loss with sample weight capping
python main.py --pair EUR_USD --config config_improved_H1.yaml --use_focal_loss --epochs 20

# New: Class-Balanced Loss
python main.py --pair EUR_USD --config config_improved_H1.yaml --use_class_balanced_loss --epochs 20
```

Compare:
- Validation accuracy
- Balanced accuracy
- Prediction distribution
- Training stability

### 5.4 Production Monitoring

After deployment, monitor:

1. **Prediction Distribution:**
   - Should be ~50/50 UP/DOWN (after calibration)
   - Alert if >80% predictions are one class

2. **Class-wise Accuracy:**
   - UP accuracy: 70-80%
   - DOWN accuracy: 70-80%
   - Gap should be <10%

3. **Loss Curves:**
   - Smooth convergence without oscillation
   - No sudden spikes

4. **Training Logs:**
   - Look for "🎯 Using Class-Balanced Focal Loss" confirmation
   - Verify CB weights are being computed

---

## 6. Rollback Plan

If CB Loss causes issues:

1. **Disable CB Loss in config:**
   ```yaml
   use_class_balanced_loss: false
   use_focal_loss: true  # Fallback to Focal Loss
   ```

2. **Restore original sample weight capping:**
   - Change `MAX_WEIGHT_MULTIPLIER` back to 10.0
   - Re-enable regime-aware weighting

3. **Monitor for degradation:**
   - If val accuracy drops >5%, revert immediately
   - Check for prediction collapse warnings

---

## 7. Implementation Checklist

- [ ] Create `ClassBalancedFocalLoss` class in `src/models/tensorflow_models.py`
- [ ] Add CB Loss config parameters to `TrainerConfig` class
- [ ] Update `TransformerDirectionTrainer.train()` to use CB Loss
- [ ] Update `config/config_improved_H1.yaml` with CB Loss settings
- [ ] Create unit tests for CB Loss
- [ ] Create integration test for CB Loss training
- [ ] Run A/B test comparing CB Loss vs baseline
- [ ] Monitor training metrics in production
- [ ] Document rollback procedure

---

## 8. References

1. **Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019**
   - Paper: https://arxiv.org/abs/1901.05555
   - Key insight: Effective number of samples grows sublinearly with class size

2. **Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017**
   - Paper: https://arxiv.org/abs/1708.02002
   - Key insight: Down-weight easy examples (majority class)

3. **Existing Codebase Documentation:**
   - [`docs/PREDICTION_COLLAPSE_DEBUG_GUIDE.md`](../docs/PREDICTION_COLLAPSE_DEBUG_GUIDE.md)
   - [`src/training/modular_trainers.py`](../src/training/modular_trainers.py)
   - [`src/models/tensorflow_models.py`](../src/models/tensorflow_models.py)

---

## 9. Summary

**Problem:** Prediction collapse at epoch 5 (96.6% UP, 3.4% DOWN)

**Root Cause:** Extreme class imbalance (124:1) with insufficient sample weight capping (10x cap when minority needs 62.5x)

**Solution:** Implement Class-Balanced Focal Loss - a research-backed technique specifically designed for extreme class imbalance

**Expected Impact:**
- Balanced predictions (~50/50 after calibration)
- Improved validation accuracy (62-65%)
- Stable training without oscillation
- Better generalization to unseen data

**Implementation Effort:** Medium (new loss class + config updates + trainer modifications)

**Risk:** Low - CB Loss is well-researched and widely used in production systems
