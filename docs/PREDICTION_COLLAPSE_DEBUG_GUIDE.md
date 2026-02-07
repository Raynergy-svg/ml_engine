# Prediction Collapse Debugging Guide
## Transformer Model Predicting 96.8% DOWN / 3.2% UP at Epoch 5

**Date:** 2026-01-27  
**Issue:** Severe output bias where model collapses to predicting mostly one class

---

## Executive Summary

Your codebase already implements **extensive anti-collapse mechanisms**, yet the model is still experiencing prediction collapse. This suggests one or more of the following issues:

1. **Class imbalance in training data** (most likely cause)
2. **Learning rate too high** for the data distribution
3. **Incorrect loss function configuration**
4. **Data leakage or target misalignment**
5. **Improper warm-start configuration**

---

## Root Cause Analysis

### 1. Loss Function Implementation ✅ (GOOD)

**File:** `src/models/tensorflow_models.py` (lines 63-166)

Your [`AntiCollapseFocalLoss`](src/models/tensorflow_models.py:63-166) is well-designed:

```python
# Key features:
- Variance regularization: Penalizes when batch predictions have LOW variance
- Dynamic alpha adjustment: Boosts weight for ignored class
- Gradient floor: Ensures minority class always contributes gradients
- Label smoothing: Prevents overconfident predictions
```

**Status:** ✅ Implementation is correct. This is NOT the issue.

---

### 2. Transformer Architecture ✅ (GOOD)

**File:** `src/training/modular_trainers.py` (lines 2561-2698)

Your [`TransformerDirectionTrainer._build_model`](src/training/modular_trainers.py:2561-2653) has proper anti-collapse design:

```python
# Anti-collapse features:
- Zero bias initialization: bias_initializer=keras.initializers.Zeros()  # Starts at 0.5
- Input noise: GaussianNoise(0.03)  # Prevents overfitting
- Spatial dropout: SpatialDropout1D(0.15)  # Drops entire features
- L2 regularization: l2_reg=0.005  # Prevents weight explosion
- Tanh activation: Outputs [-1,1] instead of [0,∞] for balanced inputs
```

**Status:** ✅ Architecture is well-designed. This is NOT the issue.

---

### 3. Optimizer Configuration ⚠️ (POTENTIAL ISSUE)

**File:** `config/config_improved_H1.yaml` (lines 126-140)

```yaml
optimizer:
  type: adamw
  learning_rate: 0.0001  # Very low LR - GOOD
  weight_decay: 0.05
  betas: [0.9, 0.999]
  clipnorm: 0.5
```

**Status:** ⚠️ Learning rate looks appropriate, but **AdamW with weight_decay=0.05 may be too aggressive** for early training.

---

### 4. Class Imbalance Handling ⚠️ (LIKELY ROOT CAUSE)

**File:** `src/training/modular_trainers.py` (lines 2874-2962)

Your code implements **inverse frequency weighting**:

```python
# Lines 2924-2952:
total = n_up + n_down
up_weight = total / (2 * n_up)      # Inverse frequency
down_weight = total / (2 * n_down)
```

**Problem:** If your training data has **extreme class imbalance** (e.g., 124:1 ratio), the inverse frequency weights will be:
- UP weight: 0.508 (minority class gets 50.8x weight)
- DOWN weight: 62.5 (majority class gets 62.5x weight)

This creates **unstable gradients** and can cause the model to oscillate or collapse.

---

### 5. Data Pipeline Issues ⚠️ (POTENTIAL ISSUE)

**File:** `src/data/data_processing.py` (lines 507-616)

Your [`prepare_sequences`](src/data/data_processing.py:507-616) function:

```python
# Line 560: train_end_idx = int(n_rows * train_split_fraction)
# Line 562: feature_scaler.fit(feature_data[:train_end_idx])  # FIT ONLY ON TRAINING DATA
```

**Status:** ✅ Data leakage prevention is correctly implemented.

**However**, check if your **target labels are correct**:
- Are labels binary (0=down, 1=up)?
- Is there label noise or misalignment?
- Are you using the correct `direction_threshold`?

---

## Actionable Debugging Steps

### Priority 1: Verify Class Distribution (CRITICAL)

**Action:** Check actual class distribution in your training data.

```python
# Add this to your training script:
import numpy as np

# After creating sequences, check distribution:
print(f"Training UP: {(y_train == 1).sum()} / len(y_train) * 100:.1f}%")
print(f"Training DOWN: {(y_train == 0).sum()} / len(y_train) * 100:.1f}%")
print(f"Imbalance ratio: {max(n_up, n_down) / min(n_up, n_down):.1f}:1")
```

**Expected:** If imbalance > 3:1, you need to address it.

---

### Priority 2: Reduce Sample Weight Aggressiveness

**File:** `src/training/modular_trainers.py` (line 2931)

**Current:**
```python
sample_weights[y_train_filtered == 1] *= up_weight  # 50.8x for minority
sample_weights[y_train_filtered == 0] *= down_weight  # 62.5x for majority
```

**Fix:** Cap the maximum weight to prevent instability:

```python
# Replace lines 2933-2935 with:
MAX_WEIGHT_MULTIPLIER = 10.0  # Cap at 10x weight
up_weight = min(total / (2 * n_up), MAX_WEIGHT_MULTIPLIER)
down_weight = min(total / (2 * n_down), MAX_WEIGHT_MULTIPLIER)
```

---

### Priority 3: Reduce Learning Rate for Warm-Start

**File:** `config/config_improved_H1.yaml` (line 130)

**Current:** `learning_rate: 0.0001`

**For warm-start:** The code reduces LR by 100x (line 3093):
```python
effective_lr = self.config.learning_rate * self.config.warm_start_lr_factor  # 0.01
# effective_lr = 0.0001 * 0.01 = 0.000001
```

**Problem:** LR of 0.000001 may be **too low** to escape local minima.

**Fix:** Increase warm-start LR factor:

```yaml
# In config_improved_H1.yaml, change line 122:
warm_start_lr_factor: 0.1  # Changed from 0.01 to 0.1
# New effective LR: 0.0001 * 0.1 = 0.00001
```

---

### Priority 4: Reduce Weight Decay for AdamW

**File:** `config/config_improved_H1.yaml` (line 133)

**Current:** `weight_decay: 0.05`

**Problem:** For AdamW, weight_decay of 0.05 is **very aggressive** and can cause gradient explosion or collapse.

**Fix:** Reduce weight decay:

```yaml
# In config_improved_H1.yaml, change line 133:
weight_decay: 0.001  # Changed from 0.05 to 0.001
```

---

### Priority 5: Verify Label Generation

**File:** `config/config_improved_H1.yaml` (lines 64-67)

```yaml
direction_lookahead: 24
direction_threshold: 0.0015  # Min 0.15% move
```

**Check:** Are your labels generated correctly?

```python
# Verify label distribution:
# After creating labels, check:
print(f"Label threshold: {direction_threshold}")
print(f"Label distribution: {np.unique(y_train, return_counts=True)}")
```

**If labels are heavily skewed (>70% one class):** Increase `direction_threshold` to get more balanced labels.

---

### Priority 6: Enable Prediction Collapse Detection Logging

**File:** `src/training/modular_trainers.py` (lines 3273-3305)

Your [`PredictionCollapseCallback`](src/training/modular_trainers.py:3273-3305) is already implemented:

```python
class PredictionCollapseCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.check_every != 0:
            return
        
        preds = self.model.predict(self.X_val, verbose=0)
        pred_classes = (preds > 0.5).astype(float).flatten()
        
        pred_up_pct = pred_classes.mean() * 100
        pred_down_pct = 100 - pred_up_pct
        
        # Check for collapse (>95% same prediction)
        if pred_up_pct > 95 or pred_down_pct > 95:
            if not self.collapse_warned:
                dominant = "UP" if pred_up_pct > 95 else "DOWN"
                logger.warning(f"⚠️ PREDICTION COLLAPSE at epoch {epoch+1}: "
                                      f"Model predicts {pred_up_pct:.1f}% UP, {pred_down_pct:.1f}% DOWN "
                                      f"(all {dominant})")
```

**Status:** ✅ Detection is in place. Check your training logs for collapse warnings.

---

## Configuration Fixes Summary

### Apply these changes to `config/config_improved_H1.yaml`:

```yaml
# 1. Reduce weight decay (line 133)
weight_decay: 0.001  # Was 0.05

# 2. Increase warm-start LR factor (line 122)
warm_start_lr_factor: 0.1  # Was 0.01

# 3. Increase direction threshold for more balanced labels (line 67)
direction_threshold: 0.003  # Was 0.0015 (0.3% instead of 0.15%)
```

### Apply these changes to `src/training/modular_trainers.py`:

```python
# Add after line 2930 (before weight application):
# Cap sample weights to prevent instability
MAX_WEIGHT_MULTIPLIER = 10.0
up_weight = min(total / (2 * n_up), MAX_WEIGHT_MULTIPLIER)
down_weight = min(total / (2 * n_down), MAX_WEIGHT_MULTIPLIER)
```

---

## Diagnostic Commands

### Run these to identify the root cause:

```bash
# 1. Check class distribution in training data
python3 -c "
import pandas as pd
import numpy as np

# Load your training data
df = pd.read_parquet('trained_data/data/your_data_file.parquet')
print('Class distribution:')
print(df['direction'].value_counts(normalize=True))
print(f'UP %: {(df[\"direction\"] == 1).mean() * 100:.1f}%')
"

# 2. Monitor training in real-time
# Check logs for:
# - "PREDICTION COLLAPSE" warnings
# - Class distribution logs
# - Validation accuracy trends

# 3. Check if warm-start is causing issues
# Look for: "WARM-START: Loading weights" in logs
# Check if val_acc degrades immediately after warm-start
```

---

## Expected Outcomes After Fixes

### If Class Imbalance is the Root Cause:
- **Before:** 96.8% DOWN, 3.2% UP (extreme bias)
- **After:** ~50-55% DOWN, ~45-50% UP (balanced predictions)

### If Learning Rate is the Root Cause:
- **Before:** Model collapses to one class by epoch 5
- **After:** Model maintains balanced predictions throughout training

### If Data Pipeline is the Root Cause:
- **Before:** Labels are misaligned or contain noise
- **After:** Clean labels with proper distribution

---

## Verification Checklist

After applying fixes, verify:

- [ ] Class distribution is balanced (40-60% each class)
- [ ] No "PREDICTION COLLAPSE" warnings in logs
- [ ] Validation accuracy improves (target: >55%)
- [ ] Balanced accuracy (UP and DOWN) are both >45%
- [ ] Model variance increases (predictions spread across [0.3, 0.7])

---

## Additional Resources

### Academic Papers on Prediction Collapse:
1. **"On the difficulty of training recurrent neural networks"** - Pascanu et al. (1994)
   - Discusses vanishing/exploding gradients in RNNs
   
2. **"Focal Loss for Dense Object Detection"** - Lin et al. (2017)
   - Your implementation follows this paper's approach
   
3. **"Label Smoothing for Improved Calibration"** - Müller et al. (2019)
   - Prevents overconfident predictions
   
4. **"Understanding the Difficulty of Training Deep Feedforward Neural Networks"** - Glorot & Bengio (2010)
   - Discusses initialization strategies

### GitHub Issues:
1. **TensorFlow issue #41097** - "Binary cross entropy loss is not decreasing"
2. **Keras issue #12692** - "Model predicts all zeros or all ones"
3. **PyTorch issue #38898** - "Binary classification with class imbalance"

---

## Summary

Your codebase has **excellent anti-collapse mechanisms** already implemented. The most likely cause is **extreme class imbalance in training data** combined with **overly aggressive sample weighting**.

**Priority actions:**
1. Check class distribution in your data
2. Cap sample weights to max 10x
3. Reduce weight decay from 0.05 to 0.001
4. Increase warm-start LR factor from 0.01 to 0.1
5. Monitor for collapse warnings in logs
6. Verify label generation threshold

Apply these fixes and retrain. The model should converge to balanced predictions (~50/50 split) instead of collapsing to one class.
