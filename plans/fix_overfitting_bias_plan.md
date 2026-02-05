# Implementation Plan: Fix Overfitting and 94.7% DOWN Prediction Bias

## Executive Summary

This plan addresses five critical errors identified through investigation:
1. Label threshold too high (0.3% → 0.15%) - increases clear samples from 0.8% to ~3.5%
2. BinaryCrossentropy instead of Focal Loss - better for class imbalance
3. Sample weights incorrect (1.5x → 62x) - required for 124:1 imbalance
4. Overly aggressive regularization - prevents minority class learning
5. No feature selection for 199 noisy features - reduce to ~50 important features

---

## Error #1: Label Threshold Too High

### Current State
- **File**: [`config/config_improved_H1.yaml`](config/config_improved_H1.yaml:67)
- **Current Value**: `direction_threshold: 0.003` (0.3%)
- **Problem**: Only 0.8% clear samples, 100% DOWN predictions
- **Root Cause**: Threshold too restrictive, filtering out most UP samples

### Required Fix
```yaml
# config/config_improved_H1.yaml (line 67)
direction_threshold: 0.0015  # Reduced from 0.003 (0.3% → 0.15%)
```

**Expected Impact**:
- Clear samples increase from ~0.8% to ~3.5% (4.4x more training data)
- UP/DOWN ratio becomes more balanced (~50:50 instead of 0.8:99.2)
- Model can learn both classes

---

## Error #2: BinaryCrossentropy Instead of Focal Loss

### Current State
- **File**: [`src/training/modular_trainers.py`](src/training/modular_trainers.py:2807-2813)
- **Current Code**:
```python
if self.config and self.config.use_focal_loss:
    logger.info(f"🎯 Using BinaryCrossentropy with label_smoothing={label_smoothing}")
    base_loss = keras.losses.BinaryCrossentropy(
        label_smoothing=label_smoothing,
    )
else:
    base_loss = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
```

**Problem**: Even when `use_focal_loss: true`, it uses BinaryCrossentropy!

### Required Fix
The Focal Loss class already exists in [`src/models/tensorflow_models.py`](src/models/tensorflow_models.py:170-214) as `BinaryFocalLoss`. We need to:

1. **Import Focal Loss** in modular_trainers.py:
```python
from src.models.tensorflow_models import BinaryFocalLoss
```

2. **Replace BinaryCrossentropy with Focal Loss**:
```python
# src/training/modular_trainers.py (lines 2807-2813)
if self.config and self.config.use_focal_loss:
    logger.info(f"🎯 Using Focal Loss for class imbalance (gamma=2.0, alpha=0.5)")
    base_loss = BinaryFocalLoss(
        gamma=2.0,  # Focusing parameter
        alpha=0.5,  # Class balance (0.5 = balanced)
        label_smoothing=label_smoothing,
    )
else:
    base_loss = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
```

**Expected Impact**:
- Focal Loss down-weights easy examples (majority class)
- Focuses learning on hard examples (minority class)
- Better handles 124:1 class imbalance

---

## Error #3: Sample Weights Incorrect (1.5x → 62x)

### Current State
- **File**: [`src/training/modular_trainers.py`](src/training/modular_trainers.py:2592-2602)
- **Current Code**:
```python
# === MINORITY CLASS BOOSTING (Anti-Bias) ===
# NOTE: 3x was too aggressive, causing gradient instability and training degradation
# 1.5x provides balance without destabilizing training
minority_class = 1 if n_up < n_down else 0
minority_boost = 1.5  # 1.5x boost for minority class (reduced from 3x - was too aggressive)
sample_weights[y_train_filtered == minority_class] *= minority_boost
```

**Problem**: For 124:1 imbalance, we need 62x weight, not 1.5x!

### Required Fix
```python
# src/training/modular_trainers.py (lines 2592-2620)
# === MINORITY CLASS BOOSTING (Anti-Bias) ===
# Calculate proper inverse frequency weighting for 124:1 imbalance
minority_class = 1 if n_up < n_down else 0
majority_class = 0 if n_up < n_down else 1

# Inverse frequency weighting: weight = total / (2 * class_count)
total = n_up + n_down
up_weight = total / (2 * n_up) if n_up > 0 else 1.0
down_weight = total / (2 * n_down) if n_down > 0 else 1.0

# Apply inverse frequency weights (not just a boost factor)
sample_weights[y_train_filtered == 1] = up_weight
sample_weights[y_train_filtered == 0] = down_weight

# Log the actual weights for verification
logger.info(f"🎯 Inverse frequency sample weights: UP={up_weight:.2f}, DOWN={down_weight:.2f}")
logger.info(f"🎯 Imbalance ratio: {max(n_up, n_down) / min(n_up, n_down):.1f}:1")
```

**Expected Impact**:
- Minority class gets 62x weight (124:1 imbalance ratio)
- Model receives proper gradient signal from minority class
- Balanced learning between UP and DOWN classes

---

## Error #4: Overly Aggressive Regularization

### Current State

**File**: [`config/config_improved_H1.yaml`](config/config_improved_H1.yaml)

Current regularization settings:
```yaml
# Model regularization (lines 80-87)
dropout: 0.5                    # Very high
recurrent_dropout: 0.0
kernel_regularizer: 0.005         # Strong L2

# Transformer regularization (lines 102-110)
transformer:
  dropout: 0.6                 # Very high
  l2_reg: 0.02                 # Strong L2
  input_noise: 0.08              # High noise
  label_smoothing: 0.15           # Strong smoothing
```

**Problem**: Over-regularization prevents model from learning minority class patterns.

### Required Fix
```yaml
# config/config_improved_H1.yaml

# Model regularization (lines 80-87)
dropout: 0.3                    # Reduced from 0.5
recurrent_dropout: 0.0
kernel_regularizer: 0.001         # Reduced from 0.005

# Transformer regularization (lines 102-110)
transformer:
  dropout: 0.3                 # Reduced from 0.6
  l2_reg: 0.005                 # Reduced from 0.02
  input_noise: 0.03              # Reduced from 0.08
  label_smoothing: 0.05           # Reduced from 0.15
```

**Also update in [`src/training/modular_trainers.py`](src/training/modular_trainers.py:2316-2324)**:
```python
# Line 2316 - L2 regularization
l2_weight = getattr(self.config, 'l2_reg', 0.005) if self.config else 0.005
# Change to:
l2_weight = getattr(self.config, 'l2_reg', 0.001) if self.config else 0.001

# Line 2323 - Input noise
noise_level = getattr(self.config, 'input_noise', 0.05) if self.config else 0.05
# Change to:
noise_level = getattr(self.config, 'input_noise', 0.03) if self.config else 0.03
```

**Expected Impact**:
- Model capacity increased, allowing minority class learning
- Reduced over-regularization prevents suppression of minority signals
- Better convergence with Focal Loss + proper sample weights

---

## Error #5: No Feature Selection for 199 Noisy Features

### Current State
- **File**: [`src/data/feature_engineering.py`](src/data/feature_engineering.py:1067-1122)
- **Feature Selection Method**: `select_features()` exists but NOT USED in training pipeline
- **Current**: All 199 features used without selection

### Required Fix

**Option A: Enable existing feature selection in training pipeline**

Add feature selection to [`src/training/modular_trainers.py`](src/training/modular_trainers.py:2432-2444):
```python
# After line 2444, add feature selection
from src.data.feature_engineering import FeatureEngineering

# ... existing code ...
self.feature_names = feature_names
self.n_features = X_train.shape[-1]

# === NEW: Feature Selection ===
if self.config.get('use_feature_selection', True):
    fe = FeatureEngineering()
    X_train_selected, selected_feature_names = fe.select_features(
        pd.DataFrame(X_train.reshape(-1, X_train.shape[-1]), 
                    columns=[f"feat_{i}" for i in range(X_train.shape[-1])]),
        target_col="feat_0",  # Use first feature as proxy
        method="correlation",
        top_k=50  # Reduce from 199 to ~50
    )
    
    # Apply same selection to validation
    X_val_selected = fe.select_features(
        pd.DataFrame(X_val.reshape(-1, X_val.shape[-1]), 
                    columns=[f"feat_{i}" for i in range(X_val.shape[-1])]),
        target_col="feat_0",
        method="correlation",
        top_k=50
    )[0]
    
    # Reshape back to sequence format
    X_train = X_train_selected.values.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_val = X_val_selected.values.reshape(X_val.shape[0], X_val.shape[1], -1)
    self.n_features = 50
    self.feature_names = selected_feature_names
    
    logger.info(f"🎯 Feature selection: {X_train.shape[-1]} → {self.n_features} features")
```

**Option B: Simpler correlation-based filtering**

Add to config:
```yaml
# config/config_improved_H1.yaml (add after line 74)
# Feature selection
use_feature_selection: true
feature_selection_method: correlation  # correlation, f_test, mutual_info
top_k_features: 50  # Select top 50 most important features
```

**Expected Impact**:
- Reduce features from 199 to ~50 (75% reduction)
- Remove noisy/uninformative features
- Faster training, less overfitting
- Better generalization

---

## Implementation Order

1. **Fix Error #1** (Label threshold) - Quick config change
2. **Fix Error #2** (Focal Loss) - Import and use existing class
3. **Fix Error #3** (Sample weights) - Replace 1.5x with proper inverse frequency
4. **Fix Error #4** (Regularization) - Reduce dropout, L2, noise in config
5. **Fix Error #5** (Feature selection) - Enable existing method in training pipeline
6. **Test** - Run training and verify:
   - Class distribution more balanced
   - Model converges properly
   - No prediction collapse
   - Validation accuracy improves

---

## Verification Metrics

After implementation, verify:
- [ ] Clear samples > 3% (was 0.8%)
- [ ] UP/DOWN ratio closer to 50:50 (was 0.8:99.2)
- [ ] Sample weights ~62x for minority class (was 1.5x)
- [ ] Model uses Focal Loss (check logs)
- [ ] Regularization parameters reduced (check config)
- [ ] Feature count ~50 (was 199)
- [ ] Validation accuracy improves (target: >55%)
- [ ] No prediction collapse (UP/DOWN both >5%)
- [ ] Balanced accuracy >50% for both classes

---

## Files to Modify

1. [`config/config_improved_H1.yaml`](config/config_improved_H1.yaml)
   - Lines 67: direction_threshold
   - Lines 80-87: Model regularization
   - Lines 102-110: Transformer regularization
   - Add: feature_selection config

2. [`src/training/modular_trainers.py`](src/training/modular_trainers.py)
   - Lines 2807-2813: Replace BinaryCrossentropy with Focal Loss
   - Lines 2592-2602: Fix sample weights (1.5x → 62x)
   - Lines 2316-2324: Update regularization parameters
   - Lines 2444: Add feature selection

3. [`src/models/tensorflow_models.py`](src/models/tensorflow_models.py)
   - Already has BinaryFocalLoss - just needs import

---

## Risk Assessment

**Low Risk Changes**:
- Label threshold reduction (config only)
- Regularization reduction (config only)

**Medium Risk Changes**:
- Sample weights (requires careful testing)
- Feature selection (may change model input shape)

**High Risk Changes**:
- Focal Loss integration (needs thorough testing)

**Mitigation Strategy**:
1. Make changes incrementally
2. Test each change independently
3. Compare metrics before/after
4. Keep backup of working config
5. Use validation set to prevent overfitting

---

## Success Criteria

The implementation is successful when:
1. Clear samples > 3% (4x increase from 0.8%)
2. UP predictions > 5% (currently ~0.3%)
3. DOWN predictions < 95% (currently 94.7%)
4. Balanced accuracy > 50% for both classes
5. Validation accuracy > 55% (baseline)
6. No severe overfitting (train-val gap < 15%)
7. Model converges within 100 epochs (not stuck)
