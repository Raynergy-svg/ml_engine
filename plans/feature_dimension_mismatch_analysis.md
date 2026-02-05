# TensorFlow Matrix Dimension Mismatch - Root Cause Analysis & Solution

## Problem Summary

**Error Location**: `transformer_direction_1/input_projection_1/MatMul` layer  
**Error Message**: Input tensor shape `[7, 60, 80]` incompatible with weight matrix expecting 37 features  
**Context**: Walk-forward validation prediction phase in enterprise ML training pipeline

---

## Root Cause Analysis

### 1. Feature Generation Pipeline

**File**: [`src/data/feature_engineering.py`](../src/data/feature_engineering.py)

The [`create_features()`](../src/data/feature_engineering.py:938) method generates **80+ features**:
- Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, etc.)
- Statistical features (returns, volatility, skewness, kurtosis, z-scores)
- Time features (hour, day of week, trading sessions)
- Lag features
- Rolling window features
- Regime detection features
- Momentum divergence features
- Feature interaction terms

### 2. Data Loader Feature Selection

**File**: [`src/core/modular_data_loaders.py`](../src/core/modular_data_loaders.py)

The [`load_direction_data()`](../src/core/modular_data_loaders.py:1527) function applies **variance-based feature selection**:

```python
# Line 1576: Selects up to 80 uncorrelated features
max_features = 80
correlation_threshold = 0.80

# Lines 1636-1660: Removes highly correlated features
# Result: ~60 features selected
```

### 3. Trainer Feature Selection

**File**: [`src/training/modular_trainers.py`](../src/training/modular_trainers.py)

The [`TransformerDirectionTrainer`](../src/training/modular_trainers.py:3175) applies **Random Forest importance-based feature selection**:

```python
# Lines 3419-3594: Feature selection logic
use_feature_selection = True  # Default
feature_selection_method = 'random_forest'  # Default
top_k_features = 50  # Default from config

# Lines 3478-3593: RF importance selection
# Result: Reduces ~60 features → 37 features
```

**Critical Metadata Saved** (lines 4650-4703):
```python
metadata = {
    'feature_names': selected_feature_names,  # 37 feature names
    'n_features': len(selected_feature_names),  # 37
    'selected_indices': selected_indices,  # Indices of selected features
    'seq_len': self.seq_len,  # 60
}
```

### 4. Walk-Forward Validation Prediction

**File**: [`src/training/walkforward_validation.py`](../src/training/walkforward_validation.py)

The [`train_direction_with_walkforward()`](../src/training/walkforward_validation.py:1025) function **does not apply feature selection**:

```python
# Lines 1084-1090: Direct data split without feature selection
X_train_fold = X[train_idx]  # Passes ALL 80 features!
y_train_fold = y[train_idx]

# Lines 1102-1108: Trains fresh model on each fold
fold_trainer = TransformerDirectionTrainer(trainer.config)
metrics = fold_trainer.train(
    X_train_fold, y_train_fold,  # 80 features passed
    ...
)
```

### 5. The Dimension Mismatch

**Training Phase**:
1. Feature engineering generates 80 features
2. Data loader selects ~60 uncorrelated features
3. Trainer applies RF selection → 37 features
4. Model built with `input_shape=(60, 37)`
5. Metadata saved: `n_features=37, selected_indices=[...]`

**Walk-Forward Validation Prediction Phase**:
1. Feature engineering generates 80 features
2. Data loader selects ~60 uncorrelated features
3. **Feature selection NOT applied** → 80 features passed
4. Model expects 37 features but receives 80
5. **ERROR**: Matrix dimension mismatch at `input_projection_1/MatMul`

---

## Solution Design

### Option 1: Apply Feature Selection During Walk-Forward (Recommended)

**Modify**: [`src/training/walkforward_validation.py`](../src/training/walkforward_validation.py:1025)

**Changes**:
1. Load model metadata to get `selected_indices`
2. Apply same feature selection during prediction
3. Ensure consistent feature order

```python
# After line 1100: Load model metadata
model_meta = load_model_metadata(trainer.model_path)

# Extract selected indices
selected_indices = model_meta.get('selected_indices', None)

# Apply feature selection before training
if selected_indices is not None:
    X_train_fold = X_train_fold[:, selected_indices]
    X_val_fold = X_val_fold[:, selected_indices]
    X_test_fold = X_test_fold[:, selected_indices] if X_test_fold is not None else None
else:
    # Fallback: use all features
    X_train_fold = X_train_fold
    X_val_fold = X_val_fold
```

### Option 2: Disable Feature Selection During Training

**Modify**: [`src/training/modular_trainers.py`](../src/training/modular_trainers.py:3175)

**Changes**:
1. Set `use_feature_selection = False` in config
2. Use all features from data loader (~60)
3. Update model input shape accordingly

```python
# In config file (e.g., config/config_improved_H1.yaml):
transformer_direction:
  use_feature_selection: false  # Disable RF selection
  top_k_features: null  # Not used when disabled
```

### Option 3: Fix Data Loader to Return Consistent Features

**Modify**: [`src/core/modular_data_loaders.py`](../src/core/modular_data_loaders.py:1527)

**Changes**:
1. Return selected indices from data loader
2. Apply same selection in trainer
3. Ensure consistency across training and inference

```python
# In load_direction_data(), return selected_indices:
result = {
    'X_train': X_train_scaled,
    'y_train': y_train,
    'selected_indices': selected_indices,  # NEW
    'feature_names': features,
    ...
}
```

---

## Recommended Implementation Path

### Phase 1: Immediate Fix (Option 1)

**Priority**: High  
**Risk**: Low  
**Impact**: Resolves walk-forward validation error

**Steps**:
1. Modify [`src/training/walkforward_validation.py`](../src/training/walkforward_validation.py:1025) to load and apply `selected_indices`
2. Add metadata loading function
3. Apply feature selection before training each fold
4. Test with existing trained models

### Phase 2: Long-term Improvement (Option 2)

**Priority**: Medium  
**Risk**: Low  
**Impact**: Simplifies pipeline, prevents future mismatches

**Steps**:
1. Update config files to disable feature selection
2. Verify model performance with ~60 features
3. Retrain models if needed

### Phase 3: Architectural Fix (Option 3)

**Priority**: Low  
**Risk**: Medium  
**Impact**: Most robust solution, requires more changes

**Steps**:
1. Refactor data loaders to return selected indices
2. Update trainers to use loader-provided indices
3. Ensure consistency across all model types

---

## Code Changes Required

### File: `src/training/walkforward_validation.py`

**Location**: After line 1100  
**Add**:
```python
def load_model_metadata(model_path: str) -> dict:
    """Load model metadata including selected feature indices."""
    import json
    from pathlib import Path
    
    meta_path = Path(model_path).parent / 'model_metadata.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}
```

**Modify** `train_direction_with_walkforward()`:
```python
# After line 1100: Load metadata
model_meta = load_model_metadata(trainer.model_path)
selected_indices = model_meta.get('selected_indices', None)

# Before line 1102: Apply feature selection
if selected_indices is not None:
    X_train_fold = X_train_fold[:, selected_indices]
    X_val_fold = X_val_fold[:, selected_indices]
    X_test_fold = X_test_fold[:, selected_indices] if X_test_fold is not None else None
    logger.info(f"Applied feature selection: {len(selected_indices)} features")
else:
    logger.warning("No selected_indices in metadata, using all features")
    X_train_fold = X_train_fold
    X_val_fold = X_val_fold
    X_test_fold = X_test_fold
```

---

## Verification Steps

1. **Test with existing trained model**:
   - Load model with 37 features
   - Run walk-forward validation with 80-feature data
   - Verify feature selection reduces to 37 before training
   - Confirm no dimension mismatch

2. **Validate feature alignment**:
   - Check that selected feature names match between training and inference
   - Verify feature order is consistent
   - Log feature names for debugging

3. **Monitor model performance**:
   - Compare performance with 37 vs 60 features
   - Ensure accuracy doesn't degrade significantly
   - Adjust feature selection parameters if needed

---

## Summary

**Root Cause**: Feature selection applied during training (80→60→37) but NOT applied during walk-forward validation prediction (80→80→ERROR)

**Immediate Fix**: Load `selected_indices` from model metadata and apply during walk-forward validation

**Long-term Solution**: Disable feature selection or refactor pipeline to ensure consistency

**Files to Modify**:
1. [`src/training/walkforward_validation.py`](../src/training/walkforward_validation.py) - Add metadata loading and feature selection
2. [`src/training/modular_trainers.py`](../src/training/modular_trainers.py) - Ensure metadata saves selected_indices
3. Config files - Optionally disable feature selection

**Expected Outcome**:
- Walk-forward validation runs without dimension mismatch
- Model receives consistent feature count (37) during training and inference
- No degradation in model performance
