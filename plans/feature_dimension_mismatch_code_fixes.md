# Code Modifications to Fix TensorFlow Matrix Dimension Mismatch

## Overview

This document provides the specific code modifications needed to resolve the TensorFlow matrix dimension mismatch error that occurs during walk-forward validation.

**Error**: Input tensor shape `[7, 60, 80]` incompatible with weight matrix expecting 37 features at `transformer_direction_1/input_projection_1/MatMul` layer

---

## Solution 1: Add Feature Selection to Walk-Forward Validation (Recommended)

### File: `src/training/walkforward_validation.py`

**Location**: After line 1100 in [`train_direction_with_walkforward()`](../src/training/walkforward_validation.py:1025)

**Add This Function** (insert after imports, before the function definition):

```python
def load_model_metadata(model_path: str) -> dict:
    """
    Load model metadata including selected feature indices.
    
    Args:
        model_path: Path to the model file (e.g., 'trained_data/models/joint/transformer_direction.keras')
    
    Returns:
        Dictionary containing model metadata (selected_indices, feature_names, n_features, etc.)
        Returns empty dict if metadata file not found
    """
    import json
    from pathlib import Path
    
    try:
        model_dir = Path(model_path).parent
        
        # Try multiple metadata file locations
        meta_path = model_dir / 'model_metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✓ Loaded metadata from {meta_path}")
            return metadata
        
        # Check for modular_ensemble.meta.json
        meta_path = model_dir / 'modular_ensemble.meta.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✓ Loaded metadata from {meta_path}")
            return metadata
        
        logger.warning(f"No valid metadata found in {model_dir}")
        return {}
        
    except Exception as e:
        logger.error(f"Failed to load model metadata from {model_path}: {e}")
        return {}
```

**Modify `train_direction_with_walkforward()` Function**:

Replace lines 1080-1108 with:

```python
        # Load model metadata to get selected_indices
        model_meta = load_model_metadata(trainer.model_path)
        selected_indices = model_meta.get('selected_indices', None)
        stored_n_features = model_meta.get('n_features', None)
        
        logger.info(f"Model metadata: n_features={stored_n_features}, selected_indices={len(selected_indices) if selected_indices else 'None'}")
        
        # Apply feature selection if selected_indices are available
        if selected_indices is not None and len(selected_indices) > 0:
            X_train_fold = X_train_fold[:, selected_indices]
            X_val_fold = X_val_fold[:, selected_indices]
            if X_test_fold is not None:
                X_test_fold = X_test_fold[:, selected_indices]
            logger.info(f"✓ Applied feature selection: {len(selected_indices)} features")
        else:
            logger.warning("No selected_indices in metadata, using all features")
            X_train_fold = X_train_fold
            X_val_fold = X_val_fold
            X_test_fold = X_test_fold
```

---

## Solution 2: Ensure Model Metadata Saves Selected Indices

### File: `src/training/modular_trainers.py`

**Location**: Lines 4650-4703 in [`TransformerDirectionTrainer.save()`](../src/training/modular_trainers.py:4650)

**Verify Metadata Includes Selected Indices**:

The metadata should already include `selected_indices` (line 4655), but verify it's being saved correctly:

```python
# Around line 4655, ensure this exists:
metadata = {
    'feature_names': self.feature_names,
    'n_features': self.n_features,
    'selected_indices': self.selected_indices,  # CRITICAL: Must be saved
    'seq_len': self.seq_len,
    'n_classes': self.n_classes,
    'architecture': {
        'input_shape': (self.seq_len, self.n_features),
        'transformer_d_model': self.transformer_d_model,
        ...
    },
    ...
}
```

---

## Solution 3: Update Config to Disable Feature Selection (Alternative)

### File: `config/config_improved_H1.yaml`

**Location**: Around line 71 (transformer_direction section)

**Add or Modify**:

```yaml
transformer_direction:
  # ... existing config ...
  
  # Disable feature selection to use all features from data loader
  use_feature_selection: false  # Set to false
  feature_selection_method: null  # Not used when disabled
  top_k_features: null  # Not used when disabled
```

---

## Solution 4: Update Data Loader to Return Selected Indices

### File: `src/core/modular_data_loaders.py`

**Location**: Lines 1800-1817 in [`load_direction_data()`](../src/core/modular_data_loaders.py:1527)

**Modify Return Statement** (around line 1815):

```python
result = {
    'X_train': X_train_scaled.astype(np.float32),
    'y_train': y[train_idx],
    'w_train': weights[train_idx],
    'X_val': X_val_scaled.astype(np.float32),
    'y_val': y[val_idx],
    'w_val': weights[val_idx],
    'X_test': X_test_scaled.astype(np.float32),
    'y_test': y[test_idx],
    'feature_names': features,
    'selected_indices': selected_indices,  # ADD THIS LINE
    'label_stats': label_stats,
    'scaler': scaler,  # Save scaler for inference
}
```

**Add Selected Indices Calculation** (before line 1660, after feature selection):

```python
# After line 1659 (final_selected = [...]), add:
selected_indices = [features.index(f) for f in final_selected if f in features]
```

---

## Implementation Steps

### Step 1: Add Metadata Loading Function

1. Add [`load_model_metadata()`](../src/training/walkforward_validation.py) function to [`src/training/walkforward_validation.py`](../src/training/walkforward_validation.py)
2. Import at top of file: `from pathlib import Path`

### Step 2: Modify Walk-Forward Validation

1. Load model metadata at start of [`train_direction_with_walkforward()`](../src/training/walkforward_validation.py:1025)
2. Extract `selected_indices` from metadata
3. Apply feature selection to X_train, X_val, X_test before training
4. Log feature selection application for debugging

### Step 3: Verify Model Metadata

1. Check that [`TransformerDirectionTrainer.save()`](../src/training/modular_trainers.py:4650) saves `selected_indices`
2. Verify metadata is saved correctly with model file
3. Test loading metadata to ensure it contains expected fields

### Step 4: Test the Fix

1. Run walk-forward validation with existing trained model
2. Verify no dimension mismatch errors
3. Check logs for "Applied feature selection" message
4. Confirm model receives 37 features during training and inference

---

## Expected Behavior After Fix

### Before Fix:
```
Training: 80 features → 60 selected → 37 selected → Model built with (60, 37)
Walk-Forward: 80 features → 60 selected → NO SELECTION → Model receives (60, 80)
Result: ERROR at input_projection_1/MatMul (80 vs 37)
```

### After Fix:
```
Training: 80 features → 60 selected → 37 selected → Model built with (60, 37)
Walk-Forward: 80 features → 60 selected → 37 selected → Model receives (60, 37)
Result: SUCCESS - Consistent feature dimensions
```

---

## Alternative Solutions

### Option A: Disable Feature Selection Entirely

Set `use_feature_selection: false` in config to use all ~60 features from data loader.

**Pros**:
- Simpler pipeline
- No feature selection overhead
- Consistent features across training and inference

**Cons**:
- May use suboptimal features
- Model performance may degrade
- Requires retraining models

### Option B: Increase top_k_features

Set `top_k_features: 60` in config to match data loader output.

**Pros**:
- Minimal code changes
- Uses more features (may improve performance)

**Cons**:
- Still has feature selection overhead
- May not match model's 37 features if already trained

### Option C: Retrain Models with Disabled Feature Selection

1. Disable feature selection in config
2. Retrain all models
3. Walk-forward validation will work consistently

**Pros**:
- Cleanest solution
- No compatibility issues

**Cons**:
- Requires full retraining
- Time-consuming

---

## Verification Checklist

- [ ] Add `load_model_metadata()` function to walkforward_validation.py
- [ ] Import `Path` and `json` modules
- [ ] Modify `train_direction_with_walkforward()` to load and apply selected_indices
- [ ] Verify model metadata saves `selected_indices`
- [ ] Test walk-forward validation with existing trained model
- [ ] Confirm no dimension mismatch errors
- [ ] Check logs for feature selection application
- [ ] Verify model performance is acceptable

---

## Notes

1. **Root Cause Confirmed**: Feature selection applied during training (80→60→37) but NOT during walk-forward validation (80→60→ERROR)

2. **Primary Fix**: Load `selected_indices` from model metadata and apply during walk-forward validation

3. **Backup Plans**: Alternative solutions provided if primary fix doesn't work or causes other issues

4. **Testing**: Always test with existing trained models before retraining

5. **Logging**: Enhanced logging added to track feature selection application for debugging
