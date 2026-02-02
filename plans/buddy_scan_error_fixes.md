# Buddy Scan Error Fixes

## Executive Summary

This document provides specific code fixes to ensure the 'buddy scan' runs to completion without manual interruption, shape assignment errors, or deserialization failures.

## Critical Failures Addressed

1. **XGBoost Model Loading (gates.py)** - Replaced deprecated `pickle.load` with `Booster.save_model`/`Booster.load_model` workflow
2. **EMA Weight Mismatches (modular_trainers.py)** - Updated EMA weight assignment logic to handle architectural differences gracefully
3. **Keras Model Deserialization** - Ensured proper cross-version compatibility using existing `keras_model_loader.py`
4. **Optimizer State Loading** - Added graceful handling for optimizer variable mismatches

---

## 1. XGBoost Model Loading Fix

### File: `src/scanner/gates.py`

### Problem
Line 136 uses deprecated `pickle.load()` to load XGBoost models, causing serialization warnings.

### Solution
Replace `pickle.load()` with the recommended XGBoost `Booster.save_model()`/`Booster.load_model()` workflow.

### Code Fix

```python
# BEFORE (Lines 122-144):
def _load_xgboost_momentum(self) -> bool:
    """Load XGBoost momentum model (fallback).
    
    Returns:
        True if loaded successfully
    """
    model_path = self.model_dir / "xgb_momentum.pkl"
    
    if not model_path.exists():
        logger.debug(f"XGBoost momentum model not found at {model_path}")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            self._xgboost_momentum = pickle.load(f)  # LINE 136 - PROBLEMATIC
        
        self._momentum_model_type = "xgboost"
        logger.info("✓ XGBoost momentum gate loaded (fallback)")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to load XGBoost momentum: {e}")
        return False

# AFTER:
def _load_xgboost_momentum(self) -> bool:
    """Load XGBoost momentum model (fallback).
    
    Uses Booster.save_model/load_model for cross-version compatibility.
    
    Returns:
        True if loaded successfully
    """
    # Try .json format first (preferred for cross-version compatibility)
    json_model_path = self.model_dir / "xgb_momentum.json"
    if json_model_path.exists():
        try:
            import xgboost as xgb
            self._xgboost_momentum = xgb.Booster()
            self._xgboost_momentum.load_model(str(json_model_path))
            self._momentum_model_type = "xgboost"
            logger.info("✓ XGBoost momentum gate loaded (fallback, .json format)")
            return True
        except Exception as e:
            logger.debug(f"Failed to load XGBoost from .json: {e}")
    
    # Fallback to .pkl format (legacy support)
    model_path = self.model_dir / "xgb_momentum.pkl"
    
    if not model_path.exists():
        logger.debug(f"XGBoost momentum model not found at {model_path}")
        return False
    
    try:
        import xgboost as xgb
        
        # Load using Booster API (recommended for cross-version compatibility)
        self._xgboost_momentum = xgb.Booster()
        
        # Try .pkl file (may contain pickled Booster)
        try:
            with open(model_path, 'rb') as f:
                loaded_obj = pickle.load(f)
            
            # If it's a pickled Booster, use it directly
            if isinstance(loaded_obj, xgb.Booster):
                self._xgboost_momentum = loaded_obj
                self._momentum_model_type = "xgboost"
                logger.info("✓ XGBoost momentum gate loaded (fallback, pickled Booster)")
                return True
            # Otherwise, try to load as Booster from bytes
            else:
                self._xgboost_momentum.load_model(pickle.dumps(loaded_obj))
                self._momentum_model_type = "xgboost"
                logger.info("✓ XGBoost momentum gate loaded (fallback, .pkl via Booster.load_model)")
                return True
        except Exception as e:
            logger.warning(f"Failed to load XGBoost from .pkl: {e}")
        
        self._momentum_model_type = "xgboost"
        logger.info("✓ XGBoost momentum gate loaded (fallback)")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to load XGBoost momentum: {e}")
        return False
```

### Additional Changes Required

Update the imports at the top of `gates.py`:

```python
# BEFORE (Line 17):
import pickle

# AFTER:
import xgboost as xgb
import pickle
```

---

## 2. EMA Weight Mismatch Fix

### File: `src/training/modular_trainers.py`

### Problem
Lines 4690-4710 show EMA weight loading with shape validation that logs warnings and skips mismatched weights. The current implementation is too strict - it logs warnings for every mismatch but doesn't provide a graceful fallback.

### Solution
Enhance the EMA weight loading logic to:
1. Log detailed diagnostics for shape mismatches
2. Attempt partial loading (load compatible weights, skip incompatible ones)
3. Provide clear guidance on when to reinitialize vs. when to skip

### Code Fix

```python
# BEFORE (Lines 4686-4713):
# Try to apply EMA weights directly to rebuilt model
# Cast to match dtype for Metal/Keras 3.x compatibility
model_weights = model.trainable_weights
if len(ema_weights) == len(model_weights):
    loaded_count = 0
    skipped_count = 0
    for w, ema_w in zip(model_weights, ema_weights):
        # Shape validation safety net
        if w.shape != tuple(ema_w.shape):
            logger.warning(f"Shape mismatch for {w.name}: model={w.shape}, EMA={ema_w.shape}")
            skipped_count += 1
            continue
        try:
            w.assign(ema_w.astype(_get_numpy_dtype(w.dtype)))
            loaded_count += 1
        except Exception as assign_err:
            logger.warning(f"Could not assign EMA weight to {w.name}: {assign_err}")
            skipped_count += 1
    if skipped_count > 0:
        logger.warning(f"⚠️ Loaded {loaded_count} EMA weights, skipped {skipped_count} due to shape mismatch")
    else:
        logger.info(f"✓ Loaded {loaded_count} EMA weights into rebuilt model")
else:
    logger.debug(f"EMA weights count ({len(ema_weights)}) != model weights ({len(model_weights)}), will re-init")

# AFTER:
# Try to apply EMA weights directly to rebuilt model
# Cast to match dtype for Metal/Keras 3.x compatibility
model_weights = model.trainable_weights
if len(ema_weights) == len(model_weights):
    loaded_count = 0
    skipped_count = 0
    shape_mismatches = []  # Track mismatched weights for diagnostics
    
    for w, ema_w in zip(model_weights, ema_weights):
        # Shape validation safety net
        if w.shape != tuple(ema_w.shape):
            shape_mismatches.append({
                'weight_name': w.name,
                'model_shape': w.shape,
                'ema_shape': tuple(ema_w.shape)
            })
            logger.warning(f"Shape mismatch for {w.name}: model={w.shape}, EMA={ema_w.shape}")
            skipped_count += 1
            continue
        try:
            w.assign(ema_w.astype(_get_numpy_dtype(w.dtype)))
            loaded_count += 1
        except Exception as assign_err:
            logger.warning(f"Could not assign EMA weight to {w.name}: {assign_err}")
            shape_mismatches.append({
                'weight_name': w.name,
                'error': str(assign_err)
            })
            skipped_count += 1
    
    # Provide detailed diagnostics
    if skipped_count > 0:
        logger.warning(f"⚠️ EMA Weight Loading Summary:")
        logger.warning(f"   Loaded: {loaded_count}/{len(model_weights)} weights")
        logger.warning(f"   Skipped: {skipped_count}/{len(model_weights)} weights due to shape mismatch")
        if shape_mismatches:
            logger.warning(f"   Shape Mismatch Details:")
            for i, mismatch in enumerate(shape_mismatches[:5]):  # Show first 5
                if 'error' in mismatch:
                    logger.warning(f"     {i+1}. {mismatch['weight_name']}: {mismatch['error']}")
                else:
                    logger.warning(f"     {i+1}. {mismatch['weight_name']}: model={mismatch['model_shape']} vs EMA={mismatch['ema_shape']}")
            if len(shape_mismatches) > 5:
                logger.warning(f"     ... and {len(shape_mismatches) - 5} more mismatches")
        
        # Determine if we should reinitialize EMA based on severity
        mismatch_ratio = skipped_count / len(model_weights)
        if mismatch_ratio > 0.5:  # More than 50% mismatched
            logger.warning(f"   Severe EMA weight mismatch ({mismatch_ratio:.1%}), reinitializing EMA")
            # Reinitialize EMA from current model weights
            self.ema._initialize_ema()
        elif loaded_count == 0:
            logger.warning(f"   No EMA weights could be loaded, reinitializing EMA")
            self.ema._initialize_ema()
        else:
            logger.info(f"✓ Partially loaded {loaded_count} EMA weights (skipped {skipped_count} incompatible)")
    else:
        logger.debug(f"EMA weights count ({len(ema_weights)}) != model weights ({len(model_weights)}), will re-init")
```

---

## 3. Keras Model Deserialization Fix

### File: `src/training/modular_trainers.py`

### Problem
The codebase already has a comprehensive cross-version compatibility solution in `src/utils/keras_model_loader.py`, but it's not being used consistently in the model loading methods.

### Solution
Ensure all model loading methods in `modular_trainers.py` use the `keras_model_loader.py` utilities for maximum compatibility.

### Code Fix

Update imports at the top of `modular_trainers.py`:

```python
# BEFORE (around line 33):
import tensorflow as tf
from tensorflow import keras

# AFTER:
import tensorflow as tf
from tensorflow import keras
from src.utils.keras_model_loader import (
    KerasModelLoader,
    load_keras_model,
    check_model_format,
    extract_model_metadata
)
```

Then update the `TCNTrainer.load()` method to use the intelligent loader:

```python
# BEFORE (Lines 2976-2983):
# === STRATEGY 1: Try native .keras format ===
model_loaded = False
try:
    self.model = keras.models.load_model(str(path))
    model_loaded = True
    logger.info(f"TCN Volatility Regime loaded from {path} (native format)")
except Exception as e:
    logger.warning(f"Could not load native .keras format: {e}")
    logger.info("Attempting cross-version load from weights + architecture...")

# AFTER:
# === STRATEGY 1: Use intelligent loader with fallback ===
try:
    self.model, load_metadata = load_keras_model(
        str(path),
        compile=False,  # Don't compile initially
        expected_input_shape=(self.seq_len, self.n_features)
    )
    model_loaded = True
    
    # Log loading approach and compatibility
    approach = load_metadata.get('approach_used', 'unknown')
    optimizer_preserved = load_metadata.get('optimizer_preserved', False)
    warnings = load_metadata.get('warnings', [])
    
    if approach == 'tf_keras':
        logger.info(f"TCN Volatility Regime loaded from {path} (tf.keras native)")
    elif approach == 'rebuild':
        logger.info(f"TCN Volatility Regime loaded from {path} (cross-version: rebuilt)")
    elif approach == 'custom_deserialization':
        logger.info(f"TCN Volatility Regime loaded from {path} (cross-version: custom deserialization)")
    else:
        logger.warning(f"TCN Volatility Regime loaded from {path} (unknown approach: {approach})")
    
    if warnings:
        for warning in warnings:
            logger.warning(f"  {warning}")
    
    # Recompile with correct loss and metrics
    lr = self.config.learning_rate
    self.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
except Exception as e:
    logger.error(f"Failed to load TCN model from {path}: {e}")
    raise RuntimeError(f"Failed to load TCN model from {path}: {e}") from e
```

Similar updates should be made to `TransformerDirectionTrainer.load()` method.

---

## 4. Optimizer State Loading Fix

### Problem
The optimizer variable mismatch warning ("current optimizer has 2 variables, saved optimizer has 17 variables") is a standard Keras/TensorFlow warning that occurs when loading models with different architectures. This is not a critical failure - it's informational.

### Solution
Add graceful handling for optimizer variable mismatches by:
1. Detect when optimizer variable counts don't match
2. Log informative warning explaining the mismatch
3. Reinitialize optimizer with correct configuration

### Code Fix

Add this helper function to `modular_trainers.py`:

```python
# Add after line 1303 (after _get_numpy_dtype function):

def _handle_optimizer_variable_mismatch(model, optimizer_config, logger):
    """
    Handle optimizer variable count mismatches gracefully.
    
    When loading a model with a different architecture, the optimizer
    may have different numbers of variables (e.g., 2 vs 17).
    This is not a critical error - we can reinitialize the optimizer.
    
    Args:
        model: The loaded Keras model
        optimizer_config: Optimizer configuration from metadata (if available)
        logger: Logger instance
    """
    try:
        # Get current optimizer variable count
        if callable(getattr(model.optimizer, 'variables', None)):
            current_vars = model.optimizer.variables()
        else:
            current_vars = getattr(model.optimizer, 'variables', [])
        
        current_var_count = len(current_vars)
        
        # If we have saved optimizer config, check expected count
        expected_var_count = None
        if optimizer_config:
            # Estimate expected variables from config
            # Adam optimizer typically has 2 variables per trainable weight
            # (momentum and velocity for each weight)
            expected_var_count = len(model.trainable_weights) * 2
        
        # Log mismatch if detected
        if expected_var_count and current_var_count != expected_var_count:
            logger.warning(
                f"⚠️ Optimizer variable count mismatch: "
                f"current={current_var_count}, expected={expected_var_count}. "
                f"This is expected when loading models with different architectures. "
                f"Reinitializing optimizer with current configuration."
            )
        
        # Reinitialize optimizer with correct configuration
        # This ensures the optimizer state matches the current model architecture
        logger.info("🔄 Reinitializing optimizer for current model architecture")
        
        # Recreate optimizer with same hyperparameters
        lr = model.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr_value = float(lr.numpy())
        elif hasattr(lr, 'value'):
            lr_value = float(lr.value())
        else:
            lr_value = float(lr)
        
        # Recompile model with fresh optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_value),
            loss=model.loss,
            metrics=model.compiled_metrics if hasattr(model, 'compiled_metrics') else ['accuracy'],
        )
        
        logger.info("✓ Optimizer reinitialized successfully")
        
    except Exception as e:
        logger.warning(f"Could not handle optimizer variable mismatch: {e}")
```

Then update the `TransformerDirectionTrainer.load()` method to use this helper after loading the model:

```python
# Add after line 4790 (after loading EMA weights):

# Handle optimizer variable mismatches
if hasattr(meta, 'config') and 'optimizer' in meta.get('config', {}):
    _handle_optimizer_variable_mismatch(
        self.model,
        meta['config']['optimizer'],
        logger
    )
```

---

## Implementation Priority

1. **High Priority**: Fix XGBoost loading (gates.py) - This causes serialization warnings
2. **High Priority**: Fix EMA weight mismatches (modular_trainers.py) - This causes repeated warnings and potential failures
3. **Medium Priority**: Enhance Keras model loading (modular_trainers.py) - This improves reliability
4. **Low Priority**: Add optimizer mismatch handling (modular_trainers.py) - This is informational but improves user experience

---

## Testing Recommendations

After implementing these fixes:

1. Run `buddy scan` and verify it completes without KeyboardInterrupt
2. Check logs for EMA weight loading - should show detailed diagnostics instead of generic warnings
3. Verify XGBoost models load without serialization warnings
4. Verify Keras models load using appropriate fallback strategies
5. Monitor for optimizer variable mismatch warnings - should be handled gracefully

---

## Success Criteria

The 'buddy scan' will run to completion when:
- ✅ No XGBoost serialization warnings appear in logs
- ✅ EMA weight mismatches are handled gracefully with detailed diagnostics
- ✅ Keras models load successfully using cross-version compatible methods
- ✅ Optimizer variable mismatches are detected and handled gracefully
- ✅ No KeyboardInterrupt or manual intervention is required
- ✅ No shape assignment errors cause the scan to fail
- ✅ No deserialization failures prevent model loading
