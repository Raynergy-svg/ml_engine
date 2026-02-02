# Inference Failure Root Cause Analysis
## Executive Summary

This document provides a comprehensive analysis of the critical inference failure where:
1. RL GATES and Ensemble models are not loading
2. Meta model is being skipped
3. Sub-models (Ridge, XGBoost, Random Forest) produce zero-signal outputs
4. Calibration is failing and defaulting to raw inputs

**Date**: 2026-01-31
**Analysis Method**: Code review of inference pipeline, model loading logic, and calibration system
**Severity**: CRITICAL - System is producing no trading signals

---

## Table of Contents

1. [RL GATES Loading Failure](#1-rl-gates-loading-failure)
2. [Ensemble Model Loading Issues](#2-ensemble-model-loading-issues)
3. [Meta Model Skipping Issue](#3-meta-model-skipping-issue)
4. [Calibration Failure and Raw Input Fallback](#4-calibration-failure-and-raw-input-fallback)
5. [Root Cause Summary](#5-root-cause-summary)
6. [Recommended Fixes](#6-recommended-fixes)

---

## 1. RL GATES Loading Failure

### 1.1 Current State

**RL Models Available**:
- `trained_data/models/sac_gate_thresholds.zip` - RL Gate Threshold Optimizer (SAC)
- `trained_data/models/ppo_optimal_exit.zip` - RL Optimal Exit Timing (PPO)
- `trained_data/models/rl_position_sizer.zip` - RL Position Sizer (PPO)

**RL Loading Code** (`src/core/modular_inference.py`):
```python
# Lines 125-126: RL model paths
RL_MODEL_PATH = Path("trained_data/models/rl_position_sizer.zip")
RL_ONNX_PATH = Path("trained_data/models/rl_position_sizer.onnx")
RL_GATE_MODEL_PATH = Path("trained_data/models/sac_gate_thresholds.zip")
RL_EXIT_MODEL_PATH = Path("trained_data/models/ppo_optimal_exit.zip")

# Lines 1214-1257: RL gates auto-detection
if self.use_rl_gates is None:
    if RL_GATE_MODEL_PATH.exists():
        logger.info("🔍 RL Gate model detected - auto-enabling RL gates")
        self.use_rl_gates = True
    else:
        self.use_rl_gates = False
```

### 1.2 Root Cause Analysis

**PRIMARY ISSUE**: RL models are NOT loading due to import failures

The RL loading code has several layers of protection:

1. **Timeout Protection** (Lines 131-166):
   ```python
   def _lazy_load_rl_sizer():
       from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
       
       with ThreadPoolExecutor(max_workers=1) as executor:
           future = executor.submit(_lazy_load_rl_sizer_unsafe)
           try:
               _RLPositionSizer, _ = future.result(timeout=RL_LOAD_TIMEOUT_SECONDS)
           except FutureTimeout:
               logger.warning(f"RL sizer import timed out after {RL_LOAD_TIMEOUT_SECONDS}s")
               RL_AVAILABLE = False
   ```

2. **Import Error Handling** (Lines 158-166):
   ```python
   except ImportError:
       RL_AVAILABLE = False
       RLPositionSizer = None
   except Exception as e:
       logger.warning(f"RL sizer import failed: {type(e).__name__}: {e}")
       RL_AVAILABLE = False
   ```

3. **Lazy Loading with No Fallback**:
   - When `_lazy_load_rl_gates()` or `_lazy_load_rl_exits()` is called, if the import fails or times out, the model simply becomes `None`
   - No retry mechanism
   - No error logging to help diagnose the actual failure

### 1.3 Evidence from Code

**Evidence 1**: The `_lazy_load_rl_gates()` function does NOT use timeout protection:
```python
# Lines 169-182: NO TIMEOUT PROTECTION
def _lazy_load_rl_gates():
    """Lazy load GateThresholdRL to avoid TF/PyTorch GPU conflicts."""
    global GateThresholdRL, RL_GATES_AVAILABLE
    if GateThresholdRL is not None:
        return GateThresholdRL, RL_GATES_AVAILABLE
    
    try:
        from src.rl.gate_threshold_env import GateThresholdRL as _GateThresholdRL
        GateThresholdRL = _GateThresholdRL
        RL_GATES_AVAILABLE = True
        return GateThresholdRL, RL_GATES_AVAILABLE
    except ImportError:
        RL_GATES_AVAILABLE = False
        GateThresholdRL = None
        return None, False
```

**Evidence 2**: The `_lazy_load_rl_exits()` function also lacks timeout protection (Lines 185-198).

**Evidence 3**: In `load_models()` (Lines 1229-1257), when RL loading fails, there's no fallback or retry:
```python
# Lines 1230-1242: RL gates loading
if self.use_rl_gates:
    GateRL, gates_available = _lazy_load_rl_gates()
    if gates_available and GateRL is not None:
        self.rl_gate_optimizer = GateRL()
        if self.rl_gate_optimizer.load():
            logger.info("✓ RL Gate Threshold Optimizer loaded (SAC)")
            self._rl_gates_loaded = True
        else:
            logger.info("ℹ RL Gates not trained - using fixed thresholds")
            self.rl_gate_optimizer = None
```

**Evidence 4**: RL gates are loaded but `_rl_gates_loaded` flag is not checked before use in inference (Lines 2856-2879).

### 1.4 Impact

**When RL GATES FAIL TO LOAD**:
1. `self._rl_gates_loaded = False`
2. `self.rl_gate_optimizer = None`
3. In `predict()` method (Lines 2856-2879), RL gate thresholds are not used:
   ```python
   if self._rl_gates_loaded and self.rl_gate_optimizer is not None:
       adjusted = self.rl_gate_optimizer.get_adjusted_thresholds(...)
   ```
4. System falls back to fixed config thresholds instead of learned RL-optimized thresholds

---

## 2. Ensemble Model Loading Issues

### 2.1 Current State

**Models in Joint Directory** (`trained_data/models/joint/`):
- `lgbm_momentum.pkl` - LightGBM momentum (from joint training)
- `lgbm_risk.pkl` - LightGBM risk (from joint training)
- `ridge_confidence.pkl` - Ridge confidence (from joint training)
- `transformer_direction.keras` - Transformer direction model
- `transformer_direction.ema.pkl` - EMA checkpoint
- `transformer_direction.ewc.pkl` - EWC checkpoint
- `transformer_direction.meta.pkl` - Metadata

**Models in Generic Directory** (`trained_data/models/`):
- `xgb_momentum.pkl` - XGBoost momentum (generic)
- `rf_risk.pkl` - Random Forest risk (generic)
- `ridge_confidence.pkl` - Ridge confidence (generic)
- `transformer_direction.keras` - Transformer direction (generic)

**Model Loading Code** (`src/core/modular_inference.py`):

#### 2.2 Model Path Resolution Order (Lines 961-989):
```python
def _get_model_path(self, model_name: str, extension: str = ".keras") -> Path:
    """Get path to a model file, auto-preferring pair-specific then joint models.
    
    Lookup order:
    1. trained_data/models/{instrument}/{model_name}{extension}  (pair-specific)
    2. trained_data/models/joint/{model_name}{extension}  (joint training)
    3. trained_data/models/{model_name}{extension}  (generic fallback)
    """
```

#### 2.3 Transformer/Direction Model Loading (Lines 1059-1095):
```python
elif transformer_path.exists():
    try:
        self.tcn = TransformerDirectionTrainer()
        self.tcn.load(str(transformer_path))
        self.use_regime = False
        logger.info(f"✓ Transformer direction model loaded from {transformer_path}")
        
        # Log training state from lineage (v2)
        if hasattr(self.tcn, 'lineage') and self.tcn.lineage:
            lineage = self.tcn.lineage
            variance_weight = getattr(lineage, 'auto_variance_weight', 'N/A')
            lr_reductions = getattr(lineage, 'lr_reductions_count', 0)
            collapse_recoveries = getattr(lineage, 'collapse_recovery_count', 0)
            lineage_ver = getattr(lineage, 'lineage_version', 1)
            instrument_trained = getattr(lineage, 'instrument', 'unknown')
            logger.info(
                f"📊 Model training state: instrument={instrument_trained}, "
                f"variance_weight={variance_weight}, lr_reductions={lr_reductions}, "
                f"collapse_recoveries={collapse_recoveries}, lineage_v{lineage_ver}"
            )
    except Exception as e:
        self.tcn = None
        self.use_regime = False
        logger.warning(f"⚠️ Transformer direction model failed to load: {e}")
        logger.warning(f"⚠️ Model may have been saved with different Keras version")
        if not self.config.permissive_mode:
            logger.info("ℹ Auto-enabling permissive_mode due to model load failure")
            self.config.permissive_mode = True
```

#### 2.4 LightGBM Loading (Lines 1137-1156):
```python
if lgbm_momentum_path.exists():
    try:
        from src.training.modular_trainers import LightGBMMomentumTrainer
        self.xgb = LightGBMMomentumTrainer()
        self.xgb.load(str(lgbm_momentum_path))
        logger.info(f"✓ LightGBM Momentum loaded from {lgbm_momentum_path}")
    except Exception as e:
        logger.warning(f"Failed to load LightGBM momentum: {e}")
        # Fall back to XGBoost
        if xgb_path.exists():
            self.xgb = XGBoostTrainer()
            self.xgb.load(str(xgb_path))
            logger.info(f"✓ XGBoost loaded from {xgb_path} (fallback)")
```

### 2.5 Root Cause Analysis

**PRIMARY ISSUE**: Model loading failures due to Keras version incompatibility

**Evidence 1**: The code explicitly logs "Model may have been saved with different Keras version" when loading fails.

**Evidence 2**: When Transformer fails, `permissive_mode` is auto-enabled, which bypasses gate models and uses only Transformer direction.

**Evidence 3**: LightGBM models are used from joint training, but the system may be looking for XGBoost models in generic directory.

**Evidence 4**: The `_get_model_path()` method checks joint directory first, so if instrument is not specified, it will use joint-trained models.

### 2.6 Impact

**When MODELS FAIL TO LOAD**:
1. `self.tcn = None` - No direction/regime model
2. `self.xgb = None` - No momentum model
3. `self.rf = None` - No risk model
4. `self.ridge = None` - No confidence model
5. All gates fail - `confidence_gate_passed = False`, `momentum_gate_passed = False`, `risk_gate_passed = False`
6. System returns zero-signal outputs (all gates return False)

---

## 3. Meta Model Skipping Issue

### 3.1 Current State

**Meta-Labeling Code** (`src/training/meta_labeling.py`):
```python
# Lines 65-72: Configuration
class MetaLabelingConfig:
    min_confidence_threshold: float = 0.55  # Trades where meta-confidence >= this
    use_triple_barrier: bool = True  # Use TP/SL outcomes
    include_primary_proba: bool = True  # Include primary model's probability
```

**Meta-Labeler Loading** (`src/core/modular_inference.py` Lines 1458-1505):
```python
def _load_meta_labeler(self) -> None:
    """Load meta-labeler model for trade filtering.
    
    Lookup order:
    1. trained_data/models/{instrument}/meta_labeler.pkl  (pair-specific)
    2. trained_data/models/meta_labeler.pkl  (generic fallback)
    """
    
    # Build list of paths to check
    meta_labeler_paths = [
        self.model_dir / "meta_labeler.pkl",
    ]
    
    # Add pair-specific path first if instrument is set
    if self.instrument and self.instrument != "GENERIC":
        meta_labeler_paths.insert(0, self.model_dir / self.instrument / "meta_labeler.pkl")
    
    for path in meta_labeler_paths:
        if path.exists():
            try:
                self.meta_labeler = MetaLabeler.load(path)
                self._meta_labeler_loaded = True
                threshold = self.config.min_meta_confidence
                primary_acc = getattr(self.meta_labeler, '_primary_accuracy', 0.5)
                logger.info(f"✓ Meta-labeler loaded from {path}")
                logger.info(f"  Threshold: {threshold:.0%}, Primary accuracy: {primary_acc:.1%}")
                return
            except Exception as e:
                logger.warning(f"Failed to load meta-labeler from {path}: {e}")
    
    # No meta-labeler found
    if self.config.enable_meta_labeling:
        logger.info("ℹ Meta-labeling enabled but no trained model found")
        logger.info("  To train: models save meta_labeler.pkl during buddy training")
```

### 3.2 Root Cause Analysis

**PRIMARY ISSUE**: Meta model is not being loaded because `meta_labeler.pkl` file does not exist

**Evidence 1**: The code checks for `meta_labeler.pkl` in:
   - `trained_data/models/meta_labeler.pkl` (generic fallback)
   - `trained_data/models/{instrument}/meta_labeler.pkl` (pair-specific)

**Evidence 2**: When meta-labeler is not loaded, `_meta_labeler_loaded = False`, and in `predict()` (Lines 3114-3318):
   ```python
   if self._meta_labeler_loaded and self.meta_labeler is not None:
       # Meta-labeler expects direction indicator
       # (direction encoded as 1 for LONG, -1 for SHORT)
       
       # Check if model has predict_proba
       if hasattr(self._meta_labeler, 'predict_proba'):
           proba = self._meta_labeler.predict_proba(X)
           success_prob = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0])
       else:
           pred = self._meta_labeler.predict(X)
           success_prob = float(pred[0])
       
       # Get meta-labeler confidence (probability that primary signal is correct)
       meta_conf_array = self.meta_labeler.predict_meta_confidence(meta_features, primary_prob.flatten())
       meta_confidence = float(meta_conf_array[0]) if len(meta_conf_array) > 0 else 0.0
       
       # Check gate threshold
       meta_gate_passed = meta_confidence >= self.config.min_meta_confidence
       
       if not meta_gate_passed:
           meta_reason = f"low_meta_conf({meta_confidence:.2f}<{self.config.min_meta_confidence})"
   ```

**Evidence 3**: When meta-labeler is not loaded, `meta_gate_passed = True` by default (Line 3317), so the gate does nothing.

### 3.3 Impact

**WHEN META MODEL IS NOT LOADED**:
1. `meta_gate_passed = True` - Gate always passes (no filtering)
2. Meta model provides no additional filtering
3. System loses the ability to filter trades based on meta-model confidence
4. All trades proceed regardless of meta-model's prediction of success probability

---

## 4. Calibration Failure and Raw Input Fallback

### 4.1 Current State

**Calibration Loading Code** (`src/core/modular_inference.py` Lines 1360-1456):
```python
def _load_calibration(self) -> None:
    """Load confidence calibration from model metadata or standalone calibration file.
    
    Calibration sources (in priority order):
    1. Standalone calibration file: trained_data/models/confidence_calibrator.pkl
    2. Pair-specific calibration: trained_data/models/{instrument}/confidence_calibrator.pkl
    3. Calibration data in model metadata: modular_ensemble.meta.json -> calibration
    4. Calibration data in Transformer metadata: transformer_direction.meta.pkl -> output_calibration
    """
    
    # Try loading standalone calibration file first
    calibration_paths = [
        self.model_dir / "confidence_calibrator.pkl",
    ]
    
    # Add pair-specific path if instrument is set
    if self.instrument and self.instrument != "GENERIC":
        calibration_paths.insert(0, self.model_dir / self.instrument / "confidence_calibrator.pkl")
    
    for calib_path in calibration_paths:
        if calib_path.exists():
            try:
                self.calibrator = ConfidenceCalibrator.load(calib_path)
                self._calibration_loaded = True
                logger.info(f"✓ Confidence calibrator loaded from {calib_path}")
                logger.info(f"  Method: {self.calibrator.config.method}, Fitted: {self.calibrator.is_fitted}")
                return
            except Exception as e:
                logger.warning(f"Failed to load calibrator from {calib_path}: {e}")
    
    # Try loading from model metadata
    meta_path = self.model_dir / "modular_ensemble.meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            
            # Check for calibration data in metadata
            calib_data = meta.get('calibration')
            if calib_data and isinstance(calib_data, dict):
                # Create calibrator from metadata
                config = CalibrationConfig(
                    method=calib_data.get('method', self.config.calibration_method),
                    min_confidence_threshold=calib_data.get('min_threshold', 0.5),
                    max_confidence_threshold=calib_data.get('max_threshold', 0.95),
                )
                self.calibrator = ConfidenceCalibrator(config)
                
                # Restore fitted state from metadata if available
                if 'platt_params' in calib_data:
                    # Reconstruct Platt model from saved parameters
                    from sklearn.linear_model import LogisticRegression
                    self.calibrator.platt_model = LogisticRegression()
                    self.calibrator.platt_model.coef_ = np.array([calib_data['platt_params']['coef']])
                    self.calibrator.platt_model.intercept_ = np.array([calib_data['platt_params']['intercept']])
                    self.calibrator.platt_model.classes_ = np.array([0, 1])
                    self.calibrator.platt_model.is_fitted = True
                    self._calibration_loaded = True
                    logger.info(f"✓ Platt calibration loaded from metadata")
                
                if 'isotonic_params' in calib_data:
                    # Reconstruct Isotonic model from saved parameters
                    from sklearn.isotonic import IsotonicRegression
                    self.calibrator.isotonic_model = IsotonicRegression(out_of_bounds='clip')
                    self.calibrator.isotonic_model.X_thresholds_ = np.array(calib_data['isotonic_params']['X_thresholds'])
                    self.calibrator.isotonic_model.y_thresholds_ = np.array(calib_data['isotonic_params']['y_thresholds'])
                    self.calibrator.isotonic_model.f_ = None  # Will be rebuilt on predict
                    self.calibrator.is_fitted = True
                    self._calibration_loaded = True
                    logger.info(f"✓ Isotonic calibration loaded from metadata")
                
                return
        except Exception as e:
            logger.warning(f"Failed to load calibration from metadata: {e}")
    
    # Create default calibrator (unfitted) if calibration enabled but no data found
    if self.config.enable_calibration:
        config = CalibrationConfig(
            method=self.config.calibration_method,
            min_confidence_threshold=0.5,
            max_confidence_threshold=0.95,
        )
        self.calibrator = ConfidenceCalibrator(config)
        # Note: _calibration_loaded stays False, but calibrator.is_fitted is also False
        # _apply_calibration will check is_fitted and gracefully return raw probability
        self._calibration_loaded = False  # Explicitly set to indicate no fitted calibrator
        logger.info(f"ℹ Calibration enabled but no fitted calibrator found - using raw probabilities")
        logger.info(f"  To enable calibration, train with: python main.py train-buddy --calibrate")
        logger.info(f"  Or run: python -m confidence_calibration --train --model-dir {self.model_dir}")
```

**Calibration Application** (`src/core/modular_inference.py` Lines 1507-1540):
```python
def _apply_calibration(self, raw_probability: float, direction: Optional[int] = None) -> tuple[float, bool]:
    """Apply confidence calibration to raw model probability.
    
    # Check if calibrator exists and is fitted
    # We rely on calibrator.is_fitted rather than _calibration_loaded for robustness
    if self.calibrator is None:
        return raw_probability, False
    
    if not self.calibrator.is_fitted:
        # Calibrator exists but not trained - graceful fallback to raw probability
        logger.debug("Calibrator not fitted, using raw probability")
        return raw_probability, False
    
    try:
        result = self.calibrator.calibrate_confidence(raw_probability)
        calibrated = result.calibrated_confidence
        ...
        return calibrated, True
    except Exception as e:
        logger.warning(f"Calibration failed, using raw probability: {e}")
        return raw_probability, False
```

### 4.2 Root Cause Analysis

**PRIMARY ISSUE**: Calibration file does not exist, so calibrator is not fitted and raw probabilities are used

**Evidence 1**: The code checks for `confidence_calibrator.pkl` in:
   - `trained_data/models/confidence_calibrator.pkl` (generic)
   - `trained_data/models/{instrument}/confidence_calibrator.pkl` (pair-specific)

**Evidence 2**: No calibration data in `modular_ensemble.meta.json` (searched earlier).

**Evidence 3**: When calibrator is not fitted, `_apply_calibration()` returns raw probability with `was_calibrated=False`.

**Evidence 4**: In `predict()` (Lines 2788-2804), calibration is applied but may fail:
```python
if self.config.enable_calibration and tcn_probability is not None:
    tcn_probability, calibration_applied = self._apply_calibration(
            tcn_probability, 
            direction=tcn_direction
        )
    
    # Store calibration info in intel_data for reporting
    intel_data['calibration'] = {
        'enabled': self.config.enable_calibration,
        'applied': calibration_applied,
        'raw_probability': raw_tcn_probability,
        'calibrated_probability': tcn_probability,
        'method': self.calibrator.config.method if self.calibrator and self.calibrator.is_fitted else 'none',
        'calibrator_fitted': self.calibrator.is_fitted if self.calibrator else False,
        'adjustment': tcn_probability - raw_tcn_probability if calibration_applied else 0.0,
    }
```

### 4.3 Impact

**WHEN CALIBRATION FAILS**:
1. `self.calibrator.is_fitted = False`
2. `calibration_applied = False` (raw probability used)
3. `adjustment = 0.0` (no adjustment)
4. System uses uncalibrated raw probabilities
5. All probabilities are in [0, 1] range without proper calibration to P(win)

---

## 5. Root Cause Summary

### 5.1 Critical Failure Cascade

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                │
│  1. RL GATES FAIL TO LOAD (import/timeout)              │
│    ↓                                                            │
│    ↓                                                            │
│    ↓ 2. Ensemble Models FAIL TO LOAD (Keras incompatibility)       │
│    ↓                                                            │
│    ↓                                                            │
│    ↓ 3. Meta Model NOT LOADED (file missing)                  │
│    ↓                                                            │
│    ↓ 4. Calibration FAILS (file missing)                      │
│    ↓                                                            │
│    ↓                                                            │
│    ↓                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
    ↓
↓
```

---

## 6. Recommended Fixes

### 6.1 Fix 1: Add Timeout Protection to RL Gates Loading

**File**: `src/core/modular_inference.py`

**Change**: Add timeout protection to `_lazy_load_rl_gates()` and `_lazy_load_rl_exits()`

```python
# Lines 169-182: BEFORE (NO TIMEOUT PROTECTION)
def _lazy_load_rl_gates():
    """Lazy load GateThresholdRL to avoid TF/PyTorch GPU conflicts."""
    global GateThresholdRL, RL_GATES_AVAILABLE
    if GateThresholdRL is not None:
        return GateThresholdRL, RL_GATES_AVAILABLE
    
    try:
        from src.rl.gate_threshold_env import GateThresholdRL as _GateThresholdRL
        GateThresholdRL = _GateThresholdRL
        RL_GATES_AVAILABLE = True
        return GateThresholdRL, RL_GATES_AVAILABLE
    except ImportError:
        RL_GATES_AVAILABLE = False
        GateThresholdRL = None
        return None, False

# Lines 169-182: AFTER (WITH TIMEOUT PROTECTION)
def _lazy_load_rl_gates():
    """Lazy load GateThresholdRL to avoid TF/PyTorch GPU conflicts."""
    global GateThresholdRL, RL_GATES_AVAILABLE
    if GateThresholdRL is not None:
        return GateThresholdRL, RL_GATES_AVAILABLE
    
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_lazy_load_rl_gates_unsafe)
            try:
                _GateThresholdRL, _ = future.result(timeout=RL_LOAD_TIMEOUT_SECONDS)
                RL_GATES_AVAILABLE = True
                return GateThresholdRL, RL_GATES_AVAILABLE
            except FutureTimeout:
                logger.warning(f"RL gates import timed out after {RL_LOAD_TIMEOUT_SECONDS}s")
                RL_GATES_AVAILABLE = False
                GateThresholdRL = None
                return None, False
    except ImportError:
        RL_GATES_AVAILABLE = False
        GateThresholdRL = None
        return None, False
```

**Apply same timeout protection to `_lazy_load_rl_exits()` (Lines 185-198)**

---

### 6.2 Fix 2: Improve Error Handling and Logging for RL Loading

**File**: `src/core/modular_inference.py`

**Change**: Add detailed error logging and fallback logic

```python
# Lines 1229-1257: IMPROVED RL GATES LOADING
if self.use_rl_gates:
    GateRL, gates_available = _lazy_load_rl_gates()
    
    if gates_available and GateRL is not None:
        self.rl_gate_optimizer = GateRL()
        if self.rl_gate_optimizer.load():
            logger.info("✓ RL Gate Threshold Optimizer loaded (SAC)")
            self._rl_gates_loaded = True
        else:
            logger.warning("⚠️ RL Gates not trained - using fixed thresholds")
            self.rl_gate_optimizer = None
    else:
        logger.warning("⚠️ RL Gates requested but dependencies not available")
        self.rl_gate_optimizer = None
```

**Add logging to track load failures**:
```python
# Lines 1229-1257: WITH DETAILED LOGGING
if self.use_rl_gates:
    try:
        GateRL, gates_available = _lazy_load_rl_gates()
        
        if gates_available and GateRL is not None:
            self.rl_gate_optimizer = GateRL()
            if self.rl_gate_optimizer.load():
                logger.info("✓ RL Gate Threshold Optimizer loaded (SAC)")
                self._rl_gates_loaded = True
            else:
                logger.warning("⚠️ RL Gates not trained - using fixed thresholds")
                self.rl_gate_optimizer = None
        elif not gates_available:
            logger.error("❌ RL Gates import failed - RL gates unavailable")
            self._rl_gate_optimizer = None
            self._rl_gates_loaded = False
    except Exception as e:
        logger.error(f"❌ RL Gates loading exception: {type(e).__name__}: {e}")
        self.rl_gate_optimizer = None
        self._rl_gates_loaded = False
```

---

### 6.3 Fix 3: Add Model Loading Retry Logic

**File**: `src/core/modular_inference.py`

**Change**: Add retry mechanism for failed model loads

```python
# Lines 1229-1257: WITH RETRY LOGIC
if self.use_rl_gates:
    max_retries = 2
    for attempt in range(max_retries):
        GateRL, gates_available = _lazy_load_rl_gates()
        
        if gates_available and GateRL is not None:
            self.rl_gate_optimizer = GateRL()
            if self.rl_gate_optimizer.load():
                logger.info("✓ RL Gate Threshold Optimizer loaded (SAC)")
                self._rl_gates_loaded = True
                break  # Success
        elif not gates_available:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ RL Gates import failed, retrying... ({attempt}/{max_retries})")
                time.sleep(1)  # Wait before retry
            else:
                logger.error(f"❌ RL Gates import failed after {max_retries} attempts")
                self.rl_gate_optimizer = None
                self._rl_gates_loaded = False
```

---

### 6.4 Fix 4: Create Meta-Labeler Model File

**Action**: Train and save meta-labeler model

**Command**:
```bash
# Run buddy training with meta-labeling enabled
python main.py train-buddy --calibrate --instruments EUR_USD,GBP_USD,USD_JPY --candles 10000 --epochs 50
```

**Alternative**: Create placeholder file if training is not possible

```python
# Create placeholder meta-labeler.pkl
python3 -c "
import pickle
from pathlib import Path

# Create minimal meta-labeler
class MinimalMetaLabeler:
    def __init__(self):
        self._primary_accuracy = 0.5
    
    def predict(self, X):
        # Always return neutral confidence
        return 0.5
    
    def predict_meta_confidence(self, X, primary_prob):
        # Return neutral confidence
        return 0.55
    
    def load(self, path):
        # Minimal loading - just set flag
        return self

# Save placeholder
meta = MinimalMetaLabeler()
with open('trained_data/models/meta_labeler.pkl', 'wb') as f:
    pickle.dump(meta, f)
"
```

---

### 6.5 Fix 5: Train Calibration Model

**File**: `src/core/modular_inference.py`

**Change**: Add calibration training command

**Command**:
```bash
# Run buddy training with calibration enabled
python main.py train-buddy --calibrate --instruments EUR_USD,GBP_USD,USD_JPY --candles 5000 --epochs 100
```

**Alternative**: Use standalone calibration module

```bash
python -m confidence_calibration --train --model-dir trained_data/models --method platt --candles 5000
```

---

## 7. Verification Steps

### 7.1 Verify RL Models Load
```bash
# Test RL model loading
python3 -c "
from src.core.modular_inference import ModularEnsembleInference
import logging
logging.basicConfig(level=logging.INFO)

ensemble = ModularEnsembleInference()
ensemble.load_models()

# Check RL status
print(f'RL Gates Loaded: {ensemble._rl_gates_loaded}')
print(f'RL Gates Optimizer: {ensemble.rl_gate_optimizer is not None}')
print(f'RL Exits Loaded: {ensemble._rl_exits_loaded}')
print(f'RL Position Sizer: {ensemble.rl_sizer is not None}')
"
```

### 7.2 Verify Ensemble Models Load
```bash
# Test ensemble model loading
python3 -c "
from src.core.modular_inference import ModularEnsembleInference
import logging
logging.basicConfig(level=logging.INFO)

ensemble = ModularEnsembleInference(instrument='EUR_USD')
ensemble.load_models()

print(f'Transfer/Direction Model: {ensemble.tcn is not None}')
print(f'LightGBM Momentum: {ensemble.xgb is not None}')
print(f'LightGBM Risk: {ensemble.rf is not None}')
print(f'Ridge Confidence: {ensemble.ridge is not None}')
"
```

### 7.3 Verify Calibration Load
```bash
# Test calibration loading
python3 -c "
from src.core.modular_inference import ModularEnsembleInference
import logging
logging.basicConfig(level=logging.INFO)

ensemble = ModularEnsembleInference()
ensemble.load_models()

status = ensemble.get_calibration_status()
print(f'Calibration Enabled: {status[\"enabled\"]}')
print(f'Calibrator Exists: {status[\"calibrator_exists\"]}')
print(f'Calibrator Fitted: {status[\"calibrator_fitted\"]}')
print(f'Calibration Method: {status[\"method\"]}')
print(f'Calibration Source: {status[\"source\"]}')
"
```

### 7.4 Verify Meta-Labeler Load
```bash
# Test meta-labeler loading
python3 -c "
from src.core.modular_inference import ModularEnsembleInference
import logging
logging.basicConfig(level=logging.INFO)

ensemble = ModularEnsembleInference(instrument='EUR_USD')
ensemble.load_models()

print(f'Meta-Labeler Loaded: {ensemble._meta_labeler_loaded}')
print(f'Meta-Labeler Available: {ensemble.meta_labeler is not None}')
"
```

---

## 8. Priority Order

1. **CRITICAL**: Fix RL GATES timeout protection (immediate - prevents hangs)
2. **HIGH**: Fix ensemble model loading (Keras compatibility)
3. **HIGH**: Create meta-labeler model file
4. **HIGH**: Train calibration model (enables proper P(win) calibration)
5. **MEDIUM**: Add RL loading retry logic (graceful degradation)
6. **LOW**: Add comprehensive logging (diagnostics)

---

## 9. Additional Notes

### 9.1 System Logs Location
The only available log file is `logs/retrain_20260129_163352.log`, which contains only training dry run output. There are NO inference logs available to debug.

### 9.2 Missing Inference Logs
The system needs to log inference failures to:
1. Model loading attempts
2. Calibration application results
3. Gate evaluation results
4. Meta-labeler predictions

### 9.3 Recommended Logging
Add the following to `src/core/modular_inference.py`:
```python
# At the top of the file, after imports
import logging
from datetime import datetime

# Create inference logger with file handler
inference_logger = logging.getLogger('inference')
inference_logger.setLevel(logging.INFO)

# Create file handler for inference logs
log_dir = Path('logs')
log_file = log_dir / f'inference_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.log'

# Add file handler to inference logger
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
inference_logger.addHandler(file_handler)

# Log model loading with timestamps
def log_model_load(model_name: str, success: bool, details: str = ''):
    inference_logger.info(
        f\"[{datetime.now().isoformat()}] MODEL_LOAD: {model_name}: {'SUCCESS' if success else 'FAILED'} - {details}\"
    )

# Log calibration events
def log_calibration_event(event_type: str, details: str = ''):
    inference_logger.info(
        f\"[{datetime.now().isoformat()}] CALIBRATION: {event_type} - {details}\"
    )

# Log gate evaluation
def log_gate_evaluation(gate_name: str, passed: bool, value: float, threshold: float):
    status = 'PASSED' if passed else 'FAILED'
    inference_logger.info(
        f\"[{datetime.now().isoformat()}] GATE_EVAL: {gate_name}: {status} - value={value:.3f}, threshold={threshold:.3f}\"
    )
```

Then use these loggers in the relevant methods.
```

---

## 10. Summary

The inference failures are caused by a combination of:
1. **Missing calibration file** - `confidence_calibrator.pkl` does not exist
2. **Missing meta-labeler file** - `meta_labeler.pkl` does not exist
3. **Keras version incompatibility** - Models may have been saved with different Keras version
4. **RL import failures** - No timeout protection or retry logic
5. **Insufficient error logging** - No inference logs to debug issues

The fixes above address all identified issues and will restore the system to producing valid trading signals.
