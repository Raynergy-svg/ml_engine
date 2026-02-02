# Model Training Pipeline Consistency Bug Fix Plan

## Executive Summary

This document describes a critical consistency bug in the model training pipeline where initialization, data fetching, training, and saving operations are misaligned. The bug causes TCN and LightGBM models to be excluded from data loading and training, and causes incorrect model-to-output mappings during saving, which can lead to financial errors.

## Bug Analysis

### Current State: Inconsistent Model Handling

#### 1. Initialization List (from `train_all_modular`)
The training function initializes and trains these models:
- **TransformerDirectionTrainer** or **TCNTrainer** (for direction)
- **XGBoostTrainer** (for momentum)
- **RandomForestTrainer** (for risk)
- **RidgeTrainer** (for confidence - now uses LightGBM as primary)
- **HistGradientBoostingDirectionTrainer** (optional, for hybrid voting)

#### 2. Data Fetching Logic (from `load_all_modular_data`)
The data loading function returns:
```python
{
    'xgboost': load_xgboost_data(...),      # ✓
    'rf': load_rf_data(...),                 # ✓
    'ridge': load_ridge_data(...),           # ✓
    'direction': load_direction_data(...),     # ✓
    'tcn': direction_data,                  # ✗ Alias only
}
```

**Issues:**
- No explicit `load_tcn_data()` function (TCN is aliased to direction)
- No explicit `load_lightgbm_data()` function (RidgeTrainer uses LightGBM)
- Missing keys: `transformer`, `lightgbm`, `histgb`

#### 3. Training Stage (from `train_all_modular`)
Training uses data keys:
```python
# Direction/Regime
data['direction'] or data['tcn']  # Works but confusing

# Gate models
data['xgboost']  # ✓
data['rf']  # ✓
data['ridge']  # ✓ (but uses LightGBM in trainer)

# Hybrid voting
data['direction']  # ✓ (reused)
```

**Issues:**
- TCN and LightGBM are not explicitly trained with dedicated data
- RidgeTrainer uses LightGBM internally but data loader doesn't reflect this
- No clear mapping between model type and data key

#### 4. Saving Stage (from `train_all_modular`)
Models are saved with these filenames:
```python
# Direction/Regime models
'transformer_direction.keras'  # ✓
'tcn_direction.keras'  # ✓
'transformer_regime.keras'  # ✓
'histgb_direction.pkl'  # ✓

# Gate models
'xgb_momentum.pkl'  # ✓
'rf_risk.pkl'  # ✓
'ridge_confidence.pkl'  # ✗ Should be 'lightgbm_confidence.pkl'
```

**Issues:**
- Filename `ridge_confidence.pkl` doesn't reflect that LightGBM is used
- No explicit `lightgbm_confidence.pkl` file
- Inference expects `ridge_confidence.pkl` but model is actually LightGBM

#### 5. Inference Loading (from `load_models`)
Models are loaded with these paths:
```python
'transformer_regime.keras'  # ✓
'transformer_direction.keras'  # ✓
'tcn_direction.keras'  # ✓
'histgb_direction.pkl'  # ✓
'xgb_momentum.pkl'  # ✓
'rf_risk.pkl'  # ✓
'ridge_confidence.pkl'  # ✗ Loads RidgeTrainer which uses LightGBM
```

**Issues:**
- `ridge_confidence.pkl` filename is misleading (model is LightGBM)
- No `lightgbm_confidence.pkl` path checked

## Root Cause

The inconsistency stems from:
1. **No unified model configuration**: Model definitions are scattered across multiple files
2. **Implicit model aliases**: TCN is aliased to direction, LightGBM is hidden inside RidgeTrainer
3. **Inconsistent naming**: Filenames don't match actual model implementations
4. **Missing explicit data loaders**: No dedicated loaders for TCN and LightGBM

## Solution Design

### 1. Create Unified Model Configuration

Define a single source of truth for all models:

```python
# src/training/model_config.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for a single model in the ensemble."""
    name: str  # 'transformer_direction', 'tcn_direction', etc.
    model_type: str  # 'transformer', 'tcn', 'xgboost', 'rf', 'lightgbm', 'ridge'
    task: str  # 'direction', 'regime', 'momentum', 'risk', 'confidence'
    data_key: str  # 'direction', 'tcn', 'xgboost', 'rf', 'lightgbm'
    trainer_class: type  # Trainer class
    data_loader_func: callable  # Data loading function
    save_extension: str  # '.keras' or '.pkl'
    enabled: bool = True
    priority: int = 0  # Lower = higher priority

# Define all models in one place
MODEL_REGISTRY = {
    'transformer_direction': ModelConfig(
        name='transformer_direction',
        model_type='transformer',
        task='direction',
        data_key='direction',
        trainer_class=TransformerDirectionTrainer,
        data_loader_func=load_direction_data,
        save_extension='.keras',
        enabled=True,
        priority=1,
    ),
    'tcn_direction': ModelConfig(
        name='tcn_direction',
        model_type='tcn',
        task='direction',
        data_key='tcn',
        trainer_class=TCNTrainer,
        data_loader_func=load_tcn_data,  # NEW: dedicated loader
        save_extension='.keras',
        enabled=True,
        priority=2,
    ),
    'transformer_regime': ModelConfig(
        name='transformer_regime',
        model_type='transformer',
        task='regime',
        data_key='regime',
        trainer_class=TransformerRegimeTrainer,
        data_loader_func=load_regime_data,
        save_extension='.keras',
        enabled=True,
        priority=3,
    ),
    'xgboost_momentum': ModelConfig(
        name='xgboost_momentum',
        model_type='xgboost',
        task='momentum',
        data_key='xgboost',
        trainer_class=XGBoostTrainer,
        data_loader_func=load_xgboost_data,
        save_extension='.pkl',
        enabled=True,
        priority=4,
    ),
    'rf_risk': ModelConfig(
        name='rf_risk',
        model_type='random_forest',
        task='risk',
        data_key='rf',
        trainer_class=RandomForestTrainer,
        data_loader_func=load_rf_data,
        save_extension='.pkl',
        enabled=True,
        priority=5,
    ),
    'lightgbm_confidence': ModelConfig(
        name='lightgbm_confidence',
        model_type='lightgbm',
        task='confidence',
        data_key='lightgbm',
        trainer_class=LightGBMTrainer,  # NEW: dedicated trainer
        data_loader_func=load_lightgbm_data,  # NEW: dedicated loader
        save_extension='.pkl',
        enabled=True,
        priority=6,
    ),
    'ridge_confidence': ModelConfig(
        name='ridge_confidence',
        model_type='ridge',
        task='confidence',
        data_key='ridge',
        trainer_class=RidgeTrainer,
        data_loader_func=load_ridge_data,
        save_extension='.pkl',
        enabled=False,  # Disabled in favor of LightGBM
        priority=7,
    ),
    'histgb_direction': ModelConfig(
        name='histgb_direction',
        model_type='histgradientboosting',
        task='direction',
        data_key='histgb',
        trainer_class=HistGradientBoostingDirectionTrainer,
        data_loader_func=load_direction_data,  # Reuses direction data
        save_extension='.pkl',
        enabled=False,  # Optional hybrid voting
        priority=8,
    ),
}
```

### 2. Create Dedicated Data Loaders

Add explicit data loaders for TCN and LightGBM:

```python
# src/core/modular_data_loaders.py

def load_tcn_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    lookahead: int = 6,
    threshold: float = 0.001,
) -> Dict[str, np.ndarray]:
    """
    Load data for TCN model (explicit loader, not an alias).
    
    Uses same features as direction model but with TCN-specific preprocessing.
    """
    # Reuse direction data loader with TCN-specific parameters
    return load_direction_data(df, split, lookahead, threshold)

def load_lightgbm_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    confidence_window: int = 10,
    instrument: Optional[str] = None,
    include_instrument_features: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load data for LightGBM confidence model.
    
    Features: Same as Ridge (confidence features)
    Target: Confidence score (0-100)
    """
    # Reuse ridge data loader (same features and targets)
    return load_ridge_data(df, split, confidence_window, instrument, include_instrument_features)
```

### 3. Update Data Loading Orchestration

Modify `load_all_modular_data` to use MODEL_REGISTRY:

```python
def load_all_modular_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    direction_threshold: float = 0.005,
    direction_lookahead: int = 6,
    use_regime: bool = False,
    regime_lookback: int = 20,
    regime_lookahead: int = 12,
    models_to_load: Optional[List[str]] = None,  # NEW: selective loading
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load data for all enabled models using unified configuration.
    
    Args:
        df: DataFrame with OHLCV and features
        split: (train_frac, val_frac, test_frac)
        direction_threshold: Min price change for clear direction labels
        direction_lookahead: Bars ahead for direction prediction
        use_regime: If True, load regime data instead of direction data
        regime_lookback: Bars to look back for regime detection
        regime_lookahead: Bars ahead to confirm regime
        models_to_load: List of model names to load (None = all enabled)
    
    Returns:
        Dict mapping model names to their data dictionaries
    """
    from src.training.model_config import MODEL_REGISTRY
    
    # Compute normalized features first
    logger.info("Computing normalized features for instrument-agnostic training...")
    df_normalized = compute_normalized_features(df)
    
    # Apply FeatureEngineering
    try:
        from src.data.feature_engineering import FeatureEngineering
        fe = FeatureEngineering({})
        df_fe = fe.create_features(df.copy(), include_all=True, apply_candle_smoothing=False)
        new_cols = [c for c in df_fe.columns if c not in df_normalized.columns]
        if new_cols:
            df_fe_aligned = df_fe[new_cols].reindex(df_normalized.index)
            df_normalized = pd.concat([df_normalized, df_fe_aligned], axis=1)
            logger.info(f"Added {len(new_cols)} features from FeatureEngineering")
    except Exception as e:
        logger.warning(f"FeatureEngineering failed: {e}")
    
    # Clean NaN/inf
    df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    
    # Determine which models to load
    if models_to_load is None:
        models_to_load = [name for name, config in MODEL_REGISTRY.items() if config.enabled]
    
    # Load data for each model
    result = {}
    for model_name in models_to_load:
        config = MODEL_REGISTRY[model_name]
        
        # Skip if data loader not available
        if config.data_loader_func is None:
            logger.warning(f"No data loader for {model_name}, skipping")
            continue
        
        # Load data with model-specific parameters
        try:
            if model_name == 'transformer_regime':
                data = config.data_loader_func(
                    df_normalized, split, regime_lookback, regime_lookahead
                )
            elif model_name in ['transformer_direction', 'tcn_direction']:
                data = config.data_loader_func(
                    df_normalized, split, direction_lookahead, direction_threshold
                )
            else:
                # Default: no extra params
                data = config.data_loader_func(df_normalized, split)
            
            result[config.data_key] = data
            logger.info(f"✓ Loaded data for {model_name} ({len(data['X_train'])} train samples)")
        except Exception as e:
            logger.error(f"Failed to load data for {model_name}: {e}")
            result[config.data_key] = None
    
    # Handle direction/regime exclusivity
    if use_regime:
        if 'regime' in result:
            logger.info("Using REGIME classification mode")
        else:
            logger.warning("Regime data requested but not loaded")
    else:
        if 'direction' in result:
            logger.info("Using DIRECTION prediction mode")
        elif 'tcn' in result:
            logger.info("Using TCN direction mode")
    
    return result
```

### 4. Update Training Orchestration

Modify `train_all_modular` to use MODEL_REGISTRY:

```python
def train_all_modular(
    df: pd.DataFrame,
    model_dir: str = "trained_data/models",
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    direction_threshold: float = 0.005,
    direction_lookahead: int = 6,
    use_regime: bool = False,
    regime_lookback: int = 20,
    regime_lookahead: int = 12,
    models_to_train: Optional[List[str]] = None,  # NEW: selective training
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Train all enabled models using unified configuration.
    
    Args:
        df: DataFrame with OHLCV and features
        model_dir: Directory to save models
        split: (train_frac, val_frac, test_frac)
        direction_threshold: Min price change for clear direction labels
        direction_lookahead: Bars ahead for direction prediction
        use_regime: If True, train regime model instead of direction
        regime_lookback: Bars to look back for regime detection
        regime_lookahead: Bars ahead to confirm regime
        models_to_train: List of model names to train (None = all enabled)
        force_retrain: If True, retrain even if model exists
    
    Returns:
        Dict with training results for each model
    """
    from src.training.model_config import MODEL_REGISTRY
    
    # Load data for all models
    logger.info("Loading data for all models...")
    data = load_all_modular_data(
        df, split, direction_threshold, direction_lookahead,
        use_regime, regime_lookback, regime_lookahead,
        models_to_load=models_to_train,
    )
    
    # Determine which models to train
    if models_to_train is None:
        models_to_train = [name for name, config in MODEL_REGISTRY.items() if config.enabled]
    
    # Train each model
    results = {}
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    
    for model_name in models_to_train:
        config = MODEL_REGISTRY[model_name]
        
        # Skip if data not available
        data_key = config.data_key
        if data_key not in data or data[data_key] is None:
            logger.warning(f"No data available for {model_name}, skipping")
            continue
        
        model_data = data[data_key]
        
        # Check if model already exists
        model_path = model_dir_path / f"{config.name}{config.save_extension}"
        if model_path.exists() and not force_retrain:
            logger.info(f"Model {config.name} already exists, skipping (use force_retrain=True)")
            continue
        
        # Initialize trainer
        trainer = config.trainer_class()
        
        # Train model
        logger.info(f"Training {config.name}...")
        try:
            if model_name in ['transformer_direction', 'tcn_direction', 'transformer_regime']:
                # Direction/Regime models
                if config.task == 'regime':
                    result = trainer.train(
                        X_train=model_data['X_train'],
                        y_train=model_data['y_train'],
                        X_val=model_data['X_val'],
                        y_val=model_data['y_val'],
                        epochs=50,
                        batch_size=32,
                    )
                else:
                    result = trainer.train(
                        X_train=model_data['X_train'],
                        y_train=model_data['y_train'],
                        w_train=model_data.get('w_train'),  # Sample weights
                        X_val=model_data['X_val'],
                        y_val=model_data['y_val'],
                        w_val=model_data.get('w_val'),
                        epochs=50,
                        batch_size=32,
                    )
            else:
                # Gate models (XGBoost, RF, LightGBM, Ridge)
                result = trainer.train(
                    X_train=model_data['X_train'],
                    y_train=model_data['y_train'],
                    X_val=model_data['X_val'],
                    y_val=model_data['y_val'],
                )
            
            # Save model
            trainer.save(str(model_path))
            logger.info(f"✓ Saved {config.name} to {model_path}")
            
            results[model_name] = {
                'status': 'success',
                'metrics': result,
                'path': str(model_path),
            }
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            results[model_name] = {
                'status': 'error',
                'error': str(e),
            }
    
    # Save ensemble metadata
    save_ensemble_metadata(model_dir, results, MODEL_REGISTRY)
    
    return results
```

### 5. Fix Model Saving Logic

Ensure correct model-to-output mapping:

```python
def save_ensemble_metadata(
    model_dir: Path,
    results: Dict[str, Any],
    model_registry: Dict[str, ModelConfig],
) -> None:
    """
    Save ensemble metadata with correct model-to-output mapping.
    
    This ensures inference knows which models are available and what they predict.
    """
    import json
    
    meta = {
        'version': '2.0',
        'trained_at': datetime.utcnow().isoformat(),
        'models': {},
        'ensemble_mode': 'regime' if any('regime' in r for r in results) else 'direction',
    }
    
    for model_name, config in model_registry.items():
        if model_name not in results:
            continue
        
        result = results[model_name]
        if result['status'] != 'success':
            continue
        
        meta['models'][model_name] = {
            'name': config.name,
            'model_type': config.model_type,
            'task': config.task,
            'output': _get_output_for_task(config.task),
            'path': result['path'],
            'metrics': result.get('metrics', {}),
        }
    
    meta_path = model_dir / "modular_ensemble.meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Saved ensemble metadata to {meta_path}")

def _get_output_for_task(task: str) -> str:
    """Get output name for a given task."""
    outputs = {
        'direction': 'direction',
        'regime': 'regime',
        'momentum': 'momentum_score',
        'risk': 'expected_drawdown_pct',
        'confidence': 'confidence_score',
    }
    return outputs.get(task, 'unknown')
```

### 6. Update Inference Loading

Use MODEL_REGISTRY for consistent loading:

```python
def load_models(self, instrument: Optional[str] = None) -> None:
    """
    Load all models using unified configuration.
    
    Args:
        instrument: Optional instrument (e.g., 'EUR_USD') to load pair-specific models.
                   If None, uses self.instrument or loads generic models.
    """
    from src.training.model_config import MODEL_REGISTRY
    
    # Update instrument if provided
    if instrument:
        self.instrument = instrument
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    warnings.filterwarnings('ignore', message='.*serialized model.*')
    
    pair_info = f" for {self.instrument}" if self.instrument and self.instrument != "GENERIC" else ""
    logger.info(f"Loading modular ensemble models{pair_info}...")
    
    # Load models based on registry
    for model_name, config in MODEL_REGISTRY.items():
        if not config.enabled:
            continue
        
        model_path = self._get_model_path(config.name, config.save_extension)
        
        if not model_path.exists():
            logger.debug(f"{config.name} not found at {model_path}")
            continue
        
        try:
            # Initialize trainer
            trainer = config.trainer_class()
            
            # Load model
            trainer.load(str(model_path))
            
            # Store in appropriate attribute
            if config.task == 'direction':
                if config.model_type == 'transformer':
                    self.tcn = trainer
                    self.use_regime = False
                elif config.model_type == 'tcn':
                    self.tcn = trainer
                    self.use_regime = False
            elif config.task == 'regime':
                self.regime_model = trainer
                self.use_regime = True
            elif config.task == 'momentum':
                self.xgb = trainer
            elif config.task == 'risk':
                self.rf = trainer
            elif config.task == 'confidence':
                if config.model_type == 'lightgbm':
                    self.ridge = trainer  # Store in ridge attribute for compatibility
                elif config.model_type == 'ridge':
                    self.ridge = trainer
            elif config.task == 'direction' and config.model_type == 'histgradientboosting':
                self.histgb = trainer
                self.use_hybrid = True
            
            logger.info(f"✓ {config.name} loaded from {model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load {config.name}: {e}")
    
    self._loaded = True
    self._loaded_instrument = self.instrument
    logger.info(f"Modular ensemble loaded{pair_info}.")
```

## Implementation Roadmap

### Phase 1: Configuration Architecture
**Objective**: Create unified model configuration as single source of truth

**Files to Create/Modify**:
- `src/training/model_config.py` (NEW)

**Actionable Steps**:
1. Create `ModelConfig` dataclass with fields: name, model_type, task, data_key, trainer_class, data_loader_func, save_extension, enabled, priority
2. Define `MODEL_REGISTRY` dictionary with all model configurations:
   - `transformer_direction`
   - `tcn_direction`
   - `transformer_regime`
   - `xgboost_momentum`
   - `rf_risk`
   - `lightgbm_confidence` (NEW)
   - `ridge_confidence` (deprecated, disabled)
   - `histgb_direction`
3. Add helper functions: `get_enabled_models()`, `get_models_by_task()`, `get_model_config()`
4. Add validation function: `validate_model_registry()` to check for missing trainers/loaders

**Success Criteria**:
- [ ] `ModelConfig` dataclass defined
- [ ] `MODEL_REGISTRY` contains all 8 model configurations
- [ ] Helper functions implemented
- [ ] Validation function passes

---

### Phase 2: Data Loader Modifications
**Objective**: Add explicit data loaders for TCN and LightGBM, update orchestration

**Files to Modify**:
- `src/core/modular_data_loaders.py`

**Actionable Steps**:

#### Step 2.1: Add TCN Data Loader
1. Create `load_tcn_data()` function (lines ~1340-1350)
2. Reuse `load_direction_data()` with TCN-specific parameters
3. Return data dict with keys: X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test, feature_names, label_stats, scaler

#### Step 2.2: Add LightGBM Data Loader
1. Create `load_lightgbm_data()` function (lines ~1720-1730)
2. Reuse `load_ridge_data()` (same features and targets)
3. Return data dict with keys: X_train, y_train, X_val, y_val, X_test, y_test, feature_names, adx_p25, adx_p75, instrument

#### Step 2.3: Update `load_all_modular_data()`
1. Import `MODEL_REGISTRY` from `src.training.model_config`
2. Add parameter `models_to_load: Optional[List[str]] = None`
3. Replace hardcoded model list with registry-based loop
4. For each model in registry:
   - Call `config.data_loader_func()` with appropriate parameters
   - Store result in `result[config.data_key]`
5. Handle direction/regime exclusivity (only one enabled at a time)
6. Add logging for each loaded model

**Success Criteria**:
- [ ] `load_tcn_data()` function created
- [ ] `load_lightgbm_data()` function created
- [ ] `load_all_modular_data()` uses MODEL_REGISTRY
- [ ] Selective loading works
- [ ] All data keys match registry data_key values

---

### Phase 3: Training Orchestration
**Objective**: Update training to use unified configuration with correct model-to-output mapping

**Files to Modify**:
- `src/training/modular_trainers.py`

**Actionable Steps**:

#### Step 3.1: Update `train_all_modular()` Signature
1. Add parameter `models_to_train: Optional[List[str]] = None`
2. Add parameter `force_retrain: bool = False`

#### Step 3.2: Import and Use MODEL_REGISTRY
1. Import `MODEL_REGISTRY` from `src.training.model_config`
2. Call `load_all_modular_data()` with `models_to_load=models_to_train`

#### Step 3.3: Train Models Using Registry
1. Replace hardcoded model training with registry loop
2. For each model in registry:
   - Check if data available: `data[config.data_key]`
   - Check if model exists and not `force_retrain`
   - Initialize trainer: `trainer = config.trainer_class()`
   - Train with appropriate parameters based on task
   - Save to: `model_dir / f"{config.name}{config.save_extension}"`
3. Collect results in dict with model names as keys

#### Step 3.4: Update Model Saving Logic
1. Ensure correct filename mapping:
   - `transformer_direction.keras`
   - `tcn_direction.keras`
   - `transformer_regime.keras`
   - `xgb_momentum.pkl`
   - `rf_risk.pkl`
   - `lightgbm_confidence.pkl` (NOT `ridge_confidence.pkl`)
   - `histgb_direction.pkl`

#### Step 3.5: Add Ensemble Metadata Saving
1. Create `save_ensemble_metadata()` function
2. Save to: `model_dir / "modular_ensemble.meta.json"`
3. Include: version, trained_at, models dict (name, model_type, task, output, path, metrics), ensemble_mode
4. Map tasks to outputs: direction→direction, regime→regime, momentum→momentum_score, risk→expected_drawdown_pct, confidence→confidence_score

**Success Criteria**:
- [ ] Training uses MODEL_REGISTRY
- [ ] Selective training works
- [ ] Models saved with correct filenames
- [ ] `lightgbm_confidence.pkl` saved (not `ridge_confidence.pkl`)
- [ ] Ensemble metadata saved with correct mappings

---

### Phase 4: Inference Updates
**Objective**: Update model loading to use unified configuration

**Files to Modify**:
- `src/core/modular_inference.py`

**Actionable Steps**:

#### Step 4.1: Update `load_models()` Method
1. Import `MODEL_REGISTRY` from `src.training.model_config`
2. Replace hardcoded model loading with registry loop
3. For each model in registry:
   - Get path: `self._get_model_path(config.name, config.save_extension)`
   - Initialize trainer: `trainer = config.trainer_class()`
   - Load model: `trainer.load(str(model_path))`
   - Store in appropriate attribute based on task

#### Step 4.2: Handle LightGBM Correctly
1. When loading `lightgbm_confidence`, store in `self.ridge` attribute
2. Add comment explaining compatibility with existing inference code

#### Step 4.3: Update Feature Extraction
1. Ensure `_extract_tcn_features()` works for both Transformer and TCN
2. Ensure `_extract_xgb_features()` loads correct features
3. Ensure `_extract_rf_features()` loads correct features
4. Ensure `_extract_ridge_features()` loads correct features for LightGBM

#### Step 4.4: Load Ensemble Metadata
1. Add `_load_ensemble_metadata()` method
2. Load from: `model_dir / "modular_ensemble.meta.json"`
3. Use metadata to determine available models and their tasks
4. Fall back to file existence check if metadata not available

**Success Criteria**:
- [ ] Loading uses MODEL_REGISTRY
- [ ] LightGBM loaded correctly
- [ ] All models loaded with correct paths
- [ ] Feature extraction works for all models
- [ ] Ensemble metadata loaded

---

### Phase 5: Testing
**Objective**: Verify consistency across initialization, data loading, training, and saving

**Files to Create**:
- `tests/test_model_consistency.py` (NEW)

**Actionable Steps**:

#### Step 5.1: Test MODEL_REGISTRY
1. Test `ModelConfig` dataclass instantiation
2. Test `MODEL_REGISTRY` contains all expected models
3. Test `get_enabled_models()` returns correct list
4. Test `get_models_by_task()` filters correctly
5. Test `validate_model_registry()` catches missing trainers/loaders

#### Step 5.2: Test Data Loading
1. Test `load_tcn_data()` returns expected structure
2. Test `load_lightgbm_data()` returns expected structure
3. Test `load_all_modular_data()` loads all enabled models
4. Test selective loading with `models_to_load` parameter
5. Test data keys match registry data_key values

#### Step 5.3: Test Training Orchestration
1. Test `train_all_modular()` trains all enabled models
2. Test selective training with `models_to_train` parameter
3. Test `force_retrain` re-trains existing models
4. Test models saved with correct filenames
5. Test ensemble metadata saved with correct mappings

#### Step 5.4: Test Inference Loading
1. Test `load_models()` loads all available models
2. Test LightGBM loaded correctly
3. Test ensemble metadata loaded
4. Test inference works with all models loaded
5. Test fallback to file existence if metadata missing

#### Step 5.5: Integration Test
1. Create end-to-end test: data loading → training → saving → inference
2. Verify TCN trained and loaded correctly
3. Verify LightGBM trained and loaded correctly
4. Verify all models saved with correct filenames
5. Verify model-to-output mapping correct

**Success Criteria**:
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] TCN explicitly trained and loaded
- [ ] LightGBM explicitly trained and loaded
- [ ] All models saved with correct filenames
- [ ] Model-to-output mapping verified

---

### Phase 6: Documentation
**Objective**: Document changes and provide migration guide

**Files to Create/Modify**:
- `README.md` (update)
- `docs/MODEL_CONSISTENCY_FIX.md` (NEW)
- `docs/MIGRATION_GUIDE.md` (NEW)

**Actionable Steps**:

#### Step 6.1: Update README
1. Document new MODEL_REGISTRY architecture
2. Update training command examples
3. Add section on selective model training
4. Update model list with LightGBM

#### Step 6.2: Create Fix Documentation
1. Document the bug and its symptoms
2. Explain the solution architecture
3. Provide code examples
4. Include diagrams of data flow

#### Step 6.3: Create Migration Guide
1. Document backup procedure
2. Provide retraining instructions
3. Document manual migration option (not recommended)
4. Provide troubleshooting steps

#### Step 6.4: Update API Documentation
1. Document `load_all_modular_data()` parameters
2. Document `train_all_modular()` parameters
3. Document `load_models()` behavior
4. Document MODEL_REGISTRY structure

**Success Criteria**:
- [ ] README updated
- [ ] Fix documentation created
- [ ] Migration guide created
- [ ] API documentation updated

## Migration Guide

### For Existing Deployments

1. **Backup current models**:
   ```bash
   cp -r trained_data/models trained_data/models.backup
   ```

2. **Update code**:
   ```bash
   git pull  # Get updated code
   pip install -r requirements.txt  # Install any new dependencies
   ```

3. **Retrain models** (recommended):
   ```bash
   python main.py train --model-type ensemble --force-retrain
   ```

4. **Or migrate existing models** (not recommended):
   - Rename `ridge_confidence.pkl` to `lightgbm_confidence.pkl` if using LightGBM
   - Update `modular_ensemble.meta.json` manually

### For New Deployments

Simply run:
```bash
python main.py train --model-type ensemble
```

The new unified configuration will automatically:
- Load data for all enabled models
- Train all models with correct data
- Save models with correct filenames
- Load models with correct mappings

## Benefits

1. **Consistency**: Single source of truth for all model configurations
2. **Clarity**: Explicit model names and tasks, no aliases
3. **Maintainability**: Easy to add/remove models
4. **Testing**: Easy to test individual components
5. **Flexibility**: Selective training/loading of models
6. **Correctness**: Proper model-to-output mapping prevents financial errors

## Risk Assessment

### Low Risk
- Configuration changes are additive (backward compatible)
- Existing models continue to work
- Gradual migration path available

### Medium Risk
- Need to retrain models for full benefits
- Inference code changes require testing
- Potential breaking changes for custom integrations

### Mitigation
- Comprehensive testing before deployment
- Backup existing models
- Rollback plan available
- Documentation for migration

## Success Criteria

1. ✓ TCN has dedicated data loader and is trained explicitly
2. ✓ LightGBM has dedicated data loader and trainer
3. ✓ All models saved with correct filenames
4. ✓ Inference loads models with correct mappings
5. ✓ MODEL_REGISTRY is single source of truth
6. ✓ Tests verify consistency across all stages
7. ✓ Documentation updated
8. ✓ Migration guide available

## Timeline

- **Phase 1**: 1 day (Configuration structure)
- **Phase 2**: 1 day (Data loading)
- **Phase 3**: 2 days (Training orchestration)
- **Phase 4**: 1 day (Inference)
- **Phase 5**: 2 days (Testing)
- **Phase 6**: 1 day (Documentation)

**Total**: 8 days

## Next Steps

1. Review and approve this plan
2. Switch to Code mode to implement
3. Execute implementation phases
4. Run comprehensive tests
5. Deploy with monitoring
