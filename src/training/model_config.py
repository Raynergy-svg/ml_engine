"""
Unified Model Configuration for Modular Ensemble Training.

This module provides a single source of truth for all model configurations,
ensuring consistency across initialization, data loading, training, and saving.

Models are defined with:
- name: Unique identifier for the model
- model_type: Type of model (transformer, tcn, xgboost, etc.)
- task: What the model predicts (direction, regime, momentum, etc.)
- data_key: Key used in data dictionary
- trainer_class: Trainer class for this model
- data_loader_func: Data loading function for this model
- save_extension: File extension for saving (.keras or .pkl)
- enabled: Whether this model is currently enabled
- priority: Training priority (lower = higher priority)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type, Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model in the ensemble.
    
    This class defines all metadata needed to train, save, and load a model
    in a consistent way across the entire pipeline.
    
    Attributes:
        name: Unique model identifier (e.g., 'transformer_direction')
        model_type: Type of model architecture (e.g., 'transformer', 'xgboost')
        task: Prediction task (e.g., 'direction', 'momentum', 'risk')
        data_key: Key used in data dictionary from data loaders
        trainer_class: Trainer class to instantiate for this model
        data_loader_func: Data loading function for this model
        save_extension: File extension for saving (.keras or .pkl)
        enabled: Whether this model is currently enabled for training
        priority: Training priority (lower numbers train first)
    """
    name: str
    model_type: str
    task: str
    data_key: str
    trainer_class: Type[Any]
    data_loader_func: Optional[Callable]
    save_extension: str
    enabled: bool = True
    priority: int = 0


# =============================================================================
# MODEL REGISTRY - Single Source of Truth
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    'transformer_direction': ModelConfig(
        name='transformer_direction',
        model_type='transformer',
        task='direction',
        data_key='direction',
        trainer_class=None,  # Will be imported at runtime to avoid circular imports
        data_loader_func=None,  # Will be imported at runtime
        save_extension='.keras',
        enabled=True,
        priority=1,
    ),
    'tcn_direction': ModelConfig(
        name='tcn_direction',
        model_type='tcn',
        task='direction',
        data_key='tcn',
        trainer_class=None,  # Will be imported at runtime
        data_loader_func=None,  # Will be imported at runtime
        save_extension='.keras',
        enabled=True,
        priority=2,
    ),
    # TCN Volatility Regime: 4-class volatility classification (LOW/NORMAL/HIGH/EXTREME)
    # Used as entry timing filter - only trade in HIGH or EXTREME volatility regimes
    # This is DIFFERENT from tcn_direction which predicts price direction
    'tcn_volatility_regime': ModelConfig(
        name='tcn_volatility_regime',
        model_type='tcn',
        task='volatility_regime',
        data_key='volatility_regime',
        trainer_class=None,  # Will be imported at runtime (TCNTrainer)
        data_loader_func=None,  # Will be imported at runtime (load_volatility_regime_data)
        save_extension='.keras',
        enabled=True,
        priority=2,  # Same priority as tcn_direction - both are optional
    ),
    'transformer_regime': ModelConfig(
        name='transformer_regime',
        model_type='transformer',
        task='regime',
        data_key='regime',
        trainer_class=None,  # Will be imported at runtime
        data_loader_func=None,  # Will be imported at runtime
        save_extension='.keras',
        enabled=True,
        priority=3,
    ),
    'xgboost_momentum': ModelConfig(
        name='xgboost_momentum',
        model_type='xgboost',
        task='momentum',
        data_key='xgboost',
        trainer_class=None,  # Will be imported at runtime
        data_loader_func=None,  # Will be imported at runtime
        save_extension='.pkl',
        enabled=True,
        priority=4,
    ),
    'rf_risk': ModelConfig(
        name='rf_risk',
        model_type='random_forest',
        task='risk',
        data_key='rf',
        trainer_class=None,  # Will be imported at runtime
        data_loader_func=None,  # Will be imported at runtime
        save_extension='.pkl',
        enabled=True,
        priority=5,
    ),
    # NOTE: lightgbm_confidence is disabled because RidgeTrainer already uses LightGBM internally.
    # The 'ridge' name is historical - the actual implementation uses LightGBM when available.
    'lightgbm_confidence': ModelConfig(
        name='lightgbm_confidence',
        model_type='lightgbm',
        task='confidence',
        data_key='lightgbm',
        trainer_class=None,  # Will be imported at runtime
        data_loader_func=None,  # Will be imported at runtime
        save_extension='.pkl',
        enabled=False,  # Disabled - RidgeTrainer already uses LightGBM internally
        priority=6,
    ),
    # NOTE: ridge_confidence uses RidgeTrainer which internally uses LightGBM when available.
    # The name 'ridge' is historical - actual implementation uses LightGBM for better performance.
    'ridge_confidence': ModelConfig(
        name='ridge_confidence',
        model_type='ridge',
        task='confidence',
        data_key='ridge',
        trainer_class=None,  # Will be imported at runtime
        data_loader_func=None,  # Will be imported at runtime
        save_extension='.pkl',
        enabled=True,  # The active confidence model (uses LightGBM internally)
        priority=6,
    ),
    'histgb_direction': ModelConfig(
        name='histgb_direction',
        model_type='histgradientboosting',
        task='direction',
        data_key='histgb',
        trainer_class=None,  # Will be imported at runtime
        data_loader_func=None,  # Will be imported at runtime
        save_extension='.pkl',
        enabled=False,  # Optional hybrid voting
        priority=8,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_enabled_models() -> List[str]:
    """Get list of enabled model names."""
    return [name for name, config in MODEL_REGISTRY.items() if config.enabled]


def get_models_by_task(task: str) -> List[str]:
    """Get list of model names that perform a specific task."""
    return [name for name, config in MODEL_REGISTRY.items() 
            if config.task == task and config.enabled]


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model by name."""
    return MODEL_REGISTRY.get(model_name)


def get_data_key_for_model(model_name: str) -> Optional[str]:
    """Get the data key used for a specific model."""
    config = MODEL_REGISTRY.get(model_name)
    return config.data_key if config else None


def get_output_for_task(task: str) -> str:
    """Get the output name for a given task.

    This maps tasks to their prediction outputs:
    - direction → 'direction'
    - regime → 'regime'
    - volatility_regime → 'volatility_regime' (4-class: LOW/NORMAL/HIGH/EXTREME)
    - momentum → 'momentum_score'
    - risk → 'expected_drawdown_pct'
    - confidence → 'confidence_score'
    """
    outputs = {
        'direction': 'direction',
        'regime': 'regime',
        'volatility_regime': 'volatility_regime',
        'momentum': 'momentum_score',
        'risk': 'expected_drawdown_pct',
        'confidence': 'confidence_score',
    }
    return outputs.get(task, 'unknown')


def validate_model_registry() -> Dict[str, Any]:
    """Validate MODEL_REGISTRY for missing trainers or data loaders.
    
    Returns:
        Dict with validation results:
        - valid: Overall validity
        - errors: List of error messages
        - warnings: List of warning messages
    """
    errors = []
    warnings = []
    
    for model_name, config in MODEL_REGISTRY.items():
        # Check if enabled
        if not config.enabled:
            continue
        
        # Check trainer class
        if config.trainer_class is None:
            errors.append(f"{model_name}: trainer_class is None")
        
        # Check data loader function
        if config.data_loader_func is None:
            errors.append(f"{model_name}: data_loader_func is None")
        
        # Check save extension
        if config.save_extension not in ['.keras', '.pkl']:
            errors.append(f"{model_name}: invalid save_extension '{config.save_extension}'")
        
        # Check task is valid
        valid_tasks = ['direction', 'regime', 'volatility_regime', 'momentum', 'risk', 'confidence']
        if config.task not in valid_tasks:
            errors.append(f"{model_name}: invalid task '{config.task}'")
        
        # Check model_type is valid
        valid_types = ['transformer', 'tcn', 'xgboost', 'random_forest', 
                      'lightgbm', 'ridge', 'histgradientboosting']
        if config.model_type not in valid_types:
            errors.append(f"{model_name}: invalid model_type '{config.model_type}'")
    
    # Check for duplicate data keys
    data_keys = [config.data_key for config in MODEL_REGISTRY.values()]
    seen_keys = set()
    for key in data_keys:
        if key in seen_keys:
            warnings.append(f"Duplicate data_key '{key}' found")
        seen_keys.add(key)
    
    valid = len(errors) == 0
    
    result = {
        'valid': valid,
        'errors': errors,
        'warnings': warnings,
        'n_models': len(MODEL_REGISTRY),
        'n_enabled': len(get_enabled_models()),
    }
    
    if not valid:
        logger.error(f"MODEL_REGISTRY validation failed: {errors}")
    elif warnings:
        logger.warning(f"MODEL_REGISTRY validation warnings: {warnings}")
    else:
        logger.info(f"MODEL_REGISTRY validation passed: {result['n_enabled']}/{result['n_models']} models enabled")
    
    return result


def _initialize_registry_with_imports() -> None:
    """Initialize MODEL_REGISTRY with actual trainer classes and data loader functions.
    
    This is called lazily to avoid circular imports at module load time.
    """
    global MODEL_REGISTRY
    
    # Import trainers
    from .modular_trainers import (
        TransformerDirectionTrainer,
        TransformerRegimeTrainer,
        TCNTrainer,
        TCNVolatilityRegimeTrainer,  # NEW: Research-backed TCN for 4-class volatility regime
        XGBoostTrainer,
        RandomForestTrainer,
        RidgeTrainer,
        HistGradientBoostingDirectionTrainer,
    )
    
    # Import data loaders
    from ..core.modular_data_loaders import (
        load_direction_data,
        load_tcn_data,
        load_regime_data,
        load_xgboost_data,
        load_rf_data,
        load_ridge_data,
        load_lightgbm_data,
        load_volatility_regime_data,
    )
    
    # Update trainer classes
    MODEL_REGISTRY['transformer_direction'].trainer_class = TransformerDirectionTrainer
    MODEL_REGISTRY['transformer_direction'].data_loader_func = load_direction_data
    
    MODEL_REGISTRY['tcn_direction'].trainer_class = TCNTrainer
    MODEL_REGISTRY['tcn_direction'].data_loader_func = load_tcn_data

    # TCN Volatility Regime: Uses specialized trainer with research-backed architecture
    MODEL_REGISTRY['tcn_volatility_regime'].trainer_class = TCNVolatilityRegimeTrainer
    MODEL_REGISTRY['tcn_volatility_regime'].data_loader_func = load_volatility_regime_data

    MODEL_REGISTRY['transformer_regime'].trainer_class = TransformerRegimeTrainer
    MODEL_REGISTRY['transformer_regime'].data_loader_func = load_regime_data
    
    MODEL_REGISTRY['xgboost_momentum'].trainer_class = XGBoostTrainer
    MODEL_REGISTRY['xgboost_momentum'].data_loader_func = load_xgboost_data
    
    MODEL_REGISTRY['rf_risk'].trainer_class = RandomForestTrainer
    MODEL_REGISTRY['rf_risk'].data_loader_func = load_rf_data
    
    MODEL_REGISTRY['lightgbm_confidence'].trainer_class = RidgeTrainer  # RidgeTrainer now uses LightGBM
    MODEL_REGISTRY['lightgbm_confidence'].data_loader_func = load_lightgbm_data
    
    MODEL_REGISTRY['ridge_confidence'].trainer_class = RidgeTrainer
    MODEL_REGISTRY['ridge_confidence'].data_loader_func = load_ridge_data
    
    MODEL_REGISTRY['histgb_direction'].trainer_class = HistGradientBoostingDirectionTrainer
    MODEL_REGISTRY['histgb_direction'].data_loader_func = load_direction_data
    
    logger.debug("MODEL_REGISTRY initialized with trainer classes and data loader functions")


# =============================================================================
# PUBLIC API
# =============================================================================

def get_registry() -> Dict[str, ModelConfig]:
    """Get the MODEL_REGISTRY, initializing imports if needed.
    
    This is the main entry point for accessing the model registry.
    It ensures that all trainer classes and data loader functions are
    imported before returning the registry.
    
    Returns:
        Dict mapping model names to ModelConfig objects
    """
    # Check if registry needs initialization
    first_config = next(iter(MODEL_REGISTRY.values()), None)
    if first_config and first_config.trainer_class is None:
        _initialize_registry_with_imports()
    
    return MODEL_REGISTRY


def get_model_names_by_priority() -> List[str]:
    """Get list of enabled model names sorted by priority (lowest first)."""
    enabled = get_enabled_models()
    return sorted(enabled, key=lambda name: MODEL_REGISTRY[name].priority)


def print_registry_summary() -> None:
    """Print a summary of the MODEL_REGISTRY for debugging."""
    logger.info("=" * 60)
    logger.info("MODEL REGISTRY SUMMARY")
    logger.info("=" * 60)
    
    registry = get_registry()
    enabled = get_enabled_models()
    
    logger.info(f"Total models: {len(registry)}")
    logger.info(f"Enabled models: {len(enabled)}")
    logger.info("")
    
    for model_name in get_model_names_by_priority():
        config = registry[model_name]
        status = "✓" if config.enabled else "✗"
        logger.info(f"{status} {model_name:30s} | "
                   f"type={config.model_type:20s} | "
                   f"task={config.task:15s} | "
                   f"data_key={config.data_key:15s} | "
                   f"priority={config.priority}")
    
    logger.info("=" * 60)


# Auto-initialize on first access
_registry_initialized = False


def ensure_registry_initialized() -> None:
    """Ensure MODEL_REGISTRY is initialized with imports."""
    global _registry_initialized
    if not _registry_initialized:
        _initialize_registry_with_imports()
        _registry_initialized = True
        logger.debug("MODEL_REGISTRY initialized")
