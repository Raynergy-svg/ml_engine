"""
Shim module - Re-exports from src.training.modular_trainers for backward compatibility.

This module provides backward compatibility for code that imports from the old
'modular_trainers' location. The actual implementation has been moved to
src.training.modular_trainers as part of the project restructure.

DEPRECATION NOTICE:
    This shim module is provided for backward compatibility only. New code should
    import directly from src.training.modular_trainers:
        from src.training.modular_trainers import TrainerConfig, BaseTrainer

Usage:
    # Old style (still works via this shim):
    from modular_trainers import TrainerConfig, TransformerDirectionTrainer
    
    # New style (recommended):
    from src.training.modular_trainers import TrainerConfig, TransformerDirectionTrainer

Performance:
    Imports are lazy-loaded on first access to minimize startup overhead.
    Explicit imports are used instead of wildcard for better IDE support
    and static analysis.
"""

import sys
import warnings
from typing import TYPE_CHECKING

# Use TYPE_CHECKING for type hints without circular imports
if TYPE_CHECKING:
    from src.training.modular_trainers import (
        TrainerConfig,
        EMACallback,
        EWCPenalty,
        OverfitPreventionCallback,
        EWCTrainingCallback,
        RichEpochCallback,
        ReplayBuffer,
        DriftDetector,
        TrainingLineage,
        BaseTrainer,
        TCNTrainer,
        TransformerDirectionTrainer,
        TransformerRegimeTrainer,
        XGBoostTrainer,
        RandomForestTrainer,
        RidgeTrainer,
        HistGradientBoostingDirectionTrainer,
    )
    from src.training.modular_trainers import (
        create_ewc_loss,
        migrate_xgboost_model,
        migrate_all_models,
        train_all_modular,
    )

# Module-level cache for lazy-loaded imports
_module_cache = None
_import_error = None


def _get_source_module():
    """
    Lazily import the source module with error handling.
    
    Returns:
        The src.training.modular_trainers module
        
    Raises:
        ImportError: If the source module cannot be imported
    """
    global _module_cache, _import_error
    
    # Return cached module if already loaded
    if _module_cache is not None:
        return _module_cache
    
    # Return cached error if import previously failed
    if _import_error is not None:
        raise _import_error
    
    try:
        from src.training import modular_trainers as source_module
        _module_cache = source_module
        return source_module
    except ImportError as e:
        _import_error = ImportError(
            f"Cannot import from src.training.modular_trainers: {e}\n"
            f"Ensure the module exists and all dependencies are installed."
        )
        raise _import_error
    except Exception as e:
        _import_error = ImportError(
            f"Unexpected error importing src.training.modular_trainers: {e}"
        )
        raise _import_error


# Define public API explicitly for better IDE support and static analysis
__all__ = [
    # Configuration
    'TrainerConfig',
    
    # Callbacks
    'EMACallback',
    'EWCPenalty',
    'OverfitPreventionCallback',
    'EWCTrainingCallback',
    'RichEpochCallback',
    
    # Utilities
    'create_ewc_loss',
    'ReplayBuffer',
    'DriftDetector',
    'TrainingLineage',
    
    # Base Trainer
    'BaseTrainer',
    
    # Trainer Implementations
    'TCNTrainer',
    'TransformerDirectionTrainer',
    'TransformerRegimeTrainer',
    'XGBoostTrainer',
    'RandomForestTrainer',
    'RidgeTrainer',
    'HistGradientBoostingDirectionTrainer',
    
    # Convenience Functions
    'migrate_xgboost_model',
    'migrate_all_models',
    'train_all_modular',
    
    # Version info
    '__version__',
    'get_source_module',
]


def __getattr__(name: str):
    """
    Lazy-load attributes from the source module.
    
    This function is called when an attribute is accessed that doesn't exist
    in this module's __dict__. It provides lazy loading for better performance.
    
    Args:
        name: The attribute name being accessed
        
    Returns:
        The requested attribute from the source module
        
    Raises:
        AttributeError: If the attribute doesn't exist in the source module
    """
    # Emit deprecation warning on first access
    if not hasattr(sys, '_modular_trainers_warned'):
        warnings.warn(
            "Importing from 'modular_trainers' is deprecated. "
            "Please import from 'src.training.modular_trainers' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        sys._modular_trainers_warned = True
    
    source_module = _get_source_module()
    
    try:
        return getattr(source_module, name)
    except AttributeError:
        available = ', '.join(sorted(dir(source_module)))
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Available attributes: {available}"
        ) from None


def __dir__():
    """Provide directory listing including source module attributes."""
    try:
        source_module = _get_source_module()
        return sorted(set(__all__) | set(dir(source_module)))
    except ImportError:
        return __all__


# Version information for compatibility checking
__version__ = '1.0.0'
__deprecated__ = True


# Convenience function for explicit module access
def get_source_module():
    """
    Get the source module explicitly.
    
    This is useful for code that needs to access the module directly
    without triggering the lazy loading mechanism.
    
    Returns:
        The src.training.modular_trainers module
    """
    return _get_source_module()


# Explicitly export version and module info
__doc__ += f"""

Version: {__version__}
Deprecated: {__deprecated__}
"""
