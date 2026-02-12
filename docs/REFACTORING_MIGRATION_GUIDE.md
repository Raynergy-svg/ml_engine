# Refactoring Migration Guide

## Overview

The ML Engine codebase has been refactored to improve maintainability and readability. The monolithic `modular_trainers.py` file (10,820 lines) has been decomposed into 18 focused modules under `src/training/trainers/`.

## What Changed

### Before: Single Monolithic File
```
src/training/modular_trainers.py (10,820 lines)
```

### After: Modular Structure
```
src/training/trainers/
├── __init__.py              # Public API exports
├── base.py                  # BaseTrainer abstract class
├── config.py                # TrainerConfig, OverfitPreventionConfig
├── display.py               # TrainingDisplay for output
├── callbacks.py             # 12 callback classes
├── utils.py                 # Helper functions and constants
├── tcn_trainer.py           # TCN volatility regime trainer
├── tcn_volatility_trainer.py # TCN volatility predictor
├── transformer_trainer.py   # Transformer direction trainer
├── transformer_regime_trainer.py # Transformer regime classifier
├── xgboost_trainer.py       # XGBoost momentum gate
├── random_forest_trainer.py # RandomForest risk gate
├── ridge_trainer.py         # Ridge confidence gate
├── lightgbm_trainers.py     # 3 LightGBM trainers
├── histgb_trainer.py        # HistGradientBoosting baseline
├── joint_trainer.py         # Multi-pair joint training
├── migration.py             # Model migration utilities
└── train_all.py             # Training orchestration
```

## Migration Path

### Option 1: No Changes Required (Recommended)

**The refactoring maintains 100% backward compatibility.** All existing imports continue to work:

```python
# These imports still work - no changes needed
from src.training.modular_trainers import (
    TransformerDirectionTrainer,
    XGBoostTrainer,
    RandomForestTrainer,
    RidgeTrainer,
    TrainerConfig,
    BaseTrainer,
)
```

The `src/training/modular_trainers.py` file now acts as a **facade** that re-exports everything from the new modular structure.

### Option 2: Adopt New Imports (Recommended for New Code)

For new code or when refactoring existing code, use the new modular imports:

```python
# New modular imports - more explicit
from src.training.trainers import (
    TransformerDirectionTrainer,
    XGBoostTrainer,
    RandomForestTrainer,
    RidgeTrainer,
)
from src.training.trainers.config import TrainerConfig
from src.training.trainers.callbacks import EMACallback, ReplayBuffer
from src.training.trainers.utils import compute_auto_variance_weight
```

## Benefits of New Structure

### 1. **Improved Maintainability**
- Each module has a single, focused responsibility
- Easier to locate and understand specific functionality
- Reduced cognitive load when reading code

### 2. **Better Testability**
- Individual modules can be tested in isolation
- Easier to mock dependencies
- Faster test execution for unit tests

### 3. **Enhanced Reusability**
- Components can be imported precisely where needed
- No need to import entire monolithic file
- Clear separation of concerns

### 4. **Easier Collaboration**
- Smaller files reduce merge conflicts
- Clearer ownership boundaries
- Easier code reviews

### 5. **Better IDE Support**
- Faster autocomplete
- More accurate code navigation
- Better refactoring tools support

## Module Reference

### Core Modules

#### `base.py` - Abstract Base Class
```python
from src.training.trainers.base import BaseTrainer
```
Contains the abstract base class that all trainers inherit from with standard interface: `train()`, `predict()`, `save()`, `load()`.

#### `config.py` - Configuration Classes
```python
from src.training.trainers.config import TrainerConfig, OverfitPreventionConfig
```
Contains all training configuration dataclasses.

#### `callbacks.py` - Training Callbacks
```python
from src.training.trainers.callbacks import (
    EMACallback,              # Exponential Moving Average
    EWCPenalty,               # Elastic Weight Consolidation
    OverfitPreventionCallback, # SWA + cosine restarts
    ReplayBuffer,             # Memory replay
    DriftDetector,            # Performance drift detection
    TrainingLineage,          # Training history tracking
)
```

#### `utils.py` - Utility Functions
```python
from src.training.trainers.utils import (
    compute_auto_variance_weight,
    create_sequences,
    get_config_seq_len,
    PRODUCTION_MODELS_DIR,
    VOLATILE_PAIRS,
)
```

### Trainer Modules

Each trainer is in its own module for easy discovery and maintenance:

```python
from src.training.trainers import (
    TCNTrainer,                          # Current volatility regime
    TCNVolatilityRegimeTrainer,          # Future volatility prediction
    TransformerDirectionTrainer,         # Direction prediction (Gate 1)
    TransformerRegimeTrainer,            # Market regime classification
    XGBoostTrainer,                      # Momentum analysis (Gate 3)
    RandomForestTrainer,                 # Risk assessment (Gate 4)
    RidgeTrainer,                        # Confidence scoring (Gate 2)
    RegimeLGBMTrainer,                   # Regime-specific LightGBM
    LightGBMMomentumTrainer,             # LightGBM momentum
    LightGBMRiskTrainer,                 # LightGBM risk
    HistGradientBoostingDirectionTrainer, # Baseline direction
    JointMultiPairTrainer,               # Multi-pair joint training
)
```

### Utility Modules

#### `migration.py` - Model Migration
```python
from src.training.trainers.migration import (
    migrate_xgboost_model,
    migrate_all_models,
)
```

#### `train_all.py` - Training Orchestration
```python
from src.training.trainers.train_all import train_all_modular
```

## Common Migration Scenarios

### Scenario 1: Training Pipeline Code

**Before:**
```python
from src.training.modular_trainers import (
    TransformerDirectionTrainer,
    XGBoostTrainer,
    TrainerConfig,
)

config = TrainerConfig()
trainer = TransformerDirectionTrainer(config)
trainer.train(X_train, y_train, X_val, y_val)
```

**After (no changes needed):**
```python
# Same code - still works!
from src.training.modular_trainers import (
    TransformerDirectionTrainer,
    XGBoostTrainer,
    TrainerConfig,
)

config = TrainerConfig()
trainer = TransformerDirectionTrainer(config)
trainer.train(X_train, y_train, X_val, y_val)
```

**After (new style - optional):**
```python
# More explicit imports
from src.training.trainers import TransformerDirectionTrainer, XGBoostTrainer
from src.training.trainers.config import TrainerConfig

config = TrainerConfig()
trainer = TransformerDirectionTrainer(config)
trainer.train(X_train, y_train, X_val, y_val)
```

### Scenario 2: Using Callbacks

**Before:**
```python
from src.training.modular_trainers import EMACallback, ReplayBuffer

ema = EMACallback()
replay = ReplayBuffer()
```

**After (new style):**
```python
from src.training.trainers.callbacks import EMACallback, ReplayBuffer

ema = EMACallback()
replay = ReplayBuffer()
```

### Scenario 3: Utility Functions

**Before:**
```python
from src.training.modular_trainers import compute_auto_variance_weight

weight = compute_auto_variance_weight("EUR_USD", 0.55)
```

**After (new style):**
```python
from src.training.trainers.utils import compute_auto_variance_weight

weight = compute_auto_variance_weight("EUR_USD", 0.55)
```

## Testing

All existing tests continue to work without modification due to backward compatibility:

```bash
# Run tests - no changes needed
pytest tests/test_modular_trainers.py -v
pytest tests/ -k "trainer" -v
```

## Troubleshooting

### Import Errors

**Problem:** `ImportError: cannot import name 'SomeClass' from 'src.training.modular_trainers'`

**Solution:** The class may have been in the original file but not exported. Check `src/training/trainers/__init__.py` for the full list of exported items.

### Circular Import Errors

**Problem:** Circular import when using new modular imports

**Solution:** Use the facade import from `src.training.modular_trainers` or restructure imports to avoid circular dependencies.

### Missing Dependencies

**Problem:** `ModuleNotFoundError` when importing from new modules

**Solution:** Ensure you're importing from the correct module. Refer to the Module Reference section above.

## Rollback Plan

If issues arise, you can temporarily revert to the original file:

```bash
# Restore original monolithic file
git show HEAD~6:src/training/modular_trainers.py > src/training/modular_trainers.py
```

However, this should not be necessary as backward compatibility is maintained.

## Future Plans

1. **Phase 2**: Refactor `modular_inference.py` into layered architecture
2. **Phase 3**: Add type stubs (`.pyi` files) for better IDE support
3. **Phase 4**: Extract common patterns into shared utilities
4. **Phase 5**: Performance profiling and optimization

## Questions?

For questions or issues related to the refactoring:
1. Check this migration guide
2. Review module docstrings in `src/training/trainers/`
3. Open an issue on GitHub with the `refactoring` label

## Summary

- ✅ **100% backward compatible** - existing code works without changes
- ✅ **18 focused modules** - each with clear responsibility
- ✅ **Better organization** - easier to find and maintain code
- ✅ **Improved testability** - modules can be tested in isolation
- ✅ **Future-ready** - foundation for further improvements

**Bottom line:** You don't need to change anything unless you want to adopt the new modular import style for clarity.
