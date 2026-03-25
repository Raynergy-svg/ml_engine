"""
Trainers package - modular components for model trainers.

This package contains:
- base: Abstract base class for all trainers (BaseTrainer)
- config: Configuration classes (TrainerConfig, OverfitPreventionConfig)
- display: Training display utilities (TrainingDisplay)
- callbacks: Training callbacks for advanced features (lazy-loaded, requires TensorFlow)
- utils: Utility functions and constants
- exceptions: Custom exception classes (WeightMismatchError, CheckpointValidationError)
- tcn_trainer: TCN trainer for current volatility regime classification
- tcn_volatility_trainer: TCN trainer for forward-looking volatility prediction
- transformer_trainer: Transformer trainer for direction prediction
- transformer_regime_trainer: Transformer trainer for regime classification
- lightgbm_trainers: LightGBM-based trainers (regime, momentum, risk)
- histgb_trainer: HistGradientBoosting baseline trainer
- joint_trainer: Joint multi-pair training with contrastive learning
- migration: Model migration utilities for version compatibility
- train_all: Main training orchestration function

NOTE: Heavy-dependency submodules (TF callbacks, TF trainers, XGBoost, LightGBM)
are lazy-imported to allow the CLI to boot without TensorFlow installed.
Per improvement.md rule: "ALWAYS use lazy imports in package __init__.py
when submodules have heavy dependencies."
"""

import importlib as _importlib
from typing import TYPE_CHECKING

# ── Always-available (no heavy deps) ────────────────────────────────────
from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig, OverfitPreventionConfig
from src.training.trainers.display import TrainingDisplay
from src.training.trainers.exceptions import (
    WeightMismatchError,
    CheckpointValidationError,
)
from src.training.trainers.utils import (
    # Constants
    MODEL_NOT_TRAINED_ERROR,
    PRODUCTION_MODELS_DIR,
    META_PKL_SUFFIX,
    WEIGHTS_H5_SUFFIX,
    ARCH_JSON_SUFFIX,
    EWC_PKL_SUFFIX,
    EMA_PKL_SUFFIX,
    SERIALIZED_MODEL_WARNING,
    UNPICKLE_ESTIMATOR_WARNING,
    LGBM_NOT_INSTALLED_ERROR,
    JOINT_MODELS_DIR,
    TRANSFORMER_DIRECTION_FILENAME,
    LGBM_MOMENTUM_FILENAME,
    LGBM_RISK_FILENAME,
    RIDGE_CONFIDENCE_FILENAME,
    NO_DIRECTION_DATA_ERROR,
    WEIGHTS_LOADED_FULL_MODEL_MSG,
    VOLATILE_PAIRS,
    STABLE_PAIRS,
    REGIME_LGBM_PARAMS,
    REGIME_NAMES_LIST,
    # Functions
    compute_auto_variance_weight,
    _get_numpy_dtype,
    create_sequences,
    create_sequences_with_weights,
    get_config_seq_len,
    _safe_load_weights_ignoring_optimizer,
    _safe_reset_optimizer_state,
    _validate_weight_shapes,
    create_ewc_loss,
    _safe_get_learning_rate,
    _safe_set_learning_rate,
    get_regime_lgbm_params,
    _create_lgbm_regressor,
    _create_lgbm_classifier,
)
# migration and train_all are lazy-loaded (they pull in TF-dependent models)

# ── Lazy imports for heavy-dependency modules ────────────────────────────
# These are only resolved on first attribute access, keeping CLI startup fast.

# Map of attribute name -> (module_path, attribute_name_in_module)
_LAZY_IMPORTS = {
    # TF-dependent trainers
    "TCNTrainer": ("src.training.trainers.tcn_trainer", "TCNTrainer"),
    "TCNVolatilityRegimeTrainer": ("src.training.trainers.tcn_volatility_trainer", "TCNVolatilityRegimeTrainer"),
    "TransformerDirectionTrainer": ("src.training.trainers.transformer_trainer", "TransformerDirectionTrainer"),
    "TransformerRegimeTrainer": ("src.training.trainers.transformer_regime_trainer", "TransformerRegimeTrainer"),
    "JointMultiPairTrainer": ("src.training.trainers.joint_trainer", "JointMultiPairTrainer"),
    # XGBoost-dependent
    "XGBoostTrainer": ("src.training.trainers.xgboost_trainer", "XGBoostTrainer"),
    # scikit-learn dependent
    "RandomForestTrainer": ("src.training.trainers.random_forest_trainer", "RandomForestTrainer"),
    "RidgeTrainer": ("src.training.trainers.ridge_trainer", "RidgeTrainer"),
    "HistGradientBoostingDirectionTrainer": ("src.training.trainers.histgb_trainer", "HistGradientBoostingDirectionTrainer"),
    # LightGBM-dependent
    "RegimeLGBMTrainer": ("src.training.trainers.lightgbm_trainers", "RegimeLGBMTrainer"),
    "LightGBMMomentumTrainer": ("src.training.trainers.lightgbm_trainers", "LightGBMMomentumTrainer"),
    "LightGBMRiskTrainer": ("src.training.trainers.lightgbm_trainers", "LightGBMRiskTrainer"),
    # TF callbacks (requires tensorflow)
    "EMACallback": ("src.training.trainers.callbacks", "EMACallback"),
    "EWCPenalty": ("src.training.trainers.callbacks", "EWCPenalty"),
    "EWCLoss": ("src.training.trainers.callbacks", "EWCLoss"),
    "OverfitPreventionCallback": ("src.training.trainers.callbacks", "OverfitPreventionCallback"),
    "EWCTrainingCallback": ("src.training.trainers.callbacks", "EWCTrainingCallback"),
    "QuietProgressCallback": ("src.training.trainers.callbacks", "QuietProgressCallback"),
    "GradualUnfreezeCallback": ("src.training.trainers.callbacks", "GradualUnfreezeCallback"),
    "RichEpochCallback": ("src.training.trainers.callbacks", "RichEpochCallback"),
    "AutoAdjustCallback": ("src.training.trainers.callbacks", "AutoAdjustCallback"),
    "ReplayBuffer": ("src.training.trainers.callbacks", "ReplayBuffer"),
    "DriftDetector": ("src.training.trainers.callbacks", "DriftDetector"),
    "TrainingLineage": ("src.training.trainers.callbacks", "TrainingLineage"),
    # Migration (depends on xgboost_model -> TF model_builders)
    "migrate_xgboost_model": ("src.training.trainers.migration", "migrate_xgboost_model"),
    "migrate_all_models": ("src.training.trainers.migration", "migrate_all_models"),
    # Training orchestration (depends on TF trainers)
    "train_all_modular": ("src.training.trainers.train_all", "train_all_modular"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache in module namespace so __getattr__ isn't called again
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── TYPE_CHECKING block for IDE support ──────────────────────────────────
if TYPE_CHECKING:
    from src.training.trainers.tcn_trainer import TCNTrainer
    from src.training.trainers.tcn_volatility_trainer import TCNVolatilityRegimeTrainer
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
    from src.training.trainers.transformer_regime_trainer import TransformerRegimeTrainer
    from src.training.trainers.xgboost_trainer import XGBoostTrainer
    from src.training.trainers.random_forest_trainer import RandomForestTrainer
    from src.training.trainers.ridge_trainer import RidgeTrainer
    from src.training.trainers.lightgbm_trainers import (
        RegimeLGBMTrainer,
        LightGBMMomentumTrainer,
        LightGBMRiskTrainer,
    )
    from src.training.trainers.histgb_trainer import HistGradientBoostingDirectionTrainer
    from src.training.trainers.joint_trainer import JointMultiPairTrainer
    from src.training.trainers.callbacks import (
        EMACallback,
        EWCPenalty,
        EWCLoss,
        OverfitPreventionCallback,
        EWCTrainingCallback,
        QuietProgressCallback,
        GradualUnfreezeCallback,
        RichEpochCallback,
        AutoAdjustCallback,
        ReplayBuffer,
        DriftDetector,
        TrainingLineage,
    )

__all__ = [
    # Base class
    "BaseTrainer",
    # Config classes
    "TrainerConfig",
    "OverfitPreventionConfig",
    # Display
    "TrainingDisplay",
    # Exceptions
    "WeightMismatchError",
    "CheckpointValidationError",
    # Trainers (lazy)
    "TCNTrainer",
    "TCNVolatilityRegimeTrainer",
    "TransformerDirectionTrainer",
    "TransformerRegimeTrainer",
    "XGBoostTrainer",
    "RandomForestTrainer",
    "RidgeTrainer",
    # LightGBM Trainers (lazy)
    "RegimeLGBMTrainer",
    "LightGBMMomentumTrainer",
    "LightGBMRiskTrainer",
    # Other Trainers (lazy)
    "HistGradientBoostingDirectionTrainer",
    "JointMultiPairTrainer",
    # Callbacks (lazy)
    "EMACallback",
    "EWCPenalty",
    "EWCLoss",
    "OverfitPreventionCallback",
    "EWCTrainingCallback",
    "QuietProgressCallback",
    "GradualUnfreezeCallback",
    "RichEpochCallback",
    "AutoAdjustCallback",
    "ReplayBuffer",
    "DriftDetector",
    "TrainingLineage",
    # Constants
    "MODEL_NOT_TRAINED_ERROR",
    "PRODUCTION_MODELS_DIR",
    "META_PKL_SUFFIX",
    "WEIGHTS_H5_SUFFIX",
    "ARCH_JSON_SUFFIX",
    "EWC_PKL_SUFFIX",
    "EMA_PKL_SUFFIX",
    "SERIALIZED_MODEL_WARNING",
    "UNPICKLE_ESTIMATOR_WARNING",
    "LGBM_NOT_INSTALLED_ERROR",
    "JOINT_MODELS_DIR",
    "TRANSFORMER_DIRECTION_FILENAME",
    "LGBM_MOMENTUM_FILENAME",
    "LGBM_RISK_FILENAME",
    "RIDGE_CONFIDENCE_FILENAME",
    "NO_DIRECTION_DATA_ERROR",
    "WEIGHTS_LOADED_FULL_MODEL_MSG",
    "VOLATILE_PAIRS",
    "STABLE_PAIRS",
    "REGIME_LGBM_PARAMS",
    "REGIME_NAMES_LIST",
    # Functions
    "compute_auto_variance_weight",
    "_get_numpy_dtype",
    "create_sequences",
    "create_sequences_with_weights",
    "get_config_seq_len",
    "_safe_load_weights_ignoring_optimizer",
    "_safe_reset_optimizer_state",
    "_validate_weight_shapes",
    "create_ewc_loss",
    "_safe_get_learning_rate",
    "_safe_set_learning_rate",
    "get_regime_lgbm_params",
    "_create_lgbm_regressor",
    "_create_lgbm_classifier",
    # Migration utilities
    "migrate_xgboost_model",
    "migrate_all_models",
    # Training orchestration
    "train_all_modular",
]
