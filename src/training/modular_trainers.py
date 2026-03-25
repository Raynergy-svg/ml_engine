"""Backward-compatibility facade for training components.

Historically, this project exposed many trainer classes, callbacks, and helpers
from a single (very large) module. The implementation has since been refactored
into the modular package under `src/training/trainers/`.

This module keeps old imports working:

    from src.training.modular_trainers import TransformerDirectionTrainer

Heavy-dependency symbols are lazy-loaded so the CLI can boot without TensorFlow.
"""

from __future__ import annotations

import importlib as _importlib
from typing import TYPE_CHECKING

# ── Always-available (no heavy deps) ────────────────────────────────────
from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import OverfitPreventionConfig, TrainerConfig
from src.training.trainers.display import TrainingDisplay
from src.training.trainers.utils import (
    ARCH_JSON_SUFFIX,
    EMA_PKL_SUFFIX,
    EWC_PKL_SUFFIX,
    META_PKL_SUFFIX,
    MODEL_NOT_TRAINED_ERROR,
    PRODUCTION_MODELS_DIR,
    STABLE_PAIRS,
    VOLATILE_PAIRS,
    WEIGHTS_H5_SUFFIX,
    compute_auto_variance_weight,
    create_ewc_loss,
    create_sequences,
    create_sequences_with_weights,
    get_config_seq_len,
    get_regime_lgbm_params,
)

# ── Lazy imports for TF/XGB/LightGBM-dependent symbols ──────────────────
_LAZY_IMPORTS = {
    # Callbacks (TF-dependent)
    "AutoAdjustCallback": ("src.training.trainers.callbacks", "AutoAdjustCallback"),
    "DriftDetector": ("src.training.trainers.callbacks", "DriftDetector"),
    "EMACallback": ("src.training.trainers.callbacks", "EMACallback"),
    "EWCLoss": ("src.training.trainers.callbacks", "EWCLoss"),
    "EWCPenalty": ("src.training.trainers.callbacks", "EWCPenalty"),
    "EWCTrainingCallback": ("src.training.trainers.callbacks", "EWCTrainingCallback"),
    "GradualUnfreezeCallback": ("src.training.trainers.callbacks", "GradualUnfreezeCallback"),
    "OverfitPreventionCallback": ("src.training.trainers.callbacks", "OverfitPreventionCallback"),
    "QuietProgressCallback": ("src.training.trainers.callbacks", "QuietProgressCallback"),
    "ReplayBuffer": ("src.training.trainers.callbacks", "ReplayBuffer"),
    "RichEpochCallback": ("src.training.trainers.callbacks", "RichEpochCallback"),
    "TrainingLineage": ("src.training.trainers.callbacks", "TrainingLineage"),
    # TF trainers
    "TCNTrainer": ("src.training.trainers.tcn_trainer", "TCNTrainer"),
    "TCNVolatilityRegimeTrainer": ("src.training.trainers.tcn_volatility_trainer", "TCNVolatilityRegimeTrainer"),
    "TransformerDirectionTrainer": ("src.training.trainers.transformer_trainer", "TransformerDirectionTrainer"),
    "TransformerRegimeTrainer": ("src.training.trainers.transformer_regime_trainer", "TransformerRegimeTrainer"),
    "JointMultiPairTrainer": ("src.training.trainers.joint_trainer", "JointMultiPairTrainer"),
    # XGBoost
    "XGBoostTrainer": ("src.training.trainers.xgboost_trainer", "XGBoostTrainer"),
    # scikit-learn
    "RandomForestTrainer": ("src.training.trainers.random_forest_trainer", "RandomForestTrainer"),
    "RidgeTrainer": ("src.training.trainers.ridge_trainer", "RidgeTrainer"),
    "HistGradientBoostingDirectionTrainer": ("src.training.trainers.histgb_trainer", "HistGradientBoostingDirectionTrainer"),
    # LightGBM
    "RegimeLGBMTrainer": ("src.training.trainers.lightgbm_trainers", "RegimeLGBMTrainer"),
    "LightGBMMomentumTrainer": ("src.training.trainers.lightgbm_trainers", "LightGBMMomentumTrainer"),
    "LightGBMRiskTrainer": ("src.training.trainers.lightgbm_trainers", "LightGBMRiskTrainer"),
    # Migration (TF-dependent via model_builders)
    "migrate_xgboost_model": ("src.training.trainers.migration", "migrate_xgboost_model"),
    "migrate_all_models": ("src.training.trainers.migration", "migrate_all_models"),
    # Training orchestration
    "train_all_modular": ("src.training.trainers.train_all", "train_all_modular"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from src.training.trainers.callbacks import (
        AutoAdjustCallback,
        DriftDetector,
        EMACallback,
        EWCLoss,
        EWCPenalty,
        EWCTrainingCallback,
        GradualUnfreezeCallback,
        OverfitPreventionCallback,
        QuietProgressCallback,
        ReplayBuffer,
        RichEpochCallback,
        TrainingLineage,
    )
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
    from src.training.trainers.migration import migrate_all_models, migrate_xgboost_model
    from src.training.trainers.train_all import train_all_modular

__all__ = [
    # Configuration
    "TrainerConfig",
    "OverfitPreventionConfig",
    # Display
    "TrainingDisplay",
    # Base
    "BaseTrainer",
    # Callbacks
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
    # Trainer Classes
    "TCNTrainer",
    "TCNVolatilityRegimeTrainer",
    "TransformerDirectionTrainer",
    "TransformerRegimeTrainer",
    "XGBoostTrainer",
    "RandomForestTrainer",
    "RidgeTrainer",
    "RegimeLGBMTrainer",
    "LightGBMMomentumTrainer",
    "LightGBMRiskTrainer",
    "HistGradientBoostingDirectionTrainer",
    "JointMultiPairTrainer",
    # Utilities
    "compute_auto_variance_weight",
    "create_sequences",
    "create_sequences_with_weights",
    "get_config_seq_len",
    "create_ewc_loss",
    "get_regime_lgbm_params",
    # Constants
    "MODEL_NOT_TRAINED_ERROR",
    "PRODUCTION_MODELS_DIR",
    "META_PKL_SUFFIX",
    "WEIGHTS_H5_SUFFIX",
    "ARCH_JSON_SUFFIX",
    "EWC_PKL_SUFFIX",
    "EMA_PKL_SUFFIX",
    "VOLATILE_PAIRS",
    "STABLE_PAIRS",
    # Migration
    "migrate_xgboost_model",
    "migrate_all_models",
    # Training
    "train_all_modular",
]
