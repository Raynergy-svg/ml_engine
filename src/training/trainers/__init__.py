"""
Trainers package - modular components for model trainers.

This package contains:
- base: Abstract base class for all trainers (BaseTrainer)
- config: Configuration classes (TrainerConfig, OverfitPreventionConfig)
- display: Training display utilities (TrainingDisplay)
- callbacks: Training callbacks for advanced features
- utils: Utility functions and constants
- tcn_trainer: TCN trainer for current volatility regime classification
- tcn_volatility_trainer: TCN trainer for forward-looking volatility prediction
- transformer_trainer: Transformer trainer for direction prediction
- transformer_regime_trainer: Transformer trainer for regime classification
- lightgbm_trainers: LightGBM-based trainers (regime, momentum, risk)
- histgb_trainer: HistGradientBoosting baseline trainer
- joint_trainer: Joint multi-pair training with contrastive learning
"""

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig, OverfitPreventionConfig
from src.training.trainers.display import TrainingDisplay
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

__all__ = [
    # Base class
    "BaseTrainer",
    # Config classes
    "TrainerConfig",
    "OverfitPreventionConfig",
    # Display
    "TrainingDisplay",
    # Trainers
    "TCNTrainer",
    "TCNVolatilityRegimeTrainer",
    "TransformerDirectionTrainer",
    "TransformerRegimeTrainer",
    "XGBoostTrainer",
    "RandomForestTrainer",
    "RidgeTrainer",
    # LightGBM Trainers
    "RegimeLGBMTrainer",
    "LightGBMMomentumTrainer",
    "LightGBMRiskTrainer",
    # Other Trainers
    "HistGradientBoostingDirectionTrainer",
    "JointMultiPairTrainer",
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
]
