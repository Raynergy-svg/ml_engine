"""
Modular Trainers - Backward Compatibility Facade.

This module provides backward compatibility by re-exporting all trainer classes
and utilities from the new modular structure under src/training/trainers/.

Original file (10,820 lines) has been refactored into focused modules:
- base.py: BaseTrainer abstract class
- config.py: TrainerConfig, OverfitPreventionConfig
- display.py: TrainingDisplay
- callbacks.py: 12 callback classes (EMA, EWC, etc.)
- utils.py: Helper functions and constants
- tcn_trainer.py: TCNTrainer
- tcn_volatility_trainer.py: TCNVolatilityRegimeTrainer
- transformer_trainer.py: TransformerDirectionTrainer
- transformer_regime_trainer.py: TransformerRegimeTrainer
- xgboost_trainer.py: XGBoostTrainer
- random_forest_trainer.py: RandomForestTrainer
- ridge_trainer.py: RidgeTrainer
- lightgbm_trainers.py: RegimeLGBMTrainer, LightGBMMomentumTrainer, LightGBMRiskTrainer
- histgb_trainer.py: HistGradientBoostingDirectionTrainer
- joint_trainer.py: JointMultiPairTrainer
- migration.py: migrate_xgboost_model, migrate_all_models
- train_all.py: train_all_modular

All imports remain unchanged for backward compatibility.
"""

# Re-export everything from the new modular structure
from src.training.trainers import *  # noqa: F401, F403

# Maintain module-level __all__ for explicit exports
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
