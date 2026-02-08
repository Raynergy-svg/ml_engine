"""
Models module for Buddy Trading System.

This module provides:
- Model builders (TCN, LSTM, XGBoost)
- TensorFlow model implementations
- Ensemble model support

Usage:
    from src.models import ModelFactory, ModelConfig, ModelType
    from src.models import build_tcn_model, build_shared_encoder_model
"""

from src.models.model_builders import (
    # Enums and config
    ModelType,
    ModelConfig,

    # Factory class
    ModelFactory,

    # Builder functions (new API)
    build_tcn_model,
    build_lstm_model,
    build_shared_encoder_model,
    build_xgboost_model,

    # Backward compatibility aliases
    _build_buddy_model,
    _build_buddy_model_shared_encoder,
    _build_buddy_model_tcn,
    _build_xgboost_model,
    _build_buddy_model_for_type,
)

__all__ = [
    # Enums and config
    "ModelType",
    "ModelConfig",

    # Factory class
    "ModelFactory",

    # Builder functions (new API)
    "build_tcn_model",
    "build_lstm_model",
    "build_shared_encoder_model",
    "build_xgboost_model",

    # Backward compatibility aliases
    "_build_buddy_model",
    "_build_buddy_model_shared_encoder",
    "_build_buddy_model_tcn",
    "_build_xgboost_model",
    "_build_buddy_model_for_type",
]
