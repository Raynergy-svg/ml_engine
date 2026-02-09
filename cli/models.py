#!/usr/bin/env python3
"""Model architecture builders for ML Engine Trading Bot.

Re-exports from src.models.model_builders. Uses deferred imports
to avoid loading TensorFlow at module-import time.
"""
from __future__ import annotations


def _build_buddy_model(**kwargs):
    """Create the Buddy model: 5 parallel LSTM heads + shared dense + direction+confidence outputs."""
    from src.models.model_builders import _build_buddy_model as _impl
    return _impl(**kwargs)


def _build_buddy_model_shared_encoder(**kwargs):
    """Create a faster Buddy model with a shared LSTM encoder + dense heads."""
    from src.models.model_builders import _build_buddy_model_shared_encoder as _impl
    return _impl(**kwargs)


def _build_buddy_model_tcn(**kwargs):
    """Create a Buddy model with TCN encoder - FASTER on M1 Metal than LSTM."""
    from src.models.model_builders import _build_buddy_model_tcn as _impl
    return _impl(**kwargs)


class ModelType:
    """Enumeration of supported model types. Re-export from src.models.model_builders."""
    TCN = 'tcn'
    LSTM = 'lstm'
    SHARED_ENCODER = 'shared_encoder'
    XGBOOST = 'xgboost'
    ATTENTION_LSTM = 'attention_lstm'

    @classmethod
    def all(cls) -> list[str]:
        return [cls.TCN, cls.LSTM, cls.SHARED_ENCODER, cls.XGBOOST, cls.ATTENTION_LSTM]

    @classmethod
    def is_valid(cls, model_type: str) -> bool:
        return model_type.lower().strip() in cls.all()


def _validate_model_params(
    feature_dim: int,
    seq_len: int,
    head_hidden: int,
    head_layers: int,
    dense_hidden: int,
) -> None:
    """Validate model configuration parameters."""
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if head_hidden <= 0:
        raise ValueError(f"head_hidden must be positive, got {head_hidden}")
    if head_layers < 1:
        raise ValueError(f"head_layers must be >= 1, got {head_layers}")
    if dense_hidden <= 0:
        raise ValueError(f"dense_hidden must be positive, got {dense_hidden}")


def _build_shared_encoder_model(**kwargs):
    """Build a shared encoder model with given parameters."""
    return _build_buddy_model_shared_encoder(**kwargs)


def _build_xgboost_model(
    feature_dim: int,
    seq_len: int,
    config: dict | None = None,
):
    """Build an XGBoost model for Buddy training."""
    from src.models.model_builders import _build_xgboost_model as _impl
    return _impl(feature_dim, seq_len, config)


def _build_buddy_model_for_type(
    model_type: str,
    *,
    feature_dim: int,
    seq_len: int,
    head_hidden: int = 64,
    head_layers: int = 2,
    head_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
    kernel_regularizer: float = 0.002,
    noise_std: float = 0.03,
):
    """Build a Buddy model based on the specified architecture type.

    Delegates to src.models.model_builders.ModelFactory.build_from_type_string()
    with the same backward-compatible API.
    """
    from src.models.model_builders import _build_buddy_model_for_type as _impl
    return _impl(
        model_type,
        feature_dim=feature_dim,
        seq_len=seq_len,
        head_hidden=head_hidden,
        head_layers=head_layers,
        head_dropout=head_dropout,
        dense_hidden=dense_hidden,
        dense_dropout=dense_dropout,
        kernel_regularizer=kernel_regularizer,
        noise_std=noise_std,
    )
