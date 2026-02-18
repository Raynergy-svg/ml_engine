"""Deprecated compatibility shim for RL position sizing.

This module has moved to:
- ``src.training.rl.position_sizer``
- ``src.training.rl.environments.trading_env``
"""

from __future__ import annotations

import warnings

from src.training.rl.position_sizer import (
    GYM_AVAILABLE,
    POSITION_LEVELS,
    RLConfig,
    RLPositionSizer,
    RL_MODEL_PATH,
    RL_SCALER_PATH,
    SB3_AVAILABLE,
    _ensure_gym_imported,
    _ensure_sb3_imported,
    get_position_sizer,
    train_rl_position_sizer,
)
from src.training.rl.environments.trading_env import TradingEnv

warnings.warn(
    "Importing from 'rl_position_sizing' is deprecated. "
    "Use 'src.training.rl.position_sizer' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "GYM_AVAILABLE",
    "POSITION_LEVELS",
    "RLConfig",
    "RLPositionSizer",
    "RL_MODEL_PATH",
    "RL_SCALER_PATH",
    "SB3_AVAILABLE",
    "TradingEnv",
    "_ensure_gym_imported",
    "_ensure_sb3_imported",
    "get_position_sizer",
    "train_rl_position_sizer",
]

