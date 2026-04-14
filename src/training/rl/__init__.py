"""RL integration package aligned with strategy documentation."""

from src.training.rl.config import (
    RLExitTimingConfig,
    RLGateOptimizationConfig,
    RLInferenceConfig,
    RLIntegrationConfig,
    RLPositionSizingConfig,
    RLRewardModelConfig,
)
from src.training.rl.environments.trading_env import TradingEnv
from src.training.rl.position_sizer import (
    POSITION_LEVELS,
    RLConfig,
    RLPositionSizer,
    get_position_sizer,
    train_rl_position_sizer,
)
from src.training.rl.reward_model import EnsembleRewardModel, RewardModelConfig

__all__ = [
    "EnsembleRewardModel",
    "POSITION_LEVELS",
    "RLConfig",
    "RewardModelConfig",
    "RLPositionSizer",
    "RLExitTimingConfig",
    "RLGateOptimizationConfig",
    "RLInferenceConfig",
    "RLIntegrationConfig",
    "RLPositionSizingConfig",
    "RLRewardModelConfig",
    "TradingEnv",
    "get_position_sizer",
    "train_rl_position_sizer",
]


def __getattr__(name: str):
    """Lazy imports for offline sizer to avoid heavy d3rlpy load at startup."""
    _offline_names = {
        "OfflineSizerIQL",
        "InsufficientTrajectoriesError",
        "train_offline_sizer",
        "load_offline_sizer",
    }
    if name in _offline_names:
        from src.training.rl.offline_sizer import (
            InsufficientTrajectoriesError,
            OfflineSizerIQL,
            load_offline_sizer,
            train_offline_sizer,
        )
        _map = {
            "OfflineSizerIQL": OfflineSizerIQL,
            "InsufficientTrajectoriesError": InsufficientTrajectoriesError,
            "train_offline_sizer": train_offline_sizer,
            "load_offline_sizer": load_offline_sizer,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
