"""
Reinforcement Learning modules for ML Engine Trading Bot.

This package provides RL-based enhancements to the trading system:
- Gate threshold optimization (SAC)
- Optimal exit timing (PPO)
- Position sizing (PPO) - see rl_position_sizing.py in root

All RL models use CPU-only execution to avoid Metal/TensorFlow conflicts.

Note: All imports are lazy to avoid 8+ second startup penalty from stable-baselines3.
"""


# These are imported lazily - the actual SB3/gym import happens on first use
def __getattr__(name):
    """Lazy import for RL classes to avoid slow startup."""
    if name == 'GateThresholdEnv':
        from .gate_threshold_env import GateThresholdEnv
        return GateThresholdEnv
    elif name == 'GateThresholdRL':
        from .gate_threshold_env import GateThresholdRL
        return GateThresholdRL
    elif name == 'OptimalExitEnv':
        from .optimal_exit_env import OptimalExitEnv
        return OptimalExitEnv
    elif name == 'OptimalExitRL':
        from .optimal_exit_env import OptimalExitRL
        return OptimalExitRL
    elif name == 'TradingCallback':
        from .callbacks import TradingCallback
        return TradingCallback
    elif name == 'SharpeStopCallback':
        from .callbacks import SharpeStopCallback
        return SharpeStopCallback
    # Utils functions
    elif name == 'detect_regime':
        from .utils import detect_regime
        return detect_regime
    elif name == 'calculate_sharpe':
        from .utils import calculate_sharpe
        return calculate_sharpe
    elif name == 'normalize_features_for_rl':
        from .utils import normalize_features_for_rl
        return normalize_features_for_rl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Environments
    'GateThresholdEnv',
    'OptimalExitEnv',
    # RL Wrappers
    'GateThresholdRL',
    'OptimalExitRL',
    # Callbacks
    'TradingCallback',
    'SharpeStopCallback',
    # Utilities
    'detect_regime',
    'calculate_sharpe',
    'normalize_features_for_rl',
]
