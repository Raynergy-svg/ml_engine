"""Shared RL runtime utilities (lazy imports + compatibility guards)."""

from __future__ import annotations

import logging
import os

# =============================================================================
# CRITICAL: Force PyTorch CPU-only BEFORE any imports to prevent TF/Metal hang
# =============================================================================
# On macOS, TensorFlow Metal and PyTorch MPS can deadlock when both try to
# access the GPU. Force PyTorch to CPU-only mode.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# Force PyTorch to never use MPS (Apple Silicon GPU)
os.environ["PYTORCH_MPS_DISABLE"] = "1"

logger = logging.getLogger(__name__)

# Optional dependencies (lazy-loaded)
GYM_AVAILABLE = None
SB3_AVAILABLE = None
gym = None
spaces = None
PPO = None
EvalCallback = None
StopTrainingOnNoModelImprovement = None
DummyVecEnv = None
BaseCallback = None
Monitor = None


def _ensure_gym_imported():
    """Lazy import gymnasium only when needed."""
    global gym, spaces, GYM_AVAILABLE
    if GYM_AVAILABLE is None:
        try:
            import gymnasium as _gym
            from gymnasium import spaces as _spaces

            gym = _gym
            spaces = _spaces
            GYM_AVAILABLE = True
        except ImportError:
            GYM_AVAILABLE = False
    return GYM_AVAILABLE


def _ensure_sb3_imported():
    """Lazy import stable-baselines3 only when needed."""
    global PPO, EvalCallback, StopTrainingOnNoModelImprovement
    global DummyVecEnv, BaseCallback, Monitor, SB3_AVAILABLE
    if SB3_AVAILABLE is None:
        try:
            # Force torch to CPU before SB3 imports (prevents MPS/Metal deadlock)
            import torch
            # set_default_device added in PyTorch 2.0
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
            logger.debug("PyTorch forced to CPU device")

            from stable_baselines3 import PPO as _PPO
            from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
            from stable_baselines3.common.callbacks import EvalCallback as _EvalCallback
            from stable_baselines3.common.callbacks import (
                StopTrainingOnNoModelImprovement as _StopTrainingOnNoModelImprovement,
            )
            from stable_baselines3.common.monitor import Monitor as _Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv

            PPO = _PPO
            BaseCallback = _BaseCallback
            EvalCallback = _EvalCallback
            StopTrainingOnNoModelImprovement = _StopTrainingOnNoModelImprovement
            Monitor = _Monitor
            DummyVecEnv = _DummyVecEnv
            SB3_AVAILABLE = True
        except Exception as e:  # noqa: BLE001
            SB3_AVAILABLE = False
            logger.debug("stable-baselines3 not available: %s", type(e).__name__)
    return SB3_AVAILABLE
