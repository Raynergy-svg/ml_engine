"""Compatibility wrapper for RL gate threshold optimization."""

from __future__ import annotations

from src.rl.gate_threshold_env import (
    GATE_RL_MODEL_PATH,
    GATE_RL_SCALER_PATH,
    GateRLConfig as GateOptimizerConfig,
    GateThresholdEnv,
    GateThresholdRL as RLGateOptimizer,
)

__all__ = [
    "GATE_RL_MODEL_PATH",
    "GATE_RL_SCALER_PATH",
    "GateOptimizerConfig",
    "GateThresholdEnv",
    "RLGateOptimizer",
]

