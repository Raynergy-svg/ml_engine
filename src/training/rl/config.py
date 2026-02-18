"""Configuration schema for RL integration components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from src.utils import load_config


@dataclass
class RLPositionSizingConfig:
    """Configuration for PPO-based RL position sizing."""

    enabled: bool = True
    model_path: str = "trained_data/models/rl_position_sizer.zip"
    scaler_path: str = "trained_data/models/rl_scaler.pkl"

    # Training settings (optimized for CPU training)
    timesteps: int = 50_000  # Reduced from 500k - 31x data reuse was excessive
    learning_rate: float = 3e-4
    n_steps: int = 256  # Reduced from 2048 for faster first rollout on CPU
    batch_size: int = 32  # Reduced from 64 for lower memory usage
    n_epochs: int = 5  # Reduced from 10 for faster training cycles
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Environment settings
    sequence_length: int = 60
    max_position_pct: float = 0.10
    min_position_pct: float = 0.01

    # Reward shaping
    sharpe_weight: float = 0.1
    drawdown_penalty: float = 0.5
    win_rate_bonus: float = 0.2

    # Risk limits
    max_drawdown_pct: float = 0.10
    daily_loss_limit_pct: float = 0.03

    # Graceful fallback when RL is unavailable
    fallback_pct: float = 0.01


@dataclass
class RLGateOptimizationConfig:
    """Configuration for SAC-based gate threshold optimization."""

    enabled: bool = False
    model_path: str = "trained_data/models/sac_gate_thresholds.zip"

    # Training settings
    timesteps: int = 100_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256

    # Bounds for learned thresholds
    confidence_bounds: Tuple[float, float] = (0.5, 0.8)
    momentum_bounds: Tuple[float, float] = (0.1, 0.5)
    risk_bounds: Tuple[float, float] = (0.01, 0.05)
    meta_bounds: Tuple[float, float] = (0.5, 0.7)

    # Defaults when model is unavailable
    default_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "confidence": 0.60,
            "momentum": 0.20,
            "risk": 0.025,
            "meta": 0.55,
        }
    )


@dataclass
class RLExitTimingConfig:
    """Configuration for PPO-based optimal exit timing."""

    enabled: bool = False
    model_path: str = "trained_data/models/ppo_optimal_exit.zip"

    # Training settings
    timesteps: int = 100_000
    learning_rate: float = 3e-4

    # Action encoding
    hold_action: int = 0
    take_profit_action: int = 1
    stop_loss_action: int = 2

    # Exit thresholds
    min_profit_for_tp: float = 0.002
    max_loss_for_sl: float = -0.003
    max_bars_held: int = 24


@dataclass
class RLRewardModelConfig:
    """Configuration for optional learned reward shaping model."""

    enabled: bool = False
    use_ensemble_predictions: bool = True
    include_risk_metrics: bool = True
    reward_horizon_bars: int = 24
    normalize_rewards: bool = True
    weight_in_env: float = 0.3


@dataclass
class RLInferenceConfig:
    """Configuration for combining RL outputs with ensemble inference."""

    position_size_source: str = "rl"  # rl, fixed, kelly
    threshold_adaptation_weight: float = 0.5
    exit_timing_confidence: float = 0.7
    fallback_on_error: bool = True
    log_rl_decisions: bool = True


@dataclass
class RLIntegrationConfig:
    """Top-level RL integration configuration."""

    enabled: bool = False
    auto_train_after_ensemble: bool = True
    position_sizing: RLPositionSizingConfig = field(
        default_factory=RLPositionSizingConfig
    )
    gate_optimization: RLGateOptimizationConfig = field(
        default_factory=RLGateOptimizationConfig
    )
    exit_timing: RLExitTimingConfig = field(default_factory=RLExitTimingConfig)
    reward_model: RLRewardModelConfig = field(default_factory=RLRewardModelConfig)
    inference: RLInferenceConfig = field(default_factory=RLInferenceConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "RLIntegrationConfig":
        """Build configuration from either root config dict or rl_integration subsection."""
        if data is None:
            return cls()

        if "rl_integration" in data and isinstance(data["rl_integration"], dict):
            rl_data = data["rl_integration"]
        else:
            rl_data = data

        pos_data = rl_data.get("position_sizing", {}) or {}
        gate_data = rl_data.get("gate_optimization", {}) or {}
        exit_data = rl_data.get("exit_timing", {}) or {}
        reward_data = rl_data.get("reward_model", {}) or {}
        inf_data = rl_data.get("inference", {}) or {}

        return cls(
            enabled=bool(rl_data.get("enabled", False)),
            auto_train_after_ensemble=bool(
                rl_data.get("auto_train_after_ensemble", True)
            ),
            position_sizing=RLPositionSizingConfig(**pos_data),
            gate_optimization=RLGateOptimizationConfig(**gate_data),
            exit_timing=RLExitTimingConfig(**exit_data),
            reward_model=RLRewardModelConfig(**reward_data),
            inference=RLInferenceConfig(**inf_data),
        )

    @classmethod
    def from_yaml(cls, path: str = "config/config_improved_H1.yaml") -> "RLIntegrationConfig":
        """Load configuration from YAML config file."""
        cfg = load_config(path)
        return cls.from_dict(cfg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for metadata/logging."""
        return asdict(self)

