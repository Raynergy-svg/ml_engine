"""Tests for RL integration configuration schema."""

from __future__ import annotations

from pathlib import Path

from src.training.rl.config import RLIntegrationConfig


def test_rl_integration_config_from_dict_defaults() -> None:
    cfg = RLIntegrationConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.position_sizing.enabled is True
    assert cfg.position_sizing.timesteps == 50_000
    assert cfg.gate_optimization.enabled is False


def test_rl_integration_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(
        "\n".join(
            [
                "rl_integration:",
                "  enabled: true",
                "  auto_train_after_ensemble: true",
                "  position_sizing:",
                "    timesteps: 12345",
                "    learning_rate: 0.001",
                "  gate_optimization:",
                "    enabled: true",
            ]
        )
    )
    cfg = RLIntegrationConfig.from_yaml(str(config_file))
    assert cfg.enabled is True
    assert cfg.auto_train_after_ensemble is True
    assert cfg.position_sizing.timesteps == 12345
    assert cfg.position_sizing.learning_rate == 0.001
    assert cfg.gate_optimization.enabled is True
