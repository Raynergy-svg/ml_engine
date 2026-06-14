"""
End-to-end integration test for SOTA Core + Neural Agents.

Verifies that the factory functions return the correct types when
toggled on, and fall back gracefully when toggled off.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from src.scanner.config import ScannerConfig
from src.scanner.sota_integration import resolve_inference_engine, resolve_agent_team, is_sota_active


class TestSOTAIntegration:
    def test_legacy_fallback_by_default(self):
        cfg = ScannerConfig()
        assert cfg.use_sota_inference is False
        assert cfg.use_neural_agents is False
        assert is_sota_active(cfg) is False

    def test_sota_inference_factory_when_enabled(self):
        cfg = ScannerConfig()
        cfg.use_sota_inference = True
        cfg.sota_model_path = "/nonexistent/model.keras"  # forces fallback
        engine = resolve_inference_engine(cfg)
        # Factory honors the flag and returns SOTAInference
        from src.sota_core.inference import SOTAInference
        assert isinstance(engine, SOTAInference)

    def test_neural_team_factory_when_enabled(self):
        cfg = ScannerConfig()
        cfg.use_neural_agents = True
        team = resolve_agent_team(cfg)
        from src.scanner.agents.neural import NeuralAgentTeam
        assert isinstance(team, NeuralAgentTeam)

    def test_neural_team_fallback_when_disabled(self):
        cfg = ScannerConfig()
        cfg.use_neural_agents = False
        team = resolve_agent_team(cfg)
        from src.scanner.agents import ScannerAgentTeam
        assert isinstance(team, ScannerAgentTeam)

    def test_is_sota_active_true_when_either_enabled(self):
        cfg = ScannerConfig()
        cfg.use_sota_inference = True
        assert is_sota_active(cfg) is True

        cfg.use_sota_inference = False
        cfg.use_neural_agents = True
        assert is_sota_active(cfg) is True
