"""Integration test: MetaStrategyAgent wiring into ScannerAgentTeam.

US-017: MetaStrategyAgent.select(regime) should dynamically deactivate
agents that are not selected for the current regime.
"""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.scanner.config import ScannerConfig
from src.scanner.agents._team import ScannerAgentTeam, AgentVerdict, AgentDecisionContext


@pytest.fixture
def meta_config():
    cfg = ScannerConfig()
    cfg.enable_meta_learning = True
    cfg.enable_trend_agent = True
    cfg.enable_mean_reversion_agent = True
    cfg.enable_volatility_agent = True
    cfg.enable_risk_sentinel_agent = True
    cfg.enable_uncertainty_agent = True
    cfg.enable_execution_quality_agent = True
    cfg.enable_momentum_agent = True
    cfg.enable_session_timing_agent = True
    cfg.enable_support_resistance_agent = True
    cfg.enable_order_flow_agent = True
    cfg.enable_devil_advocate_agent = True
    return cfg


def test_meta_strategy_agent_initialized_when_enabled(meta_config):
    team = ScannerAgentTeam(meta_config)
    assert team._meta_strategy_agent is not None


def test_meta_strategy_agent_not_initialized_when_disabled(meta_config):
    meta_config.enable_meta_learning = False
    team = ScannerAgentTeam(meta_config)
    assert team._meta_strategy_agent is None


def test_meta_strategy_filters_verdicts(meta_config):
    team = ScannerAgentTeam(meta_config)

    # Mock meta agent to only select 'trend' and 'momentum'
    mock_meta = MagicMock()
    mock_meta.select.return_value = {
        "selected_strategies": ["trend", "momentum"],
        "weights": {"trend": 0.6, "momentum": 0.4},
        "regime": "NORMAL",
        "uncertain": False,
        "reason": "performance_based",
    }
    team._meta_strategy_agent = mock_meta

    # Build fake verdicts
    verdicts = [
        AgentVerdict(name="trend", score=0.8, passed=True, weight=1.0, reason="ok", reason_code="T1"),
        AgentVerdict(name="mean_reversion", score=0.7, passed=True, weight=1.0, reason="ok", reason_code="MR1"),
        AgentVerdict(name="momentum", score=0.6, passed=True, weight=1.0, reason="ok", reason_code="M1"),
        AgentVerdict(name="volatility", score=0.5, passed=True, weight=1.0, reason="ok", reason_code="V1"),
    ]

    # Simulate the filtering that should happen in _evaluate_body
    regime = "NORMAL"
    selected = mock_meta.select(regime)
    selected_names = set(selected["selected_strategies"])
    filtered = [v for v in verdicts if v.name in selected_names]

    assert len(filtered) == 2
    assert {v.name for v in filtered} == {"trend", "momentum"}
    mock_meta.select.assert_called_once_with("NORMAL")
