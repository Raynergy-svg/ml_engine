"""Tests for LLM Macro Agent integration with ScannerAgentTeam (US-005)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.scanner.agents._team import AgentDecisionContext, ScannerAgentTeam


class MockConfig:
    """Minimal config mock that satisfies ScannerAgentTeam."""

    def __init__(self, **kwargs):
        self.enable_llm_macro_agent = False
        self.enable_llm_macro_shadow = True
        self.enable_trend_agent = False
        self.enable_mean_reversion_agent = False
        self.enable_volatility_agent = False
        self.enable_risk_sentinel_agent = False
        self.enable_uncertainty_agent = False
        self.enable_execution_quality_agent = False
        self.enable_news_risk_agent = False
        self.enable_multi_timeframe_agent = False
        self.enable_pair_performance_agent = False
        self.enable_momentum_agent = False
        self.enable_session_timing_agent = False
        self.enable_support_resistance_agent = False
        self.enable_order_flow_agent = False
        self.enable_trader_readiness_agent = False
        self.enable_devil_advocate_agent = False
        self.enable_devil_advocate = False
        self.regime_disabled_agents = {}
        self.enable_episodic_memory = False
        self.briefing_snapshot_every_n_cycles = 12
        self.disable_briefing_snapshot = True
        self.enable_calendar_blackout = False
        self.news_sentiment_fast_mode = True
        self.enable_news_sentiment = False
        self.devil_advocate_block_threshold = 0.60
        self.devil_advocate_warn_threshold = 0.40
        self.enable_trend_agent_hard_veto = False
        self.enable_mean_reversion_veto = False
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestLLMMacroIntegration:

    def test_shadow_mode_instantiated(self):
        """When shadow=True and agent=False, LLM macro agent still exists for logging."""
        config = MockConfig(enable_llm_macro_shadow=True, enable_llm_macro_agent=False)
        team = ScannerAgentTeam(config)
        assert team._llm_macro_agent is not None

    def test_agent_disabled_no_agent(self):
        """When both flags are False, no LLM macro agent."""
        config = MockConfig(enable_llm_macro_shadow=False, enable_llm_macro_agent=False)
        team = ScannerAgentTeam(config)
        assert team._llm_macro_agent is None

    def test_config_fields_exist(self):
        """ScannerConfig should have LLM macro agent fields."""
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        assert hasattr(cfg, "enable_llm_macro_agent")
        assert hasattr(cfg, "enable_llm_macro_shadow")
