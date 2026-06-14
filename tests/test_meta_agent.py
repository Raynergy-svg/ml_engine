"""Tests for meta-agent strategy selection (US-017)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.autonomy.meta_agent import MetaAgentConfig, MetaStrategyAgent


class TestMetaStrategyAgent:
    def test_init(self):
        agent = MetaStrategyAgent()
        assert "trend" in agent.cfg.strategies

    def test_observe_and_select(self):
        agent = MetaStrategyAgent(MetaAgentConfig(strategies=["A", "B", "C"], top_k=2))
        for i in range(10):
            agent.observe("A", 1.0)
            agent.observe("B", -0.5)
            agent.observe("C", 0.2)
        result = agent.select("NORMAL")
        assert len(result["selected_strategies"]) <= 2
        assert result["regime"] == "NORMAL"
        assert "weights" in result

    def test_exploration(self):
        np.random.seed(42)
        agent = MetaStrategyAgent(MetaAgentConfig(exploration_rate=1.0, top_k=2))
        result = agent.select("HIGH")
        assert result["uncertain"] is True
        assert result["reason"] == "exploration"

    def test_uncertain_when_scores_close(self):
        agent = MetaStrategyAgent(MetaAgentConfig(strategies=["A", "B"], top_k=2))
        # Give identical scores
        for _ in range(5):
            agent.observe("A", 0.0)
            agent.observe("B", 0.0)
        result = agent.select("LOW")
        assert result["uncertain"] is True
