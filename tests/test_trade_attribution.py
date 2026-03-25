"""
Tests for TradeAttributionEngine (US-304, Phase 48).

Covers: trade attribution, alignment logic, aggregate performance,
weight recommendations, persistence, disabled filter.
"""

import json
import os
import pytest
import tempfile

from src.scanner.trade_attribution import (
    TradeAttributionEngine,
    AttributionConfig,
    AgentContribution,
    TradeAttribution,
    AgentPerformance,
    create_default_attribution_engine,
)


def _sample_verdicts():
    """Helper: typical agent verdicts for a BUY trade."""
    return {
        "trend": ("LONG", 0.85, 1.0),
        "mean_reversion": ("SHORT", 0.60, 0.90),
        "momentum": ("LONG", 0.75, 0.95),
        "risk_sentinel": ("NEUTRAL", 0.50, 1.0),
        "volatility": ("LONG", 0.70, 0.85),
    }


class TestAgentContribution:
    def test_basic_creation(self):
        c = AgentContribution(
            agent_name="trend", verdict="LONG", weight=1.0,
            confidence=0.85, aligned_with_outcome=True,
            contribution_score=0.85,
        )
        assert c.agent_name == "trend"
        assert c.aligned_with_outcome is True


class TestTradeAttribution:
    def test_aligned_agents(self):
        agents = [
            AgentContribution("a", "LONG", 1.0, 0.8, True, 0.8),
            AgentContribution("b", "SHORT", 0.9, 0.6, False, -0.54),
            AgentContribution("c", "LONG", 0.95, 0.7, True, 0.665),
        ]
        attr = TradeAttribution(
            trade_id="T1", pair="EUR_USD", direction="BUY",
            outcome="WIN", pnl=25.0, agents=agents,
        )
        assert len(attr.aligned_agents) == 2
        assert len(attr.misaligned_agents) == 1

    def test_alignment_ratio(self):
        agents = [
            AgentContribution("a", "LONG", 1.0, 0.8, True, 0.8),
            AgentContribution("b", "SHORT", 1.0, 0.6, False, -0.6),
        ]
        attr = TradeAttribution(
            trade_id="T1", pair="EUR_USD", direction="BUY",
            outcome="WIN", pnl=25.0, agents=agents,
        )
        assert attr.alignment_ratio == pytest.approx(0.5)

    def test_alignment_ratio_empty(self):
        attr = TradeAttribution(
            trade_id="T1", pair="EUR_USD", direction="BUY",
            outcome="WIN", pnl=25.0, agents=[],
        )
        assert attr.alignment_ratio == 0.0


class TestTradeAttributionEngine:
    def test_attribute_winning_buy(self):
        e = TradeAttributionEngine()
        attr = e.attribute_trade(
            trade_id="T1", pair="EUR_USD", direction="BUY",
            outcome="WIN", pnl=25.0,
            agent_verdicts=_sample_verdicts(),
        )
        assert attr.outcome == "WIN"
        assert len(attr.agents) == 5
        # trend=LONG on BUY WIN → aligned
        trend = next(a for a in attr.agents if a.agent_name == "trend")
        assert trend.aligned_with_outcome is True
        # mean_reversion=SHORT on BUY WIN → misaligned
        mr = next(a for a in attr.agents if a.agent_name == "mean_reversion")
        assert mr.aligned_with_outcome is False

    def test_attribute_losing_buy(self):
        e = TradeAttributionEngine()
        attr = e.attribute_trade(
            trade_id="T2", pair="EUR_USD", direction="BUY",
            outcome="LOSS", pnl=-15.0,
            agent_verdicts=_sample_verdicts(),
        )
        # trend=LONG on BUY LOSS → misaligned
        trend = next(a for a in attr.agents if a.agent_name == "trend")
        assert trend.aligned_with_outcome is False
        # mean_reversion=SHORT on BUY LOSS → aligned (correctly bearish)
        mr = next(a for a in attr.agents if a.agent_name == "mean_reversion")
        assert mr.aligned_with_outcome is True

    def test_neutral_agent_aligned_on_win(self):
        e = TradeAttributionEngine()
        attr = e.attribute_trade(
            trade_id="T3", pair="EUR_USD", direction="BUY",
            outcome="WIN", pnl=10.0,
            agent_verdicts={"risk": ("NEUTRAL", 0.5, 1.0)},
        )
        risk = attr.agents[0]
        assert risk.aligned_with_outcome is True  # Neutral + WIN = aligned

    def test_neutral_agent_misaligned_on_loss(self):
        e = TradeAttributionEngine()
        attr = e.attribute_trade(
            trade_id="T4", pair="EUR_USD", direction="BUY",
            outcome="LOSS", pnl=-10.0,
            agent_verdicts={"risk": ("NEUTRAL", 0.5, 1.0)},
        )
        risk = attr.agents[0]
        assert risk.aligned_with_outcome is False  # Neutral + LOSS = misaligned

    def test_get_agent_performance(self):
        e = TradeAttributionEngine()
        verdicts = {"trend": ("LONG", 0.85, 1.0)}
        for i in range(8):
            e.attribute_trade(f"T{i}", "EUR_USD", "BUY", "WIN", 10.0, verdicts)
        for i in range(2):
            e.attribute_trade(f"L{i}", "EUR_USD", "BUY", "LOSS", -5.0, verdicts)

        perf = e.get_agent_performance()
        assert "trend" in perf
        assert perf["trend"].trades_participated == 10
        assert perf["trend"].accuracy_rate == pytest.approx(0.8)

    def test_weight_recommendations_good_agent(self):
        e = TradeAttributionEngine()
        verdicts = {"good_agent": ("LONG", 0.90, 1.0)}
        for i in range(12):
            e.attribute_trade(f"T{i}", "EUR_USD", "BUY", "WIN", 10.0, verdicts)

        recs = e.get_weight_recommendations()
        assert recs["good_agent"] >= 1.0  # Should boost

    def test_weight_recommendations_bad_agent(self):
        e = TradeAttributionEngine()
        verdicts = {"bad_agent": ("LONG", 0.90, 1.0)}
        for i in range(12):
            e.attribute_trade(f"T{i}", "EUR_USD", "BUY", "LOSS", -10.0, verdicts)

        recs = e.get_weight_recommendations()
        assert recs["bad_agent"] < 1.0  # Should reduce

    def test_weight_recommendations_insufficient_data(self):
        e = TradeAttributionEngine()
        verdicts = {"new_agent": ("LONG", 0.80, 1.0)}
        for i in range(5):
            e.attribute_trade(f"T{i}", "EUR_USD", "BUY", "WIN", 10.0, verdicts)

        recs = e.get_weight_recommendations()
        assert recs["new_agent"] == 1.0  # Insufficient data

    def test_disabled_engine(self):
        config = AttributionConfig(enabled=False)
        e = TradeAttributionEngine(config)
        attr = e.attribute_trade("T1", "EUR_USD", "BUY", "WIN", 10.0, _sample_verdicts())
        assert len(attr.agents) == 0  # No attribution when disabled

    def test_history_bounded(self):
        config = AttributionConfig(max_history=20)
        e = TradeAttributionEngine(config)
        for i in range(30):
            e.attribute_trade(f"T{i}", "EUR_USD", "BUY", "WIN", 10.0,
                              {"trend": ("LONG", 0.8, 1.0)})
        assert len(e.history) == 20

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "attr.json")
            config = AttributionConfig(state_file=state_file)
            e = TradeAttributionEngine(config)

            e.attribute_trade("T1", "EUR_USD", "BUY", "WIN", 25.0,
                              {"trend": ("LONG", 0.85, 1.0)})
            e.save_state()

            e2 = TradeAttributionEngine(config)
            assert len(e2.history) == 1
            assert e2.history[0].trade_id == "T1"
            assert len(e2.history[0].agents) == 1

    def test_sell_direction_attribution(self):
        e = TradeAttributionEngine()
        attr = e.attribute_trade(
            "T1", "EUR_USD", "SELL", "WIN", 20.0,
            agent_verdicts={"trend": ("SHORT", 0.85, 1.0)},
        )
        assert attr.agents[0].aligned_with_outcome is True  # SHORT + SELL WIN


class TestAttributionFactory:
    def test_create_default(self):
        e = create_default_attribution_engine()
        assert isinstance(e, TradeAttributionEngine)

    def test_create_with_config(self):
        config = AttributionConfig(max_history=200)
        e = create_default_attribution_engine(config=config)
        assert e.config.max_history == 200
