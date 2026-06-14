"""
Smoke tests for neural scanner agents.

Verifies:
  1. Each policy extracts features and emits a verdict.
  2. Verdict scores are in [0, 1].
  3. NeuralAgentTeam.evaluate() runs end-to-end with mock context.
  4. Trainer records outcomes and triggers online updates.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import pytest

from src.scanner.agents.neural.policies import (
    TrendPolicy, MomentumPolicy, MeanReversionPolicy, VolatilityPolicy,
)
from src.scanner.agents.neural.team_bridge import NeuralAgentTeam
from src.scanner.agents.neural.trainer import NeuralAgentTrainer
from src.scanner.agents._team import AgentDecisionContext, AgentVerdict
from src.scanner.results import PairAnalysis


class MockConfig:
    min_agent_consensus_ratio = 0.50
    weighted_vote_threshold = 0.55


def _make_mock_context():
    dates = pd.date_range("2026-01-01", periods=30, freq="h")
    df_raw = pd.DataFrame({
        "open": np.random.randn(30).cumsum() + 1.10,
        "high": np.random.randn(30).cumsum() + 1.11,
        "low": np.random.randn(30).cumsum() + 1.09,
        "close": np.random.randn(30).cumsum() + 1.10,
        "volume": np.random.randint(100, 1000, 30),
    }, index=dates)
    df_feat = pd.DataFrame({
        "sma_20": df_raw["close"].rolling(20).mean(),
        "sma_50": df_raw["close"].rolling(20).mean(),
        "adx": np.linspace(20, 30, 30),
        "rsi": np.linspace(45, 55, 30),
        "macd_hist": np.zeros(30),
        "roc_10": np.zeros(30),
        "bb_upper": df_raw["close"] * 1.02,
        "bb_lower": df_raw["close"] * 0.98,
    }, index=dates)
    analysis = PairAnalysis(pair="EUR_USD", direction="LONG", confidence=0.6)
    return AgentDecisionContext(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details={},
        config=MockConfig(),
    )


class TestTrendPolicy:
    def test_evaluate(self):
        agent = TrendPolicy()
        ctx = _make_mock_context()
        vd = agent.evaluate(ctx)
        assert isinstance(vd, AgentVerdict)
        assert vd.name == "neural_trend"
        assert 0.0 <= vd.score <= 1.0
        assert isinstance(vd.passed, bool)


class TestMomentumPolicy:
    def test_evaluate(self):
        agent = MomentumPolicy()
        ctx = _make_mock_context()
        vd = agent.evaluate(ctx)
        assert vd.name == "neural_momentum"
        assert 0.0 <= vd.score <= 1.0


class TestMeanReversionPolicy:
    def test_evaluate(self):
        agent = MeanReversionPolicy()
        ctx = _make_mock_context()
        vd = agent.evaluate(ctx)
        assert vd.name == "neural_mean_reversion"
        assert 0.0 <= vd.score <= 1.0


class TestVolatilityPolicy:
    def test_evaluate(self):
        agent = VolatilityPolicy()
        ctx = _make_mock_context()
        vd = agent.evaluate(ctx)
        assert vd.name == "neural_volatility"
        assert 0.0 <= vd.score <= 1.0


class TestNeuralAgentTeam:
    def test_evaluate_end_to_end(self):
        team = NeuralAgentTeam(MockConfig())
        analysis = PairAnalysis(pair="EUR_USD", direction="LONG", confidence=0.6)
        ctx = _make_mock_context()
        result = team.evaluate(
            analysis=analysis,
            df_raw=ctx.df_raw,
            df_feat=ctx.df_feat,
            gate_details={},
        )
        assert hasattr(result, "weighted_vote_score")
        assert 0.0 <= result.weighted_vote_score <= 1.0

    def test_outcome_update(self):
        team = NeuralAgentTeam(MockConfig())
        verdicts = [
            {"name": "neural_trend", "passed": True, "metadata": {"features": [0.0] * 120}},
            {"name": "neural_momentum", "passed": True, "metadata": {"features": [0.0] * 30}},
        ]
        result = team.update_weights_from_outcome(verdicts, trade_won=True)
        # Should not crash even with few samples
        assert isinstance(result, dict)


class TestNeuralAgentTrainer:
    def test_instantiates(self):
        trainer = NeuralAgentTrainer()
        assert len(trainer.agents) == 4

    def test_batch_train_with_mock_data(self):
        trainer = NeuralAgentTrainer()
        # Create synthetic trade history for trend agent
        history = []
        for _ in range(60):
            history.append({
                "agent_name": "neural_trend",
                "features": np.random.randn(120).astype(np.float32).tolist(),
                "score": 0.7,
                "passed": True,
                "trade_won": np.random.rand() > 0.4,
            })
        losses = trainer.batch_train(history, epochs=2)
        assert "neural_trend" in losses
        assert losses["neural_trend"] >= 0.0
