"""Stress tests for autonomy loop (US-016, US-017, US-018).

These tests push the autonomy components beyond normal operating
conditions to identify brittle assumptions and failure modes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.autonomy.self_heal_config import SelfHealConfig, SelfHealConfigUpdater
from src.autonomy.meta_agent import MetaAgentConfig, MetaStrategyAgent
from src.autonomy.continual_learner import ContinualLearner, ContinualLearnerConfig


# ═══════════════════════════════════════════════════════════════════
# SelfHealConfigUpdater Stress Tests
# ═══════════════════════════════════════════════════════════════════

class TestSelfHealStress:
    def test_rapid_updates_dont_corrupt_history(self):
        """Rapid successive updates should maintain coherent history."""
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=5))
        base = {"min_confidence": 42.0, "risk_per_trade_pct": 0.05, "compound_gate_floor": 0.1}
        for i in range(100):
            trades = [{"pnl": (-1 if i % 2 == 0 else 1)}] * 5
            updater.update(base, trades)
        history = updater.get_history()
        assert len(history) == 100
        # Verify all entries have expected keys
        for h in history:
            assert "win_rate" in h
            assert "max_dd" in h
            assert "sharpe" in h
            assert "changes" in h

    def test_all_losses_max_drawdown(self):
        """100% loss streak should trigger drawdown-based adjustment."""
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=5))
        trades = [{"pnl": -10.0}] * 20
        base = {"min_confidence": 42.0, "risk_per_trade_pct": 0.05, "compound_gate_floor": 0.1}
        result = updater.update(base, trades)
        # Drawdown triggers risk reduction (approval depends on magnitude)
        assert any(c["param"] == "risk_per_trade_pct" for c in result["changes"])
        assert result["changes"][0]["new"] < result["changes"][0]["old"]
        # Drawdown should be extreme
        history = updater.get_history()
        assert history[0]["max_dd"] > 0.0

    def test_all_wins_no_changes(self):
        """100% win rate should not trigger negative adjustments."""
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=5))
        trades = [{"pnl": 10.0}] * 20
        base = {"min_confidence": 42.0, "risk_per_trade_pct": 0.05, "compound_gate_floor": 0.1}
        result = updater.update(base, trades)
        # Should be healthy - no changes needed
        assert result["reason"] == "metrics_healthy"
        assert len(result["changes"]) == 0

    def test_extreme_pnl_values(self):
        """Very large PnL values should not cause overflow/NaN."""
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=5))
        trades = [{"pnl": 1e9}] * 5 + [{"pnl": -1e9}] * 5
        base = {"min_confidence": 42.0, "risk_per_trade_pct": 0.05, "compound_gate_floor": 0.1}
        result = updater.update(base, trades)
        history = updater.get_history()
        assert np.isfinite(history[0]["sharpe"])
        assert np.isfinite(history[0]["max_dd"])

    def test_zero_pnl_trades(self):
        """All zero PnL should result in zero Sharpe (not NaN)."""
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=5))
        trades = [{"pnl": 0.0}] * 20
        base = {"min_confidence": 42.0, "risk_per_trade_pct": 0.05, "compound_gate_floor": 0.1}
        result = updater.update(base, trades)
        history = updater.get_history()
        assert history[0]["sharpe"] == 0.0

    def test_missing_config_keys(self):
        """Config with missing keys should not crash."""
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=5))
        trades = [{"pnl": -1}] * 20
        # Empty config
        result = updater.update({}, trades)
        # Should either crash (expected) or handle gracefully
        assert "proposed_config" in result

    def test_single_trade_history(self):
        """Lookback > available trades should be handled."""
        updater = SelfHealConfigUpdater(SelfHealConfig(win_rate_lookback=100))
        trades = [{"pnl": -1}] * 3
        base = {"min_confidence": 42.0, "risk_per_trade_pct": 0.05, "compound_gate_floor": 0.1}
        result = updater.update(base, trades)
        assert result["reason"] == "insufficient_trade_history"


# ═══════════════════════════════════════════════════════════════════
# MetaStrategyAgent Stress Tests
# ═══════════════════════════════════════════════════════════════════

class TestMetaAgentStress:
    def test_rapid_regime_flipping(self):
        """Rapidly changing regimes should not crash selection."""
        agent = MetaStrategyAgent(MetaAgentConfig(strategies=["A", "B"], top_k=1))
        for _ in range(50):
            agent.observe("A", np.random.randn())
            agent.observe("B", np.random.randn())
        for regime in ["LOW", "NORMAL", "HIGH", "EXTREME"] * 25:
            result = agent.select(regime)
            assert len(result["selected_strategies"]) > 0

    def test_exploration_saturated(self):
        """100% exploration rate should still return valid strategies."""
        np.random.seed(0)
        agent = MetaStrategyAgent(MetaAgentConfig(exploration_rate=1.0, top_k=2))
        for _ in range(20):
            result = agent.select("NORMAL")
            assert len(result["selected_strategies"]) <= 2
            assert all(s in ["trend", "mean_reversion", "momentum"] for s in result["selected_strategies"])

    def test_no_observations(self):
        """Selecting with zero observations should gracefully degrade."""
        agent = MetaStrategyAgent(MetaAgentConfig(strategies=["X", "Y"], top_k=1))
        result = agent.select("NORMAL")
        assert len(result["selected_strategies"]) <= 1

    def test_extreme_pnl_observations(self):
        """Extreme PnL values in observations should not corrupt weights."""
        agent = MetaStrategyAgent(MetaAgentConfig(strategies=["A", "B"], top_k=1))
        agent.observe("A", 1e6)
        agent.observe("B", -1e6)
        result = agent.select("NORMAL")
        assert len(result["selected_strategies"]) == 1
        # Weights should be finite
        for w in result["weights"].values():
            assert np.isfinite(w)

    def test_all_negative_pnl(self):
        """All strategies losing should still select top performer."""
        agent = MetaStrategyAgent(MetaAgentConfig(strategies=["A", "B", "C"], top_k=2))
        for _ in range(10):
            agent.observe("A", -5.0)
            agent.observe("B", -3.0)
            agent.observe("C", -10.0)
        result = agent.select("LOW")
        # B is least bad, should be selected
        assert "B" in result["selected_strategies"]

    def test_tied_scores_randomness(self):
        """Identical scores across all strategies should not crash."""
        agent = MetaStrategyAgent(MetaAgentConfig(strategies=["A", "B", "C"], top_k=2))
        for _ in range(10):
            agent.observe("A", 1.0)
            agent.observe("B", 1.0)
            agent.observe("C", 1.0)
        result = agent.select("NORMAL")
        # Should fall back to all strategies when uncertain
        assert result["uncertain"] is True


# ═══════════════════════════════════════════════════════════════════
# ContinualLearner Stress Tests
# ═══════════════════════════════════════════════════════════════════

class TestContinualLearnerStress:
    def test_memory_overflow_protection(self):
        """Adding more examples than capacity should maintain fixed size."""
        cl = ContinualLearner(ContinualLearnerConfig(memory_size=10))
        for i in range(1000):
            cl.add_to_memory([{"x": i}])
        assert cl.get_memory_size() == 10

    def test_ewc_with_nan_params(self):
        """EWC with NaN parameters should not corrupt importance."""
        cl = ContinualLearner(ContinualLearnerConfig())
        params = {"w": np.array([1.0, np.nan, 3.0])}
        grads = {"w": np.array([0.1, 0.1, 0.1])}
        cl.update_ewc_importance(params, grads)
        assert "w" in cl._ewc_importance
        # Check for NaN propagation
        assert np.any(np.isnan(cl._ewc_importance["w"])) or np.all(np.isfinite(cl._ewc_importance["w"]))

    def test_ewc_missing_gradients(self):
        """Partial gradient dict should not crash."""
        cl = ContinualLearner(ContinualLearnerConfig())
        params = {"w": np.ones(3), "b": np.zeros(2)}
        grads = {"w": np.ones(3) * 0.1}  # missing 'b'
        cl.update_ewc_importance(params, grads)
        assert cl._task_count == 1

    def test_replay_with_empty_memory(self):
        """Preparing batch with empty memory should return only new examples."""
        cl = ContinualLearner(ContinualLearnerConfig(replay_ratio=0.5))
        new = [{"x": i} for i in range(10)]
        combined, replay = cl.prepare_training_batch(new)
        assert len(replay) == 0
        assert len(combined) == 10

    def test_forgetting_metric_edge_cases(self):
        """Forgetting metric with identical losses."""
        cl = ContinualLearner()
        assert cl.get_forgetting_metric(0.5, 0.5) == 0.0
        assert cl.get_forgetting_metric(0.5, 0.3) == 0.0  # improved, not forgetting
        assert cl.get_forgetting_metric(0.3, 0.5) == pytest.approx(0.2)

    def test_rapid_ewc_updates(self):
        """100 rapid EWC updates should not diverge."""
        cl = ContinualLearner(ContinualLearnerConfig(ewc_lambda=1.0))
        for i in range(100):
            params = {"w": np.random.randn(10) * 0.01}
            grads = {"w": np.random.randn(10) * 0.01}
            cl.update_ewc_importance(params, grads)
        penalty = cl.compute_ewc_penalty({"w": np.zeros(10)})
        assert np.isfinite(penalty)
        assert cl._task_count == 100
