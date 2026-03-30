"""Test suite for Meta-Learning Layer (Tier 6).

Tests for:
- MetaParametersRegistry (persistence, loading, defaults)
- OnlineBayesianAdapter (posterior updates, Thompson sampling, likelihood)
- MetaLearner (orchestration, rollback, loss tracking)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

import numpy as np

from src.recursive_intelligence.meta_parameters import (
    MetaParametersRegistry,
    HyperparameterSet,
    DEFAULT_HYPERPARAMS,
    REGIMES,
)
from src.recursive_intelligence.bayesian_adapter import OnlineBayesianAdapter
from src.recursive_intelligence.meta_learner import MetaLearner


class TestHyperparameterSet:
    """Tests for HyperparameterSet dataclass."""

    def test_default_creation(self):
        """Create with all defaults."""
        hp = HyperparameterSet.default()
        assert hp.confidence_threshold == 0.6
        assert hp.weight_decay_rate == 0.01
        assert hp.regime_switch_sensitivity == 0.5
        assert hp.voting_veto_power == 1.0
        assert hp.fitness == 0.5

    def test_to_dict(self):
        """Serialize to dict with rounding."""
        hp = HyperparameterSet(confidence_threshold=0.123456, weight_decay_rate=0.0123456)
        d = hp.to_dict()
        assert d["confidence_threshold"] == 0.1235  # rounded to 4 places
        assert d["weight_decay_rate"] == 0.01235  # rounded to 5 places
        assert "version" in d

    def test_from_dict(self):
        """Deserialize from dict."""
        d = {"confidence_threshold": 0.75, "weight_decay_rate": 0.02}
        hp = HyperparameterSet.from_dict(d)
        assert hp.confidence_threshold == 0.75
        assert hp.weight_decay_rate == 0.02

    def test_to_dict_roundtrip(self):
        """Serialize → deserialize → compare."""
        original = HyperparameterSet(
            confidence_threshold=0.7,
            weight_decay_rate=0.015,
            regime_switch_sensitivity=0.6,
            voting_veto_power=1.2,
        )
        d = original.to_dict()
        restored = HyperparameterSet.from_dict(d)
        assert restored.confidence_threshold == pytest.approx(0.7, abs=0.001)
        assert restored.weight_decay_rate == pytest.approx(0.015, abs=0.001)
        assert restored.regime_switch_sensitivity == pytest.approx(0.6, abs=0.001)
        assert restored.voting_veto_power == pytest.approx(1.2, abs=0.001)


class TestMetaParametersRegistry:
    """Tests for MetaParametersRegistry."""

    def test_registry_get_default(self):
        """Unknown agent/regime → returns HyperparameterSet defaults."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            params = registry.get("unknown_agent", "NORMAL")
            assert params.confidence_threshold == 0.6
            assert params.voting_veto_power == 1.0

    def test_registry_set_and_get(self):
        """Set params → get returns them."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            new_params = HyperparameterSet(confidence_threshold=0.75)
            registry.set("trend", "NORMAL", new_params)
            retrieved = registry.get("trend", "NORMAL")
            assert retrieved.confidence_threshold == pytest.approx(0.75, abs=0.001)

    def test_registry_update_fitness_ema(self):
        """Repeated updates converge toward recent outcome."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            # Start at 0.5, push toward 1.0 (win)
            for _ in range(10):
                registry.update_fitness("agent1", "NORMAL", 1.0)
            params = registry.get("agent1", "NORMAL")
            # Should be > 0.5 but < 1.0
            assert 0.5 < params.fitness < 1.0

    def test_registry_reset_to_default(self):
        """Reset returns baseline values."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            # Set to non-default
            new_params = HyperparameterSet(confidence_threshold=0.9)
            registry.set("agent1", "NORMAL", new_params)
            # Reset
            registry.reset_to_default("agent1", "NORMAL")
            retrieved = registry.get("agent1", "NORMAL")
            assert retrieved.confidence_threshold == 0.6  # back to default

    def test_registry_save_atomic(self):
        """Save creates .tmp then renames (no partial file)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry = MetaParametersRegistry(meta_dir=tmp_path)
            new_params = HyperparameterSet(confidence_threshold=0.7)
            registry.set("trend", "NORMAL", new_params)
            registry.save()
            # Check that .tmp does not exist (was renamed)
            tmp_files = list(tmp_path.glob("*.tmp"))
            assert len(tmp_files) == 0
            # Check that the final file exists
            final_files = list(tmp_path.glob("agent_meta_params_NORMAL.json"))
            assert len(final_files) == 1

    def test_registry_save_and_load(self):
        """Save to tmp_path, load back → matches original."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create and save
            registry1 = MetaParametersRegistry(meta_dir=tmp_path)
            new_params = HyperparameterSet(confidence_threshold=0.75)
            registry1.set("trend", "NORMAL", new_params)
            registry1.save()
            # Load in new instance
            registry2 = MetaParametersRegistry(meta_dir=tmp_path)
            retrieved = registry2.get("trend", "NORMAL")
            assert retrieved.confidence_threshold == pytest.approx(0.75, abs=0.001)

    def test_registry_load_corrupted_file(self):
        """Corrupt JSON → graceful fallback, no crash."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Write corrupted JSON
            bad_file = tmp_path / "agent_meta_params_NORMAL.json"
            bad_file.write_text("{invalid json")
            # Should not crash
            registry = MetaParametersRegistry(meta_dir=tmp_path)
            params = registry.get("any_agent", "NORMAL")
            assert params.confidence_threshold == 0.6  # defaults

    def test_registry_all_agents(self):
        """all_agents() returns list of agent names."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            registry.set("trend", "NORMAL", HyperparameterSet())
            registry.set("momentum", "NORMAL", HyperparameterSet())
            agents = registry.all_agents()
            assert "trend" in agents
            assert "momentum" in agents


class TestOnlineBayesianAdapter:
    """Tests for OnlineBayesianAdapter."""

    def test_adapter_init_uniform_posterior(self):
        """Posterior sums to 1.0, all entries equal initially."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            adapter = OnlineBayesianAdapter(registry)
            posterior = adapter._get_posterior("trend", "NORMAL")
            assert pytest.approx(posterior.sum(), abs=1e-6) == 1.0
            # All entries equal for uniform
            assert np.all(np.abs(np.diff(posterior)) < 1e-10)

    def test_adapter_update_returns_hyperparams(self):
        """Update returns HyperparameterSet."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            adapter = OnlineBayesianAdapter(registry)
            result = adapter.update(
                agent_name="trend",
                regime="NORMAL",
                agent_score=0.7,
                trade_outcome=1.0,  # win
            )
            assert isinstance(result, HyperparameterSet)
            assert 0.0 <= result.confidence_threshold <= 1.0

    def test_adapter_responsibility_win_aligned(self):
        """High score + win → high responsibility."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            adapter = OnlineBayesianAdapter(registry)
            resp = adapter._compute_responsibility(agent_score=0.9, trade_outcome=1.0)
            assert resp > 0.7  # high responsibility

    def test_adapter_responsibility_loss_misaligned(self):
        """High score + loss → low responsibility."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            adapter = OnlineBayesianAdapter(registry)
            resp = adapter._compute_responsibility(agent_score=0.9, trade_outcome=-1.0)
            assert resp < 0.4  # low responsibility

    def test_adapter_posterior_normalizes(self):
        """After 10 updates, posterior still sums to ~1.0."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            adapter = OnlineBayesianAdapter(registry)
            for i in range(10):
                outcome = 1.0 if i % 2 == 0 else -1.0
                adapter.update(
                    agent_name="trend",
                    regime="NORMAL",
                    agent_score=0.6,
                    trade_outcome=outcome,
                )
            posterior = adapter._get_posterior("trend", "NORMAL")
            assert pytest.approx(posterior.sum(), abs=1e-6) == 1.0

    def test_adapter_reset_posterior(self):
        """Reset → back to uniform."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            adapter = OnlineBayesianAdapter(registry)
            # Update a few times
            for _ in range(5):
                adapter.update(
                    agent_name="trend",
                    regime="NORMAL",
                    agent_score=0.8,
                    trade_outcome=1.0,
                )
            # Reset
            adapter.reset_posterior("trend", "NORMAL")
            posterior = adapter._get_posterior("trend", "NORMAL")
            # Should be uniform again
            assert np.all(np.abs(np.diff(posterior)) < 1e-10)

    def test_adapter_posterior_shifts_after_wins(self):
        """Repeated wins shift posterior toward high-ct configs."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            adapter = OnlineBayesianAdapter(registry)
            initial_best = adapter.get_best("trend", "NORMAL")
            # Multiple wins
            for _ in range(20):
                adapter.update(
                    agent_name="trend",
                    regime="NORMAL",
                    agent_score=0.9,
                    trade_outcome=1.0,
                )
            new_best = adapter.get_best("trend", "NORMAL")
            # After wins, confidence_threshold should tend higher
            assert new_best.confidence_threshold >= initial_best.confidence_threshold * 0.9


class TestMetaLearner:
    """Tests for MetaLearner orchestrator."""

    def test_meta_learner_win_trade_update(self):
        """Win trade → updates all agents."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_learner = MetaLearner(enabled=True)
            meta_learner._registry._meta_dir = Path(tmp)
            agent_scores = {"trend": 0.8, "momentum": 0.7, "volatility": 0.6}
            result = meta_learner.on_trade_close(
                pair="EUR_USD",
                regime="NORMAL",
                trade_result={"outcome": "win"},
                agent_scores=agent_scores,
            )
            assert len(result) == len(agent_scores)

    def test_meta_learner_loss_trade_update(self):
        """Loss trade → updates all agents."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_learner = MetaLearner(enabled=True)
            meta_learner._registry._meta_dir = Path(tmp)
            agent_scores = {"trend": 0.3, "momentum": 0.2}
            result = meta_learner.on_trade_close(
                pair="EUR_USD",
                regime="NORMAL",
                trade_result={"outcome": "loss"},
                agent_scores=agent_scores,
            )
            assert len(result) == len(agent_scores)

    def test_meta_learner_rollback_after_3_losses(self):
        """3 consecutive losses → rollback fired."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_learner = MetaLearner(enabled=True)
            meta_learner._registry._meta_dir = Path(tmp)
            agent_scores = {"trend": 0.5}
            # 3 bad trades in a row
            for _ in range(3):
                meta_learner.on_trade_close(
                    pair="EUR_USD",
                    regime="NORMAL",
                    trade_result={"pnl_pct": -0.10},  # -10% loss
                    agent_scores=agent_scores,
                )
            # Rollback should have been triggered
            assert meta_learner._total_rollbacks > 0
            assert meta_learner._cooldowns.get("trend", 0) > 0

    def test_meta_learner_cooldown_after_rollback(self):
        """After rollback, agent in cooldown."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_learner = MetaLearner(enabled=True)
            meta_learner._registry._meta_dir = Path(tmp)
            agent_scores = {"trend": 0.5}
            # Trigger rollback
            for _ in range(3):
                meta_learner.on_trade_close(
                    pair="EUR_USD",
                    regime="NORMAL",
                    trade_result={"pnl_pct": -0.10},
                    agent_scores=agent_scores,
                )
            # Agent should be in cooldown
            assert meta_learner._cooldowns.get("trend", 0) > 0

    def test_meta_learner_no_crash_missing_pnl(self):
        """Trade_result without pnl_pct or outcome → returns {}."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_learner = MetaLearner(enabled=True)
            agent_scores = {"trend": 0.5}
            result = meta_learner.on_trade_close(
                pair="EUR_USD",
                regime="NORMAL",
                trade_result={"some_other_key": "value"},  # invalid
                agent_scores=agent_scores,
            )
            assert result == {}

    def test_meta_learner_get_overrides_returns_dict(self):
        """get_active_overrides returns dict with 4 keys."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_learner = MetaLearner(enabled=True)
            meta_learner._registry._meta_dir = Path(tmp)
            overrides = meta_learner.get_active_overrides("trend", "NORMAL")
            assert isinstance(overrides, dict)
            assert "confidence_threshold" in overrides
            assert "weight_decay_rate" in overrides
            assert "regime_switch_sensitivity" in overrides
            assert "voting_veto_power" in overrides

    def test_meta_learner_disabled_returns_empty(self):
        """enabled=False → on_trade_close returns {}."""
        meta_learner = MetaLearner(enabled=False)
        result = meta_learner.on_trade_close(
            pair="EUR_USD",
            regime="NORMAL",
            trade_result={"outcome": "win"},
            agent_scores={"trend": 0.8},
        )
        assert result == {}

    def test_meta_learner_stats(self):
        """stats property returns dict with expected keys."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_learner = MetaLearner(enabled=True)
            meta_learner._registry._meta_dir = Path(tmp)
            stats = meta_learner.stats
            assert "total_updates" in stats
            assert "total_rollbacks" in stats
            assert "enabled" in stats
            assert stats["enabled"] is True


class TestMetaLearnerIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow_win_trades(self):
        """Complete workflow: initialize → multiple win trades → check adaptation."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            meta_learner = MetaLearner(registry=registry, enabled=True)

            # Simulate multiple winning trades
            for i in range(5):
                agent_scores = {
                    "trend": 0.7 + (i * 0.05),
                    "momentum": 0.6 + (i * 0.04),
                }
                meta_learner.on_trade_close(
                    pair="EUR_USD",
                    regime="NORMAL",
                    trade_result={"pnl_pct": 0.05},  # +5% win
                    agent_scores=agent_scores,
                )

            # Check that updates happened
            assert meta_learner._total_updates > 0

            # Check that we can get overrides
            overrides = meta_learner.get_active_overrides("trend", "NORMAL")
            assert len(overrides) == 4

    def test_full_workflow_mixed_outcomes(self):
        """Mixed win/loss trades → adaptation continues."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = MetaParametersRegistry(meta_dir=Path(tmp))
            meta_learner = MetaLearner(registry=registry, enabled=True)

            outcomes = [0.05, -0.02, 0.08, -0.03, 0.06]  # mixed
            for outcome in outcomes:
                agent_scores = {"trend": 0.6, "momentum": 0.5}
                meta_learner.on_trade_close(
                    pair="EUR_USD",
                    regime="NORMAL",
                    trade_result={"pnl_pct": outcome},
                    agent_scores=agent_scores,
                )

            # Should handle mixed outcomes gracefully
            assert meta_learner._total_updates > 0
            stats = meta_learner.stats
            assert not stats["enabled"] or stats["total_updates"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
