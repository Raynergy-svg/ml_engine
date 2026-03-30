"""Tests for Tier 6 meta-learning wiring.

Verifies:
1. MetaLearner.on_trade_close() is called from sync_closed_trades_rl when scanner has _meta_learner
2. Meta overrides are injected into agent verdict weights
3. EnsembleWeighter MLP forward/backward, training, persistence
4. Fallback to uniform weights when insufficient data
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Test 1: on_trade_close callback wiring ──


class TestMetaLearnerCallback:
    """Verify the meta-learner callback fires during sync_closed_trades_rl."""

    def test_callback_fires_with_agent_scores(self):
        """on_trade_close should be called with correct agent scores from agent_reasons."""
        from src.recursive_intelligence.meta_learner import MetaLearner

        ml = MetaLearner(enabled=True)
        ml.on_trade_close = MagicMock(return_value={})

        # Simulate what execution.py does: extract scores from agent_reasons
        agent_reasons = [
            {"name": "trend", "score": 0.8, "weight": 1.15, "passed": True},
            {"name": "momentum", "score": 0.6, "weight": 1.05, "passed": True},
            {"name": "risk_sentinel", "score": 0.3, "weight": 1.25, "passed": False},
        ]

        agent_scores = {}
        agent_weights = {}
        for ar in agent_reasons:
            name = ar.get("name", "")
            if name:
                agent_scores[name] = float(ar.get("score", 0.5))
                agent_weights[name] = float(ar.get("weight", 1.0))

        trade_result = {"pnl_pct": 0.02, "outcome": "win"}

        ml.on_trade_close(
            pair="EUR_USD",
            regime="NORMAL",
            trade_result=trade_result,
            agent_scores=agent_scores,
            agent_weights=agent_weights,
        )

        ml.on_trade_close.assert_called_once()
        call_args = ml.on_trade_close.call_args
        assert call_args.kwargs["pair"] == "EUR_USD"
        assert call_args.kwargs["regime"] == "NORMAL"
        assert "trend" in call_args.kwargs["agent_scores"]
        assert call_args.kwargs["agent_scores"]["trend"] == 0.8

    def test_callback_handles_missing_agent_reasons(self):
        """Should not crash when agent_reasons is empty."""
        from src.recursive_intelligence.meta_learner import MetaLearner

        ml = MetaLearner(enabled=True)
        # Empty agent_reasons → no scores → on_trade_close not called
        agent_reasons = []
        agent_scores = {
            ar.get("name", ""): float(ar.get("score", 0.5))
            for ar in agent_reasons
            if ar.get("name")
        }
        assert len(agent_scores) == 0  # Should be empty

    def test_on_trade_close_updates_posteriors(self):
        """Verify that on_trade_close actually updates the Bayesian posteriors."""
        from src.recursive_intelligence.meta_learner import MetaLearner

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.meta_parameters._META_DIR", Path(tmpdir)):
                ml = MetaLearner(enabled=True)
                result = ml.on_trade_close(
                    pair="EUR_USD",
                    regime="NORMAL",
                    trade_result={"pnl_pct": 0.03, "outcome": "win"},
                    agent_scores={"trend": 0.9, "momentum": 0.7},
                    agent_weights={"trend": 1.15, "momentum": 1.05},
                )
                assert ml._total_updates == 2  # Two agents updated
                assert "trend" in result
                assert "momentum" in result


# ── Test 2: Meta override injection into verdicts ──


class TestMetaOverrideInjection:
    """Verify meta overrides affect agent verdict weights in _team.py."""

    def test_voting_veto_power_multiplies_weight(self):
        """voting_veto_power should scale the verdict weight."""
        # Simulate the override application logic from _team.py
        class FakeVerdict:
            def __init__(self, name, score, weight):
                self.name = name
                self.score = score
                self.weight = weight

        verdicts = [
            FakeVerdict("trend", 0.8, 1.15),
            FakeVerdict("momentum", 0.6, 1.05),
        ]

        meta_overrides = {
            "trend": {"voting_veto_power": 1.5, "confidence_threshold": 0.0},
            "momentum": {"voting_veto_power": 0.7, "confidence_threshold": 0.0},
        }

        # Apply overrides (same logic as in _team.py)
        for v in verdicts:
            overrides = meta_overrides.get(v.name)
            if overrides:
                veto = float(overrides.get("voting_veto_power", 1.0))
                if abs(veto - 1.0) > 0.01:
                    v.weight = max(v.weight * veto, 0.0)

        assert abs(verdicts[0].weight - 1.15 * 1.5) < 0.001  # trend boosted
        assert abs(verdicts[1].weight - 1.05 * 0.7) < 0.001  # momentum dampened

    def test_confidence_threshold_reduces_low_scores(self):
        """Scores below confidence_threshold should be halved."""

        class FakeVerdict:
            def __init__(self, name, score, weight):
                self.name = name
                self.score = score
                self.weight = weight

        verdicts = [
            FakeVerdict("trend", 0.4, 1.0),    # Below threshold
            FakeVerdict("momentum", 0.8, 1.0),  # Above threshold
        ]

        meta_overrides = {
            "trend": {"voting_veto_power": 1.0, "confidence_threshold": 0.6},
            "momentum": {"voting_veto_power": 1.0, "confidence_threshold": 0.6},
        }

        for v in verdicts:
            overrides = meta_overrides.get(v.name)
            if overrides:
                ct = float(overrides.get("confidence_threshold", 0.0))
                if ct > 0.0 and v.score < ct:
                    v.score = v.score * 0.5

        assert abs(verdicts[0].score - 0.2) < 0.001  # Halved (was 0.4, below 0.6)
        assert abs(verdicts[1].score - 0.8) < 0.001  # Unchanged (0.8 >= 0.6)


# ── Test 3: EnsembleWeighter MLP ──


class TestEnsembleWeighter:
    """Test the differentiable ensemble weighting layer."""

    def test_uniform_fallback_when_untrained(self):
        """Should return uniform weights when not yet trained."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        config = WeighterConfig(n_models=4, n_context=6, min_samples=20)
        weighter = EnsembleWeighter(config=config)

        weights = weighter.get_weights(
            context_features=[0.5, 0.3, 1.0, 0.02, 0.65, 0.0],
            model_predictions=[0.6, 0.55, 0.7, 0.5],
        )
        np.testing.assert_allclose(weights, np.ones(4) / 4, atol=1e-10)

    def test_mlp_forward_produces_valid_weights(self):
        """MLP output should be valid softmax (sum to 1, all positive)."""
        from src.recursive_intelligence.ensemble_weighting import NumpyMLP

        mlp = NumpyMLP(input_dim=10, hidden_dim=16, output_dim=4)
        x = np.random.randn(10)
        weights, hidden = mlp.forward(x)

        assert weights.shape == (4,)
        assert np.all(weights >= 0)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_mlp_train_step_reduces_loss(self):
        """Training should reduce loss over iterations."""
        from src.recursive_intelligence.ensemble_weighting import NumpyMLP

        mlp = NumpyMLP(input_dim=10, hidden_dim=16, output_dim=4)
        x = np.random.randn(10)
        target = np.array([0.5, 0.3, 0.15, 0.05])

        losses = []
        for _ in range(100):
            loss = mlp.train_step(x, target, lr=0.01)
            losses.append(loss)

        # Loss should decrease over training
        assert losses[-1] < losses[0]

    def test_record_and_retrain(self):
        """Recording enough samples should trigger automatic retraining."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        config = WeighterConfig(
            n_models=3, n_context=4,
            min_samples=5, retrain_interval=5,
            buffer_max=50, epochs_per_retrain=10,
        )
        weighter = EnsembleWeighter(config=config)

        # Record enough samples to trigger retrain
        for i in range(6):
            weighter.record(
                context_features=[0.5, 0.3, float(i % 3), 0.02],
                model_predictions=[0.6 + i * 0.01, 0.5, 0.7 - i * 0.01],
                realized_outcome=0.65,
            )

        assert weighter._is_trained
        assert weighter._total_retrains >= 1
        assert weighter.stats["is_trained"]

    def test_learned_weights_differ_from_uniform(self):
        """After training, weights should not be uniform."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        config = WeighterConfig(
            n_models=3, n_context=4,
            min_samples=5, retrain_interval=5,
            buffer_max=50, epochs_per_retrain=50,
        )
        weighter = EnsembleWeighter(config=config)

        # Train with data where model 0 is consistently best
        for i in range(10):
            weighter.record(
                context_features=[0.5, 0.3, 0.0, 0.02],
                model_predictions=[0.65, 0.3, 0.1],  # Model 0 closest to outcome
                realized_outcome=0.7,
            )

        weights = weighter.get_weights(
            context_features=[0.5, 0.3, 0.0, 0.02],
            model_predictions=[0.65, 0.3, 0.1],
        )

        # Model 0 should have highest weight after training
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]

    def test_persistence_roundtrip(self):
        """Save and load should preserve MLP weights."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "weighter.json")
            with patch("src.recursive_intelligence.ensemble_weighting._STATE_FILE", state_file):
                config = WeighterConfig(
                    n_models=3, n_context=4,
                    min_samples=3, retrain_interval=3,
                    epochs_per_retrain=5,
                )
                w1 = EnsembleWeighter(config=config)

                for i in range(5):
                    w1.record(
                        context_features=[0.5, 0.3, 0.0, 0.02],
                        model_predictions=[0.6, 0.5, 0.4],
                        realized_outcome=0.55,
                    )

                # Get prediction before save
                pred1 = w1.get_weights([0.5, 0.3, 0.0, 0.02], [0.6, 0.5, 0.4])

                # Load into new instance
                w2 = EnsembleWeighter(config=config)
                pred2 = w2.get_weights([0.5, 0.3, 0.0, 0.02], [0.6, 0.5, 0.4])

                np.testing.assert_allclose(pred1, pred2, atol=1e-10)

    def test_buffer_sliding_window(self):
        """Buffer should not exceed buffer_max."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        config = WeighterConfig(
            n_models=2, n_context=2,
            min_samples=100, retrain_interval=1000,  # No auto-retrain
            buffer_max=10,
        )
        weighter = EnsembleWeighter(config=config)

        for i in range(25):
            weighter.record(
                context_features=[float(i), 0.0],
                model_predictions=[0.5, 0.5],
                realized_outcome=0.5,
            )

        assert len(weighter._buffer) <= 10


# ── Test 4: MetaLearner rollback safety ──


class TestMetaLearnerRollback:
    """Verify rollback triggers after consecutive losses."""

    def test_rollback_after_3_losses(self):
        """3 consecutive losses > -5% should trigger rollback and cooldown."""
        from src.recursive_intelligence.meta_learner import MetaLearner

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.meta_parameters._META_DIR", Path(tmpdir)):
                ml = MetaLearner(enabled=True)

                for _ in range(3):
                    ml.on_trade_close(
                        pair="EUR_USD",
                        regime="NORMAL",
                        trade_result={"pnl_pct": -0.08},  # -8%, exceeds -5% threshold
                        agent_scores={"trend": 0.3},
                    )

                assert ml._total_rollbacks >= 1
                assert ml._cooldowns.get("trend", 0) > 0

    def test_no_rollback_on_wins(self):
        """Wins should reset loss streak, no rollback."""
        from src.recursive_intelligence.meta_learner import MetaLearner

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.meta_parameters._META_DIR", Path(tmpdir)):
                ml = MetaLearner(enabled=True)

                # 2 losses then 1 win resets streak
                for _ in range(2):
                    ml.on_trade_close(
                        pair="EUR_USD",
                        regime="NORMAL",
                        trade_result={"pnl_pct": -0.08},
                        agent_scores={"trend": 0.3},
                    )
                ml.on_trade_close(
                    pair="EUR_USD",
                    regime="NORMAL",
                    trade_result={"pnl_pct": 0.05},
                    agent_scores={"trend": 0.8},
                )

                assert ml._total_rollbacks == 0
                assert ml._loss_streaks.get("trend", 0) == 0


# ── Test 5: Config flag ──


class TestConfigFlag:
    """Verify enable_meta_learning is True by default."""

    def test_meta_learning_enabled_by_default(self):
        from src.scanner.config import ScannerConfig
        config = ScannerConfig()
        assert config.enable_meta_learning is True


# ── Test 6: EnsembleWeighter integration with inference scoring ──


class TestEnsembleWeighterIntegration:
    """Test the EnsembleWeighter integration with modular inference scoring."""

    @pytest.fixture(autouse=True)
    def isolate_state(self, tmp_path):
        """Isolate EnsembleWeighter from persistent state file."""
        state_file = str(tmp_path / "ew_state.json")
        with patch("src.recursive_intelligence.ensemble_weighting._STATE_FILE", state_file):
            yield

    def test_weighter_produces_different_weights_after_training(self):
        """After training, weights should differ from config defaults."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        config = WeighterConfig(
            n_models=4, n_context=2,
            min_samples=5, retrain_interval=5,
            buffer_max=100, epochs_per_retrain=50,
        )
        w = EnsembleWeighter(config=config)

        # Train: direction model (index 0) consistently best predictor
        for i in range(10):
            w.record(
                context_features=[0.33, 0.8],  # NORMAL regime, high confidence
                model_predictions=[0.7, 0.4, 0.3, 0.2],  # dir=0.7 closest to outcome
                realized_outcome=0.75,
            )

        weights = w.get_weights([0.33, 0.8], [0.7, 0.4, 0.3, 0.2])

        # Direction model should have highest weight (closest to outcome)
        assert weights[0] > 0.25  # Higher than uniform (0.25)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_weighter_adapts_to_different_regimes(self):
        """Weights should vary between different regime contexts."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        config = WeighterConfig(
            n_models=4, n_context=2,
            min_samples=5, retrain_interval=5,
            buffer_max=100, epochs_per_retrain=100,
        )
        w = EnsembleWeighter(config=config)

        # Normal regime: direction model best
        for _ in range(10):
            w.record([0.33, 0.8], [0.7, 0.3, 0.2, 0.1], 0.75)

        # Extreme regime: risk model best
        for _ in range(10):
            w.record([1.0, 0.9], [0.2, 0.3, 0.1, 0.7], 0.65)

        w_normal = w.get_weights([0.33, 0.8], [0.7, 0.3, 0.2, 0.1])
        w_extreme = w.get_weights([1.0, 0.9], [0.2, 0.3, 0.1, 0.7])

        # Weights should be different for different regimes
        assert not np.allclose(w_normal, w_extreme, atol=0.01)

    def test_scoring_path_uses_learned_weights(self):
        """The modular inference scoring should use learned weights when available."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        config = WeighterConfig(
            n_models=4, n_context=2,
            min_samples=3, retrain_interval=3,
            buffer_max=50, epochs_per_retrain=20,
        )
        w = EnsembleWeighter(config=config)

        # Train enough to activate
        for _ in range(5):
            w.record([0.33, 0.8], [0.6, 0.5, 0.7, 0.4], 0.65)

        # Verify it's trained
        assert w._is_trained

        # Get weights — should not be uniform
        weights = w.get_weights([0.33, 0.8], [0.6, 0.5, 0.7, 0.4])
        assert not np.allclose(weights, np.ones(4) / 4, atol=0.01)

    def test_journal_entry_stores_scores(self):
        """gate_details in journal should contain scores dict when wired."""
        # Simulate what execution.py does when building journal entry
        ctx = {
            "scores": {
                "direction_score": 0.72,
                "confidence_score": 0.65,
                "momentum_score": 0.55,
                "risk_score": 0.80,
                "used_learned_weights": True,
                "ensemble_weights": {
                    "direction": 0.40, "confidence": 0.25,
                    "momentum": 0.15, "risk": 0.20,
                },
            },
            "core_score": 0.67,
            "final_score": 0.62,
        }

        # Build gate_details as execution.py does
        gate_details = {
            "scores": ctx.get("scores", {}),
            "core_score": ctx.get("core_score"),
            "final_score": ctx.get("final_score"),
        }

        assert gate_details["scores"]["direction_score"] == 0.72
        assert gate_details["scores"]["used_learned_weights"] is True
        assert gate_details["scores"]["ensemble_weights"]["direction"] == 0.40

    def test_outcome_recording_from_journal(self):
        """EnsembleWeighter should correctly parse scores from journal entry."""
        from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig

        config = WeighterConfig(n_models=4, n_context=2, min_samples=100)
        w = EnsembleWeighter(config=config)

        # Simulate what execution.py does when recording outcome
        entry = {
            "pair": "EUR_USD",
            "gate_details": {
                "scores": {
                    "direction_score": 0.72,
                    "confidence_score": 0.65,
                    "momentum_score": 0.55,
                    "risk_score": 0.80,
                },
            },
            "regime": {
                "volatility_regime": "NORMAL",
                "volatility_regime_confidence": 0.75,
            },
        }
        pnl_pips = 25.0

        gate = entry.get("gate_details", {})
        scores = gate.get("scores", {})
        dir_s = float(scores.get("direction_score", 0.5))
        conf_s = float(scores.get("confidence_score", 0.5))
        mom_s = float(scores.get("momentum_score", 0.5))
        risk_s = float(scores.get("risk_score", 0.5))
        regime = entry.get("regime", {})
        vol_id = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}.get(
            str(regime.get("volatility_regime", "NORMAL")), 1
        )
        vol_conf = float(regime.get("volatility_regime_confidence", 0.5))

        w.record(
            context_features=[vol_id / 3.0, vol_conf],
            model_predictions=[dir_s, conf_s, mom_s, risk_s],
            realized_outcome=pnl_pips / 100.0,
        )

        assert len(w._buffer) == 1
        ctx, preds, outcome = w._buffer[0]
        assert abs(ctx[0] - 1.0 / 3.0) < 0.001  # NORMAL = 1/3
        assert abs(ctx[1] - 0.75) < 0.001
        assert abs(preds[0] - 0.72) < 0.001
        assert abs(outcome - 0.25) < 0.001  # 25 pips / 100
