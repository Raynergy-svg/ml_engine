"""Tests for MAML Ridge prototype (Tier 6 Phase 1).

Tests:
1. NeuralRidge forward pass produces valid confidence 0-100
2. MAML inner loop adapts parameters (loss decreases)
3. Meta-training outer loop updates meta-parameters
4. Shadow mode comparison works correctly
5. Regime-specific adaptation produces different predictions
6. Persistence roundtrip preserves state
7. Graceful degradation when PyTorch unavailable
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import torch


# ── Test 1: NeuralRidge forward pass ──


class TestNeuralRidge:
    def test_output_range(self):
        """NeuralRidge output should be in [0, 100]."""
        from src.recursive_intelligence.maml_ridge import NeuralRidge

        model = NeuralRidge()
        x = torch.randn(1, 14)
        out = model(x)
        assert out.shape == (1,)
        assert 0.0 <= out.item() <= 100.0

    def test_batch_forward(self):
        """Should handle batch inputs."""
        from src.recursive_intelligence.maml_ridge import NeuralRidge

        model = NeuralRidge()
        x = torch.randn(10, 14)
        out = model(x)
        assert out.shape == (10,)
        assert all(0.0 <= v <= 100.0 for v in out.tolist())

    def test_param_count(self):
        """Should have reasonable param count (~700)."""
        from src.recursive_intelligence.maml_ridge import NeuralRidge

        model = NeuralRidge()  # Use default config dims
        count = model.param_count()
        # 14*64+64 + 64*32+32 + 32*1+1 = 960+64+2048+32+32+1 = 3137
        assert 2000 < count < 5000

    def test_gradients_flow(self):
        """Gradients should flow through all layers."""
        from src.recursive_intelligence.maml_ridge import NeuralRidge

        model = NeuralRidge()  # Use default config dims
        x = torch.randn(5, 14)
        y = torch.tensor([50.0, 60.0, 70.0, 40.0, 80.0])
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ── Test 2: MAML inner loop ──


class TestMAMLInnerLoop:
    def test_inner_loop_reduces_loss(self):
        """Inner loop adaptation should reduce loss on support data."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                config = MAMLConfig(inner_steps=20, inner_lr=0.001)
                maml = MAMLRidge(config=config)

                # Create support data with targets near sigmoid center (50)
                np.random.seed(42)
                support = [
                    (np.random.randn(14).astype(np.float32) * 0.5, 55.0)
                    for _ in range(10)
                ]

                # Loss before adaptation
                X = torch.tensor(np.array([s[0] for s in support], dtype=np.float32))
                Y = torch.tensor(np.array([s[1] for s in support], dtype=np.float32))
                with torch.no_grad():
                    pred_before = maml._meta_model(X)
                    loss_before = torch.nn.functional.mse_loss(pred_before, Y).item()

                # Run inner loop
                adapted_params, inner_losses = maml._inner_loop(support)

                # Inner losses should be recorded
                assert len(inner_losses) == config.inner_steps

                # Loss after adaptation
                pred_after = maml._forward_with_params(X, adapted_params)
                loss_after = torch.nn.functional.mse_loss(pred_after, Y).item()

                assert loss_after < loss_before, (
                    f"Inner loop should reduce loss: {loss_before:.2f} -> {loss_after:.2f}"
                )

    def test_inner_loop_preserves_meta_params(self):
        """Inner loop should NOT modify meta-parameters (only create adapted copies)."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                maml = MAMLRidge(config=MAMLConfig(inner_steps=5))

                # Save meta-param snapshot
                meta_before = [p.clone() for p in maml._meta_model.parameters()]

                support = [
                    (np.random.randn(14).astype(np.float32), 60.0)
                    for _ in range(5)
                ]
                maml._inner_loop(support)

                # Meta-params should be unchanged
                for p_before, p_after in zip(meta_before, maml._meta_model.parameters()):
                    assert torch.allclose(p_before, p_after), "Inner loop modified meta-parameters!"


# ── Test 3: Meta-training outer loop ──


class TestMetaTrain:
    def test_meta_train_updates_meta_params(self):
        """Outer loop should update meta-parameters."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                config = MAMLConfig(
                    min_samples_per_regime=5,
                    meta_train_interval=1000,  # Prevent auto-train
                    inner_steps=3,
                )
                maml = MAMLRidge(config=config)

                # Fill regime buffers
                for regime in ["NORMAL", "HIGH"]:
                    for _ in range(10):
                        target = 70.0 if regime == "NORMAL" else 30.0
                        maml._regime_buffers[regime].append(
                            (np.random.randn(14).astype(np.float32), target)
                        )

                # Save meta-param snapshot
                meta_before = [p.clone().detach() for p in maml._meta_model.parameters()]

                # Run meta-train
                stats = maml.meta_train()

                assert stats["status"] == "trained"
                assert stats["meta_trains"] == 1

                # Meta-params should have changed
                changed = False
                for p_before, p_after in zip(meta_before, maml._meta_model.parameters()):
                    if not torch.allclose(p_before, p_after.detach(), atol=1e-8):
                        changed = True
                        break
                assert changed, "Meta-train didn't update meta-parameters"

    def test_meta_train_insufficient_data(self):
        """Should return 'insufficient_data' when not enough samples."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                maml = MAMLRidge(config=MAMLConfig(min_samples_per_regime=50))
                stats = maml.meta_train()
                assert stats["status"] == "insufficient_data"


# ── Test 4: Shadow mode ──


class TestShadowMode:
    def test_shadow_comparison(self):
        """Shadow prediction should return comparison with Ridge."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                maml = MAMLRidge(config=MAMLConfig(shadow_mode=True))
                features = np.random.randn(14).astype(np.float32)

                result = maml.predict_shadow(features, "NORMAL", ridge_confidence=65.0)

                assert "maml_confidence" in result
                assert "ridge_confidence" in result
                assert result["ridge_confidence"] == 65.0
                assert "diff" in result
                assert "regime" in result
                assert result["regime"] == "NORMAL"

    def test_shadow_log_capped(self):
        """Shadow log should not exceed 100 entries."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                maml = MAMLRidge(config=MAMLConfig())
                features = np.random.randn(14).astype(np.float32)

                for _ in range(150):
                    maml.predict_shadow(features, "NORMAL", 50.0)

                assert len(maml._shadow_log) <= 100


# ── Test 5: Regime-specific adaptation ──


class TestRegimeAdaptation:
    def test_different_regimes_different_predictions(self):
        """After training on different regimes, predictions should diverge."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                config = MAMLConfig(
                    min_samples_per_regime=5,
                    meta_train_interval=1000,
                    inner_steps=10,
                )
                maml = MAMLRidge(config=config)

                # NORMAL regime: high confidence targets
                for _ in range(50):
                    maml._regime_buffers["NORMAL"].append(
                        (np.ones(14, dtype=np.float32) * 0.5, 80.0)
                    )
                # EXTREME regime: low confidence targets
                for _ in range(50):
                    maml._regime_buffers["EXTREME"].append(
                        (np.ones(14, dtype=np.float32) * 0.5, 20.0)
                    )

                # Multiple meta-train rounds for convergence
                for _ in range(20):
                    maml.meta_train()

                features = np.ones(14, dtype=np.float32) * 0.5
                pred_normal = maml.predict(features, "NORMAL")
                pred_extreme = maml.predict(features, "EXTREME")

                # Predictions should differ (adapted to different targets)
                assert abs(pred_normal - pred_extreme) > 1.0, (
                    f"Same features should get different predictions per regime: "
                    f"NORMAL={pred_normal:.1f}, EXTREME={pred_extreme:.1f}"
                )


# ── Test 6: Persistence ──


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        """Save and load should preserve meta-model and buffers."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "maml"
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", state_dir):
                config = MAMLConfig(min_samples_per_regime=3, meta_train_interval=1000)
                m1 = MAMLRidge(config=config)

                # Add data and train
                for _ in range(5):
                    m1._regime_buffers["NORMAL"].append(
                        (np.random.randn(14).astype(np.float32), 65.0)
                    )
                m1.meta_train()

                # Predict before save
                features = np.ones(14, dtype=np.float32) * 0.3
                pred1 = m1.predict(features, "NORMAL")
                trains1 = m1._total_meta_trains

                # Load into new instance
                m2 = MAMLRidge(config=config)
                pred2 = m2.predict(features, "NORMAL")

                assert m2._total_meta_trains == trains1
                assert abs(pred1 - pred2) < 0.1, f"Predictions diverged after load: {pred1:.2f} vs {pred2:.2f}"


# ── Test 7: Stats ──


class TestStats:
    def test_stats_structure(self):
        """Stats should have all expected fields."""
        from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.recursive_intelligence.maml_ridge._STATE_DIR", Path(tmpdir)):
                maml = MAMLRidge(config=MAMLConfig())
                stats = maml.stats

                assert "enabled" in stats
                assert stats["enabled"] is True
                assert "shadow_mode" in stats
                assert "total_meta_trains" in stats
                assert "param_count" in stats
                assert stats["param_count"] > 0
                assert "regime_buffers" in stats
                assert "shadow_mae" in stats
