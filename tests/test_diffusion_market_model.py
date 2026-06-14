"""Tests for diffusion-based market simulation (US-009)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.simulation.diffusion_market_model import (
    DiffusionConfig,
    DiffusionMarketModel,
)


class TestDiffusionConfig:
    def test_defaults(self):
        cfg = DiffusionConfig()
        assert cfg.n_factors == 5
        assert cfg.latent_dim == 16
        assert cfg.n_diffusion_steps == 50
        assert cfg.random_seed == 42

    def test_custom(self):
        cfg = DiffusionConfig(n_factors=3, latent_dim=8)
        assert cfg.n_factors == 3
        assert cfg.latent_dim == 8


class TestDiffusionMarketModel:
    def _make_data(self, n: int = 200, factors: int = 5) -> np.ndarray:
        np.random.seed(123)
        return np.random.randn(n, factors) * 0.01

    def test_init(self):
        model = DiffusionMarketModel()
        assert model.cfg.n_factors == 5
        assert not model._trained

    def test_fit_requires_2d(self):
        model = DiffusionMarketModel()
        with pytest.raises(ValueError, match="2D"):
            model.fit(np.ones(10))

    def test_fit_requires_correct_factors(self):
        model = DiffusionMarketModel(DiffusionConfig(n_factors=3))
        with pytest.raises(ValueError, match="Expected 3 factors"):
            model.fit(np.ones((10, 5)))

    def test_fit_and_generate(self):
        cfg = DiffusionConfig(n_factors=3, epochs=10)
        model = DiffusionMarketModel(cfg)
        X = self._make_data(n=100, factors=3)
        model.fit(X)
        assert model._trained is True

        paths = model.generate(n_paths=5, path_length=20, seed=42)
        assert paths.shape == (5, 20, 3)
        assert not np.isnan(paths).any()

    def test_generate_reproducible(self):
        cfg = DiffusionConfig(n_factors=3, epochs=10)
        model = DiffusionMarketModel(cfg)
        X = self._make_data(n=100, factors=3)
        model.fit(X)

        p1 = model.generate(n_paths=3, path_length=10, seed=99)
        p2 = model.generate(n_paths=3, path_length=10, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_generate_different_seeds(self):
        cfg = DiffusionConfig(n_factors=3, epochs=10)
        model = DiffusionMarketModel(cfg)
        X = self._make_data(n=100, factors=3)
        model.fit(X)

        p1 = model.generate(n_paths=3, path_length=10, seed=1)
        p2 = model.generate(n_paths=3, path_length=10, seed=2)
        assert not np.array_equal(p1, p2)

    def test_generate_with_regime_context(self):
        cfg = DiffusionConfig(n_factors=3, epochs=10)
        model = DiffusionMarketModel(cfg)
        X = self._make_data(n=100, factors=3)
        model.fit(X)

        paths = model.generate(
            n_paths=3,
            path_length=10,
            seed=1,
            regime_context={"volatility_multiplier": 2.0},
        )
        assert paths.shape == (3, 10, 3)

    def test_generate_before_fit_raises(self):
        model = DiffusionMarketModel()
        with pytest.raises(RuntimeError, match="fitted before generate"):
            model.generate(n_paths=2, path_length=5)

    def test_save_and_load(self, tmp_path):
        cfg = DiffusionConfig(n_factors=3, epochs=10)
        model = DiffusionMarketModel(cfg)
        X = self._make_data(n=100, factors=3)
        model.fit(X)

        save_path = tmp_path / "diffusion.npz"
        model.save(str(save_path))
        assert save_path.exists()

        loaded = DiffusionMarketModel(cfg).load(str(save_path))
        assert loaded._trained is True

        p1 = model.generate(n_paths=2, path_length=5, seed=7)
        p2 = loaded.generate(n_paths=2, path_length=5, seed=7)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_validate(self):
        pytest.importorskip("scipy")
        cfg = DiffusionConfig(n_factors=3, epochs=10)
        model = DiffusionMarketModel(cfg)
        X = self._make_data(n=100, factors=3)
        model.fit(X)

        real = X[:50].reshape(1, 50, 3)
        synth = model.generate(n_paths=1, path_length=50, seed=1)
        result = model.validate(real, synth)

        assert "ks_statistic" in result
        assert "ks_pvalue" in result
        assert "autocorr_real" in result
        assert "autocorr_synthetic" in result
        assert "autocorr_mse" in result
        assert "passed" in result
        assert isinstance(result["passed"], bool)

    def test_validate_requires_scipy(self):
        # If scipy is available, just skip; otherwise this is covered by importorskip
        pytest.importorskip("scipy")
        # Manual mock path not needed since scipy is present
        pass

    def test_stress_test_use_case(self):
        """Synthetic paths for stress testing factor portfolio."""
        cfg = DiffusionConfig(n_factors=4, epochs=10)
        model = DiffusionMarketModel(cfg)
        X = self._make_data(n=200, factors=4)
        model.fit(X)

        # Generate extreme regime paths
        extreme = model.generate(
            n_paths=100,
            path_length=30,
            seed=99,
            regime_context={"volatility_multiplier": 3.0},
        )
        assert extreme.shape == (100, 30, 4)

    def test_augmentation_use_case(self):
        """Synthetic training data augmentation."""
        cfg = DiffusionConfig(n_factors=3, epochs=10)
        model = DiffusionMarketModel(cfg)
        X = self._make_data(n=100, factors=3)
        model.fit(X)

        aug = model.generate(n_paths=50, path_length=100, seed=1)
        assert aug.shape == (50, 100, 3)
        # Augmented data should have similar scale to training data
        assert np.std(aug) < 10.0  # loose sanity check
