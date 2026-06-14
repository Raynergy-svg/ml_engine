"""Tests for causal discovery (US-013)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.causal.causal_discovery import CausalDiscovery, CausalDiscoveryConfig


class TestCausalDiscovery:
    def test_init(self):
        cd = CausalDiscovery()
        assert cd.cfg.max_factors == 20

    def test_fit_requires_2d(self):
        cd = CausalDiscovery()
        with pytest.raises(ValueError, match="2D"):
            cd.fit(np.ones(10))

    def test_fit_too_many_factors(self):
        cd = CausalDiscovery(CausalDiscoveryConfig(max_factors=5))
        with pytest.raises(ValueError, match="Too many factors"):
            cd.fit(np.random.randn(100, 10))

    def test_fit_and_get_adjacency(self):
        cd = CausalDiscovery(CausalDiscoveryConfig(max_factors=5, alpha=0.1))
        # X1 causes X2, X2 causes X3
        n = 500
        x1 = np.random.randn(n)
        x2 = 0.5 * x1 + np.random.randn(n)
        x3 = 0.3 * x2 + np.random.randn(n)
        x4 = np.random.randn(n)
        x5 = 0.2 * x3 + np.random.randn(n)
        X = np.stack([x1, x2, x3, x4, x5], axis=1)
        cd.fit(X, factor_names=["X1", "X2", "X3", "X4", "X5"])
        adj = cd.get_adjacency()
        assert adj.shape == (5, 5)
        assert np.diag(adj).sum() == 0  # no self-loops

    def test_get_edges(self):
        cd = CausalDiscovery(CausalDiscoveryConfig(max_factors=4, alpha=0.1))
        X = np.random.randn(100, 4)
        cd.fit(X)
        edges = cd.get_edges()
        assert isinstance(edges, list)

    def test_get_confidences(self):
        cd = CausalDiscovery(CausalDiscoveryConfig(max_factors=3))
        cd.fit(np.random.randn(200, 3))
        conf = cd.get_confidences()
        assert conf.shape == (3, 3)
        assert np.all(conf >= 0)
