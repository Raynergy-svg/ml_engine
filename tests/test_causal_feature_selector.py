"""Tests for causal feature selector (US-015)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.causal.causal_feature_selector import CausalFeatureSelector, CausalFeatureSelectorConfig


class TestCausalFeatureSelector:
    def test_init(self):
        cfs = CausalFeatureSelector()
        assert cfs.cfg.max_features == 10

    def test_select_with_causal_graph(self):
        cfs = CausalFeatureSelector(CausalFeatureSelectorConfig(target_index=0, max_features=2))
        X = np.random.randn(100, 5)
        graph = np.zeros((5, 5))
        graph[1, 0] = 1  # F1 -> F0
        graph[2, 0] = 1  # F2 -> F0
        conf = np.ones((5, 5)) * 0.5
        selected = cfs.select(X, graph, conf)
        assert selected == [1, 2]

    def test_fallback_to_correlation(self):
        cfs = CausalFeatureSelector(CausalFeatureSelectorConfig(target_index=0, max_features=2))
        X = np.random.randn(100, 5)
        # Make F1 highly correlated with F0
        X[:, 1] = X[:, 0] * 2 + np.random.randn(100) * 0.1
        selected = cfs.select(X)
        assert len(selected) <= 2
        assert 0 not in selected

    def test_no_fallback(self):
        cfs = CausalFeatureSelector(CausalFeatureSelectorConfig(fallback_to_correlation=False))
        X = np.random.randn(50, 3)
        selected = cfs.select(X)
        assert selected == []
