"""Smoke tests for meta_ensemble.py."""
import pytest
import numpy as np

from src.evaluation.meta_ensemble import MetaEnsemble, compute_meta_weights
from src.evaluation.model_comparison import BarPrediction


class TestMetaEnsemble:
    def test_meta_ensemble_predict(self):
        me = MetaEnsemble(pair="EUR_USD", legacy_weight=0.5, sota_weight=0.5)
        leg = BarPrediction(timestamp="1", direction="LONG", confidence=0.6, score=0.6)
        sota = BarPrediction(timestamp="1", direction="LONG", confidence=0.7, score=0.7)
        pred = me.predict(leg, sota, regime="NORMAL")
        assert pred.direction in ("LONG", "SHORT", "HOLD")
        assert 0.0 <= pred.confidence <= 1.0

    def test_meta_ensemble_update(self):
        me = MetaEnsemble(pair="EUR_USD", legacy_weight=0.5, sota_weight=0.5)
        leg = BarPrediction(timestamp="1", direction="LONG", confidence=0.6, score=0.6)
        sota = BarPrediction(timestamp="1", direction="LONG", confidence=0.7, score=0.7)
        me.update_from_outcome(leg, sota, actual_ret=0.01, regime="NORMAL")
        assert len(me.weight_history) == 1

    def test_compute_meta_weights(self):
        np.random.seed(5)
        legacy_returns = np.random.randn(100) * 0.01
        sota_returns = np.random.randn(100) * 0.01
        w_leg, w_sota = compute_meta_weights(legacy_returns, sota_returns)
        assert abs(w_leg + w_sota - 1.0) < 1e-6
        assert w_leg >= 0.0
        assert w_sota >= 0.0
