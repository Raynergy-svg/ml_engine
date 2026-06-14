"""Tests for counterfactual evaluator (US-014)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.causal.counterfactual_evaluator import CounterfactualConfig, CounterfactualEvaluator


class TestCounterfactualEvaluator:
    def test_init(self):
        ce = CounterfactualEvaluator()
        assert ce.cfg.n_bootstrap == 100

    def test_evaluate(self):
        ce = CounterfactualEvaluator(CounterfactualConfig(n_bootstrap=50))
        market_state = np.array([1.0, 2.0, 3.0])
        causal_graph = np.zeros((3, 3))
        actual = {"position_size": 1.0, "direction": "LONG"}
        alternatives = [
            {"position_size": 0.5, "direction": "LONG", "name": "half_size"},
            {"position_size": 1.0, "direction": "SHORT", "name": "reverse"},
        ]
        result = ce.evaluate(market_state, causal_graph, actual, alternatives)
        assert "actual_pnl" in result
        assert "counterfactuals" in result
        assert len(result["counterfactuals"]) == 2
        assert "recommendation" in result

    def test_evaluate_with_returns(self):
        ce = CounterfactualEvaluator()
        market_state = np.ones(4)
        actual = {"position_size": 1.0, "direction": "LONG"}
        alts = [{"position_size": 2.0, "direction": "LONG"}]
        returns = np.random.randn(100) * 0.01
        result = ce.evaluate(market_state, np.zeros((4, 4)), actual, alts, factor_returns=returns)
        assert result["actual_pnl"]["mean"] != 0.0
