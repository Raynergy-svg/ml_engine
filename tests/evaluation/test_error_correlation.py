"""Smoke tests for error_correlation.py."""
import pytest
import numpy as np
import pandas as pd

from src.evaluation.error_correlation import (
    RegimeSlice,
    ErrorCorrelation,
    analyze_error_correlation,
    _make_recommendation,
)
from src.evaluation.model_comparison import ModelComparisonResult, BarPrediction


def make_result() -> ModelComparisonResult:
    n = 100
    np.random.seed(7)
    legacy_preds = [
        BarPrediction(timestamp=str(i), direction="LONG", confidence=0.6, score=0.6)
        for i in range(n)
    ]
    sota_preds = [
        BarPrediction(timestamp=str(i), direction="LONG" if i % 2 == 0 else "SHORT", confidence=0.6, score=0.6)
        for i in range(n)
    ]
    actuals = [0.01 if i % 3 != 0 else -0.01 for i in range(n)]
    return ModelComparisonResult(
        pair="EUR_USD",
        n_bars=n,
        legacy_predictions=legacy_preds,
        sota_predictions=sota_preds,
        actual_returns=actuals,
    )


class TestErrorCorrelation:
    def test_regime_slice(self):
        rs = RegimeSlice(regime="NORMAL", n_bars=100, legacy_accuracy=0.5, sota_accuracy=0.5)
        assert rs.legacy_accuracy == pytest.approx(0.5)
        assert rs.sota_accuracy == pytest.approx(0.5)

    def test_analyze_error_correlation(self):
        result = make_result()
        ec = analyze_error_correlation(result)
        assert ec.pair == "EUR_USD"
        assert isinstance(ec.error_correlation, float)
        assert -1.0 <= ec.error_correlation <= 1.0

    def test_make_recommendation(self):
        result = make_result()
        ec = analyze_error_correlation(result)
        rec, conf = _make_recommendation(ec, result)
        assert rec in ("STACK", "REPLACE", "DISCARD", "UNDECIDED")
        assert 0.0 <= conf <= 1.0
