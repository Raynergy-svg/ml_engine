"""Smoke tests for model_comparison.py."""
import pytest
import numpy as np
import pandas as pd

from src.evaluation.model_comparison import (
    BarPrediction,
    ModelComparisonResult,
    compare_models_on_history,
    _direction_correct,
    _compute_accuracy,
    _compute_pnl,
)


def make_df(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    price = 1.1000 + np.cumsum(np.random.randn(n) * 0.0001)
    df = pd.DataFrame({
        "open": price + np.random.randn(n) * 0.0001,
        "high": price + abs(np.random.randn(n) * 0.0002),
        "low": price - abs(np.random.randn(n) * 0.0002),
        "close": price,
        "volume": np.random.randint(100, 1000, size=n),
    }, index=idx)
    return df


def dummy_legacy_fn(window: pd.DataFrame) -> BarPrediction:
    return BarPrediction(
        timestamp=str(window.index[-1]),
        direction="LONG" if window["close"].iloc[-1] > window["close"].iloc[-2] else "SHORT",
        confidence=0.6,
        score=0.6,
    )


def dummy_sota_fn(window: pd.DataFrame) -> BarPrediction:
    return BarPrediction(
        timestamp=str(window.index[-1]),
        direction="LONG",
        confidence=0.55,
        score=0.55,
    )


class TestModelComparison:
    def test_direction_correct(self):
        assert _direction_correct("LONG", 0.01) is True
        assert _direction_correct("SHORT", -0.01) is True
        assert _direction_correct("LONG", -0.01) is False
        assert _direction_correct("SHORT", 0.01) is False

    def test_compute_accuracy(self):
        preds = [
            BarPrediction(timestamp="1", direction="LONG", confidence=0.6, score=0.6),
            BarPrediction(timestamp="2", direction="SHORT", confidence=0.6, score=0.6),
            BarPrediction(timestamp="3", direction="LONG", confidence=0.6, score=0.6),
        ]
        actuals = [0.01, -0.01, 0.01]
        acc, long_acc, short_acc = _compute_accuracy(preds, actuals)
        assert acc == pytest.approx(1.0)
        assert long_acc == pytest.approx(1.0)
        assert short_acc == pytest.approx(1.0)

    def test_compute_pnl(self):
        preds = [
            BarPrediction(timestamp="1", direction="LONG", confidence=0.6, score=0.6),
            BarPrediction(timestamp="2", direction="SHORT", confidence=0.6, score=0.6),
        ]
        actuals = [0.01, -0.005]
        pnl, sharpe, max_dd = _compute_pnl(preds, actuals)
        assert pnl > 0
        assert isinstance(sharpe, float)
        assert max_dd >= 0

    def test_compare_models_on_history(self):
        df = make_df(n=300)
        result = compare_models_on_history(
            pair="EUR_USD",
            df=df,
            legacy_signal_fn=dummy_legacy_fn,
            sota_signal_fn=dummy_sota_fn,
            warmup_bars=50,
        )
        assert result.pair == "EUR_USD"
        assert result.n_bars == 300
        assert isinstance(result.legacy_accuracy, float)
        assert isinstance(result.sota_accuracy, float)
        assert isinstance(result.legacy_pnl, float)
        assert isinstance(result.sota_pnl, float)
        assert isinstance(result.error_correlation, float)

    def test_insufficient_data(self):
        df = make_df(n=50)
        result = compare_models_on_history(
            pair="EUR_USD",
            df=df,
            legacy_signal_fn=dummy_legacy_fn,
            sota_signal_fn=dummy_sota_fn,
            warmup_bars=128,
        )
        assert result.legacy_accuracy == 0.0
        assert result.sota_accuracy == 0.0
