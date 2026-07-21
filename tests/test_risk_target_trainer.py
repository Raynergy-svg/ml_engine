"""Tests for `src/training/trainers/risk_target_trainer.py`.

NO MOCKS. Real LightGBM models, real disk via `tmp_path`.

Coverage:
    1. train() requires y_train_drawdown/y_val_drawdown kwargs.
    2. train()/predict() round trip produces plausible outputs (positive
       vol, probabilities in [0,1]).
    3. Vol predictions are ALWAYS positive (log-space fit + exponentiate)
       — regression test for the QLIKE-blowup bug found 2026-07-08 (an
       unconstrained raw-level fit produced non-positive predictions on
       ~7.6% of rows).
    4. save()/load() round trip preserves predictions and metrics.
    5. predict() before train()/load() raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.trainers.config import TrainerConfig
from src.training.trainers.risk_target_trainer import (
    RiskTargetTrainer,
    brier_score,
    pinball_loss,
    qlike_loss,
)


def _synthetic_dataset(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.uniform(0, 1, n),
    })
    # Vol target correlated with f1's magnitude (learnable signal), always positive.
    y_vol = np.abs(X["f1"].values) * 0.05 + 0.01 + rng.normal(0, 0.002, n).clip(min=-0.005)
    y_vol = np.clip(y_vol, 1e-4, None)
    y_dd = (X["f2"].values > np.quantile(X["f2"].values, 0.75)).astype(np.int64)
    return X, y_vol, y_dd


def _small_trainer_kwargs():
    # Small/fast hyperparameters for unit tests.
    return dict(n_estimators=30, max_depth=3, early_stopping_rounds=10)


def test_train_requires_drawdown_kwargs():
    X, y_vol, y_dd = _synthetic_dataset(200)
    trainer = RiskTargetTrainer(TrainerConfig())
    with pytest.raises(ValueError):
        trainer.train(X[:150], y_vol[:150], X[150:], y_vol[150:], feature_names=list(X.columns))


def test_train_predict_round_trip_plausible_outputs():
    X, y_vol, y_dd = _synthetic_dataset(1000)
    split = 700
    trainer = RiskTargetTrainer(TrainerConfig())
    metrics = trainer.train(
        X[:split], y_vol[:split], X[split:], y_vol[split:],
        feature_names=list(X.columns),
        y_train_drawdown=y_dd[:split], y_val_drawdown=y_dd[split:],
        **_small_trainer_kwargs(),
    )
    assert "val_qlike" in metrics
    assert "val_auc_drawdown" in metrics

    pred = trainer.predict(X[split:])
    assert np.all(pred["predicted_forward_volatility"] > 0)
    assert np.all(pred["predicted_drawdown_stressed_prob"] >= 0.0)
    assert np.all(pred["predicted_drawdown_stressed_prob"] <= 1.0)


def test_vol_predictions_always_positive_even_with_low_targets():
    """Regression test for the 2026-07-08 QLIKE-blowup bug: an
    unconstrained raw-level regressor produced non-positive predictions on
    low-vol rows. The log-space fit must never produce a non-positive
    prediction, however low the true target."""
    n = 2000
    rng = np.random.default_rng(99)
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    # Many rows with a VERY low true vol target (the failure-inducing regime).
    y_vol = np.where(rng.uniform(0, 1, n) < 0.5, 1e-4, 0.05) + rng.normal(0, 1e-5, n).clip(min=0)
    y_vol = np.clip(y_vol, 1e-6, None)
    y_dd = (rng.uniform(0, 1, n) < 0.25).astype(np.int64)

    split = 1500
    trainer = RiskTargetTrainer(TrainerConfig())
    trainer.train(
        X[:split], y_vol[:split], X[split:], y_vol[split:],
        feature_names=list(X.columns),
        y_train_drawdown=y_dd[:split], y_val_drawdown=y_dd[split:],
        **_small_trainer_kwargs(),
    )
    pred = trainer.predict(X[split:])["predicted_forward_volatility"]
    assert np.all(pred > 0), "vol prediction must always be strictly positive (log-space fit)"
    # QLIKE must be finite and not pathologically large on well-posed data.
    q = qlike_loss(y_vol[split:], pred)
    assert np.isfinite(q)


def test_save_load_round_trip(tmp_path):
    X, y_vol, y_dd = _synthetic_dataset(800)
    split = 600
    trainer = RiskTargetTrainer(TrainerConfig())
    trainer.train(
        X[:split], y_vol[:split], X[split:], y_vol[split:],
        feature_names=list(X.columns),
        y_train_drawdown=y_dd[:split], y_val_drawdown=y_dd[split:],
        **_small_trainer_kwargs(),
    )
    pred_before = trainer.predict(X[split:])

    model_path = tmp_path / "risk_target_model"
    trainer.save(str(model_path))
    assert (tmp_path / "risk_target_model.pkl").exists()
    assert (tmp_path / "risk_target_model.meta.json").exists()

    loaded = RiskTargetTrainer(TrainerConfig())
    loaded.load(str(model_path))
    pred_after = loaded.predict(X[split:])

    assert np.allclose(pred_before["predicted_forward_volatility"], pred_after["predicted_forward_volatility"])
    assert np.allclose(pred_before["predicted_drawdown_stressed_prob"], pred_after["predicted_drawdown_stressed_prob"])
    assert loaded.feature_names == list(X.columns)
    assert loaded.metrics.get("val_qlike") == trainer.metrics.get("val_qlike")


def test_predict_before_train_raises():
    trainer = RiskTargetTrainer(TrainerConfig())
    with pytest.raises(RuntimeError):
        trainer.predict(pd.DataFrame({"f1": [0.0]}))


def test_metric_helpers_basic_sanity():
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred_perfect = y_true.copy()
    assert qlike_loss(y_true, y_pred_perfect) == pytest.approx(0.0, abs=1e-9)
    assert pinball_loss(y_true, y_pred_perfect, 0.5) == pytest.approx(0.0, abs=1e-9)
    assert brier_score(np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0])) == pytest.approx(0.0)
    assert brier_score(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.25)
