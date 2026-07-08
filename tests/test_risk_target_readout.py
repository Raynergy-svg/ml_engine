"""Tests for `src/training/risk_target_readout.py` (read-only consumption
seam). NO MOCKS — real artifact trained on synthetic data via tmp_path,
real disk reads.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.training.risk_target_readout import predict_risk_state
from src.training.trainers.config import TrainerConfig
from src.training.trainers.risk_target_trainer import RiskTargetTrainer
from src.training.risk_target_features import (
    FEATURE_COLUMNS,
    compute_risk_target_features,
)


def _synthetic_ohlcv(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = 0.004 + 0.01 * (np.sin(np.linspace(0, 8 * np.pi, n)) ** 2)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol))
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": rng.integers(1000, 5000, n).astype(float),
    })


def _train_artifact(tmp_path):
    df = _synthetic_ohlcv(600)
    feat = compute_risk_target_features(df)
    feat["pair"] = pd.Categorical(["TEST_PAIR"] * len(feat))
    feat["day_of_week"] = feat["day_of_week"].astype("category")
    valid = feat[FEATURE_COLUMNS].notna().all(axis=1)
    feat = feat[valid].reset_index(drop=True)

    cols = list(FEATURE_COLUMNS) + ["pair"]
    n = len(feat)
    rng = np.random.default_rng(0)
    y_vol = np.clip(feat["realized_vol_20"].values * 1.1, 1e-4, None)
    y_dd = (rng.uniform(0, 1, n) < 0.25).astype(np.int64)
    split = int(n * 0.8)

    trainer = RiskTargetTrainer(TrainerConfig())
    trainer.train(
        feat[cols][:split], y_vol[:split], feat[cols][split:], y_vol[split:],
        feature_names=cols,
        y_train_drawdown=y_dd[:split], y_val_drawdown=y_dd[split:],
        n_estimators=20, max_depth=3, early_stopping_rounds=8,
    )
    model_path = tmp_path / "risk_target_model"
    trainer.save(str(model_path))
    return model_path, df


def test_predict_risk_state_happy_path(tmp_path):
    model_path, df = _train_artifact(tmp_path)
    out = predict_risk_state(df, "TEST_PAIR", model_path=model_path)
    assert out["available"] is True
    assert out["predicted_forward_volatility_annualized"] > 0
    assert 0.0 <= out["predicted_drawdown_stressed_prob"] <= 1.0
    # Honest-consumption flag: drawdown head failed its pre-registered
    # calibration bar — must be flagged untrustworthy until re-cleared.
    assert out["drawdown_head_trustworthy"] is False
    assert "oos_metrics" in out["model_meta"]


def test_missing_artifact_fails_closed(tmp_path):
    df = _synthetic_ohlcv(200)
    out = predict_risk_state(df, "TEST_PAIR", model_path=tmp_path / "nonexistent")
    assert out["available"] is False
    assert "no_artifact_at" in out["reason"]


def test_version_mismatch_fails_closed(tmp_path):
    model_path, df = _train_artifact(tmp_path)
    meta_path = model_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["risk_target_feature_pipeline_version"] = "1999-01-01-v0"
    meta_path.write_text(json.dumps(meta))

    out = predict_risk_state(df, "TEST_PAIR", model_path=model_path)
    assert out["available"] is False
    assert "feature_pipeline_version_mismatch" in out["reason"]


def test_insufficient_history_fails_closed(tmp_path):
    model_path, _ = _train_artifact(tmp_path)
    short_df = _synthetic_ohlcv(30)  # < 61-row warm-up
    out = predict_risk_state(short_df, "TEST_PAIR", model_path=model_path)
    assert out["available"] is False
    assert out["reason"] == "insufficient_trailing_history_for_features"


def test_no_directional_output_keys(tmp_path):
    """The output must contain NO directional trade-signal fields — risk
    estimate only, per the hard constraint."""
    model_path, df = _train_artifact(tmp_path)
    out = predict_risk_state(df, "TEST_PAIR", model_path=model_path)
    forbidden = {"direction", "side", "signal", "entry", "long", "short", "buy", "sell"}
    assert not (set(k.lower() for k in out) & forbidden)
