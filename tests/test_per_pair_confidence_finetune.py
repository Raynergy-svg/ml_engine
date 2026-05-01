"""Tests for per-pair confidence fine-tune behavior (re-enabled 2026-05-01).

Covers:
- Fine-tuned RidgeTrainer produces a valid loadable model.
- Predict returns scores in the [0, 100] gate API contract.
- Leak-detection assertion (logger.error) fires when R² > expected_band[1]
  by training on a synthetic over-fit case where target == feature.
"""

from __future__ import annotations

import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.training.trainers.config import TrainerConfig  # noqa: E402
from src.training.trainers.ridge_trainer import RidgeTrainer  # noqa: E402


def _make_realistic_data(n_train: int = 800, n_val: int = 200, n_features: int = 24,
                        seed: int = 42):
    """Synthesize features + binary-rescaled labels (loss=20, win=95).

    The feature/label relationship is intentionally weak (~0.1 signal /
    0.9 noise) so val R² lands in the realistic [0.05, 0.30] band.
    """
    rng = np.random.default_rng(seed)
    X_train = rng.normal(0, 1, size=(n_train, n_features)).astype(np.float32)
    X_val = rng.normal(0, 1, size=(n_val, n_features)).astype(np.float32)
    # Hidden signal in feature 0 only; weak coupling
    sig_tr = (X_train[:, 0] > 0).astype(np.int32)
    sig_va = (X_val[:, 0] > 0).astype(np.int32)
    # 30% noise flips
    flip_tr = rng.random(n_train) < 0.3
    flip_va = rng.random(n_val) < 0.3
    y_tr_bin = np.where(flip_tr, 1 - sig_tr, sig_tr)
    y_va_bin = np.where(flip_va, 1 - sig_va, sig_va)
    y_train = (20.0 + 75.0 * y_tr_bin).astype(np.float32)
    y_val = (20.0 + 75.0 * y_va_bin).astype(np.float32)
    return X_train, y_train, X_val, y_val


def _make_leaky_data(n_train: int = 400, n_val: int = 100, n_features: int = 24,
                     seed: int = 7):
    """Synthesize a perfect leak: feature 0 == binary label, no noise.

    Used to verify the leak-detection ERROR fires.
    """
    rng = np.random.default_rng(seed)
    y_tr_bin = (rng.random(n_train) > 0.5).astype(np.float32)
    y_va_bin = (rng.random(n_val) > 0.5).astype(np.float32)
    X_train = rng.normal(0, 1, size=(n_train, n_features)).astype(np.float32)
    X_val = rng.normal(0, 1, size=(n_val, n_features)).astype(np.float32)
    # Plant the label as feature 0 — model should recover R² ≈ 1
    X_train[:, 0] = y_tr_bin
    X_val[:, 0] = y_va_bin
    y_train = (20.0 + 75.0 * y_tr_bin).astype(np.float32)
    y_val = (20.0 + 75.0 * y_va_bin).astype(np.float32)
    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_finetune_produces_loadable_model(tmp_path: Path):
    """Train a small per-pair model, save, reload, predict, verify schema."""
    X_tr, y_tr, X_va, y_va = _make_realistic_data()
    feature_names = [f"feature_{i}" for i in range(X_tr.shape[1])]

    trainer = RidgeTrainer(TrainerConfig())
    metrics = trainer.train(
        X_tr, y_tr, X_va, y_va,
        feature_names=feature_names,
        label_metadata={
            "label_mode": "journal_outcome_blend",
            "n_real": 40, "n_pseudo": 1000,
            "class_balance_real": 0.4, "class_balance_pseudo": 0.5,
        },
    )

    # Save + reload
    path = tmp_path / "ridge_confidence.pkl"
    trainer.save(str(path))
    assert path.exists()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    assert "model" in data
    assert data["n_features"] == X_tr.shape[1]
    assert data["feature_names"] == feature_names
    assert data["model_type"] in ("lightgbm", "elasticnet")
    # Metrics persisted
    assert "r2_score" in data["metrics"]
    assert "expected_r2_band" in data["metrics"]


def test_finetune_predict_returns_in_api_contract(tmp_path: Path):
    """predict() must return confidence in [0, 100] (clamped by RidgeTrainer)."""
    X_tr, y_tr, X_va, y_va = _make_realistic_data()
    feature_names = [f"f{i}" for i in range(X_tr.shape[1])]
    trainer = RidgeTrainer(TrainerConfig())
    trainer.train(X_tr, y_tr, X_va, y_va, feature_names=feature_names,
                  label_metadata={"label_mode": "journal_outcome_blend"})

    # Predict on the val set, row by row
    for i in range(min(20, X_va.shape[0])):
        out = trainer.predict(X_va[i:i + 1])
        c = out["confidence"]
        assert 0.0 <= c <= 100.0, f"score out of [0,100]: {c}"


def test_finetune_metrics_metadata_passthrough(tmp_path: Path):
    """label_metadata fields must propagate into trainer.metrics."""
    X_tr, y_tr, X_va, y_va = _make_realistic_data()
    feature_names = [f"f{i}" for i in range(X_tr.shape[1])]
    label_meta = {
        "label_mode": "journal_outcome_blend",
        "n_real": 47,
        "n_pseudo": 9234,
        "class_balance_real": 0.36,
        "class_balance_pseudo": 0.55,
        "leak_fix_version": "2026-04-30",
        "score_range": [20.0, 95.0],
        "score_formula": "score = 20 + 75 * binary_label",
    }
    trainer = RidgeTrainer(TrainerConfig())
    metrics = trainer.train(X_tr, y_tr, X_va, y_va, feature_names=feature_names,
                            label_metadata=label_meta)
    assert metrics["label_mode"] == "journal_outcome_blend"
    assert metrics["n_real_labels"] == 47
    assert metrics["n_pseudo_labels"] == 9234
    assert metrics["class_balance_real"] == 0.36
    assert metrics["class_balance_pseudo"] == 0.55
    assert metrics["expected_r2_band"] == [0.05, 0.30]


def test_leak_detection_fires_on_overfit(caplog):
    """Plant the label as a feature → R² ≈ 1 → ERROR log must fire."""
    X_tr, y_tr, X_va, y_va = _make_leaky_data()
    feature_names = [f"f{i}" for i in range(X_tr.shape[1])]
    trainer = RidgeTrainer(TrainerConfig())

    with caplog.at_level(logging.ERROR, logger="src.training.trainers.ridge_trainer"):
        metrics = trainer.train(X_tr, y_tr, X_va, y_va, feature_names=feature_names,
                                label_metadata={"label_mode": "journal_outcome_blend"})

    # R² should be very high on the over-fit case
    assert metrics["r2_score"] > 0.30, (
        f"Expected R² > 0.30 on leaky data; got {metrics['r2_score']}"
    )
    # The leak-detection ERROR log must have fired
    leak_messages = [r for r in caplog.records
                     if r.levelno == logging.ERROR
                     and "SUSPECTED LABEL LEAK" in r.message]
    assert leak_messages, (
        "Expected SUSPECTED LABEL LEAK error log on overfit; got no ERROR. "
        f"All records: {[(r.levelname, r.message[:80]) for r in caplog.records]}"
    )


def test_no_leak_log_on_realistic_data(caplog):
    """Realistic noisy data → R² in band → no leak ERROR."""
    X_tr, y_tr, X_va, y_va = _make_realistic_data()
    feature_names = [f"f{i}" for i in range(X_tr.shape[1])]
    trainer = RidgeTrainer(TrainerConfig())

    with caplog.at_level(logging.ERROR, logger="src.training.trainers.ridge_trainer"):
        metrics = trainer.train(X_tr, y_tr, X_va, y_va, feature_names=feature_names,
                                label_metadata={"label_mode": "journal_outcome_blend"})

    # Should land in the [0, 0.30] band roughly
    assert metrics["r2_score"] <= 0.40
    leak_messages = [r for r in caplog.records
                     if r.levelno == logging.ERROR
                     and "SUSPECTED LABEL LEAK" in r.message]
    if metrics["r2_score"] <= 0.30:
        assert not leak_messages, (
            f"Unexpected leak ERROR with R²={metrics['r2_score']:.4f} (in band)"
        )
