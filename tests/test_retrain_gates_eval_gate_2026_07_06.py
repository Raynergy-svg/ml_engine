"""P0 (2026-07-06): eval gate for `cli.training_ops.retrain_gates()`.

docs/ENGINEERING_BRAIN.md flagged `retrain_gates()` (invoked via `main.py
retrain-gates`, and as the subprocess fallback in
`src/core/modular_inference.py._incremental_retrain` when the ALREADY-gated
`online_retrainer.OnlineRetrainer` is unavailable) as the one ungated
model-write path: it fetched fresh OANDA data, trained XGBoost/RF/Ridge, and
called `.save()` unconditionally — no comparison against the existing on-disk
model.

These tests exercise the new `_apply_*_gate` functions directly with REAL
trainer classes and REAL disk (tmp_path), no mocks, no network — per the
project's no-mock policy (OANDA fetch is orthogonal to the gate logic and is
correctly out of scope: skip/integration-mark, don't mock). Reproduce-then-
resolve: each REFUSED case demonstrates the exact defect (a worse candidate
would previously have silently overwritten the incumbent); each PASSED case
demonstrates the promotion path still works when warranted.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from cli.training_ops import (
    CLI_GATE_PASSED,
    CLI_GATE_REFUSED,
    _apply_ridge_gate,
    _apply_rf_gate,
    _apply_xgb_gate,
    _cli_gate_verdict,
)
from src.training.trainers.config import TrainerConfig
from src.training.trainers.random_forest_trainer import RandomForestTrainer
from src.training.trainers.ridge_trainer import RidgeTrainer
from src.training.trainers.xgboost_trainer import XGBoostTrainer


# --------------------------------------------------------------------------
# Synthetic data helpers — clean, learnable signal vs. pure noise. No mocks:
# real numpy arrays fed straight into the real trainer classes.
# --------------------------------------------------------------------------

def _clean_xgb_data(rng: np.random.Generator, n: int, n_features: int = 8):
    X = rng.normal(size=(n, n_features))
    momentum = 1.0 / (1.0 + np.exp(-X[:, 0] - 0.5 * X[:, 1]))  # learnable, in (0, 1)
    momentum += rng.normal(scale=0.01, size=n)
    momentum = np.clip(momentum, 0.0, 1.0)
    accel = (X[:, 0] + X[:, 1] > 0).astype(int)
    y = np.column_stack([momentum, accel])
    return X, y


def _noisy_xgb_data(rng: np.random.Generator, n: int, n_features: int = 8):
    X = rng.normal(size=(n, n_features))
    momentum = rng.uniform(0.0, 1.0, size=n)  # unrelated to X — unlearnable
    accel = rng.integers(0, 2, size=n)
    y = np.column_stack([momentum, accel])
    return X, y


def _clean_rf_data(rng: np.random.Generator, n: int, n_features: int = 8):
    X = rng.normal(size=(n, n_features))
    drawdown = 20.0 + 5.0 * X[:, 0] + rng.normal(scale=0.5, size=n)
    streak = np.clip(0.5 + 0.1 * X[:, 1] + rng.normal(scale=0.02, size=n), 0.0, 1.0)
    y = np.column_stack([drawdown, streak])
    return X, y


def _noisy_rf_data(rng: np.random.Generator, n: int, n_features: int = 8):
    X = rng.normal(size=(n, n_features))
    drawdown = rng.uniform(0.0, 100.0, size=n)
    streak = rng.uniform(0.0, 1.0, size=n)
    y = np.column_stack([drawdown, streak])
    return X, y


def _clean_ridge_data(rng: np.random.Generator, n: int, n_features: int = 8):
    X = rng.normal(size=(n, n_features))
    conf = np.clip(60.0 + 15.0 * X[:, 0] + rng.normal(scale=1.0, size=n), 0.0, 100.0)
    return X, conf


def _noisy_ridge_data(rng: np.random.Generator, n: int, n_features: int = 8):
    X = rng.normal(size=(n, n_features))
    conf = rng.uniform(0.0, 100.0, size=n)
    return X, conf


FEATURE_NAMES = [f"f{i}" for i in range(8)]


# --------------------------------------------------------------------------
# _cli_gate_verdict — pure decision logic
# --------------------------------------------------------------------------

def test_verdict_refuses_worse_candidate():
    gate = _cli_gate_verdict(candidate_mae=1.0, existing_mae=0.5, n_test=50)
    assert gate["verdict"] == CLI_GATE_REFUSED
    assert "candidate_worse_than_existing" in gate["reason"]


def test_verdict_passes_better_candidate():
    gate = _cli_gate_verdict(candidate_mae=0.4, existing_mae=0.5, n_test=50)
    assert gate["verdict"] == CLI_GATE_PASSED


def test_verdict_passes_within_tolerance():
    # 5% tolerance: 0.524 is within 0.5 * 1.05 = 0.525
    gate = _cli_gate_verdict(candidate_mae=0.524, existing_mae=0.5, n_test=50)
    assert gate["verdict"] == CLI_GATE_PASSED


def test_verdict_no_existing_model_passes_bootstrap():
    gate = _cli_gate_verdict(candidate_mae=999.0, existing_mae=None, n_test=50)
    assert gate["verdict"] == CLI_GATE_PASSED
    assert gate["reason"] == "no_scorable_existing_model_candidate_sane"


def test_verdict_refuses_tiny_holdout():
    gate = _cli_gate_verdict(candidate_mae=0.1, existing_mae=0.5, n_test=5)
    assert gate["verdict"] == CLI_GATE_REFUSED
    assert "test_holdout_too_small" in gate["reason"]


def test_verdict_refuses_nan_candidate():
    gate = _cli_gate_verdict(candidate_mae=float("nan"), existing_mae=0.5, n_test=50)
    assert gate["verdict"] == CLI_GATE_REFUSED
    assert gate["reason"] == "candidate_mae_nan"


# --------------------------------------------------------------------------
# XGBoost gate — real trainer, real disk
# --------------------------------------------------------------------------

def test_xgb_gate_refuses_worse_candidate_incumbent_untouched(tmp_path: Path):
    rng = np.random.default_rng(42)
    config = TrainerConfig(xgb_n_estimators=30, xgb_max_depth=3)
    model_path = tmp_path / "xgb_momentum.pkl"

    # Ship a GOOD incumbent trained on a clean, learnable signal.
    X_tr, y_tr = _clean_xgb_data(rng, 400)
    X_val, y_val = _clean_xgb_data(rng, 80)
    good = XGBoostTrainer(config)
    good.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)
    good.save(str(model_path))
    incumbent_bytes_before = model_path.read_bytes()

    # Fresh BAD candidate: trained on pure noise, will score far worse.
    X_tr2, y_tr2 = _noisy_xgb_data(rng, 60)
    X_val2, y_val2 = _noisy_xgb_data(rng, 20)
    bad = XGBoostTrainer(config)
    bad.train(X_tr2, y_tr2, X_val2, y_val2, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_xgb_data(rng, 50)  # untouched gate holdout
    gate = _apply_xgb_gate(bad, model_path, config, X_test, y_test)

    assert gate["verdict"] == CLI_GATE_REFUSED
    assert gate["existing_mae"] is not None
    assert gate["candidate_mae"] > gate["existing_mae"]

    # Mirror retrain_gates()'s call-site contract: only save() on PASSED.
    if gate["verdict"] == CLI_GATE_PASSED:
        bad.save(str(model_path))
    assert model_path.read_bytes() == incumbent_bytes_before, (
        "REFUSED candidate must never touch the incumbent .pkl bytes"
    )


def test_xgb_gate_passes_better_candidate_and_write_replaces_incumbent(tmp_path: Path):
    rng = np.random.default_rng(7)
    config = TrainerConfig(xgb_n_estimators=30, xgb_max_depth=3)
    model_path = tmp_path / "xgb_momentum.pkl"

    # Ship a WEAK incumbent trained on noise.
    X_tr, y_tr = _noisy_xgb_data(rng, 60)
    X_val, y_val = _noisy_xgb_data(rng, 20)
    weak = XGBoostTrainer(config)
    weak.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)
    weak.save(str(model_path))
    incumbent_bytes_before = model_path.read_bytes()

    # Fresh GOOD candidate trained on a clean, learnable signal.
    X_tr2, y_tr2 = _clean_xgb_data(rng, 400)
    X_val2, y_val2 = _clean_xgb_data(rng, 80)
    good = XGBoostTrainer(config)
    good.train(X_tr2, y_tr2, X_val2, y_val2, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_xgb_data(rng, 50)
    gate = _apply_xgb_gate(good, model_path, config, X_test, y_test)

    assert gate["verdict"] == CLI_GATE_PASSED
    if gate["verdict"] == CLI_GATE_PASSED:
        good.save(str(model_path))
    assert model_path.read_bytes() != incumbent_bytes_before, (
        "PASSED candidate must actually replace the incumbent .pkl"
    )


def test_xgb_gate_bootstraps_when_no_incumbent(tmp_path: Path):
    rng = np.random.default_rng(1)
    config = TrainerConfig(xgb_n_estimators=30, xgb_max_depth=3)
    model_path = tmp_path / "xgb_momentum.pkl"  # never written

    X_tr, y_tr = _noisy_xgb_data(rng, 60)
    X_val, y_val = _noisy_xgb_data(rng, 20)
    candidate = XGBoostTrainer(config)
    candidate.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_xgb_data(rng, 50)
    gate = _apply_xgb_gate(candidate, model_path, config, X_test, y_test)

    assert gate["verdict"] == CLI_GATE_PASSED
    assert gate["existing_mae"] is None
    assert gate["reason"] == "no_scorable_existing_model_candidate_sane"


# --------------------------------------------------------------------------
# Random Forest gate — two sub-targets, refuse if EITHER regresses
# --------------------------------------------------------------------------

def test_rf_gate_refuses_worse_candidate_incumbent_untouched(tmp_path: Path):
    rng = np.random.default_rng(3)
    config = TrainerConfig(rf_n_estimators=30, rf_max_depth=4)
    model_path = tmp_path / "rf_risk.pkl"

    X_tr, y_tr = _clean_rf_data(rng, 400)
    X_val, y_val = _clean_rf_data(rng, 80)
    good = RandomForestTrainer(config)
    good.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)
    good.save(str(model_path))
    incumbent_bytes_before = model_path.read_bytes()

    X_tr2, y_tr2 = _noisy_rf_data(rng, 60)
    X_val2, y_val2 = _noisy_rf_data(rng, 20)
    bad = RandomForestTrainer(config)
    bad.train(X_tr2, y_tr2, X_val2, y_val2, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_rf_data(rng, 50)
    gate = _apply_rf_gate(bad, model_path, config, X_test, y_test)

    assert gate["verdict"] == CLI_GATE_REFUSED
    if gate["verdict"] == CLI_GATE_PASSED:
        bad.save(str(model_path))
    assert model_path.read_bytes() == incumbent_bytes_before


def test_rf_gate_passes_better_candidate(tmp_path: Path):
    rng = np.random.default_rng(11)
    config = TrainerConfig(rf_n_estimators=30, rf_max_depth=4)
    model_path = tmp_path / "rf_risk.pkl"

    X_tr, y_tr = _noisy_rf_data(rng, 60)
    X_val, y_val = _noisy_rf_data(rng, 20)
    weak = RandomForestTrainer(config)
    weak.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)
    weak.save(str(model_path))
    incumbent_bytes_before = model_path.read_bytes()

    X_tr2, y_tr2 = _clean_rf_data(rng, 400)
    X_val2, y_val2 = _clean_rf_data(rng, 80)
    good = RandomForestTrainer(config)
    good.train(X_tr2, y_tr2, X_val2, y_val2, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_rf_data(rng, 50)
    gate = _apply_rf_gate(good, model_path, config, X_test, y_test)

    assert gate["verdict"] == CLI_GATE_PASSED
    good.save(str(model_path))
    assert model_path.read_bytes() != incumbent_bytes_before


# --------------------------------------------------------------------------
# Ridge/ElasticNet gate
# --------------------------------------------------------------------------

def test_ridge_gate_refuses_worse_candidate_incumbent_untouched(tmp_path: Path):
    rng = np.random.default_rng(5)
    config = TrainerConfig()
    model_path = tmp_path / "ridge_confidence.pkl"

    X_tr, y_tr = _clean_ridge_data(rng, 400)
    X_val, y_val = _clean_ridge_data(rng, 80)
    good = RidgeTrainer(config)
    good.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)
    good.save(str(model_path))
    incumbent_bytes_before = model_path.read_bytes()

    X_tr2, y_tr2 = _noisy_ridge_data(rng, 60)
    X_val2, y_val2 = _noisy_ridge_data(rng, 20)
    bad = RidgeTrainer(config)
    bad.train(X_tr2, y_tr2, X_val2, y_val2, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_ridge_data(rng, 50)
    gate = _apply_ridge_gate(bad, model_path, config, X_test, y_test)

    assert gate["verdict"] == CLI_GATE_REFUSED
    if gate["verdict"] == CLI_GATE_PASSED:
        bad.save(str(model_path))
    assert model_path.read_bytes() == incumbent_bytes_before


def test_ridge_gate_passes_better_candidate(tmp_path: Path):
    rng = np.random.default_rng(13)
    config = TrainerConfig()
    model_path = tmp_path / "ridge_confidence.pkl"

    X_tr, y_tr = _noisy_ridge_data(rng, 60)
    X_val, y_val = _noisy_ridge_data(rng, 20)
    weak = RidgeTrainer(config)
    weak.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)
    weak.save(str(model_path))
    incumbent_bytes_before = model_path.read_bytes()

    X_tr2, y_tr2 = _clean_ridge_data(rng, 400)
    X_val2, y_val2 = _clean_ridge_data(rng, 80)
    good = RidgeTrainer(config)
    good.train(X_tr2, y_tr2, X_val2, y_val2, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_ridge_data(rng, 50)
    gate = _apply_ridge_gate(good, model_path, config, X_test, y_test)

    assert gate["verdict"] == CLI_GATE_PASSED
    good.save(str(model_path))
    assert model_path.read_bytes() != incumbent_bytes_before


def test_ridge_gate_refuses_degenerate_candidate(tmp_path: Path):
    rng = np.random.default_rng(9)
    config = TrainerConfig()
    model_path = tmp_path / "ridge_confidence.pkl"  # no incumbent

    n = 60
    X_tr = rng.normal(size=(n, 8))
    y_tr = np.full(n, 50.0)  # zero-variance target -> degenerate constant model
    X_val = rng.normal(size=(20, 8))
    y_val = np.full(20, 50.0)
    degenerate = RidgeTrainer(config)
    degenerate.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_ridge_data(rng, 50)
    gate = _apply_ridge_gate(degenerate, model_path, config, X_test, y_test)

    assert gate["verdict"] == CLI_GATE_REFUSED
    assert gate["reason"] == "degenerate_candidate_constant_predictions"


# --------------------------------------------------------------------------
# Cross-model schema isolation: an incumbent saved by online_retrainer.py's
# OWN ad-hoc RF schema ('model' key, not 'drawdown_model'/'streak_model')
# must not crash the gate — unscorable incumbent falls back to bootstrap.
# --------------------------------------------------------------------------

def test_rf_gate_handles_incompatible_incumbent_schema(tmp_path: Path):
    rng = np.random.default_rng(17)
    config = TrainerConfig(rf_n_estimators=30, rf_max_depth=4)
    model_path = tmp_path / "rf_risk.pkl"
    # Simulate online_retrainer.py's divergent save schema.
    with open(model_path, "wb") as f:
        pickle.dump({"model": object(), "scaler": None}, f)

    X_tr, y_tr = _clean_rf_data(rng, 200)
    X_val, y_val = _clean_rf_data(rng, 40)
    candidate = RandomForestTrainer(config)
    candidate.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)

    X_test, y_test = _clean_rf_data(rng, 50)
    gate = _apply_rf_gate(candidate, model_path, config, X_test, y_test)

    # Unscorable incumbent (wrong schema) -> treated as no-baseline bootstrap,
    # never a crash and never a silent false "worse" refusal.
    assert gate["verdict"] == CLI_GATE_PASSED


# --------------------------------------------------------------------------
# Adjacent bug found while building the gate: XGBoostTrainer.save()+.load()
# didn't round-trip under the pinned xgboost 3.2.0 (src/utils/
# xgboost_artifacts.py's atomic-write temp name was "<sidecar>.json.tmp" —
# an unrecognized extension at save time, so XGBoost silently wrote UBJSON
# bytes under a ".json" name; load_model() then failed to parse them). This
# is exactly what the gate needs working to ever score a real incumbent, so
# it's fixed in the same change. The pre-existing test_xgboost_native_
# artifacts.py suite didn't catch it because it mocks out the xgboost
# module entirely (fake save_model/load_model ignore format) — this test
# uses the REAL xgboost library, no mocks, to prove the actual round trip.
# --------------------------------------------------------------------------

def test_xgboost_trainer_real_save_load_round_trip(tmp_path: Path):
    rng = np.random.default_rng(21)
    config = TrainerConfig(xgb_n_estimators=10, xgb_max_depth=2)
    model_path = tmp_path / "xgb_momentum.pkl"

    X_tr, y_tr = _clean_xgb_data(rng, 60)
    X_val, y_val = _clean_xgb_data(rng, 20)
    trainer = XGBoostTrainer(config)
    trainer.train(X_tr, y_tr, X_val, y_val, feature_names=FEATURE_NAMES)
    trainer.save(str(model_path))

    reloaded = XGBoostTrainer(config)
    reloaded.load(str(model_path))  # must not raise

    X_test, _ = _clean_xgb_data(rng, 10)
    original_pred = trainer.momentum_model.predict(trainer.scaler.transform(X_test))
    reloaded_pred = reloaded.momentum_model.predict(reloaded.scaler.transform(X_test))
    assert np.allclose(original_pred, reloaded_pred), (
        "reloaded model must produce identical predictions to the saved one"
    )
