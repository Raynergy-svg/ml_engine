"""
Eval gate on the online retrainer — fail-closed refuse-to-write (2026-07-03).

Audit: docs/training-architecture-audit-2026-07-03.md §1.4 item 5 — the online
retrainer rewrote trained_data/models/{xgb_momentum,rf_risk,ridge_confidence}.pkl
in place with cooldowns but ZERO evaluation gate. These tests exercise the new
gate through the REAL code path: real ``OnlineRetrainer``, real sklearn models,
real pickle files on real disk (tmp_path via the class's ``model_dir`` /
``RetrainConfig.state_file`` seams). No mocks (repo hard rule).

Covered:
(a) better candidate passes and the file is written
(b) worse candidate refused and the ORIGINAL file bytes are untouched
(c) tiny holdout refused fail-closed
(d) missing existing model + sane candidate passes
(+) degenerate (constant-prediction) candidate refused without a baseline
"""

import logging

import numpy as np
import pytest

from online_retrainer import (
    GATE_HOLDOUT_FRACTION,
    GATE_MIN_HOLDOUT_SAMPLES,
    GATE_VERDICT_PASSED,
    GATE_VERDICT_REFUSED,
    OnlineRetrainer,
    RetrainConfig,
)

N_FEATURES = 8
_W = np.linspace(1.0, 2.0, N_FEATURES)


@pytest.fixture(autouse=True)
def _quiet_wandb(monkeypatch):
    """Keep the (real, offline-safe) W&B control plane inert in tests."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


def make_data(n=200, seed=7):
    """Learnable synthetic replay data: y = 1[X @ w > 0]."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    y = (X @ _W > 0).astype(int)
    return X, y


def make_retrainer(tmp_path, **config_overrides):
    """Real OnlineRetrainer with all state/artifacts redirected to tmp_path."""
    kwargs = {
        "state_file": str(tmp_path / "online_retrain_state.json"),
        "retrain_xgboost": False,
        "retrain_rf": False,
        "retrain_ridge": True,
    }
    kwargs.update(config_overrides)
    cfg = RetrainConfig(**kwargs)
    return OnlineRetrainer(config=cfg, model_dir=str(tmp_path / "models"))


def test_better_candidate_passes_and_file_is_written(tmp_path):
    """(a) A candidate that beats the on-disk model must be written."""
    retrainer = make_retrainer(tmp_path)
    model_path = retrainer.model_dir / "ridge_confidence.pkl"

    # Seed a BAD existing model via the real path: train on flipped labels.
    X1, y1 = make_data(seed=1)
    seed_result = retrainer.trigger_retrain(
        X_replay=X1, y_replay=1 - y1, reason="seed_bad_model", force=True
    )
    assert seed_result["status"] == "completed"
    assert model_path.exists()
    bad_bytes = model_path.read_bytes()

    # Retrain on clean data: candidate is far better on the clean holdout.
    X2, y2 = make_data(seed=2)
    result = retrainer.trigger_retrain(
        X_replay=X2, y_replay=y2, reason="clean_retrain", force=True
    )

    gate = result["eval_gate"]["ridge"]
    assert gate["verdict"] == GATE_VERDICT_PASSED
    assert gate["candidate_mse"] < gate["existing_mse"]
    assert "ridge" in result["models_retrained"]
    assert result["status"] == "completed"
    assert model_path.read_bytes() != bad_bytes  # file actually rewritten


def test_worse_candidate_refused_and_original_bytes_untouched(tmp_path, caplog):
    """(b) A worse candidate is refused; the old .pkl stays byte-identical."""
    caplog.set_level(logging.INFO, logger="online_retrainer")
    retrainer = make_retrainer(tmp_path)
    model_path = retrainer.model_dir / "ridge_confidence.pkl"

    # Seed a GOOD existing model on clean data.
    X1, y1 = make_data(seed=3)
    seed_result = retrainer.trigger_retrain(
        X_replay=X1, y_replay=y1, reason="seed_good_model", force=True
    )
    assert seed_result["status"] == "completed"
    good_bytes = model_path.read_bytes()

    # Poison ONLY the fit portion (first 80%): candidate learns the inverted
    # mapping, while the temporal-tail holdout keeps true labels — so the
    # existing model scores well and the candidate scores badly on the SAME
    # holdout slice.
    X2, y2 = make_data(seed=4)
    n_hold = int(len(X2) * GATE_HOLDOUT_FRACTION)
    y_poisoned = y2.copy()
    y_poisoned[: len(y2) - n_hold] = 1 - y_poisoned[: len(y2) - n_hold]

    result = retrainer.trigger_retrain(
        X_replay=X2, y_replay=y_poisoned, reason="poisoned_retrain", force=True
    )

    gate = result["eval_gate"]["ridge"]
    assert gate["verdict"] == GATE_VERDICT_REFUSED
    assert gate["reason"].startswith("candidate_worse_than_existing")
    assert gate["candidate_mse"] > gate["existing_mse"]
    assert "ridge" not in result["models_retrained"]
    assert result["status"] == "refused"
    # THE core guarantee: old artifact byte-for-byte untouched.
    assert model_path.read_bytes() == good_bytes
    # Structured, grep-friendly refusal line was emitted.
    assert any(
        "[ONLINE_RETRAIN_GATE]" in rec.message and "verdict=REFUSED" in rec.message
        for rec in caplog.records
    )


def test_tiny_holdout_refused_fail_closed(tmp_path):
    """(c) Holdout below GATE_MIN_HOLDOUT_SAMPLES ⇒ refuse, write nothing."""
    retrainer = make_retrainer(tmp_path)
    model_path = retrainer.model_dir / "ridge_confidence.pkl"

    # 60 samples clears min_samples_for_retrain (50) but leaves a 12-sample
    # holdout (< 20) — too small to judge, so the gate must fail closed.
    X, y = make_data(n=60, seed=5)
    assert int(len(X) * GATE_HOLDOUT_FRACTION) < GATE_MIN_HOLDOUT_SAMPLES

    result = retrainer.trigger_retrain(
        X_replay=X, y_replay=y, reason="tiny_holdout", force=True
    )

    gate = result["eval_gate"]["ridge"]
    assert gate["verdict"] == GATE_VERDICT_REFUSED
    assert gate["reason"].startswith("holdout_too_small")
    assert "ridge" not in result["models_retrained"]
    assert result["status"] == "refused"
    assert not model_path.exists()  # nothing was ever written


def test_missing_existing_model_and_sane_candidate_passes(tmp_path):
    """(d) No baseline on disk + non-degenerate candidate ⇒ pass and write."""
    retrainer = make_retrainer(tmp_path, retrain_rf=True)  # cover 2 heads
    ridge_path = retrainer.model_dir / "ridge_confidence.pkl"
    rf_path = retrainer.model_dir / "rf_risk.pkl"
    assert not ridge_path.exists() and not rf_path.exists()

    X, y = make_data(seed=6)
    result = retrainer.trigger_retrain(
        X_replay=X, y_replay=y, reason="first_deploy", force=True
    )

    for head, path in (("ridge", ridge_path), ("rf", rf_path)):
        gate = result["eval_gate"][head]
        assert gate["verdict"] == GATE_VERDICT_PASSED
        assert gate["reason"] == "no_scorable_existing_model_candidate_sane"
        assert gate["existing_mse"] is None
        assert head in result["models_retrained"]
        assert path.exists()
    assert result["status"] == "completed"


def test_degenerate_candidate_refused_without_baseline(tmp_path):
    """No baseline + constant-prediction candidate ⇒ refuse fail-closed."""
    retrainer = make_retrainer(tmp_path)
    model_path = retrainer.model_dir / "ridge_confidence.pkl"

    # Constant target ⇒ Ridge fits coef=0, intercept=const ⇒ constant preds.
    X, _ = make_data(seed=8)
    y_const = np.ones(len(X), dtype=int)

    result = retrainer.trigger_retrain(
        X_replay=X, y_replay=y_const, reason="degenerate", force=True
    )

    gate = result["eval_gate"]["ridge"]
    assert gate["verdict"] == GATE_VERDICT_REFUSED
    assert gate["reason"] == "degenerate_candidate_constant_predictions"
    assert "ridge" not in result["models_retrained"]
    assert not model_path.exists()
