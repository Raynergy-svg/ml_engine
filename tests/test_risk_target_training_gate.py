"""Tests for `cli/risk_target_training.py`'s gated retrain path.

NO MOCKS. Real disk via `tmp_path`, real LightGBM training (small/fast
hyperparameters). Verifies the gate reuses the exact same primitives as
`cli/training_ops.py::retrain_gates()` and behaves correctly on:
    1. First deploy (no incumbent) — always writes.
    2. A candidate that regresses on EITHER head — refused, incumbent
       artifact left untouched (byte-for-byte).
    3. A candidate that improves both heads — passes, overwrites.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cli.risk_target_training import train_risk_targets
from cli.training_ops import CLI_GATE_PASSED, CLI_GATE_REFUSED


def _dataset(n: int, seed: int, vol_scale: float = 1.0, dd_signal: bool = True):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
    })
    y_vol = np.clip(np.abs(X["f1"].values) * 0.05 * vol_scale + 0.01, 1e-4, None)
    if dd_signal:
        y_dd = (X["f2"].values > np.quantile(X["f2"].values, 0.75)).astype(np.int64)
    else:
        y_dd = rng.integers(0, 2, n)  # pure noise target — degrades the candidate
    return X, y_vol, y_dd


def _small_kwargs():
    return dict(n_estimators=25, max_depth=3, early_stopping_rounds=8)


def test_first_deploy_always_writes(tmp_path):
    X, y_vol, y_dd = _dataset(600, seed=1)
    split = 450
    result = train_risk_targets(
        X[:split], y_vol[:split], y_dd[:split],
        X[split:], y_vol[split:], y_dd[split:],
        feature_names=list(X.columns),
        model_dir=tmp_path,
        **_small_kwargs(),
    )
    assert result["written"] is True
    assert result["verdict"] == CLI_GATE_PASSED
    assert (tmp_path / "risk_target_model.pkl").exists()
    assert (tmp_path / "risk_target_model.meta.json").exists()
    assert result["vol_gate"]["reason"] == "no_scorable_existing_model_candidate_sane"


def test_regressing_candidate_is_refused_and_incumbent_untouched(tmp_path):
    X1, y_vol1, y_dd1 = _dataset(700, seed=2)
    split = 500
    first = train_risk_targets(
        X1[:split], y_vol1[:split], y_dd1[:split],
        X1[split:], y_vol1[split:], y_dd1[split:],
        feature_names=list(X1.columns),
        model_dir=tmp_path,
        **_small_kwargs(),
    )
    assert first["written"] is True
    incumbent_meta_before = json.loads((tmp_path / "risk_target_model.meta.json").read_text())
    incumbent_pkl_bytes_before = (tmp_path / "risk_target_model.pkl").read_bytes()

    # Second candidate: drawdown target replaced with pure noise -> the
    # classifier head should score materially worse (higher Brier) than
    # the incumbent, refusing the whole (both-heads) artifact.
    X2, y_vol2, y_dd2 = _dataset(700, seed=3, dd_signal=False)
    second = train_risk_targets(
        X2[:split], y_vol2[:split], y_dd2[:split],
        X2[split:], y_vol2[split:], y_dd2[split:],
        feature_names=list(X2.columns),
        model_dir=tmp_path,
        **_small_kwargs(),
    )

    assert second["drawdown_gate"]["verdict"] == CLI_GATE_REFUSED
    assert second["verdict"] == CLI_GATE_REFUSED
    assert second["written"] is False

    # Incumbent artifact must be byte-identical to before the refused attempt.
    incumbent_meta_after = json.loads((tmp_path / "risk_target_model.meta.json").read_text())
    incumbent_pkl_bytes_after = (tmp_path / "risk_target_model.pkl").read_bytes()
    assert incumbent_meta_before == incumbent_meta_after
    assert incumbent_pkl_bytes_before == incumbent_pkl_bytes_after


def test_improving_candidate_passes_and_overwrites(tmp_path):
    # First candidate: noisy drawdown target (weak incumbent).
    X1, y_vol1, y_dd1 = _dataset(900, seed=4, dd_signal=False)
    split = 650
    first = train_risk_targets(
        X1[:split], y_vol1[:split], y_dd1[:split],
        X1[split:], y_vol1[split:], y_dd1[split:],
        feature_names=list(X1.columns),
        model_dir=tmp_path,
        **_small_kwargs(),
    )
    assert first["written"] is True

    # Second candidate: real drawdown signal — should clearly beat the
    # weak incumbent and pass the gate.
    X2, y_vol2, y_dd2 = _dataset(900, seed=4, dd_signal=True)
    second = train_risk_targets(
        X2[:split], y_vol2[:split], y_dd2[:split],
        X2[split:], y_vol2[split:], y_dd2[split:],
        feature_names=list(X2.columns),
        model_dir=tmp_path,
        **_small_kwargs(),
    )
    assert second["drawdown_gate"]["verdict"] == CLI_GATE_PASSED
    assert second["written"] is True
