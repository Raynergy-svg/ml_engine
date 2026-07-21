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

from cli.risk_target_training import train_risk_targets
from cli.training_ops import CLI_GATE_PASSED, CLI_GATE_REFUSED


def _dataset(n: int, seed: int, vol_scale: float = 1.0, dd_signal: bool = True, dd_flip_frac: float = 0.0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
    })
    y_vol = np.clip(np.abs(X["f1"].values) * 0.05 * vol_scale + 0.01, 1e-4, None)
    if dd_signal:
        y_dd = (X["f2"].values > np.quantile(X["f2"].values, 0.75)).astype(np.int64)
        if dd_flip_frac > 0.0:
            # Weaken (but don't destroy) the real signal by randomly
            # flipping a fraction of labels — still clears the absolute
            # pre-registered bar (AUC>=0.55), just by less margin than the
            # unflipped quantile signal.
            flip_mask = rng.random(n) < dd_flip_frac
            y_dd = np.where(flip_mask, 1 - y_dd, y_dd)
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


def test_first_deploy_refuses_drawdown_head_that_fails_absolute_prereg_bar(tmp_path):
    """Reproduces the live bug found in the 2026-07-12 roadmap QA audit:
    `trained_data/risk_targets/models/risk_target_model.meta.json` on disk
    carries val_auc_drawdown=0.586 / val_brier_drawdown=0.249, a drawdown
    head that fails the SAME learnability-bar formula pre-registered in
    docs/prereg-risk-target-vol-drawdown-2026-07-08.md (that doc's own
    frozen-OOS evaluation of this exact head reports AUC=0.625/Brier=0.232
    vs. a 0.143 OOS baseline — a different split, same qualitative honest
    FAIL) — yet the on-disk artifact was written and logged as
    CLI_GATE_PASSED, because the gate only ever compared the candidate to
    an incumbent (or auto-passed on first deploy, when there is no
    incumbent at all). A first-deploy candidate whose drawdown target is
    pure noise (AUC ~0.5, well under the 0.55 bar) must now be refused
    even though there is no incumbent to regress against.
    """
    X, y_vol, y_dd = _dataset(700, seed=11, dd_signal=False)
    split = 500
    result = train_risk_targets(
        X[:split], y_vol[:split], y_dd[:split],
        X[split:], y_vol[split:], y_dd[split:],
        feature_names=list(X.columns),
        model_dir=tmp_path,
        **_small_kwargs(),
    )
    assert result["metrics"]["drawdown_learnable"] is False
    assert result["drawdown_gate"]["verdict"] == CLI_GATE_REFUSED
    assert "failed_prereg_absolute_bar" in result["drawdown_gate"]["reason"]
    assert result["verdict"] == CLI_GATE_REFUSED
    assert result["written"] is False
    assert not (tmp_path / "risk_target_model.pkl").exists()
    # The volatility head is a real, learnable signal in this synthetic
    # dataset — confirm the refusal is driven by the drawdown head's
    # absolute bar, not a coincidental volatility regression too.
    assert result["vol_gate"]["verdict"] == CLI_GATE_PASSED


def test_improving_candidate_passes_and_overwrites(tmp_path):
    # First candidate: real but flipped (weak) drawdown signal — clears the
    # absolute pre-registered bar with a comfortable margin (measured
    # val_auc_drawdown~0.77 vs the 0.55 bar, val_brier_drawdown~0.186 vs
    # baseline~0.222 at this exact seed/flip-frac — well clear of the
    # thin-margin flakiness a smaller flip-frac would risk on a library/
    # platform version bump) — unlike pure noise, which the gate now
    # correctly refuses even on first deploy — but still leaves clear room
    # for the second, unflipped candidate to improve on.
    X1, y_vol1, y_dd1 = _dataset(900, seed=4, dd_signal=True, dd_flip_frac=0.20)
    split = 650
    first = train_risk_targets(
        X1[:split], y_vol1[:split], y_dd1[:split],
        X1[split:], y_vol1[split:], y_dd1[split:],
        feature_names=list(X1.columns),
        model_dir=tmp_path,
        **_small_kwargs(),
    )
    assert first["written"] is True
    # Explicit check so a future library/seed drift that erodes this
    # margin fails loudly here, at the fixture assumption, rather than
    # silently failing test_improving_candidate_passes_and_overwrites for
    # an unrelated reason.
    assert first["metrics"]["drawdown_learnable"] is True

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
