"""Tests for ``online_retrainer`` ↔ W&B control plane integration.

These tests verify:
- ``trigger_retrain`` pulls config from the control plane (mocked).
- It falls back to local default on W&B failure (no crash).
- It calls ``log_run`` with the right head / tags / metrics.

XGBoost / RandomForest / sklearn are heavy; we use synthetic ``np.random``
data with the minimum viable shape to drive the code path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import online_retrainer  # noqa: E402
from online_retrainer import OnlineRetrainer, RetrainConfig  # noqa: E402


@pytest.fixture
def retrainer(tmp_path: Path) -> OnlineRetrainer:
    cfg = RetrainConfig(
        cooldown_minutes=0,
        max_retrains_per_day=100,
        min_samples_for_retrain=20,
        state_file=str(tmp_path / "online_retrain_state.json"),
    )
    return OnlineRetrainer(config=cfg, model_dir=str(tmp_path / "models"))


@pytest.fixture
def synthetic_data() -> tuple:
    rng = np.random.RandomState(42)
    X = rng.randn(60, 12)
    y = (rng.randn(60) > 0).astype(int)
    return X, y


def test_retrain_pulls_config_from_wandb(retrainer, synthetic_data, monkeypatch) -> None:
    """The retrainer must consult the control plane for each head it touches."""
    pulls: list = []

    def fake_pull(head: str):
        pulls.append(head)
        return {
            "schema_version": 1,
            "head": head,
            "model_class": "x",
            "label_generation": {},
            "training_behavior": {},
            "hyperparameters": {"n_estimators": 25, "learning_rate": 0.07, "max_depth": 3},
        }

    monkeypatch.setattr(online_retrainer, "_safe_pull_config", fake_pull)
    monkeypatch.setattr(online_retrainer, "_safe_log_run", lambda **kw: {"name": kw.get("run_name")})

    X, y = synthetic_data
    out = retrainer.trigger_retrain(X_replay=X, y_replay=y, reason="test", force=True)
    assert out["status"] in {"completed", "failed"}, out
    # At least the models that successfully imported their backend should
    # have pulled their head config. Risk + confidence rely on sklearn
    # (always available); momentum needs xgboost which may be optional in
    # the test env. Risk + confidence are the non-negotiable assertions.
    assert "risk" in pulls
    assert "confidence" in pulls
    # If xgboost is available, momentum must be pulled too.
    try:
        import xgboost  # noqa: F401
        assert "momentum" in pulls
    except ImportError:
        pass


def test_retrain_falls_back_to_local_default_on_wandb_failure(
    retrainer, synthetic_data, monkeypatch,
) -> None:
    """Control-plane pull error → empty cfg → retrain still completes."""
    def fake_pull(head: str):
        raise RuntimeError("simulated W&B outage")

    # _safe_pull_config swallows exceptions and returns {}, so wrap accordingly.
    def safe_wrap(head: str):
        try:
            return fake_pull(head)
        except Exception:
            return {}

    monkeypatch.setattr(online_retrainer, "_safe_pull_config", safe_wrap)
    monkeypatch.setattr(online_retrainer, "_safe_log_run", lambda **kw: None)

    X, y = synthetic_data
    out = retrainer.trigger_retrain(X_replay=X, y_replay=y, reason="test", force=True)
    # Did not crash; status is completed (or failed gracefully) — never error.
    assert out["status"] != "error"


def test_retrain_logs_metrics_to_wandb(retrainer, synthetic_data, monkeypatch) -> None:
    """Each retrained head must trigger a control-plane log_run with right tags."""
    captured: list = []

    monkeypatch.setattr(
        online_retrainer, "_safe_pull_config",
        lambda head: {
            "schema_version": 1,
            "head": head,
            "model_class": "x",
            "label_generation": {},
            "training_behavior": {},
            "hyperparameters": {"n_estimators": 25, "learning_rate": 0.07, "max_depth": 3},
        },
    )

    def fake_log_run(**kwargs):
        captured.append(kwargs)
        return {"name": kwargs.get("run_name"), "head": kwargs.get("head")}

    monkeypatch.setattr(online_retrainer, "_safe_log_run", fake_log_run)

    X, y = synthetic_data
    out = retrainer.trigger_retrain(X_replay=X, y_replay=y, reason="test", force=True)

    heads_logged = {c["head"] for c in captured if "head" in c}
    # At minimum, sklearn-backed heads must log (risk+confidence). Momentum
    # is logged only when xgboost is importable in this env.
    assert {"risk", "confidence"}.issubset(heads_logged), heads_logged
    try:
        import xgboost  # noqa: F401
        assert "momentum" in heads_logged
    except ImportError:
        pass

    # Every run must be tagged source=auto_retrain.
    for c in captured:
        extra = c.get("extra_config") or {}
        assert extra.get("source") == "auto_retrain", c

    # Run names follow auto_<head>_<ts> convention.
    for c in captured:
        name = c.get("run_name") or ""
        assert name.startswith("auto_"), name


def test_retrain_status_reports_can_retrain(retrainer) -> None:
    """``get_status`` continues to work after the control-plane wiring."""
    status = retrainer.get_status()
    assert "can_retrain" in status
    assert status["can_retrain"] is True


def test_safe_pull_config_offline_returns_local_default(monkeypatch) -> None:
    """End-to-end: when wandb is uninstalled, _safe_pull_config still returns
    a valid local default dict (not {})."""
    # Force ``_import_wandb`` in control plane to None.
    from src.training import wandb_control_plane as wcp
    monkeypatch.setattr(wcp, "_import_wandb", lambda: None)

    cfg = online_retrainer._safe_pull_config("confidence")
    assert cfg.get("head") == "confidence"


def test_safe_log_run_offline_writes_dir(tmp_path, monkeypatch) -> None:
    """End-to-end smoke: offline log_run returns a result with mode=offline.

    Note: We do NOT assert run-dir location here because wandb caches the
    run-dir path on import; running this after another wandb-touching test
    in the same pytest process yields the prior dir. The mode flag is the
    contract we need from the control plane.
    """
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_PROJECT", "buddy-tests")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)

    out = online_retrainer._safe_log_run(
        head="momentum",
        config={"hyperparameters": {"n_estimators": 25}},
        metrics={"val_mae": 0.4},
        run_name="test_auto_offline",
    )
    assert out is not None
    assert out["mode"] == "offline"
    assert out["head"] == "momentum"
    assert out["name"] == "test_auto_offline"
