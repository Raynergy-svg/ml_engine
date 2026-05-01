"""Tests for the confidence-training W&B integration.

Covers:
- ``WANDB_DISABLED=1`` short-circuits all calls (no wandb.init invoked).
- Offline mode is selected when ``WANDB_API_KEY`` is unset.
- Online mode is selected when ``WANDB_API_KEY`` is set.
- The metric payload contains the spec-mandated keys.
- Default project name + run name format match the spec.
- Leak-detection-fired flag fires correctly when R² > expected_band[1].
- Artifact upload is gated by ``WANDB_LOG_ARTIFACTS=1``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.training import wandb_confidence as wc  # noqa: E402


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all W&B env vars so each test starts from a known baseline."""
    for k in [
        "WANDB_DISABLED", "WANDB_API_KEY", "WANDB_MODE",
        "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_LOG_ARTIFACTS",
    ]:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _make_metrics(r2: float = 0.12, mae: float = 36.5) -> dict:
    return {
        "r2_score": r2,
        "confidence_mae": mae,
        "model_type": "lightgbm",
        "n_estimators": 100,
        "expected_r2_band": [0.05, 0.30],
        "label_mode": "journal_outcome_blend",
        "n_real_labels": 240,
        "n_pseudo_labels": 61810,
        "class_balance_real": 0.37,
        "class_balance_pseudo": 0.57,
        "class_balance": None,
        "leak_fix_version": "2026-04-30",
        "score_range": [20.0, 95.0],
    }


def _make_label_meta() -> dict:
    return {
        "label_mode": "journal_outcome_blend",
        "n_real": 240,
        "n_pseudo": 61810,
        "class_balance_real": 0.37,
        "class_balance_pseudo": 0.57,
        "leak_fix_version": "2026-04-30",
    }


# ─────────────────────────────────────────────────────────────────
# build_metric_payload — pure unit tests (no W&B import required)
# ─────────────────────────────────────────────────────────────────

def test_payload_contains_spec_keys():
    payload = wc.build_metric_payload(_make_metrics(), _make_label_meta())
    for required in (
        "val_r2", "val_mae",
        "n_real", "n_pseudo",
        "class_balance_real", "class_balance_pseudo",
        "expected_r2_band_low", "expected_r2_band_high",
        "leak_detection_fired",
    ):
        assert required in payload, f"missing required metric: {required}"


def test_leak_detection_fires_when_r2_exceeds_band():
    """R² above ``expected_r2_band[1]`` (0.30 default) → flag set."""
    payload = wc.build_metric_payload(_make_metrics(r2=0.95), _make_label_meta())
    assert payload["leak_detection_fired"] == 1.0


def test_leak_detection_does_not_fire_in_band():
    payload = wc.build_metric_payload(_make_metrics(r2=0.18), _make_label_meta())
    assert payload["leak_detection_fired"] == 0.0


def test_payload_handles_missing_label_metadata():
    payload = wc.build_metric_payload(_make_metrics(), label_metadata=None)
    # Falls back to metrics dict for n_real / n_pseudo
    assert payload["n_real"] == 240.0
    assert payload["n_pseudo"] == 61810.0


def test_payload_drops_none_values():
    """Class balance None should be dropped, not sent as 'None' string."""
    metrics = _make_metrics()
    metrics["class_balance"] = None
    payload = wc.build_metric_payload(metrics, _make_label_meta())
    # class_balance_overall comes from metrics["class_balance"]
    assert "class_balance_overall" not in payload


# ─────────────────────────────────────────────────────────────────
# Disabled / offline / online mode resolution
# ─────────────────────────────────────────────────────────────────

def test_disabled_short_circuits(clean_env):
    """WANDB_DISABLED=1 must skip wandb.init entirely."""
    clean_env.setenv("WANDB_DISABLED", "1")
    fake_wandb = mock.MagicMock()

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        result = wc.log_confidence_training_run(
            run_name="confidence_joint",
            metrics=_make_metrics(),
            label_metadata=_make_label_meta(),
        )
    assert result is None
    fake_wandb.init.assert_not_called()


def test_disabled_handles_truthy_variants(clean_env):
    """All truthy WANDB_DISABLED variants short-circuit."""
    fake_wandb = mock.MagicMock()
    for variant in ("1", "true", "True", "yes", "ON"):
        clean_env.setenv("WANDB_DISABLED", variant)
        with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
            assert wc.log_confidence_training_run(
                run_name="x", metrics=_make_metrics()
            ) is None
    fake_wandb.init.assert_not_called()


def test_offline_when_api_key_unset(clean_env):
    """No WANDB_API_KEY → offline mode."""
    fake_run = mock.MagicMock()
    fake_run.url = None
    fake_run.id = "offline-id"
    fake_wandb = mock.MagicMock()
    fake_wandb.init.return_value = fake_run

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        result = wc.log_confidence_training_run(
            run_name="confidence_EUR_USD",
            metrics=_make_metrics(),
            label_metadata=_make_label_meta(),
        )
    assert result is not None
    assert result["mode"] == "offline"
    assert fake_wandb.init.call_args.kwargs["mode"] == "offline"


def test_online_when_api_key_set(clean_env):
    """WANDB_API_KEY present → online mode."""
    clean_env.setenv("WANDB_API_KEY", "fake-key")
    fake_run = mock.MagicMock()
    fake_run.url = "https://wandb.ai/x/y/runs/z"
    fake_run.id = "online-id"
    fake_wandb = mock.MagicMock()
    fake_wandb.init.return_value = fake_run

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        result = wc.log_confidence_training_run(
            run_name="confidence_joint",
            metrics=_make_metrics(),
            label_metadata=_make_label_meta(),
        )
    assert result is not None
    assert result["mode"] == "online"
    assert result["run_url"] == "https://wandb.ai/x/y/runs/z"


# ─────────────────────────────────────────────────────────────────
# Project + run-name format
# ─────────────────────────────────────────────────────────────────

def test_default_project_name(clean_env):
    fake_run = mock.MagicMock()
    fake_run.url = None
    fake_run.id = "x"
    fake_wandb = mock.MagicMock()
    fake_wandb.init.return_value = fake_run

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        result = wc.log_confidence_training_run(
            run_name="confidence_joint", metrics=_make_metrics(),
        )
    assert result["project"] == "buddy-confidence-training"


def test_project_name_overridable(clean_env):
    clean_env.setenv("WANDB_PROJECT", "custom-project")
    fake_run = mock.MagicMock()
    fake_run.url = None
    fake_run.id = "x"
    fake_wandb = mock.MagicMock()
    fake_wandb.init.return_value = fake_run

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        result = wc.log_confidence_training_run(
            run_name="confidence_joint", metrics=_make_metrics(),
        )
    assert result["project"] == "custom-project"


def test_run_name_format(clean_env):
    """Run name = ``{run_name}_{YYYYMMDD}``."""
    fake_run = mock.MagicMock()
    fake_run.url = None
    fake_run.id = "x"
    fake_wandb = mock.MagicMock()
    fake_wandb.init.return_value = fake_run

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        result = wc.log_confidence_training_run(
            run_name="confidence_AUD_USD", metrics=_make_metrics(),
        )
    final_name = result["name"]
    assert final_name.startswith("confidence_AUD_USD_")
    assert len(final_name) == len("confidence_AUD_USD_") + 8  # 8-digit date


# ─────────────────────────────────────────────────────────────────
# Artifact upload gate
# ─────────────────────────────────────────────────────────────────

def test_artifact_upload_skipped_by_default(clean_env, tmp_path):
    pkl = tmp_path / "ridge_confidence.pkl"
    pkl.write_bytes(b"\x80\x04\x00")  # any bytes

    fake_run = mock.MagicMock()
    fake_run.url = None
    fake_run.id = "x"
    fake_wandb = mock.MagicMock()
    fake_wandb.init.return_value = fake_run

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        wc.log_confidence_training_run(
            run_name="confidence_joint",
            metrics=_make_metrics(),
            model_path=str(pkl),
        )
    fake_wandb.Artifact.assert_not_called()


def test_artifact_upload_enabled_with_env(clean_env, tmp_path):
    clean_env.setenv("WANDB_LOG_ARTIFACTS", "1")
    pkl = tmp_path / "ridge_confidence.pkl"
    pkl.write_bytes(b"\x80\x04\x00")

    fake_artifact = mock.MagicMock()
    fake_run = mock.MagicMock()
    fake_run.url = None
    fake_run.id = "x"
    fake_wandb = mock.MagicMock()
    fake_wandb.init.return_value = fake_run
    fake_wandb.Artifact.return_value = fake_artifact

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        wc.log_confidence_training_run(
            run_name="confidence_joint",
            metrics=_make_metrics(),
            model_path=str(pkl),
        )
    fake_wandb.Artifact.assert_called_once()
    fake_artifact.add_file.assert_called_once()
    fake_run.log_artifact.assert_called_once_with(fake_artifact)


# ─────────────────────────────────────────────────────────────────
# Robustness — never crash training
# ─────────────────────────────────────────────────────────────────

def test_wandb_unavailable_returns_none(clean_env):
    """No wandb installed → graceful no-op."""
    with mock.patch.object(wc, "_import_wandb", return_value=None):
        result = wc.log_confidence_training_run(
            run_name="confidence_joint", metrics=_make_metrics(),
        )
    assert result is None


def test_init_exception_does_not_propagate(clean_env):
    """If wandb.init blows up, the function returns None; caller continues."""
    fake_wandb = mock.MagicMock()
    fake_wandb.init.side_effect = RuntimeError("network down")

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        # Must NOT raise
        result = wc.log_confidence_training_run(
            run_name="confidence_joint", metrics=_make_metrics(),
        )
    assert result is None


def test_log_called_with_payload(clean_env):
    """The payload built by ``build_metric_payload`` gets logged."""
    fake_run = mock.MagicMock()
    fake_run.url = None
    fake_run.id = "x"
    fake_wandb = mock.MagicMock()
    fake_wandb.init.return_value = fake_run

    with mock.patch.object(wc, "_import_wandb", return_value=fake_wandb):
        wc.log_confidence_training_run(
            run_name="confidence_joint",
            metrics=_make_metrics(),
            label_metadata=_make_label_meta(),
        )
    # log called once with a dict containing the spec metric keys
    assert fake_run.log.called
    logged = fake_run.log.call_args.args[0]
    assert "val_r2" in logged
    assert "val_mae" in logged
    assert "leak_detection_fired" in logged
