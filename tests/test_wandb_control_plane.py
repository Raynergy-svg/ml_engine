"""Tests for the W&B Training Control Plane (``src/training/wandb_control_plane``).

Covers:
- pull_config returns local default when artifact absent
- pull_config returns artifact contents when present (mocked)
- push_config creates a new version (mocked)
- log_run handles offline mode gracefully (no API key → wandb/ dir written)
- log_run handles network failures (mocked exception → no crash)
- seed_default is idempotent (existing artifact → no duplicate upload)
- schema validation: every default JSON has required keys + sane ranges

All W&B network access is mocked. No real API calls.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# Ensure project root on path (in case test runs from a different cwd).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import wandb_control_plane as wcp  # noqa: E402


# ---------------------------------------------------------------------------
# Schema validation tests for the 7 default JSON configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head", list(wcp.SUPPORTED_HEADS))
def test_local_default_present_and_valid(head: str) -> None:
    cfg = wcp.load_local_default(head)
    assert cfg is not None, f"Default JSON missing for head={head}"
    assert wcp.validate_schema(cfg) is True
    assert cfg["head"] == head


@pytest.mark.parametrize("head", list(wcp.SUPPORTED_HEADS))
def test_default_required_subkeys(head: str) -> None:
    cfg = wcp.load_local_default(head)
    label = cfg.get("label_generation") or {}
    behavior = cfg.get("training_behavior") or {}
    hp = cfg.get("hyperparameters") or {}

    # Label generation must have the strategic keys.
    for k in ("lookahead_bars", "sl_atr_mult", "tp_atr_mult", "label_mode"):
        assert k in label, f"{head} missing label.{k}"
    # Training behavior must have at least these.
    for k in ("expected_r2_band", "min_real_labels_per_pair"):
        assert k in behavior, f"{head} missing training_behavior.{k}"
    # Top-3 HPs (count varies per model class but at least 3 keys).
    assert len(hp) >= 3, f"{head} hyperparameters has fewer than 3 keys: {list(hp)}"


@pytest.mark.parametrize("head", list(wcp.SUPPORTED_HEADS))
def test_default_ranges_are_consistent(head: str) -> None:
    """Every HP / label_gen value should be inside the declared range."""
    cfg = wcp.load_local_default(head)
    ranges = cfg.get("ranges") or {}
    flat = {**(cfg.get("hyperparameters") or {}),
            **(cfg.get("label_generation") or {}),
            **(cfg.get("training_behavior") or {})}
    for key, rng in ranges.items():
        if key not in flat:
            continue
        val = flat[key]
        if not isinstance(rng, list) or not rng:
            continue
        # Categorical (e.g., label_mode)
        if isinstance(rng[0], str):
            assert val in rng, f"{head}.{key}={val} not in {rng}"
            continue
        # Numeric range
        if len(rng) == 2 and all(isinstance(x, (int, float)) for x in rng):
            assert rng[0] <= val <= rng[1], f"{head}.{key}={val} outside {rng}"


def test_validate_schema_rejects_missing_keys() -> None:
    bad = {"schema_version": 1, "head": "x"}
    assert wcp.validate_schema(bad) is False


def test_validate_schema_rejects_non_dict() -> None:
    assert wcp.validate_schema(["not", "a", "dict"]) is False  # type: ignore[arg-type]


def test_validate_schema_rejects_non_int_version() -> None:
    cfg = wcp.load_local_default("confidence")
    cfg["schema_version"] = "1"  # str instead of int
    assert wcp.validate_schema(cfg) is False


# ---------------------------------------------------------------------------
# pull_config
# ---------------------------------------------------------------------------


def test_pull_config_returns_default_when_artifact_absent(monkeypatch) -> None:
    """No W&B available → fall back to local default."""
    # Force "wandb not installed" path.
    monkeypatch.setattr(wcp, "_import_wandb", lambda: None)

    cfg = wcp.pull_config("confidence")
    assert cfg["head"] == "confidence"
    assert cfg["schema_version"] == wcp.CURRENT_SCHEMA_VERSION


def test_pull_config_returns_artifact_when_present(monkeypatch, tmp_path) -> None:
    """Mock W&B artifact; pull_config should return the artifact contents."""
    payload = {
        "schema_version": 1,
        "head": "confidence",
        "model_class": "lightgbm",
        "label_generation": {"lookahead_bars": 24, "sl_atr_mult": 1.0,
                             "tp_atr_mult": 1.5, "label_mode": "real"},
        "training_behavior": {"expected_r2_band": [0.05, 0.30],
                              "min_real_labels_per_pair": 50},
        "hyperparameters": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
    }

    # Build a fake wandb module that returns a fake Api with a fake artifact.
    class FakeArtifact:
        def download(self, root: str) -> str:
            (Path(root) / "confidence_training_config.json").write_text(json.dumps(payload))
            return root

    class FakeApi:
        def __init__(self, *a, **kw):  # noqa: D401
            pass

        def artifact(self, name, type=None):  # noqa: A002 — mirror W&B signature
            assert "confidence_training_config:latest" in name
            return FakeArtifact()

    fake_wandb = SimpleNamespace(Api=FakeApi, Artifact=MagicMock(), init=MagicMock())
    monkeypatch.setattr(wcp, "_import_wandb", lambda: fake_wandb)
    # Force ONLINE mode so the artifact path is taken.
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    cfg = wcp.pull_config("confidence")
    assert cfg["hyperparameters"]["n_estimators"] == 300
    assert cfg["label_generation"]["lookahead_bars"] == 24


def test_pull_config_falls_back_on_artifact_error(monkeypatch) -> None:
    """Artifact fetch raises → fall back to local default; never raises."""
    class FakeApi:
        def __init__(self, *a, **kw):
            pass

        def artifact(self, name, type=None):  # noqa: A002
            raise RuntimeError("network down")

    fake_wandb = SimpleNamespace(Api=FakeApi)
    monkeypatch.setattr(wcp, "_import_wandb", lambda: fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    cfg = wcp.pull_config("confidence")
    assert cfg["head"] == "confidence"  # local default


def test_pull_config_returns_empty_for_unknown_head_with_no_default(monkeypatch) -> None:
    monkeypatch.setattr(wcp, "_import_wandb", lambda: None)
    out = wcp.pull_config("nonsense_head")
    assert out == {}


def test_pull_config_disabled_short_circuits(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_DISABLED", "1")
    # Even if wandb is installed, disabled→no fetch attempted.
    cfg = wcp.pull_config("confidence")
    assert cfg["head"] == "confidence"  # came from local default


# ---------------------------------------------------------------------------
# push_config
# ---------------------------------------------------------------------------


def test_push_config_creates_new_version(monkeypatch) -> None:
    fake_run = MagicMock()
    fake_run.url = "https://wandb.ai/x/y/runs/abc"
    fake_run.id = "abc"
    fake_init = MagicMock(return_value=fake_run)
    fake_artifact_cls = MagicMock()
    fake_artifact = MagicMock()
    fake_artifact_cls.return_value = fake_artifact

    fake_wandb = SimpleNamespace(
        init=fake_init,
        Artifact=fake_artifact_cls,
        Api=MagicMock(),
    )
    monkeypatch.setattr(wcp, "_import_wandb", lambda: fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    cfg = wcp.load_local_default("confidence")
    out = wcp.push_config("confidence", cfg)
    assert out is not None
    assert out["head"] == "confidence"
    assert out["run_id"] == "abc"
    fake_init.assert_called_once()
    # Artifact created with correct name + type
    fake_artifact_cls.assert_called_once()
    kwargs = fake_artifact_cls.call_args.kwargs
    assert kwargs["name"] == "confidence_training_config"
    assert kwargs["type"] == "training-config"
    fake_run.log_artifact.assert_called_once()


def test_push_config_rejects_invalid_schema(monkeypatch) -> None:
    fake_wandb = SimpleNamespace(init=MagicMock(), Artifact=MagicMock(), Api=MagicMock())
    monkeypatch.setattr(wcp, "_import_wandb", lambda: fake_wandb)
    out = wcp.push_config("confidence", {"bogus": "config"})
    assert out is None
    fake_wandb.init.assert_not_called()


# ---------------------------------------------------------------------------
# log_run
# ---------------------------------------------------------------------------


def test_log_run_handles_offline_mode_gracefully(monkeypatch, tmp_path) -> None:
    """No API key → offline mode → wandb/ dir written, no crash."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_PROJECT", "buddy-tests")
    monkeypatch.setenv("WANDB_SILENT", "true")

    cfg = wcp.load_local_default("confidence")
    out = wcp.log_run(
        head="confidence",
        config=cfg,
        metrics={"val_r2": 0.21, "val_mae": 18.4},
        run_name="test_offline_run",
        extra_config={"source": "test"},
    )
    assert out is not None
    assert out["mode"] == "offline"
    assert out["head"] == "confidence"
    # wandb/ side effect: at least one offline run dir created.
    runs = list(Path(tmp_path).glob("**/run-*"))
    assert runs, f"No offline run dirs under {tmp_path}"


def test_log_run_handles_network_failure(monkeypatch) -> None:
    """wandb.init raising must not crash the caller."""
    fake_wandb = SimpleNamespace(init=MagicMock(side_effect=RuntimeError("boom")),
                                 Artifact=MagicMock(), Api=MagicMock())
    monkeypatch.setattr(wcp, "_import_wandb", lambda: fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")

    out = wcp.log_run(
        head="confidence",
        config=wcp.load_local_default("confidence"),
        metrics={"val_r2": 0.18},
    )
    assert out is None  # graceful degradation


def test_log_run_disabled_short_circuits(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_DISABLED", "1")
    out = wcp.log_run(head="confidence", config={}, metrics={"val_r2": 0.1})
    assert out is None


def test_log_run_coerces_metrics(monkeypatch, tmp_path) -> None:
    """Non-numeric metric values should be dropped, booleans coerced."""
    captured: dict = {}

    fake_run = MagicMock()
    fake_run.url = "u"
    fake_run.id = "i"

    def _log(payload):
        captured.update(payload)

    fake_run.log = _log
    fake_run.summary = MagicMock()
    fake_init = MagicMock(return_value=fake_run)
    fake_wandb = SimpleNamespace(init=fake_init, Artifact=MagicMock(), Api=MagicMock())
    monkeypatch.setattr(wcp, "_import_wandb", lambda: fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")

    wcp.log_run(
        head="confidence",
        config={},
        metrics={"val_r2": 0.21, "label": "string-skipped",
                 "leak_fired": True, "n_real": None, "val_mae": "0.4"},
    )
    assert "val_r2" in captured and captured["val_r2"] == pytest.approx(0.21)
    assert "label" not in captured  # non-numeric dropped
    assert captured.get("leak_fired") == 1.0  # bool coerced
    assert "n_real" not in captured  # None dropped
    assert captured.get("val_mae") == pytest.approx(0.4)  # numeric string coerced


# ---------------------------------------------------------------------------
# seed_default
# ---------------------------------------------------------------------------


def test_seed_default_idempotent_when_artifact_exists(monkeypatch) -> None:
    """If artifact exists in W&B, seed_default must NOT push again."""
    class FakeApi:
        def __init__(self, *a, **kw):
            pass

        def artifact(self, name, type=None):  # noqa: A002 — exists path
            return MagicMock()

    fake_init = MagicMock()
    fake_wandb = SimpleNamespace(init=fake_init, Artifact=MagicMock(), Api=FakeApi)
    monkeypatch.setattr(wcp, "_import_wandb", lambda: fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    out1 = wcp.seed_default("confidence")
    out2 = wcp.seed_default("confidence")
    assert out1 is not None and out2 is not None
    # No init call → no push happened.
    fake_init.assert_not_called()


def test_seed_default_pushes_when_artifact_missing(monkeypatch) -> None:
    """If artifact missing in W&B, seed_default must push the local default."""
    class FakeApi:
        def __init__(self, *a, **kw):
            pass

        def artifact(self, name, type=None):  # noqa: A002 — missing
            raise RuntimeError("artifact not found")

    fake_run = MagicMock()
    fake_run.url = "u"
    fake_run.id = "i"
    fake_init = MagicMock(return_value=fake_run)
    fake_wandb = SimpleNamespace(init=fake_init, Artifact=MagicMock(), Api=FakeApi)
    monkeypatch.setattr(wcp, "_import_wandb", lambda: fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    out = wcp.seed_default("confidence")
    assert out is not None
    fake_init.assert_called()  # push happened


# ---------------------------------------------------------------------------
# apply_config_to_trainer
# ---------------------------------------------------------------------------


def test_apply_config_routes_lightgbm_hps_to_trainer_attrs() -> None:
    cfg = wcp.load_local_default("confidence")
    cfg["hyperparameters"] = {"n_estimators": 333, "learning_rate": 0.07, "max_depth": 5}

    class FakeTrainer:
        def __init__(self):
            self.n_estimators = 100
            self.learning_rate = 0.1
            self.max_depth = 6

    t = FakeTrainer()
    wcp.apply_config_to_trainer(t, cfg)
    assert t.n_estimators == 333
    assert t.learning_rate == pytest.approx(0.07)
    assert t.max_depth == 5


def test_apply_config_routes_keras_hps_to_inner_config() -> None:
    cfg = wcp.load_local_default("direction")
    cfg["hyperparameters"] = {"epochs": 77, "learning_rate": 0.001, "dropout": 0.55}

    class InnerCfg:
        epochs = 100
        learning_rate = 0.0003
        transformer_dropout = 0.2
        tcn_dropout = 0.6

    class FakeTrainer:
        def __init__(self):
            self.config = InnerCfg()

    t = FakeTrainer()
    wcp.apply_config_to_trainer(t, cfg)
    assert t.config.epochs == 77
    assert t.config.learning_rate == pytest.approx(0.001)
    # ``dropout`` aliased onto both transformer_dropout and tcn_dropout.
    assert t.config.transformer_dropout == pytest.approx(0.55)
    assert t.config.tcn_dropout == pytest.approx(0.55)


def test_apply_config_unknown_key_does_not_raise() -> None:
    class FakeTrainer:
        pass

    t = FakeTrainer()
    # Should not raise even though no attrs match.
    wcp.apply_config_to_trainer(t, {
        "schema_version": 1,
        "head": "x",
        "model_class": "y",
        "label_generation": {"unknown_key": 5},
        "training_behavior": {},
        "hyperparameters": {},
    })
    # apply_config_to_trainer is safe-by-default: it only sets attrs that
    # the trainer already declared (via hasattr), so unknown keys are dropped
    # rather than creating orphan attrs (mirroring the
    # ``Config Adjustment Consumer Verification`` rule from
    # .claude/rules/improvement.md).
    assert not hasattr(t, "unknown_key")
