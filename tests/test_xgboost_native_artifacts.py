from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

from src.training.meta_labeling import MetaLabeler, MetaLabelingConfig
from src.training.meta_labeler_retrainer import MetaLabelerRetrainer
from src.training.trainers.xgboost_trainer import XGBoostTrainer
from src.utils.meta_labeler_artifacts import load_meta_labeler_artifact
from src.utils.xgboost_artifacts import (
    XGBOOST_NATIVE_ARTIFACT_FORMAT,
    load_xgboost_bundle,
    save_native_xgboost_bundle,
)


class _FakeBaseModel:
    def __init__(self):
        self.loaded_from: str | None = None

    def save_model(self, path: str) -> None:
        Path(path).write_text(self.__class__.__name__, encoding="utf-8")

    def load_model(self, path: str) -> None:
        self.loaded_from = str(path)


class XGBRegressor(_FakeBaseModel):
    pass


class XGBClassifier(_FakeBaseModel):
    pass


class Booster(_FakeBaseModel):
    pass


def _install_fake_xgboost(monkeypatch):
    fake_module = SimpleNamespace(
        __version__="9.9.9",
        set_config=lambda **_: None,
        XGBRegressor=XGBRegressor,
        XGBClassifier=XGBClassifier,
        Booster=Booster,
    )
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)
    monkeypatch.setattr("src.utils.xgboost_artifacts._load_xgboost_module", lambda: fake_module)
    return fake_module


def test_save_and_load_native_xgboost_bundle(tmp_path: Path, monkeypatch) -> None:
    _install_fake_xgboost(monkeypatch)
    artifact_path = tmp_path / "xgb_momentum.pkl"

    save_native_xgboost_bundle(
        artifact_path,
        metadata={"feature_names": ["f1", "f2"]},
        models={
            "momentum_model": XGBRegressor(),
            "accel_model": XGBClassifier(),
        },
        alias_map={"momentum_model": "momentum", "accel_model": "accel"},
    )

    assert (tmp_path / "xgb_momentum.momentum.json").exists()
    assert (tmp_path / "xgb_momentum.accel.json").exists()

    payload = pickle.loads(artifact_path.read_bytes())
    assert payload["artifact_format"] == XGBOOST_NATIVE_ARTIFACT_FORMAT
    assert "momentum_model" not in payload
    assert "accel_model" not in payload

    loaded_payload, native_models, source = load_xgboost_bundle(
        artifact_path,
        required_models=("momentum_model", "accel_model"),
        alias_map={"momentum_model": "momentum", "accel_model": "accel"},
    )

    assert source == "native"
    assert loaded_payload["feature_names"] == ["f1", "f2"]
    assert isinstance(native_models["momentum_model"], XGBRegressor)
    assert isinstance(native_models["accel_model"], XGBClassifier)


def test_xgboost_trainer_save_load_uses_native_sidecars(tmp_path: Path, monkeypatch) -> None:
    _install_fake_xgboost(monkeypatch)
    artifact_path = tmp_path / "xgb_momentum.pkl"

    trainer = XGBoostTrainer()
    trainer.momentum_model = XGBRegressor()
    trainer.accel_model = XGBClassifier()
    trainer.scaler = {"scale": 1.0}
    trainer.metrics = {"momentum_mae": 0.1, "acceleration_accuracy": 0.8}
    trainer.feature_names = ["f1"]
    trainer.n_features = 1
    trainer.momentum_norm_factor = 0.42

    trainer.save(str(artifact_path))

    payload = pickle.loads(artifact_path.read_bytes())
    assert payload["artifact_format"] == XGBOOST_NATIVE_ARTIFACT_FORMAT
    assert "momentum_model" not in payload
    assert "accel_model" not in payload

    loaded = XGBoostTrainer()
    loaded.load(str(artifact_path))

    assert isinstance(loaded.momentum_model, XGBRegressor)
    assert isinstance(loaded.accel_model, XGBClassifier)
    assert loaded.scaler == {"scale": 1.0}
    assert loaded.metrics["momentum_mae"] == 0.1
    assert loaded.momentum_norm_factor == 0.42


def test_meta_labeler_save_load_uses_native_sidecar_for_xgboost(tmp_path: Path, monkeypatch) -> None:
    _install_fake_xgboost(monkeypatch)
    artifact_path = tmp_path / "meta_labeler.pkl"

    labeler = MetaLabeler(MetaLabelingConfig())
    labeler.meta_model = XGBClassifier()
    labeler.is_fitted = True
    labeler.feature_names = ["f1", "f2"]
    labeler._primary_accuracy = 0.71
    labeler.primary_selected_indices = None
    labeler.pair_name = "EUR_USD"
    labeler.model_dir = str(tmp_path)

    labeler.save(artifact_path)

    payload = pickle.loads(artifact_path.read_bytes())
    assert payload["artifact_format"] == XGBOOST_NATIVE_ARTIFACT_FORMAT
    assert "meta_model" not in payload
    assert (tmp_path / "meta_labeler.model.json").exists()

    loaded = MetaLabeler.load(artifact_path)

    assert isinstance(loaded.meta_model, XGBClassifier)
    assert loaded.is_fitted is True
    assert loaded.feature_names == ["f1", "f2"]
    assert loaded._primary_accuracy == 0.71


def test_meta_labeler_artifact_loader_supports_retrainer_format(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_xgboost(monkeypatch)
    artifact_path = tmp_path / "meta_labeler.pkl"

    retrainer = MetaLabelerRetrainer(model_dir=tmp_path, output_dir=tmp_path)
    retrainer.xgb_model = XGBClassifier()
    retrainer.scaler = {"scale": 1.0}
    retrainer.ensemble_weights = {"xgb": 0.6, "ridge": 0.2, "rf": 0.2}
    retrainer.feature_names = ["f1"]
    retrainer.is_fitted = True
    retrainer.training_metrics = {"ensemble_val_accuracy": 0.73}

    retrainer.save(artifact_path)

    loaded, source = load_meta_labeler_artifact(artifact_path)

    assert source == "retrainer"
    assert isinstance(loaded.xgb_model, XGBClassifier)
    assert loaded.is_fitted is True
    assert loaded.ensemble_weights["xgb"] == 0.6
