from __future__ import annotations

import pickle
from pathlib import Path

from src.scanner.filters.volatility import VolatilityFilter
from src.scanner.gates import GateEvaluator


class _DummyModel:
    pass


def test_gate_evaluator_uses_compat_loader_for_transformer_and_tcn(
    tmp_path: Path,
    monkeypatch,
):
    joint_dir = tmp_path / "joint"
    joint_dir.mkdir()
    (joint_dir / "transformer_direction.keras").touch()
    (joint_dir / "tcn_volatility_regime.keras").touch()

    with open(joint_dir / "tcn_volatility_regime.meta.pkl", "wb") as fh:
        pickle.dump({"scaler": "dummy_scaler", "feature_names": ["f1", "f2"]}, fh)

    calls: list[tuple[str, bool]] = []
    dummy_model = _DummyModel()

    def _fake_load_keras_model(model_path: str, compile: bool = True, **_: object):
        calls.append((str(model_path), compile))
        return dummy_model, {"approach_used": "custom_deserialization"}

    monkeypatch.setattr("src.utils.keras_model_loader.load_keras_model", _fake_load_keras_model)

    evaluator = GateEvaluator(tmp_path)

    assert evaluator._load_transformer() is True
    assert evaluator._transformer is dummy_model

    assert evaluator._load_tcn_volatility() is True
    assert evaluator._tcn_volatility is dummy_model
    assert evaluator._tcn_scaler == "dummy_scaler"
    assert evaluator._tcn_feature_names == ["f1", "f2"]

    assert calls == [
        (str(joint_dir / "transformer_direction.keras"), False),
        (str(joint_dir / "tcn_volatility_regime.keras"), False),
    ]


def test_volatility_filter_uses_compat_loader_for_tcn(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / "tcn_volatility_regime.keras").touch()

    with open(tmp_path / "tcn_volatility_regime.meta.pkl", "wb") as fh:
        pickle.dump({"scaler": "dummy_scaler"}, fh)

    calls: list[tuple[str, bool]] = []
    dummy_model = _DummyModel()

    def _fake_load_keras_model(model_path: str, compile: bool = True, **_: object):
        calls.append((str(model_path), compile))
        return dummy_model, {"approach_used": "keras_native"}

    monkeypatch.setattr("src.utils.keras_model_loader.load_keras_model", _fake_load_keras_model)

    volatility_filter = VolatilityFilter(model_dir=tmp_path)

    assert volatility_filter._load_tcn_model() is True
    assert volatility_filter._tcn_model is dummy_model
    assert volatility_filter._tcn_scaler == "dummy_scaler"
    assert calls == [(str(tmp_path / "tcn_volatility_regime.keras"), False)]
