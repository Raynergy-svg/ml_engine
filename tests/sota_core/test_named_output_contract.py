"""Consumer regressions for the named SOTA prediction-head contract."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow import keras

from src.sota_core.inference import SOTAInference, SOTAInferenceConfig
from src.sota_core.raw_sequence_model import ModelConfig, RawSequenceModel
from src.sota_core.trainer import SOTATrainer, TrainerConfig


@pytest.fixture
def frame():
    close = 100.0 + np.sin(np.arange(40, dtype=np.float32) / 3)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": np.arange(40) + 100},
        index=pd.date_range("2024-01-01", periods=40, freq="h", name="time"),
    )


def _wrapper(tmp_path, payload):
    # Disk loading is deliberately disabled; only the model's return value is a fixture.
    wrapper = SOTAInference(SOTAInferenceConfig(model_path=str(tmp_path / "absent.keras"), seq_len=8))
    wrapper.model.model = SimpleNamespace(predict=lambda *_args, **_kwargs: payload)
    wrapper._loaded = True
    return wrapper


def _heads():
    return {
        "direction_5": np.array([[0.8]], dtype=np.float32),
        "regime": np.array([[0.1, 0.6, 0.2, 0.1]], dtype=np.float32),
    }


@pytest.mark.parametrize("layout", ["named", "reversed_named", "legacy_named", "legacy_list", "legacy_tuple"])
def test_valid_prediction_contract_is_decoded_without_positional_dict_unpacking(tmp_path, frame, layout):
    heads = _heads()
    if layout == "reversed_named":
        payload = dict(reversed(list(heads.items())))
    elif layout == "legacy_named":
        payload = {"direction": heads["direction_5"], "regime": heads["regime"]}
    elif layout in ("legacy_list", "legacy_tuple"):
        payload = [heads["direction_5"], heads["regime"]]
        if layout == "legacy_tuple":
            payload = tuple(payload)
    else:
        payload = heads
    wrapper = _wrapper(tmp_path, payload)
    signal = wrapper.predict("TEST", frame)
    assert signal.reason == "sota_signal"
    assert signal.tcn_probability == pytest.approx(0.8)
    assert signal.metadata["regime"] == "NORMAL"
    assert wrapper._prediction_history == pytest.approx([0.8])
    regime, probabilities = wrapper.predict_regime_only("TEST", frame)
    assert regime == "NORMAL"
    np.testing.assert_allclose(probabilities, heads["regime"][0])
    assert wrapper._prediction_history == pytest.approx([0.8])


@pytest.mark.parametrize(
    "bad",
    [
        None,
        (),
        (np.zeros((1, 1)),),
        {},
        {"direction_5": np.array([[0.8]])},
        {"regime": np.array([[0.1, 0.6, 0.2, 0.1]])},
        {"direction_10": np.array([[0.8]]), "regime": np.array([[0.1, 0.6, 0.2, 0.1]])},
        {
            "direction_5": np.array([[0.8]]),
            "direction_10": np.array([[0.2]]),
            "regime": np.array([[0.1, 0.6, 0.2, 0.1]]),
        },
        {"direction": np.array([[0.8]]), "direction_5": np.array([[0.2]]), "regime": np.array([[0.1, 0.6, 0.2, 0.1]])},
    ],
)
def test_unsupported_or_ambiguous_direction_heads_fail_closed(tmp_path, frame, bad):
    wrapper = _wrapper(tmp_path, bad)
    signal = wrapper.predict("TEST", frame)
    assert signal.direction == "HOLD" and signal.trade is False
    assert signal.reason == "prediction_error"
    assert wrapper._prediction_history == []


@pytest.mark.parametrize(
    "head, bad",
    [
        ("direction_5", np.array([[np.nan]])),
        ("direction_5", np.array([[np.inf]])),
        ("direction_5", np.array([[-0.1]])),
        ("direction_5", np.array([[1.1]])),
        ("direction_5", np.array([0.8])),
        ("direction_5", np.array([[0.8], [0.2]])),
        ("direction_5", np.array([[0.1, 0.8, 0.1]])),
        ("direction_5", np.array([["0.8"]])),
        ("regime", np.array([[0.1, np.nan, 0.2, 0.1]])),
        ("regime", np.array([[0.1, np.inf, 0.2, 0.1]])),
        ("regime", np.array([[-0.1, 0.8, 0.2, 0.1]])),
        ("regime", np.array([[0.1, 1.2, 0.2, 0.1]])),
        ("regime", np.array([[0.1, 0.1, 0.1, 0.1]])),
        ("regime", np.array([0.1, 0.6, 0.2, 0.1])),
        ("regime", np.array([[0.1, 0.8, 0.1]])),
        ("regime", np.array([[0.1, 0.6, 0.2, 0.1]] * 2)),
        ("regime", np.array([["0.1", "0.6", "0.2", "0.1"]])),
    ],
)
def test_malformed_probabilities_do_not_escape_or_enter_calibration(tmp_path, frame, head, bad):
    payload = _heads()
    payload[head] = bad
    wrapper = _wrapper(tmp_path, payload)
    signal = wrapper.predict("TEST", frame)
    assert signal.direction == "HOLD" and signal.trade is False
    assert signal.reason == "prediction_error"
    assert wrapper._prediction_history == []
    if head == "regime":
        regime, probabilities = wrapper.predict_regime_only("TEST", frame)
        assert regime == "UNKNOWN"
        np.testing.assert_array_equal(probabilities, np.full(4, 0.25, dtype=np.float32))


@pytest.mark.parametrize("payload", [None, (), {}, {"direction_5": np.array([[0.8]])}])
def test_missing_regime_head_uses_unknown_fallback(tmp_path, frame, payload):
    wrapper = _wrapper(tmp_path, payload)
    regime, probabilities = wrapper.predict_regime_only("TEST", frame)
    assert regime == "UNKNOWN"
    np.testing.assert_array_equal(probabilities, np.full(4, 0.25, dtype=np.float32))
    assert wrapper._prediction_history == []


def test_regime_only_path_does_not_select_a_direction_horizon(tmp_path, frame):
    payload = _heads()
    payload["direction_10"] = np.array([[0.2]], dtype=np.float32)
    wrapper = _wrapper(tmp_path, payload)
    regime, probabilities = wrapper.predict_regime_only("TEST", frame)
    assert regime == "NORMAL"
    np.testing.assert_allclose(probabilities, payload["regime"][0])
    assert wrapper._prediction_history == []


def _small_model(direction_classes=2, direction_horizons=(5,)):
    return RawSequenceModel(
        ModelConfig(
            seq_len=8,
            d_model=8,
            num_heads=2,
            num_layers=1,
            cnn_filters=[8],
            dense_units=8,
            dropout=0.0,
            direction_classes=direction_classes,
            direction_horizons=direction_horizons,
        )
    )


def test_actual_keras_named_outputs_pass_through_the_real_wrapper(tmp_path, frame):
    raw = _small_model()
    with tf.device("/CPU:0"):
        raw.build()
        wrapper = _wrapper(tmp_path, None)
        wrapper.model = raw
        signal = wrapper.predict("TEST", frame)
        regime, probabilities = wrapper.predict_regime_only("TEST", frame)
    assert signal.reason == "sota_signal"
    assert np.isfinite(signal.tcn_probability)
    assert regime in wrapper.REGIME_NAMES
    np.testing.assert_allclose(probabilities, signal.metadata["regime_probs"])
    keras.backend.clear_session()


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("temporal_split", [True, False])
def test_real_finetuning_targets_and_monitors_match_named_model_heads(
    tmp_path, frame, monkeypatch, classes, temporal_split
):
    csv = tmp_path / "candles.csv"
    frame.to_csv(csv)
    raw = _small_model(direction_classes=classes)
    trainer = SOTATrainer(
        raw,
        TrainerConfig(
            batch_size=2, finetune_epochs=1, train_split=0.5, val_split=0.25, use_temporal_split=temporal_split
        ),
    )
    captured = {}
    with tf.device("/CPU:0"):
        raw.build()
        actual_fit = raw.model.fit

        def checked_fit(x, y, **kwargs):
            captured["train_keys"] = set(y)
            captured["val_keys"] = set(kwargs["validation_data"][1])
            captured["monitors"] = [callback.monitor for callback in kwargs["callbacks"]]
            return actual_fit(x, y, **kwargs)

        monkeypatch.setattr(raw.model, "fit", checked_fit)
        result = trainer.finetune(
            [str(csv)], pretrain_weights_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "saved")
        )
    assert captured["train_keys"] == captured["val_keys"] == {"direction_5", "regime"}
    assert captured["monitors"] == ["val_direction_5_loss", "val_direction_5_loss"]
    assert {"direction_5_loss", "regime_loss", "val_direction_5_loss", "val_regime_loss"} <= set(result)
    assert all(np.isfinite(value) for value in result.values())
    assert int(raw.model.optimizer.iterations) > 0
    assert (tmp_path / "saved/sota_model.keras").is_file()
    keras.backend.clear_session()


@pytest.mark.parametrize("horizons", [(), (10,), (5, 10), (5, 5), (5.0,), [5]])
def test_five_bar_trainer_never_invents_labels_for_unsupported_horizons(tmp_path, monkeypatch, horizons):
    trainer = SOTATrainer(_small_model(direction_horizons=horizons))

    def unexpected_load(_path):
        raise AssertionError("unsupported configuration reached data loading")

    monkeypatch.setattr(trainer, "_load_candles_csv", unexpected_load)
    with pytest.raises(ValueError, match="direction_horizons"):
        trainer.finetune(["unused.csv"], save_dir=str(tmp_path / "saved"))
    assert not (tmp_path / "saved").exists()
