"""Tests for Keras loader compatibility helpers."""

from src.utils.keras_model_loader import (
    _is_expected_compat_load_error,
    _model_prefers_tf_keras,
    _model_prefers_keras_native,
)


def test_expected_compat_error_detection():
    msg = (
        "Could not deserialize class 'Functional' because its parent module "
        "keras.src.engine.functional cannot be imported."
    )
    assert _is_expected_compat_load_error(msg) is True


def test_model_prefers_tf_keras_from_version():
    md = {"keras_version": "2.15.0", "config": None}
    assert _model_prefers_tf_keras(md) is True
    assert _model_prefers_keras_native(md) is False


def test_model_prefers_keras_native_from_version():
    md = {"keras_version": "3.4.1", "config": None}
    assert _model_prefers_keras_native(md) is True
    assert _model_prefers_tf_keras(md) is False


def test_model_prefers_tf_keras_from_config_module_path():
    md = {
        "keras_version": None,
        "config": {"module": "keras.src.engine.functional"},
    }
    assert _model_prefers_tf_keras(md) is True
    assert _model_prefers_keras_native(md) is False
