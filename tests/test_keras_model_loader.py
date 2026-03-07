"""Tests for Keras loader compatibility helpers."""

from pathlib import Path

from src.utils.keras_model_loader import (
    _collapse_serialized_dtype_policy,
    _extract_layer_ref_name,
    _is_expected_compat_load_error,
    _model_prefers_tf_keras,
    _model_prefers_keras_native,
    _normalize_layer_config_for_compat,
    extract_model_metadata,
    load_by_rebuilding,
    load_keras_model,
    load_with_custom_deserialization,
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


def test_input_layer_compat_normalization_rewrites_batch_shape():
    normalized = _normalize_layer_config_for_compat(
        "InputLayer",
        {
            "batch_shape": [None, 60, 50],
            "dtype": "float32",
            "sparse": False,
            "ragged": False,
            "name": "features",
            "optional": False,
        },
    )

    assert normalized["input_shape"] == [60, 50]
    assert "batch_shape" not in normalized
    assert "batch_input_shape" not in normalized
    assert "optional" not in normalized


def test_serialized_dtype_policy_collapses_to_plain_name():
    collapsed = _collapse_serialized_dtype_policy(
        {
            "module": "keras",
            "class_name": "DTypePolicy",
            "config": {"name": "float32"},
            "registered_name": None,
        }
    )

    assert collapsed == "float32"


def test_batch_normalization_axis_singleton_list_collapses_to_int():
    normalized = _normalize_layer_config_for_compat(
        "BatchNormalization",
        {
            "axis": [2],
            "momentum": 0.99,
            "epsilon": 0.001,
        },
    )

    assert normalized["axis"] == 2


def test_extract_layer_ref_name_supports_flat_and_nested_refs():
    assert _extract_layer_ref_name(["features", 0, 0]) == "features"
    assert _extract_layer_ref_name([["features", 0, 0]]) == "features"


def test_extract_model_metadata_reads_metadata_json_from_keras_archive():
    metadata = extract_model_metadata(Path("trained_data/models/joint/transformer_direction.keras"))

    assert metadata["keras_version"] == "3.13.2"
    assert metadata["model_name"] == "transformer_direction"


def test_custom_deserialization_supports_transformer_arch_json():
    model, metadata = load_with_custom_deserialization(
        Path("trained_data/models/transformer_direction.arch.json")
    )

    assert model is not None
    assert metadata["success"] is True
    assert model.input_shape[1] == 60
    assert model.input_shape[2] > 0
    assert tuple(model.output_shape[1:]) == (1,)


def test_custom_deserialization_supports_tcn_arch_json():
    model, metadata = load_with_custom_deserialization(
        Path("trained_data/models/tcn_volatility_regime.arch.json")
    )

    assert model is not None
    assert metadata["success"] is True
    assert model.input_shape[1] == 60
    assert model.input_shape[2] > 0
    assert tuple(model.output_shape[1:]) == (4,)


def test_custom_deserialization_supports_transformer_keras_archive():
    model, metadata = load_with_custom_deserialization(
        Path("trained_data/models/joint/transformer_direction.keras")
    )

    assert model is not None
    assert metadata["success"] is True
    assert tuple(model.input_shape[1:]) == (60, 50)
    assert tuple(model.output_shape[1:]) == (1,)


def test_rebuild_supports_legacy_tcn_keras_archive():
    model, metadata = load_by_rebuilding(
        Path("trained_data/models/GBP_USD/tcn_volatility_regime.keras")
    )

    assert model is not None
    assert metadata["success"] is True
    assert tuple(model.input_shape[1:]) == (60, 14)
    assert tuple(model.output_shape[1:]) == (4,)


def test_load_keras_model_supports_legacy_tcn_keras_archive():
    model, metadata = load_keras_model(
        Path("trained_data/models/GBP_USD/tcn_volatility_regime.keras"),
        compile=False,
    )

    assert model is not None
    assert metadata["success"] is True
    assert tuple(model.input_shape[1:]) == (60, 14)
    assert tuple(model.output_shape[1:]) == (4,)
