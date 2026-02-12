"""Tests for per-pair model path helpers.

These tests validate pure path/utility behavior and should not require:
- OANDA credentials
- TensorFlow
- Local trained model artifacts
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("csv_path", "expected"),
    [
        ("market_data/EUR_USD_H1.csv", "EUR_USD"),
        ("market_data/oanda_GBP_USD_H1_live_20250114.csv", "GBP_USD"),
        ("market_data/USD_JPY_M5.csv", "USD_JPY"),
        ("market_data/EURUSD.csv", "EUR_USD"),
        ("market_data/aud_nzd_h4.csv", "AUD_NZD"),
        ("market_data/random_data.csv", "GENERIC"),
        (None, "GENERIC"),
        ("", "GENERIC"),
    ],
)
def test_extract_instrument_from_csv_path(csv_path: str | None, expected: str):
    from src.cli.helpers import extract_instrument_from_csv_path

    assert extract_instrument_from_csv_path(csv_path) == expected


def test_get_pair_model_paths_direction_filename():
    from src.cli.helpers import get_pair_model_paths

    model_dir = Path("trained_data/models")
    eur_paths = get_pair_model_paths(model_dir, "EUR_USD", "transformer")
    generic_paths = get_pair_model_paths(model_dir, "GENERIC", "transformer")

    assert str(eur_paths["direction"]).endswith("EUR_USD/transformer_direction.keras")
    assert str(generic_paths["direction"]).endswith("models/transformer_direction.keras")


def test_modular_inference_get_model_path_prefers_joint_for_generic():
    from src.core.modular_inference import ModularEnsembleInference

    generic_ensemble = ModularEnsembleInference(instrument=None)
    generic_path = generic_ensemble._get_model_path("transformer_direction", ".keras")

    # Current behavior: generic inference prefers joint-trained artifacts when present.
    assert str(generic_path).endswith("trained_data/models/joint/transformer_direction.keras")
