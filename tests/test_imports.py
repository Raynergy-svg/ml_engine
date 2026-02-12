"""Import smoke tests.

These tests ensure key modules can be imported after repo refactors.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_path", "items"),
    [
        # Core utilities
        ("src.utils", ["setup_logging", "load_config"]),

        # Core modules
        ("src.core.modular_data_loaders", None),
        ("src.core.modular_inference", ["ModularEnsembleInference", "InferenceConfig"]),

        # Models
        ("src.models.tensorflow_models", None),
        ("src.models.tensorflow_engine", None),
        ("src.models.ensemble_model", None),

        # Data
        ("src.data.data_processing", None),
        ("src.data.feature_engineering", None),

        # Risk
        ("src.risk.confidence_calibration", None),
        ("src.risk.fx_guardrails", None),

        # OANDA client is importable (does not require env vars for import)
        ("src.utils.oanda_practice", ["OandaPracticeClient"]),
    ],
)
def test_import_smoke(module_path: str, items: list[str] | None):
    if module_path.startswith("src.models.tensorflow") or module_path == "src.models.ensemble_model":
        pytest.importorskip("tensorflow")

    mod = importlib.import_module(module_path)
    if items:
        for item in items:
            assert hasattr(mod, item)
