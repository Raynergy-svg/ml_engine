"""Shared helpers for loading meta-labeler artifacts across legacy formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

from src.utils.xgboost_artifacts import load_pickle_quietly

RETRAINER_META_KEYS = {
    "ensemble_weights",
    "ridge_model",
    "rf_model",
    "training_metrics",
    "xgb_model",
}


def is_retrainer_meta_labeler_payload(payload: Mapping[str, Any]) -> bool:
    """Whether a metadata dict matches the retrainer meta-labeler format."""
    if not isinstance(payload, Mapping):
        return False

    if RETRAINER_META_KEYS.intersection(payload.keys()):
        return True

    model_specs = payload.get("xgboost_models")
    return isinstance(model_specs, Mapping) and "xgb_model" in model_specs


def load_meta_labeler_artifact(path: str | Path) -> Tuple[Any, str]:
    """Load either MetaLabeler or MetaLabelerRetrainer artifacts."""
    artifact_path = Path(path)
    payload = load_pickle_quietly(artifact_path)

    if is_retrainer_meta_labeler_payload(payload):
        from src.training.meta_labeler_retrainer import MetaLabelerRetrainer

        return MetaLabelerRetrainer.load(artifact_path), "retrainer"

    from src.training.meta_labeling import MetaLabeler

    return MetaLabeler.load(artifact_path), "meta_labeler"
