"""Shared helpers for native XGBoost artifact persistence.

The repository historically pickled XGBoost estimators directly, which triggers
cross-version warnings when loading newer XGBoost releases. These helpers keep
the existing top-level artifact paths (for example ``xgb_momentum.pkl``) but
store the actual XGBoost estimators in native ``.json`` sidecars instead.
"""

from __future__ import annotations

import contextlib
import io
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

XGBOOST_NATIVE_ARTIFACT_FORMAT = "xgboost_native_v1"
SERIALIZED_MODEL_WARNING = ".*serialized model.*"

DEFAULT_ALIAS_MAP: Dict[str, str] = {
    "momentum_model": "momentum",
    "accel_model": "accel",
    "direction_model": "direction",
    "confidence_model": "confidence",
    "meta_model": "model",
    "xgb_model": "xgb",
}


@contextlib.contextmanager
def _suppress_native_stderr():
    """Suppress native-library stderr output during model deserialization.

    Mythos audit 2026-05-04 — same root-cause fix as gates.py:107
    and lightgbm_trainers.py:57. The previous implementation's
    `os.dup2(devnull, sys.stderr.fileno())` corrupted file descriptors
    in the Textual TUI process (Textual replaces sys.stderr with its
    own pipe wrapper). Detect TUI context and skip OS-level dup2 —
    fall through to a Python-only `redirect_stderr(StringIO)`.
    """
    real_stderr = getattr(sys, "__stderr__", None)
    in_tui = real_stderr is None or sys.stderr is not real_stderr

    if in_tui:
        with contextlib.redirect_stderr(io.StringIO()):
            yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        with contextlib.redirect_stderr(io.StringIO()):
            yield
        return

    try:
        saved_fd = os.dup(stderr_fd)
    except OSError:
        with contextlib.redirect_stderr(io.StringIO()):
            yield
        return

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            with contextlib.redirect_stderr(devnull):
                yield
    finally:
        try:
            os.dup2(saved_fd, stderr_fd)
        except OSError:
            pass
        try:
            os.close(saved_fd)
        except OSError:
            pass


def _load_xgboost_module():
    import xgboost as xgb

    xgb.set_config(verbosity=0)
    return xgb


def _resolve_alias(model_key: str, alias_map: Optional[Mapping[str, str]] = None) -> str:
    mapping = dict(DEFAULT_ALIAS_MAP)
    if alias_map:
        mapping.update(alias_map)
    alias = mapping.get(model_key, model_key)
    alias = alias.replace(" ", "_")
    return alias


def xgboost_sidecar_path(
    base_path: str | Path,
    model_key: str,
    *,
    alias_map: Optional[Mapping[str, str]] = None,
    extension: str = ".json",
) -> Path:
    """Return the sidecar path for one native XGBoost estimator."""
    path = Path(base_path)
    stem = path.with_suffix("") if path.suffix else path
    alias = _resolve_alias(model_key, alias_map=alias_map)
    return stem.parent / f"{stem.name}.{alias}{extension}"


def is_xgboost_model(model: Any) -> bool:
    """Best-effort check for XGBoost estimator/booster instances."""
    if model is None:
        return False

    cls = type(model)
    if cls.__name__ in {"Booster", "XGBClassifier", "XGBRegressor", "XGBModel"}:
        return True

    module_name = getattr(cls, "__module__", "")
    return isinstance(module_name, str) and module_name.startswith("xgboost")


def has_native_xgboost_sidecars(
    base_path: str | Path,
    *,
    model_keys: Sequence[str],
    alias_map: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether all requested native estimator sidecars exist."""
    return all(
        xgboost_sidecar_path(base_path, key, alias_map=alias_map).exists()
        for key in model_keys
    )


def load_pickle_quietly(path: str | Path) -> Any:
    """Load a pickle while suppressing XGBoost serialization warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)
        warnings.filterwarnings("ignore", message=".*older version of XGBoost.*")
        warnings.filterwarnings("ignore", message=".*If you are loading a serialized model.*")
        with open(path, "rb") as handle:
            with _suppress_native_stderr():
                return pickle.load(handle)


def save_native_xgboost_bundle(
    base_path: str | Path,
    *,
    metadata: Mapping[str, Any],
    models: Mapping[str, Any],
    alias_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Persist XGBoost estimators as native sidecars plus pickle metadata."""
    path = Path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(metadata)
    model_specs: Dict[str, Dict[str, str]] = {}

    for model_key, model in models.items():
        if model is None:
            continue
        if not is_xgboost_model(model):
            raise TypeError(f"{model_key} is not an XGBoost model: {type(model)!r}")

        sidecar = xgboost_sidecar_path(path, model_key, alias_map=alias_map)
        # Atomic (2026-06-10): save to tmp-in-same-dir then os.replace so an
        # interrupted write can't leave a truncated sidecar for routing.
        _tmp_sidecar = sidecar.with_name(sidecar.name + ".tmp")
        model.save_model(str(_tmp_sidecar))
        os.replace(_tmp_sidecar, sidecar)
        model_specs[model_key] = {
            "file": sidecar.name,
            "class_name": type(model).__name__,
        }

    payload["artifact_format"] = XGBOOST_NATIVE_ARTIFACT_FORMAT
    payload["xgboost_models"] = model_specs

    from src.training.trainers.utils import atomic_pickle_dump
    atomic_pickle_dump(payload, path)

    return payload


def _load_native_model(path: Path, class_name: str) -> Any:
    xgb = _load_xgboost_module()
    if class_name == "Booster":
        model = xgb.Booster()
    else:
        model_cls = getattr(xgb, class_name, None)
        if model_cls is None:
            raise ValueError(f"Unsupported XGBoost class in artifact metadata: {class_name}")
        model = model_cls()
    model.load_model(str(path))
    return model


def load_xgboost_bundle(
    base_path: str | Path,
    *,
    required_models: Sequence[str] = (),
    alias_map: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Load either a native XGBoost bundle or a legacy pickle payload.

    Returns ``(payload, native_models, source)``, where ``source`` is either
    ``"native"`` or ``"legacy"``.
    """
    payload = load_pickle_quietly(base_path)
    if not isinstance(payload, dict):
        return {"legacy_payload": payload}, {}, "legacy"

    if payload.get("artifact_format") != XGBOOST_NATIVE_ARTIFACT_FORMAT:
        return payload, {}, "legacy"

    model_specs = payload.get("xgboost_models") or {}
    missing = [model_key for model_key in required_models if model_key not in model_specs]
    if missing:
        raise FileNotFoundError(
            f"Native XGBoost bundle at {base_path} is missing model specs for: {missing}"
        )

    native_models: Dict[str, Any] = {}
    for model_key, spec in model_specs.items():
        model_file = spec.get("file")
        class_name = spec.get("class_name")
        if not isinstance(model_file, str) or not isinstance(class_name, str):
            raise ValueError(f"Invalid native XGBoost model spec for {model_key}: {spec!r}")
        model_path = Path(base_path).parent / model_file
        if not model_path.exists():
            raise FileNotFoundError(
                f"Native XGBoost sidecar missing for {model_key}: {model_path}"
            )
        native_models[model_key] = _load_native_model(model_path, class_name)

    return payload, native_models, "native"
