"""
Lightweight, fail-open MLflow mirror for training-session metadata.

Mirrors training-session records already logged locally (memory_client.py's
``training_sessions.json``, offline_learning_cycle.py) into a local,
file-backed MLflow tracking store (SQLite under ``trained_data/mlruns/`` by
default — no server, no cloud) so training history is browsable via
``mlflow ui --backend-store-uri sqlite:///trained_data/mlruns/mlflow.db``.

This module is pure observability. It never touches trading, execution,
halt, or config state, and it must never raise: any MLflow error is logged
and swallowed so the caller's own (already-authoritative) local logging is
unaffected. Mirrors the ``MLFLOW_AVAILABLE`` guard pattern already used in
``src/training/enterprise_training.py`` so the two stay consistent if
MLflow is ever absent from the environment.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

DEFAULT_TRACKING_DIR = Path("trained_data/mlruns")
DEFAULT_EXPERIMENT = "buddy_training_sessions"

_MAX_PARAM_KEY_LEN = 250
_MAX_PARAM_VAL_LEN = 500


def _flatten_params(prefix: str, value: Any, out: Dict[str, str]) -> None:
    """Flatten nested hyperparams into flat string params (MLflow params are strings)."""
    if isinstance(value, dict):
        for key, sub_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_params(child_prefix, sub_value, out)
        return

    if not prefix:
        return
    key = prefix[:_MAX_PARAM_KEY_LEN]
    out[key] = str(value)[:_MAX_PARAM_VAL_LEN]


def mirror_training_session(
    session: Dict[str, Any],
    *,
    tracking_dir: Optional[Path] = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> bool:
    """
    Mirror one memory_client TrainingSession dict into MLflow.

    Fail-open by design: any MLflow error (missing dependency, locked
    sqlite file, disk full, etc.) is logged at WARNING and swallowed —
    this function never raises, so it can never break the training run or
    live loop that called it.

    Returns True if a run was successfully logged, False otherwise.
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        tracking_dir = Path(tracking_dir or DEFAULT_TRACKING_DIR)
        tracking_dir.mkdir(parents=True, exist_ok=True)
        db_path = (tracking_dir / "mlflow.db").absolute()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(experiment_name)

        pair = str(session.get("pair", "unknown"))
        timestamp = str(session.get("timestamp", ""))
        run_name = f"{pair}_{timestamp}" if timestamp else pair

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("source", "memory_client")
            mlflow.set_tag("pair", pair)
            mlflow.set_tag("granularity", str(session.get("granularity", "")))
            mlflow.set_tag("timestamp", timestamp)

            mlflow.log_param("warm_start", session.get("warm_start", False))
            mlflow.log_param("duration_seconds", session.get("duration_seconds", 0.0))
            models_trained = session.get("models_trained") or []
            mlflow.log_param("models_trained", ",".join(str(m) for m in models_trained)[:_MAX_PARAM_VAL_LEN])

            flat_params: Dict[str, str] = {}
            _flatten_params("", session.get("hyperparams") or {}, flat_params)
            for key, val in list(flat_params.items())[:100]:
                mlflow.log_param(key, val)

            for metric_name, metric_val in (session.get("metrics") or {}).items():
                try:
                    mlflow.log_metric(str(metric_name), float(metric_val))
                except (TypeError, ValueError):
                    continue

        return True
    except Exception as exc:  # noqa: BLE001 - fail-open by design, see module docstring
        logger.warning("MLflow mirror failed (non-fatal, local JSON logging unaffected): %s", exc)
        return False
