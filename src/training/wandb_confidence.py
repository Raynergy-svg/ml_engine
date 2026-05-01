"""Weights & Biases logging for the confidence training pipeline.

This is a focused W&B integration for the 2026-04-30 confidence leak-fix
follow-up. Separate from ``src/training/trainers/wandb_callback.py`` (which
serves the broader retrain orchestrator with ``BUDDY_WANDB_REGISTRY_ENABLED``)
because the confidence pipeline uses different env-var conventions per the
controller brief:

Env vars (precedence order):
    WANDB_DISABLED=1          → short-circuit ALL W&B calls (no-op).
                                Highest precedence — for ops who don't want it.
    WANDB_API_KEY (set)       → online mode (default).
    WANDB_API_KEY (unset)     → WANDB_MODE=offline (logs to ./wandb/ locally).
    WANDB_PROJECT             → project name (default: buddy-confidence-training).
    WANDB_ENTITY              → entity / org (default: logged-in user).
    WANDB_LOG_ARTIFACTS=1     → also upload model .pkl as artifact.

Run name format: ``confidence_{joint|<pair>}_{YYYYMMDD}`` so runs sort
chronologically and joint vs per-pair are visually distinguishable.

Metrics logged:
    train_r2, val_r2, train_mae, val_mae,
    label_mode, n_real, n_pseudo, class_balance_real, class_balance_pseudo,
    expected_r2_band_low, expected_r2_band_high,
    leak_detection_fired (bool: r2 > band[1]).

If anything fails (W&B unavailable, network error, schema mismatch), the
function returns silently — observability MUST NOT break training. All
exceptions are logged at DEBUG.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "buddy-confidence-training"


def _is_disabled() -> bool:
    """Top-priority kill switch."""
    return os.getenv("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_mode() -> str:
    """Return ``"online"`` or ``"offline"`` based on env. Never raises."""
    if os.getenv("WANDB_MODE"):
        return os.getenv("WANDB_MODE", "online")
    if os.getenv("WANDB_API_KEY"):
        return "online"
    return "offline"


def _import_wandb():
    """Lazy import. Returns module or None if unavailable."""
    try:
        import wandb  # type: ignore
        return wandb
    except ImportError as exc:
        logger.debug("wandb not installed: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover — protective
        logger.debug("wandb import failed: %s", exc)
        return None


def _project_name() -> str:
    return os.getenv("WANDB_PROJECT", DEFAULT_PROJECT)


def _should_log_artifacts() -> bool:
    return os.getenv("WANDB_LOG_ARTIFACTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_metric(v: Any) -> Optional[float]:
    """Coerce to float for W&B; None for non-numerics so they're filtered."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def build_metric_payload(
    metrics: Mapping[str, Any],
    label_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the metric dict that gets sent to W&B.

    Pure function — used by tests directly.
    """
    label_metadata = label_metadata or {}
    band = metrics.get("expected_r2_band") or [0.05, 0.30]
    if not isinstance(band, (list, tuple)) or len(band) < 2:
        band = [0.05, 0.30]
    r2 = metrics.get("r2_score") or 0.0
    leak_fired = False
    try:
        leak_fired = float(r2) > float(band[1])
    except (TypeError, ValueError):
        pass

    payload: Dict[str, Any] = {
        # Core training metrics
        "val_r2": _coerce_metric(metrics.get("r2_score")),
        "val_mae": _coerce_metric(metrics.get("confidence_mae")),
        # Label-mode metadata
        "n_real": _coerce_metric(label_metadata.get("n_real")
                                 or metrics.get("n_real_labels")),
        "n_pseudo": _coerce_metric(label_metadata.get("n_pseudo")
                                   or metrics.get("n_pseudo_labels")),
        "class_balance_real": _coerce_metric(
            label_metadata.get("class_balance_real")
            or metrics.get("class_balance_real")
        ),
        "class_balance_pseudo": _coerce_metric(
            label_metadata.get("class_balance_pseudo")
            or metrics.get("class_balance_pseudo")
        ),
        "class_balance_overall": _coerce_metric(metrics.get("class_balance")),
        # Leak detection
        "expected_r2_band_low": _coerce_metric(band[0]),
        "expected_r2_band_high": _coerce_metric(band[1]),
        "leak_detection_fired": float(leak_fired),
    }
    # Drop Nones (wandb tolerates them but cleaner)
    return {k: v for k, v in payload.items() if v is not None}


def _build_run_name(run_name: str) -> str:
    """Final run name: ``{run_name}_{YYYYMMDD}``."""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{run_name}_{date}"


def log_confidence_training_run(
    run_name: str,
    metrics: Mapping[str, Any],
    label_metadata: Optional[Mapping[str, Any]] = None,
    model_path: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Log a single confidence training run to W&B.

    Args:
        run_name: e.g. ``"confidence_joint"`` or ``"confidence_EUR_USD"``.
            ``_{YYYYMMDD}`` is appended automatically.
        metrics: dict from RidgeTrainer.train() return value
            (``r2_score``, ``confidence_mae``, ``expected_r2_band``, etc.).
        label_metadata: dict from compute_realized_confidence_labels
            (``n_real``, ``n_pseudo``, ``class_balance_*``, ``label_mode``).
        model_path: optional .pkl path to upload as artifact (only if
            ``WANDB_LOG_ARTIFACTS=1``).
        config: extra config to log (e.g. ``{"instruments": [...], ...}``).
        tags: list of run tags.

    Returns:
        Dict with ``{run_id, run_url, mode, payload}`` for caller logging,
        or None if disabled / unavailable.
    """
    if _is_disabled():
        logger.info("W&B disabled (WANDB_DISABLED=1) — skipping confidence run log")
        return None

    wandb = _import_wandb()
    if wandb is None:
        logger.info("W&B not installed — skipping confidence run log")
        return None

    mode = _resolve_mode()
    project = _project_name()
    final_name = _build_run_name(run_name)
    payload = build_metric_payload(metrics, label_metadata)

    init_kwargs: Dict[str, Any] = {
        "project": project,
        "name": final_name,
        "mode": mode,
        "job_type": "confidence-training",
        "tags": tags or [],
        "reinit": True,
    }
    entity = os.getenv("WANDB_ENTITY")
    if entity:
        init_kwargs["entity"] = entity

    cfg = {
        "run_name": final_name,
        "label_mode": (label_metadata or {}).get("label_mode")
                      or metrics.get("label_mode"),
        "leak_fix_version": (label_metadata or {}).get("leak_fix_version")
                            or metrics.get("leak_fix_version"),
    }
    if config:
        cfg.update(dict(config))
    init_kwargs["config"] = cfg

    run = None
    try:
        run = wandb.init(**init_kwargs)
        run.log(payload)
        # Some surfaces want metrics in summary too
        try:
            run.summary.update(payload)
        except Exception as exc:  # pragma: no cover
            logger.debug("W&B summary update failed: %s", exc)

        # Optional artifact upload
        if _should_log_artifacts() and model_path:
            mp = Path(model_path)
            if mp.exists():
                try:
                    art = wandb.Artifact(
                        name=final_name,
                        type="model",
                        metadata=dict(metrics),
                    )
                    if mp.is_dir():
                        art.add_dir(str(mp))
                    else:
                        art.add_file(str(mp))
                    run.log_artifact(art)
                except Exception as exc:  # pragma: no cover
                    logger.debug("W&B artifact log failed: %s", exc)

        run_url = getattr(run, "url", None)
        run_id = getattr(run, "id", None)
        result = {
            "run_id": run_id,
            "run_url": run_url,
            "mode": mode,
            "project": project,
            "name": final_name,
            "payload": payload,
        }
        logger.info("W&B confidence run logged: %s (mode=%s, url=%s)",
                    final_name, mode, run_url)
        return result
    except Exception as exc:
        logger.warning("W&B confidence run logging failed: %s", exc)
        return None
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
