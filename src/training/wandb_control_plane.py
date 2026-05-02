"""W&B Training Control Plane.

Single source of truth for training-head configurations. Each of the 7 ML
heads (direction, confidence, momentum, risk, volatility_regime,
trend_regime, meta_labeler) has a versioned W&B config artifact named
``<head>_training_config``.

Flow:
    1. Trainer (manual script or online_retrainer) calls ``pull_config(head)``
       at start of run.
    2. Helper fetches ``<head>_training_config:latest`` from W&B. On any
       failure (network, no auth, missing artifact, schema mismatch) it
       falls back to ``src/training/training_defaults/<head>_training_config.json``.
    3. Trainer applies the config (label gen, training behavior toggles,
       top-3 hyperparameters), runs, then calls ``log_run(head, config,
       metrics, artifacts)`` which creates a W&B run, logs metrics + config,
       optionally uploads model artifact.
    4. Operator edits the config via W&B UI / CLI between retrains; next
       retrain picks it up automatically.

Strategic configs exposed (operator-tweakable):
    label_generation:    lookahead_bars, sl_atr_mult, tp_atr_mult,
                         label_mode (real/pseudo/blend), winsorize_bounds
    training_behavior:   enable_per_pair_finetune, enable_joint_training,
                         expected_r2_band, min_real_labels_per_pair
    hyperparameters:     top-3 per model class (see defaults JSONs)

Implementation details kept hidden in code:
    - Architecture (d_model, heads, layers, kernel sizes)
    - Validation infra (walk-forward window, embargo gap, k-folds)
    - Loss / optimizer / EMA decay
    - LightGBM internals (num_leaves, reg_alpha/lambda, min_child_weight)

Graceful degradation:
    Every public function returns a sensible value on any failure rather
    than raising. Training pipeline observability MUST NOT break training.

Env vars (same conventions as ``wandb_confidence``):
    WANDB_DISABLED=1          → no-op everything.
    WANDB_API_KEY (set)       → online mode. Unset → offline (./wandb/ dir).
    WANDB_MODE                → explicit override (online|offline|disabled).
    WANDB_PROJECT             → default ``buddy-training-control-plane``.
    WANDB_ENTITY              → entity / org. Optional.
    WANDB_LOG_ARTIFACTS=1     → also upload model artifacts on log_run.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

DEFAULT_PROJECT = "buddy-training-control-plane"

# All 7 heads recognized by the control plane. Order matters only for docs.
SUPPORTED_HEADS: tuple = (
    "direction",
    "confidence",
    "momentum",
    "risk",
    "volatility_regime",
    "trend_regime",
    "meta_labeler",
)

# Schema version of locally-stored defaults. Bump if you change required keys.
CURRENT_SCHEMA_VERSION = 1

# Required top-level keys every head config must have for ``pull_config`` to
# accept it. Anything else is allowed and passed through.
REQUIRED_TOP_KEYS = (
    "schema_version",
    "head",
    "model_class",
    "label_generation",
    "training_behavior",
    "hyperparameters",
)

# Where local defaults live — used as seed and fallback.
DEFAULTS_DIR = Path(__file__).resolve().parent / "training_defaults"


# --- Env helpers -------------------------------------------------------------


def _is_disabled() -> bool:
    return os.getenv("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_mode() -> str:
    """Return ``"online"`` or ``"offline"`` based on env. Never raises."""
    explicit = os.getenv("WANDB_MODE")
    if explicit:
        return explicit
    if os.getenv("WANDB_API_KEY"):
        return "online"
    return "offline"


def _project_name() -> str:
    return os.getenv("WANDB_PROJECT", DEFAULT_PROJECT)


def _should_log_artifacts() -> bool:
    return os.getenv("WANDB_LOG_ARTIFACTS", "").strip().lower() in {"1", "true", "yes", "on"}


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


# --- Local default loading ---------------------------------------------------


def _default_path(head: str) -> Path:
    return DEFAULTS_DIR / f"{head}_training_config.json"


def load_local_default(head: str) -> Optional[Dict[str, Any]]:
    """Read the on-disk default config for ``head``. Returns None on failure."""
    path = _default_path(head)
    if not path.exists():
        logger.warning("No local default for head=%s at %s", head, path)
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to parse local default for %s: %s", head, exc)
        return None


def validate_schema(config: Mapping[str, Any]) -> bool:
    """Return True iff ``config`` has every required top-level key.

    Pure function — used by tests directly. Does NOT raise.
    """
    if not isinstance(config, Mapping):
        return False
    for key in REQUIRED_TOP_KEYS:
        if key not in config:
            return False
    # Accept any schema_version as long as it is an int — migration logic
    # lives below.
    if not isinstance(config.get("schema_version"), int):
        return False
    return True


def _migrate_if_needed(config: Dict[str, Any]) -> Dict[str, Any]:
    """Forward-migrate config dicts produced under older schemas.

    For schema_version=1 (current) this is a no-op. Future migrations live
    here as ``if version == N: ...`` blocks. Always returns a dict with
    ``schema_version == CURRENT_SCHEMA_VERSION`` on success.
    """
    cfg = deepcopy(dict(config))
    version = cfg.get("schema_version", 0)
    if version == CURRENT_SCHEMA_VERSION:
        return cfg
    # Add migrations here when CURRENT_SCHEMA_VERSION > 1.
    logger.info(
        "Migrating config from schema_version=%s to %s for head=%s",
        version,
        CURRENT_SCHEMA_VERSION,
        cfg.get("head"),
    )
    cfg["schema_version"] = CURRENT_SCHEMA_VERSION
    return cfg


# --- Public API --------------------------------------------------------------


def pull_config(head: str) -> Dict[str, Any]:
    """Fetch the latest training config for ``head``.

    Tries W&B artifact ``<head>_training_config:latest`` first, falls back to
    the on-disk default. Always returns a dict (never raises). Always
    schema-validated and migrated to ``CURRENT_SCHEMA_VERSION``.

    On total failure (no artifact AND no default file), returns ``{}``.
    """
    if head not in SUPPORTED_HEADS:
        logger.warning("pull_config: unknown head=%s (supported: %s)", head, SUPPORTED_HEADS)

    # Step 1: try W&B artifact (skip if disabled or no wandb).
    cfg: Optional[Dict[str, Any]] = None
    if not _is_disabled():
        wandb = _import_wandb()
        if wandb is not None and _resolve_mode() == "online":
            cfg = _try_pull_from_wandb(wandb, head)

    # Step 2: fall back to local default.
    if cfg is None:
        cfg = load_local_default(head)

    if cfg is None:
        logger.warning("pull_config(%s): no W&B artifact and no local default — returning empty", head)
        return {}

    if not validate_schema(cfg):
        logger.warning(
            "pull_config(%s): schema mismatch on fetched config; falling back to local default",
            head,
        )
        cfg = load_local_default(head)
        if cfg is None or not validate_schema(cfg):
            return {}

    return _migrate_if_needed(cfg)


def _try_pull_from_wandb(wandb, head: str) -> Optional[Dict[str, Any]]:
    """Best-effort artifact fetch via the public ``Api`` (no run needed)."""
    try:
        api = wandb.Api(timeout=15)
        entity = os.getenv("WANDB_ENTITY")
        project = _project_name()
        qualifier = f"{entity}/{project}" if entity else project
        artifact_name = f"{qualifier}/{head}_training_config:latest"
        artifact = api.artifact(artifact_name, type="training-config")
        with tempfile.TemporaryDirectory() as tmp:
            artifact.download(root=tmp)
            json_path = Path(tmp) / f"{head}_training_config.json"
            if not json_path.exists():
                # Some uploads may use a different filename — pick the only json.
                candidates = list(Path(tmp).glob("*.json"))
                if not candidates:
                    logger.debug("No JSON inside artifact %s", artifact_name)
                    return None
                json_path = candidates[0]
            data = json.loads(json_path.read_text())
            logger.info("Pulled %s from W&B artifact %s", head, artifact_name)
            return data
    except Exception as exc:
        logger.debug("W&B artifact pull failed for %s: %s", head, exc)
        return None


def push_config(head: str, config: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Upload a new version of ``<head>_training_config`` artifact.

    Returns dict with run/artifact info, or ``None`` if W&B is disabled or
    upload failed. Never raises.
    """
    if head not in SUPPORTED_HEADS:
        logger.warning("push_config: unknown head=%s", head)

    if not validate_schema(config):
        logger.warning("push_config(%s): schema validation failed; aborting upload", head)
        return None

    if _is_disabled():
        logger.info("W&B disabled — push_config(%s) is a no-op", head)
        return None

    wandb = _import_wandb()
    if wandb is None:
        logger.info("W&B not installed — push_config(%s) is a no-op", head)
        return None

    mode = _resolve_mode()
    project = _project_name()
    run_name = f"push_{head}_config_{_yyyymmdd()}"
    init_kwargs: Dict[str, Any] = {
        "project": project,
        "name": run_name,
        "mode": mode,
        "job_type": "config-push",
        "tags": [f"head={head}", "config-push"],
        "reinit": True,
    }
    entity = os.getenv("WANDB_ENTITY")
    if entity:
        init_kwargs["entity"] = entity

    run = None
    try:
        run = wandb.init(**init_kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / f"{head}_training_config.json"
            target.write_text(json.dumps(dict(config), indent=2, sort_keys=True))
            artifact = wandb.Artifact(
                name=f"{head}_training_config",
                type="training-config",
                description=f"Training config for {head} head",
                metadata={"head": head, "schema_version": config.get("schema_version")},
            )
            artifact.add_file(str(target))
            run.log_artifact(artifact)
        run_url = getattr(run, "url", None)
        logger.info("Pushed %s_training_config to W&B (mode=%s, url=%s)", head, mode, run_url)
        return {
            "head": head,
            "mode": mode,
            "project": project,
            "run_url": run_url,
            "run_id": getattr(run, "id", None),
        }
    except Exception as exc:
        logger.warning("push_config(%s) failed: %s", head, exc)
        return None
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass


def seed_default(head: str, defaults: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Idempotent first-time setup.

    If a ``<head>_training_config:latest`` artifact already exists in W&B,
    this is a no-op (returns the existing config). Otherwise, uploads
    ``defaults`` (or the local default if ``defaults`` is None) as v0.

    Safe to call from every training script — it will only create the
    artifact once per head.
    """
    if head not in SUPPORTED_HEADS:
        logger.warning("seed_default: unknown head=%s", head)

    if defaults is None:
        defaults = load_local_default(head)
    if defaults is None:
        logger.warning("seed_default(%s): no defaults to seed", head)
        return None

    if not validate_schema(defaults):
        logger.warning("seed_default(%s): defaults failed schema validation", head)
        return None

    if _is_disabled():
        return dict(defaults)

    wandb = _import_wandb()
    if wandb is None:
        return dict(defaults)

    # Online check: if an artifact exists, do nothing. In offline mode we
    # cannot query — fall through to push (offline pushes go to ./wandb/).
    if _resolve_mode() == "online":
        try:
            api = wandb.Api(timeout=15)
            entity = os.getenv("WANDB_ENTITY")
            project = _project_name()
            qualifier = f"{entity}/{project}" if entity else project
            api.artifact(f"{qualifier}/{head}_training_config:latest", type="training-config")
            logger.info("seed_default(%s): artifact already exists — no-op", head)
            return dict(defaults)
        except Exception:
            # Artifact doesn't exist or unreachable — proceed to upload.
            pass

    push_config(head, defaults)
    return dict(defaults)


def log_run(
    head: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Optional[List[Path]] = None,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    job_type: Optional[str] = None,
    extra_config: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Single entry point used by every trainer to log a training run.

    Args:
        head: One of ``SUPPORTED_HEADS``.
        config: The full training config dict that was used for this run
            (the one returned by ``pull_config``).
        metrics: Scalar metrics from training (val_r2, val_mae, etc.).
        artifacts: Optional list of model file paths to upload as artifact
            (only when ``WANDB_LOG_ARTIFACTS=1``).
        run_name: Override the auto-generated name.
        tags: Extra tags (the helper always adds ``head=<name>`` and a
            ``source=*`` tag — see ``extra_config['source']``).
        job_type: Override default ``training`` job_type.
        extra_config: Merged into the W&B run config alongside the training
            config. Use this to log the trigger reason, dataset stats, etc.
            Convention: include ``{"source": "manual"|"auto_retrain"}``.

    Returns dict with ``{run_id, run_url, mode, name}`` or ``None`` if W&B
    is disabled / unavailable / errored. Never raises.
    """
    if _is_disabled():
        logger.info("W&B disabled — log_run(%s) is a no-op", head)
        return None

    wandb = _import_wandb()
    if wandb is None:
        logger.info("W&B not installed — log_run(%s) is a no-op", head)
        return None

    mode = _resolve_mode()
    project = _project_name()
    name = run_name or f"{head}_{_yyyymmdd_hhmm()}"

    base_tags = [f"head={head}"]
    src_tag = (extra_config or {}).get("source")
    if src_tag:
        base_tags.append(f"source={src_tag}")
    if tags:
        base_tags.extend(tags)

    init_kwargs: Dict[str, Any] = {
        "project": project,
        "name": name,
        "mode": mode,
        "job_type": job_type or "training",
        "tags": base_tags,
        "reinit": True,
    }
    entity = os.getenv("WANDB_ENTITY")
    if entity:
        init_kwargs["entity"] = entity

    cfg: Dict[str, Any] = dict(config) if config else {}
    if extra_config:
        cfg.setdefault("_run_meta", {}).update(dict(extra_config))
    init_kwargs["config"] = cfg

    run = None
    try:
        run = wandb.init(**init_kwargs)
        # Coerce non-numeric metrics out
        clean_metrics = _coerce_metrics(metrics)
        run.log(clean_metrics)
        try:
            run.summary.update(clean_metrics)
        except Exception as exc:  # pragma: no cover
            logger.debug("summary.update failed: %s", exc)

        if _should_log_artifacts() and artifacts:
            for path in artifacts:
                try:
                    p = Path(path)
                    if not p.exists():
                        logger.debug("Artifact path missing: %s", p)
                        continue
                    art = wandb.Artifact(name=f"{head}_model_{_yyyymmdd_hhmm()}", type="model")
                    if p.is_dir():
                        art.add_dir(str(p))
                    else:
                        art.add_file(str(p))
                    run.log_artifact(art)
                except Exception as exc:  # pragma: no cover
                    logger.debug("Artifact upload failed for %s: %s", path, exc)

        run_url = getattr(run, "url", None)
        run_id = getattr(run, "id", None)
        logger.info("W&B run logged: %s (mode=%s, url=%s)", name, mode, run_url)
        return {
            "head": head,
            "run_id": run_id,
            "run_url": run_url,
            "mode": mode,
            "project": project,
            "name": name,
        }
    except Exception as exc:
        logger.warning("log_run(%s) failed: %s", head, exc)
        return None
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass


# --- Internals ---------------------------------------------------------------


def _coerce_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Pure function — coerce metric values to float, drop non-numerics."""
    out: Dict[str, float] = {}
    if not metrics:
        return out
    for k, v in metrics.items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = float(v)
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _yyyymmdd_hhmm() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# --- Convenience: trainer-side helper ----------------------------------------


def apply_config_to_trainer(trainer: Any, config: Mapping[str, Any]) -> None:
    """Best-effort: apply a control-plane config to a trainer instance.

    Walks the standard config layout and sets matching attributes on the
    trainer. Special handling:
      - Trainers expose top-3 HPs as instance attrs (LightGBM, Ridge).
      - Keras-style trainers (Transformer, TCN) read HPs from
        ``trainer.config`` (a TrainerConfig dataclass). We set the matching
        field on ``trainer.config`` if it exists.

    This is non-fatal: unknown keys are logged at DEBUG and skipped.
    """
    if not config:
        return
    label = config.get("label_generation") or {}
    behavior = config.get("training_behavior") or {}
    hp = config.get("hyperparameters") or {}

    inner_config = getattr(trainer, "config", None)

    for key, value in {**label, **behavior, **hp}.items():
        applied = False
        # 1. Try setting on the trainer instance directly (LightGBM/Ridge path).
        try:
            if hasattr(trainer, key):
                setattr(trainer, key, value)
                applied = True
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("apply_config_to_trainer: set trainer.%s failed: %s", key, exc)

        # 2. Also try setting on ``trainer.config`` (TrainerConfig dataclass path).
        if inner_config is not None:
            try:
                if hasattr(inner_config, key):
                    setattr(inner_config, key, value)
                    applied = True
                # Map ``dropout`` → ``transformer_dropout`` / ``tcn_dropout``
                # because those are the active fields on TrainerConfig.
                if key == "dropout":
                    for alias in ("transformer_dropout", "tcn_dropout"):
                        if hasattr(inner_config, alias):
                            setattr(inner_config, alias, value)
                            applied = True
            except Exception as exc:  # pragma: no cover
                logger.debug("apply_config_to_trainer: set config.%s failed: %s", key, exc)

        if not applied:
            logger.debug("apply_config_to_trainer: %s did not match any attr", key)
