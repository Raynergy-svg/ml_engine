"""W&B Artifact logging + Model Registry access for Buddy.

Thin, safe wrapper around `wandb.log_artifact` / `wandb.use_artifact`. All calls
are env-gated by `BUDDY_WANDB_REGISTRY_ENABLED`; when off, local .pkl files in
`trained_data/models/` remain the source of truth and nothing touches W&B.

Design rules:
- Never crash the trading/training loop because of artifact logging.
- Graceful fallback to local paths when wandb missing or disabled.
- Separate from Weave init — wandb + weave share auth but initialize independently.

Layers:
1. `log_model_artifact(...)` — called after a trainer saves to disk.
2. `get_production_metrics(...)` — pulls the current production model's metrics
   for comparison against a freshly-trained candidate.
3. `promote_to_production(...)` — moves the `production` alias to a new version
   (effectively a one-line rollback-able promotion).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Env gates ────────────────────────────────────────────────────
_ENV_FLAG = "BUDDY_WANDB_REGISTRY_ENABLED"
_ENV_PROJECT = "BUDDY_WANDB_PROJECT"
_ENV_ENTITY = "BUDDY_WANDB_ENTITY"  # optional; defaults to logged-in user
_ENV_REGISTRY = "BUDDY_WANDB_REGISTRY"  # e.g. "wandb-registry-model"

_DEFAULT_PROJECT = "buddy-agents"
_DEFAULT_REGISTRY = "wandb-registry-model"


def _env_truthy(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Return True iff registry integration is env-enabled AND wandb importable."""
    if not _env_truthy(os.getenv(_ENV_FLAG)):
        return False
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        logger.warning("%s=1 but `wandb` is not installed.", _ENV_FLAG)
        return False


# ── Data shapes ──────────────────────────────────────────────────

@dataclass
class ArtifactRef:
    """Handle to a logged artifact — version id, alias list, and local path."""
    name: str            # "buddy-fx-scorer" (project-local artifact name)
    version: str         # "v12" or "latest"
    qualified: str       # "entity/project/name:v12"
    aliases: List[str] = field(default_factory=list)
    local_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# ── Public API ───────────────────────────────────────────────────

def log_model_artifact(
    name: str,
    paths: List[Path],
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    aliases: Optional[List[str]] = None,
    run_name: Optional[str] = None,
) -> Optional[ArtifactRef]:
    """Log model files as a W&B Artifact with metrics and metadata.

    Args:
        name: Artifact name (e.g. "buddy-fx-scorer", "lgbm-momentum-EUR_USD").
        paths: Files or directories to include in the artifact.
        metrics: Training/validation metrics (accuracy, auc, brier, etc.).
        metadata: Free-form metadata (feature list, training window, git sha).
        aliases: Extra aliases to apply on top of the auto-assigned "latest".
                 New candidates should typically get "staging".
        run_name: Optional W&B run name; defaults to f"train-{name}-{ts}".

    Returns:
        ArtifactRef on success, None when disabled or on any failure.
    """
    if not is_enabled():
        logger.debug("Artifact logging disabled — skipping %s", name)
        return None

    try:
        import wandb

        project = os.getenv(_ENV_PROJECT, _DEFAULT_PROJECT)
        entity = os.getenv(_ENV_ENTITY) or None
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = run_name or f"train-{name}-{ts}"

        with wandb.init(
            entity=entity,
            project=project,
            job_type="training",
            name=run_name,
            reinit=True,
            config=metadata or {},
        ) as run:
            run.log(metrics)

            art = wandb.Artifact(
                name=name,
                type="model",
                metadata={**(metadata or {}), **metrics, "logged_at": ts},
            )
            for p in paths:
                p = Path(p)
                if not p.exists():
                    logger.warning("Artifact %s: path missing, skipping: %s", name, p)
                    continue
                if p.is_dir():
                    art.add_dir(str(p))
                else:
                    art.add_file(str(p))

            final_aliases = list(aliases or [])
            logged = run.log_artifact(art, aliases=final_aliases)
            logged.wait()

            ref = ArtifactRef(
                name=name,
                version=logged.version,
                qualified=f"{entity + '/' if entity else ''}{project}/{name}:{logged.version}",
                aliases=["latest"] + final_aliases,
                metrics=dict(metrics),
            )
            logger.info(
                "Artifact logged: %s (aliases=%s, metrics=%s)",
                ref.qualified, ref.aliases, metrics,
            )
            return ref
    except Exception as exc:  # noqa: BLE001
        logger.error("log_model_artifact(%s) failed: %s", name, exc, exc_info=True)
        return None


def get_production_metrics(
    name: str,
    alias: str = "production",
) -> Optional[Dict[str, Any]]:
    """Fetch the current production model's metrics (from artifact metadata).

    Returns None if no production version exists yet (first-run case) or on
    any error — callers must treat None as "no prior production to beat".
    """
    if not is_enabled():
        return None

    try:
        import wandb
        api = wandb.Api()
        project = os.getenv(_ENV_PROJECT, _DEFAULT_PROJECT)
        entity = os.getenv(_ENV_ENTITY) or api.default_entity
        qualified = f"{entity}/{project}/{name}:{alias}"
        art = api.artifact(qualified)
        meta = dict(art.metadata or {})
        logger.info("Fetched %s metadata: %s", qualified, meta)
        return meta
    except Exception as exc:  # noqa: BLE001
        logger.info("No :%s alias for %s yet (%s) — first-run behavior", alias, name, exc)
        return None


def promote_to_production(
    ref: ArtifactRef,
    link_to_registry: bool = False,
) -> bool:
    """Move the `production` alias to this artifact version.

    When `link_to_registry=True`, also links the artifact into the org-scoped
    W&B Registry collection (cross-project governance). Otherwise alias is
    project-local, which is fine for a single-project setup.

    Returns True on success, False otherwise.
    """
    if not is_enabled():
        return False

    try:
        import wandb
        api = wandb.Api()
        project = os.getenv(_ENV_PROJECT, _DEFAULT_PROJECT)
        entity = os.getenv(_ENV_ENTITY) or api.default_entity
        qualified = f"{entity}/{project}/{ref.name}:{ref.version}"
        art = api.artifact(qualified)

        # Archive prior production: remove `production` alias from whatever had it.
        try:
            prior = api.artifact(f"{entity}/{project}/{ref.name}:production")
            if prior.version != ref.version:
                prior.aliases = [a for a in prior.aliases if a != "production"]
                prior.aliases.append("archived")
                prior.save()
                logger.info("Prior production v=%s archived", prior.version)
        except Exception as exc:  # noqa: BLE001
            logger.debug("No prior production to archive: %s", exc)

        # Apply `production` alias to new version.
        if "production" not in art.aliases:
            art.aliases.append("production")
        if "staging" in art.aliases:
            art.aliases = [a for a in art.aliases if a != "staging"]
        art.save()
        logger.info("PROMOTED %s → :production", qualified)

        if link_to_registry:
            registry = os.getenv(_ENV_REGISTRY, _DEFAULT_REGISTRY)
            collection = f"{registry}/{ref.name}"
            try:
                with wandb.init(project=project, entity=entity, job_type="promotion", reinit=True) as run:
                    run.link_artifact(art, collection, aliases=["production"])
                logger.info("Linked to registry: %s", collection)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Registry link failed (alias still set): %s", exc)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("promote_to_production failed: %s", exc, exc_info=True)
        return False


def mark_staging(ref: ArtifactRef) -> bool:
    """Tag a just-logged artifact as :staging (candidate, not yet production)."""
    if not is_enabled():
        return False
    try:
        import wandb
        api = wandb.Api()
        project = os.getenv(_ENV_PROJECT, _DEFAULT_PROJECT)
        entity = os.getenv(_ENV_ENTITY) or api.default_entity
        art = api.artifact(f"{entity}/{project}/{ref.name}:{ref.version}")
        if "staging" not in art.aliases:
            art.aliases.append("staging")
            art.save()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("mark_staging failed: %s", exc)
        return False
