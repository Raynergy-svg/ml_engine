"""SOTA model artifact versioning and rollback.

US-012: Every trained model is saved to a timestamped directory with:
  - model.keras (weights)
  - config.json (training hyperparameters)
  - lineage.json (data provenance)
  - metrics.json (validation metrics)

A symlink `trained_data/models/sota_finetuned/latest` always points to
newest version.  Rollback is a single symlink swap.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VERSION_ROOT = Path("trained_data/models/sota_versions")
LATEST_SYMLINK = Path("trained_data/models/sota_finetuned/latest")
MANIFEST_PATH = Path("trained_data/models/sota_versions/manifest.json")
DEFAULT_MAX_VERSIONS = 10


class ModelVersion:
    """Represents one versioned SOTA model artifact."""

    def __init__(self, version_dir: Path) -> None:
        self.dir = version_dir
        self.model_path = version_dir / "model.keras"
        self.config_path = version_dir / "config.json"
        self.lineage_path = version_dir / "lineage.json"
        self.metrics_path = version_dir / "metrics.json"
        self.timestamp = version_dir.name

    @property
    def exists(self) -> bool:
        return self.model_path.exists()


def save_versioned_artifact(
    model_path: Path,
    config: Dict[str, Any],
    lineage: Dict[str, Any],
    metrics: Dict[str, Any],
    *,
    version_root: Path = VERSION_ROOT,
    max_versions: int = DEFAULT_MAX_VERSIONS,
) -> Path:
    """Save a model artifact with full versioning metadata.

    Args:
        model_path: Path to the trained .keras model file.
        config: Training hyperparameters dict.
        lineage: Data provenance dict (source_data_range, n_bars, etc.).
        metrics: Validation metrics dict (val_loss, accuracy, etc.).
        version_root: Root directory for versioned artifacts.
        max_versions: Maximum number of versions to retain.

    Returns:
        Path to the new version directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_dir = version_root / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    dest_model = version_dir / "model.keras"
    shutil.copy2(str(model_path), str(dest_model))

    # Write metadata
    (version_dir / "config.json").write_text(json.dumps(config, indent=2))
    (version_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Update manifest
    _update_manifest(timestamp, config, lineage, metrics)

    # Update latest symlink (atomic on Unix)
    _update_latest_symlink(version_dir)

    # Retention: remove oldest versions beyond max_versions
    _enforce_retention(version_root, max_versions)

    logger.info("SOTA model version saved: %s", version_dir)
    return version_dir


def _update_manifest(timestamp: str, config: Dict, lineage: Dict, metrics: Dict) -> None:
    manifest: Dict[str, Any] = {"versions": []}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            # Surface corruption instead of silently dropping the entire
            # version history that rollback_sota.py --list/--to depend on.
            logger.warning(
                "Manifest at %s failed to parse (%s); starting fresh — "
                "prior version history may be lost", MANIFEST_PATH, e,
            )
    manifest["versions"].append({
        "timestamp": timestamp,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config_summary": {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))},
        "lineage_summary": lineage.get("source_data_range", "unknown"),
        "metrics_summary": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    })
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to .tmp then os.replace() so an interrupted write
    # cannot truncate/corrupt the live manifest.
    tmp_path = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    os.replace(str(tmp_path), str(MANIFEST_PATH))


def _update_latest_symlink(target_dir: Path) -> None:
    LATEST_SYMLINK.parent.mkdir(parents=True, exist_ok=True)
    if LATEST_SYMLINK.is_symlink() or LATEST_SYMLINK.exists():
        LATEST_SYMLINK.unlink()
    try:
        LATEST_SYMLINK.symlink_to(target_dir, target_is_directory=True)
    except OSError:
        # Windows may not support symlinks without privileges; copy instead
        if LATEST_SYMLINK.exists():
            shutil.rmtree(str(LATEST_SYMLINK))
        shutil.copytree(str(target_dir), str(LATEST_SYMLINK))


def _enforce_retention(version_root: Path, max_versions: int) -> None:
    # Never delete the directory that LATEST_SYMLINK currently points at —
    # after a rollback, 'latest' may target an older timestamp, and removing
    # it would leave a dangling symlink and break inference loads.
    protected: Optional[Path] = None
    try:
        if LATEST_SYMLINK.is_symlink() or LATEST_SYMLINK.exists():
            protected = LATEST_SYMLINK.resolve()
    except OSError as e:
        logger.warning("Retention: could not resolve latest symlink: %s", e)
    dirs = sorted([d for d in version_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    while len(dirs) > max_versions:
        oldest = dirs.pop(0)
        try:
            if protected is not None and oldest.resolve() == protected:
                # Skip the rollback target; retention removes the next-oldest.
                continue
        except OSError as e:
            logger.warning("Retention: could not resolve %s: %s", oldest.name, e)
        try:
            shutil.rmtree(str(oldest))
            logger.info("Retention: removed old version %s", oldest.name)
        except Exception as e:
            logger.warning("Retention: failed to remove %s: %s", oldest.name, e)


def list_versions() -> List[Dict[str, Any]]:
    """Return all version metadata from manifest."""
    if not MANIFEST_PATH.exists():
        return []
    try:
        return json.loads(MANIFEST_PATH.read_text()).get("versions", [])
    except Exception:
        return []


def rollback_to_version(timestamp: str) -> bool:
    """Rollback to a specific version by updating the latest symlink.

    Args:
        timestamp: Version timestamp string (e.g. "20260615_123000").

    Returns:
        True if rollback succeeded.
    """
    target = VERSION_ROOT / timestamp
    if not target.exists():
        logger.error("Rollback target not found: %s", timestamp)
        return False
    _update_latest_symlink(target)
    logger.info("Rollback complete: latest now points to %s", timestamp)
    return True
