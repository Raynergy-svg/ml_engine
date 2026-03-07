"""
Migration utilities for model format conversion.

This module contains functions to migrate models between serialization formats,
particularly for handling version compatibility issues with scikit-learn and XGBoost.

Functions:
    - migrate_xgboost_model(): Migrate XGBoost model from pickle to native format
    - migrate_all_models(): Batch migrate all models in a directory
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

from src.models.xgboost_model import XGBoostTradingModel
from src.training.trainers.utils import PRODUCTION_MODELS_DIR
from src.training.trainers.xgboost_trainer import XGBoostTrainer
from src.utils.meta_labeler_artifacts import load_meta_labeler_artifact

logger = logging.getLogger(__name__)


def migrate_xgboost_model(model_path: str, output_path: Optional[str] = None) -> bool:
    """
    Migrate XGBoost model from pickle to native format to avoid version warnings.

    XGBoost models serialized with pickle may cause warnings when loaded with
    different XGBoost versions. This function re-saves the model in the native
    XGBoost format which is more portable.

    Args:
        model_path: Path to existing pickle model
        output_path: Path for migrated model (default: same as input)

    Returns:
        True if migration successful, False otherwise
    """
    model_path = Path(model_path)
    output_path = Path(output_path) if output_path else model_path

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False

    try:
        backup_path = model_path.with_suffix(".pkl.bak")
        if model_path == output_path:
            shutil.copy(model_path, backup_path)

        if model_path.name in {"meta_labeler.pkl", "buddy_xgb_meta.pkl"}:
            model, _source = load_meta_labeler_artifact(model_path)
            model.save(output_path)
        elif model_path.name.endswith(".xgb.pkl"):
            model = XGBoostTradingModel.load(model_path)
            model.save(output_path)
        else:
            trainer = XGBoostTrainer()
            trainer.load(str(model_path))
            trainer.save(str(output_path))

        logger.info(f"✅ XGBoost model migrated to native sidecars: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def migrate_all_models(model_dir: str = PRODUCTION_MODELS_DIR) -> Dict[str, bool]:
    """
    Migrate all pickle-based models to reduce version warnings.

    Args:
        model_dir: Directory containing models

    Returns:
        Dict mapping model name to migration success status
    """
    model_dir = Path(model_dir)
    results = {}

    xgb_candidates = set()
    for pattern in ("xgb_momentum.pkl", "meta_labeler.pkl", "buddy_xgb_meta.pkl", "*.xgb.pkl"):
        xgb_candidates.update(model_dir.rglob(pattern))

    for model_path in sorted(xgb_candidates):
        try:
            rel_name = str(model_path.relative_to(model_dir))
        except ValueError:
            rel_name = str(model_path)
        results[rel_name] = migrate_xgboost_model(str(model_path))

    return results
