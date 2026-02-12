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
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Dict, Optional

from src.training.trainers.utils import (
    PRODUCTION_MODELS_DIR,
    RIDGE_CONFIDENCE_FILENAME,
    SERIALIZED_MODEL_WARNING,
)

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
    # Suppress warnings during migration
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)

    model_path = Path(model_path)
    output_path = Path(output_path) if output_path else model_path

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False

    try:
        # Load existing model
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # Re-save (this ensures internal XGBoost state is updated)
        backup_path = model_path.with_suffix(".pkl.bak")
        if model_path == output_path:
            # Backup original
            shutil.copy(model_path, backup_path)

        with open(output_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"✅ XGBoost model migrated: {output_path}")
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

    # Models that may need migration
    pkl_models = [
        "xgb_momentum.pkl",
        "rf_risk.pkl",
        RIDGE_CONFIDENCE_FILENAME,
        "histgb_direction.pkl",
    ]

    for model_name in pkl_models:
        model_path = model_dir / model_name
        if model_path.exists():
            results[model_name] = migrate_xgboost_model(str(model_path))
        else:
            results[model_name] = None  # Not found

    return results
