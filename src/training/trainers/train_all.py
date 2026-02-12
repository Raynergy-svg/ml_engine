"""
Train all modular models in the ensemble.

This module provides the main training orchestration function that trains all
models in the ensemble (Transformer/TCN, XGBoost, RandomForest, Ridge, and
optionally HistGradientBoosting).

Functions:
    - train_all_modular(): Train all ensemble models independently
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig
from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
from src.training.trainers.transformer_regime_trainer import TransformerRegimeTrainer
from src.training.trainers.tcn_trainer import TCNTrainer
from src.training.trainers.xgboost_trainer import XGBoostTrainer
from src.training.trainers.random_forest_trainer import RandomForestTrainer
from src.training.trainers.ridge_trainer import RidgeTrainer
from src.training.trainers.histgb_trainer import HistGradientBoostingDirectionTrainer
from src.training.trainers.utils import (
    PRODUCTION_MODELS_DIR,
    TRANSFORMER_DIRECTION_FILENAME,
    RIDGE_CONFIDENCE_FILENAME,
    NO_DIRECTION_DATA_ERROR,
)

logger = logging.getLogger(__name__)


def train_all_modular(
    data: Dict[str, Dict[str, np.ndarray]],
    config: Optional[TrainerConfig] = None,
    save_dir: str = PRODUCTION_MODELS_DIR,
    use_transformer: bool = True,
    use_regime: bool = False,
    warm_start: bool = False,
    train_histgb: bool = True,
    instrument: str = "EUR_USD",
    data_range: str = "",
) -> Dict[str, BaseTrainer]:
    """
    Train all 4 models independently.

    Args:
        data: Dict from load_all_modular_data() with 'direction'/'regime', 'xgboost', 'rf', 'ridge' keys
        config: Optional trainer configuration
        save_dir: Directory to save models
        use_transformer: If True, use Transformer; if False, use TCN (only for direction mode)
        use_regime: If True, train regime classifier instead of direction predictor
        warm_start: If True, load existing model weights and continue training (compounding learning)
        train_histgb: If True, also train HistGradientBoosting for hybrid voting with Transformer
        instrument: Trading instrument name for replay buffer storage (e.g., "EUR_USD")
        data_range: Date range of training data for lineage tracking

    Returns:
        Dict with trained trainer instances
    """
    config = config or TrainerConfig()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainers = {}

    # Determine warm-start checkpoint path
    transformer_checkpoint = (
        save_dir / TRANSFORMER_DIRECTION_FILENAME if warm_start else None
    )

    # 1. Regime or Direction Model
    logger.info("\n" + "=" * 50)

    if use_regime:
        # REGIME MODE: 3-class classification (trend/chop/mean_revert)
        logger.info("Training Transformer (REGIME Classifier - 3 classes)")
        logger.info("  Classes: trend, chop, mean_revert")
        logger.info("=" * 50)

        regime_data = data.get("regime")
        if regime_data is None:
            raise ValueError(
                "No regime data found. Set use_regime=True in load_all_modular_data()"
            )

        regime_trainer = TransformerRegimeTrainer(config)
        regime_trainer.train(
            regime_data["X_train"],
            regime_data["y_train"],
            regime_data["X_val"],
            regime_data["y_val"],
            feature_names=regime_data.get("feature_names"),
            class_names=regime_data.get("class_names"),
        )
        regime_trainer.save(str(save_dir / "transformer_regime.keras"))
        trainers["regime"] = regime_trainer
        trainers["transformer"] = regime_trainer  # Alias

    else:
        # DIRECTION MODE: Binary classification (legacy)
        # Check if we should use TCN+Transformer ensemble
        use_ensemble = (
            getattr(config, "use_tcn_transformer_ensemble", True) if config else True
        )

        if use_ensemble:
            # ENSEMBLE MODE: Train BOTH TCN and Transformer for direction
            logger.info("Training TCN + Transformer Ensemble (Direction Predictors)")
            logger.info("=" * 50)

            # Get direction data (try 'direction' key first, fallback to 'tcn')
            dir_data = data.get("direction", data.get("tcn"))
            if dir_data is None:
                raise ValueError(
                    NO_DIRECTION_DATA_ERROR
                )

            # 1. Train Transformer
            logger.info("\n[1/2] Training Transformer (Direction)")
            logger.info("-" * 40)
            transformer_trainer = TransformerDirectionTrainer(config)

            if (
                warm_start
                and transformer_checkpoint
                and transformer_checkpoint.exists()
            ):
                logger.info(
                    f"🔥 WARM-START enabled: Loading weights from {transformer_checkpoint}"
                )

            transformer_trainer.train(
                dir_data["X_train"],
                dir_data["y_train"],
                dir_data["X_val"],
                dir_data["y_val"],
                feature_names=dir_data.get("feature_names"),
                w_train=dir_data.get("w_train"),
                w_val=dir_data.get("w_val"),
                warm_start_path=str(transformer_checkpoint) if warm_start else None,
                instrument=instrument,
                data_range=data_range,
            )
            transformer_trainer.save(str(save_dir / TRANSFORMER_DIRECTION_FILENAME))
            trainers["transformer"] = transformer_trainer

            # 2. Train TCN Volatility Regime Filter (replaces old direction-based TCN)
            logger.info("\n[2/2] Training TCN (Volatility Regime Filter)")
            logger.info("-" * 40)
            tcn_trainer = TCNTrainer(config)

            # Get volatility regime data from the data dict (loaded by load_all_modular_data)
            # This data uses 4-class classification: LOW/NORMAL/HIGH/EXTREME based on ATR percentile
            vol_regime_data = data.get("volatility_regime")
            if vol_regime_data is not None:
                try:
                    # Check for existing TCN model for warm-start
                    tcn_checkpoint = save_dir / "tcn_volatility_regime.keras"
                    warm_start_tcn_path = str(tcn_checkpoint) if warm_start and tcn_checkpoint.exists() else None

                    tcn_trainer.train(
                        vol_regime_data["X_train"],
                        vol_regime_data["y_train"],
                        vol_regime_data["X_val"],
                        vol_regime_data["y_val"],
                        feature_names=vol_regime_data.get("feature_names"),
                        warm_start_path=warm_start_tcn_path,
                        instrument=instrument,
                    )
                    tcn_trainer.save(str(save_dir / "tcn_volatility_regime.keras"))
                    trainers["tcn_volatility"] = tcn_trainer
                    logger.info("✓ TCN Volatility Regime model trained and saved")
                except Exception as e:
                    logger.warning(f"TCN Volatility Regime training failed: {e}")
                    logger.warning(
                        "Continuing with Transformer-only (no volatility filter)"
                    )
            else:
                logger.warning("No volatility_regime data found in data dict")
                logger.warning(
                    "Add 'volatility_regime' to load_all_modular_data or train TCN separately"
                )

            # Use transformer as primary direction model for backward compatibility
            trainers["direction"] = transformer_trainer

            logger.info("\n✓ TCN + Transformer Ensemble training complete")
            logger.info(
                f"  - Transformer saved: {save_dir / TRANSFORMER_DIRECTION_FILENAME}"
            )
            logger.info(f"  - TCN saved: {save_dir / 'tcn_direction.keras'}")

        elif use_transformer:
            logger.info("Training Transformer (Direction Predictor)")
            logger.info("=" * 50)

            # Get direction data (try 'direction' key first, fallback to 'tcn')
            dir_data = data.get("direction", data.get("tcn"))
            if dir_data is None:
                raise ValueError(
                    NO_DIRECTION_DATA_ERROR
                )

            dir_trainer = TransformerDirectionTrainer(config)

            # Log warm-start status
            if (
                warm_start
                and transformer_checkpoint
                and transformer_checkpoint.exists()
            ):
                logger.info(
                    f"🔥 WARM-START enabled: Loading weights from {transformer_checkpoint}"
                )

            dir_trainer.train(
                dir_data["X_train"],
                dir_data["y_train"],
                dir_data["X_val"],
                dir_data["y_val"],
                feature_names=dir_data.get("feature_names"),
                w_train=dir_data.get("w_train"),
                w_val=dir_data.get("w_val"),
                warm_start_path=str(transformer_checkpoint) if warm_start else None,
                instrument=instrument,
                data_range=data_range,
            )
            dir_trainer.save(str(save_dir / TRANSFORMER_DIRECTION_FILENAME))
            trainers["direction"] = dir_trainer
            trainers["transformer"] = dir_trainer  # Alias
        else:
            logger.info("Training TCN (Direction Predictor)")
            logger.info("=" * 50)

            # Get direction data
            dir_data = data.get("direction", data.get("tcn"))
            if dir_data is None:
                raise ValueError(
                    NO_DIRECTION_DATA_ERROR
                )

            dir_trainer = TCNTrainer(config)
            dir_trainer.train(
                dir_data["X_train"],
                dir_data["y_train"],
                dir_data["X_val"],
                dir_data["y_val"],
                feature_names=dir_data.get("feature_names"),
            )
            dir_trainer.save(str(save_dir / "tcn_direction.keras"))
            trainers["direction"] = dir_trainer
            trainers["tcn"] = dir_trainer  # Alias

    # 2. XGBoost
    logger.info("\n" + "=" * 50)
    logger.info("Training XGBoost (Momentum Analyzer)")
    logger.info("=" * 50)
    xgb_data = data["xgboost"]
    xgb_trainer = XGBoostTrainer(config)
    xgb_trainer.train(
        xgb_data["X_train"],
        xgb_data["y_train"],
        xgb_data["X_val"],
        xgb_data["y_val"],
        feature_names=xgb_data.get("feature_names"),
    )
    xgb_trainer.save(str(save_dir / "xgb_momentum.pkl"))
    trainers["xgboost"] = xgb_trainer

    # 3. Random Forest
    logger.info("\n" + "=" * 50)
    logger.info("Training Random Forest (Risk Assessor)")
    logger.info("=" * 50)
    rf_data = data["rf"]
    rf_trainer = RandomForestTrainer(config)
    rf_trainer.train(
        rf_data["X_train"],
        rf_data["y_train"],
        rf_data["X_val"],
        rf_data["y_val"],
        feature_names=rf_data.get("feature_names"),
    )
    rf_trainer.save(str(save_dir / "rf_risk.pkl"))
    trainers["rf"] = rf_trainer

    # 4. Ridge
    logger.info("\n" + "=" * 50)
    logger.info("Training Ridge (Confidence Scorer)")
    logger.info("=" * 50)
    ridge_data = data["ridge"]
    ridge_trainer = RidgeTrainer(config)
    ridge_trainer.train(
        ridge_data["X_train"],
        ridge_data["y_train"],
        ridge_data["X_val"],
        ridge_data["y_val"],
        feature_names=ridge_data.get("feature_names"),
    )
    ridge_trainer.save(str(save_dir / RIDGE_CONFIDENCE_FILENAME))
    trainers["ridge"] = ridge_trainer

    # 5. HistGradientBoosting (Optional - for hybrid voting)
    if train_histgb and not use_regime:
        logger.info("\n" + "=" * 50)
        logger.info(
            "Training HistGradientBoosting (Direction Baseline for Hybrid Voting)"
        )
        logger.info("=" * 50)

        dir_data = data.get("direction", data.get("tcn"))
        if dir_data is not None:
            histgb_trainer = HistGradientBoostingDirectionTrainer(config)
            histgb_trainer.train(
                dir_data["X_train"],
                dir_data["y_train"],
                dir_data["X_val"],
                dir_data["y_val"],
                feature_names=dir_data.get("feature_names"),
            )
            histgb_trainer.save(str(save_dir / "histgb_direction.pkl"))
            trainers["histgb"] = histgb_trainer
            logger.info("✓ HistGB trained for hybrid voting with Transformer")
        else:
            logger.warning("No direction data found for HistGB training")

    logger.info("\n" + "=" * 50)
    logger.info("All 4 models trained independently!")
    logger.info("=" * 50)

    return trainers
