"""
Joint multi-pair trainer using contrastive learning.

This module orchestrates joint training across multiple currency pairs using
instrument one-hot encoding for transfer learning. Supports pair-specific
fine-tuning with full re-optimization via LightGBM warm-start.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.training.trainers.config import TrainerConfig
from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
from src.training.trainers.lightgbm_trainers import (
    LightGBMMomentumTrainer,
    LightGBMRiskTrainer,
)
from src.training.trainers.ridge_trainer import RidgeTrainer
from src.training.trainers.utils import (
    JOINT_MODELS_DIR,
    PRODUCTION_MODELS_DIR,
    TRANSFORMER_DIRECTION_FILENAME,
    LGBM_MOMENTUM_FILENAME,
    LGBM_RISK_FILENAME,
    RIDGE_CONFIDENCE_FILENAME,
    create_sequences_with_weights,
    get_config_seq_len,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class JointMultiPairTrainer:
    """
    Orchestrates joint training across multiple currency pairs.

    Combines data from multiple pairs with instrument one-hot encoding for
    transfer learning. Supports pair-specific fine-tuning with full re-optimization.

    Models trained:
    - TransformerDirectionTrainer: Direction prediction (Keras)
    - LightGBMMomentumTrainer: Momentum analysis (LightGBM)
    - LightGBMRiskTrainer: Risk assessment (LightGBM)
    - RidgeTrainer: Confidence scoring (LightGBM)

    Save location: trained_data/models/joint/
    Fine-tuned models: trained_data/models/{instrument}/
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.instruments: List[str] = []
        self.joint_data: Optional[Dict[str, Any]] = None
        self.normalization_stats: Dict[str, Dict] = {}

        # Individual trainers
        self.direction_trainer: Optional[TransformerDirectionTrainer] = None
        self.momentum_trainer: Optional[LightGBMMomentumTrainer] = None
        self.risk_trainer: Optional[LightGBMRiskTrainer] = None
        self.confidence_trainer: Optional[RidgeTrainer] = None

        # Per-instrument results
        self.per_instrument_metrics: Dict[str, Dict[str, float]] = {}
        self.joint_metrics: Dict[str, float] = {}

        # Fine-tune decisions
        self.fine_tune_decisions: Dict[str, bool] = {}

        self.is_trained = False

    def prepare_joint_data(
        self,
        dfs: Dict[str, "pd.DataFrame"],
        instruments: List[str],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare combined data with per-pair normalization and instrument encoding.

        Args:
            dfs: Dict mapping instrument to DataFrame with features
            instruments: List of instrument names to include

        Returns:
            Dict with combined data for each model type (direction, momentum, risk, confidence)
        """
        try:
            from src.core.modular_data_loaders import (
                append_instrument_features,
                load_direction_data,
                load_xgboost_data,
                load_rf_data,
                load_ridge_data,
            )
        except ImportError as e:
            logger.error(f"Could not import data loaders: {e}")
            raise

        self.instruments = instruments
        combined_data = {
            "direction": {
                "X_train": [],
                "y_train": [],
                "w_train": [],
                "X_val": [],
                "y_val": [],
                "w_val": [],
                "feature_names": None,
            },
            "momentum": {
                "X_train": [],
                "y_train": [],
                "X_val": [],
                "y_val": [],
                "feature_names": None,
            },
            "risk": {
                "X_train": [],
                "y_train": [],
                "X_val": [],
                "y_val": [],
                "feature_names": None,
            },
            "confidence": {
                "X_train": [],
                "y_train": [],
                "X_val": [],
                "y_val": [],
                "feature_names": None,
            },
        }

        # Pair-specific sample weighting: underperforming pairs get higher weight
        # so the model is forced to learn harder cases instead of averaging away.
        # weight = 1 / max(avg_accuracy, 0.30), floored to prevent >3.3x multiplier.
        _pair_weights: Dict[str, float] = {}
        try:
            import json as _json
            _acc_path = Path("trained_data/agent_accuracy_matrix.json")
            if _acc_path.exists():
                _acc_data = _json.loads(_acc_path.read_text())
                for _pair_name, _pair_data in _acc_data.items():
                    if isinstance(_pair_data, dict):
                        _accs = [v for v in _pair_data.values()
                                if isinstance(v, (int, float)) and 0 <= v <= 1]
                        if _accs:
                            _avg = sum(_accs) / len(_accs)
                            _pair_weights[_pair_name] = 1.0 / max(_avg, 0.30)
                if _pair_weights:
                    logger.info("Pair weighting: %s", {k: f"{v:.2f}x" for k, v in _pair_weights.items()})
        except Exception as _pw_err:
            logger.debug("Pair weighting skipped: %s", _pw_err)

        for instrument in instruments:
            if instrument not in dfs:
                logger.warning(f"Skipping {instrument}: not in provided DataFrames")
                continue

            df = dfs[instrument]
            _pw = _pair_weights.get(instrument, 1.0)
            logger.info(f"Processing {instrument}: {len(df)} rows (pair_weight={_pw:.2f}x)")

            # Load each model's data (pass individual params, not config object)
            try:
                # Direction data (for Transformer)
                dir_result = load_direction_data(df)
                if dir_result:
                    # Add instrument features - pass original feature names to get combined list
                    x_train, feature_names = append_instrument_features(
                        dir_result["X_train"],
                        instrument,
                        instruments,
                        original_feature_names=dir_result.get("feature_names"),
                    )
                    x_val, _ = append_instrument_features(
                        dir_result["X_val"],
                        instrument,
                        instruments,
                        original_feature_names=dir_result.get("feature_names"),
                    )
                    combined_data["direction"]["X_train"].append(x_train)
                    combined_data["direction"]["y_train"].append(dir_result["y_train"])
                    # Apply pair-specific weight multiplier to sample weights
                    _base_w = dir_result.get("w_train", np.ones(len(dir_result["y_train"])))
                    combined_data["direction"]["w_train"].append(_base_w * _pw)
                    combined_data["direction"]["X_val"].append(x_val)
                    combined_data["direction"]["y_val"].append(dir_result["y_val"])
                    combined_data["direction"]["w_val"].append(
                        dir_result.get("w_val", np.ones(len(dir_result["y_val"])))
                    )
                    if combined_data["direction"]["feature_names"] is None:
                        combined_data["direction"]["feature_names"] = feature_names

                # Momentum data (for LightGBM)
                mom_result = load_xgboost_data(df)
                if mom_result:
                    x_train, feature_names = append_instrument_features(
                        mom_result["X_train"],
                        instrument,
                        instruments,
                        original_feature_names=mom_result.get("feature_names"),
                    )
                    x_val, _ = append_instrument_features(
                        mom_result["X_val"],
                        instrument,
                        instruments,
                        original_feature_names=mom_result.get("feature_names"),
                    )
                    combined_data["momentum"]["X_train"].append(x_train)
                    combined_data["momentum"]["y_train"].append(mom_result["y_train"])
                    combined_data["momentum"]["X_val"].append(x_val)
                    combined_data["momentum"]["y_val"].append(mom_result["y_val"])
                    if combined_data["momentum"]["feature_names"] is None:
                        combined_data["momentum"]["feature_names"] = feature_names

                # Risk data (for LightGBM)
                risk_result = load_rf_data(df)
                if risk_result:
                    x_train, feature_names = append_instrument_features(
                        risk_result["X_train"],
                        instrument,
                        instruments,
                        original_feature_names=risk_result.get("feature_names"),
                    )
                    x_val, _ = append_instrument_features(
                        risk_result["X_val"],
                        instrument,
                        instruments,
                        original_feature_names=risk_result.get("feature_names"),
                    )
                    combined_data["risk"]["X_train"].append(x_train)
                    combined_data["risk"]["y_train"].append(risk_result["y_train"])
                    combined_data["risk"]["X_val"].append(x_val)
                    combined_data["risk"]["y_val"].append(risk_result["y_val"])
                    if combined_data["risk"]["feature_names"] is None:
                        combined_data["risk"]["feature_names"] = feature_names

                # Confidence data (for Ridge/LightGBM)
                conf_result = load_ridge_data(df)
                if conf_result:
                    x_train, feature_names = append_instrument_features(
                        conf_result["X_train"],
                        instrument,
                        instruments,
                        original_feature_names=conf_result.get("feature_names"),
                    )
                    x_val, _ = append_instrument_features(
                        conf_result["X_val"],
                        instrument,
                        instruments,
                        original_feature_names=conf_result.get("feature_names"),
                    )
                    combined_data["confidence"]["X_train"].append(x_train)
                    combined_data["confidence"]["y_train"].append(
                        conf_result["y_train"]
                    )
                    combined_data["confidence"]["X_val"].append(x_val)
                    combined_data["confidence"]["y_val"].append(conf_result["y_val"])
                    if combined_data["confidence"]["feature_names"] is None:
                        combined_data["confidence"]["feature_names"] = feature_names

            except Exception as e:
                logger.error(f"Error loading data for {instrument}: {e}")
                continue

        # Concatenate all instruments
        for model_type in combined_data:
            if combined_data[model_type]["X_train"]:
                combined_data[model_type]["X_train"] = np.vstack(
                    combined_data[model_type]["X_train"]
                )
                combined_data[model_type]["y_train"] = np.concatenate(
                    combined_data[model_type]["y_train"]
                )
                combined_data[model_type]["X_val"] = np.vstack(
                    combined_data[model_type]["X_val"]
                )
                combined_data[model_type]["y_val"] = np.concatenate(
                    combined_data[model_type]["y_val"]
                )

                # Concatenate weights if present (for direction model)
                if (
                    "w_train" in combined_data[model_type]
                    and combined_data[model_type]["w_train"]
                ):
                    combined_data[model_type]["w_train"] = np.concatenate(
                        combined_data[model_type]["w_train"]
                    )
                    combined_data[model_type]["w_val"] = np.concatenate(
                        combined_data[model_type]["w_val"]
                    )

                logger.info(
                    f"Combined {model_type}: train={len(combined_data[model_type]['X_train'])}, "
                    f"val={len(combined_data[model_type]['X_val'])}"
                )

        self.joint_data = combined_data
        return combined_data

    def train(
        self,
        dfs: Dict[str, "pd.DataFrame"],
        instruments: List[str],
        save_dir: str = JOINT_MODELS_DIR,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models on joint data from multiple instruments.

        Args:
            dfs: Dict mapping instrument to DataFrame
            instruments: List of instruments to include
            save_dir: Directory to save joint models

        Returns:
            Dict with metrics for each model type
        """
        logger.info("\n" + "=" * 60)
        logger.info("JOINT MULTI-PAIR TRAINING")
        logger.info(f"Instruments: {instruments}")
        logger.info("=" * 60)

        # Prepare joint data
        data = self.prepare_joint_data(dfs, instruments)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Train Direction (Transformer)
        if (
            data["direction"]["X_train"] is not None
            and len(data["direction"]["X_train"]) > 0
        ):
            logger.info("\n" + "-" * 50)
            logger.info("Training Joint Transformer (Direction)")
            logger.info("-" * 50)

            self.direction_trainer = TransformerDirectionTrainer(self.config)
            # Use "JOINT" + instrument list for multi-pair training identification
            joint_instrument = (
                f"JOINT_{'+'.join(instruments[:3])}"
                if len(instruments) <= 3
                else f"JOINT_{len(instruments)}pairs"
            )
            metrics = self.direction_trainer.train(
                data["direction"]["X_train"],
                data["direction"]["y_train"],
                data["direction"]["X_val"],
                data["direction"]["y_val"],
                feature_names=data["direction"]["feature_names"],
                w_train=data["direction"].get(
                    "w_train"
                ),  # Pass weights to filter unclear samples
                w_val=data["direction"].get("w_val"),
                skip_scaling=True,  # Data from load_direction_data is already scaled
                instrument=joint_instrument,
            )
            self.direction_trainer.save(str(save_path / TRANSFORMER_DIRECTION_FILENAME))
            results["direction"] = metrics

        # 2. Train Momentum (LightGBM)
        if (
            data["momentum"]["X_train"] is not None
            and len(data["momentum"]["X_train"]) > 0
        ):
            logger.info("\n" + "-" * 50)
            logger.info("Training Joint LightGBM (Momentum)")
            logger.info("-" * 50)

            self.momentum_trainer = LightGBMMomentumTrainer(self.config)
            metrics = self.momentum_trainer.train(
                data["momentum"]["X_train"],
                data["momentum"]["y_train"],
                data["momentum"]["X_val"],
                data["momentum"]["y_val"],
                feature_names=data["momentum"]["feature_names"],
            )
            self.momentum_trainer.save(str(save_path / LGBM_MOMENTUM_FILENAME))
            results["momentum"] = metrics

        # 3. Train Risk (LightGBM)
        if data["risk"]["X_train"] is not None and len(data["risk"]["X_train"]) > 0:
            logger.info("\n" + "-" * 50)
            logger.info("Training Joint LightGBM (Risk)")
            logger.info("-" * 50)

            self.risk_trainer = LightGBMRiskTrainer(self.config)
            metrics = self.risk_trainer.train(
                data["risk"]["X_train"],
                data["risk"]["y_train"],
                data["risk"]["X_val"],
                data["risk"]["y_val"],
                feature_names=data["risk"]["feature_names"],
            )
            self.risk_trainer.save(str(save_path / LGBM_RISK_FILENAME))
            results["risk"] = metrics

        # 4. Train Confidence (Ridge/LightGBM)
        if (
            data["confidence"]["X_train"] is not None
            and len(data["confidence"]["X_train"]) > 0
        ):
            logger.info("\n" + "-" * 50)
            logger.info("Training Joint Ridge/LightGBM (Confidence)")
            logger.info("-" * 50)

            self.confidence_trainer = RidgeTrainer(self.config)
            metrics = self.confidence_trainer.train(
                data["confidence"]["X_train"],
                data["confidence"]["y_train"],
                data["confidence"]["X_val"],
                data["confidence"]["y_val"],
                feature_names=data["confidence"]["feature_names"],
            )
            self.confidence_trainer.save(str(save_path / RIDGE_CONFIDENCE_FILENAME))
            results["confidence"] = metrics

        self.joint_metrics = results
        self.is_trained = True

        # Save metadata
        self._save_metadata(save_path, instruments, results)

        logger.info("\n" + "=" * 60)
        logger.info("Joint training complete!")
        logger.info(f"Models saved to: {save_path}")
        logger.info("=" * 60)

        return results

    def evaluate_per_instrument(
        self,
        dfs: Dict[str, "pd.DataFrame"],
        instruments: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate joint models on each instrument separately.

        Returns accuracy per instrument to identify fine-tuning candidates.
        """
        if not self.is_trained:
            raise RuntimeError("Models not trained. Call train() first.")

        try:
            from src.core.modular_data_loaders import (
                append_instrument_features,
                load_direction_data,
            )
        except ImportError as e:
            logger.error(f"Could not import data loaders: {e}")
            raise

        results = {}

        for instrument in instruments:
            if instrument not in dfs:
                continue

            df = dfs[instrument]

            try:
                # Evaluate direction model
                dir_result = load_direction_data(df)
                if dir_result and self.direction_trainer:
                    # Get expected feature count and names from trained model
                    expected_n_features = self.direction_trainer.n_features
                    expected_feature_names = self.direction_trainer.feature_names
                    # Note: raw feature count and names from dir_result are available via:
                    # dir_result["X_val"].shape[-1] and dir_result.get("feature_names")

                    # PRIMARY FIX: If model has saved feature names, extract features directly from df
                    # Handle instrument one-hot columns separately since they won't exist in per-instrument df
                    if expected_feature_names and len(expected_feature_names) > 0:
                        # Separate base features from instrument one-hot features
                        instrument_cols = [f for f in expected_feature_names if f.startswith('instrument_')]
                        base_features = [f for f in expected_feature_names if not f.startswith('instrument_')]

                        # Check which base features exist in dataframe
                        available_base = [f for f in base_features if f in df.columns]

                        if len(available_base) == len(base_features):
                            # All base features exist - extract directly
                            # CRITICAL: Use dir_result's split indices, NOT raw df indices!
                            # dir_result already has lookahead rows dropped and proper temporal split applied
                            n_dir_result = len(dir_result["X_val"])
                            
                            # Get the validation portion from dir_result (already properly aligned)
                            # dir_result["X_val"] has the correct samples, we just need to match feature count
                            x_val_2d = dir_result["X_val"]
                            
                            # If feature count matches, use as-is
                            if x_val_2d.shape[-1] == expected_n_features:
                                logger.debug(f"{instrument}: Using dir_result X_val directly ({n_dir_result} samples, {x_val_2d.shape[-1]} features)")
                            elif x_val_2d.shape[-1] + len(instrument_cols) == expected_n_features:
                                # Need to add instrument one-hot encoding
                                n_samples = x_val_2d.shape[0]
                                instrument_onehot = np.zeros((n_samples, len(instrument_cols)), dtype=np.float32)
                                current_col = f"instrument_{instrument}"
                                if current_col in instrument_cols:
                                    col_idx = instrument_cols.index(current_col)
                                    instrument_onehot[:, col_idx] = 1.0
                                x_val_2d = np.hstack([x_val_2d, instrument_onehot])
                                logger.debug(f"{instrument}: Added {len(instrument_cols)} instrument one-hot features")
                            else:
                                # Feature mismatch - will be handled by alignment logic below
                                logger.debug(f"{instrument}: Feature count mismatch (data={x_val_2d.shape[-1]}, expected={expected_n_features})")
                        else:
                            # Some base features missing - fall through to existing logic
                            missing = set(base_features) - set(available_base)
                            logger.warning(f"{instrument}: Missing {len(missing)} base features: {list(missing)[:5]}...")
                            x_val_2d = dir_result["X_val"]
                    else:
                        # FALLBACK: Using raw dir_result without feature alignment
                        # This may indicate a model/data mismatch that should be investigated
                        logger.warning(
                            f"{instrument}: Falling back to raw dir_result features without alignment. "
                            f"Model expects {expected_n_features if expected_n_features else 'unknown'} features, "
                            f"data has {dir_result['X_val'].shape[-1]} features. "
                            f"This may cause prediction errors if feature order differs."
                        )
                        x_val_2d = dir_result["X_val"]

                    # Determine if we should add instrument features (fallback for older models)
                    # Only add them if:
                    # 1. Model expects more features than current data provides, AND
                    # 2. The difference equals the number of instrument one-hot columns
                    n_instrument_features = len(instruments)
                    should_add_instrument_features = (
                        expected_n_features is not None
                        and expected_n_features
                        == x_val_2d.shape[-1] + n_instrument_features
                    )

                    if should_add_instrument_features:
                        x_val_2d, _ = append_instrument_features(
                            x_val_2d, instrument, instruments
                        )
                        logger.debug(
                            f"{instrument}: Added instrument features, shape {x_val_2d.shape}"
                        )
                    elif expected_n_features is not None and x_val_2d.shape[-1] != expected_n_features:
                        # Feature count still doesn't match - try alignment or truncation
                        if x_val_2d.shape[-1] > expected_n_features:
                            # Too many features - truncate
                            logger.warning(
                                f"{instrument}: Feature mismatch - "
                                f"data has {x_val_2d.shape[-1]}, "
                                f"model expects {expected_n_features}. "
                                f"Using first {expected_n_features} features."
                            )
                            x_val_2d = x_val_2d[:, :expected_n_features]
                        else:
                            # Too few features - pad with zeros (better than failing)
                            n_pad = expected_n_features - x_val_2d.shape[-1]
                            logger.warning(
                                f"{instrument}: Feature mismatch - "
                                f"data has {x_val_2d.shape[-1]}, "
                                f"model expects {expected_n_features}. "
                                f"Padding with {n_pad} zero columns."
                            )
                            padding = np.zeros((x_val_2d.shape[0], n_pad), dtype=np.float32)
                            x_val_2d = np.hstack([x_val_2d, padding])

                    # Properly create sequences from 2D data
                    seq_len = get_config_seq_len(self.config)
                    x_val_seq, y_val_seq, _ = create_sequences_with_weights(
                        x_val_2d, dir_result["y_val"], dir_result.get("w_val"), seq_len
                    )

                    if len(x_val_seq) == 0:
                        logger.warning(
                            f"{instrument}: No sequences created for evaluation"
                        )
                        continue

                    # Get predictions
                    preds = self.direction_trainer.model.predict(x_val_seq)
                    y_pred = (preds > 0.5).astype(int).flatten()
                    y_true = y_val_seq[: len(y_pred)]

                    accuracy = float(np.mean(y_pred == y_true))
                    results[instrument] = {"direction_accuracy": accuracy}

                    logger.info(f"{instrument}: direction_accuracy={accuracy:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {instrument}: {e}")
                results[instrument] = {"error": str(e)}

        self.per_instrument_metrics = results
        return results

    def should_fine_tune(
        self,
        performance_threshold: float = 0.05,
    ) -> Dict[str, bool]:
        """
        Determine which pairs need fine-tuning.

        Fine-tune if accuracy is >threshold below average.

        Args:
            performance_threshold: Fine-tune if accuracy below avg - threshold

        Returns:
            Dict mapping instrument to should_fine_tune (True/False)
        """
        if not self.per_instrument_metrics:
            logger.warning(
                "No per-instrument metrics. Call evaluate_per_instrument() first."
            )
            return {}

        # Get accuracies
        accuracies = []
        for instrument, metrics in self.per_instrument_metrics.items():
            if "direction_accuracy" in metrics:
                accuracies.append((instrument, metrics["direction_accuracy"]))

        if not accuracies:
            return {}

        avg_accuracy = np.mean([a for _, a in accuracies])

        decisions = {}
        for instrument, accuracy in accuracies:
            needs_tuning = (avg_accuracy - accuracy) > performance_threshold
            decisions[instrument] = needs_tuning

            if needs_tuning:
                logger.info(
                    f"{instrument} needs fine-tuning: "
                    f"{accuracy:.2%} vs avg {avg_accuracy:.2%} "
                    f"(gap={avg_accuracy - accuracy:.2%})"
                )
            else:
                logger.info(f"{instrument} OK: {accuracy:.2%}")

        self.fine_tune_decisions = decisions
        return decisions

    def fine_tune_for_pair(
        self,
        df: "pd.DataFrame",
        instrument: str,
        save_dir: str = PRODUCTION_MODELS_DIR,
        _fine_tune_epochs: int = 10,  # Reserved for future Transformer fine-tuning
        _fine_tune_lr_factor: float = 0.1,  # Reserved for future use
    ) -> Dict[str, float]:
        """
        Fine-tune joint models for specific pair with full re-optimization.

        Uses LightGBM init_model for warm-start, allowing full re-optimization
        rather than freezing early trees.

        Args:
            df: DataFrame for the instrument
            instrument: Currency pair name
            save_dir: Base directory for saving
            fine_tune_epochs: Number of epochs for Transformer fine-tuning
            fine_tune_lr_factor: Learning rate multiplier (0.1 = 10% of original)

        Returns:
            Dict with fine-tuned metrics
        """
        if not self.is_trained:
            raise RuntimeError("Joint models not trained. Call train() first.")

        try:
            from src.core.modular_data_loaders import (
                append_instrument_features,
                load_xgboost_data,
                load_rf_data,
                load_ridge_data,
            )
        except ImportError as e:
            logger.error(f"Could not import data loaders: {e}")
            raise

        logger.info("\n" + "-" * 50)
        logger.info(f"Fine-tuning for {instrument}")
        logger.info("-" * 50)

        pair_save_dir = Path(save_dir) / instrument
        pair_save_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Fine-tune Momentum (LightGBM with init_model)
        if self.momentum_trainer and self.momentum_trainer.is_trained:
            try:
                mom_result = load_xgboost_data(df)
                if mom_result:
                    x_train, feature_names = append_instrument_features(
                        mom_result["X_train"], instrument, self.instruments
                    )
                    x_val, _ = append_instrument_features(
                        mom_result["X_val"], instrument, self.instruments
                    )

                    # Create new trainer with warm-start
                    fine_tuned_momentum = LightGBMMomentumTrainer(self.config)
                    metrics = fine_tuned_momentum.train(
                        x_train,
                        mom_result["y_train"],
                        x_val,
                        mom_result["y_val"],
                        feature_names=feature_names,
                        init_momentum_model=self.momentum_trainer.get_booster(
                            "momentum"
                        ),
                        init_accel_model=self.momentum_trainer.get_booster("accel"),
                    )
                    fine_tuned_momentum.save(str(pair_save_dir / LGBM_MOMENTUM_FILENAME))
                    results["momentum"] = metrics
                    logger.info(f"  Momentum fine-tuned: {metrics}")
            except Exception as e:
                logger.error(f"  Momentum fine-tune failed: {e}")

        # Fine-tune Risk (LightGBM with init_model)
        if self.risk_trainer and self.risk_trainer.is_trained:
            try:
                risk_result = load_rf_data(df)
                if risk_result:
                    x_train, feature_names = append_instrument_features(
                        risk_result["X_train"], instrument, self.instruments
                    )
                    x_val, _ = append_instrument_features(
                        risk_result["X_val"], instrument, self.instruments
                    )

                    fine_tuned_risk = LightGBMRiskTrainer(self.config)
                    metrics = fine_tuned_risk.train(
                        x_train,
                        risk_result["y_train"],
                        x_val,
                        risk_result["y_val"],
                        feature_names=feature_names,
                        init_drawdown_model=self.risk_trainer.get_booster("drawdown"),
                        init_streak_model=self.risk_trainer.get_booster("streak"),
                    )
                    fine_tuned_risk.save(str(pair_save_dir / LGBM_RISK_FILENAME))
                    results["risk"] = metrics
                    logger.info(f"  Risk fine-tuned: {metrics}")
            except Exception as e:
                logger.error(f"  Risk fine-tune failed: {e}")

        # Fine-tune Confidence (Ridge/LightGBM)
        if self.confidence_trainer and self.confidence_trainer.is_trained:
            try:
                conf_result = load_ridge_data(df)
                if conf_result:
                    x_train, feature_names = append_instrument_features(
                        conf_result["X_train"], instrument, self.instruments
                    )
                    x_val, _ = append_instrument_features(
                        conf_result["X_val"], instrument, self.instruments
                    )

                    fine_tuned_conf = RidgeTrainer(self.config)
                    metrics = fine_tuned_conf.train(
                        x_train,
                        conf_result["y_train"],
                        x_val,
                        conf_result["y_val"],
                        feature_names=feature_names,
                    )
                    fine_tuned_conf.save(str(pair_save_dir / RIDGE_CONFIDENCE_FILENAME))
                    results["confidence"] = metrics
                    logger.info(f"  Confidence fine-tuned: {metrics}")
            except Exception as e:
                logger.error(f"  Confidence fine-tune failed: {e}")

        logger.info(f"Fine-tuned models saved to: {pair_save_dir}")
        return results

    def _save_metadata(
        self,
        save_path: Path,
        instruments: List[str],
        results: Dict[str, Dict],
    ) -> None:
        """Save joint training metadata."""
        meta = {
            "training_type": "joint_multi_pair",
            "instruments": instruments,
            "metrics": results,
            "per_instrument_metrics": self.per_instrument_metrics,
            "fine_tune_decisions": self.fine_tune_decisions,
            "saved_at": datetime.now().isoformat(),
        }

        meta_path = save_path / "joint_training_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Metadata saved to: {meta_path}")

    def save(self, save_dir: str = JOINT_MODELS_DIR) -> Dict[str, Path]:
        """Save all joint models."""
        if not self.is_trained:
            raise RuntimeError("Models not trained")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        if self.direction_trainer:
            p = save_path / TRANSFORMER_DIRECTION_FILENAME
            self.direction_trainer.save(str(p))
            paths["direction"] = p

        if self.momentum_trainer:
            p = save_path / LGBM_MOMENTUM_FILENAME
            self.momentum_trainer.save(str(p))
            paths["momentum"] = p

        if self.risk_trainer:
            p = save_path / LGBM_RISK_FILENAME
            self.risk_trainer.save(str(p))
            paths["risk"] = p

        if self.confidence_trainer:
            p = save_path / RIDGE_CONFIDENCE_FILENAME
            self.confidence_trainer.save(str(p))
            paths["confidence"] = p

        return paths

    def load(self, load_dir: str = JOINT_MODELS_DIR) -> bool:
        """Load all joint models."""
        load_path = Path(load_dir)

        if not load_path.exists():
            logger.warning(f"Joint model directory not found: {load_path}")
            return False

        loaded = False

        # Direction (Transformer)
        dir_path = load_path / TRANSFORMER_DIRECTION_FILENAME
        if dir_path.exists():
            self.direction_trainer = TransformerDirectionTrainer(self.config)
            self.direction_trainer.load(str(dir_path))
            loaded = True

        # Momentum (LightGBM)
        mom_path = load_path / LGBM_MOMENTUM_FILENAME
        if mom_path.exists():
            self.momentum_trainer = LightGBMMomentumTrainer(self.config)
            self.momentum_trainer.load(str(mom_path))
            loaded = True

        # Risk (LightGBM)
        risk_path = load_path / LGBM_RISK_FILENAME
        if risk_path.exists():
            self.risk_trainer = LightGBMRiskTrainer(self.config)
            self.risk_trainer.load(str(risk_path))
            loaded = True

        # Confidence (Ridge)
        conf_path = load_path / RIDGE_CONFIDENCE_FILENAME
        if conf_path.exists():
            self.confidence_trainer = RidgeTrainer(self.config)
            self.confidence_trainer.load(str(conf_path))
            loaded = True

        self.is_trained = loaded
        return loaded

    def warm_start_from_master(
        self,
        master_model_path: str,
        target_instrument: str,
    ) -> bool:
        """
        Initialize trainers from a master model for transfer learning.

        Loads the master model and prepares it for fine-tuning on a target
        instrument. The direction trainer is marked for warm-start which
        enables reduced learning rate and layer freezing during training.

        Args:
            master_model_path: Path to master model directory
            target_instrument: Target instrument for fine-tuning

        Returns:
            True if master model was loaded successfully
        """
        master_path = Path(master_model_path)

        if not master_path.exists():
            logger.warning(f"Master model path not found: {master_path}")
            return False

        loaded = False

        # Load direction trainer with warm-start flag
        direction_path = master_path / TRANSFORMER_DIRECTION_FILENAME
        if direction_path.exists():
            self.direction_trainer = TransformerDirectionTrainer(self.config)
            self.direction_trainer.load(str(direction_path))
            # Mark as warm-start for transfer learning
            self.direction_trainer._is_warm_start = True
            logger.info(
                f"Loaded master direction model from {direction_path} "
                f"for transfer to {target_instrument}"
            )
            loaded = True

        # Load LightGBM models (they support init_model for warm-start)
        mom_path = master_path / LGBM_MOMENTUM_FILENAME
        if mom_path.exists():
            self.momentum_trainer = LightGBMMomentumTrainer(self.config)
            self.momentum_trainer.load(str(mom_path))
            logger.info(f"Loaded master momentum model from {mom_path}")
            loaded = True

        risk_path = master_path / LGBM_RISK_FILENAME
        if risk_path.exists():
            self.risk_trainer = LightGBMRiskTrainer(self.config)
            self.risk_trainer.load(str(risk_path))
            logger.info(f"Loaded master risk model from {risk_path}")
            loaded = True

        conf_path = master_path / RIDGE_CONFIDENCE_FILENAME
        if conf_path.exists():
            self.confidence_trainer = RidgeTrainer(self.config)
            self.confidence_trainer.load(str(conf_path))
            logger.info(f"Loaded master confidence model from {conf_path}")
            loaded = True

        self.is_trained = loaded
        return loaded
