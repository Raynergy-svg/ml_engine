"""
HistGradientBoosting-based trainer for direction prediction.

This module provides a fast, accurate sklearn-based baseline that serves as a
sanity check for deep learning models. HistGradientBoostingClassifier handles
NaN/missing values natively and uses histogram-based splits for speed.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig
from src.training.trainers.utils import (
    MODEL_NOT_TRAINED_ERROR,
    UNPICKLE_ESTIMATOR_WARNING,
)

logger = logging.getLogger(__name__)


class HistGradientBoostingDirectionTrainer(BaseTrainer):
    """
    HistGradientBoostingClassifier for direction prediction.

    This is a fast, accurate sklearn-based baseline that:
    - Handles NaN/missing values natively
    - Uses histogram-based splits for speed
    - Provides competitive accuracy without deep learning overhead

    Use as sanity check: If Transformer beats this by <2%, Transformer may be overfitting.
    If this baseline achieves 55-60%, the features are informative.

    Input: Flattened feature matrix (no sequences needed)
    Output: Binary direction (0=down, 1=up) with probabilities
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.pca = None
        self.feature_names = None
        self.n_features = None

        # HistGB specific config
        # NOTE 2026-05-18 (capacity-shrink for 10% gap gate):
        # Defaults below tightened from (max_iter=200, max_depth=8, lr=0.05, l2=0.1,
        # min_samples_leaf=sklearn-default-20, max_leaf_nodes=sklearn-default-31,
        # n_iter_no_change=20, validation_fraction=0.15) which produced
        # train=96.11% / val=48.82% / gap=47.29% on USD_JPY M15/25k (smoke 2026-05-13
        # post commit 983892a).
        # Goal: gap < 0.10 per .claude/rules/improvement.md "Hard Ship Gate".
        # Reverse by restoring the old getattr defaults — every change is a single
        # literal swap.
        self.max_iter = getattr(config, "histgb_max_iter", 100) if config else 100
        self.max_depth = getattr(config, "histgb_max_depth", 4) if config else 4
        self.learning_rate = (
            getattr(config, "histgb_learning_rate", 0.03) if config else 0.03
        )
        self.l2_regularization = (
            getattr(config, "histgb_l2_reg", 1.0) if config else 1.0
        )
        self.min_samples_leaf = (
            getattr(config, "histgb_min_samples_leaf", 50) if config else 50
        )
        self.max_leaf_nodes = (
            getattr(config, "histgb_max_leaf_nodes", 15) if config else 15
        )
        self.n_iter_no_change = (
            getattr(config, "histgb_n_iter_no_change", 10) if config else 10
        )
        self.validation_fraction = (
            getattr(config, "histgb_validation_fraction", 0.2) if config else 0.2
        )
        self.use_pca = getattr(config, "histgb_use_pca", True) if config else True
        self.pca_variance = (
            getattr(config, "histgb_pca_variance", 0.95) if config else 0.95
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        w_train: Optional[np.ndarray] = None,
        w_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train HistGradientBoostingClassifier for direction.

        Uses optional PCA to reduce dimensionality and prevent overfitting.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        logger.info("Training HistGradientBoosting (Direction Baseline)...")

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Flatten if 3D (sequences) - use last timestep or mean
        if X_train.ndim == 3:
            logger.info(f"Flattening 3D input: {X_train.shape} -> using last timestep")
            X_train = X_train[:, -1, :]  # Use last timestep
            x_val = x_val[:, -1, :]

        # Filter by weights if provided (keep only clear labels)
        if w_train is not None:
            clear_mask_train = w_train > 0
            clear_mask_val = (
                w_val > 0 if w_val is not None else np.ones(len(y_val), dtype=bool)
            )
            x_train_filtered = X_train[clear_mask_train]
            y_train_filtered = y_train[clear_mask_train]
            x_val_filtered = x_val[clear_mask_val]
            y_val_filtered = y_val[clear_mask_val]
            logger.info(
                f"Filtered to clear labels: train={len(x_train_filtered)}, val={len(x_val_filtered)}"
            )
        else:
            x_train_filtered = X_train
            y_train_filtered = y_train
            x_val_filtered = x_val
            y_val_filtered = y_val

        # Ensure labels are integer (sklearn classifier requires discrete labels)
        y_train_filtered = np.asarray(y_train_filtered).astype(int)
        y_val_filtered = np.asarray(y_val_filtered).astype(int)

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train_filtered)
        x_val_scaled = self.scaler.transform(x_val_filtered)

        # Optional PCA for dimensionality reduction
        if self.use_pca and x_train_scaled.shape[1] > 20:
            self.pca = PCA(n_components=self.pca_variance, random_state=42)
            x_train_pca = self.pca.fit_transform(x_train_scaled)
            x_val_pca = self.pca.transform(x_val_scaled)
            logger.info(
                f"PCA: {x_train_scaled.shape[1]} features -> {x_train_pca.shape[1]} components "
                f"(explaining {self.pca_variance * 100:.0f}% variance)"
            )
            x_train_final = x_train_pca
            x_val_final = x_val_pca
        else:
            x_train_final = x_train_scaled
            x_val_final = x_val_scaled
            self.pca = None

        # Log class distribution
        train_up_pct = (y_train_filtered == 1).mean() * 100
        val_up_pct = (y_val_filtered == 1).mean() * 100
        logger.info(
            "Class distribution: train=%.1f%% up, val=%.1f%% up",
            train_up_pct, val_up_pct
        )

        # Train HistGradientBoosting with early stopping
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=42,
            verbose=0,
        )

        self.model.fit(x_train_final, y_train_filtered)
        self.is_trained = True

        # Evaluate
        train_pred = self.model.predict(x_train_final)
        y_pred = self.model.predict(x_val_final)
        y_prob = self.model.predict_proba(x_val_final)[:, 1]

        # Metrics
        train_acc = np.mean(train_pred == y_train_filtered)
        val_acc = np.mean(y_pred == y_val_filtered)

        # Balanced accuracy
        up_mask = y_val_filtered == 1
        down_mask = y_val_filtered == 0
        up_acc = np.mean(y_pred[up_mask] == 1) if up_mask.sum() > 0 else 0
        down_acc = np.mean(y_pred[down_mask] == 0) if down_mask.sum() > 0 else 0
        balanced_acc = (up_acc + down_acc) / 2

        # AUC
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_val_filtered, y_prob)
        except Exception:
            auc = 0.5

        self.metrics = {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "val_balanced_accuracy": float(balanced_acc),
            "val_up_accuracy": float(up_acc),
            "val_down_accuracy": float(down_acc),
            "auc": float(auc),
            "n_train_samples": len(x_train_filtered),
            "n_val_samples": len(x_val_filtered),
            "n_features_used": x_train_final.shape[1],
            "n_iterations": self.model.n_iter_,
        }

        logger.info(
            f"HistGB trained: train_accuracy={train_acc:.4f}, "
            f"val_accuracy={val_acc:.4f}, balanced={balanced_acc:.4f}, "
            f"auc={auc:.4f}, iters={self.model.n_iter_}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict direction (0 or 1) with probability."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Flatten if 3D
        if X.ndim == 3:
            X = X[:, -1, :]

        # Scale
        x_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))

        # Apply PCA if used
        if self.pca is not None:
            x_final = self.pca.transform(x_scaled)
        else:
            x_final = x_scaled

        # Get last row for prediction
        x_last = x_final[-1:] if len(x_final) > 1 else x_final

        prob = float(self.model.predict_proba(x_last)[0, 1])
        direction = int(self.model.predict(x_last)[0])

        return {
            "direction": direction,
            "probability": prob,
        }

    def save(self, path: str) -> None:
        """Save HistGB model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "pca": self.pca,
            "metrics": self.metrics,
            "config": self.config.__dict__ if self.config else {},
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "model_type": "histgb",
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"HistGB saved to {path}")

    def load(self, path: str) -> None:
        """Load HistGB model."""
        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", message=UNPICKLE_ESTIMATOR_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.pca = data.get("pca")
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        logger.info(f"HistGB loaded from {path}")
