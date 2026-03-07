"""
Meta-Labeling for Trading ML Models.

Meta-labeling is a technique that trains a secondary model to predict whether
the primary model's trade signal will be profitable. This provides:

1. Better trade filtering - only trade when meta-model is confident
2. Position sizing - scale position based on meta-model confidence
3. Improved risk management - know when NOT to trade

The key insight: We don't try to improve direction prediction (which is hard).
Instead, we learn WHEN we can trust the primary model's signal.

Flow:
    Primary Model: "Signal: LONG" (52% direction accuracy)
    Meta Model: "This signal has 70% chance of being correct" -> TRADE
    Meta Model: "This signal has 35% chance of being correct" -> SKIP

Reference: Advances in Financial Machine Learning - Marcos Lopez de Prado
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# [2026-02-14] Feature alignment fix: Import feature alignment utilities
from src.training.feature_alignment import save_feature_indices
from src.utils.xgboost_artifacts import (
    is_xgboost_model,
    load_xgboost_bundle,
    save_native_xgboost_bundle,
)

logger = logging.getLogger(__name__)

# Try to import xgboost for meta-model
try:
    import xgboost  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class MetaLabelingConfig:
    """Configuration for meta-labeling system."""

    # Meta-model parameters
    use_xgboost: bool = True  # XGBoost often outperforms DL for this
    n_estimators: int = 100  # Reduced from 200 to prevent overfitting
    max_depth: int = 2  # Reduced from 4 - shallower trees generalize better
    learning_rate: float = 0.05  # Reduced from 0.1 for smoother convergence

    # Regularization parameters (NEW - to fix overfitting)
    reg_alpha: float = 1.0  # L1 regularization
    reg_lambda: float = 5.0  # L2 regularization (strong)
    subsample: float = 0.7  # Row subsampling
    colsample_bytree: float = 0.7  # Column subsampling
    min_child_weight: int = 5  # Min samples per leaf
    early_stopping_rounds: int = 30  # Early stopping patience
    max_overfitting_gap: float = 0.10  # Reduced from 0.15 for stricter threshold
    auto_tune: bool = True  # Enable auto-tuning fallback
    dropout: float = 0.35  # Add explicit dropout control

    # Label generation
    # Updated from 0.55 to 0.52 (2024-02) to align with retrained meta-labeler
    # achieving 70.8% accuracy. Configurable via ScannerConfig.meta_labeler_threshold.
    min_confidence_threshold: float = 0.52  # Trades where meta-confidence >= this
    use_triple_barrier: bool = True  # Use TP/SL outcomes for meta-labels

    # Features for meta-model
    include_primary_proba: bool = True  # Include primary model's probability
    include_market_regime: bool = True  # Include volatility/trend regime
    include_position_features: bool = True  # Recent trade history
    use_reduced_features: bool = True  # Use only essential meta-features (recommended)


@dataclass
class MetaLabelSample:
    """A single sample for meta-labeling."""

    index: int
    features: np.ndarray
    primary_prediction: float  # Primary model's probability
    primary_correct: bool  # Whether primary model was correct
    actual_pnl: float  # Actual PnL if trade was taken


class MetaLabeler:
    """
    Meta-labeling system for trade filtering.

    Trains a secondary model to predict whether the primary model's
    signal will result in a profitable trade.
    """

    def __init__(
        self,
        config: Optional[MetaLabelingConfig] = None,
        pair_name: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        self.config = config or MetaLabelingConfig()
        self.meta_model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self._primary_accuracy: float = 0.5
        # [2026-02-14] Feature alignment fix: Store primary model's feature selection
        self.primary_selected_indices: Optional[np.ndarray] = None
        # [2026-02-14] Feature alignment fix: Store pair name and model dir for persistence
        self.pair_name = pair_name
        self.model_dir = model_dir

    def set_primary_feature_indices(self, indices: np.ndarray) -> None:
        """Store primary model's feature selection indices for alignment.
        
        [2026-02-14] Feature alignment fix: This must be called during training
        after the primary model performs feature selection, so that during inference
        the meta-labeler can apply the same feature selection before prediction.
        
        Args:
            indices: Array of feature indices selected by the primary model
        """
        self.primary_selected_indices = np.asarray(indices)
        logger.info(f"Meta-labeler: Stored {len(indices)} primary feature indices for alignment")

    def generate_meta_labels(
        self,
        features: np.ndarray,
        primary_predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        *,
        pnl_outcomes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate meta-labels from primary model predictions and outcomes.

        Args:
            features: Original features (n, features)
            primary_predictions: Primary model's probability predictions (n,)
            actual_outcomes: True outcomes (1=profitable long, 0=profitable short)
            pnl_outcomes: Optional actual PnL values

        Returns:
            Tuple of:
            - meta_features: Enhanced features for meta-model
            - meta_labels: Binary labels (1=primary was correct)
            - meta_weights: Sample weights
        """
        # [2026-02-14] Feature alignment fix: VALIDATION POINT 1 - Check input dimensions
        assert features.ndim in [2, 3], f"Features must be 2D or 3D array, got {features.ndim}D"
        assert len(features) == len(primary_predictions), (
            f"Feature/prediction count mismatch: {len(features)} vs {len(primary_predictions)}"
        )
        assert len(features) == len(actual_outcomes), (
            f"Feature/outcome count mismatch: {len(features)} vs {len(actual_outcomes)}"
        )
        
        # Primary model's binary predictions
        primary_binary = (primary_predictions >= 0.5).astype(np.float32)

        # Meta-labels: was primary model correct?
        actual_binary = (actual_outcomes >= 0.5).astype(np.float32)
        meta_labels = (primary_binary == actual_binary).astype(np.float32)

        # Track primary accuracy
        self._primary_accuracy = float(np.mean(meta_labels))
        logger.info(f"Primary model accuracy: {self._primary_accuracy:.1%}")

        # Build meta-features
        meta_feature_list = []

        # Original features (or aggregated if sequences)
        if features.ndim == 3:
            # Sequence: use last timestep only (reduced feature set)
            meta_feature_list.append(features[:, -1, :])
            # Only add aggregates if NOT using reduced features (reduces overfitting)
            if not self.config.use_reduced_features:
                meta_feature_list.append(features.mean(axis=1))
                meta_feature_list.append(features.std(axis=1))
        else:
            meta_feature_list.append(features)

        # Primary model's probability (key feature!)
        if self.config.include_primary_proba:
            meta_feature_list.append(primary_predictions.reshape(-1, 1))
            # Also add distance from 0.5 (confidence)
            meta_feature_list.append(np.abs(primary_predictions - 0.5).reshape(-1, 1))

        meta_features = np.concatenate(meta_feature_list, axis=1).astype(np.float32)

        # Weights: higher for samples where primary model was more confident
        base_weight = 1.0
        confidence = np.abs(primary_predictions - 0.5) * 2  # Scale to [0, 1]
        meta_weights = (base_weight + confidence * 0.5).astype(np.float32)

        return meta_features, meta_labels, meta_weights

    def fit(
        self,
        features: np.ndarray,
        primary_predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        *,
        features_val: Optional[np.ndarray] = None,
        predictions_val: Optional[np.ndarray] = None,
        outcomes_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train the meta-model.

        Args:
            features: Training features
            primary_predictions: Primary model's predictions on training data
            actual_outcomes: True outcomes
            features_val, predictions_val, outcomes_val: Optional validation data
            verbose: Print training progress

        Returns:
            Training metrics dict
        """
        # Generate meta-labels
        meta_X, meta_y, meta_w = self.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        if verbose:
            logger.info(f"Meta-training: {len(meta_X)} samples, {meta_X.shape[1]} features")
            logger.info(f"Meta-label balance: {np.mean(meta_y):.1%} correct")

        # Validation data
        meta_X_val, meta_y_val = None, None
        if features_val is not None and predictions_val is not None and outcomes_val is not None:
            meta_X_val, meta_y_val, _ = self.generate_meta_labels(
                features_val, predictions_val, outcomes_val
            )

        # Auto-tune hyperparameters if enabled
        if self.config.auto_tune and meta_X_val is not None and meta_y_val is not None:
            if verbose:
                logger.info("Auto-tuning meta-labeler hyperparameters...")
            best_params = self.tune_hyperparameters(meta_X, meta_y, meta_w, verbose=verbose)
            # Update config with best params
            for key, value in best_params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            if verbose:
                logger.info(f"Best params: {best_params}")

        # Train meta-model
        if self.config.use_xgboost and XGBOOST_AVAILABLE:
            metrics = self._fit_xgboost(meta_X, meta_y, meta_w, meta_X_val, meta_y_val, verbose)

            # Check for overfitting and fallback to logistic if needed
            train_acc = metrics.get("meta_train_accuracy", 0)
            val_acc = metrics.get("meta_val_accuracy", train_acc)
            gap = train_acc - val_acc

            if gap > self.config.max_overfitting_gap:
                logger.warning(
                    f"Meta-labeler overfitting detected: train={train_acc:.1%}, val={val_acc:.1%}, "
                    f"gap={gap:.1%} > {self.config.max_overfitting_gap:.1%}. "
                    f"Falling back to logistic regression."
                )
                return self._fit_logistic(meta_X, meta_y, meta_w, meta_X_val, meta_y_val, verbose)

            return metrics
        else:
            return self._fit_logistic(meta_X, meta_y, meta_w, meta_X_val, meta_y_val, verbose)

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        *,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Tune meta-model hyperparameters using TimeSeriesSplit grid search.

        Args:
            X: Meta-features
            y: Meta-labels
            w: Sample weights
            verbose: Print progress

        Returns:
            Best hyperparameters dict
        """
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        import xgboost as xgb

        # Define search grid (conservative to prevent overfitting)
        param_grid = {
            "max_depth": [2, 3],
            "reg_lambda": [3.0, 5.0, 10.0],
            "subsample": [0.6, 0.7],
            "min_child_weight": [5, 10],
        }

        # Base estimator with fixed params
        base_model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            reg_alpha=self.config.reg_alpha,
            colsample_bytree=self.config.colsample_bytree,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )

        # Use TimeSeriesSplit to avoid look-ahead bias
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1 if verbose else 0,
        )

        grid_search.fit(X, y, sample_weight=w)

        if verbose:
            logger.info(f"Grid search best score: {grid_search.best_score_:.4f}")
            logger.info(f"Grid search best params: {grid_search.best_params_}")

        return grid_search.best_params_

    def tune_threshold(
        self,
        features_val: np.ndarray,
        predictions_val: np.ndarray,
        outcomes_val: np.ndarray,
        *,
        threshold_range: Tuple[float, float] = (0.50, 0.70),
        n_steps: int = 21,
        verbose: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Tune confidence threshold on validation set to maximize filtered accuracy improvement.

        Args:
            features_val: Validation features
            predictions_val: Primary model predictions on validation
            outcomes_val: True outcomes on validation
            threshold_range: (min, max) threshold to search
            n_steps: Number of threshold values to try
            verbose: Print progress

        Returns:
            Tuple of (best_threshold, metrics_dict)
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-model must be fitted before threshold tuning")

        # Get meta-confidence
        meta_conf = self.predict_meta_confidence(features_val, predictions_val)

        # Primary model accuracy (baseline)
        primary_binary = (predictions_val >= 0.5)
        actual_binary = (outcomes_val >= 0.5)
        baseline_acc = float(np.mean(primary_binary == actual_binary))

        best_threshold = self.config.min_confidence_threshold
        best_improvement = 0.0
        best_metrics = {}

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

        for thresh in thresholds:
            trade_mask = meta_conf >= thresh
            n_trades = trade_mask.sum()

            if n_trades < 10:  # Need minimum trades for reliable estimate
                continue

            filtered_acc = float(np.mean(primary_binary[trade_mask] == actual_binary[trade_mask]))
            improvement = filtered_acc - baseline_acc
            trade_rate = n_trades / len(meta_conf)

            # Prefer threshold that maximizes improvement while keeping reasonable trade rate
            if improvement > best_improvement and trade_rate >= 0.1:
                best_improvement = improvement
                best_threshold = float(thresh)
                best_metrics = {
                    "threshold": float(thresh),
                    "baseline_accuracy": baseline_acc,
                    "filtered_accuracy": filtered_acc,
                    "improvement": improvement,
                    "trade_rate": trade_rate,
                    "n_trades": int(n_trades),
                }

        if verbose and best_metrics:
            logger.info(
                f"Optimal threshold: {best_threshold:.2f} "
                f"(acc: {best_metrics.get('baseline_accuracy', 0):.1%} -> {best_metrics.get('filtered_accuracy', 0):.1%}, "
                f"+{best_metrics.get('improvement', 0)*100:.1f}%, trades: {best_metrics.get('trade_rate', 0):.0%})"
            )

        # Update config
        self.config.min_confidence_threshold = best_threshold

        return best_threshold, best_metrics

    def _fit_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        verbose: bool,
    ) -> Dict[str, float]:
        """Train XGBoost meta-model with regularization and early stopping."""
        import xgboost as xgb

        # Build XGBClassifier with all params in constructor (avoids deprecation warning)
        xgb_params = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            # Regularization parameters
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "min_child_weight": self.config.min_child_weight,
            # Other params
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
            "n_jobs": -1,
            "random_state": 42,
        }

        # Add early stopping to constructor if validation data provided
        if X_val is not None and y_val is not None and self.config.early_stopping_rounds > 0:
            xgb_params["early_stopping_rounds"] = self.config.early_stopping_rounds

        self.meta_model = xgb.XGBClassifier(**xgb_params)

        fit_kwargs = {"sample_weight": w}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = verbose

        self.meta_model.fit(X, y, **fit_kwargs)
        self.is_fitted = True

        # Calculate metrics
        train_acc = float(np.mean((self.meta_model.predict(X) > 0.5) == y))
        metrics = {"meta_train_accuracy": train_acc}

        if X_val is not None and y_val is not None:
            val_preds = self.meta_model.predict(X_val)
            val_acc = float(np.mean((val_preds > 0.5) == y_val))
            metrics["meta_val_accuracy"] = val_acc
            metrics["overfitting_gap"] = train_acc - val_acc

            # AUC, precision, recall for richer reporting
            try:
                from sklearn.metrics import roc_auc_score, precision_score, recall_score
                val_proba = self.meta_model.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) > 1:
                    metrics["val_auc"] = float(roc_auc_score(y_val, val_proba))
                val_binary = (val_preds > 0.5).astype(int)
                metrics["val_precision"] = float(precision_score(y_val, val_binary, zero_division=0))
                metrics["val_recall"] = float(recall_score(y_val, val_binary, zero_division=0))
            except Exception:
                pass  # Non-critical; base accuracy is enough

            if verbose:
                gap = train_acc - val_acc
                gap_status = "✓" if gap <= self.config.max_overfitting_gap else "⚠️ HIGH"
                auc_str = f", auc={metrics.get('val_auc', 0):.3f}" if 'val_auc' in metrics else ""
                logger.info(f"Meta-model: train={train_acc:.1%}, val={val_acc:.1%}, gap={gap:.1%} {gap_status}{auc_str}")

        return metrics

    def _fit_logistic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        verbose: bool,
    ) -> Dict[str, float]:
        """Train logistic regression meta-model (fallback)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self.meta_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.meta_model.fit(X_scaled, y, sample_weight=w)
        self.is_fitted = True

        # Calculate metrics
        train_acc = float(np.mean(self.meta_model.predict(X_scaled) == y))
        metrics = {"meta_train_accuracy": train_acc}

        if X_val is not None and y_val is not None:
            X_val_scaled = self._scaler.transform(X_val)
            val_acc = float(np.mean(self.meta_model.predict(X_val_scaled) == y_val))
            metrics["meta_val_accuracy"] = val_acc

        return metrics

    def predict_meta_confidence(
        self,
        features: np.ndarray,
        primary_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Predict meta-confidence for primary model's signals.

        Args:
            features: Input features
            primary_predictions: Primary model's probability predictions

        Returns:
            Meta-confidence scores (probability that primary model is correct)
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-model not fitted. Call fit() first.")
        
        # [2026-02-14] Feature alignment fix: Apply stored primary feature selection
        if self.primary_selected_indices is not None:
            original_shape = features.shape
            indices = np.asarray(self.primary_selected_indices, dtype=np.int64).reshape(-1)
            if indices.size > 0:
                max_idx = int(np.max(indices))
                current_n = int(original_shape[-1])
                if max_idx < current_n:
                    if features.ndim == 2:
                        features = features[:, indices]
                    elif features.ndim == 3:
                        features = features[..., indices]
                    logger.debug(
                        f"Applied primary feature selection: {original_shape[-1]} -> {features.shape[-1]} features"
                    )
                else:
                    logger.warning(
                        "Stored primary feature indices are out of bounds for inference features "
                        "(max_idx=%d, available=%d). Skipping index-based alignment.",
                        max_idx,
                        current_n,
                    )

        # Build meta-features (same as training, but no outcomes needed)
        meta_feature_list = []

        if features.ndim == 3:
            meta_feature_list.append(features[:, -1, :])
            # Only add aggregates if NOT using reduced features
            if not self.config.use_reduced_features:
                meta_feature_list.append(features.mean(axis=1))
                meta_feature_list.append(features.std(axis=1))
        else:
            meta_feature_list.append(features)

        if self.config.include_primary_proba:
            meta_feature_list.append(primary_predictions.reshape(-1, 1))
            meta_feature_list.append(np.abs(primary_predictions - 0.5).reshape(-1, 1))

        meta_X = np.concatenate(meta_feature_list, axis=1).astype(np.float32)
        
        # [2026-02-14] Feature alignment fix: keep visibility on mismatch but do not
        # hard-fail before alignment fallback is attempted.
        if self.is_fitted and hasattr(self.meta_model, 'n_features_in_'):
            expected = self.meta_model.n_features_in_
            actual = meta_X.shape[-1]
            if actual != expected:
                logger.warning(
                    "Feature mismatch before alignment: meta-model expects %d features, got %d. "
                    "Primary indices stored: %s",
                    expected,
                    actual,
                    self.primary_selected_indices is not None,
                )

        # Handle feature shape mismatch between training and inference.
        # The meta-labeler may have been trained on a different feature set
        # (e.g., legacy pipeline with 80 features vs modular with 49).
        meta_X = self._align_features(meta_X)

        # Get probability predictions
        if hasattr(self.meta_model, "predict_proba"):
            if hasattr(self, "_scaler"):
                meta_X = self._scaler.transform(meta_X)
            proba = self.meta_model.predict_proba(meta_X)[:, 1]
        else:
            proba = self.meta_model.predict(meta_X)

        return proba.astype(np.float32)

    def _align_features(self, meta_X: np.ndarray) -> np.ndarray:
        """
        Align features to match model expectations.

        Handles feature dimension mismatches between training and inference
        by padding with zeros or truncating as needed.

        Args:
            meta_X: Meta-feature array

        Returns:
            Aligned feature array
        """
        expected_n = getattr(self.meta_model, 'n_features_in_', None)
        if expected_n is None:
            return meta_X
        
        actual_n = meta_X.shape[1]
        
        if actual_n == expected_n:
            return meta_X
        
        if actual_n < expected_n:
            # Pad with zeros to match expected feature count
            logger.warning(
                f"Meta-labeler feature mismatch: got {actual_n}, expected {expected_n}. "
                f"Padding with {expected_n - actual_n} zero features. "
                f"Retrain meta-labeler with 'buddy train' to fix permanently."
            )
            padding = np.zeros((meta_X.shape[0], expected_n - actual_n), dtype=np.float32)
            meta_X = np.concatenate([meta_X, padding], axis=1)
        else:
            # Truncate to expected feature count
            logger.warning(
                f"Meta-labeler feature mismatch: got {actual_n}, expected {expected_n}. "
                f"Truncating to first {expected_n} features. "
                f"Retrain meta-labeler with 'buddy train' to fix permanently."
            )
            meta_X = meta_X[:, :expected_n]
        
        return meta_X

    def _validate_features(self, X: np.ndarray, context: str = "training") -> np.ndarray:
        """Validate and align features for meta-labeler.

        Provides stricter validation than _align_features by raising errors
        when feature alignment cannot be performed reliably.

        Args:
            X: Input features
            context: Context string for error messages

        Returns:
            Validated/aligned features

        Raises:
            ValueError: If features cannot be aligned and no fallback is available
        """
        expected = getattr(self.meta_model, 'n_features_in_', None)
        actual = X.shape[1]

        if expected is None:
            # First training - set expected
            self.n_features_in_ = actual
            logger.info(f"Meta-labeler: Setting expected features to {actual}")
            return X

        if actual == expected:
            return X

        # Try to align using saved indices
        if hasattr(self, 'feature_indices_') and self.feature_indices_ is not None:
            if len(self.feature_indices_) <= actual:
                aligned = X[:, self.feature_indices_]
                logger.info(f"Aligned features: {actual} → {aligned.shape[1]}")
                return aligned

        # Fall back to _align_features padding/truncation
        logger.warning(
            f"Feature mismatch during {context}: got {actual} features, "
            f"expected {expected}. Using fallback alignment."
        )
        return self._align_features(X)

    def retrain_with_current_features(
        self,
        features: np.ndarray,
        primary_predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        *,
        features_val: Optional[np.ndarray] = None,
        predictions_val: Optional[np.ndarray] = None,
        outcomes_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Retrain meta-labeler with current feature pipeline.

        Use this when feature mismatch errors occur during inference.
        This resets the expected feature count and retrains the meta-model
        on the current feature set.

        Args:
            features: Current training features (n, features)
            primary_predictions: Primary model's predictions on training data
            actual_outcomes: True outcomes for training data
            features_val: Optional validation features
            predictions_val: Primary model predictions on validation
            outcomes_val: True outcomes for validation
            verbose: Print training progress

        Returns:
            Training metrics dict
        """
        logger.info("Retraining meta-labeler with current features...")

        # Generate meta-features from current data
        meta_features, meta_labels, meta_weights = self.generate_meta_labels(
            features, primary_predictions, actual_outcomes
        )

        # Reset feature expectation
        self.n_features_in_ = meta_features.shape[1]
        if verbose:
            logger.info(f"Meta-labeler retraining on {meta_features.shape[1]} features")

        # Validate dimensions
        if meta_features.shape[0] != len(meta_labels):
            raise ValueError(
                f"Meta-features and labels mismatch: "
                f"{meta_features.shape[0]} samples vs {len(meta_labels)} labels"
            )

        # Reset model to force retraining
        self.is_fitted = False
        self.meta_model = None

        # Train with new features
        return self.fit(
            features,
            primary_predictions,
            actual_outcomes,
            features_val=features_val,
            predictions_val=predictions_val,
            outcomes_val=outcomes_val,
            verbose=verbose,
        )

    def should_trade(
        self,
        features: np.ndarray,
        primary_predictions: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine which trades to take based on meta-confidence.

        Args:
            features: Input features
            primary_predictions: Primary model's predictions
            threshold: Confidence threshold (default: config value)

        Returns:
            Tuple of:
            - trade_mask: Boolean mask of trades to take
            - meta_confidence: Confidence scores
        """
        threshold = threshold or self.config.min_confidence_threshold

        meta_confidence = self.predict_meta_confidence(features, primary_predictions)
        trade_mask = meta_confidence >= threshold

        return trade_mask, meta_confidence

    def get_position_size(
        self,
        meta_confidence: np.ndarray,
        *,
        base_size: float = 1.0,
        min_size: float = 0.25,
        max_size: float = 2.0,
    ) -> np.ndarray:
        """
        Calculate position sizes based on meta-confidence.

        Higher confidence = larger position (Kelly-like sizing).

        Args:
            meta_confidence: Meta-model confidence scores
            base_size: Base position size
            min_size, max_size: Size bounds

        Returns:
            Position sizes
        """
        # Scale confidence to position size
        # confidence 0.5 -> min_size, confidence 1.0 -> max_size
        normalized = np.clip((meta_confidence - 0.5) * 2, 0, 1)
        sizes = min_size + normalized * (max_size - min_size)

        return sizes * base_size

    def save(self, path: Union[str, Path]) -> None:
        """Save meta-labeler to disk with feature indices."""
        import joblib
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "config": self.config.__dict__,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "primary_accuracy": self._primary_accuracy,
            "n_features_in_": getattr(self.meta_model, 'n_features_in_', None),
            # [2026-02-14] Feature alignment fix: Store primary model's feature selection
            "primary_selected_indices": self.primary_selected_indices,
            # [2026-02-14] Feature alignment fix: Store pair name and model dir
            "pair_name": self.pair_name,
            "model_dir": self.model_dir,
        }

        if hasattr(self, "_scaler"):
            save_dict["scaler"] = self._scaler

        native_models = {}
        if is_xgboost_model(self.meta_model):
            native_models["meta_model"] = self.meta_model
        else:
            save_dict["meta_model"] = self.meta_model

        if native_models:
            save_native_xgboost_bundle(
                path,
                metadata=save_dict,
                models=native_models,
                alias_map={"meta_model": "model"},
            )
        else:
            with open(path, "wb") as f:
                pickle.dump(save_dict, f)

        # Also save feature indices separately for easier access
        if hasattr(self.meta_model, 'n_features_in_'):
            indices_path = path.parent / "meta_feature_indices.pkl"
            joblib.dump({
                'n_features_in_': self.meta_model.n_features_in_,
                'feature_names': self.feature_names,
                # [2026-02-14] Feature alignment fix: Include primary indices
                'primary_selected_indices': self.primary_selected_indices,
            }, indices_path)
            logger.info(f"Saved meta-feature indices to {indices_path}")

        # [2026-02-14] Feature alignment fix: Save primary feature indices using feature_alignment module
        # This ensures the indices are saved in a standardized location for inference
        if self.primary_selected_indices is not None:
            model_dir = self.model_dir or str(path.parent)
            save_feature_indices(
                selected_indices=self.primary_selected_indices,
                model_dir=model_dir,
                filename="meta_primary_feature_indices.pkl",
            )
            logger.info(f"Saved primary feature indices to {model_dir}/meta_primary_feature_indices.pkl")

        logger.info(f"Saved meta-labeler to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MetaLabeler":
        """Load meta-labeler from disk."""
        save_dict, native_models, source = load_xgboost_bundle(
            path,
            alias_map={"meta_model": "model"},
        )

        config_data = save_dict.get("config", {})
        valid_config_keys = {field.name for field in fields(MetaLabelingConfig)}
        filtered_config = {
            key: value for key, value in config_data.items() if key in valid_config_keys
        }
        config = MetaLabelingConfig(**filtered_config)
        labeler = cls(config)
        labeler.meta_model = native_models.get("meta_model", save_dict.get("meta_model"))
        labeler.is_fitted = save_dict.get("is_fitted", False)
        labeler.feature_names = save_dict.get("feature_names", [])
        labeler._primary_accuracy = save_dict.get("primary_accuracy", 0.5)

        if "scaler" in save_dict:
            labeler._scaler = save_dict["scaler"]
        
        # [2026-02-14] Feature alignment fix: Restore primary feature selection indices
        labeler.primary_selected_indices = save_dict.get("primary_selected_indices")
        if labeler.primary_selected_indices is not None:
            logger.info(f"Loaded {len(labeler.primary_selected_indices)} primary feature indices from saved model")
        
        # [2026-02-14] Feature alignment fix: Restore pair name and model dir
        labeler.pair_name = save_dict.get("pair_name")
        labeler.model_dir = save_dict.get("model_dir")

        logger.debug("Loaded meta-labeler artifact from %s (%s)", path, source)
        return labeler


def train_meta_labeler(
    primary_model=None,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    *,
    primary_probs_train: np.ndarray = None,
    primary_probs_val: np.ndarray = None,
    config: Optional[MetaLabelingConfig] = None,
    verbose: bool = True,
    pair_name: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> Tuple[MetaLabeler, Dict[str, float]]:
    """
    Train a meta-labeler for an existing primary model.

    This is the main entry point for adding meta-labeling to training.
    Supports two modes:
      1. Pass primary_model → predictions are computed internally
      2. Pass primary_probs_train/primary_probs_val → skip model.predict()

    Args:
        primary_model: Trained primary model (Keras or XGBoost). Optional if probs provided.
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        primary_probs_train: Pre-computed primary model probabilities for training set
        primary_probs_val: Pre-computed primary model probabilities for validation set
        config: Meta-labeling configuration
        verbose: Print progress
        pair_name: [2026-02-14] Feature alignment fix: Pair name for feature index persistence
        model_dir: [2026-02-14] Feature alignment fix: Model directory for feature index persistence

    Returns:
        Tuple of (trained MetaLabeler, metrics dict)
    """
    if verbose:
        logger.info("Training meta-labeler...")

    # Mode 1: Pre-computed probabilities provided (preferred — avoids re-prediction)
    if primary_probs_train is not None and primary_probs_val is not None:
        preds_train = np.asarray(primary_probs_train).flatten()
        preds_val = np.asarray(primary_probs_val).flatten()
    # Mode 2: Primary model provided — compute predictions
    elif primary_model is not None and hasattr(primary_model, "predict"):
        preds_train_dict = primary_model.predict(X_train, verbose=0)
        preds_val_dict = primary_model.predict(X_val, verbose=0)

        # Handle dict output (multi-head model)
        if isinstance(preds_train_dict, dict):
            preds_train = preds_train_dict.get("direction", preds_train_dict.get("output_0"))
            preds_val = preds_val_dict.get("direction", preds_val_dict.get("output_0"))
        else:
            preds_train = preds_train_dict
            preds_val = preds_val_dict

        preds_train = np.asarray(preds_train).flatten()
        preds_val = np.asarray(preds_val).flatten()
    else:
        raise ValueError(
            "Must provide either primary_model (with .predict()) or both "
            "primary_probs_train and primary_probs_val"
        )

    # Create and train meta-labeler
    labeler = MetaLabeler(config, pair_name=pair_name, model_dir=model_dir)

    metrics = labeler.fit(
        X_train,
        preds_train,
        y_train.flatten(),
        features_val=X_val,
        predictions_val=preds_val,
        outcomes_val=y_val.flatten(),
        verbose=verbose,
    )

    if verbose:
        logger.info(f"Meta-labeler training complete: {metrics}")

    return labeler, metrics


class MetaEnsembleModel:
    """
    Combined primary + meta model for inference.

    Wraps primary model and meta-labeler for easy deployment.
    """

    def __init__(
        self,
        primary_model,
        meta_labeler: MetaLabeler,
        *,
        # Updated from 0.55 to 0.52 (2024-02) to align with retrained meta-labeler
        trade_threshold: float = 0.52,
    ):
        self.primary = primary_model
        self.meta = meta_labeler
        self.trade_threshold = trade_threshold

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions with trade filtering.

        Returns:
            Dict with:
            - direction: Direction probabilities
            - meta_confidence: Meta-model confidence
            - should_trade: Boolean trade decisions
            - position_size: Recommended position sizes
        """
        # Primary predictions
        primary_out = self.primary.predict(X, verbose=0)

        if isinstance(primary_out, dict):
            direction = np.asarray(primary_out.get("direction", primary_out.get("output_0"))).flatten()
            confidence = np.asarray(primary_out.get("confidence", direction)).flatten()
        else:
            direction = np.asarray(primary_out).flatten()
            confidence = direction

        # Meta predictions
        meta_conf = self.meta.predict_meta_confidence(X, direction)
        should_trade = meta_conf >= self.trade_threshold
        position_sizes = self.meta.get_position_size(meta_conf)

        return {
            "direction": direction,
            "confidence": confidence,
            "meta_confidence": meta_conf,
            "should_trade": should_trade,
            "position_size": position_sizes,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save ensemble model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save primary model
        primary_path = path.with_suffix(".primary.keras")
        if hasattr(self.primary, "save"):
            self.primary.save(primary_path)

        # Save meta-labeler
        meta_path = path.with_suffix(".meta.pkl")
        self.meta.save(meta_path)

        # Save config
        config_path = path.with_suffix(".ensemble.json")
        with open(config_path, "w") as f:
            json.dump({
                "primary_path": str(primary_path),
                "meta_path": str(meta_path),
                "trade_threshold": self.trade_threshold,
            }, f, indent=2)

        logger.info(f"Saved ensemble to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], primary_loader=None) -> "MetaEnsembleModel":
        """Load ensemble model."""
        path = Path(path)

        config_path = path.with_suffix(".ensemble.json")
        with open(config_path) as f:
            config = json.load(f)

        # Load primary model
        if primary_loader:
            primary = primary_loader(config["primary_path"])
        else:
            import tensorflow as tf
            primary = tf.keras.models.load_model(config["primary_path"], compile=False)

        # Load meta-labeler
        meta = MetaLabeler.load(config["meta_path"])

        return cls(
            primary,
            meta,
            # Updated from 0.55 to 0.52 (2024-02) to align with retrained meta-labeler
            trade_threshold=config.get("trade_threshold", 0.52),
        )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Meta-Labeling Test")
    print("-" * 40)

    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Simulate primary model: 55% accuracy
    true_direction = (np.random.rand(n_samples) > 0.5).astype(np.float32)
    noise = np.random.rand(n_samples) < 0.45  # 45% wrong
    primary_preds = np.where(noise, 1 - true_direction, true_direction)
    primary_preds = primary_preds + np.random.randn(n_samples) * 0.1  # Add noise
    primary_preds = np.clip(primary_preds, 0, 1).astype(np.float32)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = true_direction[:split], true_direction[split:]
    preds_train, preds_val = primary_preds[:split], primary_preds[split:]

    # Train meta-labeler
    labeler = MetaLabeler()
    metrics = labeler.fit(
        X_train,
        preds_train,
        y_train,
        features_val=X_val,
        predictions_val=preds_val,
        outcomes_val=y_val,
        verbose=True,
    )

    print(f"\nMetrics: {metrics}")

    # Test predictions
    meta_conf = labeler.predict_meta_confidence(X_val, preds_val)
    print(f"\nMeta-confidence stats: mean={np.mean(meta_conf):.2f}, std={np.std(meta_conf):.2f}")

    # Trade filtering
    trade_mask, _ = labeler.should_trade(X_val, preds_val, threshold=0.6)
    print(f"Trade filter: {np.sum(trade_mask)}/{len(trade_mask)} trades ({100*np.mean(trade_mask):.1f}%)")

    # Check if filtered trades are more accurate
    primary_binary = (preds_val >= 0.5)
    actual_binary = (y_val >= 0.5)
    all_accuracy = float(np.mean(primary_binary == actual_binary))
    filtered_accuracy = float(np.mean(primary_binary[trade_mask] == actual_binary[trade_mask]))

    print(f"\nPrimary accuracy (all): {all_accuracy:.1%}")
    print(f"Primary accuracy (filtered): {filtered_accuracy:.1%}")
    print(f"Improvement: +{100*(filtered_accuracy - all_accuracy):.1f}%")

    print("\nMeta-labeling test complete!")
