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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import xgboost for meta-model
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class MetaLabelingConfig:
    """Configuration for meta-labeling system."""
    
    # Meta-model parameters
    use_xgboost: bool = True  # XGBoost often outperforms DL for this
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    
    # Label generation
    min_confidence_threshold: float = 0.55  # Trades where meta-confidence >= this
    use_triple_barrier: bool = True  # Use TP/SL outcomes for meta-labels
    
    # Features for meta-model
    include_primary_proba: bool = True  # Include primary model's probability
    include_market_regime: bool = True  # Include volatility/trend regime
    include_position_features: bool = True  # Recent trade history


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
    
    def __init__(self, config: Optional[MetaLabelingConfig] = None):
        self.config = config or MetaLabelingConfig()
        self.meta_model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self._primary_accuracy: float = 0.5
        
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
        n = len(features)
        
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
            # Sequence: use last timestep + aggregates
            meta_feature_list.append(features[:, -1, :])
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
        
        # Train meta-model
        if self.config.use_xgboost and XGBOOST_AVAILABLE:
            return self._fit_xgboost(meta_X, meta_y, meta_w, meta_X_val, meta_y_val, verbose)
        else:
            return self._fit_logistic(meta_X, meta_y, meta_w, meta_X_val, meta_y_val, verbose)
    
    def _fit_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        verbose: bool,
    ) -> Dict[str, float]:
        """Train XGBoost meta-model."""
        import xgboost as xgb
        
        self.meta_model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )
        
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
            val_acc = float(np.mean((self.meta_model.predict(X_val) > 0.5) == y_val))
            metrics["meta_val_accuracy"] = val_acc
        
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
        
        # Build meta-features (same as training, but no outcomes needed)
        meta_feature_list = []
        
        if features.ndim == 3:
            meta_feature_list.append(features[:, -1, :])
            meta_feature_list.append(features.mean(axis=1))
            meta_feature_list.append(features.std(axis=1))
        else:
            meta_feature_list.append(features)
        
        if self.config.include_primary_proba:
            meta_feature_list.append(primary_predictions.reshape(-1, 1))
            meta_feature_list.append(np.abs(primary_predictions - 0.5).reshape(-1, 1))
        
        meta_X = np.concatenate(meta_feature_list, axis=1).astype(np.float32)
        
        # Get probability predictions
        if hasattr(self.meta_model, "predict_proba"):
            if hasattr(self, "_scaler"):
                meta_X = self._scaler.transform(meta_X)
            proba = self.meta_model.predict_proba(meta_X)[:, 1]
        else:
            proba = self.meta_model.predict(meta_X)
        
        return proba.astype(np.float32)
    
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
        """Save meta-labeler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "config": self.config.__dict__,
            "meta_model": self.meta_model,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "primary_accuracy": self._primary_accuracy,
        }
        
        if hasattr(self, "_scaler"):
            save_dict["scaler"] = self._scaler
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved meta-labeler to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MetaLabeler":
        """Load meta-labeler from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        config = MetaLabelingConfig(**save_dict["config"])
        labeler = cls(config)
        labeler.meta_model = save_dict["meta_model"]
        labeler.is_fitted = save_dict["is_fitted"]
        labeler.feature_names = save_dict["feature_names"]
        labeler._primary_accuracy = save_dict["primary_accuracy"]
        
        if "scaler" in save_dict:
            labeler._scaler = save_dict["scaler"]
        
        return labeler


def train_meta_labeler(
    primary_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Optional[MetaLabelingConfig] = None,
    verbose: bool = True,
) -> Tuple[MetaLabeler, Dict[str, float]]:
    """
    Train a meta-labeler for an existing primary model.
    
    This is the main entry point for adding meta-labeling to training.
    
    Args:
        primary_model: Trained primary model (Keras or XGBoost)
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        config: Meta-labeling configuration
        verbose: Print progress
    
    Returns:
        Tuple of (trained MetaLabeler, metrics dict)
    """
    if verbose:
        logger.info("Training meta-labeler...")
    
    # Get primary model predictions on training data
    if hasattr(primary_model, "predict"):
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
        raise ValueError("Primary model must have a predict() method")
    
    # Create and train meta-labeler
    labeler = MetaLabeler(config)
    
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
        trade_threshold: float = 0.55,
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
            trade_threshold=config.get("trade_threshold", 0.55),
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

