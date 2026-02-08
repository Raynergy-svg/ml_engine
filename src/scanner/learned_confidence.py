"""
Learned Confidence Model.

Replaces the fixed-weight heuristic confidence calculation with a model
trained on actual trade outcomes from the trade journal.

Features used (same as _compute_confidence_direct()):
- ADX: Trend strength
- RSI: Momentum oscillator
- ATR percent: Volatility measure
- Bollinger Band position: Mean reversion signal
- Volume ratio: Volume confirmation
- Spread (optional): Liquidity indicator

Target: Binary win/loss outcome from closed trades.

Model: ElasticNet with cross-validation for automatic feature selection.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum trades required for reliable model training
MIN_TRADES_FOR_TRAINING = 50

# Feature names matching the heuristic confidence calculation
CONFIDENCE_FEATURES = [
    'adx',
    'rsi',
    'atr_pct',
    'bb_position',
    'volume_ratio',
]


@dataclass
class LearnedConfidenceConfig:
    """Configuration for learned confidence model."""

    # ElasticNet parameters
    alpha: float = 0.1  # Regularization strength
    l1_ratio: float = 0.5  # Mix ratio (0=Ridge, 1=Lasso)

    # Cross-validation
    cv_folds: int = 5

    # Minimum training data
    min_trades: int = MIN_TRADES_FOR_TRAINING

    # Output bounds
    min_confidence: float = 0.0
    max_confidence: float = 1.0


class LearnedConfidenceModel:
    """Learned confidence model trained on trade outcomes.

    Uses ElasticNet regression to predict trade success probability
    based on market features. Trained from closed trades in the journal.

    Advantages over heuristic:
    - Learns optimal feature weights from actual outcomes
    - L1 regularization automatically selects relevant features
    - Updates as more trade data accumulates
    """

    def __init__(self, config: Optional[LearnedConfidenceConfig] = None):
        """Initialize learned confidence model.

        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or LearnedConfidenceConfig()
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names: List[str] = CONFIDENCE_FEATURES.copy()
        self.training_stats: Dict[str, Any] = {}

    def fit(
        self,
        features: np.ndarray,
        outcomes: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fit model on trade data.

        Args:
            features: Feature matrix (n_samples, n_features)
            outcomes: Binary outcomes (1=win, 0=loss)
            feature_names: Optional feature names for logging

        Returns:
            Dict with training metrics:
                - cv_score: Mean cross-validation score
                - n_samples: Number of training samples
                - feature_importance: Dict of feature weights

        Raises:
            ValueError: If insufficient training data
        """
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        n_samples = len(outcomes)

        if n_samples < self.config.min_trades:
            raise ValueError(
                f"Insufficient training data: {n_samples} trades, "
                f"need at least {self.config.min_trades}. "
                f"Continue trading to accumulate more data."
            )

        if feature_names:
            self.feature_names = feature_names

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        # Fit ElasticNet with cross-validation for alpha selection
        self.model = ElasticNetCV(
            l1_ratio=self.config.l1_ratio,
            cv=min(self.config.cv_folds, n_samples // 2),
            random_state=42,
            max_iter=2000,
        )
        self.model.fit(X_scaled, outcomes)

        # Calculate cross-validation score
        cv_scores = cross_val_score(
            self.model, X_scaled, outcomes,
            cv=min(self.config.cv_folds, n_samples // 2),
            scoring='r2',
        )

        self.is_fitted = True

        # Feature importance from coefficients
        feature_importance = {}
        if hasattr(self.model, 'coef_'):
            for i, name in enumerate(self.feature_names):
                if i < len(self.model.coef_):
                    feature_importance[name] = float(self.model.coef_[i])

        self.training_stats = {
            'cv_score': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'n_samples': n_samples,
            'alpha': float(self.model.alpha_),
            'feature_importance': feature_importance,
            'n_features': features.shape[1] if len(features.shape) > 1 else 1,
        }

        logger.info(
            f"✓ Learned confidence model trained on {n_samples} trades, "
            f"CV R²={self.training_stats['cv_score']:.3f}±{self.training_stats['cv_std']:.3f}"
        )

        return self.training_stats

    def predict(self, features: np.ndarray) -> float:
        """Predict confidence for given features.

        Args:
            features: Feature vector (n_features,) or (1, n_features)

        Returns:
            Confidence score in [0, 1] range
        """
        if not self.is_fitted:
            logger.warning("Learned confidence model not fitted, returning 0.5")
            return 0.5

        try:
            # Ensure 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Scale
            X_scaled = self.scaler.transform(features)

            # Predict
            raw_pred = float(self.model.predict(X_scaled)[0])

            # Clip to valid range
            confidence = np.clip(
                raw_pred,
                self.config.min_confidence,
                self.config.max_confidence,
            )

            return float(confidence)

        except Exception as e:
            logger.debug(f"Learned confidence prediction failed: {e}")
            return 0.5

    def save(self, filepath: Path) -> None:
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'config': self.config,
            'training_stats': self.training_stats,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"✓ Learned confidence model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'LearnedConfidenceModel':
        """Load model from file.

        Args:
            filepath: Path to load model from

        Returns:
            Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(config=data.get('config'))
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = data['is_fitted']
        instance.feature_names = data.get('feature_names', CONFIDENCE_FEATURES)
        instance.training_stats = data.get('training_stats', {})

        logger.info(f"✓ Learned confidence model loaded from {filepath}")
        return instance


def extract_confidence_features_from_trade(trade: Any) -> Optional[np.ndarray]:
    """Extract confidence features from a trade entry.

    Args:
        trade: TradeEntry object from trade journal

    Returns:
        Feature vector or None if extraction fails
    """
    try:
        # Try to get feature vector if stored
        if hasattr(trade, 'feature_vector') and trade.feature_vector is not None:
            fv = trade.feature_vector
            if len(fv) >= 5:
                return np.array(fv[:5], dtype=float)

        # Fallback: extract from individual fields
        features = []

        # ADX (trend strength) - use ridge_confidence as proxy
        adx = getattr(trade, 'ridge_confidence', 0.5) * 50  # Scale back to 0-50 range
        features.append(adx)

        # RSI - not typically stored, use neutral value
        rsi = getattr(trade, 'rsi', 50.0)
        features.append(rsi)

        # ATR percent - use position sizing info as proxy
        atr_pct = getattr(trade, 'atr_pct', 0.01)
        features.append(atr_pct)

        # BB position - not stored, use 0.5 (middle)
        bb_pos = getattr(trade, 'bb_position', 0.5)
        features.append(bb_pos)

        # Volume ratio
        volume_ratio = getattr(trade, 'volume_ratio', 1.0)
        features.append(volume_ratio)

        return np.array(features, dtype=float)

    except Exception as e:
        logger.debug(f"Failed to extract features from trade: {e}")
        return None


def train_from_journal(
    journal_path: Path,
    model_save_path: Path,
    min_trades: int = MIN_TRADES_FOR_TRAINING,
) -> Optional[Dict[str, Any]]:
    """Train learned confidence model from trade journal.

    Args:
        journal_path: Path to trade journal file
        model_save_path: Path to save trained model
        min_trades: Minimum trades required

    Returns:
        Training stats dict or None if training failed
    """
    from src.utils.trade_journal import TradeJournal

    # Load journal
    try:
        journal = TradeJournal(journal_path)
    except Exception as e:
        logger.error(f"Failed to load trade journal: {e}")
        return None

    # Filter closed trades with outcomes
    closed_trades = [
        t for t in journal.trades
        if getattr(t, 'status', None) in ('win', 'loss')
    ]

    if len(closed_trades) < min_trades:
        logger.warning(
            f"Insufficient closed trades: {len(closed_trades)}/{min_trades}. "
            f"Need {min_trades - len(closed_trades)} more trades."
        )
        return None

    # Extract features and outcomes
    features_list = []
    outcomes_list = []

    for trade in closed_trades:
        feats = extract_confidence_features_from_trade(trade)
        if feats is not None:
            features_list.append(feats)
            outcomes_list.append(1.0 if trade.status == 'win' else 0.0)

    if len(features_list) < min_trades:
        logger.warning(
            f"Only {len(features_list)} trades had extractable features. "
            f"Need at least {min_trades}."
        )
        return None

    # Train model
    features = np.array(features_list)
    outcomes = np.array(outcomes_list)

    model = LearnedConfidenceModel()
    try:
        stats = model.fit(features, outcomes)
        model.save(model_save_path)
        return stats
    except ValueError as e:
        logger.warning(f"Training failed: {e}")
        return None
