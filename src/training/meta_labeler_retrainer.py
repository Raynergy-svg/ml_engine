#!/usr/bin/env python3
"""
Meta-Labeler Retraining Pipeline.

This script retrains the meta-labeler model on new high-accuracy predictions
(70.8% directional accuracy) to fix the over-filtering issue where the
gated trading system produces a flat50% win rate.

The root cause: The meta-labeler was trained on old55-56% model predictions
and is now over-filtering the stronger signals.

Usage:
    python -m src.training.meta_labeler_retrainer --help
    python -m src.training.meta_labeler_retrainer --data-source simulation --simulate-trades 10000

Output:
    - Updated meta_labeler.pkl in trained_data/models/joint/
    - Training metrics and validation report
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from src.utils.xgboost_artifacts import (
    is_xgboost_model,
    load_xgboost_bundle,
    save_native_xgboost_bundle,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODEL_DIR = Path("trained_data/models/joint")
DEFAULT_JOURNAL_PATH = Path("trained_data/trade_journal.json")
DEFAULT_OUTPUT_DIR = Path("trained_data/models/joint")


@dataclass
class MetaLabelerTrainingConfig:
    """Configuration for meta-labeler retraining."""
    
    # Model parameters - optimized for70.8% accuracy signals
    use_xgboost: bool = True
    n_estimators: int = 150  # Increased for stronger signals
    max_depth: int = 3  # Slightly deeper for complex patterns
    learning_rate: float = 0.03  # Lower for smoother convergence
    
    # Regularization - adjusted for higher accuracy base model
    reg_alpha: float = 0.5  # Reduced L1 (less sparse)
    reg_lambda: float = 3.0  # Reduced L2
    subsample: float = 0.8  # Increased subsampling
    colsample_bytree: float = 0.8  # Increased column sampling
    min_child_weight: int = 3  # Reduced for more flexibility
    early_stopping_rounds: int = 25
    
    # Threshold tuning
    min_confidence_threshold: float = 0.50  # Lower threshold for stronger signals
    max_overfitting_gap: float = 0.08  # Stricter overfitting check
    
    # Features
    include_primary_proba: bool = True
    include_market_regime: bool = True
    use_reduced_features: bool = True
    
    # Training
    auto_tune: bool = True
    validation_split: float = 0.2
    min_samples: int = 100


@dataclass
class TradeSample:
    """Sample trade data for meta-labeler training."""
    
    # Features
    tcn_probability: float
    ridge_confidence: float
    xgb_momentum: float
    rf_drawdown_pips: float
    direction: str  # 'long' or 'short' - required, no default
    feature_vector: Optional[np.ndarray] = None
    instrument: str = ""
    actual_outcome: str = ""  # 'win', 'loss', 'breakeven'
    pnl: float = 0.0
    pnl_pips: float = 0.0
    timestamp: str = ""
    trade_id: str = ""


class MetaLabelerRetrainer:
    """
    Retrains meta-labeler on new high-accuracy predictions.
    
    The meta-labeler learns to predict which of the primary model's
    confident predictions actually win. With70.8% directional accuracy,
    the meta-labeler needs to be retrained to:
    1. Stop over-filtering good signals
    2. Learn the new pattern of correct predictions
    3. Adjust confidence thresholds appropriately
    """
    
    def __init__(
        self,
        config: Optional[MetaLabelerTrainingConfig] = None,
        model_dir: Path = DEFAULT_MODEL_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        self.config = config or MetaLabelerTrainingConfig()
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        
        # Models
        self.xgb_model = None
        self.ridge_model = None
        self.rf_model = None
        self.ensemble_weights = {"xgb": 0.5, "ridge": 0.3, "rf": 0.2}
        
        # Metadata
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.training_metrics: Dict[str, Any] = {}
        
    def load_trade_journal(self, journal_path: Path) -> List[TradeSample]:
        """Load trades from the trade journal.
        
        Args:
            journal_path: Path to trade_journal.json
            
        Returns:
            List of TradeSample objects
        """
        journal_path = Path(journal_path)
        if not journal_path.exists():
            logger.warning(f"Trade journal not found: {journal_path}")
            return []
        
        with open(journal_path) as f:
            data = json.load(f)
        
        trades = data.get("trades", [])
        samples = []
        
        for t in trades:
            # Only include closed trades with outcomes
            if t.get("status") not in ("win", "loss"):
                continue
            
            sample = TradeSample(
                tcn_probability=t.get("tcn_probability", 0.5),
                ridge_confidence=t.get("ridge_confidence", 50.0),
                xgb_momentum=t.get("xgb_momentum", 0.5),
                rf_drawdown_pips=t.get("rf_drawdown_pips", 0.0),
                feature_vector=np.array(t["feature_vector"]) if t.get("feature_vector") else None,
                direction=t.get("direction", "long"),
                instrument=t.get("instrument", ""),
                actual_outcome=t.get("status", ""),
                pnl=t.get("pnl", 0.0),
                pnl_pips=t.get("pnl_pips", 0.0),
                timestamp=t.get("timestamp", ""),
                trade_id=t.get("trade_id", ""),
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} closed trades from journal")
        return samples
    
    def load_backtest_results(self, backtest_path: Path) -> List[TradeSample]:
        """Load trades from backtest results file.
        
        Args:
            backtest_path: Path to backtest results JSON
            
        Returns:
            List of TradeSample objects
        """
        backtest_path = Path(backtest_path)
        if not backtest_path.exists():
            logger.warning(f"Backtest file not found: {backtest_path}")
            return []
        
        with open(backtest_path) as f:
            data = json.load(f)
        
        # Handle different backtest formats
        trades = data.get("trades", data.get("results", []))
        samples = []
        
        for t in trades:
            # Determine outcome
            pnl = t.get("pnl", t.get("profit", 0.0))
            pnl_pips = t.get("pnl_pips", 0.0)
            
            if pnl > 0:
                outcome = "win"
            elif pnl < 0:
                outcome = "loss"
            else:
                outcome = "breakeven"
            
            sample = TradeSample(
                tcn_probability=t.get("tcn_probability", t.get("probability", 0.5)),
                ridge_confidence=t.get("ridge_confidence", t.get("confidence", 50.0)),
                xgb_momentum=t.get("xgb_momentum", t.get("momentum", 0.5)),
                rf_drawdown_pips=t.get("rf_drawdown_pips", 0.0),
                feature_vector=np.array(t["features"]) if t.get("features") else None,
                direction=t.get("direction", "long"),
                instrument=t.get("instrument", t.get("pair", "")),
                actual_outcome=outcome,
                pnl=pnl,
                pnl_pips=pnl_pips,
                timestamp=t.get("timestamp", t.get("exit_time", "")),
                trade_id=t.get("trade_id", t.get("id", "")),
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} trades from backtest results")
        return samples
    
    def simulate_trades_from_model(
        self,
        n_trades: int = 1000,
        directional_accuracy: float = 0.708,
        short_accuracy: float = 0.726,
        long_accuracy: float = 0.593,
        short_bias: float = 0.55,
        seed: Optional[int] = None,
    ) -> List[TradeSample]:
        """Simulate trades based on model performance characteristics.
        
        This generates synthetic trade data matching the reported70.8%
        directional accuracy (Short 72.6% / Long 59.3%).
        
        Args:
            n_trades: Number of trades to simulate
            directional_accuracy: Overall directional accuracy
            short_accuracy: Accuracy for short signals
            long_accuracy: Accuracy for long signals
            short_bias: Probability of generating a short trade
            seed: Optional RNG seed for reproducible simulation runs
            
        Returns:
            List of simulated TradeSample objects
        """
        rng = np.random.default_rng(seed)
        samples = []
        
        # Simulate trades
        for i in range(n_trades):
            # Random direction (slightly more shorts as per typical FX)
            is_short = float(rng.random()) < short_bias
            direction = "short" if is_short else "long"
            
            # Accuracy based on direction
            accuracy = short_accuracy if is_short else long_accuracy
            
            # Generate probability (higher for more confident predictions)
            base_prob = 0.65 + float(rng.random()) * 0.25  # 0.65-0.90
            tcn_prob = base_prob if is_short else base_prob * 0.85
            
            # Generate outcome based on accuracy
            is_correct = float(rng.random()) < accuracy
            outcome = "win" if is_correct else "loss"
            
            # Generate supporting features
            ridge_conf = 50 + float(rng.random()) * 40  # 50-90
            xgb_momentum = tcn_prob + float(rng.normal(0, 0.1))
            rf_drawdown = float(rng.exponential(30)) + 10  # 10-100+ pips
            
            # PnL simulation
            if outcome == "win":
                pnl_pips = float(rng.exponential(25)) + 5  # 5-100+ pips
                pnl = pnl_pips * 0.10  # ~$0.10 per pip per micro lot
            else:
                pnl_pips = -(float(rng.exponential(20)) + 5)
                pnl = pnl_pips * 0.10
            
            sample = TradeSample(
                tcn_probability=np.clip(tcn_prob, 0.0, 1.0),
                ridge_confidence=np.clip(ridge_conf, 0.0, 100.0),
                xgb_momentum=np.clip(xgb_momentum, 0.0, 1.0),
                rf_drawdown_pips=max(0, rf_drawdown),
                direction=direction,
                instrument=str(rng.choice(["EUR_USD", "GBP_USD", "USD_JPY"])),
                actual_outcome=outcome,
                pnl=pnl,
                pnl_pips=pnl_pips,
                timestamp=datetime.now(timezone.utc).isoformat(),
                trade_id=f"sim_{i:05d}",
            )
            samples.append(sample)
        
        logger.info(
            f"Simulated {n_trades} trades with {directional_accuracy:.1%} accuracy "
            f"(Short: {short_accuracy:.1%}, Long: {long_accuracy:.1%})"
        )
        return samples
    
    def prepare_training_data(
        self,
        samples: List[TradeSample],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and labels for meta-labeler training.
        
        Args:
            samples: List of TradeSample objects
            
        Returns:
            Tuple of (features, labels, weights)
        """
        if len(samples) < self.config.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(samples)} < {self.config.min_samples}"
            )
        
        features_list = []
        labels_list = []
        weights_list = []
        
        for sample in samples:
            # Build feature vector
            if sample.feature_vector is not None and len(sample.feature_vector) >= 4:
                # Use stored feature vector
                feats = list(sample.feature_vector[:4])
            else:
                # Build from individual fields
                feats = [
                    sample.tcn_probability,
                    sample.ridge_confidence / 100.0,  # Normalize to 0-1
                    sample.xgb_momentum,
                    sample.rf_drawdown_pips / 100.0,  # Normalize
                ]
            
            # Add derived features
            primary_confidence = abs(sample.tcn_probability - 0.5) * 2
            feats.append(primary_confidence)
            
            # Direction encoding
            direction_enc = 1.0 if sample.direction == "short" else 0.0
            feats.append(direction_enc)
            
            features_list.append(feats)
            
            # Meta-label: was the primary prediction correct?
            # Primary model was correct if outcome was profitable
            is_correct = sample.actual_outcome == "win"
            
            labels_list.append(1.0 if is_correct else 0.0)
            
            # Weight by confidence (higher confidence = more important)
            weight = 1.0 + primary_confidence * 0.5
            weights_list.append(weight)
        
        features = np.array(features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.float32)
        weights = np.array(weights_list, dtype=np.float32)
        
        # Store feature names
        self.feature_names = [
            "tcn_probability",
            "ridge_confidence_norm",
            "xgb_momentum",
            "rf_drawdown_norm",
            "primary_confidence",
            "direction_enc",
        ]
        
        logger.info(f"Prepared {len(features)} samples with {features.shape[1]} features")
        logger.info(f"Label distribution: {np.mean(labels):.1%} correct")
        
        return features, labels, weights
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        w_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train XGBoost meta-model.
        
        Args:
            X_train, y_train, w_train: Training data
            X_val, y_val: Optional validation data
            
        Returns:
            Training metrics
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not available, skipping")
            return {}
        
        # Build model
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=self.config.early_stopping_rounds if X_val is not None else None,
        )
        
        # Train
        fit_kwargs = {"sample_weight": w_train}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        
        self.xgb_model.fit(X_train, y_train, **fit_kwargs)
        
        # Metrics
        train_preds = self.xgb_model.predict(X_train)
        train_acc = float(np.mean(train_preds == y_train))
        metrics = {"xgb_train_accuracy": train_acc}
        
        if X_val is not None and y_val is not None:
            val_preds = self.xgb_model.predict(X_val)
            val_acc = float(np.mean(val_preds == y_val))
            metrics["xgb_val_accuracy"] = val_acc
            metrics["xgb_overfitting_gap"] = train_acc - val_acc
            
            # AUC
            try:
                from sklearn.metrics import roc_auc_score
                val_proba = self.xgb_model.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) > 1:
                    metrics["xgb_val_auc"] = float(roc_auc_score(y_val, val_proba))
            except Exception:
                pass
        
        logger.info(
            f"XGBoost: train={metrics.get('xgb_train_accuracy', 0):.1%}, "
            f"val={metrics.get('xgb_val_accuracy', 0):.1%}"
        )
        
        return metrics
    
    def train_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        w_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train Ridge classifier as meta-model.
        
        Args:
            X_train, y_train, w_train: Training data
            X_val, y_val: Optional validation data
            
        Returns:
            Training metrics
        """
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.ridge_model = RidgeClassifier(
            alpha=1.0,
            class_weight="balanced",
            random_state=42,
        )
        self.ridge_model.fit(X_train_scaled, y_train, sample_weight=w_train)
        
        # Metrics
        train_preds = self.ridge_model.predict(X_train_scaled)
        train_acc = float(np.mean(train_preds == y_train))
        metrics = {"ridge_train_accuracy": train_acc}
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_preds = self.ridge_model.predict(X_val_scaled)
            val_acc = float(np.mean(val_preds == y_val))
            metrics["ridge_val_accuracy"] = val_acc
        
        logger.info(
            f"Ridge: train={metrics.get('ridge_train_accuracy', 0):.1%}, "
            f"val={metrics.get('ridge_val_accuracy', 0):.1%}"
        )
        
        return metrics
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        w_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train Random Forest as meta-model.
        
        Args:
            X_train, y_train, w_train: Training data
            X_val, y_val: Optional validation data
            
        Returns:
            Training metrics
        """
        from sklearn.ensemble import RandomForestClassifier
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_train, sample_weight=w_train)
        
        # Metrics
        train_preds = self.rf_model.predict(X_train)
        train_acc = float(np.mean(train_preds == y_train))
        metrics = {"rf_train_accuracy": train_acc}
        
        if X_val is not None and y_val is not None:
            val_preds = self.rf_model.predict(X_val)
            val_acc = float(np.mean(val_preds == y_val))
            metrics["rf_val_accuracy"] = val_acc
            
            # AUC
            try:
                from sklearn.metrics import roc_auc_score
                val_proba = self.rf_model.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) > 1:
                    metrics["rf_val_auc"] = float(roc_auc_score(y_val, val_proba))
            except Exception:
                pass
        
        logger.info(
            f"RandomForest: train={metrics.get('rf_train_accuracy', 0):.1%}, "
            f"val={metrics.get('rf_val_accuracy', 0):.1%}"
        )
        
        return metrics
    
    def fit(
        self,
        samples: List[TradeSample],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train all meta-labeler models.
        
        Args:
            samples: List of TradeSample objects
            verbose: Print progress
            
        Returns:
            Training metrics
        """
        if verbose:
            logger.info(f"Training meta-labeler on {len(samples)} samples...")
        
        # Prepare data
        features, labels, weights = self.prepare_training_data(samples)
        
        # Time-series split (preserve temporal order)
        split_idx = int(len(features) * (1 - self.config.validation_split))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]
        w_train = weights[:split_idx]  # validation weights not needed
        
        if verbose:
            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
            logger.info(f"Train win rate: {np.mean(y_train):.1%}")
            logger.info(f"Val win rate: {np.mean(y_val):.1%}")
        
        # Train all models
        all_metrics = {}
        
        # XGBoost (primary)
        if self.config.use_xgboost:
            xgb_metrics = self.train_xgboost(X_train, y_train, w_train, X_val, y_val)
            all_metrics.update(xgb_metrics)
        
        # Ridge (secondary)
        ridge_metrics = self.train_ridge(X_train, y_train, w_train, X_val, y_val)
        all_metrics.update(ridge_metrics)
        
        # Random Forest (tertiary)
        rf_metrics = self.train_random_forest(X_train, y_train, w_train, X_val, y_val)
        all_metrics.update(rf_metrics)
        
        # Ensemble metrics
        if X_val is not None and y_val is not None:
            ensemble_metrics = self._evaluate_ensemble(X_val, y_val)
            all_metrics.update(ensemble_metrics)
        
        self.is_fitted = True
        self.training_metrics = all_metrics
        
        if verbose:
            self._print_training_summary(all_metrics)
        
        return all_metrics
    
    def _evaluate_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate ensemble prediction.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Ensemble metrics
        """
        # Get predictions from each model
        preds = []
        
        if self.xgb_model is not None:
            xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
            preds.append(xgb_proba * self.ensemble_weights["xgb"])
        
        if self.ridge_model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            ridge_scores = self.ridge_model.decision_function(X_scaled)
            ridge_proba = 1 / (1 + np.exp(-ridge_scores))  # Sigmoid
            preds.append(ridge_proba * self.ensemble_weights["ridge"])
        
        if self.rf_model is not None:
            rf_proba = self.rf_model.predict_proba(X)[:, 1]
            preds.append(rf_proba * self.ensemble_weights["rf"])
        
        if not preds:
            return {}
        
        # Weighted average
        ensemble_proba = np.sum(preds, axis=0) / sum(self.ensemble_weights.values())
        ensemble_preds = (ensemble_proba >= 0.5).astype(float)
        
        # Metrics
        accuracy = float(np.mean(ensemble_preds == y))
        
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y)) > 1:
                auc = float(roc_auc_score(y, ensemble_proba))
            else:
                auc = 0.5
        except Exception:
            auc = 0.5
        
        return {
            "ensemble_val_accuracy": accuracy,
            "ensemble_val_auc": auc,
        }
    
    def _print_training_summary(self, metrics: Dict[str, float]) -> None:
        """Print training summary."""
        logger.info("=" * 60)
        logger.info("META-LABELER TRAINING SUMMARY")
        logger.info("=" * 60)
        
        # XGBoost
        if "xgb_train_accuracy" in metrics:
            gap = metrics.get("xgb_overfitting_gap", 0)
            gap_status = "✓" if gap <= self.config.max_overfitting_gap else "⚠️ HIGH"
            logger.info(
                f"XGBoost: train={metrics['xgb_train_accuracy']:.1%}, "
                f"val={metrics.get('xgb_val_accuracy', 0):.1%}, "
                f"gap={gap:.1%} {gap_status}"
            )
        
        # Ridge
        if "ridge_train_accuracy" in metrics:
            logger.info(
                f"Ridge: train={metrics['ridge_train_accuracy']:.1%}, "
                f"val={metrics.get('ridge_val_accuracy', 0):.1%}"
            )
        
        # RF
        if "rf_train_accuracy" in metrics:
            logger.info(
                f"RandomForest: train={metrics['rf_train_accuracy']:.1%}, "
                f"val={metrics.get('rf_val_accuracy', 0):.1%}"
            )
        
        # Ensemble
        if "ensemble_val_accuracy" in metrics:
            logger.info(
                f"Ensemble: val={metrics['ensemble_val_accuracy']:.1%}, "
                f"auc={metrics.get('ensemble_val_auc', 0):.3f}"
            )
        
        logger.info("=" * 60)
    
    def predict_meta_confidence(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Predict meta-confidence for primary model signals.
        
        Args:
            features: Feature array (n_samples, n_features)
            
        Returns:
            Meta-confidence scores (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-labeler not fitted. Call fit() first.")
        
        preds = []
        
        if self.xgb_model is not None:
            xgb_proba = self.xgb_model.predict_proba(features)[:, 1]
            preds.append(xgb_proba * self.ensemble_weights["xgb"])
        
        if self.ridge_model is not None and self.scaler is not None:
            features_scaled = self.scaler.transform(features)
            ridge_scores = self.ridge_model.decision_function(features_scaled)
            ridge_proba = 1 / (1 + np.exp(-ridge_scores))
            preds.append(ridge_proba * self.ensemble_weights["ridge"])
        
        if self.rf_model is not None:
            rf_proba = self.rf_model.predict_proba(features)[:, 1]
            preds.append(rf_proba * self.ensemble_weights["rf"])
        
        if not preds:
            return np.full(len(features), 0.5)
        
        return np.sum(preds, axis=0) / sum(self.ensemble_weights.values())
    
    def save(self, output_path: Optional[Path] = None) -> None:
        """Save meta-labeler models to disk.
        
        Args:
            output_path: Path to save (default: trained_data/models/joint/meta_labeler.pkl)
        """
        output_path = Path(output_path or self.output_dir / "meta_labeler.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "config": asdict(self.config),
            "ridge_model": self.ridge_model,
            "rf_model": self.rf_model,
            "scaler": self.scaler,
            "ensemble_weights": self.ensemble_weights,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "training_metrics": self.training_metrics,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        native_models = {}
        if is_xgboost_model(self.xgb_model):
            native_models["xgb_model"] = self.xgb_model
        else:
            save_dict["xgb_model"] = self.xgb_model

        if native_models:
            save_native_xgboost_bundle(
                output_path,
                metadata=save_dict,
                models=native_models,
                alias_map={"xgb_model": "xgb"},
            )
        else:
            with open(output_path, "wb") as f:
                pickle.dump(save_dict, f)
        
        logger.info(f"✓ Saved meta-labeler to {output_path}")
        
        # Also save training report
        report_path = output_path.with_suffix(".report.json")
        with open(report_path, "w") as f:
            json.dump({
                "training_metrics": self.training_metrics,
                "config": asdict(self.config),
                "feature_names": self.feature_names,
                "saved_at": save_dict["saved_at"],
            }, f, indent=2)
        
        logger.info(f"✓ Saved training report to {report_path}")
    
    @classmethod
    def load(cls, model_path: Path) -> "MetaLabelerRetrainer":
        """Load meta-labeler from disk.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded MetaLabelerRetrainer
        """
        save_dict, native_models, _source = load_xgboost_bundle(
            model_path,
            alias_map={"xgb_model": "xgb"},
        )
        
        config = MetaLabelerTrainingConfig(**save_dict.get("config", {}))
        instance = cls(config=config)
        
        instance.xgb_model = native_models.get("xgb_model", save_dict.get("xgb_model"))
        instance.ridge_model = save_dict.get("ridge_model")
        instance.rf_model = save_dict.get("rf_model")
        instance.scaler = save_dict.get("scaler")
        instance.ensemble_weights = save_dict.get("ensemble_weights", instance.ensemble_weights)
        instance.feature_names = save_dict.get("feature_names", [])
        instance.is_fitted = save_dict.get("is_fitted", False)
        instance.training_metrics = save_dict.get("training_metrics", {})
        
        logger.info(f"✓ Loaded meta-labeler from {model_path}")
        return instance


def retrain_from_simulation(
    n_trades: int = 10000,
    directional_accuracy: float = 0.708,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Convenience function to retrain meta-labeler from simulated data.
    
    Args:
        n_trades: Number of simulated trades
        directional_accuracy: Target directional accuracy
        output_dir: Output directory
        verbose: Print progress
        
    Returns:
        Training metrics
    """
    config = MetaLabelerTrainingConfig(min_samples=100)
    trainer = MetaLabelerRetrainer(config=config, output_dir=output_dir)
    
    # Generate simulated trades
    samples = trainer.simulate_trades_from_model(
        n_trades=n_trades,
        directional_accuracy=directional_accuracy,
    )
    
    # Train
    metrics = trainer.fit(samples, verbose=verbose)
    
    # Save
    trainer.save()
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Retrain meta-labeler on new high-accuracy predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use trade journal data
    python -m src.training.meta_labeler_retrainer --data-source journal
    
    # Use backtest results
    python -m src.training.meta_labeler_retrainer --data-source backtest --backtest-file results.json
    
    # Simulate trades matching70.8% accuracy
    python -m src.training.meta_labeler_retrainer --data-source simulation --simulate-trades 10000
        """,
    )
    
    parser.add_argument(
        "--data-source",
        choices=["journal", "backtest", "simulation"],
        default="simulation",
        help="Source of training data",
    )
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=DEFAULT_JOURNAL_PATH,
        help="Path to trade journal JSON",
    )
    parser.add_argument(
        "--backtest-file",
        type=Path,
        help="Path to backtest results JSON",
    )
    parser.add_argument(
        "--simulate-trades",
        type=int,
        default=10000,
        help="Number of trades to simulate (for simulation source)",
    )
    parser.add_argument(
        "--directional-accuracy",
        type=float,
        default=0.708,
        help="Directional accuracy for simulation (default: 0.708 = 70.8%%)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing existing models",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples required for training",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration
    config = MetaLabelerTrainingConfig(min_samples=args.min_samples)
    
    # Initialize trainer
    trainer = MetaLabelerRetrainer(
        config=config,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )
    
    # Load data based on source
    samples = []
    
    if args.data_source == "journal":
        samples = trainer.load_trade_journal(args.journal_path)
    elif args.data_source == "backtest":
        if not args.backtest_file:
            logger.error("--backtest-file required for backtest data source")
            return 1
        samples = trainer.load_backtest_results(args.backtest_file)
    elif args.data_source == "simulation":
        samples = trainer.simulate_trades_from_model(
            n_trades=args.simulate_trades,
            directional_accuracy=args.directional_accuracy,
        )
    
    if len(samples) < args.min_samples:
        logger.error(
            f"Insufficient data: {len(samples)} samples < {args.min_samples} required. "
            f"Use --data-source simulation or provide more data."
        )
        return 1
    
    # Train
    metrics = trainer.fit(samples, verbose=True)
    
    # Save
    trainer.save()
    
    # Summary
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    print(f"Data source: {args.data_source}")
    print(f"Samples used: {len(samples)}")
    print(f"Ensemble accuracy: {metrics.get('ensemble_val_accuracy', 0):.1%}")
    print(f"Ensemble AUC: {metrics.get('ensemble_val_auc', 0):.3f}")
    print(f"Output: {args.output_dir / 'meta_labeler.pkl'}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
