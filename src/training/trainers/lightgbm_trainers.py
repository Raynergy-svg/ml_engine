"""
LightGBM Trainer Variants.

This module contains LightGBM-based trainers for specialized tasks:
- RegimeLGBMTrainer: Trains 5 separate models optimized for each market regime
- LightGBMMomentumTrainer: Momentum analysis using LightGBM (replaces XGBoost)
- LightGBMRiskTrainer: Risk assessment using LightGBM (replaces RandomForest)

LightGBM advantages:
- Faster training with categorical feature support
- Better memory efficiency for large datasets
- Supports init_model for warm-start fine-tuning
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.training.trainers.base import BaseTrainer
from src.training.trainers.config import TrainerConfig
from src.training.trainers.utils import (
    MODEL_NOT_TRAINED_ERROR,
    PRODUCTION_MODELS_DIR,
    SERIALIZED_MODEL_WARNING,
    LGBM_NOT_INSTALLED_ERROR,
    REGIME_NAMES_LIST,
    get_regime_lgbm_params,
    _create_lgbm_classifier,
    _create_lgbm_regressor,
)

logger = logging.getLogger(__name__)


class RegimeLGBMTrainer(BaseTrainer):
    """
    Regime-specific LightGBM trainer that trains 5 separate models,
    one optimized for each market regime.

    Models are saved to: trained_data/models/{instrument}/lgbm_regime_{REGIME}.pkl

    Each regime has different hyperparameters:
    - STRONG_TREND: Deeper trees, less regularization, captures momentum
    - WEAK_TREND: Conservative, moderate regularization for noisy signals
    - CHOP: Shallow trees, heavy regularization to avoid noise overfitting
    - MEAN_REVERT: More iterations for reversal pattern learning
    - BREAKOUT: Fast learning for volatility expansion detection

    Input: Features + regime labels for filtering
    Output: Direction prediction per regime
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.regime_models: Dict[str, Any] = {}
        self.regime_scalers: Dict[str, Any] = {}
        self.regime_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: Optional[List[str]] = None
        self.n_features: Optional[int] = None

    def train_regime_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        regimes_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        regimes_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        min_samples_per_regime: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train 5 separate LightGBM models, one per market regime.

        Args:
            X_train: Training features
            y_train: Training labels (0=down, 1=up)
            regimes_train: Regime labels for each training sample (0-4)
            x_val: Validation features
            y_val: Validation labels
            regimes_val: Regime labels for validation samples
            feature_names: Feature names for LightGBM
            min_samples_per_regime: Skip regime if fewer samples

        Returns:
            Dict mapping regime name to metrics dict
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error(LGBM_NOT_INSTALLED_ERROR)
            raise

        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score
        import pandas as pd

        # Import regime names from data loaders
        try:
            from src.core.modular_data_loaders import REGIME_NAMES
        except ImportError:
            REGIME_NAMES = {
                0: "STRONG_TREND",
                1: "WEAK_TREND",
                2: "CHOP",
                3: "MEAN_REVERT",
                4: "BREAKOUT",
            }

        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Generate feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        all_metrics = {}

        for regime_id in range(5):
            regime_name = REGIME_NAMES.get(regime_id, f"REGIME_{regime_id}")

            # Filter data for this regime
            train_mask = regimes_train == regime_id
            val_mask = regimes_val == regime_id

            n_train = train_mask.sum()
            n_val = val_mask.sum()

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {regime_name} model (train={n_train}, val={n_val})")
            logger.info(f"{'=' * 60}")

            # Skip if insufficient samples
            if n_train < min_samples_per_regime:
                logger.warning(
                    f"Skipping {regime_name}: only {n_train} samples "
                    f"(min={min_samples_per_regime})"
                )
                all_metrics[regime_name] = {
                    "status": "skipped",
                    "reason": f"insufficient_samples ({n_train})",
                    "n_train": n_train,
                    "n_val": n_val,
                }
                continue

            x_train_regime = X_train[train_mask]
            y_train_regime = y_train[train_mask]
            x_val_regime = x_val[val_mask] if n_val > 0 else x_train_regime[:100]
            y_val_regime = y_val[val_mask] if n_val > 0 else y_train_regime[:100]

            # Scale features per regime
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train_regime)
            x_val_scaled = scaler.transform(x_val_regime)

            # Convert to DataFrame for LightGBM
            x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
            x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)

            # Get regime-specific hyperparameters
            params = get_regime_lgbm_params(regime_name)

            # Create and train model
            model = lgb.LGBMClassifier(**params)

            try:
                model.fit(
                    x_train_df,
                    y_train_regime,
                    eval_set=[(x_val_df, y_val_regime)],
                )
            except Exception as e:
                logger.error(f"Training failed for {regime_name}: {e}")
                all_metrics[regime_name] = {
                    "status": "failed",
                    "reason": str(e),
                }
                continue

            # Calculate metrics
            y_pred_train = model.predict(x_train_df)
            y_pred_val = model.predict(x_val_df)

            train_acc = accuracy_score(y_train_regime, y_pred_train)
            val_acc = accuracy_score(y_val_regime, y_pred_val) if n_val > 0 else 0.0
            val_f1 = (
                f1_score(y_val_regime, y_pred_val, zero_division=0)
                if n_val > 0
                else 0.0
            )

            # Feature importance
            importances = model.feature_importances_
            top_features = sorted(
                zip(self.feature_names, importances), key=lambda x: x[1], reverse=True
            )[:5]

            # Store model and scaler
            self.regime_models[regime_name] = model
            self.regime_scalers[regime_name] = scaler

            metrics = {
                "status": "trained",
                "n_train": n_train,
                "n_val": n_val,
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "val_f1": float(val_f1),
                "params": params,
                "top_features": [(f, float(i)) for f, i in top_features],
            }

            self.regime_metrics[regime_name] = metrics
            all_metrics[regime_name] = metrics

            logger.info(
                f"{regime_name}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
                f"val_f1={val_f1:.3f}"
            )

        self.is_trained = True
        return all_metrics

    def predict(
        self,
        X: np.ndarray,
        regime: str = "WEAK_TREND",
    ) -> Dict[str, Any]:
        """
        Predict direction using regime-specific model.

        Args:
            X: Feature matrix (last row used for prediction)
            regime: Market regime name for model selection

        Returns:
            Dict with 'direction' (0/1) and 'probability'
        """
        import pandas as pd

        if not self.is_trained:
            raise RuntimeError("Models not trained. Call train_regime_models() first.")

        # Get model for regime (fallback to WEAK_TREND)
        model = self.regime_models.get(regime)
        scaler = self.regime_scalers.get(regime)

        if model is None:
            logger.warning(f"No model for regime {regime}, using WEAK_TREND")
            model = self.regime_models.get("WEAK_TREND")
            scaler = self.regime_scalers.get("WEAK_TREND")

        if model is None:
            raise RuntimeError("No trained models available")

        # Prepare input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_last = X[-1:] if len(X) > 1 else X

        # Scale
        x_scaled = scaler.transform(x_last)
        x_df = pd.DataFrame(x_scaled, columns=self.feature_names)

        # Predict
        direction = int(model.predict(x_df)[0])
        proba = model.predict_proba(x_df)[0]
        probability = float(proba[1])  # Probability of up

        return {
            "direction": direction,
            "probability": probability,
            "regime_used": regime,
        }

    def save(
        self,
        base_path: str = PRODUCTION_MODELS_DIR,
        instrument: str = "EUR_USD"
    ) -> Dict[str, Path]:
        """
        Save all regime models to disk.

        Args:
            base_path: Base directory (e.g., 'trained_data/models')
            instrument: Currency pair name

        Returns:
            Dict mapping regime name to saved file path
        """
        save_dir = Path(base_path) / instrument
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for regime_name in REGIME_NAMES_LIST:
            if regime_name not in self.regime_models:
                continue

            model_path = save_dir / f"lgbm_regime_{regime_name}.pkl"

            data = {
                "model": self.regime_models[regime_name],
                "scaler": self.regime_scalers[regime_name],
                "metrics": self.regime_metrics.get(regime_name, {}),
                "feature_names": self.feature_names,
                "n_features": self.n_features,
                "regime": regime_name,
                "saved_at": datetime.now().isoformat(),
            }

            with open(model_path, "wb") as f:
                pickle.dump(data, f)

            saved_paths[regime_name] = model_path
            logger.info(f"Saved {regime_name} model to {model_path}")

        # Save metadata
        meta_path = save_dir / "regime_lgbm_meta.json"
        import json

        meta = {
            "instrument": instrument,
            "regimes_trained": list(self.regime_models.keys()),
            "metrics": self.regime_metrics,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "saved_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return saved_paths

    def load(
        self,
        base_path: str = PRODUCTION_MODELS_DIR,
        instrument: str = "EUR_USD"
    ) -> List[str]:
        """
        Load all available regime models from disk.

        Args:
            base_path: Base directory
            instrument: Currency pair name

        Returns:
            List of loaded regime names
        """
        load_dir = Path(base_path) / instrument
        loaded_regimes = []

        for regime_name in REGIME_NAMES_LIST:
            model_path = load_dir / f"lgbm_regime_{regime_name}.pkl"

            if not model_path.exists():
                continue

            with open(model_path, "rb") as f:
                data = pickle.load(f)

            self.regime_models[regime_name] = data["model"]
            self.regime_scalers[regime_name] = data["scaler"]
            self.regime_metrics[regime_name] = data.get("metrics", {})

            if self.feature_names is None:
                self.feature_names = data.get("feature_names")
                self.n_features = data.get("n_features")

            loaded_regimes.append(regime_name)
            logger.info(f"Loaded {regime_name} model from {model_path}")

        if loaded_regimes:
            self.is_trained = True

        return loaded_regimes

    def get_model_for_regime(self, regime: str) -> Optional[Any]:
        """Get the LightGBM model for a specific regime."""
        return self.regime_models.get(regime)

    def list_trained_regimes(self) -> List[str]:
        """List all trained regime names."""
        return list(self.regime_models.keys())


class LightGBMMomentumTrainer(BaseTrainer):
    """
    LightGBM model for momentum analysis (replaces XGBoost for Phase 4 joint training).

    Input: Lagged returns + spread dynamics
    Output:
        - momentum_score (0-1): How fast price is moving
        - acceleration (bool): Is momentum growing?

    Advantages over XGBoost:
    - Faster training with categorical feature support
    - Better memory efficiency for large datasets
    - Supports init_model for warm-start fine-tuning
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.momentum_model = None
        self.accel_model = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.momentum_norm_factor = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        momentum_norm_factor: Optional[float] = None,
        init_momentum_model: Optional[Any] = None,
        init_accel_model: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train LightGBM for momentum analysis (2 models) with optional warm-start.

        Args:
            init_momentum_model: Pre-trained momentum model for fine-tuning
            init_accel_model: Pre-trained acceleration model for fine-tuning
        """
        try:
            import lightgbm  # noqa: F401 - availability check
        except ImportError:
            raise ImportError(LGBM_NOT_INSTALLED_ERROR)

        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        self.momentum_norm_factor = momentum_norm_factor

        logger.info("Training LightGBM (Momentum)...")

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Generate feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # Convert to DataFrame for LightGBM
        x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
        x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)

        # Split targets: y[:, 0] = momentum_score, y[:, 1] = acceleration
        # Use .copy() to avoid LightGBM warnings about sliced arrays doubling memory
        y_train_momentum = y_train[:, 0].copy()
        y_train_accel = y_train[:, 1].astype(int).copy()
        y_val_momentum = y_val[:, 0].copy()
        y_val_accel = y_val[:, 1].astype(int).copy()

        # Train momentum regressor
        self.momentum_model = _create_lgbm_regressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
        )

        if self.momentum_model is None:
            raise RuntimeError("Could not create LightGBM regressor")

        # Warm-start from init_model if provided (full re-optimization)
        self.momentum_model.fit(
            x_train_df,
            y_train_momentum,
            eval_set=[(x_val_df, y_val_momentum)],
            init_model=init_momentum_model,
        )

        # Train acceleration classifier
        self.accel_model = _create_lgbm_classifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
        )

        if self.accel_model is None:
            raise RuntimeError("Could not create LightGBM classifier")

        self.accel_model.fit(
            x_train_df,
            y_train_accel,
            eval_set=[(x_val_df, y_val_accel)],
            init_model=init_accel_model,
        )

        self.is_trained = True

        # Calculate metrics
        momentum_pred = self.momentum_model.predict(x_val_df)
        accel_pred = self.accel_model.predict(x_val_df)

        momentum_mae = float(np.mean(np.abs(momentum_pred - y_val_momentum)))
        accel_acc = float(np.mean(accel_pred == y_val_accel))

        self.metrics = {
            "momentum_mae": momentum_mae,
            "acceleration_accuracy": accel_acc,
            "model_type": "lightgbm",
        }

        logger.info(
            f"LightGBM Momentum trained: momentum_mae={momentum_mae:.4f}, accel_acc={accel_acc:.4f}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict momentum score and acceleration."""
        import pandas as pd

        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # Convert to DataFrame
        x_df = pd.DataFrame(x_last, columns=self.feature_names)

        momentum = float(self.momentum_model.predict(x_df)[0])
        acceleration = bool(self.accel_model.predict(x_df)[0])

        # Clamp momentum to 0-1
        momentum = max(0.0, min(1.0, momentum))

        return {
            "momentum": momentum,
            "acceleration": acceleration,
        }

    def save(self, path: str) -> None:
        """Save LightGBM momentum models with version metadata."""
        import sklearn

        try:
            import lightgbm

            lgbm_version = lightgbm.__version__
        except ImportError:
            lgbm_version = "unknown"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "momentum_model": self.momentum_model,
            "accel_model": self.accel_model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__ if self.config else {},
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "momentum_norm_factor": self.momentum_norm_factor,
            "model_type": "lightgbm_momentum",
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgbm_version,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"LightGBM Momentum saved to {path}")

    def load(self, path: str) -> None:
        """Load LightGBM momentum models."""
        import warnings

        warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.momentum_model = data["momentum_model"]
        self.accel_model = data["accel_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.momentum_norm_factor = data.get("momentum_norm_factor")
        self.is_trained = True

        logger.info(f"LightGBM Momentum loaded from {path}")

    def get_booster(self, model_type: str = "momentum") -> Any:
        """Get underlying booster for warm-start fine-tuning."""
        if model_type == "momentum":
            return self.momentum_model.booster_ if self.momentum_model else None
        else:
            return self.accel_model.booster_ if self.accel_model else None


class LightGBMRiskTrainer(BaseTrainer):
    """
    LightGBM model for risk assessment (replaces RandomForest for Phase 4 joint training).

    Input: ATR, historical drawdowns, streak patterns
    Output:
        - expected_drawdown_pct: Max adverse excursion in next N bars
        - streak_prob: Probability losing streak continues

    Advantages over RandomForest:
    - Faster training especially on large datasets
    - Better handling of feature interactions
    - Supports init_model for warm-start fine-tuning
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.drawdown_model = None
        self.streak_model = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        init_drawdown_model: Optional[Any] = None,
        init_streak_model: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train LightGBM for risk assessment (2 models) with optional warm-start.

        Args:
            init_drawdown_model: Pre-trained drawdown model for fine-tuning
            init_streak_model: Pre-trained streak model for fine-tuning
        """
        try:
            import lightgbm  # noqa: F401 - availability check
        except ImportError:
            raise ImportError(LGBM_NOT_INSTALLED_ERROR)

        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        logger.info("Training LightGBM (Risk)...")

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Generate feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # Convert to DataFrame for LightGBM
        x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
        x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)

        # Split targets: y[:, 0] = drawdown_pct, y[:, 1] = streak_prob
        # Use .copy() to avoid LightGBM warnings about sliced arrays doubling memory
        y_train_drawdown = y_train[:, 0].copy()
        y_train_streak = y_train[:, 1].copy()
        y_val_drawdown = y_val[:, 0].copy()
        y_val_streak = y_val[:, 1].copy()

        # Train drawdown regressor
        self.drawdown_model = _create_lgbm_regressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
        )

        if self.drawdown_model is None:
            raise RuntimeError("Could not create LightGBM regressor for drawdown")

        self.drawdown_model.fit(
            x_train_df,
            y_train_drawdown,
            eval_set=[(x_val_df, y_val_drawdown)],
            init_model=init_drawdown_model,
        )

        # Train streak probability regressor
        self.streak_model = _create_lgbm_regressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
        )

        if self.streak_model is None:
            raise RuntimeError("Could not create LightGBM regressor for streak")

        self.streak_model.fit(
            x_train_df,
            y_train_streak,
            eval_set=[(x_val_df, y_val_streak)],
            init_model=init_streak_model,
        )

        self.is_trained = True

        # Calculate metrics
        drawdown_pred = self.drawdown_model.predict(x_val_df)
        streak_pred = self.streak_model.predict(x_val_df)

        # Clip predictions to valid range
        drawdown_pred = np.clip(drawdown_pred, 0, 0.10)
        streak_pred = np.clip(streak_pred, 0, 1.0)

        drawdown_mae = float(np.mean(np.abs(drawdown_pred - y_val_drawdown)))
        streak_mae = float(np.mean(np.abs(streak_pred - y_val_streak)))

        # Convert to basis points for display
        drawdown_mae_bps = drawdown_mae * 10000

        self.metrics = {
            "drawdown_mae_pct": drawdown_mae,
            "drawdown_mae_bps": drawdown_mae_bps,
            "streak_prob_mae": streak_mae,
            "model_type": "lightgbm",
        }

        logger.info(
            f"LightGBM Risk trained: drawdown_mae={drawdown_mae_bps:.1f} bps, streak_mae={streak_mae:.4f}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict expected drawdown and streak probability."""
        import pandas as pd

        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # Convert to DataFrame
        x_df = pd.DataFrame(x_last, columns=self.feature_names)

        expected_drawdown_pct = float(self.drawdown_model.predict(x_df)[0])
        streak_prob = float(self.streak_model.predict(x_df)[0])

        # Clamp values
        expected_drawdown_pct = max(0.0, min(0.10, expected_drawdown_pct))
        streak_prob = max(0.0, min(1.0, streak_prob))

        return {
            "expected_drawdown_pct": expected_drawdown_pct,
            "expected_drawdown_pips": expected_drawdown_pct * 10000,  # Legacy key
            "streak_prob": streak_prob,
        }

    def save(self, path: str) -> None:
        """Save LightGBM risk models with version metadata."""
        import sklearn

        try:
            import lightgbm

            lgbm_version = lightgbm.__version__
        except ImportError:
            lgbm_version = "unknown"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "drawdown_model": self.drawdown_model,
            "streak_model": self.streak_model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__ if self.config else {},
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "model_type": "lightgbm_risk",
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgbm_version,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"LightGBM Risk saved to {path}")

    def load(self, path: str) -> None:
        """Load LightGBM risk models."""
        import warnings

        warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.drawdown_model = data["drawdown_model"]
        self.streak_model = data["streak_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        logger.info(f"LightGBM Risk loaded from {path}")

    def get_booster(self, model_type: str = "drawdown") -> Any:
        """Get underlying booster for warm-start fine-tuning."""
        if model_type == "drawdown":
            return self.drawdown_model.booster_ if self.drawdown_model else None
        else:
            return self.streak_model.booster_ if self.streak_model else None
