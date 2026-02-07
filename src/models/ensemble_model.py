"""
Ensemble Model for FX Trading.

Combines multiple diverse models for better prediction accuracy:
1. TCN (Temporal Convolutional Network) - Deep learning, sequential patterns
2. XGBoost - Gradient boosting, non-linear interactions
3. Random Forest - Bagging, different error patterns
4. Ridge/Logistic - Linear baseline, regularization

The ensemble uses stacking: base models' predictions become features
for a meta-model that learns optimal combination weights.

Research shows 2-4 diverse models provide significant improvement
over single models, with diminishing returns beyond that.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check available libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble model."""
    
    # Which models to include
    use_tcn: bool = True
    use_xgboost: bool = True
    use_random_forest: bool = True
    use_ridge: bool = True
    
    # XGBoost params
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    
    # Random Forest params
    rf_n_estimators: int = 200
    rf_max_depth: int = 10
    rf_min_samples_leaf: int = 20
    
    # Stacking meta-model params
    stack_method: str = "xgboost"  # "xgboost", "logistic", "average"
    stack_n_estimators: int = 100
    stack_max_depth: int = 3
    
    # Early stopping
    early_stopping_rounds: int = 30


class BaseModelWrapper:
    """Base class for model wrappers in the ensemble."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class XGBoostWrapper(BaseModelWrapper):
    """XGBoost model wrapper."""
    
    def __init__(self, config: EnsembleConfig):
        super().__init__("xgboost")
        self.config = config
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        # Flatten sequences if needed
        X_flat = self._flatten(X)
        X_val_flat = self._flatten(X_val) if X_val is not None else None
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )
        
        fit_kwargs = {}
        if X_val_flat is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val_flat, y_val)]
            fit_kwargs["verbose"] = False
        
        self.model.fit(X_flat, y, **fit_kwargs)
        self.is_fitted = True
        
        # Calculate accuracy
        train_acc = float(np.mean((self.model.predict(X_flat) > 0.5) == y))
        val_acc = float(np.mean((self.model.predict(X_val_flat) > 0.5) == y_val)) if X_val_flat is not None else None
        
        return {"train_acc": train_acc, "val_acc": val_acc}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        return self.model.predict_proba(X_flat)[:, 1]
    
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X is None:
            return None
        if X.ndim == 3:
            # Take last 5 timesteps + aggregates
            batch_size, seq_len, n_features = X.shape
            use_steps = min(5, seq_len)
            X_recent = X[:, -use_steps:, :].reshape(batch_size, -1)
            X_mean = X.mean(axis=1)
            X_std = X.std(axis=1)
            X_last = X[:, -1, :]
            return np.concatenate([X_recent, X_mean, X_std, X_last], axis=1)
        return X


class RandomForestWrapper(BaseModelWrapper):
    """Random Forest model wrapper."""
    
    def __init__(self, config: EnsembleConfig):
        super().__init__("random_forest")
        self.config = config
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        X_flat = self._flatten(X)
        X_val_flat = self._flatten(X_val) if X_val is not None else None
        
        self.model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        
        self.model.fit(X_flat, y)
        self.is_fitted = True
        
        train_acc = float(np.mean(self.model.predict(X_flat) == y))
        val_acc = float(np.mean(self.model.predict(X_val_flat) == y_val)) if X_val_flat is not None else None
        
        return {"train_acc": train_acc, "val_acc": val_acc}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        return self.model.predict_proba(X_flat)[:, 1]
    
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X is None:
            return None
        if X.ndim == 3:
            X_mean = X.mean(axis=1)
            X_std = X.std(axis=1)
            X_last = X[:, -1, :]
            return np.concatenate([X_mean, X_std, X_last], axis=1)
        return X


class RidgeWrapper(BaseModelWrapper):
    """Ridge/Logistic regression wrapper with FULL sequence utilization.
    
    FIXED: Now uses ALL timesteps instead of just the last one.
    Extracts rich temporal features:
    - Last K timesteps (recent context)
    - Rolling statistics (mean, std, min, max) over multiple windows
    - Lag features for momentum detection
    - Trend indicators (slope, acceleration)
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__("ridge")
        self.config = config
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        X_flat = self._flatten(X)
        X_val_flat = self._flatten(X_val) if X_val is not None else None
        
        # Scale features for linear model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)
        X_val_scaled = self.scaler.transform(X_val_flat) if X_val_flat is not None else None
        
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=0.1,  # Regularization
        )
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        train_acc = float(np.mean(self.model.predict(X_scaled) == y))
        val_acc = float(np.mean(self.model.predict(X_val_scaled) == y_val)) if X_val_scaled is not None else None
        
        return {"train_acc": train_acc, "val_acc": val_acc}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Extract rich temporal features from full sequence.
        
        IMPROVED: Uses ALL timesteps with multiple feature types:
        1. Last 5 timesteps (recent values) - captures recent patterns
        2. Rolling mean/std for windows [5, 10, 20] - trend & volatility
        3. Diff features (momentum) - velocity of change
        4. Slope features (acceleration) - rate of momentum change
        5. Min/max positions - extreme value locations
        """
        if X is None:
            return None
        if X.ndim != 3:
            return X
            
        batch_size, seq_len, n_features = X.shape
        features_list = []
        
        # 1. Last 5 timesteps (flattened) - recent context
        use_steps = min(5, seq_len)
        X_recent = X[:, -use_steps:, :].reshape(batch_size, -1)
        features_list.append(X_recent)
        
        # 2. Rolling statistics over multiple windows
        for window in [5, 10, 20]:
            if seq_len >= window:
                # Mean of last `window` timesteps
                X_window = X[:, -window:, :]
                features_list.append(X_window.mean(axis=1))
                features_list.append(X_window.std(axis=1))
                features_list.append(X_window.min(axis=1))
                features_list.append(X_window.max(axis=1))
        
        # 3. Global statistics (full sequence)
        features_list.append(X.mean(axis=1))  # Overall mean
        features_list.append(X.std(axis=1))   # Overall volatility
        
        # 4. First-order differences (momentum)
        if seq_len >= 2:
            X_diff = np.diff(X, axis=1)
            features_list.append(X_diff[:, -1, :])  # Most recent change
            features_list.append(X_diff.mean(axis=1))  # Average momentum
        
        # 5. Second-order differences (acceleration)
        if seq_len >= 3:
            X_diff2 = np.diff(X, n=2, axis=1)
            features_list.append(X_diff2[:, -1, :])  # Most recent acceleration
            features_list.append(X_diff2.mean(axis=1))  # Average acceleration
        
        # 6. Trend features: slope from first to last timestep
        if seq_len >= 2:
            slope = (X[:, -1, :] - X[:, 0, :]) / seq_len
            features_list.append(slope)
        
        # 7. Position of min/max within sequence (normalized to [0,1])
        argmax_pos = np.argmax(X, axis=1) / seq_len  # Where was max?
        argmin_pos = np.argmin(X, axis=1) / seq_len  # Where was min?
        features_list.append(argmax_pos)
        features_list.append(argmin_pos)
        
        # Concatenate all features
        return np.concatenate(features_list, axis=1)


# Check if LightGBM is available
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LightGBMWrapper(BaseModelWrapper):
    """LightGBM wrapper - BETTER than Ridge for sequence classification.
    
    LightGBM handles feature interactions and non-linear patterns that
    Ridge misses. Uses the same rich temporal features as RidgeWrapper.
    
    Benefits over Ridge:
    - Handles feature interactions automatically
    - Non-linear decision boundaries
    - Built-in feature importance
    - Faster training on large datasets
    - Better handling of high-dimensional features
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__("lightgbm")
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_importance_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        X_flat = self._flatten(X)
        X_val_flat = self._flatten(X_val) if X_val is not None else None
        
        # Scale features (helps convergence)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)
        X_val_scaled = self.scaler.transform(X_val_flat) if X_val_flat is not None else None
        
        # LightGBM parameters optimized for FX direction prediction
        self.model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            n_estimators=200,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        
        # Fit with early stopping if validation data available
        if X_val_scaled is not None and y_val is not None:
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
        else:
            self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        train_acc = float(np.mean(self.model.predict(X_scaled) == y))
        val_acc = float(np.mean(self.model.predict(X_val_scaled) == y_val)) if X_val_scaled is not None and y_val is not None else None
        
        return {"train_acc": train_acc, "val_acc": val_acc, "n_estimators": self.model.n_estimators_}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LightGBM model not fitted")
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importances for interpretability."""
        return self.feature_importance_
    
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Extract rich temporal features - same as RidgeWrapper for consistency."""
        if X is None:
            return None
        if X.ndim != 3:
            return X
            
        batch_size, seq_len, n_features = X.shape
        features_list = []
        
        # 1. Last 5 timesteps (flattened) - recent context
        use_steps = min(5, seq_len)
        X_recent = X[:, -use_steps:, :].reshape(batch_size, -1)
        features_list.append(X_recent)
        
        # 2. Rolling statistics over multiple windows
        for window in [5, 10, 20]:
            if seq_len >= window:
                X_window = X[:, -window:, :]
                features_list.append(X_window.mean(axis=1))
                features_list.append(X_window.std(axis=1))
                features_list.append(X_window.min(axis=1))
                features_list.append(X_window.max(axis=1))
        
        # 3. Global statistics
        features_list.append(X.mean(axis=1))
        features_list.append(X.std(axis=1))
        
        # 4. First-order differences (momentum)
        if seq_len >= 2:
            X_diff = np.diff(X, axis=1)
            features_list.append(X_diff[:, -1, :])
            features_list.append(X_diff.mean(axis=1))
        
        # 5. Second-order differences (acceleration)
        if seq_len >= 3:
            X_diff2 = np.diff(X, n=2, axis=1)
            features_list.append(X_diff2[:, -1, :])
            features_list.append(X_diff2.mean(axis=1))
        
        # 6. Trend features
        if seq_len >= 2:
            slope = (X[:, -1, :] - X[:, 0, :]) / seq_len
            features_list.append(slope)
        
        # 7. Position of min/max
        argmax_pos = np.argmax(X, axis=1) / seq_len
        argmin_pos = np.argmin(X, axis=1) / seq_len
        features_list.append(argmax_pos)
        features_list.append(argmin_pos)
        
        return np.concatenate(features_list, axis=1)


class TCNWrapper(BaseModelWrapper):
    """TCN model wrapper (uses existing Keras model)."""
    
    def __init__(self, keras_model=None):
        super().__init__("tcn")
        self.model = keras_model
        self.is_fitted = keras_model is not None
    
    def set_model(self, model):
        """Set the Keras TCN model after training."""
        self.model = model
        self.is_fitted = True
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        # TCN is trained separately via the main training pipeline
        # This is a placeholder - the model should be set via set_model()
        raise NotImplementedError("TCN should be trained via main pipeline and set via set_model()")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TCN model not set")
        
        preds = self.model.predict(X, verbose=0)
        
        if isinstance(preds, dict):
            return np.asarray(preds.get("direction", preds.get("output_0"))).flatten()
        return np.asarray(preds).flatten()


class EnsembleModel:
    """
    4-Model Ensemble for FX Direction Prediction.
    
    Uses stacking: base models' predictions become features for a meta-model.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.base_models: Dict[str, BaseModelWrapper] = {}
        self.stacker = None
        self.is_fitted = False
        self._base_model_metrics: Dict[str, Dict] = {}
    
    def _init_base_models(self, tcn_model=None):
        """Initialize base models based on config."""
        self.base_models = {}
        
        if self.config.use_tcn and tcn_model is not None:
            wrapper = TCNWrapper(tcn_model)
            self.base_models["tcn"] = wrapper
        
        if self.config.use_xgboost and XGBOOST_AVAILABLE:
            self.base_models["xgboost"] = XGBoostWrapper(self.config)
        
        if self.config.use_random_forest and SKLEARN_AVAILABLE:
            self.base_models["random_forest"] = RandomForestWrapper(self.config)
        
        # Prefer LightGBM over Ridge when available (better for non-linear patterns)
        if self.config.use_ridge:
            if LIGHTGBM_AVAILABLE:
                self.base_models["lightgbm"] = LightGBMWrapper(self.config)
                logger.info("Using LightGBM instead of Ridge (better accuracy)")
            elif SKLEARN_AVAILABLE:
                self.base_models["ridge"] = RidgeWrapper(self.config)
        
        logger.info(f"Initialized {len(self.base_models)} base models: {list(self.base_models.keys())}")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tcn_model=None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the ensemble.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            tcn_model: Pre-trained TCN Keras model (optional)
            verbose: Print progress
        
        Returns:
            Training metrics dict
        """
        self._init_base_models(tcn_model)
        
        if len(self.base_models) < 2:
            raise ValueError(f"Ensemble requires at least 2 models, got {len(self.base_models)}")
        
        # Ensure binary labels
        y_train = (np.asarray(y_train).flatten() > 0.5).astype(np.int32)
        y_val = (np.asarray(y_val).flatten() > 0.5).astype(np.int32)
        
        # Train each base model (except TCN which is pre-trained)
        for name, model in self.base_models.items():
            if name == "tcn":
                # TCN already trained
                if verbose:
                    logger.info("Using pre-trained TCN model")
                continue
            
            if verbose:
                logger.info(f"Training {name}...")
            
            try:
                metrics = model.fit(X_train, y_train, X_val, y_val)
                self._base_model_metrics[name] = metrics
                
                if verbose:
                    logger.info(f"  {name}: train_acc={metrics['train_acc']:.1%}, val_acc={metrics.get('val_acc', 0):.1%}")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                del self.base_models[name]
        
        # Get base model predictions for stacking
        train_preds = self._get_base_predictions(X_train)
        val_preds = self._get_base_predictions(X_val)
        
        if verbose:
            logger.info(f"Stacking {train_preds.shape[1]} base model predictions...")
        
        # Train stacking meta-model
        if self.config.stack_method == "xgboost" and XGBOOST_AVAILABLE:
            self.stacker = xgb.XGBClassifier(
                n_estimators=self.config.stack_n_estimators,
                max_depth=self.config.stack_max_depth,
                learning_rate=0.1,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
                random_state=42,
            )
            self.stacker.fit(
                train_preds, y_train,
                eval_set=[(val_preds, y_val)],
                verbose=False,
            )
        elif self.config.stack_method == "logistic" and SKLEARN_AVAILABLE:
            self.stacker = LogisticRegression(max_iter=1000, random_state=42)
            self.stacker.fit(train_preds, y_train)
        else:
            # Simple averaging (no stacker)
            self.stacker = None
        
        self.is_fitted = True
        
        # Calculate ensemble metrics
        ensemble_train_preds = self.predict_proba(X_train)
        ensemble_val_preds = self.predict_proba(X_val)
        
        train_acc = float(np.mean((ensemble_train_preds > 0.5) == y_train))
        val_acc = float(np.mean((ensemble_val_preds > 0.5) == y_val))
        
        if verbose:
            logger.info(f"Ensemble: train_acc={train_acc:.1%}, val_acc={val_acc:.1%}")
        
        return {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "base_models": list(self.base_models.keys()),
            "base_model_metrics": self._base_model_metrics,
            "stack_method": self.config.stack_method,
        }
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models."""
        preds = []
        for name, model in self.base_models.items():
            if model.is_fitted:
                try:
                    p = model.predict_proba(X)
                    preds.append(p.reshape(-1, 1))
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}: {e}")
        
        return np.concatenate(preds, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")
        
        base_preds = self._get_base_predictions(X)
        
        if self.stacker is not None:
            return self.stacker.predict_proba(base_preds)[:, 1]
        else:
            # Simple average
            return base_preds.mean(axis=1)
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict with Keras-like interface."""
        proba = self.predict_proba(X)
        return {
            "direction": proba,
            "confidence": np.abs(proba - 0.5) * 2,  # Distance from 0.5, scaled to [0, 1]
        }
    
    def get_base_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual predictions from each base model."""
        result = {}
        for name, model in self.base_models.items():
            if model.is_fitted:
                result[name] = model.predict_proba(X)
        return result
    
    def save(self, path: Union[str, Path]) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save non-Keras models
        save_dict = {
            "config": self.config.__dict__,
            "base_models": {},
            "stacker": self.stacker,
            "base_model_metrics": self._base_model_metrics,
            "is_fitted": self.is_fitted,
        }
        
        for name, model in self.base_models.items():
            if name != "tcn":  # TCN saved separately as Keras model
                save_dict["base_models"][name] = {
                    "model": model.model,
                    "scaler": getattr(model, "scaler", None),
                }
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved ensemble to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], tcn_model=None) -> "EnsembleModel":
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        config = EnsembleConfig(**save_dict["config"])
        ensemble = cls(config)
        ensemble.stacker = save_dict["stacker"]
        ensemble._base_model_metrics = save_dict["base_model_metrics"]
        ensemble.is_fitted = save_dict["is_fitted"]
        
        # Restore base models
        for name, data in save_dict["base_models"].items():
            if name == "xgboost":
                wrapper = XGBoostWrapper(config)
                wrapper.model = data["model"]
                wrapper.is_fitted = True
            elif name == "random_forest":
                wrapper = RandomForestWrapper(config)
                wrapper.model = data["model"]
                wrapper.is_fitted = True
            elif name == "ridge":
                wrapper = RidgeWrapper(config)
                wrapper.model = data["model"]
                wrapper.scaler = data["scaler"]
                wrapper.is_fitted = True
            else:
                continue
            ensemble.base_models[name] = wrapper
        
        # Add TCN if provided
        if tcn_model is not None:
            wrapper = TCNWrapper(tcn_model)
            ensemble.base_models["tcn"] = wrapper
        
        return ensemble


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tcn_model=None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[EnsembleModel, Dict[str, Any]]:
    """
    Train a 4-model ensemble.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        tcn_model: Optional pre-trained TCN model
        config: Optional config dict
        verbose: Print progress
    
    Returns:
        Tuple of (trained ensemble, metrics dict)
    """
    # Build config
    ensemble_config = EnsembleConfig()
    if config:
        for key, value in config.items():
            if hasattr(ensemble_config, key):
                setattr(ensemble_config, key, value)
    
    # Create and train ensemble
    ensemble = EnsembleModel(ensemble_config)
    metrics = ensemble.fit(
        X_train, y_train,
        X_val, y_val,
        tcn_model=tcn_model,
        verbose=verbose,
    )
    
    return ensemble, metrics


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Ensemble Model Test")
    print("-" * 40)
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    seq_len = 60
    n_features = 100
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = (np.random.rand(n_samples) > 0.5).astype(np.float32)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Train ensemble (without TCN for testing)
    config = EnsembleConfig(use_tcn=False)
    ensemble = EnsembleModel(config)
    metrics = ensemble.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    print(f"\nEnsemble metrics: {metrics}")
    
    # Test predictions
    preds = ensemble.predict(X_val[:5])
    print("\nSample predictions:")
    print(f"  Direction: {preds['direction']}")
    print(f"  Confidence: {preds['confidence']}")
    
    # Base model predictions
    base_preds = ensemble.get_base_model_predictions(X_val[:5])
    print("\nBase model predictions:")
    for name, p in base_preds.items():
        print(f"  {name}: {p}")
    
    print("\nEnsemble test complete!")
