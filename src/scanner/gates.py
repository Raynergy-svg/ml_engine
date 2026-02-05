"""
Gate Evaluator Module.

Loads and evaluates trading gates using:
- CatBoost (primary) - best for time-series with ordered boosting
- XGBoost (fallback) - if CatBoost unavailable
- Transformer (direction) - deep learning direction prediction
- Meta-labeler (optional) - trade success probability
- TCN Forward Volatility (REQUIRED) - PREDICTS FUTURE volatility regime (FIRST GATE)

Gate models:
1. TCN Forward Volatility Gate: REQUIRED FIRST - predicts FUTURE volatility regime
   - QUIET_NEXT (0): BLOCK - expect low volatility next 48 bars
   - STABLE_NEXT (1): ALLOW - expect moderate, predictable volatility
   - ACTIVE_NEXT (2): ALLOW - expect high, tradeable volatility (best for entries)
   - EXTREME_NEXT (3): BLOCK - expect extreme volatility (news events, high risk)
2. Momentum Gate: CatBoost/XGBoost predicts momentum percentile (0-1)
3. Confidence Gate: Ridge model predicts ADX-based confidence (0-1 normalized)
4. Risk Gate: RandomForest predicts expected drawdown (0-1)
5. Transformer Gate: Direction probability (>= 0.60 for trade)
6. Meta-labeler Gate: Trade success probability (>= 0.55)

IMPORTANT: 
- ALL confidence values are 0-1 scale internally
- Scanner uses JOINT-TRAINED models exclusively from trained_data/models/joint/
- Models must be trained with: python main.py train-joint
- If TCN model missing, scan FAILS with ScannerError
- TCN predicts FUTURE regime (48-bar lookahead), not current regime

Feature Alignment:
Uses compute_normalized_features() from modular_data_loaders to ensure
consistent feature computation between scanner and inference pipeline.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import normalized feature computation for alignment with inference
try:
    from src.core.modular_data_loaders import (
        compute_normalized_features,
        get_normalized_feature_names,
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    compute_normalized_features = None
    get_normalized_feature_names = None

# Import confidence calibration for scanner accuracy
try:
    from src.risk.confidence_calibration import (
        ConfidenceCalibrator,
        CalibrationConfig,
    )
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    ConfidenceCalibrator = None
    CalibrationConfig = None

logger = logging.getLogger(__name__)

# Joint model directory constant
JOINT_MODEL_DIR = "joint"


class ScannerError(Exception):
    """Raised when scanner encounters a critical error (e.g., missing required model)."""
    pass


def _normalize_confidence(value: float, from_scale: int = 100) -> float:
    """Normalize confidence from legacy scale to 0-1.
    
    Args:
        value: Raw value (e.g., 50 on 0-100 scale)
        from_scale: Source scale maximum (default 100)
        
    Returns:
        Normalized value in 0-1 range
    """
    if from_scale <= 0:
        return 0.0
    normalized = value / from_scale
    return max(0.0, min(1.0, normalized))


def _format_confidence_pct(confidence: float) -> str:
    """Format confidence (0-1) as percentage string.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted string like "65%"
    """
    return f"{confidence * 100:.0f}%"


class GateEvaluator:
    """Evaluates trading gates using CatBoost primary + XGBoost fallback.
    
    IMPORTANT: 
    - Scanner uses JOINT-TRAINED models exclusively from trained_data/models/joint/
    - ALL confidence values are normalized to 0-1 scale
    - TCN Volatility Regime is REQUIRED and evaluated FIRST
    
    This class handles:
    - Loading TCN Volatility Regime model (REQUIRED FIRST GATE)
    - Loading CatBoost momentum model (primary)
    - Falling back to XGBoost if CatBoost unavailable
    - Loading Ridge confidence and RF risk models
    - Loading Transformer direction model (optional)
    - Loading Meta-labeler model (optional)
    - Evaluating all gates and returning pass/fail status
    
    Feature Alignment:
    Uses compute_normalized_features() for consistent feature computation.
    """
    
    # Momentum gate thresholds (0-1 scale)
    ACCELERATION_THRESHOLD: float = 0.5
    NEUTRAL_MOMENTUM_SCORE: float = 0.5
    
    # Transformer gate threshold (0-1 scale)
    TRANSFORMER_THRESHOLD: float = 0.60
    
    # Meta-labeler threshold (0-1 scale)
    META_LABELER_THRESHOLD: float = 0.55
    
    # TCN Forward Volatility Regime constants (predicts FUTURE regime)
    # QUIET_NEXT = expect low volatility - BLOCK trades
    # STABLE_NEXT = expect moderate volatility - ALLOW trades
    # ACTIVE_NEXT = expect high volatility - ALLOW trades (best entry conditions)
    # EXTREME_NEXT = expect extreme volatility - BLOCK trades (news events)
    REGIME_QUIET_NEXT = 0
    REGIME_STABLE_NEXT = 1
    REGIME_ACTIVE_NEXT = 2
    REGIME_EXTREME_NEXT = 3
    REGIME_NAMES = ["QUIET_NEXT", "STABLE_NEXT", "ACTIVE_NEXT", "EXTREME_NEXT"]
    
    # Legacy aliases for compatibility
    REGIME_LOW = 0
    REGIME_NORMAL = 1
    REGIME_HIGH = 2
    REGIME_EXTREME = 3
    
    # Gate logic: ALLOW STABLE_NEXT (1) and ACTIVE_NEXT (2), BLOCK QUIET_NEXT (0) and EXTREME_NEXT (3)
    # This is different from old logic which was just >= 2
    OPPORTUNITY_REGIMES = {1, 2}  # STABLE_NEXT and ACTIVE_NEXT
    
    def __init__(self, model_dir: Path, use_joint_only: bool = True):
        """Initialize gate evaluator.
        
        Args:
            model_dir: Base path to model directory (e.g., trained_data/models)
            use_joint_only: If True, load ONLY from joint/ subdirectory (default for scanner)
        """
        self.base_model_dir = Path(model_dir)
        self.use_joint_only = use_joint_only
        
        # Set actual model directory based on joint-only mode
        if use_joint_only:
            self.model_dir = self.base_model_dir / JOINT_MODEL_DIR
        else:
            self.model_dir = self.base_model_dir
        
        # Model instances
        self._catboost_momentum = None
        self._lgbm_momentum = None     # LightGBM momentum (joint training)
        self._xgboost_momentum = None
        self._ridge_confidence = None
        self._rf_risk = None
        self._transformer = None
        self._meta_labeler = None
        self._tcn_volatility = None  # TCN Volatility Regime model (REQUIRED)
        
        # Ridge scaler for proper ADX scaling
        self._ridge_scaler = None
        
        # XGBoost metadata (loaded from dict wrapper)
        self._xgb_feature_names = None
        self._xgb_scaler = None
        
        # LightGBM metadata (loaded from dict wrapper)
        self._lgbm_feature_names = None
        self._lgbm_scaler = None
        
        # TCN metadata
        self._tcn_scaler = None
        self._tcn_feature_names = None
        
        # Track which model is active for momentum
        self._momentum_model_type: str = "none"
        
        # Model metadata
        self._feature_names: Optional[list] = None
        self._ridge_feature_names: Optional[list] = None
        
        # Track if models are loaded
        self._models_loaded = False
        
        # Config for EXTREME regime warnings
        self._extreme_regime_warning: bool = True
        
        # Confidence calibrator for transformer probability calibration
        self._calibrator: Optional['ConfidenceCalibrator'] = None
        self._use_calibration: bool = True  # Can be disabled via config
        
    @property
    def tcn_volatility_available(self) -> bool:
        """Check if TCN volatility regime model is loaded."""
        return self._tcn_volatility is not None
    
    @property
    def has_calibrator(self) -> bool:
        """Check if confidence calibrator is loaded and can calibrate probabilities."""
        return self._calibrator is not None and self._calibrator.is_fitted
    
    def set_use_calibration(self, use_calibration: bool) -> None:
        """Enable or disable calibration for transformer probabilities.
        
        Args:
            use_calibration: If True, apply calibration when available
        """
        self._use_calibration = use_calibration
    
    def calibrate_probability(self, raw_prob: float) -> Tuple[float, bool]:
        """Calibrate a raw probability using the fitted calibrator.
        
        Args:
            raw_prob: Raw probability from model (0-1)
            
        Returns:
            Tuple of (calibrated_prob, was_calibrated):
                - calibrated_prob: Calibrated probability if calibrator available, else raw
                - was_calibrated: True if calibration was applied
        """
        if not self._use_calibration or not self.has_calibrator:
            return raw_prob, False
        
        try:
            result = self._calibrator.calibrate_confidence(raw_prob)
            return result.calibrated_confidence, True
        except Exception as e:
            logger.debug(f"Calibration failed, using raw probability: {e}")
            return raw_prob, False
        
    def load_models(self, require_tcn: bool = True, extreme_regime_warning: bool = True) -> Dict[str, bool]:
        """Load all gate models from joint model directory.
        
        Args:
            require_tcn: If True, raise ScannerError if TCN volatility model not found
            extreme_regime_warning: If True, log warning when EXTREME regime detected
        
        Returns:
            Dict mapping model name to load success status
            
        Raises:
            ScannerError: If joint model directory doesn't exist or TCN required but missing
        """
        self._extreme_regime_warning = extreme_regime_warning
        
        # Validate joint model directory exists
        if self.use_joint_only and not self.model_dir.exists():
            raise ScannerError(
                f"Joint model directory not found: {self.model_dir}\n"
                f"Joint TCN model required. Run `python main.py train-joint` first."
            )
        
        status = {
            "tcn_volatility": False,  # REQUIRED FIRST GATE
            "momentum": False,
            "confidence": False,
            "risk": False,
        }
        
        # Load TCN Volatility Regime model FIRST (REQUIRED for scanner)
        status["tcn_volatility"] = self._load_tcn_volatility()
        
        # Check if TCN is required but missing - FAIL IMMEDIATELY
        if require_tcn and not status["tcn_volatility"]:
            raise ScannerError(
                f"Joint TCN model required. Run `python main.py train-joint` first.\n"
                f"Expected: {self.model_dir / 'tcn_volatility_regime.keras'}"
            )
        
        # Try CatBoost first for momentum (better for time-series)
        status["momentum"] = self._load_catboost_momentum()
        
        # Fallback to LightGBM (used by joint training)
        if not status["momentum"]:
            status["momentum"] = self._load_lgbm_momentum()
        
        # Fallback to XGBoost if LightGBM also failed
        if not status["momentum"]:
            status["momentum"] = self._load_xgboost_momentum()
        
        # Load other gate models
        status["confidence"] = self._load_ridge_confidence()
        status["risk"] = self._load_rf_risk()
        
        # Load optional advanced gates
        status["transformer"] = self._load_transformer()
        status["meta_labeler"] = self._load_meta_labeler()
        
        # Load confidence calibrator (optional - improves accuracy if available)
        status["calibrator"] = self._load_calibrator()
        
        loaded = [k for k, v in status.items() if v]
        if loaded:
            logger.debug(f"Gate models loaded: {', '.join(loaded)} (momentum: {self._momentum_model_type})")
        else:
            logger.warning("No gate models loaded - gates will be bypassed")
        
        self._models_loaded = True
        
        return status
    
    def _load_tcn_volatility(self) -> bool:
        """Load TCN Forward Volatility model (REQUIRED FIRST GATE for scanner).
        
        This model PREDICTS FUTURE volatility regime (48-bar lookahead):
        - QUIET_NEXT (0): BLOCK - expect low volatility, insufficient movement
        - STABLE_NEXT (1): ALLOW - expect moderate volatility, good for trades
        - ACTIVE_NEXT (2): ALLOW - expect high volatility, best for entries
        - EXTREME_NEXT (3): BLOCK - expect extreme volatility, news event risk
        
        The model may be dual-head (classification + regression) for regression fallback.
        
        Returns:
            True if loaded successfully
            
        Raises:
            ScannerError: If model file exists but fails to load
        """
        model_path = self.model_dir / "tcn_volatility_regime.keras"
        meta_path = self.model_dir / "tcn_volatility_regime.meta.pkl"
        
        if not model_path.exists():
            logger.warning(f"TCN Forward Volatility model not found at {model_path}")
            return False
        
        try:
            import keras
            
            # CRITICAL: Import custom classes BEFORE loading model
            # These are decorated with @register_keras_serializable() 
            from src.models.tensorflow_models import (
                TCNVolatilityDualHead,
                DualHeadLoss,
            )
            
            # Load model (may be dual-head with classification + regression outputs)
            self._tcn_volatility = keras.models.load_model(str(model_path), compile=False)
            
            # Load metadata if available
            self._tcn_is_dual_head = False
            self._tcn_regression_thresholds = None
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    self._tcn_scaler = meta.get('scaler')
                    self._tcn_feature_names = meta.get('feature_names')
                    self._tcn_is_dual_head = meta.get('is_dual_head', False)
                    self._tcn_regression_thresholds = meta.get('regression_thresholds')
                    logger.debug(f"TCN metadata loaded: {meta.get('metrics', {})}")
                    if self._tcn_is_dual_head:
                        logger.debug("TCN model is dual-head (classification + regression fallback)")
            
            # Also try loading separate scaler file (newer format)
            scaler_path = self.model_dir / "tcn_feature_scaler.pkl"
            if scaler_path.exists() and self._tcn_scaler is None:
                with open(scaler_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                    self._tcn_scaler = scaler_data.get('scaler')
                    self._tcn_feature_names = scaler_data.get('feature_cols', self._tcn_feature_names)
                    logger.debug(f"TCN scaler loaded from {scaler_path}")
            
            logger.debug("✓ TCN Forward Volatility gate loaded")
            return True
            
        except ImportError:
            logger.warning("Keras not available for TCN Forward Volatility gate")
            return False
        except Exception as e:
            logger.error(f"Failed to load TCN Forward Volatility: {e}")
            return False
    
    def evaluate_volatility_regime(
        self,
        features: pd.DataFrame,
        seq_len: int = 60,
    ) -> Tuple[int, float, bool]:
        """Evaluate TCN Forward Volatility gate (FIRST GATE - evaluated before all others).
        
        This predicts FUTURE volatility regime (48-bar lookahead).
        Gate logic: ALLOW STABLE_NEXT (1) and ACTIVE_NEXT (2), BLOCK others.
        
        If model is dual-head and classification confidence < 60%, uses regression
        fallback to determine regime from predicted % volatility change.
        
        Args:
            features: DataFrame with engineered features
            seq_len: Sequence length for TCN input
            
        Returns:
            Tuple of (regime, confidence, is_opportunity):
                - regime: 0=QUIET_NEXT, 1=STABLE_NEXT, 2=ACTIVE_NEXT, 3=EXTREME_NEXT
                - confidence: Model confidence in prediction (0-1)
                - is_opportunity: True if regime is STABLE_NEXT or ACTIVE_NEXT
        """
        if self._tcn_volatility is None:
            logger.error("TCN Forward Volatility model not loaded - blocking ALL trades")
            return self.REGIME_QUIET_NEXT, 0.0, False
        
        try:
            import numpy as np
            
            # Prepare sequence input
            if len(features) < seq_len:
                logger.warning(f"Insufficient data for TCN ({len(features)} < {seq_len})")
                return self.REGIME_QUIET_NEXT, 0.0, False
            
            # Use feature names from metadata if available, otherwise use default volatility features
            if self._tcn_feature_names is not None:
                feature_cols = [f for f in self._tcn_feature_names if f in features.columns]
            else:
                # Default volatility-focused features - matches load_forward_volatility_data() output
                volatility_features = [
                    'atr_pct_5', 'atr_pct_10', 'atr_pct_20',
                    'volatility_5', 'volatility_10', 'volatility_20',
                    'returns_1', 'returns_5', 'returns_10',
                    'volume_ratio_10', 'volume_ratio_20',
                    # Time features for session patterns
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                    'session_london', 'session_ny', 'session_overlap',
                    'volume_ma_ratio',
                ]
                feature_cols = [f for f in volatility_features if f in features.columns]
            
            # Check feature count matches scaler expectations
            if self._tcn_scaler is not None:
                expected_features = getattr(self._tcn_scaler, 'n_features_in_', len(feature_cols))
                if len(feature_cols) != expected_features:
                    logger.warning(
                        f"TCN feature mismatch: got {len(feature_cols)}, expected {expected_features}. "
                        f"Using first {expected_features} features."
                    )
                    feature_cols = feature_cols[:expected_features]
            
            if len(feature_cols) < 5:
                logger.warning(f"Insufficient volatility features ({len(feature_cols)}) for TCN")
                return self.REGIME_QUIET_NEXT, 0.0, False
            
            # Get last seq_len rows with selected features only
            X = features[feature_cols].iloc[-seq_len:].values
            
            # Apply scaler if available
            if self._tcn_scaler is not None:
                X = self._tcn_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Add batch dimension
            X = np.expand_dims(X, axis=0)
            
            # Predict - may return dict, list, or single array depending on model architecture
            raw_output = self._tcn_volatility.predict(X, verbose=0)
            
            # Handle dual-head model output (Keras 3.x returns dict for multi-output)
            if isinstance(raw_output, dict):
                # Keras 3.x dict output: {'classification': ..., 'regression': ...}
                class_proba = raw_output.get('classification', raw_output.get('class_output'))
                regression_output = raw_output.get('regression', raw_output.get('reg_output'))
                
                if class_proba is not None and len(class_proba.shape) > 1 and class_proba.shape[-1] > 1:
                    regime = int(np.argmax(class_proba[0]))
                    confidence = float(class_proba[0][regime])
                else:
                    regime = int(round(float(class_proba[0][0]))) if class_proba is not None else 0
                    confidence = 0.5
                
                # Regression fallback: if classification confidence < 60%, use regression
                if confidence < 0.60 and regression_output is not None and self._tcn_regression_thresholds is not None:
                    vol_change_pred = float(regression_output[0][0])
                    thresholds = self._tcn_regression_thresholds
                    
                    # Map regression prediction to regime
                    if vol_change_pred < thresholds.get('quiet', -0.15):
                        regime = self.REGIME_QUIET_NEXT
                    elif vol_change_pred < thresholds.get('stable', 0.0):
                        regime = self.REGIME_STABLE_NEXT
                    elif vol_change_pred < thresholds.get('active', 0.25):
                        regime = self.REGIME_ACTIVE_NEXT
                    else:
                        regime = self.REGIME_EXTREME_NEXT
                    
                    logger.debug(f"TCN regression fallback: vol_change_pred={vol_change_pred:.3f} -> {self.REGIME_NAMES[regime]}")
            elif isinstance(raw_output, (list, tuple)) and len(raw_output) >= 2:
                # Dual-head: [classification_output, regression_output]
                class_proba = raw_output[0]
                regression_output = raw_output[1]
                
                if len(class_proba.shape) > 1 and class_proba.shape[-1] > 1:
                    regime = int(np.argmax(class_proba[0]))
                    confidence = float(class_proba[0][regime])
                else:
                    regime = int(round(float(class_proba[0][0])))
                    confidence = 0.5
                
                # Regression fallback: if classification confidence < 60%, use regression
                if confidence < 0.60 and self._tcn_regression_thresholds is not None:
                    vol_change_pred = float(regression_output[0][0])
                    thresholds = self._tcn_regression_thresholds
                    
                    # Map regression prediction to regime
                    if vol_change_pred < thresholds.get('quiet', -0.15):
                        regime = self.REGIME_QUIET_NEXT
                    elif vol_change_pred < thresholds.get('stable', 0.0):
                        regime = self.REGIME_STABLE_NEXT
                    elif vol_change_pred < thresholds.get('active', 0.25):
                        regime = self.REGIME_ACTIVE_NEXT
                    else:
                        regime = self.REGIME_EXTREME_NEXT
                    
                    logger.debug(f"TCN regression fallback: vol_change_pred={vol_change_pred:.3f} -> {self.REGIME_NAMES[regime]}")
            else:
                # Single-head classification output
                proba = raw_output
                if len(proba.shape) > 1 and proba.shape[-1] > 1:
                    regime = int(np.argmax(proba[0]))
                    confidence = float(proba[0][regime])
                else:
                    regime = int(round(float(proba[0][0])))
                    confidence = 0.5
            
            # Determine if trade is allowed (OPPORTUNITY = STABLE_NEXT or ACTIVE_NEXT)
            is_opportunity = regime in self.OPPORTUNITY_REGIMES
            regime_name = self.REGIME_NAMES[regime] if 0 <= regime <= 3 else "UNKNOWN"
            
            # Log EXTREME_NEXT regime warning if configured (high risk period ahead)
            if regime == self.REGIME_EXTREME_NEXT and self._extreme_regime_warning:
                logger.debug(
                    f"EXTREME_NEXT volatility predicted (confidence={_format_confidence_pct(confidence)}). "
                    f"High risk period - blocking trades."
                )
            
            # Log QUIET_NEXT reason
            if regime == self.REGIME_QUIET_NEXT:
                logger.debug(
                    f"QUIET_NEXT volatility predicted (confidence={_format_confidence_pct(confidence)}). "
                    f"Insufficient expected price movement - blocking trades."
                )
            
            logger.debug(
                f"TCN Forward Volatility: {regime_name} ({regime}), "
                f"confidence={_format_confidence_pct(confidence)}, is_opportunity={is_opportunity}"
            )
            
            return regime, confidence, is_opportunity
            
        except Exception as e:
            logger.error(f"TCN Volatility evaluation failed: {e}")
            return self.REGIME_LOW, 0.0, False
    
    def _load_catboost_momentum(self) -> bool:
        """Load CatBoost momentum model (primary).
        
        CatBoost is preferred because:
        - ordered boosting prevents target leakage in time-series
        - has_time=True preserves temporal ordering
        - Better accuracy on FX prediction benchmarks
        
        Returns:
            True if loaded successfully
        """
        model_path = self.model_dir / "catboost_momentum.cbm"
        
        if not model_path.exists():
            logger.debug(f"CatBoost momentum model not found at {model_path}")
            return False
        
        try:
            from catboost import CatBoostClassifier
            
            self._catboost_momentum = CatBoostClassifier()
            self._catboost_momentum.load_model(str(model_path), format='cbm')
            
            self._momentum_model_type = "catboost"
            logger.debug("✓ CatBoost momentum gate loaded")
            return True
            
        except ImportError:
            logger.debug("CatBoost not installed, will try XGBoost fallback")
            return False
        except Exception as e:
            logger.warning(f"Failed to load CatBoost momentum: {e}")
            return False
    
    def _load_lgbm_momentum(self) -> bool:
        """Load LightGBM momentum model (joint training default).
        
        LightGBM is used by joint training pipeline and provides:
        - Fast training and inference
        - Good handling of missing values
        - Strong gradient boosting performance
        
        Returns:
            True if loaded successfully
        """
        model_path = self.model_dir / "lgbm_momentum.pkl"
        
        if not model_path.exists():
            logger.debug(f"LightGBM momentum model not found at {model_path}")
            return False
        
        try:
            import pickle
            with open(model_path, 'rb') as f:
                loaded_obj = pickle.load(f)
            
            # Handle dict wrapper (from modular_trainers.py)
            if isinstance(loaded_obj, dict):
                if 'model' in loaded_obj:
                    self._lgbm_momentum = loaded_obj['model']
                elif 'momentum_model' in loaded_obj:
                    self._lgbm_momentum = loaded_obj['momentum_model']
                else:
                    self._lgbm_momentum = loaded_obj
                
                # Extract feature names and scaler
                self._lgbm_feature_names = loaded_obj.get('feature_names')
                self._lgbm_scaler = loaded_obj.get('scaler')
                
                if self._lgbm_feature_names:
                    logger.debug(f"LightGBM expects {len(self._lgbm_feature_names)} features")
            else:
                self._lgbm_momentum = loaded_obj
            
            self._momentum_model_type = "lightgbm"
            logger.debug("✓ LightGBM momentum gate loaded")
            return True
            
        except ImportError:
            logger.debug("LightGBM not installed, will try XGBoost fallback")
            return False
        except Exception as e:
            logger.warning(f"Failed to load LightGBM momentum: {e}")
            return False
    
    def _load_xgboost_momentum(self) -> bool:
        """Load XGBoost momentum model (fallback).
        
        Uses Booster.save_model/load_model for cross-version compatibility.
        
        Returns:
            True if loaded successfully
        """
        # Try .pkl format first (contains sklearn wrapper with proper predict method)
        if self._try_load_xgboost_pkl():
            return True
        
        # Fallback to .json format (raw Booster, requires DMatrix)
        return self._try_load_xgboost_json()
    
    def _try_load_xgboost_json(self) -> bool:
        """Try loading XGBoost model from .json format.
        
        Returns:
            True if loaded successfully
        """
        json_model_path = self.model_dir / "xgb_momentum.json"
        
        if not json_model_path.exists():
            return False
        
        try:
            import xgboost as xgb
            self._xgboost_momentum = xgb.Booster()
            self._xgboost_momentum.load_model(str(json_model_path))
            self._momentum_model_type = "xgboost"
            logger.debug("✓ XGBoost momentum gate loaded (.json)")
            return True
        except Exception as e:
            logger.debug(f"Failed to load XGBoost from .json: {e}")
            return False
    
    def _try_load_xgboost_pkl(self) -> bool:
        """Try loading XGBoost model from .pkl format.
        
        Returns:
            True if loaded successfully
        """
        model_path = self.model_dir / "xgb_momentum.pkl"
        
        if not model_path.exists():
            logger.debug(f"XGBoost momentum model not found at {model_path}")
            return False
        
        try:
            import xgboost as xgb
            
            # Load using Booster API (recommended for cross-version compatibility)
            self._xgboost_momentum = xgb.Booster()
            
            # Try .pkl file (may contain pickled Booster)
            if self._load_xgboost_pickled_booster(model_path, xgb):
                return True
            
            # If .pkl also fails, try loading from file path directly
            if self._load_xgboost_from_file_path(model_path):
                return True
            
            # Default fallback
            self._momentum_model_type = "xgboost"
            logger.debug("✓ XGBoost momentum gate loaded")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load XGBoost momentum: {e}")
            return False
    
    def _load_xgboost_pickled_booster(self, model_path: Path, xgb_module: Any) -> bool:
        """Try loading XGBoost model from pickled Booster object.

        Converts legacy .pkl files to .json format for cross-version compatibility.
        This avoids XGBoost's pickle deprecation warning.

        Args:
            model_path: Path to the .pkl file
            xgb_module: XGBoost module reference

        Returns:
            True if loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                # Suppress XGBoost pickle deprecation warning - model loads fine
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    loaded_obj = pickle.load(f)

            booster = None
            sklearn_model = None

            # Handle different pickle contents
            if isinstance(loaded_obj, xgb_module.Booster):
                booster = loaded_obj
            elif hasattr(loaded_obj, 'get_booster'):
                # XGBClassifier, XGBRegressor, or similar sklearn wrapper
                sklearn_model = loaded_obj
                booster = loaded_obj.get_booster()
            elif isinstance(loaded_obj, dict):
                # Model saved as dict with 'momentum_model' key (from modular_trainers.py)
                if 'momentum_model' in loaded_obj:
                    sklearn_model = loaded_obj['momentum_model']
                    if hasattr(sklearn_model, 'get_booster'):
                        booster = sklearn_model.get_booster()
                    else:
                        booster = sklearn_model
                    
                    # Store feature names if available
                    if 'feature_names' in loaded_obj:
                        self._xgb_feature_names = loaded_obj['feature_names']
                    if 'scaler' in loaded_obj:
                        self._xgb_scaler = loaded_obj['scaler']
                    
                    logger.debug(f"Loaded XGBoost model from dict with {len(loaded_obj.get('feature_names', []))} features")
                else:
                    # Model saved as dict - try to reconstruct as Booster config
                    booster = xgb_module.Booster()
                    import tempfile
                    import json as json_module
                    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tmp:
                        json_module.dump(loaded_obj, tmp)
                        tmp_path = tmp.name
                    booster.load_model(tmp_path)
                    import os
                    os.unlink(tmp_path)
            else:
                logger.debug(f"Unknown XGBoost pickle type: {type(loaded_obj)}")
                return False

            # Prefer sklearn model if available (has predict method)
            if sklearn_model is not None:
                self._xgboost_momentum = sklearn_model
            else:
                self._xgboost_momentum = booster
            self._momentum_model_type = "xgboost"

            # Convert to .json format for future cross-version compatibility
            # This eliminates the pickle deprecation warning on subsequent loads
            if booster is not None:
                json_path = self.model_dir / "xgb_momentum.json"
                try:
                    booster.save_model(str(json_path))
                    logger.info(f"✓ Converted legacy .pkl to {json_path} for cross-version compatibility")
                except Exception as save_err:
                    logger.debug(f"Could not save converted model to .json: {save_err}")

            logger.debug("✓ XGBoost momentum gate loaded (legacy .pkl)")
            return True

        except Exception as e:
            logger.debug(f"Failed to load XGBoost from .pkl: {e}")
            return False
    
    def _load_xgboost_from_file_path(self, model_path: Path) -> bool:
        """Try loading XGBoost model directly from file path.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if loaded successfully
        """
        try:
            self._xgboost_momentum.load_model(str(model_path))
            self._momentum_model_type = "xgboost"
            logger.debug("✓ XGBoost momentum gate loaded (.pkl)")
            return True
        except Exception as e:
            logger.debug(f"Failed to load XGBoost from .pkl file path: {e}")
            return False
    
    def _load_ridge_confidence(self) -> bool:
        """Load Ridge confidence model and metadata.
        
        Also loads scaler and feature names from meta file if available
        for proper output scaling to 0-100 range.
        
        Returns:
            True if loaded successfully
        """
        model_path = self.model_dir / "ridge_confidence.pkl"
        meta_path = self.model_dir / "ridge_confidence.meta.pkl"
        
        if not model_path.exists():
            logger.debug(f"Ridge confidence model not found at {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self._ridge_confidence = pickle.load(f)
            
            # Try to load metadata with scaler and feature names
            if meta_path.exists():
                try:
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    self._ridge_scaler = meta.get('scaler')
                    self._ridge_feature_names = meta.get('feature_names')
                    logger.debug("✓ Ridge metadata loaded (scaler + feature names)")
                except Exception as e:
                    logger.debug(f"Ridge metadata not available: {e}")
            
            logger.debug("✓ Ridge confidence gate loaded")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load Ridge confidence: {e}")
            return False
    
    def _load_rf_risk(self) -> bool:
        """Load RandomForest risk model.
        
        Returns:
            True if loaded successfully
        """
        model_path = self.model_dir / "rf_risk.pkl"
        
        if not model_path.exists():
            logger.debug(f"RF risk model not found at {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self._rf_risk = pickle.load(f)
            
            logger.debug("✓ RF risk gate loaded")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load RF risk: {e}")
            return False
    
    def _load_transformer(self) -> bool:
        """Load Transformer direction model.
        
        Returns:
            True if loaded successfully
        """
        model_path = self.model_dir / "transformer_direction.keras"
        
        if not model_path.exists():
            # Try .h5 extension as fallback
            model_path = self.model_dir / "transformer_direction.h5"
            if not model_path.exists():
                logger.debug(f"Transformer direction model not found")
                return False
        
        try:
            import tensorflow as tf
            
            # Suppress TF warnings during load
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._transformer = tf.keras.models.load_model(
                    str(model_path),
                    compile=False,
                )
            
            logger.debug("✓ Transformer direction gate loaded")
            return True
            
        except ImportError:
            logger.debug("TensorFlow not available for Transformer gate")
            return False
        except Exception as e:
            logger.debug(f"Failed to load Transformer: {e}")
            return False
    
    def _load_meta_labeler(self) -> bool:
        """Load Meta-labeler model.
        
        Returns:
            True if loaded successfully
        """
        model_path = self.model_dir / "meta_labeler.pkl"
        
        if not model_path.exists():
            logger.debug(f"Meta-labeler model not found at {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self._meta_labeler = pickle.load(f)
            
            logger.debug("✓ Meta-labeler gate loaded")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to load Meta-labeler: {e}")
            return False
    
    def _load_calibrator(self) -> bool:
        """Load confidence calibrator for probability calibration.
        
        Tries to load from:
        1. Joint model directory (preferred for scanner)
        2. Base model directory (fallback)
        
        Returns:
            True if loaded successfully
        """
        if not CALIBRATION_AVAILABLE:
            logger.debug("Calibration module not available")
            return False
        
        # Try joint directory first, then base
        search_paths = [
            self.model_dir / "confidence_calibrator.pkl",
            self.base_model_dir / "confidence_calibrator.pkl",
        ]
        
        for calibrator_path in search_paths:
            if calibrator_path.exists():
                try:
                    self._calibrator = ConfidenceCalibrator.load(calibrator_path)
                    
                    if self._calibrator.is_fitted:
                        logger.info(f"✓ Confidence calibrator loaded from {calibrator_path}")
                        return True
                    else:
                        logger.debug(f"Calibrator at {calibrator_path} is not fitted")
                        self._calibrator = None
                        
                except Exception as e:
                    logger.debug(f"Failed to load calibrator from {calibrator_path}: {e}")
                    continue
        
        logger.debug("No fitted confidence calibrator found - using raw probabilities")
        return False
    
    def _prepare_features_input(
        self,
        features: pd.DataFrame,
        target_feature_names: Optional[list] = None,
    ) -> np.ndarray:
        """Prepare features for model prediction.
        
        Converts DataFrame to numpy array and ensures proper shape.
        Optionally selects only specific features if target_feature_names provided.
        
        Args:
            features: DataFrame with engineered features
            target_feature_names: If provided, select only these features
            
        Returns:
            2D numpy array ready for model prediction
            
        Raises:
            ValueError: If features is empty or invalid
        """
        if features is None or (hasattr(features, '__len__') and len(features) == 0):
            raise ValueError("Features cannot be empty or None")
        
        # Select specific features if requested
        if target_feature_names is not None and hasattr(features, 'columns'):
            available = [f for f in target_feature_names if f in features.columns]
            missing = [f for f in target_feature_names if f not in features.columns]
            
            if missing:
                logger.debug(f"Missing {len(missing)} features for momentum model: {missing[:5]}...")
            
            if available:
                features = features[available]
                logger.debug(f"Selected {len(available)}/{len(target_feature_names)} features for model")
            else:
                logger.warning(f"No matching features found for model, using all {len(features.columns)} features")
        
        # Convert to numpy array
        X = features.values if hasattr(features, 'values') else features
        
        # Ensure 2D shape for single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        elif len(X.shape) != 2:
            raise ValueError(f"Features must be 1D or 2D, got shape {X.shape}")
        
        return X
    
    def _prepare_lgbm_features(
        self,
        features: pd.DataFrame,
        instrument: str,
    ) -> np.ndarray:
        """Prepare features specifically for LightGBM momentum model.
        
        The LightGBM model expects 34 features:
        - 19 numeric features (returns, volatility, indicators)
        - 15 instrument one-hot encoding columns
        
        Args:
            features: DataFrame with engineered features (single row)
            instrument: Currency pair name (e.g., 'EUR_USD')
            
        Returns:
            2D numpy array with shape (1, 34) ready for LightGBM
        """
        if self._lgbm_feature_names is None:
            logger.warning("LightGBM feature names not available, falling back to raw features")
            return self._prepare_features_input(features)
        
        # The expected numeric features (first 19)
        numeric_features = [
            'returns_1', 'returns_2', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
            'log_returns_1', 'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 
            'volatility_5', 'volatility_10', 'rsi_norm', 'stoch_k_norm',
            'macd_norm', 'macd_hist_norm', 'volume_ratio_5', 'volume_ratio_10', 'volume_zscore'
        ]
        
        # The instrument columns (last 15)
        instrument_columns = [
            'instrument_EUR_USD', 'instrument_GBP_USD', 'instrument_USD_JPY',
            'instrument_USD_CHF', 'instrument_AUD_USD', 'instrument_USD_CAD',
            'instrument_NZD_USD', 'instrument_EUR_GBP', 'instrument_EUR_JPY',
            'instrument_GBP_JPY', 'instrument_AUD_JPY', 'instrument_EUR_AUD',
            'instrument_GBP_AUD', 'instrument_EUR_CHF', 'instrument_GBP_CHF'
        ]
        
        # Build feature array
        feature_values = []
        
        # Extract numeric features
        for feat in numeric_features:
            if feat in features.columns:
                val = features[feat].iloc[-1] if len(features) > 0 else 0.0
            else:
                # Try mapping from similar feature names
                val = self._get_fallback_feature_value(features, feat)
            feature_values.append(float(val) if not pd.isna(val) else 0.0)
        
        # Add instrument one-hot encoding
        instrument_key = f'instrument_{instrument}'
        for inst_col in instrument_columns:
            feature_values.append(1.0 if inst_col == instrument_key else 0.0)
        
        # Convert to numpy array
        X = np.array(feature_values, dtype=np.float32).reshape(1, -1)
        
        # Apply scaler if available
        if self._lgbm_scaler is not None:
            try:
                X = self._lgbm_scaler.transform(X)
            except Exception as e:
                logger.debug(f"Scaler transform failed: {e}, using unscaled features")
        
        return X
    
    def _get_fallback_feature_value(
        self,
        features: pd.DataFrame,
        target_feat: str,
    ) -> float:
        """Get fallback value for a missing feature by mapping similar names.
        
        Args:
            features: DataFrame with available features
            target_feat: Target feature name we need
            
        Returns:
            Feature value or 0.0 if not found
        """
        # Common mappings between feature naming conventions
        mappings = {
            'returns_1': ['ret_1', 'return_1', 'pct_change_1'],
            'returns_2': ['ret_2', 'return_2', 'pct_change_2'],
            'returns_3': ['ret_3', 'return_3', 'pct_change_3'],
            'returns_5': ['ret_5', 'return_5', 'pct_change_5'],
            'returns_10': ['ret_10', 'return_10', 'pct_change_10'],
            'returns_20': ['ret_20', 'return_20', 'pct_change_20'],
            'log_returns_1': ['log_ret_1', 'log_return_1'],
            'atr_pct_5': ['atr_5', 'atr_pct_5d'],
            'atr_pct_10': ['atr_10', 'atr_pct_10d'],
            'atr_pct_14': ['atr_14', 'atr_pct_14d', 'atr'],
            'volatility_5': ['vol_5', 'std_5'],
            'volatility_10': ['vol_10', 'std_10'],
            'rsi_norm': ['rsi', 'rsi_14'],
            'stoch_k_norm': ['stoch_k', 'stoch_14_k'],
            'macd_norm': ['macd', 'macd_12_26'],
            'macd_hist_norm': ['macd_hist', 'macd_histogram'],
            'volume_ratio_5': ['vol_ratio_5', 'volume_5'],
            'volume_ratio_10': ['vol_ratio_10', 'volume_10'],
            'volume_zscore': ['vol_zscore', 'volume_z'],
        }
        
        alternatives = mappings.get(target_feat, [])
        for alt in alternatives:
            if alt in features.columns:
                return features[alt].iloc[-1]
        
        return 0.0
    
    def _extract_momentum_score(
        self,
        proba: np.ndarray,
    ) -> float:
        """Extract momentum score from probability array.
        
        Args:
            proba: Probability array from model prediction
            
        Returns:
            Momentum score in range [0, 1]
            
        Raises:
            ValueError: If probability array is invalid or out of range
        """
        if proba is None or proba.size == 0:
            raise ValueError("Probability array is empty or None")
        
        # Extract score based on array shape
        if len(proba.shape) > 1:
            # Multi-class probabilities: take probability of class 1
            momentum_score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
        else:
            # Single probability value
            momentum_score = float(proba[0])
        
        # Validate range
        if not (0.0 <= momentum_score <= 1.0):
            logger.warning(f"Momentum score {momentum_score} out of range [0, 1], clamping")
            momentum_score = max(0.0, min(1.0, momentum_score))
        
        return momentum_score
    
    def _predict_with_catboost(
        self,
        X: np.ndarray,
    ) -> Tuple[float, bool]:
        """Predict momentum using CatBoost model.
        
        Args:
            X: Prepared feature array
            
        Returns:
            Tuple of (momentum_score, acceleration_detected)
            
        Raises:
            RuntimeError: If CatBoost prediction fails
        """
        try:
            proba = self._catboost_momentum.predict_proba(X)
            momentum_score = self._extract_momentum_score(proba)
            acceleration = momentum_score > self.ACCELERATION_THRESHOLD
            
            logger.debug(
                f"CatBoost momentum prediction: score={momentum_score:.3f}, "
                f"acceleration={acceleration}"
            )
            
            return momentum_score, acceleration
            
        except (ValueError, IndexError, AttributeError) as e:
            raise RuntimeError(f"CatBoost prediction error: {e}") from e
    
    def _predict_with_lgbm(
        self,
        X: np.ndarray,
    ) -> Tuple[float, bool]:
        """Predict momentum using LightGBM model.
        
        Args:
            X: Prepared feature array
            
        Returns:
            Tuple of (momentum_score, acceleration_detected)
            
        Raises:
            RuntimeError: If LightGBM prediction fails
        """
        try:
            import warnings
            # Suppress sklearn feature names warning (model still works fine)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
                
                if hasattr(self._lgbm_momentum, 'predict_proba'):
                    # LGBMClassifier with probability output
                    proba = self._lgbm_momentum.predict_proba(X)
                    momentum_score = self._extract_momentum_score(proba)
                elif hasattr(self._lgbm_momentum, 'predict'):
                    # LGBMRegressor or Booster
                    prediction = self._lgbm_momentum.predict(X)
                    momentum_score = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
                    
                    # Clamp to [0, 1] - regressor output may be outside this range
                    if momentum_score < 0.0 or momentum_score > 1.0:
                        logger.debug(f"LightGBM output {momentum_score:.3f} clamped to [0, 1]")
                        momentum_score = max(0.0, min(1.0, momentum_score))
                else:
                    raise RuntimeError("LightGBM model has no predict method")
            
            acceleration = momentum_score > self.ACCELERATION_THRESHOLD
            
            logger.debug(
                f"LightGBM momentum prediction: score={momentum_score:.3f}, "
                f"acceleration={acceleration}"
            )
            
            return momentum_score, acceleration
            
        except Exception as e:
            raise RuntimeError(f"LightGBM prediction error: {e}") from e
    
    def _predict_with_xgboost(
        self,
        X: np.ndarray,
    ) -> Tuple[float, bool]:
        """Predict momentum using XGBoost model.
        
        Args:
            X: Prepared feature array
            
        Returns:
            Tuple of (momentum_score, acceleration_detected)
            
        Raises:
            RuntimeError: If XGBoost prediction fails
        """
        try:
            import xgboost as xgb
            
            # Apply scaler if available
            if self._xgb_scaler is not None:
                try:
                    X = self._xgb_scaler.transform(X)
                except Exception as e:
                    logger.debug(f"Scaler transform failed, using raw features: {e}")
            
            # Check model type and predict accordingly
            if hasattr(self._xgboost_momentum, 'predict_proba'):
                # XGBClassifier with probability output
                proba = self._xgboost_momentum.predict_proba(X)
                momentum_score = self._extract_momentum_score(proba)
            elif hasattr(self._xgboost_momentum, 'predict') and hasattr(self._xgboost_momentum, 'get_booster'):
                # sklearn wrapper (XGBRegressor or XGBClassifier without proba)
                prediction = self._xgboost_momentum.predict(X)
                momentum_score = float(prediction[0])
                
                # Clamp to [0, 1] - regressor output may be outside this range
                if momentum_score < 0.0 or momentum_score > 1.0:
                    logger.debug(f"XGBoost regressor output {momentum_score:.3f} clamped to [0, 1]")
                    momentum_score = max(0.0, min(1.0, momentum_score))
            else:
                # Native XGBoost Booster requires DMatrix
                dmatrix = xgb.DMatrix(X)
                prediction = self._xgboost_momentum.predict(dmatrix)
                momentum_score = float(prediction[0])
                
                # Validate range
                if not (0.0 <= momentum_score <= 1.0):
                    logger.warning(f"XGBoost prediction {momentum_score} out of range [0, 1], clamping")
                    momentum_score = max(0.0, min(1.0, momentum_score))
            
            acceleration = momentum_score > self.ACCELERATION_THRESHOLD
            
            logger.debug(
                f"XGBoost momentum prediction: score={momentum_score:.3f}, "
                f"acceleration={acceleration}"
            )
            
            return momentum_score, acceleration
            
        except (ValueError, IndexError, AttributeError, TypeError) as e:
            raise RuntimeError(f"XGBoost prediction error: {e}") from e
    
    def evaluate_momentum(
        self,
        features: pd.DataFrame,
        instrument: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """Evaluate momentum gate using CatBoost (primary) or XGBoost (fallback).
        
        This method attempts to predict momentum using the loaded model(s) in order:
        1. CatBoost (preferred for time-series with ordered boosting)
        2. LightGBM (joint training default)
        3. XGBoost (fallback if CatBoost/LightGBM unavailable or fails)
        4. Neutral value if no models are available
        
        Args:
            features: DataFrame with engineered features for prediction
            instrument: Currency pair name (e.g., 'EUR_USD') for LightGBM one-hot encoding
            
        Returns:
            Tuple of (momentum_score, acceleration_detected):
                - momentum_score: Float in range [0, 1] representing momentum percentile
                  (median ~0.3 for typical FX data)
                - acceleration_detected: True if momentum exceeds acceleration threshold
                  (default 0.5), indicating strong upward momentum
        
        Raises:
            ValueError: If features is empty or has invalid structure
            
        Examples:
            >>> evaluator = GateEvaluator(model_dir)
            >>> evaluator.load_models()
            >>> features = pd.DataFrame({...})
            >>> score, acceleration = evaluator.evaluate_momentum(features, 'EUR_USD')
            >>> print(f"Momentum: {score:.2f}, Acceleration: {acceleration}")
        """
        # Input validation
        if features is None:
            raise ValueError("Features cannot be None")
        
        # Use only the last row for sklearn models (XGBoost, CatBoost)
        last_row = features.iloc[-1:] if len(features) > 1 else features
        
        # Prepare input - use model's expected features if available
        try:
            X = self._prepare_features_input(last_row, self._xgb_feature_names)
        except ValueError as e:
            logger.error(f"Failed to prepare features for momentum evaluation: {e}")
            raise
        
        # Try CatBoost first (primary model)
        if self._catboost_momentum is not None:
            try:
                momentum_score, acceleration = self._predict_with_catboost(X)
                logger.debug(f"Momentum: {momentum_score:.3f} (CatBoost)")
                return momentum_score, acceleration
                
            except RuntimeError as e:
                logger.debug(f"CatBoost momentum prediction failed, trying LightGBM: {e}")
        
        # Try LightGBM (joint training default) - needs its own feature prep
        if self._lgbm_momentum is not None:
            try:
                # Prepare features specifically for LightGBM (includes instrument encoding)
                X_lgbm = self._prepare_lgbm_features(last_row, instrument or 'EUR_USD')
                momentum_score, acceleration = self._predict_with_lgbm(X_lgbm)
                logger.debug(f"Momentum: {momentum_score:.3f} (LightGBM)")
                return momentum_score, acceleration
                
            except RuntimeError as e:
                logger.warning(f"LightGBM momentum prediction failed: {e}")
        
        # Fallback to XGBoost
        if self._xgboost_momentum is not None:
            try:
                momentum_score, acceleration = self._predict_with_xgboost(X)
                logger.debug(f"Momentum: {momentum_score:.3f} (XGBoost)")
                return momentum_score, acceleration
                
            except RuntimeError as e:
                logger.debug(f"XGBoost momentum prediction failed: {e}")
        
        # No model available - return neutral value
        logger.warning(
            "No momentum model available (CatBoost, LightGBM, and XGBoost all missing or failed), "
            "returning neutral value"
        )
        return self.NEUTRAL_MOMENTUM_SCORE, False
    
    def evaluate_confidence(
        self,
        features: pd.DataFrame,
    ) -> float:
        """Evaluate confidence gate (ADX-based).
        
        Returns a confidence score NORMALIZED to 0-1 range based on:
        1. Ridge model prediction (if available) - normalized from 0-100 to 0-1
        2. Direct ADX value from features (fallback) - normalized from 0-100 to 0-1
        3. Computed ADX from price data (last resort) - normalized from 0-100 to 0-1
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Confidence score 0-1 (0.50+ is strong trend)
        """
        # Use only the last row for Ridge model
        last_row = features.iloc[-1:] if len(features) > 1 else features
        
        # Try Ridge model first
        if self._ridge_confidence is not None:
            try:
                X = last_row.values if hasattr(last_row, 'values') else last_row
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                
                raw_confidence = float(self._ridge_confidence.predict(X)[0])
                
                # Normalize to 0-1 range
                # Ridge trained on ADX outputs 0-100, so normalize
                if raw_confidence > 1.0:
                    # Legacy 0-100 scale - normalize to 0-1
                    confidence = _normalize_confidence(raw_confidence, from_scale=100)
                elif raw_confidence < 0:
                    # Negative values from standardized output - use sigmoid-like scaling
                    confidence = 0.5 + 0.25 * np.tanh(raw_confidence)
                else:
                    # Already in 0-1 range
                    confidence = raw_confidence
                
                # Clamp to valid range
                confidence = max(0.0, min(1.0, confidence))
                
                logger.debug(f"Ridge confidence: raw={raw_confidence:.2f} → normalized={_format_confidence_pct(confidence)}")
                return confidence
                    
            except Exception as e:
                logger.debug(f"Ridge confidence prediction failed: {e}")
        
        # Fallback: use ADX directly if available in features (normalize from 0-100)
        if 'adx' in features.columns:
            adx_val = features['adx'].iloc[-1]
            if pd.notna(adx_val) and 0 <= adx_val <= 100:
                return _normalize_confidence(float(adx_val), from_scale=100)
        
        # Last resort: compute ADX from price data if available
        if all(col in features.columns for col in ['high', 'low', 'close']):
            adx = self._compute_adx(features)
            if adx is not None:
                return _normalize_confidence(adx, from_scale=100)
        
        return 0.50  # Neutral fallback (0-1 scale)
    
    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Compute ADX (Average Directional Index) from price data.
        
        Args:
            df: DataFrame with high, low, close columns
            period: ADX period (default 14)
            
        Returns:
            ADX value (0-100) or None if insufficient data
        """
        try:
            if len(df) < period + 1:
                return None
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # True Range
            tr = np.zeros(len(df))
            for i in range(1, len(df)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
            
            # Directional Movement
            plus_dm = np.zeros(len(df))
            minus_dm = np.zeros(len(df))
            for i in range(1, len(df)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # Smoothed averages (Wilder's smoothing)
            def wilder_smooth(arr, period):
                result = np.zeros(len(arr))
                result[period] = np.sum(arr[1:period+1])
                for i in range(period + 1, len(arr)):
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
                return result
            
            atr = wilder_smooth(tr, period)
            plus_di = 100 * wilder_smooth(plus_dm, period) / np.maximum(atr, 1e-10)
            minus_di = 100 * wilder_smooth(minus_dm, period) / np.maximum(atr, 1e-10)
            
            # DX and ADX
            dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
            adx = wilder_smooth(dx, period) / period
            
            return float(np.clip(adx[-1], 0, 100))
            
        except Exception as e:
            logger.debug(f"ADX computation failed: {e}")
            return None
    
    def evaluate_risk(
        self,
        features: pd.DataFrame,
    ) -> Tuple[float, float]:
        """Evaluate risk gate (expected drawdown).
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Tuple of (drawdown_pct, streak_probability)
        """
        if self._rf_risk is None:
            return 0.02, 0.5  # Default 2% drawdown, 50% streak
        
        # Use only the last row for RF model
        last_row = features.iloc[-1:] if len(features) > 1 else features
        
        try:
            X = last_row.values if hasattr(last_row, 'values') else last_row
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # RF predicts drawdown percentage
            drawdown = float(self._rf_risk.predict(X)[0])
            
            # Estimate streak probability from drawdown
            streak_prob = min(0.9, drawdown * 10)
            
            return max(0, drawdown), streak_prob
            
        except Exception as e:
            logger.debug(f"RF risk prediction failed: {e}")
            return 0.02, 0.5
    
    def evaluate_transformer(
        self,
        features: pd.DataFrame,
        seq_len: int = 60,
    ) -> Tuple[Optional[str], float, float, bool]:
        """Evaluate Transformer direction gate with optional calibration.
        
        If a fitted calibrator is available and calibration is enabled,
        the raw probability will be calibrated to better reflect true win rates.
        
        Args:
            features: DataFrame with engineered features
            seq_len: Sequence length for Transformer input
            
        Returns:
            Tuple of (direction, calibrated_prob, raw_prob, was_calibrated):
                - direction: "LONG", "SHORT", or None if model unavailable
                - calibrated_prob: Calibrated probability (0-1), or raw if no calibrator
                - raw_prob: Raw model probability (0-1)
                - was_calibrated: True if calibration was applied
        """
        if self._transformer is None:
            return None, 0.5, 0.5, False
        
        try:
            # Prepare sequence input
            if len(features) < seq_len:
                logger.debug(f"Insufficient data for Transformer: {len(features)} < {seq_len}")
                return None, 0.5, 0.5, False
            
            # Get normalized features if available
            if FEATURES_AVAILABLE and compute_normalized_features is not None:
                norm_features = compute_normalized_features(features)
            else:
                norm_features = features
            
            # Select numeric columns only
            numeric_cols = norm_features.select_dtypes(include=[np.number]).columns
            X = norm_features[numeric_cols].iloc[-seq_len:].values
            
            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Reshape for batch input: (1, seq_len, features)
            X = X.reshape(1, seq_len, -1).astype(np.float32)
            
            # Predict
            proba = self._transformer.predict(X, verbose=0)
            
            # Extract probability (assumes binary classification)
            if len(proba.shape) > 1 and proba.shape[1] > 1:
                prob_long = float(proba[0, 1])
            else:
                prob_long = float(proba[0])
            
            # Store raw probability before calibration
            raw_prob = prob_long
            
            # Apply calibration if available
            calibrated_prob, was_calibrated = self.calibrate_probability(prob_long)
            
            direction = "LONG" if calibrated_prob > 0.5 else "SHORT"
            
            if was_calibrated:
                logger.debug(f"Transformer prob: raw={raw_prob:.3f} → calibrated={calibrated_prob:.3f}")
            
            return direction, calibrated_prob, raw_prob, was_calibrated
            
        except Exception as e:
            logger.debug(f"Transformer prediction failed: {e}")
            return None, 0.5, 0.5, False
    
    def evaluate_meta_labeler(
        self,
        features: pd.DataFrame,
        primary_direction: str,
    ) -> float:
        """Evaluate Meta-labeler gate.
        
        Meta-labeler predicts whether the primary model's trade signal
        will be successful (used as a quality filter).
        
        Args:
            features: DataFrame with engineered features
            primary_direction: Direction from primary model ("LONG"/"SHORT")
            
        Returns:
            Success probability (0-1), higher = more likely to succeed
        """
        if self._meta_labeler is None:
            return 0.6  # Default optimistic if unavailable
        
        try:
            X = features.values if hasattr(features, 'values') else features
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Add direction indicator if meta-labeler expects it
            # (direction encoded as 1 for LONG, -1 for SHORT)
            direction_indicator = 1.0 if primary_direction == "LONG" else -1.0
            
            # Check if model has predict_proba
            if hasattr(self._meta_labeler, 'predict_proba'):
                proba = self._meta_labeler.predict_proba(X)
                success_prob = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0])
            else:
                # Use predict directly
                pred = self._meta_labeler.predict(X)
                success_prob = float(pred[0])
            
            return max(0.0, min(1.0, success_prob))
            
        except Exception as e:
            logger.debug(f"Meta-labeler prediction failed: {e}")
            return 0.6  # Default optimistic on failure
    
    def evaluate_all_gates(
        self,
        features: pd.DataFrame,
        min_confidence: float = 0.50,
        min_momentum: float = 0.20,
        max_drawdown_pct: float = 0.025,
        min_transformer_prob: float = 0.60,
        min_meta_confidence: float = 0.55,
        instrument: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate all trading gates including advanced gates.
        
        ALL thresholds use 0-1 scale internally.
        
        Args:
            features: DataFrame with engineered features
            min_confidence: Minimum ADX confidence threshold (0-1)
            min_momentum: Minimum momentum threshold (0-1)
            max_drawdown_pct: Maximum drawdown threshold (0-1, e.g., 0.025 = 2.5%)
            min_transformer_prob: Minimum Transformer probability threshold (0-1)
            min_meta_confidence: Minimum meta-labeler confidence threshold (0-1)
            
        Returns:
            Dict with comprehensive gate results including:
            - Basic gates: momentum, confidence, risk (all 0-1 scale)
            - Advanced gates: transformer, meta_labeler (all 0-1 scale)
            - Combined pass status
        """
        # Evaluate basic gates (all return 0-1 scale)
        momentum, acceleration = self.evaluate_momentum(features, instrument)
        confidence = self.evaluate_confidence(features)  # Now returns 0-1
        drawdown, streak_prob = self.evaluate_risk(features)
        
        # Check basic thresholds (all 0-1 scale)
        momentum_passed = momentum >= min_momentum or acceleration
        confidence_passed = confidence >= min_confidence
        risk_passed = drawdown <= max_drawdown_pct
        
        # Evaluate advanced gates (transformer returns calibrated prob if calibrator available)
        transformer_direction, transformer_prob, raw_transformer_prob, was_calibrated = self.evaluate_transformer(features)
        transformer_passed = transformer_prob >= min_transformer_prob if transformer_direction else True
        
        # Meta-labeler requires a direction from transformer or fallback
        direction = transformer_direction or ("LONG" if momentum > 0.5 else "SHORT")
        meta_confidence = self.evaluate_meta_labeler(features, direction)
        meta_passed = meta_confidence >= min_meta_confidence if self._meta_labeler else True
        
        # Combined pass status
        # Basic gates must all pass
        basic_passed = momentum_passed and confidence_passed and risk_passed
        
        # Advanced gates are optional but improve quality if available
        # all_passed requires basic gates + transformer if available
        if self._transformer is not None:
            all_passed = basic_passed and transformer_passed
        else:
            all_passed = basic_passed
        
        # Gates count for display
        gates_passed = sum([momentum_passed, confidence_passed, risk_passed])
        total_basic_gates = 3
        
        if self._transformer is not None:
            gates_passed += int(transformer_passed)
            total_basic_gates += 1
        
        # Log gate results with percentage format
        logger.debug(
            f"Gates: momentum={_format_confidence_pct(momentum)} (passed={momentum_passed}), "
            f"confidence={_format_confidence_pct(confidence)} (passed={confidence_passed}), "
            f"drawdown={_format_confidence_pct(drawdown)} (passed={risk_passed})"
        )
        
        return {
            # Basic gates (ALL 0-1 SCALE)
            "momentum": momentum,
            "momentum_acceleration": acceleration,
            "momentum_passed": momentum_passed,
            "confidence": confidence,  # NOW 0-1 (was 0-100)
            "confidence_passed": confidence_passed,
            "drawdown": drawdown,
            "streak_prob": streak_prob,
            "risk_passed": risk_passed,
            # Advanced gates (ALL 0-1 SCALE)
            "transformer_direction": transformer_direction,
            "transformer_prob": transformer_prob,  # Calibrated if calibrator available
            "transformer_prob_raw": raw_transformer_prob,  # Raw model output
            "transformer_calibrated": was_calibrated,  # True if calibration was applied
            "transformer_passed": transformer_passed,
            "meta_confidence": meta_confidence,
            "meta_passed": meta_passed,
            # Combined
            "basic_passed": basic_passed,
            "all_passed": all_passed,
            "gates_passed": gates_passed,
            "total_gates": total_basic_gates,
            "model_type": self._momentum_model_type,
            "direction": direction,
            "has_calibrator": self.has_calibrator,
        }
    
    @property
    def momentum_model_type(self) -> str:
        """Get the type of momentum model in use."""
        return self._momentum_model_type
    
    @property
    def is_loaded(self) -> bool:
        """Check if any models are loaded (including TCN volatility for scanner)."""
        return any([
            self._catboost_momentum is not None,
            self._lgbm_momentum is not None,
            self._xgboost_momentum is not None,
            self._ridge_confidence is not None,
            self._rf_risk is not None,
            self._tcn_volatility is not None,
        ])
    
    @property
    def has_transformer(self) -> bool:
        """Check if Transformer model is loaded."""
        return self._transformer is not None
    
    @property
    def has_meta_labeler(self) -> bool:
        """Check if Meta-labeler model is loaded."""
        return self._meta_labeler is not None
