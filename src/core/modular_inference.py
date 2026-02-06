"""
Modular Ensemble Inference Pipeline.

Supports TWO modes:

1. DIRECTION MODE (legacy):
   - Transformer/TCN gives direction (long/short)
   - Trade only if all gates pass

2. REGIME MODE (new):
   - Transformer classifies market regime (trend/chop/mean_revert)
   - TREND: Let XGBoost/Ridge/RF decide direction via momentum
   - CHOP: Skip trading entirely
   - MEAN_REVERT: Fade 2-bar momentum
   
Gates (both modes):
- Ridge confidence > 75
- XGBoost momentum fresh OR accelerating
- RF expected drawdown < threshold

Market Intelligence (NEW):
- News sentiment analysis via FinBERT
- Economic calendar integration (blocks trades before/after high-impact events)
- Online learning from trade outcomes

Position sized for 2% max risk.

IMPORTANT: Uses NORMALIZED features (returns, z-scores, ratios) that are
instrument-agnostic. Models trained on GBP_USD work on USD_JPY, EUR_USD, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Import normalized feature computation from data loaders
from .modular_data_loaders import compute_normalized_features, get_normalized_feature_names

# Import confidence calibration module
try:
    from confidence_calibration import (
        ConfidenceCalibrator,
        CalibrationConfig,
        CalibrationResult,
    )
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    ConfidenceCalibrator = None
    CalibrationConfig = None
    CalibrationResult = None

# Import meta-labeling module
try:
    from meta_labeling import MetaLabeler, MetaLabelingConfig
    META_LABELING_AVAILABLE = True
except ImportError:
    META_LABELING_AVAILABLE = False
    MetaLabeler = None
    MetaLabelingConfig = None

# Import market intelligence module
try:
    from market_intelligence import MarketIntelligence, fetch_forex_news
    MARKET_INTEL_AVAILABLE = True
except ImportError:
    MARKET_INTEL_AVAILABLE = False
    MarketIntelligence = None
    fetch_forex_news = None

# Import online retrainer module for drift-triggered retraining
try:
    from online_retrainer import OnlineRetrainer, RetrainConfig, create_retrain_callback
    ONLINE_RETRAINER_AVAILABLE = True
except ImportError:
    ONLINE_RETRAINER_AVAILABLE = False
    OnlineRetrainer = None
    RetrainConfig = None
    create_retrain_callback = None

# RL position sizer availability is checked lazily to avoid TF/PyTorch GPU conflicts
# The actual import happens only when use_rl_sizer=True and the model is loaded
RL_AVAILABLE = True  # We assume it's available, actual check is deferred
RLPositionSizer = None  # Lazy loaded
RL_MODEL_PATH = Path("trained_data/models/rl_position_sizer.zip")

def _lazy_load_rl_sizer():
    """Lazy load RLPositionSizer to avoid TF/PyTorch GPU conflicts at import time."""
    global RLPositionSizer, RL_AVAILABLE
    if RLPositionSizer is not None:
        return RLPositionSizer, RL_AVAILABLE
    try:
        from rl_position_sizing import RLPositionSizer as _RLPositionSizer, RL_MODEL_PATH as _RL_MODEL_PATH
        RLPositionSizer = _RLPositionSizer
        RL_AVAILABLE = True
        return RLPositionSizer, RL_AVAILABLE
    except ImportError:
        RL_AVAILABLE = False
        RLPositionSizer = None
        return None, False

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference gates.
    
    Thresholds calibrated for gate models trained on:
    - Confidence: ADX-based trend strength (0-100)
    - Momentum: Percentile-normalized (median=0.3, P90=0.7)
    - Risk: ATR-based expected drawdown (typically 0.5-3%)
    """
    # TCN/Transformer probability gate - CRITICAL: reject coin-flip signals
    # 0.5 = uncertain, 0.55+ = slight edge, 0.60+ = good edge
    min_tcn_probability: float = 0.60  # Require at least 60% confidence in direction (tightened from 0.55)
    
    # Confidence gate - ADX-based, 50+ is strong trend
    min_confidence: float = 50.0  # 0-100 scale (tightened from 45)
    
    # Momentum gate - median momentum is 0.3, so 0.20 catches bottom 40%
    min_momentum: float = 0.20  # 0-1 scale (tightened from 0.15)
    require_fresh_or_accel: bool = True
    
    # Risk gate - ATR-based drawdown (2x ATR, typically 0.5-3%)
    max_drawdown_pct: float = 0.025  # 2.5% max expected drawdown
    max_streak_prob: float = 0.6  # 60% max streak continuation
    
    # Legacy pip-based (kept for backward compatibility)
    max_drawdown_pips: float = 250.0  # ~2.5% for majors
    
    # Permissive mode: Ignore failing gates from sklearn models when version mismatch detected
    # When True, only Transformer direction is used for decision
    permissive_mode: bool = False
    
    # Risk gate bypass: When True AND permissive_mode is True, also bypass the risk gate
    # By default, risk gate is ALWAYS active even in permissive mode for safety
    bypass_risk_gate_in_permissive: bool = False

    # Sentiment gate
    sentiment_block_enabled: bool = True
    sentiment_block_threshold: float = 0.60  # Absolute sentiment score to block
    sentiment_min_headlines: int = 3  # Require minimum headlines before using sentiment
    
    # Confidence calibration
    enable_calibration: bool = True  # Apply Platt/Isotonic calibration to raw probabilities
    calibration_method: str = 'platt'  # 'platt', 'isotonic', or 'none'
    
    # Meta-labeling gate - predicts whether primary model's signal will succeed
    enable_meta_labeling: bool = True  # Use meta-labeler to filter trades
    min_meta_confidence: float = 0.55  # Minimum meta-confidence to allow trade
    
    # Position sizing - RISK-BASED (5% aggressive mode)
    risk_per_trade_pct: float = 0.05  # 5% risk per trade (~$5k on $100k)
    account_equity: float = 103000.0  # User's account balance
    pip_value: float = 10.0  # ~$10 per pip per standard lot
    
    # LIQUIDITY LIMITS - Maximum lots by pair (increased for aggressive trading)
    max_lots_by_pair: dict = None  # Set in __post_init__
    
    def __post_init__(self):
        if self.max_lots_by_pair is None:
            self.max_lots_by_pair = {
                'EUR_USD': 50.0,   # Most liquid - 50 lots max
                'USD_JPY': 40.0,   # Very liquid
                'GBP_USD': 30.0,   # Liquid
                'USD_CHF': 30.0,
                'AUD_USD': 30.0,
                'USD_CAD': 30.0,
                'NZD_USD': 20.0,   # Less liquid
                'GBP_JPY': 15.0,   # Cross
                'EUR_GBP': 15.0,
                'EUR_JPY': 15.0,
                'DEFAULT': 10.0,   # Unknown pairs
            }


@dataclass
class TradeSignal:
    """Result of inference pipeline."""
    trade: bool
    direction: Optional[str]  # 'long' or 'short' or None
    size: float  # Position size (lots or units)
    confidence: float
    
    # Regime results (new)
    regime: Optional[str] = None  # 'trend', 'chop', 'mean_revert', or None
    regime_confidence: float = 0.0
    
    # Gate results
    tcn_direction: Optional[int] = None
    tcn_probability: float = 0.5
    ridge_confidence: float = 0.0
    xgb_momentum: float = 0.0
    xgb_acceleration: bool = False
    rf_drawdown_pips: float = 0.0
    rf_streak_prob: float = 0.0
    
    # Hybrid voting (HistGB + Transformer)
    histgb_direction: Optional[int] = None
    histgb_probability: float = 0.5
    models_agree: bool = True  # True if Transformer and HistGB agree
    
    # Gate status
    confidence_gate_passed: bool = False
    momentum_gate_passed: bool = False
    risk_gate_passed: bool = False
    regime_gate_passed: bool = True  # True if not in CHOP regime
    meta_gate_passed: bool = True  # True if meta-labeler allows trade (or not loaded)
    
    # Meta-labeler confidence (predicts trade SUCCESS, not direction)
    meta_confidence: float = 0.0
    
    # Rejection reason if no trade
    reason: Optional[str] = None
    
    # Market intelligence data (NEW)
    metadata: Optional[Dict[str, Any]] = None


class ModularEnsembleInference:
    """
    Inference pipeline for modular ensemble.
    
    Loads 4 independent models and combines their predictions using gated logic.
    No shared processing - each model sees its own feature subset.
    
    Supports three modes:
    - DIRECTION MODE: Transformer/TCN predicts direction directly
    - REGIME MODE: Transformer classifies regime, direction derived from momentum
    - HYBRID MODE: Transformer + HistGB voting for higher confidence trades
    
    Hybrid voting logic:
    - If both models agree: trade with full confidence
    - If models disagree in low-vol regime: use HistGB (more stable)
    - If models disagree in high-vol regime: use Transformer (better at trends)
    """
    
    def __init__(
        self,
        model_dir: str = "trained_data/models",
        config: Optional[InferenceConfig] = None,
        use_rl_sizer: bool = False,
        instrument: Optional[str] = None,  # For pair-specific model loading
        enable_market_intelligence: bool = True,  # Enable sentiment/calendar/online learning
        enable_llm_integration: bool = False,  # Enable LLM-powered enhancements
        enable_drift_detection: bool = True,  # Enable drift detection and auto-retrain
        drift_config: Optional[Dict[str, Any]] = None,  # Drift detection config overrides
    ):
        self.model_dir = Path(model_dir)
        self.config = config or InferenceConfig()
        self.instrument = instrument  # Store for pair-specific loading
        
        self.tcn = None  # Direction model (legacy)
        self.histgb = None  # HistGB baseline for hybrid voting
        self.regime_model = None  # Regime classifier (new)
        self.xgb = None
        self.rf = None
        self.ridge = None
        
        # LLM Integration (NEW)
        self.enable_llm = enable_llm_integration
        self._llm_sentiment_cache = {}  # Cache LLM sentiment by headlines hash
        
        # Drift detection config
        self.enable_drift_detection = enable_drift_detection
        self._drift_config_overrides = drift_config or {}
        
        # Market Intelligence with drift detection
        self.market_intel = None
        if enable_market_intelligence and MARKET_INTEL_AVAILABLE:
            try:
                # Create drift config from overrides
                drift_cfg = None
                if enable_drift_detection:
                    from market_intelligence import DriftConfig
                    drift_cfg = DriftConfig(
                        feature_drift_threshold=self._drift_config_overrides.get(
                            'feature_drift_threshold', 0.10
                        ),
                        performance_drift_threshold=self._drift_config_overrides.get(
                            'performance_drift_threshold', 0.05
                        ),
                        min_trades_for_drift_check=self._drift_config_overrides.get(
                            'min_trades_for_drift_check', 20
                        ),
                        incremental_retrain_epochs=self._drift_config_overrides.get(
                            'incremental_retrain_epochs', 3
                        ),
                        auto_retrain_on_drift=self._drift_config_overrides.get(
                            'auto_retrain_on_drift', True
                        ),
                        max_retrains_per_day=self._drift_config_overrides.get(
                            'max_retrains_per_day', 3
                        ),
                        cooldown_minutes=self._drift_config_overrides.get(
                            'cooldown_minutes', 60
                        ),
                    )
                
                self.market_intel = MarketIntelligence(
                    enable_sentiment=True,
                    enable_calendar=True,
                    enable_online_learning=True,
                    enable_drift_detection=enable_drift_detection,
                    drift_config=drift_cfg,
                    retrain_callback=self._incremental_retrain if enable_drift_detection else None,
                )
                features_str = "sentiment + calendar + online learning"
                if enable_drift_detection:
                    features_str += " + drift detection"
                logger.info(f"✓ Market Intelligence enabled ({features_str})")
            except Exception as e:
                logger.warning(f"Market Intelligence initialization failed: {e}")
                self.market_intel = None
        elif enable_market_intelligence and not MARKET_INTEL_AVAILABLE:
            logger.info("ℹ Market Intelligence not available (install transformers for sentiment)")
        
        # RL Position Sizer (NEW)
        self.rl_sizer: Optional[RLPositionSizer] = None
        self.use_rl_sizer = use_rl_sizer
        
        # Confidence Calibrator (NEW)
        self.calibrator: Optional['ConfidenceCalibrator'] = None
        self._calibration_loaded = False
        
        # Meta-Labeler (predicts trade SUCCESS, not direction)
        self.meta_labeler: Optional['MetaLabeler'] = None
        self._meta_labeler_loaded = False
        
        self.use_regime = False  # Will be set during load_models
        self.use_hybrid = False  # Enable HistGB voting
        self._loaded = False
        self._loaded_instrument = None  # Track which pair's models are loaded
        
        # Drift tracking for retrain decisions
        self._last_features: Optional[np.ndarray] = None
        self._pending_retrain: bool = False
        self._retrain_reason: Optional[str] = None
        
        # Online retrainer for drift-triggered incremental learning
        self._online_retrainer: Optional['OnlineRetrainer'] = None
        if enable_drift_detection and ONLINE_RETRAINER_AVAILABLE:
            try:
                retrain_cfg = RetrainConfig(
                    cooldown_minutes=self._drift_config_overrides.get('cooldown_minutes', 60),
                    max_retrains_per_day=self._drift_config_overrides.get('max_retrains_per_day', 3),
                    min_samples_for_retrain=self._drift_config_overrides.get('min_samples_for_retrain', 50),
                )
                self._online_retrainer = OnlineRetrainer(config=retrain_cfg)
                logger.info("✓ OnlineRetrainer initialized for drift-triggered retraining")
            except Exception as e:
                logger.warning(f"OnlineRetrainer initialization failed: {e}")
    
    # =========================================================================
    # ONLINE LEARNING & DRIFT DETECTION
    # =========================================================================
    
    def _incremental_retrain(self) -> Dict[str, Any]:
        """
        Callback triggered by drift detection to perform incremental retraining.
        
        This method is called automatically by the DriftDetectionManager when
        drift thresholds are exceeded. It uses accumulated replay buffer data
        (from online learning) to retrain gate models in-process.
        
        Returns:
            Dictionary with retrain status and details
        """
        from datetime import datetime
        
        result = {
            'triggered_at': datetime.utcnow().isoformat(),
            'status': 'pending',
            'method': None,
            'details': {},
        }
        
        logger.info("🔄 Drift-triggered incremental retrain starting...")
        
        # Method 1: Use OnlineRetrainer with replay buffer (preferred)
        if self._online_retrainer is not None and ONLINE_RETRAINER_AVAILABLE:
            try:
                # Get replay data from market intelligence
                X_replay, y_replay = None, None
                if self.market_intel is not None:
                    X_replay, y_replay = self.market_intel.get_replay_data()
                    logger.info(f"  Got {len(X_replay) if X_replay is not None else 0} samples from replay buffer")
                
                # Trigger in-process retraining
                retrain_result = self._online_retrainer.trigger_retrain(
                    X_replay=X_replay,
                    y_replay=y_replay,
                    reason="drift_detected",
                )
                
                result['status'] = retrain_result.get('status', 'unknown')
                result['method'] = 'online_retrainer'
                result['details'] = retrain_result
                
                # Mark model as updated if successful
                if result['status'] == 'completed':
                    if self.market_intel is not None:
                        self.market_intel.mark_model_updated()
                    logger.info(f"✅ Incremental retrain completed: {retrain_result.get('models_retrained', [])}")
                elif result['status'] == 'blocked':
                    logger.info(f"⏳ Retrain blocked: {retrain_result.get('blocked_reason', 'cooldown')}")
                elif result['status'] == 'skipped':
                    logger.info(f"⚠️ Retrain skipped: {retrain_result.get('skipped_reason', 'insufficient data')}")
                else:
                    logger.warning(f"❌ Retrain failed: {retrain_result}")
                
                return result
                
            except Exception as e:
                logger.warning(f"OnlineRetrainer failed: {e}, falling back to subprocess")
        
        # Method 2: Fallback to subprocess-based retrain (less ideal)
        import subprocess
        import shutil
        
        main_script = Path(__file__).parent / "main.py"
        
        if main_script.exists():
            try:
                logger.info("🔧 Fallback: Triggering retrain-gates subprocess...")
                
                cmd = [
                    shutil.which('python') or 'python',
                    str(main_script),
                    'retrain-gates',
                    '--candles', '3000',
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                
                result['status'] = 'triggered_background'
                result['method'] = 'subprocess_retrain_gates'
                result['details'] = {
                    'command': ' '.join(cmd),
                    'pid': process.pid,
                }
                
                logger.info(f"✓ Background retrain triggered (PID: {process.pid})")
                
                if self.market_intel and self.market_intel.drift_manager:
                    self.market_intel.drift_manager._last_retrain_time = datetime.utcnow()
                
            except Exception as e:
                logger.warning(f"Subprocess retrain failed: {e}")
                self._pending_retrain = True
                self._retrain_reason = f"Drift detected, retrain failed: {e}"
                
                result['status'] = 'queued'
                result['method'] = 'manual'
                result['details'] = {'error': str(e)}
        else:
            self._pending_retrain = True
            self._retrain_reason = "Drift detected, no retrain method available"
            
            result['status'] = 'queued'
            result['method'] = 'manual'
            result['details'] = {'error': 'No retrain method available'}
            
            logger.warning("⚠️ Cannot auto-retrain. Run manually: python main.py retrain-gates")
        
        return result
    
    def get_retrainer_status(self) -> Dict[str, Any]:
        """Get the current status of the online retrainer."""
        if self._online_retrainer is not None:
            return self._online_retrainer.get_status()
        return {
            'available': False,
            'reason': 'OnlineRetrainer not initialized',
        }
    
    def force_retrain(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Force an incremental retrain, bypassing cooldown.
        
        Use sparingly - this exists for manual intervention when drift is severe.
        
        Args:
            reason: Why the retrain is being forced
            
        Returns:
            Retrain result dictionary
        """
        if self._online_retrainer is None:
            return {'status': 'error', 'error': 'OnlineRetrainer not available'}
        
        X_replay, y_replay = None, None
        if self.market_intel is not None:
            X_replay, y_replay = self.market_intel.get_replay_data()
        
        return self._online_retrainer.trigger_retrain(
            X_replay=X_replay,
            y_replay=y_replay,
            reason=reason,
            force=True,
        )
    
    def record_trade_result(
        self,
        trade_id: str,
        instrument: str,
        direction: str,  # 'long' or 'short'
        entry_price: float,
        exit_price: float,
        pnl_pips: float,
        prediction: float,
        confidence: float,
        features: Optional[np.ndarray] = None,
        entry_time: Optional['datetime'] = None,
        exit_time: Optional['datetime'] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Record a completed trade for online learning and drift detection.
        
        Call this method after each trade closes to:
        1. Update the online learning buffer
        2. Feed drift detection with actual outcomes
        3. Potentially trigger automatic model retraining
        
        Args:
            trade_id: Unique trade identifier
            instrument: Trading pair (e.g., 'EUR_USD')
            direction: Trade direction ('long' or 'short')
            entry_price: Entry price
            exit_price: Exit price
            pnl_pips: Profit/loss in pips
            prediction: Model's prediction probability at entry
            confidence: Ridge confidence score at entry
            features: Feature array at trade entry (for drift detection)
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            
        Returns:
            Dictionary with drift detection result, or None if no drift
            
        Example:
            # After closing a trade:
            drift_result = ensemble.record_trade_result(
                trade_id="trade_123",
                instrument="EUR_USD",
                direction="long",
                entry_price=1.0850,
                exit_price=1.0870,
                pnl_pips=20.0,
                prediction=0.72,
                confidence=65.0,
                features=last_features,
            )
            
            if drift_result and drift_result.get('drift_detected'):
                print(f"⚠️ Drift detected: {drift_result['reason']}")
        """
        from datetime import datetime
        
        if entry_time is None:
            entry_time = datetime.utcnow()
        if exit_time is None:
            exit_time = datetime.utcnow()
        
        # Use last stored features if not provided
        if features is None and self._last_features is not None:
            features = self._last_features
        
        # Create dummy features if still None (for basic tracking)
        if features is None:
            features = np.zeros(10, dtype=np.float32)
        
        # Convert direction to int
        direction_int = 1 if direction.lower() == 'long' else 0
        
        result = None
        
        if self.market_intel is not None:
            try:
                drift_result = self.market_intel.record_trade_outcome(
                    trade_id=trade_id,
                    instrument=instrument,
                    direction=direction_int,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pips=pnl_pips,
                    features=features,
                    prediction=prediction,
                    confidence=confidence,
                )
                
                if drift_result is not None:
                    result = {
                        'drift_detected': drift_result.drift_detected,
                        'feature_drift': drift_result.feature_drift,
                        'performance_drift': drift_result.performance_drift,
                        'reason': drift_result.reason,
                        'recommendation': drift_result.recommendation,
                        'timestamp': drift_result.timestamp,
                    }
                    
                    if drift_result.drift_detected:
                        logger.warning(
                            f"📊 Drift detected after trade {trade_id}: "
                            f"{drift_result.reason} → {drift_result.recommendation}"
                        )
                        
                        # AUTO-TRIGGER: Check if retraining should happen now
                        if drift_result.recommendation in ('retrain', 'full_retrain'):
                            retrain_result = self.trigger_retraining_if_needed(async_mode=True)
                            result['retrain_triggered'] = retrain_result.get('triggered', False)
                            result['retrain_status'] = retrain_result.get('status', 'unknown')
                            if retrain_result.get('triggered'):
                                logger.info(
                                    f"🔄 Auto-retrain triggered after drift detection: "
                                    f"{retrain_result.get('result', {}).get('models_retrained', [])}"
                                )
                        
            except Exception as e:
                logger.warning(f"Failed to record trade for online learning: {e}")
        
        return result
    
    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get current drift detection status and online learning stats.
        
        Returns:
            Dictionary with:
            - drift_enabled: Whether drift detection is enabled
            - pending_retrain: Whether a retrain is queued
            - retrain_reason: Why retrain is pending (if any)
            - online_learning_stats: Buffer size, accuracy, etc.
            - drift_stats: Recent drift events, thresholds, etc.
        """
        status = {
            'drift_enabled': self.enable_drift_detection,
            'pending_retrain': self._pending_retrain,
            'retrain_reason': self._retrain_reason,
            'online_learning_stats': None,
            'drift_stats': None,
        }
        
        if self.market_intel is not None:
            # Online learning stats
            if self.market_intel.online_learner is not None:
                ol = self.market_intel.online_learner
                status['online_learning_stats'] = {
                    'buffer_size': len(ol.trade_buffer),
                    'trades_since_retrain': ol.trades_since_retrain,
                    'retrain_threshold': ol.retrain_threshold,
                    'should_retrain': ol.should_retrain(),
                    'performance': ol.get_performance_stats(),
                }
            
            # Drift detection stats
            if self.market_intel.drift_manager is not None:
                status['drift_stats'] = self.market_intel.drift_manager.get_drift_stats()
        
        return status
    
    def check_and_maybe_retrain(self) -> Optional[Dict[str, Any]]:
        """
        Check if model should be retrained and trigger if needed.
        
        This is a convenience method to manually check drift status and
        trigger retraining if thresholds are exceeded.
        
        Returns:
            Retrain result if triggered, None otherwise
        """
        if self.market_intel is None:
            return None
        
        should_update, reason = self.market_intel.should_update_model()
        
        if should_update:
            logger.info(f"🔄 Retrain triggered: {reason}")
            return self._incremental_retrain()
        
        return None
    
    def trigger_retraining_if_needed(
        self,
        force: bool = False,
        async_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Check drift and trigger model retraining if thresholds exceeded.
        
        This is the recommended method to call periodically during inference
        to ensure models stay adapted to current market conditions.
        
        Call this:
        - After recording trade outcomes
        - Periodically during idle time (e.g., every hour)
        - When drift warnings are logged
        
        Args:
            force: Bypass cooldown and daily limits (use sparingly)
            async_mode: If True, queue for background execution (non-blocking)
            
        Returns:
            Dictionary with:
            - triggered: Whether retrain was triggered
            - status: 'not_needed', 'triggered', 'blocked', 'queued', etc.
            - reason: Explanation
            - result: Retrain result if triggered
            
        Example:
            # During inference loop
            result = ensemble.trigger_retraining_if_needed()
            if result['triggered']:
                print(f"✓ Models updated: {result['result']['models_retrained']}")
        """
        response = {
            'triggered': False,
            'status': 'not_needed',
            'reason': 'No drift detected or drift detection disabled',
            'result': None,
        }
        
        # Check if drift detection is enabled
        if not self.enable_drift_detection:
            response['status'] = 'disabled'
            response['reason'] = 'Drift detection is disabled'
            return response
        
        if self.market_intel is None:
            response['status'] = 'unavailable'
            response['reason'] = 'Market intelligence not initialized'
            return response
        
        # Use MarketIntelligence's trigger method
        mi_result = self.market_intel.trigger_retraining_if_needed(
            force=force,
            queue_if_blocked=async_mode,
        )
        
        response.update(mi_result)
        
        # Clear pending flag if retrain was triggered
        if mi_result.get('triggered', False):
            self._pending_retrain = False
            self._retrain_reason = None
        elif mi_result.get('status') == 'queued':
            self._pending_retrain = True
            self._retrain_reason = mi_result.get('reason', 'Drift detected, queued for retrain')
        
        return response
    
    def process_pending_retrain(self) -> Optional[Dict[str, Any]]:
        """
        Process any pending retrain requests that were queued.
        
        Call this during idle periods to process queued retrains.
        
        Returns:
            Retrain result if processed, None if nothing pending
        """
        if not self._pending_retrain:
            return None
        
        logger.info("🔄 Processing pending retrain request...")
        result = self.trigger_retraining_if_needed(force=False)
        
        if result.get('triggered'):
            self._pending_retrain = False
            self._retrain_reason = None
        
        return result

    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get the current calibration status for debugging/verification.
        
        Returns:
            Dictionary with calibration status information:
            - enabled: Whether calibration is enabled in config
            - calibrator_exists: Whether a calibrator object exists
            - calibrator_fitted: Whether the calibrator has been trained
            - method: The calibration method being used
            - source: Where the calibrator was loaded from (if fitted)
        """
        status = {
            'enabled': self.config.enable_calibration,
            'calibrator_exists': self.calibrator is not None,
            'calibrator_fitted': self.calibrator.is_fitted if self.calibrator else False,
            'method': self.config.calibration_method,
            'source': None,
        }
        
        if self.calibrator and self.calibrator.is_fitted:
            # Determine source
            if self._calibration_loaded:
                status['source'] = 'loaded_from_file'
            else:
                status['source'] = 'unknown'
        elif self.calibrator and not self.calibrator.is_fitted:
            status['source'] = 'not_trained'
        
        return status
    
    def _get_model_path(self, model_name: str, extension: str = ".keras") -> Path:
        """
        Get the path to a model file, preferring pair-specific models.
        
        Lookup order:
        1. trained_data/models/{instrument}/{model_name}{extension}  (pair-specific)
        2. trained_data/models/{model_name}{extension}  (generic fallback)
        
        Returns the first existing path, or the generic path if none exist.
        """
        if self.instrument and self.instrument != "GENERIC":
            pair_path = self.model_dir / self.instrument / f"{model_name}{extension}"
            if pair_path.exists():
                return pair_path
        
        # Fallback to generic path
        return self.model_dir / f"{model_name}{extension}"
    
    def load_models(self, instrument: Optional[str] = None) -> None:
        """
        Load all 4 models from disk.
        
        Args:
            instrument: Optional instrument (e.g., 'EUR_USD') to load pair-specific models.
                       If None, uses self.instrument or loads generic models.
        
        Lookup order for each model:
        1. trained_data/models/{instrument}/ (pair-specific)
        2. trained_data/models/ (generic fallback)
        """
        # Update instrument if provided
        if instrument:
            self.instrument = instrument
        import warnings
        from modular_trainers import (
            TCNTrainer, TransformerDirectionTrainer, TransformerRegimeTrainer,
            XGBoostTrainer, RandomForestTrainer, RidgeTrainer,
            HistGradientBoostingDirectionTrainer
        )
        
        # Suppress XGBoost version warnings (models serialized with older versions)
        warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
        warnings.filterwarnings('ignore', message='.*serialized model.*')
        warnings.filterwarnings('ignore', message='.*older version of XGBoost.*')
        
        # Suppress sklearn version warnings (common when loading across versions)
        try:
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
        warnings.filterwarnings('ignore', message='.*unpickle estimator.*')
        
        pair_info = f" for {self.instrument}" if self.instrument and self.instrument != "GENERIC" else ""
        logger.info(f"Loading modular ensemble models{pair_info}...")
        
        # Use pair-specific paths with fallback to generic
        regime_path = self._get_model_path("transformer_regime", ".keras")
        transformer_path = self._get_model_path("transformer_direction", ".keras")
        tcn_path = self._get_model_path("tcn_direction", ".keras")
        histgb_path = self._get_model_path("histgb_direction", ".pkl")
        
        if regime_path.exists():
            # REGIME MODE
            self.regime_model = TransformerRegimeTrainer()
            self.regime_model.load(str(regime_path))
            self.use_regime = True
            logger.info(f"✓ Transformer REGIME model loaded from {regime_path}")
        elif transformer_path.exists():
            # DIRECTION MODE (Transformer)
            self.tcn = TransformerDirectionTrainer()
            self.tcn.load(str(transformer_path))
            self.use_regime = False
            logger.info(f"✓ Transformer direction model loaded from {transformer_path}")
        elif tcn_path.exists():
            # DIRECTION MODE (TCN legacy)
            self.tcn = TCNTrainer()
            self.tcn.load(str(tcn_path))
            self.use_regime = False
            logger.info(f"✓ TCN direction model loaded from {tcn_path}")
        else:
            logger.warning(f"No direction/regime model found for {self.instrument or 'generic'}")
        
        # Load HistGB for hybrid voting (if available)
        if histgb_path.exists():
            self.histgb = HistGradientBoostingDirectionTrainer()
            self.histgb.load(str(histgb_path))
            self.use_hybrid = True
            logger.info(f"✓ HistGB baseline loaded from {histgb_path}")
        else:
            self.use_hybrid = False
            logger.info("ℹ HistGB not found - single-model mode")
        
        # XGBoost (use pair-specific path)
        xgb_path = self._get_model_path("xgb_momentum", ".pkl")
        if xgb_path.exists():
            self.xgb = XGBoostTrainer()
            self.xgb.load(str(xgb_path))
            logger.info(f"✓ XGBoost loaded from {xgb_path}")
        else:
            logger.warning(f"XGBoost model not found at {xgb_path}")
        
        # Random Forest (use pair-specific path)
        rf_path = self._get_model_path("rf_risk", ".pkl")
        if rf_path.exists():
            self.rf = RandomForestTrainer()
            self.rf.load(str(rf_path))
            logger.info(f"✓ Random Forest loaded from {rf_path}")
        else:
            logger.warning(f"Random Forest model not found at {rf_path}")
        
        # Ridge (use pair-specific path)
        ridge_path = self._get_model_path("ridge_confidence", ".pkl")
        if ridge_path.exists():
            self.ridge = RidgeTrainer()
            self.ridge.load(str(ridge_path))
            logger.info(f"✓ Ridge loaded from {ridge_path}")
        else:
            logger.warning(f"Ridge model not found at {ridge_path}")
        
        # RL Position Sizer (lazy loaded to avoid TF/PyTorch GPU conflicts)
        if self.use_rl_sizer:
            RLSizer, rl_available = _lazy_load_rl_sizer()
            if rl_available and RLSizer is not None:
                self.rl_sizer = RLSizer()
                if self.rl_sizer.load():
                    logger.info("✓ RL Position Sizer loaded")
                else:
                    logger.info("ℹ RL Position Sizer not trained - using heuristic sizing")
                    self.rl_sizer = None
            else:
                logger.warning("⚠️ RL requested but dependencies not available. Install: pip install gymnasium stable-baselines3")
                self.rl_sizer = None
        
        # Load confidence calibration if available
        if self.config.enable_calibration:
            self._load_calibration()
        
        # Load meta-labeler if available (predicts trade SUCCESS, not direction)
        if self.config.enable_meta_labeling:
            self._load_meta_labeler()
        
        # Auto-detect sklearn version mismatch and enable permissive mode
        self._check_sklearn_version_mismatch()
        
        self._loaded = True
        self._loaded_instrument = self.instrument
        logger.info(f"Modular ensemble loaded{pair_info}.")
    
    def _check_sklearn_version_mismatch(self) -> None:
        """
        Check sklearn version compatibility for each gate model independently.
        
        GRACEFUL DEGRADATION: Instead of all-or-nothing permissive mode,
        we track which gates are usable based on:
        1. Version compatibility (major.minor match)
        2. Model quality (MAE thresholds)
        3. Load success
        
        Gates that fail checks are marked unusable but don't disable others.
        """
        import sklearn
        current_version = sklearn.__version__
        current_major_minor = current_version.split('.')[0:2]
        
        # Track gate status: True = usable, False = skip (use fallback)
        self._gate_status = {
            'xgboost': True,
            'random_forest': True,
            'ridge': True,
        }
        self._gate_issues = {}
        
        # Check XGBoost version compatibility
        if self.xgb and hasattr(self.xgb, '_saved_sklearn_version'):
            saved_ver = self.xgb._saved_sklearn_version
            if saved_ver:
                saved_major_minor = saved_ver.split('.')[0:2]
                if saved_major_minor != current_major_minor:
                    self._gate_status['xgboost'] = False
                    self._gate_issues['xgboost'] = f"sklearn {saved_ver} → {current_version}"
                    logger.warning(f"⚠️ XGBoost gate: sklearn version mismatch ({saved_ver} → {current_version})")
        
        # Check RandomForest version compatibility
        if self.rf and hasattr(self.rf, '_saved_sklearn_version'):
            saved_ver = self.rf._saved_sklearn_version
            if saved_ver:
                saved_major_minor = saved_ver.split('.')[0:2]
                if saved_major_minor != current_major_minor:
                    self._gate_status['random_forest'] = False
                    self._gate_issues['random_forest'] = f"sklearn {saved_ver} → {current_version}"
                    logger.warning(f"⚠️ RF gate: sklearn version mismatch ({saved_ver} → {current_version})")
        
        # Check Ridge version compatibility
        if self.ridge and hasattr(self.ridge, '_saved_sklearn_version'):
            saved_ver = self.ridge._saved_sklearn_version
            if saved_ver:
                saved_major_minor = saved_ver.split('.')[0:2]
                if saved_major_minor != current_major_minor:
                    self._gate_status['ridge'] = False
                    self._gate_issues['ridge'] = f"sklearn {saved_ver} → {current_version}"
                    logger.warning(f"⚠️ Ridge gate: sklearn version mismatch ({saved_ver} → {current_version})")
        
        # Check model quality from metadata
        meta_path = self.model_dir / "modular_ensemble.meta.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            
            # Check for poor RF model quality (drawdown MAE > 50% means useless)
            results = meta.get('results', {})
            rf_results = results.get('random_forest', {})
            drawdown_mae_bps = rf_results.get('drawdown_mae_bps', 0)
            
            if drawdown_mae_bps > 5000:  # > 50% MAE = unreliable
                self._gate_status['random_forest'] = False
                self._gate_issues['random_forest'] = f"High MAE ({drawdown_mae_bps/100:.1f}%)"
                logger.warning(f"⚠️ RF gate: high error (MAE={drawdown_mae_bps/100:.1f}%)")
        
        # Count usable gates
        usable_gates = sum(self._gate_status.values())
        total_gates = len(self._gate_status)
        
        if usable_gates < total_gates:
            # Some gates have issues - enable permissive mode but with partial degradation
            self.config.permissive_mode = True
            logger.info(f"📊 Gate status: {usable_gates}/{total_gates} gates usable. "
                       f"Issues: {self._gate_issues}")
            if usable_gates == 0:
                logger.warning("⚠️ All sklearn gates have issues - using Transformer-only mode")
                logger.info("💡 Run 'python retrain_gates.py' to fix sklearn version mismatch")
        else:
            logger.info(f"✓ All {total_gates} gates compatible with sklearn {current_version}")

    def _load_calibration(self) -> None:
        """
        Load confidence calibration from model metadata or standalone calibration file.
        
        Calibration sources (in priority order):
        1. Standalone calibration file: trained_data/models/confidence_calibrator.pkl
        2. Pair-specific calibration: trained_data/models/{instrument}/confidence_calibrator.pkl
        3. Calibration data in model metadata: modular_ensemble.meta.json -> calibration
        4. Calibration data in Transformer metadata: transformer_direction.meta.pkl -> output_calibration
        """
        if not CALIBRATION_AVAILABLE:
            logger.debug("Confidence calibration module not available")
            return
        
        self._calibration_loaded = False
        self.calibrator = None
        
        # Try loading standalone calibration file first
        calibration_paths = [
            self.model_dir / "confidence_calibrator.pkl",
        ]
        
        # Add pair-specific path if instrument is set
        if self.instrument and self.instrument != "GENERIC":
            calibration_paths.insert(0, self.model_dir / self.instrument / "confidence_calibrator.pkl")
        
        for calib_path in calibration_paths:
            if calib_path.exists():
                try:
                    self.calibrator = ConfidenceCalibrator.load(calib_path)
                    self._calibration_loaded = True
                    logger.info(f"✓ Confidence calibrator loaded from {calib_path}")
                    logger.info(f"  Method: {self.calibrator.config.method}, Fitted: {self.calibrator.is_fitted}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load calibrator from {calib_path}: {e}")
        
        # Try loading from model metadata
        meta_path = self.model_dir / "modular_ensemble.meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                
                # Check for calibration data in metadata
                calib_data = meta.get('calibration')
                if calib_data and isinstance(calib_data, dict):
                    # Create calibrator from metadata
                    config = CalibrationConfig(
                        method=calib_data.get('method', self.config.calibration_method),
                        min_confidence_threshold=calib_data.get('min_threshold', 0.5),
                        max_confidence_threshold=calib_data.get('max_threshold', 0.95),
                    )
                    self.calibrator = ConfidenceCalibrator(config)
                    
                    # Restore fitted state from metadata if available
                    if 'platt_params' in calib_data:
                        # Reconstruct Platt model from saved parameters
                        from sklearn.linear_model import LogisticRegression
                        self.calibrator.platt_model = LogisticRegression()
                        self.calibrator.platt_model.coef_ = np.array([calib_data['platt_params']['coef']])
                        self.calibrator.platt_model.intercept_ = np.array([calib_data['platt_params']['intercept']])
                        self.calibrator.platt_model.classes_ = np.array([0, 1])
                        self.calibrator.is_fitted = True
                        self._calibration_loaded = True
                        logger.info(f"✓ Platt calibration loaded from metadata")
                    
                    if 'isotonic_params' in calib_data:
                        # Reconstruct Isotonic model from saved parameters
                        from sklearn.isotonic import IsotonicRegression
                        self.calibrator.isotonic_model = IsotonicRegression(out_of_bounds='clip')
                        self.calibrator.isotonic_model.X_thresholds_ = np.array(calib_data['isotonic_params']['X_thresholds'])
                        self.calibrator.isotonic_model.y_thresholds_ = np.array(calib_data['isotonic_params']['y_thresholds'])
                        self.calibrator.isotonic_model.f_ = None  # Will be rebuilt on predict
                        self.calibrator.is_fitted = True
                        self._calibration_loaded = True
                        logger.info(f"✓ Isotonic calibration loaded from metadata")
                    
                    return
            except Exception as e:
                logger.warning(f"Failed to load calibration from metadata: {e}")
        
        # Create default calibrator (unfitted) if calibration enabled but no data found
        # This allows the _apply_calibration method to gracefully fallback to raw probabilities
        if self.config.enable_calibration:
            config = CalibrationConfig(
                method=self.config.calibration_method,
                min_confidence_threshold=0.5,
                max_confidence_threshold=0.95,
            )
            self.calibrator = ConfidenceCalibrator(config)
            # Note: _calibration_loaded stays False, but calibrator.is_fitted is also False
            # _apply_calibration will check is_fitted and gracefully return raw probability
            self._calibration_loaded = False  # Explicitly set to indicate no fitted calibrator
            logger.info(f"ℹ Calibration enabled but no fitted calibrator found - using raw probabilities")
            logger.info(f"  To enable calibration, train with: python main.py train-buddy --calibrate")
            logger.info(f"  Or run: python -m confidence_calibration --train --model-dir {self.model_dir}")

    def _load_meta_labeler(self) -> None:
        """
        Load meta-labeler model for trade filtering.
        
        The meta-labeler predicts whether the primary model's signal will result
        in a profitable trade. This is DIFFERENT from direction prediction:
        - Primary model: "Signal is LONG"
        - Meta-labeler: "This signal has X% chance of being correct"
        
        If meta-confidence < threshold, we skip the trade even if direction is clear.
        
        Lookup order:
        1. trained_data/models/{instrument}/meta_labeler.pkl  (pair-specific)
        2. trained_data/models/meta_labeler.pkl  (generic fallback)
        """
        if not META_LABELING_AVAILABLE:
            logger.debug("Meta-labeling module not available")
            return
        
        self._meta_labeler_loaded = False
        self.meta_labeler = None
        
        # Build list of paths to check
        meta_labeler_paths = [
            self.model_dir / "meta_labeler.pkl",
        ]
        
        # Add pair-specific path first if instrument is set
        if self.instrument and self.instrument != "GENERIC":
            meta_labeler_paths.insert(0, self.model_dir / self.instrument / "meta_labeler.pkl")
        
        for path in meta_labeler_paths:
            if path.exists():
                try:
                    self.meta_labeler = MetaLabeler.load(path)
                    self._meta_labeler_loaded = True
                    threshold = self.config.min_meta_confidence
                    primary_acc = getattr(self.meta_labeler, '_primary_accuracy', 0.5)
                    logger.info(f"✓ Meta-labeler loaded from {path}")
                    logger.info(f"  Threshold: {threshold:.0%}, Primary accuracy: {primary_acc:.1%}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load meta-labeler from {path}: {e}")
        
        # No meta-labeler found
        if self.config.enable_meta_labeling:
            logger.info("ℹ Meta-labeling enabled but no trained model found")
            logger.info("  To train: models save meta_labeler.pkl during buddy training")

    def _apply_calibration(self, raw_probability: float, direction: Optional[int] = None) -> tuple[float, bool]:
        """
        Apply confidence calibration to raw model probability.
        
        Args:
            raw_probability: Raw probability from TCN/Transformer (0-1)
            direction: Predicted direction (0=short, 1=long) for directional adjustment
            
        Returns:
            Tuple of (calibrated_probability, was_calibrated)
        """
        # Check if calibrator exists and is fitted
        # We rely on calibrator.is_fitted rather than _calibration_loaded for robustness
        if self.calibrator is None:
            return raw_probability, False
        
        if not self.calibrator.is_fitted:
            # Calibrator exists but not trained - graceful fallback to raw probability
            logger.debug("Calibrator not fitted, using raw probability")
            return raw_probability, False
        
        try:
            result = self.calibrator.calibrate_confidence(raw_probability)
            calibrated = result.calibrated_confidence
            
            # Log calibration adjustment if significant
            adjustment = calibrated - raw_probability
            if abs(adjustment) > 0.02:
                logger.debug(f"📐 Calibration: {raw_probability:.3f} → {calibrated:.3f} (Δ={adjustment:+.3f})")
            
            return calibrated, True
        except Exception as e:
            logger.warning(f"Calibration failed, using raw probability: {e}")
            return raw_probability, False

    def _find_features_by_pattern(self, df: pd.DataFrame, patterns: List[str]) -> List[str]:
        """Find features by partial name matching."""
        found = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    if col not in found:
                        found.append(col)
                    break
        return found
    
    def _extract_features_by_names(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]],
        fallback_preferred: List[str],
        fallback_patterns: List[str],
        exclude: List[str] = None,
    ) -> np.ndarray:
        """
        Extract features using saved feature names from training.
        
        CRITICAL: If feature_names is provided, we MUST return features in that
        exact order with the exact count. Missing features are filled with 0.0
        to maintain compatibility with the trained model's scaler and weights.
        
        Falls back to pattern matching only if feature_names is not available.
        """
        exclude = exclude or ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp']
        
        # If we have saved feature names from training, use them with proper ordering
        if feature_names:
            # Build array with exact feature count and order
            n_features = len(feature_names)
            n_rows = len(df)
            result = np.zeros((n_rows, n_features), dtype=np.float32)
            
            missing_features = []
            for i, fname in enumerate(feature_names):
                if fname in df.columns:
                    result[:, i] = df[fname].values.astype(np.float32)
                else:
                    missing_features.append(fname)
            
            if missing_features and len(missing_features) < len(feature_names) // 2:
                # Log missing features but continue (fill with 0)
                logger.debug(f"Missing {len(missing_features)} features (filled with 0): {missing_features[:5]}...")
            elif missing_features:
                logger.warning(f"Many features missing ({len(missing_features)}/{n_features}). "
                              f"First 10: {missing_features[:10]}")
            
            return result
        
        # Fallback: Try exact matches first
        available = [f for f in fallback_preferred if f in df.columns]
        
        # If not enough, try pattern matching
        if len(available) < 5:
            pattern_features = self._find_features_by_pattern(df, fallback_patterns)
            for f in pattern_features:
                if f not in available and f not in exclude:
                    available.append(f)
        
        # If still not enough, use any numeric columns
        if len(available) < 5:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col not in available and col not in exclude:
                    available.append(col)
        
        if not available:
            # Last resort: use all numeric columns
            available = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return df[available].values.astype(np.float32)
    
    def _add_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add computed directional features to dataframe for inference.
        Must match the features computed during training.
        """
        df = df.copy()
        
        # SMA crossover: 1 if sma_5 > sma_20, else 0
        if 'sma_5' in df.columns and 'sma_20' in df.columns:
            df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(np.float32)
        
        # MACD crossover: 1 if macd > signal, else 0
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(np.float32)
        
        # Higher high count: How many of last 10 bars made higher highs
        if 'high' in df.columns:
            highs = df['high'].values
            hh_count = np.zeros(len(highs), dtype=np.float32)
            for i in range(10, len(highs)):
                count = 0
                for j in range(1, 10):
                    if highs[i-j] > highs[i-j-1]:
                        count += 1
                hh_count[i] = count / 9.0  # Normalize to 0-1
            df['higher_high_count'] = hh_count
        
        # Lower low count: How many of last 10 bars made lower lows
        if 'low' in df.columns:
            lows = df['low'].values
            ll_count = np.zeros(len(lows), dtype=np.float32)
            for i in range(10, len(lows)):
                count = 0
                for j in range(1, 10):
                    if lows[i-j] < lows[i-j-1]:
                        count += 1
                ll_count[i] = count / 9.0  # Normalize to 0-1
            df['lower_low_count'] = ll_count
        
        # Volume direction: Are up bars getting more volume?
        if 'volume' in df.columns and 'close' in df.columns:
            close = df['close'].values
            volume = df['volume'].values
            vol_dir = np.zeros(len(close), dtype=np.float32)
            for i in range(10, len(close)):
                up_vol = 0.0
                down_vol = 0.0
                for j in range(10):
                    if close[i-j] > close[i-j-1]:
                        up_vol += volume[i-j]
                    else:
                        down_vol += volume[i-j]
                total = up_vol + down_vol
                vol_dir[i] = up_vol / max(total, 1e-8) if total > 0 else 0.5
            df['volume_direction'] = vol_dir
        
        # Trend direction: Combined signal from multiple indicators
        trend_components = []
        if 'sma_cross_5_20' in df.columns:
            trend_components.append(df['sma_cross_5_20'].values)
        if 'macd_cross' in df.columns:
            trend_components.append(df['macd_cross'].values)
        if 'rsi' in df.columns:
            trend_components.append((df['rsi'].values > 50).astype(np.float32))
        
        if trend_components:
            df['trend_direction'] = np.mean(trend_components, axis=0)
        
        return df
    
    def _extract_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for regime classification model."""
        # Features that describe market state (regime indicators)
        regime_features = [
            # Trend strength
            'adx', 'trend_strength',
            # Volatility state
            'atr_pct_14', 'atr_pct_20', 'volatility_10', 'volatility_20',
            # Mean reversion signals
            'zscore_20', 'zscore_50', 'rsi', 'rsi_norm',
            'bb_position_20', 'pct_rank_20', 'pct_rank_50',
            # Momentum consistency
            'returns_1', 'returns_5', 'returns_10', 'returns_20',
            # Crossover state
            'sma_cross_5_20', 'macd_cross',
            # Volume context
            'volume_ratio_10', 'volume_zscore',
        ]
        
        fallback_patterns = ['adx', 'rsi', 'zscore', 'returns', 'volatility', 'atr_pct', 'pct_rank']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.regime_model, 'feature_names', None) if self.regime_model else None
        return self._extract_features_by_names(df, feature_names, regime_features, fallback_patterns)
    
    def _extract_tcn_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for direction model (TCN or Transformer) using saved feature names."""
        # NORMALIZED features for instrument-agnostic inference
        normalized_features = get_normalized_feature_names()['direction']
        
        # Legacy fallback features
        legacy_fallback = [
            'adx', 'macd', 'macd_signal', 'macd_hist',
            'rsi', 'stoch_k', 'stoch_d',
            'returns', 'momentum_10', 'roc_5', 'roc_10',
            'volatility_20', 'sma_cross_5_20', 'macd_cross',
        ]
        
        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['return', 'zscore', 'ratio', 'norm', 'pct_rank', 'cross', 'rsi', 'macd']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.tcn, 'feature_names', None) if self.tcn else None
        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
    
    def _extract_xgb_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for XGBoost model using saved feature names."""
        # NORMALIZED features for instrument-agnostic inference
        normalized_features = get_normalized_feature_names()['momentum']
        
        # Legacy fallback features
        legacy_fallback = [
            'returns', 'roc_5', 'roc_10', 'roc_20',
            'high_low_ratio', 'momentum_10', 'macd', 'macd_hist',
            'stoch_k', 'stoch_d', 'mfi',
        ]
        
        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['return', 'atr_pct', 'volatility', 'volume_ratio', 'macd_norm', 'rsi_norm']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.xgb, 'feature_names', None) if self.xgb else None
        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
    
    def _extract_rf_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for Random Forest model using saved feature names."""
        # NORMALIZED features for instrument-agnostic inference
        normalized_features = get_normalized_feature_names()['risk']
        
        # Legacy fallback features
        legacy_fallback = [
            'atr', 'volatility_5', 'volatility_10', 'volatility_20',
            'high_low_ratio', 'bb_width_20',
            'returns', 'momentum_10',
        ]
        
        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['atr_pct', 'volatility', 'tr_pct', 'hl_range', 'zscore', 'return']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.rf, 'feature_names', None) if self.rf else None
        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
    
    def _extract_ridge_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for Ridge model using saved feature names."""
        # NORMALIZED features for instrument-agnostic inference
        normalized_features = get_normalized_feature_names()['confidence']
        
        # Legacy fallback features
        legacy_fallback = [
            'volatility_5', 'volatility_10', 'volatility_20',
            'atr', 'bb_width_20', 'bb_position_20',
            'adx', 'returns',
        ]
        
        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['atr_pct', 'volatility', 'volume_ratio', 'sma_ratio', 'return', 'zscore']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.ridge, 'feature_names', None) if self.ridge else None
        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
    
    def _compute_confidence_direct(self, df: pd.DataFrame) -> float:
        """
        Compute confidence score directly from indicators (0-100 scale).
        
        This bypasses the ElasticNet model and uses the formula directly,
        which is faster and avoids learning a synthetic target.
        
        Formula weights:
        - ADX (40%): Trend strength - higher = more confident
        - Volatility (20%): Moderate vol = high confidence
        - RSI (15%): Not extreme = high confidence
        - BB Position (10%): Position within Bollinger Bands
        - Volume (15%): Above average volume = confirmation
        """
        import numpy as np
        
        # Get the last row of data
        row = df.iloc[-1]
        
        # Extract indicators with safe defaults
        adx = row.get('adx', 25.0) if 'adx' in df.columns else 25.0
        rsi = row.get('rsi', 50.0) if 'rsi' in df.columns else 50.0
        atr_pct = row.get('atr_pct_14', 0.01) if 'atr_pct_14' in df.columns else 0.01
        bb_pos = row.get('bb_position_20', 0.5) if 'bb_position_20' in df.columns else 0.5
        volume_ratio = row.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in df.columns else 1.0
        
        # Handle NaN values
        adx = float(adx) if not np.isnan(adx) else 25.0
        rsi = float(rsi) if not np.isnan(rsi) else 50.0
        atr_pct = float(atr_pct) if not np.isnan(atr_pct) else 0.01
        bb_pos = float(bb_pos) if not np.isnan(bb_pos) else 0.5
        volume_ratio = float(volume_ratio) if not np.isnan(volume_ratio) else 1.0
        
        # ADX score (40%) - higher ADX = more confident trend
        # Use percentile-based scaling: 15-35 is typical range
        adx_normalized = (adx - 15) / 20  # 15->0, 35->1
        adx_score = np.clip(adx_normalized * 0.5 + 0.25, 0.0, 1.0)
        
        # Volatility score (20%) - moderate vol is ideal
        if atr_pct < 0.005:  # Too low - no movement
            vol_score = 0.5
        elif atr_pct > 0.02:  # Too high - chaotic
            vol_score = max(0.0, 1.0 - (atr_pct - 0.02) / 0.02)
        else:  # Sweet spot
            vol_score = 0.8 + 0.2 * (1.0 - abs(atr_pct - 0.01) / 0.01)
        vol_score = np.clip(vol_score, 0.0, 1.0)
        
        # RSI score (15%) - not extreme = high confidence
        rsi_distance = abs(rsi - 50)
        rsi_score = max(0.0, 1.0 - rsi_distance / 30.0)
        
        # BB position score (10%) - extremes can be good for reversals
        if bb_pos < 0.2 or bb_pos > 0.8:  # Near bands
            bb_score = 0.7
        else:  # Middle zone
            bb_score = 0.5 + 0.5 * (1.0 - abs(bb_pos - 0.5) * 2)
        bb_score = np.clip(bb_score, 0.0, 1.0)
        
        # Volume confirmation (15%) - above average = confirmation
        vol_conf_score = np.clip((volume_ratio - 0.5) / 1.0, 0.0, 1.0)
        
        # Combine with weights
        raw_conf = (
            adx_score * 0.40 +
            vol_score * 0.20 +
            rsi_score * 0.15 +
            bb_score * 0.10 +
            vol_conf_score * 0.15
        )
        
        # Map to 15-95 range (never fully 0 or 100)
        confidence = 15 + raw_conf * 80
        
        return float(confidence)
    
    def _calculate_position_size(
        self,
        expected_drawdown_pips: float,
        equity: Optional[float] = None,
        instrument: Optional[str] = None,
        features: Optional[np.ndarray] = None,
        tcn_probability: float = 0.5,
        ridge_confidence: float = 50.0,
    ) -> float:
        """
        Calculate position size for target risk percentage with liquidity limits.
        
        SUPPORTS RL POSITION SIZING:
        If rl_sizer is loaded and features are provided, uses learned optimal sizing.
        Otherwise falls back to heuristic risk-based sizing.
        
        Formula (heuristic): size = (equity * risk_pct) / (stop_loss_pips * pip_value)
        
        CRITICAL: Large positions cause slippage that destroys edge!
        - 10 lots on EUR_USD = 6+ pips slippage
        - 1 lot on EUR_USD = <1 pip slippage
        
        Args:
            expected_drawdown_pips: Stop loss distance in pips
            equity: Account equity (default: config value)
            instrument: Trading pair for liquidity limit lookup
            features: Market features for RL sizer (optional)
            tcn_probability: TCN direction probability for RL observation
            ridge_confidence: Ridge confidence score for RL observation
        
        Returns:
            Position size in lots, capped by liquidity limits
        """
        equity = equity or self.config.account_equity
        
        # =====================================================================
        # RL POSITION SIZING (if available)
        # =====================================================================
        if self.rl_sizer is not None and features is not None:
            try:
                # Construct ensemble prediction for RL observation
                ensemble_pred = np.array([tcn_probability, ridge_confidence / 100.0])
                
                # Get RL-optimized position size (returns $ amount)
                size_dollars = self.rl_sizer.get_position_size(
                    features=features[-1] if features.ndim > 1 else features,
                    ensemble_prediction=ensemble_pred,
                    account_equity=equity,
                )
                
                # Convert to lots (assume ~$100k per lot)
                size_lots = size_dollars / 100000
                
                # Apply liquidity limit
                max_lots = self.config.max_lots_by_pair.get(
                    instrument, 
                    self.config.max_lots_by_pair.get('DEFAULT', 0.5)
                ) if instrument else 1.0
                size_lots = min(size_lots, max_lots)
                
                logger.debug(f"RL position size: {size_lots:.2f} lots (${size_dollars:.0f})")
                return round(max(0.01, size_lots), 2)
                
            except Exception as e:
                logger.warning(f"RL sizing failed, falling back to heuristic: {e}")
        
        # =====================================================================
        # HEURISTIC RISK-BASED SIZING (fallback)
        # =====================================================================
        risk_amount = equity * self.config.risk_per_trade_pct
        
        # Minimum stop loss to prevent oversizing
        if expected_drawdown_pips <= 5.0:
            expected_drawdown_pips = 10.0  # Minimum 10 pip stop
        
        # Risk-based position size: lots = risk_$ / (pips * pip_value)
        # pip_value ~= $10 per pip per standard lot for most pairs
        size_lots = risk_amount / (expected_drawdown_pips * self.config.pip_value)
        
        # Apply liquidity limit for instrument
        max_lots = self.config.max_lots_by_pair.get(
            instrument, 
            self.config.max_lots_by_pair.get('DEFAULT', 0.5)
        ) if instrument else 1.0
        
        # Hard cap at liquidity limit
        size_lots = min(size_lots, max_lots)
        
        # Minimum position size
        size_lots = max(0.01, size_lots)
        
        return round(size_lots, 2)
    
    def predict(
        self,
        df: pd.DataFrame,
        equity: Optional[float] = None,
        instrument: Optional[str] = None,
        headlines: Optional[List[str]] = None,  # NEW: Optional news headlines for sentiment
    ) -> TradeSignal:
        """
        Run inference through all models and apply gates.
        
        REGIME MODE:
        - Transformer classifies regime (trend/chop/mean_revert)
        - TREND: Direction from XGBoost momentum sign
        - CHOP: Skip trading entirely
        - MEAN_REVERT: Fade 2-bar momentum
        
        DIRECTION MODE (legacy):
        - Transformer/TCN predicts direction directly
        
        IMPORTANT: Computes normalized features first for instrument-agnostic inference.
        
        Args:
            df: DataFrame with features (must have all required columns)
            equity: Account equity for position sizing
            instrument: Trading pair (e.g., 'EUR_USD') for liquidity limits
            headlines: Optional list of news headlines for sentiment analysis
        
        Returns:
            TradeSignal with trade decision and all model outputs
        """
        if not self._loaded:
            self.load_models()
        
        # Store instrument for position sizing
        self._current_instrument = instrument
        
        # === MARKET INTELLIGENCE PRE-TRADE CHECK (NEW) ===
        intel_data = {}
        if self.market_intel and instrument:
            try:
                if headlines is None and fetch_forex_news is not None:
                    headlines = fetch_forex_news(instrument)
                    intel_data['headlines_count'] = len(headlines)

                can_trade, block_reason, intel_data = self.market_intel.pre_trade_check(
                    instrument,
                    headlines=headlines,
                )

                if headlines is not None and 'headlines_count' not in intel_data:
                    intel_data['headlines_count'] = len(headlines)
                
                if not can_trade:
                    # Blocked by economic event or sentiment
                    logger.info(f"🚫 Trade blocked by market intelligence: {block_reason}")
                    return TradeSignal(
                        trade=False,
                        direction=None,
                        size=0.0,
                        confidence=0.0,
                        reason=f"Market Intelligence: {block_reason}",
                        regime=None,
                        regime_confidence=0.0,
                        tcn_direction=None,
                        tcn_probability=0.5,
                        ridge_confidence=0.0,
                        xgb_momentum=0.0,
                        xgb_acceleration=False,
                        rf_drawdown_pips=0.0,
                        rf_streak_prob=0.0,
                        metadata={'intel_data': intel_data},
                    )
                
                # Log intelligence insights
                if 'sentiment' in intel_data:
                    sent = intel_data['sentiment']
                    logger.info(f"📰 Sentiment: {sent['aggregate_label']} ({sent['aggregate_score']:+.2f}) "
                               f"from {sent['num_headlines']} headlines")
                
                if 'next_high_impact' in intel_data:
                    event = intel_data['next_high_impact']
                    logger.info(f"📅 Next high-impact: {event['name']} ({event['currency']}) "
                               f"in {int(event['minutes_until'])} minutes")
            
            except Exception as e:
                logger.warning(f"Market intelligence check failed: {e}")
                intel_data = {'error': str(e)}
        
        # FIRST: Compute normalized features for instrument-agnostic inference
        if 'returns_1' not in df.columns:
            df = compute_normalized_features(df)
        
        # Initialize with defaults
        regime = None
        regime_confidence = 0.0
        tcn_direction = None
        tcn_probability = 0.5
        ridge_confidence = 0.0
        xgb_momentum = 0.0
        xgb_acceleration = False
        rf_drawdown_pips = 100.0
        rf_drawdown_pct = 1.0
        rf_streak_prob = 1.0
        
        # === GET REGIME OR DIRECTION ===
        if self.use_regime and self.regime_model is not None:
            # REGIME MODE
            try:
                regime_features = self._extract_regime_features(df)
                regime_pred = self.regime_model.predict(regime_features)
                regime = regime_pred['regime_name']  # 'trend', 'chop', 'mean_revert'
                regime_confidence = regime_pred['confidence']
                logger.debug(f"Regime: {regime} (confidence={regime_confidence:.2f})")
            except Exception as e:
                logger.warning(f"Regime prediction failed: {e}")
                regime = 'chop'  # Default to skip on error
        else:
            # DIRECTION MODE (legacy)
            try:
                if self.tcn is not None:
                    tcn_features = self._extract_tcn_features(df)
                    tcn_pred = self.tcn.predict(tcn_features)
                    tcn_direction = tcn_pred['direction']
                    tcn_probability = tcn_pred['probability']
            except Exception as e:
                logger.warning(f"TCN prediction failed: {e}")
        
        # === HYBRID VOTING (HistGB + Transformer) ===
        histgb_direction = None
        histgb_probability = 0.5
        models_agree = True
        
        if self.use_hybrid and self.histgb is not None and not self.use_regime:
            try:
                histgb_features = self._extract_tcn_features(df)  # Same features as Transformer
                histgb_pred = self.histgb.predict(histgb_features)
                histgb_direction = histgb_pred['direction']
                histgb_probability = histgb_pred['probability']
                
                # Calculate confidence (distance from 0.5)
                tcn_confidence = abs(tcn_probability - 0.5) * 2  # 0-1 scale
                histgb_confidence = abs(histgb_probability - 0.5) * 2  # 0-1 scale
                
                # Check if models agree
                if tcn_direction is not None and histgb_direction is not None:
                    models_agree = (tcn_direction == histgb_direction)
                    
                    if models_agree:
                        # Both agree - boost confidence
                        tcn_probability = (tcn_probability + histgb_probability) / 2
                        logger.debug(f"Hybrid: AGREE (Transformer={tcn_direction}, HistGB={histgb_direction})")
                    else:
                        # Models disagree - use confidence-weighted decision
                        # Only trust HistGB if it's significantly more confident
                        atr_pct = df['atr_pct_14'].iloc[-1] if 'atr_pct_14' in df.columns else 0.01
                        
                        # NEW: Consider confidence, not just volatility
                        # HistGB wins only if: (1) higher confidence AND (2) low volatility
                        if atr_pct < 0.005 and histgb_confidence > tcn_confidence * 1.2:
                            # Low-vol AND HistGB is 20%+ more confident - trust HistGB
                            tcn_direction = histgb_direction
                            tcn_probability = histgb_probability
                            logger.debug(f"Hybrid: DISAGREE, low-vol + HistGB more confident -> HistGB ({histgb_direction})")
                        else:
                            # Trust Transformer (primary model), reduce confidence due to disagreement
                            tcn_probability = tcn_probability * 0.8
                            logger.debug(f"Hybrid: DISAGREE -> Transformer ({tcn_direction}), conf={tcn_confidence:.2f} vs HistGB={histgb_confidence:.2f}")
            except Exception as e:
                logger.warning(f"HistGB prediction failed: {e}")
        
        # === APPLY CONFIDENCE CALIBRATION (NEW) ===
        # Calibrate raw TCN/Transformer probability before gate checks
        raw_tcn_probability = tcn_probability
        calibration_applied = False
        
        if self.config.enable_calibration and tcn_probability is not None:
            tcn_probability, calibration_applied = self._apply_calibration(
                tcn_probability, 
                direction=tcn_direction
            )
            if calibration_applied:
                logger.debug(f"📐 TCN probability calibrated: {raw_tcn_probability:.3f} → {tcn_probability:.3f}")
        
        # Store calibration info in intel_data for reporting
        intel_data['calibration'] = {
            'enabled': self.config.enable_calibration,
            'applied': calibration_applied,
            'raw_probability': raw_tcn_probability,
            'calibrated_probability': tcn_probability,
            'method': self.calibrator.config.method if self.calibrator and self.calibrator.is_fitted else 'none',
            'calibrator_fitted': self.calibrator.is_fitted if self.calibrator else False,
            'adjustment': tcn_probability - raw_tcn_probability if calibration_applied else 0.0,
        }
        
        # === GET SUPPORTING MODEL PREDICTIONS ===
        try:
            # OPTION A: Use direct formula instead of Ridge model (faster, more reliable)
            # This computes confidence from indicators directly rather than learning it
            ridge_confidence = self._compute_confidence_direct(df)
            logger.debug(f"Direct confidence formula: {ridge_confidence:.1f}")
        except Exception as e:
            logger.warning(f"Direct confidence calculation failed: {e}, falling back to Ridge")
            try:
                if self.ridge is not None:
                    ridge_features = self._extract_ridge_features(df)
                    ridge_pred = self.ridge.predict(ridge_features)
                    ridge_confidence = ridge_pred['confidence']
            except Exception as e2:
                logger.warning(f"Ridge prediction also failed: {e2}")
                ridge_confidence = 50.0  # Neutral default
        
        try:
            if self.xgb is not None:
                xgb_features = self._extract_xgb_features(df)
                xgb_pred = self.xgb.predict(xgb_features)
                xgb_momentum = xgb_pred['momentum']
                xgb_acceleration = xgb_pred['acceleration']
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
        
        try:
            if self.rf is not None:
                rf_features = self._extract_rf_features(df)
                rf_pred = self.rf.predict(rf_features)
                rf_drawdown_pct = rf_pred.get('expected_drawdown_pct', rf_pred.get('expected_drawdown_pips', 0) / 10000)
                rf_drawdown_pips = rf_pred.get('expected_drawdown_pips', rf_drawdown_pct * 10000)
                rf_streak_prob = rf_pred['streak_prob']
        except Exception as e:
            logger.warning(f"Random Forest prediction failed: {e}")
        
        # === APPLY GATES ===
        # SMART GATING: Transformer probability is the primary signal
        # Gate models provide confirmation, but TCN confidence can override
        #
        # GRACEFUL DEGRADATION: When permissive_mode is enabled, we check
        # _gate_status to determine which gates to use vs bypass
        
        # Initialize gate status tracking if not done during load
        if not hasattr(self, '_gate_status'):
            self._gate_status = {'xgboost': True, 'random_forest': True, 'ridge': True}
            self._gate_issues = {}
        
        if self.config.permissive_mode:
            # GRACEFUL DEGRADATION: Use gates that are still usable
            
            # Confidence gate: use Ridge if available, else pass
            if self._gate_status.get('ridge', False) and self.ridge is not None:
                confidence_gate_passed = ridge_confidence >= self.config.min_confidence
                logger.debug(f"Ridge gate ACTIVE: confidence={ridge_confidence:.1f}, passed={confidence_gate_passed}")
            else:
                confidence_gate_passed = True
                logger.warning(f"⚠️  Ridge gate BYPASSED (permissive mode): {self._gate_issues.get('ridge', 'not loaded')}")
            
            # Momentum gate: use XGBoost if available, else pass
            if self._gate_status.get('xgboost', False) and self.xgb is not None:
                momentum_fresh = xgb_momentum >= self.config.min_momentum
                momentum_gate_passed = momentum_fresh or xgb_acceleration
                logger.debug(f"XGBoost gate ACTIVE: momentum={xgb_momentum:.3f}, accel={xgb_acceleration}, passed={momentum_gate_passed}")
            else:
                momentum_gate_passed = True
                logger.warning(f"⚠️  XGBoost gate BYPASSED (permissive mode): {self._gate_issues.get('xgboost', 'not loaded')}")
            
            # Risk gate: use RF if available AND not explicitly bypassed
            if self.config.bypass_risk_gate_in_permissive:
                risk_gate_passed = True
                logger.warning("⚠️  Risk gate BYPASSED (permissive mode): bypass_risk_gate_in_permissive=True")
            elif self._gate_status.get('random_forest', False) and self.rf is not None:
                risk_gate_passed = (
                    rf_drawdown_pct <= self.config.max_drawdown_pct and
                    rf_streak_prob <= self.config.max_streak_prob
                )
                logger.debug(f"RF gate ACTIVE: drawdown={rf_drawdown_pct:.4f}, streak={rf_streak_prob:.2f}, passed={risk_gate_passed}")
            else:
                # RF unusable - use conservative fallback based on ATR
                # If ATR > 2%, be cautious
                if 'atr_pct_14' in df.columns:
                    atr_pct = df['atr_pct_14'].iloc[-1]
                    risk_gate_passed = atr_pct <= 0.02  # Max 2% ATR
                    logger.warning(f"⚠️  RF gate BYPASSED (permissive mode), using ATR fallback: atr={atr_pct:.4f}, passed={risk_gate_passed}")
                else:
                    risk_gate_passed = True  # No ATR available, trust Transformer
                    logger.warning(f"⚠️  RF gate BYPASSED (permissive mode): {self._gate_issues.get('random_forest', 'not loaded')}, no ATR fallback")
        else:
            # TCN confidence = how far from 0.5 (uncertain)
            # 0.5 -> 0%, 0.6 -> 20%, 0.7 -> 40%, 0.8 -> 60%, 0.9 -> 80%, 1.0 -> 100%
            tcn_confidence = abs(tcn_probability - 0.5) * 200
            
            # Strong TCN signal (>55% or <45% probability) can override weak gate models
            tcn_strong = abs(tcn_probability - 0.5) > 0.05  # >55% or <45%
            tcn_very_strong = abs(tcn_probability - 0.5) > 0.15  # >65% or <35%
            
            # CONFIDENCE GATE: Use TCN-derived confidence OR Ridge, whichever is higher
            effective_confidence = max(ridge_confidence, tcn_confidence)
            confidence_gate_passed = effective_confidence >= self.config.min_confidence
            
            # MOMENTUM GATE: Pass if any of these are true:
            # 1. XGBoost momentum is fresh (above threshold)
            # 2. XGBoost detects acceleration
            # 3. TCN is strong (override weak momentum readings)
            momentum_fresh = xgb_momentum >= self.config.min_momentum
            momentum_gate_passed = momentum_fresh or xgb_acceleration or tcn_strong
            
            # RISK GATE: Slightly more lenient when TCN is confident (but not too much)
            # Tightened multipliers to prevent excessive risk in high-drawdown scenarios
            if tcn_very_strong:
                # Very confident TCN - relax risk thresholds modestly
                risk_gate_passed = (
                    rf_drawdown_pct <= self.config.max_drawdown_pct * 1.3 and
                    rf_streak_prob <= self.config.max_streak_prob * 1.3
                )
            elif tcn_strong:
                # Strong TCN - relax risk thresholds slightly
                risk_gate_passed = (
                    rf_drawdown_pct <= self.config.max_drawdown_pct * 1.15 and
                    rf_streak_prob <= self.config.max_streak_prob * 1.15
                )
            else:
                # Weak TCN - use strict thresholds
                risk_gate_passed = (
                    rf_drawdown_pct <= self.config.max_drawdown_pct and
                    rf_streak_prob <= self.config.max_streak_prob
                )
        
        # === DETERMINE DIRECTION AND TRADE DECISION ===
        direction_str = None
        regime_gate_passed = True
        
        if self.use_regime:
            # REGIME-BASED DIRECTION LOGIC
            if regime == 'chop':
                # CHOP: Skip trading entirely
                regime_gate_passed = False
                direction_str = None
            elif regime == 'trend':
                # TREND: Direction from recent momentum sign
                # Use 2-bar return to determine trend direction
                if 'returns_2' in df.columns:
                    recent_return = df['returns_2'].iloc[-1]
                elif 'returns_1' in df.columns:
                    recent_return = df['returns_1'].iloc[-1]
                else:
                    recent_return = 0
                
                # Follow the trend
                if recent_return > 0:
                    direction_str = 'long'
                    tcn_direction = 1
                else:
                    direction_str = 'short'
                    tcn_direction = 0
                tcn_probability = regime_confidence
            elif regime == 'mean_revert':
                # MEAN REVERT: Fade 2-bar momentum
                if 'returns_2' in df.columns:
                    recent_return = df['returns_2'].iloc[-1]
                elif 'returns_1' in df.columns:
                    recent_return = df['returns_1'].iloc[-1]
                else:
                    recent_return = 0
                
                # Fade (opposite of recent move)
                if recent_return > 0:
                    direction_str = 'short'  # Fade the up move
                    tcn_direction = 0
                else:
                    direction_str = 'long'  # Fade the down move
                    tcn_direction = 1
                tcn_probability = regime_confidence
        
        # === TCN PROBABILITY GATE (applies to all modes) ===
        # Reject signals where TCN is too uncertain (near 50%)
        tcn_probability_gate_passed = abs(tcn_probability - 0.5) >= (self.config.min_tcn_probability - 0.5)
        
        # === SENTIMENT GATE (NEW) ===
        sentiment_gate_passed = True
        sentiment_reason = None
        if self.config.sentiment_block_enabled and intel_data:
            sent = intel_data.get('sentiment')
            if sent:
                score = float(sent.get('aggregate_score', 0.0))
                label = sent.get('aggregate_label', 'neutral')
                n_headlines = int(sent.get('num_headlines', 0))
                if n_headlines >= self.config.sentiment_min_headlines and abs(score) >= self.config.sentiment_block_threshold:
                    # Map sentiment to directional bias
                    sentiment_dir = 'long' if label == 'bullish' else 'short' if label == 'bearish' else None
                    if sentiment_dir and direction_str and sentiment_dir != direction_str:
                        sentiment_gate_passed = False
                        sentiment_reason = f"sentiment_{label}({score:+.2f})"

        # === DIRECTION FOR GATE CHECKS ===
        # In direction mode, direction_str is set late, so use tcn_direction for gate checks
        gate_check_direction = direction_str  # Already set if use_regime=True
        if gate_check_direction is None and tcn_direction is not None:
            gate_check_direction = 'long' if tcn_direction == 1 else 'short'

        # === DYNAMIC THRESHOLDS (LLM-powered, edge cases only) ===
        rsi_low_threshold = 10.0
        rsi_high_threshold = 90.0
        adx_trend_threshold = 35.0
        
        if self.enable_llm:
            try:
                from buddy_intelligent_mode import get_dynamic_thresholds
                
                rsi_val_check = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50.0
                adx_val_check = df['adx'].iloc[-1] if 'adx' in df.columns else 20.0
                
                market_context = {
                    'rsi': rsi_val_check,
                    'adx': adx_val_check,
                    'atr_pct': float(df['atr_pct_14'].iloc[-1]) if 'atr_pct_14' in df.columns else None,
                    'trend': 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down',
                    'news_summary': intel_data.get('sentiment', {}).get('aggregate_label', 'No news'),
                }
                
                thresholds = get_dynamic_thresholds(
                    instrument=instrument or "Unknown",
                    market_context=market_context,
                    only_for_edge_cases=True,  # Only call LLM for borderline cases
                )
                
                if thresholds.adjust_thresholds:
                    rsi_low_threshold = thresholds.rsi_extreme_low
                    rsi_high_threshold = thresholds.rsi_extreme_high
                    adx_trend_threshold = thresholds.adx_strong_trend
                    logger.info(f"🧠 LLM adjusted thresholds: RSI {rsi_low_threshold:.0f}/{rsi_high_threshold:.0f}, ADX {adx_trend_threshold:.0f} - {thresholds.reason}")
            except ImportError:
                pass  # LLM module not available
            except Exception as e:
                logger.debug(f"LLM threshold adjustment skipped: {e}")

        # === RSI EXTREME GATE ===
        # Block trades that fight extreme RSI conditions
        rsi_gate_passed = True
        rsi_reason = None
        rsi_val = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50.0
        
        # Extreme oversold: block LONG (don't catch falling knife)
        # Extreme overbought: block SHORT (don't short a rocket)
        if rsi_val < rsi_low_threshold and gate_check_direction == 'long':
            rsi_gate_passed = False
            rsi_reason = f"rsi_extreme_low({rsi_val:.1f})"
        elif rsi_val > rsi_high_threshold and gate_check_direction == 'short':
            rsi_gate_passed = False
            rsi_reason = f"rsi_extreme_high({rsi_val:.1f})"
        
        # === TREND CONTRADICTION GATE ===
        # Block trades against strong established trends
        trend_gate_passed = True
        trend_reason = None
        adx_val = df['adx'].iloc[-1] if 'adx' in df.columns else 0.0
        
        if adx_val > adx_trend_threshold:  # Strong trend (threshold may be dynamic)
            # Determine trend direction from price action
            price_trend = 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down'
            
            # Block counter-trend trades in strong trends
            if price_trend == 'down' and gate_check_direction == 'long':
                trend_gate_passed = False
                trend_reason = f"trend_contra(ADX={adx_val:.1f},trend=down)"
            elif price_trend == 'up' and gate_check_direction == 'short':
                trend_gate_passed = False
                trend_reason = f"trend_contra(ADX={adx_val:.1f},trend=up)"

        # === META-LABELING GATE (5th GATE - predicts trade SUCCESS) ===
        # Meta-labeler answers: "Given this signal, should we actually trade?"
        # This is DIFFERENT from direction - it predicts whether the trade will be PROFITABLE
        meta_gate_passed = True
        meta_confidence = 0.0
        meta_reason = None
        
        if self._meta_labeler_loaded and self.meta_labeler is not None:
            try:
                # Extract features for meta-labeler (uses same features as TCN + primary probability)
                meta_features = self._extract_tcn_features(df)
                
                # Ensure we have the primary model's probability
                primary_prob = np.array([[tcn_probability]], dtype=np.float32)
                
                # Handle 2D features (single row)
                if meta_features.ndim == 1:
                    meta_features = meta_features.reshape(1, -1)
                elif len(meta_features) > 1:
                    # Use only the last row
                    meta_features = meta_features[-1:, :]
                
                # Get meta-labeler confidence (probability that primary signal is correct)
                meta_conf_array = self.meta_labeler.predict_meta_confidence(
                    meta_features, 
                    primary_prob.flatten()
                )
                meta_confidence = float(meta_conf_array[0]) if len(meta_conf_array) > 0 else 0.0
                
                # Check gate threshold
                meta_gate_passed = meta_confidence >= self.config.min_meta_confidence
                
                if not meta_gate_passed:
                    meta_reason = f"low_meta_conf({meta_confidence:.2f}<{self.config.min_meta_confidence})"
                
                logger.debug(
                    f"🏷️ Meta-labeler: confidence={meta_confidence:.2f}, "
                    f"threshold={self.config.min_meta_confidence}, passed={meta_gate_passed}"
                )
            except Exception as e:
                logger.warning(f"Meta-labeler prediction failed: {e}")
                meta_gate_passed = True  # Fail open - don't block trades on error
                meta_confidence = 0.0
        else:
            # No meta-labeler loaded - pass by default
            meta_gate_passed = True
            meta_confidence = 0.0

        if self.use_regime:
            # All gates for regime mode
            all_gates_passed = (
                regime_gate_passed and
                direction_str is not None and
                tcn_probability_gate_passed and
                confidence_gate_passed and
                momentum_gate_passed and
                risk_gate_passed and
                sentiment_gate_passed and
                rsi_gate_passed and
                trend_gate_passed and
                meta_gate_passed  # 5th gate: meta-labeler
            )
        else:
            # DIRECTION MODE (legacy)
            all_gates_passed = (
                tcn_direction is not None and
                tcn_probability_gate_passed and
                confidence_gate_passed and
                momentum_gate_passed and
                risk_gate_passed and
                sentiment_gate_passed and
                rsi_gate_passed and
                trend_gate_passed and
                meta_gate_passed  # 5th gate: meta-labeler
            )
            if all_gates_passed and tcn_direction is not None:
                direction_str = 'long' if tcn_direction == 1 else 'short'
        
        # === BUILD REJECTION REASON ===
        reason = None
        if not all_gates_passed:
            reasons = []
            if not tcn_probability_gate_passed:
                reasons.append(f"weak_tcn({tcn_probability:.2f}<{self.config.min_tcn_probability})")
            if self.use_regime:
                if not regime_gate_passed:
                    reasons.append(f"regime=CHOP (skip)")
                if direction_str is None and regime != 'chop':
                    reasons.append("no_direction")
            else:
                if tcn_direction is None:
                    reasons.append("no_direction")
            if not confidence_gate_passed:
                reasons.append(f"low_confidence({ridge_confidence:.0f}<{self.config.min_confidence})")
            if not momentum_gate_passed:
                reasons.append(f"dead_momentum({xgb_momentum:.2f})")
            if not risk_gate_passed:
                if rf_drawdown_pct > self.config.max_drawdown_pct:
                    reasons.append(f"high_drawdown({rf_drawdown_pct:.2%})")
                if rf_streak_prob > self.config.max_streak_prob:
                    reasons.append(f"streak_risk({rf_streak_prob:.2f})")
            if not sentiment_gate_passed and sentiment_reason:
                reasons.append(sentiment_reason)
            if not rsi_gate_passed and rsi_reason:
                reasons.append(rsi_reason)
            if not trend_gate_passed and trend_reason:
                reasons.append(trend_reason)
            if not meta_gate_passed and meta_reason:
                reasons.append(meta_reason)
            reason = ", ".join(reasons)
        
        # === CALCULATE POSITION SIZE ===
        size = 0.0
        if all_gates_passed:
            # Prepare features for RL sizer (if enabled)
            # Use RF features as they contain market conditions
            rl_features = None
            if self.rl_sizer is not None:
                try:
                    rl_features = self._extract_rf_features(df)
                except Exception:
                    pass
            
            size = self._calculate_position_size(
                rf_drawdown_pips, 
                equity,
                instrument=getattr(self, '_current_instrument', None),
                features=rl_features,
                tcn_probability=tcn_probability,
                ridge_confidence=ridge_confidence,
            )
        
        # Store features for drift detection when trade result is recorded
        try:
            # Use TCN features as the primary feature set for drift tracking
            self._last_features = self._extract_tcn_features(df)
            if self._last_features.ndim > 1:
                self._last_features = self._last_features[-1]  # Keep last row only
        except Exception:
            self._last_features = None
        
        return TradeSignal(
            trade=all_gates_passed,
            direction=direction_str,
            size=size,
            confidence=ridge_confidence,
            regime=regime,
            regime_confidence=regime_confidence,
            tcn_direction=tcn_direction,
            tcn_probability=tcn_probability,
            ridge_confidence=ridge_confidence,
            xgb_momentum=xgb_momentum,
            xgb_acceleration=xgb_acceleration,
            rf_drawdown_pips=rf_drawdown_pips,
            rf_streak_prob=rf_streak_prob,
            histgb_direction=histgb_direction,
            histgb_probability=histgb_probability,
            models_agree=models_agree,
            confidence_gate_passed=confidence_gate_passed,
            momentum_gate_passed=momentum_gate_passed,
            risk_gate_passed=risk_gate_passed,
            regime_gate_passed=regime_gate_passed,
            meta_gate_passed=meta_gate_passed,
            meta_confidence=meta_confidence,
            reason=reason,
            metadata={'intel_data': intel_data},
        )
    
    def predict_verbose(
        self,
        df: pd.DataFrame,
        equity: Optional[float] = None,
        instrument: Optional[str] = None,
        headlines: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference with verbose output for logging/display.
        
        Returns dict with all details formatted for display.
        """
        signal = self.predict(df, equity, instrument=instrument, headlines=headlines)
        
        # Format gate checks
        gate_checks = []
        
        # === MARKET INTELLIGENCE STATUS (NEW) ===
        intel_data = signal.metadata.get('intel_data', {}) if signal.metadata else {}
        
        # Calendar check
        if 'calendar_error' in intel_data:
            gate_checks.append(f"📅 Calendar: feed error ({intel_data['calendar_error']})")
        elif 'next_high_impact' in intel_data:
            event = intel_data['next_high_impact']
            mins = int(event.get('minutes_until', 999))
            event_name = event.get('name', 'Unknown')
            gate_checks.append(f"📅 Calendar: {event_name} in {mins}m ✓")
        elif self.market_intel:
            total = intel_data.get('calendar_events')
            high = intel_data.get('calendar_high_impact')
            if total is not None and high is not None:
                gate_checks.append(f"📅 Calendar: {total} events ({high} high-impact) ✓")
            else:
                gate_checks.append("📅 Calendar: No events ✓")
        
        # Sentiment check
        if 'sentiment' in intel_data:
            sent = intel_data['sentiment']
            label = sent.get('aggregate_label', 'neutral')
            score = sent.get('aggregate_score', 0.0)
            n_headlines = sent.get('num_headlines', 0)
            if n_headlines > 0:
                gate_checks.append(f"📰 Sentiment: {label} ({score:+.2f}) [{n_headlines} headlines] ✓")
            else:
                gate_checks.append("📰 Sentiment: No headlines (skipped)")
        elif self.market_intel and self.market_intel.sentiment is not None:
            headlines_count = intel_data.get('headlines_count')
            if headlines_count == 0:
                gate_checks.append("📰 Sentiment: No headlines found (RSS)")
            else:
                gate_checks.append("📰 Sentiment: Ready (no headlines provided)")
        
        # Online learning status
        if self.market_intel and self.market_intel.online_learner:
            buffer_size = len(self.market_intel.online_learner.trade_buffer)
            gate_checks.append(f"🔄 Online Learning: {buffer_size}/50 trades buffered")
        
        # Calibration status (NEW)
        calib_info = intel_data.get('calibration', {})
        if calib_info.get('applied'):
            raw_prob = calib_info.get('raw_probability', 0)
            cal_prob = calib_info.get('calibrated_probability', 0)
            method = calib_info.get('method', 'unknown')
            gate_checks.append(f"📐 Calibration: {method} ({raw_prob:.3f} → {cal_prob:.3f}) ✓")
        elif self.config.enable_calibration:
            gate_checks.append("📐 Calibration: Not fitted (using raw)")
        
        gate_checks.append("")  # Spacer
        
        # TCN direction
        if signal.tcn_direction is not None:
            dir_str = "LONG" if signal.tcn_direction == 1 else "SHORT"
            # Show both raw and calibrated if calibration was applied
            calib_info = intel_data.get('calibration', {})
            if calib_info.get('applied'):
                raw_prob = calib_info.get('raw_probability', signal.tcn_probability)
                gate_checks.append(f"TCN: {dir_str} (raw={raw_prob:.2f}, calibrated={signal.tcn_probability:.2f})")
            else:
                gate_checks.append(f"TCN: {dir_str} (prob={signal.tcn_probability:.2f})")
        else:
            gate_checks.append("TCN: NO SIGNAL")
        
        # Ridge confidence
        status = "✓" if signal.confidence_gate_passed else "✗"
        gate_checks.append(f"Ridge: {signal.ridge_confidence:.0f}/100 {status}")
        
        # XGBoost momentum
        status = "✓" if signal.momentum_gate_passed else "✗"
        accel_str = "accel=true" if signal.xgb_acceleration else "accel=false"
        gate_checks.append(f"XGBoost: momentum={signal.xgb_momentum:.2f}, {accel_str} {status}")
        
        # RF risk
        status = "✓" if signal.risk_gate_passed else "✗"
        gate_checks.append(f"RF: drawdown={signal.rf_drawdown_pips:.1f}pips, streak={signal.rf_streak_prob:.2f} {status}")
        
        # Meta-labeling gate (5th gate - predicts trade SUCCESS)
        if self._meta_labeler_loaded and self.meta_labeler is not None:
            status = "✓" if signal.meta_gate_passed else "✗"
            gate_checks.append(f"Meta: confidence={signal.meta_confidence:.2f} (threshold={self.config.min_meta_confidence}) {status}")
        elif self.config.enable_meta_labeling:
            gate_checks.append("Meta: not loaded (skipped)")
        
        # Final decision
        if signal.trade:
            decision = f"→ TRADE: {signal.direction.upper()}, size={signal.size} lots"
        else:
            decision = f"→ NO TRADE: {signal.reason}"
        
        return {
            'trade': signal.trade,
            'direction': signal.direction,
            'size': signal.size,
            'gate_checks': gate_checks,
            'decision': decision,
            'raw_signal': signal,
            'intel_data': intel_data,
        }


def train_calibration(
    model_dir: str = "trained_data/models",
    validation_predictions: Optional[np.ndarray] = None,
    validation_outcomes: Optional[np.ndarray] = None,
    method: str = 'platt',
    instrument: Optional[str] = None,
) -> Optional['ConfidenceCalibrator']:
    """
    Train and save confidence calibration from validation data.
    
    This function should be called after model training to fit the calibrator
    on validation set predictions and actual outcomes.
    
    Args:
        model_dir: Directory where models are stored
        validation_predictions: Raw model probabilities on validation set (0-1)
        validation_outcomes: Actual outcomes (0=loss, 1=win)
        method: Calibration method ('platt', 'isotonic', or 'both')
        instrument: Optional instrument for pair-specific calibration
        
    Returns:
        Fitted ConfidenceCalibrator or None if calibration not available
        
    Example:
        # After training, get validation predictions:
        val_preds = model.predict(X_val)[:, 1]  # Probability of positive class
        val_outcomes = (y_val == 1).astype(int)  # Actual outcomes
        
        # Train calibration:
        calibrator = train_calibration(
            model_dir="trained_data/models",
            validation_predictions=val_preds,
            validation_outcomes=val_outcomes,
            method='platt',
        )
    """
    if not CALIBRATION_AVAILABLE:
        logger.warning("Confidence calibration module not available")
        return None
    
    if validation_predictions is None or validation_outcomes is None:
        logger.warning("No validation data provided for calibration training")
        return None
    
    if len(validation_predictions) < 20:
        logger.warning(f"Insufficient validation samples ({len(validation_predictions)}) for calibration. Need at least 20.")
        return None
    
    model_dir = Path(model_dir)
    
    # Create calibrator
    config = CalibrationConfig(
        method=method,
        min_confidence_threshold=0.5,
        max_confidence_threshold=0.95,
        apply_directional_adjustment=False,  # Pure calibration only
        apply_win_probability_adjustment=False,
        apply_trading_context_adjustment=False,
    )
    calibrator = ConfidenceCalibrator(config)
    
    # Fit calibration
    try:
        calibrator.fit(validation_predictions, validation_outcomes)
        logger.info(f"✓ Calibration fitted using {len(validation_predictions)} samples ({method} method)")
        
        # Evaluate calibration quality
        eval_metrics = calibrator.evaluate_calibration(validation_predictions, validation_outcomes)
        logger.info(f"  Brier score: {eval_metrics['original_brier_score']:.4f} → {eval_metrics['calibrated_brier_score']:.4f}")
        logger.info(f"  Improvement: {eval_metrics['relative_improvement']*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Calibration fitting failed: {e}")
        return None
    
    # Save calibrator
    if instrument and instrument != "GENERIC":
        save_path = model_dir / instrument / "confidence_calibrator.pkl"
    else:
        save_path = model_dir / "confidence_calibrator.pkl"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(save_path)
    logger.info(f"✓ Calibrator saved to {save_path}")
    
    # Also save calibration parameters to ensemble metadata for portability
    meta_path = model_dir / "modular_ensemble.meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Add calibration data
            calib_meta = {
                'method': method,
                'min_threshold': 0.5,
                'max_threshold': 0.95,
                'n_samples': len(validation_predictions),
                'brier_original': float(eval_metrics['original_brier_score']),
                'brier_calibrated': float(eval_metrics['calibrated_brier_score']),
            }
            
            # Save Platt parameters
            if calibrator.platt_model is not None:
                calib_meta['platt_params'] = {
                    'coef': float(calibrator.platt_model.coef_[0][0]),
                    'intercept': float(calibrator.platt_model.intercept_[0]),
                }
            
            # Save Isotonic parameters (if used)
            if calibrator.isotonic_model is not None:
                calib_meta['isotonic_params'] = {
                    'X_thresholds': calibrator.isotonic_model.X_thresholds_.tolist(),
                    'y_thresholds': calibrator.isotonic_model.y_thresholds_.tolist(),
                }
            
            meta['calibration'] = calib_meta
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            logger.info(f"✓ Calibration metadata saved to {meta_path}")
            
        except Exception as e:
            logger.warning(f"Failed to update ensemble metadata with calibration: {e}")
    
    return calibrator


def run_inference_test():
    """Quick test of inference pipeline."""
    import pandas as pd
    import numpy as np
    
    # Create dummy data
    n = 100
    df = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n) * 0.01) + 150,
        'high': np.cumsum(np.random.randn(n) * 0.01) + 150.1,
        'low': np.cumsum(np.random.randn(n) * 0.01) + 149.9,
        'volume': np.random.randint(1000, 10000, n),
        'returns': np.random.randn(n) * 0.01,
        'volatility_5': np.abs(np.random.randn(n)) * 0.01,
        'volatility_10': np.abs(np.random.randn(n)) * 0.01,
        'volatility_20': np.abs(np.random.randn(n)) * 0.01,
        'atr': np.abs(np.random.randn(n)) * 0.5,
        'rsi': np.random.uniform(30, 70, n),
        'momentum_10': np.random.randn(n) * 0.01,
        'macd': np.random.randn(n) * 0.001,
        'macd_hist': np.random.randn(n) * 0.001,
        'obv': np.cumsum(np.random.randint(-1000, 1000, n)),
        'mfi': np.random.uniform(20, 80, n),
        'adx': np.random.uniform(15, 40, n),
    })
    
    # Test inference
    ensemble = ModularEnsembleInference()
    
    # Check if models exist (check for both Transformer and TCN)
    model_dir = Path("trained_data/models")
    if not (model_dir / "transformer_direction.keras").exists() and not (model_dir / "tcn_direction.keras").exists():
        print("Models not found. Train first with: buddy train --model-type ensemble")
        return
    
    result = ensemble.predict_verbose(df)
    
    print("\n" + "="*60)
    print("MODULAR ENSEMBLE INFERENCE TEST")
    print("="*60)
    for check in result['gate_checks']:
        print(check)
    print("-"*60)
    print(result['decision'])
    print("="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference_test()

