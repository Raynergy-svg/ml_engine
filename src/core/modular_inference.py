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

from datetime import datetime, timezone
import json
import logging
import pickle
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import INFERENCE_DEFAULTS

# Import normalized feature computation from data loaders
from .modular_data_loaders import (
    compute_normalized_features,
    get_normalized_feature_names,
    align_features_to_model,
)

# [2026-02-14] Feature alignment fix: Import feature index loader for inference
from src.training.feature_alignment import load_feature_indices as load_saved_feature_indices

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
    from src.training.meta_labeling import MetaLabeler, MetaLabelingConfig
    META_LABELING_AVAILABLE = True
except ImportError:
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

# Import learned confidence model (Phase 3: Scanner Accuracy)
try:
    from src.scanner.learned_confidence import LearnedConfidenceModel
    LEARNED_CONFIDENCE_AVAILABLE = True
except ImportError:
    LEARNED_CONFIDENCE_AVAILABLE = False
    LearnedConfidenceModel = None

# Import custom Keras serializable classes to ensure proper deserialization
# This MUST be imported before loading any models that use these schedulers
try:
    from src.training.m1_metal_optimizer import WarmupCosineDecaySchedule, CosineDecayRestarts  # noqa: F401
    CUSTOM_LR_SCHEDULES_AVAILABLE = True
except ImportError:
    CUSTOM_LR_SCHEDULES_AVAILABLE = False
    # Models may still load if compile=False, but recompilation will be needed

# Import Pandera for optional DataFrame schema validation
# Provides clear error messages when features are missing/mistyped
try:
    import pandera as pa
    import pandera.pandas as pa_pandas
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    pa = None
    pa_pandas = None

# RL position sizer availability is checked lazily to avoid TF/PyTorch GPU conflicts
# The actual import happens only when use_rl_sizer=True and the model is loaded
# NOTE: stable_baselines3 import can take 9-10+ seconds due to PyTorch dependencies
RL_AVAILABLE = True  # We assume it's available, actual check is deferred
RLPositionSizer = None  # Lazy loaded
RL_MODEL_PATH = Path("trained_data/models/rl_position_sizer.zip")
RL_ONNX_PATH = Path("trained_data/models/rl_position_sizer.onnx")

# RL Gate threshold and exit timing (NEW)
GateThresholdRL = None  # Lazy loaded
OptimalExitRL = None  # Lazy loaded
RL_GATES_AVAILABLE = True
RL_EXITS_AVAILABLE = True
RL_GATE_MODEL_PATH = Path("trained_data/models/sac_gate_thresholds.zip")
RL_EXIT_MODEL_PATH = Path("trained_data/models/ppo_optimal_exit.zip")

# Timeout for RL loading to prevent hangs
RL_LOAD_TIMEOUT_SECONDS = 10
RL_SUBPROCESS_LOAD_TIMEOUT_SECONDS = 20


def _env_flag_true(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _allow_rl_gates_exits_autoload() -> bool:
    """Whether RL gates/exits should be auto-enabled when model files exist.

    You can force-enable/disable via env vars.
    """
    if _env_flag_true("BUDDY_ENABLE_RL_GATES_EXITS"):
        return True

    if _env_flag_true("BUDDY_DISABLE_RL_GATES_EXITS"):
        return False
    return True


def _rl_gates_worker_main(request_queue, response_queue) -> None:
    """Child process: load RL gates model and serve threshold requests."""
    try:
        from src.rl.gate_threshold_env import GateThresholdRL

        optimizer = GateThresholdRL()
        loaded = optimizer.load()
        response_queue.put({'type': 'status', 'ok': bool(loaded)})
        if not loaded:
            return

        while True:
            msg = request_queue.get()
            if not isinstance(msg, dict):
                continue
            if msg.get('type') == 'shutdown':
                return
            if msg.get('type') != 'thresholds':
                continue

            req_id = msg.get('id')
            features = msg.get('features') or {}
            win_rate = float(msg.get('win_rate', 0.5))
            drawdown = float(msg.get('drawdown', 0.0))
            adjusted = optimizer.get_adjusted_thresholds(
                features=features,
                win_rate=win_rate,
                drawdown=drawdown,
            )
            response_queue.put({'type': 'thresholds', 'id': req_id, 'ok': True, 'result': adjusted})
    except BaseException as e:  # noqa: BLE001
        try:
            response_queue.put({'type': 'status', 'ok': False, 'error': str(e)})
        except Exception:
            pass


def _rl_exits_worker_main(request_queue, response_queue) -> None:
    """Child process: load RL exits model and serve exit-decision requests."""
    try:
        from src.rl.optimal_exit_env import OptimalExitRL

        optimizer = OptimalExitRL()
        loaded = optimizer.load()
        response_queue.put({'type': 'status', 'ok': bool(loaded)})
        if not loaded:
            return

        while True:
            msg = request_queue.get()
            if not isinstance(msg, dict):
                continue
            if msg.get('type') == 'shutdown':
                return
            if msg.get('type') != 'exit_decision':
                continue

            req_id = msg.get('id')
            action, confidence = optimizer.get_exit_decision(
                unrealized_pnl_pips=float(msg.get('unrealized_pnl_pips', 0.0)),
                bars_in_trade=int(msg.get('bars_in_trade', 0)),
                momentum=float(msg.get('momentum', 0.0)),
                atr=float(msg.get('atr', 0.001)),
            )
            response_queue.put(
                {
                    'type': 'exit_decision',
                    'id': req_id,
                    'ok': True,
                    'result': {'action': int(action), 'confidence': float(confidence)},
                }
            )
    except BaseException as e:  # noqa: BLE001
        try:
            response_queue.put({'type': 'status', 'ok': False, 'error': str(e)})
        except Exception:
            pass


def _call_with_timeout(fn, timeout_s: float):
    """Run `fn` in a daemon thread and wait up to `timeout_s` seconds.

    IMPORTANT: This intentionally does NOT attempt to stop the worker thread on timeout.
    It is designed to avoid deadlocking the main thread when TF/PyTorch imports hang.

    Returns:
        (timed_out, result)

    Raises:
        Re-raises any exception raised by `fn` if it completes within the timeout.
    """
    import queue
    import threading

    result_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)

    def _runner():
        try:
            result_queue.put((True, fn(), None))
        except Exception as e:
            result_queue.put((True, None, e))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        return True, None

    _, result, err = result_queue.get_nowait()
    if err is not None:
        raise err
    return False, result


def _lazy_load_rl_sizer_unsafe():
    """Internal: Load RLPositionSizer (may hang on GPU conflicts)."""
    from src.training.rl.position_sizer import RLPositionSizer as _RLPositionSizer
    return _RLPositionSizer, True


def _lazy_load_rl_sizer():
    """Lazy load RLPositionSizer with timeout protection to avoid TF/PyTorch GPU conflicts."""
    global RLPositionSizer, RL_AVAILABLE
    if RLPositionSizer is not None:
        return RLPositionSizer, RL_AVAILABLE

    try:
        timed_out, result = _call_with_timeout(_lazy_load_rl_sizer_unsafe, RL_LOAD_TIMEOUT_SECONDS)
        if timed_out:
            logger.warning(
                f"RL sizer import timed out after {RL_LOAD_TIMEOUT_SECONDS}s - using heuristic sizing"
            )
            RL_AVAILABLE = False
            RLPositionSizer = None
            return None, False

        rl_position_sizer_cls, _ = result
        RLPositionSizer = rl_position_sizer_cls
        RL_AVAILABLE = True
        return RLPositionSizer, RL_AVAILABLE
    except ImportError:
        RL_AVAILABLE = False
        return None, False
    except Exception as e:
        logger.warning(f"RL sizer import failed: {type(e).__name__}: {e}")
        RL_AVAILABLE = False
        RLPositionSizer = None
        return None, False


def _lazy_load_rl_gates_unsafe():
    """Internal: Load GateThresholdRL (may hang on GPU conflicts)."""
    from src.rl.gate_threshold_env import GateThresholdRL as _GateThresholdRL
    return _GateThresholdRL, True


def _lazy_load_rl_gates():
    """Lazy load GateThresholdRL with timeout protection to avoid TF/PyTorch GPU conflicts."""
    global GateThresholdRL, RL_GATES_AVAILABLE
    if GateThresholdRL is not None:
        return GateThresholdRL, RL_GATES_AVAILABLE

    try:
        timed_out, result = _call_with_timeout(_lazy_load_rl_gates_unsafe, RL_LOAD_TIMEOUT_SECONDS)
        if timed_out:
            logger.warning(
                f"RL gates import timed out after {RL_LOAD_TIMEOUT_SECONDS}s - using fixed thresholds"
            )
            RL_GATES_AVAILABLE = False
            GateThresholdRL = None
            return None, False

        gate_threshold_rl_cls, _ = result
        GateThresholdRL = gate_threshold_rl_cls
        RL_GATES_AVAILABLE = True
        return GateThresholdRL, RL_GATES_AVAILABLE
    except ImportError:
        RL_GATES_AVAILABLE = False
        GateThresholdRL = None
        return None, False
    except Exception as e:
        logger.warning(f"RL gates import failed: {type(e).__name__}: {e}")
        RL_GATES_AVAILABLE = False
        GateThresholdRL = None
        return None, False


def _lazy_load_rl_exits_unsafe():
    """Internal: Load OptimalExitRL (may hang on GPU conflicts)."""
    from src.rl.optimal_exit_env import OptimalExitRL as _OptimalExitRL
    return _OptimalExitRL, True


def _lazy_load_rl_exits():
    """Lazy load OptimalExitRL with timeout protection to avoid TF/PyTorch GPU conflicts."""
    global OptimalExitRL, RL_EXITS_AVAILABLE
    if OptimalExitRL is not None:
        return OptimalExitRL, RL_EXITS_AVAILABLE

    try:
        timed_out, result = _call_with_timeout(_lazy_load_rl_exits_unsafe, RL_LOAD_TIMEOUT_SECONDS)
        if timed_out:
            logger.warning(
                f"RL exits import timed out after {RL_LOAD_TIMEOUT_SECONDS}s - using fixed R:R ratios"
            )
            RL_EXITS_AVAILABLE = False
            OptimalExitRL = None
            return None, False

        optimal_exit_rl_cls, _ = result
        OptimalExitRL = optimal_exit_rl_cls
        RL_EXITS_AVAILABLE = True
        return OptimalExitRL, RL_EXITS_AVAILABLE
    except ImportError:
        RL_EXITS_AVAILABLE = False
        OptimalExitRL = None
        return None, False
    except Exception as e:
        logger.warning(f"RL exits import failed: {type(e).__name__}: {e}")
        RL_EXITS_AVAILABLE = False
        OptimalExitRL = None
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
    # === TCN VOLATILITY REGIME FILTER ===
    # TCN predicts volatility regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)
    # Used as informational entry timing filter - volatility is logged but does not
    # short-circuit the remaining gate computations.  When the regime is below
    # min_volatility_regime the trade is still blocked at the aggregation step,
    # but all gate values are computed and reported for full diagnostics.
    use_tcn_volatility_filter: bool = True  # Use TCN as volatility gate (not direction ensemble)
    min_volatility_regime: int = 0  # Minimum regime: 0=LOW (permissive default)
    tcn_required: bool = False  # Allow graceful degradation if TCN unavailable

    # Confidence gate - ADX-based, 50+ is strong trend
    min_confidence: float = 50.0  # 0-100 scale (tightened from 45)

    # Direction-confidence gate (Transformer/primary direction model)
    # 0.60 means require >=60% (or <=40%) to consider the direction actionable.
    min_tcn_probability: float = INFERENCE_DEFAULTS['min_tcn_probability']

    # Momentum gate - median momentum is 0.3, so 0.20 catches bottom 40%
    min_momentum: float = 0.20  # 0-1 scale (tightened from 0.15)
    require_fresh_or_accel: bool = True

    # Risk gate - ATR-based drawdown (2x ATR, typically 0.5-3%)
    max_drawdown_pct: float = 0.025  # 2.5% max expected drawdown
    max_streak_prob: float = 0.95  # 95% max streak continuation (increased from 0.6 for more permissive trading)

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

    # Transformer weight for direction (TCN no longer in ensemble, Transformer-only)
    transformer_weight: float = 1.0  # Transformer-only for direction prediction

    # Confidence calibration
    enable_calibration: bool = True  # Apply Platt/Isotonic calibration to raw probabilities
    calibration_method: str = 'platt'  # 'platt', 'isotonic', or 'none'

    # Meta-labeling gate - predicts whether primary model's signal will succeed
    enable_meta_labeling: bool = True  # Use meta-labeler to filter trades
    min_meta_confidence: float = 0.55  # Minimum meta-confidence to allow trade

    # Monte Carlo Dropout for uncertainty estimation
    mc_dropout_samples: int = 20  # Number of forward passes for MC Dropout (0 to disable)
    max_uncertainty_std: float = 0.15  # Block trades if prediction std > this threshold

    # Volatility filter - skip low-volatility conditions
    min_atr_pips: float = 8.0  # Minimum ATR in pips to allow trading

    # Heuristic directional veto gates (additional to core model gates).
    # Keep enabled by default for safety; aggressive profile can disable.
    use_rsi_extreme_gate: bool = True
    use_trend_contra_gate: bool = True
    rsi_extreme_low_threshold: float = 10.0
    rsi_extreme_high_threshold: float = 90.0
    adx_trend_contra_threshold: float = 35.0

    # Position sizing - RISK-BASED (5% aggressive mode)
    risk_per_trade_pct: float = 0.05  # 5% risk per trade (~$5k on $100k)
    account_equity: float = 103000.0  # User's account balance
    pip_value: float = 10.0  # ~$10 per pip per standard lot
    min_stop_loss_pips: float = 10.0  # Clamp stop distance to avoid pathological oversizing
    min_position_lots: float = 0.01  # Broker floor for meaningful FX size (1k units)
    position_sizing_mode: str = "rl_hybrid"  # "rl_hybrid", "rl_only", "risk_only"
    rl_min_risk_utilization: float = 0.08  # Minimum fraction of configured risk budget to deploy with RL sizing

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

    # Regime results (market regime from Transformer)
    regime: Optional[str] = None  # 'trend', 'chop', 'mean_revert', or None
    regime_confidence: float = 0.0

    # Volatility regime (TCN filter - NEW)
    volatility_regime: Optional[int] = None  # 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
    volatility_regime_name: Optional[str] = None  # Human-readable name
    volatility_regime_confidence: float = 0.0  # Softmax confidence
    volatility_gate_passed: bool = False  # Did it pass the minimum regime threshold?

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

    # RL-optimized parameters (NEW)
    rl_thresholds_used: bool = False  # True if RL gate thresholds were applied
    rl_exit_suggestion: Optional[str] = None  # 'hold', 'exit_profit', 'exit_loss', or None
    rl_exit_confidence: float = 0.0  # Confidence in exit suggestion

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
        use_rl_sizer: Optional[bool] = None,  # None = auto-detect from model existence
        use_rl_gates: Optional[bool] = None,  # None = auto-detect from model existence
        use_rl_exits: Optional[bool] = None,  # None = auto-detect from model existence
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

        # Model load report (for CLI visibility)
        self._model_load_report: List[Dict[str, Any]] = []

        # LLM Integration (NEW)
        self.enable_llm = enable_llm_integration
        self._llm_sentiment_cache = {}  # Cache LLM sentiment by headlines hash

        # Drift detection config
        self.enable_drift_detection = enable_drift_detection
        self._drift_config_overrides = drift_config or {}

        # RL-based enhancements (NEW)
        self.use_rl_gates = use_rl_gates
        self.use_rl_exits = use_rl_exits
        self.rl_gate_optimizer: Optional['GateThresholdRL'] = None
        self.rl_exit_optimizer: Optional['OptimalExitRL'] = None
        self._rl_gates_loaded = False
        self._rl_exits_loaded = False

        # On macOS, load RL gates/exits in a subprocess to avoid TF+PyTorch segfaults.
        self._rl_gates_proc = None
        self._rl_gates_req_q = None
        self._rl_gates_res_q = None
        self._rl_exits_proc = None
        self._rl_exits_req_q = None
        self._rl_exits_res_q = None
        self._rl_req_counter = 0

        # Track recent performance for RL gate adaptation
        self._recent_trades: List[Dict] = []
        self._recent_win_rate: float = 0.5
        self._current_drawdown: float = 0.0

        # Last sizing details (for transparency in output)
        self._last_size_details: Dict[str, Any] = {}

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
                logger.debug(f"Market Intelligence enabled ({features_str})")
            except Exception as e:
                logger.warning(f"Market Intelligence initialization failed: {e}")
                self.market_intel = None
        elif enable_market_intelligence and not MARKET_INTEL_AVAILABLE:
            logger.debug("Market Intelligence not available (install transformers for sentiment)")

        # RL Position Sizer (NEW)
        self.rl_sizer: Optional[RLPositionSizer] = None
        self.use_rl_sizer = use_rl_sizer

        # Confidence Calibrator (NEW)
        self.calibrator: Optional['ConfidenceCalibrator'] = None
        self._calibration_loaded = False

        # Learned Confidence Model (Phase 3: Scanner Accuracy)
        self.learned_confidence: Optional['LearnedConfidenceModel'] = None
        self._learned_confidence_loaded = False

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

        # Thread safety for model loading
        self._load_lock = threading.Lock()

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
                logger.debug("OnlineRetrainer initialized")
            except Exception as e:
                logger.warning(f"OnlineRetrainer initialization failed: {e}")

        # === REGIME-SPECIFIC LGBM MODELS (Phase 3) ===
        # 5 separate LightGBM models, one per market regime
        self._regime_lgbm_trainer = None  # RegimeLGBMTrainer instance
        self._regime_lgbm_loaded = False

        # Regime tracking for transition detection
        self._regime_history: List[str] = []  # Last N regimes for transition detection
        self._regime_history_max_size: int = 10  # Keep last 10 regimes
        self._current_regime: Optional[str] = None
        self._regime_transition_cooldown: int = 0  # Bars since last transition
        self._regime_transition_cooldown_target: int = 3  # 3-bar cooldown after transition

    def _next_rl_req_id(self) -> int:
        self._rl_req_counter += 1
        return self._rl_req_counter

    def _start_rl_gates_worker(self) -> Tuple[bool, Optional[str]]:
        """Start RL gates worker subprocess. Returns (ok, error_msg)."""
        try:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            self._rl_gates_req_q = ctx.Queue(maxsize=8)
            self._rl_gates_res_q = ctx.Queue(maxsize=8)
            self._rl_gates_proc = ctx.Process(
                target=__import__('src.rl.rl_worker', fromlist=['rl_gates_worker_main']).rl_gates_worker_main,
                args=(self._rl_gates_req_q, self._rl_gates_res_q),
                daemon=True,
            )
            self._rl_gates_proc.start()

            start = time.time()
            while time.time() - start < RL_SUBPROCESS_LOAD_TIMEOUT_SECONDS:
                try:
                    msg = self._rl_gates_res_q.get(timeout=0.5)
                except Exception:
                    if self._rl_gates_proc is not None and not self._rl_gates_proc.is_alive():
                        break
                    continue
                if isinstance(msg, dict) and msg.get('type') == 'status':
                    ok = bool(msg.get('ok'))
                    err = msg.get('error')
                    return ok, err
            return False, 'Timed out waiting for worker status'
        except Exception as e:
            logger.warning(f"RL gates worker start failed: {e}")
            return False, str(e)

    def _start_rl_exits_worker(self) -> Tuple[bool, Optional[str]]:
        """Start RL exits worker subprocess. Returns (ok, error_msg)."""
        try:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            self._rl_exits_req_q = ctx.Queue(maxsize=8)
            self._rl_exits_res_q = ctx.Queue(maxsize=8)
            self._rl_exits_proc = ctx.Process(
                target=__import__('src.rl.rl_worker', fromlist=['rl_exits_worker_main']).rl_exits_worker_main,
                args=(self._rl_exits_req_q, self._rl_exits_res_q),
                daemon=True,
            )
            self._rl_exits_proc.start()

            start = time.time()
            while time.time() - start < RL_SUBPROCESS_LOAD_TIMEOUT_SECONDS:
                try:
                    msg = self._rl_exits_res_q.get(timeout=0.5)
                except Exception:
                    if self._rl_exits_proc is not None and not self._rl_exits_proc.is_alive():
                        break
                    continue
                if isinstance(msg, dict) and msg.get('type') == 'status':
                    ok = bool(msg.get('ok'))
                    err = msg.get('error')
                    return ok, err
            return False, 'Timed out waiting for worker status'
        except Exception as e:
            logger.warning(f"RL exits worker start failed: {e}")
            return False, str(e)

    def _get_rl_gate_thresholds_from_worker(self, features: Dict[str, Any], win_rate: float, drawdown: float) -> Optional[Dict[str, Any]]:
        if self._rl_gates_proc is None or not self._rl_gates_proc.is_alive():
            return None
        if self._rl_gates_req_q is None or self._rl_gates_res_q is None:
            return None

        req_id = self._next_rl_req_id()
        try:
            self._rl_gates_req_q.put(
                {'type': 'thresholds', 'id': req_id, 'features': features, 'win_rate': win_rate, 'drawdown': drawdown},
                timeout=0.5,
            )
        except Exception:
            return None

        start = time.time()
        while time.time() - start < 1.5:
            try:
                msg = self._rl_gates_res_q.get(timeout=0.5)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get('type') == 'thresholds' and msg.get('id') == req_id and msg.get('ok'):
                return msg.get('result')
        return None

    def _get_rl_exit_decision_from_worker(self, unrealized_pnl_pips: float, bars_in_trade: int, momentum: float, atr: float) -> Optional[Tuple[int, float]]:
        if self._rl_exits_proc is None or not self._rl_exits_proc.is_alive():
            return None
        if self._rl_exits_req_q is None or self._rl_exits_res_q is None:
            return None

        req_id = self._next_rl_req_id()
        try:
            self._rl_exits_req_q.put(
                {
                    'type': 'exit_decision',
                    'id': req_id,
                    'unrealized_pnl_pips': unrealized_pnl_pips,
                    'bars_in_trade': bars_in_trade,
                    'momentum': momentum,
                    'atr': atr,
                },
                timeout=0.5,
            )
        except Exception:
            return None

        start = time.time()
        while time.time() - start < 1.5:
            try:
                msg = self._rl_exits_res_q.get(timeout=0.5)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get('type') == 'exit_decision' and msg.get('id') == req_id and msg.get('ok'):
                result = msg.get('result') or {}
                return int(result.get('action', 0)), float(result.get('confidence', 0.5))
        return None

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
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
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
        Get the path to a model file, auto-preferring pair-specific then joint models.

        Lookup order:
        1. trained_data/models/{instrument}/{model_name}{extension}  (pair-specific)
        2. trained_data/models/joint/{model_name}{extension}  (joint training)
        3. trained_data/models/{model_name}{extension}  (generic fallback)

        Returns the first existing path, or the generic path if none exist.
        Logs which model source is being used.
        """
        # 1. Check pair-specific path first
        if self.instrument and self.instrument != "GENERIC":
            pair_path = self.model_dir / self.instrument / f"{model_name}{extension}"
            if pair_path.exists():
                logger.debug(f"Using pair-specific model: {pair_path}")
                return pair_path

        # 2. Check joint training path
        joint_path = self.model_dir / "joint" / f"{model_name}{extension}"
        if joint_path.exists():
            logger.debug(f"Using joint-trained model: {joint_path}")
            return joint_path

        # 3. Fallback to generic path
        generic_path = self.model_dir / f"{model_name}{extension}"
        logger.debug(f"Using generic model: {generic_path}")
        return generic_path

    def _get_pair_metadata_model_path(
        self,
        model_key: str,
        expected_suffix: str = ".pkl",
    ) -> Optional[Path]:
        """
        Resolve preferred model path from pair-specific modular metadata.

        This prevents inference from accidentally picking joint/generic artifacts
        when pair training just produced explicit model files.
        """
        if not self.instrument or self.instrument == "GENERIC":
            return None

        pair_meta = self.model_dir / self.instrument / "modular_ensemble.meta.json"
        if not pair_meta.exists():
            return None

        try:
            with open(pair_meta) as f:
                meta = json.load(f)
            models = meta.get("models", {})
            entry = models.get(model_key, {})
            path_str = entry.get("path") if isinstance(entry, dict) else None
            if not isinstance(path_str, str) or not path_str.strip():
                return None

            raw_path = Path(path_str)
            candidates: list[Path] = []
            if raw_path.is_absolute():
                candidates.append(raw_path)
            else:
                # Most metadata stores workspace-relative paths.
                candidates.append(raw_path)
                # Defensive fallback: same basename in pair model dir.
                candidates.append(self.model_dir / self.instrument / raw_path.name)

            resolved = next((p for p in candidates if p.exists()), None)
            if resolved is None:
                return None

            # Ensure pair-specific metadata cannot point to another instrument artifact.
            if self.instrument not in resolved.parts:
                logger.warning(
                    "Ignoring pair metadata path for %s: %s (instrument mismatch: %s)",
                    model_key,
                    resolved,
                    self.instrument,
                )
                return None

            if expected_suffix and resolved.suffix.lower() != expected_suffix.lower():
                logger.warning(
                    "Pair metadata path for %s has unexpected suffix: %s (expected %s)",
                    model_key,
                    resolved,
                    expected_suffix,
                )

            return resolved
        except Exception as e:
            logger.warning("Failed reading pair metadata for model path resolution: %s", e)
            return None

    def load_models(self, instrument: Optional[str] = None) -> None:
        """
        Load all 4 models from disk.

        Thread-safe: Uses lock to prevent duplicate loading when called
        from multiple threads (e.g., scanner parallel execution).

        Args:
            instrument: Optional instrument (e.g., 'EUR_USD') to load pair-specific models.
                       If None, uses self.instrument or loads generic models.

        Lookup order for each model:
        1. trained_data/models/{instrument}/ (pair-specific)
        2. trained_data/models/joint/ (joint training)
        3. trained_data/models/ (generic fallback)
        """
        # Update instrument if provided
        if instrument:
            self.instrument = instrument
        import warnings
        from src.training.modular_trainers import (
            TCNTrainer, TCNVolatilityRegimeTrainer, TransformerDirectionTrainer, TransformerRegimeTrainer,
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

        # Prefer explicit pair-specific paths recorded at training time.
        meta_direction_path = self._get_pair_metadata_model_path("direction", ".keras")
        meta_xgb_path = self._get_pair_metadata_model_path("xgboost", ".pkl")
        meta_rf_path = self._get_pair_metadata_model_path("rf", ".pkl")
        meta_ridge_path = self._get_pair_metadata_model_path("ridge", ".pkl")

        # Use pair-specific paths with fallback to generic
        regime_path = self._get_model_path("transformer_regime", ".keras")
        transformer_path = meta_direction_path or self._get_model_path("transformer_direction", ".keras")
        tcn_path = self._get_model_path("tcn_volatility_regime", ".keras")  # NEW: volatility regime model
        tcn_legacy_path = self._get_model_path("tcn_direction", ".keras")  # Legacy: direction model
        histgb_path = self._get_model_path("histgb_direction", ".pkl")

        # Track TCN volatility filter status
        self.use_tcn_volatility_filter = False
        self.tcn_volatility_model = None  # TCN for volatility regime classification
        self.use_tcn_transformer_ensemble = False  # Legacy ensemble mode disabled
        self.tcn_model = None  # Legacy TCN model for direction

        if regime_path.exists():
            # REGIME MODE
            try:
                self.regime_model = TransformerRegimeTrainer()
                self.regime_model.load(str(regime_path))
                self.use_regime = True
                logger.info(f"✓ Transformer REGIME model loaded from {regime_path}")
                self._log_model_trained_at(regime_path, "Transformer regime", category="direction")
            except Exception as e:
                self.regime_model = None
                self.use_regime = False
                logger.warning(f"⚠️ Transformer REGIME model failed to load: {e}")
                logger.warning("⚠️ Model may have been saved with different Keras version")
                if not self.config.permissive_mode:
                    logger.info("ℹ Auto-enabling permissive_mode due to model load failure")
                    self.config.permissive_mode = True
        elif transformer_path.exists():
            # DIRECTION MODE (Transformer)
            try:
                self.tcn = TransformerDirectionTrainer()
                self.tcn.load(str(transformer_path))
                self.use_regime = False
                logger.info(f"✓ Transformer direction model loaded from {transformer_path}")
                self._log_model_trained_at(transformer_path, "Transformer direction", category="direction")

                # === Log training state from lineage (v2) ===
                if hasattr(self.tcn, 'lineage') and self.tcn.lineage:
                    lineage = self.tcn.lineage
                    variance_weight = getattr(lineage, 'auto_variance_weight', 'N/A')
                    lr_reductions = getattr(lineage, 'lr_reductions_count', 0)
                    collapse_recoveries = getattr(lineage, 'collapse_recovery_count', 0)
                    lineage_ver = getattr(lineage, 'lineage_version', 1)
                    instrument_trained = getattr(lineage, 'instrument', 'unknown')
                    logger.info(
                        f"📊 Model training state: instrument={instrument_trained}, "
                        f"variance_weight={variance_weight}, lr_reductions={lr_reductions}, "
                        f"collapse_recoveries={collapse_recoveries}, lineage_v{lineage_ver}"
                    )
            except Exception as e:
                self.tcn = None
                self.use_regime = False
                logger.warning(f"⚠️ Transformer direction model failed to load: {e}")
                logger.warning("⚠️ Model may have been saved with different Keras version")
                if not self.config.permissive_mode:
                    logger.info("ℹ Auto-enabling permissive_mode due to model load failure")
                    self.config.permissive_mode = True
        elif tcn_legacy_path.exists():
            # DIRECTION MODE (TCN legacy - old direction model)
            self.tcn = TCNTrainer()
            self.tcn.load(str(tcn_legacy_path))
            self.use_regime = False
            logger.info(f"✓ TCN direction model loaded from {tcn_legacy_path} (legacy)")
            self._log_model_trained_at(tcn_legacy_path, "TCN direction", category="direction")
        else:
            logger.warning(f"No direction/regime model found for {self.instrument or 'generic'}")

        # === TCN VOLATILITY REGIME FILTER (NEW) ===
        # TCN now predicts volatility regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)
        # Used as mandatory entry timing filter - blocks trades in LOW/NORMAL volatility
        if tcn_path.exists():
            try:
                self.tcn_volatility_model = TCNVolatilityRegimeTrainer()
                self.tcn_volatility_model.load(str(tcn_path))
                self.use_tcn_volatility_filter = True
                logger.info(f"✓ TCN Volatility Regime filter loaded from {tcn_path}")
                logger.info(f"🎯 TCN Volatility Filter ENABLED (min_regime={self.config.min_volatility_regime})")
                self._log_model_trained_at(tcn_path, "TCN volatility regime", category="volatility")
            except Exception as e:
                # Handle cross-Keras-version incompatibility gracefully
                self.tcn_volatility_model = None
                self.use_tcn_volatility_filter = False
                logger.warning(f"⚠️ TCN Volatility model failed to load: {e}")
                logger.warning("⚠️ Model may have been saved with different Keras version (2.x vs 3.x)")
                logger.warning("⚠️ To fix: Retrain TCN volatility on current environment, or save with cross-version format")
                logger.info("ℹ Continuing without TCN volatility filter - trades allowed without volatility gate")
        else:
            self.use_tcn_volatility_filter = False
            if self.config.tcn_required:
                logger.warning(f"⚠️ TCN Volatility Regime model NOT FOUND at {tcn_path}")
                logger.warning("⚠️ tcn_required=True: ALL TRADES WILL BE BLOCKED until TCN is trained")
            else:
                logger.info("ℹ TCN Volatility filter not found - trades allowed without volatility gate")

        # Load HistGB for hybrid voting (if available)
        if histgb_path.exists():
            self.histgb = HistGradientBoostingDirectionTrainer()
            self.histgb.load(str(histgb_path))
            self.use_hybrid = True
            logger.info(f"✓ HistGB baseline loaded from {histgb_path}")
            self._log_model_trained_at(histgb_path, "HistGB baseline", category="ensemble")
        else:
            self.use_hybrid = False
            logger.info("ℹ HistGB not found - single-model mode")

        # XGBoost or LightGBM Momentum (check LightGBM first for joint models)
        lgbm_momentum_path = self._get_model_path("lgbm_momentum", ".pkl")
        xgb_path = self._get_model_path("xgb_momentum", ".pkl")
        pair_xgb_path = (
            self.model_dir / self.instrument / "xgb_momentum.pkl"
            if self.instrument and self.instrument != "GENERIC"
            else None
        )

        # Priority: metadata-pinned pair artifact > explicit pair xgb > freshest fallback.
        if meta_xgb_path is not None and meta_xgb_path.exists():
            momentum_candidates = [meta_xgb_path]
        elif pair_xgb_path is not None and pair_xgb_path.exists():
            momentum_candidates = [pair_xgb_path]
        else:
            momentum_candidates = [p for p in [lgbm_momentum_path, xgb_path] if p.exists()]
        momentum_path = None
        if momentum_candidates:
            momentum_path = (
                momentum_candidates[0]
                if len(momentum_candidates) == 1
                else max(momentum_candidates, key=lambda p: p.stat().st_mtime)
            )

        if momentum_path is not None:
            # Try LightGBM loader first (some runs save LightGBM momentum under xgb_momentum.pkl).
            try:
                from src.training.modular_trainers import LightGBMMomentumTrainer

                self.xgb = LightGBMMomentumTrainer()
                self.xgb.load(str(momentum_path))
                logger.info(f"✓ LightGBM Momentum loaded from {momentum_path}")
                self._log_model_trained_at(momentum_path, "LightGBM momentum", category="gates")
            except Exception as e:
                logger.warning(f"Failed to load LightGBM momentum from {momentum_path}: {e}")
                # Fall back to XGBoost loader
                try:
                    self.xgb = XGBoostTrainer()
                    self.xgb.load(str(momentum_path))
                    logger.info(f"✓ XGBoost loaded from {momentum_path}")
                except Exception as e2:
                    logger.warning(f"Failed to load XGBoost momentum from {momentum_path}: {e2}")
                    self.xgb = None
                else:
                    self._log_model_trained_at(momentum_path, "XGBoost momentum", category="gates")
        else:
            logger.warning("Momentum model not found (tried lgbm_momentum.pkl and xgb_momentum.pkl)")

        # Random Forest or LightGBM Risk (check LightGBM first for joint models)
        lgbm_risk_path = self._get_model_path("lgbm_risk", ".pkl")
        rf_path = self._get_model_path("rf_risk", ".pkl")
        pair_rf_path = (
            self.model_dir / self.instrument / "rf_risk.pkl"
            if self.instrument and self.instrument != "GENERIC"
            else None
        )

        # Prefer pair RF risk artifact when available (matches training outputs + metadata).
        preferred_rf_path = None
        if meta_rf_path is not None and meta_rf_path.exists():
            preferred_rf_path = meta_rf_path
        elif pair_rf_path is not None and pair_rf_path.exists():
            preferred_rf_path = pair_rf_path

        risk_loaded = False
        if preferred_rf_path is not None:
            try:
                self.rf = RandomForestTrainer()
                self.rf.load(str(preferred_rf_path))
                logger.info(f"✓ Random Forest loaded from {preferred_rf_path}")
                self._log_model_trained_at(preferred_rf_path, "Random Forest risk", category="gates")
                risk_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load preferred RF risk from {preferred_rf_path}: {e}")

        if not risk_loaded and lgbm_risk_path.exists():
            # Use LightGBM risk (from joint training)
            try:
                from src.training.modular_trainers import LightGBMRiskTrainer
                self.rf = LightGBMRiskTrainer()
                self.rf.load(str(lgbm_risk_path))
                logger.info(f"✓ LightGBM Risk loaded from {lgbm_risk_path}")
                self._log_model_trained_at(lgbm_risk_path, "LightGBM risk", category="gates")
            except Exception as e:
                logger.warning(f"Failed to load LightGBM risk: {e}")
                # Fall back to RandomForest
                if rf_path.exists():
                    self.rf = RandomForestTrainer()
                    self.rf.load(str(rf_path))
                    logger.info(f"✓ Random Forest loaded from {rf_path} (fallback)")
                    self._log_model_trained_at(rf_path, "Random Forest risk", category="gates")
        elif not risk_loaded and rf_path.exists():
            self.rf = RandomForestTrainer()
            self.rf.load(str(rf_path))
            logger.info(f"✓ Random Forest loaded from {rf_path}")
            self._log_model_trained_at(rf_path, "Random Forest risk", category="gates")
        elif not risk_loaded:
            logger.warning("Risk model not found (tried lgbm_risk.pkl and rf_risk.pkl)")

        # Ridge (use pair-specific path)
        ridge_path = meta_ridge_path or self._get_model_path("ridge_confidence", ".pkl")
        if ridge_path.exists():
            self.ridge = RidgeTrainer()
            self.ridge.load(str(ridge_path))
            logger.info(f"✓ Ridge loaded from {ridge_path}")
            self._log_model_trained_at(ridge_path, "Ridge confidence", category="gates")
        else:
            logger.warning(f"Ridge model not found at {ridge_path}")

        # RL Position Sizer (lazy loaded with timeout to avoid TF/PyTorch GPU conflicts)
        # Check pair-specific path first, then generic
        rl_model_path = None
        rl_scaler_path = None
        if self.instrument and self.instrument != "GENERIC":
            pair_rl = self.model_dir / self.instrument / "rl_position_sizer.zip"
            pair_rl_onnx = self.model_dir / self.instrument / "rl_position_sizer.onnx"
            pair_rl_scaler = self.model_dir / self.instrument / "rl_scaler.pkl"
            if pair_rl.exists() or pair_rl_onnx.exists():
                rl_model_path = pair_rl if pair_rl.exists() else pair_rl_onnx
                rl_scaler_path = pair_rl_scaler if pair_rl_scaler.exists() else None
                logger.info(f"🤖 Pair-specific RL model found: {rl_model_path}")
        if rl_model_path is None:
            if RL_MODEL_PATH.exists():
                rl_model_path = RL_MODEL_PATH
            elif RL_ONNX_PATH.exists():
                rl_model_path = RL_ONNX_PATH

        # Auto-detect: if use_rl_sizer is None, enable if model exists
        if self.use_rl_sizer is None:
            if rl_model_path is not None:
                logger.info("🤖 RL model detected - auto-enabling RL position sizer")
                self.use_rl_sizer = True
            else:
                self.use_rl_sizer = False

        if self.use_rl_sizer:
            RLSizer, rl_available = _lazy_load_rl_sizer()
            if rl_available and RLSizer is not None:
                self.rl_sizer = RLSizer()
                load_kwargs = {}
                if rl_model_path is not None:
                    load_kwargs['model_path'] = rl_model_path
                if rl_scaler_path is not None:
                    load_kwargs['scaler_path'] = rl_scaler_path
                if self.rl_sizer.load(**load_kwargs):
                    logger.info(f"✓ RL Position Sizer loaded (PPO) from {rl_model_path}")
                    self._log_model_trained_at(
                        rl_model_path or RL_MODEL_PATH,
                        "RL position sizer",
                        category="rl_sizer",
                    )
                else:
                    logger.info("ℹ RL Position Sizer not trained - using heuristic sizing")
                    self.rl_sizer = None
            else:
                logger.warning("⚠️ RL sizer load failed/timed out - using heuristic sizing")
                self.rl_sizer = None

        # Auto-detect RL gates and exits if not explicitly set
        if self.use_rl_gates is None:
            if RL_GATE_MODEL_PATH.exists():
                if _allow_rl_gates_exits_autoload():
                    logger.info("🔍 RL Gate model detected - auto-enabling RL gates")
                    self.use_rl_gates = True
                else:
                    logger.warning(
                        "⚠️ RL Gate model detected but auto-enable is disabled. "
                        "Unset BUDDY_DISABLE_RL_GATES_EXITS or set BUDDY_ENABLE_RL_GATES_EXITS=1 to override."
                    )
                    self.use_rl_gates = False
            else:
                self.use_rl_gates = False

        if self.use_rl_exits is None:
            if RL_EXIT_MODEL_PATH.exists():
                if _allow_rl_gates_exits_autoload():
                    logger.info("🔍 RL Exit model detected - auto-enabling RL exits")
                    self.use_rl_exits = True
                else:
                    logger.warning(
                        "⚠️ RL Exit model detected but auto-enable is disabled. "
                        "Unset BUDDY_DISABLE_RL_GATES_EXITS or set BUDDY_ENABLE_RL_GATES_EXITS=1 to override."
                    )
                    self.use_rl_exits = False
            else:
                self.use_rl_exits = False

        # RL Gate Threshold Optimizer + RL Optimal Exit Timing
        # On macOS (Darwin) these are loaded in child processes to avoid
        # TF+PyTorch segfaults.  We start both workers IN PARALLEL to avoid
        # a sequential 2×timeout wait that can appear to hang (~40-120 s).
        import sys as _sys
        _is_darwin = _sys.platform == 'darwin'

        if (self.use_rl_gates or self.use_rl_exits) and _is_darwin:
            import threading

            _gates_result: dict = {}
            _exits_result: dict = {}

            def _boot_gates():
                logger.info("⏳ Starting RL Gates worker subprocess…")
                ok, err = self._start_rl_gates_worker()
                _gates_result['ok'] = ok
                _gates_result['error'] = err

            def _boot_exits():
                logger.info("⏳ Starting RL Exits worker subprocess…")
                ok, err = self._start_rl_exits_worker()
                _exits_result['ok'] = ok
                _exits_result['error'] = err

            threads = []
            if self.use_rl_gates:
                t = threading.Thread(target=_boot_gates, daemon=True)
                t.start()
                threads.append(t)
            if self.use_rl_exits:
                t = threading.Thread(target=_boot_exits, daemon=True)
                t.start()
                threads.append(t)

            for t in threads:
                t.join(timeout=RL_SUBPROCESS_LOAD_TIMEOUT_SECONDS + 2)

            # Process gates result
            if self.use_rl_gates:
                if _gates_result.get('ok'):
                    logger.info("✓ RL Gate Threshold Optimizer loaded (SAC) [subprocess]")
                    self._rl_gates_loaded = True
                    self.rl_gate_optimizer = None
                    self._log_model_trained_at(RL_GATE_MODEL_PATH, "RL gate thresholds", category="rl_gates")
                else:
                    err = _gates_result.get('error', 'unknown')
                    logger.warning(f"⚠️ RL Gates worker failed to start: {err} — using fixed thresholds")
                    self._rl_gates_loaded = False
                    self.rl_gate_optimizer = None

            # Process exits result
            if self.use_rl_exits:
                if _exits_result.get('ok'):
                    logger.info("✓ RL Optimal Exit Timing loaded (PPO) [subprocess]")
                    self._rl_exits_loaded = True
                    self.rl_exit_optimizer = None
                    self._log_model_trained_at(RL_EXIT_MODEL_PATH, "RL optimal exits", category="rl_exits")
                else:
                    err = _exits_result.get('error', 'unknown')
                    logger.warning(f"⚠️ RL Exits worker failed to start: {err} — using fixed R:R ratios")
                    self._rl_exits_loaded = False
                    self.rl_exit_optimizer = None

        else:
            # Non-Darwin (Linux/Windows) or neither gates nor exits enabled
            if self.use_rl_gates:
                GateRL, gates_available = _lazy_load_rl_gates()
                if gates_available and GateRL is not None:
                    try:
                        self.rl_gate_optimizer = GateRL()
                        timed_out, loaded = _call_with_timeout(self.rl_gate_optimizer.load, RL_LOAD_TIMEOUT_SECONDS)
                        if timed_out:
                            raise TimeoutError("RL gates load timed out")
                        if loaded:
                            logger.info("✓ RL Gate Threshold Optimizer loaded (SAC)")
                            self._rl_gates_loaded = True
                            self._log_model_trained_at(RL_GATE_MODEL_PATH, "RL gate thresholds", category="rl_gates")
                        else:
                            logger.info("ℹ RL Gates not trained - using fixed thresholds")
                            self.rl_gate_optimizer = None
                    except TimeoutError:
                        logger.warning(
                            f"⚠️ RL Gates load timed out after {RL_LOAD_TIMEOUT_SECONDS}s - using fixed thresholds"
                        )
                        self.rl_gate_optimizer = None
                    except Exception as e:
                        logger.warning(f"⚠️ RL Gates load failed: {e}")
                        self.rl_gate_optimizer = None
                else:
                    logger.warning("⚠️ RL Gates requested but dependencies not available")
                    self.rl_gate_optimizer = None

            if self.use_rl_exits:
                ExitRL, exits_available = _lazy_load_rl_exits()
                if exits_available and ExitRL is not None:
                    try:
                        self.rl_exit_optimizer = ExitRL()
                        timed_out, loaded = _call_with_timeout(self.rl_exit_optimizer.load, RL_LOAD_TIMEOUT_SECONDS)
                        if timed_out:
                            raise TimeoutError("RL exits load timed out")
                        if loaded:
                            logger.info("✓ RL Optimal Exit Timing loaded (PPO)")
                            self._rl_exits_loaded = True
                            self._log_model_trained_at(RL_EXIT_MODEL_PATH, "RL optimal exits", category="rl_exits")
                        else:
                            logger.info("ℹ RL Exits not trained - using fixed R:R ratios")
                            self.rl_exit_optimizer = None
                    except TimeoutError:
                        logger.warning(
                            f"⚠️ RL Exits load timed out after {RL_LOAD_TIMEOUT_SECONDS}s - using fixed R:R ratios"
                        )
                        self.rl_exit_optimizer = None
                    except Exception as e:
                        logger.warning(f"⚠️ RL Exits load failed: {e}")
                        self.rl_exit_optimizer = None
                else:
                    logger.warning("⚠️ RL Exits requested but dependencies not available")
                    self.rl_exit_optimizer = None

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

        # === MODEL LOAD SUMMARY ===
        # Log explicit status for each model so loading failures are
        # immediately visible and distinguishable from gate invocation issues.
        _status = lambda obj, label: f"✓ {label}" if obj is not None else f"✗ {label} (not loaded)"  # noqa: E731
        summary_lines = [
            _status(self.tcn, "Transformer/Direction"),
            _status(self.tcn_volatility_model, "TCN Volatility Regime"),
            _status(self.xgb, "XGBoost Momentum"),
            _status(self.rf, "RandomForest Risk"),
            _status(self.ridge, "Ridge Confidence"),
        ]
        if self.config.enable_meta_labeling:
            meta = getattr(self, 'meta_labeler', None)
            summary_lines.append(_status(meta, "Meta-Labeler"))
        if self.config.enable_calibration:
            cal_fitted = self.calibrator.is_fitted if getattr(self, 'calibrator', None) else False
            cal_label = "Calibrator (fitted)" if cal_fitted else "Calibrator (not fitted)"
            summary_lines.append(f"{'✓' if cal_fitted else '⚠'} {cal_label}")

        logger.info(f"Modular ensemble loaded{pair_info}.")
        logger.info("Model load summary:")
        for line in summary_lines:
            logger.info(f"  {line}")

    def _log_model_trained_at(self, model_path: Path, label: str, category: str = "model") -> None:
        """Log trained_at metadata for a model if available."""
        if model_path is None:
            return

        category = category or "model"

        try:
            display_path = str(model_path.relative_to(self.model_dir))
        except Exception:
            display_path = str(model_path)

        meta_candidates = [
            model_path.with_suffix(".meta.pkl"),
            model_path.with_suffix(".meta.json"),
        ]

        meta: Optional[Dict[str, Any]] = None
        for candidate in meta_candidates:
            if not candidate.exists():
                continue
            try:
                if candidate.suffix == ".pkl":
                    with open(candidate, "rb") as handle:
                        meta = pickle.load(handle)
                else:
                    with open(candidate, "r") as handle:
                        meta = json.load(handle)
                break
            except Exception:
                meta = None

        trained_at = None
        if meta:
            trained_at = meta.get("trained_at") or meta.get("retrained_at") or meta.get("gate_models_retrained_at")

        source = "metadata"
        if trained_at is None:
            try:
                mtime = model_path.stat().st_mtime
                trained_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                source = "file_mtime"
            except Exception:
                trained_at = None
                source = "unknown"

        trained_at_display = str(trained_at) if trained_at else "unknown"
        age_days = None
        if trained_at:
            try:
                trained_dt = datetime.fromisoformat(trained_at_display.replace("Z", "+00:00"))
                if trained_dt.tzinfo is None:
                    trained_dt = trained_dt.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - trained_dt).days
            except Exception:
                age_days = None

        metrics = None
        if meta:
            metrics = meta.get("metrics") or meta.get("results")
            if metrics is None:
                # Common single-metric keys
                metrics = {
                    "val_accuracy": meta.get("val_accuracy"),
                    "best_val_accuracy": meta.get("best_val_accuracy"),
                }

        self._model_load_report.append(
            {
                "label": label,
                "path": display_path,
                "path_source": self._get_model_source(model_path),
                "category": category,
                "trained_at": trained_at_display,
                "age_days": age_days,
                "timestamp_source": source,
                "metrics": metrics,
            }
        )

        if trained_at is None:
            logger.info(f"🕒 {label} trained_at=unknown")
        elif age_days is not None:
            logger.info(f"🕒 {label} trained_at={trained_at_display} ({age_days}d ago, {source})")
        else:
            logger.info(f"🕒 {label} trained_at={trained_at_display} ({source})")

    def _get_model_source(self, model_path: Path) -> str:
        """Return model source: pair, joint, or generic."""
        try:
            if self.instrument and self.instrument != "GENERIC":
                pair_dir = self.model_dir / self.instrument
                if pair_dir in model_path.parents:
                    return "pair"
            joint_dir = self.model_dir / "joint"
            if joint_dir in model_path.parents:
                return "joint"
        except Exception:
            return "unknown"
        return "generic"

    def get_model_load_report(self) -> List[Dict[str, Any]]:
        """Return model load metadata collected during load_models."""
        return list(self._model_load_report)

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

        # Check model quality from metadata (prefer pair-specific)
        meta_path = None
        if self.instrument and self.instrument != "GENERIC":
            _pair_meta = self.model_dir / self.instrument / "modular_ensemble.meta.json"
            if _pair_meta.exists():
                meta_path = _pair_meta
        if meta_path is None:
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
            logger.info(
                f"📊 Gate status: {usable_gates}/{total_gates} gates usable. "
                f"Issues: {self._gate_issues}"
            )
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

        # Try loading from model metadata (prefer pair-specific)
        meta_path = None
        if self.instrument and self.instrument != "GENERIC":
            _pair_meta = self.model_dir / self.instrument / "modular_ensemble.meta.json"
            if _pair_meta.exists():
                meta_path = _pair_meta
        if meta_path is None:
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
                        logger.info("✓ Platt calibration loaded from metadata")

                    if 'isotonic_params' in calib_data:
                        # Reconstruct Isotonic model from saved parameters
                        from sklearn.isotonic import IsotonicRegression
                        self.calibrator.isotonic_model = IsotonicRegression(out_of_bounds='clip')
                        self.calibrator.isotonic_model.X_thresholds_ = np.array(calib_data['isotonic_params']['X_thresholds'])
                        self.calibrator.isotonic_model.y_thresholds_ = np.array(calib_data['isotonic_params']['y_thresholds'])
                        self.calibrator.isotonic_model.f_ = None  # Will be rebuilt on predict
                        self.calibrator.is_fitted = True
                        self._calibration_loaded = True
                        logger.info("✓ Isotonic calibration loaded from metadata")

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
            logger.info("ℹ Calibration enabled but no fitted calibrator found - using raw probabilities")
            logger.info("  To enable calibration, train with: python main.py train-buddy --calibrate")
            logger.info(f"  Or run: python -m confidence_calibration --train --model-dir {self.model_dir}")

    def _load_learned_confidence(self) -> None:
        """
        Load learned confidence model (Phase 3: Scanner Accuracy).

        This model replaces the heuristic _compute_confidence_direct() formula
        with weights learned from actual trade outcomes.

        Lookup order:
        1. trained_data/models/{instrument}/learned_confidence.pkl  (pair-specific)
        2. trained_data/models/learned_confidence.pkl  (generic fallback)
        """
        if not LEARNED_CONFIDENCE_AVAILABLE:
            logger.debug("Learned confidence module not available")
            return

        self._learned_confidence_loaded = False
        self.learned_confidence = None

        # Build list of paths to check
        learned_paths = []
        if self.instrument and self.instrument != "GENERIC":
            learned_paths.append(self.model_dir / self.instrument / "learned_confidence.pkl")
        learned_paths.append(self.model_dir / "learned_confidence.pkl")

        for path in learned_paths:
            if path.exists():
                try:
                    self.learned_confidence = LearnedConfidenceModel.load(path)
                    if self.learned_confidence.is_fitted:
                        self._learned_confidence_loaded = True
                        logger.info(f"✓ Learned confidence model loaded (CV R²={self.learned_confidence.training_stats.get('cv_score', 0):.3f})")
                        return
                    else:
                        logger.debug(f"Learned confidence model at {path} is not fitted")
                        self.learned_confidence = None
                except Exception as e:
                    logger.debug(f"Failed to load learned confidence from {path}: {e}")

        logger.debug("No fitted learned confidence model found - using heuristic formula")

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
            self.model_dir / "buddy_xgb_meta.pkl",  # Training saves with suffix
        ]

        # Add pair-specific path first if instrument is set
        if self.instrument and self.instrument != "GENERIC":
            meta_labeler_paths.insert(0, self.model_dir / self.instrument / "meta_labeler.pkl")
            meta_labeler_paths.insert(1, self.model_dir / self.instrument / "buddy_xgb_meta.pkl")  # Also check suffixed filename

        for path in meta_labeler_paths:
            if path.exists():
                try:
                    loaded_labeler = MetaLabeler.load(path)

                    # Guardrail: skip stale meta-labelers trained on materially different
                    # feature dimensions, as they can over-filter valid trades.
                    if not self._is_meta_labeler_compatible(loaded_labeler):
                        logger.warning(
                            "Skipping incompatible meta-labeler at %s (feature-dimension mismatch)",
                            path,
                        )
                        continue

                    self.meta_labeler = loaded_labeler
                    self._meta_labeler_loaded = True
                    threshold = self.config.min_meta_confidence
                    primary_acc = getattr(self.meta_labeler, '_primary_accuracy', 0.5)
                    logger.info(f"✓ Meta-labeler loaded from {path}")
                    logger.info(f"  Threshold: {threshold:.0%}, Primary accuracy: {primary_acc:.1%}")
                    self._log_model_trained_at(path, "Meta labeler", category="meta")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load meta-labeler from {path}: {e}")

        # No meta-labeler found
        if self.config.enable_meta_labeling:
            logger.info("ℹ Meta-labeling enabled but no trained model found")
            logger.info("  To train: models save meta_labeler.pkl during buddy training")

    def _is_meta_labeler_compatible(self, labeler: Any) -> bool:
        """Return True when meta-labeler feature shape is plausible for current pipeline."""
        try:
            meta_model = getattr(labeler, "meta_model", None)
            expected = getattr(meta_model, "n_features_in_", None)
            if expected is None:
                return True

            # Estimate expected meta-feature dimension from current direction pipeline.
            direction_features = get_normalized_feature_names().get("direction", [])
            if not isinstance(direction_features, list) or len(direction_features) == 0:
                return True

            n_primary = len(direction_features)
            meta_cfg = getattr(labeler, "config", None)
            use_reduced = bool(getattr(meta_cfg, "use_reduced_features", True))
            include_primary_proba = bool(getattr(meta_cfg, "include_primary_proba", True))

            estimated = n_primary if use_reduced else (n_primary * 3)
            if include_primary_proba:
                estimated += 2

            tolerance = max(8, int(0.20 * estimated))
            compatible = abs(int(expected) - int(estimated)) <= tolerance
            if not compatible:
                logger.warning(
                    "Meta-labeler feature mismatch: model expects %d, current pipeline estimates %d "
                    "(tolerance=%d).",
                    int(expected),
                    int(estimated),
                    int(tolerance),
                )
            return compatible
        except Exception:
            return True

    def _apply_calibration(self, raw_probability: float, direction: Optional[int] = None) -> tuple[float, bool]:
        """
        Apply confidence calibration to raw model probability.

        Args:
            raw_probability: Raw probability from TCN/Transformer (0-1)
            direction: Predicted direction (0=short, 1=long)

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
            # Calibrate confidence of the predicted class, then map back to
            # directional probability. This preserves short-side (<0.5) outputs.
            pred_dir = direction
            if pred_dir is None:
                pred_dir = 1 if raw_probability >= 0.5 else 0

            raw_conf = raw_probability if pred_dir == 1 else (1.0 - raw_probability)
            result = self.calibrator.calibrate_confidence(raw_conf)
            calibrated_conf = float(result.calibrated_confidence)
            calibrated = calibrated_conf if pred_dir == 1 else (1.0 - calibrated_conf)

            # Safety clamp after mapping back
            calibrated = max(0.0, min(1.0, calibrated))

            # Log calibration adjustment if significant
            adjustment = calibrated - raw_probability
            if abs(adjustment) > 0.02:
                logger.debug(
                    "📐 Calibration: raw_prob=%.3f raw_conf=%.3f → cal_conf=%.3f cal_prob=%.3f (Δ=%+.3f)",
                    raw_probability,
                    raw_conf,
                    calibrated_conf,
                    calibrated,
                    adjustment,
                )

            return calibrated, True
        except Exception as e:
            logger.warning(f"Calibration failed, using raw probability: {e}")
            return raw_probability, False

    def _predict_with_uncertainty(self, features: np.ndarray, n_samples: int = 20) -> tuple[float, float, int]:
        """
        Monte Carlo Dropout for uncertainty estimation.

        Runs the model N times with dropout enabled (training=True) to get
        a distribution of predictions. High std indicates model uncertainty.

        Args:
            features: Input features array (seq_len, n_features)
            n_samples: Number of forward passes (default: 20)

        Returns:
            Tuple of (mean_probability, std_probability, predicted_direction)
        """
        if self.tcn is None or not hasattr(self.tcn, 'model'):
            return 0.5, 0.0, None

        try:
            import tensorflow as tf  # noqa: F401

            # Ensure features have batch dimension
            if features.ndim == 2:
                features = np.expand_dims(features, axis=0)

            # Extract only the last seq_len timesteps
            seq_len = getattr(self.tcn, 'seq_len', 60)
            if features.shape[1] > seq_len:
                features = features[:, -seq_len:, :]  # Take last 60 timesteps

            # Run N forward passes with dropout enabled
            predictions = []
            for _ in range(n_samples):
                # training=True keeps dropout active
                pred = self.tcn.model(features, training=True)
                if isinstance(pred, dict):
                    prob = pred.get('direction', pred.get('probability', 0.5))
                else:
                    prob = pred

                # Handle tensor output
                if hasattr(prob, 'numpy'):
                    prob = prob.numpy()
                if isinstance(prob, np.ndarray):
                    prob = float(prob.flatten()[-1])
                predictions.append(prob)

            predictions = np.array(predictions)
            mean_prob = float(np.mean(predictions))
            std_prob = float(np.std(predictions))

            # Direction from mean probability
            direction = 1 if mean_prob > 0.5 else 0

            logger.debug(f"🎲 MC Dropout (n={n_samples}): mean={mean_prob:.3f}, std={std_prob:.3f}")

            return mean_prob, std_prob, direction

        except Exception as e:
            logger.warning(f"MC Dropout failed: {e}, falling back to single prediction")
            return 0.5, 0.0, None

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
                logger.warning(
                    f"Many features missing ({len(missing_features)}/{n_features}). "
                    f"First 10: {missing_features[:10]}"
                )

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

    # =========================================================================
    # 5-CLASS REGIME DETECTION (Phase 3)
    # =========================================================================

    def _detect_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current market regime using 5-class classification.

        Uses real-time ADX, RSI, and ATR expansion to classify:
        - STRONG_TREND: ADX >= 40 + high volatility
        - WEAK_TREND: ADX 25-40
        - CHOP: ADX < 20 + neutral RSI
        - MEAN_REVERT: RSI < 30 or > 70
        - BREAKOUT: ATR expanding + ADX rising

        Returns:
            Dict with 'regime', 'confidence', 'transition_detected', 'cooldown_active'
        """
        # Import regime classification from data loaders
        try:
            from .modular_data_loaders import (
                classify_market_regime,
                REGIME_NAMES,
            )
        except ImportError:
            logger.warning("Regime classification not available")
            return {
                'regime': 'WEAK_TREND',
                'regime_id': 1,
                'confidence': 0.5,
                'transition_detected': False,
                'cooldown_active': False,
            }

        # Classify regime
        regimes, confidences = classify_market_regime(df, return_confidence=True)

        # Get current (last) regime
        current_regime_id = int(regimes.iloc[-1])
        current_regime = REGIME_NAMES.get(current_regime_id, 'WEAK_TREND')
        current_confidence = float(confidences.iloc[-1])

        # Track regime history for transition detection
        previous_regime = self._current_regime
        self._current_regime = current_regime

        # Update history
        self._regime_history.append(current_regime)
        if len(self._regime_history) > self._regime_history_max_size:
            self._regime_history = self._regime_history[-self._regime_history_max_size:]

        # Detect regime transition
        transition_detected = False
        if previous_regime is not None and previous_regime != current_regime:
            # Check if this is a real transition (not noise)
            # Require 2 out of last 3 to be the new regime to confirm
            recent = self._regime_history[-3:] if len(self._regime_history) >= 3 else self._regime_history
            if recent.count(current_regime) >= 2:
                transition_detected = True
                self._regime_transition_cooldown = self._regime_transition_cooldown_target
                logger.info(f"🔄 Regime transition: {previous_regime} → {current_regime}")

        # Update cooldown counter
        cooldown_active = self._regime_transition_cooldown > 0
        if cooldown_active:
            self._regime_transition_cooldown -= 1

        return {
            'regime': current_regime,
            'regime_id': current_regime_id,
            'confidence': current_confidence,
            'transition_detected': transition_detected,
            'cooldown_active': cooldown_active,
            'cooldown_bars_remaining': self._regime_transition_cooldown,
            'regime_history': self._regime_history[-5:],  # Last 5 for logging
        }

    def _track_regime_transition(self, current_regime: str) -> bool:
        """
        Check if regime has transitioned from previous state.

        Args:
            current_regime: Current detected regime name

        Returns:
            True if regime changed, False otherwise
        """
        if not self._regime_history:
            return False

        # Compare current to most recent
        previous = self._regime_history[-1]
        return previous != current_regime

    def _get_regime_lgbm(self, regime: str) -> Optional[Any]:
        """
        Get the LightGBM model for a specific regime.

        Lazy-loads regime-specific models if not already loaded.
        Falls back to WEAK_TREND model if requested regime not available.

        Args:
            regime: Regime name (STRONG_TREND, WEAK_TREND, CHOP, MEAN_REVERT, BREAKOUT)

        Returns:
            LightGBM model for the regime, or None if not available
        """
        if self._regime_lgbm_trainer is None:
            return None

        model = self._regime_lgbm_trainer.get_model_for_regime(regime)

        if model is None and regime != 'WEAK_TREND':
            # Fallback to WEAK_TREND (most conservative)
            logger.debug(f"No model for {regime}, falling back to WEAK_TREND")
            model = self._regime_lgbm_trainer.get_model_for_regime('WEAK_TREND')

        return model

    def _load_regime_lgbm_models(self, instrument: str) -> bool:
        """
        Load all regime-specific LightGBM models for an instrument.

        Args:
            instrument: Currency pair (e.g., 'EUR_USD')

        Returns:
            True if at least one model loaded, False otherwise
        """
        try:
            from src.training.modular_trainers import RegimeLGBMTrainer
        except ImportError:
            logger.warning("RegimeLGBMTrainer not available")
            return False

        self._regime_lgbm_trainer = RegimeLGBMTrainer()
        loaded_regimes = self._regime_lgbm_trainer.load(
            str(self.model_dir),
            instrument
        )

        if loaded_regimes:
            self._regime_lgbm_loaded = True
            logger.info(f"✓ Loaded regime LightGBM models: {loaded_regimes}")
            return True
        else:
            logger.info("ℹ No regime-specific LightGBM models found")
            return False

    def _predict_with_regime_lgbm(
        self,
        df: pd.DataFrame,
        regime: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get direction prediction from regime-specific LightGBM model.

        Args:
            df: DataFrame with features
            regime: Current market regime

        Returns:
            Dict with 'direction', 'probability', 'regime_used' or None if unavailable
        """
        if not self._regime_lgbm_loaded or self._regime_lgbm_trainer is None:
            return None

        try:
            # Extract features (use same features as XGBoost momentum)
            features = self._extract_xgb_features(df)

            # Get prediction from regime-specific model
            result = self._regime_lgbm_trainer.predict(features, regime=regime)

            return result

        except Exception as e:
            logger.warning(f"Regime LightGBM prediction failed: {e}")
            return None

    def get_volatility_regime(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Get volatility regime prediction for scanner use.

        This is a lightweight helper method for the buddy_scanner to check
        volatility regime without running full inference.

        Args:
            df: DataFrame with OHLCV data and computed features

        Returns:
            Tuple of (regime: int, confidence: float)
            - regime: 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
            - confidence: softmax confidence (0.0-1.0)

        Raises:
            RuntimeError: If TCN volatility model not loaded
        """
        if not self.use_tcn_volatility_filter or self.tcn_volatility_model is None:
            raise RuntimeError("TCN Volatility model not loaded - cannot get regime")

        # Ensure normalized features exist
        required_features = set(get_normalized_feature_names().get('volatility', []))
        if any(fname not in df.columns for fname in required_features):
            df = compute_normalized_features(df)

        # Extract VOLATILITY-SPECIFIC features and predict
        tcn_features = self._extract_tcn_volatility_features(df)
        vol_pred = self.tcn_volatility_model.predict(tcn_features)

        if 'volatility_regime' in vol_pred:
            regime = vol_pred['volatility_regime']
            regime_name = vol_pred.get('volatility_regime_name')
            confidence = vol_pred.get('regime_confidence', vol_pred.get('confidence', 0.0))
        elif 'regime' in vol_pred:
            regime = vol_pred['regime']
            regime_name = vol_pred.get('regime_name')
            confidence = vol_pred.get('confidence', 0.0)
        else:
            raise KeyError("volatility_regime")

        logger.debug(f"Volatility regime: {regime_name} ({regime}) conf={confidence:.1%}")

        return regime, confidence

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

    def _extract_tcn_volatility_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for TCN volatility regime model using saved feature names.

        This is SEPARATE from _extract_tcn_features() because the volatility model
        was trained on ~12-16 volatility-specific features, not the ~50 direction features.

        Includes optional Pandera schema validation (if installed) for early error detection.
        """
        # VOLATILITY-FOCUSED features for regime classification (matches training)
        normalized_features = get_normalized_feature_names().get('volatility', [
            'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'bb_width_norm', 'bb_width_20',
            'high_low_range',
            'returns_1', 'returns_5', 'returns_10',
            'volume_ratio_10', 'volume_ratio_20',
            'adx',
        ])

        # Legacy fallback features for volatility
        legacy_fallback = [
            'atr', 'atr_pct', 'volatility', 'vol_5', 'vol_10', 'vol_20',
            'bb_width', 'high_low_range', 'tr_pct',
        ]

        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['atr', 'volatility', 'vol_', 'range', 'bb_width', 'tr_pct']

        # Use saved feature names from training if available
        feature_names = getattr(self.tcn_volatility_model, 'feature_names', None) if self.tcn_volatility_model else None

        # Feature validation guard: log warning if feature_names not saved
        if feature_names is None:
            logger.warning("TCN volatility model has no saved feature_names, using default volatility features")
        else:
            # Validate expected feature count matches scaler (if available)
            scaler = getattr(self.tcn_volatility_model, 'scaler', None)
            if scaler is not None and hasattr(scaler, 'n_features_in_'):
                expected_n = scaler.n_features_in_
                if len(feature_names) != expected_n:
                    logger.warning(
                        f"Feature count mismatch: model has {len(feature_names)} feature_names "
                        f"but scaler expects {expected_n} features"
                    )

        # Optional Pandera schema validation for early error detection
        if PANDERA_AVAILABLE and feature_names:
            try:
                # Build schema requiring only the features we need
                schema_columns = {
                    fname: pa_pandas.Column(float, required=False, nullable=True)
                    for fname in feature_names if fname in df.columns
                }
                if schema_columns:
                    schema = pa_pandas.DataFrameSchema(schema_columns, strict=False)
                    schema.validate(df, lazy=True)
            except pa.errors.SchemaErrors as exc:
                logger.warning(f"Pandera schema validation warnings: {exc}")
            except Exception as exc:
                logger.debug(f"Pandera validation skipped: {exc}")

        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)

    def _extract_xgb_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for XGBoost model using saved feature names.

        Uses the feature alignment system for consistent handling across all models.
        """
        # Use saved feature names from training if available
        feature_names = getattr(self.xgb, 'feature_names', None) if self.xgb else None

        # If we have saved feature names, use the alignment system
        if feature_names:
            return align_features_to_model(df, feature_names, fill_value=0.0)

        # Fallback to pattern-based extraction
        normalized_features = get_normalized_feature_names()['momentum']

        legacy_fallback = [
            'returns', 'roc_5', 'roc_10', 'roc_20',
            'high_low_ratio', 'momentum_10', 'macd', 'macd_hist',
            'stoch_k', 'stoch_d', 'mfi',
        ]

        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['return', 'atr_pct', 'volatility', 'volume_ratio', 'macd_norm', 'rsi_norm']

        return self._extract_features_by_names(df, None, fallback_preferred, fallback_patterns)

    def _extract_rf_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for Random Forest model using saved feature names.

        Uses the feature alignment system to handle mismatches between stored
        model features and current pipeline features, preventing dimension errors.
        """
        # Use saved feature names from training if available
        feature_names = getattr(self.rf, 'feature_names', None) if self.rf else None
        # Note: n_features could be used for validation but currently unused
        _ = getattr(self.rf, 'n_features', None) if self.rf else None  # Reserved for future validation

        # If we have saved feature names, use the alignment system
        if feature_names:
            # Use align_features_to_model for proper legacy feature mapping
            result = align_features_to_model(df, feature_names, fill_value=0.0)

            # Validate shape matches scaler expectations (if available)
            if hasattr(self.rf, 'scaler') and self.rf.scaler is not None:
                scaler = self.rf.scaler
                expected_n = getattr(scaler, 'n_features_in_', len(feature_names))
                if result.shape[1] != expected_n:
                    logger.warning(
                        f"RF feature count mismatch after alignment: got {result.shape[1]}, "
                        f"scaler expects {expected_n}. Padding/truncating."
                    )
                    # Pad or truncate to match
                    if result.shape[1] < expected_n:
                        padding = np.zeros((result.shape[0], expected_n - result.shape[1]), dtype=np.float32)
                        result = np.concatenate([result, padding], axis=1)
                    elif result.shape[1] > expected_n:
                        result = result[:, :expected_n]

            return result

        # Fallback to pattern-based extraction if no saved feature names
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

        return self._extract_features_by_names(df, None, fallback_preferred, fallback_patterns)

    def _extract_rl_sizer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for RL Position Sizer using direction features.

        The RL position sizer was trained on direction model features (from training data),
        so we must use direction features here, NOT RF features.

        Uses the RL sizer's saved scaler to determine expected feature count.
        """
        # Get expected feature count from RL sizer's scaler
        expected_n_features = None
        if self.rl_sizer is not None and hasattr(self.rl_sizer, 'scaler'):
            scaler = self.rl_sizer.scaler
            if scaler is not None and hasattr(scaler, 'n_features_in_'):
                expected_n_features = scaler.n_features_in_

        # Use DIRECTION features (same as what RL sizer was trained on)
        normalized_features = get_normalized_feature_names()['direction']

        # Legacy fallback features for direction
        legacy_fallback = [
            'returns', 'momentum_10', 'momentum_20',
            'rsi', 'rsi_14', 'stoch_k', 'stoch_d',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'bb_position_20',
        ]

        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['returns_', 'zscore_', 'sma_ratio', 'ema_ratio', 'macd', 'rsi', 'pct_rank']

        # Extract features using the standard method
        features = self._extract_features_by_names(df, None, fallback_preferred, fallback_patterns)

        # Pad or truncate to match expected feature count
        if expected_n_features is not None and features.shape[1] != expected_n_features:
            logger.debug(
                f"RL sizer feature adjustment: got {features.shape[1]}, expected {expected_n_features}"
            )
            if features.shape[1] < expected_n_features:
                # Pad with zeros
                padding = np.zeros((features.shape[0], expected_n_features - features.shape[1]), dtype=np.float32)
                features = np.concatenate([features, padding], axis=1)
            else:
                # Truncate
                features = features[:, :expected_n_features]

        return features

    def _extract_ridge_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for Ridge model using saved feature names.

        Uses the feature alignment system for consistent handling across all models.
        """
        # Use saved feature names from training if available
        feature_names = getattr(self.ridge, 'feature_names', None) if self.ridge else None

        # If we have saved feature names, use the alignment system
        if feature_names:
            return align_features_to_model(df, feature_names, fill_value=0.0)

        # Fallback to pattern-based extraction
        normalized_features = get_normalized_feature_names()['confidence']

        legacy_fallback = [
            'volatility_5', 'volatility_10', 'volatility_20',
            'atr', 'bb_width_20', 'bb_position_20',
            'adx', 'returns',
        ]

        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['atr_pct', 'volatility', 'volume_ratio', 'sma_ratio', 'return', 'zscore']

        return self._extract_features_by_names(df, None, fallback_preferred, fallback_patterns)

    def _compute_confidence_direct(self, df: pd.DataFrame) -> float:
        """
        Compute confidence score directly from indicators (0-100 scale).

        Phase 3 Enhancement: If a learned confidence model is available,
        use it instead of the heuristic formula. The learned model was
        trained on actual trade outcomes and should provide more accurate
        confidence estimates.

        Heuristic formula weights (fallback):
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

        # Phase 3: Try learned confidence model first
        if self._learned_confidence_loaded and self.learned_confidence is not None:
            try:
                features = np.array([adx, rsi, atr_pct, bb_pos, volume_ratio])
                learned_conf = self.learned_confidence.predict(features)
                # Scale to 0-100 range (learned model outputs 0-1)
                confidence = 15 + learned_conf * 80
                logger.debug(f"Using learned confidence: {learned_conf:.3f} -> {confidence:.1f}")
                return float(confidence)
            except Exception as e:
                logger.debug(f"Learned confidence prediction failed, using heuristic: {e}")

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

    def _predict_technical_fallback(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Fallback prediction using technical indicators when models are unavailable.

        This method is ported from multi_pair_inference.py and provides a robust
        fallback when the primary Transformer/TCN models fail to load or predict.

        Uses RSI and MACD to generate direction signals:
        - RSI < 30: LONG (oversold)
        - RSI > 70: SHORT (overbought)
        - MACD histogram > 0 and MACD > 0: LONG (bullish momentum)
        - MACD histogram < 0 and MACD < 0: SHORT (bearish momentum)
        - Otherwise: HOLD

        Args:
            df: DataFrame with OHLCV data and computed features

        Returns:
            Tuple of (direction: str, confidence: float)
            - direction: "LONG", "SHORT", or "HOLD"
            - confidence: 0.0 to 1.0
        """
        try:
            # Compute basic features if not present
            df_feat = df.copy()
            close = df_feat["close"]

            # Compute RSI if not present
            if "rsi" not in df_feat.columns:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, 1e-8)
                df_feat["rsi"] = 100 - (100 / (1 + rs))

            # Compute MACD if not present
            if "macd" not in df_feat.columns or "macd_hist" not in df_feat.columns:
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                df_feat["macd"] = ema12 - ema26
                df_feat["macd_signal"] = df_feat["macd"].ewm(span=9).mean()
                df_feat["macd_hist"] = df_feat["macd"] - df_feat["macd_signal"]

            # Get latest values
            rsi = df_feat["rsi"].iloc[-1]
            macd = df_feat["macd"].iloc[-1]
            macd_hist = df_feat["macd_hist"].iloc[-1]

            # Handle NaN values
            if np.isnan(rsi):
                rsi = 50.0
            if np.isnan(macd):
                macd = 0.0
            if np.isnan(macd_hist):
                macd_hist = 0.0

            # RSI signals (stronger signal)
            if rsi < 30:
                confidence = 0.55 + (30 - rsi) / 100
                logger.debug(f"Technical fallback: RSI oversold ({rsi:.1f}) -> LONG, conf={confidence:.2f}")
                return "LONG", min(confidence, 0.75)
            elif rsi > 70:
                confidence = 0.55 + (rsi - 70) / 100
                logger.debug(f"Technical fallback: RSI overbought ({rsi:.1f}) -> SHORT, conf={confidence:.2f}")
                return "SHORT", min(confidence, 0.75)

            # MACD signals (moderate signal)
            if macd_hist > 0 and macd > 0:
                logger.debug("Technical fallback: MACD bullish -> LONG, conf=0.52")
                return "LONG", 0.52
            elif macd_hist < 0 and macd < 0:
                logger.debug("Technical fallback: MACD bearish -> SHORT, conf=0.52")
                return "SHORT", 0.52

            # No clear signal
            logger.debug(f"Technical fallback: No clear signal (RSI={rsi:.1f}, MACD={macd:.4f}) -> HOLD")
            return "HOLD", 0.5

        except Exception as e:
            logger.warning(f"Technical fallback prediction failed: {e}")
            return "HOLD", 0.5

    def _calculate_position_size(
        self,
        expected_drawdown_pips: float,
        equity: Optional[float] = None,
        instrument: Optional[str] = None,
        current_price: Optional[float] = None,
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
        # Base risk-sized lots (shared by risk-only and RL-hybrid modes)
        # =====================================================================
        risk_amount = equity * self.config.risk_per_trade_pct
        min_stop_loss_pips = max(float(getattr(self.config, "min_stop_loss_pips", 10.0)), 1.0)
        expected_drawdown_pips = max(float(expected_drawdown_pips), min_stop_loss_pips)
        risk_lots = risk_amount / (expected_drawdown_pips * self.config.pip_value)

        # Apply liquidity limit for instrument
        max_lots = self.config.max_lots_by_pair.get(
            instrument,
            self.config.max_lots_by_pair.get('DEFAULT', 0.5)
        ) if instrument else 1.0
        risk_lots = min(risk_lots, max_lots)

        sizing_mode = str(getattr(self.config, "position_sizing_mode", "rl_hybrid")).lower()
        if sizing_mode not in {"rl_hybrid", "rl_only", "risk_only"}:
            logger.warning("Unknown position_sizing_mode='%s' - falling back to rl_hybrid", sizing_mode)
            sizing_mode = "rl_hybrid"

        # =====================================================================
        # RL POSITION SIZING (if available)
        # =====================================================================
        if sizing_mode != "risk_only" and self.rl_sizer is not None and features is not None:
            try:
                # Construct ensemble prediction for RL observation
                ensemble_pred = np.array([tcn_probability, ridge_confidence / 100.0])

                # Get RL-optimized position size (returns $ amount)
                size_dollars = self.rl_sizer.get_position_size(
                    features=features[-1] if features.ndim > 1 else features,
                    ensemble_prediction=ensemble_pred,
                    account_equity=equity,
                )

                # Convert notional USD exposure to lots using current instrument price.
                # 1 lot = 100,000 base units.
                per_lot_notional_usd = 100000.0
                if instrument and isinstance(instrument, str):
                    parts = instrument.split("_")
                    if len(parts) == 2:
                        base_ccy, quote_ccy = parts[0], parts[1]
                        if quote_ccy == "USD" and current_price and current_price > 0:
                            per_lot_notional_usd = float(current_price) * 100000.0
                        elif base_ccy == "USD":
                            per_lot_notional_usd = 100000.0

                rl_lots = max(0.0, float(size_dollars)) / max(per_lot_notional_usd, 1.0)
                rl_lots = min(rl_lots, max_lots)

                # Hybrid mode: RL proposes aggressiveness, but keep risk-budget utilization meaningful.
                # This prevents tiny RL notional outputs from creating effectively zero-risk orders.
                min_util = float(getattr(self.config, "rl_min_risk_utilization", 0.08))
                min_util = min(max(min_util, 0.0), 1.0)
                min_risk_lots = risk_lots * min_util

                if sizing_mode == "rl_only":
                    size_lots = rl_lots
                    floor_lots = float(getattr(self.config, "min_position_lots", 0.01))
                else:
                    size_lots = max(rl_lots, min_risk_lots)
                    floor_lots = max(float(getattr(self.config, "min_position_lots", 0.01)), min_risk_lots)

                floor_lots = min(floor_lots, max_lots)
                size_lots = min(size_lots, max_lots)
                size_lots = max(floor_lots, size_lots)
                size_lots = min(size_lots, max_lots)

                self._last_size_details = {
                    "mode": "rl",
                    "position_sizing_mode": sizing_mode,
                    "equity": equity,
                    "size_dollars": size_dollars,
                    "position_pct": size_dollars / equity if equity else 0.0,
                    "per_lot_notional_usd": per_lot_notional_usd,
                    "rl_lots": rl_lots,
                    "risk_lots": risk_lots,
                    "min_risk_lots": min_risk_lots,
                    "size_lots": size_lots,
                    "max_lots": max_lots,
                    "min_position_lots": float(getattr(self.config, "min_position_lots", 0.01)),
                    "rl_min_risk_utilization": min_util,
                    "expected_drawdown_pips": expected_drawdown_pips,
                }

                logger.debug(
                    "RL position size (%s): rl=%.2f lots, risk_ref=%.2f lots, final=%.2f lots ($%.0f notional)",
                    sizing_mode,
                    rl_lots,
                    risk_lots,
                    size_lots,
                    size_dollars,
                )
                min_position_lots = min(float(getattr(self.config, "min_position_lots", 0.01)), max_lots)
                return round(max(min_position_lots, size_lots), 2)

            except Exception as e:
                logger.warning(f"RL sizing failed, falling back to heuristic: {e}")

        # =====================================================================
        # HEURISTIC RISK-BASED SIZING (fallback)
        # =====================================================================
        min_position_lots = min(float(getattr(self.config, "min_position_lots", 0.01)), max_lots)
        size_lots = min(max_lots, max(min_position_lots, risk_lots))

        logger.debug(
            f"Position sizing: equity=${equity:,.0f} risk={risk_amount:.0f} "
            f"dd_pips={expected_drawdown_pips:.1f} → {size_lots:.2f} lots "
            f"({int(size_lots * 100000):,} units) [max={max_lots:.1f}]"
        )

        self._last_size_details = {
            "mode": "heuristic",
            "equity": equity,
            "risk_amount": risk_amount,
            "risk_pct": self.config.risk_per_trade_pct,
            "expected_drawdown_pips": expected_drawdown_pips,
            "pip_value": self.config.pip_value,
            "size_lots": size_lots,
            "max_lots": max_lots,
            "min_lots": min_position_lots,
            "position_sizing_mode": sizing_mode,
        }

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
        # Reload models if instrument changed or not loaded
        if not self._loaded:
            if instrument:
                self.instrument = instrument
            self.load_models()
        elif instrument and instrument != self._loaded_instrument:
            logger.info(f"📊 Instrument changed ({self._loaded_instrument} → {instrument}), reloading pair-specific models")
            self._loaded = False
            self.instrument = instrument
            # Preserve explicit user choice for RL sizer (only auto-detect if originally None)
            # Don't reset to None when user explicitly passed False
            if self.use_rl_sizer is not False:
                self.use_rl_sizer = None
            self.rl_sizer = None
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
                    logger.info(
                        f"📰 Sentiment: {sent['aggregate_label']} ({sent['aggregate_score']:+.2f}) "
                        f"from {sent['num_headlines']} headlines"
                    )

                if 'next_high_impact' in intel_data:
                    event = intel_data['next_high_impact']
                    logger.info(
                        f"📅 Next high-impact: {event['name']} ({event['currency']}) "
                        f"in {int(event['minutes_until'])} minutes"
                    )

            except Exception as e:
                logger.warning(f"Market intelligence check failed: {e}")
                intel_data = {'error': str(e)}

        # === ATR VOLATILITY FILTER (NEW) ===
        # Skip low-volatility conditions where spreads eat into profits
        # Ensure normalized features exist early (needed by TCN volatility model + some gates)
        if 'returns_1' not in df.columns:
            df = compute_normalized_features(df)

        if self.config.min_atr_pips > 0 and 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
            # Determine pip value based on instrument (JPY pairs have different pip)
            pip_value = 0.01 if instrument and 'JPY' in instrument else 0.0001
            atr_pips = atr / pip_value

            if atr_pips < self.config.min_atr_pips:
                logger.info(f"🔕 Trade blocked: Low volatility (ATR={atr_pips:.1f} pips < {self.config.min_atr_pips} min)")
                return TradeSignal(
                    trade=False,
                    direction=None,
                    size=0.0,
                    confidence=0.0,
                    reason=f"Low volatility: ATR={atr_pips:.1f} pips (min {self.config.min_atr_pips})",
                    regime=None,
                    regime_confidence=0.0,
                    tcn_direction=None,
                    tcn_probability=0.5,
                    ridge_confidence=0.0,
                    xgb_momentum=0.0,
                    xgb_acceleration=False,
                    rf_drawdown_pips=0.0,
                    rf_streak_prob=0.0,
                    metadata={'atr_pips': atr_pips, 'intel_data': intel_data},
                )
            else:
                intel_data['atr_pips'] = atr_pips

        # Ensure normalized features exist for volatility + gate models
        required_groups = ['direction', 'momentum', 'risk', 'confidence', 'volatility']
        required_features = set()
        for group in required_groups:
            required_features.update(get_normalized_feature_names().get(group, []))
        if any(fname not in df.columns for fname in required_features):
            df = compute_normalized_features(df)

        # === TCN VOLATILITY REGIME FILTER ===
        # TCN predicts volatility regime: 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
        # Instead of returning early with zeroed gate values, we set a flag
        # and continue computing all gates for full diagnostic visibility.
        volatility_regime = None
        volatility_regime_name = None
        volatility_regime_confidence = 0.0
        volatility_gate_passed = False
        volatility_block_reason = None

        VOL_REGIME_NAMES = {0: 'LOW', 1: 'NORMAL', 2: 'HIGH', 3: 'EXTREME'}

        if self.config.use_tcn_volatility_filter:
            if self.tcn_volatility_model is None:
                # TCN not available
                if self.config.tcn_required:
                    logger.warning("⛔ TCN Volatility filter required but not loaded — will block trade")
                    volatility_block_reason = "TCN Volatility filter required but not loaded"
                else:
                    # Graceful degradation: continue without volatility gate
                    logger.info("ℹ TCN Volatility model not loaded — skipping volatility gate (tcn_required=False)")
                    volatility_gate_passed = True
            else:
                # Get TCN volatility regime prediction
                try:
                    tcn_features = self._extract_tcn_volatility_features(df)
                    vol_pred = self.tcn_volatility_model.predict(tcn_features)

                    if 'volatility_regime' in vol_pred:
                        volatility_regime = vol_pred['volatility_regime']
                        volatility_regime_name = vol_pred.get('volatility_regime_name')
                        volatility_regime_confidence = vol_pred.get(
                            'regime_confidence', vol_pred.get('confidence', 0.0)
                        )
                    elif 'regime' in vol_pred:
                        volatility_regime = vol_pred['regime']
                        volatility_regime_name = vol_pred.get('regime_name')
                        volatility_regime_confidence = vol_pred.get('confidence', 0.0)
                    else:
                        raise KeyError("volatility_regime")

                    # Check if regime meets minimum threshold
                    volatility_gate_passed = volatility_regime >= self.config.min_volatility_regime

                    logger.info(
                        f"🌡️ Volatility Regime: {volatility_regime_name} ({volatility_regime}) "
                        f"conf={volatility_regime_confidence:.1%} "
                        f"(min={self.config.min_volatility_regime}={VOL_REGIME_NAMES.get(self.config.min_volatility_regime, '?')})"
                    )

                    if not volatility_gate_passed:
                        min_regime_name = VOL_REGIME_NAMES.get(self.config.min_volatility_regime, '?')
                        logger.info(
                            f"⏸️ Volatility too low ({volatility_regime_name} < {min_regime_name}) "
                            f"— continuing gate computation for diagnostics"
                        )
                        volatility_block_reason = f"Low volatility regime: {volatility_regime_name} (need {min_regime_name}+)"
                except Exception as e:
                    logger.warning(f"TCN volatility prediction failed: {e}")
                    if self.config.tcn_required:
                        volatility_block_reason = f"TCN Volatility prediction failed: {e}"
                    else:
                        # Graceful degradation: continue without volatility gate
                        logger.info("ℹ Continuing without volatility gate (tcn_required=False)")
                        volatility_gate_passed = True
        else:
            # Volatility filter disabled in config
            volatility_gate_passed = True

        # FIRST: Compute normalized features for instrument-agnostic inference
        # (already ensured above, but keep as defensive fallback)
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
        mc_uncertainty_blocked = False
        mc_uncertainty_reason = None
        mc_uncertainty_std = None  # Track the actual std value for metadata

        # === GET REGIME OR DIRECTION ===
        # Track if we're using technical fallback for logging
        using_technical_fallback = False
        
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

                    # === MONTE CARLO DROPOUT (if enabled) ===
                    if self.config.mc_dropout_samples > 0:
                        mc_prob, mc_std, mc_dir = self._predict_with_uncertainty(
                            tcn_features,
                            n_samples=self.config.mc_dropout_samples
                        )

                        # Flag high uncertainty (but continue to compute all gates for display)
                        if mc_std > self.config.max_uncertainty_std:
                            logger.info(f"🎲 High uncertainty detected: std={mc_std:.3f} > {self.config.max_uncertainty_std}")
                            mc_uncertainty_blocked = True
                            mc_uncertainty_reason = f"High model uncertainty: std={mc_std:.3f}"
                            mc_uncertainty_std = float(mc_std)
                        else:
                            mc_uncertainty_blocked = False
                            mc_uncertainty_reason = None

                        # Use MC Dropout results
                        tcn_probability = mc_prob
                        tcn_direction = mc_dir
                    else:
                        # Standard single prediction
                        tcn_pred = self.tcn.predict(tcn_features)
                        tcn_direction = tcn_pred['direction']
                        tcn_probability = tcn_pred['probability']

                        # === TCN + TRANSFORMER ENSEMBLE (if available) ===
                        if self.use_tcn_transformer_ensemble and self.tcn_model is not None:
                            try:
                                tcn_only_pred = self.tcn_model.predict(tcn_features)
                                tcn_only_direction = tcn_only_pred['direction']
                                tcn_only_probability = tcn_only_pred['probability']

                                # Get ensemble weights from config (default: 60% Transformer, 40% TCN)
                                transformer_weight = getattr(self.config, 'transformer_weight', 0.6)
                                tcn_weight = getattr(self.config, 'tcn_weight', 0.4)

                                # Weighted average of probabilities
                                ensemble_probability = (
                                    tcn_probability * transformer_weight +
                                    tcn_only_probability * tcn_weight
                                )

                                # Direction from ensemble probability
                                ensemble_direction = 1 if ensemble_probability > 0.5 else 0

                                # Check agreement
                                models_agree_ensemble = (tcn_direction == tcn_only_direction)

                                if models_agree_ensemble:
                                    logger.debug(
                                        f"🎯 TCN+Transformer AGREE: direction={ensemble_direction}, "
                                        f"prob={ensemble_probability:.3f} "
                                        f"(TF={tcn_probability:.3f}, TCN={tcn_only_probability:.3f})"
                                    )
                                else:
                                    # Models disagree - reduce confidence
                                    ensemble_probability = ensemble_probability * 0.85
                                    logger.debug(
                                        f"⚠️ TCN+Transformer DISAGREE: using weighted avg, "
                                        f"prob={ensemble_probability:.3f} "
                                        f"(TF={tcn_direction}/{tcn_probability:.3f}, "
                                        f"TCN={tcn_only_direction}/{tcn_only_probability:.3f})"
                                    )

                                # Update to ensemble values
                                tcn_probability = ensemble_probability
                                tcn_direction = ensemble_direction

                            except Exception as e:
                                logger.warning(f"TCN ensemble prediction failed: {e}, using Transformer only")
                else:
                    # No direction model available - use technical fallback
                    logger.warning("⚠️ No direction model loaded - falling back to technical indicators")
                    tech_direction, tech_confidence = self._predict_technical_fallback(df)
                    if tech_direction == "LONG":
                        tcn_direction = 1
                        tcn_probability = tech_confidence
                    elif tech_direction == "SHORT":
                        tcn_direction = 0
                        tcn_probability = 1.0 - tech_confidence
                    else:
                        # HOLD - no clear direction
                        tcn_direction = None
                        tcn_probability = 0.5
                    using_technical_fallback = True
                    
            except Exception as e:
                logger.warning(f"⚠️ Direction model prediction failed: {e} - falling back to technical indicators")
                # Use technical indicator fallback
                tech_direction, tech_confidence = self._predict_technical_fallback(df)
                if tech_direction == "LONG":
                    tcn_direction = 1
                    tcn_probability = tech_confidence
                elif tech_direction == "SHORT":
                    tcn_direction = 0
                    tcn_probability = 1.0 - tech_confidence
                else:
                    # HOLD - no clear direction
                    tcn_direction = None
                    tcn_probability = 0.5
                using_technical_fallback = True

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

        # === RL-OPTIMIZED GATE THRESHOLDS ===
        # If RL gate optimizer is loaded, use learned thresholds instead of fixed config
        rl_min_confidence = self.config.min_confidence
        rl_min_momentum = self.config.min_momentum
        rl_max_drawdown = self.config.max_drawdown_pct

        if self._rl_gates_loaded and self.rl_gate_optimizer is not None:
            try:
                # Extract features for regime detection
                rl_features = {}
                if 'atr_pct_14' in df.columns:
                    rl_features['volatility'] = df['atr_pct_14'].iloc[-1]
                if 'adx_14' in df.columns:
                    rl_features['adx'] = df['adx_14'].iloc[-1]
                if 'returns_2' in df.columns:
                    rl_features['momentum'] = df['returns_2'].iloc[-1]

                # Get RL-optimized thresholds based on current market regime
                adjusted = self.rl_gate_optimizer.get_adjusted_thresholds(
                    features=rl_features,
                    win_rate=self._recent_win_rate,
                    drawdown=self._current_drawdown
                )
                rl_min_confidence = adjusted['min_confidence']
                rl_min_momentum = adjusted['min_momentum']
                rl_max_drawdown = adjusted['max_drawdown_pct']

                logger.debug(f"RL Gate Thresholds: conf={rl_min_confidence:.1f}, mom={rl_min_momentum:.3f}, dd={rl_max_drawdown:.4f}")
            except Exception as e:
                logger.warning(f"RL gate threshold adjustment failed, using config defaults: {e}")

        # macOS subprocess path: RL gates loaded but optimizer lives in a worker
        if self._rl_gates_loaded and self.rl_gate_optimizer is None and self._rl_gates_proc is not None:
            try:
                rl_features = {}
                if 'atr_pct_14' in df.columns:
                    rl_features['volatility'] = df['atr_pct_14'].iloc[-1]
                if 'adx_14' in df.columns:
                    rl_features['adx'] = df['adx_14'].iloc[-1]
                if 'returns_2' in df.columns:
                    rl_features['momentum'] = df['returns_2'].iloc[-1]

                adjusted = self._get_rl_gate_thresholds_from_worker(
                    features=rl_features,
                    win_rate=self._recent_win_rate,
                    drawdown=self._current_drawdown,
                )
                if adjusted:
                    rl_min_confidence = adjusted.get('min_confidence', rl_min_confidence)
                    rl_min_momentum = adjusted.get('min_momentum', rl_min_momentum)
                    rl_max_drawdown = adjusted.get('max_drawdown_pct', rl_max_drawdown)
            except Exception as e:
                logger.warning(f"RL gate thresholds (subprocess) failed, using config defaults: {e}")

        # Initialize gate status tracking if not done during load
        if not hasattr(self, '_gate_status'):
            self._gate_status = {'xgboost': True, 'random_forest': True, 'ridge': True}
            self._gate_issues = {}

        # Track gate internals for diagnostics/transparency in CLI output.
        momentum_fresh = False
        momentum_override_by_tcn = False
        tcn_strong = False
        tcn_very_strong = False

        if self.config.permissive_mode:
            # GRACEFUL DEGRADATION: Use gates that are still usable

            # Confidence gate: use Ridge if available, else pass
            if self._gate_status.get('ridge', False) and self.ridge is not None:
                confidence_gate_passed = ridge_confidence >= rl_min_confidence
                logger.debug(f"Ridge gate ACTIVE: confidence={ridge_confidence:.1f}, threshold={rl_min_confidence:.1f}, passed={confidence_gate_passed}")
            else:
                confidence_gate_passed = True
                logger.warning(f"⚠️  Ridge gate BYPASSED (permissive mode): {self._gate_issues.get('ridge', 'not loaded')}")

            # Momentum gate: use XGBoost if available, else pass
            if self._gate_status.get('xgboost', False) and self.xgb is not None:
                momentum_fresh = xgb_momentum >= rl_min_momentum
                if self.config.require_fresh_or_accel:
                    momentum_gate_passed = momentum_fresh or xgb_acceleration
                else:
                    momentum_gate_passed = True
                logger.debug(f"XGBoost gate ACTIVE: momentum={xgb_momentum:.3f}, threshold={rl_min_momentum:.3f}, passed={momentum_gate_passed}")
            else:
                momentum_gate_passed = True
                logger.warning(f"⚠️  XGBoost gate BYPASSED (permissive mode): {self._gate_issues.get('xgboost', 'not loaded')}")

            # Risk gate: use RF if available AND not explicitly bypassed
            if self.config.bypass_risk_gate_in_permissive:
                risk_gate_passed = True
                logger.warning("⚠️  Risk gate BYPASSED (permissive mode): bypass_risk_gate_in_permissive=True")
            elif self._gate_status.get('random_forest', False) and self.rf is not None:
                risk_gate_passed = (
                    rf_drawdown_pct <= rl_max_drawdown and
                    rf_streak_prob <= self.config.max_streak_prob
                )
                logger.debug(f"RF gate ACTIVE: drawdown={rf_drawdown_pct:.4f}, threshold={rl_max_drawdown:.4f}, passed={risk_gate_passed}")
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
            confidence_gate_passed = effective_confidence >= rl_min_confidence

            # MOMENTUM GATE: Pass if any of these are true:
            # 1. XGBoost momentum is fresh (above threshold)
            # 2. XGBoost detects acceleration
            # 3. Optional legacy behavior: TCN strong can override stale momentum
            momentum_fresh = xgb_momentum >= rl_min_momentum
            if self.config.require_fresh_or_accel:
                momentum_gate_passed = momentum_fresh or xgb_acceleration
            else:
                momentum_gate_passed = momentum_fresh or xgb_acceleration or tcn_strong
                momentum_override_by_tcn = (not (momentum_fresh or xgb_acceleration)) and tcn_strong

            # RISK GATE: Slightly more lenient when TCN is confident (but not too much)
            # Tightened multipliers to prevent excessive risk in high-drawdown scenarios
            if tcn_very_strong:
                # Very confident TCN - relax risk thresholds modestly
                risk_gate_passed = (
                    rf_drawdown_pct <= rl_max_drawdown * 1.3 and
                    rf_streak_prob <= self.config.max_streak_prob * 1.3
                )
            elif tcn_strong:
                # Strong TCN - relax risk thresholds slightly
                risk_gate_passed = (
                    rf_drawdown_pct <= rl_max_drawdown * 1.15 and
                    rf_streak_prob <= self.config.max_streak_prob * 1.15
                )
            else:
                # Weak TCN - use strict thresholds
                risk_gate_passed = (
                    rf_drawdown_pct <= rl_max_drawdown and
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

        # === TRANSFORMER CONFIDENCE GATE (direction confidence) ===
        # Require reasonable confidence in direction prediction.
        # Prefer config-driven threshold so this can be tuned without code changes.
        # Example: min_tcn_probability=0.60 means require >=60% (or <=40%).
        min_dir_prob = float(getattr(self.config, 'min_tcn_probability', INFERENCE_DEFAULTS['min_tcn_probability']))
        min_dir_prob = max(0.5, min(0.99, min_dir_prob))
        direction_confidence_gate_passed = (
            tcn_probability >= min_dir_prob or tcn_probability <= (1.0 - min_dir_prob)
        )

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
        rsi_low_threshold = float(getattr(self.config, "rsi_extreme_low_threshold", 10.0))
        rsi_high_threshold = float(getattr(self.config, "rsi_extreme_high_threshold", 90.0))
        adx_trend_threshold = float(getattr(self.config, "adx_trend_contra_threshold", 35.0))

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
        # Optional heuristic veto (profile-configurable).
        rsi_gate_enabled = bool(getattr(self.config, "use_rsi_extreme_gate", True))
        rsi_gate_passed = True
        rsi_reason = None
        rsi_val = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50.0

        if rsi_gate_enabled:
            # Extreme oversold: block LONG (don't catch falling knife)
            # Extreme overbought: block SHORT (don't short a rocket)
            # Also block at absolute extremes (RSI=100 or RSI=0) regardless of direction
            if rsi_val <= 2.0 or rsi_val >= 98.0:
                # Absolute extreme — data quality issue or extreme conditions, block all
                rsi_gate_passed = False
                rsi_reason = f"rsi_extreme({f'{rsi_val:.1f}'})"
            elif rsi_val < rsi_low_threshold and gate_check_direction == 'long':
                rsi_gate_passed = False
                rsi_reason = f"rsi_extreme_low({rsi_val:.1f})"
            elif rsi_val > rsi_high_threshold and gate_check_direction == 'short':
                rsi_gate_passed = False
                rsi_reason = f"rsi_extreme_high({rsi_val:.1f})"

        # === TREND CONTRADICTION GATE ===
        # Optional heuristic veto (profile-configurable).
        trend_gate_enabled = bool(getattr(self.config, "use_trend_contra_gate", True))
        trend_gate_passed = True
        trend_reason = None
        adx_val = df['adx'].iloc[-1] if 'adx' in df.columns else 0.0

        if trend_gate_enabled and adx_val > adx_trend_threshold:  # Strong trend (threshold may be dynamic)
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

                # [2026-02-14] Feature alignment fix: Load and apply feature indices for meta-labeler.
                # Try pair-specific indices first, then generic.
                try:
                    pair_name = getattr(self, 'instrument', None) or getattr(self, '_current_instrument', None)
                    feature_indices = None

                    candidate_dirs: List[Path] = []
                    if pair_name:
                        candidate_dirs.append(self.model_dir / pair_name)
                    candidate_dirs.append(self.model_dir)

                    for model_dir in candidate_dirs:
                        loaded = load_saved_feature_indices(str(model_dir))
                        if loaded is not None:
                            feature_indices = loaded
                            break

                    if feature_indices is not None:
                        # Persist indices into meta-labeler to keep its internal
                        # feature-construction path consistent with training.
                        if getattr(self.meta_labeler, "primary_selected_indices", None) is None:
                            self.meta_labeler.set_primary_feature_indices(np.asarray(feature_indices))

                        # Also proactively align features here to avoid dimension drift
                        # when primary_selected_indices is absent in older artifacts.
                        idx = np.asarray(feature_indices, dtype=np.int64).reshape(-1)
                        if idx.size > 0:
                            max_idx = int(np.max(idx))
                            cur_n = int(meta_features.shape[-1])
                            if max_idx < cur_n and cur_n >= idx.size:
                                meta_features = meta_features[..., idx]
                                logger.debug(
                                    "Feature alignment applied for meta-labeler: %d -> %d features",
                                    cur_n,
                                    meta_features.shape[-1],
                                )
                except Exception as align_err:
                    # Log but continue - meta gate is fail-open on prediction errors.
                    logger.debug(f"Feature alignment skipped for meta-labeler: {align_err}")

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
                meta_reason = f"prediction_error({e})"
        else:
            # No meta-labeler loaded - pass by default
            meta_gate_passed = True
            meta_confidence = 0.0

        if self.use_regime:
            # All gates for regime mode
            all_gates_passed = (
                regime_gate_passed and
                direction_str is not None and
                not mc_uncertainty_blocked and  # MC Dropout uncertainty gate
                volatility_gate_passed and  # TCN volatility filter (replaces old tcn_probability_gate)
                direction_confidence_gate_passed and
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
                not mc_uncertainty_blocked and  # MC Dropout uncertainty gate
                volatility_gate_passed and  # TCN volatility filter (replaces old tcn_probability_gate)
                direction_confidence_gate_passed and
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
            if mc_uncertainty_blocked:
                reasons.append(mc_uncertainty_reason)
            if not volatility_gate_passed:
                if volatility_block_reason:
                    reasons.append(volatility_block_reason)
                else:
                    reasons.append(f"low_volatility_regime({volatility_regime_name or 'N/A'})")
            if not direction_confidence_gate_passed:
                min_dir_prob = float(getattr(self.config, 'min_tcn_probability', INFERENCE_DEFAULTS['min_tcn_probability']))
                min_dir_prob = max(0.5, min(0.99, min_dir_prob))
                reasons.append(f"weak_direction_conf({tcn_probability:.2f}<{min_dir_prob:.2f})")
            if self.use_regime:
                if not regime_gate_passed:
                    reasons.append("regime=CHOP (skip)")
                if direction_str is None and regime != 'chop':
                    reasons.append("no_direction")
            else:
                if tcn_direction is None:
                    reasons.append("no_direction")
            if not confidence_gate_passed:
                reasons.append(f"low_confidence({ridge_confidence:.0f}<{self.config.min_confidence})")
            if not momentum_gate_passed:
                reasons.append(
                    f"dead_momentum({xgb_momentum:.2f}<{rl_min_momentum:.2f},accel={str(xgb_acceleration).lower()})"
                )
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
            # Use DIRECTION features (same as RL sizer was trained on)
            rl_features = None
            if self.rl_sizer is not None:
                try:
                    rl_features = self._extract_rl_sizer_features(df)
                except Exception:
                    pass

            size = self._calculate_position_size(
                rf_drawdown_pips,
                equity,
                instrument=getattr(self, '_current_instrument', None),
                current_price=float(df['close'].iloc[-1]) if 'close' in df.columns else None,
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

        # === RL EXIT TIMING ===
        # If RL exit optimizer is loaded, get exit suggestion for ongoing trades
        rl_exit_suggestion = None
        rl_exit_confidence = 0.0
        if self._rl_exits_loaded and self.rl_exit_optimizer is not None and all_gates_passed:
            try:
                # Build observation for exit decision
                # For new trades, PnL=0 and bars=0; for ongoing trades, caller must provide actual values
                unrealized_pnl_pips = 0.0  # Placeholder - real PnL comes from check_open_position_exit()
                bars_in_trade = 0  # Fresh trade entry
                current_momentum = xgb_momentum
                current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001

                action, confidence = self.rl_exit_optimizer.get_exit_decision(
                    unrealized_pnl_pips=unrealized_pnl_pips,
                    bars_in_trade=bars_in_trade,
                    momentum=current_momentum,
                    atr=current_atr,
                )
                action_map = {0: 'hold', 1: 'exit_profit', 2: 'exit_loss'}
                rl_exit_suggestion = action_map.get(action, 'hold')
                rl_exit_confidence = confidence
                logger.debug(f"RL Exit Suggestion: {rl_exit_suggestion} (confidence={rl_exit_confidence:.2f})")
            except Exception as e:
                logger.warning(f"RL exit decision failed: {e}")
        elif self._rl_exits_loaded and self.rl_exit_optimizer is None and self._rl_exits_proc is not None and all_gates_passed:
            try:
                unrealized_pnl_pips = 0.0
                bars_in_trade = 0
                current_momentum = xgb_momentum
                current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001

                decision = self._get_rl_exit_decision_from_worker(
                    unrealized_pnl_pips=unrealized_pnl_pips,
                    bars_in_trade=bars_in_trade,
                    momentum=current_momentum,
                    atr=current_atr,
                )
                if decision is not None:
                    action, confidence = decision
                    action_map = {0: 'hold', 1: 'exit_profit', 2: 'exit_loss'}
                    rl_exit_suggestion = action_map.get(action, 'hold')
                    rl_exit_confidence = confidence
            except Exception as e:
                logger.warning(f"RL exit decision (subprocess) failed: {e}")

        return TradeSignal(
            trade=all_gates_passed,
            direction=direction_str,
            size=size,
            confidence=ridge_confidence,
            regime=regime,
            regime_confidence=regime_confidence,
            volatility_regime=volatility_regime,
            volatility_regime_name=volatility_regime_name,
            volatility_regime_confidence=volatility_regime_confidence,
            volatility_gate_passed=volatility_gate_passed,
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
            rl_thresholds_used=self._rl_gates_loaded,
            rl_exit_suggestion=rl_exit_suggestion,
            rl_exit_confidence=rl_exit_confidence,
            reason=reason,
            metadata={
                'intel_data': intel_data,
                'volatility_regime': volatility_regime_name,
                'using_technical_fallback': using_technical_fallback,
                'gate_thresholds': {
                    'min_tcn_probability': min_dir_prob,
                    'min_confidence': rl_min_confidence,
                    'min_momentum': rl_min_momentum,
                    'max_drawdown_pct': rl_max_drawdown,
                    'max_streak_prob': self.config.max_streak_prob,
                    'require_fresh_or_accel': self.config.require_fresh_or_accel,
                    'use_rsi_extreme_gate': rsi_gate_enabled,
                    'use_trend_contra_gate': trend_gate_enabled,
                    'rl_thresholds_applied': self._rl_gates_loaded,
                },
                'momentum_gate_details': {
                    'fresh': momentum_fresh,
                    'acceleration': bool(xgb_acceleration),
                    'tcn_override': momentum_override_by_tcn,
                },
                **(
                    {'mc_uncertainty_std': mc_uncertainty_std}
                    if mc_uncertainty_std is not None
                    else {}
                ),
                **(
                    {'meta_reason': meta_reason}
                    if meta_reason is not None
                    else {}
                ),
            },
        )

    def check_open_position_exit(
        self,
        df: pd.DataFrame,
        unrealized_pnl_pips: float,
        bars_in_trade: int,
        direction: str,  # 'long' or 'short'
        entry_price: float,
        stop_loss_pips: float = 15.0,
        take_profit_pips: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Check if an open position should be exited using RL Exit optimizer.

        Call this method periodically (e.g., each bar) to get exit recommendations
        for open positions. The RL model learns optimal exit timing based on:
        - Current unrealized P/L
        - Time in trade
        - Market momentum
        - Volatility (ATR)

        Args:
            df: Current market data DataFrame (needs momentum, ATR columns)
            unrealized_pnl_pips: Current position P/L in pips
            bars_in_trade: Number of bars since trade entry
            direction: 'long' or 'short'
            entry_price: Original entry price
            stop_loss_pips: Stop loss distance in pips
            take_profit_pips: Take profit distance in pips

        Returns:
            Dict with:
                - should_exit: bool - Whether to exit now
                - action: str - 'hold', 'exit_profit', or 'exit_loss'
                - confidence: float - Model confidence in decision
                - reason: str - Human-readable reason
                - rl_available: bool - Whether RL model was used
        """
        result = {
            'should_exit': False,
            'action': 'hold',
            'confidence': 0.5,
            'reason': '',
            'rl_available': False,
            'pnl_pips': unrealized_pnl_pips,
            'bars_held': bars_in_trade,
        }

        # === FIXED STOP/TARGET CHECK (Always active as safety) ===
        if unrealized_pnl_pips <= -stop_loss_pips:
            result['should_exit'] = True
            result['action'] = 'exit_loss'
            result['confidence'] = 0.95
            result['reason'] = f"Stop loss hit: {unrealized_pnl_pips:.1f} pips"
            return result

        if unrealized_pnl_pips >= take_profit_pips:
            result['should_exit'] = True
            result['action'] = 'exit_profit'
            result['confidence'] = 0.95
            result['reason'] = f"Take profit hit: {unrealized_pnl_pips:.1f} pips"
            return result

        # === RL EXIT TIMING ===
        if self._rl_exits_loaded and self.rl_exit_optimizer is not None:
            try:
                # Extract features
                current_momentum = 0.0
                if 'momentum_5' in df.columns:
                    current_momentum = df['momentum_5'].iloc[-1]
                elif 'xgb_momentum' in df.columns:
                    current_momentum = df['xgb_momentum'].iloc[-1]

                current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001

                # Get RL decision
                action, confidence = self.rl_exit_optimizer.get_exit_decision(
                    unrealized_pnl_pips=unrealized_pnl_pips,
                    bars_in_trade=bars_in_trade,
                    momentum=current_momentum,
                    atr=current_atr,
                )

                result['rl_available'] = True
                result['confidence'] = confidence

                action_map = {0: 'hold', 1: 'exit_profit', 2: 'exit_loss'}
                result['action'] = action_map.get(action, 'hold')

                if action == 1:  # EXIT_PROFIT
                    result['should_exit'] = True
                    result['reason'] = f"RL suggests take profit at {unrealized_pnl_pips:.1f} pips (conf={confidence:.2f})"
                elif action == 2:  # EXIT_LOSS
                    result['should_exit'] = True
                    result['reason'] = f"RL suggests cut loss at {unrealized_pnl_pips:.1f} pips (conf={confidence:.2f})"
                else:  # HOLD
                    result['reason'] = f"RL suggests hold (PnL={unrealized_pnl_pips:.1f}, bars={bars_in_trade})"

                logger.debug(f"RL Exit Check: action={result['action']}, conf={confidence:.2f}, pnl={unrealized_pnl_pips:.1f}")

            except Exception as e:
                logger.warning(f"RL exit check failed: {e}")
                result['reason'] = f"RL unavailable: {e}"
        elif self._rl_exits_loaded and self.rl_exit_optimizer is None and self._rl_exits_proc is not None:
            try:
                current_momentum = 0.0
                if 'momentum_5' in df.columns:
                    current_momentum = df['momentum_5'].iloc[-1]
                elif 'xgb_momentum' in df.columns:
                    current_momentum = df['xgb_momentum'].iloc[-1]

                current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001

                decision = self._get_rl_exit_decision_from_worker(
                    unrealized_pnl_pips=unrealized_pnl_pips,
                    bars_in_trade=bars_in_trade,
                    momentum=current_momentum,
                    atr=current_atr,
                )
                if decision is None:
                    result['reason'] = "RL subprocess unavailable - using heuristics"
                    return result

                action, confidence = decision
                result['rl_available'] = True
                result['confidence'] = confidence

                action_map = {0: 'hold', 1: 'exit_profit', 2: 'exit_loss'}
                result['action'] = action_map.get(action, 'hold')

                if action == 1:  # EXIT_PROFIT
                    result['should_exit'] = True
                    result['reason'] = (
                        f"RL suggests take profit at {unrealized_pnl_pips:.1f} pips (conf={confidence:.2f})"
                    )
                elif action == 2:  # EXIT_LOSS
                    result['should_exit'] = True
                    result['reason'] = (
                        f"RL suggests cut loss at {unrealized_pnl_pips:.1f} pips (conf={confidence:.2f})"
                    )
                else:  # HOLD
                    result['reason'] = f"RL suggests hold (PnL={unrealized_pnl_pips:.1f}, bars={bars_in_trade})"
            except Exception as e:
                logger.warning(f"RL exit check (subprocess) failed: {e}")
                result['reason'] = f"RL unavailable: {e}"
        else:
            # === HEURISTIC EXIT (Fallback when RL not loaded) ===
            result['reason'] = "RL not loaded - using heuristics"

            # Trail-stop logic: tighten stop as profit grows
            if unrealized_pnl_pips > take_profit_pips * 0.6:
                # In profit zone - check for reversal signals
                current_momentum = df.get('momentum_5', pd.Series([0])).iloc[-1]

                # Exit on momentum reversal in profit
                if direction == 'long' and current_momentum < -0.2 or direction == 'short' and current_momentum > 0.2:
                    result['should_exit'] = True
                    result['action'] = 'exit_profit'
                    result['confidence'] = 0.7
                    result['reason'] = f"Momentum reversal in profit zone ({unrealized_pnl_pips:.1f} pips)"

            # Time-decay: exit after too many bars if not profitable
            if bars_in_trade > 48 and unrealized_pnl_pips < take_profit_pips * 0.2:  # 48 hours in H1
                result['should_exit'] = True
                result['action'] = 'exit_loss' if unrealized_pnl_pips < 0 else 'exit_profit'
                result['confidence'] = 0.6
                result['reason'] = f"Time decay: {bars_in_trade} bars with {unrealized_pnl_pips:.1f} pips"

        return result

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

        # MC Dropout uncertainty (if applicable)
        mc_std = signal.metadata.get('mc_uncertainty_std') if signal.metadata else None
        if mc_std is not None:
            mc_status = "✗" if mc_std > self.config.max_uncertainty_std else "✓"
            gate_checks.append(f"MC Dropout: std={mc_std:.3f} (max={self.config.max_uncertainty_std}) {mc_status}")

        gate_thresholds = signal.metadata.get('gate_thresholds', {}) if signal.metadata else {}
        momentum_details = signal.metadata.get('momentum_gate_details', {}) if signal.metadata else {}
        if gate_thresholds:
            rl_note = "RL" if gate_thresholds.get('rl_thresholds_applied') else "CFG"
            gate_checks.append(
                "Thresholds: "
                f"dir={float(gate_thresholds.get('min_tcn_probability', 0.0)):.2f}, "
                f"conf={float(gate_thresholds.get('min_confidence', 0.0)):.1f}, "
                f"mom={float(gate_thresholds.get('min_momentum', 0.0)):.2f}, "
                f"dd={float(gate_thresholds.get('max_drawdown_pct', 0.0)):.2%} "
                f"[{rl_note}]"
            )

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
        min_conf = float(gate_thresholds.get('min_confidence', self.config.min_confidence)) if gate_thresholds else self.config.min_confidence
        gate_checks.append(f"Ridge: {signal.ridge_confidence:.0f}/100 (min={min_conf:.1f}) {status}")

        # XGBoost momentum
        status = "✓" if signal.momentum_gate_passed else "✗"
        min_mom = float(gate_thresholds.get('min_momentum', self.config.min_momentum)) if gate_thresholds else self.config.min_momentum
        fresh = bool(momentum_details.get('fresh', signal.xgb_momentum >= min_mom))
        override_note = ", override=tcn_strong" if bool(momentum_details.get('tcn_override', False)) else ""
        rule_str = "fresh_or_accel" if bool(gate_thresholds.get('require_fresh_or_accel', self.config.require_fresh_or_accel)) else "allow_tcn_override"
        accel_str = "accel=true" if signal.xgb_acceleration else "accel=false"
        gate_checks.append(
            f"XGBoost: momentum={signal.xgb_momentum:.2f} (min={min_mom:.2f}, fresh={str(fresh).lower()}, {accel_str}, rule={rule_str}{override_note}) {status}"
        )

        # RF risk (display as percentage - the actual gate unit)
        status = "✓" if signal.risk_gate_passed else "✗"
        rf_dd_pct = signal.rf_drawdown_pips / 10000 if signal.rf_drawdown_pips > 1.0 else signal.rf_drawdown_pips
        gate_checks.append(f"RF: drawdown={rf_dd_pct:.2%}, streak={signal.rf_streak_prob:.2f} {status}")

        # Meta-labeling gate (5th gate - predicts trade SUCCESS)
        if self._meta_labeler_loaded and self.meta_labeler is not None:
            # Check if meta prediction actually ran (confidence > 0 or gate explicitly failed)
            if signal.meta_confidence == 0.0 and signal.meta_gate_passed and 'prediction_error' in (signal.metadata.get('meta_reason', '') if signal.metadata else ''):
                gate_checks.append("Meta: prediction failed (skipped) ⚠️")
            else:
                status = "✓" if signal.meta_gate_passed else "✗"
                gate_checks.append(f"Meta: confidence={signal.meta_confidence:.2f} (threshold={self.config.min_meta_confidence}) {status}")
        elif self.config.enable_meta_labeling:
            gate_checks.append("Meta: not loaded (skipped)")

        # Position sizing transparency
        if self._last_size_details:
            details = self._last_size_details
            if details.get("mode") == "rl":
                pct = details.get("position_pct", 0.0) * 100
                size_usd = details.get("size_dollars", 0.0)
                size_lots = details.get("size_lots", 0.0)
                max_lots = details.get("max_lots", 0.0)
                sizing_mode = details.get("position_sizing_mode", "rl_hybrid")
                rl_lots = details.get("rl_lots", size_lots)
                min_risk_lots = details.get("min_risk_lots", 0.0)
                gate_checks.append(
                    f"Sizing: RL[{sizing_mode}] {pct:.1f}% equity (${size_usd:,.0f}) -> "
                    f"rl={rl_lots:.2f} / floor={min_risk_lots:.2f} / final={size_lots:.2f} lots "
                    f"(cap {max_lots:.2f})"
                )
            else:
                risk_pct = details.get("risk_pct", 0.0) * 100
                risk_amount = details.get("risk_amount", 0.0)
                dd_pips = details.get("expected_drawdown_pips", 0.0)
                size_lots = details.get("size_lots", 0.0)
                max_lots = details.get("max_lots", 0.0)
                gate_checks.append(
                    f"Sizing: {risk_pct:.1f}% risk (${risk_amount:,.0f}) / {dd_pips:.1f} pips -> {size_lots:.2f} lots (cap {max_lots:.2f})"
                )

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
    # Prefer pair-specific meta if it exists
    meta_path = None
    if instrument and instrument != "GENERIC":
        _pair_meta = model_dir / instrument / "modular_ensemble.meta.json"
        if _pair_meta.exists():
            meta_path = _pair_meta
    if meta_path is None:
        meta_path = model_dir / "modular_ensemble.meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
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
