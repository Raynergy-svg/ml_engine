"""
Scanner Execution Module.

Provides ExecutionManager for trade execution with:
- Daily trade limit enforcement
- ATR-based SL/TP calculation
- High probability TP bonus
- Kelly-based position sizing
- Live NAV fetching for compounding
- RL position sizer integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.brokers.base import BrokerClient
from src.brokers.registry import get_registry

logger = logging.getLogger(__name__)

# Lazy import for episodic memory (suppression gate)
try:
    from src.scanner.automation.episodic_memory import EpisodicMemory
    _EPISODIC_MEMORY_AVAILABLE = True
except ImportError:
    _EPISODIC_MEMORY_AVAILABLE = False

# NOTE: PIP_VALUES has been moved to InstrumentRegistry (src.brokers.registry)
# Use get_registry().get(symbol).pip_value to access pip values


def _get_pip_value(pair: str, fallback: float = 0.0001) -> float:
    """Get pip value for a pair from InstrumentRegistry.

    Args:
        pair: Trading pair symbol (e.g., "EUR_USD").
        fallback: Default pip value if pair not found.

    Returns:
        Pip value for the pair, or fallback if not found.
    """
    try:
        return get_registry().get(pair).pip_value
    except KeyError:
        return fallback


@dataclass
class ExecutionConfig:
    """Configuration for trade execution.

    Attributes:
        # Position sizing
        account_equity: Account balance (0 = fetch from OANDA)
        risk_per_trade_pct: Risk percentage per trade (0.05 = 5%)
        leverage: Account leverage (default 50:1)
        position_sizing_enabled: Enable dynamic position sizing
        aggressive_mode: Enable larger position sizes for compounding

        # SL/TP settings
        atr_sl_multiplier: ATR multiplier for stop loss (1.0 = 1x ATR)
        atr_tp_multiplier: ATR multiplier for take profit (1.5 = 1.5x ATR)
        min_sl_pips: Minimum stop loss in pips
        max_sl_pips: Maximum stop loss in pips (fixed for tight scalping)
        min_tp_pips: Minimum take profit in pips
        max_tp_pips: Maximum base take profit in pips

        # High probability bonus
        high_prob_threshold: Confidence threshold for TP bonus (0.65 = 65%)
        high_prob_tp_bonus: Extra pips added at high probability

        # Daily limits
        max_trades_per_day: Maximum trades allowed per day
    """
    # Position sizing
    account_equity: float = 0.0  # 0 = fetch from OANDA
    risk_per_trade_pct: float = 0.05  # 5% risk per trade
    leverage: int = 50
    position_sizing_enabled: bool = True
    aggressive_mode: bool = True
    regime_scaling_enabled: bool = True
    aggressive_scale_high_vol: float = 1.5
    aggressive_scale_extreme_vol: float = 1.75
    aggressive_min_meta_confidence: float = 0.52

    # SL/TP settings (tight scalping)
    atr_sl_multiplier: float = 1.0
    atr_tp_multiplier: float = 1.5
    min_sl_pips: float = 15.0
    max_sl_pips: float = 15.0  # Fixed for tight scalping
    min_tp_pips: float = 20.0
    max_tp_pips: float = 30.0

    # High probability bonus
    high_prob_threshold: float = 0.65
    high_prob_tp_bonus: float = 20.0

    # Daily limits
    max_trades_per_day: int = 30

    # Portfolio risk limits
    max_open_risk_pct: float = 0.15  # Max 15% of NAV at risk across all open trades

    # Execution recovery and fallback params (US-048)
    max_order_attempts: int = 2  # Original + N-1 retries on rejection
    rejection_downsize_factor: float = 0.5  # Multiply lots by this on order rejection
    high_slippage_alert_pips: float = 5.0  # Log observation when |slippage| exceeds
    fallback_tp_pips: float = 25.0  # TP fallback when ATR unavailable
    fallback_atr_pips: float = 15.0  # ATR fallback for trailing stop when ATR unavailable
    min_lot_size: float = 0.01  # Minimum order size in lots
    max_lot_size: float = 50.0  # Maximum order size in lots
    trailing_stop_breakeven_pct: float = 0.50  # Progress % to move SL to breakeven
    trailing_stop_lock_pct: float = 0.75  # Progress % to lock 50% of profit


@dataclass
class ExecutionResult:
    """Result of trade execution.

    Attributes:
        success: Whether execution succeeded
        trade_id: OANDA trade ID if filled
        fill_price: Actual fill price
        units: Position size in units
        lots: Position size in lots
        sl_price: Stop loss price
        tp_price: Take profit price
        error: Error message if failed
        regime_scale: Regime-based scaling factor applied
        regime_name: Volatility regime name
        aggressive_scaling_reason: Reason for aggressive scaling if applied
    """
    success: bool
    trade_id: Optional[str] = None
    fill_price: float = 0.0
    units: int = 0
    lots: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    risk_pct: float = 0.0
    confidence_level: str = "medium"
    error: Optional[str] = None
    # Regime-based scaling fields
    regime_scale: float = 1.0
    regime_name: str = "UNKNOWN"
    aggressive_scaling_reason: str = ""
    # Execution quality fields (Phase 7)
    fill_status: str = "UNKNOWN"  # FULL, PARTIAL, REJECTED, RETRIED
    slippage_pips: float = 0.0  # fill_price - expected_price in pips


class ExecutionManager:
    """Manages trade execution with position sizing and daily limits.

    Features:
    - Daily trade limit enforcement from OANDA
    - Live NAV fetching for proper compounding
    - ATR-based SL/TP calculation
    - High probability TP bonus
    - Kelly-based position sizing
    - RL position sizer integration
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        broker: Optional[BrokerClient] = None,
        oanda_client: Optional[Any] = None,
    ):
        """Initialize execution manager.

        Args:
            config: Execution configuration
            broker: BrokerClient instance (optional)
            oanda_client: OANDA API client (deprecated, use broker instead)
        """
        self.config = config or ExecutionConfig()
        # Support both new broker interface and legacy oanda_client
        self._broker = broker
        self._legacy_oanda = oanda_client

        # Lazy-loaded components
        self._position_sizer = None
        self._risk_manager = None
        self._rl_sizer = None
        self._memory_client = None

        # Cached account info
        self._cached_nav: Optional[float] = None
        self._trades_today: int = 0
        self._last_nav_fetch: float = 0.0

        # Phase 30 (US-183): Shared observation logger (lazy-init)
        self._observer = None

        # Phase 24 (US-148): Rolling slippage tracker per pair (last 20 trades)
        self._slippage_history: Dict[str, List[float]] = {}
        self._slippage_window: int = 20
        self._slippage_threshold_pips: float = 2.5  # Avg above this triggers reduction
        self._slippage_size_reduction: float = 0.15  # 15% size reduction

        # Phase 44 (US-279): Adaptive Exit Manager (Chandelier + partial profit + vol trail + time + confidence decay)
        self._ohlc_cache: Dict[str, Any] = {}  # pair -> DataFrame with close/high/low columns
        self._ohlc_cache_timestamp: Optional[float] = None  # Phase 55: freshness tracking
        self._adaptive_exit_manager = None
        try:
            from src.scanner.adaptive_exits import create_default_exit_manager
            self._adaptive_exit_manager = create_default_exit_manager()
            logger.info("Phase 44: Adaptive exit manager initialized")
        except Exception as e:
            logger.warning(f"Adaptive exit manager unavailable — exits will use basic SL/TP only: {e}")

        # Phase 45 (US-282): Adaptive Position Sizer (Kelly + Sigmoid + Drawdown + Regime)
        self._adaptive_position_sizer = None
        self._last_regime_name = "NORMAL"
        self._current_drawdown_pct = 0.0

        # Phase 45 (US-285): EWMA Correlation Engine for dynamic diversification
        self._ewma_correlation = None
        self._ewma_price_cache: Dict[str, float] = {}  # pair -> last seen price

        # Phase 47 (US-296): Session Detector for session-aware position sizing
        self._session_detector = None

        # Phase 47 (US-297): Expectancy Tracker for per-agent per-regime performance
        self._expectancy_tracker = None

        # DynamicRiskAllocator: P/L distribution-based risk multiplier (injected from engine.py)
        self._dynamic_risk_allocator = None

        # Position timeout manager: regime-dependent time-decay exits
        self._position_timeout = None
        try:
            from src.scanner.automation.position_timeout import PositionTimeoutManager
            self._position_timeout = PositionTimeoutManager()
            logger.info("PositionTimeoutManager initialized")
        except Exception as e:
            logger.debug(f"PositionTimeoutManager init deferred: {e}")

        # Phase 49 (US-306): Pre-execution gates from Phase 48 modules
        self._spread_filter = None
        try:
            from src.scanner.spread_filter import create_default_spread_filter
            self._spread_filter = create_default_spread_filter()
            logger.info("Phase 49: Spread filter initialized")
        except Exception as e:
            logger.debug(f"Phase 49: Spread filter init deferred: {e}")

        self._calendar_filter = None
        try:
            from src.scanner.economic_calendar import create_default_calendar_filter
            self._calendar_filter = create_default_calendar_filter()
            logger.info("Phase 49: Economic calendar filter initialized")
        except Exception as e:
            logger.debug(f"Phase 49: Calendar filter init deferred: {e}")

        self._strategy_fitness = None
        try:
            from src.scanner.strategy_fitness import create_default_fitness_detector
            self._strategy_fitness = create_default_fitness_detector()
            logger.info("Phase 49: Strategy fitness detector initialized")
        except Exception as e:
            logger.debug(f"Phase 49: Strategy fitness init deferred: {e}")

        # Phase 49 (US-307): Pair performance tracker for blacklisting + feedback
        self._pair_tracker = None
        try:
            from src.scanner.pair_blacklist import create_default_pair_tracker
            self._pair_tracker = create_default_pair_tracker()
            logger.info("Phase 49: Pair performance tracker initialized")
        except Exception as e:
            logger.debug(f"Phase 49: Pair tracker init deferred: {e}")

        # Phase 49 (US-308): Trade attribution engine for SHAP-lite feedback
        self._attribution_engine = None
        try:
            from src.scanner.trade_attribution import create_default_attribution_engine
            self._attribution_engine = create_default_attribution_engine()
            logger.info("Phase 49: Trade attribution engine initialized")
        except Exception as e:
            logger.debug(f"Phase 49: Attribution engine init deferred: {e}")

        # Regime-aware RL reward shaping (replaces flat PnL reward with regime-conditioned signal)
        self._regime_reward = None
        try:
            from src.scanner.automation.regime_reward import RegimeAwareReward
            self._regime_reward = RegimeAwareReward()
            logger.info("Regime-aware RL reward shaping initialized in ExecutionManager")
        except Exception as e:
            logger.debug(f"Regime reward init deferred: {e}")

        # Phase 49 (US-309): Walk-forward optimizer for parameter tuning
        self._walkforward_optimizer = None
        try:
            from src.scanner.walkforward_optimizer import create_default_walkforward_optimizer
            self._walkforward_optimizer = create_default_walkforward_optimizer()
            logger.info("Phase 49: Walk-forward optimizer initialized")
        except Exception as e:
            logger.debug(f"Phase 49: Walk-forward optimizer init deferred: {e}")

        # Phase 52 (US-323): Pattern Gate — journal-mined pattern confidence adjustments
        self._pattern_gate = None
        try:
            from src.scanner.pattern_gate import PatternGate
            self._pattern_gate = PatternGate()
            logger.info("Phase 52 (US-323): Pattern gate initialized")
        except Exception as e:
            logger.debug(f"Phase 52: Pattern gate init deferred: {e}")

        # Phase 52 (US-323): Setup Quality Filter — composite weak-setup rejection
        self._setup_quality_filter = None
        try:
            from src.scanner.setup_quality import SetupQualityFilter
            self._setup_quality_filter = SetupQualityFilter()
            logger.info("Phase 52 (US-323): Setup quality filter initialized")
        except Exception as e:
            logger.debug(f"Phase 52: Setup quality filter init deferred: {e}")

        # Phase 52 (US-323): Tranche Tracker — breakeven SL after partial profits
        self._tranche_tracker = None
        try:
            from src.scanner.tranche_tracker import TrancheTracker
            self._tranche_tracker = TrancheTracker()
            logger.info("Phase 52 (US-323): Tranche tracker initialized")
        except Exception as e:
            logger.debug(f"Phase 52: Tranche tracker init deferred: {e}")

        # Phase 52 (US-323): Walk-Forward Retrainer — periodic agent weight retraining
        self._walkforward_retrainer = None
        try:
            from src.scanner.walkforward_retrainer import WalkForwardRetrainer
            self._walkforward_retrainer = WalkForwardRetrainer()
            logger.info("Phase 52 (US-323): Walk-forward retrainer initialized")
        except Exception as e:
            logger.debug(f"Phase 52: Walk-forward retrainer init deferred: {e}")

        # Phase 52 (US-325): Adaptive R:R Calculator — regime/confidence-based TP targeting
        self._adaptive_rr = None
        try:
            from src.scanner.adaptive_rr import AdaptiveRRCalculator
            self._adaptive_rr = AdaptiveRRCalculator()
            logger.info("Phase 52 (US-325): Adaptive R:R calculator initialized")
        except Exception as e:
            logger.debug(f"Phase 52: Adaptive R:R init deferred: {e}")

        # Phase 52 (US-326): Drawdown Adapter — progressive trade restriction under drawdown
        self._drawdown_adapter = None
        try:
            from src.scanner.drawdown_adapter import DrawdownAdapter
            self._drawdown_adapter = DrawdownAdapter()
            logger.info("Phase 52 (US-326): Drawdown adapter initialized")
        except Exception as e:
            logger.debug(f"Phase 52: Drawdown adapter init deferred: {e}")

        # Phase 53 (US-328): Gate Attribution Engine — track gate decisions for effectiveness analysis
        self._gate_attribution = None
        try:
            from src.scanner.gate_attribution import GateAttributionEngine
            self._gate_attribution = GateAttributionEngine()
            logger.info("Phase 53 (US-328): Gate attribution engine initialized")
        except Exception as e:
            logger.debug(f"Phase 53: Gate attribution init deferred: {e}")

        # Phase 53 (US-330): Signal Freshness Guard — recompute RSI/ADX at execution
        self._signal_freshness = None
        try:
            from src.scanner.signal_freshness import SignalFreshnessGuard
            self._signal_freshness = SignalFreshnessGuard()
            logger.info("Phase 53 (US-330): Signal freshness guard initialized")
        except Exception as e:
            logger.debug(f"Phase 53: Signal freshness init deferred: {e}")

        # Phase 53 (US-329): Pair-Specific Bayesian Weights — per-pair agent tuning
        self._pair_bayesian_weights = None
        try:
            from src.scanner.pair_bayesian_weights import PairSpecificBayesianWeights
            self._pair_bayesian_weights = PairSpecificBayesianWeights()
            logger.info("Phase 53 (US-329): Pair-specific Bayesian weights initialized")
        except Exception as e:
            logger.debug(f"Phase 53: Pair Bayesian weights init deferred: {e}")

        # Phase 53 (US-332): WFE Stability Monitor — pause retraining on WFE decline
        self._wfe_monitor = None
        try:
            from src.scanner.wfe_monitor import WFEStabilityMonitor
            self._wfe_monitor = WFEStabilityMonitor()
            logger.info("Phase 53 (US-332): WFE stability monitor initialized")
        except Exception as e:
            logger.debug(f"Phase 53: WFE monitor init deferred: {e}")

        # Phase 54 (US-334): Dynamic Gate Threshold Optimizer — auto-tune gate thresholds
        self._gate_threshold_optimizer = None
        try:
            from src.scanner.gate_threshold_optimizer import GateThresholdOptimizer
            self._gate_threshold_optimizer = GateThresholdOptimizer()
            logger.info("Phase 54 (US-334): Gate threshold optimizer initialized")
        except Exception as e:
            logger.debug(f"Phase 54: Gate threshold optimizer init deferred: {e}")

        # Phase 54 (US-335): Feature Drift Detector — KS test + Page-Hinkley
        self._feature_drift_detector = None
        try:
            from src.scanner.feature_drift import FeatureDriftDetector
            self._feature_drift_detector = FeatureDriftDetector()
            logger.info("Phase 54 (US-335): Feature drift detector initialized")
        except Exception as e:
            logger.debug(f"Phase 54: Feature drift detector init deferred: {e}")

        # Phase 55 (US-344): Feature Health Monitor — PSI per feature with importance downweighting
        # Phase 56 (US-349): ExecutionQualityTracker — TCA fill feedback loop
        self._exec_quality_tracker = None
        try:
            from src.scanner.automation.execution_quality_tracker import ExecutionQualityTracker
            self._exec_quality_tracker = ExecutionQualityTracker()
            logger.info("Phase 56 (US-349): Execution quality tracker initialized")
        except Exception as _eqt_err:
            logger.debug(f"Phase 56: Execution quality tracker init deferred: {_eqt_err}")

        # Tier 6 (US-400): AuraQueryGate — Pre-trade co-cognition from Aura
        self._aura_gate = None
        try:
            from src.aura.bridge.query_gate import AuraQueryGate
            self._aura_gate = AuraQueryGate(timeout_sec=25.0)
            logger.info("Tier 6 (US-400): Aura co-cognition query gate initialized")
        except Exception as _aura_err:
            logger.debug(f"Tier 6: Aura gate init deferred: {_aura_err}")

        self._feature_health_monitor = None
        try:
            from src.scanner.feature_health import FeatureHealthMonitor
            self._feature_health_monitor = FeatureHealthMonitor()
            logger.info("Phase 55 (US-344): Feature health monitor initialized")
        except Exception as e:
            logger.debug(f"Phase 55: Feature health monitor init deferred: {e}")

        # US-013: Futures Margin Validation
        self._margin_checker = None
        try:
            from src.risk.margin import MarginChecker
            self._margin_checker = MarginChecker(max_margin_pct=0.80)
            logger.info("US-013: Futures margin checker initialized (max_margin_pct=0.80)")
        except Exception as e:
            logger.debug(f"US-013: Margin checker init deferred: {e}")

        # Phase 54 (US-336): Trade Cluster Analyzer — identify high/low win clusters
        self._trade_cluster_analyzer = None
        try:
            from src.scanner.trade_clusters import TradeClusterAnalyzer
            self._trade_cluster_analyzer = TradeClusterAnalyzer()
            logger.info("Phase 54 (US-336): Trade cluster analyzer initialized")
        except Exception as e:
            logger.debug(f"Phase 54: Trade cluster analyzer init deferred: {e}")

        # Phase 54 (US-337): Cluster Confidence Gate — boost/penalize based on cluster match
        self._cluster_gate = None
        try:
            from src.scanner.cluster_gate import ClusterConfidenceGate
            self._cluster_gate = ClusterConfidenceGate(
                cluster_analyzer=self._trade_cluster_analyzer
            )
            logger.info("Phase 54 (US-337): Cluster confidence gate initialized")
        except Exception as e:
            logger.debug(f"Phase 54: Cluster gate init deferred: {e}")

        # Phase 54 (US-338): Regime Confidence Scorer — scale sizing by regime certainty
        self._regime_confidence_scorer = None
        try:
            from src.scanner.regime_confidence import RegimeConfidenceScorer
            self._regime_confidence_scorer = RegimeConfidenceScorer()
            logger.info("Phase 54 (US-338): Regime confidence scorer initialized")
        except Exception as e:
            logger.debug(f"Phase 54: Regime confidence scorer init deferred: {e}")

        # Phase 50 (US-310): Pre-execution filter chain for extensible gate management
        self._filter_chain = None
        try:
            from src.scanner.execution_filters import FilterChain
            self._filter_chain = FilterChain()
            logger.info("Phase 50 (US-310): Execution filter chain initialized")
        except Exception as e:
            logger.debug(f"Phase 50: Execution filter chain init deferred: {e}")

        # Phase 81 (US-P81-002): PositionTimeoutManager reference — injected by Scanner._init_executor()
        self._position_timeout_mgr: Optional[Any] = None

        # Phase 79: Cached DynamicDrawdownManager (avoids fresh creation every trailing stop check)
        self._dynamic_drawdown_mgr: Optional[Any] = None
        try:
            from src.scanner.automation.dynamic_drawdown import DynamicDrawdownManager
            lock_pct = getattr(self.config, "trailing_stop_lock_pct", 0.75)
            be_pct = getattr(self.config, "trailing_stop_breakeven_pct", 0.50)
            self._dynamic_drawdown_mgr = DynamicDrawdownManager(
                normal_breakeven_pct=be_pct,
                normal_lock_pct=lock_pct,
            )
            logger.info("Phase 79: DynamicDrawdownManager cached (no per-call creation)")
        except Exception as e:
            logger.debug(f"Phase 79: DynamicDrawdownManager init deferred: {e}")

        # Phase 80 (US-P80-004): Equity HWM + drawdown tracking for DynamicDrawdownManager
        # get_dynamic_thresholds() reads _current_drawdown — previously always 0.0.
        self._equity_hwm: float = 0.0       # High-water mark of account NAV
        self._current_drawdown: float = 0.0  # Current drawdown fraction (0.0 = no drawdown)

        # Episodic suppression gate — blocks setups with repeated historical losses
        self._episodic_memory: Optional[Any] = None
        if _EPISODIC_MEMORY_AVAILABLE and getattr(self.config, 'enable_episodic_memory', True):
            try:
                self._episodic_memory = EpisodicMemory(max_episodes=getattr(self.config, 'episodic_memory_max_episodes', 500))
                logger.info("EpisodicMemory suppression gate active in ExecutionManager")
            except Exception as e:
                logger.warning(f"EpisodicMemory init failed in ExecutionManager: {e}")

        # Lazy-init attributes that were previously missing from __init__
        self._correlation_block_counts: Dict[str, int] = {}
        self._open_trade_pairs: set = set()
        self._fasttrack_stats: Dict[str, int] = {"wins": 0, "losses": 0, "total": 0}

    def _init_broker(self) -> bool:
        """Initialize broker client (lazy initialization).

        Returns:
            True if broker initialized successfully
        """
        if self._broker is not None:
            return True

        # Try legacy OANDA client first (backward compatibility)
        # Wrap raw OandaPracticeClient in OandaBroker so get_nav() etc. work
        if self._legacy_oanda is not None:
            try:
                from src.brokers.oanda import OandaBroker
                self._broker = OandaBroker(client=self._legacy_oanda)
                logger.info("ExecutionManager: Wrapped legacy OANDA client in OandaBroker")
            except Exception:
                self._broker = self._legacy_oanda  # fallback to raw client
            return True

        # Lazy-init OandaBroker from environment
        try:
            from src.brokers.oanda import OandaBroker
            self._broker = OandaBroker.from_env()
            logger.info("ExecutionManager: OandaBroker initialized from environment")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize broker: {e}")
            return False

    @property
    def _oanda_base_url(self) -> str:
        """OANDA REST API base URL from environment (practice or live)."""
        return os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")

    def _retry_oanda(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute an OANDA API call with retry and circuit breaker.

        Args:
            func: Callable to execute.
            *args, **kwargs: Passed to func.

        Returns:
            Result from func.

        Raises:
            RuntimeError if circuit breaker is tripped.
            Last exception if all retries exhausted.
        """
        try:
            from src.scanner.automation.api_retry import retry_api_call, get_oanda_breaker
            return retry_api_call(
                func, *args,
                max_retries=3,
                backoff_base=1.0,
                circuit_breaker=get_oanda_breaker(),
                **kwargs,
            )
        except ImportError:
            # Fallback — direct call without retry
            return func(*args, **kwargs)

    def _get_observer(self):
        """Phase 30 (US-183): Shared observation logger instance (lazy-init)."""
        if self._observer is None:
            try:
                from .automation.observation_log import ObservationLog
                self._observer = ObservationLog()
            except ImportError:
                pass
        return self._observer

    def _init_position_sizer(self) -> None:
        """Initialize position sizer with regime-aware scaling."""
        if self._position_sizer is not None:
            return

        try:
            from src.risk.position_sizing import (
                DynamicPositionSizer, PositionSizingConfig,
                create_regime_aware_position_sizer
            )

            if self.config.aggressive_mode:
                # Use regime-aware position sizer with aggressive scaling
                # This is the "highest-ROI toggle in the entire system"
                self._position_sizer = create_regime_aware_position_sizer(
                    aggressive_scale_high_vol=float(self.config.aggressive_scale_high_vol),
                    aggressive_scale_extreme_vol=float(self.config.aggressive_scale_extreme_vol),
                    aggressive_min_meta_confidence=float(self.config.aggressive_min_meta_confidence),
                    regime_scaling_enabled=bool(self.config.regime_scaling_enabled),
                )
                logger.info("✓ Regime-aware position sizer initialized (aggressive mode)")
            else:
                config = PositionSizingConfig(
                    risk_per_trade_pct=self.config.risk_per_trade_pct,
                    min_confidence_threshold=0.5,
                )
                self._position_sizer = DynamicPositionSizer(config)
        except ImportError:
            logger.debug("DynamicPositionSizer not available")

    def _init_adaptive_position_sizer(self) -> None:
        """Phase 45 (US-282): Initialize adaptive position sizer with Kelly+Sigmoid+Drawdown+Regime."""
        if self._adaptive_position_sizer is not None:
            return
        try:
            from pathlib import Path
            from src.scanner.adaptive_position_sizing import (
                create_default_position_sizer as create_default_adaptive_sizer,
                create_aggressive_position_sizer as create_aggressive_adaptive_sizer,
            )
            if self.config.aggressive_mode:
                self._adaptive_position_sizer = create_aggressive_adaptive_sizer()
            else:
                self._adaptive_position_sizer = create_default_adaptive_sizer()

            # Load persisted state if available
            _state_path = Path("trained_data/adaptive_sizer_state.json")
            if _state_path.exists():
                try:
                    self._adaptive_position_sizer.load_state(str(_state_path))
                    logger.info("Phase 45: Loaded adaptive position sizer state")
                except Exception as e:
                    logger.warning(f"Phase 45: Could not load adaptive sizer state: {e}")

            logger.info("Phase 45: Adaptive position sizer initialized")
        except Exception as e:
            logger.debug(f"Phase 45: Adaptive position sizer init deferred: {e}")

    def _init_ewma_correlation(self) -> None:
        """Phase 45 (US-285): Initialize EWMA correlation engine."""
        if self._ewma_correlation is not None:
            return
        try:
            from src.risk.ewma_correlation import (
                create_default_ewma_engine,
            )
            # Get tracked pairs from config
            _pairs = getattr(self.config, 'pairs', None)
            if _pairs and isinstance(_pairs, (list, tuple)):
                _pair_list = list(_pairs)
            else:
                _pair_list = None

            self._ewma_correlation = create_default_ewma_engine(pairs=_pair_list)

            # Load persisted state
            _state_path = "trained_data/ewma_correlation_state.json"
            import os
            if os.path.exists(_state_path):
                try:
                    self._ewma_correlation.load_state(_state_path)
                    logger.info("Phase 45: Loaded EWMA correlation state")
                except Exception as e:
                    logger.warning(f"Phase 45: Could not load EWMA state: {e}")

            logger.info("Phase 45: EWMA correlation engine initialized")
        except Exception as e:
            logger.debug(f"Phase 45: EWMA correlation init deferred: {e}")

    def _init_risk_manager(self) -> None:
        """Initialize risk manager."""
        if self._risk_manager is not None:
            return

        try:
            from src.risk.risk_management import (
                ConfidenceBasedRiskManager, RiskManagementConfig
            )
            self._risk_manager = ConfidenceBasedRiskManager(RiskManagementConfig())
        except ImportError:
            logger.debug("ConfidenceBasedRiskManager not available")

    def _init_memory_client(self) -> None:
        """Initialize memory client for trade logging."""
        if self._memory_client is not None:
            return

        try:
            from memory_client import MLEngineMemory
            self._memory_client = MLEngineMemory()
        except ImportError:
            logger.debug("MemoryClient not available")

    def _init_session_detector(self) -> None:
        """Phase 47 (US-296): Initialize session detector for session-aware position sizing."""
        if self._session_detector is not None:
            return
        try:
            from src.scanner.session_detector import SessionDetector
            self._session_detector = SessionDetector()
            logger.info("Phase 47: Session detector initialized")
        except Exception as e:
            logger.debug(f"Phase 47: Session detector init deferred: {e}")

    def _init_expectancy_tracker(self) -> None:
        """Phase 47 (US-297): Initialize expectancy tracker for per-agent per-regime performance."""
        if self._expectancy_tracker is not None:
            return
        try:
            from src.scanner.expectancy_tracker import (
                create_default_expectancy_tracker,
            )
            self._expectancy_tracker = create_default_expectancy_tracker()
            self._expectancy_tracker.load_state()
            logger.info("Phase 47: Expectancy tracker initialized")
        except Exception as e:
            logger.debug(f"Phase 47: Expectancy tracker init deferred: {e}")

    def _update_ewma_correlation(self, pair: str, current_price: float, previous_price: float) -> None:
        """Phase 45 (US-285): Update EWMA correlation with latest return.

        Args:
            pair: Currency pair name
            current_price: Current price
            previous_price: Previous price
        """
        if self._ewma_correlation is None:
            self._init_ewma_correlation()
        if self._ewma_correlation is None:
            return

        try:
            import numpy as np
            if previous_price > 0 and current_price > 0:
                log_return = float(np.log(current_price / previous_price))
                self._ewma_correlation.update({pair: log_return})
        except Exception as e:
            logger.debug(f"Phase 45: EWMA correlation update failed: {e}")

    def fetch_live_nav(self) -> Optional[float]:
        """Fetch live NAV from broker for proper compounding.

        Returns:
            Account NAV or None if unavailable
        """
        if not self._init_broker():
            return None

        try:
            nav = self._broker.get_nav()
            if nav > 0:
                self._cached_nav = nav
                logger.debug(f"Fetched live NAV: ${nav:,.2f}")
                return nav
        except RuntimeError as e:
            # Circuit breaker tripped
            logger.warning(f"Broker circuit breaker open: {e}")
        except Exception as e:
            logger.debug(f"Could not fetch live NAV: {e}")
        return self._cached_nav

    def fetch_trades_today(self) -> int:
        """Fetch count of currently open trades from broker.

        Returns:
            Number of open trades (0 if unable to fetch).
            Uses OPEN state filter to avoid counting historical closed trades.
        """
        if not self._init_broker():
            return 0

        try:
            # Only count OPEN positions — get_trades(state="ALL") returns all history
            # which inflates the count and blocks new trades via daily limit.
            trades = self._broker.get_open_positions()
            trades_today = len(trades) if trades else 0

            self._trades_today = trades_today
            return trades_today

        except Exception as e:
            logger.debug(f"Could not fetch open trades: {e}")
            return self._trades_today

    def get_account_status(self) -> Tuple[float, int, int]:
        """Get account NAV and trade limits.

        Returns:
            Tuple of (nav, trades_today, trades_remaining)
        """
        nav = self.fetch_live_nav() or self.config.account_equity or 10000.0
        trades_today = self.fetch_trades_today()
        trades_remaining = max(0, self.config.max_trades_per_day - trades_today)

        return nav, trades_today, trades_remaining

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        trades_today = self.fetch_trades_today()

        if trades_today >= self.config.max_trades_per_day:
            return False, f"Daily limit reached ({trades_today}/{self.config.max_trades_per_day})"

        return True, f"OK ({trades_today}/{self.config.max_trades_per_day} trades today)"

    def _check_portfolio_risk_limit(self) -> Tuple[bool, str]:
        """Check if total open risk across all positions is within limit.

        Uses SL distance * position size to estimate risk per trade.
        Phase 45 (US-285): Uses EWMA correlation regime to dynamically
        reduce the limit during RISK_OFF conditions.

        Returns:
            Tuple of (within_limit, reason)
        """
        try:
            max_risk_pct = self._get_portfolio_risk_limit()
        except Exception:
            max_risk_pct = 0.15  # Fallback to base 15% limit
        if max_risk_pct <= 0:
            return True, "risk limit disabled"

        try:
            statuses = self.monitor_open_trades(evaluate_exits=False)
            if not statuses:
                return True, "no open trades"

            nav = self.fetch_live_nav() or self.config.account_equity or 100000.0

            total_risk = 0.0
            for s in statuses:
                sl_dist = s.get("sl_dist_pips", 0)
                units = abs(s.get("units", 0))
                pip_val = _get_pip_value(s.get("pair", ""), 0.0001)
                # Risk = SL distance in price * units
                risk_amount = sl_dist * pip_val * units
                total_risk += risk_amount

            risk_pct = total_risk / nav if nav > 0 else 0.0

            if risk_pct >= max_risk_pct:
                return False, (
                    f"Portfolio risk limit ({risk_pct:.1%} >= {max_risk_pct:.0%}): "
                    f"${total_risk:,.0f} at risk on ${nav:,.0f} NAV"
                )
            return True, f"risk {risk_pct:.1%} of {max_risk_pct:.0%} limit"

        except Exception as e:
            logger.warning(f"Risk limit check failed (fail-safe BLOCK): {e}")
            return False, "risk check error — blocking trade (fail-safe)"

    def _check_correlation_exposure(self, pair: str) -> Tuple[bool, str]:
        """Check if opening a trade on this pair would breach correlation exposure limits.

        Uses correlation group config to determine if the pair belongs to a
        correlated group, then counts how many open trades already exist in
        that group.  If the count >= max_correlated_exposure, the trade is blocked.

        Returns:
            Tuple of (allowed, reason)
        """
        max_corr = getattr(self.config, "max_correlated_exposure", 2)
        if max_corr <= 0:
            return True, "correlation limit disabled"

        try:
            from src.training.correlation_group_config import get_correlation_group

            group = get_correlation_group(pair)
            if not group:
                return True, "no correlation group"

            # Get open trades (read-only — no exit side effects)
            statuses = self.monitor_open_trades(evaluate_exits=False)
            if not statuses:
                return True, "no open trades"

            # Count how many open trades are in the same correlation group
            corr_count = 0
            corr_pairs = []
            for s in statuses:
                open_pair = s.get("pair", "")
                open_group = get_correlation_group(open_pair)
                if open_group and open_group.name == group.name:
                    corr_count += 1
                    corr_pairs.append(open_pair)

            if corr_count >= max_corr:
                reason = (
                    f"Correlated pair blocked: {pair} in group '{group.name}' "
                    f"({corr_count}/{max_corr} already open: {', '.join(corr_pairs)})"
                )
                # Log to observations
                try:
                    # Phase 30 (US-183): Use shared observer
                    obs = self._get_observer()
                    obs.log(
                        category="correlation_block",
                        detail=reason,
                        data={
                            "pair": pair, "group": group.name,
                            "open_correlated": corr_pairs,
                            "max_allowed": max_corr,
                        },
                    )
                except Exception as e:
                    logger.debug(f"Execution correlation check skipped: {e}")
                return False, reason

            return True, f"correlation ok ({corr_count}/{max_corr} in {group.name})"

        except ImportError:
            logger.debug("correlation_group_config not available — skipping (no groups defined)")
            return True, "correlation check unavailable (no config)"
        except Exception as e:
            logger.warning(f"Correlation check failed (fail-safe BLOCK): {e}")
            return False, "correlation check error — blocking trade (fail-safe)"

    def _check_projected_portfolio_risk(
        self,
        pair: str,
        sl_pips: float,
        lots: float,
    ) -> Tuple[bool, str, float]:
        """Check if adding the proposed trade would breach portfolio risk limit.

        Unlike _check_portfolio_risk_limit() which only checks existing positions,
        this projects the new trade's risk INTO the total before deciding.
        Phase 45 (US-285): Uses EWMA correlation regime to dynamically
        reduce the limit during RISK_OFF conditions.

        Returns:
            Tuple of (within_limit, reason, max_safe_lots)
            max_safe_lots is the largest position that fits within the risk budget
            (0.0 if even minimum lot breaches the limit).
        """
        max_risk_pct = self._get_portfolio_risk_limit()
        if max_risk_pct <= 0:
            return True, "risk limit disabled", lots

        try:
            nav = self.fetch_live_nav() or self.config.account_equity or 100000.0
            max_risk_amount = nav * max_risk_pct

            # Calculate existing risk across open positions (read-only — no exit side effects)
            existing_risk = 0.0
            statuses = self.monitor_open_trades(evaluate_exits=False)
            if statuses:
                for s in statuses:
                    sl_dist = s.get("sl_dist_pips", 0)
                    units = abs(s.get("units", 0))
                    pip_val = _get_pip_value(s.get("pair", ""), 0.0001)
                    existing_risk += sl_dist * pip_val * units

            remaining_budget = max(0.0, max_risk_amount - existing_risk)

            # Calculate the new trade's projected risk
            pip_value = _get_pip_value(pair)
            new_trade_risk = sl_pips * pip_value * lots * 100_000  # lots → units

            projected_total = existing_risk + new_trade_risk
            projected_pct = projected_total / nav if nav > 0 else 0.0

            if projected_pct <= max_risk_pct:
                return True, f"projected risk {projected_pct:.1%} within {max_risk_pct:.0%}", lots

            # Over budget — calculate max lots that fit
            if remaining_budget > 0 and sl_pips > 0 and pip_value > 0:
                max_units = remaining_budget / (sl_pips * pip_value)
                max_safe_lots = round(max_units / 100_000, 2)
                max_safe_lots = max(max_safe_lots, 0.0)
            else:
                max_safe_lots = 0.0

            reason = (
                f"Projected risk {projected_pct:.1%} exceeds {max_risk_pct:.0%} limit "
                f"(existing=${existing_risk:,.0f} + new=${new_trade_risk:,.0f} = "
                f"${projected_total:,.0f} on ${nav:,.0f} NAV). "
                f"Max safe lots: {max_safe_lots:.2f}"
            )
            return False, reason, max_safe_lots

        except Exception as e:
            logger.warning(f"Projected risk check failed (fail-safe BLOCK): {e}")
            return False, "projected risk check error — blocking trade (fail-safe)", 0.0

    def _apply_diversification_adjustment(self, lots: float, pair: str) -> float:
        """Phase 45 (US-285): Apply EWMA diversification multiplier to position size.

        Args:
            lots: Base position size in lots
            pair: Currency pair name

        Returns:
            Adjusted position size in lots
        """
        if self._ewma_correlation is None:
            return lots

        try:
            # Get currently open trade pairs
            _open_pairs = []
            _open_pairs = list(self._open_trade_pairs)

            if not _open_pairs:
                return lots

            multiplier = self._ewma_correlation.diversification_multiplier(pair, _open_pairs)
            adjusted_lots = round(lots * multiplier, 2)

            if multiplier < 1.0:
                logger.info(
                    f"Phase 45: Diversification adjustment for {pair}: "
                    f"{lots} -> {adjusted_lots} lots (mult={multiplier:.3f}, "
                    f"portfolio={_open_pairs})"
                )

            return max(adjusted_lots, self.config.min_lot_size)
        except Exception as e:
            logger.debug(f"Phase 45: Diversification adjustment failed: {e}")
            return lots

    def _get_portfolio_risk_limit(self) -> float:
        """Phase 45 (US-285): Get portfolio risk limit adjusted for correlation regime.

        Returns:
            Portfolio risk limit as fraction (e.g., 0.15 for 15%)
        """
        base_limit = getattr(self.config, "max_open_risk_pct", 0.15)  # from config, default 15%

        if self._ewma_correlation is None:
            try:
                self._init_ewma_correlation()
            except Exception as e:
                logger.debug(f"Phase 45: Correlation init failed in risk limit: {e}")
                return base_limit

        if self._ewma_correlation is not None:
            try:
                regime = self._ewma_correlation.detect_correlation_regime()
                if regime == "RISK_OFF":
                    adjusted = base_limit * 0.85  # Soften: was 0.67 (10%), now 0.85 (25.5%)
                    logger.info(f"Phase 45: RISK_OFF correlation regime — portfolio risk limit reduced to {adjusted:.1%}")
                    return adjusted
            except Exception as e:
                logger.debug(f"Phase 45: Correlation regime detection failed: {e}")

        return base_limit

    def _check_spread_conditions(self, pair: str) -> Tuple[bool, float, str]:
        """Check if current bid-ask spread is acceptable for entry.

        Fetches live pricing from broker and compares spread to max_spread_pips.

        Args:
            pair: Instrument name

        Returns:
            Tuple of (spread_ok, spread_pips, reason)
        """
        max_spread = float(getattr(self.config, "max_spread_pips", 3.0))
        pip_value = _get_pip_value(pair)

        if not self._init_broker():
            return True, 0.0, "spread check skipped (no broker client)"

        try:
            # Create instrument for broker interface
            from src.brokers.instrument import Instrument
            instrument = Instrument.fx(
                symbol=pair,
                broker_symbol=pair,
                pip_value=pip_value,
            )

            quote = self._broker.get_price_quote(instrument)
            if not quote or "bid" not in quote or "ask" not in quote:
                return True, 0.0, "spread check skipped (no pricing data)"

            ask = float(quote.get("ask", 0))
            bid = float(quote.get("bid", 0))

            if ask > 0 and bid > 0 and pip_value > 0:
                spread_pips = (ask - bid) / pip_value
                spread_pips = round(spread_pips, 1)

                if spread_pips > max_spread:
                    reason = (
                        f"Spread too wide: {spread_pips:.1f} pips > "
                        f"{max_spread:.1f} max for {pair}"
                    )
                    logger.warning(reason)
                    return False, spread_pips, reason

                return True, spread_pips, f"spread {spread_pips:.1f} pips OK"

            return True, 0.0, "spread check skipped (incomplete pricing)"

        except Exception as e:
            logger.debug(f"Spread check failed for {pair}: {e}")
            return True, 0.0, f"spread check unavailable: {e}"

    def _calculate_base_tp_pips(self, atr: float, pip_value: float, confidence: float) -> float:
        """Calculate base take profit pips from ATR with high probability bonus.

        Args:
            atr: ATR value
            pip_value: Pip value for the pair
            confidence: Model confidence (0-1)

        Returns:
            Take profit in pips
        """
        if pip_value > 0 and atr > 0:
            base_tp = (atr * self.config.atr_tp_multiplier) / pip_value
        else:
            base_tp = self.config.fallback_tp_pips

        tp_pips = max(self.config.min_tp_pips, min(base_tp, self.config.max_tp_pips))

        # High probability TP bonus
        if confidence >= self.config.high_prob_threshold:
            tp_pips += self.config.high_prob_tp_bonus
            logger.debug(
                f"High probability ({confidence:.1%}) - "
                f"TP bonus +{self.config.high_prob_tp_bonus} pips → {tp_pips:.1f} pips"
            )

        return tp_pips

    def _apply_risk_manager(
        self, pair: str, confidence: float, sl_pips: float, tp_pips: float
    ) -> Tuple[float, float, str]:
        """Apply risk manager adjustments to SL/TP.

        Args:
            pair: Instrument name
            confidence: Model confidence
            sl_pips: Base stop loss pips
            tp_pips: Base take profit pips

        Returns:
            Tuple of (adjusted_sl_pips, adjusted_tp_pips, confidence_level)
        """
        if self._risk_manager is None:
            return sl_pips, tp_pips, "medium"

        try:
            risk_result = self._risk_manager.calculate_risk_levels(
                entry_price=1.0,
                raw_confidence=confidence,
                base_stop_loss_pips=sl_pips,
                instrument=pair,
            )
            if risk_result.is_valid:
                return risk_result.stop_loss_pips, risk_result.take_profit_pips, risk_result.confidence_level
        except Exception as e:
            logger.warning(f"Risk calculation failed: {e}")

        return sl_pips, tp_pips, "medium"

    def _calculate_lots_from_sizer(
        self, equity: float, sl_pips: float, pair: str, confidence: float
    ) -> Tuple[float, float, str]:
        """Calculate lots using position sizer.

        Args:
            equity: Account equity
            sl_pips: Stop loss pips
            pair: Instrument name
            confidence: Model confidence

        Returns:
            Tuple of (lots, risk_pct, confidence_level)
        """
        def _fallback_lots() -> Tuple[float, float, str]:
            risk_pct = float(self.config.risk_per_trade_pct)
            risk_amount = max(0.0, float(equity) * risk_pct)
            pip_value_usd = 7.5 if str(pair).upper().endswith("JPY") else 10.0
            denom = max(float(sl_pips), 1.0) * pip_value_usd
            lots = (risk_amount / denom) if denom > 0 else 0.0
            lots = round(min(max(lots, self.config.min_lot_size), self.config.max_lot_size), 2)
            return lots, risk_pct, "fallback"

        if self._position_sizer is None:
            return _fallback_lots()

        try:
            pos_result = self._position_sizer.calculate_position_size(
                account_equity=equity,
                stop_loss_pips=sl_pips,
                instrument=pair,
                raw_confidence=confidence,
            )
            if pos_result.is_valid:
                lots = pos_result.units / 100_000
                risk_pct = pos_result.risk_amount / equity if equity > 0 else 0
                if lots > 0:
                    return lots, risk_pct, pos_result.confidence_level
        except Exception as e:
            logger.warning(f"Position sizing failed: {e}")

        return _fallback_lots()
    
    def calculate_regime_aware_position_size(
        self,
        pair: str,
        confidence: float,
        atr: float,
        volatility_regime: int,
        meta_confidence: float,
        account_equity: Optional[float] = None,
    ) -> Tuple[float, float, float, float, str, float, str]:
        """Calculate position sizing with regime-based aggressive scaling.
        
        This is the key method for enabling aggressive volatility-regime sizing.
        When regime == HIGH/EXTREME and meta_confidence > threshold, positions
        are scaled UP to capitalize on high-quality signals.
        
        Args:
            pair: Instrument name
            confidence: Model confidence (0-1)
            atr: ATR value for SL calculation
            volatility_regime: Volatility regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)
            meta_confidence: Meta-labeler confidence (0-1)
            account_equity: Account equity (fetches from OANDA if None)
            
        Returns:
            Tuple of (lots, risk_pct, sl_pips, tp_pips, confidence_level, regime_scale, regime_name)
        """
        if not self.config.position_sizing_enabled:
            return 0.0, 0.0, 0.0, 0.0, "disabled", 1.0, "UNKNOWN"
        
        self._init_position_sizer()
        self._init_risk_manager()
        
        # Get pip value for pair
        pip_value = _get_pip_value(pair)
        
        # Get account equity
        equity = account_equity or self.fetch_live_nav() or self.config.account_equity or 10000.0

        # ATR-based SL (rule: SL = ATR * atr_sl_multiplier / pip_value, clamped to [min_sl_pips, max_sl_pips])
        # Fixes C-3: previously hardcoded to max_sl_pips regardless of volatility
        if atr > 0 and pip_value > 0:
            sl_pips = max(
                self.config.min_sl_pips,
                min(
                    (atr * self.config.atr_sl_multiplier) / pip_value,
                    self.config.max_sl_pips,
                ),
            )
        else:
            logger.warning(
                f"calculate_regime_aware_position_size: atr={atr}, pip_value={pip_value} — "
                f"cannot compute ATR-based SL, falling back to min_sl_pips"
            )
            sl_pips = self.config.min_sl_pips

        # Calculate base TP with high probability bonus
        tp_pips = self._calculate_base_tp_pips(atr, pip_value, confidence)

        # Apply risk manager adjustments
        sl_pips, tp_pips, confidence_level = self._apply_risk_manager(
            pair, confidence, sl_pips, tp_pips
        )
        
        # Calculate position size with regime-based scaling
        if hasattr(self._position_sizer, 'calculate_regime_scaled_position_size'):
            try:
                pos_result = self._position_sizer.calculate_regime_scaled_position_size(
                    account_equity=equity,
                    stop_loss_pips=sl_pips,
                    instrument=pair,
                    volatility_regime=volatility_regime,
                    meta_confidence=meta_confidence,
                    raw_confidence=confidence,
                )
                if pos_result.is_valid:
                    lots = pos_result.units / 100_000
                    risk_pct = pos_result.risk_amount / equity if equity > 0 else 0
                    return (
                        lots,
                        risk_pct,
                        sl_pips,
                        tp_pips,
                        pos_result.confidence_level,
                        pos_result.regime_scale_applied,
                        pos_result.regime_name,
                    )
            except Exception as e:
                logger.warning(f"Regime-aware position sizing failed: {e}")
        
        # Fallback to standard calculation
        lots, risk_pct, confidence_level = self._calculate_lots_from_sizer(
            equity, sl_pips, pair, confidence
        )
        
        regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
        regime_name = regime_names[volatility_regime] if 0 <= volatility_regime <= 3 else "UNKNOWN"
        
        return lots, risk_pct, sl_pips, tp_pips, confidence_level, 1.0, regime_name

    def _calculate_adaptive_position_size(
        self,
        equity: float,
        confidence: float,
        regime_name: str = "NORMAL",
        current_drawdown_pct: float = 0.0,
    ) -> Optional[Tuple[float, float, str]]:
        """Phase 45 (US-282): Calculate position size using adaptive sizer.

        Args:
            equity: Account equity in dollars
            confidence: Model confidence [0, 1]
            regime_name: Market regime (LOW, NORMAL, HIGH, EXTREME)
            current_drawdown_pct: Current drawdown percentage [0, 1]

        Returns:
            Tuple of (lots, risk_pct, strategy_label) or None if adaptive sizer unavailable.
        """
        if self._adaptive_position_sizer is None:
            self._init_adaptive_position_sizer()

        if self._adaptive_position_sizer is None:
            return None

        try:
            result = self._adaptive_position_sizer.calculate_size(
                account_equity=equity,
                confidence=confidence,
                current_drawdown_pct=current_drawdown_pct,
                regime_name=regime_name,
            )

            # Convert position size result to lots (units / 100,000)
            lots = result.final_size / 100_000
            lots = round(min(max(lots, self.config.min_lot_size), self.config.max_lot_size), 2)

            risk_pct = result.final_size / equity if equity > 0 else 0

            strategy_label = (
                f"adaptive(K={result.kelly_factor:.2f},"
                f"C={result.confidence_factor:.2f},"
                f"D={result.drawdown_factor:.2f},"
                f"R={result.regime_factor:.2f})"
            )

            logger.info(
                f"Phase 45: Adaptive sizing: {lots} lots, risk={risk_pct:.3f}, "
                f"regime={regime_name}, {strategy_label}"
            )

            if lots > 0:
                return lots, risk_pct, strategy_label

            return None
        except Exception as e:
            logger.warning(f"Phase 45: Adaptive position sizing failed: {e}")
            return None

    def calculate_position_size(
        self,
        pair: str,
        confidence: float,
        atr: float,
        account_equity: Optional[float] = None,
    ) -> Tuple[float, float, float, float, str]:
        """Calculate position sizing based on confidence.

        Args:
            pair: Instrument name
            confidence: Model confidence (0-1)
            atr: ATR value for SL calculation
            account_equity: Account equity (fetches from OANDA if None)

        Returns:
            Tuple of (lots, risk_pct, sl_pips, tp_pips, confidence_level)
        """
        if not self.config.position_sizing_enabled:
            return 0.0, 0.0, 0.0, 0.0, "disabled"

        self._init_position_sizer()
        self._init_risk_manager()

        # Get pip value for pair
        pip_value = _get_pip_value(pair)

        # Get account equity
        equity = account_equity or self.fetch_live_nav() or self.config.account_equity or 10000.0

        # ATR-based SL (rule: SL = ATR * atr_sl_multiplier / pip_value, clamped to [min_sl_pips, max_sl_pips])
        # Fixes C-3: previously hardcoded to max_sl_pips (15.0) regardless of volatility
        if atr > 0 and pip_value > 0:
            sl_pips = max(
                self.config.min_sl_pips,
                min(
                    (atr * self.config.atr_sl_multiplier) / pip_value,
                    self.config.max_sl_pips,
                ),
            )
        else:
            logger.warning(
                f"calculate_position_size: atr={atr}, pip_value={pip_value} — "
                f"cannot compute ATR-based SL, falling back to min_sl_pips"
            )
            sl_pips = self.config.min_sl_pips

        # Calculate base TP with high probability bonus
        tp_pips = self._calculate_base_tp_pips(atr, pip_value, confidence)

        # Apply risk manager adjustments
        sl_pips, tp_pips, confidence_level = self._apply_risk_manager(pair, confidence, sl_pips, tp_pips)

        # Phase 34 (US-222): Guard against zero SL/TP from risk manager edge cases
        if sl_pips <= 0:
            logger.warning(f"calculate_position_size: sl_pips={sl_pips} after risk manager — using min_sl_pips")
            sl_pips = max(self.config.min_sl_pips, 1.0)
        if tp_pips <= 0:
            logger.warning(f"calculate_position_size: tp_pips={tp_pips} after risk manager — using min_tp_pips")
            tp_pips = max(self.config.min_tp_pips, 1.0)

        # Phase 45 (US-282): Try adaptive position sizer first
        _regime_name = getattr(self, '_last_regime_name', 'NORMAL') or 'NORMAL'
        _drawdown_pct = getattr(self, '_current_drawdown_pct', 0.0) or 0.0
        _adaptive_result = self._calculate_adaptive_position_size(
            equity=equity,
            confidence=confidence,
            regime_name=_regime_name,
            current_drawdown_pct=_drawdown_pct,
        )
        if _adaptive_result is not None:
            lots, risk_pct, _strategy_label = _adaptive_result
            logger.debug(f"Phase 45: Using adaptive position sizing {_strategy_label}")

            # DynamicRiskAllocator: P/L distribution multiplier (after adaptive, before regime)
            lots = self._apply_dynamic_risk_multiplier(lots, _regime_name)

            # Phase 47 (US-298): Apply regime position multiplier
            lots = self._apply_regime_position_multiplier(lots, _regime_name)

            # Phase 47 (US-296): Apply session position multiplier
            lots = self._apply_session_position_multiplier(lots)

            return lots, risk_pct, sl_pips, tp_pips, confidence_level

        # Calculate position size (fallback)
        lots, risk_pct, confidence_level = self._calculate_lots_from_sizer(
            equity, sl_pips, pair, confidence
        )

        # DynamicRiskAllocator: P/L distribution multiplier (after base, before regime)
        lots = self._apply_dynamic_risk_multiplier(lots, _regime_name)

        # Phase 47 (US-298): Apply regime position multiplier
        lots = self._apply_regime_position_multiplier(lots, _regime_name)

        # Phase 47 (US-296): Apply session position multiplier
        lots = self._apply_session_position_multiplier(lots)

        return lots, risk_pct, sl_pips, tp_pips, confidence_level

    def set_dynamic_risk_allocator(self, allocator) -> None:
        """Inject DynamicRiskAllocator instance (called from engine.py).

        Args:
            allocator: DynamicRiskAllocator instance (or None).
        """
        self._dynamic_risk_allocator = allocator

    def _apply_dynamic_risk_multiplier(self, lots: float, regime_name: str) -> float:
        """Apply DynamicRiskAllocator P/L-distribution multiplier to position size.

        Uses rolling P/L statistics and streak detection to scale risk
        allocation per regime. Returns multiplier in [0.50, 1.30].

        Args:
            lots: Base position size in lots
            regime_name: Current volatility regime (LOW, NORMAL, HIGH, EXTREME)

        Returns:
            Adjusted position size after applying dynamic risk multiplier
        """
        if self._dynamic_risk_allocator is None:
            return lots

        try:
            risk_mult = self._dynamic_risk_allocator.get_risk_multiplier(regime_name)
            if risk_mult == 1.0:
                return lots
            adjusted_lots = lots * risk_mult
            logger.debug(
                f"DynamicRiskAllocator: Regime {regime_name} applied {risk_mult:.4f}x "
                f"to position size ({lots:.2f} -> {adjusted_lots:.2f} lots)"
            )
            return adjusted_lots
        except Exception as e:
            logger.debug(f"DynamicRiskAllocator: multiplier error: {e}, using original lots")
            return lots

    def _apply_regime_position_multiplier(self, lots: float, regime_name: str) -> float:
        """Phase 47 (US-298): Apply regime-based position size multiplier.

        Args:
            lots: Base position size in lots
            regime_name: Current volatility regime (LOW, NORMAL, HIGH, EXTREME)

        Returns:
            Adjusted position size after applying regime multiplier
        """
        try:
            from src.scanner.regime_gates import get_regime_profile
            profile = get_regime_profile(regime_name)
            if profile is None:
                logger.debug(f"Phase 47: No regime profile for {regime_name}, using 1.0x multiplier")
                return lots

            regime_mult = profile.position_size_multiplier
            adjusted_lots = lots * regime_mult
            logger.debug(
                f"Phase 47 (US-298): Regime {regime_name} applied {regime_mult:.2f}x "
                f"to position size ({lots:.2f} → {adjusted_lots:.2f} lots)"
            )
            return adjusted_lots
        except Exception as e:
            logger.debug(f"Phase 47 (US-298): Regime multiplier error: {e}, using original lots")
            return lots

    def _apply_session_position_multiplier(self, lots: float) -> float:
        """Phase 47 (US-296): Apply session-based position size multiplier.

        Args:
            lots: Base position size in lots

        Returns:
            Adjusted position size after applying session multiplier
        """
        if self._session_detector is None:
            self._init_session_detector()
        if self._session_detector is None:
            return lots

        try:
            session_mult = self._session_detector.get_position_size_multiplier()
            adjusted_lots = lots * session_mult
            session_info = self._session_detector.get_current_session()
            logger.debug(
                f"Phase 47 (US-296): Session {session_info.name} applied {session_mult:.2f}x "
                f"to position size ({lots:.2f} → {adjusted_lots:.2f} lots)"
            )
            return adjusted_lots
        except Exception as e:
            logger.debug(f"Phase 47 (US-296): Session multiplier error: {e}, using original lots")
            return lots

    def execute_trade(
        self,
        pair: str,
        direction: str,
        confidence: float,
        current_price: float,
        atr: float,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        lots: Optional[float] = None,
        volatility_regime: Optional[int] = None,
        meta_confidence: Optional[float] = None,
        analysis_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a single trade on OANDA.

        Args:
            pair: Instrument name (e.g., "EUR_USD")
            direction: Trade direction ("LONG" or "SHORT")
            confidence: Model confidence (0-1)
            current_price: Current market price
            atr: ATR value for SL/TP calculation
            sl_pips: Override stop loss pips
            tp_pips: Override take profit pips
            lots: Override position size in lots
            volatility_regime: Volatility regime for aggressive scaling (0-3)
            meta_confidence: Meta-labeler confidence for aggressive scaling

        Returns:
            ExecutionResult with trade details
        """
        # Phase 84 (US-P84-004): CircuitBreaker — reject if API is degraded
        _cb = getattr(self, "_api_circuit_breaker", None)
        if _cb is not None:
            try:
                if _cb.is_tripped:
                    _cb_status = _cb.get_status()
                    logger.warning(
                        "Phase 84 (US-P84-004): Circuit breaker TRIPPED — "
                        "failure_rate=%.0f%% trips=%d — skipping execution for %s",
                        _cb_status.get("failure_rate", 0) * 100,
                        _cb_status.get("total_trips", 0),
                        pair,
                    )
                    return ExecutionResult(
                        success=False,
                        error="Circuit breaker tripped: API degraded",
                    )
            except Exception as _cb_err:
                logger.debug("Phase 84: CircuitBreaker check error: %s", _cb_err)

        # Check data freshness
        ctx = analysis_context or {}
        scan_time = ctx.get("scan_time") or ctx.get("analysis_timestamp")
        if scan_time:
            try:
                if isinstance(scan_time, str):
                    scan_dt = datetime.fromisoformat(scan_time.replace("Z", "+00:00"))
                else:
                    scan_dt = scan_time
                age_seconds = (datetime.now(timezone.utc) - scan_dt).total_seconds()
                max_age = getattr(self.config, "max_data_age_seconds", 60.0)
                if age_seconds > max_age:
                    reason = f"Stale data: analysis {age_seconds:.0f}s old (max {max_age:.0f}s)"
                    try:
                        # Phase 30 (US-183): Use shared observer
                        self._get_observer().log(
                            category="stale_data_skip",
                            detail=reason,
                            data={"pair": pair, "age_seconds": age_seconds},
                        )
                    except Exception as e:
                        logger.debug(f"Execution pair operation skipped: {e}")
                    return ExecutionResult(success=False, error=reason)
            except Exception as e:
                logger.debug(f"Data freshness check error: {e}")

        # ── DEFENSE-IN-DEPTH: Signal quality validation ──────────
        # These checks are the last safety net before real money leaves the account.
        # They enforce trading rules even if upstream gate logic has a bug.
        if not ctx.get("agent_passed", False):
            return ExecutionResult(
                success=False,
                error=f"BLOCKED: agent_passed=False (agents voted NO)",
            )
        _ctx_disagreement = float(ctx.get("model_disagreement", 0.0))
        _max_disagreement = getattr(self.config, "max_model_disagreement", 0.65)
        if _ctx_disagreement > _max_disagreement:
            return ExecutionResult(
                success=False,
                error=f"BLOCKED: model_disagreement={_ctx_disagreement:.2f} > {_max_disagreement:.2f}",
            )
        _ctx_regime = str(ctx.get("volatility_regime", "UNKNOWN")).upper()
        if _ctx_regime == "UNKNOWN":
            return ExecutionResult(
                success=False,
                error="BLOCKED: cannot execute with UNKNOWN regime",
            )
        _ctx_confidence = float(ctx["confidence"]) if "confidence" in ctx else confidence
        if _ctx_confidence < 0.45:
            return ExecutionResult(
                success=False,
                error=f"BLOCKED: confidence={_ctx_confidence:.3f} < 0.45 minimum",
            )
        logger.info(
            "PRE-ORDER AUDIT: %s %s conf=%.3f agent_passed=%s disagreement=%.2f regime=%s",
            pair, direction, _ctx_confidence,
            ctx.get("agent_passed"), _ctx_disagreement, _ctx_regime,
        )

        # Phase 85 (US-P85-003): ClusterGate — adjust confidence based on trade cluster membership
        if self._cluster_gate is not None:
            try:
                _cg_trade = {
                    "pair": pair, "direction": direction,
                    "confidence": confidence, "regime": _ctx_regime,
                    "atr": atr, "spread_pips": ctx.get("spread_pips", 0.0),
                }
                _cg_result = self._cluster_gate.evaluate(_cg_trade, confidence)
                if _cg_result.adjusted_confidence != confidence:
                    confidence = _cg_result.adjusted_confidence
                    logger.info(
                        "Phase 85 (US-P85-003): ClusterGate adjusted confidence %.3f → %.3f "
                        "(cluster=%s adj=%+.3f)",
                        _cg_result.original_confidence, confidence,
                        _cg_result.cluster_label or "?",
                        _cg_result.confidence_adjustment,
                    )
            except Exception as _cg_err:
                logger.debug("Phase 85: ClusterGate evaluate error: %s", _cg_err)

        # Duplicate position check — don't open same pair twice
        try:
            _open_positions = self.monitor_open_trades(evaluate_exits=False) or []
            _open_pairs = {s.get("pair", "") for s in _open_positions}
            if pair in _open_pairs:
                return ExecutionResult(
                    success=False,
                    error=f"SKIP: {pair} already has an open position",
                )
        except Exception as _dup_err:
            logger.debug(f"Duplicate position check error (non-blocking): {_dup_err}")

        # Phase 45 (US-285): Feed price return into EWMA correlation matrix
        _prev_price = self._ewma_price_cache.get(pair, 0.0)
        if _prev_price > 0 and current_price > 0:
            self._update_ewma_correlation(pair, current_price, _prev_price)
        self._ewma_price_cache[pair] = current_price

        # ── Phase 52 (US-326): Drawdown Adapter — EARLIEST gate ──
        # Must run before all other gates to halt/restrict trading under drawdown.
        if self._drawdown_adapter is not None:
            try:
                _peak_nav = getattr(self, "_peak_nav_tracked", 0.0)
                _current_nav = self.fetch_live_nav() or self.config.account_equity or 10000.0
                if _peak_nav <= 0:
                    _peak_nav = _current_nav
                # Get session name from context
                _session_name = str(ctx.get("session", ctx.get("trading_session", "LONDON"))).upper()
                # Get pair win rates from strategy fitness if available
                _pair_win_rates = {}
                if self._strategy_fitness is not None:
                    try:
                        _regime_for_dd = _ctx_regime or "NORMAL"
                        for _p in get_registry().get_all_symbols():
                            try:
                                _pf = self._strategy_fitness.check_fitness(_p, _regime_for_dd)
                                _wr = getattr(_pf, "win_rate", 0.0) or 0.0
                                if _wr > 0:
                                    _pair_win_rates[_p] = _wr
                            except Exception as _exc:
                                logger.debug("H-4: Silent exception in check_fitness: %s", _exc)
                    except Exception as _exc:
                        logger.debug("H-4: Silent exception in debug: %s", _exc)

                _dd_result = self._drawdown_adapter.check(
                    current_nav=_current_nav,
                    peak_nav=_peak_nav,
                    pair=pair,
                    confidence=confidence,
                    session_name=_session_name,
                    pair_win_rates=_pair_win_rates if _pair_win_rates else None,
                )
                self._peak_nav_tracked = max(_peak_nav, _current_nav)
                ctx["_drawdown_pct"] = _dd_result.drawdown_pct
                ctx["_drawdown_tier"] = _dd_result.active_tier

                if not _dd_result.allowed:
                    # Phase 53 (US-328): Record gate rejection
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "drawdown_adapter", rejected=True,
                                reason=_dd_result.reason, confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)
                    return ExecutionResult(
                        success=False,
                        error=_dd_result.reason,
                    )
                else:
                    # Phase 53 (US-328): Record gate pass
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "drawdown_adapter", rejected=False,
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)

                # Apply confidence tightening from Tier 1+
                if _dd_result.confidence_penalty > 0:
                    _old_conf_dd = confidence
                    confidence = _dd_result.adjusted_confidence
                    logger.info(
                        "Phase 52 (US-326): %s drawdown tier %d — confidence %.3f → %.3f",
                        pair, _dd_result.active_tier, _old_conf_dd, confidence,
                    )
                    # Re-check minimum confidence after tightening
                    if confidence < 0.45:
                        return ExecutionResult(
                            success=False,
                            error=f"BLOCKED: confidence {confidence:.3f} < 0.45 after drawdown penalty (tier {_dd_result.active_tier}, dd={_dd_result.drawdown_pct:.1f}%)",
                        )
            except Exception as _dd_err:
                logger.debug(f"Phase 52: Drawdown adapter error: {_dd_err}")

        # ── Phase 53 (US-330): Signal Freshness Guard — recompute RSI/ADX ──
        if self._signal_freshness is not None:
            try:
                _scan_rsi = float(ctx.get("rsi", ctx.get("rsi_14", 0.0)))
                _scan_adx = float(ctx.get("adx", ctx.get("adx_14", 0.0)))
                _recent_closes = ctx.get("recent_closes", ctx.get("close_prices", None))
                _recent_highs = ctx.get("recent_highs", ctx.get("high_prices", None))
                _recent_lows = ctx.get("recent_lows", ctx.get("low_prices", None))

                if _scan_rsi > 0 or _scan_adx > 0:
                    _fresh_result = self._signal_freshness.check(
                        pair=pair,
                        scan_rsi=_scan_rsi,
                        scan_adx=_scan_adx,
                        current_prices=_recent_closes,
                        current_highs=_recent_highs,
                        current_lows=_recent_lows,
                        confidence=confidence,
                    )
                    ctx["_signal_stale"] = _fresh_result.stale
                    ctx["_staleness_rate"] = _fresh_result.staleness_rate

                    if _fresh_result.stale:
                        confidence = _fresh_result.adjusted_confidence
                        # Record stale gate decision
                        if self._gate_attribution is not None:
                            try:
                                if _fresh_result.rejected:
                                    self._gate_attribution.record_decision(
                                        "staleness_guard", rejected=True,
                                        reason=_fresh_result.reason,
                                        confidence=_fresh_result.adjusted_confidence, pair=pair,
                                    )
                                else:
                                    self._gate_attribution.record_decision(
                                        "staleness_guard", rejected=False,
                                        confidence=confidence, pair=pair,
                                    )
                            except Exception as _exc:
                                logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)
                        if _fresh_result.rejected:
                            return ExecutionResult(
                                success=False,
                                error=_fresh_result.reason,
                            )
                    else:
                        # Record fresh pass
                        if self._gate_attribution is not None:
                            try:
                                self._gate_attribution.record_decision(
                                    "staleness_guard", rejected=False,
                                    confidence=confidence, pair=pair,
                                )
                            except Exception as _exc:
                                logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)
            except Exception as _sf_err:
                logger.debug(f"Phase 53: Signal freshness check error: {_sf_err}")

        # ── Phase 49 (US-306): Pre-execution gates from Phase 48 modules ──
        # Size multiplier compounds across all gates (spread × calendar × fitness)
        _phase49_size_mult = 1.0
        ctx["_spread_mult"] = 1.0
        ctx["_calendar_mult"] = 1.0
        ctx["_fitness_mult"] = 1.0

        # ── Episodic Suppression Gate (Earliest Gate) ──
        # Block setups with repeated historical losses before any other gate
        if self._episodic_memory is not None:
            try:
                _regime = ctx.get("regime", ctx.get("volatility_regime", "unknown"))
                if isinstance(_regime, int):
                    _regime_map = {0: "NORMAL", 1: "TRENDING", 2: "RANGING", 3: "VOLATILE"}
                    _regime = _regime_map.get(_regime, "unknown")
                else:
                    _regime = str(_regime).upper() if _regime else "unknown"
                _session = ctx.get("session", ctx.get("trading_session", "unknown"))
                _session = str(_session).upper() if _session else "unknown"
                _news_risk = float(ctx.get("news_risk_score", 0.0))
                _uncertainty = float(ctx.get("uncertainty_score", 0.0))

                _suppressed = self._episodic_memory.get_suppression_signal(
                    pair=pair,
                    direction=direction,
                    regime=_regime,
                    session=_session,
                    news_risk_score=_news_risk,
                    uncertainty_score=_uncertainty,
                )
                if _suppressed:
                    logger.warning(
                        f"[EPISODIC BLOCK] {pair} {direction} suppressed (regime={_regime}, session={_session}) — "
                        f"repeated losing pattern detected in episodic memory"
                    )
                    # Phase 53 (US-328): Record gate rejection
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "episodic_suppression", rejected=True,
                                reason="Pattern loss rate >= 70% with 3+ similar episodes",
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)
                    return ExecutionResult(
                        success=False,
                        error=f"EPISODIC_SUPPRESSION: {pair} {direction} blocked by episodic memory"
                    )
            except Exception as _em_err:
                logger.debug(f"Episodic suppression check failed (fail-open): {_em_err}")
                # Fail-open: any error allows trade to proceed

        # Gate 1: Spread filter — reject if spread > ATR threshold
        if self._spread_filter is not None:
            try:
                _spread_pips = ctx.get("spread_pips", 0.0)
                _atr_pips = ctx.get("atr_pips", 0.0) or (atr / _get_pip_value(pair, 0.0001) if atr > 0 else 0.0)
                if _spread_pips > 0 and _atr_pips > 0:
                    _spread_result = self._spread_filter.check_spread_from_price(
                        pair=pair, price=current_price,
                        spread_pips=_spread_pips, atr_pips=_atr_pips,
                    )
                    if not _spread_result.passed:
                        logger.info(
                            "Phase 49 (US-306): %s BLOCKED by spread filter — "
                            "normalized=%.3f threshold=%.3f",
                            pair, _spread_result.normalized_spread, _spread_result.threshold,
                        )
                        # Phase 53 (US-328): Record gate rejection
                        if self._gate_attribution is not None:
                            try:
                                self._gate_attribution.record_decision(
                                    "spread_filter", rejected=True,
                                    reason=f"spread {_spread_result.normalized_spread:.3f} > {_spread_result.threshold:.3f}",
                                    confidence=confidence, pair=pair,
                                )
                            except Exception as _exc:
                                logger.debug("H-4: Silent exception in unknown: %s", _exc)
                        return ExecutionResult(
                            success=False,
                            error=f"BLOCKED: spread too wide ({_spread_result.normalized_spread:.3f} > {_spread_result.threshold:.3f})",
                        )
                    else:
                        # Phase 53 (US-328): Record gate pass
                        if self._gate_attribution is not None:
                            try:
                                self._gate_attribution.record_decision(
                                    "spread_filter", rejected=False,
                                    confidence=confidence, pair=pair,
                                )
                            except Exception as _exc:
                                logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)
                    if _spread_result.size_multiplier < 1.0:
                        _phase49_size_mult *= _spread_result.size_multiplier
                        ctx["_spread_mult"] = _spread_result.size_multiplier
                        logger.info(
                            "Phase 49 (US-306): %s spread REDUCE_SIZE %.2fx (normalized=%.3f)",
                            pair, _spread_result.size_multiplier, _spread_result.normalized_spread,
                        )
            except Exception as _sf_err:
                logger.debug(f"Phase 49: Spread filter check error: {_sf_err}")

        # Gate 2: Economic calendar — blackout around high-impact events
        if self._calendar_filter is not None:
            try:
                _cal_result = self._calendar_filter.check_pair(pair)
                if _cal_result.action == "BLACKOUT":
                    _evt_name = _cal_result.nearest_event.event_name if _cal_result.nearest_event else "unknown"
                    logger.info(
                        "Phase 49 (US-306): %s BLOCKED by calendar blackout — %s (%s)",
                        pair, _cal_result.reason, _evt_name,
                    )
                    # Phase 53 (US-328): Record gate rejection
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "calendar_filter", rejected=True,
                                reason=f"blackout: {_evt_name}",
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in unknown: %s", _exc)
                    return ExecutionResult(
                        success=False,
                        error=f"BLOCKED: economic calendar blackout ({_evt_name})",
                    )
                else:
                    # Phase 53 (US-328): Record gate pass
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "calendar_filter", rejected=False,
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)
                if _cal_result.action == "REDUCE_SIZE" and _cal_result.size_multiplier < 1.0:
                    _phase49_size_mult *= _cal_result.size_multiplier
                    ctx["_calendar_mult"] = _cal_result.size_multiplier
                    logger.info(
                        "Phase 49 (US-306): %s calendar REDUCE_SIZE %.2fx — %s",
                        pair, _cal_result.size_multiplier, _cal_result.reason,
                    )
            except Exception as _cf_err:
                logger.debug(f"Phase 49: Calendar filter check error: {_cf_err}")

        # Gate 3: Strategy fitness — skip if strategy underperforming for this pair/regime
        if self._strategy_fitness is not None:
            try:
                _regime_for_fitness = _ctx_regime or "NORMAL"
                _fitness_result = self._strategy_fitness.check_fitness(pair, _regime_for_fitness)
                if _fitness_result.action == "SKIP":
                    logger.info(
                        "Phase 49 (US-306): %s BLOCKED by fitness detector — "
                        "score=%.3f (regime=%s)",
                        pair, _fitness_result.fitness_score, _regime_for_fitness,
                    )
                    # Phase 53 (US-328): Record gate rejection
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "fitness_detector", rejected=True,
                                reason=f"fitness {_fitness_result.fitness_score:.3f} in {_regime_for_fitness}",
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in unknown: %s", _exc)
                    return ExecutionResult(
                        success=False,
                        error=f"BLOCKED: strategy fitness too low ({_fitness_result.fitness_score:.3f} in {_regime_for_fitness})",
                    )
                else:
                    # Phase 53 (US-328): Record gate pass
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "fitness_detector", rejected=False,
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)
                if _fitness_result.action == "REDUCE_SIZE" and _fitness_result.size_multiplier < 1.0:
                    _phase49_size_mult *= _fitness_result.size_multiplier
                    ctx["_fitness_mult"] = _fitness_result.size_multiplier
                    logger.info(
                        "Phase 49 (US-306): %s fitness REDUCE_SIZE %.2fx (score=%.3f)",
                        pair, _fitness_result.size_multiplier, _fitness_result.fitness_score,
                    )
            except Exception as _fit_err:
                logger.debug(f"Phase 49: Fitness check error: {_fit_err}")

        # Gate 4: Pair blacklist — block pairs that have been auto-blacklisted by PairPerformanceTracker
        # _pair_tracker.update_from_trade() accumulates outcomes after close;
        # _pair_tracker.check_pair() enforces the blacklist here before entry.
        if self._pair_tracker is not None:
            try:
                _bl_result = self._pair_tracker.check_pair(pair)
                if _bl_result.is_blacklisted:
                    logger.info(
                        "Phase 49 (US-307): %s BLOCKED by pair blacklist — %s (cooldown=%d cycles)",
                        pair, _bl_result.reason, _bl_result.cooldown_remaining,
                    )
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "pair_blacklist", rejected=True,
                                reason=_bl_result.reason,
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("Phase 49: gate_attribution record error: %s", _exc)
                    return ExecutionResult(
                        success=False,
                        error=f"BLOCKED: pair blacklisted ({_bl_result.reason})",
                    )
                else:
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "pair_blacklist", rejected=False,
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("Phase 49: gate_attribution record error: %s", _exc)
            except Exception as _bl_err:
                logger.debug(f"Phase 49: Pair blacklist check error: {_bl_err}")

        # ── Phase 52 (US-323): Pattern Gate — journal-mined confidence adjustments ──
        ctx["_pattern_boost"] = 0.0
        if self._pattern_gate is not None:
            try:
                _pg_direction = direction.upper() if isinstance(direction, str) else "LONG"
                _pg_result = self._pattern_gate.evaluate(
                    pair=pair,
                    direction=_pg_direction,
                    confidence=confidence,
                )
                if _pg_result.adjusted_confidence != confidence:
                    _old_conf = confidence
                    confidence = _pg_result.adjusted_confidence
                    ctx["_pattern_boost"] = round(_pg_result.pair_boost + _pg_result.direction_boost, 4)
                    logger.info(
                        "Phase 52 (US-323): %s pattern gate adjusted confidence %.3f → %.3f "
                        "(pair_boost=%.3f, dir_boost=%.3f)",
                        pair, _old_conf, confidence,
                        _pg_result.pair_boost, _pg_result.direction_boost,
                    )
                if _pg_result.duration_warning:
                    logger.info(
                        "Phase 52 (US-323): %s duration warning — %s",
                        pair, "; ".join(_pg_result.warnings),
                    )
            except Exception as _pg_err:
                logger.debug(f"Phase 52: Pattern gate check error: {_pg_err}")

        # ── Phase 52 (US-323): Setup Quality Filter — reject weak setups ──
        if self._setup_quality_filter is not None:
            try:
                _agents_info = ctx.get("agents", {})
                _disagreement = float(_agents_info.get("model_disagreement", 0.0))
                _uncertainty = float(_agents_info.get("uncertainty_score", 0.0))
                _pair_wr = 0.5  # Default if no fitness data
                if self._strategy_fitness is not None:
                    try:
                        _regime_for_wr = _ctx_regime or "NORMAL"
                        _fit_check = self._strategy_fitness.check_fitness(pair, _regime_for_wr)
                        _pair_wr = getattr(_fit_check, "win_rate", 0.5) or 0.5
                    except Exception as _exc:
                        logger.debug("H-4: Silent exception in check_fitness: %s", _exc)

                _sq_result = self._setup_quality_filter.evaluate(
                    pair=pair,
                    confidence=confidence,
                    agent_disagreement=_disagreement,
                    uncertainty_score=_uncertainty,
                    pair_recent_win_rate=_pair_wr,
                )
                if not _sq_result.passed:
                    logger.info(
                        "Phase 52 (US-323): %s REJECTED by setup quality filter — "
                        "score=%.3f < %.3f — %s",
                        pair, _sq_result.quality_score, _sq_result.details.get("threshold", 0.35),
                        "; ".join(_sq_result.rejection_reasons),
                    )
                    # Phase 53 (US-328): Record gate rejection
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "setup_quality", rejected=True,
                                reason=f"quality {_sq_result.quality_score:.3f}: {'; '.join(_sq_result.rejection_reasons)}",
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in unknown: %s", _exc)
                    return ExecutionResult(
                        success=False,
                        error=f"BLOCKED: setup quality too low ({_sq_result.quality_score:.3f}): {'; '.join(_sq_result.rejection_reasons)}",
                    )
                else:
                    # Phase 53 (US-328): Record gate pass
                    if self._gate_attribution is not None:
                        try:
                            self._gate_attribution.record_decision(
                                "setup_quality", rejected=False,
                                confidence=confidence, pair=pair,
                            )
                        except Exception as _exc:
                            logger.debug("H-4: Silent exception in gate_attribution: %s", _exc)
                ctx["_setup_quality_score"] = _sq_result.quality_score
                ctx["_setup_quality_dims"] = _sq_result.dimension_scores
            except Exception as _sq_err:
                logger.debug(f"Phase 52: Setup quality filter error: {_sq_err}")

        # ── Phase 52 (US-325): Adaptive R:R — override atr_tp_multiplier by regime/confidence ──
        ctx["_adaptive_rr_applied"] = False
        if self._adaptive_rr is not None:
            try:
                _regime_for_rr = _ctx_regime or "NORMAL"
                # Get pair recent win rate from fitness detector if available
                _pair_wr_for_rr = 0.50
                if self._strategy_fitness is not None:
                    try:
                        _fit_for_rr = self._strategy_fitness.check_fitness(pair, _regime_for_rr)
                        _pair_wr_for_rr = getattr(_fit_for_rr, "win_rate", 0.50) or 0.50
                    except Exception as _exc:
                        logger.debug("H-4: Silent exception in check_fitness: %s", _exc)

                _rr_result = self._adaptive_rr.calculate(
                    regime=_regime_for_rr,
                    confidence=confidence,
                    atr_sl_multiplier=self.config.atr_sl_multiplier,
                    pair_recent_win_rate=_pair_wr_for_rr,
                )
                _old_tp_mult = self.config.atr_tp_multiplier
                if abs(_rr_result.tp_multiplier - _old_tp_mult) > 0.001:
                    self.config.atr_tp_multiplier = _rr_result.tp_multiplier
                    ctx["_adaptive_rr_applied"] = True
                    ctx["_adaptive_rr_old_tp_mult"] = _old_tp_mult
                    ctx["_adaptive_rr_new_tp_mult"] = _rr_result.tp_multiplier
                    ctx["_adaptive_rr_ratio"] = _rr_result.adjusted_rr
                    ctx["_adaptive_rr_floor"] = _rr_result.floor_applied
                    logger.info(
                        "Phase 52 (US-325): %s adaptive R:R %s → %.2f:1 "
                        "(tp_mult %.3f → %.3f, conf_adj=%.2f, form_adj=%.2f%s)",
                        pair, _regime_for_rr, _rr_result.adjusted_rr,
                        _old_tp_mult, _rr_result.tp_multiplier,
                        _rr_result.confidence_adj, _rr_result.form_adj,
                        " FLOOR" if _rr_result.floor_applied else "",
                    )
            except Exception as _rr_err:
                logger.debug(f"Phase 52: Adaptive R:R error: {_rr_err}")

        # Phase 50 (US-315): Apply compound gate floor to prevent near-zero sizing
        _compound_floor = float(getattr(self.config, "compound_gate_floor", 0.10))
        if _phase49_size_mult < _compound_floor and _phase49_size_mult > 0:
            logger.warning(
                "Phase 50 (US-315): %s compound multiplier %.3f below floor %.3f — "
                "clamping to floor (spread=%.3f, calendar=%.3f, fitness=%.3f)",
                pair, _phase49_size_mult, _compound_floor,
                ctx.get("_spread_mult", 1.0),
                ctx.get("_calendar_mult", 1.0),
                ctx.get("_fitness_mult", 1.0),
            )
            _phase49_size_mult = _compound_floor

        # Store compound multiplier for later application to position size
        if _phase49_size_mult < 1.0:
            logger.info(
                "Phase 49 (US-306): %s compound size multiplier = %.3f",
                pair, _phase49_size_mult,
            )
        ctx["_phase49_size_multiplier"] = _phase49_size_mult

        # ── Phase 81 (US-P81-001): FilterChain.apply() — extensible pre-execution gate ──
        # Fail-open: any exception in a filter returns PASS so no broken filter blocks trading.
        if self._filter_chain is not None and self._filter_chain.get_filter_names():
            try:
                from src.scanner.execution_filters import FilterResult as _FR
                _fc_context = {
                    "pair": pair,
                    "direction": direction,
                    "confidence": confidence,
                    "atr": atr,
                    "spread_pips": ctx.get("spread_pips", 0.0),
                    "atr_pips": ctx.get("atr_pips", 0.0),
                    "volatility_regime": ctx.get("_ctx_regime", "NORMAL"),
                    "phase49_size_multiplier": _phase49_size_mult,
                }
                _fc_result = self._filter_chain.apply(_fc_context)
                if not _fc_result.passed and _fc_result.action == "BLOCK":
                    logger.info(
                        "Phase 81 (FilterChain): %s BLOCKED — %s [filter=%s]",
                        pair, _fc_result.reason, _fc_result.filter_name,
                    )
                    return ExecutionResult(
                        success=False,
                        error=f"BLOCKED by filter chain: {_fc_result.reason}",
                    )
                if _fc_result.action == "REDUCE_SIZE" and _fc_result.size_multiplier < 1.0:
                    _phase49_size_mult *= _fc_result.size_multiplier
                    ctx["_phase49_size_multiplier"] = _phase49_size_mult
                    logger.info(
                        "Phase 81 (FilterChain): %s REDUCE_SIZE %.3fx — %s",
                        pair, _fc_result.size_multiplier, _fc_result.reason,
                    )
            except Exception as _fc_err:
                logger.debug("Phase 81 (FilterChain): apply() failed (fail-open): %s", _fc_err)

        # Check daily limit
        can_trade, reason = self.can_trade()
        if not can_trade:
            return ExecutionResult(success=False, error=reason)

        # Check portfolio risk limit
        risk_ok, risk_reason = self._check_portfolio_risk_limit()
        if not risk_ok:
            # Phase 26 (US-157): Log portfolio risk block to observations
            try:
                # Phase 30 (US-183): Use shared observer
                self._get_observer().log_observation(
                    pair=pair,
                    category="portfolio_risk_block",
                    description=f"US-157: Trade blocked — {risk_reason}",
                    metadata={"pair": pair, "reason": risk_reason, "gate": "portfolio_risk_limit"},
                )
            except Exception as e:
                logger.debug(f"Execution pair operation skipped: {e}")
            return ExecutionResult(success=False, error=risk_reason)

        # Check correlation exposure limit
        corr_ok, corr_reason = self._check_correlation_exposure(pair)
        if not corr_ok:
            return ExecutionResult(success=False, error=corr_reason)

        # Initialize broker
        if not self._init_broker():
            return ExecutionResult(success=False, error="Broker client not available")

        # Phase 31 (US-191): Cache NAV once per execute_trade to avoid redundant API calls
        _cached_nav = self.fetch_live_nav() or self.config.account_equity or 10000.0

        # Initialize regime scaling info
        regime_scale = 1.0
        regime_name = "UNKNOWN"
        aggressive_reason = ""

        # Calculate position sizing if not provided
        if (
            lots is None or float(lots) <= 0.0 or
            sl_pips is None or float(sl_pips) <= 0.0 or
            tp_pips is None or float(tp_pips) <= 0.0
        ):
            # When scan already provides SL/TP but not lots, use the scan SL
            # for sizing (not the internal max_sl_pips) so risk is accurate.
            scan_sl = float(sl_pips) if sl_pips and float(sl_pips) > 0 else None

            # Use regime-aware sizing if regime info provided
            if volatility_regime is not None and meta_confidence is not None:
                (
                    calc_lots,
                    risk_pct,
                    calc_sl,
                    calc_tp,
                    conf_level,
                    regime_scale,
                    regime_name,
                ) = self.calculate_regime_aware_position_size(
                    pair=pair,
                    confidence=confidence,
                    atr=atr,
                    volatility_regime=volatility_regime,
                    meta_confidence=meta_confidence,
                )
                # Re-calculate lots using actual SL if scan provided one
                if scan_sl and lots is None and calc_lots > 0:
                    equity = _cached_nav  # Phase 31 (US-191): cached NAV
                    calc_lots, risk_pct, conf_level = self._calculate_lots_from_sizer(
                        equity, scan_sl, pair, confidence
                    )
            else:
                calc_lots, risk_pct, calc_sl, calc_tp, conf_level = self.calculate_position_size(
                    pair, confidence, atr
                )
                # Re-calculate lots using actual SL if scan provided one
                if scan_sl and lots is None:
                    equity = _cached_nav  # Phase 31 (US-191): cached NAV
                    calc_lots, risk_pct, conf_level = self._calculate_lots_from_sizer(
                        equity, scan_sl, pair, confidence
                    )
            lots = lots or calc_lots
            sl_pips = sl_pips or calc_sl
            tp_pips = tp_pips or calc_tp
        else:
            risk_pct = self.config.risk_per_trade_pct
            conf_level = "custom"

        # Phase 22 (US-135): Analysis-level risk_pct override — DISABLED
        # Was overriding position sizer's risk_pct with Kelly's conservative output,
        # making positions 20x smaller than intended. Portfolio risk is bounded
        # by _check_projected_portfolio_risk() instead.

        # Apply regime risk modifier from Markov chain tracker (Phase 8)
        regime_risk_mod = float(
            (analysis_context or {}).get("regime_risk_modifier", 1.0)
        )
        if 0.0 < regime_risk_mod < 1.0:
            original_lots = lots
            lots = round(lots * regime_risk_mod, 2)
            logger.info(
                f"Regime risk modifier {regime_risk_mod:.2f}x applied: "
                f"{original_lots:.2f} → {lots:.2f} lots"
            )

        # Apply predictive size modifier from Markov forward-looking predictions (Phase 9)
        predictive_mod = float(
            (analysis_context or {}).get("predictive_size_modifier", 1.0)
        )
        if 0.0 < predictive_mod < 1.0:
            original_lots = lots
            lots = round(lots * predictive_mod, 2)
            logger.info(
                f"Predictive size modifier {predictive_mod:.2f}x applied: "
                f"{original_lots:.2f} → {lots:.2f} lots"
            )

        # Apply macro stress modifier (Phase 10, US-066)
        macro_mod = float(
            (analysis_context or {}).get("macro_stress_modifier", 1.0)
        )
        if 0.0 < macro_mod < 1.0:
            original_lots = lots
            lots = round(lots * macro_mod, 2)
            logger.info(
                f"Macro stress modifier {macro_mod:.2f}x applied: "
                f"{original_lots:.2f} → {lots:.2f} lots"
            )

        if lots <= 0:
            return ExecutionResult(success=False, error="Invalid position size")

        # Phase 52 (US-325): Restore original atr_tp_multiplier after sizing
        if ctx.get("_adaptive_rr_applied"):
            self.config.atr_tp_multiplier = ctx["_adaptive_rr_old_tp_mult"]

        # Minimum risk:reward ratio gate
        # C-7: reject immediately if SL or TP is zero — fail-closed, not fail-open
        min_rr = getattr(self.config, "min_risk_reward_ratio", 1.2)
        if sl_pips <= 0 or tp_pips <= 0:
            return ExecutionResult(
                success=False,
                error=f"Invalid SL/TP values: sl={sl_pips:.4f}, tp={tp_pips:.4f} — trade rejected",
            )
        rr_ratio = tp_pips / sl_pips
        if rr_ratio < min_rr:
            return ExecutionResult(
                success=False,
                error=f"R:R {rr_ratio:.2f}:1 below minimum {min_rr}:1 (SL {sl_pips:.1f}p / TP {tp_pips:.1f}p)",
            )

        # ── PROJECTED PORTFOLIO RISK CHECK (US-031) ────────────────────────
        # Must run AFTER sizing but BEFORE order placement.
        # Projects the new trade's risk into total portfolio risk to prevent
        # breaching the 15% NAV limit (the old check only looked at existing).
        proj_ok, proj_reason, max_safe_lots = self._check_projected_portfolio_risk(
            pair, sl_pips, lots
        )
        if not proj_ok:
            if max_safe_lots >= 0.01:
                # Downsize to fit within risk budget
                logger.info(
                    f"⚠ Risk budget: downsizing {pair} from {lots:.2f} to "
                    f"{max_safe_lots:.2f} lots. {proj_reason}"
                )
                lots = max_safe_lots
            else:
                # Can't fit even minimum lot — skip trade
                # Phase 26 (US-157): Log projected risk block
                try:
                    # Phase 30 (US-183): Use shared observer
                    self._get_observer().log_observation(
                        pair=pair,
                        category="portfolio_risk_block",
                        description=f"US-157: Projected risk block — {proj_reason}",
                        metadata={
                            "pair": pair,
                            "reason": proj_reason,
                            "gate": "projected_portfolio_risk",
                            "max_safe_lots": max_safe_lots,
                            "requested_lots": lots,
                        },
                    )
                except Exception as e:
                    logger.debug(f"Execution non-critical operation skipped: {e}")
                return ExecutionResult(
                    success=False,
                    error=f"Portfolio risk breach (projected): {proj_reason}",
                )

        # ── Phase 24 (US-148): Slippage-based position size reduction ──────
        if pair in self._slippage_history and len(self._slippage_history[pair]) >= 5:
            _avg_slip = sum(self._slippage_history[pair]) / len(self._slippage_history[pair])
            if _avg_slip > self._slippage_threshold_pips:
                _reduction = 1.0 - self._slippage_size_reduction
                _old_lots = lots
                lots = max(self.config.min_lot_size, lots * _reduction)
                logger.info(
                    f"US-148: Slippage reduction for {pair}: "
                    f"{_old_lots:.2f} → {lots:.2f} lots "
                    f"(avg slippage {_avg_slip:.1f} pips > {self._slippage_threshold_pips})"
                )
                # Phase 27 (US-165): Log slippage event for trending
                try:
                    # Phase 30 (US-183): Use shared observer
                    self._get_observer().log_observation(
                        pair=pair,
                        category="high_slippage",
                        description=(
                            f"US-165: High slippage on {pair} — "
                            f"avg {_avg_slip:.1f} pips, "
                            f"size reduced {_old_lots:.2f}→{lots:.2f}"
                        ),
                        metadata={
                            "pair": pair,
                            "avg_slippage_pips": round(_avg_slip, 2),
                            "threshold_pips": self._slippage_threshold_pips,
                            "sample_count": len(self._slippage_history[pair]),
                            "lot_reduction": round(_old_lots - lots, 4),
                        },
                    )
                except Exception as e:
                    logger.debug(f"Execution slippage tracking skipped: {e}")

        # ── SPREAD CHECK (US-053) ──────────────────────────────────────────
        spread_ok, spread_pips, spread_reason = self._check_spread_conditions(pair)
        if not spread_ok:
            return ExecutionResult(success=False, error=spread_reason)
        # Inject spread data into context for journal logging
        ctx["spread_pips"] = spread_pips

        # ── Phase 25 (US-152): Real-time correlation enforcement ──────────
        try:
            open_trades = self.monitor_open_trades(evaluate_exits=False)
            if open_trades:
                _CORR_THRESHOLD = 0.80
                # Known high-correlation pairs (static fallback)
                _CORR_PAIRS = {
                    "EUR_USD": {"GBP_USD", "EUR_GBP", "EUR_CHF"},
                    "GBP_USD": {"EUR_USD", "EUR_GBP", "GBP_JPY"},
                    "USD_JPY": {"USD_CHF", "EUR_JPY"},
                    "USD_CHF": {"USD_JPY", "EUR_CHF"},
                    "EUR_JPY": {"USD_JPY", "EUR_USD"},
                    "AUD_USD": {"NZD_USD", "AUD_NZD"},
                    "NZD_USD": {"AUD_USD", "AUD_NZD"},
                    "EUR_GBP": {"EUR_USD", "GBP_USD"},
                    "EUR_CHF": {"EUR_USD", "USD_CHF"},
                    "GBP_JPY": {"GBP_USD", "USD_JPY"},
                }
                _open_pairs = {t["pair"] for t in open_trades if t.get("pair")}
                _corr_set = _CORR_PAIRS.get(pair, set())
                _conflicting = _open_pairs & _corr_set
                if _conflicting:
                    _block_msg = (
                        f"US-152: Correlation block — {pair} has >{_CORR_THRESHOLD:.0%} "
                        f"correlation with open position(s): {', '.join(_conflicting)}"
                    )
                    logger.info(_block_msg)
                    # Phase 26 (US-162): Enrich with blocking pair P/L and duration
                    _blocking_detail = []
                    for _ot in open_trades:
                        _ot_pair = _ot.get("pair", "")
                        if _ot_pair in _conflicting:
                            _blocking_detail.append({
                                "pair": _ot_pair,
                                "unrealized_pl": _ot.get("unrealizedPL", _ot.get("unrealized_pl", 0)),
                                "units": _ot.get("units", 0),
                                "open_time": _ot.get("openTime", _ot.get("open_time", "")),
                            })
                    # Track conflict frequency

                    _conflict_key = f"{pair}_vs_{'|'.join(sorted(_conflicting))}"
                    self._correlation_block_counts[_conflict_key] = (
                        self._correlation_block_counts.get(_conflict_key, 0) + 1
                    )
                    try:
                        # Phase 30 (US-183): Use shared observer
                        self._get_observer().log_observation(
                            pair=pair,
                            category="correlation_block",
                            description=_block_msg,
                            metadata={
                                "blocked_pair": pair,
                                "conflicting_open": list(_conflicting),
                                "threshold": _CORR_THRESHOLD,
                                "blocking_pair_detail": _blocking_detail,
                                "conflict_frequency": self._correlation_block_counts.get(_conflict_key, 1),
                            },
                        )
                    except Exception as e:
                        logger.debug(f"Execution correlation check skipped: {e}")
                    return ExecutionResult(
                        success=False,
                        error=_block_msg,
                    )
        except Exception as _corr_err:
            logger.debug(f"US-152: Correlation check skipped: {_corr_err}")

        # US-075: Smart execution planning (log execution plan for larger positions)
        _smart_plan = None
        if getattr(self.config, "enable_smart_execution", False) and lots > 0:
            try:
                from src.scanner.automation.smart_execution import SmartExecutionEngine
                _se = SmartExecutionEngine(
                    strategy=getattr(self.config, "execution_strategy", "TWAP"),
                    window_seconds=getattr(self.config, "execution_window_minutes", 5.0) * 60,
                )
                if _se.should_use_smart_execution(lots, atr_pips=float(sl_pips or 0)):
                    _smart_plan = _se.plan_execution(
                        position_size_lots=lots,
                        pair=pair,
                        direction=direction,
                        market_data={"atr_pips": float(atr / _get_pip_value(pair, 0.0001)) if atr > 0 else 0},
                    )
                    logger.info(
                        "SmartExecution: %s plan for %s — %d slices, %.4f lots total",
                        _smart_plan.strategy, pair, _smart_plan.slice_count, _smart_plan.total_lots,
                    )
            except Exception as _se_err:
                logger.debug(f"Smart execution planning skipped: {_se_err}")

        # Phase 45 (US-285): Apply EWMA diversification adjustment to position size
        if lots > 0:
            lots = self._apply_diversification_adjustment(lots, pair)

        # Phase 49 (US-306): Apply compound size multiplier from spread/calendar/fitness gates
        _phase49_mult = ctx.get("_phase49_size_multiplier", 1.0)
        if _phase49_mult < 1.0 and lots > 0:
            _lots_before = lots
            lots = round(lots * _phase49_mult, 2)
            lots = max(lots, self.config.min_lot_size)  # Never go below min
            logger.info(
                "Phase 49 (US-306): %s lots adjusted %.4f → %.4f (×%.3f from gates)",
                pair, _lots_before, lots, _phase49_mult,
            )

        # Calculate SL/TP prices
        pip_value = _get_pip_value(pair)

        if direction.upper() == "LONG":
            sl_price = current_price - (sl_pips * pip_value)
            tp_price = current_price + (tp_pips * pip_value)
            units = round(lots * 100_000)  # M-10: round instead of int() to prevent truncation bias
        else:
            sl_price = current_price + (sl_pips * pip_value)
            tp_price = current_price - (tp_pips * pip_value)
            units = -round(lots * 100_000)  # M-10: round instead of int() to prevent truncation bias

        # ── Tier 6 (US-400): Aura Co-Cognition Query Gate ──
        # Pre-trade gate querying Aura about trader readiness before high-risk trades.
        # Non-blocking by design: if Aura is offline or slow (>25s), we proceed.
        if self._aura_gate is not None:
            try:
                # Build trade context for Aura query
                _aura_ctx = {
                    "pair": pair,
                    "direction": direction.upper(),
                    "confidence": confidence,
                    "weighted_vote_score": float(ctx.get("weighted_vote_score", 0.65)),
                    "portfolio_risk_pct": float(ctx.get("portfolio_risk_pct", 0.05)),
                    "drawdown_today_pct": float(ctx.get("_drawdown_pct", 0.0)),
                    "regime": str(ctx.get("volatility_regime", "NORMAL")).upper(),
                    "rr_ratio": round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0.0,
                }
                _aura_result = self._aura_gate.evaluate(_aura_ctx)

                if _aura_result.get("hard_veto"):
                    logger.warning(
                        "Tier 6 (US-400): AURA HARD VETO on %s %s: %s",
                        pair, direction, _aura_result.get("message", "unknown")
                    )
                    return ExecutionResult(
                        success=False,
                        error=f"AURA HARD VETO: {_aura_result.get('message', 'unknown')}",
                    )
                elif _aura_result.get("confidence_modifier", 0.0) != 0.0:
                    _old_conf_aura = confidence
                    _modifier = _aura_result["confidence_modifier"]
                    confidence = max(0.0, min(1.0, confidence + _modifier))
                    if not _aura_result.get("proceed"):
                        logger.info(
                            "Tier 6 (US-400): Aura caution on %s — confidence %.3f → %.3f (modifier %.3f)",
                            pair, _old_conf_aura, confidence, _modifier,
                        )
                    # Re-check minimum confidence after adjustment
                    if confidence < 0.45:
                        return ExecutionResult(
                            success=False,
                            error=f"BLOCKED: confidence {confidence:.3f} < 0.45 after Aura adjustment",
                        )
                elif _aura_result.get("response_received"):
                    logger.info(
                        "Tier 6 (US-400): Aura clears %s %s (readiness=%.2f)",
                        pair, direction, _aura_result.get("readiness_score", 0.0),
                    )
            except Exception as _aura_err:
                logger.debug(f"Tier 6: Aura gate evaluation error: {_aura_err}")
                # Non-blocking — errors don't prevent execution

        try:
            # Attempt order with rejection recovery (downsize by 50% on rejection)
            fill_status = "UNKNOWN"
            attempt_units = units
            attempt_lots = lots

            # Initialize broker
            if not self._init_broker():
                return ExecutionResult(success=False, error="Failed to initialize broker")

            # Create instrument object for broker interface
            from src.brokers.instrument import Instrument
            pip_value = _get_pip_value(pair)
            instrument = Instrument.fx(
                symbol=pair,
                broker_symbol=pair,
                pip_value=pip_value,
            )

            # US-013: Futures margin validation — check only for FUTURES instruments
            if self._margin_checker is not None and instrument.asset_class == "FUTURES":
                try:
                    account_nav = self.fetch_live_nav() or self.config.account_equity or 10000.0
                    # Convert lots to contracts for futures
                    contracts = int(round(attempt_lots))
                    if contracts > 0:
                        margin_passes = self._margin_checker.check_margin(
                            instrument=instrument,
                            contracts=contracts,
                            account_nav=account_nav,
                        )
                        if not margin_passes:
                            utilization = self._margin_checker.get_margin_utilization()
                            margin_msg = (
                                f"BLOCKED: Insufficient margin for {pair} "
                                f"(projected={contracts * instrument.margin_requirement:.2f}, "
                                f"max_allowed={account_nav * self._margin_checker.max_margin_pct:.2f}, "
                                f"current_used={utilization['total_margin_used']:.2f})"
                            )
                            logger.warning(f"US-013: {margin_msg}")
                            return ExecutionResult(success=False, error=margin_msg)
                        # Track the margin for this position (will be released when position closes)
                        self._margin_checker.track_margin(
                            position_id=f"{pair}_{direction}_{len(self._margin_checker.positions_margin)}",
                            instrument=instrument,
                            contracts=contracts,
                        )
                        logger.info(
                            f"US-013: Margin check passed for {pair} "
                            f"({contracts} contracts @ {instrument.margin_requirement:.2f} ea)"
                        )
                except Exception as e:
                    logger.warning(f"US-013: Margin check error (non-blocking): {e}")

            # Convert LONG/SHORT to BUY/SELL for broker API
            _broker_direction = "BUY" if direction.upper() == "LONG" else "SELL"

            # Phase 85 (US-P85-004): ExecutionRouter — log routing strategy (observational)
            _er = getattr(self, "_execution_router", None)
            if _er is not None:
                try:
                    _er_plan = _er.route_execution(
                        pair=pair, lot_size=attempt_lots, regime=regime_name,
                    )
                    logger.debug(
                        "Phase 85 (US-P85-004): ExecutionRouter — %s %s: strategy=%s slices=%d",
                        pair, direction, _er_plan.strategy, _er_plan.n_slices,
                    )
                except Exception as _er_err:
                    logger.debug("Phase 85: ExecutionRouter error: %s", _er_err)

            for order_attempt in range(self.config.max_order_attempts):
                order_result = self._broker.place_order(
                    instrument=instrument,
                    direction=_broker_direction,
                    quantity=abs(attempt_units),  # OANDA needs units (27000), not lots (0.27)
                    sl_price=round(sl_price, 5),
                    tp_price=round(tp_price, 5),
                    entry_price=current_price,
                )

                if order_result and order_result.status == "FILLED":
                    # Phase 84 (US-P84-004): Record API success
                    if _cb is not None:
                        try:
                            _cb.record_success()
                        except Exception:
                            pass
                    fill_price = order_result.fill_price
                    trade_id = order_result.trade_id

                    # Calculate slippage
                    pip_value = _get_pip_value(pair)
                    slippage = (fill_price - current_price) / pip_value if pip_value > 0 else 0.0
                    if direction.upper() == "SHORT":
                        slippage = -slippage
                    slippage = round(slippage, 1)

                    fill_status = "RETRIED" if order_attempt > 0 else "FULL"

                    # Log slippage warning if excessive
                    if abs(slippage) > self.config.high_slippage_alert_pips:
                        try:
                            # Phase 30 (US-183): Use shared observer
                            self._get_observer().log(
                                category="high_slippage",
                                detail=f"{pair} slippage {slippage:.1f} pips (fill {fill_price} vs expected {current_price})",
                                data={"pair": pair, "slippage_pips": slippage, "fill_price": fill_price},
                            )
                        except Exception as e:
                            logger.debug(f"Execution pair operation skipped: {e}")

                    # Phase 24 (US-148): Record slippage for rolling tracker
                    if pair not in self._slippage_history:
                        self._slippage_history[pair] = []
                    self._slippage_history[pair].append(abs(slippage))
                    if len(self._slippage_history[pair]) > self._slippage_window:
                        self._slippage_history[pair] = self._slippage_history[pair][-self._slippage_window:]
                    _avg_slip = sum(self._slippage_history[pair]) / len(self._slippage_history[pair])
                    if _avg_slip > self._slippage_threshold_pips:
                        logger.info(
                            f"US-148: {pair} avg slippage {_avg_slip:.1f} pips "
                            f"(>{self._slippage_threshold_pips}) over "
                            f"{len(self._slippage_history[pair])} trades — "
                            f"size reduction {self._slippage_size_reduction:.0%} will apply"
                        )
                        try:
                            # Phase 30 (US-183): Use shared observer
                            self._get_observer().log_observation(
                                pair=pair,
                                category="slippage_feedback",
                                description=(
                                    f"US-148: Rolling avg slippage {_avg_slip:.1f} pips "
                                    f"exceeds threshold {self._slippage_threshold_pips} — "
                                    f"sizing reduced by {self._slippage_size_reduction:.0%}"
                                ),
                                metadata={
                                    "avg_slippage_pips": round(_avg_slip, 2),
                                    "threshold": self._slippage_threshold_pips,
                                    "size_reduction": self._slippage_size_reduction,
                                    "trade_count": len(self._slippage_history[pair]),
                                },
                            )
                        except Exception as e:
                            logger.debug(f"Execution slippage tracking skipped: {e}")

                    # Log to journal AFTER successful fill (not before)
                    self._log_trade(
                        pair=pair,
                        direction=direction,
                        confidence=confidence,
                        lots=attempt_lots,
                        entry=fill_price,
                        sl=sl_price,
                        tp=tp_price,
                        trade_id=trade_id,
                        analysis_context=analysis_context,
                    )

                    # Phase 56 (US-349): TCA fill feedback — record fill metrics to ExecutionQualityTracker
                    if self._exec_quality_tracker is not None:
                        try:
                            import time as _time_mod
                            _fill_pip_value = _get_pip_value(pair, 0.0001)
                            _fill_slippage_abs = abs(slippage) if slippage is not None else 0.0
                            _fill_status_str = "filled" if fill_status == "FULL" else (
                                "partial" if fill_status == "RETRIED" else "rejected"
                            )
                            self._exec_quality_tracker.track_fill(
                                pair=pair,
                                expected_price=float(current_price),
                                fill_price=float(fill_price),
                                slippage_pips=float(_fill_slippage_abs),
                                fill_status=_fill_status_str,
                                execution_ms=0.0,  # Timing data not available at this point
                                regime=str(analysis_context.get("volatility_regime", "NORMAL"))
                                if analysis_context else "NORMAL",
                            )
                            self._exec_quality_tracker.save_state()
                            logger.debug(
                                "Phase 56 (US-349): TCA recorded — %s fill=%.5f expected=%.5f slip=%.1fpips",
                                pair, fill_price, current_price, _fill_slippage_abs,
                            )
                        except Exception as _eqt_fill_err:
                            logger.debug(f"Phase 56: ExecutionQualityTracker track_fill error: {_eqt_fill_err}")

                    # Phase 52 (US-323): Register trade with tranche tracker for SL management
                    if self._tranche_tracker is not None:
                        try:
                            self._tranche_tracker.register_trade(
                                trade_id=str(trade_id),
                                pair=pair,
                                entry_price=fill_price,
                                original_sl=sl_price,
                                direction=direction.upper(),
                            )
                            logger.info(
                                "Phase 52 (US-323): Registered %s #%s with tranche tracker (entry=%.5f, sl=%.5f)",
                                pair, trade_id, fill_price, sl_price,
                            )
                        except Exception as _tt_reg_err:
                            logger.debug(f"Phase 52: Tranche tracker registration error: {_tt_reg_err}")

                    # Phase 81 (US-P81-002): Register with PositionTimeoutManager post-fill
                    if self._position_timeout_mgr is not None:
                        try:
                            _pt_regime = (regime_name or "NORMAL").lower()
                            self._position_timeout_mgr.register_trade(
                                trade_id=str(trade_id),
                                regime=_pt_regime,
                                confidence=float(confidence),
                                pair=pair,
                                direction=direction.upper(),
                            )
                            logger.info(
                                "Phase 81 (US-P81-002): Registered %s #%s with PositionTimeoutManager (regime=%s)",
                                pair, trade_id, _pt_regime,
                            )
                        except Exception as _pt_reg_err:
                            logger.debug("Phase 81: PositionTimeout register_trade error: %s", _pt_reg_err)

                    # Log aggressive scaling if applied
                    if regime_scale > 1.0:
                        logger.info(
                            f"Trade executed with AGGRESSIVE SCALING: {regime_name} vol, "
                            f"{regime_scale:.2f}x scale, {attempt_lots:.2f} lots"
                        )

                    return ExecutionResult(
                        success=True,
                        trade_id=trade_id,
                        fill_price=fill_price,
                        units=abs(attempt_units),
                        lots=attempt_lots,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        sl_pips=sl_pips,
                        tp_pips=tp_pips,
                        risk_pct=risk_pct,
                        confidence_level=conf_level,
                        regime_scale=regime_scale,
                        regime_name=regime_name,
                        aggressive_scaling_reason=aggressive_reason,
                        fill_status=fill_status,
                        slippage_pips=slippage,
                    )
                else:
                    # Order rejected — try downsizing by 50% on first rejection
                    if order_attempt < self.config.max_order_attempts - 1:
                        new_lots = max(self.config.min_lot_size, attempt_lots * self.config.rejection_downsize_factor)
                        logger.warning(
                            f"Order rejected for {pair}: downsizing from {attempt_lots:.2f} to {new_lots:.2f} lots. "
                            f"Error: {order_result.error_message}"
                        )
                        attempt_lots = new_lots
                        attempt_units = int(new_lots * 100_000) * (1 if direction.upper() == "LONG" else -1)
                        continue
                    else:
                        # Second rejection — give up
                        return ExecutionResult(
                            success=False,
                            error=f"Order rejected twice: {order_result.error_message}",
                            fill_status="REJECTED",
                )

        except Exception as e:
            # Phase 84 (US-P84-004): Record API failure on exception
            if _cb is not None:
                try:
                    _cb.record_failure()
                except Exception:
                    pass
            return ExecutionResult(success=False, error=str(e))

    def execute_trades(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[ExecutionResult]:
        """Execute multiple trades with daily limit enforcement.

        Args:
            trades: List of trade dicts with keys:
                - pair: Instrument name
                - direction: "LONG" or "SHORT"
                - confidence: Model confidence
                - current_price: Current price
                - atr: ATR value
                - sl_pips (optional): Stop loss pips
                - tp_pips (optional): Take profit pips
                - recommended_lots (optional): Position size

        Returns:
            List of ExecutionResult for each trade
        """
        if not trades:
            return []

        # Check daily limit
        _, trades_today, trades_remaining = self.get_account_status()

        if trades_remaining <= 0:
            logger.warning(f"Daily trade limit reached ({trades_today}/{self.config.max_trades_per_day})")
            return [ExecutionResult(success=False, error="Daily limit reached") for _ in trades]

        # Limit trades to remaining slots
        if len(trades) > trades_remaining:
            logger.warning(f"Executing {trades_remaining} of {len(trades)} trades (daily limit)")
            trades = trades[:trades_remaining]

        results = []
        for trade in trades:
            # Extract volatility regime and meta-confidence from analysis context
            ctx = trade.get("analysis_context") or {}
            vol_regime = None
            meta_conf = None
            regime_str = str(ctx.get("volatility_regime", "") or "").upper()
            regime_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}
            if regime_str in regime_map:
                vol_regime = regime_map[regime_str]
            meta_conf = ctx.get("tcn_confidence") or ctx.get("weighted_vote_score")

            result = self.execute_trade(
                pair=trade.get("pair", ""),
                direction=trade.get("direction", "HOLD"),
                confidence=trade.get("confidence", 0.5),
                current_price=trade.get("current_price", 0.0),
                atr=trade.get("atr", 0.0),
                sl_pips=trade.get("sl_pips"),
                tp_pips=trade.get("tp_pips"),
                lots=trade.get("recommended_lots"),
                volatility_regime=vol_regime,
                meta_confidence=float(meta_conf) if meta_conf is not None else None,
                analysis_context=trade.get("analysis_context"),
            )
            results.append(result)

        return results

    def _log_trade(
        self,
        pair: str,
        direction: str,
        confidence: float,
        lots: float,
        entry: float,
        sl: float,
        tp: float,
        trade_id: str,
        analysis_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log trade to memory client with full gate/agent context."""
        self._init_memory_client()

        if self._memory_client is None:
            return

        try:
            nav = self._cached_nav or self.config.account_equity or 10000.0
            record: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instrument": pair,
                "direction": direction,
                "confidence": confidence,
                "lots": lots,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "trade_id": trade_id,
                "model": "scanner",
                "account_balance": nav,
            }
            if analysis_context:
                record["metadata"] = analysis_context
            self._memory_client.log_trade(record)
        except Exception as e:
            logger.debug(f"Failed to log trade: {e}")

        # Also append to trade_journal.json for RL feedback
        self._append_journal_entry(trade_id, pair, direction, confidence, entry, sl, tp, lots, analysis_context)

    def _append_journal_entry(
        self,
        trade_id: str,
        pair: str,
        direction: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        lots: float,
        analysis_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append trade entry to trained_data/trade_journal.json for RL feedback."""
        import json
        from pathlib import Path

        journal_path = Path("trained_data/trade_journal_rl.json")
        journal_path.parent.mkdir(parents=True, exist_ok=True)

        entries: List[Dict[str, Any]] = []
        if journal_path.exists():
            try:
                from src.scanner.automation.safe_json import safe_json_read
                entries = safe_json_read(journal_path, default=[])
            except ImportError:
                try:
                    entries = json.loads(journal_path.read_text())
                except Exception as e:
                    logger.debug("US-178: Journal read failed in _log_trade: %s", e)
                    entries = []
            if not isinstance(entries, list):
                logger.warning("Trade journal corrupted (not a list), resetting: %s", type(entries))
                entries = []
            else:
                # H-5: Schema validation — filter out malformed entries
                _required_keys = {"pair", "direction", "timestamp"}
                _valid = []
                for _entry in entries:
                    if isinstance(_entry, dict) and _required_keys.issubset(_entry.keys()):
                        _valid.append(_entry)
                    else:
                        logger.warning("H-5: Skipping malformed journal entry: %s", type(_entry))
                if len(_valid) < len(entries):
                    logger.warning("H-5: Filtered %d malformed entries from journal", len(entries) - len(_valid))
                entries = _valid

        ctx = analysis_context or {}

        # Record setup to episodic memory for pattern suppression
        _episode_id = ""
        if self._episodic_memory is not None:
            try:
                _regime = ctx.get("volatility_regime", "unknown")
                if isinstance(_regime, int):
                    _regime_map = {0: "NORMAL", 1: "TRENDING", 2: "RANGING", 3: "VOLATILE"}
                    _regime = _regime_map.get(_regime, "unknown")
                else:
                    _regime = str(_regime).upper() if _regime else "unknown"
                _session = ctx.get("session", ctx.get("trading_session", "unknown"))
                _session = str(_session).upper() if _session else "unknown"
                _news_risk = float(ctx.get("news_risk_score", 0.0))
                _uncertainty = float(ctx.get("uncertainty_score", 0.0))
                _atr_norm = float(ctx.get("atr_normalized", 0.0))
                _spread = float(ctx.get("spread_pips", 0.0))
                _rr_ratio = float(ctx.get("rr_ratio", 0.0))

                _episode_id = self._episodic_memory.record_setup(
                    pair=pair,
                    direction=direction,
                    regime=_regime,
                    session=_session,
                    news_risk_score=_news_risk,
                    uncertainty_score=_uncertainty,
                    atr_normalized=_atr_norm,
                    spread_pips=_spread,
                    confidence=confidence,
                    weighted_vote_score=float(ctx.get("weighted_vote_score", 0.0)),
                    rr_ratio=_rr_ratio,
                )
                logger.debug(f"Episodic memory: Recorded setup {_episode_id} for {pair} {direction}")
            except Exception as _em_err:
                logger.debug(f"Episodic memory record_setup error: {_em_err}")

        entry_record = {
            "trade_id": trade_id,
            "episode_id": _episode_id,  # Store episode_id for later outcome recording
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "direction": direction,
            "confidence": confidence,
            "entry_price": entry,
            "sl_price": sl,
            "tp_price": tp,
            "lots": lots,
            # US-014: Multi-broker and asset class support
            "broker": ctx.get("broker", "oanda"),  # 'oanda' or 'ibkr'
            "asset_class": ctx.get("asset_class", "FX"),  # 'FX' or 'FUTURES'
            "contract_month": ctx.get("contract_month", None),  # YYYYMM for futures, null for FX
            "tick_value": ctx.get("tick_value", None),  # null for FX, float for futures
            "multiplier": ctx.get("multiplier", None),  # null for FX, float for futures
            "gates": {
                "momentum_passed": ctx.get("momentum_passed"),
                "confidence_passed": ctx.get("confidence_passed"),
                "risk_passed": ctx.get("risk_passed"),
                "gate_summary": ctx.get("gate_summary"),
            },
            "agents": {
                "agent_votes": ctx.get("agent_votes"),
                "agent_total": ctx.get("agent_total"),
                "agent_passed": ctx.get("agent_passed"),
                "weighted_vote_score": ctx.get("weighted_vote_score"),
                "agent_reasons": ctx.get("agent_reasons", []),
            },
            "regime": {
                "volatility_regime": ctx.get("volatility_regime"),
                "atr_pips": ctx.get("atr_pips"),
                "uncertainty_score": ctx.get("uncertainty_score"),
                "model_disagreement": ctx.get("model_disagreement"),
                "regime_risk_modifier": ctx.get("regime_risk_modifier"),
                "predictive_size_modifier": ctx.get("predictive_size_modifier"),
                "regime_atr_multipliers": ctx.get("regime_atr_multipliers"),
            },
            "spread_pips": ctx.get("spread_pips"),
            "group_momentum": ctx.get("group_momentum"),
            "gate_details": {
                "scores": ctx.get("scores", {}),
                "core_score": ctx.get("core_score"),
                "final_score": ctx.get("final_score"),
            },
            # Tier 6 Phase 2: Real Ridge features for MAML training (replaces proxy)
            "ridge_features": ctx.get("ridge_features"),
            "outcome": None,  # Filled by sync_closed_trades_rl
            "fill_price": entry,  # Actual fill price from OANDA
            "expected_price": ctx.get("current_price", entry),
            "slippage_pips": round(
                (entry - float(ctx.get("current_price", entry)))
                / _get_pip_value(pair, 0.0001), 1
            ) if ctx.get("current_price") else 0.0,
        }

        # Deduplicate by trade_id
        entries = [e for e in entries if e.get("trade_id") != trade_id]
        entries.append(entry_record)

        try:
            from src.scanner.automation.safe_json import safe_json_write
            safe_json_write(journal_path, entries)
        except ImportError:
            # Phase 32 (US-193): Fallback with atomic temp+rename pattern
            _data = json.dumps(entries, indent=2, sort_keys=True, default=str)
            import tempfile
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(journal_path.parent), suffix=".tmp"
                )
                with os.fdopen(tmp_fd, "w") as _f:
                    _f.write(_data)
                    _f.flush()
                    os.fsync(_f.fileno())
                os.rename(tmp_path, str(journal_path))
            except OSError as _e:
                logger.warning("Atomic journal write failed, direct write fallback: %s", _e)
                journal_path.write_text(_data)
        logger.debug(f"Journal entry appended for trade #{trade_id}")

    def fetch_actual_win_rate(self) -> Tuple[float, int]:
        """Fetch actual win rate from broker closed trades.

        Returns:
            Tuple of (win_rate, total_trades)
        """
        if not self._init_broker():
            return 0.0, 0

        try:
            trades = self._broker.get_trades()

            if not trades:
                return 0.0, 0

            # Note: get_trades() returns TradeInfo objects without realized_pnl
            # For now, assume historical trades are available and estimate win rate
            # based on closed positions. A more complete implementation would need
            # TradeInfo to include realized P/L or a separate method for closed trades.

            # As a workaround, count trades (incomplete but better than nothing)
            total = len(trades)

            # Without realized PnL in TradeInfo, we can't accurately calculate win rate
            # Return 0.5 (neutral) until TradeInfo is extended
            logger.debug(f"Actual trading performance estimation: {total} recent trades available")
            return 0.5, total

        except Exception as e:
            logger.debug(f"Could not fetch actual win rate: {e}")
            return 0.0, 0

    def sync_journal(self) -> int:
        """Sync trade journal with broker and check for retraining.

        Returns:
            Number of trades synced
        """
        if not self._init_broker():
            return 0

        try:
            from src.utils.trade_journal import TradeJournal

            journal = TradeJournal()

            # For backward compatibility: if broker is legacy OANDA client, use old method
            if hasattr(self._broker, '_config'):
                # Legacy OANDA client
                updated = journal.update_from_oanda(self._broker)
            else:
                # New BrokerClient interface - get trades and sync
                trades = self._broker.get_trades()
                # Note: update_from_oanda expects OANDA client; this is a simplified version
                # A full implementation would need to extend TradeJournal to handle BrokerClient
                updated = len(trades) if trades else 0
                logger.debug(f"Broker provided {updated} recent trade(s) for journal sync")

            if updated > 0:
                logger.info(f"Journal synced: {updated} trade(s) updated")

            return updated

        except ImportError:
            logger.debug("TradeJournal not available")
            return 0
        except Exception as e:
            logger.debug(f"Journal sync failed: {e}")
            return 0

    def set_ohlc_cache(self, ohlc_by_pair: Dict[str, Any]) -> None:
        """Inject OHLC DataFrames from the Scanner's raw snapshots.

        Called by the scan loop after fetching pair data so that evaluate_exits()
        can use real price history instead of degraded synthetic arrays.

        Args:
            ohlc_by_pair: Dict mapping pair name to a DataFrame with at least
                'close', 'high', 'low' columns.
        """
        import time as _time
        self._ohlc_cache = ohlc_by_pair
        self._ohlc_cache_timestamp = _time.time()
        logger.debug(
            "Phase 55 (US-340): OHLC cache updated — %d pairs, %s",
            len(ohlc_by_pair),
            ", ".join(sorted(ohlc_by_pair.keys())[:5]) + ("..." if len(ohlc_by_pair) > 5 else ""),
        )

    def evaluate_exits(
        self,
        trade_statuses: List[Dict[str, Any]],
        regime_name: str = "NORMAL",
        current_confidence: float = 0.5,
        atr_values: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Phase 44 (US-279): Evaluate adaptive exit strategies for open trades.

        Uses AdaptiveExitManager to determine if trades should be closed,
        partially closed, or have stops adjusted based on regime, confidence,
        and price action.

        Note: Full exit evaluation requires OHLC data (price arrays, bar indices)
        which are not currently available in monitor_open_trades. This method
        gracefully degrades when those are unavailable and logs warnings.

        Args:
            trade_statuses: Output from monitor_open_trades()
            regime_name: Current market regime (LOW/NORMAL/HIGH/EXTREME)
            current_confidence: Current calibrated confidence (0-1)
            atr_values: Dict of pair -> ATR value. If None, uses fallback.

        Returns:
            List of exit action dicts with trade_id, action, reason, details
        """
        if self._adaptive_exit_manager is None:
            return []

        exit_actions = []
        for status in trade_statuses:
            try:
                from src.scanner.adaptive_exits import TradeContext
                import numpy as np

                pair = status.get("pair", "")
                entry = status.get("entry", 0.0)
                direction = status.get("direction", "LONG")
                sl_price = status.get("sl_price", 0.0)
                tp_price = status.get("tp_price", 0.0)
                time_in_minutes = status.get("time_in_minutes", 0)
                trade_id = status.get("trade_id")

                # Safely get units, default to 1 to avoid division by zero
                units = status.get("units", 0)
                safe_units = max(abs(units), 1)

                # Estimate current price from unrealized P/L
                unrealized_pl = status.get("unrealized_pl", 0.0)
                if direction == "LONG":
                    current_price = entry + (unrealized_pl / safe_units)
                    highest_price = max(entry, current_price)
                    lowest_price = min(entry, current_price)
                else:  # SHORT
                    current_price = entry - (unrealized_pl / safe_units)
                    highest_price = max(entry, current_price)
                    lowest_price = min(entry, current_price)

                # Get ATR or estimate from SL distance
                atr = (atr_values or {}).get(pair, 0.0)
                if atr <= 0:
                    if sl_price > 0:
                        atr = abs(entry - sl_price)
                    else:
                        atr = entry * 0.001
                atr = max(atr, 1e-6)

                # Convert direction string
                trade_direction = "BUY" if direction == "LONG" else "SELL"

                # Approximate bars held (assume 15-min bars)
                bars_held = max(1, time_in_minutes // 15)

                # Build TradeContext with price arrays.
                # Prefer real OHLC data from Scanner._raw_snapshots if available
                # (set via set_ohlc_cache() from the scan loop). Fall back to
                # synthetic arrays when real data is unavailable.
                # Phase 55 (US-340): Check cache freshness — warn if stale
                import time as _time
                if self._ohlc_cache_timestamp is not None:
                    _cache_age = _time.time() - self._ohlc_cache_timestamp
                    if _cache_age > 600:  # > 10 minutes (2 scan cycles)
                        logger.warning(
                            "Phase 55 (US-340): OHLC cache is %.0fs stale for %s — "
                            "Chandelier exit may use outdated data",
                            _cache_age, pair,
                        )
                price_array_size = 50
                _used_real_ohlc = False

                if self._ohlc_cache and pair in self._ohlc_cache:
                    try:
                        _ohlc_df = self._ohlc_cache[pair]
                        if hasattr(_ohlc_df, "tail") and len(_ohlc_df) >= 10:
                            _tail = _ohlc_df.tail(price_array_size)
                            prices_close = _tail["close"].values.astype(np.float64)
                            prices_high = _tail["high"].values.astype(np.float64)
                            prices_low = _tail["low"].values.astype(np.float64)
                            price_array_size = len(prices_close)
                            _used_real_ohlc = True
                    except Exception as _ohlc_err:
                        logger.debug(f"Phase 44: OHLC cache read failed for {pair}: {_ohlc_err}")

                if not _used_real_ohlc:
                    # Phase 55 (US-340): OHLC cache wiring completed — this fallback
                    # should only trigger on first scan cycle before snapshots are
                    # available, or if a pair has no cached data.
                    logger.debug(
                        "Phase 44: Using synthetic price arrays for %s — "
                        "chandelier exit accuracy is degraded without real OHLC data",
                        pair,
                    )
                    prices_close = np.full(price_array_size, current_price, dtype=np.float64)
                    prices_high = np.full(price_array_size, max(highest_price, current_price), dtype=np.float64)
                    prices_low = np.full(price_array_size, min(lowest_price, current_price), dtype=np.float64)

                # Set the last price to actual current
                prices_close[-1] = current_price
                prices_high[-1] = max(highest_price, current_price)
                prices_low[-1] = min(lowest_price, current_price)

                ctx = TradeContext(
                    entry_price=entry,
                    entry_time_bar=max(0, price_array_size - bars_held - 1),
                    current_bar=price_array_size - 1,
                    direction=trade_direction,
                    prices_close=prices_close,
                    prices_high=prices_high,
                    prices_low=prices_low,
                    sl_pips=abs(entry - sl_price) if sl_price > 0 else 15.0,
                    tp_pips=abs(tp_price - entry) if tp_price > 0 else 20.0,
                    atr_current=atr,
                    current_price=current_price,
                    highest_price_since_entry=highest_price,
                    lowest_price_since_entry=lowest_price,
                    initial_confidence=max(current_confidence, 0.50),
                    current_confidence=current_confidence,
                    regime_name=regime_name,
                    tranches_closed=[],
                )

                action = self._adaptive_exit_manager.evaluate_exit(ctx)

                if action.action != "HOLD":
                    exit_actions.append({
                        "trade_id": trade_id,
                        "pair": pair,
                        "action": action.action,
                        "reason": action.reason,
                        "strategy": action.strategy,
                        "new_stop": action.new_sl_price,
                        "close_fraction": action.partial_close_ratio,
                        "confidence": action.confidence,
                        "urgency": action.urgency,
                        "details": action.metadata,
                    })
                    logger.info(
                        "Phase 44: Exit signal for %s trade %s: %s (%s) — %s",
                        pair, trade_id, action.action, action.strategy, action.reason,
                    )

            except ValueError as e:
                logger.debug(f"Phase 44: TradeContext validation failed for trade {status.get('trade_id')}: {e}")
            except Exception as e:
                logger.warning(f"Phase 44: Exit evaluation failed for trade {status.get('trade_id')}: {e}")

        return exit_actions

    def monitor_open_trades(self, evaluate_exits: bool = True) -> List[Dict[str, Any]]:
        """Monitor open trades and return status for each.

        Args:
            evaluate_exits: If True (default), run adaptive exit evaluation and
                act on exit signals. Set to False for read-only callers that only
                need position status (risk checks, correlation checks) to avoid
                unintended side effects like closing trades.

        Returns:
            List of dicts with trade status: pair, direction, entry, current_pl,
            distance_to_sl_pips, distance_to_tp_pips, time_in_trade_minutes.
        """
        if not self._init_broker():
            return []

        try:
            # Get open positions from broker
            positions = self._broker.get_open_positions()
        except Exception as e:
            logger.warning(f"Failed to fetch open positions: {e}")
            return []

        statuses = []
        for pos in positions:
            pair = pos.instrument
            pip_value = _get_pip_value(pair)
            units = int(pos.quantity)
            direction = pos.direction  # Already "BUY" or "SELL" from PositionInfo
            entry = pos.avg_price
            unrealized_pl = pos.unrealized_pnl

            # Note: PositionInfo doesn't include SL/TP prices directly
            # For now, set to 0 (unknown). A more complete implementation would
            # need to fetch or track these separately.
            sl_price = 0.0
            tp_price = 0.0

            # Calculate distances in pips (if SL/TP available)
            if sl_price > 0:
                sl_dist_pips = abs(entry - sl_price) / pip_value
            else:
                sl_dist_pips = 0
            if tp_price > 0:
                tp_dist_pips = abs(tp_price - entry) / pip_value
            else:
                tp_dist_pips = 0

            # Time in trade - not available in PositionInfo
            # Approximate as 0 (unknown) or implement separate tracking
            time_in_minutes = 0

            statuses.append({
                "trade_id": pair,  # Use pair as ID since PositionInfo doesn't have trade_id
                "pair": pair,
                "direction": direction,
                "units": units,
                "entry": entry,
                "unrealized_pl": unrealized_pl,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_dist_pips": round(sl_dist_pips, 1),
                "tp_dist_pips": round(tp_dist_pips, 1),
                "time_in_minutes": time_in_minutes,
            })

        # Phase 44 (US-279): Run adaptive exit evaluation on collected statuses
        # Only evaluate and act on exits when explicitly requested (default True).
        # Read-only callers (risk checks, correlation checks) pass evaluate_exits=False
        # to avoid side effects like closing trades during pre-trade validation.
        if evaluate_exits and self._adaptive_exit_manager is not None and statuses:
            try:
                exit_actions = self.evaluate_exits(statuses)
                for status in statuses:
                    tid = status.get("trade_id")
                    matching = [a for a in exit_actions if a.get("trade_id") == tid]
                    if matching:
                        status["exit_signal"] = matching[0]

                # Phase 44 fix: Actually act on exit signals (close trades / move stops)
                if exit_actions:
                    self._act_on_exit_signals(exit_actions, statuses)
            except Exception as e:
                logger.warning(f"Phase 44: Adaptive exit evaluation failed: {e}")

        # Position timeout: regime-dependent time-decay exits
        # Evaluates each open trade against its regime's max holding period
        # and confidence decay schedule. Adds exit signals for timed-out trades.
        if evaluate_exits and self._position_timeout is not None and statuses:
            try:
                ptm = self._position_timeout
                ptm.tick()  # Advance bar counter each monitoring cycle

                for status in statuses:
                    tid = status.get("trade_id", "")
                    pair = status.get("pair", "")

                    # Auto-register trades not yet tracked by the timeout manager
                    if tid and tid not in ptm._positions:
                        regime = getattr(self, "_last_regime_name", "normal")
                        ptm.register_trade(
                            trade_id=tid,
                            regime=regime,
                            confidence=0.5,
                            pair=pair,
                            direction=status.get("direction", ""),
                        )

                    # Evaluate timeout
                    pnl_pips = status.get("unrealized_pl", 0.0)
                    result = ptm.evaluate(tid, pnl_pips=pnl_pips)

                    if result.should_exit:
                        # Attach timeout exit signal (same shape as adaptive exits)
                        timeout_signal = {
                            "trade_id": tid,
                            "pair": pair,
                            "action": "EXIT_FULL",
                            "reason": f"Position timeout: {result.exit_reason}",
                            "strategy": "position_timeout",
                            "confidence_decay": result.confidence_decay,
                            "bars_open": result.bars_open,
                        }
                        status["timeout_exit_signal"] = timeout_signal
                        logger.warning(
                            "PositionTimeout EXIT: %s (%s) — %s",
                            tid, pair, result.exit_reason,
                        )

                # Collect timeout exits and act on them
                timeout_exits = [
                    s["timeout_exit_signal"]
                    for s in statuses
                    if "timeout_exit_signal" in s
                ]
                if timeout_exits:
                    self._act_on_exit_signals(timeout_exits, statuses)

                # Clean up closed trades from timeout tracker
                open_ids = {s.get("trade_id") for s in statuses}
                for tracked_id in list(ptm._positions.keys()):
                    if tracked_id not in open_ids:
                        ptm.close_trade(tracked_id)

            except Exception as e:
                logger.warning(f"PositionTimeout evaluation failed: {e}")

        return statuses

    def _act_on_exit_signals(
        self,
        exit_actions: List[Dict[str, Any]],
        statuses: List[Dict[str, Any]],
    ) -> List[str]:
        """Phase 44 (US-279): Execute adaptive exit signals on OANDA.

        Processes exit actions produced by evaluate_exits() and actually closes
        trades, takes partial profit, or tightens stops via the OANDA API.

        Actions handled:
        - EXIT_FULL: Close the entire trade via close_trade_oanda()
        - TAKE_PARTIAL: Close a fraction of the position (units * close_fraction)
        - TIGHTEN_SL: Move stop loss tighter via modify_sl_oanda()

        Args:
            exit_actions: List of exit action dicts from evaluate_exits().
            statuses: List of trade status dicts (used to look up units).

        Returns:
            List of human-readable action messages.
        """
        import requests
        import os

        messages = []
        # Build a lookup from trade_id -> status for unit info
        status_by_id = {s.get("trade_id"): s for s in statuses}

        for action in exit_actions:
            trade_id = action.get("trade_id")
            pair = action.get("pair", "unknown")
            action_type = action.get("action", "")
            reason = action.get("reason", "")
            strategy = action.get("strategy", "")

            if not trade_id:
                logger.warning("Phase 44: Exit action missing trade_id, skipping: %s", action)
                continue

            try:
                if action_type == "EXIT_FULL":
                    success = self.close_trade_oanda(str(trade_id))
                    if success:
                        msg = (
                            f"Phase 44 EXIT_FULL: {pair} #{trade_id} closed "
                            f"({strategy}: {reason})"
                        )
                        logger.info(msg)
                        messages.append(msg)
                    else:
                        logger.warning(
                            "Phase 44: Failed to close trade %s %s (%s)",
                            pair, trade_id, strategy,
                        )

                elif action_type == "TAKE_PARTIAL":
                    close_fraction = action.get("close_fraction", 0.5)
                    status = status_by_id.get(trade_id, {})
                    total_units = abs(status.get("units", 0))

                    if total_units <= 0:
                        logger.warning(
                            "Phase 44: Cannot take partial profit on %s #%s — no units info",
                            pair, trade_id,
                        )
                        continue

                    close_units = max(1, int(total_units * close_fraction))

                    # OANDA partial close: PUT /trades/{id}/close with {"units": "N"}
                    token = os.getenv("OANDA_API_TOKEN", "")
                    acct = os.getenv("OANDA_ACCOUNT_ID", "")
                    base = self._oanda_base_url
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    }

                    resp = self._retry_oanda(
                        requests.put,
                        f"{base}/v3/accounts/{acct}/trades/{trade_id}/close",
                        headers=headers,
                        json={"units": str(close_units)},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        msg = (
                            f"Phase 44 TAKE_PARTIAL: {pair} #{trade_id} closed "
                            f"{close_units}/{total_units} units ({close_fraction:.0%}) "
                            f"({strategy}: {reason})"
                        )
                        logger.info(msg)
                        messages.append(msg)

                        # Phase 52 (US-323): Tranche tracker — move SL after partial close
                        if self._tranche_tracker is not None:
                            try:
                                _metadata = action.get("metadata", {})
                                _tranche_idx = int(_metadata.get("tranche_index", 0))
                                _current_r = float(_metadata.get("tranche_config", {}).get("r_target", 1.0))
                                _sl_adj = self._tranche_tracker.record_tranche_close(
                                    str(trade_id), tranche_idx=_tranche_idx, r_at_close=_current_r,
                                )
                                if _sl_adj.should_move:
                                    _sl_moved = self.modify_sl_oanda(str(trade_id), _sl_adj.new_sl)
                                    if _sl_moved:
                                        logger.info(
                                            "Phase 52 (US-323): %s #%s SL moved to %.5f (%s)",
                                            pair, trade_id, _sl_adj.new_sl, _sl_adj.reason,
                                        )
                                        messages.append(
                                            f"Phase 52 TRANCHE_SL: {pair} #{trade_id} "
                                            f"SL → {_sl_adj.new_sl:.5f} ({_sl_adj.adjustment_type})"
                                        )
                                    else:
                                        logger.warning(
                                            "Phase 52 (US-323): %s #%s tranche SL move FAILED — keeping existing SL",
                                            pair, trade_id,
                                        )
                            except Exception as _tt_err:
                                logger.debug(f"Phase 52: Tranche tracker SL move error: {_tt_err}")

                    else:
                        logger.warning(
                            "Phase 44: Partial close failed for %s #%s: %s",
                            pair, trade_id, resp.text,
                        )

                elif action_type == "TIGHTEN_SL":
                    new_stop = action.get("new_stop")
                    if new_stop is not None and new_stop > 0:
                        success = self.modify_sl_oanda(str(trade_id), float(new_stop))
                        if success:
                            msg = (
                                f"Phase 44 TIGHTEN_SL: {pair} #{trade_id} SL moved to "
                                f"{new_stop:.5f} ({strategy}: {reason})"
                            )
                            logger.info(msg)
                            messages.append(msg)
                        else:
                            logger.warning(
                                "Phase 44: SL modification failed for %s #%s",
                                pair, trade_id,
                            )
                    else:
                        logger.debug(
                            "Phase 44: TIGHTEN_SL for %s #%s has no new_stop price, skipping",
                            pair, trade_id,
                        )

                else:
                    logger.debug(
                        "Phase 44: Unknown exit action type '%s' for %s #%s",
                        action_type, pair, trade_id,
                    )

            except Exception as e:
                logger.warning(
                    "Phase 44: Error executing %s on %s #%s: %s",
                    action_type, pair, trade_id, e,
                )

        return messages

    def _batch_fetch_prices(self, pairs: List[str]) -> Dict[str, float]:
        """Phase 30 (US-180): Fetch mid prices for multiple pairs in a single OANDA call.

        Eliminates N+1 pricing API calls by batching instruments into one request.

        Args:
            pairs: List of instrument names (e.g. ["EUR_USD", "GBP_USD"]).

        Returns:
            Dict mapping pair → mid price. Missing pairs are omitted.
        """
        if not pairs:
            return {}

        import requests
        import os

        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = self._oanda_base_url
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Deduplicate pairs
        unique_pairs = list(dict.fromkeys(pairs))
        instruments = ",".join(unique_pairs)

        prices_map: Dict[str, float] = {}
        try:
            resp = self._retry_oanda(
                requests.get,
                f"{base}/v3/accounts/{acct}/pricing?instruments={instruments}",
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            for p in resp.json().get("prices", []):
                instrument = p.get("instrument", "")
                asks = p.get("asks", [{}])
                bids = p.get("bids", [{}])
                if asks and bids:
                    ask_price = float(asks[0].get("price", 0))
                    bid_price = float(bids[0].get("price", 0))
                    if ask_price > 0 and bid_price > 0:
                        prices_map[instrument] = (ask_price + bid_price) / 2
        except Exception as e:
            logger.debug("US-180: Batch price fetch failed, falling back to individual: %s", e)
            # Fallback: fetch individually
            for pair in unique_pairs:
                try:
                    resp = requests.get(
                        f"{base}/v3/accounts/{acct}/pricing?instruments={pair}",
                        headers=headers,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    price_list = resp.json().get("prices", [])
                    if price_list:
                        ask_p = float(price_list[0].get("asks", [{}])[0].get("price", 0))
                        bid_p = float(price_list[0].get("bids", [{}])[0].get("price", 0))
                        if ask_p > 0 and bid_p > 0:
                            prices_map[pair] = (ask_p + bid_p) / 2
                except Exception as e2:
                    logger.debug("US-180: Individual price fetch failed (%s): %s", pair, e2)

        return prices_map

    def apply_drawdown_guardian(self) -> List[str]:
        """Check open trades and tighten SL when trade moves in favor.

        - At 50% of TP distance: move SL to breakeven
        - At 75% of TP distance: lock in 50% of unrealized profit

        Returns:
            List of modification messages.
        """
        import requests
        import os

        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = self._oanda_base_url
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        statuses = self.monitor_open_trades()
        modifications = []

        # Phase 30 (US-180): Batch fetch all prices in one call
        _all_pairs = list({s["pair"] for s in statuses if s.get("pair")})
        _price_cache = self._batch_fetch_prices(_all_pairs)

        for status in statuses:
            trade_id = status["trade_id"]
            pair = status["pair"]
            direction = status["direction"]
            entry = status["entry"]
            sl_price = status["sl_price"]
            tp_price = status["tp_price"]
            pip_value = _get_pip_value(pair)

            if tp_price <= 0 or sl_price <= 0:
                continue

            # Phase 30 (US-180): Use batch-cached price
            mid = _price_cache.get(pair)
            if mid is None:
                logger.debug("US-180: No cached price for %s, skipping guardian", pair)
                continue

            # Calculate progress toward TP
            if direction == "LONG":
                tp_distance = tp_price - entry
                current_progress = mid - entry
            else:
                tp_distance = entry - tp_price
                current_progress = entry - mid

            if tp_distance <= 0:
                continue

            progress_pct = current_progress / tp_distance

            new_sl = None
            reason = ""

            # Dynamic drawdown thresholds (Phase 9: US-058)
            lock_pct = getattr(self.config, "trailing_stop_lock_pct", 0.75)
            be_pct = getattr(self.config, "trailing_stop_breakeven_pct", 0.50)
            try:
                # Phase 79: Use cached instance instead of creating fresh each call
                dd_mgr = self._dynamic_drawdown_mgr
                if dd_mgr is None:
                    from src.scanner.automation.dynamic_drawdown import DynamicDrawdownManager
                    dd_mgr = DynamicDrawdownManager(
                        normal_breakeven_pct=be_pct,
                        normal_lock_pct=lock_pct,
                    )
                current_dd = getattr(self, "_current_drawdown", 0.0)
                max_dd = getattr(self.config, "max_drawdown_pct", 0.15)
                dd_thresholds = dd_mgr.get_dynamic_thresholds(current_dd, max_dd)
                lock_pct = dd_thresholds.lock_pct
                be_pct = dd_thresholds.breakeven_pct
            except Exception as e:
                logger.debug(f"Execution Fall back to static thresholds: {e}")

            if progress_pct >= lock_pct:
                # Lock in 50% of unrealized profit
                if direction == "LONG":
                    new_sl = entry + (current_progress * 0.50)
                    if new_sl > sl_price:
                        reason = f"{lock_pct:.0%} TP reached, locking 50% profit (SL {sl_price:.5f} -> {new_sl:.5f})"
                    else:
                        new_sl = None
                else:
                    new_sl = entry - (current_progress * 0.50)
                    if new_sl < sl_price:
                        reason = f"{lock_pct:.0%} TP reached, locking 50% profit (SL {sl_price:.5f} -> {new_sl:.5f})"
                    else:
                        new_sl = None
            elif progress_pct >= be_pct:
                # Move SL to breakeven
                if direction == "LONG" and sl_price < entry:
                    new_sl = entry + (pip_value * 2)  # Breakeven + 2 pips
                    reason = f"{be_pct:.0%} TP reached, SL to breakeven (SL {sl_price:.5f} -> {new_sl:.5f})"
                elif direction == "SHORT" and sl_price > entry:
                    new_sl = entry - (pip_value * 2)
                    reason = f"{be_pct:.0%} TP reached, SL to breakeven (SL {sl_price:.5f} -> {new_sl:.5f})"

            if new_sl is not None:
                try:
                    # Modify the trade's SL via OANDA API
                    modify_data = {
                        "stopLoss": {"price": f"{new_sl:.5f}"}
                    }
                    # Phase 30 (US-181): Wrap SL modification in retry
                    resp = self._retry_oanda(
                        requests.put,
                        f"{base}/v3/accounts/{acct}/trades/{trade_id}/orders",
                        headers=headers,
                        json=modify_data,
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        modifications.append(f"{pair} #{trade_id}: {reason}")
                        logger.info(f"Drawdown guardian: {pair} #{trade_id}: {reason}")
                    else:
                        logger.warning(f"SL modification failed for {pair} #{trade_id}: {resp.text}")
                except Exception as e:
                    logger.warning(f"SL modification error for {pair} #{trade_id}: {e}")

        return modifications

    # ------------------------------------------------------------------
    # Phase 24 (US-147): OANDA helpers for position management
    # ------------------------------------------------------------------

    def close_trade_oanda(self, trade_id: str) -> bool:
        """Close an open trade on OANDA.

        Args:
            trade_id: OANDA trade ID to close.

        Returns:
            True if closed successfully.
        """
        import requests
        import os

        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = self._oanda_base_url
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            # Phase 30 (US-181): Wrap close trade in retry
            resp = self._retry_oanda(
                requests.put,
                f"{base}/v3/accounts/{acct}/trades/{trade_id}/close",
                headers=headers,
                json={},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"US-147: Closed trade #{trade_id} via position manager")
                return True
            else:
                logger.warning(f"US-147: Close trade #{trade_id} failed: {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"US-147: Close trade #{trade_id} error: {e}")
            return False

    def modify_sl_oanda(self, trade_id: str, new_sl: float) -> bool:
        """Modify the stop loss of an open trade on OANDA.

        Args:
            trade_id: OANDA trade ID.
            new_sl: New stop loss price.

        Returns:
            True if modified successfully.
        """
        import requests
        import os

        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = self._oanda_base_url
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            modify_data = {"stopLoss": {"price": f"{new_sl:.5f}"}}
            # Phase 30 (US-181): Wrap SL modification in retry
            resp = self._retry_oanda(
                requests.put,
                f"{base}/v3/accounts/{acct}/trades/{trade_id}/orders",
                headers=headers,
                json=modify_data,
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"US-147: Modified SL for trade #{trade_id} to {new_sl:.5f}")
                return True
            else:
                logger.warning(f"US-147: Modify SL #{trade_id} failed: {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"US-147: Modify SL #{trade_id} error: {e}")
            return False

    def update_trailing_stops(self, atr_by_pair: Optional[Dict[str, float]] = None) -> List[str]:
        """ATR-based trailing stop: move SL to trail behind price by ATR * multiplier.

        US-038: Only moves SL in the profitable direction (never widens).
        Trail distance = ATR * trailing_atr_multiplier.

        Args:
            atr_by_pair: Optional dict of pair → current ATR value.
                If None, uses a default trail of 15 pips.

        Returns:
            List of modification messages.
        """
        if not getattr(self.config, "enable_trailing_stop", True):
            return []

        # Phase 80 (US-P80-004): Refresh _current_drawdown from cached NAV before evaluating stops
        # DynamicDrawdownManager.get_dynamic_thresholds() reads _current_drawdown — was always 0.
        try:
            _nav_for_dd = getattr(self, "_cached_nav", None) or getattr(self.config, "account_equity", 0.0) or 0.0
            _nav_for_dd = float(_nav_for_dd)
            if _nav_for_dd > 0:
                if self._equity_hwm <= 0 or _nav_for_dd > self._equity_hwm:
                    self._equity_hwm = _nav_for_dd
                if self._equity_hwm > 0:
                    self._current_drawdown = max(
                        0.0, (self._equity_hwm - _nav_for_dd) / self._equity_hwm
                    )
        except Exception as _dd_refresh_err:
            logger.debug("Phase 80 (US-P80-004): Drawdown refresh skipped: %s", _dd_refresh_err)

        import requests
        import os

        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = self._oanda_base_url
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        trail_mult = float(getattr(self.config, "trailing_atr_multiplier", 1.5))

        # Phase 50 (US-314): Regime-aware trailing multiplier
        _regime_trail_mults = {
            "LOW": 2.0,
            "NORMAL": 2.5,
            "HIGH": 3.0,
            "EXTREME": 4.0,
        }
        try:
            if hasattr(self, "_regime_detector") and self._regime_detector is not None:
                _current_regime = getattr(self._regime_detector, "current_regime", None)
                if _current_regime and _current_regime in _regime_trail_mults:
                    trail_mult = _regime_trail_mults[_current_regime]
                    logger.debug(
                        "Phase 50 US-314: Regime-aware trail_mult=%.2f for regime=%s",
                        trail_mult, _current_regime,
                    )
        except Exception as _rt_err:
            logger.debug("Phase 50 US-314: Regime trail fallback: %s", _rt_err)

        statuses = self.monitor_open_trades()
        modifications: List[str] = []

        # Phase 30 (US-180): Batch fetch all prices in one call
        _all_pairs = list({s["pair"] for s in statuses if s.get("pair")})
        _price_cache = self._batch_fetch_prices(_all_pairs)

        for status in statuses:
            trade_id = status["trade_id"]
            pair = status["pair"]
            direction = status["direction"]
            entry = status["entry"]
            sl_price = status["sl_price"]
            pip_value = _get_pip_value(pair)

            if sl_price <= 0:
                continue  # No SL set — skip

            # Get current ATR for trail distance
            if atr_by_pair and pair in atr_by_pair:
                atr = atr_by_pair[pair]
            else:
                # Fallback ATR from config
                atr = getattr(self.config, "fallback_atr_pips", 15.0) * pip_value

            trail_distance = atr * trail_mult

            # Phase 50 (US-314): Confidence scaling for trail distance
            # Higher confidence → looser trail (let winners run)
            # Lower confidence → tighter trail (protect gains)
            _trade_conf = status.get("confidence", 0.0)
            if isinstance(_trade_conf, (int, float)) and _trade_conf > 0:
                _conf_scale = 0.8 + (_trade_conf * 0.4)  # Range: 0.8-1.2x
                trail_distance *= _conf_scale
                logger.debug(
                    "Phase 50 US-314: Confidence-scaled trail for %s: conf=%.2f, scale=%.2f",
                    pair, _trade_conf, _conf_scale,
                )

            # Phase 30 (US-180): Use batch-cached price
            mid = _price_cache.get(pair)
            if mid is None:
                logger.debug("US-180: No cached price for %s, skipping trailing SL", pair)
                continue

            new_sl = None
            if direction == "LONG":
                # Trail SL below current price
                candidate_sl = mid - trail_distance
                # Only move SL up (tighter), never down (wider)
                if candidate_sl > sl_price and candidate_sl > entry:
                    new_sl = candidate_sl
            else:  # SHORT
                candidate_sl = mid + trail_distance
                if candidate_sl < sl_price and candidate_sl < entry:
                    new_sl = candidate_sl

            if new_sl is not None:
                try:
                    modify_data = {"stopLoss": {"price": f"{new_sl:.5f}"}}
                    # Phase 30 (US-181): Wrap trailing SL modification in retry
                    resp = self._retry_oanda(
                        requests.put,
                        f"{base}/v3/accounts/{acct}/trades/{trade_id}/orders",
                        headers=headers,
                        json=modify_data,
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        msg = (
                            f"{pair} #{trade_id}: Trailing SL "
                            f"{sl_price:.5f} → {new_sl:.5f} "
                            f"(trail={trail_distance/pip_value:.1f} pips)"
                        )
                        modifications.append(msg)
                        logger.info(f"Trailing stop: {msg}")

                        # Log to observations (Phase 30 US-183: uses shared _observer)
                        try:
                            self._get_observer().log_observation(
                                pair=pair,
                                category="trailing_stop",
                                description=msg,
                                metadata={
                                    "trade_id": trade_id,
                                    "old_sl": sl_price,
                                    "new_sl": new_sl,
                                    "trail_pips": round(trail_distance / pip_value, 1),
                                    "mid_price": mid,
                                },
                            )
                        except Exception as e:
                            logger.debug(f"Execution trade tracking skipped: {e}")
                    else:
                        logger.warning(f"Trailing SL modify failed {pair} #{trade_id}: {resp.text}")
                except Exception as e:
                    logger.warning(f"Trailing SL error {pair} #{trade_id}: {e}")

        return modifications

    @staticmethod
    def _get_current_regime(scanner: Any, entry: Dict[str, Any]) -> str:
        """Phase 24 (US-145): Derive current volatility regime for exit recording.

        Tries: (1) scanner's cached regime, (2) scanner config default, (3) entry regime fallback.
        """
        # Try scanner's last known regime from cached results
        if scanner is not None:
            try:
                _cached = getattr(scanner, "_cached_results", {})
                pair = entry.get("pair", "")
                if pair in _cached:
                    _regime = str(getattr(_cached[pair], "volatility_regime", "")).upper()
                    if _regime and _regime != "UNKNOWN":
                        return _regime
                # Try any cached result's regime as proxy
                for _r in _cached.values():
                    _regime = str(getattr(_r, "volatility_regime", "")).upper()
                    if _regime and _regime != "UNKNOWN":
                        return _regime
                # Fall back to config default
                _default = getattr(getattr(scanner, "config", None), "volatility_regime_default", None)
                if _default:
                    return str(_default).upper()
            except Exception as e:
                logger.debug(f"Execution regime detection skipped: {e}")
        # Fall back to entry's regime_at_entry
        _entry_regime = (entry.get("regime") or {}).get("volatility_regime", "NORMAL")
        return str(_entry_regime).upper() if _entry_regime else "NORMAL"

    @staticmethod
    def _determine_exit_reason(
        entry: Dict[str, Any],
        close_price: float,
        entry_price: float,
        pip_value: float,
        duration_minutes: Optional[float] = None,
    ) -> str:
        """Determine why a trade was closed (US-061).

        Compares close_price against SL/TP levels to classify exit reason.

        Returns:
            One of: tp_hit, sl_hit, trailing_stop, breakeven_stop, timeout, manual_close
        """
        sl_price = float(entry.get("sl_price", 0))
        tp_price = float(entry.get("tp_price", 0))
        direction = str(entry.get("direction", "")).upper()

        # Tolerance: within 2 pips of SL/TP counts as a hit
        tolerance = pip_value * 2 if pip_value > 0 else 0.0002

        if tp_price > 0 and abs(close_price - tp_price) <= tolerance:
            return "tp_hit"

        if sl_price > 0 and abs(close_price - sl_price) <= tolerance:
            return "sl_hit"

        # Check if SL was moved to breakeven (close near entry)
        if entry_price > 0 and abs(close_price - entry_price) <= tolerance:
            return "breakeven_stop"

        # Check for trailing stop: trade was profitable but closed before TP
        if direction == "LONG" and close_price > entry_price and tp_price > 0 and close_price < tp_price:
            return "trailing_stop"
        if direction == "SHORT" and close_price < entry_price and tp_price > 0 and close_price > tp_price:
            return "trailing_stop"

        # Timeout: if duration exceeds 24 hours (1440 minutes)
        if duration_minutes is not None and duration_minutes > 1440:
            return "timeout"

        # Default: can't determine specific reason
        return "manual_close"

    def sync_closed_trades_rl(self, scanner: Any = None) -> Dict[str, Any]:
        """Check OANDA for recently closed trades and run RL feedback on agent weights.

        Reads the RL journal, matches against OANDA closed trades, updates outcomes,
        and calls ScannerAgentTeam.update_weights_from_outcome for each closed trade.

        Args:
            scanner: Optional Scanner instance for Phase 18 calibration feedback.
                     When provided, trade outcomes feed threshold optimizer,
                     dynamic risk allocator, execution quality profiler, etc.

        Returns:
            Dict with sync results: trades_synced, weights_updated, new_weights.
        """
        import json
        import requests
        import os
        from pathlib import Path

        journal_path = Path("trained_data/trade_journal_rl.json")
        if not journal_path.exists():
            return {"trades_synced": 0, "weights_updated": False, "detail": "no journal"}

        try:
            entries = json.loads(journal_path.read_text())
        except Exception as e:
            logger.debug("US-178: Journal parse failed in sync_closed_trades_rl: %s", e)
            return {"trades_synced": 0, "weights_updated": False, "detail": f"journal parse error: {e}"}

        # Find entries without outcomes
        pending = [e for e in entries if e.get("outcome") is None]
        if not pending:
            return {"trades_synced": 0, "weights_updated": False, "detail": "no pending trades"}

        # Fetch closed trades from OANDA
        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = self._oanda_base_url
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            # Phase 30 (US-181): Wrap closed trades fetch in retry
            resp = self._retry_oanda(
                requests.get,
                f"{base}/v3/accounts/{acct}/trades",
                params={"state": "CLOSED", "count": 50},
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            closed_trades = {t["id"]: t for t in resp.json().get("trades", [])}
        except Exception as e:
            return {"trades_synced": 0, "weights_updated": False, "detail": f"oanda error: {e}"}

        synced = 0
        synced_entries = []  # Track which entries were synced this call (avoid double-counting)
        rl_updates = []

        for entry in pending:
            tid = str(entry.get("trade_id", ""))
            if tid not in closed_trades:
                continue

            ct = closed_trades[tid]
            realized_pl = float(ct.get("realizedPL", 0))
            trade_won = realized_pl > 0
            close_price = float(ct.get("averageClosePrice", 0))
            entry_price = float(ct.get("price", entry.get("entry_price", 0)))

            # US-014: Asset class-aware P&L calculation
            asset_class = entry.get("asset_class", "FX")
            pnl_pips = 0.0

            if asset_class == "FUTURES":
                # Futures P&L: (exit_price - entry_price) * multiplier * contracts * direction_sign
                multiplier = float(entry.get("multiplier", 1.0))
                contracts = float(entry.get("lots", 1.0))
                direction_sign = 1.0 if entry.get("direction", "").upper() == "LONG" else -1.0
                pnl_pips = (close_price - entry_price) * multiplier * contracts * direction_sign
            else:
                # FX P&L: unchanged from original logic
                pip_value = _get_pip_value(entry.get("pair", ""), 0.0001)
                if pip_value > 0 and entry_price > 0:
                    pnl_pips = (close_price - entry_price) / pip_value
                    if entry.get("direction", "").upper() == "SHORT":
                        pnl_pips = -pnl_pips

            # Calculate duration
            close_time_str = ct.get("closeTime", "")
            duration_minutes = None
            entry_ts = entry.get("timestamp", "")
            try:
                if entry_ts and close_time_str:
                    open_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
                    close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                    duration_minutes = round((close_dt - open_dt).total_seconds() / 60.0, 1)
            except Exception as e:
                logger.debug(f"Execution non-critical operation skipped: {e}")

            # US-061: Determine exit reason by comparing close_price to SL/TP
            exit_reason = self._determine_exit_reason(
                entry=entry,
                close_price=close_price,
                entry_price=entry_price,
                pip_value=pip_value,
                duration_minutes=duration_minutes,
            )

            # Phase 80 (US-P80-001): Compute MAE/MFE/bars_in_trade for TradeOutcomePredictor
            # MAE proxy: on a loss, equals |pnl_pips|; on a win, 0 (no intrabar OANDA data)
            _mae_pips = round(max(0.0, -min(pnl_pips, 0.0)), 1)
            # MFE proxy: at minimum equals realized profit if positive; 0 on a loss
            _mfe_pips = round(max(0.0, pnl_pips), 1)
            # bars_in_trade: H1 bar count from duration; defaults to 1 when duration unknown
            _bars_in_trade = max(1, round(duration_minutes / 60.0)) if duration_minutes else 1

            outcome_data = {
                "realized_pl": realized_pl,
                "pnl_pips": round(pnl_pips, 1),
                "trade_won": trade_won,
                "close_price": close_price,
                "close_time": close_time_str,
                "exit_price": close_price,
                "exit_time": close_time_str,
                "duration_minutes": duration_minutes,
                "exit_reason": exit_reason,
                "mae_pips": _mae_pips,
                "mfe_pips": _mfe_pips,
                "bars_in_trade": _bars_in_trade,
                "regime_at_entry": entry.get("regime", {}).get("volatility_regime"),
                # Phase 24 (US-145): Derive regime at exit from scanner or entry fallback
                "regime_at_exit": self._get_current_regime(scanner, entry)
            }
            entry["outcome"] = outcome_data
            synced += 1
            synced_entries.append(entry)

            # Feed RL replay buffer for sizer/gates/exits models
            try:
                self._sync_trade_outcome_to_rl(entry, outcome_data)
            except Exception as rl_err:
                logger.debug(f"RL replay buffer error for {tid}: {rl_err}")

            # Record trade outcome to episodic memory for pattern suppression learning
            if self._episodic_memory is not None:
                try:
                    _episode_id = entry.get("episode_id", "")
                    if _episode_id:
                        _outcome_str = "WIN" if trade_won else "LOSS"
                        _success = self._episodic_memory.record_outcome(
                            episode_id=_episode_id,
                            outcome=_outcome_str,
                            pnl_pips=float(pnl_pips),
                        )
                        if _success:
                            logger.debug(
                                f"Episodic memory: Recorded {_outcome_str} ({pnl_pips:.1f} pips) for episode {_episode_id}"
                            )
                        else:
                            logger.debug(f"Episodic memory: Episode {_episode_id} not found for outcome recording")
                    else:
                        logger.debug(f"Trade {tid} has no episode_id for episodic memory recording")
                except Exception as _em_err:
                    logger.debug(f"Episodic memory record_outcome error for {tid}: {_em_err}")

            # Phase 81 (US-P81-002): Deregister from PositionTimeoutManager on confirmed close
            if self._position_timeout_mgr is not None:
                try:
                    self._position_timeout_mgr.close_trade(tid)
                except Exception as _pt_close_err:
                    logger.debug("Phase 81: PositionTimeout close_trade error for %s: %s", tid, _pt_close_err)

            # Phase 78: Record outcome to accuracy gate (per-pair rolling accuracy)
            if scanner is not None:
                _ag = getattr(scanner, "_accuracy_gate", None)
                if _ag is not None:
                    try:
                        _direction = entry.get("direction", "LONG")
                        _conf = float(entry.get("confidence", 0.5))
                        _ag.record_outcome(
                            pair=pair,
                            predicted_direction=_direction,
                            actual_outcome=trade_won,
                            confidence=_conf,
                        )
                        logger.debug(
                            "Phase 78: AccuracyGate recorded %s %s outcome=%s",
                            pair, _direction, "WIN" if trade_won else "LOSS",
                        )
                    except Exception as _ag_err:
                        logger.debug(f"Phase 78: AccuracyGate record error for {tid}: {_ag_err}")

            # Phase 78: Record prediction to retrain trigger (drift detection)
            if scanner is not None:
                _rt = getattr(scanner, "_retrain_trigger", None)
                if _rt is not None:
                    try:
                        _rt.record_prediction(pair=pair, correct=trade_won)
                        logger.debug(
                            "Phase 78: RetrainTrigger recorded %s correct=%s",
                            pair, trade_won,
                        )
                    except Exception as _rt_err:
                        logger.debug(f"Phase 78: RetrainTrigger record error for {tid}: {_rt_err}")

            # ── Tier 6: Meta-learner hyperparameter adaptation ──
            if scanner is not None and getattr(scanner, '_meta_learner', None) is not None:
                try:
                    _ml_agents_info = entry.get("agents", {})
                    _ml_agent_reasons = _ml_agents_info.get("agent_reasons", [])
                    _ml_agent_scores: Dict[str, float] = {}
                    _ml_agent_weights: Dict[str, float] = {}
                    for _ar in _ml_agent_reasons:
                        _name = _ar.get("name", "")
                        if _name:
                            _ml_agent_scores[_name] = float(_ar.get("score", 0.5))
                            _ml_agent_weights[_name] = float(_ar.get("weight", 1.0))

                    if _ml_agent_scores:
                        _ml_regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
                        _ml_notional = abs(entry_price * float(ct.get("initialUnits", 1)))
                        _ml_pnl_pct = realized_pl / max(_ml_notional, 1.0)
                        _ml_trade_result = {
                            "pnl_pct": _ml_pnl_pct,
                            "outcome": "win" if trade_won else "loss",
                        }
                        scanner._meta_learner.on_trade_close(
                            pair=entry.get("pair", ""),
                            regime=_ml_regime,
                            trade_result=_ml_trade_result,
                            agent_scores=_ml_agent_scores,
                            agent_weights=_ml_agent_weights,
                        )
                        logger.info(
                            "Tier 6: Meta-learner updated for %s/%s (pnl_pct=%.4f, agents=%d)",
                            entry.get("pair", ""), _ml_regime, _ml_pnl_pct, len(_ml_agent_scores),
                        )
                except Exception as _ml_err:
                    logger.debug("Tier 6: Meta-learner on_trade_close error: %s", _ml_err)

            # Tier 6: Record outcome to EnsembleWeighter for learned model weights
            if scanner is not None and getattr(scanner, '_ensemble_weighter', None) is not None:
                try:
                    _ew_gate = entry.get("gate_details", {})
                    _ew_scores = _ew_gate.get("scores", {})
                    _ew_dir = float(_ew_scores.get("direction_score", 0.5))
                    _ew_conf = float(_ew_scores.get("confidence_score", 0.5))
                    _ew_mom = float(_ew_scores.get("momentum_score", 0.5))
                    _ew_risk = float(_ew_scores.get("risk_score", 0.5))
                    _ew_regime = entry.get("regime", {})
                    _ew_vol_id = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}.get(
                        str(_ew_regime.get("volatility_regime", "NORMAL")), 1
                    )
                    _ew_vol_conf = float(_ew_regime.get("volatility_regime_confidence", 0.5))
                    # Normalize outcome: direction-adjusted pnl_pips as proxy for model accuracy
                    _ew_outcome = pnl_pips / 100.0  # Scale pips to approximate 0-centered outcome
                    scanner._ensemble_weighter.record(
                        context_features=[_ew_vol_id / 3.0, _ew_vol_conf],
                        model_predictions=[_ew_dir, _ew_conf, _ew_mom, _ew_risk],
                        realized_outcome=_ew_outcome,
                    )
                    logger.debug(
                        "Tier 6: EnsembleWeighter recorded outcome for %s (pnl_pips=%.1f)",
                        entry.get("pair", ""), pnl_pips,
                    )
                except Exception as _ew_err:
                    logger.debug("Tier 6: EnsembleWeighter record error: %s", _ew_err)

            # Tier 6 Phase 1: MAML Ridge outcome recording
            if scanner is not None and getattr(scanner, '_maml_ridge', None) is not None:
                try:
                    _maml_regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
                    if isinstance(_maml_regime, int):
                        _maml_regime = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}.get(_maml_regime, "NORMAL")
                    # Target confidence: map outcome to 0-100 scale
                    # Wins with good R:R → high confidence was correct → target near ridge_conf
                    # Losses → confidence was wrong → target = 100 - ridge_conf
                    _maml_ridge_conf = float(entry.get("regime", {}).get("ridge_confidence",
                                             entry.get("agents", {}).get("ridge_confidence", 50)))
                    if trade_won:
                        _maml_target = min(100.0, _maml_ridge_conf + abs(pnl_pips) * 0.5)
                    else:
                        _maml_target = max(0.0, _maml_ridge_conf - abs(pnl_pips) * 0.5)
                    _maml_target = max(0.0, min(100.0, _maml_target))

                    # Tier 6 Phase 2: Use real Ridge features if available, else proxy
                    import numpy as np
                    _stored_ridge_features = entry.get("ridge_features")
                    if _stored_ridge_features is not None and len(_stored_ridge_features) >= 14:
                        # Real 14D Ridge features from prediction time
                        _maml_features = np.array(_stored_ridge_features[:14], dtype=np.float32)
                    else:
                        # Fallback: proxy from gate_details (Phase 1 legacy)
                        _maml_gate = entry.get("gate_details", {})
                        _maml_scores = _maml_gate.get("scores", {})
                        _maml_features = np.array([
                            float(_maml_scores.get("direction_score", 0.5)),
                            float(_maml_scores.get("confidence_score", 0.5)),
                            float(_maml_scores.get("momentum_score", 0.5)),
                            float(_maml_scores.get("risk_score", 0.5)),
                        ] + [0.0] * 10, dtype=np.float32) if _maml_scores else None
                    if _maml_features is not None:
                        scanner._maml_ridge.record_outcome(
                            features=_maml_features,
                            regime=str(_maml_regime),
                            target_confidence=_maml_target,
                        )
                        _feat_source = "real_ridge" if _stored_ridge_features is not None else "proxy"
                        logger.debug(
                            "Tier 6 MAML: Recorded outcome for %s regime=%s target=%.1f features=%s",
                            entry.get("pair", ""), _maml_regime, _maml_target, _feat_source,
                        )

                        # Tier 6 Phase 2: Record outcome in benchmark
                        _maml_bench = getattr(scanner, '_maml_benchmark', None)
                        if _maml_bench is not None:
                            try:
                                # Get MAML's prediction at trade time from entry metadata
                                _maml_shadow = entry.get("maml_shadow", {})
                                _bench_maml_conf = float(_maml_shadow.get("maml_confidence", _maml_target))
                                _maml_bench.record_outcome(
                                    trade_id=entry.get("oanda_trade_id", entry.get("pair", "unknown")),
                                    regime=str(_maml_regime),
                                    maml_confidence=_bench_maml_conf,
                                    ridge_confidence=_maml_ridge_conf,
                                    realized_confidence=_maml_target,
                                    pnl_pips=float(pnl_pips),
                                    won=bool(trade_won),
                                    exit_reason=str(outcome_data.get("exit_reason", "")),
                                    duration_minutes=outcome_data.get("duration_minutes"),
                                    regime_at_exit=str(outcome_data.get("regime_at_exit", "")),
                                )
                            except Exception as _bench_err:
                                logger.debug("Tier 6 Benchmark: record error: %s", _bench_err)
                except Exception as _maml_err:
                    logger.debug("Tier 6 MAML: outcome record error: %s", _maml_err)

            # Collect agent verdicts for RL feedback (with regime from entry)
            agents = entry.get("agents", {})
            regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")

            # Regime-aware reward shaping: compute shaped reward to augment flat PnL signal
            _shaped_reward = None
            if self._regime_reward is not None:
                try:
                    _max_dd = abs(min(0.0, realized_pl)) / max(abs(realized_pl), 0.01)
                    _dur_bars = int((duration_minutes or 0) / 5)  # Approx 5-min bars
                    _entry_conf = float(entry.get("confidence", 0.5))
                    # Map volatility regime names to reward profile keys
                    _regime_map = {
                        "LOW": "normal", "NORMAL": "normal",
                        "HIGH": "choppy", "EXTREME": "crisis",
                    }
                    _reward_regime = _regime_map.get(regime, "normal")
                    _reward_result = self._regime_reward.compute_reward(
                        pnl=realized_pl,
                        max_drawdown=_max_dd,
                        regime=_reward_regime,
                        duration_bars=_dur_bars,
                        entry_confidence=_entry_conf,
                    )
                    _shaped_reward = _reward_result.shaped_reward
                    outcome_data["shaped_reward"] = round(_shaped_reward, 6)
                    outcome_data["regime_reward_details"] = _reward_result.to_dict()
                    if _reward_result.is_reward_hack_alert:
                        logger.warning(
                            f"RegimeReward hack alert for {tid}: "
                            f"PnL-reward correlation={_reward_result.pnl_reward_correlation:.3f}"
                        )
                except Exception as _rr_err:
                    logger.debug(f"Regime reward computation skipped for {tid}: {_rr_err}")

            if agents.get("agent_reasons"):
                rl_updates.append({
                    "agent_verdicts": agents["agent_reasons"],
                    "trade_won": trade_won,
                    "regime": regime,
                    # Phase 24 (US-149): Pass exit_reason for trailing stop differentiation
                    "exit_reason": exit_reason,
                    "pnl": realized_pl,
                    "shaped_reward": _shaped_reward,
                })

        # Phase 47 (US-297): Record trade outcomes to expectancy tracker for per-agent per-regime learning
        self._record_expectancy_from_trades(pending, closed_trades)

        # ── Phase 49 (US-307/308/309): Feed outcomes to Phase 48 modules ──

        # US-307: Update pair performance tracker for auto-blacklisting
        if self._pair_tracker is not None and synced > 0:
            try:
                _pt_recorded = 0
                for entry in pending:
                    tid = str(entry.get("trade_id", ""))
                    if tid not in closed_trades:
                        continue
                    ct = closed_trades[tid]
                    _pair = entry.get("pair", "")
                    _realized_pl = float(ct.get("realizedPL", 0))
                    _won = _realized_pl > 0
                    self._pair_tracker.update_from_trade(pair=_pair, pnl=_realized_pl, won=_won)
                    _pt_recorded += 1
                if _pt_recorded > 0:
                    self._pair_tracker.save_state()
                    logger.info("Phase 49 (US-307): Fed %d outcomes to pair tracker", _pt_recorded)
            except Exception as _pt_err:
                logger.debug(f"Phase 49: Pair tracker update error: {_pt_err}")

        # US-308: Trade attribution engine — SHAP-lite alignment scoring
        if self._attribution_engine is not None and synced > 0:
            try:
                _attr_recorded = 0
                for entry in pending:
                    tid = str(entry.get("trade_id", ""))
                    if tid not in closed_trades:
                        continue
                    ct = closed_trades[tid]
                    _pair = entry.get("pair", "")
                    _direction = entry.get("direction", "BUY").upper()
                    _realized_pl = float(ct.get("realizedPL", 0))
                    _won = _realized_pl > 0
                    _outcome_str = "WIN" if _won else "LOSS"
                    _agents_info = entry.get("agents", {})
                    _agent_reasons = _agents_info.get("agent_reasons", [])

                    # Build agent_verdicts dict: {name: (verdict, confidence, weight)}
                    _verdicts = {}
                    for _ar in _agent_reasons:
                        _name = _ar.get("name", "")
                        if _name:
                            _verdict = "LONG" if _ar.get("passed", False) else "SHORT"
                            _weight = float(_ar.get("weight", 1.0))
                            _conf = float(_ar.get("confidence", 0.5))
                            _verdicts[_name] = (_verdict, _conf, _weight)

                    if _verdicts:
                        _attribution = self._attribution_engine.attribute_trade(
                            trade_id=tid,
                            pair=_pair,
                            direction=_direction,
                            outcome=_outcome_str,
                            pnl=_realized_pl,
                            agent_verdicts=_verdicts,
                        )
                        # Store attribution summary in outcome for audit trail
                        entry.setdefault("outcome", {})["attribution"] = {
                            "alignment_ratio": round(_attribution.alignment_ratio, 3),
                            "aligned": [c.agent_name for c in _attribution.aligned_agents],
                            "misaligned": [c.agent_name for c in _attribution.misaligned_agents],
                        }
                        _attr_recorded += 1

                if _attr_recorded > 0:
                    self._attribution_engine.save_state()
                    # Phase 50 (US-308-ext): Pipe attribution weight recommendations to gate threshold optimizer
                    _weight_recs = self._attribution_engine.get_weight_recommendations()
                    _notable = {k: f"{v:.2f}x" for k, v in _weight_recs.items() if abs(v - 1.0) > 0.05}
                    logger.info(
                        "Phase 49 (US-308): Attributed %d trades. Weight recs: %s",
                        _attr_recorded, _notable or "no notable adjustments",
                    )

                    # Phase 50 (US-308-ext): Feed attribution feedback to gate threshold optimizer
                    if self._gate_threshold_optimizer is not None and _weight_recs:
                        try:
                            for _agent_name, _weight_delta in _weight_recs.items():
                                # Submit agent weight delta as feedback
                                # Interpret weight_delta > 1.0 as "increase confidence in this agent"
                                # and weight_delta < 1.0 as "decrease confidence"
                                self._gate_threshold_optimizer.record_agent_feedback(
                                    agent_name=_agent_name,
                                    weight_delta=_weight_delta,
                                    source="trade_attribution",
                                )
                            logger.info(
                                "Phase 50 (US-308-ext): %d attribution weight recommendations piped to gate threshold optimizer",
                                len(_weight_recs),
                            )
                        except AttributeError:
                            # If gate_threshold_optimizer doesn't have record_agent_feedback, skip silently
                            logger.debug(
                                "Phase 50: gate_threshold_optimizer missing record_agent_feedback method, skipping"
                            )
                        except Exception as _atf_err:
                            logger.debug(f"Phase 50 (US-308-ext): Attribution feedback pipe error: {_atf_err}")
            except Exception as _attr_err:
                logger.debug(f"Phase 49: Attribution engine error: {_attr_err}")

        # ── Phase 54 (US-334): Gate Threshold Optimizer — auto-tune gates after attribution ──
        if self._gate_threshold_optimizer is not None and self._attribution_engine is not None and synced > 0:
            try:
                _gate_report = self._attribution_engine.generate_report()
                if _gate_report:
                    _gate_recs = self._gate_threshold_optimizer.compute_recommendations(_gate_report)
                    if _gate_recs:
                        _applied = self._gate_threshold_optimizer.apply_recommendations(_gate_recs)
                        if _applied:
                            self._gate_threshold_optimizer.save_state()
                            logger.info(
                                "Phase 54 (US-334): Applied %d gate threshold adjustments: %s",
                                len(_applied),
                                {r.gate_name: f"{r.current_threshold:.4f}→{r.recommended_threshold:.4f}" for r in _applied},
                            )
                    else:
                        logger.debug("Phase 54 (US-334): No gate threshold adjustments recommended")
            except Exception as _gto_err:
                logger.debug(f"Phase 54: Gate threshold optimizer error: {_gto_err}")

        # ── Phase 54 (US-335): Feature Drift Detector — record outcomes and check periodically ──
        if self._feature_drift_detector is not None and synced > 0:
            try:
                for entry in pending:
                    tid = str(entry.get("trade_id", ""))
                    if tid in closed_trades:
                        _won = float(closed_trades[tid].get("realizedPL", 0)) > 0
                        self._feature_drift_detector.record_trade(won=_won)

                if self._feature_drift_detector.should_check():
                    # Split journal into recent vs baseline
                    _all_with_outcomes = [e for e in entries if e.get("outcome") is not None]
                    _recent_n = min(self._feature_drift_detector.config.recent_window, len(_all_with_outcomes))
                    _baseline_n = min(self._feature_drift_detector.config.baseline_window, len(_all_with_outcomes) - _recent_n)
                    if _recent_n >= 50 and _baseline_n >= 50:
                        _recent = _all_with_outcomes[-_recent_n:]
                        _baseline = _all_with_outcomes[-(_recent_n + _baseline_n):-_recent_n]
                        _drift_result = self._feature_drift_detector.check_drift(_recent, _baseline)
                        self._feature_drift_detector.save_state()
                        if _drift_result.drift_detected:
                            logger.warning(
                                "Phase 54 (US-335): Drift detected — %s",
                                _drift_result.reason,
                            )
            except Exception as _drift_err:
                logger.debug(f"Phase 54: Feature drift detector error: {_drift_err}")

        # ── Phase 55 (US-344): Feature Health Monitor — PSI per feature, every 50 trades ──
        if self._feature_health_monitor is not None and synced > 0:
            try:
                for _ in range(synced):
                    self._feature_health_monitor.record_trade()
                if self._feature_health_monitor.should_check():
                    _all_with_outcomes = [e for e in entries if e.get("outcome") is not None]
                    _recent_n = min(self._feature_health_monitor.recent_window, len(_all_with_outcomes))
                    _baseline_n = min(
                        self._feature_health_monitor.baseline_window,
                        len(_all_with_outcomes) - _recent_n,
                    )
                    if _recent_n >= 50 and _baseline_n >= 50:
                        _fh_recent = _all_with_outcomes[-_recent_n:]
                        _fh_baseline = _all_with_outcomes[-(_recent_n + _baseline_n):-_recent_n]
                        _fh_result = self._feature_health_monitor.check_health(
                            recent_trades=_fh_recent,
                            baseline_trades=_fh_baseline,
                            wfe_monitor=getattr(self, "_wfe_monitor", None),
                        )
                        self._feature_health_monitor.save_state()
                        if _fh_result.alert_triggered:
                            logger.warning(
                                "Phase 55 (US-344): Feature health alert — "
                                "score=%.3f, drifted=%s",
                                _fh_result.feature_health_score,
                                _fh_result.significant_drift_features,
                            )
            except Exception as _fh_err:
                logger.debug(f"Phase 55: Feature health monitor error: {_fh_err}")

        # ── Phase 54 (US-336): Trade Cluster Analyzer — recluster when interval reached ──
        if self._trade_cluster_analyzer is not None and synced > 0:
            try:
                for _ in range(synced):
                    self._trade_cluster_analyzer.record_trade()
                if self._trade_cluster_analyzer.should_recluster():
                    _all_with_outcomes = [e for e in entries if e.get("outcome") is not None]
                    if len(_all_with_outcomes) >= 50:
                        self._trade_cluster_analyzer.fit(_all_with_outcomes)
                        self._trade_cluster_analyzer.save_state()
            except Exception as _cluster_err:
                logger.debug(f"Phase 54: Trade cluster analyzer error: {_cluster_err}")

        # US-309: Walk-forward optimizer — record outcomes and trigger optimization
        if self._walkforward_optimizer is not None and synced > 0:
            try:
                import time as _time_mod
                _wfo_recorded = 0
                for entry in pending:
                    tid = str(entry.get("trade_id", ""))
                    if tid not in closed_trades:
                        continue
                    ct = closed_trades[tid]
                    _pair = entry.get("pair", "")
                    _direction = entry.get("direction", "BUY").upper()
                    _realized_pl = float(ct.get("realizedPL", 0))
                    _won = _realized_pl > 0
                    _confidence = float(entry.get("confidence", 0.5))
                    _momentum = float(entry.get("agents", {}).get("weighted_vote_score", 0.5))
                    _sl_pips = float(entry.get("sl_pips", 0))
                    _tp_pips = float(entry.get("tp_pips", 0))
                    _regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
                    _entry_ts = entry.get("timestamp", "")
                    _ts = 0.0
                    if _entry_ts:
                        try:
                            _ts = datetime.fromisoformat(_entry_ts.replace("Z", "+00:00")).timestamp()
                        except Exception:
                            _ts = _time_mod.time()
                    else:
                        _ts = _time_mod.time()

                    from src.scanner.walkforward_optimizer import TradeOutcome as WFTradeOutcome
                    self._walkforward_optimizer.record_outcome(WFTradeOutcome(
                        pair=_pair,
                        direction=_direction,
                        pnl=_realized_pl,
                        confidence=_confidence,
                        momentum_score=_momentum,
                        atr_sl_pips=_sl_pips,
                        atr_tp_pips=_tp_pips,
                        regime=_regime,
                        won=_won,
                        timestamp=_ts,
                    ))
                    _wfo_recorded += 1

                if _wfo_recorded > 0:
                    logger.info("Phase 49 (US-309): Recorded %d outcomes to walk-forward optimizer", _wfo_recorded)

                # Check if optimization should run
                if self._walkforward_optimizer.should_optimize():
                    _opt_result = self._walkforward_optimizer.run_optimization()
                    logger.info(
                        "Phase 49 (US-309): Walk-forward optimization complete — "
                        "WFE=%.3f IS_PF=%.2f OOS_PF=%.2f recommendation=%s",
                        _opt_result.wfe, _opt_result.is_profit_factor,
                        _opt_result.oos_profit_factor, _opt_result.recommendation,
                    )
                    # Save results to config_adjustments.json (NOT auto-applied)
                    self._walkforward_optimizer.save_state()
            except Exception as _wfo_err:
                logger.debug(f"Phase 49: Walk-forward optimizer error: {_wfo_err}")

        # ── Phase 52 (US-323): Walk-Forward Retrainer — periodic agent weight retraining ──
        # Only run after sufficient new trades (every 20 synced trades as per walk-forward config)
        if self._walkforward_retrainer is not None and synced >= 5:
            try:
                _journal_str = str(journal_path)
                _wfr_epochs = self._walkforward_retrainer.run(
                    journal_path=_journal_str,
                    current_weights_path="trained_data/models/agent_weights.json",
                    output_path="trained_data/models/walkforward_epochs.json",
                )
                if _wfr_epochs:
                    _n_epochs = len(_wfr_epochs)
                    _accepted = sum(1 for e in _wfr_epochs if e.accepted)
                    # Compute latest WFE for monitoring
                    _latest_wfe = _wfr_epochs[-1].wfe if _wfr_epochs else 0.0
                    logger.info(
                        "Phase 52 (US-323): Walk-forward retrainer: %d epochs, %d accepted, latest WFE=%.4f",
                        _n_epochs, _accepted, _latest_wfe,
                    )

                    # Phase 53 (US-332): Record WFE and check stability before applying weights
                    _wfe_allow_retrain = True
                    if self._wfe_monitor is not None:
                        try:
                            _wfe_state = self._wfe_monitor.record_wfe(
                                wfe=_latest_wfe,
                                epoch_count=_n_epochs,
                                accepted_count=_accepted,
                            )
                            _wfe_allow_retrain = self._wfe_monitor.should_retrain()
                            if not _wfe_allow_retrain:
                                logger.warning(
                                    "Phase 53 (US-332): WFE monitor state=%s — "
                                    "retraining PAUSED, using frozen weights",
                                    _wfe_state,
                                )
                            self._wfe_monitor.save_state()
                        except Exception as _wfe_err:
                            logger.debug(f"Phase 53: WFE monitor error: {_wfe_err}")

                    if _accepted > 0 and _wfe_allow_retrain:
                        _new_weights = self._walkforward_retrainer.get_current_weights()
                        if _new_weights:
                            self._walkforward_retrainer.save_weights(
                                "trained_data/models/agent_weights.json"
                            )
                            # Freeze these as last known good weights
                            if self._wfe_monitor is not None:
                                try:
                                    self._wfe_monitor.freeze_weights(_new_weights)
                                except Exception as _exc:
                                    logger.debug("H-4: Silent exception in unknown: %s", _exc)
                            logger.info(
                                "Phase 52 (US-323): Retrained agent weights saved — %s",
                                {k: f"{v:.3f}" for k, v in list(_new_weights.items())[:5]},
                            )
                    elif _accepted > 0 and not _wfe_allow_retrain:
                        logger.info(
                            "Phase 53 (US-332): %d accepted epochs skipped — WFE unstable",
                            _accepted,
                        )
            except Exception as _wfr_err:
                logger.debug(f"Phase 52: Walk-forward retrainer error: {_wfr_err}")

        # ── Phase 52 (US-323): Clean up tranche tracker for closed trades ──
        if self._tranche_tracker is not None and synced > 0:
            try:
                _tt_cleaned = 0
                for entry in pending:
                    tid = str(entry.get("trade_id", ""))
                    if tid in closed_trades:
                        self._tranche_tracker.remove_trade(tid)
                        _tt_cleaned += 1
                if _tt_cleaned > 0:
                    logger.info("Phase 52 (US-323): Cleaned %d closed trades from tranche tracker", _tt_cleaned)
            except Exception as _tt_cleanup_err:
                logger.debug(f"Phase 52: Tranche tracker cleanup error: {_tt_cleanup_err}")

        # ── Phase 52 (US-325): Adaptive R:R outcome recording for learning ──
        if self._adaptive_rr is not None and synced > 0:
            try:
                _rr_recorded = 0
                for entry in pending:
                    tid = str(entry.get("trade_id", ""))
                    if tid in closed_trades:
                        _sl = float(entry.get("sl_pips", 0))
                        _tp = float(entry.get("tp_pips", 0))
                        _regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
                        _realized = float(closed_trades[tid].get("realizedPL", 0))
                        _rr_target = _tp / _sl if _sl > 0 else 1.5
                        _actual_rr = abs(_realized) / _sl if _sl > 0 and _realized != 0 else 0.0
                        self._adaptive_rr.record_outcome(
                            rr_target=_rr_target,
                            actual_rr=_actual_rr,
                            won=_realized > 0,
                            regime=str(_regime),
                        )
                        _rr_recorded += 1
                if _rr_recorded > 0:
                    _rr_stats = self._adaptive_rr.get_stats()
                    logger.info(
                        "Phase 52 (US-325): Recorded %d R:R outcomes — "
                        "avg_target=%.2f avg_actual=%.2f win_rate=%.3f",
                        _rr_recorded, _rr_stats.get("avg_target_rr", 0),
                        _rr_stats.get("avg_actual_rr", 0), _rr_stats.get("win_rate", 0),
                    )
            except Exception as _rr_rec_err:
                logger.debug(f"Phase 52: Adaptive R:R recording error: {_rr_rec_err}")

        # Write updated journal (atomic)
        try:
            from src.scanner.automation.safe_json import safe_json_write
            safe_json_write(journal_path, entries)
        except ImportError:
            journal_path.write_text(json.dumps(entries, indent=2, default=str))

        # ── Phase 80 (US-P80-002): TradeOutcomePredictor — auto-train at 50-trade threshold ──
        # Must run AFTER journal write so fit_from_journal() reads the enriched on-disk data.
        if scanner is not None and synced > 0:
            _top = getattr(scanner, "_trade_outcome_predictor", None)
            if _top is not None:
                try:
                    _complete_trades = [
                        e for e in entries
                        if isinstance(e.get("outcome"), dict)
                        and e["outcome"].get("pnl_pips") is not None
                        and e["outcome"].get("mae_pips") is not None
                    ]
                    _n_complete = len(_complete_trades)
                    _should_train = (
                        _n_complete >= 50
                        and (_n_complete % 10 == 0 or not _top._is_trained)
                    )
                    if _should_train:
                        _top_trained = _top.fit_from_journal()
                        if _top_trained:
                            logger.info(
                                "Phase 80 (US-P80-002): TradeOutcomePredictor trained on "
                                "%d trades (accuracy=%.3f)",
                                _n_complete,
                                getattr(_top, "_train_accuracy", 0.0),
                            )
                        else:
                            logger.debug(
                                "Phase 80 (US-P80-002): TradeOutcomePredictor training skipped "
                                "(%d complete trades)", _n_complete
                            )
                    else:
                        logger.debug(
                            "Phase 80 (US-P80-002): TradeOutcomePredictor not yet ready "
                            "(%d / 50 complete trades)", _n_complete
                        )
                except Exception as _top_err:
                    logger.debug("Phase 80 (US-P80-002): TradeOutcomePredictor train error: %s", _top_err)

        # ── Phase 80 (US-P80-003): PairAffinityTracker — update matrix post-trade-close ──
        # Must run AFTER journal write so update_from_journal() reads current on-disk data.
        if scanner is not None and synced > 0:
            _pa = getattr(scanner, "_pair_affinity", None)
            if _pa is not None:
                try:
                    _co_pairs = _pa.update_from_journal()
                    if isinstance(_co_pairs, int) and _co_pairs > 0:
                        logger.info(
                            "Phase 80 (US-P80-003): PairAffinityTracker updated — "
                            "%d co-occurrence pairs processed", _co_pairs
                        )
                    else:
                        logger.debug(
                            "Phase 80 (US-P80-003): PairAffinityTracker updated (0 co-occurrence pairs)"
                        )
                except Exception as _pa_err:
                    logger.debug("Phase 80 (US-P80-003): PairAffinityTracker update error: %s", _pa_err)

        # ── Phase 81 (US-P81-004): CausalCounterfactual — post-loss analysis ──
        # Runs after journal write so counterfactual log has consistent trade IDs.
        if scanner is not None and synced > 0:
            _cc = getattr(scanner, "_causal_counterfactual", None)
            if _cc is not None:
                try:
                    _cc_analyzed = 0
                    _cc_cap = 5  # Max losses to analyze per sync (prevent latency spike)
                    for entry in pending:
                        if _cc_analyzed >= _cc_cap:
                            break
                        tid = str(entry.get("trade_id", ""))
                        if tid not in closed_trades:
                            continue
                        ct = closed_trades[tid]
                        _cc_pl = float(ct.get("realizedPL", 0))
                        if _cc_pl >= 0:
                            continue  # Only analyze losses
                        # Build trade context from journal entry
                        _cc_features = {
                            "atr_14": float(entry.get("regime", {}).get("atr_pips", 0.001)),
                            "sma_trend": float(entry.get("regime", {}).get("uncertainty_score", 0.5)),
                            "momentum_signal": float(entry.get("confidence", 0.5)),
                        }
                        _cc_context = {
                            "pair": entry.get("pair", ""),
                            "features": _cc_features,
                            "regime": str(entry.get("regime", {}).get("volatility_regime", "NORMAL")),
                            "outcome": 0.0,  # Loss
                        }
                        _cc_scenarios = _cc.generate_standard_scenarios(_cc_context)
                        _cc_results = _cc.batch_analyze(_cc_context, _cc_scenarios)
                        _cc.save_analysis(tid, _cc_results)
                        # Log the most impactful reversal scenario
                        _cc_reversals = [r for r in _cc_results if getattr(r, "outcome_reversal", False)]
                        if _cc_reversals:
                            _cc_best = max(_cc_reversals, key=lambda r: abs(getattr(r, "probability_delta", 0.0)))
                            logger.info(
                                "Phase 81 (US-P81-004): Trade %s (%s) loss — "
                                "reversal: %s delta=%.2f",
                                tid, entry.get("pair", ""),
                                getattr(getattr(_cc_best, "scenario", None), "scenario_name", "?"),
                                getattr(_cc_best, "probability_delta", 0.0),
                            )
                        _cc_analyzed += 1
                    if _cc_analyzed > 0:
                        logger.info(
                            "Phase 81 (US-P81-004): CausalCounterfactual analyzed %d losing trades",
                            _cc_analyzed,
                        )
                except Exception as _cc_err:
                    logger.debug("Phase 81 (US-P81-004): CausalCounterfactual error: %s", _cc_err)

        # Phase 45 (US-282): Feed trade outcomes to adaptive position sizer
        if self._adaptive_position_sizer is not None:
            try:
                for entry in pending:
                    tid = str(entry.get("trade_id", ""))
                    if tid not in closed_trades:
                        continue
                    ct = closed_trades[tid]
                    realized_pl = float(ct.get("realizedPL", 0))
                    trade_won = realized_pl > 0
                    self._adaptive_position_sizer.record_trade(
                        pnl=realized_pl,
                        win=trade_won,
                    )
            except Exception as _aps_err:
                logger.debug(f"Phase 45: Adaptive sizer trade recording failed: {_aps_err}")

        # Run RL agent weight updates (regime-aware)
        weights_updated = False
        new_weights: Dict[str, float] = {}
        if rl_updates:
            try:
                from src.scanner.agents import ScannerAgentTeam
                # Phase 24 (US-146): Use scanner config instead of None for proper boost/penalty rates
                _rl_config = getattr(scanner, "config", None) if scanner is not None else None
                if _rl_config is None:
                    try:
                        from src.scanner.config import ScannerConfig
                        _rl_config = ScannerConfig()
                    except Exception as e:
                        logger.debug(f"Execution non-critical operation skipped: {e}")
                agent_team = ScannerAgentTeam(config=_rl_config)
                _weights_before = dict(agent_team._learned_weights.get("_global", agent_team._BASE_WEIGHTS))
                # Phase 24 (US-149): Store original penalty for restoration after trailing-stop override
                _orig_penalty = getattr(_rl_config, "weight_penalty_on_loss", 0.15) if _rl_config else 0.15
                for upd in rl_updates:
                    # US-149: Trailing stop exit learning — reduced penalty for profitable exits
                    _exit_reason = upd.get("exit_reason", "")
                    _is_trailing = "trailing" in str(_exit_reason).lower()
                    _pnl = float(upd.get("pnl", 0))
                    if _is_trailing and _pnl > 0:
                        # Profitable trailing stop: reduced penalty (0.05x instead of 0.15x)
                        # because trailing stop is a SUCCESS mechanism — it locked in profit
                        if _rl_config is not None:
                            setattr(_rl_config, "weight_penalty_on_loss", 0.05)
                        logger.info(
                            f"US-149: Trailing stop profitable exit — "
                            f"reduced penalty 0.05 (was {_orig_penalty:.2f})"
                        )
                    elif _is_trailing and _pnl <= 0:
                        # Trailing stop hit at a loss — SL was tightened but still hit
                        # Normal penalty applies
                        if _rl_config is not None:
                            setattr(_rl_config, "weight_penalty_on_loss", _orig_penalty)
                    else:
                        # Non-trailing exit: use original penalty
                        if _rl_config is not None:
                            setattr(_rl_config, "weight_penalty_on_loss", _orig_penalty)

                    # Regime-aware reward: modulate boost/penalty based on shaped reward
                    _shaped = upd.get("shaped_reward")
                    _orig_boost = getattr(_rl_config, "weight_boost_on_win", 0.10) if _rl_config else 0.10
                    if _shaped is not None and _rl_config is not None:
                        try:
                            # Scale boost/penalty by |shaped_reward| relative to |pnl|
                            # Difficult regimes (HIGH/EXTREME) with profit get amplified reward
                            _abs_pnl = abs(float(upd.get("pnl", 0)))
                            _abs_shaped = abs(_shaped)
                            if _abs_pnl > 0.001:
                                _reward_ratio = min(2.0, max(0.3, _abs_shaped / _abs_pnl))
                            else:
                                _reward_ratio = 1.0
                            setattr(_rl_config, "weight_boost_on_win", round(_orig_boost * _reward_ratio, 4))
                            _current_penalty = getattr(_rl_config, "weight_penalty_on_loss", 0.15)
                            setattr(_rl_config, "weight_penalty_on_loss", round(_current_penalty * _reward_ratio, 4))
                        except Exception as _mod_err:
                            logger.debug(f"Regime reward modulation skipped: {_mod_err}")

                    new_weights = agent_team.update_weights_from_outcome(
                        agent_verdicts=upd["agent_verdicts"],
                        trade_won=upd["trade_won"],
                        regime=upd.get("regime", "NORMAL"),
                    )

                    # Restore boost after each trade (penalty restored below or in trailing logic)
                    if _rl_config is not None:
                        setattr(_rl_config, "weight_boost_on_win", _orig_boost)

                # Restore original penalty
                if _rl_config is not None:
                    setattr(_rl_config, "weight_penalty_on_loss", _orig_penalty)

                weights_updated = True
                # Log weight changes for observability
                _changed = {
                    k: f"{_weights_before.get(k, 0):.3f}→{new_weights.get(k, 0):.3f}"
                    for k in new_weights
                    if abs(new_weights.get(k, 0) - _weights_before.get(k, 0)) > 0.001
                }
                # US-149: Include exit reason distribution in log
                _exit_reasons = {}
                for upd in rl_updates:
                    _er = upd.get("exit_reason", "unknown")
                    _exit_reasons[_er] = _exit_reasons.get(_er, 0) + 1
                logger.info(
                    f"RL feedback: updated weights from {len(rl_updates)} closed trades "
                    f"(regime-aware, exit_reasons={_exit_reasons}, changed: {_changed or 'none'})"
                )
            except Exception as e:
                logger.warning(f"RL weight update failed: {e}")

        # Phase 53 (US-329): Update pair-specific Bayesian weights
        if self._pair_bayesian_weights is not None and rl_updates:
            try:
                _pair_updates = 0
                for upd in rl_updates:
                    _pair = upd.get("pair", "")
                    _verdicts = upd.get("agent_verdicts", {})
                    _won = upd.get("trade_won", False)
                    if _pair and _verdicts:
                        # Convert verdicts to scores dict {agent: score}
                        _scores = {}
                        for agent_name, verdict in _verdicts.items():
                            if isinstance(verdict, dict):
                                _scores[agent_name] = float(verdict.get("score", 0.5))
                            elif isinstance(verdict, (int, float)):
                                _scores[agent_name] = float(verdict)
                        if _scores:
                            self._pair_bayesian_weights.update(
                                pair=_pair,
                                agent_scores=_scores,
                                outcome="win" if _won else "loss",
                            )
                            _pair_updates += 1
                if _pair_updates > 0:
                    self._pair_bayesian_weights.save_state()
                    logger.info(
                        "Phase 53 (US-329): Updated pair-specific weights for %d trades",
                        _pair_updates,
                    )
            except Exception as _pw_err:
                logger.debug(f"Phase 53: Pair weight update error: {_pw_err}")

        # Phase 25 (US-155): Record agent verdicts to accuracy matrix for per-pair tracking
        # Uses synced_entries (not full journal) to avoid double-counting on repeated sync calls
        if scanner is not None and synced > 0:
            _acc_matrix = getattr(scanner, "_agent_accuracy_matrix", None)
            if _acc_matrix is not None:
                try:
                    _am_recorded = 0
                    for entry in synced_entries:
                        _outcome = entry.get("outcome")
                        if _outcome is None:
                            continue
                        _pair = entry.get("pair", "")
                        _regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
                        _direction = entry.get("direction", "BUY").upper()
                        _trade_won = bool(_outcome.get("trade_won", False))
                        _actual = "win" if _trade_won else "loss"
                        _agents_info = entry.get("agents", {})
                        _agent_reasons = _agents_info.get("agent_reasons", [])
                        for _ar in _agent_reasons:
                            _agent_name = _ar.get("name", "")
                            if not _agent_name:
                                continue
                            # Only record agents that voted FOR the trade (passed=True)
                            # Their prediction was the trade direction — outcome tells if correct
                            if _ar.get("passed", False):
                                _acc_matrix.record_verdict(
                                    agent_name=_agent_name,
                                    pair=_pair,
                                    regime=_regime,
                                    predicted_direction=_direction,
                                    actual_outcome=_actual,
                                )
                                _am_recorded += 1
                    if _am_recorded > 0:
                        _acc_matrix.save_state()
                        logger.info(
                            "US-155: Recorded %d agent verdicts to accuracy matrix "
                            "(%d closed trades)",
                            _am_recorded, synced,
                        )
                except Exception as _am_err:
                    logger.debug(f"US-155: Accuracy matrix recording error: {_am_err}")

        # Phase 27 (US-167): Feed outcomes to pair_regime_agent_matrix
        # Uses synced_entries (not full journal) to avoid double-counting
        if scanner is not None and synced > 0:
            _pram = getattr(scanner, "_pair_regime_agent_matrix", None)
            if _pram is not None:
                try:
                    _pram_recorded = 0
                    for entry in synced_entries:
                        _outcome = entry.get("outcome")
                        if _outcome is None:
                            continue
                        _pair = entry.get("pair", "")
                        _regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
                        _direction = entry.get("direction", "BUY").upper()
                        _won = bool(_outcome.get("trade_won", False))
                        _agent_reasons = entry.get("agents", {}).get("agent_reasons", [])
                        for _ar in _agent_reasons:
                            _aname = _ar.get("name", "")
                            if _aname and _ar.get("passed", False):
                                _pram.record_vote(
                                    pair=_pair,
                                    regime=_regime,
                                    agent=_aname,
                                    vote=_direction,
                                    won=_won,
                                )
                                _pram_recorded += 1
                    if _pram_recorded > 0:
                        _pram.save_state()
                        logger.info(
                            "US-167: Fed %d agent outcomes to pair_regime_agent_matrix",
                            _pram_recorded,
                        )
                except Exception as _pram_err:
                    logger.debug(f"US-167: PRAM feed error: {_pram_err}")

        # Phase 27 (US-166): Track fast-track trade A/B performance
        if synced > 0:
            try:

                for entry in entries:
                    _outcome = entry.get("outcome")
                    if _outcome is None:
                        continue
                    _ctx = entry.get("analysis_context", entry.get("agents", {}))
                    _is_ft = _ctx.get("fasttrack", False)
                    if _is_ft:
                        self._fasttrack_stats["total"] += 1
                        if _outcome.get("trade_won", False):
                            self._fasttrack_stats["wins"] += 1
                        else:
                            self._fasttrack_stats["losses"] += 1
                _ft_total = self._fasttrack_stats["total"]
                if _ft_total > 0 and _ft_total % 5 == 0:
                    _ft_wr = self._fasttrack_stats["wins"] / _ft_total
                    logger.info(
                        "US-166: Fast-track A/B — %d trades, "
                        "win_rate=%.1f%% (%d W / %d L)",
                        _ft_total, _ft_wr * 100,
                        self._fasttrack_stats["wins"],
                        self._fasttrack_stats["losses"],
                    )
            except Exception as _ft_err:
                logger.debug(f"US-166: Fast-track tracking error: {_ft_err}")

        # Phase 18: Feed trade outcomes to calibration modules via Scanner
        if scanner is not None and synced > 0:
            try:
                _phase18_method = getattr(scanner, "record_trade_outcome_phase18", None)
                if _phase18_method is not None:
                    for entry in entries:
                        _outcome = entry.get("outcome")
                        if _outcome is None:
                            continue
                        _gate_vals = None
                        _agents_info = entry.get("agents", {})
                        if _agents_info:
                            _gate_vals = {
                                "confidence": float(entry.get("confidence", 0)),
                                "momentum": float(_agents_info.get("weighted_vote_score", 0)),
                                "rr_ratio": (
                                    float(entry.get("tp_pips", 0)) / max(float(entry.get("sl_pips", 1)), 0.1)
                                ),
                            }
                        _phase18_method(
                            pair=entry.get("pair", ""),
                            regime=entry.get("regime", {}).get("volatility_regime", "NORMAL"),
                            pnl_pips=float(_outcome.get("pnl_pips", 0)),
                            trade_won=bool(_outcome.get("trade_won", False)),
                            duration_bars=int(_outcome.get("duration_minutes", 0) / 5),
                            gate_values=_gate_vals,
                            agent_verdicts=_agents_info.get("agent_reasons"),
                            model=entry.get("model", "ensemble"),
                        )
                    logger.info(f"Phase 18: Fed {synced} trade outcomes to calibration modules")
            except Exception as _p18_err:
                logger.debug(f"Phase 18 outcome feedback error: {_p18_err}")

        # Update per-pair performance summary after syncing
        if synced > 0:
            try:
                self._update_pair_performance()
            except Exception as perf_err:
                logger.debug(f"Pair performance update error: {perf_err}")

        # Phase 45 (US-282): Persist adaptive sizer state
        if self._adaptive_position_sizer is not None:
            try:
                self._adaptive_position_sizer.save_state("trained_data/adaptive_sizer_state.json")
            except Exception as _save_err:
                logger.debug(f"Phase 45: Adaptive sizer state save failed: {_save_err}")

        # Phase 45 (US-285): Persist EWMA correlation state
        if self._ewma_correlation is not None:
            try:
                self._ewma_correlation.save_state("trained_data/ewma_correlation_state.json")
            except Exception as _ewma_save_err:
                logger.debug(f"Phase 45: EWMA state save failed: {_ewma_save_err}")

        # Phase 53 (US-328): Persist gate attribution state
        if self._gate_attribution is not None:
            try:
                self._gate_attribution.save_state()
            except Exception as _ga_save_err:
                logger.debug(f"Phase 53: Gate attribution state save failed: {_ga_save_err}")

        # Persist regime reward log after all trades processed
        if self._regime_reward is not None and synced > 0:
            try:
                self._regime_reward.save_log()
            except Exception as _rr_save_err:
                logger.debug(f"Regime reward log save failed: {_rr_save_err}")

        return {
            "trades_synced": synced,
            "weights_updated": weights_updated,
            "new_weights": new_weights,
            "detail": f"synced {synced} trades, {len(rl_updates)} RL updates",
        }

    def _record_expectancy_from_trades(
        self, pending: List[Dict[str, Any]], closed_trades: Dict[str, Dict[str, Any]]
    ) -> None:
        """Phase 47 (US-297): Record trade outcomes to expectancy tracker for per-agent per-regime learning.

        Iterates through pending trades, finds matches in closed_trades, and records each
        agent's performance in the regime where the trade executed.

        Args:
            pending: List of pending entries from trade journal
            closed_trades: Dict mapping trade_id -> OANDA trade data
        """
        if self._expectancy_tracker is None:
            self._init_expectancy_tracker()
        if self._expectancy_tracker is None:
            return

        recorded_count = 0
        try:
            for entry in pending:
                tid = str(entry.get("trade_id", ""))
                if tid not in closed_trades:
                    continue

                ct = closed_trades[tid]
                realized_pl = float(ct.get("realizedPL", 0))
                trade_won = realized_pl > 0

                # Get regime and agent verdicts
                regime = entry.get("regime", {}).get("volatility_regime", "NORMAL")
                agents_info = entry.get("agents", {})
                agent_reasons = agents_info.get("agent_reasons", [])

                # Record each agent that participated
                for agent_reason in agent_reasons:
                    agent_name = agent_reason.get("name", "")
                    if not agent_name:
                        continue

                    try:
                        self._expectancy_tracker.record_trade(
                            agent_name=agent_name,
                            regime=regime,
                            pnl=realized_pl,
                            won=trade_won,
                        )
                        recorded_count += 1
                    except ValueError as e:
                        # Graceful skip for unknown agent/regime
                        logger.debug(
                            f"Phase 47 (US-297): Could not record expectancy for {agent_name}/{regime}: {e}"
                        )

            # Save state after recording
            if recorded_count > 0:
                try:
                    self._expectancy_tracker.save_state()
                    logger.debug(
                        f"Phase 47 (US-297): Recorded expectancy for {recorded_count} agent-trade outcomes"
                    )
                except Exception as save_err:
                    logger.warning(f"Phase 47 (US-297): Failed to save expectancy state: {save_err}")

        except Exception as e:
            logger.warning(f"Phase 47 (US-297): Expectancy recording failed: {e}")

    def _sync_trade_outcome_to_rl(self, entry: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """Append (state, action, reward) tuple to RL replay buffer.

        This feeds the RL sizer/gates/exits models with live trade outcomes.
        Reward = normalized P/L relative to risk taken (pnl / sl_pips).

        Args:
            entry: Journal entry with trade context (pair, direction, confidence, etc.)
            outcome: Outcome dict with realized_pl, pnl_pips, trade_won, close_price, close_time.
        """
        import json
        from pathlib import Path

        buffer_path = Path("trained_data/rl_replay_buffer.jsonl")

        try:
            sl_pips = float(entry.get("sl_pips", 0) or entry.get("stop_loss_pips", 0) or 1.0)
            pnl_pips = float(outcome.get("pnl_pips", 0))
            # Reward = normalized P/L relative to risk taken
            reward = pnl_pips / sl_pips if sl_pips > 0 else 0.0

            state = {
                "pair": entry.get("pair"),
                "direction": entry.get("direction"),
                "confidence": entry.get("confidence"),
                "atr": entry.get("atr"),
                "regime": entry.get("regime", {}).get("volatility_regime", "NORMAL"),
                "agent_consensus": entry.get("agents", {}).get("weighted_vote_score"),
                "uncertainty": entry.get("agents", {}).get("uncertainty_score"),
                "risk_pct": entry.get("risk_pct"),
                "lots": entry.get("lots"),
            }

            action = {
                "type": "trade",
                "direction": entry.get("direction"),
                "lots": entry.get("lots"),
                "sl_pips": sl_pips,
                "tp_pips": entry.get("tp_pips"),
            }

            record = {
                "timestamp": outcome.get("close_time"),
                "state": state,
                "action": action,
                "reward": round(reward, 4),
                "pnl_pips": pnl_pips,
                "realized_pl": outcome.get("realized_pl", 0),
                "trade_won": outcome.get("trade_won", False),
            }

            # Phase 55 (US-341): Atomic JSONL append with file locking
            try:
                from src.scanner.safe_io import safe_jsonl_append
                safe_jsonl_append(buffer_path, record)
            except ImportError:
                with open(buffer_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")

            logger.debug(f"RL replay: {entry.get('pair')} reward={reward:.3f}")

        except Exception as e:
            logger.warning(f"RL replay buffer write failed: {e}")  # L-6: elevated from debug; RL data loss must surface

    def _update_pair_performance(self) -> None:
        """Aggregate per-pair performance stats from the trade journal."""
        import json
        from pathlib import Path
        from collections import defaultdict

        journal_path = Path("trained_data/trade_journal_rl.json")
        perf_path = Path("trained_data/models/pair_performance.json")

        if not journal_path.exists():
            return

        # Phase 31 (US-188): Guard JSON parse with try/except
        try:
            entries = json.loads(journal_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"_update_pair_performance: journal parse failed: {e}")
            return
        if not isinstance(entries, list):
            logger.warning("_update_pair_performance: journal is not a list")
            return
        closed = [e for e in entries if e.get("outcome") is not None]
        if not closed:
            return

        stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
            "total_pnl_pips": 0.0, "avg_confidence": 0.0,
            "best_pnl_pips": 0.0, "worst_pnl_pips": 0.0,
        })

        for e in closed:
            pair = e.get("pair", "UNKNOWN")
            outcome = e["outcome"]
            s = stats[pair]
            s["trades"] += 1
            pnl_pips = float(outcome.get("pnl_pips", 0))
            realized_pl = float(outcome.get("realized_pl", 0))
            won = bool(outcome.get("trade_won", False))

            if won:
                s["wins"] += 1
            else:
                s["losses"] += 1

            s["total_pnl"] += realized_pl
            s["total_pnl_pips"] += pnl_pips
            s["avg_confidence"] += float(e.get("confidence", 0))
            s["best_pnl_pips"] = max(s["best_pnl_pips"], pnl_pips)
            s["worst_pnl_pips"] = min(s["worst_pnl_pips"], pnl_pips)

        # Finalize averages
        for pair, s in stats.items():
            if s["trades"] > 0:
                s["avg_confidence"] = round(s["avg_confidence"] / s["trades"], 3)
                s["win_rate"] = round(s["wins"] / s["trades"], 3)
            else:
                s["win_rate"] = 0.0
            s["total_pnl"] = round(s["total_pnl"], 2)
            s["total_pnl_pips"] = round(s["total_pnl_pips"], 1)
            s["best_pnl_pips"] = round(s["best_pnl_pips"], 1)
            s["worst_pnl_pips"] = round(s["worst_pnl_pips"], 1)

        perf_path.parent.mkdir(parents=True, exist_ok=True)
        # Phase 55 (US-341): Atomic write with locking via safe_io
        try:
            from src.scanner.safe_io import safe_json_write
            safe_json_write(perf_path, dict(stats))
        except ImportError:
            # M-4: Fallback atomic write
            _perf_tmp = perf_path.with_suffix(".tmp")
            _perf_tmp.write_text(json.dumps(dict(stats), indent=2), encoding="utf-8")
            import os as _os; _os.replace(str(_perf_tmp), str(perf_path))

    def get_pair_performance(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """Read per-pair performance stats.

        Args:
            pair: Specific pair to query, or None for all pairs.

        Returns:
            Dict with pair -> stats mapping, or single pair stats.
        """
        import json
        from pathlib import Path

        perf_path = Path("trained_data/models/pair_performance.json")
        if not perf_path.exists():
            return {}

        try:
            data = json.loads(perf_path.read_text())
        except Exception as e:
            logger.debug("US-178: Pair performance data parse failed: %s", e)
            return {}

        if pair:
            return data.get(pair, {})
        return data

    # ── Trade Journal Summary Stats ──────────────────────────────────

    def win_rate_by_pair(self) -> Dict[str, float]:
        """Calculate win rate for each pair from the trade journal.

        Returns:
            Dict mapping pair name to win rate (0.0 - 1.0).
        """
        import json
        from pathlib import Path
        from collections import defaultdict

        journal_path = Path("trained_data/trade_journal_rl.json")
        if not journal_path.exists():
            return {}

        try:
            entries = json.loads(journal_path.read_text())
        except Exception as e:
            logger.debug("US-178: Journal parse failed in win_rate_by_pair: %s", e)
            return {}

        wins: Dict[str, int] = defaultdict(int)
        total: Dict[str, int] = defaultdict(int)

        for e in entries:
            outcome = e.get("outcome")
            if outcome is None:
                continue
            pair = e.get("pair", "UNKNOWN")
            total[pair] += 1
            if outcome.get("trade_won"):
                wins[pair] += 1

        return {
            pair: round(wins[pair] / total[pair], 3) if total[pair] > 0 else 0.0
            for pair in total
        }

    def avg_duration(self, pair: Optional[str] = None) -> float:
        """Calculate average trade duration in minutes.

        Args:
            pair: Optional pair filter. None = all pairs.

        Returns:
            Average duration in minutes, or 0.0 if no data.
        """
        import json
        from pathlib import Path

        journal_path = Path("trained_data/trade_journal_rl.json")
        if not journal_path.exists():
            return 0.0

        try:
            entries = json.loads(journal_path.read_text())
        except Exception as e:
            logger.debug("US-178: Journal parse failed in avg_trade_duration: %s", e)
            return 0.0

        durations = []
        for e in entries:
            outcome = e.get("outcome")
            if outcome is None:
                continue
            if pair and e.get("pair") != pair:
                continue
            dur = outcome.get("duration_minutes")
            if dur is not None and dur > 0:
                durations.append(dur)

        return round(sum(durations) / len(durations), 1) if durations else 0.0

    def avg_rr_ratio(self, pair: Optional[str] = None) -> float:
        """Calculate average realized risk:reward ratio from closed trades.

        R:R = |pnl_pips| / sl_distance_pips for winners, negative for losers.

        Args:
            pair: Optional pair filter. None = all pairs.

        Returns:
            Average R:R ratio, or 0.0 if no data.
        """
        import json
        from pathlib import Path

        journal_path = Path("trained_data/trade_journal_rl.json")
        if not journal_path.exists():
            return 0.0

        try:
            entries = json.loads(journal_path.read_text())
        except Exception as e:
            logger.debug("US-178: Journal parse failed in avg_rr_ratio: %s", e)
            return 0.0

        ratios = []
        for e in entries:
            outcome = e.get("outcome")
            if outcome is None:
                continue
            if pair and e.get("pair") != pair:
                continue

            pnl_pips = abs(float(outcome.get("pnl_pips", 0)))
            # Estimate SL distance from entry
            entry_price = float(e.get("entry_price", 0))
            sl_price = float(e.get("sl_price", 0))
            pip_val = _get_pip_value(e.get("pair", ""), 0.0001)

            if entry_price > 0 and sl_price > 0 and pip_val > 0:
                sl_dist = abs(entry_price - sl_price) / pip_val
                if sl_dist > 0:
                    rr = pnl_pips / sl_dist
                    if not outcome.get("trade_won"):
                        rr = -rr
                    ratios.append(rr)

        return round(sum(ratios) / len(ratios), 2) if ratios else 0.0

    def execution_quality_summary(self) -> Dict[str, Any]:
        """Calculate execution quality metrics from trade journal.

        Returns:
            Dict with avg_slippage_pips, max_slippage_pips, fill_rate,
            total_trades, rejected_count.
        """
        import json
        from pathlib import Path

        journal_path = Path("trained_data/trade_journal_rl.json")
        if not journal_path.exists():
            return {"avg_slippage_pips": 0.0, "max_slippage_pips": 0.0,
                    "fill_rate": 1.0, "total_trades": 0, "rejected_count": 0}

        try:
            entries = json.loads(journal_path.read_text())
        except Exception as e:
            logger.debug("US-178: Journal parse failed in execution_quality_stats: %s", e)
            return {"avg_slippage_pips": 0.0, "max_slippage_pips": 0.0,
                    "fill_rate": 1.0, "total_trades": 0, "rejected_count": 0}

        slippages = []
        for e in entries:
            slip = e.get("slippage_pips")
            if slip is not None:
                slippages.append(abs(float(slip)))

        total = len(entries)
        avg_slip = round(sum(slippages) / len(slippages), 2) if slippages else 0.0
        max_slip = round(max(slippages), 2) if slippages else 0.0

        # Count outcomes to estimate fill rate
        filled = sum(1 for e in entries if e.get("outcome") is not None or e.get("fill_price"))
        fill_rate = round(filled / total, 3) if total > 0 else 1.0

        return {
            "avg_slippage_pips": avg_slip,
            "max_slippage_pips": max_slip,
            "fill_rate": fill_rate,
            "total_trades": total,
            "rejected_count": total - filled,
        }
