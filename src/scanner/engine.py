"""
Scanner Engine Module.

Core scanning logic using ThreadPoolExecutor for parallel pair analysis.
Handles data fetching, feature engineering, and incremental caching.
"""

from __future__ import annotations

from copy import deepcopy
import json as _json
import logging
import math
import os
import platform
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ScannerConfig, load_yaml_config
from .agents import ScannerAgentTeam
from .gates import GateEvaluator
from .backtest_gate import BacktestGate
from .results import PairAnalysis, ScanResult
from .execution import ExecutionManager, ExecutionConfig, ExecutionResult
from .analysis import QuickBacktester, CorrelationAnalyzer, DriftDetector
from .filters import VolatilityFilter, DiversificationFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# M1 Metal / TensorFlow environment defaults (set before TF is imported)
# ---------------------------------------------------------------------------
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')       # Suppress TF info/warning spam
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')      # Disable oneDNN (x86 only, noise on arm64)

_m1_metal_configured = False


def _configure_m1_metal() -> bool:
    """Configure TensorFlow for optimal M1 Metal GPU usage.

    Safe to call multiple times (idempotent via module-level flag).
    Must be called BEFORE any model loading or tf.function compilation.

    On Apple Silicon with 8 GB unified memory the Metal GPU shares the same
    pool as WindowServer.  Without memory growth limits TF can allocate the
    entire pool, starving the OS and triggering a kernel panic (userspace
    watchdog timeout).

    Returns True if M1 Metal configuration was applied, False otherwise.
    """
    global _m1_metal_configured
    if _m1_metal_configured:
        return True
    _m1_metal_configured = True

    if not (platform.machine() == 'arm64' and platform.system() == 'Darwin'):
        return False

    try:
        import tensorflow as tf
    except ImportError:
        return False

    try:
        # 1. Threading — M1 has 4 perf + 4 efficiency cores.
        #    Keep TF on performance cores; leave efficiency for OS / Python.
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)

        # 2. GPU memory growth — prevent TF from grabbing all Metal memory.
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # Already initialized
            logger.info(
                "M1 Metal GPU detected: %d device(s). Memory growth enabled.",
                len(gpus),
            )

            # Hard cap Metal memory — default 1536 MB on 8GB M1 to leave room
            # for OS, WindowServer, and Python heap.  Override via env var.
            mem_limit = os.environ.get('BUDDY_GPU_MEMORY_LIMIT_MB', '1536')
            if mem_limit:
                try:
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=int(mem_limit))],
                    )
                    logger.info("M1 Metal: GPU memory hard-capped at %s MB", mem_limit)
                except (RuntimeError, ValueError):
                    pass  # memory_growth and logical config can conflict
        else:
            logger.info("M1 Apple Silicon detected (CPU-only — tensorflow-metal not installed)")

        # 3. Mixed precision for Metal — halves memory, ~1.5x speedup.
        #    Only enable when GPU is available (no benefit on CPU-only).
        if gpus:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("M1 Metal: mixed precision (float16) enabled")
            except Exception as e:
                logger.debug("M1 Metal: mixed precision not available: %s", e)

        logger.info(
            "M1 Metal: threading set to 2 inter-op, 4 intra-op (TF %s)",
            tf.__version__,
        )
        return True

    except Exception as e:
        logger.debug("M1 Metal config failed (non-fatal): %s", e)
        return False


class Scanner:
    """FX Pair Scanner with parallel execution and incremental updates.

    Key features:
    - ThreadPoolExecutor for parallel pair scanning
    - Incremental caching to skip unchanged pairs
    - CatBoost/XGBoost gate models
    - Non-interactive mode for scripts
    - Rich display integration
    """

    def __init__(
        self,
        config: Optional[ScannerConfig] = None,
        oanda_client: Optional[Any] = None,
        broker: Optional[Any] = None,
    ):
        """Initialize scanner.

        Args:
            config: Scanner configuration (loads default if None)
            oanda_client: OANDA API client (created if None, deprecated — use broker instead)
            broker: BrokerClient instance for data fetching (created if None)
        """
        # M1 Metal / TF configuration — MUST run before any model loading
        _configure_m1_metal()

        self.config = config or ScannerConfig()
        self._oanda = oanda_client
        self._broker = broker

        # M1 memory safety: detect low-RAM systems and cap resource usage
        try:
            import psutil as _psutil
            _mem = _psutil.virtual_memory()
            _available_mb = _mem.available / (1024 * 1024)
            _total_mb = _mem.total / (1024 * 1024)
        except ImportError:
            _available_mb = 4096  # Assume enough if can't check
            _total_mb = 8192
        self._is_low_memory = _available_mb < 2048  # Less than 2GB free
        if self._is_low_memory:
            logger.warning(
                "Low memory detected: %.0fMB available of %.0fMB total. "
                "Limiting module loading and thread workers.",
                _available_mb, _total_mb,
            )
            # Cap parallel workers to reduce concurrent memory pressure
            if self.config.parallel_workers > 2:
                self.config.parallel_workers = 2
            if getattr(self.config, 'sub_inference_workers', 3) > 2:
                self.config.sub_inference_workers = 2

        # Phase 69: TF/Keras warm-up — eagerly load the ensemble models
        # before heavy Scanner init. Prevents segfault on Apple Silicon
        # (TF 2.21 + Keras 3.13) when keras.models.load_model() is called
        # after many Python modules have been imported and fragmented the
        # process memory space.
        try:
            from modular_inference import ModularEnsembleInference as _MEI
            self._modular_ensemble = _MEI(
                model_dir=str(self.config.model_dir),
                use_rl_sizer=bool(self.config.use_rl_sizer),
                use_rl_gates=bool(self.config.use_rl_gates),
                use_rl_exits=bool(self.config.use_rl_exits),
                enable_market_intelligence=False,
            )
            # Scanner uses opportunity mode: no model contracts, no required models
            if hasattr(self._modular_ensemble, 'config'):
                self._modular_ensemble.config.enforce_model_contracts = False
                self._modular_ensemble.config.require_direction_model = False
                self._modular_ensemble.config.require_volatility_model = False
            self._modular_ensemble.load_models()
            self._ensemble_loaded = (
                self._modular_ensemble._loaded
                and (
                    self._modular_ensemble.tcn is not None
                    or getattr(self._modular_ensemble, "regime_model", None) is not None
                    or getattr(self._modular_ensemble, "histgb", None) is not None
                )
            )
            if self._ensemble_loaded:
                self._ensemble_type = "ModularEnsembleInference"
            logger.info("Phase 69: Keras models pre-loaded (loaded=%s)", self._ensemble_loaded)
        except Exception as _warmup_err:
            logger.debug(f"Phase 69: Keras warm-up deferred: {_warmup_err}")
            self._modular_ensemble = None

        # Hot-reload handler: subscribe to MODEL_PROMOTED so this scanner
        # picks up fresh models without a restart. Safe no-op if bus unavailable.
        try:
            from src.scanner.model_reload_handler import register_model_reload_handler
            register_model_reload_handler(self)
        except Exception as _rh_err:
            logger.debug("Model reload handler registration deferred: %s", _rh_err)

        # Lazy-loaded components
        self._gate_evaluator: Optional[GateEvaluator] = None
        self._feature_engineer = None
        # NOTE: _modular_ensemble may already be set by Phase 69 warm-up above
        if not hasattr(self, '_modular_ensemble') or self._modular_ensemble is None:
            self._modular_ensemble = None
        self._executor: Optional[ExecutionManager] = None

        # Ensemble health tracking (US-033)
        self._ensemble_health: Dict[str, Any] = {
            "available": [],
            "missing": [],
            "expected_count": 5,
            "degraded": False,
            "penalty": 1.0,
        }

        # Analysis and filter components
        self._backtester: Optional[QuickBacktester] = None
        self._backtest_gate: Optional[BacktestGate] = None
        self._correlation_analyzer: Optional[CorrelationAnalyzer] = None
        self._drift_detector: Optional[DriftDetector] = None
        self._volatility_filter: Optional[VolatilityFilter] = None
        self._diversification_filter: Optional[DiversificationFilter] = None

        # EWMA correlation engine — fed from scan loop so ALL pairs contribute
        self._ewma_engine = None
        self._ewma_prev_prices: Dict[str, float] = {}

        # Incremental caching
        self._last_scan_times: Dict[str, float] = {}
        self._last_prices: Dict[str, float] = {}
        self._cached_results: Dict[str, PairAnalysis] = {}
        self._cache_contexts: Dict[str, Tuple[Any, ...]] = {}
        self._pair_returns: Dict[str, pd.Series] = {}
        self._feature_snapshots: Dict[str, pd.DataFrame] = {}
        self._raw_snapshots: Dict[str, pd.DataFrame] = {}
        self._scan_cycle_count: int = 0
        self._ensemble_lock = threading.Lock()
        self._agent_team = ScannerAgentTeam(self.config)

        # Init automation modules (regime tracker, chain memory, etc.)
        self._init_automation_modules()

    def clear_scan_caches(self) -> int:
        """Clear inter-scan caches to free memory. Returns number of entries freed."""
        freed = 0
        for cache in (self._cached_results, self._cache_contexts,
                      self._pair_returns, self._feature_snapshots, self._raw_snapshots):
            freed += len(cache)
            cache.clear()
        return freed

    def shutdown_rl_workers(self) -> None:
        """Terminate RL subprocess workers if running."""
        mei = getattr(self, "_modular_ensemble", None)
        if mei is None:
            return
        for attr in ("_rl_gates_proc", "_rl_exits_proc"):
            proc = getattr(mei, attr, None)
            if proc is None or not proc.is_alive():
                continue
            try:
                proc.terminate()
                proc.join(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            finally:
                setattr(mei, attr, None)

    def _init_automation_modules(self) -> None:
        """Initialize all automation/Phase-50+ modules.

        Previously this code was accidentally nested inside shutdown_rl_workers
        due to indentation, meaning none of these attributes were set during
        __init__. Now called explicitly from __init__.
        """
        # Shared observation logger for feedback hooks (lazy-init)
        self._feedback_observer = None

        # Per-pair previous regime cache (for RegimeBroadcaster transition detection)
        self._pair_last_regime: Dict[str, str] = {}

        # Regime transition tracker (Phase 8: Markov chain)
        self._regime_tracker = None
        try:
            from src.scanner.automation.regime_tracker import RegimeTransitionTracker
            self._regime_tracker = RegimeTransitionTracker()
        except Exception as e:
            logger.debug(f"Regime tracker init deferred: {e}")

        # Group momentum aggregator (Phase 8: cross-pair signal aggregation)
        self._group_momentum = None
        try:
            from src.scanner.automation.group_momentum import GroupMomentumAggregator
            self._group_momentum = GroupMomentumAggregator()
        except Exception as e:
            logger.debug(f"Group momentum aggregator init deferred: {e}")

        # Tier 6: meta-learning (optional)
        self._meta_learner: Optional[Any] = None
        if getattr(self.config, 'enable_meta_learning', False):
            try:
                from src.recursive_intelligence.meta_learner import MetaLearner
                self._meta_learner = MetaLearner(enabled=True)
                logger.info("Meta-learner initialized (Tier 6)")
            except Exception as e:
                logger.warning("Meta-learner init failed: %s — continuing without it", e)

        # Tier 6: Differentiable ensemble weighting (learned regime→model weights)
        self._ensemble_weighter: Optional[Any] = None
        if getattr(self.config, 'enable_meta_learning', False):
            try:
                from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig
                self._ensemble_weighter = EnsembleWeighter(
                    config=WeighterConfig(n_models=4, n_context=2)
                )
                logger.info("Ensemble weighter initialized (Tier 6)")
            except Exception as e:
                logger.warning("Ensemble weighter init failed: %s — continuing without it", e)

        # Tier 6: MAML Ridge prototype (shadow mode, auto-promotes when benchmark passes)
        self._maml_ridge: Optional[Any] = None
        self._maml_benchmark: Optional[Any] = None
        if getattr(self.config, 'enable_meta_learning', False):
            try:
                from src.recursive_intelligence.maml_ridge import MAMLRidge, MAMLConfig
                self._maml_ridge = MAMLRidge(config=MAMLConfig(
                    shadow_mode=True,     # Starts in shadow; promotion flips this
                    auto_promote=True,    # Self-promote when benchmark says PROMOTE
                ))
                _mode = "PROMOTED" if self._maml_ridge.is_promoted else "shadow"
                logger.info("MAML Ridge initialized (%s mode)", _mode)
            except Exception as e:
                logger.debug("MAML Ridge init skipped: %s", e)

            # Tier 6 Phase 2: Benchmarking infrastructure
            try:
                from src.recursive_intelligence.maml_benchmark import MAMLBenchmark
                self._maml_benchmark = MAMLBenchmark()
                # Connect benchmark to MAML Ridge for meta-train event recording
                if self._maml_ridge is not None:
                    self._maml_ridge._benchmark = self._maml_benchmark
                logger.info("MAML Benchmark initialized (Phase 2)")
            except Exception as e:
                logger.debug("MAML Benchmark init skipped: %s", e)

        # Macro stress detector (Phase 10: derived stress indicator)
        self._macro_stress = None
        try:
            from src.scanner.automation.macro_stress import MacroStressDetector
            self._macro_stress = MacroStressDetector()
        except Exception as e:
            logger.debug(f"Macro stress detector init deferred: {e}")

        # Seasonality tracker (Phase 10: hour-of-day confidence adjustment)
        self._seasonality = None
        try:
            from src.scanner.automation.seasonality import SeasonalityTracker
            self._seasonality = SeasonalityTracker()
        except Exception as e:
            logger.debug(f"Seasonality tracker init deferred: {e}")

        # Ensemble disagreement signal (Phase 14: meta-signal from member disagreement)
        self._ensemble_disagreement = None
        if getattr(self.config, "enable_ensemble_disagreement", False):
            try:
                from src.scanner.automation.ensemble_disagreement import EnsembleDisagreementSignal
                self._ensemble_disagreement = EnsembleDisagreementSignal()
                logger.info("Ensemble disagreement signal initialized")
            except Exception as e:
                logger.debug(f"Ensemble disagreement init deferred: {e}")

        # Position timeout manager (Phase 14: regime-dependent time-decay for open trades)
        self._position_timeout = None
        if getattr(self.config, "enable_position_timeout", False):
            try:
                from src.scanner.automation.position_timeout import PositionTimeoutManager
                self._position_timeout = PositionTimeoutManager()
                logger.info("Position timeout manager initialized")
            except Exception as e:
                logger.debug(f"Position timeout init deferred: {e}")

        # Lead-lag detector (Phase 14: cross-pair lagged correlation for entry timing)
        self._lead_lag_detector = None
        if getattr(self.config, "enable_lead_lag_detection", False):
            try:
                from src.scanner.automation.lead_lag_detector import LeadLagDetector
                self._lead_lag_detector = LeadLagDetector()
                logger.info("Lead-lag detector initialized")
            except Exception as e:
                logger.debug(f"Lead-lag detector init deferred: {e}")

        # Roll calendar (US-016: contract roll management for futures)
        self._roll_calendar = None
        try:
            from src.brokers.roll_calendar import RollCalendar
            self._roll_calendar = RollCalendar()
            logger.info("Roll calendar initialized")
        except Exception as e:
            logger.debug(f"Roll calendar init deferred: {e}")

        # Feature attention layer (Phase 14: regime-conditioned softmax attention)
        self._feature_attention = None
        if getattr(self.config, "enable_feature_attention", False):
            try:
                from src.scanner.automation.feature_attention import FeatureAttentionLayer
                self._feature_attention = FeatureAttentionLayer()
                logger.info("Feature attention layer initialized")
            except Exception as e:
                logger.debug(f"Feature attention init deferred: {e}")

        # Regime-aware RL reward shaping (Phase 14: regime-conditioned reward function)
        self._regime_reward = None
        if getattr(self.config, "enable_regime_reward", False):
            try:
                from src.scanner.automation.regime_reward import RegimeAwareReward
                self._regime_reward = RegimeAwareReward()
                logger.info("Regime-aware RL reward shaping initialized")
            except Exception as e:
                logger.debug(f"Regime reward init deferred: {e}")

        # Entropy position sizer (Phase 16: Shannon entropy of agent votes → size multiplier)
        self._entropy_sizer = None
        if getattr(self.config, "enable_entropy_sizing", False):
            try:
                from src.scanner.automation.entropy_sizer import EntropyPositionSizer
                self._entropy_sizer = EntropyPositionSizer()
                logger.info("Entropy position sizer initialized")
            except Exception as e:
                logger.debug(f"Entropy sizer init deferred: {e}")

        # Execution quality tracker (Phase 16: TCA feedback loop)
        self._exec_quality_tracker = None
        if getattr(self.config, "enable_execution_quality_tracking", False):
            try:
                from src.scanner.automation.execution_quality_tracker import ExecutionQualityTracker
                self._exec_quality_tracker = ExecutionQualityTracker()
                logger.info("Execution quality tracker initialized")
            except Exception as e:
                logger.debug(f"Execution quality tracker init deferred: {e}")

        # Phase 44 (US-278): Market Regime Detector (BOCPD + Hurst + ADX ensemble)
        self._market_regime_detector = None
        try:
            from src.scanner.regime_detector import MarketRegimeDetector
            self._market_regime_detector = MarketRegimeDetector()
            logger.info("Phase 44: Market regime detector initialized (BOCPD + Hurst + ADX)")
        except Exception as e:
            logger.debug(f"Market regime detector init deferred: {e}")

        # Live position manager (Phase 17: prediction-driven position management)
        self._position_manager = None
        if getattr(self.config, "enable_live_position_management", False):
            try:
                from src.scanner.automation.position_manager import LivePositionManager
                self._position_manager = LivePositionManager()
                logger.info("Live position manager initialized")
            except Exception as e:
                logger.debug(f"Position manager init deferred: {e}")

        # Affinity portfolio sizer (Phase 17: pair affinity → sizing)
        self._affinity_portfolio = None
        if getattr(self.config, "enable_affinity_portfolio", False):
            try:
                from src.scanner.automation.affinity_portfolio import AffinityPortfolioSizer
                self._affinity_portfolio = AffinityPortfolioSizer()
                logger.info("Affinity portfolio sizer initialized")
            except Exception as e:
                logger.debug(f"Affinity portfolio init deferred: {e}")

        # Execution router (Phase 17: smart order slicing)
        self._execution_router = None
        if getattr(self.config, "enable_execution_routing", False):
            try:
                from src.scanner.automation.execution_router import ExecutionRouter
                self._execution_router = ExecutionRouter()
                logger.info("Execution router initialized")
            except Exception as e:
                logger.debug(f"Execution router init deferred: {e}")

        # Model router (Phase 17: bandit-based pair-specific model selection)
        self._model_router = None
        if getattr(self.config, "enable_model_routing", False):
            try:
                from src.scanner.automation.model_router import ModelRouter
                self._model_router = ModelRouter()
                logger.info("Model router initialized")
            except Exception as e:
                logger.debug(f"Model router init deferred: {e}")

        # Training augmenter (Phase 17: synthetic data injection)
        self._training_augmenter = None
        if getattr(self.config, "enable_training_augmentation", False):
            try:
                from src.scanner.automation.training_augmenter import TrainingAugmenter
                self._training_augmenter = TrainingAugmenter()
                logger.info("Training augmenter initialized")
            except Exception as e:
                logger.debug(f"Training augmenter init deferred: {e}")

        # Attention feedback loop (Phase 17: temporal attention consensus feedback)
        self._attention_feedback = None
        if getattr(self.config, "enable_attention_feedback", False):
            try:
                from src.scanner.automation.attention_feedback import AttentionFeedbackLoop
                self._attention_feedback = AttentionFeedbackLoop()
                logger.info("Attention feedback loop initialized")
            except Exception as e:
                logger.debug(f"Attention feedback init deferred: {e}")

        # Execution quality optimizer (Phase 18: cost profiling by model+regime)
        self._execution_quality_optimizer = None
        if getattr(self.config, "enable_execution_quality_optimizer", False):
            try:
                from src.scanner.automation.execution_quality_optimizer import ExecutionQualityOptimizer
                self._execution_quality_optimizer = ExecutionQualityOptimizer()
                logger.info("Execution quality optimizer initialized")
            except Exception as e:
                logger.debug(f"Execution quality optimizer init deferred: {e}")

        # Threshold optimizer (Phase 18: adaptive gate threshold tuning)
        self._threshold_optimizer = None
        if getattr(self.config, "enable_threshold_optimizer", False):
            try:
                from src.scanner.automation.threshold_optimizer import ThresholdOptimizer
                self._threshold_optimizer = ThresholdOptimizer()
                # Phase 22 (US-136): Warm-start from journal if optimizer has 0 outcomes
                if self._threshold_optimizer._total_outcomes == 0:
                    try:
                        replayed = self._threshold_optimizer.replay_from_journal()
                        if replayed > 0:
                            logger.info(f"Threshold optimizer warm-started with {replayed} journal trades")
                    except Exception as _replay_err:
                        logger.debug(f"Journal replay skipped: {_replay_err}")
                logger.info("Threshold optimizer initialized")
            except Exception as e:
                logger.debug(f"Threshold optimizer init deferred: {e}")

        # Phase 21 (US-131): Central config adjustment manager
        self._config_adjuster = None
        if getattr(self.config, "enable_config_adjuster", True):
            try:
                from src.scanner.automation.config_adjuster import ConfigAdjuster
                self._config_adjuster = ConfigAdjuster()
                logger.info("Config adjuster initialized")
            except Exception as e:
                logger.debug(f"Config adjuster init deferred: {e}")

        # Dynamic risk allocator (Phase 18: P/L distribution risk sizing)
        self._dynamic_risk_allocator = None
        if getattr(self.config, "enable_dynamic_risk_allocation", False):
            try:
                from src.scanner.automation.dynamic_risk_allocator import DynamicRiskAllocator
                self._dynamic_risk_allocator = DynamicRiskAllocator()
                logger.info("Dynamic risk allocator initialized")
            except Exception as e:
                logger.debug(f"Dynamic risk allocator init deferred: {e}")

        # Model calibration tracker (Phase 18: per-model confidence calibration)
        self._model_calibration = None
        if getattr(self.config, "enable_model_calibration", False):
            try:
                from src.scanner.automation.model_calibration import ModelCalibrationTracker
                self._model_calibration = ModelCalibrationTracker()
                logger.info("Model calibration tracker initialized")
            except Exception as e:
                logger.debug(f"Model calibration init deferred: {e}")

        # Pair-regime-agent matrix (Phase 18: 3-way interaction tracking)
        self._pair_regime_agent_matrix = None
        if getattr(self.config, "enable_pair_regime_agent_matrix", False):
            try:
                from src.scanner.automation.pair_regime_agent_matrix import PairRegimeAgentMatrix
                self._pair_regime_agent_matrix = PairRegimeAgentMatrix()
                logger.info("Pair-regime-agent matrix initialized")
            except Exception as e:
                logger.debug(f"Pair-regime-agent matrix init deferred: {e}")

        # Signal timing optimizer (Phase 18: scan frequency optimization)
        self._signal_timing = None
        if getattr(self.config, "enable_signal_timing", False):
            try:
                from src.scanner.automation.signal_timing import SignalTimingOptimizer
                self._signal_timing = SignalTimingOptimizer()
                logger.info("Signal timing optimizer initialized")
            except Exception as e:
                logger.debug(f"Signal timing init deferred: {e}")

        # Memory manager (Phase 19: LRU cache and log rotation)
        self._memory_manager = None
        if getattr(self.config, "enable_memory_manager", False):
            try:
                from src.scanner.automation.memory_manager import MemoryManager
                self._memory_manager = MemoryManager()
                logger.info("Memory manager initialized")
            except Exception as e:
                logger.debug(f"Memory manager init deferred: {e}")

        # Module activation map (Phase 19: module dispatch scheduling)
        self._module_activation = None
        if getattr(self.config, "enable_module_activation", False):
            try:
                from src.scanner.automation.module_activation import ModuleActivationMap
                self._module_activation = ModuleActivationMap()
                logger.info("Module activation map initialized")
            except Exception as e:
                logger.debug(f"Module activation init deferred: {e}")

        # Agent lifecycle manager (Phase 19: agent fitness and quarantine)
        self._agent_lifecycle = None
        if getattr(self.config, "enable_agent_lifecycle", False):
            try:
                from src.scanner.automation.agent_lifecycle import AgentLifecycleManager
                self._agent_lifecycle = AgentLifecycleManager()
                logger.info("Agent lifecycle manager initialized")
            except Exception as e:
                logger.debug(f"Agent lifecycle init deferred: {e}")

        # Health registry (Phase 19: orchestrator health monitoring)
        self._health_registry = None
        if getattr(self.config, "enable_health_registry", False):
            try:
                from src.scanner.automation.health_registry import HealthRegistry
                self._health_registry = HealthRegistry()
                logger.info("Health registry initialized")
            except Exception as e:
                logger.debug(f"Health registry init deferred: {e}")

        # Observation consumer (Phase 19: pattern detection from observations)
        self._observation_consumer = None
        if getattr(self.config, "enable_observation_consumer", False):
            try:
                from src.scanner.automation.observation_consumer import ObservationConsumer
                self._observation_consumer = ObservationConsumer()
                logger.info("Observation consumer initialized")
            except Exception as e:
                logger.debug(f"Observation consumer init deferred: {e}")

        # Replay validator (Phase 19: historical replay and drift detection)
        self._replay_validator = None
        if getattr(self.config, "enable_replay_validator", False):
            try:
                from src.scanner.automation.replay_validator import ReplayValidator
                self._replay_validator = ReplayValidator()
                logger.info("Replay validator initialized")
            except Exception as e:
                logger.debug(f"Replay validator init deferred: {e}")

        # Regime broadcaster (Phase 16: event-driven regime transition propagation)
        self._regime_broadcaster = None
        if getattr(self.config, "enable_regime_broadcast", False):
            try:
                from src.scanner.automation.regime_broadcaster import RegimeBroadcaster
                self._regime_broadcaster = RegimeBroadcaster()
                logger.info("Regime broadcaster initialized")
            except Exception as e:
                logger.debug(f"Regime broadcaster init deferred: {e}")

        # Agent accuracy matrix (Phase 16: per-agent accuracy by regime/pair)
        self._agent_accuracy_matrix = None
        if getattr(self.config, "enable_agent_accuracy_matrix", False):
            try:
                from src.scanner.automation.agent_accuracy_matrix import AgentAccuracyMatrix
                self._agent_accuracy_matrix = AgentAccuracyMatrix()
                logger.info("Agent accuracy matrix initialized")
            except Exception as e:
                logger.debug(f"Agent accuracy matrix init deferred: {e}")

        # Temporal attention (Phase 16: learned multi-timeframe signal weighting)
        self._temporal_attention = None
        if getattr(self.config, "enable_temporal_attention", False):
            try:
                from src.scanner.automation.temporal_attention import TemporalAttentionLayer
                self._temporal_attention = TemporalAttentionLayer()
                logger.info("Temporal attention layer initialized")
            except Exception as e:
                logger.debug(f"Temporal attention init deferred: {e}")

        # Pair transfer learning (Phase 16: cross-pair weight sharing)
        self._pair_transfer = None
        if getattr(self.config, "enable_pair_transfer", False):
            try:
                from src.scanner.automation.pair_transfer import PairTransferLearning
                self._pair_transfer = PairTransferLearning()
                logger.info("Pair transfer learning initialized")
            except Exception as e:
                logger.debug(f"Pair transfer init deferred: {e}")

        # Causal signal filter (Phase 15: Granger causality-based signal consistency)
        self._causal_filter = None
        if getattr(self.config, "enable_causal_filtering", False):
            try:
                from src.scanner.automation.causal_filter import CausalSignalFilter
                self._causal_filter = CausalSignalFilter()
                logger.info("Causal signal filter initialized")
            except Exception as e:
                logger.debug(f"Causal filter init deferred: {e}")

        # Trade explainer (Phase 15: SHAP-based explainability and consistency filtering)
        # Phase 57 (US-352): Confidence Penalty Ceiling — hard floor on stacked penalties
        self._penalty_ceiling = None
        try:
            from src.scanner.confidence_penalty_ceiling import ConfidencePenaltyCeiling
            self._penalty_ceiling = ConfidencePenaltyCeiling()
            logger.info("Phase 57 (US-352): Confidence penalty ceiling initialized")
        except Exception as _cpc_err:
            logger.debug(f"Phase 57: Confidence penalty ceiling init deferred: {_cpc_err}")

        # Phase 57 (US-353): Calibration Guard — suppress overconfidence penalty when data insufficient
        self._calibration_guard = None
        try:
            from src.scanner.calibration_guard import CalibrationGuard
            self._calibration_guard = CalibrationGuard()
            logger.info("Phase 57 (US-353): Calibration guard initialized")
        except Exception as _cg_err:
            logger.debug(f"Phase 57: Calibration guard init deferred: {_cg_err}")

        # Phase 57 (US-354): Stall Recovery Manager — suspend penalties during stall recovery
        self._stall_recovery = None
        try:
            from src.scanner.stall_recovery import StallRecoveryManager
            self._stall_recovery = StallRecoveryManager()
            logger.info("Phase 57 (US-354): Stall recovery manager initialized")
        except Exception as _sr_err:
            logger.debug(f"Phase 57: Stall recovery manager init deferred: {_sr_err}")

        # Phase 57 (US-355): Drift Proxy Guard — prevent circular drift detection
        self._drift_proxy_guard = None
        try:
            from src.scanner.drift_proxy_guard import DriftProxyGuard
            self._drift_proxy_guard = DriftProxyGuard()
            logger.info("Phase 57 (US-355): Drift proxy guard initialized")
        except Exception as _dpg_err:
            logger.debug(f"Phase 57: Drift proxy guard init deferred: {_dpg_err}")

        # Phase 58 (US-358): Raw Confidence Recorder — pre-penalty signal distribution
        self._raw_conf_recorder = None
        try:
            from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
            self._raw_conf_recorder = RawConfidenceRecorder()
            logger.info("Phase 58 (US-358): Raw confidence recorder initialized")
        except Exception as _rcr_err:
            logger.debug(f"Phase 58: Raw confidence recorder init deferred: {_rcr_err}")

        # Phase 59 (US-364): Gate Proximity Reporter — near-miss confidence rejects
        self._gate_proximity = None
        try:
            from src.scanner.gate_proximity_reporter import GateProximityReporter
            self._gate_proximity = GateProximityReporter()
            logger.info("Phase 59 (US-364): Gate proximity reporter initialized")
        except Exception as _gpr_err:
            logger.debug(f"Phase 59: Gate proximity reporter init deferred: {_gpr_err}")

        # Phase 59 (US-365): Ensemble Divergence Monitor — TCN vs Ridge disagreement
        self._ensemble_divergence = None
        try:
            from src.scanner.ensemble_divergence_monitor import EnsembleDivergenceMonitor
            self._ensemble_divergence = EnsembleDivergenceMonitor()
            logger.info("Phase 59 (US-365): Ensemble divergence monitor initialized")
        except Exception as _edm_err:
            logger.debug(f"Phase 59: Ensemble divergence monitor init deferred: {_edm_err}")

        # Phase 61 (US-371): Momentum Score Recorder — raw XGBoost momentum distribution
        self._momentum_recorder = None
        try:
            from src.scanner.momentum_score_recorder import MomentumScoreRecorder
            self._momentum_recorder = MomentumScoreRecorder()
            logger.info("Phase 61 (US-371): Momentum score recorder initialized")
        except Exception as _msr_err:
            logger.debug(f"Phase 61: Momentum score recorder init deferred: {_msr_err}")

        # Phase 62 (US-375): Adaptive Momentum Floor — auto-reduce min_momentum on NEAR_THRESHOLD
        self._adaptive_momentum_floor = None
        try:
            from src.scanner.adaptive_momentum_floor import AdaptiveMomentumFloor
            self._adaptive_momentum_floor = AdaptiveMomentumFloor()
            logger.info("Phase 62 (US-375): Adaptive momentum floor initialized")
        except Exception as _amf_err:
            logger.debug(f"Phase 62: Adaptive momentum floor init deferred: {_amf_err}")

        # Phase 63 (US-378): RiskRejectRecorder — rolling buffer of rf_drawdown_pct per pair
        self._risk_recorder = None
        try:
            from src.scanner.risk_reject_recorder import RiskRejectRecorder
            self._risk_recorder = RiskRejectRecorder()
            logger.info("Phase 63 (US-378): Risk reject recorder initialized")
        except Exception as _rrr_err:
            logger.debug(f"Phase 63: Risk reject recorder init deferred: {_rrr_err}")

        # Phase 64 (US-382): AdaptiveDrawdownCeiling — auto-raise max_drawdown_pct on NEAR_THRESHOLD
        self._adaptive_drawdown_ceiling = None
        try:
            from src.scanner.adaptive_drawdown_ceiling import AdaptiveDrawdownCeiling
            self._adaptive_drawdown_ceiling = AdaptiveDrawdownCeiling()
            logger.info("Phase 64 (US-382): Adaptive drawdown ceiling initialized")
        except Exception as _adc_err:
            logger.debug(f"Phase 64: Adaptive drawdown ceiling init deferred: {_adc_err}")

        # Phase 66 (US-388): VoteScoreRecorder — weighted_vote_score distribution
        self._vote_recorder = None
        try:
            from src.scanner.vote_score_recorder import VoteScoreRecorder
            self._vote_recorder = VoteScoreRecorder()
            logger.info("Phase 66 (US-388): Vote score recorder initialized")
        except Exception as _vsr_err:
            logger.debug(f"Phase 66: Vote score recorder init deferred: {_vsr_err}")

        # Phase 66 (US-390): AdaptiveVoteThreshold — auto-reduce weighted_vote_threshold on NEAR_THRESHOLD
        self._adaptive_vote_threshold = None
        try:
            from src.scanner.adaptive_vote_threshold import AdaptiveVoteThreshold
            self._adaptive_vote_threshold = AdaptiveVoteThreshold()
            logger.info("Phase 66 (US-390): Adaptive vote threshold initialized")
        except Exception as _avt_err:
            logger.debug(f"Phase 66: Adaptive vote threshold init deferred: {_avt_err}")

        # Phase 67 (US-391): SubInferenceRecorder — consensus ratio distribution
        self._sub_inference_recorder = None
        try:
            from src.scanner.sub_inference_recorder import SubInferenceRecorder
            self._sub_inference_recorder = SubInferenceRecorder()
            logger.info("Phase 67 (US-391): Sub-inference recorder initialized")
        except Exception as _sir_err:
            logger.debug(f"Phase 67: Sub-inference recorder init deferred: {_sir_err}")

        # Phase 67 (US-393): AdaptiveSubInferenceThreshold — auto-reduce sub_inference_vote_threshold
        self._adaptive_sub_inference_threshold = None
        try:
            from src.scanner.adaptive_sub_inference_threshold import AdaptiveSubInferenceThreshold
            self._adaptive_sub_inference_threshold = AdaptiveSubInferenceThreshold()
            logger.info("Phase 67 (US-393): Adaptive sub-inference threshold initialized")
        except Exception as _asit_err:
            logger.debug(f"Phase 67: Adaptive sub-inference threshold init deferred: {_asit_err}")

        # Phase 75: DisagreementRecorder — heuristic disagreement distribution
        self._disagreement_recorder = None
        try:
            from src.scanner.disagreement_recorder import DisagreementRecorder
            self._disagreement_recorder = DisagreementRecorder()
            logger.info("Phase 75: Disagreement recorder initialized")
        except Exception as _dr_err:
            logger.debug(f"Phase 75: Disagreement recorder init deferred: {_dr_err}")

        # Phase 75: AdaptiveDisagreementFloor — auto-raise disagreement_hard_floor
        self._adaptive_disagreement_floor = None
        try:
            from src.scanner.adaptive_disagreement_floor import AdaptiveDisagreementFloor
            self._adaptive_disagreement_floor = AdaptiveDisagreementFloor()
            logger.info("Phase 75: Adaptive disagreement floor initialized")
        except Exception as _adf_err:
            logger.debug(f"Phase 75: Adaptive disagreement floor init deferred: {_adf_err}")

        # Phase 56 (US-348): Virtual Trade Logger — capture rejected setups for offline RL
        self._virtual_trade_logger = None
        try:
            from src.scanner.virtual_trade_logger import VirtualTradeLogger
            self._virtual_trade_logger = VirtualTradeLogger()
            logger.info("Phase 56 (US-348): Virtual trade logger initialized")
        except Exception as _vtl_err:
            logger.debug(f"Phase 56: Virtual trade logger init deferred: {_vtl_err}")

        # Phase 56 (US-347): Confidence Penalty Monitor — detect stacked penalties
        self._confidence_penalty_monitor = None
        try:
            from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor
            self._confidence_penalty_monitor = ConfidencePenaltyMonitor()
            logger.info("Phase 56 (US-347): Confidence penalty monitor initialized")
        except Exception as _cpm_err:
            logger.debug(f"Phase 56: Confidence penalty monitor init deferred: {_cpm_err}")

        # Phase 56 (US-346): Gate Rejection Tracker — surface execution bottleneck
        self._gate_rejection_tracker = None
        try:
            from src.scanner.gate_rejection_tracker import GateRejectionTracker
            self._gate_rejection_tracker = GateRejectionTracker()
            logger.info("Phase 56 (US-346): Gate rejection tracker initialized")
        except Exception as _grt_err:
            logger.debug(f"Phase 56: Gate rejection tracker init deferred: {_grt_err}")

        self._trade_explainer = None
        if getattr(self.config, "enable_trade_explainability", False):
            try:
                from src.scanner.automation.trade_explainer import TradeExplainer
                self._trade_explainer = TradeExplainer()
                logger.info("Trade explainer initialized")
            except Exception as e:
                logger.debug(f"Trade explainer init deferred: {e}")

        # Model bandit selector (Phase 15: EXP3 bandit for dynamic model routing)
        self._model_bandit = None
        if getattr(self.config, "enable_model_bandit", False):
            try:
                from src.scanner.automation.model_bandit import ModelBanditSelector
                self._model_bandit = ModelBanditSelector()
                logger.info("Model bandit selector initialized")
            except Exception as e:
                logger.debug(f"Model bandit init deferred: {e}")

        # Market impact model (Phase 15: Almgren-Chriss slippage-aware position sizing)
        self._market_impact = None
        if getattr(self.config, "enable_market_impact", False):
            try:
                from src.scanner.automation.market_impact import MarketImpactModel
                self._market_impact = MarketImpactModel()
                logger.info("Market impact model initialized")
            except Exception as e:
                logger.debug(f"Market impact init deferred: {e}")

        # Kelly portfolio optimizer (Phase 13: portfolio-level risk budgeting)
        self._kelly_optimizer = None
        if getattr(self.config, "enable_kelly_sizing", False):
            try:
                from src.scanner.automation.kelly_portfolio import KellyPortfolioOptimizer
                self._kelly_optimizer = KellyPortfolioOptimizer()
                logger.info("Kelly portfolio optimizer initialized")
            except Exception as e:
                logger.debug(f"Kelly portfolio optimizer init deferred: {e}")

        # Concept drift detector (Phase 13: rolling z-score drift monitoring)
        self._concept_drift = None
        if getattr(self.config, "enable_concept_drift_detection", False):
            try:
                from src.scanner.automation.concept_drift import ConceptDriftDetector
                self._concept_drift = ConceptDriftDetector()
                logger.info("Concept drift detector initialized")
            except Exception as e:
                logger.debug(f"Concept drift detector init deferred: {e}")

        # Multi-horizon signal fusion (Phase 13: Bayesian TF agreement grading)
        self._multi_horizon_fusion = None
        if getattr(self.config, "enable_multi_horizon_fusion", False):
            try:
                from src.scanner.automation.multi_horizon_fusion import MultiHorizonFusion
                self._multi_horizon_fusion = MultiHorizonFusion()
                logger.info("Multi-horizon fusion initialized")
            except Exception as e:
                logger.debug(f"Multi-horizon fusion init deferred: {e}")

        # Confidence calibrator (Phase 13: Platt scaling for calibrated probabilities)
        self._confidence_calibrator = None
        if getattr(self.config, "enable_confidence_calibration", False):
            try:
                from src.scanner.automation.confidence_calibrator import ConfidenceCalibrator
                self._confidence_calibrator = ConfidenceCalibrator()
                self._confidence_calibrator.fit_from_journal()
                logger.info("Confidence calibrator initialized")
            except Exception as e:
                logger.debug(f"Confidence calibrator init deferred: {e}")

        # Microstructure regime detector (Phase 12: spread-driven early regime signals)
        self._microstructure_detector = None
        if getattr(self.config, "enable_microstructure_regime", False):
            try:
                from src.scanner.automation.microstructure_regime import MicrostructureRegimeDetector
                self._microstructure_detector = MicrostructureRegimeDetector()
                logger.info("Microstructure regime detector initialized")
            except Exception as e:
                logger.debug(f"Microstructure regime detector init deferred: {e}")

        # Phase 20 (US-125): Drift monitor with auto-correction
        self._drift_monitor = None
        if getattr(self.config, "enable_replay_validator", False):
            try:
                from src.scanner.automation.drift_monitor import DriftMonitor
                self._drift_monitor = DriftMonitor(
                    replay_validator=self._replay_validator,
                    config_tuner=None,  # Injected later if available
                    threshold_optimizer=self._threshold_optimizer,
                )
                logger.info("Drift monitor initialized")
            except Exception as e:
                logger.debug(f"Drift monitor init deferred: {e}")

        # Adaptive LR scheduler (US-091: cosine-annealing + warmup for RL learning rate)
        self._adaptive_lr = None
        if getattr(self.config, "enable_adaptive_lr", False):
            try:
                from src.scanner.automation.adaptive_lr import AdaptiveLRScheduler
                self._adaptive_lr = AdaptiveLRScheduler()
                logger.info("Adaptive LR scheduler initialized")
            except Exception as e:
                logger.debug(f"Adaptive LR scheduler init deferred: {e}")

        # Adversarial trainer (US-089: adversarial robustness training for ensemble models)
        self._adversarial_trainer = None
        if getattr(self.config, "enable_adversarial_training", False):
            try:
                from src.scanner.automation.adversarial_trainer import AdversarialTrainer
                self._adversarial_trainer = AdversarialTrainer()
                logger.info("Adversarial trainer initialized")
            except Exception as e:
                logger.debug(f"Adversarial trainer init deferred: {e}")

        # crisis_generator: initialized at line ~718, dispatched via module_dispatcher
        # (save_log every 50 cycles) + explicit generate_scenario_batch call in continuous.py

        # Dynamic hedging (US-077: auto-open inverse positions during macro stress)
        self._dynamic_hedging = None
        if getattr(self.config, "enable_dynamic_hedging", False):
            try:
                from src.scanner.automation.dynamic_hedging import DynamicHedgeManager
                self._dynamic_hedging = DynamicHedgeManager()
                logger.info("Dynamic hedge manager initialized")
            except Exception as e:
                logger.debug(f"Dynamic hedge manager init deferred: {e}")

        # Trade outcome predictor (US-082: predict open trade win/loss from post-entry features)
        self._trade_outcome_predictor = None
        if getattr(self.config, "enable_trade_outcome_prediction", False):
            try:
                from src.scanner.automation.trade_outcome_predictor import TradeOutcomePredictor
                self._trade_outcome_predictor = TradeOutcomePredictor()
                logger.info("Trade outcome predictor initialized")
            except Exception as e:
                logger.debug(f"Trade outcome predictor init deferred: {e}")

        # Phase 50: Retrain trigger (accuracy drift -> retrain request)
        self._retrain_trigger = None
        try:
            from src.scanner.automation.retrain_trigger import RetrainTrigger
            self._retrain_trigger = RetrainTrigger()
            logger.info("Retrain trigger initialized")
        except Exception as e:
            logger.debug(f"Retrain trigger init deferred: {e}")

        # Phase 50: Pair model selector (per-pair best model selection)
        self._pair_model_selector = None
        try:
            from src.scanner.automation.pair_model_selector import PairModelSelector
            self._pair_model_selector = PairModelSelector()
            logger.info("Pair model selector initialized")
        except Exception as e:
            logger.debug(f"Pair model selector init deferred: {e}")

        # Phase 50: Portfolio optimizer (pair ranking by rolling Sharpe)
        self._portfolio_optimizer = None
        try:
            from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer
            self._portfolio_optimizer = PortfolioOptimizer()
            logger.info("Portfolio optimizer initialized")
        except Exception as e:
            logger.debug(f"Portfolio optimizer init deferred: {e}")

        # Phase 50: Session snapshot manager (cross-session state blending)
        self._session_snapshot = None
        try:
            from src.scanner.automation.session_snapshot import SessionSnapshotManager
            self._session_snapshot = SessionSnapshotManager()
            logger.info("Session snapshot manager initialized")
        except Exception as e:
            logger.debug(f"Session snapshot init deferred: {e}")

        # Phase 50: Alert manager (drawdown, loss streak, weight instability alerts)
        self._alert_manager = None
        try:
            from src.scanner.automation.alert_manager import AlertManager
            self._alert_manager = AlertManager()
            logger.info("Alert manager initialized")
        except Exception as e:
            logger.debug(f"Alert manager init deferred: {e}")

        # Phase 50: Dynamic drawdown manager (equity-curve-adaptive trailing stops)
        self._dynamic_drawdown = None
        try:
            from src.scanner.automation.dynamic_drawdown import DynamicDrawdownManager
            self._dynamic_drawdown = DynamicDrawdownManager()
            logger.info("Dynamic drawdown manager initialized")
        except Exception as e:
            logger.debug(f"Dynamic drawdown init deferred: {e}")

        # Phase 50: Smart execution engine (VWAP/TWAP order slicing)
        self._smart_execution = None
        if getattr(self.config, "enable_smart_execution", False):
            try:
                from src.scanner.automation.smart_execution import SmartExecutionEngine
                self._smart_execution = SmartExecutionEngine()
                logger.info("Smart execution engine initialized")
            except Exception as e:
                logger.debug(f"Smart execution init deferred: {e}")

        # Phase 50: Accuracy gate (per-pair accuracy tracking and auto-blocking)
        self._accuracy_gate = None
        try:
            from src.scanner.automation.accuracy_gate import AccuracyGate
            self._accuracy_gate = AccuracyGate()
            logger.info("Accuracy gate initialized")
        except Exception as e:
            logger.debug(f"Accuracy gate init deferred: {e}")

        # Phase 50: Pair affinity tracker (co-occurrence outcome tracking)
        self._pair_affinity = None
        try:
            from src.scanner.automation.pair_affinity import PairAffinityTracker
            self._pair_affinity = PairAffinityTracker()
            logger.info("Pair affinity tracker initialized")
        except Exception as e:
            logger.debug(f"Pair affinity init deferred: {e}")

        # Phase 50: HRP optimizer (hierarchical risk parity portfolio weights)
        self._hrp_optimizer = None
        try:
            from src.scanner.automation.hierarchical_risk_parity import HRPOptimizer
            self._hrp_optimizer = HRPOptimizer()
            logger.info("HRP optimizer initialized")
        except Exception as e:
            logger.debug(f"HRP optimizer init deferred: {e}")

        # Phase 50: Observational learner (synthetic trade pattern extraction)
        self._observational_learner = None
        if getattr(self.config, "enable_observational_learning", False):
            try:
                from src.scanner.automation.observational_learning import ObservationalLearner
                self._observational_learner = ObservationalLearner()
                logger.info("Observational learner initialized")
            except Exception as e:
                logger.debug(f"Observational learner init deferred: {e}")

        # Phase 50: Model manager (version tracking, shadow testing, safe promotion)
        self._model_manager = None
        try:
            from src.scanner.automation.model_manager import ModelManager
            self._model_manager = ModelManager()
            logger.info("Model manager initialized")
        except Exception as e:
            logger.debug(f"Model manager init deferred: {e}")

        # Phase 50: Crisis event generator (synthetic crisis scenarios for RL)
        self._crisis_generator = None
        if getattr(self.config, "enable_synthetic_crisis", False):
            try:
                from src.scanner.automation.crisis_generator import CrisisEventGenerator
                self._crisis_generator = CrisisEventGenerator()
                logger.info("Crisis event generator initialized")
            except Exception as e:
                logger.debug(f"Crisis generator init deferred: {e}")

        # Phase 50: API retry circuit breaker (failure rate tracking and cooldown)
        self._api_retry = None
        if getattr(self.config, "enable_circuit_breakers", True):
            try:
                from src.scanner.automation.api_retry import CircuitBreaker
                self._api_retry = CircuitBreaker(name="oanda_api")
                logger.info("API retry circuit breaker initialized")
            except Exception as e:
                logger.debug(f"API retry circuit breaker init deferred: {e}")

        # US-006–US-019: Previously unregistered modules — init for dispatcher
        self._counterfactual_learner = None
        try:
            from src.scanner.automation.counterfactual_learner import CounterfactualLearner
            self._counterfactual_learner = CounterfactualLearner()
            logger.info("Counterfactual learner initialized")
        except Exception as e:
            logger.debug(f"Counterfactual learner init deferred: {e}")

        self._chain_memory = None
        self._chain_memory_synced: bool = False  # Phase 82 (US-P82-003): sync once per session
        try:
            from src.scanner.automation.chain_memory import ChainMemory
            self._chain_memory = ChainMemory()
            logger.info("Chain memory initialized")
        except Exception as e:
            logger.debug(f"Chain memory init deferred: {e}")

        self._drift_projector = None
        try:
            from src.scanner.automation.drift_projector import DriftProjector
            self._drift_projector = DriftProjector()
            logger.info("Drift projector initialized")
        except Exception as e:
            logger.debug(f"Drift projector init deferred: {e}")

        # Phase 81 (US-P81-003): RegimeConfidenceScorer in Scanner for _scan_pair() risk_pct scaling
        # (ExecutionManager has its own instance; Scanner needs its own for pre-execution scaling)
        self._regime_confidence_scorer_scan = None
        try:
            from src.scanner.regime_confidence import RegimeConfidenceScorer as _RCS
            self._regime_confidence_scorer_scan = _RCS()
            logger.info("Phase 81 (US-P81-003): RegimeConfidenceScorer initialized in Scanner")
        except Exception as e:
            logger.debug(f"Phase 81: RegimeConfidenceScorer (scanner) init deferred: {e}")

        self._episodic_memory = None
        try:
            from src.scanner.automation.episodic_memory import EpisodicMemory as _EpisodicMemoryScanner
            self._episodic_memory = _EpisodicMemoryScanner()
            logger.info("Episodic memory (scanner-level) initialized")
        except Exception as e:
            logger.debug(f"Episodic memory init deferred: {e}")

        self._self_model_state = None
        try:
            from src.scanner.automation.self_model_state import SelfModelState
            self._self_model_state = SelfModelState()
            logger.info("Self-model state initialized")
        except Exception as e:
            logger.debug(f"Self-model state init deferred: {e}")

        self._causal_counterfactual = None
        try:
            from src.scanner.automation.causal_counterfactual import CounterfactualEngine
            self._causal_counterfactual = CounterfactualEngine()
            logger.info("Causal counterfactual engine initialized")
        except Exception as e:
            logger.debug(f"Causal counterfactual init deferred: {e}")

        # Phase 49 (US-307): Pair performance tracker for auto-blacklisting in scan loop
        self._pair_tracker = None
        try:
            from src.scanner.pair_blacklist import create_default_pair_tracker
            self._pair_tracker = create_default_pair_tracker()
            logger.info("Phase 49: Pair performance tracker initialized in Scanner")
        except Exception as e:
            logger.debug(f"Phase 49: Pair tracker init deferred: {e}")

        # Phase 20 (US-122): Inject agent lifecycle into agent team for weight modifiers
        if self._agent_lifecycle is not None:
            self._agent_team._agent_lifecycle = self._agent_lifecycle

        # Phase 25 (US-155): Inject accuracy matrix into agent team for per-pair weighting
        if self._agent_accuracy_matrix is not None:
            self._agent_team._accuracy_matrix = self._agent_accuracy_matrix

        # Phase 20 (US-121): Register all modules with health registry
        self._register_health_checks()

        # Load config
        self._load_yaml_config()

        # Phase 74: Restore mutable scan state from last session
        self.load_scan_state()

    def _get_feedback_observer(self):
        """Shared observation logger for module feedback hooks (lazy-init).

        Mirrors ExecutionManager._get_observer() pattern. All 13 dead-end
        module outputs route through this to feed the observation consumer.
        """
        if self._feedback_observer is None:
            try:
                from .automation.observation_log import ObservationLog
                self._feedback_observer = ObservationLog()
            except ImportError:
                pass
        return self._feedback_observer

    def _load_yaml_config(self) -> None:
        """Load YAML configuration and update settings."""
        try:
            yaml_config = load_yaml_config(self.config.config_path)

            # Update config from YAML
            if yaml_config:
                buddy_config = yaml_config.get("buddy", {})
                self.config.risk_per_trade_pct = buddy_config.get(
                    "risk_per_trade_pct", self.config.risk_per_trade_pct
                )
                self.config.sl_pips = buddy_config.get("stop_loss_pips", self.config.sl_pips)
                self.config.tp_pips = buddy_config.get("take_profit_pips", self.config.tp_pips)

                # Gate thresholds from inference config
                inference_config = yaml_config.get("inference", {})
                self.config.min_tcn_probability = inference_config.get(
                    "min_tcn_probability", self.config.min_tcn_probability
                )
                self.config.min_confidence = inference_config.get(
                    "min_confidence", self.config.min_confidence
                )
                self.config.min_momentum = inference_config.get(
                    "min_momentum", self.config.min_momentum
                )
                self.config.max_drawdown_pct = inference_config.get(
                    "max_drawdown_pct", self.config.max_drawdown_pct
                )
                self.config.final_score_threshold = inference_config.get(
                    "final_score_threshold", self.config.final_score_threshold
                )
                self.config.max_uncertainty_std = inference_config.get(
                    "max_uncertainty_std", self.config.max_uncertainty_std
                )
                self.config.use_tcn_volatility_filter = bool(
                    inference_config.get(
                        "use_tcn_volatility_filter",
                        self.config.use_tcn_volatility_filter,
                    )
                )
                self.config.require_tcn_volatility = bool(
                    inference_config.get("tcn_required", self.config.require_tcn_volatility)
                )
                min_regime = inference_config.get(
                    "min_volatility_regime", self.config.min_volatility_regime
                )
                try:
                    self.config.min_volatility_regime = max(0, min(3, int(min_regime)))
                except (TypeError, ValueError):
                    logger.warning(
                        f"Invalid min_volatility_regime={min_regime!r}; keeping "
                        f"{self.config.min_volatility_regime}"
                    )
                scan_config = yaml_config.get("scan", {})
                yaml_profile = scan_config.get("profile")
                # CLI-provided profile (non-balanced) should take precedence over YAML default.
                if yaml_profile and self.config.profile == "balanced":
                    self.config.profile = str(yaml_profile).strip().lower()
                self.config.enable_sub_inference_agents = bool(
                    scan_config.get(
                        "enable_sub_inference_agents",
                        self.config.enable_sub_inference_agents,
                    )
                )
                self.config.sub_inference_tradeable_only = bool(
                    scan_config.get(
                        "sub_inference_tradeable_only",
                        self.config.sub_inference_tradeable_only,
                    )
                )
                self.config.sub_inference_min_confidence = float(
                    scan_config.get(
                        "sub_inference_min_confidence",
                        self.config.sub_inference_min_confidence,
                    )
                )
                self.config.sub_inference_vote_threshold = float(
                    scan_config.get(
                        "sub_inference_vote_threshold",
                        self.config.sub_inference_vote_threshold,
                    )
                )
                self.config.sub_inference_window_checks = max(
                    1,
                    int(
                        scan_config.get(
                            "sub_inference_window_checks",
                            self.config.sub_inference_window_checks,
                        )
                    ),
                )
                self.config.sub_inference_workers = max(
                    1,
                    int(
                        scan_config.get(
                            "sub_inference_workers",
                            self.config.sub_inference_workers,
                        )
                    ),
                )
                self.config.sub_inference_max_candidates = max(
                    1,
                    int(
                        scan_config.get(
                            "sub_inference_max_candidates",
                            self.config.sub_inference_max_candidates,
                        )
                    ),
                )
                self.config.enable_agent_trade_promotion = bool(
                    scan_config.get(
                        "enable_agent_trade_promotion",
                        self.config.enable_agent_trade_promotion,
                    )
                )
                self.config.agent_promotion_min_confidence = float(
                    scan_config.get(
                        "agent_promotion_min_confidence",
                        self.config.agent_promotion_min_confidence,
                    )
                )
                self.config.agent_promotion_requires_risk = bool(
                    scan_config.get(
                        "agent_promotion_requires_risk",
                        self.config.agent_promotion_requires_risk,
                    )
                )
                self.config.use_master_pair_models = bool(
                    scan_config.get(
                        "use_master_pair_models",
                        self.config.use_master_pair_models,
                    )
                )
                self.config.position_sizing_enabled = bool(
                    scan_config.get(
                        "position_sizing_enabled",
                        self.config.position_sizing_enabled,
                    )
                )
                self.config.aggressive_mode = bool(
                    scan_config.get(
                        "aggressive_mode",
                        self.config.aggressive_mode,
                    )
                )
                self.config.regime_scaling_enabled = bool(
                    scan_config.get(
                        "regime_scaling_enabled",
                        self.config.regime_scaling_enabled,
                    )
                )
                self.config.aggressive_scale_high_vol = float(
                    scan_config.get(
                        "aggressive_scale_high_vol",
                        self.config.aggressive_scale_high_vol,
                    )
                )
                self.config.aggressive_scale_extreme_vol = float(
                    scan_config.get(
                        "aggressive_scale_extreme_vol",
                        self.config.aggressive_scale_extreme_vol,
                    )
                )
                self.config.aggressive_min_meta_confidence = float(
                    scan_config.get(
                        "aggressive_min_meta_confidence",
                        self.config.aggressive_min_meta_confidence,
                    )
                )
                # Preserve CLI overrides (e.g. --force disables session filter)
                # before apply_profile re-applies profile defaults.
                _prev_session_filter = self.config.enable_session_filter
                _prev_blocked_pairs = list(self.config.blocked_pairs)
                self.config.apply_profile(self.config.profile)
                self.config.enable_session_filter = _prev_session_filter
                # Restore blocked_pairs if user/CLI explicitly set them
                if _prev_blocked_pairs:
                    self.config.blocked_pairs = _prev_blocked_pairs

                logger.debug(f"Loaded config from {self.config.config_path}")

        except Exception as e:
            logger.warning(f"Failed to load YAML config: {e}")

    def _init_oanda_client(self) -> bool:
        """Initialize OANDA client.

        Returns:
            True if client initialized successfully
        """
        if self._oanda is not None:
            return True

        try:
            from src.utils.oanda_practice import OandaPracticeClient
            self._oanda = OandaPracticeClient.from_env()
            return True
        except ImportError:
            logger.error("OANDA client not available - install oanda-api-v20")
            return False
        except OSError as e:
            logger.warning(f"OANDA credentials not configured: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OANDA client: {e}")
            return False

    def _init_broker_client(self) -> bool:
        """Initialize BrokerClient for data fetching.

        Uses config.broker_type to select the correct broker via factory.
        Falls back to OandaBroker.from_env() for backward compatibility.

        Returns:
            True if broker initialized successfully
        """
        if self._broker is not None:
            return True

        broker_type = getattr(self.config, "broker_type", "oanda")

        if broker_type == "ibkr":
            try:
                from src.brokers.ibkr import IBKRBroker
                ibkr_host = getattr(self.config, "ibkr_host", "127.0.0.1")
                ibkr_port = getattr(self.config, "ibkr_port", 7497)
                ibkr_client_id = getattr(self.config, "ibkr_client_id", 1)
                self._broker = IBKRBroker(
                    host=ibkr_host,
                    port=ibkr_port,
                    client_id=ibkr_client_id,
                )
                self._broker.connect()
                logger.info(
                    "Scanner: IBKRBroker initialized (%s:%d, client_id=%d)",
                    ibkr_host, ibkr_port, ibkr_client_id,
                )
                return True
            except ImportError:
                logger.error("IBKRBroker not available - check src/brokers/ibkr imports")
                return False
            except Exception as e:
                logger.error("Failed to initialize IBKR broker: %s", e)
                return False

        # Default: OANDA
        try:
            from src.brokers.oanda import OandaBroker
            self._broker = OandaBroker.from_env()
            return True
        except ImportError:
            logger.error("BrokerClient not available - check src/brokers/ imports")
            return False
        except OSError as e:
            logger.warning(f"Broker credentials not configured: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize broker client: {e}")
            return False

    def _init_gate_evaluator(self) -> bool:
        """Initialize gate evaluator with JOINT models (preferred) or per-pair fallback.

        Scanner prefers joint-trained models from:
        trained_data/models/joint/

        Falls back to per-pair models if joint models not available.

        Returns:
            True if gates loaded successfully
        """
        if self._gate_evaluator is not None:
            return self._gate_evaluator.is_loaded

        model_dir = Path(self.config.model_dir)
        joint_dir = model_dir / "joint"

        # Check if joint models exist
        use_joint_only = getattr(self.config, 'use_joint_models_only', True)
        require_tcn = getattr(self.config, 'require_tcn_volatility', True)

        # If joint directory doesn't exist, fall back to per-pair models
        if use_joint_only and not joint_dir.exists():
            logger.warning(
                f"Joint model directory not found: {joint_dir}\n"
                f"Falling back to per-pair models. For best results, run:\n"
                f"  python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY"
            )
            use_joint_only = False
            require_tcn = False  # Can't require TCN if using fallback

        # Check if TCN model exists in joint dir - if not, don't require it
        tcn_path = joint_dir / "tcn_volatility_regime.keras"
        if use_joint_only and require_tcn and not tcn_path.exists():
            logger.warning(
                "TCN Volatility Regime model not found in joint directory.\n"
                "Scanner will run without volatility filtering.\n"
                "To train TCN model, run: python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY"
            )
            require_tcn = False

        self._gate_evaluator = GateEvaluator(
            model_dir,
            use_joint_only=use_joint_only,
            min_volatility_regime=self.config.min_volatility_regime,
        )

        try:
            status = self._gate_evaluator.load_models(require_tcn=require_tcn)
            loaded = [k for k, v in status.items() if v]
            if not loaded:
                logger.warning(
                    "No gate models could be loaded (missing packages?).\n"
                    "Scanner will use technical indicator fallback only."
                )
            return any(status.values())
        except FileNotFoundError as e:
            logger.warning(f"Model loading issue: {e}")
            # Still return False but don't raise - allow scan to continue with reduced functionality
            return False
        except Exception as e:
            logger.warning(f"Gate evaluator init failed: {e}")
            return False

    def reload_models(self, reason: str = "manual") -> bool:
        """Hot-reload all cached models from disk.

        Called by the MODEL_PROMOTED event subscriber so the running scanner
        picks up freshly-trained models without a restart. Also callable
        manually from TUI or CLI.

        Returns True if reload succeeded, False otherwise.
        """
        logger.warning("Scanner.reload_models triggered (reason=%s)", reason)
        try:
            # 1. Invalidate gate evaluator — forces full re-init
            self._gate_evaluator = None

            # 2. Reload modular ensemble (TF/Keras models)
            if self._modular_ensemble is not None:
                try:
                    self._modular_ensemble.load_models()
                    self._ensemble_loaded = getattr(self._modular_ensemble, '_loaded', False)
                    logger.info("Modular ensemble reloaded (loaded=%s)", self._ensemble_loaded)
                except Exception as _me_err:
                    logger.error("Modular ensemble reload failed: %s", _me_err)

            # 3. Reload agent weights (Bayesian + learned)
            if hasattr(self, '_agent_team') and self._agent_team is not None:
                try:
                    self._agent_team.reload_learned_weights()
                    logger.info("Agent weights reloaded")
                except Exception as _aw_err:
                    logger.debug("Agent weight reload failed: %s", _aw_err)

            # 4. Re-init gate evaluator immediately (don't wait for next scan)
            gate_ok = self._init_gate_evaluator()
            logger.info("Gate evaluator re-initialized: %s", gate_ok)
            return True
        except Exception as exc:
            logger.error("reload_models failed: %s", exc, exc_info=True)
            return False

    # ═══════════════════════════════════════════════════════════════════════
    # State Persistence (Phase 74)
    # ═══════════════════════════════════════════════════════════════════════
    _SCAN_STATE_PATH = Path("trained_data/.scanner_state.json")

    def save_scan_state(self) -> None:
        """Persist mutable scanner state to disk (atomic write)."""
        state = {
            "scan_cycle_count": self._scan_cycle_count,
            "ewma_prev_prices": dict(self._ewma_prev_prices),
            "last_scan_times": {k: v for k, v in self._last_scan_times.items()},
            "last_prices": dict(self._last_prices),
            "pair_last_regime": dict(self._pair_last_regime),
        }
        tmp_path = self._SCAN_STATE_PATH.with_suffix(".tmp")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w") as f:
                _json.dump(state, f, indent=2, sort_keys=True, default=str)
            import os as _os
            _os.rename(str(tmp_path), str(self._SCAN_STATE_PATH))
            logger.debug("Scanner state saved (%d pairs tracked)", len(state["last_prices"]))
        except Exception as e:
            logger.warning("Failed to save scanner state: %s", e)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def load_scan_state(self) -> None:
        """Restore mutable scanner state from disk (graceful on missing/corrupt)."""
        if not self._SCAN_STATE_PATH.exists():
            return
        try:
            with open(self._SCAN_STATE_PATH) as f:
                state = _json.load(f)
            self._scan_cycle_count = int(state.get("scan_cycle_count", 0))
            self._ewma_prev_prices = state.get("ewma_prev_prices", {})
            self._last_scan_times = {k: float(v) for k, v in state.get("last_scan_times", {}).items()}
            self._last_prices = {k: float(v) for k, v in state.get("last_prices", {}).items()}
            self._pair_last_regime = state.get("pair_last_regime", {})
            logger.info(
                "Scanner state restored: cycle=%d, %d pairs",
                self._scan_cycle_count, len(self._last_prices),
            )
        except (ValueError, KeyError, TypeError, _json.JSONDecodeError) as e:
            logger.warning("Corrupt scanner state file, starting fresh: %s", e)
        except Exception as e:
            logger.warning("Failed to load scanner state: %s", e)

    def _init_feature_engineer(self) -> bool:
        """Initialize feature engineering module.

        Returns:
            True if initialized successfully
        """
        if self._feature_engineer is not None:
            return True

        try:
            from src.core.modular_data_loaders import compute_normalized_features
            # Store function reference directly
            self._feature_engineer = compute_normalized_features
            return True
        except ImportError:
            logger.debug("compute_normalized_features not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize feature engineer: {e}")
            return False

    def _init_modular_ensemble(self) -> bool:
        """Initialize modular ensemble for direction prediction.

        Fallback chain:
        1. Try ModularEnsembleInference (primary)
        2. If fails, try MultiPairInference (fallback from multi_pair_inference.py)

        Returns:
            True if ensemble loaded successfully
        """
        if self._modular_ensemble is not None:
            self._sync_modular_ensemble_config()
            return self._ensemble_loaded

        self._ensemble_loaded = False
        self._ensemble_type = None  # Track which ensemble type loaded

        # === PRIMARY: Try ModularEnsembleInference ===
        try:
            from src.core.modular_inference import InferenceConfig, ModularEnsembleInference

            # Scanner is an opportunity finder, not an execution gatekeeper.
            # Use a softer inference profile so model-contract mismatches don't
            # zero out all candidate signals.
            scan_inference_cfg = InferenceConfig(
                min_tcn_probability=float(self.config.min_tcn_probability),
                min_confidence=float(self.config.min_confidence),
                min_momentum=float(self.config.min_momentum),
                max_drawdown_pct=float(self.config.max_drawdown_pct),
                max_uncertainty_std=float(self.config.max_uncertainty_std),
                use_tcn_volatility_filter=bool(self.config.use_tcn_volatility_filter),
                min_volatility_regime=int(self.config.min_volatility_regime),
                tcn_required=bool(self.config.require_tcn_volatility),
                # Opportunity-mode: allow partial model stacks and feature-schema drift.
                enforce_model_contracts=False,
                require_direction_model=False,
                require_volatility_model=False,
                require_regime_model=False,
                require_xgb_model=False,
                require_rf_model=False,
                require_ridge_model=False,
                # Scanner should not block on contextual overlays.
                sentiment_block_enabled=False,
                enable_meta_labeling=False,
                overlay_meta_enabled=False,
                final_score_threshold=float(self.config.final_score_threshold),
            )

            self._modular_ensemble = ModularEnsembleInference(
                model_dir=str(self.config.model_dir),
                config=scan_inference_cfg,
                use_rl_sizer=self.config.use_rl_sizer,
                use_rl_gates=self.config.use_rl_gates,
                use_rl_exits=self.config.use_rl_exits,
                enable_market_intelligence=False,
            )
            # Eagerly load models so we can check what actually loaded.
            # ModularEnsembleInference lazy-loads in predict(), but the
            # scanner needs to know the loaded state *now* to decide
            # whether to fall back to the gate evaluator.
            self._modular_ensemble.load_models()

            # Tier 6: Inject ensemble weighter into inference module
            if self._ensemble_weighter is not None:
                self._modular_ensemble._ensemble_weighter = self._ensemble_weighter

            # Tier 6 Phase 1: Inject MAML Ridge shadow into inference module
            if self._maml_ridge is not None:
                self._modular_ensemble._maml_ridge = self._maml_ridge
            # Tier 6 Phase 2: Inject benchmark into inference module
            if self._maml_benchmark is not None:
                self._modular_ensemble._maml_benchmark = self._maml_benchmark

            # Check if at least the main direction model loaded
            self._ensemble_loaded = (
                self._modular_ensemble._loaded
                and (
                    self._modular_ensemble.tcn is not None
                    or self._modular_ensemble.regime_model is not None
                    or self._modular_ensemble.histgb is not None
                )
            )

            if self._ensemble_loaded:
                self._sync_modular_ensemble_config()
                self._ensemble_type = "ModularEnsembleInference"
                self._assess_ensemble_health()
                logger.info("✓ ModularEnsembleInference loaded successfully")
                return True
            else:
                logger.warning("⚠️ ModularEnsembleInference loaded but no direction model available")

        except ImportError as e:
            logger.debug(f"ModularEnsembleInference not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize ModularEnsembleInference: {e}")

        # === FALLBACK: Try MultiPairInference ===
        logger.info("Attempting MultiPairInference fallback...")
        try:
            from multi_pair_inference import MultiPairInference

            self._modular_ensemble = MultiPairInference(
                models_dir=str(self.config.model_dir),
            )

            # MultiPairInference may have different loading mechanism
            if hasattr(self._modular_ensemble, 'load_models'):
                self._modular_ensemble.load_models()
            elif hasattr(self._modular_ensemble, 'load'):
                self._modular_ensemble.load()

            # Check if loaded successfully
            # MultiPairInference stores models differently
            has_models = (
                hasattr(self._modular_ensemble, 'models') and self._modular_ensemble.models
            ) or (
                hasattr(self._modular_ensemble, '_pair_models') and self._modular_ensemble._pair_models
            ) or (
                hasattr(self._modular_ensemble, '_loaded') and self._modular_ensemble._loaded
            )

            if has_models or hasattr(self._modular_ensemble, 'predict'):
                self._ensemble_loaded = True
                self._ensemble_type = "MultiPairInference"
                logger.info("✓ MultiPairInference loaded as fallback")
                return True
            else:
                logger.warning("⚠️ MultiPairInference loaded but may not be functional")
                # Still mark as loaded if predict method exists - it has technical fallback
                if hasattr(self._modular_ensemble, 'predict'):
                    self._ensemble_loaded = True
                    self._ensemble_type = "MultiPairInference"
                    return True

        except ImportError as e:
            logger.debug(f"MultiPairInference not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize MultiPairInference fallback: {e}")

        # No ensemble available - will fall back to gate evaluator
        logger.warning("No ensemble inference available - using gate evaluator or technical indicators")
        return False

    def _sync_modular_ensemble_config(self) -> None:
        """Push current scanner thresholds into the mutable ensemble config."""
        ensemble_cfg = getattr(getattr(self, "_modular_ensemble", None), "config", None)
        if ensemble_cfg is None:
            return

        ensemble_cfg.min_tcn_probability = float(self.config.min_tcn_probability)
        ensemble_cfg.min_confidence = float(self.config.min_confidence)
        ensemble_cfg.min_momentum = float(self.config.min_momentum)
        ensemble_cfg.max_drawdown_pct = float(self.config.max_drawdown_pct)
        ensemble_cfg.max_uncertainty_std = float(self.config.max_uncertainty_std)
        ensemble_cfg.use_tcn_volatility_filter = bool(self.config.use_tcn_volatility_filter)
        ensemble_cfg.min_volatility_regime = int(self.config.min_volatility_regime)
        ensemble_cfg.tcn_required = bool(self.config.require_tcn_volatility)
        ensemble_cfg.final_score_threshold = float(self.config.final_score_threshold)

    def _assess_ensemble_health(self) -> Dict[str, Any]:
        """Check which model types loaded and compute degradation penalty.

        US-033: When fewer than 3 of 5 expected model types are available,
        apply a proportional confidence penalty: confidence *= available/expected.

        Returns:
            Dict with available, missing, penalty, and degraded status.
        """
        expected_models = {
            "tcn": "tcn",
            "histgb": "histgb",
            "regime_model": "regime_model",
            "xgb": "xgb",
            "ridge": "ridge",
        }
        available = []
        missing = []

        ensemble = self._modular_ensemble
        if ensemble is not None and self._ensemble_type == "ModularEnsembleInference":
            for name, attr in expected_models.items():
                model = getattr(ensemble, attr, None)
                if model is not None:
                    available.append(name)
                else:
                    missing.append(name)
        elif ensemble is not None:
            # MultiPairInference fallback — treat as fully degraded
            available = ["fallback"]
            missing = list(expected_models.keys())

        expected_count = len(expected_models)
        available_count = len(available) if "fallback" not in available else 1
        penalty = available_count / expected_count if expected_count > 0 else 0.0
        degraded = available_count < 3

        health = {
            "available": available,
            "missing": missing,
            "expected_count": expected_count,
            "available_count": available_count,
            "degraded": degraded,
            "penalty": round(penalty, 3),
        }
        self._ensemble_health = health

        # Log to observations.jsonl at startup
        if degraded:
            try:
                from .automation.observation_log import ObservationLog
                obs = ObservationLog()
                obs.log_observation(
                    pair="SYSTEM",
                    category="degraded_ensemble",
                    description=(
                        f"Ensemble running degraded: {available_count}/{expected_count} models. "
                        f"Available: {available}. Missing: {missing}. "
                        f"Confidence penalty: {penalty:.1%}"
                    ),
                    metadata=health,
                )
            except Exception as e:
                logger.debug(f"Could not log ensemble health observation: {e}")

            logger.warning(
                "⚠️ DEGRADED ENSEMBLE: %d/%d models loaded. "
                "Missing: %s. Confidence penalty: %.1f%%",
                available_count, expected_count, missing, penalty * 100,
            )
        else:
            logger.info(
                "✓ Ensemble health: %d/%d models loaded (%s)",
                available_count, expected_count, available,
            )

        return health

    def _init_analysis_tools(self) -> None:
        """Initialize analysis and filter components."""
        if self._backtester is None:
            self._backtester = QuickBacktester(
                window=50,
                granularity=self.config.granularity,
            )

        if self._backtest_gate is None:
            self._backtest_gate = BacktestGate(
                min_win_rate=0.55,
                lookback_candles=20,
                confidence_penalty=0.15,
            )

        if self._correlation_analyzer is None:
            self._correlation_analyzer = CorrelationAnalyzer(
                threshold=0.7,
                lookback_periods=100,
            )

        if self._drift_detector is None:
            self._drift_detector = DriftDetector(
                threshold=0.03,
                model_dir=self.config.model_dir,
            )

        if self._volatility_filter is None:
            self._volatility_filter = VolatilityFilter(
                min_atr_pips=self.config.min_atr_pips,
                model_dir=self.config.model_dir,
            )

        if self._diversification_filter is None:
            self._diversification_filter = DiversificationFilter(
                correlation_threshold=0.7,
            )

    def _fetch_pair_data(
        self,
        pair: str,
        count: int = 500,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a pair.

        Tries BrokerClient (or legacy OANDA) first, falls back to local CSV files in market_data/.

        Args:
            pair: Instrument name (e.g., "EUR_USD")
            count: Number of candles to fetch

        Returns:
            DataFrame with OHLCV columns or None on failure
        """
        # === Try BrokerClient first (preferred) ===
        if self._init_broker_client():
            try:
                from src.brokers.instrument import Instrument

                # Create Instrument from pair string (e.g., "EUR_USD")
                # This is a basic mapping; adjust as needed for your broker
                instrument = Instrument(
                    symbol=pair,
                    broker_symbol=pair,  # Phase 79: OANDA requires underscore format (GBP_USD not GBP/USD)
                    asset_class="FX",
                    price_precision=5,
                    margin_requirement=0.02,
                    exchange="OANDA",
                    currency="USD",
                    pip_value=0.0001 * 100000,  # Standard FX pip value: 0.0001 * 100k = $10 per pip
                )

                candle_list = self._broker.fetch_candles(
                    instrument=instrument,
                    granularity=self.config.granularity,
                    count=count,
                )

                if candle_list:
                    data = []
                    for candle in candle_list:
                        data.append({
                            "time": candle.time,
                            "open": candle.open,
                            "high": candle.high,
                            "low": candle.low,
                            "close": candle.close,
                            "volume": candle.volume,
                        })

                    if data:
                        df = pd.DataFrame(data)
                        df["time"] = pd.to_datetime(df["time"])
                        df = df.set_index("time")

                        if len(df) >= self.config.min_candles:
                            return df
                        else:
                            logger.debug(f"{pair}: Insufficient broker data ({len(df)} candles)")

            except Exception as e:
                logger.debug(f"{pair}: Broker fetch failed - {e}")

        # === Fallback to legacy OANDA client ===
        if self._init_oanda_client():
            try:
                raw = self._oanda.get_candles(
                    instrument=pair,
                    granularity=self.config.granularity,
                    count=count,
                )

                # Parse OANDA JSON response into DataFrame
                candles = raw.get("candles", [])
                if candles:
                    data = []
                    for c in candles:
                        if not c.get("complete", True):
                            continue
                        mid = c.get("mid", {})
                        data.append({
                            "time": c.get("time"),
                            "open": float(mid.get("o", 0)),
                            "high": float(mid.get("h", 0)),
                            "low": float(mid.get("l", 0)),
                            "close": float(mid.get("c", 0)),
                            "volume": int(c.get("volume", 0)),
                        })

                    if data:
                        df = pd.DataFrame(data)
                        df["time"] = pd.to_datetime(df["time"])
                        df = df.set_index("time")

                        if len(df) >= self.config.min_candles:
                            return df
                        else:
                            logger.debug(f"{pair}: Insufficient OANDA data ({len(df)} candles)")

            except Exception as e:
                logger.debug(f"{pair}: OANDA fetch failed - {e}")

        # === Fallback to local CSV files ===
        df = self._load_local_csv(pair)
        if df is not None:
            return df

        logger.warning(f"{pair}: No data available (broker offline, OANDA offline, no local CSV)")
        return None

    def _load_local_csv(self, pair: str) -> Optional[pd.DataFrame]:
        """Load most recent local CSV data for a pair.

        Searches market_data/ for the most recent H1 CSV matching the pair.

        Args:
            pair: Instrument name (e.g., "EUR_USD")

        Returns:
            DataFrame with OHLCV columns or None if not found
        """
        data_dir = Path(self.config.model_dir).parent.parent / "market_data"
        if not data_dir.exists():
            return None

        # Find matching CSVs for this pair and granularity
        pattern = f"oanda_{pair}_{self.config.granularity}_*.csv"
        csv_files = sorted(data_dir.glob(pattern), reverse=True)  # Most recent first

        if not csv_files:
            logger.debug(f"{pair}: No local CSV files matching {pattern}")
            return None

        csv_path = csv_files[0]
        logger.info(f"{pair}: Using local CSV: {csv_path.name}")

        try:
            df = pd.read_csv(csv_path)

            # Normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Try common time column names
            time_col = None
            for col in ["time", "timestamp", "date", "datetime"]:
                if col in df.columns:
                    time_col = col
                    break

            if time_col:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col)

            # Ensure required OHLCV columns exist
            required = ["open", "high", "low", "close"]
            if not all(c in df.columns for c in required):
                logger.debug(f"{pair}: CSV missing required columns: {required}")
                return None

            if "volume" not in df.columns:
                df["volume"] = 0

            if len(df) < self.config.min_candles:
                logger.debug(f"{pair}: Insufficient CSV data ({len(df)} candles)")
                return None

            return df

        except Exception as e:
            logger.warning(f"{pair}: Failed to load CSV {csv_path.name}: {e}")
            return None

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical features for analysis.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            DataFrame with engineered features
        """
        if not self._init_feature_engineer():
            # Basic fallback features
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            if 'atr' not in df.columns:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = tr.rolling(14).mean()
            return df.ffill().bfill().fillna(0)

        try:
            # _feature_engineer is compute_normalized_features — matches training pipeline
            df_feat = self._feature_engineer(df.copy())
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            df_feat = df_feat.ffill().bfill().fillna(0.0)
            return df_feat
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}")
            return df.ffill().bfill().fillna(0)

    def _check_volatility_regime(
        self,
        df_feat: pd.DataFrame,
        pair: str,
    ) -> Tuple[bool, Optional[int]]:
        """Check TCN Volatility Regime as a global gate.

        Args:
            df_feat: Engineered features
            pair: Instrument name

        Returns:
            Tuple of (vol_allowed, volatility_regime)
        """
        if not self._init_gate_evaluator() or self._gate_evaluator is None:
            return True, None
        if not getattr(self.config, "use_tcn_volatility_filter", True):
            return True, None

        if not self._gate_evaluator.tcn_volatility_available:
            return True, None

        regime, _vol_conf, vol_allowed = self._gate_evaluator.evaluate_volatility_regime(df_feat)

        if not vol_allowed:
            regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
            logger.info(f"{pair}: Blocked by TCN Volatility Regime ({regime_names[regime]})")

        return vol_allowed, regime

    def _resolve_model_source_pair(self, pair: str) -> str:
        """Resolve which pair's model artifacts should source inference."""
        normalized_pair = str(pair).upper().replace("/", "_")
        if not bool(getattr(self.config, "use_master_pair_models", True)):
            return normalized_pair
        try:
            from src.training.correlation_group_config import get_correlation_group

            group = get_correlation_group(normalized_pair)
            if group is None:
                return normalized_pair
            master = str(group.master_pair).upper().replace("/", "_")
            return master or normalized_pair
        except Exception as e:
            logger.debug("pair %s: correlation group normalization skipped: %s", pair, e)
            return normalized_pair

    def _infer_from_ensemble(
        self,
        df_feat: pd.DataFrame,
        pair: str,
    ) -> Tuple[
        Optional[str],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[bool],
        Optional[int],
        Optional[Dict[str, Any]],
    ]:
        """Run inference using modular ensemble if available.

        Args:
            df_feat: Engineered features
            pair: Instrument name

        Returns:
            Tuple of
            (direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, gate_details)
            or all None if not available.
        """
        if not self._init_modular_ensemble() or self._modular_ensemble is None:
            return None, None, None, None, None, None, None

        try:
            model_source_pair = self._resolve_model_source_pair(pair)

            # --- MultiPairInference fallback path ---
            # MultiPairInference.predict() has a different signature and returns
            # a simple (direction, confidence) tuple instead of a signal object.
            if self._ensemble_type == "MultiPairInference":
                with self._ensemble_lock:
                    direction_raw, conf_raw = self._modular_ensemble.predict(
                        df_feat, pair,
                    )
                direction = direction_raw.upper() if direction_raw else "HOLD"
                confidence = float(conf_raw) if conf_raw is not None else 0.5
                confidence = confidence / 100.0 if confidence > 1.0 else confidence
                confidence = min(max(confidence, 0.0), 1.0)
                details = {
                    "tcn_probability": 0.5,
                    "momentum": 0.0,
                    "momentum_acceleration": False,
                    "momentum_passed": True,
                    "confidence_score": confidence,
                    "confidence_passed": confidence >= 0.55,
                    "risk_passed": True,
                    "volatility_gate_passed": True,
                    "drawdown": None,
                    "meta_confidence": confidence,
                    "trade_allowed": confidence >= 0.55,
                    "volatility_regime_name": "UNKNOWN",
                    "reason": "MultiPairInference fallback",
                    "reason_codes": [],
                    "core_score": confidence,
                    "final_score": confidence,
                    "model_source_pair": model_source_pair,
                }
                return (
                    direction,
                    confidence,
                    0.0,   # tcn_conf
                    confidence,  # ridge_conf
                    confidence >= 0.55,  # gates_passed
                    None,  # volatility_regime
                    details,
                )

            # --- Primary: ModularEnsembleInference path ---
            # Shared ModularEnsembleInference is mutable (instrument-specific loading),
            # so serialize calls across scanner worker threads.
            with self._ensemble_lock:
                signal = self._modular_ensemble.predict(
                    df_feat,
                    instrument=pair,
                    model_instrument=model_source_pair,
                )

            tcn_conf = float(abs(float(signal.tcn_probability) - 0.5) * 200.0)
            ridge_conf = float(signal.ridge_confidence or 0.0)

            # Phase 59 (US-365): Record TCN vs Ridge ensemble divergence
            if self._ensemble_divergence is not None:
                try:
                    self._ensemble_divergence.record(
                        pair=pair,
                        tcn_confidence=tcn_conf,
                        ridge_confidence=ridge_conf,
                        scan_id=str(getattr(self, "_scan_cycle", 0)),
                    )
                except Exception as _edm_err:
                    logger.debug(f"Phase 59: ensemble_divergence.record failed: {_edm_err}")

            raw_conf = float(signal.confidence or 0.0)
            confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
            confidence = min(max(confidence, 0.0), 1.0)
            direction = signal.direction.upper() if signal.direction is not None else "HOLD"
            gates_passed = signal.trade
            signal_meta = signal.metadata if isinstance(signal.metadata, dict) else {}
            score_meta = signal_meta.get("scores") if isinstance(signal_meta.get("scores"), dict) else {}

            # Phase 61 (US-371): Record raw XGBoost momentum score
            if self._momentum_recorder is not None:
                try:
                    self._momentum_recorder.record(
                        pair=pair,
                        momentum_score=float(signal.xgb_momentum),
                        momentum_passed=bool(signal.momentum_gate_passed),
                        scan_id=str(getattr(self, "_scan_cycle", 0)),
                    )
                except Exception as _msr_err:
                    logger.debug(f"Phase 61: momentum_recorder.record failed: {_msr_err}")

            # Phase 63 (US-378): Record rf_drawdown_pct and rf_streak_prob at risk gate site
            if self._risk_recorder is not None:
                try:
                    _rf_drawdown_pct = (
                        float(signal.rf_drawdown_pips) / 10000.0
                        if signal.rf_drawdown_pips
                        else 0.0
                    )
                    self._risk_recorder.record(
                        pair=pair,
                        rf_drawdown_pct=_rf_drawdown_pct,
                        rf_streak_prob=float(signal.rf_streak_prob),
                        risk_passed=bool(signal.risk_gate_passed),
                        max_drawdown_pct=float(self.config.max_drawdown_pct),
                        scan_id=str(getattr(self, "_scan_cycle", 0)),
                    )
                except Exception as _rrr_err:
                    logger.debug(f"Phase 63: _risk_recorder.record( failed: {_rrr_err}")

            details = {
                "tcn_probability": float(signal.tcn_probability),
                "momentum": float(signal.xgb_momentum),
                "momentum_acceleration": bool(signal.xgb_acceleration),
                "momentum_passed": bool(signal.momentum_gate_passed),
                "confidence_score": float(signal.ridge_confidence),
                "confidence_passed": bool(signal.confidence_gate_passed),
                "risk_passed": bool(signal.risk_gate_passed),
                "volatility_gate_passed": bool(signal.volatility_gate_passed),
                "drawdown": float(signal.rf_drawdown_pips) / 10000.0 if signal.rf_drawdown_pips else None,
                "meta_confidence": float(signal.meta_confidence),
                "trade_allowed": bool(signal.trade),
                "volatility_regime_name": str(signal.volatility_regime_name or "UNKNOWN"),
                "reason": signal.reason,
                "reason_codes": signal_meta.get("reason_codes", []),
                "core_score": score_meta.get("core_score"),
                "final_score": score_meta.get("final_score"),
                "scores": {
                    "direction_score": score_meta.get("direction_score"),
                    "confidence_score": score_meta.get("confidence_score"),
                    "momentum_score": score_meta.get("momentum_score"),
                    "risk_score": score_meta.get("risk_score"),
                    "used_learned_weights": score_meta.get("used_learned_weights", False),
                    "ensemble_weights": score_meta.get("ensemble_weights"),
                },
                "model_source_pair": model_source_pair,
            }
            return (
                direction,
                float(confidence),
                float(tcn_conf),
                float(ridge_conf),
                bool(gates_passed),
                signal.volatility_regime,
                details,
            )

        except Exception as e:
            logger.warning(f"{pair}: Ensemble inference failed - {e}")
            return None, None, None, None, None, None, None

    def _infer_from_gates(
        self,
        df_feat: pd.DataFrame,
        pair: str,
    ) -> Tuple[
        Optional[str],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[bool],
        Optional[int],
        Optional[Dict[str, Any]],
    ]:
        """Run inference using gate evaluator if available.

        Args:
            df_feat: Engineered features
            pair: Instrument name

        Returns:
            Tuple of
            (direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, gate_details)
            or all None if not available.
        """
        if not self._init_gate_evaluator() or self._gate_evaluator is None:
            return None, None, None, None, None, None, None

        try:
            gate_result = self._gate_evaluator.evaluate_all_gates(
                df_feat,
                min_confidence=self.config.min_confidence,
                min_momentum=self.config.min_momentum,
                max_drawdown_pct=self.config.max_drawdown_pct,
                instrument=pair,
            )

            ridge_conf = gate_result["confidence"]
            gates_passed = gate_result["all_passed"]

            # Use direction from gates if Transformer provided it
            if gate_result.get("transformer_direction"):
                direction = gate_result["transformer_direction"]
                tcn_conf = gate_result.get("transformer_prob", 0.5) * 100
            elif "rsi" in df_feat.columns:
                rsi = df_feat["rsi"].iloc[-1]
                direction = "LONG" if rsi < 50 else "SHORT"
                tcn_conf = 0.0
            else:
                return None, None, None, None, None, None, None

            # Overall confidence: use ridge confidence (0-100) scaled to 0-1,
            # consistent with the ensemble path which does max(tcn_conf, ridge_conf) / 100.
            confidence = max(tcn_conf, ridge_conf) / 100

            details = {
                "momentum": float(gate_result.get("momentum", 0.0)),
                "momentum_acceleration": bool(gate_result.get("momentum_acceleration", False)),
                "momentum_passed": bool(gate_result.get("momentum_passed", False)),
                "confidence_score": float(gate_result.get("confidence", 0.0)),
                "confidence_passed": bool(gate_result.get("confidence_passed", False)),
                "risk_passed": bool(gate_result.get("risk_passed", False)),
                "volatility_gate_passed": True,
                "drawdown": float(gate_result.get("drawdown", 0.0)),
                "model_source_pair": str(pair).upper().replace("/", "_"),
            }
            return direction, min(confidence, 0.95), tcn_conf, ridge_conf, gates_passed, None, details

        except Exception as e:
            logger.warning(f"{pair}: Gate evaluation failed - {e}")
            return None, None, None, None, None, None, None

    def _infer_from_technicals(
        self,
        df_feat: pd.DataFrame,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Run inference using technical indicators as fallback.

        Args:
            df_feat: Engineered features

        Returns:
            Tuple of (direction, confidence) or (None, None) if not available
        """
        if "rsi" not in df_feat.columns:
            return None, None

        rsi = df_feat["rsi"].iloc[-1]

        if rsi < 30:
            direction = "LONG"
            confidence = 0.5 + (30 - rsi) / 60
        elif rsi > 70:
            direction = "SHORT"
            confidence = 0.5 + (rsi - 70) / 60
        elif "macd" in df_feat.columns:
            macd = df_feat["macd"].iloc[-1]
            direction = "LONG" if macd > 0 else "SHORT"
            confidence = 0.5 + min(abs(macd) * 10, 0.2)
        else:
            direction = "LONG" if rsi < 50 else "SHORT"
            confidence = 0.5 + abs(rsi - 50) / 100

        return direction, min(confidence, 0.95)

    def _run_inference(
        self,
        df_raw: pd.DataFrame,
        df_feat: pd.DataFrame,
        pair: str,
    ) -> Tuple[str, float, float, float, bool, Optional[int], bool, Dict[str, Any]]:
        """Run model inference on features.

        Inference architecture priority:
        1) Modular ensemble (includes full architecture gates + volatility gate)
        2) Gate evaluator fallback (legacy scanner path)
        3) Technical indicator fallback (no gates)

        Args:
            df_raw: Raw OHLCV data
            df_feat: Engineered features
            pair: Instrument name

        Returns:
            Tuple of (
                direction,
                confidence,
                tcn_conf,
                ridge_conf,
                gates_passed,
                volatility_regime,
                inference_succeeded,
                gate_details,
            )
        """
        direction = "HOLD"
        confidence = 0.0
        tcn_conf = 0.0
        ridge_conf = 0.0
        gates_passed = False
        volatility_regime = None
        _inference_succeeded = False
        details: Dict[str, Any] = {}

        # === FIRST: Try full modular inference architecture ===
        result = self._infer_from_ensemble(df_feat, pair)
        if result[0] is not None:
            direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, details = result

            # US-106: Model router — bandit-based per-pair model weighting
            if self._model_router is not None:
                try:
                    _regime_name = str(
                        (details or {}).get("volatility_regime_name", "NORMAL")
                    ).upper()
                    _selection = self._model_router.select_models(pair, _regime_name)
                    details = details or {}
                    details["model_routing"] = _selection.to_dict()
                    if _selection.strategy == "ROUTED" and tcn_conf > 0 and ridge_conf > 0:
                        _model_conf_map = {
                            "TCN": tcn_conf,
                            "Ridge": ridge_conf,
                            "RF": ridge_conf,
                        }
                        _primary_conf = _model_conf_map.get(_selection.primary_model)
                        _fallback_conf = _model_conf_map.get(_selection.fallback_model)
                        if _primary_conf is not None and _fallback_conf is not None:
                            _routed_conf = self._model_router.blend_predictions(
                                _primary_conf, _fallback_conf,
                            )
                            _pre_route_conf = confidence
                            confidence = confidence * 0.70 + _routed_conf * 0.30
                            confidence = min(max(confidence, 0.0), 1.0)
                            details["model_routing"]["routed_confidence"] = round(_routed_conf, 4)
                            details["model_routing"]["pre_route_confidence"] = round(_pre_route_conf, 4)
                            details["model_routing"]["post_route_confidence"] = round(confidence, 4)
                            logger.debug(
                                "%s: ModelRouter %s conf %.3f (ensemble=%.3f, routed=%.3f)",
                                pair, _selection.primary_model, confidence,
                                _pre_route_conf, _routed_conf,
                            )
                except Exception as _route_err:
                    logger.debug(f"{pair}: ModelRouter error (fallback to ensemble): {_route_err}")

            # US-033: Apply degraded ensemble confidence penalty
            health = self._ensemble_health
            if health.get("degraded", False):
                penalty = health.get("penalty", 1.0)
                original_confidence = confidence
                confidence = confidence * penalty
                details = details or {}
                details["degraded_ensemble"] = True
                details["ensemble_penalty"] = penalty
                details["original_confidence"] = original_confidence
                details["available_models"] = health.get("available", [])
                details["missing_models"] = health.get("missing", [])
                logger.info(
                    "Degraded ensemble penalty applied to %s: %.3f → %.3f (×%.2f)",
                    pair, original_confidence, confidence, penalty,
                )

            return direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, True, (details or {})

        # === SECOND: Legacy fallback volatility gate ===
        vol_allowed, volatility_regime = self._check_volatility_regime(df_feat, pair)
        if not vol_allowed:
            details["volatility_gate_passed"] = False
            return direction, confidence, tcn_conf, ridge_conf, False, volatility_regime, True, details

        # === THIRD: Try gate evaluator ===
        result = self._infer_from_gates(df_feat, pair)
        if result[0] is not None:
            direction, confidence, tcn_conf, ridge_conf, gates_passed, _regime, details = result
            if _regime is not None:
                volatility_regime = _regime
            return direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, True, (details or {})

        # === FOURTH: Technical indicator fallback ===
        result = self._infer_from_technicals(df_feat)
        if result[0] is not None:
            direction, confidence = result
            return direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, True, details

        return direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, _inference_succeeded, details

    def _detect_regime(self, df_raw: pd.DataFrame, pair: str) -> Optional[Any]:
        """Phase 44 (US-278): Run BOCPD+Hurst+ADX regime detection.

        Args:
                df_raw: Raw OHLCV DataFrame with 'close', 'high', 'low' columns
                pair: Instrument name for logging

        Returns:
                RegimeDetectionResult or None on failure
        """
        if self._market_regime_detector is None:
            return None
        try:
            prices = df_raw["close"].values.astype(float)
            highs = df_raw["high"].values.astype(float) if "high" in df_raw.columns else prices
            lows = df_raw["low"].values.astype(float) if "low" in df_raw.columns else prices
            result = self._market_regime_detector.update(prices, highs, lows)
            logger.debug(
                "Phase 44: %s regime=%s conf=%.2f hurst=%.2f adx=%.1f changepoint=%.3f",
                pair, result.regime_name, result.confidence, result.hurst_exponent, result.adx_value, result.changepoint_probability,
            )
            return result
        except Exception as e:
            logger.warning(f"Phase 44: Regime detection failed for {pair}: {e}")
            return None

    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        pair: str,
    ) -> Dict[str, float]:
        """Calculate technical metrics.

        Args:
            df: DataFrame with features
            pair: Instrument name

        Returns:
            Dict of metric values
        """
        metrics = {
            "current_price": df["close"].iloc[-1],
            "atr": (
                float(df["atr"].iloc[-1])
                if "atr" in df.columns
                else (
                    # Feature engineering produces atr_pct_14 (ATR/price ratio).
                    # Recover raw ATR: atr_pct_14 * close price.
                    float(df["atr_pct_14"].iloc[-1] * df["close"].iloc[-1])
                    if "atr_pct_14" in df.columns
                    else 0.0
                )
            ),
            "volatility_percentile": 0.5,
            "trend_strength": 0.5,
            "entry_score": 0.5,
        }

        # Volatility percentile (use atr_pct_14 if raw atr column missing)
        _atr_col = "atr" if "atr" in df.columns else ("atr_pct_14" if "atr_pct_14" in df.columns else None)
        if _atr_col:
            atr_series = df[_atr_col].dropna()
            if len(atr_series) > 20:
                metrics["volatility_percentile"] = float(
                    (atr_series < atr_series.iloc[-1]).mean()
                )

        # Trend strength from ADX
        if "adx" in df.columns:
            adx_val = df["adx"].iloc[-1]
            metrics["trend_strength"] = min(float(adx_val) / 60, 1.0)

        # Entry score
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if 30 < rsi < 70:
                metrics["entry_score"] += 0.15

        if "sma_20" in df.columns and "sma_50" in df.columns:
            sma_20 = df["sma_20"].iloc[-1]
            sma_50 = df["sma_50"].iloc[-1]
            close = df["close"].iloc[-1]

            if (close > sma_20 > sma_50) or (close < sma_20 < sma_50):
                metrics["entry_score"] += 0.15

        return metrics

    def _check_price_changed(
        self,
        pair: str,
        current_price: float,
        threshold: float = 0.0001,
    ) -> bool:
        """Check if price changed enough to warrant rescan.

        Args:
            pair: Instrument name
            current_price: Current price
            threshold: Minimum price change ratio

        Returns:
            True if price changed significantly
        """
        if pair not in self._last_prices:
            self._last_prices[pair] = current_price
            return True

        last_price = self._last_prices[pair]
        change_ratio = abs(current_price - last_price) / last_price

        if change_ratio >= threshold:
            self._last_prices[pair] = current_price
            return True

        return False

    def _build_cache_context(self, pair: str) -> Tuple[Any, ...]:
        """Return the config/model context that makes a cached scan reusable."""
        gate_model_type = None
        if self._gate_evaluator is not None and self._gate_evaluator.is_loaded:
            gate_model_type = self._gate_evaluator.momentum_model_type

        return (
            str(pair).upper().replace("/", "_"),
            self.config.granularity,
            int(self.config.lookback_candles),
            str(self.config.profile),
            bool(self.config.use_master_pair_models),
            self._resolve_model_source_pair(pair),
            round(float(self.config.min_tcn_probability), 6),
            round(float(self.config.min_confidence), 6),
            round(float(self.config.min_momentum), 6),
            round(float(self.config.max_drawdown_pct), 6),
            round(float(self.config.final_score_threshold), 6),
            round(float(self.config.max_uncertainty_std), 6),
            round(float(self.config.min_atr_pips), 6),
            int(self.config.min_volatility_regime),
            bool(self.config.use_tcn_volatility_filter),
            bool(self.config.require_tcn_volatility),
            bool(self.config.enable_sub_inference_agents),
            bool(self.config.enable_trend_agent),
            bool(self.config.enable_mean_reversion_agent),
            bool(self.config.enable_volatility_agent),
            bool(self.config.enable_risk_sentinel_agent),
            bool(self.config.enable_news_risk_agent),
            bool(self.config.enable_uncertainty_agent),
            bool(self.config.enable_execution_quality_agent),
            round(float(self.config.weighted_vote_threshold), 6),
            round(float(self.config.sub_inference_min_confidence), 6),
            round(float(self.config.sub_inference_vote_threshold), 6),
            round(float(self.config.agent_promotion_min_confidence), 6),
            round(float(self.config.max_uncertainty_score), 6),
            round(float(self.config.max_model_disagreement), 6),
            round(float(self.config.max_spread_pips), 6),
            round(float(self.config.max_slippage_pips), 6),
            round(float(self.config.min_liquidity_score), 6),
            bool(getattr(self, "_ensemble_loaded", False)),
            getattr(self, "_ensemble_type", None),
            gate_model_type,
        )

    def _get_cached_result(
        self,
        pair: str,
        cache_context: Tuple[Any, ...],
    ) -> Optional[PairAnalysis]:
        """Return a defensive copy of cached analysis when context still matches."""
        if self._cache_contexts.get(pair) != cache_context:
            return None

        cached = self._cached_results.get(pair)
        if cached is None:
            return None

        cached_copy = deepcopy(cached)
        now = datetime.now(timezone.utc)
        cached_copy.scan_time = now
        cached_copy.timestamp = now
        return cached_copy

    def _is_sub_inference_candidate(self, analysis: PairAnalysis) -> bool:
        """Check whether a pair should run sub-inference confirmation agents."""
        if analysis.error is not None:
            return False
        if analysis.direction not in {"LONG", "SHORT"}:
            return False
        if bool(getattr(self.config, "sub_inference_tradeable_only", True)):
            return bool(analysis.gates_passed)
        if analysis.gates_passed:
            return True
        if analysis.confidence >= float(self.config.sub_inference_min_confidence):
            return True
        return analysis.momentum_passed and (
            analysis.confidence >= float(self.config.sub_inference_min_confidence) * 0.9
        )

    def _run_sub_inference_agents_for_pair(self, analysis: PairAnalysis) -> PairAnalysis:
        """Run rolling-window sub inference agents for one candidate pair."""
        if not self._is_sub_inference_candidate(analysis):
            return analysis

        df_raw = self._raw_snapshots.get(analysis.pair)
        df_feat = self._feature_snapshots.get(analysis.pair)
        if df_raw is None or df_feat is None or len(df_feat) < max(40, self.config.min_candles // 2):
            return analysis

        expected_direction = analysis.direction.upper()
        votes = 0
        total = 0
        score_acc = 0.0
        window_checks = max(1, int(self.config.sub_inference_window_checks))
        min_conf = float(self.config.sub_inference_min_confidence)

        for lag in range(window_checks):
            if lag >= len(df_raw) - 5 or lag >= len(df_feat) - 5:
                break

            raw_window = df_raw if lag == 0 else df_raw.iloc[:-lag]
            feat_window = df_feat if lag == 0 else df_feat.iloc[:-lag]
            if len(raw_window) < 40 or len(feat_window) < 40:
                continue

            sub_direction, sub_conf, _tcn_conf, _ridge_conf, _gates_passed, _regime, ok, _details = (
                self._run_inference(raw_window, feat_window, analysis.pair)
            )
            if not ok or sub_direction not in {"LONG", "SHORT"}:
                continue

            sub_conf = min(max(float(sub_conf), 0.0), 1.0)
            total += 1

            if sub_direction == expected_direction:
                score_acc += sub_conf
                if sub_conf >= min_conf:
                    votes += 1
            else:
                # Opposite-direction confirmations should weaken consensus.
                # Phase 23 (US-141): Configurable penalty (was hardcoded 0.20 — too harsh)
                _opp_penalty = float(getattr(self.config, "sub_inference_opposite_penalty", 0.50))
                score_acc += sub_conf * _opp_penalty

        if total <= 0:
            return analysis

        vote_required = max(1, math.ceil(total * float(self.config.sub_inference_vote_threshold)))
        consensus_ratio = votes / total
        agent_passed = votes >= vote_required
        raw_agent_score = min(max(score_acc / total, 0.0), 1.0)
        agent_score = min(max(raw_agent_score * 0.70 + consensus_ratio * 0.30, 0.0), 1.0)

        analysis.agent_votes = votes
        analysis.agent_total = total
        analysis.agent_score = agent_score
        analysis.agent_passed = agent_passed

        # Phase 67 (US-391): Record sub-inference consensus ratio
        if self._sub_inference_recorder is not None:
            try:
                self._sub_inference_recorder.record(
                    pair=analysis.pair,
                    votes=votes,
                    total=total,
                    agent_passed=agent_passed,
                    scan_id=str(getattr(self, "_scan_cycle", 0)),
                )
            except Exception as _sir_err:
                logger.debug(f"Phase 67: _sub_inference_recorder.record() failed: {_sir_err}")

        # Redo confidence logic: blend base confidence with consensus quality.
        # Phase 29 (US-179): Configurable blend weights from ScannerConfig.
        base_conf = min(max(float(analysis.confidence), 0.0), 1.0)
        _w_base = float(getattr(self.config, "confidence_blend_base_weight", 0.55))
        _w_agent = float(getattr(self.config, "confidence_blend_agent_weight", 0.30))
        _w_cons = float(getattr(self.config, "confidence_blend_consensus_weight", 0.15))
        blended_conf = (
            base_conf * _w_base +
            agent_score * _w_agent +
            consensus_ratio * _w_cons
        )
        if agent_passed:
            _boost_cap = float(getattr(self.config, "confidence_boost_cap", 0.08))
            _boost_factor = float(getattr(self.config, "confidence_boost_consensus_factor", 0.06))
            boosted = blended_conf + min(_boost_cap, consensus_ratio * _boost_factor)
            analysis.confidence = min(max(max(base_conf, boosted), 0.0), 0.99)
        else:
            softened = base_conf * 0.92 + blended_conf * 0.08
            analysis.confidence = min(max(softened, 0.0), 0.99)

        # ── Minimum agent consensus hard gate (US-032 + US-130) ─────────
        # Phase 21 (US-130): Regime-specific consensus thresholds.
        # EXTREME market = directional, fewer agents need to agree.
        # LOW market = noisy, require stronger consensus.
        _regime_str = str(getattr(analysis, "volatility_regime", "UNKNOWN")).upper()
        if getattr(self.config, "enable_regime_consensus", True):
            _regime_map = getattr(self.config, "regime_consensus_map", {})
            min_consensus_ratio = float(_regime_map.get(
                _regime_str, getattr(self.config, "min_agent_consensus_ratio", 0.25)
            ))
        else:
            min_consensus_ratio = float(getattr(
                self.config, "min_agent_consensus_ratio", 0.25
            ))
        if consensus_ratio < min_consensus_ratio and analysis.is_tradeable:
            _absolute_min = float(getattr(self.config, "soft_consensus_absolute_min", 0.10))
            if getattr(self.config, "enable_soft_consensus_gate", True) and consensus_ratio >= _absolute_min:
                # Phase 23 (US-139): Soft penalty — reduce confidence instead of blocking
                _penalty = float(getattr(self.config, "soft_consensus_penalty", 0.85))
                _old_conf = analysis.confidence
                analysis.confidence = min(max(analysis.confidence * _penalty, 0.0), 0.99)
                logger.info(
                    f"{analysis.pair}: soft consensus penalty applied "
                    f"({votes}/{total} = {consensus_ratio:.0%} < {min_consensus_ratio:.0%}), "
                    f"confidence {_old_conf:.3f} → {analysis.confidence:.3f}"
                )
                # Log to observation_log if available
                if self._observation_log is not None:
                    try:
                        self._observation_log.log_observation(
                            category="consensus_penalty",
                            pair=analysis.pair,
                            data={
                                "consensus_ratio": round(consensus_ratio, 3),
                                "min_required": round(min_consensus_ratio, 3),
                                "penalty": _penalty,
                                "old_confidence": round(_old_conf, 4),
                                "new_confidence": round(analysis.confidence, 4),
                            },
                        )
                    except Exception as e:
                        logger.debug("%s: consensus_penalty observation log write skipped: %s", analysis.pair, e)
            else:
                # Hard block: consensus below absolute minimum safety threshold
                analysis.gates_passed = False
                # Note: is_tradeable is a @property — setting gates_passed=False
                # already makes it return False. No need to set is_tradeable directly.
                why = getattr(analysis, "why_no_trade", None) or []
                if isinstance(why, list):
                    why.append(
                        f"agent consensus too low ({votes}/{total} = "
                        f"{consensus_ratio:.0%} < {min_consensus_ratio:.0%} minimum)"
                    )
                    analysis.why_no_trade = why
                logger.info(
                    f"{analysis.pair}: blocked by min agent consensus "
                    f"({votes}/{total} = {consensus_ratio:.0%})"
                )

        # Agent-consensus promotion: optionally promote to executable setup.
        # SAFETY: promotion must NEVER override specialist agent blocks,
        # dangerous model disagreement, or unknown regime conditions.
        promote = (
            bool(getattr(self.config, "enable_agent_trade_promotion", True))
            and agent_passed
            and analysis.direction in {"LONG", "SHORT"}
            and analysis.confidence >= float(getattr(self.config, "agent_promotion_min_confidence", 0.52))
        )
        if promote and bool(getattr(self.config, "agent_promotion_requires_risk", True)):
            promote = bool(analysis.risk_passed)
        # Hard safety checks — promotion cannot bypass these
        if promote and analysis.blocked_by_circuit_breaker:
            promote = False
            logger.info(f"{analysis.pair}: promotion blocked — circuit breaker active")
        if promote and getattr(analysis, "weighted_vote_score", 0) < float(
            getattr(self.config, "weighted_vote_threshold", 0.55)
        ):
            promote = False
            logger.info(f"{analysis.pair}: promotion blocked — specialist vote too low")
        if promote and str(getattr(analysis, "volatility_regime", "UNKNOWN")).upper() == "UNKNOWN":
            promote = False
            logger.info(f"{analysis.pair}: promotion blocked — regime UNKNOWN")
        if promote and float(getattr(analysis, "model_disagreement", 1.0)) > 0.30:
            promote = False
            logger.info(
                f"{analysis.pair}: promotion blocked — model_disagreement "
                f"{getattr(analysis, 'model_disagreement', 'N/A')} > 0.30"
            )
        # H-7: Explicit three-gate verification before promotion — agent consensus
        # cannot substitute for failed momentum, confidence, or risk gates
        if promote and not (
            getattr(analysis, "momentum_passed", False)
            and getattr(analysis, "confidence_passed", False)
            and getattr(analysis, "risk_passed", False)
        ):
            promote = False
            logger.info(
                f"{analysis.pair}: promotion blocked — not all three gates passed "
                f"(momentum={getattr(analysis, 'momentum_passed', False)}, "
                f"confidence={getattr(analysis, 'confidence_passed', False)}, "
                f"risk={getattr(analysis, 'risk_passed', False)})"
            )
        if promote:
            analysis.gates_passed = True
            analysis.agent_promoted = True

        # Phase 66 (US-388): Record weighted_vote_score at agent consensus site
        if self._vote_recorder is not None:
            try:
                self._vote_recorder.record(
                    pair=analysis.pair,
                    vote_score=float(getattr(analysis, "weighted_vote_score", 0.0)),
                    agent_passed=bool(getattr(analysis, "agent_passed", True)),
                    scan_id=str(getattr(self, "_scan_cycle", 0)),
                )
            except Exception as _vsr_err:
                logger.debug(f"Phase 66: _vote_recorder.record() failed: {_vsr_err}")

        return analysis

    def _run_sub_inference_pass(
        self,
        analyses: List[PairAnalysis],
        on_pair_complete: Optional[Callable[[PairAnalysis], None]] = None,
    ) -> List[PairAnalysis]:
        """Run sub-inference agents in parallel for candidate opportunities."""
        if not self.config.enable_sub_inference_agents:
            return analyses

        candidates = [a for a in analyses if self._is_sub_inference_candidate(a)]
        if not candidates:
            return analyses

        candidates = sorted(
            candidates,
            key=lambda a: (a.gates_passed, a.confidence, a.momentum),
            reverse=True,
        )
        if not bool(getattr(self.config, "sub_inference_tradeable_only", True)):
            max_candidates = max(1, int(self.config.sub_inference_max_candidates))
            candidates = candidates[:max_candidates]

        workers = max(
            1,
            min(
                int(self.config.sub_inference_workers),
                len(candidates),
            ),
        )
        updates: Dict[str, PairAnalysis] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._run_sub_inference_agents_for_pair, analysis): analysis.pair
                for analysis in candidates
            }
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    updated = future.result()
                    updates[pair] = updated
                    if on_pair_complete:
                        on_pair_complete(updated)
                except Exception as e:
                    logger.warning(f"{pair}: sub-inference agent failed - {e}")

        if not updates:
            return analyses
        return [updates.get(a.pair, a) for a in analyses]

    def _apply_specialist_agents(
        self,
        analysis: PairAnalysis,
        df_raw: pd.DataFrame,
        df_feat: pd.DataFrame,
        gate_details: Optional[Dict[str, Any]] = None,
    ) -> PairAnalysis:
        """Apply the scanner specialist-agent layer to one analysis."""
        try:
            # Tier 6: Inject meta-learner overrides before agent evaluation
            if self._meta_learner is not None:
                try:
                    _regime = str(getattr(analysis, "volatility_regime", "NORMAL") or "NORMAL").upper()
                    if _regime == "LOW":
                        _regime = "NORMAL"
                    _overrides: Dict[str, Dict[str, float]] = {}
                    for _agent_name in self._agent_team._BASE_WEIGHTS:
                        _ov = self._meta_learner.get_active_overrides(_agent_name, _regime)
                        if _ov:
                            _overrides[_agent_name] = _ov
                    if _overrides:
                        self._agent_team.apply_meta_overrides(_overrides)
                except Exception as _ml_err:
                    logger.debug("Tier 6: Meta override injection failed: %s", _ml_err)

            return self._agent_team.evaluate(
                analysis=analysis,
                df_raw=df_raw,
                df_feat=df_feat,
                gate_details=gate_details,
            )
        except Exception as e:
            logger.warning(f"{analysis.pair}: specialist agent evaluation failed - {e}")
            return analysis

    def _scan_pair(self, pair: str) -> Optional[PairAnalysis]:
        """Scan a single pair (atomic unit for parallel execution).

        Args:
            pair: Instrument name

        Returns:
            PairAnalysis or None on failure
        """
        try:
            # Check if pair is blocked (model accuracy issues or other constraints)
            if pair in self.config.blocked_pairs:
                logger.warning(f"Skipping {pair}: blocked (model accuracy issue)")
                return PairAnalysis(
                    pair=pair,
                    direction="HOLD",
                    confidence=0.0,
                    error=f"Pair blocked: {pair} is in blocked_pairs list",
                )

            # Phase 49 (US-307): Check pair performance blacklist
            if self._pair_tracker is not None:
                try:
                    _bl_result = self._pair_tracker.check_pair(pair)
                    if _bl_result.is_blacklisted:
                        logger.info(
                            "Phase 49 (US-307): Skipping %s — blacklisted (%s)",
                            pair, _bl_result.reason,
                        )
                        return PairAnalysis(
                            pair=pair,
                            direction="HOLD",
                            confidence=0.0,
                            error=f"Pair blacklisted: {_bl_result.reason}",
                        )
                except Exception as _bl_err:
                    logger.debug(f"Phase 49: Pair blacklist check error for {pair}: {_bl_err}")

            # Phase 78: Accuracy gate — block pairs with poor rolling accuracy
            if self._accuracy_gate is not None:
                try:
                    _ag_allowed, _ag_accuracy, _ag_reason = self._accuracy_gate.check_pair(pair)
                    if not _ag_allowed:
                        logger.info(
                            "Phase 78: Skipping %s — accuracy gate blocked (acc=%.2f, %s)",
                            pair, _ag_accuracy or 0.0, _ag_reason,
                        )
                        return PairAnalysis(
                            pair=pair,
                            direction="HOLD",
                            confidence=0.0,
                            error=f"Accuracy gate: {_ag_reason}",
                            kill_reason="accuracy_gate",
                        )
                except Exception as _ag_err:
                    logger.debug(f"Phase 78: Accuracy gate check error for {pair}: {_ag_err}")

            # Check session timing
            session_status = self.config.get_trading_session_status()
            if not session_status["session_allowed"]:
                return PairAnalysis(
                    pair=pair,
                    direction="HOLD",
                    confidence=0.0,
                    error=str(session_status.get("message") or "Trading currently blocked"),
                )

            # US-016: Check roll calendar for FUTURES instruments
            if self._roll_calendar is not None:
                try:
                    # Get instrument from registry to check asset class
                    from src.brokers.registry import get_registry
                    registry = get_registry()
                    instrument = registry.get_optional(pair)

                    # Only apply roll calendar checks to FUTURES (FX skips entirely)
                    if instrument is not None and instrument.asset_class == "FUTURES":
                        # Get active contract month for this symbol
                        active_contract = self._roll_calendar.get_active_contract(pair)

                        # Check if within roll window
                        if self._roll_calendar.should_roll(pair):
                            logger.warning(
                                f"US-016: {pair} within roll window (active: {active_contract})"
                            )

                        # Check if contract has expired
                        days_until_roll = self._roll_calendar.days_until_roll(pair)
                        if days_until_roll <= 0:
                            logger.warning(
                                f"US-016: {pair} contract expired (active: {active_contract}, "
                                f"days_until_roll: {days_until_roll}) — skipping scan"
                            )
                            return PairAnalysis(
                                pair=pair,
                                direction="HOLD",
                                confidence=0.0,
                                error=f"Contract expired (active: {active_contract})",
                            )

                        # Update instrument contract_month to current active contract
                        # Note: Instrument is frozen, so we track this separately or log intent
                        logger.debug(
                            f"US-016: {pair} active contract: {active_contract}, "
                            f"days_until_roll: {days_until_roll}"
                        )
                except ValueError as vc_err:
                    # Symbol not in roll calendar (e.g., not a supported futures contract)
                    logger.debug(f"US-016: {pair} not in roll calendar: {vc_err}")
                except Exception as rc_err:
                    logger.debug(f"US-016: Roll calendar check error for {pair}: {rc_err}")

            # Fetch data
            df_raw = self._fetch_pair_data(pair, self.config.lookback_candles + 200)
            if df_raw is None:
                return PairAnalysis(
                    pair=pair,
                    direction="HOLD",
                    confidence=0.0,
                    error="No data (OANDA offline, no CSV)",
                )

            # Check incremental cache — but only skip full re-scan if agents
            # are disabled.  When agents are enabled, we still need to run them
            # on every cycle so trend/MTF/conflict checks stay fresh.
            current_price = df_raw["close"].iloc[-1]
            if self.config.incremental_enabled and not getattr(self.config, "enable_sub_inference_agents", True):
                if not self._check_price_changed(pair, current_price, self.config.price_change_threshold):
                    cached = self._get_cached_result(pair, self._build_cache_context(pair))
                    if cached is not None:
                        return cached

            # Compute features
            df_feat = self._compute_features(df_raw)
            self._raw_snapshots[pair] = df_raw.tail(self.config.lookback_candles + 50).copy()
            self._feature_snapshots[pair] = df_feat.tail(self.config.lookback_candles + 50).copy()

            # Store returns for correlation analysis
            if "close" in df_feat.columns:
                self._pair_returns[pair] = df_feat["close"].pct_change().dropna()

            # Check minimum volatility
            # compute_normalized_features produces atr_pct_14, convert to absolute ATR
            pip_value = self.config.pip_values.get(pair, 0.0001)
            if "atr" in df_feat.columns:
                atr = df_feat["atr"].iloc[-1]
            elif "atr_pct_14" in df_feat.columns and "close" in df_raw.columns:
                # Convert percentage ATR back to absolute
                atr = df_feat["atr_pct_14"].iloc[-1] * df_raw["close"].iloc[-1]
            else:
                atr = 0.0
            atr_pips = atr / pip_value if pip_value > 0 else 0

            # US-074: Microstructure regime detection — use high-low range as spread proxy
            _micro_signals = None
            if self._microstructure_detector is not None:
                try:
                    # Spread proxy: average (high - low) in pips over last 3 candles
                    _hl_spread = (df_raw["high"] - df_raw["low"]).iloc[-3:].mean()
                    _spread_pips = _hl_spread / pip_value if pip_value > 0 else 0.0
                    # Volume imbalance proxy: sign of close-open normalized by range
                    _range = df_raw["high"].iloc[-1] - df_raw["low"].iloc[-1]
                    _body = df_raw["close"].iloc[-1] - df_raw["open"].iloc[-1]
                    _imbalance = (_body / _range) if _range > 1e-10 else 0.0
                    _imbalance = max(-1.0, min(1.0, _imbalance))
                    _micro_signals = self._microstructure_detector.update(
                        current_spread_pips=float(_spread_pips),
                        bid_ask_imbalance=float(_imbalance),
                        volatility_regime=str(
                            getattr(self._regime_tracker, "current_regime", "NORMAL")
                            if self._regime_tracker else "NORMAL"
                        ).upper(),
                    )
                except Exception as _micro_err:
                    logger.debug(f"{pair}: Microstructure update error: {_micro_err}")

            # Phase 44 (US-278): Run regime detection before inference
            _regime_result = self._detect_regime(df_raw, pair)

            # Run inference
            direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime, _inference_ok, gate_details = (
                self._run_inference(df_raw, df_feat, pair)
            )

            # Phase 44 (US-278): Override volatility_regime with BOCPD+Hurst+ADX if available
            if _regime_result is not None:
                volatility_regime = _regime_result.regime_index
                gate_details = dict(gate_details or {})
                gate_details["regime_detector"] = {
                    "regime_name": _regime_result.regime_name,
                    "regime_index": _regime_result.regime_index,
                    "confidence": round(_regime_result.confidence, 3),
                    "changepoint_probability": round(_regime_result.changepoint_probability, 3),
                    "hurst": round(_regime_result.hurst_exponent, 3),
                    "adx": round(_regime_result.adx_value, 1),
                    "strategy_hint": _regime_result.strategy_hint,
                    "risk_multiplier": round(_regime_result.risk_multiplier, 2),
                }

            # Scanner role: surface opportunities even when hard execution gates fail.
            # If the architecture returns HOLD, attach a lightweight technical bias.
            if direction == "HOLD":
                tech_direction, tech_conf = self._infer_from_technicals(df_feat)
                if tech_direction is not None and tech_conf is not None:
                    direction = tech_direction
                    confidence = max(float(confidence), float(tech_conf) * 0.85)
                    gate_details = dict(gate_details or {})
                    gate_details.setdefault("opportunity_bias", "technical_fallback")

            # Phase 82 (US-P82-002): PairModelSelector — log active model variant for this pair
            if self._pair_model_selector is not None:
                try:
                    _active_model = self._pair_model_selector.get_active_model(pair)
                    logger.debug(
                        "Phase 82 (US-P82-002): %s active model variant: %s",
                        pair, _active_model,
                    )
                except Exception as _pms_err:
                    logger.debug("Phase 82: PairModelSelector get_active_model error for %s: %s", pair, _pms_err)

            # Calculate metrics
            metrics = self._calculate_metrics(df_feat, pair)

            # Dynamic ATR-based SL/TP (replaces fixed 15/30 defaults)
            # Per-pair adaptive multipliers (US-006) override global config
            _pair_sl_mult = self.config.atr_sl_multiplier
            _pair_tp_mult = self.config.atr_tp_multiplier
            try:
                import json as _json
                from pathlib import Path as _Path
                _pair_cfg_path = _Path("trained_data/models/pair_sl_tp_config.json")
                if _pair_cfg_path.exists():
                    _pair_cfg = _json.loads(_pair_cfg_path.read_text())
                    if pair in _pair_cfg:
                        _pair_sl_mult = _pair_cfg[pair].get("atr_sl_multiplier", _pair_sl_mult)
                        _pair_tp_mult = _pair_cfg[pair].get("atr_tp_multiplier", _pair_tp_mult)
            except Exception:
                pass

            if atr_pips > 0:
                sl_pips = max(
                    self.config.min_sl_pips,
                    min(atr_pips * _pair_sl_mult, self.config.max_sl_pips),
                )
                tp_pips = max(
                    self.config.min_tp_pips,
                    min(atr_pips * _pair_tp_mult, self.config.max_tp_pips),
                )
                # High-confidence TP bonus
                if confidence >= self.config.high_prob_threshold:
                    tp_pips = min(tp_pips + self.config.high_prob_tp_bonus, self.config.max_tp_pips + self.config.high_prob_tp_bonus)
            else:
                sl_pips = self.config.sl_pips
                tp_pips = self.config.tp_pips

            # CausalSignalFilter: penalize confidence if signal contradicts causal structure
            if self._causal_filter is not None and direction != "HOLD":
                try:
                    # Build lightweight feature dict from metrics already computed
                    _causal_features: Dict[str, float] = {
                        "trend_strength": metrics.get("trend_strength", 0.5),
                        "volatility_percentile": metrics.get("volatility_percentile", 0.5),
                        "entry_score": metrics.get("entry_score", 0.5),
                        "atr_pips": min(1.0, atr_pips / 50.0),
                    }
                    if "rsi" in df_feat.columns:
                        _causal_features["rsi"] = float(df_feat["rsi"].iloc[-1])
                    if "momentum" in df_feat.columns:
                        _causal_features["momentum"] = float(df_feat["momentum"].iloc[-1])

                    # Derive regime locally (regime_name not yet resolved at this point)
                    _causal_regime = "NORMAL"
                    if isinstance(volatility_regime, int) and 0 <= volatility_regime <= 3:
                        _causal_regime = ["LOW", "NORMAL", "HIGH", "EXTREME"][volatility_regime]
                    elif isinstance(volatility_regime, str) and volatility_regime in {"LOW", "NORMAL", "HIGH", "EXTREME"}:
                        _causal_regime = volatility_regime

                    _causal_score = self._causal_filter.filter_signal(
                        direction=direction,
                        features=_causal_features,
                        regime=_causal_regime,
                    )
                    # Apply confidence penalty for low causal consistency
                    if _causal_score < 0.5:
                        _causal_penalty = (0.5 - _causal_score) * 0.15  # Max ~7.5% penalty
                        _old_conf_causal = confidence
                        confidence = max(0.0, confidence - _causal_penalty)
                        logger.info(
                            f"{pair}: Causal filter [{_causal_regime}] "
                            f"consistency={_causal_score:.2f} — "
                            f"conf {_old_conf_causal:.3f}->{confidence:.3f} "
                            f"(penalty={_causal_penalty:.4f})"
                        )
                    elif _causal_score < 1.0:
                        logger.debug(
                            f"{pair}: Causal filter consistency={_causal_score:.2f} (no penalty)"
                        )
                    # Feedback hook: log causal filter output to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _obs.log_observation(
                                pair=pair,
                                category="causal_filter",
                                description=f"Causal consistency={_causal_score:.2f} regime={_causal_regime}",
                                metadata={
                                    "causal_score": round(_causal_score, 4),
                                    "regime": _causal_regime,
                                    "penalty_applied": _causal_score < 0.5,
                                    "direction": direction,
                                },
                            )
                    except Exception:
                        pass
                except Exception as _causal_err:
                    logger.debug(f"{pair}: Causal filter skipped: {_causal_err}")

            # US-085: Ensemble disagreement as meta-signal
            if self._ensemble_disagreement is not None and direction != "HOLD":
                try:
                    _dis_result = self._ensemble_disagreement.compute_disagreement(
                        tcn_confidence=tcn_conf,
                        ridge_confidence=ridge_conf,
                        tcn_direction=direction,
                        ridge_direction=direction,
                    )
                    if _dis_result.confidence_multiplier < 1.0:
                        _old_conf_dis = confidence
                        confidence = confidence * _dis_result.confidence_multiplier
                        logger.info(
                            f"{pair}: Ensemble disagreement mult={_dis_result.confidence_multiplier:.2f} "
                            f"conf {_old_conf_dis:.3f}→{confidence:.3f} "
                            f"(var={_dis_result.prediction_variance:.4f}, "
                            f"regime_shift={_dis_result.is_regime_shift_signal})"
                        )
                    # Feedback hook: log ensemble disagreement to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _obs.log_observation(
                                pair=pair,
                                category="ensemble_disagreement",
                                description=f"Disagreement mult={_dis_result.confidence_multiplier:.2f} var={_dis_result.prediction_variance:.4f}",
                                metadata={
                                    "confidence_multiplier": round(_dis_result.confidence_multiplier, 4),
                                    "prediction_variance": round(_dis_result.prediction_variance, 4),
                                    "is_regime_shift_signal": _dis_result.is_regime_shift_signal,
                                    "direction": direction,
                                },
                            )
                    except Exception:
                        pass
                except Exception as _dis_err:
                    logger.debug(f"{pair}: Ensemble disagreement skipped: {_dis_err}")

            # US-087: Cross-pair lead-lag signal boost
            if self._lead_lag_detector is not None and direction != "HOLD":
                try:
                    # Record this pair as a leader so lagging pairs can boost later
                    self._lead_lag_detector.get_lagging_signals(
                        leader_pair=pair,
                        leader_direction=direction,
                        leader_confidence=confidence,
                    )
                    # If this pair is being *led* by another pair that already signaled,
                    # boost confidence slightly based on lead-lag corroboration
                    _leaders = (
                        self._lead_lag_detector.get_matrix().get_leaders_for(pair)
                        if self._lead_lag_detector.get_matrix() is not None
                        else []
                    )
                    if _leaders:
                        _best_leader = max(_leaders, key=lambda r: r.confidence)
                        if _best_leader.confidence > 0.3:
                            _lead_boost = min(0.05, _best_leader.confidence * 0.08)
                            _old_conf_ll = confidence
                            confidence = min(1.0, confidence + _lead_boost)
                            logger.info(
                                f"{pair}: Lead-lag boost from {_best_leader.leader} "
                                f"(lag={_best_leader.optimal_lag}, corr={_best_leader.correlation:.2f}) "
                                f"conf {_old_conf_ll:.3f}→{confidence:.3f}"
                            )
                    # Feedback hook: log lead-lag relationships to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _ll_meta = {
                                "direction": direction,
                                "leaders_found": len(_leaders) if _leaders else 0,
                            }
                            if _leaders:
                                _ll_meta["best_leader"] = _best_leader.leader
                                _ll_meta["best_lag"] = _best_leader.optimal_lag
                                _ll_meta["best_correlation"] = round(_best_leader.correlation, 4)
                                _ll_meta["best_confidence"] = round(_best_leader.confidence, 4)
                            _obs.log_observation(
                                pair=pair,
                                category="lead_lag_signal",
                                description=f"Lead-lag: {len(_leaders) if _leaders else 0} leaders detected",
                                metadata=_ll_meta,
                            )
                    except Exception:
                        pass
                except Exception as _ll_err:
                    logger.debug(f"{pair}: Lead-lag check skipped: {_ll_err}")

            # US-088: Attention-based feature weighting
            _attention_result = None
            if self._feature_attention is not None and direction != "HOLD":
                try:
                    _feat_vec = {
                        "trend_strength": metrics.get("trend_strength", 0.5),
                        "volatility_percentile": metrics.get("volatility_percentile", 0.5),
                        "entry_score": metrics.get("entry_score", 0.5),
                        "atr_pips": min(1.0, atr_pips / 50.0),  # Normalize
                        "rsi": metrics.get("rsi", 0.5),
                        "momentum": metrics.get("momentum", 0.5),
                    }
                    _attention_result = self._feature_attention.compute_attention(
                        feature_vector=_feat_vec,
                        regime=regime_name if regime_name != "UNKNOWN" else "NORMAL",
                    )
                    # Use attention-weighted entry_score to nudge confidence
                    if "entry_score" in _attention_result.weighted_features:
                        _weighted_entry = _attention_result.weighted_features["entry_score"]
                        _raw_entry = _feat_vec.get("entry_score", 0.5)
                        _attn_delta = (_weighted_entry - _raw_entry * (1.0 / len(_feat_vec))) * 0.1
                        if abs(_attn_delta) > 0.005:
                            _old_conf_attn = confidence
                            confidence = max(0.0, min(1.0, confidence + _attn_delta))
                            logger.debug(
                                f"{pair}: Attention [{regime_name}] dominant={_attention_result.dominant_feature} "
                                f"conf {_old_conf_attn:.3f}→{confidence:.3f} (delta={_attn_delta:+.4f})"
                            )
                    # Feedback hook: log feature attention regime priorities to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _obs.log_observation(
                                pair=pair,
                                category="feature_attention",
                                description=f"Attention dominant={_attention_result.dominant_feature} regime={regime_name}",
                                metadata={
                                    "dominant_feature": _attention_result.dominant_feature,
                                    "regime": regime_name,
                                    "attention_weights": {
                                        k: round(v, 4) for k, v in _attention_result.weighted_features.items()
                                    },
                                    "direction": direction,
                                },
                            )
                    except Exception:
                        pass
                except Exception as _attn_err:
                    logger.debug(f"{pair}: Feature attention skipped: {_attn_err}")

            # US-081: Multi-horizon signal fusion — register primary TF and apply grade
            _fusion_grade = "B"  # Default: no adjustment
            if self._multi_horizon_fusion is not None and direction != "HOLD":
                try:
                    self._multi_horizon_fusion.clear_signals()
                    # Register primary timeframe signal (M5 — the default scanning TF)
                    self._multi_horizon_fusion.add_signal(
                        timeframe="M5",
                        direction=direction,
                        confidence=confidence,
                        features={"atr_pips": atr_pips, "trend_strength": metrics.get("trend_strength", 0.5)},
                    )
                    # Derive H1/H4 signals from trend strength and momentum
                    _ts = metrics.get("trend_strength", 0.5)
                    _h1_dir = direction if _ts > 0.45 else "HOLD"
                    self._multi_horizon_fusion.add_signal("H1", _h1_dir, min(confidence, _ts), {})
                    _h4_dir = direction if _ts > 0.55 else ("HOLD" if _ts > 0.35 else (
                        "SHORT" if direction == "LONG" else "LONG"
                    ))
                    self._multi_horizon_fusion.add_signal("H4", _h4_dir, min(confidence * 0.9, _ts), {})
                    _fusion = self._multi_horizon_fusion.fuse()
                    _fusion_grade = _fusion.agreement_grade
                    if _fusion.confidence_adjustment != 0:
                        _old_conf_f = confidence
                        confidence = max(0.0, min(1.0, confidence + _fusion.confidence_adjustment))
                        logger.info(
                            f"{pair}: Fusion grade={_fusion_grade} "
                            f"conf {_old_conf_f:.3f}→{confidence:.3f} "
                            f"(adj={_fusion.confidence_adjustment:+.2f})"
                        )
                    # Feedback hook: log multi-horizon fusion to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _obs.log_observation(
                                pair=pair,
                                category="multi_horizon_fusion",
                                description=f"Fusion grade={_fusion_grade} adj={_fusion.confidence_adjustment:+.3f}",
                                metadata={
                                    "agreement_grade": _fusion_grade,
                                    "confidence_adjustment": round(_fusion.confidence_adjustment, 4),
                                    "direction": direction,
                                },
                            )
                    except Exception:
                        pass
                except Exception as _fuse_err:
                    logger.debug(f"{pair}: Multi-horizon fusion skipped: {_fuse_err}")

            # US-080: Regime-conditioned dynamic SL/TP optimization
            if (
                getattr(self.config, "enable_dynamic_sl_tp", False)
                and direction != "HOLD"
                and sl_pips > 0
                and tp_pips > 0
            ):
                try:
                    from src.scanner.automation.dynamic_sl_tp import DynamicSLTPOptimizer
                    _sltp_opt = DynamicSLTPOptimizer(
                        aggressiveness=getattr(self.config, "sl_tp_aggressiveness", 1.0),
                        min_sl_pips=self.config.min_sl_pips,
                        max_sl_pips=self.config.max_sl_pips,
                        min_tp_pips=self.config.min_tp_pips,
                        max_tp_pips=self.config.max_tp_pips,
                    )
                    _sltp_result = _sltp_opt.optimize(
                        base_sl_pips=sl_pips,
                        base_tp_pips=tp_pips,
                        regime=regime_name if regime_name != "UNKNOWN" else "NORMAL",
                        trend_strength=metrics.get("trend_strength", 0.5),
                        confidence=confidence,
                        atr_pips=atr_pips,
                    )
                    if _sltp_result.rules_applied > 0:
                        _old_sl, _old_tp = sl_pips, tp_pips
                        sl_pips = _sltp_result.optimized_sl_pips
                        tp_pips = _sltp_result.optimized_tp_pips
                        logger.info(
                            f"{pair}: Dynamic SL/TP [{regime_name}] "
                            f"SL {_old_sl:.1f}→{sl_pips:.1f}, TP {_old_tp:.1f}→{tp_pips:.1f} "
                            f"({_sltp_result.rules_applied} rules)"
                        )
                except Exception as _sltp_err:
                    logger.debug(f"{pair}: Dynamic SL/TP skipped: {_sltp_err}")

            # US-074: Apply microstructure confidence lift
            if (
                _micro_signals is not None
                and _micro_signals.confidence_lift > 0
                and getattr(self.config, "microstructure_confidence_boost", True)
            ):
                _old_conf = confidence
                confidence = min(1.0, confidence + _micro_signals.confidence_lift)
                logger.info(
                    f"{pair}: Microstructure lift +{_micro_signals.confidence_lift:.3f} "
                    f"({_old_conf:.3f} → {confidence:.3f}), "
                    f"micro_regime={_micro_signals.predicted_regime}, "
                    f"score={_micro_signals.microstructure_score:.3f}"
                )
                # Feedback hook: log microstructure regime detection to observation pipeline
                try:
                    _obs = self._get_feedback_observer()
                    if _obs is not None:
                        _obs.log_observation(
                            pair=pair,
                            category="microstructure_regime",
                            description=f"Micro regime={_micro_signals.predicted_regime} score={_micro_signals.microstructure_score:.3f}",
                            metadata={
                                "predicted_regime": _micro_signals.predicted_regime,
                                "microstructure_score": round(_micro_signals.microstructure_score, 4),
                                "confidence_lift": round(_micro_signals.confidence_lift, 4),
                                "direction": direction,
                            },
                        )
                except Exception:
                    pass

            # US-079: Adaptive confidence calibration (Platt scaling)
            if self._confidence_calibrator is not None and direction != "HOLD":
                try:
                    _cal_result = self._confidence_calibrator.calibrate(confidence)
                    if _cal_result.is_calibrated:
                        _old_conf_cal = confidence
                        confidence = _cal_result.calibrated_score
                        logger.info(
                            f"{pair}: Platt calibration {_old_conf_cal:.3f} → "
                            f"{confidence:.3f} ({_cal_result.reason})"
                        )
                    # Check if refit needed
                    if self._confidence_calibrator.should_refit():
                        self._confidence_calibrator.fit_from_journal()
                    # Feedback hook: log confidence calibration to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _obs.log_observation(
                                pair=pair,
                                category="confidence_calibration",
                                description=f"Platt calibration is_calibrated={_cal_result.is_calibrated} reason={_cal_result.reason}",
                                metadata={
                                    "is_calibrated": _cal_result.is_calibrated,
                                    "calibrated_score": round(_cal_result.calibrated_score, 4),
                                    "reason": _cal_result.reason,
                                    "direction": direction,
                                },
                            )
                    except Exception:
                        pass
                except Exception as _cal_err:
                    logger.debug(f"{pair}: Calibration skipped: {_cal_err}")

            # US-083: Concept drift detection — update and apply penalty
            if self._concept_drift is not None:
                try:
                    # Phase 57 (US-354): Skip concept drift penalty during stall recovery
                    _drift_recovery_suspended = (
                        self._stall_recovery is not None
                        and self._stall_recovery.penalties_suspended()
                    )
                    # Phase 57 (US-355): Use pre-penalty confidence for drift proxies (break circular ref)
                    # Phase 79: confidence is already 0-1 scale — no division needed
                    _pre_penalty_conf_norm = confidence
                    if self._drift_proxy_guard is not None:
                        _drift_proxies = self._drift_proxy_guard.compute_proxies(
                            pre_penalty_confidence=_pre_penalty_conf_norm,
                            post_penalty_confidence=_pre_penalty_conf_norm,  # same at this point
                            pair=pair,
                        )
                        _pred_err = _drift_proxies.prediction_error
                        _agent_dis = _drift_proxies.agent_disagreement
                    else:
                        # Feed observation: prediction error proxy, agent disagreement proxy
                        _pred_err = abs(confidence - 0.5)  # Distance from uncertainty
                        _agent_dis = 1.0 - confidence  # Proxy for disagreement
                    _micro_score = (
                        _micro_signals.microstructure_score
                        if _micro_signals is not None else 0.0
                    )
                    self._concept_drift.update(
                        prediction_error=_pred_err,
                        agent_disagreement=_agent_dis,
                        micro_anomaly_score=_micro_score,
                    )
                    _drift = self._concept_drift.detect()
                    if _drift.drift_detected and _drift.confidence_penalty < 1.0 and not _drift_recovery_suspended:
                        _old_conf_d = confidence
                        # Phase 57 (US-352): Apply ceiling before multiplier
                        _safe_drift_mult = _drift.confidence_penalty
                        if self._penalty_ceiling is not None:
                            _ceil_drift = self._penalty_ceiling.check_multiplier(
                                confidence, _drift.confidence_penalty, pair=pair, source="concept_drift"
                            )
                            _safe_drift_mult = _ceil_drift.applied_penalty
                        confidence = confidence * _safe_drift_mult
                        logger.warning(
                            f"{pair}: Concept drift [{_drift.severity}] "
                            f"conf {_old_conf_d:.3f}→{confidence:.3f} "
                            f"(penalty={_drift.confidence_penalty:.2f}, "
                            f"streams={_drift.drifting_streams})"
                        )
                    # Feedback hook: log concept drift detection to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _obs.log_observation(
                                pair=pair,
                                category="concept_drift",
                                description=f"Drift detected={_drift.drift_detected} severity={_drift.severity}",
                                metadata={
                                    "drift_detected": _drift.drift_detected,
                                    "severity": _drift.severity,
                                    "confidence_penalty": round(_drift.confidence_penalty, 4),
                                    "drifting_streams": _drift.drifting_streams,
                                },
                            )
                    except Exception:
                        pass
                except Exception as _drift_err:
                    logger.debug(f"{pair}: Drift detection skipped: {_drift_err}")

            # US-094: Trade explainability — penalize confidence for inconsistent explanations
            if self._trade_explainer is not None and direction != "HOLD":
                try:
                    # Build feature dict from available scan data
                    _explain_features: Dict[str, float] = {}
                    if isinstance(features, dict):
                        _explain_features = {k: float(v) for k, v in features.items()
                                             if isinstance(v, (int, float))}
                    elif hasattr(features, '__iter__') and len(features) > 0:
                        _feature_names = ["momentum", "trend_strength", "rsi", "volatility",
                                          "spread", "volume", "mean_reversion",
                                          "support_distance", "resistance_distance", "session_strength"]
                        for i, name in enumerate(_feature_names):
                            if i < len(features):
                                _explain_features[name] = float(features[i])

                    if _explain_features:
                        _explain_result = self._trade_explainer.explain_trade(
                            features=_explain_features,
                            confidence=confidence,
                            direction=direction,
                        )
                        if _explain_result.confidence_penalty > 0:
                            _old_conf_e = confidence
                            confidence = max(0.0, confidence - _explain_result.confidence_penalty)
                            logger.info(
                                f"{pair}: Explainability penalty: "
                                f"conf {_old_conf_e:.3f}→{confidence:.3f} "
                                f"(score={_explain_result.explainability_score:.2f}, "
                                f"contradictions={len(_explain_result.contradictions)})"
                            )
                        # Feedback hook: log trade explainer output to observation pipeline
                        try:
                            _obs = self._get_feedback_observer()
                            if _obs is not None:
                                _obs.log_observation(
                                    pair=pair,
                                    category="trade_explainer",
                                    description=f"Explainability score={_explain_result.explainability_score:.2f} penalty={_explain_result.confidence_penalty:.4f}",
                                    metadata={
                                        "explainability_score": round(_explain_result.explainability_score, 4),
                                        "confidence_penalty": round(_explain_result.confidence_penalty, 4),
                                        "contradictions_count": len(_explain_result.contradictions),
                                        "direction": direction,
                                    },
                                )
                        except Exception:
                            pass
                except Exception as _explain_err:
                    logger.debug(f"{pair}: Explainability skipped: {_explain_err}")

            # US-089: Adversarial robustness — penalize confidence when ensemble robustness is low
            if self._adversarial_trainer is not None and direction != "HOLD":
                try:
                    _adv_score = self._adversarial_trainer.get_latest_score()
                    if _adv_score is not None and _adv_score.robustness_ratio > 0:
                        # Penalize confidence when robustness ratio drops below 0.7
                        # (model struggles with adversarial scenarios)
                        if _adv_score.robustness_ratio < 0.7:
                            _adv_penalty = (0.7 - _adv_score.robustness_ratio) * 0.15  # Max ~10.5% penalty
                            _old_conf_adv = confidence
                            confidence = max(0.0, confidence - _adv_penalty)
                            logger.info(
                                f"{pair}: Adversarial robustness penalty: "
                                f"conf {_old_conf_adv:.3f}->{confidence:.3f} "
                                f"(ratio={_adv_score.robustness_ratio:.3f}, "
                                f"clean={_adv_score.clean_accuracy:.3f}, "
                                f"adv={_adv_score.adversarial_accuracy:.3f})"
                            )
                        else:
                            logger.debug(
                                f"{pair}: Adversarial robustness OK "
                                f"(ratio={_adv_score.robustness_ratio:.3f})"
                            )
                    # Feedback hook: log adversarial robustness to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _obs.log_observation(
                                pair=pair,
                                category="adversarial_robustness",
                                description=(
                                    f"Robustness ratio={_adv_score.robustness_ratio:.3f}"
                                    if _adv_score is not None else "No score available"
                                ),
                                metadata={
                                    "robustness_ratio": round(_adv_score.robustness_ratio, 4) if _adv_score else 0.0,
                                    "clean_accuracy": round(_adv_score.clean_accuracy, 4) if _adv_score else 0.0,
                                    "adversarial_accuracy": round(_adv_score.adversarial_accuracy, 4) if _adv_score else 0.0,
                                    "scenario_scores": _adv_score.scenario_scores if _adv_score else {},
                                    "direction": direction,
                                },
                            )
                    except Exception:
                        pass
                except Exception as _adv_err:
                    logger.debug(f"{pair}: Adversarial robustness check skipped: {_adv_err}")

            # US-084: Portfolio-level Kelly criterion sizing
            risk_pct = self.config.risk_per_trade_pct
            if self._kelly_optimizer is not None and direction != "HOLD" and sl_pips > 0:
                try:
                    _kelly_result = self._kelly_optimizer.portfolio_optimal_size(
                        pair=pair,
                        direction=direction,
                        confidence=confidence,
                        sl_pips=sl_pips,
                        open_positions=[],  # Populated by execution manager in live mode
                        nav=10000.0,  # Default; overridden in live trading
                    )
                    if _kelly_result.recommended_risk_pct > 0:
                        # Floor Kelly at 50% of config risk — never let Kelly shrink below half the configured risk
                        risk_pct = max(_kelly_result.recommended_risk_pct, self.config.risk_per_trade_pct * 0.5)
                        logger.info(
                            f"{pair}: Kelly sizing → risk={risk_pct:.4f} "
                            f"(half_k={_kelly_result.half_kelly_fraction:.4f}, "
                            f"corr={_kelly_result.correlation_reduction:.2f})"
                        )
                    # Feedback hook: log Kelly portfolio sizing to observation pipeline
                    try:
                        _obs = self._get_feedback_observer()
                        if _obs is not None:
                            _obs.log_observation(
                                pair=pair,
                                category="kelly_portfolio",
                                description=f"Kelly risk={_kelly_result.recommended_risk_pct:.4f} half_k={_kelly_result.half_kelly_fraction:.4f}",
                                metadata={
                                    "recommended_risk_pct": round(_kelly_result.recommended_risk_pct, 6),
                                    "half_kelly_fraction": round(_kelly_result.half_kelly_fraction, 6),
                                    "correlation_reduction": round(_kelly_result.correlation_reduction, 4),
                                    "direction": direction,
                                },
                            )
                    except Exception:
                        pass
                except Exception as _kelly_err:
                    logger.debug(f"{pair}: Kelly sizing fallback: {_kelly_err}")
                    # Fallback to simple sizing
                    if confidence > 0.65:
                        risk_pct = min(risk_pct * 1.25, self.config.risk_per_trade_pct)
            elif confidence > 0.65:
                # Simple fallback: Higher confidence = slightly larger position
                risk_pct = min(risk_pct * 1.25, self.config.risk_per_trade_pct)

            # US-092: Market impact — reduce position size if estimated slippage is too high
            if self._market_impact is not None and direction != "HOLD" and risk_pct > 0:
                try:
                    _spread = getattr(self, '_last_spread_pips', 1.0)
                    # Estimate lot size from risk_pct (approximate for impact calc)
                    _approx_lots = risk_pct * 10000.0 / max(sl_pips, 1.0)
                    _impact = self._market_impact.adjust_position_size(
                        base_size_lots=_approx_lots,
                        spread_pips=_spread,
                        regime=regime_name,
                    )
                    if _impact.exceeds_threshold:
                        _reduction = _impact.adjusted_size_lots / max(_approx_lots, 0.01)
                        risk_pct = risk_pct * _reduction
                        logger.info(
                            f"{pair}: Market impact [{regime_name}]: "
                            f"slippage={_impact.total_impact_pips:.2f} pips, "
                            f"position reduced by {_impact.size_reduction_pct:.0f}%"
                        )
                except Exception as _impact_err:
                    logger.debug(f"{pair}: Market impact skipped: {_impact_err}")

            # US-098: Entropy-based position sizing — scale by agent consensus entropy
            if self._entropy_sizer is not None and direction != "HOLD" and risk_pct > 0:
                try:
                    # Collect agent votes from verdicts if available
                    _agent_votes: Dict[str, Any] = {}
                    if hasattr(self, '_last_agent_verdicts') and self._last_agent_verdicts:
                        _agent_votes = self._last_agent_verdicts
                    elif isinstance(agent_result, dict) and "verdicts" in agent_result:
                        for _v in agent_result["verdicts"]:
                            if isinstance(_v, dict) and "agent" in _v:
                                _agent_votes[_v["agent"]] = _v.get("vote", 0)

                    if _agent_votes:
                        _entropy = self._entropy_sizer.size_from_votes(
                            agent_votes=_agent_votes,
                            regime=regime_name,
                        )
                        if abs(_entropy.size_multiplier - 1.0) > 0.01:
                            _old_risk = risk_pct
                            risk_pct *= _entropy.size_multiplier
                            logger.info(
                                f"{pair}: Entropy sizing [{regime_name}]: "
                                f"H={_entropy.normalized_entropy:.3f}, "
                                f"mult={_entropy.size_multiplier:.3f}, "
                                f"risk {_old_risk:.4f}→{risk_pct:.4f}"
                            )
                except Exception as _entropy_err:
                    logger.debug(f"{pair}: Entropy sizing skipped: {_entropy_err}")

            # Build error message if volatility regime blocked the trade or inference failed
            error_msg = None
            if not _inference_ok and direction == "HOLD" and confidence < 0.01:
                error_msg = "No models loaded"
            regime_name = "UNKNOWN"
            if volatility_regime is not None:
                regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
                if isinstance(volatility_regime, int) and 0 <= volatility_regime <= 3:
                    regime_name = regime_names[volatility_regime]
                elif isinstance(volatility_regime, str) and volatility_regime in {"LOW", "NORMAL", "HIGH", "EXTREME"}:
                    regime_name = volatility_regime

            # Phase 21 (US-128): ATR-based regime fallback when TCN unavailable
            if regime_name == "UNKNOWN" and atr_pips > 0:
                try:
                    # Compute ATR percentile from recent candles for fallback regime
                    if "close" in df_raw.columns and "high" in df_raw.columns and len(df_raw) >= 50:
                        _atr_series = df_raw["high"].iloc[-100:] - df_raw["low"].iloc[-100:]
                        _current_atr = _atr_series.iloc[-1] if len(_atr_series) > 0 else 0
                        _atr_pctile = float((_atr_series < _current_atr).mean())
                        if _atr_pctile < 0.25:
                            regime_name = "LOW"
                        elif _atr_pctile < 0.50:
                            regime_name = "NORMAL"
                        elif _atr_pctile < 0.75:
                            regime_name = "HIGH"
                        else:
                            regime_name = "EXTREME"
                        logger.info(
                            f"{pair}: Regime fallback ATR-percentile={_atr_pctile:.2f} → {regime_name}"
                        )
                        # Log to observation_log
                        try:
                            from src.scanner.automation.observation_log import ObservationLog
                            ObservationLog().log_observation(
                                pair=pair, category="regime_fallback",
                                description=f"TCN unavailable, ATR-percentile={_atr_pctile:.2f} → {regime_name}",
                                metadata={"atr_pctile": _atr_pctile, "regime": regime_name},
                            )
                        except Exception as e:
                            logger.debug("%s: regime_fallback observation log skipped: %s", pair, e)
                except Exception as _rf_err:
                    logger.debug(f"{pair}: Regime fallback error: {_rf_err}")

            # Phase 23 (US-140): If regime is STILL UNKNOWN after ATR fallback, use config default
            if regime_name == "UNKNOWN":
                _default_regime = getattr(self.config, "volatility_regime_default", "NORMAL")
                regime_name = _default_regime
                logger.info(f"{pair}: Regime defaulted to {regime_name} (ATR fallback insufficient)")
                if self._observation_log is not None:
                    try:
                        self._observation_log.log_observation(
                            category="regime_default",
                            pair=pair,
                            data={"default_regime": regime_name, "reason": "atr_fallback_insufficient"},
                        )
                    except Exception as e:
                        logger.debug("%s: regime_default observation log skipped: %s", pair, e)

            # RegimeBroadcaster: broadcast transition if regime changed for this pair
            if self._regime_broadcaster is not None:
                try:
                    _prev_regime_for_pair = self._pair_last_regime.get(pair, "NORMAL")
                    if regime_name != _prev_regime_for_pair:
                        _rb_confidence = (
                            _regime_result.confidence if _regime_result is not None else 0.5
                        )
                        self._regime_broadcaster.broadcast_transition(
                            old_regime=_prev_regime_for_pair,
                            new_regime=regime_name,
                            confidence=_rb_confidence,
                            source="scan_pipeline",
                            metadata={"pair": pair},
                        )
                    self._pair_last_regime[pair] = regime_name
                except Exception as _rb_err:
                    logger.debug(f"{pair}: RegimeBroadcaster error: {_rb_err}")

            # Phase 81 (US-P81-003): RegimeConfidenceScorer — scale risk_pct by regime certainty
            # Runs after regime_name is fully finalized (ATR fallback + default applied above).
            if self._regime_confidence_scorer_scan is not None and direction != "HOLD" and risk_pct > 0:
                try:
                    _adx_rcs = 25.0
                    if "adx" in df_feat.columns:
                        _adx_rcs = float(df_feat["adx"].iloc[-1])
                    elif "adx_14" in df_feat.columns:
                        _adx_rcs = float(df_feat["adx_14"].iloc[-1])
                    _atr_pct_rcs = float(metrics.get("volatility_percentile", 0.5))
                    _trend_rcs = float(metrics.get("trend_strength", 0.5))
                    _rcs_result = self._regime_confidence_scorer_scan.score(
                        adx=_adx_rcs,
                        atr_percentile=_atr_pct_rcs,
                        trend_strength=_trend_rcs,
                        current_regime=regime_name,
                    )
                    if _rcs_result.sizing_factor < 1.0:
                        _old_risk_pct = risk_pct
                        risk_pct *= _rcs_result.sizing_factor
                        logger.info(
                            "Phase 81 (US-P81-003): %s regime_confidence tier=%s "
                            "sizing_factor=%.2f risk %.4f→%.4f (transition=%s)",
                            pair, getattr(_rcs_result, "tier", "?"), _rcs_result.sizing_factor,
                            _old_risk_pct, risk_pct, getattr(_rcs_result, "is_transition", False),
                        )
                    else:
                        logger.debug(
                            "Phase 81 (US-P81-003): %s regime_confidence tier=%s — full size",
                            pair, getattr(_rcs_result, "tier", "?"),
                        )
                except Exception as _rcs_err:
                    logger.debug("Phase 81 (US-P81-003): RegimeConfidenceScorer error: %s", _rcs_err)

            # Compute recent drawdown from close prices (lookback ~100 bars)
            _drawdown_val = 0.02  # Sensible default fallback
            if "close" in df_raw.columns and len(df_raw) >= 20:
                _close = df_raw["close"].iloc[-100:]
                _running_max = _close.cummax()
                _dd_series = (_running_max - _close) / _running_max.replace(0, np.nan)
                _dd_series = _dd_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                _drawdown_val = max(float(_dd_series.max()), 0.001)  # Floor at 0.1%

            # Keep scanner output aligned with the underlying inference architecture.
            _momentum_val = float(gate_details.get("momentum", confidence)) if gate_details else float(confidence)
            _momentum_ok = bool(gate_details.get("momentum_passed", False)) if gate_details else False
            _momentum_accel = bool(gate_details.get("momentum_acceleration", False)) if gate_details else False
            _confidence_ok = bool(gate_details.get("confidence_passed", False)) if gate_details else False
            _confidence_score = float(gate_details.get("confidence_score", ridge_conf)) if gate_details else float(ridge_conf)
            _risk_ok = bool(gate_details.get("risk_passed", False)) if gate_details else False
            _drawdown_model = gate_details.get("drawdown") if gate_details else None
            _drawdown_used = float(_drawdown_model) if _drawdown_model is not None else _drawdown_val
            _vol_gate_ok = bool(gate_details.get("volatility_gate_passed", False)) if gate_details else False
            _final_gates_passed = bool(gates_passed) and error_msg is None

            # Phase 90 US-403: Determine kill_reason for gate telemetry
            _disagree_val = float(gate_details.get("model_disagreement", 0.0)) if gate_details else 0.0
            _disagree_floor = float(getattr(self.config, "disagreement_hard_floor", 0.50))
            if _final_gates_passed and direction in {"LONG", "SHORT"} and error_msg is None:
                _kill_reason = None  # tradeable — no kill reason
            elif _disagree_val > _disagree_floor:
                _kill_reason = "disagreement_gate"
            elif not _confidence_ok and not _final_gates_passed:
                _kill_reason = "confidence_gate"
            elif not _momentum_ok and not _final_gates_passed:
                _kill_reason = "momentum_gate"
            elif not _risk_ok and not _final_gates_passed:
                _kill_reason = "drawdown_gate"
            elif error_msg is not None:
                _kill_reason = "other"
            else:
                _kill_reason = "other"

            result = PairAnalysis(
                pair=pair,
                direction=direction,
                confidence=confidence,
                tcn_confidence=tcn_conf,
                tcn_probability=float(gate_details.get("tcn_probability", 0.5)) if gate_details else 0.5,
                ridge_confidence=ridge_conf,
                momentum=_momentum_val,
                xgb_momentum=_momentum_val,
                momentum_acceleration=_momentum_accel,
                momentum_passed=_momentum_ok,
                confidence_score=_confidence_score,
                confidence_passed=_confidence_ok,
                drawdown=_drawdown_used,
                rf_drawdown=_drawdown_used,
                risk_passed=_risk_ok,
                volatility_gate_passed=_vol_gate_ok,
                gates_passed=_final_gates_passed,
                current_price=metrics["current_price"],
                atr=metrics["atr"],
                atr_pips=atr_pips,
                volatility_percentile=metrics["volatility_percentile"],
                trend_strength=metrics["trend_strength"],
                entry_score=metrics["entry_score"],
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                risk_pct=risk_pct,
                scan_time=datetime.now(timezone.utc),
                volatility_regime=regime_name,  # TCN volatility regime as string
                master_pair=str(gate_details.get("model_source_pair", pair)).upper().replace("/", "_")
                if gate_details else str(pair).upper().replace("/", "_"),
                error=error_msg,
                kill_reason=_kill_reason,
            )
            # Tier 6: Stash ensemble component scores for downstream feedback
            if gate_details:
                _scores_dict = gate_details.get("scores", {})
                if isinstance(_scores_dict, dict):
                    result._ensemble_scores = _scores_dict
                result._core_score = gate_details.get("core_score")
                result._final_score = gate_details.get("final_score")

            result = self._apply_specialist_agents(
                analysis=result,
                df_raw=df_raw,
                df_feat=df_feat,
                gate_details=gate_details,
            )

            # Phase 70: Record weighted_vote_score from specialist agents for ALL pairs
            if self._vote_recorder is not None and getattr(result, "weighted_vote_score", None) is not None:
                try:
                    self._vote_recorder.record(
                        pair=pair,
                        vote_score=float(result.weighted_vote_score),
                        agent_passed=bool(result.weighted_vote_score >= result.weighted_vote_threshold),
                        scan_id=str(getattr(self, "_scan_cycle", 0)),
                    )
                except Exception as _vsr_err:
                    logger.debug(f"Phase 70: _vote_recorder.record() in _scan_pair failed: {_vsr_err}")

            # Phase 61+63: Record momentum/risk scores in _scan_pair for non-ensemble inference paths
            if self._momentum_recorder is not None:
                try:
                    self._momentum_recorder.record(
                        pair=pair,
                        momentum_score=float(getattr(result, "xgb_momentum", 0.0)),
                        momentum_passed=bool(getattr(result, "momentum_passed", False)),
                        scan_id=str(getattr(self, "_scan_cycle", 0)),
                    )
                except Exception as _msr_sp_err:
                    logger.debug(f"Phase 61: momentum_recorder.record in _scan_pair failed: {_msr_sp_err}")

            if self._risk_recorder is not None:
                try:
                    _rf_dd = float(getattr(result, "rf_drawdown", 0.0))
                    _rf_dd_pct = _rf_dd / 10000.0 if _rf_dd > 1.0 else _rf_dd
                    self._risk_recorder.record(
                        pair=pair,
                        rf_drawdown_pct=_rf_dd_pct,
                        rf_streak_prob=0.0,  # Not available outside ensemble signal
                        risk_passed=bool(getattr(result, "risk_passed", False)),
                        max_drawdown_pct=float(self.config.max_drawdown_pct),
                        scan_id=str(getattr(self, "_scan_cycle", 0)),
                    )
                except Exception as _rrr_sp_err:
                    logger.debug(f"Phase 63: risk_recorder.record in _scan_pair failed: {_rrr_sp_err}")

            # Phase 75: Record heuristic disagreement for ALL pairs
            if self._disagreement_recorder is not None:
                try:
                    _disagree = float(getattr(result, "model_disagreement", 0.0))
                    _hard_floor = float(getattr(self.config, "disagreement_hard_floor", 0.50))
                    self._disagreement_recorder.record(
                        pair=pair,
                        disagreement=_disagree,
                        hard_blocked=bool(_disagree > _hard_floor),
                        scan_id=str(getattr(self, "_scan_cycle", 0)),
                    )
                except Exception as _dr_err:
                    logger.debug(f"Phase 75: _disagreement_recorder.record() failed: {_dr_err}")

            # Attention feedback: compute timeframe quality from agent votes and feed to temporal attention
            if self._attention_feedback is not None and direction != "HOLD":
                try:
                    _agent_votes_for_attn = {}
                    for _ar in (result.agent_reasons or []):
                        _aname = _ar.get("name", "")
                        if _aname:
                            _avote = "BUY" if _ar.get("passed", False) else "SELL"
                            if direction == "SHORT":
                                _avote = "SELL" if _ar.get("passed", False) else "BUY"
                            _agent_votes_for_attn[_aname] = _avote

                    # Build timeframe signals from available metrics
                    _ts = metrics.get("trend_strength", 0.5)
                    _mom = metrics.get("momentum", 0.5)
                    _dir_sign = 1.0 if direction == "LONG" else -1.0
                    _tf_signals = {
                        5: {"direction": _dir_sign * _ts, "strength": _ts},
                        60: {"direction": _dir_sign * _ts * 0.8, "strength": _ts * 0.9},
                        240: {"direction": _dir_sign * _mom * 0.7, "strength": _mom * 0.85},
                    }

                    _quality_scores = self._attention_feedback.compute_timeframe_quality(
                        agent_votes=_agent_votes_for_attn,
                        timeframe_signals=_tf_signals,
                    )
                    if _quality_scores:
                        self._attention_feedback.update_attention(
                            quality_scores=_quality_scores,
                            regime=regime_name if regime_name != "UNKNOWN" else "NORMAL",
                        )
                        logger.debug(
                            f"{pair}: Attention feedback [{regime_name}] "
                            f"quality={_quality_scores}"
                        )
                except Exception as _attn_fb_err:
                    logger.debug(f"{pair}: Attention feedback skipped: {_attn_fb_err}")

            # Phase 18: Model calibration — record predictions and check flocking
            if self._model_calibration is not None and direction != "HOLD":
                try:
                    # Record per-model predictions for calibration tracking
                    if tcn_conf > 0:
                        self._model_calibration.record_prediction("TCN", tcn_conf, won=False)
                    if ridge_conf > 0:
                        self._model_calibration.record_prediction("Ridge", ridge_conf, won=False)
                    # Check flocking (won field updated later via record_trade_outcome_phase18)
                    _model_preds = {}
                    if tcn_conf > 0:
                        _model_preds["TCN"] = tcn_conf
                    if ridge_conf > 0:
                        _model_preds["Ridge"] = ridge_conf
                    if len(_model_preds) >= 2:
                        _flock = self._model_calibration.detect_flocking(_model_preds)
                        if _flock.get("is_flocking"):
                            _flock_acc = _flock.get("historical_flock_accuracy")
                            if _flock_acc is not None and _flock_acc < 0.45:
                                _old_conf_flock = confidence
                                confidence = confidence * 0.90
                                result.confidence = confidence
                                logger.warning(
                                    f"{pair}: Flocking detected (agreement={_flock['agreement_ratio']:.2f}, "
                                    f"historical_accuracy={_flock_acc:.2f}) — "
                                    f"conf {_old_conf_flock:.3f}→{confidence:.3f}"
                                )
                            else:
                                logger.info(
                                    f"{pair}: Flocking detected (agreement={_flock['agreement_ratio']:.2f}) "
                                    f"— no penalty (accuracy={_flock_acc})"
                                )
                    # Phase 30 (US-184): Consume calibration ECE for dynamic confidence adj
                    _oc_applied = False
                    # Phase 57 (US-354): Skip overconfidence penalty during stall recovery
                    _oc_recovery_suspended = (
                        self._stall_recovery is not None
                        and self._stall_recovery.penalties_suspended()
                    )
                    for _model_name in ("TCN", "Ridge"):
                        _cal_report = self._model_calibration.get_calibration_report(_model_name)
                        _oc_ratio = _cal_report.get("overconfidence_ratio")
                        # Phase 57 (US-353): Calibration guard — check data sufficiency
                        _guard_ok = True
                        if self._calibration_guard is not None:
                            _guard_ok = self._calibration_guard.should_apply_penalty(_cal_report)
                        if _oc_ratio is not None and _oc_ratio > 0.15 and _guard_ok and not _oc_recovery_suspended:
                            _old_conf_oc = confidence
                            # Phase 57 (US-352): Apply ceiling before subtraction
                            # Phase 79: Fixed from 3.0 (0-100 scale) to 0.03 (0-1 scale)
                            _raw_sub = 0.03
                            if self._penalty_ceiling is not None:
                                _ceil_result = self._penalty_ceiling.check_subtraction(
                                    confidence, _raw_sub, pair=pair, source="overconfidence"
                                )
                                _raw_sub = _ceil_result.applied_penalty
                            confidence = max(confidence - _raw_sub, 0.0)
                            result.confidence = confidence
                            _oc_applied = True
                            logger.info(
                                f"{pair}: {_model_name} overconfident (ratio={_oc_ratio:.2f}>0.15) — "
                                f"conf {_old_conf_oc:.4f}→{confidence:.4f} (-0.03)"
                            )
                            # Log observation with metadata (US-184 AC)
                            try:
                                from src.scanner.automation.observation_log import ObservationLog
                                ObservationLog().log_observation(
                                    pair=pair,
                                    category="overconfidence_detected",
                                    description=f"{_model_name} overconfidence_ratio={_oc_ratio:.3f}",
                                    metadata={
                                        "model": _model_name,
                                        "overconfidence_ratio": _oc_ratio,
                                        "ece": _cal_report.get("ece"),
                                        "confidence_offset": -0.03,
                                    },
                                )
                            except Exception as e:
                                logger.debug("calibration overconfidence penalty observation log skipped: %s", e)
                            break  # Apply penalty once, not per-model
                    # US-184: Reset offset when overconfidence ratio drops below 0.10
                    if not _oc_applied:
                        for _model_name in ("TCN", "Ridge"):
                            _cal_report = self._model_calibration.get_calibration_report(_model_name)
                            _oc_ratio = _cal_report.get("overconfidence_ratio")
                            if _oc_ratio is not None and _oc_ratio < 0.10:
                                logger.debug(
                                    f"{pair}: {_model_name} overconfidence cleared (ratio={_oc_ratio:.3f}<0.10)"
                                )

                except Exception as _cal_track_err:
                    logger.debug(f"{pair}: Model calibration tracking skipped: {_cal_track_err}")

            # Phase 58 (US-358): Record raw pre-penalty confidence for ALL pairs (not just tradeable)
            if self._raw_conf_recorder is not None:
                try:
                    self._raw_conf_recorder.record(
                        pair=pair,
                        raw_confidence=confidence,
                        scan_id=str(getattr(self, "_scan_cycle", 0)),
                    )
                except Exception as _rcr_err:
                    logger.debug(f"Phase 58: raw_conf_recorder.record failed: {_rcr_err}")

            # Phase 18: Pair-regime-agent matrix — record agent votes
            if self._pair_regime_agent_matrix is not None and direction != "HOLD":
                try:
                    _agent_reasons = getattr(result, "agent_reasons", []) or []
                    for _reason in _agent_reasons[:12]:
                        if isinstance(_reason, dict) and "agent" in _reason:
                            self._pair_regime_agent_matrix.record_vote(
                                pair=pair,
                                regime=regime_name,
                                agent=_reason["agent"],
                                vote=_reason.get("vote", direction),
                                won=False,  # Updated later via record_trade_outcome_phase18
                            )
                except Exception as _matrix_err:
                    logger.debug(f"{pair}: Pair-regime-agent matrix skipped: {_matrix_err}")

            # Apply pre-trade backtest validation (soft gate: adjusts confidence, never hard-blocks)
            self._init_analysis_tools()
            if self._backtest_gate and result.direction != "HOLD":
                backtest_result = self._backtest_gate.validate(
                    pair=pair,
                    direction=result.direction,
                    sl_pips=result.sl_pips,
                    tp_pips=result.tp_pips,
                    recent_candles=df_raw,
                    pip_value=pip_value,
                )

                # Log backtest result and apply soft penalty if needed
                if backtest_result.trades_tested > 0:
                    logger.info(
                        f"{pair}: Backtest validation - {backtest_result.reason} "
                        f"(trades_tested={backtest_result.trades_tested}, win_rate={backtest_result.win_rate:.1%})"
                    )

                # SOFT gate: adjust confidence, never block
                if not backtest_result.passed and backtest_result.confidence_adjustment < 0:
                    old_confidence = result.confidence
                    result.confidence = max(0.0, result.confidence + backtest_result.confidence_adjustment)
                    logger.debug(
                        f"{pair}: Backtest gate penalty applied: {old_confidence:.3f} → {result.confidence:.3f}"
                    )

            # Apply trade cluster penalty/boost (US-030: avoid repeated exposure to same losing setup)
            if result.direction != "HOLD":
                try:
                    from src.scanner.analysis.trade_clustering import TradeClusterAnalyzer
                    cluster_analyzer = TradeClusterAnalyzer()
                    cluster_analyzer.load_from_journal()
                    regime_name = gate_details.get("volatility_regime_name", "NORMAL") if gate_details else "NORMAL"
                    cluster_adj = cluster_analyzer.get_cluster_penalty(
                        pair=pair,
                        direction=result.direction,
                        regime=str(regime_name),
                        confidence=result.confidence,
                    )
                    if cluster_adj != 0.0:
                        old_conf = result.confidence
                        result.confidence = max(0.0, min(1.0, result.confidence + cluster_adj))
                        logger.info(
                            f"{pair}: Cluster {'penalty' if cluster_adj < 0 else 'boost'}: "
                            f"{old_conf:.3f} → {result.confidence:.3f} ({cluster_adj:+.2f})"
                        )
                except Exception as cluster_err:
                    logger.debug(f"Trade cluster analysis error: {cluster_err}")

            # Phase 21 (US-127): Apply ThresholdOptimizer overrides to is_tradeable
            if self._threshold_optimizer is not None and result.direction != "HOLD":
                try:
                    _opt_thresholds = self._threshold_optimizer.optimize_thresholds(regime_name)
                    _opt_conf = _opt_thresholds.get("confidence", self.config.min_confidence / 100.0)
                    # If optimizer has enough data and recommends lower threshold, re-check
                    if result.confidence < _opt_conf and result.is_tradeable:
                        # Optimizer says threshold should be _opt_conf — confidence too low
                        pass  # Gate already checked; optimizer tightens, doesn't loosen beyond bounds
                    elif result.confidence >= _opt_conf and not result.is_tradeable:
                        # Optimizer loosened threshold — pair may now qualify
                        if result.gates_passed or (result.confidence_passed and result.risk_passed and result.momentum_passed):
                            result.is_tradeable = True
                            result.gates_passed = True
                            logger.info(
                                f"{pair}: US-127 ThresholdOptimizer re-qualified "
                                f"(conf={result.confidence:.3f} >= opt={_opt_conf:.3f})"
                            )
                    # Feed optimized thresholds to ConfigAdjuster
                    if self._config_adjuster is not None:
                        for _tkey, _tval in _opt_thresholds.items():
                            self._config_adjuster.collect_adjustment(
                                source="threshold_optimizer",
                                key=f"optimized_{_tkey}_threshold",
                                value=_tval,
                                reason=f"Rolling win rate optimization for {regime_name}",
                            )
                except Exception as _to_err:
                    logger.debug(f"{pair}: ThresholdOptimizer apply error: {_to_err}")

            # Phase 21 (US-132): Extreme regime execution policy
            # In EXTREME, reduce position size instead of blocking; lower confidence threshold
            _result_regime = str(getattr(result, "volatility_regime", "UNKNOWN")).upper()
            if (
                getattr(self.config, "enable_extreme_regime_policy", True)
                and _result_regime == "EXTREME"
                and result.direction != "HOLD"
            ):
                try:
                    _conf_offset = getattr(self.config, "extreme_regime_confidence_offset", -5.0)
                    _effective_min = max(0.30, (self.config.min_confidence + _conf_offset) / 100.0)
                    # If confidence is above the lowered threshold but below normal, re-qualify
                    if result.confidence >= _effective_min and not result.is_tradeable:
                        if result.risk_passed and result.confidence_passed and result.momentum_passed:
                            result.is_tradeable = True
                            result.gates_passed = True
                            # Mark for reduced position sizing (consumed in execution.py)
                            result.risk_pct = result.risk_pct * getattr(
                                self.config, "extreme_regime_size_multiplier", 0.50
                            )
                            logger.info(
                                f"{pair}: US-132 EXTREME policy re-qualified "
                                f"(conf={result.confidence:.3f} >= {_effective_min:.3f}, "
                                f"size_mult={getattr(self.config, 'extreme_regime_size_multiplier', 0.50)})"
                            )
                    elif result.is_tradeable and _result_regime == "EXTREME":
                        # Already tradeable in EXTREME — still reduce position size
                        result.risk_pct = result.risk_pct * getattr(
                            self.config, "extreme_regime_size_multiplier", 0.50
                        )
                except Exception as _ext_err:
                    logger.debug(f"{pair}: Extreme regime policy error: {_ext_err}")

            # Phase 23 (US-142): Execution quality fast-track
            # If trade was blocked only by agent consensus (gates_passed but not is_tradeable),
            # allow through with reduced risk if agent_score is high enough
            if (
                getattr(self.config, "enable_execution_fasttrack", True)
                and not result.is_tradeable
                and result.direction in {"LONG", "SHORT"}
                and result.confidence_passed
                and result.momentum_passed
                and result.risk_passed
                and getattr(result, "agent_score", 0) >= float(getattr(self.config, "fasttrack_min_quality", 0.70))
                and result.confidence >= float(getattr(self.config, "fasttrack_min_confidence", 0.50))
            ):
                _ft_risk_mult = float(getattr(self.config, "fasttrack_risk_multiplier", 0.75))
                result.is_tradeable = True
                result.gates_passed = True
                result.risk_pct = result.risk_pct * _ft_risk_mult
                # Phase 27 (US-166): Tag as fast-track for A/B outcome tracking
                result.fasttrack = True
                logger.info(
                    f"{pair}: US-142 fast-track activated "
                    f"(agent_score={getattr(result, 'agent_score', 0):.3f}, "
                    f"conf={result.confidence:.3f}, risk_mult={_ft_risk_mult})"
                )
                if self._observation_log is not None:
                    try:
                        self._observation_log.log_observation(
                            category="execution_fasttrack",
                            pair=pair,
                            data={
                                "agent_score": round(getattr(result, "agent_score", 0), 4),
                                "confidence": round(result.confidence, 4),
                                "risk_multiplier": _ft_risk_mult,
                                "regime": _result_regime,
                            },
                        )
                    except Exception as e:
                        logger.debug("%s: fast-track observation log skipped: %s", pair, e)
                        pass

            # Phase 21 (US-129): Trade block reason logging
            if (
                getattr(self.config, "enable_trade_block_logging", True)
                and not result.is_tradeable
                and result.confidence > 0.30  # Only log meaningful candidates
                and result.direction != "HOLD"
            ):
                try:
                    _block_reasons = []
                    if not result.confidence_passed:
                        _block_reasons.append(
                            f"confidence={result.confidence:.3f} < min={self.config.min_confidence}"
                        )
                    if not result.momentum_passed:
                        _block_reasons.append(
                            f"momentum={result.momentum:.3f} < min={self.config.min_momentum}"
                        )
                    if not result.risk_passed:
                        _block_reasons.append(
                            f"drawdown={result.drawdown:.4f} > max={self.config.max_drawdown_pct}"
                        )
                    if not result.volatility_gate_passed:
                        _block_reasons.append(f"volatility_gate failed (regime={_result_regime})")
                    # Check why_no_trade for agent consensus failures
                    _why = getattr(result, "why_no_trade", None) or []
                    if isinstance(_why, list):
                        for _w in _why:
                            if "consensus" in str(_w).lower():
                                _block_reasons.append(str(_w))
                    if _block_reasons:
                        from src.scanner.automation.observation_log import ObservationLog
                        ObservationLog().log_observation(
                            pair=pair,
                            category="trade_block_reason",
                            description="; ".join(_block_reasons),
                            metadata={
                                "confidence": result.confidence,
                                "momentum": result.momentum,
                                "drawdown": result.drawdown,
                                "regime": _result_regime,
                                "direction": result.direction,
                                "gates": {
                                    "confidence": result.confidence_passed,
                                    "momentum": result.momentum_passed,
                                    "risk": result.risk_passed,
                                    "volatility": result.volatility_gate_passed,
                                },
                            },
                        )

                        # Phase 56 (US-346): Record gate rejections to tracker for bottleneck analysis
                        if self._gate_rejection_tracker is not None:
                            try:
                                for _block_reason in _block_reasons:
                                    # Extract gate name from reason string prefix
                                    _gate_name = _block_reason.split("=")[0].strip().split(" ")[0]
                                    self._gate_rejection_tracker.record_rejection(
                                        pair=pair,
                                        gate_name=_gate_name,
                                        reason=_block_reason,
                                        confidence=float(result.confidence),
                                    )
                            except Exception as _grt_rec_err:
                                logger.debug(
                                    "%s: Phase 56 gate rejection record failed: %s",
                                    pair, _grt_rec_err,
                                )
                except Exception as e:
                    logger.debug("%s: trade_block observation log skipped: %s", pair, e)  # Non-blocking

            # Phase 56 (US-347): Record penalty sources to confidence penalty monitor
            if self._confidence_penalty_monitor is not None and result.direction != "HOLD":
                try:
                    _cpm_penalties: dict = {}
                    # Overconfidence: -3pt on 0-100 scale = 0.03 on 0-1 scale
                    if hasattr(result, "_overconfidence_applied") and result._overconfidence_applied:
                        _cpm_penalties["overconfidence"] = 0.03
                    # Concept drift multiplier < 1.0 implies a penalty was applied
                    _drift_mult = getattr(result, "_drift_penalty_mult", None)
                    if _drift_mult is not None and float(_drift_mult) < 1.0:
                        _cpm_penalties["concept_drift"] = round(1.0 - float(_drift_mult), 4)
                    # Disagreement tracker penalty (negative float)
                    _dt_penalty = getattr(result, "_disagreement_penalty", None)
                    if _dt_penalty is not None and float(_dt_penalty) < 0.0:
                        _cpm_penalties["disagreement"] = abs(float(_dt_penalty))
                    # Explainability penalty
                    _exp_penalty = getattr(result, "_explainability_penalty", None)
                    if _exp_penalty is not None and float(_exp_penalty) > 0.0:
                        _cpm_penalties["explainability"] = float(_exp_penalty)
                    if _cpm_penalties:
                        self._confidence_penalty_monitor.record_penalties(pair, _cpm_penalties)
                except Exception as _cpm_err:
                    logger.debug(
                        "%s: Phase 56 confidence penalty monitor record failed: %s",
                        pair, _cpm_err,
                    )

            # Phase 56 (US-348): Log rejected-but-promising setups as virtual trades
            if (
                self._virtual_trade_logger is not None
                and not result.is_tradeable
                and result.direction in {"LONG", "SHORT"}
                and result.confidence >= self._virtual_trade_logger.min_confidence
            ):
                try:
                    # Build gate failure list from result
                    _vtl_failures = []
                    if not result.confidence_passed:
                        _vtl_failures.append(f"confidence={result.confidence:.3f} < min={self.config.min_confidence}")
                    if not result.momentum_passed:
                        _vtl_failures.append(f"momentum={result.momentum:.3f}")
                    if not result.risk_passed:
                        _vtl_failures.append(f"risk_passed=False")
                    if not result.volatility_gate_passed:
                        _vtl_failures.append(f"volatility_gate_passed=False")
                    if not getattr(result, "agent_passed", True):
                        _vtl_failures.append("agent_passed=False")

                    # Build agent scores from result metadata
                    _vtl_agents = {}
                    _agents_info = getattr(result, "agents", {}) or {}
                    for _ar in (_agents_info.get("agent_reasons") or []):
                        _name = _ar.get("name", "")
                        _score = float(_ar.get("score", _ar.get("confidence", 0.0)))
                        if _name:
                            _vtl_agents[_name] = _score

                    self._virtual_trade_logger.log_virtual_trade(
                        pair=pair,
                        direction=result.direction,
                        confidence=float(result.confidence),
                        gate_failures=_vtl_failures,
                        features={},
                        agent_scores=_vtl_agents,
                    )
                except Exception as _vtl_err:
                    logger.debug("%s: Phase 56 virtual trade log failed: %s", pair, _vtl_err)

            # Cache result
            self._cached_results[pair] = deepcopy(result)
            self._cache_contexts[pair] = self._build_cache_context(pair)
            self._last_scan_times[pair] = time.time()

            return result

        except Exception as e:
            error_msg = str(e)
            error_class = type(e).__name__

            # Classify common errors into actionable messages
            if "not supported between instances" in error_msg:
                friendly = f"Type mismatch in scan pipeline: {error_msg}"
            elif "No module" in error_msg or "cannot import" in error_msg.lower():
                friendly = f"Missing dependency: {error_msg}"
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                friendly = f"Data fetch timeout — check OANDA connection"
            elif "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg:
                friendly = "OANDA auth failed — check API token in env"
            elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                friendly = "Connection error — check network/OANDA status"
            elif "shape" in error_msg.lower() or "dimension" in error_msg.lower():
                friendly = f"Model input shape error: {error_msg}"
            elif "NaN" in error_msg or "inf" in error_msg.lower():
                friendly = f"Bad data in features: {error_msg}"
            elif "KeyError" in error_class:
                friendly = f"Missing data column: {error_msg}"
            else:
                friendly = f"{error_class}: {error_msg}"

            logger.error(f"[{pair}] Scan failed ({error_class}): {error_msg}")
            return PairAnalysis(
                pair=pair,
                direction="HOLD",
                confidence=0.0,
                error=friendly,
                kill_reason="other",
            )

    def scan(
        self,
        pairs: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        on_pair_complete: Optional[Callable[[PairAnalysis], None]] = None,
    ) -> ScanResult:
        """Scan multiple pairs in parallel.

        Args:
            pairs: List of pairs to scan (uses default if None)
            max_workers: Number of parallel workers
            on_pair_complete: Callback when each pair completes (for live updates)

        Returns:
            ScanResult with all pair analyses
        """
        # CRITICAL: Reload agent weights at start of each scan to apply RL updates
        self._agent_team.reload_learned_weights()

        # Phase 82 (US-P82-003): ChainMemory — sync from PRD once per session
        if self._chain_memory is not None and not self._chain_memory_synced:
            try:
                self._chain_memory.sync_from_prd()
                self._chain_memory_synced = True
                logger.info("Phase 82 (US-P82-003): ChainMemory synced from PRD")
            except Exception as _cm_sync_err:
                logger.debug(f"Phase 82: ChainMemory sync_from_prd error: {_cm_sync_err}")

        # Phase 49 (US-307): Tick cooldown on pair blacklist each scan cycle
        if self._pair_tracker is not None:
            try:
                _removed = self._pair_tracker.tick_cooldown()
                if _removed:
                    logger.info("Phase 49 (US-307): Pair blacklist cooldown expired for: %s", _removed)
            except Exception as _tc_err:
                logger.debug(f"Phase 49: Pair tracker tick_cooldown error: {_tc_err}")

        # Phase 20 (US-121): Health registry pre-flight check
        if self._health_registry is not None:
            try:
                preflight = self._health_registry.run_preflight()
                if not self._health_registry.should_continue_cycle():
                    health_score = self._health_registry.get_system_health_score()
                    failed = [n for n, ok in preflight.items() if not ok]
                    logger.warning(
                        "Health preflight FAILED (score=%.2f, failed=%s) — skipping scan cycle",
                        health_score, failed,
                    )
                    self._health_registry.save_state()
                    return ScanResult(analyses=[], model_type="skipped_unhealthy", granularity=self.config.granularity)
                logger.debug("Health preflight passed (score=%.2f)", self._health_registry.get_system_health_score())
            except Exception as hr_err:
                logger.debug(f"Health preflight error (non-blocking): {hr_err}")

        pair_list = pairs if pairs is not None else (self.config.pairs or self.config.default_pairs)
        worker_count = max(1, int(max_workers or self.config.parallel_workers))

        # M1 memory safety: cap thread workers to prevent CPU/memory starvation
        if getattr(self, '_is_low_memory', False) and worker_count > 2:
            logger.info("Low-memory mode: capping scan workers from %d to 2", worker_count)
            worker_count = 2
        # Hard cap: never exceed 4 workers regardless of config (M1 has 8 cores
        # but TensorFlow inference + GC already saturates them)
        worker_count = min(worker_count, 4)

        # Check session before starting
        if not self.config.non_interactive:
            session_status = self.config.get_trading_session_status()
            if not session_status["session_allowed"] and session_status.get("message"):
                logger.warning(str(session_status["message"]))

        # Initialize components
        self._init_oanda_client()
        self._init_gate_evaluator()
        self._init_feature_engineer()
        self._init_modular_ensemble()

        # Parallel scan
        analyses: List[PairAnalysis] = []

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self._scan_pair, pair): pair
                for pair in pair_list
            }

            for future in as_completed(futures, timeout=300):
                pair = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result is not None:
                        analyses.append(result)

                        # Callback for live display updates
                        if on_pair_complete:
                            on_pair_complete(result)

                except TimeoutError:
                    logger.error(f"Scan TIMEOUT for {pair} (>60s)")
                except Exception as e:
                    logger.error(f"Scan failed for {pair}: {e}")
                    analyses.append(PairAnalysis(
                        pair=pair,
                        direction="HOLD",
                        confidence=0.0,
                        error=str(e),
                    ))

        analyses = self._run_sub_inference_pass(
            analyses,
            on_pair_complete=on_pair_complete,
        )

        # Update regime tracker with dominant regime from this scan cycle
        if self._regime_tracker and analyses:
            valid_regime_names = {"LOW", "NORMAL", "HIGH", "EXTREME"}
            regimes_seen = [
                a.volatility_regime for a in analyses
                if a.volatility_regime is not None and a.volatility_regime in valid_regime_names
            ]
            if regimes_seen:
                # Use the most common regime across pairs
                from collections import Counter
                dominant_regime = Counter(regimes_seen).most_common(1)[0][0]
                self._regime_tracker.update(dominant_regime)

        # Compute group momentum from scan results (Phase 8: US-054)
        self._last_group_momentum = {}
        if self._group_momentum and analyses:
            scan_signals = []
            for a in analyses:
                if a.direction and a.direction.upper() != "HOLD" and a.confidence > 0:
                    scan_signals.append({
                        "pair": a.pair,
                        "direction": a.direction.upper(),
                        "confidence": a.confidence,
                    })
            if scan_signals:
                self._last_group_momentum = self._group_momentum.compute_group_momentum(
                    scan_signals
                )

        # Phase 18: Save state for calibration modules after each scan cycle
        self._save_phase18_state()

        # Phase 20 (US-121): Health registry post-flight check + save Phase 19 state
        if self._health_registry is not None:
            try:
                postflight = self._health_registry.run_postflight()
                failed_post = [n for n, ok in postflight.items() if not ok]
                if failed_post:
                    logger.warning("Health postflight: %d module(s) failed: %s", len(failed_post), failed_post)
                self._health_registry.save_state()
            except Exception as hr_err:
                logger.debug(f"Health postflight error: {hr_err}")
        self._save_phase19_state()

        # Phase 81 (US-P81-002): PositionTimeoutManager — tick bar counter each scan cycle
        if self._position_timeout is not None:
            try:
                self._position_timeout.tick()
            except Exception as _pt_tick_err:
                logger.debug("Phase 81: PositionTimeout tick error: %s", _pt_tick_err)

        # Phase 81 (US-P81-002): PositionTimeoutManager — evaluate open positions for forced exits
        if self._position_timeout is not None and self._executor is not None:
            try:
                _timeout_results = self._position_timeout.evaluate_all()
                for _tr in _timeout_results:
                    if getattr(_tr, "should_exit", False):
                        _exit_reason = getattr(_tr, "exit_reason", "timeout")
                        _timed_out_id = getattr(_tr, "trade_id", "")
                        logger.warning(
                            "Phase 81 (US-P81-002): Timeout exit for trade %s — %s",
                            _timed_out_id, _exit_reason,
                        )
                        try:
                            self._executor.close_trade(
                                trade_id=_timed_out_id, reason=_exit_reason
                            )
                            self._position_timeout.close_trade(_timed_out_id)
                        except Exception as _pt_close_err:
                            logger.warning(
                                "Phase 81: PositionTimeout force-close failed for %s: %s",
                                _timed_out_id, _pt_close_err,
                            )
            except Exception as _pt_eval_err:
                logger.debug("Phase 81: PositionTimeout evaluate_all error: %s", _pt_eval_err)

        # Phase 20 (US-125): Run drift monitor every 20 scan cycles
        self._scan_cycle_count += 1
        self._run_drift_check_cycle(analyses)

        # Phase 80 (US-P80-005): Forward drift projection every 20 scan cycles
        self._run_drift_projection_cycle(analyses)

        # US-089: Periodic adversarial robustness check every 10 scan cycles
        self._run_adversarial_robustness_check(analyses)

        # Phase 80 (US-P80-006): Adversarial deep test (train_robust) every 100 scan cycles
        self._run_adversarial_deep_test(analyses)

        # Phase 83 (US-P83-001): AlertManager — check drawdown, consecutive losses, weight instability
        if self._alert_manager is not None:
            try:
                _am_nav = getattr(self._executor, "_cached_nav", 0.0) or 0.0 if self._executor else 0.0
                _am_peak = getattr(self._executor, "_equity_hwm", _am_nav) or _am_nav if self._executor else _am_nav
                # Load last 20 closed trades for loss streak detection
                _am_trades = None
                try:
                    import json as _am_json
                    with open("trained_data/trade_journal_rl.json", "r") as _am_f:
                        _am_all = _am_json.load(_am_f)
                    _am_trades = [t for t in _am_all if isinstance(t.get("outcome"), dict)][-20:]
                except Exception:
                    pass
                _am_cur_weights = getattr(self._agent_team, "_learned_weights", None)
                _am_prev_weights = getattr(self, "_previous_weights", None)
                _am_alerts = self._alert_manager.check_all(
                    nav=_am_nav, peak_nav=_am_peak,
                    recent_trades=_am_trades,
                    current_weights=_am_cur_weights,
                    previous_weights=_am_prev_weights,
                )
                self._previous_weights = _am_cur_weights  # store for next cycle
                for _alert in (_am_alerts or []):
                    logger.warning(
                        "Phase 83 (US-P83-001): ALERT [%s] severity=%s — %s",
                        getattr(_alert, "type", "?"),
                        getattr(_alert, "severity", "?"),
                        getattr(_alert, "message", ""),
                    )
            except Exception as _am_err:
                logger.debug("Phase 83: AlertManager.check_all error: %s", _am_err)

        # Phase 83 (US-P83-003): RetainTrigger.check_drift() every 20 scan cycles
        if self._retrain_trigger is not None and self._scan_cycle_count % 20 == 0:
            try:
                _rt_request = self._retrain_trigger.check_drift()
                if _rt_request is not None:
                    logger.warning(
                        "Phase 83 (US-P83-003): Retrain drift detected — pairs=%s reason=%s",
                        getattr(_rt_request, "pairs_to_retrain", []),
                        getattr(_rt_request, "reason", "accuracy_decline"),
                    )
            except Exception as _rt_err:
                logger.debug("Phase 83: RetainTrigger.check_drift error: %s", _rt_err)

        # Phase 87: DynamicHedgeManager — evaluate hedges when macro stress detected
        if self._dynamic_hedging is not None:
            try:
                _dh_stress = self._macro_stress.get_stress_modifier() if self._macro_stress else 1.0
                if _dh_stress < 1.0:
                    _dh_positions = []
                    if self._executor is not None:
                        try:
                            _dh_open = self._executor.monitor_open_trades(evaluate_exits=False) or []
                            _dh_positions = [
                                {"pair": t.get("pair", ""), "direction": t.get("direction", "BUY"),
                                 "lots": float(t.get("lots", 0)), "unrealized_pnl": float(t.get("unrealized_pnl", 0))}
                                for t in _dh_open
                            ]
                        except Exception:
                            pass
                    if _dh_positions:
                        _dh_status = self._dynamic_hedging.evaluate(
                            open_positions=_dh_positions, stress_modifier=_dh_stress,
                        )
                        if getattr(_dh_status, "is_hedging_active", False):
                            _dh_candidates = getattr(_dh_status, "candidates", [])
                            if _dh_candidates:
                                logger.warning(
                                    "Phase 87: DynamicHedge ACTIVE — stress=%.2f, %d hedge candidates",
                                    _dh_stress, len(_dh_candidates),
                                )
                        _dh_close = getattr(_dh_status, "hedges_to_close", [])
                        if _dh_close:
                            logger.info("Phase 87: DynamicHedge releasing %d hedges (stress receding)", len(_dh_close))
            except Exception as _dh_err:
                logger.debug("Phase 87: DynamicHedgeManager error: %s", _dh_err)

        # Feed ALL pairs' prices into EWMA correlation engine so the
        # correlation matrix reflects the full universe, not just traded pairs.
        self._feed_ewma_from_scan(analyses)

        # Build scan result
        model_type = "technical"
        if bool(getattr(self, "_ensemble_loaded", False)):
            model_type = "ensemble"
        elif self._gate_evaluator is not None and self._gate_evaluator.is_loaded:
            model_type = self._gate_evaluator.momentum_model_type

        # Free DataFrame caches after every scan (not just continuous mode)
        # to prevent OOM on 8GB M1 when running single-scan commands.
        self.clear_scan_caches()

        return ScanResult(
            analyses=analyses,
            model_type=model_type,
            granularity=self.config.granularity,
        )

    # ── EWMA correlation: scan-loop feeder ──────────────────────────

    def _init_ewma_engine(self) -> None:
        """Lazily initialize the scanner-owned EWMA correlation engine."""
        if self._ewma_engine is not None:
            return
        try:
            from src.risk.ewma_correlation import create_default_ewma_engine

            _pairs = list(self.config.pairs or self.config.default_pairs)
            self._ewma_engine = create_default_ewma_engine(pairs=_pairs)

            # Load persisted state so we continue from where we left off
            import os
            _state_path = "trained_data/ewma_correlation_state.json"
            if os.path.exists(_state_path):
                try:
                    self._ewma_engine.load_state(_state_path)
                    logger.info("EWMA: Loaded persisted correlation state into scanner engine")
                except Exception as e:
                    logger.warning(f"EWMA: Could not load persisted state: {e}")

            logger.info("EWMA: Scanner correlation engine initialized (%d pairs)", len(_pairs))
        except Exception as e:
            logger.debug(f"EWMA: Engine init failed (non-blocking): {e}")

    def _feed_ewma_from_scan(self, analyses: list) -> None:
        """Feed ALL scanned pairs' log returns into the EWMA correlation engine.

        Computes log returns from consecutive scan cycles using cached previous
        prices, then batch-updates the EWMA engine with a single returns dict.

        Args:
            analyses: List of PairAnalysis results from this scan cycle.
        """
        if not analyses:
            return

        self._init_ewma_engine()
        if self._ewma_engine is None:
            return

        try:
            import numpy as np

            returns_dict: Dict[str, float] = {}
            new_price_cache: Dict[str, float] = {}

            for a in analyses:
                price = getattr(a, "current_price", 0.0)
                if not price or price <= 0:
                    continue

                new_price_cache[a.pair] = price
                prev_price = self._ewma_prev_prices.get(a.pair, 0.0)

                if prev_price > 0:
                    log_ret = float(np.log(price / prev_price))
                    # Sanity: skip absurd returns (>10% per scan cycle)
                    if abs(log_ret) < 0.10:
                        returns_dict[a.pair] = log_ret

            # Update previous price cache for next cycle
            self._ewma_prev_prices.update(new_price_cache)

            if returns_dict:
                state = self._ewma_engine.update(returns_dict)
                logger.info(
                    "EWMA: Fed %d/%d pairs (obs=%d, avg_corr=%.3f, regime=%s)",
                    len(returns_dict),
                    len(analyses),
                    state.observation_count,
                    state.avg_correlation,
                    state.regime,
                )

                # Persist so ExecutionManager picks up the warmed matrix
                try:
                    self._ewma_engine.save_state("trained_data/ewma_correlation_state.json")
                except Exception as e:
                    logger.debug(f"EWMA: State persistence failed: {e}")
            else:
                logger.debug("EWMA: No returns to feed (first cycle or all prices unchanged)")

        except Exception as e:
            logger.warning(f"EWMA: Scan-loop feed failed (non-blocking): {e}")

    def _save_phase18_state(self) -> None:
        """Persist Phase 18 module state after scan cycles."""
        for module in (
            self._execution_quality_optimizer,
            self._threshold_optimizer,
            self._dynamic_risk_allocator,
            self._model_calibration,
            self._pair_regime_agent_matrix,
            self._signal_timing,
            self._model_router,
            self._model_bandit,
            self._attention_feedback,
        ):
            if module is not None:
                try:
                    module.save_state()
                except Exception as e:
                    logger.debug(f"Phase 18 save_state error: {e}")

    def _save_phase19_state(self) -> None:
        """Persist Phase 19 module state after scan cycles (US-121)."""
        for module in (
            self._memory_manager,
            self._module_activation,
            self._agent_lifecycle,
            self._observation_consumer,
            self._replay_validator,
            self._regime_broadcaster,
            self._causal_filter,
        ):
            if module is not None:
                try:
                    module.save_state()
                except Exception as e:
                    logger.debug(f"Phase 19 save_state error: {e}")

    # ── Drift monitor: scan-loop integration ────────────────────────

    _DRIFT_CHECK_INTERVAL = 20  # Run drift check every N scan cycles

    def _run_drift_check_cycle(self, analyses: list) -> None:
        """Run DriftMonitor on scanned pairs at periodic intervals.

        Converts cached raw OHLCV DataFrames into candle dicts that
        DriftMonitor.run_drift_check() expects, checks each pair for
        feature/concept drift, and logs results.  On critical drift,
        writes a retrain request file that scheduled_retrain.py can
        pick up.

        Args:
            analyses: List of PairAnalysis results from this scan cycle.
        """
        if self._drift_monitor is None:
            return
        if self._scan_cycle_count % self._DRIFT_CHECK_INTERVAL != 0:
            return
        if not analyses:
            return

        drift_pairs_flagged: list = []

        for analysis in analyses:
            pair = analysis.pair
            df_raw = self._raw_snapshots.get(pair)
            if df_raw is None or df_raw.empty:
                continue

            try:
                # Convert DataFrame rows to list-of-dicts for replay
                candle_cols = [c for c in ("time", "open", "high", "low", "close", "volume") if c in df_raw.columns]
                historical_candles = df_raw[candle_cols].tail(200).to_dict(orient="records")

                report = self._drift_monitor.run_drift_check(
                    pair=pair,
                    historical_candles=historical_candles,
                )

                if report.get("severity") == "critical":
                    logger.warning(
                        "DriftMonitor: CRITICAL drift on %s (score=%.3f) — "
                        "retrain recommended, sizing reduced",
                        pair, report.get("drift_score", 0.0),
                    )
                    drift_pairs_flagged.append(pair)
                elif report.get("severity") == "mild":
                    logger.warning(
                        "DriftMonitor: Mild drift on %s (score=%.3f)",
                        pair, report.get("drift_score", 0.0),
                    )
                    drift_pairs_flagged.append(pair)

            except Exception as e:
                logger.debug("DriftMonitor: Check failed for %s (non-blocking): %s", pair, e)

        # Bridge to RetrainTrigger: write retrain request if critical drift found
        if drift_pairs_flagged:
            self._write_drift_retrain_request(drift_pairs_flagged)

        # Persist drift monitor state
        try:
            self._drift_monitor.save_state()
        except Exception as e:
            logger.debug("DriftMonitor: save_state failed: %s", e)

    # ── Adversarial robustness: periodic scan-loop check ─────────────

    _ADVERSARIAL_CHECK_INTERVAL = 10  # Run adversarial check every N scan cycles

    def _run_adversarial_robustness_check(self, analyses: list) -> None:
        """Run adversarial robustness evaluation on cached feature data periodically.

        Uses feature snapshots accumulated during the scan cycle to generate
        adversarial perturbations and measure ensemble robustness. Results are
        persisted and consumed per-pair in _scan_pair() via get_latest_score().

        Args:
            analyses: List of PairAnalysis results from this scan cycle.
        """
        if self._adversarial_trainer is None:
            return
        if self._scan_cycle_count % self._ADVERSARIAL_CHECK_INTERVAL != 0:
            return
        if not analyses:
            return

        try:
            import numpy as np

            # Build feature matrix from cached feature snapshots
            X_parts = []
            for analysis in analyses:
                df_feat = self._feature_snapshots.get(analysis.pair)
                if df_feat is None or df_feat.empty:
                    continue
                # Use numeric columns only, take last row as representative sample
                numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    continue
                row = df_feat[numeric_cols].iloc[-1:].values
                if not np.isfinite(row).all():
                    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
                X_parts.append(row)

            if len(X_parts) < 3:
                logger.debug("AdversarialTrainer: Not enough feature data (%d pairs), skipping", len(X_parts))
                return

            X_clean = np.vstack(X_parts)
            # Generate synthetic binary labels from confidence (above/below median)
            y_clean = np.array([
                1 if (a.confidence or 0.0) >= 0.5 else 0
                for a in analyses[:len(X_parts)]
            ])

            # Generate adversarial perturbations
            adv_batches = self._adversarial_trainer.generate_adversarial(
                X_clean, y_clean,
            )

            # If the modular ensemble exposes a predict-compatible sub-model, evaluate robustness
            _model = None
            if hasattr(self._modular_ensemble, 'ridge') and self._modular_ensemble.ridge is not None:
                _model = self._modular_ensemble.ridge
            elif hasattr(self._modular_ensemble, 'histgb') and self._modular_ensemble.histgb is not None:
                _model = self._modular_ensemble.histgb

            if _model is not None and len(adv_batches) > 0:
                # Stack all adversarial data
                adv_X_list = [b.X for b in adv_batches.values() if b.n_samples > 0]
                if adv_X_list:
                    X_adv = np.vstack(adv_X_list)
                    _score = self._adversarial_trainer.compute_robustness_score(
                        model=_model,
                        X_clean=X_clean,
                        X_adversarial=X_adv,
                        y=y_clean,
                    )
                    # Persist so per-pair lookups get fresh data
                    self._adversarial_trainer._history.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "score": _score.to_dict(),
                    })
                    self._adversarial_trainer.save_scores()
                    logger.info(
                        "AdversarialTrainer: Periodic check — "
                        "clean=%.3f, adv=%.3f, ratio=%.3f (%d pairs)",
                        _score.clean_accuracy,
                        _score.adversarial_accuracy,
                        _score.robustness_ratio,
                        len(X_parts),
                    )
            else:
                # No sklearn-compatible sub-model available; just generate & persist
                # so the perturbation stats are available for future training runs
                total_adv = sum(b.n_samples for b in adv_batches.values())
                logger.info(
                    "AdversarialTrainer: Generated %d adversarial samples "
                    "(no sub-model available for robustness eval)",
                    total_adv,
                )

            # Feedback hook: log periodic adversarial check to observation pipeline
            try:
                _obs = self._get_feedback_observer()
                if _obs is not None:
                    _latest = self._adversarial_trainer.get_latest_score()
                    _obs.log_observation(
                        pair="PORTFOLIO",
                        category="adversarial_robustness_check",
                        description=(
                            f"Periodic adversarial check: ratio={_latest.robustness_ratio:.3f}"
                            if _latest else "Adversarial check completed (no score)"
                        ),
                        metadata={
                            "robustness_ratio": round(_latest.robustness_ratio, 4) if _latest else 0.0,
                            "clean_accuracy": round(_latest.clean_accuracy, 4) if _latest else 0.0,
                            "adversarial_accuracy": round(_latest.adversarial_accuracy, 4) if _latest else 0.0,
                            "n_pairs_evaluated": len(X_parts),
                            "cycle_count": self._scan_cycle_count,
                        },
                    )
            except Exception:
                pass

        except Exception as _adv_check_err:
            logger.debug("AdversarialTrainer: Periodic check failed (non-blocking): %s", _adv_check_err)

    # ── Phase 80 (US-P80-005): Drift projector — forward projection ────────────────

    _DRIFT_PROJECTION_INTERVAL = 20  # Match _DRIFT_CHECK_INTERVAL

    def _run_drift_projection_cycle(self, analyses: list) -> None:
        """Project future accuracy degradation every 20 scan cycles.

        Uses DriftProjector.project_all_pairs() to generate t+6h, t+12h, t+24h,
        t+48h accuracy projections for all scanned pairs. Results persisted to
        trained_data/drift_projections.json for dashboard consumption.
        Current accuracy supplied from AccuracyGate stats when available.
        """
        if self._drift_projector is None:
            return
        if self._scan_cycle_count % self._DRIFT_PROJECTION_INTERVAL != 0:
            return
        if not analyses:
            return

        try:
            pairs = [a.pair for a in analyses if a.pair]
            if not pairs:
                return

            # Build current accuracy map from accuracy gate stats
            current_accuracies: Dict[str, float] = {}
            _ag = getattr(self, "_accuracy_gate", None)
            if _ag is not None:
                for _pair in pairs:
                    try:
                        _stats = _ag.get_stats(_pair)
                        if isinstance(_stats, dict) and "accuracy" in _stats:
                            current_accuracies[_pair] = float(_stats["accuracy"])
                    except Exception:
                        pass

            projections = self._drift_projector.project_all_pairs(
                pairs=pairs,
                current_accuracies=current_accuracies if current_accuracies else None,
            )

            self._drift_projector.save_projections(projections)

            # Log any high-urgency projections as warnings
            critical_pairs = [
                p for p, c in projections.items()
                if getattr(c, "action_urgency", "") in ("high", "critical")
            ]
            if critical_pairs:
                logger.warning(
                    "Phase 80 (US-P80-005): DriftProjector — %d pair(s) with high/critical urgency: %s",
                    len(critical_pairs), critical_pairs,
                )
            else:
                logger.info(
                    "Phase 80 (US-P80-005): DriftProjector — projected %d pairs (cycle=%d)",
                    len(projections), self._scan_cycle_count,
                )

        except Exception as _dp_err:
            logger.debug(
                "Phase 80 (US-P80-005): DriftProjector cycle failed (non-blocking): %s", _dp_err
            )

    # ── Phase 80 (US-P80-006): Adversarial deep test — 100-cycle cadence ──────────

    _ADVERSARIAL_DEEP_TEST_INTERVAL = 100  # Full train_robust every 100 scans

    def _run_adversarial_deep_test(self, analyses: list) -> None:
        """Run full adversarial robustness training every 100 scan cycles.

        Complements the lightweight 10-cycle check (_run_adversarial_robustness_check)
        with a full train_robust() pass (3 rounds). A robustness_ratio below 0.70
        (adversarial accuracy < 70% of clean accuracy) is logged as a WARNING.
        """
        if self._adversarial_trainer is None:
            return
        if self._scan_cycle_count % self._ADVERSARIAL_DEEP_TEST_INTERVAL != 0:
            return
        if not analyses:
            return

        try:
            import numpy as np

            # Build feature matrix from cached feature snapshots
            X_parts = []
            for analysis in analyses:
                df_feat = self._feature_snapshots.get(analysis.pair)
                if df_feat is None or df_feat.empty:
                    continue
                numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    continue
                row = df_feat[numeric_cols].iloc[-1:].values
                if not np.isfinite(row).all():
                    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
                X_parts.append(row)

            if len(X_parts) < 3:
                logger.debug(
                    "Phase 80 (US-P80-006): AdversarialTrainer deep test: "
                    "insufficient data (%d pairs)", len(X_parts)
                )
                return

            X_clean = np.vstack(X_parts)
            y_clean = np.array([
                1 if (a.confidence or 0.0) >= 0.5 else 0
                for a in analyses[:len(X_parts)]
            ])

            # Select best available sklearn sub-model
            _model = None
            if hasattr(self._modular_ensemble, "ridge") and self._modular_ensemble.ridge is not None:
                _model = self._modular_ensemble.ridge
            elif hasattr(self._modular_ensemble, "histgb") and self._modular_ensemble.histgb is not None:
                _model = self._modular_ensemble.histgb

            if _model is None:
                logger.debug(
                    "Phase 80 (US-P80-006): AdversarialTrainer deep test: "
                    "no sklearn sub-model available"
                )
                return

            _model_after, _score = self._adversarial_trainer.train_robust(
                model=_model,
                X_train=X_clean,
                y_train=y_clean,
                n_rounds=3,
            )

            if _score.robustness_ratio < 0.70:
                logger.warning(
                    "Phase 80 (US-P80-006): AdversarialTrainer deep test FAILED — "
                    "robustness_ratio=%.3f < 0.70 (clean=%.3f adv=%.3f rounds=%d cycle=%d)",
                    _score.robustness_ratio,
                    _score.clean_accuracy,
                    _score.adversarial_accuracy,
                    getattr(_score, "training_rounds", 3),
                    self._scan_cycle_count,
                )
            else:
                logger.info(
                    "Phase 80 (US-P80-006): AdversarialTrainer deep test PASSED — "
                    "robustness_ratio=%.3f (clean=%.3f adv=%.3f) cycle=%d",
                    _score.robustness_ratio,
                    _score.clean_accuracy,
                    _score.adversarial_accuracy,
                    self._scan_cycle_count,
                )

        except Exception as _deep_err:
            logger.debug(
                "Phase 80 (US-P80-006): AdversarialTrainer deep test failed (non-blocking): %s",
                _deep_err
            )

    def _write_drift_retrain_request(self, pairs: list) -> None:
        """Write a retrain request file when DriftMonitor flags pairs.

        This bridges DriftMonitor output into the same retrain_request.json
        that RetrainTrigger and scheduled_retrain.py already consume.

        Args:
            pairs: List of pair codes flagged for drift.
        """
        try:
            import fcntl
            import json as _json_local
            import os as _os_local
            from pathlib import Path

            request_path = Path("trained_data/retrain_request.json")
            request_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if a request already exists (don't overwrite higher-priority)
            if request_path.exists():
                try:
                    existing = _json_local.loads(request_path.read_text())
                    if existing.get("priority") == "urgent":
                        logger.debug("DriftMonitor: Urgent retrain request already pending, skipping write")
                        return
                except Exception as e:
                    logger.debug("drift monitor retrain request check failed: %s", e)
                    pass

            request_data = {
                "pairs": pairs,
                "reason": f"DriftMonitor: {len(pairs)} pair(s) showing feature/concept drift",
                "accuracy_snapshot": {
                    p: {"drift_severity": self._drift_monitor.get_size_multiplier(p)}
                    for p in pairs
                },
                "triggered_at": datetime.now(timezone.utc).isoformat(),
                "priority": "normal",
                "source": "drift_monitor",
            }

            with open(request_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    _json_local.dump(request_data, f, indent=2, sort_keys=True)
                    f.flush()
                    _os_local.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.info(
                "DriftMonitor: Retrain request written for %d pair(s): %s",
                len(pairs), ", ".join(pairs),
            )
        except Exception as e:
            logger.debug("DriftMonitor: Failed to write retrain request: %s", e)

    def _register_health_checks(self) -> None:
        """Register Phase 19 modules with health registry (US-121).

        Called once after all modules are initialized. Each module gets a
        health check lambda that verifies it's not None and responsive.
        """
        if self._health_registry is None:
            return

        # Register core components as critical
        self._health_registry.register_module(
            "engine", lambda: self._agent_team is not None, critical=True,
        )
        self._health_registry.register_module(
            "agents", lambda: self._agent_team is not None, critical=True,
        )

        # Register Phase 19 modules as non-critical (graceful degradation)
        phase19_modules = {
            "memory_manager": self._memory_manager,
            "module_activation": self._module_activation,
            "agent_lifecycle": self._agent_lifecycle,
            "observation_consumer": self._observation_consumer,
            "replay_validator": self._replay_validator,
        }
        for name, mod in phase19_modules.items():
            if mod is not None:
                self._health_registry.register_module(
                    name,
                    health_check=lambda m=mod: hasattr(m, "get_status"),
                    critical=False,
                )

        # Register Phase 18 modules
        phase18_modules = {
            "execution_quality_optimizer": self._execution_quality_optimizer,
            "threshold_optimizer": self._threshold_optimizer,
            "dynamic_risk_allocator": self._dynamic_risk_allocator,
            "model_calibration": self._model_calibration,
            "pair_regime_agent_matrix": self._pair_regime_agent_matrix,
            "signal_timing": self._signal_timing,
        }
        for name, mod in phase18_modules.items():
            if mod is not None:
                self._health_registry.register_module(name, critical=False)

        logger.info(
            "Health registry: %d modules registered",
            len(self._health_registry._modules),
        )

    def record_trade_outcome_phase18(
        self,
        pair: str,
        regime: str,
        pnl_pips: float,
        trade_won: bool,
        duration_bars: int = 0,
        gate_values: Optional[Dict[str, float]] = None,
        agent_verdicts: Optional[List[Dict[str, Any]]] = None,
        slippage_pips: float = 0.0,
        fill_time_ms: float = 0.0,
        fill_rate: float = 1.0,
        model: str = "ensemble",
        signal_time: float = 0.0,
        entry_time: float = 0.0,
        peak_time: float = 0.0,
    ) -> None:
        """Feed Phase 18 calibration modules with a trade outcome.

        Called by ExecutionManager after trade closure to update:
        - ThresholdOptimizer (gate tuning from win/loss)
        - DynamicRiskAllocator (P/L distribution)
        - ExecutionQualityOptimizer (cost profiling)
        - ModelCalibrationTracker (prediction accuracy)
        - PairRegimeAgentMatrix (agent accuracy per context)
        - SignalTimingOptimizer (signal-to-entry-to-peak gaps)
        """
        if self._threshold_optimizer is not None and gate_values:
            try:
                self._threshold_optimizer.update_outcome(regime, gate_values, trade_won)
            except Exception as e:
                logger.debug(f"ThresholdOptimizer outcome error: {e}")

        if self._dynamic_risk_allocator is not None:
            try:
                self._dynamic_risk_allocator.record_trade(pnl_pips, duration_bars, regime)
            except Exception as e:
                logger.debug(f"DynamicRiskAllocator outcome error: {e}")

        if self._execution_quality_optimizer is not None:
            try:
                self._execution_quality_optimizer.record_execution(
                    pair, regime, model, slippage_pips, fill_time_ms, fill_rate
                )
            except Exception as e:
                logger.debug(f"ExecutionQualityOptimizer outcome error: {e}")

        if self._model_calibration is not None:
            try:
                self._model_calibration.record_flock_outcome(trade_won)
            except Exception as e:
                logger.debug(f"ModelCalibrationTracker outcome error: {e}")

        if self._pair_regime_agent_matrix is not None and agent_verdicts:
            try:
                for verdict in agent_verdicts:
                    if isinstance(verdict, dict) and "agent" in verdict:
                        self._pair_regime_agent_matrix.record_vote(
                            pair=pair,
                            regime=regime,
                            agent=verdict["agent"],
                            vote=verdict.get("vote", "HOLD"),
                            won=trade_won,
                        )
            except Exception as e:
                logger.debug(f"PairRegimeAgentMatrix outcome error: {e}")

        if self._signal_timing is not None and signal_time > 0 and entry_time > 0:
            try:
                self._signal_timing.record_signal(pair, signal_time, entry_time, peak_time)
            except Exception as e:
                logger.debug(f"SignalTimingOptimizer outcome error: {e}")

        # US-106: Model router + bandit reward update from trade outcome
        if self._model_router is not None:
            try:
                self._model_router.record_outcome(pair, regime, model, pnl_pips)
            except Exception as e:
                logger.debug(f"ModelRouter outcome error: {e}")

        if self._model_bandit is not None:
            try:
                self._model_bandit.update_reward(model, pnl_pips)
            except Exception as e:
                logger.debug(f"ModelBandit outcome error: {e}")

        # Attention feedback: reinforce timeframe quality based on trade outcome
        if self._attention_feedback is not None and agent_verdicts:
            try:
                # Build agent votes from verdicts
                _attn_votes = {}
                for _v in agent_verdicts:
                    if isinstance(_v, dict):
                        _aname = _v.get("name", _v.get("agent", ""))
                        if _aname:
                            _attn_votes[_aname] = "BUY" if _v.get("passed", False) else "SELL"

                # Outcome-weighted quality: winning trades reinforce current attention,
                # losing trades produce anti-reinforcement via inverted quality
                _outcome_mult = 1.0 if trade_won else -0.5
                _quality_override = {
                    5: max(0.0, min(1.0, 0.5 + _outcome_mult * 0.3)),
                    60: max(0.0, min(1.0, 0.5 + _outcome_mult * 0.2)),
                    240: max(0.0, min(1.0, 0.5 + _outcome_mult * 0.15)),
                }
                self._attention_feedback.update_attention(
                    quality_scores=_quality_override,
                    regime=regime,
                )
                logger.debug(
                    f"Attention feedback outcome: {pair} regime={regime} "
                    f"won={trade_won} quality={_quality_override}"
                )
            except Exception as _attn_out_err:
                logger.debug(f"Attention feedback outcome error: {_attn_out_err}")

        # Phase 82 (US-P82-002): PairModelSelector — record prediction accuracy
        if self._pair_model_selector is not None:
            try:
                _pms_model_type = model if model in ("joint", "per_pair") else "joint"
                # Use trade_won as a proxy: win=LONG/SHORT (direction matched), loss=HOLD proxy
                _pms_actual = "LONG" if trade_won else "HOLD"
                self._pair_model_selector.record_prediction(
                    pair=pair,
                    model_type=_pms_model_type,
                    predicted_direction=_pms_actual,  # best proxy without entry direction
                    actual_direction=_pms_actual,
                )
            except Exception as _pms_rec_err:
                logger.debug(f"Phase 82: PairModelSelector record_prediction error: {_pms_rec_err}")

        # Phase 82 (US-P82-004): AdaptiveLR — advance scheduler after each trade outcome
        if self._adaptive_lr is not None:
            try:
                _alr_loss = 0.0 if trade_won else 1.0
                _alr_state = self._adaptive_lr.step(loss=_alr_loss)
                logger.debug(
                    "Phase 82 (US-P82-004): AdaptiveLR step — phase=%s lr=%.6f trade=%d",
                    getattr(_alr_state, "phase", "?"),
                    getattr(_alr_state, "current_lr", 0.0),
                    getattr(_alr_state, "trade_count", 0),
                )
            except Exception as _alr_err:
                logger.debug(f"Phase 82: AdaptiveLR step error: {_alr_err}")

        # Phase 83 (US-P83-004): CounterfactualLearner — analyze losing trades for learnings
        if self._counterfactual_learner is not None and not trade_won:
            try:
                _cl_entry = {
                    "trade_id": f"{pair}_{regime}",
                    "pair": pair,
                    "pnl": pnl_pips,
                    "entry_features": gate_values or {},
                    "regime": regime,
                    "closed": True,
                }
                _cl_result = self._counterfactual_learner.analyze_closed_trade(_cl_entry)
                if _cl_result:
                    self._counterfactual_learner.append_learning(_cl_result)
                    logger.info(
                        "Phase 83 (US-P83-004): CounterfactualLearner insight for %s: %s",
                        pair, _cl_result[:120],
                    )
            except Exception as _cl_err:
                logger.debug("Phase 83: CounterfactualLearner error: %s", _cl_err)

        # Phase 85 (US-P85-001): TemporalAttentionLayer — outcome-based weight learning
        if self._temporal_attention is not None:
            try:
                _ta_result = "win" if trade_won else "loss"
                _ta_dir = "BUY"  # proxy — full direction context not available here
                self._temporal_attention.update_from_outcome(
                    trade_result=_ta_result,
                    timeframe_signals=[],  # empty = use cached signals
                    regime=regime,
                    trade_direction=_ta_dir,
                )
                logger.debug(
                    "Phase 85 (US-P85-001): TemporalAttention update_from_outcome — %s regime=%s",
                    _ta_result, regime,
                )
            except Exception as _ta_err:
                logger.debug("Phase 85: TemporalAttention update_from_outcome error: %s", _ta_err)

        # Persist after recording outcomes
        self._save_phase18_state()

    def scan_with_analysis(
        self,
        pairs: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        run_backtest: bool = True,
        run_correlation: bool = True,
        run_drift_check: bool = True,
        apply_diversification: bool = True,
    ) -> ScanResult:
        """Scan with full analysis pipeline.

        Runs the standard scan plus:
        - Quick backtests on tradeable pairs
        - Correlation analysis and diversification filter
        - Model drift detection

        Args:
            pairs: List of pairs to scan
            max_workers: Parallel workers
            run_backtest: Run quick backtests
            run_correlation: Analyze correlations
            run_drift_check: Check model drift
            apply_diversification: Filter correlated pairs

        Returns:
            ScanResult with enhanced analytics
        """
        # Initialize analysis tools
        self._init_analysis_tools()

        # Run base scan
        result = self.scan(pairs=pairs, max_workers=max_workers)

        # Build data cache for analysis
        data_cache: Dict[str, pd.DataFrame] = {}
        for pair in self._cached_results:
            if pair in self._pair_returns:
                # We need full DataFrame - fetch if needed
                df = self._fetch_pair_data(pair, self.config.lookback_candles + 200)
                if df is not None:
                    data_cache[pair] = self._compute_features(df)

        # Run quick backtests on tradeable pairs
        if run_backtest and self._backtester:
            for analysis in result.analyses:
                if analysis.is_tradeable and analysis.pair in data_cache:
                    bt_result = self._backtester.run(
                        data_cache[analysis.pair],
                        analysis.direction,
                    )
                    if bt_result.success:
                        analysis.backtest_pnl = bt_result.total_pnl
                        analysis.backtest_sharpe = bt_result.sharpe

        # Correlation analysis
        if run_correlation and self._correlation_analyzer:
            confidence_lookup = {a.pair: a.confidence for a in result.analyses}
            corr_results = self._correlation_analyzer.analyze(
                self._pair_returns,
                confidence_lookup,
            )

            # Update analyses with correlation info
            for analysis in result.analyses:
                if analysis.pair in corr_results:
                    cr = corr_results[analysis.pair]
                    analysis.correlation_group = cr.correlation_group

            # Apply diversification filter
            if apply_diversification and self._diversification_filter:
                pre_filter_pairs = {a.pair for a in result.analyses}
                result.analyses = self._diversification_filter.filter(
                    result.analyses,
                    returns_data=self._pair_returns,
                    apply_position_reduction=True,
                )
                logger.debug(
                    "Diversification filter: %d → %d tradeable pairs",
                    len(pre_filter_pairs),
                    len(result.analyses),
                )
                # US-059: Log correlation conflicts to observation log
                try:
                    filtered_out = self._diversification_filter.get_filtered_pairs()
                    if filtered_out:
                        from .automation.observation_log import ObservationLog
                        obs_log = ObservationLog()
                        kept_pairs = {a.pair for a in result.analyses}
                        for removed_pair in filtered_out:
                            # Find which kept pair caused the conflict
                            conflict_with = next(iter(kept_pairs), "unknown")
                            obs_log.log_correlation_conflict(
                                pair=removed_pair,
                                conflicting_pair=conflict_with,
                                correlation=0.7,  # threshold used by filter
                                reason="Diversification filter removed correlated pair",
                            )
                except Exception as e:
                    logger.debug("correlation conflict observation log skipped: %s", e)
                    pass

            # US-073: HRP portfolio weight optimization (replaces binary filter)
            if getattr(self.config, "use_hrp", False) and self._pair_returns:
                try:
                    from .automation.hierarchical_risk_parity import HRPOptimizer
                    hrp = HRPOptimizer(
                        linkage_method=getattr(self.config, "hrp_linkage_method", "average"),
                    )
                    active_pairs = [a.pair for a in result.analyses]
                    hrp_result = hrp.get_weights(active_pairs, self._pair_returns)
                    # Inject HRP weight into each analysis for position sizing
                    for analysis in result.analyses:
                        hrp_weight = hrp_result.weights.get(analysis.pair, 1.0 / max(len(active_pairs), 1))
                        # Store as position_size_modifier for execution layer
                        if not hasattr(analysis, "hrp_weight"):
                            analysis.hrp_weight = hrp_weight
                        else:
                            analysis.hrp_weight = hrp_weight
                    logger.info(
                        "HRP weights applied: %d pairs, effective_n=%.1f",
                        hrp_result.n_assets, hrp_result.effective_n,
                    )
                except Exception as e:
                    logger.debug("HRP optimization skipped: %s", e)

        # Model drift detection
        if run_drift_check and self._drift_detector:
            global_drift = self._drift_detector.detect()

            if global_drift.needs_retrain:
                for analysis in result.analyses:
                    analysis.needs_training = True
                    analysis.training_reason = global_drift.retrain_reason
                    analysis.model_drift_score = global_drift.drift_amount

        # Phase 59 (US-364): Record gate proximity for confidence rejects
        if self._gate_proximity is not None and result.analyses:
            try:
                self._gate_proximity.record_scan(
                    result.analyses,
                    min_confidence=float(self.config.min_confidence),
                )
            except Exception as _gpr_err:
                logger.debug(f"Phase 59: gate_proximity.record_scan failed: {_gpr_err}")

        # Phase 61 (US-373): Persist momentum scores + periodic gap analysis
        if self._momentum_recorder is not None:
            try:
                self._momentum_recorder.save_state()
            except Exception as _msr_save_err:
                logger.debug(f"Phase 61: momentum_recorder.save_state failed: {_msr_save_err}")
            if self._scan_cycle_count % 50 == 0:
                try:
                    from src.scanner.momentum_gap_analyzer import analyze as _mga_analyze
                    _mga_result = _mga_analyze(
                        self._momentum_recorder,
                        float(self.config.min_momentum),
                    )
                    _mga_cls = _mga_result.get("classification", "INSUFFICIENT_DATA")
                    if _mga_cls == "NEAR_THRESHOLD":
                        logger.warning(
                            "Phase 61 (US-373): Momentum gap NEAR_THRESHOLD — "
                            "near_miss_rate=%.1f%% | %s",
                            _mga_result.get("near_miss_rate_pct", 0.0),
                            _mga_result.get("recommendation", ""),
                        )
                    else:
                        logger.info(
                            "Phase 61 (US-373): Momentum gap analysis — "
                            "classification=%s gap_p50=%.3f pass_rate=%.2f",
                            _mga_cls,
                            _mga_result.get("gap_at_p50", 0.0),
                            _mga_result.get("pass_rate", 0.0),
                        )

                    # Phase 62 (US-376): Adaptive momentum floor — may reduce min_momentum
                    if self._adaptive_momentum_floor is not None:
                        try:
                            _new_momentum = self._adaptive_momentum_floor.tick(
                                _mga_cls, float(self.config.min_momentum)
                            )
                            if _new_momentum is not None:
                                _old_momentum = float(self.config.min_momentum)
                                self.config.min_momentum = _new_momentum
                                _amf_status = self._adaptive_momentum_floor.get_status()
                                logger.info(
                                    "Phase 62 (US-376): AdaptiveMomentumFloor — "
                                    "min_momentum reduced %.4f → %.4f "
                                    "(adaptation %d/%d)",
                                    _old_momentum, _new_momentum,
                                    _amf_status.get("adaptation_count", 0),
                                    _amf_status.get("max_adaptations", 5),
                                )
                        except Exception as _amf_tick_err:
                            logger.debug(f"Phase 62: adaptive_momentum_floor.tick() failed: {_amf_tick_err}")
                except Exception as _mga_err:
                    logger.debug(f"Phase 61: MomentumGapAnalyzer.analyze() failed: {_mga_err}")

        # Phase 62 (US-376): Persist adaptive momentum floor state
        if self._adaptive_momentum_floor is not None:
            try:
                self._adaptive_momentum_floor.save_state()
            except Exception as _amf_save_err:
                logger.debug(f"Phase 62: adaptive_momentum_floor.save_state failed: {_amf_save_err}")

        # Phase 63 (US-378 + US-379): Persist risk recorder + periodic gap analysis
        if self._risk_recorder is not None:
            try:
                self._risk_recorder.save_state()
            except Exception as _rrr_save_err:
                logger.debug(f"Phase 63: _risk_recorder.save_state failed: {_rrr_save_err}")

        # Phase 64 (US-383): Persist adaptive drawdown ceiling state
        if self._adaptive_drawdown_ceiling is not None:
            try:
                self._adaptive_drawdown_ceiling.save_state()
            except Exception as _adc_save_err:
                logger.debug(f"Phase 64: adaptive_drawdown_ceiling.save_state failed: {_adc_save_err}")
            if self._scan_cycle_count % 50 == 0:
                try:
                    from src.scanner.risk_gate_analyzer import analyze as _rga_analyze
                    _rga_result = _rga_analyze(
                        self._risk_recorder,
                        float(self.config.max_drawdown_pct),
                    )
                    _rga_cls = _rga_result.get("classification", "INSUFFICIENT_DATA")
                    if _rga_cls == "NEAR_THRESHOLD":
                        logger.warning(
                            "Phase 63 (US-379): Risk gate NEAR_THRESHOLD — "
                            "near_miss_rate=%.1f%% drawdown_kill=%.1f%% | %s",
                            _rga_result.get("near_miss_rate_pct", 0.0),
                            _rga_result.get("drawdown_kill_pct", 0.0),
                            _rga_result.get("recommendation", ""),
                        )
                    else:
                        logger.info(
                            "Phase 63 (US-379): Risk gate analysis — "
                            "classification=%s gap_p50=%.5f pass_rate=%.2f",
                            _rga_cls,
                            _rga_result.get("gap_at_p50", 0.0),
                            _rga_result.get("pass_rate", 0.0),
                        )
                    # Phase 64 (US-383): Adaptive drawdown ceiling
                    if self._adaptive_drawdown_ceiling is not None:
                        try:
                            _new_max_drawdown = self._adaptive_drawdown_ceiling.tick(
                                _rga_cls, float(self.config.max_drawdown_pct)
                            )
                            if _new_max_drawdown is not None:
                                _old_max_drawdown = float(self.config.max_drawdown_pct)
                                self.config.max_drawdown_pct = _new_max_drawdown
                                _adc_status = self._adaptive_drawdown_ceiling.get_status()
                                logger.info(
                                    "Phase 64 (US-383): AdaptiveDrawdownCeiling — "
                                    "max_drawdown_pct raised %.4f → %.4f (adaptation %d/%d)",
                                    _old_max_drawdown, _new_max_drawdown,
                                    _adc_status.get("adaptation_count", 0),
                                    _adc_status.get("max_adaptations", 3),
                                )
                        except Exception as _adc_tick_err:
                            logger.debug(f"Phase 64: adaptive_drawdown_ceiling.tick() failed: {_adc_tick_err}")
                except Exception as _rga_err:
                    logger.debug(f"Phase 63: RiskGateAnalyzer.analyze() failed: {_rga_err}")

        # Phase 66 (US-388 + US-389): Persist vote recorder + periodic gap analysis
        if self._vote_recorder is not None:
            try:
                self._vote_recorder.save_state()
            except Exception as _vsr_save_err:
                logger.debug(f"Phase 66: _vote_recorder.save_state failed: {_vsr_save_err}")
            if self._scan_cycle_count % 50 == 0:
                try:
                    from src.scanner.vote_gap_analyzer import analyze as _vga_analyze
                    _vga_result = _vga_analyze(
                        self._vote_recorder,
                        float(self.config.weighted_vote_threshold),
                    )
                    _vga_cls = _vga_result.get("classification", "INSUFFICIENT_DATA")
                    if _vga_cls == "NEAR_THRESHOLD":
                        logger.warning(
                            "Phase 66 (US-389): Agent consensus NEAR_THRESHOLD — "
                            "near_miss_rate=%.1f%% vote_pass_rate=%.2f | %s",
                            _vga_result.get("near_miss_rate_pct", 0.0),
                            _vga_result.get("pass_rate", 0.0),
                            _vga_result.get("recommendation", ""),
                        )
                    else:
                        logger.info(
                            "Phase 66 (US-389): Vote gap analysis — "
                            "classification=%s gap_p50=%.5f pass_rate=%.2f",
                            _vga_cls,
                            _vga_result.get("gap_at_p50", 0.0),
                            _vga_result.get("pass_rate", 0.0),
                        )
                    # Phase 66 (US-390): Adaptive vote threshold
                    if self._adaptive_vote_threshold is not None:
                        try:
                            _new_vote_thresh = self._adaptive_vote_threshold.tick(
                                _vga_cls, float(self.config.weighted_vote_threshold)
                            )
                            if _new_vote_thresh is not None:
                                _old_vote_thresh = float(self.config.weighted_vote_threshold)
                                self.config.weighted_vote_threshold = _new_vote_thresh
                                _avt_status = self._adaptive_vote_threshold.get_status()
                                logger.info(
                                    "Phase 66 (US-390): AdaptiveVoteThreshold — "
                                    "weighted_vote_threshold reduced %.4f → %.4f (adaptation %d/%d)",
                                    _old_vote_thresh, _new_vote_thresh,
                                    _avt_status.get("adaptation_count", 0),
                                    _avt_status.get("max_adaptations", 5),
                                )
                        except Exception as _avt_tick_err:
                            logger.debug(f"Phase 66: adaptive_vote_threshold.tick() failed: {_avt_tick_err}")
                except Exception as _vga_err:
                    logger.debug(f"Phase 66: VoteGapAnalyzer.analyze() failed: {_vga_err}")

        # Phase 66 (US-390): Persist adaptive vote threshold state
        if self._adaptive_vote_threshold is not None:
            try:
                self._adaptive_vote_threshold.save_state()
            except Exception as _avt_save_err:
                logger.debug(f"Phase 66: adaptive_vote_threshold.save_state failed: {_avt_save_err}")

        # Phase 67 (US-391 + US-392): Persist sub-inference recorder + periodic gap analysis
        if self._sub_inference_recorder is not None:
            try:
                self._sub_inference_recorder.save_state()
            except Exception as _sir_save_err:
                logger.debug(f"Phase 67: _sub_inference_recorder.save_state failed: {_sir_save_err}")
            if self._scan_cycle_count % 50 == 0:
                try:
                    from src.scanner.sub_inference_gap_analyzer import analyze as _siga_analyze
                    _siga_result = _siga_analyze(
                        self._sub_inference_recorder,
                        float(self.config.sub_inference_vote_threshold),
                    )
                    _siga_cls = _siga_result.get("classification", "INSUFFICIENT_DATA")
                    if _siga_cls == "NEAR_THRESHOLD":
                        logger.warning(
                            "Phase 67 (US-392): Sub-inference NEAR_THRESHOLD — "
                            "near_miss_rate=%.1f%% pass_rate=%.2f | %s",
                            _siga_result.get("near_miss_rate_pct", 0.0),
                            _siga_result.get("pass_rate", 0.0),
                            _siga_result.get("recommendation", ""),
                        )
                    else:
                        logger.info(
                            "Phase 67 (US-392): Sub-inference gap analysis — "
                            "classification=%s gap_p50=%.5f pass_rate=%.2f",
                            _siga_cls,
                            _siga_result.get("gap_at_p50", 0.0),
                            _siga_result.get("pass_rate", 0.0),
                        )
                    # Phase 67 (US-393): Adaptive sub-inference threshold
                    if self._adaptive_sub_inference_threshold is not None:
                        try:
                            _new_si_thresh = self._adaptive_sub_inference_threshold.tick(
                                _siga_cls, float(self.config.sub_inference_vote_threshold),
                                total_agents=int(self.config.sub_inference_window_checks),
                            )
                            if _new_si_thresh is not None:
                                _old_si_thresh = float(self.config.sub_inference_vote_threshold)
                                self.config.sub_inference_vote_threshold = _new_si_thresh
                                _asit_status = self._adaptive_sub_inference_threshold.get_status()
                                logger.info(
                                    "Phase 67 (US-393): AdaptiveSubInferenceThreshold — "
                                    "sub_inference_vote_threshold reduced %.4f → %.4f (adaptation %d/%d)",
                                    _old_si_thresh, _new_si_thresh,
                                    _asit_status.get("adaptation_count", 0),
                                    _asit_status.get("max_adaptations", 3),
                                )
                        except Exception as _asit_tick_err:
                            logger.debug(f"Phase 67: adaptive_sub_inference_threshold.tick() failed: {_asit_tick_err}")
                except Exception as _siga_err:
                    logger.debug(f"Phase 67: SubInferenceGapAnalyzer.analyze() failed: {_siga_err}")

        # Phase 67 (US-393): Persist adaptive sub-inference threshold state
        if self._adaptive_sub_inference_threshold is not None:
            try:
                self._adaptive_sub_inference_threshold.save_state()
            except Exception as _asit_save_err:
                logger.debug(f"Phase 67: adaptive_sub_inference_threshold.save_state failed: {_asit_save_err}")

        # Phase 75: Persist disagreement recorder + periodic gap analysis
        if self._disagreement_recorder is not None:
            try:
                self._disagreement_recorder.save_state()
            except Exception as _dr_save_err:
                logger.debug(f"Phase 75: _disagreement_recorder.save_state failed: {_dr_save_err}")
            if self._scan_cycle_count % 50 == 0:
                try:
                    from src.scanner.disagreement_gap_analyzer import analyze as _dga_analyze
                    _dga_result = _dga_analyze(
                        self._disagreement_recorder,
                        float(self.config.disagreement_hard_floor),
                    )
                    _dga_cls = _dga_result.get("classification", "INSUFFICIENT_DATA")
                    if _dga_cls == "NEAR_THRESHOLD":
                        logger.warning(
                            "Phase 75: Disagreement NEAR_THRESHOLD — "
                            "near_miss_rate=%.1f%% pass_rate=%.2f | %s",
                            _dga_result.get("near_miss_rate_pct", 0.0),
                            _dga_result.get("pass_rate", 0.0),
                            _dga_result.get("recommendation", ""),
                        )
                    else:
                        logger.info(
                            "Phase 75: Disagreement gap analysis — "
                            "classification=%s gap_p50=%.5f pass_rate=%.2f",
                            _dga_cls,
                            _dga_result.get("gap_at_p50", 0.0),
                            _dga_result.get("pass_rate", 0.0),
                        )
                    # Phase 75: Adaptive disagreement floor
                    if self._adaptive_disagreement_floor is not None:
                        try:
                            _new_disagree_floor = self._adaptive_disagreement_floor.tick(
                                _dga_cls, float(self.config.disagreement_hard_floor)
                            )
                            if _new_disagree_floor is not None:
                                _old_disagree_floor = float(self.config.disagreement_hard_floor)
                                self.config.disagreement_hard_floor = _new_disagree_floor
                                _adf_status = self._adaptive_disagreement_floor.get_status()
                                logger.info(
                                    "Phase 75: AdaptiveDisagreementFloor — "
                                    "disagreement_hard_floor raised %.4f → %.4f (adaptation %d/%d)",
                                    _old_disagree_floor, _new_disagree_floor,
                                    _adf_status.get("adaptation_count", 0),
                                    _adf_status.get("max_adaptations", 2),
                                )
                        except Exception as _adf_tick_err:
                            logger.debug(f"Phase 75: adaptive_disagreement_floor.tick() failed: {_adf_tick_err}")
                except Exception as _dga_err:
                    logger.debug(f"Phase 75: DisagreementGapAnalyzer.analyze() failed: {_dga_err}")

        # Phase 75: Persist adaptive disagreement floor state
        if self._adaptive_disagreement_floor is not None:
            try:
                self._adaptive_disagreement_floor.save_state()
            except Exception as _adf_save_err:
                logger.debug(f"Phase 75: adaptive_disagreement_floor.save_state failed: {_adf_save_err}")

        # Phase 82 (US-P82-004): Persist RegimeConfidenceScorer state
        if self._regime_confidence_scorer_scan is not None:
            try:
                self._regime_confidence_scorer_scan.save_state()
            except Exception as _rcs_save_err:
                logger.debug(f"Phase 82: regime_confidence_scorer_scan.save_state failed: {_rcs_save_err}")

        # Phase 82 (US-P82-002): PairModelSelector — periodic switch check every 50 cycles
        if self._pair_model_selector is not None and self._scan_cycle_count % 50 == 0:
            try:
                _pms_pairs = pair_list if pair_list else (self.config.pairs or self.config.default_pairs)
                for _pms_pair in (_pms_pairs or []):
                    _switch_to = self._pair_model_selector.check_switch(_pms_pair)
                    if _switch_to is not None:
                        self._pair_model_selector.execute_switch(_pms_pair, _switch_to)
                        logger.info(
                            "Phase 82 (US-P82-002): PairModelSelector switched %s → %s",
                            _pms_pair, _switch_to,
                        )
            except Exception as _pms_sw_err:
                logger.debug(f"Phase 82: PairModelSelector switch check error: {_pms_sw_err}")

        # Phase 83 (US-P83-002): PortfolioOptimizer — periodic Sharpe-based pair ranking
        if self._portfolio_optimizer is not None:
            try:
                if self._portfolio_optimizer.should_rotate(self._scan_cycle_count):
                    _po_rankings = self._portfolio_optimizer.rank_pairs()
                    if _po_rankings:
                        self._portfolio_optimizer.save_rankings(_po_rankings)
                        _po_top = _po_rankings[:3]
                        _po_bottom = _po_rankings[-3:] if len(_po_rankings) > 3 else []
                        logger.info(
                            "Phase 83 (US-P83-002): Portfolio rotation — top=%s bottom=%s",
                            [(r.pair, round(r.sharpe_ratio, 2)) for r in _po_top],
                            [(r.pair, round(r.sharpe_ratio, 2)) for r in _po_bottom],
                        )
                    _po_observe = self._portfolio_optimizer.get_observe_only_pairs()
                    if _po_observe:
                        logger.info(
                            "Phase 83 (US-P83-002): Observe-only pairs (low Sharpe): %s",
                            _po_observe,
                        )
            except Exception as _po_err:
                logger.debug("Phase 83: PortfolioOptimizer error: %s", _po_err)

        # Phase 83 (US-P83-004): SelfModelState — periodic health summary
        if self._self_model_state is not None and self._scan_cycle_count % 50 == 0:
            try:
                _sms_summary = self._self_model_state.get_summary()
                _sms_health = _sms_summary.get("overall_health", "unknown")
                if _sms_health == "critical":
                    logger.warning(
                        "Phase 83 (US-P83-004): SelfModel CRITICAL — worst=%s — %s",
                        _sms_summary.get("worst_pair", "?"),
                        _sms_summary.get("summary", ""),
                    )
                else:
                    logger.info(
                        "Phase 83 (US-P83-004): SelfModel health=%s pairs_monitored=%d — %s",
                        _sms_health,
                        _sms_summary.get("pairs_monitored", 0),
                        _sms_summary.get("summary", ""),
                    )
                if self._self_model_state.needs_immediate_action():
                    logger.warning(
                        "Phase 83 (US-P83-004): SelfModel NEEDS IMMEDIATE ACTION — worst_pair=%s",
                        _sms_summary.get("worst_pair", "?"),
                    )
            except Exception as _sms_err:
                logger.debug("Phase 83: SelfModelState.get_summary error: %s", _sms_err)

        # Phase 82 (US-P82-003): ChainMemory — record chain completion after each scan
        if self._chain_memory is not None:
            try:
                _cm_signals = sum(
                    1 for _a in (result.analyses if result else [])
                    if getattr(_a, "should_trade", False)
                )
                self._chain_memory.record_chain_complete({
                    "phase": self._scan_cycle_count,
                    "pairs_scanned": len(result.analyses) if result else 0,
                    "signals": _cm_signals,
                })
            except Exception as _cm_rec_err:
                logger.debug(f"Phase 82: ChainMemory record_chain_complete error: {_cm_rec_err}")

        # Phase 84 (US-P84-001): SessionSnapshotManager — save cross-session state every 100 cycles
        if self._session_snapshot is not None and self._scan_cycle_count % 100 == 0:
            try:
                _ss_weights = getattr(self._agent_team, "_learned_weights", {}) or {}
                _ss_path = self._session_snapshot.save_snapshot(
                    agent_weights=_ss_weights,
                    scan_cycles=self._scan_cycle_count,
                    profile=getattr(self.config, "profile", "balanced") or "balanced",
                )
                if _ss_path:
                    logger.info("Phase 84 (US-P84-001): Session snapshot saved → %s", _ss_path)
            except Exception as _ss_err:
                logger.debug("Phase 84: SessionSnapshotManager.save_snapshot error: %s", _ss_err)

        # Phase 84 (US-P84-002): CrisisEventGenerator — synthetic stress scenarios every 100 cycles
        if self._crisis_generator is not None and self._scan_cycle_count % 100 == 0:
            try:
                import numpy as np
                _cg_pair = next(iter(self._raw_snapshots), None) if self._raw_snapshots else None
                if _cg_pair is not None:
                    _cg_df = self._raw_snapshots[_cg_pair]
                    _cg_cols = [c for c in ("open", "high", "low", "close", "volume") if c in _cg_df.columns]
                    if len(_cg_cols) >= 4:
                        _cg_candles = _cg_df[_cg_cols].values.astype(np.float64)
                        _cg_scenarios = self._crisis_generator.generate_scenario_batch(
                            base_candles=_cg_candles, n_scenarios=5,
                        )
                        logger.info(
                            "Phase 84 (US-P84-002): Generated %d synthetic crisis scenarios from %s",
                            len(_cg_scenarios), _cg_pair,
                        )
            except Exception as _cg_err:
                logger.debug("Phase 84: CrisisEventGenerator error: %s", _cg_err)

        # Phase 84 (US-P84-003): PairTransferLearning — update stats + periodic auto_transfer
        if self._pair_transfer is not None:
            try:
                for _pt_a in (result.analyses if result else []):
                    if _pt_a and getattr(_pt_a, "pair", None) and getattr(_pt_a, "confidence", 0) > 0:
                        _pt_samples = len(self._raw_snapshots.get(_pt_a.pair, [])) if self._raw_snapshots else 0
                        self._pair_transfer.update_pair_stats(
                            pair=_pt_a.pair,
                            sample_count=max(_pt_samples, self._scan_cycle_count),
                            accuracy=float(_pt_a.confidence),
                        )
                if self._scan_cycle_count % 50 == 0:
                    _pt_count = self._pair_transfer.auto_transfer()
                    if _pt_count > 0:
                        logger.info(
                            "Phase 84 (US-P84-003): PairTransfer — %d cross-pair weight transfers applied",
                            _pt_count,
                        )
            except Exception as _pt_err:
                logger.debug("Phase 84: PairTransferLearning error: %s", _pt_err)

        # Phase 85 (US-P85-001): TemporalAttentionLayer — persist learned weights
        if self._temporal_attention is not None:
            try:
                self._temporal_attention.save_state()
            except Exception as _ta_save_err:
                logger.debug("Phase 85: temporal_attention.save_state failed: %s", _ta_save_err)

        # Phase 85 (US-P85-002): TrainingAugmenter — periodic augmented dataset generation
        if self._training_augmenter is not None and self._scan_cycle_count % 100 == 0:
            try:
                import json as _aug_json
                _aug_trades = []
                try:
                    with open("trained_data/trade_journal_rl.json", "r") as _aug_f:
                        _aug_all = _aug_json.load(_aug_f)
                    _aug_trades = [t for t in _aug_all if isinstance(t.get("outcome"), dict)]
                except Exception:
                    pass
                if len(_aug_trades) >= 10:
                    _aug_pair = next(
                        (a.pair for a in (result.analyses if result else []) if a and getattr(a, "pair", None)),
                        "EUR_USD",
                    )
                    _aug_result = self._training_augmenter.generate_augmented_dataset(
                        real_trades=_aug_trades, pair=_aug_pair,
                    )
                    self._training_augmenter.save_state()
                    logger.info(
                        "Phase 85 (US-P85-002): TrainingAugmenter — %d real + %d augmented = %d total",
                        len(_aug_trades), len(_aug_result) - len(_aug_trades), len(_aug_result),
                    )
            except Exception as _aug_err:
                logger.debug("Phase 85: TrainingAugmenter error: %s", _aug_err)

        # Phase 85 (US-P85-004): ExecutionRouter — persist routing state
        if self._execution_router is not None:
            try:
                self._execution_router.save_state()
            except Exception as _er_save_err:
                logger.debug("Phase 85: execution_router.save_state failed: %s", _er_save_err)

        # Phase 86 (US-P86-001): HRPOptimizer — periodic portfolio weight computation
        if self._hrp_optimizer is not None and self._scan_cycle_count % 50 == 0:
            try:
                _hrp_pairs = [a.pair for a in (result.analyses if result else [])
                              if a and getattr(a, "pair", None)]
                if len(_hrp_pairs) >= 2 and self._pair_returns:
                    _hrp_result = self._hrp_optimizer.get_weights(
                        pairs=_hrp_pairs, returns_data=self._pair_returns,
                    )
                    _hrp_top = sorted(
                        _hrp_result.weights.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    logger.info(
                        "Phase 86 (US-P86-001): HRP portfolio weights — top=%s effective_n=%.1f",
                        [(p, round(w, 3)) for p, w in _hrp_top],
                        getattr(_hrp_result, "effective_n", 0),
                    )
            except Exception as _hrp_err:
                logger.debug("Phase 86: HRPOptimizer error: %s", _hrp_err)

        # Phase 86 (US-P86-002): ObservationalLearner — extract patterns from synthetic output
        if self._observational_learner is not None and self._scan_cycle_count % 100 == 0:
            try:
                # Use crisis generator's generated scenarios as synthetic trades proxy
                _ol_scenarios = []
                if self._crisis_generator is not None:
                    _ol_status = self._crisis_generator.get_status()
                    _ol_count = _ol_status.get("total_generated", 0)
                    if _ol_count > 0:
                        _ol_scenarios = [{"type": "synthetic", "regime": "CRISIS", "pnl": 0.0}] * min(_ol_count, 20)
                if _ol_scenarios:
                    _ol_result = self._observational_learner.extract_patterns(_ol_scenarios)
                    _ol_n_patterns = len(getattr(_ol_result, "patterns", []))
                    if _ol_n_patterns > 0:
                        logger.info(
                            "Phase 86 (US-P86-002): ObservationalLearner — %d patterns extracted from %d synthetic",
                            _ol_n_patterns, len(_ol_scenarios),
                        )
            except Exception as _ol_err:
                logger.debug("Phase 86: ObservationalLearner error: %s", _ol_err)

        # Phase 86 (US-P86-003): LivePositionManager — evaluate open positions (observational)
        if self._position_manager is not None:
            try:
                _pm_trades = []
                if self._executor is not None:
                    try:
                        _pm_trades = self._executor.monitor_open_trades(evaluate_exits=False) or []
                    except Exception:
                        pass
                for _pm_trade in _pm_trades:
                    try:
                        from src.scanner.automation.position_manager import PositionContext
                        _pm_ctx = PositionContext(
                            trade_id=str(_pm_trade.get("trade_id", "")),
                            pair=str(_pm_trade.get("pair", "")),
                            direction=str(_pm_trade.get("direction", "BUY")),
                            entry_price=float(_pm_trade.get("entry_price", 0)),
                            current_price=float(_pm_trade.get("current_price", 0)),
                            sl_price=float(_pm_trade.get("sl_price", 0)),
                            tp_price=float(_pm_trade.get("tp_price", 0)),
                            lot_size=float(_pm_trade.get("lots", 0)),
                            bars_in_trade=int(_pm_trade.get("bars_in_trade", 1)),
                            regime=str(_pm_trade.get("regime", "NORMAL")),
                        )
                        _pm_action = self._position_manager.evaluate_position(_pm_ctx)
                        if getattr(_pm_action, "action", "HOLD") != "HOLD":
                            logger.info(
                                "Phase 86 (US-P86-003): PositionManager recommends %s for %s — %s (wp=%.2f)",
                                _pm_action.action, _pm_ctx.pair,
                                getattr(_pm_action, "reason", ""),
                                getattr(_pm_action, "win_probability", 0),
                            )
                    except Exception as _pm_eval_err:
                        logger.debug("Phase 86: PositionManager evaluate error: %s", _pm_eval_err)
            except Exception as _pm_err:
                logger.debug("Phase 86: LivePositionManager error: %s", _pm_err)

        # Phase 86 (US-P86-003): LivePositionManager — persist state
        if self._position_manager is not None:
            try:
                self._position_manager.save_state()
            except Exception as _pm_save_err:
                logger.debug("Phase 86: position_manager.save_state failed: %s", _pm_save_err)

        # Phase 88: Missing save_state calls — agent_accuracy_matrix + pair_transfer
        if self._agent_accuracy_matrix is not None:
            try:
                self._agent_accuracy_matrix.save_state()
            except Exception as _aam_err:
                logger.debug("Phase 88: agent_accuracy_matrix.save_state failed: %s", _aam_err)
        if self._pair_transfer is not None:
            try:
                self._pair_transfer.save_state()
            except Exception as _pt_save_err:
                logger.debug("Phase 88: pair_transfer.save_state failed: %s", _pt_save_err)

        return result

    def scan_incremental(
        self,
        pairs: Optional[List[str]] = None,
        interval_seconds: float = 60.0,
        max_iterations: Optional[int] = None,
        on_update: Optional[Callable[[ScanResult], None]] = None,
    ) -> None:
        """Run incremental scanning loop.

        Only re-scans pairs whose price has changed significantly.

        Args:
            pairs: List of pairs to scan
            interval_seconds: Time between scans
            max_iterations: Stop after N iterations (None = infinite)
            on_update: Callback with updated results
        """
        pair_list = pairs if pairs is not None else (self.config.pairs or self.config.default_pairs)
        iteration = 0

        while max_iterations is None or iteration < max_iterations:
            try:
                # Scan all pairs (incremental cache handles skipping)
                result = self.scan(
                    pairs=pair_list,
                    max_workers=self.config.max_workers,
                )

                # Callback with results
                if on_update:
                    on_update(result)

                # Wait for next interval
                time.sleep(interval_seconds)
                iteration += 1

            except KeyboardInterrupt:
                logger.info("Scan loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Scan iteration failed: {e}")
                time.sleep(interval_seconds)

    def get_model_health(self) -> Dict[str, Any]:
        """Report ALL runtime-loaded models so the TUI/health check reflects
        reality rather than a hardcoded "3/3" counter.

        Covers four model groups:
          1. Modular ensemble (Transformer regime/direction, TCN volatility,
             HistGB baseline) — loaded via Scanner.__init__ Phase 69 warm-up
          2. Gate evaluator (momentum/confidence/risk/transformer/meta_labeler
             /TCN gate) — loaded via _init_gate_evaluator()
          3. Agent team weights (trained_data/models/agent_weights.json)
          4. Supporting artifacts (calibration params, pair affinity, RL)

        Returns a dict with `loaded` (dict of name→bool), `count` (int,
        number of True), and `total` (int, number of expected slots).
        Any call site can summarize as f"{count}/{total}" for display.
        """
        health: Dict[str, bool] = {}

        # ── Group 1: modular ensemble ──────────────────────────────
        mei = getattr(self, "_modular_ensemble", None)
        if mei is not None:
            health["transformer_direction"] = getattr(mei, "transformer", None) is not None
            health["tcn_volatility"] = getattr(mei, "tcn", None) is not None
            # Only count optional models when they're expected/configured
            if getattr(getattr(mei, "config", None), "require_regime_model", False):
                health["transformer_regime"] = getattr(mei, "regime_model", None) is not None
            if getattr(mei, "histgb", None) is not None:
                health["histgb_baseline"] = True
        else:
            health["transformer_direction"] = False
            health["tcn_volatility"] = False

        # ── Group 2: gate evaluator ────────────────────────────────
        ge = getattr(self, "_gate_evaluator", None)
        if ge is not None and hasattr(ge, "get_model_health"):
            gate_health = ge.get_model_health()
            health.update(gate_health)
        else:
            for k in (
                "momentum_catboost",
                "momentum_xgboost",
                "momentum_lightgbm",
                "confidence_ridge",
                "risk_rf",
                "transformer_gate",
                "meta_labeler",
                "tcn_volatility_gate",
            ):
                health[k] = False

        # ── Group 3: agent team weights ────────────────────────────
        from pathlib import Path as _Path
        agent_weights = _Path("trained_data/models/agent_weights.json")
        health["agent_team_weights"] = agent_weights.exists()

        # ── Group 4: supporting artifacts (advisory) ───────────────
        support_dir = _Path("trained_data/models")
        health["calibration_params"] = (support_dir / "calibration_params.json").exists()
        health["pair_affinity_matrix"] = (support_dir / "pair_affinity_matrix.json").exists()
        health["modular_ensemble_meta"] = (support_dir / "modular_ensemble.meta.json").exists()

        # ── Group 5: RL models ─────────────────────────────────────
        health["rl_position_sizer"] = (support_dir / "rl_position_sizer.zip").exists()
        health["rl_gate_thresholds"] = (support_dir / "sac_gate_thresholds.zip").exists()
        health["rl_optimal_exit"] = (support_dir / "ppo_optimal_exit.zip").exists()

        loaded_count = sum(1 for v in health.values() if v)
        total = len(health)

        # "momentum_*" are mutually exclusive (cascade: catboost → xgb → lgbm).
        # Normalize: only count as one momentum model even if multiple flags
        # would be True (defensive; in practice only one is loaded at a time).
        momentum_loaded = any(
            health[k] for k in ("momentum_catboost", "momentum_xgboost", "momentum_lightgbm")
        )
        # Don't let the cascade double-count toward total
        momentum_total_slots = 3  # 3 cascade options but only 1 expected loaded
        if momentum_loaded:
            # Subtract the unused cascade slots from total and keep +1 for the one that loaded
            adjusted_total = total - (momentum_total_slots - 1)
            adjusted_loaded = loaded_count - (
                sum(1 for k in ("momentum_catboost", "momentum_xgboost", "momentum_lightgbm") if health[k])
                - 1
            )
        else:
            adjusted_total = total - (momentum_total_slots - 1)
            adjusted_loaded = loaded_count

        # Model freshness — when was each group last trained? Stale models
        # are a major cause of losing streaks; surface this in every snapshot
        # so Claude reflections (and TUI) can act on it.
        try:
            from src.scanner.automation.model_freshness import get_model_freshness
            freshness = get_model_freshness()
        except Exception as _fr_err:
            logger.debug("model freshness lookup failed: %s", _fr_err)
            freshness = {"status": "UNKNOWN", "oldest_age_days": None, "groups": []}

        return {
            "loaded": health,
            "count": int(adjusted_loaded),
            "total": int(adjusted_total),
            "momentum_type": getattr(ge, "_momentum_model_type", "none") if ge else "none",
            "freshness": freshness,
        }

    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information.

        Returns:
            Dict with account details (equity, trades today, etc.)
        """
        if not self._init_oanda_client():
            return {"error": "OANDA client not available"}

        try:
            raw = self._oanda.get_account_summary()
            # OANDA v20 nests data under {"account": {...}} — unwrap it.
            account = raw.get("account", raw) if isinstance(raw, dict) else raw
            balance = account.get("balance", 0)
            nav = account.get("NAV", account.get("nav", 0))
            unrealized = account.get("unrealizedPL", account.get("unrealized_pl", 0))
            margin_used = account.get("marginUsed", account.get("margin_used", 0))
            margin_available = account.get("marginAvailable", account.get("margin_available", 0))
            open_positions = account.get("openPositionCount", account.get("open_positions", 0))
            open_trades = account.get("openTradeCount", account.get("open_trades", 0))
            return {
                "balance": balance,
                "NAV": nav,
                "nav": nav,
                "unrealizedPL": unrealized,
                "unrealized_pl": unrealized,
                "marginUsed": margin_used,
                "margin_used": margin_used,
                "marginAvailable": margin_available,
                "margin_available": margin_available,
                "openPositionCount": open_positions,
                "open_positions": open_positions,
                "openTradeCount": open_trades,
                "open_trades": open_trades,
            }
        except Exception as e:
            return {"error": str(e)}

    def _init_executor(self) -> bool:
        """Initialize execution manager.

        Returns:
            True if initialized successfully
        """
        if self._executor is not None:
            return True

        try:
            exec_config = ExecutionConfig(
                account_equity=self.config.account_equity,
                risk_per_trade_pct=self.config.risk_per_trade_pct,
                leverage=self.config.leverage,
                position_sizing_enabled=self.config.position_sizing_enabled,
                aggressive_mode=self.config.aggressive_mode,
                atr_sl_multiplier=self.config.atr_sl_multiplier,
                atr_tp_multiplier=self.config.atr_tp_multiplier,
                min_sl_pips=self.config.min_sl_pips,
                max_sl_pips=self.config.max_sl_pips,
                min_tp_pips=self.config.min_tp_pips,
                max_tp_pips=self.config.max_tp_pips,
                high_prob_threshold=self.config.high_prob_threshold,
                high_prob_tp_bonus=self.config.high_prob_tp_bonus,
                max_trades_per_day=self.config.daily_trade_limit,
                max_order_attempts=getattr(self.config, "max_order_attempts", 2),
                rejection_downsize_factor=getattr(self.config, "rejection_downsize_factor", 0.5),
                high_slippage_alert_pips=getattr(self.config, "high_slippage_alert_pips", 5.0),
                fallback_tp_pips=getattr(self.config, "fallback_tp_pips", 25.0),
                fallback_atr_pips=getattr(self.config, "fallback_atr_pips", 15.0),
                min_lot_size=getattr(self.config, "min_lot_size", 0.01),
                max_lot_size=getattr(self.config, "max_lot_size", 50.0),
                trailing_stop_breakeven_pct=getattr(self.config, "trailing_stop_breakeven_pct", 0.50),
                trailing_stop_lock_pct=getattr(self.config, "trailing_stop_lock_pct", 0.75),
                # Pass max_data_age_seconds so batch scans (15 pairs, 2-5 min) don't
                # silently block every trade as stale.  Old default 60s was too tight.
                max_data_age_seconds=getattr(self.config, "max_data_age_seconds", 300.0),
                # Pass max_spread_pips so profile overrides reach the spread filter.
                # Default 3.0 works for majors; smart profile uses 4.5 for AUD/GBP crosses.
                # When --force (session filter off), widen by off_hours_spread_multiplier
                # to tolerate thinner off-hours liquidity.
                max_spread_pips=(
                    getattr(self.config, "max_spread_pips", 3.0)
                    * getattr(self.config, "off_hours_spread_multiplier", 3.0)
                    if not getattr(self.config, "enable_session_filter", True)
                    else getattr(self.config, "max_spread_pips", 3.0)
                ),
                # Broker config passthrough for ExecutionManager lazy init
                broker_type=getattr(self.config, "broker_type", "oanda"),
                ibkr_host=getattr(self.config, "ibkr_host", "127.0.0.1"),
                ibkr_port=getattr(self.config, "ibkr_port", 7497),
                ibkr_client_id=getattr(self.config, "ibkr_client_id", 1),
            )
            self._executor = ExecutionManager(
                config=exec_config,
                broker=self._broker,
                oanda_client=self._oanda,
            )
            # Phase 81 (US-P81-002): Inject PositionTimeoutManager reference into executor
            if self._position_timeout is not None:
                self._executor._position_timeout_mgr = self._position_timeout
                logger.info("Phase 81 (US-P81-002): PositionTimeoutManager injected into executor")
            # Phase 84 (US-P84-004): Inject CircuitBreaker into executor
            if self._api_retry is not None:
                self._executor._api_circuit_breaker = self._api_retry
                logger.info("Phase 84 (US-P84-004): CircuitBreaker injected into executor")
            # Phase 85 (US-P85-004): Inject ExecutionRouter into executor
            if self._execution_router is not None:
                self._executor._execution_router = self._execution_router
                logger.info("Phase 85 (US-P85-004): ExecutionRouter injected into executor")
            # Phase 86 (US-P86-004): Inject SmartExecution into executor
            if self._smart_execution is not None:
                self._executor._smart_execution = self._smart_execution
                logger.info("Phase 86 (US-P86-004): SmartExecution injected into executor")

            # Inject DynamicRiskAllocator so position sizing uses P/L distribution
            if self._dynamic_risk_allocator is not None:
                self._executor.set_dynamic_risk_allocator(self._dynamic_risk_allocator)
            # Phase 55 (US-340): Push OHLC cache to ExecutionManager so
            # Chandelier exits have real price data instead of synthetic arrays.
            if self._raw_snapshots:
                self._executor.set_ohlc_cache(self._raw_snapshots)
                logger.info(
                    "Phase 55 (US-340): Pushed %d pair OHLC snapshots to ExecutionManager",
                    len(self._raw_snapshots),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize executor: {e}")
            return False

    def execute_trades(
        self,
        analyses: List[PairAnalysis],
        max_trades: Optional[int] = None,
    ) -> List[ExecutionResult]:
        """Execute trades for tradeable pairs.

        Args:
            analyses: List of PairAnalysis results to execute
            max_trades: Maximum number of trades to execute

        Returns:
            List of ExecutionResult for each trade attempted
        """
        if not self.config.enable_execution:
            logger.warning("Execution disabled in config. Set enable_execution=True")
            return []

        if not self._init_executor():
            return []

        # ── Phase 29 (US-178): Pre-flight validation ──────────────────────
        # Verify account connectivity and basic health before placing orders.
        try:
            nav = self._executor.fetch_live_nav()
            if nav is None or nav <= 0:
                logger.warning("Pre-flight FAIL: could not fetch live NAV — skipping execution")
                return []
            # Verify we haven't exceeded portfolio risk ceiling (15% NAV)
            _open_count = len(getattr(self, "_open_positions", []))
            if _open_count >= getattr(self.config, "max_concurrent_trades", 10):
                logger.warning(
                    f"Pre-flight FAIL: {_open_count} open trades >= max_concurrent_trades"
                )
                return []
        except Exception as e:
            logger.warning(f"Pre-flight check error (non-fatal): {e}")

        # Filter to tradeable pairs only
        tradeable = [a for a in analyses if a.is_tradeable]

        if max_trades:
            tradeable = tradeable[:max_trades]

        if not tradeable:
            logger.info("No tradeable pairs to execute")
            return []

        # Phase 29 (US-178): Validate each trade's parameters before execution
        validated = []
        for a in tradeable:
            # Must have valid price, ATR, and SL/TP
            if not a.current_price or float(a.current_price) <= 0:
                logger.warning(f"Pre-flight skip {a.pair}: invalid current_price={a.current_price}")
                continue
            if not a.atr or float(a.atr) <= 0:
                logger.warning(f"Pre-flight skip {a.pair}: invalid ATR={a.atr}")
                continue
            if not a.sl_pips or float(a.sl_pips) <= 0:
                logger.warning(f"Pre-flight skip {a.pair}: invalid SL={a.sl_pips}")
                continue
            if not a.tp_pips or float(a.tp_pips) <= 0:
                logger.warning(f"Pre-flight skip {a.pair}: invalid TP={a.tp_pips}")
                continue
            # Economic calendar blackout gate (Phase 48, US-301)
            if getattr(self.config, "enable_calendar_blackout", True):
                try:
                    from src.scanner.economic_calendar import EconomicCalendarFilter, EconomicCalendarConfig
                    cal_config = EconomicCalendarConfig()
                    cal_filter = EconomicCalendarFilter(cal_config)
                    cal_result = cal_filter.check_pair(a.pair)
                    if cal_result.action == "BLACKOUT":
                        logger.warning(
                            "Calendar BLACKOUT skip %s: %s (event in %s min)",
                            a.pair, cal_result.reason,
                            cal_result.minutes_until_event,
                        )
                        continue
                    if cal_result.action == "REDUCE_SIZE":
                        if hasattr(a, "recommended_lots") and a.recommended_lots:
                            original = float(a.recommended_lots)
                            a.recommended_lots = original * cal_result.size_multiplier
                            logger.info(
                                "Calendar REDUCE_SIZE %s: %.2f -> %.2f lots (multiplier %.1f)",
                                a.pair, original, float(a.recommended_lots),
                                cal_result.size_multiplier,
                            )
                except Exception as cal_err:
                    logger.debug("Calendar filter skipped for %s: %s", a.pair, cal_err)
            # R:R ratio gate (rules/trading.md: minimum 1.2:1)
            rr = float(a.tp_pips) / float(a.sl_pips)
            # Hard safety floor from trading rules: enforce >= 1.2:1 even if
            # configured threshold is tuned lower.
            configured_min_rr = getattr(self.config, "min_risk_reward_ratio", 1.2)
            try:
                min_rr = max(1.2, float(configured_min_rr))
            except (TypeError, ValueError):
                min_rr = 1.2
            if rr < min_rr:
                logger.warning(f"Pre-flight skip {a.pair}: R:R {rr:.2f} < minimum")
                continue
            validated.append(a)
        tradeable = validated

        # Convert to execution format with full analysis context for journal
        trades = [
            {
                "pair": a.pair,
                "direction": a.direction,
                "confidence": a.confidence,
                "current_price": a.current_price,
                "atr": a.atr,
                "sl_pips": a.sl_pips,
                "tp_pips": a.tp_pips,
                "recommended_lots": (a.recommended_lots if float(a.recommended_lots) > 0 else None),
                "analysis_context": {
                    "momentum_passed": a.momentum_passed,
                    "confidence_passed": a.confidence_passed,
                    "risk_passed": a.risk_passed,
                    "gate_summary": a.gate_summary,
                    "agent_votes": a.agent_votes,
                    "agent_total": a.agent_total,
                    "agent_passed": a.agent_passed,
                    "agent_promoted": a.agent_promoted,
                    "fasttrack": getattr(a, "fasttrack", False),  # US-166
                    "weighted_vote_score": a.weighted_vote_score,
                    "agent_reasons": a.agent_reasons[:5] if a.agent_reasons else [],
                    "volatility_regime": a.volatility_regime,
                    "atr_pips": a.atr_pips,
                    "uncertainty_score": a.uncertainty_score,
                    "model_disagreement": a.model_disagreement,
                    "tcn_confidence": a.tcn_confidence,
                    "ridge_confidence": a.ridge_confidence,
                    "momentum": a.momentum,
                    "drawdown": a.drawdown,
                    "overall_score": a.overall_score,
                    "scores": getattr(a, "_ensemble_scores", {}),
                    "core_score": getattr(a, "_core_score", None),
                    "final_score": getattr(a, "_final_score", None),
                    "scan_time": a.scan_time.isoformat() if a.scan_time else None,
                    "regime_risk_modifier": (
                        self._regime_tracker.get_regime_risk_modifier()
                        if self._regime_tracker else 1.0
                    ),
                    "predictive_size_modifier": (
                        self._regime_tracker.get_predictive_size_modifier()
                        if self._regime_tracker else 1.0
                    ),
                    "regime_atr_multipliers": (
                        self.config.regime_atr_multipliers.get(
                            a.volatility_regime, {}
                        )
                        if getattr(self.config, "enable_regime_atr_adaptation", False)
                        and a.volatility_regime in {"LOW", "NORMAL", "HIGH", "EXTREME"}
                        else {}
                    ),
                    # US-066: Macro stress modifier
                    "macro_stress_modifier": (
                        self._macro_stress.get_stress_modifier()
                        if self._macro_stress else 1.0
                    ),
                    # US-064: Hour-of-day seasonality modifier
                    "seasonality_modifier": (
                        self._seasonality.get_hour_modifier(
                            a.pair, datetime.now(timezone.utc).hour
                        )
                        if self._seasonality else 1.0
                    ),
                    # Phase 22 (US-135): Pass regime-adjusted risk_pct to execution
                    "risk_pct": a.risk_pct,
                    # Tier 6 Phase 2: Store real Ridge features for MAML training
                    "ridge_features": (
                        a.metadata.get("ridge_features")
                        if isinstance(getattr(a, "metadata", None), dict)
                        else None
                    ),
                },
            }
            for a in tradeable
        ]

        # ── Trade outcome predictor (US-082): advisory pre-execution prediction ──
        if self._trade_outcome_predictor is not None:
            for trade in trades:
                try:
                    ctx = trade.get("analysis_context", {})
                    # Synthesize pre-execution features from analysis context.
                    # Post-entry features (MAE, MFE, bars) are zero at entry.
                    pred_features = {
                        "dist_to_sl_pct": 0.0,
                        "dist_to_tp_pct": 0.0,
                        "mae_pips": 0.0,
                        "mfe_pips": 0.0,
                        "bars_in_trade": 0,
                        "momentum_since_entry": float(ctx.get("momentum", 0.0) or 0.0),
                        "current_vol_vs_entry_vol": 1.0,
                    }
                    prediction = self._trade_outcome_predictor.predict(pred_features)
                    trade["analysis_context"]["outcome_prediction"] = prediction.to_dict()
                    if prediction.is_trained:
                        logger.info(
                            "US-082 %s: outcome_predictor advisory — win_prob=%.2f action=%s (%s)",
                            trade.get("pair", "?"),
                            prediction.win_probability,
                            prediction.suggested_action,
                            prediction.reason,
                        )
                except Exception as e:
                    logger.debug("US-082 outcome predictor skipped for %s: %s", trade.get("pair", "?"), e)

        # Feedback hook: log seasonality modifiers for tradeable setups to observation pipeline
        if self._seasonality is not None:
            try:
                _obs = self._get_feedback_observer()
                if _obs is not None:
                    _hour = datetime.now(timezone.utc).hour
                    for a in tradeable:
                        _season_mod = self._seasonality.get_hour_modifier(a.pair, _hour)
                        if _season_mod != 1.0:
                            _obs.log_observation(
                                pair=a.pair,
                                category="seasonality",
                                description=f"Seasonality modifier={_season_mod:.3f} hour={_hour}",
                                metadata={
                                    "modifier": round(_season_mod, 4),
                                    "hour_utc": _hour,
                                    "direction": a.direction,
                                },
                            )
            except Exception:
                pass

        # ── GROUP MOMENTUM FILTER (US-054) ────────────────────────────────
        group_scores = getattr(self, "_last_group_momentum", {})
        if self._group_momentum and group_scores:
            filtered_trades = []
            for trade in trades:
                gm_result = self._group_momentum.evaluate_trade(
                    pair=trade["pair"],
                    direction=trade["direction"],
                    confidence=trade["confidence"],
                    group_scores=group_scores,
                )
                # Inject group momentum data into analysis context
                trade["analysis_context"]["group_momentum"] = {
                    "action": gm_result.action,
                    "alignment": gm_result.alignment,
                    "base_score": gm_result.base_score,
                    "quote_score": gm_result.quote_score,
                    "position_modifier": gm_result.position_modifier,
                }

                if gm_result.action == "block":
                    logger.info(f"Group momentum BLOCKED: {gm_result.reason}")
                    # US-059: Log blocked trades to observation log
                    try:
                        from .automation.observation_log import ObservationLog
                        ObservationLog().log_group_momentum(
                            pair=trade["pair"],
                            action="block",
                            alignment=gm_result.alignment,
                            base_score=gm_result.base_score,
                            quote_score=gm_result.quote_score,
                            reason=gm_result.reason,
                        )
                    except Exception as e:
                        logger.debug("%s: group_momentum block observation log skipped: %s", trade.get('pair','?'), e)
                        pass
                    continue
                elif gm_result.action == "boost":
                    # Apply position size boost
                    if trade.get("recommended_lots") is not None:
                        trade["recommended_lots"] = round(
                            float(trade["recommended_lots"]) * gm_result.position_modifier, 2
                        )
                    logger.info(f"Group momentum BOOST: {gm_result.reason}")
                    # US-059: Log boosted trades to observation log
                    try:
                        from .automation.observation_log import ObservationLog
                        ObservationLog().log_group_momentum(
                            pair=trade["pair"],
                            action="boost",
                            alignment=gm_result.alignment,
                            base_score=gm_result.base_score,
                            quote_score=gm_result.quote_score,
                            reason=gm_result.reason,
                        )
                    except Exception as e:
                        logger.debug("%s: group_momentum boost observation log skipped: %s", trade.get('pair','?'), e)
                        pass
                filtered_trades.append(trade)
            trades = filtered_trades

        # Phase 55 (US-340): Refresh OHLC cache before execution so
        # evaluate_exits() in monitor_open_trades() has latest price data
        if self._raw_snapshots and self._executor is not None:
            self._executor.set_ohlc_cache(self._raw_snapshots)

        return self._executor.execute_trades(trades)

    def scan_and_execute(
        self,
        pairs: Optional[List[str]] = None,
        max_trades: int = 3,
    ) -> Tuple[ScanResult, List[ExecutionResult]]:
        """Scan pairs and execute top trades.

        Args:
            pairs: List of pairs to scan (uses default if None)
            max_trades: Maximum number of trades to execute

        Returns:
            Tuple of (ScanResult, list of ExecutionResult)
        """
        # Scan first
        result = self.scan(pairs=pairs)

        # Execute if enabled
        if self.config.enable_execution:
            executions = self.execute_trades(result.tradeable, max_trades=max_trades)
        else:
            executions = []

        return result, executions

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status including daily limits.

        Returns:
            Dict with execution status
        """
        if not self._init_executor():
            return {"error": "Executor not available"}

        nav, trades_today, trades_remaining = self._executor.get_account_status()
        can_trade, reason = self._executor.can_trade()

        return {
            "can_trade": can_trade,
            "reason": reason,
            "nav": nav,
            "trades_today": trades_today,
            "trades_remaining": trades_remaining,
            "daily_limit": self.config.daily_trade_limit,
            "execution_enabled": self.config.enable_execution,
        }
