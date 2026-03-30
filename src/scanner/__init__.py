"""
FX Scanner Module - Clean, refactored multi-pair scanner.

This module provides a robust scanner for FX pairs using:
- CatBoost (primary) + XGBoost (fallback) for momentum gates
- ThreadPoolExecutor for parallel pair scanning
- Rich Live displays for real-time progress
- Incremental caching to reduce API calls in watch mode
- Trade execution with daily limits and position sizing
- Analysis tools: backtest, correlation, drift detection
- Filters: volatility regime, diversification

Usage:
    from src.scanner import Scanner, ScannerConfig, ScanResult

    scanner = Scanner()
    results = scanner.scan(pairs=["EUR_USD", "GBP_USD"])

    # Watch mode with incremental updates
    scanner.scan_watch(pairs=["EUR_USD"], interval_seconds=300)

    # Execute trades
    from src.scanner import ExecutionManager, ExecutionConfig

    executor = ExecutionManager()
    result = executor.execute_trade(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.72,
        current_price=1.0850,
        atr=0.0015,
    )

    # Analysis tools
    from src.scanner.analysis import QuickBacktester, CorrelationAnalyzer, DriftDetector

    # Filters
    from src.scanner.filters import VolatilityFilter, DiversificationFilter
"""

import importlib as _importlib
import logging as _logging

_logger = _logging.getLogger(__name__)

# --- Lazy imports ---
# Eager imports of display/engine/analysis pull in heavy deps (rich, tensorflow, etc.)
# which breaks lightweight consumers (automation modules, CLI tools, tests).
# All public names are still available via `from src.scanner import X` thanks to __getattr__.

def __getattr__(name):
    """Lazy import: only load heavy modules when their symbols are actually accessed."""
    _LAZY_MAP = {
        # Config (lightweight — safe to import eagerly but kept lazy for consistency)
        "ScannerConfig": ("src.scanner.config", "ScannerConfig"),
        "DEFAULT_PAIRS": ("src.scanner.config", "DEFAULT_PAIRS"),
        # NOTE: PIP_VALUES has been moved to InstrumentRegistry — use get_registry() instead
        "load_yaml_config": ("src.scanner.config", "load_yaml_config"),
        "PROJECT_ROOT": ("src.scanner.config", "PROJECT_ROOT"),
        "DEFAULT_CONFIG_PATH": ("src.scanner.config", "DEFAULT_CONFIG_PATH"),
        # Results
        "ScanResult": ("src.scanner.results", "ScanResult"),
        "PairAnalysis": ("src.scanner.results", "PairAnalysis"),
        # Agents
        "AgentDecisionContext": ("src.scanner.agents", "AgentDecisionContext"),
        "AgentVerdict": ("src.scanner.agents", "AgentVerdict"),
        "ScannerAgentTeam": ("src.scanner.agents", "ScannerAgentTeam"),
        # Gates
        "GateEvaluator": ("src.scanner.gates", "GateEvaluator"),
        # Engine
        "Scanner": ("src.scanner.engine", "Scanner"),
        # Display (requires rich)
        "ScannerDisplay": ("src.scanner.display", "ScannerDisplay"),
        # Execution
        "ExecutionManager": ("src.scanner.execution", "ExecutionManager"),
        "ExecutionConfig": ("src.scanner.execution", "ExecutionConfig"),
        "ExecutionResult": ("src.scanner.execution", "ExecutionResult"),
        # Analysis
        "QuickBacktester": ("src.scanner.analysis", "QuickBacktester"),
        "BacktestResult": ("src.scanner.analysis", "BacktestResult"),
        "CorrelationAnalyzer": ("src.scanner.analysis", "CorrelationAnalyzer"),
        "CorrelationResult": ("src.scanner.analysis", "CorrelationResult"),
        "DriftDetector": ("src.scanner.analysis", "DriftDetector"),
        "DriftResult": ("src.scanner.analysis", "DriftResult"),
        # Filters
        "VolatilityFilter": ("src.scanner.filters", "VolatilityFilter"),
        "VolatilityResult": ("src.scanner.filters", "VolatilityResult"),
        "DiversificationFilter": ("src.scanner.filters", "DiversificationFilter"),
        # Automation
        "ContinuousScanner": ("src.scanner.automation", "ContinuousScanner"),
        "IdleMaintenance": ("src.scanner.automation", "IdleMaintenance"),
        # Phase 48: Trade Intelligence & Adaptive Filtering
        "SpreadFilter": ("src.scanner.spread_filter", "SpreadFilter"),
        "SpreadFilterConfig": ("src.scanner.spread_filter", "SpreadFilterConfig"),
        "SpreadCheckResult": ("src.scanner.spread_filter", "SpreadCheckResult"),
        "EconomicCalendarFilter": ("src.scanner.economic_calendar", "EconomicCalendarFilter"),
        "EconomicCalendarConfig": ("src.scanner.economic_calendar", "EconomicCalendarConfig"),
        "EconomicEvent": ("src.scanner.economic_calendar", "EconomicEvent"),
        "StrategyFitnessDetector": ("src.scanner.strategy_fitness", "StrategyFitnessDetector"),
        "PairPerformanceTracker": ("src.scanner.pair_blacklist", "PairPerformanceTracker"),
        "TradeAttributionEngine": ("src.scanner.trade_attribution", "TradeAttributionEngine"),
        "WalkForwardOptimizer": ("src.scanner.walkforward_optimizer", "WalkForwardOptimizer"),
        # Phase 50: Data Activation & Seeding
        "JournalAnalyzer": ("src.scanner.journal_analyzer", "JournalAnalyzer"),
        "CalibrationSeeder": ("src.scanner.calibration_seeder", "CalibrationSeeder"),
        "FitnessSeeder": ("src.scanner.fitness_seeder", "FitnessSeeder"),
        # Phase 51: Win-Rate Engine
        "IsotonicCalibrator": ("src.scanner.isotonic_calibrator", "IsotonicCalibrator"),
        "TrancheTracker": ("src.scanner.tranche_tracker", "TrancheTracker"),
        "PatternGate": ("src.scanner.pattern_gate", "PatternGate"),
        "WalkForwardRetrainer": ("src.scanner.walkforward_retrainer", "WalkForwardRetrainer"),
        "SetupQualityFilter": ("src.scanner.setup_quality", "SetupQualityFilter"),
        # Phase 52: Pipeline Integration & Adaptive Intelligence
        "ConfidenceDecomposer": ("src.scanner.confidence_decomposer", "ConfidenceDecomposer"),
        "AdaptiveRRCalculator": ("src.scanner.adaptive_rr", "AdaptiveRRCalculator"),
        "AdaptiveRRConfig": ("src.scanner.adaptive_rr", "AdaptiveRRConfig"),
        "AdaptiveRRResult": ("src.scanner.adaptive_rr", "AdaptiveRRResult"),
        "DrawdownAdapter": ("src.scanner.drawdown_adapter", "DrawdownAdapter"),
        "DrawdownAdapterConfig": ("src.scanner.drawdown_adapter", "DrawdownAdapterConfig"),
        "DrawdownCheckResult": ("src.scanner.drawdown_adapter", "DrawdownCheckResult"),
        "GateAttributionEngine": ("src.scanner.gate_attribution", "GateAttributionEngine"),
        "GateAttributionConfig": ("src.scanner.gate_attribution", "GateAttributionConfig"),
        "GateDecision": ("src.scanner.gate_attribution", "GateDecision"),
        "GateStats": ("src.scanner.gate_attribution", "GateStats"),
        "PairSpecificBayesianWeights": ("src.scanner.pair_bayesian_weights", "PairSpecificBayesianWeights"),
        "PairWeightsConfig": ("src.scanner.pair_bayesian_weights", "PairWeightsConfig"),
        "PairWeightSample": ("src.scanner.pair_bayesian_weights", "PairWeightSample"),
        "SignalFreshnessGuard": ("src.scanner.signal_freshness", "SignalFreshnessGuard"),
        "FreshnessConfig": ("src.scanner.signal_freshness", "FreshnessConfig"),
        "FreshnessResult": ("src.scanner.signal_freshness", "FreshnessResult"),
        "UnifiedConfidencePipeline": ("src.scanner.confidence_pipeline", "UnifiedConfidencePipeline"),
        "ConfidencePipelineConfig": ("src.scanner.confidence_pipeline", "ConfidencePipelineConfig"),
        "CalibrationMonitor": ("src.scanner.confidence_pipeline", "CalibrationMonitor"),
        # Phase 53: WFE Stability Monitor
        "WFEStabilityMonitor": ("src.scanner.wfe_monitor", "WFEStabilityMonitor"),
        "WFEMonitorConfig": ("src.scanner.wfe_monitor", "WFEMonitorConfig"),
        # Phase 54: Dynamic Gate Threshold Optimizer
        "GateThresholdOptimizer": ("src.scanner.gate_threshold_optimizer", "GateThresholdOptimizer"),
        "GateThresholdOptimizerConfig": ("src.scanner.gate_threshold_optimizer", "GateThresholdOptimizerConfig"),
        "ThresholdRecommendation": ("src.scanner.gate_threshold_optimizer", "ThresholdRecommendation"),
        # Phase 54: Feature Drift Detector
        "FeatureDriftDetector": ("src.scanner.feature_drift", "FeatureDriftDetector"),
        "FeatureDriftConfig": ("src.scanner.feature_drift", "FeatureDriftConfig"),
        "DriftCheckResult": ("src.scanner.feature_drift", "DriftCheckResult"),
        # Phase 54: Trade Cluster Analyzer
        "TradeClusterAnalyzer": ("src.scanner.trade_clusters", "TradeClusterAnalyzer"),
        "TradeClusterConfig": ("src.scanner.trade_clusters", "TradeClusterConfig"),
        "ClusterInfo": ("src.scanner.trade_clusters", "ClusterInfo"),
        # Phase 54: Cluster Confidence Gate
        "ClusterConfidenceGate": ("src.scanner.cluster_gate", "ClusterConfidenceGate"),
        "ClusterGateResult": ("src.scanner.cluster_gate", "ClusterGateResult"),
        # Phase 54: Regime Confidence Scorer
        "RegimeConfidenceScorer": ("src.scanner.regime_confidence", "RegimeConfidenceScorer"),
        "RegimeConfidenceConfig": ("src.scanner.regime_confidence", "RegimeConfidenceConfig"),
        "RegimeScoreResult": ("src.scanner.regime_confidence", "RegimeScoreResult"),
        # Phase 55: Safe I/O
        "safe_json_write": ("src.scanner.safe_io", "safe_json_write"),
        "safe_json_read": ("src.scanner.safe_io", "safe_json_read"),
        "safe_jsonl_append": ("src.scanner.safe_io", "safe_jsonl_append"),
        # Phase 55: Retry & Resilient Save (US-342)
        "retry_with_backoff": ("src.scanner.safe_io", "retry_with_backoff"),
        "resilient_save": ("src.scanner.safe_io", "resilient_save"),
        # Phase 55: Disagreement Tracker (US-343)
        "DisagreementTracker": ("src.scanner.disagreement_tracker", "DisagreementTracker"),
    }
    if name in _LAZY_MAP:
        module_path, attr_name = _LAZY_MAP[name]
        try:
            module = _importlib.import_module(module_path)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as e:
            _logger.debug(f"Lazy import {module_path}.{attr_name} failed: {e}")
            raise ImportError(f"Cannot import {attr_name} from {module_path}: {e}") from e
    raise AttributeError(f"module 'src.scanner' has no attribute {name!r}")

__all__ = [
    # Core scanner
    "Scanner",
    "ScannerConfig",
    "ScanResult",
    "PairAnalysis",
    "AgentDecisionContext",
    "AgentVerdict",
    "ScannerAgentTeam",
    "ScannerDisplay",
    "GateEvaluator",
    # Execution
    "ExecutionManager",
    "ExecutionConfig",
    "ExecutionResult",
    # Analysis
    "QuickBacktester",
    "BacktestResult",
    "CorrelationAnalyzer",
    "CorrelationResult",
    "DriftDetector",
    "DriftResult",
    # Filters
    "VolatilityFilter",
    "VolatilityResult",
    "DiversificationFilter",
    # Automation
    "ContinuousScanner",
    "IdleMaintenance",
    # Config helpers
    "DEFAULT_PAIRS",
    # NOTE: PIP_VALUES moved to InstrumentRegistry — use get_registry() instead
    "load_yaml_config",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
    # Phase 48: Trade Intelligence & Adaptive Filtering
    "SpreadFilter",
    "SpreadFilterConfig",
    "SpreadCheckResult",
    "EconomicCalendarFilter",
    "EconomicCalendarConfig",
    "EconomicEvent",
    "StrategyFitnessDetector",
    "PairPerformanceTracker",
    "TradeAttributionEngine",
    "WalkForwardOptimizer",
    # Phase 50: Data Activation & Seeding
    "JournalAnalyzer",
    "CalibrationSeeder",
    "FitnessSeeder",
    # Phase 51: Win-Rate Engine
    "IsotonicCalibrator",
    "TrancheTracker",
    "PatternGate",
    "WalkForwardRetrainer",
    "SetupQualityFilter",
    # Phase 52: Pipeline Integration & Adaptive Intelligence
    "ConfidenceDecomposer",
    "AdaptiveRRCalculator",
    "AdaptiveRRConfig",
    "AdaptiveRRResult",
    "DrawdownAdapter",
    "DrawdownAdapterConfig",
    "DrawdownCheckResult",
    # Phase 53: Gate Attribution & Pair-Specific Tuning
    "GateAttributionEngine",
    "GateAttributionConfig",
    "GateDecision",
    "GateStats",
    "PairSpecificBayesianWeights",
    "PairWeightsConfig",
    "PairWeightSample",
    "SignalFreshnessGuard",
    "FreshnessConfig",
    "FreshnessResult",
    "UnifiedConfidencePipeline",
    "ConfidencePipelineConfig",
    "CalibrationMonitor",
    # Phase 53: WFE Stability Monitor
    "WFEStabilityMonitor",
    "WFEMonitorConfig",
    # Phase 54: Dynamic Gate Threshold Optimizer
    "GateThresholdOptimizer",
    "GateThresholdOptimizerConfig",
    "ThresholdRecommendation",
    # Phase 54: Feature Drift Detector
    "FeatureDriftDetector",
    "FeatureDriftConfig",
    "DriftCheckResult",
    # Phase 54: Trade Cluster Analyzer
    "TradeClusterAnalyzer",
    "TradeClusterConfig",
    "ClusterInfo",
    # Phase 54: Cluster Confidence Gate
    "ClusterConfidenceGate",
    "ClusterGateResult",
    # Phase 54: Regime Confidence Scorer
    "RegimeConfidenceScorer",
    "RegimeConfidenceConfig",
    "RegimeScoreResult",
]
