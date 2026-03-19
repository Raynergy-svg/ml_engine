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
        "PIP_VALUES": ("src.scanner.config", "PIP_VALUES"),
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
    "PIP_VALUES",
    "load_yaml_config",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
]
