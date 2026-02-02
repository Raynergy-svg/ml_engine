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

from src.scanner.config import ScannerConfig, DEFAULT_PAIRS, PIP_VALUES, load_yaml_config, PROJECT_ROOT, DEFAULT_CONFIG_PATH
from src.scanner.results import ScanResult, PairAnalysis
from src.scanner.gates import GateEvaluator
from src.scanner.engine import Scanner
from src.scanner.display import ScannerDisplay
from src.scanner.execution import ExecutionManager, ExecutionConfig, ExecutionResult

# Analysis submodule
from src.scanner.analysis import (
    QuickBacktester,
    BacktestResult,
    CorrelationAnalyzer,
    CorrelationResult,
    DriftDetector,
    DriftResult,
)

# Filters submodule
from src.scanner.filters import (
    VolatilityFilter,
    VolatilityResult,
    DiversificationFilter,
)

# Automation submodule
from src.scanner.automation import (
    ContinuousScanner,
    IdleMaintenance,
)

__all__ = [
    # Core scanner
    "Scanner",
    "ScannerConfig",
    "ScanResult", 
    "PairAnalysis",
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
