"""
Scanner Analysis Subpackage.

Provides analytics modules for enhanced scanning:
- QuickBacktester: 50-candle rapid backtests with Sharpe calculation
- CorrelationAnalyzer: Pair correlation analysis with Union-Find clustering
- DriftDetector: Model accuracy drift detection
"""

from src.scanner.analysis.backtest import QuickBacktester, BacktestResult
from src.scanner.analysis.correlation import CorrelationAnalyzer, CorrelationResult
from src.scanner.analysis.drift import DriftDetector, DriftResult

__all__ = [
    "QuickBacktester",
    "BacktestResult",
    "CorrelationAnalyzer", 
    "CorrelationResult",
    "DriftDetector",
    "DriftResult",
]
