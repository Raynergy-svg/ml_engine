"""
Buddy Scanner - DEPRECATED SHIM
================================
This module is deprecated. Please use src.scanner instead.

Migration guide:
    # Old usage (deprecated):
    from buddy_scanner import BuddyScanner
    scanner = BuddyScanner(config_path="config.yaml")
    results = scanner.scan(pairs=["EUR_USD"])
    
    # New usage (recommended):
    from src.scanner import Scanner, ScannerConfig
    config = ScannerConfig()
    scanner = Scanner(config=config)
    result = scanner.scan(pairs=["EUR_USD"])

Key changes:
- TCN Volatility Regime is now REQUIRED (fails if missing)
- Uses joint-trained models from trained_data/models/joint/
- ALL confidence values are 0-1 internally
- Position multipliers: 0.5x (50-65%), 1.0x (65-80%), 2.0x (80%+)
- Run 'python main.py train-joint' to train required models
"""

from __future__ import annotations

import warnings
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Issue deprecation warning on import
warnings.warn(
    "buddy_scanner is deprecated. Use 'from src.scanner import Scanner, ScannerConfig' instead. "
    "See buddy_scanner.py docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from src.scanner for backward compatibility
try:
    from src.scanner import (
        Scanner,
        ScannerConfig,
        ScanResult,
        PairAnalysis,
        ScannerError,
    )
except ImportError as e:
    logger.error(f"Failed to import from src.scanner: {e}")
    raise ImportError(
        "src.scanner module not available. Ensure src/scanner/__init__.py exists."
    ) from e


# Legacy aliases for backward compatibility
class BuddyScanner(Scanner):
    """
    DEPRECATED: Use Scanner from src.scanner instead.
    
    This is a compatibility shim that redirects to the new Scanner class.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        account_equity: Optional[float] = None,
        use_rl_sizer: bool = False,
        **kwargs: Any,
    ):
        """Initialize BuddyScanner (deprecated, use Scanner instead)."""
        warnings.warn(
            "BuddyScanner is deprecated. Use Scanner from src.scanner instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        config = ScannerConfig()
        if account_equity is not None:
            config.account_equity = account_equity
        
        super().__init__(config=config)
    
    def scan(
        self,
        pairs: Optional[List[str]] = None,
        granularity: str = "H1",
        top_n: int = 5,
        verbose: bool = False,
        prompt_train: bool = False,
        diversified: bool = False,
        **kwargs: Any,
    ) -> List["EnhancedScanResult"]:
        """
        Scan pairs and return results in legacy format.
        
        Note: Returns list of EnhancedScanResult for compatibility,
        but internally uses the new ScanResult format.
        """
        # Call parent scan method
        result: ScanResult = super().scan(pairs=pairs, granularity=granularity)
        
        # Convert to legacy EnhancedScanResult format
        legacy_results = []
        for analysis in result.ranked[:top_n] if result.ranked else []:
            legacy = EnhancedScanResult(
                pair=analysis.pair,
                direction=analysis.direction,
                confidence=analysis.confidence,
                current_price=analysis.current_price,
                atr=analysis.atr,
                volatility_percentile=analysis.regime_confidence * 100 if analysis.regime_confidence else 50.0,
                trend_strength=analysis.confidence,  # Use confidence as proxy
                entry_score=analysis.confidence,
                historical_accuracy=analysis.backtest_win_rate or 0.0,
                sl_pips=analysis.stop_loss_pips or 15.0,
                tp_pips=analysis.take_profit_pips or 30.0,
                recommended_lots=analysis.position_size_units / 100000 if analysis.position_size_units else 0.0,
                risk_pct=0.02,  # Default 2%
                gates_passed=analysis.gates_passed,
                needs_training=not analysis.gates_passed,
                overall_score=analysis.confidence,
                timestamp=analysis.timestamp,
            )
            legacy_results.append(legacy)
        
        return legacy_results
    
    def scan_continuous(
        self,
        pairs: Optional[List[str]] = None,
        granularity: str = "H1",
        interval_minutes: int = 5,
        auto_execute: bool = False,
        top_n: int = 5,
    ) -> None:
        """Continuous scan mode (deprecated, use scan_watch instead)."""
        warnings.warn(
            "scan_continuous is deprecated. Use Scanner.scan_watch() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.scan_watch(
            pairs=pairs,
            granularity=granularity,
            interval_seconds=interval_minutes * 60,
            max_iterations=0,  # Infinite
        )


class EnhancedScanResult:
    """
    DEPRECATED: Legacy result format for backward compatibility.
    
    New code should use PairAnalysis from src.scanner.results.
    """
    
    def __init__(
        self,
        pair: str,
        direction: str,
        confidence: float,
        current_price: float,
        atr: float,
        volatility_percentile: float = 50.0,
        trend_strength: float = 0.0,
        entry_score: float = 0.0,
        historical_accuracy: float = 0.0,
        sl_pips: float = 15.0,
        tp_pips: float = 30.0,
        recommended_lots: float = 0.0,
        risk_pct: float = 0.02,
        gates_passed: bool = False,
        needs_training: bool = True,
        overall_score: float = 0.0,
        timestamp: Optional[str] = None,
    ):
        self.pair = pair
        self.direction = direction
        self.confidence = confidence
        self.current_price = current_price
        self.atr = atr
        self.volatility_percentile = volatility_percentile
        self.trend_strength = trend_strength
        self.entry_score = entry_score
        self.historical_accuracy = historical_accuracy
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.recommended_lots = recommended_lots
        self.risk_pct = risk_pct
        self.gates_passed = gates_passed
        self.needs_training = needs_training
        self.overall_score = overall_score
        self.timestamp = timestamp or ""


# Legacy exports
__all__ = [
    "BuddyScanner",
    "EnhancedScanResult",
    # Also export new classes for migration
    "Scanner",
    "ScannerConfig",
    "ScanResult",
    "PairAnalysis",
    "ScannerError",
]


# ============================================================================
# ORIGINAL IMPLEMENTATION BELOW (kept for reference, not used)
