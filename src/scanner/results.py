"""
Scanner Results Module.

Provides dataclasses for scan results:
- PairAnalysis: Analysis result for a single pair
- ScanResult: Collection of pair analyses with metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PairAnalysis:
    """Analysis result for a single FX pair.
    
    Attributes:
        pair: Instrument name (e.g., "EUR_USD")
        direction: Trade direction ("LONG", "SHORT", "HOLD")
        confidence: Overall confidence score (0-1)
        gates_passed: Whether all trading gates passed
        
        # Gate details (aligned with engine.py)
        tcn_confidence: Transformer/TCN direction confidence
        ridge_confidence: Ridge ADX confidence score (0-100)
        momentum: Momentum score (0-1)
        momentum_acceleration: Whether momentum is accelerating
        momentum_passed: Whether momentum gate passed
        confidence_score: Confidence score from Ridge model
        confidence_passed: Whether confidence gate passed
        drawdown: Expected drawdown percentage
        risk_passed: Whether risk gate passed
        
        # Market data
        current_price: Latest close price
        atr: Average True Range (raw)
        atr_pips: ATR in pips
        volatility_percentile: Volatility percentile (0-1)
        trend_strength: Trend strength (0-1)
        entry_score: Entry timing score
        
        # Position sizing
        sl_pips: Suggested stop loss in pips
        tp_pips: Suggested take profit in pips
        risk_pct: Risk percentage
        
        # Error handling
        error: Error message if scan failed, None otherwise
        scan_time: When this pair was scanned
    """
    # Core fields
    pair: str
    direction: str = "HOLD"
    confidence: float = 0.0
    gates_passed: bool = False
    
    # Gate details (matching engine.py)
    tcn_confidence: float = 0.0
    ridge_confidence: float = 0.0
    momentum: float = 0.0
    momentum_acceleration: bool = False
    momentum_passed: bool = False
    confidence_score: float = 0.0
    confidence_passed: bool = False
    drawdown: float = 0.0
    risk_passed: bool = False
    
    # Legacy gate fields for compatibility
    tcn_probability: float = 0.5
    xgb_momentum: float = 0.0
    rf_drawdown: float = 0.0
    confidence_gate_passed: bool = False
    momentum_gate_passed: bool = False
    risk_gate_passed: bool = False
    volatility_gate_passed: bool = False
    
    # Market data
    current_price: float = 0.0
    atr: float = 0.0
    atr_pips: float = 0.0
    volatility_percentile: float = 0.5
    volatility_regime: str = "UNKNOWN"
    trend_strength: float = 0.0
    entry_score: float = 0.5
    
    # Position sizing
    recommended_lots: float = 0.0
    sl_pips: float = 15.0
    tp_pips: float = 30.0
    risk_pct: float = 0.02
    risk_amount: float = 0.0
    
    # Model info
    model_type: str = "unknown"
    has_pair_model: bool = False
    pair_model_accuracy: float = 0.0
    
    # Training status (from buddy_scanner)
    needs_training: bool = False
    training_reason: Optional[str] = None
    
    # Analytics (from buddy_scanner - Phase 2 will populate these)
    backtest_pnl: float = 0.0
    backtest_sharpe: float = 0.0
    correlation_group: Optional[str] = None
    model_drift_score: float = 0.0
    memory_accuracy: float = 0.0
    
    # Error handling
    error: Optional[str] = None
    scan_time_ms: float = 0.0
    scan_time: Optional[datetime] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_tradeable(self) -> bool:
        """Check if this pair has a valid trade signal."""
        return self.gates_passed and self.direction is not None and self.error is None
    
    @property
    def overall_score(self) -> float:
        """Combined score for ranking pairs."""
        if self.error:
            return 0.0
        
        return (
            self.confidence * 0.35 +
            (self.ridge_confidence / 100) * 0.25 +
            self.xgb_momentum * 0.20 +
            self.trend_strength * 0.10 +
            (1.0 - min(self.rf_drawdown * 20, 1.0)) * 0.10  # Lower drawdown = higher score
        )
    
    @property
    def gate_summary(self) -> str:
        """Human-readable gate status."""
        passed = 0
        total = 3
        
        if self.confidence_passed or self.confidence_gate_passed:
            passed += 1
        if self.momentum_passed or self.momentum_gate_passed:
            passed += 1
        if self.risk_passed or self.risk_gate_passed:
            passed += 1
        
        return f"{passed}/{total}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pair": self.pair,
            "direction": self.direction,
            "confidence": self.confidence,
            "gates_passed": self.gates_passed,
            "tcn_probability": self.tcn_probability,
            "ridge_confidence": self.ridge_confidence,
            "xgb_momentum": self.xgb_momentum,
            "rf_drawdown": self.rf_drawdown,
            "current_price": self.current_price,
            "atr": self.atr,
            "volatility_regime": self.volatility_regime,
            "recommended_lots": self.recommended_lots,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "risk_amount": self.risk_amount,
            "model_type": self.model_type,
            "error": self.error,
            "scan_time_ms": self.scan_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "is_tradeable": self.is_tradeable,
        }


@dataclass
class ScanResult:
    """Collection of pair analyses from a scan.
    
    Attributes:
        analyses: List of PairAnalysis results
        tradeable: Filtered list of tradeable pairs (gates passed)
        non_tradeable: Filtered list of non-tradeable pairs
        errors: Filtered list of pairs with errors
        
        # Metadata
        model_type: Type of model used (catboost, xgboost, ensemble, technical)
        granularity: Timeframe used (H1, M15, etc.)
        scan_time_total_ms: Total scan time in milliseconds
        account_equity: Account balance used for sizing
        timestamp: When scan was performed
        config_used: Scanner config summary
    """
    analyses: List[PairAnalysis] = field(default_factory=list)
    
    # Model metadata
    model_type: str = "unknown"
    granularity: str = "H1"
    
    # Timing metadata
    scan_time_total_ms: float = 0.0
    account_equity: float = 0.0
    scan_time: datetime = field(default_factory=datetime.now)  # Alias for timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    config_summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tradeable_pairs(self) -> List[PairAnalysis]:
        """Get tradeable pairs sorted by score (alias for tradeable)."""
        return self.tradeable
    
    @property
    def tradeable(self) -> List[PairAnalysis]:
        """Get tradeable pairs sorted by score."""
        return sorted(
            [a for a in self.analyses if a.is_tradeable],
            key=lambda x: x.overall_score,
            reverse=True
        )
    
    @property
    def non_tradeable(self) -> List[PairAnalysis]:
        """Get non-tradeable pairs (gates failed but no error)."""
        return [a for a in self.analyses if not a.is_tradeable and a.error is None]
    
    @property
    def errors(self) -> List[PairAnalysis]:
        """Get pairs that had errors during scan."""
        return [a for a in self.analyses if a.error is not None]
    
    @property
    def success_rate(self) -> float:
        """Percentage of pairs scanned without error."""
        if not self.analyses:
            return 0.0
        return (len(self.analyses) - len(self.errors)) / len(self.analyses)
    
    def get_top_n(self, n: int = 5) -> List[PairAnalysis]:
        """Get top N tradeable pairs by score."""
        return self.tradeable[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tradeable_count": len(self.tradeable),
            "non_tradeable_count": len(self.non_tradeable),
            "error_count": len(self.errors),
            "scan_time_total_ms": self.scan_time_total_ms,
            "account_equity": self.account_equity,
            "timestamp": self.timestamp.isoformat(),
            "success_rate": self.success_rate,
            "tradeable": [a.to_dict() for a in self.tradeable],
            "non_tradeable": [a.to_dict() for a in self.non_tradeable],
            "errors": [a.to_dict() for a in self.errors],
        }
