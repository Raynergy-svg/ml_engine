"""
Scanner Results Module.

Provides dataclasses for scan results:
- PairAnalysis: Analysis result for a single pair (enhanced with BuddyScanner fields)
- ScanResult: Collection of pair analyses with metadata

ALL confidence values use 0-1 scale internally.
Display methods convert to percentage format (e.g., 65%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _format_pct(value: float) -> str:
    """Format 0-1 value as percentage string."""
    return f"{value * 100:.0f}%"


@dataclass
class PairAnalysis:
    """Analysis result for a single FX pair.
    
    ALL confidence/probability fields use 0-1 scale internally.
    Use format methods for percentage display.
    
    Attributes:
        pair: Instrument name (e.g., "EUR_USD")
        direction: Trade direction ("LONG", "SHORT", "HOLD")
        confidence: Overall confidence score (0-1)
        gates_passed: Whether all trading gates passed
        
        # TCN Forward Volatility (FIRST GATE) - predicts FUTURE regime
        volatility_regime: int (0=QUIET_NEXT, 1=STABLE_NEXT, 2=ACTIVE_NEXT, 3=EXTREME_NEXT)
        volatility_regime_name: str ("QUIET_NEXT", "STABLE_NEXT", "ACTIVE_NEXT", "EXTREME_NEXT")
        regime_confidence: Model confidence in regime prediction (0-1)
        extreme_warning: True if regime is EXTREME_NEXT (news event risk)
        
        # Gate details (ALL 0-1 SCALE)
        tcn_confidence: Transformer/TCN direction confidence (0-1)
        ridge_confidence: Ridge ADX confidence score (0-1, normalized from 0-100)
        momentum: Momentum score (0-1)
        drawdown: Expected drawdown percentage (0-1)
        
        # Position sizing
        position_size_units: Calculated position size in units
        sl_pips: Stop loss in pips (ATR-based)
        tp_pips: Take profit in pips (ATR-based)
        position_multiplier: Confidence-tiered multiplier (0.5x/1.0x/2.0x)
        
        # Analytics (from BuddyScanner)
        backtest_win_rate: Quick backtest win rate
        backtest_pnl: Quick backtest P&L
        correlation_group: Correlation cluster identifier
        
        # Trade status
        is_tradeable: Combined check for all gates + no errors
        block_reason: Why trade was blocked (if any)
    """
    # Core fields
    pair: str
    direction: str = "HOLD"
    confidence: float = 0.0  # 0-1 scale
    gates_passed: bool = False
    
    # ==========================================================================
    # TCN FORWARD VOLATILITY (FIRST GATE - predicts FUTURE regime)
    # ==========================================================================
    volatility_regime: int = 0  # 0=QUIET_NEXT, 1=STABLE_NEXT, 2=ACTIVE_NEXT, 3=EXTREME_NEXT
    volatility_regime_name: str = "UNKNOWN"  # Human readable
    regime_confidence: float = 0.0  # 0-1 scale
    extreme_warning: bool = False  # True if EXTREME_NEXT regime detected
    volatility_gate_passed: bool = False
    
    # ==========================================================================
    # GATE DETAILS (ALL 0-1 SCALE)
    # ==========================================================================
    # Transformer direction gate
    tcn_confidence: float = 0.0  # 0-1 scale
    tcn_probability: float = 0.5  # Raw probability (0-1)
    tcn_probability_raw: float = 0.5  # Raw (uncalibrated) probability
    tcn_calibrated: bool = False  # True if probability was calibrated
    tcn_gate_passed: bool = False
    
    # Ridge confidence gate (ADX-based)
    ridge_confidence: float = 0.0  # 0-1 scale (normalized from 0-100)
    confidence_score: float = 0.0  # Alias for compatibility
    confidence_passed: bool = False
    confidence_gate_passed: bool = False  # Alias
    
    # XGBoost momentum gate
    momentum: float = 0.0  # 0-1 scale
    xgb_momentum: float = 0.0  # Alias for compatibility
    momentum_acceleration: bool = False
    momentum_passed: bool = False
    momentum_gate_passed: bool = False  # Alias
    
    # RF risk gate
    drawdown: float = 0.0  # Expected drawdown (0-1 scale, e.g., 0.025 = 2.5%)
    rf_drawdown: float = 0.0  # Alias for compatibility
    risk_passed: bool = False
    risk_gate_passed: bool = False  # Alias
    
    # ==========================================================================
    # MARKET DATA
    # ==========================================================================
    current_price: float = 0.0
    atr: float = 0.0  # Raw ATR
    atr_pips: float = 0.0  # ATR in pips
    volatility_percentile: float = 0.5  # 0-1 scale
    trend_strength: float = 0.0  # 0-1 scale (from ADX)
    entry_score: float = 0.5  # 0-1 scale
    
    # ==========================================================================
    # POSITION SIZING (from BuddyScanner)
    # ==========================================================================
    position_size_units: int = 0  # Calculated units to trade
    recommended_lots: float = 0.0  # Position in lots (units/100000)
    sl_pips: float = 15.0  # Stop loss (ATR-based, clamped to min/max)
    tp_pips: float = 30.0  # Take profit (ATR-based, clamped to min/max)
    risk_pct: float = 0.02  # Risk per trade (0-1 scale, e.g., 0.02 = 2%)
    risk_amount: float = 0.0  # Risk amount in account currency
    position_multiplier: float = 1.0  # Confidence-tiered (0.5x/1.0x/2.0x)
    
    # ==========================================================================
    # MODEL INFO
    # ==========================================================================
    model_type: str = "unknown"  # catboost, xgboost, ensemble, technical
    has_pair_model: bool = False  # Pair-specific model exists
    pair_model_accuracy: float = 0.0  # Historical accuracy (0-1)
    
    # ==========================================================================
    # TRAINING STATUS (from BuddyScanner)
    # ==========================================================================
    needs_training: bool = False  # Model needs retraining
    training_reason: Optional[str] = None  # Why retraining needed
    
    # ==========================================================================
    # ANALYTICS (from BuddyScanner - enhanced)
    # ==========================================================================
    backtest_win_rate: float = 0.0  # Quick backtest win rate (0-1)
    backtest_pnl: float = 0.0  # Quick backtest P&L
    backtest_sharpe: float = 0.0  # Sharpe ratio from backtest
    correlation_group: Optional[str] = None  # Correlation cluster ID
    model_drift_score: float = 0.0  # Drift amount (0-1)
    memory_accuracy: float = 0.0  # Historical accuracy from memory client (0-1)
    
    # ==========================================================================
    # TRADE STATUS
    # ==========================================================================
    is_tradeable_override: Optional[bool] = None  # Manual override
    block_reason: Optional[str] = None  # Why trade was blocked
    error: Optional[str] = None  # Error message if scan failed
    
    # ==========================================================================
    # TIMING
    # ==========================================================================
    scan_time_ms: float = 0.0
    scan_time: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_tradeable(self) -> bool:
        """Check if this pair has a valid trade signal.
        
        Requires:
        - All gates passed (including TCN Volatility)
        - Valid direction (not HOLD)
        - No errors
        - No block reason
        """
        if self.is_tradeable_override is not None:
            return self.is_tradeable_override
        return (
            self.gates_passed and 
            self.direction in ("LONG", "SHORT") and 
            self.error is None and
            self.block_reason is None
        )
    
    @property
    def overall_score(self) -> float:
        """Combined score for ranking pairs (0-1 scale)."""
        if self.error or self.block_reason:
            return 0.0
        
        # All inputs are now 0-1 scale
        return (
            self.confidence * 0.35 +
            self.ridge_confidence * 0.25 +  # Already 0-1
            self.momentum * 0.20 +
            self.trend_strength * 0.10 +
            (1.0 - min(self.drawdown * 20, 1.0)) * 0.10  # Lower drawdown = higher score
        )
    
    @property
    def gate_summary(self) -> str:
        """Human-readable gate status."""
        passed = sum([
            self.volatility_gate_passed,
            self.confidence_passed or self.confidence_gate_passed,
            self.momentum_passed or self.momentum_gate_passed,
            self.risk_passed or self.risk_gate_passed,
        ])
        total = 4  # VOL + CONF + MOM + RISK
        return f"{passed}/{total}"
    
    def format_confidence(self) -> str:
        """Format overall confidence as percentage."""
        return _format_pct(self.confidence)
    
    def format_ridge_confidence(self) -> str:
        """Format Ridge confidence as percentage."""
        return _format_pct(self.ridge_confidence)
    
    def format_momentum(self) -> str:
        """Format momentum as percentage."""
        return _format_pct(self.momentum)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pair": self.pair,
            "direction": self.direction,
            "confidence": self.confidence,
            "confidence_pct": self.format_confidence(),
            "gates_passed": self.gates_passed,
            # TCN Volatility
            "volatility_regime": self.volatility_regime,
            "volatility_regime_name": self.volatility_regime_name,
            "regime_confidence": self.regime_confidence,
            "extreme_warning": self.extreme_warning,
            # Gates (0-1 scale)
            "tcn_probability": self.tcn_probability,
            "ridge_confidence": self.ridge_confidence,
            "ridge_confidence_pct": self.format_ridge_confidence(),
            "xgb_momentum": self.momentum,
            "rf_drawdown": self.drawdown,
            # Market data
            "current_price": self.current_price,
            "atr": self.atr,
            "atr_pips": self.atr_pips,
            # Position sizing
            "position_size_units": self.position_size_units,
            "recommended_lots": self.recommended_lots,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "risk_amount": self.risk_amount,
            "position_multiplier": self.position_multiplier,
            # Analytics
            "backtest_win_rate": self.backtest_win_rate,
            "backtest_pnl": self.backtest_pnl,
            "correlation_group": self.correlation_group,
            # Status
            "model_type": self.model_type,
            "block_reason": self.block_reason,
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
        
        # Enhanced metadata (from BuddyScanner)
        tcn_model_loaded: Whether TCN Volatility model was available
        joint_models_used: Whether joint models were used exclusively
    """
    analyses: List[PairAnalysis] = field(default_factory=list)
    
    # Model metadata
    model_type: str = "unknown"
    granularity: str = "H1"
    
    # TCN/Joint model status
    tcn_model_loaded: bool = False
    joint_models_used: bool = True
    
    # Timing metadata
    scan_time_total_ms: float = 0.0
    account_equity: float = 0.0
    nav: float = 0.0  # Live NAV from OANDA
    scan_time: datetime = field(default_factory=datetime.now)  # Alias for timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    config_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Trade limits (from BuddyScanner)
    trades_today: int = 0
    trades_remaining: int = 3
    daily_limit: int = 3
    
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
    def blocked_by_volatility(self) -> List[PairAnalysis]:
        """Get pairs blocked by QUIET_NEXT or EXTREME_NEXT volatility regime.
        
        New forward-looking logic:
        - QUIET_NEXT (0): Blocked - expect insufficient price movement
        - STABLE_NEXT (1): Allowed - expect moderate, tradeable volatility
        - ACTIVE_NEXT (2): Allowed - expect high, best for entries
        - EXTREME_NEXT (3): Blocked - expect news event, high risk
        """
        return [
            a for a in self.analyses 
            if a.volatility_regime not in {1, 2} and a.error is None  # Block if not STABLE or ACTIVE
        ]
    
    @property
    def extreme_volatility_warnings(self) -> List[PairAnalysis]:
        """Get tradeable pairs with EXTREME volatility warning."""
        return [a for a in self.tradeable if a.extreme_warning]
    
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
            "blocked_by_volatility_count": len(self.blocked_by_volatility),
            "extreme_warnings_count": len(self.extreme_volatility_warnings),
            "error_count": len(self.errors),
            "tcn_model_loaded": self.tcn_model_loaded,
            "joint_models_used": self.joint_models_used,
            "scan_time_total_ms": self.scan_time_total_ms,
            "account_equity": self.account_equity,
            "nav": self.nav,
            "trades_today": self.trades_today,
            "trades_remaining": self.trades_remaining,
            "daily_limit": self.daily_limit,
            "timestamp": self.timestamp.isoformat(),
            "success_rate": self.success_rate,
            "tradeable": [a.to_dict() for a in self.tradeable],
            "non_tradeable": [a.to_dict() for a in self.non_tradeable],
            "errors": [a.to_dict() for a in self.errors],
        }
