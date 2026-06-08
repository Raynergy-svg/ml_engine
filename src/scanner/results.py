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
    risk_pct: float = 0.05  # Phase 81: was 0.02, practice account $100K
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

    # Sub-inference confirmation (opportunity mode)
    agent_votes: int = 0
    agent_total: int = 0
    agent_score: float = 0.0
    agent_passed: bool = False
    agent_promoted: bool = False
    weighted_vote_score: float = 0.0
    weighted_vote_threshold: float = 0.0
    agent_context_key: Optional[str] = None
    agent_reasons: List[Dict[str, Any]] = field(default_factory=list)
    agent_reason_codes: List[str] = field(default_factory=list)
    why_trade: List[str] = field(default_factory=list)
    why_no_trade: List[str] = field(default_factory=list)
    uncertainty_score: float = 0.0
    confidence_variance: float = 0.0
    model_disagreement: float = 0.0
    execution_quality_passed: bool = True
    execution_quality_score: float = 1.0
    spread_pips: float = 0.0
    est_slippage_pips: float = 0.0
    liquidity_score: float = 1.0
    blocked_by_circuit_breaker: bool = False
    circuit_breakers_triggered: List[str] = field(default_factory=list)
    master_pair: Optional[str] = None

    # Phase 98 kept this as observability for near-miss setups, but live
    # execution now treats agent_passed=False as a fast veto. The execution
    # layer has always hard-blocked agent_passed=False; keeping the scanner
    # property aligned prevents "selected for execution, then rejected" churn.
    agent_soft_penalty_applied: bool = False

    # 2026-05-10 (Agent B observability fix): override flag honored by the
    # `is_tradeable` setter. Pre-fix `is_tradeable` was a read-only @property,
    # yet engine.py:4693/4745/4778 did `result.is_tradeable = True` — which
    # raised AttributeError. The two wrapped sites (US-127, US-132) swallowed
    # the error under `except Exception` and logged at debug only, so the
    # `logger.info("re-qualified")` lines NEVER fired (log-format dishonesty).
    # The third site (US-142 fast-track) raised an uncaught AttributeError
    # caught only by the outer scan-level handler, dropping the pair silently.
    # The setter (see below) now sets this flag and the property short-circuits
    # to True when it is set — making the three `logger.info` lines honest.
    _force_tradeable_override: bool = False

    # Gate kill reason (Phase 90 US-403): set at each gate rejection site.
    # None when pair is tradeable. Values: 'confidence_gate', 'momentum_gate',
    # 'disagreement_gate', 'drawdown_gate', 'accuracy_gate', 'other'.
    kill_reason: Optional[str] = None

    # Mythos audit 2026-05-04 — operator-facing rejection reason consumed by
    # embedded_scanner.py:521 when populating brain-feed reject lines.
    # Read pattern was `analysis.rejection_reason or "gates"` — pre-fix this
    # raised AttributeError because the field didn't exist on PairAnalysis,
    # crashing every Scanner.scan_pair invocation:
    #   ERROR: Scan failed for GBP_USD: 'PairAnalysis' object has no attribute 'rejection_reason'
    # which is why every scan returned 0/N tradeable with momentum=0.000 in
    # the virtual_trade_logger. None default means "no operator-overlaid
    # reason set"; the consumer falls back to "gates" via `or "gates"`.
    rejection_reason: Optional[str] = None

    # Error handling
    error: Optional[str] = None
    scan_time_ms: float = 0.0
    scan_time: Optional[datetime] = None

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_tradeable(self) -> bool:
        """Check if this pair has a valid trade signal.

        Safety invariants enforced here (last line of defense before execute_trades):
        - agent_passed must be True
        - model_disagreement must be <= max_model_disagreement (from config, default 0.50)
        - volatility_regime must not be UNKNOWN

        Phase 76: Uses _max_model_disagreement (set by agent team from config) instead
        of hardcoded 0.30. When soft_uncertainty_blocking=True, disagreement between
        the hard_floor and max is penalized (confidence reduced) rather than blocked.

        2026-05-14: agent_passed is hard-aligned with execution. The order
        layer rejects agent_passed=False as a final safety invariant, so the
        scanner must veto those setups before they enter execution selection.

        2026-05-10 (Agent B observability fix): honors `_force_tradeable_override`
        flag set by the `is_tradeable` setter. Required by engine.py re-qualify
        sites (US-127 ThresholdOptimizer, US-132 EXTREME regime, US-142 fast-track)
        that need to override the base computation. Pre-fix those sites silently
        AttributeError'd because `is_tradeable` had no setter.
        """
        # 2026-05-10: explicit override takes precedence over base computation.
        # Still requires direction in {LONG, SHORT} and no error — those are
        # invariants no override can violate.
        if (
            getattr(self, "_force_tradeable_override", False)
            and self.agent_passed
            and self.direction in {"LONG", "SHORT"}
            and self.error is None
        ):
            return True

        _max_disagree = float(getattr(self, "_max_model_disagreement", 0.50))
        agent_ok = bool(self.agent_passed)

        return (
            self.gates_passed
            and self.direction in {"LONG", "SHORT"}
            and self.error is None
            and not self.blocked_by_circuit_breaker
            and bool(self.execution_quality_passed)
            and agent_ok
            and self.model_disagreement <= _max_disagree
            and self.volatility_regime.upper() != "UNKNOWN"
        )

    @is_tradeable.setter
    def is_tradeable(self, value: bool) -> None:
        """Override the computed `is_tradeable` verdict.

        Added 2026-05-10 (Agent B observability fix). Pre-existing engine.py
        re-qualification sites (US-127 ThresholdOptimizer, US-132 EXTREME
        regime, US-142 fast-track) do `result.is_tradeable = True` to re-admit
        a pair that the base computation rejected. Without this setter, the
        assignment raises AttributeError — silently swallowed by two of the
        three sites (the `logger.info("...re-qualified")` lines never fired).

        Setting `True` flips `_force_tradeable_override` so the property
        returns True when direction is LONG/SHORT, no error is present, and
        agent_passed=True. Setting `False` clears the override, restoring the
        base computation.

        Note: this is a one-way override for re-qualification — it does NOT
        force `gates_passed=True`. Callers that need the gate fields aligned
        should set those explicitly (engine.py:4694/4746/4779 already do).
        """
        self._force_tradeable_override = bool(value)

    @property
    def overall_score(self) -> float:
        """Combined score for ranking pairs."""
        if self.error:
            return 0.0

        momentum_val = float(self.xgb_momentum) if float(self.xgb_momentum) > 0 else float(self.momentum)
        drawdown_val = float(self.rf_drawdown) if float(self.rf_drawdown) > 0 else float(self.drawdown)

        base_score = (
            self.confidence * 0.35 +
            (self.ridge_confidence / 100) * 0.25 +
            momentum_val * 0.20 +
            self.trend_strength * 0.10 +
            (1.0 - min(drawdown_val * 20, 1.0)) * 0.10  # Lower drawdown = higher score
        )
        if self.agent_total > 0:
            base_score = base_score * 0.88 + min(max(self.agent_score, 0.0), 1.0) * 0.12
        if self.weighted_vote_score > 0:
            base_score = base_score * 0.90 + min(max(self.weighted_vote_score, 0.0), 1.0) * 0.10
        return min(max(base_score, 0.0), 1.0)

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
            "agent_votes": self.agent_votes,
            "agent_total": self.agent_total,
            "agent_score": self.agent_score,
            "agent_passed": self.agent_passed,
            "agent_soft_penalty_applied": self.agent_soft_penalty_applied,
            "agent_promoted": self.agent_promoted,
            "weighted_vote_score": self.weighted_vote_score,
            "weighted_vote_threshold": self.weighted_vote_threshold,
            "agent_context_key": self.agent_context_key,
            "agent_reasons": self.agent_reasons,
            "agent_reason_codes": self.agent_reason_codes,
            "why_trade": self.why_trade,
            "why_no_trade": self.why_no_trade,
            "uncertainty_score": self.uncertainty_score,
            "confidence_variance": self.confidence_variance,
            "model_disagreement": self.model_disagreement,
            "execution_quality_passed": self.execution_quality_passed,
            "execution_quality_score": self.execution_quality_score,
            "spread_pips": self.spread_pips,
            "est_slippage_pips": self.est_slippage_pips,
            "liquidity_score": self.liquidity_score,
            "blocked_by_circuit_breaker": self.blocked_by_circuit_breaker,
            "circuit_breakers_triggered": self.circuit_breakers_triggered,
            "master_pair": self.master_pair,
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
