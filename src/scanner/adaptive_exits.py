"""
Adaptive Exit Strategy Manager for Buddy FX Trading Bot.

Combines multiple exit strategies with priority-based decision making:
- Chandelier Exit: ATR-based trailing stop from peak price
- Partial Profit Taking: Scale-out at configurable R-multiples
- Volatility-Adjusted Trailing Stop: Regime-dependent ATR multipliers
- Time-Based Exit: Maximum holding period by volatility regime
- Confidence-Decay Exit: Exit when calibrated confidence drops below threshold

All evaluations are stateless (context passed in via TradeContext dataclass).
Thread-safe for concurrent trade monitoring.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExitStrategyConfig:
    """Configuration for all exit strategies."""

    # Chandelier Exit parameters
    chandelier_period: int = 22
    chandelier_atr_multiplier: float = 3.0

    # Partial Profit Taking configuration
    enable_partial_profits: bool = True
    partial_tranches: List[Dict] = field(
        default_factory=lambda: [
            {"ratio": 0.50, "r_target": 1.0, "type": "fixed"},
            {"ratio": 0.25, "r_target": 1.5, "type": "trailing"},
            {"ratio": 0.25, "r_target": 2.0, "type": "fixed_or_time"},
        ]
    )

    # Volatility-adjusted trailing stop multipliers (by regime)
    vol_trail_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "LOW": 1.5,
            "NORMAL": 2.5,
            "HIGH": 3.5,
            "EXTREME": 4.0,
        }
    )

    # Time-based exit: max bars in trade by regime
    time_exit_bars: Dict[str, int] = field(
        default_factory=lambda: {
            "LOW": 40,
            "NORMAL": 25,
            "HIGH": 15,
            "EXTREME": 8,
        }
    )

    # Confidence decay exit threshold (exit if confidence < this)
    confidence_exit_threshold: float = 0.35

    # Confidence decay rates (multiplicative per bar, by regime)
    confidence_decay_rates: Dict[str, float] = field(
        default_factory=lambda: {
            "LOW": 0.99,
            "NORMAL": 0.97,
            "HIGH": 0.95,
            "EXTREME": 0.92,
        }
    )

    def validate(self) -> None:
        """Validate configuration values at load time."""
        if self.chandelier_period < 5:
            raise ValueError(
                f"chandelier_period must be >= 5, got {self.chandelier_period}"
            )
        if self.chandelier_atr_multiplier <= 0:
            raise ValueError(
                f"chandelier_atr_multiplier must be > 0, got {self.chandelier_atr_multiplier}"
            )
        if not 0 < self.confidence_exit_threshold < 1:
            raise ValueError(
                f"confidence_exit_threshold must be between 0 and 1, got {self.confidence_exit_threshold}"
            )
        if len(self.partial_tranches) == 0 and self.enable_partial_profits:
            logger.warning(
                "Partial profits enabled but no tranches configured; will be disabled"
            )
        for regime in ["LOW", "NORMAL", "HIGH", "EXTREME"]:
            if regime not in self.vol_trail_multipliers:
                raise ValueError(f"Missing vol_trail_multiplier for regime {regime}")
            if regime not in self.time_exit_bars:
                raise ValueError(f"Missing time_exit_bars for regime {regime}")
            if regime not in self.confidence_decay_rates:
                raise ValueError(f"Missing confidence_decay_rate for regime {regime}")
            if self.vol_trail_multipliers[regime] <= 0:
                raise ValueError(
                    f"vol_trail_multiplier[{regime}] must be > 0, got {self.vol_trail_multipliers[regime]}"
                )
            if self.time_exit_bars[regime] < 1:
                raise ValueError(
                    f"time_exit_bars[{regime}] must be >= 1, got {self.time_exit_bars[regime]}"
                )
            if not 0 < self.confidence_decay_rates[regime] <= 1:
                raise ValueError(
                    f"confidence_decay_rate[{regime}] must be in (0, 1], got {self.confidence_decay_rates[regime]}"
                )


@dataclass
class TradeContext:
    """All information needed to evaluate exit for an open trade."""

    # Entry parameters
    entry_price: float
    entry_time_bar: int  # Bar index at entry
    current_bar: int  # Current bar index
    direction: str  # "BUY" or "SELL"

    # Price data (recent bars, sufficient for chandelier lookback)
    prices_close: np.ndarray  # Close prices
    prices_high: np.ndarray  # High prices
    prices_low: np.ndarray  # Low prices

    # Risk parameters
    sl_pips: float  # Original stop loss in pips (always positive)
    tp_pips: float  # Original take profit in pips (always positive)
    atr_current: float  # Current ATR value

    # Current state
    current_price: float
    highest_price_since_entry: float  # Max high for long trades
    lowest_price_since_entry: float  # Min low for short trades

    # Calibrated confidence
    initial_confidence: float  # Confidence at entry (0 to 1)
    current_confidence: float  # Current calibrated confidence (may have decayed)

    # Volatility regime
    regime_name: str  # "LOW", "NORMAL", "HIGH", or "EXTREME"

    # Partial profit state
    tranches_closed: List[int] = field(
        default_factory=list
    )  # Indices of tranches already closed

    def validate(self) -> None:
        """Validate trade context at evaluation time."""
        if self.direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got {self.direction}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be > 0, got {self.entry_price}")
        if self.current_price <= 0:
            raise ValueError(f"current_price must be > 0, got {self.current_price}")
        if self.sl_pips <= 0:
            raise ValueError(f"sl_pips must be > 0, got {self.sl_pips}")
        if self.tp_pips <= 0:
            raise ValueError(f"tp_pips must be > 0, got {self.tp_pips}")
        if self.atr_current <= 0:
            raise ValueError(f"atr_current must be > 0, got {self.atr_current}")
        if not 0 <= self.initial_confidence <= 1:
            raise ValueError(
                f"initial_confidence must be in [0, 1], got {self.initial_confidence}"
            )
        if not 0 <= self.current_confidence <= 1:
            raise ValueError(
                f"current_confidence must be in [0, 1], got {self.current_confidence}"
            )
        if self.regime_name not in ("LOW", "NORMAL", "HIGH", "EXTREME"):
            raise ValueError(
                f"regime_name must be LOW/NORMAL/HIGH/EXTREME, got {self.regime_name}"
            )
        if self.current_bar < self.entry_time_bar:
            raise ValueError(
                f"current_bar ({self.current_bar}) < entry_time_bar ({self.entry_time_bar})"
            )
        if len(self.prices_close) == 0:
            raise ValueError("prices_close cannot be empty")
        if len(self.prices_high) != len(self.prices_close):
            raise ValueError("prices_high length mismatch")
        if len(self.prices_low) != len(self.prices_close):
            raise ValueError("prices_low length mismatch")

    def bars_held(self) -> int:
        """Number of bars this trade has been open."""
        return self.current_bar - self.entry_time_bar

    def unrealized_pips(self) -> float:
        """Calculate unrealized P&L in pips (direction-aware)."""
        if self.direction == "BUY":
            return self.current_price - self.entry_price
        else:  # SELL
            return self.entry_price - self.current_price

    def unrealized_r(self) -> float:
        """Calculate risk-reward multiple (how many risk units in profit/loss)."""
        if self.sl_pips == 0:
            return 0.0
        return self.unrealized_pips() / self.sl_pips


@dataclass
class ExitAction:
    """Represents a decision to hold or exit a trade."""

    action: str  # "HOLD", "TIGHTEN_SL", "TAKE_PARTIAL", "EXIT_FULL"
    reason: str  # Human-readable reason for the action
    strategy: str  # Strategy that triggered: "chandelier", "partial_profit", "vol_trail", "time_exit", "confidence_decay"

    # Action parameters (populated based on action type)
    new_sl_price: Optional[float] = None  # For TIGHTEN_SL actions
    partial_close_ratio: Optional[float] = None  # For TAKE_PARTIAL (e.g., 0.5 = close 50%)

    # Metadata and priority
    confidence: float = 0.0  # Confidence in this decision (0 to 1)
    urgency: str = "normal"  # "low", "normal", "high", "critical"
    metadata: Dict = field(default_factory=dict)  # Additional context

    def is_exit(self) -> bool:
        """Check if this is any form of exit decision."""
        return self.action in ("TAKE_PARTIAL", "EXIT_FULL")


# ============================================================================
# ADAPTIVE EXIT MANAGER
# ============================================================================


class AdaptiveExitManager:
    """
    Multi-strategy exit system for FX trades.

    Evaluates five exit strategies concurrently and returns the highest-priority
    action. All evaluation is stateless; trade state is passed in via TradeContext.

    Thread-safe for concurrent trade monitoring.
    """

    def __init__(self, config: Optional[ExitStrategyConfig] = None):
        """
        Initialize exit manager with configuration.

        Args:
            config: ExitStrategyConfig instance. If None, uses defaults.

        Raises:
            ValueError: If config validation fails.
        """
        self.config = config or ExitStrategyConfig()
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Invalid exit strategy config: {e}")
            raise

    def evaluate_exit(self, ctx: TradeContext) -> ExitAction:
        """
        Evaluate all exit strategies for the given trade context.

        Evaluates strategies in priority order:
        1. Confidence Decay Exit (highest priority - protects against stale ideas)
        2. Time Exit (hard backstop on holding period)
        3. Chandelier Exit (technical exit based on peak)
        4. Volatility-Adjusted Trail (regime-aware tightening)
        5. Partial Profit Taking (lowest priority - scaling out)

        Returns the highest-priority non-HOLD action, or HOLD if all strategies agree.

        Args:
            ctx: TradeContext with all trade state and market data.

        Returns:
            ExitAction with action, reason, strategy, and parameters.

        Raises:
            ValueError: If context validation fails.
        """
        try:
            ctx.validate()
        except ValueError as e:
            logger.error(f"Invalid trade context: {e}")
            raise

        actions = []

        # Evaluate all strategies (order doesn't matter; priority is in merge)
        try:
            actions.append(self._evaluate_confidence_decay(ctx))
            actions.append(self._evaluate_time_exit(ctx))
            actions.append(self._evaluate_chandelier(ctx))
            actions.append(self._evaluate_vol_adjusted_trail(ctx))
            actions.append(self._evaluate_partial_profit(ctx))
        except Exception as e:
            logger.error(f"Unexpected error evaluating exit strategies: {e}", exc_info=True)
            # Return a critical exit on unexpected error rather than silent failure
            return ExitAction(
                action="EXIT_FULL",
                reason=f"Critical error in exit evaluation: {str(e)}",
                strategy="error_handler",
                confidence=0.0,
                urgency="critical",
            )

        # Merge actions by priority
        result = self._priority_merge(actions)
        logger.debug(
            f"Exit evaluation for {ctx.direction} trade (bars_held={ctx.bars_held()}): "
            f"action={result.action}, strategy={result.strategy}, confidence={result.confidence:.2f}"
        )
        return result

    def _evaluate_confidence_decay(self, ctx: TradeContext) -> ExitAction:
        """
        Evaluate confidence decay exit.

        Applies exponential decay to initial confidence over time held.
        If decayed confidence falls below threshold, exit the trade.

        Formula: decayed = initial_confidence * (decay_rate ^ bars_held)

        Args:
            ctx: TradeContext with confidence and regime.

        Returns:
            ExitAction with EXIT_FULL if threshold breached, else HOLD.
        """
        bars = ctx.bars_held()
        decay_rate = self.config.confidence_decay_rates[ctx.regime_name]

        # Compute decayed confidence (exponential decay)
        decayed_confidence = ctx.initial_confidence * (decay_rate ** bars)

        metadata = {
            "initial_confidence": ctx.initial_confidence,
            "decay_rate": decay_rate,
            "bars_held": bars,
            "decayed_confidence": decayed_confidence,
        }

        if decayed_confidence < self.config.confidence_exit_threshold:
            reason = (
                f"Confidence decayed from {ctx.initial_confidence:.2f} to {decayed_confidence:.2f} "
                f"(threshold {self.config.confidence_exit_threshold}) over {bars} bars in {ctx.regime_name} regime"
            )
            logger.info(f"Confidence decay exit triggered: {reason}")
            return ExitAction(
                action="EXIT_FULL",
                reason=reason,
                strategy="confidence_decay",
                confidence=1.0 - decayed_confidence,  # Inverse of confidence = exit urgency
                urgency="high",
                metadata=metadata,
            )

        return ExitAction(
            action="HOLD",
            reason="Confidence still above decay exit threshold",
            strategy="confidence_decay",
            confidence=decayed_confidence,
            metadata=metadata,
        )

    def _evaluate_time_exit(self, ctx: TradeContext) -> ExitAction:
        """
        Evaluate time-based exit backstop.

        If trade held longer than max_bars for the current regime, exit.
        Prevents indefinite holding of losing/uncertain positions.

        Args:
            ctx: TradeContext with bars held and regime.

        Returns:
            ExitAction with EXIT_FULL if max time exceeded, else HOLD.
        """
        bars = ctx.bars_held()
        max_bars = self.config.time_exit_bars[ctx.regime_name]

        metadata = {
            "bars_held": bars,
            "max_bars_regime": max_bars,
            "regime": ctx.regime_name,
        }

        if bars >= max_bars:
            reason = (
                f"Trade held for {bars} bars, max for {ctx.regime_name} regime is {max_bars}. "
                f"Time-based exit backstop triggered."
            )
            logger.info(f"Time exit triggered: {reason}")
            return ExitAction(
                action="EXIT_FULL",
                reason=reason,
                strategy="time_exit",
                confidence=0.95,  # High confidence in time-based rule
                urgency="high",
                metadata=metadata,
            )

        pct_time_used = bars / max_bars
        return ExitAction(
            action="HOLD",
            reason=f"Time backstop not yet hit ({bars}/{max_bars} bars)",
            strategy="time_exit",
            confidence=pct_time_used,  # Confidence increases as time limit approaches
            metadata=metadata,
        )

    def _evaluate_chandelier(self, ctx: TradeContext) -> ExitAction:
        """
        Evaluate Chandelier Exit (ATR-based trailing from peak).

        For LONG: CE = max(high[-period:]) - ATR * multiplier
        Exit if current_price < CE (broken below the chandelier)

        For SHORT: CE = min(low[-period:]) + ATR * multiplier
        Exit if current_price > CE (broken above the chandelier)

        The chandelier exit is a technical trailing stop that moves with peaks/troughs.

        Args:
            ctx: TradeContext with price data and ATR.

        Returns:
            ExitAction with TIGHTEN_SL or HOLD.
        """
        period = min(self.config.chandelier_period, len(ctx.prices_high))
        atr_mult = self.config.chandelier_atr_multiplier
        trail_distance = ctx.atr_current * atr_mult

        metadata = {
            "period": period,
            "atr_multiplier": atr_mult,
            "trail_distance": trail_distance,
        }

        if ctx.direction == "BUY":
            # For longs: chandelier = highest high in period - ATR_distance
            highest = float(np.max(ctx.prices_high[-period:]))
            chandelier = highest - trail_distance
            metadata["highest_high"] = highest
            metadata["chandelier_level"] = chandelier

            if ctx.current_price < chandelier:
                reason = (
                    f"Chandelier exit: price {ctx.current_price:.5f} below "
                    f"chandelier level {chandelier:.5f} (highest={highest:.5f} - ATR*{atr_mult})"
                )
                logger.info(f"Chandelier exit triggered (LONG): {reason}")
                return ExitAction(
                    action="EXIT_FULL",
                    reason=reason,
                    strategy="chandelier",
                    confidence=0.85,
                    urgency="normal",
                    metadata=metadata,
                )
            else:
                return ExitAction(
                    action="HOLD",
                    reason=f"Price {ctx.current_price:.5f} still above chandelier {chandelier:.5f}",
                    strategy="chandelier",
                    metadata=metadata,
                )

        else:  # SELL
            # For shorts: chandelier = lowest low in period + ATR_distance
            lowest = float(np.min(ctx.prices_low[-period:]))
            chandelier = lowest + trail_distance
            metadata["lowest_low"] = lowest
            metadata["chandelier_level"] = chandelier

            if ctx.current_price > chandelier:
                reason = (
                    f"Chandelier exit: price {ctx.current_price:.5f} above "
                    f"chandelier level {chandelier:.5f} (lowest={lowest:.5f} + ATR*{atr_mult})"
                )
                logger.info(f"Chandelier exit triggered (SHORT): {reason}")
                return ExitAction(
                    action="EXIT_FULL",
                    reason=reason,
                    strategy="chandelier",
                    confidence=0.85,
                    urgency="normal",
                    metadata=metadata,
                )
            else:
                return ExitAction(
                    action="HOLD",
                    reason=f"Price {ctx.current_price:.5f} still below chandelier {chandelier:.5f}",
                    strategy="chandelier",
                    metadata=metadata,
                )

    def _evaluate_vol_adjusted_trail(self, ctx: TradeContext) -> ExitAction:
        """
        Evaluate volatility-adjusted trailing stop.

        Tightens trailing stop based on regime. Higher volatility = wider trail.
        For LONG: trail_stop = highest_since_entry - (ATR * regime_multiplier)
        For SHORT: trail_stop = lowest_since_entry + (ATR * regime_multiplier)

        If breached, suggests tightening the SL or exiting.

        Args:
            ctx: TradeContext with peak prices, ATR, and regime.

        Returns:
            ExitAction with TIGHTEN_SL, EXIT_FULL, or HOLD.
        """
        multiplier = self.config.vol_trail_multipliers[ctx.regime_name]
        trail_distance = ctx.atr_current * multiplier

        metadata = {
            "regime": ctx.regime_name,
            "multiplier": multiplier,
            "trail_distance": trail_distance,
        }

        if ctx.direction == "BUY":
            trail_stop = ctx.highest_price_since_entry - trail_distance
            metadata["highest_since_entry"] = ctx.highest_price_since_entry
            metadata["trail_stop"] = trail_stop

            if ctx.current_price < trail_stop:
                # Trail stop breached
                reason = (
                    f"Vol-adjusted trail breach (LONG): price {ctx.current_price:.5f} < "
                    f"trail_stop {trail_stop:.5f} (highest={ctx.highest_price_since_entry:.5f} - {multiplier}*ATR)"
                )
                logger.info(f"Vol-adjusted trail exit triggered: {reason}")
                return ExitAction(
                    action="EXIT_FULL",
                    reason=reason,
                    strategy="vol_trail",
                    confidence=0.75,
                    urgency="normal",
                    metadata=metadata,
                )
            else:
                return ExitAction(
                    action="HOLD",
                    reason=f"Price {ctx.current_price:.5f} still above vol trail {trail_stop:.5f}",
                    strategy="vol_trail",
                    metadata=metadata,
                )

        else:  # SELL
            trail_stop = ctx.lowest_price_since_entry + trail_distance
            metadata["lowest_since_entry"] = ctx.lowest_price_since_entry
            metadata["trail_stop"] = trail_stop

            if ctx.current_price > trail_stop:
                # Trail stop breached
                reason = (
                    f"Vol-adjusted trail breach (SHORT): price {ctx.current_price:.5f} > "
                    f"trail_stop {trail_stop:.5f} (lowest={ctx.lowest_price_since_entry:.5f} + {multiplier}*ATR)"
                )
                logger.info(f"Vol-adjusted trail exit triggered: {reason}")
                return ExitAction(
                    action="EXIT_FULL",
                    reason=reason,
                    strategy="vol_trail",
                    confidence=0.75,
                    urgency="normal",
                    metadata=metadata,
                )
            else:
                return ExitAction(
                    action="HOLD",
                    reason=f"Price {ctx.current_price:.5f} still below vol trail {trail_stop:.5f}",
                    strategy="vol_trail",
                    metadata=metadata,
                )

    def _evaluate_partial_profit(self, ctx: TradeContext) -> ExitAction:
        """
        Evaluate partial profit taking (scale-out).

        Checks each configured tranche (defined in config.partial_tranches).
        Each tranche has:
        - ratio: % of position to close (0.5 = close 50% of remaining)
        - r_target: risk-reward multiple at which to close
        - type: "fixed" (close at r_target), "trailing" (close if touch r_target then trail),
                or "fixed_or_time" (close at r_target OR after certain bars)

        Returns TAKE_PARTIAL for the first eligible tranche not yet closed.

        Args:
            ctx: TradeContext with current R and tranche state.

        Returns:
            ExitAction with TAKE_PARTIAL (if eligible tranche found) or HOLD.
        """
        if not self.config.enable_partial_profits or len(self.config.partial_tranches) == 0:
            return ExitAction(
                action="HOLD",
                reason="Partial profits disabled",
                strategy="partial_profit",
            )

        current_r = ctx.unrealized_r()
        metadata = {
            "current_r": current_r,
            "tranches_closed": ctx.tranches_closed,
        }

        # Find first unclosed tranche that has hit its target
        for idx, tranche in enumerate(self.config.partial_tranches):
            if idx in ctx.tranches_closed:
                continue  # Already closed

            r_target = tranche.get("r_target", 1.0)
            tranche_type = tranche.get("type", "fixed")
            ratio = tranche.get("ratio", 0.25)

            # Check if R target met
            if current_r >= r_target:
                reason = (
                    f"Partial profit taking: tranche {idx} ({ratio:.0%}) at R={current_r:.2f} "
                    f">= target {r_target:.2f} (type={tranche_type})"
                )
                logger.info(f"Partial profit exit triggered: {reason}")
                return ExitAction(
                    action="TAKE_PARTIAL",
                    reason=reason,
                    strategy="partial_profit",
                    partial_close_ratio=ratio,
                    confidence=0.80,
                    urgency="normal",
                    metadata={**metadata, "tranche_index": idx, "tranche_config": tranche},
                )

        # No tranches ready; return HOLD with nearest target info
        next_target = min(
            (t.get("r_target", 1.0) for i, t in enumerate(self.config.partial_tranches) if i not in ctx.tranches_closed),
            default=None,
        )
        if next_target is not None:
            reason = f"Partial profits: current R={current_r:.2f}, next target R={next_target:.2f}"
        else:
            reason = "Partial profits: all tranches closed or no targets defined"

        return ExitAction(
            action="HOLD",
            reason=reason,
            strategy="partial_profit",
            confidence=min(1.0, current_r / (next_target or 1.0)) if next_target else 0.0,
            metadata=metadata,
        )

    def _priority_merge(self, actions: List[ExitAction]) -> ExitAction:
        """
        Merge multiple exit actions by priority.

        Priority order (highest to lowest):
        1. EXIT_FULL with urgency="critical"
        2. EXIT_FULL with urgency="high"
        3. TAKE_PARTIAL
        4. EXIT_FULL with urgency="normal"
        5. TIGHTEN_SL
        6. HOLD (default if all strategies agree to hold)

        Returns the first non-HOLD action in priority order, or HOLD if all are HOLD.

        Args:
            actions: List of ExitAction from all strategies.

        Returns:
            Highest-priority action.
        """
        # Filter out HOLD actions to find the highest priority exit
        exit_actions = [a for a in actions if a.is_exit()]

        if not exit_actions:
            # All strategies agree to hold; return the one with highest confidence
            return max(actions, key=lambda a: a.confidence)

        # Sort by priority: critical exits first, then high urgency, then partials, then normal
        def priority_key(action: ExitAction) -> tuple:
            urgency_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
            action_order = {"EXIT_FULL": 0, "TAKE_PARTIAL": 1, "TIGHTEN_SL": 2}
            return (
                action_order.get(action.action, 3),
                urgency_order.get(action.urgency, 3),
            )

        return min(exit_actions, key=priority_key)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_default_exit_manager() -> AdaptiveExitManager:
    """
    Create an AdaptiveExitManager with default configuration.

    Returns:
        Configured AdaptiveExitManager instance.
    """
    return AdaptiveExitManager(config=ExitStrategyConfig())


def create_conservative_exit_manager() -> AdaptiveExitManager:
    """
    Create an AdaptiveExitManager with conservative (tight) exit rules.

    Useful for high-risk or uncertain market conditions.

    Returns:
        Configured AdaptiveExitManager with tighter parameters.
    """
    config = ExitStrategyConfig()
    config.chandelier_atr_multiplier = 2.0  # Tighter than default 3.0
    config.vol_trail_multipliers = {
        "LOW": 1.0,
        "NORMAL": 1.5,
        "HIGH": 2.0,
        "EXTREME": 3.0,
    }
    config.time_exit_bars = {
        "LOW": 30,
        "NORMAL": 20,
        "HIGH": 12,
        "EXTREME": 6,
    }
    config.confidence_exit_threshold = 0.45  # Exit sooner on confidence decay
    return AdaptiveExitManager(config=config)


def create_aggressive_exit_manager() -> AdaptiveExitManager:
    """
    Create an AdaptiveExitManager with aggressive (loose) exit rules.

    Useful for trending markets where trades need breathing room.

    Returns:
        Configured AdaptiveExitManager with looser parameters.
    """
    config = ExitStrategyConfig()
    config.chandelier_atr_multiplier = 4.0  # Looser than default 3.0
    config.vol_trail_multipliers = {
        "LOW": 2.0,
        "NORMAL": 3.0,
        "HIGH": 4.0,
        "EXTREME": 5.0,
    }
    config.time_exit_bars = {
        "LOW": 60,
        "NORMAL": 40,
        "HIGH": 25,
        "EXTREME": 12,
    }
    config.confidence_exit_threshold = 0.25  # Hold longer before confidence decay exit
    return AdaptiveExitManager(config=config)
