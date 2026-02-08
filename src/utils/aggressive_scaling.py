"""Aggressive Scaling Module for $100K → $1M Strategy.

This module implements research-backed aggressive scaling strategies:
1. Kelly Criterion with slippage adjustment
2. Smart order execution (limit orders, split orders)
3. Intraday compounding
4. Drawdown protection with circuit breakers

Based on 78% win rate, 1.67 R:R edge optimization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

class ScalingPhase(Enum):
    """Scaling phases for progressive position sizing."""
    VALIDATION = "validation"      # Weeks 1-2: Prove model
    OPTIMIZATION = "optimization"  # Weeks 3-4: Optimize execution
    SCALING = "scaling"            # Month 2+: Full Kelly
    ACCELERATION = "acceleration"  # Month 3+: Multiple pairs/accounts


@dataclass
class TradingStats:
    """Historical trading statistics for Kelly calculation."""
    # Core stats
    win_rate: float = 0.78              # Validation win rate
    avg_win_pips: float = 41.5          # Average winning trade
    avg_loss_pips: float = 24.9         # Average losing trade

    # Sample size tracking
    total_trades: int = 6               # 6/6 wins so far
    winning_trades: int = 6
    losing_trades: int = 0

    # Execution costs
    avg_slippage_pips: float = 6.0      # 6 pips at 10 lots
    spread_pips: float = 1.5            # Typical spread

    # Confidence intervals
    min_trades_for_kelly: int = 20      # Minimum trades before full Kelly
    confidence_factor: float = 0.5      # Scale down Kelly until sample size met

    def update_from_trades(self, trades: List[Dict]) -> None:
        """Update stats from trade history."""
        if not trades:
            return

        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) <= 0]

        self.total_trades = len(trades)
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)

        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

        if wins:
            self.avg_win_pips = sum(abs(t.get('pnl_pips', 0)) for t in wins) / len(wins)
        if losses:
            self.avg_loss_pips = sum(abs(t.get('pnl_pips', 0)) for t in losses) / len(losses)

        # Update slippage from actual execution
        slippages = [abs(t.get('slippage_pips', 0)) for t in trades if t.get('slippage_pips') is not None]
        if slippages:
            self.avg_slippage_pips = sum(slippages) / len(slippages)

    @property
    def risk_reward_ratio(self) -> float:
        """Raw R:R before costs."""
        if self.avg_loss_pips <= 0:
            return 2.0  # Default
        return self.avg_win_pips / self.avg_loss_pips

    @property
    def effective_risk_reward(self) -> float:
        """R:R after slippage costs."""
        effective_win = self.avg_win_pips - self.avg_slippage_pips
        effective_loss = self.avg_loss_pips + self.avg_slippage_pips
        if effective_loss <= 0:
            return 1.0
        return effective_win / effective_loss


@dataclass
class KellyConfig:
    """Kelly Criterion configuration."""
    # Kelly fraction scaling
    full_kelly_fraction: float = 1.0    # 1.0 = full Kelly, 0.5 = half Kelly
    max_kelly_fraction: float = 0.65    # Never exceed 65% even with perfect stats
    min_kelly_fraction: float = 0.05    # Minimum 5% position

    # Safety adjustments
    slippage_adjustment: bool = True    # Adjust for execution costs
    sample_size_scaling: bool = True    # Scale down until 20+ trades

    # Phase-based Kelly multipliers
    validation_multiplier: float = 0.25   # 25% Kelly during validation
    optimization_multiplier: float = 0.50  # 50% Kelly during optimization
    scaling_multiplier: float = 0.75      # 75% Kelly during scaling
    acceleration_multiplier: float = 1.0  # Full Kelly during acceleration


@dataclass
class ExecutionConfig:
    """Smart order execution configuration."""
    # Limit order settings
    use_limit_orders: bool = True
    limit_offset_atr_mult: float = 0.3   # Place limit 0.3 ATR away
    limit_wait_seconds: int = 900        # 15 min wait before market order

    # Split order settings
    use_split_orders: bool = True
    split_threshold_lots: float = 5.0    # Split orders > 5 lots
    split_count: int = 3                 # Split into 3 orders
    split_ratios: Tuple[float, ...] = (0.4, 0.3, 0.3)  # 40%, 30%, 30%
    split_delay_seconds: int = 30        # Delay between splits

    # Slippage targets
    target_slippage_pips: float = 1.5    # Target < 2 pips
    max_acceptable_slippage_pips: float = 4.0


@dataclass
class CompoundingConfig:
    """Intraday compounding configuration."""
    enabled: bool = True
    recalculate_after_each_trade: bool = True
    compound_frequency: str = "intraday"  # intraday, daily, weekly

    # Reserve settings
    profit_reserve_pct: float = 0.10     # Keep 10% as drawdown buffer
    min_compound_amount: float = 1000    # Min $1K before compounding


@dataclass
class DrawdownConfig:
    """Drawdown protection configuration."""
    # Daily circuit breakers
    max_daily_drawdown_pct: float = 0.10  # Stop at -10% daily
    max_trades_per_day: int = 6

    # Losing streak protection
    losing_streak_reduction: Dict[int, float] = field(default_factory=lambda: {
        1: 0.75,  # 75% after 1 loss
        2: 0.50,  # 50% after 2 losses
        3: 0.25,  # 25% after 3+ losses
    })

    # Recovery settings
    recovery_win_streak: int = 2         # 2 wins to restore full size
    pause_after_max_streak: int = 3      # Pause trading after 3 losses


@dataclass
class AggressiveScalingConfig:
    """Master configuration for aggressive scaling."""
    # Phase configuration
    current_phase: ScalingPhase = ScalingPhase.VALIDATION
    phase_start_date: str = ""

    # Sub-configurations
    stats: TradingStats = field(default_factory=TradingStats)
    kelly: KellyConfig = field(default_factory=KellyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    compounding: CompoundingConfig = field(default_factory=CompoundingConfig)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)

    # Lot size settings
    standard_lot_units: int = 100_000    # 1 lot = 100K units
    pip_value_per_lot: float = 10.0      # $10/pip for standard pairs
    jpy_pip_value_per_lot: float = 9.3   # ~$9.3/pip for JPY pairs

    # Account limits
    max_leverage: float = 50.0           # Broker max leverage
    min_lot_size: float = 0.01           # Micro lots


# ============================================================================
# Kelly Criterion Calculator
# ============================================================================

class KellyCalculator:
    """
    Optimal position sizing using Kelly Criterion with slippage adjustment.

    Full Kelly Formula:
        f* = (p*b - q) / b

    Where:
        p = probability of winning (win rate)
        q = probability of losing (1 - p)
        b = ratio of win to loss (R:R)

    Slippage-Adjusted Kelly:
        - Reduces effective win by slippage
        - Increases effective loss by slippage
        - Results in more conservative sizing
    """

    def __init__(self, config: AggressiveScalingConfig):
        self.config = config

    def calculate_kelly_fraction(self) -> Dict[str, Any]:
        """Calculate optimal Kelly fraction with all adjustments."""
        stats = self.config.stats
        kelly_cfg = self.config.kelly

        # 1. Calculate raw Kelly
        raw_rr = stats.risk_reward_ratio
        raw_kelly = self._kelly_formula(stats.win_rate, raw_rr)

        # 2. Calculate slippage-adjusted Kelly
        if kelly_cfg.slippage_adjustment:
            effective_rr = stats.effective_risk_reward
            adjusted_kelly = self._kelly_formula(stats.win_rate, effective_rr)
        else:
            adjusted_kelly = raw_kelly

        # 3. Apply sample size scaling
        if kelly_cfg.sample_size_scaling and stats.total_trades < stats.min_trades_for_kelly:
            sample_factor = stats.total_trades / stats.min_trades_for_kelly
            sample_factor = max(0.25, sample_factor)  # At least 25%
            adjusted_kelly *= sample_factor

        # 4. Apply phase multiplier
        phase_mult = self._get_phase_multiplier()
        phase_kelly = adjusted_kelly * phase_mult

        # 5. Apply fraction scaling (half Kelly, quarter Kelly, etc.)
        final_kelly = phase_kelly * kelly_cfg.full_kelly_fraction

        # 6. Apply bounds
        final_kelly = max(kelly_cfg.min_kelly_fraction,
                          min(kelly_cfg.max_kelly_fraction, final_kelly))

        return {
            "raw_kelly": raw_kelly,
            "raw_rr": raw_rr,
            "slippage_adjusted_kelly": adjusted_kelly if kelly_cfg.slippage_adjustment else None,
            "effective_rr": stats.effective_risk_reward if kelly_cfg.slippage_adjustment else None,
            "phase": self.config.current_phase.value,
            "phase_multiplier": phase_mult,
            "final_kelly_fraction": final_kelly,
            "recommended_risk_pct": final_kelly * 100,
            "stats_used": {
                "win_rate": stats.win_rate,
                "avg_win_pips": stats.avg_win_pips,
                "avg_loss_pips": stats.avg_loss_pips,
                "slippage_pips": stats.avg_slippage_pips,
                "sample_size": stats.total_trades,
            }
        }

    def _kelly_formula(self, win_rate: float, rr_ratio: float) -> float:
        """Core Kelly formula: f* = (p*b - q) / b"""
        p = win_rate
        q = 1 - p
        b = rr_ratio

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b
        return max(0.0, kelly)

    def _get_phase_multiplier(self) -> float:
        """Get Kelly multiplier for current phase."""
        phase = self.config.current_phase
        kelly_cfg = self.config.kelly

        if phase == ScalingPhase.VALIDATION:
            return kelly_cfg.validation_multiplier
        elif phase == ScalingPhase.OPTIMIZATION:
            return kelly_cfg.optimization_multiplier
        elif phase == ScalingPhase.SCALING:
            return kelly_cfg.scaling_multiplier
        else:  # ACCELERATION
            return kelly_cfg.acceleration_multiplier

    def calculate_position_size(
        self,
        account_equity: float,
        stop_loss_pips: float,
        instrument: str = "EUR_USD"
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using Kelly.

        Args:
            account_equity: Current account equity in USD
            stop_loss_pips: Stop loss distance in pips
            instrument: Trading instrument

        Returns:
            Dict with lots, units, risk amount, and calculations
        """
        kelly_result = self.calculate_kelly_fraction()
        kelly_fraction = kelly_result["final_kelly_fraction"]

        # Calculate risk amount
        risk_amount = account_equity * kelly_fraction

        # Get pip value
        is_jpy = instrument.upper().endswith("_JPY")
        pip_value_per_lot = (self.config.jpy_pip_value_per_lot if is_jpy
                             else self.config.pip_value_per_lot)

        # Calculate lots: Risk / (SL pips * pip value per lot)
        if stop_loss_pips <= 0 or pip_value_per_lot <= 0:
            return {"error": "Invalid stop loss or pip value", "lots": 0, "units": 0}

        lots = risk_amount / (stop_loss_pips * pip_value_per_lot)

        # Apply minimum lot constraint
        lots = max(self.config.min_lot_size, lots)

        # Check leverage constraint
        position_value = lots * self.config.standard_lot_units
        leverage_used = position_value / account_equity if account_equity > 0 else 0

        if leverage_used > self.config.max_leverage:
            # Scale down to max leverage
            lots = (account_equity * self.config.max_leverage) / self.config.standard_lot_units

        units = int(lots * self.config.standard_lot_units)
        actual_risk = lots * stop_loss_pips * pip_value_per_lot

        return {
            "lots": round(lots, 2),
            "units": units,
            "risk_amount_usd": round(actual_risk, 2),
            "risk_pct": round((actual_risk / account_equity) * 100, 2) if account_equity > 0 else 0,
            "leverage_used": round(leverage_used, 2),
            "kelly_calculation": kelly_result,
            "instrument": instrument,
            "stop_loss_pips": stop_loss_pips,
        }


# ============================================================================
# Smart Order Execution
# ============================================================================

@dataclass
class OrderPlan:
    """Plan for smart order execution."""
    total_lots: float
    split_orders: List[Dict[str, Any]]
    use_limit: bool
    limit_price: Optional[float]
    limit_offset: Optional[float]
    total_expected_slippage_pips: float
    estimated_savings_usd: float


class SmartOrderExecutor:
    """
    Smart order execution to minimize slippage.

    Strategies:
    1. Limit orders with mean reversion entry
    2. Split large orders into smaller chunks
    3. Time-based execution to avoid market impact
    """

    def __init__(self, config: AggressiveScalingConfig):
        self.config = config

    def plan_execution(
        self,
        direction: str,  # 'long' or 'short'
        total_lots: float,
        current_price: float,
        atr: float,
        instrument: str = "EUR_USD"
    ) -> OrderPlan:
        """
        Plan optimal order execution.

        Args:
            direction: Trade direction
            total_lots: Total position size in lots
            current_price: Current market price
            atr: Average True Range for limit offset
            instrument: Trading instrument

        Returns:
            OrderPlan with execution strategy
        """
        exec_cfg = self.config.execution

        # Determine if we should split
        should_split = (exec_cfg.use_split_orders and
                        total_lots > exec_cfg.split_threshold_lots)

        # Calculate limit order offset
        limit_offset = None
        limit_price = None
        if exec_cfg.use_limit_orders:
            limit_offset = atr * exec_cfg.limit_offset_atr_mult
            if direction.lower() == 'long':
                limit_price = current_price - limit_offset
            else:
                limit_price = current_price + limit_offset

        # Create order splits
        split_orders = []
        if should_split:
            remaining_lots = total_lots
            for i, ratio in enumerate(exec_cfg.split_ratios):
                order_lots = total_lots * ratio
                remaining_lots -= order_lots

                order_type = "limit" if i == 0 else ("limit" if i == 1 else "passive")
                delay = 0 if i == 0 else (i * exec_cfg.split_delay_seconds)

                split_orders.append({
                    "order_num": i + 1,
                    "lots": round(order_lots, 2),
                    "order_type": order_type,
                    "delay_seconds": delay,
                    "expected_slippage_pips": self._estimate_slippage(order_lots, order_type),
                })
        else:
            order_type = "limit" if exec_cfg.use_limit_orders else "market"
            split_orders.append({
                "order_num": 1,
                "lots": round(total_lots, 2),
                "order_type": order_type,
                "delay_seconds": 0,
                "expected_slippage_pips": self._estimate_slippage(total_lots, order_type),
            })

        # Calculate total expected slippage
        total_slippage = sum(o["expected_slippage_pips"] * o["lots"] for o in split_orders)
        avg_slippage = total_slippage / total_lots if total_lots > 0 else 0

        # Calculate savings vs naive market order
        naive_slippage = self.config.stats.avg_slippage_pips * total_lots
        savings_pips = naive_slippage - total_slippage
        savings_usd = savings_pips * self.config.pip_value_per_lot

        return OrderPlan(
            total_lots=total_lots,
            split_orders=split_orders,
            use_limit=exec_cfg.use_limit_orders,
            limit_price=round(limit_price, 5) if limit_price else None,
            limit_offset=round(limit_offset, 5) if limit_offset else None,
            total_expected_slippage_pips=round(avg_slippage, 2),
            estimated_savings_usd=round(savings_usd, 2),
        )

    def _estimate_slippage(self, lots: float, order_type: str) -> float:
        """Estimate slippage based on order size and type."""
        base_slippage = self.config.stats.avg_slippage_pips

        # Limit orders typically get better fills
        if order_type == "limit":
            return max(0.5, base_slippage * 0.25)  # 75% reduction
        elif order_type == "passive":
            return max(0.3, base_slippage * 0.15)  # 85% reduction (maker rebate)
        else:  # market
            # Larger orders have more slippage
            size_factor = 1.0 + (lots / 10) * 0.1  # 10% more per 10 lots
            return base_slippage * size_factor


# ============================================================================
# Intraday Compounding Engine
# ============================================================================

@dataclass
class IntradayState:
    """Track intraday trading state for compounding."""
    date: str
    starting_equity: float
    current_equity: float
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0
    trade_history: List[Dict] = field(default_factory=list)


class IntradayCompounder:
    """
    Intraday compounding engine.

    Reinvests profits within the trading day for exponential growth.
    """

    def __init__(self, config: AggressiveScalingConfig):
        self.config = config
        self.state: Optional[IntradayState] = None

    def start_session(self, starting_equity: float) -> IntradayState:
        """Initialize a new trading session."""
        today = datetime.now(timezone.utc).date().isoformat()
        self.state = IntradayState(
            date=today,
            starting_equity=starting_equity,
            current_equity=starting_equity,
        )
        return self.state

    def update_after_trade(
        self,
        pnl: float,
        is_win: bool
    ) -> Dict[str, Any]:
        """
        Update state after a trade and calculate next position size.

        Args:
            pnl: Trade P/L in USD
            is_win: Whether the trade was a win

        Returns:
            Updated state and next recommended size
        """
        if self.state is None:
            raise ValueError("Session not started. Call start_session first.")

        # Update state
        self.state.current_equity += pnl
        self.state.pnl_today += pnl
        self.state.trades_today += 1

        if is_win:
            self.state.wins_today += 1
        else:
            self.state.losses_today += 1

        self.state.trade_history.append({
            "trade_num": self.state.trades_today,
            "pnl": pnl,
            "is_win": is_win,
            "equity_after": self.state.current_equity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Calculate compounding stats
        compound_cfg = self.config.compounding

        # Reserve profit buffer
        usable_equity = self.state.current_equity
        if self.state.pnl_today > 0:
            reserve = self.state.pnl_today * compound_cfg.profit_reserve_pct
            usable_equity = self.state.current_equity - reserve

        daily_return = (self.state.current_equity - self.state.starting_equity) / self.state.starting_equity

        return {
            "current_equity": round(self.state.current_equity, 2),
            "usable_equity": round(usable_equity, 2),
            "pnl_today": round(self.state.pnl_today, 2),
            "daily_return_pct": round(daily_return * 100, 2),
            "trades_today": self.state.trades_today,
            "win_rate_today": round(self.state.wins_today / self.state.trades_today * 100, 1) if self.state.trades_today > 0 else 0,
            "compounding_gain": self._calculate_compounding_gain(),
        }

    def _calculate_compounding_gain(self) -> Dict[str, float]:
        """Calculate extra gains from intraday compounding."""
        if self.state is None or not self.state.trade_history:
            return {"extra_gain_usd": 0, "compound_factor": 1.0}

        # Calculate what we would have made without compounding
        # (fixed size based on starting equity)
        avg_trade_pnl = self.state.pnl_today / self.state.trades_today
        simple_pnl = avg_trade_pnl * self.state.trades_today

        # Actual PnL includes compounding effects
        actual_pnl = self.state.pnl_today

        extra_gain = actual_pnl - simple_pnl
        compound_factor = actual_pnl / simple_pnl if simple_pnl != 0 else 1.0

        return {
            "extra_gain_usd": round(extra_gain, 2),
            "compound_factor": round(compound_factor, 3),
        }


# ============================================================================
# Aggressive Risk Manager
# ============================================================================

@dataclass
class RiskState:
    """Track risk state for drawdown protection."""
    losing_streak: int = 0
    winning_streak: int = 0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    is_paused: bool = False
    pause_reason: Optional[str] = None
    position_multiplier: float = 1.0


class AggressiveRiskManager:
    """
    Risk management with drawdown protection and losing streak handling.

    Features:
    - Daily loss circuit breaker
    - Position size reduction after losses
    - Recovery mode after winning streak
    """

    def __init__(self, config: AggressiveScalingConfig):
        self.config = config
        self.state = RiskState()
        self._state_path = Path("trained_data") / "aggressive_risk_state.json"

    def check_can_trade(self, account_equity: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk rules.

        Returns:
            Tuple of (can_trade, reason)
        """
        dd_cfg = self.config.drawdown

        # Check if paused
        if self.state.is_paused:
            return False, f"Trading paused: {self.state.pause_reason}"

        # Check daily trade limit
        if self.state.daily_trades >= dd_cfg.max_trades_per_day:
            return False, f"Daily trade limit reached ({dd_cfg.max_trades_per_day})"

        # Check daily drawdown
        if account_equity > 0:
            daily_dd = self.state.daily_pnl / account_equity
            if daily_dd < -dd_cfg.max_daily_drawdown_pct:
                self.state.is_paused = True
                self.state.pause_reason = f"Daily drawdown limit hit ({daily_dd:.1%})"
                return False, self.state.pause_reason

        # Check losing streak pause
        if self.state.losing_streak >= dd_cfg.pause_after_max_streak:
            self.state.is_paused = True
            self.state.pause_reason = f"Losing streak of {self.state.losing_streak}"
            return False, self.state.pause_reason

        return True, "OK"

    def adjust_size_for_streak(self, base_lots: float) -> Tuple[float, str]:
        """
        Adjust position size based on losing streak.

        Returns:
            Tuple of (adjusted_lots, reason)
        """
        dd_cfg = self.config.drawdown

        if self.state.losing_streak == 0:
            return base_lots, "Full size - no losing streak"

        # Get reduction factor
        reduction = dd_cfg.losing_streak_reduction.get(
            self.state.losing_streak,
            dd_cfg.losing_streak_reduction.get(3, 0.25)  # Default to 25% for 3+
        )

        adjusted = base_lots * reduction
        reason = f"Reduced to {reduction*100:.0f}% after {self.state.losing_streak} loss(es)"

        self.state.position_multiplier = reduction

        return adjusted, reason

    def update_after_trade(self, pnl: float, is_win: bool) -> Dict[str, Any]:
        """
        Update risk state after a trade.

        Args:
            pnl: Trade P/L in USD
            is_win: Whether the trade was a win

        Returns:
            Updated risk state
        """
        dd_cfg = self.config.drawdown

        self.state.daily_pnl += pnl
        self.state.daily_trades += 1

        if is_win:
            self.state.winning_streak += 1
            self.state.losing_streak = 0

            # Check if we can restore full size
            if self.state.winning_streak >= dd_cfg.recovery_win_streak:
                self.state.position_multiplier = 1.0
        else:
            self.state.losing_streak += 1
            self.state.winning_streak = 0

        # Save state
        self._save_state()

        return {
            "losing_streak": self.state.losing_streak,
            "winning_streak": self.state.winning_streak,
            "daily_pnl": round(self.state.daily_pnl, 2),
            "daily_trades": self.state.daily_trades,
            "position_multiplier": self.state.position_multiplier,
            "is_paused": self.state.is_paused,
            "pause_reason": self.state.pause_reason,
        }

    def reset_daily(self) -> None:
        """Reset daily counters (call at session start)."""
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.is_paused = False
        self.state.pause_reason = None
        # Keep streak counters - they persist across days
        self._save_state()

    def _save_state(self) -> None:
        """Persist risk state to disk."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "losing_streak": self.state.losing_streak,
            "winning_streak": self.state.winning_streak,
            "position_multiplier": self.state.position_multiplier,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self._state_path.write_text(json.dumps(data, indent=2))

    def load_state(self) -> None:
        """Load risk state from disk."""
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text())
                self.state.losing_streak = data.get("losing_streak", 0)
                self.state.winning_streak = data.get("winning_streak", 0)
                self.state.position_multiplier = data.get("position_multiplier", 1.0)
            except Exception as e:
                logger.warning(f"Failed to load risk state: {e}")


# ============================================================================
# Main Scaling Engine
# ============================================================================

class AggressiveScalingEngine:
    """
    Main engine coordinating all aggressive scaling components.

    Usage:
        engine = AggressiveScalingEngine()

        # Get optimal position size
        position = engine.calculate_position(
            account_equity=100000,
            stop_loss_pips=24.9,
            current_price=1.0850,
            atr=0.0012
        )

        # After trade closes
        engine.update_after_trade(pnl=3550, is_win=True)
    """

    def __init__(self, config: Optional[AggressiveScalingConfig] = None):
        self.config = config or AggressiveScalingConfig()

        # Initialize components
        self.kelly = KellyCalculator(self.config)
        self.executor = SmartOrderExecutor(self.config)
        self.compounder = IntradayCompounder(self.config)
        self.risk_manager = AggressiveRiskManager(self.config)

        # Load persisted state
        self.risk_manager.load_state()

    def start_session(self, account_equity: float) -> Dict[str, Any]:
        """
        Start a new trading session.

        Args:
            account_equity: Starting account equity

        Returns:
            Session initialization info
        """
        self.risk_manager.reset_daily()
        self.compounder.start_session(account_equity)

        return {
            "session_start": datetime.now(timezone.utc).isoformat(),
            "starting_equity": account_equity,
            "phase": self.config.current_phase.value,
            "kelly_info": self.kelly.calculate_kelly_fraction(),
        }

    def calculate_position(
        self,
        account_equity: float,
        stop_loss_pips: float,
        current_price: float,
        atr: float,
        instrument: str = "EUR_USD",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Calculate optimal position with all adjustments.

        Args:
            account_equity: Current account equity
            stop_loss_pips: Stop loss in pips
            current_price: Current market price
            atr: Average True Range
            instrument: Trading instrument
            direction: Trade direction

        Returns:
            Complete position sizing and execution plan
        """
        # 1. Check if we can trade
        can_trade, reason = self.risk_manager.check_can_trade(account_equity)
        if not can_trade:
            return {
                "can_trade": False,
                "reason": reason,
                "lots": 0,
                "units": 0,
            }

        # 2. Get compounded equity if applicable
        if self.config.compounding.enabled and self.compounder.state:
            effective_equity = self.compounder.state.current_equity
        else:
            effective_equity = account_equity

        # 3. Calculate Kelly-optimal position
        kelly_position = self.kelly.calculate_position_size(
            account_equity=effective_equity,
            stop_loss_pips=stop_loss_pips,
            instrument=instrument
        )

        base_lots = kelly_position.get("lots", 0)

        # 4. Adjust for losing streak
        adjusted_lots, streak_reason = self.risk_manager.adjust_size_for_streak(base_lots)

        # 5. Plan execution
        execution_plan = self.executor.plan_execution(
            direction=direction,
            total_lots=adjusted_lots,
            current_price=current_price,
            atr=atr,
            instrument=instrument
        )

        return {
            "can_trade": True,
            "lots": round(adjusted_lots, 2),
            "units": int(adjusted_lots * self.config.standard_lot_units),
            "kelly_calculation": kelly_position,
            "streak_adjustment": streak_reason,
            "execution_plan": asdict(execution_plan),
            "effective_equity": round(effective_equity, 2),
            "phase": self.config.current_phase.value,
        }

    def update_after_trade(self, pnl: float, is_win: bool) -> Dict[str, Any]:
        """
        Update all components after a trade closes.

        Args:
            pnl: Trade P/L in USD
            is_win: Whether the trade was a win

        Returns:
            Updated state from all components
        """
        # Update risk manager
        risk_update = self.risk_manager.update_after_trade(pnl, is_win)

        # Update compounder
        compound_update = {}
        if self.config.compounding.enabled and self.compounder.state:
            compound_update = self.compounder.update_after_trade(pnl, is_win)

        return {
            "risk_state": risk_update,
            "compound_state": compound_update,
        }

    def advance_phase(self) -> str:
        """Advance to next scaling phase."""
        phases = list(ScalingPhase)
        current_idx = phases.index(self.config.current_phase)

        if current_idx < len(phases) - 1:
            self.config.current_phase = phases[current_idx + 1]
            self.config.phase_start_date = datetime.now(timezone.utc).date().isoformat()

        return self.config.current_phase.value

    def get_projection(
        self,
        starting_equity: float,
        trades_per_day: int = 3,
        trading_days: int = 22,  # ~1 month
        months: int = 6
    ) -> Dict[str, Any]:
        """
        Project growth based on current stats and Kelly sizing.

        Args:
            starting_equity: Starting capital
            trades_per_day: Expected trades per day
            trading_days: Trading days per month
            months: Projection period in months

        Returns:
            Monthly projections with risk scenarios
        """
        stats = self.config.stats
        kelly = self.kelly.calculate_kelly_fraction()

        projections = []
        equity = starting_equity

        for month in range(1, months + 1):
            month_trades = trades_per_day * trading_days

            # Expected wins/losses
            expected_wins = int(month_trades * stats.win_rate)
            expected_losses = month_trades - expected_wins

            # Calculate expected PnL
            # Average win after slippage
            net_win_pips = stats.avg_win_pips - stats.avg_slippage_pips
            net_loss_pips = stats.avg_loss_pips + stats.avg_slippage_pips

            # Position sizing at this equity level
            kelly_risk = kelly["final_kelly_fraction"]
            avg_lots = (equity * kelly_risk) / (stats.avg_loss_pips * self.config.pip_value_per_lot)

            expected_win_pnl = expected_wins * net_win_pips * avg_lots * self.config.pip_value_per_lot
            expected_loss_pnl = expected_losses * net_loss_pips * avg_lots * self.config.pip_value_per_lot

            month_pnl = expected_win_pnl - expected_loss_pnl
            month_return = month_pnl / equity if equity > 0 else 0

            equity += month_pnl

            projections.append({
                "month": month,
                "starting_equity": round(equity - month_pnl, 2),
                "ending_equity": round(equity, 2),
                "month_pnl": round(month_pnl, 2),
                "month_return_pct": round(month_return * 100, 1),
                "avg_lots": round(avg_lots, 1),
                "trades": month_trades,
            })

        # Risk scenarios
        conservative = equity * 0.6   # 60% of projection
        aggressive = equity * 1.5     # 150% of projection

        return {
            "projections": projections,
            "final_equity_expected": round(equity, 2),
            "total_return_pct": round((equity - starting_equity) / starting_equity * 100, 1),
            "scenarios": {
                "conservative": round(conservative, 2),
                "expected": round(equity, 2),
                "aggressive": round(aggressive, 2),
            },
            "assumptions": {
                "win_rate": stats.win_rate,
                "slippage_pips": stats.avg_slippage_pips,
                "kelly_fraction": kelly["final_kelly_fraction"],
                "trades_per_day": trades_per_day,
            }
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_validation_phase_engine() -> AggressiveScalingEngine:
    """Create engine configured for validation phase (weeks 1-2)."""
    config = AggressiveScalingConfig(
        current_phase=ScalingPhase.VALIDATION,
    )
    # More conservative settings
    config.kelly.full_kelly_fraction = 0.25  # Quarter Kelly
    config.execution.use_limit_orders = True
    config.execution.use_split_orders = False  # Smaller sizes don't need splits
    config.drawdown.max_daily_drawdown_pct = 0.05  # 5% max

    return AggressiveScalingEngine(config)


def create_scaling_phase_engine() -> AggressiveScalingEngine:
    """Create engine configured for scaling phase (month 2+)."""
    config = AggressiveScalingConfig(
        current_phase=ScalingPhase.SCALING,
    )
    config.kelly.full_kelly_fraction = 0.75  # 3/4 Kelly
    config.execution.use_limit_orders = True
    config.execution.use_split_orders = True

    return AggressiveScalingEngine(config)


def create_full_kelly_engine() -> AggressiveScalingEngine:
    """Create engine configured for full Kelly (month 3+)."""
    config = AggressiveScalingConfig(
        current_phase=ScalingPhase.ACCELERATION,
    )
    config.kelly.full_kelly_fraction = 1.0
    config.execution.use_limit_orders = True
    config.execution.use_split_orders = True

    return AggressiveScalingEngine(config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: $100K account, targeting $1M
    print("=" * 60)
    print("AGGRESSIVE SCALING ENGINE - $100K → $1M Strategy")
    print("=" * 60)

    # Start with validation phase
    engine = create_validation_phase_engine()

    # Start session
    session = engine.start_session(account_equity=100_000)
    print("\n📊 Session Started:")
    print(f"   Phase: {session['phase']}")
    print(f"   Kelly Fraction: {session['kelly_info']['final_kelly_fraction']:.1%}")

    # Calculate position for a trade
    position = engine.calculate_position(
        account_equity=100_000,
        stop_loss_pips=24.9,
        current_price=1.0850,
        atr=0.0012,
        instrument="EUR_USD",
        direction="long"
    )

    print("\n📈 Position Calculation:")
    print(f"   Can Trade: {position['can_trade']}")
    print(f"   Lots: {position['lots']}")
    print(f"   Units: {position['units']:,}")
    print(f"   Effective Equity: ${position['effective_equity']:,.2f}")

    if position.get('execution_plan'):
        plan = position['execution_plan']
        print("\n🎯 Execution Plan:")
        print(f"   Expected Slippage: {plan['total_expected_slippage_pips']} pips")
        print(f"   Estimated Savings: ${plan['estimated_savings_usd']}")
        if plan['use_limit']:
            print(f"   Limit Price: {plan['limit_price']}")

    # Simulate a winning trade
    result = engine.update_after_trade(pnl=3550, is_win=True)
    print("\n✅ After Winning Trade:")
    print(f"   Daily PnL: ${result['risk_state']['daily_pnl']:,.2f}")
    if result.get('compound_state'):
        print(f"   Current Equity: ${result['compound_state']['current_equity']:,.2f}")
        print(f"   Daily Return: {result['compound_state']['daily_return_pct']:.1f}%")

    # Get 6-month projection
    projection = engine.get_projection(
        starting_equity=100_000,
        trades_per_day=3,
        months=6
    )

    print("\n📊 6-Month Projection:")
    print(f"   {'Month':<8} {'Starting':<15} {'Ending':<15} {'Return':<10}")
    print(f"   {'-'*48}")
    for p in projection['projections']:
        print(f"   {p['month']:<8} ${p['starting_equity']:>12,.0f} ${p['ending_equity']:>12,.0f} {p['month_return_pct']:>7.1f}%")

    print(f"\n   Final Expected: ${projection['final_equity_expected']:,.0f}")
    print(f"   Total Return: {projection['total_return_pct']:.0f}%")
    print("\n   Scenarios:")
    print(f"   - Conservative: ${projection['scenarios']['conservative']:,.0f}")
    print(f"   - Expected:     ${projection['scenarios']['expected']:,.0f}")
    print(f"   - Aggressive:   ${projection['scenarios']['aggressive']:,.0f}")
