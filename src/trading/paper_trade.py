"""Paper Trade Planning Module.

This module provides paper trade planning and simulation utilities,
extracted from main.py lines 6157-6414 as part of Phase 3a decomposition.

Paper trading allows testing strategies on OANDA PRACTICE without real money risk.
All trades are simulated against live market data with full risk management.

Example usage:
    from src.trading.paper_trade import (
        setup_paper_trade_context,
        build_paper_trade_plan,
        execute_paper_trade_plan,
    )

    # Setup context (validates policy, connects to OANDA)
    context = setup_paper_trade_context(config, instrument="EUR_USD", granularity="H1")
    if context is None:
        print("Setup failed - check policy/account state")
        return

    # Build trade plan (analyzes market, calculates sizing)
    plan = build_paper_trade_plan(context, candles=300, equity=10000.0)
    if plan is None:
        print("No valid setup found")
        return

    # Execute (or dry-run)
    execute_paper_trade_plan(context, plan, execute=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PaperTradePlan:
    """Plan for a paper trade execution.

    Contains all computed values needed to execute a trade:
    - Entry signal and direction
    - Position sizing
    - Risk levels (SL/TP)
    - Confidence metrics
    - Cost analysis

    Attributes:
        instrument: OANDA instrument name (e.g., "EUR_USD").
        granularity: Candle timeframe (e.g., "H1", "M5").
        signal: Trade direction signal ("buy", "sell", or "hold").
        last_close: Most recent closing price.
        atr_value: 14-period ATR value.
        units: Signed position size (negative for sells).
        stop_price: Stop loss price.
        tp_price: Take profit price.
        spread_pips: Current bid-ask spread in pips.
        slippage_pips: Estimated slippage in pips.
        slippage_price: Slippage in price units.
        confidence: Trading confidence score (0-1).
        band: Confidence band ("low", "medium", "high").
        conf_reasons: List of reasons affecting confidence.
    """
    instrument: str
    granularity: str
    signal: Literal["buy", "sell", "hold"]
    last_close: float
    atr_value: float
    units: int
    stop_price: float
    tp_price: float
    spread_pips: float
    slippage_pips: float
    slippage_price: float
    confidence: float
    band: str
    conf_reasons: List[str] = field(default_factory=list)

    @property
    def direction(self) -> str:
        """Trade direction as 'long' or 'short'."""
        return "long" if self.signal == "buy" else "short"

    @property
    def is_tradeable(self) -> bool:
        """Whether this plan represents a tradeable setup."""
        return self.signal in ("buy", "sell") and self.band != "low" and self.units != 0

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio based on SL/TP distances."""
        sl_distance = abs(self.last_close - self.stop_price)
        tp_distance = abs(self.tp_price - self.last_close)
        if sl_distance <= 0:
            return 0.0
        return tp_distance / sl_distance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instrument": self.instrument,
            "granularity": self.granularity,
            "signal": self.signal,
            "last_close": self.last_close,
            "atr_value": self.atr_value,
            "units": self.units,
            "stop_price": self.stop_price,
            "tp_price": self.tp_price,
            "spread_pips": self.spread_pips,
            "slippage_pips": self.slippage_pips,
            "confidence": self.confidence,
            "band": self.band,
            "conf_reasons": self.conf_reasons,
            "risk_reward_ratio": self.risk_reward_ratio,
        }

    def __str__(self) -> str:
        return (
            f"PaperTradePlan({self.signal.upper()} {self.instrument} "
            f"units={self.units} SL={self.stop_price:.5f} TP={self.tp_price:.5f} "
            f"conf={self.confidence:.2f} band={self.band})"
        )


@dataclass
class PaperTradeContext:
    """Context for paper trade execution.

    Contains all state needed for trade planning and execution:
    - Configuration
    - Policy rules
    - OANDA client connection
    - Account state

    This is created once per trading session and reused across multiple plans.

    Attributes:
        config: Application configuration dict.
        policy: FX trading policy (from fx_guardrails).
        client: OANDA practice client.
        state: Current trading state (entries today, etc.).
        pnl: Current P&L and account metrics.
        instrument: Target instrument.
        granularity: Target timeframe.
    """
    config: Dict[str, Any]
    policy: Any  # FxPolicy
    client: Any  # OandaPracticeClient
    state: Any  # FxState
    pnl: Dict[str, Any]
    instrument: str
    granularity: str

    @property
    def nav(self) -> float:
        """Current account NAV."""
        return float(self.pnl.get("nav", 0) or 0)

    @property
    def balance(self) -> float:
        """Current account balance."""
        return float(self.pnl.get("balance", 0) or 0)

    @property
    def equity(self) -> float:
        """Current account equity (prefer NAV)."""
        return self.nav if self.nav > 0 else self.balance

    @property
    def drawdown_pct(self) -> Optional[float]:
        """Current drawdown percentage if available."""
        dd = self.pnl.get("drawdown_pct")
        return float(dd) if dd is not None else None


# =============================================================================
# Utility Functions
# =============================================================================

def _pip_size(instrument: str) -> float:
    """Get pip size for an instrument."""
    return 0.01 if instrument.endswith("_JPY") else 0.0001


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range.

    Args:
        df: OHLCV DataFrame.
        period: ATR period (default 14).

    Returns:
        ATR value.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_series = tr.rolling(period).mean()
    value = float(atr_series.iloc[-1])

    if not np.isfinite(value) or value <= 0:
        # Fallback to simple range
        return float((high - low).abs().mean())

    return value


def _calculate_spread_pips(df: pd.DataFrame, instrument: str) -> Optional[float]:
    """Calculate spread in pips from bid/ask closes.

    Args:
        df: DataFrame with bid_close and ask_close columns.
        instrument: OANDA instrument name.

    Returns:
        Spread in pips, or None if unavailable.
    """
    if "bid_close" not in df.columns or "ask_close" not in df.columns:
        return None

    bid = df["bid_close"].iloc[-1]
    ask = df["ask_close"].iloc[-1]

    if pd.isna(bid) or pd.isna(ask):
        return None

    pip = _pip_size(instrument)
    return (float(ask) - float(bid)) / pip


def _conservative_slippage_pips(
    spread_pips: float,
    min_pips: float = 0.5,
    spread_mult: float = 0.5,
) -> float:
    """Estimate conservative slippage based on spread.

    Args:
        spread_pips: Current spread in pips.
        min_pips: Minimum slippage assumption.
        spread_mult: Multiplier of spread to add.

    Returns:
        Estimated slippage in pips.
    """
    return max(min_pips, spread_pips * spread_mult)


def _setup_signal(df: pd.DataFrame) -> Literal["buy", "sell", "hold"]:
    """Simple rule-based setup signal (placeholder for ML).

    Uses EMA crossover and RSI for basic signal generation.
    This is intentionally conservative and meant for testing.

    Args:
        df: OHLCV DataFrame.

    Returns:
        Signal: "buy", "sell", or "hold".
    """
    if len(df) < 50:
        return "hold"

    close = df["close"].astype(float)

    # EMA crossover
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()

    # Simple RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))

    current_rsi = float(rsi.iloc[-1])
    ema_diff = float(ema_fast.iloc[-1] - ema_slow.iloc[-1])
    ema_prev_diff = float(ema_fast.iloc[-2] - ema_slow.iloc[-2])

    # Conservative signals
    if ema_diff > 0 and ema_prev_diff <= 0 and current_rsi < 70:
        return "buy"
    if ema_diff < 0 and ema_prev_diff >= 0 and current_rsi > 30:
        return "sell"

    return "hold"


def _position_size_units(
    instrument: str,
    equity: float,
    risk_per_trade_pct: float,
    stop_distance_price: float,
    price: float,
) -> int:
    """Calculate position size in units.

    Uses basic formula: risk_amount / stop_distance_price

    Args:
        instrument: OANDA instrument name.
        equity: Account equity.
        risk_per_trade_pct: Risk as decimal (e.g., 0.02 for 2%).
        stop_distance_price: Stop loss distance in price units.
        price: Current price.

    Returns:
        Position size in units.
    """
    if stop_distance_price <= 0:
        return 0

    risk_amount = equity * risk_per_trade_pct
    units = risk_amount / stop_distance_price

    # Clamp to reasonable bounds
    min_units = 1
    max_units = 10_000_000  # 100 lots max

    return min(max_units, max(min_units, int(units)))


# =============================================================================
# Main Functions
# =============================================================================

def setup_paper_trade_context(
    config: Dict[str, Any],
    *,
    instrument: str,
    granularity: str,
    execute: bool = False,
    console_print: Optional[Callable[[str], None]] = None,
) -> Optional[PaperTradeContext]:
    """Setup context for paper trading.

    Validates policy, connects to OANDA, checks account state,
    and prepares for trade execution.

    Args:
        config: Application configuration dict.
        instrument: OANDA instrument name.
        granularity: Candle timeframe.
        execute: Whether execution is intended (affects force-flat).
        console_print: Optional print function for output.

    Returns:
        PaperTradeContext if setup succeeds, None otherwise.
    """
    _print = console_print or (lambda x: None)

    try:
        from src.utils.oanda_practice import OandaPracticeClient
        import fx_guardrails as fxg
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        _print(f"[red]Error: Missing required modules: {e}[/red]")
        return None

    # Load and validate policy
    try:
        policy = fxg.load_fx_policy(config)
    except Exception as e:
        logger.error(f"Failed to load FX policy: {e}")
        _print(f"[red]Error: Failed to load FX policy: {e}[/red]")
        return None

    # Validate instrument
    if instrument not in set(policy.instruments):
        _print(f"[bold red]Blocked[/bold red]: instrument not allowed ({instrument}).")
        _print(f"Allowed: {', '.join(policy.instruments)}")
        return None

    # Validate granularity
    if granularity != policy.granularity:
        _print(f"[bold red]Blocked[/bold red]: granularity must be {policy.granularity} (got {granularity}).")
        return None

    # Connect to OANDA
    try:
        client = OandaPracticeClient.from_env()
    except Exception as e:
        logger.error(f"Failed to connect to OANDA: {e}")
        _print(f"[red]Error: Failed to connect to OANDA: {e}[/red]")
        return None

    # Load trading state
    state = fxg.load_state(config, policy)

    # Refresh account state
    try:
        pnl = fxg.update_state_from_account_summary(policy, state, client.get_account_summary())
    except Exception as e:
        logger.error(f"Failed to refresh account state: {e}")
        _print(f"[red]Error: Failed to get account info: {e}[/red]")
        return None

    # Check daily stops
    stop_hit, stop_reason, stop_kind = fxg.check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=pnl.get("realized_pct"),
        confidence_band="medium",
    )
    if stop_hit and stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        fxg.save_state(config, state)
        _print(f"[bold red]Trading disabled[/bold red]: {stop_reason}")
        return None

    # Check force-flat
    if fxg.should_force_flat(policy) or (state.disabled_reason and getattr(state, "disabled_kind", None) == "loss"):
        positions = client.get_open_positions()
        open_instruments = [p.get("instrument") for p in (positions.get("positions") or []) if p.get("instrument")]

        if open_instruments:
            _print("[yellow]Force-flat: open positions detected.[/yellow]")
            if not execute:
                _print("[dim]Dry-run: would close all open positions.[/dim]")
            else:
                for inst in open_instruments:
                    try:
                        client.close_position(instrument=inst)
                        _print(f"Closed position: {inst}")
                    except Exception as e:
                        _print(f"[red]Failed closing {inst}: {e}[/red]")
            return None

    # Check entry gate
    ok, why = fxg.can_open_new_trade(policy, state)
    if not ok:
        _print(f"[bold red]Blocked[/bold red]: {why}")
        return None

    # Check max positions
    positions = client.get_open_positions()
    open_instruments = [p.get("instrument") for p in (positions.get("positions") or []) if p.get("instrument")]
    if len(open_instruments) >= policy.limits.max_open_positions:
        _print("[bold red]Blocked[/bold red]: max open positions reached.")
        return None

    # Validate account metrics
    nav = pnl.get("nav")
    balance = pnl.get("balance")
    if nav is None or balance is None:
        _print("[bold red]Blocked[/bold red]: missing account NAV/balance from broker.")
        return None

    return PaperTradeContext(
        config=config,
        policy=policy,
        client=client,
        state=state,
        pnl=pnl,
        instrument=instrument,
        granularity=granularity,
    )


def build_paper_trade_plan(
    context: PaperTradeContext,
    *,
    candles: int = 300,
    equity: Optional[float] = None,
    risk_per_trade_pct: Optional[float] = None,
    console_print: Optional[Callable[[str], None]] = None,
) -> Optional[PaperTradePlan]:
    """Build a paper trade plan from market analysis.

    Fetches market data, analyzes for setups, and computes position sizing
    with full risk management.

    Args:
        context: Paper trade context from setup_paper_trade_context.
        candles: Number of candles to fetch.
        equity: Override account equity.
        risk_per_trade_pct: Override risk per trade.
        console_print: Optional print function for output.

    Returns:
        PaperTradePlan if valid setup found, None otherwise.
    """
    _print = console_print or (lambda x: None)

    try:
        from src.utils.fx_paper import candles_to_ohlcv_df
        import fx_guardrails as fxg
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return None

    instrument = context.instrument
    granularity = context.granularity
    policy = context.policy
    client = context.client
    state = context.state
    pnl = context.pnl
    config = context.config

    # Fetch market data
    try:
        resp = client.get_candles(instrument, granularity=granularity, count=candles, price="MBA")
        df = candles_to_ohlcv_df(resp)
    except Exception as e:
        logger.error(f"Failed to fetch candles: {e}")
        _print(f"[red]Error: Failed to fetch market data: {e}[/red]")
        return None

    # Check spread
    spread_pips = _calculate_spread_pips(df, instrument)
    if spread_pips is None:
        # Use fallback from policy
        spread_pips = float(policy.costs.spread_fallback_pips.get(instrument, 0.0) or 0.0)

    max_spread = float(policy.costs.max_spread_pips.get(instrument, 0.0) or 0.0)
    if max_spread > 0 and spread_pips > max_spread:
        _print(f"[bold red]Blocked[/bold red]: spread too wide ({spread_pips:.2f} pips > {max_spread:.2f} pips).")
        return None

    # Calculate slippage
    slippage_pips = _conservative_slippage_pips(
        spread_pips,
        min_pips=float(policy.costs.slippage_pips_min),
        spread_mult=float(policy.costs.slippage_pips_spread_mult),
    )
    pip = _pip_size(instrument)
    slippage_price = slippage_pips * pip

    # Get signal
    signal = _setup_signal(df)
    last_close = float(df["close"].iloc[-1])
    atr_value = _calculate_atr(df, period=14)

    if signal == "hold":
        _print(f"FX setup: HOLD  {instrument} {granularity}  close={last_close:.5f}  ATR14={atr_value:.5f}")
        return None

    # Build risk rules
    nav = pnl.get("nav")
    base_equity = float(nav) if nav is not None else float(equity or 10000.0)

    if equity is not None:
        base_equity = float(equity)

    rpt = risk_per_trade_pct or float(policy.risk.risk_per_trade_pct)

    # Calculate confidence
    try:
        from reasoning_enhanced import fx_confidence_v1
    except ImportError:
        # Simple fallback confidence
        confidence = 0.5
        conf_reasons = ["Confidence model unavailable"]
    else:
        max_spread_for_conf = max_spread if max_spread > 0 else max(1e-9, spread_pips)
        params = dict((config.get("fx", {}) or {}).get("confidence_model") or {})

        payload = fx_confidence_v1(
            signal=signal,
            price=last_close,
            atr=atr_value,
            spread_pips=spread_pips,
            max_spread_pips=max_spread_for_conf,
            params=params,
        )
        confidence = float(payload.get("confidence") or 0.0)
        conf_reasons = [str(x) for x in (payload.get("reasons") or [])]

    # Get confidence band
    band = fxg.confidence_band(policy, confidence)

    if band == "low":
        _print(f"[bold red]Blocked[/bold red]: low confidence ({confidence:.2f}) => no trades.")
        _print(f"[dim]Reasons: {', '.join(conf_reasons)}[/dim]")
        return None

    # Check daily stops with band
    stop_hit, stop_reason, stop_kind = fxg.check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=pnl.get("realized_pct"),
        confidence_band=band,
    )
    if stop_hit and stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        fxg.save_state(config, state)
        _print(f"[bold red]Trading disabled[/bold red]: {stop_reason}")
        return None

    # Calculate stop and take profit
    atr_stop_mult = float(getattr(policy.risk, "atr_stop_mult", 1.5))
    rr_take_profit = float(getattr(policy.risk, "rr_take_profit", 1.5))

    stop_distance = (atr_value * atr_stop_mult) + slippage_price
    if stop_distance <= 0:
        stop_distance = 15.0 * pip  # Fallback

    tp_distance = stop_distance * rr_take_profit

    if signal == "buy":
        stop_price = last_close - stop_distance
        tp_price = last_close + tp_distance
        side = 1
    else:
        stop_price = last_close + stop_distance
        tp_price = last_close - tp_distance
        side = -1

    # Calculate position size
    units_abs = _position_size_units(
        instrument=instrument,
        equity=base_equity,
        risk_per_trade_pct=rpt,
        stop_distance_price=stop_distance,
        price=last_close,
    )

    units = side * units_abs

    return PaperTradePlan(
        instrument=instrument,
        granularity=granularity,
        signal=signal,
        last_close=last_close,
        atr_value=atr_value,
        units=units,
        stop_price=stop_price,
        tp_price=tp_price,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        slippage_price=slippage_price,
        confidence=confidence,
        band=band,
        conf_reasons=conf_reasons,
    )


def execute_paper_trade_plan(
    context: PaperTradeContext,
    plan: PaperTradePlan,
    *,
    execute: bool = False,
    force_units: Optional[int] = None,
    verbose: bool = False,
    max_hold_minutes: Optional[float] = None,
    console_print: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute a paper trade plan.

    Places the order on OANDA PRACTICE or performs a dry-run.

    Args:
        context: Paper trade context.
        plan: Trade plan to execute.
        execute: If True, place real order; if False, dry-run only.
        force_units: Override position size.
        verbose: Enable verbose output.
        max_hold_minutes: Auto-close after this many minutes.
        console_print: Optional print function for output.

    Returns:
        Order result dict if executed, None otherwise.
    """
    _print = console_print or (lambda x: None)

    try:
        import fx_guardrails as fxg
    except ImportError as e:
        logger.error(f"Failed to import fx_guardrails: {e}")
        return None

    policy = context.policy
    client = context.client
    state = context.state
    config = context.config

    # Display plan
    _print(
        f"FX setup: {plan.signal.upper()}  {plan.instrument} {plan.granularity}  "
        f"close={plan.last_close:.5f}  units={plan.units}  "
        f"SL={plan.stop_price:.5f}  TP={plan.tp_price:.5f}  "
        f"spread={plan.spread_pips:.2f}p  slip={plan.slippage_pips:.2f}p  "
        f"conf={plan.confidence:.2f}  band={plan.band}"
    )

    if not execute:
        _print("[dim]Dry-run (no order). Pass --execute to place a PRACTICE order.[/dim]")
        return None

    # Confirmation if required
    if policy.require_confirmation:
        try:
            ans = input("Confirm PLACE ORDER? (y/N): ") or ""
            if ans.strip().lower() != "y":
                _print("[dim]Order cancelled.[/dim]")
                return None
        except Exception:
            _print("[dim]Order cancelled (input error).[/dim]")
            return None

    # Apply force units override
    units = plan.units
    if force_units is not None:
        units = -abs(force_units) if plan.signal == "sell" else abs(force_units)
        if verbose:
            _print(f"[dim]Force-units override[/dim]: using units={units}")

    # Ensure integer units
    try:
        units_final = int(units)
    except Exception:
        units_final = int(float(units))

    if verbose:
        _print(f"[dim]Order sizing[/dim]: computed_units={plan.units} final_submitted_units={units_final}")

    # Get price bound
    pip = _pip_size(plan.instrument)
    try:
        quote = client.get_price_quote(instrument=plan.instrument)
        bid = float(quote["bid"])
        ask = float(quote["ask"])

        buffer_pips = 1.0  # Default buffer
        buffer_price = buffer_pips * pip

        if units_final > 0:
            price_bound = ask + buffer_price
        else:
            price_bound = bid - buffer_price
    except Exception as e:
        _print(f"[bold red]Blocked[/bold red]: could not fetch live quote: {e}")
        return None

    # Place order
    try:
        result = client.create_market_order(
            instrument=plan.instrument,
            units=units_final,
            stop_loss_price=float(plan.stop_price),
            take_profit_price=float(plan.tp_price),
            price_bound=price_bound,
        )
    except Exception as e:
        logger.error(f"Order execution failed: {e}")
        _print(f"[bold red]Order failed[/bold red]: {e}")
        return None

    # Update state
    state.entries_today += 1
    fxg.save_state(config, state)

    _print("[bold green]Order submitted (PRACTICE).[/bold green]")
    _print(str(result))

    # Parse trade ID for auto-close
    try:
        tx = result.get("orderFillTransaction") or result.get("orderCreateTransaction") or {}
        trade_id = None
        if isinstance(tx, dict):
            to = tx.get("tradeOpened")
            if isinstance(to, dict):
                trade_id = to.get("tradeID") or to.get("id")
            tro = tx.get("tradesOpened")
            if trade_id is None and isinstance(tro, list) and len(tro) > 0:
                trade_id = tro[0].get("tradeID") or tro[0].get("id")
        if trade_id:
            client._last_trade_id = str(trade_id)
    except Exception:
        pass

    # Schedule auto-close if requested
    if max_hold_minutes is not None and max_hold_minutes > 0:
        try:
            from src.trading.execution import TradeExecutor
            executor = TradeExecutor(client)
            executor.schedule_auto_close(
                plan.instrument,
                delay_seconds=max_hold_minutes * 60.0,
                verbose=verbose,
            )
        except Exception as e:
            logger.warning(f"Failed to schedule auto-close: {e}")

    return result
