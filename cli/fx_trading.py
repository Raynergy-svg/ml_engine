#!/usr/bin/env python3
"""FX trading execution and paper trading utilities.

This module provides:
- FX paper trading with OANDA PRACTICE
- Policy enforcement and risk management
- Confidence-based trade gating
- Legacy integrated prediction compatibility shims
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel

from cli.io_utils import (
    console,
    logger,
    load_config,
    BUDDY_META_FILENAME,
    TABLE_HEADER_STYLE,
)

if TYPE_CHECKING:
    # NeuralNetworkIntegrator is a legacy class; avoid import at runtime.
    pass


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.65
DEFAULT_SEQUENCE_LENGTH: Final[int] = 60
DEFAULT_ATR_PERIOD: Final[int] = 14
DEFAULT_SMA_PERIOD: Final[int] = 20
DEFAULT_RSI_PERIOD: Final[int] = 14
DEFAULT_BB_PERIOD: Final[int] = 20
DEFAULT_BB_STD_DEV: Final[float] = 2.0
DEFAULT_PRICE_BOUND_BUFFER_PIPS: Final[float] = 1.0
MIN_SPREAD_PIPS: Final[float] = 1e-9
NUMPY_EPSILON: Final[float] = 1e-8

# Column name mappings for normalization
OHLCV_COLUMN_MAP: Final[Dict[str, str]] = {
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "adj close": "close",
    "adj_close": "close",
    "adjclose": "close",
    "volume": "volume",
    "vol": "volume",
    "v": "volume",
}

SLM_COLUMN_MAP: Final[Dict[str, str]] = {
    "close": "Close",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "volume": "Volume",
}

# =============================================================================
# Custom Exceptions
# =============================================================================


class FXTradingError(Exception):
    """Base exception for FX trading operations."""
    pass


class PolicyViolationError(FXTradingError):
    """Raised when FX policy constraints are violated."""
    pass


class AccountMetricsError(FXTradingError):
    """Raised when account metrics cannot be retrieved."""
    pass


class MarketDataError(FXTradingError):
    """Raised when market data operations fail."""
    pass


class OrderExecutionError(FXTradingError):
    """Raised when order execution fails."""
    pass


class ConfigurationError(FXTradingError):
    """Raised when configuration is invalid."""
    pass


# =============================================================================
# Enums
# =============================================================================

class Signal(str, Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ConfidenceBand(str, Enum):
    """Confidence band classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StopKind(str, Enum):
    """Types of trading stops."""
    LOSS = "loss"
    PROFIT = "profit"
    TIME = "time"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True, slots=True)
class FxPaperTradePlan:
    """Immutable plan for a paper trade execution."""
    instrument: str
    granularity: str
    signal: str
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
    conf_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OrderPricing:
    """Pricing information for order execution."""
    stop_distance: float
    tp_distance: float
    stop_price: float
    tp_price: float
    side: int  # 1 for buy, -1 for sell


@dataclass(frozen=True, slots=True)
class SpreadAndSlippage:
    """Spread and slippage calculations."""
    is_valid: bool
    spread_pips: float
    slippage_pips: float
    slippage_price: float


@dataclass(frozen=True, slots=True)
class SignalContext:
    """Context information for a trading signal."""
    signal: str
    last_close: float
    atr_value: float


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """Result of confidence calculation."""
    confidence: float
    band: str
    reasons: list[str]


# =============================================================================
# Configuration Helpers
# =============================================================================

def _buddy_live_enabled_from_meta() -> tuple[bool, float]:
    """Return (live_enabled, confidence_threshold) from last Buddy training.

    Returns:
        Tuple of (is_live_enabled, confidence_threshold)

    Raises:
        ConfigurationError: If meta file exists but is corrupted.
    """
    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    
    if not meta_path.exists():
        return False, DEFAULT_CONFIDENCE_THRESHOLD
    
    try:
        meta = json.loads(meta_path.read_text())
        live_enabled = bool(meta.get("live_enabled", False))
        confidence_threshold = float(
            meta.get("live_confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        )
        return live_enabled, confidence_threshold
    except (json.JSONDecodeError, ValueError, KeyError, IOError) as e:
        logger.warning(f"Failed to parse buddy meta file: {e}")
        return False, DEFAULT_CONFIDENCE_THRESHOLD


def _get_policy_config_value(
    policy: Any,
    config_path: str,
    default: Any = None,
    *,
    required: bool = False,
) -> Any:
    """Safely retrieve a nested configuration value from policy.

    Args:
        policy: Policy object or dict
        config_path: Dot-separated path (e.g., 'costs.max_spread_pips')
        default: Default value if not found
        required: If True, raise ConfigurationError when not found

    Returns:
        The configuration value or default

    Raises:
        ConfigurationError: If required value is not found
    """
    value = policy
    parts = config_path.split(".")
    
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        
        if value is None:
            if required:
                raise ConfigurationError(f"Required policy configuration not found: {config_path}")
            return default
    
    return value


# =============================================================================
# User Interaction
# =============================================================================

def _fx_confirm(prompt: str) -> bool:
    """Prompt user for yes/no confirmation.

    Args:
        prompt: The confirmation prompt to display

    Returns:
        True if user confirms with 'y', False otherwise
    """
    ans = (console.input(prompt) or "").strip().lower()
    return ans == "y"


# =============================================================================
# Position Management
# =============================================================================

def _fx_open_position_instruments(payload: Any) -> list[str]:
    """Extract list of instruments from open positions payload.

    Args:
        payload: Response payload containing positions data

    Returns:
        List of instrument names with open positions
    """
    positions = (payload or {}).get("positions") or []
    return [
        str(p.get("instrument"))
        for p in positions
        if p and p.get("instrument")
    ]


# =============================================================================
# Policy Enforcement
# =============================================================================

def _fx_enforce_fx_policy(
    cfg: Dict[str, Any],
    *,
    instrument: str,
    granularity: str,
) -> Optional[Any]:
    """Validate instrument and granularity against FX policy.

    Args:
        cfg: Configuration dictionary
        instrument: FX instrument to validate
        granularity: Timeframe granularity to validate

    Returns:
        Policy object if validation passes, None otherwise

    Raises:
        PolicyViolationError: If policy loading fails
    """
    try:
        from fx_guardrails import load_fx_policy
    except ImportError as e:
        raise PolicyViolationError(f"Failed to import fx_guardrails: {e}") from e

    policy = load_fx_policy(cfg)
    
    # Validate instrument
    allowed_instruments = set(getattr(policy, "instruments", []))
    if instrument not in allowed_instruments:
        console.print(
            f"[bold red]Blocked[/bold red]: instrument not allowed ({instrument})."
        )
        console.print(f"Allowed: {', '.join(sorted(allowed_instruments))}")
        return None
    
    # Validate granularity
    policy_granularity = getattr(policy, "granularity", None)
    if granularity != policy_granularity:
        console.print(
            f"[bold red]Blocked[/bold red]: granularity must be {policy_granularity} "
            f"(got {granularity})."
        )
        return None
    
    return policy


def _fx_gate_fx_entry(policy: Any, state: Any, client: Any) -> bool:
    """Check if new trade entry is allowed based on policy and state.

    Args:
        policy: FX policy object
        state: Trading state object
        client: OANDA client instance

    Returns:
        True if entry is allowed, False otherwise
    """
    try:
        from fx_guardrails import can_open_new_trade
    except ImportError as e:
        logger.error(f"Failed to import can_open_new_trade: {e}")
        return False

    is_allowed, reason = can_open_new_trade(policy, state)
    if not is_allowed:
        console.print(f"[bold red]Blocked[/bold red]: {reason}")
        return False

    # Check max open positions
    open_instruments = _fx_open_position_instruments(client.get_open_positions())
    max_positions = _get_policy_config_value(
        policy, "limits.max_open_positions", default=1
    )
    
    if len(open_instruments) >= max_positions:
        console.print("[bold red]Blocked[/bold red]: max open positions reached.")
        return False

    return True


# =============================================================================
# State Management
# =============================================================================

def _fx_refresh_fx_state(
    cfg: Dict[str, Any],
    policy: Any,
    state: Any,
    client: Any,
) -> tuple[dict[str, Any], str]:
    """Refresh FX trading state with current account data.

    Args:
        cfg: Configuration dictionary
        policy: FX policy object
        state: Trading state object
        client: OANDA client instance

    Returns:
        Tuple of (pnl_dict, confidence_band)

    Raises:
        AccountMetricsError: If account summary cannot be retrieved
    """
    try:
        from fx_guardrails import check_daily_stops, update_state_from_account_summary
    except ImportError as e:
        raise AccountMetricsError(f"Failed to import fx_guardrails: {e}") from e

    try:
        account_summary = client.get_account_summary()
        pnl = update_state_from_account_summary(policy, state, account_summary)
    except Exception as e:
        raise AccountMetricsError(f"Failed to get account summary: {e}") from e

    # Evaluate daily circuit breakers early
    stop_hit, stop_reason, stop_kind = check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=pnl.get("realized_pct"),
        confidence_band="medium",
    )
    
    if stop_hit and stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        try:
            from fx_guardrails import save_state
            save_state(cfg, state)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        console.print(f"[bold red]Trading disabled[/bold red]: {stop_reason}")

    # Band is computed later; keep placeholder here for display
    return pnl, "unknown"


def _fx_maybe_force_flat(
    policy: Any,
    state: Any,
    client: Any,
    *,
    execute: bool,
) -> bool:
    """Force close all positions if required by policy or state.

    Args:
        policy: FX policy object
        state: Trading state object
        client: OANDA client instance
        execute: If True, actually close positions; if False, dry-run

    Returns:
        True if positions were closed or no positions existed, False otherwise
    """
    try:
        from fx_guardrails import should_force_flat
    except ImportError as e:
        logger.error(f"Failed to import should_force_flat: {e}")
        return False

    # Only force-flat automatically for the time cutoff or loss-stop
    force_flat_due_to_stop = (
        bool(state.disabled_reason) and
        getattr(state, "disabled_kind", None) == StopKind.LOSS.value
    )
    
    if not should_force_flat(policy) and not force_flat_due_to_stop:
        return False

    open_instruments = _fx_open_position_instruments(client.get_open_positions())
    if not open_instruments:
        console.print("[dim]No open positions to close.[/dim]")
        return True

    console.print("[yellow]Force-flat: open positions detected.[/yellow]")
    
    if not execute:
        console.print("[dim]Dry-run: would close all open positions.[/dim]")
        return True

    require_confirmation = getattr(policy, "require_confirmation", False)
    if require_confirmation and not _fx_confirm("Confirm CLOSE ALL positions? (y/N): "):
        console.print("[dim]Close cancelled.[/dim]")
        return True

    # Close all positions
    closed_count = 0
    for instrument in open_instruments:
        try:
            client.close_position(instrument=instrument)
            console.print(f"Closed position: {instrument}")
            closed_count += 1
        except Exception as e:
            console.print(f"[bold red]Failed[/bold red] closing {instrument}: {e}")
            logger.error(f"Failed to close position {instrument}: {e}")

    return closed_count > 0


def _fx_apply_daily_stops(
    cfg: Dict[str, Any],
    policy: Any,
    state: Any,
    pnl: Dict[str, Any],
    *,
    band: str,
) -> bool:
    """Check and apply daily stop limits based on confidence band.

    Args:
        cfg: Configuration dictionary
        policy: FX policy object
        state: Trading state object
        pnl: PnL dictionary with drawdown and realized percentages
        band: Current confidence band

    Returns:
        True if stop was triggered, False otherwise
    """
    try:
        from fx_guardrails import check_daily_stops, save_state
    except ImportError as e:
        logger.error(f"Failed to import fx_guardrails: {e}")
        return False

    stop_hit, stop_reason, stop_kind = check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=pnl.get("realized_pct"),
        confidence_band=str(band),
    )
    
    if not stop_hit:
        return False

    if stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        try:
            save_state(cfg, state)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        console.print(f"[bold red]Trading disabled[/bold red]: {stop_reason}")
    
    return True


# =============================================================================
# Account Metrics
# =============================================================================

def _fx_require_account_metrics(pnl: Dict[str, Any]) -> bool:
    """Validate that required account metrics are available.

    Args:
        pnl: PnL dictionary from broker

    Returns:
        True if metrics are valid, False otherwise

    Raises:
        AccountMetricsError: If required metrics are missing
    """
    nav = pnl.get("nav")
    balance = pnl.get("balance")
    
    if nav is None or balance is None:
        console.print("[bold red]Blocked[/bold red]: missing account NAV/balance from broker.")
        console.print(f"[dim]pnl payload keys: {sorted((pnl or {}).keys())}[/dim]")
        return False
    
    return True


# =============================================================================
# Market Data Loading
# =============================================================================

def _fx_load_fx_df(
    client: Any,
    *,
    instrument: str,
    granularity: str,
    candles: int,
) -> pd.DataFrame:
    """Load OHLCV data for FX instrument.

    Args:
        client: OANDA client instance
        instrument: FX instrument symbol
        granularity: Timeframe granularity
        candles: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data

    Raises:
        MarketDataError: If data fetch fails
    """
    try:
        from fx_paper import candles_to_ohlcv_df
    except ImportError as e:
        raise MarketDataError(f"Failed to import fx_paper: {e}") from e

    try:
        response = client.get_candles(
            instrument,
            granularity=granularity,
            count=candles,
            price="MBA"
        )
        return candles_to_ohlcv_df(response)
    except Exception as e:
        raise MarketDataError(
            f"Failed to load candles for {instrument} {granularity}: {e}"
        ) from e


# =============================================================================
# Signal and Analysis
# =============================================================================

def _fx_get_signal_context(df: pd.DataFrame) -> SignalContext:
    """Calculate trading signal and market context.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        SignalContext with signal, last close price, and ATR value

    Raises:
        MarketDataError: If signal calculation fails
    """
    try:
        from fx_paper import atr as fx_atr
        from fx_paper import setup_signal
    except ImportError as e:
        raise MarketDataError(f"Failed to import fx_paper: {e}") from e

    try:
        signal = setup_signal(df)
        last_close = float(df["close"].iloc[-1])
        atr_value = float(fx_atr(df, period=DEFAULT_ATR_PERIOD))
        return SignalContext(
            signal=str(signal),
            last_close=last_close,
            atr_value=atr_value,
        )
    except (IndexError, KeyError, ValueError) as e:
        raise MarketDataError(f"Failed to calculate signal context: {e}") from e


# =============================================================================
# Spread and Slippage
# =============================================================================

def _fx_spread_and_slippage(
    policy: Any,
    df: pd.DataFrame,
    *,
    instrument: str,
) -> SpreadAndSlippage:
    """Calculate spread and slippage for instrument.

    Args:
        policy: FX policy object
        df: DataFrame with OHLCV data
        instrument: FX instrument symbol

    Returns:
        SpreadAndSlippage object with calculations

    Raises:
        MarketDataError: If calculation fails
    """
    try:
        from fx_paper import (
            conservative_slippage_pips,
            pip_size,
            spread_pips_from_df,
        )
    except ImportError as e:
        raise MarketDataError(f"Failed to import fx_paper: {e}") from e

    # Get spread from data or fallback
    spread_pips = spread_pips_from_df(df, instrument)
    if spread_pips is None:
        spread_pips = float(
            _get_policy_config_value(
                policy,
                "costs.spread_fallback_pips",
                default=0.0
            ).get(instrument, 0.0) or 0.0
        )

    # Check max spread limit
    max_spread = float(
        _get_policy_config_value(
            policy,
            "costs.max_spread_pips",
            default={}
        ).get(instrument, 0.0) or 0.0
    )
    
    if max_spread > 0 and spread_pips > max_spread:
        console.print(
            f"[bold red]Blocked[/bold red]: spread too wide "
            f"({spread_pips:.2f} pips > {max_spread:.2f} pips)."
        )
        return SpreadAndSlippage(
            is_valid=False,
            spread_pips=float(spread_pips),
            slippage_pips=0.0,
            slippage_price=0.0,
        )

    # Calculate slippage
    slippage_pips = conservative_slippage_pips(
        spread_pips=spread_pips,
        min_pips=_get_policy_config_value(
            policy, "costs.slippage_pips_min", default=0.0
        ),
        spread_mult=_get_policy_config_value(
            policy, "costs.slippage_pips_spread_mult", default=1.0
        ),
    )
    slippage_price = slippage_pips * pip_size(instrument)

    return SpreadAndSlippage(
        is_valid=True,
        spread_pips=float(spread_pips),
        slippage_pips=float(slippage_pips),
        slippage_price=float(slippage_price),
    )


# =============================================================================
# Risk Management
# =============================================================================

def _fx_build_risk_rules(
    policy: Any,
    pnl: Dict[str, Any],
    *,
    equity: float,
    risk_per_trade_pct: Optional[float] = None,
) -> Any:
    """Build risk rules for trade sizing.

    Args:
        policy: FX policy object
        pnl: PnL dictionary
        equity: Base equity for calculations
        risk_per_trade_pct: Override risk percentage (uses policy default if None)

    Returns:
        RiskRules object

    Raises:
        ConfigurationError: If required risk parameters are missing
    """
    try:
        from fx_paper import RiskRules
    except ImportError as e:
        raise ConfigurationError(f"Failed to import fx_paper: {e}") from e

    # Prefer live NAV when available
    nav = pnl.get("nav")
    base_equity = float(nav) if nav is not None else float(equity)

    # Use explicit CLI flag if provided, else fallback to policy
    if risk_per_trade_pct is not None:
        rpt = float(risk_per_trade_pct)
    else:
        rpt = float(
            _get_policy_config_value(
                policy, "risk.risk_per_trade_pct", required=True
            )
        )

    return RiskRules(
        equity=base_equity,
        risk_per_trade_pct=rpt,
        max_daily_loss_pct=float(
            _get_policy_config_value(policy, "risk.daily_loss_stop_pct", default=0.02)
        ),
        max_open_positions=int(
            _get_policy_config_value(policy, "limits.max_open_positions", default=1)
        ),
        atr_stop_mult=float(
            _get_policy_config_value(policy, "risk.atr_stop_mult", default=1.5)
        ),
        rr_take_profit=float(
            _get_policy_config_value(policy, "risk.rr_take_profit", default=1.5)
        ),
    )


def _fx_build_order_units_and_prices(
    policy: Any,
    rules: Any,
    *,
    instrument: str,
    signal: str,
    price: float,
    atr_value: float,
    slippage_price: float,
) -> Tuple[int, OrderPricing]:
    """Calculate order units and pricing for trade.

    Args:
        policy: FX policy object
        rules: Risk rules object
        instrument: FX instrument symbol
        signal: Trading signal (buy/sell)
        price: Current price
        atr_value: ATR value
        slippage_price: Slippage in price units

    Returns:
        Tuple of (units, OrderPricing object)

    Raises:
        ValueError: If stop distance is not positive
        ConfigurationError: If required parameters are missing
    """
    try:
        from fx_paper import position_size_units
    except ImportError as e:
        raise ConfigurationError(f"Failed to import fx_paper: {e}") from e

    # Calculate stop distance
    atr_mult = float(
        _get_policy_config_value(policy, "risk.atr_stop_mult", default=1.5)
    )
    stop_distance = float(atr_value) * atr_mult
    stop_distance = float(max(stop_distance, 0.0)) + float(max(slippage_price, 0.0))
    
    if stop_distance <= 0:
        raise ValueError("Computed stop distance is not positive")

    # Calculate take profit distance
    rr_ratio = float(
        _get_policy_config_value(policy, "risk.rr_take_profit", default=1.5)
    )
    tp_distance = stop_distance * rr_ratio

    # Calculate prices based on signal
    if signal == Signal.BUY.value:
        stop_price = float(price) - stop_distance
        tp_price = float(price) + tp_distance
        side = 1
    else:
        stop_price = float(price) + stop_distance
        tp_price = float(price) - tp_distance
        side = -1

    # Calculate position size
    units_abs = position_size_units(
        instrument=instrument,
        equity=float(getattr(rules, "equity", 0.0)),
        risk_per_trade_pct=float(getattr(rules, "risk_per_trade_pct", 0.005)),
        stop_distance_price=stop_distance,
        price=float(price),
        pip_value_per_unit=None,
    )

    pricing = OrderPricing(
        stop_distance=stop_distance,
        tp_distance=tp_distance,
        stop_price=stop_price,
        tp_price=tp_price,
        side=side,
    )

    return int(side * units_abs), pricing


# =============================================================================
# Confidence Calculation
# =============================================================================

def _fx_compute_confidence_and_band(
    cfg: Dict[str, Any],
    policy: Any,
    *,
    instrument: str,
    signal: str,
    price: float,
    atr_value: float,
    spread_pips: float,
) -> ConfidenceResult:
    """Compute confidence score and band for trade gating.

    Args:
        cfg: Configuration dictionary
        policy: FX policy object
        instrument: FX instrument symbol
        signal: Trading signal
        price: Current price
        atr_value: ATR value
        spread_pips: Current spread in pips

    Returns:
        ConfidenceResult with confidence, band, and reasons

    Raises:
        ConfigurationError: If confidence calculation fails
    """
    try:
        from fx_guardrails import confidence_band
        from reasoning_enhanced import fx_confidence_v1
    except ImportError as e:
        raise ConfigurationError(f"Failed to import confidence modules: {e}") from e

    # Get max spread for confidence calculation
    max_spread_pips = float(
        _get_policy_config_value(
            policy, "costs.max_spread_pips", default={}
        ).get(instrument, 0.0) or 0.0
    )
    
    msp = max_spread_pips if max_spread_pips > 0 else max(MIN_SPREAD_PIPS, float(spread_pips))
    
    # Get confidence model parameters
    params = dict(
        (cfg.get("fx", {}) or {}).get("confidence_model") or {}
    )

    # Calculate confidence
    try:
        payload = fx_confidence_v1(
            signal=str(signal),
            price=float(price),
            atr=float(atr_value),
            spread_pips=float(spread_pips),
            max_spread_pips=float(msp),
            params=params,
        )
    except Exception as e:
        logger.error(f"Confidence calculation failed: {e}")
        return ConfidenceResult(
            confidence=0.0,
            band=ConfidenceBand.LOW.value,
            reasons=[f"Confidence calculation error: {e}"]
        )

    confidence = float(payload.get("confidence") or 0.0)
    reasons = [str(x) for x in (payload.get("reasons") or [])]
    band = confidence_band(policy, confidence)

    return ConfidenceResult(
        confidence=confidence,
        band=str(band),
        reasons=reasons,
    )


# =============================================================================
# Order Execution
# =============================================================================

def _fx_execution_guard_price_bound(
    policy: Any,
    client: Any,
    *,
    instrument: str,
    units: int,
) -> Optional[float]:
    """Calculate price bound for order execution guard.

    Args:
        policy: FX policy object
        client: OANDA client instance
        instrument: FX instrument symbol
        units: Number of units (positive for buy, negative for sell)

    Returns:
        Price bound value, or None if quote cannot be fetched
    """
    try:
        from fx_paper import pip_size
    except ImportError as e:
        logger.error(f"Failed to import pip_size: {e}")
        return None

    # Fetch live quote
    try:
        quote = client.get_price_quote(instrument=instrument)
    except Exception as e:
        console.print(f"[bold red]Blocked[/bold red]: could not fetch live quote: {e}")
        return None

    bid = float(quote["bid"])
    ask = float(quote["ask"])

    # Get buffer configuration
    buffer_pips = None
    try:
        buffer_pips = float(getattr(policy.costs, "price_bound_buffer_pips"))
    except AttributeError:
        try:
            # Support dict-like policy.costs
            buffer_pips = float(
                (policy.get("costs") or {}).get("price_bound_buffer_pips", None)
            )
        except (AttributeError, TypeError):
            buffer_pips = None

    # Default buffer
    if buffer_pips is None:
        buffer_pips = DEFAULT_PRICE_BOUND_BUFFER_PIPS

    buffer_price = float(buffer_pips) * pip_size(instrument)

    # Calculate bound based on direction
    if int(units) > 0:
        return float(ask + buffer_price)
    return float(bid - buffer_price)


def _schedule_auto_close(
    client: Any,
    instrument: str,
    delay_s: float,
    *,
    verbose: bool = False,
) -> None:
    """Spawn a daemon thread to close position after delay.

    This is a best-effort helper for PRACTICE mode to ensure scalping-style
    trades are not left open beyond the desired timeframe.

    Args:
        client: OANDA client instance
        instrument: FX instrument symbol
        delay_s: Delay in seconds before closing
        verbose: If True, log verbose messages
    """
    def _worker() -> None:
        """Worker function that runs in the daemon thread."""
        try:
            if verbose:
                console.print(
                    f"[dim]Auto-close thread[/dim]: sleeping {delay_s:.1f}s "
                    f"before closing {instrument}"
                )
            time.sleep(max(0.0, float(delay_s)))
            
            # Try to close the specific trade first
            trade_id = getattr(client, "_last_trade_id", None)
            if (
                hasattr(client, "close_trade") and
                trade_id
            ):
                try:
                    result = client.close_trade(trade_id=trade_id)
                    console.print(
                        f"[dim]Auto-close[/dim]: closed trade {trade_id} "
                        f"for {instrument}: {result}"
                    )
                    return
                except Exception:
                    # Fall back to closing by instrument
                    logger.debug("close_trade failed, falling back to close_position")
            
            # Fallback: close by instrument
            result = client.close_position(instrument=instrument)
            console.print(
                f"[dim]Auto-close[/dim]: closed position for {instrument}: {result}"
            )
        except Exception as e:
            console.print(
                f"[yellow]Auto-close failed[/yellow]: could not close {instrument}: {e}"
            )
            logger.error(f"Auto-close failed for {instrument}: {e}")

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name=f"auto-close-{instrument}"
    )
    thread.start()


# =============================================================================
# Trade Plan Execution
# =============================================================================

def _fx_execute_paper_trade_plan(
    cfg: Dict[str, Any],
    policy: Any,
    client: Any,
    state: Any,
    plan: FxPaperTradePlan,
    rules: Any,
    *,
    execute: bool,
    verbose: bool,
) -> None:
    """Execute a paper trade plan.

    Args:
        cfg: Configuration dictionary
        policy: FX policy object
        client: OANDA client instance
        state: Trading state object
        plan: Trade plan to execute
        rules: Risk rules object
        execute: If True, actually place order; if False, dry-run
        verbose: If True, log verbose messages

    Raises:
        OrderExecutionError: If order execution fails
    """
    if not execute:
        console.print(
            "[dim]Dry-run (no order). Pass --execute to place a PRACTICE order.[/dim]"
        )
        return

    require_confirmation = getattr(policy, "require_confirmation", False)
    if require_confirmation and not _fx_confirm("Confirm PLACE ORDER? (y/N): "):
        console.print("[dim]Order cancelled.[/dim]")
        return

    # Calculate final units with potential override
    units = int(plan.units)
    
    try:
        if hasattr(rules, "force_units") and rules.force_units is not None:
            force_units = int(rules.force_units)
            units = -abs(force_units) if units < 0 else abs(force_units)
            console.print(
                f"[dim]Force-units override (rules.force_units)[/dim]: "
                f"using units={units}"
            )
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to apply force_units override: {e}")

    # Ensure units is an integer
    try:
        units_final = int(units)
    except (ValueError, TypeError):
        try:
            units_final = int(float(units))
        except (ValueError, TypeError) as e:
            raise OrderExecutionError(f"Invalid units value: {units}") from e

    if verbose:
        console.print(
            f"[dim]Order sizing[/dim]: computed_units={units} "
            f"final_submitted_units={units_final}"
        )

    # Get price bound
    price_bound = _fx_execution_guard_price_bound(
        policy, client, instrument=plan.instrument, units=units_final
    )
    if price_bound is None:
        raise OrderExecutionError("Could not determine price bound")

    # Submit order
    try:
        result = client.create_market_order(
            instrument=plan.instrument,
            units=units_final,
            stop_loss_price=float(plan.stop_price),
            take_profit_price=float(plan.tp_price),
            price_bound=price_bound,
        )
    except Exception as e:
        raise OrderExecutionError(f"Order submission failed: {e}") from e

    # Update state
    state.entries_today += 1
    try:
        from fx_guardrails import save_state
        save_state(cfg, state)
    except Exception as e:
        logger.error(f"Failed to save state after order: {e}")

    console.print("[bold green]Order submitted (PRACTICE).[/bold green]")
    console.print(result)

    # Extract trade ID for auto-close
    _extract_and_store_trade_id(client, result)

    # Schedule auto-close if configured
    _maybe_schedule_auto_close(cfg, client, plan, verbose)


def _extract_and_store_trade_id(client: Any, result: Dict[str, Any]) -> None:
    """Extract trade ID from order result and store on client.

    Args:
        client: OANDA client instance
        result: Order result dictionary
    """
    try:
        transaction = (
            (result or {}).get("orderFillTransaction") or
            (result or {}).get("orderCreateTransaction") or
            {}
        )
        
        if not isinstance(transaction, dict):
            return

        # Try tradeOpened structure
        trade_opened = transaction.get("tradeOpened")
        if isinstance(trade_opened, dict):
            trade_id = trade_opened.get("tradeID") or trade_opened.get("id")
            if trade_id:
                setattr(client, "_last_trade_id", str(trade_id))
                return

        # Try tradesOpened structure
        trades_opened = transaction.get("tradesOpened")
        if isinstance(trades_opened, list) and trades_opened:
            trade_id = trades_opened[0].get("tradeID") or trades_opened[0].get("id")
            if trade_id:
                setattr(client, "_last_trade_id", str(trade_id))
    except Exception as e:
        logger.debug(f"Failed to extract trade ID: {e}")


def _maybe_schedule_auto_close(
    cfg: Dict[str, Any],
    client: Any,
    plan: FxPaperTradePlan,
    verbose: bool,
) -> None:
    """Schedule auto-close if configured.

    Args:
        cfg: Configuration dictionary
        client: OANDA client instance
        plan: Trade plan
        verbose: If True, log verbose messages
    """
    try:
        max_hold_minutes = cfg.get("buddy", {}).get("max_hold_minutes")
        if max_hold_minutes is not None:
            delay_seconds = float(max_hold_minutes) * 60.0
            if delay_seconds > 0:
                _schedule_auto_close(
                    client,
                    plan.instrument,
                    delay_s=delay_seconds,
                    verbose=verbose
                )
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to schedule auto-close: {e}")


# =============================================================================
# Paper Trade Setup
# =============================================================================

def _fx_setup_paper_trade(
    cfg: Dict[str, Any],
    *,
    instrument: str,
    granularity: str,
    execute: bool,
) -> Optional[Tuple[Any, Any, Any, Dict[str, Any]]]:
    """Setup paper trading environment.

    Args:
        cfg: Configuration dictionary
        instrument: FX instrument symbol
        granularity: Timeframe granularity
        execute: If True, actual execution mode

    Returns:
        Tuple of (policy, client, state, pnl) or None if setup fails
    """
    try:
        from src.utils.oanda_practice import OandaPracticeClient
        import fx_guardrails as fxg
    except ImportError as e:
        console.print(f"[bold red]Failed[/bold red] to import required modules: {e}")
        return None

    # Enforce policy
    policy = _fx_enforce_fx_policy(cfg, instrument=instrument, granularity=granularity)
    if policy is None:
        return None

    # Initialize client and state
    client = OandaPracticeClient.from_env()
    state = fxg.load_state(cfg, policy)

    # Refresh state and check for force-flat
    try:
        pnl, _ = _fx_refresh_fx_state(cfg, policy, state, client)
    except AccountMetricsError as e:
        console.print(f"[bold red]Failed[/bold red] to refresh state: {e}")
        return None

    if _fx_maybe_force_flat(policy, state, client, execute=execute):
        return None

    # Validate account metrics
    if not _fx_require_account_metrics(pnl):
        return None

    # Check entry gates
    if not _fx_gate_fx_entry(policy, state, client):
        return None

    return policy, client, state, pnl


def _fx_build_paper_trade_plan(
    cfg: Dict[str, Any],
    policy: Any,
    client: Any,
    state: Any,
    pnl: Dict[str, Any],
    *,
    instrument: str,
    granularity: str,
    candles: int,
    equity: float,
    risk_per_trade_pct: Optional[float] = None,
) -> Optional[Tuple[FxPaperTradePlan, Any]]:
    """Build a paper trade plan.

    Args:
        cfg: Configuration dictionary
        policy: FX policy object
        client: OANDA client instance
        state: Trading state object
        pnl: PnL dictionary
        instrument: FX instrument symbol
        granularity: Timeframe granularity
        candles: Number of candles to analyze
        equity: Base equity for calculations
        risk_per_trade_pct: Override risk percentage

    Returns:
        Tuple of (plan, rules) or None if plan cannot be created
    """
    # Load market data
    try:
        df = _fx_load_fx_df(
            client,
            instrument=instrument,
            granularity=granularity,
            candles=candles
        )
    except MarketDataError as e:
        console.print(f"[bold red]Failed[/bold red] to load market data: {e}")
        return None

    # Check spread and slippage
    spread_info = _fx_spread_and_slippage(policy, df, instrument=instrument)
    if not spread_info.is_valid:
        return None

    # Get signal context
    try:
        signal_context = _fx_get_signal_context(df)
    except MarketDataError as e:
        console.print(f"[bold red]Failed[/bold red] to calculate signal: {e}")
        return None

    if signal_context.signal == Signal.HOLD.value:
        console.print(
            f"FX setup: HOLD  {instrument} {granularity}  "
            f"close={signal_context.last_close:.5f}  "
            f"ATR14={signal_context.atr_value:.5f}"
        )
        return None

    # Build risk rules
    try:
        rules = _fx_build_risk_rules(
            policy,
            pnl,
            equity=equity,
            risk_per_trade_pct=risk_per_trade_pct
        )
    except ConfigurationError as e:
        console.print(f"[bold red]Failed[/bold red] to build risk rules: {e}")
        return None

    # Calculate confidence
    try:
        confidence_result = _fx_compute_confidence_and_band(
            cfg,
            policy,
            instrument=instrument,
            signal=signal_context.signal,
            price=signal_context.last_close,
            atr_value=signal_context.atr_value,
            spread_pips=spread_info.spread_pips,
        )
    except ConfigurationError as e:
        console.print(f"[bold red]Failed[/bold red] to calculate confidence: {e}")
        return None

    if confidence_result.band == ConfidenceBand.LOW.value:
        console.print(
            f"[bold red]Blocked[/bold red]: low confidence "
            f"({confidence_result.confidence:.2f}) => no trades."
        )
        if confidence_result.reasons:
            console.print(
                f"[dim]Reasons: {', '.join(confidence_result.reasons)}[/dim]"
            )
        return None

    # Check daily stops
    if _fx_apply_daily_stops(cfg, policy, state, pnl, band=confidence_result.band):
        return None

    # Calculate order units and prices
    try:
        units, pricing = _fx_build_order_units_and_prices(
            policy,
            rules,
            instrument=instrument,
            signal=signal_context.signal,
            price=signal_context.last_close,
            atr_value=signal_context.atr_value,
            slippage_price=spread_info.slippage_price,
        )
    except (ValueError, ConfigurationError) as e:
        console.print(f"[bold red]Failed[/bold red] to calculate order parameters: {e}")
        return None

    # Create plan
    plan = FxPaperTradePlan(
        instrument=str(instrument),
        granularity=str(granularity),
        signal=str(signal_context.signal),
        last_close=float(signal_context.last_close),
        atr_value=float(signal_context.atr_value),
        units=int(units),
        stop_price=float(pricing.stop_price),
        tp_price=float(pricing.tp_price),
        spread_pips=float(spread_info.spread_pips),
        slippage_pips=float(spread_info.slippage_pips),
        slippage_price=float(spread_info.slippage_price),
        confidence=float(confidence_result.confidence),
        band=str(confidence_result.band),
        conf_reasons=list(confidence_result.reasons),
    )

    return plan, rules


# =============================================================================
# Public API
# =============================================================================

def fx_paper_trade(
    config_path: str,
    instrument: str,
    granularity: str = "M5",
    candles: int = 300,
    execute: bool = False,
    verbose: bool = False,
    equity: float = 10_000.0,
    risk_per_trade_pct: Optional[float] = None,
) -> None:
    """Paper trade forex on OANDA PRACTICE using a simple setup rule.

    This is intentionally conservative and defaults to DRY-RUN (no order placed).

    Args:
        config_path: Path to configuration file
        instrument: FX instrument symbol (e.g., "EUR_USD")
        granularity: Timeframe granularity (e.g., "M5", "H1")
        candles: Number of candles to analyze
        execute: If True, actually place orders; if False, dry-run
        verbose: If True, log verbose messages
        equity: Base equity for risk calculations
        risk_per_trade_pct: Risk per trade percentage (uses policy default if None)

    Raises:
        FXTradingError: If trading setup or execution fails
    """
    cfg = load_config(config_path)

    # Setup
    setup = _fx_setup_paper_trade(
        cfg,
        instrument=instrument,
        granularity=granularity,
        execute=execute
    )
    if setup is None:
        return
    policy, client, state, pnl = setup

    # Build plan
    planned = _fx_build_paper_trade_plan(
        cfg,
        policy,
        client,
        state,
        pnl,
        instrument=instrument,
        granularity=granularity,
        candles=candles,
        equity=equity,
        risk_per_trade_pct=risk_per_trade_pct,
    )
    if planned is None:
        return
    plan, rules = planned

    # Display plan
    console.print(
        f"FX setup: {plan.signal.upper()}  {plan.instrument} {plan.granularity}  "
        f"close={plan.last_close:.5f}  units={plan.units}  "
        f"SL={plan.stop_price:.5f}  TP={plan.tp_price:.5f}  "
        f"spread={plan.spread_pips:.2f}p  slip={plan.slippage_pips:.2f}p  "
        f"conf={plan.confidence:.2f}  band={plan.band}"
    )

    # Execute
    try:
        _fx_execute_paper_trade_plan(
            cfg,
            policy,
            client,
            state,
            plan,
            rules,
            execute=execute,
            verbose=verbose,
        )
    except OrderExecutionError as e:
        console.print(f"[bold red]Order execution failed[/bold red]: {e}")
        logger.error(f"Order execution failed: {e}")


# =============================================================================
# Legacy Integrated Prediction (Compatibility Shims)
# =============================================================================

def _build_integrated_engines(config: Dict[str, Any]) -> None:
    """Retired: Legacy integrated ML/MT/MR pipeline.

    The integrated pipeline has been retired as part of the TensorFlow-only
    Buddy migration.

    Args:
        config: Configuration dictionary

    Raises:
        RuntimeError: Always raised as this is deprecated
    """
    raise RuntimeError(
        "Retired: integrated engines are no longer supported. "
        "Use `train-buddy`/`buddy`."
    )


def integrated_predict_once(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility shim for legacy integrated prediction.

    The production path is Buddy (`train-buddy`/`buddy`). Some unit tests still
    exercise the historical integrated API. This is a no-network, lightweight
    stub implementation.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with stub prediction and reasoning
    """
    batch_size = int((config or {}).get("batch_size") or 1)
    prediction = np.zeros((batch_size, 1), dtype=np.float32)
    reasoning = {
        "insights": ["Integrated pipeline is deprecated; returned stub prediction."]
    }
    return {"prediction": prediction, "reasoning": reasoning}


def integrated_predict(config_path: str, csv_path: str | None = None) -> None:
    """Legacy integrated predict - retired.

    Args:
        config_path: Path to configuration file
        csv_path: Path to CSV file (unused)

    Raises:
        RuntimeError: Always raised as this is deprecated
    """
    raise RuntimeError(
        "Retired: integrated predict is no longer supported. "
        "Use `train-buddy`/`buddy`."
    )


# =============================================================================
# SLM Indicators
# =============================================================================

def compute_slm_indicators(market_data_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute a small subset of SLM technical indicators without network calls.

    Args:
        market_data_df: DataFrame with OHLCV data

    Returns:
        Dictionary with computed indicators

    Raises:
        ImportError: If SLM module cannot be imported
        MarketDataError: If indicator calculation fails
    """
    try:
        from SLM.stock_slm import (
            calculate_bollinger_bands,
            calculate_ema,
            calculate_macd,
            calculate_rsi,
            calculate_sma,
        )
    except ImportError as e:
        raise ImportError(f"Failed to import SLM indicators: {e}") from e

    # Normalize column names for SLM
    df_for_slm = market_data_df
    if hasattr(market_data_df, "columns") and "Close" not in market_data_df.columns:
        rename_map = {}
        for col in market_data_df.columns:
            key = str(col).strip().lower()
            if key in SLM_COLUMN_MAP:
                rename_map[col] = SLM_COLUMN_MAP[key]
        if rename_map:
            df_for_slm = market_data_df.rename(columns=rename_map)

    # Calculate indicators
    try:
        sma_20 = calculate_sma(df_for_slm, DEFAULT_SMA_PERIOD)
        ema_20 = calculate_ema(df_for_slm, DEFAULT_SMA_PERIOD)
        rsi_14 = calculate_rsi(df_for_slm, DEFAULT_RSI_PERIOD)
        macd_line, macd_signal, macd_hist = calculate_macd(df_for_slm)
        bb_upper, bb_lower = calculate_bollinger_bands(
            df_for_slm, DEFAULT_BB_PERIOD, DEFAULT_BB_STD_DEV
        )
    except Exception as e:
        raise MarketDataError(f"Failed to calculate SLM indicators: {e}") from e

    return {
        "sma_20": sma_20,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }


# =============================================================================
# Market Data Utilities
# =============================================================================

def _pick_default_market_csv(config: Dict[str, Any]) -> str:
    """Find default market CSV file from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the first CSV file found

    Raises:
        FileNotFoundError: If no CSV files are found
    """
    data_dir = (
        config.get("data", {}).get("data_dir")
        or config.get("paths", {}).get("data_dir")
        or config.get("DATA_DIR")
        or config.get("data_dir")
        or "market_data"
    )
    
    candidates = sorted(Path(data_dir).glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No .csv files found under: {data_dir}")
    
    return str(candidates[0])


def _normalize_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized OHLCV column names.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with normalized column names

    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If required columns are missing
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("market_data_df must be a pandas DataFrame")

    # Create copy and normalize column names
    out = df.copy()
    rename_map = {}
    
    for col in out.columns:
        key = str(col).strip().lower()
        if key in OHLCV_COLUMN_MAP:
            rename_map[col] = OHLCV_COLUMN_MAP[key]

    out = out.rename(columns=rename_map)
    
    # Validate required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Market DataFrame missing required columns: {missing}")
    
    return out


def build_integrated_feature_tensors(
    market_data_df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    allow_pad: bool = False,
) -> Dict[str, np.ndarray]:
    """Compatibility shim for legacy integrated feature tensors.

    Tests expect 3 tensors:
    - ml_features: (1, seq_len, 7)
    - mt_features: (1, seq_len, 11)
    - mr_features: (1, seq_len, 11)

    This implementation is intentionally simple and deterministic.

    Args:
        market_data_df: DataFrame with OHLCV data
        config: Configuration dictionary
        allow_pad: If True, pad sequences shorter than seq_len

    Returns:
        Dictionary with feature tensors

    Raises:
        MarketDataError: If tensor building fails
    """
    df = _normalize_market_dataframe(market_data_df)
    seq_len = int(
        (config or {}).get("data", {}).get("sequence_length", DEFAULT_SEQUENCE_LENGTH)
    )

    # Extract base numeric series
    try:
        o = df["open"].to_numpy(dtype=np.float32)
        h = df["high"].to_numpy(dtype=np.float32)
        low = df["low"].to_numpy(dtype=np.float32)
        c = df["close"].to_numpy(dtype=np.float32)
        v = df["volume"].to_numpy(dtype=np.float32)
    except (KeyError, ValueError) as e:
        raise MarketDataError(f"Failed to extract OHLCV data: {e}") from e

    n = len(df)
    if n <= 0:
        raise ValueError("market_data_df must have at least 1 row")

    # Pad or truncate to sequence length
    if n < seq_len:
        if not allow_pad:
            raise ValueError(f"Not enough rows ({n}) for sequence_length={seq_len}")
        
        pad = seq_len - n
        # Left-pad using the first row (conservative, stable)
        o = np.concatenate([np.repeat(o[:1], pad), o])
        h = np.concatenate([np.repeat(h[:1], pad), h])
        low = np.concatenate([np.repeat(low[:1], pad), low])
        c = np.concatenate([np.repeat(c[:1], pad), c])
        v = np.concatenate([np.repeat(v[:1], pad), v])
    else:
        o = o[-seq_len:]
        h = h[-seq_len:]
        low = low[-seq_len:]
        c = c[-seq_len:]
        v = v[-seq_len:]

    # Derived features
    prev_c = np.concatenate([c[:1], c[:-1]])
    ret = (c - prev_c) / np.maximum(prev_c, NUMPY_EPSILON)
    rng = (h - low) / np.maximum(c, NUMPY_EPSILON)

    ml = np.stack([o, h, low, c, v, ret, rng], axis=-1).astype(np.float32)

    # MT/MR historical shapes include extra engineered slots; fill deterministically
    zeros4 = np.zeros((seq_len, 4), dtype=np.float32)
    mt = np.concatenate([ml, zeros4], axis=-1)
    mr = np.concatenate([ml, zeros4], axis=-1)

    return {
        "ml_features": ml[None, :, :],
        "mt_features": mt[None, :, :],
        "mr_features": mr[None, :, :],
    }


def _fetch_live_market_data(
    ticker: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    """Fetch live market data using yfinance.

    Args:
        ticker: Ticker symbol
        period: Time period (e.g., "5d")
        interval: Data interval (e.g., "1h")

    Returns:
        DataFrame with market data

    Raises:
        ImportError: If yfinance is not available
        RuntimeError: If data fetch fails
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as e:
        raise ImportError("Live data fetch requires `yfinance`.") from e

    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data is None or data.empty:
        raise RuntimeError(f"No live data returned for {ticker} ({period}, {interval})")
    
    # Keep the index as a column (Date/Datetime) for display/debugging
    return data.reset_index()


def _df_last_value(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Any:
    """Get the last value from the first matching column.

    Args:
        df: DataFrame to search
        candidates: Tuple of column names to try in order

    Returns:
        The last value from the first matching column, or None
    """
    for col in candidates:
        if col in df.columns:
            return df[col].iloc[-1]
    return None


def _print_last_bar(df: pd.DataFrame) -> None:
    """Print the last bar information.

    Args:
        df: DataFrame with market data
    """
    last_close = _df_last_value(df, ("Close", "close"))
    last_ts = _df_last_value(df, ("Datetime", "Date", "date", "datetime"))
    
    if last_ts is None and last_close is None:
        return
    
    console.print(f"[dim]Last bar:[/dim] {last_ts}  close={last_close}")


def _print_reasoning_insights(reasoning: Dict[str, Any]) -> None:
    """Print reasoning insights.

    Args:
        reasoning: Dictionary with reasoning information
    """
    if not isinstance(reasoning, dict):
        return
    
    insights = reasoning.get("insights")
    if not insights:
        return
    
    console.print("[bold green]Reasoning insights:[/bold green]")
    for insight in insights:
        console.print(f"- {insight}")


def _integrated_predict_live_once(
    *,
    ticker: str,
    period: str,
    interval: str,
    config: Dict[str, Any],
    integrator: Any,
) -> None:
    """Run a single live prediction iteration.

    Args:
        ticker: Ticker symbol
        period: Time period
        interval: Data interval
        config: Configuration dictionary
        integrator: Integrator object (deprecated)
    """
    df = _fetch_live_market_data(ticker, period=period, interval=interval)
    features = build_integrated_feature_tensors(df, config, allow_pad=True)
    result = integrator.predict(features)

    console.print(f"[bold blue]Live predict[/bold blue] {ticker} ({period}, {interval})")
    _print_last_bar(df)
    console.print(f"[bold yellow]Prediction:[/bold yellow] {result.get('prediction')}")
    console.print(f"[bold yellow]Uncertainty:[/bold yellow] {result.get('uncertainty')}")

    weights = result.get("weights")
    if weights is not None:
        console.print(f"[cyan]Weights (ML/MT/MR):[/cyan] {weights}")

    _print_reasoning_insights(result.get("reasoning"))


def integrated_predict_live(
    config_path: str,
    ticker: str,
    period: str = "5d",
    interval: str = "1h",
    watch_seconds: Optional[int] = None,
) -> None:
    """Run the unified prediction using live-ish OHLCV from yfinance.

    Args:
        config_path: Path to configuration file
        ticker: Ticker symbol
        period: Time period
        interval: Data interval
        watch_seconds: If set, watch continuously with this interval

    Raises:
        RuntimeError: If integrated engines are not available
    """
    config = load_config(config_path)
    integrator = _build_integrated_engines(config)

    _integrated_predict_live_once(
        ticker=ticker,
        period=period,
        interval=interval,
        config=config,
        integrator=integrator,
    )

    if watch_seconds is None:
        return

    interval_seconds = max(1, int(watch_seconds))
    while True:  # pragma: no cover
        time.sleep(interval_seconds)
        _integrated_predict_live_once(
            ticker=ticker,
            period=period,
            interval=interval,
            config=config,
            integrator=integrator,
        )


# =============================================================================
# Dashboard Generation
# =============================================================================

def generate_dashboard(
    epoch: int,
    total_epochs: int,
    latency: int,
    train_loss: float,
    val_loss: float,
    lr_delta: float,
    api_logs: list,
    config: Dict[str, Any],
) -> Layout:
    """Generate the rich dashboard layout.

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        latency: API latency in milliseconds
        train_loss: Training loss value
        val_loss: Validation loss value
        lr_delta: Learning rate change percentage
        api_logs: List of API log messages
        config: Configuration dictionary

    Returns:
        Rich Layout object with dashboard
    """
    # Create main layout structure
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", size=8),
        Layout(name="footer", size=8),
    )

    # Build metrics table
    metrics_table = _build_metrics_table(
        epoch, total_epochs, latency, train_loss, val_loss, lr_delta, config
    )
    layout["header"].update(
        Panel(metrics_table, title="[bold]Training Metrics[/bold]", border_style="blue")
    )

    # Build body with progress and logs
    body_layout = _build_body_layout(epoch, total_epochs, latency, api_logs)
    layout["body"].update(body_layout)

    # Build footer with config and log info
    footer_layout = _build_footer_layout(config, api_logs)
    layout["footer"].update(footer_layout)

    return layout


def _build_metrics_table(
    epoch: int,
    total_epochs: int,
    latency: int,
    train_loss: float,
    val_loss: float,
    lr_delta: float,
    config: Dict[str, Any],
) -> Table:
    """Build the metrics table for the dashboard.

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        latency: API latency in milliseconds
        train_loss: Training loss value
        val_loss: Validation loss value
        lr_delta: Learning rate change percentage
        config: Configuration dictionary

    Returns:
        Rich Table object with metrics
    """
    metrics_table = Table(
        show_header=True,
        header_style=TABLE_HEADER_STYLE,
        expand=True
    )
    metrics_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    metrics_table.add_column("Value", style="green")

    metrics_table.add_row("Epoch", f"{epoch}/{total_epochs}")
    metrics_table.add_row("Training Loss", f"{train_loss:.6f}")
    metrics_table.add_row("Validation Loss", f"{val_loss:.6f}")
    metrics_table.add_row("API Latency", f"{latency}ms")

    lr_color = "green" if lr_delta >= 0 else "red"
    metrics_table.add_row(
        "LR Change",
        f"[{lr_color}]{lr_delta:+.2f}%[/{lr_color}]"
    )
    metrics_table.add_row(
        "Device",
        f"{config.get('hardware', {}).get('device', 'cpu')}"
    )

    status = (
        "[bold green]Active[/bold green]"
        if epoch < total_epochs
        else "[bold yellow]Complete[/bold yellow]"
    )
    metrics_table.add_row("Status", status)

    return metrics_table


def _build_body_layout(epoch: int, total_epochs: int, latency: int, api_logs: list) -> Layout:
    """Build the body layout with progress and logs.

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        latency: API latency in milliseconds
        api_logs: List of API log messages

    Returns:
        Rich Layout object with body content
    """
    body_layout = Layout()
    body_layout.split_row(
        Layout(name="progress", ratio=1),
        Layout(name="logs", ratio=2)
    )

    # Progress panel
    progress_pct = (epoch / total_epochs * 100) if total_epochs > 0 else 0
    bar = "█" * int(progress_pct / 2) + "░" * (50 - int(progress_pct / 2))
    eta = (total_epochs - epoch) * latency / 1000
    progress_text = f"{progress_pct:.1f}%\n{bar}\nETA: {eta:.1f}s"
    body_layout["progress"].update(
        Panel(progress_text, title="[bold]Progress[/bold]", border_style="green")
    )

    # Logs panel
    current_time = time.strftime("%H:%M:%S")
    if not api_logs:
        api_logs.append(f"[grey]{current_time}[/grey] Starting training...")

    formatted_logs = [
        log if log.startswith("[") else f"[grey]{current_time}[/grey] {log}"
        for log in api_logs[-5:]
    ]
    logs_content = "\n".join(formatted_logs)
    body_layout["logs"].update(
        Panel(logs_content, title="[bold]Recent Updates[/bold]", border_style="yellow")
    )

    return body_layout


def _build_footer_layout(config: Dict[str, Any], api_logs: list) -> Layout:
    """Build the footer layout with config and log info.

    Args:
        config: Configuration dictionary
        api_logs: List of API log messages

    Returns:
        Rich Layout object with footer content
    """
    # Config table
    config_table = Table(
        show_header=True,
        header_style=TABLE_HEADER_STYLE,
        expand=True
    )
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    model_config = config.get("model", {})
    config_table.add_row("Learning Rate", f"{model_config.get('learning_rate', 'N/A')}")
    config_table.add_row("Batch Size", f"{model_config.get('batch_size', 'N/A')}")
    config_table.add_row("Architecture", f"{model_config.get('architecture', 'N/A')}")
    config_table.add_row("Hidden Size", f"{model_config.get('hidden_size', 'N/A')}")
    config_table.add_row("Num Layers", f"{model_config.get('num_layers', 'N/A')}")
    config_table.add_row("Dropout", f"{model_config.get('dropout', 'N/A')}")

    optimizer_config = model_config.get("optimizer", {})
    config_table.add_row("Optimizer", f"{optimizer_config.get('type', 'Adam')}")

    # Latest log table
    logs_table = Table(
        show_header=True,
        header_style=TABLE_HEADER_STYLE,
        expand=True
    )
    logs_table.add_column("Latest Log", style="green")
    latest_log = api_logs[-1] if api_logs else "No logs available"
    logs_table.add_row(latest_log)

    # Combine into footer
    footer_layout = Layout()
    footer_layout.split_row(
        Panel(config_table, title="[bold]Configuration[/bold]", border_style="blue"),
        Panel(logs_table, title="[bold]Log Info[/bold]", border_style="blue"),
    )

    return footer_layout
