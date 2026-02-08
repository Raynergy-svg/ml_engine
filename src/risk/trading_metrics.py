"""
Trading Metrics Module - RAPI and Transaction Cost Analysis.

Implements Risk-Adjusted Performance Index (RAPI) and per-pair transaction cost
simulation for realistic backtesting of FX trading strategies.

Key Components:
- Per-pair transaction costs (majors, crosses, emerging)
- RAPI calculation (directional profitability adjusted for costs)
- Cost-adjusted Sharpe ratio
- Detailed trade-by-trade cost breakdown

Usage:
    from src.risk.trading_metrics import compute_rapi, TradingCostsConfig

    config = TradingCostsConfig.for_instrument("EUR_USD")
    result = compute_rapi(predictions, returns, config)

    print(f"RAPI: {result.rapi_score:.4f}")
    print(f"Net return: {result.net_return:.4%}")

Reference: MADL optimization and RAPI metrics from quantitative finance research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PairTier(Enum):
    """Currency pair tier classification based on liquidity."""
    MAJOR = "major"      # EUR/USD, GBP/USD, USD/JPY, USD/CHF
    CROSS = "cross"      # EUR/GBP, AUD/USD, NZD/USD, USD/CAD
    EMERGING = "emerging"  # USD/MXN, USD/TRY, USD/ZAR


# Default spread costs by tier (round-trip basis points)
DEFAULT_COSTS_BY_TIER = {
    PairTier.MAJOR: 0.005,     # 0.5% (5 bps) round-trip
    PairTier.CROSS: 0.007,     # 0.7% (7 bps) round-trip
    PairTier.EMERGING: 0.009,  # 0.9% (9 bps) round-trip
}

# Per-pair spread costs (pips) - aligned with fx_guardrails.py FxCosts
DEFAULT_SPREAD_PIPS = {
    # Majors
    "EUR_USD": 1.0,
    "GBP_USD": 1.6,
    "USD_JPY": 1.2,
    "USD_CHF": 1.5,
    # Crosses
    "EUR_GBP": 2.0,
    "AUD_USD": 1.4,
    "NZD_USD": 2.0,
    "USD_CAD": 1.8,
    "EUR_JPY": 2.0,
    # Emerging
    "USD_MXN": 50.0,
    "USD_TRY": 80.0,
    "USD_ZAR": 100.0,
}

# Pip values for converting spread to percentage
PIP_VALUES = {
    # Most pairs: 0.0001 = 1 pip
    "default": 0.0001,
    # JPY pairs: 0.01 = 1 pip
    "USD_JPY": 0.01,
    "EUR_JPY": 0.01,
    "GBP_JPY": 0.01,
    "AUD_JPY": 0.01,
    "NZD_JPY": 0.01,
    "CAD_JPY": 0.01,
    "CHF_JPY": 0.01,
}


def _classify_pair_tier(instrument: str) -> PairTier:
    """Classify currency pair into tier based on liquidity."""
    majors = {"EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF"}
    crosses = {"EUR_GBP", "AUD_USD", "NZD_USD", "USD_CAD", "EUR_JPY", "GBP_JPY"}

    if instrument in majors:
        return PairTier.MAJOR
    elif instrument in crosses:
        return PairTier.CROSS
    else:
        return PairTier.EMERGING


def _get_pip_value(instrument: str) -> float:
    """Get pip value for instrument."""
    return PIP_VALUES.get(instrument, PIP_VALUES["default"])


@dataclass(frozen=True)
class TradingCostsConfig:
    """Configuration for trading cost calculations.

    Attributes:
        instrument: Currency pair (e.g., "EUR_USD")
        tier: Pair tier classification
        spread_pips: Spread in pips
        slippage_pips: Expected slippage in pips
        commission_per_side: Commission per trade side (as decimal)
        round_trip_cost_pct: Total round-trip cost as percentage
    """
    instrument: str
    tier: PairTier
    spread_pips: float
    slippage_pips: float = 0.3
    commission_per_side: float = 0.0  # Most FX is commission-free
    round_trip_cost_pct: float = 0.005  # Default 0.5%

    @classmethod
    def for_instrument(
        cls,
        instrument: str,
        spread_pips: Optional[float] = None,
        slippage_pips: float = 0.3,
    ) -> "TradingCostsConfig":
        """Create cost config for specific instrument.

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            spread_pips: Override spread (uses default if None)
            slippage_pips: Slippage estimate

        Returns:
            TradingCostsConfig for the instrument
        """
        tier = _classify_pair_tier(instrument)

        if spread_pips is None:
            spread_pips = DEFAULT_SPREAD_PIPS.get(instrument, 2.0)

        # Convert spread to percentage
        pip_value = _get_pip_value(instrument)
        spread_pct = spread_pips * pip_value
        slippage_pct = slippage_pips * pip_value

        # Round-trip cost = 2 * (spread + slippage)
        round_trip_cost = 2 * (spread_pct + slippage_pct)

        return cls(
            instrument=instrument,
            tier=tier,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            round_trip_cost_pct=round_trip_cost,
        )

    @classmethod
    def default_by_tier(cls, tier: PairTier) -> "TradingCostsConfig":
        """Create default cost config for tier."""
        cost_pct = DEFAULT_COSTS_BY_TIER[tier]
        return cls(
            instrument=f"DEFAULT_{tier.value.upper()}",
            tier=tier,
            spread_pips=cost_pct * 10000 / 2,  # Approx
            round_trip_cost_pct=cost_pct,
        )


@dataclass
class RAPIResult:
    """Result of RAPI calculation.

    Attributes:
        gross_return: Total return before costs
        total_costs: Total transaction costs incurred
        net_return: Return after costs
        n_trades: Number of trades executed
        n_winning: Number of winning trades
        n_losing: Number of losing trades
        win_rate: Win rate (gross)
        win_rate_net: Win rate after costs
        rapi_score: Risk-Adjusted Performance Index
        sharpe_gross: Sharpe ratio before costs
        sharpe_net: Sharpe ratio after costs
        cost_drag: Return reduction due to costs
    """
    gross_return: float
    total_costs: float
    net_return: float
    n_trades: int
    n_winning: int
    n_losing: int
    win_rate: float
    win_rate_net: float
    rapi_score: float
    sharpe_gross: float
    sharpe_net: float
    cost_drag: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'gross_return': self.gross_return,
            'total_costs': self.total_costs,
            'net_return': self.net_return,
            'n_trades': self.n_trades,
            'n_winning': self.n_winning,
            'n_losing': self.n_losing,
            'win_rate': self.win_rate,
            'win_rate_net': self.win_rate_net,
            'rapi_score': self.rapi_score,
            'sharpe_gross': self.sharpe_gross,
            'sharpe_net': self.sharpe_net,
            'cost_drag': self.cost_drag,
        }


def compute_rapi(
    predictions: np.ndarray,
    returns: np.ndarray,
    config: Optional[TradingCostsConfig] = None,
    annualization_factor: float = np.sqrt(252),
) -> RAPIResult:
    """
    Compute Risk-Adjusted Performance Index (RAPI).

    RAPI measures directional prediction quality adjusted for transaction costs:

    RAPI = (Mean Net Return) / (Std Net Return) * √252

    Higher RAPI indicates better risk-adjusted performance after costs.

    Args:
        predictions: Direction predictions (0=short, 1=long) or probabilities
        returns: Actual price returns
        config: Trading costs config (uses major pair defaults if None)
        annualization_factor: Annualization (√252 for daily, √52 for weekly)

    Returns:
        RAPIResult with detailed metrics
    """
    if config is None:
        config = TradingCostsConfig.default_by_tier(PairTier.MAJOR)

    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    # Align arrays
    n = min(len(predictions), len(returns))
    predictions = predictions[:n]
    returns = returns[:n]

    # Convert probabilities to positions
    if predictions.max() <= 1 and predictions.min() >= 0:
        # Probabilities: convert to position (-1, 0, +1)
        positions = np.where(predictions > 0.5, 1, -1)
    else:
        # Already positions
        positions = np.sign(predictions)

    # Detect position changes (trades)
    position_changes = np.abs(np.diff(positions, prepend=0))
    trade_indices = np.where(position_changes > 0)[0]
    n_trades = len(trade_indices)

    # Strategy returns: position * market return
    strategy_returns_gross = positions * returns

    # Apply transaction costs on position changes
    # Cost is applied when entering or exiting a position
    transaction_costs = position_changes * config.round_trip_cost_pct / 2

    # Net returns
    strategy_returns_net = strategy_returns_gross - transaction_costs

    # Aggregate metrics
    gross_return = float(np.sum(strategy_returns_gross))
    total_costs = float(np.sum(transaction_costs))
    net_return = float(np.sum(strategy_returns_net))

    # Win/loss counts (gross)
    winning_mask = strategy_returns_gross > 0
    losing_mask = strategy_returns_gross < 0
    n_winning = int(np.sum(winning_mask))
    n_losing = int(np.sum(losing_mask))

    # Win rate
    n_periods = len(strategy_returns_gross)
    win_rate = n_winning / max(n_periods, 1)

    # Win rate after costs (net positive periods)
    win_rate_net = float(np.sum(strategy_returns_net > 0)) / max(n_periods, 1)

    # Sharpe ratios
    if np.std(strategy_returns_gross) > 0:
        sharpe_gross = annualization_factor * np.mean(strategy_returns_gross) / np.std(strategy_returns_gross)
    else:
        sharpe_gross = 0.0

    if np.std(strategy_returns_net) > 0:
        sharpe_net = annualization_factor * np.mean(strategy_returns_net) / np.std(strategy_returns_net)
    else:
        sharpe_net = 0.0

    # RAPI = annualized net Sharpe
    rapi_score = float(sharpe_net)

    # Cost drag
    cost_drag = gross_return - net_return if gross_return != 0 else 0.0

    logger.info(
        f"RAPI ({config.instrument}): {rapi_score:.4f} | "
        f"Gross: {gross_return:.4%}, Net: {net_return:.4%}, Costs: {total_costs:.4%}"
    )

    return RAPIResult(
        gross_return=gross_return,
        total_costs=total_costs,
        net_return=net_return,
        n_trades=n_trades,
        n_winning=n_winning,
        n_losing=n_losing,
        win_rate=win_rate,
        win_rate_net=win_rate_net,
        rapi_score=rapi_score,
        sharpe_gross=float(sharpe_gross),
        sharpe_net=float(sharpe_net),
        cost_drag=cost_drag,
    )


def simulate_transaction_costs(
    predictions: np.ndarray,
    returns: np.ndarray,
    config: Optional[TradingCostsConfig] = None,
) -> Dict[str, any]:
    """
    Detailed transaction cost simulation with trade-by-trade breakdown.

    Args:
        predictions: Direction predictions
        returns: Actual returns
        config: Trading costs config

    Returns:
        Dictionary with detailed cost analysis
    """
    if config is None:
        config = TradingCostsConfig.default_by_tier(PairTier.MAJOR)

    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    n = min(len(predictions), len(returns))
    predictions = predictions[:n]
    returns = returns[:n]

    # Convert to positions
    positions = np.where(predictions > 0.5, 1, -1)

    # Identify trade entries/exits
    position_changes = np.diff(positions, prepend=0)

    trades = []
    current_position = 0
    entry_idx = None

    for i in range(n):
        if position_changes[i] != 0:
            # Position change
            if current_position != 0 and entry_idx is not None:
                # Close previous position
                exit_return = np.sum(returns[entry_idx:i] * current_position)
                cost = config.round_trip_cost_pct
                net = exit_return - cost
                trades.append({
                    'entry': entry_idx,
                    'exit': i,
                    'direction': 'long' if current_position > 0 else 'short',
                    'gross_return': exit_return,
                    'cost': cost,
                    'net_return': net,
                    'profitable': net > 0,
                })

            current_position = positions[i]
            entry_idx = i

    # Close final position
    if current_position != 0 and entry_idx is not None:
        exit_return = np.sum(returns[entry_idx:] * current_position)
        cost = config.round_trip_cost_pct
        trades.append({
            'entry': entry_idx,
            'exit': n - 1,
            'direction': 'long' if current_position > 0 else 'short',
            'gross_return': exit_return,
            'cost': cost,
            'net_return': exit_return - cost,
            'profitable': (exit_return - cost) > 0,
        })

    # Summary statistics
    if trades:
        gross_returns = [t['gross_return'] for t in trades]
        net_returns = [t['net_return'] for t in trades]
        costs = [t['cost'] for t in trades]

        return {
            'n_trades': len(trades),
            'total_gross_return': sum(gross_returns),
            'total_net_return': sum(net_returns),
            'total_costs': sum(costs),
            'avg_trade_return': np.mean(net_returns),
            'win_rate': sum(1 for t in trades if t['profitable']) / len(trades),
            'avg_winner': np.mean([t['net_return'] for t in trades if t['profitable']]) if any(t['profitable'] for t in trades) else 0,
            'avg_loser': np.mean([t['net_return'] for t in trades if not t['profitable']]) if any(not t['profitable'] for t in trades) else 0,
            'profit_factor': abs(sum(t['net_return'] for t in trades if t['profitable'])) / max(abs(sum(t['net_return'] for t in trades if not t['profitable'])), 1e-10),
            'trades': trades,
            'config': {
                'instrument': config.instrument,
                'tier': config.tier.value,
                'round_trip_cost_pct': config.round_trip_cost_pct,
            },
        }
    else:
        return {
            'n_trades': 0,
            'total_gross_return': 0.0,
            'total_net_return': 0.0,
            'total_costs': 0.0,
            'trades': [],
            'config': {
                'instrument': config.instrument,
                'tier': config.tier.value,
                'round_trip_cost_pct': config.round_trip_cost_pct,
            },
        }


def compute_cost_adjusted_accuracy(
    predictions: np.ndarray,
    returns: np.ndarray,
    config: Optional[TradingCostsConfig] = None,
) -> Tuple[float, float]:
    """
    Compute accuracy adjusted for transaction costs.

    A "correct" prediction that doesn't cover costs isn't truly profitable.

    Args:
        predictions: Direction predictions (0/1 or probabilities)
        returns: Actual returns
        config: Trading costs config

    Returns:
        (gross_accuracy, cost_adjusted_accuracy)
    """
    if config is None:
        config = TradingCostsConfig.default_by_tier(PairTier.MAJOR)

    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    n = min(len(predictions), len(returns))
    predictions = predictions[:n]
    returns = returns[:n]

    # Direction predictions
    pred_direction = predictions > 0.5  # True = long
    actual_direction = returns > 0  # True = price went up

    # Gross accuracy (direction match)
    gross_accuracy = np.mean(pred_direction == actual_direction)

    # Cost-adjusted: correct AND return exceeds cost
    cost_threshold = config.round_trip_cost_pct / 2  # Half since it's entry cost

    correct_and_profitable = (
        (pred_direction == actual_direction) &
        (np.abs(returns) > cost_threshold)
    )

    cost_adjusted_accuracy = np.mean(correct_and_profitable)

    return float(gross_accuracy), float(cost_adjusted_accuracy)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Trading Metrics Module")
    print("=" * 60)

    # Test pair classification
    print("\nPair Tier Classification:")
    for pair in ["EUR_USD", "AUD_USD", "USD_MXN"]:
        tier = _classify_pair_tier(pair)
        config = TradingCostsConfig.for_instrument(pair)
        print(f"  {pair}: {tier.value}, cost={config.round_trip_cost_pct:.4%}")

    # Generate synthetic data
    np.random.seed(42)
    n = 500

    # Generate returns
    returns = np.random.randn(n) * 0.001  # ~0.1% daily vol

    # Generate predictions (60% accuracy)
    true_direction = returns > 0
    predictions = true_direction.copy().astype(float)
    noise_idx = np.random.choice(n, size=int(n * 0.4), replace=False)
    predictions[noise_idx] = 1 - predictions[noise_idx]

    print(f"\nSynthetic data: {n} periods, 60% gross accuracy")

    # Test RAPI for different pairs
    print("\nRAPI by Pair Tier:")
    for pair in ["EUR_USD", "AUD_USD", "USD_MXN"]:
        config = TradingCostsConfig.for_instrument(pair)
        result = compute_rapi(predictions, returns, config)
        print(f"  {pair} ({config.tier.value}): RAPI={result.rapi_score:.4f}, "
              f"Net={result.net_return:.4%}, Drag={result.cost_drag:.4%}")

    # Test accuracy adjustment
    print("\nCost-Adjusted Accuracy:")
    config = TradingCostsConfig.for_instrument("EUR_USD")
    gross_acc, adj_acc = compute_cost_adjusted_accuracy(predictions, returns, config)
    print(f"  Gross: {gross_acc:.2%}, Adjusted: {adj_acc:.2%}")

    # Test detailed simulation
    print("\nDetailed Trade Simulation (EUR_USD):")
    sim = simulate_transaction_costs(predictions, returns, config)
    print(f"  Trades: {sim['n_trades']}")
    print(f"  Total Net Return: {sim['total_net_return']:.4%}")
    print(f"  Win Rate: {sim['win_rate']:.2%}")
    print(f"  Profit Factor: {sim['profit_factor']:.2f}")

    print("\n✓ Trading metrics module ready")
