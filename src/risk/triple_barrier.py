"""
Triple Barrier Labeling for ML Trading Models.

The triple barrier method provides better labels for trading ML by considering
actual trade outcomes rather than just price direction.

Three barriers:
1. Upper barrier (Take Profit): Price moves enough in favorable direction
2. Lower barrier (Stop Loss): Price moves enough in unfavorable direction  
3. Time barrier (Timeout): Neither TP nor SL hit within time horizon

Labels:
- 1: TP hit first (profitable trade)
- 0: Timeout (no clear signal - skip these in training)
- -1: SL hit first (losing trade)

For binary classification (direction), we can map:
- 1 (TP hit on long) -> 1 (bullish)
- 1 (TP hit on short) -> 0 (bearish)
- 0 (timeout) -> excluded or 0.5
- -1 (SL hit) -> opposite of intended direction

Reference: Advances in Financial Machine Learning - Marcos Lopez de Prado
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BarrierHit(Enum):
    """Which barrier was hit first."""
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"
    TIMEOUT = "timeout"
    
    @property
    def is_profitable(self) -> bool:
        return self == BarrierHit.TAKE_PROFIT


@dataclass(frozen=True)
class TripleBarrierResult:
    """Result of triple barrier labeling for a single sample."""
    
    signal_index: int
    direction: Literal["long", "short"]  # Intended direction
    barrier_hit: BarrierHit
    bars_held: int
    entry_price: float
    exit_price: float
    pnl_pips: float
    
    @property
    def is_win(self) -> bool:
        return self.barrier_hit == BarrierHit.TAKE_PROFIT
    
    @property
    def is_timeout(self) -> bool:
        return self.barrier_hit == BarrierHit.TIMEOUT
    
    def to_direction_label(self) -> float:
        """Convert to direction label (1=long profitable, 0=short profitable, 0.5=unclear).
        
        For the direction head, we want to learn which direction would have been profitable.
        """
        if self.barrier_hit == BarrierHit.TIMEOUT:
            return 0.5  # No clear signal
        
        if self.barrier_hit == BarrierHit.TAKE_PROFIT:
            # TP hit = intended direction was correct
            return 1.0 if self.direction == "long" else 0.0
        else:
            # SL hit = intended direction was wrong
            return 0.0 if self.direction == "long" else 1.0
    
    def to_confidence_label(self) -> float:
        """Convert to confidence label (1=TP hit, 0=SL/timeout)."""
        return 1.0 if self.barrier_hit == BarrierHit.TAKE_PROFIT else 0.0
    
    def to_ternary_label(self) -> int:
        """Convert to ternary label (1=TP, 0=timeout, -1=SL)."""
        if self.barrier_hit == BarrierHit.TAKE_PROFIT:
            return 1
        elif self.barrier_hit == BarrierHit.TIMEOUT:
            return 0
        else:
            return -1


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    *,
    instrument: str,
    stop_loss_pips: float,
    take_profit_pips: float,
    max_horizon_candles: int,
    spread_pips: float = 0.0,
    primary_direction: Optional[np.ndarray] = None,
    use_both_directions: bool = True,
    seq_len: int = 60,
    label_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[TripleBarrierResult]]:
    """
    Compute triple barrier labels for all valid indices.
    
    Args:
        df: OHLC DataFrame
        instrument: Trading pair (e.g., "USD_JPY")
        stop_loss_pips: SL distance in pips
        take_profit_pips: TP distance in pips
        max_horizon_candles: Max bars to wait for barrier hit
        spread_pips: Spread cost in pips
        primary_direction: If provided, use as intended direction (else try both)
        use_both_directions: If True, compute best direction at each point
        seq_len: Sequence length for features (labels start after this)
        label_stride: Only compute labels every N bars (for speed)
    
    Returns:
        Tuple of:
        - direction_labels: (n,) array of direction labels [0, 1]
        - confidence_labels: (n,) array of confidence labels [0, 1]
        - label_weights: (n,) array of weights (0 for excluded samples)
        - results: List of TripleBarrierResult for detailed analysis
    """
    from src.utils.fx_paper import pip_size, simulate_tp_sl_outcome
    
    n = len(df)
    direction_labels = np.full((n,), 0.5, dtype=np.float32)
    confidence_labels = np.zeros((n,), dtype=np.float32)
    label_weights = np.zeros((n,), dtype=np.float32)
    results: List[TripleBarrierResult] = []
    
    pip = float(pip_size(instrument))
    start_idx = max(0, seq_len - 1)
    end_idx = n - max_horizon_candles - 2
    
    if end_idx <= start_idx:
        logger.warning(f"Insufficient data for triple barrier labeling: start={start_idx}, end={end_idx}")
        return direction_labels, confidence_labels, label_weights, results
    
    for i in range(start_idx, end_idx, label_stride):
        try:
            # Determine direction(s) to try
            if primary_direction is not None:
                directions_to_try = ["long" if primary_direction[i] >= 0.5 else "short"]
            elif use_both_directions:
                directions_to_try = ["long", "short"]
            else:
                directions_to_try = ["long"]  # Default to long
            
            best_result: Optional[TripleBarrierResult] = None
            best_pnl = float("-inf")
            
            for direction in directions_to_try:
                try:
                    res = simulate_tp_sl_outcome(
                        df,
                        instrument=instrument,
                        signal_index=i,
                        direction=direction,
                        stop_loss_pips=stop_loss_pips,
                        take_profit_pips=take_profit_pips,
                        spread_pips=spread_pips,
                        max_horizon_candles=max_horizon_candles,
                        tie_break="sl",  # Conservative
                    )
                    
                    # Calculate PnL in pips
                    if direction == "long":
                        pnl = (res.exit_price - res.entry_price) / pip
                    else:
                        pnl = (res.entry_price - res.exit_price) / pip
                    
                    # Convert to TripleBarrierResult
                    barrier = BarrierHit.TAKE_PROFIT if res.hit_type == "tp" else (
                        BarrierHit.STOP_LOSS if res.hit_type == "sl" else BarrierHit.TIMEOUT
                    )
                    
                    result = TripleBarrierResult(
                        signal_index=i,
                        direction=direction,
                        barrier_hit=barrier,
                        bars_held=res.bars_held,
                        entry_price=res.entry_price,
                        exit_price=res.exit_price,
                        pnl_pips=pnl,
                    )
                    
                    # Keep best direction (highest PnL or TP hit)
                    if result.is_win or (not best_result or pnl > best_pnl):
                        best_result = result
                        best_pnl = pnl
                        
                        # If we found a winning direction, use it
                        if result.is_win:
                            break
                            
                except Exception as e:
                    logger.debug(f"Triple barrier failed for idx={i}, dir={direction}: {e}")
                    continue
            
            if best_result is not None:
                results.append(best_result)
                direction_labels[i] = best_result.to_direction_label()
                confidence_labels[i] = best_result.to_confidence_label()
                
                # Weight: full for TP/SL, reduced for timeout
                if best_result.is_timeout:
                    label_weights[i] = 0.3  # Downweight unclear samples
                else:
                    label_weights[i] = 1.0
                    
        except Exception as e:
            logger.debug(f"Triple barrier labeling failed at idx={i}: {e}")
            continue
    
    n_labeled = int(np.sum(label_weights > 0))
    n_wins = sum(1 for r in results if r.is_win)
    n_timeouts = sum(1 for r in results if r.is_timeout)
    
    logger.info(
        f"Triple barrier labeling: {n_labeled} samples, "
        f"{n_wins} wins ({100*n_wins/max(n_labeled,1):.1f}%), "
        f"{n_timeouts} timeouts ({100*n_timeouts/max(n_labeled,1):.1f}%)"
    )
    
    return direction_labels, confidence_labels, label_weights, results


def create_triple_barrier_dataset(
    df: pd.DataFrame,
    features: np.ndarray,
    *,
    instrument: str,
    stop_loss_pips: float,
    take_profit_pips: float,
    max_horizon_candles: int,
    spread_pips: float = 0.0,
    seq_len: int = 60,
    exclude_timeouts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a filtered dataset with triple barrier labels.
    
    Optionally excludes timeout samples where there's no clear signal.
    
    Args:
        df: OHLC DataFrame
        features: Feature array (n, features)
        instrument: Trading pair
        stop_loss_pips, take_profit_pips: TP/SL distances
        max_horizon_candles: Time barrier
        spread_pips: Spread cost
        seq_len: Sequence length
        exclude_timeouts: If True, remove timeout samples
    
    Returns:
        Tuple of (X_filtered, y_direction, y_confidence)
    """
    direction_labels, confidence_labels, weights, results = compute_triple_barrier_labels(
        df,
        instrument=instrument,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips,
        max_horizon_candles=max_horizon_candles,
        spread_pips=spread_pips,
        use_both_directions=True,
        seq_len=seq_len,
        label_stride=1,
    )
    
    if exclude_timeouts:
        # Keep only samples with clear outcomes (TP or SL hit)
        mask = weights >= 1.0
    else:
        mask = weights > 0
    
    indices = np.where(mask)[0]
    
    X_filtered = features[indices]
    y_direction = direction_labels[indices]
    y_confidence = confidence_labels[indices]
    
    logger.info(f"Filtered dataset: {len(indices)} samples from {len(features)} original")
    
    return X_filtered, y_direction, y_confidence


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Triple Barrier Labeling Test")
    print("-" * 40)
    
    # Create dummy OHLC data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="H")
    
    # Random walk price
    returns = np.random.randn(n) * 0.001
    close = 150.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.001)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.001)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    }, index=dates)
    
    # Compute labels
    dir_labels, conf_labels, weights, results = compute_triple_barrier_labels(
        df,
        instrument="USD_JPY",
        stop_loss_pips=15.0,
        take_profit_pips=30.0,
        max_horizon_candles=48,
        spread_pips=1.5,
        seq_len=60,
        label_stride=5,
    )
    
    print(f"\nResults: {len(results)} samples")
    if results:
        wins = sum(1 for r in results if r.is_win)
        timeouts = sum(1 for r in results if r.is_timeout)
        losses = len(results) - wins - timeouts
        
        print(f"  Wins: {wins} ({100*wins/len(results):.1f}%)")
        print(f"  Losses: {losses} ({100*losses/len(results):.1f}%)")
        print(f"  Timeouts: {timeouts} ({100*timeouts/len(results):.1f}%)")
        
        print("\nSample results:")
        for r in results[:5]:
            print(f"  idx={r.signal_index}: {r.direction} -> {r.barrier_hit.value}, PnL={r.pnl_pips:+.1f} pips")
    
    print("\nTriple Barrier test complete!")

