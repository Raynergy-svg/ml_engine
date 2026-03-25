"""
Example: Market Regime Detector Usage

Demonstrates how to integrate the MarketRegimeDetector into the scanner pipeline.
"""

import logging
from datetime import datetime

import numpy as np

from src.scanner.regime_detector import (
    MarketRegimeDetector,
    RegimeDetectorConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic usage: detect regime from price series."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Regime Detection")
    logger.info("=" * 80)

    # Create detector with default config
    detector = MarketRegimeDetector()

    # Generate synthetic price data (trending market)
    np.random.seed(42)
    prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))

    # Detect regime
    result = detector.update(prices)

    logger.info(f"\nPrice Series: {len(prices)} bars")
    logger.info(f"Regime: {result.regime_name} (index={result.regime_index})")
    logger.info(f"Confidence: {result.confidence:.2%}")
    logger.info(f"\nComponent Signals:")
    logger.info(f"  - Changepoint Probability: {result.changepoint_probability:.2%}")
    logger.info(f"  - Hurst Exponent: {result.hurst_exponent:.3f}")
    logger.info(f"  - Trend: {result.trend_classification}")
    logger.info(f"  - ADX: {result.adx_value:.1f} ({result.adx_strength})")
    logger.info(f"  - Volatility Percentile: {result.volatility_percentile:.1f}%")
    logger.info(f"\nDerived Signals:")
    logger.info(f"  - Strategy: {result.strategy_hint}")
    logger.info(f"  - Risk Multiplier: {result.risk_multiplier:.2f}")


def example_with_ohlc():
    """Usage with OHLC data for complete analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Regime Detection with OHLC Data")
    logger.info("=" * 80)

    # Create detector with custom config
    config = RegimeDetectorConfig(
        hazard_rate=0.02,  # Shorter expected regime duration
        bocpd_threshold=0.4,  # Higher changepoint sensitivity
        hurst_window=80,  # Shorter window
    )
    detector = MarketRegimeDetector(config)

    # Generate synthetic OHLC data
    np.random.seed(42)
    closes = np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))
    highs = closes + np.abs(np.random.normal(0.005, 0.002, 200))
    lows = closes - np.abs(np.random.normal(0.005, 0.002, 200))

    # Detect regime with OHLC
    result = detector.update(closes, highs, lows)

    logger.info(f"\nPrice Series: {len(closes)} bars OHLC")
    logger.info(f"Regime: {result.regime_name} (index={result.regime_index})")
    logger.info(f"Confidence: {result.confidence:.2%}")
    logger.info(f"\nTechnical Indicators:")
    logger.info(f"  - ADX: {result.adx_value:.1f} → {result.adx_strength} Trend")
    logger.info(f"  - Hurst: {result.hurst_exponent:.3f} → {result.trend_classification}")
    logger.info(f"  - Volatility: {result.volatility_percentile:.1f}th percentile")
    logger.info(f"\nDecision:")
    logger.info(f"  - Recommended Strategy: {result.strategy_hint}")
    logger.info(f"  - Position Sizing Factor: {result.risk_multiplier:.2f}x")


def example_regime_transitions():
    """Detect regime changes in a volatile market."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Detecting Regime Transitions")
    logger.info("=" * 80)

    detector = MarketRegimeDetector()

    # Simulate market with volatility regime change
    np.random.seed(42)
    quiet_period = np.cumprod(1 + np.random.normal(0.0001, 0.002, 100))
    volatile_period = np.cumprod(1 + np.random.normal(0.001, 0.05, 100))
    prices = np.concatenate([quiet_period, volatile_period])

    # Detect regime for entire series
    result_full = detector.update(prices)
    logger.info(f"\nFull Series ({len(prices)} bars):")
    logger.info(f"  Regime: {result_full.regime_name}")
    logger.info(f"  Changepoint Alert: {result_full.changepoint_alert}")
    logger.info(f"  Changepoint Prob: {result_full.changepoint_probability:.2%}")

    # Detect regime for each half
    result_first = detector.update(prices[:100])
    result_second = detector.update(prices[100:])

    logger.info(f"\nFirst Half (bars 1-100, quiet):")
    logger.info(f"  Regime: {result_first.regime_name}")
    logger.info(f"  Volatility: {result_first.volatility_percentile:.1f}th percentile")

    logger.info(f"\nSecond Half (bars 101-200, volatile):")
    logger.info(f"  Regime: {result_second.regime_name}")
    logger.info(f"  Volatility: {result_second.volatility_percentile:.1f}th percentile")

    if result_full.changepoint_alert:
        logger.info(
            f"\n✓ Regime change detected! Use {result_second.strategy_hint} strategy."
        )


def example_strategy_guidance():
    """Show how regime detection guides trading strategy."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Strategy Guidance from Regime Detection")
    logger.info("=" * 80)

    detector = MarketRegimeDetector()

    # Test different market scenarios
    scenarios = {
        "Trending": np.cumprod(
            1 + np.random.normal(0.001, 0.005, 200)
        ),  # Positive drift
        "Ranging": 100 + 5 * np.sin(np.linspace(0, 10 * np.pi, 200)),  # Oscillating
        "Volatile": np.cumprod(
            1 + np.random.normal(0.0001, 0.05, 200)
        ),  # High vol
    }

    for scenario_name, prices in scenarios.items():
        np.random.seed(42)  # Reset for consistency
        result = detector.update(prices)

        logger.info(f"\n{scenario_name} Market:")
        logger.info(f"  Regime: {result.regime_name}")
        logger.info(f"  Hurst: {result.hurst_exponent:.3f} ({result.trend_classification})")
        logger.info(f"  ADX: {result.adx_value:.1f} ({result.adx_strength})")
        logger.info(f"  Strategy: {result.strategy_hint}")
        logger.info(f"  Risk Multiplier: {result.risk_multiplier:.2f}x")


def example_integration_with_scanner():
    """Show how to integrate with the scanner pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: Integration with Scanner Pipeline")
    logger.info("=" * 80)

    logger.info(
        """
    # In your scanner code, after fetching OHLC bars:

    from src.scanner.regime_detector import MarketRegimeDetector

    # Initialize detector (once, in __init__)
    self.regime_detector = MarketRegimeDetector()

    # In scan loop:
    bars = fetch_bars_for_pair(pair, count=200)

    # Run regime detection
    regime_result = self.regime_detector.update(
        prices=bars['close'].values,
        highs=bars['high'].values,
        lows=bars['low'].values
    )

    # Use result to adjust execution:
    execution_config.regime_scale = regime_result.risk_multiplier

    # Apply strategy hint to agent weights:
    if regime_result.strategy_hint == "MOMENTUM":
        agent_weights['momentum_agent'] *= 1.2
    elif regime_result.strategy_hint == "MEAN_REVERSION":
        agent_weights['mean_reversion_agent'] *= 1.2
    elif regime_result.strategy_hint == "CAUTION":
        final_confidence *= 0.8  # Reduce confidence on changepoint alerts

    # Log for learning extraction:
    trade_context['regime'] = regime_result.regime_name
    trade_context['changepoint_alert'] = regime_result.changepoint_alert
    trade_context['hurst'] = regime_result.hurst_exponent
    """
    )


if __name__ == "__main__":
    example_basic_usage()
    example_with_ohlc()
    example_regime_transitions()
    example_strategy_guidance()
    example_integration_with_scanner()

    logger.info("\n" + "=" * 80)
    logger.info("All examples complete!")
    logger.info("=" * 80)
