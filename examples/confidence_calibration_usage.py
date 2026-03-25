#!/usr/bin/env python3
"""
Example usage of the Enhanced Confidence Calibration System.

Demonstrates:
1. Initialization and configuration
2. Calibrating a trade setup
3. Applying time decay
4. Recording outcomes
5. Automatic refitting
6. Integration with trading loops
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scanner.confidence_calibration import (
    CalibrationConfig,
    ConfidenceCalibrationSystem,
)


class MockAgentVerdict:
    """Mock agent verdict for demonstration."""
    def __init__(self, name, score, weight=1.0, passed=True):
        self.name = name
        self.score = score
        self.weight = weight
        self.passed = passed


def example_1_basic_initialization():
    """Example 1: Initialize the calibration system."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Initialization")
    print("="*70)

    # Use default configuration
    system = ConfidenceCalibrationSystem()
    print(f"✓ System initialized")
    print(f"  - Calibration file: {system._config.calibration_file}")
    print(f"  - Min trades for calibration: {system._config.min_trades_for_calibration}")
    print(f"  - Refit interval: {system._config.refit_interval}")

    # Or use custom configuration
    custom_config = CalibrationConfig(
        min_trades_for_calibration=50,
        refit_interval=25,
        calibration_file="trained_data/my_calibration.json",
    )
    system2 = ConfidenceCalibrationSystem(custom_config)
    print(f"\n✓ Custom system initialized")
    print(f"  - Min trades: {system2._config.min_trades_for_calibration}")
    print(f"  - Refit interval: {system2._config.refit_interval}")


def example_2_calibrate_trade():
    """Example 2: Calibrate a trade setup."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Calibrate a Trade Setup")
    print("="*70)

    system = ConfidenceCalibrationSystem()

    # Create agent verdicts (from ScannerAgentTeam.evaluate())
    agent_verdicts = [
        MockAgentVerdict("trend", 0.72, weight=1.15),
        MockAgentVerdict("momentum", 0.68, weight=1.05),
        MockAgentVerdict("mean_reversion", 0.75, weight=0.90),
        MockAgentVerdict("volatility", 0.70, weight=1.00),
        MockAgentVerdict("risk_sentinel", 0.65, weight=1.25),
        MockAgentVerdict("uncertainty", 0.73, weight=1.10),
        MockAgentVerdict("execution_quality", 0.71, weight=1.05),
        MockAgentVerdict("news_risk", 0.74, weight=0.95),
        MockAgentVerdict("multi_timeframe", 0.69, weight=1.10),
        MockAgentVerdict("pair_performance", 0.70, weight=0.85),
        MockAgentVerdict("session_timing", 0.66, weight=0.80),
        MockAgentVerdict("support_resistance", 0.72, weight=1.00),
    ]

    # Calibrate for NORMAL volatility regime
    result = system.calibrate(agent_verdicts, regime_name="NORMAL")

    print(f"✓ Calibration complete")
    print(f"\n  Raw Inputs:")
    print(f"    - Raw weighted score: {result.raw_weighted_score:.3f}")
    print(f"    - Agent scores range: [{min(result.agent_scores):.3f}, {max(result.agent_scores):.3f}]")

    print(f"\n  Ensemble Disagreement:")
    print(f"    - Std dev: {result.ensemble_disagreement:.4f}")
    print(f"    - Level: {result.disagreement_level}")

    print(f"\n  Platt Scaling:")
    print(f"    - Calibrated score: {result.platt_calibrated:.3f}")
    print(f"    - Is calibrated: {result.is_calibrated}")
    print(f"    - Regime used: {result.calibration_regime}")

    print(f"\n  Agent Agreement:")
    print(f"    - Agreement quality: {result.agent_agreement:.3f}")
    print(f"    - Agents reporting: {result.agents_reporting}")

    print(f"\n  Meta-Confidence:")
    print(f"    - Meta confidence: {result.meta_confidence:.3f}")

    print(f"\n  Final Result:")
    print(f"    - Final confidence: {result.final_confidence:.3f}")

    return system, result


def example_3_time_decay(system, initial_result):
    """Example 3: Apply time decay as position ages."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Apply Time Decay")
    print("="*70)

    # Apply decay after 5 bars
    decayed_5 = system.apply_time_decay(initial_result, bars_held=5, regime_name="NORMAL")
    print(f"After 5 bars in NORMAL regime:")
    print(f"  - Original confidence: {initial_result.final_confidence:.3f}")
    print(f"  - Decay factor: {decayed_5.time_decay_factor:.4f}")
    print(f"  - Decayed confidence: {decayed_5.final_confidence:.3f}")

    # Apply decay after 20 bars
    decayed_20 = system.apply_time_decay(initial_result, bars_held=20, regime_name="NORMAL")
    print(f"\nAfter 20 bars in NORMAL regime:")
    print(f"  - Original confidence: {initial_result.final_confidence:.3f}")
    print(f"  - Decay factor: {decayed_20.time_decay_factor:.4f}")
    print(f"  - Decayed confidence: {decayed_20.final_confidence:.3f}")

    # Compare regimes
    decayed_high = system.apply_time_decay(initial_result, bars_held=10, regime_name="HIGH")
    decayed_normal = system.apply_time_decay(initial_result, bars_held=10, regime_name="NORMAL")
    print(f"\nAfter 10 bars, comparing regimes:")
    print(f"  - HIGH volatility: {decayed_high.final_confidence:.3f} (fast decay)")
    print(f"  - NORMAL volatility: {decayed_normal.final_confidence:.3f} (moderate decay)")


def example_4_record_and_learn(system):
    """Example 4: Record trade outcomes for learning."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Record Outcomes & Learn")
    print("="*70)

    print(f"Initial state:")
    print(f"  - Platt params: {len(system._platt_params)}")
    print(f"  - Trade history: {len(system._trade_history)}")

    # Simulate a series of trades
    trades = [
        (0.72, 1.0, "NORMAL"),  # Win with high confidence
        (0.68, 0.0, "NORMAL"),  # Loss with medium confidence
        (0.75, 1.0, "NORMAL"),  # Win with high confidence
        (0.65, 0.0, "NORMAL"),  # Loss with lower confidence
        (0.70, 1.0, "NORMAL"),  # Win with medium-high confidence
    ]

    print(f"\nRecording {len(trades)} trades:")
    for i, (raw_score, outcome, regime) in enumerate(trades, 1):
        system.record_outcome(raw_score=raw_score, outcome=outcome, regime_name=regime)
        status = "WIN" if outcome == 1.0 else "LOSS"
        print(f"  {i}. Score={raw_score:.2f}, Outcome={status}, Regime={regime}")

    print(f"\nAfter recording:")
    print(f"  - Trade history: {len(system._trade_history)} trades")
    print(f"  - Trades since refit: {system._trades_since_refit}")

    # Show win rate
    outcomes = [t[1] for t in system._trade_history]
    win_rate = sum(outcomes) / len(outcomes) if outcomes else 0
    print(f"  - Win rate: {win_rate:.1%}")


def example_5_bulk_calibration(system):
    """Example 5: Build up calibration with bulk trades."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Bulk Calibration Building")
    print("="*70)

    print(f"Current history: {len(system._trade_history)} trades")
    print(f"Platt params available: {list(system._platt_params.keys())}")

    # Add enough trades to trigger Platt calibration (need 30+ total)
    print(f"\nAdding trades to reach calibration threshold ({system._config.min_trades_for_calibration} trades)...")

    import random
    random.seed(42)
    target_trades = system._config.min_trades_for_calibration + 10

    while len(system._trade_history) < target_trades:
        # Simulate: higher raw scores have higher win probability
        raw_score = random.uniform(0.4, 0.8)
        # Win probability increases with raw score
        win_prob = (raw_score - 0.4) / 0.4  # Maps [0.4, 0.8] to [0, 1]
        outcome = 1.0 if random.random() < win_prob else 0.0
        regime = random.choice(["LOW", "NORMAL", "HIGH"])
        system.record_outcome(raw_score=raw_score, outcome=outcome, regime_name=regime)

    print(f"✓ Built up to {len(system._trade_history)} trades")
    print(f"✓ Platt calibration activated with params: {list(system._platt_params.keys())}")

    # Show calibration quality
    for regime, params in system._platt_params.items():
        coef = params.get("coef", 0)
        intercept = params.get("intercept", 0)
        print(f"  - {regime}: coef={coef:.4f}, intercept={intercept:.4f}")


def example_6_integration_with_trading_loop():
    """Example 6: Integration pattern for trading loops."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Integration with Trading Loop")
    print("="*70)

    system = ConfidenceCalibrationSystem()

    print("""
    # Pattern for integration in Scanner/Execution:

    def scan_and_execute():
        verdicts = agent_team.evaluate(pair, timeframe)

        # 1. Calibrate
        calibration = system.calibrate(verdicts, regime_name=volatility_regime)

        # 2. Check confidence gate
        if calibration.final_confidence < 0.55:
            logger.info(f"Rejected: confidence={calibration.final_confidence:.3f} < 0.55")
            return None

        # 3. Log for audit
        logger.info(f"Calibration: raw={calibration.raw_weighted_score:.3f}, "
                   f"platt={calibration.platt_calibrated:.3f}, "
                   f"final={calibration.final_confidence:.3f}, "
                   f"disagreement={calibration.disagreement_level}")

        # 4. Execute trade
        trade = execute_trade(pair, sl, tp, size=position_size)

        # 5. Store calibration result in trade record
        trade_record = {
            "trade_id": trade.id,
            "calibration_result": calibration,
            "entry_time": datetime.now(),
        }

        # 6. Later, when trade closes...
        outcome = 1.0 if trade.profit > 0 else 0.0
        system.record_outcome(
            raw_score=calibration.raw_weighted_score,
            outcome=outcome,
            regime_name=volatility_regime,
        )

        # System auto-refits when refit_interval trades accumulated
    """)

    print("✓ Integration pattern documented above")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ENHANCED CONFIDENCE CALIBRATION SYSTEM - USAGE EXAMPLES")
    print("="*70)

    example_1_basic_initialization()

    system, result = example_2_calibrate_trade()

    example_3_time_decay(system, result)

    example_4_record_and_learn(system)

    example_5_bulk_calibration(system)

    example_6_integration_with_trading_loop()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
