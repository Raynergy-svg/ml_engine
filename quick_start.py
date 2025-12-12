#!/usr/bin/env python3
"""
Quick Start Example - Train a Model with Enhanced Features

This script demonstrates how to use the enhanced ML engine to:
1. Load and preprocess data
2. Create features
3. Train a model
4. Evaluate performance
5. Run a backtest

Usage:
    python quick_start.py
"""

from pathlib import Path

# Import enhanced modules
from data_loader import MarketDataLoader
from evaluation import Backtester, generate_simple_strategy_signals
from utils import load_config, setup_logging
from train_enhanced import EnhancedTrainer

# Setup logging
logger = setup_logging(log_file="quick_start.log")


def main():
    """Quick start example."""

    logger.info("=" * 60)
    logger.info("ML Engine Trading Bot - Quick Start")
    logger.info("=" * 60)

    # Step 1: Load configuration
    logger.info("\n1. Loading configuration...")
    config = load_config("config.yaml")
    logger.info("✓ Configuration loaded")

    # Step 2: Load data
    logger.info("\n2. Loading market data...")
    data_file = "market_data/TSLA_data.csv"

    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please ensure TSLA_data.csv exists in market_data/ directory")
        return

    loader = MarketDataLoader(config)
    df = loader.load_csv(data_file)
    logger.info(f"✓ Loaded {len(df)} rows of data")

    # Step 3: Preprocess data with feature engineering
    logger.info("\n3. Preprocessing data and creating features...")
    logger.info("   This will create 100+ technical indicators and features...")

    X_train, y_train, x_val, y_val, X_test, y_test = loader.preprocess(
        df,
        add_features=True,
        scaler_type="standard",
        sequence_length=60,
        test_size=0.2,
        validation_size=0.1,
    )

    logger.info("✓ Data preprocessed:")
    logger.info(f"   Train: {X_train.shape}")
    logger.info(f"   Val:   {x_val.shape}")
    logger.info(f"   Test:  {X_test.shape}")

    # Save scaler for future use
    loader.save_scaler("trained_data/models/scaler.pkl")
    logger.info("✓ Scaler saved")

    # Step 4: Initialize trainer
    logger.info("\n4. Initializing enhanced trainer...")
    trainer = EnhancedTrainer(config)
    logger.info(f"✓ Using device: {trainer.device}")

    # Step 5: Train model
    logger.info("\n5. Training model...")
    logger.info("   This may take several minutes depending on your hardware...")
    logger.info("   Model: Attention-LSTM with 3 layers")

    train_losses, val_losses = trainer.train(
        X_train,
        y_train,
        x_val,
        y_val,
        num_epochs=50,  # Reduced for quick start
    )

    logger.info("✓ Training complete!")
    logger.info(f"   Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"   Final val loss:   {val_losses[-1]:.6f}")

    # Step 6: Evaluate on test set
    logger.info("\n6. Evaluating on test set...")
    predictions, targets, metrics = trainer.evaluate_on_test(X_test, y_test)

    logger.info("✓ Evaluation complete:")
    logger.info(f"   RMSE: {metrics['test_rmse']:.6f}")
    logger.info(f"   MAE:  {metrics['test_mae']:.6f}")
    logger.info(f"   R²:   {metrics['test_r2']:.6f}")
    logger.info(
        f"   Direction Accuracy: {metrics.get('test_direction_accuracy', 0):.2f}%"
    )

    # Step 7: Run backtest
    logger.info("\n7. Running backtest...")
    backtester = Backtester(initial_capital=10000.0)
    signals = generate_simple_strategy_signals(predictions, targets)
    backtest_metrics = backtester.run_backtest(targets, signals)

    logger.info("✓ Backtest complete:")
    logger.info(
        f"   Total Return:   {backtest_metrics.get('total_return_pct', 0):.2f}%"
    )
    logger.info(f"   Sharpe Ratio:   {backtest_metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(
        f"   Max Drawdown:   {backtest_metrics.get('max_drawdown_pct', 0):.2f}%"
    )
    logger.info(f"   Win Rate:       {backtest_metrics.get('win_rate_pct', 0):.2f}%")
    logger.info(f"   Num Trades:     {backtest_metrics.get('num_trades', 0)}")
    logger.info(f"   Final Value:    ${backtest_metrics.get('final_value', 0):.2f}")

    # Step 8: Summary
    logger.info("\n" + "=" * 60)
    logger.info("Quick Start Complete!")
    logger.info("=" * 60)
    logger.info("\nFiles created:")
    logger.info("  • trained_data/models/best_model.pth     - Best model checkpoint")
    logger.info("  • trained_data/models/scaler.pkl         - Feature scaler")
    logger.info("  • trained_data/visualizations/           - Evaluation plots")
    logger.info("  • training.log                           - Training logs")
    logger.info("  • quick_start.log                        - This run's logs")

    logger.info("\nNext steps:")
    logger.info("  1. Check the visualizations in trained_data/visualizations/")
    logger.info("  2. Review the logs for detailed training information")
    logger.info("  3. Try different model architectures in config.yaml")
    logger.info("  4. Experiment with different features and parameters")
    logger.info("  5. Use train_enhanced.py for full training with more epochs")

    logger.info("\nFor more information, see:")
    logger.info("  • IMPROVEMENTS.md    - Comprehensive documentation")
    logger.info("  • CHANGES_SUMMARY.md - Summary of all improvements")
    logger.info("  • config.yaml        - Configuration options")

    logger.info("\nThank you for using ML Engine Trading Bot!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"\n\nError: {e}", exc_info=True)
        logger.info(
            "\nIf you need help, check the documentation or open an issue on GitHub"
        )
