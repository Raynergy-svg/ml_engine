#!/usr/bin/env python3
"""
Test script for Modular Ensemble.

Each model can be tested independently. All models share the same 8,000 H1 candles
but use different feature subsets and predict different targets.

Usage:
    # Test all models together
    python test_modular_ensemble.py
    
    # Test individual models
    python test_modular_ensemble.py --model tcn
    python test_modular_ensemble.py --model xgboost
    python test_modular_ensemble.py --model rf
    python test_modular_ensemble.py --model ridge
    
    # Test inference pipeline
    python test_modular_ensemble.py --inference
    
    # Train all models (same as: buddy train --model-type ensemble)
    python test_modular_ensemble.py --train

One-line test commands:
    buddy train --model-type ensemble --candles 8000    # Train modular ensemble
    buddy predict                                        # Run inference with gates
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic test data with all required features."""
    np.random.seed(42)
    
    # Simulate price series
    close = 150 + np.cumsum(np.random.randn(n_samples) * 0.1)
    
    df = pd.DataFrame({
        # OHLCV
        'open': close + np.random.randn(n_samples) * 0.05,
        'high': close + np.abs(np.random.randn(n_samples)) * 0.1,
        'low': close - np.abs(np.random.randn(n_samples)) * 0.1,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_samples),
        
        # Returns
        'returns': np.concatenate([[0], np.diff(close) / close[:-1]]),
        
        # Volatility features (for TCN)
        'volatility_5': np.abs(np.random.randn(n_samples)) * 0.01,
        'volatility_10': np.abs(np.random.randn(n_samples)) * 0.01,
        'volatility_20': np.abs(np.random.randn(n_samples)) * 0.01,
        'volatility_log_5': np.log(1 + np.abs(np.random.randn(n_samples)) * 0.01),
        'volatility_log_10': np.log(1 + np.abs(np.random.randn(n_samples)) * 0.01),
        'volatility_log_20': np.log(1 + np.abs(np.random.randn(n_samples)) * 0.01),
        'volatility_regime': np.random.randint(0, 3, n_samples),
        'atr': np.abs(np.random.randn(n_samples)) * 0.5,
        'bb_width_20': np.abs(np.random.randn(n_samples)) * 0.02,
        'high_low_ratio': 1 + np.abs(np.random.randn(n_samples)) * 0.001,
        
        # Trend features (for TCN)
        'sma_5': close + np.random.randn(n_samples) * 0.1,
        'sma_10': close + np.random.randn(n_samples) * 0.1,
        'sma_20': close + np.random.randn(n_samples) * 0.1,
        'ema_12': close + np.random.randn(n_samples) * 0.1,
        'ema_26': close + np.random.randn(n_samples) * 0.1,
        
        # Momentum features (for TCN, XGBoost)
        'rsi': np.random.uniform(30, 70, n_samples),
        'rsi_momentum': np.random.randn(n_samples) * 0.1,
        'momentum_10': np.random.randn(n_samples) * 0.01,
        
        # Lag features (for XGBoost)
        'close_lag_1': np.roll(close, 1),
        'close_lag_2': np.roll(close, 2),
        'close_lag_3': np.roll(close, 3),
        'close_lag_5': np.roll(close, 5),
        'close_lag_10': np.roll(close, 10),
        
        # ROC features (for XGBoost)
        'roc_5': np.random.randn(n_samples) * 0.01,
        'roc_10': np.random.randn(n_samples) * 0.01,
        'roc_20': np.random.randn(n_samples) * 0.01,
        
        # Price range (for XGBoost, RF)
        'price_range_5': np.abs(np.random.randn(n_samples)) * 0.2,
        'price_range_10': np.abs(np.random.randn(n_samples)) * 0.3,
        
        # Volume features (for XGBoost, Ridge)
        'obv': np.cumsum(np.random.randint(-1000, 1000, n_samples)),
        'mfi': np.random.uniform(20, 80, n_samples),
        
        # MACD features (for XGBoost)
        'macd': np.random.randn(n_samples) * 0.01,
        'macd_hist': np.random.randn(n_samples) * 0.005,
        
        # Stochastic (for XGBoost)
        'stoch_k': np.random.uniform(20, 80, n_samples),
        'stoch_d': np.random.uniform(20, 80, n_samples),
        
        # Risk features (for RF)
        'close_min_5': close - np.abs(np.random.randn(n_samples)) * 0.2,
        'close_max_5': close + np.abs(np.random.randn(n_samples)) * 0.2,
        'close_min_10': close - np.abs(np.random.randn(n_samples)) * 0.3,
        'close_max_10': close + np.abs(np.random.randn(n_samples)) * 0.3,
        'close_min_20': close - np.abs(np.random.randn(n_samples)) * 0.4,
        'close_max_20': close + np.abs(np.random.randn(n_samples)) * 0.4,
        'cum_returns': np.cumsum(np.random.randn(n_samples) * 0.001),
        
        # Confidence features (for Ridge)
        'bb_position_20': np.random.uniform(-1, 1, n_samples),
        'kurt_20': np.random.uniform(-0.5, 0.5, n_samples),
        'adx': np.random.uniform(15, 40, n_samples),
    })
    
    return df


def test_tcn():
    """Test TCN model alone."""
    print("\n" + "="*60)
    print("TESTING TCN (Direction Predictor)")
    print("="*60)
    
    from src.core.modular_data_loaders import load_tcn_data
    from src.training.modular_trainers import TCNTrainer, TrainerConfig
    
    # Create test data
    df = create_test_data(500)
    
    # Load data
    data = load_tcn_data(df, split=(0.7, 0.2, 0.1))
    print(f"Features: {len(data['feature_names'])}")
    print(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    # Train with minimal epochs for testing
    config = TrainerConfig(epochs=5, verbose=0)
    trainer = TCNTrainer(config)
    metrics = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    print(f"Metrics: {metrics}")
    
    # Test prediction
    pred = trainer.predict(data['X_test'][:60])
    print(f"Prediction: direction={pred['direction']}, probability={pred['probability']:.3f}")
    
    print("✓ TCN test passed")
    return True


def test_xgboost():
    """Test XGBoost model alone."""
    print("\n" + "="*60)
    print("TESTING XGBOOST (Momentum Analyzer)")
    print("="*60)
    
    from src.core.modular_data_loaders import load_xgboost_data
    from src.training.modular_trainers import XGBoostTrainer, TrainerConfig
    
    # Create test data
    df = create_test_data(500)
    
    # Load data
    data = load_xgboost_data(df, split=(0.7, 0.2, 0.1))
    print(f"Features: {len(data['feature_names'])}")
    print(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    # Train
    config = TrainerConfig(xgb_n_estimators=10)  # Minimal for testing
    trainer = XGBoostTrainer(config)
    metrics = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    print(f"Metrics: {metrics}")
    
    # Test prediction
    pred = trainer.predict(data['X_test'][:10])
    print(f"Prediction: momentum={pred['momentum']:.3f}, acceleration={pred['acceleration']}")
    
    print("✓ XGBoost test passed")
    return True


def test_rf():
    """Test Random Forest model alone."""
    print("\n" + "="*60)
    print("TESTING RANDOM FOREST (Risk Assessor)")
    print("="*60)
    
    from src.core.modular_data_loaders import load_rf_data
    from src.training.modular_trainers import RandomForestTrainer, TrainerConfig
    
    # Create test data
    df = create_test_data(500)
    
    # Load data
    data = load_rf_data(df, split=(0.7, 0.2, 0.1))
    print(f"Features: {len(data['feature_names'])}")
    print(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    # Train
    config = TrainerConfig(rf_n_estimators=10)  # Minimal for testing
    trainer = RandomForestTrainer(config)
    metrics = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    print(f"Metrics: {metrics}")
    
    # Test prediction
    pred = trainer.predict(data['X_test'][:10])
    print(f"Prediction: expected_drawdown={pred['expected_drawdown_pips']:.2f} pips, streak_prob={pred['streak_prob']:.3f}")
    
    print("✓ Random Forest test passed")
    return True


def test_ridge():
    """Test Ridge model alone."""
    print("\n" + "="*60)
    print("TESTING RIDGE (Confidence Scorer)")
    print("="*60)
    
    from src.core.modular_data_loaders import load_ridge_data
    from src.training.modular_trainers import RidgeTrainer, TrainerConfig
    
    # Create test data
    df = create_test_data(500)
    
    # Load data
    data = load_ridge_data(df, split=(0.7, 0.2, 0.1))
    print(f"Features: {len(data['feature_names'])}")
    print(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    # Train
    config = TrainerConfig()
    trainer = RidgeTrainer(config)
    metrics = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    print(f"Metrics: {metrics}")
    
    # Test prediction
    pred = trainer.predict(data['X_test'][:10])
    print(f"Prediction: confidence={pred['confidence']:.1f}/100")
    
    print("✓ Ridge test passed")
    return True


def test_inference():
    """Test full inference pipeline."""
    print("\n" + "="*60)
    print("TESTING INFERENCE PIPELINE")
    print("="*60)
    
    model_dir = Path("trained_data/models")
    
    # Check if models exist
    required_models = [
        "tcn_direction.keras",
        "xgb_momentum.pkl",
        "rf_risk.pkl",
        "ridge_confidence.pkl",
    ]
    
    missing = [m for m in required_models if not (model_dir / m).exists()]
    if missing:
        print(f"Missing models: {missing}")
        print("Train first with: buddy train --model-type ensemble --candles 8000")
        return False
    
    from src.core.modular_inference import ModularEnsembleInference, InferenceConfig
    
    # Create test data
    df = create_test_data(100)
    
    # Configure inference
    config = InferenceConfig(
        min_confidence=50,  # Lower threshold for testing
        min_momentum=0.3,
        max_drawdown_pips=50,
        max_streak_prob=0.5,
    )
    
    # Run inference
    ensemble = ModularEnsembleInference(config=config)
    result = ensemble.predict_verbose(df)
    
    print("\nGate Checks:")
    for check in result['gate_checks']:
        print(f"  {check}")
    
    print(f"\nDecision: {result['decision']}")
    
    print("✓ Inference test passed")
    return True


def test_all():
    """Run all tests."""
    print("\n" + "="*70)
    print("MODULAR ENSEMBLE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test each model
    try:
        results['tcn'] = test_tcn()
    except Exception as e:
        print(f"✗ TCN failed: {e}")
        results['tcn'] = False
    
    try:
        results['xgboost'] = test_xgboost()
    except Exception as e:
        print(f"✗ XGBoost failed: {e}")
        results['xgboost'] = False
    
    try:
        results['rf'] = test_rf()
    except Exception as e:
        print(f"✗ Random Forest failed: {e}")
        results['rf'] = False
    
    try:
        results['ridge'] = test_ridge()
    except Exception as e:
        print(f"✗ Ridge failed: {e}")
        results['ridge'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {name:12s}: {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    return all(results.values())


def train_ensemble():
    """Train full modular ensemble (wrapper for buddy train)."""
    print("\n" + "="*60)
    print("TRAINING MODULAR ENSEMBLE")
    print("="*60)
    print("This is equivalent to: buddy train --model-type ensemble --candles 8000")
    print("")
    
    import subprocess
    result = subprocess.run(
        ["./Buddy", "train", "--model-type", "ensemble", "--candles", "8000"],
        capture_output=False,
        text=True,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Test modular ensemble models")
    parser.add_argument("--model", choices=["tcn", "xgboost", "rf", "ridge"], 
                        help="Test specific model")
    parser.add_argument("--inference", action="store_true", 
                        help="Test inference pipeline")
    parser.add_argument("--train", action="store_true", 
                        help="Train full ensemble")
    parser.add_argument("--all", action="store_true", 
                        help="Run all tests (default)")
    
    args = parser.parse_args()
    
    if args.train:
        success = train_ensemble()
    elif args.model == "tcn":
        success = test_tcn()
    elif args.model == "xgboost":
        success = test_xgboost()
    elif args.model == "rf":
        success = test_rf()
    elif args.model == "ridge":
        success = test_ridge()
    elif args.inference:
        success = test_inference()
    else:
        success = test_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
