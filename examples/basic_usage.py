#!/usr/bin/env python3
"""
Basic usage example for ML Engine.

This script demonstrates how to:
1. Load configuration
2. Initialize the ML engine
3. Train a model
4. Evaluate performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ml_engine import EnhancedMLEngine
from utils import load_config

def main():
    print("ML Engine - Basic Usage Example")
    print("=" * 60)
    
    # Simple example configuration
    config = {
        "device": "cpu",
        "model": {"type": "lstm", "input_size": 7, "hidden_size": 64, "num_layers": 2, "dropout": 0.2},
        "optimizer": {"type": "adam", "learning_rate": 0.001},
        "training": {"epochs": 3},
        "batch_size": 32,
        "early_stopping_patience": 10
    }
    
    # Create dummy data
    X_train = np.random.randn(100, 60, 7).astype(np.float32)
    y_train = np.random.randn(100, 1).astype(np.float32)
    
    # Initialize and train
    engine = EnhancedMLEngine(config)
    for epoch, metrics in engine.train(X_train, y_train):
        print(f"Epoch {epoch}: Train Loss={metrics['train_loss']:.6f}")
    
    print("\n✓ Example completed!")

if __name__ == "__main__":
    main()
