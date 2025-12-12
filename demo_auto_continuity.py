#!/usr/bin/env python3
"""
Demo: Auto Session Continuity

Shows how the model now automatically resumes from the best checkpoint
and continues improving across multiple training runs.
"""

import torch
import numpy as np
from ml_engine_enhanced import EnhancedMLEngine
from utils import load_config

print("=" * 70)
print("AUTO SESSION CONTINUITY DEMO")
print("=" * 70)
print()
print("The model now automatically:")
print("  ✓ Resumes from the best checkpoint if it exists")
print("  ✓ Continues training from where it left off")
print("  ✓ Preserves optimizer state, learning rate, and training history")
print("  ✓ Improves incrementally across multiple runs")
print()

# Load config
config = load_config("config.yaml")
config["epochs"] = 5  # Short runs for demo
config["batch_size"] = 32
config["auto_resume"] = True  # Enable auto-resume (now default)

print("Configuration:")
print(f"  • auto_resume: {config['auto_resume']}")
print(f"  • epochs per run: {config['epochs']}")
print(f"  • early_stopping_patience: {config.get('early_stopping_patience', 10)}")
print()

# Create dummy data for demo
print("Creating dummy training data...")
np.random.seed(42)
n_samples = 200
seq_len = 60
n_features = config.get("model", {}).get("input_size", 7)

X_train = torch.randn(n_samples, seq_len, n_features)
y_train = torch.randn(n_samples)
X_val = torch.randn(50, seq_len, n_features)
y_val = torch.randn(50)

print(f"  • Training samples: {n_samples}")
print(f"  • Validation samples: {len(y_val)}")
print()

# Simulate multiple training sessions
print("=" * 70)
print("SIMULATING MULTIPLE TRAINING SESSIONS")
print("=" * 70)
print()

for session in range(1, 4):
    print(f"{'='*70}")
    print(f"SESSION {session}: Running {config['epochs']} epochs")
    print(f"{'='*70}")
    
    # Create engine (will auto-resume if checkpoint exists)
    engine = EnhancedMLEngine(config)
    
    # Train
    result = engine.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config['epochs']
    )
    
    print()
    print(f"Session {session} Results:")
    print(f"  • Resumed: {result.get('resumed', False)}")
    print(f"  • Total epochs trained: {result.get('total_epochs', 'N/A')}")
    print(f"  • Best validation loss: {result['best_val_loss']:.6f}")
    print(f"  • Training history length: {len(result['train_losses'])}")
    print()

print("=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print()
print("Key Observations:")
print("  • Session 1: Trains from scratch (epochs 1-5)")
print("  • Session 2: Auto-resumes from best checkpoint (epochs 6-10)")
print("  • Session 3: Continues improving (epochs 11-15)")
print()
print("The model maintains continuity across sessions without manual intervention!")
print()
print("To disable auto-resume, set in config.yaml:")
print("  auto_resume: false")
print()
