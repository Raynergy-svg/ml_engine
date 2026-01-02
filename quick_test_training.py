#!/usr/bin/env python3
"""
Quick test to verify training pipeline works.
Run with: python quick_test_training.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np

print("=" * 60)
print("Quick Training Test")
print("=" * 60)

# Import TensorFlow
print("\n1. Loading TensorFlow...")
import tensorflow as tf
print(f"   TensorFlow: {tf.__version__}")
print(f"   GPUs: {tf.config.list_physical_devices('GPU')}")

# Import local modules
print("\n2. Importing local modules...")
from m1_metal_optimizer import (
    configure_metal_runtime,
    create_optimized_dataset,
    create_metal_optimized_tcn,
)
from tensorflow_data_pipeline import load_tensorflow_multitask_data

# Configure runtime
print("\n3. Configuring M1 Metal runtime...")
runtime = configure_metal_runtime(mixed_precision=True)
print(f"   {runtime}")

# Load data
print("\n4. Loading data...")
DATA_FILE = "market_data/oanda_USD_JPY_M5_live_20251230_221406.csv"

config = {
    'sequence_length': 60,
    'target_shift': 1,
    'risk_window': 14,
    'state_classes': 2,
    'enable_data_scaling': True,
}

X_train, y_train, X_val, y_val, X_test, y_test, meta = load_tensorflow_multitask_data(
    DATA_FILE, config, state_classes=2, validation_split=0.2, add_all_features=True
)

print(f"   Train shape: {X_train.shape}")
print(f"   Val shape: {X_val.shape}")
print(f"   Test shape: {X_test.shape}")
print(f"   Features: {X_train.shape[2]}")

# Check class balance
direction = y_train['direction'].flatten()
up_pct = 100 * np.mean(direction)
print(f"   Direction balance: {up_pct:.1f}% up / {100-up_pct:.1f}% down")

# Create dataset
print("\n5. Creating tf.data datasets...")
train_ds = create_optimized_dataset(X_train, y_train, batch_size=128, shuffle=True, cache=True)
val_ds = create_optimized_dataset(X_val, y_val, batch_size=128, shuffle=False, cache=True)

# Create model
print("\n6. Creating model...")
input_shape = (X_train.shape[1], X_train.shape[2])
model = create_metal_optimized_tcn(
    input_shape=input_shape,
    hidden_size=32,
    num_layers=2,
    dropout=0.35,
    l2_reg=0.002,
    output_heads={
        'price': 1, 'trend': 1, 'direction': 1, 'risk': 1, 'state_logits': 2,
    },
)

# Compile
print("\n7. Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.01, clipnorm=1.0),
    loss={
        'price': tf.keras.losses.Huber(delta=1.0),
        'trend': tf.keras.losses.Huber(delta=0.5),
        'direction': tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        'risk': tf.keras.losses.Huber(delta=0.01),  # Huber for stability with scaled values
        'state_logits': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    },
    loss_weights={'price': 1.0, 'trend': 0.5, 'direction': 5.0, 'risk': 2.0, 'state_logits': 3.0},
    metrics={'direction': [tf.keras.metrics.BinaryAccuracy(name='dir_acc')]},
)

model.summary()

# Quick training (5 epochs)
print("\n8. Quick training (5 epochs)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=1,
)

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_dir_acc = history.history['direction_dir_acc'][-1]
final_val_dir_acc = history.history['val_direction_dir_acc'][-1]

print(f"Final Loss:       {final_loss:.4f} (val: {final_val_loss:.4f})")
print(f"Direction Acc:    {100*final_dir_acc:.1f}% (val: {100*final_val_dir_acc:.1f}%)")

if final_val_dir_acc > 0.51:
    print("\n✅ Model is learning! Direction accuracy > 51%")
    print("   Ready for full training with: python train_optimized_m1.py --epochs 50")
else:
    print("\n⚠️  Direction accuracy is at ~50% (random)")
    print("   This is normal for 5 epochs - model needs more training")
    print("   Run full training: python train_optimized_m1.py --epochs 100")

print("\n" + "=" * 60)

