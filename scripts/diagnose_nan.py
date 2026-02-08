#!/usr/bin/env python3
"""Diagnose NaN issues in training data."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

print("=" * 60)
print("Diagnosing NaN Issues in Training Data")
print("=" * 60)

# Load data
DATA_FILE = "market_data/oanda_USD_JPY_M5_live_20251230_221406.csv"
print(f"\nLoading: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
print(f"Rows: {len(df)}")

# Check for NaN in raw data
print("\n1. NaN in raw data:")
for col in df.columns:
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f"   {col}: {nan_count} NaN values")

# Check close prices
close = df['close'].values
print("\n2. Close prices:")
print(f"   min: {close.min():.4f}")
print(f"   max: {close.max():.4f}")
print(f"   NaN: {np.isnan(close).sum()}")
print(f"   Inf: {np.isinf(close).sum()}")
print(f"   Zero: {(close == 0).sum()}")

# Calculate returns like multitask_labels.py
returns = np.zeros_like(close, dtype=float)
returns[1:] = (close[1:] / np.clip(close[:-1], 1e-12, None)) - 1.0

print("\n3. Returns:")
print(f"   min: {returns.min():.6f}")
print(f"   max: {returns.max():.6f}")
print(f"   mean: {returns.mean():.6f}")
print(f"   std: {returns.std():.6f}")
print(f"   NaN: {np.isnan(returns).sum()}")
print(f"   Inf: {np.isinf(returns).sum()}")

# Calculate risk (rolling std)
risk_window = 14
risk = np.zeros_like(returns, dtype=float)
for i in range(len(returns)):
    start = max(0, i - risk_window + 1)
    window = returns[start : i + 1]
    risk[i] = float(np.std(window))

print("\n4. Risk (rolling std of returns):")
print(f"   min: {risk.min():.8f}")
print(f"   max: {risk.max():.8f}")
print(f"   mean: {risk.mean():.8f}")
print(f"   NaN: {np.isnan(risk).sum()}")
print(f"   Inf: {np.isinf(risk).sum()}")
print(f"   Zero: {(risk == 0).sum()}")

# The issue: risk values are VERY SMALL (order of 1e-4 to 1e-3)
# When squared in MSE, they become 1e-8 to 1e-6, causing numerical issues

print("\n5. Risk value distribution:")
print(f"   < 1e-5: {(risk < 1e-5).sum()}")
print(f"   < 1e-4: {(risk < 1e-4).sum()}")
print(f"   < 1e-3: {(risk < 1e-3).sum()}")

# Now test with feature engineering
print("\n6. Testing full data pipeline...")
from tensorflow_data_pipeline import load_tensorflow_multitask_data  # noqa: E402

config = {
    'sequence_length': 60,
    'target_shift': 1,
    'risk_window': 14,
    'state_classes': 2,
    'enable_data_scaling': True,
}

try:
    X_train, y_train, X_val, y_val, X_test, y_test, meta = load_tensorflow_multitask_data(
        DATA_FILE, config, state_classes=2, validation_split=0.2, add_all_features=True
    )
    
    print(f"\n   X_train shape: {X_train.shape}")
    print(f"   X_train NaN: {np.isnan(X_train).sum()}")
    print(f"   X_train Inf: {np.isinf(X_train).sum()}")
    
    for key, arr in y_train.items():
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        print(f"\n   {key}:")
        print(f"      shape: {arr.shape}")
        print(f"      NaN: {nan_count}")
        print(f"      Inf: {inf_count}")
        print(f"      min: {arr.min():.8f}")
        print(f"      max: {arr.max():.8f}")
        print(f"      mean: {arr.mean():.8f}")
        
        if nan_count > 0 or inf_count > 0:
            print("      ⚠️ WARNING: Contains invalid values!")
            
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)
