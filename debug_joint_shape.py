#!/usr/bin/env python3
"""Debug script to check shapes in joint training."""

from src.utils.oanda_practice import OandaPracticeClient
from src.core.modular_data_loaders import (
    compute_normalized_features, 
    load_direction_data,
    load_xgboost_data,
    load_rf_data,
    load_ridge_data,
)
import pandas as pd
import numpy as np

# Fetch small batch
print("Fetching data...")
client = OandaPracticeClient.from_env()
response = client.get_candles('USD_JPY', granularity='H1', count=3000, price='MBA')
candles = response.get('candles', [])

rows = []
for c in candles:
    rows.append({
        'time': c.get('time'),
        'open': float(c.get('mid', {}).get('o', 0)),
        'high': float(c.get('mid', {}).get('h', 0)),
        'low': float(c.get('mid', {}).get('l', 0)),
        'close': float(c.get('mid', {}).get('c', 0)),
        'volume': int(c.get('volume', 0)),
    })

df = pd.DataFrame(rows)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = compute_normalized_features(df)

print(f"DataFrame shape after features: {df.shape}")
print(f"DataFrame columns: {len(df.columns)}")

# Check each data loader
print("\n=== load_direction_data ===")
dir_result = load_direction_data(df)
print(f'X_train shape: {dir_result["X_train"].shape}')
print(f'y_train shape: {dir_result["y_train"].shape}')
print(f'feature_names count: {len(dir_result["feature_names"])}')

print("\n=== load_xgboost_data ===")
xgb_result = load_xgboost_data(df)
print(f'X_train shape: {xgb_result["X_train"].shape}')
print(f'y_train shape: {xgb_result["y_train"].shape}')
print(f'y_train sample: {xgb_result["y_train"][:3]}')
print(f'feature_names count: {len(xgb_result["feature_names"])}')

print("\n=== load_rf_data ===")
rf_result = load_rf_data(df)
print(f'X_train shape: {rf_result["X_train"].shape}')
print(f'y_train shape: {rf_result["y_train"].shape}')
print(f'feature_names count: {len(rf_result["feature_names"])}')

print("\n=== load_ridge_data ===")
ridge_result = load_ridge_data(df)
print(f'X_train shape: {ridge_result["X_train"].shape}')
print(f'y_train shape: {ridge_result["y_train"].shape}')
print(f'feature_names count: {len(ridge_result["feature_names"])}')

print("\n=== Checking if shapes match ===")
# These should all be the same (X features)
print(f"Direction X cols: {dir_result['X_train'].shape[1]}")
print(f"XGBoost X cols:   {xgb_result['X_train'].shape[1]}")
print(f"RF X cols:        {rf_result['X_train'].shape[1]}")
print(f"Ridge X cols:     {ridge_result['X_train'].shape[1]}")
